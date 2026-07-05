"""Orchestrator — platform-agnostic agent loop.

Improvements integrated:
  1  Tool result size cap (max_tool_result_chars, default 8000)
  3  Structured error classification (ErrorKind — not string matching)
  4  Rolling context window (compact_after_turns, LLM summarisation)
  5  In-session read-only tool result cache
  6  Parallel execution of independent tool calls
  7  Disk quota guard before shell/python (min_free_disk_mb, 0=unlimited)
  8  TaskRouter invalidated on /reload (handled in app.py)
  9  Discord edit throttle (handled in discord/adapter.py)
 10  Per-session token/cost tracking, auto-pause on QUOTA_EXHAUSTED,
     state persisted to ~/.rika/data/paused/ for /resume after restart.
"""
from __future__ import annotations
import asyncio, hashlib, json, re, shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.config import Config
from src.core.errors import ErrorKind, classify_error, is_fatal, is_quota, is_retryable
from src.core.models import EventType, SessionEvent, ToolCall, ToolResult
from src.core.event_bus import emit as _emit
from src.core.session_usage import SessionUsage
from src.core.task_router import TaskRouter
from src.core.skill_registry import SkillRegistry
from src.providers.provider_pool import get_pool
from src.tools.schemas import get_all_schemas, SCHEMA_MAP
from src.utils.logger import logger

# ---------------------------------------------------------------------------
_CORRECTION_WARNING = (
    "[SYSTEM — DO NOT REPEAT THIS TO THE USER]\n"
    "Your last response contained NO actual tool call.\n"
    "You MUST respond with a JSON function call — not text, not markdown, not Python syntax.\n"
    "  WRONG: ```run_shell_command(\"ls\")``` — this is text, nothing executes\n"
    "  RIGHT:  Call run_shell_command via the API function-calling interface\n\n"
    "Valid options:\n"
    "  1. end_thinking(message=\"<complete answer>\") — only if the task is fully done\n"
    "  2. Any other tool from the schema list\n\n"
    "Your unsent text (paste into end_thinking.message if this IS your final answer):\n"
)
_LOOP_WARNING = (
    "[SYSTEM — LOOP DETECTION]\n"
    "You have completed {n} turns. Verify you are not stuck in a cycle.\n"
    "If the task is done or cannot continue, call end_thinking() now.\n"
)
_QUOTA_PAUSE_MSG = (
    "All AI providers have reached their quota limits. "
    "Your task has been paused and saved.\n"
    "Use /resume after adding fresh keys (/addkey) or when quota resets.\n"
    "Paused task ID: {task_id}"
)
_COMPACT_SYSTEM = (
    "Summarise the following conversation turns into a dense progress report. "
    "Preserve: what was accomplished, key findings, files written, errors, what comes next. "
    "Plain text only."
)
_PLANNER_SYSTEM = (
    "You are a task planner. Decompose the user request into an ordered subtask list.\n"
    "Output ONLY a valid JSON array, no extra text.\n"
    "Tiers: complex=code/analysis, mid=execution/research, low=summarise/format.\n"
    "Round UP when uncertain.\n"
    'Each entry: {"id":N,"task":"...","tier":"complex|mid|low","skills":["..."]}\n'
    "Skills from AVAILABLE_SKILLS only. Empty array if no skill needed."
)
_REPLAN_SYSTEM = (
    "You are a task replanner. A subtask failed. Update the plan.\n"
    "Output ONLY the updated JSON array. Omit already-completed steps."
)

_BUILTIN_TOOLS = frozenset({
    "declare_step","end_thinking","run_shell_command","run_python",
    "save_memory","get_memories","use_skill","skill_manager",
})
_PARALLEL_SAFE = frozenset({
    "web_search","wikipedia_search","read_file","curl","get_memories",
    "use_skill","skill_manager","weather","list_workspace","get_job_result","db_query",
})
_SEQUENTIAL = frozenset({
    "run_shell_command","run_python","write_file","send_file","save_memory",
    "declare_step","end_thinking","update_todo","apply_patch","git",
    "remind_me","generate_qr",
})

_LEGACY_RE = re.compile(
    r"(TOOL:\s*[\w_]+\s*\|?\s*QUERY:.*|<function[=\[]\s*[\w_]+[\s>{].*?</function>)",
    re.IGNORECASE | re.DOTALL,
)
_END_THINKING_LEAK_RE = re.compile(
    r'<function[=\[\[]end_thinking[\s>]\s*\{.*?"message"\s*:\s*"((?:[^"\\]|\\.)*)"',
    re.DOTALL,
)
_TOOL_NAMES_PAT = (
    r"run_shell_command|run_python|web_search|wikipedia_search|curl|"
    r"read_file|write_file|send_file|save_memory|get_memories|"
    r"declare_step|end_thinking|use_skill|skill_manager|list_workspace"
)
_MARKDOWN_TOOL_RE = re.compile(
    r"(?:```[a-z]*\s*\n)?(?:" + _TOOL_NAMES_PAT + r")\s*\(",
    re.IGNORECASE,
)
_TEXT_END_THINKING_RE = re.compile(
    r'end_thinking\s*\(\s*message\s*=\s*["\']'
    r'((?:[^"\'\\]|\\.)*)'
    r'["\']s*\)',
    re.DOTALL,
)


def is_fatal_provider_error(err_str: str) -> bool:
    return is_fatal(classify_error(None, err_str))


def friendly_provider_error(last_error: Optional[str]) -> str:
    e = (last_error or "").lower()
    if any(t in e for t in ("quota","credit","billing","insufficient")):
        return "All providers hit their quota limit. Task paused — use /resume."
    if "429" in e or "rate limit" in e or "throttl" in e:
        return "Rate-limited. Retrying automatically."
    if "401" in e or "403" in e or "auth" in e or "api key" in e:
        return "No working API key. Add one with /addkey."
    if "timeout" in e or "timed out" in e:
        return "Provider timed out. Retrying."
    return "All providers failed. Retrying automatically."


def strip_legacy_tool_syntax(text: str) -> str:
    m = _END_THINKING_LEAK_RE.search(text)
    if m: return m.group(1).strip()
    return _LEGACY_RE.sub("", text).strip()


def _cache_key(name: str, args: Dict) -> str:
    h = hashlib.sha256(f"{name}:{json.dumps(args,sort_keys=True)}".encode()).hexdigest()
    return h[:16]


class Orchestrator:
    def __init__(
        self,
        config: Optional[Config] = None,
        sink=None, handle: str = "",
        on_message: Optional[Callable] = None,
        on_status_update: Optional[Callable] = None,
        on_tool_call: Optional[Callable] = None,
        on_tool_result: Optional[Callable] = None,
        is_cancelled: Optional[Callable] = None,
        resume_thought_history: Optional[List[Dict]] = None,
        resume_plan: Optional[List[Dict]] = None,
        resume_step_idx: int = 0,
        resume_agent_results: Optional[Dict] = None,
        resume_usage: Optional[SessionUsage] = None,
    ) -> None:
        self.config  = config or Config.get()
        self.pool    = get_pool()
        self.router  = TaskRouter.get(self.config)
        self.skills  = SkillRegistry.get()
        self.sink    = sink
        self.handle  = handle
        self.on_message       = on_message
        self.on_status_update = on_status_update
        self.on_tool_call     = on_tool_call
        self.on_tool_result   = on_tool_result
        self.is_cancelled     = is_cancelled
        self._skill_schemas: Dict[str, Any] = {}
        self._cache: Dict[str, str] = {}
        self._resume_history  = resume_thought_history
        self._resume_plan     = resume_plan
        self._resume_step_idx = resume_step_idx
        self._resume_results  = resume_agent_results or {}
        self._resume_usage    = resume_usage

    async def run(
        self,
        user_id: int, channel_id: str, message: str,
        context_str: str, system_prompt: str,
        history: Optional[List[Dict]] = None,
        summary: Optional[str] = None,
        max_turns: int = 20,
        platform: str = "telegram",
    ) -> str:
        cfg  = self.config
        pool = self.pool
        pool.reset_tool_caps(session_id=user_id)

        skill_list  = self.skills.skill_list_prompt()
        full_system = system_prompt + skill_list
        fragment    = await self._memory_fragment(user_id, message)

        if self._resume_history is not None:
            thought_history = self._resume_history
            plan     = self._resume_plan
            step_idx = self._resume_step_idx
            results  = self._resume_results
            usage    = self._resume_usage or SessionUsage(session_id=channel_id)
            logger.info("session_resumed", step=step_idx)
        else:
            from src.core.complexity import classify_complexity
            complexity = await classify_complexity(message, cfg, pool, user_id)
            logger.info("complexity", v=complexity, text=message[:80])
            thought_history = [
                {"role": "system", "content": full_system},
                {"role": "user",   "content": f"Request: {message}{fragment}\n\nContext:\n{context_str}"},
            ]
            plan = None; step_idx = 0; results: Dict = {}
            usage = SessionUsage(session_id=channel_id)
            if complexity == "complex_multi" and getattr(cfg, "planning_enabled", True):
                plan = await self._plan(message, context_str, full_system, user_id, pool)
                if plan:
                    await _emit(channel_id, SessionEvent(EventType.INTENT,
                        payload={"title": f"Plan: {len(plan)} steps", "status": "running"}))

        base_schemas = [s.to_openai() for s in get_all_schemas() if s.name in _BUILTIN_TOOLS]

        _final: Optional[str] = None
        _no_tc = 0; _retry = 0; _pending: Optional[str] = None
        resp: Any = None; last_error: Optional[str] = None

        for turn in range(max_turns):
            if self._cancelled(channel_id):
                await self._deliver(channel_id, "Task stopped.")
                return "Task stopped."

            if turn > 0 and turn % 20 == 0:
                thought_history.append({"role":"user","content":_LOOP_WARNING.format(n=turn)})

            compact_at = getattr(cfg, "compact_after_turns", 12)
            if compact_at > 0 and len(thought_history) > compact_at * 2 + 2:
                thought_history = await self._compact(thought_history, user_id, pool)

            await _emit(channel_id, SessionEvent(EventType.THINKING, turn=turn))

            tier = self.router.tier_for_step(plan[step_idx]) if plan and step_idx < len(plan) else "mid"
            schemas = base_schemas + list(self._skill_schemas.values())

            resp, last_error, ekind = await self._provider_request(
                user_id, tier, thought_history, schemas
            )

            if resp is None:
                if ekind == ErrorKind.FATAL:
                    msg = friendly_provider_error(last_error)
                    await self._deliver(channel_id, msg)
                    return msg

                if ekind == ErrorKind.QUOTA_EXHAUSTED and getattr(cfg, "auto_pause_on_quota", True):
                    tid = await self._pause(user_id, channel_id, platform, message,
                                           thought_history, plan, step_idx, results,
                                           usage, full_system, summary)
                    notice = _QUOTA_PAUSE_MSG.format(task_id=tid)
                    await self._deliver(channel_id, notice)
                    return notice

                _retry += 1
                if _retry >= 3 and _pending is not None:
                    _final = _pending; break

                action = await self._countdown(channel_id, 30, _retry, last_error)
                if action == "stop": return "Task stopped."
                continue

            _retry = 0
            # Token tracking
            if resp.usage and getattr(cfg, "track_token_usage", True):
                usage.record(
                    provider=getattr(resp, "provider_name", ""),
                    model=getattr(resp, "model", ""),
                    input_tokens=resp.usage.get("prompt_tokens", 0),
                    output_tokens=resp.usage.get("completion_tokens", 0),
                )

            if resp.has_tool_calls:
                _no_tc = 0; step_failed = False
                safe = [tc for tc in resp.tool_calls if tc.name in _PARALLEL_SAFE and tc.name not in _SEQUENTIAL]
                seq  = [tc for tc in resp.tool_calls if tc.name not in _PARALLEL_SAFE or tc.name in _SEQUENTIAL]
                all_safe = bool(safe) and not seq

                if all_safe:
                    raw_results = await asyncio.gather(
                        *[self._exec_tool(tc, user_id, full_system, channel_id, turn) for tc in safe],
                        return_exceptions=True,
                    )
                    pairs = list(zip(safe, raw_results))
                else:
                    pairs = []
                    for tc in resp.tool_calls:
                        if self._cancelled(channel_id): return "Task stopped."
                        r = await self._exec_tool(tc, user_id, full_system, channel_id, turn)
                        pairs.append((tc, r))

                for tc, tr in pairs:
                    if isinstance(tr, BaseException): tr = f"Error: {tr}"
                    tr = str(tr)
                    max_c = getattr(cfg, "max_tool_result_chars", 8000)
                    if max_c > 0 and len(tr) > max_c:
                        tr = tr[:max_c] + f"\n...[{len(tr)-max_c} chars truncated]"
                    if tr.startswith("__SEND_FILE__:"):
                        parts = tr.split(":", 2)
                        fp, cap = (parts[1] if len(parts)>1 else ""), (parts[2] if len(parts)>2 else "")
                        if fp and self.sink:
                            ok = await self.sink.send_file(channel_id, fp, cap)
                            tr = f"File sent: {fp}" if ok else f"Failed: {fp}"
                    if tr.startswith("__END_THINKING__:"):
                        _final = tr[len("__END_THINKING__:"):]
                        thought_history += [self._tc_msg(tc), self._tr_msg(tc, "[acknowledged]")]
                        break
                    await self._fire_tr(channel_id, tc, tr, turn)
                    if tr.lower().startswith("error") or "failed" in tr.lower()[:40]:
                        step_failed = True
                    thought_history += [self._tc_msg(tc), self._tr_msg(tc, tr)]
                    results.setdefault(f"turn_{turn}", []).append({"output": tr, "tool_used": tc.name})

                if _final is not None: break

                if plan and step_failed and step_idx < len(plan) and getattr(cfg, "dynamic_replan_on_failure", True):
                    np = await self._replan(plan, step_idx, thought_history, user_id, pool)
                    if np: plan = np; step_idx = 0
                else:
                    if plan and step_idx < len(plan): step_idx += 1
                _pending = None; turn_inc = True
            else:
                _no_tc += 1
                content = (resp.content or "").strip()

                # Fast-path: model wrote end_thinking as text
                m = _TEXT_END_THINKING_RE.search(content)
                if m: _final = m.group(1).strip(); break

                has_text_tools = bool(_MARKDOWN_TOOL_RE.search(content))
                if has_text_tools:
                    names = ", ".join(m.group().rstrip("(") for m in _MARKDOWN_TOOL_RE.finditer(content))
                    correction = (
                        f"[SYSTEM] You wrote tool calls as TEXT ({names}). Nothing executed.\n"
                        "Tool calls must be JSON API calls, NOT text/markdown.\n"
                        "Re-read the OPERATIONAL RULES and call the FIRST tool as a real API call."
                    )
                    _pending = None
                else:
                    correction = _CORRECTION_WARNING + content
                    _pending = content or _pending

                if _no_tc > 3:
                    _final = _pending or content or "Task complete."; break

                thought_history.append({"role": "user", "content": correction})
                _retry = 0
                continue

            # Explicit continue used above — this is reached only after tool_calls branch
            # (Python for-loop continues automatically, but we keep explicit for clarity)

        if _final is None:
            _final = (getattr(resp,"content",None) or "").strip() or "Task complete."
            logger.warning("max_turns_reached")

        output = strip_legacy_tool_syntax(_final) or _final
        if getattr(cfg, "track_token_usage", True) and usage.total_calls:
            output += f"\n\n---\n{usage.summary()}"

        await _emit(channel_id, SessionEvent(EventType.MESSAGE,
            payload={"text": output, "final": True}))
        if self.on_message:
            await self.on_message(output, results)
        return output

    # ── Tool execution ────────────────────────────────────────────────────────

    async def _exec_tool(self, tc, user_id: int, system_prompt: str,
                         channel_id: str, turn: int) -> str:
        await self._fire_tc(channel_id, tc, turn)
        cfg = self.config

        if tc.name in _PARALLEL_SAFE and tc.name not in ("use_skill","skill_manager"):
            ck = _cache_key(tc.name, tc.arguments)
            if ck in self._cache:
                logger.debug("cache_hit", tool=tc.name); return self._cache[ck]

        min_free = getattr(cfg, "min_free_disk_mb", 0)
        if min_free > 0 and tc.name in ("run_shell_command","run_python"):
            ws = str(Path.home() / ".rika" / "shared")
            free_mb = shutil.disk_usage(ws).free // (1024*1024)
            if free_mb < min_free:
                return f"Error: only {free_mb} MB free, minimum {min_free} MB required."

        if tc.name == "use_skill":
            sn = tc.arguments.get("skill_name","")
            result = self.skills.activate_skill(sn)
            meta   = self.skills.get_skill(sn)
            if meta:
                for sname in meta.get("tools",[]):
                    if sname in SCHEMA_MAP:
                        self._skill_schemas[sname] = SCHEMA_MAP[sname].to_openai()
            return result

        from src.agents.agent_factory import execute_tool
        result = str(await execute_tool(tc.name, tc.arguments, user_id, system_prompt=system_prompt))

        if tc.name in _PARALLEL_SAFE:
            self._cache[_cache_key(tc.name, tc.arguments)] = result
        return result

    # ── History compaction ────────────────────────────────────────────────────

    async def _compact(self, history: List[Dict], user_id: int, pool) -> List[Dict]:
        keep = max(4, len(history)//3)
        sys_msgs  = [m for m in history if m.get("role")=="system"]
        rest      = [m for m in history if m.get("role")!="system"]
        old, new  = rest[:-keep], rest[-keep:]
        if not old: return history
        turns_text = "\n".join(
            f"{m['role'].upper()}: {(m.get('content') or '')[:400]}"
            for m in old
        )[:6000]
        msgs = [{"role":"system","content":_COMPACT_SYSTEM},
                {"role":"user","content":turns_text}]
        try:
            provider, payload = self.router.payload_for_tier("low", pool, msgs)
            resp, _, _ = await self._raw_call(user_id, provider, payload, [])
            if resp and resp.content:
                compacted = sys_msgs + [{"role":"system","content":f"[SUMMARY]\n{resp.content.strip()}"}] + new
                logger.info("history_compacted", before=len(history), after=len(compacted))
                return compacted
        except Exception as e:
            logger.warning("compact_failed", error=str(e))
        return history

    # ── Pause task ────────────────────────────────────────────────────────────

    async def _pause(self, user_id, channel_id, platform, message,
                     thought_history, plan, step_idx, results,
                     usage, system_prompt, summary) -> str:
        from src.core.session_store import PausedTask, TaskStore
        task = PausedTask.create(
            user_id=user_id, channel_id=channel_id, platform=platform,
            original_message=message, thought_history=thought_history,
            plan=plan, current_step_idx=step_idx, agent_results=results,
            session_usage=usage.to_dict(), system_prompt=system_prompt,
            summary=summary, reason="quota_exhausted",
        )
        TaskStore.get().save(task)
        return task.task_id

    # ── Planning ──────────────────────────────────────────────────────────────

    async def _plan(self, message, ctx, system_prompt, user_id, pool):
        skills = ", ".join(s["name"] for s in self.skills.all_skills()) or "none"
        msgs = [{"role":"system","content":_PLANNER_SYSTEM},
                {"role":"user","content":f"AVAILABLE_SKILLS: {skills}\n\nREQUEST: {message}\n\nCONTEXT: {ctx[:500]}"}]
        try:
            provider, payload = self.router.payload_for_tier("mid", pool, msgs)
            resp, _, _ = await self._raw_call(user_id, provider, payload, [])
            if resp is None: return None
            raw = re.sub(r"^```(?:json)?\s*|\s*```$","", (resp.content or "").strip())
            p = json.loads(raw)
            return p if isinstance(p, list) and p else None
        except Exception as e:
            logger.warning("plan_failed", error=str(e)); return None

    async def _replan(self, plan, idx, history, user_id, pool):
        msgs = [{"role":"system","content":_REPLAN_SYSTEM},
                {"role":"user","content":
                 f"FAILED: {json.dumps(plan[idx])}\n"
                 f"REMAINING: {json.dumps(plan[idx+1:])}\n"
                 f"LAST: {history[-1].get('content','')[:400]}"}]
        try:
            provider, payload = self.router.payload_for_tier("mid", pool, msgs)
            resp, _, _ = await self._raw_call(user_id, provider, payload, [])
            if resp is None: return None
            raw = re.sub(r"^```(?:json)?\s*|\s*```$","", (resp.content or "").strip())
            p = json.loads(raw)
            return p if isinstance(p, list) else None
        except Exception as e:
            logger.warning("replan_failed", error=str(e)); return None

    # ── Provider request ──────────────────────────────────────────────────────

    async def _provider_request(self, user_id, tier, messages, schemas
                                ) -> Tuple[Any, Optional[str], ErrorKind]:
        cfg   = self.config
        chain = self.router.get_attempt_chain(tier)
        last_err: Optional[str] = None
        last_kind = ErrorKind.TRANSIENT

        for _, provider, model in chain:
            if self._cancelled(""): return None, "cancelled", ErrorKind.FATAL
            payload = {"model": model, "messages": messages}
            resp, err, kind = await self._raw_call(
                user_id, provider, payload, schemas,
                max_retries=getattr(cfg,"provider_max_retries",2),
                base_delay=getattr(cfg,"provider_retry_delay",2.0),
            )
            if resp is not None: return resp, None, ErrorKind.TRANSIENT
            last_err = err; last_kind = kind
            if kind == ErrorKind.FATAL: break
            await asyncio.sleep(0.5)

        return None, last_err, last_kind

    async def _raw_call(self, user_id, provider, payload, schemas,
                        max_retries=2, base_delay=2.0
                        ) -> Tuple[Any, Optional[str], ErrorKind]:
        last_err: Optional[str] = None; last_kind = ErrorKind.TRANSIENT
        for attempt in range(max_retries+1):
            try:
                resp = await asyncio.wait_for(
                    self.pool.request_with_key_structured(user_id, provider, payload, schemas),
                    timeout=60.0,
                )
                if resp and (resp.has_tool_calls or resp.content):
                    return resp, None, ErrorKind.TRANSIENT
                break
            except asyncio.TimeoutError as e:
                last_err = f"Provider {provider} timed out"; last_kind = ErrorKind.NETWORK
            except Exception as e:
                last_err = str(e); last_kind = classify_error(e, last_err)
            if is_fatal(last_kind) or is_quota(last_kind): break
            if attempt < max_retries and is_retryable(last_kind):
                await asyncio.sleep(base_delay * (2**attempt))
        return None, last_err, last_kind

    # ── Countdown ─────────────────────────────────────────────────────────────

    async def _countdown(self, channel_id, secs, attempt, err) -> str:
        if self.sink is None:
            await asyncio.sleep(secs); return "done"
        self.sink.init_countdown_events(channel_id)
        if self.handle:
            try:
                line = friendly_provider_error(err).split("\n")[0]
                note = "\nConsider /addkey to add fresh keys." if attempt > 1 else ""
                await self.sink.edit_message(self.handle, f"{line}{note}")
            except Exception: pass
        return await self.sink.show_countdown(
            handle=self.handle, wait_seconds=secs, attempt=attempt,
            on_stop=asyncio.Event(), on_retry=asyncio.Event(),
        )

    # ── Memory ────────────────────────────────────────────────────────────────

    async def _memory_fragment(self, user_id, message) -> str:
        try:
            from src.db.chat_store import get_relevant_memories
            mems = await get_relevant_memories(user_id, message, k=5)
            if not mems: return ""
            return "\n\nRELEVANT PAST CONTEXT:\n" + "\n".join(f"  {k}: {v}" for k,v in mems.items())
        except Exception as e:
            logger.debug("mem_frag_failed", error=str(e)); return ""

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _cancelled(self, cid) -> bool:
        if self.is_cancelled:
            try: return bool(self.is_cancelled(cid))
            except: pass
        return False

    async def _deliver(self, cid, text):
        if self.sink:
            try: await self.sink.send_text(cid, text)
            except: pass

    def _tc_msg(self, tc) -> Dict:
        return {"role":"assistant","content":None,
                "tool_calls":[{"id":tc.call_id,"type":"function",
                    "function":{"name":tc.name,"arguments":json.dumps(tc.arguments)}}]}

    def _tr_msg(self, tc, result: str) -> Dict:
        return {"role":"tool","content":result,"tool_call_id":tc.call_id}

    async def _fire_tc(self, cid, tc, turn):
        await _emit(cid, SessionEvent(EventType.TOOL_CALL,
            payload={"tool":tc.name,"args":tc.arguments}, turn=turn))
        if self.on_tool_call:
            try: await self.on_tool_call(ToolCall(name=tc.name,arguments=tc.arguments,call_id=tc.call_id))
            except: pass

    async def _fire_tr(self, cid, tc, result, turn):
        await _emit(cid, SessionEvent(EventType.TOOL_RESULT,
            payload={"tool":tc.name,"result":result[:300],"success":not result.lower().startswith("error")},
            turn=turn))
        if self.on_tool_result:
            try: await self.on_tool_result(ToolResult(tool_name=tc.name,result=result,call_id=tc.call_id))
            except: pass
