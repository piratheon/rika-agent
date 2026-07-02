"""Orchestrator — platform-agnostic agent loop with multi-tier model routing.

Features restored/added vs the friends' version:
  - 3-way complexity classifier drives planning phase (complex_multi only)
  - Planning phase: MidLLM produces structured subtask plan with tier labels
  - Per-step model selection via TaskRouter (complex/mid/low -> provider/model)
  - Tier fallback chain: complex->mid->low on missing key (error only if all tiers empty)
  - On-demand skill loading via SkillRegistry.activate_skill()
  - Skill list injected into every system prompt (names + descriptions only)
  - Unlimited auto-retry with countdown on rate-limit/transient errors
  - _pending_content deadlock guard (deliver last good answer after 3 failed retries)
  - end_thinking sentinel (__END_THINKING__:) and __SEND_FILE__: routing
  - _no_tool_corr correction loop with CORRECTION_WARNING injection
  - pool.reset_tool_caps() per message
  - Dynamic replanning on step failure (not pre-planned conditional branches)
  - Loop-detection warning injected after 20 turns
  - Memory fragment injection
  - strip_legacy_tool_syntax safety net
  - is_fatal_provider_error, friendly_provider_error exported for reuse
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.config import Config
from src.core.models import EventType, SessionEvent, ToolCall, ToolResult
from src.core.event_bus import emit as _emit_event
from src.core.task_router import TaskRouter
from src.core.skill_registry import SkillRegistry
from src.providers.provider_pool import get_pool
from src.tools.schemas import get_all_schemas, get_schemas_for_tools, SCHEMA_MAP
from src.utils.logger import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CORRECTION_WARNING = (
    "[SYSTEM — DO NOT REPEAT THIS TO THE USER]\n"
    "You produced text output without a tool call. This is not allowed.\n"
    "You MUST respond with a tool call. Either:\n"
    '  1. end_thinking(message="...") to deliver your final answer\n'
    "  2. Any other tool to continue processing\n\n"
    "Your unsent text (copy into end_thinking if it is your final answer):\n"
)

_LOOP_WARNING = (
    "[SYSTEM — LOOP DETECTION]\n"
    "You have completed {n} turns. Check whether you are stuck in a repetitive cycle.\n"
    "If the task is complete or cannot be completed, call end_thinking() now.\n"
    "If you genuinely need more steps, continue — but be concise.\n"
)

# Built-in tools always included regardless of skills loaded
_BUILTIN_TOOL_NAMES = {
    "declare_step", "end_thinking", "run_shell_command", "run_python",
    "save_memory", "get_memories", "use_skill",
}

_LEGACY_TOOL_RE = re.compile(
    r"(TOOL:\s*[\w_]+\s*\|?\s*QUERY:.*"
    r"|<function[=\[]\s*[\w_]+[\s>{].*?</function>)",
    re.IGNORECASE | re.DOTALL,
)
_END_THINKING_LEAK_RE = re.compile(
    r'<function[=\[\[]end_thinking[\s>]\s*\{.*?"message"\s*:\s*"((?:[^"\\]|\\.)*)"',
    re.DOTALL,
)

_PLANNER_SYSTEM = """You are a task planner. Given a user request, decompose it into
an ordered list of concrete subtasks. For each subtask output ONLY valid JSON.

Complexity tiers:
  complex — writing/debugging code from scratch, architectural decisions,
            analysing ambiguous/contradictory information, creative generation
  mid     — executing/running code, multi-step tool chains with clear instructions,
            structured data extraction, calling APIs, web research
  low     — summarising already-fetched short content, translation, formatting,
            deterministic single-tool calls (read file, list dir, simple math)

When uncertain, round UP to the higher tier.

Output format — a JSON array, no extra text:
[
  {"id": 1, "task": "...", "tier": "complex|mid|low",
   "skills": ["skill_name", ...]},
  ...
]

skill names must come from the AVAILABLE_SKILLS list provided.
If no skill is needed, use an empty array.
"""

_REPLAN_SYSTEM = """You are a task replanner. A subtask failed.
Given the failure details and remaining steps, produce an updated JSON plan array
(same format as before). You may split, merge, or replace steps as needed.
Do not repeat already-completed steps. Output ONLY the updated JSON array."""


# ---------------------------------------------------------------------------
# Exported helpers (importable by Discord bot, API server, tests)
# ---------------------------------------------------------------------------

def is_fatal_provider_error(err_str: str) -> bool:
    """True for errors that must NOT be retried (auth, schema rejection, etc.)."""
    e = (err_str or "").lower()
    return any(t in e for t in (
        "tool_use_failed", "401", "403",
        "invalid_api_key", "authentication", "permission_denied",
        "access_denied", "account_deactivated", "account suspended",
        "model_not_found", "no such model", "invalid request",
    ))


def friendly_provider_error(last_error: str | None) -> str:
    err = (last_error or "").lower()
    if "403" in err or "access denied" in err or "network" in err:
        return (
            "Could not reach the AI provider — likely a network or VPN issue.\n"
            "Retrying — or click Stop to cancel."
        )
    if "429" in err or "quota" in err or "rate limit" in err or "exhausted" in err:
        return (
            "All providers are rate-limited or over quota.\n"
            "Retrying automatically — or click Stop to cancel."
        )
    if "401" in err or "unauthorized" in err or "no api key" in err:
        return (
            "No working API key found for any provider.\n"
            "Add a key with /addkey or check /status."
        )
    if "timeout" in err or "timed out" in err:
        return (
            "The AI provider timed out.\n"
            "Retrying — or click Stop to cancel."
        )
    return (
        "All AI providers failed to respond right now.\n"
        "Retrying automatically — or click Stop to cancel."
    )


def strip_legacy_tool_syntax(text: str) -> str:
    leak = _END_THINKING_LEAK_RE.search(text)
    if leak:
        return leak.group(1).strip()
    return _LEGACY_TOOL_RE.sub("", text).strip()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    """Platform-agnostic orchestration loop.

    All I/O goes through the EventSink (sink) and on_message callback.
    No Telegram or Discord imports.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        sink=None,
        handle: str = "",
        on_message: Optional[Callable] = None,
        on_status_update: Optional[Callable] = None,
        on_tool_call: Optional[Callable] = None,
        on_tool_result: Optional[Callable] = None,
        is_cancelled: Optional[Callable] = None,
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

        # Active skill schemas accumulated during session
        self._active_skill_tools: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        user_id: int,
        channel_id: str,
        message: str,
        context_str: str,
        system_prompt: str,
        history: Optional[List[Dict]] = None,
        summary: Optional[str] = None,
        max_turns: int = 20,
    ) -> str:
        cfg  = self.config
        pool = self.pool

        pool.reset_tool_caps(session_id=user_id)

        # Inject skill list into system prompt
        skill_list = self.skills.skill_list_prompt()
        full_system = system_prompt + skill_list

        fragment_str = await self._get_memory_fragment(user_id, message)

        thought_history: List[Dict] = [
            {"role": "system", "content": full_system},
            {
                "role": "user",
                "content": (
                    f"Request: {message}{fragment_str}\n\nContext:\n{context_str}"
                ),
            },
        ]

        # Classify complexity to decide whether planning is needed
        from src.core.complexity import classify_complexity
        complexity = await classify_complexity(message, cfg, pool, user_id)
        logger.info("complexity_result", complexity=complexity, text=message[:80])

        plan: Optional[List[Dict]] = None
        if complexity == "complex_multi":
            plan = await self._plan(message, context_str, full_system, user_id, pool)
            if plan:
                logger.info("plan_generated", steps=len(plan))
                await _emit_event(channel_id, SessionEvent(
                    EventType.INTENT,
                    payload={"title": f"Plan: {len(plan)} steps", "status": "running"},
                ))

        # Build tool schemas: built-ins always present, skills loaded on demand
        base_schemas = self._build_base_schemas()
        agent_results: Dict = {}
        completed_steps: List[int] = []
        current_step_idx = 0

        _final_message: str | None = None
        _no_tool_corr: int = 0
        _provider_retry: int = 0
        _pending_content: str | None = None

        turn = 0
        while turn < max_turns:
            if self._cancelled(channel_id):
                await self._deliver(channel_id, "Task stopped by user.")
                return "Task stopped by user."

            # Loop detection warning
            if turn > 0 and turn % 20 == 0:
                thought_history.append({
                    "role": "user",
                    "content": _LOOP_WARNING.format(n=turn),
                })

            await _emit_event(channel_id, SessionEvent(EventType.THINKING, turn=turn))

            # Select tier for this turn
            tier = "mid"
            if plan and current_step_idx < len(plan):
                tier = self.router.tier_for_step(plan[current_step_idx])

            # Build schemas (built-ins + active skill tools)
            current_schemas = base_schemas + list(self._active_skill_tools.values())

            resp, last_error = await self._provider_request(
                user_id, tier, thought_history, current_schemas,
            )

            # All providers failed
            if resp is None:
                # No-key error → do NOT retry, surface immediately
                if last_error and ("no api key" in last_error.lower()
                                   or "401" in last_error
                                   or "no working" in last_error.lower()):
                    msg = friendly_provider_error(last_error)
                    await self._deliver(channel_id, msg)
                    return msg

                _provider_retry += 1

                if _provider_retry >= 3 and _pending_content is not None:
                    logger.warning("orchestration_deadlock_guard", turn=turn)
                    _final_message = _pending_content
                    break

                action = await self._countdown(channel_id, 30, _provider_retry, last_error)
                if action == "stop":
                    return "Task stopped by user."
                continue

            _provider_retry = 0

            # Tool calls
            if resp.has_tool_calls:
                _no_tool_corr = 0
                step_failed = False

                for tool_call in resp.tool_calls:
                    if self._cancelled(channel_id):
                        return "Task stopped by user."

                    t_name = tool_call.name
                    t_args = tool_call.arguments

                    await self._fire_tc_event(channel_id, tool_call, turn)

                    # Skill activation intercept
                    if t_name == "use_skill":
                        skill_name = t_args.get("skill_name", "")
                        tool_result = self.skills.activate_skill(skill_name)
                        # Inject skill schemas into active set
                        skill_meta = self.skills.get_skill(skill_name)
                        if skill_meta:
                            for sname in skill_meta.get("tools", []):
                                if sname in SCHEMA_MAP:
                                    self._active_skill_tools[sname] = (
                                        SCHEMA_MAP[sname].to_openai()
                                    )
                    else:
                        from src.agents.agent_factory import execute_tool
                        tool_result: str = await execute_tool(
                            t_name, t_args, user_id,
                            system_prompt=full_system,
                        )

                    # File delivery sentinel
                    if tool_result.startswith("__SEND_FILE__:"):
                        parts = tool_result.split(":", 2)
                        file_path = parts[1] if len(parts) > 1 else ""
                        caption   = parts[2] if len(parts) > 2 else ""
                        if file_path and self.sink:
                            ok = await self.sink.send_file(channel_id, file_path, caption)
                            tool_result = (
                                f"File sent: {file_path}" if ok
                                else f"Failed to send file: {file_path}"
                            )

                    # end_thinking sentinel
                    if tool_result.startswith("__END_THINKING__:"):
                        _final_message = tool_result[len("__END_THINKING__:"):]
                        thought_history.append(self._tc_msg(tool_call))
                        thought_history.append(self._tr_msg(tool_call, "[end_thinking acknowledged]"))
                        break

                    await self._fire_tr_event(channel_id, tool_call, tool_result, turn)

                    # Step failure detection
                    if (tool_result.lower().startswith("error") or
                            "failed" in tool_result.lower()[:30]):
                        step_failed = True

                    thought_history.append(self._tc_msg(tool_call))
                    thought_history.append(self._tr_msg(tool_call, tool_result))

                    key = f"turn_{turn}"
                    agent_results.setdefault(key, []).append(
                        {"output": tool_result, "tool_used": t_name}
                    )

                if _final_message is not None:
                    break

                # Dynamic replanning on step failure
                if (plan and step_failed and current_step_idx < len(plan)
                        and getattr(cfg, "dynamic_replan_on_failure", True)):
                    new_plan = await self._replan(
                        plan, current_step_idx, thought_history, user_id, pool
                    )
                    if new_plan:
                        logger.info("replan_accepted", new_steps=len(new_plan))
                        plan = new_plan
                        current_step_idx = 0
                else:
                    if plan and current_step_idx < len(plan):
                        completed_steps.append(plan[current_step_idx]["id"])
                        current_step_idx += 1

                turn += 1
                _pending_content = None
                continue

            # No tool call — correction loop
            _no_tool_corr += 1
            logger.warning("no_tool_call", turn=turn, correction=_no_tool_corr,
                           preview=(resp.content or "")[:80])

            if _no_tool_corr > 3:
                _final_message = (resp.content or "").strip() or "Task complete."
                break

            _pending_content = (resp.content or "").strip() or _pending_content
            thought_history.append({
                "role": "user",
                "content": _CORRECTION_WARNING + (resp.content or ""),
            })
            _provider_retry = 0

        # Finalize
        if _final_message is None:
            if resp is not None:
                _final_message = (resp.content or "").strip()
            if not _final_message:
                _final_message = "Task complete."
            logger.warning("orchestration_max_turns")

        output = strip_legacy_tool_syntax(_final_message) or _final_message

        await _emit_event(channel_id, SessionEvent(
            EventType.MESSAGE, payload={"text": output, "final": True}
        ))
        if self.on_message:
            await self.on_message(output, agent_results)

        return output

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    async def _plan(
        self, message: str, context_str: str, system_prompt: str,
        user_id: int, pool,
    ) -> Optional[List[Dict]]:
        """Call MidLLM to produce a structured subtask plan."""
        skill_names = [s["name"] for s in self.skills.all_skills()]
        available_skills = ", ".join(skill_names) if skill_names else "none"

        messages = [
            {"role": "system", "content": _PLANNER_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"AVAILABLE_SKILLS: {available_skills}\n\n"
                    f"USER REQUEST: {message}\n\n"
                    f"CONTEXT: {context_str[:500]}"
                ),
            },
        ]
        try:
            provider, payload = self.router.payload_for_tier("mid", pool, messages)
        except RuntimeError as exc:
            logger.warning("planner_no_key", error=str(exc))
            return None

        try:
            resp, _ = await self._raw_provider_call(user_id, provider, payload, [])
            if resp is None:
                return None
            raw = (resp.content or "").strip()
            # Strip markdown code fences if present
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            plan = json.loads(raw)
            if isinstance(plan, list) and plan:
                return plan
        except (json.JSONDecodeError, Exception) as exc:
            logger.warning("planner_parse_failed", error=str(exc))
        return None

    async def _replan(
        self, current_plan: List[Dict], failed_step_idx: int,
        thought_history: List[Dict], user_id: int, pool,
    ) -> Optional[List[Dict]]:
        """Call MidLLM to replan after a step failure."""
        failed_step = current_plan[failed_step_idx]
        remaining   = current_plan[failed_step_idx + 1:]

        messages = [
            {"role": "system", "content": _REPLAN_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"FAILED_STEP: {json.dumps(failed_step)}\n"
                    f"REMAINING_STEPS: {json.dumps(remaining)}\n"
                    f"LAST_TOOL_RESULT: "
                    f"{thought_history[-1].get('content', '')[:400]}"
                ),
            },
        ]
        try:
            provider, payload = self.router.payload_for_tier("mid", pool, messages)
        except RuntimeError:
            return None

        try:
            resp, _ = await self._raw_provider_call(user_id, provider, payload, [])
            if resp is None:
                return None
            raw = re.sub(r"^```(?:json)?\s*", "", (resp.content or "").strip())
            raw = re.sub(r"\s*```$", "", raw)
            new_plan = json.loads(raw)
            if isinstance(new_plan, list):
                return new_plan
        except (json.JSONDecodeError, Exception) as exc:
            logger.warning("replan_parse_failed", error=str(exc))
        return None

    # ------------------------------------------------------------------
    # Provider request with per-tier model selection + fallback chain
    # ------------------------------------------------------------------

    async def _provider_request(
        self, user_id: int, tier: str,
        messages: List[Dict], tool_schemas: List[Dict],
    ) -> Tuple[Any, Optional[str]]:
        """Iterate the full model chain for the given tier, returning (resp, last_error).

        Within-tier: models tried in list order.
        Cross-tier:  on all-fail, falls through to the next tier in fallback_chain.
        Exhausted:   returns (None, last_error); the orchestrator countdown loop
                     retries from the top, cycling back to the first model.
        """
        # patch_model_chains: full_chain_iteration
        cfg = self.config
        _max_retries = getattr(cfg, "provider_max_retries", 2)
        _base_delay  = getattr(cfg, "provider_retry_delay", 2.0)

        chain = self.router.get_attempt_chain(tier)
        last_error: Optional[str] = None

        for attempt_tier, provider, model in chain:
            if self._cancelled(""):
                return None, "cancelled"

            payload = {**{"messages": messages}, "model": model}
            resp, err = await self._raw_provider_call(
                user_id, provider, payload, tool_schemas,
                max_retries=_max_retries, base_delay=_base_delay,
            )
            if resp is not None:
                if attempt_tier != tier:
                    logger.warning(
                        "task_router_tier_fallback",
                        requested=tier, used=attempt_tier,
                        provider=provider, model=model,
                    )
                return resp, None

            last_error = err
            err_lower = (err or "").lower()

            # Hard auth failure for this specific key — skip to next model.
            # Rate-limit / quota / timeout are retried within _raw_provider_call
            # already; if we reach here they've all been exhausted for this model.
            auth_fail = any(t in err_lower for t in (
                "401", "403", "invalid_api_key", "authentication",
                "permission_denied", "account_deactivated",
            ))
            if auth_fail:
                logger.warning(
                    "task_router_skip_auth_fail",
                    provider=provider, model=model, error=err,
                )
                continue  # try next model in chain

            # No-key error for an entire provider — skip to next model
            if "no working" in err_lower or "no api key" in err_lower:
                continue

            # Any other error (quota, timeout already exhausted): continue chain
            logger.warning(
                "task_router_model_failed",
                provider=provider, model=model,
                tier=attempt_tier, error=(err or "")[:120],
            )
            # Small pause between model switches to avoid hammering APIs
            import asyncio as _asyncio
            await _asyncio.sleep(1.0)

        # All models in the full chain exhausted for this round.
        # Return (None, last_error) — the orchestrator countdown loop will
        # call us again, restarting from the top of the chain (natural cycle-back).
        logger.warning(
            "task_router_chain_exhausted",
            tier=tier,
            chain_len=len(chain),
            last_error=(last_error or "")[:120],
        )
        return None, last_error

    async def _raw_provider_call(
        self, user_id: int, provider: str, payload: Dict,
        tool_schemas: List[Dict],
        max_retries: int = 2, base_delay: float = 2.0,
    ) -> Tuple[Any, Optional[str]]:
        last_error: Optional[str] = None
        for attempt in range(max_retries + 1):
            try:
                resp = await asyncio.wait_for(
                    self.pool.request_with_key_structured(
                        user_id, provider, payload, tool_schemas,
                    ),
                    timeout=60.0,
                )
                if resp and (resp.has_tool_calls or resp.content):
                    return resp, None
                break
            except asyncio.TimeoutError:
                last_error = f"Provider {provider} timed out"
                if attempt < max_retries:
                    await asyncio.sleep(base_delay * (2 ** attempt))
            except Exception as exc:
                last_error = str(exc)
                if is_fatal_provider_error(last_error):
                    break
                if attempt < max_retries:
                    await asyncio.sleep(base_delay * (2 ** attempt))
        return None, last_error

    # ------------------------------------------------------------------
    # Schema helpers
    # ------------------------------------------------------------------

    def _build_base_schemas(self) -> List[Dict]:
        """Built-in schemas always present in every call."""
        schemas = get_all_schemas()
        return [
            s.to_openai()
            for s in schemas
            if s.name in _BUILTIN_TOOL_NAMES
        ]

    # ------------------------------------------------------------------
    # Countdown / retry
    # ------------------------------------------------------------------

    async def _countdown(
        self, channel_id: str, wait_seconds: int,
        attempt: int, last_error: Optional[str],
    ) -> str:
        if self.sink is None:
            await asyncio.sleep(wait_seconds)
            return "done"

        on_stop  = asyncio.Event()
        on_retry = asyncio.Event()
        self.sink.init_countdown_events(channel_id)

        if self.handle:
            try:
                err_line = friendly_provider_error(last_error).split("\n")[0]
                note = (
                    "\n\nAll providers seem over quota — consider /addkey."
                    if attempt > 1 else ""
                )
                await self.sink.edit_message(self.handle, f"{err_line}{note}")
            except Exception:
                pass

        return await self.sink.show_countdown(
            handle=self.handle,
            wait_seconds=wait_seconds,
            attempt=attempt,
            on_stop=on_stop,
            on_retry=on_retry,
        )

    # ------------------------------------------------------------------
    # Memory fragment
    # ------------------------------------------------------------------

    async def _get_memory_fragment(self, user_id: int, message: str) -> str:
        try:
            from src.db.chat_store import get_relevant_memories
            memories = await get_relevant_memories(user_id, message, k=5)
            if not memories:
                return ""
            lines = [f"  {k}: {v}" for k, v in memories.items()]
            return "\n\nRELEVANT PAST CONTEXT:\n" + "\n".join(lines)
        except Exception as exc:
            logger.debug("memory_fragment_failed", error=str(exc))
            return ""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cancelled(self, channel_id: str) -> bool:
        if self.is_cancelled:
            try:
                return bool(self.is_cancelled(channel_id))
            except Exception:
                pass
        return False

    async def _deliver(self, channel_id: str, text: str) -> None:
        if self.sink:
            try:
                await self.sink.send_text(channel_id, text)
            except Exception:
                pass

    def _tc_msg(self, tc) -> Dict:
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": tc.call_id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
            }],
        }

    def _tr_msg(self, tc, result: str) -> Dict:
        return {"role": "tool", "content": result, "tool_call_id": tc.call_id}

    async def _fire_tc_event(self, channel_id: str, tc: Any, turn: int) -> None:
        await _emit_event(channel_id, SessionEvent(
            EventType.TOOL_CALL,
            payload={"tool": tc.name, "args": tc.arguments},
            turn=turn,
        ))
        if self.on_tool_call:
            try:
                await self.on_tool_call(ToolCall(
                    name=tc.name, arguments=tc.arguments, call_id=tc.call_id
                ))
            except Exception:
                pass

    async def _fire_tr_event(
        self, channel_id: str, tc: Any, result: str, turn: int
    ) -> None:
        await _emit_event(channel_id, SessionEvent(
            EventType.TOOL_RESULT,
            payload={
                "tool": tc.name,
                "result": result[:300],
                "success": not result.lower().startswith("error"),
            },
            turn=turn,
        ))
        if self.on_tool_result:
            try:
                await self.on_tool_result(ToolResult(
                    tool_name=tc.name, result=result, call_id=tc.call_id
                ))
            except Exception:
                pass
