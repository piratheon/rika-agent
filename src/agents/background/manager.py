"""BackgroundAgentManager — AI-driven watcher orchestration.

Key changes vs v2:
- Natural language watcher creation: user says "watch my server" and the AI
  decides what to monitor, writes scripts, and registers watchers.
- CronWatcher and ScriptWatcher fire a ConcreteAgent turn WITH TOOLS, not
  just a plain LLM message. The agent can run shell commands, write files,
  restart services — it actually acts, not just describes.
- ScriptWatcher: executes agent-authored Python/shell scripts, fires on
  non-zero exit or ALERT: prefix in output.
- "ai" watcher type: plain natural language goal, AI decides the watcher type
  and config autonomously.

Interface-agnostic via EventSink — no Telegram dependency.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from src.agents.agent_models import BackgroundAgentConfig, WakeSignal
from src.agents.background.watchers import (
    CronWatcher, LogPatternWatcher, PortWatcher,
    ProcessWatcher, SystemWatcher, URLWatcher, WatcherBase,
)
from src.agents.background.script_watcher import ScriptWatcher
from src.config import Config
from src.core.event_sink import EventSink
from src.db.background_store import (
    disable_background_agent, list_user_background_agents,
    load_all_background_agents, save_background_agent,
    save_wake_event, update_agent_trigger_count,
)
from src.providers.provider_pool import get_pool
from src.utils.logger import logger


def _watcher_dir(workspace: str) -> Path:
    p = Path(workspace).expanduser() / "watchers"
    p.mkdir(parents=True, exist_ok=True)
    return p


class BackgroundAgentManager:
    """Singleton. One instance per process."""

    _instance: Optional["BackgroundAgentManager"] = None

    def __init__(self, sink: EventSink) -> None:
        self._sink = sink
        self._tasks: Dict[str, asyncio.Task] = {}
        self._wake_queue: asyncio.Queue[WakeSignal] = asyncio.Queue(maxsize=512)
        self._processor: Optional[asyncio.Task] = None
        self._started = False

    @classmethod
    def initialize(cls, sink: EventSink) -> "BackgroundAgentManager":
        cls._instance = cls(sink)
        return cls._instance

    @classmethod
    def get(cls) -> Optional["BackgroundAgentManager"]:
        return cls._instance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._processor = asyncio.create_task(
            self._wake_processor_loop(), name="wake_processor"
        )
        agents = await load_all_background_agents()
        for cfg in agents:
            if cfg.enabled:
                await self._spawn_watcher(cfg)
        logger.info("background_manager_started", active=len(self._tasks))

    async def register(self, cfg: BackgroundAgentConfig) -> None:
        await save_background_agent(cfg)
        await self._spawn_watcher(cfg)
        logger.info("background_agent_registered", id=cfg.id, type=cfg.watcher_type)

    async def create_from_natural_language(
        self, user_id: int, channel_id: str, goal: str
    ) -> List[BackgroundAgentConfig]:
        """AI-driven watcher creation from a natural language goal.

        The LLM is given the full watcher schema and asked to produce
        a JSON list of BackgroundAgentConfig-compatible objects.
        Returns the list of created configs (already registered).
        """
        cfg = Config.get()
        pool = get_pool()

        import uuid
        ws = str(Path(cfg.workspace_path).expanduser())

        schema_description = """
Available watcher types and their config fields:
- "system": {} (no config needed) — monitors CPU, memory, disk
- "process": {"process_name": "nginx"} — alerts if process stops
- "url": {"url": "https://...", "expected_status": 200} — HTTP health check
- "port": {"host": "localhost", "port": 5432} — TCP port check
- "log": {"file_path": "/var/log/app.log", "pattern": "ERROR|FATAL"} — regex log tail
- "cron": {"task_description": "check and report disk usage"} — scheduled AI task WITH tools
- "script": {"script_path": "/path/to/script.py", "cooldown_seconds": 120} — run a script file
  For "script" type, first write the script to """ + ws + """/watchers/<name>.py
  The script should: print ALERT: <message> if something is wrong, exit 0 if fine.
"""

        prompt = f"""The user wants to set up autonomous background monitoring with this goal:
"{goal}"

Based on this goal, decide what monitoring agents to create.
You have shell access and can write monitoring scripts.

{schema_description}

Return ONLY a JSON array of objects with these fields:
- watcher_type: string (one of the types above)
- name: string (short descriptive name)
- description: string (what this agent does and why)
- config: object (watcher-specific config)
- interval_seconds: integer (how often to check — minimum 30)

Be practical. For a "be my server guardian" goal, create: system watcher + any relevant process watchers.
For scheduling tasks, use "cron" type.
If the task needs a custom script, use "script" type and specify the script path.

Respond with ONLY the JSON array, no explanation."""

        payload = {
            "model": cfg.default_model,
            "messages": [
                {"role": "system", "content": cfg.system_prompt},
                {"role": "user", "content": prompt},
            ],
        }

        raw_json = ""
        for provider in (cfg.default_provider_priority or ["gemini", "groq", "openrouter"]):
            try:
                resp = await pool.request_with_key(user_id, provider, payload)
                raw_json = resp.get("output", "").strip()
                if raw_json:
                    break
            except Exception as exc:
                logger.warning("nl_watcher_llm_failed", provider=provider, error=str(exc))

        if not raw_json:
            return []

        # Strip markdown code fences if present
        if raw_json.startswith("```"):
            raw_json = "\n".join(
                line for line in raw_json.splitlines()
                if not line.strip().startswith("```")
            ).strip()

        try:
            watcher_list = json.loads(raw_json)
            if isinstance(watcher_list, dict):
                watcher_list = [watcher_list]
        except json.JSONDecodeError as exc:
            logger.error("nl_watcher_json_parse_failed", error=str(exc), raw=raw_json[:300])
            return []

        created: List[BackgroundAgentConfig] = []
        for entry in watcher_list[:8]:  # cap at 8 watchers per request
            try:
                agent_id = f"{entry.get('watcher_type', 'ai')[:4]}_{str(uuid.uuid4())[:6]}"
                agent_cfg = BackgroundAgentConfig(
                    id=agent_id,
                    user_id=user_id,
                    channel_id=channel_id,
                    watcher_type=entry["watcher_type"],
                    name=entry.get("name", agent_id),
                    description=entry.get("description", goal),
                    config=entry.get("config", {}),
                    interval_seconds=max(30, int(entry.get("interval_seconds", 60))),
                )
                await self.register(agent_cfg)
                created.append(agent_cfg)
            except Exception as exc:
                logger.error("nl_watcher_create_failed", entry=str(entry)[:200], error=str(exc))

        return created

    async def stop(self) -> None:
        """Cancel all watcher tasks and the wake processor. Called on shutdown."""
        if self._processor and not self._processor.done():
            self._processor.cancel()
            try:
                await self._processor
            except asyncio.CancelledError:
                pass
            self._processor = None

        if self._tasks:
            for task in list(self._tasks.values()):
                task.cancel()
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
            self._tasks.clear()

        self._started = False
        logger.info("background_manager_stopped")

    async def stop_agent(self, agent_id: str) -> bool:
        task = self._tasks.pop(agent_id, None)
        if task is None:
            return False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await disable_background_agent(agent_id)
        logger.info("background_agent_stopped", id=agent_id)
        return True

    async def list_for_user(self, user_id: int) -> list:
        return await list_user_background_agents(user_id)

    # ------------------------------------------------------------------
    # Internal — watcher spawning
    # ------------------------------------------------------------------

    def _build_watcher(self, cfg: BackgroundAgentConfig) -> Optional[WatcherBase]:
        c = cfg.config
        t = cfg.watcher_type
        try:
            if t == "system":
                return SystemWatcher(
                    cpu_threshold=float(c.get("cpu_threshold", 85)),
                    mem_threshold=float(c.get("mem_threshold", 90)),
                    disk_threshold=float(c.get("disk_threshold", 90)),
                    load_threshold=float(c.get("load_threshold", 4.0)),
                )
            if t == "process":
                return ProcessWatcher(process_name=c["process_name"])
            if t == "url":
                return URLWatcher(
                    url=c["url"],
                    expected_status=int(c.get("expected_status", 200)),
                    timeout=int(c.get("timeout", 10)),
                )
            if t == "port":
                return PortWatcher(
                    host=c.get("host", "localhost"),
                    port=int(c["port"]),
                )
            if t == "log":
                return LogPatternWatcher(
                    file_path=c["file_path"],
                    pattern=c["pattern"],
                    cooldown_seconds=int(c.get("cooldown_seconds", 300)),
                )
            if t in ("cron", "ai"):
                return CronWatcher(
                    task_description=c.get("task_description", cfg.description)
                )
            if t == "script":
                return ScriptWatcher(
                    script_path=c["script_path"],
                    working_dir=c.get("working_dir"),
                    cooldown_seconds=int(c.get("cooldown_seconds", 120)),
                )
        except (KeyError, ValueError, TypeError) as exc:
            logger.error("watcher_build_failed", agent_id=cfg.id, type=t, error=str(exc))
        return None

    async def _spawn_watcher(self, cfg: BackgroundAgentConfig) -> None:
        watcher = self._build_watcher(cfg)
        if watcher is None:
            return
        old = self._tasks.pop(cfg.id, None)
        if old and not old.done():
            old.cancel()
        task = asyncio.create_task(
            self._watcher_loop(cfg, watcher),
            name=f"watcher_{cfg.id}",
        )
        self._tasks[cfg.id] = task

    async def _watcher_loop(self, cfg: BackgroundAgentConfig, watcher: WatcherBase) -> None:
        logger.info("watcher_loop_started", id=cfg.id, type=cfg.watcher_type, interval=cfg.interval_seconds)
        while True:
            try:
                signal = await watcher.check()
                if signal is not None:
                    signal.agent_id = cfg.id
                    signal.user_id = cfg.user_id
                    signal.channel_id = cfg.channel_id
                    signal.ai_context = cfg.description
                    try:
                        self._wake_queue.put_nowait(signal)
                    except asyncio.QueueFull:
                        logger.warning("wake_queue_full_dropping", agent_id=cfg.id)
                    await update_agent_trigger_count(cfg.id)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("watcher_loop_error", agent_id=cfg.id, error=str(exc))
            await asyncio.sleep(cfg.interval_seconds)

    # ------------------------------------------------------------------
    # Internal — wake signal processing
    # ------------------------------------------------------------------

    async def _wake_processor_loop(self) -> None:
        logger.info("wake_processor_started")
        while True:
            try:
                signal = await self._wake_queue.get()
                asyncio.create_task(self._handle_signal(signal))
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("wake_processor_error", error=str(exc))

    async def _handle_signal(self, signal: WakeSignal) -> None:
        severity_tag = {"info": "INFO", "warning": "WARN", "critical": "CRIT"}.get(
            signal.severity, "WARN"
        )

        # Non-AI signals — send raw
        if not signal.needs_ai_analysis:
            raw = "\n".join(f"{k}: {v}" for k, v in signal.raw_data.items())
            text = f"<b>[{severity_tag}] {signal.agent_id}</b>\n\n{raw}"
            await self._send(signal.channel_id, text)
            await save_wake_event(signal, raw)
            return

        # Cron and script signals — run a FULL AGENT TURN with tools
        # so the AI can actually act (run commands, write files, etc.)
        if signal.event_type in ("scheduled_task", "script_alert"):
            await self._handle_with_agent(signal, severity_tag)
            return

        # Other signals — plain LLM analysis call (no tools needed for threshold breach)
        await self._handle_with_llm(signal, severity_tag)

    async def _handle_with_agent(self, signal: WakeSignal, severity_tag: str) -> None:
        """Run a ConcreteAgent turn with full tool access for cron/script signals."""
        from src.agents.agent_factory import execute_tool
        from src.agents.agent_models import AgentSpec
        from src.agents.agent_factory import ConcreteAgent

        cfg = Config.get()
        context_str = json.dumps(signal.raw_data, indent=2)

        task_msg = (
            f"Background task triggered.\n"
            f"Purpose: {signal.ai_context}\n"
            f"Event: {signal.event_type} [{signal.severity}]\n\n"
            f"Data:\n{context_str}\n\n"
            f"Execute the task. Use tools as needed. "
            f"Be concise in your response — this goes directly to the user."
        )

        spec = AgentSpec(
            id=f"bg_{signal.agent_id}",
            name="BackgroundAgent",
            role="Autonomous background agent",
            system_prompt=(
                cfg.system_prompt
                + "\n\nYou are running as a background agent. "
                "You have full tool access. Complete the task autonomously. "
                "Keep your final response to the user under 300 words."
            ),
            tools=["run_shell_command", "run_python", "web_search", "curl",
                   "wikipedia_search", "save_memory", "send_file", "list_workspace"],
        )

        try:
            agent = ConcreteAgent(spec, depth=0)
            result = await asyncio.wait_for(
                agent.run({"user_id": signal.user_id, "message": task_msg,
                           "full_context": task_msg}),
                timeout=120,
            )
            output = result.get("output", "").strip() or "Task completed."

            # Handle any file sends the agent queued
            send_files = result.get("send_files", [])
            if send_files:
                ws = str(Path(cfg.workspace_path).expanduser())
                for finfo in send_files:
                    try:
                        await self._sink.send_file(
                            signal.channel_id,
                            finfo.get("path", ""),
                            finfo.get("caption", ""),
                            workspace=ws,
                        )
                    except Exception as exc:
                        logger.error("bg_agent_file_send_failed", error=str(exc))

        except asyncio.TimeoutError:
            output = "Background task timed out after 2 minutes."
        except Exception as exc:
            logger.error("bg_agent_run_failed", agent_id=signal.agent_id, error=str(exc))
            output = "Background task encountered an internal error."

        text = f"<b>[{severity_tag}] {signal.agent_id}</b>\n\n{output}"
        await self._send(signal.channel_id, text)
        await save_wake_event(signal, output)

    async def _handle_with_llm(self, signal: WakeSignal, severity_tag: str) -> None:
        """Single LLM call for simple anomaly analysis (no tools needed)."""
        cfg = Config.get()
        pool = get_pool()
        context_str = json.dumps(signal.raw_data, indent=2)

        payload = {
            "model": cfg.default_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        cfg.system_prompt
                        + "\n\nYou are a background sentinel. An anomaly was detected. "
                        "Write a concise notification (max 120 words): what happened, "
                        "what it means, and one remediation suggestion."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Agent: {signal.agent_id}\n"
                        f"Purpose: {signal.ai_context}\n"
                        f"Event: {signal.event_type} [{signal.severity}]\n\n"
                        f"Data:\n{context_str}"
                    ),
                },
            ],
        }

        analysis = ""
        for provider in (cfg.default_provider_priority or ["gemini", "groq", "openrouter"]):
            try:
                resp = await pool.request_with_key(signal.user_id, provider, payload)
                analysis = resp.get("output", "").strip()
                if analysis:
                    break
            except Exception as exc:
                logger.warning("wake_llm_failed", provider=provider, error=str(exc))

        if not analysis:
            analysis = context_str[:400]

        text = f"<b>[{severity_tag}] {signal.agent_id}</b>\n\n{analysis}"
        await self._send(signal.channel_id, text)
        await save_wake_event(signal, analysis)

    async def _send(self, channel_id: str, text: str) -> None:
        try:
            await self._sink.send_text(channel_id, text)
        except Exception as exc:
            logger.error("wake_send_failed", channel_id=channel_id, error=str(exc))
