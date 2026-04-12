"""Tool executor — unified tool execution with timeout.

Extracted from agent_factory.py execute_tool() function.
Same logic, just wrapped in a class for reusability.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from src.config import Config
from src.tools.registry import get_registry
from src.utils.logger import logger


class ToolExecutor:
    """Executes tools with timeout and error handling.
    
    Same logic as app.py execute_tool() but as a reusable class.
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config.get()
        self.registry = get_registry()
        self.tool_timeout = getattr(self.config, "tool_timeout_seconds", 10)
    
    async def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        user_id: int = 0,
        system_prompt: str = "",
        bubble=None,
    ) -> str:
        """Execute a tool with timeout.
        
        Same logic as execute_tool() from agent_factory.py.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments as dict
            user_id: User ID for permission checks
            system_prompt: System prompt for context
            bubble: LiveBubble for status updates (optional)
            
        Returns:
            Tool execution result as string
        """
        # Memory tools
        if tool_name == "save_memory":
            from src.db.chat_store import save_rika_memory
            k = arguments.get("key", "")
            v = arguments.get("value", "")
            if not k:
                return "Error: 'key' is required."
            pinned = arguments.get("pinned", False)
            await save_rika_memory(user_id, k, v, "memory", pinned=bool(pinned))
            return f"Memory saved: {k}" + (" (pinned)" if pinned else "")

        if tool_name == "save_skill":
            from src.db.chat_store import save_skill
            k = arguments.get("name", "")
            v = arguments.get("code", "")
            if not k:
                return "Error: 'name' is required."
            await save_skill(user_id, k, v)
            return f"Skill saved: {k}"

        if tool_name == "get_memories":
            from src.db.chat_store import (
                get_pinned_memories, get_rika_memories, list_skill_names
            )
            pinned = await get_pinned_memories(user_id)
            all_mems = await get_rika_memories(user_id, "memory")
            # Merge: pinned first, then recents not already in pinned, cap at 10
            merged = dict(pinned)
            for k, v in all_mems.items():
                if k not in merged and len(merged) < 10:
                    merged[k] = v
            skills = list(await list_skill_names(user_id))
            pinned_keys = set(pinned.keys())
            lines = []
            for k, v in merged.items():
                tag = " [pinned]" if k in pinned_keys else ""
                lines.append(f"  {k}{tag}: {v}")
            mem_block = "\n".join(lines) if lines else "  (none)"
            skill_block = ", ".join(skills) if skills else "none"
            return (
                f"MEMORIES ({len(merged)} shown, {len(all_mems)} total):\n{mem_block}\n\n"
                f"SKILLS ({len(skills)}):\n  {skill_block}"
            )

        # Skill lazy-load — the key innovation
        if tool_name == "use_skill":
            from src.db.chat_store import get_skill, list_skill_names
            skill_name = arguments.get("skill_name", "").strip()
            if not skill_name:
                return "Error: 'skill_name' is required."
            content = await get_skill(user_id, skill_name)
            if content is None:
                available = await list_skill_names(user_id)
                return (
                    f"Skill '{skill_name}' not found.\n"
                    f"Available skills: {', '.join(available) if available else 'none'}"
                )
            return f"SKILL '{skill_name}':\n{content}"

        # Delegation
        if tool_name == "delegate_task":
            from src.agents.agent_factory import ConcreteAgent
            from src.agents.agent_models import AgentSpec
            query = arguments.get("query", "")
            sub_spec = AgentSpec(
                id="sub_research",
                name="ResearchAgent",
                role="Specialized researcher",
                system_prompt=system_prompt,
                tools=["web_search", "curl", "wikipedia_search", "run_shell_command"],
            )
            sub = ConcreteAgent(sub_spec, bubble=bubble, depth=0)
            res = await sub.run({"user_id": user_id, "message": query, "full_context": query})
            return res.get("output", "No result.")

        # File send signal
        if tool_name == "send_file":
            path = arguments.get("path", "").strip()
            caption = arguments.get("caption", "")
            return f"__SEND_FILE__:{path}:{caption}"

        # Narration tool — intercepted before registry; emits a SessionEvent
        if tool_name == "declare_step":
            from src.core.event_bus import emit as _emit
            from src.core.models import EventType, SessionEvent
            _session_id = getattr(self, "_current_session_id", 0)
            status = arguments.get("status", "running")
            etype = {
                "running": EventType.INTENT,
                "done":    EventType.STEP_DONE,
                "failed":  EventType.STEP_FAILED,
            }.get(status, EventType.INTENT)
            import asyncio as _aio
            try:
                _aio.get_event_loop().create_task(
                    _emit(_session_id, SessionEvent(
                        etype,
                        payload={"title": arguments.get("title", ""), "status": status},
                    ))
                )
            except RuntimeError:
                pass  # no running loop (e.g. in tests)
            return ""

        # Registry tools
        tool_fn = self.registry.get(tool_name)
        if tool_fn is None:
            return f"Error: tool '{tool_name}' not found."

        try:
            # Execute tool with timeout
            if tool_name == "run_shell_command":
                cmd = arguments.get("command") or arguments.get("query", "")
                if not cmd:
                    return "Error: no command provided."
                raw = await asyncio.wait_for(
                    tool_fn(cmd, user_id=user_id) if asyncio.iscoroutinefunction(tool_fn) else tool_fn(cmd, user_id=user_id),
                    timeout=self.tool_timeout
                )
            elif tool_name == "run_python":
                code = arguments.get("code") or arguments.get("query", "")
                if not code:
                    return "Error: no code provided."
                timeout = min(arguments.get("timeout_seconds", 30), self.tool_timeout * 2)
                raw = await asyncio.wait_for(
                    tool_fn(code, timeout_seconds=int(timeout)) if asyncio.iscoroutinefunction(tool_fn) else tool_fn(code),
                    timeout=timeout + 5
                )
            elif tool_name == "watch_task_logs":
                raw = await asyncio.wait_for(
                    tool_fn(
                        arguments.get("file_path", ""),
                        arguments.get("timeout", "30"),
                    ),
                    timeout=self.tool_timeout
                )
            elif tool_name == "write_file":
                raw = await asyncio.wait_for(
                    tool_fn(
                        arguments.get("path", ""),
                        arguments.get("content", ""),
                        arguments.get("mode", "w"),
                    ),
                    timeout=self.tool_timeout
                )
            elif tool_name == "read_file":
                raw = await asyncio.wait_for(
                    tool_fn(arguments.get("path", ""), arguments.get("max_lines", 200)),
                    timeout=self.tool_timeout
                )
            elif len(arguments) == 0:
                raw = await asyncio.wait_for(
                    tool_fn("") if asyncio.iscoroutinefunction(tool_fn) else tool_fn(""),
                    timeout=self.tool_timeout
                )
            elif len(arguments) == 1:
                val = next(iter(arguments.values()), "")
                raw = await asyncio.wait_for(
                    tool_fn(str(val)) if asyncio.iscoroutinefunction(tool_fn) else tool_fn(str(val)),
                    timeout=self.tool_timeout
                )
            else:
                kwargs = {k: str(v) for k, v in arguments.items()}
                raw = await asyncio.wait_for(
                    tool_fn(**kwargs) if asyncio.iscoroutinefunction(tool_fn) else tool_fn(**kwargs),
                    timeout=self.tool_timeout
                )

            if isinstance(raw, dict):
                if raw.get("blocked"):
                    return raw.get("message", "Command blocked by security policy.")
                parts = []
                if raw.get("stdout"):
                    parts.append(f"stdout:\n{raw['stdout']}")
                if raw.get("stderr"):
                    parts.append(f"stderr:\n{raw['stderr']}")
                if "exit_code" in raw:
                    parts.append(f"exit_code: {raw['exit_code']}")
                if raw.get("cwd"):
                    parts.append(f"cwd: {raw['cwd']}")
                if raw.get("error"):
                    parts.append(f"error: {raw['error']}")
                return "\n".join(parts) or "Done."
            return str(raw)

        except asyncio.TimeoutError:
            logger.error("tool_execution_timeout", tool=tool_name, timeout=self.tool_timeout)
            return f"Error: Tool '{tool_name}' timed out after {self.tool_timeout} seconds. For long-running tasks, use /watch to schedule as a background task."
        except Exception as exc:
            logger.error("tool_execution_failed", tool=tool_name, error=str(exc))
            return f"Error executing {tool_name}: {exc}"
