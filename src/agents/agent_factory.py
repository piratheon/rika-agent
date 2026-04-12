"""ConcreteAgent — function-calling ReAct loop with token-efficient context.

Token efficiency changes:
- Memories: pinned (always) + semantic retrieval (top-4 relevant) only.
  No more full JSON dump of all memories on every call.
- Skills: lazy-loaded via use_skill tool. Never injected into system prompt.
  Saves 100-400 tokens per call when skills exist.
- Runtime system message is built once per run, not per turn.
- Skill names listed in use_skill tool description so agent knows what exists.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Dict, List, Optional

from src.agents.agent_models import AgentSpec
from src.agents.base_agent import BaseAgent
from src.config import Config
from src.providers.base_provider import StructuredResponse, ToolCall
from src.providers.provider_pool import get_pool
from src.utils.logger import logger

MAX_TOOL_TURNS = 8
MAX_AGENT_DEPTH = 2


# ---------------------------------------------------------------------------
# Context builder — token-efficient
# ---------------------------------------------------------------------------

async def build_system_context(
    user_id: int,
    base_system_prompt: str,
    current_message: str,
    dep_outputs: Optional[Dict[str, str]] = None,
) -> str:
    """Build a token-efficient system message.

    Injects:
    - base_system_prompt (soul + tools + mandates)
    - pinned memories (max 5, always relevant)
    - semantically relevant memories (top 4 for this specific message)
    - skill names only (not skill content — lazy-loaded via use_skill tool)
    - dependency outputs from teammate agents (if any)

    Does NOT inject:
    - all memories as a JSON dump
    - all skills as a JSON dump
    - action logs (those belong in the watcher context, not here)
    """
    from src.db.chat_store import (
        get_pinned_memories,
        get_relevant_memories,
        list_skill_names,
    )

    parts = [base_system_prompt]

    # Pinned memories — always injected, zero retrieval cost
    pinned = await get_pinned_memories(user_id)
    if pinned:
        pinned_lines = "\n".join(f"  {k}: {v}" for k, v in pinned.items())
        parts.append(f"[PINNED CONTEXT]\n{pinned_lines}")

    # Semantically relevant memories — retrieved for this specific message
    if current_message:
        relevant = await get_relevant_memories(user_id, current_message, limit=4)
        # Remove any that were already in pinned
        relevant = {k: v for k, v in relevant.items() if k not in pinned}
        if relevant:
            rel_lines = "\n".join(f"  {k}: {v}" for k, v in relevant.items())
            parts.append(f"[RELEVANT MEMORY]\n{rel_lines}")

    # Skill names only — agent calls use_skill to load content
    skill_names = await list_skill_names(user_id)
    if skill_names:
        parts.append(
            f"[AVAILABLE SKILLS] (call use_skill to load any of these):\n"
            + ", ".join(skill_names)
        )

    # Teammate context (multi-agent runs)
    if dep_outputs:
        non_empty = {k: v for k, v in dep_outputs.items() if v and v != "N/A"}
        if non_empty:
            parts.append("[TEAMMATE CONTEXT]\n" + json.dumps(non_empty, indent=2))

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Tool executor — delegates to src.core.tools.ToolExecutor
# ---------------------------------------------------------------------------

async def execute_tool(
    tool_name: str,
    arguments: Dict[str, Any],
    user_id: int,
    depth: int = 0,
    system_prompt: str = "",
    bubble=None,
) -> str:
    """Thin shim — delegates to ToolExecutor in src.core.tools.

    Depth tracking for delegate_task recursion prevention is handled here
    by temporarily patching the max depth check before delegating.
    """
    from src.core.tools import ToolExecutor
    executor = ToolExecutor()
    executor._current_depth = depth          # used by delegate_task guard
    executor._current_system_prompt = system_prompt
    executor._bubble = bubble
    return await executor.execute(
        tool_name=tool_name,
        arguments=arguments,
        user_id=user_id,
        system_prompt=system_prompt,
    )


# ---------------------------------------------------------------------------
# Text-protocol fallback
# ---------------------------------------------------------------------------

def _parse_text_tool_call(text: str):
    match = re.search(
        r"TOOL:\s*([\w_]+)\s*\|?\s*QUERY:\s*(.*)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if match:
        return match.group(1).strip(), {"query": match.group(2).strip()}
    return None, None


# ---------------------------------------------------------------------------
# ConcreteAgent
# ---------------------------------------------------------------------------

class ConcreteAgent(BaseAgent):
    def __init__(self, spec: AgentSpec, bubble=None, depth: int = 0) -> None:
        self.spec = spec
        self.bubble = bubble
        self.depth = depth

    async def _get_tool_schemas(self, user_id: int):
        from src.db.chat_store import list_skill_names
        from src.tools.schemas import get_schemas_for_tools, SCHEMA_MAP

        if not self.spec.tools:
            return []
        # Core tools from spec
        base = list(self.spec.tools)
        # Always add memory tools + use_skill
        extras = ["save_memory", "get_memories", "save_skill", "use_skill", "delegate_task"]
        all_tools = base + [e for e in extras if e not in base]

        schemas = get_schemas_for_tools(all_tools)

        # Inject skill names into use_skill description dynamically
        skill_names = await list_skill_names(user_id)
        for s in schemas:
            if s.name == "use_skill" and skill_names:
                s.description = s.description + f" Available: {', '.join(skill_names[:20])}"
        return schemas

    async def _request_structured(
        self, user_id: int, messages: list, schemas: list
    ) -> StructuredResponse:
        cfg = Config.get()
        pool = get_pool()
        payload = {"model": cfg.default_model, "messages": messages}
        for provider in (cfg.default_provider_priority or ["gemini", "groq", "openrouter"]):
            try:
                resp = await asyncio.wait_for(
                    pool.request_with_key_structured(user_id, provider, payload, schemas),
                    timeout=30.0,
                )
                return resp
            except asyncio.TimeoutError:
                logger.warning("agent_request_timeout", provider=provider)
                continue
            except Exception as exc:
                logger.warning("agent_provider_failed", provider=provider, error=str(exc))
        raise RuntimeError("All providers failed.")

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        cfg = Config.get()
        user_id = int(context.get("user_id", 0))
        message = context.get("message", "")
        full_context = context.get("full_context", message)
        dep_outputs = {
            dep: context.get("results", {}).get(dep, {}).get("output", "N/A")
            for dep in self.spec.depends_on
        }

        # Build token-efficient system message ONCE per run
        base_prompt = self.spec.system_prompt or cfg.system_prompt
        sys_msg = await build_system_context(
            user_id=user_id,
            base_system_prompt=base_prompt,
            current_message=message,
            dep_outputs=dep_outputs if dep_outputs else None,
        )

        schemas = await self._get_tool_schemas(user_id) if self.spec.tools else []
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": f"Context: {full_context}\n\nTask: {message}"},
        ]
        tool_used: Optional[str] = None
        send_files: List[Dict] = []

        for _turn in range(MAX_TOOL_TURNS):
            try:
                response = await self._request_structured(user_id, messages, schemas)
            except Exception as exc:
                return {"id": self.spec.id, "output": f"Error: {exc}"}

            # Function calling path
            if response.has_tool_calls:
                tool_results = []
                for tc in response.tool_calls:
                    if self.bubble:
                        self.bubble.update(self.spec.id, f"calling {tc.name}...")
                    result_str = await execute_tool(
                        tc.name, tc.arguments, user_id,
                        depth=self.depth, system_prompt=sys_msg, bubble=self.bubble,
                    )
                    tool_used = tc.name
                    if result_str.startswith("__SEND_FILE__:"):
                        parts = result_str[len("__SEND_FILE__:"):].split(":", 1)
                        send_files.append({
                            "path": parts[0],
                            "caption": parts[1] if len(parts) > 1 else "",
                        })
                        result_str = f"File queued: {parts[0]}"
                    tool_results.append({
                        "tool_call_id": tc.call_id,
                        "name": tc.name,
                        "content": result_str,
                    })

                messages.append({
                    "role": "assistant",
                    "content": response.content or "",
                    "tool_calls": [
                        {
                            "id": tc.call_id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in response.tool_calls
                    ],
                })
                for tr in tool_results:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tr["tool_call_id"],
                        "content": tr["content"],
                    })
                continue

            # Text-protocol fallback
            if schemas:
                t_name, t_args = _parse_text_tool_call(response.content)
                if t_name:
                    if self.bubble:
                        self.bubble.update(self.spec.id, f"using {t_name}...")
                    result_str = await execute_tool(
                        t_name, t_args or {}, user_id,
                        depth=self.depth, system_prompt=sys_msg,
                    )
                    tool_used = t_name
                    if result_str.startswith("__SEND_FILE__:"):
                        parts = result_str[len("__SEND_FILE__:"):].split(":", 1)
                        send_files.append({
                            "path": parts[0],
                            "caption": parts[1] if len(parts) > 1 else "",
                        })
                        result_str = f"File queued: {parts[0]}"
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({
                        "role": "user",
                        "content": f"Result:\n{result_str}\n\nContinue.",
                    })
                    continue

            # Final response
            return {
                "id": self.spec.id,
                "output": response.content.strip(),
                "tool_used": tool_used,
                "send_files": send_files,
            }

        # Turn limit — synthesize
        try:
            final = await self._request_structured(user_id, messages, [])
            return {
                "id": self.spec.id,
                "output": final.content,
                "tool_used": tool_used,
                "send_files": send_files,
            }
        except Exception as exc:
            return {"id": self.spec.id, "output": f"Error: {exc}", "send_files": send_files}


class AgentFactory:
    @staticmethod
    def create(spec: AgentSpec, bubble=None, depth: int = 0) -> BaseAgent:
        return ConcreteAgent(spec, bubble=bubble, depth=depth)
