"""Orchestrator — main agent orchestration loop.

Extracted from app.py run_orchestration_background() function.
Same logic, but interface-agnostic via callbacks.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Callable, Dict, List, Optional

from src.config import Config
from src.core.complexity import classify_complexity
from src.core.context import ContextBuilder
from src.core.models import AgentState, AgentStatus, ToolCall, ToolResult
from src.core.tools import ToolExecutor
from src.providers.provider_pool import get_pool
from src.tools.schemas import get_all_schemas
from src.utils.logger import logger
from src.core.event_bus import emit as _emit_event
from src.core.models import EventType, SessionEvent


class Orchestrator:
    """Main orchestration loop for agent execution.
    
    Extracted from app.py run_orchestration_background().
    Same logic, but uses callbacks for interface-specific logic.
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        on_status_update: Optional[Callable] = None,
        on_tool_call: Optional[Callable] = None,
        on_tool_result: Optional[Callable] = None,
        on_message: Optional[Callable] = None,
        is_cancelled: Optional[Callable[[int], bool]] = None,
    ):
        """Initialize orchestrator.
        
        Args:
            config: Config instance
            on_status_update: Callback(status: AgentState)
            on_tool_call: Callback(tool_call: ToolCall)
            on_tool_result: Callback(result: ToolResult)
            on_message: Callback(message: str)
            is_cancelled: Callback(user_id: int) -> bool
        """
        self.config = config or Config.get()
        self.pool = get_pool()
        self.tool_executor = ToolExecutor(self.config)
        self.context_builder = ContextBuilder()
        
        # Callbacks for interface-specific logic
        self.on_status_update = on_status_update
        self.on_tool_call = on_tool_call
        self.on_tool_result = on_tool_result
        self.on_message = on_message
        self.is_cancelled = is_cancelled
        
        # Provider-specific models
        self.model_map = {
            "groq": getattr(self.config, "groq_model", "llama-3.3-70b-versatile"),
            "openrouter": getattr(self.config, "openrouter_model", "google/gemini-2.0-flash-001"),
            "gemini": getattr(self.config, "gemini_model", "gemini-2.0-flash"),
            "ollama": getattr(self.config, "ollama_model", "llama3.2"),
            "g4f": getattr(self.config, "g4f_model", "MiniMaxAI/MiniMax-M2.5"),
        }
    
    async def run(
        self,
        user_id: int,
        chat_id: int,
        message: str,
        context_str: str,
        system_prompt: str,
        history: Optional[List[Dict]] = None,
        summary: Optional[str] = None,
        max_turns: int = 10,
    ) -> str:
        """Run the orchestration loop.
        
        Same logic as app.py run_orchestration_background().
        
        Args:
            user_id: User ID
            chat_id: Chat/session ID
            message: User message
            context_str: Context string
            system_prompt: System prompt
            history: Chat history
            summary: Conversation summary
            max_turns: Maximum reasoning turns
            
        Returns:
            Final AI response
        """
        state = AgentState(
            status=AgentStatus.THINKING,
            user_id=user_id,
            chat_id=chat_id,
        )
        
        # Build message history
        thought_history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ]
        
        # Get tool schemas for function calling
        tool_schemas = get_all_schemas()
        openai_schemas = [s.to_openai() for s in tool_schemas]
        
        priorities = self.config.default_provider_priority or ["gemini", "groq", "openrouter"]
        agent_results: dict = {}
        narrative_chunks: list = []
        
        try:
            for turn in range(max_turns):
                # Check cancel flag
                if self.is_cancelled and self.is_cancelled(user_id):
                    state.status = AgentStatus.CANCELLED
                    await self._notify_status(state)
                    return "Task cancelled by user."
                
                state.turn_count = turn + 1
                await self._notify_status(state)
                await _emit_event(chat_id, SessionEvent(
                    EventType.THINKING, turn=turn
                ))
                
                # Get response from provider
                resp = await self._get_provider_response(
                    user_id=user_id,
                    priorities=priorities,
                    messages=thought_history,
                    tool_schemas=openai_schemas,
                )
                
                if not resp:
                    break
                
                # Handle tool calls
                if resp.has_tool_calls:
                    for tool_call in resp.tool_calls:
                        # Check cancel between tool calls
                        if self.is_cancelled and self.is_cancelled(user_id):
                            state.status = AgentStatus.CANCELLED
                            await self._notify_status(state)
                            return "Task cancelled by user."
                        
                        # Execute tool
                        state.status = AgentStatus.EXECUTING_TOOL
                        state.current_tool = ToolCall(
                            name=tool_call.name,
                            arguments=tool_call.arguments,
                            call_id=tool_call.call_id,
                        )
                        await self._notify_status(state)
                        await self._notify_tool_call(state.current_tool)
                        await _emit_event(chat_id, SessionEvent(
                            EventType.TOOL_CALL,
                            payload={"tool": tool_call.name, "args": tool_call.arguments},
                            turn=turn,
                        ))
                        
                        result = await self.tool_executor.execute(
                            tool_name=tool_call.name,
                            arguments=tool_call.arguments,
                            user_id=user_id,
                            system_prompt=system_prompt,
                        )
                        
                        tool_result = ToolResult(
                            tool_name=tool_call.name,
                            result=result,
                            call_id=tool_call.call_id,
                        )
                        await self._notify_tool_result(tool_result)
                        await _emit_event(chat_id, SessionEvent(
                            EventType.TOOL_RESULT,
                            payload={
                                "tool": tool_call.name,
                                "result": result[:300],
                                "success": not result.startswith("Error"),
                            },
                            turn=turn,
                        ))
                        
                        # Add to history
                        thought_history.append({
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{
                                "id": tool_call.call_id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.name,
                                    "arguments": json.dumps(tool_call.arguments),
                                },
                            }],
                        })
                        thought_history.append({
                            "role": "tool",
                            "content": result,
                            "tool_call_id": tool_call.call_id,
                        })
                        
                        agent_results[f"turn_{turn}"] = {
                            "output": result,
                            "tool_used": tool_call.name,
                        }
                    continue  # Continue to next turn
                
                # No tool calls - final response
                output = resp.content
                state.status = AgentStatus.IDLE
                await self._notify_status(state)
                
                # Process final response
                final = re.sub(r"TOOL:\s*[\w_]+\s*\|?\s*QUERY:.*", "", output, flags=re.IGNORECASE | re.DOTALL).strip()
                if final:
                    narrative_chunks.append(final)
                
                unique = []
                for c in narrative_chunks:
                    if c not in unique and len(c) > 2:
                        unique.append(c)
                full_text = "\n\n".join(unique) or "Task complete."
                
                # Strip leaked tool markers from final display
                for marker in ("RESEARCH_FINDINGS", "SYSTEM_DATA", "TOOL_RESULT"):
                    if marker in full_text:
                        full_text = full_text.split(marker)[0].strip()
                
                findings_block = ""
                if agent_results:
                    findings_block = "\n\n<b>Process log:</b>\n"
                    for aid, res in agent_results.items():
                        tool = res.get("tool_used", "analysis")
                        preview = str(res.get("output", "done"))
                        preview = (preview[:120] + "...") if len(preview) > 120 else preview
                        findings_block += f"  {tool}: {preview}\n"
                
                full_response = full_text + findings_block
                
                # Notify interface
                await _emit_event(chat_id, SessionEvent(
                    EventType.MESSAGE,
                    payload={"text": full_response, "final": True},
                    turn=turn,
                ))
                if self.on_message:
                    await self.on_message(full_response, agent_results)
                
                return full_response
            
            return "The reasoning loop reached its turn limit. Please rephrase your request."
            
        except Exception as exc:
            state.status = AgentStatus.ERROR
            await self._notify_status(state)
            logger.exception("orchestration_failed", user_id=user_id, error=str(exc))
            return f"An error occurred: {str(exc)}"
    
    async def _get_provider_response(
        self,
        user_id: int,
        priorities: List[str],
        messages: List[Dict],
        tool_schemas: List[Dict],
    ) -> Optional[Any]:
        """Get response from provider with failover."""
        from src.providers.base_provider import StructuredResponse
        
        for provider in priorities:
            # Check cancel
            if self.is_cancelled and self.is_cancelled(user_id):
                return None
            
            try:
                payload = {
                    "model": self.model_map.get(provider, self.config.default_model),
                    "messages": messages,
                }
                
                resp = await asyncio.wait_for(
                    self.pool.request_with_key_structured(user_id, provider, payload, tool_schemas),
                    timeout=60.0,
                )
                
                if resp and (resp.has_tool_calls or resp.content):
                    return resp
                    
            except asyncio.TimeoutError:
                logger.warning("provider_timeout", provider=provider, timeout=60)
            except Exception as exc:
                logger.warning("provider_failed", provider=provider, error=str(exc))
        
        return None
    
    async def _notify_status(self, state: AgentState):
        """Notify interface of status update."""
        if self.on_status_update:
            try:
                await self.on_status_update(state)
            except Exception as exc:
                logger.warning("status_callback_failed", error=str(exc))
    
    async def _notify_tool_call(self, tool_call: ToolCall):
        """Notify interface of tool call."""
        if self.on_tool_call:
            try:
                await self.on_tool_call(tool_call)
            except Exception as exc:
                logger.warning("tool_call_callback_failed", error=str(exc))
    
    async def _notify_tool_result(self, result: ToolResult):
        """Notify interface of tool result."""
        if self.on_tool_result:
            try:
                await self.on_tool_result(result)
            except Exception as exc:
                logger.warning("tool_result_callback_failed", error=str(exc))
