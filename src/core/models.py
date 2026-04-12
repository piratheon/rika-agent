"""Shared data models for agent interfaces.

Extracted from app.py - these are used by both Telegram and Web.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class AgentStatus(Enum):
    """Agent execution status."""
    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING_TOOL = "executing_tool"
    ERROR = "error"
    CANCELLED = "cancelled"


class MessageType(Enum):
    """Message types for interface communication."""
    USER_MESSAGE = "user_message"
    AI_MESSAGE = "ai_message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    STATUS_UPDATE = "status_update"
    ERROR = "error"


@dataclass
class ToolCall:
    """Represents a tool execution request."""
    name: str
    arguments: Dict[str, Any]
    call_id: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "arguments": self.arguments,
            "call_id": self.call_id,
        }


@dataclass
class ToolResult:
    """Represents a tool execution result."""
    tool_name: str
    result: str
    success: bool = True
    call_id: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "result": self.result,
            "success": self.success,
            "call_id": self.call_id,
        }


@dataclass
class AgentMessage:
    """Unified message format for all interfaces."""
    type: MessageType
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


@dataclass
class AgentState:
    """Current agent execution state."""
    status: AgentStatus = AgentStatus.IDLE
    current_tool: Optional[ToolCall] = None
    turn_count: int = 0
    user_id: int = 0
    chat_id: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "current_tool": self.current_tool.to_dict() if self.current_tool else None,
            "turn_count": self.turn_count,
            "user_id": self.user_id,
            "chat_id": self.chat_id,
        }


# ---------------------------------------------------------------------------
# Event stream — typed session events consumed by all interfaces
# ---------------------------------------------------------------------------

import time as _time


class EventType(Enum):
    """All events the orchestration layer can emit."""
    THINKING      = "thinking"       # agent is reasoning between tool calls
    INTENT        = "intent"         # agent declared a step via declare_step tool
    TOOL_CALL     = "tool_call"      # tool about to be executed
    TOOL_RESULT   = "tool_result"    # tool returned a result
    STEP_DONE     = "step_done"      # declare_step(status="done")
    STEP_FAILED   = "step_failed"    # declare_step(status="failed")
    MESSAGE       = "message"        # final or streaming text chunk
    ERROR         = "error"          # unrecoverable error in orchestration
    CANCELLED     = "cancelled"      # user cancelled the task
    BUDGET        = "budget"         # token budget update


@dataclass
class SessionEvent:
    """A single typed event emitted by the orchestration layer.

    Interfaces subscribe to these events and render them differently:
    - Telegram: collapses to a work log message, edits in place
    - TUI: expands to a live panel with spinners
    - Web (v3.0): streams over WebSocket

    Attributes:
        type:    The event category.
        payload: Event-specific data dict. Keys vary by type:
                   INTENT/STEP_*: {"title": str, "status": str}
                   TOOL_CALL:     {"tool": str, "args": dict}
                   TOOL_RESULT:   {"tool": str, "result": str, "success": bool}
                   MESSAGE:       {"text": str, "final": bool}
                   ERROR:         {"reason": str}
                   BUDGET:        {"input": int, "output": int, "total": int}
        turn:    Orchestration turn number this event belongs to.
        ts:      Monotonic timestamp at creation.
    """
    type: EventType
    payload: Dict[str, Any] = field(default_factory=dict)
    turn: int = 0
    ts: float = field(default_factory=_time.monotonic)
