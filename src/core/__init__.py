"""Core agentic logic — interface-agnostic.

This module contains the shared logic used by ALL interfaces:
- Telegram bot
- Web UI (WebSocket)
- CLI (future)
- API (future)

No interface-specific imports here. Only pure agent logic.
"""
from src.core.complexity import classify_complexity
from src.core.context import ContextBuilder
from src.core.models import AgentState, AgentStatus, MessageType, ToolCall, ToolResult
from src.core.orchestrator import Orchestrator
from src.core.tools import ToolExecutor

__all__ = [
    "Orchestrator",
    "ContextBuilder",
    "ToolExecutor",
    "AgentState",
    "AgentStatus",
    "MessageType",
    "ToolCall",
    "ToolResult",
    "classify_complexity",
]
from src.core.event_bus import emit, get_bus, subscribe, teardown as teardown_session
from src.core.models import EventType, SessionEvent
