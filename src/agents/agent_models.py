from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class AgentSpec(BaseModel):
    id: str
    name: str
    role: str
    system_prompt: str = ""
    tools: List[str] = []
    output_schema: Dict[str, Any] = {}
    depends_on: List[str] = []
    model_preference: Literal["fast", "smart"] = "smart"


class TaskPlan(BaseModel):
    reasoning: str
    agents: List[AgentSpec] = []
    final_synthesis_prompt: Optional[str] = None


class WakeSignal(BaseModel):
    """A trigger event produced by a background watcher script.
    Carries no LLM output — it is the raw anomaly detected by pure Python code.
    The WakeProcessor consumes it and decides whether to call an LLM.
    """
    agent_id: str = ""
    user_id: int = 0
    channel_id: str = ""
    event_type: str  # threshold_breach | process_down | process_recovered |
                     # url_unreachable | url_recovered | port_closed | port_recovered |
                     # pattern_match | scheduled_task
    severity: Literal["info", "warning", "critical"] = "warning"
    raw_data: Dict[str, Any] = Field(default_factory=dict)
    ai_context: str = ""   # user's original description — injected by manager
    needs_ai_analysis: bool = True
    timestamp: str = Field(default_factory=_utcnow)


class BackgroundAgentConfig(BaseModel):
    """Persisted configuration for one background monitoring agent."""
    id: str
    user_id: int
    channel_id: str
    watcher_type: str   # system | process | url | port | log | cron
    name: str
    description: str    # user's natural-language intent, becomes AI context
    config: Dict[str, Any] = Field(default_factory=dict)
    interval_seconds: int = 60
    enabled: bool = True
    created_at: str = Field(default_factory=_utcnow)
