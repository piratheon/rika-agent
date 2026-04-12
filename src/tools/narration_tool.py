"""Narration tool — declare_step for structured work log narration.

The agent calls declare_step() before and after each logical work block.
This has zero external side effects — ToolExecutor intercepts it before
registry lookup and emits a SessionEvent(EventType.INTENT, ...) into the
event bus. The tool itself returns an empty string.

Example LLM usage:
    declare_step(title="Setting up project structure")
    ... file write tools ...
    declare_step(title="Setting up project structure", status="done")
"""
from __future__ import annotations

DECLARE_STEP_SCHEMA = {
    "name": "declare_step",
    "description": (
        "Announce the start or end of a logical work block before executing tools for it. "
        "Call with status='running' when starting a new goal. "
        "Call with status='done' on success or status='failed' on error. "
        "This is mandatory for any task that involves 2 or more tool calls."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Short descriptive label, e.g. 'Creating backend server', 'Running tests', 'Fixing bug in main.py'.",
            },
            "status": {
                "type": "string",
                "enum": ["running", "done", "failed"],
                "description": "Current status of this work block.",
                "default": "running",
            },
        },
        "required": ["title"],
    },
}


async def declare_step(title: str, status: str = "running") -> str:
    """No-op execution — the event is emitted by ToolExecutor before this runs."""
    return ""
