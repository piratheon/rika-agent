"""Work log renderer — converts SessionEvents to Telegram-formatted work log.

Produces the structured work log format:
    [+] Setting up project structure
      * thinking finished
      * written main.py
      * execution finished (exit 0)
    [!] Tests failed
      * main.py execution failed (exit code: 2)
      * edited main.py
    [...] Archiving project
      * executing tar ...

Usage:
    renderer = WorkLogRenderer()
    for event in events:
        renderer.consume(event)
    text = renderer.render()
"""
from __future__ import annotations

import html
from dataclasses import dataclass, field
from typing import Optional

from src.core.models import EventType, SessionEvent


@dataclass
class _Step:
    title: str
    status: str = "running"       # running | done | failed
    events: list[str] = field(default_factory=list)

    def render(self) -> str:
        if self.status == "done":
            header = f"[+] {html.escape(self.title)}"
        elif self.status == "failed":
            header = f"[!] {html.escape(self.title)}"
        else:
            header = f"[...] {html.escape(self.title)}"
        lines = [header]
        for e in self.events:
            lines.append(f"  * {html.escape(e)}")
        return "\n".join(lines)


class WorkLogRenderer:
    """Stateful renderer: consume SessionEvents, render the work log."""

    def __init__(self) -> None:
        self._steps: list[_Step] = []
        self._current: Optional[_Step] = None
        self._last_thinking: bool = False
        self._final_message: str = ""

    def consume(self, event: SessionEvent) -> None:
        t = event.type
        p = event.payload

        if t == EventType.THINKING:
            if self._current and self._last_thinking:
                # Collapse consecutive THINKING into one line
                return
            if self._current:
                self._current.events.append("thinking...")
            self._last_thinking = True
            return

        self._last_thinking = False

        if t == EventType.INTENT:
            title = p.get("title", "")
            status = p.get("status", "running")
            if status == "running":
                # Close previous "thinking..." line as "thinking finished"
                if self._current and self._current.events and self._current.events[-1] == "thinking...":
                    self._current.events[-1] = "thinking finished"
                self._current = _Step(title=title)
                self._steps.append(self._current)
            elif self._current and self._current.title == title:
                if self._current.events and self._current.events[-1] == "thinking...":
                    self._current.events[-1] = "thinking finished"
                self._current.status = status

        elif t == EventType.STEP_DONE:
            if self._current:
                if self._current.events and self._current.events[-1] == "thinking...":
                    self._current.events[-1] = "thinking finished"
                self._current.status = "done"

        elif t == EventType.STEP_FAILED:
            if self._current:
                self._current.status = "failed"

        elif t == EventType.TOOL_CALL:
            tool = p.get("tool", "?")
            args = p.get("args", {})
            # Build a short human-readable description
            arg_preview = ""
            for key in ("path", "command", "query", "title", "name", "skill_name", "key"):
                if key in args:
                    val = str(args[key])
                    arg_preview = f" {val[:60]}" if len(val) <= 60 else f" {val[:57]}..."
                    break
            line = f"{tool}{arg_preview}"
            if self._current:
                # Replace trailing "thinking..." with "thinking finished" + this line
                if self._current.events and self._current.events[-1] == "thinking...":
                    self._current.events[-1] = "thinking finished"
                self._current.events.append(line)
            else:
                # Tool called outside a declared step — create an implicit step
                implicit = _Step(title=f"Running {tool}", status="running")
                implicit.events.append(line)
                self._steps.append(implicit)
                self._current = implicit

        elif t == EventType.TOOL_RESULT:
            tool = p.get("tool", "?")
            success = p.get("success", True)
            result_preview = str(p.get("result", ""))[:80]
            if not success:
                line = f"{tool} failed: {result_preview}"
            else:
                line = f"{tool} finished"
            if self._current:
                self._current.events.append(line)

        elif t == EventType.MESSAGE:
            if p.get("final"):
                self._final_message = p.get("text", "")
                # Mark any in-progress step as done
                if self._current and self._current.status == "running":
                    self._current.status = "done"

        elif t in (EventType.ERROR, EventType.CANCELLED):
            if self._current:
                self._current.status = "failed"
                reason = p.get("reason", str(t.value))
                self._current.events.append(reason)

    def render(self, include_final: bool = False) -> str:
        """Render current state as Telegram HTML."""
        parts = [s.render() for s in self._steps]
        if include_final and self._final_message:
            parts.append(f"\n{html.escape(self._final_message)}")
        return "\n".join(parts) if parts else "Working..."

    def has_steps(self) -> bool:
        return bool(self._steps)

    def final_message(self) -> str:
        return self._final_message
