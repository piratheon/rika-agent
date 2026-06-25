import asyncio
import time
from typing import Dict


class LiveBubble:
    """Manage a single Telegram message bubble that is updated incrementally.

    This class stores agent sections and serialises updates through an internal
    asyncio.Queue. Call `start(flush_cb)` to begin the flush loop where `flush_cb`
    is an async callable receiving the rendered text and should perform the
    `edit_message_text` call.
    """

    def __init__(self, throttle_ms: int = 800):
        self.sections: Dict[str, str] = {}
        self.queue: asyncio.Queue = asyncio.Queue()
        self.throttle = throttle_ms / 1000.0
        self._task = None
        self._last_flush = 0.0
        self._pulse_idx = 0
        self._pulse_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self._icons = {
            "pending": "◌",
            "running": "⚙️",
            "done": "✅",
            "error": "❌",
            "thinking": "🧠",
        }

    def update(self, agent_id: str, text: str):
        # text may include status prefix like 'running...' or 'done'
        self.sections[agent_id] = text
        # non-blocking enqueue
        try:
            self.queue.put_nowait(True)
        except asyncio.QueueFull:
            pass

    def render(self) -> str:
        parts = ["<b>Processing request...</b>", "", "<b>Active Agents & Fragments:</b>"]
        self._pulse_idx = (self._pulse_idx + 1) % len(self._pulse_frames)
        pulse = self._pulse_frames[self._pulse_idx]
        
        for aid, txt in self.sections.items():
            # pick icon based on keywords
            icon = self._icons.get("pending")
            low_txt = txt.lower()
            if "running" in low_txt or "using" in low_txt or "fetching" in low_txt:
                icon = f"{pulse} {self._icons.get('running')}"
            elif "thinking" in low_txt or "orchestrating" in low_txt:
                icon = f"{pulse} {self._icons.get('thinking')}"
            elif "done" in low_txt or "stored" in low_txt or "validated" in low_txt or "success" in low_txt:
                icon = self._icons.get("done")
            elif "error" in low_txt or "failed" in low_txt:
                icon = self._icons.get("error")
                
            parts.append(f"{icon} <code>{aid}</code> — {txt}")
        return "\n".join(parts)

    async def start(self, flush_cb):
        if self._task:
            return
        self._task = asyncio.create_task(self._loop(flush_cb))

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _loop(self, flush_cb):
        while True:
            # Wait for an update or a pulse timeout (0.2s for smooth animation)
            try:
                await asyncio.wait_for(self.queue.get(), timeout=0.2)
            except asyncio.TimeoutError:
                pass
                
            now = time.time()
            elapsed = now - self._last_flush
            if elapsed < self.throttle:
                # Still check for throttle but don't sleep the whole duration
                # just to allow the pulse to move if throttle is long.
                continue
                
            text = self.render()
            try:
                await flush_cb(text)
                self._last_flush = time.time()
            except Exception:
                # If editing fails, maybe the message was deleted or same content
                pass
