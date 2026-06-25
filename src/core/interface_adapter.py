from __future__ import annotations

from abc import ABC, abstractmethod
from asyncio import Event
from typing import Optional


class InterfaceAdapter(ABC):
    """Contract between orchestration logic and a specific platform.

    Each platform (Telegram, Discord, API, library mode) implements this.
    The orchestrator and BackgroundAgentManager hold a reference to one
    via EventSink.
    """

    @abstractmethod
    async def send_text(
        self, channel_id: str, text: str, **kwargs
    ) -> str:
        """Send text to a channel. Returns an opaque message handle."""
        ...

    @abstractmethod
    async def edit_message(
        self, handle: str, text: str, **kwargs
    ) -> None:
        """Edit an existing message by handle. No-op if unsupported."""
        ...

    @abstractmethod
    async def send_file(
        self, channel_id: str, file_path: str, caption: str = "", **kwargs
    ) -> bool:
        """Send a file from the local filesystem. Returns True on success."""
        ...

    @abstractmethod
    async def show_countdown(
        self,
        handle: str,
        wait_seconds: int,
        attempt: int = 1,
        on_stop: Optional[Event] = None,
        on_retry: Optional[Event] = None,
    ) -> str:
        """Show a countdown with stop/retry affordances.

        Returns one of: "stop" | "retry" | "done"
        The on_stop/on_retry events are set by the platform's
        callback mechanism (e.g., button press).
        """
        ...

    @abstractmethod
    def supports_interactive(self) -> bool:
        """True if the platform supports buttons, callbacks, countdown UX.
        False for WhatsApp, API mode, library mode.
        """
        ...

    @abstractmethod
    def format_text(self, text: str, mode: str = "HTML") -> str:
        """Convert between HTML/Markdown/plain for the target platform."""
        ...

    @abstractmethod
    async def send_chunked(
        self, channel_id: str, text: str, max_len: int = 4000
    ) -> list[str]:
        """Send long text in chunks."""
        ...

    @abstractmethod
    async def send_typing(self, channel_id: str) -> None:
        """Show a typing indicator / progress pulse.
        No-op for platforms that don't support it (API, library mode).
        """
        ...

    @abstractmethod
    def get_channel_id(self, source: str, platform_id: str) -> str:
        """Build a canonical channel_id string from platform identifiers.
        E.g. Telegram: "tg:123456789"
             Discord:  "discord:guild/channel"
             API:      "api:token_hash"
        """
        ...
