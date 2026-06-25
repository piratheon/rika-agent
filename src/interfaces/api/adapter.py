from __future__ import annotations

import asyncio
from typing import Optional

from src.core.interface_adapter import InterfaceAdapter


class APIAdapter(InterfaceAdapter):
    """Non-interactive adapter for REST API / HTTP clients.

    Stores outgoing messages in a per-channel buffer. The Flask server
    drains the buffer after orchestration completes and returns the
    accumulated responses to the HTTP caller.
    """

    def __init__(self) -> None:
        self._buffers: dict[str, list[str]] = {}
        self._handle_counter = 0

    # ------------------------------------------------------------------
    # InterfaceAdapter implementation
    # ------------------------------------------------------------------

    async def send_text(
        self, channel_id: str, text: str, **kwargs
    ) -> str:
        self._buffers.setdefault(channel_id, []).append(text)
        self._handle_counter += 1
        return f"api:{channel_id}:{self._handle_counter}"

    async def edit_message(
        self, handle: str, text: str, **kwargs
    ) -> None:
        pass

    async def send_file(
        self, channel_id: str, file_path: str, caption: str = "", **kwargs
    ) -> bool:
        self._buffers.setdefault(channel_id, []).append(
            f"[file: {file_path}] {caption}".strip()
        )
        return True

    async def show_countdown(
        self,
        handle: str,
        wait_seconds: int = 30,
        attempt: int = 1,
        on_stop: Optional[asyncio.Event] = None,
        on_retry: Optional[asyncio.Event] = None,
    ) -> str:
        await asyncio.sleep(wait_seconds)
        return "done"

    def supports_interactive(self) -> bool:
        return False

    def format_text(self, text: str, mode: str = "HTML") -> str:
        return text

    async def send_chunked(
        self, channel_id: str, text: str, max_len: int = 4000
    ) -> list[str]:
        parts = []
        for i in range(0, len(text), max_len):
            chunk = text[i:i + max_len]
            parts.append(chunk)
            self._buffers.setdefault(channel_id, []).append(chunk)
        return parts

    async def send_typing(self, channel_id: str) -> None:
        pass

    def get_channel_id(self, source: str, platform_id: str) -> str:
        return f"api:{platform_id}"

    # ------------------------------------------------------------------
    # API-specific: drain buffered responses for a channel
    # ------------------------------------------------------------------

    def drain(self, channel_id: str) -> list[str]:
        return self._buffers.pop(channel_id, [])
