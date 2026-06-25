from __future__ import annotations

import asyncio
from typing import Optional, Union

from src.core.interface_adapter import InterfaceAdapter
from src.core.router import PlatformRouter


class EventSink:
    """Delivery abstraction — wraps one or more InterfaceAdapters via PlatformRouter.

    This is what the Orchestrator and BackgroundAgentManager hold.
    All methods route to the correct platform adapter based on channel_id prefix.

    Accepts either a single InterfaceAdapter (single-platform, backward compat)
    or a PlatformRouter (multi-platform).
    """

    def __init__(self, adapter_or_router: Union[InterfaceAdapter, PlatformRouter]) -> None:
        if isinstance(adapter_or_router, PlatformRouter):
            self._router = adapter_or_router
        else:
            self._router = PlatformRouter()
            self._router.register("", adapter_or_router)

    # ------------------------------------------------------------------
    # Basic text/file operations
    # ------------------------------------------------------------------

    async def send_text(
        self, channel_id: str, text: str, **kwargs
    ) -> str:
        adapter = self._router.route(channel_id)
        return await adapter.send_text(channel_id, text, **kwargs)

    async def edit_message(
        self, handle: str, text: str, **kwargs
    ) -> None:
        adapter = self._router.primary
        if adapter is not None:
            await adapter.edit_message(handle, text, **kwargs)

    async def send_file(
        self, channel_id: str, file_path: str, caption: str = "", **kwargs
    ) -> bool:
        adapter = self._router.route(channel_id)
        return await adapter.send_file(channel_id, file_path, caption, **kwargs)

    async def send_chunked(
        self, channel_id: str, text: str, max_len: int = 4000
    ) -> list[str]:
        adapter = self._router.route(channel_id)
        return await adapter.send_chunked(channel_id, text, max_len)

    async def send_typing(self, channel_id: str) -> None:
        adapter = self._router.route(channel_id)
        await adapter.send_typing(channel_id)

    # ------------------------------------------------------------------
    # Interactive countdown
    # ------------------------------------------------------------------

    async def show_countdown(
        self,
        handle: str,
        wait_seconds: int = 30,
        attempt: int = 1,
        on_stop: Optional[asyncio.Event] = None,
        on_retry: Optional[asyncio.Event] = None,
    ) -> str:
        adapter = self._router.primary
        if adapter is not None and adapter.supports_interactive():
            return await adapter.show_countdown(
                handle, wait_seconds, attempt,
                on_stop=on_stop, on_retry=on_retry,
            )
        await asyncio.sleep(wait_seconds)
        return "done"

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_text(self, text: str, mode: str = "HTML") -> str:
        adapter = self._router.primary
        if adapter is not None:
            return adapter.format_text(text, mode)
        return text

    # ------------------------------------------------------------------
    # Channel identity
    # ------------------------------------------------------------------

    def get_channel_id(self, source: str, platform_id: str) -> str:
        adapter = self._router.primary
        if adapter is not None:
            return adapter.get_channel_id(source, platform_id)
        return f"{source}:{platform_id}"

    # ------------------------------------------------------------------
    # Task lifecycle (delegated to the correct adapter by channel_id)
    # ------------------------------------------------------------------

    def is_cancelled(self, channel_id: str) -> bool:
        adapter = self._router.route(channel_id)
        if hasattr(adapter, "is_cancelled"):
            return adapter.is_cancelled(channel_id)
        return False

    def clear_cancel(self, channel_id: str) -> None:
        adapter = self._router.route(channel_id)
        if hasattr(adapter, "clear_cancel"):
            adapter.clear_cancel(channel_id)

    def track_task(self, channel_id: str, task: asyncio.Task) -> None:
        adapter = self._router.route(channel_id)
        if hasattr(adapter, "track_task"):
            adapter.track_task(channel_id, task)

    def untrack_task(self, channel_id: str) -> None:
        adapter = self._router.route(channel_id)
        if hasattr(adapter, "untrack_task"):
            adapter.untrack_task(channel_id)

    def init_countdown_events(self, channel_id: str) -> None:
        adapter = self._router.route(channel_id)
        if hasattr(adapter, "init_countdown_events"):
            adapter.init_countdown_events(channel_id)

    @property
    def router(self) -> PlatformRouter:
        return self._router
