from __future__ import annotations

from typing import Optional

from src.core.interface_adapter import InterfaceAdapter


class PlatformRouter:
    """Routes channel_ids to the correct InterfaceAdapter by prefix.

    Each platform registers with a prefix (e.g. "tg:", "discord:", "api:").
    When a channel_id comes in, the first matching prefix wins.
    Falls back to the primary (first-registered) adapter.
    """

    def __init__(self) -> None:
        self._adapters: dict[str, InterfaceAdapter] = {}
        self._primary: Optional[InterfaceAdapter] = None

    def register(self, prefix: str, adapter: InterfaceAdapter) -> None:
        self._adapters[prefix] = adapter
        if self._primary is None:
            self._primary = adapter

    def route(self, channel_id: str) -> InterfaceAdapter:
        for prefix, adapter in self._adapters.items():
            if prefix and channel_id.startswith(prefix):
                return adapter
        if self._primary is not None:
            return self._primary
        raise ValueError(f"No adapter registered for channel_id: {channel_id!r}")

    @property
    def primary(self) -> Optional[InterfaceAdapter]:
        return self._primary
