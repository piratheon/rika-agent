"""Tests for EventSink."""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.core.event_sink import EventSink
from src.core.router import PlatformRouter
from src.core.interface_adapter import InterfaceAdapter


class MyAdapter(InterfaceAdapter):
    """Minimal adapter for testing."""
    def __init__(self):
        self.sent_texts = []
        self.sent_files = []
        self.edited = []
        self._cancelled = set()

    async def send_text(self, channel_id, text, **kw):
        self.sent_texts.append((channel_id, text))
        return "msg_1"

    async def send_file(self, channel_id, path, caption="", **kw):
        self.sent_files.append((channel_id, path, caption))
        return True

    async def edit_message(self, handle, text, **kw):
        self.edited.append((handle, text))

    async def send_chunked(self, channel_id, text, max_len=4000):
        return [text]

    async def send_typing(self, channel_id):
        pass

    async def show_countdown(self, handle, wait_seconds=30, attempt=1, **kw):
        import asyncio
        await asyncio.sleep(wait_seconds)
        return "done"

    def supports_interactive(self):
        return False

    def format_text(self, text, mode="HTML"):
        return text

    def get_channel_id(self, source, platform_id):
        return f"{source}:{platform_id}"

    def is_cancelled(self, channel_id):
        return channel_id in self._cancelled

    def clear_cancel(self, channel_id):
        self._cancelled.discard(channel_id)

    def track_task(self, channel_id, task):
        pass

    def untrack_task(self, channel_id):
        pass

    def init_countdown_events(self, channel_id):
        pass


class AnotherAdapter(InterfaceAdapter):
    """Second adapter for multi-platform tests."""
    def __init__(self):
        self.sent_texts = []

    async def send_text(self, channel_id, text, **kw):
        self.sent_texts.append((channel_id, text))
        return "msg_2"

    async def send_file(self, channel_id, path, caption="", **kw):
        return True

    async def edit_message(self, handle, text, **kw):
        pass

    async def send_chunked(self, channel_id, text, max_len=4000):
        return [text]

    async def send_typing(self, channel_id):
        pass

    async def show_countdown(self, handle, wait_seconds=30, attempt=1, **kw):
        import asyncio
        await asyncio.sleep(wait_seconds)
        return "done"

    def supports_interactive(self):
        return False

    def format_text(self, text, mode="HTML"):
        return text

    def get_channel_id(self, source, platform_id):
        return f"{source}:{platform_id}"


class TestEventSinkSingleAdapter:
    @pytest.fixture
    def sink(self):
        return EventSink(MyAdapter())

    @pytest.mark.asyncio
    async def test_send_text(self, sink):
        mid = await sink.send_text("tg:1", "hello")
        assert mid == "msg_1"

    @pytest.mark.asyncio
    async def test_send_file(self, sink):
        ok = await sink.send_file("tg:1", "/tmp/f.txt", "caption")
        assert ok is True

    @pytest.mark.asyncio
    async def test_edit_message(self, sink):
        await sink.edit_message("handle_1", "new text")
        assert sink._router.primary.edited == [("handle_1", "new text")]

    @pytest.mark.asyncio
    async def test_is_cancelled_default(self, sink):
        assert sink.is_cancelled("tg:1") is False

    def test_is_cancelled_after_mark(self, sink):
        sink._router.primary._cancelled.add("tg:1")
        assert sink.is_cancelled("tg:1") is True

    def test_clear_cancel(self, sink):
        sink._router.primary._cancelled.add("tg:1")
        sink.clear_cancel("tg:1")
        assert sink.is_cancelled("tg:1") is False

    @pytest.mark.asyncio
    async def test_show_countdown_non_interactive(self, sink):
        result = await sink.show_countdown("h", wait_seconds=0.1)
        assert result == "done"

    def test_format_text(self, sink):
        assert sink.format_text("hi") == "hi"

    def test_get_channel_id(self, sink):
        cid = sink.get_channel_id("tg", "123")
        assert cid == "tg:123"

    @pytest.mark.asyncio
    async def test_send_chunked(self, sink):
        chunks = await sink.send_chunked("tg:1", "long text")
        assert chunks == ["long text"]


class TestEventSinkMultiAdapter:
    @pytest.fixture
    def router(self):
        r = PlatformRouter()
        r.register("tg:", MyAdapter())
        r.register("api:", AnotherAdapter())
        return r

    @pytest.fixture
    def sink(self, router):
        return EventSink(router)

    @pytest.mark.asyncio
    async def test_routes_to_tg_adapter(self, sink):
        await sink.send_text("tg:123", "hello tg")
        tg_adapter = sink._router._adapters["tg:"]
        assert tg_adapter.sent_texts == [("tg:123", "hello tg")]

    @pytest.mark.asyncio
    async def test_routes_to_api_adapter(self, sink):
        await sink.send_text("api:abc", "hello api")
        api_adapter = sink._router._adapters["api:"]
        assert api_adapter.sent_texts == [("api:abc", "hello api")]

    @pytest.mark.asyncio
    async def test_edit_uses_primary(self, sink):
        await sink.edit_message("h", "edit")
        tg_adapter = sink._router._adapters["tg:"]
        assert getattr(tg_adapter, "edited", []) == [("h", "edit")]

    @pytest.mark.asyncio
    async def test_is_cancelled_routes_to_correct_adapter(self, sink):
        tg_adapter = sink._router._adapters["tg:"]

        assert sink.is_cancelled("tg:1") is False
        tg_adapter._cancelled.add("tg:1")
        assert sink.is_cancelled("tg:1") is True

    @pytest.mark.asyncio
    async def test_sink_created_with_platform_router(self, router):
        sink = EventSink(router)
        assert sink.router is router


class TestEventSinkEdgeCases:
    def test_init_with_platform_router(self):
        router = PlatformRouter()
        sink = EventSink(router)
        assert sink.router is router

    def test_primary_routes_to_first_registered(self):
        router = PlatformRouter()
        tg = MyAdapter()
        api = AnotherAdapter()
        router.register("tg:", tg)
        router.register("api:", api)
        sink = EventSink(router)
        assert sink._router.primary is tg
