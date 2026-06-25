"""Tests for PlatformRouter and EventSink."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock

from src.core.router import PlatformRouter


class TestPlatformRouter:
    def test_register_and_route_by_prefix(self):
        router = PlatformRouter()
        tg = AsyncMock()
        api = AsyncMock()
        router.register("tg:", tg)
        router.register("api:", api)

        assert router.route("tg:12345") is tg
        assert router.route("api:abcde") is api

    def test_route_falls_back_to_primary(self):
        router = PlatformRouter()
        tg = AsyncMock()
        api = AsyncMock()
        router.register("tg:", tg)
        router.register("api:", api)

        result = router.route("unknown:999")
        assert result is tg  # primary = first registered

    def test_route_with_no_prefix_match_uses_primary(self):
        router = PlatformRouter()
        tg = AsyncMock()
        router.register("tg:", tg)

        assert router.route("something") is tg

    def test_route_raises_on_empty_router(self):
        router = PlatformRouter()
        with pytest.raises(ValueError, match="No adapter registered"):
            router.route("anything")

    def test_register_replaces_existing_prefix(self):
        router = PlatformRouter()
        old = AsyncMock()
        new = AsyncMock()
        router.register("tg:", old)
        router.register("tg:", new)

        assert router.route("tg:1") is new

    def test_primary_property(self):
        router = PlatformRouter()
        assert router.primary is None

        tg = AsyncMock()
        router.register("tg:", tg)
        assert router.primary is tg

    def test_exact_prefix_match(self):
        router = PlatformRouter()
        tg = AsyncMock()
        api = AsyncMock()
        router.register("tg:", tg)
        router.register("api:", api)

        assert router.route("tg:") is tg      # exact prefix
        assert router.route("api:x") is api    # prefix + suffix

    def test_empty_router_raises(self):
        """A router with no adapters raises on any route."""
        router = PlatformRouter()
        with pytest.raises(ValueError, match="No adapter registered"):
            router.route("anything")
