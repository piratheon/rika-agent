"""Shared test fixtures and mocks."""
from __future__ import annotations

import os
from typing import AsyncGenerator
from unittest.mock import AsyncMock

import pytest


@pytest.fixture(autouse=True)
def clear_env():
    """Clear relevant env vars before each test and restore after."""
    saved = {}
    keys = [
        "TELEGRAM_BOT_TOKEN", "POSTGRES_URL", "DATABASE_PATH",
        "GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY",
    ]
    for k in keys:
        saved[k] = os.environ.pop(k, None)
    yield
    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v


@pytest.fixture
def mock_event_sink():
    """Create a fully mocked EventSink."""
    from src.core.event_sink import EventSink
    sink = AsyncMock(spec=EventSink)
    sink.send_text = AsyncMock()
    sink.send_file = AsyncMock()
    sink.edit_message = AsyncMock()
    sink.is_cancelled = AsyncMock(return_value=False)
    sink.clear_cancel = AsyncMock()
    return sink


@pytest.fixture
def mock_adapter():
    """Create a generic mocked InterfaceAdapter."""
    adapter = AsyncMock()
    adapter.request = AsyncMock(return_value={"output": "ok", "usage": {"total_tokens": 10}})
    return adapter
