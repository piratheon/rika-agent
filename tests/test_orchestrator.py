"""Tests for Orchestrator class."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.orchestrator import Orchestrator
from src.core.models import AgentState, AgentStatus, ToolCall, ToolResult


@pytest.fixture
def mock_config():
    """Return a mock Config with sensible defaults."""
    cfg = MagicMock()
    cfg.default_provider_priority = ["gemini", "groq"]
    cfg.default_model = "gemini-2.0-flash"
    cfg.groq_model = "llama-3.3-70b-versatile"
    cfg.openrouter_model = "google/gemini-2.0-flash-001"
    cfg.gemini_model = "gemini-2.0-flash"
    cfg.ollama_model = "llama3.2"
    cfg.g4f_model = "MiniMaxAI/MiniMax-M2.5"
    cfg.system_prompt = "You are helpful"
    cfg.max_turns = 20
    cfg.provider_max_retries = 2
    cfg.provider_retry_delay = 2.0
    cfg.live_bubble_throttle_ms = 800
    cfg.tool_timeout_seconds = 10
    return cfg


@pytest.fixture
def orchestrator(mock_config):
    with patch("src.core.orchestrator.Config.get", return_value=mock_config), \
         patch("src.core.orchestrator.get_pool"), \
         patch("src.core.orchestrator.ToolExecutor"), \
         patch("src.core.orchestrator.ContextBuilder"), \
         patch("src.core.orchestrator.get_all_schemas", return_value=[]):
        o = Orchestrator(
            on_status_update=AsyncMock(),
            on_tool_call=AsyncMock(),
            on_tool_result=AsyncMock(),
            on_message=AsyncMock(),
            is_cancelled=lambda cid: False,
        )
        return o


class TestOrchestratorInit:
    def test_default_model_map(self, mock_config):
        with patch("src.core.orchestrator.Config.get", return_value=mock_config), \
             patch("src.core.orchestrator.get_pool"), \
             patch("src.core.orchestrator.ToolExecutor"), \
             patch("src.core.orchestrator.ContextBuilder"):
            o = Orchestrator()
            assert "groq" in o.model_map
            assert "gemini" in o.model_map
            assert "openrouter" in o.model_map
            assert "g4f" in o.model_map

    def test_callbacks_stored(self, mock_config):
        cb = AsyncMock()
        with patch("src.core.orchestrator.Config.get", return_value=mock_config), \
             patch("src.core.orchestrator.get_pool"), \
             patch("src.core.orchestrator.ToolExecutor"), \
             patch("src.core.orchestrator.ContextBuilder"):
            o = Orchestrator(on_message=cb)
            assert o.on_message is cb


class TestNotify:
    @pytest.mark.asyncio
    async def test_notify_status_calls_callback(self, orchestrator):
        state = AgentState(status=AgentStatus.THINKING, user_id=1, channel_id="tg:1")
        await orchestrator._notify_status(state)
        orchestrator.on_status_update.assert_called_once_with(state)

    @pytest.mark.asyncio
    async def test_notify_status_swallows_error(self, orchestrator):
        orchestrator.on_status_update.side_effect = Exception("cb error")
        state = AgentState(status=AgentStatus.THINKING, user_id=1, channel_id="tg:1")
        await orchestrator._notify_status(state)  # should not raise

    @pytest.mark.asyncio
    async def test_notify_tool_call(self, orchestrator):
        tc = ToolCall(name="test_tool", arguments={}, call_id="1")
        await orchestrator._notify_tool_call(tc)
        orchestrator.on_tool_call.assert_called_once_with(tc)

    @pytest.mark.asyncio
    async def test_notify_tool_result(self, orchestrator):
        tr = ToolResult(tool_name="test", result="ok", call_id="1")
        await orchestrator._notify_tool_result(tr)
        orchestrator.on_tool_result.assert_called_once_with(tr)


class TestGetProviderResponse:
    @pytest.mark.asyncio
    async def test_returns_response_from_first_provider(self, orchestrator):
        mock_resp = MagicMock()
        mock_resp.has_tool_calls = False
        mock_resp.content = "hello"
        orchestrator.pool.request_with_key_structured = AsyncMock(return_value=mock_resp)

        resp = await orchestrator._get_provider_response(
            user_id=1,
            priorities=["gemini"],
            messages=[{"role": "user", "content": "hi"}],
            tool_schemas=[],
        )
        assert resp is mock_resp

    @pytest.mark.asyncio
    async def test_returns_none_when_all_providers_fail(self, orchestrator):
        orchestrator.pool.request_with_key_structured = AsyncMock(
            side_effect=Exception("fail"),
        )

        resp = await orchestrator._get_provider_response(
            user_id=1,
            priorities=["gemini", "groq"],
            messages=[{"role": "user", "content": "hi"}],
            tool_schemas=[],
        )
        assert resp is None

    @pytest.mark.asyncio
    async def test_skips_provider_on_timeout(self, orchestrator):
        mock_resp = MagicMock()
        mock_resp.has_tool_calls = False
        mock_resp.content = "ok"

        async def side_effect(*a, **kw):
            import asyncio
            raise asyncio.TimeoutError()

        orchestrator.pool.request_with_key_structured = AsyncMock(
            side_effect=[side_effect(), mock_resp],
        )

        resp = await orchestrator._get_provider_response(
            user_id=1,
            priorities=["gemini", "openrouter"],
            messages=[{"role": "user", "content": "hi"}],
            tool_schemas=[],
        )
        assert resp is mock_resp


class TestRun:
    @pytest.mark.asyncio
    async def test_cancelled_returns_early(self, orchestrator):
        orchestrator.is_cancelled = lambda cid: True
        result = await orchestrator.run(
            user_id=1,
            channel_id="tg:1",
            message="hello",
            context_str="",
            system_prompt="You are helpful",
        )
        assert result == "Task cancelled by user."

    @pytest.mark.asyncio
    async def test_max_turns_no_provider_response(self, orchestrator):
        orchestrator.pool.request_with_key_structured = AsyncMock(
            side_effect=Exception("fail"),
        )
        result = await orchestrator.run(
            user_id=1,
            channel_id="tg:1",
            message="hello",
            context_str="",
            system_prompt="You are helpful",
            max_turns=3,
        )
        assert "turn limit" in result.lower()

    @pytest.mark.asyncio
    async def test_provider_returns_content_no_tools(self, orchestrator):
        mock_resp = MagicMock()
        mock_resp.has_tool_calls = False
        mock_resp.content = "Hello world"
        orchestrator.pool.request_with_key_structured = AsyncMock(return_value=mock_resp)

        result = await orchestrator.run(
            user_id=1,
            channel_id="tg:1",
            message="say hi",
            context_str="",
            system_prompt="You are helpful",
        )
        assert "Hello world" in result

    @pytest.mark.asyncio
    async def test_tool_execution_chain(self, orchestrator):
        """One tool call, then final content response."""
        tool_resp = MagicMock()
        tool_resp.has_tool_calls = True
        tool_resp.tool_calls = [MagicMock(name="web_search", arguments={"q": "test"}, call_id="1")]
        tool_resp.content = None

        final_resp = MagicMock()
        final_resp.has_tool_calls = False
        final_resp.content = "Final answer"

        orchestrator.pool.request_with_key_structured = AsyncMock(
            side_effect=[tool_resp, final_resp],
        )
        orchestrator.tool_executor.execute = AsyncMock(return_value="search results")

        result = await orchestrator.run(
            user_id=1,
            channel_id="tg:1",
            message="search for X",
            context_str="",
            system_prompt="You are helpful",
        )
        assert "Final answer" in result
        assert orchestrator.tool_executor.execute.called
