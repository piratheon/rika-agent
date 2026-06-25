import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.providers.provider_pool import ProviderPool
from src.providers.base_provider import ProviderQuotaError, ProviderAuthError, ProviderTransientError

@pytest.mark.asyncio
async def test_request_with_key_failover():
    """Test that ProviderPool fails over to the next key on quota error."""
    pool = ProviderPool()

    mock_keys = [
        {"id": 1, "provider": "gemini", "is_blacklisted": False, "last_used_at": None},
        {"id": 2, "provider": "gemini", "is_blacklisted": False, "last_used_at": None},
    ]

    with (
        patch("src.db.key_store.list_api_keys", AsyncMock(return_value=mock_keys)),
        patch("src.db.key_store.get_api_key_raw", AsyncMock(side_effect=[b"key1", b"key2"])),
        patch("src.db.key_store.blacklist_key", AsyncMock()) as mock_blacklist,
        patch("src.db.key_store.update_key_last_used", AsyncMock()),
    ):
        mock_adapter1 = AsyncMock()
        mock_adapter1.request.side_effect = ProviderQuotaError("Quota exceeded")

        mock_adapter2 = AsyncMock()
        mock_adapter2.request.return_value = {"output": "success", "usage": {"total_tokens": 10}}

        with patch.object(pool, "_make_adapter", side_effect=[mock_adapter1, mock_adapter2]):
            resp = await pool.request_with_key(user_id=1, provider="gemini", payload={"text": "hi"})

            assert resp["output"] == "success"
            mock_blacklist.assert_called_once()
            assert mock_blacklist.call_args[0][0] == 1
            assert mock_blacklist.call_args[1]["reason"] == "quota_exceeded"


@pytest.mark.asyncio
async def test_request_with_key_all_fail():
    """Test that ProviderPool raises error if all keys fail."""
    pool = ProviderPool()

    mock_keys = [
        {"id": 1, "provider": "gemini", "is_blacklisted": False, "last_used_at": None},
    ]

    with (
        patch("src.db.key_store.list_api_keys", AsyncMock(return_value=mock_keys)),
        patch("src.db.key_store.get_api_key_raw", AsyncMock(return_value=b"key1")),
        patch("src.db.key_store.blacklist_key", AsyncMock()),
    ):
        mock_adapter = AsyncMock()
        mock_adapter.request.side_effect = ProviderAuthError("Invalid key")

        with patch.object(pool, "_make_adapter", return_value=mock_adapter):
            with pytest.raises(RuntimeError, match="No gemini API key found"):
                await pool.request_with_key(user_id=1, provider="gemini", payload={"text": "hi"})


@pytest.mark.asyncio
async def test_request_with_key_transient_retry():
    """Test that ProviderPool retries on transient errors (handled by decorator)."""
    pool = ProviderPool()

    mock_keys = [{"id": 1, "provider": "gemini", "is_blacklisted": False, "last_used_at": None}]

    with (
        patch("src.db.key_store.list_api_keys", AsyncMock(return_value=mock_keys)),
        patch("src.db.key_store.get_api_key_raw", AsyncMock(return_value=b"key1")),
        patch("src.db.key_store.update_key_last_used", AsyncMock()),
    ):
        mock_adapter = AsyncMock()
        mock_adapter.request.side_effect = [ProviderTransientError("Timeout"), {"output": "ok"}]

        with patch.object(pool, "_make_adapter", return_value=mock_adapter):
            resp = await pool.request_with_key(user_id=1, provider="gemini", payload={"text": "hi"})
            assert resp["output"] == "ok"
            assert mock_adapter.request.call_count == 2
