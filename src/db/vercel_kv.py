"""Vercel KV client — thin async wrapper around the Upstash Redis REST API.

Vercel KV is powered by Upstash. When a KV database is connected to your
Vercel project, these env vars are injected automatically:

  KV_REST_API_URL          https://<name>.upstash.io
  KV_REST_API_TOKEN        the read-write token
  KV_REST_API_READ_ONLY_TOKEN  (not used here — write access required)

Usage:
    from src.db.vercel_kv import kv
    await kv.set("session:123", json.dumps(data), ex=3600)
    raw = await kv.get("session:123")
    await kv.delete("session:123")

All methods return None and log a warning when KV is not configured, so
call sites do not need to guard against missing env vars.
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

import httpx

from src.utils.logger import logger

_KV_URL   = os.environ.get("KV_REST_API_URL", "").rstrip("/")
_KV_TOKEN = os.environ.get("KV_REST_API_TOKEN", "")
_ENABLED  = bool(_KV_URL and _KV_TOKEN)


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {_KV_TOKEN}",
        "Content-Type": "application/json",
    }


async def _cmd(*args: Any) -> Any:
    """Execute a raw Upstash Redis REST command and return the result."""
    if not _ENABLED:
        return None
    url = f"{_KV_URL}/pipeline"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(url, headers=_headers(), json=[list(args)])
            r.raise_for_status()
            data = r.json()
            # pipeline response: [{"result": ..., "error": null}]
            if isinstance(data, list) and data:
                entry = data[0]
                if entry.get("error"):
                    logger.warning("vercel_kv_error", cmd=args[0], error=entry["error"])
                    return None
                return entry.get("result")
    except Exception as exc:
        logger.warning("vercel_kv_request_failed", cmd=args[0], error=str(exc))
    return None


class _KVClient:
    """Async Vercel KV / Upstash REST client."""

    @property
    def enabled(self) -> bool:
        return _ENABLED

    async def get(self, key: str) -> Optional[str]:
        """Return value for key, or None if not found."""
        result = await _cmd("GET", key)
        return result  # already a string or None

    async def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        """Set key to value. ex = TTL in seconds (optional)."""
        if ex is not None:
            result = await _cmd("SET", key, value, "EX", str(ex))
        else:
            result = await _cmd("SET", key, value)
        return result == "OK"

    async def delete(self, key: str) -> int:
        """Delete key. Returns number of keys deleted (0 or 1)."""
        result = await _cmd("DEL", key)
        return int(result or 0)

    async def exists(self, key: str) -> bool:
        """Return True if key exists."""
        result = await _cmd("EXISTS", key)
        return int(result or 0) > 0

    async def expire(self, key: str, seconds: int) -> bool:
        """Set TTL on existing key. Returns True if key exists."""
        result = await _cmd("EXPIRE", key, str(seconds))
        return int(result or 0) == 1

    async def incr(self, key: str) -> int:
        """Increment integer value atomically."""
        result = await _cmd("INCR", key)
        return int(result or 0)

    async def get_json(self, key: str) -> Optional[Any]:
        """Convenience: get + JSON decode."""
        raw = await self.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw

    async def set_json(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        """Convenience: JSON encode + set."""
        return await self.set(key, json.dumps(value, ensure_ascii=False), ex=ex)


# Singleton — import and use directly
kv = _KVClient()
