"""Vercel AI Gateway provider — OpenAI-compatible with function calling.

Auto-enabled when the VERCEL environment variable is set (Vercel deployment)
and VERCEL_API_KEY is present. Can also be used from any environment by
setting VERCEL_API_KEY and enabling it in config.json.

Gateway docs: https://sdk.vercel.ai/docs/ai-sdk-core/provider-management
"""
from __future__ import annotations

import json
import os
from typing import Any, AsyncGenerator, Dict, List

import httpx

from src.providers.base_provider import (
    BaseProvider,
    ProviderAuthError,
    ProviderQuotaError,
    ProviderTransientError,
    StructuredResponse,
    ToolCall,
)
from src.utils.logger import logger

_DEFAULT_MODEL = "openai/gpt-4o-mini"
_GATEWAY_BASE = "https://ai-gateway.vercel.sh/v1"


class VercelProvider(BaseProvider):
    SUPPORTS_FUNCTION_CALLING = True

    def __init__(self, api_key: str, provider_name: str = "vercel") -> None:
        super().__init__(api_key, provider_name)
        self.base_url = os.environ.get("VERCEL_GATEWAY_URL", _GATEWAY_BASE).rstrip("/")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _fix_model(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        model = payload.get("model", "")
        if not model:
            payload = dict(payload)
            payload["model"] = _DEFAULT_MODEL
        return payload

    def _raise_for_status(self, r: httpx.Response) -> None:
        if r.status_code == 401:
            raise ProviderAuthError(f"Vercel auth failed: {r.text[:200]}")
        if r.status_code == 429:
            raise ProviderQuotaError(f"Vercel rate limit: {r.text[:200]}")
        if r.status_code >= 400:
            raise ProviderTransientError(f"Vercel HTTP {r.status_code}: {r.text[:200]}")

    # ------------------------------------------------------------------
    # BaseProvider interface
    # ------------------------------------------------------------------

    async def request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        payload = self._fix_model(payload)
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
            )
            self._raise_for_status(r)
            data = r.json()
            content = self._extract_openai_content(data.get("choices", []))
            return {
                "output": content or "",
                "usage": data.get("usage", {}),
                "raw_response": data,
            }

    async def request_with_tools(
        self, payload: Dict[str, Any], tool_schemas: List[Any]
    ) -> StructuredResponse:
        payload = dict(self._fix_model(payload))
        if tool_schemas:
            payload["tools"] = [s.to_openai() for s in tool_schemas]
            payload["tool_choice"] = "auto"
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
            )
            self._raise_for_status(r)
            data = r.json()
            choices = data.get("choices", [])
            tool_calls = self._parse_openai_tool_calls(choices) if tool_schemas else []
            content = self._extract_openai_content(choices) or ""
            return StructuredResponse(
                content=content,
                tool_calls=tool_calls,
                usage=data.get("usage", {}),
                model=data.get("model", ""),
            )

    async def stream(self, payload: Dict[str, Any]) -> AsyncGenerator[str, None]:
        payload = dict(self._fix_model(payload))
        payload["stream"] = True
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
            ) as resp:
                self._raise_for_status(resp)
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    raw = line[6:]
                    if raw == "[DONE]":
                        break
                    try:
                        delta = json.loads(raw)["choices"][0].get("delta", {})
                        if chunk := delta.get("content"):
                            yield chunk
                    except Exception:
                        continue

    async def test_key(self) -> bool:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": _DEFAULT_MODEL,
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 5,
                },
                headers=self._headers(),
            )
            if r.status_code == 200:
                return True
            if r.status_code == 401:
                raise ProviderAuthError("Vercel auth failed")
            if r.status_code == 429:
                raise ProviderQuotaError("Vercel rate limit")
            raise ProviderTransientError(f"Vercel test_key: {r.status_code}")


def is_vercel_environment() -> bool:
    """Return True when running inside a Vercel deployment."""
    return os.environ.get("VERCEL", "") == "1" or bool(
        os.environ.get("VERCEL_ENV", "")
    )
