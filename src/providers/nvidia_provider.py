"""NVIDIA NIM provider — OpenAI-compatible API.

NVIDIA NIM exposes hosted models (Llama, Mistral, Nemotron, Gemma, GLM, Phi-3)
via an OpenAI-compatible REST API at https://integrate.api.nvidia.com/v1

Environment variables:
  NVIDIA_API_KEY   — API key from https://build.nvidia.com

Auto-detection:
  If NVIDIA_API_KEY is present the provider is prepended to provider priority.

Default model: meta/llama-3.1-70b-instruct
All models: https://build.nvidia.com/models
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
)
from src.utils.logger import logger


class NvidiaProvider(BaseProvider):
    """NVIDIA NIM adapter — matches the BaseProvider contract used by
    ProviderPool (request, request_with_tools, stream, test_key).

    Previously this was a standalone class with chat()/chat_with_tools()/
    stream_chat() methods that did not match the interface ProviderPool
    actually calls (request/request_with_tools/stream/test_key). Every call
    to adapter.test_key() raised AttributeError, was swallowed by a bare
    except Exception: pass in get_healthy_key(), and silently reported the
    key as invalid regardless of whether it was actually valid.
    """

    SUPPORTS_FUNCTION_CALLING = True

    def __init__(self, api_key: str, provider_name: str = "nvidia"):
        super().__init__(api_key, provider_name)
        self.base_url = os.environ.get(
            "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"
        ).rstrip("/")

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _raise(self, r: httpx.Response) -> None:
        if r.status_code == 401:
            raise ProviderAuthError(f"NVIDIA NIM auth: {r.text[:200]}")
        if r.status_code == 429:
            raise ProviderQuotaError(f"NVIDIA NIM quota: {r.text[:200]}")
        if r.status_code >= 400:
            raise ProviderTransientError(f"NVIDIA NIM {r.status_code}: {r.text[:200]}")

    async def request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=90.0) as client:
            r = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
            )
            self._raise(r)
            data = r.json()
            content = self._extract_openai_content(data.get("choices", []))
            return {"output": content or "", "usage": data.get("usage", {})}

    async def request_with_tools(
        self, payload: Dict[str, Any], tool_schemas: List[Any]
    ) -> StructuredResponse:
        payload = dict(payload)
        if tool_schemas:
            payload["tools"] = [s.to_openai() for s in tool_schemas]
            payload["tool_choice"] = "auto"
        async with httpx.AsyncClient(timeout=90.0) as client:
            r = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
            )
            self._raise(r)
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
        payload = dict(payload)
        payload["stream"] = True
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
            ) as r:
                self._raise(r)
                async for line in r.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    raw = line[6:].strip()
                    if raw == "[DONE]":
                        break
                    try:
                        delta = json.loads(raw)["choices"][0].get("delta", {})
                        if content := delta.get("content"):
                            yield content
                    except Exception:
                        continue

    async def test_key(self) -> bool:
        """Validate the key with a minimal real completion call.

        NIM does not expose a lightweight /models endpoint suitable for key
        validation the way OpenRouter does, so a 1-token completion against
        a small, reliably-hosted model is used instead.
        """
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": "meta/llama-3.1-8b-instruct",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 1,
                },
                headers=self._headers(),
            )
            if r.status_code == 200:
                return True
            if r.status_code == 401:
                raise ProviderAuthError("NVIDIA NIM auth failed")
            if r.status_code == 429:
                raise ProviderQuotaError("NVIDIA NIM quota exceeded")
            raise ProviderTransientError(f"NVIDIA NIM test: {r.status_code} {r.text[:200]}")
