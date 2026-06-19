"""NVIDIA NIM provider — OpenAI-compatible API.

NVIDIA NIM exposes hosted models (Llama, Mistral, Nemotron, Gemma, Phi-3)
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
from typing import Any, Dict, List, Optional

import httpx

from src.utils.logger import logger

NVIDIA_BASE_URL = os.environ.get(
    "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"
).rstrip("/")

SUPPORTS_FUNCTION_CALLING = True
SUPPORTS_STREAMING = True


def _headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _raise_for_status(r: httpx.Response) -> None:
    if r.status_code == 401:
        raise ValueError("NVIDIA NIM 401: invalid API key")
    if r.status_code == 429:
        raise ValueError("NVIDIA NIM 429: rate limit exceeded")
    if r.status_code >= 400:
        raise ValueError(f"NVIDIA NIM HTTP {r.status_code}: {r.text[:200]}")


async def request(
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    async with httpx.AsyncClient(timeout=90.0) as client:
        r = await client.post(
            f"{NVIDIA_BASE_URL}/chat/completions",
            json=payload,
            headers=_headers(api_key),
        )
        _raise_for_status(r)
        data = r.json()
        return {"output": data["choices"][0]["message"]["content"] or "", "raw": data}


async def request_with_tools(
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    async with httpx.AsyncClient(timeout=90.0) as client:
        r = await client.post(
            f"{NVIDIA_BASE_URL}/chat/completions",
            json=payload,
            headers=_headers(api_key),
        )
        _raise_for_status(r)
        return r.json()


async def stream(
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float = 0.7,
    max_tokens: int = 2048,
):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    async with httpx.AsyncClient(timeout=90.0) as client:
        async with client.stream(
            "POST",
            f"{NVIDIA_BASE_URL}/chat/completions",
            json=payload,
            headers=_headers(api_key),
        ) as r:
            _raise_for_status(r)
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    text = chunk["choices"][0].get("delta", {}).get("content") or ""
                    if text:
                        yield text
                except Exception:
                    continue


async def test_key(api_key: str) -> bool:
    try:
        await request(
            api_key=api_key,
            model="meta/llama-3.1-8b-instruct",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
        )
        return True
    except Exception as exc:
        logger.warning("nvidia_key_test_failed", error=str(exc))
        return False


class NvidiaProvider:
    """Class-based adapter matching the project's provider pattern."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    async def chat(self, model: str, messages: list, **kwargs) -> Dict[str, Any]:
        return await request(self.api_key, model, messages, **kwargs)

    async def chat_with_tools(self, model: str, messages: list, tools: list = None, **kwargs) -> Dict[str, Any]:
        return await request_with_tools(self.api_key, model, messages, tools=tools, **kwargs)

    async def stream_chat(self, model: str, messages: list, **kwargs):
        async for chunk in stream(self.api_key, model, messages, **kwargs):
            yield chunk
