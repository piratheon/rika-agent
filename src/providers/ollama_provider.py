"""Ollama provider — local LLM inference, zero API cost.

Connects to a locally running Ollama instance (default: http://localhost:11434).
Configure OLLAMA_BASE_URL in .env to point at a remote host.

Workflow:
  1. test_key() pings /api/tags to verify the instance is reachable.
  2. request() posts to /api/chat with stream=false.
  3. stream() posts with stream=true and reads NDJSON chunks.

Model handling:
  - Uses the model name from payload["model"] as-is.
  - If the requested model is not found (404), falls back to the first
    available model returned by /api/tags.
  - "No API key" concept: the api_key field is ignored; pass any string.
"""
from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx

from src.providers.base_provider import (
    BaseProvider,
    ProviderAuthError,
    ProviderQuotaError,
    ProviderTransientError,
)
from src.utils.logger import logger

import os

_DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaProvider(BaseProvider):
    def __init__(self, api_key: str = "", provider_name: str = "ollama") -> None:
        super().__init__(api_key or "ollama", provider_name)
        self.base_url = os.environ.get("OLLAMA_BASE_URL", _DEFAULT_BASE_URL).rstrip("/")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _chat_url(self) -> str:
        return f"{self.base_url}/api/chat"

    def _tags_url(self) -> str:
        return f"{self.base_url}/api/tags"

    def _extract_messages(self, payload: Dict[str, Any]) -> List[Dict[str, str]]:
        """Convert payload messages to Ollama format (same as OpenAI format)."""
        messages = payload.get("messages", [])
        # Ollama accepts role/content dicts directly
        return [
            {"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in messages
            if isinstance(m, dict)
        ]

    async def _get_available_models(self) -> List[str]:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(self._tags_url())
                if r.status_code == 200:
                    data = r.json()
                    return [m["name"] for m in data.get("models", [])]
        except Exception:
            pass
        return []

    async def _resolve_model(self, requested: str) -> str:
        """Return requested model if available; otherwise first available model."""
        available = await self._get_available_models()
        if not available:
            return requested  # let the request fail naturally
        if requested in available:
            return requested
        # Try prefix match (e.g. "llama3" matches "llama3.2:latest")
        for m in available:
            if m.startswith(requested.split(":")[0]):
                logger.info("ollama_model_fallback", requested=requested, using=m)
                return m
        logger.info("ollama_model_fallback_first", requested=requested, using=available[0])
        return available[0]

    # ------------------------------------------------------------------
    # BaseProvider interface
    # ------------------------------------------------------------------

    async def request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        model = await self._resolve_model(payload.get("model", "llama3.2"))
        messages = self._extract_messages(payload)

        body = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": payload.get("max_tokens", 2048)},
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                r = await client.post(self._chat_url(), json=body)

                if r.status_code == 404:
                    raise ProviderTransientError(f"Ollama model '{model}' not found (404).")
                if r.status_code >= 500:
                    raise ProviderTransientError(f"Ollama server error: {r.status_code}")
                r.raise_for_status()

                data = r.json()
                content = data.get("message", {}).get("content", "")
                tokens_in = data.get("prompt_eval_count", 0)
                tokens_out = data.get("eval_count", 0)

                return {
                    "output": content,
                    "usage": {
                        "prompt_tokens": tokens_in,
                        "completion_tokens": tokens_out,
                        "total_tokens": tokens_in + tokens_out,
                    },
                    "model": data.get("model", model),
                }

        except httpx.ConnectError:
            raise ProviderTransientError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running: ollama serve"
            )
        except httpx.TimeoutException:
            raise ProviderTransientError("Ollama request timed out (120s).")
        except (ProviderTransientError, ProviderAuthError, ProviderQuotaError):
            raise
        except Exception as exc:
            logger.error("ollama_request_failed", error=str(exc))
            raise ProviderTransientError(f"Ollama error: {exc}")

    async def stream(self, payload: Dict[str, Any]) -> AsyncGenerator[str, None]:
        model = await self._resolve_model(payload.get("model", "llama3.2"))
        messages = self._extract_messages(payload)
        body = {"model": model, "messages": messages, "stream": True}

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream("POST", self._chat_url(), json=body) as resp:
                    if resp.status_code >= 400:
                        raise ProviderTransientError(f"Ollama stream error: {resp.status_code}")
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                            chunk = data.get("message", {}).get("content", "")
                            if chunk:
                                yield chunk
                            if data.get("done"):
                                break
                        except json.JSONDecodeError:
                            continue
        except httpx.ConnectError:
            raise ProviderTransientError(f"Cannot connect to Ollama at {self.base_url}.")
        except (ProviderTransientError, ProviderAuthError, ProviderQuotaError):
            raise
        except Exception as exc:
            raise ProviderTransientError(f"Ollama stream error: {exc}")

    async def test_key(self) -> bool:
        """Test reachability by listing available models."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(self._tags_url())
                if r.status_code == 200:
                    models = [m["name"] for m in r.json().get("models", [])]
                    logger.info("ollama_available_models", count=len(models), models=models[:5])
                    return True
                return False
        except httpx.ConnectError:
            raise ProviderTransientError(
                f"Ollama not reachable at {self.base_url}. Start it with: ollama serve"
            )
        except Exception as exc:
            raise ProviderTransientError(f"Ollama test failed: {exc}")

    async def list_models(self) -> List[str]:
        """Return list of locally available Ollama models."""
        return await self._get_available_models()
