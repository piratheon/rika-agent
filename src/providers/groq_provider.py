"""Groq provider — OpenAI-compatible with native function calling."""
from __future__ import annotations
import json, os
import httpx
from typing import Any, AsyncGenerator, Dict, List
from src.providers.base_provider import (BaseProvider, ProviderAuthError, ProviderQuotaError,
                                          ProviderTransientError, StructuredResponse, ToolCall)
from src.utils.logger import logger

_DEFAULT_MODEL = "llama-3.3-70b-versatile"
_VISION_MODEL  = "llama-3.2-11b-vision-preview"

class GroqToolUseFailedError(Exception):
    """Groq rejected the tool schema (too many tools / malformed generation).

    This is NOT a key problem. The key is valid. The fix is to retry
    with a smaller tool set, not a different key.
    """


class GroqProvider(BaseProvider):
    SUPPORTS_FUNCTION_CALLING = True

    def __init__(self, api_key: str, provider_name: str = "groq"):
        super().__init__(api_key, provider_name)
        self.base_url = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai")

    def _fix_model(self, payload: dict) -> dict:
        model = payload.get("model", "")
        if not model or any(x in model.lower() for x in ["gemini", "gpt", "claude"]):
            payload = dict(payload)
            payload["model"] = _DEFAULT_MODEL
        return payload

    def _make_headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _raise_for_status(self, r: httpx.Response) -> None:
        if r.status_code == 401: raise ProviderAuthError(f"Groq auth failed: {r.text[:200]}")
        if r.status_code == 429: raise ProviderQuotaError(f"Groq quota: {r.text[:200]}")
        if r.status_code == 400:
            try:
                body = r.json()
                code = body.get("error", {}).get("code", "")
            except Exception:
                code = ""
            if code == "tool_use_failed":
                raise GroqToolUseFailedError(
                    f"Groq tool_use_failed (too many tools or malformed schema): "
                    f"{body.get('error',{}).get('message','')[:200]}"
                )
            raise ProviderTransientError(f"Groq HTTP 400: {r.text[:200]}")
        if r.status_code >= 400: raise ProviderTransientError(f"Groq HTTP {r.status_code}: {r.text[:200]}")

    async def request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        payload = self._fix_model(payload)
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(f"{self.base_url}/v1/chat/completions",
                                  json=payload, headers=self._make_headers())
            self._raise_for_status(r)
            data = r.json()
            content = self._extract_openai_content(data.get("choices", []))
            return {"output": content or "", "usage": data.get("usage", {}), "raw_response": data}

    async def request_with_tools(self, payload: Dict[str, Any], tool_schemas: List[Any]) -> StructuredResponse:
        payload = dict(self._fix_model(payload))
        if tool_schemas:
            payload["tools"] = [s.to_openai(strip_enum=True) for s in tool_schemas]
            payload["tool_choice"] = "auto"
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(f"{self.base_url}/v1/chat/completions",
                                  json=payload, headers=self._make_headers())
            self._raise_for_status(r)
            data = r.json()
            choices = data.get("choices", [])
            tool_calls = self._parse_openai_tool_calls(choices) if tool_schemas else []
            content = self._extract_openai_content(choices) or ""
            return StructuredResponse(content=content, tool_calls=tool_calls,
                                      usage=data.get("usage", {}), model=data.get("model", ""))

    async def stream(self, payload: Dict[str, Any]) -> AsyncGenerator[str, None]:
        payload = dict(self._fix_model(payload))
        payload["stream"] = True
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", f"{self.base_url}/v1/chat/completions",
                                     json=payload, headers=self._make_headers()) as resp:
                self._raise_for_status(resp)
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "): continue
                    raw = line[6:]
                    if raw == "[DONE]": break
                    try:
                        delta = json.loads(raw)["choices"][0].get("delta", {})
                        if c := delta.get("content"): yield c
                    except Exception: continue

    async def test_key(self) -> bool:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(f"{self.base_url}/v1/chat/completions",
                                  json={"model": _DEFAULT_MODEL,
                                        "messages": [{"role": "user", "content": "hi"}],
                                        "max_tokens": 5},
                                  headers=self._make_headers())
            if r.status_code == 200: return True
            if r.status_code == 401: raise ProviderAuthError("Groq auth failed")
            if r.status_code == 429: raise ProviderQuotaError("Groq quota exceeded")
            raise ProviderTransientError(f"Groq test: {r.status_code}")
