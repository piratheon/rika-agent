"""Gemini provider — Google genai SDK with native function calling and vision."""
from __future__ import annotations
import asyncio, json
from typing import Any, AsyncGenerator, Dict, List, Optional
from src.providers.base_provider import (BaseProvider, ProviderAuthError, ProviderQuotaError,
                                          ProviderTransientError, StructuredResponse, ToolCall)
from src.utils.logger import logger
from src.config import Config

try:
    from google import genai
    from google.genai import types, errors
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

_DEFAULT = "gemini-2.0-flash"

def _part_text(text: str):
    """Create a text Part compatible with all google-genai SDK versions.

    Newer SDK (1.x+) uses keyword-only: Part.from_text(text=...).
    This helper handles both old and new API transparently.
    """
    try:
        return types.Part.from_text(text=str(text))
    except TypeError:
        return types.Part.from_text(str(text))


def _part_function_call(name: str, args: dict):
    """Create a FunctionCall Part — keyword-only safe."""
    try:
        return types.Part.from_function_call(name=name, args=args)
    except Exception:
        try:
            return types.Part(function_call={"name": name, "args": args})
        except Exception:
            return _part_text(f"[fn_call] {name}({args})")


def _part_function_response(name: str, response: dict):
    """Create a FunctionResponse Part — keyword-only safe."""
    try:
        return types.Part.from_function_response(name=name, response=response)
    except Exception:
        try:
            return types.Part(function_response={"name": name, "response": response})
        except Exception:
            return _part_text(f"[fn_result] {name}: {response}")


class GeminiProvider(BaseProvider):
    SUPPORTS_FUNCTION_CALLING = True

    def __init__(self, api_key: str, provider_name: str = "gemini"):
        super().__init__(api_key, provider_name)
        self.default_model = Config.get().default_model or _DEFAULT

    def _client(self):
        if not HAS_GENAI: raise ProviderTransientError("google-genai not installed")
        return genai.Client(api_key=self.api_key)

    def _loop(self):
        return asyncio.get_running_loop()

    def _extract_messages(self, payload: dict):
        """Convert OpenAI-format messages to Gemini contents + system_instruction.

        Handles all four OpenAI message roles:
          system    → system_instruction string (not a Content entry)
          user      → role="user" Content with text or multimodal parts
          assistant → role="model" Content; tool_calls become FunctionCall parts
          tool      → role="user" Content with FunctionResponse part
        """
        messages = payload.get("messages", [])
        system = ""
        contents = []

        for m in messages:
            role = m.get("role", "user")
            content = m.get("content") or ""

            if role == "system":
                system = content if isinstance(content, str) else str(content)
                continue

            # tool result ────────────────────────────────────────────────────
            if role == "tool":
                result_text = content if isinstance(content, str) else str(content)
                tool_call_id = m.get("tool_call_id", "")
                # Resolve tool name from preceding model turn
                tool_name = tool_call_id
                for prev in reversed(contents):
                    if getattr(prev, "role", None) == "model":
                        for part in (prev.parts or []):
                            fc = getattr(part, "function_call", None)
                            if fc and getattr(fc, "id", None) == tool_call_id:
                                tool_name = getattr(fc, "name", tool_name)
                                break
                        break
                contents.append(types.Content(
                    role="user",
                    parts=[_part_function_response(tool_name, {"result": result_text})],
                ))
                continue

            # assistant / model ───────────────────────────────────────────────
            if role == "assistant":
                parts = []
                if content and isinstance(content, str) and content.strip():
                    parts.append(_part_text(content))
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    raw_args = fn.get("arguments", "{}")
                    try:
                        import json as _j
                        args = _j.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except Exception:
                        args = {}
                    parts.append(_part_function_call(name, args))
                if parts:
                    contents.append(types.Content(role="model", parts=parts))
                continue

            # user ────────────────────────────────────────────────────────────
            if isinstance(content, list):
                parts = []
                for item in content:
                    if item.get("type") == "text":
                        parts.append(_part_text(item["text"]))
                    elif item.get("type") == "image_url":
                        url = item.get("image_url", {}).get("url", "")
                        if url.startswith("data:"):
                            media_type, b64 = url.split(";base64,", 1)
                            media_type = media_type.replace("data:", "")
                            import base64 as _b64
                            parts.append(types.Part.from_bytes(
                                data=_b64.b64decode(b64), mime_type=media_type,
                            ))
                if parts:
                    contents.append(types.Content(role="user", parts=parts))
            else:
                text = content if isinstance(content, str) else str(content)
                if text.strip():
                    contents.append(types.Content(role="user", parts=[_part_text(text)]))

        return contents, system


    def _resolve_model(self, payload: Dict[str, Any]) -> str:
        model = payload.get("model") or self.default_model
        if "gemini" not in model.lower():
            return self.default_model
        return model

    async def request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        client = self._client()
        model = self._resolve_model(payload)
        contents, system = self._extract_messages(payload)
        config = types.GenerateContentConfig(system_instruction=system) if system else None
        try:
            resp = await self._loop().run_in_executor(
                None, lambda: client.models.generate_content(
                    model=model, contents=contents, config=config
                )
            )
            return {
                "output": resp.text or "",
                "usage": {
                    "prompt_tokens": getattr(resp.usage_metadata, "prompt_token_count", 0),
                    "completion_tokens": getattr(resp.usage_metadata, "candidates_token_count", 0),
                    "total_tokens": getattr(resp.usage_metadata, "total_token_count", 0),
                }
            }
        except errors.ClientError as e:
            code = getattr(e, "code", 500)
            if code == 401: raise ProviderAuthError(str(e))
            if code == 429: raise ProviderQuotaError(str(e))
            raise ProviderTransientError(str(e))
        except Exception as e:
            raise ProviderTransientError(str(e))

    async def request_with_tools(self, payload: Dict[str, Any], tool_schemas: List[Any]) -> StructuredResponse:
        client = self._client()
        model = self._resolve_model(payload)
        contents, system = self._extract_messages(payload)
        config_kwargs: Dict[str, Any] = {}
        if system:
            config_kwargs["system_instruction"] = system
        if tool_schemas:
            try:
                declarations = [s.to_gemini() for s in tool_schemas]
                config_kwargs["tools"] = [types.Tool(function_declarations=declarations)]
            except Exception as e:
                logger.warning("gemini_tool_schema_build_failed", error=str(e))
        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None
        try:
            resp = await self._loop().run_in_executor(
                None, lambda: client.models.generate_content(
                    model=model, contents=contents, config=config
                )
            )
            tool_calls: List[ToolCall] = []
            content_text = ""
            for part in (resp.candidates[0].content.parts if resp.candidates else []):
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    args = dict(fc.args) if fc.args else {}
                    tool_calls.append(ToolCall(name=fc.name, arguments=args))
                elif hasattr(part, "text") and part.text:
                    content_text += part.text
            usage = {
                "prompt_tokens": getattr(resp.usage_metadata, "prompt_token_count", 0),
                "completion_tokens": getattr(resp.usage_metadata, "candidates_token_count", 0),
                "total_tokens": getattr(resp.usage_metadata, "total_token_count", 0),
            }
            return StructuredResponse(content=content_text, tool_calls=tool_calls,
                                      usage=usage, model=model)
        except errors.ClientError as e:
            code = getattr(e, "code", 500)
            if code == 401: raise ProviderAuthError(str(e))
            if code == 429: raise ProviderQuotaError(str(e))
            raise ProviderTransientError(str(e))
        except Exception as e:
            raise ProviderTransientError(str(e))

    async def stream(self, payload: Dict[str, Any]) -> AsyncGenerator[str, None]:
        client = self._client()
        model = self._resolve_model(payload)
        contents, system = self._extract_messages(payload)
        config = types.GenerateContentConfig(system_instruction=system) if system else None
        try:
            stream_gen = await self._loop().run_in_executor(
                None, lambda: client.models.generate_content_stream(
                    model=model, contents=contents, config=config
                )
            )
            for chunk in stream_gen:
                if chunk.text: yield chunk.text
        except errors.ClientError as e:
            code = getattr(e, "code", 500)
            if code == 401: raise ProviderAuthError(str(e))
            if code == 429: raise ProviderQuotaError(str(e))
            raise ProviderTransientError(str(e))

    async def test_key(self) -> bool:
        client = self._client()
        try:
            await self._loop().run_in_executor(None, client.models.list)
            return True
        except errors.ClientError as e:
            code = getattr(e, "code", 500)
            if code in (401, 403): raise ProviderAuthError(str(e))
            if code == 429: return True
            raise ProviderTransientError(str(e))
