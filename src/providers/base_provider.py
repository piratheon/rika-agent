"""Base provider — abstract interface with structured function-calling support."""
from __future__ import annotations
import abc, json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

class ProviderError(Exception): pass
class ProviderAuthError(ProviderError): pass
class ProviderQuotaError(ProviderError): pass
class ProviderTransientError(ProviderError): pass

@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]
    call_id: str = ""
    def get_arg(self, key: str, default: Any = "") -> Any:
        return self.arguments.get(key, default)

@dataclass
class StructuredResponse:
    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=dict)
    model: str = ""
    @property
    def has_tool_calls(self) -> bool: return bool(self.tool_calls)
    @property
    def output(self) -> str: return self.content
    def to_legacy_dict(self) -> Dict[str, Any]:
        return {"output": self.content, "usage": self.usage}

class BaseProvider(abc.ABC):
    SUPPORTS_FUNCTION_CALLING: bool = False

    def __init__(self, api_key: str, provider_name: str) -> None:
        self.api_key = api_key
        self.provider_name = provider_name

    @abc.abstractmethod
    async def request(self, payload: Dict[str, Any]) -> Dict[str, Any]: ...
    @abc.abstractmethod
    async def stream(self, payload: Dict[str, Any]) -> AsyncIterator[str]: ...
    @abc.abstractmethod
    async def test_key(self) -> bool: ...

    async def request_with_tools(self, payload: Dict[str, Any], tool_schemas: List[Any]) -> StructuredResponse:
        """Default: ignore schemas, call request(). Override for real function calling."""
        result = await self.request(payload)
        return StructuredResponse(content=result.get("output", ""), usage=result.get("usage", {}))

    @staticmethod
    def _parse_openai_tool_calls(choices: List[Dict]) -> List[ToolCall]:
        calls: List[ToolCall] = []
        if not choices: return calls
        msg = choices[0].get("message", {})
        for tc in msg.get("tool_calls", []) or []:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            raw_args = fn.get("arguments", "{}")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                args = {"query": raw_args}
            calls.append(ToolCall(name=name, arguments=args, call_id=tc.get("id", "")))
        return calls

    @staticmethod
    def _extract_openai_content(choices: List[Dict]) -> str:
        if not choices: return ""
        return choices[0].get("message", {}).get("content") or ""
