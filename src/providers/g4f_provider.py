"""G4F (gpt4free) provider — free access to multiple AI models.

Uses the `g4f` library which routes requests through various free providers
(Bing, You.com, Liaobots, etc.) without requiring API keys.

WARNING: G4F providers are unstable by nature — they rely on reverse-engineered
APIs and can break without notice. Use as a last-resort fallback, not primary.

Default configuration:
  Provider: DeepInfra
  Model: MiniMaxAI/MiniMax-M2.5
"""
from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator, Dict, List

from src.providers.base_provider import (
    BaseProvider,
    ProviderAuthError,
    ProviderQuotaError,
    ProviderTransientError,
)
from src.utils.logger import logger

# Default G4F configuration
DEFAULT_G4F_PROVIDER = "DeepInfra"
DEFAULT_G4F_MODEL = "MiniMaxAI/MiniMax-M2.5"

# Fallback chain if the requested model isn't available via g4f
_FALLBACK_MODELS = [
    DEFAULT_G4F_MODEL,
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4",
    "llama-3.1-70b",
    "gemini-pro",
    "claude-3-haiku",
]


class G4FProvider(BaseProvider):
    """Free LLM access via gpt4free. No API key required."""

    def __init__(self, api_key: str = "", provider_name: str = "g4f") -> None:
        super().__init__(api_key or "g4f", provider_name)
        self._g4f_available: bool = self._check_g4f()

    def _check_g4f(self) -> bool:
        try:
            import g4f  # noqa: F401
            return True
        except ImportError:
            return False

    def _require_g4f(self):
        if not self._g4f_available:
            raise ProviderTransientError(
                "g4f is not installed. Install it with: pip install g4f"
            )

    # ------------------------------------------------------------------
    # Internal: sync wrapper (g4f is sync)
    # ------------------------------------------------------------------

    def _sync_request(self, model: str, messages: List[Dict]) -> str:
        """Synchronous g4f call — runs in executor to avoid blocking."""
        try:
            from g4f.client import Client
            client = Client()
            # Use DeepInfra provider with MiniMax-M2.5 model
            response = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            raise RuntimeError(str(exc))

    def _try_models(self, messages: List[Dict], requested: str) -> str:
        """Try requested model, then fallback chain."""
        chain = [requested] + [m for m in _FALLBACK_MODELS if m != requested]
        last_err = ""
        for model in chain:
            try:
                result = self._sync_request(model, messages)
                if result:
                    logger.debug("g4f_model_success", model=model)
                    return result
            except Exception as exc:
                last_err = str(exc)
                logger.debug("g4f_model_failed", model=model, error=last_err)
                continue
        raise RuntimeError(f"All g4f models failed. Last error: {last_err}")

    # ------------------------------------------------------------------
    # BaseProvider interface
    # ------------------------------------------------------------------

    async def request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._require_g4f()
        model = payload.get("model", "gpt-4o-mini")
        messages = payload.get("messages", [])

        loop = asyncio.get_running_loop()
        try:
            content = await loop.run_in_executor(None, self._try_models, messages, model)
            return {
                "output": content,
                "usage": {
                    "prompt_tokens": 0,   # g4f doesn't expose token counts
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
        except RuntimeError as exc:
            err = str(exc).lower()
            if any(x in err for x in ["rate limit", "429", "too many"]):
                raise ProviderQuotaError(f"G4F rate limited: {exc}")
            if any(x in err for x in ["auth", "401", "403", "blocked"]):
                raise ProviderAuthError(f"G4F auth error: {exc}")
            raise ProviderTransientError(f"G4F failed: {exc}")
        except Exception as exc:
            logger.error("g4f_request_failed", error=str(exc))
            raise ProviderTransientError(f"G4F unexpected error: {exc}")

    async def stream(self, payload: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """G4F streaming — falls back to single-shot since g4f streaming is unreliable."""
        result = await self.request(payload)
        content = result.get("output", "")
        # Yield in chunks to simulate streaming feel
        chunk_size = 80
        for i in range(0, len(content), chunk_size):
            yield content[i : i + chunk_size]
            await asyncio.sleep(0)

    async def test_key(self) -> bool:
        """Verify g4f is installed and reachable with a minimal call."""
        self._require_g4f()
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                None,
                self._sync_request,
                "gpt-4o-mini",
                [{"role": "user", "content": "hi"}],
            )
            return bool(result)
        except Exception as exc:
            raise ProviderTransientError(f"G4F test failed: {exc}")

    @staticmethod
    def list_providers() -> List[str]:
        """Return list of available g4f provider names."""
        try:
            import g4f
            return [p.__name__ for p in g4f.Provider.__providers__]
        except Exception:
            return []
