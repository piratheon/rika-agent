"""TaskRouter — per-step model selection based on complexity tier.

Maps low/mid/complex subtask tiers to configured provider+model pairs.
Fallback chain: if the configured tier's provider has no working key,
try the next tier down. Raises only when NO tier has a working key
(caller should surface this as an explicit no-key error, not a retry).

Config example (config.json):
    "model_tiers": {
        "complex": {"provider": "openrouter", "model": "anthropic/claude-opus-4.6"},
        "mid":     {"provider": "groq",       "model": "llama-3.3-70b-versatile"},
        "low":     {"provider": "groq",       "model": "llama-3.1-8b-instant"}
    },
    "tier_fallback_chain": ["complex", "mid", "low"]

Platform-agnostic — no Telegram or Discord imports.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from src.utils.logger import logger

VALID_TIERS = ("complex", "mid", "low")

_DEFAULT_TIERS: Dict[str, Dict[str, str]] = {
    "complex": {"provider": "openrouter", "model": "google/gemini-2.5-pro"},
    "mid":     {"provider": "groq",       "model": "llama-3.3-70b-versatile"},
    "low":     {"provider": "groq",       "model": "llama-3.1-8b-instant"},
}
_DEFAULT_FALLBACK_CHAIN: List[str] = ["complex", "mid", "low"]


class TaskRouter:
    """Resolves provider+model for a given complexity tier with fallback."""

    _instance: Optional["TaskRouter"] = None

    def __init__(self, cfg=None) -> None:
        from src.config import Config
        self._cfg = cfg or Config.get()
        self._tiers: Dict[str, Dict[str, str]] = getattr(
            self._cfg, "model_tiers", None
        ) or _DEFAULT_TIERS
        self._chain: List[str] = getattr(
            self._cfg, "tier_fallback_chain", None
        ) or _DEFAULT_FALLBACK_CHAIN

    @classmethod
    def get(cls, cfg=None) -> "TaskRouter":
        if cls._instance is None or cfg is not None:
            cls._instance = cls(cfg)
        return cls._instance

    @classmethod
    def invalidate(cls) -> None:
        cls._instance = None

    def resolve(self, tier: str, pool) -> Tuple[str, str]:
        """Return (provider, model) for the given tier, following the fallback chain.

        Falls back to the next tier only when the tier's configured provider has
        no healthy key. Raises RuntimeError when no tier has a key — caller should
        surface this as an actionable no-key error, not trigger an auto-retry.
        """
        tier = tier if tier in VALID_TIERS else "mid"
        start_idx = self._chain.index(tier) if tier in self._chain else 0

        for candidate in self._chain[start_idx:]:
            t_cfg = self._tiers.get(candidate)
            if not t_cfg:
                continue
            provider = t_cfg.get("provider", "groq")
            model    = t_cfg.get("model", "")
            try:
                has_key = pool.has_healthy_key(provider)
            except Exception:
                has_key = True  # assume yes if pool can't answer; will fail at call time
            if has_key:
                if candidate != tier:
                    logger.warning(
                        "task_router_fallback",
                        requested=tier, using=candidate, provider=provider,
                    )
                return provider, model

        raise RuntimeError(
            f"No working API key found for any tier in the fallback chain "
            f"({', '.join(self._chain)}). Add a key with /addkey."
        )

    def payload_for_tier(
        self, tier: str, pool, messages: list, extra: dict | None = None
    ) -> Tuple[str, Dict]:
        """Build (provider, payload) ready for pool.request_with_key_structured()."""
        provider, model = self.resolve(tier, pool)
        payload: Dict = {"model": model, "messages": messages}
        if extra:
            payload.update(extra)
        return provider, payload

    def tier_for_step(self, step: Dict) -> str:
        """Extract and validate tier from a plan step dict."""
        raw = (step.get("tier") or "mid").lower().strip()
        return raw if raw in VALID_TIERS else "mid"
