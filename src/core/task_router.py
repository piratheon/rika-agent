"""TaskRouter — per-step model selection with per-tier model chains.

Supports three config formats:

  Single model for everything ("all" key):
    model_tiers = {"all": {"provider": "groq", "model": "llama-3.3-70b-versatile"}}

  Per-tier single model (dict per tier):
    model_tiers = {
      "complex": {"provider": "openrouter", "model": "anthropic/claude-opus-4-5"},
      "mid":     {"provider": "groq",       "model": "llama-3.3-70b-versatile"},
      "low":     {"provider": "groq",       "model": "llama-3.1-8b-instant"},
    }

  Per-tier model chain (list per tier — tried in order on failure):
    model_tiers = {
      "complex": [
        {"provider": "openrouter", "model": "anthropic/claude-opus-4-5"},
        {"provider": "openrouter", "model": "z-ai/glm-4-32b"},
      ],
      "mid": {"provider": "groq", "model": "llama-3.3-70b-versatile"},
      "low": {"provider": "groq", "model": "llama-3.1-8b-instant"},
    }

Fallback behaviour:
  1. Try each model in the requested tier's list (in order).
  2. On all-fail, move to the next tier in tier_fallback_chain.
  3. On all tiers exhausted, the orchestrator's countdown loop retries
     from the top — naturally cycling back to the first model.

Platform-agnostic — no Telegram or Discord imports.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from src.utils.logger import logger

VALID_TIERS = ("complex", "mid", "low")

_DEFAULT_TIERS: Dict = {
    "complex": [{"provider": "openrouter", "model": "google/gemini-2.5-pro"}],
    "mid":     [{"provider": "groq",       "model": "llama-3.3-70b-versatile"}],
    "low":     [{"provider": "groq",       "model": "llama-3.1-8b-instant"}],
}
_DEFAULT_FALLBACK_CHAIN: List[str] = ["complex", "mid", "low"]


def _normalise_tier_entry(raw) -> List[Dict[str, str]]:
    """Coerce a tier config value to a list of {provider, model} dicts."""
    if raw is None:
        return []
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, list):
        return [m for m in raw if isinstance(m, dict) and "provider" in m]
    return []


class TaskRouter:
    """Resolves an ordered (provider, model) attempt list for a complexity tier."""

    _instance: Optional["TaskRouter"] = None

    def __init__(self, cfg=None) -> None:
        from src.config import Config
        self._cfg   = cfg or Config.get()
        raw_tiers   = getattr(self._cfg, "model_tiers", None) or _DEFAULT_TIERS
        self._chain: List[str] = (
            getattr(self._cfg, "tier_fallback_chain", None) or _DEFAULT_FALLBACK_CHAIN
        )
        # Normalise every tier entry to a list
        self._tiers: Dict[str, List[Dict[str, str]]] = {}
        for k, v in raw_tiers.items():
            self._tiers[k] = _normalise_tier_entry(v)

    @classmethod
    def get(cls, cfg=None) -> "TaskRouter":
        if cls._instance is None or cfg is not None:
            cls._instance = cls(cfg)
        return cls._instance

    @classmethod
    def invalidate(cls) -> None:
        cls._instance = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_attempt_chain(self, start_tier: str) -> List[Tuple[str, str, str]]:
        """Return ordered [(tier, provider, model), ...] to try for start_tier.

        Single-model mode: "all" key → same model for every tier.
        Chain mode: models within a tier tried in array order before falling
        through to the next tier in tier_fallback_chain.

        If the chain is empty (no model configured anywhere), returns a
        safe-fallback entry pointing at Groq so the orchestrator countdown
        loop still has something to retry against.
        """
        # Single-model mode
        all_models = self._tiers.get("all")
        if all_models:
            m = all_models[0]
            p, mo = m.get("provider", "groq"), m.get("model", "")
            logger.debug("task_router_single_model_mode", provider=p, model=mo)
            return [("all", p, mo)]

        start_idx = (
            self._chain.index(start_tier)
            if start_tier in self._chain
            else 0
        )

        chain: List[Tuple[str, str, str]] = []
        for tier in self._chain[start_idx:]:
            for entry in self._tiers.get(tier, []):
                p  = entry.get("provider", "groq")
                mo = entry.get("model", "")
                if p and mo:
                    chain.append((tier, p, mo))

        if not chain:
            # No model configured for start_tier or later — try all tiers
            for tier in self._chain:
                for entry in self._tiers.get(tier, []):
                    p  = entry.get("provider", "groq")
                    mo = entry.get("model", "")
                    if p and mo:
                        chain.append((tier, p, mo))

        if not chain:
            logger.error("task_router_no_models_configured")
            # Safe-fallback: groq with a known model so the countdown loop
            # fires a real error ("no key") rather than a silent empty queue.
            chain = [("mid", "groq", "llama-3.3-70b-versatile")]

        return chain

    def models_for_tier(self, tier: str) -> List[Dict[str, str]]:
        """Return the raw model list for a single tier (used by planner/replanner)."""
        all_models = self._tiers.get("all")
        if all_models:
            return list(all_models)
        return list(self._tiers.get(tier, []))

    def first_model_for_tier(self, tier: str) -> Tuple[str, str]:
        """Return (provider, model) for the first model in a tier."""
        models = self.models_for_tier(tier)
        if models:
            return models[0].get("provider", "groq"), models[0].get("model", "")
        # Fallback through chain
        for t in self._chain:
            models = self.models_for_tier(t)
            if models:
                return models[0].get("provider", "groq"), models[0].get("model", "")
        return "groq", "llama-3.3-70b-versatile"

    def payload_for_tier(
        self, tier: str, pool, messages: list, extra: dict | None = None
    ) -> Tuple[str, Dict]:
        """Build (provider, payload) for the first model in a tier.

        Used by the planner/replanner which always uses mid-tier.
        Full chain iteration is done by Orchestrator._provider_request.
        """
        provider, model = self.first_model_for_tier(tier)
        payload: Dict = {"model": model, "messages": messages}
        if extra:
            payload.update(extra)
        return provider, payload

    def tier_for_step(self, step: Dict) -> str:
        """Extract and validate tier label from a plan step dict."""
        raw = (step.get("tier") or "mid").lower().strip()
        return raw if raw in VALID_TIERS else "mid"

    def describe(self) -> str:
        """Human-readable summary for /status."""
        if "all" in self._tiers:
            m = self._tiers["all"][0] if self._tiers["all"] else {}
            return f"single-model: {m.get('provider')}/{m.get('model')}"
        lines = []
        for tier in self._chain:
            models = self._tiers.get(tier, [])
            if models:
                chain_str = " → ".join(
                    f"{m.get('provider')}/{m.get('model')}" for m in models
                )
                lines.append(f"{tier}: {chain_str}")
        return "; ".join(lines) if lines else "default"
