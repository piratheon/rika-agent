"""Per-session token and cost tracking."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict

_COSTS: Dict[str, Dict[str, float]] = {
    "anthropic/claude-opus-4-5":         {"in": 15.0,  "out": 75.0},
    "anthropic/claude-opus-4-6":         {"in": 15.0,  "out": 75.0},
    "anthropic/claude-sonnet-4-6":       {"in": 3.0,   "out": 15.0},
    "anthropic/claude-haiku-4-5":        {"in": 0.80,  "out": 4.0},
    "google/gemini-2.5-pro":             {"in": 1.25,  "out": 5.0},
    "google/gemini-2.0-flash":           {"in": 0.10,  "out": 0.40},
    "llama-3.3-70b-versatile":           {"in": 0.59,  "out": 0.79},
    "llama-3.1-8b-instant":              {"in": 0.05,  "out": 0.08},
    "gemini-2.0-flash":                  {"in": 0.10,  "out": 0.40},
    "gemini-2.5-pro":                    {"in": 1.25,  "out": 5.0},
    "meta/llama-3.1-70b-instruct":       {"in": 0.97,  "out": 0.97},
    "z-ai/glm-4-32b":                    {"in": 0.10,  "out": 0.10},
}


@dataclass
class ModelUsage:
    provider: str; model: str
    input_tokens: int = 0; output_tokens: int = 0; calls: int = 0

    @property
    def estimated_cost_usd(self) -> float:
        r = _COSTS.get(self.model) or _COSTS.get(self.model.split("/")[-1])
        if not r: return 0.0
        return self.input_tokens / 1e6 * r["in"] + self.output_tokens / 1e6 * r["out"]


@dataclass
class SessionUsage:
    session_id: str
    _bd: Dict[str, ModelUsage] = field(default_factory=dict)

    def record(self, provider: str, model: str, input_tokens: int, output_tokens: int) -> None:
        k = f"{provider}:{model}"
        if k not in self._bd:
            self._bd[k] = ModelUsage(provider=provider, model=model)
        m = self._bd[k]
        m.input_tokens += input_tokens; m.output_tokens += output_tokens; m.calls += 1

    @property
    def total_tokens(self) -> int:
        return sum(m.input_tokens + m.output_tokens for m in self._bd.values())

    @property
    def total_calls(self) -> int:
        return sum(m.calls for m in self._bd.values())

    @property
    def estimated_cost_usd(self) -> float:
        return sum(m.estimated_cost_usd for m in self._bd.values())

    def summary(self) -> str:
        if not self._bd: return "No LLM calls recorded."
        lines = [f"{self.total_calls} call(s) | {self.total_tokens:,} tokens | ~${self.estimated_cost_usd:.4f}"]
        for m in sorted(self._bd.values(), key=lambda x: -x.input_tokens - x.output_tokens):
            cost = f" ~${m.estimated_cost_usd:.4f}" if m.estimated_cost_usd else ""
            lines.append(f"  {m.provider}/{m.model}: {m.input_tokens:,}in+{m.output_tokens:,}out ({m.calls}x){cost}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"session_id": self.session_id, "breakdown": {
            k: {"provider": v.provider, "model": v.model,
                "input_tokens": v.input_tokens, "output_tokens": v.output_tokens, "calls": v.calls}
            for k, v in self._bd.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "SessionUsage":
        obj = cls(session_id=d.get("session_id", ""))
        for k, v in d.get("breakdown", {}).items():
            obj._bd[k] = ModelUsage(provider=v["provider"], model=v["model"],
                input_tokens=v.get("input_tokens",0), output_tokens=v.get("output_tokens",0),
                calls=v.get("calls",0))
        return obj
