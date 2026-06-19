import re
from typing import Dict

# Accept formats like: openrouter:"sk-..." groq:gsk_... google=AIza... etc.
# Narrowed down to avoid catching URLs like https://...
RE = re.compile(r"(?P<provider>gemini|google|openrouter|groq|anthropic|openai|nvidia|vercel|ollama|g4f)\s*[:=]\s*(?:\"(?P<quoted>[^\"]+)\"|(?P<bare>[A-Za-z0-9_\-]{10,}))", re.IGNORECASE)


def parse_keys(text: str) -> Dict[str, str]:
    """
    Parses provider:key pairs from text. 
    Only matches known providers and keys that look like typical API keys (at least 10 chars, alphanumeric).
    """
    matches = RE.finditer(text)
    out = {}
    for m in matches:
        provider = m.group("provider").lower()
        val = m.group("quoted") or m.group("bare")
        # Secondary check: ensure the value doesn't start with // (like in https://)
        if val and not val.startswith("//"):
            out[provider] = val
    return out
