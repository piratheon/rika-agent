"""Complexity classifier — determines if message needs orchestration.

Extracted from app.py _classify_complexity() function.
Same logic, just wrapped in a function for reusability.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from src.config import Config
from src.utils.logger import logger


async def classify_complexity(
    text: str,
    cfg: Config,
    pool,
    user_id: int,
) -> bool:
    """Classify whether a message requires tools / orchestration.
    
    Same logic as app.py _classify_complexity() - three-tier approach:
    1. Obvious simple: short greeting/question → False immediately
    2. Obvious complex: contains explicit tool keywords → True immediately
    3. Ambiguous: single cheap LLM classification call
    
    Returns:
        True if complex (needs orchestration), False if simple (direct reply)
    """
    t = text.lower().strip()
    logger.debug("complexity_check", text=text[:50], lower=t[:50])

    # Tier 1 — definitely simple (no LLM call)
    _SIMPLE_PATTERNS = (
        r"^(hi|hello|hey|yo|sup|greetings|good morning|good evening|good night|what's up|whats up)[\s!?.]*$",
        r"^(thanks|thank you|thx|ty|ok|okay|yes|no|sure|np|nice|cool|great|perfect|got it|understood)[\s!?.]*$",
        r"^(who are you|what are you|what can you do|help me|what's your name)[\s?]*$",
    )
    for i, p in enumerate(_SIMPLE_PATTERNS):
        if re.match(p, t):
            logger.debug("complexity_simple_match", pattern=i, text=text[:50])
            return False

    # Tier 2 — definitely complex (no LLM call)
    _COMPLEX_KEYWORDS = [
        # web / network
        "search", "find", "fetch", "curl ", "wikipedia", "browse", "lookup",
        "scrape", "download ", "latest news", "what is the price",
        "who is the ceo", "current", "today", "weather", "stock", "news",
        # code / system
        "run ", "execute", "shell", "install ", "git ", "docker ",
        "systemctl", "grep ", "ls ", "pwd", "cat ", "write a script",
        "write a program", "create a file", "build", "compile", "deploy",
        "python", "bash", "script", "code",
        # agent / memory
        "memory", "remember ", "delegate", "save", "skill", "workspace",
        "uptime", "disk usage", "monitor", "check ", "analyze", "research",
        "calculate",
        # tool-testing phrases that triggered the bug
        "test", "try", "demo", "show me", "use the", "use your",
        "tool", "tools", "capability", "capabilities",
        "what can you", "can you", "try to", "attempt",
        # file operations
        "read file", "write file", "list file", "create", "modify",
    ]
    # Messages longer than 80 chars are almost always non-trivial
    if len(text) > 80:
        logger.debug("complexity_length_match", text=text[:50])
        return True
    if any(kw in t for kw in _COMPLEX_KEYWORDS):
        logger.debug("complexity_keyword_match", text=text[:50])
        return True

    # Tier 3 — ambiguous: ask the LLM
    logger.debug("complexity_ambiguous", text=text[:50])
    try:
        payload = {
            "model": cfg.default_model,
            "messages": [
                {"role": "system", "content": (
                    "Classify the user message. Reply with ONE word only: "
                    "SIMPLE or COMPLEX. "
                    "SIMPLE = casual chat, greetings, factual questions answerable from memory. "
                    "COMPLEX = needs web search, code execution, file operations, real-time data, "
                    "multi-step research, or system interaction."
                )},
                {"role": "user", "content": text[:200]},
            ],
        }
        for p in (cfg.default_provider_priority or ["gemini", "groq", "openrouter"]):
            try:
                resp = await pool.request_with_key(user_id, p, payload)
                answer = (resp.get("output") or "").strip().upper()
                result = "COMPLEX" in answer
                logger.debug("complexity_llm_result", provider=p, answer=answer, result=result)
                return result
            except Exception as exc:
                logger.debug("complexity_llm_failed", provider=p, error=str(exc))
                continue
    except Exception as exc:
        logger.debug("complexity_llm_error", error=str(exc))
    
    return len(text) > 100
