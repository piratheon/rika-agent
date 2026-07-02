"""Complexity classifier — 3-way output: simple | complex_single | complex_multi.

simple        → direct LLM reply, no orchestration
complex_single → full orchestration, no planning phase
complex_multi  → full orchestration WITH planning phase (TaskRouter activated)

Detection order (no LLM call for tiers 1 and 2):
  1. Obvious-simple heuristics (greeting, short acknowledgement)
  2. Multi-step heuristics: sequencing words + distinct-action-verb count
     → complex_multi if score >= MULTI_THRESHOLD
  3. Single-complex heuristics: explicit tool/domain keywords, length
     → complex_single
  4. Ambiguous: cheap LLM triage call → SIMPLE | COMPLEX_SINGLE | COMPLEX_MULTI
"""
from __future__ import annotations

import re
from typing import Literal

from src.utils.logger import logger

ComplexityResult = Literal["simple", "complex_single", "complex_multi"]

# ---------------------------------------------------------------------------
# Sequencing signals (language-agnostic list — EN + common AR/FR equivalents)
# ---------------------------------------------------------------------------
_SEQUENCING = re.compile(
    r"\b("
    r"then|after that|next|afterwards|subsequently|following that|once done|"
    r"when finished|finally|lastly|also|additionally|and then|step \d|"
    r"ثم|بعد ذلك|بعدها|كذلك|أيضا|في النهاية|"
    r"ensuite|puis|après ça|finalement"
    r")\b",
    re.IGNORECASE,
)

# Action-verb clusters — each item in a distinct semantic cluster.
# Counting across clusters avoids double-counting synonyms.
_ACTION_CLUSTERS = [
    re.compile(r"\b(download|télécharger|تحميل|fetch|pull|get)\b",  re.I),
    re.compile(r"\b(summarize|summarise|resume|résumer|تلخيص|recap)\b", re.I),
    re.compile(r"\b(send|email|notify|message|إرسال|envoyer)\b",    re.I),
    re.compile(r"\b(run|execute|launch|تشغيل|exécuter|compile)\b",   re.I),
    re.compile(r"\b(write|create|generate|build|كتابة|créer)\b",     re.I),
    re.compile(r"\b(search|find|lookup|بحث|chercher|look up)\b",     re.I),
    re.compile(r"\b(analyse|analyze|inspect|تحليل|analyser)\b",      re.I),
    re.compile(r"\b(convert|transform|تحويل|convertir|format)\b",    re.I),
    re.compile(r"\b(upload|push|deploy|نشر|déployer|publish)\b",     re.I),
    re.compile(r"\b(delete|remove|clean|حذف|supprimer)\b",           re.I),
    re.compile(r"\b(translate|ترجمة|traduire|localize)\b",           re.I),
    re.compile(r"\b(install|setup|configure|تثبيت|installer)\b",     re.I),
]

_MULTI_SEQ_THRESHOLD    = 1   # at least 1 sequencing word
_MULTI_CLUSTER_THRESHOLD = 2  # at least 2 distinct action clusters

# ---------------------------------------------------------------------------
# Explicit complex keywords (single-tool or domain triggers)
# ---------------------------------------------------------------------------
_COMPLEX_KEYWORDS = [
    # web / network
    "search", "find", "fetch", "curl ", "wikipedia", "browse", "lookup",
    "scrape", "download ", "latest news", "current", "today", "weather",
    "stock", "news",
    # code / system
    "run ", "execute", "shell", "install ", "git ", "docker ", "systemctl",
    "grep ", "write a script", "write a program", "create a file", "build",
    "compile", "deploy", "python", "bash", "script", "code",
    # agent / memory
    "memory", "remember ", "delegate", "save", "skill", "workspace",
    "uptime", "disk usage", "monitor", "check ", "analyze", "research",
    "calculate",
    # tool-testing phrases
    "test", "try", "demo", "show me", "use the", "use your",
    "tool", "tools", "capability", "capabilities",
    "what can you", "can you", "try to", "attempt",
    # file ops
    "read file", "write file", "list file", "create", "modify",
]

# Obvious-simple patterns
_SIMPLE_RE = re.compile(
    r"^("
    r"hi|hello|hey|yo|sup|greetings|good morning|good evening|good night"
    r"|what's up|whats up|thanks|thank you|thx|ty|ok|okay|yes|no|sure"
    r"|np|nice|cool|great|perfect|got it|understood"
    r"|who are you|what are you|what can you do|help me|what's your name"
    r")[\s!?.]*$",
    re.IGNORECASE,
)


def classify_complexity_sync(text: str) -> ComplexityResult:
    """Heuristic-only classification — no LLM call.

    Returns complex_multi, complex_single, or simple.
    Used as the fast path; async wrapper below adds LLM triage for ambiguous cases.
    """
    t = text.strip()
    tl = t.lower()

    # Tier 1 — obvious simple
    if _SIMPLE_RE.match(t) and len(t) < 60:
        return "simple"

    # Tier 2 — multi-step detection
    seq_hits     = len(_SEQUENCING.findall(tl))
    cluster_hits = sum(1 for pat in _ACTION_CLUSTERS if pat.search(tl))
    if seq_hits >= _MULTI_SEQ_THRESHOLD and cluster_hits >= _MULTI_CLUSTER_THRESHOLD:
        logger.debug("complexity_multi", seq=seq_hits, clusters=cluster_hits, text=t[:60])
        return "complex_multi"

    # Tier 3 — single-complex keywords / length
    if len(text) > 80 or any(kw in tl for kw in _COMPLEX_KEYWORDS):
        return "complex_single"

    return "simple"


async def classify_complexity(
    text: str,
    cfg,
    pool,
    user_id: int,
) -> ComplexityResult:
    """Full classifier: heuristics first, LLM triage for ambiguous cases.

    Returns: 'simple' | 'complex_single' | 'complex_multi'
    """
    fast = classify_complexity_sync(text)
    if fast != "simple":
        # Fast path already determined complex — trust it
        return fast

    # Ambiguous (passed simple heuristics but not obviously simple) →
    # cheap LLM triage only if text is non-trivial length
    if len(text) < 30:
        return "simple"

    logger.debug("complexity_ambiguous_llm_triage", text=text[:60])
    try:
        payload = {
            "model": cfg.default_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Classify the user message into ONE of three categories. "
                        "Reply with ONLY the category name, no other text.\n\n"
                        "SIMPLE — casual chat, greetings, factual questions from memory\n"
                        "COMPLEX_SINGLE — needs web search, code, file ops, or "
                        "a single tool chain\n"
                        "COMPLEX_MULTI — requires multiple distinct tasks in sequence "
                        "(e.g. download THEN summarise THEN send; research THEN "
                        "write THEN deploy)"
                    ),
                },
                {"role": "user", "content": text[:300]},
            ],
        }
        for p in (cfg.default_provider_priority or ["groq", "openrouter", "gemini"]):
            try:
                resp   = await pool.request_with_key(user_id, p, payload)
                answer = (resp.get("output") or "").strip().upper()
                if "COMPLEX_MULTI" in answer:
                    logger.debug("complexity_llm_multi", provider=p)
                    return "complex_multi"
                if "COMPLEX" in answer:
                    logger.debug("complexity_llm_single", provider=p)
                    return "complex_single"
                logger.debug("complexity_llm_simple", provider=p)
                return "simple"
            except Exception as exc:
                logger.debug("complexity_llm_failed", provider=p, error=str(exc))
                continue
    except Exception as exc:
        logger.debug("complexity_llm_error", error=str(exc))

    # Default: if text is non-trivial and we couldn't triage, treat as single-complex
    return "complex_single" if len(text) > 60 else "simple"
