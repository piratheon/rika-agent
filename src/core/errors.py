"""Provider error classification — type-based first, string-content fallback."""
from __future__ import annotations
import re
from enum import Enum, auto
from typing import Optional


class ErrorKind(Enum):
    FATAL           = auto()   # auth/bad-request — surface immediately
    QUOTA_EXHAUSTED = auto()   # monthly limit — pause task
    RATE_LIMITED    = auto()   # temporary 429 — retry with backoff
    NETWORK         = auto()   # connection/timeout — retry immediately
    TRANSIENT       = auto()   # 5xx — retry with backoff


_QUOTA_RE = re.compile(
    r"insufficient.{0,20}(credits?|quota|funds?|balance)"
    r"|monthly.{0,20}(limit|budget|quota)"
    r"|account.{0,20}(quota|limit|budget)"
    r"|billing.{0,20}(required|limit|exceeded)"
    r"|out.{0,10}of.{0,10}credits?"
    r"|free.{0,20}tier.{0,20}(limit|reached|exhausted)"
    r"|RESOURCE_EXHAUSTED|payment.{0,20}required|prepaid.{0,20}balance",
    re.IGNORECASE,
)
_RATE_RE = re.compile(
    r"rate.{0,10}limit|too.{0,10}many.{0,10}requests?"
    r"|requests?.{0,10}per.{0,10}(minute|second|hour)"
    r"|\brpm\b|\btpm\b|\brps\b|retry.{0,10}after|throttl",
    re.IGNORECASE,
)
_FATAL_RE = re.compile(
    r"invalid.{0,10}api.{0,10}key|api.{0,10}key.{0,10}(invalid|expired|revoked|not.{0,5}found)"
    r"|authentication.{0,20}failed|permission.{0,10}denied|access.{0,10}denied"
    r"|account.{0,10}(deactivated|suspended|disabled)"
    r"|model.{0,10}not.{0,10}found|tool_use_failed|invalid.{0,10}request|no.{0,10}such.{0,10}model",
    re.IGNORECASE,
)


def classify_error(
    exc: Optional[Exception],
    err_str: str = "",
    status_code: Optional[int] = None,
) -> ErrorKind:
    import asyncio
    if exc is not None:
        if isinstance(exc, asyncio.TimeoutError):
            return ErrorKind.NETWORK
        tn = type(exc).__name__
        if any(t in tn for t in ("ConnectError","ConnectionError","ConnectionRefusedError",
                                  "ClientConnectorError","ServerDisconnectedError","NetworkError")):
            return ErrorKind.NETWORK

    code = status_code
    if code is None and exc is not None:
        code = getattr(exc,"status_code",None) or getattr(exc,"status",None) or getattr(exc,"code",None)

    if code is not None:
        if code in (401, 403):            return ErrorKind.FATAL
        if code == 429:
            return ErrorKind.QUOTA_EXHAUSTED if _QUOTA_RE.search(err_str) else ErrorKind.RATE_LIMITED
        if code == 400:
            return ErrorKind.FATAL if _FATAL_RE.search(err_str) else ErrorKind.TRANSIENT
        if code >= 500:                   return ErrorKind.TRANSIENT

    if _QUOTA_RE.search(err_str):         return ErrorKind.QUOTA_EXHAUSTED
    if _RATE_RE.search(err_str):          return ErrorKind.RATE_LIMITED
    if _FATAL_RE.search(err_str):         return ErrorKind.FATAL
    if any(t in err_str.lower() for t in ("timeout","timed out","connection refused",
                                           "network","eof","broken pipe","reset by peer")):
        return ErrorKind.NETWORK
    return ErrorKind.TRANSIENT


def is_retryable(k: ErrorKind) -> bool:
    return k in (ErrorKind.RATE_LIMITED, ErrorKind.NETWORK, ErrorKind.TRANSIENT)

def is_fatal(k: ErrorKind) -> bool:   return k == ErrorKind.FATAL
def is_quota(k: ErrorKind) -> bool:   return k == ErrorKind.QUOTA_EXHAUSTED
