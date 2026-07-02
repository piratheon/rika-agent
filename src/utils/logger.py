"""Logger — structured logging to ~/.rika/logs/.

Log files rotate per-process-start:
  ~/.rika/logs/rk-YYYYMMDD-HHMMSS-mmm.log

Falls back to ./logs/ if ~/.rika is not yet initialised (e.g. unit tests).
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path


def _log_path() -> str:
    now = datetime.now()
    ms  = now.strftime("%f")[:3]
    fname = now.strftime(f"rk-%Y%m%d-%H%M%S-{ms}.log")
    rika_logs = Path.home() / ".rika" / "logs"
    try:
        rika_logs.mkdir(parents=True, exist_ok=True)
        return str(rika_logs / fname)
    except OSError:
        fallback = Path("logs")
        fallback.mkdir(parents=True, exist_ok=True)
        return str(fallback / fname)


LOG_PATH  = _log_path()
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

# stdlib root logger
_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
_handler.setFormatter(
    logging.Formatter(
        "[%(asctime)s] [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
)
_root = logging.getLogger()
_root.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
if not any(
    isinstance(h, logging.FileHandler)
    and getattr(h, "baseFilename", None) == os.path.abspath(LOG_PATH)
    for h in _root.handlers
):
    _root.addHandler(_handler)

# Silence noisy third-party loggers
for _noisy in ("httpx", "httpcore", "telegram", "discord", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# structlog
import structlog

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)


def get_logger(name: str | None = None):
    return structlog.get_logger(name)


logger = get_logger("app")
