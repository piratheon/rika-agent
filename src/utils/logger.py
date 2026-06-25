import logging
import os
import structlog
from datetime import datetime


def _ensure_log_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _get_log_path():
    # Format: logs/rk-[date]-[time:(hour-minute-second-ms)].log
    now = datetime.now()
    # ms is %f, we'll take first 3 digits for ms
    ms = now.strftime("%f")[:3]
    filename = now.strftime(f"rk-%Y%m%d-%H%M%S-{ms}.log")
    path = os.path.join("logs", filename)
    _ensure_log_dir(path)
    return path


LOG_PATH = _get_log_path()
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

# Configure stdlib logging
handler = logging.FileHandler(LOG_PATH)
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
root = logging.getLogger()
root.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

# Silence noisy dependencies
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)

if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', None) == os.path.abspath(LOG_PATH) for h in root.handlers):
    root.addHandler(handler)

# Configure structlog
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
