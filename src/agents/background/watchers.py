"""Background watchers — pure Python anomaly detectors.

Every watcher runs in a tight asyncio loop. They NEVER call an LLM.
When a threshold is breached or a state change is detected, they return
a WakeSignal. The BackgroundAgentManager queues that signal for the
AI-powered WakeProcessor to handle.
"""
from __future__ import annotations

import asyncio
import os
import re
import socket
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

from src.agents.agent_models import WakeSignal
from src.utils.logger import logger


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class WatcherBase:
    """Abstract watcher. Subclasses implement `check()` only."""

    async def check(self) -> Optional[WakeSignal]:
        """Return a WakeSignal if something needs attention; None otherwise."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# SystemWatcher
# ---------------------------------------------------------------------------

class SystemWatcher(WatcherBase):
    """Monitor CPU load, memory, and disk via /proc (no psutil required)."""

    def __init__(
        self,
        cpu_threshold: float = 85.0,
        mem_threshold: float = 90.0,
        disk_threshold: float = 90.0,
        load_threshold: float = 4.0,
    ):
        self.cpu_threshold = cpu_threshold
        self.mem_threshold = mem_threshold
        self.disk_threshold = disk_threshold
        self.load_threshold = load_threshold
        self._prev_cpu: Optional[Dict[str, int]] = None

    # --- helpers ---

    def _read_proc(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}

        # Load average
        try:
            with open("/proc/loadavg") as f:
                parts = f.read().split()
            result["load_1"] = float(parts[0])
            result["load_5"] = float(parts[1])
            result["load_15"] = float(parts[2])
        except Exception:
            result["load_1"] = 0.0

        # Memory
        try:
            with open("/proc/meminfo") as f:
                mem: Dict[str, int] = {}
                for line in f:
                    k, _, v = line.partition(":")
                    mem[k.strip()] = int(v.strip().split()[0])
            total = mem.get("MemTotal", 1)
            available = mem.get("MemAvailable", total)
            used = total - available
            result["mem_percent"] = round((used / total) * 100, 1)
            result["mem_used_mb"] = round(used / 1024, 1)
            result["mem_total_mb"] = round(total / 1024, 1)
        except Exception:
            result["mem_percent"] = 0.0

        # Disk (root)
        try:
            import shutil
            u = shutil.disk_usage("/")
            result["disk_percent"] = round((u.used / u.total) * 100, 1)
            result["disk_used_gb"] = round(u.used / 1e9, 2)
            result["disk_total_gb"] = round(u.total / 1e9, 2)
        except Exception:
            result["disk_percent"] = 0.0

        return result

    async def check(self) -> Optional[WakeSignal]:
        loop = asyncio.get_running_loop()
        try:
            m = await loop.run_in_executor(None, self._read_proc)
        except Exception as exc:
            logger.error("system_watcher_read_failed", error=str(exc))
            return None

        breaches: List[str] = []
        severity: str = "warning"

        if m.get("load_1", 0) > self.load_threshold:
            breaches.append(f"load_1min={m['load_1']} (limit {self.load_threshold})")

        if m.get("mem_percent", 0) > self.mem_threshold:
            breaches.append(f"memory={m['mem_percent']}% (limit {self.mem_threshold}%)")

        disk = m.get("disk_percent", 0)
        if disk > self.disk_threshold:
            breaches.append(f"disk={disk}% (limit {self.disk_threshold}%)")
            if disk > 95:
                severity = "critical"

        if not breaches:
            return None

        return WakeSignal(
            event_type="threshold_breach",
            severity=severity,
            raw_data={"breaches": breaches, "metrics": m},
            needs_ai_analysis=True,
        )


# ---------------------------------------------------------------------------
# ProcessWatcher
# ---------------------------------------------------------------------------

class ProcessWatcher(WatcherBase):
    """Detect when a named process stops or (re)starts."""

    def __init__(self, process_name: str):
        self.process_name = process_name.lower()
        self._was_running: Optional[bool] = None

    def _is_running(self) -> bool:
        try:
            for pid in os.listdir("/proc"):
                if not pid.isdigit():
                    continue
                try:
                    comm_path = f"/proc/{pid}/comm"
                    if os.path.exists(comm_path):
                        with open(comm_path) as f:
                            if self.process_name in f.read().strip().lower():
                                return True
                    with open(f"/proc/{pid}/cmdline") as f:
                        if self.process_name in f.read().replace("\0", " ").lower():
                            return True
                except (PermissionError, FileNotFoundError, OSError):
                    continue
        except Exception:
            pass
        return False

    async def check(self) -> Optional[WakeSignal]:
        loop = asyncio.get_running_loop()
        running = await loop.run_in_executor(None, self._is_running)
        was = self._was_running
        self._was_running = running

        if was is None:          # first poll — establish baseline silently
            return None
        if was and not running:
            return WakeSignal(
                event_type="process_down",
                severity="critical",
                raw_data={
                    "process": self.process_name,
                    "status": "stopped",
                    "message": f"Process '{self.process_name}' stopped.",
                },
            )
        if not was and running:
            return WakeSignal(
                event_type="process_recovered",
                severity="info",
                raw_data={
                    "process": self.process_name,
                    "status": "recovered",
                    "message": f"Process '{self.process_name}' is running again.",
                },
                needs_ai_analysis=False,
            )
        return None


# ---------------------------------------------------------------------------
# URLWatcher
# ---------------------------------------------------------------------------

class URLWatcher(WatcherBase):
    """HTTP health check. Fires on state change (up→down or down→up)."""

    def __init__(self, url: str, expected_status: int = 200, timeout: int = 10):
        self.url = url
        self.expected_status = expected_status
        self.timeout = timeout
        self._last_up: Optional[bool] = None

    async def check(self) -> Optional[WakeSignal]:
        status_code = 0
        error_msg = ""
        try:
            async with httpx.AsyncClient(timeout=float(self.timeout), follow_redirects=True) as client:
                r = await client.get(self.url)
                status_code = r.status_code
                is_up = r.status_code == self.expected_status
        except Exception as exc:
            is_up = False
            error_msg = str(exc)

        was = self._last_up
        self._last_up = is_up

        if was is None:
            return None
        if was and not is_up:
            return WakeSignal(
                event_type="url_unreachable",
                severity="critical",
                raw_data={
                    "url": self.url,
                    "expected_status": self.expected_status,
                    "actual_status": status_code,
                    "error": error_msg,
                },
            )
        if not was and is_up:
            return WakeSignal(
                event_type="url_recovered",
                severity="info",
                raw_data={"url": self.url, "status": status_code, "message": "URL reachable again."},
                needs_ai_analysis=False,
            )
        return None


# ---------------------------------------------------------------------------
# PortWatcher
# ---------------------------------------------------------------------------

class PortWatcher(WatcherBase):
    def __init__(self, host: str = "localhost", port: int = 80):
        self.host = host
        self.port = port
        self._last_open: Optional[bool] = None

    def _check(self) -> bool:
        try:
            with socket.create_connection((self.host, self.port), timeout=5):
                return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            return False

    async def check(self) -> Optional[WakeSignal]:
        loop = asyncio.get_running_loop()
        is_open = await loop.run_in_executor(None, self._check)
        was = self._last_open
        self._last_open = is_open

        if was is None:
            return None
        if was and not is_open:
            return WakeSignal(
                event_type="port_closed",
                severity="critical",
                raw_data={"host": self.host, "port": self.port, "status": "closed"},
            )
        if not was and is_open:
            return WakeSignal(
                event_type="port_recovered",
                severity="info",
                raw_data={"host": self.host, "port": self.port, "status": "open"},
                needs_ai_analysis=False,
            )
        return None


# ---------------------------------------------------------------------------
# LogPatternWatcher
# ---------------------------------------------------------------------------

class LogPatternWatcher(WatcherBase):
    """Tail a log file and fire when a regex pattern is matched.
    Tracks file position across calls. Has a per-fire cooldown to avoid spam.
    """

    def __init__(self, file_path: str, pattern: str, cooldown_seconds: int = 300):
        self.file_path = file_path
        self.pattern = pattern
        self.cooldown = cooldown_seconds
        self._pos: int = 0
        self._last_fired: float = 0.0
        try:
            self._re = re.compile(pattern, re.IGNORECASE)
        except re.error:
            self._re = re.compile(re.escape(pattern), re.IGNORECASE)

        # Seek to end on first load so we don't flood alerts with old lines
        try:
            self._pos = os.path.getsize(file_path)
        except OSError:
            self._pos = 0

    def _scan(self) -> List[str]:
        matches: List[str] = []
        if not os.path.exists(self.file_path):
            return matches
        try:
            with open(self.file_path, errors="replace") as f:
                f.seek(self._pos)
                content = f.read()
                self._pos = f.tell()
            for line in content.splitlines():
                if self._re.search(line):
                    matches.append(line.strip())
        except Exception as exc:
            logger.error("log_watcher_scan_error", path=self.file_path, error=str(exc))
        return matches[-15:]  # cap to prevent huge messages

    async def check(self) -> Optional[WakeSignal]:
        now = time.monotonic()
        loop = asyncio.get_running_loop()
        matches = await loop.run_in_executor(None, self._scan)

        if not matches:
            return None
        if now - self._last_fired < self.cooldown:
            return None  # position advanced, signal suppressed by cooldown

        self._last_fired = now
        return WakeSignal(
            event_type="pattern_match",
            severity="warning",
            raw_data={
                "file": self.file_path,
                "pattern": self.pattern,
                "sample_matches": matches,
                "match_count": len(matches),
            },
        )


# ---------------------------------------------------------------------------
# CronWatcher — always fires; used for scheduled AI tasks
# ---------------------------------------------------------------------------

class CronWatcher(WatcherBase):
    """Fires on every interval cycle. Used for scheduled AI work."""

    def __init__(self, task_description: str):
        self.task_description = task_description

    async def check(self) -> Optional[WakeSignal]:
        return WakeSignal(
            event_type="scheduled_task",
            severity="info",
            raw_data={
                "task": self.task_description,
                "triggered_at": _utcnow(),
            },
            needs_ai_analysis=True,
        )
