"""ScriptWatcher — executes a script file on each cycle and feeds stdout to the AI.

This is the key missing piece for fully-agentic background monitoring.

Flow:
  1. Agent writes a monitoring script to ~/.rika/shared/watchers/<id>.py (or .sh)
  2. ScriptWatcher executes it on every interval
  3. If the script exits non-zero OR outputs a line starting with "ALERT:",
     a WakeSignal is fired with the full output as context
  4. The WakeProcessor runs a ConcreteAgent turn (with tools) on the signal
     so the AI can take real action — not just describe what's wrong

The script protocol:
  - Exit 0 = everything fine, output is discarded
  - Exit non-zero = problem detected, output included in WakeSignal
  - Print "ALERT: <message>" on any line = always fires (even exit 0)
  - Print "METRIC: <key>=<value>" to record time-series data (future)

Agent-written scripts live in: ~/.rika/shared/watchers/
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.agents.agent_models import WakeSignal
from src.utils.logger import logger


class ScriptWatcher:
    """Executes an agent-authored script file and fires on non-zero exit or ALERT: output."""

    def __init__(
        self,
        script_path: str,
        working_dir: Optional[str] = None,
        cooldown_seconds: int = 120,
        alert_on_exit_nonzero: bool = True,
        alert_on_pattern: str = "ALERT:",
    ):
        self.script_path = script_path
        self.working_dir = working_dir or str(Path(script_path).parent)
        self.cooldown = cooldown_seconds
        self.alert_on_exit_nonzero = alert_on_exit_nonzero
        self.alert_pattern = alert_on_pattern.upper()
        self._last_fired: float = 0.0
        self._consecutive_failures: int = 0

    def _resolve_interpreter(self) -> List[str]:
        """Pick the right interpreter from the script extension."""
        path = Path(self.script_path)
        ext = path.suffix.lower()
        if ext == ".py":
            return [sys.executable, str(path)]
        if ext in (".sh", ".bash"):
            return ["bash", str(path)]
        # Try shebang
        try:
            with open(path) as f:
                first = f.readline().strip()
            if first.startswith("#!"):
                return first[2:].split() + [str(path)]
        except Exception:
            pass
        return ["bash", str(path)]

    async def check(self) -> Optional[WakeSignal]:
        now = time.monotonic()

        # Script file must exist
        if not os.path.exists(self.script_path):
            logger.warning("script_watcher_missing", path=self.script_path)
            return WakeSignal(
                event_type="script_missing",
                severity="warning",
                raw_data={
                    "path": self.script_path,
                    "message": f"Monitoring script not found: {self.script_path}",
                },
                needs_ai_analysis=False,
            )

        cmd = self._resolve_interpreter()
        loop = asyncio.get_running_loop()

        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=60,
                        cwd=self.working_dir,
                    ),
                ),
                timeout=70,
            )
        except asyncio.TimeoutError:
            self._consecutive_failures += 1
            return WakeSignal(
                event_type="script_timeout",
                severity="warning",
                raw_data={
                    "path": self.script_path,
                    "message": "Script timed out after 60 seconds.",
                    "consecutive_failures": self._consecutive_failures,
                },
            )
        except Exception as exc:
            self._consecutive_failures += 1
            logger.error("script_watcher_exec_failed", path=self.script_path, error=str(exc))
            return None

        stdout = result.stdout or ""
        stderr = result.stderr or ""
        exit_code = result.returncode

        # Check alert conditions
        alert_lines = [
            line for line in stdout.splitlines()
            if line.upper().startswith(self.alert_pattern)
        ]
        has_explicit_alert = bool(alert_lines)
        has_failure = self.alert_on_exit_nonzero and exit_code != 0

        if not has_explicit_alert and not has_failure:
            self._consecutive_failures = 0
            return None  # All good

        # Cooldown check
        if now - self._last_fired < self.cooldown:
            return None  # Suppress repeated firing

        self._last_fired = now
        self._consecutive_failures += 1 if has_failure else 0

        severity = "critical" if exit_code != 0 and self._consecutive_failures >= 3 else "warning"
        if has_explicit_alert:
            severity = "warning"

        return WakeSignal(
            event_type="script_alert",
            severity=severity,
            raw_data={
                "script": self.script_path,
                "exit_code": exit_code,
                "stdout": stdout[:2000],
                "stderr": stderr[:500],
                "alert_lines": alert_lines,
                "consecutive_failures": self._consecutive_failures,
            },
            needs_ai_analysis=True,
        )
