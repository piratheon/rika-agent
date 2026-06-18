"""Shell tool — command execution with security checking and workspace isolation."""
from __future__ import annotations
import re

import asyncio
import os
import subprocess
import time
from typing import Any, Dict, Optional

from src.utils.logger import logger

MAX_OUTPUT_CHARS = 4000
_CONFIRM_PREFIX = "CONFIRM:"


def _get_workspace() -> str:
    from src.tools.workspace import get_workspace_path
    return str(get_workspace_path())


def _get_security_level() -> str:
    try:
        from src.config import Config
        return getattr(Config.get(), "command_security_level", "standard")
    except Exception:
        return "standard"


def _is_security_enabled() -> bool:
    try:
        from src.config import Config
        return getattr(Config.get(), "enable_command_security", True)
    except Exception:
        return True


async def _audit(user_id, command, workspace, result=None,
                 blocked=False, block_reason="", block_severity="", confirmed=False):
    try:
        from src.db.connection import get_db
        stdout_head = (result.get("stdout", "") or "")[:500] if result else ""
        stderr_head = (result.get("stderr", "") or "")[:200] if result else ""
        exit_code = result.get("exit_code") if result else None
        async with get_db() as db:
            await db.execute(
                "INSERT INTO command_audit "
                "(user_id, command, exit_code, stdout_head, stderr_head, "
                "was_blocked, block_reason, block_severity, confirmed_override, workspace_path) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (user_id, command, exit_code, stdout_head, stderr_head,
                 1 if blocked else 0, block_reason, block_severity,
                 1 if confirmed else 0, workspace),
            )
            await db.commit()
    except Exception as exc:
        logger.warning("command_audit_write_failed", error=str(exc))


async def run_shell_command(command: str, user_id: int = 0, workspace: Optional[str] = None) -> Dict[str, Any]:
    """Execute a shell command with security checking and workspace isolation.

    Prefix the command with 'CONFIRM: ' to override a MEDIUM-severity block after warning.
    All commands run with cwd=~/.rika/shared by default.
    """
    ws = workspace or _get_workspace()

    confirmed = False
    cmd = command.strip()
    if cmd.upper().startswith(_CONFIRM_PREFIX):
        confirmed = True
        cmd = cmd[len(_CONFIRM_PREFIX):].strip()

    if not cmd:
        return {"error": "Empty command."}

    if _is_security_enabled():
        from src.tools.command_security import check_command, format_block_message
        sec = check_command(cmd, workspace_path=ws, security_level=_get_security_level())

        if not sec.allowed:
            if sec.requires_confirmation and confirmed:
                logger.warning("command_confirmed_override", command=cmd[:200],
                               severity=sec.severity, rule=sec.matched_rule)
                asyncio.create_task(_audit(user_id, cmd, ws, confirmed=True,
                                       block_severity=sec.severity, block_reason=sec.reason))
            else:
                asyncio.create_task(_audit(user_id, cmd, ws, blocked=True,
                                           block_reason=sec.reason, block_severity=sec.severity))
                return {
                    "blocked": True,
                    "severity": sec.severity,
                    "message": format_block_message(sec),
                    "stdout": "",
                    "stderr": "",
                    "exit_code": -1,
                }

    # Secondary structural check — blocks injection patterns that evade
    # the regex-based command_security checker regardless of security level.
    _INJECT_RE = re.compile(r'`[^`]+`|\$\([^)]+\)|;\s*(?:rm|curl|wget|nc|bash|sh)\b',
                             re.IGNORECASE)
    if _INJECT_RE.search(cmd):
        logger.warning("shell_structural_injection_blocked", command=cmd[:200])
        return {
            "blocked": True,
            "severity": "critical",
            "message": "Command blocked: contains shell injection pattern ($(), backticks, or chained rm/curl/wget/nc/bash/sh).",
            "stdout": "", "stderr": "", "exit_code": -1,
        }

    logger.info("executing_shell_command", command=cmd[:200], workspace=ws)

    try:
        loop = asyncio.get_running_loop()
        proc = await loop.run_in_executor(None, lambda: subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=120, cwd=ws
        ))
        stdout = proc.stdout
        stderr = proc.stderr
        truncated = False
        if len(stdout) > MAX_OUTPUT_CHARS:
            stdout = stdout[:MAX_OUTPUT_CHARS] + f"\n...[truncated, {len(proc.stdout)-MAX_OUTPUT_CHARS} chars omitted]"
            truncated = True
        if len(stderr) > 1000:
            stderr = stderr[:1000] + "\n...[stderr truncated]"
        out = {"stdout": stdout, "stderr": stderr, "exit_code": proc.returncode, "cwd": ws}
        if truncated:
            out["truncated"] = True
        asyncio.create_task(_audit(user_id, cmd, ws, result=out))
        return out
    except subprocess.TimeoutExpired:
        return {"error": "Command timed out after 120 seconds.", "cwd": ws}
    except Exception as exc:
        return {"error": str(exc), "cwd": ws}


async def watch_task_logs(file_path: str, timeout: str = "30") -> Dict[str, Any]:
    """Async log watcher — non-blocking."""
    try:
        timeout_sec = max(5, int(timeout))
    except (TypeError, ValueError):
        timeout_sec = 30

    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}

    start = time.monotonic()
    try:
        initial_size = os.path.getsize(file_path)
        content = ""
        while time.monotonic() - start < timeout_sec:
            current_size = os.path.getsize(file_path)
            if current_size > initial_size:
                with open(file_path, errors="replace") as f:
                    f.seek(initial_size)
                    content += f.read(MAX_OUTPUT_CHARS - len(content))
                    initial_size = current_size
                if len(content) >= MAX_OUTPUT_CHARS:
                    content += "\n...[truncated]"
                    break
            await asyncio.sleep(1)
        return {
            "log_content": content or "No new log entries during watch window.",
            "duration_seconds": int(time.monotonic() - start),
            "file": file_path,
        }
    except Exception as exc:
        return {"error": str(exc)}


async def get_command_history(user_id: int, limit: int = 15) -> str:
    """Return recent command audit entries for a user."""
    try:
        from src.db.connection import get_db
        async with get_db() as db:
            cur = await db.execute(
                "SELECT command, exit_code, was_blocked, block_severity, executed_at "
                "FROM command_audit WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            )
            rows = await cur.fetchall()
        if not rows:
            return "No command history yet."
        lines = [f"Last {len(rows)} commands:\n"]
        for cmd, code, blocked, sev, at in reversed(rows):
            at_str = (at or "")[:16]
            status = f"[BLOCKED:{sev}]" if blocked else f"[exit:{code}]"
            lines.append(f"{at_str}  {status}  {cmd[:80]}")
        return "\n".join(lines)
    except Exception as exc:
        return f"Error reading history: {exc}"
