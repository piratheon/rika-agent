"""Code execution sandbox with configurable isolation levels.

Level 0 — None (RestrictedPython):
  No filesystem or network access. Safe for calculations.
  No subprocess spawn. Runs in-process.

Level 1 — Process (ulimit):
  Real Python with ulimit resource limits. Can use any installed package,
  write files to workspace. No network isolation (can't do without root/Docker).
  Works everywhere Linux is available, no extra dependencies.

Level 2 — Docker:
  Maximum isolation. No network, memory-capped, read-only root FS.
  Requires Docker running on the host. Agent can install packages inside
  the container (they don't persist). Best for untrusted code.

Configuration in config.json:
  "sandbox_level": 0 | 1 | 2

Detection in setup script:
  - If Docker available and running → recommend 2
  - Else if Linux → recommend 1
  - Else → 0 (fallback)
"""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from src.utils.logger import logger

MAX_OUTPUT = 8000
_DOCKER_IMAGE = "python:3.12-slim"


# ---------------------------------------------------------------------------
# Level 0 — RestrictedPython (in-process, no isolation)
# ---------------------------------------------------------------------------

async def _run_level0(code: str, timeout: int) -> Dict[str, Any]:
    """RestrictedPython: no file I/O, no network, no imports."""
    try:
        from RestrictedPython import compile_restricted_exec
        from RestrictedPython.Guards import safe_builtins
    except ImportError:
        return {"error": "RestrictedPython not installed: pip install RestrictedPython", "stdout": "", "exit_code": 1}

    import contextlib
    import io

    builtins = dict(safe_builtins)
    for banned in ("open", "eval", "exec", "compile", "__import__"):
        builtins.pop(banned, None)

    # RestrictedPython rewrites print() → _print_(), iteration → _getiter_(),
    # and attribute access → _getattr_(). All must be in globals.
    try:
        from RestrictedPython.PrintCollector import PrintCollector as _PC
        _print_impl = _PC
    except ImportError:
        # Fallback: simple print that captures to stdout
        import io as _io
        _buf_ref: list = []
        class _PC:
            def __init__(self, _getiter_=None): pass
            def __call__(self, *a, **kw): _buf_ref.append(" ".join(str(x) for x in a))
            def __str__(self): return chr(10).join(_buf_ref)
        _print_impl = _PC
    gdict: Dict[str, Any] = {
        "__builtins__": builtins,
        "_print_": _print_impl,
        "_getiter_": iter,
        "_getattr_": getattr,
        "_write_": lambda x: x,
    }
    ldict: Dict[str, Any] = {}

    def _run() -> Dict[str, Any]:
        buf = io.StringIO()
        try:
            result = compile_restricted_exec(code)
            if result.errors:
                return {"stdout": "", "stderr": "\n".join(result.errors),
                        "exit_code": 1, "isolation": "none"}
            # Inject _getiter_ into print collector if it needs it
            if "_print_" in gdict:
                try:
                    gdict["_print_"] = gdict["_print_"](_getiter_=iter)
                except Exception:
                    pass
            with contextlib.redirect_stdout(buf):
                exec(result.code, gdict, ldict)  # noqa: S102
            return {"stdout": buf.getvalue(), "stderr": "", "exit_code": 0,
                    "result": repr(ldict.get("result")), "isolation": "none"}
        except Exception as exc:
            return {"stdout": buf.getvalue(), "stderr": f"{type(exc).__name__}: {exc}",
                    "exit_code": 1, "isolation": "none"}

    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(loop.run_in_executor(None, _run), timeout=float(timeout))
    except asyncio.TimeoutError:
        return {"error": "Execution timed out", "stdout": "", "exit_code": 124, "isolation": "none"}


# ---------------------------------------------------------------------------
# Level 1 — Subprocess + ulimit (process isolation, Linux/macOS)
# ---------------------------------------------------------------------------

async def _run_level1(code: str, timeout: int, workspace: str) -> Dict[str, Any]:
    exec_dir = Path(workspace) / ".exec"
    exec_dir.mkdir(parents=True, exist_ok=True)
    script = exec_dir / f"_{uuid4().hex[:8]}.py"
    script.write_text(code, encoding="utf-8")

    # Build resource-limited command
    # ulimit: -v 512MB virtual memory, -t CPU seconds, -f 50MB file size
    mem_kb = 524288  # 512 MB
    cmd = (
        f"ulimit -v {mem_kb} -t {timeout} -f 51200 2>/dev/null; "
        f"timeout {timeout + 2} {sys.executable} {script}"
    )

    try:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workspace,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=float(timeout + 5)
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            script.unlink(missing_ok=True)
            return {"error": "Execution timed out", "stdout": "", "exit_code": 124, "isolation": "process"}

        stdout = stdout_b.decode("utf-8", errors="replace")[:MAX_OUTPUT]
        stderr = stderr_b.decode("utf-8", errors="replace")[:2000]
        return {
            "stdout": stdout, "stderr": stderr,
            "exit_code": proc.returncode or 0,
            "script": str(script.name),
            "isolation": "process",
        }
    finally:
        script.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Level 2 — Docker (full isolation)
# ---------------------------------------------------------------------------

async def _check_docker() -> bool:
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "info",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
        return proc.returncode == 0
    except FileNotFoundError:
        return False


async def _run_level2(code: str, timeout: int, workspace: str) -> Dict[str, Any]:
    exec_dir = Path(workspace) / ".exec"
    exec_dir.mkdir(parents=True, exist_ok=True)
    script = exec_dir / f"_{uuid4().hex[:8]}.py"
    script.write_text(code, encoding="utf-8")

    # Map workspace into container so agent can write files there
    cmd = [
        "docker", "run", "--rm",
        "--network", "none",
        "--memory", "512m",
        "--memory-swap", "512m",
        "--cpus", "0.5",
        "--read-only",
        "--tmpfs", "/tmp:size=64m,mode=1777",
        "-v", f"{script.resolve()}:/code/script.py:ro",
        "-v", f"{Path(workspace).resolve()}:/workspace:rw",
        "--workdir", "/workspace",
        _DOCKER_IMAGE,
        "timeout", str(timeout), "python3", "/code/script.py",
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=float(timeout + 15)
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            script.unlink(missing_ok=True)
            return {"error": "Docker execution timed out", "stdout": "", "exit_code": 124, "isolation": "docker"}

        stdout = stdout_b.decode("utf-8", errors="replace")[:MAX_OUTPUT]
        stderr = stderr_b.decode("utf-8", errors="replace")[:2000]
        return {
            "stdout": stdout, "stderr": stderr,
            "exit_code": proc.returncode or 0,
            "isolation": "docker",
        }
    except FileNotFoundError:
        return {"error": "Docker not found. Set sandbox_level to 0 or 1.", "stdout": "", "exit_code": 1}
    finally:
        script.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

async def run_sandboxed(
    code: str,
    level: int = 0,
    timeout: int = 30,
    workspace: Optional[str] = None,
) -> Dict[str, Any]:
    """Run Python code at the specified isolation level.

    Args:
        code:      Python source code to execute.
        level:     0 = RestrictedPython, 1 = process+ulimit, 2 = Docker.
        timeout:   Max execution time in seconds.
        workspace: Path to agent workspace (required for levels 1 and 2).

    Returns:
        Dict with stdout, stderr, exit_code, and isolation level used.
    """
    if not code or not code.strip():
        return {"error": "No code provided", "stdout": "", "exit_code": 1}

    ws = workspace or os.path.expanduser("~/.Rika-Workspace")

    logger.info("sandbox_execute", level=level, code_len=len(code), timeout=timeout)

    if level >= 2:
        if await _check_docker():
            return await _run_level2(code, timeout, ws)
        else:
            logger.warning("sandbox_docker_unavailable_fallback_level1")
            return await _run_level1(code, timeout, ws)

    if level >= 1:
        # ulimit works reliably on Linux; macOS has partial support
        if sys.platform.startswith(("linux", "darwin")):
            return await _run_level1(code, timeout, ws)
        else:
            logger.warning("sandbox_ulimit_unsupported_fallback_level0", platform=sys.platform)
            return await _run_level0(code, timeout)

    return await _run_level0(code, timeout)


async def detect_best_level() -> int:
    """Detect the highest isolation level available on this system."""
    if await _check_docker():
        return 2
    if sys.platform.startswith(("linux", "darwin")):
        return 1
    return 0


def describe_level(level: int) -> str:
    return {
        0: "Level 0 — RestrictedPython (no file/network access, in-process)",
        1: "Level 1 — Process + ulimit (real Python, resource-limited, can write to workspace)",
        2: "Level 2 — Docker (full isolation: no network, memory-capped, ephemeral container)",
    }.get(level, f"Unknown level {level}")
