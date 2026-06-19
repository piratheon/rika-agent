"""Agent workspace management.

~/.rika/shared is the agent's private sandbox:
  - All shell commands run with cwd=workspace by default
  - Agent uses it freely for temp files, analysis, scripts, etc.
  - Never touches system paths unless the user explicitly requests it
  - Size is monitored; warns at configurable threshold
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from src.utils.logger import logger

_DEFAULT_WORKSPACE = "~/.rika/shared"


def get_workspace_path(configured: Optional[str] = None) -> Path:
    """Return the resolved, guaranteed-to-exist workspace path."""
    raw = configured or os.environ.get("RIKA_WORKSPACE", _DEFAULT_WORKSPACE)
    p = Path(raw).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def workspace_info(path: Optional[Path] = None) -> Dict:
    """Return size, file count, and subdirectory count for the workspace."""
    ws = path or get_workspace_path()
    try:
        total_size = 0
        file_count = 0
        dir_count = 0
        for entry in ws.rglob("*"):
            if entry.is_file():
                total_size += entry.stat().st_size
                file_count += 1
            elif entry.is_dir():
                dir_count += 1
        return {
            "path": str(ws),
            "size_bytes": total_size,
            "size_human": _human_size(total_size),
            "file_count": file_count,
            "dir_count": dir_count,
        }
    except Exception as exc:
        return {"path": str(ws), "error": str(exc)}


def list_workspace(path: Optional[Path] = None, depth: int = 2) -> str:
    """Return a tree-like string listing of workspace contents."""
    ws = path or get_workspace_path()
    if not ws.exists():
        return f"Workspace does not exist yet: {ws}"
    lines = [f"Workspace: {ws}"]
    _tree(ws, lines, prefix="", depth=depth, current=0)
    info = workspace_info(ws)
    lines.append(f"\n{info.get('file_count', 0)} files, {info.get('size_human', '0 B')}")
    return "\n".join(lines)


def clean_workspace(path: Optional[Path] = None) -> str:
    """Delete all contents of the workspace but keep the directory."""
    ws = path or get_workspace_path()
    removed = 0
    errors: List[str] = []
    for entry in ws.iterdir():
        try:
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()
            removed += 1
        except Exception as exc:
            errors.append(f"{entry.name}: {exc}")
    msg = f"Cleaned workspace: {removed} items removed."
    if errors:
        msg += f"\nErrors ({len(errors)}): " + "; ".join(errors[:3])
    logger.info("workspace_cleaned", path=str(ws), removed=removed)
    return msg


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _tree(path: Path, lines: List[str], prefix: str, depth: int, current: int) -> None:
    if current >= depth:
        return
    try:
        entries = sorted(path.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
    except PermissionError:
        return
    for i, entry in enumerate(entries):
        connector = "└── " if i == len(entries) - 1 else "├── "
        size = f"  ({_human_size(entry.stat().st_size)})" if entry.is_file() else ""
        lines.append(f"{prefix}{connector}{entry.name}{size}")
        if entry.is_dir():
            extension = "    " if i == len(entries) - 1 else "│   "
            _tree(entry, lines, prefix + extension, depth, current + 1)


def _human_size(size_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes //= 1024
    return f"{size_bytes:.1f} TB"
