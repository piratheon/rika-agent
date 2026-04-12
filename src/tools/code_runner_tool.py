"""Code runner — delegates to sandbox.py at the configured isolation level."""
from __future__ import annotations
import asyncio
from typing import Any, Dict, Optional
from src.utils.logger import logger

async def run_python(code: str, timeout_seconds: int = 30,
                     workspace: Optional[str] = None) -> Dict[str, Any]:
    """Execute Python code at the configured sandbox isolation level."""
    from src.config import Config
    from src.tools.sandbox import run_sandboxed
    from src.tools.workspace import get_workspace_path
    cfg = Config.get()
    level = getattr(cfg, "sandbox_level", 0)
    ws = workspace or str(get_workspace_path(cfg.workspace_path))
    # Level 0 (RestrictedPython) blocks __import__ entirely.
    # Auto-escalate to level 1 (subprocess) when code has import statements.
    import re as _re
    _has_imports = bool(_re.search(r"^\s*(?:import|from)\s+\w", code, _re.MULTILINE))
    if level == 0 and _has_imports:
        import sys as _sys
        if _sys.platform.startswith(("linux", "darwin")):
            level = 1  # use subprocess — supports imports
    result = await run_sandboxed(code, level=level, timeout=int(timeout_seconds), workspace=ws)
    # Format for agent consumption
    if result.get("error") and not result.get("stdout"):
        return {"error": result["error"], "stdout": "", "exit_code": result.get("exit_code", 1)}
    return result
