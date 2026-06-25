"""Tool registry — module-level singleton."""
from __future__ import annotations
from typing import Callable, Dict, Optional

_registry: Optional[Dict[str, Callable]] = None

def get_registry() -> Dict[str, Callable]:
    global _registry
    if _registry is None:
        _registry = _build()
    return _registry

def invalidate_registry() -> None:
    global _registry
    _registry = None

def _build() -> Dict[str, Callable]:
    from src.config import Config
    cfg = Config.get()
    registry: Dict[str, Callable] = {}

    if cfg.enable_web_search:
        from src.tools.web_search_tool import web_search as _web_search

        async def web_search_tool(query: str, max_results: int = 5) -> str:
            # Coerce max_results to int — ToolExecutor may pass it as str
            try:
                max_results = int(max_results)
            except (TypeError, ValueError):
                max_results = 5
            return await _web_search(query=query, max_results=max_results)

        registry["web_search"] = web_search_tool

    if cfg.enable_wikipedia_search:
        from src.tools.wikipedia_tool import wikipedia_search
        registry["wikipedia_search"] = wikipedia_search

    if cfg.enable_web_fetch:
        from src.tools.curl_tool import curl_fetch
        registry["curl"] = curl_fetch

    if cfg.enable_code_execution:
        from src.tools.shell_tool import run_shell_command, watch_task_logs
        registry["run_shell_command"] = run_shell_command
        registry["watch_task_logs"] = watch_task_logs
        from src.tools.code_runner_tool import run_python
        registry["run_python"] = run_python

    async def list_workspace_tool(query: str = "") -> str:
        from src.tools.workspace import get_workspace_path, list_workspace
        return list_workspace(get_workspace_path(cfg.workspace_path), depth=3)

    registry["list_workspace"] = list_workspace_tool

    # NOTE: Memory tools (save_memory, get_memories, save_skill, use_skill,
    #        delegate_task) are intentionally NOT in the registry.
    # They require user_id context and are handled by ToolExecutor.execute()
    # in src/core/tools.py which receives user_id from the orchestration layer.

    # File delivery tool — returns a marker for the orchestration layer to handle
    async def send_file_tool(path: str, caption: str = "") -> str:
        """Signal to send a file. The orchestration layer handles actual delivery."""
        if not path:
            return "Error: 'path' is required."
        return f"__SEND_FILE__:{path}:{caption}"

    registry["send_file"] = send_file_tool

    # File write tool
    async def write_file_tool(path: str, content: str, mode: str = "w") -> str:
        from src.tools.workspace import get_workspace_path
        if not path:
            return "Error: path is required."
        if content is None:
            return "Error: content is required."
        workspace = get_workspace_path(cfg.workspace_path).resolve()
        try:
            full_path = (workspace / path).resolve()
        except Exception:
            return "Error: invalid path."
        if not full_path.is_relative_to(workspace):
            return "Error: path outside workspace."
        rel = str(full_path.relative_to(workspace))
        try:
            full_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_path, mode, encoding="utf-8") as fh:
                fh.write(content)
            return f"File written: {rel} ({len(content)} bytes)"
        except Exception as e:
            return f"Error writing file: {e}"

    registry["write_file"] = write_file_tool

    # File read tool
    async def read_file_tool(path: str, max_lines: int = 200) -> str:
        from src.tools.workspace import get_workspace_path
        if not path:
            return "Error: path is required."
        workspace = get_workspace_path(cfg.workspace_path).resolve()
        try:
            full_path = (workspace / path).resolve()
        except Exception:
            return "Error: invalid path."
        if not full_path.is_relative_to(workspace):
            return "Error: path outside workspace."
        rel = str(full_path.relative_to(workspace))
        if not full_path.exists():
            return f"Error: file not found: {rel}"
        if full_path.stat().st_size > 1_048_576:
            return "Error: file too large. Use run_shell_command with head/tail."
        try:
            lines = full_path.read_text(encoding="utf-8").splitlines(keepends=True)
            max_lines = int(max_lines)
            if len(lines) > max_lines:
                preview = "".join(lines[:max_lines])
                header = f"File: {rel} (first {max_lines}/{len(lines)} lines):"
                return header + chr(10) + preview + chr(10) + "[... truncated ...]"
            return f"File: {rel}" + chr(10) + "".join(lines)
        except Exception as e:
            return f"Error reading file: {e}"

    registry["read_file"] = read_file_tool

    return registry

