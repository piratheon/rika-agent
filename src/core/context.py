"""Runtime context builder — creates system context for agent.

Extracted from app.py _build_runtime_context() function.
Same logic, just wrapped in a class for reusability.
"""
from __future__ import annotations

import os
import platform
import socket
from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.config import Config
from src.utils.logger import logger


class ContextBuilder:
    """Builds runtime context injected into agent messages."""
    
    @staticmethod
    def build_runtime_context(tg_user, cfg: Config) -> str:
        """Build slim runtime context (~40 tokens vs ~120 for verbose version).
        
        Same code as app.py _build_runtime_context() but as a static method.
        """
        now = datetime.now(timezone.utc)
        # Also get local time with offset for the user's likely context
        local_now = datetime.now()

        username = tg_user.username or tg_user.first_name or f"user_{tg_user.id}"
        full_name = " ".join(filter(None, [tg_user.first_name, tg_user.last_name]))

        try:
            hostname = socket.gethostname()
        except Exception:
            hostname = "unknown"

        try:
            os_info = f"{platform.system()} {platform.release()} ({platform.machine()})"
        except Exception:
            os_info = "unknown"

        try:
            python_ver = platform.python_version()
        except Exception:
            python_ver = "unknown"

        try:
            from src.tools.workspace import get_workspace_path
            workspace = str(get_workspace_path(getattr(cfg, "workspace_path", None)))
        except Exception:
            workspace = "~/.rika/shared"

        # Slim context — ~40 tokens vs ~120 for the full version
        lines = [
            f"[CTX] {now.strftime('%Y-%m-%d %H:%M')} UTC | "
            f"user:{username} | host:{hostname} | "
            f"model:{cfg.default_model} | ws:{workspace}",
        ]
        return "\n".join(lines)
    
    @staticmethod
    async def build_full_context(
        tg_user,
        cfg: Config,
        user_id: int,
        summary: Optional[str] = None,
        history: Optional[List[Dict]] = None,
        user_message: str = "",
    ) -> str:
        """Build full context with history and summary.
        
        Same logic as app.py _process_message() context building.
        """
        context_parts = []
        
        # Runtime context
        context_parts.append(ContextBuilder.build_runtime_context(tg_user, cfg))
        
        # Summary if exists
        if summary:
            context_parts.append(f"[Earlier context summary]\n{summary}")
        
        # History
        if history:
            for m in history[:-1]:  # Exclude current message
                context_parts.append(f"{m['role']}: {m['content']}")
        
        # Current user message
        if user_message:
            context_parts.append(f"user: {user_message}")
        
        return "\n".join(context_parts)
