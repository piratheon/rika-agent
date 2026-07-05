"""Config — loaded from config.json + soul.md, cached with a short TTL.

Use Config.get() everywhere instead of Config.load() to avoid re-reading
disk on every incoming message. Config.reload() forces a fresh read.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import ClassVar, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

_cache: Optional["Config"] = None
_cache_at: float = 0.0
_CACHE_TTL: float = 30.0


class Config(BaseModel):
    model_config = ConfigDict(extra="allow")

    bot_name: str = "rk-agent"
    access_mode: str = "allowlist"
    allowed_user_ids: List[int] = []
    default_provider_priority: List[str] = ["groq", "openrouter", "gemini"]
    max_api_keys_per_user: int = 10
    max_context_messages: int = 10
    summarization_interval: int = 8    # trigger incremental summary every N messages
    max_pinned_memories: int = 5       # hard cap on always-injected memories
    max_relevant_memories: int = 4     # semantic retrieval per call
    max_agents_per_task: int = 6
    agent_task_timeout_seconds: int = 90
    live_bubble_throttle_ms: int = 800
    enable_code_execution: bool = True
    enable_wikipedia_search: bool = True
    enable_web_fetch: bool = True
    enable_web_search: bool = True
    enable_telegram: bool = True
    enable_web_ui: bool = False
    enable_command_security: bool = True
    command_security_level: str = "standard"
    workspace_path: str = "~/.rika/shared"
    workspace_max_size_mb: int = 500
    log_level: str = "info"
    default_model: str = "gemini-2.0-flash"
    gemini_quota_reset_utc_hour: int = 8
    groq_quota_reset_utc_hour: int = 0
    openrouter_quota_reset_utc_hour: int = 0
    # Code sandbox isolation level (0=RestrictedPython, 1=ulimit, 2=Docker)
    sandbox_level: int = 0

    # Provider retry policy
    provider_max_retries: int = 2       # retries per provider on transient errors
    provider_retry_delay: float = 2.0   # base delay in seconds (doubles each attempt)

    ollama_enabled: bool = False
    ollama_base_url: str = "http://localhost:11434"
    ollama_default_model: str = "llama3.2"
    g4f_enabled: bool = False

    # Vercel AI Gateway
    vercel_enabled: bool = False
    vercel_model: str = "openai/gpt-4o-mini"
    vercel_auto_detect: bool = True   # auto-add to priority when VERCEL=1

    # NVIDIA NIM (auto-detected from NVIDIA_API_KEY)
    nvidia_enabled: bool = False
    nvidia_model: str = "meta/llama-3.1-70b-instruct"
    nvidia_auto_detect: bool = True

    # Vercel Postgres (auto-detected from POSTGRES_URL env var)
    vercel_postgres_enabled: bool = False
    vercel_postgres_pool_min: int = 2
    vercel_postgres_pool_max: int = 10

    # Vercel KV / Upstash Redis (auto-detected from KV_REST_API_URL)
    vercel_kv_enabled: bool = False
    
    # Per-provider model configuration (fallbacks if not set)
    groq_model: str = "llama-3.3-70b-versatile"
    openrouter_model: str = "google/gemini-2.0-flash-001"
    gemini_model: str = "gemini-2.0-flash"
    ollama_model: str = "llama3.2"
    g4f_model: str = "MiniMaxAI/MiniMax-M2.5"  # DeepInfra provider
    
    # ── Multi-tier model routing ──────────────────────────────────────────────
    # Map complexity tier to provider + model. If a tier is missing, falls back
    # to the next tier in tier_fallback_chain.
    model_tiers: dict = {}          # populated below in load(); see defaults
    tier_fallback_chain: list = ["complex", "mid", "low"]
    planning_enabled: bool = True   # set False to disable planning phase globally
    dynamic_replan_on_failure: bool = True
    # Log directory (overrides ~/.rika/logs default when set explicitly)
    log_dir: str = ""

    max_tool_result_chars: int = 8000
    compact_after_turns: int = 12
    min_free_disk_mb: int = 0
    discord_edit_throttle_ms: int = 800
    track_token_usage: bool = True
    auto_pause_on_quota: bool = True
    max_background_agents_per_user: int = 10
    wake_event_retention_days: int = 30
    max_concurrent_orchestrations_per_user: int = 2
    system_prompt: str = ""
    
    # Per-provider tool schema limits
    # Groq llama models fail with >8-10 function declarations
    max_tools_groq: int = 8

    # Tool execution timeout in seconds
    tool_timeout_seconds: int = 10

    TECHNICAL_MANDATES: ClassVar[str] = (
        "\n\n--- OPERATIONAL RULES (READ CAREFULLY) ---\n"
        "\n"
        "TOOL CALLING — CRITICAL:\n"
        "Tools are invoked ONLY through the JSON function-calling API.\n"
        "Writing tool calls as text (e.g. run_shell_command(\"ls\") in prose or markdown)\n"
        "does NOTHING — no code runs, no result is returned, the loop stalls.\n"
        "NEVER write tool calls as text, markdown, code blocks, or Python syntax.\n"
        "Every response must be EITHER a JSON function call OR end_thinking().\n"
        "Mixing prose + tool-call text is forbidden and will be corrected.\n"
        "\n"
        "STEP-BY-STEP THINKING DISCIPLINE:\n"
        "Step 1 — Planning: call declare_step(title=\"Plan: <goal>\", status=\"running\")\n"
        "          then execute the FIRST concrete action only.\n"
        "Step 2+ — For each subsequent action: call declare_step(title=\"<action>\")\n"
        "           then execute that single action. Do NOT batch multiple actions.\n"
        "Final    — When ALL steps are done: call end_thinking(message=\"<final answer>\").\n"
        "           end_thinking MUST be your ONLY response in that turn. No other text.\n"
        "\n"
        "RULES:\n"
        "1. ACCURACY: Ground responses in reality. Use tools to verify facts.\n"
        "2. NO HALLUCINATION: Never fabricate tool results. If a tool fails, report honestly.\n"
        "3. WORKSPACE: Default working directory is ~/.rika/shared.\n"
        "4. COMMAND SECURITY: Destructive commands are auto-blocked.\n"
        "   Prefix medium-risk with CONFIRM: after warning the user.\n"
        "5. DECLARE STEPS: Every distinct goal needs a declare_step() call before execution.\n"
        "   Keep titles short: \'Downloading captions\' not \'I will now download the captions\'.\n"
        "6. ONE ACTION PER TURN: Call one tool at a time unless the tools are fully independent\n"
        "   reads (e.g. two web_search calls that do not depend on each other).\n"
    )

    def get_tools_prompt(self) -> str:
        tools: List[str] = []
        if self.enable_web_search:
            tools.append("- web_search: Search the web (DuckDuckGo, no API key).")
        if self.enable_wikipedia_search:
            tools.append("- wikipedia_search: Get Wikipedia summaries.")
        if self.enable_web_fetch:
            tools.append("- curl: Fetch and extract text from a URL.")
        if self.enable_code_execution:
            tools.append("- run_shell_command: Execute shell commands (cwd = workspace).")
            tools.append("- run_python: Execute Python in a sandboxed environment.")
        tools += [
            "- list_workspace: List files in the workspace.",
            "- read_file: Read content from a file (path, max_lines=200).",
            "- write_file: Write text/JSON/code to a file (path, content, mode='w').",
            "- send_file: Send a workspace file to the user (path, caption='').",
            "- save_memory: Persist key-value pair. Format: 'key | value'.",
            "- get_memories: Retrieve all stored memories and skills.",
            "- save_skill: Store a reusable skill/code snippet. Format: 'name | code'.",
            "- use_skill: Load a stored skill by name. Format: 'skill_name'.",
            "- delegate_task: Spawn a research sub-agent for a specific query.",
        ]
        if not tools:
            return "\nNote: No external tools enabled."
        return (
            "\n--- AVAILABLE TOOLS ---\n"
            + "\n".join(tools)
        )

    def get_system_prompt_for_fc(self) -> str:
        """System prompt stripped of text tool list for function-calling mode.

        When sending tool schemas as JSON functions, the LLM already knows
        all tools. The text "--- AVAILABLE TOOLS ---" block is redundant,
        wastes tokens, and can make models output text-protocol calls instead
        of JSON function calls.
        """
        import re as _re
        stripped = _re.sub(
            r"\n--- AVAILABLE TOOLS ---.*?(?=\n---|$)",
            "",
            self.system_prompt,
            flags=_re.DOTALL,
        )
        return stripped.strip()

    @classmethod
    def detect_platforms(cls) -> dict[str, bool]:
        """Auto-detect which platforms are enabled from env vars + config.

        Returns dict like {"telegram": True, ...}
        """
        import os as _os
        platforms: dict[str, bool] = {}

        # Telegram: TELEGRAM_BOT_TOKEN env var
        has_token = bool(_os.environ.get("TELEGRAM_BOT_TOKEN", "").strip())
        platforms["telegram"] = has_token

        # Discord: DISCORD_BOT_TOKEN env var
        platforms["discord"] = bool(_os.environ.get("DISCORD_BOT_TOKEN", "").strip())

        # API: ENABLE_API env var
        _api = _os.environ.get("ENABLE_API", "").strip().lower()
        platforms["api"] = _api in ("1", "true", "yes")

        # library: always False — not a server, always started programmatically

        return platforms

    @classmethod
    def log_platform_status(cls, platforms: dict[str, bool]) -> None:
        """Log which platforms are enabled/disabled in a clean table."""
        sep = "+" + "-" * 20 + "+" + "-" * 12 + "+"
        print(sep)
        print(f"| {'Platform':<18} | {'Status':<10} |")
        print(sep)
        for name, enabled in sorted(platforms.items()):
            status = "ENABLED" if enabled else "disabled"
            print(f"| {name:<18} | {status:<10} |")
        print(sep)
        enabled_count = sum(1 for v in platforms.values() if v)
        if enabled_count == 0:
            print("No interface enabled. Running background tasks only.")
        else:
            print(f"{enabled_count} interface(s) active — starting...")

    @classmethod
    def load(cls, path: str = "config.json") -> "Config":
        load_dotenv()
        p = Path(path)
        if not p.exists():
            import logging as _logging
            _logging.warning("config.json not found — using defaults. Copy config.json.template to config.json to customize.")
        data = json.loads(p.read_text()) if p.exists() else {}
        cfg = cls(**data)

        # Auto-detect Vercel deployment and prepend vercel to provider priority
        import os as _os
        _on_vercel = _os.environ.get("VERCEL", "") == "1" or bool(
            _os.environ.get("VERCEL_ENV", "")
        )
        _has_vercel_key = bool(_os.environ.get("VERCEL_API_KEY", "").strip())
        if cfg.nvidia_auto_detect and _os.environ.get("NVIDIA_API_KEY", "").strip():
            if "nvidia" not in cfg.default_provider_priority:
                cfg.default_provider_priority = ["nvidia"] + cfg.default_provider_priority
            cfg.nvidia_enabled = True
        if cfg.vercel_auto_detect and _on_vercel and _has_vercel_key:
            if "vercel" not in cfg.default_provider_priority:
                cfg.default_provider_priority = [
                    "vercel"
                ] + cfg.default_provider_priority
            cfg.vercel_enabled = True
        soul = Path("soul.md")
        identity = (
            soul.read_text(encoding="utf-8")
            if soul.exists()
            else "You are a helpful, precise, and thoughtful AI assistant."
        )
        cfg.system_prompt = f"{identity}\n{cfg.get_tools_prompt()}\n{cls.TECHNICAL_MANDATES}"

        # Populate model_tiers defaults if not set in config.json
        if not cfg.model_tiers:
            priority = cfg.default_provider_priority or ["groq", "openrouter", "gemini"]
            _primary = priority[0] if priority else "groq"
            _groq_mid   = "llama-3.3-70b-versatile"
            _groq_low   = "llama-3.1-8b-instant"
            _or_complex = "anthropic/claude-opus-4-5"
            cfg.model_tiers = {
                "complex": {"provider": "openrouter", "model": _or_complex},
                "mid":     {"provider": _primary,     "model": cfg.groq_model or _groq_mid},
                "low":     {"provider": "groq",       "model": _groq_low},
            }

        return cfg

    @classmethod
    def get(cls) -> "Config":
        global _cache, _cache_at
        now = time.monotonic()
        if _cache is None or (now - _cache_at) >= _CACHE_TTL:
            _cache = cls.load()
            _cache_at = now
        return _cache

    @classmethod
    def reload(cls) -> "Config":
        global _cache, _cache_at
        _cache = cls.load()
        _cache_at = time.monotonic()
        return _cache

    @classmethod
    def invalidate(cls) -> None:
        global _cache, _cache_at
        _cache = None
        _cache_at = 0.0
