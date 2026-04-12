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
    summarization_interval: int = 8
    # sliding window — raw messages kept
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
    workspace_path: str = "~/.Rika-Workspace"
    workspace_max_size_mb: int = 500
    log_level: str = "info"
    default_model: str = "gemini-2.0-flash"
    gemini_quota_reset_utc_hour: int = 8
    groq_quota_reset_utc_hour: int = 0
    openrouter_quota_reset_utc_hour: int = 0
    # Code sandbox isolation level (0=RestrictedPython, 1=ulimit, 2=Docker)
    sandbox_level: int = 0

    ollama_enabled: bool = False
    ollama_base_url: str = "http://localhost:11434"
    ollama_default_model: str = "llama3.2"
    g4f_enabled: bool = False
    
    # Per-provider model configuration (fallbacks if not set)
    groq_model: str = "llama-3.3-70b-versatile"
    openrouter_model: str = "google/gemini-2.0-flash-001"
    gemini_model: str = "gemini-2.0-flash"
    ollama_model: str = "llama3.2"
    g4f_model: str = "MiniMaxAI/MiniMax-M2.5"  # DeepInfra provider
    
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
        "\n\n--- OPERATIONAL RULES ---\n"
        "1. ACCURACY: Ground responses in reality. Use tools to verify facts.\n"
        "2. RESPONSE: After gathering information, respond naturally and completely.\n"
        "3. REASONING: You can reason, think, and respond directly without tools for:\n"
        "   - Casual conversation, greetings, questions\n"
        "   - Factual questions from your training knowledge\n"
        "   - Analysis, explanations, creative tasks\n"
        "   - Any task that doesn't require real-time data or system access\n"
        "4. TOOLS require background watchers: For tool execution (web_search, shell commands, etc.),\n"
        "   the user must have active watchers. Suggest /watch or /autowatch if they need tools.\n"
        "5. NO HALLUCINATION: If a tool fails or isn't available, be honest. Never fabricate results.\n"
        "6. WORKSPACE: Your sandbox is ~/.Rika-Workspace (path in runtime context).\n"
        "   Write temp files, scripts, and analysis artifacts there by default.\n"
        "7. COMMAND SECURITY: Destructive commands are blocked automatically.\n"
        "   Prefix medium-risk commands with 'CONFIRM: ' after warning the user.\n"
        "8. NARRATION: Before starting any group of 2+ tool calls for a distinct goal,\n"
        "   call declare_step(title=\"...\") first. After completion call\n"
        "   declare_step(title=\"...\", status=\"done\") or status=\"failed\".\n"
        "   Keep titles short and action-oriented: 'Creating backend', not 'I will now create...'\n"
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
            + "\n\nTo call a tool: TOOL: tool_name | QUERY: your query"
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
    def load(cls, path: str = "config.json") -> "Config":
        load_dotenv()
        p = Path(path)
        data = json.loads(p.read_text()) if p.exists() else {}
        cfg = cls(**data)
        soul = Path("soul.md")
        identity = (
            soul.read_text(encoding="utf-8")
            if soul.exists()
            else "You are a helpful, precise, and thoughtful AI assistant."
        )
        cfg.system_prompt = f"{identity}\n{cfg.get_tools_prompt()}\n{cls.TECHNICAL_MANDATES}"
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
