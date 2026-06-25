"""Telegram bot entry point.

Changes vs v1:
- cfg = Config.get() moved before any branch that uses it (NameError fix).
- document_handler no longer mutates the frozen Update object.
- Per-user orchestration semaphore (max 2 concurrent tasks).
- Background agent manager initialized in main().
- /watch, /watchers, /stopwatch, /wakelog commands.
- /memory, /deletememory commands.
- Sentinel promoted to BackgroundAgentManager with system watcher.
- Streaming path wired through LiveBubble for direct replies.
- Core logic extracted to src.core for reuse across interfaces.
"""

from __future__ import annotations

print("""
 ██████╗ ██╗██╗  ██╗ █████╗       █████╗  ██████╗ ███████╗███╗   ██╗████████╗
 ██╔══██╗██║██║ ██╔╝██╔══██╗     ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝
 ██████╔╝██║█████╔╝ ███████║     ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   
 ██╔══██╗██║██╔═██╗ ██╔══██║     ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   
 ██║  ██║██║██║  ██╗██║  ██║     ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   
 ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝╚═╝  ╚═╝     ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   
""")

import asyncio
import json
import os
import re
import uuid
from typing import Dict, Optional

from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

try:
    from src.config import Config
    from src.providers.provider_pool import get_pool
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from src.config import Config
    from src.providers.provider_pool import get_pool

# Import core logic
from src.core import (
    AgentState,
    AgentStatus,
    ContextBuilder,
    Orchestrator,
    ToolCall,
    ToolResult,
    classify_complexity,
)

from src.core.event_sink import EventSink
from src.core.router import PlatformRouter
from src.interfaces.telegram.adapter import TelegramAdapter
from src.utils.logger import logger

# Global adapter, sink and router — set in build_application()
_adapter: Optional[TelegramAdapter] = None
_sink: Optional[EventSink] = None
_router: Optional[PlatformRouter] = None

# Per-user semaphore map — limits concurrent orchestration tasks
_USER_SEMAPHORES: Dict[str, asyncio.Semaphore] = {}

def _get_semaphore(channel_id: str) -> asyncio.Semaphore:
    cfg = Config.get()
    limit = cfg.max_concurrent_orchestrations_per_user or 2
    if channel_id not in _USER_SEMAPHORES:
        _USER_SEMAPHORES[channel_id] = asyncio.Semaphore(limit)
    return _USER_SEMAPHORES[channel_id]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _extract_response_text(resp: dict) -> str:
    if not resp:
        return ""
    if "output" in resp:
        return resp["output"] or ""
    try:
        choices = resp.get("choices")
        if choices and isinstance(choices, list):
            first = choices[0]
            msg = first.get("message") if isinstance(first, dict) else None
            if msg and isinstance(msg, dict):
                return msg.get("content") or ""
            text = first.get("text")
            if text:
                return text
    except Exception as exc:
        logger.warning("extract_response_failed", error=str(exc))
    for k in ("output_text", "text"):
        if isinstance(resp.get(k), str):
            return resp[k]
    return str(resp)


# ---------------------------------------------------------------------------
def _is_fatal_provider_error(err_str: str) -> bool:
    """Return True for errors that should skip to the next provider immediately.

    Includes tool_use_failed: Groq's function-calling schema rejection is a
    capability/context issue for this turn — retrying burns TPM quota.
    """
    e = err_str.lower()
    return (
        "429" in e
        or "quota" in e
        or "exhausted" in e
        or "rate limit" in e
        or "401" in e
        or "unauthorized" in e
        or "invalid api key" in e
        or "tool_use_failed" in e
        or "no api key available" in e
    )


def _is_retryable_provider_error(exc: Exception | None, err_str: str) -> bool:
    """Return True for transient errors worth retrying (timeout, 403, network)."""
    if isinstance(exc, asyncio.TimeoutError):
        return True
    e = err_str.lower()
    return (
        "403" in e
        or "access denied" in e
        or "network" in e
        or "timed out" in e
        or "timeout" in e
        or "connection" in e
    )


# /start
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Global auth gate — runs before every handler via TypeHandler group=-1
# ---------------------------------------------------------------------------

async def _auth_gate(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Reject all updates from non-owner users before any handler runs.

    PTB 21+ supports negative group numbers. TypeHandler at group=-1 fires
    first. Raising ApplicationHandlerStop prevents every subsequent handler
    from receiving the update — no per-command guard needed.
    """
    owner = os.environ.get("OWNER_USER_ID", "").strip()
    if not owner:
        logger.warning(
            "auth_gate_disabled",
            message="OWNER_USER_ID not set — all Telegram users can access the bot. "
                    "Set OWNER_USER_ID in .env for single-user mode.",
        )
        return  # open mode (development only)
    uid = update.effective_user.id if update.effective_user else None
    if uid is None or str(uid) != owner:
        # Silent to the sender (do not reveal the bot exists), but logged
        # so the real owner can diagnose a misconfigured OWNER_USER_ID
        # (typo, stray whitespace/quotes from .env, wrong ID copied).
        logger.warning(
            "auth_gate_rejected",
            configured_owner=owner,
            sender_id=uid,
            sender_username=getattr(update.effective_user, "username", None),
        )
        from telegram.ext import ApplicationHandlerStop
        raise ApplicationHandlerStop


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    cfg = Config.get()
    from src.db.key_store import init_db, upsert_user
    await init_db()
    tg_user = update.effective_user
    user_id = await upsert_user("telegram", str(tg_user.id), tg_user.username)

    await update.message.reply_html(
        f"<b>{cfg.bot_name}</b> is active.\n"
        "Send /help for commands.\n\n"
        "To start background monitoring: /watch system\n"
        "To add API keys: /addkey provider:\"key\""
    )

    # Start background agent manager if not already running
    from src.agents.background.manager import BackgroundAgentManager
    manager = BackgroundAgentManager.get()
    if manager is not None:
        # Register a sentinel system watcher for this user if they don't have one already
        existing = await manager.list_for_user(user_id)
        has_system = any(a.watcher_type == "system" for a in existing if a.enabled)
        if not has_system:
            pass  # User must explicitly call /watch system — don't auto-register


# ---------------------------------------------------------------------------
# /help
# ---------------------------------------------------------------------------

async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    cfg = Config.get()
    text = (
        f"<b>{cfg.bot_name} commands</b>\n\n"
        "/start — Initialize bot\n"
        "/help — Show this message\n"
        "/stop — Stop current task/research\n"
        "/addkey provider:\"key\" — Add an API key\n"
        "/status — Key status, model, active agents\n"
        "/providers — All providers and connectivity status\n"
        "/reload — Reload config + tool registry from disk (owner only)\n"
        "/memory — List stored memories and skills\n"
        "/pinmemory &lt;key&gt; — Always inject this memory (max 5)\n"
        "/unpinmemory &lt;key&gt; — Remove from always-injected list\n"
        "/deletememory key — Delete a memory entry\n"
        "/delete_me — Delete all your stored data\n\n"
        "<b>Workspace</b>\n"
        "/files — List files in ~/.rika/shared\n"
        "/files 4 — List with depth 4\n"
        "/cleanworkspace — Wipe workspace contents\n"
        "/cmdhistory — Recent command execution log\n\n"
        "<b>Background monitoring</b>\n"
        "/autowatch &lt;goal&gt; — AI decides what to monitor and sets it up\n"
        "/watch system — Monitor CPU / memory / disk\n"
        "/watch process &lt;name&gt; — Watch if a process is running\n"
        "/watch url &lt;url&gt; — HTTP health check\n"
        "/watch port &lt;port&gt; — TCP port availability\n"
        "/watch log &lt;path&gt; &lt;pattern&gt; — Watch log for regex\n"
        "/watch cron &lt;interval&gt; &lt;task&gt; — Scheduled AI task\n"
        "/watchers — List active background agents\n"
        "/stopwatch &lt;id&gt; — Stop a background agent\n"
        "/wakelog — Recent wake events\n\n"
        "To add API keys, send a message with provider:key pairs:\n"
        "<code>openrouter:\"sk-...\"  groq:\"gsk_...\"  google:AIza...</code>\n"
        "For Ollama (local): enable in config.json, no key needed.\n"
        "For G4F (free): pip install g4f, then enable in config.json.\n\n"
        "<b>Command security:</b> dangerous commands are blocked automatically.\n"
        "Prefix with <code>CONFIRM: </code> to override a warning-level block."
    )
    await update.message.reply_html(text)


# ---------------------------------------------------------------------------
# /addkey
# ---------------------------------------------------------------------------

async def addkey_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    from src.db.key_store import add_api_key, init_db, upsert_user
    from src.utils.parse_keys import parse_keys

    raw = " ".join(context.args) if context.args else (update.message.text or "")
    if raw.startswith("/addkey"):
        parts = raw.split(None, 1)
        raw = parts[1] if len(parts) > 1 else ""

    keys = parse_keys(raw)
    if not keys:
        allowed = ["gemini", "google", "openrouter", "groq", "anthropic", "openai", "nvidia", "vercel", "ollama", "g4f"]
        tokens = [t.strip() for t in raw.replace(",", " ").split() if t.strip()]
        i = 0
        while i + 1 < len(tokens):
            prov = tokens[i].lower()
            val = tokens[i + 1].strip('"\'')
            if prov in allowed:
                keys[prov] = val
            i += 2

    if not keys:
        await update.message.reply_text('Usage: /addkey provider:"key" — e.g. /addkey groq:"gsk_..."')
        return

    await init_db()
    tg_user = update.effective_user
    user_id = await upsert_user("telegram", str(tg_user.id), tg_user.username)
    pool = get_pool()
    results = []
    for provider, raw_key in keys.items():
        try:
            kid = await add_api_key(user_id, provider, raw_key)
            try:
                ok = await pool.get_healthy_key(user_id, provider)
                status = "valid" if ok else "invalid"
            except Exception as exc:
                status = f"validation error: {exc}"
            results.append(f"{provider}: stored (id={kid}), {status}")
        except Exception as exc:
            results.append(f"{provider}: error — {exc}")
    await update.message.reply_text("\n".join(results))


# ---------------------------------------------------------------------------
# /delete_me
# ---------------------------------------------------------------------------

async def delete_me_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = [[
        InlineKeyboardButton("Yes, delete everything", callback_data="confirm_delete"),
        InlineKeyboardButton("Cancel", callback_data="cancel_delete"),
    ]]
    await update.message.reply_text(
        "Are you sure you want to delete all your data? This cannot be undone.",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


# ---------------------------------------------------------------------------
# /status
# ---------------------------------------------------------------------------

async def status_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info("status_handler_called", user_id=update.effective_user.id)
    from src.db.key_store import list_api_keys, upsert_user
    tg_user = update.effective_user
    cfg = Config.get()
    logger.info("status_handler_config_loaded")
    user_id = await upsert_user("telegram", str(tg_user.id), tg_user.username)
    logger.info("status_handler_user_upserted", user_id=user_id)
    keys = await list_api_keys(user_id)
    logger.info("status_handler_keys_loaded", count=len(keys))
    active = [k for k in keys if not k["is_blacklisted"]]
    env_count = sum(1 for p in ["GEMINI", "GROQ", "OPENROUTER"] if os.environ.get(f"{p}_API_KEY"))

    from src.agents.background.manager import BackgroundAgentManager
    manager = BackgroundAgentManager.get()
    bg_count = 0
    if manager:
        agents = await manager.list_for_user(user_id)
        bg_count = sum(1 for a in agents if a.enabled)
    logger.info("status_handler_sending_response")

    msg = (
        f"<b>Status — {cfg.bot_name}</b>\n\n"
        f"Your Telegram ID: <code>{tg_user.id}</code>\n"
        f"DB keys: {len(keys)} ({len(active)} active)\n"
        f"Env keys: {env_count}\n"
        f"Current model: <code>{cfg.default_model}</code>\n"
        f"Background agents: {bg_count} active"
    )
    await update.message.reply_html(msg)
    logger.info("status_handler_response_sent")


# ---------------------------------------------------------------------------
# /memory, /deletememory
# ---------------------------------------------------------------------------

async def memory_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    from src.db.chat_store import list_rika_memories
    from src.db.key_store import upsert_user
    tg_user = update.effective_user
    user_id = await upsert_user("telegram", str(tg_user.id), tg_user.username)
    entries = await list_rika_memories(user_id)
    if not entries:
        await update.message.reply_text("No memories stored yet.")
        return
    lines = ["<b>Stored memories and skills:</b>\n"]
    for e in entries:
        tag = "M" if e["type"] == "memory" else "S"
        pin = " [pinned]" if e.get("pinned") else ""
        acc = f" (used {e.get('access_count', 0)}x)" if e.get("access_count", 0) > 0 else ""
        lines.append(f"[{tag}]{pin} <b>{_escape_html(e['key'])}</b>: {_escape_html(str(e['value'])[:80])}{acc}")
    await update.message.reply_html("\n".join(lines))


async def deletememory_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    from src.db.chat_store import delete_rika_memory
    from src.db.key_store import upsert_user
    key = " ".join(context.args).strip() if context.args else ""
    if not key:
        await update.message.reply_text("Usage: /deletememory <key>")
        return
    tg_user = update.effective_user
    user_id = await upsert_user("telegram", str(tg_user.id), tg_user.username)
    for mem_type in ("memory", "skill"):
        await delete_rika_memory(user_id, key, mem_type)
    await update.message.reply_text(f"Deleted memory entry: {key}")


# ---------------------------------------------------------------------------
# /watch — register a background monitoring agent
# ---------------------------------------------------------------------------

async def watch_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    from src.agents.agent_models import BackgroundAgentConfig
    from src.agents.background.manager import BackgroundAgentManager
    from src.db.key_store import upsert_user

    args = context.args or []
    if not args:
        await update.message.reply_html(
            "<b>Usage examples:</b>\n"
            "/watch system — monitor CPU/RAM/disk\n"
            "/watch process nginx — watch if nginx is running\n"
            "/watch url https://example.com — HTTP health check\n"
            "/watch port 8080 — check if port is open\n"
            "/watch log /var/log/syslog ERROR — watch for pattern\n"
            "/watch cron 10m check disk usage and report\n"
        )
        return

    tg_user = update.effective_user
    chat_id = update.effective_chat.id
    user_id = await upsert_user("telegram", str(tg_user.id), tg_user.username)
    manager = BackgroundAgentManager.get()
    if manager is None:
        await update.message.reply_text("Error: background agent manager is not running.")
        return

    watcher_type = args[0].lower()
    rest = args[1:]
    agent_id = str(uuid.uuid4())[:8]

    try:
        if watcher_type == "system":
            cfg_dict: dict = {}
            # Optional threshold overrides: cpu:90 mem:95 disk:85
            for token in rest:
                if ":" in token:
                    k, v = token.split(":", 1)
                    cfg_dict[f"{k.strip()}_threshold"] = float(v.strip())
            agent_cfg = BackgroundAgentConfig(
                id=f"sys_{agent_id}",
                user_id=user_id,
                chat_id=chat_id,
                watcher_type="system",
                name="System Monitor",
                description="Monitor system health: CPU load, memory usage, and disk space.",
                config=cfg_dict,
                interval_seconds=120,
            )
            description = "System health monitor (CPU / RAM / disk)"

        elif watcher_type == "process":
            if not rest:
                await update.message.reply_text("Usage: /watch process <process_name>")
                return
            proc = rest[0]
            interval = 30
            agent_cfg = BackgroundAgentConfig(
                id=f"proc_{agent_id}",
                user_id=user_id,
                chat_id=chat_id,
                watcher_type="process",
                name=f"Process: {proc}",
                description=f"Alert me if the '{proc}' process stops running.",
                config={"process_name": proc},
                interval_seconds=interval,
            )
            description = f"Process watcher: {proc}"

        elif watcher_type == "url":
            if not rest:
                await update.message.reply_text("Usage: /watch url <url> [expect:<status>]")
                return
            url = rest[0]
            expected = 200
            for token in rest[1:]:
                if token.startswith("expect:"):
                    expected = int(token.split(":")[1])
            agent_cfg = BackgroundAgentConfig(
                id=f"url_{agent_id}",
                user_id=user_id,
                chat_id=chat_id,
                watcher_type="url",
                name=f"URL: {url[:40]}",
                description=f"Alert me when {url} becomes unreachable (expected HTTP {expected}).",
                config={"url": url, "expected_status": expected},
                interval_seconds=60,
            )
            description = f"URL health check: {url}"

        elif watcher_type == "port":
            if not rest:
                await update.message.reply_text("Usage: /watch port <port> [host:<host>]")
                return
            port = int(rest[0])
            host = "localhost"
            for token in rest[1:]:
                if token.startswith("host:"):
                    host = token.split(":", 1)[1]
            agent_cfg = BackgroundAgentConfig(
                id=f"port_{agent_id}",
                user_id=user_id,
                chat_id=chat_id,
                watcher_type="port",
                name=f"Port {port} on {host}",
                description=f"Alert me if port {port} on {host} closes.",
                config={"host": host, "port": port},
                interval_seconds=60,
            )
            description = f"Port watcher: {host}:{port}"

        elif watcher_type == "log":
            if len(rest) < 2:
                await update.message.reply_text("Usage: /watch log <file_path> <pattern>")
                return
            file_path = rest[0]
            pattern = " ".join(rest[1:])
            agent_cfg = BackgroundAgentConfig(
                id=f"log_{agent_id}",
                user_id=user_id,
                chat_id=chat_id,
                watcher_type="log",
                name=f"Log: {os.path.basename(file_path)}",
                description=f"Alert me when pattern '{pattern}' appears in {file_path}.",
                config={"file_path": file_path, "pattern": pattern},
                interval_seconds=30,
            )
            description = f"Log watcher: {file_path} pattern={pattern}"

        elif watcher_type == "cron":
            if len(rest) < 2:
                await update.message.reply_text(
                    "Usage: /watch cron <interval> <task description>\n"
                    "Example: /watch cron 1h check disk usage and warn if above 80%"
                )
                return
            interval_str = rest[0].lower()
            task_desc = " ".join(rest[1:])
            # Parse interval: 30s, 5m, 1h, 2h
            interval_seconds = _parse_interval(interval_str)
            agent_cfg = BackgroundAgentConfig(
                id=f"cron_{agent_id}",
                user_id=user_id,
                chat_id=chat_id,
                watcher_type="cron",
                name=f"Cron: {task_desc[:30]}",
                description=task_desc,
                config={"task_description": task_desc},
                interval_seconds=interval_seconds,
            )
            description = f"Scheduled task every {interval_str}: {task_desc[:60]}"
        elif watcher_type == "script":
            if not rest:
                await update.message.reply_text(
                    "Usage: /watch script <script_path> [cooldown:<seconds>]\n"
                    "The script should print ALERT: <msg> on problems, exit 0 if fine."
                )
                return
            script_path = rest[0]
            cooldown = 120
            for token in rest[1:]:
                if token.startswith("cooldown:"):
                    cooldown = int(token.split(":")[1])
            agent_cfg = BackgroundAgentConfig(
                id=f"scr_{agent_id}",
                user_id=user_id,
                chat_id=chat_id,
                watcher_type="script",
                name=f"Script: {os.path.basename(script_path)}",
                description=f"Run {script_path} and alert on non-zero exit or ALERT: output.",
                config={"script_path": script_path, "cooldown_seconds": cooldown},
                interval_seconds=60,
            )
            description = f"Script watcher: {script_path}"

        else:
            await update.message.reply_text(
                f"Unknown watcher type: '{watcher_type}'. "
                "Use: system, process, url, port, log, cron, script"
            )
            return

        await manager.register(agent_cfg)
        await update.message.reply_html(
            f"Background agent started.\n\n"
            f"<b>ID:</b> <code>{agent_cfg.id}</code>\n"
            f"<b>Type:</b> {watcher_type}\n"
            f"<b>Task:</b> {_escape_html(description)}\n\n"
            f"Use /stopwatch {agent_cfg.id} to stop it."
        )
    except ValueError as exc:
        await update.message.reply_text(f"Configuration error: {exc}")
    except Exception as exc:
        logger.exception("watch_handler_failed", error=str(exc))
        await update.message.reply_text(f"Failed to start background agent: {exc}")


def _parse_interval(s: str) -> int:
    """Parse interval strings like '30s', '5m', '1h' into seconds."""
    s = s.strip().lower()
    if s.endswith("s"):
        return max(10, int(s[:-1]))
    if s.endswith("m"):
        return max(60, int(s[:-1]) * 60)
    if s.endswith("h"):
        return max(300, int(s[:-1]) * 3600)
    return max(60, int(s))


# ---------------------------------------------------------------------------
# /watchers — list background agents
# ---------------------------------------------------------------------------

async def watchers_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    from src.agents.background.manager import BackgroundAgentManager
    from src.db.key_store import upsert_user
    tg_user = update.effective_user
    user_id = await upsert_user("telegram", str(tg_user.id), tg_user.username)
    manager = BackgroundAgentManager.get()
    if manager is None:
        await update.message.reply_text("Background agent manager not running.")
        return
    agents = await manager.list_for_user(user_id)
    if not agents:
        await update.message.reply_text("No background agents registered.")
        return
    lines = ["<b>Your background agents:</b>\n"]
    for a in agents:
        status = "active" if a.enabled else "stopped"
        lines.append(
            f"<code>{a.id}</code> [{status}] <b>{_escape_html(a.name)}</b>\n"
            f"  Type: {a.watcher_type} | Every {a.interval_seconds}s"
        )
    lines.append(f"\n/stopwatch &lt;id&gt; to stop one")
    await update.message.reply_html("\n".join(lines))


# ---------------------------------------------------------------------------
# /stopwatch — stop a background agent
# ---------------------------------------------------------------------------

async def stopwatch_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    from src.agents.background.manager import BackgroundAgentManager
    agent_id = " ".join(context.args).strip() if context.args else ""
    if not agent_id:
        await update.message.reply_text("Usage: /stopwatch <agent_id>")
        return
    manager = BackgroundAgentManager.get()
    if manager is None:
        await update.message.reply_text("Background agent manager not running.")
        return
    stopped = await manager.stop_agent(agent_id)
    if stopped:
        await update.message.reply_text(f"Agent {agent_id} stopped.")
    else:
        await update.message.reply_text(f"No active agent with id: {agent_id}")


# ---------------------------------------------------------------------------
# /wakelog — show recent wake events
# ---------------------------------------------------------------------------

async def wakelog_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    from src.db.background_store import get_wake_events
    from src.db.key_store import upsert_user
    tg_user = update.effective_user
    user_id = await upsert_user("telegram", str(tg_user.id), tg_user.username)
    limit = 10
    if context.args:
        try:
            limit = int(context.args[0])
        except ValueError:
            pass
    events = await get_wake_events(user_id, limit)
    if not events:
        await update.message.reply_text("No wake events recorded yet.")
        return
    lines = [f"<b>Last {len(events)} wake events:</b>\n"]
    for e in events:
        sev = e["severity"].upper()
        at = e["at"][:16] if e["at"] else "?"
        analysis = (e["analysis"] or "")[:100]
        lines.append(
            f"[{sev}] <code>{e['agent_id']}</code> — {e['event_type']}\n"
            f"  {at}: {_escape_html(analysis)}"
        )
    await update.message.reply_html("\n\n".join(lines))


# ---------------------------------------------------------------------------
# /broadcast (owner only)
# ---------------------------------------------------------------------------

async def broadcast_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    owner = os.environ.get("OWNER_USER_ID")
    if not owner or str(update.effective_user.id) != str(owner):
        await update.message.reply_text("Unauthorized.")
        return
    text = " ".join(context.args) if context.args else ""
    if not text:
        await update.message.reply_text("Usage: /broadcast <message>")
        return
    from src.db.connection import get_db
    cfg = Config.get()
    async with get_db() as db:
        cur = await db.execute("SELECT platform_user_id FROM users WHERE platform = 'telegram'")
        users = await cur.fetchall()
    count = 0
    for (pu_id,) in users:
        try:
            await context.bot.send_message(
                chat_id=int(pu_id),
                text=f"<b>Broadcast — {cfg.bot_name}:</b>\n\n{text}",
                parse_mode="HTML",
            )
            count += 1
            await asyncio.sleep(0.05)
        except Exception as exc:
            logger.warning("broadcast_send_failed", error=str(exc))
            continue
    await update.message.reply_text(f"Broadcast sent to {count} users.")


# ---------------------------------------------------------------------------
# Callback query (inline buttons)
# ---------------------------------------------------------------------------

async def callback_query_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    from src.db.key_store import delete_user_by_platform
    cfg = Config.get()
    if query.data == "confirm_delete":
        deleted = await delete_user_by_platform("telegram", str(query.from_user.id))
        text = "All your data has been deleted." if deleted else "No data found for your account."
        await query.edit_message_text(text)
    elif query.data == "confirm_cleanws":
        from src.tools.workspace import clean_workspace, get_workspace_path
        ws = get_workspace_path(getattr(cfg, "workspace_path", None))
        msg = clean_workspace(ws)
        await query.edit_message_text(msg)
    else:
        await query.edit_message_text("Cancelled.")


# ---------------------------------------------------------------------------
# Stop task button handler
# ---------------------------------------------------------------------------

async def stop_task_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle stop button click — cancels active task or countdown wait."""
    query = update.callback_query
    channel_id = f"tg:{query.from_user.id}"

    if _adapter is not None:
        _adapter.handle_stop(channel_id)

    await query.answer("Stopping…")
    try:
        await query.edit_message_text("⏹ Task stopped by user.", reply_markup=None)
    except Exception as exc:
        logger.debug("stop_edit_failed", error=str(exc))
    logger.info("task_cancelled_by_user", channel_id=channel_id)




# ---------------------------------------------------------------------------
# Retry-now button handler
# ---------------------------------------------------------------------------

async def retry_now_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle Retry-now button — breaks the active countdown immediately."""
    query = update.callback_query
    data = query.data or ""
    try:
        uid = int(data.split(":", 1)[1])
    except (ValueError, IndexError):
        uid = query.from_user.id

    if _adapter is not None:
        _adapter.handle_retry(f"tg:{uid}")

    await query.answer("Retrying now…")


# ---------------------------------------------------------------------------
# /stop command — cancel current task
# ---------------------------------------------------------------------------

async def stop_command_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /stop command — cancels active task or countdown wait."""
    channel_id = f"tg:{update.effective_user.id}"

    if _adapter is not None:
        _adapter.handle_stop(channel_id)

    logger.info("task_cancelled_by_command", channel_id=channel_id)
    await update.message.reply_text("⏹ Task stopped.")


# ---------------------------------------------------------------------------
# Document handler — fix: no longer mutates update.message.text
# ---------------------------------------------------------------------------

async def document_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    doc = update.message.document
    mime = doc.mime_type or ""
    name = doc.file_name or ""
    text_mimes = ("text/", "json", "javascript", "xml")
    text_exts = (".txt", ".py", ".md", ".json", ".js", ".html", ".css", ".sql", ".sh", ".svg", ".yaml", ".yml")
    is_text = any(s in mime for s in text_mimes) or any(name.lower().endswith(e) for e in text_exts)
    if not is_text:
        await update.message.reply_text("Only text-based files are accepted (py, md, json, sh, etc.)")
        return
    try:
        import io
        f = await doc.get_file()
        buf = io.BytesIO()
        await f.download_to_memory(buf)
        content = buf.getvalue().decode("utf-8", errors="replace")
    except Exception as exc:
        await update.message.reply_text(f"Failed to read file: {exc}")
        return

    # Build a synthetic message text and feed it to the main handler
    # without touching the frozen Update object
    file_msg = f"Analyze this file: {name}\n\nUPLOADED_FILE ({name}):\n---\n{content[:8000]}\n---"
    await _process_message(update, context, override_text=file_msg)


# ---------------------------------------------------------------------------
# Runtime context injection
# ---------------------------------------------------------------------------

def _agent_name(cfg=None) -> str:
    name = os.environ.get("AGENT_NAME", "").strip()
    if name: return name
    if cfg: return cfg.bot_name
    return Config.get().bot_name


def _build_runtime_context(tg_user, cfg) -> str:
    """Build a compact runtime context block injected at the top of every request.
    
    Wrapper around core ContextBuilder for backward compatibility.
    """
    return ContextBuilder.build_runtime_context(tg_user, cfg)


# ---------------------------------------------------------------------------
# Main message handler
# ---------------------------------------------------------------------------

async def key_submission_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (update.message.text or "").strip()
    await _process_message(update, context, override_text=text)


async def _process_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    override_text: Optional[str] = None,
) -> None:
    from src.db.chat_store import (
        add_chat_message,
        get_chat_history,
        get_summary_data,
        count_messages_since,
    )
    from src.db.key_store import add_api_key, init_db, list_api_keys, upsert_user
    from src.live.live_bubble import LiveBubble
    from src.utils.parse_keys import parse_keys

    text = override_text or (update.message.text or "").strip()
    logger.info("incoming_message", user_id=update.effective_user.id, length=len(text))

    await init_db()
    tg_user = update.effective_user
    cfg = Config.get()  # FIXED: cfg loaded before any branch uses it
    user_id = await upsert_user("telegram", str(tg_user.id), tg_user.username)
    keys_list = await list_api_keys(user_id)

    env_keys: list = [
        p for p in ["GEMINI", "GROQ", "OPENROUTER"]
        if os.environ.get(f"{p}_API_KEY")
    ]

    # Log user message
    await add_chat_message(user_id, "user", text)

    # Try key submission first
    parsed_keys = parse_keys(text)
    if parsed_keys:
        await _handle_key_submission(update, context, user_id, parsed_keys, cfg)
        return

    # No keys — check availability
    if not keys_list and not env_keys:
        await update.message.reply_text(
            f"No API keys stored yet.\n"
            f"Add one: /addkey groq:\"gsk_...\" or /addkey openrouter:\"sk-...\""
        )
        return

    # Load context
    pool = get_pool()
    summary_data = await get_summary_data(user_id)
    summary = summary_data["summary"] if summary_data else None
    last_msg_id = summary_data["last_msg_id"] if summary_data else 0
    history = await get_chat_history(user_id, limit=cfg.max_context_messages, after_id=last_msg_id)

    context_parts = []
    context_parts.append(_build_runtime_context(tg_user, cfg))
    if summary:
        context_parts.append(f"[Earlier context summary]\n{summary}")
    for m in history[:-1]:
        context_parts.append(f"{m['role']}: {m['content']}")
    context_parts.append(f"user: {text}")
    context_str = "\n".join(context_parts)

    # Complexity check — LLM-based routing
    is_complex = await _classify_complexity(text, cfg, pool, user_id)
    logger.debug("complexity_routing", is_complex=is_complex, text_len=len(text))

    if not is_complex:
        await _handle_direct_reply(update, context, user_id, text, context_str, history, summary, cfg, pool, last_msg_id)
    else:
        channel_id = f"tg:{update.effective_chat.id}"

        # Create stop button with proper InlineKeyboardMarkup
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("⏹ Stop", callback_data=f"stop_task:{tg_user.id}")]])
        sent = await update.message.reply_text(
            f"{_agent_name(cfg)} is thinking...\n\n_This may take up to 60 seconds. Click Stop to cancel._",
            reply_markup=keyboard,
            parse_mode="Markdown"
        )
        handle = f"tg:{sent.chat_id}:{sent.message_id}"
        sem = _get_semaphore(channel_id)

        if _adapter is not None:
            _adapter.clear_cancel(channel_id)
            _adapter.init_countdown_events(channel_id)

        # Create and track task
        task = asyncio.create_task(
            _run_orchestration_guarded(
                sem, channel_id, handle,
                user_id, context_str, text, history, summary, cfg, keyboard,
            )
        )
        if _adapter is not None:
            _adapter.track_task(channel_id, task)

        # Clean up task tracking when done
        def cleanup(t):
            if _adapter is not None:
                _adapter.untrack_task(channel_id)
        task.add_done_callback(cleanup)


async def _classify_complexity(text: str, cfg: Config, pool, user_id: int) -> bool:
    """Classify whether a message requires tools / orchestration.
    
    Wrapper around core classify_complexity for backward compatibility.
    """
    return await classify_complexity(text, cfg, pool, user_id)


async def _handle_direct_reply(
    update, context, user_id, text, context_str, history, summary, cfg, pool, last_msg_id
) -> None:
    logger.info("direct_reply_handler", user_id=user_id, text_len=len(text), context_len=len(context_str))
    from src.db.chat_store import add_chat_message
    payload = {
        "model": cfg.default_model,
        "messages": [
            {"role": "system", "content": cfg.system_prompt},
            {"role": "user", "content": context_str},
        ],
    }
    # Filter priorities to only providers with valid keys
    all_priorities = cfg.default_provider_priority or ["gemini", "groq", "openrouter"]
    available = await pool.get_available_providers(user_id)
    priorities = [p for p in all_priorities if p in available]
    
    if not priorities:
        logger.error("direct_reply_no_available_providers", available=available, configured=all_priorities)
        await update.message.reply_text("No API keys configured. Add one with /addkey provider:\"key\"")
        return
    
    logger.info("direct_reply_starting", user_id=user_id, priorities=priorities, available=available)
    reply = None
    last_error = None
    
    # Try each provider in priority order
    for p in priorities:
        # Check cancel via adapter
        if _adapter is not None and _adapter.is_cancelled(f"tg:{update.effective_user.id}"):
            logger.info("direct_reply_cancelled_before_provider", provider=p)
            if _adapter is not None:
                _adapter.clear_cancel(f"tg:{update.effective_user.id}")
            await update.message.reply_text(f"{_agent_name(cfg)} task cancelled.")
            return
        
        try:
            logger.info("direct_reply_try_provider", provider=p, user_id=user_id)
            # 60s timeout for HTTP requests (prevents hanging on bad connections/keys)
            resp = await asyncio.wait_for(
                pool.request_with_key(user_id, p, payload),
                timeout=60.0
            )
            reply = resp.get("output", "").strip()
            if reply:
                logger.info("direct_reply_got_response", provider=p, reply_len=len(reply), first_50=reply[:50])
                break
        except asyncio.TimeoutError:
            last_error = f"Provider {p} timed out"
            logger.warning("direct_reply_timeout", provider=p)
            _dr_retries = getattr(cfg, "provider_max_retries", 2)
            _dr_delay   = getattr(cfg, "provider_retry_delay", 2.0)
            for _r in range(_dr_retries):
                await asyncio.sleep(_dr_delay * (2 ** _r))
                try:
                    resp = await asyncio.wait_for(
                        pool.request_with_key(user_id, p, payload),
                        timeout=60.0,
                    )
                    reply = resp.get("output", "").strip()
                    if reply:
                        break
                except Exception as exc:
                    logger.warning("direct_reply_provider_error", provider=p, error=str(exc))
            if reply:
                break
            continue
        except Exception as exc:
            last_error = str(exc)
            if _is_retryable_provider_error(exc, last_error):
                _dr_retries = getattr(cfg, "provider_max_retries", 2)
                _dr_delay   = getattr(cfg, "provider_retry_delay", 2.0)
                logger.warning("direct_reply_transient", provider=p, error=last_error)
                for _r in range(_dr_retries):
                    await asyncio.sleep(_dr_delay * (2 ** _r))
                    try:
                        resp = await asyncio.wait_for(
                            pool.request_with_key(user_id, p, payload),
                            timeout=60.0,
                        )
                        reply = resp.get("output", "").strip()
                        if reply:
                            break
                    except Exception as exc:
                        logger.warning("direct_reply_retry_failed", provider=p, error=str(exc))
                if reply:
                    break
            else:
                logger.warning("direct_reply_provider_failed", provider=p, error=last_error)
            continue
    
    if not reply:
        logger.error("direct_reply_no_response", priorities=priorities, available=available, last_error=last_error)
        await update.message.reply_text(_friendly_provider_error(last_error))
        return

    # If LLM spontaneously tried to use a tool in "simple" mode, re-route
    # Detect any text-protocol tool invocation the LLM may have output
    _TOOL_PROTO_RE = re.compile(
        r"(?:TOOL:|WEB_SEARCH:|WIKIPEDIA_SEARCH:|RUN_PYTHON:|RUN_SHELL_COMMAND:"
        r"|CURL:|LIST_WORKSPACE:|READ_FILE:|WRITE_FILE:|SAVE_MEMORY:"
        r"|GET_MEMORIES:|SAVE_SKILL:|USE_SKILL:|DELEGATE_TASK:|SEND_FILE:)",
        re.IGNORECASE,
    )
    if _TOOL_PROTO_RE.search(reply):
        logger.info("direct_reply_reroute_to_orchestration")
        sent = await update.message.reply_text(f"{_agent_name(cfg)} is processing...")
        channel_id = f"tg:{update.effective_chat.id}"
        handle = f"tg:{sent.chat_id}:{sent.message_id}"
        sem = _get_semaphore(channel_id)
        asyncio.create_task(
            _run_orchestration_guarded(
                sem, channel_id, handle,
                user_id, context_str, text, history, summary, cfg
            )
        )
        return

    await add_chat_message(user_id, "assistant", reply)
    logger.info("direct_reply_sending", reply_len=len(reply))
    try:
        await update.message.reply_html(reply)
        logger.info("direct_reply_sent_success")
    except Exception as exc:
        logger.exception("direct_reply_send_failed", error=str(exc))


async def _run_orchestration_guarded(
    sem: asyncio.Semaphore, channel_id: str, handle: str, user_id: int,
    context_str, original_text, history, summary, cfg, keyboard=None,
) -> None:
    if sem.locked() and sem._value == 0:  # type: ignore[attr-defined]
        if _sink is not None:
            try:
                await _sink.edit_message(
                    handle,
                    "Another task is already running. Please wait.",
                    reply_markup=keyboard,
                )
            except Exception as exc:
                logger.debug("semaphore_busy_edit_failed", error=str(exc))
        return
    async with sem:
        try:
            await run_orchestration_background(
                channel_id, handle, user_id,
                context_str, original_text, history, summary, keyboard,
            )
        finally:
            _USER_SEMAPHORES.pop(channel_id, None)


async def _handle_key_submission(update, context, user_id, keys, cfg) -> None:
    from src.db.key_store import add_api_key
    from src.live.live_bubble import LiveBubble
    
    sent = await update.message.reply_text(f"Validating keys...")
    bubble = LiveBubble(throttle_ms=800)

    async def flush(text: str) -> None:
        try:
            await context.bot.edit_message_text(
                chat_id=sent.chat_id, message_id=sent.message_id, text=text
            )
        except Exception as exc:
            logger.debug("key_submission_flush_failed", error=str(exc))

    await bubble.start(flush)
    pool = get_pool()
    lines = []
    for provider, raw_key in keys.items():
        bubble.update(provider, "storing...")
        try:
            kid = await add_api_key(user_id, provider, raw_key)
            bubble.update(provider, "validating...")
            ok = await pool.get_healthy_key(user_id, provider)
            status = "valid" if ok else "invalid"
            lines.append(f"{provider}: stored (id={kid}), {status}")
            bubble.update(provider, status)
        except Exception as exc:
            lines.append(f"{provider}: error — {exc}")
            bubble.update(provider, f"error: {exc}")
    await bubble.stop()
    try:
        await context.bot.edit_message_text(
            chat_id=sent.chat_id,
            message_id=sent.message_id,
            text="Key submission results:\n" + "\n".join(lines),
        )
    except Exception as exc:
        logger.debug("key_submission_edit_failed", error=str(exc))


# ---------------------------------------------------------------------------
# Orchestration background loop
# ---------------------------------------------------------------------------

_CORRECTION_WARNING = (
    "[SYSTEM — DO NOT REPEAT THIS TO THE USER]\n"
    "You produced text output without a tool call. This is not allowed.\n"
    "You MUST respond with a tool call. Either:\n"
    '  1. end_thinking(message=\"...\") to deliver your final answer\n'
    "  2. Any other tool to continue processing\n\n"
    "Your unsent text (copy into end_thinking if it is your final answer):\n"
)



def _friendly_provider_error(last_error: str | None) -> str:
    """Map a raw provider error to a clean user-facing message."""
    err = (last_error or "").lower()
    if "403" in err or "access denied" in err or "network" in err:
        return (
            "Couldn\'t reach the AI provider — this is usually a network or VPN issue.\n"
            "Try again in a moment or check /status."
        )
    if "429" in err or "quota" in err or "rate limit" in err or "exhausted" in err:
        return (
            "All providers are currently rate-limited or over quota.\n"
            "Wait a few minutes, then try again. Check /status for key health."
        )
    if "401" in err or "unauthorized" in err or "no.*api key" in err or "no api key" in err:
        return (
            "No working API key found for any provider.\n"
            "Add a key with /addkey or check existing ones with /status."
        )
    if "timeout" in err or "timed out" in err:
        return (
            "The AI provider took too long to respond.\n"
            "Try again — if it keeps happening check /status."
        )
    return (
        "All AI providers failed to respond right now.\n"
        "Check /status for details, or try again in a moment."
    )


async def _wait_event(event: asyncio.Event) -> None:
    await event.wait()



import re as _re

_LEGACY_TOOL_LINE_RE = _re.compile(
    r'^[ \t]*TOOL:\s*[\w_]+\s*\|.*$',
    _re.IGNORECASE | _re.MULTILINE,
)
_FUNCTION_TAG_RE = _re.compile(
    r'<function[=\[].*?(?:</function>|$)',
    _re.IGNORECASE | _re.DOTALL,
)
_END_THINKING_MSG_RE = _re.compile(
    r'<function[=\[]end_thinking[ >]\s*\{.*?"message"\s*:\s*"((?:[^"\\\\]|\\\\.)*)"',
    _re.IGNORECASE | _re.DOTALL,
)


def _strip_legacy_tool_syntax(text: str) -> str:
    """Strip leaked non-JSON tool-call syntax from model output.

    Covers two formats that leak when function calling is unavailable
    (e.g. tool schema cap dropped to 0):
      - Legacy text protocol:  TOOL: name | KEY: value  (any key)
      - Llama native tags:     <function=name>{...}</function>
                                or an unterminated <function=name>{...}
                                running to end of string.

    Special case: if the leaked block is specifically
    <function=end_thinking>{\"message\": \"...\"} -- the most common
    leak -- extract the message text instead of discarding it, since it
    usually contains the model's real (correct) answer.
    """
    if not text:
        return text

    extracted = _END_THINKING_MSG_RE.search(text)

    cleaned = _FUNCTION_TAG_RE.sub('', text)
    cleaned = _LEGACY_TOOL_LINE_RE.sub('', cleaned)
    cleaned = _re.sub(r'\n{3,}', '\n\n', cleaned).strip()

    if not cleaned and extracted:
        msg = extracted.group(1)
        msg = msg.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
        return msg.strip()

    return cleaned





async def run_orchestration_background(
    channel_id: str, handle: str, user_id: int,
    context_str: str, original_text: str,
    history: list, summary: Optional[str], keyboard=None,
    *,
    sink: Optional[EventSink] = None,
) -> None:
    from src.live.live_bubble import LiveBubble

    sink = sink or _sink
    cfg = Config.get()
    pool = get_pool()
    bubble = LiveBubble(throttle_ms=cfg.live_bubble_throttle_ms)

    async def flush(text: str) -> None:
        if sink is not None:
            try:
                await sink.edit_message(handle, text, reply_markup=keyboard)
            except Exception as exc:
                logger.debug("bubble_flush_failed", error=str(exc))

    await bubble.start(flush)

    # Semantic memory fragments
    from src.db.vector_store import vector_store
    fragments = await vector_store.search_memories(user_id, original_text, limit=3)
    fragment_str = ""
    if fragments:
        fragment_str = "\n\nRELEVANT PAST CONTEXT:\n" + "\n".join(
            f"- {f['text']}" for f in fragments
        )

    thought_history = [
        {"role": "system", "content": cfg.system_prompt},
        {
            "role": "user",
            "content": f"Request: {original_text}{fragment_str}\n\nContext:\n{context_str}",
        },
    ]

    agent_results: dict = {}
    narrative_chunks: list = []
    priorities = cfg.default_provider_priority or ["gemini", "groq", "openrouter"]
    _final_message: str | None = None
    _no_tool_corr: int = 0
    _provider_retry: int = 0
    _pending_content: str | None = None

    pool.reset_tool_caps()

    try:
        max_turns = getattr(cfg, "max_turns", 20)
        turn = 0

        while turn < max_turns:
            if sink is not None and sink.is_cancelled(channel_id):
                await flush(f"{_agent_name(cfg)} task stopped by user.")
                if sink is not None:
                    sink.clear_cancel(channel_id)
                return

            bubble.update("Thinking", f"turn {turn + 1}...")

            from src.tools.schemas import get_all_schemas
            tool_schemas = get_all_schemas()
            openai_schemas = tool_schemas

            model_map = {
                "groq": getattr(cfg, "groq_model", "llama-3.3-70b-versatile"),
                "openrouter": getattr(cfg, "openrouter_model", "google/gemini-2.0-flash-001"),
                "gemini": getattr(cfg, "gemini_model", "gemini-2.0-flash"),
                "ollama": getattr(cfg, "ollama_model", "llama3.2"),
                "g4f": getattr(cfg, "g4f_model", "MiniMaxAI/MiniMax-M2.5"),
            }

            resp, last_error = await _request_from_providers(
                pool, priorities, model_map, cfg, thought_history,
                openai_schemas, channel_id, sink, flush, _agent_name(cfg), user_id,
            )

            if not resp:
                _provider_retry += 1
                logger.warning("all_providers_failed_retrying", turn=turn,
                               retry=_provider_retry, last_error=last_error)

                if _provider_retry >= 3 and _pending_content is not None:
                    logger.warning("orchestration_same_turn_retry_limit", turn=turn,
                                   has_pending=_pending_content is not None)
                    await bubble.stop()
                    if _pending_content:
                        _final_message = _pending_content
                    else:
                        _final_message = ('I ran into repeated provider errors and could not finish this task. ' + _friendly_provider_error(last_error))
                    break

                await bubble.stop()
                if sink is not None:
                    action = await sink.show_countdown(
                        handle, wait_seconds=30, attempt=_provider_retry,
                    )
                    if action == "stop":
                        sink.clear_cancel(channel_id)
                        return
                await bubble.start(flush)
                bubble.update("Thinking", f"turn {turn + 1} (retry {_provider_retry})...")
                continue

            if resp.has_tool_calls:
                stop, msg = await _handle_tool_calls(
                    resp, turn, user_id, cfg, sink, channel_id,
                    thought_history, agent_results, bubble,
                )
                if stop:
                    _final_message = msg
                    break

                turn += 1
                _provider_retry = 0
                _pending_content = None
                continue

            _no_tool_corr += 1
            logger.warning(
                "orchestration_no_tool_call",
                turn=turn, correction=_no_tool_corr,
                preview=(resp.content or "")[:80],
            )
            if _no_tool_corr > 3:
                logger.warning("orchestration_correction_limit_reached", turn=turn)
                _final_message = (resp.content or "").strip() or "Task complete."
                break
            _pending_content = (resp.content or "").strip() or _pending_content
            thought_history.append({
                "role": "user",
                "content": _CORRECTION_WARNING + (resp.content or ""),
            })
            _provider_retry = 0
            continue

        _final_message = await _finalize_orchestration(
            _final_message, resp, narrative_chunks, agent_results, cfg,
            bubble, sink, handle, channel_id, user_id, history, summary, pool,
        )
        if _final_message is not None:
            return

    except asyncio.CancelledError:
        logger.info("orchestration_cancelled", user_id=user_id)
        await bubble.stop()
        return

    except Exception as exc:
        logger.exception("orchestration_loop_failed", error=str(exc))
        await bubble.stop()
        if sink is not None:
            await sink.send_text(channel_id, f"A fatal error occurred: {_escape_html(str(exc))}")


async def _request_from_providers(
    pool, priorities, model_map, cfg, thought_history,
    openai_schemas, channel_id, sink, flush, agent_name, user_id,
):
    """Try each provider in priority order with retry logic.
    Returns (resp, last_error) — resp is None if all providers failed.
    """
    resp = None
    last_error = None
    for p_name in priorities:
        if sink is not None and sink.is_cancelled(channel_id):
            await flush(f"{agent_name} task stopped by user.")
            if sink is not None:
                sink.clear_cancel(channel_id)
            return None, last_error

        payload = {
            "model": model_map.get(p_name, cfg.default_model),
            "messages": thought_history,
        }

        _max_retries = getattr(cfg, "provider_max_retries", 2)
        _base_delay = getattr(cfg, "provider_retry_delay", 2.0)
        for _attempt in range(_max_retries + 1):
            try:
                resp = await asyncio.wait_for(
                    pool.request_with_key_structured(
                        user_id, p_name, payload, openai_schemas,
                    ),
                    timeout=60.0,
                )
                if resp.has_tool_calls or resp.content:
                    return resp, last_error
            except asyncio.TimeoutError as _exc:
                last_error = f"Provider {p_name} timed out"
                logger.warning(
                    "orchestration_provider_timeout",
                    provider=p_name, attempt=_attempt + 1, timeout=60,
                )
                if _attempt < _max_retries:
                    await asyncio.sleep(_base_delay * (2 ** _attempt))
                    continue
                break
            except Exception as _exc:
                last_error = str(_exc)
                if _is_fatal_provider_error(last_error):
                    logger.warning(
                        "orchestration_provider_failed",
                        provider=p_name, error=last_error,
                    )
                    break
                logger.warning(
                    "orchestration_provider_transient",
                    provider=p_name, attempt=_attempt + 1, error=last_error,
                )
                if _attempt < _max_retries:
                    await asyncio.sleep(_base_delay * (2 ** _attempt))
                    continue
                break
    return None, last_error


async def _handle_tool_calls(
    resp, turn, user_id, cfg, sink, channel_id,
    thought_history, agent_results, bubble,
):
    """Execute each tool call in the response.
    Returns (stop: bool, final_message: str | None).
    stop=True and final_message=message if __END_THINKING__ was encountered.
    """
    from src.agents.agent_factory import execute_tool

    for tool_call in resp.tool_calls:
        t_name = tool_call.name
        t_args = tool_call.arguments

        logger.info("tool_call_received", tool=t_name, arguments=t_args)

        bubble.update("Tool", f"running {t_name}...")
        tool_result = await execute_tool(t_name, t_args, user_id, system_prompt=cfg.system_prompt, bubble=bubble)

        logger.info("tool_result", tool=t_name, result_preview=tool_result[:100] if tool_result else None)

        if tool_result.startswith("__SEND_FILE__:"):
            parts = tool_result.split(":", 2)
            file_path = parts[1] if len(parts) > 1 else ""
            file_caption = parts[2] if len(parts) > 2 else ""
            if file_path and sink is not None:
                sent = await sink.send_file(channel_id, file_path, file_caption, workspace=cfg.workspace_path)
                tool_result = f"File sent: {file_path}" if sent else f"Failed to send file: {file_path}"

        if tool_result.startswith("__END_THINKING__:"):
            msg = tool_result[len("__END_THINKING__:"):]
            thought_history.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": tool_call.call_id,
                    "type": "function",
                    "function": {"name": t_name, "arguments": json.dumps(t_args)},
                }],
            })
            thought_history.append({
                "role": "tool",
                "content": "[end_thinking acknowledged]",
                "tool_call_id": tool_call.call_id,
            })
            logger.info("end_thinking_received", preview=msg[:80])
            return True, msg

        thought_history.append({"role": "assistant", "content": None, "tool_calls": [{"id": tool_call.call_id, "type": "function", "function": {"name": t_name, "arguments": json.dumps(t_args)}}]})
        thought_history.append({"role": "tool", "content": tool_result, "tool_call_id": tool_call.call_id})
        if f"turn_{turn}" not in agent_results:
            agent_results[f"turn_{turn}"] = []
        agent_results[f"turn_{turn}"].append({"output": tool_result, "tool_used": t_name})

    return False, None


async def _finalize_orchestration(
    final_message, resp, narrative_chunks, agent_results, cfg,
    bubble, sink, handle, channel_id, user_id, history, summary, pool,
):
    """Post-loop processing: final message assembly, dedup, sending.
    Returns the final message string, or None if turn-limit was reached.
    """
    from src.db.chat_store import add_chat_message

    if final_message is None:
        final_message = (resp.content or "").strip() if resp else ""
        if not final_message:
            final_message = "Task complete."
        logger.warning("orchestration_max_turns_no_end_thinking")

    bubble.update("Thinking", "done")
    await bubble.stop()

    final = _strip_legacy_tool_syntax(final_message)
    if final:
        narrative_chunks.append(final)

        unique = []
        for c in narrative_chunks:
            if c not in unique and len(c) > 2:
                unique.append(c)
        full_text = "\n\n".join(unique) or "Task complete."

        for marker in ("RESEARCH_FINDINGS", "SYSTEM_DATA", "TOOL_RESULT"):
            if marker in full_text:
                full_text = full_text.split(marker)[0].strip()

        findings_block = ""
        cfg_debug = getattr(cfg, "log_level", "info").lower() == "debug"
        if agent_results and cfg_debug:
            findings_block = "\n\n<b>Process log:</b>\n"
            for aid, res in agent_results.items():
                if res.get("tool_used") == "declare_step":
                    continue
                tool = res.get("tool_used", "analysis")
                preview = str(res.get("output", "done"))
                preview = (preview[:120] + "...") if len(preview) > 120 else preview
                findings_block += f"  {tool}: {_escape_html(preview)}\n"

        full_response = full_text + findings_block
        await add_chat_message(user_id, "assistant", full_text, metadata=agent_results)

        if sink is not None:
            chunks = _split_message(full_response, 4000)
            for i, chunk in enumerate(chunks):
                if i == 0:
                    try:
                        await sink.edit_message(handle, chunk)
                    except Exception:
                        await sink.edit_message(handle, chunk)
                else:
                    await sink.send_text(channel_id, chunk)

        if len(history) >= cfg.max_context_messages:
            asyncio.create_task(_trigger_summarization(user_id, history, summary, pool, cfg))
        return final_message

    await bubble.stop()
    if sink is not None:
        await sink.edit_message(
            handle,
            "The reasoning loop reached its turn limit. Please rephrase your request."
        )
    return None


def _split_message(text: str, max_len: int) -> list:
    chunks = []
    while len(text) > max_len:
        split_at = text.rfind("\n", 0, max_len)
        if split_at == -1:
            split_at = max_len
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip()
    chunks.append(text)
    return chunks


async def _trigger_summarization(user_id, history, old_summary, pool, cfg) -> None:
    """Incremental summarization: only compress the oldest messages into the rolling summary.
    Cheap and frequent is better than expensive and rare.
    """
    from src.db.chat_store import update_summary, count_messages_since
    logger.info("triggering_summarization", user_id=user_id)

    # Only summarize the oldest messages (not the whole history)
    interval = getattr(cfg, "summarization_interval", 8)
    to_compress = history[:max(1, len(history) - 4)]  # keep last 4 raw, compress the rest
    if not to_compress:
        return

    history_text = ""
    for m in to_compress:
        role = m["role"]
        content = m["content"][:400]  # cap individual message length
        history_text += f"{role}: {content}\n"

    prompt = (
        "Update the knowledge state with new conversation turns. "
        "Be dense and factual. Max 150 words. "
        "Remove outdated info if superseded.\n\n"
        f"Current state: {old_summary or 'None'}\n\n"
        f"New turns to integrate:\n{history_text}"
    )
    payload = {
        "model": cfg.default_model,
        "messages": [
            {"role": "system", "content": "You are a high-fidelity knowledge compressor."},
            {"role": "user", "content": prompt},
        ],
    }
    priorities = cfg.default_provider_priority or ["gemini", "groq", "openrouter"]
    for p in priorities:
        try:
            resp = await pool.request_with_key(user_id, p, payload)
            new_summary = resp.get("output")
            if new_summary:
                await update_summary(user_id, new_summary, history[-1]["id"])
                return
        except Exception as exc:
            logger.warning("summarization_provider_failed", provider=p, error=str(exc))
            continue



async def providers_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/providers — show all configured providers and test connectivity."""
    cfg = Config.get()
    from src.db.key_store import upsert_user, list_api_keys
    tg_user = update.effective_user
    user_id = await upsert_user("telegram", str(tg_user.id), tg_user.username)
    pool = get_pool()

    lines = ["<b>Provider status</b>\n"]

    # Keyed providers
    db_keys = await list_api_keys(user_id)
    for provider in ["gemini", "groq", "openrouter"]:
        env_key = bool(os.environ.get(f"{provider.upper()}_API_KEY"))
        db_count = sum(1 for k in db_keys if k.get("provider", "").lower() == provider and not k["is_blacklisted"])
        blacklisted = sum(1 for k in db_keys if k.get("provider", "").lower() == provider and k["is_blacklisted"])
        status = f"{db_count} key(s) active"
        if env_key: status += " + env key"
        if blacklisted: status += f", {blacklisted} blacklisted"
        if db_count == 0 and not env_key: status = "no keys"
        lines.append(f"  {provider}: {status}")

    # Ollama
    if cfg.ollama_enabled:
        try:
            from src.providers.ollama_provider import OllamaProvider
            op = OllamaProvider()
            models = await op.list_models()
            lines.append(f"  ollama: {len(models)} model(s) — {', '.join(models[:3])}" + ("..." if len(models) > 3 else ""))
        except Exception as exc:
            lines.append(f"  ollama: unreachable ({exc})")
    else:
        lines.append("  ollama: disabled (set ollama_enabled: true in config.json)")

    # G4F
    if cfg.g4f_enabled:
        try:
            import g4f  # noqa
            lines.append("  g4f: installed and enabled")
        except ImportError:
            lines.append("  g4f: enabled in config but not installed (pip install g4f)")
    else:
        lines.append("  g4f: disabled (set g4f_enabled: true in config.json)")

    lines.append(f"\n<b>Priority order:</b> {' → '.join(cfg.default_provider_priority)}")
    lines.append(f"<b>Default model:</b> <code>{cfg.default_model}</code>")
    await update.message.reply_html("\n".join(lines))


async def reload_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/reload — reload config and tool registry from disk."""
    owner = os.environ.get("OWNER_USER_ID")
    if owner and str(update.effective_user.id) != str(owner):
        await update.message.reply_text("Only the owner can reload config.")
        return
    from src.config import Config
    from src.tools.registry import invalidate_registry
    Config.reload()
    invalidate_registry()
    cfg = Config.get()
    await update.message.reply_text(
        f"Config reloaded.\nModel: {cfg.default_model}\n"
        f"Security: {cfg.command_security_level}\n"
        f"Priority: {' → '.join(cfg.default_provider_priority)}"
    )



# ---------------------------------------------------------------------------
# Photo / vision handler
# ---------------------------------------------------------------------------

async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle photo messages — encodes and feeds to vision-capable providers."""
    import base64, io
    from src.db.key_store import upsert_user, list_api_keys
    cfg = Config.get()
    tg_user = update.effective_user
    user_id = await upsert_user("telegram", str(tg_user.id), tg_user.username)
    keys_list = await list_api_keys(user_id)
    env_keys = [p for p in ["GEMINI", "GROQ", "OPENROUTER"] if os.environ.get(f"{p}_API_KEY")]
    if not keys_list and not env_keys:
        await update.message.reply_text("No API keys stored. Add one with /addkey first.")
        return

    # Download the highest-resolution photo
    photo = update.message.photo[-1]
    caption = update.message.caption or "Describe and analyze this image in detail."
    sent = await update.message.reply_text(f"{_agent_name(cfg)} is analyzing the image...")

    try:
        photo_file = await photo.get_file()
        buf = io.BytesIO()
        await photo_file.download_to_memory(buf)
        b64 = base64.b64encode(buf.getvalue()).decode()
        mime = "image/jpeg"

        pool = get_pool()
        payload = {
            "model": cfg.default_model,
            "messages": [
                {"role": "system", "content": cfg.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                        {"type": "text", "text": caption},
                    ],
                },
            ],
        }

        # Try vision-capable providers first (Gemini > OpenRouter)
        reply = None
        for provider in (["gemini", "openrouter"] + [p for p in cfg.default_provider_priority
                          if p not in ("gemini", "openrouter")]):
            try:
                resp = await pool.request_with_key(user_id, provider, payload)
                reply = resp.get("output", "").strip()
                if reply: break
            except Exception as exc:
                logger.debug("vision_provider_failed", provider=provider, error=str(exc))
                continue

        if not reply:
            reply = "Could not analyze the image — no vision-capable provider responded."

        from src.db.chat_store import add_chat_message
        await add_chat_message(user_id, "user", f"[Image] {caption}")
        await add_chat_message(user_id, "assistant", reply)

        try:
            await context.bot.edit_message_text(chat_id=sent.chat_id,
                                                message_id=sent.message_id,
                                                text=reply, parse_mode="HTML")
        except Exception:
            await context.bot.edit_message_text(chat_id=sent.chat_id,
                                                message_id=sent.message_id, text=reply)
    except Exception as exc:
        logger.exception("photo_handler_failed", error=str(exc))
        await context.bot.edit_message_text(chat_id=sent.chat_id, message_id=sent.message_id,
                                            text=f"Failed to process image: {_escape_html(str(exc))}")


# ---------------------------------------------------------------------------
# File sending helper (called by orchestration when agent uses send_file tool)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# /autowatch — natural language watcher creation
# ---------------------------------------------------------------------------

async def autowatch_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/autowatch <goal> — AI decides what to monitor and sets it up."""
    from src.agents.background.manager import BackgroundAgentManager
    from src.db.key_store import upsert_user
    cfg = Config.get()

    goal = " ".join(context.args).strip() if context.args else ""
    if not goal:
        await update.message.reply_html(
            "<b>/autowatch</b> — let the AI decide what to monitor\n\n"
            "<b>Examples:</b>\n"
            "/autowatch be a guardian to this server\n"
            "/autowatch monitor my nginx and postgres\n"
            "/autowatch alert me if disk gets above 80% and report weekly usage\n"
            "/autowatch watch for OOM killer in dmesg every 5 minutes\n\n"
            "The AI will decide which watcher types to create, write any needed "
            "scripts, and start them automatically."
        )
        return

    tg_user = update.effective_user
    channel_id = f"tg:{update.effective_chat.id}"
    user_id = await upsert_user("telegram", str(tg_user.id), tg_user.username)
    manager = BackgroundAgentManager.get()

    if manager is None:
        await update.message.reply_text("Background manager not running.")
        return

    sent = await update.message.reply_text(
        f"{_agent_name(cfg)} is deciding what to monitor..."
    )

    try:
        created = await manager.create_from_natural_language(user_id, channel_id, goal)
        if not created:
            await context.bot.edit_message_text(
                chat_id=sent.chat_id, message_id=sent.message_id,
                text="Could not create any watchers. Try rephrasing or use /watch manually."
            )
            return

        lines = [f"Started {len(created)} background agent(s):\n"]
        for c in created:
            lines.append(
                f"<code>{c.id}</code> [{c.watcher_type}] — {_escape_html(c.name)}\n"
                f"  {_escape_html(c.description[:80])} (every {c.interval_seconds}s)"
            )
        lines.append(f"\nUse /watchers to see all · /stopwatch &lt;id&gt; to stop one")

        await context.bot.edit_message_text(
            chat_id=sent.chat_id, message_id=sent.message_id,
            text="\n".join(lines), parse_mode="HTML"
        )
    except Exception as exc:
        logger.exception("autowatch_failed", error=str(exc))
        await context.bot.edit_message_text(
            chat_id=sent.chat_id, message_id=sent.message_id,
            text=f"Failed: {_escape_html(str(exc))}"
        )


async def pinmemory_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/pinmemory &lt;key&gt; — pin a memory so it's always injected into context."""
    from src.db.chat_store import pin_memory
    from src.db.key_store import upsert_user
    key = " ".join(context.args).strip() if context.args else ""
    if not key:
        await update.message.reply_text("Usage: /pinmemory &lt;key&gt;")
        return
    user_id = await upsert_user("telegram", str(update.effective_user.id), update.effective_user.username)
    ok = await pin_memory(user_id, key)
    if ok:
        await update.message.reply_text(f"Pinned: {key}\nThis memory will always be included in context.")
    else:
        await update.message.reply_text(f"Memory '{key}' not found. Check /memory for available keys.")


async def unpinmemory_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/unpinmemory &lt;key&gt; — unpin a memory (retrieved semantically instead)."""
    from src.db.chat_store import unpin_memory
    from src.db.key_store import upsert_user
    key = " ".join(context.args).strip() if context.args else ""
    if not key:
        await update.message.reply_text("Usage: /unpinmemory &lt;key&gt;")
        return
    user_id = await upsert_user("telegram", str(update.effective_user.id), update.effective_user.username)
    ok = await unpin_memory(user_id, key)
    if ok:
        await update.message.reply_text(f"Unpinned: {key}")
    else:
        await update.message.reply_text(f"Memory '{key}' not found.")

# ---------------------------------------------------------------------------
# Workspace commands
# ---------------------------------------------------------------------------

async def files_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/files — list workspace contents."""
    cfg = Config.get()
    from src.tools.workspace import get_workspace_path, list_workspace
    ws = get_workspace_path(getattr(cfg, "workspace_path", None))
    depth = 3
    if context.args:
        try:
            depth = int(context.args[0])
        except ValueError:
            pass
    listing = list_workspace(ws, depth=depth)
    await update.message.reply_text(listing)


async def cleanworkspace_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/cleanworkspace — wipe all files in the workspace."""
    keyboard = [[
        InlineKeyboardButton("Yes, clean it", callback_data="confirm_cleanws"),
        InlineKeyboardButton("Cancel", callback_data="cancel_cleanws"),
    ]]
    await update.message.reply_text(
        "Clean the workspace? All files in ~/.rika/shared will be deleted.",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def cmdhistory_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/cmdhistory — show recent command history."""
    from src.db.key_store import upsert_user
    from src.tools.shell_tool import get_command_history
    tg_user = update.effective_user
    user_id = await upsert_user("telegram", str(tg_user.id), tg_user.username)
    limit = 15
    if context.args:
        try:
            limit = int(context.args[0])
        except ValueError:
            pass
    history = await get_command_history(user_id, limit)
    await update.message.reply_text(history)


# ---------------------------------------------------------------------------
# App factory and main
# ---------------------------------------------------------------------------

_shared_infra_started = False


async def start_shared_infra(config: Config) -> None:
    """Initialize shared infrastructure (DB + workspace only).
    Idempotent — safe to call multiple times, even from temp event loops.
    """
    global _shared_infra_started
    if _shared_infra_started:
        return
    _shared_infra_started = True
    from src.db.key_store import init_db
    await init_db()

    # Ensure workspace directory exists
    try:
        from src.tools.workspace import get_workspace_path
        ws = get_workspace_path(getattr(config, "workspace_path", None))
        logger.info("workspace_ready", path=str(ws))
    except Exception as exc:
        logger.warning("workspace_init_failed", error=str(exc))


async def start_background_tasks(config: Config) -> None:
    """Start scheduler + unblacklist loop.
    Must be called from within a persistent event loop (not asyncio.run()).
    Safe to call multiple times — scheduler and create_task are idempotent.
    """
    try:
        from src.providers.unblacklist_scheduler import unblacklist_loop
        from src.scheduler import start_scheduler
        try:
            start_scheduler(config)
        except Exception as exc:
            logger.warning("scheduler_start_failed", error=str(exc))
        _unblacklist_task = asyncio.create_task(
            unblacklist_loop(), name="unblacklist_loop"
        )
    except Exception as exc:
        logger.warning("background_tasks_failed", error=str(exc))


def build_application(config: Config):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN not set in environment")

    async def _post_init(application) -> None:
        """Runs inside PTB's event loop after the bot is ready."""
        global _adapter, _sink, _router

        await start_shared_infra(config)
        await start_background_tasks(config)

        # Create adapter and sink for Telegram
        _adapter = TelegramAdapter(
            application.bot,
            owner_user_id=os.environ.get("OWNER_USER_ID", "").strip() or None,
        )
        _router = PlatformRouter()
        _router.register("tg:", _adapter)
        _sink = EventSink(_router)

        # Background agent manager — must start inside the running event loop
        from src.agents.background.manager import BackgroundAgentManager
        manager = BackgroundAgentManager.initialize(_sink)
        await manager.start()

    async def _post_shutdown(application) -> None:
        """Cancel background tasks on clean shutdown."""
        from src.agents.background.manager import BackgroundAgentManager
        manager = BackgroundAgentManager.get()
        if manager is not None:
            await manager.stop()
        # Cancel the unblacklist loop if still running
        for task in asyncio.all_tasks():
            if task.get_name() in ("unblacklist_loop", "wake_processor"):
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        logger.info("shutdown_cleanup_complete")

    app = (
        ApplicationBuilder()
        .token(token)
        .post_init(_post_init)
        .post_shutdown(_post_shutdown)
        .build()
    )
    # Auth gate — must be first (group=-1 runs before all other groups)
    from telegram.ext import TypeHandler
    app.add_handler(TypeHandler(Update, _auth_gate), group=-1)

    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("help", help_handler))
    app.add_handler(CommandHandler("addkey", addkey_handler))
    app.add_handler(CommandHandler("status", status_handler))
    app.add_handler(CommandHandler("memory", memory_handler))
    app.add_handler(CommandHandler("pinmemory", pinmemory_handler))
    app.add_handler(CommandHandler("unpinmemory", unpinmemory_handler))
    app.add_handler(CommandHandler("deletememory", deletememory_handler))
    app.add_handler(CommandHandler("autowatch", autowatch_handler))
    app.add_handler(CommandHandler("watch", watch_handler))
    app.add_handler(CommandHandler("watchers", watchers_handler))
    app.add_handler(CommandHandler("stopwatch", stopwatch_handler))
    app.add_handler(CommandHandler("stop", stop_command_handler))
    app.add_handler(CommandHandler("wakelog", wakelog_handler))
    app.add_handler(CommandHandler("providers", providers_handler))
    app.add_handler(CommandHandler("reload", reload_handler))
    app.add_handler(CommandHandler("files", files_handler))
    app.add_handler(CommandHandler("cleanworkspace", cleanworkspace_handler))
    app.add_handler(CommandHandler("cmdhistory", cmdhistory_handler))
    app.add_handler(CommandHandler("broadcast", broadcast_handler))
    app.add_handler(CommandHandler("delete_me", delete_me_handler))
    app.add_handler(CallbackQueryHandler(callback_query_handler, pattern="^(confirm_delete|confirm_cleanws)$"))
    app.add_handler(CallbackQueryHandler(stop_task_handler, pattern="^stop_task:"))
    app.add_handler(CallbackQueryHandler(retry_now_handler, pattern="^retry_now:"))
    app.add_handler(MessageHandler(filters.PHOTO, photo_handler))
    app.add_handler(MessageHandler(filters.Document.ALL, document_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, key_submission_handler))

    # Error handler — log all unhandled exceptions
    async def error_handler(update: Optional[Update], context: ContextTypes.DEFAULT_TYPE) -> None:
        logger.exception("unhandled_error", error=context.error, update=update)
    app.add_error_handler(error_handler)

    return app


# ---------------------------------------------------------------------------
# Graceful shutdown — all modes
# ---------------------------------------------------------------------------

import atexit
import signal as _signal

_shutting_down = False


def _cleanup_resources() -> None:
    """Clean up shared resources on exit (atexit handler)."""
    global _shutting_down
    if _shutting_down:
        return
    _shutting_down = True
    logger.info("shutdown_cleanup_started")

    # Cancel background tasks
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            for task in asyncio.all_tasks(loop):
                if task.get_name() in ("unblacklist_loop", "wake_processor"):
                    task.cancel()
    except RuntimeError:
        pass

    # Stop background agent manager
    try:
        from src.agents.background.manager import BackgroundAgentManager
        manager = BackgroundAgentManager.get()
        if manager is not None:
            import asyncio as _a
            try:
                _a.run(manager.stop())
            except RuntimeError:
                pass
    except Exception:
        pass

    # Close DB pool
    try:
        from src.db.connection import close_pg_pool
        close_pg_pool()
    except Exception:
        pass

    logger.info("shutdown_cleanup_complete")


atexit.register(_cleanup_resources)

# Register signal handlers for non-PTB modes (PTB has its own signal handling)
import signal as _sig

def _signal_handler(signum, frame):
    logger.info("shutdown_signal_received", signal=signum)
    _cleanup_resources()
    raise SystemExit(0)

_sig.signal(_sig.SIGTERM, _signal_handler)
# SIGINT is handled by PTB in Telegram mode, but we add it for API-only/headless
_sig.signal(_sig.SIGINT, _signal_handler)


import threading


def main() -> None:
    load_dotenv()
    config = Config.get()
    platforms = Config.detect_platforms()
    Config.log_platform_status(platforms)

    telegram_enabled = platforms.get("telegram", False)
    discord_enabled = platforms.get("discord", False)
    api_enabled = platforms.get("api", False)

    enabled_count = sum([telegram_enabled, discord_enabled, api_enabled])

    # Shared infra + background tasks
    asyncio.run(start_shared_infra(config))

    if enabled_count == 0:
        _start_headless(config)
        return

    if api_enabled:
        _start_api_background(config)

    if telegram_enabled and discord_enabled:
        _run_telegram_discord(config)
    elif telegram_enabled and api_enabled and not discord_enabled:
        _run_multi_platform(config)  # Telegram + API (legacy)
    elif telegram_enabled:
        _run_telegram(config)
    elif discord_enabled:
        _run_discord(config)
    elif api_enabled:
        _run_api(config)


def _run_telegram(config: Config) -> None:
    """Start Telegram interface only (backward-compat default)."""
    print(f"  → Starting {config.bot_name} Telegram interface (polling)")
    app = build_application(config)
    app.run_polling()


def _run_api(config: Config) -> None:
    """Start API interface only — no Telegram."""
    asyncio.run(start_shared_infra(config))

    _start_api_background(config)

    from src.interfaces.api.server import run_api_server
    run_api_server(config)


def _start_api_background(config: Config) -> None:
    """Start background event loop for API (or Discord+API)."""
    def _bg_loop():
        asyncio.run(_run_bg(config))
    threading.Thread(target=_bg_loop, daemon=True).start()


def _run_multi_platform(config: Config) -> None:
    """Run Telegram + API concurrently."""
    asyncio.run(start_shared_infra(config))

    # Start API in background thread
    from src.interfaces.api.server import run_api_server
    api_thread = threading.Thread(
        target=run_api_server,
        args=(config,),
        daemon=True,
    )
    api_thread.start()
    print("  → API server started in background thread")

    # Start Telegram in main thread — _post_init calls start_background_tasks
    # inside PTB's persistent event loop
    print(f"  → Starting {config.bot_name} Telegram interface (polling)")
    app = build_application(config)
    app.run_polling()


async def _run_bg(config: Config) -> None:
    """Persistent async loop for background tasks (API-only / headless mode)."""
    await start_background_tasks(config)
    await asyncio.Event().wait()


def _run_discord(config: Config) -> None:
    """Start Discord interface only — standalone."""
    print(f"  → Starting {config.bot_name} Discord interface")
    from src.interfaces.discord.bot import DiscordBot
    token = os.environ.get("DISCORD_BOT_TOKEN", "").strip()
    owner_id = os.environ.get("OWNER_USER_ID", "").strip()
    bot = DiscordBot(token, owner_user_id=owner_id)
    bot.run()


def _run_telegram_discord(config: Config) -> None:
    """Run Telegram + Discord concurrently."""
    print(f"  → Starting {config.bot_name} Telegram + Discord interfaces")

    # Start Discord in a background thread
    import threading
    token = os.environ.get("DISCORD_BOT_TOKEN", "").strip()
    owner_id = os.environ.get("OWNER_USER_ID", "").strip()
    discord_thread = threading.Thread(
        target=_run_discord_isolated,
        args=(token, owner_id),
        daemon=True,
    )
    discord_thread.start()
    print("  → Discord bot started in background thread")
    print("    Make sure DISCORD_BOT_INTENTS (Message Content) is ON in Discord Developer Portal")

    # Start Telegram in main thread
    print(f"  → Starting {config.bot_name} Telegram interface (polling)")
    app = build_application(config)
    app.run_polling()


def _run_discord_isolated(token: str, owner_id: str) -> None:
    """Run Discord bot in its own thread (no shared event loop issues)."""
    from src.interfaces.discord.bot import DiscordBot
    bot = DiscordBot(token, owner_user_id=owner_id)
    bot.run()


def _start_headless(config: Config) -> None:
    """Start with no interactive interface — shared infra + background tasks."""
    async def _run():
        await start_shared_infra(config)
        await start_background_tasks(config)
        await asyncio.Event().wait()
    asyncio.run(_run())


if __name__ == "__main__":
    main()
