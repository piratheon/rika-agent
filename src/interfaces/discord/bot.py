"""Discord bot client — receives messages, dispatches to orchestration.

Connects to the Discord Gateway, receives messages/button clicks, and
routes them through the same orchestration pipeline as Telegram.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Dict, Optional

import discord
from discord import Intents

from src.core.event_sink import EventSink
from src.core.router import PlatformRouter
from src.interfaces.discord.adapter import DiscordAdapter
from src.utils.logger import logger


def _cfg():
    from src.config import Config
    return Config.get()


def _pool():
    from src.providers.provider_pool import get_pool
    return get_pool()


class DiscordBot:
    """Wraps discord.Client with orchestration dispatch.

    Args:
        token: Discord bot token from DISCORD_BOT_TOKEN.
        owner_user_id: Discord user ID allowed to use the bot
                       (from OWNER_USER_ID).  Empty = open mode.
    """

    def __init__(self, token: str, owner_user_id: str = "") -> None:
        self._token = token
        self._owner_user_id = owner_user_id
        self._adapter: Optional[DiscordAdapter] = None
        self._sink: Optional[EventSink] = None
        self._semaphores: Dict[str, asyncio.Semaphore] = {}
        self._initialised = False
        self._client = self._build_client()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        await self._client.start(self._token, reconnect=True)

    def run(self) -> None:
        self._client.run(self._token, reconnect=True)

    async def close(self) -> None:
        await self._client.close()

    # ------------------------------------------------------------------
    # Client setup
    # ------------------------------------------------------------------

    def _build_client(self) -> discord.Client:
        intents = Intents.default()
        intents.message_content = True

        client = discord.Client(intents=intents)

        @client.event
        async def on_ready():
            await self._on_ready()

        @client.event
        async def on_message(message: discord.Message):
            await self._on_message(message)

        @client.event
        async def on_interaction(interaction: discord.Interaction):
            await self._on_interaction(interaction)

        return client

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    async def _on_ready(self) -> None:
        logger.info("discord_bot_ready", user=str(self._client.user))
        print(f"  → Logged in as {self._client.user}")

        if self._initialised:
            return
        self._initialised = True

        cfg = _cfg()
        from src.bot.app import start_shared_infra, start_background_tasks
        await start_shared_infra(cfg)
        await start_background_tasks(cfg)

        # Own adapter + sink (standalone)
        self._adapter = DiscordAdapter(self._client, owner_user_id=self._owner_user_id)
        _router = PlatformRouter()
        _router.register("discord:", self._adapter)
        self._sink = EventSink(_router)

        # Background agent manager (singleton — only initialize once)
        from src.agents.background.manager import BackgroundAgentManager
        mgr = BackgroundAgentManager.get()
        if mgr is None:
            mgr = BackgroundAgentManager.initialize(self._sink)
            await mgr.start()
        else:
            # Update existing manager's sink to include Discord
            mgr._sink = self._sink

        # Also plug into app.py globals so orchestration helpers work
        import importlib
        app_mod = importlib.import_module("src.bot.app")
        if app_mod._sink is None:
            app_mod._sink = self._sink
            app_mod._adapter = self._adapter
            app_mod._router = _router
        else:
            app_mod._router.register("discord:", self._adapter)
            app_mod._sink = EventSink(app_mod._router)
            mgr = BackgroundAgentManager.get()
            if mgr is not None:
                mgr._sink = app_mod._sink

    # ------------------------------------------------------------------
    # Message handler
    # ------------------------------------------------------------------

    async def _on_message(self, message: discord.Message) -> None:
        if message.author.bot:
            return

        # Auth gate
        if self._owner_user_id:
            if str(message.author.id) != self._owner_user_id:
                return

        channel_id = f"discord:{message.channel.id}"
        if self._adapter:
            self._adapter.clear_cancel(channel_id)

        # File attachment
        if message.attachments:
            await self._handle_file_attachment(message, channel_id)
            return

        content = message.content.strip()
        if not content:
            return

        # Built-in commands
        if content.startswith("/"):
            await self._handle_command(message, content, channel_id)
            return

        # Dispatch to orchestration
        await self._dispatch_orchestration(message, content, channel_id)

    # ------------------------------------------------------------------
    # Orchestration dispatch (mirrors Telegram's key_submission_handler)
    # ------------------------------------------------------------------

    async def _dispatch_orchestration(
        self, message: discord.Message, text: str, channel_id: str
    ) -> None:
        from src.db.chat_store import get_or_create_user, add_chat_message, get_chat_history, get_chat_summary

        user_id_int = message.author.id
        channel = message.channel

        # Get or create user
        user = await get_or_create_user(str(user_id_int), message.author.display_name or "DiscordUser")
        user_id = user["id"]

        # History & summary
        history = await get_chat_history(user_id, limit=10)
        summary = await get_chat_summary(user_id)
        cfg = _cfg()

        # Build context string (same as Telegram handler)
        from src.db.key_store import get_api_keys_by_user
        keys = await get_api_keys_by_user(user_id)
        available = [k["provider"] for k in keys] if keys else []
        if available:
            context_parts = [f"user configured providers: {', '.join(available)}"]
        else:
            context_parts = ["user has NO providers configured"]
        priority = cfg.get_active_providers(available)
        if priority:
            context_parts.append(f"active provider priority: {', '.join(priority)}")

        context_str = "\n".join(context_parts)

        # Complexity check
        from src.core.complexity import classify_complexity
        is_complex = await classify_complexity(text, cfg, _pool(), user_id)

        if not is_complex:
            await self._handle_direct_reply(
                channel, user_id, text, context_str, history, summary, cfg
            )
        else:
            # Send "thinking" message
            sent = await channel.send(f"{cfg.bot_name or 'Agent'} is thinking...")
            handle = f"discord:{channel.id}:{sent.id}"

            if self._adapter:
                self._adapter.clear_cancel(channel_id)
                self._adapter.init_countdown_events(channel_id)

            sem = self._get_semaphore(channel_id)
            task = asyncio.create_task(
                self._run_orchestration_guarded(
                    sem, channel_id, handle, user_id, context_str,
                    text, history, summary, cfg,
                )
            )
            if self._adapter:
                self._adapter.track_task(channel_id, task)

            def cleanup(_t):
                if self._adapter:
                    self._adapter.untrack_task(channel_id)
            task.add_done_callback(cleanup)

    async def _handle_direct_reply(
        self, channel, user_id: int, text: str,
        context_str: str, history: list, summary: Optional[str], cfg,
    ) -> None:
        """Simple LLM call without tools (same as Telegram's _handle_direct_reply)."""
        from src.db.chat_store import add_chat_message
        from src.providers.model_router import model_router

        await add_chat_message(user_id, "user", text)
        payload = {
            "model": cfg.default_model,
            "messages": [
                {"role": "system", "content": cfg.system_prompt},
            ],
        }
        if context_str:
            payload["messages"].append({"role": "system", "content": context_str})
        if summary:
            payload["messages"].append({"role": "system", "content": f"Session summary: {summary}"})
        for msg in history:
            payload["messages"].append({"role": msg["role"], "content": msg["content"]})
        payload["messages"].append({"role": "user", "content": text})

        last_error = None
        pool = _pool()
        available_keys = await _get_keys(user_id)
        available = [k["provider"] for k in available_keys]
        priority = cfg.get_active_providers(available)

        for provider_name in priority:
            keys_for_provider = [k for k in available_keys if k["provider"] == provider_name]
            for key in keys_for_provider:
                try:
                    reply = await model_router.route_to_provider(
                        provider_name, key["key"], payload, cfg
                    )
                    if reply:
                        await add_chat_message(user_id, "assistant", reply)
                        await channel.send(reply)
                        return
                except Exception as exc:
                    last_error = exc
                    logger.debug("direct_reply_provider_failed", provider=provider_name, error=str(exc))
                    continue

        # All providers failed
        from src.bot.app import _friendly_provider_error
        await channel.send(_friendly_provider_error(last_error))

    async def _run_orchestration_guarded(
        self, sem: asyncio.Semaphore, channel_id: str, handle: str,
        user_id: int, context_str: str, original_text: str,
        history: list, summary: Optional[str], cfg,
    ) -> None:
        if sem.locked() and sem._value == 0:
            if self._sink:
                try:
                    await self._sink.edit_message(handle, "Another task is already running. Please wait.")
                except Exception:
                    pass
            return
        async with sem:
            from src.bot.app import run_orchestration_background
            await run_orchestration_background(
                channel_id, handle, user_id, context_str,
                original_text, history, summary, None,
                sink=self._sink,
            )

    # ------------------------------------------------------------------
    # Buttons / interactions
    # ------------------------------------------------------------------

    async def _on_interaction(self, interaction: discord.Interaction) -> None:
        if not interaction.is_component():
            return
        if not self._adapter:
            return

        channel_id = f"discord:{interaction.channel_id}"
        custom_id = interaction.data.get("custom_id", "") if interaction.data else ""

        if custom_id == "stop":
            await interaction.response.defer()
            self._adapter.handle_stop(channel_id)
        elif custom_id == "retry":
            await interaction.response.defer()
            self._adapter.handle_retry(channel_id)

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    async def _handle_command(
        self, message: discord.Message, content: str, channel_id: str
    ) -> None:
        parts = content.split()
        cmd = parts[0].lstrip("/").lower()
        args = content[len(parts[0]):].strip()

        builtin = {
            "help": self._cmd_help,
            "status": self._cmd_sink,
            "memory": self._cmd_sink,
            "providers": self._cmd_sink,
            "stop": self._cmd_stop,
            "cancel": self._cmd_stop,
        }
        handler = builtin.get(cmd)
        if handler:
            await handler(message, args, channel_id, cmd)
        else:
            # Unknown commands go to orchestration
            await self._dispatch_orchestration(message, content, channel_id)

    async def _cmd_help(
        self, msg: discord.Message, args: str, channel_id: str, cmd: str
    ) -> None:
        lines = [
            "**Available commands**",
            "",
            "`/help` — Show this message",
            "`/status` — Agent status & resource usage",
            "`/memory` — View agent memory",
            "`/providers` — List configured providers",
            "`/stop` — Cancel current task",
            "",
            "Any other message is sent to the agent for processing.",
        ]
        await msg.channel.send("\n".join(lines))

    async def _cmd_sink(
        self, msg: discord.Message, args: str, channel_id: str, cmd: str
    ) -> None:
        await self._dispatch_orchestration(msg, f"/{cmd} {args}".strip(), channel_id)

    async def _cmd_stop(
        self, msg: discord.Message, args: str, channel_id: str, cmd: str
    ) -> None:
        if self._adapter:
            self._adapter.handle_stop(channel_id)
        await msg.channel.send("⏹ Stopped.")

    # ------------------------------------------------------------------
    # File attachment handler
    # ------------------------------------------------------------------

    async def _handle_file_attachment(
        self, message: discord.Message, channel_id: str
    ) -> None:
        if not message.attachments:
            return
        attachment = message.attachments[0]
        temp_dir = Path("/tmp/rika_discord_uploads")
        temp_dir.mkdir(parents=True, exist_ok=True)
        file_path = temp_dir / attachment.filename
        await attachment.save(str(file_path))

        await self._dispatch_orchestration(
            message,
            f"attachment:{attachment.filename}",
            channel_id,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_semaphore(self, channel_id: str) -> asyncio.Semaphore:
        cfg = _cfg()
        limit = cfg.max_concurrent_orchestrations_per_user or 2
        if channel_id not in self._semaphores:
            self._semaphores[channel_id] = asyncio.Semaphore(limit)
        return self._semaphores[channel_id]


async def _get_keys(user_id: int, provider_name: Optional[str] = None) -> list:
    from src.db.key_store import get_api_keys_by_user
    keys = await get_api_keys_by_user(user_id)
    if provider_name:
        keys = [k for k in keys if k["provider"] == provider_name]
    return keys
