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
        # patch_discord_intents_error: privileged_intents_handler
        try:
            self._client.run(self._token, reconnect=True)
        except discord.errors.PrivilegedIntentsRequired:
            print(
                "\n  [Discord] Bot requires the Message Content privileged intent.\n"
                "  Enable it at:\n"
                "    https://discord.com/developers/applications/\n"
                "  Steps:\n"
                "    1. Select your application\n"
                "    2. Bot → Privileged Gateway Intents\n"
                "    3. Enable \"Message Content Intent\"\n"
                "    4. Save Changes and restart the bot\n",
                flush=True,
            )

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
        # patch_deep_scan: bug2_model_router_replaced
        from src.db.chat_store import add_chat_message

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

        last_error: Exception | None = None
        pool = _pool()
        all_priorities = cfg.default_provider_priority or ["gemini", "groq", "openrouter"]
        available_keys = await _get_keys(user_id)
        available = {k["provider"] for k in available_keys}
        priority = [p for p in all_priorities if p in available]

        for provider_name in priority:
            try:
                resp = await pool.request_with_key(user_id, provider_name, payload)
                reply = (
                    resp.get("content") or
                    resp.get("text") or
                    (resp.get("choices") or [{}])[0].get("message", {}).get("content", "")
                ).strip()
                if reply:
                    await add_chat_message(user_id, "assistant", reply)
                    await channel.send(reply)
                    return
            except Exception as exc:
                last_error = exc
                logger.debug("direct_reply_provider_failed",
                             provider=provider_name, error=str(exc))

        err = str(last_error or "").lower()
        if "429" in err or "quota" in err or "rate limit" in err:
            msg = "All providers are rate-limited. Wait a few minutes and try again."
        elif "401" in err or "unauthorized" in err or "no api key" in err:
            msg = "No working API key found. Add one with /addkey."
        elif "timeout" in err or "timed out" in err:
            msg = "The AI provider timed out. Please try again."
        else:
            msg = "All AI providers failed to respond. Try again in a moment."
        await channel.send(msg)

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
            # patch_deep_scan: bug3_orchestrator_direct
            from src.core.orchestrator import Orchestrator
            _cfg_o = _cfg()

            async def _on_msg(text: str, _results: dict) -> None:
                if self._sink:
                    await self._sink.send_text(channel_id, text)

            orch = Orchestrator(
                on_message=_on_msg,
                is_cancelled=lambda cid: (
                    self._sink.is_cancelled(cid) if self._sink else False
                ),
            )
            await orch.run(
                user_id=user_id,
                channel_id=channel_id,
                message=original_text,
                context_str=context_str,
                system_prompt=_cfg_o.system_prompt,
                history=history,
                summary=summary,
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
