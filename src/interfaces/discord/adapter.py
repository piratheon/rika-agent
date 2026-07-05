"""Discord adapter — InterfaceAdapter for discord.py."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import discord

from src.core.interface_adapter import InterfaceAdapter
from src.utils.logger import logger


def _parse_handle(handle: str) -> Tuple[int, int]:
    """Parse a handle string 'discord:{channel_id}:{message_id}' back to ints."""
    parts = handle.split(":")
    return int(parts[1]), int(parts[2])


class DiscordAdapter(InterfaceAdapter):
    """InterfaceAdapter implementation for Discord.

    Requires: discord.py (pip install discord.py)
    Enable: set DISCORD_BOT_TOKEN in .env
    """

    def __init__(self, client: discord.Client, owner_user_id: Optional[str] = None) -> None:
        self._client = client
        self._owner_user_id = owner_user_id or os.environ.get("OWNER_USER_ID", "").strip()

        # Per-channel state
        self._cancel_flags: Dict[str, bool] = {}
        self._last_edit_ms: Dict[str, float] = {}
        self._active_tasks: Dict[str, asyncio.Task] = {}
        self._retry_events: Dict[str, asyncio.Event] = {}
        self._stop_events: Dict[str, asyncio.Event] = {}

    # ------------------------------------------------------------------
    # InterfaceAdapter implementation
    # ------------------------------------------------------------------

    async def send_text(self, channel_id: str, text: str, **kwargs) -> str:
        channel = await self._get_channel(channel_id)
        if channel is None:
            return ""
        sent = await channel.send(text, **kwargs)
        handle = f"discord:{channel.id}:{sent.id}"
        return handle

    async def edit_message(self, handle: str, text: str, **kwargs) -> None:
        import time
        from src.config import Config
        throttle_ms = getattr(Config.get(), "discord_edit_throttle_ms", 800)
        now_ms = time.monotonic() * 1000
        if now_ms - self._last_edit_ms.get(handle, 0.0) < throttle_ms:
            return
        channel_id, message_id = _parse_handle(handle)
        channel = self._client.get_channel(channel_id)
        if channel is None:
            return
        try:
            msg = await channel.fetch_message(message_id)
            await msg.edit(content=text, **kwargs)
            self._last_edit_ms[handle] = time.monotonic() * 1000
        except (discord.NotFound, discord.Forbidden, discord.HTTPException):
            pass

    async def send_file(
        self, channel_id: str, file_path: str, caption: str = "", **kwargs
    ) -> bool:
        workspace = kwargs.get("workspace", "~/.rika/shared")
        channel = await self._get_channel(channel_id)
        if channel is None:
            return False

        ws = Path(workspace).expanduser().resolve()
        safe_rel = file_path.lstrip("/")
        full_path = (ws / safe_rel).resolve()

        if not str(full_path).startswith(str(ws)):
            logger.warning("send_file_path_traversal_blocked", path=file_path)
            return False

        if not full_path.exists():
            await channel.send(f"Agent tried to send {file_path!r} but the file does not exist.")
            return False

        if full_path.stat().st_size > 50 * 1024 * 1024:
            await channel.send(f"File {file_path!r} is too large to send (>50 MB).")
            return False

        try:
            discord_file = discord.File(full_path, filename=full_path.name)
            await channel.send(file=discord_file, content=caption or None)
            return True
        except Exception as exc:
            logger.error("send_file_failed", path=str(full_path), error=str(exc))
            await channel.send(f"Failed to send file: {exc}")
            return False

    async def show_countdown(
        self,
        handle: str,
        wait_seconds: int = 30,
        attempt: int = 1,
        on_stop: Optional[asyncio.Event] = None,
        on_retry: Optional[asyncio.Event] = None,
    ) -> str:
        channel_id_num, message_id = _parse_handle(handle)
        channel = await self._get_channel(f"discord:{channel_id_num}")
        if channel is None:
            await asyncio.sleep(wait_seconds)
            return "done"

        # Reconstruct the full channel_id (prefixed) for state dict keys
        channel_key = f"discord:{channel_id_num}"

        if on_stop is None:
            on_stop = asyncio.Event()
        if on_retry is None:
            on_retry = asyncio.Event()
        self._stop_events[channel_key] = on_stop
        self._retry_events[channel_key] = on_retry

        view = discord.ui.View(timeout=wait_seconds)
        stop_button = discord.ui.Button(label="⏹ Stop", style=discord.ButtonStyle.danger)
        retry_button = discord.ui.Button(label="↩ Retry now", style=discord.ButtonStyle.primary)

        async def stop_cb(interaction: discord.Interaction):
            await interaction.response.defer()
            on_stop.set()

        async def retry_cb(interaction: discord.Interaction):
            await interaction.response.defer()
            on_retry.set()

        stop_button.callback = stop_cb
        retry_button.callback = retry_cb
        view.add_item(stop_button)
        view.add_item(retry_button)

        note = ""
        if attempt > 1:
            note = (
                "\n\n*All providers seem to be over quota. "
                "Consider adding fresh keys.*"
            )

        try:
            msg = await channel.fetch_message(message_id)
            await msg.edit(
                content=f"⏳ **Rate limit reached** — retrying in **{wait_seconds}s**{note}",
                view=view,
            )
        except (discord.NotFound, discord.Forbidden, discord.HTTPException):
            msg = await channel.send(
                content=f"⏳ **Rate limit reached** — retrying in **{wait_seconds}s**{note}",
                view=view,
            )
            handle = f"discord:{channel.id}:{msg.id}"

        for remaining in range(wait_seconds - 1, -1, -1):
            retry_task = asyncio.create_task(on_retry.wait())
            stop_task = asyncio.create_task(on_stop.wait())
            try:
                await asyncio.wait(
                    {retry_task, stop_task},
                    timeout=1.0,
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                for t_ in (retry_task, stop_task):
                    if not t_.done():
                        t_.cancel()
            if on_stop.is_set():
                self._cleanup_countdown(channel_key)
                view.stop()
                return "stop"
            if on_retry.is_set():
                self._cleanup_countdown(channel_key)
                view.stop()
                return "retry"
            try:
                msg = await channel.fetch_message(int(handle.split(":")[2]))
                bar_done = wait_seconds - remaining
                filled = "█" * bar_done + "░" * remaining
                await msg.edit(
                    content=f"⏳ **Rate limit reached** — retrying in **{remaining}s**\n`{filled}`{note}",
                    view=view,
                )
            except (discord.NotFound, discord.Forbidden, discord.HTTPException):
                pass

        self._cleanup_countdown(channel_key)
        view.stop()
        return "done"

    def supports_interactive(self) -> bool:
        return True

    def format_text(self, text: str, mode: str = "HTML") -> str:  # patch_friends_fixes: format_text_fixed
        if mode == "HTML":
            import re
            # Step 1 — convert known structural tags to Markdown while
            # &lt;/&gt; are still escaped (protects text content from the
            # tag-stripping regex that follows).
            text = text.replace("<br>", "\n").replace("<br/>", "\n")
            text = text.replace("<b>", "**").replace("</b>", "**")
            text = text.replace("<i>", "_").replace("</i>", "_")
            text = text.replace("<u>", "__").replace("</u>", "__")
            text = text.replace("<s>", "~~").replace("</s>", "~~")
            text = text.replace("<code>", "`").replace("</code>", "`")
            text = text.replace("<pre>", "```").replace("</pre>", "```")
            # Step 2 — strip any remaining unknown HTML tags.
            # Safe: &lt; and &gt; in text content are still escaped entities,
            # not bare < >, so the regex cannot eat them.
            text = re.sub(r"<[^>]+>", "", text)
            # Step 3 — unescape HTML entities now that no tags remain.
            # &lt;/&gt; in text content (e.g. numeric comparisons, C++ templates,
            # escaped LLM output) are restored to < > for the Discord user.
            text = text.replace("&lt;", "<").replace("&gt;", ">")
            text = text.replace("&amp;", "&").replace("&quot;", '"'
                               ).replace("&apos;", "'")
        return text

    async def send_chunked(
        self, channel_id: str, text: str, max_len: int = 2000
    ) -> list[str]:
        channel = await self._get_channel(channel_id)
        handles: list[str] = []
        if channel is None:
            return handles
        chunks = self._split_message(text, max_len)
        for chunk in chunks:
            sent = await channel.send(chunk)
            handles.append(f"discord:{channel.id}:{sent.id}")
        return handles

    async def send_typing(self, channel_id: str) -> None:
        channel = await self._get_channel(channel_id)
        if channel is None:
            return
        async with channel.typing():
            await asyncio.sleep(0.5)

    def get_channel_id(self, source: str, platform_id: str) -> str:
        return f"discord:{platform_id}"

    # ------------------------------------------------------------------
    # Task lifecycle (called by EventSink)
    # ------------------------------------------------------------------

    def is_cancelled(self, channel_id: str) -> bool:
        return self._cancel_flags.get(channel_id, False)

    def clear_cancel(self, channel_id: str) -> None:
        self._cancel_flags[channel_id] = False

    def track_task(self, channel_id: str, task: asyncio.Task) -> None:
        self._active_tasks[channel_id] = task

    def untrack_task(self, channel_id: str) -> None:
        self._active_tasks.pop(channel_id, None)

    def init_countdown_events(self, channel_id: str) -> None:
        self._retry_events[channel_id] = asyncio.Event()
        self._stop_events[channel_id] = asyncio.Event()

    # ------------------------------------------------------------------
    # Discord-specific event handlers — called by bot.py
    # ------------------------------------------------------------------

    def handle_stop(self, channel_id: str) -> None:
        self._cancel_flags[channel_id] = True
        event = self._stop_events.get(channel_id)
        if event is not None:
            event.set()
        task = self._active_tasks.get(channel_id)
        if task is not None and not task.done():
            task.cancel()

    def handle_retry(self, channel_id: str) -> None:
        event = self._retry_events.get(channel_id)
        if event is not None:
            event.set()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get_channel(self, channel_id: str) -> Optional[discord.abc.Messageable]:
        cid = int(channel_id.split(":", 1)[1])
        channel = self._client.get_channel(cid)
        if channel is None:
            try:
                channel = await self._client.fetch_channel(cid)
            except Exception:
                pass
        return channel

    def _cleanup_countdown(self, channel_key: str) -> None:
        self._retry_events.pop(channel_key, None)
        self._stop_events.pop(channel_key, None)

    @staticmethod
    def _split_message(text: str, max_len: int) -> list[str]:
        chunks = []
        while len(text) > max_len:
            split_at = text.rfind("\n", 0, max_len)
            if split_at == -1:
                split_at = max_len
            chunks.append(text[:split_at])
            text = text[split_at:].lstrip()
        chunks.append(text)
        return chunks
