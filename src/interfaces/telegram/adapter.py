from __future__ import annotations

import asyncio
import html
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters
from telegram.ext._application import ApplicationHandlerStop

from src.core.interface_adapter import InterfaceAdapter
from src.utils.logger import logger


def _parse_handle(handle: str) -> Tuple[int, int]:
    """Parse a handle string 'tg:{chat_id}:{message_id}' back to ints."""
    parts = handle.split(":")
    return int(parts[1]), int(parts[2])


class TelegramAdapter(InterfaceAdapter):
    """InterfaceAdapter implementation for Telegram (PTB)."""

    def __init__(self, bot: Any, owner_user_id: Optional[str] = None) -> None:
        self._bot = bot
        self._owner_user_id = owner_user_id or os.environ.get("OWNER_USER_ID", "").strip()

        # Per-channel state
        self._cancel_flags: Dict[str, bool] = {}
        self._active_tasks: Dict[str, asyncio.Task] = {}
        self._retry_events: Dict[str, asyncio.Event] = {}
        self._stop_events: Dict[str, asyncio.Event] = {}

    # ------------------------------------------------------------------
    # InterfaceAdapter implementation
    # ------------------------------------------------------------------

    async def send_text(self, channel_id: str, text: str, **kwargs) -> str:
        parse_mode = kwargs.get("parse_mode", "HTML")
        reply_markup = kwargs.get("reply_markup")
        chat_id = self._channel_to_chat_id(channel_id)
        sent = await self._bot.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode=parse_mode,
            reply_markup=reply_markup,
        )
        handle = f"tg:{chat_id}:{sent.message_id}"
        return handle

    async def edit_message(self, handle: str, text: str, **kwargs) -> None:
        chat_id, message_id = _parse_handle(handle)
        parse_mode = kwargs.get("parse_mode", "HTML")
        reply_markup = kwargs.get("reply_markup")
        try:
            await self._bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=text,
                parse_mode=parse_mode,
                reply_markup=reply_markup,
            )
        except Exception:
            pass

    async def send_file(
        self, channel_id: str, file_path: str, caption: str = "", **kwargs
    ) -> bool:
        workspace = kwargs.get("workspace", "~/.rika/shared")
        chat_id = self._channel_to_chat_id(channel_id)
        ws = Path(workspace).expanduser().resolve()
        # Prevent absolute-path escape; resolve() + startswith() catches .. traversal
        safe_rel = file_path.lstrip("/")
        full_path = (ws / safe_rel).resolve()

        if not str(full_path).startswith(str(ws)):
            logger.warning("send_file_path_traversal_blocked", path=file_path)
            return False

        if not full_path.exists():
            await self._bot.send_message(
                chat_id=chat_id,
                text=f"Agent tried to send {file_path!r} but the file does not exist.",
            )
            return False

        if full_path.stat().st_size > 50 * 1024 * 1024:
            await self._bot.send_message(
                chat_id=chat_id,
                text=f"File {file_path!r} is too large to send (>50 MB).",
            )
            return False

        try:
            with open(full_path, "rb") as f:
                await self._bot.send_document(
                    chat_id=chat_id,
                    document=f,
                    filename=full_path.name,
                    caption=caption or f"File: {full_path.name}",
                )
            return True
        except Exception as exc:
            logger.error("send_file_failed", path=str(full_path), error=str(exc))
            await self._bot.send_message(
                chat_id=chat_id,
                text=f"Failed to send file: {exc}",
            )
            return False

    async def show_countdown(
        self,
        handle: str,
        wait_seconds: int = 30,
        attempt: int = 1,
        on_stop: Optional[asyncio.Event] = None,
        on_retry: Optional[asyncio.Event] = None,
    ) -> str:
        chat_id, message_id = _parse_handle(handle)
        channel_id = self._chat_id_to_channel(chat_id)

        if on_stop is None:
            on_stop = asyncio.Event()
        if on_retry is None:
            on_retry = asyncio.Event()
        self._stop_events[channel_id] = on_stop
        self._retry_events[channel_id] = on_retry

        kb = InlineKeyboardMarkup([[
            InlineKeyboardButton("⏹ Stop", callback_data=f"stop_task:{chat_id}"),
            InlineKeyboardButton("↩ Retry now", callback_data=f"retry_now:{chat_id}"),
        ]])

        note = ""
        if attempt > 1:
            note = (
                "\n\n<i>All providers seem to be over quota. "
                "Consider adding fresh keys with /addkey.</i>"
            )

        async def _edit(remaining: int) -> None:
            bar_done = wait_seconds - remaining
            filled = "█" * bar_done + "░" * remaining
            txt = (
                f"⏳ <b>Rate limit reached</b> — retrying in <b>{remaining}s</b>\n"
                f"<code>{filled}</code>{note}"
            )
            try:
                await self._bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=message_id,
                    text=txt,
                    parse_mode="HTML",
                    reply_markup=kb,
                )
            except Exception:
                pass

        await _edit(wait_seconds)

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
                self._cleanup_countdown(channel_id)
                return "stop"
            if on_retry.is_set():
                self._cleanup_countdown(channel_id)
                return "retry"
            await _edit(remaining)

        self._cleanup_countdown(channel_id)
        return "done"

    def supports_interactive(self) -> bool:
        return True

    def format_text(self, text: str, mode: str = "HTML") -> str:
        if mode == "HTML":
            return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return text

    async def send_chunked(
        self, channel_id: str, text: str, max_len: int = 4000
    ) -> list[str]:
        chat_id = self._channel_to_chat_id(channel_id)
        handles: list[str] = []
        chunks = self._split_message(text, max_len)
        for i, chunk in enumerate(chunks):
            if i == 0:
                handles.append(await self.send_text(channel_id, chunk))
                continue
            sent = await self._bot.send_message(chat_id=chat_id, text=chunk, parse_mode="HTML")
            handles.append(f"tg:{chat_id}:{sent.message_id}")
        return handles

    async def send_typing(self, channel_id: str) -> None:
        chat_id = self._channel_to_chat_id(channel_id)
        try:
            await self._bot.send_chat_action(chat_id=chat_id, action="typing")
        except Exception:
            pass

    def get_channel_id(self, source: str, platform_id: str) -> str:
        return f"tg:{platform_id}"

    # ------------------------------------------------------------------
    # PTB callback handlers — called by app.py
    # ------------------------------------------------------------------

    def handle_stop(self, channel_id: str) -> None:
        """Called when user clicks the Stop button or sends /stop."""
        self._cancel_flags[channel_id] = True
        event = self._stop_events.get(channel_id)
        if event is not None:
            event.set()
        task = self._active_tasks.get(channel_id)
        if task is not None and not task.done():
            task.cancel()

    def handle_retry(self, channel_id: str) -> None:
        """Called when user clicks the Retry now button."""
        event = self._retry_events.get(channel_id)
        if event is not None:
            event.set()

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
    # Internal helpers
    # ------------------------------------------------------------------

    def _channel_to_chat_id(self, channel_id: str) -> int:
        return int(channel_id.split(":", 1)[1])

    def _chat_id_to_channel(self, chat_id: int) -> str:
        return f"tg:{chat_id}"

    def _cleanup_countdown(self, channel_id: str) -> None:
        self._retry_events.pop(channel_id, None)
        self._stop_events.pop(channel_id, None)

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
