from __future__ import annotations

import asyncio
import hashlib
import os
import threading
import time
from typing import Optional

from flask import Flask, jsonify, request

from src.config import Config
from src.interfaces.api.adapter import APIAdapter

app = Flask(__name__)

_adapter: Optional[APIAdapter] = None
_config: Optional[Config] = None
_loop: Optional[asyncio.AbstractEventLoop] = None


def _run_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Run a persistent event loop forever (daemon thread target)."""
    asyncio.set_event_loop(loop)
    loop.run_forever()


def _submit(coro) -> asyncio.Future:
    """Submit a coroutine to the persistent event loop from any thread."""
    global _loop
    if _loop is None:
        raise RuntimeError("Event loop not started")
    return asyncio.run_coroutine_threadsafe(coro, _loop)


def _get_or_create_user(api_key: str) -> tuple[int, str]:
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
    from src.db.key_store import get_or_create_user
    future = _submit(
        get_or_create_user("api", key_hash, username=f"api_{key_hash[:8]}")
    )
    user_id = future.result()
    return user_id, key_hash


def _build_runtime_context() -> str:
    import platform
    import time as _time
    parts = [
        f"Current time: {_time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"Host: {platform.node()}",
        f"OS: {platform.system()} {platform.release()}",
        f"Interface: API (REST)",
    ]
    return "\n".join(parts)


async def _process_message(
    adapter: APIAdapter,
    config: Config,
    channel_id: str,
    user_id: int,
    message: str,
    platform_user_id: str,
) -> list[str]:
    from src.bot.app import run_orchestration_background
    from src.core.event_sink import EventSink
    from src.db.chat_store import add_chat_message, get_chat_history
    from src.db.key_store import get_summary_data

    sink = EventSink(adapter)
    await add_chat_message(user_id, "user", message)

    summary_data = await get_summary_data(user_id)
    summary = summary_data["summary"] if summary_data else None
    last_msg_id = summary_data["last_msg_id"] if summary_data else 0
    history = await get_chat_history(
        user_id, limit=config.max_context_messages, after_id=last_msg_id
    )

    context_parts = [_build_runtime_context()]
    if summary:
        context_parts.append(f"[Earlier context summary]\n{summary}")
    for m in history[:-1]:
        context_parts.append(f"{m['role']}: {m['content']}")
    context_parts.append(f"user: {message}")
    context_str = "\n".join(context_parts)

    handle = f"api:{channel_id}:{user_id}"
    await run_orchestration_background(
        channel_id=channel_id,
        handle=handle,
        user_id=user_id,
        context_str=context_str,
        original_text=message,
        history=history,
        summary=summary,
        sink=sink,
    )

    return adapter.drain(channel_id)


# ------------------------------------------------------------------
# Flask routes
# ------------------------------------------------------------------


@app.route("/api/chat", methods=["POST"])
def chat():
    if _adapter is None:
        return jsonify({"error": "API adapter not initialized"}), 503

    data = request.get_json(silent=True)
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' in request body"}), 400

    message = data["message"].strip()
    if not message:
        return jsonify({"error": "Message cannot be empty"}), 400

    api_key = (
        request.headers.get("X-API-Key", "")
        or request.args.get("api_key", "")
        or "anonymous"
    )
    user_id, platform_user_id = _get_or_create_user(api_key)
    channel_id = f"api:{platform_user_id}"

    future = _submit(
        _process_message(
            _adapter, _config, channel_id, user_id, message, platform_user_id
        )
    )
    try:
        responses = future.result(timeout=180)
    except TimeoutError:
        return jsonify({"error": "Request timed out"}), 504
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify({
        "responses": responses,
        "channel_id": channel_id,
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "platforms": ["api"],
    })


# ------------------------------------------------------------------
# Server runner
# ------------------------------------------------------------------


def run_api_server(config: Config, adapter: Optional[APIAdapter] = None) -> None:
    """Start the Flask API server (blocking).

    Launches a persistent event loop in a daemon thread so async
    operations (orchestration, DB) share the same loop and can safely
    use asyncio.Lock and other loop-bound primitives.

    Can be called in a thread alongside other platforms (Telegram, etc.).
    """
    global _adapter, _config, _loop

    _config = config
    _adapter = adapter or APIAdapter()

    # Start persistent event loop in daemon thread
    _loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=_run_loop, args=(_loop,), daemon=True)
    loop_thread.start()

    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "5000"))
    debug = os.environ.get("API_DEBUG", "").lower() in ("1", "true")

    print(f"  → API server listening on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)
