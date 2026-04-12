"""Event bus — per-session async queue connecting orchestrator to interfaces.

The orchestrator emits SessionEvent objects into the bus via emit().
Interfaces (Telegram, TUI, Web) consume events from get_bus().

Design:
- One asyncio.Queue per session (keyed by chat_id / session integer).
- The queue is bounded (200 items). If the consumer falls behind, the
  oldest event is dropped rather than blocking the orchestrator.
- teardown() must be called when a session ends to release memory.
- thread-safe: all operations are async, no locks needed.
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import AsyncIterator

from src.core.models import EventType, SessionEvent
from src.utils.logger import logger

_buses: dict[int, asyncio.Queue[SessionEvent]] = defaultdict(
    lambda: asyncio.Queue(maxsize=200)
)

_fanout: dict[int, list[asyncio.Queue[SessionEvent]]] = defaultdict(list)


def get_bus(session_id: int) -> asyncio.Queue[SessionEvent]:
    """Return the primary event queue for a session."""
    return _buses[session_id]


async def emit(session_id: int, event: SessionEvent) -> None:
    """Emit an event into a session's bus.

    Non-blocking: if the queue is full, the oldest item is dropped
    before inserting the new one. The orchestrator never blocks on emit.
    """
    q = _buses[session_id]
    if q.full():
        try:
            q.get_nowait()
            logger.warning("event_bus_dropped_oldest", session_id=session_id)
        except asyncio.QueueEmpty:
            pass
    try:
        q.put_nowait(event)
    except asyncio.QueueFull:
        logger.error("event_bus_put_failed", session_id=session_id, event_type=event.type.value)

    # Fan-out to additional subscribers (e.g., web socket handler)
    for extra_q in _fanout.get(session_id, []):
        try:
            extra_q.put_nowait(event)
        except asyncio.QueueFull:
            pass


def subscribe(session_id: int) -> asyncio.Queue[SessionEvent]:
    """Create and return an additional subscriber queue for fan-out.

    Used by the web interface (v3.0) to receive the same events as
    the primary Telegram consumer.
    """
    q: asyncio.Queue[SessionEvent] = asyncio.Queue(maxsize=200)
    _fanout[session_id].append(q)
    return q


def unsubscribe(session_id: int, q: asyncio.Queue[SessionEvent]) -> None:
    """Remove a subscriber queue."""
    try:
        _fanout[session_id].remove(q)
    except ValueError:
        pass


async def drain(session_id: int, timeout: float = 0.1) -> list[SessionEvent]:
    """Drain all currently queued events for a session (non-blocking)."""
    q = _buses[session_id]
    events: list[SessionEvent] = []
    try:
        while True:
            events.append(q.get_nowait())
    except asyncio.QueueEmpty:
        pass
    return events


async def iter_events(
    session_id: int,
    sentinel_type: EventType = EventType.MESSAGE,
) -> AsyncIterator[SessionEvent]:
    """Async generator that yields events until a MESSAGE(final=True) event."""
    q = _buses[session_id]
    while True:
        event = await q.get()
        yield event
        if (
            event.type == sentinel_type
            and event.payload.get("final", False)
        ) or event.type in (EventType.ERROR, EventType.CANCELLED):
            break


def teardown(session_id: int) -> None:
    """Release all resources for a session. Call when the session ends."""
    _buses.pop(session_id, None)
    _fanout.pop(session_id, None)
