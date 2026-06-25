"""Database operations for background agents and wake events."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from src.agents.agent_models import BackgroundAgentConfig, WakeSignal
from src.db.connection import get_db


async def save_background_agent(cfg: BackgroundAgentConfig) -> None:
    async with get_db() as db:
        await db.execute(
            """
            INSERT OR REPLACE INTO background_agents
              (id, user_id, channel_id, watcher_type, name, description, config,
               interval_seconds, enabled, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cfg.id,
                cfg.user_id,
                cfg.channel_id,
                cfg.watcher_type,
                cfg.name,
                cfg.description,
                json.dumps(cfg.config),
                cfg.interval_seconds,
                1 if cfg.enabled else 0,
                cfg.created_at,
            ),
        )
        await db.commit()


async def load_all_background_agents() -> List[BackgroundAgentConfig]:
    """Load all enabled background agents across all users (called on bot start)."""
    async with get_db() as db:
        cur = await db.execute(
            "SELECT id, user_id, channel_id, watcher_type, name, description, config, "
            "interval_seconds, enabled, created_at FROM background_agents WHERE enabled = 1"
        )
        rows = await cur.fetchall()
    return [_row_to_cfg(r) for r in rows]


async def list_user_background_agents(user_id: int) -> List[BackgroundAgentConfig]:
    async with get_db() as db:
        cur = await db.execute(
            "SELECT id, user_id, channel_id, watcher_type, name, description, config, "
            "interval_seconds, enabled, created_at FROM background_agents WHERE user_id = ?",
            (user_id,),
        )
        rows = await cur.fetchall()
    return [_row_to_cfg(r) for r in rows]


async def disable_background_agent(agent_id: str) -> None:
    async with get_db() as db:
        await db.execute(
            "UPDATE background_agents SET enabled = 0 WHERE id = ?", (agent_id,)
        )
        await db.commit()


async def update_agent_trigger_count(agent_id: str) -> None:
    async with get_db() as db:
        await db.execute(
            "UPDATE background_agents SET trigger_count = trigger_count + 1, "
            "last_triggered_at = datetime('now') WHERE id = ?",
            (agent_id,),
        )
        await db.commit()


async def save_wake_event(signal: WakeSignal, analysis: str) -> None:
    async with get_db() as db:
        await db.execute(
            "INSERT INTO wake_events (agent_id, user_id, event_type, severity, raw_data, ai_analysis, sent_to_user) "
            "VALUES (?, ?, ?, ?, ?, ?, 1)",
            (
                signal.agent_id,
                signal.user_id,
                signal.event_type,
                signal.severity,
                json.dumps(signal.raw_data),
                analysis,
            ),
        )
        await db.commit()


async def get_wake_events(user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
    async with get_db() as db:
        cur = await db.execute(
            "SELECT w.id, w.agent_id, w.event_type, w.severity, w.ai_analysis, w.created_at "
            "FROM wake_events w "
            "JOIN background_agents a ON w.agent_id = a.id "
            "WHERE a.user_id = ? ORDER BY w.id DESC LIMIT ?",
            (user_id, limit),
        )
        rows = await cur.fetchall()
    return [
        {
            "id": r[0],
            "agent_id": r[1],
            "event_type": r[2],
            "severity": r[3],
            "analysis": r[4],
            "at": r[5],
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _row_to_cfg(row) -> BackgroundAgentConfig:
    return BackgroundAgentConfig(
        id=row[0],
        user_id=row[1],
        channel_id=row[2],
        watcher_type=row[3],
        name=row[4],
        description=row[5],
        config=json.loads(row[6]) if row[6] else {},
        interval_seconds=row[7],
        enabled=bool(row[8]),
        created_at=row[9] or "",
    )
