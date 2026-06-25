"""chat_store — conversation history, memory, and summarization.

Token efficiency changes:
- get_pinned_memories(): returns only pinned=1 rows (always injected, max 5)
- get_relevant_memories(): semantic search via vector store (top-k by relevance)
- get_skill(): load a single skill by name (lazy load, not bulk inject)
- list_skill_names(): returns just names for the use_skill tool description
- Incremental summarization: summarize oldest N messages, not full history
- access_count / last_accessed tracking for memory pruning
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.db.connection import get_db
from src.db.vector_store import vector_store


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------

async def add_chat_message(
    user_id: int,
    role: str,
    content: str,
    metadata: Optional[Dict] = None,
) -> None:
    async with get_db() as db:
        await db.execute(
            "INSERT INTO chat_history (user_id, role, content, metadata) VALUES (?, ?, ?, ?)",
            (user_id, role, content, json.dumps(metadata) if metadata else None),
        )
        await db.commit()
    if role in ("user", "assistant"):
        asyncio.create_task(
            vector_store.add_memory(
                user_id=user_id,
                text=content,
                metadata={"role": role, "timestamp": _utcnow()},
            )
        )


async def get_chat_history(
    user_id: int, limit: int = 10, after_id: int = 0
) -> List[Dict]:
    async with get_db() as db:
        cur = await db.execute(
            "SELECT role, content, id, metadata FROM chat_history "
            "WHERE user_id = ? AND id > ? ORDER BY id DESC LIMIT ?",
            (user_id, after_id, limit),
        )
        rows = await cur.fetchall()
    return [
        {
            "role": r[0],
            "content": r[1],
            "id": r[2],
            "metadata": json.loads(r[3]) if r[3] else None,
        }
        for r in reversed(rows)
    ]


async def count_messages_since(user_id: int, after_id: int) -> int:
    """Count messages since the last summarization checkpoint."""
    async with get_db() as db:
        cur = await db.execute(
            "SELECT COUNT(*) FROM chat_history WHERE user_id = ? AND id > ?",
            (user_id, after_id),
        )
        row = await cur.fetchone()
    return row[0] if row else 0


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------

async def get_summary_data(user_id: int) -> Optional[Dict]:
    async with get_db() as db:
        cur = await db.execute(
            "SELECT summary, last_msg_id FROM chat_summaries WHERE user_id = ?",
            (user_id,),
        )
        row = await cur.fetchone()
    return {"summary": row[0], "last_msg_id": row[1]} if row else None


async def update_summary(user_id: int, summary: str, last_msg_id: int) -> None:
    async with get_db() as db:
        await db.execute(
            "INSERT OR REPLACE INTO chat_summaries "
            "(user_id, summary, last_msg_id, updated_at) VALUES (?, ?, ?, datetime('now'))",
            (user_id, summary, last_msg_id),
        )
        await db.commit()


# ---------------------------------------------------------------------------
# Memory — tiered injection
# ---------------------------------------------------------------------------

async def save_rika_memory(
    user_id: int,
    key: str,
    value: str,
    mem_type: str = "memory",
    pinned: bool = False,
) -> None:
    token_est = _estimate_tokens(f"{key}: {value}")
    async with get_db() as db:
        await db.execute(
            "INSERT OR REPLACE INTO rika_memory "
            "(user_id, mem_key, mem_value, mem_type, pinned, token_estimate) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, key, value, mem_type, 1 if pinned else 0, token_est),
        )
        await db.commit()
    # Also index in vector store for semantic retrieval
    asyncio.create_task(
        vector_store.add_memory(
            user_id=user_id,
            text=f"{key}: {value}",
            metadata={"mem_type": mem_type, "key": key, "pinned": pinned},
        )
    )


async def pin_memory(user_id: int, key: str, mem_type: str = "memory") -> bool:
    """Pin a memory so it's always injected. Returns True if found."""
    async with get_db() as db:
        cur = await db.execute(
            "UPDATE rika_memory SET pinned = 1 "
            "WHERE user_id = ? AND mem_key = ? AND mem_type = ?",
            (user_id, key, mem_type),
        )
        await db.commit()
        return cur.rowcount > 0


async def unpin_memory(user_id: int, key: str, mem_type: str = "memory") -> bool:
    async with get_db() as db:
        cur = await db.execute(
            "UPDATE rika_memory SET pinned = 0 "
            "WHERE user_id = ? AND mem_key = ? AND mem_type = ?",
            (user_id, key, mem_type),
        )
        await db.commit()
        return cur.rowcount > 0


async def get_pinned_memories(user_id: int) -> Dict[str, str]:
    """Always-injected memories (pinned=1). Keep under 5 for token sanity."""
    async with get_db() as db:
        cur = await db.execute(
            "SELECT mem_key, mem_value FROM rika_memory "
            "WHERE user_id = ? AND pinned = 1 AND mem_type = 'memory' "
            "ORDER BY last_accessed DESC LIMIT 5",
            (user_id,),
        )
        rows = await cur.fetchall()
    if rows:
        await _touch_memories(user_id, [r[0] for r in rows])
    return {r[0]: r[1] for r in rows}


async def get_relevant_memories(
    user_id: int, query: str, limit: int = 4
) -> Dict[str, str]:
    """Semantically retrieve the most relevant non-pinned memories for this query."""
    try:
        results = await vector_store.search_memories(user_id, query, limit=limit + 4)
        memories: Dict[str, str] = {}
        keys_found: List[str] = []
        for r in results:
            meta = r.get("metadata", {})
            if meta.get("mem_type") == "memory" and not meta.get("pinned"):
                key = meta.get("key", "")
                if key and key not in memories:
                    # Fetch actual value from DB (vector store only stores text)
                    val = await _get_memory_value(user_id, key)
                    if val:
                        memories[key] = val
                        keys_found.append(key)
            if len(memories) >= limit:
                break
        if keys_found:
            await _touch_memories(user_id, keys_found)
        return memories
    except Exception:
        return {}


async def _get_memory_value(user_id: int, key: str) -> Optional[str]:
    async with get_db() as db:
        cur = await db.execute(
            "SELECT mem_value FROM rika_memory WHERE user_id = ? AND mem_key = ? AND mem_type = 'memory'",
            (user_id, key),
        )
        row = await cur.fetchone()
    return row[0] if row else None


async def _touch_memories(user_id: int, keys: List[str]) -> None:
    """Update access_count and last_accessed for accessed memories."""
    if not keys:
        return
    placeholders = ",".join("?" * len(keys))
    async with get_db() as db:
        await db.execute(
            f"UPDATE rika_memory SET access_count = access_count + 1, "
            f"last_accessed = datetime('now') "
            f"WHERE user_id = ? AND mem_key IN ({placeholders})",
            [user_id] + keys,
        )
        await db.commit()


async def prune_stale_memories(user_id: int, keep: int = 100) -> int:
    """Remove least-accessed memories beyond `keep` count. Returns rows deleted."""
    async with get_db() as db:
        cur = await db.execute(
            "SELECT COUNT(*) FROM rika_memory WHERE user_id = ? AND mem_type = 'memory' AND pinned = 0",
            (user_id,),
        )
        row = await cur.fetchone()
        count = row[0] if row else 0
        if count <= keep:
            return 0
        to_delete = count - keep
        del_cur = await db.execute(
            "DELETE FROM rika_memory WHERE user_id = ? AND mem_type = 'memory' AND pinned = 0 "
            "AND mem_key IN ("
            "  SELECT mem_key FROM rika_memory WHERE user_id = ? AND mem_type = 'memory' AND pinned = 0 "
            "  ORDER BY COALESCE(last_accessed, created_at) ASC LIMIT ?"
            ")",
            (user_id, user_id, to_delete),
        )
        await db.commit()
        return del_cur.rowcount


# ---------------------------------------------------------------------------
# Skills — lazy load (never bulk-injected)
# ---------------------------------------------------------------------------

async def get_skill(user_id: int, skill_name: str) -> Optional[str]:
    """Load a single skill by exact name. Used by use_skill tool."""
    async with get_db() as db:
        cur = await db.execute(
            "SELECT mem_value FROM rika_memory WHERE user_id = ? AND mem_key = ? AND mem_type = 'skill'",
            (user_id, skill_name),
        )
        row = await cur.fetchone()
    if row:
        await _touch_memories(user_id, [skill_name])
        return row[0]
    return None


async def list_skill_names(user_id: int) -> List[str]:
    """Return just skill names — for building the use_skill tool description."""
    async with get_db() as db:
        cur = await db.execute(
            "SELECT mem_key FROM rika_memory WHERE user_id = ? AND mem_type = 'skill' ORDER BY mem_key",
            (user_id,),
        )
        rows = await cur.fetchall()
    return [r[0] for r in rows]


async def save_skill(user_id: int, name: str, code: str) -> None:
    """Save a skill (same as save_rika_memory with mem_type='skill')."""
    await save_rika_memory(user_id, name, code, mem_type="skill")


# ---------------------------------------------------------------------------
# Generic helpers (backward compat)
# ---------------------------------------------------------------------------

async def get_rika_memories(user_id: int, mem_type: str = "memory") -> Dict[str, str]:
    """Legacy: returns ALL memories of a type. Use get_relevant_memories instead."""
    async with get_db() as db:
        cur = await db.execute(
            "SELECT mem_key, mem_value FROM rika_memory WHERE user_id = ? AND mem_type = ?",
            (user_id, mem_type),
        )
        rows = await cur.fetchall()
    return {r[0]: r[1] for r in rows}


async def delete_rika_memory(user_id: int, key: str, mem_type: str = "memory") -> None:
    async with get_db() as db:
        await db.execute(
            "DELETE FROM rika_memory WHERE user_id = ? AND mem_key = ? AND mem_type = ?",
            (user_id, key, mem_type),
        )
        await db.commit()


async def list_rika_memories(user_id: int) -> List[Dict]:
    async with get_db() as db:
        cur = await db.execute(
            "SELECT mem_key, mem_value, mem_type, pinned, access_count, created_at "
            "FROM rika_memory WHERE user_id = ? ORDER BY mem_type, pinned DESC, mem_key",
            (user_id,),
        )
        rows = await cur.fetchall()
    return [
        {
            "key": r[0], "value": r[1], "type": r[2],
            "pinned": bool(r[3]), "access_count": r[4], "created_at": r[5],
        }
        for r in rows
    ]
