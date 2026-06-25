"""Postgres schema bootstrap — applies all tables in one pass.

Equivalent to all SQLite migration files combined, but in Postgres syntax.
Uses IF NOT EXISTS / IF NOT EXISTS guards so it is safe to run on every
startup (idempotent).

Called by migrate.apply_migrations() when the DB backend is Postgres.
"""
from __future__ import annotations

import asyncpg
from src.utils.logger import logger

# All CREATE TABLE and CREATE INDEX statements — Postgres syntax.
# Differences from SQLite:
#   INTEGER PRIMARY KEY AUTOINCREMENT -> BIGSERIAL PRIMARY KEY
#   TEXT PRIMARY KEY                  -> TEXT PRIMARY KEY (unchanged)
#   BLOB                              -> BYTEA
#   TEXT DEFAULT (datetime('now'))    -> TIMESTAMPTZ DEFAULT NOW()
#   INTEGER 0/1 bool columns          -> SMALLINT (preserved for code compat)
#   ALTER TABLE ... ADD COLUMN        -> ADD COLUMN IF NOT EXISTS
_DDL_STATEMENTS = [
    # ---- migrations tracker ----
    """
    CREATE TABLE IF NOT EXISTS migrations (
        name       TEXT PRIMARY KEY,
        applied_at TIMESTAMPTZ DEFAULT NOW()
    )
    """,

    # ---- users ----
    """
    CREATE TABLE IF NOT EXISTS users (
        id                BIGSERIAL PRIMARY KEY,
        platform          TEXT NOT NULL DEFAULT 'telegram',
        platform_user_id  TEXT NOT NULL,
        username          TEXT,
        created_at        TIMESTAMPTZ DEFAULT NOW(),
        last_active_at    TIMESTAMPTZ,
        UNIQUE(platform, platform_user_id)
    )
    """,

    # ---- api_keys ----
    """
    CREATE TABLE IF NOT EXISTS api_keys (
        id               BIGSERIAL PRIMARY KEY,
        user_id          BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        provider         TEXT NOT NULL,
        key_hash         TEXT NOT NULL,
        key_encrypted    BYTEA NOT NULL,
        is_blacklisted   SMALLINT DEFAULT 0,
        blacklisted_at   TIMESTAMPTZ,
        quota_resets_at  TIMESTAMPTZ,
        last_used_at     TIMESTAMPTZ,
        tokens_used_today BIGINT DEFAULT 0,
        created_at       TIMESTAMPTZ DEFAULT NOW()
    )
    """,

    # ---- conversations ----
    """
    CREATE TABLE IF NOT EXISTS conversations (
        id              BIGSERIAL PRIMARY KEY,
        user_id         BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        created_at      TIMESTAMPTZ DEFAULT NOW(),
        last_message_at TIMESTAMPTZ
    )
    """,

    # ---- messages ----
    """
    CREATE TABLE IF NOT EXISTS messages (
        id              BIGSERIAL PRIMARY KEY,
        conversation_id BIGINT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
        role            TEXT NOT NULL,
        content         TEXT,
        agent_name      TEXT,
        token_count     BIGINT,
        created_at      TIMESTAMPTZ DEFAULT NOW()
    )
    """,

    # ---- agent_tasks ----
    """
    CREATE TABLE IF NOT EXISTS agent_tasks (
        id              BIGSERIAL PRIMARY KEY,
        conversation_id BIGINT REFERENCES conversations(id),
        parent_task_id  BIGINT,
        agent_spec      TEXT,
        status          TEXT,
        input           TEXT,
        output          TEXT,
        error           TEXT,
        started_at      TIMESTAMPTZ,
        finished_at     TIMESTAMPTZ
    )
    """,

    # ---- key_blacklist_log ----
    """
    CREATE TABLE IF NOT EXISTS key_blacklist_log (
        id               BIGSERIAL PRIMARY KEY,
        api_key_id       BIGINT NOT NULL REFERENCES api_keys(id) ON DELETE CASCADE,
        reason           TEXT,
        blacklisted_at   TIMESTAMPTZ DEFAULT NOW(),
        unblacklisted_at TIMESTAMPTZ
    )
    """,

    # ---- chat_history ----
    """
    CREATE TABLE IF NOT EXISTS chat_history (
        id        BIGSERIAL PRIMARY KEY,
        user_id   BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        role      TEXT NOT NULL,
        content   TEXT NOT NULL,
        metadata  TEXT,
        timestamp TIMESTAMPTZ DEFAULT NOW()
    )
    """,

    # ---- chat_summaries ----
    """
    CREATE TABLE IF NOT EXISTS chat_summaries (
        user_id    BIGINT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
        summary    TEXT NOT NULL,
        last_msg_id BIGINT NOT NULL,
        updated_at TIMESTAMPTZ DEFAULT NOW()
    )
    """,

    # ---- rika_memory ----
    """
    CREATE TABLE IF NOT EXISTS rika_memory (
        id             BIGSERIAL PRIMARY KEY,
        user_id        BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        mem_key        TEXT NOT NULL,
        mem_value      TEXT NOT NULL,
        mem_type       TEXT NOT NULL DEFAULT 'memory',
        created_at     TIMESTAMPTZ DEFAULT NOW(),
        pinned         SMALLINT NOT NULL DEFAULT 0,
        access_count   BIGINT NOT NULL DEFAULT 0,
        last_accessed  TIMESTAMPTZ,
        token_estimate BIGINT NOT NULL DEFAULT 0,
        UNIQUE (user_id, mem_key, mem_type)
    )
    """,

    # ---- background_agents ----
    """
    CREATE TABLE IF NOT EXISTS background_agents (
        id               TEXT PRIMARY KEY,
        user_id          BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        channel_id       TEXT NOT NULL,
        watcher_type     TEXT NOT NULL,
        name             TEXT NOT NULL,
        description      TEXT NOT NULL DEFAULT '',
        config           TEXT NOT NULL DEFAULT '{}',
        interval_seconds BIGINT NOT NULL DEFAULT 60,
        enabled          SMALLINT NOT NULL DEFAULT 1,
        last_triggered_at TIMESTAMPTZ,
        trigger_count    BIGINT NOT NULL DEFAULT 0,
        created_at       TIMESTAMPTZ DEFAULT NOW()
    )
    """,

    # ---- wake_events ----
    """
    CREATE TABLE IF NOT EXISTS wake_events (
        id          BIGSERIAL PRIMARY KEY,
        agent_id    TEXT NOT NULL REFERENCES background_agents(id) ON DELETE CASCADE,
        user_id     BIGINT NOT NULL,
        event_type  TEXT NOT NULL,
        severity    TEXT NOT NULL DEFAULT 'warning',
        raw_data    TEXT NOT NULL DEFAULT '{}',
        ai_analysis TEXT,
        sent_to_user SMALLINT NOT NULL DEFAULT 0,
        created_at  TIMESTAMPTZ DEFAULT NOW()
    )
    """,

    # ---- command_audit ----
    """
    CREATE TABLE IF NOT EXISTS command_audit (
        id                 BIGSERIAL PRIMARY KEY,
        user_id            BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        command            TEXT NOT NULL,
        exit_code          BIGINT,
        stdout_head        TEXT,
        stderr_head        TEXT,
        was_blocked        SMALLINT NOT NULL DEFAULT 0,
        block_reason       TEXT,
        block_severity     TEXT,
        confirmed_override SMALLINT NOT NULL DEFAULT 0,
        workspace_path     TEXT,
        executed_at        TIMESTAMPTZ DEFAULT NOW()
    )
    """,

    # ---- indexes ----
    "CREATE INDEX IF NOT EXISTS idx_background_agents_user ON background_agents(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_wake_events_agent ON wake_events(agent_id)",
    "CREATE INDEX IF NOT EXISTS idx_command_audit_user ON command_audit(user_id, executed_at)",
    "CREATE INDEX IF NOT EXISTS idx_rika_memory_pinned ON rika_memory(user_id, pinned, mem_type)",
    "CREATE INDEX IF NOT EXISTS idx_rika_memory_accessed ON rika_memory(user_id, last_accessed)",
]

_SCHEMA_MIGRATION_NAME = "pg_schema_v2.2.0"


async def apply_pg_migrations(pool: asyncpg.Pool) -> None:
    """Idempotently apply the full Postgres schema."""
    async with pool.acquire() as conn:
        async with conn.transaction():
            # Ensure migrations table exists first
            await conn.execute(_DDL_STATEMENTS[0])

            # Check if schema has already been applied
            row = await conn.fetchrow(
                "SELECT 1 FROM migrations WHERE name = $1",
                _SCHEMA_MIGRATION_NAME,
            )
            if row:
                logger.info("pg_schema_already_applied", migration=_SCHEMA_MIGRATION_NAME)
                return

            # Apply all DDL
            for stmt in _DDL_STATEMENTS:
                stmt = stmt.strip()
                if stmt:
                    await conn.execute(stmt)

            # Record the migration
            await conn.execute(
                "INSERT INTO migrations (name) VALUES ($1) ON CONFLICT DO NOTHING",
                _SCHEMA_MIGRATION_NAME,
            )

    logger.info("pg_schema_applied", migration=_SCHEMA_MIGRATION_NAME)
