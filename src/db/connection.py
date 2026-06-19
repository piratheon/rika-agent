"""DB connection layer — SQLite (default) or Postgres (Vercel / POSTGRES_URL).

Backend is selected at import time via environment variables:
  POSTGRES_URL              -> Postgres via asyncpg (preferred on Vercel)
  POSTGRES_PRISMA_URL       -> same, Prisma-style URL (fallback)
  POSTGRES_URL_NON_POOLING  -> same, direct connection URL (fallback)
  DATABASE_PATH             -> SQLite path (default: ./data/rk.db)

The Postgres path uses a connection pool (min 2, max 10) initialised lazily
on first use and reused for the lifetime of the process.

All store files (key_store, chat_store, background_store) use get_db() and
see the same cursor interface regardless of backend.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

# ------------------------------------------------------------------
# Backend detection
# ------------------------------------------------------------------

_POSTGRES_URL: Optional[str] = (
    os.environ.get("POSTGRES_URL")
    or os.environ.get("POSTGRES_PRISMA_URL")
    or os.environ.get("POSTGRES_URL_NON_POOLING")
)
DB_BACKEND: str = "postgres" if _POSTGRES_URL else "sqlite"
DB_PATH: str = os.environ.get("DATABASE_PATH", "./data/rk.db")

# ------------------------------------------------------------------
# Postgres pool (lazy)
# ------------------------------------------------------------------

_pg_pool = None


async def _get_pg_pool():
    global _pg_pool
    if _pg_pool is not None:
        return _pg_pool
    import asyncpg  # optional dependency — only needed when POSTGRES_URL is set
    from src.utils.logger import logger
    try:
        cfg_min = int(os.environ.get("VERCEL_PG_POOL_MIN", "2"))
        cfg_max = int(os.environ.get("VERCEL_PG_POOL_MAX", "10"))
        _pg_pool = await asyncpg.create_pool(
            _POSTGRES_URL,
            min_size=cfg_min,
            max_size=cfg_max,
            command_timeout=60,
        )
        logger.info("pg_pool_created", min=cfg_min, max=cfg_max)
    except Exception as exc:
        from src.utils.logger import logger
        logger.error("pg_pool_failed", error=str(exc))
        raise
    return _pg_pool


async def close_pg_pool() -> None:
    """Gracefully close the Postgres connection pool on shutdown."""
    global _pg_pool
    if _pg_pool is not None:
        await _pg_pool.close()
        _pg_pool = None


# ------------------------------------------------------------------
# get_db() — unified async context manager
# ------------------------------------------------------------------

if DB_BACKEND == "postgres":
    @asynccontextmanager
    async def get_db():
        """Yield a PgConnectionWrapper inside an asyncpg transaction."""
        from src.db.pg_compat import PgConnectionWrapper
        pool = await _get_pg_pool()
        async with pool.acquire() as conn:
            tr = conn.transaction()
            await tr.start()
            wrapper = PgConnectionWrapper(conn, tr)
            try:
                yield wrapper
                await tr.commit()
            except Exception:
                await tr.rollback()
                raise

else:
    import aiosqlite

    @asynccontextmanager
    async def get_db():
        """Yield an aiosqlite connection (SQLite backend)."""
        Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(DB_PATH) as conn:
            await conn.execute("PRAGMA foreign_keys = ON;")
            yield conn
