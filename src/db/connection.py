"""DB connection layer — SQLite (default) or Postgres (POSTGRES_URL).

Backend is selected on first get_db() call, not at import time.
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
# Backend detection (lazy — resolved on first get_db() call)
# ------------------------------------------------------------------

_pg_pool = None
_db_backend: Optional[str] = None


def _detect_backend() -> str:
    global _db_backend
    if _db_backend is not None:
        return _db_backend
    postgres_url = (
        os.environ.get("POSTGRES_URL")
        or os.environ.get("POSTGRES_PRISMA_URL")
        or os.environ.get("POSTGRES_URL_NON_POOLING")
    )
    _db_backend = "postgres" if postgres_url else "sqlite"
    return _db_backend


def get_db_path() -> str:
    return os.environ.get("DATABASE_PATH", "./data/rk.db")


# ------------------------------------------------------------------
# Postgres pool (lazy)
# ------------------------------------------------------------------


async def _get_pg_pool():
    global _pg_pool
    if _pg_pool is not None:
        return _pg_pool
    import asyncpg
    from src.utils.logger import logger

    postgres_url = (
        os.environ.get("POSTGRES_URL")
        or os.environ.get("POSTGRES_PRISMA_URL")
        or os.environ.get("POSTGRES_URL_NON_POOLING")
    )
    if not postgres_url:
        raise RuntimeError("No POSTGRES_URL set")
    try:
        cfg_min = int(os.environ.get("VERCEL_PG_POOL_MIN", "2"))
        cfg_max = int(os.environ.get("VERCEL_PG_POOL_MAX", "10"))
        _pg_pool = await asyncpg.create_pool(
            postgres_url,
            min_size=cfg_min,
            max_size=cfg_max,
            command_timeout=60,
        )
        logger.info("pg_pool_created", min=cfg_min, max=cfg_max)
    except Exception as exc:
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
# get_db() — unified async context manager (lazy backend selection)
# ------------------------------------------------------------------


async def _get_db_sqlite():
    import aiosqlite
    db_path = get_db_path()
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = await aiosqlite.connect(db_path)
    await conn.execute("PRAGMA foreign_keys = ON;")
    return conn


async def _get_db_postgres():
    from src.db.pg_compat import PgConnectionWrapper
    pool = await _get_pg_pool()
    conn = await pool.acquire()
    tr = conn.transaction()
    await tr.start()
    return PgConnectionWrapper(conn, tr)


@asynccontextmanager
async def get_db():
    """Yield a DB connection matching the configured backend.
    Backend is detected on first call, not at import time.
    """
    backend = _detect_backend()

    if backend == "postgres":
        wrapper = await _get_db_postgres()
        try:
            yield wrapper
            await wrapper._tr.commit()
        except Exception:
            await wrapper._tr.rollback()
            raise
        finally:
            await wrapper._conn.close()

    else:
        import aiosqlite
        db_path = get_db_path()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(db_path) as conn:
            await conn.execute("PRAGMA foreign_keys = ON;")
            yield conn
