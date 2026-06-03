import aiosqlite
from pathlib import Path
import asyncio

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


async def apply_migrations(db_path: str):
    # Route to Postgres migration when POSTGRES_URL is present
    from src.db.connection import DB_BACKEND
    if DB_BACKEND == "postgres":
        from src.db.pg_migrate import apply_pg_migrations
        from src.db.connection import _get_pg_pool
        pool = await _get_pg_pool()
        await apply_pg_migrations(pool)
        return

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(db_path) as conn:
        await conn.execute("PRAGMA foreign_keys = ON;")
        # Ensure migrations table exists (created by migration file too, but be defensive)
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS migrations (
            name TEXT PRIMARY KEY,
            applied_at TEXT DEFAULT (datetime('now'))
        );
        """)
        await conn.commit()

        for sql_file in sorted(MIGRATIONS_DIR.glob("*.sql")):
            name = sql_file.name
            cur = await conn.execute("SELECT 1 FROM migrations WHERE name = ?", (name,))
            row = await cur.fetchone()
            if row:
                continue
            sql = sql_file.read_text()
            await conn.executescript(sql)
            await conn.execute("INSERT INTO migrations(name) VALUES(?)", (name,))
            await conn.commit()


def run_sync(db_path: str = "./data/rk.db"):
    # Use asyncio.run() which is compatible with modern Python event loop policy
    asyncio.run(apply_migrations(db_path))


if __name__ == "__main__":
    run_sync()
