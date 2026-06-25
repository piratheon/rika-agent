
"""Postgres compatibility shim — mimics the aiosqlite cursor/connection API.

Translates SQLite SQL to Postgres:
  ?             -> $1, $2, ...      (positional parameters)
  datetime('now') -> NOW()          (timestamp function)
  INSERT OR REPLACE -> INSERT ... ON CONFLICT DO UPDATE SET ...
  AUTOINCREMENT    -> handled in schema (BIGSERIAL)

All existing store files (key_store, chat_store, background_store) work
without modification — they see the same cursor interface they used with
aiosqlite.
"""
from __future__ import annotations

import re
from typing import Any, List, Optional, Sequence

# Per-table upsert conflict columns for INSERT OR REPLACE -> ON CONFLICT translation.
# Key: table name (lowercase). Value: conflict column expression (with parens).
_UPSERT_CONFLICTS: dict[str, str] = {
    "chat_summaries":    "(user_id)",
    "rika_memory":       "(user_id, mem_key, mem_type)",
    "background_agents": "(id)",
}


# ---------------------------------------------------------------------------
# SQL translation
# ---------------------------------------------------------------------------

def _replace_placeholders(sql: str) -> str:
    """Replace SQLite ? placeholders with Postgres $N positional parameters."""
    counter = 0
    out: List[str] = []
    for ch in sql:
        if ch == "?":
            counter += 1
            out.append(f"${counter}")
        else:
            out.append(ch)
    return "".join(out)


def _replace_datetime(sql: str) -> str:
    sql = re.sub(r"datetime\(\'now\'\)", "NOW()", sql, flags=re.IGNORECASE)
    sql = re.sub(r"date\(\'now\'\)", "CURRENT_DATE", sql, flags=re.IGNORECASE)
    return sql


def _safe_ident(name: str) -> str:
    """Validate a SQL identifier contains only safe characters.
    Raises ValueError if unsafe characters are found.
    """
    import string as _string
    allowed = set(_string.ascii_letters + _string.digits + "_")
    if not all(c in allowed for c in name):
        raise ValueError(f"Unsafe SQL identifier: {name!r}")
    return name


def _translate_insert_or_replace(sql: str) -> str:
    """Translate INSERT OR REPLACE INTO to INSERT ... ON CONFLICT DO UPDATE."""
    pattern = re.compile(
        r"INSERT\s+OR\s+REPLACE\s+INTO\s+(\w+)\s*\(([^)]+)\)\s*VALUES\s*(\([^)]+\))",
        re.IGNORECASE | re.DOTALL,
    )
    m = pattern.search(sql)
    if not m:
        return sql

    table = _safe_ident(m.group(1).strip().lower())
    cols_raw = m.group(2).strip()
    cols = [_safe_ident(c.strip()) for c in cols_raw.split(",")]
    vals = m.group(3).strip()
    conflict_expr = _UPSERT_CONFLICTS.get(table)

    if not conflict_expr:
        cols_str = ", ".join(cols)
        pg = f"INSERT INTO {table} ({cols_str}) VALUES {vals} ON CONFLICT DO NOTHING"
        return sql[: m.start()] + pg + sql[m.end() :]

    conflict_cols = {c.strip() for c in conflict_expr.strip("()").split(",")}
    update_cols = [c for c in cols if c not in conflict_cols]

    if not update_cols:
        cols_str = ", ".join(cols)
        pg = (
            f"INSERT INTO {table} ({cols_str}) VALUES {vals} "
            f"ON CONFLICT {conflict_expr} DO NOTHING"
        )
    else:
        set_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)
        cols_str = ", ".join(cols)
        pg = (
            f"INSERT INTO {table} ({cols_str}) VALUES {vals} "
            f"ON CONFLICT {conflict_expr} DO UPDATE SET {set_clause}"
        )

    return sql[: m.start()] + pg + sql[m.end() :]


def translate_sql(sql: str) -> str:
    """Full SQLite -> Postgres translation pipeline."""
    sql = _translate_insert_or_replace(sql)
    sql = _replace_datetime(sql)
    sql = _replace_placeholders(sql)
    return sql


# ---------------------------------------------------------------------------
# Cursor / connection wrappers
# ---------------------------------------------------------------------------

class PgCursor:
    """Mimics aiosqlite cursor: supports fetchone(), fetchall(), lastrowid."""

    __slots__ = ("_rows", "lastrowid")

    def __init__(self, rows: List[Any], last_id: Optional[int] = None) -> None:
        self._rows = rows
        self.lastrowid: Optional[int] = last_id

    async def fetchone(self) -> Optional[Any]:
        return self._rows[0] if self._rows else None

    async def fetchall(self) -> List[Any]:
        return list(self._rows)


class PgConnectionWrapper:
    """Wraps asyncpg connection + transaction to look like an aiosqlite connection."""

    def __init__(self, conn: Any, transaction: Any) -> None:
        self._conn = conn
        self._tr = transaction

    async def execute(self, sql: str, params: Sequence[Any] = ()) -> PgCursor:
        pg_sql = translate_sql(sql)
        stripped = pg_sql.strip().upper()
        is_insert = stripped.startswith("INSERT")
        is_select = stripped.startswith("SELECT")

        # Append RETURNING id to INSERT so we can populate lastrowid
        if is_insert and "RETURNING" not in stripped:
            pg_sql = pg_sql.rstrip().rstrip(";") + " RETURNING id"

        if is_insert or is_select:
            rows = list(await self._conn.fetch(pg_sql, *params))
            last_id: Optional[int] = None
            if is_insert and rows:
                try:
                    last_id = rows[0]["id"]
                except (KeyError, IndexError):
                    pass
            return PgCursor(rows, last_id)

        # UPDATE, DELETE, CREATE, DROP, ALTER, etc.
        await self._conn.execute(pg_sql, *params)
        return PgCursor([])

    async def executemany(self, sql: str, params_seq: Sequence[Sequence[Any]]) -> None:
        pg_sql = translate_sql(sql)
        await self._conn.executemany(pg_sql, [list(p) for p in params_seq])

    async def executescript(self, sql: str) -> None:
        """Split on ; and execute each statement individually."""
        for stmt in sql.split(";"):
            stmt = stmt.strip()
            if not stmt:
                continue
            upper = stmt.upper()
            if upper.startswith(("BEGIN", "COMMIT", "ROLLBACK")):
                continue
            try:
                await self._conn.execute(translate_sql(stmt))
            except Exception:
                pass  # CREATE TABLE IF NOT EXISTS guards handle most conflicts

    async def commit(self) -> None:
        """No-op — the transaction commits when the get_db() context exits cleanly."""
