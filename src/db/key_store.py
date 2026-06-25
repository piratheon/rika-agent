import hashlib
from typing import List, Optional
from .migrate import apply_migrations
from ..crypto import encrypt, decrypt
from .connection import get_db, get_db_path

async def init_db():
    await apply_migrations(get_db_path())


async def upsert_user(
    platform: str,
    platform_user_id: str,
    username: Optional[str] = None,
) -> int:
    async with get_db() as conn:
        cur = await conn.execute(
            "SELECT id FROM users WHERE platform = ? AND platform_user_id = ?",
            (platform, platform_user_id),
        )
        row = await cur.fetchone()
        if row:
            uid = row[0]
            if username:
                await conn.execute(
                    "UPDATE users SET username = ?, last_active_at = datetime('now') WHERE id = ?",
                    (username, uid),
                )
                await conn.commit()
            return uid

        cur = await conn.execute(
            "INSERT INTO users(platform, platform_user_id, username, last_active_at) VALUES(?,?,?,datetime('now'))",
            (platform, platform_user_id, username),
        )
        await conn.commit()
        return cur.lastrowid


def _hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def add_api_key(user_id: int, provider: str, raw_key: str) -> int:
    key_hash = _hash_key(raw_key)
    encrypted = encrypt(raw_key.encode("utf-8"))
    async with get_db() as conn:
        cur = await conn.execute(
            "INSERT INTO api_keys(user_id, provider, key_hash, key_encrypted, created_at) VALUES(?,?,?,?,datetime('now'))",
            (user_id, provider, key_hash, encrypted),
        )
        await conn.commit()
        return cur.lastrowid


async def list_api_keys(user_id: int) -> List[dict]:
    out = []
    async with get_db() as conn:
        cur = await conn.execute(
            "SELECT id, provider, key_hash, is_blacklisted, created_at, quota_resets_at, last_used_at FROM api_keys WHERE user_id = ?",
            (user_id,),
        )
        rows = await cur.fetchall()
        for r in rows:
            out.append({
                "id": r[0],
                "provider": r[1],
                "key_hash": r[2],
                "is_blacklisted": bool(r[3]),
                "created_at": r[4],
                "quota_resets_at": r[5],
                "last_used_at": r[6],
            })
    return out


async def get_api_key_raw(key_id: int) -> bytes:
    """Return the decrypted raw API key bytes for a given api_keys.id."""
    async with get_db() as conn:
        cur = await conn.execute("SELECT key_encrypted FROM api_keys WHERE id = ?", (key_id,))
        row = await cur.fetchone()
        if not row:
            raise KeyError(f"api_key id={key_id} not found")
        blob = row[0]
        # decrypt using crypto.decrypt
        try:
            raw = decrypt(blob)
            return raw
        except Exception as e:
            raise RuntimeError("Failed to decrypt API key") from e


async def delete_user_by_platform(platform: str, platform_user_id: str) -> int:
    """Delete a user and cascade-delete related rows. Returns number of rows deleted."""
    async with get_db() as conn:
        cur = await conn.execute(
            "SELECT id FROM users WHERE platform = ? AND platform_user_id = ?",
            (platform, platform_user_id),
        )
        row = await cur.fetchone()
        if not row:
            return 0
        user_id = row[0]
        await conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        await conn.commit()
        return 1


async def list_blacklisted_due() -> list[int]:
    async with get_db() as conn:
        cur = await conn.execute(
            "SELECT id FROM api_keys WHERE is_blacklisted = 1 AND quota_resets_at IS NOT NULL AND datetime(quota_resets_at) <= datetime('now')"
        )
        rows = await cur.fetchall()
        return [r[0] for r in rows]


async def blacklist_key(key_id: int, reason: str = "quota_exceeded", quota_resets_at: str | None = None) -> None:
    async with get_db() as conn:
        await conn.execute(
            "UPDATE api_keys SET is_blacklisted = 1, blacklisted_at = datetime('now'), quota_resets_at = ?, last_used_at = last_used_at WHERE id = ?",
            (quota_resets_at, key_id),
        )
        await conn.execute(
            "INSERT INTO key_blacklist_log(api_key_id, reason, blacklisted_at) VALUES(?,?,datetime('now'))",
            (key_id, reason),
        )
        await conn.commit()


async def unblacklist_key(key_id: int) -> None:
    async with get_db() as conn:
        await conn.execute(
            "UPDATE api_keys SET is_blacklisted = 0, blacklisted_at = NULL, quota_resets_at = NULL WHERE id = ?",
            (key_id,),
        )
        await conn.execute(
            "UPDATE key_blacklist_log SET unblacklisted_at = datetime('now') WHERE api_key_id = ? AND unblacklisted_at IS NULL",
            (key_id,),
        )
        await conn.commit()


async def update_key_last_used(key_id: int) -> None:
    async with get_db() as conn:
        await conn.execute(
            "UPDATE api_keys SET last_used_at = datetime('now') WHERE id = ?", (key_id,)
        )
        await conn.commit()


async def increment_tokens_used(key_id: int, tokens: int) -> None:
    async with get_db() as conn:
        await conn.execute(
            "UPDATE api_keys SET tokens_used_today = tokens_used_today + ? WHERE id = ?",
            (tokens, key_id),
        )
        await conn.commit()
