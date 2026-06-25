-- Migration v008: Multi-platform identity + channel_id
--
-- Replaces telegram_user_id with (platform, platform_user_id) pair.
-- Replaces chat_id BIGINT with channel_id TEXT.
-- Migrates existing Telegram data with "tg:" prefix for channel_id.

-- ── users: migrate from telegram_user_id to (platform, platform_user_id) ──

CREATE TABLE IF NOT EXISTS users_new (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    platform TEXT NOT NULL DEFAULT 'telegram',
    platform_user_id TEXT NOT NULL,
    username TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    last_active_at TEXT,
    UNIQUE(platform, platform_user_id)
);

INSERT INTO users_new (id, platform, platform_user_id, username, created_at, last_active_at)
SELECT id, 'telegram', CAST(telegram_user_id AS TEXT), username, created_at, last_active_at FROM users;

DROP TABLE users;
ALTER TABLE users_new RENAME TO users;

-- ── background_agents: chat_id → channel_id (prefixed with "tg:") ──

CREATE TABLE IF NOT EXISTS background_agents_new (
    id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    channel_id TEXT NOT NULL,
    watcher_type TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    config TEXT NOT NULL DEFAULT '{}',
    interval_seconds INTEGER NOT NULL DEFAULT 60,
    enabled INTEGER NOT NULL DEFAULT 1,
    last_triggered_at TEXT,
    trigger_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now'))
);

INSERT INTO background_agents_new (id, user_id, channel_id, watcher_type, name, description, config, interval_seconds, enabled, last_triggered_at, trigger_count, created_at)
SELECT id, user_id, 'tg:' || CAST(chat_id AS TEXT), watcher_type, name, description, config, interval_seconds, enabled, last_triggered_at, trigger_count, created_at FROM background_agents;

DROP TABLE background_agents;
ALTER TABLE background_agents_new RENAME TO background_agents;

CREATE INDEX IF NOT EXISTS idx_background_agents_user ON background_agents(user_id);
CREATE INDEX IF NOT EXISTS idx_wake_events_agent ON wake_events(agent_id);
