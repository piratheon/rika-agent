-- Background agent monitoring system
BEGIN TRANSACTION;

CREATE TABLE IF NOT EXISTS background_agents (
    id                 TEXT    PRIMARY KEY,
    user_id            INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    chat_id            INTEGER NOT NULL,
    watcher_type       TEXT    NOT NULL,
    name               TEXT    NOT NULL,
    description        TEXT    NOT NULL DEFAULT '',
    config             TEXT    NOT NULL DEFAULT '{}',
    interval_seconds   INTEGER NOT NULL DEFAULT 60,
    enabled            INTEGER NOT NULL DEFAULT 1,
    last_triggered_at  TEXT,
    trigger_count      INTEGER NOT NULL DEFAULT 0,
    created_at         TEXT    DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS wake_events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id     TEXT    NOT NULL REFERENCES background_agents(id) ON DELETE CASCADE,
    user_id      INTEGER NOT NULL,
    event_type   TEXT    NOT NULL,
    severity     TEXT    NOT NULL DEFAULT 'warning',
    raw_data     TEXT    NOT NULL DEFAULT '{}',
    ai_analysis  TEXT,
    sent_to_user INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT    DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_background_agents_user
    ON background_agents(user_id);

CREATE INDEX IF NOT EXISTS idx_wake_events_agent
    ON wake_events(agent_id);

COMMIT;
