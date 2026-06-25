-- Command execution audit log
BEGIN TRANSACTION;

CREATE TABLE IF NOT EXISTS command_audit (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    command       TEXT    NOT NULL,
    exit_code     INTEGER,
    stdout_head   TEXT,
    stderr_head   TEXT,
    was_blocked   INTEGER NOT NULL DEFAULT 0,
    block_reason  TEXT,
    block_severity TEXT,
    confirmed_override INTEGER NOT NULL DEFAULT 0,
    workspace_path TEXT,
    executed_at   TEXT    DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_command_audit_user
    ON command_audit(user_id, executed_at);

COMMIT;
