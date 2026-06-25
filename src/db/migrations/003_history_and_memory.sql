-- Chat history storage
CREATE TABLE IF NOT EXISTS chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role TEXT NOT NULL, -- 'user' or 'assistant'
    content TEXT NOT NULL,
    timestamp TEXT DEFAULT (datetime('now'))
);

-- Context summaries to handle long histories
CREATE TABLE IF NOT EXISTS chat_summaries (
    user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    summary TEXT NOT NULL,
    last_msg_id INTEGER NOT NULL, -- last message ID included in this summary
    updated_at TEXT DEFAULT (datetime('now'))
);

-- Rika's persistent memory and skills
CREATE TABLE IF NOT EXISTS rika_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    mem_key TEXT NOT NULL,
    mem_value TEXT NOT NULL,
    mem_type TEXT NOT NULL DEFAULT 'memory', -- 'memory' or 'skill'
    created_at TEXT DEFAULT (datetime('now')),
    UNIQUE(user_id, mem_key, mem_type)
);
