-- Token efficiency: pinned memories always injected (max 5),
-- non-pinned retrieved via semantic search only when relevant.
-- access_count and last_accessed enable LRU pruning of stale memories.
BEGIN TRANSACTION;

ALTER TABLE rika_memory ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0;
ALTER TABLE rika_memory ADD COLUMN access_count INTEGER NOT NULL DEFAULT 0;
ALTER TABLE rika_memory ADD COLUMN last_accessed TEXT;
ALTER TABLE rika_memory ADD COLUMN token_estimate INTEGER NOT NULL DEFAULT 0;

-- Index for fast pinned lookup
CREATE INDEX IF NOT EXISTS idx_rika_memory_pinned
    ON rika_memory(user_id, pinned, mem_type);

-- Index for access tracking
CREATE INDEX IF NOT EXISTS idx_rika_memory_accessed
    ON rika_memory(user_id, last_accessed);

COMMIT;
