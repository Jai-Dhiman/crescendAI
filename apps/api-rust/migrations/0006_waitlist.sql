-- Waitlist for beta signups
CREATE TABLE IF NOT EXISTS waitlist (
    email TEXT PRIMARY KEY,
    context TEXT,
    source TEXT NOT NULL DEFAULT 'web',
    created_at TEXT NOT NULL
);

CREATE INDEX idx_waitlist_created ON waitlist(created_at);
