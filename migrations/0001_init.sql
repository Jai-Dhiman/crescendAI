-- migrations/0001_init.sql
-- D1 schema for shareable sessions
-- Applies to both preview and production when executed with wrangler

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sessions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  slug TEXT NOT NULL UNIQUE,
  prompt TEXT NOT NULL,
  feedback TEXT NOT NULL,
  teacher_comment TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT
);

-- Helpful index for lookups by slug (unique already creates an index, but explicit naming can help tooling)
CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_slug ON sessions (slug);
