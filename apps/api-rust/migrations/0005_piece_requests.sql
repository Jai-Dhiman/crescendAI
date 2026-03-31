-- Track piece identification requests for catalog prioritization.
-- Unmatched queries (matched_piece_id IS NULL) form a scoreboard
-- of student demand that guides which scores to source next.

CREATE TABLE IF NOT EXISTS piece_requests (
  id TEXT PRIMARY KEY,
  query TEXT NOT NULL,
  student_id TEXT NOT NULL,
  matched_piece_id TEXT,
  match_confidence REAL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_piece_requests_unmatched
  ON piece_requests(matched_piece_id) WHERE matched_piece_id IS NULL;
