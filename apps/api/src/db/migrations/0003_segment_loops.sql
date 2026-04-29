CREATE TABLE IF NOT EXISTS segment_loops (
  id TEXT PRIMARY KEY DEFAULT gen_random_uuid(),
  student_id TEXT NOT NULL,
  piece_id TEXT NOT NULL,
  conversation_id TEXT,
  bars_start INTEGER NOT NULL,
  bars_end INTEGER NOT NULL,
  dimension TEXT,
  required_correct INTEGER NOT NULL DEFAULT 5,
  attempts_completed INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'pending',
  trigger TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX segment_loops_active_unique
  ON segment_loops (student_id, piece_id)
  WHERE status = 'active';
