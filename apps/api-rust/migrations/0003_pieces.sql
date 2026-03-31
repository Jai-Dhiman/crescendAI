-- Score MIDI Library: piece catalog table
-- Design spec: docs/superpowers/specs/2026-03-14-score-midi-library-design.md

CREATE TABLE IF NOT EXISTS pieces (
  piece_id TEXT PRIMARY KEY,
  composer TEXT NOT NULL,
  title TEXT NOT NULL,
  key_signature TEXT,
  time_signature TEXT,
  tempo_bpm INTEGER,
  bar_count INTEGER NOT NULL,
  duration_seconds REAL,
  note_count INTEGER NOT NULL,
  pitch_range_low INTEGER,
  pitch_range_high INTEGER,
  has_time_sig_changes INTEGER NOT NULL DEFAULT 0,
  has_tempo_changes INTEGER NOT NULL DEFAULT 0,
  source TEXT NOT NULL DEFAULT 'asap',
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_pieces_composer ON pieces(composer);
