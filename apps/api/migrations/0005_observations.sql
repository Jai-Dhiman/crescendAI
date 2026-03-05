-- Migration: 0005_observations
-- Add observations table for storing teacher pipeline outputs

CREATE TABLE IF NOT EXISTS observations (
    id TEXT PRIMARY KEY,
    student_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    chunk_index INTEGER,
    dimension TEXT NOT NULL,
    observation_text TEXT NOT NULL,
    elaboration_text TEXT,
    reasoning_trace TEXT,
    framing TEXT,
    dimension_score REAL,
    student_baseline REAL,
    piece_context TEXT,
    learning_arc TEXT,
    is_fallback BOOLEAN DEFAULT FALSE,
    created_at TEXT NOT NULL
);

CREATE INDEX idx_observations_student ON observations(student_id, created_at);
CREATE INDEX idx_observations_session ON observations(session_id);
