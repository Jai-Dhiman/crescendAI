-- Migration: Provider-agnostic auth schema
-- Adds student_id UUID as primary key, auth_identities lookup table

-- Create new students table with UUID primary key
CREATE TABLE IF NOT EXISTS students_v2 (
    student_id TEXT PRIMARY KEY,
    email TEXT,
    display_name TEXT,
    inferred_level TEXT,
    baseline_dynamics REAL,
    baseline_timing REAL,
    baseline_pedaling REAL,
    baseline_articulation REAL,
    baseline_phrasing REAL,
    baseline_interpretation REAL,
    baseline_session_count INTEGER DEFAULT 0,
    explicit_goals TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Migrate existing data from students to students_v2
INSERT INTO students_v2 (student_id, email, inferred_level, baseline_dynamics, baseline_timing, baseline_pedaling, baseline_articulation, baseline_phrasing, baseline_interpretation, baseline_session_count, explicit_goals, created_at, updated_at)
SELECT apple_user_id, email, inferred_level, baseline_dynamics, baseline_timing, baseline_pedaling, baseline_articulation, baseline_phrasing, baseline_interpretation, baseline_session_count, explicit_goals, updated_at, updated_at
FROM students;

-- Drop old table and rename
DROP TABLE IF EXISTS students;
ALTER TABLE students_v2 RENAME TO students;

-- Auth identities lookup table
CREATE TABLE IF NOT EXISTS auth_identities (
    provider TEXT NOT NULL,
    provider_user_id TEXT NOT NULL,
    student_id TEXT NOT NULL REFERENCES students(student_id),
    created_at TEXT NOT NULL,
    PRIMARY KEY (provider, provider_user_id)
);

CREATE INDEX IF NOT EXISTS idx_auth_identities_student ON auth_identities(student_id);

-- Migrate existing Apple users into auth_identities
INSERT INTO auth_identities (provider, provider_user_id, student_id, created_at)
SELECT 'apple', student_id, student_id, updated_at FROM students;
