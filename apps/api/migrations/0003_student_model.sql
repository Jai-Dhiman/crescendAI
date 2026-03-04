-- Slice 5: Student Model + Auth
-- D1 schema for student data, practice sessions, and check-ins

CREATE TABLE IF NOT EXISTS students (
    apple_user_id TEXT PRIMARY KEY,
    email TEXT,
    inferred_level TEXT,
    baseline_dynamics REAL,
    baseline_timing REAL,
    baseline_pedaling REAL,
    baseline_articulation REAL,
    baseline_phrasing REAL,
    baseline_interpretation REAL,
    baseline_session_count INTEGER DEFAULT 0,
    explicit_goals TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    student_id TEXT NOT NULL REFERENCES students(apple_user_id),
    started_at TEXT NOT NULL,
    ended_at TEXT,
    avg_dynamics REAL,
    avg_timing REAL,
    avg_pedaling REAL,
    avg_articulation REAL,
    avg_phrasing REAL,
    avg_interpretation REAL,
    observations_json TEXT,
    chunks_summary_json TEXT
);

CREATE TABLE IF NOT EXISTS student_check_ins (
    id TEXT PRIMARY KEY,
    student_id TEXT NOT NULL REFERENCES students(apple_user_id),
    session_id TEXT REFERENCES sessions(id),
    question TEXT NOT NULL,
    answer TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_student ON sessions(student_id, started_at);
CREATE INDEX IF NOT EXISTS idx_checkins_student ON student_check_ins(student_id);
