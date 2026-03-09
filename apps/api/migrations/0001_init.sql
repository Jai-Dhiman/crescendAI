-- CrescendAI D1 Schema (consolidated)
-- Students, auth, sessions, observations, conversations, memory

-- Students with UUID primary key
CREATE TABLE IF NOT EXISTS students (
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

-- Provider-agnostic auth lookup
CREATE TABLE IF NOT EXISTS auth_identities (
    provider TEXT NOT NULL,
    provider_user_id TEXT NOT NULL,
    student_id TEXT NOT NULL REFERENCES students(student_id),
    created_at TEXT NOT NULL,
    PRIMARY KEY (provider, provider_user_id)
);

CREATE INDEX IF NOT EXISTS idx_auth_identities_student ON auth_identities(student_id);

-- Practice sessions
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    student_id TEXT NOT NULL REFERENCES students(student_id),
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

CREATE INDEX IF NOT EXISTS idx_sessions_student ON sessions(student_id, started_at);

-- Check-in Q&A
CREATE TABLE IF NOT EXISTS student_check_ins (
    id TEXT PRIMARY KEY,
    student_id TEXT NOT NULL REFERENCES students(student_id),
    session_id TEXT REFERENCES sessions(id),
    question TEXT NOT NULL,
    answer TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_checkins_student ON student_check_ins(student_id);

-- Teacher pipeline observations
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

CREATE INDEX IF NOT EXISTS idx_observations_student ON observations(student_id, created_at);
CREATE INDEX IF NOT EXISTS idx_observations_session ON observations(session_id);

-- Chat conversations
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    student_id TEXT NOT NULL,
    title TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_conversations_student ON conversations(student_id, updated_at);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, created_at);

-- Memory system
CREATE TABLE IF NOT EXISTS synthesized_facts (
    id TEXT PRIMARY KEY,
    student_id TEXT NOT NULL,
    fact_text TEXT NOT NULL,
    fact_type TEXT NOT NULL,
    dimension TEXT,
    piece_context TEXT,
    valid_at TEXT NOT NULL,
    invalid_at TEXT,
    trend TEXT,
    confidence TEXT NOT NULL,
    evidence TEXT NOT NULL,
    source_type TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expired_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_synthesized_facts_student ON synthesized_facts(student_id);
CREATE INDEX IF NOT EXISTS idx_synthesized_facts_active ON synthesized_facts(student_id, invalid_at, expired_at);

CREATE TABLE IF NOT EXISTS teaching_approaches (
    id TEXT PRIMARY KEY,
    student_id TEXT NOT NULL,
    observation_id TEXT NOT NULL,
    dimension TEXT NOT NULL,
    framing TEXT NOT NULL,
    approach_summary TEXT NOT NULL,
    engaged INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_teaching_approaches_student ON teaching_approaches(student_id);
CREATE INDEX IF NOT EXISTS idx_teaching_approaches_observation ON teaching_approaches(observation_id);

CREATE TABLE IF NOT EXISTS student_memory_meta (
    student_id TEXT PRIMARY KEY,
    last_synthesis_at TEXT,
    total_observations INTEGER DEFAULT 0,
    total_facts INTEGER DEFAULT 0
);
