-- Student memory system: synthesized facts, teaching approaches, memory meta
-- See docs/plans/2026-03-06-memory-system-design.md

CREATE TABLE synthesized_facts (
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

CREATE INDEX idx_synthesized_facts_student ON synthesized_facts(student_id);
CREATE INDEX idx_synthesized_facts_active ON synthesized_facts(student_id, invalid_at, expired_at);

CREATE TABLE teaching_approaches (
    id TEXT PRIMARY KEY,
    student_id TEXT NOT NULL,
    observation_id TEXT NOT NULL,
    dimension TEXT NOT NULL,
    framing TEXT NOT NULL,
    approach_summary TEXT NOT NULL,
    engaged INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE INDEX idx_teaching_approaches_student ON teaching_approaches(student_id);
CREATE INDEX idx_teaching_approaches_observation ON teaching_approaches(observation_id);

CREATE TABLE student_memory_meta (
    student_id TEXT PRIMARY KEY,
    last_synthesis_at TEXT,
    total_observations INTEGER DEFAULT 0,
    total_facts INTEGER DEFAULT 0
);
