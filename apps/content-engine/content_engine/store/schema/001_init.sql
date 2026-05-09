CREATE TABLE IF NOT EXISTS episode (
    id TEXT PRIMARY KEY,
    candidate_url TEXT NOT NULL,
    source_type TEXT NOT NULL,
    state TEXT NOT NULL,
    config_versions TEXT NOT NULL,
    model_output TEXT,
    observation TEXT,
    script_text TEXT,
    voiceover_path TEXT,
    render_path TEXT,
    posts TEXT,
    analytics TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_episode_state ON episode(state);
