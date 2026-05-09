CREATE TABLE IF NOT EXISTS config_version (
    key TEXT NOT NULL,
    version INTEGER NOT NULL,
    value TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (key, version)
);

CREATE INDEX IF NOT EXISTS idx_config_key_version ON config_version(key, version DESC);
