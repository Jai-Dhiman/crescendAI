-- Pedagogy Chunks Schema for RAG System
-- Stores pedagogical content with full citation metadata

-- Main table for pedagogy chunks
CREATE TABLE IF NOT EXISTS pedagogy_chunks (
    chunk_id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    text_with_context TEXT NOT NULL,
    source_type TEXT NOT NULL CHECK (source_type IN ('book', 'letter', 'masterclass', 'journal')),
    source_title TEXT NOT NULL,
    source_author TEXT NOT NULL,
    source_url TEXT,
    page_number INTEGER,
    section_title TEXT,
    paragraph_index INTEGER,
    char_start INTEGER,
    char_end INTEGER,
    timestamp_start REAL,
    timestamp_end REAL,
    speaker TEXT,
    composers TEXT,  -- JSON array stored as text
    pieces TEXT,     -- JSON array stored as text
    techniques TEXT, -- JSON array stored as text
    ingested_at TEXT NOT NULL DEFAULT (datetime('now')),
    source_hash TEXT NOT NULL UNIQUE
);

-- FTS5 virtual table for BM25 full-text search
CREATE VIRTUAL TABLE IF NOT EXISTS pedagogy_chunks_fts USING fts5(
    text,
    source_title,
    source_author,
    composers,
    pieces,
    techniques,
    content='pedagogy_chunks',
    content_rowid='rowid'
);

-- Triggers to keep FTS in sync with main table

-- After INSERT: add to FTS index
CREATE TRIGGER IF NOT EXISTS pedagogy_chunks_ai AFTER INSERT ON pedagogy_chunks BEGIN
    INSERT INTO pedagogy_chunks_fts(rowid, text, source_title, source_author, composers, pieces, techniques)
    VALUES (NEW.rowid, NEW.text, NEW.source_title, NEW.source_author, NEW.composers, NEW.pieces, NEW.techniques);
END;

-- After DELETE: remove from FTS index
CREATE TRIGGER IF NOT EXISTS pedagogy_chunks_ad AFTER DELETE ON pedagogy_chunks BEGIN
    INSERT INTO pedagogy_chunks_fts(pedagogy_chunks_fts, rowid, text, source_title, source_author, composers, pieces, techniques)
    VALUES ('delete', OLD.rowid, OLD.text, OLD.source_title, OLD.source_author, OLD.composers, OLD.pieces, OLD.techniques);
END;

-- After UPDATE: update FTS index
CREATE TRIGGER IF NOT EXISTS pedagogy_chunks_au AFTER UPDATE ON pedagogy_chunks BEGIN
    INSERT INTO pedagogy_chunks_fts(pedagogy_chunks_fts, rowid, text, source_title, source_author, composers, pieces, techniques)
    VALUES ('delete', OLD.rowid, OLD.text, OLD.source_title, OLD.source_author, OLD.composers, OLD.pieces, OLD.techniques);
    INSERT INTO pedagogy_chunks_fts(rowid, text, source_title, source_author, composers, pieces, techniques)
    VALUES (NEW.rowid, NEW.text, NEW.source_title, NEW.source_author, NEW.composers, NEW.pieces, NEW.techniques);
END;

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_chunks_source_type ON pedagogy_chunks(source_type);
CREATE INDEX IF NOT EXISTS idx_chunks_source_hash ON pedagogy_chunks(source_hash);
CREATE INDEX IF NOT EXISTS idx_chunks_ingested_at ON pedagogy_chunks(ingested_at);
