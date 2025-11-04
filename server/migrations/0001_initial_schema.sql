-- CrescendAI Server - Initial Database Schema
-- Version: 2.0
-- Created: 2025-01-30

-- Chat Sessions
-- Stores metadata for each chat session, optionally linked to a recording
CREATE TABLE IF NOT EXISTS chat_sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    recording_id TEXT,
    title TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    FOREIGN KEY (recording_id) REFERENCES recordings(id) ON DELETE SET NULL
);

-- Chat Messages
-- Stores all messages in a chat session (user, assistant, tool calls)
CREATE TABLE IF NOT EXISTS chat_messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system', 'tool')),
    content TEXT NOT NULL,
    tool_calls TEXT, -- JSON array of tool calls
    tool_call_id TEXT, -- For tool response messages
    metadata TEXT, -- JSON object for additional metadata
    created_at INTEGER NOT NULL,
    FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
);

-- Recordings
-- Stores metadata for uploaded audio recordings
CREATE TABLE IF NOT EXISTS recordings (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    duration REAL, -- Duration in seconds
    mime_type TEXT NOT NULL,
    r2_key TEXT NOT NULL, -- R2 storage key
    status TEXT NOT NULL DEFAULT 'uploaded' CHECK(status IN ('uploaded', 'processing', 'analyzed', 'failed')),
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

-- Analysis Results
-- Stores AST model outputs (16D scores + temporal segments)
CREATE TABLE IF NOT EXISTS analysis_results (
    id TEXT PRIMARY KEY,
    recording_id TEXT NOT NULL UNIQUE,
    scores TEXT NOT NULL, -- JSON object with 16D scores
    temporal_segments TEXT NOT NULL, -- JSON array of temporal segments with scores
    processing_time_ms INTEGER,
    model_version TEXT,
    created_at INTEGER NOT NULL,
    FOREIGN KEY (recording_id) REFERENCES recordings(id) ON DELETE CASCADE
);

-- Knowledge Documents
-- Stores metadata for pedagogy documents (PDFs, articles, etc.)
CREATE TABLE IF NOT EXISTS knowledge_documents (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    source TEXT, -- URL or reference
    doc_type TEXT NOT NULL CHECK(doc_type IN ('pdf', 'article', 'book', 'other')),
    author TEXT,
    year INTEGER,
    file_path TEXT, -- R2 key if stored
    metadata TEXT, -- JSON object for additional metadata
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

-- Knowledge Chunks
-- Stores chunked text content with Vectorize IDs for semantic search
CREATE TABLE IF NOT EXISTS knowledge_chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    vectorize_id TEXT UNIQUE, -- ID in Vectorize index
    token_count INTEGER,
    metadata TEXT, -- JSON object for chunk-level metadata (e.g., page number)
    created_at INTEGER NOT NULL,
    FOREIGN KEY (document_id) REFERENCES knowledge_documents(id) ON DELETE CASCADE,
    UNIQUE(document_id, chunk_index)
);

-- User Contexts
-- Stores user-specific context for personalized feedback
CREATE TABLE IF NOT EXISTS user_contexts (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL UNIQUE,
    goals TEXT, -- JSON array of learning goals
    constraints TEXT, -- JSON array of constraints (e.g., time, physical limitations)
    repertoire TEXT, -- JSON array of pieces being studied
    experience_level TEXT CHECK(experience_level IN ('beginner', 'intermediate', 'advanced', 'professional')),
    preferred_feedback_style TEXT, -- JSON object with preferences
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

-- Feedback History (Optional - for tracking feedback given)
-- Stores generated feedback for recordings
CREATE TABLE IF NOT EXISTS feedback_history (
    id TEXT PRIMARY KEY,
    recording_id TEXT NOT NULL,
    session_id TEXT,
    feedback_text TEXT NOT NULL, -- JSON object with structured feedback
    citations TEXT, -- JSON array of knowledge chunk references
    created_at INTEGER NOT NULL,
    FOREIGN KEY (recording_id) REFERENCES recordings(id) ON DELETE CASCADE,
    FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE SET NULL
);
