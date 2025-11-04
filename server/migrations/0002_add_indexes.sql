-- CrescendAI Server - Performance Indexes
-- Version: 2.0
-- Created: 2025-01-30

-- Chat Sessions Indexes
CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id ON chat_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_recording_id ON chat_sessions(recording_id);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_created_at ON chat_sessions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_created ON chat_sessions(user_id, created_at DESC);

-- Chat Messages Indexes
CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_created_at ON chat_messages(created_at);
CREATE INDEX IF NOT EXISTS idx_chat_messages_session_created ON chat_messages(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_chat_messages_role ON chat_messages(role);

-- Recordings Indexes
CREATE INDEX IF NOT EXISTS idx_recordings_user_id ON recordings(user_id);
CREATE INDEX IF NOT EXISTS idx_recordings_status ON recordings(status);
CREATE INDEX IF NOT EXISTS idx_recordings_created_at ON recordings(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_recordings_user_created ON recordings(user_id, created_at DESC);

-- Analysis Results Indexes
CREATE INDEX IF NOT EXISTS idx_analysis_results_recording_id ON analysis_results(recording_id);
CREATE INDEX IF NOT EXISTS idx_analysis_results_created_at ON analysis_results(created_at DESC);

-- Knowledge Documents Indexes
CREATE INDEX IF NOT EXISTS idx_knowledge_documents_doc_type ON knowledge_documents(doc_type);
CREATE INDEX IF NOT EXISTS idx_knowledge_documents_created_at ON knowledge_documents(created_at DESC);

-- Knowledge Chunks Indexes
CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_document_id ON knowledge_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_vectorize_id ON knowledge_chunks(vectorize_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_doc_chunk ON knowledge_chunks(document_id, chunk_index);

-- Full-text search index for knowledge chunks content
-- D1 supports FTS5 for full-text search
CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_chunks_fts USING fts5(
    chunk_id UNINDEXED,
    content,
    tokenize = 'porter'
);

-- User Contexts Indexes
CREATE INDEX IF NOT EXISTS idx_user_contexts_user_id ON user_contexts(user_id);
CREATE INDEX IF NOT EXISTS idx_user_contexts_experience_level ON user_contexts(experience_level);

-- Feedback History Indexes
CREATE INDEX IF NOT EXISTS idx_feedback_history_recording_id ON feedback_history(recording_id);
CREATE INDEX IF NOT EXISTS idx_feedback_history_session_id ON feedback_history(session_id);
CREATE INDEX IF NOT EXISTS idx_feedback_history_created_at ON feedback_history(created_at DESC);
