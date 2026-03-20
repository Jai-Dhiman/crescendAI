-- Unified chat: link practice sessions to conversations, extend messages for observations

-- Link practice sessions to conversations
ALTER TABLE sessions ADD COLUMN conversation_id TEXT;
CREATE INDEX IF NOT EXISTS idx_sessions_conversation ON sessions(conversation_id);

-- Extend messages to carry observation metadata
ALTER TABLE messages ADD COLUMN message_type TEXT DEFAULT 'chat';
ALTER TABLE messages ADD COLUMN dimension TEXT;
ALTER TABLE messages ADD COLUMN framing TEXT;
ALTER TABLE messages ADD COLUMN components_json TEXT;
ALTER TABLE messages ADD COLUMN session_id TEXT;
ALTER TABLE messages ADD COLUMN observation_id TEXT;

-- Back-link observations to messages and conversations
ALTER TABLE observations ADD COLUMN message_id TEXT;
ALTER TABLE observations ADD COLUMN conversation_id TEXT;
