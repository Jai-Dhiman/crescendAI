-- ============================================================================
-- EXTENSIONS
-- ============================================================================

-- Enable pgvector for vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- ENUMS
-- ============================================================================

-- User roles
CREATE TYPE user_role AS ENUM ('teacher', 'student', 'admin');

-- Project access levels
CREATE TYPE access_level AS ENUM ('view', 'edit', 'admin');

-- Annotation types
CREATE TYPE annotation_type AS ENUM ('highlight', 'note', 'drawing');

-- Knowledge base processing status
CREATE TYPE processing_status AS ENUM ('pending', 'processing', 'completed', 'failed');

-- ============================================================================
-- TABLES
-- ============================================================================

-- Users table (extends Supabase auth.users)
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email TEXT UNIQUE NOT NULL,
    role user_role NOT NULL DEFAULT 'student',
    full_name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Indexes
    CONSTRAINT users_email_check CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$')
);

-- Teacher-Student relationships (many-to-many)
CREATE TABLE teacher_student_relationships (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    teacher_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    student_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT different_users CHECK (teacher_id != student_id),
    CONSTRAINT unique_relationship UNIQUE (teacher_id, student_id)
);

-- PDF Projects
CREATE TABLE projects (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    owner_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    description TEXT,

    -- R2 storage paths
    r2_bucket TEXT NOT NULL DEFAULT 'piano-pdfs',
    r2_key TEXT NOT NULL,  -- e.g., 'projects/{project_id}/{filename}.pdf'

    -- Metadata
    file_size_bytes BIGINT,
    page_count INTEGER,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT r2_key_unique UNIQUE (r2_bucket, r2_key)
);

-- Project access control
CREATE TABLE project_access (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    access_level access_level NOT NULL DEFAULT 'view',
    granted_by UUID NOT NULL REFERENCES users(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT unique_project_access UNIQUE (project_id, user_id)
);

-- Annotations on PDFs
CREATE TABLE annotations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL,
    annotation_type annotation_type NOT NULL,

    -- JSONB payload for flexibility
    -- For highlights: {x, y, width, height, color, text}
    -- For notes: {x, y, text, color}
    -- For drawings: {paths: [{x, y}, ...], color, strokeWidth}
    content JSONB NOT NULL,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT page_number_positive CHECK (page_number > 0)
);

-- Knowledge Base Documents
CREATE TABLE knowledge_base_docs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title TEXT NOT NULL,
    source_type TEXT NOT NULL,  -- 'pdf', 'video', 'text', 'web'
    source_url TEXT,             -- YouTube URL, web URL, or R2 key

    -- Ownership
    owner_id UUID REFERENCES users(id) ON DELETE CASCADE,  -- NULL = public/base content
    is_public BOOLEAN NOT NULL DEFAULT false,

    -- Processing status
    status processing_status NOT NULL DEFAULT 'pending',
    error_message TEXT,

    -- Metadata
    total_chunks INTEGER DEFAULT 0,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    processed_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Document chunks (for RAG)
CREATE TABLE document_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    doc_id UUID NOT NULL REFERENCES knowledge_base_docs(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,

    -- Vector embedding (BGE-base-v1.5 = 768 dimensions)
    embedding vector(768),

    -- Metadata (JSONB for flexibility)
    -- {page: 1, start_char: 0, end_char: 512, teacher_id: uuid, is_public: true}
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT chunk_index_positive CHECK (chunk_index >= 0),
    CONSTRAINT unique_chunk UNIQUE (doc_id, chunk_index)
);

-- Chat sessions (RAG conversations)
CREATE TABLE chat_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,  -- NULL = general chat
    title TEXT,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Chat messages
CREATE TABLE chat_messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,

    -- Source citations (for assistant messages)
    sources JSONB,  -- [{chunk_id, doc_id, score, snippet}, ...]

    -- Confidence score (for assistant messages)
    confidence DECIMAL(3, 2) CHECK (confidence >= 0 AND confidence <= 1),

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================================
-- INDEXES
-- ============================================================================

-- Users
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_role ON users(role);

-- Relationships
CREATE INDEX idx_relationships_teacher ON teacher_student_relationships(teacher_id);
CREATE INDEX idx_relationships_student ON teacher_student_relationships(student_id);

-- Projects
CREATE INDEX idx_projects_owner ON projects(owner_id);
CREATE INDEX idx_projects_created ON projects(created_at DESC);

-- Project access
CREATE INDEX idx_project_access_project ON project_access(project_id);
CREATE INDEX idx_project_access_user ON project_access(user_id);

-- Annotations
CREATE INDEX idx_annotations_project ON annotations(project_id);
CREATE INDEX idx_annotations_user ON annotations(user_id);
CREATE INDEX idx_annotations_project_page ON annotations(project_id, page_number);

-- Knowledge base
CREATE INDEX idx_kb_docs_owner ON knowledge_base_docs(owner_id);
CREATE INDEX idx_kb_docs_status ON knowledge_base_docs(status);
CREATE INDEX idx_kb_docs_public ON knowledge_base_docs(is_public) WHERE is_public = true;

-- Document chunks
CREATE INDEX idx_chunks_doc ON document_chunks(doc_id);
CREATE INDEX idx_chunks_metadata_teacher ON document_chunks USING gin ((metadata -> 'teacher_id'));
CREATE INDEX idx_chunks_metadata_public ON document_chunks USING gin ((metadata -> 'is_public'));

-- BM25 full-text search index on chunk content
CREATE INDEX idx_chunks_content_fts ON document_chunks USING gin(to_tsvector('english', content));

-- HNSW vector similarity index (99% recall, <8ms search)
-- m=16: max connections per node (higher = better recall)
-- ef_construction=64: build-time search depth (higher = better quality)
CREATE INDEX idx_chunks_embedding ON document_chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Chat sessions
CREATE INDEX idx_chat_sessions_user ON chat_sessions(user_id);
CREATE INDEX idx_chat_sessions_project ON chat_sessions(project_id);
CREATE INDEX idx_chat_sessions_created ON chat_sessions(created_at DESC);

-- Chat messages
CREATE INDEX idx_chat_messages_session ON chat_messages(session_id);
CREATE INDEX idx_chat_messages_created ON chat_messages(created_at);

-- ============================================================================
-- FUNCTIONS & TRIGGERS
-- ============================================================================

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply to tables with updated_at
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_projects_updated_at BEFORE UPDATE ON projects
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_annotations_updated_at BEFORE UPDATE ON annotations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_kb_docs_updated_at BEFORE UPDATE ON knowledge_base_docs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_chat_sessions_updated_at BEFORE UPDATE ON chat_sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- PGVECTOR OPTIMIZATION
-- ============================================================================

-- Set query-time HNSW search depth (balance speed/recall)
-- Can be adjusted per query: SET LOCAL hnsw.ef_search = 40;
ALTER DATABASE postgres SET hnsw.ef_search = 40;

-- ============================================================================
-- ROW LEVEL SECURITY (RLS)
-- ============================================================================

-- Enable RLS on all tables (we'll add policies later)
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE teacher_student_relationships ENABLE ROW LEVEL SECURITY;
ALTER TABLE projects ENABLE ROW LEVEL SECURITY;
ALTER TABLE project_access ENABLE ROW LEVEL SECURITY;
ALTER TABLE annotations ENABLE ROW LEVEL SECURITY;
ALTER TABLE knowledge_base_docs ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_messages ENABLE ROW LEVEL SECURITY;

-- For development: Allow service role to bypass RLS
-- In production, we'll add proper policies

-- ============================================================================
-- SEED DATA (Optional - for testing)
-- ============================================================================

-- Insert a test admin user (password will be set via Supabase Auth)
-- Uncomment if you want test data:
-- INSERT INTO users (email, role, full_name) VALUES
--     ('admin@piano.dev', 'admin', 'Test Admin'),
--     ('teacher@piano.dev', 'teacher', 'Test Teacher'),
--     ('student@piano.dev', 'student', 'Test Student');

-- ============================================================================
-- SUCCESS
-- ============================================================================

SELECT
    'Database schema created successfully!' as status,
    COUNT(*) as total_tables
FROM information_schema.tables
WHERE table_schema = 'public';
