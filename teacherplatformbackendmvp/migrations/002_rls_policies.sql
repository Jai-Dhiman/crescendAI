-- ============================================================================
-- ROW LEVEL SECURITY (RLS) POLICIES
-- ============================================================================

-- Note: Tables were already enabled for RLS in 001_initial_schema.sql
-- This migration adds the actual policy rules

-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Function to get current user's ID from JWT
-- Note: In practice, this is handled by application-level filtering
-- These policies are for defense-in-depth when using service role key
CREATE OR REPLACE FUNCTION auth.user_id() RETURNS UUID AS $$
BEGIN
    -- This would be set by Supabase Auth or application context
    -- For now, we rely on application-level enforcement
    RETURN NULL::UUID;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to check if user is a teacher of a student
CREATE OR REPLACE FUNCTION is_teacher_of_student(teacher_id UUID, student_id UUID)
RETURNS BOOLEAN AS $$
BEGIN
    RETURN EXISTS (
        SELECT 1 FROM teacher_student_relationships
        WHERE teacher_student_relationships.teacher_id = $1
          AND teacher_student_relationships.student_id = $2
    );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ============================================================================
-- USERS TABLE POLICIES
-- ============================================================================

-- Users can read their own record
CREATE POLICY users_select_own
ON users FOR SELECT
USING (id = auth.user_id());

-- Admins can read all users
CREATE POLICY users_select_admin
ON users FOR SELECT
USING (
    EXISTS (
        SELECT 1 FROM users u
        WHERE u.id = auth.user_id()
        AND u.role = 'admin'
    )
);

-- Teachers can see their students
CREATE POLICY users_select_teacher_students
ON users FOR SELECT
USING (
    role = 'student'
    AND EXISTS (
        SELECT 1 FROM teacher_student_relationships r
        WHERE r.student_id = users.id
        AND r.teacher_id = auth.user_id()
    )
);

-- Students can see their teachers
CREATE POLICY users_select_student_teachers
ON users FOR SELECT
USING (
    role = 'teacher'
    AND EXISTS (
        SELECT 1 FROM teacher_student_relationships r
        WHERE r.teacher_id = users.id
        AND r.student_id = auth.user_id()
    )
);

-- ============================================================================
-- TEACHER_STUDENT_RELATIONSHIPS POLICIES
-- ============================================================================

-- Teachers can view their own student relationships
CREATE POLICY relationships_select_teacher
ON teacher_student_relationships FOR SELECT
USING (teacher_id = auth.user_id());

-- Students can view their own teacher relationships
CREATE POLICY relationships_select_student
ON teacher_student_relationships FOR SELECT
USING (student_id = auth.user_id());

-- Admins can view all relationships
CREATE POLICY relationships_select_admin
ON teacher_student_relationships FOR SELECT
USING (
    EXISTS (
        SELECT 1 FROM users
        WHERE id = auth.user_id()
        AND role = 'admin'
    )
);

-- Only teachers can create relationships with their students
CREATE POLICY relationships_insert_teacher
ON teacher_student_relationships FOR INSERT
WITH CHECK (
    teacher_id = auth.user_id()
    AND EXISTS (
        SELECT 1 FROM users
        WHERE id = auth.user_id()
        AND role = 'teacher'
    )
);

-- Only teachers can delete their own student relationships
CREATE POLICY relationships_delete_teacher
ON teacher_student_relationships FOR DELETE
USING (teacher_id = auth.user_id());

-- Admins can delete any relationship
CREATE POLICY relationships_delete_admin
ON teacher_student_relationships FOR DELETE
USING (
    EXISTS (
        SELECT 1 FROM users
        WHERE id = auth.user_id()
        AND role = 'admin'
    )
);

-- ============================================================================
-- PROJECTS POLICIES
-- ============================================================================

-- Users can view their own projects
CREATE POLICY projects_select_own
ON projects FOR SELECT
USING (owner_id = auth.user_id());

-- Users can view projects they have access to
CREATE POLICY projects_select_with_access
ON projects FOR SELECT
USING (
    EXISTS (
        SELECT 1 FROM project_access
        WHERE project_id = projects.id
        AND user_id = auth.user_id()
    )
);

-- Users can create their own projects
CREATE POLICY projects_insert_own
ON projects FOR INSERT
WITH CHECK (owner_id = auth.user_id());

-- Users can update their own projects
CREATE POLICY projects_update_own
ON projects FOR UPDATE
USING (owner_id = auth.user_id());

-- Users can delete their own projects
CREATE POLICY projects_delete_own
ON projects FOR DELETE
USING (owner_id = auth.user_id());

-- ============================================================================
-- PROJECT_ACCESS POLICIES
-- ============================================================================

-- Project owners can manage access
CREATE POLICY project_access_select_owner
ON project_access FOR SELECT
USING (
    EXISTS (
        SELECT 1 FROM projects
        WHERE id = project_access.project_id
        AND owner_id = auth.user_id()
    )
);

-- Users can see their own access grants
CREATE POLICY project_access_select_own
ON project_access FOR SELECT
USING (user_id = auth.user_id());

-- Project owners can grant access
CREATE POLICY project_access_insert_owner
ON project_access FOR INSERT
WITH CHECK (
    EXISTS (
        SELECT 1 FROM projects
        WHERE id = project_access.project_id
        AND owner_id = auth.user_id()
    )
);

-- Project owners can revoke access
CREATE POLICY project_access_delete_owner
ON project_access FOR DELETE
USING (
    EXISTS (
        SELECT 1 FROM projects
        WHERE id = project_access.project_id
        AND owner_id = auth.user_id()
    )
);

-- ============================================================================
-- ANNOTATIONS POLICIES
-- ============================================================================

-- Users can view annotations on projects they have access to
CREATE POLICY annotations_select
ON annotations FOR SELECT
USING (
    EXISTS (
        SELECT 1 FROM projects
        WHERE id = annotations.project_id
        AND (
            owner_id = auth.user_id()
            OR EXISTS (
                SELECT 1 FROM project_access
                WHERE project_id = projects.id
                AND user_id = auth.user_id()
            )
        )
    )
);

-- Users can create annotations on projects they have access to
CREATE POLICY annotations_insert
ON annotations FOR INSERT
WITH CHECK (
    user_id = auth.user_id()
    AND EXISTS (
        SELECT 1 FROM projects
        WHERE id = annotations.project_id
        AND (
            owner_id = auth.user_id()
            OR EXISTS (
                SELECT 1 FROM project_access
                WHERE project_id = projects.id
                AND user_id = auth.user_id()
                AND access_level IN ('edit', 'admin')
            )
        )
    )
);

-- Users can update their own annotations
CREATE POLICY annotations_update_own
ON annotations FOR UPDATE
USING (user_id = auth.user_id());

-- Users can delete their own annotations
CREATE POLICY annotations_delete_own
ON annotations FOR DELETE
USING (user_id = auth.user_id());

-- ============================================================================
-- KNOWLEDGE_BASE_DOCS POLICIES
-- ============================================================================

-- Everyone can view public knowledge base docs
CREATE POLICY kb_docs_select_public
ON knowledge_base_docs FOR SELECT
USING (is_public = true);

-- Users can view their own docs
CREATE POLICY kb_docs_select_own
ON knowledge_base_docs FOR SELECT
USING (owner_id = auth.user_id());

-- Students can view their teachers' docs
CREATE POLICY kb_docs_select_teacher_content
ON knowledge_base_docs FOR SELECT
USING (
    EXISTS (
        SELECT 1 FROM users current_user
        WHERE current_user.id = auth.user_id()
        AND current_user.role = 'student'
        AND is_teacher_of_student(knowledge_base_docs.owner_id, current_user.id)
    )
);

-- Admins can view all docs
CREATE POLICY kb_docs_select_admin
ON knowledge_base_docs FOR SELECT
USING (
    EXISTS (
        SELECT 1 FROM users
        WHERE id = auth.user_id()
        AND role = 'admin'
    )
);

-- Users can create their own docs
CREATE POLICY kb_docs_insert_own
ON knowledge_base_docs FOR INSERT
WITH CHECK (owner_id = auth.user_id());

-- Users can update their own docs
CREATE POLICY kb_docs_update_own
ON knowledge_base_docs FOR UPDATE
USING (owner_id = auth.user_id());

-- Users can delete their own docs
CREATE POLICY kb_docs_delete_own
ON knowledge_base_docs FOR DELETE
USING (owner_id = auth.user_id());

-- ============================================================================
-- DOCUMENT_CHUNKS POLICIES
-- ============================================================================

-- Public chunks are visible to everyone
CREATE POLICY chunks_select_public
ON document_chunks FOR SELECT
USING (
    EXISTS (
        SELECT 1 FROM knowledge_base_docs
        WHERE id = document_chunks.doc_id
        AND is_public = true
    )
);

-- Users can view chunks from their own docs
CREATE POLICY chunks_select_own
ON document_chunks FOR SELECT
USING (
    EXISTS (
        SELECT 1 FROM knowledge_base_docs
        WHERE id = document_chunks.doc_id
        AND owner_id = auth.user_id()
    )
);

-- Students can view chunks from their teachers' docs
CREATE POLICY chunks_select_teacher_content
ON document_chunks FOR SELECT
USING (
    EXISTS (
        SELECT 1 FROM knowledge_base_docs kb
        JOIN users current_user ON current_user.id = auth.user_id()
        WHERE kb.id = document_chunks.doc_id
        AND current_user.role = 'student'
        AND is_teacher_of_student(kb.owner_id, current_user.id)
    )
);

-- Admins can view all chunks
CREATE POLICY chunks_select_admin
ON document_chunks FOR SELECT
USING (
    EXISTS (
        SELECT 1 FROM users
        WHERE id = auth.user_id()
        AND role = 'admin'
    )
);

-- Only document owners can insert chunks
CREATE POLICY chunks_insert_own
ON document_chunks FOR INSERT
WITH CHECK (
    EXISTS (
        SELECT 1 FROM knowledge_base_docs
        WHERE id = document_chunks.doc_id
        AND owner_id = auth.user_id()
    )
);

-- Only document owners can delete chunks
CREATE POLICY chunks_delete_own
ON document_chunks FOR DELETE
USING (
    EXISTS (
        SELECT 1 FROM knowledge_base_docs
        WHERE id = document_chunks.doc_id
        AND owner_id = auth.user_id()
    )
);

-- ============================================================================
-- CHAT_SESSIONS POLICIES
-- ============================================================================

-- Users can view their own chat sessions
CREATE POLICY chat_sessions_select_own
ON chat_sessions FOR SELECT
USING (user_id = auth.user_id());

-- Users can create their own chat sessions
CREATE POLICY chat_sessions_insert_own
ON chat_sessions FOR INSERT
WITH CHECK (user_id = auth.user_id());

-- Users can update their own chat sessions
CREATE POLICY chat_sessions_update_own
ON chat_sessions FOR UPDATE
USING (user_id = auth.user_id());

-- Users can delete their own chat sessions
CREATE POLICY chat_sessions_delete_own
ON chat_sessions FOR DELETE
USING (user_id = auth.user_id());

-- ============================================================================
-- CHAT_MESSAGES POLICIES
-- ============================================================================

-- Users can view messages from their own sessions
CREATE POLICY chat_messages_select_own
ON chat_messages FOR SELECT
USING (
    EXISTS (
        SELECT 1 FROM chat_sessions
        WHERE id = chat_messages.session_id
        AND user_id = auth.user_id()
    )
);

-- Users can create messages in their own sessions
CREATE POLICY chat_messages_insert_own
ON chat_messages FOR INSERT
WITH CHECK (
    EXISTS (
        SELECT 1 FROM chat_sessions
        WHERE id = chat_messages.session_id
        AND user_id = auth.user_id()
    )
);

-- ============================================================================
-- SUCCESS
-- ============================================================================

SELECT
    'RLS policies created successfully!' as status,
    COUNT(*) as total_policies
FROM pg_policies
WHERE schemaname = 'public';
