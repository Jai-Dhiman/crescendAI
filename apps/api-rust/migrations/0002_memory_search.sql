-- Add entities column for entity-based retrieval and useful indexes for memory search.

ALTER TABLE synthesized_facts ADD COLUMN entities TEXT;

CREATE INDEX IF NOT EXISTS idx_sf_student_dimension ON synthesized_facts(student_id, dimension);
CREATE INDEX IF NOT EXISTS idx_sf_student_source ON synthesized_facts(student_id, source_type);
