-- Insert test user
INSERT INTO users (id, email, role, full_name)
VALUES ('11111111-1111-1111-1111-111111111111'::uuid, 'teacher@test.com', 'teacher', 'Test Teacher')
ON CONFLICT (email) DO NOTHING;

-- Insert knowledge base document
INSERT INTO knowledge_base_docs (id, title, source_type, source_url, owner_id, is_public, status, total_chunks, processed_at)
VALUES (
    '22222222-2222-2222-2222-222222222222'::uuid,
    'Piano Pedagogy Fundamentals',
    'text',
    NULL,
    '11111111-1111-1111-1111-111111111111'::uuid,
    true,
    'completed',
    5,
    NOW()
)
ON CONFLICT (id) DO NOTHING;

-- Insert document chunks with content about piano pedagogy
-- Chunk 1: Posture and Hand Position
INSERT INTO document_chunks (id, doc_id, chunk_index, content, embedding, metadata)
VALUES (
    '33333333-3333-3333-3333-333333333331'::uuid,
    '22222222-2222-2222-2222-222222222222'::uuid,
    0,
    'Proper posture and hand position are fundamental to piano playing. Students should sit with their back straight, feet flat on the floor, and arms relaxed. The wrists should be level with the keyboard, not drooping or raised. Curved fingers, as if holding a ball, allow for optimal control and prevent injury. The thumb should rest on its side, not flat on the key.',
    array_fill(0.1, ARRAY[768])::vector(768),
    '{"page": 1, "topic": "posture", "is_public": true}'::jsonb
)
ON CONFLICT (doc_id, chunk_index) DO NOTHING;

-- Chunk 2: Scale Practice
INSERT INTO document_chunks (id, doc_id, chunk_index, content, embedding, metadata)
VALUES (
    '33333333-3333-3333-3333-333333333332'::uuid,
    '22222222-2222-2222-2222-222222222222'::uuid,
    1,
    'Scale practice is essential for developing finger independence and evenness. Begin with major scales using proper fingering patterns. Practice hands separately first, then together. Use a metronome starting at 60 BPM and gradually increase speed. Focus on evenness of tone and rhythm rather than raw speed. Practice scales in all keys, starting with C major and adding sharps and flats progressively.',
    array_fill(0.2, ARRAY[768])::vector(768),
    '{"page": 2, "topic": "scales", "is_public": true}'::jsonb
)
ON CONFLICT (doc_id, chunk_index) DO NOTHING;

-- Chunk 3: Sight Reading
INSERT INTO document_chunks (id, doc_id, chunk_index, content, embedding, metadata)
VALUES (
    '33333333-3333-3333-3333-333333333333'::uuid,
    '22222222-2222-2222-2222-222222222222'::uuid,
    2,
    'Sight reading skills develop through consistent daily practice. Start with simple pieces below your current level. Look ahead at least one measure while playing. Identify the key signature and time signature before beginning. Practice reading both treble and bass clef independently. Never stop to correct mistakes during sight reading - maintaining steady tempo is more important than perfect accuracy.',
    array_fill(0.3, ARRAY[768])::vector(768),
    '{"page": 3, "topic": "sight reading", "is_public": true}'::jsonb
)
ON CONFLICT (doc_id, chunk_index) DO NOTHING;

-- Chunk 4: Practice Techniques
INSERT INTO document_chunks (id, doc_id, chunk_index, content, embedding, metadata)
VALUES (
    '33333333-3333-3333-3333-333333333334'::uuid,
    '22222222-2222-2222-2222-222222222222'::uuid,
    3,
    'Effective practice requires focused attention and strategic repetition. Break difficult passages into small sections of 2-4 measures. Practice hands separately before combining. Use varied rhythms to build muscle memory. The slow practice method is crucial: play passages at half tempo with perfect accuracy before increasing speed. Set specific goals for each practice session.',
    array_fill(0.4, ARRAY[768])::vector(768),
    '{"page": 4, "topic": "practice methods", "is_public": true}'::jsonb
)
ON CONFLICT (doc_id, chunk_index) DO NOTHING;

-- Chunk 5: Rhythm and Timing
INSERT INTO document_chunks (id, doc_id, chunk_index, content, embedding, metadata)
VALUES (
    '33333333-3333-3333-3333-333333333335'::uuid,
    '22222222-2222-2222-2222-222222222222'::uuid,
    4,
    'Developing a strong sense of rhythm is fundamental to musical performance. Use a metronome regularly to build internal timing. Practice counting aloud while playing to reinforce rhythmic understanding. Subdivide complex rhythms into smaller units. Tap out rhythms away from the piano to internalize patterns. Record yourself playing to identify timing inconsistencies.',
    array_fill(0.5, ARRAY[768])::vector(768),
    '{"page": 5, "topic": "rhythm", "is_public": true}'::jsonb
)
ON CONFLICT (doc_id, chunk_index) DO NOTHING;
