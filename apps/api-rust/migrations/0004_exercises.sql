-- Exercise catalog, dimension junction, and student tracking tables

CREATE TABLE IF NOT EXISTS exercises (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    instructions TEXT NOT NULL,
    difficulty TEXT NOT NULL,
    category TEXT NOT NULL,
    repertoire_tags TEXT,
    notation_content TEXT,
    notation_format TEXT,
    midi_content BLOB,
    source TEXT NOT NULL,
    variants_json TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_exercises_difficulty ON exercises(difficulty);

CREATE TABLE IF NOT EXISTS exercise_dimensions (
    exercise_id TEXT NOT NULL REFERENCES exercises(id),
    dimension TEXT NOT NULL,
    PRIMARY KEY (exercise_id, dimension)
);

CREATE INDEX IF NOT EXISTS idx_exercise_dimensions_dim ON exercise_dimensions(dimension);

CREATE TABLE IF NOT EXISTS student_exercises (
    id TEXT PRIMARY KEY,
    student_id TEXT NOT NULL,
    exercise_id TEXT NOT NULL,
    session_id TEXT,
    assigned_at TEXT NOT NULL,
    completed BOOLEAN DEFAULT 0,
    response TEXT,
    dimension_before_json TEXT,
    dimension_after_json TEXT,
    notes TEXT,
    times_assigned INTEGER DEFAULT 1,
    UNIQUE(student_id, exercise_id, session_id)
);

CREATE INDEX IF NOT EXISTS idx_student_exercises ON student_exercises(student_id, exercise_id);

-- Seed exercises

-- WARMUP

INSERT INTO exercises (id, title, description, instructions, difficulty, category, repertoire_tags, source, created_at) VALUES (
    'ex-wrm-001',
    'Five-Finger Warm-Up',
    'Activates each finger independently with even tone and timing. Essential start to any session.',
    'Place your right hand on C-D-E-F-G. Play each note slowly and evenly, listening for equal volume on every finger. Repeat with left hand. Then play both hands together in parallel motion, ascending and descending. Focus on relaxed wrists and consistent touch.',
    'beginner',
    'warmup',
    NULL,
    'curated',
    '2026-03-15T00:00:00Z'
);
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-wrm-001', 'articulation');
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-wrm-001', 'timing');

INSERT INTO exercises (id, title, description, instructions, difficulty, category, repertoire_tags, source, created_at) VALUES (
    'ex-wrm-002',
    'Major Scale Cycle',
    'Builds fluency across all keys with consistent fingering and tone.',
    'Play C major scale over 2 octaves, hands together, at a comfortable tempo. Move to G major, then D major, continuing through the cycle of fifths. Keep each scale even in dynamics and tempo. If a key feels unfamiliar, slow down rather than stumbling through.',
    'beginner',
    'warmup',
    NULL,
    'curated',
    '2026-03-15T00:00:00Z'
);
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-wrm-002', 'timing');
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-wrm-002', 'articulation');

INSERT INTO exercises (id, title, description, instructions, difficulty, category, repertoire_tags, source, created_at) VALUES (
    'ex-wrm-003',
    'Arpeggio Warm-Up',
    'Develops hand shape awareness and smooth wrist rotation across broken chords.',
    'Play C major arpeggio over 2 octaves with the right hand, then the left. Focus on smooth wrist rotation at each thumb crossing. Play at mf with even tone. Move through C, F, G, and D major arpeggios. Keep fingers close to the keys.',
    'beginner',
    'warmup',
    NULL,
    'curated',
    '2026-03-15T00:00:00Z'
);
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-wrm-003', 'articulation');

-- DYNAMICS

INSERT INTO exercises (id, title, description, instructions, difficulty, category, repertoire_tags, source, created_at) VALUES (
    'ex-dyn-001',
    'Dynamic Contrast Scales',
    'Trains smooth dynamic gradients from pp to ff and back, building control over the full dynamic range.',
    'Play a C major scale ascending over 2 octaves. Start at pp and arrive at ff by the top. Descend from ff back to pp. Focus on making the gradient as smooth as possible -- no sudden jumps. Repeat in the key of your current piece.',
    'intermediate',
    'technique',
    NULL,
    'curated',
    '2026-03-15T00:00:00Z'
);
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-dyn-001', 'dynamics');

INSERT INTO exercises (id, title, description, instructions, difficulty, category, repertoire_tags, source, created_at) VALUES (
    'ex-dyn-002',
    'Terraced Dynamics',
    'Develops the ability to shift cleanly between distinct dynamic levels without gradual transition.',
    'Choose a simple scale or passage you know well. Play it 4 times: first at pp, then p, then f, then ff. Each repetition should be at a consistent, clearly different volume. The goal is distinct levels, not a gradient. Record and listen back: can you hear four separate volumes?',
    'beginner',
    'technique',
    NULL,
    'curated',
    '2026-03-15T00:00:00Z'
);
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-dyn-002', 'dynamics');

INSERT INTO exercises (id, title, description, instructions, difficulty, category, repertoire_tags, source, created_at) VALUES (
    'ex-dyn-003',
    'Sforzando Control',
    'Builds control over sudden accents within a quiet context, crucial for Beethoven and similar repertoire.',
    'Play a C major scale at pp. On every 4th note, play a sforzando (sf) -- a sudden accent -- then immediately return to pp. The accent should be sharp and deliberate, not a gradual swell. Practice until the contrast between sf and pp is dramatic but controlled.',
    'intermediate',
    'technique',
    '["Beethoven"]',
    'curated',
    '2026-03-15T00:00:00Z'
);
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-dyn-003', 'dynamics');
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-dyn-003', 'articulation');

INSERT INTO exercises (id, title, description, instructions, difficulty, category, repertoire_tags, source, created_at) VALUES (
    'ex-dyn-004',
    'Chopin Cantabile Voicing',
    'Develops the ability to project a singing melody over a soft accompaniment, essential for Romantic repertoire.',
    'Choose a passage where the right hand has a melody over left hand chords or arpeggios. Play the LH alone at pp. Now add the RH melody at mf. The melody should float clearly above the accompaniment. If the LH starts competing, reduce it further. Record and listen: is the melody always audible?',
    'advanced',
    'musicality',
    '["Chopin", "romantic"]',
    'curated',
    '2026-03-15T00:00:00Z'
);
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-dyn-004', 'dynamics');
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-dyn-004', 'phrasing');

-- TIMING

INSERT INTO exercises (id, title, description, instructions, difficulty, category, repertoire_tags, source, created_at) VALUES (
    'ex-tim-001',
    'Metronome Pulse Training',
    'Builds internal pulse by gradually removing external timing support.',
    'Set a metronome to 72 bpm. Play a simple scale or passage for 8 bars with the metronome. Then mute the metronome for 4 bars, keeping the same tempo internally. Un-mute and check: are you still aligned? Repeat, gradually extending the silent bars to 8, then 16.',
    'beginner',
    'technique',
    NULL,
    'curated',
    '2026-03-15T00:00:00Z'
);
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-tim-001', 'timing');

INSERT INTO exercises (id, title, description, instructions, difficulty, category, repertoire_tags, source, created_at) VALUES (
    'ex-tim-002',
    'Mozart Evenness Drill',
    'Develops rhythmic clarity and evenness in fast passage work, essential for Classical repertoire.',
    'Choose a fast passage (Alberti bass, running 16ths, or scales). Play it at half tempo with a metronome, focusing on absolute evenness -- every note the same length and weight. Record and listen: do any notes rush or drag? Gradually increase tempo by 4 bpm increments only when the current tempo is perfectly even.',
    'intermediate',
    'technique',
    '["Mozart", "classical"]',
    'curated',
    '2026-03-15T00:00:00Z'
);
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-tim-002', 'timing');
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-tim-002', 'articulation');

INSERT INTO exercises (id, title, description, instructions, difficulty, category, repertoire_tags, source, created_at) VALUES (
    'ex-tim-003',
    'Rubato Awareness',
    'Develops intentional rubato by separating rhythmic freedom from rhythmic imprecision.',
    'Choose an expressive passage from your piece. First, play it strictly in tempo with a metronome -- no rubato at all. Now play it again without the metronome, adding rubato where you feel it should go. Record both versions. Listen: does your rubato version sound intentional, or does it just sound like uneven timing? The difference is whether the rubato serves the musical phrase.',
    'advanced',
    'musicality',
    '["Chopin", "romantic"]',
    'curated',
    '2026-03-15T00:00:00Z'
);
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-tim-003', 'timing');
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-tim-003', 'phrasing');

-- PEDALING

INSERT INTO exercises (id, title, description, instructions, difficulty, category, repertoire_tags, source, created_at) VALUES (
    'ex-ped-001',
    'Legato Pedal Harmonic Changes',
    'Develops clean pedal transitions at harmonic boundaries, preventing harmonic bleed between chords.',
    'Take a 4-bar phrase from your Chopin piece where the harmony changes every bar. Play with full sustain pedal. Now: lift the pedal exactly on each new harmony and re-engage immediately. Listen for any overlap or gap. The goal is a seamless legato with zero harmonic bleed.',
    'intermediate',
    'technique',
    '["Chopin", "romantic"]',
    'curated',
    '2026-03-15T00:00:00Z'
);
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-ped-001', 'pedaling');

INSERT INTO exercises (id, title, description, instructions, difficulty, category, repertoire_tags, source, created_at) VALUES (
    'ex-ped-002',
    'Half-Pedal Color Palette',
    'Explores the range of pedal depths for tonal color, moving beyond binary pedal on/off.',
    'Choose a sustained chord passage. Play it with full pedal, then half pedal, then quarter pedal, then no pedal. Listen to how each depth changes the resonance and color. Now play the passage with varying pedal depth -- deeper on rich harmonies, shallower on moving passages. Develop sensitivity to the pedal as a continuous control, not a switch.',
    'advanced',
    'technique',
    '["Debussy", "impressionist"]',
    'curated',
    '2026-03-15T00:00:00Z'
);
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-ped-002', 'pedaling');

INSERT INTO exercises (id, title, description, instructions, difficulty, category, repertoire_tags, source, created_at) VALUES (
    'ex-ped-003',
    'Debussy Layered Pedaling',
    'Develops the ability to create tonal layers through pedal technique, essential for Impressionist repertoire.',
    'Choose a Debussy passage with sustained bass notes and moving upper voices. Play the bass note with full pedal, then gradually thin the pedal as the upper voices move. The bass should sustain (via finger legato or partial pedal) while upper voices remain clear. Listen for muddy vs. clear textures.',
    'advanced',
    'musicality',
    '["Debussy", "impressionist"]',
    'curated',
    '2026-03-15T00:00:00Z'
);
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-ped-003', 'pedaling');
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-ped-003', 'phrasing');

-- ARTICULATION

INSERT INTO exercises (id, title, description, instructions, difficulty, category, repertoire_tags, source, created_at) VALUES (
    'ex-art-001',
    'Legato vs Staccato Contrast',
    'Develops deliberate control over touch quality by alternating between connected and detached playing.',
    'Play a C major scale ascending in legato -- each note connects smoothly to the next with no gaps. Descend in staccato -- each note short and crisp. Now alternate: legato ascending, staccato descending, 4 times. The contrast should be dramatic and consistent.',
    'beginner',
    'technique',
    NULL,
    'curated',
    '2026-03-15T00:00:00Z'
);
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-art-001', 'articulation');

INSERT INTO exercises (id, title, description, instructions, difficulty, category, repertoire_tags, source, created_at) VALUES (
    'ex-art-002',
    'Bach Voice Independence',
    'Builds the ability to maintain independent articulation across voices, essential for contrapuntal music.',
    'Choose a Bach two-part invention or a fugue exposition. Play each voice alone with its own articulation -- the subject might be legato while the counter-subject is more detached. Now combine both voices while maintaining their independent articulation. If one voice starts mimicking the other, isolate and retry.',
    'advanced',
    'musicality',
    '["Bach", "baroque"]',
    'curated',
    '2026-03-15T00:00:00Z'
);
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-art-002', 'articulation');
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-art-002', 'phrasing');

INSERT INTO exercises (id, title, description, instructions, difficulty, category, repertoire_tags, source, created_at) VALUES (
    'ex-art-003',
    'Beethoven Legato-Staccato Passages',
    'Practices the rapid articulation changes that Beethoven demands, particularly sforzando within legato lines.',
    'Find a Beethoven passage where articulation shifts quickly -- legato melody interrupted by staccato chords, or sforzando accents within a smooth line. Practice each articulation type separately at half tempo. Then combine, focusing on making each transition instant and deliberate, not gradual.',
    'intermediate',
    'technique',
    '["Beethoven", "classical"]',
    'curated',
    '2026-03-15T00:00:00Z'
);
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-art-003', 'articulation');
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-art-003', 'dynamics');

INSERT INTO exercises (id, title, description, instructions, difficulty, category, repertoire_tags, source, created_at) VALUES (
    'ex-art-004',
    'Mozart Clarity Drill',
    'Develops the crystal-clear articulation that Mozart requires, where every note is distinct and precisely placed.',
    'Choose a Mozart passage with running eighth or sixteenth notes. Play at half tempo, lifting each finger cleanly before the next note. Every note should have a clear beginning and ending. No pedal. Record and listen: can you hear every single note distinctly? Gradually increase tempo while maintaining clarity.',
    'intermediate',
    'technique',
    '["Mozart", "classical"]',
    'curated',
    '2026-03-15T00:00:00Z'
);
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-art-004', 'articulation');
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-art-004', 'timing');

INSERT INTO exercises (id, title, description, instructions, difficulty, category, repertoire_tags, source, created_at) VALUES (
    'ex-art-005',
    'Melody Extraction',
    'Develops the ability to project a melody above accompaniment through touch differentiation.',
    'Choose a passage where the RH has melody over LH chords. Play the melody alone at mf. Now add the LH at pp -- the melody should remain just as clear. If the LH starts to overpower, reduce it further. Record and listen back: can you hear every note of the melody clearly?',
    'advanced',
    'musicality',
    NULL,
    'curated',
    '2026-03-15T00:00:00Z'
);
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-art-005', 'articulation');
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-art-005', 'dynamics');

-- PHRASING

INSERT INTO exercises (id, title, description, instructions, difficulty, category, repertoire_tags, source, created_at) VALUES (
    'ex-phr-001',
    'Phrase Breathing',
    'Develops awareness of phrase structure by treating musical phrases like vocal breaths.',
    'Choose a lyrical passage. Identify the phrase boundaries -- where would a singer breathe? Mark them. Play through, making a tiny lift (not a full stop) at each breath point. The lift should be barely audible but enough to create shape. Now play without the lifts and notice how the music loses direction.',
    'intermediate',
    'musicality',
    NULL,
    'curated',
    '2026-03-15T00:00:00Z'
);
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-phr-001', 'phrasing');

INSERT INTO exercises (id, title, description, instructions, difficulty, category, repertoire_tags, source, created_at) VALUES (
    'ex-phr-002',
    'Chopin Long Line',
    'Trains the ability to shape extended melodic lines with direction and arrival points.',
    'Take an 8-bar Chopin melody. Identify the highest note or the harmonic climax -- that is your arrival point. Play the phrase building toward that point, then relaxing away from it. Every note should either be going toward or coming from the climax. If the phrase feels aimless, your arrival point is wrong or your shaping is not committed enough.',
    'intermediate',
    'musicality',
    '["Chopin", "romantic"]',
    'curated',
    '2026-03-15T00:00:00Z'
);
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-phr-002', 'phrasing');
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-phr-002', 'dynamics');

INSERT INTO exercises (id, title, description, instructions, difficulty, category, repertoire_tags, source, created_at) VALUES (
    'ex-phr-003',
    'Bach Phrase Architecture',
    'Develops awareness of phrase structure in contrapuntal music where multiple voices phrase independently.',
    'In a Bach fugue or invention, mark the subject entries in each voice. Each subject entry is a phrase with its own shape. Play each voice alone and shape each subject entry with a clear arc (beginning, peak, resolution). Now play all voices together -- can each subject entry still be heard as a shaped phrase?',
    'advanced',
    'musicality',
    '["Bach", "baroque"]',
    'curated',
    '2026-03-15T00:00:00Z'
);
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-phr-003', 'phrasing');
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-phr-003', 'articulation');

INSERT INTO exercises (id, title, description, instructions, difficulty, category, repertoire_tags, source, created_at) VALUES (
    'ex-phr-004',
    'Debussy Color Phrasing',
    'Develops phrase shaping through color and texture changes rather than just dynamic changes.',
    'Choose a Debussy passage. Instead of shaping phrases with volume alone, experiment with touch quality: deeper key contact for warm tones, lighter touch for transparent textures. Use pedal depth as part of the phrase shape. The phrase should have a clear arc even if the volume stays relatively constant.',
    'advanced',
    'musicality',
    '["Debussy", "impressionist"]',
    'curated',
    '2026-03-15T00:00:00Z'
);
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-phr-004', 'phrasing');
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-phr-004', 'pedaling');

-- INTERPRETATION

INSERT INTO exercises (id, title, description, instructions, difficulty, category, repertoire_tags, source, created_at) VALUES (
    'ex-int-001',
    'Character Study',
    'Develops interpretive imagination by playing the same music with different emotional characters.',
    'Choose a short passage (4-8 bars). Play it three times with different characters: joyful, melancholy, and agitated. Use dynamics, articulation, tempo, and pedaling to create each character. Record all three. Listen back: are the three versions genuinely different in character, or just in volume?',
    'intermediate',
    'musicality',
    NULL,
    'curated',
    '2026-03-15T00:00:00Z'
);
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-int-001', 'interpretation');

INSERT INTO exercises (id, title, description, instructions, difficulty, category, repertoire_tags, source, created_at) VALUES (
    'ex-int-002',
    'Comparative Listening',
    'Develops interpretive awareness by analyzing how professionals make different choices with the same score.',
    'Find two professional recordings of the passage you are working on (YouTube is fine). Listen to each twice, noting: where do they differ in tempo, dynamics, rubato, and pedaling? Which choices do you prefer? Why? Now play the passage yourself, consciously borrowing one choice from each performer. Your interpretation should be informed by theirs but not a copy.',
    'intermediate',
    'ear-training',
    NULL,
    'curated',
    '2026-03-15T00:00:00Z'
);
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-int-002', 'interpretation');

INSERT INTO exercises (id, title, description, instructions, difficulty, category, repertoire_tags, source, created_at) VALUES (
    'ex-int-003',
    'Structural Emphasis',
    'Builds interpretive depth by connecting musical structure to performance decisions.',
    'Analyze the harmonic structure of your passage: where are the tensions, resolutions, surprises, and cadences? For each structural event, decide how you will signal it: a slight ritardando before a resolution? A dynamic push into a modulation? Write your decisions on the score. Now play, executing each decision. The structure should be audible to a listener who does not have the score.',
    'advanced',
    'musicality',
    NULL,
    'curated',
    '2026-03-15T00:00:00Z'
);
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-int-003', 'interpretation');
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-int-003', 'phrasing');
