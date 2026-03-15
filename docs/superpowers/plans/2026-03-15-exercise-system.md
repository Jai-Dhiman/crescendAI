# Exercise System Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the exercise database, API endpoints, seed data, and web UI integration for the CrescendAI feedback loop: observe weakness -> prescribe exercise -> evaluate improvement.

**Architecture:** D1 migration creates 3 tables (exercises, exercise_dimensions, student_exercises) with 25 curated seed exercises. Rust service exposes 3 authenticated endpoints (GET catalog, POST assign, POST complete). Web UI enhances the existing ExerciseSetCard with assign capability and adds a "Try exercises" action to chat observations.

**Tech Stack:** Rust/WASM (Cloudflare Workers), D1 (SQLite), React 19, Tailwind CSS v4, bun, wrangler/Miniflare

**Spec:** `docs/superpowers/specs/2026-03-15-exercise-system-design.md`

**Scope Boundaries:**
- Working in `apps/api/` and `apps/web/src/`
- NOT touching `model/`, `usePracticeSession.ts`, `ListeningMode.tsx`, `stop.rs`, `teaching_moments.rs`
- Migration is 0004 (0003 exists for Score MIDI Library)

---

## Chunk 1: D1 Migration + Seed Data

### Task 1: Create D1 migration with tables and indexes

**Files:**
- Create: `apps/api/migrations/0004_exercises.sql`

- [ ] **Step 1: Write the migration DDL**

Create `apps/api/migrations/0004_exercises.sql` with the three tables and indexes:

```sql
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
```

- [ ] **Step 2: Verify migration file exists and SQL is valid**

Run: `cat apps/api/migrations/0004_exercises.sql | head -5`
Expected: First lines of the migration file

- [ ] **Step 3: Commit the DDL**

```bash
git add apps/api/migrations/0004_exercises.sql
git commit -m "add exercise system D1 tables (0004)"
```

### Task 2: Add seed data -- warmup exercises (3)

**Files:**
- Modify: `apps/api/migrations/0004_exercises.sql`

- [ ] **Step 1: Append warmup exercise INSERTs**

Append to `0004_exercises.sql`:

```sql
-- ============================================================
-- Seed data: 25 curated exercises
-- ============================================================

-- WARMUP exercises (3)

INSERT INTO exercises (id, title, description, instructions, difficulty, category, repertoire_tags, source, created_at)
VALUES ('ex-wrm-001', 'Five-Finger Warm-Up', 'Activates each finger independently with even tone and timing. Essential start to any session.', 'Place your right hand on C-D-E-F-G. Play each note slowly and evenly, listening for equal volume on every finger. Repeat with left hand. Then play both hands together in parallel motion, ascending and descending. Focus on relaxed wrists and consistent touch.', 'beginner', 'warmup', NULL, 'curated', '2026-03-15T00:00:00Z');
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-wrm-001', 'articulation');
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-wrm-001', 'timing');

INSERT INTO exercises (id, title, description, instructions, difficulty, category, repertoire_tags, source, created_at)
VALUES ('ex-wrm-002', 'Major Scale Cycle', 'Builds fluency across all keys with consistent fingering and tone.', 'Play C major scale over 2 octaves, hands together, at a comfortable tempo. Move to G major, then D major, continuing through the cycle of fifths. Keep each scale even in dynamics and tempo. If a key feels unfamiliar, slow down rather than stumbling through.', 'beginner', 'warmup', NULL, 'curated', '2026-03-15T00:00:00Z');
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-wrm-002', 'timing');
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-wrm-002', 'articulation');

INSERT INTO exercises (id, title, description, instructions, difficulty, category, repertoire_tags, source, created_at)
VALUES ('ex-wrm-003', 'Arpeggio Warm-Up', 'Develops hand shape awareness and smooth wrist rotation across broken chords.', 'Play C major arpeggio over 2 octaves with the right hand, then the left. Focus on smooth wrist rotation at each thumb crossing. Play at mf with even tone. Move through C, F, G, and D major arpeggios. Keep fingers close to the keys.', 'beginner', 'warmup', NULL, 'curated', '2026-03-15T00:00:00Z');
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-wrm-003', 'articulation');
```

- [ ] **Step 2: Commit warmup seed data**

```bash
git add apps/api/migrations/0004_exercises.sql
git commit -m "seed 3 warmup exercises"
```

### Task 3: Add seed data -- dynamics exercises (4)

**Files:**
- Modify: `apps/api/migrations/0004_exercises.sql`

- [ ] **Step 1: Append dynamics exercise INSERTs**

Append to `0004_exercises.sql`:

```sql
-- DYNAMICS exercises (4)

INSERT INTO exercises (id, title, description, instructions, difficulty, category, repertoire_tags, source, created_at)
VALUES ('ex-dyn-001', 'Dynamic Contrast Scales', 'Trains smooth dynamic gradients from pp to ff and back, building control over the full dynamic range.', 'Play a C major scale ascending over 2 octaves. Start at pp and arrive at ff by the top. Descend from ff back to pp. Focus on making the gradient as smooth as possible -- no sudden jumps. Repeat in the key of your current piece.', 'intermediate', 'technique', NULL, 'curated', '2026-03-15T00:00:00Z');
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-dyn-001', 'dynamics');

INSERT INTO exercises (id, title, description, instructions, difficulty, category, repertoire_tags, source, created_at)
VALUES ('ex-dyn-002', 'Terraced Dynamics', 'Develops the ability to shift cleanly between distinct dynamic levels without gradual transition.', 'Choose a simple scale or passage you know well. Play it 4 times: first at pp, then p, then f, then ff. Each repetition should be at a consistent, clearly different volume. The goal is distinct levels, not a gradient. Record and listen back: can you hear four separate volumes?', 'beginner', 'technique', NULL, 'curated', '2026-03-15T00:00:00Z');
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-dyn-002', 'dynamics');

INSERT INTO exercises (id, title, description, instructions, difficulty, category, repertoire_tags, source, created_at)
VALUES ('ex-dyn-003', 'Sforzando Control', 'Builds control over sudden accents within a quiet context, crucial for Beethoven and similar repertoire.', 'Play a C major scale at pp. On every 4th note, play a sforzando (sf) -- a sudden accent -- then immediately return to pp. The accent should be sharp and deliberate, not a gradual swell. Practice until the contrast between sf and pp is dramatic but controlled.', 'intermediate', 'technique', '["Beethoven"]', 'curated', '2026-03-15T00:00:00Z');
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-dyn-003', 'dynamics');
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-dyn-003', 'articulation');

INSERT INTO exercises (id, title, description, instructions, difficulty, category, repertoire_tags, source, created_at)
VALUES ('ex-dyn-004', 'Chopin Cantabile Voicing', 'Develops the ability to project a singing melody over a soft accompaniment, essential for Romantic repertoire.', 'Choose a passage where the right hand has a melody over left hand chords or arpeggios. Play the LH alone at pp. Now add the RH melody at mf. The melody should float clearly above the accompaniment. If the LH starts competing, reduce it further. Record and listen: is the melody always audible?', 'advanced', 'musicality', '["Chopin", "romantic"]', 'curated', '2026-03-15T00:00:00Z');
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-dyn-004', 'dynamics');
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-dyn-004', 'phrasing');
```

- [ ] **Step 2: Commit dynamics seed data**

```bash
git add apps/api/migrations/0004_exercises.sql
git commit -m "seed 4 dynamics exercises"
```

### Task 4: Add seed data -- timing exercises (3)

**Files:**
- Modify: `apps/api/migrations/0004_exercises.sql`

- [ ] **Step 1: Append timing exercise INSERTs**

Append to `0004_exercises.sql`:

```sql
-- TIMING exercises (3)

INSERT INTO exercises (id, title, description, instructions, difficulty, category, repertoire_tags, source, created_at)
VALUES ('ex-tim-001', 'Metronome Pulse Training', 'Builds internal pulse by gradually removing external timing support.', 'Set a metronome to 72 bpm. Play a simple scale or passage for 8 bars with the metronome. Then mute the metronome for 4 bars, keeping the same tempo internally. Un-mute and check: are you still aligned? Repeat, gradually extending the silent bars to 8, then 16.', 'beginner', 'technique', NULL, 'curated', '2026-03-15T00:00:00Z');
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-tim-001', 'timing');

INSERT INTO exercises (id, title, description, instructions, difficulty, category, repertoire_tags, source, created_at)
VALUES ('ex-tim-002', 'Mozart Evenness Drill', 'Develops rhythmic clarity and evenness in fast passage work, essential for Classical repertoire.', 'Choose a fast passage (Alberti bass, running 16ths, or scales). Play it at half tempo with a metronome, focusing on absolute evenness -- every note the same length and weight. Record and listen: do any notes rush or drag? Gradually increase tempo by 4 bpm increments only when the current tempo is perfectly even.', 'intermediate', 'technique', '["Mozart", "classical"]', 'curated', '2026-03-15T00:00:00Z');
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-tim-002', 'timing');
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-tim-002', 'articulation');

INSERT INTO exercises (id, title, description, instructions, difficulty, category, repertoire_tags, source, created_at)
VALUES ('ex-tim-003', 'Rubato Awareness', 'Develops intentional rubato by separating rhythmic freedom from rhythmic imprecision.', 'Choose an expressive passage from your piece. First, play it strictly in tempo with a metronome -- no rubato at all. Now play it again without the metronome, adding rubato where you feel it should go. Record both versions. Listen: does your rubato version sound intentional, or does it just sound like uneven timing? The difference is whether the rubato serves the musical phrase.', 'advanced', 'musicality', '["Chopin", "romantic"]', 'curated', '2026-03-15T00:00:00Z');
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-tim-003', 'timing');
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-tim-003', 'phrasing');
```

- [ ] **Step 2: Commit timing seed data**

```bash
git add apps/api/migrations/0004_exercises.sql
git commit -m "seed 3 timing exercises"
```

### Task 5: Add seed data -- pedaling exercises (3)

**Files:**
- Modify: `apps/api/migrations/0004_exercises.sql`

- [ ] **Step 1: Append pedaling exercise INSERTs**

Append to `0004_exercises.sql`:

```sql
-- PEDALING exercises (3)

INSERT INTO exercises (id, title, description, instructions, difficulty, category, repertoire_tags, source, created_at)
VALUES ('ex-ped-001', 'Legato Pedal Harmonic Changes', 'Develops clean pedal transitions at harmonic boundaries, preventing harmonic bleed between chords.', 'Take a 4-bar phrase from your Chopin piece where the harmony changes every bar. Play with full sustain pedal. Now: lift the pedal exactly on each new harmony and re-engage immediately. Listen for any overlap or gap. The goal is a seamless legato with zero harmonic bleed.', 'intermediate', 'technique', '["Chopin", "romantic"]', 'curated', '2026-03-15T00:00:00Z');
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-ped-001', 'pedaling');

INSERT INTO exercises (id, title, description, instructions, difficulty, category, repertoire_tags, source, created_at)
VALUES ('ex-ped-002', 'Half-Pedal Color Palette', 'Explores the range of pedal depths for tonal color, moving beyond binary pedal on/off.', 'Choose a sustained chord passage. Play it with full pedal, then half pedal, then quarter pedal, then no pedal. Listen to how each depth changes the resonance and color. Now play the passage with varying pedal depth -- deeper on rich harmonies, shallower on moving passages. Develop sensitivity to the pedal as a continuous control, not a switch.', 'advanced', 'technique', '["Debussy", "impressionist"]', 'curated', '2026-03-15T00:00:00Z');
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-ped-002', 'pedaling');

INSERT INTO exercises (id, title, description, instructions, difficulty, category, repertoire_tags, source, created_at)
VALUES ('ex-ped-003', 'Debussy Layered Pedaling', 'Develops the ability to create tonal layers through pedal technique, essential for Impressionist repertoire.', 'Choose a Debussy passage with sustained bass notes and moving upper voices. Play the bass note with full pedal, then gradually thin the pedal as the upper voices move. The bass should sustain (via finger legato or partial pedal) while upper voices remain clear. Listen for muddy vs. clear textures.', 'advanced', 'musicality', '["Debussy", "impressionist"]', 'curated', '2026-03-15T00:00:00Z');
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-ped-003', 'pedaling');
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-ped-003', 'phrasing');
```

- [ ] **Step 2: Commit pedaling seed data**

```bash
git add apps/api/migrations/0004_exercises.sql
git commit -m "seed 3 pedaling exercises"
```

### Task 6: Add seed data -- articulation exercises (4)

**Files:**
- Modify: `apps/api/migrations/0004_exercises.sql`

- [ ] **Step 1: Append articulation exercise INSERTs**

Append to `0004_exercises.sql`:

```sql
-- ARTICULATION exercises (4)

INSERT INTO exercises (id, title, description, instructions, difficulty, category, repertoire_tags, source, created_at)
VALUES ('ex-art-001', 'Legato vs Staccato Contrast', 'Develops deliberate control over touch quality by alternating between connected and detached playing.', 'Play a C major scale ascending in legato -- each note connects smoothly to the next with no gaps. Descend in staccato -- each note short and crisp. Now alternate: legato ascending, staccato descending, 4 times. The contrast should be dramatic and consistent.', 'beginner', 'technique', NULL, 'curated', '2026-03-15T00:00:00Z');
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-art-001', 'articulation');

INSERT INTO exercises (id, title, description, instructions, difficulty, category, repertoire_tags, source, created_at)
VALUES ('ex-art-002', 'Bach Voice Independence', 'Builds the ability to maintain independent articulation across voices, essential for contrapuntal music.', 'Choose a Bach two-part invention or a fugue exposition. Play each voice alone with its own articulation -- the subject might be legato while the counter-subject is more detached. Now combine both voices while maintaining their independent articulation. If one voice starts mimicking the other, isolate and retry.', 'advanced', 'musicality', '["Bach", "baroque"]', 'curated', '2026-03-15T00:00:00Z');
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-art-002', 'articulation');
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-art-002', 'phrasing');

INSERT INTO exercises (id, title, description, instructions, difficulty, category, repertoire_tags, source, created_at)
VALUES ('ex-art-003', 'Beethoven Legato-Staccato Passages', 'Practices the rapid articulation changes that Beethoven demands, particularly sforzando within legato lines.', 'Find a Beethoven passage where articulation shifts quickly -- legato melody interrupted by staccato chords, or sforzando accents within a smooth line. Practice each articulation type separately at half tempo. Then combine, focusing on making each transition instant and deliberate, not gradual.', 'intermediate', 'technique', '["Beethoven", "classical"]', 'curated', '2026-03-15T00:00:00Z');
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-art-003', 'articulation');
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-art-003', 'dynamics');

INSERT INTO exercises (id, title, description, instructions, difficulty, category, repertoire_tags, source, created_at)
VALUES ('ex-art-004', 'Mozart Clarity Drill', 'Develops the crystal-clear articulation that Mozart requires, where every note is distinct and precisely placed.', 'Choose a Mozart passage with running eighth or sixteenth notes. Play at half tempo, lifting each finger cleanly before the next note. Every note should have a clear beginning and ending. No pedal. Record and listen: can you hear every single note distinctly? Gradually increase tempo while maintaining clarity.', 'intermediate', 'technique', '["Mozart", "classical"]', 'curated', '2026-03-15T00:00:00Z');
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-art-004', 'articulation');
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-art-004', 'timing');
```

- [ ] **Step 2: Commit articulation seed data**

```bash
git add apps/api/migrations/0004_exercises.sql
git commit -m "seed 4 articulation exercises"
```

### Task 7: Add seed data -- phrasing exercises (4)

**Files:**
- Modify: `apps/api/migrations/0004_exercises.sql`

- [ ] **Step 1: Append phrasing exercise INSERTs**

Append to `0004_exercises.sql`:

```sql
-- PHRASING exercises (4)

INSERT INTO exercises (id, title, description, instructions, difficulty, category, repertoire_tags, source, created_at)
VALUES ('ex-phr-001', 'Phrase Breathing', 'Develops awareness of phrase structure by treating musical phrases like vocal breaths.', 'Choose a lyrical passage. Identify the phrase boundaries -- where would a singer breathe? Mark them. Play through, making a tiny lift (not a full stop) at each breath point. The lift should be barely audible but enough to create shape. Now play without the lifts and notice how the music loses direction.', 'intermediate', 'musicality', NULL, 'curated', '2026-03-15T00:00:00Z');
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-phr-001', 'phrasing');

INSERT INTO exercises (id, title, description, instructions, difficulty, category, repertoire_tags, source, created_at)
VALUES ('ex-phr-002', 'Chopin Long Line', 'Trains the ability to shape extended melodic lines with direction and arrival points.', 'Take an 8-bar Chopin melody. Identify the highest note or the harmonic climax -- that is your arrival point. Play the phrase building toward that point, then relaxing away from it. Every note should either be going toward or coming from the climax. If the phrase feels aimless, your arrival point is wrong or your shaping is not committed enough.', 'intermediate', 'musicality', '["Chopin", "romantic"]', 'curated', '2026-03-15T00:00:00Z');
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-phr-002', 'phrasing');
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-phr-002', 'dynamics');

INSERT INTO exercises (id, title, description, instructions, difficulty, category, repertoire_tags, source, created_at)
VALUES ('ex-phr-003', 'Bach Phrase Architecture', 'Develops awareness of phrase structure in contrapuntal music where multiple voices phrase independently.', 'In a Bach fugue or invention, mark the subject entries in each voice. Each subject entry is a phrase with its own shape. Play each voice alone and shape each subject entry with a clear arc (beginning, peak, resolution). Now play all voices together -- can each subject entry still be heard as a shaped phrase?', 'advanced', 'musicality', '["Bach", "baroque"]', 'curated', '2026-03-15T00:00:00Z');
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-phr-003', 'phrasing');
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-phr-003', 'articulation');

INSERT INTO exercises (id, title, description, instructions, difficulty, category, repertoire_tags, source, created_at)
VALUES ('ex-phr-004', 'Debussy Color Phrasing', 'Develops phrase shaping through color and texture changes rather than just dynamic changes.', 'Choose a Debussy passage. Instead of shaping phrases with volume alone, experiment with touch quality: deeper key contact for warm tones, lighter touch for transparent textures. Use pedal depth as part of the phrase shape. The phrase should have a clear arc even if the volume stays relatively constant.', 'advanced', 'musicality', '["Debussy", "impressionist"]', 'curated', '2026-03-15T00:00:00Z');
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-phr-004', 'phrasing');
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-phr-004', 'pedaling');
```

- [ ] **Step 2: Commit phrasing seed data**

```bash
git add apps/api/migrations/0004_exercises.sql
git commit -m "seed 4 phrasing exercises"
```

### Task 8: Add seed data -- interpretation exercises (3) + Melody Extraction

**Files:**
- Modify: `apps/api/migrations/0004_exercises.sql`

- [ ] **Step 1: Append interpretation exercise INSERTs and the Melody Extraction exercise**

Append to `0004_exercises.sql`:

```sql
-- INTERPRETATION exercises (3)

INSERT INTO exercises (id, title, description, instructions, difficulty, category, repertoire_tags, source, created_at)
VALUES ('ex-int-001', 'Character Study', 'Develops interpretive imagination by playing the same music with different emotional characters.', 'Choose a short passage (4-8 bars). Play it three times with different characters: joyful, melancholy, and agitated. Use dynamics, articulation, tempo, and pedaling to create each character. Record all three. Listen back: are the three versions genuinely different in character, or just in volume?', 'intermediate', 'musicality', NULL, 'curated', '2026-03-15T00:00:00Z');
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-int-001', 'interpretation');

INSERT INTO exercises (id, title, description, instructions, difficulty, category, repertoire_tags, source, created_at)
VALUES ('ex-int-002', 'Comparative Listening', 'Develops interpretive awareness by analyzing how professionals make different choices with the same score.', 'Find two professional recordings of the passage you are working on (YouTube is fine). Listen to each twice, noting: where do they differ in tempo, dynamics, rubato, and pedaling? Which choices do you prefer? Why? Now play the passage yourself, consciously borrowing one choice from each performer. Your interpretation should be informed by theirs but not a copy.', 'intermediate', 'ear-training', NULL, 'curated', '2026-03-15T00:00:00Z');
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-int-002', 'interpretation');

INSERT INTO exercises (id, title, description, instructions, difficulty, category, repertoire_tags, source, created_at)
VALUES ('ex-int-003', 'Structural Emphasis', 'Builds interpretive depth by connecting musical structure to performance decisions.', 'Analyze the harmonic structure of your passage: where are the tensions, resolutions, surprises, and cadences? For each structural event, decide how you will signal it: a slight ritardando before a resolution? A dynamic push into a modulation? Write your decisions on the score. Now play, executing each decision. The structure should be audible to a listener who does not have the score.', 'advanced', 'musicality', NULL, 'curated', '2026-03-15T00:00:00Z');
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-int-003', 'interpretation');
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-int-003', 'phrasing');

-- MELODY EXTRACTION (spec example: articulation + dynamics, advanced, musicality)

INSERT INTO exercises (id, title, description, instructions, difficulty, category, repertoire_tags, source, created_at)
VALUES ('ex-art-005', 'Melody Extraction', 'Develops the ability to project a melody above accompaniment through touch differentiation.', 'Choose a passage where the RH has melody over LH chords. Play the melody alone at mf. Now add the LH at pp -- the melody should remain just as clear. If the LH starts to overpower, reduce it further. Record and listen back: can you hear every note of the melody clearly?', 'advanced', 'musicality', NULL, 'curated', '2026-03-15T00:00:00Z');
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-art-005', 'articulation');
INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES ('ex-art-005', 'dynamics');
```

- [ ] **Step 2: Verify final exercise count**

Run: `grep -c "^INSERT INTO exercises" apps/api/migrations/0004_exercises.sql`
Expected: `25`

Run: `grep -c "^INSERT INTO exercise_dimensions" apps/api/migrations/0004_exercises.sql`
Expected: A number greater than 25 (many exercises have 2 dimensions)

- [ ] **Step 3: Commit interpretation + Melody Extraction seed data**

```bash
git add apps/api/migrations/0004_exercises.sql
git commit -m "seed interpretation + melody extraction exercises (25 total)

3 warmup + 4 dynamics + 3 timing + 3 pedaling + 4 articulation +
4 phrasing + 3 interpretation + 1 melody extraction = 25"
```

---

## Chunk 2: Rust Exercise Service

### Task 9: Create exercise service module with types

**Files:**
- Create: `apps/api/src/services/exercises.rs`
- Modify: `apps/api/src/services/mod.rs`

- [ ] **Step 1: Create exercises.rs with types and module registration**

Create `apps/api/src/services/exercises.rs`:

```rust
use wasm_bindgen::JsValue;
use worker::{console_error, Env};

// -- Response types --

#[derive(serde::Serialize)]
pub struct Exercise {
    pub id: String,
    pub title: String,
    pub description: String,
    pub instructions: String,
    pub difficulty: String,
    pub category: String,
    pub repertoire_tags: Option<String>,
    pub source: String,
    pub dimensions: Vec<String>,
}

#[derive(serde::Serialize)]
pub struct StudentExercise {
    pub id: String,
    pub student_id: String,
    pub exercise_id: String,
    pub session_id: Option<String>,
    pub assigned_at: String,
    pub completed: bool,
    pub response: Option<String>,
    pub times_assigned: i64,
}

// -- Request types --

#[derive(serde::Deserialize)]
pub struct AssignRequest {
    pub exercise_id: String,
    pub session_id: Option<String>,
}

#[derive(serde::Deserialize)]
pub struct CompleteRequest {
    pub student_exercise_id: String,
    pub response: Option<String>,
    pub dimension_before_json: Option<String>,
    pub dimension_after_json: Option<String>,
    pub notes: Option<String>,
}

// -- Query param parsing --

pub struct ExerciseQueryParams {
    pub dimension: Option<String>,
    pub level: Option<String>,
    pub repertoire: Option<String>,
}

pub fn parse_exercise_query_params(query: &str) -> ExerciseQueryParams {
    let params: std::collections::HashMap<String, String> = query
        .split('&')
        .filter_map(|pair| {
            let mut parts = pair.splitn(2, '=');
            let key = parts.next()?.to_string();
            let value = parts.next()?.to_string();
            if value.is_empty() {
                None
            } else {
                Some((key, value))
            }
        })
        .collect();

    ExerciseQueryParams {
        dimension: params.get("dimension").cloned(),
        level: params.get("level").cloned(),
        repertoire: params.get("repertoire").cloned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_all_params() {
        let params = parse_exercise_query_params("dimension=dynamics&level=intermediate&repertoire=Chopin");
        assert_eq!(params.dimension.as_deref(), Some("dynamics"));
        assert_eq!(params.level.as_deref(), Some("intermediate"));
        assert_eq!(params.repertoire.as_deref(), Some("Chopin"));
    }

    #[test]
    fn test_parse_empty_query() {
        let params = parse_exercise_query_params("");
        assert!(params.dimension.is_none());
        assert!(params.level.is_none());
        assert!(params.repertoire.is_none());
    }

    #[test]
    fn test_parse_partial_params() {
        let params = parse_exercise_query_params("dimension=pedaling");
        assert_eq!(params.dimension.as_deref(), Some("pedaling"));
        assert!(params.level.is_none());
        assert!(params.repertoire.is_none());
    }

    #[test]
    fn test_parse_empty_value_ignored() {
        let params = parse_exercise_query_params("dimension=&level=beginner");
        assert!(params.dimension.is_none());
        assert_eq!(params.level.as_deref(), Some("beginner"));
    }

    #[test]
    fn test_assign_request_deserialization() {
        let json = r#"{"exercise_id": "ex-dyn-001", "session_id": "sess-123"}"#;
        let req: AssignRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.exercise_id, "ex-dyn-001");
        assert_eq!(req.session_id.as_deref(), Some("sess-123"));
    }

    #[test]
    fn test_assign_request_optional_session() {
        let json = r#"{"exercise_id": "ex-dyn-001"}"#;
        let req: AssignRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.exercise_id, "ex-dyn-001");
        assert!(req.session_id.is_none());
    }

    #[test]
    fn test_complete_request_all_fields() {
        let json = r#"{"student_exercise_id": "se-abc", "response": "positive", "dimension_before_json": "{}", "dimension_after_json": "{}", "notes": "felt good"}"#;
        let req: CompleteRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.student_exercise_id, "se-abc");
        assert_eq!(req.response.as_deref(), Some("positive"));
        assert_eq!(req.notes.as_deref(), Some("felt good"));
    }

    #[test]
    fn test_complete_request_minimal() {
        let json = r#"{"student_exercise_id": "se-abc"}"#;
        let req: CompleteRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.student_exercise_id, "se-abc");
        assert!(req.response.is_none());
        assert!(req.dimension_before_json.is_none());
    }

    #[test]
    fn test_exercise_serialization() {
        let exercise = Exercise {
            id: "ex-dyn-001".to_string(),
            title: "Test".to_string(),
            description: "Desc".to_string(),
            instructions: "Do this".to_string(),
            difficulty: "intermediate".to_string(),
            category: "technique".to_string(),
            repertoire_tags: Some(r#"["Chopin"]"#.to_string()),
            source: "curated".to_string(),
            dimensions: vec!["dynamics".to_string()],
        };
        let json = serde_json::to_value(&exercise).unwrap();
        assert_eq!(json["id"], "ex-dyn-001");
        assert_eq!(json["dimensions"][0], "dynamics");
        // Intentionally omits notation_content, notation_format, midi_content
        assert!(json.get("notation_content").is_none());
    }
}
```

Add to `apps/api/src/services/mod.rs`:

```rust
pub mod exercises;
```

- [ ] **Step 2: Run unit tests**

Run: `cd apps/api && cargo test`
Expected: All tests pass

- [ ] **Step 3: Commit types and tests**

```bash
git add apps/api/src/services/exercises.rs apps/api/src/services/mod.rs
git commit -m "add exercise service types with unit tests"
```

### Task 10: Implement GET /api/exercises handler

**Files:**
- Modify: `apps/api/src/services/exercises.rs`

- [ ] **Step 1: Add the handle_exercises function**

Append to `apps/api/src/services/exercises.rs` (before the `#[cfg(test)]` module):

```rust
pub async fn handle_exercises(
    env: &Env,
    headers: &http::HeaderMap,
    query_string: &str,
) -> http::Response<axum::body::Body> {
    use axum::body::Body;
    use http::{Response, StatusCode};

    // Verify auth
    let student_id = match crate::auth::verify_auth_header(headers, env) {
        Ok(id) => id,
        Err(err_response) => return err_response,
    };

    let params = parse_exercise_query_params(query_string);

    let db = match env.d1("DB") {
        Ok(db) => db,
        Err(e) => {
            console_error!("D1 binding failed: {:?}", e);
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Database connection failed"}"#))
                .unwrap();
        }
    };

    // Build dynamic SQL based on which params are provided
    let mut sql = String::from(
        "SELECT DISTINCT e.id, e.title, e.description, e.instructions, \
         e.difficulty, e.category, e.repertoire_tags, e.source \
         FROM exercises e \
         JOIN exercise_dimensions ed ON ed.exercise_id = e.id \
         LEFT JOIN student_exercises se ON se.exercise_id = e.id AND se.student_id = ?1 \
         WHERE se.id IS NULL"
    );

    let mut bind_values: Vec<JsValue> = vec![JsValue::from_str(&student_id)];
    let mut bind_index = 2;

    if let Some(ref dim) = params.dimension {
        sql.push_str(&format!(" AND ed.dimension = ?{}", bind_index));
        bind_values.push(JsValue::from_str(dim));
        bind_index += 1;
    }

    if let Some(ref level) = params.level {
        sql.push_str(&format!(" AND e.difficulty = ?{}", bind_index));
        bind_values.push(JsValue::from_str(level));
        bind_index += 1;
    }

    // Soft-prefer repertoire match if provided
    if let Some(ref rep) = params.repertoire {
        sql.push_str(&format!(
            " ORDER BY CASE WHEN e.repertoire_tags LIKE '%' || ?{} || '%' THEN 0 ELSE 1 END, RANDOM()",
            bind_index
        ));
        bind_values.push(JsValue::from_str(rep));
    } else {
        sql.push_str(" ORDER BY RANDOM()");
    }

    sql.push_str(" LIMIT 3");

    let stmt = match db.prepare(&sql).bind(&bind_values) {
        Ok(stmt) => stmt,
        Err(e) => {
            console_error!("Failed to prepare exercise query: {:?}", e);
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Query preparation failed"}"#))
                .unwrap();
        }
    };

    let rows = match stmt.all().await {
        Ok(result) => result.results::<serde_json::Value>().unwrap_or_default(),
        Err(e) => {
            console_error!("Exercise query failed: {:?}", e);
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Query execution failed"}"#))
                .unwrap();
        }
    };

    // For each exercise, fetch its dimensions
    let mut exercises: Vec<Exercise> = Vec::new();
    for row in &rows {
        let id = row["id"].as_str().unwrap_or_default().to_string();

        let dimensions = match fetch_exercise_dimensions(&db, &id).await {
            Ok(dims) => dims,
            Err(_) => vec![],
        };

        exercises.push(Exercise {
            id,
            title: row["title"].as_str().unwrap_or_default().to_string(),
            description: row["description"].as_str().unwrap_or_default().to_string(),
            instructions: row["instructions"].as_str().unwrap_or_default().to_string(),
            difficulty: row["difficulty"].as_str().unwrap_or_default().to_string(),
            category: row["category"].as_str().unwrap_or_default().to_string(),
            repertoire_tags: row["repertoire_tags"].as_str().map(|s| s.to_string()),
            source: row["source"].as_str().unwrap_or_default().to_string(),
            dimensions,
        });
    }

    let response_body = serde_json::json!({ "exercises": exercises });
    let json = serde_json::to_string(&response_body).unwrap_or_else(|_| r#"{"exercises":[]}"#.to_string());

    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Body::from(json))
        .unwrap()
}

async fn fetch_exercise_dimensions(
    db: &worker::D1Database,
    exercise_id: &str,
) -> Result<Vec<String>, String> {
    let stmt = db
        .prepare("SELECT dimension FROM exercise_dimensions WHERE exercise_id = ?1")
        .bind(&[JsValue::from_str(exercise_id)])
        .map_err(|e| format!("Failed to bind dimensions query: {:?}", e))?;

    let result = stmt
        .all()
        .await
        .map_err(|e| format!("Dimensions query failed: {:?}", e))?;

    let rows = result.results::<serde_json::Value>().unwrap_or_default();
    Ok(rows
        .iter()
        .filter_map(|r| r["dimension"].as_str().map(|s| s.to_string()))
        .collect())
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cd apps/api && cargo check`
Expected: No errors

- [ ] **Step 3: Commit GET handler**

```bash
git add apps/api/src/services/exercises.rs
git commit -m "implement GET /api/exercises handler"
```

### Task 11: Implement POST /api/exercises/assign handler

**Files:**
- Modify: `apps/api/src/services/exercises.rs`

- [ ] **Step 1: Add the handle_assign_exercise function**

Append to `apps/api/src/services/exercises.rs` (before the `#[cfg(test)]` module):

```rust
pub async fn handle_assign_exercise(
    env: &Env,
    headers: &http::HeaderMap,
    body: &[u8],
) -> http::Response<axum::body::Body> {
    use axum::body::Body;
    use http::{Response, StatusCode};

    let student_id = match crate::auth::verify_auth_header(headers, env) {
        Ok(id) => id,
        Err(err_response) => return err_response,
    };

    let request: AssignRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => {
            console_error!("Failed to parse assign request: {:?}", e);
            return Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Invalid request body"}"#))
                .unwrap();
        }
    };

    let db = match env.d1("DB") {
        Ok(db) => db,
        Err(e) => {
            console_error!("D1 binding failed: {:?}", e);
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Database connection failed"}"#))
                .unwrap();
        }
    };

    // Get current max times_assigned for this (student, exercise) pair
    let times_assigned = match db
        .prepare("SELECT MAX(times_assigned) as max_ta FROM student_exercises WHERE student_id = ?1 AND exercise_id = ?2")
        .bind(&[
            JsValue::from_str(&student_id),
            JsValue::from_str(&request.exercise_id),
        ]) {
        Ok(stmt) => {
            match stmt.first::<serde_json::Value>(None).await {
                Ok(Some(row)) => {
                    row["max_ta"].as_i64().map(|v| v + 1).unwrap_or(1)
                }
                _ => 1,
            }
        }
        Err(_) => 1,
    };

    // Generate ID (reuse existing UUID generator from ask service)
    let id = format!("se-{}", crate::services::ask::generate_uuid());

    let now = js_sys::Date::new_0()
        .to_iso_string()
        .as_string()
        .unwrap_or_default();

    let session_id_val = match &request.session_id {
        Some(s) => JsValue::from_str(s),
        None => JsValue::NULL,
    };

    let insert_result = db
        .prepare(
            "INSERT INTO student_exercises (id, student_id, exercise_id, session_id, assigned_at, completed, times_assigned) \
             VALUES (?1, ?2, ?3, ?4, ?5, 0, ?6)"
        )
        .bind(&[
            JsValue::from_str(&id),
            JsValue::from_str(&student_id),
            JsValue::from_str(&request.exercise_id),
            session_id_val,
            JsValue::from_str(&now),
            JsValue::from_f64(times_assigned as f64),
        ]);

    match insert_result {
        Ok(stmt) => {
            if let Err(e) = stmt.run().await {
                console_error!("Failed to insert student_exercise: {:?}", e);
                return Response::builder()
                    .status(StatusCode::INTERNAL_SERVER_ERROR)
                    .header("Content-Type", "application/json")
                    .body(Body::from(r#"{"error":"Failed to assign exercise"}"#))
                    .unwrap();
            }
        }
        Err(e) => {
            console_error!("Failed to prepare insert: {:?}", e);
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Failed to assign exercise"}"#))
                .unwrap();
        }
    }

    let record = StudentExercise {
        id,
        student_id,
        exercise_id: request.exercise_id,
        session_id: request.session_id,
        assigned_at: now,
        completed: false,
        response: None,
        times_assigned,
    };

    let json = serde_json::to_string(&record).unwrap_or_else(|_| "{}".to_string());

    Response::builder()
        .status(StatusCode::CREATED)
        .header("Content-Type", "application/json")
        .body(Body::from(json))
        .unwrap()
}

```

- [ ] **Step 2: Verify it compiles**

Run: `cd apps/api && cargo check`
Expected: No errors

- [ ] **Step 3: Commit assign handler**

```bash
git add apps/api/src/services/exercises.rs
git commit -m "implement POST /api/exercises/assign handler"
```

### Task 12: Implement POST /api/exercises/complete handler

**Files:**
- Modify: `apps/api/src/services/exercises.rs`

- [ ] **Step 1: Add the handle_complete_exercise function**

Append to `apps/api/src/services/exercises.rs` (before the `#[cfg(test)]` module):

```rust
pub async fn handle_complete_exercise(
    env: &Env,
    headers: &http::HeaderMap,
    body: &[u8],
) -> http::Response<axum::body::Body> {
    use axum::body::Body;
    use http::{Response, StatusCode};

    let student_id = match crate::auth::verify_auth_header(headers, env) {
        Ok(id) => id,
        Err(err_response) => return err_response,
    };

    let request: CompleteRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => {
            console_error!("Failed to parse complete request: {:?}", e);
            return Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Invalid request body"}"#))
                .unwrap();
        }
    };

    let db = match env.d1("DB") {
        Ok(db) => db,
        Err(e) => {
            console_error!("D1 binding failed: {:?}", e);
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Database connection failed"}"#))
                .unwrap();
        }
    };

    // Verify the record belongs to this student
    let existing = match db
        .prepare("SELECT id, student_id, exercise_id, session_id, assigned_at, times_assigned FROM student_exercises WHERE id = ?1")
        .bind(&[JsValue::from_str(&request.student_exercise_id)])
    {
        Ok(stmt) => match stmt.first::<serde_json::Value>(None).await {
            Ok(Some(row)) => row,
            Ok(None) => {
                return Response::builder()
                    .status(StatusCode::NOT_FOUND)
                    .header("Content-Type", "application/json")
                    .body(Body::from(r#"{"error":"Student exercise record not found"}"#))
                    .unwrap();
            }
            Err(e) => {
                console_error!("Failed to query student_exercise: {:?}", e);
                return Response::builder()
                    .status(StatusCode::INTERNAL_SERVER_ERROR)
                    .header("Content-Type", "application/json")
                    .body(Body::from(r#"{"error":"Query failed"}"#))
                    .unwrap();
            }
        },
        Err(e) => {
            console_error!("Failed to prepare query: {:?}", e);
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Query preparation failed"}"#))
                .unwrap();
        }
    };

    // Check ownership
    let record_student_id = existing["student_id"].as_str().unwrap_or_default();
    if record_student_id != student_id {
        return Response::builder()
            .status(StatusCode::FORBIDDEN)
            .header("Content-Type", "application/json")
            .body(Body::from(r#"{"error":"Not authorized to complete this exercise"}"#))
            .unwrap();
    }

    // Update the record
    let response_val = match &request.response {
        Some(r) => JsValue::from_str(r),
        None => JsValue::NULL,
    };
    let before_val = match &request.dimension_before_json {
        Some(v) => JsValue::from_str(v),
        None => JsValue::NULL,
    };
    let after_val = match &request.dimension_after_json {
        Some(v) => JsValue::from_str(v),
        None => JsValue::NULL,
    };
    let notes_val = match &request.notes {
        Some(n) => JsValue::from_str(n),
        None => JsValue::NULL,
    };

    let update_result = db
        .prepare(
            "UPDATE student_exercises SET completed = 1, response = ?1, \
             dimension_before_json = ?2, dimension_after_json = ?3, notes = ?4 \
             WHERE id = ?5"
        )
        .bind(&[
            response_val,
            before_val,
            after_val,
            notes_val,
            JsValue::from_str(&request.student_exercise_id),
        ]);

    match update_result {
        Ok(stmt) => {
            if let Err(e) = stmt.run().await {
                console_error!("Failed to update student_exercise: {:?}", e);
                return Response::builder()
                    .status(StatusCode::INTERNAL_SERVER_ERROR)
                    .header("Content-Type", "application/json")
                    .body(Body::from(r#"{"error":"Failed to complete exercise"}"#))
                    .unwrap();
            }
        }
        Err(e) => {
            console_error!("Failed to prepare update: {:?}", e);
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Failed to complete exercise"}"#))
                .unwrap();
        }
    }

    let record = StudentExercise {
        id: request.student_exercise_id,
        student_id,
        exercise_id: existing["exercise_id"].as_str().unwrap_or_default().to_string(),
        session_id: existing["session_id"].as_str().map(|s| s.to_string()),
        assigned_at: existing["assigned_at"].as_str().unwrap_or_default().to_string(),
        completed: true,
        response: request.response,
        times_assigned: existing["times_assigned"].as_i64().unwrap_or(1),
    };

    let json = serde_json::to_string(&record).unwrap_or_else(|_| "{}".to_string());

    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Body::from(json))
        .unwrap()
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cd apps/api && cargo check`
Expected: No errors

- [ ] **Step 3: Commit complete handler**

```bash
git add apps/api/src/services/exercises.rs
git commit -m "implement POST /api/exercises/complete handler"
```

### Task 13: Add routes to server.rs

**Files:**
- Modify: `apps/api/src/server.rs`

- [ ] **Step 1: Add three route blocks before the health check**

In `apps/api/src/server.rs`, add these three blocks just before the `// Health check` comment. All three use exact-match `path ==` (not `starts_with`) to avoid sub-path conflicts:

```rust
    // Exercise catalog (authenticated)
    if path == "/api/exercises" && method == http::Method::GET {
        let headers = req.headers().clone();
        let query_string = req.uri().query().unwrap_or_default().to_string();
        return into_worker_response(with_cors(
            crate::services::exercises::handle_exercises(&env, &headers, &query_string).await,
            origin.as_deref(),
        )).await;
    }

    // Assign exercise to student (authenticated)
    if path == "/api/exercises/assign" && method == http::Method::POST {
        let headers = req.headers().clone();
        let body = req
            .into_body()
            .collect()
            .await
            .map(|b| b.to_bytes().to_vec())
            .unwrap_or_default();
        return into_worker_response(with_cors(
            crate::services::exercises::handle_assign_exercise(&env, &headers, &body).await,
            origin.as_deref(),
        )).await;
    }

    // Complete exercise (authenticated)
    if path == "/api/exercises/complete" && method == http::Method::POST {
        let headers = req.headers().clone();
        let body = req
            .into_body()
            .collect()
            .await
            .map(|b| b.to_bytes().to_vec())
            .unwrap_or_default();
        return into_worker_response(with_cors(
            crate::services::exercises::handle_complete_exercise(&env, &headers, &body).await,
            origin.as_deref(),
        )).await;
    }
```

- [ ] **Step 2: Verify it compiles**

Run: `cd apps/api && cargo check`
Expected: No errors

- [ ] **Step 3: Commit routing**

```bash
git add apps/api/src/server.rs
git commit -m "add exercise endpoint routes to server.rs"
```

---

## Chunk 3: Web UI Changes

### Task 14: Update types.ts

**Files:**
- Modify: `apps/web/src/lib/types.ts`

- [ ] **Step 1: Add hands, exercise_id to ExerciseSetConfig and dimension to RichMessage**

In `apps/web/src/lib/types.ts`, update the `ExerciseSetConfig` exercises array type to add `hands` and `exercise_id`, and add `dimension` to `RichMessage`:

Update the `ExerciseSetConfig` interface:

```typescript
export interface ExerciseSetConfig {
	source_passage: string;
	target_skill: string;
	exercises: Array<{
		title: string;
		instruction: string;
		focus_dimension: string;
		hands?: "left" | "right" | "both";
		exercise_id?: string;
	}>;
}
```

Update the `RichMessage` interface:

```typescript
export interface RichMessage {
	id: string;
	role: "user" | "assistant";
	content: string;
	created_at: string;
	streaming?: boolean;
	components?: InlineComponent[];
	dimension?: string;
}
```

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd apps/web && bunx tsc --noEmit`
Expected: No errors (or only pre-existing errors unrelated to this change)

- [ ] **Step 3: Commit types update**

```bash
git add apps/web/src/lib/types.ts
git commit -m "add hands, exercise_id, and dimension to exercise types"
```

### Task 15: Add exercises namespace to api.ts

**Files:**
- Modify: `apps/web/src/lib/api.ts`

- [ ] **Step 1: Add Exercise response type and exercises namespace**

At the end of the type definitions section (after `ChatStreamEvent`), add:

```typescript
// --- Exercise types ---

export interface Exercise {
	id: string;
	title: string;
	description: string;
	instructions: string;
	difficulty: string;
	category: string;
	repertoire_tags: string | null;
	source: string;
	dimensions: string[];
}

export interface StudentExercise {
	id: string;
	student_id: string;
	exercise_id: string;
	session_id: string | null;
	assigned_at: string;
	completed: boolean;
	response: string | null;
	times_assigned: number;
}
```

In the `api` object, after the `chat` namespace, add:

```typescript
	exercises: {
		fetch(params?: {
			dimension?: string;
			level?: string;
			repertoire?: string;
		}): Promise<{ exercises: Exercise[] }> {
			const searchParams = new URLSearchParams();
			if (params?.dimension) searchParams.set("dimension", params.dimension);
			if (params?.level) searchParams.set("level", params.level);
			if (params?.repertoire) searchParams.set("repertoire", params.repertoire);
			const qs = searchParams.toString();
			return request(`/api/exercises${qs ? `?${qs}` : ""}`);
		},

		assign(body: {
			exercise_id: string;
			session_id?: string;
		}): Promise<StudentExercise> {
			return request("/api/exercises/assign", {
				method: "POST",
				body: JSON.stringify(body),
			});
		},

		complete(body: {
			student_exercise_id: string;
			response?: string;
			dimension_before_json?: string;
			dimension_after_json?: string;
			notes?: string;
		}): Promise<StudentExercise> {
			return request("/api/exercises/complete", {
				method: "POST",
				body: JSON.stringify(body),
			});
		},
	},
```

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd apps/web && bunx tsc --noEmit`
Expected: No errors

- [ ] **Step 3: Commit API client**

```bash
git add apps/web/src/lib/api.ts
git commit -m "add exercises namespace to API client"
```

### Task 16: Enhance ExerciseSetCard with hands badge and assign button

**Files:**
- Modify: `apps/web/src/components/cards/ExerciseSetCard.tsx`

- [ ] **Step 1: Update ExerciseSetCard with hands badge and "Try this" button**

Replace the contents of `apps/web/src/components/cards/ExerciseSetCard.tsx`:

```tsx
import { useState } from "react";
import { api } from "../../lib/api";
import type { ExerciseSetConfig } from "../../lib/types";

interface ExerciseSetCardProps {
	config: ExerciseSetConfig;
}

const HANDS_LABELS: Record<string, string> = {
	left: "LH",
	right: "RH",
	both: "Both",
};

export function ExerciseSetCard({ config }: ExerciseSetCardProps) {
	const [expandedIndex, setExpandedIndex] = useState<number | null>(null);

	return (
		<div className="bg-surface-card border border-border rounded-xl p-4 mt-3">
			<h4 className="text-body-sm font-medium text-accent mb-1">
				{config.target_skill}
			</h4>
			<p className="text-body-xs text-text-secondary mb-3">
				{config.source_passage}
			</p>
			<div className="space-y-2">
				{config.exercises.map((exercise, i) => {
					const isExpanded = expandedIndex === i;
					return (
						<ExerciseItem
							key={exercise.title}
							exercise={exercise}
							isExpanded={isExpanded}
							onToggle={() => setExpandedIndex(isExpanded ? null : i)}
						/>
					);
				})}
			</div>
		</div>
	);
}

function ExerciseItem({
	exercise,
	isExpanded,
	onToggle,
}: {
	exercise: ExerciseSetConfig["exercises"][number];
	isExpanded: boolean;
	onToggle: () => void;
}) {
	const [assignState, setAssignState] = useState<
		"idle" | "loading" | "assigned" | "error"
	>("idle");

	async function handleAssign() {
		if (!exercise.exercise_id || assignState !== "idle") return;
		setAssignState("loading");
		try {
			await api.exercises.assign({ exercise_id: exercise.exercise_id });
			setAssignState("assigned");
		} catch {
			setAssignState("error");
		}
	}

	return (
		<div className="border border-border rounded-lg overflow-hidden">
			<button
				type="button"
				onClick={onToggle}
				className="w-full flex items-center justify-between px-3 py-2 text-left hover:bg-surface transition"
			>
				<span className="text-body-sm text-cream font-medium">
					{exercise.title}
				</span>
				<span className="flex items-center gap-2 ml-2">
					{exercise.hands && (
						<span className="text-body-xs text-text-tertiary bg-surface px-1.5 py-0.5 rounded">
							{HANDS_LABELS[exercise.hands] ?? exercise.hands}
						</span>
					)}
					<span className="text-body-xs text-text-tertiary">
						{exercise.focus_dimension}
					</span>
				</span>
			</button>
			{isExpanded && (
				<div className="px-3 pb-3 pt-1 border-t border-border">
					<p className="text-body-sm text-text-secondary">
						{exercise.instruction}
					</p>
					{exercise.exercise_id && (
						<div className="mt-2">
							{assignState === "idle" && (
								<button
									type="button"
									onClick={handleAssign}
									className="text-body-xs text-accent hover:text-accent-lighter transition"
								>
									Try this
								</button>
							)}
							{assignState === "loading" && (
								<span className="text-body-xs text-text-tertiary">
									Assigning...
								</span>
							)}
							{assignState === "assigned" && (
								<span className="text-body-xs text-text-tertiary">
									Assigned
								</span>
							)}
							{assignState === "error" && (
								<span className="text-body-xs text-red-400">
									Could not assign exercise
								</span>
							)}
						</div>
					)}
				</div>
			)}
		</div>
	);
}
```

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd apps/web && bunx tsc --noEmit`
Expected: No errors

- [ ] **Step 3: Commit ExerciseSetCard enhancement**

```bash
git add apps/web/src/components/cards/ExerciseSetCard.tsx
git commit -m "enhance ExerciseSetCard with hands badge and assign button"
```

### Task 17: Add "Try exercises" action to ChatMessages

**Files:**
- Modify: `apps/web/src/components/ChatMessages.tsx`

- [ ] **Step 1: Add onTryExercises prop and action button**

Update `ChatMessages.tsx` to accept and pass `onTryExercises`:

```tsx
import { memo, useCallback, useEffect, useRef, useState } from "react";
import type { RichMessage } from "../lib/types";
import { InlineCard } from "./InlineCard";
import { MessageContent } from "./MessageContent";

interface ChatMessagesProps {
	messages: RichMessage[];
	children?: React.ReactNode;
	onTryExercises?: (dimension: string) => Promise<void>;
}

export function ChatMessages({ messages, children, onTryExercises }: ChatMessagesProps) {
	const scrollContainerRef = useRef<HTMLDivElement>(null);
	const isNearBottomRef = useRef(true);
	const prevMessageCountRef = useRef(0);

	const scrollToBottom = useCallback(
		(behavior: ScrollBehavior = "instant") => {
			const container = scrollContainerRef.current;
			if (!container) return;
			if (behavior === "smooth") {
				container.scrollTo({ top: container.scrollHeight, behavior: "smooth" });
			} else {
				container.scrollTop = container.scrollHeight;
			}
		},
		[],
	);

	// Track whether user is near the bottom
	useEffect(() => {
		const container = scrollContainerRef.current;
		if (!container) return;

		function handleScroll() {
			if (!container) return;
			const threshold = 150;
			const distanceFromBottom =
				container.scrollHeight - container.scrollTop - container.clientHeight;
			isNearBottomRef.current = distanceFromBottom <= threshold;
		}

		container.addEventListener("scroll", handleScroll, { passive: true });
		return () => container.removeEventListener("scroll", handleScroll);
	}, []);

	// Auto-scroll on content changes
	useEffect(() => {
		if (!isNearBottomRef.current) return;

		const isNewMessage = messages.length > prevMessageCountRef.current;
		prevMessageCountRef.current = messages.length;

		// Instant scroll for streaming-related changes (avoids jerk);
		// smooth only for new non-streaming message additions
		const lastMsg = messages[messages.length - 1];
		const behavior =
			lastMsg?.streaming
				? "instant"
				: isNewMessage
					? "smooth"
					: "instant";
		scrollToBottom(behavior);
	}, [messages, scrollToBottom]);

	// Scroll on mount
	useEffect(() => {
		scrollToBottom("instant");
	}, [scrollToBottom]);

	if (messages.length === 0) {
		return null;
	}

	return (
		<div
			ref={scrollContainerRef}
			className="flex-1 overflow-y-auto px-6 pt-8 flex flex-col"
			style={{ scrollBehavior: "auto" }}
		>
			<div className="flex-1 max-w-3xl mx-auto space-y-6 w-full">
				{messages.map((msg) => (
					<MessageBubble key={msg.id} message={msg} onTryExercises={onTryExercises} />
				))}
			</div>
			{children}
		</div>
	);
}

const MessageBubble = memo(function MessageBubble({
	message,
	onTryExercises,
}: { message: RichMessage; onTryExercises?: (dimension: string) => Promise<void> }) {
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState<string | null>(null);

	if (message.role === "user") {
		return (
			<div className="flex justify-end">
				<div className="bg-surface border border-border rounded-2xl px-5 py-3 max-w-[80%]">
					<p className="text-body-md text-cream whitespace-pre-wrap">
						{message.content}
					</p>
				</div>
			</div>
		);
	}

	async function handleTryExercises() {
		if (!message.dimension || !onTryExercises || loading) return;
		setLoading(true);
		setError(null);
		try {
			await onTryExercises(message.dimension);
		} catch {
			setError("Could not load exercises");
		} finally {
			setLoading(false);
		}
	}

	return (
		<div className="flex justify-start animate-fade-in">
			<div className="max-w-[80%]">
				<MessageContent content={message.content} />
				{message.components?.map((component, i) => (
					// biome-ignore lint/suspicious/noArrayIndexKey: components have no stable id
					<InlineCard key={`${message.id}-card-${i}`} component={component} />
				))}
				{message.dimension && onTryExercises && !message.streaming && (
					<div className="mt-2">
						{loading ? (
							<span className="text-body-xs text-text-tertiary">Loading exercises...</span>
						) : error ? (
							<span className="text-body-xs text-red-400">{error}</span>
						) : (
							<button
								type="button"
								onClick={handleTryExercises}
								className="text-body-xs text-accent hover:text-accent-lighter transition"
							>
								Try exercises for this
							</button>
						)}
					</div>
				)}
			</div>
		</div>
	);
});
```

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd apps/web && bunx tsc --noEmit`
Expected: No errors

- [ ] **Step 3: Commit ChatMessages update**

```bash
git add apps/web/src/components/ChatMessages.tsx
git commit -m "add try-exercises action button to ChatMessages"
```

### Task 18: Wire onTryExercises callback in AppChat

**Files:**
- Modify: `apps/web/src/components/AppChat.tsx`

- [ ] **Step 1: Add the handleTryExercises callback and pass to ChatMessages**

In `apps/web/src/components/AppChat.tsx`, add the callback function inside the `AppChat` component (after `handleSend`):

```typescript
	const handleTryExercises = useCallback(
		async (dimension: string) => {
			try {
				const { exercises } = await api.exercises.fetch({ dimension });
				if (exercises.length === 0) return;

				const exerciseMsg: RichMessage = {
					id: `exercises-${Date.now()}`,
					role: "assistant",
					content: `Here are some exercises to work on your ${dimension}:`,
					created_at: new Date().toISOString(),
					components: [
						{
							type: "exercise_set" as const,
							config: {
								source_passage: "Based on your recent practice",
								target_skill: `${dimension} improvement`,
								exercises: exercises.map((e) => ({
									title: e.title,
									instruction: e.instructions,
									focus_dimension: e.dimensions[0] ?? dimension,
									exercise_id: e.id,
								})),
							},
						},
					],
				};

				setMessages((prev) => [...prev, exerciseMsg]);
			} catch (e) {
				const errorMessage =
					e instanceof Error ? e.message : "Failed to load exercises";
				addToast({ type: "error", message: errorMessage });
			}
		},
		[addToast],
	);
```

Then update the `ChatMessages` component usage (around line 742) to pass the callback:

Change:
```tsx
<ChatMessages messages={messages}>
```

To:
```tsx
<ChatMessages messages={messages} onTryExercises={handleTryExercises}>
```

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd apps/web && bunx tsc --noEmit`
Expected: No errors

- [ ] **Step 3: Commit AppChat wiring**

```bash
git add apps/web/src/components/AppChat.tsx
git commit -m "wire onTryExercises callback in AppChat"
```

---

## Chunk 4: Integration Tests

### Task 19: Create test runner script

**Files:**
- Create: `apps/api/tests/run.sh`

- [ ] **Step 1: Write the test runner script**

Create `apps/api/tests/run.sh`:

```bash
#!/bin/bash
set -e

cd "$(dirname "$0")/.."

echo "Applying D1 migrations..."
for migration in migrations/0*.sql; do
    echo "  $migration"
    bunx wrangler d1 execute crescendai-db --local --file="$migration" 2>/dev/null || true
done

echo "Starting wrangler dev..."
bunx wrangler dev --local --port 8787 &
WRANGLER_PID=$!

cleanup() {
    echo "Stopping wrangler dev (PID $WRANGLER_PID)..."
    kill $WRANGLER_PID 2>/dev/null || true
    wait $WRANGLER_PID 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for server to be ready..."
for i in $(seq 1 60); do
    if curl -sf http://localhost:8787/health > /dev/null 2>&1; then
        echo "Server ready."
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "Server failed to start within 60 seconds."
        exit 1
    fi
    sleep 1
done

echo "Running tests..."
bun test tests/exercises.test.ts
EXIT_CODE=$?

exit $EXIT_CODE
```

- [ ] **Step 2: Make it executable**

Run: `chmod +x apps/api/tests/run.sh`

- [ ] **Step 3: Commit test runner**

```bash
git add apps/api/tests/run.sh
git commit -m "add exercise integration test runner script"
```

### Task 20: Create integration tests

**Files:**
- Create: `apps/api/tests/exercises.test.ts`

- [ ] **Step 1: Write the integration test file**

Create `apps/api/tests/exercises.test.ts`:

```typescript
import { describe, test, expect, beforeAll } from "bun:test";

const BASE = "http://localhost:8787";
let cookie = "";

beforeAll(async () => {
	// Authenticate via debug endpoint (dev-only, returns JWT cookie)
	const res = await fetch(`${BASE}/api/auth/debug`, {
		method: "POST",
		credentials: "include",
	});
	expect(res.ok).toBe(true);
	const setCookie = res.headers.get("set-cookie");
	expect(setCookie).toBeTruthy();
	// Extract just the cookie name=value pair
	cookie = setCookie!.split(";")[0];
});

function authedFetch(path: string, options: RequestInit = {}) {
	return fetch(`${BASE}${path}`, {
		...options,
		headers: {
			"Content-Type": "application/json",
			Cookie: cookie,
			...options.headers,
		},
	});
}

describe("GET /api/exercises", () => {
	test("returns exercises without filters", async () => {
		const res = await authedFetch("/api/exercises");
		expect(res.ok).toBe(true);
		const data = (await res.json()) as { exercises: unknown[] };
		expect(data.exercises).toBeDefined();
		expect(data.exercises.length).toBeGreaterThan(0);
		expect(data.exercises.length).toBeLessThanOrEqual(3);
	});

	test("filters by dimension", async () => {
		const res = await authedFetch("/api/exercises?dimension=dynamics");
		expect(res.ok).toBe(true);
		const data = (await res.json()) as {
			exercises: Array<{ dimensions: string[] }>;
		};
		for (const exercise of data.exercises) {
			expect(exercise.dimensions).toContain("dynamics");
		}
	});

	test("filters by level", async () => {
		const res = await authedFetch("/api/exercises?level=beginner");
		expect(res.ok).toBe(true);
		const data = (await res.json()) as {
			exercises: Array<{ difficulty: string }>;
		};
		for (const exercise of data.exercises) {
			expect(exercise.difficulty).toBe("beginner");
		}
	});

	test("combined dimension and level filter", async () => {
		const res = await authedFetch(
			"/api/exercises?dimension=dynamics&level=intermediate",
		);
		expect(res.ok).toBe(true);
		const data = (await res.json()) as {
			exercises: Array<{ difficulty: string; dimensions: string[] }>;
		};
		for (const exercise of data.exercises) {
			expect(exercise.difficulty).toBe("intermediate");
			expect(exercise.dimensions).toContain("dynamics");
		}
	});

	test("requires auth", async () => {
		const res = await fetch(`${BASE}/api/exercises`);
		expect(res.status).toBe(401);
	});
});

describe("POST /api/exercises/assign", () => {
	test("assigns an exercise", async () => {
		const res = await authedFetch("/api/exercises/assign", {
			method: "POST",
			body: JSON.stringify({ exercise_id: "ex-dyn-001" }),
		});
		expect(res.status).toBe(201);
		const data = (await res.json()) as {
			id: string;
			exercise_id: string;
			completed: boolean;
			times_assigned: number;
		};
		expect(data.id).toMatch(/^se-/);
		expect(data.exercise_id).toBe("ex-dyn-001");
		expect(data.completed).toBe(false);
		expect(data.times_assigned).toBe(1);
	});

	test("excludes assigned exercise from GET results", async () => {
		// ex-dyn-001 was assigned above, so it should not appear in results
		const res = await authedFetch("/api/exercises?dimension=dynamics");
		expect(res.ok).toBe(true);
		const data = (await res.json()) as {
			exercises: Array<{ id: string }>;
		};
		const ids = data.exercises.map((e) => e.id);
		expect(ids).not.toContain("ex-dyn-001");
	});

	test("increments times_assigned on re-assignment", async () => {
		const res = await authedFetch("/api/exercises/assign", {
			method: "POST",
			body: JSON.stringify({
				exercise_id: "ex-dyn-001",
				session_id: "sess-2",
			}),
		});
		expect(res.status).toBe(201);
		const data = (await res.json()) as { times_assigned: number };
		expect(data.times_assigned).toBe(2);
	});
});

describe("POST /api/exercises/complete", () => {
	let studentExerciseId: string;

	test("assign exercise for completion test", async () => {
		const res = await authedFetch("/api/exercises/assign", {
			method: "POST",
			body: JSON.stringify({
				exercise_id: "ex-tim-001",
				session_id: "sess-complete",
			}),
		});
		expect(res.status).toBe(201);
		const data = (await res.json()) as { id: string };
		studentExerciseId = data.id;
	});

	test("completes an exercise", async () => {
		const res = await authedFetch("/api/exercises/complete", {
			method: "POST",
			body: JSON.stringify({
				student_exercise_id: studentExerciseId,
				response: "positive",
				notes: "felt good",
			}),
		});
		expect(res.ok).toBe(true);
		const data = (await res.json()) as {
			completed: boolean;
			response: string;
		};
		expect(data.completed).toBe(true);
		expect(data.response).toBe("positive");
	});

	test("returns 404 for non-existent record", async () => {
		const res = await authedFetch("/api/exercises/complete", {
			method: "POST",
			body: JSON.stringify({
				student_exercise_id: "se-nonexistent",
			}),
		});
		expect(res.status).toBe(404);
	});
});
```

- [ ] **Step 2: Commit integration tests**

```bash
git add apps/api/tests/exercises.test.ts
git commit -m "add exercise integration tests"
```

### Task 21: Run integration tests

- [ ] **Step 1: Run the test suite**

Run: `cd apps/api && ./tests/run.sh`
Expected: All tests pass

- [ ] **Step 2: Fix any failures and re-run**

If tests fail, fix the issue in the relevant file and re-run. Commit fixes separately.

- [ ] **Step 3: Final commit if needed**

```bash
git add -A
git commit -m "fix integration test issues"
```
