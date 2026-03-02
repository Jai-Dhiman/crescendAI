# Slice 7: Exercise Database

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a structured, queryable database of piano exercises that can be filtered by dimension, difficulty, and repertoire context. Supports both curated exercises (human-authored) and LLM-generated custom exercises adapted from the student's music.

**Architecture:** D1 tables for exercises and student-exercise tracking. Exercises stored as structured records with optional MusicXML/Lilypond for notation rendering. LLM generates custom exercises as text instructions referencing the student's specific passages.

**Tech Stack:** D1 (SQLite), MusicXML/Lilypond (notation content), OSMD or VexFlow (rendering -- deferred to Slice 9)

---

## Context

The exercise database replaces RAG. Instead of retrieving book quotes, the system draws from exercises that the student can actually do. This powers Focus Mode (Slice 8) and enriches the "Tell me more" responses (Slice 6).

## Design

### Exercise Schema

```sql
CREATE TABLE exercises (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT NOT NULL,        -- what this exercise trains, why it matters
    instructions TEXT NOT NULL,       -- step-by-step how to practice it
    target_dimensions TEXT NOT NULL,  -- JSON array: ["dynamics", "pedaling"]
    difficulty TEXT NOT NULL,         -- beginner / intermediate / advanced
    category TEXT NOT NULL,           -- technique / musicality / ear-training / warmup
    repertoire_tags TEXT,             -- JSON array: ["Chopin", "nocturne", "romantic", "LH independence"]
    notation_content TEXT,            -- MusicXML or Lilypond source (nullable for text-only exercises)
    notation_format TEXT,             -- "musicxml" / "lilypond" / null
    midi_content BLOB,               -- MIDI bytes for playback (nullable)
    source TEXT,                      -- "curated" / "method:Hanon" / "method:Czerny" / etc.
    variants_json TEXT,               -- JSON: alternate versions (different keys, tempos, dynamics)
    created_at TEXT NOT NULL
);

CREATE INDEX idx_exercises_dimension ON exercises(target_dimensions);
CREATE INDEX idx_exercises_difficulty ON exercises(difficulty);
```

### Student-Exercise Tracking

```sql
CREATE TABLE student_exercises (
    id TEXT PRIMARY KEY,
    student_id TEXT NOT NULL REFERENCES students(id),
    exercise_id TEXT NOT NULL REFERENCES exercises(id),
    session_id TEXT REFERENCES student_sessions(id),
    assigned_at TEXT NOT NULL,
    completed BOOLEAN DEFAULT FALSE,
    response TEXT,                    -- positive / neutral / negative / skipped
    dimension_before_json TEXT,       -- JSON: 6-dim scores before exercise
    dimension_after_json TEXT,        -- JSON: 6-dim scores after exercise
    notes TEXT,                       -- student feedback on the exercise
    times_assigned INTEGER DEFAULT 1,
    UNIQUE(student_id, exercise_id, session_id)
);

CREATE INDEX idx_student_exercises ON student_exercises(student_id, exercise_id);
```

### Exercise Query Logic

```sql
-- Find exercises for a student working on dynamics at intermediate level,
-- preferably related to Chopin, not done before

SELECT e.* FROM exercises e
LEFT JOIN student_exercises se
    ON se.exercise_id = e.id AND se.student_id = ?
WHERE e.target_dimensions LIKE '%dynamics%'
    AND e.difficulty = 'intermediate'
    AND se.id IS NULL  -- not previously assigned
ORDER BY
    CASE WHEN e.repertoire_tags LIKE '%Chopin%' THEN 0 ELSE 1 END,  -- prefer matching repertoire
    RANDOM()
LIMIT 3;
```

### Curated Exercise Seed Data

Jai seeds the database with exercises from standard methods + custom exercises. Examples:

**Dynamics exercise (intermediate):**
```json
{
    "title": "Dynamic Contrast Scales",
    "description": "Trains awareness of volume control and smooth crescendo/diminuendo over a scale passage.",
    "instructions": "Play a C major scale ascending over 2 octaves. Start at pp and arrive at ff by the top. Descend from ff back to pp. Focus on making the gradient as smooth as possible -- no sudden jumps. Repeat in the key of your current piece.",
    "target_dimensions": ["dynamics"],
    "difficulty": "intermediate",
    "category": "technique",
    "repertoire_tags": ["scales", "all-styles"],
    "source": "curated"
}
```

**Pedaling exercise (intermediate, Chopin-specific):**
```json
{
    "title": "Legato Pedal Harmonic Changes",
    "description": "Trains clean pedal changes on harmonic boundaries in romantic repertoire.",
    "instructions": "Take a 4-bar phrase from your Chopin piece where the harmony changes every bar. Play with full sustain pedal. Now: lift the pedal exactly on each new harmony and re-engage immediately. Listen for any overlap or gap. The goal is a seamless legato with zero harmonic bleed.",
    "target_dimensions": ["pedaling"],
    "difficulty": "intermediate",
    "category": "technique",
    "repertoire_tags": ["Chopin", "romantic", "pedal"]
}
```

**Voicing exercise (advanced):**
```json
{
    "title": "Melody Extraction",
    "description": "Trains the ability to project a melodic line above accompaniment.",
    "instructions": "Choose a passage where the RH has melody over LH chords. Play the melody alone at mf. Now add the LH at pp -- the melody should remain just as clear. If the LH starts to overpower, reduce it further. Record and listen back: can you hear every note of the melody clearly?",
    "target_dimensions": ["articulation", "dynamics"],
    "difficulty": "advanced",
    "category": "musicality",
    "repertoire_tags": ["all-styles", "voicing", "balance"]
}
```

### LLM-Generated Custom Exercises

When the curated DB doesn't have a close match, or when an exercise should reference the student's specific passage:

**Input to LLM:**
```
The student is working on {piece} by {composer}. Their {dimension} needs attention, specifically around {chunk_time_description}.

Generate a practice exercise that:
1. Uses their actual music (reference specific bars/passages if possible)
2. Isolates the {dimension} skill
3. Provides 3-4 concrete steps
4. Describes what "improved" sounds like

Keep it practical -- something they can do in 5 minutes right now.
Do NOT generate musical notation -- reference the score they already have.
```

**Output:** Text instructions only. The LLM does NOT generate notation (hallucination risk). It references the student's existing score: "Take bars 20-24. Play the left hand alone..."

**Storage:** LLM-generated exercises are stored in the same table with `source = "generated"`. If a student responds positively, the exercise can be promoted to curated status after human review.

### What This Slice Does NOT Include

- Notation rendering on screen (Slice 9 -- frontend)
- MIDI playback (deferred)
- Focus mode flow (Slice 8)
- Exercise effectiveness analytics

### Tasks

**Task 1: D1 migration for exercise tables**
- Create exercises and student_exercises tables
- Run migration

**Task 2: Seed curated exercises**
- Write 20-30 exercises across all 6 dimensions and 3 difficulty levels
- Cover common repertoire contexts (Chopin, Bach, Beethoven, scales/arpeggios)
- Insert via seed migration or script

**Task 3: Implement exercise query API**
- `GET /api/exercises?dimension=dynamics&level=intermediate&student_id=...`
- Filters by dimension, difficulty, not-previously-assigned
- Prefers matching repertoire tags
- Returns top 3 candidates

**Task 4: Implement LLM custom exercise generation**
- Endpoint or internal function that generates a custom exercise given piece + dimension + context
- Stores generated exercise in DB with source="generated"
- Post-processing: reject if too long (>1000 chars) or if it tries to include notation

**Task 5: Implement student-exercise tracking**
- Record when exercises are assigned, completed, skipped
- Record student response (positive/negative)
- Record dimension scores before/after (for effectiveness measurement later)

### Open Questions

1. How many curated exercises are needed for V1? 20-30 is a starting point. At what point does the LLM-generated path become more important?
2. Should exercises have difficulty progressions (exercise A leads to exercise B)?
3. Notation content: defer entirely to V2? Or include Lilypond for a few key exercises?
