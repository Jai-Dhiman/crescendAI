# Exercises and Focused Practice

> **Status (2026-03-14):** NOT STARTED. Exercise DB tables defined in architecture.md but not migrated. No exercise data seeded. No endpoint implemented. Focus mode depends on exercise DB and teaching moment detection (02-pipeline.md).

## Why Exercises Matter

The core feedback loop in CrescendAI is: observe weakness, prescribe exercise, evaluate improvement, close the loop. Without exercises, the system can only describe problems. With exercises, it can fix them.

Today, the teacher pipeline (02-pipeline.md) detects when something sounds off and explains what to work on. But "your pedaling is blurring the harmonic changes" is only half the job. The other half is "here's a 5-minute exercise to fix it, and I'll listen to tell you if it worked."

Exercises connect two otherwise separate capabilities:

1. **The exercise database** provides the content -- curated method exercises (Hanon, Czerny), repertoire-specific drills, and LLM-generated custom exercises grounded in the student's actual music.
2. **Focus mode** provides the interaction pattern -- a guided mini-session that isolates a weak dimension, walks through targeted exercises, evaluates each attempt, and measures improvement.

The 6 target dimensions are defined in `model/02-teacher-grounded-taxonomy.md`: dynamics, timing, pedaling, articulation, phrasing, interpretation. Every exercise targets one or more of these dimensions.

---

## Exercise Database

### Schema

Two D1 tables store exercises and student-exercise tracking.

**`exercises` table:**

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT PK | Unique exercise identifier |
| `title` | TEXT NOT NULL | Exercise name |
| `description` | TEXT NOT NULL | What this exercise trains, why it matters |
| `instructions` | TEXT NOT NULL | Step-by-step how to practice it |
| `difficulty` | TEXT NOT NULL | `beginner` / `intermediate` / `advanced` |
| `category` | TEXT NOT NULL | `technique` / `musicality` / `ear-training` / `warmup` |
| `repertoire_tags` | TEXT | JSON array, e.g. `["Chopin", "nocturne", "romantic"]` |
| `notation_content` | TEXT | MusicXML or Lilypond source (nullable for text-only exercises) |
| `notation_format` | TEXT | `musicxml` / `lilypond` / null |
| `midi_content` | BLOB | MIDI bytes for playback (nullable) |
| `source` | TEXT | `curated` / `method:Hanon` / `method:Czerny` / `generated` |
| `variants_json` | TEXT | JSON: alternate versions (different keys, tempos, dynamics) |
| `created_at` | TEXT NOT NULL | ISO 8601 timestamp |

Index: `idx_exercises_difficulty` on `difficulty`.

**`exercise_dimensions` junction table:**

| Column | Type | Description |
|--------|------|-------------|
| `exercise_id` | TEXT NOT NULL | FK to `exercises(id)` |
| `dimension` | TEXT NOT NULL | One of: dynamics, timing, pedaling, articulation, phrasing, interpretation |

Primary key: `(exercise_id, dimension)`.
Index: `idx_exercise_dimensions_dim` on `dimension`.

```sql
CREATE TABLE exercise_dimensions (
    exercise_id TEXT NOT NULL REFERENCES exercises(id),
    dimension TEXT NOT NULL,
    PRIMARY KEY (exercise_id, dimension)
);
CREATE INDEX idx_exercise_dimensions_dim ON exercise_dimensions(dimension);
```

### Exercise Properties

Each exercise carries enough metadata to support filtering and pedagogical context:

- **target dimensions** -- stored in the `exercise_dimensions` junction table. Most exercises target 1-2 dimensions.
- **difficulty** -- coarse difficulty level. Maps roughly to student proficiency, not piece difficulty.
- **category** -- the type of practice activity. `technique` covers mechanical drills (scales, arpeggios, Hanon). `musicality` covers expressive work (voicing, phrasing). `ear-training` covers listening exercises. `warmup` covers session openers.
- **notation** -- optional MusicXML or Lilypond source. Many exercises are text-only (especially LLM-generated ones). Notation rendering is handled by `05-ui-system.md`.
- **variants** -- JSON describing alternate versions. A dynamics exercise might have variants in different keys or at different tempos.
- **source** -- provenance. `curated` = human-authored by Jai. `method:Hanon` = adapted from a published method. `generated` = created by the LLM for a specific student context.

### Curated Seed Data

V1 targets 20-30 curated exercises across all 6 dimensions and 3 difficulty levels, covering common repertoire contexts. Examples:

| Title | Dimension(s) | Difficulty | Category | Source |
|-------|-------------|------------|----------|--------|
| Dynamic Contrast Scales | dynamics | intermediate | technique | curated |
| Legato Pedal Harmonic Changes | pedaling | intermediate | technique | curated |
| Melody Extraction | articulation, dynamics | advanced | musicality | curated |

**Dynamic Contrast Scales (intermediate, dynamics):**
> Play a C major scale ascending over 2 octaves. Start at pp and arrive at ff by the top. Descend from ff back to pp. Focus on making the gradient as smooth as possible -- no sudden jumps. Repeat in the key of your current piece.

**Legato Pedal Harmonic Changes (intermediate, pedaling, Chopin-specific):**
> Take a 4-bar phrase from your Chopin piece where the harmony changes every bar. Play with full sustain pedal. Now: lift the pedal exactly on each new harmony and re-engage immediately. Listen for any overlap or gap. The goal is a seamless legato with zero harmonic bleed.

**Melody Extraction (advanced, articulation + dynamics):**
> Choose a passage where the RH has melody over LH chords. Play the melody alone at mf. Now add the LH at pp -- the melody should remain just as clear. If the LH starts to overpower, reduce it further. Record and listen back: can you hear every note of the melody clearly?

### LLM-Generated Custom Exercises

When the curated DB has no close match, or when an exercise should reference the student's specific passage, the LLM generates a custom exercise.

**Constraints:**
- Text instructions only. The LLM does NOT generate notation (hallucination risk for musical notation is too high).
- Instructions reference the student's existing score: "Take bars 20-24. Play the left hand alone..."
- Post-processing rejects exercises that exceed 1000 characters or attempt to include notation.

**LLM prompt template:**

```
The student is working on {piece} by {composer}. Their {dimension} needs attention,
specifically around {chunk_time_description}.

Generate a practice exercise that:
1. Uses their actual music (reference specific bars/passages if possible)
2. Isolates the {dimension} skill
3. Provides 3-4 concrete steps
4. Describes what "improved" sounds like

Keep it practical -- something they can do in 5 minutes right now.
Do NOT generate musical notation -- reference the score they already have.
```

**Storage:** Generated exercises go into the same `exercises` table with `source = "generated"`. If a student responds positively, the exercise can be promoted to curated status after human review.

### Query Logic

Exercise selection filters by dimension, difficulty, and novelty (not previously assigned to this student), with repertoire-matching as a soft preference:

```sql
SELECT e.* FROM exercises e
JOIN exercise_dimensions ed ON ed.exercise_id = e.id
LEFT JOIN student_exercises se
    ON se.exercise_id = e.id AND se.student_id = ?
WHERE ed.dimension = 'dynamics'
    AND e.difficulty = 'intermediate'
    AND se.id IS NULL  -- not previously assigned
ORDER BY
    CASE WHEN e.repertoire_tags LIKE '%Chopin%' THEN 0 ELSE 1 END,
    RANDOM()
LIMIT 3;
```

The API exposes this as `GET /api/exercises?dimension=dynamics&level=intermediate&student_id=...`, returning the top 3 candidates. On iOS, SwiftData mirrors the same filter logic locally, querying cached exercises first and falling back to the API if the cache is stale. New exercises reach iOS devices via the sync protocol (`POST /api/sync` response includes `exerciseUpdates`).

---

## Focus Mode (Guided Practice)

Focus mode transforms the app from a passive listener into an active teacher. It is the teaching loop: observe, identify, diagnose, prescribe, evaluate.

### Entry Points

**System-initiated** -- after the STOP classifier (02-pipeline.md) identifies a dimension as the top teaching moment in 3+ of the last 5 sessions:

> "I've noticed pedaling keeps coming up. Want to do a focused session on it next time?"

The student can accept, dismiss, or say "later."

**Student-initiated** -- the student says or taps "I want to work on my dynamics" or "Let's focus on pedaling." The system enters focus mode for that dimension immediately.

### Session Flow

A focus session has 5 stages:

1. **Introduction.** The teacher LLM sets context:
   > "Let's work on your pedaling. I've been hearing some harmonic bleed in your Chopin -- the sustain is carrying across chord changes. We'll do 3 exercises to sharpen that."

2. **Curated exercise.** Query the exercise DB for a matching curated exercise (target dimension, student level, repertoire match). Present the title and instructions. Student plays. MuQ evaluates the attempt, focused on the target dimension. Teacher gives brief feedback:
   > "Better -- the chord change at the top was cleaner. Try it once more and really listen for the bass note ringing over."

3. **Custom exercise.** The LLM generates an exercise using the student's actual passage where the issue was detected:
   > "Now take bars 20-24 of your Nocturne. Play just the LH with pedal. Lift and re-engage on each new harmony. Then add the RH and keep the same pedal discipline."

   Student plays. MuQ evaluates. Teacher feedback.

4. **Integration.** Student plays the full passage with everything from the previous exercises applied. MuQ evaluates. Compare the target dimension score to the original teaching moment that triggered focus mode:
   > "That's markedly cleaner. The harmonic changes in bars 22-23 aren't bleeding anymore. Keep this awareness next time you run through the whole piece."

5. **Wrap-up.** Record before/after scores in `student_exercises`. Update the student model: "Focused on pedaling in session N." Suggest next steps:
   > "Try running through the piece now and I'll listen for whether the pedaling holds in context."

The student can skip an exercise or end the session early at any point.

### Focused Evaluation

MuQ inference runs via the cloud HF endpoint on each exercise attempt, using the same inference path as regular practice. The key difference is how feedback is delivered:

- **Weight to target dimension only.** If the student is working on pedaling, suppress observations about timing even if timing is worse than usual.
- **Non-target exception.** Only surface a non-target observation if STOP probability exceeds 0.95 on that dimension -- something is severely off.
- **Before/after comparison.** Each exercise attempt records the target dimension score. The wrap-up computes overall improvement.

This keeps the session focused and avoids overwhelming the student with unrelated feedback.

### Session State Machine

```json
{
    "mode": "focus",
    "target_dimension": "pedaling",
    "trigger": "system",
    "trigger_context": "Pedaling was the top teaching moment in 3 of last 5 sessions",
    "exercises": [
        {
            "type": "curated",
            "exercise_id": "ex-ped-003",
            "status": "completed",
            "attempts": 2,
            "target_dim_before": 0.35,
            "target_dim_after": 0.52
        },
        {
            "type": "generated",
            "exercise_id": "ex-gen-001",
            "status": "in_progress",
            "attempts": 1
        },
        {
            "type": "integration",
            "status": "pending"
        }
    ],
    "overall_improvement": null
}
```

States per exercise: `pending` -> `in_progress` -> `completed` (or `skipped`). The session itself progresses linearly through the exercise list, with the student able to skip forward or exit.

---

## Student Tracking

### `student_exercises` Table

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT PK | Unique record identifier |
| `student_id` | TEXT NOT NULL | FK to `students` |
| `exercise_id` | TEXT NOT NULL | FK to `exercises` |
| `session_id` | TEXT | FK to `sessions` (nullable for standalone assignments) |
| `assigned_at` | TEXT NOT NULL | ISO 8601 timestamp |
| `completed` | BOOLEAN | Default FALSE |
| `response` | TEXT | `positive` / `neutral` / `negative` / `skipped` |
| `dimension_before_json` | TEXT | JSON: 6-dim scores before exercise |
| `dimension_after_json` | TEXT | JSON: 6-dim scores after exercise |
| `notes` | TEXT | Student feedback on the exercise |
| `times_assigned` | INTEGER | Default 1 |

Unique constraint: `(student_id, exercise_id, session_id)`.

Index: `idx_student_exercises` on `(student_id, exercise_id)`.

### How Exercise History Feeds the Subagent

The two-stage subagent pipeline (02-pipeline.md) uses exercise history as context when selecting teaching moments and generating observations:

- **Exercise effectiveness:** If a student completed an exercise targeting pedaling and their before/after scores improved, the subagent knows that dimension is responding to intervention. It may raise the bar for flagging pedaling in future sessions.
- **Negative responses:** If a student marked an exercise as `negative` or `skipped` it, the subagent avoids prescribing similar exercises and may try a different approach to the same dimension.
- **Repeat assignments:** The `times_assigned` counter and query logic (exclude previously-assigned exercises) prevent the system from recycling the same exercises.

---

## Dependencies

| Dependency | Location | Relationship |
|------------|----------|-------------|
| STOP classifier and teaching moment selection | `02-pipeline.md` | STOP triggers focus mode when a dimension persists across sessions |
| Exercise UI components (`exercise_set`, notation rendering) | `05-ui-system.md` | Frontend rendering of exercise cards and notation |
| 6-dimension taxonomy | `model/02-teacher-grounded-taxonomy.md` | All exercises and evaluations use these 6 dimensions |
| D1 sync protocol | `docs/architecture.md` (Sync section) | Exercise data syncs to iOS via `POST /api/sync` |
| MuQ cloud inference | `02-pipeline.md` | Focus mode evaluation uses the same HF endpoint inference path |

---

## Open Questions

1. **Curated exercise count for V1.** 20-30 is the starting target. At what point does the LLM-generated path become more important than expanding the curated set?
2. **Difficulty progressions.** Should exercises link to successor exercises (exercise A leads to exercise B)? Useful for building skills incrementally, but adds schema complexity.
3. **Notation content.** Defer entirely to V2, or include Lilypond for a few key exercises in V1?
4. **Exercises per focus session.** 3 (curated + custom + integration) feels right but may be too many for a student who has already been practicing for 45 minutes. Should the system adapt based on session length?
5. **Focus mode timing.** Should focus mode interrupt a regular practice session mid-stream, or always be a separate mini-session?
6. **No improvement path.** What happens if the student doesn't improve during focus mode? Encouraging message + suggest trying again tomorrow? Adapt exercises on the fly? Lower expectations?
7. **Multi-session focus plans.** "Work on pedaling for the next 3 sessions" is not in scope for V1. When does it become worth building?
