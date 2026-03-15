# Exercise System Design

Exercise database, API endpoints, seed data, and web UI integration for the CrescendAI feedback loop: observe weakness, prescribe exercise, evaluate improvement.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Student tracking model | One record per (student, exercise, session) triple | Preserves per-session before/after scores while supporting exclusion filter. `times_assigned` denormalized across sessions. |
| Exercise ID format | Prefixed human-readable: `ex-dyn-001` (curated), `ex-gen-{uuid}` (generated) | Clear provenance from the ID, readable seed SQL and logs. |
| Auth on exercise endpoints | All three endpoints require auth, `student_id` from JWT | Consistent with codebase, prevents information leakage about other students' history. No `student_id` query param needed. |
| "Try exercises" in chat | Appends a new assistant message with `exercise_set` component | Keeps message model immutable. Feels like the teacher responding to the request. |
| Scope | Full CRUD: GET + assign + complete | Complete endpoint is ~30 lines of Rust, having it ready means focus mode can use it immediately. |

## Scope Boundaries

- Working in `apps/api/` (Rust, D1 migrations) and `apps/web/src/` (React)
- NOT touching `model/`, `usePracticeSession.ts`, `ListeningMode.tsx`, `stop.rs`, `teaching_moments.rs`
- D1 migration numbered 0004 (0003 reserved for Score MIDI Library)

---

## 1. D1 Migration (0004_exercises.sql)

File: `apps/api/migrations/0004_exercises.sql`

### Tables

**`exercises`** -- the exercise catalog:

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT PK | `ex-{dim}-{nnn}` curated, `ex-gen-{uuid}` generated |
| `title` | TEXT NOT NULL | Exercise name |
| `description` | TEXT NOT NULL | What this exercise trains, why it matters |
| `instructions` | TEXT NOT NULL | Step-by-step how to practice it |
| `difficulty` | TEXT NOT NULL | `beginner` / `intermediate` / `advanced` |
| `category` | TEXT NOT NULL | `technique` / `musicality` / `ear-training` / `warmup` |
| `repertoire_tags` | TEXT | JSON array, e.g. `["Chopin", "nocturne"]` |
| `notation_content` | TEXT | MusicXML or Lilypond (nullable, unused V1) |
| `notation_format` | TEXT | `musicxml` / `lilypond` / null |
| `midi_content` | BLOB | MIDI bytes (nullable, unused V1) |
| `source` | TEXT NOT NULL | `curated` / `method:Hanon` / `generated` |
| `variants_json` | TEXT | JSON alternate versions |
| `created_at` | TEXT NOT NULL | ISO 8601 timestamp |

**`exercise_dimensions`** -- junction table:

| Column | Type |
|--------|------|
| `exercise_id` | TEXT NOT NULL REFERENCES exercises(id) |
| `dimension` | TEXT NOT NULL |

Primary key: `(exercise_id, dimension)`.

**`student_exercises`** -- tracking:

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT PK | `se-{uuid}` |
| `student_id` | TEXT NOT NULL | FK to students |
| `exercise_id` | TEXT NOT NULL | FK to exercises |
| `session_id` | TEXT | FK to sessions (nullable) |
| `assigned_at` | TEXT NOT NULL | ISO 8601 |
| `completed` | BOOLEAN DEFAULT 0 | |
| `response` | TEXT | `positive` / `neutral` / `negative` / `skipped` |
| `dimension_before_json` | TEXT | JSON 6-dim scores before |
| `dimension_after_json` | TEXT | JSON 6-dim scores after |
| `notes` | TEXT | Student feedback |
| `times_assigned` | INTEGER DEFAULT 1 | Denormalized count across sessions |

Unique constraint: `(student_id, exercise_id, session_id)`.

### Indexes

- `idx_exercises_difficulty` on `exercises(difficulty)`
- `idx_exercise_dimensions_dim` on `exercise_dimensions(dimension)`
- `idx_student_exercises` on `student_exercises(student_id, exercise_id)`

### Seed Data

25 curated exercises in the migration. Coverage:

| Dimension | Count | Difficulties |
|-----------|-------|-------------|
| dynamics | 5 | beginner, intermediate, advanced |
| timing | 4 | beginner, intermediate, advanced |
| pedaling | 4 | intermediate, advanced |
| articulation | 4 | beginner, intermediate, advanced |
| phrasing | 4 | intermediate, advanced |
| interpretation | 4 | intermediate, advanced |

Categories: ~12 technique, ~6 musicality, ~3 ear-training, ~4 warmup.

Includes the 3 spec examples from `04-exercises.md`:
- "Dynamic Contrast Scales" (dynamics, intermediate, technique)
- "Legato Pedal Harmonic Changes" (pedaling, intermediate, technique)
- "Melody Extraction" (articulation + dynamics, advanced, musicality)

Plus repertoire-tagged exercises for Bach (articulation, phrasing), Chopin (pedaling, dynamics, phrasing), Beethoven (dynamics, articulation), Mozart (articulation, timing), Debussy (pedaling, phrasing).

Plus 3-4 warmup exercises (scales, arpeggios, chord progressions).

All seed data: `source = 'curated'`. Each exercise INSERT followed by corresponding `exercise_dimensions` INSERTs.

---

## 2. Rust Exercise Service

File: `apps/api/src/services/exercises.rs`
Module registration: add `pub mod exercises;` to `services/mod.rs`.

### Types

```
Exercise { id, title, description, instructions, difficulty, category,
           repertoire_tags, source, dimensions: Vec<String> }

StudentExercise { id, student_id, exercise_id, session_id, assigned_at,
                  completed, response, times_assigned }

AssignRequest { exercise_id: String, session_id: Option<String> }

CompleteRequest { student_exercise_id: String, response: Option<String>,
                  dimension_before_json: Option<String>,
                  dimension_after_json: Option<String>, notes: Option<String> }
```

### Endpoints

**`GET /api/exercises?dimension=&level=`** -- `handle_exercises`
- Auth required, `student_id` from JWT
- Both query params optional; if neither, returns 3 random exercises
- Query: JOIN `exercise_dimensions`, LEFT JOIN `student_exercises` (WHERE `se.id IS NULL`), filter by dimension/level, soft-prefer repertoire match via `ORDER BY CASE`, `LIMIT 3`
- Returns `{ exercises: Vec<Exercise> }` with dimensions populated per exercise
- Parse query string manually (matching existing `practice/chunk` pattern)

**`POST /api/exercises/assign`** -- `handle_assign_exercise`
- Auth required, `student_id` from JWT
- Generates `id` as `se-{uuid}`
- Checks if (student, exercise) pair exists across any session; if yes, increments `times_assigned` on new record
- INSERT into `student_exercises`, returns created record

**`POST /api/exercises/complete`** -- `handle_complete_exercise`
- Auth required, `student_id` from JWT
- Validates `student_exercise_id` belongs to this student
- UPDATE `student_exercises` SET `completed = 1` + optional fields
- Returns updated record

### Routing (server.rs)

Three new `if` blocks before the health check:
- `path == "/api/exercises" && method == GET` -- query string, no body
- `path == "/api/exercises/assign" && method == POST` -- body
- `path == "/api/exercises/complete" && method == POST` -- body

---

## 3. Web UI

### ExerciseSetCard Enhancement

File: `apps/web/src/components/cards/ExerciseSetCard.tsx` (existing, modify)

Changes:
- Add `hands` indicator badge ("LH" / "RH" / "Both") next to dimension badge, only when `hands` is present
- Add "Try this" button at bottom of each expanded exercise. On click: calls `api.exercises.assign()`. Transitions to "Assigned" (disabled, checkmark) on success. Only renders when `exercise_id` is present in the exercise object.

### Types Update

File: `apps/web/src/lib/types.ts`

Add to `ExerciseSetConfig.exercises` array type:
- `hands?: "left" | "right" | "both"`
- `exercise_id?: string` (present when from DB, absent when pipeline-generated)

### API Client

File: `apps/web/src/lib/api.ts`

Add `exercises` namespace to `api` object:
- `fetch(params?)` -- `GET /api/exercises` with optional `dimension` and `level` query params
- `assign(body)` -- `POST /api/exercises/assign`
- `complete(body)` -- `POST /api/exercises/complete`

All use the existing `request()` helper with `credentials: "include"`.

### "Try exercises for this" Action

In `ChatMessages.tsx` `MessageBubble`, after `InlineCard` components for assistant messages:
- Small text button: "Try exercises for this" (accent color, no border)
- Only shows when message has a `dimension` field in metadata
- On click: calls `api.exercises.fetch({ dimension })`, transforms response into `ExerciseSetConfig`, appends a new assistant `RichMessage` with the `exercise_set` component

The `dimension` field comes from an optional `dimension` property on the `RichMessage` type (populated by the pipeline when an observation targets a specific dimension).

---

## 4. Testing

### Rust Unit Tests

In `apps/api/src/services/exercises.rs` (`#[cfg(test)]` module):
- Query param parsing
- Struct serialization/deserialization
- AssignRequest/CompleteRequest validation

### Bun Integration Tests

File: `apps/api/tests/exercises.test.ts`

Uses `fetch` against `http://localhost:8787`. Auth via `/api/auth/debug` (dev-only endpoint that returns JWT cookie).

Tests:
1. `GET /api/exercises` returns exercises
2. `GET /api/exercises?dimension=dynamics` returns only dynamics exercises
3. `GET /api/exercises?level=beginner` returns only beginner exercises
4. `GET /api/exercises?dimension=dynamics&level=intermediate` combined filter
5. `POST /api/exercises/assign` assigns an exercise, returns record
6. `GET /api/exercises` after assign -- assigned exercise excluded
7. `POST /api/exercises/complete` marks complete, returns updated record
8. `POST /api/exercises/complete` with wrong student returns 403/404

### Test Runner

File: `apps/api/tests/run.sh`

1. Start `wrangler dev --local --port 8787` in background
2. Poll `/health` until ready (timeout 60s)
3. Run `bun test apps/api/tests/exercises.test.ts`
4. Kill wrangler dev, exit with test exit code

---

## Files Changed/Created

| File | Action | Description |
|------|--------|-------------|
| `apps/api/migrations/0004_exercises.sql` | Create | Tables + seed data |
| `apps/api/src/services/exercises.rs` | Create | Exercise service (3 handlers) |
| `apps/api/src/services/mod.rs` | Edit | Add `pub mod exercises;` |
| `apps/api/src/server.rs` | Edit | Add 3 route blocks |
| `apps/web/src/lib/types.ts` | Edit | Add `hands`, `exercise_id` to ExerciseSetConfig |
| `apps/web/src/lib/api.ts` | Edit | Add `exercises` namespace |
| `apps/web/src/components/cards/ExerciseSetCard.tsx` | Edit | Hands badge, "Try this" button |
| `apps/web/src/components/ChatMessages.tsx` | Edit | "Try exercises" action button |
| `apps/api/tests/exercises.test.ts` | Create | Integration tests |
| `apps/api/tests/run.sh` | Create | Test runner script |
