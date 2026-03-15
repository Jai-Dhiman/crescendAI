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
- D1 migration numbered 0004 (0003 already exists for Score MIDI Library pieces table; 0004 is the next available slot)

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

**`GET /api/exercises?dimension=&level=&repertoire=`** -- `handle_exercises`
- Auth required, `student_id` from JWT
- All query params optional; if none provided, returns 3 random exercises
- `repertoire` is an optional string (e.g., `"Chopin"`) used for soft-preference ordering. If provided, exercises with matching `repertoire_tags` sort first. If not provided, no repertoire preference is applied (random order among matches).
- Query: JOIN `exercise_dimensions`, LEFT JOIN `student_exercises` (WHERE `se.id IS NULL`), filter by dimension/level, `ORDER BY CASE WHEN e.repertoire_tags LIKE '%' || ? || '%' THEN 0 ELSE 1 END, RANDOM()`, `LIMIT 3`
- Returns `{ exercises: Vec<Exercise> }` with dimensions populated per exercise. The `Exercise` response type intentionally omits `notation_content`, `notation_format`, and `midi_content` (unused in V1, avoids sending large blobs).
- The limit of 3 is intentional for both pipeline-generated and user-initiated exercise requests, matching the `exercise_set` component pattern (2-3 exercises per card).
- Parse query string manually (matching existing `practice/chunk` pattern)

**`POST /api/exercises/assign`** -- `handle_assign_exercise`
- Auth required, `student_id` from JWT
- Generates `id` as `se-{uuid}`
- Before INSERT, queries `SELECT MAX(times_assigned) FROM student_exercises WHERE student_id = ? AND exercise_id = ?`. If a record exists, the new record's `times_assigned` = that value + 1. If no record exists, `times_assigned` defaults to 1.
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
- Add "Try this" button at bottom of each expanded exercise. On click: calls `api.exercises.assign()`. Three states: default ("Try this"), loading (spinner), success ("Assigned", disabled). On error: show brief inline error text. Only renders when `exercise_id` is present in the exercise object (pipeline-generated exercises without DB records do not show the button).

### Types Update

File: `apps/web/src/lib/types.ts`

Add to `ExerciseSetConfig.exercises` array type:
- `hands?: "left" | "right" | "both"`
- `exercise_id?: string` (present when from DB, absent when pipeline-generated)

Add to `RichMessage`:
- `dimension?: string` -- the primary dimension this observation targets (populated by the pipeline when an observation targets a specific dimension, e.g. `"dynamics"`). Used by the "Try exercises" button to know which dimension to fetch exercises for.

### API Client

File: `apps/web/src/lib/api.ts`

Add `exercises` namespace to `api` object:
- `fetch(params?)` -- `GET /api/exercises` with optional `dimension` and `level` query params
- `assign(body)` -- `POST /api/exercises/assign`
- `complete(body)` -- `POST /api/exercises/complete`

All use the existing `request()` helper with `credentials: "include"`.

### "Try exercises for this" Action

The "Try exercises" button needs to append a new message to the chat. `ChatMessages` receives messages as a read-only prop, so:

- Add an `onTryExercises?: (dimension: string) => void` callback prop to `ChatMessages` and `MessageBubble`
- `AppChat.tsx` (which owns message state) provides the callback. When invoked, it:
  1. Calls `api.exercises.fetch({ dimension })`
  2. Transforms the API response into an `ExerciseSetConfig` (mapping DB exercises to the config shape, including `exercise_id`)
  3. Appends a new assistant `RichMessage` with the `exercise_set` component to the messages array

In `ChatMessages.tsx` `MessageBubble`, after `InlineCard` components for assistant messages:
- Small text button: "Try exercises for this" (accent color, no border)
- Only renders when `message.dimension` is present (i.e., this observation targets a specific dimension)
- On click: calls `onTryExercises(message.dimension)`
- Shows loading spinner while the fetch is in progress
- On error: shows a brief inline error message (e.g., "Could not load exercises") rather than failing silently

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

1. Apply all D1 migrations to the local database: `wrangler d1 execute crescendai-db --local --file=migrations/0001_init.sql` through `0004_exercises.sql`
2. Start `wrangler dev --local --port 8787` in background
3. Poll `/health` until ready (timeout 60s)
4. Run `bun test apps/api/tests/exercises.test.ts`
5. Kill wrangler dev, exit with test exit code

Note: `wrangler dev --local` uses Miniflare with a local SQLite database. Migrations are not auto-applied; the script must apply them explicitly before starting the dev server.

---

## Files Changed/Created

| File | Action | Description |
|------|--------|-------------|
| `apps/api/migrations/0004_exercises.sql` | Create | Tables + seed data |
| `apps/api/src/services/exercises.rs` | Create | Exercise service (3 handlers) |
| `apps/api/src/services/mod.rs` | Edit | Add `pub mod exercises;` |
| `apps/api/src/server.rs` | Edit | Add 3 route blocks |
| `apps/web/src/lib/types.ts` | Edit | Add `hands`, `exercise_id` to ExerciseSetConfig; add `dimension` to RichMessage |
| `apps/web/src/lib/api.ts` | Edit | Add `exercises` namespace |
| `apps/web/src/components/cards/ExerciseSetCard.tsx` | Edit | Hands badge, "Try this" button |
| `apps/web/src/components/ChatMessages.tsx` | Edit | "Try exercises" action button, `onTryExercises` prop |
| `apps/web/src/components/AppChat.tsx` | Edit | Provide `onTryExercises` callback (fetch + append message) |
| `apps/api/tests/exercises.test.ts` | Create | Integration tests |
| `apps/api/tests/run.sh` | Create | Test runner script |
