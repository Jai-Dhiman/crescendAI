# Reflect-then-Prescribe Session Close Design

**Goal:** On session exit, the teacher gives a light reflection plus one directional question, with a single exercise pre-staged so that when the student confirms, it reveals instantly; if the student denies, the teacher responds adaptively (offer choices / propose an alternative / stop).

**Not in scope:**
- History / timeline / "signs of improvement" conditioning of the focus or exercise (Phase 2 — the cold-start eval baseline is blind to it).
- The retrieval-based exercise system (`model/src/exercise_corpus/`, tracked in #17). This uses the existing free-form `create_exercise` tool.
- iOS surface (web only for the interactive confirm/deny UI).
- Multi-exercise menus on the *primary* path (binary confirm on one inferred focus; alternatives are a deny-path concern).
- AMT-gated bar-level facts (ASCF lift via #15 stays deferred).

## Problem

The honest DO-path baseline (#22, locked in `apps/evals/teacher_model/stage0/tests/test_aggregator.py:_SONNET_BASELINE`) judges a single synthesis turn's prose and rewards a "do everything in one turn" teacher (composite 1.060, ASCF outcome 0.959, Concrete Artifact Provision 0.452). The product vision is the opposite: a **two-step** close — reflect lightly and ask one directional question, then prescribe a concrete exercise *after* the student confirms. Today:

- `apps/api/src/services/teacher.ts:synthesize` returns `{ text, toolResults }`. Tool calls are executed but the prose-only judge never sees them, so "Concrete Artifact Provision" is structurally low.
- There is no mechanism to pre-stage an exercise, gate it behind a confirm, or reveal it instantly.
- `apps/api/src/services/exercises.ts:assignExercise` has **no ownership link** between a generated exercise and the student it was offered to — assigning on confirm via a raw `exerciseId` is an IDOR hole.
- The web (`apps/web/src/components/AppChat.tsx`) auto-renders any `exercise_set` component in a synthesis message — there is no "hidden until confirm" affordance.

## Solution (from the user's perspective)

A student finishes a practice session. The teacher says something like: *"Your pedaling smeared the line in the running passage — want a drill for that?"* with **Confirm** / **Not now** controls. On **Confirm**, the pre-staged exercise appears immediately (no spinner, one cheap fetch). On **Not now**, the teacher replies adaptively — offering a different focus if the session/memory suggests one, or gracefully closing.

## Design

### The flow

```
session exit -> SessionBrain.runSynthesisAndPersist()
  selectPrimaryFocus(topMoments) -> focus {dimension, deviation, rationale} | null
  buildSynthesisFraming(..., focus)        // task block mandates: reflect lightly,
                                           // ask ONE directional question on focus.dimension,
                                           // emit ONE create_exercise for focus.dimension
  synthesize() -> { text, toolResults, pendingExercise }
       where pendingExercise = { exerciseId, focusDimension, previewTitle } | null
  DO persists: pending_exercises row (studentId, sessionId, exerciseId, focusDimension, previewTitle, consumed=false)
  DO persists: synthesis message with a hidden `pending_exercise` marker component
  DO WS payload: { type:"synthesis", text, components, isFallback, pendingExercise }
client renders reflection text + Confirm/Not-now (exercise hidden)
  Confirm -> POST /api/exercises/assign-pending { sessionId, exerciseId }
             -> validate studentId owns an unconsumed pending row matching (sessionId, exerciseId)
             -> assignExercise(); mark pending row consumed; return full exercise_set payload
             -> client renders exercise_set via existing Artifact chain
  Not now  -> client sends a chat turn signalling decline
             -> teacher (chat path + memory) responds adaptively
```

### Key decisions and trade-offs

- **Binary confirm on one inferred focus, not a menu.** The hard requirement is instant reveal; pre-generating every menu branch wastes 2-3x tokens, and regenerating after the answer breaks "as quick as possible." One focus drives both the question and the staged drill, so they always align. Trade-off: a wrong focus guess costs a deny round-trip — acceptable since a well-chosen worst-deviation should land most of the time.
- **`pending_exercises` linkage table for IDOR.** No student↔exercise link exists pre-assignment. A dedicated table (carrying `studentId`) is the ownership source of truth, gives a clean "deny leaves a harmless unconsumed row," and is a natural home for accept/deny analytics. Chosen over a `sessions.pending_exercise_id` column (avoids depending on the sessions schema) and over storing nothing (the IDOR hole #22 warns against).
- **Hidden `pending_exercise` marker, never `exercise_set`.** The web auto-renders `exercise_set`; carrying the staged exercise as a distinct hidden marker preserves the confirm gate and survives page reload (persisted in the synthesis message).
- **Focus selector reads the DO `topMoments` shape directly** (`{ dimension, deviation, is_positive }`), not `deriveSignals` (which reads `deviation_from_mean` and would silently read `undefined`).
- **Eval re-cut + re-lock.** The prose-only composite is retired for the synthesis turn. Reflection prose is judged on reflection-appropriate dims (Autonomy, Scaffolded, Specific Positive Praise, Tone, Style); the staged exercise is judged separately with `exercise_quality_judge_v1`. CAP + ASCF are exempted on the prose turn (they move to the artifact / are AMT-gated). User explicitly approved the re-lock.
- **Synthesis prompt changes go in the synthesis-only `buildSynthesisFraming` task block**, never the shared `UNIFIED_TEACHER_SYSTEM` (also the live-chat system prompt). `SESSION_SYNTHESIS_SYSTEM` stays dead.

## Modules

### `selectPrimaryFocus` (services)
- **Interface:** `selectPrimaryFocus(topMoments: unknown[]): { dimension: string; deviation: number; rationale: string } | null`
- **Hides:** parsing the heterogeneous `topMoments` array, filtering to genuine weaknesses (`is_positive === false`), picking the largest-magnitude negative deviation, and producing a short rationale string. Returns `null` when no weakness exists (no pre-stage).
- **Depth:** DEEP — one call hides all moment-ranking logic; the result drives both the question and the exercise.
- **Tested through:** the exported function on representative `topMoments` arrays.

### `synthesize` extension (services)
- **Interface:** `TeacherResponse` gains `pendingExercise: { exerciseId: string; focusDimension: string; previewTitle: string } | null`.
- **Hides:** extracting the first `create_exercise` result from `toolResults`, pulling `exerciseId` / `focusDimension` / `title` from its `exercise_set` config, and returning `null` cleanly when the teacher emitted no exercise.
- **Depth:** DEEP — callers get a typed pending field without knowing tool-result internals.
- **Tested through:** `synthesize()` with a stubbed Anthropic call returning a `create_exercise` tool_use.

### `assignPendingExercise` (services)
- **Interface:** `assignPendingExercise(ctx: ServiceContext, args: { studentId: string; sessionId: string; exerciseId: string }): Promise<ExerciseSetPayload>`
- **Hides:** the ownership check against `pending_exercises` (unconsumed row matching studentId+sessionId+exerciseId), the `assignExercise` call, marking the row consumed, and assembling the full `exercise_set` payload from the `exercises` + `exerciseDimensions` rows. Throws `NotFoundError` (→ 403/404) on no matching pending row.
- **Depth:** DEEP — a single call encapsulates authorization + assignment + payload assembly.
- **Tested through:** the exported function and the `/api/exercises/assign-pending` route.

### Eval dual judge re-cut (`apps/evals`)
- **Interface:** `build_do_row(...)` captures a `pending_exercise` from the `SessionResult`; a reflection-prose judge scores reflection dims; an exercise-artifact judge (`exercise_quality_judge_v1`) scores the staged drill; the aggregator emits new locked baselines.
- **Hides:** the split scoring and the new baseline math.
- **Depth:** DEEP.
- **Tested through:** `build_do_row` with injected fake `SessionResult`/judge (matching existing `test_do_row.py` pattern) and the aggregator test with the new `_SONNET_BASELINE`.

### Web reflect-then-prescribe message (`apps/web`)
- **Interface:** a synthesis `RichMessage` carrying a `pending_exercise` component renders reflection text + Confirm/Not-now; Confirm calls `api.exercises.assignPending` and appends the returned `exercise_set`; Not-now sends a decline chat turn.
- **Hides:** the confirm/deny state machine and the reveal wiring.
- **Depth:** SHALLOW-ish (a UI component) — justified: it's the user-facing surface and reuses the existing `Artifact → InlineCard → ExerciseSetCard` chain for the reveal, so it adds only gating, not new rendering.
- **Tested through:** a vitest component test (render → click Confirm → exercise appears; click Not-now → decline sent).

## Verification Architecture

- **Canonical success state:**
  1. `synthesize()` returns `pendingExercise` populated when the teacher emits a `create_exercise`, `null` otherwise.
  2. `POST /api/exercises/assign-pending` returns the full `exercise_set` payload for a valid owned pending row and **403/404** for a foreign/unknown `exerciseId`.
  3. The web reflection message reveals the exercise on Confirm and sends a decline on Not-now.
  4. The eval produces new locked dual-judge baselines (reflection-dims composite + exercise-quality score), with `_SONNET_BASELINE` updated to the new shape.
- **Automated check:**
  - API: `bun test` (vitest) over `services/*.test.ts`, `routes/exercises.test.ts`, `do/session-brain.*.test.ts`.
  - Web: `bun test` (vitest) over the new component test.
  - Eval: `uv run pytest apps/evals/teaching_knowledge/tests/test_do_row.py apps/evals/teacher_model/stage0/tests/test_aggregator.py` (fixture-driven; no live DO).
- **Harness (Task Group 0):** the eval dual-judge re-cut is buildable and testable against fixtures **before** the feature ships (it only needs a fake `SessionResult` carrying a pending exercise + synthesis text). It becomes Task Group 0.
- **Manual verification (not automatable in CI):** the actual baseline re-lock requires a live `wrangler dev` DO run over the holdout (`run_eval.py --do-path`). After the feature lands, run it once to produce the new `baseline_v2_do_aggregate.json` numbers, then update `_SONNET_BASELINE` to match. Document in `apps/evals/EVAL_CHECKLIST.md`.

## File Changes

| File | Change | Type |
|------|--------|------|
| `apps/api/src/services/focus-selector.ts` | `selectPrimaryFocus` | New |
| `apps/api/src/services/focus-selector.test.ts` | unit tests | New |
| `apps/api/src/services/prompts.ts` | `buildSynthesisFraming` accepts focus; task block mandates reflection + question + one exercise | Modify |
| `apps/api/src/services/prompts.test.ts` | framing assertions | Modify |
| `apps/api/src/services/teacher.ts` | `TeacherResponse.pendingExercise`; extract in `synthesize` | Modify |
| `apps/api/src/services/teacher.test.ts` | pendingExercise extraction | Modify |
| `apps/api/src/db/schema/exercises.ts` | `pendingExercises` table | Modify |
| `apps/api/src/services/exercises.ts` | `assignPendingExercise` | Modify |
| `apps/api/src/services/exercises.test.ts` | ownership + payload tests | New/Modify |
| `apps/api/src/routes/exercises.ts` | `POST /assign-pending` | Modify |
| `apps/api/src/routes/exercises.test.ts` | route + IDOR test | Modify |
| `apps/api/src/do/session-brain.ts` | call `selectPrimaryFocus`; persist pending row + marker; add `pendingExercise` to WS payload | Modify |
| `apps/api/src/do/session-brain.*.test.ts` | pending persistence test | Modify |
| `apps/api/migrations/*` | Drizzle migration for `pending_exercises` | New (generated) |
| `apps/web/src/lib/types.ts` | `pending_exercise` component + payload type | Modify |
| `apps/web/src/lib/api.ts` | `exercises.assignPending` | Modify |
| `apps/web/src/components/ReflectionMessage.tsx` | confirm/deny + reveal | New |
| `apps/web/src/components/ReflectionMessage.test.tsx` | component test | New |
| `apps/web/src/components/AppChat.tsx` | route synthesis messages with pending_exercise to ReflectionMessage | Modify |
| `apps/evals/teaching_knowledge/run_eval.py` | capture `pending_exercise`; dual-judge wiring | Modify |
| `apps/evals/shared/pipeline_client.py` | `SynthesisResult.pending_exercise` field | Modify |
| `apps/evals/teaching_knowledge/tests/test_do_row.py` | dual-judge row tests | Modify |
| `apps/evals/teacher_model/stage0/aggregator.py` | reflection-dims composite + exercise-quality | Modify |
| `apps/evals/teacher_model/stage0/tests/test_aggregator.py` | new `_SONNET_BASELINE` | Modify |
| `apps/evals/EVAL_CHECKLIST.md` | re-lock runbook | Modify |

## Open Questions

- **Q: Decline transport from web — structured signal or a plain chat message?**
  Default: send a normal chat turn with a fixed decline string (e.g. "Not right now — something else?") plus the focus dimension as context; the teacher's adaptivity is a prompt behavior, no new endpoint.
- **Q: Does the synthesis WS `pendingExercise` field need a `messages` schema column, or is the hidden `pending_exercise` component in `componentsJson` enough for reload fidelity?**
  Default: reuse `componentsJson` (hidden marker) — no `messages` migration; the `pending_exercises` table is the validation source of truth.
- **Q: Cold-start vs returning students.**
  Default (resolved in design): both from day one; focus = worst within-session deviation for cold-start, identical logic for returning; improvement-aware conditioning is Phase 2.
