# Issue #29 — Exercise Routing Contract Design

**Goal:** Replace freeform string exercise generation with a single structured `ExerciseRoutingDecision` that the teacher LLM emits in both post-session synthesis and chat, and that the serving layer translates into an `ExerciseSetCard` or a text stub without polluting the `exercises` catalog table.

**Not in scope:**
- Own-passage audio playback (S2 / #45)
- Corpus drill asset selection and runtime display (S3 / #46, S4 / #47)
- Corpus breadth indexing (#17 / S5)
- RANK and difficulty filter (#42, #43 / S6)
- ASCF baseline re-lock (credit-gated; documented as follow-on)
- Any changes under `model/`

---

## Problem

The current exercise system has two separate debt items that together mean exercises are both wrong and polluting:

1. **`proposed_exercises: string[]` in `SynthesisArtifactSchema`** — a freeform text list with no machine-readable routing. The DO reads `proposed_exercises[0]`, creates a synthetic row in the `exercises` catalog via `stageDominantExercise`, then creates a second row in `exerciseDimensions`, then inserts into `pending_exercises` referencing a catalog ID. On assignment the serving layer reads back the catalog row to build `ExerciseSetPayload`. The exercises catalog exists for curated, human-authored drills; synthetic rows from session synthesis corrupt it.

2. **`create_exercise` chat tool** — also inserts into the `exercises` catalog on every tool call. It builds a freeform `exercise_set` InlineComponent immediately but the DB write is the same pollution pattern.

3. **`exercise-proposal` molecule** — dead harness code with no V6 caller; produces a V5-era `ExerciseArtifact` that the current `OnSessionEnd` loop never uses.

4. **`artifacts/exercise.ts`** — `ExerciseArtifactSchema`, `EXERCISE_TYPES`, `ACTION_REQUIRED_TYPES` are V5 schema definitions with no live callers in the V6 path.

---

## Solution (from the user's perspective)

After a practice session the teacher now emits a single structured prescription:

```json
{
  "kind": "own_passage_loop",
  "target_dimension": "pedaling",
  "bar_range": [12, 16],
  "tempo_factor": 0.75
}
```

or

```json
{
  "kind": "corpus_drill",
  "target_dimension": "timing",
  "bar_range": [1, 8],
  "tempo_factor": 0.8,
  "primitive_id": null
}
```

For `own_passage_loop`: an `ExerciseSetCard` appears in the synthesis message with a rendered score clip when a piece is identified.

For `corpus_drill`: a text stub renders ("Timing drill coming soon") where the card would be. The structure is already there for S3/S4 to populate.

In chat the teacher can call `prescribe_exercise` with the same fields plus `piece_id`. An `exercise_set` InlineComponent is returned immediately (no confirm gate). No DB rows are created in S1.

The reflect-then-prescribe gate from #27 (web `ReflectionMessage` confirm/reveal → `ExerciseSetCard`) is fully preserved — the flow, the endpoint, and the component shapes are untouched.

---

## Design

### Contract first

`ExerciseRoutingDecision` is a Zod discriminatedUnion on `kind`. All Zod constraints are validated at the schema level, not in the caller. The `pieceId` is intentionally absent from the contract — it is bound at the edge (from session `pieceCtx` in the DO; from `piece_id` tool input in chat) so the LLM never has to know the catalog UUID.

### Layered model depollution

The serving layer (`assignPendingExercise`) now reads `pending_exercises.routing_json` instead of looking up an `exercises` row. `stageDominantExercise` writes directly to `pending_exercises` with `title`, `instruction`, and `routing_json` columns (new migration). No writes to `exercises` or `exerciseDimensions` from the prescription path.

`assignExercise` / `studentExercises` tracking is untouched — when a user accepts the exercise from the confirm gate, we still record it.

### Chat path — no DB persistence in S1

The `prescribe_exercise` tool in `OnChatMessage` builds and returns an `exercise_set` InlineComponent immediately without writing to any table. Persistence (saving to `pending_exercises`) is deferred to a future slice when the chat accept flow is wired.

### Phase 2 schema migration

`proposed_exercises: string[]` is replaced by `prescribed_exercise: ExerciseRoutingDecision | null`. The prompt instruction in `buildPhase2Prompt` is updated accordingly. The validation-repair loop (MAX_PHASE2_ATTEMPTS) is kept as-is — only the prompt text and the schema field change.

### Error handling

- `prescribed_exercise: null` → no `pendingComponent` in the synthesis payload (same behavior as today when `proposed_exercises` was empty).
- `own_passage_loop` with no identified piece (`pieceCtx === null`) → `ExerciseSetPayload` with no `scoreClip`; renders as text-only in `ExerciseSetCard`. Explicit log at warn level. No crash.
- Zod parse failure in `assignPendingExercise` for `routing_json` → throws `InferenceError` with the parse message. No silent fallback.

---

## Modules

### `apps/api/src/harness/artifacts/exercise-routing.ts` (New)

**Interface:** `ExerciseRoutingDecision` type + `ExerciseRoutingDecisionSchema` Zod schema. Two variants: `OwnPassageLoopDecision`, `CorpusDrillDecision`.

**Hides:** all Zod refinement logic (bar_range start ≤ end, tempo_factor bounds, DIMS_6 enum, primitive_id constraint to corpus_drill only).

**Depth:** DEEP — single import to get the validated decision type across the entire codebase.

**Tested through:** `ExerciseRoutingDecisionSchema.parse(...)` assertions on valid and invalid payloads.

---

### `apps/api/src/harness/artifacts/synthesis.ts` (Modify)

**Interface:** `SynthesisArtifact` type — remove `proposed_exercises`, add `prescribed_exercise: ExerciseRoutingDecision | null`.

**Hides:** Zod schema refinements for weekly/piece_onboarding scopes, assigned_loops refinements.

**Depth:** DEEP — all tests go through `SynthesisArtifactSchema.parse(...)`.

---

### `apps/api/src/services/pending-exercise.ts` (Modify)

**Interface:** `stageDominantExercise(db, args)` — now accepts `routing: ExerciseRoutingDecision` instead of `proposedExercise: string`. Writes `title`, `instruction`, `routing_json` to `pending_exercises`. No writes to `exercises` or `exerciseDimensions`.

**Hides:** row insertion logic, title truncation, pending row ID extraction.

**Depth:** DEEP — callers only need to know the function exists and returns `PendingExercise`.

---

### `apps/api/src/services/exercises.ts` (Modify)

**Interface:** `assignPendingExercise(ctx, args)` — reads `pending_exercises.routing_json`, builds `ExerciseSetPayload` for `own_passage_loop` (with `scoreClip`) or `corpus_drill` (text stub). No `exercises` catalog join.

**Hides:** routing_json Zod parse, scoreClip construction logic, corpus_drill stub generation.

**Depth:** DEEP — routes result from any valid `ExerciseRoutingDecision` to the correct `ExerciseSetPayload` shape without the caller knowing the variant.

---

### `apps/api/src/services/tool-processor.ts` (Modify)

**Interface:** `TOOL_REGISTRY` — remove `create_exercise`, add `prescribe_exercise`. `getAnthropicToolSchemas()` unchanged signature.

**Hides:** Zod schemas, Anthropic JSON schemas, `process` function implementations.

**Depth:** DEEP.

---

### `apps/api/src/harness/loop/phase2.ts` (Modify)

**Interface:** `buildPhase2Prompt(digest, diagnoses, guardrail)` — same signature, updated instruction text for `prescribed_exercise` instead of `proposed_exercises[0]`.

**Hides:** prompt assembly, instruction wording.

**Depth:** DEEP.

---

### `apps/web/src/components/cards/ExerciseSetCard.tsx` (Modify)

**Interface:** `ExerciseSetCard({ config, onExpand, artifactId })` — existing signature unchanged. Gracefully renders when `config.scoreClip` is absent and exercises have no `exerciseId` (corpus_drill stub path).

**Hides:** conditional score-clip loading, corpus_drill fallback render.

**Depth:** DEEP.

---

## Verification Architecture

**Canonical success state:** `bun test --run` in `apps/api` returns green. `bun test --run` in `apps/web` returns green. `uv run pytest apps/evals/teaching_knowledge/` returns green with no KeyError on `proposed_exercises`.

**Automated check:**
```bash
cd apps/api && bun test --run
cd apps/web && bun test --run
cd apps/evals && uv run pytest teaching_knowledge/test_run_eval_atomic_gate.py -x
```

**Harness:** No separate Task Group 0 needed — the existing test suites and the eval gate are the harness. The schema contract test (Task 1) must be written first and must fail before implementation begins.

---

## File Changes

| File | Change | Type |
|------|--------|------|
| `apps/api/src/harness/artifacts/exercise-routing.ts` | New contract: discriminated union `ExerciseRoutingDecision` | New |
| `apps/api/src/harness/artifacts/exercise-routing.test.ts` | Schema validation tests | New |
| `apps/api/src/harness/artifacts/exercise.ts` | Delete — replaced by exercise-routing.ts | Delete |
| `apps/api/src/harness/artifacts/exercise.test.ts` | Delete (tests the deleted file) | Delete |
| `apps/api/src/harness/artifacts/index.ts` | Remove ExerciseArtifact exports; add ExerciseRoutingDecision export | Modify |
| `apps/api/src/harness/artifacts/synthesis.ts` | Remove proposed_exercises; add prescribed_exercise | Modify |
| `apps/api/src/harness/artifacts/synthesis.test.ts` | Update fixtures removing proposed_exercises, add prescribed_exercise tests | Modify |
| `apps/api/src/harness/loop/phase2.ts` | Update buildPhase2Prompt instruction | Modify |
| `apps/api/src/harness/loop/phase2.test.ts` | Update prompt assertion (remove proposed_exercises[0] check, add prescribed_exercise check); update VALID_ARTIFACT fixtures | Modify |
| `apps/api/src/harness/loop/phase2-schema.test.ts` | Update VALID_ARTIFACT fixture (remove proposed_exercises) | Modify |
| `apps/api/src/harness/loop/runHook.test.ts` | Update VALID_ARTIFACT fixture | Modify |
| `apps/api/src/harness/loop/compound-registry.test.ts` | Remove `create_exercise` name assertion; add `prescribe_exercise` assertion; update tool count | Modify |
| `apps/api/src/harness/skills/molecules/exercise-proposal.ts` | Delete — V5 dead code | Delete |
| `apps/api/src/harness/skills/molecules/exercise-proposal.test.ts` | Delete | Delete |
| `apps/api/src/harness/skills/molecules/index.ts` | Remove exerciseProposal from ALL_MOLECULES | Modify |
| `apps/api/src/harness/skills/molecules/index.test.ts` | Update count 9→8, remove exercise-proposal from name list | Modify |
| `apps/api/src/harness/skills/__catalog__/molecule-exercise-proposal.test.ts` | Delete | Delete |
| `apps/api/src/db/schema/exercises.ts` | Add title, instruction, routing_json columns to pendingExercises | Modify |
| `apps/api/src/services/pending-exercise.ts` | Rewrite stageDominantExercise: no exercises/exerciseDimensions inserts | Modify |
| `apps/api/src/services/pending-exercise.test.ts` | Update / add tests for new stageDominantExercise signature | Modify |
| `apps/api/src/services/exercises.ts` | Rewrite assignPendingExercise: read routing_json, no catalog join | Modify |
| `apps/api/src/services/exercises.test.ts` | Add routing_json path tests | Modify |
| `apps/api/src/services/tool-processor.ts` | Remove create_exercise; add prescribe_exercise | Modify |
| `apps/api/src/services/tool-processor.test.ts` | Replace create_exercise test with prescribe_exercise test | Modify |
| `apps/api/src/services/prompts.ts` | Update UNIFIED_TEACHER_SYSTEM tool description line (~96) | Modify |
| `apps/api/src/services/prompts.test.ts` | Update snapshot/text assertion for updated system prompt | Modify |
| `apps/api/src/do/session-brain.ts` | Replace proposed_exercises[0] staging with prescribed_exercise | Modify |
| `apps/api/src/do/session-brain.unit.test.ts` | Update ARTIFACT fixtures removing proposed_exercises | Modify |
| `apps/api/src/harness/skills/__catalog__/integration.test.ts` | Update VALID_SYNTHESIS_ARTIFACT fixture | Modify |
| `apps/api/src/services/teacher-synthesize-v6.test.ts` | Update VALID_ARTIFACT fixture | Modify |
| `apps/api/src/services/teacher.test.ts` | Update V6_VALID_ARTIFACT fixture | Modify |
| `apps/api/src/lib/types.ts` | No change (ExerciseRoutingDecision not needed on API lib/types) | — |
| `apps/web/src/lib/types.ts` | No change (ExerciseSetConfig.scoreClip already optional) | — |
| `apps/web/src/components/cards/ExerciseSetCard.tsx` | Extend: graceful corpus_drill render (text stub when no scoreClip + no exerciseId) | Modify |
| `apps/evals/teaching_knowledge/run_eval.py` | Update synthesis result rendering to use prescribed_exercise; remove proposed_exercises reference | Modify |
| Drizzle migration SQL file (new) | ADD COLUMN title, instruction, routing_json to pending_exercises | New |

---

## Open Questions

- Q: Should `stageDominantExercise` enforce that `routing_json` is present before insert?  
  Default: Yes — throw `InferenceError` if `prescribed_exercise` is null (caller must guard; DO already does `if (prescribed_exercise !== undefined)` check before calling).

- Q: Should `prescribe_exercise` in chat create a `pending_exercises` row in S1?  
  Default: No — S1 builds the InlineComponent inline only; DB persistence is S3+ work.
