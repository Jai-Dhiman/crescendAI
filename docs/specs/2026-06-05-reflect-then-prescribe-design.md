# Reflect-then-Prescribe Session Close Design

**Goal:** On session exit, the production (V6) teacher gives a light reflection plus one directional question, with a single exercise pre-staged so that when the student confirms, it reveals instantly; if the student denies, the teacher responds adaptively.

**Not in scope:**
- History / timeline / "signs of improvement" conditioning (Phase 2).
- The retrieval-based exercise system (`model/src/exercise_corpus/`, #17). Staging reuses the artifact's own `proposed_exercises`.
- iOS surface (web only for the interactive UI).
- Multi-exercise menus on the primary path (binary confirm on one focus).
- AMT-gated bar-level facts (#15 stays deferred).
- **Automated eval measurement of this feature.** The eval harness bypasses V6 (runs the legacy `synthesize` path); it cannot measure the production V6 behavior. Re-locking the `_SONNET_BASELINE` and making the eval V6-aware is a deferred follow-up (see Verification).

## Problem

Production runs `HARNESS_V6_ENABLED = "true"` (`apps/api/wrangler.toml:28`), so every non-eval session is synthesized by the **V6 harness**: `synthesizeV6` (`teacher.ts:746`) → `runHook("OnSessionEnd")` → a 2-phase agentic compound that emits a `SynthesisArtifact` (`harness/artifacts/synthesis.ts`). The legacy `synthesize`/`buildSynthesisFraming` path runs **only for eval sessions** (`session-brain.ts:1631-1634`). Today the V6 close:

- Produces a `headline` (300-500 char paragraph) shown as the synthesis text — not a light reflection ending in a directional question.
- Produces `proposed_exercises` as **plain strings** that are never persisted and never rendered as actionable artifacts.
- Has no pre-staged, persisted exercise (no `exerciseId`), no confirm/deny gate.

There is also no ownership link between a generated exercise and the student it was offered to — assigning on confirm via a raw `exerciseId` would be an IDOR hole.

## Solution (from the user's perspective)

A student finishes a session. The teacher's close reads as a light reflection ending in one directional question (*"...want a drill for that?"*) with **Confirm** / **Not now** controls. On **Confirm**, the pre-staged exercise appears immediately (one cheap fetch). On **Not now**, the teacher replies adaptively.

## Design

### The flow (V6 production path)

```
session exit -> SessionBrain.runSynthesisAndPersist() [V6 branch]
  synthesizeV6 -> runHook("OnSessionEnd")
       Phase 1: diagnosis molecules + assign_segment_loop (unchanged)
       Phase 2: write_synthesis_artifact -> headline = LIGHT reflection + ONE directional
                question about dominant_dimension (Phase-2 prompt change)
  artifact returned to the DO V6 block
  IF artifact.proposed_exercises.length > 0:
     stageDominantExercise(db, { studentId, sessionId, dominantDimension,
                                 proposedExercise: proposed_exercises[0], pieceMetadata })
       -> persists ONE exercises row (returns exerciseId)
       -> persists ONE pending_exercises row (studentId, sessionId, exerciseId, focusDimension, previewTitle, consumed=false)
       -> returns { exerciseId, focusDimension, previewTitle }
     pendingComponent = buildPendingExerciseComponent(staged)   // { type:"pending_exercise", config:{...} }
  buildV6WsPayload(artifact, loopComponents, pendingComponent)   // adds the hidden marker
  WS -> { type:"synthesis", text: headline, components:[...loops, pending_exercise], isFallback:false }
  persistSynthesisMessage(... components incl. pending_exercise)  // reload fidelity
client renders headline + Confirm/Not-now (pending_exercise NOT auto-rendered as an exercise)
  Confirm -> POST /api/exercises/assign-pending { sessionId, exerciseId }
             -> validate studentId owns an unconsumed pending row matching (sessionId, exerciseId)
             -> assignExercise(); mark consumed; return full exercise_set payload -> reveal
  Not now  -> client sends a decline chat turn -> teacher (chat path + memory) adapts
```

### Key decisions and trade-offs

- **Deterministic staging in the DO, copying `assigned_loops`.** The artifact already produces `proposed_exercises` + `dominant_dimension`; staging one persisted exercise from them after Phase 2 needs no new LLM tool and no artifact-schema migration — a direct parallel to the live `assigned_loops → loopComponents` path. Trade-off: the staged exercise's text is the LLM's `proposed_exercises[0]` (good enough; not a bespoke per-bar drill — that's the AMT/#17 future).
- **Only the Phase-2 prompt is LLM-facing.** Headline becomes a light reflection + one directional question on `dominant_dimension`, within the existing 300-500 char budget (≈2-4 sentences — acceptably light; not lowering the schema min, which other code depends on).
- **Binary confirm on `dominant_dimension`.** Instant reveal; the question and the staged drill always align. Deny is the rare path.
- **`pending_exercises` linkage table for IDOR.** No student↔exercise link exists pre-assignment; a dedicated table (carrying `studentId`) is the ownership source of truth and a clean deny semantic.
- **Hidden `pending_exercise` component, never `exercise_set`.** The web auto-renders `exercise_set`; the staged exercise rides as a distinct hidden marker so the confirm gate holds and reload re-shows it.
- **Eval is not the measurement here.** Eval bypasses V6. Verification is V6 harness unit tests + web component tests + manual in-app check; a follow-up issue should make the eval V6-aware so future teacher-quality work is automatable again.

## Modules

### Phase-2 prompt builder (`apps/api/src/harness/loop/phase2.ts`)
- **Interface:** extract the inline Phase-2 user-prompt assembly into a pure `buildPhase2Prompt(digest, diagnoses, guardrail): string`.
- **Hides:** the reflection+directional-question instruction and the dominant-dimension focus directive.
- **Depth:** DEEP-enough (pure, isolates the only LLM-facing change).
- **Tested through:** the exported pure function (asserts the instruction text + dominant-focus directive present; guardrail still honored).

### `stageDominantExercise` (`apps/api/src/services/pending-exercise.ts`)
- **Interface:** `stageDominantExercise(db, args: { studentId; sessionId; dominantDimension; proposedExercise; pieceMetadata }): Promise<{ exerciseId; focusDimension; previewTitle }>`
- **Hides:** persisting the `exercises` row + `exerciseDimensions` + `pending_exercises` row, deriving title/instruction, returning the staging ref.
- **Depth:** DEEP.
- **Tested through:** the exported function with a mocked `db`.

### `buildPendingExerciseComponent` + `buildV6WsPayload` extension (services / `session-brain.ts`)
- **Interface:** `buildPendingExerciseComponent(staged): InlineComponent`; `buildV6WsPayload(artifact, loopComponents?, pendingComponent?)`.
- **Hides:** the hidden-marker shape and its inclusion in the WS payload + persisted message.
- **Depth:** SHALLOW but justified (pure mappers mirroring `loopComponents`).
- **Tested through:** the exported pure functions.

### `assignPendingExercise` + `/assign-pending` (`exercises.ts`, `routes/exercises.ts`)
- **Interface:** `assignPendingExercise(ctx, { studentId; sessionId; exerciseId }): Promise<ExerciseSetPayload>`; `POST /assign-pending`.
- **Hides:** ownership check against `pending_exercises`, `assignExercise`, marking consumed, payload assembly. Throws `NotFoundError` → 404 on no owned unconsumed row (IDOR guard).
- **Depth:** DEEP.
- **Tested through:** the exported function (mocked db) and the Hono route (`testApp.request`).

### Web reflect-then-prescribe (`apps/web`)
- **Interface:** a synthesis `RichMessage` carrying a `pending_exercise` component renders headline + Confirm/Not-now; Confirm calls `api.exercises.assignPending` and reveals the `exercise_set`; Not-now sends a decline chat turn.
- **Depth:** SHALLOW (UI) — justified: reuses the `Artifact → InlineCard → ExerciseSetCard` chain for the reveal.
- **Tested through:** vitest component test.

## Verification Architecture

- **Canonical success state:**
  1. `buildPhase2Prompt` output instructs a light reflection + one directional question on `dominant_dimension`.
  2. `stageDominantExercise` persists exactly one `exercises` row + one `pending_exercises` row and returns the ref; the V6 block emits a `pending_exercise` component only when `proposed_exercises` is non-empty.
  3. `POST /api/exercises/assign-pending` returns the full `exercise_set` payload for a valid owned pending row, **404** for a foreign/unknown `exerciseId`, **401** unauthenticated.
  4. Web: Confirm reveals the exercise; Not-now sends a decline.
- **Automated check:**
  - API: `cd apps/api && bunx vitest run` over the new service/route/DO/harness tests; `bun run typecheck`.
  - Web: `cd apps/web && bunx vitest run` over the new component tests; `bunx tsc --noEmit`.
- **Harness (Task Group 0):** none buildable for production behavior (the V6 path needs live LLM calls; the eval can't measure V6). The pure helpers are the testable seams.
- **Manual verification (required):** run the web app against `wrangler dev` with `HARNESS_V6_ENABLED=true`, complete a session, confirm the reflection+question renders, Confirm reveals the exercise, Not-now declines. This replaces the (now-deferred) eval re-lock.
- **Deferred follow-up:** make the eval harness V6-aware (run V6 in eval, fix the sparse-accumulator no-op) so teacher-quality work is automatable again, then re-lock baselines. Open a tracking issue.

## File Changes

| File | Change | Type |
|------|--------|------|
| `apps/api/src/harness/loop/phase2.ts` | extract `buildPhase2Prompt`; headline = reflection + directional question on dominant_dimension | Modify |
| `apps/api/src/harness/loop/phase2.test.ts` | prompt-builder assertions | New/Modify |
| `apps/api/src/db/schema/exercises.ts` | `pendingExercises` table | Modify |
| `apps/api/migrations/*` | Drizzle migration | New (generated) |
| `apps/api/src/services/pending-exercise.ts` | `stageDominantExercise`, `buildPendingExerciseComponent` | New |
| `apps/api/src/services/pending-exercise.test.ts` | helper tests | New |
| `apps/api/src/services/exercises.ts` | `assignPendingExercise`, `ExerciseSetPayload` | Modify |
| `apps/api/src/services/exercises.test.ts` | ownership + payload tests | New/Modify |
| `apps/api/src/routes/exercises.ts` | `POST /assign-pending` | Modify |
| `apps/api/src/routes/exercises.test.ts` | route + IDOR test | Modify |
| `apps/api/src/do/session-brain.ts` | extend `buildV6WsPayload`; stage in V6 block; emit pending component; persist | Modify |
| `apps/api/src/do/session-brain.unit.test.ts` | `buildV6WsPayload` with pending component | Modify |
| `apps/web/src/lib/types.ts` | `pending_exercise` component type | Modify |
| `apps/web/src/lib/api.ts` | `exercises.assignPending` | Modify |
| `apps/web/src/components/ReflectionMessage.tsx` | confirm/deny + reveal | New |
| `apps/web/src/components/ReflectionMessage.test.tsx` | component test | New |
| `apps/web/src/components/ChatMessages.tsx` | filter `pending_exercise` from auto-render | Modify |
| `apps/web/src/components/AppChat.tsx` | route synthesis msg w/ pending_exercise to ReflectionMessage | Modify |

## Open Questions

- **Q: Decline transport from web?** Default: a normal chat turn with a fixed decline string + the focus dimension as context; adaptivity is a chat-prompt behavior (no new endpoint).
- **Q: Persist the pending marker in the message `componentsJson` vs a `messages` column?** Default: `componentsJson` hidden marker (no migration); `pending_exercises` table is the validation source of truth.
- **Q: Headline ≥300 chars vs "light" reflection?** Default: keep the schema min; instruct a concise reflection + question within 300-500 chars (≈2-4 sentences). Revisit lowering the min only if it reads as bloated.
