# Reflect-then-Prescribe — Plan Index

**Feature:** On session exit (production V6 path), the teacher gives a light reflection + one directional question, with a single exercise pre-staged so confirm reveals it instantly; deny is teacher-adaptive.

**Issue:** #27 · **Spec:** `docs/specs/2026-06-05-reflect-then-prescribe-design.md`

Four separate plan files, each its own `/build` → `/challenge` → `/review` → `/ship` cycle. They are NOT independent in arbitrary order — there is a strict dependency chain via shared type contracts. Run them in the order below, merging each to `main` before building the next.

## Run order (linear; merge between each)

| # | Plan file | Surface | Depends on | Ships alone? |
|---|-----------|---------|-----------|--------------|
| A | `2026-06-05-reflect-then-prescribe-v6-prompt.md` | V6 Phase-2 prompt (`harness/loop/phase2.ts`) | none | yes (prompt-only) |
| B | `2026-06-05-reflect-then-prescribe-pending-assign.md` | `pending_exercises` table + `assignPendingExercise` + `POST /assign-pending` | none | yes (table+endpoint, no caller) |
| C | `2026-06-05-reflect-then-prescribe-v6-staging.md` | V6 DO staging + `pending_exercise` component | **B** (table) | yes (component web ignores until D) |
| D | `2026-06-05-reflect-then-prescribe-web.md` | Web confirm/deny + reveal | **B** (endpoint) + **C** (component) | LAST — lights up the feature |

Safe linear sequence for `/autopilot`: **A → B → C → D**. (A and B have no interdependency and could swap; C must follow B; D must follow B and C.)

## Running via /autopilot

`/autopilot` normally re-plans from a design. To execute THESE plan files instead, point each cycle at the plan file (`/build <plan-file>` → `/challenge` → `/review` → `/ship`) per stage, or invoke `/autopilot` with the plan path so it skips re-planning. Run one plan fully (through `/ship` + merge) before starting the next, so each builds against the merged contracts of the previous.

## Ship-guard

`/ship` deletes the spec + plan files whose slug it is shipping. All four plans share ONE spec. Each plan's header says: **do NOT delete the shared spec until plan D ships.** Plan D's `/ship` deletes the shared spec AND all four plan files (listed in its ship-guard note) plus this index.

## Shared contract (canonical names — identical across all plans)

- `PendingExercise = { exerciseId: string; focusDimension: string; previewTitle: string }` — canonical home `apps/api/src/services/pending-exercise.ts` (plan C).
- `pending_exercise` InlineComponent: `{ type: "pending_exercise", config: { exerciseId, focusDimension, previewTitle } }` (C produces, D consumes; web defines its own `PendingExerciseConfig`).
- `ExerciseSetPayload` (API, plan B) ≅ `ExerciseSetConfig` (web, plan D): `{ sourcePassage, targetSkill, exercises: [{ title, instruction, focusDimension, hands?, exerciseId }] }`.
- `pending_exercises` table (plan B): `studentId, sessionId, exerciseId, focusDimension, previewTitle, consumed, createdAt`; unique `(studentId, sessionId, exerciseId)`.
- `assignPendingExercise(ctx, { studentId, sessionId, exerciseId }) → ExerciseSetPayload`; route `POST /assign-pending` zod `{ sessionId: uuid, exerciseId: uuid }`; `NotFoundError → 404` (IDOR guard), `requireAuth → 401`.

## Verification & the dropped eval plan

The original plan set included an eval re-cut (re-lock `_SONNET_BASELINE`). **It was dropped.** Production runs the V6 harness; the eval harness deliberately bypasses V6 and measures the legacy `synthesize` path (`session-brain.ts:1631-1634`), so it **cannot measure this feature**. Verification is therefore:

- **Automated:** V6 harness + service + route + web unit tests (in plans A–D); `bunx tsc --noEmit` / `bun run typecheck`.
- **Manual (required):** run the web app against `wrangler dev` with `HARNESS_V6_ENABLED=true`, complete a session, confirm the reflection + question renders, Confirm reveals the exercise, Not-now declines.

**Deferred follow-up (open a tracking issue):** make the eval harness V6-aware (run V6 in eval, fix the sparse-accumulator no-op that caused eval to use the legacy path), then re-lock baselines on the V6 path — so future teacher-quality work is automatable again. This is the pre-existing eval-validity gap surfaced during planning, not introduced by this feature.
