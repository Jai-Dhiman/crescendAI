# Plan C: reflect-then-prescribe — V6 DO Staging (pending_exercise component)

> **Build-agent dispatch note:** This is Plan C of 4 for the reflect-then-prescribe feature.
> Tasks 1–3 are in Group A (independent, dispatch in parallel). Task 4 is Group B (depends on
> A passing). Each task follows the exact 5-step shape: write failing test → verify fail →
> implement → verify pass → commit. Fresh Sonnet 4.6 subagent per task group.

**Goal:** Deterministically stage ONE persisted exercise in the V6 DO block after the
`SynthesisArtifact` is produced, and deliver it as a hidden `pending_exercise` component
in the WebSocket payload and persisted synthesis message — a direct parallel to the existing
`assigned_loops → loopComponents` path. No new LLM tool, no artifact-schema change.

**Spec path:** `docs/specs/2026-06-05-reflect-then-prescribe-design.md`

**Ship-guard note:** Do NOT delete the shared spec until Plan D (web) ships. This plan
implements only the API/DO surface; the spec is still needed by Plans D and E.

**Style:** Follow `apps/api/TS_STYLE.md` throughout. Key rules for this plan:
- DO state versioning: snapshot `state` before any `await`; re-read `this.getState()` after
  if you mutate it. The V6 block in Task 4 does not mutate state, so no re-read is needed —
  but note it explicitly.
- `console.log(JSON.stringify({...}))` for any new logging in the DO.
- Services receive a raw `db` (Drizzle instance), not `ServiceContext`, because
  `stageDominantExercise` is called from the DO which has no `env`-bound service context.
  Match the pattern used by `persistDiagnosisArtifacts` (called at line 1666 of
  `session-brain.ts` with bare `db`).
- Explicit exception handling: if `stageDominantExercise` throws, log and continue without
  a pending component (do not let a staging failure kill the synthesis delivery).

**Dependency note:** DEPENDS ON Plan B (`pendingExercises` table, `pending_exercises` DB
table, Drizzle migration) merged to main first. Plan C must not run until the schema
migration is applied. Consumed by Plan D (web reads the `pending_exercise` component).
Benefits from Plan A (Phase-2 prompt → light reflection + directional question) but does
not require it — the `pending_exercise` component is useful regardless of prompt framing.

---

## Shared Contract (do NOT rename — matches Plans B and D)

### `PendingExercise` type

```typescript
export type PendingExercise = {
  exerciseId: string;
  focusDimension: string;
  previewTitle: string;
};
```

Canonical home: `apps/api/src/services/pending-exercise.ts` (new file). Export it from
there. Plans B and D import from this location.

> **Check before writing:** if a future merge of Plan A already exported `PendingExercise`
> from `apps/api/src/services/teacher.ts`, import it from there instead and re-export it
> from `pending-exercise.ts`. At the time of this writing no such export exists; define it
> in `pending-exercise.ts`.

### `stageDominantExercise` contract

```typescript
stageDominantExercise(
  db: Db,
  args: {
    studentId: string;
    sessionId: string;
    dominantDimension: string;
    proposedExercise: string;
    pieceMetadata: { title?: string; composer?: string } | null;
  }
): Promise<PendingExercise>
```

Behavior (in order, each step must throw on DB failure — no silent fallback):
1. Derive `previewTitle`: first 60 chars of `args.proposedExercise` (trimmed), or if blank,
   `"${args.dominantDimension} focus drill"`.
2. `db.insert(exercises).values({ title: previewTitle, description: "Staged from session synthesis", instructions: args.proposedExercise, difficulty: "intermediate", category: "generated", source: "teacher_llm" }).returning({ id: exercises.id })` — throw `InferenceError("Failed to insert staged exercise")` if result is empty.
3. `db.insert(exerciseDimensions).values({ exerciseId: inserted.id, dimension: args.dominantDimension })`.
4. `db.insert(pendingExercises).values({ studentId: args.studentId, sessionId: args.sessionId, exerciseId: inserted.id, focusDimension: args.dominantDimension, previewTitle, consumed: false })`.
5. Return `{ exerciseId: inserted.id, focusDimension: args.dominantDimension, previewTitle }`.

### `buildPendingExerciseComponent` contract

```typescript
buildPendingExerciseComponent(staged: PendingExercise): InlineComponent
// returns: { type: "pending_exercise", config: { exerciseId, focusDimension, previewTitle } }
```

`InlineComponent` is imported from `../services/tool-processor` (line 15:
`export interface InlineComponent { type: string; config: Record<string, unknown> }`).

### `buildV6WsPayload` extended signature

```typescript
export function buildV6WsPayload(
  artifact: SynthesisArtifact,
  loopComponents?: InlineComponent[],
  pendingComponent?: InlineComponent | null,
): { type: "synthesis"; text: string; components: InlineComponent[]; isFallback: false }
```

`components` = `[...(loopComponents ?? []), ...(pendingComponent ? [pendingComponent] : [])]`.
`text` = `artifact.headline` (unchanged). `isFallback` = `false` (unchanged).

### V6 block wiring (session-brain.ts)

After the `artifact` event and the `loopComponents` build (around line 1712), add:

```typescript
let pendingComponent: InlineComponent | null = null;
if (artifact.proposed_exercises.length > 0) {
  try {
    const staged = await stageDominantExercise(db, {
      studentId: state.studentId,
      sessionId: state.sessionId,
      dominantDimension: artifact.dominant_dimension,
      proposedExercise: artifact.proposed_exercises[0],
      pieceMetadata: pieceCtx,
    });
    pendingComponent = buildPendingExerciseComponent(staged);
  } catch (err) {
    const error = err as Error;
    console.log(
      JSON.stringify({
        level: "warn",
        message: "stageDominantExercise failed; synthesis delivered without pending component",
        sessionId: state.sessionId,
        error: error.message,
      }),
    );
  }
}
const wsPayload = buildV6WsPayload(artifact, loopComponents, pendingComponent);
```

`persistSynthesisMessage` already receives `wsPayload.components` (line 1727–1729); no
change needed there — the pending component rides in `wsPayload.components` automatically.

> **DO state versioning note:** `stageDominantExercise` does not read or write DO state
> (`this.state`). The `state` variable is already snapshotted before the V6 block entry;
> no re-read of `this.getState()` is needed after these awaits. If a future change reads
> DO state after this point, it must re-snapshot.

---

## Task Groups

### Group A — pure functions + service (dispatch in parallel, Tasks 1–3)

---

### Task 1 — `stageDominantExercise` + `PendingExercise` type

**Success criteria:** `stageDominantExercise` with a mocked `db` performs exactly three
inserts (exercises, exerciseDimensions, pendingExercises) in order, with the correct column
values, and returns a `PendingExercise` ref.

**Step 1 — Write the failing test**

Create `apps/api/src/services/pending-exercise.test.ts`:

```typescript
import { describe, expect, it, vi } from "vitest";
import { stageDominantExercise } from "./pending-exercise";

describe("stageDominantExercise", () => {
  function makeMockDb(exerciseId = "ex-uuid-1") {
    // Chain: insert().values().returning() for exercises
    // Chain: insert().values()             for exerciseDimensions
    // Chain: insert().values()             for pendingExercises
    const insertedValues: unknown[] = [];
    let callCount = 0;

    const mockDb = {
      insert: vi.fn().mockImplementation(() => {
        callCount++;
        const call = callCount;
        return {
          values: vi.fn().mockImplementation((v: unknown) => {
            insertedValues.push({ call, values: v });
            if (call === 1) {
              // exercises — has .returning()
              return {
                returning: vi.fn().mockResolvedValue([{ id: exerciseId }]),
              };
            }
            // exerciseDimensions and pendingExercises — no .returning()
            return Promise.resolve(undefined);
          }),
        };
      }),
    };
    return { mockDb, insertedValues };
  }

  it("inserts exercises, exerciseDimensions, pendingExercises and returns ref", async () => {
    const { mockDb, insertedValues } = makeMockDb("ex-uuid-1");

    const result = await stageDominantExercise(mockDb as never, {
      studentId: "stu-1",
      sessionId: "sess-1",
      dominantDimension: "pedaling",
      proposedExercise: "Practice slow legato pedaling in bars 1-4.",
      pieceMetadata: { title: "Nocturne", composer: "Chopin" },
    });

    expect(mockDb.insert).toHaveBeenCalledTimes(3);

    // exercises insert (call 1)
    const exercisesInsert = insertedValues[0] as { call: number; values: Record<string, unknown> };
    expect(exercisesInsert.values).toMatchObject({
      description: "Staged from session synthesis",
      instructions: "Practice slow legato pedaling in bars 1-4.",
      difficulty: "intermediate",
      category: "generated",
      source: "teacher_llm",
    });
    expect(typeof exercisesInsert.values["title"]).toBe("string");
    expect((exercisesInsert.values["title"] as string).length).toBeGreaterThan(0);

    // exerciseDimensions insert (call 2)
    const dimInsert = insertedValues[1] as { call: number; values: Record<string, unknown> };
    expect(dimInsert.values).toMatchObject({
      exerciseId: "ex-uuid-1",
      dimension: "pedaling",
    });

    // pendingExercises insert (call 3)
    const pendingInsert = insertedValues[2] as { call: number; values: Record<string, unknown> };
    expect(pendingInsert.values).toMatchObject({
      studentId: "stu-1",
      sessionId: "sess-1",
      exerciseId: "ex-uuid-1",
      focusDimension: "pedaling",
      consumed: false,
    });
    expect(typeof pendingInsert.values["previewTitle"]).toBe("string");

    // return value
    expect(result).toEqual({
      exerciseId: "ex-uuid-1",
      focusDimension: "pedaling",
      previewTitle: expect.any(String),
    });
  });

  it("derives previewTitle from first 60 chars of proposedExercise", async () => {
    const { mockDb } = makeMockDb();
    const longExercise = "A".repeat(100);
    const result = await stageDominantExercise(mockDb as never, {
      studentId: "s",
      sessionId: "sess",
      dominantDimension: "dynamics",
      proposedExercise: longExercise,
      pieceMetadata: null,
    });
    expect(result.previewTitle).toBe("A".repeat(60));
    expect(result.previewTitle.length).toBe(60);
  });

  it("falls back to dimension focus drill when proposedExercise is blank after trim", async () => {
    const { mockDb } = makeMockDb();
    const result = await stageDominantExercise(mockDb as never, {
      studentId: "s",
      sessionId: "sess",
      dominantDimension: "timing",
      proposedExercise: "   ",
      pieceMetadata: null,
    });
    expect(result.previewTitle).toBe("timing focus drill");
  });

  it("throws InferenceError when exercises insert returns empty array", async () => {
    const mockDb = {
      insert: vi.fn().mockReturnValue({
        values: vi.fn().mockReturnValue({
          returning: vi.fn().mockResolvedValue([]),
        }),
      }),
    };

    await expect(
      stageDominantExercise(mockDb as never, {
        studentId: "s",
        sessionId: "sess",
        dominantDimension: "pedaling",
        proposedExercise: "drill",
        pieceMetadata: null,
      }),
    ).rejects.toThrow("Failed to insert staged exercise");
  });
});
```

**Step 2 — Verify fail**

```bash
cd apps/api && bunx vitest run --config vitest.node.config.ts src/services/pending-exercise.test.ts
# expect: Cannot find module './pending-exercise' (or similar)
```

**Step 3 — Implement**

Create `apps/api/src/services/pending-exercise.ts`:

```typescript
import { exerciseDimensions, exercises, pendingExercises } from "../db/schema/exercises";
import { InferenceError } from "../lib/errors";
import type { InlineComponent } from "./tool-processor";
import type { Db } from "../lib/types";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type PendingExercise = {
  exerciseId: string;
  focusDimension: string;
  previewTitle: string;
};

// ---------------------------------------------------------------------------
// stageDominantExercise
// ---------------------------------------------------------------------------

export async function stageDominantExercise(
  db: Db,
  args: {
    studentId: string;
    sessionId: string;
    dominantDimension: string;
    proposedExercise: string;
    pieceMetadata: { title?: string; composer?: string } | null;
  },
): Promise<PendingExercise> {
  const trimmed = args.proposedExercise.trim();
  const previewTitle =
    trimmed.length > 0
      ? trimmed.slice(0, 60)
      : `${args.dominantDimension} focus drill`;

  const [inserted] = await db
    .insert(exercises)
    .values({
      title: previewTitle,
      description: "Staged from session synthesis",
      instructions: args.proposedExercise,
      difficulty: "intermediate",
      category: "generated",
      source: "teacher_llm",
    })
    .returning({ id: exercises.id });

  if (!inserted) {
    throw new InferenceError("Failed to insert staged exercise");
  }

  await db.insert(exerciseDimensions).values({
    exerciseId: inserted.id,
    dimension: args.dominantDimension,
  });

  await db.insert(pendingExercises).values({
    studentId: args.studentId,
    sessionId: args.sessionId,
    exerciseId: inserted.id,
    focusDimension: args.dominantDimension,
    previewTitle,
    consumed: false,
  });

  return { exerciseId: inserted.id, focusDimension: args.dominantDimension, previewTitle };
}
```

> **Type note:** `Db` is the Drizzle database instance type from `apps/api/src/lib/types.ts`.
> Verify the exact export name by checking `apps/api/src/lib/types.ts` before writing — if
> it is exported as `DrizzleDb` or another name, use that. The mock-as-never pattern in tests
> sidesteps the exact type.

**Step 4 — Verify pass**

```bash
cd apps/api && bunx vitest run --config vitest.node.config.ts src/services/pending-exercise.test.ts
# expect: all tests green
```

**Step 5 — Commit**

```
feat(api): add stageDominantExercise service for V6 exercise staging
```

---

### Task 2 — `buildPendingExerciseComponent` (pure)

**Success criteria:** `buildPendingExerciseComponent(staged)` returns an `InlineComponent`
with `type = "pending_exercise"` and `config` containing all three fields from the input.

**Step 1 — Write the failing test**

Append to `apps/api/src/services/pending-exercise.test.ts`:

```typescript
import { buildPendingExerciseComponent } from "./pending-exercise";

describe("buildPendingExerciseComponent", () => {
  it("returns a pending_exercise InlineComponent", () => {
    const staged: PendingExercise = {
      exerciseId: "ex-1",
      focusDimension: "dynamics",
      previewTitle: "Soft entry in bars 1-4",
    };
    const comp = buildPendingExerciseComponent(staged);
    expect(comp.type).toBe("pending_exercise");
    expect(comp.config).toEqual({
      exerciseId: "ex-1",
      focusDimension: "dynamics",
      previewTitle: "Soft entry in bars 1-4",
    });
  });

  it("config is a plain object (Record<string, unknown>)", () => {
    const comp = buildPendingExerciseComponent({
      exerciseId: "ex-2",
      focusDimension: "timing",
      previewTitle: "drill",
    });
    expect(typeof comp.config).toBe("object");
    expect(comp.config).not.toBeNull();
  });
});
```

(Note: add the import for `buildPendingExerciseComponent` and `PendingExercise` at the top
of the test file alongside the existing `stageDominantExercise` import.)

**Step 2 — Verify fail**

```bash
cd apps/api && bunx vitest run --config vitest.node.config.ts src/services/pending-exercise.test.ts
# expect: buildPendingExerciseComponent is not exported (or not a function)
```

**Step 3 — Implement**

Append to `apps/api/src/services/pending-exercise.ts`:

```typescript
// ---------------------------------------------------------------------------
// buildPendingExerciseComponent
// ---------------------------------------------------------------------------

export function buildPendingExerciseComponent(staged: PendingExercise): InlineComponent {
  return {
    type: "pending_exercise",
    config: {
      exerciseId: staged.exerciseId,
      focusDimension: staged.focusDimension,
      previewTitle: staged.previewTitle,
    },
  };
}
```

**Step 4 — Verify pass**

```bash
cd apps/api && bunx vitest run --config vitest.node.config.ts src/services/pending-exercise.test.ts
# expect: all tests green (Task 1 + Task 2 cases)
```

**Step 5 — Commit**

```
feat(api): add buildPendingExerciseComponent pure mapper
```

---

### Task 3 — extend `buildV6WsPayload` + unit tests

**Success criteria:** `buildV6WsPayload(artifact, loopComponents, pendingComponent)` appends
the pending component after loop components when provided, and omits it when `null` or
`undefined`. All existing unit tests in `session-brain.unit.test.ts` remain green.

**Step 1 — Write the failing tests**

Append to `apps/api/src/do/session-brain.unit.test.ts` (no `/// <reference types` header
needed — this file runs under `vitest.node.config.ts`):

```typescript
import { buildPendingExerciseComponent } from "../services/pending-exercise";
import type { PendingExercise } from "../services/pending-exercise";

// add inside the existing describe("buildV6WsPayload") block:

it("includes pendingComponent in components when provided", () => {
  const staged: PendingExercise = {
    exerciseId: "ex-staged-1",
    focusDimension: "pedaling",
    previewTitle: "Legato pedaling drill",
  };
  const pending = buildPendingExerciseComponent(staged);
  const payload = buildV6WsPayload(ARTIFACT, [], pending);
  expect(payload.components).toHaveLength(1);
  expect(payload.components[0]).toEqual({
    type: "pending_exercise",
    config: { exerciseId: "ex-staged-1", focusDimension: "pedaling", previewTitle: "Legato pedaling drill" },
  });
});

it("appends pendingComponent after loopComponents", () => {
  const loopComp = { type: "segment_loop", config: { id: "loop-1" } };
  const pendingComp = { type: "pending_exercise", config: { exerciseId: "ex-1", focusDimension: "dynamics", previewTitle: "drill" } };
  const payload = buildV6WsPayload(ARTIFACT_WITH_LOOP, [loopComp], pendingComp);
  expect(payload.components).toHaveLength(2);
  expect(payload.components[0]?.type).toBe("segment_loop");
  expect(payload.components[1]?.type).toBe("pending_exercise");
});

it("omits pendingComponent when null", () => {
  const loopComp = { type: "segment_loop", config: { id: "loop-1" } };
  const payload = buildV6WsPayload(ARTIFACT_WITH_LOOP, [loopComp], null);
  expect(payload.components).toHaveLength(1);
  expect(payload.components[0]?.type).toBe("segment_loop");
});

it("omits pendingComponent when undefined (backward-compatible)", () => {
  const payload = buildV6WsPayload(ARTIFACT);
  expect(payload.components).toEqual([]);
});
```

**Step 2 — Verify fail**

```bash
cd apps/api && bunx vitest run src/do/session-brain.unit.test.ts
# expect: TypeScript error or test failure — buildV6WsPayload does not accept third arg yet
```

**Step 3 — Implement**

Edit `apps/api/src/do/session-brain.ts` lines 59–74 (the `buildV6WsPayload` function):

Current signature:
```typescript
export function buildV6WsPayload(
  artifact: SynthesisArtifact,
  loopComponents?: InlineComponent[],
): { ... }
```

Replace with:
```typescript
export function buildV6WsPayload(
  artifact: SynthesisArtifact,
  loopComponents?: InlineComponent[],
  pendingComponent?: InlineComponent | null,
): {
  type: "synthesis";
  text: string;
  components: InlineComponent[];
  isFallback: false;
} {
  return {
    type: "synthesis",
    text: artifact.headline,
    components: [
      ...(loopComponents ?? []),
      ...(pendingComponent ? [pendingComponent] : []),
    ],
    isFallback: false,
  };
}
```

**Step 4 — Verify pass**

```bash
cd apps/api && bunx vitest run src/do/session-brain.unit.test.ts
# expect: all tests green — new cases + all pre-existing cases
```

**Step 5 — Commit**

```
feat(do): extend buildV6WsPayload to accept optional pending_exercise component
```

---

### Group B — V6 block wiring (depends on Group A)

---

### Task 4 — wire `stageDominantExercise` + `buildPendingExerciseComponent` into the V6 block

**Behavioral coverage:** The observable behavior (correct insert shape, correct component
shape, correct payload ordering) is verified by Tasks 1–3. Task 4 is integration wiring
verified by regression + typecheck.

**Step 1 — State the wiring target**

The V6 block in `session-brain.ts` currently (around line 1698–1712):

```typescript
const loopComponents = artifact.assigned_loops.map((ref) =>
  toLoopComponent({ ... }),
);
const wsPayload = buildV6WsPayload(artifact, loopComponents);
```

After Task 4 it becomes:

```typescript
const loopComponents = artifact.assigned_loops.map((ref) =>
  toLoopComponent({ ... }),
);
let pendingComponent: InlineComponent | null = null;
if (artifact.proposed_exercises.length > 0) {
  try {
    const staged = await stageDominantExercise(db, {
      studentId: state.studentId,
      sessionId: state.sessionId,
      dominantDimension: artifact.dominant_dimension,
      proposedExercise: artifact.proposed_exercises[0],
      pieceMetadata: pieceCtx,
    });
    pendingComponent = buildPendingExerciseComponent(staged);
  } catch (err) {
    const error = err as Error;
    console.log(
      JSON.stringify({
        level: "warn",
        message: "stageDominantExercise failed; synthesis delivered without pending component",
        sessionId: state.sessionId,
        error: error.message,
      }),
    );
  }
}
const wsPayload = buildV6WsPayload(artifact, loopComponents, pendingComponent);
```

`persistSynthesisMessage` on line 1722 already consumes `wsPayload.components` — no further
change needed there.

**Step 2 — Verify pre-wiring regression baseline**

Before making changes, confirm existing tests pass:

```bash
cd apps/api && bunx vitest run src/do/session-brain.unit.test.ts src/do/session-brain.canary.test.ts
# expect: all green (baseline)
```

**Step 3 — Implement**

In `apps/api/src/do/session-brain.ts`:

1. Add imports at the top of the file (alongside existing service imports):
   ```typescript
   import { stageDominantExercise, buildPendingExerciseComponent } from "../services/pending-exercise";
   ```
2. Replace the `const wsPayload = buildV6WsPayload(artifact, loopComponents);` line (around
   line 1712) with the wiring block shown in Step 1 above.
3. The `proposed_exercises[0]` non-null access is guarded by the `length > 0` check; the
   TypeScript compiler may still flag it — use `artifact.proposed_exercises[0]!` if needed.

> **DO state versioning note:** `stageDominantExercise` does not read or write `this.state`
> (the DO's durable state). The `state` variable was snapshotted from `this.getState()` well
> before the V6 try block (see line 1634 context). No re-read of `this.getState()` is needed
> after these awaits unless a future change reads DO state here.

**Step 4 — Verify pass**

```bash
cd apps/api && bunx vitest run src/do/session-brain.unit.test.ts src/do/session-brain.canary.test.ts
# expect: all green — no regressions
cd apps/api && bun run typecheck
# expect: no type errors
```

**Step 5 — Commit**

```
feat(do): wire stageDominantExercise into V6 block; emit pending_exercise component

On session exit, if the SynthesisArtifact carries proposed_exercises, one exercise is
persisted and returned as a hidden pending_exercise component alongside loopComponents.
Staging failure is non-fatal: synthesis delivers without the component and logs a warning.
```

---

## Verification Summary

| Check | Command | Owner |
|---|---|---|
| `stageDominantExercise` inserts + return | `bunx vitest run --config vitest.node.config.ts src/services/pending-exercise.test.ts` | Task 1 |
| `buildPendingExerciseComponent` shape | same command | Task 2 |
| `buildV6WsPayload` pending ordering | `bunx vitest run src/do/session-brain.unit.test.ts` | Task 3 |
| V6 block regression | `bunx vitest run src/do/session-brain.unit.test.ts src/do/session-brain.canary.test.ts` | Task 4 |
| Type safety | `cd apps/api && bun run typecheck` | Task 4 |
| Manual (required) | Run `wrangler dev`, complete a session with `HARNESS_V6_ENABLED=true`, confirm `pending_exercise` component appears in WS payload | Post-ship |

## Files Changed

| File | Change |
|---|---|
| `apps/api/src/services/pending-exercise.ts` | NEW — `PendingExercise` type, `stageDominantExercise`, `buildPendingExerciseComponent` |
| `apps/api/src/services/pending-exercise.test.ts` | NEW — mock-db tests for both functions |
| `apps/api/src/do/session-brain.ts` | MODIFY — extend `buildV6WsPayload` signature; wire staging into V6 block |
| `apps/api/src/do/session-brain.unit.test.ts` | MODIFY — add `buildV6WsPayload` pending-component cases |

---

## Challenge Review

### CEO Pass
- **Premise (OBS):** Correct — staging deterministically from the artifact's own `proposed_exercises[0]` + `dominant_dimension` is the minimal way to add a persisted, ownable exercise to the V6 path without a new LLM tool or schema migration. Direct parallel to the live `assigned_loops` path.
- **Scope (OBS):** 1 new service file + 1 signature extension + 1 wiring block. Tight.

### Engineering Pass — all seams verified against real code
- **Db type** `export type Db = PostgresJsDatabase<typeof schema>` (lib/types.ts:4) — plan's `import type { Db }` correct. SAFE.
- **Wiring site** `const wsPayload = buildV6WsPayload(artifact, loopComponents);` at session-brain.ts:1712 — exact, singular, inside the V6 gate (1634) after `loopComponents` (1698). SAFE.
- **`db` + `pieceCtx` in scope** — `const db = createDb(this.env.HYPERDRIVE)` (1564), `pieceCtx` (1509). SAFE.
- **Test fixtures present** — `session-brain.unit.test.ts` already defines `ARTIFACT` (14), `ARTIFACT_WITH_LOOP` (28), `describe("buildV6WsPayload")` (35); Task 3's appended cases will compile. SAFE.
- **exercises insert columns** (title, description, instructions, difficulty, category, source) match the schema's notNull columns. SAFE.
- **[RISK] (confidence: 4/10)** — `artifact.proposed_exercises[0]` under `noUncheckedIndexedAccess` may be typed `string | undefined`; the `length > 0` guard doesn't narrow the index access. Plan already notes using `!`. Build implementer: add `!` or a local const if typecheck flags it.
- **[OBS]** `ExerciseSetPayload.sourcePassage` will be the generic `description = "Staged from session synthesis"` (the staged exercise has no real source passage). Acceptable for MVP; the instruction body carries the actual drill text. Revisit if the web reveal reads awkwardly.
- **[OBS]** A staged `exercises` + `pending_exercises` row is written on every session exit with `proposed_exercises` even if the student never confirms — the user's explicit "eager pre-stage for instant reveal" tradeoff. Unconsumed rows are harmless.
- **Failure modes (OBS):** staging failure is caught, logged (`level: warn`), and synthesis still delivers without the component — correct non-fatal handling, no silent swallow (it logs).
- **Test Philosophy (OBS):** mock-db tests assert insert shapes + return (behavior at the db boundary); pure-function tests for the component mapper and payload builder. Task 4 is integration wiring verified by regression + typecheck (acceptable — behavior covered by Tasks 1–3).
- **Vertical Slice (OBS):** Tasks 1/2/3 each one behavior+impl+commit; Task 4 wiring. Group A (1–3) parallel-safe (1&2 same new file → must serialize 1 before 2; 3 is a different file). NOTE: Tasks 1 and 2 both edit `pending-exercise.ts`/`.test.ts` so they CANNOT run in true parallel — serialize 1→2; 3 is independent.

### Presumption Inventory
| Assumption | Verdict | Reason |
|---|---|---|
| `Db` type from lib/types | SAFE | Verified line 4 |
| `buildV6WsPayload` call at 1712, singular | SAFE | Verified |
| `db`/`pieceCtx`/`state.*` in V6 scope | SAFE | Verified |
| unit-test fixtures ARTIFACT(_WITH_LOOP) exist | SAFE | Verified |
| pendingExercises table available (Plan B merged) | SAFE | Plan B shipped to main |
| `proposed_exercises[0]` index-access typing | VALIDATE | May need `!` |

### Summary
[BLOCKER] count: 0
[RISK]    count: 1
[QUESTION] count: 0

VERDICT: PROCEED_WITH_CAUTION — (1) Tasks 1 & 2 touch the same new file; serialize them (1 then 2), do NOT dispatch in parallel; (2) add `!` to `proposed_exercises[0]` if typecheck flags it.
