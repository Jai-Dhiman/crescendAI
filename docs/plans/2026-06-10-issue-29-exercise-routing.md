# Issue #29 — Exercise Routing Contract Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** Replace freeform string exercise generation with a single structured `ExerciseRoutingDecision` emitted by the teacher in post-session synthesis and chat, with no pollution of the `exercises` catalog table.

**Spec:** docs/specs/2026-06-10-issue-29-exercise-routing-design.md

**Style:** Follow `apps/api/TS_STYLE.md` — ServiceContext DI, never destructure `c.env`, domain errors in services, Zod with JSON error hooks, DO state versioning across awaits, `console.log(JSON.stringify({...}))` logging. `bun` not npm. Explicit exceptions over silent fallbacks. No emojis.

---

## Drizzle migration note (apply locally before Task Group C)

After Task C generates the migration, apply it locally ONLY:
```bash
cd /path/to/crescendai && DATABASE_URL="postgresql://jdhiman:postgres@localhost:5432/crescendai_dev" bun run migrate
```
NEVER run bare `bun run migrate` — it targets production.

## ASCF baseline re-lock note

The locked ASCF baseline (0.959, n=98 holdout) reflects the old `proposed_exercises` path. After this PR merges, re-locking requires a live `--do-path` eval run (Anthropic credits required). This is a documented follow-on, not an S1 blocker. The eval gate (Task Group F) updates the rendering code so the eval runs without a KeyError, but does not attempt a re-lock.

---

## Task Groups

- **Group A (parallel):** Task 1 (exercise-routing contract)
- **Group B (depends on A, parallel):** Task 2 (SynthesisArtifact schema migration), Task 3 (phase2 prompt update)
- **Group C (depends on B, parallel):** Task 4 (DB schema migration + pending-exercise service rewrite), Task 5 (post-session DO wiring)
- **Group D (depends on C):** Task 6 (assignPendingExercise rewrite)
- **Group E (depends on A, parallel with B/C/D):** Task 7 (delete dead code), Task 8 (prescribe_exercise chat tool)
- **Group F (depends on D+E):** Task 9 (web ExerciseSetCard corpus_drill stub), Task 10 (eval render update)
- **Group G (depends on all):** Task 11 (fixture sweep — update proposed_exercises in every remaining test file)

---

### Task 1: ExerciseRoutingDecision contract schema

**Group:** A

**Behavior being verified:** `ExerciseRoutingDecisionSchema` accepts valid `own_passage_loop` and `corpus_drill` payloads and rejects payloads that violate constraints (wrong kind, bar_range start > end, tempo_factor out of range, primitive_id on own_passage_loop).

**Interface under test:** `ExerciseRoutingDecisionSchema.parse(...)` and `.safeParse(...)`

**Files:**
- Create: `apps/api/src/harness/artifacts/exercise-routing.ts`
- Create: `apps/api/src/harness/artifacts/exercise-routing.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/harness/artifacts/exercise-routing.test.ts
import { describe, test, expect } from "vitest";
import { ExerciseRoutingDecisionSchema } from "./exercise-routing";

describe("ExerciseRoutingDecisionSchema — own_passage_loop", () => {
  const validLoop = {
    kind: "own_passage_loop",
    target_dimension: "pedaling",
    bar_range: [12, 16],
    tempo_factor: 0.75,
  };

  test("accepts a valid own_passage_loop", () => {
    expect(() => ExerciseRoutingDecisionSchema.parse(validLoop)).not.toThrow();
  });

  test("rejects own_passage_loop with bar_range start > end", () => {
    const result = ExerciseRoutingDecisionSchema.safeParse({
      ...validLoop,
      bar_range: [16, 12],
    });
    expect(result.success).toBe(false);
    expect(result.error?.issues.some((i) => i.path.includes("bar_range"))).toBe(true);
  });

  test("rejects own_passage_loop with tempo_factor below 0.25", () => {
    const result = ExerciseRoutingDecisionSchema.safeParse({
      ...validLoop,
      tempo_factor: 0.1,
    });
    expect(result.success).toBe(false);
  });

  test("rejects own_passage_loop with tempo_factor above 1.0", () => {
    const result = ExerciseRoutingDecisionSchema.safeParse({
      ...validLoop,
      tempo_factor: 1.1,
    });
    expect(result.success).toBe(false);
  });

  test("rejects own_passage_loop with invalid target_dimension", () => {
    const result = ExerciseRoutingDecisionSchema.safeParse({
      ...validLoop,
      target_dimension: "vibrato",
    });
    expect(result.success).toBe(false);
  });

  test("rejects own_passage_loop with bar_range containing non-positive numbers", () => {
    const result = ExerciseRoutingDecisionSchema.safeParse({
      ...validLoop,
      bar_range: [0, 4],
    });
    expect(result.success).toBe(false);
  });
});

describe("ExerciseRoutingDecisionSchema — corpus_drill", () => {
  const validDrill = {
    kind: "corpus_drill",
    target_dimension: "timing",
    bar_range: [1, 8],
    tempo_factor: 0.8,
    primitive_id: null,
  };

  test("accepts a valid corpus_drill with primitive_id null", () => {
    expect(() => ExerciseRoutingDecisionSchema.parse(validDrill)).not.toThrow();
  });

  test("accepts a valid corpus_drill with non-null primitive_id", () => {
    expect(() =>
      ExerciseRoutingDecisionSchema.parse({ ...validDrill, primitive_id: "drill-abc" })
    ).not.toThrow();
  });

  test("rejects corpus_drill with bar_range start > end", () => {
    const result = ExerciseRoutingDecisionSchema.safeParse({
      ...validDrill,
      bar_range: [8, 1],
    });
    expect(result.success).toBe(false);
  });
});

describe("ExerciseRoutingDecisionSchema — discriminant", () => {
  test("rejects an unknown kind", () => {
    const result = ExerciseRoutingDecisionSchema.safeParse({
      kind: "free_improv",
      target_dimension: "phrasing",
      bar_range: [1, 4],
      tempo_factor: 0.5,
    });
    expect(result.success).toBe(false);
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun test --run src/harness/artifacts/exercise-routing.test.ts
```
Expected: FAIL — `Cannot find module './exercise-routing'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```typescript
// apps/api/src/harness/artifacts/exercise-routing.ts
import { z } from "zod";
import { DIMS_6 } from "../../lib/dims";

const DimensionEnum = z.enum(DIMS_6 as unknown as [string, ...string[]]);

const barRangeSchema = z
  .tuple([z.number().int().positive(), z.number().int().positive()])
  .refine(([start, end]) => start <= end, {
    message: "bar_range start must be <= end",
    path: ["bar_range"],
  });

const tempoFactorSchema = z.number().min(0.25).max(1.0);

export const OwnPassageLoopSchema = z.object({
  kind: z.literal("own_passage_loop"),
  target_dimension: DimensionEnum,
  bar_range: barRangeSchema,
  tempo_factor: tempoFactorSchema,
});

export const CorpusDrillSchema = z.object({
  kind: z.literal("corpus_drill"),
  target_dimension: DimensionEnum,
  bar_range: barRangeSchema,
  tempo_factor: tempoFactorSchema,
  primitive_id: z.string().nullable(),
});

export const ExerciseRoutingDecisionSchema = z.discriminatedUnion("kind", [
  OwnPassageLoopSchema,
  CorpusDrillSchema,
]);

export type OwnPassageLoopDecision = z.infer<typeof OwnPassageLoopSchema>;
export type CorpusDrillDecision = z.infer<typeof CorpusDrillSchema>;
export type ExerciseRoutingDecision = z.infer<typeof ExerciseRoutingDecisionSchema>;
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun test --run src/harness/artifacts/exercise-routing.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai && git add apps/api/src/harness/artifacts/exercise-routing.ts apps/api/src/harness/artifacts/exercise-routing.test.ts && git commit -m "feat(#29): add ExerciseRoutingDecision contract schema"
```

---

### Task 2: SynthesisArtifact schema migration (proposed_exercises → prescribed_exercise)

**Group:** B (depends on Task 1)

**Behavior being verified:** `SynthesisArtifactSchema` rejects payloads that include `proposed_exercises` and accepts payloads with `prescribed_exercise: null` or a valid `ExerciseRoutingDecision`.

**Interface under test:** `SynthesisArtifactSchema.parse(...)` / `.safeParse(...)`

**Files:**
- Modify: `apps/api/src/harness/artifacts/synthesis.ts`
- Modify: `apps/api/src/harness/artifacts/synthesis.test.ts`

- [ ] **Step 1: Write the failing test**

Add to `apps/api/src/harness/artifacts/synthesis.test.ts` (after existing tests):

```typescript
import { ExerciseRoutingDecisionSchema } from "./exercise-routing";

// --- prescribed_exercise field tests ---

const BASE_PRESCRIBED = {
  session_id: "sess_1",
  synthesis_scope: "session" as const,
  strengths: [],
  focus_areas: [],
  dominant_dimension: "phrasing" as const,
  recurring_pattern: null,
  next_session_focus: null,
  diagnosis_refs: [],
  headline: "A".repeat(300),
  assigned_loops: [],
};

test("SynthesisArtifact accepts prescribed_exercise null", () => {
  const result = SynthesisArtifactSchema.safeParse({
    ...BASE_PRESCRIBED,
    prescribed_exercise: null,
  });
  expect(result.success).toBe(true);
});

test("SynthesisArtifact accepts a valid own_passage_loop prescribed_exercise", () => {
  const result = SynthesisArtifactSchema.safeParse({
    ...BASE_PRESCRIBED,
    prescribed_exercise: {
      kind: "own_passage_loop",
      target_dimension: "pedaling",
      bar_range: [12, 16],
      tempo_factor: 0.75,
    },
  });
  expect(result.success).toBe(true);
});

test("SynthesisArtifact accepts a valid corpus_drill prescribed_exercise", () => {
  const result = SynthesisArtifactSchema.safeParse({
    ...BASE_PRESCRIBED,
    prescribed_exercise: {
      kind: "corpus_drill",
      target_dimension: "timing",
      bar_range: [1, 8],
      tempo_factor: 0.8,
      primitive_id: null,
    },
  });
  expect(result.success).toBe(true);
});

test("SynthesisArtifact rejects invalid prescribed_exercise (bad kind)", () => {
  const result = SynthesisArtifactSchema.safeParse({
    ...BASE_PRESCRIBED,
    prescribed_exercise: {
      kind: "free_form",
      target_dimension: "phrasing",
    },
  });
  expect(result.success).toBe(false);
});

test("SynthesisArtifact without prescribed_exercise field defaults to null", () => {
  const result = SynthesisArtifactSchema.safeParse(BASE_PRESCRIBED);
  expect(result.success).toBe(true);
  if (result.success) expect(result.data.prescribed_exercise).toBeNull();
});
```

Also in Step 3 you will remove the `proposed_exercises` field from synthesis.ts. The existing test `"rejects when proposed_exercises exceeds 3 items"` will need to be removed since the field is gone. Update `baseValid` fixture by removing `proposed_exercises` and adding `prescribed_exercise: null`.

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun test --run src/harness/artifacts/synthesis.test.ts
```
Expected: FAIL — `prescribed_exercise` not in schema, and existing tests referencing `proposed_exercises` will break.

- [ ] **Step 3: Implement the minimum to make the test pass**

Update `apps/api/src/harness/artifacts/synthesis.ts`:

```typescript
import { z } from "zod";
import { DIMENSIONS, SEVERITIES } from "./diagnosis";
import { ExerciseRoutingDecisionSchema } from "./exercise-routing";

export const SYNTHESIS_SCOPES = [
  "session",
  "weekly",
  "piece_onboarding",
] as const;

const DimensionEnum = z.enum(DIMENSIONS);
const SeverityEnum = z.enum(SEVERITIES);
const SynthesisScopeEnum = z.enum(SYNTHESIS_SCOPES);

const StrengthEntry = z.object({
  dimension: DimensionEnum,
  one_liner: z.string().min(1).max(200),
});

const FocusAreaEntry = z.object({
  dimension: DimensionEnum,
  one_liner: z.string().min(1).max(200),
  severity: SeverityEnum,
});

export const SynthesisArtifactSchema = z
  .object({
    session_id: z.string().min(1),
    synthesis_scope: SynthesisScopeEnum,
    strengths: z.array(StrengthEntry).max(2),
    focus_areas: z.array(FocusAreaEntry).max(3),
    prescribed_exercise: ExerciseRoutingDecisionSchema.nullable().default(null),
    dominant_dimension: DimensionEnum,
    recurring_pattern: z.string().min(1).nullable(),
    next_session_focus: z.string().min(1).max(200).nullable(),
    diagnosis_refs: z.array(z.string().min(1)),
    headline: z.string().min(300).max(500),
    assigned_loops: z
      .array(
        z
          .object({
            id: z.string().min(1),
            pieceId: z.string().min(1),
            barsStart: z.number().int().positive(),
            barsEnd: z.number().int().positive(),
          })
          .refine((v) => v.barsEnd >= v.barsStart, {
            message: "barsEnd must be >= barsStart",
            path: ["barsEnd"],
          }),
      )
      .default([]),
  })
  .refine(
    (s) => s.synthesis_scope !== "weekly" || s.recurring_pattern !== null,
    {
      message: 'recurring_pattern is required when synthesis_scope is "weekly"',
      path: ["recurring_pattern"],
    },
  )
  .refine(
    (s) =>
      s.synthesis_scope !== "piece_onboarding" ||
      s.focus_areas.every((f) => f.severity === "minor"),
    {
      message:
        'when synthesis_scope is "piece_onboarding", all focus_areas[].severity must be "minor"',
      path: ["focus_areas"],
    },
  );

export type SynthesisArtifact = z.infer<typeof SynthesisArtifactSchema>;
```

Also update `apps/api/src/harness/artifacts/synthesis.test.ts`:
- Remove `proposed_exercises` from `baseValid` (replace with `prescribed_exercise: null`)
- Remove `BASE_VALID.proposed_exercises: []` (replace with `prescribed_exercise: null`)
- Remove the `"rejects when proposed_exercises exceeds 3 items"` test
- Update any artifact fixture that has `proposed_exercises` field

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun test --run src/harness/artifacts/synthesis.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai && git add apps/api/src/harness/artifacts/synthesis.ts apps/api/src/harness/artifacts/synthesis.test.ts && git commit -m "feat(#29): migrate SynthesisArtifact from proposed_exercises to prescribed_exercise"
```

---

### Task 3: Phase 2 prompt instruction update

**Group:** B (depends on Task 1, parallel with Task 2)

**Behavior being verified:** `buildPhase2Prompt` instructs the model to emit a `prescribed_exercise` field (single routing decision tied to `dominant_dimension`) and no longer mentions `proposed_exercises[0]`.

**Interface under test:** `buildPhase2Prompt(digest, diagnoses, guardrail)` return value string assertions.

**Files:**
- Modify: `apps/api/src/harness/loop/phase2.ts`
- Modify: `apps/api/src/harness/loop/phase2.test.ts`

- [ ] **Step 1: Write the failing test**

In `apps/api/src/harness/loop/phase2.test.ts`, replace the existing `"instructs proposed_exercises[0] to target dominant_dimension"` test with:

```typescript
it("instructs prescribed_exercise to target dominant_dimension as own_passage_loop or corpus_drill", () => {
  const prompt = buildPhase2Prompt(digest, diagnoses, "");
  expect(prompt).toContain("prescribed_exercise");
  expect(prompt).toContain("own_passage_loop");
  expect(prompt).toContain("corpus_drill");
  expect(prompt).toContain("dominant_dimension");
  expect(prompt).not.toContain("proposed_exercises[0]");
  expect(prompt).not.toContain("proposed_exercises");
});
```

Keep the remaining existing tests. They use `proposed_exercises` fixture checks — those will also need updating once the `VALID_ARTIFACT` fixtures are updated (done in Task 11 sweep, but you may update the local `VALID_ARTIFACT` const in phase2.test.ts now since it needs `prescribed_exercise: null` instead of `proposed_exercises: []`).

Update `VALID_ARTIFACT` in `phase2.test.ts`:
- Remove `proposed_exercises: []`
- Add `prescribed_exercise: null`

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun test --run src/harness/loop/phase2.test.ts
```
Expected: FAIL — assertion `not.toContain("proposed_exercises")` fails because the prompt still has `proposed_exercises[0]`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Update `buildPhase2Prompt` in `apps/api/src/harness/loop/phase2.ts`:

```typescript
export function buildPhase2Prompt(
  digest: Record<string, unknown>,
  diagnoses: unknown[],
  guardrail: string,
): string {
  const reflectionInstruction =
    "Headline instructions: write a light reflection in 2-4 sentences about what happened " +
    "in this session, ending in exactly one directional question about the dominant_dimension " +
    "(e.g. 'Want a drill targeting that?'). The headline must be 300-500 characters total. " +
    "Do not list all dimensions; focus on the one area that matters most.\n\n";

  const exerciseInstruction =
    "Exercise instructions: set prescribed_exercise to a single routing decision that targets " +
    "the dominant_dimension. Use kind='own_passage_loop' when the student has been identified " +
    "playing a specific piece and you want them to loop a bar range from it; use " +
    "kind='corpus_drill' when no piece is identified or a general technique drill would be " +
    "more appropriate. Set prescribed_exercise to null if no exercise is warranted. " +
    "Do NOT put a pieceId in prescribed_exercise — that is bound at the serving layer.\n\n";

  return (
    `Session digest:\n${JSON.stringify(digest, null, 2)}\n\n` +
    `Collected diagnoses (${diagnoses.length}):\n${JSON.stringify(diagnoses, null, 2)}\n\n` +
    guardrail +
    reflectionInstruction +
    exerciseInstruction +
    `Write the SynthesisArtifact now using the write_synthesis_artifact tool.`
  );
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun test --run src/harness/loop/phase2.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai && git add apps/api/src/harness/loop/phase2.ts apps/api/src/harness/loop/phase2.test.ts && git commit -m "feat(#29): update phase2 prompt to prescribe structured exercise decision"
```

---

### Task 4: DB schema migration + pending-exercise service rewrite

**Group:** C (depends on Group B)

**Behavior being verified:** `stageDominantExercise` writes a row to `pending_exercises` with `title`, `instruction`, and `routing_json` populated without touching `exercises` or `exerciseDimensions`.

**Interface under test:** `stageDominantExercise(db, args)` return value + observable side-effects via the pending-exercise service's public API.

**Files:**
- Modify: `apps/api/src/db/schema/exercises.ts`
- New: Drizzle migration SQL (generated via `bun drizzle-kit generate` or `just migrate-generate`)
- Modify: `apps/api/src/services/pending-exercise.ts`
- Modify: `apps/api/src/services/pending-exercise.test.ts`

- [ ] **Step 1: Write the failing test**

**RISK 3 fix:** The test actually calls `stageDominantExercise` with a mocked `db` whose `insert` is spied. It asserts: `insert` is called exactly once and only against `pendingExercises` (never `exercises` or `exerciseDimensions`), and that `routing_json`, `title`, `instruction`, and `piece_id` are persisted in the inserted row.

```typescript
// apps/api/src/services/pending-exercise.test.ts
// (Replace existing test that assumed exercises table insert)
import { describe, test, expect, vi } from "vitest";
import {
  stageDominantExercise,
  buildPendingExerciseComponent,
} from "./pending-exercise";
import { pendingExercises } from "../db/schema/exercises";
import type { ExerciseRoutingDecision } from "../harness/artifacts/exercise-routing";

const ROUTING: ExerciseRoutingDecision = {
  kind: "own_passage_loop",
  target_dimension: "pedaling",
  bar_range: [12, 16],
  tempo_factor: 0.75,
};

describe("stageDominantExercise", () => {
  test("inserts exactly once into pending_exercises with routing_json, title, instruction, piece_id; never inserts into exercises", async () => {
    const insertSpy = vi.fn();
    let capturedTable: unknown = null;
    let capturedRow: unknown = null;

    const db = {
      insert: (table: unknown) => {
        capturedTable = table;
        insertSpy(table);
        return {
          values: (row: unknown) => {
            capturedRow = row;
            return {
              returning: () => Promise.resolve([{ id: "pending-row-id-1" }]),
            };
          },
        };
      },
    };

    const result = await stageDominantExercise(db as never, {
      studentId: "stu-1",
      sessionId: "sess-1",
      dominantDimension: "pedaling",
      routing: ROUTING,
      pieceCtx: { pieceId: "chopin.ballade.1" },
    });

    // insert called exactly once — never touches exercises or exerciseDimensions
    expect(insertSpy).toHaveBeenCalledTimes(1);
    expect(capturedTable).toBe(pendingExercises);

    // persisted fields
    const row = capturedRow as Record<string, unknown>;
    expect(row.routingJson).toEqual(ROUTING);
    expect(typeof row.title).toBe("string");
    expect((row.title as string).length).toBeGreaterThan(0);
    expect(typeof row.instruction).toBe("string");
    expect(row.pieceId).toBe("chopin.ballade.1");

    // return value uses the pending row id as exerciseId
    expect(result.exerciseId).toBe("pending-row-id-1");
    expect(result.focusDimension).toBe("pedaling");
  });

  test("pieceId is null when pieceCtx is null", async () => {
    let capturedRow: unknown = null;
    const db = {
      insert: (_table: unknown) => ({
        values: (row: unknown) => {
          capturedRow = row;
          return { returning: () => Promise.resolve([{ id: "pending-row-id-2" }]) };
        },
      }),
    };

    await stageDominantExercise(db as never, {
      studentId: "stu-1",
      sessionId: "sess-1",
      dominantDimension: "pedaling",
      routing: ROUTING,
      pieceCtx: null,
    });

    expect((capturedRow as Record<string, unknown>).pieceId).toBeNull();
  });
});

describe("buildPendingExerciseComponent", () => {
  test("sets exerciseId to the pending row id returned by stageDominantExercise", () => {
    const staged = {
      exerciseId: "pending-row-id-1",
      focusDimension: "pedaling",
      previewTitle: "Own passage loop: pedaling (bars 12-16)",
    };
    const component = buildPendingExerciseComponent(staged);
    expect(component.type).toBe("pending_exercise");
    expect(component.config.exerciseId).toBe("pending-row-id-1");
    expect(component.config.focusDimension).toBe("pedaling");
    expect(component.config.previewTitle).toBe("Own passage loop: pedaling (bars 12-16)");
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun test --run src/services/pending-exercise.test.ts
```
Expected: FAIL — current `stageDominantExercise` calls `db.insert` three times (exercises, exerciseDimensions, pending_exercises), so `insertSpy` count > 1 and `capturedTable !== pendingExercises`.

- [ ] **Step 3: Implement the minimum to make the test pass**

**BLOCKER 1 fix:** The old `exerciseId uuid` column is DROPPED entirely. It existed only as a FK reference to `exercises.id`. Since `stageDominantExercise` no longer writes to `exercises`, there is no valid UUID to put there. `PendingExerciseConfig.exerciseId` is now the pending row's own PK (`id`), which is already a uuid.

**BLOCKER 2 fix:** A new nullable `pieceId text` column is added. `stageDominantExercise` receives `pieceCtx` (already available in the DO call site) and stores `pieceCtx?.pieceId ?? null` at staging time. `assignPendingExercise` reads it back in Task 6 to build `scoreClip` — no route or web changes needed.

Update `apps/api/src/db/schema/exercises.ts` — replace the `pendingExercises` table definition.

**Final column set for `pending_exercises`:** `id` (uuid PK), `studentId`, `sessionId`, `focusDimension`, `previewTitle`, `title` (text, nullable), `instruction` (text, nullable), `routingJson` (jsonb, nullable), `pieceId` (text, nullable), `consumed`, `createdAt`. The old `exerciseId` column is DROPPED.

```typescript
export const pendingExercises = pgTable(
  "pending_exercises",
  {
    id: uuid("id").defaultRandom().primaryKey(),
    studentId: text("student_id").notNull(),
    sessionId: uuid("session_id").notNull(),
    focusDimension: text("focus_dimension").notNull(),
    previewTitle: text("preview_title").notNull(),
    title: text("title"),
    instruction: text("instruction"),
    routingJson: jsonb("routing_json"),
    pieceId: text("piece_id"),
    consumed: boolean("consumed").notNull().default(false),
    createdAt: timestamp("created_at", { withTimezone: true })
      .notNull()
      .defaultNow(),
  },
  (t) => [
    uniqueIndex("idx_pending_exercises_unique").on(
      t.studentId,
      t.sessionId,
      t.id,
    ),
    index("idx_pending_exercises_lookup").on(t.studentId, t.consumed),
  ],
);
```

Note on uniqueIndex: the old unique index included `exerciseId`. Since that column is dropped, the index is updated to `(studentId, sessionId, id)`. Because `id` is a PK this is always unique; the index exists for lookup performance.

Generate migration:
```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun drizzle-kit generate
```

Review the generated SQL — it must contain:
- `ALTER TABLE pending_exercises DROP COLUMN exercise_id;`
- `ALTER TABLE pending_exercises ADD COLUMN title text;`
- `ALTER TABLE pending_exercises ADD COLUMN instruction text;`
- `ALTER TABLE pending_exercises ADD COLUMN routing_json jsonb;`
- `ALTER TABLE pending_exercises ADD COLUMN piece_id text;`
- DROP of the old unique index referencing `exercise_id`; CREATE of the new unique index on `(student_id, session_id, id)`

If `bun drizzle-kit generate` does not detect the DROP automatically, manually add the `DROP COLUMN exercise_id` statement to the generated SQL file before applying.

Apply locally:
```bash
cd /Users/jdhiman/Documents/crescendai && DATABASE_URL="postgresql://jdhiman:postgres@localhost:5432/crescendai_dev" bun run migrate
```

Rewrite `apps/api/src/services/pending-exercise.ts`:

```typescript
import { pendingExercises } from "../db/schema/exercises";
import { InferenceError } from "../lib/errors";
import type { Db } from "../lib/types";
import type { ExerciseRoutingDecision } from "../harness/artifacts/exercise-routing";
import type { InlineComponent } from "./tool-processor";

export type PendingExercise = {
  exerciseId: string; // = pending_exercises.id (the row PK)
  focusDimension: string;
  previewTitle: string;
};

export async function stageDominantExercise(
  db: Db,
  args: {
    studentId: string;
    sessionId: string;
    dominantDimension: string;
    routing: ExerciseRoutingDecision;
    pieceCtx: { pieceId?: string } | null;
  },
): Promise<PendingExercise> {
  const barLabel = `bars ${args.routing.bar_range[0]}-${args.routing.bar_range[1]}`;
  const title =
    args.routing.kind === "own_passage_loop"
      ? `Own passage loop: ${args.dominantDimension} (${barLabel})`
      : `${args.dominantDimension} drill (${barLabel})`;
  const previewTitle = title.slice(0, 60);

  const instruction =
    args.routing.kind === "own_passage_loop"
      ? `Loop ${barLabel} from your recording at ${Math.round(args.routing.tempo_factor * 100)}% tempo, focusing on ${args.dominantDimension}.`
      : `${args.dominantDimension} drill — ${barLabel} at ${Math.round(args.routing.tempo_factor * 100)}% tempo.`;

  const pieceId = args.pieceCtx?.pieceId ?? null;

  const [inserted] = await db
    .insert(pendingExercises)
    .values({
      studentId: args.studentId,
      sessionId: args.sessionId,
      focusDimension: args.dominantDimension,
      previewTitle,
      title,
      instruction,
      routingJson: args.routing as unknown as Record<string, unknown>,
      pieceId,
      consumed: false,
    })
    .returning({ id: pendingExercises.id });

  if (!inserted) {
    throw new InferenceError("Failed to insert pending exercise");
  }

  return {
    exerciseId: inserted.id, // pending row PK — this is what the web client sends back
    focusDimension: args.dominantDimension,
    previewTitle,
  };
}

export function buildPendingExerciseComponent(
  staged: PendingExercise,
): InlineComponent {
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

No write to `exercises` or `exerciseDimensions` anywhere in this file. `PendingExerciseConfig.exerciseId` is now the pending row PK. `pieceId` is stored at staging time from `pieceCtx` so the assign step can build `scoreClip` without needing the route to supply it.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun test --run src/services/pending-exercise.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai && git add apps/api/src/db/schema/exercises.ts apps/api/src/services/pending-exercise.ts apps/api/src/services/pending-exercise.test.ts && git add apps/api/drizzle/ && git commit -m "feat(#29): drop exerciseId FK col; add title/instruction/routing_json/piece_id to pending_exercises; rewrite stageDominantExercise"
```

---

### Task 5: Post-session DO wiring (SessionBrain)

**Group:** C (depends on Group B, parallel with Task 4)

**Behavior being verified:** `buildV6WsPayload` and the DO synthesis path use `artifact.prescribed_exercise` instead of `artifact.proposed_exercises[0]`; `proposed_exercises` references are removed from the test fixtures.

**Interface under test:** `buildV6WsPayload(artifact, loopComponents, pendingComponent)` — existing exported function.

**Files:**
- Modify: `apps/api/src/do/session-brain.ts`
- Modify: `apps/api/src/do/session-brain.unit.test.ts`

- [ ] **Step 1: Write the failing test**

In `apps/api/src/do/session-brain.unit.test.ts`, update the `ARTIFACT` const — remove `proposed_exercises` and add `prescribed_exercise: null`. Also add a new test:

```typescript
it("does not access proposed_exercises on the artifact (field removed)", () => {
  // If this test compiles, proposed_exercises is not on SynthesisArtifact type.
  // Runtime: buildV6WsPayload should work with no proposed_exercises.
  const payload = buildV6WsPayload({
    ...ARTIFACT,
    prescribed_exercise: {
      kind: "own_passage_loop" as const,
      target_dimension: "pedaling" as const,
      bar_range: [12, 16] as [number, number],
      tempo_factor: 0.75,
    },
  });
  expect(payload.type).toBe("synthesis");
  // The pending component is NOT on the payload (staging happens in the DO, not here)
  expect(payload.components).toEqual([]);
});
```

Update all `ARTIFACT` and `ARTIFACT_WITH_LOOP` constants to remove `proposed_exercises: []` and add `prescribed_exercise: null`.

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun test --run src/do/session-brain.unit.test.ts
```
Expected: FAIL — TypeScript error: `proposed_exercises` does not exist on `SynthesisArtifact` (after Task 2 landed), and `prescribed_exercise` missing from the existing `ARTIFACT` fixture.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `apps/api/src/do/session-brain.ts`, find the block around line 1797–1821 and replace:

```typescript
// OLD (remove):
let pendingComponent: InlineComponent | null = null;
const proposedExercise = artifact.proposed_exercises[0];
if (proposedExercise !== undefined) {
  try {
    const staged = await stageDominantExercise(db, {
      studentId: state.studentId,
      sessionId: state.sessionId,
      dominantDimension: artifact.dominant_dimension,
      proposedExercise,
      pieceMetadata: pieceCtx,
    });
    pendingComponent = buildPendingExerciseComponent(staged);
  } catch (err) {
    const error = err as Error;
    console.log(
      JSON.stringify({
        level: "warn",
        message:
          "stageDominantExercise failed; synthesis delivered without pending component",
        sessionId: state.sessionId,
        error: error.message,
      }),
    );
  }
}
```

Replace with:

```typescript
// NEW:
let pendingComponent: InlineComponent | null = null;
if (artifact.prescribed_exercise !== null) {
  try {
    const staged = await stageDominantExercise(db, {
      studentId: state.studentId,
      sessionId: state.sessionId,
      dominantDimension: artifact.dominant_dimension,
      routing: artifact.prescribed_exercise,
      pieceCtx,
    });
    pendingComponent = buildPendingExerciseComponent(staged);
  } catch (err) {
    const error = err as Error;
    console.log(
      JSON.stringify({
        level: "warn",
        message:
          "stageDominantExercise failed; synthesis delivered without pending component",
        sessionId: state.sessionId,
        error: error.message,
      }),
    );
  }
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun test --run src/do/session-brain.unit.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai && git add apps/api/src/do/session-brain.ts apps/api/src/do/session-brain.unit.test.ts && git commit -m "feat(#29): wire DO to use prescribed_exercise from SynthesisArtifact"
```

---

### Task 6: assignPendingExercise rewrite — read routing_json, no catalog join

**Group:** D (depends on Group C)

**Behavior being verified:** `assignPendingExercise` reads `pending_exercises.routing_json`, returns `ExerciseSetPayload` with `scoreClip` for `own_passage_loop` (when `pieceId` available in the pending row context) and a text stub for `corpus_drill`; never reads the `exercises` table.

**Interface under test:** `assignPendingExercise(ctx, args)` return value shape.

**Files:**
- Modify: `apps/api/src/services/exercises.ts`
- Modify: `apps/api/src/services/exercises.test.ts`

- [ ] **Step 1: Write the failing test**

Add to `apps/api/src/services/exercises.test.ts`:

```typescript
import { ExerciseRoutingDecisionSchema } from "../harness/artifacts/exercise-routing";

describe("assignPendingExercise — routing_json path", () => {
  const OWN_PASSAGE_ROUTING = ExerciseRoutingDecisionSchema.parse({
    kind: "own_passage_loop",
    target_dimension: "pedaling",
    bar_range: [12, 16],
    tempo_factor: 0.75,
  });

  const CORPUS_DRILL_ROUTING = ExerciseRoutingDecisionSchema.parse({
    kind: "corpus_drill",
    target_dimension: "timing",
    bar_range: [1, 8],
    tempo_factor: 0.8,
    primitive_id: null,
  });

  test("own_passage_loop with pieceId in row produces scoreClip in ExerciseSetPayload", async () => {
    // Mock DB returns a pending row with routing_json and pieceId stored at staging time
    const mockCtx = buildMockCtxWithPendingRow({
      routingJson: OWN_PASSAGE_ROUTING,
      focusDimension: "pedaling",
      previewTitle: "Pedaling drill",
      title: "Own passage loop",
      instruction: "Loop bars 12-16",
      pieceId: "chopin.ballade.1",
    });
    // pieceId is NOT passed as an arg — it is read from the pending row
    const payload = await assignPendingExercise(mockCtx, {
      studentId: "stu-1",
      sessionId: "sess-1",
      exerciseId: "pending-row-id",
    });
    expect(payload.scoreClip).toEqual({
      pieceId: "chopin.ballade.1",
      bars: [12, 16],
    });
    expect(payload.exercises[0].focusDimension).toBe("pedaling");
  });

  test("own_passage_loop with null pieceId in row produces no scoreClip (text-only)", async () => {
    const mockCtx = buildMockCtxWithPendingRow({
      routingJson: OWN_PASSAGE_ROUTING,
      focusDimension: "pedaling",
      previewTitle: "Pedaling drill",
      title: "Own passage loop",
      instruction: "Loop bars 12-16",
      pieceId: null,
    });
    const payload = await assignPendingExercise(mockCtx, {
      studentId: "stu-1",
      sessionId: "sess-1",
      exerciseId: "pending-row-id",
    });
    expect(payload.scoreClip).toBeUndefined();
    expect(payload.exercises[0].instruction).toContain("Loop bars 12-16");
  });

  test("corpus_drill produces text stub, no scoreClip", async () => {
    const mockCtx = buildMockCtxWithPendingRow({
      routingJson: CORPUS_DRILL_ROUTING,
      focusDimension: "timing",
      previewTitle: "Timing drill",
      title: "Timing corpus drill",
      instruction: "Timing drill — bars 1-8",
      pieceId: "chopin.ballade.1",
    });
    const payload = await assignPendingExercise(mockCtx, {
      studentId: "stu-1",
      sessionId: "sess-1",
      exerciseId: "pending-row-id",
    });
    expect(payload.scoreClip).toBeUndefined();
    expect(payload.exercises[0].instruction).toContain("coming soon");
  });
});

// Helper to build a mock ServiceContext with the given pending row.
// pieceId is on the row itself (stored at staging time) — not passed via args.
// exerciseId column has been dropped; it is not present on the row.
function buildMockCtxWithPendingRow(row: {
  routingJson: unknown;
  focusDimension: string;
  previewTitle: string;
  title: string | null;
  instruction: string | null;
  pieceId: string | null;
}) {
  const pendingRow = {
    id: "pending-row-id",
    studentId: "stu-1",
    sessionId: "sess-1",
    focusDimension: row.focusDimension,
    previewTitle: row.previewTitle,
    title: row.title,
    instruction: row.instruction,
    routingJson: row.routingJson,
    pieceId: row.pieceId,
    consumed: false,
    createdAt: new Date(),
  };

  const mockDb = {
    select: () => ({
      from: () => ({
        where: () => Promise.resolve([pendingRow]),
      }),
    }),
    update: () => ({
      set: () => ({
        where: () => Promise.resolve([]),
      }),
    }),
    insert: () => ({
      values: () => ({
        onConflictDoUpdate: () => ({
          returning: () => Promise.resolve([{ id: "se-id" }]),
        }),
      }),
    }),
    query: {
      exercises: {
        findFirst: () => Promise.resolve(null),
      },
    },
  };

  return { db: mockDb as unknown as import("../lib/types").ServiceContext["db"], env: {} } as unknown as import("../lib/types").ServiceContext;
}
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun test --run src/services/exercises.test.ts
```
Expected: FAIL — `assignPendingExercise` does not read `routing_json` or `piece_id` from the pending row; current implementation joins to the `exercises` catalog table instead.

- [ ] **Step 3: Implement the minimum to make the test pass**

**BLOCKER 2 fix:** `assignPendingExercise` reads `piece_id` from the pending row (stored at staging time) rather than from the route/args. No route or web changes needed for pieceId. If `piece_id` is null, `own_passage_loop` renders text-only with a warn log.

**RISK 4 fix:** The `assignExercise` call is explicitly REMOVED. `assignExercise` calls `ctx.db.query.exercises.findFirst()` and throws `NotFoundError` if the id is not in the `exercises` table. The pending row id is not a catalog exercise id, so this call would throw on every confirm. `studentExercises` tracking for S1 routed exercises is out of scope — noted with TODO(S3).

**RISK 5 fix:** `ExerciseSetPayload.exercises[].exerciseId` is made optional (`exerciseId?: string`). The chat-path `processPrescribeExercise` (Task 8) returns exercise objects without `exerciseId`; making it optional fixes the TypeScript error. Also update `ExerciseSetConfig` in `apps/web/src/lib/types.ts` and confirm `ExerciseSetCard` already gates on `exercise.exerciseId` before rendering the assign button (verified: line 120).

Update `assignPendingExercise` in `apps/api/src/services/exercises.ts`:

```typescript
import { and, eq } from "drizzle-orm";
import { pendingExercises } from "../db/schema/exercises";
import { NotFoundError, InferenceError } from "../lib/errors";
import type { ServiceContext } from "../lib/types";
import { ExerciseRoutingDecisionSchema } from "../harness/artifacts/exercise-routing";

// RISK 5: exerciseId is optional — corpus_drill and chat-path exercises may not have one
export type ExerciseSetPayload = {
  sourcePassage: string;
  targetSkill: string;
  scoreClip?: { pieceId: string; bars: [number, number] };
  exercises: Array<{
    title: string;
    instruction: string;
    focusDimension: string;
    hands?: "left" | "right" | "both";
    exerciseId?: string; // optional: absent for corpus_drill stub and chat-path exercises
  }>;
};

export async function assignPendingExercise(
  ctx: ServiceContext,
  args: {
    studentId: string;
    sessionId: string;
    exerciseId: string; // = pending_exercises.id (the pending row PK)
  },
): Promise<ExerciseSetPayload> {
  const [pendingRow] = await ctx.db
    .select()
    .from(pendingExercises)
    .where(
      and(
        eq(pendingExercises.studentId, args.studentId),
        eq(pendingExercises.sessionId, args.sessionId),
        eq(pendingExercises.id, args.exerciseId),
        eq(pendingExercises.consumed, false),
      ),
    );

  if (!pendingRow) {
    throw new NotFoundError("pending exercise", args.exerciseId);
  }

  if (!pendingRow.routingJson) {
    throw new InferenceError(
      `pending exercise ${args.exerciseId} has no routing_json`,
    );
  }

  const routingResult = ExerciseRoutingDecisionSchema.safeParse(
    pendingRow.routingJson,
  );
  if (!routingResult.success) {
    throw new InferenceError(
      `pending exercise ${args.exerciseId} routing_json invalid: ${routingResult.error.message}`,
    );
  }
  const routing = routingResult.data;

  // RISK 4: do NOT call assignExercise here. assignExercise does exercises.findFirst()
  // and throws NotFoundError because the pending row id is not in the exercises catalog.
  // TODO(S3): wire studentExercises tracking for routed exercises when the catalog
  // has a real exercise row to reference.

  await ctx.db
    .update(pendingExercises)
    .set({ consumed: true })
    .where(eq(pendingExercises.id, pendingRow.id));

  const title = pendingRow.title ?? pendingRow.previewTitle;
  const instruction =
    pendingRow.instruction ??
    `${pendingRow.focusDimension} exercise — bars ${routing.bar_range[0]}-${routing.bar_range[1]}`;

  if (routing.kind === "own_passage_loop") {
    // BLOCKER 2: pieceId comes from the pending row (stored at staging time), not the route
    const pieceId = pendingRow.pieceId ?? null;
    const scoreClip =
      pieceId !== null
        ? { pieceId, bars: routing.bar_range as [number, number] }
        : undefined;

    if (!scoreClip) {
      console.log(
        JSON.stringify({
          level: "warn",
          message: "assignPendingExercise: own_passage_loop has no piece_id; rendering text-only",
          exerciseId: args.exerciseId,
        }),
      );
    }

    return {
      sourcePassage: `bars ${routing.bar_range[0]}-${routing.bar_range[1]}`,
      targetSkill: pendingRow.focusDimension,
      scoreClip,
      exercises: [
        {
          title,
          instruction,
          focusDimension: pendingRow.focusDimension,
          exerciseId: pendingRow.id,
        },
      ],
    };
  }

  // corpus_drill — text stub only
  const stubInstruction =
    `${pendingRow.focusDimension} drill coming soon` +
    (routing.primitive_id ? ` (drill: ${routing.primitive_id})` : "") +
    ". In the meantime: " + instruction;

  return {
    sourcePassage: `bars ${routing.bar_range[0]}-${routing.bar_range[1]}`,
    targetSkill: pendingRow.focusDimension,
    exercises: [
      {
        title,
        instruction: stubInstruction,
        focusDimension: pendingRow.focusDimension,
        exerciseId: pendingRow.id,
      },
    ],
  };
}
```

Also update `ExerciseSetConfig` in `apps/web/src/lib/types.ts` — make `exerciseId` optional:

```typescript
// In ExerciseSetConfig (wherever it is defined in apps/web/src/lib/types.ts):
exercises: Array<{
  title: string;
  instruction: string;
  focusDimension: string;
  hands?: "left" | "right" | "both";
  exerciseId?: string; // optional — corpus_drill stub and chat-path exercises omit it
}>;
```

Update the Task 6 test helpers: remove `pieceId` from `assignPendingExercise` args (it now reads from the row) and update `buildMockCtxWithPendingRow` to use the `pieceId` field on the pending row directly. Update the mock's `select` chain to return a `pendingRow` that includes `pieceId`. The `exerciseId: null` field in the mock row is no longer needed (column dropped), remove it.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun test --run src/services/exercises.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai && git add apps/api/src/services/exercises.ts apps/api/src/services/exercises.test.ts apps/web/src/lib/types.ts && git commit -m "feat(#29): rewrite assignPendingExercise to read routing_json; no exercises catalog join; exerciseId optional"
```

---

### Task 7: Delete dead code (exercise-proposal molecule + artifacts/exercise.ts)

**Group:** E (depends on Task 1, parallel with Group B/C/D)

**Behavior being verified:** `ALL_MOLECULES` has 8 members without `exercise-proposal`; `artifacts/index.ts` exports no `ExerciseArtifact`; `compound-registry.test.ts` no longer expects `create_exercise` in `OnChatMessage`.

**Interface under test:** `ALL_MOLECULES` length and name list; `ARTIFACT_NAMES` in artifacts/index.ts.

**Files:**
- Delete: `apps/api/src/harness/skills/molecules/exercise-proposal.ts`
- Delete: `apps/api/src/harness/skills/molecules/exercise-proposal.test.ts`
- Delete: `apps/api/src/harness/artifacts/exercise.ts`
- Delete: `apps/api/src/harness/artifacts/exercise.test.ts`
- Delete: `apps/api/src/harness/skills/__catalog__/molecule-exercise-proposal.test.ts`
- Modify: `apps/api/src/harness/skills/molecules/index.ts`
- Modify: `apps/api/src/harness/skills/molecules/index.test.ts`
- Modify: `apps/api/src/harness/artifacts/index.ts`

- [ ] **Step 1: Write the failing test**

In `apps/api/src/harness/skills/molecules/index.test.ts`, update the count and name list:

```typescript
import { test, expect } from 'vitest'
import { ALL_MOLECULES } from './index'

test('ALL_MOLECULES contains 8 ToolDefinition objects with unique names', () => {
  expect(ALL_MOLECULES).toHaveLength(8)
  const names = ALL_MOLECULES.map(m => m.name)
  expect(new Set(names).size).toBe(8)
  for (const mol of ALL_MOLECULES) {
    expect(typeof mol.name).toBe('string')
    expect(mol.name.length).toBeGreaterThan(0)
    expect(typeof mol.description).toBe('string')
    expect(typeof mol.invoke).toBe('function')
    expect(typeof mol.input_schema).toBe('object')
  }
})

test('ALL_MOLECULES includes all 8 named molecules (exercise-proposal removed)', () => {
  const names = new Set(ALL_MOLECULES.map(m => m.name))
  const expected = [
    'voicing-diagnosis', 'pedal-triage', 'rubato-coaching', 'phrasing-arc-analysis',
    'tempo-stability-triage', 'dynamic-range-audit', 'articulation-clarity-check',
    'cross-modal-contradiction-check',
  ]
  for (const name of expected) {
    expect(names.has(name), `missing molecule: ${name}`).toBe(true)
  }
  expect(names.has('exercise-proposal')).toBe(false)
})
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun test --run src/harness/skills/molecules/index.test.ts
```
Expected: FAIL — `ALL_MOLECULES` has length 9, not 8.

- [ ] **Step 3: Implement the minimum to make the test pass**

Delete the files:
```bash
rm /Users/jdhiman/Documents/crescendai/apps/api/src/harness/skills/molecules/exercise-proposal.ts
rm /Users/jdhiman/Documents/crescendai/apps/api/src/harness/skills/molecules/exercise-proposal.test.ts
rm /Users/jdhiman/Documents/crescendai/apps/api/src/harness/artifacts/exercise.ts
rm /Users/jdhiman/Documents/crescendai/apps/api/src/harness/artifacts/exercise.test.ts
rm /Users/jdhiman/Documents/crescendai/apps/api/src/harness/skills/__catalog__/molecule-exercise-proposal.test.ts
```

Update `apps/api/src/harness/skills/molecules/index.ts`:

```typescript
import type { ToolDefinition } from '../../loop/types'
import { voicingDiagnosis } from './voicing-diagnosis'
import { pedalTriage } from './pedal-triage'
import { rubatoCoaching } from './rubato-coaching'
import { phrasingArcAnalysis } from './phrasing-arc-analysis'
import { tempoStabilityTriage } from './tempo-stability-triage'
import { dynamicRangeAudit } from './dynamic-range-audit'
import { articulationClarityCheck } from './articulation-clarity-check'
import { crossModalContradictionCheck } from './cross-modal-contradiction-check'

export const ALL_MOLECULES: ToolDefinition[] = [
  voicingDiagnosis,
  pedalTriage,
  rubatoCoaching,
  phrasingArcAnalysis,
  tempoStabilityTriage,
  dynamicRangeAudit,
  articulationClarityCheck,
  crossModalContradictionCheck,
]
```

Update `apps/api/src/harness/artifacts/index.ts` — remove ExerciseArtifact, add ExerciseRoutingDecision:

```typescript
import type { ZodTypeAny } from "zod";
import { DiagnosisArtifactSchema, type DiagnosisArtifact } from "./diagnosis";
import { SynthesisArtifactSchema, type SynthesisArtifact } from "./synthesis";
import { SegmentLoopArtifactSchema, type SegmentLoopArtifact } from "./segment-loop";

export { DiagnosisArtifactSchema, type DiagnosisArtifact } from "./diagnosis";
export { SynthesisArtifactSchema, type SynthesisArtifact } from "./synthesis";
export { SegmentLoopArtifactSchema, type SegmentLoopArtifact, type SegmentLoopRef } from "./segment-loop";
export { ExerciseRoutingDecisionSchema, type ExerciseRoutingDecision } from "./exercise-routing";

export const ARTIFACT_NAMES = [
  "DiagnosisArtifact",
  "SynthesisArtifact",
  "SegmentLoopArtifact",
] as const;
export type ArtifactName = (typeof ARTIFACT_NAMES)[number];

export const artifactSchemas: Record<ArtifactName, ZodTypeAny> = {
  DiagnosisArtifact: DiagnosisArtifactSchema,
  SynthesisArtifact: SynthesisArtifactSchema,
  SegmentLoopArtifact: SegmentLoopArtifactSchema,
};
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun test --run src/harness/skills/molecules/index.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai && git rm apps/api/src/harness/skills/molecules/exercise-proposal.ts apps/api/src/harness/skills/molecules/exercise-proposal.test.ts apps/api/src/harness/artifacts/exercise.ts apps/api/src/harness/artifacts/exercise.test.ts apps/api/src/harness/skills/__catalog__/molecule-exercise-proposal.test.ts && git add apps/api/src/harness/skills/molecules/index.ts apps/api/src/harness/skills/molecules/index.test.ts apps/api/src/harness/artifacts/index.ts && git commit -m "feat(#29): delete exercise-proposal molecule and legacy ExerciseArtifact schema"
```

---

### Task 8: prescribe_exercise chat tool (replace create_exercise)

**Group:** E (depends on Task 1, parallel with Tasks 2/3/4/5/6/7)

**Behavior being verified:** `TOOL_REGISTRY` contains `prescribe_exercise` and not `create_exercise`; `processToolUse(ctx, studentId, "prescribe_exercise", validInput)` returns an `exercise_set` InlineComponent with `scoreClip` for `own_passage_loop`; `compound-registry.test.ts` finds `prescribe_exercise` in `OnChatMessage` binding tools and not `create_exercise`.

**Interface under test:** `processToolUse(ctx, studentId, toolName, toolInput)` return value; `getAnthropicToolSchemas()` tool name list.

**Files:**
- Modify: `apps/api/src/services/tool-processor.ts`
- Modify: `apps/api/src/services/tool-processor.test.ts`
- Modify: `apps/api/src/services/prompts.ts`
- Modify: `apps/api/src/services/prompts.test.ts`
- Modify: `apps/api/src/harness/loop/compound-registry.test.ts`

- [ ] **Step 1: Write the failing test**

In `apps/api/src/services/tool-processor.test.ts`, add:

```typescript
import { describe, test, expect } from "vitest";
import { TOOL_REGISTRY, getAnthropicToolSchemas, processToolUse } from "./tool-processor";

describe("TOOL_REGISTRY — prescribe_exercise replaces create_exercise", () => {
  test("prescribe_exercise is registered", () => {
    expect(TOOL_REGISTRY["prescribe_exercise"]).toBeDefined();
  });

  test("create_exercise is NOT registered", () => {
    expect(TOOL_REGISTRY["create_exercise"]).toBeUndefined();
  });

  test("getAnthropicToolSchemas includes prescribe_exercise and not create_exercise", () => {
    const names = getAnthropicToolSchemas().map((s) => s.name);
    expect(names).toContain("prescribe_exercise");
    expect(names).not.toContain("create_exercise");
  });
});

describe("processToolUse — prescribe_exercise own_passage_loop", () => {
  test("returns exercise_set InlineComponent with scoreClip for own_passage_loop", async () => {
    const ctx = { db: {} } as unknown as import("../lib/types").ServiceContext;
    const result = await processToolUse(ctx, "stu-1", "prescribe_exercise", {
      kind: "own_passage_loop",
      target_dimension: "pedaling",
      bar_range: [12, 16],
      tempo_factor: 0.75,
      piece_id: "chopin.ballade.1",
    });
    expect(result.isError).toBe(false);
    expect(result.componentsJson).toHaveLength(1);
    const comp = result.componentsJson[0];
    expect(comp?.type).toBe("exercise_set");
    const config = comp?.config as { scoreClip?: { pieceId: string; bars: [number, number] } };
    expect(config.scoreClip).toEqual({
      pieceId: "chopin.ballade.1",
      bars: [12, 16],
    });
  });

  test("returns exercise_set without scoreClip for corpus_drill", async () => {
    const ctx = { db: {} } as unknown as import("../lib/types").ServiceContext;
    const result = await processToolUse(ctx, "stu-1", "prescribe_exercise", {
      kind: "corpus_drill",
      target_dimension: "timing",
      bar_range: [1, 8],
      tempo_factor: 0.8,
      primitive_id: null,
      piece_id: "chopin.ballade.1",
    });
    expect(result.isError).toBe(false);
    const comp = result.componentsJson[0];
    expect(comp?.type).toBe("exercise_set");
    const config = comp?.config as { scoreClip?: unknown; exercises: { instruction: string }[] };
    expect(config.scoreClip).toBeUndefined();
    expect(config.exercises[0].instruction).toContain("coming soon");
  });
});
```

In `apps/api/src/harness/loop/compound-registry.test.ts`, update the `OnChatMessage` test:

```typescript
it("returns a streaming binding for OnChatMessage with prescribe_exercise tool", () => {
  const binding = getCompoundBinding("OnChatMessage");
  expect(binding).toBeDefined();
  expect(binding?.compoundName).toBe("chat-response");
  expect(binding?.mode).toBe("streaming");
  expect(binding?.phases).toBe(1);
  expect(binding!.tools.length).toBeGreaterThanOrEqual(Object.values(TOOL_REGISTRY).length + 1);
  const names = binding!.tools.map((t) => t.name);
  expect(new Set(names).size).toBe(names.length);
  expect(names).toContain("prescribe_exercise");
  expect(names).not.toContain("create_exercise");
  expect(names).toContain("search_catalog");
  expect(names).toContain("assign_segment_loop");
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun test --run src/services/tool-processor.test.ts src/harness/loop/compound-registry.test.ts
```
Expected: FAIL — `prescribe_exercise` not in registry; `create_exercise` still present.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `apps/api/src/services/tool-processor.ts`:

1. Remove the `createExerciseSchema`, `createExerciseAnthropicSchema`, `persistGeneratedExercise`, and `processCreateExercise` functions entirely.

2. Remove the `exerciseDimensions` and `exercises` imports (keep only those still used by other tools — check: `listExercises` uses them; those imports stay).

3. Add `prescribeExerciseSchema`, `prescribeExerciseAnthropicSchema`, and `processPrescribeExercise`:

```typescript
import { ExerciseRoutingDecisionSchema } from "../harness/artifacts/exercise-routing";
import { DIMS_6 } from "../lib/dims";

// prescribe_exercise uses the routing decision fields + piece_id at the edge
const prescribeExerciseSchema = ExerciseRoutingDecisionSchema.and(
  z.object({ piece_id: z.string().min(1).nullable() })
);

async function processPrescribeExercise(
  _ctx: ServiceContext,
  _studentId: string,
  rawInput: unknown,
): Promise<InlineComponent[]> {
  const input = prescribeExerciseSchema.parse(rawInput);

  if (input.kind === "own_passage_loop") {
    const scoreClip =
      input.piece_id !== null
        ? { pieceId: input.piece_id, bars: input.bar_range as [number, number] }
        : undefined;

    return [
      {
        type: "exercise_set",
        config: {
          sourcePassage: `bars ${input.bar_range[0]}-${input.bar_range[1]}`,
          targetSkill: `${input.target_dimension} focus`,
          scoreClip,
          exercises: [
            {
              title: `Own passage loop: ${input.target_dimension}`,
              instruction: `Loop bars ${input.bar_range[0]}-${input.bar_range[1]} at ${Math.round(input.tempo_factor * 100)}% tempo. Focus on ${input.target_dimension}.`,
              focusDimension: input.target_dimension,
            },
          ],
        },
      },
    ];
  }

  // corpus_drill — text stub
  const stubInstruction =
    `${input.target_dimension} drill coming soon` +
    (input.primitive_id ? ` (drill: ${input.primitive_id})` : "") +
    `. Practice bars ${input.bar_range[0]}-${input.bar_range[1]} at ${Math.round(input.tempo_factor * 100)}% tempo focusing on ${input.target_dimension}.`;

  return [
    {
      type: "exercise_set",
      config: {
        sourcePassage: `bars ${input.bar_range[0]}-${input.bar_range[1]}`,
        targetSkill: `${input.target_dimension} focus`,
        exercises: [
          {
            title: `${input.target_dimension} corpus drill`,
            instruction: stubInstruction,
            focusDimension: input.target_dimension,
          },
        ],
      },
    },
  ];
}

const prescribeExerciseAnthropicSchema: AnthropicToolSchema = {
  name: "prescribe_exercise",
  description:
    "Prescribe a single targeted practice exercise for the student. Use own_passage_loop when the student has been playing a specific piece and you want them to loop a bar range. Use corpus_drill when a general technique drill is warranted. The exercise renders immediately in chat — no confirmation step.",
  input_schema: {
    type: "object",
    properties: {
      kind: {
        type: "string",
        enum: ["own_passage_loop", "corpus_drill"],
        description: "own_passage_loop: loop from the student's piece. corpus_drill: general technique drill.",
      },
      target_dimension: {
        type: "string",
        enum: DIMS_6,
        description: "The musical dimension this exercise targets.",
      },
      bar_range: {
        type: "array",
        items: { type: "integer", minimum: 1 },
        minItems: 2,
        maxItems: 2,
        description: "Bar range [start, end] for the exercise. Start must be <= end.",
      },
      tempo_factor: {
        type: "number",
        minimum: 0.25,
        maximum: 1.0,
        description: "Practice tempo as a fraction of performance tempo (0.25 = 25%, 1.0 = full tempo).",
      },
      primitive_id: {
        type: ["string", "null"],
        description: "For corpus_drill only: drill primitive identifier. Pass null in S1.",
      },
      piece_id: {
        type: ["string", "null"],
        description: "Catalog piece ID for own_passage_loop. Use search_catalog to find it. Pass null for corpus_drill.",
      },
    },
    required: ["kind", "target_dimension", "bar_range", "tempo_factor", "piece_id"],
  },
};
```

4. Replace `create_exercise` in `TOOL_REGISTRY` with `prescribe_exercise`:

```typescript
export const TOOL_REGISTRY: Record<string, ToolDefinition> = {
  prescribe_exercise: {
    name: "prescribe_exercise",
    description: prescribeExerciseAnthropicSchema.description,
    schema: prescribeExerciseSchema,
    anthropicSchema: prescribeExerciseAnthropicSchema,
    concurrencySafe: true,
    process: processPrescribeExercise,
  },
  score_highlight: { ... },
  // ... rest unchanged
};
```

5. In `apps/api/src/services/prompts.ts`, update the `UNIFIED_TEACHER_SYSTEM` tool usage line (currently ~96):

```typescript
// Replace:
// "- create_exercise: When a concrete drill would help more than verbal guidance. Use sparingly."
// With:
"- prescribe_exercise: When a concrete drill would help more than verbal guidance. Use own_passage_loop when the student is working on a specific piece and you want them to loop a bar range from it. Use corpus_drill for general technique drills. Use sparingly.",
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun test --run src/services/tool-processor.test.ts src/harness/loop/compound-registry.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai && git add apps/api/src/services/tool-processor.ts apps/api/src/services/tool-processor.test.ts apps/api/src/services/prompts.ts apps/api/src/harness/loop/compound-registry.test.ts && git commit -m "feat(#29): replace create_exercise with prescribe_exercise in chat tool registry"
```

---

### Task 9: Web ExerciseSetCard — graceful corpus_drill stub rendering

**Group:** F (depends on Groups D + E)

**Behavior being verified:** `ExerciseSetCard` renders without crashing when `config.scoreClip` is absent and `exercises[].exerciseId` is absent (corpus_drill stub path from chat tool).

**Interface under test:** React component render — verifiable by checking that no exception is thrown and a fallback string appears.

**Files:**
- Modify: `apps/web/src/components/cards/ExerciseSetCard.tsx`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/web/src/components/cards/ExerciseSetCard.test.tsx
import { render, screen } from "@testing-library/react";
import { describe, test, expect } from "vitest";
import { ExerciseSetCard } from "./ExerciseSetCard";

describe("ExerciseSetCard — corpus_drill stub rendering", () => {
  test("renders without scoreClip and without exerciseId (corpus_drill stub)", () => {
    const config = {
      sourcePassage: "bars 1-8",
      targetSkill: "timing focus",
      exercises: [
        {
          title: "Timing corpus drill",
          instruction: "Timing drill coming soon. Practice bars 1-8 at 80% tempo.",
          focusDimension: "timing",
          // no exerciseId — corpus_drill path
        },
      ],
      // no scoreClip
    };
    expect(() =>
      render(<ExerciseSetCard config={config} />)
    ).not.toThrow();
    expect(screen.getByText("timing focus")).toBeInTheDocument();
    expect(screen.getByText("Timing corpus drill")).toBeInTheDocument();
  });

  test("renders with scoreClip present (own_passage_loop path) — existing behavior preserved", () => {
    const config = {
      sourcePassage: "bars 12-16",
      targetSkill: "pedaling focus",
      scoreClip: { pieceId: "chopin.ballade.1", bars: [12, 16] as [number, number] },
      exercises: [
        {
          title: "Own passage loop: pedaling",
          instruction: "Loop bars 12-16 at 75% tempo.",
          focusDimension: "pedaling",
          exerciseId: "ex-id-1",
        },
      ],
    };
    expect(() =>
      render(<ExerciseSetCard config={config} />)
    ).not.toThrow();
    expect(screen.getByText("pedaling focus")).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/web && bun test --run src/components/cards/ExerciseSetCard.test.tsx
```
Expected: FAIL — test file does not exist yet (component likely already renders, but we need to verify the test file doesn't exist).

If the component already handles missing scoreClip and exerciseId gracefully (scoreClip is already optional in `ExerciseSetConfig`, and the assign button is only shown when `exerciseId` is present), the test may pass without code changes. In that case, the test is the only addition needed. Verify by running.

- [ ] **Step 3: Implement the minimum to make the test pass**

Review `ExerciseSetCard.tsx` — the `scoreClip` section is already conditional (`{config.scoreClip && scoreClip && !clipLoadError && ...}`). The assign button is already gated on `{exercise.exerciseId && ...}`. No component code changes needed for the stub path.

If the test runner requires `@testing-library/react` setup, ensure it is installed:
```bash
cd /Users/jdhiman/Documents/crescendai/apps/web && bun add -d @testing-library/react @testing-library/jest-dom jsdom
```

Verify `vitest.config.ts` has `environment: "jsdom"` (or add it).

The component test file is the implementation for this task.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/web && bun test --run src/components/cards/ExerciseSetCard.test.tsx
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai && git add apps/web/src/components/cards/ExerciseSetCard.test.tsx && git commit -m "test(#29): add ExerciseSetCard corpus_drill stub rendering test"
```

---

### Task 10: Eval render update (proposed_exercises → prescribed_exercise)

**Group:** F (depends on Groups D + E, parallel with Task 9)

**Behavior being verified:** `uv run pytest apps/evals/teaching_knowledge/test_run_eval_atomic_gate.py` passes without KeyError on `proposed_exercises`.

**Interface under test:** The eval's handling of synthesis result artifacts in `run_eval.py` (the rendering path that formats the artifact for the judge).

**Files:**
- Modify: `apps/evals/teaching_knowledge/run_eval.py`

- [ ] **Step 1: Write the failing test**

The existing `test_run_eval_atomic_gate.py` is the test. Run it:

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest teaching_knowledge/test_run_eval_atomic_gate.py -x -v 2>&1 | head -40
```

Read `test_run_eval_atomic_gate.py` to understand what it exercises and where `proposed_exercises` is referenced.

Also check `run_eval.py` for any direct access to `artifact["proposed_exercises"]`:

```bash
grep -n "proposed_exercises\|prescribed_exercise" apps/evals/teaching_knowledge/run_eval.py
```

If there are zero matches (as confirmed above), the issue is in the `SynthesisArtifact` shape returned by the DO WS path (`pipeline_client.py` / `SynthesisResult`). The `run_eval.py` test reads `synthesis.text` not `artifact[...]` fields directly. No code change may be needed.

Confirm by running the gate test. If it passes, this task is a no-op verification. If it fails, locate the KeyError and patch.

- [ ] **Step 2: Run test — verify current state**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest teaching_knowledge/test_run_eval_atomic_gate.py -x -v 2>&1 | head -40
```
Expected: if it passes without changes, document that. If it fails with a field access error, proceed to Step 3.

- [ ] **Step 3: Implement the minimum to make the test pass**

If `run_eval.py` has any code that reads `artifact["proposed_exercises"]` (check all files it imports too), update it to use `artifact.get("prescribed_exercise")` and render as prose:

```python
# In whatever function renders synthesis artifact fields:
prescribed = artifact.get("prescribed_exercise")
if prescribed is not None:
    kind = prescribed.get("kind", "exercise")
    dim = prescribed.get("target_dimension", "")
    bar_range = prescribed.get("bar_range", [])
    parts.append(f"Prescribed exercise: {kind} targeting {dim}, bars {bar_range[0]}-{bar_range[1]}.")
```

If the eval accesses `proposed_exercises` anywhere in the test chain, replace with `.get("prescribed_exercise")` with the above rendering.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest teaching_knowledge/test_run_eval_atomic_gate.py -x -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai && git add apps/evals/teaching_knowledge/run_eval.py && git commit -m "fix(#29): update eval render to use prescribed_exercise field"
```

If no changes needed: skip commit, document in PR.

---

### Task 11: Fixture sweep — update proposed_exercises in all remaining test files

**Group:** G (depends on all previous tasks)

**Behavior being verified:** `bun test --run` in `apps/api` passes with zero TypeScript errors related to `proposed_exercises` field not existing on `SynthesisArtifact`.

**Interface under test:** Full test suite compilation and execution.

**Files (modify all fixtures that reference proposed_exercises):**
- `apps/api/src/harness/loop/runHook.test.ts` — update `VALID_ARTIFACT` const
- `apps/api/src/harness/loop/phase2-schema.test.ts` — update `VALID_ARTIFACT` const
- `apps/api/src/harness/skills/__catalog__/integration.test.ts` — update `VALID_SYNTHESIS_ARTIFACT` const
- `apps/api/src/services/teacher-synthesize-v6.test.ts` — update `VALID_ARTIFACT` const
- `apps/api/src/services/teacher.test.ts` — update `V6_VALID_ARTIFACT` const

- [ ] **Step 1: Write the failing test** (the full suite is the test)

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun test --run 2>&1 | grep -E "proposed_exercises|FAIL|ERROR" | head -30
```
Expected: TypeScript errors in multiple files referencing `proposed_exercises` which no longer exists on `SynthesisArtifact`.

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun test --run 2>&1 | grep -c "FAIL"
```
Expected: non-zero count.

- [ ] **Step 3: Implement the minimum to make the test pass**

In each file listed above, find every occurrence of:
```typescript
proposed_exercises: [...],
```
and replace with:
```typescript
prescribed_exercise: null,
```

For any test that specifically asserts on `proposed_exercises` content (e.g., checks the array length), update or remove the assertion. The exercise content is now tested through `prescribed_exercise` assertions in Task 2's test file.

Exact replacements needed:

`apps/api/src/harness/loop/runHook.test.ts`:
```typescript
// Replace:
proposed_exercises: [],
// With:
prescribed_exercise: null,
```

`apps/api/src/harness/loop/phase2-schema.test.ts`:
```typescript
// Replace:
proposed_exercises: [],
// With:
prescribed_exercise: null,
```

`apps/api/src/harness/skills/__catalog__/integration.test.ts`:
```typescript
// Replace:
proposed_exercises: [],
// With:
prescribed_exercise: null,
```

`apps/api/src/services/teacher-synthesize-v6.test.ts`:
```typescript
// Replace:
proposed_exercises: [],
// With:
prescribed_exercise: null,
```

`apps/api/src/services/teacher.test.ts`:
```typescript
// Replace (both occurrences):
proposed_exercises: [],
// With:
prescribed_exercise: null,
```

Also check and update `apps/api/src/services/prompts.test.ts` if it references `proposed_exercises`.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun test --run
```
Expected: all tests PASS, zero TypeScript errors on `proposed_exercises`.

Also run web tests:
```bash
cd /Users/jdhiman/Documents/crescendai/apps/web && bun test --run
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai && git add apps/api/src/harness/loop/runHook.test.ts apps/api/src/harness/loop/phase2-schema.test.ts apps/api/src/harness/skills/__catalog__/integration.test.ts apps/api/src/services/teacher-synthesize-v6.test.ts apps/api/src/services/teacher.test.ts && git commit -m "chore(#29): sweep proposed_exercises fixtures to prescribed_exercise across all test files"
```

---

## Final Verification

```bash
# API full suite
cd /Users/jdhiman/Documents/crescendai/apps/api && bun test --run

# Web full suite
cd /Users/jdhiman/Documents/crescendai/apps/web && bun test --run

# Eval gate
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest teaching_knowledge/test_run_eval_atomic_gate.py -x -v

# TypeScript compilation check
cd /Users/jdhiman/Documents/crescendai/apps/api && bun tsc --noEmit
cd /Users/jdhiman/Documents/crescendai/apps/web && bun tsc --noEmit
```

All must pass before the branch is ready for `/review`.

## Success Criteria (from spec)

- After a session and in chat the teacher emits a structured `{kind,...}` decision
- `own_passage_loop` renders via `ExerciseSetCard` `scoreClip`
- `corpus_drill` stub-renders as text
- Reflect-then-prescribe gate (#27) intact
- `segment_loop` stack untouched
- `exercises` catalog no longer polluted by synthesis or chat tool calls
- All `bun test --run` green in `apps/api` + `apps/web`
- Eval gate green
- `#37` closeable (clean exercise catalog)

---

## Challenge Review

### CEO Pass

#### 1. Premise Challenge

Right problem. The current `stageDominantExercise` writes three rows per synthesis (into `exercises`, `exerciseDimensions`, `pending_exercises`) and the old `create_exercise` chat tool also pollutes `exercises`. Both are confirmed in the source. The spec correctly identifies all four debt items. No simpler framing exists — the schema migration and the route rewrite must happen together.

Direct path: yes. The plan targets the exact files that own the corrupt behavior and nothing else. Verified against source.

#### 2. Scope Check

Scope is tight for a system with this many interdependencies — 11 task groups, ~25 files. However, 8 of those files are fixture sweeps (Task 11), not behavioral changes. The behavioral surface is: new Zod schema (Task 1), synthesis schema swap (Task 2), prompt text change (Task 3), DB migration + service rewrite (Task 4), DO wiring (Task 5), assignPendingExercise rewrite (Task 6), dead code deletion (Task 7), new chat tool (Task 8), web component test (Task 9), eval no-op check (Task 10). That is the minimum needed to close both pollution paths simultaneously.

One item can be deferred without losing the core goal: Task 9 (ExerciseSetCard test). The component already handles missing `scoreClip` and missing `exerciseId` (verified in source at lines 120, 151). The test would pass without any code changes. It adds coverage value but is not a blocker for catalog cleanliness.

#### 3. Twelve-Month Alignment

```
CURRENT STATE                            THIS PLAN                              12-MONTH IDEAL
proposed_exercises: string[]      →      prescribed_exercise: {kind,...}   →    Full corpus_drill
Pollutes exercises catalog                No catalog pollution                   asset pipeline live
create_exercise DB writes                 prescribe_exercise in-memory only      Own-passage audio
No structured routing                     Structured routing contract            playback (S2)
```

This plan moves squarely toward the ideal. It introduces no tech debt — the structured `ExerciseRoutingDecision` is the foundation S2/S3/S4 build on.

#### 4. Alternatives Check

The spec documents a deliberate "no DB writes in chat path" decision for S1 with deferred persistence to S3+. The tradeoff (no chat-path exercise persistence) is stated in the spec's Open Questions. This is an owner-approved decision and is not re-litigated here.

---

### Engineering Pass

#### 5. Architecture

Data flow for synthesis path (verified against source):

```
SynthesisArtifact.prescribed_exercise (non-null)
    → session-brain.ts ~line 1798
    → stageDominantExercise(db, { routing: prescribed_exercise, ... })
    → INSERT INTO pending_exercises (title, instruction, routing_json, ...)
    → buildPendingExerciseComponent({ exerciseId: inserted.id, ... })
    → buildV6WsPayload → WebSocket → web ReflectionMessage

Student clicks Confirm →
    api.exercises.assignPending({ sessionId, exerciseId: pendingConfig.exerciseId })
    → POST /api/exercises/assign-pending
    → assignPendingExercise(ctx, { exerciseId: req.exerciseId })
    → SELECT FROM pending_exercises WHERE exerciseId = args.exerciseId  ← BROKEN (see BLOCKER-1)
```

**[BLOCKER-1] found:** The existing route at `apps/api/src/routes/exercises.ts` line 65-73 calls `assignPendingExercise` with `exerciseId` from the request body. The current `assignPendingExercise` queries `WHERE pending_exercises.exerciseId = args.exerciseId`. The plan rewrites `assignPendingExercise` to query `WHERE pending_exercises.id = args.exerciseId` (line 1077 of the plan's Task 6 implementation), but the schema column the plan makes nullable is `pending_exercises.exerciseId` (the old FK to `exercises.id`), not the `id` primary key.

The web client (`ReflectionMessage.tsx` line 31) passes `pendingConfig.exerciseId` to the API, and `pendingConfig.exerciseId` is populated from `buildPendingExerciseComponent(staged)` where `staged.exerciseId = inserted.id` (the pending row's primary key).

So after Task 4's rewrite, `stageDominantExercise` returns the pending row's `id` as `exerciseId`, the web client sends it as `exerciseId`, and the route passes it as `exerciseId`. The new `assignPendingExercise` queries `WHERE pending_exercises.id = args.exerciseId`. This actually closes correctly — the round-trip works.

However, the `assignPendingSchema` in the route (`apps/api/src/routes/exercises.ts` line 26-29) only accepts `sessionId` and `exerciseId` — it does NOT accept `pieceId`. The Task 6 implementation of `assignPendingExercise` takes a `pieceId: string | null` arg, but the route never sends it. This means `assignPendingExercise` will always receive `pieceId: null` from the HTTP path, so `own_passage_loop` exercises from the synthesis confirm-gate will always render without a `scoreClip`.

The plan's Task 6 test exercises this via mock context, not through the actual route — so the test passes but the behavior through the real path will be wrong.

**Security:** `routingJson` is written to the DB as LLM output and read back by `ExerciseRoutingDecisionSchema.safeParse(pendingRow.routingJson)`. Zod validates it before use. No injection vector. Safe.

**N+1:** No new unbounded queries introduced.

#### 6. Module Depth Audit

- `exercise-routing.ts` — Interface: 3 exports (schema + 2 types). Implementation: discriminated union with 3 refinements, bar_range constraint, tempo_factor bounds. **DEEP.**
- `synthesis.ts` (modified) — Interface: schema + type. Implementation: complex Zod object with 2 cross-field refinements. **DEEP.**
- `pending-exercise.ts` (rewritten) — Interface: 2 functions + 1 type. Implementation: DB insert with title/instruction generation logic. **DEEP.**
- `exercises.ts` (`assignPendingExercise` rewrite) — Interface: 1 function. Implementation: DB select, Zod parse, branch on kind, conditional scoreClip. **DEEP.**
- `tool-processor.ts` (`prescribe_exercise` addition) — Interface: unchanged registry + process function. Implementation: two branches (own_passage_loop / corpus_drill) building InlineComponent. **DEEP.**

#### 7. Code Quality

**Task 4 schema inconsistency (RISK):** The plan's `pendingExercises` schema block (Task 4, Step 3) includes `exerciseId: uuid("exercise_id").notNull()` in one place, then a note to make it nullable. The final proposed schema has `exerciseId: uuid("exercise_id")` with no `.notNull()`. However the plan also keeps `exerciseId` in the insert in `stageDominantExercise` using `args.studentId + "-" + args.sessionId + "-pending"` — a non-UUID string — as its value. If the column type is `uuid` (Postgres), inserting a non-UUID string will throw a `22P02 invalid input syntax` error at runtime.

**Task 6 `assignExercise` call:** The plan's `assignPendingExercise` implementation calls `assignExercise(ctx, { ... exerciseId: args.exerciseId })` at line 1102-1106, then notes in the prose to skip it with a TODO. The final implementation omits this call but the prose is contradictory — the build agent may implement it and hit the FK violation (since there is no `exercises` row for the pending row ID).

**`DIMS_6` cast:** `DIMS_6 as unknown as [string, ...string[]]` is the existing pattern; the plan replicates it correctly.

**`exerciseId` in `ExerciseSetPayload`:** The existing type has `exerciseId: string` (non-optional). The plan's Task 6 implementation returns `exerciseId: pendingRow.id` for both paths — correct. But the type definition at line 1049 still declares `exerciseId: string` (required), whereas corpus_drill exercises from chat (`processPrescribeExercise`) do not include `exerciseId` in their config. This creates a type inconsistency: the chat path produces components that fail the `ExerciseSetPayload` type. The web `ExerciseSetCard` already handles missing `exerciseId` gracefully (gated at line 120), but the TypeScript type needs `exerciseId?: string` to be optional.

#### 8. Test Philosophy Audit

**Task 4 test — shape test warning (RISK, confidence 7/10):** The test for `stageDominantExercise` (Task 4, Step 1) does not actually call `stageDominantExercise`. It constructs a mock return value manually and calls `buildPendingExerciseComponent` on it. The `"inserts into pending_exercises with routing_json..."` test name claims to verify DB behavior but verifies nothing about the DB insert path. The second test (`"stageDominantExercise args include routing not proposedExercise (TypeScript shape)"`) constructs `Args` directly without calling the function. Both tests would pass even if `stageDominantExercise` still wrote to `exercises`. This is a shape test, not a behavior test.

The plan acknowledges this: "The actual runtime test is in the worker integration layer." But the worker integration test is not written here — it is deferred. There is no existing integration test for `stageDominantExercise`. This means the core invariant ("no writes to exercises") is untested.

**Task 8 test — processToolUse ctx is empty (RISK, confidence 6/10):** The `prescribe_exercise` test creates `ctx = { db: {} } as unknown as ServiceContext`. Since `processPrescribeExercise` does not use `ctx` at all (no DB writes in S1), this is fine and the test will correctly verify behavior.

**All other tests:** Behavioral, through public interfaces, correct vertical slices.

#### 9. Vertical Slice Audit

All 11 tasks follow the one-test → one-impl → one-commit structure. No horizontal slicing found.

The one exception is Task 3 Step 1, which instructs updating `VALID_ARTIFACT` in `phase2.test.ts` (removing `proposed_exercises`, adding `prescribed_exercise: null`) alongside writing the new prompt test. This is minor: the fixture update is a prerequisite for the new test to compile, not a separate behavior. Acceptable.

Task 7 (dead code deletion) writes tests before deleting code, then deletes. This is the correct order — the test fails with current 9-length, passes after deletion. Correct vertical slice.

#### 10. Test Coverage Gaps

```
[+] exercise-routing.ts
    ├── ExerciseRoutingDecisionSchema.parse()
    │   ├── [TESTED] own_passage_loop valid — Task 1
    │   ├── [TESTED] own_passage_loop bar_range start > end — Task 1
    │   ├── [TESTED] own_passage_loop tempo_factor bounds — Task 1
    │   ├── [TESTED] corpus_drill valid — Task 1
    │   └── [TESTED] unknown kind — Task 1 ★★★

[+] pending-exercise.ts (stageDominantExercise rewrite)
    ├── happy path (routing_json written, no exercises insert)
    │   └── [GAP] no behavior test — Task 4 test only checks TypeScript shape
    ├── InferenceError on failed insert
    │   └── [GAP] untested
    └── title truncation at 60 chars
        └── [GAP] untested

[+] exercises.ts (assignPendingExercise rewrite)
    ├── own_passage_loop + pieceId → scoreClip present — Task 6 [TESTED ★★]
    ├── own_passage_loop + no pieceId → no scoreClip — Task 6 [TESTED ★★]
    ├── corpus_drill → no scoreClip, "coming soon" text — Task 6 [TESTED ★★]
    ├── pending row not found → NotFoundError
    │   └── [GAP] untested
    ├── routing_json null → InferenceError
    │   └── [GAP] untested
    └── routing_json invalid JSON Schema → InferenceError
        └── [GAP] untested (★ shape only if added)

[+] route /api/exercises/assign-pending
    ├── pieceId NOT wired from route to service
    │   └── [GAP] own_passage_loop from the real HTTP path always has pieceId=null
    │          (scoreClip never renders for synthesis-path exercises)
```

The `stageDominantExercise` gap is significant because it is the primary invariant this plan was created to enforce. Without a behavior test, a future change could re-introduce catalog writes silently.

#### 11. Failure Modes

**Task 4 migration:** The migration adds nullable columns to `pending_exercises`. If the migration runs but the deploy fails, the new columns exist in the DB but the code still writes the old shape. This is safe — old code writes `exerciseId` (FK), new columns are nullable with no NOT NULL constraint.

**`stageDominantExercise` with non-UUID `exerciseId` value:** The plan constructs `exerciseId` as `args.studentId + "-" + args.sessionId + "-pending"`. The column type is `uuid`. Postgres will reject this with a type error unless the column type is changed to `text`. The plan's final schema shows `exerciseId: uuid("exercise_id")` (nullable) which still requires a valid UUID if non-null. **This is a runtime crash path** — see BLOCKER-2.

**`own_passage_loop` with no `scoreClip` from HTTP path:** Not a crash. The `ExerciseSetCard` renders text-only gracefully. But it is a behavioral regression from what users would expect (the whole point of `own_passage_loop` is a score clip).

**`assignExercise` call skipped:** The plan says to skip the `studentExercises` insert in S1 with a TODO. If the build agent follows the plan verbatim (not the prose note), `assignExercise` is called with a pending row `id` (not a catalog `exercises.id`), hitting a FK violation on `student_exercises.exercise_id` references `exercises.id`. The migration must also drop this FK or the call must be omitted. The plan is ambiguous here.

---

### Presumption Inventory

| Assumption | Verdict | Reason |
|---|---|---|
| `DIMS_6` is importable from `../../lib/dims` in the new `exercise-routing.ts` | SAFE | File confirmed at `apps/api/src/lib/dims.ts` with named export `DIMS_6` |
| `ExerciseSetCard` already handles missing `scoreClip` | SAFE | Verified: line 151 `if (!config.scoreClip) return` and line 166 `{config.scoreClip && ...}` |
| `ExerciseSetCard` already handles missing `exerciseId` | SAFE | Verified: line 120 `{exercise.exerciseId && ...}` |
| `molecule-exercise-proposal.test.ts` uses `validateSkill` against a harness catalog doc | SAFE | Confirmed — deleting the test file is sufficient; no validator re-count assertion exists for molecules in `integration.test.ts` |
| Task 10 eval gate (`test_run_eval_atomic_gate.py`) has no `proposed_exercises` access | SAFE | Verified by grep — zero matches in gate test and in `run_eval.py` |
| `teacher-synthesize-v6.test.ts` has `proposed_exercises` fixture | SAFE | Confirmed at line 17, needs Task 11 sweep |
| `pendingExercises.exerciseId` is a FK to `exercises.id` | SAFE | Verified: no explicit FK in the Drizzle schema definition (no `.references()`), but the column exists and is used in queries |
| `studentExercises.exerciseId` is NOT an explicit FK in Drizzle schema | SAFE | Verified: `apps/api/src/db/schema/exercises.ts` line 55 — `exerciseId: uuid("exercise_id").notNull()` with no `.references()` call. No cascade constraint. Skipping the `assignExercise` call in S1 is safe. |
| `processToolUse` in chat path has no DB calls for `prescribe_exercise` | SAFE | The new handler uses no `ctx.db` — correct for S1 |
| Route `/assign-pending` passes `pieceId` to `assignPendingExercise` | RISKY | Route does NOT pass `pieceId` — the service signature accepts it but the route never sends it. `own_passage_loop` scoreClip will be absent on the confirmed exercise path. |
| `exerciseId` field in `stageDominantExercise` can hold `studentId+sessionId` string as a UUID column | RISKY | Postgres `uuid` type rejects non-UUID strings. Runtime crash on first synthesis with a valid prescription. |

---

### Summary

| Finding | Count |
|---|---|
| [BLOCKER] | 2 |
| [RISK] | 4 |
| [QUESTION] | 1 |

**[BLOCKER] (confidence 9/10) — Task 4, `stageDominantExercise` implementation: `exerciseId` column is typed `uuid` in Postgres but the plan constructs its value as `args.studentId + "-" + args.sessionId + "-pending"` (a non-UUID string). Inserting this into a `uuid` column will throw `22P02 invalid input syntax for type uuid` at runtime on the first synthesis with a valid `prescribed_exercise`. Fix: either drop the `exerciseId` column from the `pendingExercises` table entirely in this migration (it is no longer a FK and its only prior purpose was to join back to `exercises`) or make it `text` type. The simplest fix is to remove the `exerciseId` column from the Drizzle schema in Task 4 and from the migration (drop it), since `stageDominantExercise` no longer needs it and `assignPendingExercise` now queries by `pending_exercises.id`.**

**[BLOCKER] (confidence 8/10) — Task 6, route/service contract gap: The `/api/exercises/assign-pending` route (`apps/api/src/routes/exercises.ts` lines 26-73) passes `{ studentId, sessionId, exerciseId }` to `assignPendingExercise`. The Task 6 implementation adds `pieceId: string | null` to the service args, but the route never supplies it — it is always effectively `null`. This means `own_passage_loop` exercises confirmed through the ReflectionMessage flow will always produce `scoreClip: undefined`, defeating the primary UX purpose of `own_passage_loop`. The plan must update `assignPendingSchema` and the route handler to accept and forward an optional `pieceId` from the client, OR derive `pieceId` inside the service from the `pending_exercises` row (requires storing `pieceId` at staging time in `stageDominantExercise`). The client (`ReflectionMessage.tsx`) would also need to send `pieceId` if it is to be sourced from the web side. This is a missing step that no task in the plan addresses.**

**[RISK] (confidence 8/10) — Task 4 test is a shape test, not a behavior test: `stageDominantExercise` is never actually called in the Task 4 test. The core invariant ("no writes to `exercises`") is unverified. No worker integration test is scheduled. The existing `services/pending-exercise.test.ts` (if it has a worker integration harness) should be checked; if it does not, add at minimum a unit test that mocks `db.insert` and asserts it is only called once (for `pending_exercises`), not three times.**

**[RISK] (confidence 7/10) — Task 6 `assignExercise` call ambiguity: the plan's Step 3 prose at line 1171-1177 says to guard the `assignExercise` call with a comment/skip for S1, but the preceding code block at line 1102 still includes the call. The build agent will follow the code block, calling `assignExercise(ctx, { exerciseId: args.exerciseId })` where `args.exerciseId` is a pending row `id` (UUID, but not in `exercises` table). While `studentExercises` has no Drizzle-level FK to `exercises`, `assignExercise` first does `ctx.db.query.exercises.findFirst(...)` at line 96-98 and throws `NotFoundError` if not found. This will throw on every confirm attempt. The plan must explicitly remove the `assignExercise` call from the new `assignPendingExercise` body.**

**[RISK] (confidence 6/10) — `ExerciseSetPayload.exercises[].exerciseId` type: the existing type at `apps/api/src/services/exercises.ts` line 19 declares `exerciseId: string` (required). The chat-path `processPrescribeExercise` in Task 8 returns exercise objects without `exerciseId`. TypeScript will error unless `exerciseId` is made optional (`exerciseId?: string`). The plan does not update this type definition. The build agent will hit a TS compilation error in Task 8.**

**[QUESTION] — Where should `pieceId` be stored for the synthesis confirm path? Options: (A) store `pieceId` in `pending_exercises` at staging time (`stageDominantExercise` receives `pieceCtx` which has `pieceId` — already available); (B) derive it from the session state at `assignPendingExercise` time via a DB join; (C) have the web client supply it at confirm time (requires web changes). Option A is the simplest: add a nullable `pieceId text` column to `pending_exercises` in the Task 4 migration, populate it in `stageDominantExercise`, read it back in `assignPendingExercise`. This also resolves BLOCKER-2 without requiring route or web changes. The plan must decide and document this.**

VERDICT: NEEDS_REWORK — Two blockers must be resolved before execution: (1) the `exerciseId uuid` column crash from a non-UUID string insert in `stageDominantExercise`, and (2) the missing `pieceId` propagation through the `/assign-pending` route that would make `own_passage_loop` always render without a score clip. Both are fixable with small targeted changes: drop or retype `exerciseId` in the schema, and store `pieceId` in `pending_exercises` at staging time (the `pieceCtx` object is already available in the DO call site). Two additional risks (the `assignExercise` NotFoundError and the `exerciseId?: string` type omission) will also cause build failures and should be fixed in the same rework pass.
