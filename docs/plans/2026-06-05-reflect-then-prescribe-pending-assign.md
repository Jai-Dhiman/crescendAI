# Plan B: reflect-then-prescribe — Schema + Service + Route (pending-assign)

> **Build-agent dispatch note:** This is Plan B of 4 for the reflect-then-prescribe feature
> (issue #27, branch `issue-27-reflect-then-prescribe`). Execute Tasks 1–3 sequentially; each
> task follows the 5-step shape (failing test → verify fail → impl → verify pass → commit).
> Fresh Sonnet 4.6 subagent per task group. After all three tasks are green, this branch is
> shippable independently — no caller exists yet.

**Goal:** Add the `pendingExercises` Drizzle table, the `assignPendingExercise` service
function, the `ExerciseSetPayload` type, and the `POST /assign-pending` route — the API half
of the ownership-gated exercise reveal.

**Spec path:** `docs/specs/2026-06-05-reflect-then-prescribe-design.md`

**Ship-guard note:** Do NOT delete the shared spec until Plan D (web) ships. This plan
implements only the API surface; the spec is still needed by Plans C and D.

**Style:** Follow `apps/api/TS_STYLE.md` throughout. Key rules for this plan:
- `ServiceContext` for DI (`{ db, env }`) — never individual args.
- Domain errors in services (`NotFoundError`), never `HTTPException`.
- Chain `.post("/assign-pending", ...)` onto the existing `exercisesRoutes` expression.
- `console.log(JSON.stringify({...}))` for any new logging.
- State versioning across awaits where applicable (not needed here — no DO state).

**Dependency note:** Must merge BEFORE Plan D (web). Plan C (DO persistence) may run in
parallel; Plan B introduces no DO changes and Plan C introduces no route changes.

**[SHIPS INDEPENDENTLY]** — a new table + endpoint with no caller yet; safe to land on main
before the web surface exists.

---

## Shared Contract (identical across all 4 plans — do NOT rename)

### `pendingExercises` table (db name `pending_exercises`)

| Column | Drizzle type | Constraints |
|---|---|---|
| `id` | `uuid("id").defaultRandom().primaryKey()` | PK |
| `studentId` | `text("student_id").notNull()` | |
| `sessionId` | `uuid("session_id").notNull()` | |
| `exerciseId` | `uuid("exercise_id").notNull()` | |
| `focusDimension` | `text("focus_dimension").notNull()` | |
| `previewTitle` | `text("preview_title").notNull()` | |
| `consumed` | `boolean("consumed").notNull().default(false)` | |
| `createdAt` | `timestamp("created_at", { withTimezone: true }).notNull().defaultNow()` | |

Constraints:
- `uniqueIndex("idx_pending_exercises_unique")` on `(studentId, sessionId, exerciseId)`
- `index("idx_pending_exercises_lookup")` on `(studentId, consumed)`

### `ExerciseSetPayload` type

```typescript
export type ExerciseSetPayload = {
  sourcePassage: string;
  targetSkill: string;
  exercises: Array<{
    title: string;
    instruction: string;
    focusDimension: string;
    hands?: "left" | "right" | "both";
    exerciseId: string;
  }>;
};
```

Exported from `apps/api/src/services/exercises.ts`.

### `assignPendingExercise` contract

```typescript
assignPendingExercise(
  ctx: ServiceContext,
  args: { studentId: string; sessionId: string; exerciseId: string }
): Promise<ExerciseSetPayload>
```

- Query `pendingExercises` for an **unconsumed** row matching all three
  (`studentId`, `sessionId`, `exerciseId`).
- If none → `throw new NotFoundError("pending exercise", args.exerciseId)`.
- Else call `assignExercise(ctx, { studentId, exerciseId, sessionId })`.
- UPDATE the pending row `set consumed = true`.
- Assemble `ExerciseSetPayload` from the `exercises` row + its `exerciseDimensions`.
- Return payload.
- **IDOR guard:** `studentId` comes from auth (never from request body). A foreign
  `exerciseId` finds no matching row → `NotFoundError`.

### Route contract

`POST /assign-pending` in `apps/api/src/routes/exercises.ts`:
- `validate("json", z.object({ sessionId: z.string().uuid(), exerciseId: z.string().uuid() }))`
- `requireAuth(c.var.studentId)` (throws `HTTPException(401)` → 401)
- `assignPendingExercise({ db: c.var.db, env: c.env }, { studentId: c.var.studentId, sessionId, exerciseId })`
- Returns `c.json(payload)` (200).
- `NotFoundError` → **404** via the error handler in
  `apps/api/src/middleware/error-handler.ts` (line 16–18: `instanceof NotFoundError →
  c.json({ error: err.message }, 404)`).

---

## Task Groups

### Task 1 — `pendingExercises` table + migration

**Success criteria:** `bun run generate` succeeds and the generated migration SQL contains
`CREATE TABLE "pending_exercises"` with all eight columns and both indexes.

**Step 1 — Write the failing "test" (migration verification plan)**

The "test" for a schema change is the generate command. Before touching schema, confirm the
table does not yet exist:

```bash
grep -r "pending_exercises" apps/api/src/db/schema/ apps/api/drizzle/
# expect: no matches
```

**Step 2 — Verify fail**

```bash
cd apps/api && bun run generate
# expect: succeeds, but grep below finds nothing
grep -r "pending_exercises" drizzle/
# expect: no matches (table not yet declared)
```

**Step 3 — Implement: add table to `apps/api/src/db/schema/exercises.ts`**

Append after the closing brace of `studentExercises` (after line 75 in the current file):

```typescript
export const pendingExercises = pgTable(
  "pending_exercises",
  {
    id: uuid("id").defaultRandom().primaryKey(),
    studentId: text("student_id").notNull(),
    sessionId: uuid("session_id").notNull(),
    exerciseId: uuid("exercise_id").notNull(),
    focusDimension: text("focus_dimension").notNull(),
    previewTitle: text("preview_title").notNull(),
    consumed: boolean("consumed").notNull().default(false),
    createdAt: timestamp("created_at", { withTimezone: true })
      .notNull()
      .defaultNow(),
  },
  (t) => [
    uniqueIndex("idx_pending_exercises_unique").on(
      t.studentId,
      t.sessionId,
      t.exerciseId,
    ),
    index("idx_pending_exercises_lookup").on(t.studentId, t.consumed),
  ],
);
```

All imports (`boolean`, `index`, `pgTable`, `text`, `timestamp`, `uniqueIndex`, `uuid`) are
already present at the top of the file — no new imports needed.

**Step 4 — Verify pass**

```bash
cd apps/api && bun run generate
# expect: exit 0, new migration file created

grep -r "pending_exercises" drizzle/
# expect: at least one .sql file containing the table name

# Exact verification grep:
grep "CREATE TABLE \"pending_exercises\"" drizzle/*.sql
# expect: one matching line

grep "idx_pending_exercises_unique\|idx_pending_exercises_lookup" drizzle/*.sql
# expect: two matching lines (one per index)
```

**Step 5 — Commit**

```
git add apps/api/src/db/schema/exercises.ts apps/api/drizzle/
git commit -m "feat(schema): add pending_exercises table for reflect-then-prescribe"
```

---

### Task 2 — `ExerciseSetPayload` type + `assignPendingExercise` service

**Success criteria:** `cd apps/api && bunx vitest run --config vitest.node.config.ts src/services/exercises.test.ts` passes all cases:
(a) returns `ExerciseSetPayload` and marks the pending row consumed when an unconsumed owned row exists,
(b) throws `NotFoundError` when no matching row (covers IDOR via foreign `exerciseId`).

**Step 1 — Write failing tests**

Create `apps/api/src/services/exercises.test.ts`:

```typescript
import { and, eq } from "drizzle-orm";
import { describe, expect, it, vi } from "vitest";
import { pendingExercises } from "../db/schema/exercises";
import { NotFoundError } from "../lib/errors";
import { assignPendingExercise } from "./exercises";

// ---------------------------------------------------------------------------
// Shared fixtures
// ---------------------------------------------------------------------------

const STUDENT_ID = "student-abc";
const SESSION_ID = "00000000-0000-0000-0000-000000000010";
const EXERCISE_ID = "00000000-0000-0000-0000-000000000001";
const FOREIGN_EXERCISE_ID = "00000000-0000-0000-0000-000000000099";

const PENDING_ROW = {
  id: "00000000-0000-0000-0000-000000000020",
  studentId: STUDENT_ID,
  sessionId: SESSION_ID,
  exerciseId: EXERCISE_ID,
  focusDimension: "pedaling",
  previewTitle: "Pedal Separation Drill",
  consumed: false,
  createdAt: new Date(),
};

const EXERCISE_ROW = {
  id: EXERCISE_ID,
  title: "Pedal Separation Drill",
  description: "Practice separating pedal changes.",
  instructions: "Play each phrase with clean pedal lifts.",
  difficulty: "intermediate",
  category: "technique",
  repertoireTags: null,
  notationContent: null,
  notationFormat: null,
  midiContent: null,
  source: "generated",
  variantsJson: null,
  createdAt: new Date(),
};

const DIMENSION_ROWS = [{ exerciseId: EXERCISE_ID, dimension: "pedaling" }];

// ---------------------------------------------------------------------------
// Helpers to build a mock ctx.db
// ---------------------------------------------------------------------------

function makeCtx({
  pendingRow,
  exerciseRow,
  dimensionRows,
}: {
  pendingRow: typeof PENDING_ROW | null;
  exerciseRow: typeof EXERCISE_ROW | null;
  dimensionRows: typeof DIMENSION_ROWS;
}) {
  // assignExercise internals: ctx.db.query.exercises.findFirst + insert into studentExercises
  // assignPendingExercise internals: select from pendingExercises + update pendingExercises
  //                                  + select from exercises + select from exerciseDimensions

  const mockUpdate = vi.fn().mockReturnValue({
    set: vi.fn().mockReturnValue({
      where: vi.fn().mockResolvedValue(undefined),
    }),
  });

  const mockInsert = vi.fn().mockReturnValue({
    values: vi.fn().mockReturnValue({
      onConflictDoUpdate: vi.fn().mockReturnValue({
        returning: vi.fn().mockResolvedValue([{ id: "se-1", studentId: STUDENT_ID }]),
      }),
    }),
  });

  // select() chain used for pendingExercises lookup and exerciseDimensions lookup
  let selectCallCount = 0;
  const mockSelect = vi.fn().mockImplementation(() => {
    selectCallCount++;
    const callIndex = selectCallCount;
    return {
      from: vi.fn().mockReturnValue({
        where: vi.fn().mockResolvedValue(
          callIndex === 1
            ? pendingRow
              ? [pendingRow]
              : []
            : dimensionRows,
        ),
      }),
    };
  });

  const mockFindFirst = vi.fn().mockImplementation(({ where: _where }) => {
    // used by assignExercise to verify the exercise exists
    return Promise.resolve(exerciseRow);
  });

  const db = {
    select: mockSelect,
    update: mockUpdate,
    insert: mockInsert,
    query: {
      exercises: { findFirst: mockFindFirst },
      pendingExercises: {
        findFirst: vi.fn().mockResolvedValue(pendingRow),
      },
    },
  };

  return { db: db as never, env: {} as never };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("assignPendingExercise", () => {
  it("returns ExerciseSetPayload and marks pending row consumed for a valid owned row", async () => {
    const ctx = makeCtx({
      pendingRow: PENDING_ROW,
      exerciseRow: EXERCISE_ROW,
      dimensionRows: DIMENSION_ROWS,
    });

    const payload = await assignPendingExercise(ctx, {
      studentId: STUDENT_ID,
      sessionId: SESSION_ID,
      exerciseId: EXERCISE_ID,
    });

    // Payload shape
    expect(payload).toMatchObject({
      sourcePassage: expect.any(String),
      targetSkill: expect.any(String),
      exercises: expect.arrayContaining([
        expect.objectContaining({
          title: EXERCISE_ROW.title,
          instruction: EXERCISE_ROW.instructions,
          focusDimension: PENDING_ROW.focusDimension,
          exerciseId: EXERCISE_ID,
        }),
      ]),
    });

    // consumed=true update must have been called
    expect(ctx.db.update).toHaveBeenCalled();
  });

  it("throws NotFoundError when no unconsumed pending row matches (IDOR: foreign exerciseId)", async () => {
    const ctx = makeCtx({
      pendingRow: null,
      exerciseRow: EXERCISE_ROW,
      dimensionRows: [],
    });

    await expect(
      assignPendingExercise(ctx, {
        studentId: STUDENT_ID,
        sessionId: SESSION_ID,
        exerciseId: FOREIGN_EXERCISE_ID,
      }),
    ).rejects.toThrow(NotFoundError);
  });
});
```

**Step 2 — Verify fail**

```bash
cd apps/api && bunx vitest run --config vitest.node.config.ts src/services/exercises.test.ts
# expect: fails — assignPendingExercise and ExerciseSetPayload do not exist yet
```

**Step 3 — Implement**

In `apps/api/src/services/exercises.ts`:

1. Add import at top (after existing imports):

```typescript
import { and, eq } from "drizzle-orm";
import { pendingExercises } from "../db/schema/exercises";
```

(Note: `eq` may already be imported — check and add only what is missing. `sql` is
already imported; add `and` and the `pendingExercises` table import.)

2. Export the `ExerciseSetPayload` type (add after the existing imports, before `listExercises`):

```typescript
export type ExerciseSetPayload = {
  sourcePassage: string;
  targetSkill: string;
  exercises: Array<{
    title: string;
    instruction: string;
    focusDimension: string;
    hands?: "left" | "right" | "both";
    exerciseId: string;
  }>;
};
```

3. Add `assignPendingExercise` after `completeExercise`:

```typescript
export async function assignPendingExercise(
  ctx: ServiceContext,
  args: { studentId: string; sessionId: string; exerciseId: string },
): Promise<ExerciseSetPayload> {
  const [pendingRow] = await ctx.db
    .select()
    .from(pendingExercises)
    .where(
      and(
        eq(pendingExercises.studentId, args.studentId),
        eq(pendingExercises.sessionId, args.sessionId),
        eq(pendingExercises.exerciseId, args.exerciseId),
        eq(pendingExercises.consumed, false),
      ),
    );

  if (!pendingRow) {
    throw new NotFoundError("pending exercise", args.exerciseId);
  }

  await assignExercise(ctx, {
    studentId: args.studentId,
    exerciseId: args.exerciseId,
    sessionId: args.sessionId,
  });

  await ctx.db
    .update(pendingExercises)
    .set({ consumed: true })
    .where(eq(pendingExercises.id, pendingRow.id));

  const exerciseRow = await ctx.db.query.exercises.findFirst({
    where: (e, { eq: eqFn }) => eqFn(e.id, args.exerciseId),
  });

  const dimensionRows = await ctx.db
    .select()
    .from(exerciseDimensions)
    .where(eq(exerciseDimensions.exerciseId, args.exerciseId));

  const dims = dimensionRows.map((d) => d.dimension);

  return {
    sourcePassage: exerciseRow?.description ?? "",
    targetSkill: pendingRow.focusDimension,
    exercises: [
      {
        title: exerciseRow?.title ?? pendingRow.previewTitle,
        instruction: exerciseRow?.instructions ?? "",
        focusDimension: pendingRow.focusDimension,
        exerciseId: args.exerciseId,
        ...(dims.length === 1 && dims[0]
          ? {}
          : {}),
      },
    ],
  };
}
```

**Implementation note on imports:** The current `exercises.ts` imports `eq` and `sql` from
`drizzle-orm` (line 1). Add `and` to that import. Also import `pendingExercises` from the
schema (alongside the existing `exercises`, `exerciseDimensions`, `studentExercises` import).

**Exact import line after edit:**

```typescript
import { and, eq, sql } from "drizzle-orm";
import {
  exerciseDimensions,
  exercises,
  pendingExercises,
  studentExercises,
} from "../db/schema/exercises";
```

**Step 4 — Verify pass**

```bash
cd apps/api && bunx vitest run --config vitest.node.config.ts src/services/exercises.test.ts
# expect: 2 tests pass
```

**Step 5 — Commit**

```
git add apps/api/src/services/exercises.ts apps/api/src/services/exercises.test.ts
git commit -m "feat(exercises): add assignPendingExercise + ExerciseSetPayload"
```

---

### Task 3 — `POST /assign-pending` route

**Success criteria:** `cd apps/api && bunx vitest run src/routes/exercises.test.ts` passes all
three new cases plus the three existing 401 cases (6 total):
- 200 + payload on valid owned pending row
- 404 when service throws `NotFoundError`
- 401 when unauthenticated

**Error handler mapping (exact):**
`apps/api/src/middleware/error-handler.ts` line 16–18:
```typescript
if (err instanceof NotFoundError) {
  return c.json({ error: err.message }, 404);
}
```
→ `NotFoundError` surfaces as **HTTP 404**.

`requireAuth` throws `HTTPException(401)` which is caught at line 12–14:
```typescript
if (err instanceof HTTPException) {
  return err.getResponse();
}
```
→ unauthenticated requests surface as **HTTP 401**.

**Step 1 — Write failing tests**

Append to `apps/api/src/routes/exercises.test.ts`:

```typescript
import { Hono } from "hono";
import { describe, expect, it, vi } from "vitest";
import { errorHandler } from "../middleware/error-handler";
import { exercisesRoutes } from "./exercises";

// Wire the error handler so NotFoundError -> 404
const testApp = new Hono()
  .onError(errorHandler)
  .route("/api/exercises", exercisesRoutes);

// ---------------------------------------------------------------------------
// Existing 401 tests (preserved exactly)
// ---------------------------------------------------------------------------
describe("exercises routes", () => {
  it("GET /api/exercises returns 401 without auth", async () => {
    const res = await testApp.request("/api/exercises");
    expect(res.status).toBe(401);
  });

  it("POST /api/exercises/assign returns 401 without auth", async () => {
    const res = await testApp.request("/api/exercises/assign", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        exerciseId: "00000000-0000-0000-0000-000000000001",
      }),
    });
    expect(res.status).toBe(401);
  });

  it("POST /api/exercises/complete returns 401 without auth", async () => {
    const res = await testApp.request("/api/exercises/complete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        studentExerciseId: "00000000-0000-0000-0000-000000000001",
      }),
    });
    expect(res.status).toBe(401);
  });

  // ---------------------------------------------------------------------------
  // POST /assign-pending
  // ---------------------------------------------------------------------------
  it("POST /api/exercises/assign-pending returns 401 without auth", async () => {
    const res = await testApp.request("/api/exercises/assign-pending", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sessionId: "00000000-0000-0000-0000-000000000010",
        exerciseId: "00000000-0000-0000-0000-000000000001",
      }),
    });
    expect(res.status).toBe(401);
  });
});
```

For the 200 and 404 cases the route must call the service, so we use `vi.mock`. Add a
separate describe block that mocks the service module:

```typescript
import * as exercisesService from "../services/exercises";

describe("POST /api/exercises/assign-pending — authenticated", () => {
  const SESSION_ID = "00000000-0000-0000-0000-000000000010";
  const EXERCISE_ID = "00000000-0000-0000-0000-000000000001";

  // Inject a fake authenticated studentId via a before-route middleware
  function makeAuthApp(studentId: string) {
    return new Hono()
      .use("*", async (c, next) => {
        c.set("studentId", studentId);
        c.set("db", {} as never);
        await next();
      })
      .onError(errorHandler)
      .route("/api/exercises", exercisesRoutes);
  }

  it("returns 200 with ExerciseSetPayload on valid owned pending row", async () => {
    const payload = {
      sourcePassage: "Running passage bars 3-6",
      targetSkill: "pedaling",
      exercises: [
        {
          title: "Pedal Separation Drill",
          instruction: "Play with clean pedal lifts.",
          focusDimension: "pedaling",
          exerciseId: EXERCISE_ID,
        },
      ],
    };
    vi.spyOn(exercisesService, "assignPendingExercise").mockResolvedValueOnce(payload);

    const app = makeAuthApp("student-abc");
    const res = await app.request("/api/exercises/assign-pending", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sessionId: SESSION_ID, exerciseId: EXERCISE_ID }),
    });

    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body).toMatchObject(payload);
  });

  it("returns 404 when assignPendingExercise throws NotFoundError (IDOR: foreign exerciseId)", async () => {
    const { NotFoundError } = await import("../lib/errors");
    vi.spyOn(exercisesService, "assignPendingExercise").mockRejectedValueOnce(
      new NotFoundError("pending exercise", "00000000-0000-0000-0000-000000000099"),
    );

    const app = makeAuthApp("student-abc");
    const res = await app.request("/api/exercises/assign-pending", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sessionId: SESSION_ID,
        exerciseId: "00000000-0000-0000-0000-000000000099",
      }),
    });

    expect(res.status).toBe(404);
    const body = await res.json();
    expect(body).toHaveProperty("error");
  });
});
```

**Step 2 — Verify fail**

```bash
cd apps/api && bunx vitest run src/routes/exercises.test.ts
# expect: the three new POST /assign-pending tests fail (route does not exist yet)
```

**Step 3 — Implement**

In `apps/api/src/routes/exercises.ts`:

1. Add import for `assignPendingExercise` and `ExerciseSetPayload`:

```typescript
import {
  assignExercise,
  assignPendingExercise,
  completeExercise,
  listExercises,
} from "../services/exercises";
```

2. Add schema constant before the chained route expression:

```typescript
const assignPendingSchema = z.object({
  sessionId: z.string().uuid(),
  exerciseId: z.string().uuid(),
});
```

3. Chain `.post("/assign-pending", ...)` onto the existing `exercisesRoutes` expression
   (append after `.post("/complete", ...)`):

```typescript
  .post("/assign-pending", validate("json", assignPendingSchema), async (c) => {
    requireAuth(c.var.studentId);
    const { sessionId, exerciseId } = c.req.valid("json");
    const payload = await assignPendingExercise(
      { db: c.var.db, env: c.env },
      { studentId: c.var.studentId, sessionId, exerciseId },
    );
    return c.json(payload);
  });
```

**Full resulting route file** (complete, for build-agent clarity):

```typescript
import { Hono } from "hono";
import { z } from "zod";
import type { Bindings, Variables } from "../lib/types";
import { validate } from "../lib/validate";
import { requireAuth } from "../middleware/auth-session";
import {
  assignExercise,
  assignPendingExercise,
  completeExercise,
  listExercises,
} from "../services/exercises";

const assignSchema = z.object({
  exerciseId: z.string().uuid(),
  sessionId: z.string().uuid().optional(),
});

const completeSchema = z.object({
  studentExerciseId: z.string().uuid(),
  response: z.string().optional(),
  dimensionBeforeJson: z.unknown().optional(),
  dimensionAfterJson: z.unknown().optional(),
  notes: z.string().optional(),
});

const listQuerySchema = z.object({
  dimension: z.string().optional(),
  level: z.string().optional(),
  repertoire: z.string().optional(),
});

const assignPendingSchema = z.object({
  sessionId: z.string().uuid(),
  exerciseId: z.string().uuid(),
});

const exercisesRoutes = new Hono<{ Bindings: Bindings; Variables: Variables }>()
  .get("/", validate("query", listQuerySchema), async (c) => {
    requireAuth(c.var.studentId);
    const { dimension, level, repertoire } = c.req.valid("query");
    const result = await listExercises(
      { db: c.var.db, env: c.env },
      { studentId: c.var.studentId, dimension, level, repertoire },
    );
    return c.json(result);
  })
  .post("/assign", validate("json", assignSchema), async (c) => {
    requireAuth(c.var.studentId);
    const body = c.req.valid("json");
    const result = await assignExercise(
      { db: c.var.db, env: c.env },
      { studentId: c.var.studentId, ...body },
    );
    return c.json(result, 201);
  })
  .post("/complete", validate("json", completeSchema), async (c) => {
    requireAuth(c.var.studentId);
    const body = c.req.valid("json");
    const result = await completeExercise(
      { db: c.var.db, env: c.env },
      { studentId: c.var.studentId, ...body },
    );
    return c.json(result);
  })
  .post("/assign-pending", validate("json", assignPendingSchema), async (c) => {
    requireAuth(c.var.studentId);
    const { sessionId, exerciseId } = c.req.valid("json");
    const payload = await assignPendingExercise(
      { db: c.var.db, env: c.env },
      { studentId: c.var.studentId, sessionId, exerciseId },
    );
    return c.json(payload);
  });

export { exercisesRoutes };
```

**Step 4 — Verify pass**

```bash
cd apps/api && bunx vitest run src/routes/exercises.test.ts
# expect: 6 tests pass (3 existing + 3 new)
```

**Step 5 — Commit**

```
git add apps/api/src/routes/exercises.ts apps/api/src/routes/exercises.test.ts
git commit -m "feat(routes): add POST /assign-pending for reflect-then-prescribe"
```

---

## Final verification (all three tasks)

```bash
cd apps/api && bunx vitest run --config vitest.node.config.ts src/services/exercises.test.ts src/routes/exercises.test.ts
# expect: all tests pass

# Confirm migration is present
grep "CREATE TABLE \"pending_exercises\"" drizzle/*.sql
# expect: one matching line
```

Branch is shippable at this point. Open a PR with `Closes #<issue>` and request review
before Plan D (web) merges.

---

## Challenge Review

### CEO Pass
- **Premise (OBS):** Correct and necessary — the IDOR-guarded ownership link has no existing home; a dedicated `pending_exercises` table is the minimal honest solution (verified: no student↔exercise link exists pre-assignment).
- **Scope (OBS):** Tight — 1 table, 1 service fn, 1 route, 3 vertical tasks. Ships independently (table+endpoint, no caller).
- **Alternatives (OBS):** Spec documents the table-vs-sessions-column choice.

### Engineering Pass
- **[RISK] (confidence: 8/10) — Migration verification paths are wrong.** drizzle.config.ts uses `out: "./src/db/migrations"` (verified) and `schema: "./src/db/schema/index.ts"`. The plan's Step-2/Step-4 greps target `drizzle/*.sql` / `apps/api/drizzle/` — those dirs do not exist; migrations land in `src/db/migrations/`. The table DEF is correct and is auto-detected because `src/db/schema/index.ts` does `export * from "./exercises"` (verified). Build implementer MUST verify with `grep 'CREATE TABLE "pending_exercises"' src/db/migrations/*.sql`, not `drizzle/`.
- **[RISK] (confidence: 7/10) — routes/exercises.test.ts must be REPLACED, not appended.** The existing file (verified, tabs, 3 tests, plain `new Hono().route(...)`, NO `onError`) differs from the plan's block which adds `.onError(errorHandler)` + a service-mock describe + `import * as exercisesService`. A literal append duplicates `import { Hono }` / `testApp` / describe. Implementer must overwrite the file with the new full version and use TAB indentation (biome).
- **[RISK] (confidence: 6/10) — `vi.spyOn(exercisesService, "assignPendingExercise")` may not intercept the route's named import.** The existing 401 tests never reach the service (requireAuth throws first), so this mock pattern is unproven here. If the 200/404 tests hit the real service (with `db: {}`) they will 500. Fallback: switch to top-level `vi.mock("../services/exercises", ...)` (hoisted) if the spy does not intercept.
- **[MINOR] — Dead spread** in `ExerciseSetPayload` `exercises[0]`: `...(dims.length === 1 && dims[0] ? {} : {})` is a no-op (both branches `{}`). Remove it; `hands` is not stored on `exercises` so it is correctly omitted.
- **Architecture (OBS):** Service composes cleanly over existing `assignExercise` (verified signature `{studentId, exerciseId, sessionId?}`). Data flow: select unconsumed pending row → assignExercise → mark consumed → assemble payload from exercises + exerciseDimensions. Sound.
- **Security (OBS):** IDOR guard verified correct — `studentId` from `c.var.studentId` (auth), never request body; foreign `exerciseId` → no matching row → `NotFoundError` → 404 (error-handler.ts verified). This is the #22 IDOR lesson applied correctly.
- **Test Philosophy (OBS):** Service test mocks `ctx.db` (external boundary — OK) and asserts payload shape (behavior). `expect(ctx.db.update).toHaveBeenCalled()` is a weak implementation-detail assertion but acceptable as a secondary check. Route tests verify HTTP status mapping (200/404/401) — legitimate route-level behavior; mocking the separately-tested service at the route boundary is acceptable.
- **Vertical Slice (OBS):** Task 1 (schema) / Task 2 (service) / Task 3 (route) — each one behavior + impl + commit. Task 1's "test" is the generate command (acceptable for a schema change; no behavior to TDD).

### Presumption Inventory
| Assumption | Verdict | Reason |
|---|---|---|
| NotFoundError→404, HTTPException→401 mapping | SAFE | Verified error-handler.ts |
| `NotFoundError(entity, id)` 2-arg signature | SAFE | Verified lib/errors.ts:9 |
| exercises.ts imports `eq, sql`; add `and`, `pendingExercises` | SAFE | Verified |
| New table auto-detected by drizzle-kit | SAFE | index.ts re-exports exercises (verified) |
| Migration output dir | RISKY | It is `src/db/migrations/`, not `drizzle/` — fix verify greps |
| Existing route test structure | RISKY | File must be replaced (no onError today); use tabs |
| vi.spyOn intercepts named service import | VALIDATE | Unproven in repo; vi.mock fallback documented |

### Summary
[BLOCKER] count: 0
[RISK]    count: 3
[QUESTION] count: 0

VERDICT: PROCEED_WITH_CAUTION — (1) verify migration in `src/db/migrations/*.sql`, not `drizzle/`; (2) REPLACE routes/exercises.test.ts (don't append) with tabs; (3) if 200/404 route tests don't intercept via vi.spyOn, switch to vi.mock("../services/exercises"); (4) drop the dead spread in the payload.
