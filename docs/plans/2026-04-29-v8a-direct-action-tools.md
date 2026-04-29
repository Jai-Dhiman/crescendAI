# V8a — Direct-Action Tools Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** Register `assign_segment_loop` as the first action atom so both synthesis and chat paths can create durable `segment_loops` DB rows that the DO tracks across sessions.
**Spec:** docs/specs/2026-04-27-v8a-direct-action-tools-design.md
**Style:** Follow `apps/api/TS_STYLE.md`. No emojis. Explicit exceptions, no silent fallbacks.

---

## Task Groups

```
Group A (parallel): Task 1, Task 2, Task 3
Group B (parallel, depends on A): Task 4, Task 5
Group C (parallel, depends on B): Task 6, Task 7, Task 8
Group D (sequential, depends on C): Task 9, then Task 10
Group E (parallel, depends on D): Task 11, Task 12
```

---

### Task 1: DB schema + migration for segment_loops

**Group:** A (parallel with Task 2, Task 3)

**Behavior being verified:** `segment_loops` Drizzle table exports with correct name and required columns.
**Interface under test:** `apps/api/src/db/schema/segment-loops.ts` export

**Files:**
- Create: `apps/api/src/db/schema/segment-loops.ts`
- Create: `apps/api/src/db/migrations/0003_segment_loops.sql`
- Modify: `apps/api/src/db/schema/index.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/db/schema/segment-loops.test.ts
import { test, expect } from "vitest";
import { getTableName } from "drizzle-orm";
import { segmentLoops } from "./segment-loops";

test("segmentLoops table has correct name", () => {
  expect(getTableName(segmentLoops)).toBe("segment_loops");
});

test("segmentLoops table has status, piece_id, student_id columns", () => {
  const cols = Object.keys(segmentLoops);
  expect(cols).toContain("status");
  expect(cols).toContain("pieceId");
  expect(cols).toContain("studentId");
  expect(cols).toContain("barsStart");
  expect(cols).toContain("barsEnd");
  expect(cols).toContain("attemptsCompleted");
  expect(cols).toContain("requiredCorrect");
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun run test -- --run src/db/schema/segment-loops.test.ts
```
Expected: FAIL — `Cannot find module './segment-loops'`

- [ ] **Step 3: Implement**

```typescript
// apps/api/src/db/schema/segment-loops.ts
import { integer, pgTable, text, timestamp } from "drizzle-orm/pg-core";
import { sql } from "drizzle-orm";

export const SEGMENT_LOOP_STATUSES = [
  "pending",
  "active",
  "completed",
  "dismissed",
  "superseded",
] as const;

export type SegmentLoopStatus = (typeof SEGMENT_LOOP_STATUSES)[number];

export const SEGMENT_LOOP_TRIGGERS = ["chat", "synthesis"] as const;
export type SegmentLoopTrigger = (typeof SEGMENT_LOOP_TRIGGERS)[number];

export const segmentLoops = pgTable("segment_loops", {
  id: text("id")
    .primaryKey()
    .default(sql`gen_random_uuid()`),
  studentId: text("student_id").notNull(),
  pieceId: text("piece_id").notNull(),
  conversationId: text("conversation_id"),
  barsStart: integer("bars_start").notNull(),
  barsEnd: integer("bars_end").notNull(),
  dimension: text("dimension"),
  requiredCorrect: integer("required_correct").notNull().default(5),
  attemptsCompleted: integer("attempts_completed").notNull().default(0),
  status: text("status").notNull().default("pending"),
  trigger: text("trigger").notNull(),
  createdAt: timestamp("created_at", { withTimezone: true })
    .notNull()
    .defaultNow(),
  updatedAt: timestamp("updated_at", { withTimezone: true })
    .notNull()
    .defaultNow(),
});
```

```sql
-- apps/api/src/db/migrations/0003_segment_loops.sql
CREATE TABLE IF NOT EXISTS segment_loops (
  id TEXT PRIMARY KEY DEFAULT gen_random_uuid(),
  student_id TEXT NOT NULL,
  piece_id TEXT NOT NULL,
  conversation_id TEXT,
  bars_start INTEGER NOT NULL,
  bars_end INTEGER NOT NULL,
  dimension TEXT,
  required_correct INTEGER NOT NULL DEFAULT 5,
  attempts_completed INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'pending',
  trigger TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX segment_loops_active_unique
  ON segment_loops (student_id, piece_id)
  WHERE status = 'active';
```

```typescript
// apps/api/src/db/schema/index.ts  (add to existing re-exports)
export * from "./segment-loops";
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run test -- --run src/db/schema/segment-loops.test.ts
```

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/db/schema/segment-loops.ts apps/api/src/db/migrations/0003_segment_loops.sql apps/api/src/db/schema/segment-loops.test.ts apps/api/src/db/schema/index.ts && git commit -m "feat(v8a): add segment_loops DB schema and migration"
```

---

### Task 2: SegmentLoopArtifact Zod schema

**Group:** A (parallel with Task 1, Task 3)

**Behavior being verified:** `SegmentLoopArtifactSchema` accepts valid artifacts and rejects ones with `bars_end < bars_start`.
**Interface under test:** `SegmentLoopArtifactSchema.safeParse`

**Files:**
- Create: `apps/api/src/harness/artifacts/segment-loop.ts`
- Create: `apps/api/src/harness/artifacts/segment-loop.test.ts`
- Modify: `apps/api/src/harness/artifacts/index.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/harness/artifacts/segment-loop.test.ts
import { test, expect } from "vitest";
import { SegmentLoopArtifactSchema } from "./segment-loop";

const VALID = {
  kind: "segment_loop" as const,
  id: "loop-abc",
  studentId: "stu-1",
  pieceId: "chopin.ballades.1",
  barsStart: 12,
  barsEnd: 16,
  requiredCorrect: 3,
  attemptsCompleted: 0,
  status: "pending" as const,
  dimension: null,
};

test("valid SegmentLoopArtifact passes", () => {
  const r = SegmentLoopArtifactSchema.safeParse(VALID);
  expect(r.success).toBe(true);
});

test("bars_end < bars_start fails refinement", () => {
  const r = SegmentLoopArtifactSchema.safeParse({ ...VALID, barsStart: 20, barsEnd: 10 });
  expect(r.success).toBe(false);
});

test("invalid status rejected", () => {
  const r = SegmentLoopArtifactSchema.safeParse({ ...VALID, status: "unknown" });
  expect(r.success).toBe(false);
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun run test -- --run src/harness/artifacts/segment-loop.test.ts
```
Expected: FAIL — `Cannot find module './segment-loop'`

- [ ] **Step 3: Implement**

```typescript
// apps/api/src/harness/artifacts/segment-loop.ts
import { z } from "zod";
import { DIMS_6 } from "../../lib/dims";

export const SEGMENT_LOOP_STATUSES = [
  "pending",
  "active",
  "completed",
  "dismissed",
  "superseded",
] as const;

export const SegmentLoopArtifactSchema = z
  .object({
    kind: z.literal("segment_loop"),
    id: z.string().min(1),
    studentId: z.string().min(1),
    pieceId: z.string().min(1),
    barsStart: z.number().int().positive(),
    barsEnd: z.number().int().positive(),
    requiredCorrect: z.number().int().min(1).max(10),
    attemptsCompleted: z.number().int().min(0),
    status: z.enum(SEGMENT_LOOP_STATUSES),
    dimension: z.enum(DIMS_6 as unknown as [string, ...string[]]).nullable(),
  })
  .refine((v) => v.barsEnd >= v.barsStart, {
    message: "barsEnd must be >= barsStart",
    path: ["barsEnd"],
  });

export type SegmentLoopArtifact = z.infer<typeof SegmentLoopArtifactSchema>;

export interface SegmentLoopRef {
  id: string;
  pieceId: string;
  barsStart: number;
  barsEnd: number;
}
```

Add to `apps/api/src/harness/artifacts/index.ts`:
```typescript
export { SegmentLoopArtifactSchema, type SegmentLoopArtifact, type SegmentLoopRef } from "./segment-loop";
// Also add to ARTIFACT_NAMES and artifactSchemas:
// "SegmentLoopArtifact" in ARTIFACT_NAMES array
// SegmentLoopArtifact: SegmentLoopArtifactSchema in artifactSchemas record
```

The full updated `index.ts`:
```typescript
import type { ZodTypeAny } from "zod";
import { DiagnosisArtifactSchema, type DiagnosisArtifact } from "./diagnosis";
import { ExerciseArtifactSchema, type ExerciseArtifact } from "./exercise";
import { SynthesisArtifactSchema, type SynthesisArtifact } from "./synthesis";
import { SegmentLoopArtifactSchema, type SegmentLoopArtifact } from "./segment-loop";

export { DiagnosisArtifactSchema, type DiagnosisArtifact } from "./diagnosis";
export { ExerciseArtifactSchema, type ExerciseArtifact } from "./exercise";
export { SynthesisArtifactSchema, type SynthesisArtifact } from "./synthesis";
export { SegmentLoopArtifactSchema, type SegmentLoopArtifact, type SegmentLoopRef } from "./segment-loop";

export const ARTIFACT_NAMES = [
  "DiagnosisArtifact",
  "ExerciseArtifact",
  "SynthesisArtifact",
  "SegmentLoopArtifact",
] as const;
export type ArtifactName = (typeof ARTIFACT_NAMES)[number];

export const artifactSchemas: Record<ArtifactName, ZodTypeAny> = {
  DiagnosisArtifact: DiagnosisArtifactSchema,
  ExerciseArtifact: ExerciseArtifactSchema,
  SynthesisArtifact: SynthesisArtifactSchema,
  SegmentLoopArtifact: SegmentLoopArtifactSchema,
};
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run test -- --run src/harness/artifacts/segment-loop.test.ts
```

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/artifacts/segment-loop.ts apps/api/src/harness/artifacts/segment-loop.test.ts apps/api/src/harness/artifacts/index.ts && git commit -m "feat(v8a): add SegmentLoopArtifact Zod schema"
```

---

### Task 3: Extend HookContext with pieceId and trigger; extend ToolDefinition.invoke

**Group:** A (parallel with Task 1, Task 2)

**Behavior being verified:** `HookContext` carries `pieceId` and `trigger`; an action atom's `invoke` receives `PhaseContext` as second argument in `runPhase1`.
**Interface under test:** `runPhase1` event stream when a tool's `invoke` reads `ctx.trigger`

**Files:**
- Modify: `apps/api/src/harness/loop/types.ts`
- Modify: `apps/api/src/harness/loop/middleware.ts`
- Modify: `apps/api/src/harness/loop/phase1.ts`
- Modify: `apps/api/src/harness/loop/middleware.test.ts`

- [ ] **Step 1: Write the failing test**

Add to `apps/api/src/harness/loop/middleware.test.ts`:
```typescript
import { test, expect } from "vitest";
import { wrapToolCall } from "./middleware";
import type { PhaseContext } from "./types";

const MOCK_CTX = {
  env: {} as PhaseContext["env"],
  studentId: "stu-1",
  sessionId: "sess-1",
  conversationId: null,
  digest: {},
  waitUntil: () => {},
  turnCap: 5,
  trigger: "chat" as const,
} satisfies PhaseContext;

test("wrapToolCall passes through non-action tools unchanged", async () => {
  const result = await wrapToolCall("search_catalog", MOCK_CTX, async () => "result");
  expect(result).toBe("result");
});

test("wrapToolCall passes through assign_segment_loop (gating is in atom)", async () => {
  const result = await wrapToolCall("assign_segment_loop", MOCK_CTX, async () => ({ status: "pending" }));
  expect(result).toEqual({ status: "pending" });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun run test -- --run src/harness/loop/middleware.test.ts
```
Expected: FAIL — `wrapToolCall` has wrong signature (currently `(invoke) => ...`)

- [ ] **Step 3: Implement**

```typescript
// apps/api/src/harness/loop/types.ts — add to HookContext:
export interface HookContext {
  env: import("../../lib/types").Bindings;
  studentId: string;
  sessionId: string;
  conversationId: string | null;
  digest: Record<string, unknown>;
  waitUntil: (p: Promise<unknown>) => void;
  pieceId?: string;
  trigger?: "chat" | "synthesis";
}

// Also extend ToolDefinition.invoke signature:
export interface ToolDefinition {
  name: string;
  description: string;
  input_schema: Record<string, unknown>;
  invoke: (input: unknown, ctx?: PhaseContext) => Promise<unknown>;
}
```

```typescript
// apps/api/src/harness/loop/middleware.ts — update wrapToolCall:
export async function wrapToolCall(
  _toolName: string,
  _ctx: PhaseContext,
  invoke: () => Promise<unknown>,
): Promise<unknown> {
  return invoke();
}
```

```typescript
// apps/api/src/harness/loop/phase1.ts — update the invoke call site:
// Change line:
//   const output = await wrapToolCall(() => def.invoke(tu.input));
// To:
const output = await wrapToolCall(tu.name, ctx, () => def.invoke(tu.input, ctx));
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run test -- --run src/harness/loop/middleware.test.ts
```

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/loop/types.ts apps/api/src/harness/loop/middleware.ts apps/api/src/harness/loop/middleware.test.ts apps/api/src/harness/loop/phase1.ts && git commit -m "feat(v8a): extend HookContext with pieceId/trigger; thread ctx through wrapToolCall"
```

---

### Task 4: services/segment-loops.ts — lifecycle service

**Group:** B (parallel with Task 5, depends on Group A)

**Behavior being verified:** `createSegmentLoop` with `trigger='chat'` creates a row with `status='pending'`; creating a second active loop supersedes the first.
**Interface under test:** Route-level HTTP behavior — `POST /api/segment-loops/:id/accept` transitions pending → active; `POST /api/segment-loops/:id/accept` on an active loop returns 409.

**Files:**
- Create: `apps/api/src/services/segment-loops.ts`
- Create: `apps/api/src/routes/segment-loops.ts`
- Create: `apps/api/src/routes/segment-loops.test.ts`

Note: `routes/segment-loops.ts` is created in this task (thin routes delegating to service) so the route test can exercise the service's public behavior. Task 7 mounts the routes in `index.ts`.

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/routes/segment-loops.test.ts
import { Hono } from "hono";
import { describe, expect, it } from "vitest";
import { segmentLoopsRoutes } from "./segment-loops";

const testApp = new Hono().route("/api/segment-loops", segmentLoopsRoutes);

describe("segment-loops routes — auth guard", () => {
  it("POST /api/segment-loops/:id/accept returns 401 without auth", async () => {
    const res = await testApp.request(
      "/api/segment-loops/00000000-0000-0000-0000-000000000001/accept",
      { method: "POST" },
    );
    expect(res.status).toBe(401);
  });

  it("POST /api/segment-loops/:id/decline returns 401 without auth", async () => {
    const res = await testApp.request(
      "/api/segment-loops/00000000-0000-0000-0000-000000000001/decline",
      { method: "POST" },
    );
    expect(res.status).toBe(401);
  });

  it("POST /api/segment-loops/:id/dismiss returns 401 without auth", async () => {
    const res = await testApp.request(
      "/api/segment-loops/00000000-0000-0000-0000-000000000001/dismiss",
      { method: "POST" },
    );
    expect(res.status).toBe(401);
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun run test -- --run src/routes/segment-loops.test.ts
```
Expected: FAIL — `Cannot find module './segment-loops'`

- [ ] **Step 3: Implement**

```typescript
// apps/api/src/services/segment-loops.ts
import { and, eq, sql } from "drizzle-orm";
import { segmentLoops } from "../db/schema/segment-loops";
import type { SEGMENT_LOOP_STATUSES } from "../db/schema/segment-loops";
import { ConflictError, NotFoundError, ValidationError } from "../lib/errors";
import type { Db } from "../lib/types";
import type { SegmentLoopArtifact } from "../harness/artifacts/segment-loop";
import type { InlineComponent } from "./tool-processor";

type SegmentLoopStatus = (typeof SEGMENT_LOOP_STATUSES)[number];

const TERMINAL_STATUSES: SegmentLoopStatus[] = ["completed", "dismissed", "superseded"];

function rowToArtifact(row: typeof segmentLoops.$inferSelect): SegmentLoopArtifact {
  return {
    kind: "segment_loop",
    id: row.id,
    studentId: row.studentId,
    pieceId: row.pieceId,
    barsStart: row.barsStart,
    barsEnd: row.barsEnd,
    requiredCorrect: row.requiredCorrect,
    attemptsCompleted: row.attemptsCompleted,
    status: row.status as SegmentLoopStatus,
    dimension: row.dimension as SegmentLoopArtifact["dimension"],
  };
}

export interface CreateSegmentLoopInput {
  studentId: string;
  pieceId: string;
  conversationId: string | null;
  barsStart: number;
  barsEnd: number;
  requiredCorrect: number;
  dimension?: string | null;
  trigger: "chat" | "synthesis";
}

export async function createSegmentLoop(
  db: Db,
  input: CreateSegmentLoopInput,
): Promise<SegmentLoopArtifact> {
  const status = input.trigger === "chat" ? "pending" : "active";

  return await db.transaction(async (tx) => {
    // Supersede any non-terminal loop for this (student, piece)
    await tx
      .update(segmentLoops)
      .set({ status: "superseded", updatedAt: new Date() })
      .where(
        and(
          eq(segmentLoops.studentId, input.studentId),
          eq(segmentLoops.pieceId, input.pieceId),
          sql`status NOT IN ('completed', 'dismissed', 'superseded')`,
        ),
      );

    const [inserted] = await tx
      .insert(segmentLoops)
      .values({
        studentId: input.studentId,
        pieceId: input.pieceId,
        conversationId: input.conversationId,
        barsStart: input.barsStart,
        barsEnd: input.barsEnd,
        requiredCorrect: input.requiredCorrect,
        dimension: input.dimension ?? null,
        trigger: input.trigger,
        status,
      })
      .returning();

    if (!inserted) throw new ConflictError("Failed to insert segment loop");
    return rowToArtifact(inserted);
  });
}

async function assertOwnerAndLoad(
  db: Db,
  id: string,
  studentId: string,
): Promise<typeof segmentLoops.$inferSelect> {
  const [row] = await db
    .select()
    .from(segmentLoops)
    .where(eq(segmentLoops.id, id))
    .limit(1);
  if (!row || row.studentId !== studentId) throw new NotFoundError("segment_loop", id);
  return row;
}

function assertNotTerminal(status: string, id: string): void {
  if (TERMINAL_STATUSES.includes(status as SegmentLoopStatus)) {
    throw new ValidationError(`segment_loop ${id} is in terminal state '${status}'`);
  }
}

export async function acceptSegmentLoop(
  db: Db,
  id: string,
  studentId: string,
): Promise<SegmentLoopArtifact> {
  const row = await assertOwnerAndLoad(db, id, studentId);
  if (row.status !== "pending") {
    throw new ValidationError(`accept requires status 'pending', got '${row.status}'`);
  }
  const [updated] = await db
    .update(segmentLoops)
    .set({ status: "active", updatedAt: new Date() })
    .where(eq(segmentLoops.id, id))
    .returning();
  return rowToArtifact(updated!);
}

export async function declineSegmentLoop(
  db: Db,
  id: string,
  studentId: string,
): Promise<SegmentLoopArtifact> {
  const row = await assertOwnerAndLoad(db, id, studentId);
  if (row.status !== "pending") {
    throw new ValidationError(`decline requires status 'pending', got '${row.status}'`);
  }
  const [updated] = await db
    .update(segmentLoops)
    .set({ status: "dismissed", updatedAt: new Date() })
    .where(eq(segmentLoops.id, id))
    .returning();
  return rowToArtifact(updated!);
}

export async function dismissSegmentLoop(
  db: Db,
  id: string,
  studentId: string,
): Promise<SegmentLoopArtifact> {
  const row = await assertOwnerAndLoad(db, id, studentId);
  assertNotTerminal(row.status, id);
  const [updated] = await db
    .update(segmentLoops)
    .set({ status: "dismissed", updatedAt: new Date() })
    .where(eq(segmentLoops.id, id))
    .returning();
  return rowToArtifact(updated!);
}

export async function findActiveForPiece(
  db: Db,
  studentId: string,
  pieceId: string,
): Promise<SegmentLoopArtifact | null> {
  const [row] = await db
    .select()
    .from(segmentLoops)
    .where(
      and(
        eq(segmentLoops.studentId, studentId),
        eq(segmentLoops.pieceId, pieceId),
        eq(segmentLoops.status, "active"),
      ),
    )
    .limit(1);
  return row ? rowToArtifact(row) : null;
}

export async function incrementAttempts(
  db: Db,
  id: string,
  studentId: string,
): Promise<{ attemptsCompleted: number; completedNow: boolean }> {
  const row = await assertOwnerAndLoad(db, id, studentId);
  if (row.status !== "active") {
    throw new ValidationError(`incrementAttempts requires status 'active', got '${row.status}'`);
  }
  const newCount = row.attemptsCompleted + 1;
  const completedNow = newCount >= row.requiredCorrect;
  await db
    .update(segmentLoops)
    .set({
      attemptsCompleted: newCount,
      status: completedNow ? "completed" : "active",
      updatedAt: new Date(),
    })
    .where(eq(segmentLoops.id, id));
  return { attemptsCompleted: newCount, completedNow };
}

export function toLoopComponent(artifact: SegmentLoopArtifact): InlineComponent {
  return {
    type: "segment_loop",
    config: {
      id: artifact.id,
      pieceId: artifact.pieceId,
      barsStart: artifact.barsStart,
      barsEnd: artifact.barsEnd,
      requiredCorrect: artifact.requiredCorrect,
      attemptsCompleted: artifact.attemptsCompleted,
      status: artifact.status,
      dimension: artifact.dimension,
    },
  };
}
```

```typescript
// apps/api/src/routes/segment-loops.ts
import { Hono } from "hono";
import type { Bindings, Variables } from "../lib/types";
import { requireAuth } from "../middleware/auth-session";
import { NotFoundError, ValidationError } from "../lib/errors";
import * as segmentLoopsService from "../services/segment-loops";

const app = new Hono<{ Bindings: Bindings; Variables: Variables }>();

app.post("/:id/accept", async (c) => {
  requireAuth(c.var.studentId);
  const studentId = c.var.studentId!;
  const id = c.req.param("id");
  try {
    const artifact = await segmentLoopsService.acceptSegmentLoop(c.var.db, id, studentId);
    return c.json(artifact);
  } catch (err) {
    if (err instanceof NotFoundError) return c.json({ error: err.message }, 404);
    if (err instanceof ValidationError) return c.json({ error: err.message, code: "invalid_state" }, 409);
    throw err;
  }
});

app.post("/:id/decline", async (c) => {
  requireAuth(c.var.studentId);
  const studentId = c.var.studentId!;
  const id = c.req.param("id");
  try {
    const artifact = await segmentLoopsService.declineSegmentLoop(c.var.db, id, studentId);
    return c.json(artifact);
  } catch (err) {
    if (err instanceof NotFoundError) return c.json({ error: err.message }, 404);
    if (err instanceof ValidationError) return c.json({ error: err.message, code: "invalid_state" }, 409);
    throw err;
  }
});

app.post("/:id/dismiss", async (c) => {
  requireAuth(c.var.studentId);
  const studentId = c.var.studentId!;
  const id = c.req.param("id");
  try {
    const artifact = await segmentLoopsService.dismissSegmentLoop(c.var.db, id, studentId);
    return c.json(artifact);
  } catch (err) {
    if (err instanceof NotFoundError) return c.json({ error: err.message }, 404);
    if (err instanceof ValidationError) return c.json({ error: err.message, code: "invalid_state" }, 409);
    throw err;
  }
});

export { app as segmentLoopsRoutes };
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run test -- --run src/routes/segment-loops.test.ts
```

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/services/segment-loops.ts apps/api/src/routes/segment-loops.ts apps/api/src/routes/segment-loops.test.ts && git commit -m "feat(v8a): segment-loops lifecycle service + accept/decline/dismiss routes"
```

---

### Task 5: passage-loop-detector — strict-isolation detector

**Group:** B (parallel with Task 4, depends on Group A)

**Behavior being verified:** A contiguous position span bounded within ±1 bar of the assignment returns `LoopAttempt{inBounds:true}`; a span that starts at bar 1 and traverses the assigned bars returns no event.
**Interface under test:** `processPosition` function

**Files:**
- Create: `apps/api/src/do/passage-loop-detector.ts`
- Create: `apps/api/src/do/passage-loop-detector.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/do/passage-loop-detector.test.ts
import { describe, expect, test } from "vitest";
import { PassageLoopDetector } from "./passage-loop-detector";
import type { SegmentLoopArtifact } from "../harness/artifacts/segment-loop";

const ASSIGNMENT: SegmentLoopArtifact = {
  kind: "segment_loop",
  id: "loop-1",
  studentId: "stu-1",
  pieceId: "chopin.ballades.1",
  barsStart: 12,
  barsEnd: 16,
  requiredCorrect: 3,
  attemptsCompleted: 0,
  status: "active",
  dimension: null,
};

// PositionSpan: { startBar: number; endBar: number; durationMs: number }
describe("PassageLoopDetector", () => {
  test("clean isolated loop within tolerance returns inBounds=true", () => {
    const det = new PassageLoopDetector();
    const event = det.processPosition(
      { startBar: 12, endBar: 16, durationMs: 8000 },
      ASSIGNMENT,
    );
    expect(event).not.toBeNull();
    expect(event?.inBounds).toBe(true);
  });

  test("start-to-finish playthrough traversing assigned bars returns null", () => {
    const det = new PassageLoopDetector();
    const event = det.processPosition(
      { startBar: 1, endBar: 80, durationMs: 180000 },
      ASSIGNMENT,
    );
    expect(event).toBeNull();
  });

  test("span starting before tolerance window returns null", () => {
    const det = new PassageLoopDetector();
    const event = det.processPosition(
      { startBar: 5, endBar: 16, durationMs: 30000 },
      ASSIGNMENT,
    );
    expect(event).toBeNull();
  });

  test("span ending after tolerance window returns null", () => {
    const det = new PassageLoopDetector();
    const event = det.processPosition(
      { startBar: 12, endBar: 25, durationMs: 30000 },
      ASSIGNMENT,
    );
    expect(event).toBeNull();
  });

  test("same span reported twice is debounced — returns only one event", () => {
    const det = new PassageLoopDetector();
    const span = { startBar: 12, endBar: 16, durationMs: 8000 };
    det.processPosition(span, ASSIGNMENT);
    const second = det.processPosition(span, ASSIGNMENT);
    expect(second).toBeNull();
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun run test -- --run src/do/passage-loop-detector.test.ts
```
Expected: FAIL — `Cannot find module './passage-loop-detector'`

- [ ] **Step 3: Implement**

```typescript
// apps/api/src/do/passage-loop-detector.ts
import type { SegmentLoopArtifact } from "../harness/artifacts/segment-loop";

export interface PositionSpan {
  startBar: number;
  endBar: number;
  durationMs: number;
}

export interface LoopAttempt {
  inBounds: boolean;
  ts: number;
  passage: { startBar: number; endBar: number };
}

const TOLERANCE_BARS = 1;
const DEBOUNCE_MS = 2000;

export class PassageLoopDetector {
  private lastEventTs = 0;
  private lastPassageKey = "";

  processPosition(
    span: PositionSpan,
    assignment: SegmentLoopArtifact,
  ): LoopAttempt | null {
    const { startBar, endBar } = span;
    const assignStart = assignment.barsStart;
    const assignEnd = assignment.barsEnd;

    // Strict isolation: start must be within tolerance of assigned start
    const startInWindow =
      startBar >= assignStart - TOLERANCE_BARS &&
      startBar <= assignStart + TOLERANCE_BARS;
    // End must be within tolerance of assigned end
    const endInWindow =
      endBar >= assignEnd - TOLERANCE_BARS && endBar <= assignEnd + TOLERANCE_BARS;

    if (!startInWindow || !endInWindow) return null;

    const passageKey = `${startBar}-${endBar}`;
    const now = Date.now();

    // Debounce: same passage within debounce window counts once
    if (passageKey === this.lastPassageKey && now - this.lastEventTs < DEBOUNCE_MS) {
      return null;
    }

    this.lastEventTs = now;
    this.lastPassageKey = passageKey;

    return { inBounds: true, ts: now, passage: { startBar, endBar } };
  }

  reset(): void {
    this.lastEventTs = 0;
    this.lastPassageKey = "";
  }
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run test -- --run src/do/passage-loop-detector.test.ts
```

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/do/passage-loop-detector.ts apps/api/src/do/passage-loop-detector.test.ts && git commit -m "feat(v8a): passage-loop-detector strict-isolation algorithm"
```

---

### Task 6: assign-segment-loop action atom

**Group:** C (parallel with Task 7, Task 8, depends on Group B)

**Behavior being verified:** The atom with `trigger='synthesis'` creates an `active` loop; with `trigger='chat'` creates a `pending` loop; with no `piece_id` in tool input throws `ToolPreconditionError`.
**Interface under test:** `assignSegmentLoopAtom(ctx, input)` exported function

**Files:**
- Create: `apps/api/src/harness/atoms/assign-segment-loop.ts`
- Create: `apps/api/src/harness/atoms/assign-segment-loop.test.ts`
- Modify: `apps/api/src/harness/skills/atoms/index.ts` (add to ALL_ATOMS and add a ToolDefinition export)
- Modify: `apps/api/src/lib/errors.ts` (add `ToolPreconditionError`)

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/harness/atoms/assign-segment-loop.test.ts
import { describe, expect, test, vi } from "vitest";
import { assignSegmentLoopAtom, ASSIGN_SEGMENT_LOOP_TOOL } from "./assign-segment-loop";
import { ToolPreconditionError } from "../../lib/errors";
import type { PhaseContext } from "../loop/types";
import type { Bindings } from "../../lib/types";

const mockCreateLoop = vi.fn();
vi.mock("../../services/segment-loops", () => ({
  createSegmentLoop: (...args: unknown[]) => mockCreateLoop(...args),
}));

const BASE_CTX: PhaseContext = {
  env: {} as Bindings,
  studentId: "stu-1",
  sessionId: "sess-1",
  conversationId: "conv-1",
  digest: {},
  waitUntil: () => {},
  turnCap: 5,
  trigger: "synthesis",
  pieceId: "chopin.ballades.1",
};

const VALID_INPUT = {
  piece_id: "chopin.ballades.1",
  bars_start: 12,
  bars_end: 16,
  required_correct: 3,
};

describe("assignSegmentLoopAtom", () => {
  test("synthesis trigger creates active loop", async () => {
    const mockArtifact = { kind: "segment_loop", id: "loop-1", status: "active" };
    mockCreateLoop.mockResolvedValueOnce(mockArtifact);

    const result = await assignSegmentLoopAtom(BASE_CTX, VALID_INPUT);

    expect(mockCreateLoop).toHaveBeenCalledWith(
      expect.anything(),
      expect.objectContaining({ trigger: "synthesis", status: undefined }),
    );
    expect(result).toEqual(mockArtifact);
  });

  test("chat trigger creates pending loop", async () => {
    const mockArtifact = { kind: "segment_loop", id: "loop-2", status: "pending" };
    mockCreateLoop.mockResolvedValueOnce(mockArtifact);

    const chatCtx = { ...BASE_CTX, trigger: "chat" as const };
    await assignSegmentLoopAtom(chatCtx, VALID_INPUT);

    expect(mockCreateLoop).toHaveBeenCalledWith(
      expect.anything(),
      expect.objectContaining({ trigger: "chat" }),
    );
  });

  test("missing piece_id throws ToolPreconditionError", async () => {
    await expect(
      assignSegmentLoopAtom(BASE_CTX, { bars_start: 12, bars_end: 16, required_correct: 3 }),
    ).rejects.toBeInstanceOf(ToolPreconditionError);
  });

  test("bars_end < bars_start throws ValidationError", async () => {
    await expect(
      assignSegmentLoopAtom(BASE_CTX, { ...VALID_INPUT, bars_start: 20, bars_end: 10 }),
    ).rejects.toThrow();
  });

  test("ASSIGN_SEGMENT_LOOP_TOOL has correct name", () => {
    expect(ASSIGN_SEGMENT_LOOP_TOOL.name).toBe("assign_segment_loop");
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun run test -- --run src/harness/atoms/assign-segment-loop.test.ts
```
Expected: FAIL — `Cannot find module './assign-segment-loop'` and `ToolPreconditionError` missing

- [ ] **Step 3: Implement**

```typescript
// apps/api/src/lib/errors.ts — add:
export class ToolPreconditionError extends DomainError {
  constructor(reason: string) {
    super(`tool precondition failed: ${reason}`);
  }
}
```

```typescript
// apps/api/src/harness/atoms/assign-segment-loop.ts
import { z } from "zod";
import { DIMS_6 } from "../../lib/dims";
import { ToolPreconditionError, ValidationError } from "../../lib/errors";
import { createSegmentLoop } from "../../services/segment-loops";
import type { SegmentLoopArtifact } from "../artifacts/segment-loop";
import type { PhaseContext, ToolDefinition } from "../loop/types";
import { createDb } from "../../db/client";

const inputSchema = z.object({
  piece_id: z.string().min(1).optional(),
  bars_start: z.number().int().positive(),
  bars_end: z.number().int().positive(),
  required_correct: z.number().int().min(1).max(10).default(5),
  dimension: z.enum(DIMS_6 as unknown as [string, ...string[]]).optional().nullable(),
});

export async function assignSegmentLoopAtom(
  ctx: PhaseContext,
  rawInput: unknown,
): Promise<SegmentLoopArtifact> {
  const parsed = inputSchema.safeParse(rawInput);
  if (!parsed.success) {
    throw new ValidationError(parsed.error.message);
  }
  const input = parsed.data;

  if (!input.piece_id) {
    throw new ToolPreconditionError("no_piece_identified");
  }

  if (input.bars_end < input.bars_start) {
    throw new ValidationError("bars_end must be >= bars_start");
  }

  const db = createDb(ctx.env.HYPERDRIVE);
  return createSegmentLoop(db, {
    studentId: ctx.studentId,
    pieceId: input.piece_id,
    conversationId: ctx.conversationId,
    barsStart: input.bars_start,
    barsEnd: input.bars_end,
    requiredCorrect: input.required_correct,
    dimension: input.dimension ?? null,
    trigger: ctx.trigger ?? "synthesis",
  });
}

export const ASSIGN_SEGMENT_LOOP_TOOL: ToolDefinition = {
  name: "assign_segment_loop",
  description:
    "Assign a focused passage-loop practice task. The student will practice the specified bar range repeatedly until they complete the required number of isolated attempts. Use after identifying a specific passage that needs targeted work. Requires a piece to be identified first.",
  input_schema: {
    type: "object",
    properties: {
      piece_id: {
        type: "string",
        description:
          "Piece slug from search_catalog (e.g. 'chopin.ballades.1'). Pass verbatim.",
      },
      bars_start: {
        type: "integer",
        description: "First bar of the practice passage (inclusive, 1-indexed).",
        minimum: 1,
      },
      bars_end: {
        type: "integer",
        description: "Last bar of the practice passage (inclusive). Must be >= bars_start.",
        minimum: 1,
      },
      required_correct: {
        type: "integer",
        description: "Number of isolated loop attempts required to complete the assignment (1-10). Default 5.",
        minimum: 1,
        maximum: 10,
        default: 5,
      },
      dimension: {
        type: "string",
        enum: [...DIMS_6],
        description: "Optional: which musical dimension this loop targets.",
      },
    },
    required: ["piece_id", "bars_start", "bars_end"],
  },
  invoke: async (input: unknown, ctx?: PhaseContext): Promise<unknown> => {
    if (!ctx) throw new ToolPreconditionError("assign_segment_loop requires PhaseContext");
    return assignSegmentLoopAtom(ctx, input);
  },
};
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run test -- --run src/harness/atoms/assign-segment-loop.test.ts
```

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/atoms/assign-segment-loop.ts apps/api/src/harness/atoms/assign-segment-loop.test.ts apps/api/src/lib/errors.ts && git commit -m "feat(v8a): assign-segment-loop action atom + ToolPreconditionError"
```

---

### Task 7: Mount segment-loops routes in index.ts

**Group:** C (parallel with Task 6, Task 8, depends on Group B)

**Behavior being verified:** `GET /api/segment-loops/xxx/accept` (wrong method) returns 405; auth guard active.
**Interface under test:** Mounted application router

**Files:**
- Modify: `apps/api/src/index.ts`

- [ ] **Step 1: Write the failing test**

Add to `apps/api/src/routes/segment-loops.test.ts`:
```typescript
import { app } from "../../index";

describe("segment-loops routes — mounted", () => {
  it("GET /api/segment-loops/:id/accept returns 405 (method not allowed)", async () => {
    const res = await app.request(
      "/api/segment-loops/00000000-0000-0000-0000-000000000001/accept",
    );
    expect(res.status).toBe(405);
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun run test -- --run src/routes/segment-loops.test.ts
```
Expected: FAIL — returns 404 (route not mounted)

- [ ] **Step 3: Implement**

In `apps/api/src/index.ts`, add import and route:
```typescript
import { segmentLoopsRoutes } from "./routes/segment-loops";

// In the routes chain, add:
.route("/api/segment-loops", segmentLoopsRoutes)
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run test -- --run src/routes/segment-loops.test.ts
```

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/index.ts && git commit -m "feat(v8a): mount segment-loops routes at /api/segment-loops"
```

---

### Task 8: Fix HARNESS_V6_CHAT_ENABLED flag + add compound-registry OnChatMessage tool

**Group:** C (parallel with Task 6, Task 7, depends on Group B)

**Behavior being verified:** `compound-registry` `OnChatMessage` binding includes `assign_segment_loop` in its tool list; `routes/chat.ts` routes through `chatV6` when `HARNESS_V6_ENABLED=true`.
**Interface under test:** `getCompoundBinding('OnChatMessage').tools` array; `routes/chat.test.ts`

**Files:**
- Modify: `apps/api/src/routes/chat.ts`
- Modify: `apps/api/src/harness/loop/compound-registry.ts`
- Modify: `apps/api/src/harness/loop/compound-registry.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// Add to apps/api/src/harness/loop/compound-registry.test.ts:
import { test, expect } from "vitest";
import { getCompoundBinding } from "./compound-registry";

test("OnChatMessage binding includes assign_segment_loop", () => {
  const binding = getCompoundBinding("OnChatMessage");
  const toolNames = binding?.tools.map((t) => t.name) ?? [];
  expect(toolNames).toContain("assign_segment_loop");
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun run test -- --run src/harness/loop/compound-registry.test.ts
```
Expected: FAIL — `assign_segment_loop` not in tool list

- [ ] **Step 3: Implement**

In `apps/api/src/harness/loop/compound-registry.ts`, add the `assign_segment_loop` tool to the `OnChatMessage` binding:

```typescript
import { ASSIGN_SEGMENT_LOOP_TOOL } from "../../harness/atoms/assign-segment-loop";

// In the OnChatMessage binding tools array, add ASSIGN_SEGMENT_LOOP_TOOL alongside the TOOL_REGISTRY tools:
tools: [
  ...Object.values(TOOL_REGISTRY).map((t) => ({
    name: t.name,
    description: t.description,
    input_schema: t.anthropicSchema.input_schema as Record<string, unknown>,
    invoke: async (_input: unknown): Promise<unknown> => ({}),
  })),
  ASSIGN_SEGMENT_LOOP_TOOL,
],
```

In `apps/api/src/routes/chat.ts`, fix the flag:
```typescript
// Change:
c.env.HARNESS_V6_CHAT_ENABLED === "true"
// To:
c.env.HARNESS_V6_ENABLED === "true"
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run test -- --run src/harness/loop/compound-registry.test.ts
```

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/loop/compound-registry.ts apps/api/src/routes/chat.ts apps/api/src/harness/loop/compound-registry.test.ts && git commit -m "feat(v8a): add assign_segment_loop to OnChatMessage binding; fix chat flag"
```

---

### Task 9: Synthesis path wiring — synthesizeV6 + compound-registry OnSessionEnd + SynthesisArtifact

**Group:** D (sequential — Task 9 first, then Task 10; depends on Group C)

**Behavior being verified:** `synthesizeV6` passes `trigger:'synthesis'` and `pieceId` to `HookContext`; `SynthesisArtifactSchema` accepts an `assigned_loops` array; `OnSessionEnd` binding includes `ASSIGN_SEGMENT_LOOP_TOOL` in its tools.
**Interface under test:** `SynthesisArtifactSchema.safeParse`; `getCompoundBinding('OnSessionEnd').tools`; `buildV6WsPayload` includes loop components

**Files:**
- Modify: `apps/api/src/services/teacher.ts` (`synthesizeV6`)
- Modify: `apps/api/src/harness/artifacts/synthesis.ts` (add `assigned_loops`)
- Modify: `apps/api/src/harness/artifacts/synthesis.test.ts`
- Modify: `apps/api/src/harness/loop/compound-registry.ts` (add to OnSessionEnd tools)
- Modify: `apps/api/src/do/session-brain.ts` (`buildV6WsPayload`)

- [ ] **Step 1: Write the failing test**

```typescript
// Add to apps/api/src/harness/artifacts/synthesis.test.ts:
import { test, expect } from "vitest";
import { SynthesisArtifactSchema } from "./synthesis";

const BASE_VALID = {
  session_id: "sess_1",
  synthesis_scope: "session" as const,
  strengths: [],
  focus_areas: [],
  proposed_exercises: [],
  dominant_dimension: "phrasing" as const,
  recurring_pattern: null,
  next_session_focus: null,
  diagnosis_refs: [],
  headline: "A".repeat(300),
  assigned_loops: [],
};

test("SynthesisArtifact with empty assigned_loops passes", () => {
  expect(SynthesisArtifactSchema.safeParse(BASE_VALID).success).toBe(true);
});

test("SynthesisArtifact with assigned_loops entry passes", () => {
  const r = SynthesisArtifactSchema.safeParse({
    ...BASE_VALID,
    assigned_loops: [{ id: "loop-1", pieceId: "chopin.ballades.1", barsStart: 12, barsEnd: 16 }],
  });
  expect(r.success).toBe(true);
});

test("SynthesisArtifact without assigned_loops field defaults to empty array", () => {
  const { assigned_loops: _, ...without } = BASE_VALID;
  const r = SynthesisArtifactSchema.safeParse(without);
  expect(r.success).toBe(true);
  if (r.success) expect(r.data.assigned_loops).toEqual([]);
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun run test -- --run src/harness/artifacts/synthesis.test.ts
```
Expected: FAIL — `assigned_loops` field does not exist on schema

- [ ] **Step 3: Implement**

In `apps/api/src/harness/artifacts/synthesis.ts`, add `assigned_loops`:
```typescript
// Add SegmentLoopRef import at top:
import type { SegmentLoopRef } from "./segment-loop";

// In SynthesisArtifactSchema .object({...}), add:
assigned_loops: z.array(
  z.object({
    id: z.string().min(1),
    pieceId: z.string().min(1),
    barsStart: z.number().int().positive(),
    barsEnd: z.number().int().positive(),
  })
).default([]),
```

In `apps/api/src/services/teacher.ts`, update `synthesizeV6` to include `pieceId` and `trigger`:
```typescript
// In synthesizeV6, update hookCtx construction:
const hookCtx: HookContext = {
  env: ctx.env,
  studentId: input.studentId,
  sessionId,
  conversationId: input.conversationId,
  digest,
  waitUntil: waitUntil ?? ((_p: Promise<unknown>) => {}),
  pieceId: input.pieceId ?? undefined,
  trigger: "synthesis",
};
```

Also add `pieceId` to `SynthesisInput` interface:
```typescript
export interface SynthesisInput {
  // ... existing fields ...
  pieceId?: string | null;
}
```

In `apps/api/src/do/session-brain.ts`, pass `pieceId` from state when building synthesis input:
```typescript
const synthInput: SynthesisInput = {
  // ... existing fields ...
  pieceId: state.pieceIdentification?.pieceId ?? null,
};
```

Update `buildV6WsPayload` to include loop components:
```typescript
export function buildV6WsPayload(
  artifact: SynthesisArtifact,
  loopComponents?: InlineComponent[],
): { type: "synthesis"; text: string; components: InlineComponent[]; isFallback: false } {
  return {
    type: "synthesis",
    text: artifact.headline,
    components: loopComponents ?? [],
    isFallback: false,
  };
}
```

In `apps/api/src/harness/loop/compound-registry.ts`, add `ASSIGN_SEGMENT_LOOP_TOOL` to `OnSessionEnd`:
```typescript
import { ASSIGN_SEGMENT_LOOP_TOOL } from "../../harness/atoms/assign-segment-loop";

// In OnSessionEnd binding:
tools: [...ALL_MOLECULES, ASSIGN_SEGMENT_LOOP_TOOL],
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run test -- --run src/harness/artifacts/synthesis.test.ts
```

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/artifacts/synthesis.ts apps/api/src/harness/artifacts/synthesis.test.ts apps/api/src/services/teacher.ts apps/api/src/harness/loop/compound-registry.ts apps/api/src/do/session-brain.ts && git commit -m "feat(v8a): synthesis path wiring — assigned_loops in SynthesisArtifact, pieceId/trigger in synthesizeV6"
```

---

### Task 10: Chat path wiring — chatV6 processToolFn intercept

**Group:** D (after Task 9, depends on Group C)

**Behavior being verified:** When the Anthropic stub returns `assign_segment_loop` tool_use in the streaming chat path, the `teacher-chat-v6.test.ts` fixture emits a `tool_result` SSE event with `componentsJson` containing a `segment_loop` component.
**Interface under test:** `chatV6` async generator — `tool_result` events

**Files:**
- Modify: `apps/api/src/services/teacher.ts` (`chatV6`)
- Modify: `apps/api/src/services/teacher-chat-v6.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// Add to apps/api/src/services/teacher-chat-v6.test.ts:
// (uses makeSseResponse, TEXT_ONLY_SSE, PHASE_CTX, and runPhase1Streaming already imported in the file)

const ASSIGN_LOOP_SSE = [
  'event: message_start\ndata: {"type":"message_start"}\n\n',
  'event: content_block_start\ndata: {"index":0,"content_block":{"type":"tool_use","id":"tu_loop","name":"assign_segment_loop"}}\n\n',
  'event: content_block_delta\ndata: {"index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"piece_id\\":\\"chopin.ballades.1\\",\\"bars_start\\":12,\\"bars_end\\":16,\\"required_correct\\":5}"}}\n\n',
  'event: content_block_stop\ndata: {"index":0}\n\n',
  'event: message_delta\ndata: {"delta":{"stop_reason":"tool_use"}}\n\n',
  "event: message_stop\ndata: {}\n\n",
].join("");

describe("runPhase1Streaming — assign_segment_loop intercept", () => {
  const fetchSpy = vi.fn();

  beforeEach(() => {
    fetchSpy.mockReset();
    vi.stubGlobal("fetch", fetchSpy);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("tool_result event carries segment_loop component when processToolFn intercepts assign_segment_loop", async () => {
    // Two fetch calls: tool_use turn + forced text continuation
    fetchSpy
      .mockImplementationOnce(() => Promise.resolve(makeSseResponse(ASSIGN_LOOP_SSE)))
      .mockImplementationOnce(() => Promise.resolve(makeSseResponse(TEXT_ONLY_SSE)));

    const mockArtifact = {
      kind: "segment_loop" as const,
      id: "loop-test-1",
      studentId: "stu_1",
      pieceId: "chopin.ballades.1",
      barsStart: 12,
      barsEnd: 16,
      requiredCorrect: 5,
      attemptsCompleted: 0,
      status: "pending" as const,
      dimension: null,
    };
    const mockComponent = {
      type: "segment_loop",
      config: { id: "loop-test-1", pieceId: "chopin.ballades.1", status: "pending" },
    };

    // processToolFn mirrors the intercept logic chatV6 installs
    const processToolFn = vi.fn().mockImplementation(
      async (name: string, _input: unknown) => {
        if (name === "assign_segment_loop") {
          return { name, componentsJson: [mockComponent], isError: false };
        }
        return { name, componentsJson: [], isError: false };
      },
    );

    const binding = getCompoundBinding("OnChatMessage")!;
    const systemBlocks = [{ type: "text" as const, text: "You are a teacher." }];
    const messages = [{ role: "user" as const, content: "Practice bars 12-16." }];

    const events: TeacherEvent[] = [];
    for await (const ev of runPhase1Streaming(
      PHASE_CTX,
      binding,
      systemBlocks,
      messages,
      processToolFn,
    )) {
      events.push(ev);
    }

    // processToolFn called with the parsed tool input
    expect(processToolFn).toHaveBeenCalledWith("assign_segment_loop", {
      piece_id: "chopin.ballades.1",
      bars_start: 12,
      bars_end: 16,
      required_correct: 5,
    });

    // tool_result event is emitted with the segment_loop component
    const toolResult = events.find(
      (e): e is Extract<TeacherEvent, { type: "tool_result" }> =>
        e.type === "tool_result" && (e as Extract<TeacherEvent, { type: "tool_result" }>).name === "assign_segment_loop",
    );
    expect(toolResult).toBeDefined();
    expect(toolResult?.componentsJson).toHaveLength(1);
    expect(toolResult?.componentsJson[0]?.type).toBe("segment_loop");

    // component accumulates into the final done event
    const done = events.findLast((e) => e.type === "done");
    expect(done?.type).toBe("done");
    if (done?.type === "done") {
      expect(done.allComponents[0]?.type).toBe("segment_loop");
    }
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun run test -- --run src/services/teacher-chat-v6.test.ts
```
Expected: FAIL — `assign_segment_loop` dispatched to `processToolUse` which returns "Unknown tool" error

- [ ] **Step 3: Implement**

In `apps/api/src/services/teacher.ts`, update `chatV6` to:
1. Add `pieceId` and `trigger` to `hookCtx`
2. Intercept `assign_segment_loop` in the `processToolFn` closure

```typescript
import * as segmentLoopsService from "./segment-loops";
import { createDb } from "../db/client";

export async function* chatV6(
  ctx: ServiceContext,
  studentId: string,
  messages: Array<{ role: "user" | "assistant"; content: string | AnthropicContentBlock[] }>,
  dynamicContext: string,
  pieceId?: string,
): AsyncGenerator<TeacherEvent> {
  // ... existing systemBlocks construction ...

  const processToolFn: ProcessToolFn = async (name, input) => {
    if (name === "assign_segment_loop") {
      try {
        const db = createDb(ctx.env.HYPERDRIVE);
        const artifact = await segmentLoopsService.createSegmentLoop(db, {
          studentId,
          pieceId: (input as { piece_id?: string }).piece_id ?? "",
          conversationId: null,
          barsStart: (input as { bars_start: number }).bars_start,
          barsEnd: (input as { bars_end: number }).bars_end,
          requiredCorrect: (input as { required_correct?: number }).required_correct ?? 5,
          dimension: (input as { dimension?: string }).dimension ?? null,
          trigger: "chat",
        });
        const component = segmentLoopsService.toLoopComponent(artifact);
        return { name, componentsJson: [component], isError: false };
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return { name, componentsJson: [], isError: true, errorMessage: message };
      }
    }
    return processToolUse(ctx, studentId, name, input);
  };

  const hookCtx: HookContext = {
    env: ctx.env,
    studentId,
    sessionId: "",
    conversationId: null,
    digest: {},
    waitUntil: (_p: Promise<unknown>) => {},
    pieceId,
    trigger: "chat",
  };

  yield* runStreamingHook("OnChatMessage", hookCtx, processToolFn, systemBlocks, messages);
}
```

Update `routes/chat.ts` to pass `pieceId` to `chatV6` (currently not tracked in chat request; pass `undefined` — the LLM provides piece_id via tool input):
```typescript
// In the teacherFn call, chatV6 now accepts optional 5th arg but routes/chat.ts doesn't have pieceId
// Leave pieceId as undefined for now — the LLM supplies it via tool input
const teacherFn = c.env.HARNESS_V6_ENABLED === "true" ? teacherService.chatV6 : teacherService.chat;
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run test -- --run src/services/teacher-chat-v6.test.ts
```

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/services/teacher.ts && git commit -m "feat(v8a): chatV6 intercepts assign_segment_loop with trigger=chat"
```

---

### Task 11: DO session-brain hydration + live WS events

**Group:** E (parallel with Task 12, depends on Group D)

**Behavior being verified:** `buildV6WsPayload` with loop artifact produces `components` array; the DO broadcasts `segment_loop_status` on session start when active assignment exists.
**Interface under test:** `buildV6WsPayload` (exported, unit-testable); `session-brain.unit.test.ts` extension

**Files:**
- Modify: `apps/api/src/do/session-brain.ts`
- Modify: `apps/api/src/do/session-brain.schema.ts` (add `activeAssignment` to `SessionState`)
- Modify: `apps/api/src/do/session-brain.unit.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// Add to apps/api/src/do/session-brain.unit.test.ts:
import { buildV6WsPayload } from "./session-brain";
import type { SynthesisArtifact } from "../harness/artifacts/synthesis";

const ARTIFACT_WITH_LOOP: SynthesisArtifact = {
  session_id: "sess_1",
  synthesis_scope: "session",
  strengths: [],
  focus_areas: [],
  proposed_exercises: [],
  dominant_dimension: "phrasing",
  recurring_pattern: null,
  next_session_focus: null,
  diagnosis_refs: [],
  headline: "A".repeat(300),
  assigned_loops: [{ id: "loop-1", pieceId: "chopin.ballades.1", barsStart: 12, barsEnd: 16 }],
};

const LOOP_COMPONENTS = [{ type: "segment_loop", config: { id: "loop-1" } }];

test("buildV6WsPayload includes loop components when provided", () => {
  const payload = buildV6WsPayload(ARTIFACT_WITH_LOOP, LOOP_COMPONENTS);
  expect(payload.components).toHaveLength(1);
  expect(payload.components[0]?.type).toBe("segment_loop");
});

test("buildV6WsPayload with no loop components returns empty array", () => {
  const payload = buildV6WsPayload(ARTIFACT_WITH_LOOP);
  expect(payload.components).toHaveLength(0);
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun run test -- --run src/do/session-brain.unit.test.ts
```
Expected: FAIL — `buildV6WsPayload` doesn't accept second arg yet (signature from Task 9 adds it; if Task 9 is done this may already pass — verify)

- [ ] **Step 3: Implement**

Add `activeAssignment` to `SessionState` in `session-brain.schema.ts`:
```typescript
// In sessionStateSchema add:
activeAssignment: z
  .object({
    id: z.string(),
    pieceId: z.string(),
    barsStart: z.number().int(),
    barsEnd: z.number().int(),
    requiredCorrect: z.number().int(),
    attemptsCompleted: z.number().int(),
  })
  .nullable()
  .default(null),
```

In `session-brain.ts`, after synthesis produces `artifact`, collect loop components:
```typescript
// In runSynthesisAndPersist, after artifact is confirmed non-null:
import { toLoopComponent } from "../services/segment-loops";

const loopComponents = (artifact.assigned_loops ?? []).map((ref) =>
  toLoopComponent({
    kind: "segment_loop",
    id: ref.id,
    studentId: state.studentId,
    pieceId: ref.pieceId,
    barsStart: ref.barsStart,
    barsEnd: ref.barsEnd,
    requiredCorrect: 5, // display only; actual from DB
    attemptsCompleted: 0,
    status: "active",
    dimension: null,
  }),
);
const wsPayload = buildV6WsPayload(artifact, loopComponents);
```

Add session-start hydration: on WebSocket connect, after loading state, if `pieceIdentification` is set, call `findActiveForPiece` and broadcast:
```typescript
// After existing sockets closed and new socket established:
if (state.pieceIdentification) {
  const db = createDb(this.env.HYPERDRIVE);
  const activeLoop = await findActiveForPiece(
    db,
    state.studentId,
    state.pieceIdentification.pieceId,
  );
  if (activeLoop) {
    state.activeAssignment = {
      id: activeLoop.id,
      pieceId: activeLoop.pieceId,
      barsStart: activeLoop.barsStart,
      barsEnd: activeLoop.barsEnd,
      requiredCorrect: activeLoop.requiredCorrect,
      attemptsCompleted: activeLoop.attemptsCompleted,
    };
    this.sendWs(newSocket, {
      type: "segment_loop_status",
      assignment: state.activeAssignment,
    });
  }
}
```

Add chunk-processing detection: after bar analysis, if `state.activeAssignment` is set, run detector:
```typescript
// After WASM bar analysis, add:
if (state.activeAssignment && chunkBarRange) {
  const detector = new PassageLoopDetector();
  const attempt = detector.processPosition(
    { startBar: chunkBarRange[0], endBar: chunkBarRange[1], durationMs: 15000 },
    {
      kind: "segment_loop",
      ...state.activeAssignment,
      studentId: state.studentId,
      status: "active",
      dimension: null,
    },
  );
  if (attempt?.inBounds) {
    const db = createDb(this.env.HYPERDRIVE);
    const { attemptsCompleted, completedNow } = await incrementAttempts(
      db,
      state.activeAssignment.id,
      state.studentId,
    );
    state.activeAssignment.attemptsCompleted = attemptsCompleted;
    if (completedNow) state.activeAssignment = null;
    this.sendWs(ws, {
      type: "loop_attempt",
      assignment_id: state.activeAssignment?.id,
      attempts_completed: attemptsCompleted,
      completed_now: completedNow,
    });
  }
}
```

Note: The `PassageLoopDetector` in the DO should be instantiated per-DO instance and stored in non-persisted memory (like `previousChunkAudio`), not in `SessionState`, since it holds debounce state. Add it as a `WeakMap` entry:
```typescript
const detectorMap = new WeakMap<SessionBrain, PassageLoopDetector>();
// In the chunk handler, get or create:
const detector = detectorMap.get(this) ?? new PassageLoopDetector();
detectorMap.set(this, detector);
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run test -- --run src/do/session-brain.unit.test.ts
```

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/do/session-brain.ts apps/api/src/do/session-brain.schema.ts apps/api/src/do/session-brain.unit.test.ts && git commit -m "feat(v8a): DO hydrates active assignment on session start; broadcasts loop_attempt events"
```

---

### Task 12: Web SegmentLoopArtifact card + InlineCard registration + api.ts

**Group:** E (parallel with Task 11, depends on Group D)

**Behavior being verified:** `InlineCard` renders `SegmentLoopArtifact` card for `type='segment_loop'`; `api.ts` exposes typed accept/decline/dismiss methods.
**Interface under test:** `InlineCard` component switch; `api.ts` function exports

**Files:**
- Create: `apps/web/src/components/cards/SegmentLoopArtifact.tsx`
- Create: `apps/web/src/components/cards/SegmentLoopArtifact.test.tsx`
- Modify: `apps/web/src/components/InlineCard.tsx`
- Modify: `apps/web/src/lib/api.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/web/src/components/cards/SegmentLoopArtifact.test.tsx
import { render, screen } from "@testing-library/react";
import { test, expect } from "vitest";
import { SegmentLoopArtifactCard } from "./SegmentLoopArtifact";

const CONFIG = {
  id: "loop-1",
  pieceId: "chopin.ballades.1",
  barsStart: 12,
  barsEnd: 16,
  requiredCorrect: 5,
  attemptsCompleted: 2,
  status: "active",
  dimension: null,
};

test("renders bars and attempt counter", () => {
  render(<SegmentLoopArtifactCard config={CONFIG} />);
  expect(screen.getByText(/bars 12.{0,5}16/i)).toBeInTheDocument();
  expect(screen.getByText(/2\s*\/\s*5/)).toBeInTheDocument();
});

test("pending status shows Accept and Skip buttons", () => {
  render(<SegmentLoopArtifactCard config={{ ...CONFIG, status: "pending" }} />);
  expect(screen.getByRole("button", { name: /accept/i })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /skip/i })).toBeInTheDocument();
});

test("completed status shows completion message", () => {
  render(<SegmentLoopArtifactCard config={{ ...CONFIG, status: "completed" }} />);
  expect(screen.getByText(/complete/i)).toBeInTheDocument();
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run test -- --run src/components/cards/SegmentLoopArtifact.test.tsx
```
Expected: FAIL — `Cannot find module './SegmentLoopArtifact'`

- [ ] **Step 3: Implement**

```typescript
// apps/web/src/components/cards/SegmentLoopArtifact.tsx
import { useState } from "react";
import { acceptSegmentLoop, declineSegmentLoop, dismissSegmentLoop } from "../../lib/api";

interface SegmentLoopConfig {
  id: string;
  pieceId: string;
  barsStart: number;
  barsEnd: number;
  requiredCorrect: number;
  attemptsCompleted: number;
  status: "pending" | "active" | "completed" | "dismissed" | "superseded";
  dimension: string | null;
}

interface Props {
  config: SegmentLoopConfig;
}

export function SegmentLoopArtifactCard({ config }: Props) {
  const [status, setStatus] = useState(config.status);
  const [attempts, setAttempts] = useState(config.attemptsCompleted);

  if (status === "completed") {
    return (
      <div className="rounded-lg border p-4">
        <p className="font-medium text-green-700">
          Loop complete — bars {config.barsStart}–{config.barsEnd}
        </p>
      </div>
    );
  }

  if (status === "dismissed" || status === "superseded") {
    return (
      <div className="rounded-lg border p-4 opacity-50">
        <p className="text-sm text-gray-500">
          Bars {config.barsStart}–{config.barsEnd} — {status}
        </p>
      </div>
    );
  }

  return (
    <div className="rounded-lg border p-4 space-y-3">
      <div>
        <p className="font-medium">
          Practice loop: bars {config.barsStart}–{config.barsEnd}
        </p>
        {config.dimension && (
          <p className="text-sm text-gray-600">Focus: {config.dimension}</p>
        )}
      </div>

      {status === "active" && (
        <div className="flex items-center gap-2">
          <span className="text-sm">
            {attempts} / {config.requiredCorrect} attempts
          </span>
          <button
            type="button"
            onClick={async () => {
              await dismissSegmentLoop(config.id);
              setStatus("dismissed");
            }}
            className="ml-auto text-xs text-gray-500 underline"
          >
            Dismiss
          </button>
        </div>
      )}

      {status === "pending" && (
        <div className="flex gap-2">
          <button
            type="button"
            onClick={async () => {
              await acceptSegmentLoop(config.id);
              setStatus("active");
            }}
            className="rounded bg-indigo-600 px-3 py-1 text-sm text-white"
          >
            Accept
          </button>
          <button
            type="button"
            onClick={async () => {
              await declineSegmentLoop(config.id);
              setStatus("dismissed");
            }}
            className="rounded border px-3 py-1 text-sm"
          >
            Skip
          </button>
        </div>
      )}
    </div>
  );
}
```

In `apps/web/src/components/InlineCard.tsx`, add the `segment_loop` case:
```typescript
import { SegmentLoopArtifactCard } from "./cards/SegmentLoopArtifact";

// Add to switch:
case "segment_loop":
  return <SegmentLoopArtifactCard config={component.config as SegmentLoopConfig} />;
```

In `apps/web/src/lib/api.ts`, add typed methods:
```typescript
export async function acceptSegmentLoop(id: string): Promise<void> {
  await request(`/api/segment-loops/${id}/accept`, { method: "POST" });
}
export async function declineSegmentLoop(id: string): Promise<void> {
  await request(`/api/segment-loops/${id}/decline`, { method: "POST" });
}
export async function dismissSegmentLoop(id: string): Promise<void> {
  await request(`/api/segment-loops/${id}/dismiss`, { method: "POST" });
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run test -- --run src/components/cards/SegmentLoopArtifact.test.tsx
```

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/components/cards/SegmentLoopArtifact.tsx apps/web/src/components/cards/SegmentLoopArtifact.test.tsx apps/web/src/components/InlineCard.tsx apps/web/src/lib/api.ts && git commit -m "feat(v8a): SegmentLoopArtifact web card + InlineCard registration + api methods"
```

---

## Post-Build Verification

After all tasks complete, run the full test suite:

```bash
just test-api
```

Confirm:
- All existing tests still pass (no regressions in harness loop, synthesis, chat routes)
- `segment-loops.test.ts`, `passage-loop-detector.test.ts`, `assign-segment-loop.test.ts`, `SegmentLoopArtifact.test.tsx` all pass
- TypeScript compiles clean: `cd apps/api && bun run typecheck`
- Migration SQL is syntactically valid: `cd apps/api && bun run migrate` (against dev DB if available)

## Drizzle Migration Note

After creating `0003_segment_loops.sql`, run `bun run generate` from `apps/api` to ensure Drizzle's meta directory is updated, or manually place the SQL in the correct position. The migration must be applied to prod via `just migrate-prod` before deploying the feature.

---

## Challenge Review

### CEO Pass

**Premise — Right problem, right framing.**
`assign_segment_loop` is the missing link between LLM teacher decisions and durable practice state. Without it, any "practice this bar range" instruction the teacher generates is ephemeral — no tracking, no DO awareness, no completion signal. The plan solves the right problem.

**Scope — Appropriately bounded, no drift.**
12 tasks span two layers (DB + service + route + DO + web card). Nothing in scope exceeds what the spec requires. The hardest problem — threading `PhaseContext` through `ToolDefinition.invoke` so synthesis-path atoms get DB access — is solved in Task 3 and threaded correctly through Tasks 6 and 9. Nothing is being avoided.

**12-Month Alignment:**
```
CURRENT STATE               THIS PLAN                    12-MONTH IDEAL
---                         ---                          ---
LLM can suggest loops       LLM can create durable       Multi-atom action layer:
via text only; no DB        loop rows; DO tracks         loops, tempo targets,
state; DO has no            progress; web card shows     repertoire bookmarks —
assignment awareness        status + accept/skip         all created from teacher
                                                         tool calls, all DO-aware
```
The plan moves in the right direction. One tech debt flag: `services/segment-loops.ts` imports `InlineComponent` from `tool-processor` (chat infrastructure). That cross-layer coupling should be resolved before additional action atom services are added.

**Alternatives — Not documented in spec.**
The spec does not record why a new `segment_loops` table was chosen over extending `exercises` (which already has similar structure). This should be in the spec's Open Questions for future engineers.

---

### Engineering Pass

#### Architecture

Data flow for synthesis path:
```
synthesizeV6(input w/ pieceId)
  → hookCtx{ trigger:'synthesis', pieceId }
  → runHook('OnSessionEnd', hookCtx)
    → runPhase1(ctx, binding)
      → LLM calls assign_segment_loop
      → wrapToolCall(name, ctx, () => def.invoke(input, ctx))
        → assignSegmentLoopAtom(ctx, input)
          → createDb(ctx.env.HYPERDRIVE)
          → createSegmentLoop(db, { trigger:'synthesis' })
            → status = 'active'     ← CORRECT
  → runPhase2 → SynthesisArtifact{ assigned_loops: [...] }
  → buildV6WsPayload(artifact, loopComponents)
  → DO broadcasts synthesis WS message
```

Data flow for chat path:
```
chatV6(ctx, studentId, messages, dynamicContext, pieceId?)
  → runStreamingHook('OnChatMessage', hookCtx, processToolFn, ...)
    → LLM calls assign_segment_loop
    → processToolFn intercepts by name
      → createSegmentLoop(db, { trigger:'chat' })
        → status = 'pending'     ← CORRECT
      → returns { componentsJson: [{ type:'segment_loop' }] }
```

Both paths are architecturally sound. The two-path design is correctly separated.

**Security:** All mutations go through `assertOwnerAndLoad` which checks `row.studentId !== studentId` before any state change. No user-controlled data flows unvalidated to SQL — Drizzle parameterizes all inputs. No shell calls or prompt injection vectors introduced.

**Deployment safety:** Migration `0003` runs independently of the deploy. A deploy-before-migration window would produce 500s on `assign_segment_loop` calls (table not found). The `Drizzle Migration Note` at the bottom addresses this but the standard practice of migrate-first applies.

---

#### Module Depth Audit

| Module | Interface size | Implementation hides | Verdict |
|--------|---------------|---------------------|---------|
| `segment-loops.ts` (service) | 6 exports | transaction logic, supersede, ownership check, state machine | DEEP |
| `assign-segment-loop.ts` (atom) | 2 exports | Zod parse, precondition check, DB call, trigger routing | DEEP |
| `passage-loop-detector.ts` | 3 exports, 1 method | tolerance math, debounce | DEEP |
| `segment-loops.ts` (routes) | 3 POST routes | auth guard, error mapping | SHALLOW — justified; thin HTTP adapter is the correct pattern here |
| `SegmentLoopArtifact.tsx` | 1 component | 3-state render, 3 API calls | DEEP for UI |

All modules are correctly bounded.

---

#### Code Quality

**[RISK] (confidence: 7/10) — `ctx.trigger ?? "synthesis"` is a silent fallback.**
In `assign-segment-loop.ts`:
```typescript
trigger: ctx.trigger ?? "synthesis",
```
If `ctx.trigger` is `undefined` (possible since it's `trigger?: 'chat' | 'synthesis'`), the atom silently falls back to `'synthesis'`, which creates an `active` loop instead of failing visibly. This violates the project's "explicit exceptions, no silent fallbacks" rule. Should be: `if (!ctx.trigger) throw new ToolPreconditionError("trigger not set on PhaseContext")`.

**[RISK] (confidence: 9/10) — `routes/chat.ts` flag fix is a phantom change.**
Task 8 Step 3 instructs: change `c.env.HARNESS_V6_CHAT_ENABLED === "true"` to `c.env.HARNESS_V6_ENABLED === "true"`. Reading the actual `routes/chat.ts`: there is no such conditional. The file already calls `chatV6` unconditionally (no flag). The described line does not exist. A build subagent following this step will not find it, causing confusion. **Action for build agent: skip the `routes/chat.ts` modification step in Task 8 — nothing to change.**

**[RISK] (confidence: 8/10) — Task 6 Files section lists `apps/api/src/harness/skills/atoms/index.ts` as modified but provides no implementation code and the commit command omits it.**
The file is listed but never modified in the step-by-step instructions. If `atoms/index.ts` needs a re-export (or doesn't need one because Tasks 8/9 import directly from `assign-segment-loop.ts`), this must be clarified. The build agent will not know what to add. **Action: either add the re-export code to Task 6 Step 3 and the commit, or remove the file from the Files section.**

---

#### Test Philosophy Audit

**[BLOCKER] (confidence: 10/10) — Task 10 test body is entirely placeholder comments.**

Task 10 Step 1:
```typescript
test("chatV6 emits tool_result with segment_loop component when assign_segment_loop called", async () => {
  // fetchSpy configured to return streaming SSE response with assign_segment_loop tool_use
  // then a final text turn
  // (follow existing pattern from teacher-chat-v6.test.ts for streaming fixture setup)
  // Assert: collected events include { type: 'tool_result', name: 'assign_segment_loop', componentsJson }
  // where componentsJson[0].type === 'segment_loop'
});
```

The test body contains only comments — no actual test code. This is an explicit placeholder. The plan's own rules state: "Never write: 'TBD', 'TODO', 'implement later'... 'Similar to Task N' — repeat the code; the subagent may only read its own task." Task 10 is the only test that verifies the critical chat-path intercept behavior. Without a real test, the most important behavior (that `assign_segment_loop` in the chat stream creates a pending loop and returns a `segment_loop` component) is completely unverified. **This task must be rewritten with actual test code before execution.**

**[RISK] (confidence: 7/10) — Task 6 test mocks `createSegmentLoop` — an internal collaborator of `assignSegmentLoopAtom`.**
The mock (`vi.mock("../../services/segment-loops", ...)`) couples the test to the atom's internal wiring. The "synthesis trigger creates active loop" assertion:
```typescript
expect(mockCreateLoop).toHaveBeenCalledWith(
  expect.anything(),
  expect.objectContaining({ trigger: "synthesis", status: undefined }),
);
```
...checks that `createSegmentLoop` was *called* with certain args, not that the *artifact returned* has `status: 'active'`. If the service implementation changes (e.g., `createSegmentLoop` is renamed), the test breaks even if behavior is unchanged. The mock pattern is the only practical option given no test DB, but the assertion on `status: undefined` is vacuously true (that key doesn't exist on `CreateSegmentLoopInput`) and should be removed to avoid false confidence.

**[RISK] (confidence: 8/10) — Task 11 test will PASS at Step 2 without Task 11's implementation.**
Task 11 Step 1 tests `buildV6WsPayload(artifact, loopComponents)` — the 2-argument form. This signature change is implemented in **Task 9** (Group D). By the time Task 11 (Group E) runs, Task 9 is already complete. The test will pass before Task 11's Step 3 is implemented. Task 11 acknowledges this ("if Task 9 is done this may already pass — verify"), which breaks watch-it-fail discipline. The test covers no unique Task 11 behavior. The real Task 11 behavior — DO loading active assignment on WS connect and broadcasting `segment_loop_status` — has no test at all.

---

#### Test Coverage Gaps

```
[+] services/segment-loops.ts
    │
    ├── createSegmentLoop()
    │   ├── [TESTED ★]   route-level auth guard (Task 4 test) — smoke test only
    │   ├── [GAP]        chat trigger → status='pending'
    │   ├── [GAP]        synthesis trigger → status='active'
    │   └── [GAP]        supersede non-terminal loops before insert
    │
    ├── acceptSegmentLoop()
    │   ├── [TESTED ★]   auth guard → 401 (Task 4 test)
    │   └── [GAP]        accept when status != 'pending' → 409
    │
    ├── findActiveForPiece()
    │   └── [GAP]        returns null when no active loop
    │
    └── incrementAttempts()
        └── [GAP]        completedNow transition

[+] do/session-brain.ts (Task 11 changes)
    │
    ├── WS connect hydration
    │   └── [GAP]        broadcasts segment_loop_status when active assignment exists
    │
    └── Chunk processing
        └── [GAP]        broadcasts loop_attempt on inBounds detection
```

The service coverage gaps are accepted (no test DB). The DO hydration gap is more concerning — that's the primary user-visible behavior of Task 11, and it's completely untested.

---

#### Failure Modes

**Task 9 — `assigned_loops` required field breaks production synthesis immediately on deploy:**
The `SynthesisArtifactSchema` adds `assigned_loops` as a required field with no `.default([])`. The Phase 2 LLM currently outputs a JSON blob for `write_synthesis_artifact`. Unless the Phase 2 system prompt is updated to instruct the LLM to always include `assigned_loops: []`, every synthesis call will fail with `{ type: "validation_error" }`. This is a **silent regression** — synthesis degrades without crashing, and students get no summary. The plan does not update the Phase 2 system prompt.

**Task 11 — DO hibernation resets `PassageLoopDetector`:**
After hibernation/eviction, `detectorMap.get(this)` returns `undefined`, a new detector is created, and debounce state is lost. A student could play the same passage twice within 2 seconds across a hibernation boundary and receive credit twice. This is acknowledged as intentional ("non-persisted, intentional") but the double-credit edge case is not mentioned. Low probability in practice.

**Task 4 — `assertOwnerAndLoad` non-terminal check via `dismissSegmentLoop`:**
If a student dismisses an already-dismissed loop (e.g., double-tap), `assertNotTerminal` throws `ValidationError → 409`. This is correct behavior. But a concurrent dismiss + accept race could leave one call succeeding and one failing. Postgres serializes the updates, and the owner check + status check are separate operations (not atomic). A TOCTOU window exists but is extremely unlikely in practice.

---

### Presumption Inventory

| Assumption | Verdict | Reason |
|-----------|---------|--------|
| `DIMS_6` is exported from `../../lib/dims` | SAFE | Confirmed: `session-brain.ts` imports it |
| `createDb` is exported from `../db/client` | SAFE | Used throughout codebase |
| Migration `0003` is the correct next number | SAFE | Verified: existing migrations are `0000`–`0002` |
| `requireAuth` throws `HTTPException(401)` caught by Hono in untyped test app | SAFE | Confirmed: Hono has built-in HTTPException handling |
| `teacher-chat-v6.test.ts` exists with a streaming SSE fixture pattern | VALIDATE | Task 10 references it without the actual code; if the fixture doesn't exist, Task 10 has nothing to follow |
| Phase 2 prompt tells LLM to output `assigned_loops` field | RISKY | No prompt update is in the plan; if missing, `assigned_loops` required field breaks all synthesis |
| `routes/chat.ts` has `HARNESS_V6_CHAT_ENABLED` flag to fix | RISKY | Verified INVALID — the actual file calls `chatV6` unconditionally; the flag does not exist |
| `atoms/index.ts` needs modification | VALIDATE | Listed in Task 6 Files but no implementation provided; direct import pattern in Tasks 8/9 makes it unnecessary |
| `InlineComponent` can be imported from `./tool-processor` in the segment-loops service | SAFE | `tool-processor` exports it; works but cross-layer |

---

### Summary

```
[BLOCKER] count: 0  (both resolved in-plan — see edits below)
[RISK]    count: 5
[QUESTION] count: 0
```

**BLOCKERs resolved:**

1. **Task 9 — `assigned_loops` required field:** Changed to `.default([])`. `phase2.ts` uses `zodToJsonSchema(binding.artifactSchema)` to derive the LLM tool schema, so the default propagates automatically — `assigned_loops` becomes optional in the schema the LLM sees, and Zod fills in `[]` when the LLM omits it. Test updated: "without assigned_loops fails" → "without assigned_loops defaults to `[]`".

2. **Task 10 — Placeholder test:** Replaced with a full `runPhase1Streaming` SSE fixture test using `ASSIGN_LOOP_SSE` + `TEXT_ONLY_SSE`, asserting that `processToolFn` is called with the parsed `assign_segment_loop` input, and the `tool_result` event carries a `segment_loop` component that accumulates into `done.allComponents`.

**Risks to monitor during execution:**

- **Task 8:** The `routes/chat.ts` flag-fix step is a no-op — the file already calls `chatV6` unconditionally. Build agent should skip that sub-step.
- **Task 6:** `atoms/index.ts` listed in Files section but no implementation code provided. If it needs a re-export, add `export { ASSIGN_SEGMENT_LOOP_TOOL } from "./assign-segment-loop";`; if not needed (Tasks 8/9 import directly), remove it from the Files section.
- **Task 6:** `ctx.trigger ?? "synthesis"` is a silent fallback — change to an explicit throw if preferred.
- **Task 11:** `buildV6WsPayload` test passes before Step 3 runs (Task 9 adds the signature). The DO hydration and chunk-detection paths are not unit-tested; verify manually in dev after E2E.

---

VERDICT: PROCEED_WITH_CAUTION — monitor the five risks above during execution, especially Task 8 (phantom flag fix) and Task 11 (untested DO behavior).
