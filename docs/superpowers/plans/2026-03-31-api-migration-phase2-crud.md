# Phase 2: CRUD Endpoints + Data Migration

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port all CRUD endpoints from the Rust API to the new Hono + Drizzle stack, add SSE chat streaming, and create a D1-to-Postgres migration script.

**Architecture:** Each route module is a chained Hono app (for RPC type preservation) that calls stateless service functions via `ServiceContext`. Auth middleware extracts `studentId` from better-auth sessions. The LLM client routes through CF AI Gateway. Chat uses `streamSSE` from Hono for real-time Anthropic streaming.

**Tech Stack:** Hono, Drizzle ORM, better-auth, Zod, PlanetScale Postgres via Hyperdrive, CF AI Gateway, R2 buckets

**Style Guide:** ALL code must follow `apps/api/TS_STYLE.md`. Key rules: never destructure `c.env`, ServiceContext for DI, domain errors in services (never import HTTPException), chain `.route()` for RPC types, `console.log(JSON.stringify({...}))` for logging.

---

## File Structure

```
apps/api/src/
  lib/
    types.ts              -- MODIFY: add AI Gateway bindings to Bindings
    errors.ts             -- MODIFY: add ConflictError
    validate.ts           -- CREATE: reusable zValidator wrapper
  middleware/
    auth-session.ts       -- CREATE: better-auth session extractor
  routes/
    waitlist.ts           -- CREATE
    waitlist.test.ts      -- CREATE
    scores.ts             -- CREATE
    scores.test.ts        -- CREATE
    exercises.ts          -- CREATE
    exercises.test.ts     -- CREATE
    conversations.ts      -- CREATE
    conversations.test.ts -- CREATE
    sync.ts               -- CREATE
    sync.test.ts          -- CREATE
    chat.ts               -- CREATE
    chat.test.ts          -- CREATE
  services/
    waitlist.ts           -- CREATE
    scores.ts             -- CREATE
    exercises.ts          -- CREATE
    conversations.ts      -- CREATE
    sync.ts               -- CREATE
    llm.ts                -- CREATE: AI Gateway client
    memory.ts             -- CREATE: fact search + context builder
    chat.ts               -- CREATE: SSE chat orchestration
    prompts.ts            -- CREATE: LLM prompt templates
  index.ts                -- MODIFY: mount new routes
scripts/
  migrate-d1.ts           -- CREATE: one-time D1 -> Postgres migration
```

---

### Task 1: Auth Session Middleware

Auth middleware extracts the better-auth session from the request and populates `c.var.studentId`. Protected routes check this value. Phase 1 defined the `Variables` type with `studentId: string | null` but never populated it.

**Files:**
- Create: `apps/api/src/middleware/auth-session.ts`
- Modify: `apps/api/src/index.ts` (mount middleware on `/api/*`)

- [ ] **Step 1: Write the auth session middleware**

```typescript
// apps/api/src/middleware/auth-session.ts
import { createMiddleware } from "hono/factory";
import { HTTPException } from "hono/http-exception";
import type { Bindings, Variables } from "../lib/types";
import { createAuth } from "../lib/auth";
import { createDb } from "../db/client";

export const authSessionMiddleware = createMiddleware<{
  Bindings: Bindings;
  Variables: Variables;
}>(async (c, next) => {
  const db = createDb(c.env.HYPERDRIVE);
  const auth = createAuth(db, c.env);
  const session = await auth.api.getSession({ headers: c.req.raw.headers });
  c.set("studentId", session?.user?.id ?? null);
  await next();
});

/** Use in route handlers to assert authenticated access. Throws 401 if no session. */
export function requireAuth(studentId: string | null): asserts studentId is string {
  if (!studentId) {
    throw new HTTPException(401, { message: "Authentication required" });
  }
}
```

- [ ] **Step 2: Mount the middleware in index.ts**

In `apps/api/src/index.ts`, add the import and mount it on `/api/*` AFTER the db middleware:

```typescript
import { authSessionMiddleware } from "./middleware/auth-session";

// After dbMiddleware line:
app.use("/api/*", authSessionMiddleware);
```

- [ ] **Step 3: Commit**

```bash
git add apps/api/src/middleware/auth-session.ts apps/api/src/index.ts
git commit -m "feat(api): add auth session middleware with requireAuth helper"
```

---

### Task 2: Shared Validation Helper + ConflictError

The TS style guide defines a reusable `validate()` wrapper around `zValidator` that returns JSON errors instead of plain text. Every route module uses this.

**Files:**
- Create: `apps/api/src/lib/validate.ts`
- Modify: `apps/api/src/lib/errors.ts` (add ConflictError)
- Modify: `apps/api/src/middleware/error-handler.ts` (map ConflictError)

- [ ] **Step 1: Create the validate helper**

```typescript
// apps/api/src/lib/validate.ts
import { zValidator } from "@hono/zod-validator";
import type { z } from "zod";

/**
 * Typed Zod validator that returns JSON error responses.
 * Wraps @hono/zod-validator with a hook that formats validation errors as JSON.
 */
export function validate<T extends z.ZodSchema>(
  target: "json" | "query" | "param",
  schema: T,
) {
  return zValidator(target, schema, (result, c) => {
    if (!result.success) {
      return c.json(
        { error: "Validation failed", issues: result.error.issues },
        400,
      );
    }
  });
}
```

- [ ] **Step 2: Add ConflictError to errors.ts**

Add to `apps/api/src/lib/errors.ts`:

```typescript
export class ConflictError extends DomainError {
  constructor(message: string) {
    super(message);
  }
}
```

And in `apps/api/src/middleware/error-handler.ts`, add the mapping after ValidationError:

```typescript
import { ConflictError } from "../lib/errors";

// Add after ValidationError check:
if (err instanceof ConflictError) {
  return c.json({ error: err.message }, 409);
}
```

- [ ] **Step 3: Commit**

```bash
git add apps/api/src/lib/validate.ts apps/api/src/lib/errors.ts apps/api/src/middleware/error-handler.ts
git commit -m "feat(api): add shared zod validation helper and ConflictError"
```

---

### Task 3: Waitlist Route + Service

The simplest endpoint. No auth required. Upsert by email with honeypot spam check.

**Rust behavior:** `POST /api/waitlist` accepts `{ email, context?, website? }`. If `website` field is non-empty, it's a bot (honeypot). Inserts or ignores on email conflict.

**Files:**
- Create: `apps/api/src/services/waitlist.ts`
- Create: `apps/api/src/routes/waitlist.ts`
- Create: `apps/api/src/routes/waitlist.test.ts`
- Modify: `apps/api/src/index.ts` (mount route)

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/routes/waitlist.test.ts
import { describe, it, expect } from "vitest";
import { Hono } from "hono";
import { waitlistRoutes } from "./waitlist";

const testApp = new Hono().route("/api/waitlist", waitlistRoutes);

describe("POST /api/waitlist", () => {
  it("returns 400 for missing email", async () => {
    const res = await testApp.request("/api/waitlist", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    expect(res.status).toBe(400);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd apps/api && bun run test -- --run waitlist.test.ts`
Expected: FAIL (module not found)

- [ ] **Step 3: Write the service**

```typescript
// apps/api/src/services/waitlist.ts
import type { ServiceContext } from "../lib/types";
import { waitlist } from "../db/schema/catalog";

export async function addToWaitlist(
  ctx: ServiceContext,
  data: { email: string; context?: string; source?: string },
) {
  await ctx.db
    .insert(waitlist)
    .values({
      email: data.email,
      context: data.context ?? null,
      source: data.source ?? "web",
    })
    .onConflictDoNothing({ target: waitlist.email });

  return { success: true };
}
```

- [ ] **Step 4: Write the route**

```typescript
// apps/api/src/routes/waitlist.ts
import { Hono } from "hono";
import { z } from "zod";
import type { Bindings, Variables } from "../lib/types";
import { validate } from "../lib/validate";
import * as waitlistService from "../services/waitlist";

const waitlistSchema = z.object({
  email: z.string().email(),
  context: z.string().optional(),
  source: z.string().optional(),
  website: z.string().optional(), // honeypot
});

const app = new Hono<{ Bindings: Bindings; Variables: Variables }>().post(
  "/",
  validate("json", waitlistSchema),
  async (c) => {
    const body = c.req.valid("json");

    // Honeypot: bots fill hidden fields
    if (body.website) {
      return c.json({ success: true }); // silent success to not tip off bot
    }

    const result = await waitlistService.addToWaitlist(
      { db: c.var.db, env: c.env },
      body,
    );
    return c.json(result, 201);
  },
);

export { app as waitlistRoutes };
```

- [ ] **Step 5: Mount in index.ts and run tests**

Add import and chain in the routes:

```typescript
import { waitlistRoutes } from "./routes/waitlist";

// Add to route chain:
const routes = app
  .route("/health", healthRoutes)
  .route("/api/auth", authRoutes)
  .route("/api/waitlist", waitlistRoutes);
```

Run: `cd apps/api && bun run test -- --run waitlist.test.ts`

- [ ] **Step 6: Commit**

```bash
git add apps/api/src/services/waitlist.ts apps/api/src/routes/waitlist.ts apps/api/src/routes/waitlist.test.ts apps/api/src/index.ts
git commit -m "feat(api): add waitlist endpoint with honeypot"
```

---

### Task 4: Scores/Pieces Routes + Service

Read-only piece catalog. No auth required. Three endpoints: list all pieces, get piece metadata, get piece score data from R2.

**Rust behavior:**
- `GET /api/scores` returns all pieces ordered by composer, title. Optional `?composer=X` filter.
- `GET /api/scores/:pieceId` returns single piece metadata.
- `GET /api/scores/:pieceId/data` fetches raw JSON from R2 at `scores/v1/{pieceId}.json` with immutable cache headers.

**Files:**
- Create: `apps/api/src/services/scores.ts`
- Create: `apps/api/src/routes/scores.ts`
- Create: `apps/api/src/routes/scores.test.ts`
- Modify: `apps/api/src/index.ts` (mount route)

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/routes/scores.test.ts
import { describe, it, expect } from "vitest";
import { Hono } from "hono";
import { scoresRoutes } from "./scores";

const testApp = new Hono().route("/api/scores", scoresRoutes);

describe("GET /api/scores", () => {
  it("returns 200 with pieces array", async () => {
    const res = await testApp.request("/api/scores");
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body).toHaveProperty("pieces");
    expect(Array.isArray(body.pieces)).toBe(true);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd apps/api && bun run test -- --run scores.test.ts`
Expected: FAIL (module not found)

- [ ] **Step 3: Write the service**

```typescript
// apps/api/src/services/scores.ts
import { eq, asc } from "drizzle-orm";
import type { ServiceContext } from "../lib/types";
import type { Bindings } from "../lib/types";
import { pieces } from "../db/schema/catalog";
import { NotFoundError } from "../lib/errors";

export async function listPieces(ctx: ServiceContext, composer?: string) {
  if (composer) {
    return ctx.db
      .select()
      .from(pieces)
      .where(eq(pieces.composer, composer))
      .orderBy(asc(pieces.title));
  }
  return ctx.db
    .select()
    .from(pieces)
    .orderBy(asc(pieces.composer), asc(pieces.title));
}

export async function getPiece(ctx: ServiceContext, pieceId: string) {
  const piece = await ctx.db.query.pieces.findFirst({
    where: (p, { eq }) => eq(p.pieceId, pieceId),
  });
  if (!piece) throw new NotFoundError("piece", pieceId);
  return piece;
}

export async function getPieceData(env: Bindings, pieceId: string) {
  const key = `scores/v1/${pieceId}.json`;
  const object = await env.SCORES.get(key);
  if (!object) throw new NotFoundError("score data", pieceId);
  return object;
}
```

- [ ] **Step 4: Write the route**

```typescript
// apps/api/src/routes/scores.ts
import { Hono } from "hono";
import { z } from "zod";
import type { Bindings, Variables } from "../lib/types";
import { validate } from "../lib/validate";
import * as scoresService from "../services/scores";

const app = new Hono<{ Bindings: Bindings; Variables: Variables }>()
  .get("/", async (c) => {
    const composer = c.req.query("composer");
    const ctx = { db: c.var.db, env: c.env };
    const pieceList = await scoresService.listPieces(ctx, composer);
    return c.json({ pieces: pieceList });
  })
  .get(
    "/:pieceId",
    validate("param", z.object({ pieceId: z.string().min(1) })),
    async (c) => {
      const { pieceId } = c.req.valid("param");
      const ctx = { db: c.var.db, env: c.env };
      const piece = await scoresService.getPiece(ctx, pieceId);
      return c.json(piece);
    },
  )
  .get(
    "/:pieceId/data",
    validate("param", z.object({ pieceId: z.string().min(1) })),
    async (c) => {
      const { pieceId } = c.req.valid("param");
      const r2Object = await scoresService.getPieceData(c.env, pieceId);
      return new Response(r2Object.body, {
        headers: {
          "Content-Type": "application/json",
          "Cache-Control": "public, max-age=31536000, immutable",
        },
      });
    },
  );

export { app as scoresRoutes };
```

- [ ] **Step 5: Mount in index.ts and run tests**

Add to route chain. Run: `cd apps/api && bun run test -- --run scores.test.ts`

- [ ] **Step 6: Commit**

```bash
git add apps/api/src/services/scores.ts apps/api/src/routes/scores.ts apps/api/src/routes/scores.test.ts apps/api/src/index.ts
git commit -m "feat(api): add scores/pieces read-only endpoints with R2 data"
```

---

### Task 5: Exercises Routes + Service

Three endpoints for exercise browsing and assignment tracking. All require auth.

**Rust behavior:**
- `GET /api/exercises?dimension=X&level=X&repertoire=X` returns up to 3 matching exercises with their dimensions. Joins `exercise_dimensions`, excludes already-assigned-and-incomplete for the student.
- `POST /api/exercises/assign` assigns an exercise to a student. Tracks `times_assigned`.
- `POST /api/exercises/complete` marks a student exercise as completed with optional response/dimension data.

**Files:**
- Create: `apps/api/src/services/exercises.ts`
- Create: `apps/api/src/routes/exercises.ts`
- Create: `apps/api/src/routes/exercises.test.ts`
- Modify: `apps/api/src/index.ts` (mount route)

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/routes/exercises.test.ts
import { describe, it, expect } from "vitest";
import { Hono } from "hono";
import { exercisesRoutes } from "./exercises";

const testApp = new Hono().route("/api/exercises", exercisesRoutes);

describe("GET /api/exercises", () => {
  it("returns 401 without auth", async () => {
    const res = await testApp.request("/api/exercises");
    expect(res.status).toBe(401);
  });
});

describe("POST /api/exercises/assign", () => {
  it("returns 401 without auth", async () => {
    const res = await testApp.request("/api/exercises/assign", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ exerciseId: "123" }),
    });
    expect(res.status).toBe(401);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd apps/api && bun run test -- --run exercises.test.ts`
Expected: FAIL (module not found)

- [ ] **Step 3: Write the service**

```typescript
// apps/api/src/services/exercises.ts
import { eq, and, sql, asc } from "drizzle-orm";
import type { ServiceContext } from "../lib/types";
import {
  exercises,
  exerciseDimensions,
  studentExercises,
} from "../db/schema/exercises";
import { NotFoundError } from "../lib/errors";

interface ExerciseFilters {
  studentId: string;
  dimension?: string;
  level?: string;
  repertoire?: string;
}

export async function listExercises(ctx: ServiceContext, filters: ExerciseFilters) {
  const conditions = [];
  if (filters.dimension) {
    conditions.push(eq(exerciseDimensions.dimension, filters.dimension));
  }
  if (filters.level) {
    conditions.push(eq(exercises.difficulty, filters.level));
  }

  const result = await ctx.db
    .selectDistinct({
      id: exercises.id,
      title: exercises.title,
      description: exercises.description,
      instructions: exercises.instructions,
      difficulty: exercises.difficulty,
      category: exercises.category,
      repertoireTags: exercises.repertoireTags,
      source: exercises.source,
    })
    .from(exercises)
    .innerJoin(exerciseDimensions, eq(exerciseDimensions.exerciseId, exercises.id))
    .where(conditions.length > 0 ? and(...conditions) : undefined)
    .orderBy(asc(exercises.title))
    .limit(3);

  // Fetch dimensions for each exercise
  const exerciseIds = result.map((e) => e.id);
  if (exerciseIds.length === 0) return [];

  const dims = await ctx.db
    .select({
      exerciseId: exerciseDimensions.exerciseId,
      dimension: exerciseDimensions.dimension,
    })
    .from(exerciseDimensions)
    .where(sql`${exerciseDimensions.exerciseId} = ANY(${exerciseIds})`);

  const dimMap = new Map<string, string[]>();
  for (const d of dims) {
    const list = dimMap.get(d.exerciseId) ?? [];
    list.push(d.dimension);
    dimMap.set(d.exerciseId, list);
  }

  return result.map((e) => ({ ...e, dimensions: dimMap.get(e.id) ?? [] }));
}

export async function assignExercise(
  ctx: ServiceContext,
  data: { studentId: string; exerciseId: string; sessionId?: string },
) {
  const exercise = await ctx.db.query.exercises.findFirst({
    where: (e, { eq }) => eq(e.id, data.exerciseId),
  });
  if (!exercise) throw new NotFoundError("exercise", data.exerciseId);

  const [assigned] = await ctx.db
    .insert(studentExercises)
    .values({
      studentId: data.studentId,
      exerciseId: data.exerciseId,
      sessionId: data.sessionId ?? null,
    })
    .onConflictDoUpdate({
      target: [studentExercises.studentId, studentExercises.exerciseId, studentExercises.sessionId],
      set: { timesAssigned: sql`${studentExercises.timesAssigned} + 1` },
    })
    .returning();

  return assigned;
}

export async function completeExercise(
  ctx: ServiceContext,
  data: {
    studentExerciseId: string;
    studentId: string;
    response?: string;
    dimensionBeforeJson?: unknown;
    dimensionAfterJson?: unknown;
    notes?: string;
  },
) {
  const existing = await ctx.db.query.studentExercises.findFirst({
    where: (se, { eq, and }) =>
      and(eq(se.id, data.studentExerciseId), eq(se.studentId, data.studentId)),
  });
  if (!existing) throw new NotFoundError("student_exercise", data.studentExerciseId);

  const [updated] = await ctx.db
    .update(studentExercises)
    .set({
      completed: true,
      response: data.response ?? null,
      dimensionBeforeJson: data.dimensionBeforeJson ?? null,
      dimensionAfterJson: data.dimensionAfterJson ?? null,
      notes: data.notes ?? null,
    })
    .where(eq(studentExercises.id, data.studentExerciseId))
    .returning();

  return updated;
}
```

- [ ] **Step 4: Write the route**

```typescript
// apps/api/src/routes/exercises.ts
import { Hono } from "hono";
import { z } from "zod";
import type { Bindings, Variables } from "../lib/types";
import { validate } from "../lib/validate";
import { requireAuth } from "../middleware/auth-session";
import * as exercisesService from "../services/exercises";

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

const app = new Hono<{ Bindings: Bindings; Variables: Variables }>()
  .get("/", async (c) => {
    requireAuth(c.var.studentId);
    const ctx = { db: c.var.db, env: c.env };
    const dimension = c.req.query("dimension");
    const level = c.req.query("level");
    const repertoire = c.req.query("repertoire");
    const result = await exercisesService.listExercises(ctx, {
      studentId: c.var.studentId,
      dimension,
      level,
      repertoire,
    });
    return c.json({ exercises: result });
  })
  .post("/assign", validate("json", assignSchema), async (c) => {
    requireAuth(c.var.studentId);
    const body = c.req.valid("json");
    const ctx = { db: c.var.db, env: c.env };
    const assigned = await exercisesService.assignExercise(ctx, {
      studentId: c.var.studentId,
      ...body,
    });
    return c.json(assigned, 201);
  })
  .post("/complete", validate("json", completeSchema), async (c) => {
    requireAuth(c.var.studentId);
    const body = c.req.valid("json");
    const ctx = { db: c.var.db, env: c.env };
    const completed = await exercisesService.completeExercise(ctx, {
      studentId: c.var.studentId,
      ...body,
    });
    return c.json(completed);
  });

export { app as exercisesRoutes };
```

- [ ] **Step 5: Mount in index.ts and run tests**

Add to route chain. Run: `cd apps/api && bun run test -- --run exercises.test.ts`

- [ ] **Step 6: Commit**

```bash
git add apps/api/src/services/exercises.ts apps/api/src/routes/exercises.ts apps/api/src/routes/exercises.test.ts apps/api/src/index.ts
git commit -m "feat(api): add exercises CRUD endpoints"
```

---

### Task 6: Conversations Routes + Service

Three endpoints for conversation management. All require auth.

**Rust behavior:**
- `GET /api/conversations` returns list with `id`, `title`, `updatedAt`, ordered by `updatedAt` DESC.
- `GET /api/conversations/:id` returns conversation with all messages ordered by `createdAt` ASC. Verifies ownership.
- `DELETE /api/conversations/:id` deletes conversation and all messages (FK cascade). Verifies ownership.

**Files:**
- Create: `apps/api/src/services/conversations.ts`
- Create: `apps/api/src/routes/conversations.ts`
- Create: `apps/api/src/routes/conversations.test.ts`
- Modify: `apps/api/src/index.ts` (mount route)

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/routes/conversations.test.ts
import { describe, it, expect } from "vitest";
import { Hono } from "hono";
import { conversationsRoutes } from "./conversations";

const testApp = new Hono().route("/api/conversations", conversationsRoutes);

describe("GET /api/conversations", () => {
  it("returns 401 without auth", async () => {
    const res = await testApp.request("/api/conversations");
    expect(res.status).toBe(401);
  });
});

describe("DELETE /api/conversations/:id", () => {
  it("returns 401 without auth", async () => {
    const res = await testApp.request(
      "/api/conversations/00000000-0000-0000-0000-000000000001",
      { method: "DELETE" },
    );
    expect(res.status).toBe(401);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd apps/api && bun run test -- --run conversations.test.ts`
Expected: FAIL (module not found)

- [ ] **Step 3: Write the service**

```typescript
// apps/api/src/services/conversations.ts
import { eq, and, desc, asc } from "drizzle-orm";
import type { ServiceContext } from "../lib/types";
import { conversations, messages } from "../db/schema/conversations";
import { NotFoundError } from "../lib/errors";

export async function listConversations(ctx: ServiceContext, studentId: string) {
  return ctx.db
    .select({
      id: conversations.id,
      title: conversations.title,
      updatedAt: conversations.updatedAt,
    })
    .from(conversations)
    .where(eq(conversations.studentId, studentId))
    .orderBy(desc(conversations.updatedAt));
}

export async function getConversation(
  ctx: ServiceContext,
  conversationId: string,
  studentId: string,
) {
  const conv = await ctx.db.query.conversations.findFirst({
    where: (c, { eq, and }) =>
      and(eq(c.id, conversationId), eq(c.studentId, studentId)),
  });
  if (!conv) throw new NotFoundError("conversation", conversationId);

  const msgs = await ctx.db
    .select()
    .from(messages)
    .where(eq(messages.conversationId, conversationId))
    .orderBy(asc(messages.createdAt));

  return { ...conv, messages: msgs };
}

export async function deleteConversation(
  ctx: ServiceContext,
  conversationId: string,
  studentId: string,
) {
  const conv = await ctx.db.query.conversations.findFirst({
    where: (c, { eq, and }) =>
      and(eq(c.id, conversationId), eq(c.studentId, studentId)),
  });
  if (!conv) throw new NotFoundError("conversation", conversationId);

  // FK cascade deletes messages
  await ctx.db
    .delete(conversations)
    .where(eq(conversations.id, conversationId));
}
```

- [ ] **Step 4: Write the route**

```typescript
// apps/api/src/routes/conversations.ts
import { Hono } from "hono";
import { z } from "zod";
import type { Bindings, Variables } from "../lib/types";
import { validate } from "../lib/validate";
import { requireAuth } from "../middleware/auth-session";
import * as conversationsService from "../services/conversations";

const app = new Hono<{ Bindings: Bindings; Variables: Variables }>()
  .get("/", async (c) => {
    requireAuth(c.var.studentId);
    const ctx = { db: c.var.db, env: c.env };
    const list = await conversationsService.listConversations(ctx, c.var.studentId);
    return c.json({ conversations: list });
  })
  .get(
    "/:id",
    validate("param", z.object({ id: z.string().uuid() })),
    async (c) => {
      requireAuth(c.var.studentId);
      const { id } = c.req.valid("param");
      const ctx = { db: c.var.db, env: c.env };
      const conv = await conversationsService.getConversation(ctx, id, c.var.studentId);
      return c.json(conv);
    },
  )
  .delete(
    "/:id",
    validate("param", z.object({ id: z.string().uuid() })),
    async (c) => {
      requireAuth(c.var.studentId);
      const { id } = c.req.valid("param");
      const ctx = { db: c.var.db, env: c.env };
      await conversationsService.deleteConversation(ctx, id, c.var.studentId);
      return c.json({ success: true });
    },
  );

export { app as conversationsRoutes };
```

- [ ] **Step 5: Mount in index.ts and run tests**

Add to route chain. Run: `cd apps/api && bun run test -- --run conversations.test.ts`

- [ ] **Step 6: Commit**

```bash
git add apps/api/src/services/conversations.ts apps/api/src/routes/conversations.ts apps/api/src/routes/conversations.test.ts apps/api/src/index.ts
git commit -m "feat(api): add conversations CRUD with ownership checks"
```

---

### Task 7: Sync Route + Service

iOS sync endpoint. Receives student profile deltas and new session data, upserts to Postgres.

**Rust behavior:**
- `POST /api/sync` accepts `{ student: StudentDelta, newSessions: SessionDelta[], lastSyncTimestamp? }`.
- Upserts student baselines (COALESCE to keep existing non-null values).
- Inserts new sessions (ignore on conflict).
- Returns `{ syncTimestamp, exerciseUpdates: [] }`.

**Files:**
- Create: `apps/api/src/services/sync.ts`
- Create: `apps/api/src/routes/sync.ts`
- Create: `apps/api/src/routes/sync.test.ts`
- Modify: `apps/api/src/index.ts` (mount route)

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/routes/sync.test.ts
import { describe, it, expect } from "vitest";
import { Hono } from "hono";
import { syncRoutes } from "./sync";

const testApp = new Hono().route("/api/sync", syncRoutes);

describe("POST /api/sync", () => {
  it("returns 401 without auth", async () => {
    const res = await testApp.request("/api/sync", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        student: {},
        newSessions: [],
      }),
    });
    expect(res.status).toBe(401);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd apps/api && bun run test -- --run sync.test.ts`
Expected: FAIL (module not found)

- [ ] **Step 3: Write the service**

```typescript
// apps/api/src/services/sync.ts
import { eq } from "drizzle-orm";
import type { ServiceContext } from "../lib/types";
import { students } from "../db/schema/students";
import { sessions } from "../db/schema/sessions";

interface StudentDelta {
  inferredLevel?: string;
  baselineDynamics?: number;
  baselineTiming?: number;
  baselinePedaling?: number;
  baselineArticulation?: number;
  baselinePhrasing?: number;
  baselineInterpretation?: number;
  baselineSessionCount?: number;
}

interface SessionDelta {
  id: string;
  startedAt: string;
  endedAt?: string;
  avgDynamics?: number;
  avgTiming?: number;
  avgPedaling?: number;
  avgArticulation?: number;
  avgPhrasing?: number;
  avgInterpretation?: number;
  observationsJson?: unknown;
  chunksSummaryJson?: unknown;
}

export interface SyncRequest {
  student: StudentDelta;
  newSessions: SessionDelta[];
  lastSyncTimestamp?: string;
}

export async function handleSync(
  ctx: ServiceContext,
  studentId: string,
  data: SyncRequest,
) {
  // Upsert student baselines
  const updateFields: Record<string, unknown> = { updatedAt: new Date() };
  if (data.student.inferredLevel !== undefined) updateFields.inferredLevel = data.student.inferredLevel;
  if (data.student.baselineDynamics !== undefined) updateFields.baselineDynamics = data.student.baselineDynamics;
  if (data.student.baselineTiming !== undefined) updateFields.baselineTiming = data.student.baselineTiming;
  if (data.student.baselinePedaling !== undefined) updateFields.baselinePedaling = data.student.baselinePedaling;
  if (data.student.baselineArticulation !== undefined) updateFields.baselineArticulation = data.student.baselineArticulation;
  if (data.student.baselinePhrasing !== undefined) updateFields.baselinePhrasing = data.student.baselinePhrasing;
  if (data.student.baselineInterpretation !== undefined) updateFields.baselineInterpretation = data.student.baselineInterpretation;
  if (data.student.baselineSessionCount !== undefined) updateFields.baselineSessionCount = data.student.baselineSessionCount;

  await ctx.db
    .update(students)
    .set(updateFields)
    .where(eq(students.studentId, studentId));

  // Insert new sessions (ignore on conflict)
  for (const session of data.newSessions) {
    await ctx.db
      .insert(sessions)
      .values({
        id: session.id,
        studentId,
        startedAt: new Date(session.startedAt),
        endedAt: session.endedAt ? new Date(session.endedAt) : null,
        avgDynamics: session.avgDynamics ?? null,
        avgTiming: session.avgTiming ?? null,
        avgPedaling: session.avgPedaling ?? null,
        avgArticulation: session.avgArticulation ?? null,
        avgPhrasing: session.avgPhrasing ?? null,
        avgInterpretation: session.avgInterpretation ?? null,
        observationsJson: session.observationsJson ?? null,
        chunksSummaryJson: session.chunksSummaryJson ?? null,
      })
      .onConflictDoNothing({ target: sessions.id });
  }

  return {
    syncTimestamp: new Date().toISOString(),
    exerciseUpdates: [],
  };
}
```

- [ ] **Step 4: Write the route**

```typescript
// apps/api/src/routes/sync.ts
import { Hono } from "hono";
import { z } from "zod";
import type { Bindings, Variables } from "../lib/types";
import { validate } from "../lib/validate";
import { requireAuth } from "../middleware/auth-session";
import * as syncService from "../services/sync";

const studentDeltaSchema = z.object({
  inferredLevel: z.string().optional(),
  baselineDynamics: z.number().optional(),
  baselineTiming: z.number().optional(),
  baselinePedaling: z.number().optional(),
  baselineArticulation: z.number().optional(),
  baselinePhrasing: z.number().optional(),
  baselineInterpretation: z.number().optional(),
  baselineSessionCount: z.number().int().optional(),
});

const sessionDeltaSchema = z.object({
  id: z.string().uuid(),
  startedAt: z.string(),
  endedAt: z.string().optional(),
  avgDynamics: z.number().optional(),
  avgTiming: z.number().optional(),
  avgPedaling: z.number().optional(),
  avgArticulation: z.number().optional(),
  avgPhrasing: z.number().optional(),
  avgInterpretation: z.number().optional(),
  observationsJson: z.unknown().optional(),
  chunksSummaryJson: z.unknown().optional(),
});

const syncSchema = z.object({
  student: studentDeltaSchema,
  newSessions: z.array(sessionDeltaSchema),
  lastSyncTimestamp: z.string().optional(),
});

const app = new Hono<{ Bindings: Bindings; Variables: Variables }>().post(
  "/",
  validate("json", syncSchema),
  async (c) => {
    requireAuth(c.var.studentId);
    const body = c.req.valid("json");
    const ctx = { db: c.var.db, env: c.env };
    const result = await syncService.handleSync(ctx, c.var.studentId, body);
    return c.json(result);
  },
);

export { app as syncRoutes };
```

- [ ] **Step 5: Mount in index.ts and run tests**

Add to route chain. Run: `cd apps/api && bun run test -- --run sync.test.ts`

- [ ] **Step 6: Commit**

```bash
git add apps/api/src/services/sync.ts apps/api/src/routes/sync.ts apps/api/src/routes/sync.test.ts apps/api/src/index.ts
git commit -m "feat(api): add sync endpoint for iOS delta sync"
```

---

### Task 8: LLM Client + AI Gateway

LLM client that routes through CF AI Gateway. Supports Anthropic (streaming + non-streaming), Groq, and Workers AI.

**Rust behavior:** `services/llm.rs` has `call_anthropic`, `call_anthropic_stream`, `call_groq`, `call_workers_ai`. All go through AI Gateway URLs. Anthropic streaming returns a ReadableStream of SSE events.

**Files:**
- Modify: `apps/api/src/lib/types.ts` (add AI Gateway URL bindings)
- Modify: `apps/api/wrangler.toml` (add AI Gateway vars)
- Create: `apps/api/src/services/llm.ts`
- Create: `apps/api/src/services/prompts.ts`

- [ ] **Step 1: Add AI Gateway bindings to types.ts**

Add to `Bindings` interface in `apps/api/src/lib/types.ts`:

```typescript
AI_GATEWAY_TEACHER: string;    // crescendai-teacher gateway
AI_GATEWAY_BACKGROUND: string; // crescendai-background gateway
```

Add to `wrangler.toml` `[vars]`:

```toml
AI_GATEWAY_TEACHER = ""
AI_GATEWAY_BACKGROUND = ""
```

- [ ] **Step 2: Write the LLM client service**

```typescript
// apps/api/src/services/llm.ts
import { InferenceError } from "../lib/errors";
import type { Bindings } from "../lib/types";

interface LlmMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

interface AnthropicRequest {
  model: string;
  max_tokens: number;
  system?: string;
  messages: LlmMessage[];
  stream?: boolean;
  tools?: unknown[];
  tool_choice?: unknown;
}

interface AnthropicResponse {
  content: Array<{
    type: string;
    text?: string;
    id?: string;
    name?: string;
    input?: unknown;
  }>;
  stop_reason: string;
  usage: { input_tokens: number; output_tokens: number };
}

export async function callAnthropic(
  env: Bindings,
  request: AnthropicRequest,
): Promise<AnthropicResponse> {
  const url = `${env.AI_GATEWAY_TEACHER}/anthropic/v1/messages`;
  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": env.ANTHROPIC_API_KEY,
      "anthropic-version": "2023-06-01",
    },
    body: JSON.stringify({ ...request, stream: false }),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new InferenceError(`Anthropic error ${res.status}: ${text}`);
  }

  return res.json() as Promise<AnthropicResponse>;
}

export async function callAnthropicStream(
  env: Bindings,
  request: AnthropicRequest,
): Promise<ReadableStream> {
  const url = `${env.AI_GATEWAY_TEACHER}/anthropic/v1/messages`;
  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": env.ANTHROPIC_API_KEY,
      "anthropic-version": "2023-06-01",
    },
    body: JSON.stringify({ ...request, stream: true }),
  });

  if (!res.ok || !res.body) {
    const text = await res.text();
    throw new InferenceError(`Anthropic stream error ${res.status}: ${text}`);
  }

  return res.body;
}

export async function callGroq(
  env: Bindings,
  model: string,
  messages: LlmMessage[],
  maxTokens = 1024,
): Promise<string> {
  const url = `${env.AI_GATEWAY_BACKGROUND}/groq/openai/v1/chat/completions`;
  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${env.GROQ_API_KEY}`,
    },
    body: JSON.stringify({ model, messages, max_tokens: maxTokens }),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new InferenceError(`Groq error ${res.status}: ${text}`);
  }

  const data = (await res.json()) as {
    choices: Array<{ message: { content: string } }>;
  };
  return data.choices[0]?.message?.content ?? "";
}
```

- [ ] **Step 3: Write prompt templates**

```typescript
// apps/api/src/services/prompts.ts

export const CHAT_SYSTEM = `You are a warm, encouraging piano teacher. You help students improve their playing through thoughtful conversation. You give specific, actionable advice grounded in the student's actual playing data when available.

Key principles:
- Celebrate strengths before suggesting improvements
- Frame observations, not absolute judgments
- Give actionable practice strategies
- Be specific about musical elements (dynamics, timing, pedaling, articulation, phrasing, interpretation)
- Adapt to the student's level and goals`;

export function buildChatUserContext(student: {
  inferredLevel?: string | null;
  explicitGoals?: string | null;
  baselines?: Record<string, number | null>;
}): string {
  const parts: string[] = [];
  if (student.inferredLevel) {
    parts.push(`Student level: ${student.inferredLevel}`);
  }
  if (student.explicitGoals) {
    parts.push(`Student goals: ${student.explicitGoals}`);
  }
  if (student.baselines) {
    const dims = Object.entries(student.baselines)
      .filter(([, v]) => v != null)
      .map(([k, v]) => `${k}: ${(v as number).toFixed(2)}`)
      .join(", ");
    if (dims) parts.push(`Current baselines: ${dims}`);
  }
  return parts.join("\n");
}

export function buildTitlePrompt(firstMessage: string): string {
  return `Generate a concise title (3-6 words, no quotes) for a piano lesson conversation that starts with: "${firstMessage}"`;
}
```

- [ ] **Step 4: Commit**

```bash
git add apps/api/src/lib/types.ts apps/api/src/services/llm.ts apps/api/src/services/prompts.ts apps/api/wrangler.toml
git commit -m "feat(api): add LLM client with AI Gateway routing and prompts"
```

---

### Task 9: Chat Route + Service (SSE Streaming)

The most complex Phase 2 endpoint. SSE streaming chat with Anthropic.

**Rust behavior:**
- `POST /api/chat` accepts `{ conversationId?, message }`.
- If no `conversationId`, creates a new conversation.
- Saves user message to DB.
- Builds context: student profile + memory facts + recent messages.
- Streams Anthropic response as SSE events: `start` (conversationId), `delta` (text chunks), `done`.
- Saves assistant message to DB after stream completes.
- Auto-generates title on first message (via Groq, backgrounded via `waitUntil`).

**Files:**
- Create: `apps/api/src/services/memory.ts` (fact search for chat context)
- Create: `apps/api/src/services/chat.ts`
- Create: `apps/api/src/routes/chat.ts`
- Create: `apps/api/src/routes/chat.test.ts`
- Modify: `apps/api/src/index.ts` (mount route)

- [ ] **Step 1: Write the memory service (chat dependency)**

```typescript
// apps/api/src/services/memory.ts
import { and, eq, isNull, desc } from "drizzle-orm";
import type { ServiceContext } from "../lib/types";
import { synthesizedFacts } from "../db/schema/memory";
import { observations } from "../db/schema/observations";

/**
 * Build memory context string for chat/teaching pipeline.
 * Returns recent active facts + recent observations for the student.
 */
export async function buildMemoryContext(
  ctx: ServiceContext,
  studentId: string,
): Promise<string> {
  const facts = await ctx.db
    .select({
      factText: synthesizedFacts.factText,
      factType: synthesizedFacts.factType,
      dimension: synthesizedFacts.dimension,
      confidence: synthesizedFacts.confidence,
    })
    .from(synthesizedFacts)
    .where(
      and(
        eq(synthesizedFacts.studentId, studentId),
        isNull(synthesizedFacts.invalidAt),
        isNull(synthesizedFacts.expiredAt),
      ),
    )
    .orderBy(desc(synthesizedFacts.validAt))
    .limit(12);

  const recentObs = await ctx.db
    .select({
      dimension: observations.dimension,
      observationText: observations.observationText,
      framing: observations.framing,
    })
    .from(observations)
    .where(eq(observations.studentId, studentId))
    .orderBy(desc(observations.createdAt))
    .limit(5);

  const parts: string[] = [];

  if (facts.length > 0) {
    parts.push("## Known Facts About This Student");
    for (const f of facts) {
      const dim = f.dimension ? ` [${f.dimension}]` : "";
      parts.push(`- (${f.factType}${dim}, ${f.confidence}) ${f.factText}`);
    }
  }

  if (recentObs.length > 0) {
    parts.push("\n## Recent Practice Observations");
    for (const o of recentObs) {
      const framing = o.framing ? ` (${o.framing})` : "";
      parts.push(`- [${o.dimension}${framing}] ${o.observationText}`);
    }
  }

  return parts.join("\n");
}
```

- [ ] **Step 2: Write the chat service**

```typescript
// apps/api/src/services/chat.ts
import { eq, asc } from "drizzle-orm";
import type { Bindings, Db, ServiceContext } from "../lib/types";
import { conversations, messages } from "../db/schema/conversations";
import { students } from "../db/schema/students";
import { NotFoundError } from "../lib/errors";
import { callAnthropicStream, callGroq } from "./llm";
import { buildMemoryContext } from "./memory";
import { CHAT_SYSTEM, buildChatUserContext, buildTitlePrompt } from "./prompts";

interface ChatInput {
  conversationId?: string;
  message: string;
}

interface ChatContext {
  conversationId: string;
  isNewConversation: boolean;
  stream: ReadableStream;
}

export async function handleChatStream(
  ctx: ServiceContext,
  studentId: string,
  input: ChatInput,
): Promise<ChatContext> {
  const db = ctx.db;
  const env = ctx.env;

  let conversationId = input.conversationId;
  let isNewConversation = false;

  if (conversationId) {
    const conv = await db.query.conversations.findFirst({
      where: (c, { eq, and }) =>
        and(eq(c.id, conversationId!), eq(c.studentId, studentId)),
    });
    if (!conv) throw new NotFoundError("conversation", conversationId);
  } else {
    const [newConv] = await db
      .insert(conversations)
      .values({ studentId })
      .returning();
    conversationId = newConv.id;
    isNewConversation = true;
  }

  // Save user message
  await db.insert(messages).values({
    conversationId,
    role: "user",
    content: input.message,
  });

  // Build LLM context
  const student = await db.query.students.findFirst({
    where: (s, { eq }) => eq(s.studentId, studentId),
  });

  const memoryContext = await buildMemoryContext(ctx, studentId);

  const recentMessages = await db
    .select({ role: messages.role, content: messages.content })
    .from(messages)
    .where(eq(messages.conversationId, conversationId))
    .orderBy(asc(messages.createdAt))
    .limit(20);

  const userContext = buildChatUserContext({
    inferredLevel: student?.inferredLevel,
    explicitGoals: student?.explicitGoals,
    baselines: student
      ? {
          dynamics: student.baselineDynamics,
          timing: student.baselineTiming,
          pedaling: student.baselinePedaling,
          articulation: student.baselineArticulation,
          phrasing: student.baselinePhrasing,
          interpretation: student.baselineInterpretation,
        }
      : undefined,
  });

  const systemPrompt = [CHAT_SYSTEM, userContext, memoryContext]
    .filter(Boolean)
    .join("\n\n");

  const anthropicMessages = recentMessages.map((m) => ({
    role: m.role as "user" | "assistant",
    content: m.content,
  }));

  const stream = await callAnthropicStream(env, {
    model: "claude-sonnet-4-20250514",
    max_tokens: 2048,
    system: systemPrompt,
    messages: anthropicMessages,
  });

  return { conversationId, isNewConversation, stream };
}

/**
 * Save the completed assistant message and generate title if needed.
 * Called via waitUntil after the stream finishes.
 */
export async function saveAssistantMessage(
  db: Db,
  env: Bindings,
  conversationId: string,
  content: string,
  isNewConversation: boolean,
  firstUserMessage: string,
) {
  await db.insert(messages).values({
    conversationId,
    role: "assistant",
    content,
  });

  await db
    .update(conversations)
    .set({ updatedAt: new Date() })
    .where(eq(conversations.id, conversationId));

  // Generate title for new conversations (best-effort)
  if (isNewConversation) {
    try {
      const title = await callGroq(
        env,
        "llama-3.3-70b-versatile",
        [{ role: "user", content: buildTitlePrompt(firstUserMessage) }],
        30,
      );
      if (title) {
        await db
          .update(conversations)
          .set({ title: title.trim() })
          .where(eq(conversations.id, conversationId));
      }
    } catch {
      console.log(
        JSON.stringify({
          level: "warn",
          message: "Title generation failed",
          conversationId,
        }),
      );
    }
  }
}
```

- [ ] **Step 3: Write the chat route with SSE streaming**

```typescript
// apps/api/src/routes/chat.ts
import { Hono } from "hono";
import { streamSSE } from "hono/streaming";
import { z } from "zod";
import type { Bindings, Variables } from "../lib/types";
import { validate } from "../lib/validate";
import { requireAuth } from "../middleware/auth-session";
import * as chatService from "../services/chat";

const chatSchema = z.object({
  conversationId: z.string().uuid().optional(),
  message: z.string().min(1).max(10000),
});

const app = new Hono<{ Bindings: Bindings; Variables: Variables }>().post(
  "/",
  validate("json", chatSchema),
  async (c) => {
    requireAuth(c.var.studentId);
    const body = c.req.valid("json");
    const ctx = { db: c.var.db, env: c.env };

    const { conversationId, isNewConversation, stream } =
      await chatService.handleChatStream(ctx, c.var.studentId, body);

    c.header("Content-Encoding", "Identity");

    return streamSSE(c, async (sseStream) => {
      await sseStream.writeSSE({
        data: JSON.stringify({ conversationId }),
        event: "start",
        id: "0",
      });

      const reader = stream.getReader();
      const decoder = new TextDecoder();
      let fullContent = "";
      let id = 1;

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value, { stream: true });
          fullContent += chunk;

          await sseStream.writeSSE({
            data: chunk,
            event: "delta",
            id: String(id++),
          });
        }
      } finally {
        reader.releaseLock();
      }

      await sseStream.writeSSE({
        data: "[DONE]",
        event: "done",
        id: String(id),
      });

      // Save message in background
      c.executionCtx.waitUntil(
        chatService.saveAssistantMessage(
          c.var.db,
          c.env,
          conversationId,
          fullContent,
          isNewConversation,
          body.message,
        ),
      );
    });
  },
);

export { app as chatRoutes };
```

- [ ] **Step 4: Write the test**

```typescript
// apps/api/src/routes/chat.test.ts
import { describe, it, expect } from "vitest";
import { Hono } from "hono";
import { chatRoutes } from "./chat";

const testApp = new Hono().route("/api/chat", chatRoutes);

describe("POST /api/chat", () => {
  it("returns 401 without auth", async () => {
    const res = await testApp.request("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: "Hello" }),
    });
    expect(res.status).toBe(401);
  });

  it("returns 400 for empty message", async () => {
    const res = await testApp.request("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: "" }),
    });
    expect(res.status).toBe(400);
  });
});
```

- [ ] **Step 5: Mount in index.ts and run tests**

Add to route chain. Run: `cd apps/api && bun run test -- --run chat.test.ts`

- [ ] **Step 6: Commit**

```bash
git add apps/api/src/services/memory.ts apps/api/src/services/chat.ts apps/api/src/routes/chat.ts apps/api/src/routes/chat.test.ts apps/api/src/index.ts
git commit -m "feat(api): add chat SSE streaming with Anthropic via AI Gateway"
```

---

### Task 10: Final Route Mounting + Type Export + Verification

Wire all routes into index.ts with proper chaining for Hono RPC type inference. Run full test suite and type check.

**Files:**
- Modify: `apps/api/src/index.ts`

- [ ] **Step 1: Verify final index.ts has all routes chained**

The final `index.ts` route chain must include ALL routes for proper `AppType` export:

```typescript
import { waitlistRoutes } from "./routes/waitlist";
import { scoresRoutes } from "./routes/scores";
import { exercisesRoutes } from "./routes/exercises";
import { conversationsRoutes } from "./routes/conversations";
import { syncRoutes } from "./routes/sync";
import { chatRoutes } from "./routes/chat";

const routes = app
  .route("/health", healthRoutes)
  .route("/api/auth", authRoutes)
  .route("/api/waitlist", waitlistRoutes)
  .route("/api/scores", scoresRoutes)
  .route("/api/exercises", exercisesRoutes)
  .route("/api/conversations", conversationsRoutes)
  .route("/api/sync", syncRoutes)
  .route("/api/chat", chatRoutes);

export type AppType = typeof routes;
```

- [ ] **Step 2: Run all tests**

Run: `cd apps/api && bun run test -- --run`
Expected: All tests pass (health + waitlist + scores + exercises + conversations + sync + chat).

- [ ] **Step 3: Run type check**

Run: `cd apps/api && bun run typecheck`
Expected: No errors.

- [ ] **Step 4: Verify with wrangler dev**

Run: `cd apps/api && wrangler dev`
Verify:
- `curl http://localhost:8787/health` -> 200
- `curl http://localhost:8787/api/scores` -> 200 (empty or populated pieces array)
- `curl -X POST http://localhost:8787/api/waitlist -H 'Content-Type: application/json' -d '{"email":"test@test.com"}'` -> 201
- `curl http://localhost:8787/api/conversations` -> 401
- `curl http://localhost:8787/api/exercises` -> 401
- `curl -X POST http://localhost:8787/api/sync -H 'Content-Type: application/json' -d '{"student":{},"newSessions":[]}'` -> 401

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/index.ts
git commit -m "feat(api): mount all Phase 2 routes with RPC type chain"
```

---

### Task 11: D1 to Postgres Data Migration Script

One-time migration script to move data from D1 (SQLite) to PlanetScale Postgres.

**Approach:** Read from D1 via Wrangler CLI (`wrangler d1 execute`), transform data types (TEXT timestamps -> Date objects, INTEGER booleans -> real booleans, TEXT JSON -> parsed objects), write to Postgres via Drizzle. Run locally with `DATABASE_URL` env var.

**Note:** Uses `execSync` to call wrangler CLI. This is a locally-run one-time script with hardcoded SQL queries (not user input). The security hook flagged this -- it's acceptable for a migration script that only the developer runs.

**Files:**
- Create: `apps/api/scripts/migrate-d1.ts`
- Modify: `apps/api/package.json` (add script)

- [ ] **Step 1: Write the migration script**

Create `apps/api/scripts/migrate-d1.ts` with:

1. Read `DATABASE_URL` and `D1_DATABASE_ID` from env
2. For each table (in FK-safe order: students -> auth_identities -> sessions -> ... -> waitlist):
   - Query D1 via `wrangler d1 execute --json`
   - Transform: TEXT timestamps -> `new Date()`, INTEGER booleans -> `true/false`, TEXT JSON -> `JSON.parse()`
   - Insert into Postgres via Drizzle with `.onConflictDoNothing()` (idempotent)
3. Log row counts for verification

Tables in FK-safe order:
1. students
2. auth_identities
3. sessions
4. student_check_ins
5. conversations
6. messages
7. observations
8. teaching_approaches
9. synthesized_facts
10. student_memory_meta
11. exercises
12. exercise_dimensions
13. student_exercises
14. pieces
15. piece_requests
16. waitlist

- [ ] **Step 2: Add migration script to package.json**

Add to `apps/api/package.json` scripts:

```json
"migrate-d1": "bun run scripts/migrate-d1.ts"
```

- [ ] **Step 3: Test the migration script locally**

Run: `cd apps/api && DATABASE_URL="<planetscale_url>" D1_DATABASE_ID="crescendai-db" bun run migrate-d1`

Verify row counts match between D1 and Postgres.

- [ ] **Step 4: Commit**

```bash
git add apps/api/scripts/migrate-d1.ts apps/api/package.json
git commit -m "feat(api): add D1 to Postgres one-time migration script"
```

---

## Validation Gate

After all tasks complete:

1. **All tests pass:** `cd apps/api && bun run test -- --run` (all route test files)
2. **Type check passes:** `cd apps/api && bun run typecheck`
3. **wrangler dev smoke test:** Health, scores, waitlist work; auth gates on protected routes
4. **AppType export is complete:** All routes visible in the chained type
5. **Migration script:** Runs successfully with row count verification (when D1 data exists)
