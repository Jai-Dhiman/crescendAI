# TypeScript Style Guide

## CrescendAI API -- Cloudflare Workers (Hono + Drizzle + better-auth)

Grounded in: Hono official docs (2026-03), Cloudflare Workers best practices, Drizzle ORM docs,
better-auth integration guides, and production patterns from the CF ecosystem.

This document defines coding standards for `apps/api/`. Claude Code must follow these rules when
editing any file under `apps/api/src/`.

---

## 1. CF Workers Hard Constraints

Runtime: V8 isolates on Cloudflare's edge. Not Node.js. Not Deno.

### What IS available

- Full ES2022+ syntax, `async`/`await`, `fetch`, `crypto`, `TextEncoder`/`TextDecoder`
- `nodejs_compat` flag enables: `AsyncLocalStorage`, `node:crypto`, `node:buffer`, many npm packages
- Web standard APIs: `Request`, `Response`, `Headers`, `URL`, `ReadableStream`, `WebSocket`
- CF-specific: `Hyperdrive`, `R2Bucket`, `DurableObjectNamespace`, `KVNamespace`, `Vectorize`

### What does NOT exist

- No filesystem (`fs`) in production (only in Vitest via miniflare)
- No `process.env` -- use `c.env` (Hono) or `env` (Worker fetch handler)
- No long-running processes -- max 30s CPU time (paid plan)
- No threads -- single-threaded event loop

### Memory and size limits

- 128MB memory per isolate
- 10MB Worker bundle size (after gzip)
- WASM modules count toward bundle size

---

## 2. Hono Patterns

### Handler structure

Always chain validators and handler in a single expression. This preserves type inference
for Hono RPC.

```typescript
// GOOD
const app = new Hono<{ Bindings: Bindings; Variables: Variables }>()
  .post(
    "/sessions",
    zValidator("json", createSessionSchema),
    async (c) => {
      const body = c.req.valid("json");
      const db = c.var.db;
      const result = await sessionService.create({ db, env: c.env }, body);
      return c.json(result, 201);
    },
  )
  .get("/:id", async (c) => {
    const id = c.req.param("id");
    const db = c.var.db;
    const session = await sessionService.getById({ db, env: c.env }, id);
    return c.json(session);
  });
```

```typescript
// BAD: separate handler function loses type inference
const handleCreate = async (c: Context) => { ... };
app.post("/sessions", handleCreate);
```

### Route composition (critical for Hono RPC)

Chain `.route()` calls and export the type from the CHAINED result.

```typescript
// index.ts
const app = new Hono<{ Bindings: Bindings; Variables: Variables }>();
// ... middleware ...

const routes = app
  .route("/health", healthRoutes)
  .route("/api/auth", authRoutes)
  .route("/api/exercises", exerciseRoutes)
  .route("/api/conversations", conversationRoutes);

export type AppType = typeof routes;
export default app;
```

```typescript
// BAD: discarding route() return value -- AppType has no route info
app.route("/api/exercises", exerciseRoutes);
app.route("/api/conversations", conversationRoutes);
export type AppType = typeof app; // untyped client
```

### c.env: never destructure service bindings

CF service bindings (R2, KV, DO, Hyperdrive) lose their `this` context when destructured.

```typescript
// GOOD
const value = await c.env.CHUNKS.get(key);
const id = c.env.SESSION_BRAIN.idFromName(sessionId);

// GOOD: pass env as argument to helpers
async function uploadChunk(env: Bindings, key: string, data: ArrayBuffer) {
  await env.CHUNKS.put(key, data);
}
```

```typescript
// BAD: "Illegal invocation" at runtime
const { CHUNKS, SESSION_BRAIN } = c.env;
await CHUNKS.get(key); // throws

// BAD: module-level caching (stale across requests in reused isolates)
let cachedEnv: Bindings;
```

### Middleware with typed Variables

Use `createMiddleware` with explicit generics.

```typescript
import { createMiddleware } from "hono/factory";

export const dbMiddleware = createMiddleware<{
  Bindings: Bindings;
  Variables: Variables;
}>(async (c, next) => {
  const db = createDb(c.env.HYPERDRIVE);
  c.set("db", db);
  await next();
});
```

### Zod validation with custom error response

Always provide a hook to return JSON errors (default is plain text).

```typescript
import { zValidator } from "@hono/zod-validator";

// Reusable wrapper that throws HTTPException on validation failure
function validate<T extends z.ZodSchema>(target: "json" | "query" | "param", schema: T) {
  return zValidator(target, schema, (result, c) => {
    if (!result.success) {
      return c.json(
        { error: "Validation failed", issues: result.error.issues },
        400,
      );
    }
  });
}

// Usage
app.post("/sessions", validate("json", createSessionSchema), async (c) => {
  const body = c.req.valid("json"); // fully typed
  // ...
});
```

### SSE streaming (LLM proxy)

Set `Content-Encoding: Identity` to prevent wrangler dev from buffering.

```typescript
import { streamSSE } from "hono/streaming";

app.post("/chat", async (c) => {
  c.header("Content-Encoding", "Identity");

  return streamSSE(c, async (stream) => {
    stream.onAbort(() => { /* cleanup */ });

    const upstream = await fetch(llmUrl, { ... });
    if (!upstream.ok || !upstream.body) {
      await stream.writeSSE({ data: JSON.stringify({ error: "Upstream failed" }), event: "error" });
      return;
    }

    const reader = upstream.body.getReader();
    const decoder = new TextDecoder();
    let id = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      await stream.writeSSE({ data: decoder.decode(value, { stream: true }), event: "delta", id: String(id++) });
    }

    await stream.writeSSE({ data: "[DONE]", event: "done", id: String(id) });
  });
});
```

### WebSocket routing to Durable Objects

Hono validates the upgrade, then proxies the raw request to the DO.

```typescript
app.get("/ws/:sessionId", async (c) => {
  if (c.req.header("Upgrade") !== "websocket") {
    return c.text("Expected WebSocket upgrade", 426);
  }
  const id = c.env.SESSION_BRAIN.idFromName(c.req.param("sessionId"));
  const stub = c.env.SESSION_BRAIN.get(id);
  return stub.fetch(c.req.raw);
});
```

Do NOT use `upgradeWebSocket` from `hono/cloudflare-workers` when routing to DOs.
That creates a WebSocket in the Worker itself, not in the DO.

---

## 3. Error Handling

### Two-layer pattern

Services throw domain errors. Handlers map them to HTTP responses.

```typescript
// lib/errors.ts -- domain errors (HTTP-agnostic)
export class NotFoundError extends Error { ... }
export class AuthenticationError extends Error { ... }
export class ValidationError extends Error { ... }
export class InferenceError extends Error { ... }

// services/memory.ts
async function getFact(ctx: ServiceContext, factId: string) {
  const fact = await ctx.db.query.synthesizedFacts.findFirst({
    where: (f, { eq }) => eq(f.id, factId),
  });
  if (!fact) throw new NotFoundError("fact", factId);
  return fact;
}

// middleware/error-handler.ts -- maps domain errors to HTTP
app.onError((err, c) => {
  if (err instanceof HTTPException) return err.getResponse();
  if (err instanceof NotFoundError) return c.json({ error: err.message }, 404);
  if (err instanceof AuthenticationError) return c.json({ error: err.message }, 401);
  if (err instanceof ValidationError) return c.json({ error: err.message }, 400);
  console.error(JSON.stringify({ level: "error", message: err.message, stack: err.stack }));
  return c.json({ error: "Internal server error" }, 500);
});
```

Services must NEVER import `HTTPException` or return HTTP status codes.

---

## 4. Service Layer

### ServiceContext pattern

All services receive a `ServiceContext` object, not individual dependencies.

```typescript
export interface ServiceContext {
  db: Db;
  env: Bindings;
}

// GOOD
async function createObservation(ctx: ServiceContext, data: NewObservation) {
  const [obs] = await ctx.db.insert(observations).values(data).returning();
  return obs;
}

// BAD: individual args (grows unmanageable)
async function createObservation(db: Db, env: Bindings, inference: InferenceClient, data: NewObservation) { ... }
```

### No module-level state

Services must be stateless functions. No cached DB connections, no singletons.

---

## 5. Drizzle ORM

### Connection setup (Hyperdrive)

Hyperdrive transparently handles prepared statement multiplexing. Do NOT disable `prepare`.

```typescript
import postgres from "postgres";
import { drizzle } from "drizzle-orm/postgres-js";
import * as schema from "./schema/index";

export function createDb(hyperdrive: Hyperdrive) {
  const sql = postgres(hyperdrive.connectionString);
  return drizzle(sql, { schema });
}
```

```typescript
// BAD: disabling prepare (was needed before June 2024, hurts performance now)
const sql = postgres(url, { prepare: false });
```

### Query patterns

```typescript
import { eq, and, gt, desc } from "drizzle-orm";

// Select with where
const student = await db.query.students.findFirst({
  where: (s, { eq }) => eq(s.studentId, id),
});

// Select with join
const result = await db
  .select({ session: sessions, observation: observations })
  .from(sessions)
  .leftJoin(observations, eq(observations.sessionId, sessions.id))
  .where(eq(sessions.studentId, studentId))
  .orderBy(desc(sessions.startedAt));

// Insert with returning
const [newSession] = await db
  .insert(sessions)
  .values({ studentId, startedAt: new Date() })
  .returning();

// Update
await db
  .update(sessions)
  .set({ endedAt: new Date() })
  .where(eq(sessions.id, sessionId));

// Upsert
await db
  .insert(waitlist)
  .values({ email, source: "web" })
  .onConflictDoNothing({ target: waitlist.email });
```

### Transactions

Use the `tx` parameter inside the callback. Never use the outer `db` inside a transaction.

```typescript
// GOOD
await db.transaction(async (tx) => {
  const [obs] = await tx.insert(observations).values(obsData).returning();
  await tx.insert(teachingApproaches).values({ observationId: obs.id, ... });
});

// BAD: uses db instead of tx (bypasses the transaction)
await db.transaction(async (tx) => {
  await db.insert(observations).values(obsData); // <-- db, not tx!
});
```

### Schema conventions

- `uuid().defaultRandom().primaryKey()` for primary keys
- `timestamp("col", { withTimezone: true }).notNull().defaultNow()` for timestamps
- `jsonb("col").$type<MyType>()` for typed JSON columns
- `.notNull()` on every column unless genuinely nullable
- Explicit foreign keys with `references()` and cascade behavior
- Index naming: `idx_{table}_{columns}`

---

## 6. better-auth

### Per-request instance (not singleton)

On CF Workers, env bindings are per-request. Auth must be created per-request.

```typescript
export function createAuth(db: Db, env: Bindings) {
  return betterAuth({
    database: drizzleAdapter(db, { provider: "pg" }),
    baseURL: env.BETTER_AUTH_URL,
    secret: env.AUTH_SECRET,
    socialProviders: {
      apple: { clientId: env.APPLE_WEB_SERVICES_ID, clientSecret: env.APPLE_CLIENT_SECRET },
      google: { clientId: env.GOOGLE_CLIENT_ID, clientSecret: env.GOOGLE_CLIENT_SECRET },
    },
    session: {
      expiresIn: 60 * 60 * 24 * 30,
      cookieCache: { enabled: false }, // disabled: bug #4203
    },
  });
}
```

### Getting session in protected routes

```typescript
// Middleware that populates session on context
app.use("/api/*", async (c, next) => {
  const db = c.var.db;
  const auth = createAuth(db, c.env);
  const session = await auth.api.getSession({ headers: c.req.raw.headers });
  c.set("studentId", session?.user?.id ?? null);
  await next();
});

// Protected handler
app.get("/api/me", (c) => {
  const studentId = c.var.studentId;
  if (!studentId) throw new HTTPException(401, { message: "Unauthorized" });
  // ...
});
```

### cookieCache bug (#4203)

When `cookieCache` is enabled alongside `secondaryStorage`, better-auth treats an expired
cookie cache as a logout instead of falling back to secondary storage. Users get randomly
logged out after the cache TTL expires. Keep `cookieCache.enabled: false` until fixed.
Cost: one extra DB read per session check.

---

## 7. Durable Objects

### Use the Hibernation API for WebSockets

```typescript
import { DurableObject } from "cloudflare:workers";

export class SessionBrain extends DurableObject<Bindings> {
  async fetch(request: Request): Promise<Response> {
    const pair = new WebSocketPair();
    const [client, server] = Object.values(pair);
    this.ctx.acceptWebSocket(server); // hibernation API
    server.serializeAttachment({ sessionId, connectedAt: Date.now() });
    return new Response(null, { status: 101, webSocket: client });
  }

  async webSocketMessage(ws: WebSocket, message: string | ArrayBuffer) {
    const state = ws.deserializeAttachment();
    // process message
  }

  async webSocketClose(ws: WebSocket, code: number, reason: string) {
    ws.close(code, reason);
  }
}
```

Never use `server.accept()` (standard API). It prevents hibernation and bills for idle time.

### State versioning across awaits

TypeScript DOs have no borrow checker. Another event can fire during an `await` and mutate
shared state. Use clone-before-await + compare-and-swap.

```typescript
// GOOD: snapshot state before async, re-read after
async processChunk(chunkData: ArrayBuffer) {
  let state = await this.ctx.storage.get<SessionState>("state");
  const stateVersion = state.version;

  const result = await fetch(muqEndpoint, { body: chunkData }); // yields

  // Re-read: another event may have mutated state
  state = await this.ctx.storage.get<SessionState>("state");
  if (state.version !== stateVersion) {
    // State was modified during our await -- handle conflict
    return;
  }

  state.observations.push(result);
  state.version++;
  await this.ctx.storage.put("state", state);
}
```

```typescript
// BAD: read once, use stale state across awaits
async processChunk(chunkData: ArrayBuffer) {
  const state = await this.ctx.storage.get<SessionState>("state");
  const result = await fetch(muqEndpoint, { body: chunkData }); // state may be stale
  state.observations.push(result); // overwrites concurrent changes
  await this.ctx.storage.put("state", state);
}
```

### Alarms

Use alarms for scheduled work. Never use `setTimeout` (prevents hibernation, not durable).

```typescript
// Schedule
const existing = await this.ctx.storage.getAlarm();
if (!existing) {
  await this.ctx.storage.setAlarm(Date.now() + 5 * 60 * 1000);
}

// Handle
async alarm() {
  await this.runSynthesis();
}
```

Never set alarms in the constructor without checking for existing ones (clobbers pending alarms).

### Validate state with Zod on every read

```typescript
const rawState = await this.ctx.storage.get("state");
const state = sessionStateSchema.parse(rawState);
```

---

## 8. Structured Logging

Production: `console.log(JSON.stringify({...}))`. CF Workers Logs auto-indexes JSON fields.

```typescript
console.log(JSON.stringify({
  level: "info",
  method: c.req.method,
  path: c.req.path,
  status: c.res.status,
  duration_ms: Date.now() - start,
  studentId: c.var.studentId,
}));
```

For errors, always include `message` and `stack`:

```typescript
console.error(JSON.stringify({
  level: "error",
  message: err.message,
  stack: err.stack,
  name: err.name,
}));
```

Never use `console.log("some string")` in production code. Always JSON.

---

## 9. Testing

### Framework: Vitest + @cloudflare/vitest-pool-workers

Tests run inside the actual Workers runtime (workerd via Miniflare). All bindings are available.

```typescript
import { describe, it, expect } from "vitest";
import app from "../index";

describe("GET /health", () => {
  it("returns ok", async () => {
    const res = await app.request("/health");
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.status).toBe("ok");
  });
});
```

### DO alarm testing

Alarms are NOT reset between tests. Clean up manually with `runDurableObjectAlarm()`.

### DB tests

Use PGlite (WASM Postgres in-memory) for fast unit tests when Hyperdrive isn't needed.

---

## 10. Import Conventions

```typescript
// Hono core
import { Hono } from "hono";
import { HTTPException } from "hono/http-exception";
import { createMiddleware } from "hono/factory";
import { cors } from "hono/cors";
import { streamSSE } from "hono/streaming";

// Validation
import { zValidator } from "@hono/zod-validator";
import { z } from "zod";

// Drizzle
import { eq, and, gt, desc, sql } from "drizzle-orm";
import { drizzle } from "drizzle-orm/postgres-js";
import postgres from "postgres";

// CF Workers
import { DurableObject } from "cloudflare:workers";

// Sentry
import * as Sentry from "@sentry/cloudflare";

// better-auth
import { betterAuth } from "better-auth";
import { drizzleAdapter } from "better-auth/adapters/drizzle";

// Hono RPC client (web app only)
import { hc } from "hono/client";
import type { AppType } from "@crescendai/api";
```

---

## 11. File Organization

- `routes/` -- Hono route modules. Each exports a chained Hono app.
- `middleware/` -- Hono middleware. Each exports a `createMiddleware` result.
- `services/` -- Business logic. Pure functions taking `ServiceContext`. No CF bindings.
- `do/` -- Durable Object classes. Each has a `.ts` (class) and `.schema.ts` (Zod schemas).
- `db/schema/` -- Drizzle table definitions. One file per domain.
- `db/migrations/` -- drizzle-kit generated SQL. Never hand-edit.
- `lib/` -- Shared utilities, types, error classes. No business logic.

### Naming

- Files: `kebab-case.ts`
- Exports: `camelCase` for functions/variables, `PascalCase` for types/classes
- Schema tables: `camelCase` in TypeScript, `snake_case` in SQL (Drizzle handles mapping)
- Route files export a named `{ xxxRoutes }` not a default export

---

## 12. What NOT to Do

- Do not use `any`. Use `unknown` and narrow with Zod or type guards.
- Do not store request-scoped data in module-level variables.
- Do not destructure CF service bindings from `c.env`.
- Do not use `console.log` with plain strings in production.
- Do not use `setTimeout`/`setInterval` in Durable Objects (use alarms).
- Do not use `server.accept()` for WebSockets (use `this.ctx.acceptWebSocket()`).
- Do not use `db` inside a `db.transaction()` callback (use `tx`).
- Do not create better-auth as a module-level singleton.
- Do not import `HTTPException` in service files.
- Do not hand-edit files in `db/migrations/`.
