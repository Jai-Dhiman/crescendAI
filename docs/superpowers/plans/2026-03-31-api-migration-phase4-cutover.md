# Phase 4: Cutover — Web Client Migration + DNS Switch

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Switch the web app from the Rust API to the new Hono API. Migrate auth to better-auth client, replace raw fetch with Hono RPC where possible, deploy, migrate data, switch DNS, and decommission D1.

**Architecture:** The web app (`crescend.ai`) calls the API (`api.crescend.ai`). Currently it uses raw `fetch` with a hand-rolled API client (463 lines across `api.ts` + `practice-api.ts`). We're replacing the HTTP CRUD calls with Hono's typed `hc<AppType>()` client, migrating auth to better-auth's client SDK, and keeping manual `fetch`/`WebSocket` for SSE streaming and practice sessions.

**Tech Stack:** Hono RPC (`hc`), better-auth client (`@better-auth/client`), TanStack Start, bun

**Style Guide:** Web app code follows existing patterns in `apps/web/`. API code follows `apps/api/TS_STYLE.md`.

**What stays manual (no RPC):**
- SSE chat streaming (`POST /api/chat`) — RPC can't handle streaming responses
- WebSocket practice session (`/api/practice/ws/:id`) — not HTTP
- Chunk upload (`POST /api/practice/chunk`) — binary body
- Practice start (`POST /api/practice/start`) — could use RPC but keeping with practice-api for locality

---

## File Structure

```
apps/web/src/
  lib/
    api-client.ts          -- CREATE: hc<AppType>() typed RPC client
    auth-client.ts         -- CREATE: better-auth client (replaces custom auth calls)
    api.ts                 -- MODIFY: replace raw fetch with RPC client for CRUD
    practice-api.ts        -- MODIFY: update base URL handling, WS params
  hooks/
    useAuth.ts             -- MODIFY: use better-auth client
  components/
    AppChat.tsx            -- MODIFY: update extract-goals call
  routes/
    signin.tsx             -- MODIFY: use better-auth social sign-in
apps/api/src/
  routes/
    goals.ts               -- CREATE: port extract-goals from Rust
  services/
    goals.ts               -- CREATE: goal extraction service
  index.ts                 -- MODIFY: mount goals route
```

---

### Task 1: Port extract-goals Endpoint

The web app calls `POST /api/extract-goals` but this wasn't ported to the Hono API. Port it before changing the client.

**Rust behavior:** Takes `{ message: string }`, calls Workers AI to extract goals from natural language, merges into student's `explicit_goals` JSON column. Returns `{ goals: string[] }`.

**Files:**
- Create: `apps/api/src/services/goals.ts`
- Create: `apps/api/src/routes/goals.ts`
- Modify: `apps/api/src/index.ts` (mount route)

- [ ] **Step 1: Read the Rust implementation**

Read `apps/api-rust/src/services/goals.rs` for the exact logic.

- [ ] **Step 2: Write the service**

```typescript
// apps/api/src/services/goals.ts
import { eq } from "drizzle-orm";
import type { ServiceContext } from "../lib/types";
import { students } from "../db/schema/students";
import { callGroq } from "./llm";

export async function extractGoals(
  ctx: ServiceContext,
  studentId: string,
  message: string,
): Promise<string[]> {
  // Call Groq to extract goals from natural language
  const response = await callGroq(
    ctx.env,
    "llama-3.3-70b-versatile",
    [
      {
        role: "system",
        content: "Extract specific, actionable piano practice goals from the student's message. Return a JSON array of goal strings. Each goal should be concise (under 20 words). If no clear goals, return an empty array.",
      },
      { role: "user", content: message },
    ],
    200,
  );

  // Parse goals from LLM response
  let goals: string[] = [];
  try {
    const parsed = JSON.parse(response);
    if (Array.isArray(parsed)) {
      goals = parsed.filter((g): g is string => typeof g === "string");
    }
  } catch {
    // LLM didn't return valid JSON — try to extract from text
    goals = [];
  }

  if (goals.length === 0) return [];

  // Merge with existing goals
  const student = await ctx.db.query.students.findFirst({
    where: (s, { eq: e }) => e(s.studentId, studentId),
  });

  const existingGoals: string[] = student?.explicitGoals
    ? JSON.parse(student.explicitGoals)
    : [];

  const mergedGoals = [...new Set([...existingGoals, ...goals])];

  await ctx.db
    .update(students)
    .set({
      explicitGoals: JSON.stringify(mergedGoals),
      updatedAt: new Date(),
    })
    .where(eq(students.studentId, studentId));

  return goals;
}
```

- [ ] **Step 3: Write the route**

```typescript
// apps/api/src/routes/goals.ts
import { Hono } from "hono";
import { z } from "zod";
import type { Bindings, Variables } from "../lib/types";
import { validate } from "../lib/validate";
import { requireAuth } from "../middleware/auth-session";
import * as goalsService from "../services/goals";

const extractGoalsSchema = z.object({
  message: z.string().min(1).max(5000),
});

const app = new Hono<{ Bindings: Bindings; Variables: Variables }>().post(
  "/",
  validate("json", extractGoalsSchema),
  async (c) => {
    requireAuth(c.var.studentId);
    const { message } = c.req.valid("json");
    const ctx = { db: c.var.db, env: c.env };
    const goals = await goalsService.extractGoals(ctx, c.var.studentId, message);
    return c.json({ goals });
  },
);

export { app as goalsRoutes };
```

- [ ] **Step 4: Mount in index.ts**

Add to route chain: `.route("/api/extract-goals", goalsRoutes)`

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/services/goals.ts apps/api/src/routes/goals.ts apps/api/src/index.ts
git commit -m "feat(api): port extract-goals endpoint from Rust"
```

---

### Task 2: Create Hono RPC Client

Create the typed RPC client that the web app uses for all regular HTTP endpoints.

**Files:**
- Create: `apps/web/src/lib/api-client.ts`

- [ ] **Step 1: Install hono as a dev dependency in the web app**

```bash
cd apps/web && bun add hono
```

Hono is needed for the `hc` client and the `AppType` import.

- [ ] **Step 2: Create the RPC client**

```typescript
// apps/web/src/lib/api-client.ts
import { hc } from "hono/client";
import type { AppType } from "../../api/src/index";

const API_BASE = import.meta.env.PROD
  ? "https://api.crescend.ai"
  : "http://localhost:8787";

// Typed RPC client — rename a field in the API, get a TS error here
export const client = hc<AppType>(API_BASE, {
  init: {
    credentials: "include", // send HttpOnly cookie
  },
});

// Re-export for convenience
export type { AppType };
```

- [ ] **Step 3: Commit**

```bash
git add apps/web/src/lib/api-client.ts apps/web/package.json apps/web/bun.lockb
git commit -m "feat(web): create Hono RPC typed API client"
```

---

### Task 3: Create better-auth Client

Replace hand-rolled auth calls with better-auth's client SDK for sign-in, sign-out, and session management.

**Files:**
- Create: `apps/web/src/lib/auth-client.ts`
- Modify: `apps/web/src/lib/auth.tsx` (use better-auth client)
- Modify: `apps/web/src/hooks/useAuth.ts`
- Modify: `apps/web/src/routes/signin.tsx`

- [ ] **Step 1: Install better-auth client**

```bash
cd apps/web && bun add better-auth
```

The `better-auth` package includes both server and client code.

- [ ] **Step 2: Create auth client config**

```typescript
// apps/web/src/lib/auth-client.ts
import { createAuthClient } from "better-auth/client";

const API_BASE = import.meta.env.PROD
  ? "https://api.crescend.ai"
  : "http://localhost:8787";

export const authClient = createAuthClient({
  baseURL: API_BASE,
});
```

- [ ] **Step 3: Update auth.tsx to use better-auth session**

Replace the custom `GET /api/auth/me` call with `authClient.getSession()`.
The `useAuth` hook should call `authClient.useSession()` (React hook provided by better-auth).

- [ ] **Step 4: Update signin.tsx for better-auth social providers**

Replace custom Apple/Google auth calls:
- Apple: `authClient.signIn.social({ provider: "apple", callbackURL: "/" })`
- Google: `authClient.signIn.social({ provider: "google", callbackURL: "/" })`
- Sign out: `authClient.signOut()`

NOTE: The Apple JS SDK popup flow needs careful handling. better-auth expects a redirect-based flow, but the current code uses Apple's JS SDK to get an identity token client-side. Check better-auth docs for Apple social provider client-side integration.

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/lib/auth-client.ts apps/web/src/lib/auth.tsx apps/web/src/hooks/useAuth.ts apps/web/src/routes/signin.tsx
git commit -m "feat(web): migrate auth to better-auth client SDK"
```

---

### Task 4: Replace api.ts with RPC Client

Rewrite `api.ts` to use the Hono RPC client for CRUD endpoints. Keep raw fetch for SSE streaming.

**Files:**
- Modify: `apps/web/src/lib/api.ts`

- [ ] **Step 1: Read the current api.ts**

Read `apps/web/src/lib/api.ts` to understand all methods.

- [ ] **Step 2: Replace CRUD methods with RPC calls**

The Hono RPC client pattern: `client.api.exercises.$get({ query: { dimension: "dynamics" } })`

Methods to migrate to RPC:
- `api.chat.list()` → `client.api.conversations.$get()`
- `api.chat.get(id)` → `client.api.conversations[":id"].$get({ param: { id } })`
- `api.chat.delete(id)` → `client.api.conversations[":id"].$delete({ param: { id } })`
- `api.exercises.fetch(filters)` → `client.api.exercises.$get({ query: filters })`
- `api.exercises.assign(data)` → `client.api.exercises.assign.$post({ json: data })`
- `api.exercises.complete(data)` → `client.api.exercises.complete.$post({ json: data })`
- `api.waitlist.join(email, context)` → `client.api.waitlist.$post({ json: { email, context } })`
- `api.scores.list()` → `client.api.scores.$get()`
- `api.scores.get(id)` → `client.api.scores[":pieceId"].$get({ param: { pieceId: id } })`
- `checkNeedsSynthesis(cid)` → `client.api.practice["needs-synthesis"].$get({ query: { conversationId: cid } })`
- `triggerDeferredSynthesis(sid)` → `client.api.practice.synthesize.$post({ json: { sessionId: sid } })`
- `extractGoals(message)` → `client.api["extract-goals"].$post({ json: { message } })`

Methods that stay as raw fetch (SSE):
- `api.chat.send(conversationId, message)` — SSE streaming, stays as raw fetch

The RPC client returns `Response` objects. Use `.json()` to parse.

- [ ] **Step 3: Update all component imports**

Components currently import from `api.ts`. The interface should stay similar but internally use RPC.

- [ ] **Step 4: Commit**

```bash
git add apps/web/src/lib/api.ts
git commit -m "feat(web): replace raw fetch with Hono RPC for CRUD endpoints"
```

---

### Task 5: Update practice-api.ts

Update WebSocket connection params to match the new DO's expected query parameters.

**Files:**
- Modify: `apps/web/src/lib/practice-api.ts`

- [ ] **Step 1: Fix query parameter names**

The new DO expects `studentId` and `conversationId` (camelCase), not `student_id` and `conversation_id`. The route handler already sets these correctly (after our P1 fix), so the client-side WebSocket URL construction should match.

Check that `connectWebSocket` builds the URL correctly:
```typescript
const wsUrl = `${WS_BASE}/api/practice/ws/${sessionId}?conversationId=${conversationId}`;
```

- [ ] **Step 2: Verify chunk upload matches new API**

The new `POST /chunk` expects query params `sessionId` (uuid) and `chunkIndex` (number). Verify the client sends these correctly.

- [ ] **Step 3: Verify WS message types match**

Compare `PracticeWsEvent` in `practice-api.ts` with the WebSocket messages the DO sends. The new DO sends:
- `connected`, `chunk_processed`, `observation`, `synthesis`, `piece_identified`, `piece_set`, `mode_change`

The web app expects `session_summary` which the new DO doesn't send (it was a Rust-specific event). The `piece_identified` event is new. Update the types.

- [ ] **Step 4: Commit**

```bash
git add apps/web/src/lib/practice-api.ts
git commit -m "feat(web): update practice API client for new Hono DO"
```

---

### Task 6: Update Component Call Sites

Fix any remaining raw `fetch` calls in components (especially the inline `extract-goals` call in AppChat.tsx).

**Files:**
- Modify: `apps/web/src/components/AppChat.tsx` (remove inline apiBase, use RPC)

- [ ] **Step 1: Fix AppChat.tsx:293 inline extract-goals call**

Replace the inline `fetch` with the RPC client call.

- [ ] **Step 2: Verify all call sites compile**

```bash
cd apps/web && bun run typecheck
```

- [ ] **Step 3: Commit**

```bash
git add apps/web/src/components/AppChat.tsx
git commit -m "fix(web): replace inline fetch calls with RPC client"
```

---

### Task 7: Deploy + Configure + Verify

Operational steps. These require user involvement.

- [ ] **Step 1: Configure production secrets**

```bash
cd apps/api
wrangler secret put AUTH_SECRET
wrangler secret put APPLE_CLIENT_SECRET
wrangler secret put GOOGLE_CLIENT_SECRET
wrangler secret put ANTHROPIC_API_KEY
wrangler secret put GROQ_API_KEY
wrangler secret put SENTRY_DSN
```

Set production vars in wrangler.toml:
```toml
[env.production.vars]
ENVIRONMENT = "production"
ALLOWED_ORIGIN = "https://crescend.ai"
BETTER_AUTH_URL = "https://api.crescend.ai"
AI_GATEWAY_TEACHER = "<gateway-url>"
AI_GATEWAY_BACKGROUND = "<gateway-url>"
MUQ_ENDPOINT = "<hf-endpoint-url>"
AMT_ENDPOINT = "<amt-service-url>"
```

- [ ] **Step 2: Deploy new API worker**

```bash
cd apps/api && wrangler deploy
```

Verify: `curl https://api.crescend.ai/health` returns `{"status":"ok","version":"2.0.0","stack":"hono"}`

- [ ] **Step 3: Run D1 -> Postgres data migration**

```bash
cd apps/api && DATABASE_URL="<planetscale_prod_url>" D1_DATABASE_ID="crescendai-db" bun run migrate-d1
```

Verify row counts match.

- [ ] **Step 4: Deploy updated web app**

```bash
cd apps/web && bun run deploy
```

- [ ] **Step 5: Smoke test**

- Sign in with Apple/Google
- Send a chat message (SSE streaming)
- Start a practice session (WebSocket)
- Upload a chunk
- Verify observations appear
- End session, verify synthesis
- List conversations
- Check exercises

- [ ] **Step 6: Performance validation**

```bash
# Cold start
curl -w "time_total: %{time_total}\n" -o /dev/null -s https://api.crescend.ai/health
# Should be <50ms after first request (cold start is the first one)

# Auth round-trip
time curl -s -b cookie.txt https://api.crescend.ai/api/auth/session
```

- [ ] **Step 7: Verify Sentry**

Check Sentry dashboard (`crescendai-api` project) for:
- Traces appearing
- No unexpected errors
- Breadcrumbs on API requests

---

### Task 8: Cleanup

After successful cutover validation.

- [ ] **Step 1: Remove D1 binding from wrangler.toml**

Remove the `[[d1_databases]]` section if it exists.

- [ ] **Step 2: Archive Rust crate**

```bash
# Option A: Move to archive
mv apps/api-rust apps/api-rust-archive

# Option B: Delete (after confirming WASM crates extracted successfully)
rm -rf apps/api-rust
```

- [ ] **Step 3: Clean up wrangler.toml**

- Remove `localConnectionString` with PlanetScale username (review finding P1)
- Set production Hyperdrive ID
- Remove any D1 references

- [ ] **Step 4: Update CLAUDE.md**

Update the API section to reflect the new stack (Hono, not Rust). Remove references to `api-rust/`, `RUST_STYLE.md`, etc.

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "chore: complete API migration cutover, archive Rust crate"
```

---

## Validation Gate

1. **Health check:** `GET /health` returns `{"status":"ok","version":"2.0.0","stack":"hono"}`
2. **Auth flow:** Sign in -> session -> authenticated request -> sign out works end-to-end
3. **Chat flow:** Send message -> SSE stream -> response saved -> title generated
4. **Practice flow:** Start -> upload chunks -> WS observations -> end -> synthesis
5. **CRUD:** Exercises, conversations, scores all work via RPC
6. **Data migration:** Row counts match D1 -> Postgres
7. **Performance:** Cold start <50ms, p95 response <200ms
8. **Sentry:** Traces and errors reporting correctly
9. **No D1 references remaining in production config**
