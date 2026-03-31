# API Stack Migration: Rust/WASM to Hono + Drizzle + PlanetScale Postgres

**Date:** 2026-03-30
**Status:** Design approved
**Driver:** Long-term architecture (10-year plan)

## Context

CrescendAI's API is a Rust/Axum application compiled to WASM on Cloudflare Workers with D1 (SQLite) as the database and hand-rolled JWT auth. This served well for building the initial product, but creates friction at every layer:

- Rust compile times (10-30s per change) slow iteration
- WASM cold starts (800ms+) impact first-request latency
- D1 has no foreign keys, no full-text search, no proper relational integrity
- Hand-rolled JWT auth duplicates what battle-tested libraries provide
- No access to JS-only CF features (Workflows, Agents SDK, Vectorize, proper Sentry SDK)
- Two-language codebase (Rust + TypeScript) prevents type sharing between API and web app
- npm ecosystem inaccessible from Rust/WASM

## Decision

Migrate the API to **Hono + Drizzle ORM + better-auth + PlanetScale Postgres (via CF Hyperdrive)**, keeping Rust only as WASM modules for compute-heavy code (DTW, N-gram, score following). The web frontend (TanStack Start) stays unchanged. The two-worker architecture (api.crescend.ai + crescend.ai) stays unchanged.

This mirrors how Anthropic, OpenAI, and other AI product companies architect their stacks: separate frontend + separate API + Postgres.

## Target Stack

```
                    crescend.ai                    api.crescend.ai
                  +----------------+              +------------------------+
                  | TanStack       |  Hono RPC    | Hono API Worker        |
                  | Start          |----types---->|                        |
                  | (React 19)     |              | - better-auth          |
                  | Biome          |              | - Drizzle ORM          |
                  | Tailwind v4    |              | - Zod validation       |
                  +----------------+              | - WASM modules         |
                                                  |   (DTW, N-gram)        |
                                                  | - Durable Objects      |
                                                  |   (session brain)      |
                                                  +----------+-------------+
                                                             |
                                                +------------+------------+
                                                |            |            |
                                          CF Hyperdrive   HF Endpoints  CF AI Gateway
                                                |         (MuQ + AMT)   (Groq/Anthropic)
                                                |
                                        PlanetScale Postgres
                                          (Drizzle schema)
```

## What the TS Migration Unlocks

| Capability | Before (Rust/WASM) | After (Hono/TS) | CrescendAI Impact |
|---|---|---|---|
| CF Workflows | Not available | Durable execution with auto-resume | Session synthesis, deferred recovery |
| Agents SDK | Not available | DO + SQL + WebSocket + AI built-in | Teacher pipeline agent patterns |
| Sentry SDK | OTLP drain only | Full SDK (traces, breadcrumbs, crons) | Proper observability |
| Auto OTel tracing | Limited | Zero-code I/O instrumentation | Every KV/DO/fetch traced |
| Vectorize | No bindings | Native bindings | Future: semantic memory search |
| Hono RPC | N/A | End-to-end type safety, zero codegen | API rename = instant TS error in web |
| Hot reload | 10-30s Rust compile | Sub-second | Dev velocity |
| Cold starts | 800ms+ (multi-MB WASM) | <5ms | First-request latency |
| Vitest in Workers | Not available | Tests with real bindings | Integration testing |
| npm ecosystem | N/A | Full nodejs_compat | Sentry, better-auth, Drizzle, etc. |

## Project Structure

```
apps/
  api/                              -- Hono API (CF Worker)
    src/
      routes/                       -- Route modules (each exports a Hono app)
        auth.ts                     -- better-auth mount
        chat.ts                     -- LLM proxy, SSE streaming
        practice.ts                 -- Session start, chunk upload, synthesis
        exercises.ts                -- CRUD
        conversations.ts            -- CRUD
        scores.ts                   -- Piece library
        sync.ts                     -- Cross-device sync
        waitlist.ts
      middleware/                    -- Hono middleware
        auth.ts                     -- better-auth session extractor
        rate-limit.ts               -- Per-tier rate limiting
        logger.ts                   -- Structured JSON logging
        error-handler.ts            -- Global error -> JSON response
        sentry.ts                   -- Sentry integration
      do/                           -- Durable Object classes
        session-brain.ts            -- Practice state machine + WebSocket
        session-brain.schema.ts     -- Zod schemas for DO state
      services/                     -- Business logic (no CF bindings)
        teaching-pipeline.ts        -- Subagent + teacher LLM orchestration
        stop-classifier.ts          -- STOP classification
        piece-identify.ts           -- Delegates to WASM module
        observation.ts              -- Bar-aligned analysis, accumulator
        synthesis.ts                -- Session synthesis logic
        memory.ts                   -- Fact extraction + retrieval
      db/                           -- Drizzle schema + migrations
        schema/                     -- Table definitions
          students.ts
          sessions.ts
          observations.ts
          conversations.ts
          exercises.ts
          memory.ts
          index.ts                  -- Re-exports all tables
        migrations/                 -- drizzle-kit generated SQL
        client.ts                   -- DB factory (Hyperdrive -> postgres -> drizzle)
      wasm/                         -- Rust WASM modules
        score-follower/
          src/lib.rs
          Cargo.toml
          pkg/                      -- wasm-pack output
        piece-identify/
          src/lib.rs
          Cargo.toml
          pkg/
      lib/                          -- Pure utilities
        ai-gateway.ts               -- CF AI Gateway client
        inference.ts                -- HF endpoint client (MuQ + AMT)
      index.ts                      -- App composition + route mounting
    wrangler.toml
    drizzle.config.ts
  api-rust/                         -- Current Rust API, renamed from api/ (kept until cutover)
  web/                              -- TanStack Start (unchanged)
    src/
      lib/
        api-client.ts               -- hc<AppType>() typed RPC client

Note: The current apps/api/ (Rust) is renamed to apps/api-rust/ at the start of the
migration. The new Hono API takes apps/api/. This avoids confusion during the transition.
```

## Key Architectural Patterns

### Route composition

Each route file exports a Hono app, composed in index.ts:

```typescript
app.route('/api/auth', authRoutes)       // better-auth handles internally
app.route('/api/chat', chatRoutes)
app.route('/api/practice', practiceRoutes)
```

### Dependency injection via Hono context

No singletons. CF bindings come from c.env, Drizzle instance created per-request in middleware. Never destructure c.env (loses `this` binding). Never store request-scoped data in module-level variables (isolate reuse = data leaks).

### Service layer

Business logic separated from handlers. Services receive a `ServiceContext` object (`{ db, inference, aiGateway, env }`) rather than individual dependencies. Handlers are thin: validate -> call service -> respond. Services are unit testable with mock context.

Services throw domain errors (e.g., `StudentNotFoundError`, `InferenceTimeoutError`). Handlers catch and map to `HTTPException`. This keeps services HTTP-agnostic (reusable from DOs, Workflows, not just HTTP handlers).

### WASM boundary

Rust modules compiled with `wasm-pack --target bundler`. Imported in services. Complex data crosses the boundary via `serde-wasm-bindgen` (Rust structs <-> JS objects). Target: `wasm32-unknown-unknown`. No threading, no Tokio.

### Hono RPC for web client

API exports `type AppType = typeof app`. Web app uses `hc<AppType>(baseUrl)` for fully typed API calls with zero codegen. Rename a field in the API, get a TypeScript error in the web app immediately.

**Limitation:** Hono RPC covers HTTP routes only. The WebSocket path (`/api/practice/ws/{session_id}`) goes through the DO's fetch handler, not Hono routes. The `usePracticeSession` hook's WebSocket connection remains manually typed. Define shared WebSocket message types in a types file imported by both the DO and the web app.

### DO state validation and concurrency

Zod schemas validate state on every read from `ctx.storage`. This replaces the type safety Rust's serde provided at the storage boundary. Reload state at every async boundary (the DO may have been evicted between awaits).

**Critical: state versioning across awaits.** The current Rust DO uses `RefCell<SessionState>` with explicit borrow scoping, and the Rust compiler prevents holding borrows across await points. TypeScript has no equivalent. Another WebSocket message can fire during an `await` and mutate `this.state`, causing subtle corruption.

Pattern: clone-before-await + compare-and-swap. Before any async operation that reads and later writes state, snapshot the relevant fields. After the await, verify the snapshot matches current state before writing. If it doesn't, the operation was interleaved and must be retried or dropped.

This pattern must be applied to: inference dispatch (parallel MuQ + AMT), synthesis trigger, teaching moment selection, and any other path that reads state -> awaits -> writes state.

### Request validation

Use `@hono/zod-validator` (`zValidator` middleware) for request validation. Simpler than zod-openapi, no OpenAPIHono class required. OpenAPI docs can be added later if a public API is needed.

### Error handling

Global error handler via `app.onError()`. Throw `HTTPException` from `hono/http-exception` with status codes. No silent fallbacks.

### Structured logging

`console.log(JSON.stringify({...}))` for production (CF Workers Logs auto-indexes JSON fields). Hono's built-in `logger()` for dev only.

## Cloudflare Bindings Inventory (wrangler.toml)

```
[vars]
ENVIRONMENT = "development"          # dev/staging/production

[[hyperdrive]]
binding = "HYPERDRIVE"               # PlanetScale Postgres via Hyperdrive

[[d1_databases]]                     # REMOVE after cutover
binding = "DB"

[[r2_buckets]]
binding = "CHUNKS"                   # Audio chunk uploads
bucket_name = "crescendai-chunks"

[[r2_buckets]]
binding = "SCORES"                   # Score MIDI data
bucket_name = "crescendai-scores"

[[durable_objects.bindings]]
name = "SESSION_BRAIN"
class_name = "SessionBrain"

[secrets]                            # Set via wrangler secret put
# AUTH_SECRET                        -- better-auth secret
# APPLE_CLIENT_SECRET                -- Apple Sign In
# GOOGLE_CLIENT_ID                   -- Google OAuth
# GOOGLE_CLIENT_SECRET               -- Google OAuth
# SENTRY_DSN                         -- Sentry DSN
# MUQ_ENDPOINT                       -- HF MuQ inference URL
# AMT_ENDPOINT                       -- AMT service URL
# GROQ_API_KEY                       -- Groq LLM
# ANTHROPIC_API_KEY                  -- Anthropic LLM
```

Removed from current wrangler.toml:
- `DB` (D1) -- replaced by HYPERDRIVE after cutover
- `JWT_SECRET` -- replaced by better-auth's AUTH_SECRET

## Database: D1 to PlanetScale Postgres

### Schema improvements

| D1 (current) | Postgres/Drizzle (new) | Improvement |
|---|---|---|
| TEXT primary keys (UUIDs as strings) | `uuid().defaultRandom().primaryKey()` | Native UUID type, faster indexing |
| JSON columns (TEXT) | `jsonb()` | Indexable, queryable JSON operators |
| No foreign keys | Real FK constraints with `references()` | Referential integrity at DB level |
| INTEGER timestamps | `timestamp().defaultNow()` | Native date handling |
| No indexes | Composite indexes on hot queries | Query performance |
| Manual created_at | `timestamp().defaultNow().notNull()` | DB-level defaults |

### Schema conventions

- `generatedAlwaysAsIdentity()` for auto-increment IDs (serial is deprecated)
- Bounded `varchar(n)` instead of unbounded text where possible
- `jsonb` for flexible schema fields (accumulator, reasoning traces, dimension scores)
- FK cascade deletes for parent-child relationships (session -> observations)
- Composite index on `(student_id, created_at)` for "my recent sessions" query
- All columns `.notNull()` unless genuinely nullable

### Migration workflow

- Development: `drizzle-kit push` (fast, destructive, no migration files)
- Production: `drizzle-kit generate` then `drizzle-kit migrate` (auditable SQL committed to git)
- Never delete or reorder migration files after they've been applied

### Connection setup

Hyperdrive provides connection pooling + caching. Use `postgres` (Postgres.js) driver through Hyperdrive connection string. Create db instance per-request in middleware, not at module level. Hyperdrive is free on Workers Paid plan.

## Authentication: Hand-rolled JWT to better-auth

### What better-auth provides

- Apple Sign In + Google Sign In (social providers)
- Cookie-based sessions (stored in Postgres via Drizzle adapter)
- Bearer token support for mobile/API clients
- Session management, token refresh, signout
- MFA (available when needed)
- Account linking by email

### Integration with Hono

- Use `drizzleAdapter` with `provider: "pg"`
- Create auth instance per request in middleware (not singleton)
- `c.executionCtx.waitUntil()` required for background tasks (token cleanup, session writes)
- Disable `cookieCache` until bug #4203 is fixed (prevents unexpected 5-minute logouts)
- Routes auto-mounted at `/api/auth/*`

### Migration from current auth

- Current: HS256 JWT in HttpOnly cookie, 30-day expiry
- New: better-auth manages sessions in Postgres, issues its own cookies
- During transition: both systems run independently (old Rust API validates old JWTs, new Hono API uses better-auth)
- At cutover: invalidate old JWTs, users re-authenticate once

## Migration Strategy

Build new Hono API in parallel with existing Rust API. No dual-write complexity. Cut over when new stack reaches feature parity.

### Phase 1: Foundation

- Hono worker skeleton with middleware chain
- PlanetScale Postgres + Hyperdrive binding
- Drizzle schema matching current D1 tables (with Postgres improvements)
- better-auth with Apple + Google social providers
- Sentry SDK integration
- Structured logging
- Health check endpoint
- Vitest + @cloudflare/vitest-pool-workers test setup

**Validation gate:** Auth flow works end-to-end (sign in with Apple -> session -> authenticated request -> sign out).

### Phase 2: CRUD Endpoints + Data

- Exercises (CRUD + catalog)
- Conversations + messages (CRUD + list)
- Sync endpoints
- Waitlist
- Scores/pieces (read-only library)
- Chat endpoint (SSE streaming to LLM proxy)
- Data migration script (D1 -> Postgres, one-time)
- Note: D1 schema must be reverse-engineered from ~83 raw SQL statements scattered across Rust service files (ask.rs, memory.rs, chat.rs, sync.rs, exercises.rs, scores.rs). The Drizzle schema is the canonical definition going forward.

**Validation gate:** All CRUD endpoints pass integration tests. Data migration script runs successfully with row count verification.

### Phase 3: Complex Domain

- Practice Durable Objects (session brain state machine)
- WebSocket upgrade + hibernation API
- WASM module extraction (score-follower, piece-identify as standalone crates)
- Teaching pipeline (subagent + teacher LLM orchestration)
- STOP classifier
- Bar-aligned observation analysis
- Session synthesis via CF Workflow (DO alarm serializes SessionAccumulator + context, dispatches to Workflow for durable execution with auto-resume)
- Observation pacing
- AI Gateway routing (Groq/Anthropic/Workers AI)

**Validation gate:** Full practice session works end-to-end (record -> analyze -> observe -> synthesize). State machine transitions match Rust behavior.

### Phase 4: Cutover

- Web app switches to `hc<AppType>()` client
- DNS switch: api.crescend.ai -> new Hono worker
- D1 decommission
- Rust crate archived (moved to `apps/api-rust-archive/` or deleted)
- Performance validation: cold start <50ms, p95 response within 20% of Rust API
- Sentry verified and reporting

### Cutover criteria

- All endpoints passing integration tests
- Data migration validated (D1 -> Postgres, row counts match)
- Web app fully functional on new API client
- Performance baseline met
- Sentry configured and reporting
- iOS app auth flow tested (if applicable at that point)

## Workflow Improvements

### Hooks (cherry-picked from everything-claude-code)

| Hook | Type | Purpose |
|---|---|---|
| `config-protection` | PreToolUse | Block Claude from weakening Biome/TS configs to "fix" lint errors |
| `block-no-verify` | PreToolUse | Enforce git hooks mechanically |
| `commit-quality` | PreToolUse | Lint staged files + detect secrets + validate commit message |
| `suggest-compact` | PreToolUse | Suggest /compact every ~50 tool calls |

### Claude skills to create

| Skill | What it encodes |
|---|---|
| `hono-patterns` | Handler structure, middleware chain, c.env bindings (never destructure), HTTPException, app.route() composition, Zod validation with @hono/zod-openapi |
| `drizzle-patterns` | Schema conventions (generatedAlwaysAsIdentity, varchar, jsonb), migration workflow (generate for prod, push for dev), transactions, prepared statements |
| `cf-workers-patterns` | DO lifecycle (reload state at async boundaries), WebSocket hibernation API, alarm scheduling, waitUntil for background work, no module-level state |
| `better-auth-patterns` | Auth instance per request, disable cookieCache (bug #4203), waitUntil required, social provider setup |

### CLAUDE.md updates

Add "Key Patterns" section with explicit GOOD/BAD code examples for:

- Hono handler structure
- Drizzle query patterns
- DO state management
- WASM module invocation
- Error handling

This gives Claude concrete anchors for the new stack instead of falling back to generic Express/Prisma patterns.

## Testing Strategy

### Unit tests

- Services tested with mock DB (pure function + dependency injection)
- Zod schemas tested for validation edge cases

### Integration tests

- `@cloudflare/vitest-pool-workers` for handler tests with real bindings
- Runs inside actual Workers runtime (workerd via Miniflare)
- Isolated per-test-file storage

### DO tests

- `runDurableObjectAlarm()` helper for alarm testing
- Manual alarm cleanup between tests (alarms NOT reset automatically)
- WebSocket tests via Worker entry point

### DB tests

- PGlite (WASM Postgres in-memory) for fast parallel unit tests
- `drizzle-kit push` against PGlite (no migration files needed for test)

### WASM boundary tests

- Module loading: verify WASM modules instantiate correctly in Workers runtime
- Serialization: test serde-wasm-bindgen roundtrip (Rust structs <-> JS objects) for all boundary types
- Accuracy regression: run known inputs through extracted WASM modules and compare outputs against the original monolithic Rust implementation

### E2E tests

- Full practice session flow (record -> analyze -> observe -> synthesize)
- Auth flow (sign in -> session -> authenticated request -> sign out)
- Chat flow (send message -> SSE stream -> response)

## Tooling Decisions

| Tool | Decision | Rationale |
|---|---|---|
| Linter/formatter | Keep Biome | Stable, works, no reason to migrate to Oxlint |
| Build | Keep Vite (adopt Rolldown when Vite 8 ships) | Free speed win without config change |
| Package manager | bun | Already in use, good CF Workers support |
| Deployment platform | Stay on Cloudflare | All primitives needed are available |
| Void.cloud | Skip | Early access, vendor lock-in risk, already on CF |
| Vite+ toolchain | Wait for stability | Alpha, Biome covers lint/format today |

## Cost Impact

| Component | Current | New | Delta |
|---|---|---|---|
| D1 | Free tier | Removed | -$0 |
| PlanetScale Postgres | N/A | $5/mo (Scaler plan) | +$5 |
| Hyperdrive | N/A | Free (included in Workers Paid) | +$0 |
| Workers Paid | $5/mo | $5/mo (same) | $0 |
| Sentry | Already paying | Same | $0 |
| **Total** | **$5/mo** | **$10/mo** | **+$5/mo** |

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| DO migration introduces state machine bugs | Medium | High | Comprehensive integration tests, side-by-side behavior comparison |
| better-auth cookieCache bug (#4203) causes session issues | Medium | Medium | Disable cookieCache, accept one extra DB read per session check |
| PlanetScale Neki doesn't support needed PG features | Low | Medium | Using standard SQL features only, no exotic extensions |
| WASM module extraction breaks DTW/N-gram accuracy | Low | Low | Rust code unchanged, only compilation target changes |
| Hono RPC type inference breaks on complex routes | Low | Low | Fallback to manual type definitions where needed |

## Not In Scope

- Frontend framework change (TanStack Start stays)
- iOS app changes (follows web, uses same API)
- Model/ML pipeline changes
- Inference endpoint changes (MuQ + AMT unchanged)
- Landing page redesign
- Monetization/billing implementation
