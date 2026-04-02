# API Migration Phase 1: Foundation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up a new Hono API worker with Drizzle + PlanetScale Postgres + better-auth, replacing the Rust/WASM API's auth and database layers.

**Architecture:** Hono on Cloudflare Workers with Drizzle ORM connecting to PlanetScale Postgres via Hyperdrive. better-auth handles Apple + Google social auth with cookie-based sessions stored in Postgres. The existing Rust API continues running at api.crescend.ai; the new Hono API runs on a staging route until cutover.

**Tech Stack:** Hono, Drizzle ORM, postgres (Postgres.js driver), better-auth, @sentry/cloudflare, @cloudflare/vitest-pool-workers, Zod, bun

**Spec:** `docs/superpowers/specs/2026-03-30-api-stack-migration-design.md`

---

## File Structure

```
apps/api-rust/                          -- Current Rust API (renamed from apps/api/)
apps/api/                               -- New Hono API
  src/
    index.ts                            -- App composition, middleware chain, route mounting
    routes/
      auth.ts                           -- better-auth handler mount
      health.ts                         -- Health check endpoint
    middleware/
      db.ts                             -- Drizzle instance per-request
      error-handler.ts                  -- Global error -> JSON response
      logger.ts                         -- Structured JSON logging
      sentry.ts                         -- Sentry SDK middleware
    db/
      schema/
        students.ts                     -- students + auth_identities tables
        sessions.ts                     -- sessions + student_check_ins tables
        observations.ts                 -- observations + teaching_approaches tables
        conversations.ts                -- conversations + messages tables
        exercises.ts                    -- exercises + exercise_dimensions + student_exercises
        memory.ts                       -- synthesized_facts + student_memory_meta
        catalog.ts                      -- pieces + piece_requests + waitlist
        index.ts                        -- Re-exports all tables
      client.ts                         -- Drizzle client factory (Hyperdrive -> postgres -> drizzle)
      migrations/                       -- drizzle-kit generated (empty until first generate)
    lib/
      auth.ts                           -- better-auth instance factory
      errors.ts                         -- Domain error classes
      types.ts                          -- Shared types (ServiceContext, Bindings)
  wrangler.toml                         -- CF Worker config with Hyperdrive + bindings
  drizzle.config.ts                     -- Drizzle Kit configuration
  package.json
  tsconfig.json
  vitest.config.ts                      -- Vitest + CF Workers pool
```

---

### Task 1: Rename Current API and Scaffold New Project

**Files:**
- Rename: `apps/api/` -> `apps/api-rust/`
- Create: `apps/api/package.json`
- Create: `apps/api/tsconfig.json`
- Create: `apps/api/wrangler.toml`

- [ ] **Step 1: Rename the existing Rust API directory**

```bash
cd /Users/jdhiman/Documents/crescendai
mv apps/api apps/api-rust
```

- [ ] **Step 2: Update Justfile references from `apps/api` to `apps/api-rust`**

Read the Justfile and update all paths that reference `apps/api` for Rust commands (test-api, check-api, deploy-api, etc.) to `apps/api-rust`. Keep the new `apps/api` free for the Hono project.

- [ ] **Step 3: Create the new API directory and initialize**

```bash
mkdir -p apps/api/src/{routes,middleware,db/schema,db/migrations,lib}
cd apps/api
bun init -y
```

- [ ] **Step 4: Install dependencies**

```bash
cd apps/api
bun add hono @hono/zod-validator zod better-auth drizzle-orm postgres @sentry/cloudflare
bun add -d wrangler @cloudflare/vitest-pool-workers @cloudflare/workers-types vitest drizzle-kit typescript
```

- [ ] **Step 5: Create tsconfig.json**

Write `apps/api/tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ESNext",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "lib": ["ESNext"],
    "types": ["@cloudflare/workers-types/2023-07-01"],
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "allowImportingTsExtensions": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "jsxImportSource": "hono/jsx",
    "paths": {
      "@/*": ["./src/*"]
    }
  },
  "include": ["src/**/*.ts"],
  "exclude": ["node_modules"]
}
```

- [ ] **Step 6: Create wrangler.toml**

Write `apps/api/wrangler.toml`:

```toml
name = "crescendai-api-v2"
main = "src/index.ts"
compatibility_date = "2025-09-15"
compatibility_flags = ["nodejs_compat"]

[observability]
enabled = true

[observability.logs]
enabled = true
invocation_logs = true
head_sampling_rate = 1

[vars]
ENVIRONMENT = "development"
ALLOWED_ORIGIN = "http://localhost:3000"
APPLE_BUNDLE_ID = "ai.crescend.ios"
APPLE_WEB_SERVICES_ID = "ai.crescend.web"
GOOGLE_CLIENT_ID = "726399246264-lkq6jl4khal8gagn65j9t2a8bg52i7ie.apps.googleusercontent.com"
BETTER_AUTH_URL = "http://localhost:8787"

[[hyperdrive]]
binding = "HYPERDRIVE"
id = "PLACEHOLDER_HYPERDRIVE_ID"

[[r2_buckets]]
binding = "CHUNKS"
bucket_name = "crescendai-bucket"

[[r2_buckets]]
binding = "SCORES"
bucket_name = "crescendai-bucket"

[[durable_objects.bindings]]
name = "SESSION_BRAIN"
class_name = "SessionBrain"

# Secrets (set via `wrangler secret put`):
# AUTH_SECRET
# APPLE_CLIENT_SECRET
# GOOGLE_CLIENT_SECRET
# SENTRY_DSN
# HF_INFERENCE_ENDPOINT
# GROQ_API_KEY
# ANTHROPIC_API_KEY
```

Note: `PLACEHOLDER_HYPERDRIVE_ID` must be replaced after creating the Hyperdrive config via `wrangler hyperdrive create crescendai-pg --connection-string="<planetscale_connection_string>"`.

- [ ] **Step 7: Create drizzle.config.ts**

Write `apps/api/drizzle.config.ts`:

```typescript
import { defineConfig } from "drizzle-kit";

export default defineConfig({
	dialect: "postgresql",
	schema: "./src/db/schema/index.ts",
	out: "./src/db/migrations",
});
```

- [ ] **Step 8: Commit**

```bash
git add apps/api-rust apps/api Justfile
git commit -m "feat: scaffold Hono API project, rename Rust API to api-rust"
```

---

### Task 2: Shared Types and Domain Errors

**Files:**
- Create: `apps/api/src/lib/types.ts`
- Create: `apps/api/src/lib/errors.ts`

- [ ] **Step 1: Create Cloudflare bindings type and ServiceContext**

Write `apps/api/src/lib/types.ts`:

```typescript
import type { DrizzleD1Database } from "drizzle-orm/d1";
import type { PostgresJsDatabase } from "drizzle-orm/postgres-js";
import type * as schema from "../db/schema/index";

export type Db = PostgresJsDatabase<typeof schema>;

export interface Bindings {
	HYPERDRIVE: Hyperdrive;
	CHUNKS: R2Bucket;
	SCORES: R2Bucket;
	SESSION_BRAIN: DurableObjectNamespace;
	ENVIRONMENT: string;
	ALLOWED_ORIGIN: string;
	APPLE_BUNDLE_ID: string;
	APPLE_WEB_SERVICES_ID: string;
	GOOGLE_CLIENT_ID: string;
	BETTER_AUTH_URL: string;
	AUTH_SECRET: string;
	APPLE_CLIENT_SECRET: string;
	GOOGLE_CLIENT_SECRET: string;
	SENTRY_DSN: string;
	HF_INFERENCE_ENDPOINT: string;
	GROQ_API_KEY: string;
	ANTHROPIC_API_KEY: string;
}

export interface Variables {
	db: Db;
	studentId: string | null;
}

export interface ServiceContext {
	db: Db;
	env: Bindings;
}
```

- [ ] **Step 2: Create domain error classes**

Write `apps/api/src/lib/errors.ts`:

```typescript
export class DomainError extends Error {
	constructor(message: string) {
		super(message);
		this.name = this.constructor.name;
	}
}

export class NotFoundError extends DomainError {
	constructor(entity: string, id: string) {
		super(`${entity} not found: ${id}`);
	}
}

export class AuthenticationError extends DomainError {
	constructor(message = "Authentication required") {
		super(message);
	}
}

export class ValidationError extends DomainError {
	constructor(message: string) {
		super(message);
	}
}

export class InferenceError extends DomainError {
	constructor(message: string) {
		super(message);
	}
}
```

- [ ] **Step 3: Commit**

```bash
git add apps/api/src/lib/
git commit -m "feat: add shared types (Bindings, ServiceContext) and domain errors"
```

---

### Task 3: Drizzle Schema -- Students and Auth

**Files:**
- Create: `apps/api/src/db/schema/students.ts`

- [ ] **Step 1: Write the students and auth_identities schema**

Write `apps/api/src/db/schema/students.ts`:

```typescript
import {
	pgTable,
	uuid,
	text,
	real,
	integer,
	timestamp,
	primaryKey,
	index,
	uniqueIndex,
} from "drizzle-orm/pg-core";

export const students = pgTable("students", {
	studentId: uuid("student_id").defaultRandom().primaryKey(),
	email: text("email"),
	displayName: text("display_name"),
	inferredLevel: text("inferred_level"),
	baselineDynamics: real("baseline_dynamics"),
	baselineTiming: real("baseline_timing"),
	baselinePedaling: real("baseline_pedaling"),
	baselineArticulation: real("baseline_articulation"),
	baselinePhrasing: real("baseline_phrasing"),
	baselineInterpretation: real("baseline_interpretation"),
	baselineSessionCount: integer("baseline_session_count").notNull().default(0),
	explicitGoals: text("explicit_goals"),
	createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
	updatedAt: timestamp("updated_at", { withTimezone: true }).notNull().defaultNow(),
});

export const authIdentities = pgTable(
	"auth_identities",
	{
		provider: text("provider").notNull(),
		providerUserId: text("provider_user_id").notNull(),
		studentId: uuid("student_id")
			.notNull()
			.references(() => students.studentId, { onDelete: "cascade" }),
		createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
	},
	(table) => [
		primaryKey({ columns: [table.provider, table.providerUserId] }),
		uniqueIndex("idx_auth_identities_provider_user").on(table.provider, table.providerUserId),
		index("idx_auth_identities_student").on(table.studentId),
	],
);
```

- [ ] **Step 2: Commit**

```bash
git add apps/api/src/db/schema/students.ts
git commit -m "feat: add Drizzle schema for students and auth_identities"
```

---

### Task 4: Drizzle Schema -- Sessions, Observations, Conversations

**Files:**
- Create: `apps/api/src/db/schema/sessions.ts`
- Create: `apps/api/src/db/schema/observations.ts`
- Create: `apps/api/src/db/schema/conversations.ts`

- [ ] **Step 1: Write sessions schema**

Write `apps/api/src/db/schema/sessions.ts`:

```typescript
import {
	pgTable,
	uuid,
	text,
	real,
	integer,
	timestamp,
	jsonb,
	index,
	boolean,
} from "drizzle-orm/pg-core";
import { students } from "./students";

export const sessions = pgTable(
	"sessions",
	{
		id: uuid("id").defaultRandom().primaryKey(),
		studentId: uuid("student_id")
			.notNull()
			.references(() => students.studentId, { onDelete: "cascade" }),
		startedAt: timestamp("started_at", { withTimezone: true }).notNull().defaultNow(),
		endedAt: timestamp("ended_at", { withTimezone: true }),
		avgDynamics: real("avg_dynamics"),
		avgTiming: real("avg_timing"),
		avgPedaling: real("avg_pedaling"),
		avgArticulation: real("avg_articulation"),
		avgPhrasing: real("avg_phrasing"),
		avgInterpretation: real("avg_interpretation"),
		observationsJson: jsonb("observations_json"),
		chunksSummaryJson: jsonb("chunks_summary_json"),
		conversationId: text("conversation_id"),
		accumulatorJson: jsonb("accumulator_json"),
		needsSynthesis: boolean("needs_synthesis").notNull().default(false),
	},
	(table) => [
		index("idx_sessions_student").on(table.studentId, table.startedAt),
		index("idx_sessions_conversation").on(table.conversationId),
	],
);

export const studentCheckIns = pgTable(
	"student_check_ins",
	{
		id: uuid("id").defaultRandom().primaryKey(),
		studentId: uuid("student_id")
			.notNull()
			.references(() => students.studentId, { onDelete: "cascade" }),
		sessionId: uuid("session_id").references(() => sessions.id, { onDelete: "set null" }),
		question: text("question").notNull(),
		answer: text("answer"),
		createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
	},
	(table) => [index("idx_checkins_student").on(table.studentId)],
);
```

- [ ] **Step 2: Write observations schema**

Write `apps/api/src/db/schema/observations.ts`:

```typescript
import {
	pgTable,
	uuid,
	text,
	real,
	integer,
	timestamp,
	boolean,
	index,
} from "drizzle-orm/pg-core";

export const observations = pgTable(
	"observations",
	{
		id: uuid("id").defaultRandom().primaryKey(),
		studentId: uuid("student_id").notNull(),
		sessionId: uuid("session_id").notNull(),
		chunkIndex: integer("chunk_index"),
		dimension: text("dimension").notNull(),
		observationText: text("observation_text").notNull(),
		elaborationText: text("elaboration_text"),
		reasoningTrace: text("reasoning_trace"),
		framing: text("framing"),
		dimensionScore: real("dimension_score"),
		studentBaseline: real("student_baseline"),
		pieceContext: text("piece_context"),
		learningArc: text("learning_arc"),
		isFallback: boolean("is_fallback").notNull().default(false),
		createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
		messageId: text("message_id"),
		conversationId: text("conversation_id"),
	},
	(table) => [
		index("idx_observations_student").on(table.studentId, table.createdAt),
		index("idx_observations_session").on(table.sessionId),
	],
);

export const teachingApproaches = pgTable(
	"teaching_approaches",
	{
		id: uuid("id").defaultRandom().primaryKey(),
		studentId: uuid("student_id").notNull(),
		observationId: uuid("observation_id").notNull(),
		dimension: text("dimension").notNull(),
		framing: text("framing").notNull(),
		approachSummary: text("approach_summary").notNull(),
		engaged: boolean("engaged").notNull().default(false),
		createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
	},
	(table) => [
		index("idx_teaching_approaches_student").on(table.studentId),
		index("idx_teaching_approaches_observation").on(table.observationId),
	],
);
```

- [ ] **Step 3: Write conversations schema**

Write `apps/api/src/db/schema/conversations.ts`:

```typescript
import {
	pgTable,
	uuid,
	text,
	timestamp,
	jsonb,
	index,
} from "drizzle-orm/pg-core";

export const conversations = pgTable(
	"conversations",
	{
		id: uuid("id").defaultRandom().primaryKey(),
		studentId: uuid("student_id").notNull(),
		title: text("title"),
		createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
		updatedAt: timestamp("updated_at", { withTimezone: true }).notNull().defaultNow(),
	},
	(table) => [
		index("idx_conversations_student").on(table.studentId, table.updatedAt),
	],
);

export const messages = pgTable(
	"messages",
	{
		id: uuid("id").defaultRandom().primaryKey(),
		conversationId: uuid("conversation_id")
			.notNull()
			.references(() => conversations.id, { onDelete: "cascade" }),
		role: text("role").notNull(),
		content: text("content").notNull(),
		createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
		messageType: text("message_type").notNull().default("chat"),
		dimension: text("dimension"),
		framing: text("framing"),
		componentsJson: jsonb("components_json"),
		sessionId: uuid("session_id"),
		observationId: uuid("observation_id"),
	},
	(table) => [
		index("idx_messages_conversation").on(table.conversationId, table.createdAt),
	],
);
```

- [ ] **Step 4: Commit**

```bash
git add apps/api/src/db/schema/sessions.ts apps/api/src/db/schema/observations.ts apps/api/src/db/schema/conversations.ts
git commit -m "feat: add Drizzle schema for sessions, observations, conversations"
```

---

### Task 5: Drizzle Schema -- Memory, Exercises, Catalog + Index

**Files:**
- Create: `apps/api/src/db/schema/memory.ts`
- Create: `apps/api/src/db/schema/exercises.ts`
- Create: `apps/api/src/db/schema/catalog.ts`
- Create: `apps/api/src/db/schema/index.ts`

- [ ] **Step 1: Write memory schema**

Write `apps/api/src/db/schema/memory.ts`:

```typescript
import {
	pgTable,
	uuid,
	text,
	integer,
	timestamp,
	jsonb,
	index,
} from "drizzle-orm/pg-core";

export const synthesizedFacts = pgTable(
	"synthesized_facts",
	{
		id: uuid("id").defaultRandom().primaryKey(),
		studentId: uuid("student_id").notNull(),
		factText: text("fact_text").notNull(),
		factType: text("fact_type").notNull(),
		dimension: text("dimension"),
		pieceContext: text("piece_context"),
		validAt: timestamp("valid_at", { withTimezone: true }).notNull(),
		invalidAt: timestamp("invalid_at", { withTimezone: true }),
		trend: text("trend"),
		confidence: text("confidence").notNull(),
		evidence: text("evidence").notNull(),
		sourceType: text("source_type").notNull(),
		createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
		expiredAt: timestamp("expired_at", { withTimezone: true }),
		entities: jsonb("entities"),
	},
	(table) => [
		index("idx_synthesized_facts_student").on(table.studentId),
		index("idx_synthesized_facts_active").on(table.studentId, table.invalidAt, table.expiredAt),
		index("idx_sf_student_dimension").on(table.studentId, table.dimension),
		index("idx_sf_student_source").on(table.studentId, table.sourceType),
	],
);

export const studentMemoryMeta = pgTable("student_memory_meta", {
	studentId: uuid("student_id").primaryKey(),
	lastSynthesisAt: timestamp("last_synthesis_at", { withTimezone: true }),
	totalObservations: integer("total_observations").notNull().default(0),
	totalFacts: integer("total_facts").notNull().default(0),
});
```

- [ ] **Step 2: Write exercises schema**

Write `apps/api/src/db/schema/exercises.ts`:

```typescript
import {
	pgTable,
	uuid,
	text,
	integer,
	timestamp,
	boolean,
	jsonb,
	primaryKey,
	uniqueIndex,
	index,
} from "drizzle-orm/pg-core";

export const exercises = pgTable(
	"exercises",
	{
		id: uuid("id").defaultRandom().primaryKey(),
		title: text("title").notNull(),
		description: text("description").notNull(),
		instructions: text("instructions").notNull(),
		difficulty: text("difficulty").notNull(),
		category: text("category").notNull(),
		repertoireTags: jsonb("repertoire_tags"),
		notationContent: text("notation_content"),
		notationFormat: text("notation_format"),
		midiContent: text("midi_content"),
		source: text("source").notNull(),
		variantsJson: jsonb("variants_json"),
		createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
	},
	(table) => [index("idx_exercises_difficulty").on(table.difficulty)],
);

export const exerciseDimensions = pgTable(
	"exercise_dimensions",
	{
		exerciseId: uuid("exercise_id")
			.notNull()
			.references(() => exercises.id, { onDelete: "cascade" }),
		dimension: text("dimension").notNull(),
	},
	(table) => [
		primaryKey({ columns: [table.exerciseId, table.dimension] }),
		index("idx_exercise_dimensions_dim").on(table.dimension),
	],
);

export const studentExercises = pgTable(
	"student_exercises",
	{
		id: uuid("id").defaultRandom().primaryKey(),
		studentId: uuid("student_id").notNull(),
		exerciseId: uuid("exercise_id").notNull(),
		sessionId: uuid("session_id"),
		assignedAt: timestamp("assigned_at", { withTimezone: true }).notNull().defaultNow(),
		completed: boolean("completed").notNull().default(false),
		response: text("response"),
		dimensionBeforeJson: jsonb("dimension_before_json"),
		dimensionAfterJson: jsonb("dimension_after_json"),
		notes: text("notes"),
		timesAssigned: integer("times_assigned").notNull().default(1),
	},
	(table) => [
		uniqueIndex("idx_student_exercises_unique").on(
			table.studentId,
			table.exerciseId,
			table.sessionId,
		),
		index("idx_student_exercises").on(table.studentId, table.exerciseId),
	],
);
```

- [ ] **Step 3: Write catalog schema (pieces, piece_requests, waitlist)**

Write `apps/api/src/db/schema/catalog.ts`:

```typescript
import {
	pgTable,
	uuid,
	text,
	integer,
	real,
	boolean,
	timestamp,
	index,
} from "drizzle-orm/pg-core";

export const pieces = pgTable(
	"pieces",
	{
		pieceId: text("piece_id").primaryKey(),
		composer: text("composer").notNull(),
		title: text("title").notNull(),
		keySignature: text("key_signature"),
		timeSignature: text("time_signature"),
		tempoBpm: integer("tempo_bpm"),
		barCount: integer("bar_count").notNull(),
		durationSeconds: real("duration_seconds"),
		noteCount: integer("note_count").notNull(),
		pitchRangeLow: integer("pitch_range_low"),
		pitchRangeHigh: integer("pitch_range_high"),
		hasTimeSigChanges: boolean("has_time_sig_changes").notNull().default(false),
		hasTempoChanges: boolean("has_tempo_changes").notNull().default(false),
		source: text("source").notNull().default("asap"),
		createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
	},
	(table) => [index("idx_pieces_composer").on(table.composer)],
);

export const pieceRequests = pgTable(
	"piece_requests",
	{
		id: uuid("id").defaultRandom().primaryKey(),
		query: text("query").notNull(),
		studentId: uuid("student_id").notNull(),
		matchedPieceId: text("matched_piece_id"),
		matchConfidence: real("match_confidence"),
		matchMethod: text("match_method"),
		createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
	},
	(table) => [
		index("idx_piece_requests_unmatched").on(table.matchedPieceId),
	],
);

export const waitlist = pgTable(
	"waitlist",
	{
		email: text("email").primaryKey(),
		context: text("context"),
		source: text("source").notNull().default("web"),
		createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
	},
	(table) => [index("idx_waitlist_created").on(table.createdAt)],
);
```

- [ ] **Step 4: Write the index re-export**

Write `apps/api/src/db/schema/index.ts`:

```typescript
export * from "./students";
export * from "./sessions";
export * from "./observations";
export * from "./conversations";
export * from "./exercises";
export * from "./memory";
export * from "./catalog";
```

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/db/schema/
git commit -m "feat: add Drizzle schema for memory, exercises, catalog, and index"
```

---

### Task 6: Database Client Factory

**Files:**
- Create: `apps/api/src/db/client.ts`
- Create: `apps/api/src/middleware/db.ts`

- [ ] **Step 1: Write the Drizzle client factory**

Write `apps/api/src/db/client.ts`:

```typescript
import { drizzle } from "drizzle-orm/postgres-js";
import postgres from "postgres";
import * as schema from "./schema/index";

export function createDb(hyperdrive: Hyperdrive) {
	const sql = postgres(hyperdrive.connectionString);
	return drizzle(sql, { schema });
}
```

Note: Hyperdrive transparently handles prepared statement multiplexing across pooled connections (since June 2024). Do NOT set `prepare: false` -- it kills query plan caching and adds extra round-trips.

- [ ] **Step 2: Write the database middleware**

Write `apps/api/src/middleware/db.ts`:

```typescript
import { createMiddleware } from "hono/factory";
import type { Bindings, Variables } from "../lib/types";
import { createDb } from "../db/client";

export const dbMiddleware = createMiddleware<{
	Bindings: Bindings;
	Variables: Variables;
}>(async (c, next) => {
	const db = createDb(c.env.HYPERDRIVE);
	c.set("db", db);
	await next();
});
```

- [ ] **Step 3: Commit**

```bash
git add apps/api/src/db/client.ts apps/api/src/middleware/db.ts
git commit -m "feat: add Drizzle client factory and DB middleware"
```

---

### Task 7: Middleware Chain (Error Handler, Logger, Sentry)

**Files:**
- Create: `apps/api/src/middleware/error-handler.ts`
- Create: `apps/api/src/middleware/logger.ts`
- Create: `apps/api/src/middleware/sentry.ts`

- [ ] **Step 1: Write the global error handler**

Write `apps/api/src/middleware/error-handler.ts`:

```typescript
import type { ErrorHandler } from "hono";
import { HTTPException } from "hono/http-exception";
import {
	NotFoundError,
	AuthenticationError,
	ValidationError,
} from "../lib/errors";

export const errorHandler: ErrorHandler = (err, c) => {
	if (err instanceof HTTPException) {
		return c.json({ error: err.message }, err.status);
	}

	if (err instanceof NotFoundError) {
		return c.json({ error: err.message }, 404);
	}

	if (err instanceof AuthenticationError) {
		return c.json({ error: err.message }, 401);
	}

	if (err instanceof ValidationError) {
		return c.json({ error: err.message }, 400);
	}

	console.error(JSON.stringify({
		level: "error",
		message: err.message,
		stack: err.stack,
		name: err.name,
	}));

	return c.json({ error: "Internal server error" }, 500);
};
```

- [ ] **Step 2: Write the structured logger middleware**

Write `apps/api/src/middleware/logger.ts`:

```typescript
import { createMiddleware } from "hono/factory";
import type { Bindings, Variables } from "../lib/types";

export const structuredLogger = createMiddleware<{
	Bindings: Bindings;
	Variables: Variables;
}>(async (c, next) => {
	const start = Date.now();
	await next();
	const duration = Date.now() - start;

	console.log(JSON.stringify({
		level: "info",
		method: c.req.method,
		path: c.req.path,
		status: c.res.status,
		duration_ms: duration,
	}));
});
```

- [ ] **Step 3: Write the Sentry middleware**

Write `apps/api/src/middleware/sentry.ts`:

```typescript
import { createMiddleware } from "hono/factory";
import * as Sentry from "@sentry/cloudflare";
import type { Bindings, Variables } from "../lib/types";

export const sentryMiddleware = createMiddleware<{
	Bindings: Bindings;
	Variables: Variables;
}>(async (c, next) => {
	Sentry.setContext("request", {
		method: c.req.method,
		url: c.req.url,
		path: c.req.path,
	});
	try {
		await next();
	} catch (err) {
		Sentry.captureException(err);
		throw err;
	}
});
```

- [ ] **Step 4: Commit**

```bash
git add apps/api/src/middleware/error-handler.ts apps/api/src/middleware/logger.ts apps/api/src/middleware/sentry.ts
git commit -m "feat: add error handler, structured logger, and Sentry middleware"
```

---

### Task 8: better-auth Configuration

**Files:**
- Create: `apps/api/src/lib/auth.ts`
- Create: `apps/api/src/routes/auth.ts`

- [ ] **Step 1: Write the better-auth instance factory**

Write `apps/api/src/lib/auth.ts`:

```typescript
import { betterAuth } from "better-auth";
import { drizzleAdapter } from "better-auth/adapters/drizzle";
import type { Db } from "./types";
import type { Bindings } from "./types";

export function createAuth(db: Db, env: Bindings) {
	return betterAuth({
		database: drizzleAdapter(db, {
			provider: "pg",
		}),
		baseURL: env.BETTER_AUTH_URL,
		secret: env.AUTH_SECRET,
		socialProviders: {
			apple: {
				clientId: env.APPLE_WEB_SERVICES_ID,
				clientSecret: env.APPLE_CLIENT_SECRET,
			},
			google: {
				clientId: env.GOOGLE_CLIENT_ID,
				clientSecret: env.GOOGLE_CLIENT_SECRET,
			},
		},
		session: {
			expiresIn: 60 * 60 * 24 * 30,
			cookieCache: {
				enabled: false,
			},
		},
		advanced: {
			crossSubDomainCookies: {
				enabled: true,
				domain: ".crescend.ai",
			},
		},
	});
}

export type Auth = ReturnType<typeof createAuth>;
```

- [ ] **Step 2: Write the auth route handler**

Write `apps/api/src/routes/auth.ts`:

```typescript
import { Hono } from "hono";
import type { Bindings, Variables } from "../lib/types";
import { createAuth } from "../lib/auth";
import { createDb } from "../db/client";

const auth = new Hono<{ Bindings: Bindings; Variables: Variables }>();

auth.all("/*", async (c) => {
	const db = createDb(c.env.HYPERDRIVE);
	const authInstance = createAuth(db, c.env);
	return authInstance.handler(c.req.raw);
});

export { auth as authRoutes };
```

Note: better-auth handles all `/api/auth/*` routes internally (sign-in, callback, session, sign-out). We create auth + db per request as recommended for CF Workers.

- [ ] **Step 3: Commit**

```bash
git add apps/api/src/lib/auth.ts apps/api/src/routes/auth.ts
git commit -m "feat: add better-auth with Apple + Google social providers"
```

---

### Task 9: Health Check Route

**Files:**
- Create: `apps/api/src/routes/health.ts`

- [ ] **Step 1: Write the health check route**

Write `apps/api/src/routes/health.ts`:

```typescript
import { Hono } from "hono";
import type { Bindings, Variables } from "../lib/types";

const health = new Hono<{ Bindings: Bindings; Variables: Variables }>();

health.get("/", async (c) => {
	return c.json({
		status: "ok",
		version: "2.0.0",
		stack: "hono",
	});
});

export { health as healthRoutes };
```

- [ ] **Step 2: Commit**

```bash
git add apps/api/src/routes/health.ts
git commit -m "feat: add health check endpoint"
```

---

### Task 10: App Composition and Entry Point

**Files:**
- Create: `apps/api/src/index.ts`

- [ ] **Step 1: Write the app entry point**

Write `apps/api/src/index.ts`:

```typescript
import { Hono } from "hono";
import { cors } from "hono/cors";
import * as Sentry from "@sentry/cloudflare";
import type { Bindings, Variables } from "./lib/types";
import { dbMiddleware } from "./middleware/db";
import { structuredLogger } from "./middleware/logger";
import { sentryMiddleware } from "./middleware/sentry";
import { errorHandler } from "./middleware/error-handler";
import { authRoutes } from "./routes/auth";
import { healthRoutes } from "./routes/health";

const app = new Hono<{ Bindings: Bindings; Variables: Variables }>();

app.onError(errorHandler);

app.use("*", async (c, next) => {
	const origin = c.env.ALLOWED_ORIGIN || "http://localhost:3000";
	return cors({
		origin,
		allowMethods: ["GET", "POST", "OPTIONS", "DELETE"],
		allowHeaders: ["Content-Type", "Authorization", "Cookie"],
		credentials: true,
	})(c, next);
});

app.use("*", structuredLogger);
app.use("*", sentryMiddleware);
app.use("/api/*", dbMiddleware);

app.route("/health", healthRoutes);
app.route("/api/auth", authRoutes);

app.notFound((c) => c.json({ error: "Not found" }, 404));

export type AppType = typeof app;

export default Sentry.withSentry(
	(env) => ({
		dsn: env.SENTRY_DSN,
		tracesSampleRate: 1.0,
	}),
	app,
);
```

- [ ] **Step 2: Verify the worker starts locally**

```bash
cd apps/api
npx wrangler dev --local
```

Expected: Worker starts on localhost:8787. Hit `http://localhost:8787/health` and get `{"status":"ok","version":"2.0.0","stack":"hono"}`.

Note: Auth routes will fail without Hyperdrive/PlanetScale configured, but the health check should work.

- [ ] **Step 3: Commit**

```bash
git add apps/api/src/index.ts
git commit -m "feat: compose Hono app with middleware chain and route mounting"
```

---

### Task 11: Vitest Test Setup

**Files:**
- Create: `apps/api/vitest.config.ts`

- [ ] **Step 1: Write vitest config for CF Workers pool**

Write `apps/api/vitest.config.ts`:

```typescript
import { defineWorkersConfig } from "@cloudflare/vitest-pool-workers/config";

export default defineWorkersConfig({
	test: {
		poolOptions: {
			workers: {
				wrangler: { configPath: "./wrangler.toml" },
			},
		},
	},
});
```

- [ ] **Step 2: Add test script to package.json**

Edit `apps/api/package.json` to add scripts:

```json
{
  "scripts": {
    "dev": "wrangler dev",
    "test": "vitest",
    "typecheck": "tsc --noEmit",
    "generate": "drizzle-kit generate",
    "migrate": "drizzle-kit migrate"
  }
}
```

- [ ] **Step 3: Commit**

```bash
git add apps/api/vitest.config.ts apps/api/package.json
git commit -m "feat: add Vitest config with CF Workers pool"
```

---

### Task 12: Integration Test -- Health Check

**Files:**
- Create: `apps/api/src/routes/health.test.ts`

- [ ] **Step 1: Write the failing test**

Write `apps/api/src/routes/health.test.ts`:

```typescript
import { describe, it, expect } from "vitest";
import app from "../index";

describe("GET /health", () => {
	it("returns ok status", async () => {
		const res = await app.request("/health");
		expect(res.status).toBe(200);

		const body = await res.json();
		expect(body).toEqual({
			status: "ok",
			version: "2.0.0",
			stack: "hono",
		});
	});
});
```

- [ ] **Step 2: Run the test**

```bash
cd apps/api
bun run test -- --run
```

Expected: PASS. The health check should work without Hyperdrive since it doesn't use the DB middleware.

- [ ] **Step 3: Commit**

```bash
git add apps/api/src/routes/health.test.ts
git commit -m "test: add health check integration test"
```

---

### Task 13: Generate Initial Drizzle Migration

**Files:**
- Generated: `apps/api/src/db/migrations/*.sql`

- [ ] **Step 1: Generate the migration**

```bash
cd apps/api
npx drizzle-kit generate
```

Expected: Creates a SQL migration file in `src/db/migrations/` with all table definitions matching the schema.

- [ ] **Step 2: Review the generated SQL**

Read the generated migration file and verify:
- All 16 tables are present (students, auth_identities, sessions, student_check_ins, observations, teaching_approaches, conversations, messages, synthesized_facts, student_memory_meta, exercises, exercise_dimensions, student_exercises, pieces, piece_requests, waitlist)
- All indexes match the schema definitions
- Foreign key constraints are correct
- UUID columns use `uuid` type with `gen_random_uuid()` defaults

- [ ] **Step 3: Commit**

```bash
git add apps/api/src/db/migrations/
git commit -m "feat: generate initial Drizzle migration for PlanetScale Postgres"
```

---

### Task 14: Update Justfile with New API Commands

**Files:**
- Modify: `Justfile`

- [ ] **Step 1: Add new Hono API commands to Justfile**

Add these recipes to the Justfile:

```makefile
# New Hono API
api:
    cd apps/api && npx wrangler dev --local

test-api-new:
    cd apps/api && bun run test -- --run

check-api-new:
    cd apps/api && bun run typecheck

# Dev with new API (light mode)
dev-new:
    just api & just web
```

- [ ] **Step 2: Verify commands work**

```bash
just check-api-new
```

Expected: TypeScript compilation succeeds with no errors.

- [ ] **Step 3: Commit**

```bash
git add Justfile
git commit -m "feat: add Justfile commands for new Hono API"
```

---

### Task 15: PlanetScale + Hyperdrive Setup

This task requires manual setup via the PlanetScale dashboard and Cloudflare dashboard.

- [ ] **Step 1: Create PlanetScale database**

1. Go to PlanetScale dashboard
2. Create a new Postgres database named `crescendai`
3. Choose the Scaler plan ($5/mo)
4. Select a region close to your primary user base (us-east-1 recommended for HF endpoint proximity)
5. Copy the connection string

- [ ] **Step 2: Create Hyperdrive config**

```bash
cd apps/api
npx wrangler hyperdrive create crescendai-pg --connection-string="<planetscale_connection_string>"
```

Copy the returned Hyperdrive ID.

- [ ] **Step 3: Update wrangler.toml with Hyperdrive ID**

Replace `PLACEHOLDER_HYPERDRIVE_ID` in `apps/api/wrangler.toml` with the actual Hyperdrive ID from Step 2.

- [ ] **Step 4: Apply the Drizzle migration to PlanetScale**

```bash
cd apps/api
DATABASE_URL="<planetscale_connection_string>" npx drizzle-kit migrate
```

Expected: All 16 tables created in PlanetScale Postgres.

- [ ] **Step 5: Set secrets**

```bash
cd apps/api
npx wrangler secret put AUTH_SECRET
npx wrangler secret put APPLE_CLIENT_SECRET
npx wrangler secret put GOOGLE_CLIENT_SECRET
npx wrangler secret put SENTRY_DSN
```

- [ ] **Step 6: Commit the Hyperdrive ID update**

```bash
git add apps/api/wrangler.toml
git commit -m "feat: configure Hyperdrive for PlanetScale Postgres"
```

---

### Task 16: End-to-End Validation

- [ ] **Step 1: Start the new API locally**

```bash
cd apps/api
npx wrangler dev
```

- [ ] **Step 2: Verify health check**

```bash
curl http://localhost:8787/health
```

Expected: `{"status":"ok","version":"2.0.0","stack":"hono"}`

- [ ] **Step 3: Verify auth routes are mounted**

```bash
curl -X GET http://localhost:8787/api/auth/get-session
```

Expected: A JSON response from better-auth (likely `{"session":null}` since no session exists). The point is it doesn't 404.

- [ ] **Step 4: Verify DB connection via Hyperdrive**

The auth route hitting the session check validates the full chain: Hono -> better-auth -> Drizzle -> Hyperdrive -> PlanetScale Postgres. If Step 3 returns a response (even null session), the DB connection works.

- [ ] **Step 5: Run all tests**

```bash
cd apps/api
bun run test -- --run
```

Expected: All tests pass.

- [ ] **Step 6: Final commit if any adjustments were needed**

```bash
git add -A apps/api/
git commit -m "feat: Phase 1 foundation complete -- Hono + Drizzle + better-auth + PlanetScale"
```

---

## Phase 1 Validation Gate

Phase 1 is complete when:
- [x] Health check returns 200 at `/health`
- [x] better-auth routes are mounted at `/api/auth/*`
- [x] Drizzle schema matches all 16 D1 tables with Postgres improvements
- [x] Hyperdrive connects to PlanetScale Postgres
- [x] Sentry SDK is configured and middleware is active
- [x] Structured JSON logging is working
- [x] All tests pass
- [x] TypeScript compiles with no errors

## What Comes Next

**Phase 2: CRUD Endpoints + Data Migration** -- Exercises, conversations, sync, waitlist, scores/pieces, chat (SSE streaming), and the D1 -> Postgres data migration script. Separate plan.

**Phase 3: Complex Domain** -- Practice DOs (state machine, WebSocket hibernation), WASM module extraction, teaching pipeline, synthesis Workflow. Separate plan.

**Phase 4: Cutover** -- Web app switches to `hc<AppType>()`, DNS switch, D1 decommission. Separate plan.
