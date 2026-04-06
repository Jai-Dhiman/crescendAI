# Catalog Search Tool Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** The teacher LLM can resolve piece names/composers to piece_id UUIDs via a `search_catalog` tool, enabling it to chain into `score_highlight` and `reference_browser` without asking the student for a UUID.
**Spec:** docs/specs/2026-04-06-catalog-search-tool-design.md
**Style:** Follow the project's coding standards (CLAUDE.md / TS_STYLE.md)

**Note:** Part B (piece context injection into chat dynamic context) is deferred -- `pieceIdentification` lives in DO state only and isn't persisted to a queryable DB column. Adding a `piece_id` column to the `sessions` table is a separate migration task. The `search_catalog` tool alone fully solves the UX problem.

---

## Task Groups

- Group A (parallel): Task 1, Task 2
- Group B (sequential, depends on A): Task 3

---

## File Structure

| File | Responsibility | Interface | Depth | New / Modify |
|------|----------------|-----------|-------|-------------|
| `apps/api/src/services/tool-processor.ts` | Add search_catalog tool (schema, process fn, Anthropic schema, registry) | `TOOL_REGISTRY.search_catalog` | DEEP (hides ILIKE query construction, result shaping, validation) | Modify |
| `apps/api/src/services/tool-processor.test.ts` | Schema validation + integration tests | Zod safeParse + processToolUse | -- | Modify |
| `apps/api/src/services/prompts.ts` | Update tool usage docs in UNIFIED_TEACHER_SYSTEM | String constant | -- | Modify |

---

### Task 1: Add search_catalog Zod schema and validation tests
**Group:** A (parallel with Task 2)

**Behavior being verified:** The `search_catalog` tool validates input: accepts `query`, `composer`, and/or `title` (at least one required), rejects empty input, rejects when all fields are empty strings.
**Interface under test:** `TOOL_REGISTRY.search_catalog.schema` (Zod schema)

**Files:**
- Modify: `apps/api/src/services/tool-processor.ts`
- Modify: `apps/api/src/services/tool-processor.test.ts`

- [ ] **Step 1: Write the failing test**

In `apps/api/src/services/tool-processor.test.ts`, add after the `reference_browser schema validation` describe block (after line ~402):

```typescript
// ---------------------------------------------------------------------------
// search_catalog Zod validation
// ---------------------------------------------------------------------------

describe("search_catalog schema validation", () => {
	const schema = TOOL_REGISTRY.search_catalog.schema;

	it("passes with query only", () => {
		const result = schema.safeParse({ query: "Chopin waltz" });
		expect(result.success).toBe(true);
	});

	it("passes with composer only", () => {
		const result = schema.safeParse({ composer: "Chopin" });
		expect(result.success).toBe(true);
	});

	it("passes with title only", () => {
		const result = schema.safeParse({ title: "Waltz Op. 64" });
		expect(result.success).toBe(true);
	});

	it("passes with all fields", () => {
		const result = schema.safeParse({
			query: "waltz",
			composer: "Chopin",
			title: "Op. 64 No. 2",
		});
		expect(result.success).toBe(true);
	});

	it("rejects empty object", () => {
		const result = schema.safeParse({});
		expect(result.success).toBe(false);
	});

	it("rejects all empty strings", () => {
		const result = schema.safeParse({ query: "", composer: "", title: "" });
		expect(result.success).toBe(false);
	});
});
```

Also update the registry structure test. Change line ~22 from:
```typescript
expect(Object.keys(TOOL_REGISTRY)).toHaveLength(5);
```
to:
```typescript
expect(Object.keys(TOOL_REGISTRY)).toHaveLength(6);
```

And update the `toolNames` array (line ~14-20) to include `"search_catalog"`:
```typescript
const toolNames = [
	"create_exercise",
	"score_highlight",
	"keyboard_guide",
	"show_session_data",
	"reference_browser",
	"search_catalog",
] as const;
```

Also add to the concurrencySafe test (line ~45-49):
```typescript
expect(TOOL_REGISTRY.search_catalog.concurrencySafe).toBe(true);
```

And add to the maxResultChars test (line ~52-56):
```typescript
expect(TOOL_REGISTRY.search_catalog.maxResultChars).toBe(3000);
```

And update the getAnthropicToolSchemas test (line ~70):
```typescript
expect(schemas).toHaveLength(6);
```

- [ ] **Step 2: Run test -- verify it FAILS**

```bash
cd apps/api && npx vitest run src/services/tool-processor.test.ts
```
Expected: FAIL -- `TOOL_REGISTRY.search_catalog` is undefined. Registry has 5 tools, not 6.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `apps/api/src/services/tool-processor.ts`, add after the `referenceBrowserSchema` definition (after line ~365) and before the Anthropic schemas section:

```typescript
// ---------------------------------------------------------------------------
// Tool: search_catalog
// ---------------------------------------------------------------------------

const searchCatalogSchema = z
	.object({
		query: z.string().min(1).max(200).optional(),
		composer: z.string().min(1).max(200).optional(),
		title: z.string().min(1).max(200).optional(),
	})
	.refine(
		(data) =>
			(data.query !== undefined && data.query.length > 0) ||
			(data.composer !== undefined && data.composer.length > 0) ||
			(data.title !== undefined && data.title.length > 0),
		{ message: "At least one of query, composer, or title is required" },
	);

async function processSearchCatalog(
	ctx: ServiceContext,
	_studentId: string,
	rawInput: unknown,
): Promise<InlineComponent[]> {
	const input = searchCatalogSchema.parse(rawInput);

	const conditions = [];

	if (input.composer) {
		conditions.push(sql`${pieces.composer} ILIKE ${"%" + input.composer + "%"}`);
	}

	if (input.title) {
		conditions.push(sql`${pieces.title} ILIKE ${"%" + input.title + "%"}`);
	}

	if (input.query) {
		conditions.push(
			sql`(${pieces.composer} ILIKE ${"%" + input.query + "%"} OR ${pieces.title} ILIKE ${"%" + input.query + "%"})`,
		);
	}

	const whereClause = conditions.length > 1 ? and(...conditions) : conditions[0];

	const rows = await ctx.db
		.select({
			pieceId: pieces.pieceId,
			composer: pieces.composer,
			title: pieces.title,
			barCount: pieces.barCount,
		})
		.from(pieces)
		.where(whereClause)
		.orderBy(asc(pieces.composer), asc(pieces.title))
		.limit(5);

	if (rows.length === 0) {
		return [
			{
				type: "search_catalog_result",
				config: {
					matches: [],
					message: "No pieces found matching the search criteria.",
				},
			},
		];
	}

	return [
		{
			type: "search_catalog_result",
			config: {
				matches: rows.map((r) => ({
					pieceId: r.pieceId,
					composer: r.composer,
					title: r.title,
					barCount: r.barCount,
				})),
			},
		},
	];
}
```

Also add the import for `sql` at line 1 -- update from:
```typescript
import { and, desc, eq } from "drizzle-orm";
```
to:
```typescript
import { and, asc, desc, eq, sql } from "drizzle-orm";
```

Note: `asc` is already used in scores.ts but not yet imported in tool-processor.ts. Add it.

Add the Anthropic schema after `referenceBrowserAnthropicSchema` (before the Registry section):

```typescript
const searchCatalogAnthropicSchema: AnthropicToolSchema = {
	name: "search_catalog",
	description:
		"Search the piece catalog to find a piece's UUID by composer name, title, or free text query. Use this when the student mentions a piece by name and you need the piece_id for other tools like score_highlight. Returns up to 5 matches.",
	input_schema: {
		type: "object",
		properties: {
			query: {
				type: "string",
				description:
					"Free text search across composer and title fields. Use when the student gives a general reference like 'Chopin waltz'.",
			},
			composer: {
				type: "string",
				description: "Filter by composer name (case-insensitive substring match).",
			},
			title: {
				type: "string",
				description:
					"Filter by piece title (case-insensitive substring match). Include opus numbers if known.",
			},
		},
	},
};
```

Add the registry entry inside `TOOL_REGISTRY` (after the `reference_browser` entry):

```typescript
search_catalog: {
	name: "search_catalog",
	description: searchCatalogAnthropicSchema.description,
	schema: searchCatalogSchema,
	anthropicSchema: searchCatalogAnthropicSchema,
	concurrencySafe: true,
	maxResultChars: 3000,
	process: processSearchCatalog,
},
```

- [ ] **Step 4: Run test -- verify it PASSES**

```bash
cd apps/api && npx vitest run src/services/tool-processor.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd apps/api && git add src/services/tool-processor.ts src/services/tool-processor.test.ts && git commit -m "feat(api): add search_catalog tool for piece name to UUID resolution"
```

---

### Task 2: Update teacher system prompt with search_catalog tool guidance
**Group:** A (parallel with Task 1)

**Behavior being verified:** The `UNIFIED_TEACHER_SYSTEM` prompt includes guidance for the `search_catalog` tool so the teacher knows when and how to use it.
**Interface under test:** `UNIFIED_TEACHER_SYSTEM` string content

**Files:**
- Modify: `apps/api/src/services/prompts.ts`
- Create: `apps/api/src/services/prompts.test.ts`

- [ ] **Step 1: Write the failing test**

Create `apps/api/src/services/prompts.test.ts`:

```typescript
import { describe, expect, it } from "vitest";
import { UNIFIED_TEACHER_SYSTEM } from "./prompts";

describe("UNIFIED_TEACHER_SYSTEM", () => {
	it("includes search_catalog tool guidance", () => {
		expect(UNIFIED_TEACHER_SYSTEM).toContain("search_catalog");
	});

	it("instructs teacher to search before asking student for piece_id", () => {
		expect(UNIFIED_TEACHER_SYSTEM).toContain(
			"Never ask the student for a piece ID",
		);
	});

	it("instructs teacher to disambiguate multiple matches", () => {
		expect(UNIFIED_TEACHER_SYSTEM).toContain(
			"If multiple matches are returned",
		);
	});
});
```

- [ ] **Step 2: Run test -- verify it FAILS**

```bash
cd apps/api && npx vitest run src/services/prompts.test.ts
```
Expected: FAIL -- `UNIFIED_TEACHER_SYSTEM` does not contain `"search_catalog"` or the instruction text.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `apps/api/src/services/prompts.ts`, update the `## Tool Usage` section of `UNIFIED_TEACHER_SYSTEM` (around line 46-53). Replace:

```typescript
## Tool Usage
You have tools available. Use them when they add value:
- create_exercise: When a concrete drill would help more than verbal guidance. Use sparingly.
- score_highlight: When discussing a specific passage and visual reference would help.
- keyboard_guide: When fingering or hand position matters.
- show_session_data: When the student asks about progress or you want to reference their history.
- reference_browser: When suggesting the student listen to a specific performance.
Most responses should be text-only. Tools are supplements, not defaults.
```

with:

```typescript
## Tool Usage
You have tools available. Use them when they add value:
- search_catalog: When the student mentions a piece by name and you need its piece_id for other tools. Never ask the student for a piece ID -- use search_catalog to look it up yourself. If multiple matches are returned, present the options to the student and confirm before using a piece_id.
- create_exercise: When a concrete drill would help more than verbal guidance. Use sparingly.
- score_highlight: When discussing a specific passage and visual reference would help. Requires piece_id -- use search_catalog first if you don't have it.
- keyboard_guide: When fingering or hand position matters.
- show_session_data: When the student asks about progress or you want to reference their history.
- reference_browser: When suggesting the student listen to a specific performance.
Most responses should be text-only. Tools are supplements, not defaults.
```

- [ ] **Step 4: Run test -- verify it PASSES**

```bash
cd apps/api && npx vitest run src/services/prompts.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd apps/api && git add src/services/prompts.ts src/services/prompts.test.ts && git commit -m "feat(api): add search_catalog guidance to teacher system prompt"
```

---

### Task 3: Integration test -- search_catalog pass-through with mock DB
**Group:** B (depends on Group A)

**Behavior being verified:** `processToolUse("search_catalog", { query: "chopin" })` calls the DB and returns a `search_catalog_result` component with matches (or empty matches if none found).
**Interface under test:** `processToolUse("search_catalog", ...)` return shape

**Files:**
- Modify: `apps/api/src/services/tool-processor.test.ts`

- [ ] **Step 1: Write the failing test**

In `apps/api/src/services/tool-processor.test.ts`, add to the "processToolUse pass-through tools" describe block (after the score_highlight tests, before the closing `});`):

```typescript
it("search_catalog returns matches from DB", async () => {
	// Create a mockCtx that returns catalog results for the search_catalog ILIKE chain
	const searchMockCtx = {
		db: {
			select: () => ({
				from: () => ({
					where: () => ({
						orderBy: () => ({
							limit: () =>
								Promise.resolve([
									{
										pieceId: "abc-123",
										composer: "Chopin",
										title: "Waltz Op. 64 No. 2",
										barCount: 138,
									},
								]),
						}),
					}),
				}),
			}),
		} as never,
		env: {} as never,
	};

	const { processToolUse } = await import("./tool-processor");
	const result: ToolResult = await processToolUse(
		searchMockCtx,
		studentId,
		"search_catalog",
		{ query: "chopin waltz" },
	);
	expect(result.isError).toBe(false);
	expect(result.name).toBe("search_catalog");
	expect(result.componentsJson).toHaveLength(1);
	expect(result.componentsJson[0].type).toBe("search_catalog_result");

	const config = result.componentsJson[0].config as {
		matches: Array<{ pieceId: string; composer: string; title: string }>;
	};
	expect(config.matches).toHaveLength(1);
	expect(config.matches[0].pieceId).toBe("abc-123");
	expect(config.matches[0].composer).toBe("Chopin");
});

it("search_catalog returns empty matches when nothing found", async () => {
	// Mock DB returns empty array
	const emptyMockCtx = {
		db: {
			select: () => ({
				from: () => ({
					where: () => ({
						orderBy: () => ({
							limit: () => Promise.resolve([]),
						}),
					}),
				}),
			}),
		} as never,
		env: {} as never,
	};

	const { processToolUse } = await import("./tool-processor");
	const result: ToolResult = await processToolUse(
		emptyMockCtx,
		studentId,
		"search_catalog",
		{ composer: "Nonexistent" },
	);
	expect(result.isError).toBe(false);
	const config = result.componentsJson[0].config as {
		matches: unknown[];
		message?: string;
	};
	expect(config.matches).toHaveLength(0);
	expect(config.message).toContain("No pieces found");
});
```

Note: The `searchMockCtx` needs a different mock chain than the existing `mockCtx` because `processSearchCatalog` calls `select().from().where().orderBy().limit()` (with `orderBy` in the chain), while `processScoreHighlight` calls `select().from().where().limit()` (without `orderBy`). The existing `mockCtx` doesn't have `orderBy` in the chain, so it can't be reused here.

- [ ] **Step 2: Run test -- verify it PASSES**

```bash
cd apps/api && npx vitest run src/services/tool-processor.test.ts
```
Expected: PASS -- Task 1 already implemented the full process function. This task is a verification-only commit adding integration tests that confirm the DB query and result shaping behavior.

- [ ] **Step 3: Implement (not needed)**

No implementation required -- Task 1 already includes the full `processSearchCatalog` function. Proceed to commit.

- [ ] **Step 4: Run test -- verify it PASSES**

```bash
cd apps/api && npx vitest run src/services/tool-processor.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd apps/api && git add src/services/tool-processor.test.ts && git commit -m "test(api): add search_catalog integration tests with mock DB"
```

---

## Challenge Review

### CEO Pass

#### 1. Premise Challenge

**Right problem?** Yes. The concrete failure mode is real and demonstrable: `score_highlight` requires a UUID, students never know UUIDs, and the teacher LLM has no resolution path. This is not a hypothetical UX gap -- it is a blocker for any chat-initiated `score_highlight` call.

**Real pain?** High. Without this, the teacher can call `score_highlight` from the observation path (where `pieceId` is already in DO state), but it fails entirely in the chat path for any piece the student references by name. The tool exists but is unreachable from the primary user-facing surface.

**Direct path?** Yes. Adding a read-only catalog search tool is the minimal correct fix. The alternative (injecting piece context into every chat dynamic context) is deferred as Part B and documented -- this is the right scoping decision.

**Existing coverage?** `GET /api/scores` already exposes catalog data publicly. The plan correctly reuses the `pieces` table and the existing Drizzle query pattern rather than introducing a new service. Pattern match is tight.

#### 2. Scope Check

The plan has cleanly cut Part B (piece context injection into chat framing). This is the correct call -- Part B requires a schema migration (`piece_id` column on `sessions` table) and cross-service wiring (`chat.ts` -> `prompts.ts`) that would expand scope 3x.

One scope note: the spec's `## File Changes` table lists 5 files including `chat.ts` and `prompts.test.ts` (testing `buildChatFraming`). The plan's `## File Structure` table lists only 3 files and defers `chat.ts` entirely. The plan's scope is correct for the stated goal; the spec's table bleeds into Part B scope. No action needed -- the plan is right to cut.

The hardest problem here -- query design -- is actually being solved. ILIKE substring match on a 242-piece catalog is the right call (no pg_trgm, no embedding lookup). This is not being avoided.

#### 3. Twelve-Month Alignment

```
CURRENT STATE                        THIS PLAN                       12-MONTH IDEAL
Teacher LLM must ask student     ->  Teacher resolves names to   ->  Teacher always has piece context
for UUID to use score_highlight      UUIDs autonomously via           injected (Part B), with search
                                     search_catalog tool              as fallback for cross-references
```

The plan moves cleanly toward the ideal. No tech debt created. Part B is the natural next step and is already documented.

#### 4. Alternatives Check

The spec documents the alternatives consideration: ILIKE vs full-text/trigram. Rationale is clear (242-piece catalog, LLM reformulates queries, no infra complexity). The single-tool design vs embedding in `score_highlight` is also justified (shared by `reference_browser`, disambiguate before commit, single responsibility). Alternative documentation is adequate.

---

### Engineering Pass

#### 5. Architecture

**Data flow:**
```
chat.ts -> processToolUse("search_catalog", { query })
    -> TOOL_REGISTRY["search_catalog"].schema.safeParse()
    -> processSearchCatalog(ctx, studentId, input)
    -> ctx.db.select().from(pieces).where(ILIKE).orderBy().limit(5)
    -> InlineComponent[]{ type: "search_catalog_result", config: { matches } }
```

The plan follows the existing pattern exactly: schema constant -> process function -> Anthropic schema constant -> registry entry. This is correct.

**Security:** Read-only query against the `pieces` table (publicly accessible via `GET /api/scores`). Drizzle parameterizes all queries -- no SQL injection risk. The user-supplied `query`/`composer`/`title` strings are interpolated into `sql` template literals with `${"%" + input.query + "%"}` syntax, which Drizzle correctly parameterizes as prepared statement parameters. No issue.

**Import change:** The plan adds `sql` and `asc` to the drizzle-orm import. `sql` is used for `ILIKE` construction. This is the standard Drizzle pattern for raw SQL fragments in typed queries (per TS_STYLE.md section 10).

**`whereClause` edge case:** When all three fields are provided, `conditions` has 3 elements and `and(...conditions)` is used. When exactly one field is provided, `conditions[0]` is used directly. When `conditions.length === 0` (impossible after Zod validation), `conditions[0]` would be `undefined`. This is safe because Zod's `.refine()` and `.min(1)` ensure at least one non-empty field before `processSearchCatalog` is called. SAFE but not obviously so.

**No scaling concern:** 242-piece catalog, limit 5, read-only. No N+1.

#### 6. Module Depth Audit

**`search_catalog` entry in TOOL_REGISTRY (in tool-processor.ts)**
- Interface size: 1 registry key, same `ToolDefinition` interface as all other tools
- Implementation size: ~60 LOC (schema + Zod refine + ILIKE query builder + result shaping + empty-result handling)
- Verdict: DEEP -- hides query construction, ILIKE parameterization, result shaping, and zero-result messaging behind a single `process` fn

**Anthropic schema constant (`searchCatalogAnthropicSchema`)**
- Interface size: 1 exported constant
- Implementation size: static JSON (no logic)
- Verdict: DEEP for purpose -- follows identical pattern to existing 5 schemas

**Prompt update (UNIFIED_TEACHER_SYSTEM)**
- Interface size: 1 exported string constant
- Implementation size: ~60 lines of template text
- Verdict: DEEP -- hides all pedagogical framing behind a single exported constant

No shallow module concerns.

#### 7. Code Quality

**DRY:** The ILIKE construction with `sql` template strings is repeated 3 times (once per field). This is acceptable at this scale -- extracting a helper for 3 occurrences in a single function would be over-engineering. Not a DRY violation.

**Error handling:** `processSearchCatalog` calls `ctx.db.select()...` inside `processToolUse`'s `try/catch` block (lines 704-739 of tool-processor.ts). Any DB error (connection failure, query error) is caught, logged as structured JSON, and returned as `{ isError: true }`. This is correct and matches existing tool error handling.

**Edge cases:**
- Empty `query`/`composer`/`title` strings: handled by Zod `.min(1)` + `.refine()`.
- A string like `"   "` (whitespace only) would pass `.min(1)` and result in ILIKE `"% %"`, which matches everything. Minor UX concern, flagged as RISK below.
- `rows` being null/undefined: Drizzle `.limit()` always returns an array. Safe.

**TS_STYLE.md compliance:**
- No `any` used -- `rawInput: unknown` narrows via Zod parse. Compliant.
- Structured logging: errors logged by `processToolUse`'s existing catch block. Compliant.
- No `HTTPException` imported in service file. Compliant.
- ServiceContext pattern used. Compliant.

#### 8. Test Philosophy Audit

**Task 1 -- schema validation tests:** All 6 tests exercise `schema.safeParse()` directly. This is the correct interface for schema tests. Tests verify behavior (rejects empty, rejects missing, accepts optional-but-at-least-one). Not shape tests.

Registry structure tests (length=6, concurrencySafe=true, maxResultChars=3000) verify observable properties of the registry, not internal state. Acceptable.

**Task 2 -- prompt content tests:** Two tests verify `UNIFIED_TEACHER_SYSTEM` contains specific substrings. For a prompt-content test, there is no behavior to test other than "the string contains the right guidance." This is the correct test form for this artifact. However:

[RISK] (confidence: 6/10) -- Task 2's test for `"Never ask the student for a piece ID"` is brittle: if the exact phrasing changes during copy editing, the test breaks. Verify this is actually an issue before hardening -- this is acceptable for a prompt test given there is no alternative.

**Task 3 -- integration tests:** Tests call `processToolUse("search_catalog", ...)` through the public interface, asserting on the returned `ToolResult` shape. Correct behavior-through-interface testing.

The mock DB chain couples the test to the specific Drizzle call chain order (`select().from().where().orderBy().limit()`). This is a pre-existing accepted pattern in the test suite (same pattern used for `score_highlight` mock).

[RISK] (confidence: 7/10) -- Task 3's mock DB chain is structurally coupled to the exact Drizzle call order. A chain change (e.g., adding `.innerJoin()`) causes the mock to return `undefined` silently. Pre-existing pattern in the codebase; consistent with existing `mockCtx` usage.

#### 9. Vertical Slice Audit

**Task 1:** One behavior (schema validation) -> one implementation (schema + registry entry) -> one commit. CLEAN.

**Task 2:** One behavior (prompt contains tool guidance) -> one implementation (prompt string update) -> one commit. CLEAN.

**Task 3:**

[BLOCKER] (confidence: 8/10) -- Task 3's integration tests will pass immediately after Task 1 is merged because `processSearchCatalog` is fully implemented in Task 1. The plan acknowledges this at Step 2 ("If Task 1 is merged, this should PASS immediately"). However, Step 2 is labeled "Run test -- verify it FAILS", which is directly contradicted by the note. This creates an ambiguous instruction: the build agent may interpret a passing test at Step 2 as a test failure or may be confused about whether to proceed. **Required fix before execution:** Relabel Task 3's Step 2 to clearly state it is a verification-only step: "Expected: PASS (implementation already shipped in Task 1 -- this task adds integration test coverage, not new behavior)." Alternatively, move the integration test into Task 1's test step, making Task 3 unnecessary.

#### 10. Test Coverage Gaps

```
[+] apps/api/src/services/tool-processor.ts (search_catalog additions)
    │
    ├── searchCatalogSchema (Zod)
    │   ├── [TESTED ★★★] query-only, composer-only, title-only, all-fields (Task 1)
    │   ├── [TESTED ★★★] empty object rejects, all-empty-strings rejects (Task 1)
    │   └── [GAP]     whitespace-only string (e.g., { query: "   " }) -- passes .min(1)
    │
    ├── processSearchCatalog()
    │   ├── [TESTED ★★] happy path with matches (Task 3)
    │   ├── [TESTED ★★] zero matches -> empty matches + message (Task 3)
    │   ├── [GAP]     all three fields simultaneously (multi-condition AND) -- tested at schema level only
    │   └── [GAP]     DB error thrown mid-query -- covered by processToolUse generic catch, not explicitly tested
    │
    └── TOOL_REGISTRY["search_catalog"]
        ├── [TESTED ★★] registry structure (length, concurrencySafe, maxResultChars) (Task 1)
        └── [TESTED ★★] Anthropic schema count (Task 1)

[+] apps/api/src/services/prompts.ts (UNIFIED_TEACHER_SYSTEM update)
    └── [TESTED ★★]  contains "search_catalog" and exact phrase (Task 2)
```

The whitespace-only string gap is minor (242-piece catalog, non-malicious LLM context). Multi-condition AND gap is acceptable.

[RISK] (confidence: 5/10) -- Whitespace-only input strings pass `.min(1)` Zod validation and generate ILIKE `"% %"`, which matches all pieces and returns 5 random results. Low real-world impact given LLM input patterns; worth noting but not blocking.

#### 11. Failure Modes

**Task 1:** DB import failure -> `processSearchCatalog` throws -> caught by `processToolUse` catch block -> logged -> returned as `isError: true` -> teacher LLM degrades gracefully. Not silent.

**Task 2:** Prompt string update is pure data -- no failure mode beyond a TypeScript compile error (impossible for a string literal).

**Task 3:** Tests-only commit. No runtime failure mode.

**Runtime disambiguation gap:**

[RISK] (confidence: 7/10) -- The system prompt update instructs the teacher to "use search_catalog to look it up yourself" but does not specify what to do when multiple matches are returned. When `search_catalog` returns 5 pieces named "waltz", the LLM may silently pick the wrong one. **Recommended fix:** Add to the prompt update: "If search_catalog returns multiple matches, present the options to the student and confirm before using a piece_id."

#### 12. Presumption Inventory

| Assumption | Verdict | Reason |
|---|---|---|
| `pieces` table has `composer`, `title`, `barCount`, `pieceId` columns | SAFE | Verified in `catalog.ts` -- all four columns present |
| `asc` is not yet imported in `tool-processor.ts` | SAFE | Verified -- line 1: `{ and, desc, eq }`, no `asc` |
| `sql` is not yet imported in `tool-processor.ts` | SAFE | Verified -- line 1 has no `sql` import |
| Registry currently has 5 tools (tests assert `toHaveLength(5)`) | SAFE | Verified in test file line 23 and tool-processor.ts lines 604-649 |
| `processToolUse`'s catch block covers DB errors from `processSearchCatalog` | SAFE | Verified -- lines 704-739 wrap all `tool.process()` calls |
| Mock DB chain `select().from().where().orderBy().limit()` matches actual Drizzle call order | SAFE | Plan's implementation code uses exactly this chain order |
| ILIKE is supported by PlanetScale Postgres (via Hyperdrive) | SAFE | Standard PostgreSQL operator, no extension required |
| `pieces.pieceId` is a `text` primary key (not `uuid()` type) | SAFE | Verified in `catalog.ts` -- `text("piece_id").primaryKey()` |
| Existing `mockCtx` in test file does NOT have `orderBy` in its chain | SAFE | Verified -- `tool-processor.test.ts` lines 411-421: chain is `select().from().where().limit()` without `orderBy`; plan correctly notes this and uses `searchMockCtx` |
| `UNIFIED_TEACHER_SYSTEM` currently does NOT mention `search_catalog` | SAFE | Verified in `prompts.ts` lines 46-53 |
| Part B is cleanly deferrable without affecting Part A correctness | SAFE | `search_catalog` is a standalone tool; chat framing injection is independent |

---

### Summary

**[BLOCKER] count: 1**
**[RISK] count: 4**
**[QUESTION] count: 0**

**[BLOCKER]** (confidence: 8/10) -- Task 3's Step 2 is labeled "verify it FAILS" but the plan itself says it will PASS immediately if Task 1 is already merged. This contradictory instruction will confuse the build agent. Fix before executing: relabel Task 3 Step 2 as a verification step with "Expected: PASS", or move the integration tests into Task 1's test step and eliminate Task 3 as a separate task.

**[RISK]** (confidence: 7/10) -- System prompt does not guide disambiguation behavior when `search_catalog` returns multiple matches. LLM may silently pick wrong piece. Add one sentence: "If search_catalog returns multiple matches, present the options to the student before using a piece_id."

**[RISK]** (confidence: 7/10) -- Task 3's mock DB chain is structurally coupled to exact Drizzle call ordering. Pre-existing accepted pattern in this test suite; consistent with existing `mockCtx` usage.

**[RISK]** (confidence: 6/10) -- Task 2's prompt test asserts exact substring `"Never ask the student for a piece ID"` -- brittle to copy edits. Acceptable for prompt content tests.

**[RISK]** (confidence: 5/10) -- Whitespace-only input strings pass `.min(1)` Zod validation and generate overly broad ILIKE queries. Low real-world impact.

VERDICT: PROCEED_WITH_CAUTION -- [Fix Task 3 Step 2 label before dispatching build agent; add disambiguation guidance to system prompt update]
