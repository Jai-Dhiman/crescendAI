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
- search_catalog: When the student mentions a piece by name and you need its piece_id for other tools. Never ask the student for a piece ID -- use search_catalog to look it up yourself.
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

- [ ] **Step 2: Run test -- verify it FAILS**

```bash
cd apps/api && npx vitest run src/services/tool-processor.test.ts
```
Expected: FAIL if Task 1 hasn't been merged yet (search_catalog not in registry). If Task 1 is merged, this should PASS immediately since the implementation is already complete. In that case, this task is a verification-only commit confirming the integration behavior is tested.

- [ ] **Step 3: Implement (if needed)**

If tests pass immediately because Task 1 already implemented the full process function, proceed to commit. The integration test has independent value.

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
