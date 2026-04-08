# Catalog Search: Structured Retrieval Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** The `search_catalog` tool uses exact integer match on `opus_number` and `piece_number` so "Waltz Op. 64 No. 2" and "Waltz Op. 64 No. 3" are unambiguously resolved.
**Spec:** docs/specs/2026-04-08-catalog-search-structured-retrieval-design.md
**Style:** Follow `apps/api/TS_STYLE.md` for all files under `apps/api/src/`.

---

## Task Groups

- **Group A (parallel):** Task 1, Task 2
- **Group B (parallel, depends on Group A):** Task 3, Task 4
- **Group C (sequential, depends on Group B):** Task 5

---

## File Structure

| File | Responsibility | Interface | Depth | Action |
|------|----------------|-----------|-------|--------|
| `apps/api/src/services/catalog-parse.ts` | Parse ASAP title string → structured integer fields | `parseTitleFields(title): TitleFields` | DEEP | New |
| `apps/api/src/services/catalog-parse.test.ts` | Unit tests for parseTitleFields across all ASAP title patterns | Tests | -- | New |
| `apps/api/src/db/schema/catalog.ts` | Add `opusNumber`, `pieceNumber`, `catalogueType` columns to `pieces` table | Drizzle table columns | -- | Modify |
| `apps/api/src/db/schema/catalog.test.ts` | Schema structural test: new columns are defined | Tests | -- | New |
| `apps/api/src/db/migrations/` | Generated DDL SQL for three new nullable columns | -- | -- | New (generated) |
| `apps/api/src/scripts/backfill-piece-fields.ts` | One-time script: read all pieces, parse titles, write new fields | Standalone script | -- | New |
| `apps/api/src/services/tool-processor.ts` | Replace `searchCatalogSchema` + `searchCatalogAnthropicSchema` + `processSearchCatalog` | `TOOL_REGISTRY.search_catalog` | DEEP | Modify |
| `apps/api/src/services/tool-processor.test.ts` | Update search_catalog validation tests and behavior tests | Tests | -- | Modify |

---

### Task 1: `parseTitleFields` pure function + unit tests
**Group:** A (parallel with Task 2)

**Behavior being verified:** `parseTitleFields` extracts `opusNumber`, `pieceNumber`, and `catalogueType` from ASAP title strings — handles standard "Op./No." format, WTC trailing-dash format, and titles with no structured identifiers.
**Interface under test:** `parseTitleFields(title: string): TitleFields` from `catalog-parse.ts`

**Files:**
- Create: `apps/api/src/services/catalog-parse.ts`
- Create: `apps/api/src/services/catalog-parse.test.ts`

- [ ] **Step 1: Write the failing test**

Create `apps/api/src/services/catalog-parse.test.ts`:

```typescript
import { describe, expect, it } from "vitest";
import { parseTitleFields } from "./catalog-parse";

describe("parseTitleFields", () => {
	it("extracts opus and piece number from standard Op./No. title", () => {
		const result = parseTitleFields("Etudes Op. 10 No. 3");
		expect(result).toEqual({ opusNumber: 10, pieceNumber: 3, catalogueType: "op" });
	});

	it("extracts opus and piece number from waltz title", () => {
		const result = parseTitleFields("Waltz Op. 64 No. 2");
		expect(result).toEqual({ opusNumber: 64, pieceNumber: 2, catalogueType: "op" });
	});

	it("extracts piece number without opus for bare No. format", () => {
		const result = parseTitleFields("Ballades No. 1");
		expect(result).toEqual({ opusNumber: null, pieceNumber: 1, catalogueType: null });
	});

	it("identifies WTC catalogue type and extracts trailing piece number", () => {
		const result = parseTitleFields("WTC I - Prelude - 1");
		expect(result).toEqual({ opusNumber: null, pieceNumber: 1, catalogueType: "wtc" });
	});

	it("returns all null for title with no structured identifiers", () => {
		const result = parseTitleFields("Arabesques");
		expect(result).toEqual({ opusNumber: null, pieceNumber: null, catalogueType: null });
	});

	it("handles title with opus only and no piece number", () => {
		const result = parseTitleFields("Sonata Op. 27");
		expect(result).toEqual({ opusNumber: 27, pieceNumber: null, catalogueType: "op" });
	});

	it("handles WTC title without trailing number", () => {
		const result = parseTitleFields("WTC I - Prelude");
		expect(result).toEqual({ opusNumber: null, pieceNumber: null, catalogueType: "wtc" });
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun vitest run src/services/catalog-parse.test.ts
```
Expected: FAIL — `Cannot find module './catalog-parse'` (file does not exist yet).

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `apps/api/src/services/catalog-parse.ts`:

```typescript
export interface TitleFields {
	opusNumber: number | null;
	pieceNumber: number | null;
	catalogueType: string | null;
}

/**
 * Parses an ASAP dataset title string into structured catalog fields.
 * Titles follow a regular naming convention derived from directory paths:
 *   "Waltz Op. 64 No. 2"   → opus=64, piece=2, type="op"
 *   "WTC I - Prelude - 1"  → opus=null, piece=1, type="wtc"
 *   "Ballades No. 1"       → opus=null, piece=1, type=null
 *   "Arabesques"           → all null
 */
export function parseTitleFields(title: string): TitleFields {
	const opusMatch = title.match(/Op\.\s*(\d+)/i);
	const numberMatch = title.match(/No\.\s*(\d+)/i);
	const isWtc = /WTC/i.test(title);

	const opusNumber = opusMatch ? parseInt(opusMatch[1], 10) : null;
	let pieceNumber = numberMatch ? parseInt(numberMatch[1], 10) : null;

	// WTC titles use trailing "- N" format instead of "No. N"
	if (isWtc && pieceNumber === null) {
		const trailingNum = title.match(/[-\u2013]\s*(\d+)\s*$/);
		if (trailingNum) {
			pieceNumber = parseInt(trailingNum[1], 10);
		}
	}

	let catalogueType: string | null = null;
	if (isWtc) {
		catalogueType = "wtc";
	} else if (opusNumber !== null) {
		catalogueType = "op";
	}

	return { opusNumber, pieceNumber, catalogueType };
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun vitest run src/services/catalog-parse.test.ts
```
Expected: PASS — 7 tests passing.

- [ ] **Step 5: Commit**

```bash
cd apps/api && git add src/services/catalog-parse.ts src/services/catalog-parse.test.ts && git commit -m "feat(api): add parseTitleFields for ASAP title structured extraction"
```

---

### Task 2: DB schema — add `opusNumber`, `pieceNumber`, `catalogueType` columns
**Group:** A (parallel with Task 1)

**Behavior being verified:** The `pieces` Drizzle table exports `opusNumber`, `pieceNumber`, and `catalogueType` as accessible column objects — meaning downstream code can use `eq(pieces.opusNumber, 64)` without a TypeScript error.
**Interface under test:** `pieces.opusNumber`, `pieces.pieceNumber`, `pieces.catalogueType` from `db/schema/catalog.ts`

**Files:**
- Modify: `apps/api/src/db/schema/catalog.ts`
- Create: `apps/api/src/db/schema/catalog.test.ts`
- Create: `apps/api/src/db/migrations/` (SQL generated by drizzle-kit)

- [ ] **Step 1: Write the failing test**

Create `apps/api/src/db/schema/catalog.test.ts`:

```typescript
import { describe, expect, it } from "vitest";
import { pieces } from "./catalog";

describe("pieces schema", () => {
	it("has opusNumber, pieceNumber, catalogueType columns defined", () => {
		// TypeScript compile check: if these columns don't exist, this file won't compile.
		// Runtime check: Drizzle column objects are defined (not undefined).
		expect(pieces.opusNumber).toBeDefined();
		expect(pieces.pieceNumber).toBeDefined();
		expect(pieces.catalogueType).toBeDefined();
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun vitest run src/db/schema/catalog.test.ts
```
Expected: FAIL — TypeScript error: `Property 'opusNumber' does not exist on type 'PgTableWithColumns<...>'`. The columns are not in the schema yet.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `apps/api/src/db/schema/catalog.ts`, add three columns inside the `pgTable("pieces", { ... })` definition, after the `source` field and before `createdAt` (after line 28):

```typescript
		opusNumber: integer("opus_number"),
		pieceNumber: integer("piece_number"),
		catalogueType: text("catalogue_type"),
```

Full updated column block (lines 12-34, showing context):

```typescript
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
		opusNumber: integer("opus_number"),
		pieceNumber: integer("piece_number"),
		catalogueType: text("catalogue_type"),
		createdAt: timestamp("created_at", { withTimezone: true })
			.notNull()
			.defaultNow(),
	},
	(t) => [index("idx_pieces_composer").on(t.composer)],
);
```

Then generate the migration:

```bash
cd apps/api && DATABASE_URL=postgresql://jdhiman:postgres@localhost:5432/crescendai_dev bun run generate
```

This produces a new file in `apps/api/src/db/migrations/` with SQL equivalent to:

```sql
ALTER TABLE "pieces" ADD COLUMN "opus_number" integer;
ALTER TABLE "pieces" ADD COLUMN "piece_number" integer;
ALTER TABLE "pieces" ADD COLUMN "catalogue_type" text;
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun vitest run src/db/schema/catalog.test.ts
```
Expected: PASS — TypeScript compiles, column objects are defined.

- [ ] **Step 5: Commit**

```bash
cd apps/api && git add src/db/schema/catalog.ts src/db/schema/catalog.test.ts src/db/migrations/ && git commit -m "feat(api): add opus_number, piece_number, catalogue_type columns to pieces"
```

---

### Task 3: Backfill script — populate new columns from existing titles
**Group:** B (parallel with Task 4, depends on Group A)

**Behavior being verified:** Running the script against the dev DB updates pieces whose titles contain "Op. N" or "No. N" patterns with the correct integer values, and leaves truly unstructured titles (like "Arabesques") unchanged.
**Interface under test:** `bun src/scripts/backfill-piece-fields.ts` via manual run against dev DB — verified by inspecting output logs.

**Files:**
- Create: `apps/api/src/scripts/backfill-piece-fields.ts`

- [ ] **Step 1: Write the failing test**

The behavioral verification for this task is a manual run. Write the script first, then run it in dry-verify mode (no unit test framework can test a live DB script in isolation). The "failing" state is: the file does not exist. Create it in Step 3.

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun src/scripts/backfill-piece-fields.ts
```
Expected: FAIL — `Cannot find module` (file does not exist yet).

- [ ] **Step 3: Implement**

Create `apps/api/src/scripts/backfill-piece-fields.ts`:

```typescript
import { eq, isNull, or } from "drizzle-orm";
import { drizzle } from "drizzle-orm/postgres-js";
import postgres from "postgres";
import * as schema from "../db/schema/index";
import { parseTitleFields } from "../services/catalog-parse";

const DATABASE_URL = process.env.DATABASE_URL;
if (!DATABASE_URL) {
	throw new Error("DATABASE_URL environment variable is required");
}

const sql = postgres(DATABASE_URL);
const db = drizzle(sql, { schema });
const { pieces } = schema;

// Only fetch pieces that haven't been backfilled yet (idempotent)
const rows = await db
	.select({ pieceId: pieces.pieceId, title: pieces.title })
	.from(pieces)
	.where(
		or(
			isNull(pieces.opusNumber),
			isNull(pieces.pieceNumber),
			isNull(pieces.catalogueType),
		),
	);

console.log(JSON.stringify({ message: "backfill starting", total: rows.length }));

let updated = 0;
let skipped = 0;

for (const row of rows) {
	const fields = parseTitleFields(row.title);

	if (
		fields.opusNumber === null &&
		fields.pieceNumber === null &&
		fields.catalogueType === null
	) {
		skipped++;
		continue;
	}

	await db
		.update(pieces)
		.set({
			opusNumber: fields.opusNumber,
			pieceNumber: fields.pieceNumber,
			catalogueType: fields.catalogueType,
		})
		.where(eq(pieces.pieceId, row.pieceId));

	updated++;
	console.log(
		JSON.stringify({
			pieceId: row.pieceId,
			title: row.title,
			opusNumber: fields.opusNumber,
			pieceNumber: fields.pieceNumber,
			catalogueType: fields.catalogueType,
		}),
	);
}

console.log(JSON.stringify({ message: "backfill complete", updated, skipped }));

await sql.end();
```

- [ ] **Step 4: Run against dev DB — verify it PASSES**

First apply the migration to the dev DB:

```bash
cd apps/api && DATABASE_URL=postgresql://jdhiman:postgres@localhost:5432/crescendai_dev bun run migrate
```

Then run the backfill:

```bash
cd apps/api && DATABASE_URL=postgresql://jdhiman:postgres@localhost:5432/crescendai_dev bun src/scripts/backfill-piece-fields.ts
```

Expected: Structured JSON logs showing `updated` count for pieces with "Op." or "No." in their titles, `skipped` count for pieces like "Arabesques". Final log: `{"message":"backfill complete","updated":N,"skipped":M}` where N > 0.

- [ ] **Step 5: Commit**

```bash
cd apps/api && git add src/scripts/backfill-piece-fields.ts && git commit -m "feat(api): add backfill script for piece opus/number structured fields"
```

---

### Task 4: Updated `searchCatalogSchema` + `searchCatalogAnthropicSchema` + validation tests
**Group:** B (parallel with Task 3, depends on Group A)

**Behavior being verified:** The `search_catalog` Zod schema accepts the new structured fields (`composer`, `opus_number`, `piece_number`, `title_keywords`, `query`) and rejects out-of-range integer values. The old `title` field is no longer accepted as a top-level search field.
**Interface under test:** `TOOL_REGISTRY.search_catalog.schema` (Zod schema)

**Files:**
- Modify: `apps/api/src/services/tool-processor.ts`
- Modify: `apps/api/src/services/tool-processor.test.ts`

- [ ] **Step 1: Write the failing test**

In `apps/api/src/services/tool-processor.test.ts`, replace the existing `search_catalog schema validation` describe block (lines 412–448) with:

```typescript
// ---------------------------------------------------------------------------
// search_catalog Zod validation
// ---------------------------------------------------------------------------

describe("search_catalog schema validation", () => {
	const schema = TOOL_REGISTRY.search_catalog.schema;

	it("passes with composer only", () => {
		const result = schema.safeParse({ composer: "Chopin" });
		expect(result.success).toBe(true);
	});

	it("passes with opus_number only", () => {
		const result = schema.safeParse({ opus_number: 64 });
		expect(result.success).toBe(true);
	});

	it("passes with piece_number only", () => {
		const result = schema.safeParse({ piece_number: 2 });
		expect(result.success).toBe(true);
	});

	it("passes with title_keywords only", () => {
		const result = schema.safeParse({ title_keywords: "Waltz" });
		expect(result.success).toBe(true);
	});

	it("passes with query only", () => {
		const result = schema.safeParse({ query: "Chopin waltz" });
		expect(result.success).toBe(true);
	});

	it("passes with composer + opus_number + piece_number (primary use case)", () => {
		const result = schema.safeParse({
			composer: "Chopin",
			opus_number: 64,
			piece_number: 2,
		});
		expect(result.success).toBe(true);
	});

	it("rejects empty object", () => {
		const result = schema.safeParse({});
		expect(result.success).toBe(false);
	});

	it("rejects opus_number below 1", () => {
		const result = schema.safeParse({ opus_number: 0 });
		expect(result.success).toBe(false);
	});

	it("rejects opus_number above 9999", () => {
		const result = schema.safeParse({ opus_number: 10000 });
		expect(result.success).toBe(false);
	});

	it("rejects non-integer opus_number", () => {
		const result = schema.safeParse({ opus_number: 64.5 });
		expect(result.success).toBe(false);
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun vitest run src/services/tool-processor.test.ts
```
Expected: FAIL — `"passes with opus_number only"`: the OLD schema strips unknown fields (Zod default behavior), so `{ opus_number: 64 }` becomes `{}` after parsing → `.refine()` sees all fields undefined → returns `success: false` → test expects `true` → FAIL.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `apps/api/src/services/tool-processor.ts`, replace `searchCatalogSchema` (lines 410–422) with:

```typescript
const searchCatalogSchema = z
	.object({
		composer: z.string().min(1).max(200).optional(),
		opus_number: z.number().int().min(1).max(9999).optional(),
		piece_number: z.number().int().min(1).max(9999).optional(),
		title_keywords: z.string().min(1).max(200).optional(),
		query: z.string().min(1).max(300).optional(),
	})
	.refine(
		(data) =>
			data.composer !== undefined ||
			data.opus_number !== undefined ||
			data.piece_number !== undefined ||
			data.title_keywords !== undefined ||
			data.query !== undefined,
		{ message: "At least one search field is required" },
	);
```

Replace `searchCatalogAnthropicSchema` (lines 685–708) with:

```typescript
const searchCatalogAnthropicSchema: AnthropicToolSchema = {
	name: "search_catalog",
	description:
		"Search the piece catalog to find a piece's UUID. PREFER structured fields: use composer, opus_number, and piece_number when you can extract them from the student's words — { composer: 'Chopin', opus_number: 64, piece_number: 2 } is exact and unambiguous. Use title_keywords for genre words when opus/number are unknown. Only use query as a last resort.",
	input_schema: {
		type: "object",
		properties: {
			composer: {
				type: "string",
				description:
					"Composer last name. 'Chopin', 'Bach', 'Beethoven'. Case-insensitive substring match.",
			},
			opus_number: {
				type: "integer",
				description:
					"Opus number as integer. 'Op. 64' → 64. Exact match — most important for disambiguation.",
			},
			piece_number: {
				type: "integer",
				description:
					"Piece number within the opus as integer. 'No. 2' → 2. Exact match — critical to distinguish pieces within an opus.",
			},
			title_keywords: {
				type: "string",
				description:
					"Genre or title keywords when opus/number are unknown. 'Waltz', 'Nocturne', 'Ballade'. Token-split substring match.",
			},
			query: {
				type: "string",
				description:
					"Free-form fallback only. Use when you cannot identify composer or structured numbers — e.g., 'that slow Bach prelude'.",
			},
		},
	},
};
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun vitest run src/services/tool-processor.test.ts
```
Expected: PASS — all tests in the updated search_catalog schema validation describe block pass. Other existing tests are unaffected.

- [ ] **Step 5: Commit**

```bash
cd apps/api && git add src/services/tool-processor.ts src/services/tool-processor.test.ts && git commit -m "feat(api): update search_catalog schema to structured fields with exact integer matching"
```

---

### Task 5: Updated `processSearchCatalog` logic + behavior tests
**Group:** C (depends on Group B)

**Behavior being verified:** When `processSearchCatalog` finds no matches, it returns a message guiding the caller toward structured fields — specifically mentioning `opus_number` and `piece_number`. This distinguishes the new behavior from the old "No pieces found matching the search criteria." message.
**Interface under test:** `processToolUse("search_catalog", ...)` return shape

**Files:**
- Modify: `apps/api/src/services/tool-processor.ts`
- Modify: `apps/api/src/services/tool-processor.test.ts`

- [ ] **Step 1: Write the failing test**

In `apps/api/src/services/tool-processor.test.ts`, replace the two existing `search_catalog` tests inside the `processToolUse pass-through tools` describe block (lines 562–639) with:

```typescript
it("search_catalog returns matches for structured query", async () => {
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
		{ composer: "Chopin", opus_number: 64, piece_number: 2 },
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

it("search_catalog returns matches for query fallback", async () => {
	const fallbackMockCtx = {
		db: {
			select: () => ({
				from: () => ({
					where: () => ({
						orderBy: () => ({
							limit: () =>
								Promise.resolve([
									{
										pieceId: "def-456",
										composer: "Bach",
										title: "WTC I - Prelude - 1",
										barCount: 32,
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
		fallbackMockCtx,
		studentId,
		"search_catalog",
		{ query: "Bach WTC prelude" },
	);
	expect(result.isError).toBe(false);
	const config = result.componentsJson[0].config as {
		matches: Array<{ pieceId: string }>;
	};
	expect(config.matches).toHaveLength(1);
	expect(config.matches[0].pieceId).toBe("def-456");
});

it("search_catalog empty result message references opus_number and piece_number", async () => {
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
	// New message guides caller toward structured fields
	expect(config.message).toContain("opus_number");
	expect(config.message).toContain("piece_number");
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun vitest run src/services/tool-processor.test.ts
```
Expected: FAIL — `"search_catalog empty result message references opus_number and piece_number"`: old `processSearchCatalog` returns `"No pieces found matching the search criteria."` which does not contain `"opus_number"` → test expects `true` → FAIL.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `apps/api/src/services/tool-processor.ts`, replace `processSearchCatalog` (lines 424–489) with:

```typescript
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
	if (input.opus_number !== undefined) {
		conditions.push(eq(pieces.opusNumber, input.opus_number));
	}
	if (input.piece_number !== undefined) {
		conditions.push(eq(pieces.pieceNumber, input.piece_number));
	}
	if (input.title_keywords) {
		const tokens = input.title_keywords
			.trim()
			.split(/\s+/)
			.filter((t) => t.length >= 2);
		for (const token of tokens) {
			conditions.push(sql`${pieces.title} ILIKE ${"%" + token + "%"}`);
		}
	}
	// Free-form fallback: token-split across both fields.
	// Only applied when no structured fields provided.
	if (input.query && conditions.length === 0) {
		const tokens = input.query
			.trim()
			.split(/\s+/)
			.filter((t) => t.length >= 2);
		for (const token of tokens) {
			conditions.push(
				sql`(${pieces.composer} ILIKE ${"%" + token + "%"} OR ${pieces.title} ILIKE ${"%" + token + "%"})`,
			);
		}
	}

	if (conditions.length === 0) {
		throw new Error(
			"search_catalog: no conditions after parse (unreachable after validation)",
		);
	}

	const rows = await ctx.db
		.select({
			pieceId: pieces.pieceId,
			composer: pieces.composer,
			title: pieces.title,
			barCount: pieces.barCount,
		})
		.from(pieces)
		.where(and(...conditions))
		.orderBy(asc(pieces.composer), asc(pieces.title))
		.limit(5);

	if (rows.length === 0) {
		return [
			{
				type: "search_catalog_result",
				config: {
					matches: [],
					message:
						"No pieces found. The catalog contains ~242 ASAP pieces. Try fewer fields or use exact opus/piece numbers — e.g., { composer: 'Chopin', opus_number: 64, piece_number: 2 }.",
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

Note: `eq` is already imported at line 1 of `tool-processor.ts` (`import { and, asc, desc, eq, sql } from "drizzle-orm"`). No import changes needed.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun vitest run src/services/tool-processor.test.ts
```
Expected: PASS — all tests pass including the three new search_catalog behavior tests.

- [ ] **Step 5: Commit**

```bash
cd apps/api && git add src/services/tool-processor.ts src/services/tool-processor.test.ts && git commit -m "feat(api): update processSearchCatalog to use exact integer match for opus/piece number"
```

---

## Post-Implementation: Apply Migration to Production

After all tasks pass, apply the migration to the production DB and run the backfill:

```bash
# Apply migration
cd apps/api && bun run migrate

# Backfill existing pieces (requires production DATABASE_URL)
DATABASE_URL=<production-connection-string> bun src/scripts/backfill-piece-fields.ts
```

Verify the backfill log shows `updated` > 0 and the backfill completes without errors.
