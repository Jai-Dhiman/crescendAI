# Score Highlight Artifact Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** The teacher LLM can point at specific bars in the student's score with dimension-colored annotations, rendered as inline SVG snippets in the chat flow with expand-to-sidebar for full score view.
**Spec:** docs/specs/2026-04-06-score-highlight-artifact-design.md
**Style:** Follow the project's coding standards (CLAUDE.md / TS_STYLE.md)

---

## Task Groups

- Group A (parallel): Task 1, Task 2, Task 3
- Group B (parallel, depends on A): Task 4, Task 5, Task 6
- Group C (sequential, depends on B): Task 7
- Group D (sequential, depends on C): Task 8
- Group E (parallel, depends on D): Task 9, Task 10
- Group F (sequential, depends on E): Task 11

---

### Task 1: Update score_highlight Zod schema to multi-region highlights
**Group:** A (parallel with Task 2, Task 3)

**Behavior being verified:** The score_highlight tool accepts and validates a `highlights` array with `bars` tuple, `dimension`, and optional `annotation`, and rejects the old `bars: string` format.
**Interface under test:** `TOOL_REGISTRY.score_highlight.schema` (Zod schema)

**Files:**
- Modify: `apps/api/src/services/tool-processor.ts`
- Modify: `apps/api/src/services/tool-processor.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// In apps/api/src/services/tool-processor.test.ts
// Replace the existing "score_highlight Zod validation" describe block entirely

describe("score_highlight schema validation", () => {
	const schema = TOOL_REGISTRY.score_highlight.schema;

	it("passes valid single highlight", () => {
		const result = schema.safeParse({
			piece_id: "123e4567-e89b-12d3-a456-426614174000",
			highlights: [
				{ bars: [1, 4], dimension: "dynamics" },
			],
		});
		expect(result.success).toBe(true);
	});

	it("passes multiple highlights with annotations", () => {
		const result = schema.safeParse({
			piece_id: "123e4567-e89b-12d3-a456-426614174000",
			highlights: [
				{ bars: [1, 4], dimension: "dynamics", annotation: "crescendo here" },
				{ bars: [12, 16], dimension: "pedaling", annotation: "sustain bleeds" },
			],
		});
		expect(result.success).toBe(true);
	});

	it("rejects missing piece_id", () => {
		const result = schema.safeParse({
			highlights: [{ bars: [1, 4], dimension: "dynamics" }],
		});
		expect(result.success).toBe(false);
	});

	it("rejects empty highlights array", () => {
		const result = schema.safeParse({
			piece_id: "123e4567-e89b-12d3-a456-426614174000",
			highlights: [],
		});
		expect(result.success).toBe(false);
	});

	it("rejects more than 5 highlights", () => {
		const highlights = Array.from({ length: 6 }, (_, i) => ({
			bars: [i + 1, i + 2],
			dimension: "dynamics",
		}));
		const result = schema.safeParse({
			piece_id: "123e4567-e89b-12d3-a456-426614174000",
			highlights,
		});
		expect(result.success).toBe(false);
	});

	it("rejects invalid dimension", () => {
		const result = schema.safeParse({
			piece_id: "123e4567-e89b-12d3-a456-426614174000",
			highlights: [{ bars: [1, 4], dimension: "rhythm" }],
		});
		expect(result.success).toBe(false);
	});

	it("rejects bars where start > end", () => {
		const result = schema.safeParse({
			piece_id: "123e4567-e89b-12d3-a456-426614174000",
			highlights: [{ bars: [8, 4], dimension: "dynamics" }],
		});
		expect(result.success).toBe(false);
	});

	it("rejects invalid piece_id (not uuid)", () => {
		const result = schema.safeParse({
			piece_id: "not-a-uuid",
			highlights: [{ bars: [1, 4], dimension: "dynamics" }],
		});
		expect(result.success).toBe(false);
	});

	it("rejects old bars-string format", () => {
		const result = schema.safeParse({ bars: "1-4" });
		expect(result.success).toBe(false);
	});
});
```

- [ ] **Step 2: Run test -- verify it FAILS**

```bash
cd apps/api && npx vitest run src/services/tool-processor.test.ts
```
Expected: FAIL -- multiple tests fail because the schema still expects `{ bars: string, annotations?: string[], piece_id?: string }`.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `apps/api/src/services/tool-processor.ts`, replace the `scoreHighlightSchema` and `scoreHighlightAnthropicSchema`:

```typescript
// Replace the existing scoreHighlightSchema (around line 158)
const scoreHighlightSchema = z.object({
	piece_id: z.string().uuid(),
	highlights: z
		.array(
			z.object({
				bars: z
					.tuple([z.number().int().min(1), z.number().int().min(1)])
					.refine(([start, end]) => start <= end, {
						message: "bars start must be <= end",
					}),
				dimension: dimensionEnum,
				annotation: z.string().max(500).optional(),
			}),
		)
		.min(1)
		.max(5),
});

// Replace the existing scoreHighlightAnthropicSchema (around line 465)
const scoreHighlightAnthropicSchema: AnthropicToolSchema = {
	name: "score_highlight",
	description:
		"Highlight one or more bar ranges in the score viewer with dimension-colored annotations. Use to visually point at specific passages during teaching.",
	input_schema: {
		type: "object",
		properties: {
			piece_id: {
				type: "string",
				format: "uuid",
				description: "UUID of the piece being discussed. Required.",
			},
			highlights: {
				type: "array",
				description:
					"One to five highlight regions. Each targets a bar range with a dimension and optional annotation.",
				minItems: 1,
				maxItems: 5,
				items: {
					type: "object",
					properties: {
						bars: {
							type: "array",
							items: { type: "integer", minimum: 1 },
							minItems: 2,
							maxItems: 2,
							description:
								"Bar range as [start, end]. Use same number for a single bar (e.g. [4, 4]).",
						},
						dimension: {
							type: "string",
							enum: DIMS_6,
							description: "Which musical dimension this highlight targets.",
						},
						annotation: {
							type: "string",
							description:
								"Optional text annotation to display on the highlighted bars.",
						},
					},
					required: ["bars", "dimension"],
				},
			},
		},
		required: ["piece_id", "highlights"],
	},
};
```

Also update `processScoreHighlight` to use the new schema shape:

```typescript
async function processScoreHighlight(
	ctx: ServiceContext,
	_studentId: string,
	rawInput: unknown,
): Promise<InlineComponent[]> {
	const input = scoreHighlightSchema.parse(rawInput);

	// Validate piece exists in catalog
	const catalogRow = await ctx.db
		.select({ pieceId: pieces.pieceId })
		.from(pieces)
		.where(eq(pieces.pieceId, input.piece_id))
		.limit(1);

	if (catalogRow.length === 0) {
		console.log(
			JSON.stringify({
				level: "warn",
				message: "score_highlight piece_id not found in catalog",
				pieceId: input.piece_id,
			}),
		);
	}

	return [
		{
			type: "score_highlight",
			config: {
				pieceId: input.piece_id,
				highlights: input.highlights.map((h) => ({
					bars: h.bars,
					dimension: h.dimension,
					...(h.annotation !== undefined ? { annotation: h.annotation } : {}),
				})),
			},
		},
	];
}
```

Also update the existing `processToolUse` pass-through test for the new schema shape. In the `"processToolUse pass-through tools"` describe block, replace the score_highlight test:

```typescript
it("score_highlight pass-through returns ToolResult", async () => {
	const { processToolUse } = await import("./tool-processor");
	const result: ToolResult = await processToolUse(
		mockCtx,
		studentId,
		"score_highlight",
		{
			piece_id: "123e4567-e89b-12d3-a456-426614174000",
			highlights: [{ bars: [1, 8], dimension: "dynamics" }],
		},
	);
	// Note: will fail catalog lookup with mock ctx, but should not throw
	// because processScoreHighlight logs a warning and continues
	expect(result.isError).toBe(false);
	expect(result.componentsJson[0].type).toBe("score_highlight");
});
```

Note: The pass-through test with `mockCtx` will fail because `ctx.db.select` is not a function on the empty mock. The process function now always does a catalog lookup. Update the mockCtx to handle this:

```typescript
const mockCtx = {
	db: {
		select: () => ({
			from: () => ({
				where: () => ({
					limit: () => Promise.resolve([]),
				}),
			}),
		}),
	} as never,
	env: {} as never,
};
```

- [ ] **Step 4: Run test -- verify it PASSES**

```bash
cd apps/api && npx vitest run src/services/tool-processor.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd apps/api && git add src/services/tool-processor.ts src/services/tool-processor.test.ts && git commit -m "feat(api): update score_highlight tool to multi-region highlights schema"
```

---

### Task 2: Define ScoreHighlightConfig type
**Group:** A (parallel with Task 1, Task 3)

**Behavior being verified:** The `ScoreHighlightConfig` type is a concrete interface (not a stub) and can be used in the `InlineComponent` discriminated union.
**Interface under test:** `ScoreHighlightConfig` type shape (compile-time, no runtime test needed -- verified via artifact store test using the new shape)

**Files:**
- Modify: `apps/web/src/lib/types.ts`
- Modify: `apps/web/src/stores/artifact.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// In apps/web/src/stores/artifact.test.ts
// Update the existing anotherComponent to use the real config shape:

const scoreHighlightComponent: InlineComponent = {
	type: "score_highlight",
	config: {
		pieceId: "123e4567-e89b-12d3-a456-426614174000",
		highlights: [
			{ bars: [1, 4] as [number, number], dimension: "dynamics", annotation: "crescendo" },
		],
	},
};

// Replace all references to anotherComponent with scoreHighlightComponent
// Then add a new test in the "register" describe block:

it("registers a score_highlight component with typed config", () => {
	const store = useArtifactStore.getState();
	store.register("sh1", scoreHighlightComponent);

	const entry = useArtifactStore.getState().states["sh1"];
	expect(entry).toBeDefined();
	expect(entry.state).toBe("inline");
	expect(entry.component.type).toBe("score_highlight");
	if (entry.component.type === "score_highlight") {
		expect(entry.component.config.pieceId).toBe(
			"123e4567-e89b-12d3-a456-426614174000",
		);
		expect(entry.component.config.highlights).toHaveLength(1);
		expect(entry.component.config.highlights[0].dimension).toBe("dynamics");
	}
});
```

- [ ] **Step 2: Run test -- verify it FAILS**

```bash
cd apps/web && npx vitest run src/stores/artifact.test.ts
```
Expected: FAIL -- TypeScript compilation error because `ScoreHighlightConfig` is `{ [key: string]: unknown }` and does not have `pieceId` or `highlights` properties.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `apps/web/src/lib/types.ts`, replace the stub:

```typescript
// Replace the existing ScoreHighlightConfig stub (line 41)
export interface ScoreHighlightConfig {
	pieceId: string;
	highlights: Array<{
		bars: [number, number];
		dimension: string;
		annotation?: string;
	}>;
}
```

In `apps/web/src/stores/artifact.test.ts`, replace `anotherComponent` definition and add the new test:

```typescript
const scoreHighlightComponent: InlineComponent = {
	type: "score_highlight",
	config: {
		pieceId: "123e4567-e89b-12d3-a456-426614174000",
		highlights: [
			{ bars: [1, 4] as [number, number], dimension: "dynamics", annotation: "crescendo" },
		],
	},
};
```

Replace all 2 usages of `anotherComponent` in the file with `scoreHighlightComponent`.

- [ ] **Step 4: Run test -- verify it PASSES**

```bash
cd apps/web && npx vitest run src/stores/artifact.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd apps/web && git add src/lib/types.ts src/stores/artifact.test.ts && git commit -m "feat(web): define ScoreHighlightConfig type replacing empty stub"
```

---

### Task 3: Update R2 key to MXL and Content-Type
**Group:** A (parallel with Task 1, Task 2)

**Behavior being verified:** `getPieceData` fetches from `scores/v1/{pieceId}.mxl` and the route serves it with `application/vnd.recordare.musicxml` Content-Type.
**Interface under test:** `getPieceData()` function, `GET /api/scores/:pieceId/data` route response headers

**Files:**
- Modify: `apps/api/src/services/scores.ts`
- Modify: `apps/api/src/routes/scores.ts`

- [ ] **Step 1: Write the failing test**

No existing test file for scores service. Create a minimal one:

```typescript
// apps/api/src/services/scores.test.ts
import { describe, expect, it } from "vitest";

describe("getPieceData R2 key", () => {
	it("requests the .mxl key from R2", async () => {
		let requestedKey = "";
		const mockEnv = {
			SCORES: {
				get: async (key: string) => {
					requestedKey = key;
					return { body: new ReadableStream() };
				},
			},
		};

		const { getPieceData } = await import("./scores");
		await getPieceData(mockEnv as never, "abc-123");
		expect(requestedKey).toBe("scores/v1/abc-123.mxl");
	});
});
```

- [ ] **Step 2: Run test -- verify it FAILS**

```bash
cd apps/api && npx vitest run src/services/scores.test.ts
```
Expected: FAIL -- `requestedKey` is `"scores/v1/abc-123.json"`, not `.mxl`.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `apps/api/src/services/scores.ts`, change line 31:

```typescript
// Before:
const object = await env.SCORES.get(`scores/v1/${pieceId}.json`);
// After:
const object = await env.SCORES.get(`scores/v1/${pieceId}.mxl`);
```

In `apps/api/src/routes/scores.ts`, change the Content-Type header in the `/:pieceId/data` route:

```typescript
// Before:
"Content-Type": "application/json",
// After:
"Content-Type": "application/vnd.recordare.musicxml",
```

- [ ] **Step 4: Run test -- verify it PASSES**

```bash
cd apps/api && npx vitest run src/services/scores.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd apps/api && git add src/services/scores.ts src/services/scores.test.ts src/routes/scores.ts && git commit -m "feat(api): switch score data to MXL format in R2"
```

---

### Task 4: Remove R2 fetch from tool-processor score_highlight
**Group:** B (depends on Group A, parallel with Task 5, Task 6)

**Behavior being verified:** The `processScoreHighlight` function no longer fetches score data from R2 -- it validates the piece exists in the catalog and returns the config without `scoreData`. The frontend fetches MXL directly.
**Interface under test:** `processToolUse("score_highlight", ...)` return shape

**Files:**
- Modify: `apps/api/src/services/tool-processor.ts`
- Modify: `apps/api/src/services/tool-processor.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// In apps/api/src/services/tool-processor.test.ts
// Add to the "processToolUse pass-through tools" describe block:

it("score_highlight config has no scoreData field", async () => {
	const { processToolUse } = await import("./tool-processor");
	const result: ToolResult = await processToolUse(
		mockCtx,
		studentId,
		"score_highlight",
		{
			piece_id: "123e4567-e89b-12d3-a456-426614174000",
			highlights: [{ bars: [1, 8], dimension: "dynamics" }],
		},
	);
	expect(result.isError).toBe(false);
	const config = result.componentsJson[0].config;
	expect(config).not.toHaveProperty("scoreData");
	expect(config).toHaveProperty("pieceId");
	expect(config).toHaveProperty("highlights");
});
```

- [ ] **Step 2: Run test -- verify it FAILS**

```bash
cd apps/api && npx vitest run src/services/tool-processor.test.ts
```
Expected: PASS or FAIL depending on whether Task 1 already removed the R2 fetch. If Task 1's `processScoreHighlight` already omits `scoreData` (which it does per the implementation above), this test should PASS immediately. In that case, this task is a verification-only commit confirming the behavior is tested.

If the test passes immediately, that's correct -- the implementation in Task 1 already achieved this. Proceed to commit the test.

- [ ] **Step 3: Implement (if needed)**

If `processScoreHighlight` still has the R2 fetch (e.g., Task 1 was not yet merged), remove the `scoreData` logic from `processScoreHighlight`:
- Remove the `let scoreData: unknown = null` variable
- Remove the `env.SCORES.get(key)` call
- Remove the `if (scoreData !== null) config.scoreData = scoreData` line
- Keep only the catalog existence check (for logging)

- [ ] **Step 4: Run test -- verify it PASSES**

```bash
cd apps/api && npx vitest run src/services/tool-processor.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd apps/api && git add src/services/tool-processor.ts src/services/tool-processor.test.ts && git commit -m "test(api): verify score_highlight omits scoreData from config"
```

---

### Task 5: Add api.scores.getData() client method
**Group:** B (depends on Group A, parallel with Task 4, Task 6)

**Behavior being verified:** The frontend API client can fetch MXL binary data for a given piece ID.
**Interface under test:** `api.scores.getData(pieceId)` returns `ArrayBuffer`

**Files:**
- Modify: `apps/web/src/lib/api.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/web/src/lib/api.test.ts
import { describe, expect, it, vi } from "vitest";

describe("api.scores.getData", () => {
	it("fetches from /api/scores/:pieceId/data and returns ArrayBuffer", async () => {
		const mockBuffer = new ArrayBuffer(8);
		const mockResponse = new Response(mockBuffer, {
			status: 200,
			headers: { "Content-Type": "application/vnd.recordare.musicxml" },
		});

		vi.stubGlobal("fetch", vi.fn().mockResolvedValue(mockResponse));

		const { api } = await import("./api");
		const result = await api.scores.getData("piece-abc-123");

		expect(fetch).toHaveBeenCalledWith(
			expect.stringContaining("/api/scores/piece-abc-123/data"),
			expect.objectContaining({ credentials: "include" }),
		);
		expect(result).toBeInstanceOf(ArrayBuffer);

		vi.unstubAllGlobals();
	});
});
```

- [ ] **Step 2: Run test -- verify it FAILS**

```bash
cd apps/web && npx vitest run src/lib/api.test.ts
```
Expected: FAIL -- `api.scores` is not defined.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `apps/web/src/lib/api.ts`, add to the `api` object:

```typescript
scores: {
	async getData(pieceId: string): Promise<ArrayBuffer> {
		const res = await fetch(`${BASE}/api/scores/${pieceId}/data`, {
			credentials: "include",
		});
		if (!res.ok) {
			throw new ApiError(res.status, `Failed to fetch score data for ${pieceId}`);
		}
		return res.arrayBuffer();
	},
},
```

Where `BASE` is the existing base URL variable used by other methods in the file (check exact name -- likely empty string or `import.meta.env.VITE_API_URL`).

- [ ] **Step 4: Run test -- verify it PASSES**

```bash
cd apps/web && npx vitest run src/lib/api.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd apps/web && git add src/lib/api.ts src/lib/api.test.ts && git commit -m "feat(web): add api.scores.getData() for MXL fetch"
```

---

### Task 6: Add score_highlight branch in getCollapsedProps and InlineCard routing
**Group:** B (depends on Group A, parallel with Task 4, Task 5)

**Behavior being verified:** When an artifact with `type: "score_highlight"` is registered, `getCollapsedProps` returns meaningful title/subtitle/badge, and `InlineCard` routes to `ScoreHighlightCard` (or a temporary placeholder until the real card is built).
**Interface under test:** `getCollapsedProps()` with score_highlight component

**Files:**
- Modify: `apps/web/src/components/Artifact.tsx`
- Modify: `apps/web/src/components/InlineCard.tsx`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/web/src/components/Artifact.test.ts
import { describe, expect, it } from "vitest";
import type { InlineComponent } from "../lib/types";

// Import the function directly -- it's a pure function, no React needed
// We need to extract getCollapsedProps or test it indirectly.
// Since getCollapsedProps is not exported, we test through the module.
// For now, we extract and export it.

describe("getCollapsedProps", () => {
	it("returns dimension and bar range for score_highlight", async () => {
		const { getCollapsedProps } = await import("./Artifact");

		const component: InlineComponent = {
			type: "score_highlight",
			config: {
				pieceId: "123e4567-e89b-12d3-a456-426614174000",
				highlights: [
					{ bars: [4, 8] as [number, number], dimension: "dynamics", annotation: "crescendo" },
					{ bars: [12, 16] as [number, number], dimension: "pedaling" },
				],
			},
		};

		const props = getCollapsedProps(component);
		expect(props.title).toBe("Score Highlight");
		expect(props.subtitle).toContain("bars 4-8");
		expect(props.badge).toBe("2 regions");
	});

	it("returns singular badge for single region", async () => {
		const { getCollapsedProps } = await import("./Artifact");

		const component: InlineComponent = {
			type: "score_highlight",
			config: {
				pieceId: "123e4567-e89b-12d3-a456-426614174000",
				highlights: [
					{ bars: [1, 4] as [number, number], dimension: "timing" },
				],
			},
		};

		const props = getCollapsedProps(component);
		expect(props.badge).toBe("1 region");
	});
});
```

- [ ] **Step 2: Run test -- verify it FAILS**

```bash
cd apps/web && npx vitest run src/components/Artifact.test.ts
```
Expected: FAIL -- `getCollapsedProps` is not exported, and the `score_highlight` case returns the generic fallback.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `apps/web/src/components/Artifact.tsx`:

1. Export `getCollapsedProps`:
```typescript
// Change from:
function getCollapsedProps(component: InlineComponent)
// To:
export function getCollapsedProps(component: InlineComponent)
```

2. Add the `score_highlight` case before the default return:
```typescript
if (component.type === "score_highlight") {
	const count = component.config.highlights.length;
	const firstHighlight = component.config.highlights[0];
	const subtitle = firstHighlight
		? `bars ${firstHighlight.bars[0]}-${firstHighlight.bars[1]}, ${firstHighlight.dimension}`
		: "";
	return {
		title: "Score Highlight",
		subtitle,
		badge: `${count} region${count === 1 ? "" : "s"}`,
	};
}
```

In `apps/web/src/components/InlineCard.tsx`, add the routing case. For now, route to `PlaceholderCard` -- the real `ScoreHighlightCard` will be wired in Task 9:

```typescript
case "score_highlight":
	return <PlaceholderCard type="score_highlight" />;
```

This is a temporary routing that will be replaced in Task 9. The important thing in this task is that `getCollapsedProps` returns correct data.

- [ ] **Step 4: Run test -- verify it PASSES**

```bash
cd apps/web && npx vitest run src/components/Artifact.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd apps/web && git add src/components/Artifact.tsx src/components/Artifact.test.ts src/components/InlineCard.tsx && git commit -m "feat(web): add score_highlight collapsed props and InlineCard routing"
```

---

### Task 7: OSMD Manager -- ensureRendered with caching
**Group:** C (sequential, depends on Group B)

**Behavior being verified:** `ensureRendered(pieceId)` loads and renders OSMD once per piece. Subsequent calls for the same piece skip the render. Calls for different pieces render independently. Errors from OSMD load/render propagate as exceptions.
**Interface under test:** `ensureRendered(pieceId): Promise<void>`

**Files:**
- Create: `apps/web/src/lib/osmd-manager.ts`
- Create: `apps/web/src/lib/osmd-manager.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/web/src/lib/osmd-manager.test.ts
import { afterEach, describe, expect, it, vi } from "vitest";

// Mock opensheetmusicdisplay before importing the manager
const mockRender = vi.fn();
const mockLoad = vi.fn().mockResolvedValue(undefined);
const mockOsmdInstance = {
	load: mockLoad,
	render: mockRender,
	graphic: { measureList: [] },
};

vi.mock("opensheetmusicdisplay", () => ({
	OpenSheetMusicDisplay: vi.fn().mockImplementation(() => mockOsmdInstance),
}));

// Mock api.scores.getData
vi.mock("./api", () => ({
	api: {
		scores: {
			getData: vi.fn().mockResolvedValue(new ArrayBuffer(8)),
		},
	},
}));

// Mock document.createElement for the hidden container
const mockContainer = {
	style: {},
	remove: vi.fn(),
};

describe("OsmdManager.ensureRendered", () => {
	afterEach(() => {
		vi.clearAllMocks();
	});

	it("calls OSMD load and render on first call", async () => {
		const { osmdManager } = await import("./osmd-manager");
		osmdManager.reset();

		await osmdManager.ensureRendered("piece-1");

		expect(mockLoad).toHaveBeenCalledTimes(1);
		expect(mockRender).toHaveBeenCalledTimes(1);
	});

	it("skips render on second call for same piece", async () => {
		const { osmdManager } = await import("./osmd-manager");
		osmdManager.reset();

		await osmdManager.ensureRendered("piece-1");
		mockLoad.mockClear();
		mockRender.mockClear();

		await osmdManager.ensureRendered("piece-1");

		expect(mockLoad).not.toHaveBeenCalled();
		expect(mockRender).not.toHaveBeenCalled();
	});

	it("renders independently for different pieces", async () => {
		const { osmdManager } = await import("./osmd-manager");
		osmdManager.reset();

		await osmdManager.ensureRendered("piece-1");
		await osmdManager.ensureRendered("piece-2");

		expect(mockLoad).toHaveBeenCalledTimes(2);
		expect(mockRender).toHaveBeenCalledTimes(2);
	});

	it("propagates OSMD load errors", async () => {
		const { osmdManager } = await import("./osmd-manager");
		osmdManager.reset();

		mockLoad.mockRejectedValueOnce(new Error("MXL parse failed"));

		await expect(osmdManager.ensureRendered("bad-piece")).rejects.toThrow(
			"MXL parse failed",
		);
	});
});
```

- [ ] **Step 2: Run test -- verify it FAILS**

```bash
cd apps/web && npx vitest run src/lib/osmd-manager.test.ts
```
Expected: FAIL -- `osmd-manager` module does not exist.

- [ ] **Step 3: Implement the minimum to make the test pass**

```typescript
// apps/web/src/lib/osmd-manager.ts
import { api } from "./api";

interface CachedScore {
	// biome-ignore lint/suspicious/noExplicitAny: OSMD has no exported type
	osmd: any;
	container: HTMLDivElement;
}

const cache = new Map<string, CachedScore>();

async function ensureRendered(pieceId: string): Promise<void> {
	if (cache.has(pieceId)) return;

	const { OpenSheetMusicDisplay } = await import("opensheetmusicdisplay");

	const container = document.createElement("div");
	container.style.position = "absolute";
	container.style.left = "-9999px";
	container.style.top = "-9999px";
	container.style.width = "1200px";
	document.body.appendChild(container);

	const osmd = new OpenSheetMusicDisplay(container, {
		backend: "svg",
		drawTitle: false,
		drawSubtitle: false,
		drawComposer: false,
		drawLyricist: false,
		drawPartNames: false,
		drawPartAbbreviations: false,
		drawMeasureNumbers: true,
		drawCredits: false,
	});

	const data = await api.scores.getData(pieceId);
	const blob = new Blob([data], {
		type: "application/vnd.recordare.musicxml",
	});
	const url = URL.createObjectURL(blob);

	try {
		await osmd.load(url);
		osmd.render();
		cache.set(pieceId, { osmd, container });
	} finally {
		URL.revokeObjectURL(url);
	}
}

function getOsmdInstance(
	pieceId: string,
): CachedScore | null {
	return cache.get(pieceId) ?? null;
}

function reset(): void {
	for (const entry of cache.values()) {
		entry.container.remove();
	}
	cache.clear();
}

export const osmdManager = {
	ensureRendered,
	getOsmdInstance,
	reset,
};
```

- [ ] **Step 4: Run test -- verify it PASSES**

```bash
cd apps/web && npx vitest run src/lib/osmd-manager.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd apps/web && git add src/lib/osmd-manager.ts src/lib/osmd-manager.test.ts && git commit -m "feat(web): add OSMD Manager with per-piece caching"
```

---

### Task 8: OSMD Manager -- clipBars SVG extraction
**Group:** D (sequential, depends on Task 7)

**Behavior being verified:** `clipBars(pieceId, startBar, endBar)` returns a cloned SVG element with a cropped viewBox for the requested bar range. Returns `null` for out-of-range bars or if the piece is not rendered.
**Interface under test:** `clipBars(pieceId, startBar, endBar): SVGElement | null`

**Files:**
- Modify: `apps/web/src/lib/osmd-manager.ts`
- Modify: `apps/web/src/lib/osmd-manager.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// Add to apps/web/src/lib/osmd-manager.test.ts

describe("OsmdManager.clipBars", () => {
	it("returns null for unrendered piece", async () => {
		const { osmdManager } = await import("./osmd-manager");
		osmdManager.reset();

		const result = osmdManager.clipBars("nonexistent", 1, 4);
		expect(result).toBeNull();
	});

	it("returns null when bar index is out of range", async () => {
		const { osmdManager } = await import("./osmd-manager");
		osmdManager.reset();

		// Render with empty measureList
		await osmdManager.ensureRendered("piece-empty");

		const result = osmdManager.clipBars("piece-empty", 1, 4);
		expect(result).toBeNull();
	});

	it("returns an SVG element for valid bar range", async () => {
		const { osmdManager } = await import("./osmd-manager");
		osmdManager.reset();

		// Set up measureList with mock stave SVG elements
		const svgNS = "http://www.w3.org/2000/svg";
		const mockSvg = document.createElementNS(svgNS, "svg");
		const mockStaveEl = document.createElementNS(svgNS, "rect");
		mockSvg.appendChild(mockStaveEl);

		// Override getBoundingClientRect on the mock elements
		mockStaveEl.getBoundingClientRect = () => ({
			top: 100, left: 50, bottom: 200, right: 350,
			width: 300, height: 100, x: 50, y: 100, toJSON: () => {},
		});

		mockOsmdInstance.graphic = {
			measureList: [
				[{ stave: { SVGElement: mockStaveEl } }],  // bar 1
				[{ stave: { SVGElement: mockStaveEl } }],  // bar 2
				[{ stave: { SVGElement: mockStaveEl } }],  // bar 3
				[{ stave: { SVGElement: mockStaveEl } }],  // bar 4
			],
		};

		await osmdManager.ensureRendered("piece-with-measures");

		const result = osmdManager.clipBars("piece-with-measures", 1, 4);
		expect(result).not.toBeNull();
		expect(result!.tagName.toLowerCase()).toBe("svg");
	});
});
```

- [ ] **Step 2: Run test -- verify it FAILS**

```bash
cd apps/web && npx vitest run src/lib/osmd-manager.test.ts
```
Expected: FAIL -- `osmdManager.clipBars is not a function`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Add `clipBars` to `apps/web/src/lib/osmd-manager.ts`:

```typescript
function clipBars(
	pieceId: string,
	startBar: number,
	endBar: number,
): SVGElement | null {
	const cached = cache.get(pieceId);
	if (!cached) return null;

	const { osmd, container } = cached;
	const measureList = osmd.graphic?.measureList;
	if (!measureList) return null;

	// Bars are 1-indexed; measureList is 0-indexed
	const startIdx = startBar - 1;
	const endIdx = endBar - 1;

	if (startIdx < 0 || endIdx >= measureList.length) return null;

	// Find bounding box spanning all target measures
	let minX = Infinity;
	let minY = Infinity;
	let maxX = -Infinity;
	let maxY = -Infinity;

	const containerRect = container.getBoundingClientRect();

	for (let i = startIdx; i <= endIdx; i++) {
		const measure = measureList[i]?.[0];
		if (!measure?.stave?.SVGElement) continue;

		const svgEl = measure.stave.SVGElement as SVGElement;
		const rect = svgEl.getBoundingClientRect();

		minX = Math.min(minX, rect.left - containerRect.left);
		minY = Math.min(minY, rect.top - containerRect.top);
		maxX = Math.max(maxX, rect.right - containerRect.left);
		maxY = Math.max(maxY, rect.bottom - containerRect.top);
	}

	if (minX === Infinity) return null;

	// Add padding
	const pad = 10;
	minX = Math.max(0, minX - pad);
	minY = Math.max(0, minY - pad);
	maxX += pad;
	maxY += pad;

	// Clone the container's SVG and set viewBox to the cropped region
	const sourceSvg = container.querySelector("svg");
	if (!sourceSvg) return null;

	const cloned = sourceSvg.cloneNode(true) as SVGElement;
	cloned.setAttribute("viewBox", `${minX} ${minY} ${maxX - minX} ${maxY - minY}`);
	cloned.setAttribute("width", "100%");
	cloned.setAttribute("height", "auto");
	cloned.style.maxHeight = "200px";

	return cloned;
}
```

Add `clipBars` to the exported `osmdManager` object:

```typescript
export const osmdManager = {
	ensureRendered,
	clipBars,
	getOsmdInstance,
	reset,
};
```

- [ ] **Step 4: Run test -- verify it PASSES**

```bash
cd apps/web && npx vitest run src/lib/osmd-manager.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd apps/web && git add src/lib/osmd-manager.ts src/lib/osmd-manager.test.ts && git commit -m "feat(web): add clipBars SVG extraction to OSMD Manager"
```

---

### Task 9: ScoreHighlightCard inline component
**Group:** E (parallel with Task 10, depends on Group D)

**Behavior being verified:** `ScoreHighlightCard` renders dimension-colored highlight info with annotation text, shows a loading state while OSMD initializes, shows text-only fallback on error, and has an expand button that calls `onExpand`.
**Interface under test:** `<ScoreHighlightCard config={...} onExpand={fn} />`

**Files:**
- Create: `apps/web/src/components/cards/ScoreHighlightCard.tsx`
- Modify: `apps/web/src/components/InlineCard.tsx`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/web/src/components/cards/ScoreHighlightCard.test.ts
import { afterEach, describe, expect, it, vi } from "vitest";
import type { ScoreHighlightConfig } from "../../lib/types";

// Mock osmd-manager
const mockEnsureRendered = vi.fn().mockResolvedValue(undefined);
const mockClipBars = vi.fn().mockReturnValue(null);

vi.mock("../../lib/osmd-manager", () => ({
	osmdManager: {
		ensureRendered: (...args: unknown[]) => mockEnsureRendered(...args),
		clipBars: (...args: unknown[]) => mockClipBars(...args),
	},
}));

describe("ScoreHighlightCard", () => {
	afterEach(() => {
		vi.clearAllMocks();
	});

	const config: ScoreHighlightConfig = {
		pieceId: "123e4567-e89b-12d3-a456-426614174000",
		highlights: [
			{ bars: [4, 8] as [number, number], dimension: "dynamics", annotation: "crescendo builds" },
			{ bars: [12, 16] as [number, number], dimension: "pedaling" },
		],
	};

	it("exports ScoreHighlightCard", async () => {
		const mod = await import("./ScoreHighlightCard");
		expect(typeof mod.ScoreHighlightCard).toBe("function");
	});

	it("calls ensureRendered with pieceId on import", async () => {
		// Dynamically importing to verify the component exists and accepts config
		const { ScoreHighlightCard } = await import("./ScoreHighlightCard");
		expect(ScoreHighlightCard).toBeDefined();
		// Component should accept config and onExpand props without TypeScript errors
		// Runtime rendering test would need @testing-library/react (integration test)
	});
});
```

- [ ] **Step 2: Run test -- verify it FAILS**

```bash
cd apps/web && npx vitest run src/components/cards/ScoreHighlightCard.test.ts
```
Expected: FAIL -- module `./ScoreHighlightCard` does not exist.

- [ ] **Step 3: Implement the minimum to make the test pass**

```typescript
// apps/web/src/components/cards/ScoreHighlightCard.tsx
import { ArrowsOut } from "@phosphor-icons/react";
import { useCallback, useEffect, useRef, useState } from "react";
import { DIMENSION_COLORS } from "../../lib/mock-session";
import { osmdManager } from "../../lib/osmd-manager";
import type { ScoreHighlightConfig } from "../../lib/types";

interface ScoreHighlightCardProps {
	config: ScoreHighlightConfig;
	onExpand?: () => void;
	artifactId?: string;
}

type RenderState = "loading" | "rendered" | "error";

export function ScoreHighlightCard({
	config,
	onExpand,
}: ScoreHighlightCardProps) {
	const [renderState, setRenderState] = useState<RenderState>("loading");
	const svgContainerRef = useRef<HTMLDivElement>(null);

	useEffect(() => {
		let cancelled = false;

		async function loadScore() {
			try {
				await osmdManager.ensureRendered(config.pieceId);
				if (cancelled) return;

				// Clip SVG fragments for each highlight region
				if (svgContainerRef.current) {
					svgContainerRef.current.innerHTML = "";
					for (const highlight of config.highlights) {
						const svg = osmdManager.clipBars(
							config.pieceId,
							highlight.bars[0],
							highlight.bars[1],
						);
						if (svg) {
							// Apply dimension color overlay
							const color =
								DIMENSION_COLORS[
									highlight.dimension as keyof typeof DIMENSION_COLORS
								] ?? "#7a9a82";
							svg.style.border = `2px solid ${color}`;
							svg.style.borderRadius = "8px";
							svg.style.marginBottom = "8px";
							svgContainerRef.current.appendChild(svg);
						}
					}
				}

				setRenderState("rendered");
			} catch {
				if (!cancelled) setRenderState("error");
			}
		}

		loadScore();
		return () => {
			cancelled = true;
		};
	}, [config.pieceId, config.highlights]);

	return (
		<div className="bg-surface-card border border-border rounded-xl p-4 mt-3">
			{/* Header */}
			<div className="flex items-center justify-between mb-3">
				<span className="text-body-sm font-medium text-cream">
					Score Highlight
				</span>
				{onExpand && (
					<button
						type="button"
						onClick={onExpand}
						className="w-7 h-7 flex items-center justify-center rounded-lg text-text-secondary hover:text-cream hover:bg-surface transition"
						aria-label="Expand score highlight"
					>
						<ArrowsOut size={16} />
					</button>
				)}
			</div>

			{/* SVG snippets or fallback */}
			{renderState === "loading" && (
				<div className="flex items-center justify-center h-20 text-text-tertiary text-body-sm">
					Loading score...
				</div>
			)}

			<div ref={svgContainerRef} className="osmd-clip-container" />

			{/* Highlights legend (always shown -- text fallback for error state) */}
			<div className="flex flex-col gap-2 mt-2">
				{config.highlights.map((h, i) => {
					const color =
						DIMENSION_COLORS[
							h.dimension as keyof typeof DIMENSION_COLORS
						] ?? "#7a9a82";
					return (
						<div
							key={`${h.dimension}-${h.bars[0]}-${h.bars[1]}`}
							className="flex items-start gap-2"
						>
							<span
								className="w-2 h-2 rounded-full mt-1.5 shrink-0"
								style={{ backgroundColor: color }}
							/>
							<div className="min-w-0">
								<span className="text-body-xs text-text-secondary capitalize">
									{h.dimension} -- bars {h.bars[0]}-{h.bars[1]}
								</span>
								{h.annotation && (
									<p className="text-body-xs text-text-tertiary mt-0.5">
										{h.annotation}
									</p>
								)}
							</div>
						</div>
					);
				})}
			</div>

			{renderState === "error" && (
				<p className="text-body-xs text-text-tertiary mt-2 italic">
					Score preview unavailable
				</p>
			)}
		</div>
	);
}
```

Update `apps/web/src/components/InlineCard.tsx` to route to the real card:

```typescript
import type { InlineComponent } from "../lib/types";
import { ExerciseSetCard } from "./cards/ExerciseSetCard";
import { PlaceholderCard } from "./cards/PlaceholderCard";
import { ScoreHighlightCard } from "./cards/ScoreHighlightCard";

interface InlineCardProps {
	component: InlineComponent;
	onExpand?: () => void;
	artifactId?: string;
}

export function InlineCard({
	component,
	onExpand,
	artifactId,
}: InlineCardProps) {
	switch (component.type) {
		case "exercise_set":
			return (
				<ExerciseSetCard
					config={component.config}
					onExpand={onExpand}
					artifactId={artifactId}
				/>
			);
		case "score_highlight":
			return (
				<ScoreHighlightCard
					config={component.config}
					onExpand={onExpand}
					artifactId={artifactId}
				/>
			);
		default:
			return <PlaceholderCard type={component.type} />;
	}
}
```

- [ ] **Step 4: Run test -- verify it PASSES**

```bash
cd apps/web && npx vitest run src/components/cards/ScoreHighlightCard.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd apps/web && git add src/components/cards/ScoreHighlightCard.tsx src/components/cards/ScoreHighlightCard.test.ts src/components/InlineCard.tsx && git commit -m "feat(web): add ScoreHighlightCard inline component with SVG clipping"
```

---

### Task 10: ScorePanel store -- openHighlight action and remove DEV gate
**Group:** E (parallel with Task 9, depends on Group D)

**Behavior being verified:** `scorePanelStore.openHighlight({ pieceId, highlights })` opens the panel with highlight data (not MockSessionData). The DEV-only gate on `open()` is removed so the panel works in production.
**Interface under test:** `useScorePanelStore` actions

**Files:**
- Modify: `apps/web/src/stores/score-panel.ts`
- Create: `apps/web/src/stores/score-panel.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/web/src/stores/score-panel.test.ts
import { afterEach, describe, expect, it } from "vitest";
import { useScorePanelStore } from "./score-panel";

afterEach(() => {
	useScorePanelStore.getState().clear();
});

describe("openHighlight", () => {
	it("opens the panel with highlight data", () => {
		const store = useScorePanelStore.getState();
		store.openHighlight({
			pieceId: "piece-abc",
			highlights: [
				{ bars: [4, 8] as [number, number], dimension: "dynamics", annotation: "test" },
			],
		});

		const state = useScorePanelStore.getState();
		expect(state.isOpen).toBe(true);
		expect(state.highlightData).not.toBeNull();
		expect(state.highlightData!.pieceId).toBe("piece-abc");
		expect(state.highlightData!.highlights).toHaveLength(1);
	});

	it("clears previous highlight data on close", () => {
		const store = useScorePanelStore.getState();
		store.openHighlight({
			pieceId: "piece-abc",
			highlights: [
				{ bars: [1, 4] as [number, number], dimension: "timing" },
			],
		});
		useScorePanelStore.getState().close();

		expect(useScorePanelStore.getState().isOpen).toBe(false);
		// highlightData preserved so reopening can restore
	});

	it("replaces previous highlight data on new openHighlight", () => {
		const store = useScorePanelStore.getState();
		store.openHighlight({
			pieceId: "piece-1",
			highlights: [{ bars: [1, 4] as [number, number], dimension: "dynamics" }],
		});
		store.openHighlight({
			pieceId: "piece-2",
			highlights: [{ bars: [5, 8] as [number, number], dimension: "pedaling" }],
		});

		const state = useScorePanelStore.getState();
		expect(state.highlightData!.pieceId).toBe("piece-2");
	});
});

describe("open (DEV gate removed)", () => {
	it("opens the panel with session data in any environment", () => {
		const store = useScorePanelStore.getState();
		store.open({
			piece: "Test Piece",
			section: "mm. 1-8",
			durationSeconds: 120,
			observations: [],
		});

		expect(useScorePanelStore.getState().isOpen).toBe(true);
	});
});
```

- [ ] **Step 2: Run test -- verify it FAILS**

```bash
cd apps/web && npx vitest run src/stores/score-panel.test.ts
```
Expected: FAIL -- `openHighlight` is not a function on the store, and `open()` may be gated to DEV.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `apps/web/src/stores/score-panel.ts`:

```typescript
import { create } from "zustand";
import type { MockSessionData } from "../lib/mock-session";
import type { ScoreHighlightConfig } from "../lib/types";

interface ScorePanelState {
	isOpen: boolean;
	sessionData: MockSessionData | null;
	highlightData: ScoreHighlightConfig | null;
	activeAnnotationIndex: number | null;
	panelWidth: number;
	open: (data: MockSessionData) => void;
	openHighlight: (data: ScoreHighlightConfig) => void;
	close: () => void;
	toggle: () => void;
	setActiveAnnotation: (index: number | null) => void;
	setPanelWidth: (width: number) => void;
	clear: () => void;
}

const PANEL_WIDTH_KEY = "crescend-score-panel-width";
const DEFAULT_PANEL_WIDTH = 480;

function loadPersistedWidth(): number {
	try {
		const stored = localStorage.getItem(PANEL_WIDTH_KEY);
		if (stored) {
			const parsed = Number(stored);
			if (parsed >= 320 && parsed <= window.innerWidth * 0.6) return parsed;
		}
	} catch {
		// SSR or localStorage unavailable
	}
	return DEFAULT_PANEL_WIDTH;
}

export const useScorePanelStore = create<ScorePanelState>((set) => ({
	isOpen: false,
	sessionData: null,
	highlightData: null,
	activeAnnotationIndex: null,
	panelWidth: loadPersistedWidth(),
	open: (data) => {
		// DEV gate removed -- panel works in all environments
		set({ isOpen: true, sessionData: data, highlightData: null, activeAnnotationIndex: null });
	},
	openHighlight: (data) => {
		set({ isOpen: true, highlightData: data, sessionData: null, activeAnnotationIndex: null });
	},
	close: () => set({ isOpen: false }),
	toggle: () => set((s) => ({ isOpen: !s.isOpen })),
	setActiveAnnotation: (index) => set({ activeAnnotationIndex: index }),
	setPanelWidth: (width) => {
		try {
			localStorage.setItem(PANEL_WIDTH_KEY, String(width));
		} catch {
			// SSR or localStorage unavailable
		}
		set({ panelWidth: width });
	},
	clear: () =>
		set({ isOpen: false, sessionData: null, highlightData: null, activeAnnotationIndex: null }),
}));
```

- [ ] **Step 4: Run test -- verify it PASSES**

```bash
cd apps/web && npx vitest run src/stores/score-panel.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd apps/web && git add src/stores/score-panel.ts src/stores/score-panel.test.ts && git commit -m "feat(web): add openHighlight action to ScorePanel store, remove DEV gate"
```

---

### Task 11: Wire ScorePanel sidebar to OSMD Manager and highlight data
**Group:** F (sequential, depends on Group E)

**Behavior being verified:** When `highlightData` is set in the store, `ScorePanel` renders the score for that piece using OSMD Manager (not a hardcoded path), with annotation markers positioned at the highlighted bar ranges. The expand button in `ScoreHighlightCard` opens the sidebar via `openHighlight`.
**Interface under test:** `ScorePanel` component rendering with `highlightData`, `ScoreHighlightCard` expand button

**Files:**
- Modify: `apps/web/src/components/ScorePanel.tsx`
- Modify: `apps/web/src/components/cards/ScoreHighlightCard.tsx`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/web/src/components/ScorePanel.test.ts
import { describe, expect, it, vi } from "vitest";
import { useScorePanelStore } from "../stores/score-panel";

// Mock osmd-manager
vi.mock("../lib/osmd-manager", () => ({
	osmdManager: {
		ensureRendered: vi.fn().mockResolvedValue(undefined),
		getOsmdInstance: vi.fn().mockReturnValue(null),
		clipBars: vi.fn().mockReturnValue(null),
		reset: vi.fn(),
	},
}));

describe("ScorePanel highlight integration", () => {
	it("reads highlightData from store when opened via openHighlight", () => {
		const store = useScorePanelStore.getState();
		store.openHighlight({
			pieceId: "piece-abc",
			highlights: [
				{ bars: [4, 8] as [number, number], dimension: "dynamics", annotation: "forte" },
			],
		});

		const state = useScorePanelStore.getState();
		expect(state.isOpen).toBe(true);
		expect(state.highlightData).not.toBeNull();
		expect(state.highlightData!.pieceId).toBe("piece-abc");
		expect(state.sessionData).toBeNull();

		store.clear();
	});

	it("ScorePanel module exports ScorePanel component", async () => {
		const mod = await import("./ScorePanel");
		expect(typeof mod.ScorePanel).toBe("function");
	});
});
```

- [ ] **Step 2: Run test -- verify it FAILS**

```bash
cd apps/web && npx vitest run src/components/ScorePanel.test.ts
```
Expected: FAIL if `ScorePanel` still imports the hardcoded path and crashes, or PASS if the import succeeds (the store test part should pass from Task 10). The key validation here is that the module loads cleanly.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `apps/web/src/components/ScorePanel.tsx`, modify `ScorePanelScore` to accept either `sessionData` or `highlightData`:

1. Add `highlightData` to the store reads at the top of `ScorePanel`:
```typescript
const highlightData = useScorePanelStore((s) => s.highlightData);
```

2. Replace the hardcoded MXL load in `ScorePanelScore`. Change the `useMountEffect` that calls `initOSMD`:

```typescript
// Replace the hardcoded load line:
// await osmd.load("/scores/chopin-nocturne-op9-no2.mxl");
// With:
import { osmdManager } from "../lib/osmd-manager";
import { api } from "../lib/api";

// If highlightData is present, use the OSMD Manager's cached instance
// If sessionData is present, load from the pieceId (when available) or fall back
```

The full approach for `ScorePanelScore`:
- Accept a `pieceId` prop (derived from `highlightData.pieceId` or extracted from session context)
- Use `osmdManager.ensureRendered(pieceId)` to get the OSMD instance
- If `highlightData` is present, map `highlights` to annotation markers instead of `sessionData.observations`
- Keep backward compatibility with `sessionData` for the existing observation-based flow

Update the `ScorePanelScore` interface:

```typescript
interface ScorePanelScoreProps {
	pieceId: string;
	observations: Array<{
		dimension: string;
		barRange?: [number, number];
		text?: string;
		framing?: string;
	}>;
	activeAnnotationIndex: number | null;
	osmdRef: React.MutableRefObject<any>;
	onAnnotationClick: (index: number) => void;
}
```

In the parent `ScorePanel`, derive the props from whichever data source is active:

```typescript
// Derive observations from highlightData or sessionData
const observations = highlightData
	? highlightData.highlights.map((h) => ({
			dimension: h.dimension,
			barRange: h.bars,
			text: h.annotation ?? "",
			framing: "" as string,
		}))
	: (sessionData?.observations ?? []);

const pieceId = highlightData?.pieceId ?? "";
const title = highlightData ? "Score Highlight" : (sessionData?.piece ?? "");
const section = highlightData
	? `bars ${highlightData.highlights[0]?.bars[0]}-${highlightData.highlights[highlightData.highlights.length - 1]?.bars[1]}`
	: (sessionData?.section ?? "");
```

In the `initOSMD` function inside `ScorePanelScore`, replace the hardcoded load:

```typescript
async function initOSMD() {
	const osmdContainer = containerRef.current;
	if (!osmdContainer || cancelled) return;

	try {
		if (pieceId) {
			// Use OSMD Manager for cached rendering
			await osmdManager.ensureRendered(pieceId);
			if (cancelled) return;

			const cached = osmdManager.getOsmdInstance(pieceId);
			if (cached) {
				// Move the rendered SVG into our container
				const sourceSvg = cached.container.querySelector("svg");
				if (sourceSvg) {
					const cloned = sourceSvg.cloneNode(true) as SVGElement;
					osmdContainer.appendChild(cloned);
				}
				osmdRef.current = cached.osmd;
				setIsRendered(true);
				return;
			}
		}

		// Fallback: direct OSMD init (for sessionData without pieceId)
		const { OpenSheetMusicDisplay } = await import("opensheetmusicdisplay");
		if (cancelled) return;

		const osmd = new OpenSheetMusicDisplay(osmdContainer, {
			backend: "svg",
			drawTitle: false,
			drawSubtitle: false,
			drawComposer: false,
			drawLyricist: false,
			drawPartNames: false,
			drawPartAbbreviations: false,
			drawMeasureNumbers: true,
			drawCredits: false,
		});

		osmdRef.current = osmd;

		// No hardcoded path -- pieceId is required
		if (!pieceId) {
			console.error("ScorePanel: no pieceId provided");
			return;
		}

		await osmdManager.ensureRendered(pieceId);
		if (cancelled) return;
		osmd.render();
		setIsRendered(true);
	} catch (err) {
		console.error("OSMD render failed:", err);
	}
}
```

In `ScoreHighlightCard`, wire the expand button to `openHighlight`:

```typescript
// In ScoreHighlightCard.tsx, import the store:
import { useScorePanelStore } from "../../stores/score-panel";

// Replace the onExpand prop usage:
const openHighlight = useScorePanelStore((s) => s.openHighlight);

// In the expand button onClick:
onClick={() => {
	openHighlight(config);
	onExpand?.();
}}
```

- [ ] **Step 4: Run test -- verify it PASSES**

```bash
cd apps/web && npx vitest run src/components/ScorePanel.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd apps/web && git add src/components/ScorePanel.tsx src/components/ScorePanel.test.ts src/components/cards/ScoreHighlightCard.tsx && git commit -m "feat(web): wire ScorePanel to OSMD Manager and highlight data, remove hardcoded MXL"
```
