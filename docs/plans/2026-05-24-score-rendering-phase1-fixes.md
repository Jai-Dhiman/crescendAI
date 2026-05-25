# Score Rendering Phase 1 Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** Make the score rendering pipeline correct against today's API, without changing the API.
**Spec:** docs/specs/2026-05-24-score-rendering-phase1-fixes-design.md
**Style:** Follow `CLAUDE.md` + `apps/CLAUDE.md` (web tier uses Bun, TanStack Start, biome).

**SVG insertion convention.** Verovio output is a controlled string (no user input). The existing codebase inserts it via `useLayoutEffect` + `ref.current.insertAdjacentHTML("afterbegin", svgMarkup)` — see `ScorePanel.tsx:280`, the deleted `SvgClip.tsx`, etc. Phase 1 keeps this convention; do NOT introduce `dangerouslySetInnerHTML`.

## Task Groups

```
Group A (sequential — 1 task):       Task 1
Group B (depends on A, 1 task):       Task 2
Group C (depends on B, 1 task):       Task 3
Group D (depends on C, parallel):     Task 4, Task 5, Task 5b
Group E (depends on D, 1 task):       Task 6
```

Tasks 4, 5, and 5b touch disjoint files (`ScoreHighlightCard.tsx`, `PlayPassageCard.tsx`, `ExerciseSetCard.tsx`) and can run in parallel.

**Branching checkpoint inside Task 2:** Task 2 includes a manual validation step against the live sandbox before committing. If `tk.select()` produces broken output on real pieces, **halt the plan**, revise Tasks 2–5 to use Approach B (`SvgClipBBox`) as the canonical clip path instead. See the explicit instructions in Task 2.

---

### Task 1: Worker — Extract `loadPiece` and stop swallowing measure-index failures

**Group:** A (no dependencies)

**Behavior being verified:** When `buildMeasureIndex` throws (e.g., Verovio's `renderToTimemap` fails), `loadPiece` returns the `"failed"` sentinel instead of returning a degraded `CacheEntry` with an empty measure array. This prevents the silent-fallback bug where every clip method then degrades to "render page 1".

**Interface under test:** Newly exported `loadPiece(bytes, bindings, pieceId?)` function.

**Files:**
- Modify: `apps/web/src/lib/score-worker.ts`
- Modify: `apps/web/src/lib/score-worker.test.ts`

- [ ] **Step 1: Write the failing test**

Append to `apps/web/src/lib/score-worker.test.ts` (after the existing `describe("renderFullSvg")` block):

```typescript
describe("loadPiece — silent fallback regression", () => {
	it("returns 'failed' when buildMeasureIndex throws (no silent degradation to page 1)", async () => {
		const throwingTk = {
			loadZipDataBuffer: vi.fn().mockReturnValue(true),
			renderToTimemap: vi.fn(() => {
				throw new Error("timemap exploded");
			}),
			setOptions: vi.fn(),
			loadData: vi.fn().mockReturnValue(true),
		};
		const FakeToolkitClass = vi
			.fn()
			.mockImplementation(() => throwingTk) as unknown as new (
			mod: unknown,
		) => typeof throwingTk;

		const { loadPiece } = await import("./score-worker");

		const bytes = new TextEncoder().encode(
			"<?xml version='1.0'?><score-partwise/>",
		).buffer;

		const result = await loadPiece(
			bytes,
			{
				// biome-ignore lint/suspicious/noExplicitAny: test bindings
				module: {} as any,
				ToolkitClass: FakeToolkitClass,
			},
			"test-piece",
		);

		expect(result).toBe("failed");
		expect(throwingTk.renderToTimemap).toHaveBeenCalled();
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run vitest run src/lib/score-worker.test.ts -t "silent fallback regression"
```

Expected: FAIL — `loadPiece is not exported from "./score-worker"`.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `apps/web/src/lib/score-worker.ts`:

1. Add this exported interface and function at module scope, after the existing `renderClipSvgMxl` definition (around line 224) and BEFORE the `if (typeof window === "undefined")` block:

```typescript
export interface VerovioBindings {
	module: unknown;
	// biome-ignore lint/suspicious/noExplicitAny: dynamic Verovio ESM class
	ToolkitClass: new (mod: unknown) => any;
}

export async function loadPiece(
	bytes: ArrayBuffer,
	bindings: VerovioBindings,
	pieceId?: string,
): Promise<LoadResult> {
	const { module, ToolkitClass } = bindings;
	const ZIP_MAGIC = 0x04034b50;
	const isZip =
		bytes.byteLength >= 4 &&
		new DataView(bytes).getUint32(0, true) === ZIP_MAGIC;

	let xmlContent: string | null = null;
	let tk = new ToolkitClass(module);
	tk.setOptions(VEROVIO_OPTS);
	let loaded = false;

	if (isZip) {
		try {
			const clone = bytes.slice(0);
			loaded = tk.loadZipDataBuffer(clone) as boolean;
		} catch {
			tk = new ToolkitClass(module);
			tk.setOptions(VEROVIO_OPTS);
		}
	}

	if (!loaded) {
		if (isZip) {
			try {
				xmlContent = await extractXmlFromMxl(bytes);
			} catch {
				// extraction failed; loaded stays false
			}
		} else {
			try {
				xmlContent = new TextDecoder().decode(bytes);
			} catch {
				// non-text binary
			}
		}

		if (xmlContent !== null) {
			tk = new ToolkitClass(module);
			tk.setOptions(VEROVIO_OPTS);
			try {
				const clean = xmlContent.replace(
					/<!DOCTYPE\s[^>[]*(\[[^\]]*\])?\s*>/g,
					"",
				);
				loaded = tk.loadData(clean) as boolean;
			} catch {
				// loadData fallback failed
			}
		}
	}

	if (!loaded) return "failed";

	const entry: CacheEntry = { tk, measures: [], xmlContent };
	try {
		entry.measures = buildMeasureIndex(tk);
	} catch (e) {
		console.error(
			`[score-worker] buildMeasureIndex failed for ${pieceId ?? "?"}:`,
			e,
		);
		return "failed";
	}
	return entry;
}
```

2. Inside the `if (typeof window === "undefined")` block, DELETE the old inline `async function loadPiece(bytes, pieceId?)` (the existing lines 341–423 of the current file). Update the message-handler call site that today reads `loadPiece(msg.bytes, msg.pieceId)` (around line 463) to:

```typescript
const loadPromise = loadPiece(
	msg.bytes,
	{ module: verovioModule, ToolkitClass: VerovioToolkitClass },
	msg.pieceId,
);
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run vitest run src/lib/score-worker.test.ts
```

Expected: PASS (all worker tests).

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/lib/score-worker.ts apps/web/src/lib/score-worker.test.ts && git commit -m "fix(web): loadPiece returns failed when measure index cannot be built

Removes the silent-fallback swallow that caused every clip method to
degrade to page 1 when buildMeasureIndex threw. loadPiece is now
exported and parameterized over its Verovio bindings so the regression
is testable."
```

---

### Task 2: Worker — Route `render_clip` to `tk.select` and delete dead approach helpers

**Group:** B (depends on Group A)

**Behavior being verified:** The worker's `render_clip` message dispatch calls `tk.select({start, end})` with the measureOn IDs derived from the cached measure index, then renders page 1. The `method` field on `WorkerInMsg` and the `'mei'` / `'mxl'` branches are removed.

**Interface under test:** New exported `processRenderClipRequest(tk, measures, startBar, endBar)` function.

**Files:**
- Modify: `apps/web/src/lib/score-worker.ts`
- Modify: `apps/web/src/lib/score-worker.test.ts`

**Branching checkpoint — perform BEFORE Step 1:**

Validate `tk.select` produces correct clips against the live sandbox before writing code. The selector `g.measure` is correct for Verovio output (verified in `app.sandbox.tsx:799` and the investigation's Playwright probe — Verovio emits `<g class="measure">` for measure containers). The validation must scope to Approach C's specific SVG, not the whole page.

1. Start dev server: `just dev-light` (separate terminal).
2. Navigate Playwright (via MCP) to `http://localhost:3000/app/sandbox`. Wait for the "Rendering Approaches" section to finish loading.
3. Run this exact `browser_evaluate` to scope a count to Approach C and assert correctness:

```js
() => {
  // Find the label "C — tk.select()" by font-mono text content, then walk up
  // to the row container and grab the SVG underneath.
  const labels = Array.from(document.querySelectorAll('.font-mono'));
  const cLabel = labels.find(l => /^C\s/.test(l.textContent?.trim() ?? ''));
  if (!cLabel) return { error: 'Approach C label not found' };
  let scope = cLabel.closest('div');
  while (scope && !scope.querySelector('svg')) scope = scope.parentElement;
  const svg = scope?.querySelector('svg');
  if (!svg) return { error: 'Approach C SVG not found' };
  return {
    measures: svg.querySelectorAll('g.measure').length,
    bytes: svg.outerHTML.length,
  };
}
```

Expected for the default sandbox config (`startBar=135, endBar=136`): `measures === 2`.

4. Repeat the validation for two additional bar ranges. Edit `apps/web/src/routes/app.sandbox.tsx` — the `<ApproachesComparison>` JSX is at approximately line 1178 with literal numeric props. Change `startBar={135} endBar={136}` to `startBar={1} endBar={4}` (expect `measures === 4`), then to `startBar={200} endBar={208}` (expect `measures === 9`). The dev server will hot-reload. **Revert the JSX back to `startBar={135} endBar={136}` after validation.**

**If any case shows the wrong measure count or visually broken output (missing barlines, cross-system splits):** HALT THIS TASK. Revise the plan: change Tasks 2-5 to use Approach B (`SvgClipBBox`) instead. The fallback plan is: keep `renderClipSvg` (default helper) and `SvgClipBBox.tsx`; do NOT introduce `processRenderClipRequest`; do NOT change `getClip`'s return type; just wire `SvgClipBBox` into both cards (replacing `SvgClip`) and delete only `SvgClip.tsx`, `renderClipSvgMei`, `renderClipSvgMxl`, and the sandbox ApproachesComparison.

**If all three cases pass:** proceed with Step 1.

- [ ] **Step 1: Write the failing test**

In `apps/web/src/lib/score-worker.test.ts`, REPLACE the existing `describe("renderClipSvg", ...)` block (lines 31-73) with:

```typescript
describe("processRenderClipRequest", () => {
	it("calls tk.select with measureOn IDs from the index and renders page 1", async () => {
		const mockSelect = vi.fn();
		const tk = {
			...fakeTk,
			select: mockSelect,
		};
		const { processRenderClipRequest } = await import("./score-worker");
		// biome-ignore lint/suspicious/noExplicitAny: test mock
		const svg = processRenderClipRequest(tk as any, fakeMeasures, 2, 4);
		expect(mockSelect).toHaveBeenCalledWith({
			start: "measure-id-2",
			end: "measure-id-4",
		});
		expect(mockRenderToSVG).toHaveBeenCalledWith(1);
		expect(svg).toBe("<svg>clip-svg</svg>");
	});

	it("falls back to a full page-1 render when the start bar is out of range", async () => {
		const tk = { ...fakeTk, select: vi.fn() };
		const { processRenderClipRequest } = await import("./score-worker");
		// biome-ignore lint/suspicious/noExplicitAny: test mock
		const svg = processRenderClipRequest(tk as any, fakeMeasures, 999, 1000);
		expect(tk.select).not.toHaveBeenCalled();
		expect(mockRenderToSVG).toHaveBeenCalledWith(1);
		expect(svg).toBe("<svg>clip-svg</svg>");
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run vitest run src/lib/score-worker.test.ts -t "processRenderClipRequest"
```

Expected: FAIL — `processRenderClipRequest is not exported from "./score-worker"`.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `apps/web/src/lib/score-worker.ts`:

1. ADD this exported function at module scope, immediately after `renderClipSvgSelect`:

```typescript
// Canonical render_clip dispatch. Uses tk.select so Verovio engraves
// only the requested bars with correct musical context.
export function processRenderClipRequest(
	tk: VerovioTk,
	measures: MeasureEntry[],
	startBar: number,
	endBar: number,
): string {
	return renderClipSvgSelect(tk, measures, startBar, endBar);
}
```

2. DELETE the following from `apps/web/src/lib/score-worker.ts`:
   - The entire `renderClipSvg` function (the default full-page + measureIds helper, around lines 76–91).
   - The entire `renderClipSvgMei` function (around lines 122–149).
   - The entire `renderClipSvgMxl` function (around lines 154–224).
   - The `xmlContent` field from `CacheEntry` (line 14). New shape: `interface CacheEntry { tk: VerovioTk; measures: MeasureEntry[]; }`.

3. In the module-scope `loadPiece` (added in Task 1), REPLACE the fallback block that handles non-ZIP / ZIP-extract decode with this version (which drops `xmlContent` storage):

```typescript
if (!loaded) {
	let fallbackXml: string | null = null;
	if (isZip) {
		try {
			fallbackXml = await extractXmlFromMxl(bytes);
		} catch {
			// extraction failed
		}
	} else {
		try {
			fallbackXml = new TextDecoder().decode(bytes);
		} catch {
			// non-text binary
		}
	}
	if (fallbackXml !== null) {
		tk = new ToolkitClass(module);
		tk.setOptions(VEROVIO_OPTS);
		try {
			const clean = fallbackXml.replace(
				/<!DOCTYPE\s[^>[]*(\[[^\]]*\])?\s*>/g,
				"",
			);
			loaded = tk.loadData(clean) as boolean;
		} catch {
			// loadData fallback failed
		}
	}
}
```

Also: delete the top-level `let xmlContent: string | null = null;` in `loadPiece` and update the `CacheEntry` construction at the end to `const entry: CacheEntry = { tk, measures: [] };`.

4. REPLACE the `WorkerInMsg` type (around line 296) with:

```typescript
type WorkerInMsg =
	| {
			type: "render_clip";
			requestId: string;
			pieceId: string;
			startBar: number;
			endBar: number;
			bytes?: ArrayBuffer;
	  }
	| {
			type: "render_full";
			requestId: string;
			pieceId: string;
			bytes?: ArrayBuffer;
			pageWidth?: number;
	  };
```

5. In the message handler inside the `if (typeof window === "undefined")` block, REPLACE the entire `if (msg.type === "render_clip") { ... }` branch (the existing method-based dispatch between lines 481-543) with:

```typescript
if (msg.type === "render_clip") {
	const svg = processRenderClipRequest(
		tk,
		measures,
		msg.startBar,
		msg.endBar,
	);
	(self as unknown as Worker).postMessage({
		requestId: msg.requestId,
		svg,
	});
} else {
```

The `else` block (render_full handling) is unchanged.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run vitest run src/lib/score-worker.test.ts
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/lib/score-worker.ts apps/web/src/lib/score-worker.test.ts && git commit -m "refactor(web): worker render_clip uses tk.select; drop dead approaches

Default render_clip path now invokes processRenderClipRequest ->
renderClipSvgSelect, producing a self-contained SVG with correct
musical context. Deletes renderClipSvg (Approach A), renderClipSvgMei
(Approach D - 1019ms), renderClipSvgMxl (Approach E - dead code), and
xmlContent storage on CacheEntry. Removes the 'method' message field."
```

---

### Task 3: Renderer — `getClip` returns `Promise<string>`; drop `getClipMethod`

**Group:** C (depends on Group B)

**Behavior being verified:** `scoreRenderer.getClip(pieceId, startBar, endBar)` resolves with the SVG string directly (not a `ClipResult` envelope). `scoreRenderer.getClipMethod` is removed from the public API.

**Interface under test:** `scoreRenderer.getClip` return type and value.

**Files:**
- Modify: `apps/web/src/lib/score-renderer.ts`
- Modify: `apps/web/src/lib/score-renderer.test.ts`

- [ ] **Step 1: Write the failing test**

The existing test at `apps/web/src/lib/score-renderer.test.ts:32-38` already asserts the post-fix behavior:

```typescript
it("resolves with the SVG string returned by the Worker", async () => {
	const { scoreRenderer } = await import("./score-renderer");
	const svg = await scoreRenderer.getClip("chopin.ballades.1", 1, 4);
	expect(svg).toBe("<svg>mock</svg>");
	expect(mockGetData).toHaveBeenCalledWith("chopin.ballades.1");
});
```

No new test code needed — this test fails today because `getClip` returns `Promise<ClipResult>` (an object), and `expect(svg).toBe("<svg>mock</svg>")` does strict equality against an object.

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run vitest run src/lib/score-renderer.test.ts -t "resolves with the SVG string"
```

Expected: FAIL — `expected { svg: '<svg>mock</svg>', startMeasureId: null, endMeasureId: null } to be '<svg>mock</svg>'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `apps/web/src/lib/score-renderer.ts`:

1. DELETE the `ClipResult` interface export (lines 4–8).

2. REPLACE the `PendingClip` type (lines 16–21) with:

```typescript
type PendingClip = {
	kind: "clip";
	resolve: (svg: string) => void;
	reject: (err: Error) => void;
	pieceId: string;
};
```

3. REPLACE the worker `onmessage` handler body (lines 42–71) with:

```typescript
this.worker.onmessage = (
	e: MessageEvent<{
		requestId: string;
		svg?: string;
		error?: string;
	}>,
) => {
	const { requestId, svg, error } = e.data;
	const pending = this.pendingRequests.get(requestId);
	if (!pending) return;
	this.pendingRequests.delete(requestId);
	if (error !== undefined) {
		this.sentPieceIds.delete(pending.pieceId);
		pending.reject(new Error(error));
	} else if (svg !== undefined) {
		pending.resolve(svg);
	} else {
		pending.reject(new Error("Worker returned no svg and no error"));
	}
};
```

4. CHANGE `getClip`'s return type (line 136) from `Promise<ClipResult>` to `Promise<string>`. No other body change is required — the resolve call already passes the string through.

5. DELETE the entire `getClipMethod` method (lines 173–210).

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run vitest run src/lib/score-renderer.test.ts
```

Expected: PASS (all renderer tests).

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/lib/score-renderer.ts apps/web/src/lib/score-renderer.test.ts && git commit -m "refactor(web): scoreRenderer.getClip returns SVG string; drop getClipMethod

getClip now returns Promise<string> matching the worker's new contract
(self-contained SVG from tk.select). Removes ClipResult and the
getClipMethod public surface (no production callers)."
```

---

### Task 4: ScoreHighlightCard — parallel-load clips and render pre-cropped SVG

**Group:** D (parallel with Task 5; depends on Group C)

**Behavior being verified:** A `ScoreHighlightCard` with N highlights issues N concurrent calls to `scoreRenderer.getClip` (parallel, not sequential) and renders the returned SVG strings without any client-side cropping component.

**Interface under test:** The rendered DOM of `<ScoreHighlightCard config={...} />`.

**Files:**
- Modify: `apps/web/src/components/cards/ScoreHighlightCard.tsx`
- Modify: `apps/web/src/components/cards/ScoreHighlightCard.test.tsx`

- [ ] **Step 1: Write the failing test**

REPLACE the contents of `apps/web/src/components/cards/ScoreHighlightCard.test.tsx` with:

```typescript
// src/components/cards/ScoreHighlightCard.test.tsx
import { render, screen, waitFor } from "@testing-library/react";
import * as React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ScoreHighlightConfig } from "../../lib/types";

const mockGetClip = vi.fn();
vi.mock("../../lib/score-renderer", () => ({
	scoreRenderer: {
		getClip: (...args: unknown[]) => mockGetClip(...args),
	},
}));

beforeEach(() => {
	vi.clearAllMocks();
});

describe("ScoreHighlightCard", () => {
	it("renders dimension label, bar range, and annotation when getClip rejects", async () => {
		mockGetClip.mockRejectedValue(new Error("Worker unavailable"));
		const config: ScoreHighlightConfig = {
			pieceId: "chopin.ballades.1",
			highlights: [
				{
					bars: [1, 4] as [number, number],
					dimension: "dynamics",
					annotation: "hushed opening",
				},
			],
		};
		const { ScoreHighlightCard } = await import("./ScoreHighlightCard");
		render(React.createElement(ScoreHighlightCard, { config }));
		await waitFor(() => {
			expect(screen.getByText("dynamics")).toBeInTheDocument();
			expect(screen.getByText(/bars 1/)).toBeInTheDocument();
			expect(screen.getByText("hushed opening")).toBeInTheDocument();
			expect(mockGetClip).toHaveBeenCalledWith("chopin.ballades.1", 1, 4);
		});
	});

	it("issues all getClip calls in parallel for multi-highlight configs and renders each SVG", async () => {
		const resolvers: Array<(svg: string) => void> = [];
		mockGetClip.mockImplementation(
			() =>
				new Promise<string>((resolve) => {
					resolvers.push(resolve);
				}),
		);

		const config: ScoreHighlightConfig = {
			pieceId: "chopin.ballades.1",
			highlights: [
				{ bars: [1, 4] as [number, number], dimension: "dynamics", annotation: "a" },
				{ bars: [5, 8] as [number, number], dimension: "timing", annotation: "b" },
				{ bars: [9, 12] as [number, number], dimension: "pedaling", annotation: "c" },
			],
		};
		const { ScoreHighlightCard } = await import("./ScoreHighlightCard");
		render(React.createElement(ScoreHighlightCard, { config }));

		// All three getClip calls must be issued before any resolves.
		await waitFor(() => {
			expect(mockGetClip).toHaveBeenCalledTimes(3);
		});

		resolvers[0]("<svg data-bars='1-4'></svg>");
		resolvers[1]("<svg data-bars='5-8'></svg>");
		resolvers[2]("<svg data-bars='9-12'></svg>");

		// jsdom's SVG support is incomplete — query by attribute selector
		// can return null even when insertAdjacentHTML succeeds. Assert on
		// the parent container's innerHTML to verify each SVG string landed.
		await waitFor(() => {
			const html = document.body.innerHTML;
			expect(html).toContain("data-bars='1-4'");
			expect(html).toContain("data-bars='5-8'");
			expect(html).toContain("data-bars='9-12'");
		});
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run vitest run src/components/cards/ScoreHighlightCard.test.tsx
```

Expected: FAIL on the parallel test — current `for…await` calls `mockGetClip` only ONCE before the first promise must resolve. `waitFor` times out at `mockGetClip).toHaveBeenCalledTimes(3)`.

- [ ] **Step 3: Implement the minimum to make the test pass**

REPLACE the contents of `apps/web/src/components/cards/ScoreHighlightCard.tsx` with:

```typescript
import { ArrowsOut } from "@phosphor-icons/react";
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { DIMENSION_COLORS } from "../../lib/mock-session";
import { scoreRenderer } from "../../lib/score-renderer";
import type { ScoreHighlightConfig } from "../../lib/types";
import { useScorePanelStore } from "../../stores/score-panel";

interface ScoreHighlightCardProps {
	config: ScoreHighlightConfig;
	onExpand?: () => void;
	artifactId?: string;
}

type RenderState = "loading" | "rendered" | "error";

interface LoadedClip {
	svg: string;
	dimension: string;
	bars: [number, number];
	annotation?: string;
}

function ClipSvg({ svg }: { svg: string }) {
	const ref = useRef<HTMLDivElement>(null);
	useLayoutEffect(() => {
		if (!ref.current) return;
		ref.current.textContent = "";
		// biome-ignore lint/security/noDomManipulation: controlled SVG from Verovio WASM, not user input
		ref.current.insertAdjacentHTML("afterbegin", svg);
		const svgEl = ref.current.querySelector("svg");
		if (svgEl) {
			svgEl.setAttribute("width", "100%");
			svgEl.removeAttribute("height");
			(svgEl as SVGElement).style.display = "block";
		}
	}, [svg]);
	return <div ref={ref} className="[&>svg]:w-full [&>svg]:block" />;
}

export function ScoreHighlightCard({
	config,
	onExpand,
}: ScoreHighlightCardProps) {
	const [renderState, setRenderState] = useState<RenderState>("loading");
	const [clips, setClips] = useState<LoadedClip[]>([]);
	const openHighlight = useScorePanelStore((s) => s.openHighlight);

	// biome-ignore lint/correctness/useExhaustiveDependencies: highlights array identity is not stable; JSON.stringify produces a stable change signal
	useEffect(() => {
		let cancelled = false;

		Promise.all(
			config.highlights.map((highlight) =>
				scoreRenderer
					.getClip(config.pieceId, highlight.bars[0], highlight.bars[1])
					.then((svg) => ({
						svg,
						dimension: highlight.dimension,
						bars: highlight.bars,
						annotation: highlight.annotation,
					})),
			),
		)
			.then((results) => {
				if (cancelled) return;
				setClips(results);
				setRenderState("rendered");
			})
			.catch((err) => {
				console.error("ScoreHighlightCard: failed to load score", err);
				if (!cancelled) setRenderState("error");
			});

		return () => {
			cancelled = true;
		};
	}, [config.pieceId, JSON.stringify(config.highlights)]);

	return (
		<div className="bg-surface-card border border-border rounded-xl overflow-hidden mt-3">
			{renderState === "loading" && (
				<div className="h-10 flex items-center justify-center">
					<div className="w-3.5 h-3.5 rounded-full border-2 border-text-tertiary/50 border-t-transparent animate-spin" />
				</div>
			)}

			{renderState === "rendered" && clips.length > 0 && (
				<div className="px-3 pt-3 pb-0 flex flex-col gap-2">
					{clips.map((clip) => {
						const color =
							DIMENSION_COLORS[
								clip.dimension as keyof typeof DIMENSION_COLORS
							] ?? "#7a9a82";
						return (
							<div
								key={`${clip.dimension}-${clip.bars[0]}-${clip.bars[1]}`}
								style={{
									position: "relative",
									borderRadius: "6px",
									border: `1.5px solid ${color}40`,
									backgroundColor: "white",
									overflow: "hidden",
								}}
							>
								<ClipSvg svg={clip.svg} />
								<div
									style={{
										position: "absolute",
										inset: 0,
										backgroundColor: `${color}22`,
										borderRadius: "5px",
										pointerEvents: "none",
									}}
								/>
							</div>
						);
					})}
				</div>
			)}

			<div
				className={`p-4 flex flex-col gap-3.5 ${
					renderState === "rendered" && clips.length > 0
						? "border-t border-border/40"
						: ""
				}`}
			>
				<div className="flex items-center justify-between">
					<span className="text-body-xs text-text-tertiary">
						{config.highlights.length === 1
							? "1 annotation"
							: `${config.highlights.length} annotations`}
					</span>
					{onExpand && (
						<button
							type="button"
							onClick={() => {
								openHighlight(config);
								onExpand?.();
							}}
							className="w-6 h-6 flex items-center justify-center rounded text-text-tertiary hover:text-cream hover:bg-surface transition-colors"
							aria-label="Expand score highlight"
						>
							<ArrowsOut size={13} />
						</button>
					)}
				</div>

				{config.highlights.map((h) => {
					const color =
						DIMENSION_COLORS[h.dimension as keyof typeof DIMENSION_COLORS] ??
						"#7a9a82";
					return (
						<div
							key={`${h.dimension}-${h.bars[0]}-${h.bars[1]}`}
							className="flex items-start gap-3"
						>
							<div className="flex items-center gap-1.5 shrink-0 mt-1">
								<span
									className="w-1.5 h-1.5 rounded-full shrink-0"
									style={{ backgroundColor: color }}
								/>
								<span className="text-label-sm text-text-tertiary uppercase tracking-wide">
									{h.dimension}
								</span>
							</div>
							<div className="min-w-0">
								<span className="text-body-xs text-text-tertiary">
									bars {h.bars[0]}–{h.bars[1]}
								</span>
								{h.annotation && (
									<p className="text-body-sm text-text-primary mt-0.5 leading-snug">
										{h.annotation}
									</p>
								)}
							</div>
						</div>
					);
				})}
			</div>
		</div>
	);
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run vitest run src/components/cards/ScoreHighlightCard.test.tsx
```

Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/components/cards/ScoreHighlightCard.tsx apps/web/src/components/cards/ScoreHighlightCard.test.tsx && git commit -m "feat(web): ScoreHighlightCard parallel-loads clips and renders pre-cropped SVGs

useEffect uses Promise.all to issue all getClip calls concurrently.
SVG strings from getClip (pre-cropped by tk.select) render via a small
ClipSvg helper using the codebase's existing insertAdjacentHTML pattern
- no SvgClip cropping component needed."
```

---

### Task 5: PlayPassageCard — render pre-cropped SVG from new getClip

**Group:** D (parallel with Task 4; depends on Group C)

**Behavior being verified:** `PlayPassageCard` renders the SVG string returned by `scoreRenderer.getClip` directly via the codebase's `insertAdjacentHTML` pattern. No `SvgClip` cropping component is used.

**Interface under test:** The rendered DOM of `<PlayPassageCard config={...} />` with a mocked clip.

**Files:**
- Modify: `apps/web/src/components/cards/PlayPassageCard.tsx`
- Modify: `apps/web/src/components/cards/PlayPassageCard.test.tsx`

- [ ] **Step 1: Write the failing test**

Add to `apps/web/src/components/cards/PlayPassageCard.test.tsx` (keep existing tests; replace any that destructure a `ClipResult` shape from `_mockClip`):

```typescript
import { render, waitFor } from "@testing-library/react";
import * as React from "react";
import { describe, expect, it } from "vitest";
import type { PassageManifest, PlayPassageConfig } from "../../lib/types";

describe("PlayPassageCard SVG rendering", () => {
	it("renders the SVG string passed via _mockClip", async () => {
		const manifest: PassageManifest = {
			pieceId: "chopin.ballades.1",
			sessionId: "session-1",
			bars: [5, 8],
			chunks: [],
		};
		const config: PlayPassageConfig = {
			sessionId: "session-1",
			bars: [5, 8] as [number, number],
			dimension: "timing",
			annotation: "rushing here",
		};
		const { PlayPassageCard } = await import("./PlayPassageCard");
		render(
			React.createElement(PlayPassageCard, {
				config,
				_mockManifest: manifest,
				_mockClip: "<svg data-test='passage-clip'></svg>",
			}),
		);

		// jsdom SVG attribute querying is unreliable — assert on innerHTML
		// to verify insertAdjacentHTML landed the string.
		await waitFor(() => {
			expect(document.body.innerHTML).toContain("data-test='passage-clip'");
		});
	});
});
```

**Update existing mocks in `PlayPassageCard.test.tsx`.** Before adding the new test above, read the existing `PlayPassageCard.test.tsx` file. Any existing `mockGetClip.mockResolvedValue({ svg: "<svg>...</svg>", startMeasureId: null, endMeasureId: null })` call must become `mockGetClip.mockResolvedValue("<svg>...</svg>")` (a string, not an object). Apply this transformation to every occurrence. Also: any existing test that constructs `_mockClip` as an object literal (e.g., `_mockClip: { svg: "...", startMeasureId: null, endMeasureId: null }`) must change to `_mockClip: "<svg>...</svg>"` (a string).

If `PassageManifest` requires more fields than `pieceId/sessionId/bars/chunks`, look up the type in `apps/web/src/lib/types.ts` and supply realistic values for the remaining fields.

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run vitest run src/components/cards/PlayPassageCard.test.tsx -t "renders the SVG string passed via _mockClip"
```

Expected: FAIL — TypeScript reports `_mockClip` must be `ClipResult` (an object), so the test won't compile; OR if it does compile, the `data-test='passage-clip'` attribute never reaches the DOM because the current render path is `<SvgClip svgMarkup={clip.svg} ... />` and `clip.svg` of a string treated as `ClipResult` is `undefined`.

- [ ] **Step 3: Implement the minimum to make the test pass**

REPLACE the contents of `apps/web/src/components/cards/PlayPassageCard.tsx` with:

```typescript
// apps/web/src/components/cards/PlayPassageCard.tsx
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { api } from "../../lib/api";
import { DIMENSION_COLORS } from "../../lib/mock-session";
import { PassagePlayer } from "../../lib/passage-player";
import { scoreRenderer } from "../../lib/score-renderer";
import type { PassageManifest, PlayPassageConfig } from "../../lib/types";

interface PlayPassageCardProps {
	config: PlayPassageConfig;
	onExpand?: () => void;
	artifactId?: string;
	_mockManifest?: PassageManifest;
	_mockClip?: string;
	_playable?: boolean;
}

type LoadState = "loading" | "ready" | "audio_error" | "error";

function ClipSvg({ svg }: { svg: string }) {
	const ref = useRef<HTMLDivElement>(null);
	useLayoutEffect(() => {
		if (!ref.current) return;
		ref.current.textContent = "";
		// biome-ignore lint/security/noDomManipulation: controlled SVG from Verovio WASM, not user input
		ref.current.insertAdjacentHTML("afterbegin", svg);
		const svgEl = ref.current.querySelector("svg");
		if (svgEl) {
			svgEl.setAttribute("width", "100%");
			svgEl.removeAttribute("height");
			(svgEl as SVGElement).style.display = "block";
		}
	}, [svg]);
	return <div ref={ref} className="[&>svg]:w-full [&>svg]:block" />;
}

export function PlayPassageCard({
	config,
	onExpand,
	artifactId: _artifactId,
	_mockManifest,
	_mockClip,
	_playable,
}: PlayPassageCardProps) {
	const [loadState, setLoadState] = useState<LoadState>("loading");
	const [clipSvg, setClipSvg] = useState<string | null>(null);
	const [manifest, setManifest] = useState<PassageManifest | null>(null);
	const playerRef = useRef<PassagePlayer | null>(null);
	const ctxRef = useRef<AudioContext | null>(null);

	useEffect(() => {
		if (_mockManifest && _mockClip && !_playable) {
			setManifest(_mockManifest);
			setClipSvg(_mockClip);
			setLoadState("ready");
			return;
		}

		let cancelled = false;
		(async () => {
			let m: PassageManifest;
			let svg: string;
			try {
				m =
					_mockManifest ??
					(await api.sessions.getPassage(config.sessionId, config.bars));
				if (!_mockManifest && cancelled) return;
				svg =
					_mockClip ??
					(await scoreRenderer.getClip(
						m.pieceId,
						config.bars[0],
						config.bars[1],
					));
				if (!_mockClip && cancelled) return;
			} catch (err) {
				console.error("PlayPassageCard fetch failed", err);
				if (!cancelled) setLoadState("error");
				return;
			}
			setManifest(m);
			setClipSvg(svg);

			try {
				const ctx = new AudioContext();
				ctxRef.current = ctx;
				const player = new PassagePlayer(m, ctx);
				await player.load();
				if (cancelled) {
					player.destroy();
					ctxRef.current = null;
					return;
				}
				playerRef.current = player;
				setLoadState("ready");
			} catch (err) {
				console.error("PlayPassageCard audio load failed", err);
				if (!cancelled) setLoadState("audio_error");
			}
		})();
		return () => {
			cancelled = true;
			playerRef.current?.destroy();
			ctxRef.current?.close();
			ctxRef.current = null;
		};
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [config.sessionId, config.bars[0], config.bars[1], _mockManifest, _mockClip, _playable]);

	const color =
		DIMENSION_COLORS[config.dimension as keyof typeof DIMENSION_COLORS] ??
		"#7a9a82";

	return (
		<div
			className="bg-surface-card border border-border rounded-xl overflow-hidden mt-3"
			onClick={onExpand}
		>
			{loadState === "loading" && (
				<div className="h-10 flex items-center justify-center">
					<div className="w-3.5 h-3.5 rounded-full border-2 border-text-tertiary/50 border-t-transparent animate-spin" />
				</div>
			)}
			{(loadState === "ready" || loadState === "audio_error") &&
				clipSvg &&
				manifest && (
					<div className="px-3 pt-3">
						<div
							style={{
								position: "relative",
								borderRadius: "6px",
								border: `1.5px solid ${color}40`,
								backgroundColor: "white",
								overflow: "hidden",
							}}
						>
							<ClipSvg svg={clipSvg} />
						</div>
						{loadState === "ready" ? (
							<button
								type="button"
								aria-label="Play passage"
								onClick={() => void playerRef.current?.play()}
								className="mt-3 px-3 py-1.5 rounded-md border border-border text-body-sm text-text-primary hover:bg-surface transition-colors"
							>
								Play
							</button>
						) : (
							<span className="mt-3 inline-block text-body-sm text-text-tertiary">
								Audio unavailable
							</span>
						)}
					</div>
				)}
			{loadState === "error" && (
				<div className="p-4 text-body-sm text-text-tertiary">
					couldn't load audio
				</div>
			)}
			<div className="p-4 flex flex-col gap-3.5">
				<div className="flex items-center gap-1.5 shrink-0">
					<span
						className="w-1.5 h-1.5 rounded-full"
						style={{ backgroundColor: color }}
					/>
					<span className="text-label-sm text-text-tertiary uppercase tracking-wide">
						{config.dimension}
					</span>
				</div>
				<span className="text-body-xs text-text-tertiary">
					bars {config.bars[0]}–{config.bars[1]}
				</span>
				<p className="text-body-sm text-text-primary mt-0.5 leading-snug">
					{config.annotation}
				</p>
			</div>
		</div>
	);
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run vitest run src/components/cards/PlayPassageCard.test.tsx
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/components/cards/PlayPassageCard.tsx apps/web/src/components/cards/PlayPassageCard.test.tsx && git commit -m "feat(web): PlayPassageCard renders pre-cropped clip SVG

scoreRenderer.getClip now returns Promise<string>; PlayPassageCard
inserts the SVG via insertAdjacentHTML (matching the codebase
convention). The _mockClip prop type changes from ClipResult to string."
```

---

### Task 5b: ExerciseSetCard — render pre-cropped SVG from new getClip

**Group:** D (parallel with Tasks 4 and 5; depends on Group C)

**Behavior being verified:** `ExerciseSetCard` with a `scoreClip` config renders the SVG returned by `scoreRenderer.getClip` directly. No `SvgClip` component, no `ClipResult` type — the state holds a `string`.

**Interface under test:** The rendered DOM of `<ExerciseSetCard config={...} />` when `config.scoreClip` is present.

**Files:**
- Modify: `apps/web/src/components/cards/ExerciseSetCard.tsx`
- Create: `apps/web/src/components/cards/ExerciseSetCard.test.tsx`

- [ ] **Step 1: Write the failing test**

Create `apps/web/src/components/cards/ExerciseSetCard.test.tsx`:

```typescript
import { render, waitFor } from "@testing-library/react";
import * as React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ExerciseSetConfig } from "../../lib/types";

const mockGetClip = vi.fn();
vi.mock("../../lib/score-renderer", () => ({
	scoreRenderer: {
		getClip: (...args: unknown[]) => mockGetClip(...args),
	},
}));
vi.mock("../../lib/api", () => ({
	api: { exercises: { assign: vi.fn() } },
}));

beforeEach(() => {
	vi.clearAllMocks();
});

describe("ExerciseSetCard", () => {
	it("renders the SVG string returned by scoreRenderer.getClip when scoreClip is present", async () => {
		mockGetClip.mockResolvedValue("<svg data-test='exercise-clip'></svg>");
		const config: ExerciseSetConfig = {
			targetSkill: "Voicing the melody",
			sourcePassage: "bars 5-8",
			scoreClip: {
				pieceId: "chopin.ballades.1",
				bars: [5, 8],
			},
			exercises: [
				{ title: "Slow practice", instruction: "Half tempo, both hands." },
			],
		};
		const { ExerciseSetCard } = await import("./ExerciseSetCard");
		render(React.createElement(ExerciseSetCard, { config }));

		await waitFor(() => {
			expect(mockGetClip).toHaveBeenCalledWith("chopin.ballades.1", 5, 8);
			expect(document.body.innerHTML).toContain("data-test='exercise-clip'");
		});
	});

	it("renders without a clip section when scoreClip is absent", async () => {
		const config: ExerciseSetConfig = {
			targetSkill: "Voicing the melody",
			sourcePassage: "general",
			exercises: [
				{ title: "Slow practice", instruction: "Half tempo, both hands." },
			],
		};
		const { ExerciseSetCard } = await import("./ExerciseSetCard");
		render(React.createElement(ExerciseSetCard, { config }));
		await waitFor(() => {
			expect(document.body.textContent).toContain("Voicing the melody");
		});
		expect(mockGetClip).not.toHaveBeenCalled();
	});
});
```

Look up `ExerciseSetConfig` in `apps/web/src/lib/types.ts`. If the shape requires additional fields beyond what's shown, fill in realistic values (e.g., a `setId` UUID, `exerciseId` on items, `hands` enum). Match the existing shape exactly so TypeScript compilation passes.

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run vitest run src/components/cards/ExerciseSetCard.test.tsx
```

Expected: FAIL — the existing `ExerciseSetCard` stores `scoreClip` as `ClipResult` and renders via `<SvgClip svgMarkup={...} />`. The test's `data-test='exercise-clip'` attribute never reaches the DOM because `scoreClip.svg` of a `string`-resolved getClip is `undefined` under the existing type assumptions.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `apps/web/src/components/cards/ExerciseSetCard.tsx`:

1. REPLACE the top imports (lines 1-9) with:

```typescript
import { ArrowsOut, CaretDown } from "@phosphor-icons/react";
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { scoreRenderer } from "../../lib/score-renderer";
import { api } from "../../lib/api";
import { handsLabel } from "../../lib/exercise-utils";
import type { ExerciseSetConfig } from "../../lib/types";
import { useArtifactStore } from "../../stores/artifact";
```

(Removes `ClipResult` import and `SvgClip` import.)

2. ADD a `ClipSvg` helper function immediately after the imports, before `ExerciseSetCardProps`:

```typescript
function ClipSvg({ svg }: { svg: string }) {
	const ref = useRef<HTMLDivElement>(null);
	useLayoutEffect(() => {
		if (!ref.current) return;
		ref.current.textContent = "";
		// biome-ignore lint/security/noDomManipulation: controlled SVG from Verovio WASM
		ref.current.insertAdjacentHTML("afterbegin", svg);
		const svgEl = ref.current.querySelector("svg");
		if (svgEl) {
			svgEl.setAttribute("width", "100%");
			svgEl.removeAttribute("height");
			(svgEl as SVGElement).style.display = "block";
		}
	}, [svg]);
	return <div ref={ref} className="[&>svg]:w-full [&>svg]:block" />;
}
```

3. CHANGE the `scoreClip` state type (line 148) from `useState<ClipResult | null>(null)` to `useState<string | null>(null)`. Update the `.then((r) => { if (!cancelled) setScoreClip(r); })` callback shape stays the same — `r` is now the SVG string directly.

4. REPLACE the `<SvgClip ... />` JSX (lines 163-171) with:

```typescript
{config.scoreClip && scoreClip && (
	<div className="border-b border-border/60 bg-white">
		<ClipSvg svg={scoreClip} />
	</div>
)}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run vitest run src/components/cards/ExerciseSetCard.test.tsx
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/components/cards/ExerciseSetCard.tsx apps/web/src/components/cards/ExerciseSetCard.test.tsx && git commit -m "feat(web): ExerciseSetCard renders pre-cropped clip SVG

scoreClip state is now string (was ClipResult); SVG rendered via the
local ClipSvg helper using the codebase's insertAdjacentHTML pattern."
```

---

### Task 6: Delete dead helpers — SvgClip, SvgClipBBox, sandbox ApproachesComparison

**Group:** E (depends on Group D)

**Behavior being verified:** The deleted modules no longer exist in the codebase, and the sandbox route still loads without import errors.

**Interface under test:** Filesystem state + module-resolution at import time.

**Files:**
- Delete: `apps/web/src/components/SvgClip.tsx`
- Delete: `apps/web/src/components/SvgClipBBox.tsx`
- Modify: `apps/web/src/routes/app.sandbox.tsx`
- Create: `apps/web/src/components/svg-clip-deleted.test.ts`

- [ ] **Step 1: Write the failing test**

Create `apps/web/src/components/svg-clip-deleted.test.ts` with:

```typescript
// Regression test: dead SvgClip helpers are deleted and the sandbox
// route module loads without dangling imports.
import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

describe("SvgClip deletion", () => {
	it("SvgClip.tsx and SvgClipBBox.tsx files are deleted from the components dir", () => {
		const root = resolve(__dirname, "..", "..");
		expect(existsSync(`${root}/src/components/SvgClip.tsx`)).toBe(false);
		expect(existsSync(`${root}/src/components/SvgClipBBox.tsx`)).toBe(false);
	});

	it("app.sandbox route module loads without dangling imports", async () => {
		await expect(import("../routes/app.sandbox")).resolves.toBeDefined();
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run vitest run src/components/svg-clip-deleted.test.ts
```

Expected: FAIL — file-existence assertions fail (`SvgClip.tsx` and `SvgClipBBox.tsx` still exist).

- [ ] **Step 3: Implement the minimum to make the test pass**

1. Delete the two component files:

```bash
rm apps/web/src/components/SvgClip.tsx apps/web/src/components/SvgClipBBox.tsx
```

2. In `apps/web/src/routes/app.sandbox.tsx`, perform these edits:

   - DELETE the imports for `SvgClip` and `SvgClipBBox` (around lines 6-7):
     ```typescript
     import { SvgClip } from "../components/SvgClip";
     import { SvgClipBBox } from "../components/SvgClipBBox";
     ```

   - DELETE the `import type { ClipResult } from "../lib/score-renderer";` line (around line 10 — the `ClipResult` type was removed in Task 3).

   - REPLACE the `MOCK_CLIP: ClipResult = { svg: ..., startMeasureId: ..., endMeasureId: ... }` fixture (around line 179) with `const MOCK_CLIP_SVG: string = "<svg>...</svg>";` — copy the inner `svg` field's value as the new string constant. Rename to `MOCK_CLIP_SVG` to surface every call site.

   - UPDATE every `_mockClip={MOCK_CLIP}` JSX usage in the file (six occurrences at approximately lines 1137, 1386, 1397, 1408, 1419, 1430, 1441 — verify exact lines with grep) to `_mockClip={MOCK_CLIP_SVG}`. Grep first: `grep -n "MOCK_CLIP" apps/web/src/routes/app.sandbox.tsx`. Every match must be updated to reference the new string constant.

   - DELETE the entire `ApproachRowProps` interface (around lines 539-542) and `ApproachRow` function (around lines 544-558).
   - DELETE the entire `ApproachA` function (around lines 560-596).
   - DELETE the entire `ApproachB` function (around lines 598-634).
   - DELETE the entire `ApproachWorkerMethod` function (around lines 636-668).
   - DELETE the entire `ApproachesComparison` function (around lines 670-743).
   - DELETE the JSX call site `<ApproachesComparison ... />` (around line 1178).
   - In `ScoreClipPanel` (around line 993), change `const [clip, setClip] = useState<ClipResult | null>(null);` to `const [svg, setSvg] = useState<string | null>(null);` and update the `.then((r) => { if (!cancelled) setClip(r); })` callback to `.then((s) => { if (!cancelled) setSvg(s); })`. Replace the `<SvgClip svgMarkup={clip.svg} startMeasureId={...} endMeasureId={...} />` JSX (around line 1026) with a `<ClipSvg svg={svg} />` invocation, where `ClipSvg` is a local helper component identical to the one defined in Task 4 (copy the same 14-line component into `app.sandbox.tsx`, since `ClipSvg` is not yet a shared export). Update the surrounding render logic to use `svg` (string) instead of `clip` (object).

3. KEEP: `ScoreGeometryProbe`, `SvgPanel`, `ScoreResizePanel`, `ScoreClipPanel` (refactored), `ScoreClipsSection`, all fixture-card sections (ScoreHighlight, PlayPassage, ExerciseSet, SegmentLoop, KeyboardGuide).

4. After edits, type-check to surface any leftover usages:

```bash
cd apps/web && bun run tsc --noEmit
```

Expected: zero errors. If any errors surface, address each (most likely candidates: leftover references to `ClipResult`, `SvgClip`, or removed approach functions).

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run vitest run src/components/svg-clip-deleted.test.ts && bun run vitest run
```

Expected: PASS (the deletion test + the full suite).

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/components/svg-clip-deleted.test.ts apps/web/src/routes/app.sandbox.tsx && git rm apps/web/src/components/SvgClip.tsx apps/web/src/components/SvgClipBBox.tsx && git commit -m "chore(web): delete dead SvgClip helpers and sandbox ApproachesComparison

SvgClip (geometrically broken Approach A) and SvgClipBBox (now unused
after Tasks 4-5 wire pre-cropped SVG from tk.select) are deleted.
Sandbox ApproachesComparison is removed along with ApproachA, ApproachB,
ApproachWorkerMethod, and ApproachRow; the ScoreGeometryProbe (Phase 2
IR spike) stays."
```

---

## Notes for the build agent

- **Run all worker/renderer tests after every task**, not just the new test, to catch silent breakage. Command: `cd apps/web && bun run vitest run`.
- **Tasks 1 and 2 both modify `score-worker.ts`.** Task 1 must merge before Task 2 starts.
- **Task 3 changes `getClip`'s return type.** Tasks 4 and 5 cannot start until Task 3's commit lands or their TypeScript compilation will fail.
- **Tasks 4 and 5 can run in parallel** — different files, no shared state.
- **Task 2 has a manual validation step** before code changes. Do NOT skip it. If `tk.select` produces broken output, halt and revise the plan per the explicit fallback instructions in Task 2.
- After Task 6, the renderer's public API surface is `{ getFull, getClip }`. `ClipResult`, `getClipMethod`, `SvgClip`, and `SvgClipBBox` are gone. The worker's message types are `render_clip` (no `method` field) and `render_full` only.

---

## Challenge Review

### CEO Pass

#### Premise

Right problem, right now. Five defects are verified live (not speculative): the silent measure-index swallow, the geometrically broken Approach A shipping to users, the dead MXL code path, the unusably slow MEI path, and the serial clip loading. Each is confirmed by code inspection. Without this pass the user-visible clips are either wrong (truncated by ~50% of vertical context) or visibly stagger in. This is a correctness release, not a speculative improvement.

The framing is also sound: "make the existing API correct without changing the API" is a much narrower scope than the full Phase 2 IR rebuild, and the two do not conflict. There is no simpler alternative; the defects live in distinct layers (worker, renderer, consumer) and each requires its own fix.

#### Scope

The plan tightly matches the spec. Nothing is added beyond the five stated defects. The correct things are deferred: score IR, cursor module, annotation off-by-2, iOS, server-side SVG cache.

One item is under-scoped relative to the actual code: **`ExerciseSetCard.tsx` is not mentioned in the plan but imports `ClipResult` from `score-renderer` and renders via `SvgClip`** (verified at lines 3–5, 148, 165 of `ExerciseSetCard.tsx`). Task 6's cleanup deletes `SvgClip.tsx` and removes `ClipResult` from `score-renderer`. After those deletions, `ExerciseSetCard.tsx` will have dangling imports that fail the TypeScript build. Task 6 tells the build agent to run `tsc --noEmit` to surface errors, which will catch this — but the plan does not tell the agent *how* to fix `ExerciseSetCard`, so the agent will be left improvising on a file it was not briefed on.

#### Twelve-Month Alignment

```
CURRENT STATE                      THIS PLAN                       12-MONTH IDEAL
Clips are geometrically wrong;  →  Canonical clip path via      →  Score IR layer (Phase 2)
dead code masquerades as           tk.select; silent errors          makes bar coords explicit
options; card load staggers        explicit; parallel loading;       everywhere; no client-side
                                   dead code deleted                 cropping at all
```

This plan moves directly toward the ideal. It does not create debt that Phase 2 will need to unwind — the `processRenderClipRequest` surface it introduces is the same shape Phase 2 will keep; Phase 2 will add a richer IR above it, not replace it.

#### Alternatives

The spec documents the alternatives (A through E) and explains why each was eliminated. B (`SvgClipBBox`) is preserved as an explicit fallback with a clear trigger condition. The branching checkpoint is the right mechanism.

---

### Engineering Pass

#### Architecture

The data flow after this plan:

```
getClip(pieceId, s, e)
  → ensureBytes (fetch + cache)
  → postMessage render_clip {startBar, endBar}
  → worker: processRenderClipRequest
      → renderClipSvgSelect (tk.select + renderToSVG(1))
      → postMessage {svg}
  → pending.resolve(svg: string)          ← was ClipResult object
  → component: ClipSvg (insertAdjacentHTML)
```

The flow is clean. No new external boundaries, no new async state. The worker-side changes are surgical.

**Concern: Task 1's `loadPiece` implementation in the plan initializes `xmlContent: string | null = null` at the module scope of the exported function, then Task 2 removes that field from `CacheEntry` but tells the agent to also update the `loadPiece` code to remove its `xmlContent` variable. Both tasks touch the same function. Because they are sequential (Group A → B) this is safe, but the two-step rewrite increases the risk of a merge artifact where the variable is removed from `CacheEntry` but not from the function body or vice versa. The tsc guard in Task 6 will catch this, but it arrives 4 tasks later.**

**`renderClipSvgSelect` mutates shared toolkit state (Task 1/2 concern).** Reading the existing implementation at lines 100–117 of `score-worker.ts`: `renderClipSvgSelect` calls `tk.setOptions(CLIP_RENDER_OPTS)`, then `tk.select({...})`, then `tk.renderToSVG(1)`, then `tk.select({})`, then `tk.setOptions(VEROVIO_OPTS)`. The worker is single-threaded, so this mutation is safe within one message handler. But if a `render_full` message is queued while a `render_clip` is executing — impossible in the current sequential onmessage handler — the options would be wrong. Confirmed: not a bug today because the handler is not re-entrant. Safe.

#### Module Depth

| Module | Interface | Impl | Verdict |
|--------|-----------|------|---------|
| `score-worker.ts` (after changes) | 3 exports: `loadPiece`, `processRenderClipRequest`, `renderFullSvg` (+ `renderClipSvgSelect` indirectly) | ~200 LOC, hides WASM lifecycle, ZIP parsing, timemap → index join, error classification | DEEP |
| `score-renderer.ts` (after changes) | 2 public methods: `getFull`, `getClip` | Hides Worker lifecycle, request correlation, bytes cache, sentPieceIds logic | DEEP |
| `ScoreHighlightCard.tsx` + `PlayPassageCard.tsx` | Component props | Simple consumers; complexity is in the deep modules above | SHALLOW by design — spec explicitly calls this out |
| `ClipSvg` (inline helper, Task 4/5/6) | 1 function, 14 LOC | 14 LOC impl | SHALLOW — acceptable given scope; see note below |

`ClipSvg` is duplicated three times (Tasks 4, 5, 6). The plan acknowledges this explicitly ("copy the same 14-line component... since `ClipSvg` is not yet a shared export"). At 14 LOC the duplication is tolerable within a single phase, but it means Task 6's edit of `app.sandbox.tsx` introduces a third copy rather than importing from either card. If any of the three copies diverges (e.g., a Phase 2 engineer fixes a bug in one), it produces silent split behavior. This is a known tradeoff; it is the right call given Phase 2 will consolidate.

#### Code Quality

**Task 2 branching checkpoint: is it well-defined enough for a build agent?**

The checkpoint requires the agent to:
1. Start a dev server in a separate terminal (`just dev-light`).
2. Drive Playwright via MCP to the sandbox.
3. Use `browser_evaluate` to count `g.measure` elements in "Approach C's" section.
4. Repeat for two other bar ranges by temporarily editing the hardcoded values in the sandbox.

Step 4 is underspecified. The agent is told to "temporarily edit the sandbox's hardcoded bar range to `[1, 4]`" — but the plan does not specify which line or which prop to change. `ApproachesComparison` is called at line 1178 of `app.sandbox.tsx` with `startBar={135} endBar={136}`. The agent will need to find and edit that JSX, then revert it, twice. This is a file edit during the checkpoint — that's legitimate for a build agent with Edit access, but the checkpoint instructions omit the file and line so the agent must infer it.

More critically: **the checkpoint instructs the agent to count `g.measure` elements — but Verovio's SVG class for measures may not be `g.measure`**. Looking at the existing `SvgClip.tsx` code, the cropping logic uses `svgEl.querySelector('[id="..."]')` and `.closest('.system')`. The Verovio SVG structure uses class names like `system`, but the measure containers are typically `<g id="m-1">` or similar, not `<g class="measure">`. A count of `g.measure` may return 0 even on correct output, falsely triggering the halt condition. The checkpoint needs to use the actual Verovio element selector — likely `[id^="m-"]` or checking the rendered bar IDs from the measure index. **This is a BLOCKER on the checkpoint's correctness.**

**Task 3: the existing `score-renderer.test.ts` uses `MockWorker.postMessage` which always returns `{ svg: "<svg>mock</svg>" }` regardless of message type.** After Task 3, `getClip` resolves with a `string`. The test at line 33–38 asserts `expect(svg).toBe("<svg>mock</svg>")`. Today this test FAILS (because `getClip` returns `Promise<ClipResult>` = an object). The plan correctly identifies this test as the "failing test" for Task 3 — it does not write a new test, it relies on the existing one which currently fails. This is the right approach. Confirmed the existing test will fail pre-implementation and pass post-implementation.

**Task 5: existing `PlayPassageCard.test.tsx` passes `mockGetClip.mockResolvedValue({ svg: "<svg></svg>", startMeasureId: null, endMeasureId: null })` (an object) at lines 79–83, 98–101, 123–126.** After Task 5, `getClip` returns `Promise<string>`, so passing an object won't work. The plan's Task 5 instructions say: "If the existing `PlayPassageCard.test.tsx` contains tests that construct `_mockClip` as `ClipResult` (object with `svg`, `startMeasureId`, `endMeasureId`), update those tests to pass the SVG string directly." The existing tests also call `mockGetClip.mockResolvedValue(...)` — the card's existing tests do not use `_mockClip` at all; they exercise the live `scoreRenderer.getClip` path via the mock. Those mock return values will need to change from `{ svg: ..., ... }` to just the string `"<svg></svg>"`. The plan's Step 3 instruction is vague here; the agent may miss these three existing mock call sites.

**`ExerciseSetCard.tsx` will break after Task 6 (BLOCKER).** Confirmed by reading the file: it imports `ClipResult` from `score-renderer` (line 3) and `SvgClip` from `../SvgClip` (line 5), uses `ClipResult` as state type (line 148), and renders `<SvgClip ... />` (line 165). Task 6 deletes `SvgClip.tsx` and removes `ClipResult` from `score-renderer`. The `tsc --noEmit` check in Task 6, Step 3 will surface both errors, but Task 6's implementation step does not mention `ExerciseSetCard` or tell the agent how to update it. The agent will encounter two TypeScript errors on an un-briefed file and must improvise. The fix is straightforward (convert `scoreClip` state from `ClipResult` to `string`, update render to `<ClipSvg svg={scoreClip} />`), but the plan must explicitly include it.

#### Test Philosophy Audit

**Task 1 — `loadPiece` regression test**

The test uses a `FakeToolkitClass` that throws from `renderToTimemap` and passes it directly as `bindings.ToolkitClass`. This is testing the exported `loadPiece` function through its public interface with a controlled mock at the external boundary (the Verovio WASM). The mock is an external dependency (WASM), not an internal collaborator. PASS.

One concern: the test does `await import("./score-worker")` inside the test body. If vitest's module cache is not reset between tests (no `vi.resetModules()` in the test file's `beforeEach`), the newly exported `loadPiece` will be resolved on first import and subsequent tests may get the pre-Task-1 module if run in the same vitest process. The existing `score-worker.test.ts` does NOT call `vi.resetModules()` in `beforeEach`. The Task 1 test appends a new `describe` block; the existing tests import with `await import("./score-worker")` too. This should be fine in vitest since all `import()` calls within the same test file share the same module cache, and `vi.clearAllMocks()` is called. Low risk.

**Task 2 — `processRenderClipRequest` tests**

The test calls `processRenderClipRequest(tk, fakeMeasures, startBar, endBar)` and asserts on `mockSelect` and `mockRenderToSVG`. The function is a thin pass-through to `renderClipSvgSelect`, so the test is effectively testing the select-and-render behavior through the public function. This is acceptable given `renderClipSvgSelect` is also exported and tested. However: the test REPLACES the existing `describe("renderClipSvg", ...)` block (lines 31–73), which removes 5 existing tests for `renderClipSvg`. The plan instructs the agent to delete those tests as part of removing `renderClipSvg`. But `renderClipSvg` is only deleted in Task 2 Step 3, after the tests are replaced in Task 2 Step 1. Between Step 1 and Step 3, the test file references `processRenderClipRequest` (not yet exported) and no longer references `renderClipSvg` (still present). The TDD step 2 will fail with "not exported" as intended. Step 3 adds the export and deletes `renderClipSvg`. This is correct per the vertical-slice discipline.

**Task 4 — parallel-load test**

The test captures resolvers into an array and then asserts `mockGetClip).toHaveBeenCalledTimes(3)` before resolving any of them. This genuinely verifies parallelism: if the implementation uses `for...await`, only the first `getClip` call is issued before the loop awaits, so the test times out at `toHaveBeenCalledTimes(3)`. But there is a subtle weakness: **the `waitFor` that asserts `toHaveBeenCalledTimes(3)` has a default timeout (usually 1000ms in testing-library). If the component's `useEffect` fires synchronously in jsdom and issues all three calls in the same microtask queue, the assertion passes regardless.** However: `Promise.all` is what makes parallelism work, and `for...await` is what makes it fail. The test's mechanism (all resolvers captured before any resolves) is sound — if `for...await` is used, the second `mockGetClip` is never called before the first resolver resolves, so the array has only 1 entry when the `waitFor` first checks. The test will correctly fail on `for...await` implementation. PASS on test validity.

The test then checks the DOM for three specific SVG strings inserted via `insertAdjacentHTML`. The jsdom environment does not parse SVG innerHTML via `insertAdjacentHTML` by default — jsdom has historically had incomplete SVG support. `document.querySelector("svg[data-bars='1-4']")` may return `null` even when the correct SVG string is injected, because jsdom may not parse `<svg ...>` in `insertAdjacentHTML`. This is a known jsdom limitation. The plan does not account for this. **If jsdom silently drops the SVG elements, the test will fail after implementation, not before, which is the wrong failure direction.** This is a RISK.

**Task 5 — PlayPassageCard SVG rendering test**

Same jsdom concern: `document.querySelector("svg[data-test='passage-clip']")` after `insertAdjacentHTML` may return `null` in jsdom. Same risk as Task 4.

Additionally, the existing `PlayPassageCard` tests (which pass `_mockClip` as a `ClipResult` object) will become TypeScript errors after Task 5 changes the prop type to `string`. The plan instructs the agent to update them, but does not give specific instructions — only "if the existing tests construct `_mockClip` as `ClipResult`... update those tests to pass the SVG string directly." The agent needs to find and update the `mockGetClip.mockResolvedValue(...)` calls in the existing test file (3 occurrences, lines 79–83, 98–101, 123–126). This is in the mock, not in `_mockClip`. The plan's framing ("construct `_mockClip` as `ClipResult`") does not match how the existing tests actually work — they use `mockGetClip`, not `_mockClip`. The agent may miss this.

**Task 6 — file-existence test**

`existsSync` tests verify file deletion. This is a filesystem state assertion, not a behavior assertion — it's a shape test. It would pass even if the files are deleted but all their consumers are broken. However, the second test ("sandbox route module loads without dangling imports") does verify behavior through dynamic import resolution, which is meaningful. RISK (low severity): the `existsSync` test is a shape test but it is the appropriate tool for verifying deletion; the combination of the two tests is sufficient.

**Task 6 — the `svg-clip-deleted.test.ts` test file is created as a new file.** The plan's Step 5 commit adds the new file but also uses `git rm` to remove the two components — and uses `git add` for `svg-clip-deleted.test.ts`. This is correct. However, the `__dirname` resolution in the test file is `resolve(__dirname, "..", "..")`. From `apps/web/src/components/svg-clip-deleted.test.ts`, `__dirname` is `apps/web/src/components`, `..` is `apps/web/src`, `../..` is `apps/web`. Then `${root}/src/components/SvgClip.tsx` is `apps/web/src/components/SvgClip.tsx`. That is the correct path. SAFE.

#### Vertical Slice Audit

All six tasks follow one-test → one-impl → one-commit. No horizontal slicing detected. The one exception is Task 3, which reuses an existing failing test rather than writing a new one — this is explicitly called out and is the correct approach (the test already encodes the right behavior; it was written against a spec that is now being implemented).

#### Test Coverage Gaps

```
score-worker.ts
  loadPiece()
    ├── [TESTED ★★]  buildMeasureIndex throws → returns "failed" (Task 1)
    ├── [GAP]        loadZipDataBuffer throws AND extractXmlFromMxl throws → returns "failed"
    └── [GAP]        isZip=false, TextDecoder fails → returns "failed"

  processRenderClipRequest()
    ├── [TESTED ★★]  happy path: select called, renderToSVG(1) called (Task 2)
    └── [TESTED ★★]  out-of-range bar: select not called, page-1 fallback (Task 2)

score-renderer.ts
  getClip() → Promise<string>
    ├── [TESTED ★★]  happy path resolves with svg string (existing, Task 3)
    ├── [TESTED ★★]  worker error → rejects (existing)
    ├── [TESTED ★★]  api.getData fails → rejects (existing)
    └── [GAP]        worker returns neither svg nor error → rejects with "Worker returned no svg..."

ScoreHighlightCard.tsx
  ├── [TESTED ★]     getClip rejects → error state (metadata still visible) (Task 4 test 1)
  ├── [TESTED ★★]    N clips issued in parallel before any resolves (Task 4 test 2)
  ├── [TESTED ★★]    SVG strings inserted into DOM after resolution (Task 4 test 2)*
  └── [GAP]          some highlights resolve, one rejects → what does the card show?

PlayPassageCard.tsx
  ├── [TESTED ★★]    SVG string inserted into DOM via _mockClip (Task 5)
  ├── [TESTED ★★]    annotation/dimension/bar range visible (existing)
  ├── [TESTED ★★]    play button triggers player.play() (existing)
  ├── [TESTED ★★]    manifest fetch fails → error state (existing)
  └── [TESTED ★★]    audio load fails → audio_error state (existing)
```

*The `[GAP]` on "some highlights resolve, one rejects" maps to the spec's statement: "the current behavior — single rejection sets the card to error — is preserved by checking for any rejection in the resolved array." But the plan uses `Promise.all`, which rejects on the first rejection. The spec text says "we `Promise.allSettled` internally" but the Task 4 implementation code uses `Promise.all` with a `.catch`. If any clip fails, all clips fail. The spec's claim that "the others still resolve" is false under `Promise.all`. The gap is not in the test coverage per se — it's a spec/implementation mismatch that the tests don't catch.

#### Failure Modes

- **Task 2 checkpoint halt:** if `tk.select` fails validation, the plan provides a clear fallback (use Approach B). The halt is well-specified at the task level. The only weakness is the selector ambiguity noted above.
- **Mid-task failure on Task 2:** if the agent deletes `renderClipSvg`, `renderClipSvgMei`, `renderClipSvgMxl` but fails before updating the message handler, the worker would try to call the deleted default branch and throw a ReferenceError on every `render_clip` message. This would be caught by the outer `try/catch` in the message handler (line 558 of current `score-worker.ts`) and returned as an `error` response. Not silent. Acceptable.
- **`ExerciseSetCard` broken imports after Task 6:** if the `tsc --noEmit` check passes (it won't — it will surface errors), the app would fail to build. The TypeScript guard will catch it, but the plan does not give the agent instructions for resolution. The agent will stall.

---

### Presumption Inventory

| Assumption | Verdict | Reason |
|------------|---------|--------|
| `loadPiece` is the only place that calls `buildMeasureIndex` | SAFE | Verified by grep: `buildMeasureIndex` appears only inside the `if (typeof window === "undefined")` block's `loadPiece` function. |
| `getClipMethod` has no production callers (only sandbox) | SAFE | Verified: only `ApproachWorkerMethod` in `app.sandbox.tsx` calls it (line 653). No other callers. |
| `xmlContent` field on `CacheEntry` has no consumers after MXL deletion | RISKY | `renderClipSvgMxl` is the only function that reads `xmlContent`. After Task 2 deletes it, no code reads the field. BUT: Task 1 writes `loadPiece` code that still stores `xmlContent` in the entry. Task 2 must both remove the function AND the field. The two-task sequence makes this correct but fragile. |
| Verovio SVG measures are addressable as `g.measure` for the checkpoint count | RISKY | Verovio does not use class="measure" in its SVG output. The measure elements are typically `<g id="m-1">` etc. The checkpoint's count of `g.measure` will return 0 regardless, and the checkpoint will always appear to fail. |
| `Promise.all` preserves the spec's "if one highlight fails, the card goes to error" contract | SAFE | `Promise.all` rejects on first rejection. This matches the current `for...await` behavior (any error sets `error` state). The spec's parenthetical "we `Promise.allSettled` internally" is inaccurate — the code uses `Promise.all` — but the user-visible behavior (error state on any failure) is preserved. |
| jsdom can parse SVG inserted via `insertAdjacentHTML` and make it queryable | VALIDATE | jsdom has known incomplete SVG support. `document.querySelector("svg[data-bars='1-4']")` after `insertAdjacentHTML` with an `<svg>` string may return null. If so, the Task 4 and Task 5 SVG-insertion tests will fail after implementation, breaking the TDD flow. |
| `ExerciseSetCard.tsx` only needs to be updated for `SvgClip`/`ClipResult` removal | RISKY | `ExerciseSetCard` uses both `ClipResult` (as state type) and `SvgClip` (as render component). The plan does not mention this file. Task 6's `tsc --noEmit` will surface the errors but the agent has no instructions for resolution. |
| `MOCK_CLIP: ClipResult` in `app.sandbox.tsx` does not need updating | RISKY | After Task 3 removes `ClipResult` from `score-renderer`, `import type { ClipResult }` at line 10 of `app.sandbox.tsx` becomes a dangling type import. Task 6 lists removing `import type { ClipResult } from "../lib/score-renderer"` — this is in the task — but `MOCK_CLIP` is typed as `ClipResult` and is passed as `_mockClip` to `PlayPassageCard` in six places. After Task 5 changes `_mockClip` to `string`, the `MOCK_CLIP` fixture (which is a `ClipResult` object today) must become the SVG string directly (`MOCK_SCORE_SVG`). Task 6 does not mention updating the `MOCK_CLIP` fixture or the six `_mockClip={MOCK_CLIP}` call sites. |

---

### Summary

**[BLOCKER] count: 3**
**[RISK]    count: 4**
**[QUESTION] count: 0**

**[BLOCKER] (confidence: 9/10)** — `ExerciseSetCard.tsx` is not in the plan's scope but imports `ClipResult` and renders via `SvgClip`. Task 6 deletes both. `tsc --noEmit` will surface two errors on an un-briefed file with no resolution instructions. Add `ExerciseSetCard.tsx` to Task 6's implementation step: change `scoreClip` state from `ClipResult | null` to `string | null`, update `.then((r) => { setScoreClip(r); })` to `.then((s) => { setScoreClip(s); })`, replace `<SvgClip svgMarkup={scoreClip.svg} startMeasureId={...} endMeasureId={...} />` with `<ClipSvg svg={scoreClip} />` (using the same local helper pattern as Task 4/5/6). Remove the `catch (() => {})` silent swallow while there — it hides score-load failures in the card.

**[BLOCKER] (confidence: 9/10)** — `MOCK_CLIP` in `app.sandbox.tsx` is typed `ClipResult` and passed as `_mockClip` in six `PlayPassageCard` usages. After Task 5 changes `_mockClip` to `string` and Task 6 removes `ClipResult`, all six call sites break. Task 6 must also: (a) replace `const MOCK_CLIP: ClipResult = { svg: MOCK_SCORE_SVG, ... }` with `const MOCK_CLIP = MOCK_SCORE_SVG;` (or inline the constant), and (b) no changes are needed to the six `_mockClip={MOCK_CLIP}` call sites since `MOCK_CLIP` now resolves to the string. Add this to Task 6's implementation step.

**[BLOCKER] (confidence: 8/10)** — Task 2's branching checkpoint uses `g.measure` as the CSS selector to count measure elements in Verovio's SVG output. Verovio does not produce `<g class="measure">` elements; measure containers are `<g id="m-{n}">` or similar. The count will return 0 on correct and incorrect output alike, meaning the checkpoint cannot distinguish a working `tk.select` from a broken one. Rewrite the checkpoint assertion to use an actual Verovio measure selector: count elements matching `[id^="m-"]` or `g[id^="measure-"]` (verify the actual prefix by inspecting one Approach C SVG in the sandbox's current output before writing the checkpoint). Alternatively, measure the bounding height of the rendered SVG viewBox: a correct 2-bar clip has a much smaller viewBox height than a full-page render.

**[RISK] (confidence: 7/10)** — jsdom may not parse `<svg>` elements inserted via `insertAdjacentHTML`, causing `document.querySelector("svg[data-bars='1-4']")` to return null in Task 4 and Task 5 tests even after correct implementation. This would cause the tests to fail after implementation, breaking TDD. Mitigation: before writing those assertions, verify jsdom SVG parsing by running a minimal `insertAdjacentHTML` + `querySelector("svg")` test in the existing test environment. If jsdom does not support it, assert on the `ref.current.innerHTML` string instead of querying the DOM.

**[RISK] (confidence: 7/10)** — The existing `PlayPassageCard.test.tsx` has three `mockGetClip.mockResolvedValue({ svg: ..., startMeasureId: null, endMeasureId: null })` calls (lines 79–83, 98–101, 123–126) that pass a `ClipResult` object. After Task 5 changes `getClip` to return `Promise<string>`, these mocks will be returning an object where the component expects a string, causing the existing tests to fail for the wrong reason. Task 5's instructions say "update tests that construct `_mockClip` as `ClipResult`" — but the relevant mock return values are on `mockGetClip`, not on `_mockClip`. The agent may miss this. Add explicit instructions: "Also update all three `mockGetClip.mockResolvedValue({ svg: ... })` calls to `mockGetClip.mockResolvedValue('<svg></svg>')` (plain string)."

**[RISK] (confidence: 6/10)** — The spec says "the others still resolve (we `Promise.allSettled` internally)" but Task 4's implementation uses `Promise.all`, which fails all clips on the first rejection. This is a spec/impl inconsistency. The user-visible behavior (card goes to error on any failure) is consistent between old and new, so this is not a regression — but the spec's parenthetical is misleading and could cause a future engineer to "fix" `Promise.all` to `Promise.allSettled` thinking it matches the spec. Correct the spec's parenthetical or add a comment in Task 4's implementation: `// Promise.all: any clip failure → full error state (intentional)`.

**[RISK] (confidence: 6/10)** — `ClipSvg` is duplicated in `ScoreHighlightCard.tsx`, `PlayPassageCard.tsx`, and `app.sandbox.tsx` (three copies). Task 6's instructions for `app.sandbox.tsx` say "copy the same 14-line component." If the three copies diverge, Phase 2 will have a subtle split. The risk is low given the simplicity of the component, but the plan should note that Phase 2's first task should consolidate `ClipSvg` into a shared export in `apps/web/src/components/ClipSvg.tsx`.

---

VERDICT: NEEDS_REWORK — Three blockers must be resolved before execution: (1) add `ExerciseSetCard.tsx` to Task 6 scope with explicit instructions, (2) add `MOCK_CLIP` fixture update and `_mockClip` call-site type change to Task 6 scope, (3) fix the Task 2 branching checkpoint selector from `g.measure` to the actual Verovio measure element selector.
