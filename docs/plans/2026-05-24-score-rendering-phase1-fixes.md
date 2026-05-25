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
Group D (depends on C, parallel):     Task 4, Task 5
Group E (depends on D, 1 task):       Task 6
```

Tasks 4 and 5 touch different files (`ScoreHighlightCard.tsx` vs `PlayPassageCard.tsx`) and can run in parallel.

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

Validate `tk.select` produces correct clips against the live sandbox before writing code:

1. Start dev server: `just dev-light` (separate terminal).
2. Drive Playwright (via MCP) to `http://localhost:3000/app/sandbox`.
3. Locate the "Approaches Comparison" section ("Rendering Approaches — Bars 135–136").
4. Use `browser_evaluate` to count Approach C's `g.measure` elements. Expected: exactly 2 (bars 135–136).
5. Repeat by temporarily editing the sandbox's hardcoded bar range to `[1, 4]` (expect 4 measures) and `[200, 208]` (expect 9 measures). The sandbox has `<ApproachesComparison pieceId="chopin.ballades.1" startBar={135} endBar={136} />` at around line 1178.

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

		await waitFor(() => {
			expect(document.querySelector("svg[data-bars='1-4']")).toBeTruthy();
			expect(document.querySelector("svg[data-bars='5-8']")).toBeTruthy();
			expect(document.querySelector("svg[data-bars='9-12']")).toBeTruthy();
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

		await waitFor(() => {
			expect(
				document.querySelector("svg[data-test='passage-clip']"),
			).toBeTruthy();
		});
	});
});
```

If the existing `PlayPassageCard.test.tsx` contains tests that construct `_mockClip` as `ClipResult` (object with `svg`, `startMeasureId`, `endMeasureId`), update those tests to pass the SVG string directly. Read the existing file before editing to determine which tests need updates.

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

   - DELETE any `import type { ClipResult } from "../lib/score-renderer";` (the `ClipResult` type was removed in Task 3).

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
