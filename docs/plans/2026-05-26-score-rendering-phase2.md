# Score Rendering Phase 2 — ScoreIR + Playback Cursor Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** A pianist watching a rendered score in the web app sees a sub-50ms-latency cursor that tracks playback position note-by-note.
**Spec:** docs/specs/2026-05-26-score-rendering-phase2-design.md
**Style:** Follow CLAUDE.md — explicit exception handling, no silent fallbacks, bun for JS, no emojis.

---

## Task Groups

- **Group 0 (spike gate, sequential first):** Task 0
- **Group A (parallel, depends on Group 0 passing):** Task 1, Task 2
- **Group B (sequential, depends on Group A):** Task 3
- **Group C (parallel, depends on Group B):** Task 4, Task 5
- **Group D (sequential, depends on Group C):** Task 6
- **Group E (parallel, depends on Group D):** Task 7, Task 8
- **Group F (sequential, depends on Group E):** Task 9

---

## Task 0: Prerequisite Spike — Measure Ballade load() wall-clock

**Group:** 0 (sequential gate — if this task fails the >200ms threshold, halt the build and revise the spec toward lazy-IR before continuing)

**Behavior being verified:** `load()` on the Ballade fixture (56.7KB MXL, the largest fixture) completes within 200ms wall-clock, confirming that eager IR build at load time is a viable contract.

**Interface under test:** `loadPiece()` from `apps/web/src/lib/score-worker.ts` exercised directly (same pattern as the existing integration test).

**Files:**
- Modify: `apps/web/src/lib/score-worker.integration.test.ts`

- [ ] **Step 1: Write the failing test**

Add the following describe block to `apps/web/src/lib/score-worker.integration.test.ts`, after the existing describe block:

```typescript
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const BALLADE_FIXTURE_PATH = resolve(
  dirname(fileURLToPath(import.meta.url)),
  "../../public/scores/chopin-ballade-op23-no1.mxl",
);

describe("load() wall-clock spike — Ballade fixture", () => {
  it("completes loadPiece for the Ballade within 200ms", async () => {
    const esm = (await import("verovio/esm")) as any;
    const wasm = (await import("verovio/wasm")) as any;
    const VerovioToolkit = esm.VerovioToolkit ?? esm.default?.VerovioToolkit;
    const VerovioModule = wasm.default ?? wasm;
    const mod = await VerovioModule();

    const bytes = readFileSync(BALLADE_FIXTURE_PATH);
    const arrayBuf = new ArrayBuffer(bytes.byteLength);
    new Uint8Array(arrayBuf).set(bytes);

    const { loadPiece } = await import("./score-worker");

    const t0 = Date.now();
    const entry = await loadPiece(
      arrayBuf,
      { module: mod, ToolkitClass: VerovioToolkit as any },
      "chopin-ballade-op23-no1",
    );
    const elapsed = Date.now() - t0;

    expect(entry).not.toBe("failed");
    // GATE: if this assertion fails, the eager-IR contract is not viable.
    // Halt the build and revise the spec toward lazy IR before proceeding.
    expect(elapsed).toBeLessThan(200);
  }, 30_000);
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/web && bun run test src/lib/score-worker.integration.test.ts
```

Expected: FAIL — `loadPiece` currently returns a `CacheEntry` without rendering all pages or building an IR, so the new describe block's test may actually pass the timing assertion but will fail once Task 3 changes `loadPiece` to include IR build. For now, the test must at minimum run and confirm the fixture loads under 200ms with the current implementation.

> **Build-agent gate:** If the test runs and `elapsed >= 200` is reported, STOP. Do not proceed to Group A. File a note that the spec must be revised toward lazy-IR and await further instruction.

- [ ] **Step 3: Implement the minimum to make the test pass**

No implementation changes needed for this task — the test exercises the existing `loadPiece`. If the test passes, the timing contract is validated and Group A can begin. If it fails the 200ms threshold, halt.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/web && bun run test src/lib/score-worker.integration.test.ts
```

Expected: PASS — the Ballade loads within 200ms, the spike gate is cleared.

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai && git add apps/web/src/lib/score-worker.integration.test.ts && git commit -m "test(score-worker): spike gate — Ballade load() under 200ms"
```

---

## Task 1: IR Types and parseScoreIR pure function

**Group:** A (parallel with Task 2)

**Behavior being verified:** `parseScoreIR` extracts a structurally valid `ScoreIR` from Verovio-rendered SVG pages: every `NoteIR.bbox.x` and `.y` is a finite number, every `BarIR.noteIds` resolves to a key in `notes`, `qstampStart < qstampEnd` for every bar, and `bars.length` equals the number of supplied measures.

**Interface under test:** `parseScoreIR(pieceId, pageSvgs, measures, verovioVersion, pageWidth)` from `apps/web/src/lib/score-ir.ts`, exercised with real Verovio SVG fixtures produced inline in the test.

**Files:**
- Create: `apps/web/src/lib/score-ir.ts`
- Create: `apps/web/src/lib/score-ir.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/web/src/lib/score-ir.test.ts
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

// Minimal synthetic SVG that exercises the regex paths.
// Two measures, two notes each — one per staff.
const SYNTHETIC_PAGE_SVG = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 2400 800" width="2400" height="800">
  <g class="measure" id="m1">
    <g class="note" id="note-1-1"><use x="100" y="200"/></g>
    <g class="note" id="note-1-2"><use x="200" y="200"/></g>
  </g>
  <g class="measure" id="m2">
    <g class="note" id="note-2-1"><use x="400" y="200"/></g>
    <g class="note" id="note-2-2"><use x="500" y="200"/></g>
  </g>
</svg>`;

const SYNTHETIC_MEASURES = [
  { qstamp: 0, measureOn: "m1" },
  { qstamp: 4, measureOn: "m2" },
];

describe("parseScoreIR — structural invariants", () => {
  it("returns a ScoreIR whose bars.length equals the supplied measures count", async () => {
    const { parseScoreIR } = await import("./score-ir");
    const ir = parseScoreIR(
      "test-piece",
      [SYNTHETIC_PAGE_SVG],
      SYNTHETIC_MEASURES,
      "4.0.0",
      2400,
    );
    expect(ir.bars.length).toBe(SYNTHETIC_MEASURES.length);
  });

  it("every NoteIR.bbox.x and .y is a finite number", async () => {
    const { parseScoreIR } = await import("./score-ir");
    const ir = parseScoreIR(
      "test-piece",
      [SYNTHETIC_PAGE_SVG],
      SYNTHETIC_MEASURES,
      "4.0.0",
      2400,
    );
    for (const note of Object.values(ir.notes)) {
      expect(Number.isFinite(note.bbox.x)).toBe(true);
      expect(Number.isFinite(note.bbox.y)).toBe(true);
      expect(note.bbox.w).toBe(0);
      expect(note.bbox.h).toBe(0);
    }
  });

  it("every BarIR.noteIds resolves to a key in ir.notes", async () => {
    const { parseScoreIR } = await import("./score-ir");
    const ir = parseScoreIR(
      "test-piece",
      [SYNTHETIC_PAGE_SVG],
      SYNTHETIC_MEASURES,
      "4.0.0",
      2400,
    );
    for (const bar of ir.bars) {
      for (const noteId of bar.noteIds) {
        expect(ir.notes[noteId]).toBeDefined();
      }
    }
  });

  it("qstampStart < qstampEnd for every bar with notes", async () => {
    const { parseScoreIR } = await import("./score-ir");
    const ir = parseScoreIR(
      "test-piece",
      [SYNTHETIC_PAGE_SVG],
      SYNTHETIC_MEASURES,
      "4.0.0",
      2400,
    );
    for (const bar of ir.bars) {
      if (bar.noteIds.length > 0) {
        expect(bar.qstampStart).toBeLessThan(bar.qstampEnd);
      }
    }
  });

  it("stores pieceId, verovioVersion, and pageWidth on the IR", async () => {
    const { parseScoreIR } = await import("./score-ir");
    const ir = parseScoreIR(
      "my-piece",
      [SYNTHETIC_PAGE_SVG],
      SYNTHETIC_MEASURES,
      "4.1.0",
      2400,
    );
    expect(ir.pieceId).toBe("my-piece");
    expect(ir.verovioVersion).toBe("4.1.0");
    expect(ir.pageWidth).toBe(2400);
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/web && bun run test src/lib/score-ir.test.ts
```

Expected: FAIL — `Cannot find module './score-ir'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `apps/web/src/lib/score-ir.ts`:

```typescript
// apps/web/src/lib/score-ir.ts

export type Bbox = { x: number; y: number; w: number; h: number };
// NOTE: w and h are always 0. SVG-text parsing has no layout engine.
// Cursor reads only x and y. Click-to-select would need main-thread getBBox.

export type NoteIR = {
  id: string;
  bbox: Bbox;
  qstamp: number;
  staff: 1 | 2;
};

export type BarIR = {
  barNumber: number;
  measureOn: string;
  pageN: number;
  bbox: Bbox;
  noteIds: string[];
  qstampStart: number;
  qstampEnd: number;
};

export type PageIR = {
  pageN: number;
  viewBox: string;
  width: number;
  height: number;
  systemBboxes: Bbox[];
};

export type ScoreIR = {
  pieceId: string;
  verovioVersion: string;
  pageWidth: number;
  pages: PageIR[];
  bars: BarIR[];
  notes: Record<string, NoteIR>;
};

interface MeasureEntry {
  qstamp: number;
  measureOn: string;
}

// Extract the numeric value of a named SVG attribute from a tag attribute string.
// Returns NaN if the attribute is absent or non-numeric.
function extractAttr(attrs: string, name: string): number {
  const m = attrs.match(new RegExp(`${name}="([^"]+)"`));
  return m ? parseFloat(m[1]) : Number.NaN;
}

// Parse the viewBox attribute "minX minY width height" into width and height.
function parseViewBox(attrs: string): { viewBox: string; width: number; height: number } {
  const m = attrs.match(/viewBox="([^"]+)"/);
  const viewBox = m ? m[1] : "0 0 0 0";
  const parts = viewBox.split(/\s+/).map(Number);
  return { viewBox, width: parts[2] ?? 0, height: parts[3] ?? 0 };
}

// Walk SVG text to extract note elements: <g class="note" id="..."><use x="..." y="..."/>
// Returns a map of note id -> {x, y}.
function extractNotePositions(svgText: string): Map<string, { x: number; y: number }> {
  const result = new Map<string, { x: number; y: number }>();
  // Match <g ...class="...note..."... id="..."> followed (soon after) by <use x="..." y="..."/>
  const noteBlockRe = /<g\s+([^>]*class="[^"]*\bnote\b[^"]*"[^>]*)>([\s\S]*?)<\/g>/g;
  for (const blockMatch of svgText.matchAll(noteBlockRe)) {
    const gAttrs = blockMatch[1];
    const inner = blockMatch[2];
    const idMatch = gAttrs.match(/id="([^"]+)"/);
    if (!idMatch) continue;
    const id = idMatch[1];
    const useMatch = inner.match(/<use\s+([^>]*\/?>)/);
    if (!useMatch) continue;
    const useAttrs = useMatch[1];
    const x = extractAttr(useAttrs, "x");
    const y = extractAttr(useAttrs, "y");
    if (Number.isFinite(x) && Number.isFinite(y)) {
      result.set(id, { x, y });
    }
  }
  return result;
}

// Walk SVG text to extract measure element ids in document order.
// Returns an array of { measureId, noteIds[] } where noteIds are the note element
// ids contained within each measure block.
function extractMeasureNoteMap(svgText: string): Array<{ measureId: string; noteIds: string[] }> {
  const result: Array<{ measureId: string; noteIds: string[] }> = [];
  // Match outer measure blocks (non-greedy; may miss deeply nested — acceptable for v1).
  const measureRe = /<g\s+([^>]*class="[^"]*\bmeasure\b[^"]*"[^>]*)>([\s\S]*?)<\/g>/g;
  for (const measureMatch of svgText.matchAll(measureRe)) {
    const gAttrs = measureMatch[1];
    const inner = measureMatch[2];
    const idMatch = gAttrs.match(/id="([^"]+)"/);
    if (!idMatch) continue;
    const measureId = idMatch[1];
    const noteIds: string[] = [];
    const noteIdRe = /<g\s+[^>]*class="[^"]*\bnote\b[^"]*"[^>]*\sid="([^"]+)"/g;
    for (const noteMatch of inner.matchAll(noteIdRe)) {
      noteIds.push(noteMatch[1]);
    }
    result.push({ measureId, noteIds });
  }
  return result;
}

export function parseScoreIR(
  pieceId: string,
  pageSvgs: string[],
  measures: MeasureEntry[],
  verovioVersion: string,
  pageWidth: number,
): ScoreIR {
  const notes: Record<string, NoteIR> = {};
  const pages: PageIR[] = [];
  const bars: BarIR[] = [];

  // Build a lookup from measureOn id -> MeasureEntry index for qstamp resolution.
  const measureByMeasureOn = new Map<string, { qstamp: number; idx: number }>();
  for (let i = 0; i < measures.length; i++) {
    measureByMeasureOn.set(measures[i].measureOn, { qstamp: measures[i].qstamp, idx: i });
  }

  for (let pageIdx = 0; pageIdx < pageSvgs.length; pageIdx++) {
    const pageN = pageIdx + 1;
    const svgText = pageSvgs[pageIdx];

    // Extract page dimensions from the root <svg> tag.
    const svgTagMatch = svgText.match(/<svg\s+([^>]*)>/);
    const svgAttrs = svgTagMatch ? svgTagMatch[1] : "";
    const { viewBox, width, height } = parseViewBox(svgAttrs);

    pages.push({ pageN, viewBox, width, height, systemBboxes: [] });

    // Extract all note positions on this page.
    const notePositions = extractNotePositions(svgText);
    for (const [id, { x, y }] of notePositions) {
      // Staff inference: notes with y > page_midpoint are staff 2, else staff 1.
      const midY = height / 2;
      notes[id] = {
        id,
        bbox: { x, y, w: 0, h: 0 },
        qstamp: 0, // filled in below when we associate notes to bars
        staff: y > midY ? 2 : 1,
      };
    }

    // Extract measure->note mapping for this page.
    const measureNoteMap = extractMeasureNoteMap(svgText);

    for (const { measureId, noteIds } of measureNoteMap) {
      const entry = measureByMeasureOn.get(measureId);
      if (!entry) continue;

      const { qstamp: qstampStart, idx } = entry;
      // qstampEnd: next measure's qstamp, or qstampStart + 4 for the last measure.
      const nextMeasure = measures[idx + 1];
      const qstampEnd = nextMeasure ? nextMeasure.qstamp : qstampStart + 4;

      // Assign qstamp to each note in this bar using the bar's qstampStart as baseline.
      // Intra-bar interpolation is done by the cursor; notes all get qstampStart here.
      for (const noteId of noteIds) {
        if (notes[noteId]) {
          notes[noteId].qstamp = qstampStart;
        }
      }

      // Compute bar bbox from the x-positions of its notes.
      const xs = noteIds.map((id) => notes[id]?.bbox.x ?? 0).filter((x) => x > 0);
      const barX = xs.length > 0 ? Math.min(...xs) : 0;

      bars.push({
        barNumber: idx + 1,
        measureOn: measureId,
        pageN,
        bbox: { x: barX, y: 0, w: 0, h: 0 },
        noteIds,
        qstampStart,
        qstampEnd,
      });
    }
  }

  // Ensure bars are sorted by barNumber ascending.
  bars.sort((a, b) => a.barNumber - b.barNumber);

  return { pieceId, verovioVersion, pageWidth, pages, bars, notes };
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/web && bun run test src/lib/score-ir.test.ts
```

Expected: PASS — all 5 assertions pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai && git add apps/web/src/lib/score-ir.ts apps/web/src/lib/score-ir.test.ts && git commit -m "feat(score-ir): add ScoreIR types and parseScoreIR pure function"
```

---

## Task 2: Worker `get_page` and `get_ir` message handlers + updated CacheEntry

**Group:** A (parallel with Task 1)

**Behavior being verified:** The worker responds to a `get_page` message with the pre-rendered SVG for that page number, and responds to a `get_ir` message with the cached `ScoreIR`, both without re-rendering.

**Interface under test:** The worker's `onmessage` handler for `get_page` and `get_ir` message types, exercised via the mock-dispatch pattern in `score-worker.test.ts`.

**Files:**
- Modify: `apps/web/src/lib/score-worker.ts`
- Modify: `apps/web/src/lib/score-worker.test.ts`

- [ ] **Step 1: Write the failing test**

Add to `apps/web/src/lib/score-worker.test.ts` (append after existing describe blocks):

```typescript
describe("processGetPageRequest", () => {
  it("returns the pre-rendered SVG for the requested page number", async () => {
    const { processGetPageRequest } = await import("./score-worker");
    const pageSvgs = ["<svg>page1</svg>", "<svg>page2</svg>", "<svg>page3</svg>"];
    const result = processGetPageRequest(pageSvgs, 2);
    expect(result).toBe("<svg>page2</svg>");
  });

  it("returns 'failed' when the requested page does not exist", async () => {
    const { processGetPageRequest } = await import("./score-worker");
    const pageSvgs = ["<svg>page1</svg>"];
    const result = processGetPageRequest(pageSvgs, 99);
    expect(result).toBe("failed");
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/web && bun run test src/lib/score-worker.test.ts
```

Expected: FAIL — `processGetPageRequest is not a function` (export does not exist yet)

- [ ] **Step 3: Implement the minimum to make the test pass**

In `apps/web/src/lib/score-worker.ts`, make the following changes:

1. Update the `CacheEntry` interface to include `ir` and `pageSvgs`:

```typescript
// Replace the existing CacheEntry interface:
interface CacheEntry {
  tk: VerovioTk;
  measures: MeasureEntry[];
  ir: import("./score-ir").ScoreIR;
  pageSvgs: string[];
}
```

2. Add the `processGetPageRequest` export before the worker `onmessage` block:

```typescript
export function processGetPageRequest(pageSvgs: string[], pageN: number): string | "failed" {
  const svg = pageSvgs[pageN - 1];
  if (svg === undefined) return "failed";
  return svg;
}
```

Note: The `ir` field on `CacheEntry` will be wired up in Task 3 when `loadPiece` is updated. For this task, `CacheEntry` gains the fields; `loadPiece` keeps returning the old shape (no `ir` or `pageSvgs` yet). The worker message dispatch for `get_page` and `get_ir` will be added in Task 3 as well, since it requires the full updated `loadPiece`.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/web && bun run test src/lib/score-worker.test.ts
```

Expected: PASS — both new test cases and all prior tests pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai && git add apps/web/src/lib/score-worker.ts apps/web/src/lib/score-worker.test.ts && git commit -m "feat(score-worker): add processGetPageRequest export and updated CacheEntry shape"
```

---

## Task 3: Update `loadPiece` to render all pages + build IR eagerly

**Group:** B (sequential, depends on Group A — both Task 1 and Task 2 must be complete)

**Behavior being verified:** After `loadPiece` resolves for the Nocturne fixture, the returned entry contains a non-null `ir` with `bars.length === entry.measures.length`, `pageSvgs.length === ir.pages.length`, and every `NoteIR.bbox.x` is finite. Also verifies the worker's `get_ir` message handler returns the cached `ScoreIR` for a previously-loaded piece.

**Interface under test:** `loadPiece()` return value shape and the `get_ir` message response payload, exercised in `score-worker.integration.test.ts`.

**Files:**
- Modify: `apps/web/src/lib/score-worker.ts`
- Modify: `apps/web/src/lib/score-worker.integration.test.ts`

- [ ] **Step 1: Write the failing test**

Add a new describe block to `apps/web/src/lib/score-worker.integration.test.ts`:

```typescript
const NOCTURNE_FIXTURE_PATH = resolve(
  dirname(fileURLToPath(import.meta.url)),
  "../../public/scores/chopin-nocturne-op9-no2.mxl",
);

describe("loadPiece — IR and pageSvgs in returned CacheEntry", () => {
  it("returns a CacheEntry with ir and pageSvgs after loading the Nocturne", async () => {
    const esm = (await import("verovio/esm")) as any;
    const wasm = (await import("verovio/wasm")) as any;
    const VerovioToolkit = esm.VerovioToolkit ?? esm.default?.VerovioToolkit;
    const VerovioModule = wasm.default ?? wasm;
    const mod = await VerovioModule();

    const bytes = readFileSync(NOCTURNE_FIXTURE_PATH);
    const arrayBuf = new ArrayBuffer(bytes.byteLength);
    new Uint8Array(arrayBuf).set(bytes);

    const { loadPiece } = await import("./score-worker");
    const entry = await loadPiece(
      arrayBuf,
      { module: mod, ToolkitClass: VerovioToolkit as any },
      "chopin-nocturne-op9-no2",
    );

    expect(entry).not.toBe("failed");
    if (entry === "failed") return;

    // pageSvgs must be present and match page count
    expect(Array.isArray(entry.pageSvgs)).toBe(true);
    expect(entry.pageSvgs.length).toBeGreaterThan(0);

    // IR must be built
    expect(entry.ir).toBeDefined();
    expect(entry.ir.bars.length).toBe(entry.measures.length);
    expect(entry.ir.pages.length).toBe(entry.pageSvgs.length);

    // Every note must have finite x/y
    for (const note of Object.values(entry.ir.notes)) {
      expect(Number.isFinite(note.bbox.x)).toBe(true);
      expect(Number.isFinite(note.bbox.y)).toBe(true);
    }

    // At least some notes must have been found
    expect(Object.keys(entry.ir.notes).length).toBeGreaterThan(0);
  }, 30_000);
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/web && bun run test src/lib/score-worker.integration.test.ts
```

Expected: FAIL — `entry.pageSvgs is undefined` (current `loadPiece` does not render all pages or build IR)

- [ ] **Step 3: Implement the minimum to make the test pass**

Update `loadPiece` in `apps/web/src/lib/score-worker.ts`. Replace the section after `buildMeasureIndex` succeeds with the following:

```typescript
// After buildMeasureIndex succeeds, render all pages and build IR.
const pageCount = tk.getPageCount() as number;
if (pageCount === 0) return "failed";

const pageSvgs: string[] = [];
for (let n = 1; n <= pageCount; n++) {
  pageSvgs.push(tk.renderToSVG(n) as string);
}

const { parseScoreIR } = await import("./score-ir");
const ir = parseScoreIR(
  pieceId ?? "",
  pageSvgs,
  entry.measures,
  tk.getVersion() as string,
  VEROVIO_OPTS.pageWidth,
);

return { tk, measures: entry.measures, ir, pageSvgs };
```

Also add `get_ir` to the worker `onmessage` dispatch (inside the `if (typeof window === "undefined")` block), after the `get_page` dispatch case:

```typescript
// In the onmessage handler, update the WorkerInMsg union type:
type WorkerInMsg =
  | { type: "load";     requestId: string; pieceId: string; bytes: ArrayBuffer }
  | { type: "get_page"; requestId: string; pieceId: string; pageN: number }
  | { type: "get_clip"; requestId: string; pieceId: string; startBar: number; endBar: number }
  | { type: "get_ir";   requestId: string; pieceId: string };

// In the dispatch switch (replacing the existing render_full/render_clip if/else):
if (msg.type === "get_clip") {
  const svg = processRenderClipRequest(tk, measures, msg.startBar, msg.endBar);
  (self as unknown as Worker).postMessage({ requestId: msg.requestId, payload: svg });
} else if (msg.type === "get_page") {
  const svg = processGetPageRequest(result.pageSvgs, msg.pageN);
  if (svg === "failed") {
    (self as unknown as Worker).postMessage({
      requestId: msg.requestId,
      error: `page ${msg.pageN} not found for ${msg.pieceId}`,
    });
  } else {
    (self as unknown as Worker).postMessage({ requestId: msg.requestId, payload: svg });
  }
} else if (msg.type === "get_ir") {
  (self as unknown as Worker).postMessage({ requestId: msg.requestId, payload: result.ir });
}
```

The old `render_full` and `render_clip` message types are removed; callers now use `get_page` and `get_clip`. The `load` message type is added to carry bytes explicitly (wired in Task 4 on the renderer side).

Also update `renderFullSvg` — it remains exported for the unit test but is no longer used by the message handler.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/web && bun run test src/lib/score-worker.integration.test.ts
```

Expected: PASS — all integration tests pass, including the spike gate and the new IR/pageSvgs assertions.

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai && git add apps/web/src/lib/score-worker.ts apps/web/src/lib/score-worker.integration.test.ts && git commit -m "feat(score-worker): eager IR build and all-pages render in loadPiece; add get_page/get_ir handlers"
```

---

## Task 4: ScoreRenderer — `load`, `getPage`, `getIR` methods; rename `getFull` → `getPage(n)`

**Group:** C (parallel with Task 5, depends on Group B)

**Behavior being verified:** `scoreRenderer.load(pieceId)` resolves with `{ir, pageSvgs}` after fetching and loading the piece; `scoreRenderer.getIR(pieceId)` returns the same IR synchronously after load; `scoreRenderer.getPage(pieceId, 1)` returns the page-1 SVG string; `scoreRenderer.getClip` signature is unchanged.

**Interface under test:** Public methods of `ScoreRenderer` from `apps/web/src/lib/score-renderer.ts`.

**Files:**
- Modify: `apps/web/src/lib/score-renderer.ts`

Note: There is no unit test file for `score-renderer.ts` today — the renderer is covered by integration tests via `score-worker.integration.test.ts`. A focused unit-level behavior test is added here using a mock worker (the renderer's worker protocol, not Verovio internals).

- Create: `apps/web/src/lib/score-renderer.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/web/src/lib/score-renderer.test.ts
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// Intercept Worker construction so tests run without a real worker.
const mockPostMessage = vi.fn();
const mockWorkerInstance = {
  postMessage: mockPostMessage,
  onmessage: null as ((e: MessageEvent) => void) | null,
  onerror: null as ((e: ErrorEvent) => void) | null,
};

vi.stubGlobal("Worker", class {
  onmessage: ((e: MessageEvent) => void) | null = null;
  onerror: ((e: ErrorEvent) => void) | null = null;
  postMessage(data: unknown) {
    mockPostMessage(data);
    mockWorkerInstance.onmessage = this.onmessage;
    mockWorkerInstance.onerror = this.onerror;
  }
});

const FAKE_IR = {
  pieceId: "test-piece",
  verovioVersion: "4.0.0",
  pageWidth: 2400,
  pages: [{ pageN: 1, viewBox: "0 0 2400 800", width: 2400, height: 800, systemBboxes: [] }],
  bars: [],
  notes: {},
};

function simulateWorkerResponse(requestId: string, payload: unknown) {
  const handler = mockWorkerInstance.onmessage;
  if (handler) {
    handler(new MessageEvent("message", { data: { requestId, payload } }));
  }
}

describe("ScoreRenderer.load", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.resetModules();
  });

  it("resolves with {ir, pageSvgs} after a successful load message exchange", async () => {
    // Mock api.scores.getData to return a small ArrayBuffer
    vi.doMock("./api", () => ({
      api: {
        scores: {
          getData: vi.fn().mockResolvedValue(new ArrayBuffer(8)),
        },
      },
    }));

    const { ScoreRenderer } = await import("./score-renderer");
    const renderer = new ScoreRenderer();

    const payload = { ir: FAKE_IR, pageSvgs: ["<svg>page1</svg>"] };

    // Trigger load; it will postMessage to the mock worker.
    const loadPromise = renderer.load("test-piece");

    // Let the postMessage call register the worker handlers.
    await Promise.resolve();

    // Find the requestId from the postMessage call and simulate the worker response.
    const sentMsg = mockPostMessage.mock.calls[0]?.[0] as { requestId: string };
    simulateWorkerResponse(sentMsg.requestId, payload);

    const result = await loadPromise;
    expect(result).not.toBe("failed");
    if (result === "failed") return;
    expect(result.ir.pieceId).toBe("test-piece");
    expect(result.pageSvgs).toEqual(["<svg>page1</svg>"]);
  });

  it("getIR returns the cached IR synchronously after load resolves", async () => {
    vi.doMock("./api", () => ({
      api: { scores: { getData: vi.fn().mockResolvedValue(new ArrayBuffer(8)) } },
    }));

    const { ScoreRenderer } = await import("./score-renderer");
    const renderer = new ScoreRenderer();
    const payload = { ir: FAKE_IR, pageSvgs: ["<svg>page1</svg>"] };

    const loadPromise = renderer.load("test-piece");
    await Promise.resolve();
    const sentMsg = mockPostMessage.mock.calls[0]?.[0] as { requestId: string };
    simulateWorkerResponse(sentMsg.requestId, payload);
    await loadPromise;

    const ir = renderer.getIR("test-piece");
    expect(ir).not.toBeNull();
    expect(ir?.pieceId).toBe("test-piece");
  });

  it("getIR returns null when load has not been called", async () => {
    vi.doMock("./api", () => ({
      api: { scores: { getData: vi.fn().mockResolvedValue(new ArrayBuffer(8)) } },
    }));
    const { ScoreRenderer } = await import("./score-renderer");
    const renderer = new ScoreRenderer();
    expect(renderer.getIR("nonexistent-piece")).toBeNull();
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/web && bun run test src/lib/score-renderer.test.ts
```

Expected: FAIL — `ScoreRenderer is not exported` and `renderer.load is not a function` (current renderer exports only a singleton and lacks `load`/`getIR`/`getPage`)

- [ ] **Step 3: Implement the minimum to make the test pass**

Rewrite `apps/web/src/lib/score-renderer.ts`:

```typescript
// apps/web/src/lib/score-renderer.ts
import type { ScoreIR } from "./score-ir";
import { api } from "./api";

type PendingRequest = {
  resolve: (value: unknown) => void;
  reject: (err: Error) => void;
  pieceId: string;
};

export class ScoreRenderer {
  private worker: Worker | null = null;
  private pendingRequests = new Map<string, PendingRequest>();
  private bytesCache = new Map<string, ArrayBuffer>();
  private sentPieceIds = new Set<string>();
  private requestCounter = 0;
  private pendingFetches = new Map<string, Promise<void>>();
  // Main-thread IR cache populated from load() resolved payload.
  private irCache = new Map<string, ScoreIR>();

  private ensureWorker(): Worker {
    if (!this.worker) {
      if (typeof Worker === "undefined") {
        throw new Error("Web Workers are not available in this environment");
      }
      this.worker = new Worker(new URL("./score-worker.ts", import.meta.url), {
        type: "module",
      });
      this.worker.onmessage = (
        e: MessageEvent<{ requestId: string; payload?: unknown; error?: string }>,
      ) => {
        const { requestId, payload, error } = e.data;
        const pending = this.pendingRequests.get(requestId);
        if (!pending) return;
        this.pendingRequests.delete(requestId);
        if (error !== undefined) {
          this.sentPieceIds.delete(pending.pieceId);
          pending.reject(new Error(error));
        } else {
          pending.resolve(payload);
        }
      };
      this.worker.onerror = (e: ErrorEvent) => {
        const err = new Error(`Score worker crashed: ${e.message}`);
        for (const { reject, pieceId } of this.pendingRequests.values()) {
          this.sentPieceIds.delete(pieceId);
          reject(err);
        }
        this.pendingRequests.clear();
        this.worker = null;
      };
    }
    return this.worker;
  }

  private async ensureBytes(pieceId: string): Promise<void> {
    if (this.sentPieceIds.has(pieceId) || this.bytesCache.has(pieceId)) return;
    const inflight = this.pendingFetches.get(pieceId);
    if (inflight) return inflight;
    const fetchPromise = (async () => {
      const bytes = await api.scores.getData(pieceId);
      this.bytesCache.set(pieceId, bytes);
    })();
    this.pendingFetches.set(pieceId, fetchPromise);
    try {
      await fetchPromise;
    } finally {
      this.pendingFetches.delete(pieceId);
    }
  }

  private sendRequest<T>(
    pieceId: string,
    msg: Record<string, unknown>,
    bytes?: ArrayBuffer,
  ): Promise<T> {
    const worker = this.ensureWorker();
    return new Promise((resolve, reject) => {
      const requestId = `req-${++this.requestCounter}`;
      this.pendingRequests.set(requestId, {
        resolve: resolve as (v: unknown) => void,
        reject,
        pieceId,
      });
      worker.postMessage({ ...msg, requestId, pieceId, bytes });
    });
  }

  async load(
    pieceId: string,
  ): Promise<{ ir: ScoreIR; pageSvgs: string[] } | "failed"> {
    await this.ensureBytes(pieceId);
    const needsBytes = !this.sentPieceIds.has(pieceId);
    const bytes = needsBytes ? this.bytesCache.get(pieceId) : undefined;
    if (needsBytes && bytes === undefined) {
      throw new Error(`Score bytes missing after fetch for pieceId: ${pieceId}`);
    }
    if (needsBytes) this.sentPieceIds.add(pieceId);

    try {
      const payload = await this.sendRequest<{ ir: ScoreIR; pageSvgs: string[] }>(
        pieceId,
        { type: "load" },
        bytes,
      );
      this.irCache.set(pieceId, payload.ir);
      return payload;
    } catch {
      return "failed";
    }
  }

  getIR(pieceId: string): ScoreIR | null {
    return this.irCache.get(pieceId) ?? null;
  }

  async getPage(pieceId: string, pageN: number): Promise<string> {
    const worker = this.ensureWorker();
    return new Promise((resolve, reject) => {
      const requestId = `req-${++this.requestCounter}`;
      this.pendingRequests.set(requestId, {
        resolve: resolve as (v: unknown) => void,
        reject,
        pieceId,
      });
      worker.postMessage({ type: "get_page", requestId, pieceId, pageN });
    });
  }

  async getClip(
    pieceId: string,
    startBar: number,
    endBar: number,
  ): Promise<string> {
    const worker = this.ensureWorker();
    return new Promise((resolve, reject) => {
      const requestId = `req-${++this.requestCounter}`;
      this.pendingRequests.set(requestId, {
        resolve: resolve as (v: unknown) => void,
        reject,
        pieceId,
      });
      worker.postMessage({ type: "get_clip", requestId, pieceId, startBar, endBar });
    });
  }
}

export const scoreRenderer = new ScoreRenderer();
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/web && bun run test src/lib/score-renderer.test.ts
```

Expected: PASS — all 3 new tests pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai && git add apps/web/src/lib/score-renderer.ts apps/web/src/lib/score-renderer.test.ts && git commit -m "feat(score-renderer): add load/getIR/getPage methods; export ScoreRenderer class"
```

---

## Task 5: Update `ScorePanel.tsx` caller and `score-worker.test.ts` for new message types

**Group:** C (parallel with Task 4, depends on Group B)

**Behavior being verified:** `ScorePanel.tsx` calls `getPage(pieceId, 1)` instead of the removed `getFull(pieceId)`; the mock-based unit tests in `score-worker.test.ts` compile and pass against the new message type names.

**Interface under test:** `scoreRenderer.getPage(pieceId, 1)` in `ScorePanel.tsx`; the `render_full` → removal in `score-worker.test.ts`.

**Files:**
- Modify: `apps/web/src/components/ScorePanel.tsx` (line 276)
- Modify: `apps/web/src/lib/score-worker.test.ts`

- [ ] **Step 1: Write the failing test**

The test here is the TypeScript compiler: `ScorePanel.tsx` imports `scoreRenderer` and calls `getFull`, which no longer exists after Task 4. The "test" is the type-check step.

Add to `score-worker.test.ts` (replace the `renderFullSvg` describe block entirely):

```typescript
describe("processGetPageRequest", () => {
  it("returns the SVG for a valid page number from the pre-rendered cache", async () => {
    const { processGetPageRequest } = await import("./score-worker");
    const svgs = ["<svg>p1</svg>", "<svg>p2</svg>"];
    expect(processGetPageRequest(svgs, 1)).toBe("<svg>p1</svg>");
    expect(processGetPageRequest(svgs, 2)).toBe("<svg>p2</svg>");
  });

  it("returns 'failed' for an out-of-range page number", async () => {
    const { processGetPageRequest } = await import("./score-worker");
    const svgs = ["<svg>p1</svg>"];
    expect(processGetPageRequest(svgs, 0)).toBe("failed");
    expect(processGetPageRequest(svgs, 5)).toBe("failed");
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/web && bun run test src/lib/score-worker.test.ts
```

Expected: FAIL — `renderFullSvg describe block` tests that reference the old `renderFullSvg` may produce type errors; `processGetPageRequest` does not yet exist (was added structurally in Task 2 but the test file still imports `renderFullSvg`).

- [ ] **Step 3: Implement the minimum to make the test pass**

1. In `apps/web/src/components/ScorePanel.tsx` at line 276, replace:
   ```typescript
   const svg = await scoreRenderer.getFull(pieceId);
   ```
   with:
   ```typescript
   const svg = await scoreRenderer.getPage(pieceId, 1);
   ```

2. In `apps/web/src/lib/score-worker.test.ts`, remove the `renderFullSvg` describe block (lines 63-72 from the original) and replace it with the `processGetPageRequest` describe block written in Step 1 above.

3. Keep the `processRenderClipRequest` and `loadPiece` describe blocks unchanged.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/web && bun run test src/lib/score-worker.test.ts
```

Expected: PASS — all unit tests pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai && git add apps/web/src/components/ScorePanel.tsx apps/web/src/lib/score-worker.test.ts && git commit -m "fix(score-panel): call getPage(pieceId, 1) replacing removed getFull; update unit tests for new message types"
```

---

## Task 6: Integration tests for IR invariants × 3 fixtures, IR/Clip correlation, and eviction

**Group:** D (sequential, depends on Group C — Tasks 4 and 5 both complete)

**Behavior being verified:** For all 3 fixtures (Czerny, Nocturne, Ballade), `loadPiece` returns a `CacheEntry` where `bars.length === measures.length`, `pages.length > 0`, and all notes have finite x/y. Additionally: `getClip`'s first measure id matches `ir.bars[startBar-1].measureOn` (IR/clip correlation). Reloading the same `pieceId` with different bytes yields disjoint `notes` keys (eviction correctness).

**Interface under test:** `loadPiece()` and `processRenderClipRequest()` from `score-worker.ts`, exercised with real Verovio against real fixtures in `score-worker.integration.test.ts`.

**Files:**
- Modify: `apps/web/src/lib/score-worker.integration.test.ts`

- [ ] **Step 1: Write the failing test**

Append to `apps/web/src/lib/score-worker.integration.test.ts`:

```typescript
const CZERNY_FIXTURE_PATH = resolve(
  dirname(fileURLToPath(import.meta.url)),
  "../../public/scores/czerny-op299-no1.mxl",
);

async function makeBindings() {
  const esm = (await import("verovio/esm")) as any;
  const wasm = (await import("verovio/wasm")) as any;
  const VerovioToolkit = esm.VerovioToolkit ?? esm.default?.VerovioToolkit;
  const VerovioModule = wasm.default ?? wasm;
  const mod = await VerovioModule();
  return { module: mod, ToolkitClass: VerovioToolkit as any };
}

describe("IR invariants — all 3 fixtures", () => {
  const fixtures = [
    { name: "czerny-op299-no1", path: CZERNY_FIXTURE_PATH },
    { name: "chopin-nocturne-op9-no2", path: FIXTURE_PATH },
    { name: "chopin-ballade-op23-no1", path: BALLADE_FIXTURE_PATH },
  ];

  for (const { name, path } of fixtures) {
    it(`${name}: bars.length === measures.length and all notes have finite x/y`, async () => {
      const bindings = await makeBindings();
      const bytes = readFileSync(path);
      const arrayBuf = new ArrayBuffer(bytes.byteLength);
      new Uint8Array(arrayBuf).set(bytes);

      const { loadPiece } = await import("./score-worker");
      const entry = await loadPiece(arrayBuf, bindings, name);

      expect(entry).not.toBe("failed");
      if (entry === "failed") return;

      expect(entry.ir.bars.length).toBe(entry.measures.length);
      expect(entry.ir.pages.length).toBeGreaterThan(0);
      expect(entry.ir.pages.length).toBe(entry.pageSvgs.length);

      for (const note of Object.values(entry.ir.notes)) {
        expect(Number.isFinite(note.bbox.x)).toBe(true);
        expect(Number.isFinite(note.bbox.y)).toBe(true);
      }
    }, 60_000);
  }
});

describe("IR/Clip correlation", () => {
  it("getClip first measure id matches ir.bars[startBar-1].measureOn for the Nocturne", async () => {
    const bindings = await makeBindings();
    const bytes = readFileSync(FIXTURE_PATH);
    const arrayBuf = new ArrayBuffer(bytes.byteLength);
    new Uint8Array(arrayBuf).set(bytes);

    const { loadPiece, processRenderClipRequest } = await import("./score-worker");
    const entry = await loadPiece(arrayBuf, bindings, "nocturne-correlation");

    expect(entry).not.toBe("failed");
    if (entry === "failed") return;

    const startBar = 5;
    const endBar = 10;
    const svg = processRenderClipRequest(entry.tk, entry.measures, startBar, endBar);

    const expectedMeasureOn = entry.ir.bars[startBar - 1]?.measureOn;
    expect(expectedMeasureOn).toBeTruthy();

    // firstMeasureId is defined earlier in the file
    const actualId = firstMeasureId(svg);
    expect(actualId).toBe(expectedMeasureOn);
  }, 30_000);
});

describe("Cache eviction — reloading pieceId with different bytes yields disjoint note keys", () => {
  it("a second loadPiece call for the same pieceId produces a different ir.notes keyset", async () => {
    const bindings = await makeBindings();

    const nocturneBytes = readFileSync(FIXTURE_PATH);
    const nocturneBuf = new ArrayBuffer(nocturneBytes.byteLength);
    new Uint8Array(nocturneBuf).set(nocturneBytes);

    const czernyBytes = readFileSync(CZERNY_FIXTURE_PATH);
    const czernyBuf = new ArrayBuffer(czernyBytes.byteLength);
    new Uint8Array(czernyBuf).set(czernyBytes);

    const { loadPiece } = await import("./score-worker");

    const entry1 = await loadPiece(nocturneBuf, bindings, "eviction-test");
    expect(entry1).not.toBe("failed");
    if (entry1 === "failed") return;
    const firstKeys = new Set(Object.keys(entry1.ir.notes));

    const entry2 = await loadPiece(czernyBuf, bindings, "eviction-test");
    expect(entry2).not.toBe("failed");
    if (entry2 === "failed") return;
    const secondKeys = new Set(Object.keys(entry2.ir.notes));

    // Verovio randomizes element ids on every loadData call — the two keysets
    // should be disjoint (different pieces loaded into the same pieceId slot).
    const intersection = [...firstKeys].filter((k) => secondKeys.has(k));
    // Allow at most 5% overlap to account for any coincidental id collision.
    expect(intersection.length).toBeLessThan(firstKeys.size * 0.05);
  }, 60_000);
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/web && bun run test src/lib/score-worker.integration.test.ts
```

Expected: FAIL — `entry.ir is undefined` or `entry.pageSvgs is undefined` (loadPiece has not yet been updated in Task 3 if this runs before Task 3 completes — but Task 3 is in Group B which must precede Group D)

- [ ] **Step 3: Implement the minimum to make the test pass**

No new implementation — Group B (Task 3) must already be complete before Group D runs. If this test fails for any reason other than a fixture or fixture-path issue, the root cause is in Task 3's implementation, which must be fixed there.

If `BALLADE_FIXTURE_PATH` and `CZERNY_FIXTURE_PATH` are not yet declared at the top of the file, add them (they may already be present from Task 0's additions).

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/web && bun run test src/lib/score-worker.integration.test.ts
```

Expected: PASS — all integration tests pass including the three-fixture loop and correlation/eviction tests.

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai && git add apps/web/src/lib/score-worker.integration.test.ts && git commit -m "test(score-worker): IR invariants x3 fixtures, IR/clip correlation, eviction"
```

---

## Task 7: `ScoreCursor` class — rAF loop, binary search, overlay mount/unmount

**Group:** E (parallel with Task 8, depends on Group D)

**Behavior being verified:** A `ScoreCursor` instantiated with a frozen IR fixture and a `qstampSource` returning a qstamp inside bar 5 places its overlay `<line>` x1/x2 within 1px of the expected interpolated position after one rAF tick. A `qstampSource` returning `null` results in the overlay being hidden (display: none) within one rAF tick.

**Interface under test:** `ScoreCursor.start()`, `ScoreCursor.stop()`, overlay `<line>` attribute state, from `apps/web/src/lib/score-cursor.ts`.

**Files:**
- Create: `apps/web/src/lib/score-cursor.ts`
- Create: `apps/web/src/lib/score-cursor.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/web/src/lib/score-cursor.test.ts
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ScoreIR } from "./score-ir";

// jsdom does not implement requestAnimationFrame; provide a manual scheduler.
let rafCallback: FrameRequestCallback | null = null;
vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) => {
  rafCallback = cb;
  return 1;
});
vi.stubGlobal("cancelAnimationFrame", vi.fn());

function flushRaf() {
  if (rafCallback) {
    const cb = rafCallback;
    rafCallback = null;
    cb(performance.now());
  }
}

// Minimal IR: 2 bars, 2 notes per bar, one page.
const FAKE_IR: ScoreIR = {
  pieceId: "test",
  verovioVersion: "4.0.0",
  pageWidth: 2400,
  pages: [{ pageN: 1, viewBox: "0 0 2400 800", width: 2400, height: 800, systemBboxes: [] }],
  bars: [
    {
      barNumber: 1,
      measureOn: "m1",
      pageN: 1,
      bbox: { x: 100, y: 0, w: 0, h: 0 },
      noteIds: ["n1", "n2"],
      qstampStart: 0,
      qstampEnd: 4,
    },
    {
      barNumber: 2,
      measureOn: "m2",
      pageN: 1,
      bbox: { x: 500, y: 0, w: 0, h: 0 },
      noteIds: ["n3", "n4"],
      qstampStart: 4,
      qstampEnd: 8,
    },
  ],
  notes: {
    n1: { id: "n1", bbox: { x: 100, y: 200, w: 0, h: 0 }, qstamp: 0, staff: 1 },
    n2: { id: "n2", bbox: { x: 300, y: 200, w: 0, h: 0 }, qstamp: 2, staff: 1 },
    n3: { id: "n3", bbox: { x: 500, y: 200, w: 0, h: 0 }, qstamp: 4, staff: 1 },
    n4: { id: "n4", bbox: { x: 700, y: 200, w: 0, h: 0 }, qstamp: 6, staff: 1 },
  },
};

describe("ScoreCursor", () => {
  let container: HTMLElement;

  beforeEach(() => {
    container = document.createElement("div");
    document.body.appendChild(container);
    rafCallback = null;
  });

  afterEach(() => {
    document.body.removeChild(container);
  });

  it("mounts an overlay svg on start() and removes it on stop()", async () => {
    const { ScoreCursor } = await import("./score-cursor");
    const cursor = new ScoreCursor({
      pieceId: "test",
      container,
      ir: FAKE_IR,
      qstampSource: () => null,
    });
    cursor.start();
    expect(container.querySelector("svg.score-cursor-overlay")).not.toBeNull();
    cursor.stop();
    expect(container.querySelector("svg.score-cursor-overlay")).toBeNull();
  });

  it("hides the overlay line when qstampSource returns null", async () => {
    const { ScoreCursor } = await import("./score-cursor");
    const cursor = new ScoreCursor({
      pieceId: "test",
      container,
      ir: FAKE_IR,
      qstampSource: () => null,
    });
    cursor.start();
    flushRaf();
    const line = container.querySelector("svg.score-cursor-overlay line") as SVGLineElement | null;
    expect(line?.getAttribute("visibility")).toBe("hidden");
    cursor.stop();
  });

  it("positions the cursor line within 1px of the expected interpolated x for a qstamp inside bar 1", async () => {
    const { ScoreCursor } = await import("./score-cursor");
    // qstamp = 1: halfway between n1 (x=100, qstamp=0) and n2 (x=300, qstamp=2)
    // expected interpolated x = 100 + (1-0)/(2-0) * (300-100) = 200
    const cursor = new ScoreCursor({
      pieceId: "test",
      container,
      ir: FAKE_IR,
      qstampSource: () => 1,
    });
    cursor.start();
    flushRaf();
    const line = container.querySelector("svg.score-cursor-overlay line") as SVGLineElement | null;
    expect(line).not.toBeNull();
    expect(line?.getAttribute("visibility")).not.toBe("hidden");
    const x1 = parseFloat(line?.getAttribute("x1") ?? "NaN");
    expect(Math.abs(x1 - 200)).toBeLessThan(1);
    cursor.stop();
  });

  it("keeps the rAF loop alive and captures exception via Sentry when qstampSource throws", async () => {
    const sentryMock = { captureException: vi.fn() };
    vi.doMock("./sentry", () => ({ Sentry: sentryMock }));

    const { ScoreCursor } = await import("./score-cursor");
    const cursor = new ScoreCursor({
      pieceId: "test",
      container,
      ir: FAKE_IR,
      qstampSource: () => { throw new Error("source exploded"); },
    });
    cursor.start();
    flushRaf();
    expect(sentryMock.captureException).toHaveBeenCalled();
    // rAF loop must have re-scheduled (rafCallback is set again)
    expect(rafCallback).not.toBeNull();
    cursor.stop();
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/web && bun run test src/lib/score-cursor.test.ts
```

Expected: FAIL — `Cannot find module './score-cursor'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `apps/web/src/lib/score-cursor.ts`:

```typescript
// apps/web/src/lib/score-cursor.ts
import type { BarIR, NoteIR, ScoreIR } from "./score-ir";
import { Sentry } from "./sentry";

export interface ScoreCursorOptions {
  pieceId: string;
  container: HTMLElement;
  ir: ScoreIR;
  qstampSource: () => number | null;
}

export class ScoreCursor {
  private readonly container: HTMLElement;
  private readonly ir: ScoreIR;
  private readonly qstampSource: () => number | null;
  // One overlay <svg> per page in the IR.
  private overlays: SVGSVGElement[] = [];
  private rafId: number | null = null;
  private lastPageN = -1;

  constructor(opts: ScoreCursorOptions) {
    this.container = opts.container;
    this.ir = opts.ir;
    this.qstampSource = opts.qstampSource;
  }

  start(): void {
    this.mountOverlays();
    this.rafId = requestAnimationFrame(this.tick);
  }

  stop(): void {
    if (this.rafId !== null) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
    this.unmountOverlays();
  }

  private mountOverlays(): void {
    for (let i = 0; i < this.ir.pages.length; i++) {
      const page = this.ir.pages[i];
      const overlay = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      overlay.setAttribute("class", "score-cursor-overlay");
      overlay.setAttribute("viewBox", page.viewBox);
      overlay.setAttribute("style", [
        "position:absolute",
        "top:0",
        "left:0",
        "width:100%",
        "height:100%",
        "pointer-events:none",
      ].join(";"));

      const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("x1", "0");
      line.setAttribute("x2", "0");
      line.setAttribute("y1", "0");
      line.setAttribute("y2", String(page.height));
      line.setAttribute("stroke", "#2563eb");
      line.setAttribute("stroke-width", "2");
      line.setAttribute("visibility", "hidden");
      overlay.appendChild(line);

      this.container.appendChild(overlay);
      this.overlays.push(overlay);
    }
  }

  private unmountOverlays(): void {
    for (const overlay of this.overlays) {
      if (overlay.parentNode) {
        overlay.parentNode.removeChild(overlay);
      }
    }
    this.overlays = [];
  }

  private tick = (_ts: number): void => {
    try {
      const q = this.qstampSource();
      if (q === null) {
        this.hideAll();
        this.rafId = requestAnimationFrame(this.tick);
        return;
      }

      const bar = this.findBar(q);
      if (!bar) {
        this.hideAll();
        this.rafId = requestAnimationFrame(this.tick);
        return;
      }

      const x = this.interpolateX(bar, q);
      const overlayIdx = bar.pageN - 1;
      const overlay = this.overlays[overlayIdx];
      if (!overlay) {
        this.rafId = requestAnimationFrame(this.tick);
        return;
      }

      // Hide all overlays except the current page.
      for (let i = 0; i < this.overlays.length; i++) {
        const line = this.overlays[i].querySelector("line");
        if (!line) continue;
        if (i === overlayIdx) {
          line.setAttribute("x1", String(x));
          line.setAttribute("x2", String(x));
          line.setAttribute("visibility", "visible");
        } else {
          line.setAttribute("visibility", "hidden");
        }
      }

      // Scroll into view when the cursor crosses a page boundary.
      if (bar.pageN !== this.lastPageN) {
        overlay.scrollIntoView({ block: "nearest" });
        this.lastPageN = bar.pageN;
      }
    } catch (err) {
      this.hideAll();
      Sentry.captureException(err);
    }
    this.rafId = requestAnimationFrame(this.tick);
  };

  private hideAll(): void {
    for (const overlay of this.overlays) {
      const line = overlay.querySelector("line");
      if (line) line.setAttribute("visibility", "hidden");
    }
  }

  // Binary search: find the bar where qstampStart <= q < qstampEnd.
  private findBar(q: number): BarIR | null {
    const bars = this.ir.bars;
    let lo = 0;
    let hi = bars.length - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >>> 1;
      const bar = bars[mid];
      if (q < bar.qstampStart) {
        hi = mid - 1;
      } else if (q >= bar.qstampEnd) {
        lo = mid + 1;
      } else {
        return bar;
      }
    }
    // q is at or past the last bar's qstampEnd — return the last bar.
    return bars[bars.length - 1] ?? null;
  }

  // Linear interpolation of x within a bar between the two bracketing notes.
  private interpolateX(bar: BarIR, q: number): number {
    const notes = bar.noteIds
      .map((id) => this.ir.notes[id])
      .filter((n): n is NoteIR => n !== undefined)
      .sort((a, b) => a.qstamp - b.qstamp);

    if (notes.length === 0) return bar.bbox.x;
    if (notes.length === 1) return notes[0].bbox.x;

    // Find the two bracketing notes.
    let prev = notes[0];
    let next = notes[notes.length - 1];
    for (let i = 0; i < notes.length - 1; i++) {
      if (notes[i].qstamp <= q && notes[i + 1].qstamp > q) {
        prev = notes[i];
        next = notes[i + 1];
        break;
      }
    }

    if (prev === next) return prev.bbox.x;
    const t = (q - prev.qstamp) / (next.qstamp - prev.qstamp);
    return prev.bbox.x + t * (next.bbox.x - prev.bbox.x);
  }
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/web && bun run test src/lib/score-cursor.test.ts
```

Expected: PASS — all 4 cursor tests pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai && git add apps/web/src/lib/score-cursor.ts apps/web/src/lib/score-cursor.test.ts && git commit -m "feat(score-cursor): add ScoreCursor class with rAF loop, binary search, and overlay management"
```

---

## Task 8: Integration test — Ballade `load()` perf + IR invariants at scale

**Group:** E (parallel with Task 7, depends on Group D)

**Behavior being verified:** For the Ballade (largest fixture, 56.7KB), the full `loadPiece` including eager IR build completes under 200ms; `ir.bars.length === measures.length`; all notes have finite x/y. This is the final verification of the eager-IR contract at production scale.

**Interface under test:** `loadPiece()` exercised with real Verovio in `score-worker.integration.test.ts`.

**Files:**
- Modify: `apps/web/src/lib/score-worker.integration.test.ts`

- [ ] **Step 1: Write the failing test**

Add a new describe block (the Task 0 spike test measured the old `loadPiece`; this one measures the new one including IR build):

```typescript
describe("Ballade load() with full IR build — production-scale perf gate", () => {
  it("completes loadPiece + IR build for the Ballade within 200ms", async () => {
    const esm = (await import("verovio/esm")) as any;
    const wasm = (await import("verovio/wasm")) as any;
    const VerovioToolkit = esm.VerovioToolkit ?? esm.default?.VerovioToolkit;
    const VerovioModule = wasm.default ?? wasm;
    const mod = await VerovioModule();

    const bytes = readFileSync(BALLADE_FIXTURE_PATH);
    const arrayBuf = new ArrayBuffer(bytes.byteLength);
    new Uint8Array(arrayBuf).set(bytes);

    const { loadPiece } = await import("./score-worker");

    const t0 = Date.now();
    const entry = await loadPiece(
      arrayBuf,
      { module: mod, ToolkitClass: VerovioToolkit as any },
      "chopin-ballade-op23-no1-perf",
    );
    const elapsed = Date.now() - t0;

    expect(entry).not.toBe("failed");
    if (entry === "failed") return;

    expect(entry.ir.bars.length).toBe(entry.measures.length);
    expect(elapsed).toBeLessThan(200);
  }, 30_000);
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/web && bun run test src/lib/score-worker.integration.test.ts --reporter=verbose
```

Expected: FAIL — the new describe block fails because Task 3 (Group B) must be complete first. If Group B is already done, this test may pass immediately, in which case proceed to Step 5.

- [ ] **Step 3: Implement the minimum to make the test pass**

No new implementation — this test depends entirely on Task 3's changes to `loadPiece`. If Group B is complete and the test fails the 200ms threshold, halt the build and flag the issue: the spec must be revised toward lazy-IR.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/web && bun run test src/lib/score-worker.integration.test.ts
```

Expected: PASS — all integration tests pass, including the Ballade perf gate with full IR build.

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai && git add apps/web/src/lib/score-worker.integration.test.ts && git commit -m "test(score-worker): Ballade load+IR perf gate with full eager-IR build"
```

---

## Task 9: Full test suite — verify all tests pass together

**Group:** F (sequential, depends on all prior groups)

**Behavior being verified:** All four test files (`score-ir.test.ts`, `score-worker.test.ts`, `score-worker.integration.test.ts`, `score-cursor.test.ts`, `score-renderer.test.ts`) pass in a single `bun run test` invocation. TypeScript compilation produces no errors.

**Interface under test:** Full public surface of all four new/modified modules.

**Files:**
- No new files.

- [ ] **Step 1: Write the failing test**

This task has no new test code — it is the final verification sweep. Skip to Step 2.

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/web && bun run test src/lib/score-ir.test.ts src/lib/score-worker.test.ts src/lib/score-worker.integration.test.ts src/lib/score-cursor.test.ts src/lib/score-renderer.test.ts
```

Expected: Should PASS if all prior tasks are complete. If any test fails, trace to the owning task and fix it there.

- [ ] **Step 3: Implement the minimum to make the test pass**

Fix any cross-task type or import issues discovered by running the full suite together. Common issues:
- `processGetPageRequest` import in `score-worker.test.ts` not resolving if Task 2 export name differs
- `ScoreRenderer` export (named vs default) inconsistency between Task 4 impl and Task 4 test
- `Sentry` import path in `score-cursor.ts` must be `./sentry` (relative, not `@sentry/react` directly)

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/web && bun run test src/lib/score-ir.test.ts src/lib/score-worker.test.ts src/lib/score-worker.integration.test.ts src/lib/score-cursor.test.ts src/lib/score-renderer.test.ts
```

Expected: PASS — all suites green.

Also run the TypeScript type check:

```bash
cd /Users/jdhiman/Documents/crescendai/apps/web && bunx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai && git add -A && git commit -m "test(score-rendering-phase2): full suite green — IR, worker, cursor, renderer"
```

---

## Spec Coverage Checklist

| Spec requirement | Task |
|---|---|
| ScoreIR types locked | Task 1 |
| `parseScoreIR` pure function | Task 1 |
| Worker `get_page` message | Task 2, Task 3 |
| Worker `get_ir` message | Task 2, Task 3 |
| `loadPiece` renders all pages eagerly | Task 3 |
| `loadPiece` builds IR eagerly | Task 3 |
| `getPageCount() === 0` → `"failed"` | Task 3 |
| `ScoreRenderer.load` → `{ir, pageSvgs}` | Task 4 |
| `ScoreRenderer.getIR` synchronous | Task 4 |
| `ScoreRenderer.getPage(n)` replacing `getFull` | Task 4 |
| `ScoreRenderer.getClip` signature unchanged | Task 4 |
| `ScorePanel.tsx:276` getFull → getPage(_, 1) | Task 5 |
| IR invariants × 3 fixtures | Task 6 |
| IR/clip correlation | Task 6 |
| Cache eviction disjoint note keys | Task 6 |
| `ScoreCursor.start()` / `stop()` | Task 7 |
| Overlay mount/unmount | Task 7 |
| rAF loop with binary search + interpolation | Task 7 |
| Null qstamp hides overlay | Task 7 |
| `qstampSource` throws → Sentry + loop alive | Task 7 |
| Ballade load + IR build < 200ms | Task 8 |
| Full suite type-check | Task 9 |
