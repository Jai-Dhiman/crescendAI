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

## Task 0: Prerequisite Spike — Measure Ballade IR marginal cost

**Group:** 0 (sequential gate — if marginal IR-build cost exceeds 200ms, halt the build and revise the spec toward lazy-IR before continuing)

**Behavior being verified:** The MARGINAL cost of IR build (measured as Δ between with-IR and without-IR `loadPiece` runs on the Ballade fixture) is under 200ms, confirming that eager IR build at load time is a viable contract. Total `loadPiece` wall-clock (~2-4s for Ballade) is Verovio's intrinsic cost and is NOT gated here — the UI shows a loading state for the duration.

**Interface under test:** `loadPiece()` from `apps/web/src/lib/score-worker.ts` exercised directly (same pattern as the existing integration test). Two runs: one baseline (without IR, current code path) and one with-IR (after Task 3 lands), both on a warm WASM module.

**Files:**
- Modify: `apps/web/src/lib/score-worker.integration.test.ts`

- [ ] **Step 1: Write the failing test**

The test is already committed at HEAD (revised 2026-05-26 to use marginal-cost assertion). See the `"IR build marginal cost (with-IR minus without-IR) is under 200ms"` it-block in `score-worker.integration.test.ts`.

The test structure:
1. Load Ballade with a baseline pieceId (no IR — measures Verovio intrinsic cost).
2. Load Ballade with the actual pieceId (with IR — after Task 3, measures Verovio + IR walk).
3. Compute `marginalMs = max(0, withIrMs - baselineMs)`.
4. GATE (a): `entry.ir` is defined and `entry.ir.pages.length > 0`.
5. GATE (b): `marginalMs < 200`.

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/web && bun run test src/lib/score-worker.integration.test.ts
```

Expected: FAIL — `entry.ir is undefined`. The current `loadPiece` returns a `CacheEntry` without an `ir` field, so `expect(entry.ir).toBeDefined()` fails immediately. This preserves watch-it-fail discipline: test stays RED through Tasks 1-2, turns GREEN when Task 3 wires IR build into `loadPiece`.

> **Build-agent gate:** If the test fails for any reason OTHER than `entry.ir is undefined` (e.g. `marginalMs >= 200` or the fixture does not load), STOP. Do not proceed to Group A. File a note that the spike is broken and await further instruction.

- [ ] **Step 3: Implement the minimum to make the test pass**

The test cannot pass until Task 3 attaches IR build to `loadPiece`. This task's Step 3 is intentionally deferred: the test stays red through Group A, and turns green when Group B (Task 3) completes. The build agent must confirm the test is red after Step 2, then proceed to Group A, then verify it turns green after Task 3's commit. If `marginalMs >= 200` at that point, halt and flag for lazy-IR revision.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/web && bun run test src/lib/score-worker.integration.test.ts
```

Expected: PASS — `entry.ir` is defined, `marginalMs < 200` (IR walk is cheap; total load may be ~2-4s).

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai && git add apps/web/src/lib/score-worker.integration.test.ts && git commit -m "test(score-worker): spike gate — Ballade IR marginal cost under 200ms"
```

---

## Task 1: IR Types and parseScoreIR pure function

**Group:** A (parallel with Task 2)

**Behavior being verified:** `parseScoreIR` extracts a structurally valid `ScoreIR` from Verovio-rendered SVG pages: every `NoteIR.bbox.x` and `.y` is a finite number, every `BarIR.noteIds` resolves to a key in `notes`, `qstampStart < qstampEnd` for every bar, `bars.length` equals the number of supplied measures, and for any bar containing notes at two or more distinct onset positions (i.e. not all notes are a chord at the same tick), the notes carry at least two distinct `qstamp` values.

**Interface under test:** `parseScoreIR(pieceId, pageSvgs, measures, noteQstampMap, verovioVersion, pageWidth)` from `apps/web/src/lib/score-ir.ts`, exercised with real Verovio SVG fixtures produced inline in the test. `noteQstampMap` is a `Map<string, number>` from note element id to its onset qstamp, built by the worker from `tk.renderToTimemap({ includeNotes: true })` before calling `parseScoreIR`.

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

// noteQstampMap: per-note onset qstamps from the timemap.
// m1 has two notes at different onsets (0 and 2); m2 has two notes at different onsets (4 and 6).
const SYNTHETIC_NOTE_QSTAMP_MAP = new Map<string, number>([
  ["note-1-1", 0],
  ["note-1-2", 2],
  ["note-2-1", 4],
  ["note-2-2", 6],
]);

describe("parseScoreIR — structural invariants", () => {
  it("returns a ScoreIR whose bars.length equals the supplied measures count", async () => {
    const { parseScoreIR } = await import("./score-ir");
    const ir = parseScoreIR(
      "test-piece",
      [SYNTHETIC_PAGE_SVG],
      SYNTHETIC_MEASURES,
      SYNTHETIC_NOTE_QSTAMP_MAP,
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
      SYNTHETIC_NOTE_QSTAMP_MAP,
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
      SYNTHETIC_NOTE_QSTAMP_MAP,
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
      SYNTHETIC_NOTE_QSTAMP_MAP,
      "4.0.0",
      2400,
    );
    for (const bar of ir.bars) {
      if (bar.noteIds.length > 0) {
        expect(bar.qstampStart).toBeLessThan(bar.qstampEnd);
      }
    }
  });

  it("notes with distinct onset positions carry distinct qstamp values (not all qstampStart)", async () => {
    const { parseScoreIR } = await import("./score-ir");
    const ir = parseScoreIR(
      "test-piece",
      [SYNTHETIC_PAGE_SVG],
      SYNTHETIC_MEASURES,
      SYNTHETIC_NOTE_QSTAMP_MAP,
      "4.0.0",
      2400,
    );
    // m1 has notes at qstamp 0 and 2 — they must be distinct in the IR.
    const bar1 = ir.bars.find((b) => b.measureOn === "m1");
    expect(bar1).toBeDefined();
    if (bar1 && bar1.noteIds.length >= 2) {
      const qstamps = bar1.noteIds.map((id) => ir.notes[id]?.qstamp ?? -1);
      const uniqueQstamps = new Set(qstamps);
      expect(uniqueQstamps.size).toBeGreaterThan(1);
    }
  });

  it("stores pieceId, verovioVersion, and pageWidth on the IR", async () => {
    const { parseScoreIR } = await import("./score-ir");
    const ir = parseScoreIR(
      "my-piece",
      [SYNTHETIC_PAGE_SVG],
      SYNTHETIC_MEASURES,
      SYNTHETIC_NOTE_QSTAMP_MAP,
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
  noteQstampMap: Map<string, number>,
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
      // Per-note qstamp from the timemap. Fall back to 0 if the note id is absent
      // (this should not occur for well-formed Verovio output but is explicit, not silent).
      const qstamp = noteQstampMap.get(id) ?? 0;
      notes[id] = {
        id,
        bbox: { x, y, w: 0, h: 0 },
        qstamp,
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

Note: `processGetPageRequest` accepts a `pageSvgs` array and a 1-based page number. The worker message handler in Task 3 also handles an optional `pageWidth` parameter for the sandbox's responsive-width use case (see `get_page` handler in Task 3 Step 3).

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

    // Responsive-width regression: re-rendering page 1 with a different pageWidth must produce
    // an SVG whose <svg> width attribute reflects the new width, not the cached pageWidth.
    // This assertion catches any missing tk.redoLayout({}) between setOptions and renderToSVG.
    const originalPageWidth = entry.ir.pageWidth;
    const altWidth = originalPageWidth - 200;
    entry.tk.setOptions({ pageWidth: altWidth });
    entry.tk.redoLayout({});
    const altSvg = entry.tk.renderToSVG(1) as string;
    entry.tk.setOptions({ pageWidth: originalPageWidth });
    entry.tk.redoLayout({});
    const cachedWidth = entry.pageSvgs[0]?.match(/width="(\d+)"/)?.[1];
    const altSvgWidth = altSvg.match(/width="(\d+)"/)?.[1];
    expect(altSvgWidth).toBeDefined();
    expect(cachedWidth).toBeDefined();
    // The re-rendered SVG's width must differ from the cached page's width.
    // If redoLayout is missing, Verovio returns stale layout and widths will incorrectly match.
    expect(altSvgWidth).not.toBe(cachedWidth);
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

// Build a noteId -> onset qstamp map from the Verovio timemap.
// tk.renderToTimemap({ includeNotes: true }) returns an array of entries like:
//   { on: number, off: number, qon: number, qoff: number, notes: string[] }
// where `notes` is an array of note element ids sharing that onset tick.
const noteQstampMap = new Map<string, number>();
const timemap = tk.renderToTimemap({ includeNotes: true }) as Array<{
  qon: number;
  notes?: string[];
}>;
for (const entry of timemap) {
  if (Array.isArray(entry.notes)) {
    for (const noteId of entry.notes) {
      noteQstampMap.set(noteId, entry.qon);
    }
  }
}

const { parseScoreIR } = await import("./score-ir");
const ir = parseScoreIR(
  pieceId ?? "",
  pageSvgs,
  entry.measures,
  noteQstampMap,
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
  | { type: "get_page"; requestId: string; pieceId: string; pageN: number; pageWidth?: number }
  | { type: "get_clip"; requestId: string; pieceId: string; startBar: number; endBar: number }
  | { type: "get_ir";   requestId: string; pieceId: string };

// In the dispatch switch (replacing the existing render_full/render_clip if/else):
if (msg.type === "get_clip") {
  const svg = processRenderClipRequest(tk, measures, msg.startBar, msg.endBar);
  (self as unknown as Worker).postMessage({ requestId: msg.requestId, payload: svg });
} else if (msg.type === "get_page") {
  // If a custom pageWidth is supplied (e.g. from the sandbox's responsive-width logic),
  // re-render that page with the adjusted width; otherwise serve from the pre-rendered cache.
  let svg: string | "failed";
  if (msg.pageWidth !== undefined && msg.pageWidth !== result.ir.pageWidth) {
    const tk = result.tk;
    tk.setOptions({ pageWidth: msg.pageWidth });
    // redoLayout is required after every layout-changing setOptions call before renderToSVG.
    // Without it, Verovio renders with the old layout geometry (silently wrong-sized).
    tk.redoLayout({});
    const rendered = tk.renderToSVG(msg.pageN) as string;
    // Restore original pageWidth so future pre-rendered cache reads remain consistent.
    tk.setOptions({ pageWidth: result.ir.pageWidth });
    tk.redoLayout({});
    svg = rendered || "failed";
  } else {
    svg = processGetPageRequest(result.pageSvgs, msg.pageN);
  }
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

  async getPage(pieceId: string, pageN: number, pageWidth?: number): Promise<string> {
    const worker = this.ensureWorker();
    return new Promise((resolve, reject) => {
      const requestId = `req-${++this.requestCounter}`;
      this.pendingRequests.set(requestId, {
        resolve: resolve as (v: unknown) => void,
        reject,
        pieceId,
      });
      worker.postMessage({ type: "get_page", requestId, pieceId, pageN, pageWidth });
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
- Modify: `apps/web/src/routes/app.sandbox.tsx` (lines 537 and 577)

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

4. In `apps/web/src/routes/app.sandbox.tsx`, update both `getFull` call sites:

   At line 537 (initial load — no pageWidth), replace:
   ```typescript
   scoreRenderer
     .getFull(pieceId)
     .then((s) => {
   ```
   with:
   ```typescript
   scoreRenderer
     .getPage(pieceId, 1)
     .then((s) => {
   ```

   At line 577 (drag-end re-render with responsive width), replace:
   ```typescript
   scoreRenderer
     .getFull(pieceId, Math.round(w / 0.4))
     .then(setSvg)
   ```
   with:
   ```typescript
   scoreRenderer
     .getPage(pieceId, 1, Math.round(w / 0.4))
     .then(setSvg)
   ```

   The optional third argument `pageWidth` on `getPage` is supported by the updated `get_page` worker handler (Task 3 Step 3): when `pageWidth` differs from the cached IR's `pageWidth`, the worker re-renders that page with the requested width and restores the original options afterward.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/web && bun run test src/lib/score-worker.test.ts
```

Expected: PASS — all unit tests pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai && git add apps/web/src/components/ScorePanel.tsx apps/web/src/lib/score-worker.test.ts apps/web/src/routes/app.sandbox.tsx && git commit -m "fix(score-panel,sandbox): call getPage replacing removed getFull; sandbox uses optional pageWidth arg; update unit tests"
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

      // Per-note qstamp regression: for every bar with >= 2 notes, assert that either
      // (a) the notes have at least 2 distinct qstamp values (non-chord bar), or
      // (b) all notes share a single qstamp value (valid chord at one onset position).
      // What must NOT happen: all notes collapsed to qstampStart regardless of actual onset.
      // Verify at least one bar in the fixture has >= 2 distinct qstamps (catches the regression).
      const barsWithMultipleOnsets = entry.ir.bars.filter((bar) => {
        if (bar.noteIds.length < 2) return false;
        const qstamps = new Set(bar.noteIds.map((id) => entry.ir.notes[id]?.qstamp ?? -1));
        return qstamps.size >= 2;
      });
      expect(barsWithMultipleOnsets.length).toBeGreaterThan(0);
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
| `app.sandbox.tsx:537` getFull → getPage(_, 1) | Task 5 |
| `app.sandbox.tsx:577` getFull(_, pageWidth) → getPage(_, 1, pageWidth) | Task 5 |
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

---

## Challenge Review

### CEO Pass

#### 1. Premise Challenge

**Right problem?** Yes. The current codebase has no addressable note geometry on the main thread — every cursor attempt would require re-parsing SVG per rAF frame. The IR approach is the right abstraction. No simpler alternative achieves a 60fps cursor.

**Real pain?** Confirmed. The spec documents exactly what breaks without this: cursor cannot be drawn, note-level features (highlight, click-to-seek) are blocked, and `getFull` misleadingly renders only page 1 while its name implies multi-page.

**Direct path?** Yes. The plan is tightly scoped to the stated goal. The spike gate (Task 0) is appropriate risk mitigation for the one unvalidated assumption (eager-IR load time at Ballade scale).

**Existing coverage?** `score-worker.ts` already exports `processRenderClipRequest` and `renderFullSvg` as testable pure functions — the plan correctly extends this pattern for `processGetPageRequest`. `score-renderer.ts` already uses the `bytesCache`/`sentPieceIds` deduplication pattern — the plan reuses it correctly.

#### 2. Scope Check

**Missing callers — scope gap (BLOCKER):** The plan updates `ScorePanel.tsx` (1 caller of `getFull`) but `app.sandbox.tsx` calls `scoreRenderer.getFull(pieceId)` at lines 537 and 577. The call at line 577 passes a custom `pageWidth` parameter (`getFull(pieceId, Math.round(w / 0.4))`). The new `ScoreRenderer` plan has no `pageWidth` parameter on `getPage`. After Task 4 removes `getFull`, the sandbox will fail TypeScript compilation. The plan's File Changes table does not list `app.sandbox.tsx`.

**pageWidth capability gap (BLOCKER — related):** The current `render_full` worker message accepts an optional `pageWidth` and applies it via `tk.setOptions({ pageWidth })` before rendering. The new API (`getPage(pieceId, n)`) carries no `pageWidth` parameter. This capability is silently dropped without any migration path for the sandbox's responsive-width use case. If the sandbox is intentionally descoped, the plan must explicitly name it and either (a) add `app.sandbox.tsx` to the file changes table with the updated call, or (b) document the decision to break the sandbox.

**What could be deferred:** `get_ir` as a worker message (in addition to the main-thread cache) is speculative — the spec names it as "a fallback path" for renderer re-instantiation, which is not a concrete use case in the current codebase. Keeping IR only on the main thread (populated from `load()`'s resolved payload) simplifies the protocol by one message type and one round-trip without breaking any current consumer.

#### 3. Twelve-Month Alignment

```
CURRENT STATE                THIS PLAN                           12-MONTH IDEAL
SVG-only worker protocol  →  IR + cursor module                 →  Note-level features,
(render_full, render_clip)   (IR cache, 4-msg protocol,              annotations, playback
No note geometry             cursor rAF loop)                        overlay, click-to-seek
```

This plan moves squarely toward the 12-month ideal. No tech debt is introduced that conflicts with the future direction. The `NoteIR.bbox.w/h = 0` limitation is correctly documented as a known constraint (click-to-select deferred to after main-thread `getBBox`).

#### 4. Alternatives Check

The spec explicitly documents the major decision points (eager vs lazy IR, SVG-text parse vs getBBox, cursor DOM ownership, ship strategy) with rationale in the Key Decisions table. No additional question is required here.

---

### Engineering Pass

#### 5. Architecture

Data flow for the new load path:
```
api.scores.getData(pieceId) → ArrayBuffer → worker.postMessage({type:'load', bytes})
  → loadPiece() → [loadZipDataBuffer | extractXmlFromMxl + loadData]
  → buildMeasureIndex() → renderToSVG(n) × pages → parseScoreIR()
  → {ir, pageSvgs} → worker.postMessage({requestId, payload: {ir, pageSvgs}})
  → ScoreRenderer.onmessage → irCache.set(pieceId, ir) → load() resolves
```

Data flow for rAF cursor tick:
```
qstampSource() → q
  → findBar(q) [binary search ir.bars] → bar
  → interpolateX(bar, q) [linear interp notes] → x (SVG-local coords)
  → overlay<line>.setAttribute('x1', x) [direct DOM mutation]
```

**Protocol breaking change correctly handled:** The worker currently sends `{ requestId, svg }` for render responses. The plan changes this to `{ requestId, payload }` in Task 3 (worker side) and Task 4 (renderer side). These are in consecutive task groups (B then C), so the in-between state has a broken protocol for one commit. This is acceptable in a single-PR build (no deployment between Task 3 and Task 4) but the build agent must be aware that the system is intentionally broken between Group B and Group C commits.

**Security:** SVG is produced entirely by Verovio WASM from controlled MXL files (not user-supplied HTML). The `insertAdjacentHTML` in ScorePanel.tsx already has a biome-ignore for this reason. No new user-input-to-DOM path is introduced.

**Scaling:** IR build is O(notes × pages) at load time. At Ballade scale (~1291 notes, estimated), the spec cites 25ms extrapolated — acceptable. No N+1 or fan-out patterns.

#### 6. Module Depth Audit

- **score-ir.ts** — Interface: 1 exported function + 4 exported types. Implementation: SVG regex extraction, qstamp-from-measureOn lookup, staff inference, bar bbox aggregation, page dimension parsing. **DEEP.**
- **score-worker.ts** (modified) — Interface: 4-message protocol + 3 exported functions (`loadPiece`, `processRenderClipRequest`, `processGetPageRequest`). Implementation: WASM init, MXL ZIP parsing, multi-page render, IR build, cache/deduplication logic. **DEEP.**
- **score-renderer.ts** (rewritten) — Interface: 4 public methods. Implementation: worker lifecycle, fetch deduplication, postMessage protocol, request-ID generation, IR main-thread cache. **DEEP.**
- **score-cursor.ts** — Interface: 2 public methods (`start`, `stop`). Implementation: rAF loop, binary search, interpolation, per-page overlay DOM management, page-cross scroll. **DEEP.**

#### 7. Code Quality

**Catch-all in `ScoreRenderer.load()` (confidence: 9/10):** The plan's Task 4 implementation has:
```typescript
} catch {
  return "failed";
}
```
This is a bare catch-all that silently swallows all errors from the `sendRequest` call — including worker crashes, network failures, and programming errors. Per CLAUDE.md: "Explicit exception handling over silent fallbacks." The caught error should at minimum be logged (or captured to Sentry) before returning `"failed"`.

**`NoteIR.qstamp` is always `qstampStart`, not onset qstamp (confidence: 9/10):** In `parseScoreIR`, every note in a bar is assigned `qstamp = qstampStart` (the bar's start qstamp). The comment says "Intra-bar interpolation is done by the cursor; notes all get qstampStart here." But the test `FAKE_IR` in `score-cursor.test.ts` has notes with *individual* qstamps (`n1: qstamp=0, n2: qstamp=2, n3: qstamp=4, n4: qstamp=6`). The cursor's `interpolateX` sorts notes by `qstamp` and linearly interpolates between them — which works correctly only if notes have distinct onset qstamps. If all notes in a bar get `qstampStart`, then every note in the bar has the same qstamp, the sort is a no-op, and the interpolation degenerates: `prev === next` for all but the first two notes, returning `prev.bbox.x` always. The cursor will snap to the leftmost note's x for the entire bar duration rather than tracking across notes. The test fixture uses hand-crafted distinct qstamps and will pass, but the real IR will have collapsed qstamps and the cursor will exhibit step-function behavior rather than smooth interpolation.

**DRY — `makeBindings()` helper (confidence: 8/10):** The Verovio WASM initialization block (`import verovio/esm`, `import verovio/wasm`, init) appears identically in Task 0, Task 3, Task 6, and Task 8 integration tests. The plan introduces a `makeBindings()` helper in Task 6, but Tasks 0 and 3 don't use it — they inline the same 5 lines. This is a minor inconsistency (3 inlined, 1 extracted) that the build agent should normalize.

#### 8. Test Philosophy Audit

All tests exercise public interfaces (exported functions, class methods). No private methods are directly tested. External boundaries (Worker constructor, `api.scores.getData`) are mocked in unit/integration tests at appropriate levels.

**Task 7 Sentry test mocks `./sentry` (confidence: 7/10):** The test uses `vi.doMock("./sentry", () => ({ Sentry: sentryMock }))` combined with `vi.resetModules()` in `beforeEach`. This is an acceptable pattern for testing error-capture behavior at a boundary. The Sentry module is an external boundary (side-effect: error reporting), not an internal collaborator. No flag raised.

**Task 4 score-renderer.test.ts — mock Worker pattern (confidence: 7/10):** The test stubs `Worker` globally and simulates the worker response by calling `simulateWorkerResponse` manually. This tests the renderer's Promise orchestration against the worker protocol contract without requiring a real worker. The behavior being tested (load resolves with IR, getIR returns synchronously after load) is correctly behavior-level. No flag raised.

#### 9. Vertical Slice Audit

Tasks 1–9 each follow the one-test → one-impl → one-commit structure with two exceptions:

**Task 0 watch-it-fail violation (confidence: 9/10):** Task 0's Step 2 instructs the build agent to "verify it FAILS" but the plan's own text immediately below says "the test may actually pass the timing assertion." The test as written exercises the *current* `loadPiece` which does not build IR — and simply asserts `elapsed < 200ms`. This test can PASS before any implementation exists (current `loadPiece` already runs within 200ms for the Ballade). The "watch it fail first" discipline is broken: the test is a timing gate, not a behavioral assertion that fails because the feature doesn't exist yet. This means the spike gate provides no build-agent signal that "this test fails because the feature is missing" — it only fails if the timing threshold is violated. This is acceptable for a spike (the plan acknowledges it), but the task's Step 2 language should say "verify it RUNS and note elapsed time" rather than "verify it FAILS."

**Task 6 "eviction" test mislabeled (confidence: 9/10):** Task 6's "Cache eviction" describe block calls `loadPiece()` directly — a stateless pure function — not through the worker's `onmessage` handler. `loadPiece()` has no cache. `toolkitCache` lives inside the `if (typeof window === "undefined")` block and is never exercised. This test verifies that two different ArrayBuffers produce different IR note keysets (a trivially true property of the pure function), not that the worker's `toolkitCache` correctly evicts a stale entry when the same `pieceId` is loaded with new bytes. The real eviction invariant (stale IR is not returned from cache after re-load) is untested. The test title is misleading, and the actual worker-cache eviction behavior (which the spec explicitly lists as a design decision) has no test coverage.

#### 10. Test Coverage Gaps

```
[+] score-ir.ts
    │
    ├── parseScoreIR()
    │   ├── [TESTED]  structural invariants (bars.length, noteIds resolve, qstampStart < qstampEnd) ★★
    │   ├── [GAP]     multi-page SVG (all tests use single page)
    │   ├── [GAP]     measure in SVG with no matching entry in measures[] array
    │   └── [GAP]     SVG with no note elements (empty piece)

[+] score-worker.ts (new paths)
    │
    ├── loadPiece() — extended path
    │   ├── [TESTED]  IR built and pageSvgs present after load ★★
    │   ├── [TESTED]  getPageCount() === 0 → "failed" (spec requirement, but no explicit test written)
    │   └── [GAP]     getPageCount() === 0 path has no test in any task

[+] score-cursor.ts
    │
    ├── findBar()
    │   ├── [TESTED]  qstamp within bar range ★★
    │   ├── [TESTED]  qstamp = null → hidden ★★
    │   └── [GAP]     qstamp past last bar's qstampEnd (spec says return last bar — no test)
    │
    ├── interpolateX()
    │   ├── [TESTED]  midpoint interpolation within bar 1 ★★
    │   └── [GAP]     bar with zero notes (returns bar.bbox.x — no test)
    │
    ├── mountOverlays() / unmountOverlays()
    │   ├── [TESTED]  mount/unmount lifecycle ★★
    │   └── [GAP]     multi-page IR (test fixture has 1 page; plan spec says "one overlay per page")

[+] score-renderer.ts (new paths)
    │
    ├── load()
    │   ├── [TESTED]  happy path resolves with {ir, pageSvgs} ★★
    │   ├── [GAP]     worker returns error → load() returns "failed" (no test)
    │   └── [GAP]     api.scores.getData() throws → no test

    ├── getIR()
    │   ├── [TESTED]  synchronous after load ★★
    │   └── [TESTED]  null before load ★★

    ├── getPage()
    │   └── [GAP]     no behavior test (only tested as part of ScorePanel migration)

    └── worker cache eviction via worker protocol
        └── [GAP]     real toolkitCache eviction untested (Task 6's test bypasses cache)
```

**[RISK] `getPageCount() === 0` spec requirement has no test:** The spec says "Any failure in any step → resolve with 'failed', no partial cache entry" and specifically: "getPageCount() === 0 → 'failed'." The Spec Coverage Checklist lists this requirement for Task 3, but no test in any task exercises this path.

#### 11. Failure Modes

**Task 3 protocol gap (between Group B and Group C):** After Task 3's commit, the worker sends `{ requestId, payload: svg }` for `get_clip` responses, but `score-renderer.ts` (unchanged until Task 4) still reads `e.data.svg`. Any `getClip` call in this intermediate state returns a rejected promise with "Worker returned no svg and no error." Since this is a single-PR build with no deploy between groups, this is not a user-visible failure, but the build agent must not run integration tests against the live app between Group B and Group C commits.

**Cursor rAF loop after `stop()` (confidence: 8/10):** The `tick` arrow function's catch block calls `this.rafId = requestAnimationFrame(this.tick)` unconditionally at the end of the `try/catch`, AND the normal path also calls `requestAnimationFrame(this.tick)`. If `stop()` is called while a tick is in flight (before `requestAnimationFrame` returns), `rafId` is set to null by `stop()`, then overwritten by the in-flight tick's `requestAnimationFrame` call. This leaves a "ghost" rAF handle that never gets cancelled. The overlay is already unmounted, so `this.hideAll()` on the next tick will try `overlay.querySelector("line")` on a detached node — harmless in most browsers but a source of silent iteration over orphaned elements. The risk is low severity (no crash, no visible artifact) but the loop technically keeps running until garbage collected.

**`parseScoreIR` SVG regex brittle against attribute order (confidence: 6/10):** The `extractNotePositions` regex matches `<g ...class="...note..."... id="...">` and assumes class comes before id in the attribute list. Verovio's SVG output may reorder attributes across versions or platform builds. Verify against actual Verovio 4.x output (the fixture test will catch this if it fails, so this is a RISK not a BLOCKER).

#### 12. Presumption Inventory

| Assumption | Verdict | Reason |
|---|---|---|
| `api.scores.getData` returns `ArrayBuffer` | SAFE | Verified in `apps/web/src/lib/api.ts` lines 401–411 |
| `sentry.ts` exports `{ Sentry }` | SAFE | Verified: `apps/web/src/lib/sentry.ts` exports `{ Sentry }` re-exported from `@sentry/react` |
| Score fixtures exist at declared paths | SAFE | Verified: all 3 MXL files present in `apps/web/public/scores/` |
| `scoreRenderer.getFull` is called only in `ScorePanel.tsx` | RISKY | False: also called in `app.sandbox.tsx` at lines 537 and 577, one with `pageWidth` arg |
| Removing `render_full` / `render_clip` messages breaks no other callers | RISKY | `app.sandbox.tsx` uses `getFull` (which sends `render_full`); not in plan's file changes |
| All notes in a bar get distinct qstamps from the timemap | RISKY | `parseScoreIR` sets all notes in a bar to `qstampStart`; cursor interpolation requires per-note onset qstamps to produce smooth cursor movement |
| `processGetPageRequest` export added in Task 2 doesn't require CacheEntry.pageSvgs to exist | SAFE | Task 2 test passes `pageSvgs` directly as a parameter; no CacheEntry dependency |
| Task 6 "eviction test" exercises the worker's `toolkitCache` | RISKY | It calls `loadPiece()` directly — a stateless function; `toolkitCache` is never touched |
| `NoteIR.qstamp` will be populated with onset timing from timemap | RISKY | Current `parseScoreIR` sets all notes to `qstampStart`; onset data is not in the timemap's note entries |
| 200ms Ballade timing threshold holds on CI runners (not just dev machine) | VALIDATE | CI hardware typically slower; spike confirmed on dev machine (extrapolated 25ms IR) but not on actual CI |

---

### Summary

**[BLOCKER] count: 3**
**[RISK]    count: 6**
**[QUESTION] count: 0**

---

**[BLOCKER] (confidence: 9/10) — `app.sandbox.tsx` uses `scoreRenderer.getFull` in two places (lines 537 and 577), one with a custom `pageWidth` argument. The plan removes `getFull` but does not list `app.sandbox.tsx` in its File Changes table. After Task 4's rewrite of `score-renderer.ts`, TypeScript compilation will fail and the sandbox route will be broken. The plan must add `app.sandbox.tsx` to the file changes table and either (a) replace `getFull(pieceId)` with `getPage(pieceId, 1)` for the no-pageWidth call and provide a migration for the `pageWidth`-parameterized call, or (b) explicitly document that the sandbox's responsive-width render is being removed.**

**[BLOCKER] (confidence: 9/10) — `NoteIR.qstamp` is set to `qstampStart` for all notes in a bar. `interpolateX()` sorts notes by `qstamp` and interpolates between them — but if all notes share the same qstamp value, the sort is a no-op and the linear interpolation degenerates: `(q - prev.qstamp) / (next.qstamp - prev.qstamp)` produces 0/0 (NaN) whenever the two bracketing notes have equal qstamps. The cursor will either stick at the bar's leftmost note or produce NaN x-coordinates. The test fixture uses hand-crafted distinct per-note qstamps and will pass, masking this defect. Fix: populate `NoteIR.qstamp` with the actual per-note onset qstamp from the timemap (`tk.renderToTimemap({ includeNotes: true })`), or document that qstamp is bar-level only and implement a position-based interpolation (x interpolated purely by playback time fraction through the bar, not by note onset).**

**[BLOCKER] (confidence: 9/10) — Task 0's Step 2 "verify it FAILS" discipline is broken. The spike test asserts `elapsed < 200ms` against the *current* `loadPiece` (which has no IR build). The current `loadPiece` already runs in under 200ms — so the test passes before any implementation exists. This violates TDD watch-it-fail. A spike whose test passes before implementation provides no confidence that the test actually exercises the new behavior added in Task 3. The plan must either (a) change the Task 0 test to assert something that cannot pass without Task 3's changes (e.g., assert `entry.ir !== undefined`), making it a true failing test that becomes a passing test after Task 3, or (b) honestly rename Task 0's Step 2 as "verify it RUNS (not FAILS) and record elapsed time" and remove the watch-it-fail claim. As written, the build agent will skip the watch-it-fail discipline and proceed on a green test that proves nothing about future Task 3 behavior.**

---

**[RISK] (confidence: 9/10) — `ScoreRenderer.load()` has a bare `catch { return "failed"; }` that swallows all errors silently. Per CLAUDE.md: "Explicit exception handling over silent fallbacks." At minimum, the caught error should be passed to `Sentry.captureException` before returning `"failed"`. This is a production observability gap.**

**[RISK] (confidence: 9/10) — Task 6's "Cache eviction" test calls `loadPiece()` directly (a stateless pure function) and never exercises `toolkitCache`. The test title is misleading and the actual eviction invariant — that the worker's in-memory `toolkitCache` correctly drops the stale `CacheEntry` (and its IR) when the same `pieceId` is reloaded with new bytes — has no test coverage. The spec identifies cache eviction as an explicit design decision. A true eviction test should exercise the worker's `onmessage` handler via postMessage with bytes twice for the same pieceId.**

**[RISK] (confidence: 8/10) — `getPageCount() === 0 → "failed"` is listed in the Spec Coverage Checklist as Task 3's responsibility, but no test in any task explicitly exercises this path. The path is critical (zero-page load is indistinguishable from corruption) and irreversible (a `"failed"` result is cached). Add a unit test in Task 3 or Task 6 that supplies a `loadZipDataBuffer` mock returning `true` but a `getPageCount` mock returning `0`, and asserts that `loadPiece` returns `"failed"`.**

**[RISK] (confidence: 8/10) — `ScoreCursor.tick` has a rAF ghost-loop risk: if `stop()` is called while a tick is in-flight (after `qstampSource()` is called but before the final `requestAnimationFrame(this.tick)`), the in-flight tick overwrites `rafId` with a new handle that is never cancelled. The overlays are unmounted so no visual artifact occurs, but the loop burns rAF budget until the `ScoreCursor` object is garbage collected. Fix: check `this.rafId !== null` at the top of `tick` before re-scheduling.**

**[RISK] (confidence: 8/10) — The 200ms Ballade timing threshold (Task 0 gate, Task 8 final verification) was derived from a dev machine spike (25ms IR walk extrapolated). CI runners are typically 2–4x slower. The threshold has not been validated on the actual CI hardware. If CI is significantly slower, Task 8 will be a flaky gate. Consider raising the threshold to 500ms for CI, or measuring on CI before committing to 200ms.**

**[RISK] (confidence: 6/10) — `parseScoreIR`'s SVG regex for note extraction assumes attribute order (`class` before `id`, `x` before `y` on `<use>`). Verovio 4.x has been observed to maintain consistent attribute order, but this assumption is undocumented and could silently produce empty IR if Verovio changes its serialization. The integration tests with real fixtures will catch this; if they pass, the risk is low. Verify once against actual rendered SVG from the Nocturne fixture before closing.**

---

VERDICT: NEEDS_REWORK — Three blockers must be resolved before execution: (1) `app.sandbox.tsx` is an unscoped caller of the removed `getFull` method that will break TypeScript compilation; (2) `NoteIR.qstamp` is always set to `qstampStart` causing cursor interpolation to degenerate on real IR data; (3) Task 0's watch-it-fail discipline is broken — the spike test passes before implementation, giving the build agent no signal about Task 3 behavior.

---

## Challenge Review — Pass 2 (2026-05-26)

> Re-review after three targeted fixes: (1) `app.sandbox.tsx` added to Task 5 with `getPage` migration + optional `pageWidth` arg; (2) `NoteIR.qstamp` now sourced from `tk.renderToTimemap({includeNotes:true})` via `noteQstampMap` passed into `parseScoreIR`; (3) Task 0 now asserts `entry.ir !== undefined` so the test genuinely fails before Task 3.
> All source files read before forming opinions.

### CEO Pass

#### 1. Premise Challenge

No change from Pass 1. Right problem, real pain, direct path. Fixes address the scope gap and data model correctness issues that were blockers. No new strategic concerns.

#### 2. Scope Check

**`app.sandbox.tsx` is now scoped (Pass 1 BLOCKER resolved):** Task 5 adds `app.sandbox.tsx` to its file list and the commit step stages it. Both `getFull` call sites are migrated: line 537 → `getPage(pieceId, 1)`, line 577 → `getPage(pieceId, 1, Math.round(w / 0.4))`. The optional `pageWidth` argument is wired through `getPage` → `get_page` worker message → Task 3's conditional re-render path. Verified against actual sandbox code at lines 536-577.

**`get_page` handler missing `redoLayout` on pageWidth re-render (NEW BLOCKER):** Task 3's `get_page` handler re-renders with a custom pageWidth via:
```typescript
tk.setOptions({ pageWidth: msg.pageWidth });
const rendered = tk.renderToSVG(msg.pageN) as string;
tk.setOptions({ pageWidth: result.ir.pageWidth });
```
`tk.setOptions()` changes the Verovio options but does **not** trigger a layout reflow. Without `tk.redoLayout({})` after `setOptions`, `renderToSVG` returns the SVG of the previously-computed layout at the old dimensions — the output will be mis-sized. The existing `renderClipSvgSelect` in the current codebase follows the correct sequence: `setOptions` → `redoLayout` → `renderToSVG` → `select({})` → `setOptions(original)` → `redoLayout`. The plan omits `redoLayout` calls in this new path, breaking the sandbox's responsive-width use case which was specifically called out as a reason to preserve the `pageWidth` capability. This is a functional correctness bug in production-visible behavior.

#### 3. Twelve-Month Alignment

No change from Pass 1. The plan still moves toward the 12-month ideal.

#### 4. Alternatives Check

No change from Pass 1.

---

### Engineering Pass

#### 5. Architecture

**`noteQstampMap` fix verified (Pass 1 BLOCKER resolved):** `parseScoreIR` now takes `noteQstampMap: Map<string, number>` as its 4th parameter. Task 3's `loadPiece` builds it from `tk.renderToTimemap({ includeNotes: true })`, iterating entries whose `notes` array carries note element ids and mapping each to `entry.qon`. The Task 1 test uses `SYNTHETIC_NOTE_QSTAMP_MAP` with distinct per-note onsets. The Task 6 regression check verifies at least one bar in each fixture has ≥2 distinct qstamp values. The degenerate-interpolation BLOCKER from Pass 1 is resolved.

**Spec/plan signature mismatch (`parseScoreIR` parameter count):** The spec (line 107) documents `parseScoreIR(pieceId, pageSvgs, measures, tk.getVersion(), VEROVIO_OPTS.pageWidth)` — 5 parameters. The plan implements `parseScoreIR(pieceId, pageSvgs, measures, noteQstampMap, verovioVersion, pageWidth)` — 6 parameters. This is intentional (the fix added `noteQstampMap`) but the spec was not updated to reflect the new signature. The spec's Module section (line 151) still says the interface hides "qstamp-from-measureOn-via-timemap lookup" but the timemap lookup now happens in the *caller* (`loadPiece`) and the result is passed in as a parameter — so the hiding is partial. This is a documentation divergence, not a build-blocking issue, but the spec should be updated so the module description matches the actual interface.

**Task 0 watch-it-fail verified (Pass 1 BLOCKER resolved):** The test now asserts `expect(entry.ir).toBeDefined()` before the timing assertion. Since the current `loadPiece` returns `{ tk, measures }` without an `ir` field, `entry.ir` is `undefined` and Step 2 genuinely produces `FAIL — entry.ir is undefined`. After Task 3's `loadPiece` update, `entry.ir` is populated and the test turns green. The timing assertion (`expect(elapsed).toBeLessThan(200)`) is the correct final gate. Pass 1 BLOCKER is resolved.

#### 6. Module Depth Audit

No change from Pass 1. All four modules remain DEEP.

#### 7. Code Quality

**`ScoreRenderer.load()` bare catch (carry-forward RISK from Pass 1):** The bare `catch { return "failed"; }` in Task 4's `ScoreRenderer.load()` implementation is unchanged. The error is still swallowed without logging. Per CLAUDE.md, explicit exception handling is preferred over silent fallbacks. Sentry capture should be added.

**Duplicate `processGetPageRequest` describe blocks in `score-worker.test.ts` (NEW RISK):** Task 2 writes a `describe("processGetPageRequest")` block with 2 test cases (page=2, page=99). Task 5 also writes a `describe("processGetPageRequest")` block with 2 test cases (page=0, page=5) and says it "replaces the `renderFullSvg` describe block." The plan does not instruct Task 5 to remove Task 2's describe block. After both tasks complete, `score-worker.test.ts` will have two separate `describe("processGetPageRequest")` blocks. Vitest runs both, producing 4 tests under the same describe name — confusing but not a test failure. The build agent should remove Task 2's block when adding Task 5's, or consolidate into one.

**`noteQstampMap.get(id) ?? 0` silent fallback:** In `parseScoreIR`, notes not found in the timemap fall back to `qstamp = 0`. For a note genuinely absent from the timemap, `0` is indistinguishable from a bar-1-beat-1 note onset and will cause cursor misplacement silently. The plan comments this as "should not occur for well-formed Verovio output but is explicit, not silent" — the fallback is `0`, not a thrown exception. Per CLAUDE.md explicit exception handling rule, this deserves at minimum a `console.error` log so the anomaly is visible.

#### 8. Test Philosophy Audit

No new concerns. All tests exercise public interfaces. The Sentry mock in Task 7 correctly targets an external boundary.

#### 9. Vertical Slice Audit

Tasks 1–9 each follow one-test → one-impl → one-commit. Task 0 watch-it-fail is now genuine. No new horizontal slicing detected.

#### 10. Test Coverage Gaps

The coverage gaps from Pass 1 carry forward. New gaps introduced by the fixes:

```
[+] score-worker.ts — get_page handler (Task 3 new path)
    │
    ├── pageWidth === undefined
    │   └── [TESTED] served from pageSvgs cache — covered by processGetPageRequest tests ★★
    │
    └── pageWidth !== undefined (responsive re-render path)
        ├── [GAP] no test that the re-rendered SVG reflects the new width
        └── [GAP] no test that options are restored after re-render

[+] score-ir.ts — parseScoreIR
    └── noteQstampMap absent for a note id
        └── [GAP] fallback to 0 — no test, no log (silent anomaly)
```

**[RISK] `getPageCount() === 0` still has no test** (carry-forward from Pass 1).

#### 11. Failure Modes

**`get_page` responsive re-render without `redoLayout` produces wrong-sized SVG:** The in-flight Verovio toolkit has a committed layout at `VEROVIO_OPTS.pageWidth`. Calling `setOptions({ pageWidth: msg.pageWidth })` changes the option value but the layout engine has not re-run. `renderToSVG(n)` will return SVG with the old layout geometry at the new nominal width. The `viewBox` may not match. For the sandbox drag-resize use case, this produces a garbled or unstretched score. This is a behavior regression from the current `getFull` implementation which also skips `redoLayout` — but the current codebase's `render_full` handler likewise omits `redoLayout` after `setOptions({ pageWidth })`, so this is a pre-existing bug that the plan reproduces, not a new regression. Flagged for awareness.

**All other failure mode analysis carries forward from Pass 1.**

#### 12. Presumption Inventory

| Assumption | Verdict | Reason |
|---|---|---|
| `app.sandbox.tsx` getFull→getPage migration covers all callers | SAFE | Grep confirms exactly 2 `getFull` calls in sandbox (lines 537, 577); ScorePanel has 1 (line 276); all 3 are in the plan |
| `tk.renderToTimemap({ includeNotes: true })` returns entries with `notes: string[]` of element ids | VALIDATE | API shape assumed from Verovio docs; not verified against real output in any existing test. Integration test (Task 3) will catch if wrong, but the build agent should verify the timemap shape against actual Verovio 4.x output before writing production code |
| `tk.renderToSVG(n)` without `redoLayout` after `setOptions` returns a correctly-reflowed SVG | RISKY | Existing codebase evidence (`renderClipSvgSelect`) shows `redoLayout` is required after `setOptions`. Task 3's `get_page` handler omits this call for the responsive-width path |
| `noteQstampMap.get(id) ?? 0` fallback never triggers on well-formed Verovio output | VALIDATE | No existing test covers the absence case; integration tests will pass only if timemap coverage is complete for all rendered notes |
| Two `describe("processGetPageRequest")` blocks in one test file don't conflict | SAFE | Vitest runs both; no assertion conflict since they test different page numbers |
| 200ms Ballade threshold holds on CI | VALIDATE | Carry-forward from Pass 1 — not validated on CI hardware |

---

### Summary

**[BLOCKER] count: 1**
**[RISK]    count: 7**
**[QUESTION] count: 0**

---

**[BLOCKER] (confidence: 9/10) — Task 3's `get_page` handler re-renders with a custom `pageWidth` via `tk.setOptions({ pageWidth })` followed immediately by `tk.renderToSVG(n)`, without calling `tk.redoLayout({})` between them. Verovio does not reflow on `setOptions` alone — the layout engine must be triggered explicitly, as the existing `renderClipSvgSelect` function demonstrates (`setOptions` → `redoLayout` → `renderToSVG`). Without `redoLayout`, the SVG returned by `renderToSVG` reflects the previously-computed layout at the original `pageWidth`, not the requested one. The sandbox's drag-to-resize use case (`getPage(pieceId, 1, Math.round(w / 0.4))`) will silently produce wrong-sized SVG. Fix: add `tk.redoLayout({})` after `tk.setOptions({ pageWidth: msg.pageWidth })` and again after restoring the original options, matching the pattern in `renderClipSvgSelect`.**

---

**[RISK] (confidence: 8/10) — Task 2 writes a `describe("processGetPageRequest")` block and Task 5 writes a second one without removing Task 5's. After both tasks, `score-worker.test.ts` has two `describe("processGetPageRequest")` blocks. Vitest runs both without error, but the duplication is confusing and signals a plan ordering issue. The plan should explicitly instruct Task 5's Step 3 to remove Task 2's describe block when replacing `renderFullSvg`.**

**[RISK] (confidence: 9/10) — `ScoreRenderer.load()` has a bare `catch { return "failed"; }` that swallows all errors silently. Per CLAUDE.md: "Explicit exception handling over silent fallbacks." The caught error should at minimum be passed to `Sentry.captureException` before returning `"failed"`. Carry-forward from Pass 1.**

**[RISK] (confidence: 9/10) — Task 6's "Cache eviction" test calls `loadPiece()` directly — a stateless pure function — and never exercises `toolkitCache`. The test title is misleading. The actual worker-cache eviction invariant (stale IR not returned after re-load under the same pieceId) has no test. Carry-forward from Pass 1.**

**[RISK] (confidence: 8/10) — `getPageCount() === 0 → "failed"` is listed in the Spec Coverage Checklist but has no test in any task. Carry-forward from Pass 1.**

**[RISK] (confidence: 8/10) — `ScoreCursor.tick` ghost-loop risk if `stop()` races an in-flight tick. `rafId` can be overwritten after `stop()` sets it to null. Carry-forward from Pass 1.**

**[RISK] (confidence: 8/10) — The 200ms Ballade timing threshold has not been validated on CI hardware. Carry-forward from Pass 1.**

**[RISK] (confidence: 7/10) — `tk.renderToTimemap({ includeNotes: true })` shape (specifically that `notes` is an array of element id strings on each timemap entry) is assumed from Verovio docs but not verified against actual Verovio 4.x WASM output. If the field name or type differs, `noteQstampMap` will be empty and all notes will silently fall back to qstamp=0 — exactly the degenerate behavior the fix was intended to prevent. The Task 3 integration test will catch this if it actually checks per-note qstamp diversity, which it does (the Task 6 regression check). This is a VALIDATE assumption, not a guaranteed risk, but worth explicitly verifying the timemap shape before writing the production code.**

---

VERDICT: NEEDS_REWORK — One new blocker: Task 3's `get_page` responsive-rerender path calls `tk.setOptions({ pageWidth })` without `tk.redoLayout({})`, returning a mis-sized SVG for the sandbox drag-resize use case. Fix requires adding `redoLayout` calls around the re-render (matching the `renderClipSvgSelect` pattern) before execution proceeds.
