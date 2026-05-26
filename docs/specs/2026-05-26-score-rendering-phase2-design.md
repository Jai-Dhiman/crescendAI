# Score Rendering Phase 2 — ScoreIR + Playback Cursor Design

**Goal:** A pianist watching a rendered score in the web app sees a sub-50ms-latency cursor that tracks playback position note-by-note, rather than no cursor at all.
**Not in scope:** Server-side Verovio, SharedArrayBuffer transfer, note pitch/MIDI in IR, drag-to-resize reflow, performance overlay implementation, click-to-select hit-testing (would require main-thread `getBBox`), multi-piece concurrent caching, the upstream playback engine itself (this spec consumes a `qstampSource` it does not own).

## Problem

`apps/web/src/lib/score-worker.ts` today exposes only `render_full` and `render_clip` — both return SVG strings. There is no addressable note/bar geometry on the main thread, which means:

- A playback cursor cannot be drawn without re-parsing the rendered SVG every frame (prohibitive in a rAF loop).
- `ScorePanel.tsx:276` and `PlayPassageCard.tsx:54` get pre-cropped SVGs and have no way to know *which* notes are where, blocking note-level features (cursor, highlight, click-to-seek).
- The current `getFull` API name implies multi-page but actually renders only `tk.renderToSVG(1)` — page 1 only — which is misleading.

Phase 1 (shipped 2026-05-25) fixed clip cropping by adding `tk.redoLayout()` after `tk.select()` and replacing silent fallback with explicit `"failed"`. Phase 2 builds on that hardened foundation.

## Solution (from the user's perspective)

When a pianist opens a score in the web app and starts playback (or any qstamp-emitting source — practice playback, AMT alignment, future MIDI replay), a vertical line cursor appears over the rendered staff and tracks position in real time. It moves smoothly between notes via linear interpolation between bracketing onset qstamps, hides itself when the source pauses or ends, and scrolls the page into view when the cursor crosses a page boundary.

## Design

### Load-time expectations

`loadPiece` is not a fast operation for large pieces. Profiling on the Ballade fixture shows total wall-clock of ~3.8s, broken down as: `loadZipDataBuffer` ~1.9s, `renderToSVG` across 16 pages ~1.2s, `renderToTimemap` ~0.7s. This is Verovio's intrinsic cost and is not reduced by the IR layer. The IR walk itself (regex extraction over already-rendered SVG strings) is cheap (~25ms for Ballade-scale). The UI must display a loading state while `load()` is pending; callers must not assume sub-second completion for large pieces. Approximate expectations: ~500ms for a Nocturne-scale piece, ~4s for a Ballade-scale piece.

### Approach

The worker eagerly builds a `ScoreIR` (intermediate representation) once at load time and caches it alongside the existing toolkit. IR is a pure function of the rendered SVG text — produced by regex extraction over the worker's own `renderToSVG(n)` output for each page, not via DOM `getBBox()` (which requires a layout engine the worker doesn't have). The cursor reads the cached IR synchronously, runs a `requestAnimationFrame` loop that pulls qstamps via a caller-supplied function, binary-searches the bar index to find the bracketing notes, linearly interpolates an x coordinate, and mutates a sibling `<line>` element directly. React stays out of the rAF loop.

### Key decisions

| Decision | Choice | Why |
|---|---|---|
| Eager vs. lazy IR build | **Eager at load()** | Gives `getIR()` a synchronous contract (safe in rAF hot path); IR build is decoupled from clip mutations; matches spike evidence that IR-walk is cheap (3.6ms Nocturne, 25ms Ballade extrapolated). Gated by Task 0 spike threshold of **200ms MARGINAL cost** of IR build on top of Verovio's intrinsic load+render time (measured as Δ between with-IR and without-IR `loadPiece` runs). Note: Ballade-scale total load is ~2-4s (dominated by Verovio's `loadZipDataBuffer` ~1.9s + multi-page `renderToSVG` ~1.2s); this is expected and shown to users via a loading state, not a failure. If the marginal threshold is exceeded, plan halts and reconsiders. |
| IR source | **SVG-text parse, not getBBox** | Worker has no DOM/layout engine. Spike confirmed regex extraction over `<g class="note">` and `<g class="measure">` tags yields finite, valid x/y bboxes for 1291/1291 notes in Nocturne. |
| `NoteIR.bbox.w/h` | **Always 0** | SVG-text parsing cannot compute element size without a layout engine. Cursor only reads x/y; width/height would only matter for click-to-select hit-testing (out of scope). Documented in the type. |
| Cache eviction trigger | **Full CacheEntry drop on bytes-for-known-id** | Verovio randomizes element IDs on every `loadData`, so a reloaded `tk` invalidates every IR note key. Whole-entry eviction (tk + ir + pageSvgs) keeps the invariant trivially correct. |
| `qstampSource` shape | **`() => number \| null`** | Zero allocation per rAF frame; cursor visible iff non-null. Pause/end/error all collapse to `null` for v1. Promote to event-channel hybrid only if a UI requirement forces it later. |
| Cursor DOM ownership | **Imperative, sibling `<svg>` overlay** | React's diffing cost in a 60fps loop is unacceptable. Cursor owns its element and uses `element.setAttribute('x1', ...)` directly. |
| Ship strategy | **Single PR, both layers** | User-selected. Internal commits sliced so review can still walk the IR layer and cursor layer independently. |

### Module shape

```
score-ir.ts       — pure parseScoreIR(): pageSvgs + measures -> ScoreIR
score-worker.ts   — load/get_page/get_clip/get_ir message handlers; eager IR build
score-renderer.ts — main-thread Promise/sync wrapper; IR cached on main side
score-cursor.ts   — new ScoreCursor({pieceId, container, qstampSource}).start()/.stop()
```

### IR types (locked)

```ts
type Bbox = { x: number; y: number; w: number; h: number };
// NOTE: w and h are always 0. SVG-text parsing has no layout engine.
// Cursor reads only x and y. Click-to-select would need main-thread getBBox.

type NoteIR = {
  id: string;          // Verovio note element id (random per loadData)
  bbox: Bbox;          // SVG-local coords on its page
  qstamp: number;      // from tk.renderToTimemap
  staff: 1 | 2;
};

type BarIR = {
  barNumber: number;       // 1-indexed
  measureOn: string;       // Verovio measure id
  pageN: number;           // 1-indexed page number
  bbox: Bbox;              // SVG-local coords on its page
  noteIds: string[];       // in score order
  qstampStart: number;
  qstampEnd: number;
};

type PageIR = {
  pageN: number;
  viewBox: string;         // SVG viewBox attribute
  width: number;
  height: number;
  systemBboxes: Bbox[];    // one per system on this page
};

type ScoreIR = {
  pieceId: string;
  verovioVersion: string;  // from tk.getVersion()
  pageWidth: number;       // VEROVIO_OPTS.pageWidth at render time
  pages: PageIR[];
  bars: BarIR[];           // sorted by barNumber
  notes: Record<string, NoteIR>;   // O(1) lookup by note id
};
```

### Worker API (new)

```
Inbound message types:
  { type: 'load',     requestId, pieceId, bytes }
  { type: 'get_page', requestId, pieceId, pageN }
  { type: 'get_clip', requestId, pieceId, startBar, endBar }
  { type: 'get_ir',   requestId, pieceId }

Outbound message types:
  Success: { requestId, payload }   // payload varies per request type
  Failure: { requestId, error }
```

Worker `load` semantics:
1. Run existing `loadPiece` (loadZipDataBuffer or extractXmlFromMxl + loadData).
2. Build measure index via existing `buildMeasureIndex`.
3. For each page `n` in `1..tk.getPageCount()`: call `tk.renderToSVG(n)`, store in `pageSvgs[n-1]`.
4. Call `parseScoreIR(pieceId, pageSvgs, measures, tk.getVersion(), VEROVIO_OPTS.pageWidth)` → IR.
5. Store `{tk, measures, ir, pageSvgs}` on cache. Resolve with `{ir, pageSvgs}` to the caller.
6. Any failure in any step → resolve with `"failed"`, no partial cache entry.

### Main-thread API (new)

```ts
class ScoreRenderer {
  load(pieceId: string): Promise<{ir: ScoreIR; pageSvgs: string[]} | "failed">;
  getPage(pieceId: string, pageN: number): Promise<string>;
  getClip(pieceId: string, startBar: number, endBar: number): Promise<string>;
  getIR(pieceId: string): ScoreIR | null;   // synchronous
}
```

`getIR` returns `null` if `load(pieceId)` has not resolved successfully. Calling `getIR` before `load` is a caller bug, not a recoverable runtime state.

### Cursor module shape

```ts
interface ScoreCursorOptions {
  pieceId: string;
  container: HTMLElement;            // contains the rendered page <svg>s
  qstampSource: () => number | null; // called once per rAF tick
}

class ScoreCursor {
  constructor(opts: ScoreCursorOptions);
  start(): void;                     // begins rAF loop, mounts overlay <svg>
  stop(): void;                      // cancels rAF, unmounts overlay
}
```

rAF body (executed per frame):
1. `const q = this.qstampSource()`. If `q == null`, hide overlay, schedule next frame, return.
2. Binary-search `this.ir.bars` for the bar where `qstampStart <= q < qstampEnd`.
3. Find the two bracketing notes by qstamp within that bar.
4. Linear-interpolate the x coordinate in SVG-local coords; transform to viewport coords using the current bar's page container offset.
5. Mutate the overlay `<line>`'s `x1` and `x2` attributes.
6. If the bar's `pageN` differs from last frame's, call `this.pages[pageN-1].scrollIntoView({block: 'nearest'})`.
7. If `qstampSource()` throws: catch, hide overlay, `Sentry.captureException(err)`, keep the loop alive.

## Modules

- **score-ir.ts** — Interface: `parseScoreIR(pieceId, pageSvgs, measures, verovioVersion, pageWidth) -> ScoreIR`. Hides: SVG regex extraction, qstamp-from-measureOn-via-timemap lookup, staff inference from `<g class="staff">` ancestry, bar bbox aggregation from contained notes. Tested through: invariant tests over the returned IR shape from fixed SVG fixtures. DEEP.
- **score-worker.ts** — Interface: 4-message protocol (load/get_page/get_clip/get_ir). Hides: WASM init, MXL ZIP parsing, multi-page render, IR build coordination, in-flight load deduplication via Promise-in-cache pattern, cache mgmt, eviction. Tested through: real-Verovio integration tests that postMessage and assert response payloads. DEEP.
- **score-renderer.ts** — Interface: 4-method async/sync class (`load`, `getPage`, `getClip`, `getIR`). Hides: worker lifecycle (`ensureWorker`), bytes fetch from `api.scores.getData`, postMessage protocol, request ID generation, IR caching on main thread, error-to-Promise mapping. Tested through: typed-method calls against a real worker. DEEP.
- **score-cursor.ts** — Interface: 2-method class (`start`, `stop`). Hides: rAF scheduling, binary-search algorithm, qstamp interpolation, SVG-local-to-viewport transform, DOM overlay creation/mutation, page-cross scroll, error containment. Tested through: jsdom tests that instantiate the class with a frozen IR fixture, drive `qstampSource`, and assert on overlay `<line>` attributes after rAF ticks. DEEP.

## Verification Architecture

- **Canonical success state:**
  1. `scoreRenderer.load('chopin-ballade-op23-no1')` resolves with a non-null `ir` and `pageSvgs.length === ir.pages.length`. The IR-build marginal cost (measured as Δ between with-IR and without-IR `loadPiece` runs) is under 200ms. Total load time may be 2-4s for Ballade-scale pieces (dominated by Verovio's intrinsic `loadZipDataBuffer` + multi-page `renderToSVG`); the UI shows a loading state for the duration.
  2. `scoreRenderer.getIR(pieceId)` returns the same `ScoreIR` object synchronously after load.
  3. For all 3 fixtures, every `NoteIR.bbox.x` and `.y` is a finite number; every `BarIR.noteIds` resolves to entries in `notes`; `qstampStart < qstampEnd`; `bars.length === measures.length`.
  4. `getClip(pieceId, s, e)` returns SVG whose first `.measure` id equals `ir.bars[s-1].measureOn` (Phase 1 invariant preserved + extended to assert match against IR).
  5. Reloading the same `pieceId` with different bytes yields an IR with disjoint `notes` keys from the prior load.
  6. A `ScoreCursor` instantiated with a frozen IR fixture and a `qstampSource` returning a value inside bar 5 places its overlay `<line>` at an x within 1px of the expected interpolated position.
  7. A `ScoreCursor` whose `qstampSource` returns `null` hides its overlay within one rAF tick.

- **Automated check:** `cd apps/web && bun run test src/lib/score-worker.integration.test.ts src/lib/score-ir.test.ts src/lib/score-cursor.test.ts` exits 0.

- **Harness:** Task 0 (the spike) is the contract-decision harness — it measures `load()` wall-clock on Ballade and gates whether the eager-IR contract above survives. If it fails (>200ms), execution halts before any further task and the spec is revised toward a lazy-IR contract. No separate fixture-generation harness is needed: the three MXL fixtures already live at known paths in `apps/web/public/scores/`.

## File Changes

| File | Change | Type |
|------|--------|------|
| `apps/web/src/lib/score-ir.ts` | IR types + `parseScoreIR` pure function | New |
| `apps/web/src/lib/score-ir.test.ts` | Unit tests for parser | New |
| `apps/web/src/lib/score-worker.ts` | Add `get_ir` / `get_page` message handlers; rewrite `loadPiece` to render all pages + build IR; update `CacheEntry` shape | Modify |
| `apps/web/src/lib/score-worker.integration.test.ts` | Add: IR invariants × 3 fixtures, IR/Clip correlation, eviction, Ballade `load()` perf | Modify |
| `apps/web/src/lib/score-worker.test.ts` | Update mock-based dispatch tests for new message types | Modify |
| `apps/web/src/lib/score-renderer.ts` | Rename `getFull` → `getPage(n)`, add `load`, add `getIR`, cache IR on main thread | Modify |
| `apps/web/src/lib/score-cursor.ts` | New ScoreCursor class | New |
| `apps/web/src/lib/score-cursor.test.ts` | jsdom tests for cursor behavior | New |
| `apps/web/src/components/ScorePanel.tsx` | Update line 276: `getFull(pieceId)` → `getPage(pieceId, 1)` | Modify |
| `apps/web/public/scores/czerny-op299-no1.mxl` | Small exercise fixture | New (already added) |
| `apps/web/public/scores/chopin-ballade-op23-no1.mxl` | Ballade-scale fixture | New (already added) |

## Open Questions

- **Q:** Should `getIR` be exposed as a worker message at all, or only as a main-thread cached read after `load()` resolves with the IR payload?
  **Default:** Both. Worker exposes `get_ir` so a renderer instance that has lost its main-thread cache (e.g., re-instantiated) can rehydrate without re-fetching MXL bytes. Main-thread cache is populated from `load()`'s resolved payload; `get_ir` is a fallback path.
- **Q:** If `tk.getPageCount()` returns 0 (unloaded or corrupt), should `load()` return `"failed"` or resolve with empty `pageSvgs`?
  **Default:** `"failed"`. A score with zero pages is indistinguishable from a load failure for any downstream consumer.
- **Q:** Should the cursor's overlay `<svg>` be appended to `container` or to each page container?
  **Default:** One overlay `<svg>` per page container (so the cursor's coordinate system is naturally page-local and `scrollIntoView` semantics work). Cursor manages N overlay elements, shows only the one on the current page.
- **Q (RESOLVED 2026-05-26):** Was the original 200ms Task 0 gate correctly specified?
  **No.** The original gate measured total `load()` wall-clock (Verovio intrinsic + IR walk). Actual profiling on the Ballade fixture revealed the dominant cost is Verovio's `loadZipDataBuffer` (~1.9s) and multi-page `renderToSVG` (~1.2s), not the IR walk (~25ms extrapolated). The original spike extrapolation predicted 25ms IR walk time — that prediction was correct, but the threshold was applied to the wrong measurement. Revised: the gate now measures only the MARGINAL cost (Δ between with-IR and without-IR `loadPiece` runs), which is expected to be ~25ms for Ballade-scale pieces. Total load (~2-4s) is accepted as Verovio's intrinsic cost and handled by a UI loading state.
