# Verovio Score Renderer Design

**Goal:** Replace the broken OSMD-based clip rendering with a Verovio Web Worker that renders score excerpts off the main thread, serving all score consumers (inline cards, full-score panel) from a single clean interface.

**Not in scope:**
- R2 SVG pre-render cache (future optimization)
- Exercise score clips (ExerciseSetConfig does not yet have a scoreClip field)
- Keyboard guide or reference browser artifacts
- Score editing or annotation editing
- iOS score rendering
- Verovio cursor/playback integration in the panel

## Problem

`osmd-manager.ts` renders the full score off-screen then calls `clipBars()` to crop the SVG. `clipBars()` returns null — OSMD's `measureList` bounding boxes are not populated in the current rendering path. Result: every `ScoreHighlightCard` renders a loading spinner, then shows only text annotations with no score visual.

Additionally, OSMD's `render()` is synchronous and blocks the main thread. For a full piece (100+ measures), this blocks the UI for 1-4 seconds. OSMD cannot run in a Web Worker (DOM-coupled). There is no path to off-thread rendering with OSMD.

## Solution (from the user's perspective)

When the teacher generates a score highlight during a practice session or chat:
1. An inline card appears showing a cropped score excerpt with dimension-colored annotations. The main UI stays responsive while the score loads.
2. Expanding the card opens the full-score panel, also rendered by Verovio.
3. If rendering fails, the card degrades to text-only (dimension, bar range, annotation) — teaching content is never lost.

## Design

### Approach: Verovio WASM in a persistent Web Worker

Verovio is a C++ music engraving engine compiled to WebAssembly. It has no DOM dependency, runs in Web Workers (official docs recommend Worker usage for longer scores), and accepts compressed `.mxl` files natively via `loadZipDataBuffer`. It produces higher-quality engraving than OSMD/VexFlow for complex piano notation.

A single persistent Worker is spawned on first use. The Worker loads the Verovio WASM module once (~10MB, isolated to the Worker bundle — does not affect main bundle or initial page load). It maintains a per-piece toolkit cache: once a piece's MXL is loaded into a Verovio toolkit instance, subsequent renders reuse the cached toolkit without reloading.

The main thread fetches MXL bytes once per piece via `api.scores.getData()`. Bytes are cached on the main thread and sent to the Worker on first request for each piece. Concurrent requests for the same piece are deduplicated via a pending-fetch Map.

**Why Verovio everywhere:** OSMD cannot run off-thread. Keeping OSMD for the panel while using Verovio for clips produces visually inconsistent output. One renderer, consistent quality.

**Resize behavior in panel:** Three variants (reflow-on-drag-end, debounced 200ms, fixed-width CSS scale) are implemented in `app.sandbox.tsx` for visual validation. The chosen variant is locked before Task 7 executes.

**Error handling:** No silent fallback to OSMD. Any failure propagates as a rejected Promise. Components catch and render text-only fallback. Sentry captures the error.

### Verovio API

```typescript
// Init — once per Worker startup
const VerovioModule = await createVerovioModule();   // verovio/wasm
const tk = new VerovioToolkit(VerovioModule);         // verovio/esm
tk.setOptions({ pageWidth: 1800, adjustPageHeight: true, breaks: "none", footer: "none", header: "none" });

// Load a piece — once per piece, toolkit cached after this
tk.loadZipDataBuffer(mxlArrayBuffer);

// Clip render
tk.select({ measureRange: "1-4" });
const svg: string = tk.renderToSVG(1);

// Full render
const svg: string = tk.renderToSVG(1);
```

### SVG injection

The SVG string returned by the Worker is mounted into a container div via a DOM ref in a `useEffect`. Verovio produces well-formed SVG from controlled MXL sources in our R2 bucket — no user-supplied markup enters this path.

### Annotation positioning in ScorePanel

Verovio's SVG output assigns the CSS class `measure` to every measure element. After SVG injection, annotation positions are computed by querying those elements:

```typescript
const measures = Array.from(containerRef.current.querySelectorAll('.measure'));
const containerRect = containerRef.current.getBoundingClientRect();
const pos = measures[barIndex]?.getBoundingClientRect();
```

Fallback: distribute evenly if the element is not found (same as current OSMD fallback).

## Modules

### 1. `score-worker.ts`
- **Interface:** Exports `renderClipSvg(tk, buf, start, end): string` and `renderFullSvg(tk, buf): string`. Worker message handler wires these to `self.onmessage`.
- **Hides:** Verovio WASM dynamic import, `createVerovioModule` async init, per-piece toolkit cache, `loadZipDataBuffer` sequencing, `select()` + `renderToSVG()`, Worker message protocol (requestId correlation).
- **Tested through:** `renderClipSvg` and `renderFullSvg` called directly with a mock toolkit.
- **Depth verdict:** DEEP

### 2. `score-renderer.ts`
- **Interface:** `scoreRenderer.getClip(pieceId, startBar, endBar): Promise<string>`, `scoreRenderer.getFull(pieceId): Promise<string>`. Exported as a singleton.
- **Hides:** Worker construction and lifecycle, `onmessage` handler, requestId generation and correlation Map, MXL byte cache, fetch deduplication via pending Map, `sentPieceIds` Set.
- **Tested through:** `getClip` and `getFull` public methods with Worker mocked at global constructor level.
- **Depth verdict:** DEEP

### 3. `ScoreHighlightCard` (modified)
- **Interface:** unchanged — `{ config: ScoreHighlightConfig, onExpand?: () => void, artifactId?: string }`
- **Hides:** scoreRenderer.getClip call, SVG mounting via DOM ref, loading/error states, text fallback.
- **Tested through:** mocked scoreRenderer that rejects → text fallback elements appear in DOM.
- **Depth verdict:** DEEP

### 4. `ScorePanel` (modified)
- **Interface:** unchanged — no props, driven by `useScorePanelStore`.
- **Hides:** scoreRenderer.getFull call, SVG mounting, `.measure` querySelectorAll annotation positioning.
- **Tested through:** store-driven open with highlightData → component renders without error.
- **Depth verdict:** DEEP

## File Changes

| File | Change | Type |
|------|--------|------|
| `apps/web/src/lib/score-worker.ts` | Verovio Worker — pure fns + message handler | New |
| `apps/web/src/lib/score-renderer.ts` | Main-thread singleton interface | New |
| `apps/web/src/lib/score-renderer.test.ts` | Behavior tests for score-renderer | New |
| `apps/web/src/lib/score-worker.test.ts` | Pure function tests for score-worker | New |
| `apps/web/src/components/cards/ScoreHighlightCard.tsx` | Swap osmdManager to scoreRenderer | Modify |
| `apps/web/src/components/cards/ScoreHighlightCard.test.tsx` | Update mock, add text-fallback test (rename from .ts) | Modify |
| `apps/web/src/components/ScorePanel.tsx` | Swap osmdManager to scoreRenderer.getFull() | Modify |
| `apps/web/src/components/ScorePanel.test.ts` | Update mock from osmd-manager to score-renderer | Modify |
| `apps/web/src/routes/app.sandbox.tsx` | Add resize behavior section (3 variants) | Modify |
| `apps/web/package.json` | Remove opensheetmusicdisplay dependency | Modify |
| `apps/web/src/lib/osmd-manager.ts` | Replaced by score-renderer + score-worker | Delete |
| `apps/web/src/lib/osmd-manager.test.ts` | Replaced by score-renderer.test.ts | Delete |

## Open Questions

- Q: Which panel resize behavior to use (reflow-on-drag-end, debounced 200ms, fixed-width CSS scale)? Default: reflow-on-drag-end. Validated in sandbox before Task 7 executes.
