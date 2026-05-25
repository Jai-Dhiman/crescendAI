# Score Rendering Phase 1 — Bugfix Pass Design

**Goal:** Make the score rendering pipeline (worker + renderer + card consumers) correct against today's API, without changing the API.

**Not in scope:**
- Score IR layer, new worker API, cursor module, performance overlay — all in Phase 2 ([project_score_rendering_phase2.md](../../.claude/projects/-Users-jdhiman-Documents-crescendai/memory/project_score_rendering_phase2.md))
- ScorePanel annotation positioning off-by-2 fix — deferred to Phase 2 (the IR makes it trivially correct; fixing it now would be a workaround we throw away)
- Server-side rendering, persistent SVG cache
- Drag-to-resize reflow
- iOS — web only

## Problem

Five concrete defects in the score rendering pipeline, verified live against `chopin.ballades.1` in `/app/sandbox`:

1. **Silent measure-index failure.** `score-worker.ts:416-419` wraps `buildMeasureIndex` in `try { … } catch {}` that leaves `measures = []` on any error. Every per-bar lookup then returns `undefined`, and every clip method silently degrades to "render page 1." Confirmed: `getClipMethod('chopin.ballades.1', 135, 136, 'select')` returns an SVG with 25 measures (all of page 1) instead of the requested 2 bars.
2. **Approach A (`SvgClip.tsx`) is geometrically broken.** `SvgClip.tsx:36` derives `scaleY = vb.height / svgRect.height` mixing post-CSS-layout screen coords with SVG viewBox coords. For bars 135–136, A produces a viewBox of `0 196 960 205` while Approach B (`SvgClipBBox.tsx`, using `getBBox + getCTM`) produces the correct `0 96 960 407` on identical input. A is what `ScoreHighlightCard.tsx:94` and `PlayPassageCard.tsx:119` ship to users today.
3. **Approach E (MXL filter) is dead code.** `score-worker.ts:341-405` stores `xmlContent = null` on the happy `loadZipDataBuffer` path. Any subsequent `renderClipSvgMxl` call hits `xmlContent not available for mxl method`. Confirmed via Playwright.
4. **Approach D (MEI round-trip) is unusably slow.** `getClipMethod('mei')` measured at 1019ms per clip on a 264-bar piece — ~70× slower than alternatives. Rebuilds a Verovio toolkit per request.
5. **`ScoreHighlightCard` loads clips serially.** `ScoreHighlightCard.tsx:36-66` uses `for…await`. Three highlights = three worker round-trips of ~265KB each, ~50ms+ avoidable latency on warm cache.

Together these mean: highlight cards visibly stage in, the rendered clips are geometrically wrong (truncated by ~50% vertical context), and several rendering "options" in the worker are non-functional. The piece-level cache works and Verovio itself is fast (warm `getFull` 17ms, default `getClip` 11–18ms) — the failures are entirely in the layers we own.

## Solution (from the user's perspective)

After this ships, when a teacher sends a score highlight in chat:
- The clip shows the requested bars with full vertical context (both staves, including barlines and the system row).
- Three-highlight cards finish loading at the same time, not staggered.
- If the underlying score fails to load, the user sees a single deterministic error, not a partial render of the wrong content.

No visible API or UI changes beyond correctness. The renderer's external surface (`scoreRenderer.getFull`, `scoreRenderer.getClip`) is unchanged.

## Design

**Canonical clip method: Approach C (`tk.select`)**, contingent on the measure-index fix landing first. Reasons:

- Smaller SVG payload per clip (~50–80KB vs ~265KB), since Verovio engraves only the requested bars instead of the whole page.
- Verovio's `select()` auto-injects clef, key signature, and time signature at the start of the clipped range — correct musical context for free.
- No client-side cropping logic at all; the worker returns an already-correct SVG.

**Fallback: Approach B (`SvgClipBBox`)**, if Task 2's validation reveals `tk.select` produces broken output (cross-system splits, missing barlines) on any real piece in the catalog. B uses the existing full-page-SVG + `getBBox + getCTM` crop, which is geometrically correct but ships a much larger payload per clip.

**Branching plan checkpoint:** Task 2 explicitly validates C on three reference pieces (4-bar exercise, mid-piece highlight, multi-page Ballade range). If any fail visual or assertion checks, the build halts and the plan is revised to fallback B before Tasks 3–6 execute.

**Other decisions:**

- Worker keeps `xmlContent: null` on the happy path **and** removes the MXL clip method entirely. Since approach E is dead code, the only consumer of `xmlContent` is being deleted; storing it would be carrying weight for nothing. The fallback decode path in `loadPiece` (`score-worker.ts:372-388`) is preserved because it's the legitimate failure recovery, not the MXL filter.
- `renderClipSvgMei` and `renderClipSvgMxl` are deleted from the worker along with the `'mei'` and `'mxl'` branches of the message handler. `scoreRenderer.getClipMethod` is removed from the renderer surface. There are no production consumers of these — only `app.sandbox.tsx`'s `ApproachesComparison`, which is deleted in the same pass.
- The silent-fallback regression test (Task 7) encodes a project principle: the worker must propagate `loadPiece` failures as `error:` messages, never as degraded successes. This catches the entire *category* of bug we hit, not just the specific timemap failure.

**Trade-offs accepted:**

- We do not understand *why* `renderToTimemap` triggers the deprecated-WASM-`try`-instruction warning. The fix surfaces the error from `buildMeasureIndex`; if it throws on a real piece, `loadPiece` returns failure and the user sees an error state. We accept this UX in exchange for "no silent fallback" — Phase 2 will revisit the root cause when the IR walk replaces timemap.
- We delete `SvgClip` and (likely) `SvgClipBBox` rather than fix A. There is no scenario where A is the right approach; preserving it as "another option" is the trap that produced today's mess.
- `ScoreHighlightCard` parallelization uses `Promise.all`. If any clip fails, the entire card flips to error state (matching today's behavior under `for…await`). We deliberately do *not* use `Promise.allSettled` — partial-success rendering ("two clips OK, one missing") is a UX decision that should be made when there's evidence users hit this case; today's contract is all-or-nothing and the parallelization preserves it.

## Modules

**`score-worker.ts` (modify, DEEP)**
- Interface: postMessage protocol — `render_full`, `render_clip` (no `method` field after this change), responses with `{ svg, startMeasureId, endMeasureId }` or `{ error }`.
- Hides: WASM lifecycle, ZIP parsing/decompression, Verovio toolkit cache, timemap → measure index join, error classification (WASM exception vs format error vs index failure).
- Tested through: `score-worker.test.ts` exercising the message protocol with a mock Verovio toolkit. After this pass, also tested by the silent-fallback regression (Task 7).

**`score-renderer.ts` (modify, DEEP)**
- Interface: `scoreRenderer.getFull(pieceId, pageWidth?)`, `scoreRenderer.getClip(pieceId, startBar, endBar)`. `getClipMethod` is removed.
- Hides: Worker lifecycle/recovery, request correlation by `requestId`, bytes cache and `sentPieceIds` correctness, `pendingFetches` deduplication.
- Tested through: `score-renderer.test.ts` exercising public methods with a `MockWorker` stub.

**Consumer components (modify, shallow by design)**
- `ScoreHighlightCard.tsx`, `PlayPassageCard.tsx`, `app.sandbox.tsx` — these compose the deep modules and pass output to SVG containers. They get smaller in this pass (one fewer component import, parallel `Promise.all` instead of `for…await`).

## File Changes

| File | Change | Type |
|------|--------|------|
| `apps/web/src/lib/score-worker.ts` | Unswallow `buildMeasureIndex`; delete `renderClipSvgMei`, `renderClipSvgMxl`, related message handler branches, `xmlContent` field on `CacheEntry`, MXL extraction call on happy path | Modify |
| `apps/web/src/lib/score-renderer.ts` | Remove `getClipMethod` and `pendingRequests` kind narrowing for it; route `getClip` to the worker's `select` path internally (if C wins) | Modify |
| `apps/web/src/components/SvgClip.tsx` | — | **Delete** |
| `apps/web/src/components/SvgClipBBox.tsx` | — | **Delete (if C wins)** / Keep (if B wins) |
| `apps/web/src/components/cards/ScoreHighlightCard.tsx` | Render returned SVG via a new `<SvgPanel>` (or keep `SvgClipBBox` if B wins); replace `for…await` with `Promise.all` | Modify |
| `apps/web/src/components/cards/PlayPassageCard.tsx` | Same SVG-rendering swap as `ScoreHighlightCard` | Modify |
| `apps/web/src/components/cards/ExerciseSetCard.tsx` | Same SVG-rendering swap (state holds `string` instead of `ClipResult`) | Modify |
| `apps/web/src/components/cards/ExerciseSetCard.test.tsx` | New test covering the SVG render and the no-scoreClip path | New |
| `apps/web/src/routes/app.sandbox.tsx` | Delete `ApproachesComparison`, `ApproachA`, `ApproachB`, `ApproachWorkerMethod`, `ApproachRow`, related imports; keep `ScoreGeometryProbe` (Phase 2 spike) | Modify |
| `apps/web/src/lib/score-worker.test.ts` | Add regression test: when `buildMeasureIndex` throws, `loadPiece` resolves with `'failed'` (not a degraded `CacheEntry`) and the worker responds with `error:` | Modify |
| `apps/web/src/lib/score-renderer.test.ts` | Update mock worker responses to match new `getClip` contract; drop any test exercising removed methods | Modify |

## Open Questions

- **Q:** Does `tk.select` produce correct output for *all* bar ranges in the production catalog, or only the ones in the sandbox? **Default if not resolved:** Task 2 validates against three pieces; if any fail, fallback B. We do not pre-test the full catalog.
- **Q:** Does `buildMeasureIndex` actually throw on `chopin.ballades.1`, or does the deprecated-WASM-warning fire without throwing? **Default if not resolved:** Task 1 unswallows the catch and logs the underlying outcome. If it doesn't throw, the silent fallback was never the cause and Task 1 shifts to investigating why the timemap returns empty.
- **Q:** Are there any non-sandbox callers of `scoreRenderer.getClipMethod`? **Default if not resolved:** Repo grep at plan time. None found in initial search — confirm in Task 4 before deleting.
