# Corpus Drill Renderable Assets + Verovio Transpose-on-Demand Design

**Goal:** A corpus_drill exercise primitive (e.g. `hanon_001`) can be fetched from R2 and rendered in the web score panel at any in-range key, so the existing `corpus_drill` card can show a real, transposable score instead of stub text.
**Not in scope (S4+, do NOT build here):** teacher/Worker selection of a primitive; the off-keyboard transpose gate (the decision logic that picks a *legal* semitone range); drill playback/audio; corpus breadth (#17/S5); ranking (#42/#43/S6). This slice delivers the *assets* and the *render-at-key plumbing* only.

## Problem

The S1 `corpus_drill` branch of `ExerciseRoutingDecision` currently renders stub text (per MEMORY: "corpus_drill stubs text (S2/S3/S4 follow)"). The 22 exercise primitives exist as committed MusicXML at `model/data/scores/exercise_primitives/*.xml`, but:

1. **They are not in R2.** The production data endpoint `GET /api/scores/:pieceId/data` (`apps/api/src/services/scores.ts::getPieceData`) reads `scores/v1/{pieceId}.mxl` from R2. No primitive has ever been uploaded there, so any attempt to render `hanon_001` 404s.
2. **The renderer cannot transpose.** `ScoreRenderer.load(pieceId)` (`apps/web/src/lib/score-renderer.ts`) and the worker's `loadPiece(bytes, bindings, pieceId?)` (`apps/web/src/lib/score-worker.ts`) load a fixed key. A drill is "play this pattern in F# major"; without a transpose path the primitive can only render in its authored key.

Without this slice the corpus_drill card has nothing real to display.

## Solution (from the user's perspective)

When the teacher prescribes a corpus drill, the web score panel fetches the primitive's `.mxl` from R2 (served by the unchanged data endpoint) and Verovio engraves it. If the drill specifies a key offset, the panel passes a semitone integer to the renderer and Verovio engraves the same notes shifted by that many semitones, with accidentals auto-minimized. The same primitive at `transpose: 0` (or no transpose) renders byte-for-byte identically to a real catalog piece — real pieces are wholly unaffected.

## Design

Three independent vertical slices plus a verification harness:

**(A) Offline asset generation.** `model/src/exercise_corpus/build_render_assets.py` exposes a single `build()` entrypoint that, for each committed `model/data/scores/exercise_primitives/*.xml`: validates it loads in partitura, strips its DOCTYPE, wraps it in a standard MXL ZIP (reusing the proven `model/src/score_library/upload.py::wrap_as_mxl_zip` container format), and writes `model/data/exercise_primitives/mxl/{primitive_id}.mxl` (a committed, non-gitignored directory). Deterministic and idempotent (skip-if-unchanged by comparing the inner XML bytes). A `.xml` that fails partitura load **raises** naming the file — no skip-and-continue. Driven by `just build-exercise-assets`.

**(B) Local R2 seeding.** `just seed-exercise-assets` enumerates the committed `.mxl` files and `wrangler r2 object put`s each to `crescendai-bucket/scores/v1/{primitive_id}.mxl --local`, mirroring the existing `just seed-fingerprint` / `just seed-scores` recipes. Idempotent overwrite. A put failure makes the recipe exit non-zero (`set -euo pipefail`). The flat `scores/v1/` keyspace is collision-safe: primitive ids (`hanon_001`) share no namespace with real ASAP piece slugs (`beethoven.fur_elise`).

**(C) Renderer transpose plumbing.** `ScoreRenderer.load(pieceId, transpose?: number)` and the worker's `loadPiece(bytes, bindings, pieceId?, transpose?: number)` gain an optional **semitone integer**. The public surface is a `number` end-to-end; **only** at the single `tk.setOptions` call site inside `loadPiece` is it stringified to Verovio's `transpose: String(n)`. Verovio applies `transpose` at `loadData` time, so a different key is a different engraving — therefore the worker's toolkit cache key and the renderer's IR cache key become `${pieceId}:${transpose ?? 0}`. Byte fetching (`ensureBytes`/`bytesCache`) stays keyed by `pieceId` alone (transpose changes the engraving, not the source bytes). A real piece loaded with no transpose argument (or `transpose: 0`) takes a code path byte-identical to today's; the composite key for `transpose: 0` equals the legacy `${pieceId}:0` slot and produces identical output. If `loadData` returns `false` with a transpose set, the existing "failed" channel + `Sentry.captureException` fires upstream — there is NO untransposed fallback.

**(D) Hermetic oracle test.** A Python test globs the committed `*.xml` (never the gitignored `.db`/`.mid`), and for each primitive across a band of in-range semitone offsets transposes via Verovio (the `verovio` Python binding — same WASM core as the web's `verovio` npm package, runs in-process with partitura, no Node subprocess) and via `transforms.py::transpose`, then asserts the **pitch-class multiset** (`midi_pitch % 12` Counter) is equal. Both branches originate from the SAME committed `.xml`. For offsets that push a primitive off the 88-key piano, `transforms.py::transpose` already raises (lines 104–108); the test asserts both sides reject rather than asserting equivalence — the off-keyboard *gate* itself is deferred to S4.

### Key decisions and trade-offs

- **MusicXML, not MEI.** Already verified: the 22 `.xml` exist and the prod load path is MusicXML-native (`getPieceData` serves `application/vnd.recordare.musicxml+zip`; `loadPiece` uses `loadZipDataBuffer` with a raw-XML `loadData` fallback). No format conversion.
- **Reuse the unchanged data endpoint.** `GET /api/scores/:pieceId/data` only touches R2 and needs no `pieces` DB row, so no API code and no migration is required — only an R2 seed. Trade-off: primitives have no catalog metadata row, which is fine because the renderer needs only the bytes; metadata (when S4 needs it) lives in the existing gitignored `model/data/exercise_primitives.db` consulted by the local seed step, never by the CI oracle.
- **Verovio Python binding over headless Node for the oracle (refinement of the brainstorm's "headless Node").** Empirically confirmed (verovio 6.2.1, matching web verovio ^6.1.0): `setOptions({transpose:'2'})` + `loadData()` yields a pitch-class histogram exactly equal to the base histogram shifted +2 mod 12. The binding runs in the same `uv run pytest` process as partitura — strictly more hermetic than cross-runtime Node subprocessing, and it lives naturally in the existing `model/tests/exercise_corpus/` suite. This is an implementation refinement, not a design change: the oracle's contract ("transpose via Verovio AND via transforms.py, assert PC-multiset equality") is identical. Residual risk: the binding must be installed (added as a model dev dependency); the test pins it.
- **Semitone integer surface, Verovio string only at the boundary.** Maps 1:1 onto `transforms.py::transpose(part, semitones: int)` and keeps Verovio's interval-string vocabulary out of `ScoreRenderer`/React.
- **Transpose in the cache key, not the byte cache.** A primitive's bytes are fetched once; each distinct key is a distinct Verovio engraving, so only the toolkit/IR caches key on transpose.

## Modules

- **`build_render_assets.py` (build)** — Interface: `build(xml_dir: Path = <committed default>, out_dir: Path = <committed default>) -> list[Path]`. Hides: partitura validation, DOCTYPE stripping, MXL ZIP container construction (delegated to `wrap_as_mxl_zip`), idempotent skip-if-unchanged, fail-loud-on-bad-xml. Tested through: calling `build()` on the committed corpus and asserting the produced `.mxl` files are valid ZIPs whose inner note count matches the source `.xml`. DEEP — one call hides the whole offline pipeline.
- **`loadPiece` transpose path (web worker)** — Interface: `loadPiece(bytes, bindings, pieceId?, transpose?: number)`. Hides: Verovio `setOptions({transpose})` placement, the load/redo ordering, composite-cache-key derivation. Tested through: loading a real fixture twice (transpose 0 vs 2) and asserting different SVG output, and transpose 0 == no-transpose. DEEP — the transpose surface is one optional number hiding Verovio's load-time engraving contract.
- **Oracle equivalence test (Python)** — Interface: a pytest module exercising `transforms.py::transpose` and the Verovio binding through their public APIs. Hides: nothing (it IS the verification). Tested through: itself. (Not a production module; it is the harness.)

## Verification Architecture

- **Canonical success state:** For every committed primitive and every in-range semitone offset, the pitch-class multiset produced by Verovio's `transpose` equals the multiset produced by `transforms.py::transpose` — and both reject (raise) on off-keyboard offsets. Separately, every committed `.mxl` loads in Verovio (`getPageCount() > 0`) with a note count matching its source `.xml`, and `loadPiece(..., transpose: 2)` yields a different SVG than `transpose: 0` while `transpose: 0` is byte-identical to the pre-change load.
- **Automated check:** `cd model && uv run --with verovio pytest tests/exercise_corpus/test_render_assets_oracle.py` (oracle + asset integrity, hermetic — globs committed `.xml`) and `cd apps/web && bunx vitest run src/lib/score-worker.transpose.integration.test.ts` (renderer plumbing against a real Verovio fixture).
- **Harness:** Buildable before the feature — it IS Task Group 0 (the oracle test), which can be written and run against `transforms.py::transpose` (already shipped) and the Verovio binding immediately, gating the asset-gen and renderer tasks. Asset-integrity assertions depend on Task A output, so they live in Task A's own test, not Group 0.

## File Changes

| File | Change | Type |
|------|--------|------|
| `model/src/exercise_corpus/build_render_assets.py` | New `build()` asset generator | New |
| `model/data/exercise_primitives/mxl/*.mxl` | 22 committed compressed MusicXML assets (build output) | New |
| `model/tests/exercise_corpus/test_render_assets_oracle.py` | Hermetic Verovio-vs-partitura PC-multiset oracle + asset integrity | New |
| `model/pyproject.toml` | Add `verovio` to dev dependencies | Modify |
| `justfile` | Add `build-exercise-assets` + `seed-exercise-assets` recipes | Modify |
| `apps/web/src/lib/score-worker.ts` | `loadPiece` gains `transpose?: number`; composite cache key; `transpose` worker message field | Modify |
| `apps/web/src/lib/score-renderer.ts` | `ScoreRenderer.load(pieceId, transpose?)`; composite IR cache key | Modify |
| `apps/web/src/lib/score-worker.transpose.integration.test.ts` | Real-Verovio transpose plumbing + regression-lock | New |

## Open Questions

- Q: Should the oracle test cover all 22 primitives or a representative subset for CI speed? Default: all 22 across offsets `-5..+6`, with a single-primitive fast subset available if CI wall-clock exceeds ~60s (the Verovio binding loads each XML in tens of ms, so all-22 is expected to be acceptable).
- Q: Do the partitura-exported `.xml` render cleanly in Verovio as-is, or does the build step need a normalization pass? Default: assume clean (DOCTYPE-strip is the only known transform, already proven by `wrap_as_mxl_zip`); if Task A's asset-integrity assertion fails, the build step grows a normalization pass — a plan-time discovery, not a design change.
