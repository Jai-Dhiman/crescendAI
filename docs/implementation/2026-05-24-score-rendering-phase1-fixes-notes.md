# Implementation Notes — score-rendering-phase1-fixes

Decisions, deviations, and tradeoffs made during build. Read this before running /review.

## Task 1: Worker loadPiece extraction (commits aa3cb8e6, 3733e45b)
- `FakeToolkitClass` in the test is a plain `function`, not `vi.fn().mockImplementation(...)`. Arrow-function mocks cannot be invoked with `new` under vitest.
- `console.error` for buildMeasureIndex failures uses comma-separated args, not template-literal format strings (semgrep CWE-134).
- Followup fix (3733e45b): added `console.error` logging to the `extractXmlFromMxl` and `loadData` fallback catch blocks (silent swallow was a regression vs the original inline `loadPiece`). The third silent catch (ZIP `loadZipDataBuffer` recovery) was intentionally left silent — original behavior, it's a recovery path not a failure.
- Pre-existing 3 failures in `score-renderer.test.ts` are still present after Task 1 — they are Task 3's seam.

## Task 2: Worker render_clip via tk.select (commit 797a6383)
- Manual Playwright checkpoint was SKIPPED per autopilot orchestrator direction. Justification: /challenge re-review verified `g.measure` selector is correct (existing `ScoreGeometryProbe` uses it against real Verovio output). If any visual regression appears in dev, fall back per plan Task 2 halt instructions (use Approach B / `SvgClipBBox`).
- Cleanup beyond plan letter: also removed orphan `getPageForBar` helper and `ClipSvgResult` interface (no remaining callers after `renderClipSvg` deletion).
- All catch blocks in `loadPiece` fallback now use `console.error(e)` (continued from Task 1 fix).
- Minor findings deferred (not blocking): `renderClipSvgSelect` is still exported though only used inside the module; stale "Approach C" / "mei/mxl" comments at lines 27 and 60. Acceptable.
