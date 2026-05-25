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

## Task 3: getClip → Promise<string> (commits 61037a42, d19b0ba4)
- Test file needed no edits — existing assertions already encoded the post-fix behavior.
- Followup (d19b0ba4): collapsed `PendingFull | PendingClip` into a single `PendingRequest` type. After `getClip` returned `string`, both variants were structurally identical and the `kind` discriminant was never read.
- Reviewer over-flagged broken consumer imports (`ExerciseSetCard`, `PlayPassageCard`, `ScoreHighlightCard`, `app.sandbox.tsx`) as CRITICAL — those are scope-deferred to Tasks 4/5/5b/6 per plan, and the dispatcher confirmed mid-build red is expected.

## Task 4: ScoreHighlightCard parallel-load (commit 2f8087ff)
- jsdom serializes HTML attributes with double quotes; test assertions converted single→double accordingly.
- Reviewer noted minor: `.catch` branch sets error state without `cancelled` guard. Matches existing codebase pattern; React 18+ noop on unmounted; not blocking.

## Task 5: PlayPassageCard string clip (commit 4f33f36a)
- `PassageManifest` required more fields than the plan listed (`source`, `startOffsetSec`, `endOffsetSec`, `barTimeline`) — filled with realistic values.
- All three existing `mockGetClip.mockResolvedValue({svg: ..., ...})` calls updated to plain strings.
- `clip: ClipResult | null` state renamed to `clipSvg: string | null`.

## Task 5b: ExerciseSetCard string clip (commit bf9b7a38)
- New test file created.
- `ExerciseSetConfig.exercises[i].focusDimension` is required (not optional); filled with "dynamics" in test fixtures.
- Pre-existing `.catch(() => {})` silent swallow on `getClip` left untouched (out of scope for this task — flagged for future cleanup).

## Task 6: Delete dead helpers + sandbox cleanup (commit a3976051)
- Additional orphan removed beyond plan letter: `SvgDisplay` helper (was sole-use by `ApproachWorkerMethod`); deleted to clear TS6133 unused-var error.
- Sandbox's local `ClipSvg` uses `useEffect` rather than `useLayoutEffect` (cards use `useLayoutEffect`); sandbox is dev-only so the flash-of-unstyled-SVG concern doesn't apply. Acceptable per reviewer.
- Full web test suite: 63/63 green. `tsc --noEmit` for web is clean (apps/api drizzle errors are pre-existing and out of scope).
