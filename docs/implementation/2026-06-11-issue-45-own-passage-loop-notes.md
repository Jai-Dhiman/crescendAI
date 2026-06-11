# Implementation Notes

Decisions, deviations, and tradeoffs made during build. Read this before running /review.

## Task 1: tempoFactor to scoreClip type and API construction sites
- Updated `ExerciseSetPayload.scoreClip` in `exercises.ts` as well as the web-side type — both needed to compile
- The `assignPendingExercise` test required the args-object signature `{studentId, sessionId, exerciseId}` and correct mock chain (no `.limit()`)
- Did NOT touch `apps/api/src/lib/types.ts` (as specified)

## Task 2: get_clip_playback worker message
- `processGetClipPlaybackRequest` is async (uses `await import("./score-ir")`)
- Quality review caught that notes loop ran AFTER `tk.select({})` restore — fixed by restructuring into try/finally with notes built before restore
- Mock SVG in test needed `class="measure"` and `class="note"` elements for `parseScoreIR` to produce non-empty bars

## Task 3: LoopClock pure timing module
- Quality review added a constructor guard: `clipEndQ <= clipStartQ` throws rather than producing NaN
- Recalibration uses `phaseOriginQ + calibrationMs` pair to preserve position across `setTempoFactor` calls

## Task 4: smplr + soundfont
- smplr@0.26.0 — `Soundfont` is a factory function (NOT constructor, `new` is deprecated)
- `piano.load` is a `readonly Promise<Smplr>` property (not a method call); also `piano.ready: Promise<void>` is the preferred non-deprecated version
- Soundfont file: 2.3MB committed directly (under 10MB threshold)

## Task 5: LoopPlayer audio orchestrator
- Used factory form `Soundfont(ctx, opts)` per actual smplr API
- Pass-wrap detection uses `Math.floor((horizonQ - clipStartQ) / clipRange)` floor-division to fire exactly once per boundary
- `nextMetronomeBeatTime` initialized in `play()`, reset in `stop()` — both in the single authoritative class body

## Task 6: useLoopPlayer hook + LoopTransport + ExerciseSetCard
- `clipNotes` kept in a ref (not in effect dep array) to prevent player teardown on `setIsPlaying(true)` re-render
- Replaced (not added) the existing `vi.mock("../../lib/score-renderer")` with merged version — Vitest deduplicates per path
- `ExerciseSetCard` falls back to `getClip` when `!hasTempoFactor` — backward compat preserved
- `ScoreCursor` wired to `loopPlayer.qstampSource` for animated cursor during playback

## Task 7: Full test run
- Web: 145/145 passing, 35 files
- API: 282 passing, 4 pre-existing catalog failures (present on main, unrelated to this branch)
- API workerd pool: known worktree symlink limitation — not caused by this branch

## Task 8: Manual browser verification
- PENDING MANUAL VERIFICATION — headless agent cannot click browser
- All code, tests, and typecheck are green
- To verify: `just dev-light`, open chat, trigger own_passage_loop prescription, confirm score-first card, transport bar, cursor animation, piano + metronome audio, tempo slider
