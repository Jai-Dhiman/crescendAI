# Implementation Notes

Decisions, deviations, and tradeoffs made during build. Read this before running /review.

## Task 1: formatWsGap / session-review types
Implemented verbatim per plan. Types are scaffolding ahead of their consumers (used by VerdictBanner, SessionDisclosures, SessionReview in later tasks). Reviewer noted `Math.round` on sub-minute gaps is untested (0-second/negative edge cases) — accepted, out of scope.

## Task 2: createPassageQstampSource
Implemented verbatim per plan. Reviewer confirmed DEEP module: bracketing-pair search + interpolation + three-way null degradation behind a 1-function interface. Known accepted gap (per challenge review): no test for "bar missing from `bars`" — code path exists (`if (!barIR) return null`) but untested.

## Task 3: usePracticeSession sessionId
Implemented per plan, plus the challenge-review-mandated landmine comment. Added directly above `sessionIdRef.current = null;` in `cleanup()`:
`// Landmine: this nulls sessionId at the exact moment synthesis/session_summary arrives (see handleWsMessage), so any live-review surface reading sessionId from that handler sees null -- future wiring (#162) must capture the id before cleanup() runs.`
`cleanup()` itself was not otherwise modified — the sessionIdRef-cleared-too-early behavior is intentionally left as-is, flagged for #162.

## Task 4: ScoreMarkLayer renderDetail
Implemented verbatim per plan. Existing `renderLayer` test helper extended to accept an `overrides` param without touching prior test cases.

## Task 5: SessionTimelineStrip renderDetail
Implemented verbatim per plan, mirrors Task 4's pattern exactly.

## Task 6: MarkPassagePlayback
Implemented all 5 states (no_bars/loading/ready/audio_error/error) plus the two extra test cases (error, audio_error) mandated by the challenge review. Deviation: implementer omitted the plan's `_mockManifest`/`_mockClip`/`_mockIr`/`_playable` escape-hatch props (copy-pasted precedent from `PlayPassageCard.tsx`) since the test suite mocks at the module boundary instead and no caller anywhere in the plan (present or planned) ever passes those props. Reviewer traced the whole plan doc and confirmed this is genuinely dead surface — accepted as scope discipline, not a scope-narrowing bug.

## Task 7: VerdictBanner
Implemented per plan with one adaptation: the "loading" case uses `<output>` instead of `<p role="status">` because Biome's `lint/a11y/useSemanticElements` rule blocks that pairing and explicitly recommends `<output>`. `<output>` carries an implicit ARIA status role so semantics are preserved; verified sound by the reviewer.

## Task 8: SessionDisclosures
Implemented verbatim per plan. Content landed across two commits (fd1bbd8d, eed3838b) due to a shared-worktree commit-ordering race with the parallel Task 7 agent — both agents confirmed no content was lost; reviewed at HEAD.

## Task 9: ScoreStand renderDetail
Implemented verbatim per plan.

## Task 10: PieceLessMode renderDetail
Implemented verbatim per plan.

## Task 11: MarkReviewDetail
Implemented verbatim per plan, mirroring MarkDetail.tsx's evidence-rendering pattern. `MarkPassagePlayback` mocked at module boundary in the test.

## Task 12: SessionReview
Implemented verbatim per plan. Composes VerdictBanner/ScoreStand-or-PieceLessMode/SessionDisclosures/carry-forward line, with `renderDetail` closure wired to MarkReviewDetail. 10-prop composer accepted per the plan's challenge review (pure composition layer over a client contract with no server implementation yet).

## Task 13: PracticeMode review phase
Implemented verbatim per plan. `phase` defaults to "practice" so all existing callers (AppChat, /practice-preview) are unaffected — verified all 4 pre-existing PracticeMode tests still pass alongside the new one. `usePauseTracker` gated on `phase === "practice"` so auto-stop banner logic doesn't fire during review. Optional review-data props all have safe defaults matching SessionReview's required prop types.

## Task 14: /practice-preview review-ready fixture section
Implemented per plan, reusing the pre-existing `SCORE_FIXTURE_MARKS` local const. Deviation: used `style={{height:500}}` for the new section's inner wrapper instead of the existing sections' `h-1/3` flex classes, since the existing three sections already sum to 100% height (h-1/3 * 3) inside the `h-dvh` parent — a 4th h-1/3 would overflow. Reviewer confirmed this reasoning is sound and doesn't disturb the prior three sections.

## Task 15: /practice-preview review-failed fixture section (final task)
Implemented per plan. Deviation: the plan's illustrative snippet used `userPickedPieceId={null}`, which would have made this section fall back to `PieceLessMode` and collide (duplicate `data-testid="session-timeline"`) with the pre-existing pieceless fixture section elsewhere on the page, breaking that section's own test. Fixed by setting `userPickedPieceId="chopin-nocturne-op9-no2"` so this section renders `ScoreStand` instead. Reviewer verified `VerdictBanner` (which renders the failed-state alert under test) is a sibling of the ScoreStand/PieceLessMode conditional inside SessionReview, receiving only `verdict`/`onRetry` — entirely independent of `pieceId` — so the deviation does not touch what this task verifies. All 4 tests in practice-preview.test.tsx pass, including the pre-existing pieceless test, confirming the collision was genuinely fixed.

## Plan complete: all 15 tasks implemented, spec-reviewed, and quality-reviewed. See STATE comment on issue #159 and the /review pass for final verification.
