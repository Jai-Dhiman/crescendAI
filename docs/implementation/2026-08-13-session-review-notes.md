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
