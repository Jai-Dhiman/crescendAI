# Implementation Notes

Decisions, deviations, and tradeoffs made during build of #157 (mark system).
Read this before running /review.

## Task 1: Cross-canvas contract harness
Committed deliberately red with `--no-verify`. Failure is at vite import-analysis
(module resolution), 0 tests collected — not a syntax error. No stubs created for
`mark-fixtures`, `ScoreMarkLayer`, or `SessionTimelineStrip`.

## Task 2: Anchor degrades to timestamp
`resolveAnchor` intentionally ignores `bars`/`alignmentQuality` and always returns
a timestamp anchor. This is the deliberate TDD minimum; Task 3 adds the bars
branch. Reviewers must NOT flag it as incomplete.
`as unknown as MarkAnchor` is required — a plain `as` does not compile against the
branded intersection (verified independently before the plan was written).
`import type { Dimension }` deliberately deferred to Task 5: tsconfig sets
`noUnusedLocals: true`, so an unused import is a hard error, not a warning.

## Plan defect found during Task 2 (fixed in the plan)
The known-red window also makes `bunx tsc --noEmit` exit 2, not just `bun run test`.
The plan originally documented only the test-suite consequence, and most tasks'
Step 4 asked for `tsc` exit 0 — unachievable until Task 19. Plan updated: tasks
now filter with `bunx tsc --noEmit 2>&1 | grep -v "mark-canvases.contract.test.tsx"`
and expect no surviving output.
