# Implementation Notes

Decisions, deviations, and tradeoffs made during build. Read this before running /review.

## Task 1: Piece-page fixtures -- marks aging across two sessions

Implemented exactly as specified; no deviation from the plan's code was needed.
`Mark`, `resolveAnchor`, and the `MarkLifecycle`/`MarkTaxonomy`/`Dimension`
unions in `lib/mark.ts` all matched the plan's assumptions.

- The first commit attempt was rejected by the pre-commit Biome hook over
  line-wrapping of the `flatMap`/`Math.max` calls. Fixed with `bun run format`
  (the project's own formatter), tests re-verified, committed without
  `--no-verify`. No logic changed.
- Reviewers' two accepted MINOR findings, both left as-is deliberately:
  `Math.max` over `PIECE_PAGE_ALL_MARKS` would yield `-Infinity` if the fixture
  array were ever emptied (unreachable today — the data is two hardcoded
  sessions of two marks each), and the span assertion in the test is a
  regression tripwire against someone hardcoding a stale span rather than
  independent evidence the derivation is correct.

Commit: c2b2d3f1
