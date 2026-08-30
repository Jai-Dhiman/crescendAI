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

## Task 6: `buildRepertoire` -- session rows become repertoire entries

Implemented exactly as specified; no deviation from the plan's code.

- Reviewers accepted two MINOR findings, left as-is deliberately:
  tie-break when two rows share an identical `lastPracticedAt` falls back to
  `Map` insertion order (untested, unlikely in practice); and if two rows for
  the same `pieceId` carried different composer/title, first-seen wins
  silently (pieceId is assumed to key stable catalog metadata).

Commit: 00370b74

## Task 7: `listRepertoire` and `listPieceHistory` -- the queries

Implemented exactly as specified. Verified `NotFoundError`, `ServiceContext`/`Db`/`Bindings`,
`pieces`/`sessions` schemas, and `listOpenLoopsForPiece`/`OpenDrill` all matched the plan's
assumptions before writing code -- no deviations needed. Reviewer noted the added
`isNotNull(sessions.pieceId)` filter is technically redundant given the innerJoin already
excludes pieceless sessions, but kept as documentation of the invariant. Two accepted MINOR
gaps: no explicit zero-sessions/zero-drills test case, and the drills assertion only checks
length not field-level passthrough -- both low-risk, left as-is.

Commit: dbecba12

## Task 8: `GET /api/repertoire` and `GET /api/repertoire/:pieceId`

Implemented exactly as specified, following the existing `routes/scores.ts` pattern.
No deviations. Reviewers found no issues.

Commit: c819e617

## Task 9: `POST /api/practice/start` records the piece

Implemented per plan, with one necessary deviation: the plan's test-harness snippet for
detecting the target table (`(table as {_?:{name?:string}})._?.name`) does not work
against real Drizzle `PgTable` objects (the name lives behind `Symbol(drizzle:Name)`).
Fixed by using `getTableName(table)` from `drizzle-orm` instead; also avoided a
`body.pieceId!` non-null assertion (bound to a local const) to satisfy
`noNonNullAssertion`.

Code quality review caught an IMPORTANT regression: the initial implementation validated
`pieceId` against the catalog AFTER creating a new `conversations` row, so an invalid
`pieceId` left an orphaned conversation with no session/messages. Fixed in a follow-up
commit by moving the piece-catalog check to run before any conversation lookup/creation,
with a regression test asserting no `conversations` insert happens on a 404. Re-review
approved with one accepted MINOR gap: no test combines an existing `conversationId` with
an invalid `pieceId` (low risk -- that branch is read-only, never a write).

Commits: ed411526 (initial), 4f0a1e35 (ordering fix)

## Task 10: Web API client and shared query options

Implemented exactly as specified. No deviations. Reviewers found no issues; verified
WEB-QUERY-001, WEB-ERR-001, WEB-STATE-002 all held directly (grepped for queryKey
literals, catch-and-return patterns, useMemo/useCallback).

Commit: 858bcb7f

## Task 11: `RepertoireCard`

Implemented exactly as specified. Code quality review found an IMPORTANT issue: the
title span used `text-body`, which is not a defined Tailwind token in this project
(only `text-body-lg/md/sm/xs` exist) -- the class was a silent no-op, invisible to
jsdom tests. Fixed by changing to `text-body-md`, matching `ErrorBoundary.tsx`'s
pattern. Re-verified: fix is a one-line change, tests still pass.

Commits: c1ab3ee2 (initial), 075c8eba (text-body-md fix)

## Task 12: `AddPieceSearch` / Task 13: `PieceSessionHistory`

Both implemented exactly as specified. Due to a git race between two parallel build
agents in the shared worktree, both tasks' files landed in a single commit whose
message names only Task 12 -- confirmed by two independent reviewers to be a
bookkeeping artifact only; both tasks' content was verified correct and complete.

Commit: efa99041 (contains AddPieceSearch.tsx/.test.tsx and PieceSessionHistory.tsx/.test.tsx)

## Task 14: `usePracticeSession` yields a real `confidentGuess`

Implemented as specified, with one accepted deviation beyond the plan's literal
snippet: also resets `confidentGuess` to `null` in `start()`'s per-session reset
block, alongside the existing resets of `marks`, `verdict`, etc. -- confirmed
in-spirit and consistent with the established pattern, not scope creep.
`lib/piece-ladder.ts` was not modified. Reviewers accepted two MINOR gaps: no
unmount/race guard (consistent with every other WS case in the file) and only a
single happy-path test (no coverage for a second event overwriting a prior guess).

Commit: 618ea6d7

## Task 15: `/app` becomes a layout; practice moves to `/app/practice`

Implemented per plan, following the required Step 5 (build) -> Step 6 (commit) -> Step 7
(typecheck) ordering to resolve the plan's one documented BLOCKER (route-tree codegen
must precede typecheck for a new route path). Deviations, all confirmed reasonable by
review: test used `fireEvent.click()` instead of raw `.click()` (React 19 + jsdom does
not flush state synchronously on a raw DOM click; same assertions, different dispatch
mechanism); `void practice.start()` added (floating-promise lint fix, `start` returns
`Promise<void>`); handlers refactored into named functions; Tailwind class order fixed
(`h-20 w-20`). Code quality review accepted two MINOR gaps: `RouteComponent`'s name
breaks the codebase's convention of descriptive route-component names (every other
route names its component after the page), and no test directly exercises the
`confidentGuess` wiring through to `PracticeMode` (verified correct by inspection,
but not by a dedicated assertion). Full `bun run test`: 220/221 (1 pre-existing
unrelated flaky perf test failure).

Commit: 1433366a

## Task 16: Home

Implemented exactly as specified. No deviations to the component itself.
Build was resumed after two prior agents died on API rate limits, mid-way
through Group G; Tasks 1-15 were verified already committed before starting
this task. Because Task 16 and Task 17 were run sequentially (not in parallel,
per the resuming instruction, to avoid repeating the git race that collapsed
Tasks 12/13 into one commit) rather than the plan's "parallel, then a shared
route-tree regen" shape, `bun run typecheck` fails on this task's own commit
in isolation -- `navigate({ to: "/app/piece/$pieceId", ... })` doesn't
typecheck until `routeTree.gen.ts` knows that path, which Task 17 adds. This
is expected and resolved by Task 17's build step; the combined typecheck gate
is clean once both tasks have landed (verified below). Lint required `void`
on both `navigate(...)` calls (`noFloatingPromises`) and a biome format pass
on the test file, neither of which the plan's literal snippet needed since it
predates this repo's exact lint config drift.

Commit: 7beedb4b

## Task 17: Piece page

Implemented per plan, following the required Step 4 (test) -> Step 5 (build,
regenerating `routeTree.gen.ts`) -> Step 6 (commit route + generated tree
together) -> Step 7 (typecheck) ordering. `bun run build` succeeded and added
the `/app/piece/$pieceId` entry to `routeTree.gen.ts` (21 lines, verified via
diff). One lint fix beyond the plan's snippet: `void` on the `navigate(...)`
call in the "Practice this" button (`noFloatingPromises`), and a
`useSortedClasses` Tailwind class-order fix on the open-drills `<li>` (biome
`--write --unsafe`). No other deviations.

With both Task 16 and Task 17 landed, `bun run typecheck` in `apps/web` is
clean (this resolves Task 16's isolated typecheck gap above). Full gates:

- `apps/web`: `bun run test` -- 230/230 passed (the previously-flagged flaky
  perf test passed on this run); `bun run typecheck` -- clean.
- `apps/api`: `bun install` -- no changes; `bun run test` -- 527/527 passed;
  `bun run typecheck` -- clean.

Commit: 4dd67306

## Code review fix: pieceId never reached the backend (P0)

The spec (line 72-73) requires a session started from a piece to carry that
piece's id for its whole life. The backend already accepted and validated
`pieceId` on `POST /api/practice/start`, but the frontend never sent it:
`AppPractice`'s `handleStart` called `practice.start()` with zero arguments,
`usePracticeSession.start` never took a `pieceId` parameter, and
`practiceApi.start` never put it in the POST body. Threaded `pieceId` through
all three: `practiceApi.start(conversationId?, pieceId?)` builds a single
JSON payload object from whichever of the two are present (instead of the
old conversationId-only ternary), `usePracticeSession`'s `start` signature
gained a second `pieceId?: string` parameter and forwards it, and
`AppPractice.handleStart` now calls `practice.start(undefined, pieceId ??
undefined)` using the `pieceId` already available from the `?pieceId=`
search param. Added `apps/web/src/lib/practice-api.start.test.ts` asserting
the fetch body contains `pieceId` when a piece was picked, omits it entirely
when none was, and includes both `conversationId` and `pieceId` together.
No other call sites of `practiceApi.start`/`practice.start` existed to
update. `bun run test` (scoped) and `bun run typecheck` clean.

Commit: 86b52612

## Code review fix: `text-body` no-op Tailwind class (P1)

`app.css` defines only `--text-body-lg/-md/-sm/-xs`; there is no bare
`--text-body` token, so `className="text-body ..."` silently did nothing.
This exact bug was already caught and fixed once on this branch in
`RepertoireCard.tsx` (changed to `text-body-md`), but four more `<h2>`
occurrences were introduced later: `app.index.tsx:59` and
`app.piece.$pieceId.tsx:89,109,114`. Changed all four to `text-body-md` to
match `RepertoireCard.tsx`. Grepped the full branch diff
(`git diff $(git merge-base main HEAD) -- apps/web/src`) for any other bare
`text-body`/`text-display`/`text-label` token usage; none found -- only
already-suffixed classes (`text-body-md/-sm/-xs`) appear. `bun run
typecheck` clean; scoped `bun run test` (`app.index`, `app.piece`) passed.

Commit: cacbb4f3
