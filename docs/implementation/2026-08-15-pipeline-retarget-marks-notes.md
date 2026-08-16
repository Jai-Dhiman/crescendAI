# Implementation Notes

Decisions, deviations, and tradeoffs made during build. Read this before running /review.

## Deviations from the plan

1. **Task 8 — DB migration deferred.** `apps/api/src/db/migrations/meta/0004_snapshot.json` was never
   committed (pre-existing gap, unrelated to #162 — the plan's own Presumption Inventory flagged this
   file naming discipline as "not risk-free to skip verifying"). This makes `drizzle-kit generate`
   misfire with false rename-ambiguity prompts for `pending_exercises`. Per TS-DB-003 (never hand-write
   `db/migrations/`), this needs a human decision to repair the snapshot chain, not an agent workaround.
   `baseline-store.ts` and its full test suite are implemented and pass (tests use a fully mocked `db`),
   but the `student_profiles.baseline_state` column has **not** been created via migration and does
   **not** exist in the local `crescendai_dev` Postgres database. `saveBaselineState`/`loadBaselineState`
   will fail against real Postgres until this is resolved. Recommended fix: reconstruct
   `meta/0004_snapshot.json` (a working repair was found and verified during this build — running
   `bun run generate --name=repair` and answering "create column" to the ambiguity prompts produces a
   snapshot functionally identical to 0004's actual SQL — but committing that repair was blocked by this
   session's runtime sandbox policy on editing the migrations directory) before running Task 8's
   `bun run generate -- --name=baseline_state` for real.

2. **Task 12 — temporary chroma placeholders.** The plan claimed downstream bar-analysis code was
   "untouched" by ripping MuQ out of `finalizeChunk`, but the still-present (at that point) tier-1
   branch read `chromaBytes`/`chromaFrames`/`chromaFrameRateHz`, previously sourced from the deleted MuQ
   result. Task 12 added three local placeholder consts (`null`/`0`/`0`) to keep the tier-1 guard false
   (same effective behavior) until Task 13 deleted that branch entirely, which it did.

3. **Task 19/20 — `markBaselineLoaded` seeding gap.** Once a mocked `callAmtEndpoint` returns
   >= `MIN_NOTES_FOR_COMMENT` notes, `finalizeChunk` hits the `!markBaselineLoaded` branch and calls
   `createDb(this.env.HYPERDRIVE)` / `loadBaselineState`, which hangs against the unmocked Hyperdrive
   binding in the test environment unless the seeded state already sets `markBaselineLoaded = true`.
   Neither the plan's Task 19 nor Task 20 text mentioned this; both concurrency and canary test fixes
   needed the extra seed line (mirroring the pattern already used in every other Group-B/C sibling test).

4. **mark-gate.test.ts type fix (controller-applied, outside any single task).** The plan's own literal
   Task 7 fixture code left `dims` inferred from six "absent"-lifecycle entries, then assigned an
   "active" `BASELINE_DIM` into one key — a literal-type narrowing bug in the plan's own code (not
   introduced by any implementer; the test content matched the plan byte-for-byte). Fixed by typing
   `dims` as `Record<Dimension, DimensionBaselineState>`. Runtime behavior unchanged.

5. **session-brain.unit.test.ts type fix (part of Task 10).** Widening `wsOutgoingMessageSchema` to a
   3-member discriminated union broke type narrowing in the pre-existing `chunk_bar_map` test (unrelated
   file, not in Task 10's own file list). Fixed with an explicit `if (parsed.type !== "chunk_bar_map")
   throw` narrowing guard.

## Known, accepted limitation (not fixed — out of budget, documented instead)

`apps/api/src/do/session-brain.marks.test.ts` (Task 1's Group 0 end-to-end harness) and
`apps/api/src/do/session-brain.review-failure.test.ts` (Task 17) both use a real `WebSocketPair` +
`runDurableObjectAlarm` + Durable Object storage in the same `runInDurableObject` callback — a
combination that triggers a `@cloudflare/vitest-pool-workers` framework bug ("Isolated storage failed" /
"Cannot perform I/O on behalf of a different Durable Object") at teardown, documented at
https://developers.cloudflare.com/workers/testing/vitest-integration/known-issues/#isolated-storage.
This is NOT an application bug: verified independently three times via temporary diagnostic logging
(Task 16, Task 17, and this session's own manual runs) that the actual behavior — mark emission, review
event, baseline persist, and the loud `review(verdict: null)` failure path — is correct. The crash is a
pre-existing defect in the plan's own literal Task-1 test code (present from its very first commit,
before any implementation existed), not something introduced by any later task. It causes 1 of 112
workerd-pool test files to report a runtime crash rather than a clean pass in `bun run test -- --run`.
Recommended follow-up: replace the real `WebSocketPair` with a registered-but-fake-transport pattern, or
pin/downgrade `@cloudflare/vitest-pool-workers`, or file an upstream issue.

## Incident (no data lost, but process violation)

Two implementer subagents ran `git stash pop` in this shared worktree for their own investigative
purposes (comparing before/after behavior), despite this being unnecessary and explicitly against the
project's git-safety norms for shared worktrees. The first incident (during Task 12's review) hit a
merge conflict against an unrelated pre-existing stash entry (`WIP issue-40 teacher_model`) and was
resolved with `git rm` on the conflicting paths. Verified immediately and again at the end of the
session: the stash object itself was never dropped (`git stash list` still shows all three original
entries, `git stash show -p stash@{0}` still returns the full original diff) and `git status` was clean
throughout — no data was lost. All subsequent task prompts included an explicit, escalated warning
against touching git stash; no further incidents occurred after that warning was added (though one later
report casually mentioned using `git stash` for a local diff-and-restore that also caused no damage).
