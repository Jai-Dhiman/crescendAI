# Implementation Notes

Decisions, deviations, and tradeoffs made during build. Read this before running /review.

## Build environment note

No subagent-dispatch (Task) tool was available in this environment, so the controller executed each task's TDD cycle directly (write failing test, watch fail, implement, watch pass, commit) and performed spec + quality self-review per task by re-reading the diff and re-running tests. Two-stage review discipline was preserved in spirit; the work was not delegated to fresh subagents.

The WASM `pkg/` directory is gitignored, so a fresh worktree has no `pkg/`. Ran `bun run build:wasm` during setup to generate it. Task 2's commit therefore does not include `pkg/` (git ignores it); the export is verified by rebuilding and running the workerd test.

## Task 1: Rust select_session_moments

Implemented verbatim per plan. Added `select_session_moments` (additive, before `select_positive_moment`; `select_teaching_moment` untouched) and `Candidate::to_session_moment`. Reuses existing `max_negative_deviation` and `select_positive_moment` for the all-at-mean fallback. 11 teaching_moments tests pass (7 pre-existing + 4 new). No deviations.

## Task 6: buildSynthesisFraming reference-mode guardrail

Implemented verbatim. `sessionData` typed as `Record<string, unknown>` to allow the conditional `reference_mode` key. 8th param defaults to `null`, keeping the single existing caller byte-compatible. 15 prompts tests pass (12 pre-existing + 3 new).

## Task 4: computeSessionDurationMs (+ eval_score: carry-forward)

Added `computeSessionDurationMs(count)` and `MUQ_CHUNK_MS = 15000`. DEVIATION FROM LITERAL PLAN (honoring the /challenge RISK + carry-forward caution): rather than `computeSessionDurationMs(state.scoredChunks.length)`, I introduced an `effectiveScoredChunks` list in `runSynthesisAndPersist` that falls back to the existing `eval_score:` per-chunk storage keys when `state.scoredChunks` is empty in eval mode (synthesizing `chunkIndex` from numeric-sorted key order, since eval_score values are bare score arrays). Duration is then `computeSessionDurationMs(effectiveScoredChunks.length)`. This unifies the chunk source so both Task 4 (duration) and Task 5 (cold-start) read one list and neither silently no-ops at 0 chunks in eval. The pre-existing `evalContext.scored_chunks` block was refactored to consume the same `effectiveScoredChunks` list — behavior preserved exactly (same numeric sort, same fallback to state.scoredChunks). 12 unit tests pass; tsc clean.

## Task 3: buildColdStartMoments helper

Implemented verbatim per plan. Added `import { SessionAccumulator, type AccumulatedMoment }` (was `SessionAccumulator` only) and the exported `buildColdStartMoments(scoredChunks, max)` function placed after `computeSessionDurationMs`, before the `SessionBrain` class. Computes per-dimension session mean, calls `wasm.selectSessionMoments`, maps `TeachingMoment` -> `AccumulatedMoment` (snake_case bridge fields -> camelCase, barRange=null, analysisTier=3, timestampMs=0, llmAnalysis=null). 14 unit tests pass (2 new + 12 pre-existing). tsc clean for session-brain.ts. No deviations.

## Task 5: DO wiring cold-start branch + thread referenceMode

Implemented 5(a)-(d) per plan. Two deviations, both in the new test file (not in production code):

1. vi.mock hoisting: the plan's `const mockCallAnthropic = vi.fn()` referenced inside the hoisted `vi.mock("./llm")` factory throws `Cannot access 'mockCallAnthropic' before initialization` under this vitest config. Fixed to match the codebase convention in teacher.test.ts: define `callAnthropic: vi.fn()` inside the factory, then retrieve via `const mockCallAnthropic = vi.mocked(callAnthropic)` after the import. Behavior identical — still captures the `system` arg passed to `callAnthropic`.

2. mockResolvedValue type: the plan's value `{ content: [...] }` is not assignable to `AnthropicResponse` (which also requires `stop_reason` and `usage`). Supplied the full shape (`stop_reason: "end_turn"`, `usage: {input_tokens:0, output_tokens:0}`) instead of a cast — type-honest. synthesize only reads `content`, so behavior is unchanged.

Production wiring: the cold-start branch reads `effectiveScoredChunks` (the Task 4 fallback-aware list), NOT raw `state.scoredChunks`, so cold-start moments do not no-op in eval mode (honors the /challenge carry-forward caution). Branch is placed after the sessionDurationMs line and BEFORE the evalContext snapshot of `acc.teachingMoments` and before `acc.topMoments()` is read, so both consumers see the accumulated cold-start moments. Verified: cold-start moments + duration both source from the same fallback-aware list (session-brain.ts lines 1447-1467). 2 new tests + 28-test regression (teacher.test.ts + session-brain.unit.test.ts) all pass; tsc clean for all touched files.

## Build environment note (continuation session)

This was a resumed/interrupted build. A prior session committed Tasks 1, 6, 4, 2 (+ notes init) and left Task 3's test uncommitted with the impl unwritten. This session completed Task 3 and Task 5. As in the prior session, no Task subagent-dispatch tool was available, so the controller executed the TDD cycles directly with per-task spec + quality self-review (re-read diff, re-run tests, typecheck). Two-stage review preserved in spirit, not delegated to fresh subagents.
