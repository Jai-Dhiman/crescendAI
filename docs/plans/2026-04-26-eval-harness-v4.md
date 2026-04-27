# Eval Harness v4 Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task within a group, sequentially across groups).
> Do NOT start execution until `/challenge` returns `VERDICT: PROCEED`.

**Goal:** Add three eval-harness capabilities — top-2 cluster-based teacher-voice injection from a single source of truth, a 4-condition signal ablation that tests whether MuQ + AMT signals are load-bearing in synthesis, and a binary 8x5 atomic-skill rubric matrix that decomposes synthesis failures.

**Spec:** `docs/specs/2026-04-26-eval-harness-v4-design.md`

**Style:** Follow project standards (`CLAUDE.md`, `apps/api/TS_STYLE.md`). Use `uv` for Python, `bun` for JS, no emojis, explicit exception handling.

## Task Groups

```
Group A (parallel): Task 1, Task 2, Task 3, Task 4, Task 5
Group B (depends on Group A):
  Track Py (sequential): Task 6 -> 7 -> 8 -> 9 -> 10  (all touch teacher_style.py)
  Track Ts (sequential): Task 11 -> 12 -> 13 -> 14 -> 15  (all touch teacher_style.ts)
  Track Compile: Task 16 (depends on Task 1)
  Track Ablation: Task 17, 18, 19 (depend on Tasks 2-4)
  Track Atomic: Task 20 -> 21 (depends on Task 5)
Group C (depends on Group B):
  Task 22 (prompts.ts; depends on Task 15)
  Task 23 (run_eval.py cluster injection; depends on Task 10)
  Task 24 (run_eval.py atomic gate; depends on Task 23 + Task 20)
  Task 25 (justfile recipes; depends on Task 16)
```

The full per-task body (test code, exact failing-error expectations, exact implementation code, commit messages) is committed in five appendix files to keep this index readable; each appendix is self-contained for its task range so the build agent dispatches one subagent per task with the corresponding appendix file as context.

- `docs/plans/2026-04-26-eval-harness-v4/tasks-01-05.md` (Group A, 5 tasks)
- `docs/plans/2026-04-26-eval-harness-v4/tasks-06-10.md` (Python teacher_style track)
- `docs/plans/2026-04-26-eval-harness-v4/tasks-11-15.md` (TS teacher_style track)
- `docs/plans/2026-04-26-eval-harness-v4/tasks-16.md` (compile script)
- `docs/plans/2026-04-26-eval-harness-v4/tasks-17-21.md` (ablation + atomic judge)
- `docs/plans/2026-04-26-eval-harness-v4/tasks-22-25.md` (integrations + just recipes)

Each appendix follows the strict task shape:

```
### Task N: <name>
Group: <A|B|C>
Behavior being verified: <one sentence>
Interface under test: <public API>
Files: Create / Modify / Test
Step 1: failing test (exact code)
Step 2: run command + expected exact failure error
Step 3: minimum implementation (exact code)
Step 4: run command + expected pass
Step 5: commit (exact files + exact message)
```

## Plan self-review notes

- Spec coverage: Tasks 1 + 6-15 cover sub-goal 1; Tasks 2-4 + 17-19 cover sub-goal 2; Tasks 5 + 20-21 + 24 cover sub-goal 3; Tasks 16 + 25 cover infra; Tasks 22-23 cover integration.
- Vertical-slice check: every task has exactly one test + one implementation + one commit.
- Behavior-test check: all tests assert on public function returns or file outputs, not internal state.
- No mocking of internal collaborators: Tasks 17 and 20 use Protocol-typed fakes for the LLM client, which is a system boundary.
- Group correctness: tasks sharing a file are sequential within their track; cross-track parallelism preserved.
- No placeholders: every step in every appendix has exact code or commands.
