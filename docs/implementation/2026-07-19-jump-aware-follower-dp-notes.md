# Implementation Notes — Jump-Aware Follower DP (#118)

Decisions, deviations, and tradeoffs made during build. Read this before running /review.

## Task 2: backward jump (repeat/restart)
Implemented ContinuityPrior jump fields (default inf), follow() bar_boundaries param, backward-only _relax_row_jumps (suffix-max), "jump" traceback. Backward-jump test produces exactly [0,1,2,3,0,1] (cheap) and [0,1,2,3] (expensive/no-jump). Full suite 60 passed. Commit 3c6a63ad.
PROCESS NOTE: the concurrent Task 5 *reviewer* ran `git stash` which transiently reverted Task 2's WIP mid-task; Task 2 re-applied identical spec code and committed cleanly. Verified stash@{0} was fully superseded by HEAD (empty diff) and dropped it. Lesson: review subagents must be read-only (git show/diff by SHA), never stash, in a shared worktree.

Task 2 review: SPEC PASS, QUALITY APPROVED. DP correctness (same-row relocation, one-jump-per-row, source-strictly-after, cycle-freedom, defaults-disable) all independently verified. 60 passed. 1 MINOR (jumps_enabled gate references fwd penalty pre-forward-branch; harmless, resolved by Task 3).

## Task 3: forward jump (skip)
Replaced _relax_row_jumps with prefix+suffix version: forward branch (pref_val[jb-1]-jump_fwd_penalty) competes with backward branch per bar-column; still one jump/row. Forward test yields [0,1,10,11] (cheap) and [0,1]+unmatched(2,3) (inf). 61 passed. Commit e955bca8.

Task 3 review: SPEC PASS, QUALITY APPROVED. Cycle-freedom holds for both directions (fwd source < target, back source > target; both non-jump-propagated cells). 61 passed.

## Task 6: wire jump penalties into gap_report
Committed 62f891fc (subagent). CLI: --jump-back-penalty / --jump-fwd-penalty (default inf = monotonic baseline). _run_performance computes bar_boundaries from alignment.midi_score_downbeats; _run_cell builds the prior. 61 tests pass.

### Integration verification (capped subset: --per-composer 1 --seeds 3 --max-score-notes 1800)
MECHANISM VALIDATED. Penalty is load-bearing (challenge RISK #1 confirmed):
- @ jump_back=1.0 / jump_fwd=1.0 (TOO LOW): repeat/restart relock 50.6s->4.60s BUT clean false_jumps 0->274, tempo/wrong/hesit 0->~820, jump relock 0.111->0.000. Over-eager jumping wrecks non-structural clips. OVERALL FAIL.
- @ jump_back=5.0 / jump_fwd=10.0 (VALIDATED): clean/tempo/wrong_note/hesitation ALL false_jumps=0 (zero regression, lock unchanged 0.965/0.966); repeat/restart lock 0.647->0.934, median relock 50.6s->5.74s (<8s bar MET); jump lock 0.479->0.564, relock 0.111->0.222. OVERALL PASS. Wall 288s (fewer jumps fire than @1.0's 750s).

All three #118 pass conditions met at 5.0/10.0. Autoresearch starting point = jump_back~5.0, jump_fwd~10.0 (asymmetric: forward riskier). Baseline-vs-jump diffs are same-subset (Task 0 monotonic baseline vs these). Full-set (71-piece) numbers not re-run (jump-aware ~2-4x the 9.5h monotonic wall; capped subset is the fair comparison per plan).
