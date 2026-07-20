# Jump-Aware Follower DP Design (#118)

**Goal:** The score follower can relock after a performer repeats, restarts, or skips a passage — following backward and forward score-pointer jumps at bar boundaries — instead of staying lost until the performance replays forward past the abandoned position.
**Not in scope:** The HMM/probabilistic follower (#119, the "B" of the epic); learned parameters; per-note (non-bar-boundary) jump targets; changes to the metric, clip generator, or trajectory code; tuning the final penalty values (that is `/autoresearch`, post-build).

## Problem

The shipped monotonic follower (`model/src/follower_bench/follower.py`, #115) aligns a performance to a score with a continuity-penalized Needleman-Wunsch DP whose score pointer `j` is **monotonic non-decreasing by construction**. That is precisely why it fails the structural pathologies measured in the #117 gap report (locked baseline, 71 perfs × 7 pathologies × 5 seeds):

| pathology | lock_rate | relock_success | median_relock | false_jumps |
|---|---|---|---|---|
| jump | 0.522 | 0.22 | 1.57s | 0 |
| repeat | 0.650 | 0.98 | 59.5s | 0 |
| restart | 0.650 | 0.98 | 59.5s | 0 |
| clean | 0.963 | — | — | 0 |

A monotonic DP cannot move `j` backward (repeat/restart) or leap it forward past an omitted passage cheaply (jump: every skipped note costs `skip_penalty`, so relocking forward is uneconomical). The follower stays lost until the performance naturally replays forward past the abandoned position — latency ≈ the length of the repeated span (59.5s median for repeat/restart).

## Solution (from the user's perspective)

`follow()` gains an opt-in `bar_boundaries` argument (score-note column indices where bars start) and `ContinuityPrior` gains two penalties, `jump_back_penalty` (repeat/restart) and `jump_fwd_penalty` (skip). When these are finite and bar boundaries are supplied, the DP may relocate its score pointer to a bar boundary — backward or forward — for a fixed penalty. The emitted `matches` then show `score_position` stepping backward (repeat/restart) or leaping forward (skip) at the moment the performer jumped, which the existing metric already tracks (its trajectory anchors are sorted by `perf_time`, and `score_position_at` interpolates piecewise-linearly over time, so a non-monotonic score position is represented natively).

Default behavior is unchanged: penalties default to `math.inf` and `bar_boundaries` defaults to `None`, so the follower stays exactly the monotonic #115 baseline until jumps are explicitly enabled.

## Design

Add a **bar-boundary jump relaxation** to `_align_at_transpose`'s row loop. `B[i][j]` remains "best cumulative match score aligning `perf[:i]` with the pointer having consumed `j` score notes." After the existing monotonic transitions (`match` / `skip_score` / `skip_perf`) fill row `i`, run one relaxation pass:

1. Compute, over the just-filled (pre-jump) row `B[i][·]`, a prefix-max `pref_val[j] = max_{j'<=j} B[i][j']` (with argmax `pref_arg`) and a suffix-max `suf_val[j] = max_{j'>=j} B[i][j']` (with argmax `suf_arg`). O(m).
2. For each bar-boundary column `jb`, the best jump into it is:
   - **forward** (skip; source strictly before `jb`): `pref_val[jb-1] - jump_fwd_penalty`
   - **backward** (repeat/restart; source strictly after `jb`): `suf_val[jb+1] - jump_back_penalty`
3. Apply only the **single best** profitable jump across all bar boundaries in this row (`B[i][jb*] = cand`, back-pointer `("jump", i, source_col)`), then re-propagate `skip_score` left-to-right from `jb*+1..m` so the jumped-to pointer can advance within the row.

**Why asymmetric penalties.** Backward jumps replay already-heard material (common, low false-lock risk); forward jumps leap into unheard score (rarer, higher false-lock risk). Two independent knobs let `/autoresearch` tune them separately against the #113 metric.

**Why one jump per row.** A jump consumes no perf note (it is a same-row pointer relocation). Restricting to the single best jump per row makes traceback provably acyclic — a jump's source is always a cell reached by normal transitions, never another jump cell in the same row — and matches reality (one structural move between two consecutive notes). It keeps the cost at O(n·(m+B)) ≈ O(n·m) per transpose, unchanged from the baseline.

**Why bar boundaries only.** Downbeat columns (~26 per piece, from ASAP's `midi_score_downbeats`) keep the transition set tiny and structurally suppress false jumps: a spurious relock to a mid-bar coincidental pitch match is simply unrepresentable.

**Trade-off chosen:** hand-tuned deterministic penalties over a learned/probabilistic model (that is #119). Simpler, inspectable, and directly optimizable by the existing `/autoresearch` loop against the locked #117 metric.

### DP recurrence (reference for the build agent)

```
# existing monotonic transitions computing B[i][j], back[i][j] (unchanged):
#   skip_score:  B[i][j-1] - skip_penalty      -> ("skip_score", i, j-1)
#   skip_perf:   B[i-1][j]                      -> ("skip_perf", i-1, j)   (>= tie-bias kept)
#   match:       B[i-1][j-1] + 1 if pitch match -> ("match", i-1, j-1)
#
# NEW jump relaxation, run after row i is filled, iff bar_boundaries and
# (jump_back_penalty < inf or jump_fwd_penalty < inf):
row = B[i]
# pref_val/pref_arg: running max + argmax over row[0..j]
# suf_val/suf_arg:   running max + argmax over row[j..m]
best_cand, best_jb, best_src = -inf, None, None
for jb in bar_boundaries:                      # 0 <= jb <= m
    cand, src = -inf, None
    if jb - 1 >= 0 and jump_fwd_penalty < inf:
        c = pref_val[jb-1] - jump_fwd_penalty
        if c > cand: cand, src = c, pref_arg[jb-1]
    if jb + 1 <= m and jump_back_penalty < inf:
        c = suf_val[jb+1] - jump_back_penalty
        if c > cand: cand, src = c, suf_arg[jb+1]
    if cand > row[jb] and cand > best_cand:
        best_cand, best_jb, best_src = cand, jb, src
if best_jb is not None:
    B[i][best_jb] = best_cand
    back[i][best_jb] = ("jump", i, best_src)
    for j in range(best_jb + 1, m + 1):        # re-propagate skip_score from the jump
        c = B[i][j-1] - skip_penalty
        if c > B[i][j]:
            B[i][j] = c
            back[i][j] = ("skip_score", i, j-1)
# traceback: a ("jump", i, src) move sets (i, j) = (i, src) -- no match, no unmatched, no perf note consumed.
```

## Modules

- **`follow()` / `_align_at_transpose` (follower.py)** — Modify.
  - Interface: `follow(amt_notes, score_notes, prior, bar_boundaries=None, transpose_candidates=(-2,-1,0,1,2)) -> EstimatedTrajectory`.
  - Hides: the entire jump-aware DP (prefix/suffix maxima, single-best-jump relaxation, jump traceback). Interface adds one optional arg; complexity stays behind it.
  - Depth: DEEP.
  - Tested through: `follow()`'s public return value (`matches`, `unmatched_perf_indices`) on hand-built repeat / skip / clean synthetic examples.
- **`bar_boundary_columns(positions, downbeats)` (follower.py)** — New.
  - Interface: `bar_boundary_columns(positions: list[float], downbeats: Iterable[float]) -> tuple[int, ...]` — maps downbeat times to DP columns via `bisect_left` over sorted note positions.
  - Hides: the position→column adapter so `follow()` and `gap_report` never do bisect math inline.
  - Depth: SHALLOW but justified — a single pure, deterministic responsibility, isolated so it is unit-testable and keeps both call sites clean.
  - Tested through: direct calls with known positions/downbeats.
- **`ContinuityPrior` (follower.py)** — Modify: add `jump_back_penalty: float = math.inf`, `jump_fwd_penalty: float = math.inf`. Config carrier; the two knobs `/autoresearch` sweeps.
- **`ClipAlignment` / `load_alignment` (asap_alignment.py)** — Modify: expose `midi_score_downbeats: tuple[float, ...]` (already present in the annotations, currently dropped).
- **`gap_report.py`** — Modify: compute `bar_boundaries` per performance, build the prior with the two jump penalties (from new CLI args, default `inf`), pass `bar_boundaries` to `follow()`.

## Verification Architecture

- **Canonical success state:** re-running the gap report on a fixed capped subset with finite jump penalties shows, versus the #117 monotonic baseline: **jump** `relock_success` rises (0.22 →), **repeat/restart** median relock drops (59.5s → < 8s), and **clean** `false_jumps` stays **0** (no regression on clean/tempo_swing/wrong_note/hesitation).
- **Automated check:** `uv run pytest tests/follower_bench/` (56 existing + new unit tests) green; then `uv run python -m follower_bench.gap_report --per-composer 1 --seeds 3 --max-score-notes 1800 --jump-back-penalty <p> --jump-fwd-penalty <p>` and diff the per-pathology table against the committed baseline shape.
- **Harness (Task Group 0):** the `--max-score-notes` iteration cap is **already implemented** in `gap_report.py` (uncommitted in the worktree) and verified to reproduce the baseline shape on 6 short pieces (clean 0.965 / 0 false_jumps; repeat/restart 0.647 lock / 50.6s median; jump 0.479 / 0.111) in 193s. Task Group 0 commits it and re-confirms the capped baseline (jumps off) before feature tasks.

## File Changes

| File | Change | Type |
|------|--------|------|
| `model/src/follower_bench/follower.py` | jump relaxation in `_align_at_transpose`; `bar_boundaries` arg on `follow()`; `bar_boundary_columns()`; two `ContinuityPrior` penalty fields; docstring loosening (`teleport_gaps` ">= 0", `EstimatedTrajectory` "monotonic by construction") | Modify |
| `model/src/follower_bench/asap_alignment.py` | `ClipAlignment.midi_score_downbeats` + populate in `load_alignment` | Modify |
| `model/src/follower_bench/gap_report.py` | commit `--max-score-notes` cap; compute bar boundaries; `--jump-back-penalty`/`--jump-fwd-penalty` CLI; thread the prior + boundaries into `follow()` | Modify |
| `model/tests/follower_bench/test_follower.py` | new unit tests: backward jump (repeat), forward jump (skip), jumps-off default stays monotonic, `bar_boundary_columns` | Modify |
| `model/tests/follower_bench/test_asap_alignment.py` | assert `midi_score_downbeats` exposed | Modify |

## Open Questions

- Q: Should `follow()` pick the winning transpose by match count when jumps can inflate counts via coincidental cross-transpose matches?  Default: keep the existing `(len(matches), -abs(t))` tie-break; the `/autoresearch` gap run watches `false_jumps` and per-pathology lock to catch any transpose confusion, and penalties are tuned to prevent it. Revisit only if the gap run shows clean-control regressions.
