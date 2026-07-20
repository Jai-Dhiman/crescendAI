# HMM Follower Design (#119, epic #108)

**Goal:** Add an opt-in Viterbi-HMM decoder beside the untouched additive DP in
`model/src/follower_bench/follower.py`, replacing #118's hand-tuned additive
jump penalties with log-probability emission/transition costs and emitting a
calibrated per-note position confidence read off a forward-backward posterior.

**Not in scope:**
- Performance timing / tempo model (pitch-sequence only, exactly the #118 grid).
- Parameter tuning: the build lands the mechanism with sane hand-set default
  `HmmParams`; a SEPARATE post-ship `/autoresearch` pass tunes the ~7 params for
  gap_report parity. No tuning tasks in this plan.
- WASM port (#P3).
- Modifying the additive `follow()` / `_align_at_transpose` decode behavior.

## Problem

#118 shipped a jump-aware additive DP whose jump cost is a flat, hand-tuned
`jump_back_penalty` / `jump_fwd_penalty` (one global pair). #118's autoresearch
(14 runs) proved **no single global pair works**: forward-jump recovery needs
`jump_fwd < jump_back`, which always makes backward jumps too cheap and blows up
repeat relock (5.74s -> 16.35s) -- the "repeat-cliff". The root cause is a
structural one: in the additive DP an unmatched perf note (`skip_perf`) is
**free (cost 0)**, so a coincidental forward-jump-during-a-repeat pays nothing
for the replayed notes it leaves unexplained. Additionally, the additive DP
emits no confidence, so a downstream consumer cannot tell when to trust the
cursor.

## Solution (from the user's perspective)

`gap_report --hmm` runs a Viterbi-HMM follower whose transition costs are
log-probabilities and whose `p_ins` parameter charges every unexplained perf
note (no free skip). Each matched note carries a `confidence` in `[0, 1]` (the
forward-backward posterior mass on the decoded column). A new calibration
measurement reports Spearman rho(confidence, -|position error|) plus a
risk-coverage curve, proving the confidence is usable to gate the cursor.
Default `gap_report` (no `--hmm`) is byte-for-byte the untouched additive path,
so #117 reproduces and all 61 existing tests stay green.

## Design

Pitch-only HMM on the same perf-note x score-note grid as the additive DP.
Hidden state after perf note `i` = score column `j` (0..m; column `j` means the
next unconsumed score note is index `j`). All arithmetic in log-space.

**Emission** (a match move consuming perf note `i-1` at score note `j-1`, at
transpose `t`): `log(p_match)` if `score[j-1].pitch + t == perf[i-1].pitch` else
`log(p_confuse)`.

**Transitions** producing `V[i][j]` (mirrors the additive `match/skip_score/
skip_perf` moves, but log-prob weighted and with no free skip):
- match: `V[i-1][j-1] + log(p_adv) + emit(i, j)` -- consume perf note, advance one column.
- insertion: `V[i-1][j] + log(p_ins)` -- consume a spurious/noise perf note, pointer stays. **Replaces #118's free `skip_perf`; `p_ins < 1` is the load-bearing fix that defeats the repeat-cliff.**
- deletion: `V[i][j-1] + log(p_del)` -- a score note not played, pointer advances (analog of `skip_penalty`).

**Jumps (bar-boundary relocation, the #118 mechanism, reformulated for a correct
posterior):** a jump is a **row-advancing "jump-into-match"** -- it consumes
perf note `i-1` and lands it as a match on the first note of a bar
(`bar_boundary_columns`). For a target bar-start note index `b`, the jump comes
from any source column `s`: backward (`s > b`, repeat/restart) costs
`log(p_jump_back)`, forward (`s < b`, skip) costs `log(p_jump_fwd)`:
`V[i][b+1] = max_s( V[i-1][s] + (log p_jump_back if s>b else log p_jump_fwd) ) + log(p_adv) + emit(i, b+1)`.
Because every jump edge advances `i`, the `(i, j)` grid stays a DAG (topological
order: increasing `i`, then increasing `j` within a row), so both Viterbi
(max-product) and forward-backward (sum-product) are cycle-free and correct.
**This deliberately differs from #118's same-row `_relax_row_jumps`** (which is
correct for max-product but would make a summed posterior cycle-prone); the
acceptance criterion is the #113 metric, not mechanical identity. Jumps disable
by setting `p_jump_back = p_jump_fwd = 0` (log `-inf`) -> monotonic HMM.

**Confidence:** after Viterbi picks the winning transpose, a sum-product
forward (`alpha`) / backward (`beta`) pass over the same edges (log-sum-exp with
max-shift) gives `gamma[i][j] = alpha[i][j] + beta[i][j] - Z`,
`Z = logsumexp_j alpha[n][j]`. Each decoded match consuming perf note `i-1` at
column `j` carries `confidence = exp(gamma[i][j])` in `[0, 1]`.

**Parameters** (`HmmParams` dataclass, probabilities in `(0, 1]`, unnormalized
weights -- the per-step posterior normalizes by construction, so normalization
is unnecessary): `p_match, p_confuse, p_ins, p_del, p_adv, p_jump_back,
p_jump_fwd`. Seven knobs vs #118's three. Sane hand-set defaults land the
mechanism; autoresearch tunes later.

## Modules

- **`follower_bench.hmm`** (New) -- the HMM decoder.
  - Interface: `HmmParams` (dataclass), `follow_hmm(amt_notes, score_notes, params, bar_boundaries=None, transpose_candidates=(-2,-1,0,1,2)) -> EstimatedTrajectory` (matches carry `confidence`), `column_posteriors(amt_notes, score_notes, params, transpose, bar_boundaries=None) -> list[list[float]]` (the gamma matrix as probabilities), `alignment_logprob(amt_notes, score_notes, params, transpose, bar_boundaries=None) -> float` (the log marginal `Z`).
  - Hides: log-space Viterbi, the DAG jump-into-match transitions, traceback, transpose search, and the log-sum-exp forward-backward.
  - Tested through: `follow_hmm` correspondence + confidence, `column_posteriors` sums-to-1, `alignment_logprob` no-free-skip.
  - Depth: DEEP (three small pure functions hide the whole probabilistic decoder).
- **`follower_bench.calibration`** (New) -- calibration measurement, kept out of
  `metric.py` so the position scorer stays follower-agnostic.
  - Interface: `calibration_stats(matches, clip, *, sample_hz=SAMPLE_HZ, coverage_fractions=...) -> CalibrationStats` (Spearman rho + risk-coverage points).
  - Hides: zero-order-hold confidence track, grid sampling parallel to `score_clip`, Spearman + risk-coverage math.
  - Depth: DEEP.
- **`follower_bench.follower`** (Modify, minimal) -- `MatchedNote` gains
  `confidence: float | None = None` (default `None` keeps every existing
  construction and test unaffected; the additive path leaves it `None`).
- **`follower_bench.gap_report`** (Modify) -- `--hmm` flag routes cells through
  `follow_hmm` and reports a calibration line; default path untouched.

## Verification Architecture

- **Canonical success state (build gate):** the 8 behavioral unit tests below
  pass, and all 61 existing `follower_bench` tests stay green.
- **Automated check:** `cd model && uv run pytest tests/follower_bench/`.
- **Harness:** no separate golden-fixture harness is buildable (the decoder IS
  the thing under test); the unit tests themselves are the harness. Task Group 0
  is the foundation slice (`HmmParams` + `MatchedNote.confidence`).
- **Post-ship gate (separate, NOT this plan):** `gap_report --hmm --per-composer
  1 --seeds 3 --max-score-notes 1800 --workers 6` matches-or-beats the #118
  5.0/8.0 baseline (clean false_jumps=0; repeat/restart median <=8s; jump
  relock_success >= 0.222) AND calibration rho >= ~0.3, reached via
  `/autoresearch` on the seven params.

**Behavioral tests:**
1. Monotonic HMM (jumps off) decodes the obvious correspondence on a small synthetic, skipping an unmatchable note as an insertion.
2. Backward-jump relock after a repeat (jumps-on vs jumps-off contrast).
3. Forward-jump relock after a long skipped passage (jumps-on vs jumps-off contrast).
4. Cliff-crossing: one clip with BOTH a repeat and a forward skip; HMM (one param set) relocks both, while additive `follow()` at the shipped 5.0/8.0 pair misses the forward skip.
5. `column_posteriors`: each row sums to ~1 (forward-backward correctness).
6. Calibration direction: a high-confidence/low-error region vs a low-confidence/high-error region yields rho(confidence, -|error|) > 0 and top-20%-confident median error < overall.
7. No-free-skip: inserting one spurious noise note drops `alignment_logprob` by ~`log(p_ins)`, not 0.
8. gap_report routing seam: `_follow_for_cell(use_hmm=False)` uses the additive DP, `use_hmm=True` uses `follow_hmm`, verified on a tiny synthetic (no ASAP data).

## File Changes

| File | Change | Type |
|------|--------|------|
| `model/src/follower_bench/hmm.py` | HMM decoder: `HmmParams`, `follow_hmm`, `column_posteriors`, `alignment_logprob` | New |
| `model/src/follower_bench/calibration.py` | `calibration_stats`, `CalibrationStats` | New |
| `model/src/follower_bench/follower.py` | `MatchedNote.confidence: float \| None = None` | Modify |
| `model/src/follower_bench/gap_report.py` | `--hmm` flag, `_follow_for_cell` seam, calibration line | Modify |
| `model/tests/follower_bench/test_hmm.py` | tests 1-5, 7 | New |
| `model/tests/follower_bench/test_calibration.py` | test 6 | New |
| `model/tests/follower_bench/test_gap_report_routing.py` | test 8 | New |

## Open Questions

- Q: Does the jump-into-match reformulation (vs #118's same-row relocation) hurt
  metric parity? Default: accept it -- it is the only formulation with a
  provably correct posterior, and post-ship autoresearch tunes params against
  the same gap_report; if parity is unreachable, revisit in a follow-up.
- Q: Is `scipy.stats.spearmanr` available? Default: yes (partitura/lightning
  dependency); if absent the build falls back to a numpy rank-based Spearman.
