# Uncertainty and Diagnostics

> Last updated: 2026-04-20

This doc explains how the model reports what it doesn't know, how the harness
consumes those signals, and how the Chunk A diagnostics let us see dimension
collapse rather than pretend it isn't happening.

---

## The 6-vector problem

CrescendAI outputs a 6-dimensional evaluation per clip: dynamics, timing,
pedaling, articulation, phrasing, interpretation. The user-facing harness
surfaces those as 6 separate pieces of feedback.

But per `docs/model/06-label-quality.md`, the 6-vector is partially synthetic:
T5 labels (the largest labeled tier) are derived from a single ordinal.
Training on them teaches the model that all 6 dims correlate. The resulting
`dimension_collapse_score` — mean absolute off-diagonal correlation of
predicted dims — is expected to be high (we measure it; we don't paper over
it).

**The question is not "can we make collapse = 0?" It's "how does the harness
decide, per session, whether the 6-vector carries 6 dimensions of information
or is effectively a scalar replicated 6x?"**

---

## The two signals the harness consumes

### Signal 1 — Per-dim σ (heteroscedastic output head)

Every dim outputs a `(μ, σ)` pair. Plan:
`docs/plans/2026-04-20-heteroscedastic-heads.md`.

- Low σ on a dim → model is confident; surface the per-dim feedback.
- High σ on a dim → model is uncertain; suppress or soften.
- Thresholds are per-dim (some dims are intrinsically noisier) and derived
  from PercePiano validation (75th percentile σ on clean distribution).

### Signal 2 — Session-level collapse indicator

The dimension_collapse_score is a *distribution-level* statistic, computed
over a validation set. At inference time, we don't have a distribution — we
have one clip. But we can approximate session collapse by:

- The variance of the predicted 6-vector's entries (low variance → collapse).
- The σ pattern (all σs high → model didn't find dim-specific signal).
- Historical per-user: if this user's past 10 sessions all showed
  near-identical relative dim rankings, the model isn't learning anything
  user-specific and should fall back to overall quality.

---

## The Chunk A diagnostics (measurement)

Defined in `model/src/model_improvement/evaluation.py`. Emitted by
`a1_max_sweep.py` and `autoresearch_contrastive.py` in every run's JSON.

### per_dimension_correlation

6×6 Pearson correlation matrix of per-dim predictions across held-out samples.
Identity → fully independent dims. All-ones → fully collapsed.

### conditional_independence

Same 6×6 but residualized on the mean label (proxy for overall quality). This
is the critical diagnostic: if per_dimension_correlation is high, we can't
tell whether the dims are genuinely correlated (because quality is correlated)
or the model compressed them. Conditional independence strips the quality
signal; what remains is dim-specific information (if any).

Near-zero matrix → dims carry only overall-quality info (collapse confirmed).
Non-zero off-diagonals → dims carry additional information beyond overall
quality, even after controlling for it.

### dimension_collapse_score

Single scalar: `mean(|off_diag(per_dimension_correlation)|)`. Near 1.0 means
the 6-vector is a scalar replicated 6x. Near 0.0 means fully independent dims.
Published on every sweep leaderboard. Sortable metric for experiment picking.

### skill_discrimination_report

When skill tier labels are provided (e.g. from T5's 1–5 ordinal), per-dim
Cohen's d between adjacent tiers. Answers: "given input from a stronger
player, do the model's predictions actually shift?" If d < 0.4 across all
dims, the model isn't discriminating skill at all — that's an independent
failure mode from collapse and is more urgent to fix.

### gate_value_stats

For fusion models only (once MuQ+Aria fusion lands). Mean, std, and
10-bin histogram of gate activations per dimension. If a dim's gate is always
on one encoder, the other encoder is redundant for that dim. Stuck-at-0 or
stuck-at-1 gates indicate the fusion didn't learn anything per-dim-specific.

---

## Harness interpretation rules

When `inference` returns `{scores, sigmas, collapse_proxy}` for a session:

1. **If collapse_proxy > 0.85 for this session:**
   - Don't surface per-dim scores to the teacher.
   - Pass only the overall-quality signal: mean(scores) with mean(sigmas).
   - The teacher generates a single-paragraph observation, not a 6-bullet list.

2. **Else, for each dim d:**
   - If `sigmas[d] > threshold[d]`: suppress dim d. Teacher told that d was
     suppressed.
   - Else: pass (scores[d], sigmas[d]) to the teacher; allow per-dim
     feedback.

3. **If all dims are suppressed (every sigma above threshold):**
   - Teacher is given a "I can't reliably score this clip" signal.
   - Surface a meta-observation: "this clip is hard to evaluate — try a
     cleaner recording or a passage you know better."

4. **OOD case (clip is from beta user's home, not studio):**
   - `ood_harness.run_ood_test` thresholds may be stricter — once we have OOD
     calibration, the thresholds are different for phone-captured input.
   - Flagged in `ood_source` field; harness reads the flag and picks the
     right threshold table.

---

## What this buys us

- **Visibility before shipping.** We won't hand users 6-dim feedback we
  privately know is 1-dim. The collapse score says "don't trust per-dim here"
  and the harness acts accordingly.
- **Model-independent.** The harness rules are expressed in terms of (μ, σ)
  and collapse_proxy, not in terms of specific model versions. Future encoder
  swaps inherit this logic for free.
- **Tuning surface.** Thresholds are config, not hard-coded. As OOD data
  accumulates we can re-calibrate without re-training.

---

## What this does NOT fix

- **The underlying collapse.** We don't claim to have solved it. We claim to
  measure it and avoid lying to users about it.
- **Miscalibration of σ itself.** A heteroscedastic head can be confidently
  wrong. The ECE metric in `calibration.py` catches this *in aggregate*, not
  per-clip. If σ is systematically under- or over-confident on some sub-
  population, our harness rules fail silently on that sub-population. This is
  one of the reasons partnership-driven real practice data (year 2) matters.
- **The cold-start case.** First-ever session from a new beta user has no
  per-user history; the session-level collapse proxy is only the within-clip
  signal (variance of dim predictions + σ pattern). It's noisier.

Track these limitations explicitly in `docs/model/04-north-star.md`.
