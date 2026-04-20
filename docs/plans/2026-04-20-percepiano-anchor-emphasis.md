# PercePiano Anchor Emphasis — Per-Dim Supervision Reweight

**Goal:** Lift PercePiano from ~20% of the training mix to 30–35% and drop T5's
per-dim contribution to zero, since T5 labels are derived from a single ordinal
and were silently teaching the regression head that all 6 dims = overall quality.

**North star metric:** `dimension_collapse_score` (mean |off-diag| correlation of
per-dim predictions, emitted by the Chunk A diagnostics from
`model/src/model_improvement/evaluation.py`). Target: reduce collapse by ≥0.15
vs the current A1-Max baseline without sacrificing >2pp pairwise accuracy.

## Why

- **The collapse evidence.** PercePiano is the *only* tier with per-rater per-dim
  annotations. T5 ordinals are broadcast across 6 dims with a small offset; T2
  competition placements are scalar. When T5 dominates the mix (~60%) and feeds
  the regression head, gradient updates pull every dimension toward the same
  target. The 6-vector becomes a scalar replicated 6x.
- **Why not relabel T5 per-dim.** Per the solo-dev constraint (`docs/model/06-label-quality.md`),
  per-dim relabeling is out of scope. Relabeling 500+ T5 clips on 6 rubric dims
  is ≈ 6× the labeler-hours of the current 1–5 workflow. Instead, we route
  around the problem: keep T5 for skill-ranking (ListMLE / SemiSupCon), drop it
  from per-dim regression.
- **Why not more PercePiano.** PercePiano has 3 pieces. Over-anchoring on it
  risks piece overfit — the diagnostic flag is per-piece pairwise accuracy vs
  cross-piece. We'll measure this in the sweep.

## Scope

**In scope.** Training-mix config changes + a comparative sweep with Chunk A
diagnostics. Purely a reweighting experiment — no new loss terms, no new
datasets.

**Out of scope.** SemiSupCon (see `2026-04-20-semi-sup-con-loss.md`),
heteroscedastic heads (see `2026-04-20-heteroscedastic-heads.md`), practice
augmentation (see `2026-04-20-practice-augmentation.md`). Those plans can layer
on top of the new mix ratios.

## File Structure

### Modified Files

| File | Change |
|---|---|
| `model/src/model_improvement/a1_max_sweep.py` | Add `PERCEPIANO_MIX_RATIOS = [0.20, 0.30, 0.35]` to the sweep grid; thread through `BASE_CONFIG["percepiano_mix"]`. |
| `model/src/model_improvement/data.py` | Introduce `MixWeightedSampler` that respects a target ratio for PercePiano vs T2 vs T3 vs T5. Existing `HardNegativePairSampler` stays unchanged — it only affects hard negatives, not mix. |
| `model/src/model_improvement/losses.py:DimensionWiseRankingLoss` | Accept `apply_to_tiers: set[str] = {"percepiano"}` parameter. T2/T5 bypass the per-dim BCE term entirely and contribute only to ListMLE + SemiSupCon. |
| `model/src/model_improvement/training.py` | Pass the tier tag through the dataloader so the loss can gate. |

### New Files

| File | Responsibility |
|---|---|
| `model/src/model_improvement/mix_analysis.py` | Post-sweep analysis: for each mix ratio, plot `dimension_collapse_mean` vs `pairwise_mean`, with per-piece pairwise as an overfit flag. |

## Phases

### P0 — Instrument the existing baseline (0.5 day)

- [ ] Run the current A1-Max config (20% PercePiano) through the Chunk A
  diagnostics and record the baseline `dimension_collapse_mean`. This is the
  "before" number we have to beat.
- [ ] Add a `tier` field to every batch item in `data.py` so the loss function
  can identify which tier the sample came from.

### P1 — Plumb the tier gate through the loss (1 day)

- [ ] Extend `DimensionWiseRankingLoss.forward()` to skip the BCE term for
  non-PercePiano tiers. Unit-test: feeding a T5-only batch produces zero
  per-dim BCE gradient.
- [ ] Update `training.py` so the LightningModule reads `batch["tier"]` and
  passes it to the loss. No architecture changes.

### P2 — Mix-ratio sweep (2 days MPS, measured)

- [ ] Run the 3-point mix sweep × 4 folds × 3 LoRA configs from the existing
  grid. That's 36 runs at ~0.5h each on M4 = ~18h, so split across two days.
- [ ] After each fold completes, verify the new `dimension_collapse_per_fold`
  field lands in the results JSON.
- [ ] Gate-keep the jump to 35% on per-piece pairwise — if one piece's pairwise
  drops >3pp from baseline, flag piece overfit.

### P3 — Pick the winner and lock the mix (0.5 day)

- [ ] Run `mix_analysis.py` to produce the collapse-vs-pairwise plot.
- [ ] Update `BASE_CONFIG` in `a1_max_sweep.py` and
  `autoresearch_contrastive.py` to use the winning ratio.
- [ ] Commit a short report to `data/results/percepiano_mix_sweep.md`:
  baseline collapse, winning-config collapse, Δ, per-piece pairwise deltas.

## Exit Criteria

- `dimension_collapse_mean` drops ≥0.15 vs 20% baseline in at least one config.
- `pairwise_mean` does not regress by more than 2pp.
- No single piece in PercePiano regresses by >3pp pairwise (overfit guard).
- Winning mix ratio is locked in `BASE_CONFIG` and referenced by all downstream
  plans (SemiSupCon, heteroscedastic heads).

## Dependencies

- Chunk A diagnostics (`dimension_collapse_score` etc.) must be landed first —
  that's the prerequisite for measuring the win.
- None of the other three week-scale plans depend on this one, but they all
  produce cleaner numbers when run *after* this has landed.
