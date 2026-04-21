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

        <!-- BASELINE_DIAGNOSTICS -->
        #### A1-Max baseline diagnostics (config: `A1max_r32_L7-12_ls0.1`, 4-fold CV)

        | Metric | Value |
        |--------|-------|
        | `dimension_collapse_mean` | **0.3546** |
        | `dimension_collapse_per_fold` | 0.3127, 0.3572, 0.3100, 0.4387 |
        | Pairwise accuracy (4-fold mean) | 0.8027 |
        | R² (4-fold mean) | -0.1905 |
        | Skill discrimination Cohen's d | `skipped` — no_tier_labels
  Requires `data/evals/ood_practice/labels.json` populated with T5 tier labels. |

        **Per-dimension prediction correlation matrix (element-wise mean across folds):**

        ```
                      dynamic   timing  pedalin  articul  phrasin  interpr
dynamics        1.000   -0.071   -0.248   -0.004   -0.334   -0.474
timing         -0.071    1.000    0.313    0.272    0.351    0.589
pedaling       -0.248    0.313    1.000    0.224    0.238    0.642
articulation   -0.004    0.272    0.224    1.000    0.578    0.223
phrasing       -0.334    0.351    0.238    0.578    1.000    0.655
interpretation -0.474    0.589    0.642    0.223    0.655    1.000
        ```

        > Numbers captured from `data/results/a1_max_sweep_results.json`.
        > Re-run `model/scripts/stamp_baseline_diagnostics.py` to refresh after the sweep completes.


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

## Execution Timing

**Infrastructure:** merged (MixWeightedSampler, `apply_to_tiers`, sweep grid).

**Experimental runs (the 216-config sweep) are gated on T5 labeling completion.**
The mix-ratio sweep reweights T2/T5 tiers; running it before T5 labels stabilize
would just measure label noise, not the mix effect. Kick off the sweep once the
T5 single-ordinal pass is complete (tracked in `model/data/labels/t5/label_log.jsonl`)
and `kappa_report` rolling κ ≥ 0.6.

Until then, only the baseline diagnostics block above is populated — all other
mix ratios pending. See `docs/plans/2026-04-20-model-year-roadmap.md` for the
Q2-ships-code / Q3-runs-experiments split.
