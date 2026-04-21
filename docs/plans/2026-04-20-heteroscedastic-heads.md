# Heteroscedastic Output Heads — (μ, σ) Per Dimension

**Goal:** Replace the scalar sigmoid regression head with a Gaussian head that
outputs `(μ, σ)` per dimension. The harness consumes σ to decide whether per-dim
feedback has enough information to surface, or should degrade to a single
overall-quality observation.

**North star metric:** expected calibration error (ECE) per dimension. Target:
ECE ≤ 0.05 on all 6 dims on PercePiano validation. A well-calibrated σ is the
whole point — miscalibrated σ is worse than no σ at all.

## Why

- **Dimension collapse is accepted, not fixed.** The other three plans reduce
  collapse but cannot eliminate it (per `docs/model/06-label-quality.md`). The
  harness has to decide, per-session, whether to surface per-dim feedback. That
  decision needs a scalar confidence per dim.
- **Scalar sigmoid hides uncertainty.** The current head says "dynamics = 0.63"
  whether the input was 30s of Chopin from a concert pianist or 15s of noise.
  A Gaussian head with `σ=0.02` vs `σ=0.18` on those two cases gives the harness
  the lever it needs.
- **Gates the practice-augmentation win.** Practice clips are the OOD distribution.
  We expect σ to grow on practice inputs — if it doesn't, the model is
  overconfident, which is the failure mode we specifically need to surface
  before shipping feedback to beta users.

## Scope

**In scope.** New head module, Gaussian NLL loss, calibration metrics,
reliability-diagram tooling, and harness consumption path wired into
`apps/api/` so the production surface uses σ.

**Out of scope.** Anything requiring new labels. This is a pure
modeling-and-calibration change.

## File Structure

### New Files

| File | Responsibility |
|---|---|
| `model/src/model_improvement/heads.py` | `HeteroscedasticHead` nn.Module: two `Linear(hidden, num_dims)` — one for μ, one for log-σ². Output shape `(B, 2, num_dims)`. Softplus on log-σ² with a small ε floor to prevent σ→0 exploit. |
| `model/src/model_improvement/calibration.py` | `expected_calibration_error()`, `reliability_diagram()`, `per_dim_calibration_report()`. Mirrors scikit-learn's calibration API so it's recognizable. |
| `apps/api/src/services/confidence_gate.ts` | Consume (μ, σ) from inference response, compute per-dim `should_surface = σ < threshold`. Threshold per-dim, read from checkpoint metadata. |

### Modified Files

| File | Change |
|---|---|
| `model/src/model_improvement/audio_encoders.py:MuQLoRAMaxModel` | Replace `self.regression_head = nn.Sequential(Linear, Sigmoid)` with `self.head = HeteroscedasticHead(...)`. `predict_scores` returns `(μ, σ)` tuple. |
| `model/src/model_improvement/losses.py` | Add `gaussian_nll_loss(mu, log_var, target)`; replace CCC/BCE regression term in the composite loss when `use_gaussian_head: bool = True`. |
| `model/src/model_improvement/evaluation.py:evaluate_model` | When `predict_fn` returns `(μ, σ)`, route μ to the existing pairwise/r2 path and σ to a new `calibration` block. Back-compat: if predict_fn returns a bare tensor, fall back to scalar path. |
| `apps/inference/muq/handler.py` | Return σ alongside μ in the MuQ HF inference response. Update the Pydantic response schema. |
| `apps/api/src/practice/synthesis.ts` | Before passing scores to the teacher prompt, run the confidence gate. Dims below threshold are omitted from the summary payload. |

## Phases

### P0 — Head + loss (1 day)

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

- Per-dim ECE ≤ 0.05 on PercePiano validation.
- Per-dim ECE ≤ 0.10 on OOD practice set (looser — OOD is meant to expand σ).
- σ responds monotonically to Gaussian input noise (robustness check in
  `evaluation.py:run_robustness_check`).
- Harness-side: a stubbed high-σ input never reaches the teacher's per-dim
  summary.

## Execution Timing

**Infrastructure:** not yet merged. This plan is ready to implement — nothing
in the Wave 1 code changes (PercePiano mix, SemiSupCon, Practice Augmentation)
blocks the head/loss swap. The `apply_to_tiers` gate already merged in
`losses.py` composes cleanly with Gaussian NLL.

**Experimental runs (ECE sweeps, per-dim calibration reports) are gated on T5
labeling completion.** Calibration quality depends on the label distribution;
running ECE on a partial T5 pool would give a miscalibrated σ baseline that
Wave 1's reweighting then invalidates. Order of operations at training time:

1. T5 labeling complete + κ ≥ 0.6
2. PercePiano mix sweep locks winning ratio
3. SemiSupCon pretraining produces the encoder checkpoint
4. Practice-augmentation corrupted audio rendered
5. **Then** Heteroscedastic heads fine-tune on the Wave-1-locked configuration,
   and ECE is measured against both clean and OOD practice sets.

Harness wiring (`confidence_gate.ts`, `synthesis.ts`) can land independently
once σ shape is confirmed, but threshold values are frozen only after
calibration runs complete.

## Risks

- **σ collapses to ε.** Gaussian NLL has a known exploit: shrink σ and eat the
  penalty on residual. Softplus floor + warmup epochs with fixed σ=0.1 mitigate.
- **Teacher voice doesn't adapt to suppressed dims.** The system prompt update
  is necessary but not sufficient. Needs an eval pass with `apps/evals/`.
- **Calibration is per-dim and fragile.** ECE can look fine on aggregated data
  while individual dims are miscalibrated. Always report per-dim, never just
  the mean.
