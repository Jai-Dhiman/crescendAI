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

## Implementation Status

**P0 COMPLETE — 2026-04-20.** All code shipped. Deviations from original spec:

- Head returns `(mu, sigma)` tuple (not `(B, 2, num_dims)` tensor — cleaner for callers).
- `gaussian_nll_loss(mu, sigma, target)` takes `sigma` not `log_var` (wraps `F.gaussian_nll_loss` which takes variance internally).
- Inference server required two additional files beyond `handler.py`:
  `apps/inference/models/loader.py` (new `A1MaxInferenceHeadGaussian`, checkpoint key mapping)
  and `apps/inference/models/inference.py` (updated `predict_with_ensemble` return type).
- Gate fires in `session-brain.ts` at teaching moment accumulation, not `synthesis.ts`.
  `synthesis.ts` only does DB persistence; the gate needs to be upstream of `SessionAccumulator`.
- `apps/inference/models/loader.py` fixed a pre-existing `weights_only=False` → `weights_only=True`.

**Remaining before ECE numbers are meaningful:** T5 labeling complete + training run with
`use_gaussian_head=True`. Gate thresholds (currently 0.15 default) should be frozen after
calibration against PercePiano validation ECE.

## Phases

### P0 — Head + loss (complete)

        See `docs/model/08-uncertainty-and-diagnostics.md` § Baseline snapshot (pre-Wave 1).


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
