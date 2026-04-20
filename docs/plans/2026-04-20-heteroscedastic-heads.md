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

- [ ] Write `HeteroscedasticHead`. Softplus floor on σ prevents collapse.
- [ ] Write `gaussian_nll_loss`. Standard formulation:
  `0.5 * (log(2πσ²) + (y - μ)² / σ²)`.
- [ ] Unit tests: zero-grad at target, gradient sign matches residual sign,
  σ→0 is prevented by the floor.

### P1 — Wire into training + swap loss (1 day)

- [ ] Add `use_gaussian_head` flag to the LightningModule config. When true:
  head is `HeteroscedasticHead`, regression loss is Gaussian NLL, per-dim BCE
  is disabled.
- [ ] Re-run A1-Max on the current best mix ratio with the new head. Verify
  training converges — the NLL can destabilize if σ explodes early.

### P2 — Calibration metrics (1 day)

- [ ] Implement ECE: bin predictions by σ, compute `|mean(residual) - mean(σ)|`
  per bin, weighted-average.
- [ ] Reliability diagram: scatter of predicted σ vs empirical residual std,
  per dim. If the line is y=x, σ is calibrated.
- [ ] Wire both into `evaluate_model` so they emit in the same JSON blob as
  Chunk A diagnostics.

### P3 — Harness consumption (1.5 days)

- [ ] Derive per-dim σ thresholds from validation: the 75th percentile σ on
  PercePiano val becomes the "surface / suppress" threshold. Persist in
  checkpoint metadata under `confidence_thresholds`.
- [ ] Implement `confidence_gate.ts`. Test: given a stub (μ, σ) payload, verify
  low-σ dims pass through and high-σ dims are suppressed.
- [ ] Integrate with `synthesis.ts`: dims suppressed by the gate are removed
  from the teacher payload, *and* the teacher is informed in the system prompt
  that some dims were suppressed. The teacher must not pretend to have info it
  doesn't.

### P4 — OOD calibration test (0.5 day)

- [ ] Run `ood_harness.run_ood_test` on the first OOD practice clips (once
  Chunk B has data). ECE on OOD vs PercePiano tells you whether σ generalizes
  or whether the model is overconfident on practice audio.

## Exit Criteria

- Per-dim ECE ≤ 0.05 on PercePiano validation.
- Per-dim ECE ≤ 0.10 on OOD practice set (looser — OOD is meant to expand σ).
- σ responds monotonically to Gaussian input noise (robustness check in
  `evaluation.py:run_robustness_check`).
- Harness-side: a stubbed high-σ input never reaches the teacher's per-dim
  summary.

## Risks

- **σ collapses to ε.** Gaussian NLL has a known exploit: shrink σ and eat the
  penalty on residual. Softplus floor + warmup epochs with fixed σ=0.1 mitigate.
- **Teacher voice doesn't adapt to suppressed dims.** The system prompt update
  is necessary but not sufficient. Needs an eval pass with `apps/evals/`.
- **Calibration is per-dim and fragile.** ECE can look fine on aggregated data
  while individual dims are miscalibrated. Always report per-dim, never just
  the mean.
