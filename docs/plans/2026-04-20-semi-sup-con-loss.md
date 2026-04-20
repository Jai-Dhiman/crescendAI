# SemiSupCon Loss — Labeled Positives in Contrastive Matrix

**Goal:** Extend the contrastive pretraining pipeline
(`autoresearch_contrastive.py`) with a SemiSupCon-style loss that treats
same-tier T2 / T5 pairs as *labeled* positives in the InfoNCE matrix, alongside
the existing piece-based self-supervised positives from T3 MAESTRO.

**Reference.** Dufourq & Smit, *SemiSupCon* (arXiv:2407.13840). Same idea as
SupCon, but labeled positives coexist with unlabeled self-supervised positives
in one contrastive objective.

**North star metric:** `probe_pairwise` on held-out T2 competition pairs, gated
on the Chunk A `probe_dimension_collapse` staying below the SemiSupCon-free
baseline. A higher probe_pairwise with equal or worse collapse means we
"learned to rank by pretending" — we only accept the trade when both metrics
improve or hold.

## Why

- **Tier-based positives work with single-ordinal T5.** SemiSupCon doesn't
  require per-dim labels. "Same T5 bucket" and "same T2 placement" are
  sufficient labeled-positive signals. This is what makes the loss compatible
  with our labeling capacity.
- **Replaces heuristic hard-negative sampling.** Current
  `HardNegativePairSampler` picks negatives by piece dissimilarity. SemiSupCon
  turns labeled non-positive samples into *hard negatives automatically*
  — everything that isn't a same-tier positive is a negative, and the
  temperature scaling ensures the hardest ones dominate the gradient.
- **Initialization constraint (critical).** Per arXiv:2506.23869, contrastive
  fine-tuning only works when the encoder is initialized from pretrained
  weights. The existing Aria pipeline already satisfies this via
  `aria_embeddings.py`. MuQ is initialized from its pretrained checkpoint too.
  **Do not** try SemiSupCon from random init — it will fail, and the failure
  will be silent (the probe just won't improve).

## Scope

**In scope.** Extend `losses.py` with `semi_sup_con_loss()`. Extend
`autoresearch_contrastive.py` to sample labeled pairs from T2/T5 alongside
unlabeled piece pairs from T3. No architecture changes.

**Out of scope.** Any loss-weighting changes for the downstream multi-task
model — SemiSupCon is a pretraining phase only. Phase C consumes the resulting
encoder weights as frozen features for the linear probe.

## File Structure

### Modified Files

| File | Change |
|---|---|
| `model/src/model_improvement/losses.py` | Add `semi_sup_con_loss(projections, tier_ids, temperature)` that computes InfoNCE where positives include (a) same-piece pairs from T3 and (b) same-tier pairs from T2/T5. Composes with `piece_based_infonce_loss` — does not replace it. |
| `model/src/model_improvement/autoresearch_contrastive.py` | New `--lambda-semisupcon` flag (default 0.5). The training loop now produces batches with mixed tier membership so the loss can build its positive mask. |
| `model/src/model_improvement/data.py` | Add `SemiSupConBatchSampler`: ensures every batch contains ≥2 samples from each tier with labels, so the positive mask is non-degenerate. |
| `docs/specs/2026-03-19-contrastive-pretraining-design.md` | Append a new section: "SemiSupCon positive-mask construction." Update the loss composition table. Do not duplicate the existing design doc. |

## Phases

### P0 — Spec update + positive-mask construction (0.5 day)

- [ ] Update the contrastive-pretraining design spec with the positive-mask
  rules:
  - same piece → positive (piece-based InfoNCE)
  - same T2 placement (exact) → positive (SemiSupCon labeled)
  - same T5 ordinal bucket → positive (SemiSupCon labeled)
  - different piece + different tier label → negative
  - different piece + same tier label → positive (labeled signal wins)
- [ ] Unit test: on a synthetic batch of 8 samples with known tiers, the mask
  matches the expected layout.

### P1 — Loss implementation (1 day)

- [ ] Write `semi_sup_con_loss()`. Standard SemiSupCon: numerator is sum over
  labeled positives, denominator is sum over all non-self, temperature-scaled.
- [ ] Make sure the loss is well-defined when no labeled positives exist in a
  batch (graceful: fall back to piece-based InfoNCE for that batch).
- [ ] Unit test: two identical samples → loss → 0. Two anti-correlated batches
  → loss > InfoNCE baseline.

### P2 — Batch sampler (1 day)

- [ ] Write `SemiSupConBatchSampler`. Guarantees ≥2 T2 samples, ≥2 T5 samples,
  ≥2 T3 piece-pair samples per batch. Sizes: bottleneck is T2 since it has
  fewer recordings — sampler falls back to T3 if T2 is exhausted.
- [ ] Verify on MPS: batch assembly time <50ms so we don't starve the GPU.

### P3 — Autoresearch sweep (1.5 days)

- [ ] Grid: `lambda_semisupcon ∈ [0.0, 0.3, 0.5, 1.0]`, temperature ∈
  `[0.05, 0.1, 0.2]`. 12 configs × 2 encoders (MuQ, Aria).
- [ ] Track `probe_pairwise`, `probe_r2`, `probe_dimension_collapse` per run.
- [ ] Winner selection: highest `probe_pairwise` subject to
  `probe_dimension_collapse ≤ baseline + 0.05`.

### P4 — Freeze and hand off to Phase C (0.5 day)

- [ ] Export winning encoder checkpoint to
  `data/checkpoints/semisupcon_{encoder}_v1.ckpt`.
- [ ] Update `docs/model/04-north-star.md` with the new probe numbers (pre-
  and post-SemiSupCon).

## Exit Criteria

- `probe_pairwise` improves ≥1.5pp over the piece-only InfoNCE baseline on T2
  held-out pairs.
- `probe_dimension_collapse` does not regress by more than 0.05.
- Unit tests cover positive-mask edge cases (no labeled positives in batch,
  all samples same tier).

## Risks

- **Label leakage from T2 placements.** Same-placement from *different
  competitions* may not be same-quality. Mitigation: only treat same-placement
  as positive if both samples are from the same competition.
- **SemiSupCon drowning out piece-based self-sup.** If labeled positives
  dominate, the model learns a tier classifier instead of a quality ranker.
  Temperature + λ balance is the lever; sweep captures this.
- **Batch sampler stalls on small tiers.** T2 may not have enough recordings
  per batch. Fallback to T3-only is designed in but adds variance.
