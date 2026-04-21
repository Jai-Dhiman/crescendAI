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

        See `docs/model/08-uncertainty-and-diagnostics.md` § Baseline snapshot (pre-Wave 1).


## Exit Criteria

- `probe_pairwise` improves ≥1.5pp over the piece-only InfoNCE baseline on T2
  held-out pairs.
- `probe_dimension_collapse` does not regress by more than 0.05.
- Unit tests cover positive-mask edge cases (no labeled positives in batch,
  all samples same tier).

## Execution Timing

**Infrastructure:** merged (`semi_sup_con_loss`, `SemiSupConBatchSampler`,
`--lambda-semi-sup` flag on `autoresearch_contrastive.py`).

**Pretraining runs are gated on T5 labeling completion.** SemiSupCon's labeled
positives come from same-tier T2/T5 pairs; without a substantial T5 label pool,
the positive mask is too sparse to shape the InfoNCE matrix meaningfully, and
the λ sweep just measures variance. Kick off pretraining once T5 labeling is
complete and `kappa_report` rolling κ ≥ 0.6.

Probe runs against the existing T3-only piece-based checkpoint can still be
used for smoke-testing the loss path, but the headline `probe_pairwise` number
in Exit Criteria is a training-time measurement.

## Risks

- **Label leakage from T2 placements.** Same-placement from *different
  competitions* may not be same-quality. Mitigation: only treat same-placement
  as positive if both samples are from the same competition.
- **SemiSupCon drowning out piece-based self-sup.** If labeled positives
  dominate, the model learns a tier classifier instead of a quality ranker.
  Temperature + λ balance is the lever; sweep captures this.
- **Batch sampler stalls on small tiers.** T2 may not have enough recordings
  per batch. Fallback to T3-only is designed in but adds variance.
