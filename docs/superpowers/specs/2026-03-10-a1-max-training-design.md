# A1-Max: Best Audio Model for Web MVP

**Date:** 2026-03-10
**Status:** Approved

## Goal

Stack all Tier 1 improvements from the experiment roadmap into one A1 training run, deploy as a 4-fold ensemble on the cloud HF endpoint, and ship 6-dim scores to the web app.

**Target:** ~78-82% pairwise accuracy (up from A1 baseline 73.9%).

## Model Configuration

**Base:** A1 (MuQ + LoRA, attention pooling, shared 512-dim encoder, per-dimension ranking + regression heads)

**Stacked improvements:**

1. **Hard negative mining** -- After 5 warmup epochs, oversample pairs where model is wrong or uncertain (ranking logit near 0). Curriculum: start with easy pairs (|label_diff| > 0.3), progressively add harder pairs by epoch 15.
2. **ListMLE ranking loss** -- Replace pairwise BCE with ListMLE over all performances of the same piece. Falls back to pairwise BCE for pieces with only 2 performances (ListMLE needs 3+).
3. **CCC regression loss** -- Replace MSE with Concordance Correlation Coefficient. Penalizes scale and shift errors.
4. **Loss weights** -- Ranking-dominant: `L = 1.5 * L_listmle + 0.3 * L_contrastive + 0.3 * L_ccc + 0.1 * L_invariance`
5. **Label smoothing** -- Sweep {0.0, 0.05, 0.1}
6. **Mixup on embeddings** -- Beta(0.2, 0.2) interpolation on MuQ embeddings + labels
7. **LoRA sweep** -- Rank {8, 16, 32}, layers {(9-12), (7-12)}

**Hyperparameter grid:** 3 label_smoothing x 3 LoRA_rank x 2 layer_ranges = 18 configs, each with 4-fold CV = 72 training runs.

## Training Infrastructure

**Compute:** Local Mac, MPS backend.

**Memory management:**
- Pre-extracted MuQ embeddings on CPU, move to MPS per-batch only
- LoRA adapters + downstream heads only on MPS (~2-5M params)
- Batch size 32, fall back to 16 if needed
- `torch.mps.empty_cache()` + `gc.collect()` between every fold
- Cap per-piece list size in ListMLE to prevent memory spikes
- Sequential execution: one config/fold at a time

**Data:** 1,202 PercePiano segments, 4-fold piece-stratified CV from `percepiano_cache/folds.json`. Pre-extracted MuQ embeddings (1024-dim frame-level).

## Evaluation

**Per config (4-fold averaged):**
- Pairwise accuracy (primary) -- must beat 73.9%
- R2 (secondary)
- Robustness score_drop_pct (veto at >15%)
- Per-dimension pairwise breakdown (dynamics, timing, pedaling, articulation, phrasing, interpretation)
- STOP AUC >= 0.80

**Ensemble:** After selecting best config, evaluate the 4-fold ensemble (average predictions). This is what ships.

## Deployment

1. Select best config by mean pairwise accuracy across 4 folds
2. Save all 4 fold checkpoints
3. Update HF inference endpoint (`apps/inference/handler.py`):
   - Load 4-fold A1-max ensemble
   - Output 6 dimensions (not 19)
   - Return per-dimension scores
4. Update web app observation pipeline to consume 6-dim scores
5. Keep current 19-dim model as tagged rollback version

## Files

**Training (new/modified):**
- `model/src/model_improvement/losses.py` -- ListMLE, CCC loss
- `model/src/model_improvement/data.py` -- HardNegativePairSampler, mixup
- `model/src/model_improvement/audio_encoders.py` -- training_step updates
- `model/src/model_improvement/a1_max_sweep.py` -- sweep runner (new)

**Deployment:**
- `apps/inference/handler.py` -- 6-dim ensemble inference
- Web app observation pipeline -- 6-dim consumption
