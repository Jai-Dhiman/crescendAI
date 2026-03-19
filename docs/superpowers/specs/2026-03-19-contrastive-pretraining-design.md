# Phase B: Quality-Aware Contrastive Pretraining

**Date:** 2026-03-19
**Status:** Approved
**Scope:** New autoresearch script for contrastive pretraining of MuQ and Aria encoders

## Context

Phase A (frozen linear probe) validated that Aria and MuQ produce decorrelated errors (phi=0.043), confirming fusion viability. Phase B adapts both encoders to produce quality-aware embeddings before independent fine-tuning (Phase C) and gated fusion (Phase D).

The existing `piece_based_infonce_loss` clusters same-piece performances together but has no notion of quality ordering within a piece. Phase B adds this quality dimension using ordinal signals from competition placements (T2), skill buckets (T5), and continuous scores (T1).

## Design Decisions

1. **Hybrid loss** (decided over margin-weighted InfoNCE and pure triplet): Two complementary terms -- inter-piece clustering (InfoNCE) + intra-piece quality ordering (margin loss). These are geometrically orthogonal and have complementary failure modes.

2. **Pluggable encoder with `--encoder` flag** (decided over separate scripts or single-script branching): One script dispatches to encoder-specific classes conforming to a shared interface. Avoids duplication while keeping encoder logic isolated.

3. **Margin-based ordinal loss** (decided over soft ordinal InfoNCE): Simpler, more interpretable, directly compatible with the margin values from `CompetitionPairSampler`. Centroid-based formulation pushes better performances closer to the piece centroid.

4. **Contrastive val loss + linear probe diagnostic** (decided over loss-only or probe-only): Val loss for fast autoresearch optimization, linear probe as secondary sanity check for downstream quality prediction.

## Architecture

### File: `model/src/model_improvement/autoresearch_contrastive.py`

Single-fold (fold-0) contrastive pretraining script following the autoresearch pattern from `autoresearch_loss_weights.py`.

### Encoder Abstraction

Both encoders work on pre-extracted embeddings (not raw audio/MIDI) and produce fixed-dim projected embeddings.

**MuQ encoder:**
- Input: frame-level embeddings `[B, T, 1024]` + mask
- Layers: attention pooling -> 2-layer MLP (1024 -> 512) -> projection head (512 -> 256)
- All layers trainable (no LoRA backbone -- pre-extracted embeddings)

**Aria encoder:**
- Input: fixed-dim embeddings `[B, 512]`
- Layers: 2-layer MLP (512 -> 512) -> projection head (512 -> 256)
- All layers trainable (frozen Aria backbone -- embeddings already extracted)

Both produce 256-dim projected embeddings for contrastive learning.

### Loss Functions

**Term 1 -- Piece-clustering InfoNCE:**
Reuses existing `piece_based_infonce_loss(embeddings, piece_ids, temperature)` from `losses.py`. Clusters same-piece performances together, pushes different pieces apart.

**Term 2 -- Ordinal margin loss (new, added to `losses.py`):**

```python
def ordinal_margin_loss(
    embeddings: Tensor,     # [B, D] normalized projected embeddings
    piece_ids: Tensor,      # [B] piece membership
    quality_scores: Tensor, # [B] normalized quality in [0,1], higher = better
    margin_scale: float = 0.1,
) -> Tensor:
```

For each piece with 2+ segments in the batch:
1. Compute piece centroid (mean of all same-piece embeddings)
2. For each ordered pair (better, worse): `loss = max(0, margin - (sim(better, centroid) - sim(worse, centroid)))` where `margin = margin_scale * (quality_better - quality_worse)`
3. Average across all valid pairs

**Combined loss:** `L = lambda_infonce * L_infonce + lambda_ordinal * L_ordinal`

### Data Pipeline

| Tier | Segments | InfoNCE | Ordinal | Quality derivation |
|------|----------|---------|---------|-------------------|
| T1 PercePiano | 1,202 | piece_mapping.json | 6-dim composite labels | mean(labels) -> [0,1] |
| T2 Competition | 9,059 | same piece/edition | placement ordinals | 1 - (placement / max_placement) |
| T5 YouTube Skill | ~3,100 (TBD) | same piece | skill buckets 1-5 | bucket / 5 |

**Weighted sampling:** `WeightedTierSampler` draws from tiers per configurable weights (default: T1=0.2, T2=0.6, T5=0.2). If T5 data doesn't exist, weight redistributes to T2.

**Train/val split:** T1 uses fold-0 from `folds.json`. T2 holds out one competition edition. T5 holds out one piece.

**Collate:** `multi_task_collate_fn` for MuQ (variable-length padding), simple stack for Aria (fixed-dim). Each batch item carries: embedding, piece_id, quality_score.

### Lightning Module: `ContrastivePretrainModel`

- `training_step`: encode -> project -> compute both loss terms -> log
- `validation_step`: same on held-out data
- `configure_optimizers`: AdamW, LR=1e-4, weight_decay=1e-4, warmup + cosine decay

### Training Config (Autoresearch Defaults)

- Fold 0 only
- 30 epochs, patience 7
- Batch size 16, gradient accumulation 2 (effective 32)
- Gradient clipping 1.0
- Temperature 0.07 (configurable)

### Evaluation

1. **Primary -- contrastive val loss**: Combined `L_infonce + L_ordinal` on held-out data. Used for autoresearch optimization and early stopping.
2. **Secondary -- linear probe**: Freeze pretrained encoder, extract embeddings for T1 PercePiano, train `Linear(dim, 6)` on fold-0 train, evaluate pairwise accuracy + R2 on fold-0 val. Logged but not optimized.

### CLI Interface

```bash
uv run python -m model_improvement.autoresearch_contrastive \
  --encoder muq|aria \
  --lambda-infonce 1.0 \
  --lambda-ordinal 0.5 \
  --temperature 0.07 \
  --ordinal-margin 0.1 \
  --t1-weight 0.2 --t2-weight 0.6 --t5-weight 0.2 \
  --learning-rate 1e-4 \
  --weight-decay 1e-4 \
  --max-epochs 30
```

### Structured Output

```
AUTORESEARCH_RESULT
contrastive_loss=X.XXXXXX
ordinal_loss=X.XXXXXX
probe_pairwise=X.XXXXXX
probe_r2=X.XXXXXX
elapsed=Ys
```
Plus `AUTORESEARCH_JSON={...}` for machine parsing.

### Checkpoints

Best model by val contrastive loss saved to `data/checkpoints/contrastive_pretrain/{encoder}/fold_0/`. This checkpoint initializes Phase C (LoRA fine-tuning).

## Files Modified

- `model/src/model_improvement/losses.py` -- add `ordinal_margin_loss`
- **New:** `model/src/model_improvement/autoresearch_contrastive.py` -- main script

## Informed By

- Aria's contrastive training code (`aria.training.contrastive_finetune`): symmetric NT-Xent, AdamW with linear decay, EOS-position embedding extraction
- Existing autoresearch pattern (`autoresearch_loss_weights.py`): structured output, single-fold, best checkpoint loading, memory cleanup
- Existing loss/data building blocks: `piece_based_infonce_loss`, `CompetitionPairSampler`, `AudioSegmentDataset`, `multi_task_collate_fn`

## Not In Scope

- LoRA adaptation of MuQ/Aria backbones (Phase C)
- Gated fusion (Phase D)
- T5 YouTube Skill data curation (separate workstream)
- Multi-fold cross-validation (autoresearch uses fold-0 only)
