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

3. **Cross-piece anchor ordinal loss** (decided over centroid-based): For each ordered pair (better, worse) within a piece, sample a random anchor from a different piece. Enforce `sim(anchor, better) > sim(anchor, worse) + margin`. Avoids centroid leakage (centroid including the sample being compared) and works correctly with 2-segment pieces.

4. **Contrastive val loss + linear probe diagnostic** (decided over loss-only or probe-only): Val loss for fast autoresearch optimization, linear probe as secondary sanity check for downstream quality prediction.

## Architecture

### Files

- `model/src/model_improvement/losses.py` -- add `ordinal_margin_loss`
- **New:** `model/src/model_improvement/autoresearch_contrastive.py` -- main script (encoder classes, dataset, Lightning module, CLI)

### Encoder Interface

Both encoders conform to a shared protocol:

```python
class ContrastiveEncoder(Protocol):
    def encode(self, embeddings: Tensor, mask: Tensor | None = None) -> Tensor:
        """Raw embedding -> pooled hidden [B, hidden_dim]."""
        ...
    def project(self, z: Tensor) -> Tensor:
        """Hidden -> projected embedding [B, proj_dim=256]."""
        ...
```

**MuQ encoder:**
- Input: frame-level embeddings `[B, T, 1024]` + mask
- Layers: attention pooling -> 2-layer MLP (1024 -> 512) -> projection head (512 -> 256)
- Architecture reuses attention pooling + encoder MLP from `MuQLoRAModel` (no comparator/ranking/regression heads)
- All layers trainable (no LoRA backbone -- pre-extracted embeddings)

**Aria encoder:**
- Input: fixed-dim embeddings `[B, 512]`
- Layers: 2-layer MLP (512 -> 512) -> projection head (512 -> 256)
- All layers trainable (frozen Aria backbone -- embeddings already extracted)

Both produce 256-dim L2-normalized projected embeddings for contrastive learning.

### Loss Functions

**Term 1 -- Piece-clustering InfoNCE:**
Reuses existing `piece_based_infonce_loss(embeddings, piece_ids, temperature)` from `losses.py`. Clusters same-piece performances together, pushes different pieces apart.

Note: with gradient accumulation, InfoNCE negatives come from the micro-batch (batch_size=16), not the accumulated batch (32). This is acceptable for this phase -- the effective negative count is 14-15 per anchor. If this proves insufficient, a memory bank can be added later.

**Term 2 -- Ordinal margin loss (new, added to `losses.py`):**

```python
def ordinal_margin_loss(
    embeddings: Tensor,     # [B, D] L2-normalized projected embeddings
    piece_ids: Tensor,      # [B] piece membership
    quality_scores: Tensor, # [B] normalized quality in [0,1], higher = better
    margin_scale: float = 0.1,
    eps: float = 1e-8,
) -> Tensor:
```

For each piece with 2+ segments in the batch:
1. Enumerate all ordered pairs `(i, j)` where `quality_scores[i] > quality_scores[j]` within the same piece
2. For each pair, sample a random anchor `k` from a different piece in the batch
3. Compute: `loss = max(0, margin - (sim(k, i) - sim(k, j)))` where `margin = margin_scale * (quality_i - quality_j)` and `sim` is cosine similarity
4. Average across all valid pairs

If no valid pairs exist (all segments from unique pieces, or all same quality), returns 0.

**Combined loss:** `L = lambda_infonce * L_infonce + lambda_ordinal * L_ordinal`

### Data Pipeline

#### Unified Dataset: `ContrastiveSegmentDataset`

A new dataset class in `autoresearch_contrastive.py` that yields individual segments with a unified schema:

```python
# Each item:
{
    "embedding": Tensor,       # [T, D] for MuQ or [D] for Aria
    "piece_id": int,           # unified across tiers (see namespace below)
    "quality_score": float,    # normalized to [0, 1], higher = better
    "tier": str,               # "t1", "t2", or "t5" (for logging)
}
```

**Piece ID namespace:** Disjoint integer ranges per tier to avoid false cross-tier positives. T1 pieces: 0..N1-1, T2 pieces: N1..N1+N2-1, T5 pieces: N1+N2..N1+N2+N5-1. Cross-tier same-piece matching is not attempted (too error-prone with different naming conventions). The InfoNCE loss treats each tier's pieces independently.

**Quality score normalization per tier:**

| Tier | Source | Normalization |
|------|--------|---------------|
| T1 PercePiano | `composite_labels.json` (6-dim, each in [0,1]) | `mean(labels_across_6_dims)` -- already in [0,1] |
| T2 Competition | `placement` (int, lower = better) | `1.0 - (placement - 1) / (max_placement_in_group - 1)` per (competition, edition, piece) group. Ties get same score. |
| T5 YouTube Skill | `skill_bucket` (int 1-5) | `(bucket - 1) / 4.0` |

**Embedding paths:**

| Tier | MuQ embeddings | Aria embeddings |
|------|----------------|-----------------|
| T1 | `Embeddings.percepiano / "muq_embeddings.pt"` | `Embeddings.percepiano / "aria_embedding.pt"` |
| T2 | `Embeddings.competition / "muq_embeddings.pt"` | `Embeddings.competition / "aria_embedding.pt"` (to be extracted) |
| T5 | TBD (deferred) | TBD (deferred) |

**T5 is deferred for the first implementation.** The script accepts `--t5-weight` but defaults to 0.0. When T5 weight is 0, its share redistributes to T2 proportionally.

#### Weighted Tier Sampling

`WeightedTierSampler(Sampler)` draws indices from a concatenated dataset, biased by tier weights. Ensures each batch has a mix of tiers. To guarantee InfoNCE gets positive pairs, the sampler over-represents pieces with 2+ segments -- implemented by sampling piece groups (not individual segments) and then drawing segments within each group. This ensures most batches contain at least a few same-piece pairs.

#### Collation

**MuQ:** Uses a custom collate that pads variable-length `[T, 1024]` embeddings to batch max length (similar to `multi_task_collate_fn`), creates masks, and stacks `piece_id` + `quality_score`.

**Aria:** Simple stack -- all embeddings are `[512]`, no padding needed.

Both produce batches with keys: `embeddings`, `mask` (MuQ only), `piece_ids`, `quality_scores`.

#### Train/Val Split

- **T1:** fold-0 from `folds.json` (train/val keys)
- **T2:** Hold out Cliburn 2022 as validation (largest single competition, provides diverse piece coverage). Remaining competitions for training.
- **T5:** Deferred (first implementation trains without T5)

### Lightning Module: `ContrastivePretrainModel`

- `__init__(encoder, lambda_infonce, lambda_ordinal, temperature, margin_scale, lr, wd, warmup_epochs, max_epochs)`
- `training_step`: `z = encoder.encode(batch) -> proj = encoder.project(z) -> L_infonce + L_ordinal`
- `validation_step`: same, logs `val_contrastive_loss`, `val_infonce_loss`, `val_ordinal_loss`
- `configure_optimizers`: AdamW, warmup (5 epochs, `LinearLR(start_factor=0.01)`) + cosine decay to `1e-6`

### Training Config (Autoresearch Defaults)

- Fold 0 only
- 30 epochs, patience 7
- Batch size 16, gradient accumulation 2 (effective 32)
- Gradient clipping 1.0
- Temperature 0.07 (configurable)
- LR 1e-4, weight decay 1e-4
- Warmup: 5 epochs with `LinearLR(start_factor=0.01)`

### Evaluation

1. **Primary -- contrastive val loss**: Combined `lambda_infonce * L_infonce + lambda_ordinal * L_ordinal` on held-out data. Used for autoresearch optimization and early stopping.
2. **Secondary -- linear probe**: After training completes, freeze pretrained encoder, extract `encode()` (not projected) embeddings for all T1 PercePiano segments. Train `Linear(hidden_dim, 6)` on fold-0 train using `train_linear_probe` from `aria_linear_probe.py`, evaluate pairwise accuracy + R2 on fold-0 val. Logged but not optimized.

Validation uses the same `WeightedTierSampler` with the same tier weights applied to the held-out splits.

### CLI Interface

```bash
uv run python -m model_improvement.autoresearch_contrastive \
  --encoder muq|aria \
  --lambda-infonce 1.0 \
  --lambda-ordinal 0.5 \
  --temperature 0.07 \
  --ordinal-margin 0.1 \
  --t1-weight 0.2 --t2-weight 0.8 \
  --learning-rate 1e-4 \
  --weight-decay 1e-4 \
  --max-epochs 30
```

(T5 weight defaults to 0.0; when added later, T2 weight decreases accordingly.)

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

## Informed By

- Aria's contrastive training code (`aria.training.contrastive_finetune`): symmetric NT-Xent, AdamW with lr=1e-5 / weight_decay=0.1 / linear decay, EOS-position embedding extraction, trailing loss logging
- Existing autoresearch pattern (`autoresearch_loss_weights.py`): structured output, single-fold, best checkpoint loading, memory cleanup
- Existing loss/data building blocks: `piece_based_infonce_loss`, `CompetitionPairSampler`, `AudioSegmentDataset`, `multi_task_collate_fn`
- Spec review feedback: centroid leakage fix, dataset class specification, piece_id namespace strategy, warmup schedule, gradient accumulation interaction with InfoNCE

## Not In Scope

- LoRA adaptation of MuQ/Aria backbones (Phase C)
- Gated fusion (Phase D)
- T5 YouTube Skill data curation (separate workstream; script supports it when ready)
- Multi-fold cross-validation (autoresearch uses fold-0 only)
- T2 Aria embedding extraction (prerequisite; must run `aria_embeddings.py` on competition MIDIs first)
