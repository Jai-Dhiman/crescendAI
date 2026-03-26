# Training Readiness Design

**Date:** 2026-03-26
**Status:** Approved
**Scope:** Evaluation strategy, compute/storage migration, experiment tracking, autoresearch sweeps, Aria research, doc updates

---

## Context

T5 YouTube Skill Corpus labeling is in progress (2/16 pieces curated). This spec prepares the entire training pipeline so that when labeling completes, training can begin immediately. Key decisions: retire 4-fold CV in favor of single train/val/test split, migrate from Thunder Compute to HF Jobs, add HF Buckets for training data, integrate Trackio for experiment tracking, and plan comprehensive autoresearch sweeps.

## 1. Evaluation Strategy

### Single Split Replaces 4-Fold CV

The 4-fold piece-stratified CV on PercePiano (T1) is retired as the primary evaluation gate. With T5 providing ~3,100 segments across 16 pieces and 5 skill buckets, a proper three-way split is both more statistically powerful and directly measures the product goal: can the model distinguish skill levels?

### Three-Way Split

| Split | T1 (1,202) | T2 (9,059) | T3 (24,321) | T5 (~3,100) |
|-------|-----------|-----------|-------------|-------------|
| **Train** | 80% (~960) | 85% (~7,700) | 100% | 80% (~2,480) |
| **Val** | -- | -- | -- | 10% (~310) |
| **Test** | 20% (~240) | 15% (~1,360) | -- | 10% (~310) |

**Rationale:**
- **Val is T5-only** because it's the autoresearch optimization target. Keeping it narrow prevents metric dilution.
- **Test is multi-domain** for final reporting: T5 test (skill discrimination), T1 test (quality regression sanity), T2 test (competition ranking sanity).
- **T3 has no holdout** because MAESTRO provides contrastive pairs (same piece, different performer), not quality labels. No meaningful accuracy metric on held-out contrastive data.

### Stratification

- T5: stratify by **piece + bucket** (proportional representation in each split)
- T1: stratify by **piece** (2-way split, not 4-fold)
- T2: stratify by **competition + round** (hold out entire rounds to prevent same-performer leakage)

### Metrics

| Metric | Split | Purpose |
|--------|-------|---------|
| **T5 skill discrimination** (pairwise accuracy, 5 buckets) | Val | Primary autoresearch metric |
| **T5 skill discrimination** | Test | Final reporting (never seen during optimization) |
| **T1 pairwise accuracy** | Test | Sanity: quality regression among advanced players |
| **T2 Spearman rho** | Test | Sanity: competition ranking correlation |
| **Aria-only discrimination** | Val | Confound check: confirms signal is musical, not audio quality |
| **Per-dimension breakdown** | Val | Diagnostic: which dims discriminate best |

### Label Quality Validation

T5 is labeled by a single annotator. Known risks: middle buckets (2-4) are subjective, audio quality correlates with skill bucket.

**Required post-labeling step:** Test-retest reliability. Re-label 30-40 randomly selected recordings blind after a 1-week gap. Measure self-agreement. Target: >80% exact match or >90% within-one-bucket.

**Confound defense:** Aria encoder operates on MIDI (via AMT) with zero audio quality information. If Aria discriminates skill buckets, the signal is musical. If MuQ discriminates but Aria doesn't, MuQ may be exploiting audio quality as a shortcut. This check runs on every experiment automatically.

## 2. Compute Strategy

### Local-First, Cloud for Validation

| Environment | Use Case | Cost |
|-------------|----------|------|
| **Local M4 (32GB)** | All autoresearch sweeps, debugging, single-run experiments, T5 embedding extraction | Free |
| **HF Jobs L4 (24GB, $0.80/hr)** | Budget full runs, Aria LoRA if it fits in 24GB | ~$2-4/run |
| **HF Jobs A100 (80GB, $2.50/hr)** | Aria fine-tuning if >24GB needed, fusion experiments, final test-set validation | ~$5-10/run |

### Thunder Compute Retirement

Thunder Compute A100 ($0.78/hr prototyping, $1.79/hr production) is cheaper per-hour than HF Jobs ($2.50/hr for A100). However, HF Jobs provides native bucket mounting (no download step), integrated billing, and a single ecosystem with HF Buckets + Hub. For sporadic validation runs (not 24/7 training), the convenience outweighs the per-hour premium.

**Migration:** Update all docs and notebook setup cells to reference HF Jobs. Remove Thunder Compute SSH configs and setup scripts.

### HF Jobs Pricing Reference

| Hardware | VRAM | $/hour |
|----------|------|--------|
| T4 | 16 GB | $0.40 |
| L4 | 24 GB | $0.80 |
| L40S | 48 GB | $1.80 |
| A10G small | 24 GB | $1.00 |
| A10G large | 24 GB | $1.50 |
| A100 80GB | 80 GB | $2.50 |
| H200 141GB | 141 GB | $5.00 |

Default: L4 ($0.80/hr) for runs that fit in 24GB. A100 only when VRAM requires it.

## 3. Storage Strategy

### HF Buckets for Training Data

| Location | Contents | Purpose | Est. Cost |
|----------|----------|---------|-----------|
| **HF Bucket** (private) | Embeddings (~52GB), manifests (~29GB), checkpoints (~10GB), T5 audio | Training data lake, mounted as volumes in HF Jobs | ~$1.10/mo |
| **GDrive** (via rclone) | Results JSON, final model weights, labels, experiment logs | Archival backup, non-training assets | Free (existing) |
| **Local M4** | Active working set only (current experiment's embeddings + T5 data, ~8-10GB) | Fast iteration | Free |

### HF Bucket Integration

HF Buckets are NOT S3-compatible (no `s3://` endpoint). They use `hf://buckets/` paths and integrate via:
- `hf` CLI: `hf buckets sync`, `hf buckets cp`
- Python: `huggingface_hub` library
- HF Jobs: mount as volumes with `-v hf://buckets/crescendai/training-data:/data`

Training scripts read from mounted paths as if local. No code changes needed for the training loop itself -- only the data loading paths change.

Chunk-level dedup means successive checkpoint uploads only transfer changed bytes.

### GDrive Cleanup Plan

1. Audit current GDrive contents: `rclone ls gdrive:crescendai_data/`
2. Delete stale checkpoints (pre-clean-fold experiments, legacy S2 GNN weights)
3. Delete regenerable embeddings that will move to HF Bucket
4. Keep: final model weights, results JSON, labels, validated experiments
5. GDrive becomes archival backup only (results, labels, final weights)

## 4. Experiment Tracking with Trackio

### Integration

Trackio replaces the current ad-hoc `results/*.json` + GDrive rclone sync pattern for experiment tracking. Existing `sync.py` rclone code stays for GDrive backup but is no longer the primary tracking mechanism.

### What Gets Tracked Per Experiment

```
Metadata:
  - experiment_id, timestamp, git commit hash
  - phase (AR-1 through AR-6)
  - config: LR, schedule, LoRA rank, loss weights, data mix

Per-Epoch:
  - train_loss (total + per-component)
  - val_skill_discrimination (T5 val pairwise accuracy)
  - learning_rate (for schedule visualization)

Final:
  - T5_val pairwise accuracy (primary metric)
  - Aria-only confound check (above chance = pass)
  - Per-dimension breakdown
  - Training time

Autoresearch:
  - kept/reverted decision
  - delta vs baseline
  - config diff from previous experiment
```

### Dashboard

Trackio syncs to an HF Space for persistent visibility. Tracks:
- Metric progression across autoresearch iterations
- Best configuration per phase
- Overall improvement trajectory from baseline to final model

## 5. Autoresearch Sweep Plan

### Eval Metric

T5 val skill discrimination (pairwise accuracy, 5 buckets). Single training run per experiment (no k-fold).

### Phase Ordering

Each phase uses the winner from the previous phase as baseline. All sweeps run on local M4. Winners validated on HF Jobs for final test-set numbers.

| Phase | Swept | Search Space | Est. Experiments |
|-------|-------|-------------|-----------------|
| **AR-1: LR + schedule** | Peak LR, warmup, decay | LR: {1e-5, 3e-5, 1e-4, 3e-4}, warmup: {100, 500, 1000}, decay: {cosine, linear} | 12-16 |
| **AR-2: LoRA config** | Rank, target layers, alpha | Rank: {8, 16, 32, 64}, layers: {last-4, last-6, all}, alpha: {rank, 2*rank} | 12-16 |
| **AR-3: Loss weights** | All 5 loss component weights | Continuous perturbation around current optimum (contrastive, regression, ranking, ListMLE, invariance) | 10-15 |
| **AR-4: Contrastive recipe** | Temperature, margin, mining | Temp: {0.05, 0.1, 0.2}, margin: {0.1, 0.3, 0.5}, mining: {random, hard, semi-hard} | 9-12 |
| **AR-5: Fusion gates** | Gate architecture, init, regularization | Linear vs MLP, init: {0.5, audio-biased}, dropout: {0, 0.1, 0.2} | 8-10 |
| **AR-6: Data mix** | T1/T2/T3/T5 weighting | T1: {10%, 20%, 30%}, T5: {30%, 50%, 70%}, T2 fills remainder | 9-12 |

**Total: ~60-80 experiments.** At ~20 min/run on M4, ~20-27 hours of local compute spread across phases.

### Dependencies

- AR-1 through AR-3: can start as soon as T5 labeling completes and embeddings are extracted
- AR-4: requires contrastive pretraining infrastructure (Phase B code)
- AR-5: requires both MuQ and Aria independently trained (Phase C complete)
- AR-6: requires T5 complete (all 16 pieces curated)

### Autoresearch Workflow

```
for each phase:
  baseline = winner from previous phase (or current best)
  for each config in search space:
    train on {T1_train, T2_train, T3, T5_train}
    evaluate on T5_val
    run Aria confound check
    log to Trackio
    if metric > baseline: keep, update baseline
    else: revert

  validate phase winner on HF Jobs (full resources)
  log final metrics to Trackio
```

## 6. Aria Repo Research

Separate research task, not blocking the rest. Findings feed into AR-2 (LoRA config) and AR-4 (contrastive recipe).

### Research Questions

1. **Pretraining recipe:** How was the 650M model trained on 820K MIDIs? LR schedules, batch sizes, sequence lengths, tokenization choices, data augmentation.
2. **Fine-tuning patterns:** Any LoRA/adapter examples in the repo or community. How others have adapted Aria for downstream tasks.
3. **Embedding extraction:** Better pooling strategies than EOS-token (512d) or last-token (1536d). Attention-weighted pooling, CLS token, layer averaging.
4. **Training code structure:** Whether their training harness has patterns worth adopting (data loading, checkpointing, distributed training).

### Deliverable

Summary of findings with concrete recommendations for CrescendAI's training config. Incorporated into autoresearch sweep designs before AR-2 and AR-4 begin.

## 7. Doc Updates

| File | Changes |
|------|---------|
| `docs/model/01-data.md` | Replace Thunder Compute with HF Jobs, add HF Bucket storage, update eval strategy to single split |
| `model/CLAUDE.md` | Replace Thunder Compute in Stack section, add Trackio, update training workflow description |
| `docs/model/04-north-star.md` | Update eval tiers (single split replaces 4-fold), T5 as primary metric, retire E1 4-fold gate |
| `docs/model/03-encoders.md` | Add Aria confound check to evaluation criteria |

## 8. Pipeline: Labeling Done to Training Starts

The end-to-end pipeline from "T5 labeling complete" to "first autoresearch experiment runs":

```
1. Test-retest reliability check (30-40 recordings, 1 week after labeling)
2. Generate T5 train/val/test splits (stratified by piece + bucket)
3. Extract MuQ embeddings for T5 recordings
4. Extract Aria embeddings for T5 recordings
5. Upload all embeddings + manifests to HF Bucket
6. Update data loader to read T5 splits
7. Add Trackio logging to training loop
8. Run baseline experiment (current A1-Max config on new data mix)
9. Begin AR-1 (LR + schedule sweep)
```

Steps 2-6 should be automated in a single script so the transition is one command.

## Cost Estimate

| Item | Monthly Cost |
|------|-------------|
| HF Bucket (private, ~92GB) | ~$1.10 |
| HF Jobs (est. 10 validation runs/month, L4) | ~$8-16 |
| HF Jobs (est. 2 A100 runs/month for Aria/fusion) | ~$10-20 |
| GDrive | Free (existing) |
| Local M4 | Free |
| **Total** | **~$19-37/month** |
