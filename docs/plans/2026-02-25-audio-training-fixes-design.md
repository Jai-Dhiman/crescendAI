# Audio Training Fixes Design

Date: 2026-02-25
Status: Approved
Notebook: `model/notebooks/model_improvement/01_audio_training.ipynb`

## Context

Training A1 and A2 on Thunder Compute (A6000/32GB RAM) revealed four issues:

1. No checkpoint resume -- crashed run requires full re-training
2. Rising loss -- best checkpoints always epoch 0-2 for A1 (all 4 folds)
3. A2 stage1 val_loss=0.0000 -- EarlyStopping/ModelCheckpoint blind during self-supervised stage
4. CPU memory hit 90% on 32GB RAM instance

GPU was massively underutilized (44 steps/epoch with pre-extracted 1024-dim embeddings). Training moves to local M4/32GB to eliminate cloud costs.

## Changes

### 1. Local MPS Training Support

**Files**: `model/src/model_improvement/training.py`, notebook

- Auto-detect hardware in `train_model()`: CUDA -> `bf16-mixed`, MPS -> `32-true`, CPU -> `32-true`
- Skip `deterministic=True` on MPS (CUBLAS workspace config is CUDA-only)
- Default `num_workers=2` on MPS/CPU, `num_workers=4` on CUDA
- Guard `CUBLAS_WORKSPACE_CONFIG` env var behind CUDA check in notebook
- Make `DATA_DIR` / `CHECKPOINT_DIR` detect local vs cloud (presence of `/workspace/`)
- Skip rclone sync and git clone cells when running locally

### 2. Checkpoint Resume (Skip Completed Folds)

**Files**: `model/src/model_improvement/training.py`, notebook training loops

- New `find_checkpoint(ckpt_dir, model_name, fold_idx)` helper: checks local dir for existing `.ckpt` file, returns path or None
- `train_model()` gets `resume_if_exists=True` parameter: if checkpoint found, load and return without re-training
- Notebook loops check before each fold, skip if found, load checkpoint for evaluation
- GDrive sync happens once after all folds complete (not per-fold)
- Does NOT resume mid-epoch -- crashed fold restarts from scratch (simple, low cost)

### 3. Fix Rising Loss (Warmup + Lower LR)

**Files**: `model/src/model_improvement/audio_encoders.py`, notebook configs

Root cause: `lr=1e-4` with CosineAnnealingLR on ~700 samples overfits in epoch 0-1. Cosine curve is nearly flat at t=0 so LR stays high.

- Add `warmup_epochs` parameter (default 5) to all three model classes
- `configure_optimizers()` uses `SequentialLR`:
  1. `LinearLR(start_factor=0.01)` for warmup epochs
  2. `CosineAnnealingLR(T_max=max_epochs - warmup_epochs, eta_min=1e-6)` for remainder
- Notebook configs: `learning_rate` from `1e-4` to `3e-5`, add `warmup_epochs: 5`
- EarlyStopping `patience` from 20 to 10

No changes to: loss weights, optimizer type, weight decay, batch size, gradient clipping.

### 4. Fix A2 Stage 1 Validation Logging

**Files**: `model/src/model_improvement/audio_encoders.py` (`MuQStagedModel.validation_step`)

Problem: Stage1 validation only logs invariance MSE (clean vs augmented embeddings). With augmentation of `emb + randn * 0.01`, this is trivially near-zero.

- Compute full self-supervised validation loss: `lambda_contrastive * l_contrastive + lambda_invariance * l_inv`
- Matches training loss computation from `_self_supervised_step`
- EarlyStopping and ModelCheckpoint now track a meaningful metric

No changes to supervised validation branch.

### 5. Memory Optimization

**Files**: notebook `build_dataloaders()`

- `num_workers` from 4 to 2 (less fork-memory duplication, no contention on 4 vCPU / M4)
- `pin_memory = torch.cuda.is_available()` (only useful for CUDA transfers)
- MAESTRO ablation: same settings
- No changes to batch size or embedding loading strategy

## Scope Exclusions

No changes to: model architecture, loss functions, loss weights, batch size, data augmentation, evaluation pipeline. Infrastructure and LR schedule fixes only.

## Pre-run Cleanup

Delete existing A1/A2 checkpoints from GDrive and local before re-running (trained with old LR, best at epoch 0-2).
