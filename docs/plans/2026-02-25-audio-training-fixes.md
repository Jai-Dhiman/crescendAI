# Audio Training Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix training infrastructure so audio encoder experiments run reliably on local M4 with proper LR scheduling, checkpoint resume, and meaningful validation metrics.

**Architecture:** Changes span two source files (`training.py`, `audio_encoders.py`) and the notebook. Source changes are tested via pytest; notebook changes are manual-verify only. All three audio encoder classes (`MuQLoRAModel`, `MuQStagedModel`, `MuQFullUnfreezeModel`) share the same `configure_optimizers` pattern, so the warmup fix applies uniformly.

**Tech Stack:** PyTorch Lightning 2.x, PyTorch LR schedulers (SequentialLR, LinearLR, CosineAnnealingLR)

---

### Task 1: Add warmup LR schedule to MuQLoRAModel

**Files:**
- Modify: `model/src/model_improvement/audio_encoders.py:28-48` (constructor), `model/src/model_improvement/audio_encoders.py:290-295` (configure_optimizers)
- Test: `model/tests/model_improvement/test_audio_encoders.py`

**Step 1: Write the failing test**

Add to `TestMuQLoRAModel` in `test_audio_encoders.py`:

```python
def test_warmup_lr_schedule(self):
    model = MuQLoRAModel(
        input_dim=1024, hidden_dim=512, num_labels=6,
        use_pretrained_muq=False,
        learning_rate=3e-5,
        warmup_epochs=5,
        max_epochs=200,
    )
    optim_config = model.configure_optimizers()
    scheduler = optim_config["lr_scheduler"]["scheduler"]
    # SequentialLR wraps two sub-schedulers
    assert hasattr(scheduler, '_schedulers')
    assert len(scheduler._schedulers) == 2

    # First step should be near start_factor * lr (warmup beginning)
    opt = optim_config["optimizer"]
    initial_lr = opt.param_groups[0]["lr"]
    assert initial_lr < 3e-5  # Should be start_factor * lr = 0.01 * 3e-5
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/jdhiman/Documents/crescendai/model && python -m pytest tests/model_improvement/test_audio_encoders.py::TestMuQLoRAModel::test_warmup_lr_schedule -v`
Expected: FAIL -- `warmup_epochs` is not an accepted parameter

**Step 3: Implement warmup in MuQLoRAModel**

In `audio_encoders.py`, add `warmup_epochs: int = 5` to `__init__` params (line ~44). Store as `self.warmup_epochs = warmup_epochs`.

Replace `configure_optimizers` (lines 290-295):

```python
def configure_optimizers(self) -> dict:
    opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
    warmup = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.01, total_iters=self.warmup_epochs
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=self.max_epochs - self.warmup_epochs, eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[warmup, cosine], milestones=[self.warmup_epochs]
    )
    return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/jdhiman/Documents/crescendai/model && python -m pytest tests/model_improvement/test_audio_encoders.py::TestMuQLoRAModel::test_warmup_lr_schedule -v`
Expected: PASS

**Step 5: Run all existing MuQLoRAModel tests to confirm no regression**

Run: `cd /Users/jdhiman/Documents/crescendai/model && python -m pytest tests/model_improvement/test_audio_encoders.py::TestMuQLoRAModel -v`
Expected: All pass

**Step 6: Commit**

```bash
git add model/src/model_improvement/audio_encoders.py model/tests/model_improvement/test_audio_encoders.py
git commit -m "feat(model): add warmup LR schedule to MuQLoRAModel"
```

---

### Task 2: Add warmup LR schedule to MuQStagedModel

**Files:**
- Modify: `model/src/model_improvement/audio_encoders.py:643-662` (constructor), `model/src/model_improvement/audio_encoders.py:878-883` (configure_optimizers)
- Test: `model/tests/model_improvement/test_audio_encoders.py`

**Step 1: Write the failing test**

Add to `TestMuQStagedModel`:

```python
def test_warmup_lr_schedule(self):
    model = MuQStagedModel(
        input_dim=1024, hidden_dim=512, num_labels=6,
        use_pretrained_muq=False,
        learning_rate=3e-5,
        warmup_epochs=5,
        max_epochs=200,
    )
    optim_config = model.configure_optimizers()
    scheduler = optim_config["lr_scheduler"]["scheduler"]
    assert hasattr(scheduler, '_schedulers')
    assert len(scheduler._schedulers) == 2
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/jdhiman/Documents/crescendai/model && python -m pytest tests/model_improvement/test_audio_encoders.py::TestMuQStagedModel::test_warmup_lr_schedule -v`
Expected: FAIL

**Step 3: Implement warmup in MuQStagedModel**

Same pattern as Task 1. Add `warmup_epochs: int = 5` to `__init__` (line ~662), store `self.warmup_epochs`. Replace `configure_optimizers` (lines 878-883) with identical SequentialLR logic.

**Step 4: Run test to verify it passes**

Run: `cd /Users/jdhiman/Documents/crescendai/model && python -m pytest tests/model_improvement/test_audio_encoders.py::TestMuQStagedModel::test_warmup_lr_schedule -v`
Expected: PASS

**Step 5: Run all existing MuQStagedModel tests**

Run: `cd /Users/jdhiman/Documents/crescendai/model && python -m pytest tests/model_improvement/test_audio_encoders.py::TestMuQStagedModel -v`
Expected: All pass

**Step 6: Commit**

```bash
git add model/src/model_improvement/audio_encoders.py model/tests/model_improvement/test_audio_encoders.py
git commit -m "feat(model): add warmup LR schedule to MuQStagedModel"
```

---

### Task 3: Add warmup LR schedule to MuQFullUnfreezeModel

**Files:**
- Modify: `model/src/model_improvement/audio_encoders.py:316-336` (constructor), `model/src/model_improvement/audio_encoders.py:582-628` (configure_optimizers)
- Test: `model/tests/model_improvement/test_audio_encoders.py`

**Step 1: Write the failing test**

Add to `TestMuQFullUnfreezeModel`:

```python
def test_warmup_lr_schedule(self):
    model = MuQFullUnfreezeModel(
        input_dim=256, hidden_dim=128, num_labels=6,
        use_pretrained_muq=False,
        learning_rate=3e-5,
        warmup_epochs=5,
        max_epochs=200,
        unfreeze_schedule={0: [3]},
        mock_num_layers=4,
    )
    model.unfreeze_for_epoch(0)
    optim_config = model.configure_optimizers()
    scheduler = optim_config["lr_scheduler"]["scheduler"]
    assert hasattr(scheduler, '_schedulers')
    assert len(scheduler._schedulers) == 2
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/jdhiman/Documents/crescendai/model && python -m pytest tests/model_improvement/test_audio_encoders.py::TestMuQFullUnfreezeModel::test_warmup_lr_schedule -v`
Expected: FAIL

**Step 3: Implement warmup in MuQFullUnfreezeModel**

Add `warmup_epochs: int = 5` to `__init__`. This model's `configure_optimizers` has discriminative LR logic (param groups per backbone layer). Wrap the existing CosineAnnealingLR (line 625-627) in SequentialLR:

```python
def configure_optimizers(self) -> dict:
    # ... existing param_groups logic (lines 588-622) stays identical ...
    opt = torch.optim.AdamW(param_groups, weight_decay=self.wd)
    warmup = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.01, total_iters=self.warmup_epochs
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=self.max_epochs - self.warmup_epochs, eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[warmup, cosine], milestones=[self.warmup_epochs]
    )
    return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
```

**Step 4: Run test to verify it passes + existing tests**

Run: `cd /Users/jdhiman/Documents/crescendai/model && python -m pytest tests/model_improvement/test_audio_encoders.py::TestMuQFullUnfreezeModel -v`
Expected: All pass

**Step 5: Commit**

```bash
git add model/src/model_improvement/audio_encoders.py model/tests/model_improvement/test_audio_encoders.py
git commit -m "feat(model): add warmup LR schedule to MuQFullUnfreezeModel"
```

---

### Task 4: Fix A2 stage1 validation logging

**Files:**
- Modify: `model/src/model_improvement/audio_encoders.py:833-838` (MuQStagedModel.validation_step self_supervised branch)
- Test: `model/tests/model_improvement/test_audio_encoders.py`

**Step 1: Write the failing test**

Add to `TestMuQStagedModel`:

```python
def test_stage1_validation_logs_contrastive_loss(self):
    """Stage 1 val_loss should include contrastive loss, not just invariance MSE."""
    model = MuQStagedModel(
        input_dim=1024, hidden_dim=512, num_labels=6,
        use_pretrained_muq=False, stage="self_supervised",
        temperature=0.07, lambda_contrastive=0.3, lambda_invariance=0.5,
    )
    # Create batch where clean == augmented (invariance loss = 0)
    # but contrastive loss > 0 (different piece_ids exist)
    emb = torch.randn(4, 50, 1024)
    batch = {
        "embeddings_clean": emb,
        "embeddings_augmented": emb.clone(),  # Identical -> MSE = 0
        "mask": torch.ones(4, 50, dtype=torch.bool),
        "piece_ids": torch.tensor([0, 0, 1, 1]),
    }
    model.validation_step(batch, 0)
    # val_loss should be > 0 because contrastive loss is non-zero
    # (With old code, val_loss would be 0.0 since it only logged MSE)
    val_loss = model.trainer.callback_metrics.get("val_loss") if model.trainer else None
    # Without a trainer, we verify the method doesn't error.
    # The structural test: check that the method computes projection + InfoNCE.
    # We do this by monkey-patching log to capture what gets logged.
    logged = {}
    model.log = lambda name, value, **kw: logged.update({name: value})
    model.validation_step(batch, 0)
    assert "val_loss" in logged
    assert logged["val_loss"].item() > 0  # Contrastive component makes it non-zero
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/jdhiman/Documents/crescendai/model && python -m pytest tests/model_improvement/test_audio_encoders.py::TestMuQStagedModel::test_stage1_validation_logs_contrastive_loss -v`
Expected: FAIL -- `logged["val_loss"]` is 0.0 (only invariance MSE logged)

**Step 3: Fix the validation_step self_supervised branch**

Replace lines 833-838 in `audio_encoders.py`:

```python
if self.stage == "self_supervised":
    z_clean = self.encode(batch["embeddings_clean"], batch.get("mask"))
    z_aug = self.encode(batch["embeddings_augmented"], batch.get("mask"))

    # Contrastive loss (same as training)
    proj_clean = self.projection(z_clean)
    proj_aug = self.projection(z_aug)
    all_proj = torch.cat([proj_clean, proj_aug], dim=0)
    all_pieces = torch.cat([batch["piece_ids"], batch["piece_ids"]], dim=0)
    l_contrast = piece_based_infonce_loss(
        all_proj, all_pieces, temperature=self.temperature
    )

    # Invariance loss
    l_inv = F.mse_loss(z_clean, z_aug)

    val_loss = l_contrast + self.lambda_invariance * l_inv
    self.log("val_loss", val_loss, prog_bar=True)
    self.log("val_contrast_loss", l_contrast)
    self.log("val_inv_loss", l_inv)
```

**Step 4: Run test to verify it passes + existing tests**

Run: `cd /Users/jdhiman/Documents/crescendai/model && python -m pytest tests/model_improvement/test_audio_encoders.py::TestMuQStagedModel -v`
Expected: All pass

**Step 5: Commit**

```bash
git add model/src/model_improvement/audio_encoders.py model/tests/model_improvement/test_audio_encoders.py
git commit -m "fix(model): log contrastive+invariance for A2 stage1 validation"
```

---

### Task 5: Add hardware-aware training config to training.py

**Files:**
- Modify: `model/src/model_improvement/training.py`
- Test: `model/tests/model_improvement/test_training.py` (new file)

**Step 1: Write the failing tests**

Create `model/tests/model_improvement/test_training.py`:

```python
import pytest
from pathlib import Path
from unittest.mock import patch
from model_improvement.training import detect_accelerator_config, find_checkpoint


class TestDetectAcceleratorConfig:
    @patch("torch.cuda.is_available", return_value=True)
    def test_cuda_config(self, _mock):
        cfg = detect_accelerator_config()
        assert cfg["precision"] == "bf16-mixed"
        assert cfg["accelerator"] == "auto"
        assert cfg["deterministic"] is True

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=True)
    def test_mps_config(self, _mock_mps, _mock_cuda):
        cfg = detect_accelerator_config()
        assert cfg["precision"] == "32-true"
        assert cfg["accelerator"] == "auto"
        assert cfg["deterministic"] is False

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=False)
    def test_cpu_config(self, _mock_mps, _mock_cuda):
        cfg = detect_accelerator_config()
        assert cfg["precision"] == "32-true"
        assert cfg["accelerator"] == "cpu"
        assert cfg["deterministic"] is False


class TestFindCheckpoint:
    def test_finds_existing_checkpoint(self, tmp_path):
        ckpt_dir = tmp_path / "A1" / "fold_0"
        ckpt_dir.mkdir(parents=True)
        ckpt_file = ckpt_dir / "epoch=5-val_loss=0.1234.ckpt"
        ckpt_file.touch()
        result = find_checkpoint(tmp_path, "A1", 0)
        assert result == ckpt_file

    def test_returns_none_when_no_checkpoint(self, tmp_path):
        result = find_checkpoint(tmp_path, "A1", 0)
        assert result is None

    def test_returns_none_for_empty_dir(self, tmp_path):
        ckpt_dir = tmp_path / "A1" / "fold_0"
        ckpt_dir.mkdir(parents=True)
        result = find_checkpoint(tmp_path, "A1", 0)
        assert result is None
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/jdhiman/Documents/crescendai/model && python -m pytest tests/model_improvement/test_training.py -v`
Expected: FAIL -- `detect_accelerator_config` and `find_checkpoint` don't exist

**Step 3: Implement detect_accelerator_config and find_checkpoint**

Add to `training.py` before `train_model`:

```python
import torch


def detect_accelerator_config() -> dict:
    """Auto-detect hardware and return appropriate trainer kwargs."""
    if torch.cuda.is_available():
        return {
            "accelerator": "auto",
            "precision": "bf16-mixed",
            "deterministic": True,
        }
    if torch.backends.mps.is_available():
        return {
            "accelerator": "auto",
            "precision": "32-true",
            "deterministic": False,
        }
    return {
        "accelerator": "cpu",
        "precision": "32-true",
        "deterministic": False,
    }


def find_checkpoint(
    checkpoint_dir: str | Path, model_name: str, fold_idx: int
) -> Path | None:
    """Find an existing best checkpoint for a model/fold.

    Returns the path to the .ckpt file if found, None otherwise.
    """
    ckpt_dir = Path(checkpoint_dir) / model_name / f"fold_{fold_idx}"
    if not ckpt_dir.exists():
        return None
    ckpts = sorted(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        return None
    return ckpts[0]
```

**Step 4: Update train_model to use detect_accelerator_config**

Replace the hardcoded precision/deterministic in `train_model` (lines 70-80):

```python
def train_model(
    model: pl.LightningModule,
    train_loader,
    val_loader,
    model_name: str,
    fold_idx: int,
    checkpoint_dir: str | Path,
    max_epochs: int = 200,
    monitor: str = "val_loss",
    upload_remote: str | None = None,
    precision: str | None = None,
    gradient_clip_val: float = 1.0,
    patience: int = 10,
) -> pl.Trainer:
    ckpt_dir = Path(checkpoint_dir) / model_name / f"fold_{fold_idx}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    hw = detect_accelerator_config()
    if precision is not None:
        hw["precision"] = precision

    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="{epoch}-{" + monitor + ":.4f}",
            monitor=monitor,
            mode="min",
            save_top_k=1,
        ),
        EarlyStopping(
            monitor=monitor,
            patience=patience,
            mode="min",
        ),
    ]

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=hw["accelerator"],
        devices=1,
        precision=hw["precision"],
        gradient_clip_val=gradient_clip_val,
        callbacks=callbacks,
        enable_progress_bar=True,
        log_every_n_steps=10,
        deterministic=hw["deterministic"],
    )

    trainer.fit(model, train_loader, val_loader)

    if upload_remote is not None:
        upload_checkpoint(ckpt_dir, f"{model_name}/fold_{fold_idx}")

    return trainer
```

**Step 5: Run tests**

Run: `cd /Users/jdhiman/Documents/crescendai/model && python -m pytest tests/model_improvement/test_training.py -v`
Expected: All pass

**Step 6: Commit**

```bash
git add model/src/model_improvement/training.py model/tests/model_improvement/test_training.py
git commit -m "feat(model): add hardware detection and checkpoint finder to training.py"
```

---

### Task 6: Update notebook for local training, resume, and memory fixes

**Files:**
- Modify: `model/notebooks/model_improvement/01_audio_training.ipynb`

This task modifies the notebook only. No pytest -- verify by reading the cells.

**Step 1: Update Setup cell (cell-2)**

Replace the CUBLAS line:

```python
import subprocess
import sys
import os
import torch
if torch.cuda.is_available():
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
```

**Step 2: Update cell-3 (clone/install/data sync)**

Make cloud-only operations conditional:

```python
import sys
from pathlib import Path

IS_CLOUD = Path('/workspace').exists()

if IS_CLOUD:
    get_ipython().system('git clone https://github.com/Jai-Dhiman/crescendAI.git /workspace/crescendai')
    get_ipython().run_line_magic('cd', '/workspace/crescendai/model')
    get_ipython().system('curl -LsSf https://astral.sh/uv/install.sh | sh')
    get_ipython().system(f'uv pip install -e . --python {sys.executable}')
    get_ipython().system('rclone sync gdrive:crescendai_data/model_improvement/data/percepiano_cache ./data/percepiano_cache --progress')
    get_ipython().system('rclone sync gdrive:crescendai_data/model_improvement/data/competition_cache ./data/competition_cache --progress')
    get_ipython().system('rclone sync gdrive:crescendai_data/model_improvement/data/maestro_cache ./data/maestro_cache --progress')
    get_ipython().system("rclone copy gdrive:crescendai_data/model_improvement/data/composite_labels ../data/composite_labels --progress 2>/dev/null || echo 'composite_labels: using local copy'")

    DATA_DIR = Path('/workspace/crescendai/model/data')
    CHECKPOINT_DIR = Path('/workspace/crescendai/model/checkpoints/model_improvement')
else:
    DATA_DIR = Path('../data')  # Relative to model/ dir
    CHECKPOINT_DIR = Path('../data/checkpoints/model_improvement')

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
print(f'DATA_DIR: {DATA_DIR.resolve()}')
print(f'CHECKPOINT_DIR: {CHECKPOINT_DIR.resolve()}')
print(f'Running on: {"cloud" if IS_CLOUD else "local"}')
```

**Step 3: Update imports cell (cell-4)**

Add the new training imports:

```python
from model_improvement.training import train_model, find_checkpoint, detect_accelerator_config, upload_checkpoint
```

Remove the inline `train_model` and `upload_checkpoint` definitions from cell-10 entirely (they now come from `training.py`).

**Step 4: Update build_dataloaders in cell-10**

Replace `num_workers=4, pin_memory=True` with:

```python
_NUM_WORKERS = 4 if torch.cuda.is_available() else 2
_PIN_MEMORY = torch.cuda.is_available()
```

And use `num_workers=_NUM_WORKERS, pin_memory=_PIN_MEMORY` in both DataLoader calls.

Remove the inline `train_model` and `upload_checkpoint` functions from cell-10 (imported from training.py now).

**Step 5: Update notebook configs (cells 12, 15, 19)**

A1_CONFIG (cell-12):
```python
'learning_rate': 3e-5,   # was 1e-4
'warmup_epochs': 5,       # new
```

A2_CONFIG (cell-15):
```python
'learning_rate': 3e-5,   # was 1e-4
'warmup_epochs': 5,       # new
```

A3_CONFIG (cell-19):
```python
'learning_rate': 3e-5,   # was 1e-4
'warmup_epochs': 5,       # new
```

**Step 6: Add checkpoint resume to A1 training loop (cell-12)**

Replace the fold loop with resume logic:

```python
a1_trainers = []
a1_models = []
for fold_idx, fold in enumerate(folds):
    print(f'\nFold {fold_idx}/{len(folds)-1}')

    existing = find_checkpoint(CHECKPOINT_DIR, 'A1', fold_idx)
    if existing:
        print(f'  Loading existing checkpoint: {existing.name}')
        model = MuQLoRAModel.load_from_checkpoint(existing, **A1_CONFIG)
        a1_models.append(model)
        a1_trainers.append(None)
        continue

    model = MuQLoRAModel(**A1_CONFIG)
    train_loader, val_loader = build_dataloaders(
        fold, labels, piece_to_keys, embeddings, cache_dir,
        batch_size=16,
    )

    trainer = train_model(
        model, train_loader, val_loader, 'A1', fold_idx,
        checkpoint_dir=CHECKPOINT_DIR,
        max_epochs=A1_CONFIG['max_epochs'],
        patience=10,
    )
    a1_trainers.append(trainer)
    a1_models.append(trainer.lightning_module)

    best_val = trainer.callback_metrics.get('val_loss', float('inf'))
    best_acc = trainer.callback_metrics.get('val_pairwise_acc', 0.0)
    print(f'  Best val_loss={best_val:.4f}, val_pairwise_acc={best_acc:.4f}')
```

**Step 7: Add checkpoint resume to A2 training loop (cell-15)**

Same pattern. Check for `A2_stage1` and `A2` checkpoints separately:

```python
a2_trainers = []
a2_models = []
for fold_idx, fold in enumerate(folds):
    print(f'\nFold {fold_idx}/{len(folds)-1}')

    # Check if Stage 2 (final) is already done
    existing_s2 = find_checkpoint(CHECKPOINT_DIR, 'A2', fold_idx)
    if existing_s2:
        print(f'  Loading existing A2 checkpoint: {existing_s2.name}')
        model = MuQStagedModel.load_from_checkpoint(existing_s2, **A2_CONFIG, stage='supervised')
        a2_models.append(model)
        a2_trainers.append(None)
        continue

    # --- Stage 1: Self-supervised ---
    existing_s1 = find_checkpoint(CHECKPOINT_DIR, 'A2_stage1', fold_idx)
    if existing_s1:
        print(f'  Stage 1: loading existing checkpoint: {existing_s1.name}')
        model = MuQStagedModel.load_from_checkpoint(existing_s1, **A2_CONFIG, stage='self_supervised')
    else:
        print('  Stage 1: Self-supervised (contrastive + invariance)')
        model = MuQStagedModel(**A2_CONFIG, stage='self_supervised')
        train_loader_ss, val_loader_ss = build_dataloaders(
            fold, labels, piece_to_keys, embeddings, cache_dir,
            batch_size=16, include_augmented=True,
        )
        train_model(
            model, train_loader_ss, val_loader_ss,
            'A2_stage1', fold_idx,
            checkpoint_dir=CHECKPOINT_DIR,
            max_epochs=STAGE1_EPOCHS,
            patience=10,
        )

    # --- Stage 2: Supervised ---
    print('  Stage 2: Supervised (ranking + regression)')
    model.switch_to_supervised()
    train_loader, val_loader = build_dataloaders(
        fold, labels, piece_to_keys, embeddings, cache_dir,
        batch_size=16,
    )
    trainer_s2 = train_model(
        model, train_loader, val_loader,
        'A2', fold_idx,
        checkpoint_dir=CHECKPOINT_DIR,
        max_epochs=STAGE2_EPOCHS,
        patience=10,
    )
    a2_trainers.append(trainer_s2)
    a2_models.append(trainer_s2.lightning_module)

    best_val = trainer_s2.callback_metrics.get('val_loss', float('inf'))
    best_acc = trainer_s2.callback_metrics.get('val_pairwise_acc', 0.0)
    print(f'  Best val_loss={best_val:.4f}, val_pairwise_acc={best_acc:.4f}')
```

**Step 8: Add checkpoint resume to A3 training loop (cell-19)**

Same pattern as A1:

```python
a3_trainers = []
a3_models = []
for fold_idx, fold in enumerate(folds):
    print(f'\nFold {fold_idx}/{len(folds)-1}')

    existing = find_checkpoint(CHECKPOINT_DIR, 'A3', fold_idx)
    if existing:
        print(f'  Loading existing checkpoint: {existing.name}')
        model = MuQFullUnfreezeModel.load_from_checkpoint(existing, **A3_CONFIG)
        a3_models.append(model)
        a3_trainers.append(None)
        continue

    model = MuQFullUnfreezeModel(**A3_CONFIG)
    train_loader, val_loader = build_dataloaders(
        fold, labels, piece_to_keys, embeddings, cache_dir,
        batch_size=16,
    )

    trainer = train_model(
        model, train_loader, val_loader, 'A3', fold_idx,
        checkpoint_dir=CHECKPOINT_DIR,
        max_epochs=A3_CONFIG['max_epochs'],
        patience=10,
    )
    a3_trainers.append(trainer)
    a3_models.append(trainer.lightning_module)

    best_val = trainer.callback_metrics.get('val_loss', float('inf'))
    best_acc = trainer.callback_metrics.get('val_pairwise_acc', 0.0)
    print(f'  Best val_loss={best_val:.4f}, val_pairwise_acc={best_acc:.4f}')
```

**Step 9: Update evaluation section (cell-25) to use a1_models/a2_models/a3_models**

Replace `[t.lightning_module for t in a1_trainers]` with `a1_models` (and same for a2, a3), since some trainers may be None (loaded from checkpoint).

**Step 10: Update MAESTRO ablation (cell-17) DataLoader settings**

Replace `num_workers=4, pin_memory=True` with `num_workers=_NUM_WORKERS, pin_memory=_PIN_MEMORY` in both DataLoader calls in the MAESTRO ablation cell.

**Step 11: Update upload cell (cell-23)**

Make upload conditional on cloud:

```python
if IS_CLOUD:
    for model_name in ['A1', 'A2', 'A2_stage1', 'A3']:
        local = CHECKPOINT_DIR / model_name
        if local.exists():
            upload_checkpoint(local, model_name)
            print(f'Uploaded {model_name} checkpoints')
else:
    print('Local run -- skipping GDrive upload.')
    print(f'Checkpoints saved to: {CHECKPOINT_DIR.resolve()}')
```

**Step 12: Commit**

```bash
git add model/notebooks/model_improvement/01_audio_training.ipynb
git commit -m "feat(model): update audio training notebook for local MPS, resume, memory fixes"
```

---

### Task 7: Run full test suite and verify

**Files:** None (verification only)

**Step 1: Run all audio encoder tests**

Run: `cd /Users/jdhiman/Documents/crescendai/model && python -m pytest tests/model_improvement/test_audio_encoders.py tests/model_improvement/test_training.py -v`
Expected: All pass

**Step 2: Run full model_improvement test suite**

Run: `cd /Users/jdhiman/Documents/crescendai/model && python -m pytest tests/model_improvement/ -v`
Expected: All pass (no regressions in data, tokenizer, graph, etc.)

**Step 3: Commit any fixes if needed, then final commit**

```bash
git add -A
git commit -m "test: verify all audio training fixes pass"
```
