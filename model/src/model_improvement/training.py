"""Training utilities for symbolic encoder experiments."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


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


def upload_checkpoint(local_path: str | Path, remote_subdir: str) -> None:
    """Sync a local checkpoint directory to Google Drive via rclone."""
    remote = f"gdrive:crescendai_data/model_improvement/checkpoints/{remote_subdir}"
    subprocess.run(
        ["rclone", "copy", str(local_path), remote, "--progress"], check=True
    )


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
    """Train a model with standard callbacks.

    Args:
        model: PyTorch Lightning model to train.
        train_loader: Training dataloader.
        val_loader: Validation dataloader.
        model_name: Name for checkpoint subdirectory.
        fold_idx: Cross-validation fold index.
        checkpoint_dir: Base directory for saving checkpoints.
        max_epochs: Maximum training epochs.
        monitor: Metric to monitor for checkpointing/early stopping.
        upload_remote: If not None, the remote base path for rclone upload.
            Checkpoints are uploaded to ``{upload_remote}/{model_name}/fold_{fold_idx}``.
        precision: Training precision override. If None, auto-detected.
        gradient_clip_val: Max gradient norm for clipping. Default ``1.0``.
        patience: Early stopping patience in epochs. Default ``10``.

    Returns:
        The fitted Trainer instance.
    """
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
