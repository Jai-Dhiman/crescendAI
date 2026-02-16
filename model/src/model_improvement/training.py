"""Training utilities for symbolic encoder experiments."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


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

    Returns:
        The fitted Trainer instance.
    """
    ckpt_dir = Path(checkpoint_dir) / model_name / f"fold_{fold_idx}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

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
            patience=20,
            mode="min",
        ),
    ]

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        callbacks=callbacks,
        enable_progress_bar=True,
        log_every_n_steps=10,
        deterministic=True,
    )

    trainer.fit(model, train_loader, val_loader)

    if upload_remote is not None:
        upload_checkpoint(ckpt_dir, f"{model_name}/fold_{fold_idx}")

    return trainer
