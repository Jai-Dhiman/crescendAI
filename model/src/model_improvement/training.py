"""Training utilities for symbolic encoder experiments."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from model_improvement.trackio_callback import TrackioCallback


def _log_msg(msg: str, log_file: str | None = None) -> None:
    """Print to stdout and optionally append to a log file."""
    print(msg, flush=True)
    if log_file:
        with open(log_file, "a") as f:
            f.write(f"{msg}\n")


class PrintEpochCallback(pl.Callback):
    """Print epoch metrics + memory to stdout (useful when tqdm doesn't render over SSH)."""

    def __init__(self, log_file: str | None = None):
        super().__init__()
        self.log_file = log_file

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = {
            k: f"{v:.4f}"
            for k, v in trainer.callback_metrics.items()
            if isinstance(v, (int, float, torch.Tensor))
        }
        mem = ""
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            mem = f" | GPU: {alloc:.1f}GB alloc, {reserved:.1f}GB reserved"
        try:
            import psutil
            ram = psutil.Process().memory_info().rss / 1024**3
            mem += f" | RAM: {ram:.1f}GB"
        except ImportError:
            pass
        _log_msg(f"Epoch {trainer.current_epoch}: {metrics}{mem}", self.log_file)


class BatchProgressCallback(pl.Callback):
    """Print progress every N training batches so long epochs aren't silent."""

    def __init__(self, log_every: int = 50, log_file: str | None = None):
        super().__init__()
        self.log_every = log_every
        self.log_file = log_file

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (batch_idx + 1) % self.log_every == 0:
            total = trainer.num_training_batches
            loss = outputs["loss"].item() if isinstance(outputs, dict) else float("nan")
            mem = ""
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / 1024**3
                mem = f" | GPU: {alloc:.1f}GB"
            _log_msg(
                f"  batch {batch_idx + 1}/{total} loss={loss:.4f}{mem}",
                self.log_file,
            )


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
    accumulate_grad_batches: int = 1,
    log_file: str | None = None,
    accelerator: str | None = None,
    trackio_experiment_id: str | None = None,
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
        accumulate_grad_batches: Accumulate gradients over N batches before
            stepping. Default ``1`` (no accumulation).
        log_file: Optional path to a log file. Callbacks will append progress
            messages here in addition to stdout. Useful for ``tail -f`` when
            Jupyter buffers stdout.
        accelerator: Override the auto-detected accelerator (e.g. ``"cpu"``
            to force CPU training when MPS scatter kernels cause overhead).

    Returns:
        The fitted Trainer instance.
    """
    ckpt_dir = Path(checkpoint_dir) / model_name / f"fold_{fold_idx}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    hw = detect_accelerator_config()
    if precision is not None:
        hw["precision"] = precision
    if accelerator is not None:
        hw["accelerator"] = accelerator

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
        PrintEpochCallback(log_file=log_file),
        BatchProgressCallback(log_every=50, log_file=log_file),
    ]
    if trackio_experiment_id is not None:
        callbacks.append(TrackioCallback(experiment_id=trackio_experiment_id))

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=hw["accelerator"],
        devices=1,
        precision=hw["precision"],
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=accumulate_grad_batches,
        callbacks=callbacks,
        enable_progress_bar=True,
        log_every_n_steps=10,
        deterministic=hw["deterministic"],
        num_sanity_val_steps=0,
    )

    trainer.fit(model, train_loader, val_loader)

    if upload_remote is not None:
        upload_checkpoint(ckpt_dir, f"{model_name}/fold_{fold_idx}")

    return trainer
