"""PyTorch Lightning callback for Trackio experiment tracking."""

from __future__ import annotations

import subprocess
from typing import Any

import pytorch_lightning as pl
import torch


class TrackioCallback(pl.Callback):
    """Log training metrics to Trackio per epoch.

    Logs: train_loss, val_skill_discrimination, learning_rate.
    Requires trackio to be installed: uv pip install trackio
    """

    def __init__(
        self,
        experiment_id: str,
        project: str = "crescendai-training",
        config: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.experiment_id = experiment_id
        self.project = project
        self.config = config or {}
        self._run = None

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        try:
            import trackio
            self._run = trackio.Run(
                name=self.experiment_id,
                project=self.project,
                config=self.config,
            )
            # Log git commit hash
            try:
                commit = subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"],
                    text=True, stderr=subprocess.DEVNULL,
                ).strip()
                self._run.log({"git_commit": commit}, step=0)
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass
        except ImportError:
            print("WARNING: trackio not installed, skipping experiment tracking")
            self._run = None

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._run is None:
            return

        metrics: dict[str, float] = {}
        for key, val in trainer.callback_metrics.items():
            if isinstance(val, torch.Tensor):
                metrics[key] = val.item()
            elif isinstance(val, (int, float)):
                metrics[key] = float(val)

        # Extract learning rate from optimizer
        if trainer.optimizers:
            opt = trainer.optimizers[0]
            metrics["learning_rate"] = opt.param_groups[0]["lr"]

        self._run.log(metrics, step=trainer.current_epoch)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._run is not None:
            self._run.finish()
