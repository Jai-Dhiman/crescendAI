"""PyTorch Lightning callback for Trackio experiment tracking."""

from __future__ import annotations

import subprocess
from typing import Any

import pytorch_lightning as pl
import torch


class TrackioCallback(pl.Callback):
    """Log training metrics to Trackio per epoch.

    Uses the wandb-compatible trackio.init / trackio.log / trackio.finish API.
    Requires trackio to be installed: uv pip install trackio
    """

    def __init__(
        self,
        experiment_id: str,
        project: str = "crescendai-training",
        config: dict[str, Any] | None = None,
        space_id: str | None = None,
    ):
        super().__init__()
        self.experiment_id = experiment_id
        self.project = project
        self.config = config or {}
        self.space_id = space_id
        self._active = False

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        try:
            import trackio
            init_kwargs: dict[str, Any] = {
                "name": self.experiment_id,
                "project": self.project,
                "config": self.config,
            }
            if self.space_id is not None:
                init_kwargs["space_id"] = self.space_id
            trackio.init(**init_kwargs)
            self._active = True
            try:
                commit = subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"],
                    text=True, stderr=subprocess.DEVNULL,
                ).strip()
                trackio.log({"git_commit": commit}, step=0)
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass
        except ImportError:
            print("WARNING: trackio not installed, skipping experiment tracking")

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self._active:
            return

        import trackio
        metrics: dict[str, float] = {}
        for key, val in trainer.callback_metrics.items():
            if isinstance(val, torch.Tensor):
                metrics[key] = val.item()
            elif isinstance(val, (int, float)):
                metrics[key] = float(val)

        if trainer.optimizers:
            metrics["learning_rate"] = trainer.optimizers[0].param_groups[0]["lr"]

        trackio.log(metrics, step=trainer.current_epoch)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._active:
            import trackio
            trackio.finish()
            self._active = False
