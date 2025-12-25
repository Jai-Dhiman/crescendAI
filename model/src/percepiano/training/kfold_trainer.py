"""
K-Fold Cross-Validation Trainer for PercePiano

Implements 4-fold cross-validation following the PercePiano paper methodology.
Each fold trains a model with proper train/val splits and early stopping.
Final evaluation aggregates results across all folds.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

from ..data.kfold_split import get_test_samples, load_fold_assignments
from ..data.percepiano_vnet_dataset import (
    PercePianoKFoldDataModule,
    PercePianoTestDataset,
)
from ..models.percepiano_replica import PERCEPIANO_DIMENSIONS, PercePianoVNetModule


class EpochLogger(Callback):
    """Callback for logging epoch summaries during training."""

    def __init__(self, fold_id: int):
        super().__init__()
        self.fold_id = fold_id
        self.epoch_start_time = None
        self.best_val_r2 = -float("inf")
        self.best_epoch = 0

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        epoch_time = time.time() - self.epoch_start_time

        # Get metrics
        train_loss = trainer.callback_metrics.get("train/loss", 0.0)
        val_loss = trainer.callback_metrics.get("val/loss", 0.0)
        val_r2 = trainer.callback_metrics.get("val/mean_r2", 0.0)
        lr = trainer.optimizers[0].param_groups[0]["lr"]

        # Track best
        is_best = ""
        if val_r2 > self.best_val_r2:
            self.best_val_r2 = val_r2
            self.best_epoch = epoch
            is_best = " *best*"

        # Log every epoch
        print(
            f"  [Fold {self.fold_id}] Epoch {epoch:3d} | "
            f"train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | "
            f"val_r2: {val_r2:+.4f} | lr: {lr:.2e} | "
            f"time: {epoch_time:.1f}s{is_best}"
        )


class GradientMonitorCallback(Callback):
    """
    DIAGNOSTIC: Comprehensive gradient and activation monitoring.

    Logs gradient norms by layer category, detects explosion/vanishing,
    and tracks key metrics for debugging training issues.
    """

    def __init__(self, log_every_n_steps: int = 100, verbose_first_n_steps: int = 5):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.verbose_first_n_steps = verbose_first_n_steps
        self.step_losses = []

    def on_after_backward(self, trainer, pl_module):
        step = trainer.global_step
        is_verbose_step = step < self.verbose_first_n_steps
        is_log_step = step % self.log_every_n_steps == 0

        if not (is_verbose_step or is_log_step):
            return

        # Collect gradient norms by category
        grad_stats = {
            "han_encoder": {"norm": 0.0, "count": 0, "max": 0.0},
            "performance_contractor": {"norm": 0.0, "count": 0, "max": 0.0},
            "final_attention": {"norm": 0.0, "count": 0, "max": 0.0},
            "prediction_head": {"norm": 0.0, "count": 0, "max": 0.0},
            "other": {"norm": 0.0, "count": 0, "max": 0.0},
        }

        total_norm_sq = 0.0
        max_param_norm = 0.0
        max_param_name = ""

        for name, param in pl_module.named_parameters():
            if param.grad is None:
                continue

            param_norm = param.grad.data.norm(2).item()
            total_norm_sq += param_norm**2

            # Track max gradient
            if param_norm > max_param_norm:
                max_param_norm = param_norm
                max_param_name = name

            # Categorize
            if "han_encoder" in name:
                cat = "han_encoder"
            elif "performance_contractor" in name:
                cat = "performance_contractor"
            elif "final_attention" in name:
                cat = "final_attention"
            elif "prediction_head" in name:
                cat = "prediction_head"
            else:
                cat = "other"

            grad_stats[cat]["norm"] += param_norm**2
            grad_stats[cat]["count"] += 1
            grad_stats[cat]["max"] = max(grad_stats[cat]["max"], param_norm)

        total_norm = total_norm_sq**0.5

        # Compute category norms
        for cat in grad_stats:
            if grad_stats[cat]["count"] > 0:
                grad_stats[cat]["norm"] = grad_stats[cat]["norm"]**0.5

        # Get current loss from trainer
        current_loss = trainer.callback_metrics.get("train/loss", float("nan"))
        if hasattr(current_loss, "item"):
            current_loss = current_loss.item()
        self.step_losses.append(current_loss)

        # Print diagnostics
        print(f"\n  [DIAG] Step {step}: total_grad_norm={total_norm:.4f}, loss={current_loss:.6f}")
        print(f"    Gradient by category:")
        for cat, stats in grad_stats.items():
            if stats["count"] > 0:
                print(f"      {cat:<22} norm={stats['norm']:8.4f}  max={stats['max']:8.4f}  params={stats['count']}")

        if max_param_norm > 0:
            print(f"    Max gradient: {max_param_name} = {max_param_norm:.4f}")

        # Gradient health assessment
        if total_norm > 100:
            print(f"  [DIAG] WARNING: Large gradient norm (>{100}). May need gradient clipping.")
        elif total_norm > 10:
            print(f"  [DIAG] NOTE: Moderate gradient norm. Training should be stable with clip=2.0")
        elif total_norm < 1e-6:
            print(f"  [DIAG] WARNING: Vanishing gradients detected!")
        elif total_norm < 0.01:
            print(f"  [DIAG] NOTE: Small gradients. Learning may be slow.")

        # Check for NaN/Inf
        if not np.isfinite(total_norm):
            print(f"  [DIAG] CRITICAL: NaN/Inf in gradients! Training will fail.")

        # Track loss trend (after first few steps)
        if len(self.step_losses) >= 10:
            recent_losses = [l for l in self.step_losses[-10:] if np.isfinite(l)]
            if len(recent_losses) >= 5:
                loss_trend = recent_losses[-1] - recent_losses[0]
                if loss_trend > 0:
                    print(f"  [DIAG] WARNING: Loss increasing over last 10 steps ({loss_trend:+.6f})")


class ActivationDiagnosticCallback(Callback):
    """
    DIAGNOSTIC: Run activation diagnostics on first few batches.

    Calls forward with diagnose=True to print intermediate activation statistics.
    This helps identify where values explode or collapse in the forward pass.
    """

    def __init__(self, diagnose_first_n_batches: int = 3):
        super().__init__()
        self.diagnose_first_n_batches = diagnose_first_n_batches
        self.batch_count = 0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.batch_count < self.diagnose_first_n_batches:
            print(f"\n{'='*60}")
            print(f"  ACTIVATION DIAGNOSTICS - Batch {self.batch_count}")
            print(f"{'='*60}")

            # Run diagnostic forward pass
            with torch.no_grad():
                input_features = batch["input_features"]
                note_locations = {
                    "beat": batch["note_locations_beat"],
                    "measure": batch["note_locations_measure"],
                    "voice": batch["note_locations_voice"],
                }
                targets = batch["scores"]

                # Move to device
                device = next(pl_module.parameters()).device
                input_features = input_features.to(device)
                note_locations = {k: v.to(device) for k, v in note_locations.items()}
                targets = targets.to(device)

                # Run forward with diagnostics
                _ = pl_module(
                    input_features,
                    note_locations,
                    diagnose=True,
                )

                # Also print target statistics
                print(f"  [ACT] Targets: shape={targets.shape}")
                print(f"    mean={targets.mean().item():.4f}, std={targets.std().item():.4f}")
                print(f"    min={targets.min().item():.4f}, max={targets.max().item():.4f}")

            print(f"{'='*60}\n")
            self.batch_count += 1


@dataclass
class FoldMetrics:
    """Metrics for a single fold."""

    fold_id: int
    train_loss: float
    val_loss: float
    val_r2: float
    val_pearson: float
    val_spearman: float
    val_mae: float
    val_rmse: float
    per_dim_r2: Dict[str, float]
    per_dim_pearson: Dict[str, float]
    epochs_trained: int
    best_epoch: int
    training_time_seconds: float
    n_train_samples: int
    n_val_samples: int
    prediction_stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregateMetrics:
    """Aggregated metrics across all folds."""

    mean_r2: float
    std_r2: float
    mean_pearson: float
    std_pearson: float
    mean_spearman: float
    std_spearman: float
    mean_mae: float
    std_mae: float
    mean_rmse: float
    std_rmse: float
    per_dim_mean_r2: Dict[str, float]
    per_dim_std_r2: Dict[str, float]
    total_training_time: float


class KFoldTrainer:
    """
    K-Fold Cross-Validation trainer for PercePiano.

    Trains models for each fold, saves checkpoints, and aggregates metrics.
    Supports resuming from checkpoints after interruption.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        fold_assignments: Dict[str, Dict[str, Union[int, str]]],
        data_dir: Union[str, Path],
        checkpoint_dir: Union[str, Path],
        log_dir: Optional[Union[str, Path]] = None,
        n_folds: int = 4,
    ):
        self.config = config
        self.fold_assignments = fold_assignments
        self.data_dir = Path(data_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir) if log_dir else self.checkpoint_dir / "logs"
        self.n_folds = n_folds

        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Storage for results
        self.fold_metrics: List[FoldMetrics] = []
        self.fold_checkpoints: List[Path] = []
        self.total_start_time: Optional[float] = None

    def _find_checkpoint(self, fold_id: int, checkpoint_type: str = "last") -> Optional[Path]:
        """
        Find existing checkpoint for a fold.

        Args:
            fold_id: The fold number
            checkpoint_type: "last" for last.ckpt, "best" for best checkpoint

        Returns:
            Path to checkpoint if found, None otherwise
        """
        fold_dir = self.checkpoint_dir / f"fold_{fold_id}"
        if not fold_dir.exists():
            return None

        if checkpoint_type == "last":
            last_ckpt = fold_dir / "last.ckpt"
            if last_ckpt.exists():
                return last_ckpt
        elif checkpoint_type == "best":
            # Find best checkpoint (format: best-epoch=XX-val/mean_r2=X.XXXX.ckpt)
            best_ckpts = list(fold_dir.glob("best-*.ckpt"))
            if best_ckpts:
                # Return most recent best checkpoint
                return max(best_ckpts, key=lambda p: p.stat().st_mtime)

        return None

    def _load_completed_folds(self) -> Tuple[int, List[Path]]:
        """
        Detect which folds have completed training (have best checkpoints).

        Returns:
            Tuple of (first_incomplete_fold, list_of_best_checkpoints)
        """
        completed_checkpoints = []
        first_incomplete = 0

        for fold_id in range(self.n_folds):
            best_ckpt = self._find_checkpoint(fold_id, "best")
            if best_ckpt:
                completed_checkpoints.append(best_ckpt)
                first_incomplete = fold_id + 1
            else:
                break

        return first_incomplete, completed_checkpoints

    def _create_model(self) -> PercePianoVNetModule:
        """Create a new model instance."""
        return PercePianoVNetModule(
            input_size=self.config.get("input_size", 78),  # SOTA uses 78 features
            hidden_size=self.config.get("hidden_size", 256),
            note_layers=self.config.get("note_layers", 2),
            voice_layers=self.config.get("voice_layers", 2),
            beat_layers=self.config.get("beat_layers", 2),
            measure_layers=self.config.get("measure_layers", 1),
            num_attention_heads=self.config.get("num_attention_heads", 8),
            final_hidden=self.config.get("final_hidden", 128),
            dropout=self.config.get("dropout", 0.2),
            learning_rate=self.config.get("learning_rate", 2.5e-5),  # SOTA: 2.5e-5
            weight_decay=self.config.get("weight_decay", 1e-5),
        )

    def _create_callbacks(self, fold_id: int) -> List[pl.Callback]:
        """Create callbacks for training."""
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.checkpoint_dir / f"fold_{fold_id}",
            filename="best-{epoch:02d}-{val/mean_r2:.4f}",
            monitor="val/mean_r2",
            mode="max",
            save_top_k=1,
            save_last=True,
        )

        early_stopping = EarlyStopping(
            monitor="val/mean_r2",
            mode="max",
            patience=self.config.get("early_stopping_patience", 20),
            verbose=False,  # We have our own logging
        )

        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        epoch_logger = EpochLogger(fold_id=fold_id)

        # DIAGNOSTIC: Add gradient monitoring (logs every 100 steps, verbose first 5)
        grad_monitor = GradientMonitorCallback(log_every_n_steps=100, verbose_first_n_steps=5)

        # DIAGNOSTIC: Add activation diagnostics (runs on first 3 batches)
        activation_diag = ActivationDiagnosticCallback(diagnose_first_n_batches=3)

        return [checkpoint_callback, early_stopping, lr_monitor, epoch_logger, grad_monitor, activation_diag]

    def train_fold(
        self, fold_id: int, verbose: bool = True, resume_from_checkpoint: bool = False
    ) -> FoldMetrics:
        """
        Train a single fold.

        Args:
            fold_id: The fold number to train
            verbose: Whether to print progress
            resume_from_checkpoint: If True, attempt to resume from last.ckpt

        Returns:
            FoldMetrics for this fold
        """
        start_time = time.time()

        # Check for existing checkpoint to resume from
        resume_ckpt_path = None
        if resume_from_checkpoint:
            resume_ckpt_path = self._find_checkpoint(fold_id, "last")
            if resume_ckpt_path and verbose:
                print(f"  Found checkpoint to resume from: {resume_ckpt_path}")

        # Create data module
        # SOTA uses batch_size=8 and no augmentation
        # num_workers=4 is optimal for A100XL (4 vCPUs) - avoids dataloader bottleneck
        data_module = PercePianoKFoldDataModule(
            data_dir=self.data_dir,
            fold_assignments=self.fold_assignments,
            fold_id=fold_id,
            batch_size=self.config.get("batch_size", 8),  # SOTA: 8
            max_notes=self.config.get("max_notes", 1024),
            num_workers=self.config.get("num_workers", 4),  # 4 for A100XL vCPUs
            augment_train=self.config.get(
                "augment_train", False
            ),  # SOTA: no augmentation
        )
        data_module.setup("fit")

        n_train = len(data_module.train_dataset)
        n_val = len(data_module.val_dataset)

        if verbose:
            print(f"\n{'=' * 70}")
            print(f"FOLD {fold_id}/{self.n_folds - 1}")
            print(f"{'=' * 70}")
            print(f"  Train samples: {n_train} | Val samples: {n_val}")
            print(
                f"  Batch size: {self.config.get('batch_size', 32)} | "
                f"Max epochs: {self.config.get('max_epochs', 100)} | "
                f"Early stop patience: {self.config.get('early_stopping_patience', 20)}"
            )
            if resume_ckpt_path:
                print(f"  RESUMING from checkpoint: {resume_ckpt_path.name}")
            print(f"{'=' * 70}")

        # Create model
        model = self._create_model()
        if verbose and fold_id == 0 and not resume_ckpt_path:
            print(f"  Model parameters: {model.count_parameters():,}")

        # Create callbacks
        callbacks = self._create_callbacks(fold_id)
        checkpoint_callback = callbacks[0]
        epoch_logger = callbacks[3]

        # Create logger
        logger = TensorBoardLogger(
            save_dir=str(self.log_dir),
            name=f"fold_{fold_id}",
            version=0,
        )

        # Create trainer
        trainer = pl.Trainer(
            max_epochs=self.config.get("max_epochs", 100),
            accelerator="auto",
            devices=1,
            # CRITICAL: Use FP32 to match original PercePiano (no mixed precision)
            # FP16 causes numerical instability in attention and prediction collapse
            precision=self.config.get("precision", "32"),
            gradient_clip_val=self.config.get("gradient_clip_val", 2.0),  # Matches original PercePiano
            callbacks=callbacks,
            logger=logger,
            enable_progress_bar=False,  # We use our own logging
            log_every_n_steps=10,
            enable_model_summary=False,
        )

        # Train (with optional checkpoint resume)
        trainer.fit(
            model,
            data_module,
            ckpt_path=str(resume_ckpt_path) if resume_ckpt_path else None,
        )

        training_time = time.time() - start_time

        # Get best checkpoint
        best_checkpoint = Path(checkpoint_callback.best_model_path)
        self.fold_checkpoints.append(best_checkpoint)

        # Load best model and do detailed evaluation
        best_model = PercePianoVNetModule.load_from_checkpoint(
            str(best_checkpoint),
            input_size=self.config.get("input_size", 78),  # SOTA: 78 features
            hidden_size=self.config.get("hidden_size", 256),
            note_layers=self.config.get("note_layers", 2),
            voice_layers=self.config.get("voice_layers", 2),
            beat_layers=self.config.get("beat_layers", 2),
            measure_layers=self.config.get("measure_layers", 1),
            num_attention_heads=self.config.get("num_attention_heads", 8),
            final_hidden=self.config.get("final_hidden", 128),
            dropout=self.config.get("dropout", 0.2),
        )

        # Detailed evaluation
        val_metrics = self._detailed_evaluation(
            best_model, data_module.val_dataloader(), "Validation", verbose
        )

        # Create fold metrics
        fold_metrics = FoldMetrics(
            fold_id=fold_id,
            train_loss=float(trainer.callback_metrics.get("train/loss", 0.0)),
            val_loss=val_metrics["loss"],
            val_r2=val_metrics["r2"],
            val_pearson=val_metrics["pearson"],
            val_spearman=val_metrics["spearman"],
            val_mae=val_metrics["mae"],
            val_rmse=val_metrics["rmse"],
            per_dim_r2=val_metrics["per_dim_r2"],
            per_dim_pearson=val_metrics["per_dim_pearson"],
            epochs_trained=trainer.current_epoch + 1,
            best_epoch=epoch_logger.best_epoch,
            training_time_seconds=training_time,
            n_train_samples=n_train,
            n_val_samples=n_val,
            prediction_stats=val_metrics["prediction_stats"],
        )

        self.fold_metrics.append(fold_metrics)

        if verbose:
            self._print_fold_summary(fold_metrics)

        return fold_metrics

    def _detailed_evaluation(
        self,
        model: PercePianoVNetModule,
        dataloader: DataLoader,
        split_name: str,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Detailed model evaluation with prediction analysis."""
        model.eval()
        device = next(model.parameters()).device

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in dataloader:
                input_features = batch["input_features"].to(device)
                note_locations = {
                    "beat": batch["note_locations_beat"].to(device),
                    "measure": batch["note_locations_measure"].to(device),
                    "voice": batch["note_locations_voice"].to(device),
                }
                targets = batch["scores"].to(device)

                outputs = model(input_features, note_locations)
                predictions = outputs["predictions"]

                all_preds.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        # Overall metrics
        mse = np.mean((preds - targets) ** 2)
        mae = np.mean(np.abs(preds - targets))
        rmse = np.sqrt(mse)
        r2 = r2_score(targets, preds)
        pearson = pearsonr(targets.flatten(), preds.flatten())[0]
        spearman = spearmanr(targets.flatten(), preds.flatten())[0]

        # Per-dimension metrics
        per_dim_r2 = {}
        per_dim_pearson = {}
        per_dim_pred_std = {}
        per_dim_target_std = {}

        for i, dim in enumerate(PERCEPIANO_DIMENSIONS):
            if targets.shape[1] > i:
                per_dim_r2[dim] = r2_score(targets[:, i], preds[:, i])
                per_dim_pearson[dim] = pearsonr(targets[:, i], preds[:, i])[0]
                per_dim_pred_std[dim] = float(np.std(preds[:, i]))
                per_dim_target_std[dim] = float(np.std(targets[:, i]))

        # Prediction distribution analysis
        prediction_stats = {
            "pred_mean": float(np.mean(preds)),
            "pred_std": float(np.std(preds)),
            "pred_min": float(np.min(preds)),
            "pred_max": float(np.max(preds)),
            "target_mean": float(np.mean(targets)),
            "target_std": float(np.std(targets)),
            "n_collapsed_dims": sum(1 for d in per_dim_pred_std.values() if d < 0.05),
            "per_dim_pred_std": per_dim_pred_std,
            "per_dim_target_std": per_dim_target_std,
        }

        return {
            "loss": mse,
            "r2": r2,
            "pearson": pearson,
            "spearman": spearman,
            "mae": mae,
            "rmse": rmse,
            "per_dim_r2": per_dim_r2,
            "per_dim_pearson": per_dim_pearson,
            "prediction_stats": prediction_stats,
            "predictions": preds,
            "targets": targets,
        }

    def _print_fold_summary(self, m: FoldMetrics) -> None:
        """Print detailed fold summary."""
        print(f"\n  {'─' * 66}")
        print(f"  FOLD {m.fold_id} SUMMARY")
        print(f"  {'─' * 66}")
        print(
            f"  Training: {m.epochs_trained} epochs ({m.training_time_seconds:.1f}s) | "
            f"Best epoch: {m.best_epoch}"
        )
        print(f"  Samples:  train={m.n_train_samples}, val={m.n_val_samples}")
        print()
        print(f"  Overall Metrics:")
        print(f"    R2:       {m.val_r2:+.4f}")
        print(f"    Pearson:  {m.val_pearson:+.4f}")
        print(f"    Spearman: {m.val_spearman:+.4f}")
        print(f"    MAE:      {m.val_mae:.4f}")
        print(f"    RMSE:     {m.val_rmse:.4f}")

        # Prediction health check
        stats = m.prediction_stats
        print()
        print(f"  Prediction Health:")
        print(
            f"    Pred range: [{stats['pred_min']:.3f}, {stats['pred_max']:.3f}] "
            f"(target: [0, 1])"
        )
        print(
            f"    Pred std: {stats['pred_std']:.4f} (target std: {stats['target_std']:.4f})"
        )
        if stats["n_collapsed_dims"] > 0:
            print(
                f"    WARNING: {stats['n_collapsed_dims']}/19 dimensions have collapsed predictions (std < 0.05)"
            )
        else:
            print(f"    No collapsed dimensions detected")

        # Per-dimension breakdown (top 5 and bottom 5)
        sorted_dims = sorted(m.per_dim_r2.items(), key=lambda x: x[1], reverse=True)
        print()
        print(f"  Top 5 Dimensions by R2:")
        for dim, r2 in sorted_dims[:5]:
            pearson = m.per_dim_pearson.get(dim, 0)
            print(f"    {dim:<22} R2={r2:+.4f}  r={pearson:+.4f}")

        print(f"  Bottom 5 Dimensions by R2:")
        for dim, r2 in sorted_dims[-5:]:
            pearson = m.per_dim_pearson.get(dim, 0)
            status = "NEGATIVE" if r2 < 0 else ""
            print(f"    {dim:<22} R2={r2:+.4f}  r={pearson:+.4f}  {status}")

    def train_all_folds(
        self,
        verbose: bool = True,
        resume: bool = False,
        fold_order: list = None,
    ) -> AggregateMetrics:
        """
        Train all folds sequentially.

        Args:
            verbose: Whether to print progress
            resume: If True, skip completed folds and resume any in-progress fold
            fold_order: Custom order to train folds (e.g., [1, 2, 3, 0] to train
                       balanced folds first). If None, uses [0, 1, ..., n_folds-1].

        Returns:
            AggregateMetrics across all folds
        """
        if fold_order is None:
            fold_order = list(range(self.n_folds))
        self.total_start_time = time.time()

        print(f"\n{'#' * 70}")
        print(f"  {self.n_folds}-FOLD CROSS-VALIDATION")
        print(f"{'#' * 70}")
        print(f"  Data directory: {self.data_dir}")
        print(f"  Checkpoint directory: {self.checkpoint_dir}")
        print(f"  Fold order: {fold_order}")
        print(
            f"  Config: lr={self.config.get('learning_rate')}, "
            f"hidden={self.config.get('hidden_size')}, "
            f"batch={self.config.get('batch_size')}"
        )

        # Check for existing checkpoints if resuming
        start_fold = 0
        if resume:
            start_fold, completed_checkpoints = self._load_completed_folds()
            if completed_checkpoints:
                print(f"\n  RESUME MODE: Found {len(completed_checkpoints)} completed folds")
                print(f"  Skipping folds 0-{start_fold - 1}, starting from fold {start_fold}")
                # Load completed fold checkpoints
                self.fold_checkpoints = completed_checkpoints
                # Load metrics for completed folds from saved results if available
                results_file = self.checkpoint_dir / "kfold_results.json"
                if results_file.exists():
                    with open(results_file) as f:
                        saved_results = json.load(f)
                    for fm in saved_results.get("fold_metrics", [])[:start_fold]:
                        self.fold_metrics.append(
                            FoldMetrics(
                                fold_id=fm["fold_id"],
                                train_loss=fm["train_loss"],
                                val_loss=fm["val_loss"],
                                val_r2=fm["val_r2"],
                                val_pearson=fm["val_pearson"],
                                val_spearman=fm["val_spearman"],
                                val_mae=fm["val_mae"],
                                val_rmse=fm["val_rmse"],
                                per_dim_r2=fm["per_dim_r2"],
                                per_dim_pearson=fm.get("per_dim_pearson", {}),
                                epochs_trained=fm["epochs_trained"],
                                best_epoch=fm["best_epoch"],
                                training_time_seconds=fm["training_time_seconds"],
                                n_train_samples=fm["n_train_samples"],
                                n_val_samples=fm["n_val_samples"],
                            )
                        )
                    print(f"  Loaded metrics for {len(self.fold_metrics)} completed folds")

        # Filter fold_order to skip already completed folds
        folds_to_train = [f for f in fold_order if f >= start_fold or f not in range(start_fold)]
        if resume and start_fold > 0:
            # When resuming, skip folds that are already complete
            completed_fold_ids = {m.fold_id for m in self.fold_metrics}
            folds_to_train = [f for f in fold_order if f not in completed_fold_ids]
            print(f"  Training folds in order: {folds_to_train}")

        for i, fold_id in enumerate(folds_to_train):
            # For the first fold after resume, try to resume from checkpoint
            resume_from_ckpt = resume and i == 0
            self.train_fold(fold_id, verbose=verbose, resume_from_checkpoint=resume_from_ckpt)

            # Save results after each fold (for crash recovery)
            self.save_results()

            # Progress update
            elapsed = time.time() - self.total_start_time
            folds_done = i + 1
            avg_per_fold = elapsed / folds_done
            remaining = avg_per_fold * (len(folds_to_train) - i - 1)
            print(
                f"\n  Progress: {folds_done}/{len(folds_to_train)} folds complete (fold {fold_id}) | "
                f"Elapsed: {elapsed / 60:.1f}m | Est. remaining: {remaining / 60:.1f}m"
            )

        # Aggregate metrics
        aggregate = self._compute_aggregate_metrics()

        if verbose:
            self._print_aggregate_results(aggregate)

        return aggregate

    def _compute_aggregate_metrics(self) -> AggregateMetrics:
        """Compute aggregate metrics across folds."""
        r2_values = [m.val_r2 for m in self.fold_metrics]
        pearson_values = [m.val_pearson for m in self.fold_metrics]
        spearman_values = [m.val_spearman for m in self.fold_metrics]
        mae_values = [m.val_mae for m in self.fold_metrics]
        rmse_values = [m.val_rmse for m in self.fold_metrics]

        per_dim_mean_r2 = {}
        per_dim_std_r2 = {}

        for dim in PERCEPIANO_DIMENSIONS:
            dim_values = [m.per_dim_r2.get(dim, 0.0) for m in self.fold_metrics]
            per_dim_mean_r2[dim] = np.mean(dim_values)
            per_dim_std_r2[dim] = np.std(dim_values)

        total_time = sum(m.training_time_seconds for m in self.fold_metrics)

        return AggregateMetrics(
            mean_r2=np.mean(r2_values),
            std_r2=np.std(r2_values),
            mean_pearson=np.mean(pearson_values),
            std_pearson=np.std(pearson_values),
            mean_spearman=np.mean(spearman_values),
            std_spearman=np.std(spearman_values),
            mean_mae=np.mean(mae_values),
            std_mae=np.std(mae_values),
            mean_rmse=np.mean(rmse_values),
            std_rmse=np.std(rmse_values),
            per_dim_mean_r2=per_dim_mean_r2,
            per_dim_std_r2=per_dim_std_r2,
            total_training_time=total_time,
        )

    def _print_aggregate_results(self, agg: AggregateMetrics) -> None:
        """Print aggregate results."""
        print(f"\n{'=' * 70}")
        print(f"  AGGREGATE RESULTS ({self.n_folds}-Fold Cross-Validation)")
        print(f"{'=' * 70}")
        print(f"  Total training time: {agg.total_training_time / 60:.1f} minutes")
        print()
        print(f"  Overall Metrics (mean +/- std):")
        print(f"    R2:       {agg.mean_r2:+.4f} +/- {agg.std_r2:.4f}")
        print(f"    Pearson:  {agg.mean_pearson:+.4f} +/- {agg.std_pearson:.4f}")
        print(f"    Spearman: {agg.mean_spearman:+.4f} +/- {agg.std_spearman:.4f}")
        print(f"    MAE:      {agg.mean_mae:.4f} +/- {agg.std_mae:.4f}")
        print(f"    RMSE:     {agg.mean_rmse:.4f} +/- {agg.std_rmse:.4f}")

        # Per-fold summary table
        print()
        print(f"  Per-Fold Results:")
        print(f"    {'Fold':<6} {'R2':>10} {'Pearson':>10} {'MAE':>10} {'Epochs':>8}")
        print(f"    {'-' * 6} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 8}")
        for m in self.fold_metrics:
            print(
                f"    {m.fold_id:<6} {m.val_r2:>+10.4f} {m.val_pearson:>+10.4f} "
                f"{m.val_mae:>10.4f} {m.epochs_trained:>8}"
            )
        print(f"    {'-' * 6} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 8}")
        print(
            f"    {'Mean':<6} {agg.mean_r2:>+10.4f} {agg.mean_pearson:>+10.4f} "
            f"{agg.mean_mae:>10.4f}"
        )
        print(
            f"    {'Std':<6} {agg.std_r2:>10.4f} {agg.std_pearson:>10.4f} "
            f"{agg.std_mae:>10.4f}"
        )

        # Dimension analysis
        sorted_dims = sorted(
            agg.per_dim_mean_r2.items(), key=lambda x: x[1], reverse=True
        )
        positive_dims = sum(1 for _, r2 in sorted_dims if r2 > 0)
        strong_dims = sum(1 for _, r2 in sorted_dims if r2 >= 0.2)

        print()
        print(
            f"  Dimension Analysis: {positive_dims}/19 positive R2, {strong_dims}/19 strong (>=0.2)"
        )
        print(f"    Best:  {sorted_dims[0][0]} (R2={sorted_dims[0][1]:+.4f})")
        print(f"    Worst: {sorted_dims[-1][0]} (R2={sorted_dims[-1][1]:+.4f})")

        # Baseline comparison
        print()
        print(f"  Baseline Comparison:")
        print(f"    Bi-LSTM (published):  R2 = 0.185")
        print(f"    MidiBERT (published): R2 = 0.313")
        print(f"    HAN SOTA (published): R2 = 0.397")
        print(f"    Ours:                 R2 = {agg.mean_r2:.3f} +/- {agg.std_r2:.3f}")

        # Interpretation
        print()
        if agg.mean_r2 >= 0.35:
            print(f"  --> EXCELLENT: Matching published SOTA!")
        elif agg.mean_r2 >= 0.25:
            print(f"  --> GOOD: Usable for pseudo-labeling MAESTRO")
        elif agg.mean_r2 >= 0.10:
            print(f"  --> FAIR: Model is learning, needs improvement")
        elif agg.mean_r2 >= 0:
            print(f"  --> WEAK: Barely better than mean prediction")
        else:
            print(f"  --> FAILED: Worse than predicting the mean")

    def evaluate_on_test(
        self,
        normalization_stats: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate all fold models on held-out test set."""
        print(f"\n{'=' * 70}")
        print(f"  TEST SET EVALUATION")
        print(f"{'=' * 70}")

        # Create test dataset
        test_dataset = PercePianoTestDataset(
            data_dir=self.data_dir,
            fold_assignments=self.fold_assignments,
            max_notes=self.config.get("max_notes", 1024),
            normalization_stats=normalization_stats,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.get("batch_size", 32),
            shuffle=False,
            num_workers=0,
        )

        print(f"  Test samples: {len(test_dataset)}")

        # Evaluate each fold
        fold_test_metrics = []
        all_fold_preds = []
        targets = None

        print()
        print(f"  Per-Fold Test Results:")
        print(f"    {'Fold':<6} {'R2':>10} {'Pearson':>10} {'MAE':>10}")
        print(f"    {'-' * 6} {'-' * 10} {'-' * 10} {'-' * 10}")

        for fold_id, checkpoint_path in enumerate(self.fold_checkpoints):
            model = PercePianoVNetModule.load_from_checkpoint(
                str(checkpoint_path),
                input_size=self.config.get("input_size", 78),  # SOTA: 78 features
                hidden_size=self.config.get("hidden_size", 256),
                note_layers=self.config.get("note_layers", 2),
                voice_layers=self.config.get("voice_layers", 2),
                beat_layers=self.config.get("beat_layers", 2),
                measure_layers=self.config.get("measure_layers", 1),
                num_attention_heads=self.config.get("num_attention_heads", 8),
                final_hidden=self.config.get("final_hidden", 128),
                dropout=self.config.get("dropout", 0.2),
            )

            model.eval()
            device = next(model.parameters()).device

            preds = []
            tgts = []

            with torch.no_grad():
                for batch in test_loader:
                    input_features = batch["input_features"].to(device)
                    note_locations = {
                        "beat": batch["note_locations_beat"].to(device),
                        "measure": batch["note_locations_measure"].to(device),
                        "voice": batch["note_locations_voice"].to(device),
                    }

                    outputs = model(input_features, note_locations)
                    preds.append(outputs["predictions"].cpu().numpy())
                    tgts.append(batch["scores"].numpy())

            preds = np.concatenate(preds, axis=0)
            tgts = np.concatenate(tgts, axis=0)
            all_fold_preds.append(preds)
            targets = tgts

            r2 = r2_score(targets, preds)
            pearson = pearsonr(targets.flatten(), preds.flatten())[0]
            mae = np.mean(np.abs(preds - targets))

            fold_test_metrics.append(
                {
                    "fold_id": fold_id,
                    "r2": r2,
                    "pearson": pearson,
                    "mae": mae,
                    "predictions": preds,
                }
            )

            print(f"    {fold_id:<6} {r2:>+10.4f} {pearson:>+10.4f} {mae:>10.4f}")

        # Ensemble
        ensemble_preds = np.mean(all_fold_preds, axis=0)
        ensemble_r2 = r2_score(targets, ensemble_preds)
        ensemble_pearson = pearsonr(targets.flatten(), ensemble_preds.flatten())[0]
        ensemble_spearman = spearmanr(targets.flatten(), ensemble_preds.flatten())[0]
        ensemble_mae = np.mean(np.abs(ensemble_preds - targets))
        ensemble_rmse = np.sqrt(np.mean((ensemble_preds - targets) ** 2))

        print(f"    {'-' * 6} {'-' * 10} {'-' * 10} {'-' * 10}")
        print(
            f"    {'Ens.':<6} {ensemble_r2:>+10.4f} {ensemble_pearson:>+10.4f} {ensemble_mae:>10.4f}"
        )

        # Per-dimension test results
        per_dim_r2 = {}
        for i, dim in enumerate(PERCEPIANO_DIMENSIONS):
            per_dim_r2[dim] = r2_score(targets[:, i], ensemble_preds[:, i])

        sorted_dims = sorted(per_dim_r2.items(), key=lambda x: x[1], reverse=True)
        positive = sum(1 for _, r2 in sorted_dims if r2 > 0)

        print()
        print(f"  Ensemble Test Metrics:")
        print(f"    R2:       {ensemble_r2:+.4f}")
        print(f"    Pearson:  {ensemble_pearson:+.4f}")
        print(f"    Spearman: {ensemble_spearman:+.4f}")
        print(f"    MAE:      {ensemble_mae:.4f}")
        print(f"    RMSE:     {ensemble_rmse:.4f}")
        print(f"    Positive dimensions: {positive}/19")

        # Check for improvement from ensemble
        mean_fold_r2 = np.mean([m["r2"] for m in fold_test_metrics])
        ensemble_gain = ensemble_r2 - mean_fold_r2
        print()
        print(f"  Ensemble Gain: {ensemble_gain:+.4f} R2 over mean of individual folds")

        return {
            "fold_metrics": fold_test_metrics,
            "ensemble": {
                "r2": ensemble_r2,
                "pearson": ensemble_pearson,
                "spearman": ensemble_spearman,
                "mae": ensemble_mae,
                "rmse": ensemble_rmse,
                "per_dim_r2": per_dim_r2,
                "predictions": ensemble_preds,
                "targets": targets,
            },
        }

    def save_results(self, output_path: Optional[Union[str, Path]] = None) -> None:
        """Save training results to JSON."""
        output_path = output_path or self.checkpoint_dir / "kfold_results.json"
        output_path = Path(output_path)

        results = {
            "config": {k: v for k, v in self.config.items() if not callable(v)},
            "n_folds": self.n_folds,
            "fold_metrics": [
                {
                    "fold_id": m.fold_id,
                    "train_loss": float(m.train_loss),
                    "val_loss": float(m.val_loss),
                    "val_r2": float(m.val_r2),
                    "val_pearson": float(m.val_pearson),
                    "val_spearman": float(m.val_spearman),
                    "val_mae": float(m.val_mae),
                    "val_rmse": float(m.val_rmse),
                    "epochs_trained": m.epochs_trained,
                    "best_epoch": m.best_epoch,
                    "training_time_seconds": m.training_time_seconds,
                    "n_train_samples": m.n_train_samples,
                    "n_val_samples": m.n_val_samples,
                    "per_dim_r2": {k: float(v) for k, v in m.per_dim_r2.items()},
                    "per_dim_pearson": {k: float(v) for k, v in m.per_dim_pearson.items()},
                }
                for m in self.fold_metrics
            ],
            "fold_checkpoints": [str(p) for p in self.fold_checkpoints],
        }

        if self.fold_metrics:
            agg = self._compute_aggregate_metrics()
            results["aggregate"] = {
                "mean_r2": float(agg.mean_r2),
                "std_r2": float(agg.std_r2),
                "mean_pearson": float(agg.mean_pearson),
                "std_pearson": float(agg.std_pearson),
                "mean_spearman": float(agg.mean_spearman),
                "std_spearman": float(agg.std_spearman),
                "mean_mae": float(agg.mean_mae),
                "std_mae": float(agg.std_mae),
                "mean_rmse": float(agg.mean_rmse),
                "std_rmse": float(agg.std_rmse),
                "total_training_time_seconds": float(agg.total_training_time),
                "per_dim_mean_r2": {
                    k: float(v) for k, v in agg.per_dim_mean_r2.items()
                },
            }

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n  Results saved to {output_path}")
