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
from ..models.percepiano_replica import (
    PERCEPIANO_DIMENSIONS,
    PercePianoVNetModule,
    PercePianoBiLSTMBaseline,
    PercePianoBaselinePlusBeat,
    PercePianoBaselinePlusBeatMeasure,
)
from .diagnostics import DiagnosticCallback, HierarchyAblationCallback

# Model type constants
MODEL_TYPE_HAN = "han"
MODEL_TYPE_BASELINE = "baseline"
MODEL_TYPE_BASELINE_BEAT = "baseline_beat"
MODEL_TYPE_BASELINE_BEAT_MEASURE = "baseline_beat_measure"


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
    DIAGNOSTIC: Gradient monitoring for debugging training issues.

    Logs gradient norms by layer category at regular intervals.
    Focus on detecting:
    1. Gradient imbalance (prediction_head doing all the work)
    2. Context vector learning (critical for Round 3 fix validation)
    """

    def __init__(self, log_every_n_steps: int = 100, verbose_first_n_steps: int = 5):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.verbose_first_n_steps = verbose_first_n_steps

    def on_after_backward(self, trainer, pl_module):
        step = trainer.global_step
        is_verbose_step = step < self.verbose_first_n_steps
        is_log_step = step % self.log_every_n_steps == 0

        if not (is_verbose_step or is_log_step):
            return

        # Collect gradient norms by category
        grad_stats = {
            "han_encoder": {"norm": 0.0, "count": 0},
            "performance_contractor": {"norm": 0.0, "count": 0},
            "final_attention": {"norm": 0.0, "count": 0},
            "prediction_head": {"norm": 0.0, "count": 0},
        }

        # Track context vector gradients specifically (Round 3 fix validation)
        context_vector_norm = 0.0
        context_vector_count = 0

        total_norm_sq = 0.0

        for name, param in pl_module.named_parameters():
            if param.grad is None:
                continue

            param_norm = param.grad.data.norm(2).item()
            total_norm_sq += param_norm**2

            # Track context vectors specifically (critical for Round 3)
            if "context_vector" in name:
                context_vector_norm += param_norm**2
                context_vector_count += 1

            # Categorize - handle both HAN and Bi-LSTM naming conventions
            if "han_encoder" in name or "lstm" in name:
                cat = "han_encoder"
            elif "performance_contractor" in name or "note_contractor" in name:
                cat = "performance_contractor"
            elif "final_attention" in name or "note_attention" in name:
                cat = "final_attention"
            elif "prediction_head" in name or "out_fc" in name:
                cat = "prediction_head"
            else:
                continue  # Skip 'other' category (embedder, etc.)

            grad_stats[cat]["norm"] += param_norm**2
            grad_stats[cat]["count"] += 1

        total_norm = total_norm_sq**0.5
        context_vector_norm = context_vector_norm**0.5

        # Compute category norms
        for cat in grad_stats:
            if grad_stats[cat]["count"] > 0:
                grad_stats[cat]["norm"] = grad_stats[cat]["norm"]**0.5

        # Get current loss
        current_loss = trainer.callback_metrics.get("train/loss", float("nan"))
        if hasattr(current_loss, "item"):
            current_loss = current_loss.item()

        # Compute gradient balance ratio (prediction_head vs rest)
        pred_head_norm = grad_stats["prediction_head"]["norm"]
        encoder_norm = grad_stats["han_encoder"]["norm"] + grad_stats["performance_contractor"]["norm"]
        balance_ratio = pred_head_norm / (encoder_norm + 1e-8)

        # Print compact diagnostics
        print(f"\n  [GRAD] Step {step}: total={total_norm:.3f}, loss={current_loss:.6f}")
        print(f"    han={grad_stats['han_encoder']['norm']:.3f}, "
              f"contractor={grad_stats['performance_contractor']['norm']:.3f}, "
              f"attn={grad_stats['final_attention']['norm']:.4f}, "
              f"head={grad_stats['prediction_head']['norm']:.3f}")

        # Log context vector gradient (critical for Round 3 validation)
        if context_vector_count > 0 and is_verbose_step:
            if context_vector_norm < 1e-6:
                print(f"    context_vectors={context_vector_norm:.6f} [WARN: not learning!]")
            else:
                print(f"    context_vectors={context_vector_norm:.4f} [OK: learning]")

        print(f"    Balance (head/encoder): {balance_ratio:.1f}x", end="")

        # Flag issues
        if balance_ratio > 10:
            print(" [IMBALANCED - head doing all work]")
        elif total_norm > 100:
            print(" [EXPLOSION]")
        elif total_norm < 0.001:
            print(" [VANISHING]")
        elif not np.isfinite(total_norm):
            print(" [NaN/Inf DETECTED]")
        else:
            print(" [OK]")


class SliceRegenerationCallback(Callback):
    """
    Regenerates slice indices at the start of each training epoch.

    This is critical for SOTA performance - the original PercePiano calls
    update_slice_info() at the start of each epoch to create new overlapping
    slices, providing data augmentation through varied slice boundaries.

    Without this callback, the same slices would be used every epoch,
    reducing the effective training data diversity.
    """

    def on_train_epoch_start(self, trainer, pl_module):
        """Regenerate slice indices at the start of each training epoch."""
        # Access the training dataset through the datamodule
        if hasattr(trainer, 'datamodule') and trainer.datamodule is not None:
            dataset = trainer.datamodule.train_dataset
        elif hasattr(trainer, 'train_dataloader'):
            # Fallback: access through dataloader
            dataloader = trainer.train_dataloader
            if callable(dataloader):
                dataloader = dataloader()
            dataset = dataloader.dataset
        else:
            return

        # Call update_slice_info if the method exists
        if hasattr(dataset, 'update_slice_info'):
            old_count = len(dataset.slice_info) if hasattr(dataset, 'slice_info') else 0
            dataset.update_slice_info()
            new_count = len(dataset.slice_info) if hasattr(dataset, 'slice_info') else 0
            # Only log on first epoch or if count changed significantly
            if trainer.current_epoch == 0:
                print(f"  [Slice] Epoch {trainer.current_epoch}: regenerated {new_count} slices")


class ActivationDiagnosticCallback(Callback):
    """
    DIAGNOSTIC: Check key activation statistics on first batch only.

    Focus on the critical metrics for detecting issues:
    1. Model architecture validation (Round 5: prediction head size)
    2. Logits std (should be 0.5-1.5 for proper sigmoid spread)
    3. Prediction std (should be 0.10-0.15 to match target std ~0.106)
    4. Pred/Target std ratio (should be 0.8-1.5, not >2.0 like Round 3)
    5. Per-dimension std (detect collapsed dimensions)
    6. Context vectors present (Round 3 fix validation)
    """

    def __init__(self):
        super().__init__()
        self.checked = False

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.checked:
            return

        self.checked = True
        print(f"\n{'='*60}")
        print(f"  ACTIVATION CHECK - Batch 0")
        print(f"{'='*60}")

        # Model architecture check (Round 5 validation)
        total_params = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
        print(f"  Model parameters: {total_params:,}")

        # Check prediction head architecture (Round 6: should be 512->512->19)
        # NOTE: The config's final_fc_size=128 is for DECODER, not classifier!
        # See model_m2pf.py:118-124 for actual implementation.
        if hasattr(pl_module, 'prediction_head'):
            head = pl_module.prediction_head
            # Get layer sizes from Sequential
            layer_sizes = []
            for module in head:
                if hasattr(module, 'in_features'):
                    layer_sizes.append(f"{module.in_features}->{module.out_features}")
            if layer_sizes:
                head_arch = ", ".join(layer_sizes)
                print(f"  Prediction head: {head_arch}")
                # Validate Round 6 fix: 512->512->19 is correct
                if "512->512" in head_arch and "512->19" in head_arch:
                    print(f"    [OK] Prediction head architecture correct (Round 6)")
                elif "512->128" in head_arch:
                    print(f"    [WARN] Prediction head using 128 hidden (should be 512!)")

        # Log learning rate to confirm config
        lr = trainer.optimizers[0].param_groups[0]["lr"]
        print(f"  Learning rate: {lr:.2e}")

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

            # Run forward with diagnose=True to see detailed activation flow
            outputs = pl_module(input_features, note_locations, diagnose=True)
            logits = outputs.get("logits")
            predictions = outputs["predictions"]

            # Key metrics
            target_std = targets.std().item()
            pred_std = predictions.std().item()
            pred_mean = predictions.mean().item()
            pred_min = predictions.min().item()
            pred_max = predictions.max().item()
            std_ratio = pred_std / (target_std + 1e-8)

            print(f"  Targets:     mean={targets.mean().item():.3f}, std={target_std:.3f}")
            print(f"  Predictions: mean={pred_mean:.3f}, std={pred_std:.3f}, range=[{pred_min:.3f}, {pred_max:.3f}]")

            if logits is not None:
                logit_std = logits.std().item()
                logit_mean = logits.mean().item()
                print(f"  Logits:      mean={logit_mean:.3f}, std={logit_std:.3f}")

            print(f"\n  Health Check:")

            # 1. Logits std check (Round 4: expect 0.5-1.5)
            if logits is not None:
                if logit_std < 0.3:
                    print(f"    [WARN] Logits std={logit_std:.3f} too small (target: 0.5-1.5)")
                elif logit_std > 3.0:
                    print(f"    [WARN] Logits std={logit_std:.3f} too large (target: 0.5-1.5)")
                else:
                    print(f"    [OK] Logits std={logit_std:.3f} in good range")

            # 2. Prediction std absolute check
            if pred_std < 0.05:
                print(f"    [FAIL] Prediction collapse! std={pred_std:.3f} (target: 0.10-0.15)")
            elif pred_std < 0.08:
                print(f"    [WARN] Predictions under-spread: std={pred_std:.3f} (target: 0.10-0.15)")
            elif pred_std > 0.25:
                print(f"    [WARN] Predictions over-spread: std={pred_std:.3f} (target: 0.10-0.15)")
            else:
                print(f"    [OK] Prediction std={pred_std:.3f} in good range")

            # 3. Pred/Target std ratio check (Round 4 key metric)
            if std_ratio < 0.5:
                print(f"    [WARN] Pred/target std ratio={std_ratio:.2f}x too low (target: 0.8-1.5x)")
            elif std_ratio > 2.0:
                print(f"    [WARN] Pred/target std ratio={std_ratio:.2f}x too high - overshoot! (target: 0.8-1.5x)")
            else:
                print(f"    [OK] Pred/target std ratio={std_ratio:.2f}x (target: 0.8-1.5x)")

            # 4. Per-dimension std check (detect collapsed dimensions)
            dim_stds = predictions.std(dim=0)  # [19]
            collapsed_dims = (dim_stds < 0.03).sum().item()
            if collapsed_dims > 0:
                print(f"    [WARN] {collapsed_dims}/19 dimensions collapsed (std < 0.03)")
            else:
                print(f"    [OK] All 19 dimensions have healthy variance")

            # 5. Check attention context vectors exist (Round 3 validation)
            has_context_vectors = any(
                "context_vector" in name
                for name, _ in pl_module.named_parameters()
            )
            if has_context_vectors:
                print(f"    [OK] Context vectors present (Round 3 fix active)")
            else:
                print(f"    [WARN] No context vectors found - check final_attention type")

        print(f"{'='*60}\n")


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
        model_type: str = MODEL_TYPE_HAN,
    ):
        """
        Initialize K-Fold trainer.

        Args:
            config: Training configuration dictionary
            fold_assignments: Fold assignments from create_piece_based_folds()
            data_dir: Path to data directory
            checkpoint_dir: Path to save checkpoints
            log_dir: Path to save logs (default: checkpoint_dir/logs)
            n_folds: Number of folds (default: 4)
            model_type: Either MODEL_TYPE_HAN ("han") for full hierarchical model
                       or MODEL_TYPE_BASELINE ("baseline") for Bi-LSTM baseline
        """
        self.config = config
        self.fold_assignments = fold_assignments
        self.data_dir = Path(data_dir)
        self.model_type = model_type

        # Use model-specific subdirectory for checkpoints
        model_suffix = "_baseline" if model_type == MODEL_TYPE_BASELINE else ""
        self.checkpoint_dir = Path(checkpoint_dir) / f"percepiano{model_suffix}"
        self.log_dir = Path(log_dir) / f"percepiano{model_suffix}" if log_dir else self.checkpoint_dir / "logs"
        self.n_folds = n_folds

        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Storage for results
        self.fold_metrics: List[FoldMetrics] = []
        self.fold_checkpoints: List[Path] = []
        self.total_start_time: Optional[float] = None

        # Print model type info
        model_name = "Bi-LSTM Baseline (7-layer)" if model_type == MODEL_TYPE_BASELINE else "HAN (Hierarchical)"
        print(f"  Model type: {model_name}")

    def get_trained_model(self, fold_id: int) -> Optional[PercePianoVNetModule]:
        """
        Get the trained model for a specific fold.

        Models are stored in memory after train_fold() completes.
        This is useful for running diagnostics without loading from checkpoint.

        Args:
            fold_id: The fold number

        Returns:
            The trained model if available, None otherwise
        """
        return getattr(self, f"_trained_model_fold_{fold_id}", None)

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

    def _create_model(self) -> pl.LightningModule:
        """Create a new model instance based on model_type."""
        input_size = self.config.get("input_size", 79)
        hidden_size = self.config.get("hidden_size", 256)
        num_attention_heads = self.config.get("num_attention_heads", 8)
        dropout = self.config.get("dropout", 0.2)
        learning_rate = self.config.get("learning_rate", 2.5e-5)
        weight_decay = self.config.get("weight_decay", 1e-5)
        beat_layers = self.config.get("beat_layers", 2)
        measure_layers = self.config.get("measure_layers", 1)

        if self.model_type == MODEL_TYPE_BASELINE:
            # Bi-LSTM Baseline: 7-layer single LSTM (matches VirtuosoNetSingle)
            return PercePianoBiLSTMBaseline(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=7,  # Fixed: matches original VirtuosoNetSingle
                num_attention_heads=num_attention_heads,
                dropout=dropout,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
            )
        elif self.model_type == MODEL_TYPE_BASELINE_BEAT:
            # Baseline + Beat: 7-layer LSTM + beat hierarchy (Phase 2 Step 1)
            return PercePianoBaselinePlusBeat(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=7,
                beat_layers=beat_layers,
                num_attention_heads=num_attention_heads,
                dropout=dropout,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
            )
        elif self.model_type == MODEL_TYPE_BASELINE_BEAT_MEASURE:
            # Baseline + Beat + Measure: 7-layer LSTM + beat + measure (Phase 2 Step 2)
            return PercePianoBaselinePlusBeatMeasure(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=7,
                beat_layers=beat_layers,
                measure_layers=measure_layers,
                num_attention_heads=num_attention_heads,
                dropout=dropout,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
            )
        else:
            # HAN: Full hierarchical model (matches VirtuosoNetMultiLevel)
            return PercePianoVNetModule(
                input_size=input_size,
                hidden_size=hidden_size,
                note_layers=self.config.get("note_layers", 2),
                voice_layers=self.config.get("voice_layers", 2),
                beat_layers=beat_layers,
                measure_layers=measure_layers,
                num_attention_heads=num_attention_heads,
                dropout=dropout,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
            )

    def _create_callbacks(self, fold_id: int) -> List[pl.Callback]:
        """Create callbacks for training."""
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.checkpoint_dir / f"fold_{fold_id}",
            filename="best-epoch={epoch:02d}-r2={val/mean_r2:.4f}",
            monitor="val/mean_r2",
            mode="max",
            save_top_k=1,
            save_last=True,
            auto_insert_metric_name=False,  # Prevent Lightning from auto-inserting metric name
        )

        early_stopping = EarlyStopping(
            monitor="val/mean_r2",
            mode="max",
            patience=self.config.get("early_stopping_patience", 20),
            verbose=False,  # We have our own logging
        )

        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        epoch_logger = EpochLogger(fold_id=fold_id)

        # SOTA: Slice regeneration each epoch for data augmentation
        slice_regen = SliceRegenerationCallback()

        # DIAGNOSTIC: Gradient monitoring (logs every 100 steps, verbose first 5)
        grad_monitor = GradientMonitorCallback(log_every_n_steps=100, verbose_first_n_steps=5)

        # DIAGNOSTIC: Activation check on first batch
        activation_diag = ActivationDiagnosticCallback()

        # Base callbacks for both model types
        callbacks = [
            checkpoint_callback,
            early_stopping,
            lr_monitor,
            epoch_logger,
            slice_regen,
            grad_monitor,
            activation_diag,
        ]

        # Add hierarchy-specific callbacks for models with hierarchy
        # (HAN and Phase 2 incremental models)
        has_hierarchy = self.model_type in [
            MODEL_TYPE_HAN,
            MODEL_TYPE_BASELINE_BEAT,
            MODEL_TYPE_BASELINE_BEAT_MEASURE,
        ]

        if has_hierarchy:
            # DIAGNOSTIC: Comprehensive hierarchy diagnostics (every 5 epochs)
            # Captures activation variances, attention entropy, contribution analysis
            hierarchy_diag = DiagnosticCallback(
                log_every_n_steps=200,
                detailed_analysis_every_n_epochs=5,
                save_dir=self.checkpoint_dir / f"fold_{fold_id}" / "diagnostics",
            )
            callbacks.append(hierarchy_diag)

            # DIAGNOSTIC: Ablation test (only for full HAN, not incremental models)
            # For incremental models, we compare directly to baseline checkpoint
            if self.model_type == MODEL_TYPE_HAN:
                ablation_callback = HierarchyAblationCallback(run_every_n_epochs=10)
                callbacks.append(ablation_callback)

        return callbacks

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
        # SOTA uses slice sampling with slice_len=5000
        # num_workers=4 is optimal for A100XL (4 vCPUs) - avoids dataloader bottleneck
        data_module = PercePianoKFoldDataModule(
            data_dir=self.data_dir,
            fold_assignments=self.fold_assignments,
            fold_id=fold_id,
            batch_size=self.config.get("batch_size", 8),  # SOTA: 8
            max_notes=self.config.get("max_notes", 5000),  # SOTA: 5000
            num_workers=self.config.get("num_workers", 4),  # 4 for A100XL vCPUs
            augment_train=self.config.get(
                "augment_train", False
            ),  # SOTA: no augmentation
            slice_len=self.config.get("slice_len", None),  # SOTA: 5000
        )
        data_module.setup("fit")

        n_train = len(data_module.train_dataset)
        n_val = len(data_module.val_dataset)

        if verbose:
            print(f"\n{'=' * 70}")
            print(f"FOLD {fold_id}/{self.n_folds - 1}")
            print(f"{'=' * 70}")
            print(f"  Train slices: {n_train} | Val slices: {n_val}")
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
        # Use the correct model class based on model_type
        if self.model_type == MODEL_TYPE_BASELINE:
            best_model = PercePianoBiLSTMBaseline.load_from_checkpoint(
                str(best_checkpoint),
                input_size=self.config.get("input_size", 79),
                hidden_size=self.config.get("hidden_size", 256),
                num_layers=7,
                num_attention_heads=self.config.get("num_attention_heads", 8),
                dropout=self.config.get("dropout", 0.2),
                learning_rate=self.config.get("learning_rate", 2.5e-5),
                weight_decay=self.config.get("weight_decay", 1e-5),
            )
        elif self.model_type == MODEL_TYPE_BASELINE_BEAT:
            best_model = PercePianoBaselinePlusBeat.load_from_checkpoint(
                str(best_checkpoint),
                input_size=self.config.get("input_size", 79),
                hidden_size=self.config.get("hidden_size", 256),
                num_layers=7,
                beat_layers=self.config.get("beat_layers", 2),
                num_attention_heads=self.config.get("num_attention_heads", 8),
                dropout=self.config.get("dropout", 0.2),
                learning_rate=self.config.get("learning_rate", 2.5e-5),
                weight_decay=self.config.get("weight_decay", 1e-5),
            )
        elif self.model_type == MODEL_TYPE_BASELINE_BEAT_MEASURE:
            best_model = PercePianoBaselinePlusBeatMeasure.load_from_checkpoint(
                str(best_checkpoint),
                input_size=self.config.get("input_size", 79),
                hidden_size=self.config.get("hidden_size", 256),
                num_layers=7,
                beat_layers=self.config.get("beat_layers", 2),
                measure_layers=self.config.get("measure_layers", 1),
                num_attention_heads=self.config.get("num_attention_heads", 8),
                dropout=self.config.get("dropout", 0.2),
                learning_rate=self.config.get("learning_rate", 2.5e-5),
                weight_decay=self.config.get("weight_decay", 1e-5),
            )
        else:  # MODEL_TYPE_HAN
            best_model = PercePianoVNetModule.load_from_checkpoint(
                str(best_checkpoint),
                input_size=self.config.get("input_size", 79),  # SOTA: 79 features
                hidden_size=self.config.get("hidden_size", 256),
                note_layers=self.config.get("note_layers", 2),
                voice_layers=self.config.get("voice_layers", 2),
                beat_layers=self.config.get("beat_layers", 2),
                measure_layers=self.config.get("measure_layers", 1),
                num_attention_heads=self.config.get("num_attention_heads", 8),
                dropout=self.config.get("dropout", 0.2),
            )

        # Store trained model for later retrieval (e.g., for diagnostics)
        setattr(self, f"_trained_model_fold_{fold_id}", best_model)

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
                input_size=self.config.get("input_size", 79),  # SOTA: 79 features
                hidden_size=self.config.get("hidden_size", 256),
                note_layers=self.config.get("note_layers", 2),
                voice_layers=self.config.get("voice_layers", 2),
                beat_layers=self.config.get("beat_layers", 2),
                measure_layers=self.config.get("measure_layers", 1),
                num_attention_heads=self.config.get("num_attention_heads", 8),
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
