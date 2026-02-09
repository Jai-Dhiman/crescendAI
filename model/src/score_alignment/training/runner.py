"""Training orchestration for score alignment experiments.

Provides experiment runners for training alignment projection models
and evaluating with DTW-based metrics.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from tqdm import tqdm
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader


class EpochProgressCallback(Callback):
    """Print epoch-level progress to stdout (works in all Jupyter environments)."""

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        train_loss = metrics.get("train_loss", float("nan"))
        val_loss = metrics.get("val_loss", float("nan"))
        print(
            f"Epoch {epoch:3d}/{trainer.max_epochs} | "
            f"train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}"
        )

from ..alignment.dtw import align_embeddings
from ..alignment.metrics import compute_alignment_summary, evaluate_dtw_alignment
from ..config import ExperimentConfig, ProjectionConfig, TrainingConfig
from ..data.alignment_dataset import (
    FrameAlignmentDataset,
    MeasureAlignmentDataset,
    frame_alignment_collate_fn,
    measure_alignment_collate_fn,
)
from ..data.asap import ASAPDatasetIndex, ASAPPerformance, get_performance_key
from ..models.projection import AlignmentProjectionModel


def load_existing_results(exp_id: str, results_dir: Path) -> Optional[Dict]:
    """Load existing results from JSON if available."""
    results_file = Path(results_dir) / f"{exp_id}.json"
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)
    return None


def experiment_completed(exp_id: str, checkpoint_dir: Path) -> bool:
    """Check if experiment checkpoint exists."""
    exp_dir = Path(checkpoint_dir) / exp_id
    return exp_dir.exists() and (exp_dir / "best.ckpt").exists()


def run_alignment_experiment(
    exp_id: str,
    description: str,
    performances: List[ASAPPerformance],
    score_cache_dir: Path,
    perf_cache_dir: Path,
    asap_root: Path,
    train_keys: List[str],
    val_keys: List[str],
    config: Optional[ExperimentConfig] = None,
    projection_config: Optional[ProjectionConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    checkpoint_dir: Optional[Path] = None,
    results_dir: Optional[Path] = None,
    log_dir: Optional[Path] = None,
    use_measures: bool = False,
) -> Dict:
    """Run a score alignment training experiment.

    Args:
        exp_id: Experiment identifier.
        description: Human-readable description.
        performances: List of ASAPPerformance objects.
        score_cache_dir: Directory with cached score MuQ embeddings.
        perf_cache_dir: Directory with cached performance MuQ embeddings.
        asap_root: Root directory of ASAP dataset.
        train_keys: Performance keys for training.
        val_keys: Performance keys for validation.
        config: Full experiment configuration (overrides other configs).
        projection_config: Projection network configuration.
        training_config: Training configuration.
        checkpoint_dir: Directory for model checkpoints.
        results_dir: Directory for results JSON.
        log_dir: Directory for training logs.
        use_measures: If True, use measure-level alignment instead of frame-level.

    Returns:
        Results dictionary with training metrics and evaluation results.
    """
    # Use config if provided, otherwise use individual configs
    if config:
        projection_config = config.projection
        training_config = config.training
        checkpoint_dir = config.checkpoint_dir or checkpoint_dir
        results_dir = config.results_dir or results_dir
        log_dir = config.log_dir or log_dir

    # Use defaults if not provided
    if projection_config is None:
        projection_config = ProjectionConfig()
    if training_config is None:
        training_config = TrainingConfig()

    # Ensure directories exist
    if checkpoint_dir:
        checkpoint_dir = Path(checkpoint_dir)
        exp_checkpoint_dir = checkpoint_dir / exp_id
        exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    else:
        exp_checkpoint_dir = None

    if results_dir:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

    if log_dir:
        log_dir = Path(log_dir)

    # Check if already completed
    if results_dir and exp_checkpoint_dir:
        existing = load_existing_results(exp_id, results_dir)
        if existing and experiment_completed(exp_id, checkpoint_dir):
            print(f"SKIP {exp_id}: already completed")
            return existing

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {exp_id}")
    print(f"Description: {description}")
    print(f"{'='*70}")

    start_time = time.time()

    # Filter performances by train/val keys
    key_to_perf = {get_performance_key(p): p for p in performances}
    train_perfs = [key_to_perf[k] for k in train_keys if k in key_to_perf]
    val_perfs = [key_to_perf[k] for k in val_keys if k in key_to_perf]

    print(f"Train performances: {len(train_perfs)}")
    print(f"Validation performances: {len(val_perfs)}")

    # Create datasets
    if use_measures:
        train_ds = MeasureAlignmentDataset(
            train_perfs, score_cache_dir, perf_cache_dir, asap_root
        )
        val_ds = MeasureAlignmentDataset(
            val_perfs, score_cache_dir, perf_cache_dir, asap_root
        )
        collate_fn = measure_alignment_collate_fn
    else:
        train_ds = FrameAlignmentDataset(
            train_perfs,
            score_cache_dir,
            perf_cache_dir,
            asap_root,
            max_frames=training_config.max_frames,
        )
        val_ds = FrameAlignmentDataset(
            val_perfs,
            score_cache_dir,
            perf_cache_dir,
            asap_root,
            max_frames=training_config.max_frames,
        )
        collate_fn = frame_alignment_collate_fn

    if len(train_ds) == 0 or len(val_ds) == 0:
        print("ERROR: No valid samples in dataset")
        return {"error": "No valid samples", "exp_id": exp_id}

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # Create dataloaders
    train_dl = DataLoader(
        train_ds,
        batch_size=training_config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=training_config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=training_config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=training_config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Create model
    model = AlignmentProjectionModel(
        projection_config=projection_config,
        training_config=training_config,
    )

    # Callbacks
    callbacks = []
    ckpt_path = None

    if exp_checkpoint_dir:
        ckpt_callback = ModelCheckpoint(
            dirpath=exp_checkpoint_dir,
            filename="best",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        )
        callbacks.append(ckpt_callback)
        ckpt_path = exp_checkpoint_dir / "best.ckpt"

    callbacks.append(
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=training_config.patience,
            verbose=True,
        )
    )
    callbacks.append(EpochProgressCallback())

    # Logger
    logger = None
    if log_dir:
        logger = CSVLogger(save_dir=log_dir, name=exp_id)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=training_config.max_epochs,
        callbacks=callbacks,
        logger=logger,
        accelerator="auto",
        devices=1,
        gradient_clip_val=training_config.gradient_clip_val,
        enable_progress_bar=True,
        deterministic=True,
        log_every_n_steps=10,
    )

    # Train
    trainer.fit(model, train_dl, val_dl)

    # Load best checkpoint
    if ckpt_path and ckpt_path.exists():
        model = AlignmentProjectionModel.load_from_checkpoint(ckpt_path)

    # Evaluate with DTW on validation set
    print("\nEvaluating alignment quality on validation set...")
    eval_metrics = evaluate_alignment_model(
        model,
        val_ds,
        score_cache_dir,
        perf_cache_dir,
        asap_root,
    )

    # Compile results
    training_time = time.time() - start_time

    results = {
        "exp_id": exp_id,
        "description": description,
        "config": {
            "projection": {
                "input_dim": projection_config.input_dim,
                "hidden_dim": projection_config.hidden_dim,
                "output_dim": projection_config.output_dim,
                "num_layers": projection_config.num_layers,
            },
            "training": {
                "learning_rate": training_config.learning_rate,
                "batch_size": training_config.batch_size,
                "max_epochs": training_config.max_epochs,
            },
        },
        "metrics": eval_metrics,
        "training_time_seconds": training_time,
        "num_train_samples": len(train_ds),
        "num_val_samples": len(val_ds),
    }

    # Save results
    if results_dir:
        with open(results_dir / f"{exp_id}.json", "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n{exp_id} COMPLETE")
    print(f"Mean onset error: {eval_metrics.get('weighted_mean_error_ms', 0):.1f} ms")
    print(
        f"Within 30ms: {eval_metrics.get('weighted_percent_within_threshold', 0):.1f}%"
    )

    return results


def evaluate_alignment_model(
    model: AlignmentProjectionModel,
    dataset: FrameAlignmentDataset,
    score_cache_dir: Path,
    perf_cache_dir: Path,
    asap_root: Path,
    distance_metric: str = "cosine",
) -> Dict:
    """Evaluate alignment model using DTW on projected embeddings.

    Args:
        model: Trained projection model.
        dataset: Validation dataset.
        score_cache_dir: Directory with score embeddings.
        perf_cache_dir: Directory with performance embeddings.
        asap_root: ASAP dataset root.
        distance_metric: Distance metric for DTW.

    Returns:
        Aggregated alignment metrics.
    """
    model.eval()
    device = next(model.parameters()).device

    all_metrics = []

    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Evaluating alignment"):
            sample = dataset[i]

            # Get embeddings
            score_emb = sample["score_embeddings"].unsqueeze(0).to(device)
            perf_emb = sample["perf_embeddings"].unsqueeze(0).to(device)

            # Project embeddings
            proj_score, proj_perf = model(score_emb, perf_emb)

            # Convert to numpy for DTW
            proj_score_np = proj_score.squeeze(0).cpu().numpy()
            proj_perf_np = proj_perf.squeeze(0).cpu().numpy()

            # Run DTW alignment
            path_score, path_perf, cost, _ = align_embeddings(
                proj_score_np, proj_perf_np, metric=distance_metric
            )

            # Get ground truth onsets
            score_onsets = sample["score_onsets"].numpy()
            perf_onsets = sample["perf_onsets"].numpy()

            # Evaluate
            metrics = evaluate_dtw_alignment(
                score_onsets,
                perf_onsets,
                path_score,
                path_perf,
            )
            all_metrics.append(metrics)

    # Aggregate
    return compute_alignment_summary(all_metrics)


def run_dtw_baseline(
    performances: List[ASAPPerformance],
    score_cache_dir: Path,
    perf_cache_dir: Path,
    asap_root: Path,
    keys: List[str],
    distance_metric: str = "cosine",
    sakoe_chiba_radius: Optional[int] = None,
    results_dir: Optional[Path] = None,
    results_key: str = "A_dtw_baseline",
) -> Dict:
    """Run DTW baseline without learned projection.

    Args:
        performances: List of ASAPPerformance objects.
        score_cache_dir: Directory with score embeddings.
        perf_cache_dir: Directory with performance embeddings.
        asap_root: ASAP dataset root.
        keys: Performance keys to evaluate.
        distance_metric: Distance metric for DTW.
        sakoe_chiba_radius: Optional Sakoe-Chiba band constraint.
        results_dir: Directory to cache results JSON. If set and results
            exist, the cached results are returned immediately.
        results_key: Filename stem for the cached results JSON.

    Returns:
        Aggregated alignment metrics.
    """
    # Check for cached results
    if results_dir:
        results_dir = Path(results_dir)
        existing = load_existing_results(results_key, results_dir)
        if existing:
            print(f"SKIP {results_key}: loaded cached results from {results_dir / (results_key + '.json')}")
            return existing

    # Create dataset (just for loading embeddings consistently)
    key_to_perf = {get_performance_key(p): p for p in performances}
    eval_perfs = [key_to_perf[k] for k in keys if k in key_to_perf]

    # No batching for DTW baseline -- use large max_frames to avoid truncation
    dataset = FrameAlignmentDataset(
        eval_perfs, score_cache_dir, perf_cache_dir, asap_root,
        max_frames=40000,
    )

    print(f"Running DTW baseline on {len(dataset)} samples...")

    all_metrics = []

    for i in tqdm(range(len(dataset)), desc="DTW baseline"):
        sample = dataset[i]

        # Get raw embeddings (no projection)
        score_emb = sample["score_embeddings"].numpy()
        perf_emb = sample["perf_embeddings"].numpy()

        # Run DTW
        path_score, path_perf, cost, _ = align_embeddings(
            score_emb, perf_emb, metric=distance_metric
        )

        # Get ground truth
        score_onsets = sample["score_onsets"].numpy()
        perf_onsets = sample["perf_onsets"].numpy()

        # Evaluate
        metrics = evaluate_dtw_alignment(
            score_onsets, perf_onsets, path_score, path_perf
        )
        all_metrics.append(metrics)

    result = compute_alignment_summary(all_metrics)

    # Cache results
    if results_dir:
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / f"{results_key}.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved results to {results_dir / (results_key + '.json')}")

    return result
