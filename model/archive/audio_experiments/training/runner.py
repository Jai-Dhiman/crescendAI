"""Experiment runners for 4-fold cross-validation."""

import json
import time
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from ..data import MERTDataset, MelDataset, StatsDataset, DualEmbeddingDataset
from ..data import mert_collate_fn, mel_collate_fn, stats_collate_fn, dual_collate_fn
from ..models import MelCNNModel, StatsMLPModel
from .metrics import compute_comprehensive_metrics


def experiment_completed(exp_id: str, checkpoint_dir: Path) -> bool:
    """Check if experiment has all fold checkpoints."""
    exp_dir = Path(checkpoint_dir) / exp_id
    if not exp_dir.exists():
        return False
    return all((exp_dir / f"fold{i}_best.ckpt").exists() for i in range(4))


def load_existing_results(exp_id: str, results_dir: Path) -> Optional[Dict]:
    """Load results from JSON if exists."""
    results_file = Path(results_dir) / f"{exp_id}.json"
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)
    return None


def run_4fold_mert_experiment(
    exp_id: str,
    description: str,
    model_factory: Callable,
    mert_cache_dir: Path,
    labels: Dict,
    fold_assignments: Dict,
    config: Dict,
    checkpoint_root: Path,
    results_dir: Path,
    log_dir: Path,
) -> Dict:
    """Run 4-fold CV for MERT-based experiment.

    Args:
        exp_id: Experiment identifier (e.g., "B0_baseline")
        description: Human-readable description
        model_factory: Function that takes config and returns a model
        mert_cache_dir: Directory with cached MERT embeddings
        labels: Dict mapping keys to label arrays
        fold_assignments: Dict with fold assignments
        config: Training configuration
        checkpoint_root: Root directory for checkpoints
        results_dir: Directory to save results JSON
        log_dir: Directory for training logs

    Returns:
        Results dict with summary, fold_results, per_dimension metrics
    """
    exp_checkpoint_dir = Path(checkpoint_root) / exp_id
    exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Check if already done
    existing = load_existing_results(exp_id, results_dir)
    if existing and experiment_completed(exp_id, checkpoint_root):
        print(f"SKIP {exp_id}: already completed (R2={existing['summary']['avg_r2']:.4f})")
        return existing

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {exp_id}")
    print(f"Description: {description}")
    print(f"{'='*70}")

    start_time = time.time()
    fold_results = {}
    all_preds, all_labels = [], []

    for fold in range(config["n_folds"]):
        ckpt_path = exp_checkpoint_dir / f"fold{fold}_best.ckpt"

        # Create datasets
        train_ds = MERTDataset(
            mert_cache_dir, labels, fold_assignments, fold, "train", config["max_frames"]
        )
        val_ds = MERTDataset(
            mert_cache_dir, labels, fold_assignments, fold, "val", config["max_frames"]
        )

        if len(train_ds) == 0 or len(val_ds) == 0:
            print(f"Fold {fold}: No data available, skipping")
            continue

        train_dl = DataLoader(
            train_ds,
            batch_size=config["batch_size"],
            shuffle=True,
            collate_fn=mert_collate_fn,
            num_workers=config["num_workers"],
            pin_memory=True,
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=config["batch_size"],
            shuffle=False,
            collate_fn=mert_collate_fn,
            num_workers=config["num_workers"],
            pin_memory=True,
        )

        if ckpt_path.exists():
            print(f"Fold {fold}: Loading existing checkpoint")
            model = model_factory(config)
            model = model.__class__.load_from_checkpoint(ckpt_path)
        else:
            print(f"Fold {fold}: Training ({len(train_ds)} train, {len(val_ds)} val)")
            model = model_factory(config)

            callbacks = [
                ModelCheckpoint(
                    dirpath=exp_checkpoint_dir,
                    filename=f"fold{fold}_best",
                    monitor="val_r2",
                    mode="max",
                    save_top_k=1,
                ),
                EarlyStopping(
                    monitor="val_r2", mode="max", patience=config["patience"], verbose=True
                ),
            ]

            trainer = pl.Trainer(
                max_epochs=config["max_epochs"],
                callbacks=callbacks,
                logger=CSVLogger(save_dir=log_dir, name=exp_id, version=f"fold{fold}"),
                accelerator="auto",
                devices=1,
                gradient_clip_val=config["gradient_clip_val"],
                enable_progress_bar=True,
                deterministic=True,
                log_every_n_steps=10,
            )

            trainer.fit(model, train_dl, val_dl)
            fold_results[fold] = float(callbacks[0].best_model_score or 0)

            # Reload best
            model = model.__class__.load_from_checkpoint(ckpt_path)

        # Evaluate
        model.eval().to("cuda")
        with torch.no_grad():
            for batch in val_dl:
                pred = model(
                    batch["embeddings"].cuda(),
                    batch["attention_mask"].cuda(),
                    batch.get("lengths"),
                )
                all_preds.append(pred.cpu().numpy())
                all_labels.append(batch["labels"].numpy())

        del model
        torch.cuda.empty_cache()

    # Aggregate results
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    metrics = compute_comprehensive_metrics(all_preds, all_labels)

    # If fold_results is empty (loaded from checkpoints), compute from metrics
    if not fold_results:
        fold_results = {i: metrics["overall_r2"] for i in range(4)}

    avg_r2 = np.mean(list(fold_results.values()))
    std_r2 = np.std(list(fold_results.values()))

    results = {
        "experiment_id": exp_id,
        "description": description,
        "config": config,
        "summary": {
            "avg_r2": float(avg_r2),
            "std_r2": float(std_r2),
            "r2_ci_95": metrics["r2_ci_95"],
            "overall_r2": metrics["overall_r2"],
            "overall_mae": metrics["overall_mae"],
            "dispersion_ratio": metrics["dispersion_ratio"],
        },
        "fold_results": {str(k): float(v) for k, v in fold_results.items()},
        "per_dimension": metrics["per_dimension"],
        "training_time_seconds": time.time() - start_time,
    }

    # Save results
    with open(results_dir / f"{exp_id}.json", "w") as f:
        json.dump(results, f, indent=2)

    print(
        f"\n{exp_id} COMPLETE: R2={avg_r2:.4f} +/- {std_r2:.4f}, "
        f"CI=[{metrics['r2_ci_95'][0]:.4f}, {metrics['r2_ci_95'][1]:.4f}]"
    )

    return results


def run_4fold_mel_experiment(
    exp_id: str,
    description: str,
    mel_cache_dir: Path,
    labels: Dict,
    fold_assignments: Dict,
    config: Dict,
    checkpoint_root: Path,
    results_dir: Path,
    log_dir: Path,
) -> Dict:
    """Run 4-fold CV for Mel-CNN experiment."""
    exp_checkpoint_dir = Path(checkpoint_root) / exp_id
    exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    existing = load_existing_results(exp_id, results_dir)
    if existing and experiment_completed(exp_id, checkpoint_root):
        print(f"SKIP {exp_id}: already completed (R2={existing['summary']['avg_r2']:.4f})")
        return existing

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {exp_id}")
    print(f"{'='*70}")

    start_time = time.time()
    fold_results = {}
    all_preds, all_labels = [], []

    for fold in range(config["n_folds"]):
        ckpt_path = exp_checkpoint_dir / f"fold{fold}_best.ckpt"

        train_ds = MelDataset(mel_cache_dir, labels, fold_assignments, fold, "train")
        val_ds = MelDataset(mel_cache_dir, labels, fold_assignments, fold, "val")

        train_dl = DataLoader(
            train_ds,
            batch_size=config["batch_size"],
            shuffle=True,
            collate_fn=mel_collate_fn,
            num_workers=config["num_workers"],
            pin_memory=True,
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=config["batch_size"],
            shuffle=False,
            collate_fn=mel_collate_fn,
            num_workers=config["num_workers"],
            pin_memory=True,
        )

        if ckpt_path.exists():
            model = MelCNNModel.load_from_checkpoint(ckpt_path)
        else:
            print(f"Fold {fold}: Training")
            model = MelCNNModel(
                hidden_dim=config["hidden_dim"],
                learning_rate=config["learning_rate"],
                weight_decay=config["weight_decay"],
                max_epochs=config["max_epochs"],
            )

            callbacks = [
                ModelCheckpoint(
                    dirpath=exp_checkpoint_dir,
                    filename=f"fold{fold}_best",
                    monitor="val_r2",
                    mode="max",
                    save_top_k=1,
                ),
                EarlyStopping(
                    monitor="val_r2", mode="max", patience=config["patience"], verbose=True
                ),
            ]

            trainer = pl.Trainer(
                max_epochs=config["max_epochs"],
                callbacks=callbacks,
                logger=CSVLogger(save_dir=log_dir, name=exp_id, version=f"fold{fold}"),
                accelerator="auto",
                devices=1,
                gradient_clip_val=config["gradient_clip_val"],
                enable_progress_bar=True,
                deterministic=True,
            )

            trainer.fit(model, train_dl, val_dl)
            fold_results[fold] = float(callbacks[0].best_model_score or 0)
            model = MelCNNModel.load_from_checkpoint(ckpt_path)

        model.eval().to("cuda")
        with torch.no_grad():
            for batch in val_dl:
                pred = model(batch["mel"].cuda())
                all_preds.append(pred.cpu().numpy())
                all_labels.append(batch["labels"].numpy())

        del model
        torch.cuda.empty_cache()

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    metrics = compute_comprehensive_metrics(all_preds, all_labels)

    if not fold_results:
        fold_results = {i: metrics["overall_r2"] for i in range(4)}

    avg_r2 = np.mean(list(fold_results.values()))
    std_r2 = np.std(list(fold_results.values()))

    results = {
        "experiment_id": exp_id,
        "description": description,
        "summary": {
            "avg_r2": float(avg_r2),
            "std_r2": float(std_r2),
            "r2_ci_95": metrics["r2_ci_95"],
            "overall_r2": metrics["overall_r2"],
            "overall_mae": metrics["overall_mae"],
            "dispersion_ratio": metrics["dispersion_ratio"],
        },
        "fold_results": {str(k): float(v) for k, v in fold_results.items()},
        "per_dimension": metrics["per_dimension"],
        "training_time_seconds": time.time() - start_time,
    }

    with open(results_dir / f"{exp_id}.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{exp_id} COMPLETE: R2={avg_r2:.4f}")
    return results


def run_4fold_stats_experiment(
    exp_id: str,
    description: str,
    stats_cache_dir: Path,
    labels: Dict,
    fold_assignments: Dict,
    config: Dict,
    checkpoint_root: Path,
    results_dir: Path,
    log_dir: Path,
) -> Dict:
    """Run 4-fold CV for statistics MLP experiment."""
    exp_checkpoint_dir = Path(checkpoint_root) / exp_id
    exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    existing = load_existing_results(exp_id, results_dir)
    if existing and experiment_completed(exp_id, checkpoint_root):
        print(f"SKIP {exp_id}: already completed (R2={existing['summary']['avg_r2']:.4f})")
        return existing

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {exp_id}")
    print(f"{'='*70}")

    start_time = time.time()
    fold_results = {}
    all_preds, all_labels = [], []

    for fold in range(config["n_folds"]):
        ckpt_path = exp_checkpoint_dir / f"fold{fold}_best.ckpt"

        train_ds = StatsDataset(stats_cache_dir, labels, fold_assignments, fold, "train")
        val_ds = StatsDataset(stats_cache_dir, labels, fold_assignments, fold, "val")

        train_dl = DataLoader(
            train_ds,
            batch_size=config["batch_size"],
            shuffle=True,
            collate_fn=stats_collate_fn,
            num_workers=config["num_workers"],
            pin_memory=True,
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=config["batch_size"],
            shuffle=False,
            collate_fn=stats_collate_fn,
            num_workers=config["num_workers"],
            pin_memory=True,
        )

        if ckpt_path.exists():
            model = StatsMLPModel.load_from_checkpoint(ckpt_path)
        else:
            print(f"Fold {fold}: Training")
            model = StatsMLPModel(
                input_dim=49,
                hidden_dim=256,
                learning_rate=config["learning_rate"],
                weight_decay=config["weight_decay"],
                max_epochs=config["max_epochs"],
            )

            callbacks = [
                ModelCheckpoint(
                    dirpath=exp_checkpoint_dir,
                    filename=f"fold{fold}_best",
                    monitor="val_r2",
                    mode="max",
                    save_top_k=1,
                ),
                EarlyStopping(
                    monitor="val_r2", mode="max", patience=config["patience"], verbose=True
                ),
            ]

            trainer = pl.Trainer(
                max_epochs=config["max_epochs"],
                callbacks=callbacks,
                logger=CSVLogger(save_dir=log_dir, name=exp_id, version=f"fold{fold}"),
                accelerator="auto",
                devices=1,
                gradient_clip_val=config["gradient_clip_val"],
                enable_progress_bar=True,
                deterministic=True,
            )

            trainer.fit(model, train_dl, val_dl)
            fold_results[fold] = float(callbacks[0].best_model_score or 0)
            model = StatsMLPModel.load_from_checkpoint(ckpt_path)

        model.eval().to("cuda")
        with torch.no_grad():
            for batch in val_dl:
                pred = model(batch["features"].cuda())
                all_preds.append(pred.cpu().numpy())
                all_labels.append(batch["labels"].numpy())

        del model
        torch.cuda.empty_cache()

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    metrics = compute_comprehensive_metrics(all_preds, all_labels)

    if not fold_results:
        fold_results = {i: metrics["overall_r2"] for i in range(4)}

    avg_r2 = np.mean(list(fold_results.values()))
    std_r2 = np.std(list(fold_results.values()))

    results = {
        "experiment_id": exp_id,
        "description": description,
        "summary": {
            "avg_r2": float(avg_r2),
            "std_r2": float(std_r2),
            "r2_ci_95": metrics["r2_ci_95"],
            "overall_r2": metrics["overall_r2"],
            "overall_mae": metrics["overall_mae"],
            "dispersion_ratio": metrics["dispersion_ratio"],
        },
        "fold_results": {str(k): float(v) for k, v in fold_results.items()},
        "per_dimension": metrics["per_dimension"],
        "training_time_seconds": time.time() - start_time,
    }

    with open(results_dir / f"{exp_id}.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{exp_id} COMPLETE: R2={avg_r2:.4f}")
    return results


def run_4fold_dual_experiment(
    exp_id: str,
    description: str,
    model_factory: Callable,
    mert_cache_dir: Path,
    muq_cache_dir: Path,
    labels: Dict,
    fold_assignments: Dict,
    config: Dict,
    checkpoint_root: Path,
    results_dir: Path,
    log_dir: Path,
) -> Dict:
    """Run 4-fold CV for dual MERT+MuQ fusion experiment.

    Args:
        exp_id: Experiment identifier (e.g., "D9a_mert_muq_ensemble")
        description: Human-readable description
        model_factory: Function that takes config and returns a model
        mert_cache_dir: Directory with cached MERT embeddings
        muq_cache_dir: Directory with cached MuQ embeddings
        labels: Dict mapping keys to label arrays
        fold_assignments: Dict with fold assignments
        config: Training configuration
        checkpoint_root: Root directory for checkpoints
        results_dir: Directory to save results JSON
        log_dir: Directory for training logs

    Returns:
        Results dict with summary, fold_results, per_dimension metrics
    """
    exp_checkpoint_dir = Path(checkpoint_root) / exp_id
    exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Check if already done
    existing = load_existing_results(exp_id, results_dir)
    if existing and experiment_completed(exp_id, checkpoint_root):
        print(f"SKIP {exp_id}: already completed (R2={existing['summary']['avg_r2']:.4f})")
        return existing

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {exp_id}")
    print(f"Description: {description}")
    print(f"{'='*70}")

    start_time = time.time()
    fold_results = {}
    all_preds, all_labels = [], []

    # Get all keys from fold assignments
    all_keys = []
    for i in range(4):
        all_keys.extend(fold_assignments.get(f"fold_{i}", []))

    for fold in range(config["n_folds"]):
        ckpt_path = exp_checkpoint_dir / f"fold{fold}_best.ckpt"

        # Get train/val keys for this fold
        val_keys = fold_assignments.get(f"fold_{fold}", [])
        train_keys = []
        for i in range(4):
            if i != fold:
                train_keys.extend(fold_assignments.get(f"fold_{i}", []))

        # Create datasets
        train_ds = DualEmbeddingDataset(
            mert_cache_dir, muq_cache_dir, labels, train_keys, config["max_frames"]
        )
        val_ds = DualEmbeddingDataset(
            mert_cache_dir, muq_cache_dir, labels, val_keys, config["max_frames"]
        )

        if len(train_ds) == 0 or len(val_ds) == 0:
            print(f"Fold {fold}: No data available, skipping")
            continue

        train_dl = DataLoader(
            train_ds,
            batch_size=config["batch_size"],
            shuffle=True,
            collate_fn=dual_collate_fn,
            num_workers=config["num_workers"],
            pin_memory=True,
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=config["batch_size"],
            shuffle=False,
            collate_fn=dual_collate_fn,
            num_workers=config["num_workers"],
            pin_memory=True,
        )

        if ckpt_path.exists():
            print(f"Fold {fold}: Loading existing checkpoint")
            model = model_factory(config)
            model = model.__class__.load_from_checkpoint(ckpt_path)
        else:
            print(f"Fold {fold}: Training ({len(train_ds)} train, {len(val_ds)} val)")
            model = model_factory(config)

            callbacks = [
                ModelCheckpoint(
                    dirpath=exp_checkpoint_dir,
                    filename=f"fold{fold}_best",
                    monitor="val_r2",
                    mode="max",
                    save_top_k=1,
                ),
                EarlyStopping(
                    monitor="val_r2", mode="max", patience=config["patience"], verbose=True
                ),
            ]

            trainer = pl.Trainer(
                max_epochs=config["max_epochs"],
                callbacks=callbacks,
                logger=CSVLogger(save_dir=log_dir, name=exp_id, version=f"fold{fold}"),
                accelerator="auto",
                devices=1,
                gradient_clip_val=config["gradient_clip_val"],
                enable_progress_bar=True,
                deterministic=True,
                log_every_n_steps=10,
            )

            trainer.fit(model, train_dl, val_dl)
            fold_results[fold] = float(callbacks[0].best_model_score or 0)

            # Reload best
            model = model.__class__.load_from_checkpoint(ckpt_path)

        # Evaluate
        model.eval().to("cuda")
        with torch.no_grad():
            for batch in val_dl:
                pred = model(
                    batch["mert_embeddings"].cuda(),
                    batch["muq_embeddings"].cuda(),
                    batch["mert_mask"].cuda() if batch.get("mert_mask") is not None else None,
                    batch["muq_mask"].cuda() if batch.get("muq_mask") is not None else None,
                )
                all_preds.append(pred.cpu().numpy())
                all_labels.append(batch["labels"].numpy())

        del model
        torch.cuda.empty_cache()

    # Aggregate results
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    metrics = compute_comprehensive_metrics(all_preds, all_labels)

    # If fold_results is empty (loaded from checkpoints), compute from metrics
    if not fold_results:
        fold_results = {i: metrics["overall_r2"] for i in range(4)}

    avg_r2 = np.mean(list(fold_results.values()))
    std_r2 = np.std(list(fold_results.values()))

    results = {
        "experiment_id": exp_id,
        "description": description,
        "config": config,
        "summary": {
            "avg_r2": float(avg_r2),
            "std_r2": float(std_r2),
            "r2_ci_95": metrics["r2_ci_95"],
            "overall_r2": metrics["overall_r2"],
            "overall_mae": metrics["overall_mae"],
            "dispersion_ratio": metrics["dispersion_ratio"],
        },
        "fold_results": {str(k): float(v) for k, v in fold_results.items()},
        "per_dimension": metrics["per_dimension"],
        "training_time_seconds": time.time() - start_time,
    }

    # Save results
    with open(results_dir / f"{exp_id}.json", "w") as f:
        json.dump(results, f, indent=2)

    print(
        f"\n{exp_id} COMPLETE: R2={avg_r2:.4f} +/- {std_r2:.4f}, "
        f"CI=[{metrics['r2_ci_95'][0]:.4f}, {metrics['r2_ci_95'][1]:.4f}]"
    )

    return results
