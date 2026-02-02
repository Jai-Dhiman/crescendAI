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

from ..data import (
    DisentanglementDataset,
    PairwiseRankingDataset,
    TripletRankingDataset,
    build_multi_performer_pieces,
    disentanglement_collate_fn,
    get_fold_piece_mapping,
    pairwise_collate_fn,
    triplet_collate_fn,
)
from .metrics import compute_pairwise_metrics, evaluate_disentanglement


class PrintProgressCallback(pl.Callback):
    """Simple callback that prints progress for remote notebook environments."""

    def __init__(self, exp_id: str, fold: int):
        self.exp_id = exp_id
        self.fold = fold
        self.train_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        train_loss = trainer.callback_metrics.get("train_loss", float("nan"))
        val_loss = trainer.callback_metrics.get("val_loss", float("nan"))
        val_acc = trainer.callback_metrics.get("val_pairwise_acc", trainer.callback_metrics.get("val_r2", float("nan")))
        print(
            f"[{self.exp_id}][Fold {self.fold}] Epoch {epoch:3d} | "
            f"train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | val_metric: {val_acc:.4f}",
            flush=True,
        )

    def on_train_start(self, trainer, pl_module):
        print(f"[{self.exp_id}][Fold {self.fold}] Training started - {trainer.max_epochs} max epochs", flush=True)

    def on_train_end(self, trainer, pl_module):
        print(f"[{self.exp_id}][Fold {self.fold}] Training finished at epoch {trainer.current_epoch}", flush=True)


def experiment_completed(exp_id: str, checkpoint_dir: Path, n_folds: int = 4) -> bool:
    exp_dir = Path(checkpoint_dir) / exp_id
    if not exp_dir.exists():
        return False
    return all((exp_dir / f"fold{i}_best.ckpt").exists() for i in range(n_folds))


def load_existing_results(exp_id: str, results_dir: Path) -> Optional[Dict]:
    results_file = Path(results_dir) / f"{exp_id}.json"
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)
    return None


def run_pairwise_experiment(
    exp_id: str,
    description: str,
    model_factory: Callable,
    cache_dir: Path,
    labels: Dict,
    piece_to_keys: Dict[str, list],
    fold_assignments: Dict,
    config: Dict,
    checkpoint_root: Path,
    results_dir: Path,
    log_dir: Path,
    monitor_metric: str = "val_pairwise_acc",
    on_fold_complete: Optional[Callable[[str, int], None]] = None,
) -> Dict:
    exp_checkpoint_dir = Path(checkpoint_root) / exp_id
    exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Check if already done
    existing = load_existing_results(exp_id, results_dir)
    if existing and experiment_completed(exp_id, checkpoint_root):
        print(
            f"SKIP {exp_id}: already completed (acc={existing['summary']['avg_pairwise_acc']:.4f})"
        )
        return existing

    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT: {exp_id}")
    print(f"Description: {description}")
    print(f"{'=' * 70}")

    start_time = time.time()
    fold_results = {}
    all_logits, all_labels_a, all_labels_b = [], [], []

    n_folds = config.get("n_folds", 4)

    for fold in range(n_folds):
        ckpt_path = exp_checkpoint_dir / f"fold{fold}_best.ckpt"

        # Get fold-specific piece mapping
        train_piece_map, train_keys = get_fold_piece_mapping(
            piece_to_keys, fold_assignments, fold, "train"
        )
        val_piece_map, val_keys = get_fold_piece_mapping(
            piece_to_keys, fold_assignments, fold, "val"
        )

        # Create datasets
        train_ds = PairwiseRankingDataset(
            cache_dir,
            labels,
            train_piece_map,
            train_keys,
            max_frames=config.get("max_frames", 1000),
            ambiguous_threshold=config.get("ambiguous_threshold", 0.05),
        )
        val_ds = PairwiseRankingDataset(
            cache_dir,
            labels,
            val_piece_map,
            val_keys,
            max_frames=config.get("max_frames", 1000),
            ambiguous_threshold=config.get("ambiguous_threshold", 0.05),
        )

        if len(train_ds) == 0 or len(val_ds) == 0:
            print(f"Fold {fold}: No data available, skipping")
            continue

        train_dl = DataLoader(
            train_ds,
            batch_size=config.get("batch_size", 32),
            shuffle=True,
            collate_fn=pairwise_collate_fn,
            num_workers=config.get("num_workers", 2),
            pin_memory=True,
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=config.get("batch_size", 32),
            shuffle=False,
            collate_fn=pairwise_collate_fn,
            num_workers=config.get("num_workers", 2),
            pin_memory=True,
        )

        print(f"Fold {fold}: {len(train_ds)} train pairs, {len(val_ds)} val pairs")

        # Add num_pieces to config for models that need it
        config_with_pieces = {**config, "num_pieces": train_ds.get_num_pieces()}

        if ckpt_path.exists():
            print(f"Fold {fold}: Loading existing checkpoint")
            model = model_factory(config_with_pieces)
            model = model.__class__.load_from_checkpoint(ckpt_path)
        else:
            model = model_factory(config_with_pieces)

            callbacks = [
                ModelCheckpoint(
                    dirpath=exp_checkpoint_dir,
                    filename=f"fold{fold}_best",
                    monitor=monitor_metric,
                    mode="max",
                    save_top_k=1,
                ),
                EarlyStopping(
                    monitor=monitor_metric,
                    mode="max",
                    patience=config.get("patience", 15),
                    verbose=True,
                ),
                PrintProgressCallback(exp_id, fold),
            ]

            trainer = pl.Trainer(
                max_epochs=config.get("max_epochs", 200),
                callbacks=callbacks,
                logger=CSVLogger(save_dir=log_dir, name=exp_id, version=f"fold{fold}"),
                accelerator="auto",
                devices=1,
                gradient_clip_val=config.get("gradient_clip_val", 1.0),
                enable_progress_bar=False,  # Disabled for remote notebooks
                deterministic=True,
                log_every_n_steps=10,
            )

            trainer.fit(model, train_dl, val_dl)
            fold_results[fold] = float(callbacks[0].best_model_score or 0.5)
            model = model.__class__.load_from_checkpoint(ckpt_path)

            # Upload checkpoint after fold completes
            if on_fold_complete is not None:
                on_fold_complete(exp_id, fold)

        # Evaluate
        model.eval().to("cuda" if torch.cuda.is_available() else "cpu")
        device = next(model.parameters()).device

        with torch.no_grad():
            for batch in val_dl:
                outputs = model(
                    batch["embeddings_a"].to(device),
                    batch["embeddings_b"].to(device),
                    batch.get(
                        "mask_a",
                        batch["embeddings_a"].new_ones(
                            batch["embeddings_a"].shape[:2], dtype=torch.bool
                        ),
                    ).to(device),
                    batch.get(
                        "mask_b",
                        batch["embeddings_b"].new_ones(
                            batch["embeddings_b"].shape[:2], dtype=torch.bool
                        ),
                    ).to(device),
                )
                if isinstance(outputs, dict):
                    logits = outputs["ranking_logits"]
                else:
                    logits = outputs
                all_logits.append(logits.cpu().numpy())
                all_labels_a.append(batch["labels_a"].numpy())
                all_labels_b.append(batch["labels_b"].numpy())

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Aggregate results
    all_logits = np.vstack(all_logits)
    all_labels_a = np.vstack(all_labels_a)
    all_labels_b = np.vstack(all_labels_b)

    metrics = compute_pairwise_metrics(all_logits, all_labels_a, all_labels_b)

    if not fold_results:
        fold_results = {i: metrics["overall_accuracy"] for i in range(n_folds)}

    avg_acc = np.mean(list(fold_results.values()))
    std_acc = np.std(list(fold_results.values()))

    results = {
        "experiment_id": exp_id,
        "description": description,
        "config": {k: v for k, v in config.items() if not callable(v)},
        "summary": {
            "avg_pairwise_acc": float(avg_acc),
            "std_pairwise_acc": float(std_acc),
            "overall_accuracy": metrics["overall_accuracy"],
            "kendall_tau": metrics.get("kendall_tau", 0),
            "n_comparisons": metrics["n_comparisons"],
        },
        "fold_results": {str(k): float(v) for k, v in fold_results.items()},
        "per_dimension": metrics["per_dimension"],
        "training_time_seconds": time.time() - start_time,
    }

    with open(results_dir / f"{exp_id}.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{exp_id} COMPLETE: Acc={avg_acc:.4f} +/- {std_acc:.4f}")
    return results


def run_disentanglement_experiment(
    exp_id: str,
    description: str,
    model_factory: Callable,
    cache_dir: Path,
    labels: Dict,
    piece_to_keys: Dict[str, list],
    fold_assignments: Dict,
    config: Dict,
    checkpoint_root: Path,
    results_dir: Path,
    log_dir: Path,
    on_fold_complete: Optional[Callable[[str, int], None]] = None,
) -> Dict:
    exp_checkpoint_dir = Path(checkpoint_root) / exp_id
    exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    existing = load_existing_results(exp_id, results_dir)
    if existing and experiment_completed(exp_id, checkpoint_root):
        print(
            f"SKIP {exp_id}: already completed (R2={existing['summary']['avg_r2']:.4f})"
        )
        return existing

    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT: {exp_id}")
    print(f"Description: {description}")
    print(f"{'=' * 70}")

    start_time = time.time()
    fold_results = {}
    all_preds, all_labels = [], []
    all_z_style, all_z_piece, all_piece_ids = [], [], []

    n_folds = config.get("n_folds", 4)

    for fold in range(n_folds):
        ckpt_path = exp_checkpoint_dir / f"fold{fold}_best.ckpt"

        _, train_keys = get_fold_piece_mapping(
            piece_to_keys, fold_assignments, fold, "train"
        )
        _, val_keys = get_fold_piece_mapping(
            piece_to_keys, fold_assignments, fold, "val"
        )

        train_ds = DisentanglementDataset(
            cache_dir,
            labels,
            piece_to_keys,
            train_keys,
            max_frames=config.get("max_frames", 1000),
        )
        val_ds = DisentanglementDataset(
            cache_dir,
            labels,
            piece_to_keys,
            val_keys,
            max_frames=config.get("max_frames", 1000),
        )

        if len(train_ds) == 0 or len(val_ds) == 0:
            print(f"Fold {fold}: No data available, skipping")
            continue

        train_dl = DataLoader(
            train_ds,
            batch_size=config.get("batch_size", 64),
            shuffle=True,
            collate_fn=disentanglement_collate_fn,
            num_workers=config.get("num_workers", 2),
            pin_memory=True,
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=config.get("batch_size", 64),
            shuffle=False,
            collate_fn=disentanglement_collate_fn,
            num_workers=config.get("num_workers", 2),
            pin_memory=True,
        )

        print(f"Fold {fold}: {len(train_ds)} train, {len(val_ds)} val")

        config_with_pieces = {**config, "num_pieces": train_ds.get_num_pieces()}

        if ckpt_path.exists():
            model = model_factory(config_with_pieces)
            model = model.__class__.load_from_checkpoint(ckpt_path)
        else:
            model = model_factory(config_with_pieces)

            callbacks = [
                ModelCheckpoint(
                    dirpath=exp_checkpoint_dir,
                    filename=f"fold{fold}_best",
                    monitor="val_r2",
                    mode="max",
                    save_top_k=1,
                ),
                EarlyStopping(
                    monitor="val_r2",
                    mode="max",
                    patience=config.get("patience", 15),
                    verbose=True,
                ),
                PrintProgressCallback(exp_id, fold),
            ]

            trainer = pl.Trainer(
                max_epochs=config.get("max_epochs", 200),
                callbacks=callbacks,
                logger=CSVLogger(save_dir=log_dir, name=exp_id, version=f"fold{fold}"),
                accelerator="auto",
                devices=1,
                gradient_clip_val=config.get("gradient_clip_val", 1.0),
                enable_progress_bar=False,  # Disabled for remote notebooks
                deterministic=True,
                log_every_n_steps=10,
            )

            trainer.fit(model, train_dl, val_dl)
            fold_results[fold] = float(callbacks[0].best_model_score or 0)
            model = model.__class__.load_from_checkpoint(ckpt_path)

            # Upload checkpoint after fold completes
            if on_fold_complete is not None:
                on_fold_complete(exp_id, fold)

        # Evaluate
        model.eval().to("cuda" if torch.cuda.is_available() else "cpu")
        device = next(model.parameters()).device

        with torch.no_grad():
            for batch in val_dl:
                outputs = model(
                    batch["embeddings"].to(device),
                    batch.get("attention_mask").to(device)
                    if batch.get("attention_mask") is not None
                    else None,
                )
                all_preds.append(outputs["predictions"].cpu().numpy())
                all_labels.append(batch["labels"].numpy())
                all_z_style.append(outputs["z_style"].cpu().numpy())
                all_z_piece.append(outputs["z_piece"].cpu().numpy())
                all_piece_ids.append(batch["piece_ids"].numpy())

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Aggregate
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    all_z_style = np.vstack(all_z_style)
    all_z_piece = np.vstack(all_z_piece)
    all_piece_ids = np.concatenate(all_piece_ids)

    from sklearn.metrics import mean_absolute_error, r2_score

    overall_r2 = r2_score(all_labels, all_preds)
    overall_mae = mean_absolute_error(all_labels, all_preds)

    # Disentanglement metrics
    disentangle = evaluate_disentanglement(
        all_z_style, all_z_piece, all_piece_ids, all_preds
    )

    if not fold_results:
        fold_results = {i: overall_r2 for i in range(n_folds)}

    avg_r2 = np.mean(list(fold_results.values()))
    std_r2 = np.std(list(fold_results.values()))

    results = {
        "experiment_id": exp_id,
        "description": description,
        "config": {k: v for k, v in config.items() if not callable(v)},
        "summary": {
            "avg_r2": float(avg_r2),
            "std_r2": float(std_r2),
            "overall_r2": float(overall_r2),
            "overall_mae": float(overall_mae),
        },
        "disentanglement": disentangle,
        "fold_results": {str(k): float(v) for k, v in fold_results.items()},
        "training_time_seconds": time.time() - start_time,
    }

    with open(results_dir / f"{exp_id}.json", "w") as f:
        json.dump(results, f, indent=2)

    print(
        f"\n{exp_id} COMPLETE: R2={avg_r2:.4f}, style_piece_acc={disentangle['style_piece_accuracy']:.4f}"
    )
    return results


def run_triplet_experiment(
    exp_id: str,
    description: str,
    model_factory: Callable,
    cache_dir: Path,
    labels: Dict,
    piece_to_keys: Dict[str, list],
    fold_assignments: Dict,
    config: Dict,
    checkpoint_root: Path,
    results_dir: Path,
    log_dir: Path,
    monitor_metric: str = "val_pairwise_acc",
    on_fold_complete: Optional[Callable[[str, int], None]] = None,
) -> Dict:
    exp_checkpoint_dir = Path(checkpoint_root) / exp_id
    exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Check if already done
    existing = load_existing_results(exp_id, results_dir)
    if existing and experiment_completed(exp_id, checkpoint_root):
        print(
            f"SKIP {exp_id}: already completed (acc={existing['summary']['avg_pairwise_acc']:.4f})"
        )
        return existing

    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT: {exp_id}")
    print(f"Description: {description}")
    print(f"{'=' * 70}")

    start_time = time.time()
    fold_results = {}
    all_logits, all_labels_a, all_labels_n = [], [], []

    n_folds = config.get("n_folds", 4)

    for fold in range(n_folds):
        ckpt_path = exp_checkpoint_dir / f"fold{fold}_best.ckpt"

        # Get fold-specific piece mapping
        train_piece_map, train_keys = get_fold_piece_mapping(
            piece_to_keys, fold_assignments, fold, "train"
        )
        val_piece_map, val_keys = get_fold_piece_mapping(
            piece_to_keys, fold_assignments, fold, "val"
        )

        # Create triplet datasets
        train_ds = TripletRankingDataset(
            cache_dir,
            labels,
            train_piece_map,
            train_keys,
            max_frames=config.get("max_frames", 1000),
            min_score_diff=config.get("min_score_diff", 0.05),
        )
        val_ds = TripletRankingDataset(
            cache_dir,
            labels,
            val_piece_map,
            val_keys,
            max_frames=config.get("max_frames", 1000),
            min_score_diff=config.get("min_score_diff", 0.05),
        )

        if len(train_ds) == 0 or len(val_ds) == 0:
            print(f"Fold {fold}: No triplet data available, skipping")
            continue

        train_dl = DataLoader(
            train_ds,
            batch_size=config.get("batch_size", 32),
            shuffle=True,
            collate_fn=triplet_collate_fn,
            num_workers=config.get("num_workers", 2),
            pin_memory=True,
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=config.get("batch_size", 32),
            shuffle=False,
            collate_fn=triplet_collate_fn,
            num_workers=config.get("num_workers", 2),
            pin_memory=True,
        )

        print(
            f"Fold {fold}: {len(train_ds)} train triplets, {len(val_ds)} val triplets"
        )

        # Add num_pieces to config for models that need it
        config_with_pieces = {**config, "num_pieces": train_ds.get_num_pieces()}

        if ckpt_path.exists():
            print(f"Fold {fold}: Loading existing checkpoint")
            model = model_factory(config_with_pieces)
            model = model.__class__.load_from_checkpoint(ckpt_path)
        else:
            model = model_factory(config_with_pieces)

            callbacks = [
                ModelCheckpoint(
                    dirpath=exp_checkpoint_dir,
                    filename=f"fold{fold}_best",
                    monitor=monitor_metric,
                    mode="max",
                    save_top_k=1,
                ),
                EarlyStopping(
                    monitor=monitor_metric,
                    mode="max",
                    patience=config.get("patience", 15),
                    verbose=True,
                ),
                PrintProgressCallback(exp_id, fold),
            ]

            trainer = pl.Trainer(
                max_epochs=config.get("max_epochs", 100),
                callbacks=callbacks,
                logger=CSVLogger(save_dir=log_dir, name=exp_id, version=f"fold{fold}"),
                accelerator="auto",
                devices=1,
                gradient_clip_val=config.get("gradient_clip_val", 1.0),
                enable_progress_bar=False,  # Disabled for remote notebooks
                deterministic=True,
                log_every_n_steps=10,
            )

            trainer.fit(model, train_dl, val_dl)
            fold_results[fold] = float(callbacks[0].best_model_score or 0.5)
            model = model.__class__.load_from_checkpoint(ckpt_path)

            # Upload checkpoint after fold completes
            if on_fold_complete is not None:
                on_fold_complete(exp_id, fold)

        # Evaluate on validation set
        model.eval().to("cuda" if torch.cuda.is_available() else "cpu")
        device = next(model.parameters()).device

        with torch.no_grad():
            for batch in val_dl:
                outputs = model(
                    batch["embeddings_anchor"].to(device),
                    batch["embeddings_positive"].to(device),
                    batch["embeddings_negative"].to(device),
                    batch.get(
                        "mask_anchor",
                        batch["embeddings_anchor"].new_ones(
                            batch["embeddings_anchor"].shape[:2], dtype=torch.bool
                        ),
                    ).to(device),
                    batch.get(
                        "mask_positive",
                        batch["embeddings_positive"].new_ones(
                            batch["embeddings_positive"].shape[:2], dtype=torch.bool
                        ),
                    ).to(device),
                    batch.get(
                        "mask_negative",
                        batch["embeddings_negative"].new_ones(
                            batch["embeddings_negative"].shape[:2], dtype=torch.bool
                        ),
                    ).to(device),
                )
                # Use anchor vs negative logits for pairwise accuracy
                all_logits.append(outputs["ranking_logits_neg"].cpu().numpy())
                all_labels_a.append(batch["labels_anchor"].numpy())
                all_labels_n.append(batch["labels_negative"].numpy())

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Aggregate results
    all_logits = np.vstack(all_logits)
    all_labels_a = np.vstack(all_labels_a)
    all_labels_n = np.vstack(all_labels_n)

    # Compute pairwise metrics (anchor should rank above negative)
    metrics = compute_pairwise_metrics(all_logits, all_labels_a, all_labels_n)

    if not fold_results:
        fold_results = {i: metrics["overall_accuracy"] for i in range(n_folds)}

    avg_acc = np.mean(list(fold_results.values()))
    std_acc = np.std(list(fold_results.values()))

    results = {
        "experiment_id": exp_id,
        "description": description,
        "config": {k: v for k, v in config.items() if not callable(v)},
        "summary": {
            "avg_pairwise_acc": float(avg_acc),
            "std_pairwise_acc": float(std_acc),
            "overall_accuracy": metrics["overall_accuracy"],
            "kendall_tau": metrics.get("kendall_tau", 0),
            "n_comparisons": metrics["n_comparisons"],
        },
        "fold_results": {str(k): float(v) for k, v in fold_results.items()},
        "per_dimension": metrics["per_dimension"],
        "training_time_seconds": time.time() - start_time,
    }

    with open(results_dir / f"{exp_id}.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{exp_id} COMPLETE: Acc={avg_acc:.4f} +/- {std_acc:.4f}")
    return results


def run_dimension_group_experiment(
    exp_id: str,
    description: str,
    model_factory: Callable,
    cache_dir: Path,
    labels: Dict,
    piece_to_keys: Dict[str, list],
    fold_assignments: Dict,
    config: Dict,
    checkpoint_root: Path,
    results_dir: Path,
    log_dir: Path,
    dimension_groups: Dict[str, list],
    on_fold_complete: Optional[Callable[[str, int], None]] = None,
) -> Dict:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT: {exp_id}")
    print(f"Description: {description}")
    print(f"Training {len(dimension_groups)} specialized models")
    print(f"{'=' * 70}")

    start_time = time.time()
    group_results = {}
    all_per_dimension = {}

    for group_name, dim_indices in dimension_groups.items():
        group_exp_id = f"{exp_id}_{group_name}"
        print(f"\n--- Group: {group_name} (dims: {dim_indices}) ---")

        # Update config for this group
        group_config = {
            **config,
            "num_labels": len(dim_indices),
            "dimension_indices": dim_indices,
        }

        # Run pairwise experiment for this group
        group_result = run_pairwise_experiment(
            exp_id=group_exp_id,
            description=f"{description} - {group_name} group",
            model_factory=model_factory,
            cache_dir=cache_dir,
            labels=labels,
            piece_to_keys=piece_to_keys,
            fold_assignments=fold_assignments,
            config=group_config,
            checkpoint_root=checkpoint_root,
            results_dir=results_dir,
            log_dir=log_dir,
            on_fold_complete=on_fold_complete,
        )

        group_results[group_name] = {
            "avg_pairwise_acc": group_result["summary"]["avg_pairwise_acc"],
            "std_pairwise_acc": group_result["summary"]["std_pairwise_acc"],
            "dimensions": dim_indices,
        }

        # Map per-dimension results back to original indices
        for i, (dim_key, acc) in enumerate(group_result["per_dimension"].items()):
            original_dim = dim_indices[int(dim_key)]
            all_per_dimension[str(original_dim)] = acc

    # Aggregate results
    all_accs = [g["avg_pairwise_acc"] for g in group_results.values()]
    avg_acc = np.mean(all_accs)
    std_acc = np.std(all_accs)

    results = {
        "experiment_id": exp_id,
        "description": description,
        "config": {k: v for k, v in config.items() if not callable(v)},
        "summary": {
            "avg_pairwise_acc": float(avg_acc),
            "std_pairwise_acc": float(std_acc),
            "n_groups": len(dimension_groups),
        },
        "group_results": group_results,
        "per_dimension": all_per_dimension,
        "training_time_seconds": time.time() - start_time,
    }

    with open(results_dir / f"{exp_id}.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{exp_id} COMPLETE: Avg group acc={avg_acc:.4f} +/- {std_acc:.4f}")
    for group_name, g_res in group_results.items():
        print(f"  {group_name}: {g_res['avg_pairwise_acc']:.4f}")

    return results
