"""A1-Max hyperparameter sweep runner.

Runs 18 configs x 4 folds sequentially on local MPS with aggressive memory
management. Saves results to a JSON file for analysis.

Usage:
    cd model/
    uv run python -m model_improvement.a1_max_sweep
"""

from __future__ import annotations

import gc
import json
import time
from itertools import product
from pathlib import Path

import torch
import pytorch_lightning as pl
from functools import partial
from torch.utils.data import DataLoader

from src.paths import Checkpoints, Embeddings, Labels, Results

from model_improvement.audio_encoders import MuQLoRAMaxModel
from model_improvement.data import (
    PairedPerformanceDataset,
    HardNegativePairSampler,
    audio_pair_collate_fn,
)
from model_improvement.evaluation import evaluate_model
from model_improvement.taxonomy import load_composite_labels, NUM_DIMS

# Sweep grid
LORA_RANKS = [8, 16, 32]
LORA_LAYER_RANGES = [
    (9, 10, 11, 12),
    (7, 8, 9, 10, 11, 12),
]
LABEL_SMOOTHING_VALUES = [0.0, 0.05, 0.1]

BASE_CONFIG = {
    "input_dim": 1024,
    "hidden_dim": 512,
    "num_labels": NUM_DIMS,
    "learning_rate": 3e-5,
    "weight_decay": 1e-5,
    "temperature": 0.07,
    "lambda_listmle": 1.5,
    "lambda_contrastive": 0.3,
    "lambda_regression": 0.3,
    "lambda_invariance": 0.1,
    "use_ccc": True,
    "mixup_alpha": 0.2,
    "warmup_epochs": 5,
    "max_epochs": 200,
    "use_pretrained_muq": False,
}

BATCH_SIZE = 4
ACCUM_BATCHES = 4


def _cleanup_memory():
    gc.collect()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _config_name(lora_rank, lora_layers, label_smoothing):
    layer_str = f"L{lora_layers[0]}-{lora_layers[-1]}"
    return f"A1max_r{lora_rank}_{layer_str}_ls{label_smoothing}"


def run_sweep(
    checkpoint_dir: Path = Checkpoints.root / "a1_max_sweep",
    results_path: Path = Results.root / "a1_max_sweep_results.json",
):
    """Run the full 18x4 sweep."""
    pl.seed_everything(42, workers=True)

    # Load data
    composite_path = Labels.composite / "composite_labels.json"
    labels_raw = load_composite_labels(composite_path)
    labels = {k: v.tolist() for k, v in labels_raw.items()}

    emb_path = Embeddings.percepiano / "muq_embeddings.pt"
    embeddings = torch.load(emb_path, map_location="cpu", weights_only=True)

    with open(Labels.percepiano / "folds.json") as f:
        folds = json.load(f)
    with open(Labels.percepiano / "piece_mapping.json") as f:
        piece_to_keys = json.load(f)

    print(f"Loaded {len(labels)} labels, {len(embeddings)} embeddings, {len(folds)} folds")

    configs = list(product(LORA_RANKS, LORA_LAYER_RANGES, LABEL_SMOOTHING_VALUES))
    total_runs = len(configs) * len(folds)
    print(f"Sweep: {len(configs)} configs x {len(folds)} folds = {total_runs} runs")

    # Resume from existing results
    results = {}
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        print(f"Resuming: {len(results)} configs already completed")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    collate_fn = partial(audio_pair_collate_fn, embeddings=embeddings)

    from model_improvement.training import train_model

    run_count = 0
    for lora_rank, lora_layers, label_smoothing in configs:
        config_name = _config_name(lora_rank, lora_layers, label_smoothing)

        if config_name in results:
            print(f"\nSkipping {config_name} (already completed)")
            run_count += len(folds)
            continue

        config = {**BASE_CONFIG}
        config["lora_rank"] = lora_rank
        config["lora_target_layers"] = lora_layers
        config["label_smoothing"] = label_smoothing

        print(f"\n{'='*60}")
        print(f"Config: {config_name}")
        print(f"  lora_rank={lora_rank}, layers={lora_layers}, ls={label_smoothing}")

        fold_metrics = []
        for fold_idx, fold in enumerate(folds):
            run_count += 1
            print(f"\n  Fold {fold_idx} (run {run_count}/{total_runs})")
            start_time = time.time()

            _cleanup_memory()

            train_ds = PairedPerformanceDataset(
                cache_dir=Embeddings.percepiano, labels=labels,
                piece_to_keys=piece_to_keys, keys=fold["train"],
            )
            val_ds = PairedPerformanceDataset(
                cache_dir=Embeddings.percepiano, labels=labels,
                piece_to_keys=piece_to_keys, keys=fold["val"],
            )

            train_loader = DataLoader(
                train_ds, batch_size=BATCH_SIZE, shuffle=True,
                collate_fn=collate_fn, num_workers=0,
            )
            val_loader = DataLoader(
                val_ds, batch_size=BATCH_SIZE, shuffle=False,
                collate_fn=collate_fn, num_workers=0,
            )

            model = MuQLoRAMaxModel(**config)
            trainer = train_model(
                model, train_loader, val_loader,
                config_name, fold_idx,
                checkpoint_dir=checkpoint_dir,
                max_epochs=config["max_epochs"],
                patience=10,
                accumulate_grad_batches=ACCUM_BATCHES,
            )

            trained_model = trainer.lightning_module
            trained_model.cpu()
            trained_model.eval()

            fold_res = evaluate_model(
                trained_model, fold["val"], labels,
                get_input_fn=lambda key: (embeddings[key].unsqueeze(0), None),
                encode_fn=lambda m, inp, mask: m.encode(inp, mask),
                compare_fn=lambda m, z_a, z_b: m.compare(z_a, z_b),
                predict_fn=lambda m, inp, mask: m.predict_scores(inp, mask),
            )
            fold_metrics.append(fold_res)

            elapsed = time.time() - start_time
            pw = fold_res.get("pairwise", 0)
            r2 = fold_res.get("r2", 0)
            print(f"    pairwise={pw:.4f}, r2={r2:.4f} ({elapsed:.0f}s)")

            del model, trainer, trained_model
            del train_ds, val_ds, train_loader, val_loader
            _cleanup_memory()

        pw_values = [m.get("pairwise", 0) for m in fold_metrics]
        r2_values = [m.get("r2", 0) for m in fold_metrics]
        results[config_name] = {
            "config": {
                "lora_rank": lora_rank,
                "lora_layers": list(lora_layers),
                "label_smoothing": label_smoothing,
            },
            "pairwise_mean": sum(pw_values) / len(pw_values),
            "pairwise_per_fold": pw_values,
            "r2_mean": sum(r2_values) / len(r2_values),
            "r2_per_fold": r2_values,
        }

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n  {config_name}: pairwise={results[config_name]['pairwise_mean']:.4f}")

    # Print leaderboard
    print(f"\n{'='*60}")
    print("SWEEP RESULTS (sorted by pairwise accuracy)")
    print(f"{'='*60}")
    sorted_configs = sorted(
        results.items(), key=lambda x: x[1]["pairwise_mean"], reverse=True
    )
    print(f"{'Config':<40} {'Pairwise':>10} {'R2':>10}")
    print("-" * 60)
    for name, metrics in sorted_configs:
        print(f"{name:<40} {metrics['pairwise_mean']:>10.4f} {metrics['r2_mean']:>10.4f}")

    best_name, best_metrics = sorted_configs[0]
    delta = best_metrics["pairwise_mean"] - 0.7393
    print(f"\nBest: {best_name} (pairwise={best_metrics['pairwise_mean']:.4f}, delta vs A1={delta:+.4f})")

    return results


if __name__ == "__main__":
    run_sweep()
