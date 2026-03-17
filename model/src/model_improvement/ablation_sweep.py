"""Loss component ablation for ISMIR paper revision.

5 configs x 4 folds = 20 training runs.
Results saved to model/data/ablation_sweep_results.json.

Usage:
    cd model/
    uv run python -m model_improvement.ablation_sweep
    uv run python -m model_improvement.ablation_sweep --config frozen_probe --fold 0
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from functools import partial
from pathlib import Path

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.paths import Checkpoints, Embeddings, Labels, Results

from model_improvement.audio_encoders import MuQLoRAMaxModel, MuQFrozenProbeModel
from model_improvement.data import (
    PairedPerformanceDataset,
    audio_pair_collate_fn,
)
from model_improvement.evaluation import evaluate_model
from model_improvement.taxonomy import load_composite_labels, NUM_DIMS
from model_improvement.training import train_model

RESULTS_PATH = Path("data/ablation_sweep_results.json")
CHECKPOINT_DIR = Path("data/checkpoints/ablation")
DATA_DIR = Path("data")

BATCH_SIZE = 4
ACCUM_BATCHES = 4

# --------------------------------------------------------------------------- #
# Ablation configs
#
# All LoRA configs use rank-32, layers 7-12, label_smoothing=0.1 (matching
# the best A1-Max sweep config: A1max_r32_L7-12_ls0.1).
#
# The "_model_class" key selects which class to instantiate. Keys starting
# with "_" are stripped before passing kwargs to the constructor.
# --------------------------------------------------------------------------- #

ABLATION_CONFIGS = {
    "frozen_probe": {
        "_model_class": "MuQFrozenProbeModel",
        "input_dim": 1024,
        "hidden_dim": 512,
        "num_labels": NUM_DIMS,
        "learning_rate": 1e-4,  # Higher LR: only MLP head trains
        "weight_decay": 1e-5,
        "temperature": 0.07,
        "lora_target_layers": (7, 8, 9, 10, 11, 12),
        "label_smoothing": 0.0,
        "warmup_epochs": 5,
        "max_epochs": 200,
        "use_pretrained_muq": False,
    },
    "bce_ranking_only": {
        "_model_class": "MuQLoRAMaxModel",
        "input_dim": 1024,
        "hidden_dim": 512,
        "num_labels": NUM_DIMS,
        "learning_rate": 3e-5,
        "weight_decay": 1e-5,
        "temperature": 0.07,
        "lambda_listmle": 0.0,
        "lambda_contrastive": 0.0,
        "lambda_regression": 0.0,
        "lambda_invariance": 0.0,
        "use_ccc": False,
        "mixup_alpha": 0.0,
        "warmup_epochs": 5,
        "max_epochs": 200,
        "use_pretrained_muq": False,
        "lora_rank": 32,
        "lora_target_layers": (7, 8, 9, 10, 11, 12),
        "label_smoothing": 0.1,
    },
    "bce_plus_listmle": {
        "_model_class": "MuQLoRAMaxModel",
        "input_dim": 1024,
        "hidden_dim": 512,
        "num_labels": NUM_DIMS,
        "learning_rate": 3e-5,
        "weight_decay": 1e-5,
        "temperature": 0.07,
        "lambda_listmle": 1.5,
        "lambda_contrastive": 0.0,
        "lambda_regression": 0.0,
        "lambda_invariance": 0.0,
        "use_ccc": False,
        "mixup_alpha": 0.0,
        "warmup_epochs": 5,
        "max_epochs": 200,
        "use_pretrained_muq": False,
        "lora_rank": 32,
        "lora_target_layers": (7, 8, 9, 10, 11, 12),
        "label_smoothing": 0.1,
    },
    "bce_listmle_ccc": {
        "_model_class": "MuQLoRAMaxModel",
        "input_dim": 1024,
        "hidden_dim": 512,
        "num_labels": NUM_DIMS,
        "learning_rate": 3e-5,
        "weight_decay": 1e-5,
        "temperature": 0.07,
        "lambda_listmle": 1.5,
        "lambda_contrastive": 0.0,
        "lambda_regression": 0.3,
        "lambda_invariance": 0.0,
        "use_ccc": True,
        "mixup_alpha": 0.0,
        "warmup_epochs": 5,
        "max_epochs": 200,
        "use_pretrained_muq": False,
        "lora_rank": 32,
        "lora_target_layers": (7, 8, 9, 10, 11, 12),
        "label_smoothing": 0.1,
    },
    "full_a1max_repro": {
        "_model_class": "MuQLoRAMaxModel",
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
        "lora_rank": 32,
        "lora_target_layers": (7, 8, 9, 10, 11, 12),
        "label_smoothing": 0.1,
    },
}

MODEL_CLASSES = {
    "MuQLoRAMaxModel": MuQLoRAMaxModel,
    "MuQFrozenProbeModel": MuQFrozenProbeModel,
}


def _cleanup_memory():
    gc.collect()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def run_ablation_sweep(
    config_filter: str | None = None,
    fold_filter: int | None = None,
):
    """Run ablation sweep with resume support."""
    pl.seed_everything(42, workers=True)

    # Load data (same pattern as a1_max_sweep.py)
    cache_dir = DATA_DIR / "percepiano_cache"
    composite_path = DATA_DIR / "composite_labels" / "composite_labels.json"
    labels_raw = load_composite_labels(composite_path)
    labels = {k: v.tolist() for k, v in labels_raw.items()}

    emb_path = cache_dir / "muq_embeddings.pt"
    # Safe: weights_only=True prevents arbitrary code execution
    embeddings = torch.load(emb_path, map_location="cpu", weights_only=True)  # nosemgrep

    with open(cache_dir / "folds.json") as f:
        folds = json.load(f)
    with open(cache_dir / "piece_mapping.json") as f:
        piece_to_keys = json.load(f)

    print(f"Loaded {len(labels)} labels, {len(embeddings)} embeddings, {len(folds)} folds")

    # Resume from existing results
    results = {}
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            results = json.load(f)
        print(f"Resuming: {len(results)} configs already completed")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    collate_fn = partial(audio_pair_collate_fn, embeddings=embeddings)

    configs_to_run = list(ABLATION_CONFIGS.items())
    if config_filter:
        configs_to_run = [(k, v) for k, v in configs_to_run if k == config_filter]

    for config_name, config in configs_to_run:
        if config_name in results:
            print(f"\nSkipping {config_name} (already completed)")
            continue

        # Resolve model class without mutating the config dict
        model_class_name = config.get("_model_class", "MuQLoRAMaxModel")
        model_class = MODEL_CLASSES[model_class_name]
        model_kwargs = {k: v for k, v in config.items() if not k.startswith("_")}

        print(f"\n{'='*60}")
        print(f"Config: {config_name} ({model_class_name})")
        print(f"  losses: listmle={model_kwargs.get('lambda_listmle', 0)}, "
              f"contrastive={model_kwargs.get('lambda_contrastive', 0)}, "
              f"regression={model_kwargs.get('lambda_regression', 0)}, "
              f"invariance={model_kwargs.get('lambda_invariance', 0)}")

        fold_range = list(range(len(folds)))
        if fold_filter is not None:
            fold_range = [fold_filter]

        fold_metrics = []
        for fold_idx in fold_range:
            fold = folds[fold_idx]
            print(f"\n  Fold {fold_idx}")
            start_time = time.time()
            _cleanup_memory()

            train_ds = PairedPerformanceDataset(
                cache_dir=cache_dir, labels=labels,
                piece_to_keys=piece_to_keys, keys=fold["train"],
            )
            val_ds = PairedPerformanceDataset(
                cache_dir=cache_dir, labels=labels,
                piece_to_keys=piece_to_keys, keys=fold["val"],
            )
            train_loader = DataLoader(  # nosemgrep
                train_ds, batch_size=BATCH_SIZE, shuffle=True,
                collate_fn=collate_fn, num_workers=0, pin_memory=False,
            )
            val_loader = DataLoader(  # nosemgrep
                val_ds, batch_size=BATCH_SIZE, shuffle=False,
                collate_fn=collate_fn, num_workers=0, pin_memory=False,
            )

            model = model_class(**model_kwargs)
            trainer = train_model(
                model, train_loader, val_loader,
                config_name, fold_idx,
                checkpoint_dir=CHECKPOINT_DIR,
                max_epochs=model_kwargs.get("max_epochs", 200),
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

        # Only save full results (all 4 folds), not partial single-fold runs
        if fold_filter is not None:
            print(f"\n  Single-fold dry run complete for {config_name}")
            continue

        pw_values = [m.get("pairwise", 0) for m in fold_metrics]
        r2_values = [m.get("r2", 0) for m in fold_metrics]
        results[config_name] = {
            "config": {k: (list(v) if isinstance(v, tuple) else v)
                       for k, v in model_kwargs.items()},
            "pairwise_mean": sum(pw_values) / len(pw_values),
            "pairwise_per_fold": pw_values,
            "r2_mean": sum(r2_values) / len(r2_values),
            "r2_per_fold": r2_values,
        }

        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  {config_name}: pairwise={results[config_name]['pairwise_mean']:.4f}")

    # Leaderboard
    if results:
        print(f"\n{'='*60}")
        print("ABLATION RESULTS (sorted by pairwise accuracy)")
        print(f"{'='*60}")
        sorted_configs = sorted(
            results.items(), key=lambda x: x[1]["pairwise_mean"], reverse=True
        )
        print(f"{'Config':<25} {'Pairwise':>10} {'R2':>10}")
        print("-" * 45)
        for name, metrics in sorted_configs:
            print(f"{name:<25} {metrics['pairwise_mean']:>10.4f} {metrics['r2_mean']:>10.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISMIR ablation sweep")
    parser.add_argument("--config", type=str, default=None,
                        help="Run single config (e.g., frozen_probe)")
    parser.add_argument("--fold", type=int, default=None,
                        help="Run single fold (0-3) for dry run")
    args = parser.parse_args()
    run_ablation_sweep(config_filter=args.config, fold_filter=args.fold)
