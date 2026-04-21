"""A1-Max hyperparameter sweep runner.

Sweeps 18 architecture configs x 3 PercePiano mix ratios x 4 folds = 216 runs.
Ratios vary the PercePiano fraction of the training mix (20 / 30 / 35 %).
With only PercePiano data loaded the ratio is a label-only parameter; it
becomes load-bearing once T5 data is wired into MixWeightedSampler.

Usage:
    cd model/

    # Full sweep (GPU recommended — ~30h on A100):
    uv run python -m model_improvement.a1_max_sweep

    # Local overnight run — top-1 config from prior results, 4 DataLoader workers:
    uv run python -m model_improvement.a1_max_sweep --top-n-configs 1 --num-workers 4

    # Resume an interrupted run automatically (results file is checked at startup).
"""

from __future__ import annotations

import argparse
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
    MixWeightedSampler,
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
# PercePiano fraction of the training mix.  Swept here to find the ratio that
# drops dimension_collapse >= 0.15 without losing pairwise accuracy > 2pp.
# Winning ratio is baked into BASE_CONFIG so downstream experiments inherit it.
PERCEPIANO_RATIOS = [0.20, 0.30, 0.35]

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
    # Winning PercePiano mix ratio baked in after the sweep completes.
    # All downstream experiments that spread BASE_CONFIG inherit this value.
    "percepiano_ratio": 0.30,
}

BATCH_SIZE = 4
ACCUM_BATCHES = 4


def _cleanup_memory():
    gc.collect()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _config_name(lora_rank, lora_layers, label_smoothing, percepiano_ratio):
    layer_str = f"L{lora_layers[0]}-{lora_layers[-1]}"
    ratio_str = f"mix{int(percepiano_ratio * 100)}"
    return f"A1max_r{lora_rank}_{layer_str}_ls{label_smoothing}_{ratio_str}"


def _select_top_configs(
    all_configs: list,
    top_n: int,
    prior_results_path: Path,
) -> list:
    """Return the top-N configs by pairwise_mean from a prior results JSON.

    Checks prior_results_path first, then falls back to the legacy
    a1_max_sweep.json (generated before the diagnostics rewrite). Falls back
    to grid order if neither file exists.
    """
    legacy_path = Results.root / "a1_max_sweep.json"
    candidates = [p for p in (prior_results_path, legacy_path) if p.exists()]
    if not candidates:
        print(f"No prior results found; using grid order for top-{top_n}.")
        return all_configs[:top_n]

    chosen = candidates[0]
    print(f"Loading prior pairwise rankings from: {chosen.name}")
    with open(chosen) as f:
        prior = json.load(f)

    ranked = sorted(prior.items(), key=lambda x: x[1].get("pairwise_mean", 0.0), reverse=True)
    top_names = {name for name, _ in ranked[:top_n]}

    filtered = [
        (rank, layers, ls, ratio)
        for rank, layers, ls, ratio in all_configs
        if _config_name(rank, layers, ls, ratio) in top_names
    ]
    if not filtered:
        print(f"Warning: top-{top_n} names not found in grid; using grid order.")
        return all_configs[:top_n]

    print(f"Running top-{top_n} config(s) by prior pairwise: {[_config_name(*c) for c in filtered]}")
    return filtered


def run_sweep(
    checkpoint_dir: Path = Checkpoints.root / "a1_max_sweep",
    results_path: Path = Results.root / "a1_max_sweep_results.json",
    num_workers: int = 0,
    top_n_configs: int | None = None,
):
    """Run the A1-Max sweep (all 18 configs by default).

    Args:
        checkpoint_dir: Directory for per-fold checkpoints.
        results_path: JSON file for incremental results (supports resume).
        num_workers: DataLoader background workers. Use 4 for local MPS overnight
            runs to overlap batch prefetch with GPU compute. Default 0.
        top_n_configs: If set, run only the top-N configs by pairwise_mean from
            a prior completed sweep at results_path. Pass 1 to capture baseline
            diagnostics overnight on local hardware.
    """
    pl.seed_everything(42, workers=True)

    # Load data
    composite_path = Labels.composite / "composite_labels.json"
    labels_raw = load_composite_labels(composite_path)
    labels = {k: v.tolist() for k, v in labels_raw.items()}

    emb_path = Embeddings.percepiano / "muq_embeddings.pt"
    # weights_only=True limits deserialization to tensor data only.  # nosemgrep
    embeddings = torch.load(emb_path, map_location="cpu", weights_only=True)  # nosemgrep

    with open(Labels.percepiano / "folds.json") as f:
        folds = json.load(f)
    with open(Labels.percepiano / "piece_mapping.json") as f:
        piece_to_keys = json.load(f)

    print(f"Loaded {len(labels)} labels, {len(embeddings)} embeddings, {len(folds)} folds")

    all_configs = list(
        product(LORA_RANKS, LORA_LAYER_RANGES, LABEL_SMOOTHING_VALUES, PERCEPIANO_RATIOS)
    )
    if top_n_configs is not None:
        configs = _select_top_configs(all_configs, top_n_configs, results_path)
    else:
        configs = all_configs

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
    for lora_rank, lora_layers, label_smoothing, percepiano_ratio in configs:
        config_name = _config_name(lora_rank, lora_layers, label_smoothing, percepiano_ratio)

        if config_name in results:
            print(f"\nSkipping {config_name} (already completed)")
            run_count += len(folds)
            continue

        config = {**BASE_CONFIG}
        config["lora_rank"] = lora_rank
        config["lora_target_layers"] = lora_layers
        config["label_smoothing"] = label_smoothing
        config["percepiano_ratio"] = percepiano_ratio

        print(f"\n{'='*60}")
        print(f"Config: {config_name}")
        print(
            f"  lora_rank={lora_rank}, layers={lora_layers}, "
            f"ls={label_smoothing}, mix={percepiano_ratio}"
        )

        fold_metrics = []
        for fold_idx, fold in enumerate(folds):
            run_count += 1
            print(f"\n  Fold {fold_idx} (run {run_count}/{total_runs})")
            start_time = time.time()

            _cleanup_memory()

            emb_keys = set(embeddings.keys())
            train_ds = PairedPerformanceDataset(
                cache_dir=Embeddings.percepiano, labels=labels,
                piece_to_keys=piece_to_keys, keys=fold["train"],
                embedding_keys=emb_keys,
            )
            val_ds = PairedPerformanceDataset(
                cache_dir=Embeddings.percepiano, labels=labels,
                piece_to_keys=piece_to_keys, keys=fold["val"],
                embedding_keys=emb_keys,
            )

            train_sampler = MixWeightedSampler(
                dataset_sizes=[len(train_ds)],
                weights=[percepiano_ratio],
                epoch_size=len(train_ds),
            )
            train_loader = DataLoader(  # nosemgrep
                train_ds, batch_size=BATCH_SIZE, sampler=train_sampler,
                collate_fn=collate_fn, num_workers=num_workers,
                pin_memory=False,  # pin_memory=True is CUDA-only; MPS uses unified memory
            )
            val_loader = DataLoader(  # nosemgrep
                val_ds, batch_size=BATCH_SIZE, shuffle=False,
                collate_fn=collate_fn, num_workers=0,  # spawn issues on macOS with large closures
                pin_memory=False,
            )

            model = MuQLoRAMaxModel(**config)
            trainer = train_model(
                model, train_loader, val_loader,
                config_name, fold_idx,
                checkpoint_dir=checkpoint_dir,
                max_epochs=config["max_epochs"],
                patience=10,
                accumulate_grad_batches=ACCUM_BATCHES,
                trackio_experiment_id=f"{config_name}/fold_{fold_idx}",
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
        collapse_values = [
            m.get("dimension_collapse_score") for m in fold_metrics
            if m.get("dimension_collapse_score") is not None
            and not (isinstance(m.get("dimension_collapse_score"), float)
                     and m.get("dimension_collapse_score") != m.get("dimension_collapse_score"))
        ]
        results[config_name] = {
            "config": {
                "lora_rank": lora_rank,
                "lora_layers": list(lora_layers),
                "label_smoothing": label_smoothing,
                "percepiano_ratio": percepiano_ratio,
            },
            "pairwise_mean": sum(pw_values) / len(pw_values),
            "pairwise_per_fold": pw_values,
            "r2_mean": sum(r2_values) / len(r2_values),
            "r2_per_fold": r2_values,
            # Chunk A diagnostics: dimension collapse and per-dim independence.
            # Matrices kept per-fold (shape-stable JSON); scalar collapse
            # averaged across folds so the leaderboard can show it.
            "dimension_collapse_mean": (
                sum(collapse_values) / len(collapse_values)
                if collapse_values else None
            ),
            "dimension_collapse_per_fold": [
                m.get("dimension_collapse_score") for m in fold_metrics
            ],
            "per_dimension_correlation_per_fold": [
                m.get("per_dimension_correlation") for m in fold_metrics
            ],
            "conditional_independence_per_fold": [
                m.get("conditional_independence") for m in fold_metrics
            ],
            "skill_discrimination_per_fold": [
                m.get("skill_discrimination") for m in fold_metrics
            ],
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
    print(f"{'Config':<40} {'Pairwise':>10} {'R2':>10} {'Collapse':>10}")
    print("-" * 72)
    for name, metrics in sorted_configs:
        collapse = metrics.get("dimension_collapse_mean")
        collapse_str = f"{collapse:>10.4f}" if collapse is not None else f"{'n/a':>10}"
        print(
            f"{name:<40} {metrics['pairwise_mean']:>10.4f} "
            f"{metrics['r2_mean']:>10.4f} {collapse_str}"
        )

    best_name, best_metrics = sorted_configs[0]
    delta = best_metrics["pairwise_mean"] - 0.7393
    print(f"\nBest: {best_name} (pairwise={best_metrics['pairwise_mean']:.4f}, delta vs A1={delta:+.4f})")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A1-Max hyperparameter sweep with Chunk A diagnostics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full 18-config sweep (GPU recommended):
  uv run python -m model_improvement.a1_max_sweep

  # Overnight local run — best config from prior results, 4 prefetch workers:
  uv run python -m model_improvement.a1_max_sweep --top-n-configs 1 --num-workers 4

  # After completion, stamp results into Wave 1 plans:
  uv run python scripts/stamp_baseline_diagnostics.py
""",
    )
    parser.add_argument(
        "--top-n-configs",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Run only the top-N configs by pairwise_mean from a prior completed sweep. "
            "Reads data/results/a1_max_sweep_results.json or the legacy a1_max_sweep.json. "
            "Use 1 for a fast local overnight baseline diagnostic capture (~8h on MPS)."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        metavar="N",
        help=(
            "DataLoader background workers for batch prefetch. "
            "Use 4 on local MPS to overlap prefetch with GPU compute. "
            "Default 0 (synchronous, safest for cloud storage)."
        ),
    )
    args = parser.parse_args()
    run_sweep(num_workers=args.num_workers, top_n_configs=args.top_n_configs)
