"""Autoresearch wrapper: single-fold training for loss weight optimization.

Trains fold 0 only with configurable loss weights. Outputs structured
metrics (pairwise accuracy, R2) for the autoresearch loop.

Usage:
    cd model/
    uv run python -m model_improvement.autoresearch_loss_weights
    uv run python -m model_improvement.autoresearch_loss_weights \
        --lambda-listmle 2.0 --lambda-contrastive 0.5

Exit code 0 on success, 1 on failure.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from functools import partial

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.paths import Checkpoints, Embeddings, Labels

from model_improvement.audio_encoders import MuQLoRAMaxModel
from model_improvement.data import (
    PairedPerformanceDataset,
    audio_pair_collate_fn,
)
from model_improvement.evaluation import evaluate_model
from model_improvement.taxonomy import load_composite_labels, NUM_DIMS
from model_improvement.training import train_model

FOLD_IDX = 0
BATCH_SIZE = 4
ACCUM_BATCHES = 4
CHECKPOINT_DIR = Checkpoints.root / "autoresearch_loss"


def _cleanup_memory():
    gc.collect()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def run_single_fold(
    lambda_listmle: float,
    lambda_contrastive: float,
    lambda_regression: float,
    lambda_invariance: float,
    mixup_alpha: float,
) -> dict:
    """Train fold 0 and return metrics.

    Returns:
        Dict with keys: pairwise, r2, elapsed_seconds.

    Raises:
        RuntimeError: If training or evaluation fails.
    """
    pl.seed_everything(42, workers=True)

    # Load data
    composite_path = Labels.composite / "composite_labels.json"
    labels_raw = load_composite_labels(composite_path)
    labels = {k: v.tolist() for k, v in labels_raw.items()}

    emb_path = Embeddings.percepiano / "muq_embeddings.pt"
    # Safe: weights_only=True prevents arbitrary code execution
    embeddings = torch.load(emb_path, map_location="cpu", weights_only=True)  # nosemgrep

    with open(Labels.percepiano / "folds.json") as f:
        folds = json.load(f)
    with open(Labels.percepiano / "piece_mapping.json") as f:
        piece_to_keys = json.load(f)

    fold = folds[FOLD_IDX]
    collate_fn = partial(audio_pair_collate_fn, embeddings=embeddings)

    config = {
        "input_dim": 1024,
        "hidden_dim": 512,
        "num_labels": NUM_DIMS,
        "learning_rate": 3e-5,
        "weight_decay": 1e-5,
        "temperature": 0.07,
        "lambda_listmle": lambda_listmle,
        "lambda_contrastive": lambda_contrastive,
        "lambda_regression": lambda_regression,
        "lambda_invariance": lambda_invariance,
        "use_ccc": True,
        "mixup_alpha": mixup_alpha,
        "warmup_epochs": 5,
        "max_epochs": 200,
        "use_pretrained_muq": False,
        "lora_rank": 32,
        "lora_target_layers": (7, 8, 9, 10, 11, 12),
        "label_smoothing": 0.1,
    }

    print(f"Loss weights: listmle={lambda_listmle}, contrastive={lambda_contrastive}, "
          f"regression={lambda_regression}, invariance={lambda_invariance}, "
          f"mixup={mixup_alpha}")

    _cleanup_memory()
    start_time = time.time()

    train_ds = PairedPerformanceDataset(
        cache_dir=Embeddings.percepiano, labels=labels,
        piece_to_keys=piece_to_keys, keys=fold["train"],
    )
    val_ds = PairedPerformanceDataset(
        cache_dir=Embeddings.percepiano, labels=labels,
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

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    model = MuQLoRAMaxModel(**config)
    trainer = train_model(
        model, train_loader, val_loader,
        "autoresearch", FOLD_IDX,
        checkpoint_dir=CHECKPOINT_DIR,
        max_epochs=50,
        patience=7,
        accumulate_grad_batches=ACCUM_BATCHES,
    )

    # Load the BEST checkpoint, not the last epoch
    ckpt_callback = None
    for cb in trainer.callbacks:
        if isinstance(cb, pl.callbacks.ModelCheckpoint):
            ckpt_callback = cb
            break

    if ckpt_callback is not None and ckpt_callback.best_model_path:
        print(f"Loading best checkpoint: {ckpt_callback.best_model_path}")
        best_model = MuQLoRAMaxModel.load_from_checkpoint(
            ckpt_callback.best_model_path
        )
    else:
        best_model = trainer.lightning_module

    best_model.cpu()
    best_model.eval()

    fold_res = evaluate_model(
        best_model, fold["val"], labels,
        get_input_fn=lambda key: (embeddings[key].unsqueeze(0), None),
        encode_fn=lambda m, inp, mask: m.encode(inp, mask),
        compare_fn=lambda m, z_a, z_b: m.compare(z_a, z_b),
        predict_fn=lambda m, inp, mask: m.predict_scores(inp, mask),
    )

    elapsed = time.time() - start_time

    del model, trainer, best_model
    del train_ds, val_ds, train_loader, val_loader
    _cleanup_memory()

    return {
        "pairwise": fold_res.get("pairwise", 0.0),
        "r2": fold_res.get("r2", 0.0),
        "elapsed_seconds": round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Single-fold loss weight training for autoresearch"
    )
    parser.add_argument("--lambda-listmle", type=float, default=1.5)
    parser.add_argument("--lambda-contrastive", type=float, default=0.3)
    parser.add_argument("--lambda-regression", type=float, default=0.3)
    parser.add_argument("--lambda-invariance", type=float, default=0.1)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    args = parser.parse_args()

    result = run_single_fold(
        lambda_listmle=args.lambda_listmle,
        lambda_contrastive=args.lambda_contrastive,
        lambda_regression=args.lambda_regression,
        lambda_invariance=args.lambda_invariance,
        mixup_alpha=args.mixup_alpha,
    )

    # Structured output for autoresearch parsing
    print(f"\n{'='*60}")
    print("AUTORESEARCH_RESULT")
    print(f"pairwise={result['pairwise']:.6f}")
    print(f"r2={result['r2']:.6f}")
    print(f"elapsed={result['elapsed_seconds']}s")
    print(f"{'='*60}")

    # Also dump as JSON for machine parsing
    print("AUTORESEARCH_JSON=" + json.dumps(result))

    return 0


if __name__ == "__main__":
    sys.exit(main())
