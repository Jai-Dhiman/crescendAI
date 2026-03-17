"""Compute MAESTRO calibration statistics for the A1-Max ensemble.

Loads 4-fold A1-Max inference heads, runs all pre-computed MuQ embeddings
through the ensemble, and outputs per-dimension calibration statistics
(mean, std, percentiles) for normalizing scores.

Run from model/ directory:
    uv run python scripts/compute_maestro_calibration.py

Outputs:
    data/calibration/calibration_stats.json   -- aggregate stats per dimension
    data/calibration/calibration_scores.jsonl  -- per-segment scores
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import jsonlines
import numpy as np
import torch
import torch.nn as nn

DIMENSIONS = ["dynamics", "timing", "pedaling", "articulation", "phrasing", "interpretation"]

MODEL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MODEL_ROOT / "src"))

from src.paths import Embeddings, Checkpoints, Calibration

EMBEDDINGS_DIR = Embeddings.maestro / "muq_embeddings"
CHECKPOINT_DIR = Checkpoints.model_improvement / "A3"
METADATA_PATH = Embeddings.maestro / "metadata.jsonl"
OUTPUT_STATS = Calibration.root / "calibration_stats.json"
OUTPUT_SCORES = Calibration.root / "calibration_scores.jsonl"

N_FOLDS = 4


# Inference-only head replicated from apps/inference/models/loader.py.
# Copied inline to avoid cross-package import issues.
class A1MaxInferenceHead(nn.Module):
    """Predict quality scores from pre-computed MuQ frame embeddings."""

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        num_labels: int = 6,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_labels = num_labels

        self.attn = nn.Sequential(
            nn.Linear(input_dim, 256), nn.Tanh(), nn.Linear(256, 1)
        )

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_labels),
            nn.Sigmoid(),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        squeeze_output = False
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)
            squeeze_output = True

        scores = self.attn(embeddings).squeeze(-1)
        w = torch.softmax(scores, dim=-1).unsqueeze(-1)
        pooled = (embeddings * w).sum(1)

        z = self.encoder(pooled)
        result = self.regression_head(z)

        return result.squeeze(0) if squeeze_output else result


def load_head(ckpt_path: Path, device: torch.device) -> A1MaxInferenceHead:
    """Load an A1MaxInferenceHead from a PyTorch Lightning checkpoint."""
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    hparams = checkpoint.get("hyper_parameters", {})
    head = A1MaxInferenceHead(
        input_dim=hparams.get("input_dim", 1024),
        hidden_dim=hparams.get("hidden_dim", 512),
        num_labels=hparams.get("num_labels", 6),
        dropout=hparams.get("dropout", 0.2),
    )

    state_dict = checkpoint["state_dict"]
    head_state = {
        k: v for k, v in state_dict.items()
        if k.startswith("attn.") or k.startswith("encoder.") or k.startswith("regression_head.")
    }
    head.load_state_dict(head_state, strict=True)

    head.to(device)
    head.set_default_dtype = None
    head.requires_grad_(False)
    return head


def load_ensemble(device: torch.device) -> list[A1MaxInferenceHead]:
    """Load all 4 fold heads."""
    heads = []
    for fold in range(N_FOLDS):
        fold_dir = CHECKPOINT_DIR / f"fold_{fold}"
        if not fold_dir.exists():
            raise FileNotFoundError(f"Fold directory not found: {fold_dir}")

        ckpts = sorted(fold_dir.glob("*.ckpt"))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoint found in {fold_dir}")

        head = load_head(ckpts[0], device)
        heads.append(head)
        print(f"  Loaded fold {fold} from {ckpts[0].name}")

    return heads


@torch.no_grad()
def predict_ensemble(
    heads: list[A1MaxInferenceHead], embeddings: torch.Tensor
) -> torch.Tensor:
    """Run ensemble prediction: average across folds.

    Args:
        heads: List of 4 A1MaxInferenceHead models.
        embeddings: [B, T, D] float32 tensor on device.

    Returns:
        [B, 6] averaged predictions.
    """
    preds = torch.stack([head(embeddings) for head in heads], dim=0)
    return preds.mean(dim=0)


def load_metadata() -> dict[str, dict]:
    """Load MAESTRO metadata keyed by segment_id."""
    metadata = {}
    if METADATA_PATH.exists():
        with jsonlines.open(METADATA_PATH) as reader:
            for record in reader:
                metadata[record["segment_id"]] = record
    return metadata


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def main():
    parser = argparse.ArgumentParser(description="Compute MAESTRO calibration stats")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = select_device(args.device)
    batch_size = args.batch_size

    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print()

    # Load ensemble
    print("Loading A1-Max ensemble...")
    heads = load_ensemble(device)
    print(f"  {len(heads)} heads loaded\n")

    # Load metadata
    metadata = load_metadata()
    print(f"Loaded {len(metadata)} metadata records")

    # Discover embeddings
    embedding_files = sorted(EMBEDDINGS_DIR.glob("*.pt"))
    n_total = len(embedding_files)
    print(f"Found {n_total} embedding files\n")

    if n_total == 0:
        print("No embeddings found. Exiting.")
        sys.exit(1)

    # Process individually (variable T dimension prevents batching)
    all_predictions = []
    all_segment_ids = []
    start_time = time.time()

    for idx, f in enumerate(embedding_files):
        emb = torch.load(f, map_location="cpu", weights_only=True)
        emb = emb.float().to(device)  # [T, 1024]
        preds = predict_ensemble(heads, emb)  # [6]
        all_predictions.append(preds.cpu().numpy())
        all_segment_ids.append(f.stem)

        if (idx + 1) % 1000 == 0 or idx == n_total - 1:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed
            eta = (n_total - idx - 1) / rate if rate > 0 else 0
            print(f"  {idx + 1}/{n_total} segments "
                  f"[{elapsed:.1f}s elapsed, ~{eta:.0f}s remaining]")

        if (idx + 1) % 5000 == 0:
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()

    elapsed_total = time.time() - start_time
    print(f"\nInference complete: {n_total} segments in {elapsed_total:.1f}s\n")

    # Aggregate predictions
    all_predictions = np.stack(all_predictions, axis=0)  # [N, 6]

    # Compute per-dimension calibration stats
    stats = {}
    for i, dim in enumerate(DIMENSIONS):
        values = all_predictions[:, i]
        stats[dim] = {
            "mean": round(float(np.mean(values)), 6),
            "std": round(float(np.std(values)), 6),
            "p5": round(float(np.percentile(values, 5)), 6),
            "p25": round(float(np.percentile(values, 25)), 6),
            "p50": round(float(np.percentile(values, 50)), 6),
            "p75": round(float(np.percentile(values, 75)), 6),
            "p95": round(float(np.percentile(values, 95)), 6),
        }

    # Write calibration_stats.json
    with open(OUTPUT_STATS, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote {OUTPUT_STATS}")

    # Write per-segment scores
    with jsonlines.open(OUTPUT_SCORES, mode="w") as writer:
        for idx, seg_id in enumerate(all_segment_ids):
            scores = {dim: round(float(all_predictions[idx, i]), 6) for i, dim in enumerate(DIMENSIONS)}
            record = {"segment_id": seg_id, "scores": scores}
            meta = metadata.get(seg_id)
            if meta:
                record["canonical_composer"] = meta.get("canonical_composer", "")
                record["canonical_title"] = meta.get("canonical_title", "")
            writer.write(record)
    print(f"Wrote {OUTPUT_SCORES}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"MAESTRO Calibration Statistics ({n_total} segments, A1-Max 4-fold ensemble)")
    print(f"{'='*70}")
    print(f"{'Dimension':<16} {'Mean':>8} {'Std':>8} {'P5':>8} {'P25':>8} {'P50':>8} {'P75':>8} {'P95':>8}")
    print(f"{'-'*16} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for dim in DIMENSIONS:
        s = stats[dim]
        print(f"{dim:<16} {s['mean']:>8.4f} {s['std']:>8.4f} {s['p5']:>8.4f} "
              f"{s['p25']:>8.4f} {s['p50']:>8.4f} {s['p75']:>8.4f} {s['p95']:>8.4f}")
    print()


if __name__ == "__main__":
    main()
