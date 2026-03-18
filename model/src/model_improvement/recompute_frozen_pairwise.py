"""Recompute frozen probe pairwise accuracy from regression scores.

The original evaluation used the comparator (randomly initialized + frozen),
producing meaningless pairwise numbers. This script loads the trained frozen
probe checkpoints and computes pairwise accuracy from regression score
differences: if score_a[dim] > score_b[dim], predict A is better.

Usage:
    cd model/
    uv run python -m model_improvement.recompute_frozen_pairwise
"""

from __future__ import annotations

import json

import torch

from src.paths import Embeddings, Labels, Checkpoints, Results
from model_improvement.audio_encoders import MuQFrozenProbeModel
from model_improvement.metrics import MetricsSuite
from model_improvement.taxonomy import load_composite_labels, NUM_DIMS

CHECKPOINT_DIR = Checkpoints.root / "ablation"
RESULTS_PATH = Results.root / "ablation_sweep.json"


def main():
    # Load data (paths from src.paths)
    composite_path = Labels.composite / "composite_labels.json"
    labels_raw = load_composite_labels(composite_path)
    labels = {k: v.tolist() for k, v in labels_raw.items()}

    emb_path = Embeddings.percepiano / "muq_embeddings.pt"
    # Safe: weights_only=True prevents arbitrary code execution
    embeddings = torch.load(emb_path, map_location="cpu", weights_only=True)  # nosemgrep

    with open(Labels.percepiano / "folds.json") as f:
        folds = json.load(f)

    suite = MetricsSuite(ambiguous_threshold=0.05)

    print("Recomputing frozen probe pairwise from regression scores\n")

    pw_per_fold = []
    r2_per_fold = []

    for fold_idx, fold in enumerate(folds):
        # Find checkpoint
        ckpt_dir = CHECKPOINT_DIR / "frozen_probe" / f"fold_{fold_idx}"
        ckpt_files = sorted(ckpt_dir.glob("*.ckpt"))
        if not ckpt_files:
            print(f"  Fold {fold_idx}: no checkpoint found in {ckpt_dir}")
            continue

        # Pick best checkpoint (lowest val_loss from filename)
        ckpt_path = min(
            ckpt_files,
            key=lambda p: float(p.stem.split("val_loss=")[1]) if "val_loss=" in p.stem else float("inf"),
        )
        print(f"  Fold {fold_idx}: loading {ckpt_path.name}")

        model = MuQFrozenProbeModel.load_from_checkpoint(
            ckpt_path, map_location="cpu"
        )
        model.eval()

        val_keys = [k for k in fold["val"] if k in labels and k in embeddings]

        # Predict scores for all val keys
        scores_cache = {}
        with torch.no_grad():
            for key in val_keys:
                inp = embeddings[key].unsqueeze(0)
                scores_cache[key] = model.predict_scores(inp, None).squeeze(0)

        # Compute pairwise from regression score differences
        # score_a - score_b as "logits": positive means A predicted higher
        all_score_diffs = []
        all_la = []
        all_lb = []

        for i, key_a in enumerate(val_keys):
            for key_b in val_keys[i + 1:]:
                diff = scores_cache[key_a] - scores_cache[key_b]
                all_score_diffs.append(diff.unsqueeze(0))

                lab_a = torch.tensor(labels[key_a][:NUM_DIMS], dtype=torch.float32)
                lab_b = torch.tensor(labels[key_b][:NUM_DIMS], dtype=torch.float32)
                all_la.append(lab_a.unsqueeze(0))
                all_lb.append(lab_b.unsqueeze(0))

        logits = torch.cat(all_score_diffs)
        la = torch.cat(all_la)
        lb = torch.cat(all_lb)

        pw = suite.pairwise_accuracy(logits, la, lb)
        pw_per_fold.append(pw["overall"])

        # Also recompute R2
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for key in val_keys:
                inp = embeddings[key].unsqueeze(0)
                pred = model.predict_scores(inp, None)
                target = torch.tensor(
                    labels[key][:NUM_DIMS], dtype=torch.float32
                ).unsqueeze(0)
                all_preds.append(pred)
                all_targets.append(target)

        r2 = suite.regression_r2(torch.cat(all_preds), torch.cat(all_targets))
        r2_per_fold.append(r2)

        print(f"    regression-pairwise={pw['overall']:.4f}, r2={r2:.4f}")
        per_dim_str = ", ".join(
            f"{v:.3f}" for v in pw["per_dimension"].values()
        )
        print(f"    per-dim pairwise: [{per_dim_str}]")

        del model

    pw_mean = sum(pw_per_fold) / len(pw_per_fold)
    r2_mean = sum(r2_per_fold) / len(r2_per_fold)

    print(f"\nFrozen probe (regression-derived pairwise):")
    print(f"  pairwise_mean = {pw_mean:.4f}")
    print(f"  pairwise_per_fold = {pw_per_fold}")
    print(f"  r2_mean = {r2_mean:.4f}")

    # Update results JSON
    with open(RESULTS_PATH) as f:
        results = json.load(f)

    results["frozen_probe"]["pairwise_mean"] = pw_mean
    results["frozen_probe"]["pairwise_per_fold"] = pw_per_fold
    results["frozen_probe"]["pairwise_method"] = "regression_score_diff"
    results["frozen_probe"]["r2_mean"] = r2_mean
    results["frozen_probe"]["r2_per_fold"] = r2_per_fold

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nUpdated {RESULTS_PATH}")


if __name__ == "__main__":
    main()
