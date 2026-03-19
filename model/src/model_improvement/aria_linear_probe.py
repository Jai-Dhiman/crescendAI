"""Linear probe evaluation for Aria frozen embeddings.

Trains a simple Linear(dim, 6) on frozen embeddings across 4-fold CV.
Computes pairwise accuracy, R2, per-dimension breakdown, and error
correlation with MuQ.

Usage:
    python -m model_improvement.aria_linear_probe
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr

from model_improvement.evaluation import aggregate_folds
from model_improvement.metrics import MetricsSuite, format_comparison_table
from model_improvement.taxonomy import DIMENSIONS, NUM_DIMS, load_composite_labels
from paths import Embeddings, Labels

logger = logging.getLogger(__name__)


def compute_pairwise_from_regression(
    predictions: torch.Tensor,
    keys: list[str],
    labels: dict[str, np.ndarray],
    ambiguous_threshold: float = 0.05,
) -> dict:
    """Compute pairwise accuracy from pointwise regression predictions.

    Uses scalar-logit semantics: for each pair (i, j), average pred_diff
    across dims to get a scalar logit, average label_diff across dims to
    get a scalar target. A pair is correct if sign(scalar_logit) ==
    sign(scalar_target). Ambiguous pairs where mean |label_diff| <
    threshold are excluded.

    Also computes per-dimension pairwise accuracy.

    Args:
        predictions: Tensor of shape (n_samples, 6).
        keys: Ordered list of segment IDs matching prediction rows.
        labels: Dict mapping segment_id to numpy array of 6 scores.
        ambiguous_threshold: Min mean |label_diff| to include a pair.

    Returns:
        Dict with "overall", "per_dimension", "n_comparisons",
        "correct_mask", "non_ambiguous_mask".
    """
    n = len(keys)

    pred_diffs = []
    label_diffs = []

    for i in range(n):
        for j in range(i + 1, n):
            pred_diff = predictions[i] - predictions[j]
            lab_a = torch.tensor(
                labels[keys[i]][:NUM_DIMS], dtype=torch.float32,
            )
            lab_b = torch.tensor(
                labels[keys[j]][:NUM_DIMS], dtype=torch.float32,
            )
            pred_diffs.append(pred_diff)
            label_diffs.append(lab_a - lab_b)

    pred_diffs_t = torch.stack(pred_diffs)
    label_diffs_t = torch.stack(label_diffs)

    # Scalar logit: mean across dimensions
    scalar_logit = pred_diffs_t.mean(dim=1)
    scalar_target = label_diffs_t.mean(dim=1)

    # Non-ambiguous mask: mean |label_diff| >= threshold
    non_ambiguous = label_diffs_t.abs().mean(dim=1) >= ambiguous_threshold

    # Overall accuracy (scalar-logit)
    correct = (scalar_logit > 0) == (scalar_target > 0)
    n_non_ambig = non_ambiguous.sum().item()
    if n_non_ambig > 0:
        overall = float(
            (correct & non_ambiguous).sum().item() / n_non_ambig
        )
    else:
        overall = 0.5

    # Per-dimension accuracy
    per_dim: dict[int, float] = {}
    for d in range(NUM_DIMS):
        dim_non_ambig = label_diffs_t[:, d].abs() >= ambiguous_threshold
        if dim_non_ambig.sum() > 0:
            dim_correct = (
                (pred_diffs_t[:, d] > 0) == (label_diffs_t[:, d] > 0)
            )
            per_dim[d] = float(
                (dim_correct & dim_non_ambig).sum().item()
                / dim_non_ambig.sum().item()
            )
        else:
            per_dim[d] = 0.5

    return {
        "overall": overall,
        "per_dimension": per_dim,
        "n_comparisons": int(n_non_ambig),
        "correct_mask": correct,
        "non_ambiguous_mask": non_ambiguous,
    }


def train_linear_probe(
    train_emb: torch.Tensor,
    train_labels: torch.Tensor,
    val_emb: torch.Tensor,
    val_labels: torch.Tensor,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 100,
    patience: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Train a linear probe with early stopping on val MSE.

    Returns:
        Tuple of (val_predictions, train_predictions).
    """
    dim_in = train_emb.shape[1]
    probe = nn.Linear(dim_in, NUM_DIMS)
    nn.init.xavier_uniform_(probe.weight)
    nn.init.zeros_(probe.bias)

    optimizer = torch.optim.Adam(
        probe.parameters(), lr=lr, weight_decay=weight_decay,
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        probe.train()
        optimizer.zero_grad()
        pred = probe(train_emb)
        loss = criterion(pred, train_labels)
        loss.backward()
        optimizer.step()

        probe.eval()
        with torch.no_grad():
            val_pred = probe(val_emb)
            val_loss = criterion(val_pred, val_labels).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                k: v.clone() for k, v in probe.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is not None:
        probe.load_state_dict(best_state)

    probe.eval()
    with torch.no_grad():
        val_preds = probe(val_emb)
        train_preds = probe(train_emb)

    return val_preds, train_preds


def compute_error_correlation(
    correct_a: torch.Tensor,
    correct_b: torch.Tensor,
) -> float:
    """Compute phi coefficient between two binary correct/incorrect vectors.

    Returns:
        Phi coefficient (float). 1.0 = identical errors, 0.0 = independent.
    """
    a = correct_a.float().numpy()
    b = correct_b.float().numpy()
    r, _ = pearsonr(a, b)
    return float(r)


def load_embeddings_as_matrix(
    emb_dict: dict[str, torch.Tensor],
    keys: list[str],
) -> tuple[torch.Tensor, list[str]]:
    """Convert embedding dict to matrix, filtering to available keys.

    Returns:
        Tuple of (embedding_matrix [n, dim], valid_keys [n]).
    """
    valid_keys = [k for k in keys if k in emb_dict]
    if not valid_keys:
        raise ValueError("No keys found in embedding dict")
    matrix = torch.stack([emb_dict[k] for k in valid_keys])
    return matrix, valid_keys


def run_fold_evaluation(
    emb_dict: dict[str, torch.Tensor],
    labels: dict[str, np.ndarray],
    fold: dict[str, list[str]],
    fold_idx: int,
    model_name: str,
    restrict_val_keys: list[str] | None = None,
) -> dict:
    """Run linear probe evaluation for a single fold.

    Args:
        restrict_val_keys: If provided, only use these val keys (for
            ensuring consistent keys across models in error correlation).
    """
    train_keys = fold["train"]
    val_keys = restrict_val_keys if restrict_val_keys else fold["val"]

    train_emb, train_valid = load_embeddings_as_matrix(emb_dict, train_keys)
    val_emb, val_valid = load_embeddings_as_matrix(emb_dict, val_keys)

    train_labels_t = torch.tensor(
        np.array([labels[k] for k in train_valid]), dtype=torch.float32,
    )
    val_labels_t = torch.tensor(
        np.array([labels[k] for k in val_valid]), dtype=torch.float32,
    )

    logger.info(
        "Fold %d [%s]: train=%d, val=%d",
        fold_idx, model_name, len(train_valid), len(val_valid),
    )

    val_preds, _ = train_linear_probe(
        train_emb, train_labels_t, val_emb, val_labels_t,
    )

    pw = compute_pairwise_from_regression(val_preds, val_valid, labels)

    suite = MetricsSuite()
    r2 = suite.regression_r2(val_preds, val_labels_t)

    result = {
        "pairwise": pw["overall"],
        "r2": r2,
        "pairwise_detail": pw,
        "n_comparisons": pw["n_comparisons"],
    }

    for d_idx, d_name in enumerate(DIMENSIONS):
        if d_idx in pw["per_dimension"]:
            result[f"pw_{d_name}"] = pw["per_dimension"][d_idx]

    return result


def run_full_evaluation(
    emb_dict: dict[str, torch.Tensor],
    labels: dict[str, np.ndarray],
    folds: list[dict],
    model_name: str,
    restrict_val_keys_per_fold: list[list[str]] | None = None,
) -> dict:
    """Run linear probe across all folds and aggregate results.

    Args:
        restrict_val_keys_per_fold: If provided, a list of key lists
            (one per fold) to ensure consistent val keys across models.
    """
    torch.manual_seed(42)
    fold_results = []
    all_correct_masks = []
    all_non_ambiguous_masks = []

    for fold_idx, fold in enumerate(folds):
        restrict_keys = (
            restrict_val_keys_per_fold[fold_idx]
            if restrict_val_keys_per_fold
            else None
        )
        result = run_fold_evaluation(
            emb_dict, labels, fold, fold_idx, model_name,
            restrict_val_keys=restrict_keys,
        )
        fold_results.append(result)
        all_correct_masks.append(
            result["pairwise_detail"]["correct_mask"]
        )
        all_non_ambiguous_masks.append(
            result["pairwise_detail"]["non_ambiguous_mask"]
        )

    aggregated = aggregate_folds(fold_results)
    aggregated["fold_results"] = fold_results
    aggregated["all_correct_masks"] = all_correct_masks
    aggregated["all_non_ambiguous_masks"] = all_non_ambiguous_masks
    return aggregated


def mean_pool_muq(
    muq_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Mean-pool MuQ frame-level embeddings to fixed-dim vectors.

    Args:
        muq_dict: {segment_id: tensor[n_frames, 1024]}

    Returns:
        {segment_id: tensor[1024]}
    """
    return {k: v.mean(dim=0) for k, v in muq_dict.items()}


def main() -> None:
    """Run the full Aria validation experiment."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s",
    )

    # Load data
    labels = load_composite_labels(
        Labels.composite / "composite_labels.json",
    )
    with open(Labels.percepiano / "folds.json") as f:
        folds = json.load(f)

    print(f"Loaded {len(labels)} labels, {len(folds)} folds")
    print(
        f"Fold sizes: "
        f"{[(len(f['train']), len(f['val'])) for f in folds]}"
    )
    print()

    results_all = {}

    # Load all embedding dicts to compute shared keys
    emb_dicts: dict[str, dict[str, torch.Tensor]] = {}

    aria_emb_path = Embeddings.percepiano / "aria_embedding.pt"
    if aria_emb_path.exists():
        emb_dicts["Aria-Embedding"] = torch.load(
            aria_emb_path, map_location="cpu", weights_only=True,
        )

    aria_base_path = Embeddings.percepiano / "aria_base.pt"
    if aria_base_path.exists():
        emb_dicts["Aria-Base"] = torch.load(
            aria_base_path, map_location="cpu", weights_only=True,
        )

    muq_path = Embeddings.percepiano / "muq_embeddings.pt"
    if muq_path.exists():
        muq_raw = torch.load(
            muq_path, map_location="cpu", weights_only=False,
        )
        emb_dicts["MuQ"] = mean_pool_muq(muq_raw)

    if not emb_dicts:
        raise FileNotFoundError(
            "No embedding files found. Run aria_embeddings.py first."
        )

    # Compute shared val keys per fold (intersection of all models)
    shared_val_keys_per_fold: list[list[str]] = []
    for fold in folds:
        shared = set(fold["val"])
        for emb_dict in emb_dicts.values():
            shared &= set(emb_dict.keys())
        shared &= set(labels.keys())
        shared_val_keys_per_fold.append(sorted(shared))

    print(
        f"Shared val keys per fold: "
        f"{[len(k) for k in shared_val_keys_per_fold]}"
    )
    print()

    # --- Each model ---
    for name, emb_dict in emb_dicts.items():
        dim = next(iter(emb_dict.values())).shape[0]
        print(f"=== {name} ({dim}-dim) ===")
        results = run_full_evaluation(
            emb_dict, labels, folds, name,
            restrict_val_keys_per_fold=shared_val_keys_per_fold,
        )
        results_all[name] = results
        print(
            f"  Pairwise: {results['pairwise_mean']:.4f} "
            f"+/- {results['pairwise_std']:.4f}"
        )
        print(
            f"  R2:       {results['r2_mean']:.4f} "
            f"+/- {results['r2_std']:.4f}"
        )
        for d in DIMENSIONS:
            key = f"pw_{d}_mean"
            if key in results:
                print(f"  {d:>15s}: {results[key]:.4f}")
        print()

    # --- Comparison Table ---
    if len(results_all) > 1:
        print("=== Comparison Table ===")
        table_data = {}
        for name, res in results_all.items():
            table_data[name] = {
                "pairwise": res.get("pairwise_mean", 0.0),
                "r2": res.get("r2_mean", 0.0),
            }
            for d in DIMENSIONS:
                key = f"pw_{d}_mean"
                if key in res:
                    table_data[name][f"pw_{d}"] = res[key]
        print(format_comparison_table(table_data))
        print()

    # --- Error Correlation ---
    if len(results_all) >= 2 and "MuQ" in results_all:
        print("=== Error Correlation (MuQ vs best Aria) ===")
        aria_names = [n for n in results_all if n != "MuQ"]
        best_aria_name = max(
            aria_names,
            key=lambda n: results_all[n].get("pairwise_mean", 0.0),
        )
        best_aria = results_all[best_aria_name]
        muq_res = results_all["MuQ"]

        all_phi = []
        for fold_idx in range(len(folds)):
            aria_correct = best_aria["all_correct_masks"][fold_idx]
            muq_correct = muq_res["all_correct_masks"][fold_idx]
            aria_na = best_aria["all_non_ambiguous_masks"][fold_idx]
            muq_na = muq_res["all_non_ambiguous_masks"][fold_idx]

            shared_mask = aria_na & muq_na
            if shared_mask.sum() > 0:
                phi = compute_error_correlation(
                    aria_correct[shared_mask],
                    muq_correct[shared_mask],
                )
                all_phi.append(phi)

        if all_phi:
            mean_phi = float(np.mean(all_phi))
            std_phi = float(np.std(all_phi))
            print(f"  Best Aria: {best_aria_name}")
            print(f"  Phi coefficient: {mean_phi:.4f} +/- {std_phi:.4f}")
            if mean_phi < 0.50:
                print(
                    "  -> Models make DIFFERENT mistakes "
                    "-> fusion promising"
                )
            elif mean_phi < 0.70:
                print(
                    "  -> MODERATE overlap "
                    "-> fusion may help"
                )
            else:
                print(
                    "  -> Models are REDUNDANT "
                    "-> fusion unlikely to help"
                )
        print()

    # --- Decision ---
    print("=== Decision ===")
    for name, res in results_all.items():
        pw = res.get("pairwise_mean", 0.0)
        if pw > 0.60:
            print(
                f"  {name}: {pw:.4f} "
                f"-- QUALITY SIGNAL CONFIRMED (>60%)"
            )
        elif pw > 0.55:
            print(
                f"  {name}: {pw:.4f} "
                f"-- MARGINAL (55-60%), try LoRA fine-tuning"
            )
        else:
            print(
                f"  {name}: {pw:.4f} "
                f"-- NO QUALITY SIGNAL (~50%)"
            )

    # Save results
    from paths import Results
    results_path = Results.root / "aria_validation.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    save_data = {}
    for name, res in results_all.items():
        save_data[name] = {
            k: v for k, v in res.items()
            if k not in (
                "fold_results",
                "all_correct_masks",
                "all_non_ambiguous_masks",
            )
        }
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
