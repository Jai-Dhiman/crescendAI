from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score, silhouette_score
from tqdm.auto import tqdm


def compute_pairwise_accuracy(
    predictions_a: np.ndarray,
    predictions_b: np.ndarray,
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    ambiguous_threshold: float = 0.05,
) -> Dict:
    # True ranking direction
    label_diff = labels_a - labels_b
    pred_diff = predictions_a - predictions_b

    # Binary predictions
    true_ranking = label_diff > 0
    pred_ranking = pred_diff > 0

    # Mask for non-ambiguous pairs
    non_ambiguous = np.abs(label_diff) >= ambiguous_threshold

    # Overall accuracy
    if non_ambiguous.any():
        correct = (true_ranking == pred_ranking) & non_ambiguous
        overall_acc = correct.sum() / non_ambiguous.sum()
    else:
        overall_acc = 0.5

    # Per-dimension accuracy
    per_dim = {}
    for d in range(19):
        dim_mask = non_ambiguous[:, d]
        if dim_mask.any():
            dim_correct = (true_ranking[:, d] == pred_ranking[:, d]) & dim_mask
            per_dim[d] = dim_correct.sum() / dim_mask.sum()
        else:
            per_dim[d] = 0.5

    return {
        "overall_accuracy": float(overall_acc),
        "per_dimension": {k: float(v) for k, v in per_dim.items()},
        "n_pairs": int(non_ambiguous.any(axis=1).sum()),
        "n_comparisons": int(non_ambiguous.sum()),
    }


def compute_pairwise_metrics(
    ranking_logits: np.ndarray,
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    ambiguous_threshold: float = 0.05,
) -> Dict:
    label_diff = labels_a - labels_b

    # Convert logits to binary predictions
    pred_ranking = ranking_logits > 0
    true_ranking = label_diff > 0

    # Non-ambiguous mask
    non_ambiguous = np.abs(label_diff) >= ambiguous_threshold

    if not non_ambiguous.any():
        return {
            "overall_accuracy": 0.5,
            "per_dimension": {d: 0.5 for d in range(19)},
            "n_comparisons": 0,
        }

    # Overall
    correct = (pred_ranking == true_ranking) & non_ambiguous
    overall_acc = correct.sum() / non_ambiguous.sum()

    # Per-dimension
    per_dim = {}
    for d in range(19):
        mask = non_ambiguous[:, d]
        if mask.any():
            dim_correct = ((pred_ranking[:, d] == true_ranking[:, d]) & mask).sum()
            per_dim[d] = float(dim_correct / mask.sum())
        else:
            per_dim[d] = 0.5

    # Concordance (Kendall's tau-like)
    # For each pair, check if predicted ranking matches true ranking
    concordant = correct.sum()
    discordant = (non_ambiguous & ~correct).sum()
    total = concordant + discordant
    kendall_tau = (concordant - discordant) / total if total > 0 else 0

    return {
        "overall_accuracy": float(overall_acc),
        "per_dimension": per_dim,
        "n_comparisons": int(non_ambiguous.sum()),
        "kendall_tau": float(kendall_tau),
    }


def compute_intra_piece_std(
    predictions: np.ndarray,
    piece_ids: np.ndarray,
) -> Dict:
    unique_pieces = np.unique(piece_ids)
    piece_stds = []
    per_dim_stds = {d: [] for d in range(19)}

    for pid in unique_pieces:
        mask = piece_ids == pid
        if mask.sum() > 1:
            piece_preds = predictions[mask]
            # Overall std
            piece_stds.append(piece_preds.std())
            # Per-dimension std
            for d in range(19):
                per_dim_stds[d].append(piece_preds[:, d].std())

    return {
        "mean_intra_piece_std": float(np.mean(piece_stds)) if piece_stds else 0.0,
        "per_dimension": {
            d: float(np.mean(stds)) if stds else 0.0 for d, stds in per_dim_stds.items()
        },
        "n_multi_performer_pieces": len([s for s in piece_stds if s > 0]),
    }


def evaluate_disentanglement(
    z_style: np.ndarray,
    z_piece: np.ndarray,
    piece_ids: np.ndarray,
    predictions: np.ndarray,
) -> Dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    # 1. Piece classification from z_style (should be near chance if disentangled)
    try:
        clf = LogisticRegression(max_iter=1000, multi_class="multinomial", n_jobs=-1)
        style_piece_acc = cross_val_score(
            clf, z_style, piece_ids, cv=3, scoring="accuracy"
        ).mean()
    except Exception:
        style_piece_acc = 0.0

    # 2. Piece classification from z_piece (should be high)
    try:
        piece_piece_acc = cross_val_score(
            clf, z_piece, piece_ids, cv=3, scoring="accuracy"
        ).mean()
    except Exception:
        piece_piece_acc = 0.0

    # 3. Within-piece variance of style embeddings (should be high)
    unique_pieces = np.unique(piece_ids)
    within_vars = []
    for pid in unique_pieces:
        mask = piece_ids == pid
        if mask.sum() > 1:
            var = z_style[mask].var(axis=0).mean()
            within_vars.append(var)
    within_piece_var = float(np.mean(within_vars)) if within_vars else 0.0

    # 4. Silhouette scores
    n_unique = len(unique_pieces)
    if n_unique > 1 and len(piece_ids) > n_unique:
        try:
            sample_size = min(1000, len(piece_ids))
            piece_silhouette = silhouette_score(
                z_piece, piece_ids, sample_size=sample_size
            )
            style_silhouette = silhouette_score(
                z_style, piece_ids, sample_size=sample_size
            )
        except Exception:
            piece_silhouette = 0.0
            style_silhouette = 0.0
    else:
        piece_silhouette = 0.0
        style_silhouette = 0.0

    # 5. Intra-piece prediction std
    intra_piece = compute_intra_piece_std(predictions, piece_ids)

    return {
        "style_piece_accuracy": float(style_piece_acc),
        "piece_piece_accuracy": float(piece_piece_acc),
        "within_piece_variance": within_piece_var,
        "piece_silhouette": float(piece_silhouette),
        "style_silhouette": float(style_silhouette),
        "intra_piece_std": intra_piece["mean_intra_piece_std"],
    }


def bootstrap_pairwise_accuracy(
    ranking_logits: np.ndarray,
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    np.random.seed(seed)
    n_samples = len(ranking_logits)

    label_diff = labels_a - labels_b
    pred_ranking = ranking_logits > 0
    true_ranking = label_diff > 0
    non_ambiguous = np.abs(label_diff) >= 0.05

    if not non_ambiguous.any():
        return (0.5, 0.5, 0.5)

    accs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n_samples, n_samples, replace=True)
        mask = non_ambiguous[idx]
        if mask.any():
            correct = (pred_ranking[idx] == true_ranking[idx]) & mask
            acc = correct.sum() / mask.sum()
            accs.append(acc)

    return tuple(np.percentile(accs, [2.5, 50, 97.5]))


def compute_regression_pairwise_accuracy(
    model: pl.LightningModule,
    cache_dir: Path,
    labels: Dict,
    piece_to_keys: Dict[str, List[str]],
    device: str = "cuda",
    ambiguous_threshold: float = 0.05,
    max_frames: int = 1000,
) -> Dict:
    cache_dir = Path(cache_dir)
    model = model.to(device)
    model.eval()

    # Get all available keys
    available = {p.stem for p in cache_dir.glob("*.pt")}

    # Collect predictions for all samples
    key_to_pred: Dict[str, np.ndarray] = {}
    key_to_label: Dict[str, np.ndarray] = {}

    all_keys = set()
    for keys in piece_to_keys.values():
        all_keys.update(keys)
    all_keys = all_keys & available & set(labels.keys())

    with torch.no_grad():
        for key in tqdm(sorted(all_keys), desc="Predicting"):
            emb = torch.load(cache_dir / f"{key}.pt", weights_only=True)
            if emb.shape[0] > max_frames:
                emb = emb[:max_frames]

            # Add batch dimension
            emb = emb.unsqueeze(0).to(device)
            mask = torch.ones(1, emb.shape[1], dtype=torch.bool, device=device)

            # Forward pass
            pred = model(emb, mask)
            key_to_pred[key] = pred.squeeze(0).cpu().numpy()
            key_to_label[key] = np.array(labels[key][:19])

    # Compute pairwise accuracy for each piece
    all_correct = []
    all_non_ambiguous = []
    per_dim_correct = {d: [] for d in range(19)}
    per_dim_non_ambiguous = {d: [] for d in range(19)}
    per_piece_acc = {}

    for piece_id, keys in piece_to_keys.items():
        valid_keys = [k for k in keys if k in key_to_pred]
        if len(valid_keys) < 2:
            continue

        piece_correct = 0
        piece_total = 0

        for i, k_a in enumerate(valid_keys):
            for k_b in valid_keys[i + 1 :]:
                pred_a = key_to_pred[k_a]
                pred_b = key_to_pred[k_b]
                label_a = key_to_label[k_a]
                label_b = key_to_label[k_b]

                # True ranking direction
                label_diff = label_a - label_b
                pred_diff = pred_a - pred_b

                true_ranking = label_diff > 0
                pred_ranking = pred_diff > 0

                # Non-ambiguous mask
                non_ambiguous = np.abs(label_diff) >= ambiguous_threshold

                # Per-dimension metrics
                for d in range(19):
                    if non_ambiguous[d]:
                        correct = true_ranking[d] == pred_ranking[d]
                        per_dim_correct[d].append(correct)
                        per_dim_non_ambiguous[d].append(True)
                        all_correct.append(correct)
                        all_non_ambiguous.append(True)

                        if correct:
                            piece_correct += 1
                        piece_total += 1

        if piece_total > 0:
            per_piece_acc[piece_id] = piece_correct / piece_total

    # Compute overall accuracy
    if all_correct:
        overall_acc = sum(all_correct) / len(all_correct)
    else:
        overall_acc = 0.5

    # Compute per-dimension accuracy
    per_dim_acc = {}
    for d in range(19):
        if per_dim_correct[d]:
            per_dim_acc[d] = sum(per_dim_correct[d]) / len(per_dim_correct[d])
        else:
            per_dim_acc[d] = 0.5

    return {
        "overall_accuracy": float(overall_acc),
        "per_dimension": {k: float(v) for k, v in per_dim_acc.items()},
        "n_pairs": len(all_correct) // 19 if all_correct else 0,
        "n_comparisons": len(all_correct),
        "per_piece_accuracy": per_piece_acc,
    }
