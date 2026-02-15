"""Diagnostic utilities for validating Model B methodology."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from masterclass_experiments.models import train_classifier


def permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    video_ids: np.ndarray,
    n_permutations: int = 1000,
    random_state: int = 42,
) -> dict:
    """Permutation test for LOVO cross-validation AUC.

    Shuffles labels while preserving video structure, re-runs CV each time.
    If observed AUC is significantly higher than the null distribution,
    the signal is real.

    Args:
        X: Feature matrix [N, D].
        y: Binary labels [N].
        video_ids: Video ID per sample [N].
        n_permutations: Number of permutations to run.
        random_state: Random seed for reproducibility.

    Returns:
        Dict with observed_auc, p_value, null_distribution.
    """
    observed_auc = _lovo_auc(X, y, video_ids)

    rng = np.random.default_rng(random_state)
    null_aucs = []
    for _ in range(n_permutations):
        shuffled_y = rng.permutation(y)
        null_auc = _lovo_auc(X, shuffled_y, video_ids)
        null_aucs.append(null_auc)

    null_aucs = np.array(null_aucs)
    p_value = float(np.mean(null_aucs >= observed_auc))

    return {
        "observed_auc": observed_auc,
        "p_value": p_value,
        "null_distribution": null_aucs,
    }


def per_fold_metrics(
    X: np.ndarray,
    y: np.ndarray,
    video_ids: np.ndarray,
) -> list[dict]:
    """Compute per-fold metrics for LOVO cross-validation.

    Returns detailed results for each held-out video, including
    fold-level AUC and coefficients.

    Args:
        X: Feature matrix [N, D].
        y: Binary labels [N].
        video_ids: Video ID per sample [N].

    Returns:
        List of dicts, one per fold.
    """
    unique_videos = np.unique(video_ids)
    folds = []

    for held_out_video in unique_videos:
        test_mask = video_ids == held_out_video
        train_mask = ~test_mask
        train_idx = np.where(train_mask)[0].tolist()
        test_idx = np.where(test_mask)[0].tolist()

        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        if len(np.unique(y[train_mask])) < 2:
            continue

        result = train_classifier(X, y, train_idx, test_idx)

        has_both = len(np.unique(result["y_true"])) == 2
        auc = float(roc_auc_score(result["y_true"], result["y_pred_proba"])) if has_both else float("nan")

        folds.append({
            "held_out_video": held_out_video,
            "auc": auc,
            "accuracy": float(np.mean(result["y_true"] == result["y_pred"])),
            "n_test": len(test_idx),
            "n_train": len(train_idx),
            "n_stop_test": int(result["y_true"].sum()),
            "n_continue_test": int((1 - result["y_true"]).sum()),
            "coefficients": result["coefficients"],
        })

    return folds


def coefficient_stability(
    X: np.ndarray,
    y: np.ndarray,
    video_ids: np.ndarray,
) -> dict:
    """Analyze coefficient stability across LOVO folds.

    Stable coefficients (same sign, similar magnitude across folds) indicate
    the model is learning a real signal. Unstable coefficients suggest
    overfitting to specific videos.

    Args:
        X: Feature matrix [N, D].
        y: Binary labels [N].
        video_ids: Video ID per sample [N].

    Returns:
        Dict with mean_coefs, std_coefs, sign_consistency per feature.
    """
    folds = per_fold_metrics(X, y, video_ids)
    all_coefs = np.stack([f["coefficients"] for f in folds])  # [n_folds, D]

    mean_coefs = all_coefs.mean(axis=0)
    std_coefs = all_coefs.std(axis=0)

    # Sign consistency: fraction of folds that agree on the sign of the mean
    mean_signs = np.sign(mean_coefs)
    fold_signs = np.sign(all_coefs)
    sign_consistency = np.mean(fold_signs == mean_signs, axis=0)

    return {
        "mean_coefs": mean_coefs,
        "std_coefs": std_coefs,
        "sign_consistency": sign_consistency,
        "per_fold_coefs": all_coefs,
    }


def _lovo_auc(X: np.ndarray, y: np.ndarray, video_ids: np.ndarray) -> float:
    """Compute aggregate AUC from leave-one-video-out CV."""
    unique_videos = np.unique(video_ids)
    all_y_true = []
    all_y_proba = []

    for held_out_video in unique_videos:
        test_mask = video_ids == held_out_video
        train_mask = ~test_mask
        train_idx = np.where(train_mask)[0].tolist()
        test_idx = np.where(test_mask)[0].tolist()

        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        if len(np.unique(y[train_mask])) < 2:
            continue

        result = train_classifier(X, y, train_idx, test_idx)
        all_y_true.extend(result["y_true"])
        all_y_proba.extend(result["y_pred_proba"])

    if len(np.unique(all_y_true)) < 2:
        return 0.5

    return float(roc_auc_score(all_y_true, all_y_proba))
