"""Evaluation utilities for masterclass priority signal experiment."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from masterclass_experiments.models import train_classifier


def leave_one_video_out_cv(
    X: np.ndarray,
    y: np.ndarray,
    video_ids: np.ndarray,
    segment_ids: np.ndarray | None = None,
) -> dict:
    """Leave-one-video-out cross-validation.

    Args:
        X: Feature matrix [N, D].
        y: Binary labels [N].
        video_ids: Video ID per sample [N].
        segment_ids: Optional segment IDs for qualitative analysis [N].

    Returns:
        Dict with aggregate metrics and per-segment predictions.
    """
    unique_videos = np.unique(video_ids)
    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    per_segment = []

    for held_out_video in unique_videos:
        test_mask = video_ids == held_out_video
        train_mask = ~test_mask

        train_idx = np.where(train_mask)[0].tolist()
        test_idx = np.where(test_mask)[0].tolist()

        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        # Check both classes exist in training set
        if len(np.unique(y[train_mask])) < 2:
            continue

        result = train_classifier(X, y, train_idx, test_idx)

        all_y_true.extend(result["y_true"])
        all_y_pred.extend(result["y_pred"])
        all_y_proba.extend(result["y_pred_proba"])

        for i, idx in enumerate(test_idx):
            per_segment.append({
                "video_id": video_ids[idx],
                "segment_id": segment_ids[idx] if segment_ids is not None else str(idx),
                "y_true": int(result["y_true"][i]),
                "y_pred_proba": float(result["y_pred_proba"][i]),
            })

    all_y_true_arr = np.array(all_y_true)
    all_y_pred_arr = np.array(all_y_pred)
    all_y_proba_arr = np.array(all_y_proba)

    # Compute aggregate metrics
    has_both_classes = len(np.unique(all_y_true_arr)) == 2
    auc = float(roc_auc_score(all_y_true_arr, all_y_proba_arr)) if has_both_classes else 0.5
    accuracy = float(accuracy_score(all_y_true_arr, all_y_pred_arr))
    precision = float(precision_score(all_y_true_arr, all_y_pred_arr, zero_division=0))
    recall = float(recall_score(all_y_true_arr, all_y_pred_arr, zero_division=0))

    return {
        "auc": auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "n_samples": len(all_y_true_arr),
        "n_stop": int(all_y_true_arr.sum()),
        "n_continue": int((1 - all_y_true_arr).sum()),
        "per_segment": per_segment,
    }
