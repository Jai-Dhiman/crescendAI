"""Evaluation metrics for score-performance alignment.

Computes onset error metrics to evaluate alignment quality against
ground truth note alignments.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import MUQ_FRAME_RATE, ONSET_ERROR_THRESHOLD_MS


def frame_to_time(frame_idx: int, frame_rate: float = MUQ_FRAME_RATE) -> float:
    """Convert frame index to time in seconds.

    Args:
        frame_idx: Frame index.
        frame_rate: Frame rate in frames per second.

    Returns:
        Time in seconds.
    """
    return frame_idx / frame_rate


def time_to_frame(time_sec: float, frame_rate: float = MUQ_FRAME_RATE) -> int:
    """Convert time in seconds to frame index.

    Args:
        time_sec: Time in seconds.
        frame_rate: Frame rate in frames per second.

    Returns:
        Frame index (rounded).
    """
    return int(round(time_sec * frame_rate))


def onset_error(
    predicted_onsets: np.ndarray,
    ground_truth_onsets: np.ndarray,
    threshold_ms: float = ONSET_ERROR_THRESHOLD_MS,
) -> Dict[str, float]:
    """Compute onset error metrics between predicted and ground truth.

    Args:
        predicted_onsets: Predicted onset times in seconds.
        ground_truth_onsets: Ground truth onset times in seconds.
        threshold_ms: Threshold in milliseconds for "correct" classification.

    Returns:
        Dict with metrics:
            - mean_error_ms: Mean absolute error in milliseconds
            - median_error_ms: Median absolute error in milliseconds
            - std_error_ms: Standard deviation of error in milliseconds
            - max_error_ms: Maximum absolute error in milliseconds
            - percent_within_threshold: Percentage of onsets within threshold
            - rmse_ms: Root mean squared error in milliseconds
    """
    if len(predicted_onsets) != len(ground_truth_onsets):
        raise ValueError(
            f"Length mismatch: predicted={len(predicted_onsets)}, "
            f"ground_truth={len(ground_truth_onsets)}"
        )

    if len(predicted_onsets) == 0:
        return {
            "mean_error_ms": 0.0,
            "median_error_ms": 0.0,
            "std_error_ms": 0.0,
            "max_error_ms": 0.0,
            "percent_within_threshold": 100.0,
            "rmse_ms": 0.0,
        }

    # Compute errors in milliseconds
    errors_sec = np.abs(predicted_onsets - ground_truth_onsets)
    errors_ms = errors_sec * 1000

    # Metrics
    threshold_sec = threshold_ms / 1000
    within_threshold = np.sum(errors_sec <= threshold_sec)
    percent_within = 100.0 * within_threshold / len(errors_sec)

    return {
        "mean_error_ms": float(np.mean(errors_ms)),
        "median_error_ms": float(np.median(errors_ms)),
        "std_error_ms": float(np.std(errors_ms)),
        "max_error_ms": float(np.max(errors_ms)),
        "percent_within_threshold": float(percent_within),
        "rmse_ms": float(np.sqrt(np.mean(errors_ms ** 2))),
    }


def evaluate_dtw_alignment(
    score_onsets: np.ndarray,
    perf_onsets: np.ndarray,
    dtw_path_score: np.ndarray,
    dtw_path_perf: np.ndarray,
    frame_rate: float = MUQ_FRAME_RATE,
    threshold_ms: float = ONSET_ERROR_THRESHOLD_MS,
) -> Dict[str, float]:
    """Evaluate DTW alignment against ground truth onset pairs.

    Maps ground truth score onsets through the DTW path to get predicted
    performance onsets, then computes error against ground truth.

    Args:
        score_onsets: Ground truth score onset times (seconds).
        perf_onsets: Ground truth performance onset times (seconds).
        dtw_path_score: Score frame indices from DTW alignment.
        dtw_path_perf: Performance frame indices from DTW alignment.
        frame_rate: Frame rate of embeddings.
        threshold_ms: Error threshold for accuracy computation.

    Returns:
        Dict with onset error metrics.
    """
    if len(score_onsets) == 0:
        return {
            "mean_error_ms": 0.0,
            "median_error_ms": 0.0,
            "std_error_ms": 0.0,
            "max_error_ms": 0.0,
            "percent_within_threshold": 100.0,
            "rmse_ms": 0.0,
            "num_notes": 0,
        }

    # Convert ground truth onsets to frames
    score_frames = np.array([time_to_frame(t, frame_rate) for t in score_onsets])

    # For each score onset, find the corresponding performance frame via DTW path
    predicted_perf_times = []

    for score_frame in score_frames:
        # Find closest frame in DTW path
        path_distances = np.abs(dtw_path_score - score_frame)
        closest_idx = np.argmin(path_distances)
        perf_frame = dtw_path_perf[closest_idx]
        predicted_perf_times.append(frame_to_time(perf_frame, frame_rate))

    predicted_perf_times = np.array(predicted_perf_times)

    # Compute error metrics
    metrics = onset_error(predicted_perf_times, perf_onsets, threshold_ms)
    metrics["num_notes"] = len(score_onsets)

    return metrics


def compute_alignment_summary(
    all_metrics: List[Dict[str, float]],
) -> Dict[str, float]:
    """Aggregate alignment metrics across multiple performances.

    Args:
        all_metrics: List of metric dicts from evaluate_dtw_alignment.

    Returns:
        Dict with aggregated statistics.
    """
    if not all_metrics:
        return {}

    # Collect values
    mean_errors = [m["mean_error_ms"] for m in all_metrics if "mean_error_ms" in m]
    median_errors = [m["median_error_ms"] for m in all_metrics if "median_error_ms" in m]
    within_thresholds = [
        m["percent_within_threshold"]
        for m in all_metrics
        if "percent_within_threshold" in m
    ]
    num_notes = [m.get("num_notes", 0) for m in all_metrics]

    total_notes = sum(num_notes)

    # Compute weighted averages (by number of notes)
    if total_notes > 0:
        weighted_mean = sum(
            m["mean_error_ms"] * m.get("num_notes", 1)
            for m in all_metrics
            if "mean_error_ms" in m
        ) / total_notes
        weighted_within = sum(
            m["percent_within_threshold"] * m.get("num_notes", 1)
            for m in all_metrics
            if "percent_within_threshold" in m
        ) / total_notes
    else:
        weighted_mean = 0.0
        weighted_within = 0.0

    return {
        # Per-performance statistics
        "mean_of_means_ms": float(np.mean(mean_errors)) if mean_errors else 0.0,
        "std_of_means_ms": float(np.std(mean_errors)) if mean_errors else 0.0,
        "mean_of_medians_ms": float(np.mean(median_errors)) if median_errors else 0.0,
        "mean_percent_within_threshold": (
            float(np.mean(within_thresholds)) if within_thresholds else 0.0
        ),
        # Weighted by number of notes
        "weighted_mean_error_ms": float(weighted_mean),
        "weighted_percent_within_threshold": float(weighted_within),
        # Counts
        "num_performances": len(all_metrics),
        "total_notes": total_notes,
    }


def compute_frame_accuracy(
    predicted_mapping: np.ndarray,
    ground_truth_mapping: np.ndarray,
    tolerance_frames: int = 2,
) -> float:
    """Compute frame-level alignment accuracy.

    Args:
        predicted_mapping: Predicted frame mapping from score to performance.
        ground_truth_mapping: Ground truth frame mapping.
        tolerance_frames: Number of frames tolerance for "correct" prediction.

    Returns:
        Accuracy as percentage (0-100).
    """
    if len(predicted_mapping) != len(ground_truth_mapping):
        raise ValueError("Mapping lengths must match")

    if len(predicted_mapping) == 0:
        return 100.0

    errors = np.abs(predicted_mapping - ground_truth_mapping)
    correct = np.sum(errors <= tolerance_frames)

    return 100.0 * correct / len(predicted_mapping)


def compute_monotonicity_score(
    dtw_path_perf: np.ndarray,
) -> float:
    """Compute how monotonic the alignment path is.

    A perfectly monotonic path has score = 1.0.
    Lower scores indicate more back-tracking in the alignment.

    Args:
        dtw_path_perf: Performance frame indices from DTW path.

    Returns:
        Monotonicity score (0-1).
    """
    if len(dtw_path_perf) <= 1:
        return 1.0

    diffs = np.diff(dtw_path_perf)
    monotonic_steps = np.sum(diffs >= 0)

    return monotonic_steps / len(diffs)
