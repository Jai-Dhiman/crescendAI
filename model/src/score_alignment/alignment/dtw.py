"""Standard Dynamic Time Warping for sequence alignment.

Provides DTW-based alignment between score and performance embeddings
using various distance metrics.
"""

from typing import Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist


def compute_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    metric: str = "cosine",
) -> np.ndarray:
    """Compute pairwise cost matrix between two sequences.

    Args:
        x: First sequence of shape [T1, D].
        y: Second sequence of shape [T2, D].
        metric: Distance metric ("cosine", "euclidean", "sqeuclidean").

    Returns:
        Cost matrix of shape [T1, T2].
    """
    if metric == "cosine":
        # scipy cdist returns cosine distance = 1 - cosine_similarity
        return cdist(x, y, metric="cosine")
    elif metric == "euclidean":
        return cdist(x, y, metric="euclidean")
    elif metric == "sqeuclidean":
        return cdist(x, y, metric="sqeuclidean")
    else:
        raise ValueError(f"Unknown metric: {metric}")


def dtw_alignment(
    cost_matrix: np.ndarray,
    sakoe_chiba_radius: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute DTW alignment path using dynamic programming.

    Args:
        cost_matrix: Cost matrix of shape [T1, T2].
        sakoe_chiba_radius: If provided, constrains alignment to a band
            of this radius around the diagonal. Reduces complexity from
            O(T1*T2) to O(T1*radius).

    Returns:
        Tuple of (path_x, path_y, total_cost) where path_x and path_y
        are arrays of indices along each sequence forming the alignment.
    """
    T1, T2 = cost_matrix.shape

    # Initialize accumulated cost matrix
    D = np.full((T1 + 1, T2 + 1), np.inf)
    D[0, 0] = 0

    # Fill cost matrix
    for i in range(1, T1 + 1):
        # Apply Sakoe-Chiba band constraint if specified
        if sakoe_chiba_radius is not None:
            # Compute diagonal position for this row
            diag = int(round(i * T2 / T1))
            j_start = max(1, diag - sakoe_chiba_radius)
            j_end = min(T2 + 1, diag + sakoe_chiba_radius + 1)
        else:
            j_start = 1
            j_end = T2 + 1

        for j in range(j_start, j_end):
            cost = cost_matrix[i - 1, j - 1]
            D[i, j] = cost + min(
                D[i - 1, j],      # insertion
                D[i, j - 1],      # deletion
                D[i - 1, j - 1],  # match
            )

    # Backtrack to find optimal path
    path_x = []
    path_y = []
    i, j = T1, T2

    while i > 0 or j > 0:
        path_x.append(i - 1)
        path_y.append(j - 1)

        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            # Choose minimum cost predecessor
            candidates = [
                (D[i - 1, j], i - 1, j),
                (D[i, j - 1], i, j - 1),
                (D[i - 1, j - 1], i - 1, j - 1),
            ]
            _, i, j = min(candidates, key=lambda x: x[0])

    # Reverse to get forward path
    path_x = np.array(path_x[::-1])
    path_y = np.array(path_y[::-1])

    total_cost = D[T1, T2]

    return path_x, path_y, total_cost


def align_embeddings(
    score_emb: np.ndarray,
    perf_emb: np.ndarray,
    metric: str = "cosine",
    sakoe_chiba_radius: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Align score and performance embeddings using DTW.

    Args:
        score_emb: Score embeddings of shape [T1, D].
        perf_emb: Performance embeddings of shape [T2, D].
        metric: Distance metric for cost computation.
        sakoe_chiba_radius: Optional band constraint.

    Returns:
        Tuple of (path_score, path_perf, cost, cost_matrix).
    """
    cost_matrix = compute_cost_matrix(score_emb, perf_emb, metric)
    path_score, path_perf, cost = dtw_alignment(cost_matrix, sakoe_chiba_radius)

    return path_score, path_perf, cost, cost_matrix


def path_to_frame_mapping(
    path_x: np.ndarray,
    path_y: np.ndarray,
    num_frames_x: int,
) -> np.ndarray:
    """Convert DTW path to frame-level mapping from x to y.

    For each frame in sequence x, returns the corresponding frame in y.
    When multiple y frames map to the same x frame, uses the mean.

    Args:
        path_x: Indices in first sequence from DTW path.
        path_y: Indices in second sequence from DTW path.
        num_frames_x: Total number of frames in first sequence.

    Returns:
        Array of shape [num_frames_x] with mapping to y indices.
    """
    mapping = np.zeros(num_frames_x)
    counts = np.zeros(num_frames_x)

    for px, py in zip(path_x, path_y):
        if 0 <= px < num_frames_x:
            mapping[px] += py
            counts[px] += 1

    # Average for frames with multiple mappings
    valid = counts > 0
    mapping[valid] /= counts[valid]

    # Interpolate missing frames (shouldn't happen with standard DTW)
    if not np.all(valid):
        valid_indices = np.where(valid)[0]
        if len(valid_indices) > 0:
            mapping = np.interp(
                np.arange(num_frames_x),
                valid_indices,
                mapping[valid_indices],
            )

    return mapping


def get_alignment_path_times(
    path_x: np.ndarray,
    path_y: np.ndarray,
    frame_rate: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert frame indices in alignment path to time values.

    Args:
        path_x: Frame indices for first sequence.
        path_y: Frame indices for second sequence.
        frame_rate: Frame rate (frames per second).

    Returns:
        Tuple of (times_x, times_y) in seconds.
    """
    times_x = path_x / frame_rate
    times_y = path_y / frame_rate

    return times_x, times_y


def compute_warping_factor(
    path_x: np.ndarray,
    path_y: np.ndarray,
) -> np.ndarray:
    """Compute local warping factor along the alignment path.

    Warping factor > 1 means performance is slower than score at that point.
    Warping factor < 1 means performance is faster.

    Args:
        path_x: Score frame indices from DTW path.
        path_y: Performance frame indices from DTW path.

    Returns:
        Array of local warping factors for each path point.
    """
    # Compute local derivatives
    dx = np.diff(path_x, prepend=path_x[0])
    dy = np.diff(path_y, prepend=path_y[0])

    # Avoid division by zero
    dx = np.where(dx == 0, 1, dx)

    # Warping factor is dy/dx
    warping = dy / dx

    return warping
