"""Alignment algorithms and evaluation metrics.

This submodule provides:
    - Standard DTW for baseline alignment
    - Differentiable soft-DTW loss for training
    - Onset error metrics for evaluation
"""

from .dtw import (
    compute_cost_matrix,
    dtw_alignment,
    align_embeddings,
    path_to_frame_mapping,
)
from .soft_dtw import SoftDTWLoss, SoftDTWDivergence
from .metrics import (
    onset_error,
    frame_to_time,
    time_to_frame,
    evaluate_dtw_alignment,
    compute_alignment_summary,
)
