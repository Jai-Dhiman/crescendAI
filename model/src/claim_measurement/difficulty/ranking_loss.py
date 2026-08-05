"""Pairwise ranking + cumulative-link ordinal auxiliary loss for #138 Phase 1
LoRA training. Pure torch, CPU-testable -- factored out of train_fold.py
precisely so this offline suite can reach it without a GPU (see design spec's
Modules section for the rationale: the gate metric is Kendall tau-c, a rank
correlation, so pairwise ranking is the primary objective; a low-weight
ordinal auxiliary only pins the score scale, which pure pairwise loss does
not constrain).
"""
from __future__ import annotations

import torch


def ordered_pairs(grades: torch.Tensor) -> torch.Tensor:
    """All (i, j) index pairs within one batch where grades[i] > grades[j].
    Returns an (n_pairs, 2) int64 tensor; shape (0, 2) when no such pair
    exists (e.g. every piece in the batch shares one grade), never raises."""
    n = grades.shape[0]
    gi = grades.unsqueeze(1).expand(n, n)
    gj = grades.unsqueeze(0).expand(n, n)
    mask = gi > gj
    return mask.nonzero(as_tuple=False)


def pairwise_ranking_loss(scores: torch.Tensor, grades: torch.Tensor) -> torch.Tensor:
    """Pairwise logistic ranking loss: -log(sigmoid(score_i - score_j)) for
    every strictly grade-ordered pair (i higher-graded than j). Returns a
    finite 0.0 still attached to `scores`' autograd graph when the batch has
    zero ordered pairs (e.g. every piece shares one grade) rather than NaN
    from averaging an empty tensor -- see Task 8."""
    pairs = ordered_pairs(grades)
    if pairs.shape[0] == 0:
        return scores.sum() * 0.0
    hi, lo = pairs[:, 0], pairs[:, 1]
    return torch.nn.functional.softplus(-(scores[hi] - scores[lo])).mean()


def ordinal_loss(
    logits: torch.Tensor, grades: torch.Tensor, n_levels: int
) -> torch.Tensor:
    """Cumulative-link ordinal loss: n_levels - 1 binary "grade > k" targets
    per row, BCE-with-logits against `logits` (shape (batch, n_levels - 1))."""
    thresholds = torch.arange(n_levels - 1, device=grades.device)
    targets = (grades.unsqueeze(1) > thresholds.unsqueeze(0)).float()
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)


def combined_loss(
    scores: torch.Tensor,
    ordinal_logits: torch.Tensor,
    grades: torch.Tensor,
    n_levels: int,
    ordinal_weight: float,
) -> torch.Tensor:
    """The training objective train_fold.py optimizes: pairwise ranking loss
    (primary, matches the tau-c gate metric) plus a low-weight ordinal
    auxiliary (keeps the score scale from drifting freely)."""
    return (pairwise_ranking_loss(scores, grades)
            + ordinal_weight * ordinal_loss(ordinal_logits, grades, n_levels))
