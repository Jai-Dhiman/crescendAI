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
