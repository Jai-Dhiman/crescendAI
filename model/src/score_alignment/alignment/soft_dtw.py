"""Differentiable Soft-DTW loss for neural network training.

Soft-DTW replaces the hard min operation in standard DTW with a soft
minimum (log-sum-exp), making it differentiable and suitable for use
as a loss function in neural networks.

The anti-diagonal DP recurrence is JIT-compiled with torch.jit.script
to eliminate Python loop overhead (~1500 iterations for 750-frame seqs).

Reference:
    Cuturi & Blondel (2017) "Soft-DTW: a Differentiable Loss Function for Time-Series"
    https://arxiv.org/abs/1703.01541
"""

import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cosine_dist_func(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Cosine distance: 1 - cosine_similarity.

    Args:
        x: [B, T1, D]
        y: [B, T2, D]

    Returns:
        [B, T1, T2] cosine distances.
    """
    x_n = F.normalize(x, p=2, dim=-1)
    y_n = F.normalize(y, p=2, dim=-1)
    return 1 - torch.bmm(x_n, y_n.transpose(1, 2))


@torch.jit.script
def _soft_dtw_forward(D: torch.Tensor, gamma: float) -> torch.Tensor:
    """JIT-compiled anti-diagonal soft-DTW dynamic programming.

    Iterates over anti-diagonals k = i + j, vectorizing all cells on
    each diagonal into a single batch operation.  JIT compilation
    eliminates ~20 us of Python overhead per iteration.

    Args:
        D: [B, T1, T2] pairwise distance matrix.
        gamma: Smoothing parameter.

    Returns:
        [B] soft-DTW values.
    """
    B = D.shape[0]
    T1 = D.shape[1]
    T2 = D.shape[2]

    R = torch.full((B, T1 + 1, T2 + 1), float("inf"),
                    device=D.device, dtype=D.dtype)
    R[:, 0, 0] = 0.0

    for k in range(2, T1 + T2 + 1):
        i_start = max(1, k - T2)
        i_end = min(T1, k - 1)
        i_idx = torch.arange(i_start, i_end + 1, device=D.device)
        j_idx = k - i_idx

        cost = D[:, i_idx - 1, j_idx - 1]
        stacked = torch.stack([
            R[:, i_idx - 1, j_idx - 1],
            R[:, i_idx, j_idx - 1],
            R[:, i_idx - 1, j_idx],
        ], dim=0)
        soft_min = -gamma * torch.logsumexp(-stacked / gamma, dim=0)
        R[:, i_idx, j_idx] = cost + soft_min

    return R[:, T1, T2]


class SoftDTWDivergence(nn.Module):
    """Soft-DTW Divergence loss.

    Computes: sdtw(x, y) - 0.5 * (sdtw(x, x) + sdtw(y, y))

    Uses a JIT-compiled anti-diagonal DP that runs entirely on GPU.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        distance: str = "cosine",
    ):
        super().__init__()
        self.gamma = gamma
        self.distance = distance
        print(
            f"SoftDTWDivergence: JIT-compiled PyTorch "
            f"(gamma={gamma}, distance={distance})",
            file=sys.stderr,
            flush=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
        y_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute soft-DTW divergence.

        Args:
            x: [B, T1, D] or [T1, D]
            y: [B, T2, D] or [T2, D]
            x_mask: Optional boolean mask for x, shape [B, T1] or [T1].
            y_mask: Optional boolean mask for y, shape [B, T2] or [T2].

        Returns:
            Soft-DTW divergence (scalar or [B]).
        """
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
            squeeze = True

        D_xy = self._pairwise_distances(x, y)
        D_xx = self._pairwise_distances(x, x)
        D_yy = self._pairwise_distances(y, y)

        sdtw_xy = _soft_dtw_forward(D_xy, self.gamma)
        sdtw_xx = _soft_dtw_forward(D_xx, self.gamma)
        sdtw_yy = _soft_dtw_forward(D_yy, self.gamma)

        result = sdtw_xy - 0.5 * (sdtw_xx + sdtw_yy)

        if squeeze:
            result = result.squeeze(0)
        return result

    def _pairwise_distances(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distance matrix [B, T1, T2]."""
        if self.distance == "cosine":
            return _cosine_dist_func(x, y)
        elif self.distance == "sqeuclidean":
            x_sq = (x ** 2).sum(dim=-1, keepdim=True)
            y_sq = (y ** 2).sum(dim=-1, keepdim=True)
            xy = torch.bmm(x, y.transpose(1, 2))
            return x_sq + y_sq.transpose(1, 2) - 2 * xy
        elif self.distance == "euclidean":
            x_sq = (x ** 2).sum(dim=-1, keepdim=True)
            y_sq = (y ** 2).sum(dim=-1, keepdim=True)
            xy = torch.bmm(x, y.transpose(1, 2))
            return torch.sqrt(torch.clamp(
                x_sq + y_sq.transpose(1, 2) - 2 * xy, min=1e-8
            ))
        raise ValueError(f"Unknown distance metric: {self.distance}")


# Backward-compatible aliases
SoftDTWLoss = SoftDTWDivergence


def try_load_cuda_soft_dtw():
    return None
