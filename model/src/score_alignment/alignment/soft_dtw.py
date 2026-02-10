"""Differentiable Soft-DTW loss for neural network training.

Soft-DTW replaces the hard min operation in standard DTW with a soft
minimum (log-sum-exp), making it differentiable and suitable for use
as a loss function in neural networks.

When pytorch-softdtw-cuda is installed, the DP recurrence runs as a native
CUDA kernel (~10-50x faster than the pure-PyTorch fallback).

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
    """Cosine distance for pytorch-softdtw-cuda dist_func interface.

    Args:
        x: [B, T1, D]
        y: [B, T2, D]

    Returns:
        [B, T1, T2] cosine distances (1 - cosine_similarity).
    """
    x_n = F.normalize(x, p=2, dim=-1)
    y_n = F.normalize(y, p=2, dim=-1)
    return 1 - torch.bmm(x_n, y_n.transpose(1, 2))


def _try_load_cuda_soft_dtw():
    """Attempt to import the CUDA-accelerated SoftDTW class.

    Returns the class if available, None otherwise.
    """
    try:
        from ._soft_dtw_cuda import SoftDTW as CudaSoftDTW
        return CudaSoftDTW
    except ImportError:
        return None


# Resolve once at import time so repeated calls are free.
_CudaSoftDTW = _try_load_cuda_soft_dtw()


class SoftDTWDivergence(nn.Module):
    """Soft-DTW Divergence loss.

    Computes: sdtw(x, y) - 0.5 * (sdtw(x, x) + sdtw(y, y))

    Uses the CUDA kernel from pytorch-softdtw-cuda when available,
    falling back to a pure-PyTorch anti-diagonal DP implementation.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        distance: str = "cosine",
    ):
        super().__init__()
        self.gamma = gamma
        self.distance = distance

        self._cuda_impl = None

        if _CudaSoftDTW is not None and torch.cuda.is_available():
            dist_func = _cosine_dist_func if distance == "cosine" else None
            self._cuda_impl = _CudaSoftDTW(
                use_cuda=True,
                gamma=gamma,
                normalize=True,  # normalize=True IS the divergence in this library
                dist_func=dist_func,
            )
            print(
                "SoftDTWDivergence: using CUDA kernel",
                file=sys.stderr,
                flush=True,
            )
        else:
            print(
                "SoftDTWDivergence: using pure-PyTorch fallback "
                "(CUDA not available or numba import failed)",
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
            x_mask: Unused when CUDA backend is active.
            y_mask: Unused when CUDA backend is active.

        Returns:
            Soft-DTW divergence (scalar or [B]).
        """
        if self._cuda_impl is not None and x.is_cuda:
            # Ensure 3-D
            squeeze = False
            if x.dim() == 2:
                x = x.unsqueeze(0)
                y = y.unsqueeze(0)
                squeeze = True

            result = self._cuda_impl(x, y)

            if squeeze:
                result = result.squeeze(0)
            return result

        # ---------- Pure-PyTorch fallback ----------
        return self._fallback_forward(x, y, x_mask, y_mask)

    # ------------------------------------------------------------------
    # Fallback: pure-PyTorch anti-diagonal DP
    # ------------------------------------------------------------------

    def _pairwise_distances(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
            squeeze = True

        if self.distance == "cosine":
            dist = _cosine_dist_func(x, y)
        elif self.distance == "sqeuclidean":
            x_sq = (x ** 2).sum(dim=-1, keepdim=True)
            y_sq = (y ** 2).sum(dim=-1, keepdim=True)
            xy = torch.bmm(x, y.transpose(1, 2))
            dist = x_sq + y_sq.transpose(1, 2) - 2 * xy
        elif self.distance == "euclidean":
            x_sq = (x ** 2).sum(dim=-1, keepdim=True)
            y_sq = (y ** 2).sum(dim=-1, keepdim=True)
            xy = torch.bmm(x, y.transpose(1, 2))
            dist = torch.sqrt(torch.clamp(x_sq + y_sq.transpose(1, 2) - 2 * xy, min=1e-8))
        else:
            raise ValueError(f"Unknown distance metric: {self.distance}")

        if squeeze:
            dist = dist.squeeze(0)
        return dist

    def _soft_dtw_forward(self, D: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if D.dim() == 2:
            D = D.unsqueeze(0)
            squeeze = True

        B, T1, T2 = D.shape
        device = D.device
        dtype = D.dtype

        R = torch.full((B, T1 + 1, T2 + 1), float("inf"), device=device, dtype=dtype)
        R[:, 0, 0] = 0

        for k in range(2, T1 + T2 + 1):
            i_start = max(1, k - T2)
            i_end = min(T1, k - 1)
            i_idx = torch.arange(i_start, i_end + 1, device=device)
            j_idx = k - i_idx

            cost = D[:, i_idx - 1, j_idx - 1]
            stacked = torch.stack([
                R[:, i_idx - 1, j_idx - 1],
                R[:, i_idx, j_idx - 1],
                R[:, i_idx - 1, j_idx],
            ], dim=0)
            soft_min = -self.gamma * torch.logsumexp(-stacked / self.gamma, dim=0)
            R[:, i_idx, j_idx] = cost + soft_min

        result = R[:, T1, T2]
        if squeeze:
            result = result.squeeze(0)
        return result

    def _sdtw(self, x: torch.Tensor, y: torch.Tensor, x_mask, y_mask) -> torch.Tensor:
        D = self._pairwise_distances(x, y)
        if x_mask is not None or y_mask is not None:
            if D.dim() == 2:
                D = D.unsqueeze(0)
                if x_mask is not None:
                    x_mask = x_mask.unsqueeze(0)
                if y_mask is not None:
                    y_mask = y_mask.unsqueeze(0)
            if x_mask is not None:
                D = D.masked_fill(~x_mask.unsqueeze(-1), float("inf"))
            if y_mask is not None:
                D = D.masked_fill(~y_mask.unsqueeze(1), float("inf"))
        return self._soft_dtw_forward(D)

    def _fallback_forward(self, x, y, x_mask, y_mask) -> torch.Tensor:
        d_xy = self._sdtw(x, y, x_mask, y_mask)
        d_xx = self._sdtw(x, x, x_mask, x_mask)
        d_yy = self._sdtw(y, y, y_mask, y_mask)
        return d_xy - 0.5 * (d_xx + d_yy)


# Keep old names as aliases for backwards compatibility
SoftDTWLoss = SoftDTWDivergence
try_load_cuda_soft_dtw = _try_load_cuda_soft_dtw
