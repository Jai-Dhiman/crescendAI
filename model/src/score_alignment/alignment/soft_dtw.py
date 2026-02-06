"""Differentiable Soft-DTW loss for neural network training.

Soft-DTW replaces the hard min operation in standard DTW with a soft
minimum (log-sum-exp), making it differentiable and suitable for use
as a loss function in neural networks.

Reference:
    Cuturi & Blondel (2017) "Soft-DTW: a Differentiable Loss Function for Time-Series"
    https://arxiv.org/abs/1703.01541
"""

import torch
import torch.nn as nn
from typing import Optional


class SoftDTWLoss(nn.Module):
    """Differentiable Soft-DTW loss function.

    Computes the soft-DTW distance between two sequences using
    a smoothed minimum operation controlled by gamma.

    As gamma -> 0, soft-DTW approaches standard DTW.
    Larger gamma values provide smoother gradients but less accurate alignment.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        distance: str = "cosine",
        normalize: bool = False,
    ):
        """Initialize Soft-DTW loss.

        Args:
            gamma: Smoothing parameter. Smaller = closer to hard DTW.
            distance: Distance metric ("cosine", "euclidean", "sqeuclidean").
            normalize: If True, normalize by path length.
        """
        super().__init__()
        self.gamma = gamma
        self.distance = distance
        self.normalize = normalize

    def _pairwise_distances(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute pairwise distance matrix between two sequences.

        Args:
            x: First sequence of shape [B, T1, D] or [T1, D].
            y: Second sequence of shape [B, T2, D] or [T2, D].

        Returns:
            Distance matrix of shape [B, T1, T2] or [T1, T2].
        """
        # Handle batch dimension
        squeeze_batch = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
            squeeze_batch = True

        B, T1, D = x.shape
        _, T2, _ = y.shape

        if self.distance == "cosine":
            # Normalize and compute dot product
            x_norm = torch.nn.functional.normalize(x, p=2, dim=-1)
            y_norm = torch.nn.functional.normalize(y, p=2, dim=-1)
            # Cosine distance = 1 - cosine_similarity
            dist = 1 - torch.bmm(x_norm, y_norm.transpose(1, 2))

        elif self.distance == "sqeuclidean":
            # Squared Euclidean: ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x.y
            x_sq = (x ** 2).sum(dim=-1, keepdim=True)  # [B, T1, 1]
            y_sq = (y ** 2).sum(dim=-1, keepdim=True)  # [B, T2, 1]
            xy = torch.bmm(x, y.transpose(1, 2))  # [B, T1, T2]
            dist = x_sq + y_sq.transpose(1, 2) - 2 * xy

        elif self.distance == "euclidean":
            x_sq = (x ** 2).sum(dim=-1, keepdim=True)
            y_sq = (y ** 2).sum(dim=-1, keepdim=True)
            xy = torch.bmm(x, y.transpose(1, 2))
            dist = torch.sqrt(
                torch.clamp(x_sq + y_sq.transpose(1, 2) - 2 * xy, min=1e-8)
            )

        else:
            raise ValueError(f"Unknown distance metric: {self.distance}")

        if squeeze_batch:
            dist = dist.squeeze(0)

        return dist

    def _soft_min(self, *args) -> torch.Tensor:
        """Compute soft minimum using log-sum-exp."""
        stacked = torch.stack(args, dim=0)
        return -self.gamma * torch.logsumexp(-stacked / self.gamma, dim=0)

    def _soft_dtw_forward(
        self,
        D: torch.Tensor,
    ) -> torch.Tensor:
        """Compute soft-DTW using anti-diagonal vectorized dynamic programming.

        Instead of a T1*T2 double for-loop, iterates over anti-diagonals
        k = i + j (from 2 to T1+T2). All cells on a given anti-diagonal are
        independent and can be computed in a single vectorized step.

        Args:
            D: Distance matrix of shape [B, T1, T2] or [T1, T2].

        Returns:
            Soft-DTW value of shape [B] or scalar.
        """
        squeeze_batch = False
        if D.dim() == 2:
            D = D.unsqueeze(0)
            squeeze_batch = True

        B, T1, T2 = D.shape
        device = D.device
        dtype = D.dtype

        # R is (T1+1) x (T2+1) with 0-indexed border; R[0,0]=0, rest=inf
        R = torch.full((B, T1 + 1, T2 + 1), float("inf"), device=device, dtype=dtype)
        R[:, 0, 0] = 0

        # Iterate over anti-diagonals k = i + j (1-indexed into R)
        # k ranges from 2 (cell [1,1]) to T1+T2 (cell [T1,T2])
        for k in range(2, T1 + T2 + 1):
            # Valid i range: max(1, k-T2) <= i <= min(T1, k-1)
            i_start = max(1, k - T2)
            i_end = min(T1, k - 1)
            i_idx = torch.arange(i_start, i_end + 1, device=device)
            j_idx = k - i_idx  # j = k - i

            # Gather costs and predecessors for all cells on this diagonal
            cost = D[:, i_idx - 1, j_idx - 1]          # [B, diag_len]
            r_diag = R[:, i_idx - 1, j_idx - 1]        # [B, diag_len]
            r_left = R[:, i_idx, j_idx - 1]             # [B, diag_len]
            r_up = R[:, i_idx - 1, j_idx]               # [B, diag_len]

            # Soft-min over the three predecessors
            stacked = torch.stack([r_diag, r_left, r_up], dim=0)  # [3, B, diag_len]
            soft_min = -self.gamma * torch.logsumexp(
                -stacked / self.gamma, dim=0
            )  # [B, diag_len]

            R[:, i_idx, j_idx] = cost + soft_min

        result = R[:, T1, T2]

        if self.normalize:
            result = result / (T1 + T2)

        if squeeze_batch:
            result = result.squeeze(0)

        return result

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
        y_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute soft-DTW loss between two sequences.

        Args:
            x: First sequence of shape [B, T1, D] or [T1, D].
            y: Second sequence of shape [B, T2, D] or [T2, D].
            x_mask: Optional mask for x of shape [B, T1] or [T1].
            y_mask: Optional mask for y of shape [B, T2] or [T2].

        Returns:
            Soft-DTW loss value.
        """
        D = self._pairwise_distances(x, y)

        # Apply masking by setting distances to infinity for masked positions
        if x_mask is not None or y_mask is not None:
            if D.dim() == 2:
                D = D.unsqueeze(0)
                if x_mask is not None:
                    x_mask = x_mask.unsqueeze(0)
                if y_mask is not None:
                    y_mask = y_mask.unsqueeze(0)

            if x_mask is not None:
                # Expand mask to [B, T1, 1] and apply
                D = D.masked_fill(~x_mask.unsqueeze(-1), float("inf"))
            if y_mask is not None:
                # Expand mask to [B, 1, T2] and apply
                D = D.masked_fill(~y_mask.unsqueeze(1), float("inf"))

        return self._soft_dtw_forward(D)


class SoftDTWDivergence(nn.Module):
    """Soft-DTW Divergence loss.

    Computes: sdtw(x, y) - 0.5 * (sdtw(x, x) + sdtw(y, y))

    This divergence is non-negative and equals zero when x = y,
    making it a proper divergence measure.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        distance: str = "cosine",
        normalize: bool = False,
    ):
        """Initialize Soft-DTW Divergence.

        Args:
            gamma: Smoothing parameter for soft-DTW.
            distance: Distance metric.
            normalize: If True, normalize by path length.
        """
        super().__init__()
        self.sdtw = SoftDTWLoss(gamma=gamma, distance=distance, normalize=normalize)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
        y_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute soft-DTW divergence.

        Args:
            x: First sequence of shape [B, T1, D] or [T1, D].
            y: Second sequence of shape [B, T2, D] or [T2, D].
            x_mask: Optional mask for x.
            y_mask: Optional mask for y.

        Returns:
            Soft-DTW divergence value.
        """
        d_xy = self.sdtw(x, y, x_mask, y_mask)
        d_xx = self.sdtw(x, x, x_mask, x_mask)
        d_yy = self.sdtw(y, y, y_mask, y_mask)

        return d_xy - 0.5 * (d_xx + d_yy)


def try_load_cuda_soft_dtw():
    """Attempt to load GPU-accelerated soft-DTW implementation.

    Returns the CudaSoftDTW class if available, None otherwise.

    To install: pip install git+https://github.com/Maghoumi/pytorch-softdtw-cuda
    """
    try:
        from soft_dtw_cuda import SoftDTW as CudaSoftDTW
        return CudaSoftDTW
    except ImportError:
        return None
