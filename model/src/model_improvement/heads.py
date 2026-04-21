"""Heteroscedastic output head: predicts (mu, sigma) per quality dimension."""

import torch
import torch.nn as nn
import torch.nn.functional as F

SIGMA_FLOOR = 1e-4


class HeteroscedasticHead(nn.Module):
    """Two-branch linear head outputting (mu, sigma) per dimension.

    mu_head outputs unbounded means.
    log_sigma_head outputs raw logits, then softplus + sigma_floor prevents
    sigma-collapse (the NLL exploit where the model shrinks sigma to near-zero
    to eat the residual penalty).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_dims: int,
        sigma_floor: float = SIGMA_FLOOR,
    ):
        super().__init__()
        self.mu_head = nn.Linear(hidden_dim, num_dims)
        self.log_sigma_head = nn.Linear(hidden_dim, num_dims)
        self.sigma_floor = sigma_floor

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: Hidden representation [B, hidden_dim].

        Returns:
            mu: [B, num_dims] -- unbounded predicted mean.
            sigma: [B, num_dims] -- positive predicted std, >= sigma_floor.
        """
        mu = self.mu_head(z)
        sigma = F.softplus(self.log_sigma_head(z)) + self.sigma_floor
        return mu, sigma
