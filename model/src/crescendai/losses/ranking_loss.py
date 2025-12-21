"""
Pairwise ranking loss for relative comparisons.

Encourages the model to preserve relative ordering of performance quality,
even when absolute values are noisy (as with synthetic labels).
More robust than MSE loss for weak supervision scenarios.

Reference: "Learning to Rank using Gradient Descent" (Burges et al., 2005)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RankingLoss(nn.Module):
    """
    Pairwise ranking loss for relative comparisons.

    For all pairs (i, j) where target[i] > target[j],
    enforce prediction[i] > prediction[j] with margin.
    """

    def __init__(self, margin: float = 1.0):
        """
        Initialize ranking loss.

        Args:
            margin: Margin for ranking violation (default: 1.0)
                   Larger margin enforces stricter ordering
        """
        super().__init__()
        self.margin = margin

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute pairwise ranking loss.

        Args:
            predictions: Model predictions [batch, num_dims]
            targets: Target labels [batch, num_dims]

        Returns:
            Ranking loss (scalar)
        """
        batch_size = predictions.shape[0]

        if batch_size < 2:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        # Compute pairwise differences
        # diff_target[i, j] = target[i] - target[j]
        # diff_pred[i, j] = prediction[i] - prediction[j]
        diff_target = targets.unsqueeze(1) - targets.unsqueeze(0)  # [batch, batch, num_dims]
        diff_pred = predictions.unsqueeze(1) - predictions.unsqueeze(0)  # [batch, batch, num_dims]

        # Create mask for pairs where target[i] > target[j]
        mask = (diff_target > 0).float()

        # Margin ranking loss: max(0, margin - (pred[i] - pred[j]))
        loss = F.relu(self.margin - diff_pred) * mask

        # Average over all valid pairs and dimensions
        num_pairs = mask.sum()
        if num_pairs > 0:
            loss = loss.sum() / num_pairs
        else:
            loss = torch.tensor(0.0, device=predictions.device, requires_grad=True)

        return loss


if __name__ == "__main__":
    print("Ranking loss module loaded successfully")
    print("- Pairwise ranking for noisy labels")
    print("- Preserves relative ordering when absolute values are uncertain")
