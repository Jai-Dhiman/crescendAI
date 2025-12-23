"""
Bootstrap loss for learning with noisy labels.

Softens labels using the model's own predictions to reduce
the impact of label noise during training.

Reference: "Training Deep Neural Networks on Noisy Labels with Bootstrapping"
(Reed et al., ICLR Workshop 2014)
"""

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BootstrapLoss(nn.Module):
    """
    Bootstrap loss for robust learning with noisy labels.

    Interpolates between the original label and the model's prediction:
        soft_target = beta * target + (1 - beta) * prediction
        loss = MSE(prediction, soft_target)

    This has a self-correcting effect:
    - For clean labels: model learns normally
    - For noisy labels: model's consistent predictions override noise

    Two variants:
    - Hard bootstrap: Uses argmax of prediction (for classification)
    - Soft bootstrap: Uses prediction probabilities directly (for regression)

    For regression tasks, soft bootstrap is more appropriate.
    """

    def __init__(
        self,
        beta: float = 0.8,
        warmup_epochs: int = 0,
        base_loss: Literal["mse", "huber", "mae"] = "mse",
        huber_delta: float = 1.0,
    ):
        """
        Initialize bootstrap loss.

        Args:
            beta: Weight on original label vs model prediction (0.0-1.0)
                  - 1.0 = pure label (no bootstrapping)
                  - 0.0 = pure prediction (self-training)
                  - 0.8 = typical value (80% label, 20% prediction)
            warmup_epochs: Epochs to train with pure labels before bootstrapping
            base_loss: Base loss function ('mse', 'huber', or 'mae')
            huber_delta: Delta parameter for Huber loss
        """
        super().__init__()

        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"beta must be in [0, 1], got {beta}")

        self.beta = beta
        self.warmup_epochs = warmup_epochs
        self.base_loss = base_loss
        self.huber_delta = huber_delta
        self.current_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set current epoch (for warmup scheduling)."""
        self.current_epoch = epoch

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        epoch: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute bootstrap loss.

        Args:
            predictions: Model predictions [batch, num_dims]
            targets: Target labels [batch, num_dims]
            epoch: Current epoch (optional, overrides set_epoch)

        Returns:
            Bootstrap loss (scalar)
        """
        if epoch is not None:
            self.current_epoch = epoch

        # During warmup, use pure labels
        if self.current_epoch < self.warmup_epochs:
            effective_beta = 1.0
        else:
            effective_beta = self.beta

        # Compute soft targets
        # Using .detach() prevents gradients flowing through the soft target
        soft_targets = (
            effective_beta * targets + (1 - effective_beta) * predictions.detach()
        )

        # Compute loss against soft targets
        if self.base_loss == "mse":
            loss = F.mse_loss(predictions, soft_targets)
        elif self.base_loss == "huber":
            loss = F.huber_loss(predictions, soft_targets, delta=self.huber_delta)
        elif self.base_loss == "mae":
            loss = F.l1_loss(predictions, soft_targets)
        else:
            raise ValueError(f"Unknown base_loss: {self.base_loss}")

        return loss


class SymmetricBootstrapLoss(nn.Module):
    """
    Symmetric bootstrap loss for improved noise robustness.

    Combines forward (label -> prediction) and backward (prediction -> label)
    cross-entropy for more symmetric treatment of label noise.

    L = alpha * H(soft_target, prediction) + (1-alpha) * H(prediction, soft_target)

    where H is the appropriate loss function.
    """

    def __init__(
        self,
        beta: float = 0.8,
        alpha: float = 0.5,
        base_loss: Literal["mse", "huber", "mae"] = "mse",
    ):
        """
        Initialize symmetric bootstrap loss.

        Args:
            beta: Bootstrap interpolation weight
            alpha: Weight between forward and backward loss
            base_loss: Base loss function
        """
        super().__init__()

        self.beta = beta
        self.alpha = alpha
        self.base_loss = base_loss

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute symmetric bootstrap loss.

        Args:
            predictions: Model predictions [batch, num_dims]
            targets: Target labels [batch, num_dims]

        Returns:
            Symmetric bootstrap loss (scalar)
        """
        # Soft targets
        soft_targets = self.beta * targets + (1 - self.beta) * predictions.detach()

        # Forward loss: prediction trying to match soft target
        if self.base_loss == "mse":
            forward_loss = F.mse_loss(predictions, soft_targets)
            backward_loss = F.mse_loss(soft_targets, predictions.detach())
        elif self.base_loss == "huber":
            forward_loss = F.huber_loss(predictions, soft_targets)
            backward_loss = F.huber_loss(soft_targets, predictions.detach())
        elif self.base_loss == "mae":
            forward_loss = F.l1_loss(predictions, soft_targets)
            backward_loss = F.l1_loss(soft_targets, predictions.detach())
        else:
            raise ValueError(f"Unknown base_loss: {self.base_loss}")

        # Combine symmetrically
        loss = self.alpha * forward_loss + (1 - self.alpha) * backward_loss

        return loss


class AdaptiveBootstrapLoss(nn.Module):
    """
    Adaptive bootstrap loss with sample-dependent beta.

    Uses model confidence to determine how much to trust predictions
    vs labels on a per-sample basis:
    - High confidence predictions: Trust prediction more
    - Low confidence predictions: Trust label more

    This is useful when label noise varies across samples.
    """

    def __init__(
        self,
        base_beta: float = 0.8,
        confidence_scale: float = 0.2,
        base_loss: Literal["mse", "huber", "mae"] = "mse",
    ):
        """
        Initialize adaptive bootstrap loss.

        Args:
            base_beta: Base weight on original labels
            confidence_scale: How much confidence affects beta
            base_loss: Base loss function
        """
        super().__init__()

        self.base_beta = base_beta
        self.confidence_scale = confidence_scale
        self.base_loss = base_loss

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        uncertainties: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute adaptive bootstrap loss.

        Args:
            predictions: Model predictions [batch, num_dims]
            targets: Target labels [batch, num_dims]
            uncertainties: Per-sample uncertainty estimates [batch, num_dims]
                          Higher uncertainty = trust label more

        Returns:
            Adaptive bootstrap loss (scalar)
        """
        if uncertainties is None:
            # Fall back to standard bootstrap
            beta = self.base_beta
        else:
            # Compute sample-dependent beta based on uncertainty
            # Higher uncertainty -> higher beta (trust label more)
            # Normalize uncertainties to [0, 1] range approximately
            normalized_unc = torch.sigmoid(uncertainties)

            # Beta increases with uncertainty
            beta = self.base_beta + self.confidence_scale * normalized_unc
            beta = beta.clamp(0.0, 1.0)

        # Compute soft targets with sample-dependent beta
        soft_targets = beta * targets + (1 - beta) * predictions.detach()

        # Compute loss
        if self.base_loss == "mse":
            loss = F.mse_loss(predictions, soft_targets)
        elif self.base_loss == "huber":
            loss = F.huber_loss(predictions, soft_targets)
        elif self.base_loss == "mae":
            loss = F.l1_loss(predictions, soft_targets)
        else:
            raise ValueError(f"Unknown base_loss: {self.base_loss}")

        return loss


if __name__ == "__main__":
    print("Bootstrap Loss for Noisy Labels")
    print("- Softens labels using model predictions")
    print("- Reduces impact of label noise")
    print("- Variants: Standard, Symmetric, Adaptive")

    # Example usage
    bootstrap = BootstrapLoss(beta=0.8, warmup_epochs=5)

    # Simulate predictions and noisy targets
    predictions = torch.randn(32, 8) * 10 + 50  # Mean ~50
    targets = predictions + torch.randn(32, 8) * 5  # Noisy version

    # Before warmup
    bootstrap.set_epoch(0)
    loss_warmup = bootstrap(predictions, targets)
    print(f"\nLoss during warmup (epoch 0): {loss_warmup.item():.4f}")

    # After warmup
    bootstrap.set_epoch(10)
    loss_bootstrap = bootstrap(predictions, targets)
    print(f"Loss after warmup (epoch 10): {loss_bootstrap.item():.4f}")
