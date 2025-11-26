import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Literal


class UncertaintyWeightedLoss(nn.Module):
    """
    Uncertainty-weighted multi-task loss (Kendall & Gal, CVPR 2018).

    Automatically balances multiple tasks without manual weight tuning:

    L_total = Σ_i [ (1 / 2σ_i²) * L_i + log(σ_i) ]

    where:
    - L_i: Individual task loss (MSE or Huber for regression)
    - σ_i: Learned uncertainty parameter (per dimension)
    - First term: Task loss weighted by inverse uncertainty
    - Second term: Regularization preventing σ → ∞

    Benefits:
    - No manual loss weight tuning required
    - Tasks with higher inherent noise automatically downweighted
    - Provides per-dimension uncertainty estimates
    - Huber loss option for robustness to outliers
    """

    def __init__(
        self,
        num_tasks: int = 10,
        reduction: str = 'mean',
        label_smoothing: float = 0.0,
        base_loss: Literal['mse', 'huber', 'mae'] = 'mse',
        huber_delta: float = 1.0,
    ):
        """
        Initialize uncertainty-weighted loss.

        Args:
            num_tasks: Number of tasks (dimensions)
            reduction: Loss reduction method ('mean', 'sum', 'none')
            label_smoothing: Label smoothing factor (0.0 to 1.0)
            base_loss: Base loss function ('mse', 'huber', or 'mae')
            huber_delta: Delta parameter for Huber loss (default: 1.0)
        """
        super().__init__()

        self.num_tasks = num_tasks
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.base_loss = base_loss
        self.huber_delta = huber_delta

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        log_vars: torch.Tensor,
        task_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute uncertainty-weighted loss.

        Args:
            predictions: Predicted scores [batch, num_tasks]
            targets: Target scores [batch, num_tasks]
            log_vars: Log variance parameters [num_tasks]
            task_weights: Optional task-specific weights [num_tasks]

        Returns:
            Dictionary containing:
                - 'loss': Total weighted loss
                - 'task_losses': Per-task losses [num_tasks]
                - 'uncertainties': Per-task uncertainties [num_tasks]
                - 'weighted_losses': Per-task weighted losses [num_tasks]
        """
        batch_size, num_tasks = predictions.shape
        assert num_tasks == self.num_tasks, f"Expected {self.num_tasks} tasks, got {num_tasks}"

        # Apply label smoothing if requested
        if self.label_smoothing > 0:
            targets = self._apply_label_smoothing(targets)

        # Compute per-task losses using configured base loss
        if self.base_loss == 'mse':
            task_losses = F.mse_loss(predictions, targets, reduction='none')
        elif self.base_loss == 'huber':
            task_losses = F.huber_loss(predictions, targets, reduction='none', delta=self.huber_delta)
        elif self.base_loss == 'mae':
            task_losses = F.l1_loss(predictions, targets, reduction='none')
        else:
            raise ValueError(f"Unknown base_loss: {self.base_loss}")

        task_losses = task_losses.mean(dim=0)  # [batch, num_tasks] -> [num_tasks]

        # Compute precision (inverse variance) from log variance
        # precision = 1 / (2 * σ²) = exp(-log_vars) / 2
        precision = torch.exp(-log_vars) / 2.0

        # Uncertainty-weighted loss for each task:
        # (1 / 2σ²) * L_i + log(σ) = precision * L_i + 0.5 * log_var
        weighted_losses = precision * task_losses + 0.5 * log_vars

        # Apply optional task weights
        if task_weights is not None:
            weighted_losses = weighted_losses * task_weights

        # Total loss
        if self.reduction == 'mean':
            total_loss = weighted_losses.mean()
        elif self.reduction == 'sum':
            total_loss = weighted_losses.sum()
        else:  # 'none'
            total_loss = weighted_losses

        # Compute uncertainties (standard deviations)
        uncertainties = torch.exp(0.5 * log_vars)

        return {
            'loss': total_loss,
            'task_losses': task_losses,
            'uncertainties': uncertainties,
            'weighted_losses': weighted_losses,
        }

    def _apply_label_smoothing(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Apply label smoothing to targets.

        Shifts targets α% toward the mean (typically 0.1).
        This prevents overconfidence.

        Args:
            targets: Target values [batch, num_tasks]

        Returns:
            Smoothed targets
        """
        # Compute mean for each task
        task_means = targets.mean(dim=0, keepdim=True)

        # Interpolate between targets and means
        smoothed = (1 - self.label_smoothing) * targets + self.label_smoothing * task_means

        return smoothed


class WeightedMSELoss(nn.Module):
    """
    Simple weighted MSE loss for comparison.

    Uses fixed manual weights instead of learned uncertainties.
    """

    def __init__(
        self,
        num_tasks: int = 10,
        task_weights: Optional[torch.Tensor] = None,
        reduction: str = 'mean',
    ):
        """
        Initialize weighted MSE loss.

        Args:
            num_tasks: Number of tasks
            task_weights: Optional fixed weights [num_tasks]
            reduction: Loss reduction method
        """
        super().__init__()

        self.num_tasks = num_tasks
        self.reduction = reduction

        if task_weights is None:
            # Equal weights
            task_weights = torch.ones(num_tasks)
        self.register_buffer('task_weights', task_weights)

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute weighted MSE loss.

        Args:
            predictions: Predicted scores [batch, num_tasks]
            targets: Target scores [batch, num_tasks]
            **kwargs: Ignored (for compatibility)

        Returns:
            Dictionary with 'loss' and 'task_losses'
        """
        # Per-task MSE
        task_losses = F.mse_loss(predictions, targets, reduction='none')
        task_losses = task_losses.mean(dim=0)  # [num_tasks]

        # Weighted losses
        weighted_losses = task_losses * self.task_weights

        # Total loss
        if self.reduction == 'mean':
            total_loss = weighted_losses.mean()
        elif self.reduction == 'sum':
            total_loss = weighted_losses.sum()
        else:
            total_loss = weighted_losses

        return {
            'loss': total_loss,
            'task_losses': task_losses,
            'weighted_losses': weighted_losses,
        }


class CombinedLoss(nn.Module):
    """
    Combined loss with multiple components.

    Useful for adding auxiliary losses (e.g., contrastive, ranking).
    """

    def __init__(
        self,
        primary_loss: nn.Module,
        auxiliary_losses: Optional[Dict[str, tuple]] = None,
    ):
        """
        Initialize combined loss.

        Args:
            primary_loss: Main loss module (e.g., UncertaintyWeightedLoss)
            auxiliary_losses: Dict of {name: (loss_fn, weight)} pairs
        """
        super().__init__()

        self.primary_loss = primary_loss
        self.auxiliary_losses = auxiliary_losses or {}

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            predictions: Predictions
            targets: Targets
            **kwargs: Additional arguments for losses

        Returns:
            Dictionary with all loss components
        """
        # Compute primary loss
        result = self.primary_loss(predictions, targets, **kwargs)

        # Compute auxiliary losses
        total_loss = result['loss']
        for name, (loss_fn, weight) in self.auxiliary_losses.items():
            aux_loss = loss_fn(predictions, targets, **kwargs)
            result[f'aux_{name}'] = aux_loss
            total_loss = total_loss + weight * aux_loss

        result['total_loss'] = total_loss

        return result


def create_loss_function(
    loss_type: str = 'uncertainty',
    num_tasks: int = 10,
    **kwargs
) -> nn.Module:
    """
    Factory function to create loss function.

    Args:
        loss_type: 'uncertainty' or 'weighted_mse'
        num_tasks: Number of tasks
        **kwargs: Additional arguments

    Returns:
        Loss function module
    """
    if loss_type == 'uncertainty':
        return UncertaintyWeightedLoss(
            num_tasks=num_tasks,
            **kwargs
        )
    elif loss_type == 'weighted_mse':
        return WeightedMSELoss(
            num_tasks=num_tasks,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    print("Uncertainty-weighted loss module loaded successfully")
    print("- Kendall & Gal formulation (CVPR 2018)")
    print("- Automatic task balancing (no manual tuning)")
    print("- Per-dimension uncertainty estimates")
    print("- Label smoothing support")
