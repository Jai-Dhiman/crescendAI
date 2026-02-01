"""Gradient Reversal Layer for adversarial domain adaptation.

Based on:
- DANN: Domain-Adversarial Training of Neural Networks
  https://jmlr.org/papers/volume17/15-239/15-239.pdf
"""

import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    """Gradient reversal function for adversarial training.

    In the forward pass, this is an identity function.
    In the backward pass, gradients are multiplied by -lambda.
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """Gradient Reversal Layer with schedulable lambda.

    The lambda parameter controls the strength of gradient reversal:
    - lambda=0: No reversal (gradients pass through unchanged)
    - lambda=1: Full reversal (gradients are negated)

    Common schedules:
    - constant: Fixed lambda throughout training
    - linear: lambda = 2 / (1 + exp(-gamma * p)) - 1, where p is progress
    - cosine: lambda = 0.5 * (1 - cos(pi * p))
    """

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_: float):
        """Update the reversal strength."""
        self.lambda_ = lambda_


def get_grl_lambda(
    epoch: int,
    max_epochs: int,
    schedule: str = "linear",
    gamma: float = 10.0,
) -> float:
    """Compute GRL lambda based on training progress.

    Args:
        epoch: Current epoch (0-indexed).
        max_epochs: Total number of training epochs.
        schedule: Schedule type ("constant", "linear", "cosine").
        gamma: Steepness for linear schedule (default: 10.0 from DANN paper).

    Returns:
        Lambda value for gradient reversal.
    """
    import math

    p = epoch / max_epochs  # Progress [0, 1]

    if schedule == "constant":
        return 1.0
    elif schedule == "linear":
        # DANN paper schedule: starts at 0, ends at 1
        return float(2.0 / (1.0 + math.exp(-gamma * p)) - 1.0)
    elif schedule == "cosine":
        # Smooth cosine schedule: 0 -> 1
        return float(0.5 * (1.0 - math.cos(math.pi * p)))
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
