"""
Unit tests for uncertainty-weighted loss module.
"""

import pytest
import torch
from src.losses.uncertainty_loss import UncertaintyWeightedLoss, WeightedMSELoss


def test_uncertainty_loss_init():
    """Test loss initialization."""
    num_tasks = 6
    loss_fn = UncertaintyWeightedLoss(num_tasks=num_tasks)

    assert loss_fn.num_tasks == num_tasks


def test_uncertainty_loss_forward():
    """Test loss forward pass."""
    batch_size = 4
    num_tasks = 6

    # Create dummy predictions, targets, and log_vars
    predictions = torch.randn(batch_size, num_tasks)
    targets = torch.randn(batch_size, num_tasks)
    log_vars = torch.randn(num_tasks)

    loss_fn = UncertaintyWeightedLoss(num_tasks=num_tasks)
    result = loss_fn(predictions, targets, log_vars)

    # Check output is a dictionary
    assert isinstance(result, dict)
    assert "loss" in result
    assert "task_losses" in result
    assert "uncertainties" in result

    # Check shapes
    assert result["loss"].shape == ()  # Scalar
    assert result["task_losses"].shape == (num_tasks,)
    assert result["uncertainties"].shape == (num_tasks,)


def test_uncertainty_loss_positive():
    """Test that loss is positive."""
    predictions = torch.randn(4, 6)
    targets = torch.randn(4, 6)
    log_vars = torch.randn(6)

    loss_fn = UncertaintyWeightedLoss(num_tasks=6)
    result = loss_fn(predictions, targets, log_vars)

    assert result["loss"].item() >= 0


def test_uncertainty_loss_gradient():
    """Test that gradients flow through loss."""
    predictions = torch.randn(4, 6, requires_grad=True)
    targets = torch.randn(4, 6)
    log_vars = torch.randn(6, requires_grad=True)

    loss_fn = UncertaintyWeightedLoss(num_tasks=6)
    result = loss_fn(predictions, targets, log_vars)

    # Backprop
    result["loss"].backward()

    # Gradients should exist
    assert predictions.grad is not None
    assert torch.all(torch.isfinite(predictions.grad))


def test_uncertainty_loss_gradient_log_vars():
    """Test that gradients flow to uncertainty parameters."""
    predictions = torch.randn(4, 6)
    targets = torch.randn(4, 6)
    log_vars = torch.randn(6, requires_grad=True)

    loss_fn = UncertaintyWeightedLoss(num_tasks=6)
    result = loss_fn(predictions, targets, log_vars)

    # Backprop
    result["loss"].backward()

    # Gradients should exist for log_vars
    assert log_vars.grad is not None
    assert torch.all(torch.isfinite(log_vars.grad))


def test_uncertainty_loss_perfect_prediction():
    """Test loss when predictions are perfect."""
    targets = torch.randn(4, 6)
    predictions = targets.clone()
    log_vars = torch.zeros(6)  # Small uncertainties

    loss_fn = UncertaintyWeightedLoss(num_tasks=6)
    result = loss_fn(predictions, targets, log_vars)

    # Loss should be small
    assert result["loss"].item() < 1.0


def test_uncertainty_loss_task_losses():
    """Test individual task losses."""
    predictions = torch.randn(4, 6)
    targets = torch.randn(4, 6)
    log_vars = torch.randn(6)

    loss_fn = UncertaintyWeightedLoss(num_tasks=6)
    result = loss_fn(predictions, targets, log_vars)

    # All task losses should be non-negative
    assert torch.all(result["task_losses"] >= 0)


@pytest.mark.parametrize("num_tasks", [1, 3, 6, 10])
def test_uncertainty_loss_different_num_tasks(num_tasks):
    """Test loss with different numbers of tasks."""
    predictions = torch.randn(4, num_tasks)
    targets = torch.randn(4, num_tasks)
    log_vars = torch.randn(num_tasks)

    loss_fn = UncertaintyWeightedLoss(num_tasks=num_tasks)
    result = loss_fn(predictions, targets, log_vars)

    assert result["task_losses"].shape[0] == num_tasks


def test_uncertainty_loss_batch_size_one():
    """Test loss with single sample."""
    predictions = torch.randn(1, 6)
    targets = torch.randn(1, 6)
    log_vars = torch.randn(6)

    loss_fn = UncertaintyWeightedLoss(num_tasks=6)
    result = loss_fn(predictions, targets, log_vars)

    assert result["loss"].shape == ()


def test_uncertainty_loss_numerical_stability():
    """Test loss doesn't produce NaN or Inf."""
    # Extreme values
    predictions = torch.randn(4, 6) * 100
    targets = torch.randn(4, 6) * 100
    log_vars = torch.randn(6)

    loss_fn = UncertaintyWeightedLoss(num_tasks=6)
    result = loss_fn(predictions, targets, log_vars)

    assert torch.isfinite(result["loss"])
    assert torch.all(torch.isfinite(result["task_losses"]))


def test_uncertainty_loss_label_smoothing():
    """Test label smoothing."""
    predictions = torch.randn(4, 6)
    targets = torch.randn(4, 6)
    log_vars = torch.randn(6)

    loss_fn = UncertaintyWeightedLoss(num_tasks=6, label_smoothing=0.1)
    result = loss_fn(predictions, targets, log_vars)

    assert result["loss"] is not None


def test_weighted_mse_loss():
    """Test weighted MSE loss baseline."""
    predictions = torch.randn(4, 6)
    targets = torch.randn(4, 6)

    loss_fn = WeightedMSELoss(num_tasks=6)
    result = loss_fn(predictions, targets)

    assert "loss" in result
    assert "task_losses" in result
    assert result["loss"].item() >= 0


def test_task_weights():
    """Test optional task weights."""
    predictions = torch.randn(4, 6)
    targets = torch.randn(4, 6)
    log_vars = torch.randn(6)
    task_weights = torch.tensor([2.0, 1.0, 1.0, 1.0, 0.5, 0.5])

    loss_fn = UncertaintyWeightedLoss(num_tasks=6)
    result = loss_fn(predictions, targets, log_vars, task_weights=task_weights)

    assert result["loss"] is not None
