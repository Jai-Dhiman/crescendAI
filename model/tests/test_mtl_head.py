"""
Unit tests for multi-task learning head module.
"""

import torch
import pytest

from src.models.mtl_head import MultiTaskHead


def test_mtl_head_init():
    """Test MTL head initialization."""
    input_dim = 512
    dimensions = ['dim1', 'dim2', 'dim3', 'dim4', 'dim5', 'dim6']

    head = MultiTaskHead(input_dim=input_dim, dimensions=dimensions)

    assert head.input_dim == input_dim
    assert head.num_dimensions == len(dimensions)


def test_mtl_head_forward():
    """Test MTL head forward pass."""
    batch_size = 4
    input_dim = 512
    dimensions = ['dim1', 'dim2', 'dim3', 'dim4', 'dim5', 'dim6']

    head = MultiTaskHead(input_dim=input_dim, dimensions=dimensions)

    # Create dummy input
    x = torch.randn(batch_size, input_dim)

    # Forward pass (without uncertainties)
    scores, uncertainties = head(x)

    # Check shapes
    assert scores.shape == (batch_size, len(dimensions))
    assert uncertainties is None  # Not returned by default

    # Forward pass with uncertainties
    scores, uncertainties = head(x, return_uncertainties=True)
    assert uncertainties.shape == (len(dimensions),)


def test_mtl_head_score_range():
    """Test that scores are in 0-100 range."""
    batch_size = 8
    input_dim = 512
    dimensions = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6']

    head = MultiTaskHead(input_dim=input_dim, dimensions=dimensions)
    x = torch.randn(batch_size, input_dim)

    scores, _ = head(x)

    # Scores should be in [0, 100]
    assert torch.all(scores >= 0)
    assert torch.all(scores <= 100)


def test_mtl_head_uncertainties_positive():
    """Test that uncertainties are positive."""
    head = MultiTaskHead(input_dim=512, dimensions=['d1', 'd2', 'd3'])
    x = torch.randn(4, 512)

    _, uncertainties = head(x, return_uncertainties=True)

    # Uncertainties (σ) should be positive
    assert torch.all(uncertainties > 0)


def test_mtl_head_gradient_flow():
    """Test gradient flow through MTL head."""
    head = MultiTaskHead(input_dim=512, dimensions=['d1', 'd2', 'd3'])
    x = torch.randn(4, 512, requires_grad=True)

    scores, _ = head(x)

    # Create dummy loss
    loss = scores.sum()
    loss.backward()

    # Gradients should exist
    assert x.grad is not None
    assert torch.all(torch.isfinite(x.grad))


def test_mtl_head_uncertainty_gradient():
    """Test gradient flow to uncertainty parameters."""
    head = MultiTaskHead(input_dim=512, dimensions=['d1', 'd2', 'd3'])
    x = torch.randn(4, 512)

    scores, uncertainties = head(x, return_uncertainties=True)

    # Use uncertainties in loss
    loss = scores.sum() + uncertainties.sum()
    loss.backward()

    # Check that log_vars has gradients
    assert head.log_vars.grad is not None


@pytest.mark.parametrize("num_dimensions", [1, 6, 10])
def test_mtl_head_different_dimensions(num_dimensions):
    """Test MTL head with different numbers of dimensions."""
    dimensions = [f'd{i}' for i in range(num_dimensions)]
    head = MultiTaskHead(input_dim=512, dimensions=dimensions)
    x = torch.randn(4, 512)

    scores, uncertainties = head(x, return_uncertainties=True)

    assert scores.shape[1] == num_dimensions
    assert uncertainties.shape[0] == num_dimensions


@pytest.mark.parametrize("batch_size", [1, 4, 16, 32])
def test_mtl_head_different_batch_sizes(batch_size):
    """Test MTL head with different batch sizes."""
    head = MultiTaskHead(input_dim=512, dimensions=['d1', 'd2', 'd3'])
    x = torch.randn(batch_size, 512)

    scores, _ = head(x)

    assert scores.shape[0] == batch_size


def test_mtl_head_hidden_dim():
    """Test MTL head with custom hidden dimension."""
    head = MultiTaskHead(
        input_dim=512,
        dimensions=['d1', 'd2', 'd3'],
        shared_hidden=256,
        task_hidden=128
    )

    x = torch.randn(4, 512)
    scores, _ = head(x)

    # Should still work with custom dimensions
    assert scores.shape == (4, 3)


def test_mtl_head_dropout():
    """Test MTL head with dropout."""
    head = MultiTaskHead(
        input_dim=512,
        dimensions=['d1', 'd2', 'd3'],
        dropout=0.5
    )

    x = torch.randn(4, 512)

    # Training mode (dropout active)
    head.train()
    scores_train1 = head(x)[0]
    scores_train2 = head(x)[0]

    # Eval mode (dropout inactive)
    head.eval()
    scores_eval1 = head(x)[0]
    scores_eval2 = head(x)[0]

    # Eval scores should be deterministic
    torch.testing.assert_close(scores_eval1, scores_eval2)


def test_mtl_head_state_dict():
    """Test saving and loading MTL head."""
    dimensions = ['d1', 'd2', 'd3']
    head = MultiTaskHead(input_dim=512, dimensions=dimensions)
    x = torch.randn(4, 512)

    # Set to eval mode for deterministic behavior
    head.eval()

    # Get initial output
    scores1, _ = head(x)

    # Save state
    state = head.state_dict()

    # Create new head and load state
    head2 = MultiTaskHead(input_dim=512, dimensions=dimensions)
    head2.load_state_dict(state)
    head2.eval()

    # Should produce same output
    scores2, _ = head2(x)
    torch.testing.assert_close(scores1, scores2)


def test_mtl_head_per_dimension_heads():
    """Test that each dimension has independent head."""
    dimensions = ['d1', 'd2', 'd3']
    head = MultiTaskHead(input_dim=512, dimensions=dimensions)

    # Check task_heads ModuleDict
    assert len(head.task_heads) == len(dimensions)

    # Each head should be independent
    for dim_name in dimensions:
        assert dim_name in head.task_heads
        params = list(head.task_heads[dim_name].parameters())
        assert len(params) > 0


def test_mtl_head_shared_features():
    """Test shared feature extractor."""
    head = MultiTaskHead(input_dim=512, dimensions=['d1', 'd2', 'd3'], shared_hidden=256)
    x = torch.randn(4, 512)

    # All dimensions should use same shared features
    shared_features = head.shared_extractor(x)

    assert shared_features.shape == (4, 256)


def test_mtl_head_numerical_stability():
    """Test numerical stability with extreme inputs."""
    head = MultiTaskHead(input_dim=512, dimensions=['d1', 'd2', 'd3'])

    # Large values
    x_large = torch.randn(4, 512) * 100
    scores_large, _ = head(x_large)

    assert torch.all(torch.isfinite(scores_large))
    assert torch.all(scores_large >= 0)
    assert torch.all(scores_large <= 100)

    # Small values
    x_small = torch.randn(4, 512) * 0.01
    scores_small, _ = head(x_small)

    assert torch.all(torch.isfinite(scores_small))


def test_mtl_head_get_dimension_names():
    """Test getting dimension names."""
    dimensions = ['note_accuracy', 'rhythmic_precision', 'dynamics']
    head = MultiTaskHead(input_dim=512, dimensions=dimensions)

    assert head.get_dimension_names() == dimensions


def test_mtl_head_get_uncertainties():
    """Test getting current uncertainties."""
    head = MultiTaskHead(input_dim=512, dimensions=['d1', 'd2', 'd3'])

    uncertainties = head.get_uncertainties()

    assert uncertainties.shape == (3,)
    assert torch.all(uncertainties > 0)
