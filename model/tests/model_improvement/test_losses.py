import torch
import pytest
from model_improvement.losses import ListMLELoss


class TestListMLELoss:
    def test_perfect_ranking_low_loss(self):
        """Perfectly ordered predictions should yield lower loss than reversed."""
        loss_fn = ListMLELoss()
        # 5 items, 1 dimension, perfectly ordered predictions match labels
        predictions = torch.tensor([[0.9], [0.7], [0.5], [0.3], [0.1]])
        labels = torch.tensor([[0.9], [0.7], [0.5], [0.3], [0.1]])
        perfect_loss = loss_fn(predictions, labels)

        # Reversed predictions
        reversed_preds = torch.tensor([[0.1], [0.3], [0.5], [0.7], [0.9]])
        reversed_loss = loss_fn(reversed_preds, labels)

        assert perfect_loss < reversed_loss

    def test_single_item_returns_zero(self):
        """ListMLE with a single item should return 0."""
        loss_fn = ListMLELoss()
        predictions = torch.tensor([[0.5]])
        labels = torch.tensor([[0.5]])
        loss = loss_fn(predictions, labels)
        assert loss.item() == 0.0

    def test_two_items_nonzero(self):
        """Two items with different labels should produce nonzero loss."""
        loss_fn = ListMLELoss()
        predictions = torch.tensor([[0.3], [0.7]])
        labels = torch.tensor([[0.8], [0.2]])  # Wrong order
        loss = loss_fn(predictions, labels)
        assert loss.item() > 0.0

    def test_multi_dimension(self):
        """Should work independently per dimension."""
        loss_fn = ListMLELoss()
        predictions = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
        labels = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
        loss = loss_fn(predictions, labels)
        assert loss.ndim == 0

    def test_gradient_flows(self):
        """Loss must produce gradients."""
        loss_fn = ListMLELoss()
        predictions = torch.tensor([[0.5], [0.3]], requires_grad=True)
        labels = torch.tensor([[0.8], [0.2]])
        loss = loss_fn(predictions, labels)
        loss.backward()
        assert predictions.grad is not None
