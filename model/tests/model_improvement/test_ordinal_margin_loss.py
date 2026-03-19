import torch
import pytest
from model_improvement.losses import ordinal_margin_loss


class TestOrdinalMarginLoss:
    def test_no_same_piece_pairs_returns_zero(self):
        """All segments from unique pieces -> no ordinal pairs -> loss 0."""
        emb = torch.randn(4, 8)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        piece_ids = torch.tensor([0, 1, 2, 3])
        quality = torch.tensor([0.8, 0.6, 0.4, 0.2])
        loss = ordinal_margin_loss(emb, piece_ids, quality)
        assert loss.item() == 0.0

    def test_same_quality_returns_zero(self):
        """Same piece, same quality -> no ordered pairs -> loss 0."""
        emb = torch.randn(4, 8)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        piece_ids = torch.tensor([0, 0, 1, 1])
        quality = torch.tensor([0.5, 0.5, 0.5, 0.5])
        loss = ordinal_margin_loss(emb, piece_ids, quality)
        assert loss.item() == 0.0

    def test_violated_margin_produces_nonzero_loss(self):
        """Better and worse embeddings equidistant from anchor -> margin violated."""
        torch.manual_seed(42)
        emb = torch.randn(4, 8)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        piece_ids = torch.tensor([0, 0, 1, 1])
        quality = torch.tensor([0.9, 0.1, 0.8, 0.2])
        loss = ordinal_margin_loss(emb, piece_ids, quality, margin_scale=1.0)
        assert loss.item() > 0.0

    def test_gradient_flows(self):
        """Loss must produce gradients."""
        emb = torch.randn(4, 8, requires_grad=True)
        emb_norm = torch.nn.functional.normalize(emb, dim=-1)
        piece_ids = torch.tensor([0, 0, 1, 1])
        quality = torch.tensor([0.9, 0.1, 0.8, 0.2])
        loss = ordinal_margin_loss(emb_norm, piece_ids, quality, margin_scale=1.0)
        loss.backward()
        assert emb.grad is not None

    def test_no_cross_piece_anchors_returns_zero(self):
        """All segments same piece -> no cross-piece anchor -> loss 0."""
        emb = torch.randn(3, 8)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        piece_ids = torch.tensor([0, 0, 0])
        quality = torch.tensor([0.9, 0.5, 0.1])
        loss = ordinal_margin_loss(emb, piece_ids, quality)
        assert loss.item() == 0.0

    def test_larger_margin_scale_increases_loss(self):
        """Larger margin_scale should produce >= loss."""
        torch.manual_seed(42)
        emb = torch.randn(4, 8)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        piece_ids = torch.tensor([0, 0, 1, 1])
        quality = torch.tensor([0.9, 0.1, 0.8, 0.2])
        torch.manual_seed(99)
        loss_small = ordinal_margin_loss(emb, piece_ids, quality, margin_scale=0.01)
        torch.manual_seed(99)
        loss_large = ordinal_margin_loss(emb, piece_ids, quality, margin_scale=1.0)
        assert loss_large >= loss_small

    def test_batch_size_one_returns_zero(self):
        """Single segment -> no pairs -> loss 0."""
        emb = torch.randn(1, 8)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        piece_ids = torch.tensor([0])
        quality = torch.tensor([0.5])
        loss = ordinal_margin_loss(emb, piece_ids, quality)
        assert loss.item() == 0.0
