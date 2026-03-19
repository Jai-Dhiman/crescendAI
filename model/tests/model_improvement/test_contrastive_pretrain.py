"""Tests for contrastive pretraining pipeline."""

import torch
import pytest


class TestMuQContrastiveEncoder:
    def test_encode_shape(self):
        from model_improvement.autoresearch_contrastive import MuQContrastiveEncoder
        enc = MuQContrastiveEncoder(input_dim=1024, hidden_dim=512, projection_dim=256)
        x = torch.randn(2, 50, 1024)
        mask = torch.ones(2, 50, dtype=torch.bool)
        z = enc.encode(x, mask)
        assert z.shape == (2, 512)

    def test_project_shape(self):
        from model_improvement.autoresearch_contrastive import MuQContrastiveEncoder
        enc = MuQContrastiveEncoder(input_dim=1024, hidden_dim=512, projection_dim=256)
        z = torch.randn(2, 512)
        proj = enc.project(z)
        assert proj.shape == (2, 256)

    def test_project_is_normalized(self):
        from model_improvement.autoresearch_contrastive import MuQContrastiveEncoder
        enc = MuQContrastiveEncoder(input_dim=1024, hidden_dim=512, projection_dim=256)
        z = torch.randn(2, 512)
        proj = enc.project(z)
        norms = proj.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(2), atol=1e-5)

    def test_mask_affects_output(self):
        from model_improvement.autoresearch_contrastive import MuQContrastiveEncoder
        enc = MuQContrastiveEncoder(input_dim=1024, hidden_dim=512, projection_dim=256)
        enc.eval()
        x = torch.randn(1, 50, 1024)
        mask_full = torch.ones(1, 50, dtype=torch.bool)
        mask_half = torch.ones(1, 50, dtype=torch.bool)
        mask_half[0, 25:] = False
        with torch.no_grad():
            z_full = enc.encode(x, mask_full)
            z_half = enc.encode(x, mask_half)
        assert not torch.allclose(z_full, z_half, atol=1e-3)


class TestAriaContrastiveEncoder:
    def test_encode_shape(self):
        from model_improvement.autoresearch_contrastive import AriaContrastiveEncoder
        enc = AriaContrastiveEncoder(input_dim=512, hidden_dim=512, projection_dim=256)
        x = torch.randn(2, 512)
        z = enc.encode(x)
        assert z.shape == (2, 512)

    def test_project_shape(self):
        from model_improvement.autoresearch_contrastive import AriaContrastiveEncoder
        enc = AriaContrastiveEncoder(input_dim=512, hidden_dim=512, projection_dim=256)
        z = torch.randn(2, 512)
        proj = enc.project(z)
        assert proj.shape == (2, 256)

    def test_project_is_normalized(self):
        from model_improvement.autoresearch_contrastive import AriaContrastiveEncoder
        enc = AriaContrastiveEncoder(input_dim=512, hidden_dim=512, projection_dim=256)
        z = torch.randn(2, 512)
        proj = enc.project(z)
        norms = proj.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(2), atol=1e-5)

    def test_no_mask_required(self):
        from model_improvement.autoresearch_contrastive import AriaContrastiveEncoder
        enc = AriaContrastiveEncoder(input_dim=512, hidden_dim=512, projection_dim=256)
        x = torch.randn(2, 512)
        z = enc.encode(x)
        assert z.shape == (2, 512)
