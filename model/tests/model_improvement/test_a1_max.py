import torch
import pytest
import numpy as np
from pathlib import Path
from model_improvement.data import HardNegativePairSampler, PairedPerformanceDataset, apply_mixup


class TestHardNegativePairSampler:
    def _make_dataset(self):
        """Create a minimal PairedPerformanceDataset for testing."""
        labels = {
            "piece1_1": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            "piece1_2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "piece1_3": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            "piece2_1": [0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
            "piece2_2": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        }
        piece_to_keys = {
            "piece1": ["piece1_1", "piece1_2", "piece1_3"],
            "piece2": ["piece2_1", "piece2_2"],
        }
        keys = list(labels.keys())
        ds = PairedPerformanceDataset(
            cache_dir=Path("/tmp"), labels=labels,
            piece_to_keys=piece_to_keys, keys=keys,
        )
        return ds, labels

    def test_warmup_returns_all_pairs(self):
        """During warmup, should sample uniformly from all pairs."""
        ds, labels = self._make_dataset()
        sampler = HardNegativePairSampler(
            dataset=ds, warmup_epochs=5, easy_threshold=0.3,
        )
        indices = sampler.get_indices(epoch=2)
        assert len(indices) == len(ds)

    def test_post_warmup_filters_easy(self):
        """After warmup with curriculum, should filter some pairs."""
        ds, labels = self._make_dataset()
        sampler = HardNegativePairSampler(
            dataset=ds, warmup_epochs=5, easy_threshold=0.3,
        )
        indices = sampler.get_indices(epoch=10)
        assert len(indices) <= len(ds)
        assert len(indices) > 0

    def test_curriculum_progression(self):
        """Later epochs should include harder pairs."""
        ds, labels = self._make_dataset()
        sampler = HardNegativePairSampler(
            dataset=ds, warmup_epochs=2, easy_threshold=0.3,
        )
        early = sampler.get_indices(epoch=3)
        late = sampler.get_indices(epoch=50)
        assert len(late) >= len(early)


class TestMixup:
    def test_output_shapes_match_input(self):
        emb = torch.randn(4, 50, 1024)
        labels = torch.rand(4, 6)
        mixed_emb, mixed_labels = apply_mixup(emb, labels, alpha=0.2)
        assert mixed_emb.shape == emb.shape
        assert mixed_labels.shape == labels.shape

    def test_alpha_zero_returns_original(self):
        """Alpha=0 means no mixup, output equals input."""
        emb = torch.randn(4, 50, 1024)
        labels = torch.rand(4, 6)
        mixed_emb, mixed_labels = apply_mixup(emb, labels, alpha=0.0)
        assert torch.allclose(mixed_emb, emb)
        assert torch.allclose(mixed_labels, labels)

    def test_mixup_changes_values(self):
        """With alpha > 0, output should differ from input."""
        torch.manual_seed(42)
        emb = torch.randn(8, 50, 1024)
        labels = torch.rand(8, 6)
        mixed_emb, mixed_labels = apply_mixup(emb, labels, alpha=0.4)
        assert not torch.allclose(mixed_emb, emb)
