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


class TestContrastiveSegmentDataset:
    def _make_t1_data(self):
        embeddings = {
            "seg_a1": torch.randn(10, 1024),
            "seg_a2": torch.randn(12, 1024),
            "seg_b1": torch.randn(8, 1024),
        }
        labels = {
            "seg_a1": [0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
            "seg_a2": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "seg_b1": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        }
        piece_to_keys = {"piece_X": ["seg_a1", "seg_a2"], "piece_Y": ["seg_b1"]}
        keys = ["seg_a1", "seg_a2", "seg_b1"]
        return embeddings, labels, piece_to_keys, keys

    def test_t1_items_have_correct_schema(self):
        from model_improvement.autoresearch_contrastive import ContrastiveSegmentDataset
        emb, labels, p2k, keys = self._make_t1_data()
        ds = ContrastiveSegmentDataset.from_t1(emb, labels, p2k, keys, piece_id_offset=0)
        item = ds[0]
        assert "embedding" in item
        assert "piece_id" in item
        assert "quality_score" in item
        assert isinstance(item["piece_id"], int)
        assert 0.0 <= item["quality_score"] <= 1.0

    def test_t1_piece_ids_are_consistent(self):
        from model_improvement.autoresearch_contrastive import ContrastiveSegmentDataset
        emb, labels, p2k, keys = self._make_t1_data()
        ds = ContrastiveSegmentDataset.from_t1(emb, labels, p2k, keys, piece_id_offset=0)
        items = [ds[i] for i in range(len(ds))]
        ids = {item["piece_id"] for item in items}
        assert len(ids) == 2

    def test_piece_id_offset(self):
        from model_improvement.autoresearch_contrastive import ContrastiveSegmentDataset
        emb, labels, p2k, keys = self._make_t1_data()
        ds = ContrastiveSegmentDataset.from_t1(emb, labels, p2k, keys, piece_id_offset=100)
        item = ds[0]
        assert item["piece_id"] >= 100

    def test_length_matches_keys(self):
        from model_improvement.autoresearch_contrastive import ContrastiveSegmentDataset
        emb, labels, p2k, keys = self._make_t1_data()
        ds = ContrastiveSegmentDataset.from_t1(emb, labels, p2k, keys, piece_id_offset=0)
        assert len(ds) == 3


class TestContrastiveCollate:
    def test_muq_collate_pads_and_masks(self):
        from model_improvement.autoresearch_contrastive import contrastive_collate_muq
        batch = [
            {"embedding": torch.randn(10, 1024), "piece_id": 0, "quality_score": 0.8},
            {"embedding": torch.randn(20, 1024), "piece_id": 0, "quality_score": 0.3},
        ]
        out = contrastive_collate_muq(batch)
        assert out["embeddings"].shape == (2, 20, 1024)
        assert out["mask"].shape == (2, 20)
        assert out["mask"][0, 9].item() is True
        assert out["mask"][0, 10].item() is False
        assert out["mask"][1].all()
        assert out["piece_ids"].shape == (2,)
        assert out["quality_scores"].shape == (2,)

    def test_aria_collate_stacks(self):
        from model_improvement.autoresearch_contrastive import contrastive_collate_aria
        batch = [
            {"embedding": torch.randn(512), "piece_id": 0, "quality_score": 0.8},
            {"embedding": torch.randn(512), "piece_id": 1, "quality_score": 0.3},
        ]
        out = contrastive_collate_aria(batch)
        assert out["embeddings"].shape == (2, 512)
        assert "mask" not in out
        assert out["piece_ids"].shape == (2,)
