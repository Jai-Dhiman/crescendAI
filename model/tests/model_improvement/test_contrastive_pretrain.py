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


class TestWeightedTierSampler:
    def test_respects_tier_weights(self):
        from model_improvement.autoresearch_contrastive import (
            ContrastiveSegmentDataset, WeightedTierSampler,
        )
        t1_items = [{"embedding": torch.randn(8), "piece_id": i // 2, "quality_score": 0.5} for i in range(10)]
        t2_items = [{"embedding": torch.randn(8), "piece_id": 100 + i // 2, "quality_score": 0.5} for i in range(40)]
        ds_t1 = ContrastiveSegmentDataset(t1_items)
        ds_t2 = ContrastiveSegmentDataset(t2_items)
        sampler = WeightedTierSampler(
            datasets=[ds_t1, ds_t2], weights=[0.5, 0.5], total_samples=100, seed=42,
        )
        indices = list(sampler)
        t1_count = sum(1 for i in indices if i < 10)
        assert 30 <= t1_count <= 70

    def test_guarantees_multi_piece_batches(self):
        from model_improvement.autoresearch_contrastive import (
            ContrastiveSegmentDataset, WeightedTierSampler,
        )
        items = [{"embedding": torch.randn(8), "piece_id": i // 3, "quality_score": float(i % 3) / 2} for i in range(12)]
        ds = ContrastiveSegmentDataset(items)
        sampler = WeightedTierSampler(datasets=[ds], weights=[1.0], total_samples=12, seed=42)
        indices = list(sampler)
        piece_ids = {items[i]["piece_id"] for i in indices}
        assert len(piece_ids) >= 2

    def test_length(self):
        from model_improvement.autoresearch_contrastive import (
            ContrastiveSegmentDataset, WeightedTierSampler,
        )
        items = [{"embedding": torch.randn(8), "piece_id": 0, "quality_score": 0.5} for _ in range(10)]
        ds = ContrastiveSegmentDataset(items)
        sampler = WeightedTierSampler(datasets=[ds], weights=[1.0], total_samples=20, seed=42)
        assert len(sampler) == 20


class TestLoadT2Records:
    def test_train_excludes_cliburn_2022(self):
        from model_improvement.autoresearch_contrastive import _load_t2_records
        train = _load_t2_records("train")
        for r in train:
            assert not (r["competition"] == "cliburn" and r["edition"] == 2022)

    def test_val_is_cliburn_2022_only(self):
        from model_improvement.autoresearch_contrastive import _load_t2_records
        val = _load_t2_records("val")
        for r in val:
            assert r["competition"] == "cliburn" and r["edition"] == 2022

    def test_train_and_val_are_disjoint(self):
        from model_improvement.autoresearch_contrastive import _load_t2_records
        train_ids = {r["recording_id"] for r in _load_t2_records("train")}
        val_ids = {r["recording_id"] for r in _load_t2_records("val")}
        assert train_ids.isdisjoint(val_ids)


class TestContrastivePretrainModel:
    def _make_batch(self, encoder_type="muq"):
        if encoder_type == "muq":
            return {
                "embeddings": torch.randn(4, 20, 1024),
                "mask": torch.ones(4, 20, dtype=torch.bool),
                "piece_ids": torch.tensor([0, 0, 1, 1]),
                "quality_scores": torch.tensor([0.9, 0.1, 0.8, 0.2]),
            }
        else:
            return {
                "embeddings": torch.randn(4, 512),
                "piece_ids": torch.tensor([0, 0, 1, 1]),
                "quality_scores": torch.tensor([0.9, 0.1, 0.8, 0.2]),
            }

    def test_training_step_returns_loss(self):
        from model_improvement.autoresearch_contrastive import (
            ContrastivePretrainModel, MuQContrastiveEncoder,
        )
        encoder = MuQContrastiveEncoder()
        model = ContrastivePretrainModel(encoder=encoder)
        batch = self._make_batch("muq")
        loss = model.training_step(batch, 0)
        assert loss.ndim == 0
        assert loss.requires_grad

    def test_training_step_aria(self):
        from model_improvement.autoresearch_contrastive import (
            ContrastivePretrainModel, AriaContrastiveEncoder,
        )
        encoder = AriaContrastiveEncoder()
        model = ContrastivePretrainModel(encoder=encoder)
        batch = self._make_batch("aria")
        loss = model.training_step(batch, 0)
        assert loss.ndim == 0

    def test_validation_step_logs(self):
        from model_improvement.autoresearch_contrastive import (
            ContrastivePretrainModel, MuQContrastiveEncoder,
        )
        encoder = MuQContrastiveEncoder()
        model = ContrastivePretrainModel(encoder=encoder)
        batch = self._make_batch("muq")
        model.validation_step(batch, 0)

    def test_configure_optimizers(self):
        from model_improvement.autoresearch_contrastive import (
            ContrastivePretrainModel, MuQContrastiveEncoder,
        )
        encoder = MuQContrastiveEncoder()
        model = ContrastivePretrainModel(encoder=encoder)
        opt_config = model.configure_optimizers()
        assert "optimizer" in opt_config
        assert "lr_scheduler" in opt_config


class TestIntegration:
    def test_full_training_loop_muq(self):
        """End-to-end: build dataset, train 2 epochs, verify loss is finite."""
        from torch.utils.data import DataLoader
        from model_improvement.autoresearch_contrastive import (
            MuQContrastiveEncoder, ContrastivePretrainModel,
            ContrastiveSegmentDataset, WeightedTierSampler,
            contrastive_collate_muq,
        )
        import pytorch_lightning as pl

        items = []
        for piece in range(2):
            for seg in range(4):
                items.append({
                    "embedding": torch.randn(20, 1024),
                    "piece_id": piece,
                    "quality_score": float(seg) / 3.0,
                })
        ds = ContrastiveSegmentDataset(items)
        sampler = WeightedTierSampler([ds], [1.0], total_samples=8, seed=42)
        loader = DataLoader(ds, batch_size=4, sampler=sampler, collate_fn=contrastive_collate_muq)

        encoder = MuQContrastiveEncoder(input_dim=1024, hidden_dim=64, projection_dim=32)
        model = ContrastivePretrainModel(
            encoder=encoder, max_epochs=2, warmup_epochs=1,
            lambda_infonce=1.0, lambda_ordinal=0.5,
        )

        trainer = pl.Trainer(
            max_epochs=2, accelerator="cpu", devices=1,
            enable_progress_bar=False, enable_model_summary=False,
            num_sanity_val_steps=0,
        )
        trainer.fit(model, loader, loader)
        assert trainer.callback_metrics["train_loss"].isfinite()

    def test_full_training_loop_aria(self):
        """End-to-end for Aria encoder."""
        from torch.utils.data import DataLoader
        from model_improvement.autoresearch_contrastive import (
            AriaContrastiveEncoder, ContrastivePretrainModel,
            ContrastiveSegmentDataset, WeightedTierSampler,
            contrastive_collate_aria,
        )
        import pytorch_lightning as pl

        items = []
        for piece in range(2):
            for seg in range(4):
                items.append({
                    "embedding": torch.randn(512),
                    "piece_id": piece,
                    "quality_score": float(seg) / 3.0,
                })
        ds = ContrastiveSegmentDataset(items)
        sampler = WeightedTierSampler([ds], [1.0], total_samples=8, seed=42)
        loader = DataLoader(ds, batch_size=4, sampler=sampler, collate_fn=contrastive_collate_aria)

        encoder = AriaContrastiveEncoder(input_dim=512, hidden_dim=64, projection_dim=32)
        model = ContrastivePretrainModel(
            encoder=encoder, max_epochs=2, warmup_epochs=1,
        )

        trainer = pl.Trainer(
            max_epochs=2, accelerator="cpu", devices=1,
            enable_progress_bar=False, enable_model_summary=False,
            num_sanity_val_steps=0,
        )
        trainer.fit(model, loader, loader)
        assert trainer.callback_metrics["train_loss"].isfinite()
