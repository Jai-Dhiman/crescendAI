"""Tests for symbolic encoder experiments: S1 (Transformer), S2 (GNN), S2-hetero, S3 (Continuous)."""

import torch
import pytest
from model_improvement.symbolic_encoders import (
    TransformerSymbolicEncoder,
    GNNSymbolicEncoder,
    GNNHeteroSymbolicEncoder,
    ContinuousSymbolicEncoder,
)


class TestTransformerSymbolicEncoder:
    def test_forward_shape(self):
        model = TransformerSymbolicEncoder(
            vocab_size=500,
            d_model=512,
            nhead=8,
            num_layers=6,
            hidden_dim=512,
            num_labels=19,
        )
        input_ids = torch.randint(0, 500, (4, 128))
        attention_mask = torch.ones(4, 128, dtype=torch.bool)
        out = model(input_ids, attention_mask)
        assert out["z_symbolic"].shape == (4, 512)
        assert out["scores"].shape == (4, 19)

    def test_masked_lm_step(self):
        model = TransformerSymbolicEncoder(
            vocab_size=500,
            d_model=512,
            nhead=8,
            num_layers=6,
            hidden_dim=512,
            num_labels=19,
            stage="pretrain",
        )
        batch = {
            "input_ids": torch.randint(0, 500, (4, 128)),
            "labels": torch.randint(0, 500, (4, 128)),
            "attention_mask": torch.ones(4, 128, dtype=torch.bool),
        }
        loss = model.training_step(batch, 0)
        assert loss.ndim == 0

    def test_pairwise_ranking_step(self):
        model = TransformerSymbolicEncoder(
            vocab_size=500,
            d_model=512,
            nhead=8,
            num_layers=6,
            hidden_dim=512,
            num_labels=19,
            stage="finetune",
        )
        batch = {
            "input_ids_a": torch.randint(0, 500, (4, 128)),
            "input_ids_b": torch.randint(0, 500, (4, 128)),
            "mask_a": torch.ones(4, 128, dtype=torch.bool),
            "mask_b": torch.ones(4, 128, dtype=torch.bool),
            "labels_a": torch.rand(4, 19),
            "labels_b": torch.rand(4, 19),
            "piece_ids_a": torch.tensor([0, 0, 1, 1]),
            "piece_ids_b": torch.tensor([0, 0, 1, 1]),
        }
        loss = model.training_step(batch, 0)
        assert loss.ndim == 0


class TestGNNSymbolicEncoder:
    def test_forward_shape(self):
        model = GNNSymbolicEncoder(
            node_features=6,
            hidden_dim=512,
            num_layers=4,
            num_labels=19,
        )
        x = torch.randn(20, 6)  # pitch, velocity, onset, duration, pedal, voice
        edge_index = torch.randint(0, 20, (2, 40))
        batch_vec = torch.zeros(20, dtype=torch.long)
        out = model(x, edge_index, batch_vec)
        assert out["z_symbolic"].shape == (1, 512)
        assert out["scores"].shape == (1, 19)

    def test_link_prediction_step(self):
        model = GNNSymbolicEncoder(
            node_features=6,
            hidden_dim=512,
            num_layers=4,
            num_labels=19,
            stage="pretrain",
        )
        batch = {
            "x": torch.randn(20, 6),
            "edge_index": torch.randint(0, 20, (2, 40)),
            "batch": torch.zeros(20, dtype=torch.long),
            "pos_edges": torch.randint(0, 20, (2, 10)),
            "neg_edges": torch.randint(0, 20, (2, 10)),
        }
        loss = model.training_step(batch, 0)
        assert loss.ndim == 0


class TestGNNHeteroSymbolicEncoder:
    def _make_hetero_data(self, num_nodes=20):
        """Create synthetic HeteroData-like dicts for testing."""
        x_dict = {"note": torch.randn(num_nodes, 6)}
        edge_index_dict = {
            ("note", "onset", "note"): torch.randint(0, num_nodes, (2, 10)),
            ("note", "during", "note"): torch.randint(0, num_nodes, (2, 8)),
            ("note", "follow", "note"): torch.randint(0, num_nodes, (2, 12)),
            ("note", "silence", "note"): torch.randint(0, num_nodes, (2, 5)),
        }
        return x_dict, edge_index_dict

    def test_forward_shape(self):
        model = GNNHeteroSymbolicEncoder(
            node_features=6,
            hidden_dim=64,
            num_layers=3,
            num_labels=19,
        )
        x_dict, edge_index_dict = self._make_hetero_data()
        batch_vec = torch.zeros(20, dtype=torch.long)
        out = model(x_dict, edge_index_dict, batch_vec)
        assert out["z_symbolic"].shape == (1, 64)
        assert out["scores"].shape == (1, 19)

    def test_link_prediction_step(self):
        model = GNNHeteroSymbolicEncoder(
            node_features=6,
            hidden_dim=64,
            num_layers=3,
            num_labels=19,
            stage="pretrain",
        )
        x_dict, edge_index_dict = self._make_hetero_data()
        batch = {
            "x_dict": x_dict,
            "edge_index_dict": edge_index_dict,
            "batch": torch.zeros(20, dtype=torch.long),
            "pos_edges": torch.randint(0, 20, (2, 10)),
            "neg_edges": torch.randint(0, 20, (2, 10)),
        }
        loss = model.training_step(batch, 0)
        assert loss.ndim == 0

    def test_pairwise_ranking_step(self):
        model = GNNHeteroSymbolicEncoder(
            node_features=6,
            hidden_dim=64,
            num_layers=3,
            num_labels=19,
            stage="finetune",
        )
        x_dict_a, ei_dict_a = self._make_hetero_data(10)
        x_dict_b, ei_dict_b = self._make_hetero_data(15)
        batch = {
            "x_dict_a": x_dict_a,
            "edge_index_dict_a": ei_dict_a,
            "batch_a": torch.zeros(10, dtype=torch.long),
            "x_dict_b": x_dict_b,
            "edge_index_dict_b": ei_dict_b,
            "batch_b": torch.zeros(15, dtype=torch.long),
            "labels_a": torch.rand(1, 19),
            "labels_b": torch.rand(1, 19),
            "piece_ids_a": torch.tensor([0]),
            "piece_ids_b": torch.tensor([0]),
        }
        loss = model.training_step(batch, 0)
        assert loss.ndim == 0


class TestContinuousSymbolicEncoder:
    def test_forward_shape(self):
        model = ContinuousSymbolicEncoder(
            input_channels=5,
            hidden_dim=512,
            num_labels=19,
        )
        x = torch.randn(4, 200, 5)  # [B, T, C]
        mask = torch.ones(4, 200, dtype=torch.bool)
        out = model(x, mask)
        assert out["z_symbolic"].shape == (4, 512)
        assert out["scores"].shape == (4, 19)

    def test_contrastive_pretrain_step(self):
        model = ContinuousSymbolicEncoder(
            input_channels=5,
            hidden_dim=512,
            num_labels=19,
            stage="pretrain",
        )
        batch = {
            "features": torch.randn(4, 200, 5),
            "mask": torch.ones(4, 200, dtype=torch.bool),
            "masked_features": torch.randn(4, 200, 5),
            "masked_positions": torch.zeros(4, 200, dtype=torch.bool),
        }
        # Set some positions as masked
        batch["masked_positions"][:, 50:70] = True
        loss = model.training_step(batch, 0)
        assert loss.ndim == 0
