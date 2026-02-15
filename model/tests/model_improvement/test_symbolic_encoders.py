"""Tests for symbolic encoder experiments: S1 (Transformer), S2 (GNN), S3 (Continuous)."""

import torch
import pytest
from model_improvement.symbolic_encoders import TransformerSymbolicEncoder


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
