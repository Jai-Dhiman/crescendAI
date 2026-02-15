import torch
import pytest
from model_improvement.audio_encoders import MuQLoRAModel, MuQStagedModel, MuQFullUnfreezeModel


class TestMuQLoRAModel:
    def test_forward_shape(self):
        model = MuQLoRAModel(
            input_dim=1024, hidden_dim=512, num_labels=19,
            use_pretrained_muq=False,
        )
        x_a = torch.randn(4, 100, 1024)
        x_b = torch.randn(4, 80, 1024)
        mask_a = torch.ones(4, 100, dtype=torch.bool)
        mask_b = torch.ones(4, 80, dtype=torch.bool)
        out = model(x_a, x_b, mask_a, mask_b)
        assert out["ranking_logits"].shape == (4, 19)
        assert out["z_a"].shape == (4, 512)
        assert out["z_b"].shape == (4, 512)

    def test_regression_forward(self):
        model = MuQLoRAModel(
            input_dim=1024, hidden_dim=512, num_labels=19,
            use_pretrained_muq=False,
        )
        x = torch.randn(4, 100, 1024)
        mask = torch.ones(4, 100, dtype=torch.bool)
        scores = model.predict_scores(x, mask)
        assert scores.shape == (4, 19)
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_training_step_returns_loss(self):
        model = MuQLoRAModel(
            input_dim=1024, hidden_dim=512, num_labels=19,
            use_pretrained_muq=False,
        )
        batch = {
            "embeddings_a": torch.randn(4, 50, 1024),
            "embeddings_b": torch.randn(4, 50, 1024),
            "mask_a": torch.ones(4, 50, dtype=torch.bool),
            "mask_b": torch.ones(4, 50, dtype=torch.bool),
            "labels_a": torch.rand(4, 19),
            "labels_b": torch.rand(4, 19),
            "piece_ids_a": torch.tensor([0, 0, 1, 1]),
            "piece_ids_b": torch.tensor([0, 0, 1, 1]),
        }
        loss = model.training_step(batch, 0)
        assert loss.ndim == 0
        assert loss.item() > 0


class TestMuQStagedModel:
    def test_stage1_self_supervised_step(self):
        model = MuQStagedModel(
            input_dim=1024, hidden_dim=512, num_labels=19,
            use_pretrained_muq=False, stage="self_supervised",
        )
        batch = {
            "embeddings_clean": torch.randn(4, 50, 1024),
            "embeddings_augmented": torch.randn(4, 50, 1024),
            "mask": torch.ones(4, 50, dtype=torch.bool),
            "piece_ids": torch.tensor([0, 0, 1, 1]),
        }
        loss = model.training_step(batch, 0)
        assert loss.ndim == 0

    def test_stage2_supervised_step(self):
        model = MuQStagedModel(
            input_dim=1024, hidden_dim=512, num_labels=19,
            use_pretrained_muq=False, stage="supervised",
        )
        batch = {
            "embeddings_a": torch.randn(4, 50, 1024),
            "embeddings_b": torch.randn(4, 50, 1024),
            "mask_a": torch.ones(4, 50, dtype=torch.bool),
            "mask_b": torch.ones(4, 50, dtype=torch.bool),
            "labels_a": torch.rand(4, 19),
            "labels_b": torch.rand(4, 19),
            "piece_ids_a": torch.tensor([0, 0, 1, 1]),
            "piece_ids_b": torch.tensor([0, 0, 1, 1]),
        }
        loss = model.training_step(batch, 0)
        assert loss.ndim == 0

    def test_switch_stage(self):
        model = MuQStagedModel(
            input_dim=1024, hidden_dim=512, num_labels=19,
            use_pretrained_muq=False, stage="self_supervised",
        )
        model.switch_to_supervised()
        assert model.stage == "supervised"


class TestMuQFullUnfreezeModel:
    def test_gradual_unfreezing(self):
        model = MuQFullUnfreezeModel(
            input_dim=256, hidden_dim=128, num_labels=19,
            use_pretrained_muq=False,
            unfreeze_schedule={0: [3], 10: [2], 20: [1], 30: [0]},
            mock_num_layers=4,
        )
        model.unfreeze_for_epoch(0)
        unfrozen = model.get_unfrozen_layers()
        assert 3 in unfrozen

    def test_discriminative_lr(self):
        model = MuQFullUnfreezeModel(
            input_dim=256, hidden_dim=128, num_labels=19,
            use_pretrained_muq=False,
            unfreeze_schedule={0: [3], 10: [2]},
            lr_decay_factor=0.5,
            mock_num_layers=4,
        )
        model.unfreeze_for_epoch(0)
        optim_config = model.configure_optimizers()
        param_groups = optim_config["optimizer"].param_groups
        assert len(param_groups) > 1

    def test_training_step_returns_loss(self):
        model = MuQFullUnfreezeModel(
            input_dim=256, hidden_dim=128, num_labels=19,
            use_pretrained_muq=False,
            unfreeze_schedule={0: [3]},
            mock_num_layers=4,
        )
        model.unfreeze_for_epoch(0)
        batch = {
            "embeddings_a": torch.randn(4, 50, 256),
            "embeddings_b": torch.randn(4, 50, 256),
            "mask_a": torch.ones(4, 50, dtype=torch.bool),
            "mask_b": torch.ones(4, 50, dtype=torch.bool),
            "labels_a": torch.rand(4, 19),
            "labels_b": torch.rand(4, 19),
            "piece_ids_a": torch.tensor([0, 0, 1, 1]),
            "piece_ids_b": torch.tensor([0, 0, 1, 1]),
        }
        loss = model.training_step(batch, 0)
        assert loss.ndim == 0
        assert loss.item() > 0
