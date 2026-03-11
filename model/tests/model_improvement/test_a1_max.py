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


from model_improvement.audio_encoders import MuQLoRAMaxModel


class TestMuQLoRAMaxModel:
    def _make_batch(self, batch_size=4, seq_len=50, input_dim=1024, num_dims=6):
        return {
            "embeddings_a": torch.randn(batch_size, seq_len, input_dim),
            "embeddings_b": torch.randn(batch_size, seq_len, input_dim),
            "mask_a": torch.ones(batch_size, seq_len, dtype=torch.bool),
            "mask_b": torch.ones(batch_size, seq_len, dtype=torch.bool),
            "labels_a": torch.rand(batch_size, num_dims),
            "labels_b": torch.rand(batch_size, num_dims),
            "piece_ids_a": torch.tensor([0, 0, 1, 1]),
            "piece_ids_b": torch.tensor([0, 0, 1, 1]),
        }

    def test_training_step_returns_loss(self):
        model = MuQLoRAMaxModel(
            input_dim=1024, hidden_dim=512, num_labels=6,
            use_pretrained_muq=False,
        )
        batch = self._make_batch()
        loss = model.training_step(batch, 0)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_listmle_loss_included(self):
        """With lambda_listmle > 0, ListMLE should contribute to loss."""
        model = MuQLoRAMaxModel(
            input_dim=1024, hidden_dim=512, num_labels=6,
            use_pretrained_muq=False, lambda_listmle=1.5,
        )
        logged = {}
        model.log = lambda name, value, **kw: logged.update({name: value})
        batch = self._make_batch()
        model.training_step(batch, 0)
        assert "train_listmle_loss" in logged

    def test_ccc_loss_included(self):
        """With use_ccc=True, CCC loss should replace MSE."""
        model = MuQLoRAMaxModel(
            input_dim=1024, hidden_dim=512, num_labels=6,
            use_pretrained_muq=False, use_ccc=True,
        )
        logged = {}
        model.log = lambda name, value, **kw: logged.update({name: value})
        batch = self._make_batch()
        model.training_step(batch, 0)
        assert "train_ccc_loss" in logged

    def test_mixup_changes_loss(self):
        """With mixup_alpha > 0, loss should differ from no-mixup."""
        torch.manual_seed(42)
        model_no_mix = MuQLoRAMaxModel(
            input_dim=256, hidden_dim=128, num_labels=6,
            use_pretrained_muq=False, mixup_alpha=0.0,
        )
        torch.manual_seed(42)
        model_mix = MuQLoRAMaxModel(
            input_dim=256, hidden_dim=128, num_labels=6,
            use_pretrained_muq=False, mixup_alpha=0.4,
        )
        model_mix.load_state_dict(model_no_mix.state_dict())

        batch = self._make_batch(input_dim=256)
        torch.manual_seed(0)
        loss_no_mix = model_no_mix.training_step(batch, 0)
        torch.manual_seed(0)
        loss_mix = model_mix.training_step(batch, 0)
        assert not torch.isclose(loss_no_mix, loss_mix)

    def test_forward_compatible_with_parent(self):
        """forward() and predict_scores() should work identically."""
        model = MuQLoRAMaxModel(
            input_dim=1024, hidden_dim=512, num_labels=6,
            use_pretrained_muq=False,
        )
        x_a = torch.randn(2, 50, 1024)
        x_b = torch.randn(2, 50, 1024)
        out = model(x_a, x_b)
        assert out["ranking_logits"].shape == (2, 6)

        scores = model.predict_scores(x_a)
        assert scores.shape == (2, 6)

    def test_default_params_match_spec(self):
        """Default loss weights should match spec."""
        model = MuQLoRAMaxModel(
            input_dim=1024, hidden_dim=512, use_pretrained_muq=False,
        )
        assert model.hparams.lambda_listmle == 1.5
        assert model.hparams.lambda_contrastive == 0.3
        assert model.hparams.lambda_regression == 0.3
        assert model.hparams.lambda_invariance == 0.1
        assert model.hparams.use_ccc is True
        assert model.hparams.mixup_alpha == 0.2
