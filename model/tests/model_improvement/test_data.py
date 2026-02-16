import torch
import pytest
from pathlib import Path
from model_improvement.data import (
    CompetitionDataset,
    PairedPerformanceDataset,
    AugmentedEmbeddingDataset,
    MIDIPretrainingDataset,
    ContinuousPretrainDataset,
    multi_task_collate_fn,
    symbolic_collate_fn,
    continuous_collate_fn,
)


def test_competition_dataset_returns_ordinal_labels():
    mock_data = [
        {"recording_id": "r1", "placement": 1, "embeddings": torch.randn(100, 1024)},
        {"recording_id": "r2", "placement": 2, "embeddings": torch.randn(80, 1024)},
        {"recording_id": "r3", "placement": 3, "embeddings": torch.randn(120, 1024)},
    ]
    ds = CompetitionDataset(mock_data, max_frames=1000)
    item = ds[0]
    assert "embeddings" in item
    assert "placement" in item


def test_paired_performance_dataset():
    mock_labels = {"k1": [0.5] * 19, "k2": [0.7] * 19}
    mock_pieces = {"piece1": ["k1", "k2"]}
    ds = PairedPerformanceDataset(
        cache_dir=Path("/tmp/test"),
        labels=mock_labels,
        piece_to_keys=mock_pieces,
        keys=["k1", "k2"],
    )
    assert len(ds) > 0


def test_midi_pretraining_dataset():
    tokens = list(range(100))
    ds = MIDIPretrainingDataset(
        token_sequences=[tokens],
        max_seq_len=512,
        mask_prob=0.15,
        vocab_size=500,
    )
    item = ds[0]
    assert "input_ids" in item
    assert "labels" in item
    assert "attention_mask" in item


def test_multi_task_collate():
    batch = [
        {"embeddings_a": torch.randn(50, 1024), "labels_a": torch.rand(19)},
        {"embeddings_a": torch.randn(30, 1024), "labels_a": torch.rand(19)},
    ]
    collated = multi_task_collate_fn(batch)
    assert collated["embeddings_a"].shape[0] == 2
    assert collated["embeddings_a"].shape[1] == 50  # padded to max


class TestSymbolicCollateFn:
    def test_produces_correct_batch_keys(self):
        token_sequences = {
            "k1": list(range(100)),
            "k2": list(range(80)),
        }
        batch = [
            {
                "key_a": "k1",
                "key_b": "k2",
                "labels_a": torch.rand(19),
                "labels_b": torch.rand(19),
                "piece_id": 0,
            }
        ]
        result = symbolic_collate_fn(batch, token_sequences, max_len=512)
        assert result is not None
        expected_keys = {
            "input_ids_a", "input_ids_b",
            "mask_a", "mask_b",
            "labels_a", "labels_b",
            "piece_ids_a", "piece_ids_b",
        }
        assert set(result.keys()) == expected_keys

    def test_padding_and_mask(self):
        token_sequences = {"k1": list(range(50)), "k2": list(range(30))}
        batch = [
            {
                "key_a": "k1",
                "key_b": "k2",
                "labels_a": torch.rand(19),
                "labels_b": torch.rand(19),
                "piece_id": 0,
            }
        ]
        result = symbolic_collate_fn(batch, token_sequences, max_len=100)
        assert result["input_ids_a"].shape == (1, 100)
        assert result["mask_a"][0, :50].all()
        assert not result["mask_a"][0, 50:].any()
        assert result["mask_b"][0, :30].all()
        assert not result["mask_b"][0, 30:].any()

    def test_missing_keys_returns_none(self):
        result = symbolic_collate_fn(
            [{"key_a": "missing", "key_b": "also_missing",
              "labels_a": torch.rand(19), "labels_b": torch.rand(19), "piece_id": 0}],
            {},
        )
        assert result is None


class TestContinuousCollateFn:
    def test_produces_correct_batch_keys(self):
        features = {
            "k1": torch.randn(100, 5),
            "k2": torch.randn(80, 5),
        }
        batch = [
            {
                "key_a": "k1",
                "key_b": "k2",
                "labels_a": torch.rand(19),
                "labels_b": torch.rand(19),
                "piece_id": 0,
            }
        ]
        result = continuous_collate_fn(batch, features, max_len=200)
        assert result is not None
        expected_keys = {
            "features_a", "features_b",
            "mask_a", "mask_b",
            "labels_a", "labels_b",
            "piece_ids_a", "piece_ids_b",
        }
        assert set(result.keys()) == expected_keys

    def test_padding_shape(self):
        features = {"k1": torch.randn(50, 5), "k2": torch.randn(30, 5)}
        batch = [
            {
                "key_a": "k1",
                "key_b": "k2",
                "labels_a": torch.rand(19),
                "labels_b": torch.rand(19),
                "piece_id": 0,
            }
        ]
        result = continuous_collate_fn(batch, features, max_len=100)
        assert result["features_a"].shape == (1, 100, 5)
        assert result["mask_a"][0, :50].all()
        assert not result["mask_a"][0, 50:].any()

    def test_missing_keys_returns_none(self):
        result = continuous_collate_fn(
            [{"key_a": "missing", "key_b": "also_missing",
              "labels_a": torch.rand(19), "labels_b": torch.rand(19), "piece_id": 0}],
            {},
        )
        assert result is None


class TestContinuousPretrainDataset:
    def test_returns_correct_item_format(self):
        features = {"k1": torch.randn(100, 5), "k2": torch.randn(80, 5)}
        ds = ContinuousPretrainDataset(["k1", "k2"], features, max_len=200)
        assert len(ds) == 2
        item = ds[0]
        expected_keys = {"features", "mask", "masked_features", "masked_positions"}
        assert set(item.keys()) == expected_keys
        assert item["features"].shape == (200, 5)
        assert item["mask"].shape == (200,)
        assert item["masked_features"].shape == (200, 5)
        assert item["masked_positions"].shape == (200,)

    def test_masking_zeroes_features(self):
        features = {"k1": torch.ones(50, 3)}
        ds = ContinuousPretrainDataset(["k1"], features, max_len=50, mask_prob=1.0)
        item = ds[0]
        # All valid positions should be masked
        assert item["masked_positions"][:50].all()
        # Masked positions should be zero
        assert (item["masked_features"][:50] == 0.0).all()

    def test_empty_keys_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            ContinuousPretrainDataset([], {}, max_len=100)
