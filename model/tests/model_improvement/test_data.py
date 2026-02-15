import torch
import pytest
from pathlib import Path
from model_improvement.data import (
    CompetitionDataset,
    PairedPerformanceDataset,
    AugmentedEmbeddingDataset,
    MIDIPretrainingDataset,
    multi_task_collate_fn,
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
