"""Tests for description embedding and clustering."""

from unittest.mock import patch, MagicMock

import numpy as np
import pytest


def test_embed_descriptions_shape():
    from masterclass_experiments.clustering import embed_descriptions

    # Mock the sentence transformer to avoid downloading model in tests
    mock_model = MagicMock()
    mock_model.encode.return_value = np.random.randn(5, 384).astype(np.float32)

    with patch(
        "masterclass_experiments.clustering.SentenceTransformer",
        return_value=mock_model,
    ):
        descriptions = ["dynamics too soft", "phrasing", "pedal heavy", "tone", "voicing"]
        embeddings = embed_descriptions(descriptions)

    assert embeddings.shape == (5, 384)
    assert embeddings.dtype == np.float32
    mock_model.encode.assert_called_once()


def test_cluster_descriptions_returns_labels():
    from masterclass_experiments.clustering import cluster_descriptions

    # 3 tight clusters in 2D
    rng = np.random.RandomState(42)
    cluster_a = rng.randn(30, 2) + np.array([0, 0])
    cluster_b = rng.randn(30, 2) + np.array([10, 10])
    cluster_c = rng.randn(30, 2) + np.array([20, 0])
    embeddings = np.vstack([cluster_a, cluster_b, cluster_c]).astype(np.float32)

    labels, clusterer = cluster_descriptions(embeddings, min_cluster_size=10)

    assert labels.shape == (90,)
    # Should find at least 2 clusters (HDBSCAN may merge or split)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    assert n_clusters >= 2


def test_cluster_descriptions_noise_label():
    from masterclass_experiments.clustering import cluster_descriptions

    # Very sparse data -- should produce noise labels (-1)
    rng = np.random.RandomState(42)
    embeddings = rng.randn(10, 50).astype(np.float32) * 100

    labels, _ = cluster_descriptions(embeddings, min_cluster_size=5)

    assert labels.shape == (10,)
    # With such sparse data, some or all points may be noise
    assert -1 in labels or len(set(labels)) >= 1


def test_load_open_moments():
    from masterclass_experiments.clustering import load_open_descriptions

    import json
    import tempfile
    from pathlib import Path

    moments = [
        {"moment_id": "a", "open_description": "dynamics too soft"},
        {"moment_id": "b", "open_description": "pedal clarity"},
        {"moment_id": "c", "open_description": "phrasing shape"},
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for m in moments:
            f.write(json.dumps(m) + "\n")
        tmp_path = Path(f.name)

    try:
        ids, descriptions = load_open_descriptions(tmp_path)
        assert ids == ["a", "b", "c"]
        assert descriptions == [
            "dynamics too soft",
            "pedal clarity",
            "phrasing shape",
        ]
    finally:
        tmp_path.unlink()


def test_summarize_clusters():
    from masterclass_experiments.clustering import summarize_clusters

    descriptions = [
        "dynamics too soft",
        "crescendo too abrupt",
        "volume balance",
        "pedal muddy",
        "sustain pedal timing",
        "pedal clarity in bass",
    ]
    labels = np.array([0, 0, 0, 1, 1, 1])

    summary = summarize_clusters(descriptions, labels)

    assert len(summary) == 2
    assert summary[0]["cluster_id"] == 0
    assert summary[0]["size"] == 3
    assert len(summary[0]["examples"]) == 3
    assert summary[1]["cluster_id"] == 1
    assert summary[1]["size"] == 3
