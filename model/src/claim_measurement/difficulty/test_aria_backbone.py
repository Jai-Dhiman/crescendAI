"""Tests for the Aria backbone adapter (no real weights loaded).

Run: cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
"""
from pathlib import Path

import numpy as np
import torch

import model_improvement.aria_embeddings as aria_embeddings
from claim_measurement.difficulty.aria_backbone import AriaBackbone


def test_embed_wraps_extract_embedding_as_numpy(monkeypatch):
    def fake_extract_embedding(midi_path, variant="embedding"):
        assert variant == "embedding"
        return torch.tensor([1.0, 2.0, 3.0])

    monkeypatch.setattr(aria_embeddings, "extract_embedding", fake_extract_embedding)

    result = AriaBackbone().embed(Path("/fake/piece.mid"))

    assert set(result) == {"embedding"}
    assert isinstance(result["embedding"], np.ndarray)
    assert result["embedding"].dtype == np.float32
    np.testing.assert_allclose(result["embedding"], [1.0, 2.0, 3.0])
