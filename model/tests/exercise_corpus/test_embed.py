"""Tests for embed.py through its public interface.

embed_primitives is a thin adapter. The test patches the underlying
extract_all_embeddings at the module boundary (not an internal) to avoid
requiring Aria weights in CI. The behavior under test is:
  - correct variant is passed through
  - the dict returned by the underlying call is returned as-is
  - FileNotFoundError from the underlying call propagates (no swallowing)
"""
import torch
import pytest
from pathlib import Path
from unittest.mock import patch

from exercise_corpus.embed import embed_primitives


def test_embed_primitives_passes_variant_and_returns_dict(tmp_path: Path):
    fake_result = {
        "hanon_001": torch.randn(512),
        "hanon_002": torch.randn(512),
    }
    with patch(
        "exercise_corpus.embed.extract_all_embeddings",
        return_value=fake_result,
    ) as mock_fn:
        result = embed_primitives(tmp_path)

    assert mock_fn.call_count == 1
    assert mock_fn.call_args.kwargs.get("variant") == "embedding"
    assert result is fake_result


def test_embed_primitives_propagates_file_not_found(tmp_path: Path):
    with patch(
        "exercise_corpus.embed.extract_all_embeddings",
        side_effect=FileNotFoundError("No .mid files found"),
    ):
        with pytest.raises(FileNotFoundError, match="No .mid files"):
            embed_primitives(tmp_path)
