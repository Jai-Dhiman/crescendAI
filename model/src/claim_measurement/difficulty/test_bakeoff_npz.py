"""Tests for the shared .npz embedding contract.

Run: cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
"""
import numpy as np

from claim_measurement.difficulty.bakeoff_npz import read_embedding_npz, write_embedding_npz


def test_round_trip_preserves_embedding_grade_composer(tmp_path):
    path = tmp_path / "piece_001.npz"
    write_embedding_npz(path, {"embedding": np.array([1.0, 2.0, 3.0], dtype=np.float32)},
                         grade=7, composer_id=42)

    record = read_embedding_npz(path)

    np.testing.assert_array_equal(record.embeddings["embedding"], [1.0, 2.0, 3.0])
    assert record.grade == 7
    assert record.composer_id == 42


def test_round_trip_preserves_multiple_poolings(tmp_path):
    path = tmp_path / "piece_002.npz"
    write_embedding_npz(
        path,
        {"mean_pool": np.array([0.1, 0.2], dtype=np.float32),
         "last_token": np.array([0.9, 0.8], dtype=np.float32)},
        grade=3, composer_id=1,
    )

    record = read_embedding_npz(path)

    assert set(record.embeddings) == {"mean_pool", "last_token"}
    np.testing.assert_allclose(record.embeddings["mean_pool"], [0.1, 0.2])
    np.testing.assert_allclose(record.embeddings["last_token"], [0.9, 0.8])
