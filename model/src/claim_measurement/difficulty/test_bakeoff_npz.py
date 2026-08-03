"""Tests for the shared .npz embedding contract.

Run: cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
"""
import numpy as np
import pytest

from claim_measurement.difficulty import bakeoff_npz
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


def test_crash_mid_write_leaves_no_partial_or_temp_file(tmp_path, monkeypatch):
    """The resume guard treats "file exists" as "already done", so a crashed
    write must leave nothing behind at the destination path."""
    path = tmp_path / "piece_003.npz"

    def savez_that_dies_after_writing_bytes(fh, **arrays):
        fh.write(b"truncated garbage")
        raise RuntimeError("simulated OOM mid-write")

    monkeypatch.setattr(bakeoff_npz.np, "savez", savez_that_dies_after_writing_bytes)

    with pytest.raises(RuntimeError, match="simulated OOM"):
        write_embedding_npz(path, {"embedding": np.zeros(3, dtype=np.float32)},
                             grade=1, composer_id=0)

    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_existing_file_is_replaced_not_appended(tmp_path):
    path = tmp_path / "piece_004.npz"
    write_embedding_npz(path, {"embedding": np.array([1.0], dtype=np.float32)}, grade=1, composer_id=0)
    write_embedding_npz(path, {"embedding": np.array([2.0], dtype=np.float32)}, grade=5, composer_id=3)

    record = read_embedding_npz(path)

    np.testing.assert_array_equal(record.embeddings["embedding"], [2.0])
    assert (record.grade, record.composer_id) == (5, 3)
    assert list(tmp_path.iterdir()) == [path]
