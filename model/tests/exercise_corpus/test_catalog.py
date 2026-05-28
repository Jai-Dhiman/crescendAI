# model/tests/exercise_corpus/test_catalog.py
import numpy as np
import pytest
import torch
from pathlib import Path

from exercise_corpus import Primitive
from exercise_corpus.catalog import write_primitives, read_primitives, CatalogRow


def _make_primitive(n: int) -> Primitive:
    return Primitive(
        primitive_id=f"hanon_{n:03d}",
        source="hanon",
        source_exercise_number=n,
        title=f"Hanon Exercise {n}",
        musicxml_path=Path(f"/fake/scores/hanon_{n:03d}.xml"),
        midi_path=Path(f"/fake/midi/hanon_{n:03d}.mid"),
        n_notes=8 * n,
    )


def test_round_trip_preserves_all_fields(tmp_path: Path):
    db_path = tmp_path / "test_catalog.db"
    primitives = [_make_primitive(i) for i in range(1, 4)]
    embeddings = {
        p.primitive_id: torch.randn(512) for p in primitives
    }
    write_primitives(primitives, embeddings, db_path)
    rows = read_primitives(db_path)

    assert len(rows) == 3
    row_by_id = {r.primitive_id: r for r in rows}
    for p in primitives:
        row = row_by_id[p.primitive_id]
        assert row.source == p.source
        assert row.source_exercise_number == p.source_exercise_number
        assert row.title == p.title
        assert str(row.musicxml_path) == str(p.musicxml_path)
        assert str(row.midi_path) == str(p.midi_path)
        assert row.n_notes == p.n_notes
        assert isinstance(row.embedding, np.ndarray)
        assert row.embedding.shape == (512,)
        expected = embeddings[p.primitive_id].numpy()
        np.testing.assert_array_equal(row.embedding, expected)


def test_created_at_is_populated(tmp_path: Path):
    db_path = tmp_path / "test_catalog.db"
    p = _make_primitive(1)
    write_primitives([p], {p.primitive_id: torch.randn(512)}, db_path)
    rows = read_primitives(db_path)
    assert rows[0].created_at is not None
    assert len(rows[0].created_at) > 0
