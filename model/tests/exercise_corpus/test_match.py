"""Tests for match.py -- cosine-ranked retrieval over the catalog embeddings.

The matcher is pure linear algebra over what catalog.write_primitives already
stored, so these tests build a tiny synthetic catalog (random 512-dim vectors,
no Aria weights required) and assert ranking behavior through the public
match_exercises interface.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from exercise_corpus import Primitive
from exercise_corpus.catalog import write_primitives
from exercise_corpus.match import Match, load_index, match_exercises


def _make_catalog(tmp_path: Path, vectors: dict[str, np.ndarray]) -> Path:
    """Write a synthetic catalog DB from {primitive_id: 512-dim vector}."""
    primitives = []
    embeddings = {}
    for i, (pid, vec) in enumerate(vectors.items(), start=1):
        source = pid.split("_")[0]
        primitives.append(
            Primitive(
                primitive_id=pid,
                source=source,
                source_exercise_number=i,
                title=f"{source} {i}",
                musicxml_path=tmp_path / f"{pid}.xml",
                midi_path=tmp_path / f"{pid}.mid",
                n_notes=100 + i,
            )
        )
        embeddings[pid] = torch.from_numpy(vec.astype(np.float32))
    db_path = tmp_path / "cat.db"
    write_primitives(primitives, embeddings, db_path)
    return db_path


def test_self_retrieval_ranks_itself_first(tmp_path: Path):
    rng = np.random.default_rng(0)
    vectors = {f"hanon_{i:03d}": rng.standard_normal(512) for i in range(1, 6)}
    db = _make_catalog(tmp_path, vectors)

    query = vectors["hanon_003"]
    results = match_exercises(query, db_path=db, top_k=5)

    assert isinstance(results[0], Match)
    assert results[0].primitive_id == "hanon_003"
    assert results[0].score == pytest.approx(1.0, abs=1e-5)
    # scores must be sorted descending
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_top_k_bounds_result_count(tmp_path: Path):
    rng = np.random.default_rng(1)
    vectors = {f"hanon_{i:03d}": rng.standard_normal(512) for i in range(1, 11)}
    db = _make_catalog(tmp_path, vectors)
    results = match_exercises(vectors["hanon_001"], db_path=db, top_k=3)
    assert len(results) == 3


def test_source_filter_restricts_candidates(tmp_path: Path):
    rng = np.random.default_rng(2)
    vectors = {f"hanon_{i:03d}": rng.standard_normal(512) for i in range(1, 4)}
    vectors.update({f"czerny_{i:03d}": rng.standard_normal(512) for i in range(1, 4)})
    db = _make_catalog(tmp_path, vectors)
    results = match_exercises(
        vectors["hanon_001"], db_path=db, top_k=10, sources=["czerny"]
    )
    assert {r.source for r in results} == {"czerny"}
    assert len(results) == 3


def test_deterministic_tie_break_by_primitive_id(tmp_path: Path):
    # Two identical vectors -> identical cosine to the query -> tie broken by id.
    shared = np.ones(512)
    vectors = {"hanon_002": shared.copy(), "hanon_001": shared.copy()}
    db = _make_catalog(tmp_path, vectors)
    results = match_exercises(shared, db_path=db, top_k=2)
    assert [r.primitive_id for r in results] == ["hanon_001", "hanon_002"]


def test_rejects_wrong_dimension_query(tmp_path: Path):
    rng = np.random.default_rng(3)
    vectors = {f"hanon_{i:03d}": rng.standard_normal(512) for i in range(1, 6)}
    db = _make_catalog(tmp_path, vectors)
    with pytest.raises(ValueError, match="512"):
        match_exercises(np.zeros(128), db_path=db, top_k=3)


def test_requires_db_or_index(tmp_path: Path):
    with pytest.raises(ValueError, match="db_path or index"):
        match_exercises(np.zeros(512), top_k=3)


def test_load_index_reuse_matches_db_path(tmp_path: Path):
    rng = np.random.default_rng(4)
    vectors = {f"hanon_{i:03d}": rng.standard_normal(512) for i in range(1, 6)}
    db = _make_catalog(tmp_path, vectors)
    index = load_index(db)
    via_index = match_exercises(vectors["hanon_002"], index=index, top_k=5)
    via_db = match_exercises(vectors["hanon_002"], db_path=db, top_k=5)
    assert [r.primitive_id for r in via_index] == [r.primitive_id for r in via_db]
