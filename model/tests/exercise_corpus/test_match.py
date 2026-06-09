"""Tests for match.py -- cosine-ranked retrieval over the catalog embeddings.

The matcher is pure linear algebra over what catalog.write_primitives already
stored, so these tests build a tiny synthetic catalog (random 512-dim vectors,
no Aria weights required) and assert ranking behavior through the public
match_exercises interface.
"""

import math
from pathlib import Path

import numpy as np
import pytest
import torch

import exercise_corpus
from exercise_corpus import Primitive
from exercise_corpus.catalog import write_primitives
from exercise_corpus.match import (
    Match,
    NoPrimitiveForDimensionError,
    load_index,
    match_by_dimension,
    match_exercises,
)
from exercise_corpus.tags import TagSet, load_tags


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


def _dummy_catalog(tmp_path: Path, primitive_ids: list[str]) -> Path:
    """Synthetic catalog with arbitrary (unused-by-tag-retrieval) embeddings."""
    rng = np.random.default_rng(0)
    vectors = {pid: rng.standard_normal(512) for pid in primitive_ids}
    return _make_catalog(tmp_path, vectors)


def test_match_by_dimension_filters_to_tagged_primitives(tmp_path: Path):
    db = _dummy_catalog(
        tmp_path, ["hanon_001", "hanon_002", "hanon_003", "burgmuller_001"]
    )
    tags = {
        "hanon_001": TagSet(frozenset({"timing", "articulation"}), frozenset()),
        "hanon_002": TagSet(frozenset({"timing", "articulation"}), frozenset()),
        "hanon_003": TagSet(frozenset({"timing", "articulation"}), frozenset()),
        "burgmuller_001": TagSet(
            frozenset({"phrasing", "interpretation"}), frozenset()
        ),
    }

    results = match_by_dimension("timing", tags, db_path=db, top_k=5)

    ids = [m.primitive_id for m in results]
    assert ids == ["hanon_001", "hanon_002", "hanon_003"]  # burgmuller excluded
    assert "burgmuller_001" not in ids
    assert all(isinstance(m, Match) for m in results)
    # No cosine query in tag mode -> score is the nan sentinel.
    assert all(math.isnan(m.score) for m in results)
    # Deterministic across repeated calls.
    again = [
        m.primitive_id
        for m in match_by_dimension("timing", tags, db_path=db, top_k=5)
    ]
    assert again == ids
    # top_k bounds the result count.
    assert len(match_by_dimension("timing", tags, db_path=db, top_k=2)) == 2


def test_match_by_dimension_raises_for_untagged_dimension(tmp_path: Path):
    db = _dummy_catalog(tmp_path, ["hanon_001", "hanon_002"])
    tags = {
        "hanon_001": TagSet(frozenset({"timing", "articulation"}), frozenset()),
        "hanon_002": TagSet(frozenset({"timing", "articulation"}), frozenset()),
    }
    with pytest.raises(NoPrimitiveForDimensionError, match="pedaling"):
        match_by_dimension("pedaling", tags, db_path=db, top_k=5)


# Resolve the shipped tag file relative to the installed package, not CWD.
_PKG_DIR = Path(exercise_corpus.__file__).resolve().parent
SHIPPED_TAGS = _PKG_DIR / "technique_tags.toml"

# The 22 real primitive_ids (Hanon 1-20 + Czerny no.1 + Burgmuller no.1).
_REAL_IDS = [f"hanon_{i:03d}" for i in range(1, 21)] + ["czerny_001", "burgmuller_001"]


def test_shipped_tags_yield_expected_dimension_buckets(tmp_path: Path):
    # Synthetic catalog over the REAL ids -> no real DB / Aria weights needed.
    db = _dummy_catalog(tmp_path, _REAL_IDS)
    tags = load_tags(SHIPPED_TAGS, known_primitive_ids=set(_REAL_IDS))

    timing = {
        m.primitive_id
        for m in match_by_dimension("timing", tags, db_path=db, top_k=50)
    }
    assert "hanon_001" in timing
    assert "czerny_001" in timing
    assert "burgmuller_001" not in timing  # Hanon/Czerny are not phrasing studies

    phrasing = {
        m.primitive_id
        for m in match_by_dimension("phrasing", tags, db_path=db, top_k=50)
    }
    assert phrasing == {"burgmuller_001"}

    # Conservative authoring: nothing in this corpus teaches pedaling or dynamics.
    with pytest.raises(NoPrimitiveForDimensionError):
        match_by_dimension("pedaling", tags, db_path=db, top_k=50)
    with pytest.raises(NoPrimitiveForDimensionError):
        match_by_dimension("dynamics", tags, db_path=db, top_k=50)
