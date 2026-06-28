"""Unit tests for cosine_select.py (Python mirror of cosine-select.ts)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[4]))

from pipeline.exercise_routing.cosine_select import (
    cosine_select_within_dimension,
    load_catalog,
)

MANIFEST_PATH = (
    Path(__file__).parents[4]
    / "api" / "src" / "services" / "exercise_primitives_manifest.json"
)
ASSET_PATH = (
    Path(__file__).parents[4]
    / "api" / "src" / "services" / "exercise_embeddings.json"
)


def _catalog():
    manifest = json.loads(MANIFEST_PATH.read_text())
    return load_catalog(manifest, ASSET_PATH)


def _vector_of(cat, pid):
    return cat.matrix[cat.ids.index(pid)].copy()


def _first_for_dim(cat, dim):
    for pid in cat.ids:
        if dim in cat.manifest[pid]["dimensions"]:
            return pid
    raise AssertionError(f"no drill for {dim}")


def test_self_vector_selects_self():
    cat = _catalog()
    pid = _first_for_dim(cat, "timing")
    assert cosine_select_within_dimension(_vector_of(cat, pid), "timing", cat) == pid


def test_dimension_filter_respected():
    cat = _catalog()
    pedal = _first_for_dim(cat, "pedaling")
    chosen = cosine_select_within_dimension(_vector_of(cat, pedal), "timing", cat)
    assert "timing" in cat.manifest[chosen]["dimensions"]


def test_empty_bucket_returns_none():
    cat = _catalog()
    assert cosine_select_within_dimension(cat.matrix[0].copy(), "nope", cat) is None


def test_unnormalized_query_same_as_normalized():
    cat = _catalog()
    pid = _first_for_dim(cat, "phrasing")
    v = _vector_of(cat, pid) * 5.0
    assert cosine_select_within_dimension(v, "phrasing", cat) == pid


def test_zero_query_raises():
    cat = _catalog()
    with pytest.raises(ValueError, match="zero magnitude"):
        cosine_select_within_dimension(np.zeros(cat.dim, dtype=np.float32), "timing", cat)


def test_dim_mismatch_raises():
    cat = _catalog()
    with pytest.raises(ValueError, match="dim"):
        cosine_select_within_dimension(np.ones(3, dtype=np.float32), "timing", cat)


def test_parity_with_ts_default_via_self_query():
    # The TS test asserts a drill's own vector selects it within its dimension;
    # mirror that here so the two selectors cannot silently diverge.
    cat = _catalog()
    for dim in ("dynamics", "articulation", "interpretation"):
        pid = _first_for_dim(cat, dim)
        assert cosine_select_within_dimension(_vector_of(cat, pid), dim, cat) == pid
