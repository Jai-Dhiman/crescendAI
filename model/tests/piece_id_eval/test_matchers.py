"""Verify all three matchers implement the Matcher protocol and find the right piece on toy data."""
from __future__ import annotations

import numpy as np
import pytest

from piece_id_eval.matchers import ChordNgramMatcher, DtwCeilingMatcher, TwoDFTMatcher
from piece_id_eval.matchers.base import Matcher, Ranked
from piece_id_eval.score_chroma import build_score_chroma


def _simple_catalog() -> dict[str, np.ndarray]:
    """3-piece catalog: each piece has a distinct dominant pitch class."""
    catalog: dict[str, np.ndarray] = {}
    frame_rate = 10.0
    for i, pc in enumerate([0, 4, 7]):  # C, E, G
        notes = [{"pitch": 60 + pc, "onset_seconds": 0.0, "duration_seconds": 3.0}]
        catalog[f"piece_{i}"] = build_score_chroma(notes, frame_rate)
    return catalog


def test_matcher_protocol_dtw() -> None:
    catalog = _simple_catalog()
    m = DtwCeilingMatcher(catalog, oti=False)
    assert isinstance(m, Matcher)
    assert isinstance(m.name, str)


def test_matcher_protocol_chord_ngram() -> None:
    catalog = _simple_catalog()
    m = ChordNgramMatcher(catalog, oti=False, n=2)
    assert isinstance(m, Matcher)


def test_matcher_protocol_twodft() -> None:
    catalog = _simple_catalog()
    m = TwoDFTMatcher(catalog, oti=False)
    assert isinstance(m, Matcher)


def test_dtw_ceiling_ranks_own_piece_first() -> None:
    catalog = _simple_catalog()
    m = DtwCeilingMatcher(catalog, oti=False)
    # Query from piece_0's own chroma (circularity sanity)
    query = catalog["piece_0"].copy()
    ranked = m.rank(query)
    assert len(ranked) == 3
    assert ranked[0][0] == "piece_0", f"expected piece_0 first, got {ranked[0][0]}"


def test_chord_ngram_ranks_own_piece_first() -> None:
    catalog = _simple_catalog()
    m = ChordNgramMatcher(catalog, oti=False, n=2)
    query = catalog["piece_1"].copy()
    ranked = m.rank(query)
    assert ranked[0][0] == "piece_1", f"expected piece_1 first, got {ranked[0][0]}"


def test_twodft_ranks_own_piece_first() -> None:
    catalog = _simple_catalog()
    m = TwoDFTMatcher(catalog, oti=False)
    query = catalog["piece_2"].copy()
    ranked = m.rank(query)
    assert ranked[0][0] == "piece_2", f"expected piece_2 first, got {ranked[0][0]}"


def test_ranked_result_is_descending() -> None:
    catalog = _simple_catalog()
    m = TwoDFTMatcher(catalog, oti=False)
    query = catalog["piece_0"].copy()
    ranked = m.rank(query)
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True), f"scores not descending: {scores}"


def test_dtw_result_is_list_of_tuples() -> None:
    catalog = _simple_catalog()
    m = DtwCeilingMatcher(catalog, oti=False)
    ranked = m.rank(catalog["piece_0"])
    for item in ranked:
        assert isinstance(item, Ranked)
        assert isinstance(item.piece_id, str)
        assert isinstance(item.score, float)
