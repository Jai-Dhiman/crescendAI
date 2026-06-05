# model/tests/piece_id_eval/test_matchers.py
"""Verify note-based matchers implement the Matcher protocol and find the right
piece on a small synthetic catalog. Each matcher's first test is score
self-query recall@1 == 1.0.
"""
from __future__ import annotations

from piece_id_eval.matchers.base import Matcher, Ranked
from piece_id_eval.matchers.landmark import LandmarkMatcher
from piece_id_eval.matchers.note_chroma_matcher import NoteChromaMatcher
from piece_id_eval.notes import Note


def _catalog_3piece() -> dict[str, list[Note]]:
    """3-piece synthetic catalog with distinct pitch classes."""
    pitches = [60, 64, 67]  # C, E, G
    catalog: dict[str, list[Note]] = {}
    for i, p in enumerate(pitches):
        catalog[f"piece_{i}"] = [
            Note(onset=j * 0.5, offset=j * 0.5 + 0.4, pitch=p, velocity=80)
            for j in range(20)
        ]
    return catalog


def test_note_chroma_matcher_protocol_compliance() -> None:
    catalog = _catalog_3piece()
    m = NoteChromaMatcher(catalog)
    assert isinstance(m, Matcher)
    assert isinstance(m.name, str)


def test_note_chroma_matcher_self_query_recall_at_1() -> None:
    catalog = _catalog_3piece()
    m = NoteChromaMatcher(catalog)
    for piece_id, notes in catalog.items():
        ranked = m.rank(notes)
        assert len(ranked) == 3
        assert ranked[0].piece_id == piece_id, (
            f"self-query for {piece_id} not first: {ranked[0].piece_id}"
        )


def test_note_chroma_matcher_returns_ranked_list() -> None:
    catalog = _catalog_3piece()
    m = NoteChromaMatcher(catalog)
    ranked = m.rank(catalog["piece_0"])
    assert all(isinstance(r, Ranked) for r in ranked)
    scores = [r.score for r in ranked]
    assert scores == sorted(scores, reverse=True), f"not descending: {scores}"


def test_landmark_matcher_skips_empty_note_entries() -> None:
    """Empty-note catalog entries must not appear in rank() output."""
    catalog = _catalog_3piece()
    catalog["empty_piece"] = []
    m = LandmarkMatcher(catalog)
    query = catalog["piece_0"]
    ranked = m.rank(query)
    ranked_ids = {r.piece_id for r in ranked}
    assert "empty_piece" not in ranked_ids, (
        f"empty_piece must not appear in rank output, got: {ranked_ids}"
    )
    assert len(ranked) == 3, f"expected 3 results (non-empty pieces), got {len(ranked)}"
