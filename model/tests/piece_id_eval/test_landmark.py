# model/tests/piece_id_eval/test_landmark.py
"""Verify LandmarkMatcher (C2) through its public interface."""
from __future__ import annotations

from piece_id_eval.matchers.base import Matcher, Ranked
from piece_id_eval.matchers.landmark import LandmarkMatcher
from piece_id_eval.notes import Note


def _make_piece(root_pitch: int, n: int = 30) -> list[Note]:
    """Piece with a repeating melodic pattern starting at root_pitch."""
    pattern = [0, 2, 4, 5, 7, 9, 11]
    return [
        Note(onset=i * 0.3, offset=i * 0.3 + 0.25, pitch=root_pitch + pattern[i % 7], velocity=80)
        for i in range(n)
    ]


def _catalog() -> dict[str, list[Note]]:
    return {
        "piece_c": _make_piece(60),   # C major scale pattern
        "piece_f": _make_piece(65),   # F major scale pattern
        "piece_g": _make_piece(67),   # G major scale pattern
    }


def test_landmark_matcher_protocol_compliance() -> None:
    m = LandmarkMatcher(_catalog())
    assert isinstance(m, Matcher)
    assert isinstance(m.name, str)


def test_landmark_self_query_recall_at_1() -> None:
    catalog = _catalog()
    m = LandmarkMatcher(catalog)
    for piece_id, notes in catalog.items():
        ranked = m.rank(notes)
        assert ranked[0].piece_id == piece_id, (
            f"self-query for {piece_id}: expected first, got {ranked[0].piece_id}"
        )


def test_landmark_rank_returns_all_pieces() -> None:
    catalog = _catalog()
    m = LandmarkMatcher(catalog)
    ranked = m.rank(catalog["piece_c"])
    assert len(ranked) == 3
    ids = {r.piece_id for r in ranked}
    assert ids == set(catalog.keys())


def test_landmark_scores_descending() -> None:
    catalog = _catalog()
    m = LandmarkMatcher(catalog)
    ranked = m.rank(catalog["piece_c"])
    scores = [r.score for r in ranked]
    assert scores == sorted(scores, reverse=True)
