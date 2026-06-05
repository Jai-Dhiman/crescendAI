# model/tests/piece_id_eval/test_dtw_ceiling.py
"""Verify DtwCeilingMatcher (C3 note-based) through its public interface."""
from __future__ import annotations

from piece_id_eval.matchers.base import Matcher, Ranked
from piece_id_eval.matchers.dtw_ceiling import DtwCeilingMatcher
from piece_id_eval.notes import Note


def _make_piece(pitches: list[int], onset_step: float = 0.5) -> list[Note]:
    return [
        Note(onset=i * onset_step, offset=i * onset_step + 0.3, pitch=p, velocity=80)
        for i, p in enumerate(pitches)
    ]


def _catalog() -> dict[str, list[Note]]:
    return {
        "ascending": _make_piece(list(range(60, 80))),
        "descending": _make_piece(list(range(79, 59, -1))),
        "constant": _make_piece([60] * 20),
    }


def test_dtw_ceiling_protocol_compliance() -> None:
    m = DtwCeilingMatcher(_catalog())
    assert isinstance(m, Matcher)
    assert isinstance(m.name, str)


def test_dtw_ceiling_self_query_recall_at_1() -> None:
    catalog = _catalog()
    m = DtwCeilingMatcher(catalog)
    for piece_id, notes in catalog.items():
        ranked = m.rank(notes)
        assert ranked[0].piece_id == piece_id, (
            f"self-query for {piece_id}: expected first, got {ranked[0].piece_id}"
        )


def test_dtw_ceiling_ranks_all_pieces() -> None:
    catalog = _catalog()
    m = DtwCeilingMatcher(catalog)
    ranked = m.rank(catalog["ascending"])
    assert len(ranked) == 3
    ids = {r.piece_id for r in ranked}
    assert ids == set(catalog.keys())


def test_dtw_ceiling_scores_descending() -> None:
    catalog = _catalog()
    m = DtwCeilingMatcher(catalog)
    ranked = m.rank(catalog["ascending"])
    scores = [r.score for r in ranked]
    assert scores == sorted(scores, reverse=True)


def test_dtw_ceiling_subsequence_query_ranks_correct_piece() -> None:
    """A short window (first 8 notes) of ascending should still match ascending."""
    catalog = _catalog()
    m = DtwCeilingMatcher(catalog)
    query = catalog["ascending"][:8]
    ranked = m.rank(query)
    assert ranked[0].piece_id == "ascending"
