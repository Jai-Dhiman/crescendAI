# model/tests/piece_id_eval/test_chroma_seq_dtw.py
"""Verify ChromaSeqDtwMatcher (C4) through its public interface."""
from __future__ import annotations

from piece_id_eval.matchers.base import Matcher, Ranked
from piece_id_eval.matchers.chroma_seq_dtw import ChromaSeqDtwMatcher
from piece_id_eval.notes import Note


def _make_piece(pitch: int, n: int = 40) -> list[Note]:
    """All notes on a single pitch; distinct pitch -> distinct chroma."""
    return [Note(onset=i * 0.25, offset=i * 0.25 + 0.2, pitch=pitch, velocity=80) for i in range(n)]


def _catalog() -> dict[str, list[Note]]:
    return {
        "piece_c": _make_piece(60),   # pc=0
        "piece_e": _make_piece(64),   # pc=4
        "piece_g": _make_piece(67),   # pc=7
    }


def test_chroma_seq_dtw_protocol_compliance() -> None:
    m = ChromaSeqDtwMatcher(_catalog())
    assert isinstance(m, Matcher)
    assert isinstance(m.name, str)


def test_chroma_seq_dtw_self_query_recall_at_1() -> None:
    catalog = _catalog()
    m = ChromaSeqDtwMatcher(catalog)
    for piece_id, notes in catalog.items():
        ranked = m.rank(notes)
        assert ranked[0].piece_id == piece_id, (
            f"self-query for {piece_id}: expected first, got {ranked[0].piece_id}"
        )


def test_chroma_seq_dtw_ranks_all_pieces() -> None:
    catalog = _catalog()
    m = ChromaSeqDtwMatcher(catalog)
    ranked = m.rank(catalog["piece_c"])
    assert len(ranked) == 3
    ids = {r.piece_id for r in ranked}
    assert ids == set(catalog.keys())


def test_chroma_seq_dtw_scores_descending() -> None:
    catalog = _catalog()
    m = ChromaSeqDtwMatcher(catalog)
    ranked = m.rank(catalog["piece_c"])
    scores = [r.score for r in ranked]
    assert scores == sorted(scores, reverse=True)


def test_chroma_seq_dtw_subsequence_query() -> None:
    """A 10-note window from piece_c should still match piece_c first."""
    catalog = _catalog()
    m = ChromaSeqDtwMatcher(catalog)
    query = catalog["piece_c"][:10]
    ranked = m.rank(query)
    assert ranked[0].piece_id == "piece_c"
