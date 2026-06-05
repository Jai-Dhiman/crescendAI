# model/src/piece_id_eval/matchers/landmark.py
"""C2: Ordinal-landmark hash matcher.

Token: (pc_anchor, interval, ordinal_gap) where:
  - pc_anchor = anchor note pitch % 12 (absolute, key-dependent)
  - interval  = (target.pitch - anchor.pitch) clamped to [-12, 12]
  - ordinal_gap = target event index - anchor event index, in 1..MAX_GAP

The token is ordinal (event-index) not temporal: a student at half speed
produces the same tokens as at full speed. Inverted index maps token ->
list[piece_id]; rank by total hit count.

K=5 target notes per anchor; MAX_GAP=5 (per spec Open Questions defaults).
"""
from __future__ import annotations

from collections import defaultdict

from piece_id_eval.matchers.base import Ranked
from piece_id_eval.notes import Note

_K = 5          # number of target notes per anchor
_MAX_GAP = 5    # maximum ordinal gap between anchor and target

Token = tuple[int, int, int]  # (pc_anchor, interval, ordinal_gap)


def _build_tokens(notes: list[Note]) -> list[Token]:
    """Generate all (pc_anchor, interval, ordinal_gap) tokens from a note list."""
    tokens: list[Token] = []
    for i, anchor in enumerate(notes):
        pc_anchor = anchor.pitch % 12
        for gap in range(1, _MAX_GAP + 1):
            j = i + gap
            if j >= len(notes):
                break
            target = notes[j]
            interval = max(-12, min(12, target.pitch - anchor.pitch))
            tokens.append((pc_anchor, interval, gap))
    return tokens


def _build_index(catalog: dict[str, list[Note]]) -> dict[Token, list[str]]:
    """Build inverted index: token -> [piece_id, ...]."""
    index: dict[Token, list[str]] = defaultdict(list)
    for piece_id, notes in catalog.items():
        seen: set[Token] = set()
        for token in _build_tokens(notes):
            if token not in seen:
                index[token].append(piece_id)
                seen.add(token)
    return dict(index)


class LandmarkMatcher:
    """Inverted landmark-token hit-count matcher (C2)."""

    def __init__(self, catalog: dict[str, list[Note]]) -> None:
        self._piece_ids = list(catalog.keys())
        self._index = _build_index(catalog)

    @property
    def name(self) -> str:
        return "landmark"

    def rank(self, query: list[Note]) -> list[Ranked]:
        """Rank catalog pieces by landmark token hit count.

        Args:
            query: list of Note representing the query window.

        Returns:
            list of Ranked sorted descending by hit count (score).
        """
        hits: dict[str, int] = {pid: 0 for pid in self._piece_ids}
        for token in _build_tokens(query):
            for piece_id in self._index.get(token, []):
                if piece_id in hits:
                    hits[piece_id] += 1
        results = [Ranked(piece_id=pid, score=float(count)) for pid, count in hits.items()]
        results.sort(key=lambda r: r.score, reverse=True)
        return results
