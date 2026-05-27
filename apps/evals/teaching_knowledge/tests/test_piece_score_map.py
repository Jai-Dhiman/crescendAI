from __future__ import annotations

from pathlib import Path

import pytest

from teaching_knowledge.piece_score_map import PIECE_SCORE_MAP, get_score_path_for_piece


@pytest.mark.parametrize("slug", sorted(PIECE_SCORE_MAP.keys()))
def test_mapped_pieces_return_existing_path(slug: str) -> None:
    result = get_score_path_for_piece(slug)
    assert result is not None
    assert isinstance(result, Path)
    assert result.exists()
    assert result.suffix == ".json"


def test_unmapped_piece_returns_none() -> None:
    assert get_score_path_for_piece("clair_de_lune") is None
    assert get_score_path_for_piece("schumann_traumerei") is None
    assert get_score_path_for_piece("nonexistent_piece") is None
