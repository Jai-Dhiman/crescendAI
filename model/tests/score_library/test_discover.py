"""Tests for ASAP score MIDI discovery and title generation."""

from __future__ import annotations

from pathlib import Path

from score_library.discover import PieceEntry, derive_piece_id, discover_pieces
from score_library.titles import clean_title_from_path


# -- derive_piece_id ----------------------------------------------------------


def test_derive_piece_id_3_level(tmp_path: Path) -> None:
    piece_dir = tmp_path / "Chopin" / "Etudes_op_10" / "3"
    piece_dir.mkdir(parents=True)
    assert derive_piece_id(piece_dir, tmp_path) == "chopin.etudes_op_10.3"


def test_derive_piece_id_2_level(tmp_path: Path) -> None:
    piece_dir = tmp_path / "Balakirev" / "Islamey"
    piece_dir.mkdir(parents=True)
    assert derive_piece_id(piece_dir, tmp_path) == "balakirev.islamey"


# -- discover_pieces ----------------------------------------------------------


def _touch(path: Path) -> Path:
    """Create a file and its parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def test_discover_pieces_finds_both_depths(tmp_path: Path) -> None:
    # 3-level: Chopin/Etudes_op_10/3/score_chopin.mid
    _touch(tmp_path / "Chopin" / "Etudes_op_10" / "3" / "score_chopin.mid")
    # 2-level: Balakirev/Islamey/score_islamey.mid
    _touch(tmp_path / "Balakirev" / "Islamey" / "score_islamey.mid")

    entries = discover_pieces(tmp_path)

    assert len(entries) == 2
    ids = {e.piece_id for e in entries}
    assert "chopin.etudes_op_10.3" in ids
    assert "balakirev.islamey" in ids

    # Check composers
    by_id = {e.piece_id: e for e in entries}
    assert by_id["chopin.etudes_op_10.3"].composer == "Chopin"
    assert by_id["balakirev.islamey"].composer == "Balakirev"

    # Check that entries are PieceEntry instances
    for entry in entries:
        assert isinstance(entry, PieceEntry)
        assert entry.score_midi_path.exists()


def test_discover_pieces_skips_dirs_without_scores(tmp_path: Path) -> None:
    # Only performance MIDIs, no score_*.mid
    _touch(tmp_path / "Chopin" / "Etudes_op_10" / "3" / "performance_01.mid")
    _touch(tmp_path / "Chopin" / "Etudes_op_10" / "3" / "performance_02.mid")

    entries = discover_pieces(tmp_path)
    assert entries == []


def test_discover_pieces_empty_dir(tmp_path: Path) -> None:
    entries = discover_pieces(tmp_path)
    assert entries == []


# -- clean_title_from_path ----------------------------------------------------


def test_clean_title_from_path_etude(tmp_path: Path) -> None:
    piece_dir = tmp_path / "Chopin" / "Etudes_op_10" / "3"
    piece_dir.mkdir(parents=True)
    title = clean_title_from_path(piece_dir, tmp_path)
    assert title == "Etudes Op. 10 No. 3"


def test_clean_title_from_path_2_level(tmp_path: Path) -> None:
    piece_dir = tmp_path / "Balakirev" / "Islamey"
    piece_dir.mkdir(parents=True)
    title = clean_title_from_path(piece_dir, tmp_path)
    assert title == "Islamey"
