# model/tests/piece_id_eval/test_notes.py
"""Verify Note loaders through their public interface on committed fixtures."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from piece_id_eval.notes import Note, load_amt_notes, load_score_notes

REPO_ROOT = Path(__file__).resolve().parents[3]
BACH_SCORE = REPO_ROOT / "model/data/scores/bach.prelude.bwv_846.json"


def _write_amt_fixture(tmp_path: Path) -> Path:
    """Write a minimal flat AMT notes JSON with 3 notes out of onset order."""
    notes = [
        {"onset": 1.5, "offset": 2.0, "pitch": 62, "velocity": 70},
        {"onset": 0.0, "offset": 0.5, "pitch": 60, "velocity": 80},
        {"onset": 3.0, "offset": 3.5, "pitch": 64, "velocity": 90},
    ]
    p = tmp_path / "amt_notes.json"
    p.write_text(json.dumps(notes))
    return p


def test_load_amt_notes_returns_notes_sorted_by_onset(tmp_path: Path) -> None:
    p = _write_amt_fixture(tmp_path)
    notes = load_amt_notes(p)
    assert len(notes) == 3
    onsets = [n.onset for n in notes]
    assert onsets == sorted(onsets), f"not sorted: {onsets}"


def test_load_amt_notes_fields_match_fixture(tmp_path: Path) -> None:
    p = _write_amt_fixture(tmp_path)
    notes = load_amt_notes(p)
    first = notes[0]  # onset=0.0 after sorting
    assert isinstance(first, Note)
    assert first.onset == pytest.approx(0.0)
    assert first.offset == pytest.approx(0.5)
    assert first.pitch == 60
    assert first.velocity == 80


def test_load_score_notes_returns_notes_sorted_by_onset() -> None:
    if not BACH_SCORE.exists():
        pytest.skip("bach score fixture not present")
    notes = load_score_notes(BACH_SCORE)
    assert len(notes) > 0
    onsets = [n.onset for n in notes]
    assert onsets == sorted(onsets), f"not sorted: {onsets[:5]}"


def test_load_score_notes_fields_plausible() -> None:
    if not BACH_SCORE.exists():
        pytest.skip("bach score fixture not present")
    notes = load_score_notes(BACH_SCORE)
    first = notes[0]
    assert isinstance(first, Note)
    assert first.onset >= 0.0
    assert first.offset > first.onset
    assert 21 <= first.pitch <= 108  # piano range
    assert 0 <= first.velocity <= 127


def test_load_amt_notes_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_amt_notes(tmp_path / "nonexistent.json")


def test_note_namedtuple_fields() -> None:
    n = Note(onset=0.0, offset=0.5, pitch=60, velocity=80)
    assert n.onset == 0.0
    assert n.offset == 0.5
    assert n.pitch == 60
    assert n.velocity == 80
