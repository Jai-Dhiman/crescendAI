# model/tests/piece_id_eval/test_corruption.py
"""Verify corrupt_notes statistics at known rates/seeds."""
from __future__ import annotations

from piece_id_eval.corruption import corrupt_notes
from piece_id_eval.notes import Note


def _make_notes(n: int = 100) -> list[Note]:
    return [Note(onset=i * 0.5, offset=i * 0.5 + 0.3, pitch=60 + (i % 24), velocity=80) for i in range(n)]


def test_deletion_rate_removes_roughly_half() -> None:
    notes = _make_notes(200)
    corrupted = corrupt_notes(notes, deletion_rate=0.5, insertion_rate=0.0, jitter_seconds=0.0, seed=0)
    ratio = len(corrupted) / len(notes)
    assert 0.35 < ratio < 0.65, f"expected ~50% remaining, got {ratio:.2f}"


def test_insertion_rate_adds_notes() -> None:
    notes = _make_notes(100)
    corrupted = corrupt_notes(notes, deletion_rate=0.0, insertion_rate=0.5, jitter_seconds=0.0, seed=0)
    assert len(corrupted) > len(notes), "insertion_rate=0.5 should add notes"


def test_jitter_shifts_onsets() -> None:
    notes = _make_notes(50)
    corrupted = corrupt_notes(notes, deletion_rate=0.0, insertion_rate=0.0, jitter_seconds=0.05, seed=0)
    assert len(corrupted) == len(notes)
    original_onsets = [n.onset for n in notes]
    corrupted_onsets = [n.onset for n in corrupted]
    # At least some onsets should have changed
    changed = sum(abs(a - b) > 1e-9 for a, b in zip(original_onsets, corrupted_onsets))
    assert changed > 0, "jitter_seconds>0 should shift some onsets"


def test_no_corruption_is_identity() -> None:
    notes = _make_notes(20)
    corrupted = corrupt_notes(notes, deletion_rate=0.0, insertion_rate=0.0, jitter_seconds=0.0, seed=0)
    assert corrupted == notes


def test_deterministic_for_same_seed() -> None:
    notes = _make_notes(100)
    a = corrupt_notes(notes, deletion_rate=0.3, insertion_rate=0.2, jitter_seconds=0.05, seed=42)
    b = corrupt_notes(notes, deletion_rate=0.3, insertion_rate=0.2, jitter_seconds=0.05, seed=42)
    assert a == b


def test_corrupted_notes_sorted_by_onset() -> None:
    notes = _make_notes(50)
    corrupted = corrupt_notes(notes, deletion_rate=0.2, insertion_rate=0.3, jitter_seconds=0.05, seed=1)
    onsets = [n.onset for n in corrupted]
    assert onsets == sorted(onsets), "output must be sorted by onset"
