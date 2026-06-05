# model/tests/piece_id_eval/test_windowing.py
"""Verify sample_windows through its public interface on synthetic notes."""
from __future__ import annotations

from piece_id_eval.notes import Note
from piece_id_eval.windowing import sample_windows


def _make_notes(n: int, spacing: float = 0.5) -> list[Note]:
    """Create n synthetic notes at regular spacing."""
    return [Note(onset=i * spacing, offset=i * spacing + 0.3, pitch=60, velocity=80) for i in range(n)]


def test_sample_windows_count_matches_n_starts() -> None:
    notes = _make_notes(100)  # 50 seconds at 0.5s spacing
    windows = sample_windows(notes, window_seconds=10.0, n_starts=5, seed=42)
    assert len(windows) == 5


def test_sample_windows_notes_within_window() -> None:
    notes = _make_notes(100)
    windows = sample_windows(notes, window_seconds=10.0, n_starts=3, seed=0)
    for win in windows:
        if len(win) == 0:
            continue
        duration = win[-1].onset - win[0].onset
        assert duration <= 10.0 + 1e-6, f"window spans {duration:.3f}s > 10s"


def test_sample_windows_full_returns_single_window() -> None:
    notes = _make_notes(20)
    windows = sample_windows(notes, window_seconds=None, n_starts=5, seed=0)
    assert len(windows) == 1
    assert windows[0] == notes


def test_sample_windows_deterministic() -> None:
    notes = _make_notes(100)
    a = sample_windows(notes, window_seconds=10.0, n_starts=5, seed=7)
    b = sample_windows(notes, window_seconds=10.0, n_starts=5, seed=7)
    assert a == b


def test_sample_windows_different_seeds_differ() -> None:
    notes = _make_notes(100)
    a = sample_windows(notes, window_seconds=10.0, n_starts=5, seed=1)
    b = sample_windows(notes, window_seconds=10.0, n_starts=5, seed=2)
    # Different seeds should (almost certainly) produce different start offsets
    starts_a = [w[0].onset if w else None for w in a]
    starts_b = [w[0].onset if w else None for w in b]
    assert starts_a != starts_b


def test_sample_windows_short_recording_returns_full() -> None:
    """If recording is shorter than window, return full recording as single window."""
    notes = _make_notes(4)  # 2 seconds total
    windows = sample_windows(notes, window_seconds=30.0, n_starts=5, seed=0)
    assert len(windows) == 1
    assert windows[0] == notes
