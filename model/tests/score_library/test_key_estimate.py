"""Tests for the Krumhansl key estimation helper."""

from __future__ import annotations

from score_library.key_estimate import estimate_key
from score_library.schema import Bar, ScoreData, ScoreNote


def _note(pitch: int, onset_tick: int = 0) -> ScoreNote:
    return ScoreNote(
        pitch=pitch,
        pitch_name="X",
        velocity=80,
        onset_tick=onset_tick,
        onset_seconds=float(onset_tick) / 480,
        duration_ticks=240,
        duration_seconds=0.25,
        track=0,
    )


def _bar(bar_number: int, notes: list[ScoreNote], start_tick: int = 0) -> Bar:
    pitches = [n.pitch for n in notes]
    return Bar(
        bar_number=bar_number,
        start_tick=start_tick,
        start_seconds=0.0,
        time_signature="4/4",
        notes=notes,
        pedal_events=[],
        note_count=len(notes),
        pitch_range=[min(pitches), max(pitches)] if pitches else [],
        mean_velocity=80 if notes else 0,
    )


def _make_score(bars: list[Bar]) -> ScoreData:
    return ScoreData(
        piece_id="test.key",
        composer="Test",
        title="Test",
        key_signature=None,
        time_signatures=[{"tick": 0, "numerator": 4, "denominator": 4}],
        tempo_markings=[],
        total_bars=len(bars),
        bars=bars,
    )


def _c_major_pitches() -> list[int]:
    """MIDI pitches for a dense C-major scale (C4-B4) repeated many times."""
    scale_pcs = [0, 2, 4, 5, 7, 9, 11]  # C D E F G A B pitch classes
    return [(4 * 12 + pc) for pc in scale_pcs] * 10  # 70 notes, heavily C-major


def _g_major_pitches() -> list[int]:
    """Dense G-major scale pitches (G4 scale)."""
    scale_pcs = [7, 9, 11, 0, 2, 4, 6]  # G A B C D E F# pitch classes
    return [(4 * 12 + pc) for pc in scale_pcs] * 10


def _a_minor_pitches() -> list[int]:
    """Dense A-natural-minor scale pitches."""
    scale_pcs = [9, 11, 0, 2, 4, 5, 7]  # A B C D E F G
    return [(4 * 12 + pc) for pc in scale_pcs] * 10


def test_c_major_score_estimates_c_major():
    pitches = _c_major_pitches()
    notes = [_note(p, i * 240) for i, p in enumerate(pitches)]
    bar = _bar(1, notes)
    score = _make_score([bar])
    key = estimate_key(score)
    assert key == "C major", f"Expected 'C major', got {key!r}"


def test_g_major_score_estimates_g_major():
    pitches = _g_major_pitches()
    notes = [_note(p, i * 240) for i, p in enumerate(pitches)]
    bar = _bar(1, notes)
    score = _make_score([bar])
    key = estimate_key(score)
    assert key == "G major", f"Expected 'G major', got {key!r}"


def test_a_minor_score_estimates_a_minor_or_relative_major():
    """A natural-minor scale shares all pitch classes with C major.

    The Krumhansl profiles weight tonic heavily, so for a uniform-weight
    scale passage the relative major (C major) can outscore A minor.
    The invariant is that the result is one of the two relative keys.
    """
    pitches = _a_minor_pitches()
    notes = [_note(p, i * 240) for i, p in enumerate(pitches)]
    bar = _bar(1, notes)
    score = _make_score([bar])
    key = estimate_key(score)
    assert key in ("A minor", "C major"), (
        f"Expected 'A minor' or its relative major 'C major', got {key!r}"
    )


def test_empty_score_returns_default():
    """An empty score (no notes) should not raise and returns a valid key string."""
    score = _make_score([_bar(1, [])])
    key = estimate_key(score)
    assert " major" in key or " minor" in key


def test_estimate_key_format():
    """Return value always has the form 'X major' or 'X minor'."""
    pitches = _c_major_pitches()
    notes = [_note(p, i * 240) for i, p in enumerate(pitches)]
    bar = _bar(1, notes)
    score = _make_score([bar])
    key = estimate_key(score)
    parts = key.split()
    assert len(parts) == 2
    assert parts[1] in ("major", "minor")
