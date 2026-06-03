"""Tests for the chroma-independent validation gate."""

from __future__ import annotations

from score_library.schema import Bar, ScoreData, ScoreNote
from score_library.validate import ExpectedMeta, Violation, validate_score


def _note(pitch: int, onset_seconds: float, onset_tick: int) -> ScoreNote:
    return ScoreNote(
        pitch=pitch,
        pitch_name="X",
        velocity=80,
        onset_tick=onset_tick,
        onset_seconds=onset_seconds,
        duration_ticks=240,
        duration_seconds=0.25,
        track=0,
    )


def _bar(
    bar_number: int,
    start_tick: int,
    notes: list[ScoreNote],
    time_signature: str = "4/4",
) -> Bar:
    pitches = [n.pitch for n in notes]
    return Bar(
        bar_number=bar_number,
        start_tick=start_tick,
        start_seconds=float(bar_number - 1),
        time_signature=time_signature,
        notes=notes,
        pedal_events=[],
        note_count=len(notes),
        pitch_range=[min(pitches), max(pitches)] if pitches else [],
        mean_velocity=80 if notes else 0,
    )


def _make_score(bars: list[Bar], piece_id: str = "test.piece") -> ScoreData:
    return ScoreData(
        piece_id=piece_id,
        composer="Test",
        title="Test",
        key_signature=None,
        time_signatures=[{"tick": 0, "numerator": 4, "denominator": 4}],
        tempo_markings=[],
        total_bars=len(bars),
        bars=bars,
    )


def _c_major_clean_bars(n_bars: int = 3, ppb: int = 480) -> list[Bar]:
    """n_bars of clean 4/4 16th-grid C-major scale notes, monotonic, >= 20 notes.

    16th = ppb/4 ticks. Bar length = ppb*4 ticks. 8 sixteenths per bar (every
    other 16th) keeps onsets exactly on grid.
    """
    bar_ticks = ppb * 4
    sixteenth = ppb // 4
    c_major = [60, 62, 64, 65, 67, 69, 71, 72]
    bars: list[Bar] = []
    for b in range(n_bars):
        notes = []
        for i, pitch in enumerate(c_major):
            tick = b * bar_ticks + i * (2 * sixteenth)
            notes.append(_note(pitch, tick / 1000.0, tick))
        bars.append(_bar(b + 1, b * bar_ticks, notes))
    return bars


def _expected(**overrides) -> ExpectedMeta:
    base = dict(piece_id="test.piece", expected_key="C major", expected_bars=3)
    base.update(overrides)
    return ExpectedMeta(**base)


class TestDoDMinimums:
    def test_clean_score_has_no_min_violations(self) -> None:
        score = _make_score(_c_major_clean_bars())
        violations = validate_score(score, _expected())
        assert not any(v.check == "min_notes" for v in violations)
        assert not any(v.check == "total_bars" for v in violations)
        assert not any(v.check == "monotonic_onsets" for v in violations)

    def test_too_few_notes_flagged(self) -> None:
        bars = [_bar(1, 0, [_note(60, 0.0, 0), _note(62, 0.25, 240)])]
        score = _make_score(bars)
        violations = validate_score(score, _expected(expected_bars=1))
        assert any(v.check == "min_notes" for v in violations)

    def test_zero_bars_flagged(self) -> None:
        score = _make_score([])
        violations = validate_score(score, _expected())
        assert any(v.check == "total_bars" for v in violations)

    def test_non_monotonic_onsets_flagged(self) -> None:
        bars = _c_major_clean_bars()
        # Corrupt: make the last note of bar 1 precede an earlier note.
        bars[0].notes[-1] = _note(72, -1.0, -1000)
        score = _make_score(bars)
        violations = validate_score(score, _expected())
        assert any(v.check == "monotonic_onsets" for v in violations)

    def test_violation_is_frozen_dataclass(self) -> None:
        v = Violation(check="min_notes", detail="x")
        assert v.check == "min_notes"
        assert v.detail == "x"
