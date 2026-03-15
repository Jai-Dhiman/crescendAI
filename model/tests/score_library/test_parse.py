"""Tests for the MIDI parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from score_library.parse import (
    assign_notes_to_bars,
    build_bar_grid,
    parse_score_midi,
    ticks_to_seconds,
)
from score_library.schema import ScoreData

ASAP_CHOPIN_ETUDE = Path(
    "/Users/jdhiman/Documents/crescendai/model/data/asap_cache"
    "/Chopin/Etudes_op_10/3/score_SunMeiting08.mid"
)


# ---------------------------------------------------------------------------
# Bar grid tests
# ---------------------------------------------------------------------------


class TestBuildBarGrid:
    def test_build_bar_grid_simple_4_4(self) -> None:
        """4/4 at 480 tpb -> 1920 ticks per bar."""
        time_sigs = [{"tick": 0, "numerator": 4, "denominator": 4}]
        total_ticks = 1920 * 4  # 4 bars
        bars = build_bar_grid(time_sigs, total_ticks, ticks_per_beat=480)

        assert len(bars) == 4
        assert bars[0]["bar_number"] == 1
        assert bars[0]["start_tick"] == 0
        assert bars[0]["time_signature"] == "4/4"
        assert bars[1]["start_tick"] == 1920
        assert bars[3]["start_tick"] == 1920 * 3

    def test_build_bar_grid_3_4(self) -> None:
        """3/4 at 480 tpb -> 1440 ticks per bar."""
        time_sigs = [{"tick": 0, "numerator": 3, "denominator": 4}]
        total_ticks = 1440 * 3
        bars = build_bar_grid(time_sigs, total_ticks, ticks_per_beat=480)

        assert len(bars) == 3
        assert bars[0]["time_signature"] == "3/4"
        assert bars[1]["start_tick"] == 1440
        assert bars[2]["start_tick"] == 2880

    def test_build_bar_grid_time_sig_change(self) -> None:
        """4/4 for 2 bars then 3/4 for 2 bars."""
        time_sigs = [
            {"tick": 0, "numerator": 4, "denominator": 4},
            {"tick": 3840, "numerator": 3, "denominator": 4},  # after 2 bars of 4/4
        ]
        total_ticks = 3840 + 1440 * 2  # 2 bars of 4/4 + 2 bars of 3/4
        bars = build_bar_grid(time_sigs, total_ticks, ticks_per_beat=480)

        assert len(bars) == 4
        assert bars[0]["time_signature"] == "4/4"
        assert bars[1]["time_signature"] == "4/4"
        assert bars[2]["time_signature"] == "3/4"
        assert bars[2]["start_tick"] == 3840
        assert bars[3]["time_signature"] == "3/4"
        assert bars[3]["start_tick"] == 3840 + 1440

    def test_build_bar_grid_6_8(self) -> None:
        """6/8 at 480 tpb -> 6 * 480 * 4 / 8 = 1440 ticks per bar."""
        time_sigs = [{"tick": 0, "numerator": 6, "denominator": 8}]
        total_ticks = 1440 * 2
        bars = build_bar_grid(time_sigs, total_ticks, ticks_per_beat=480)

        assert len(bars) == 2
        assert bars[0]["time_signature"] == "6/8"
        assert bars[1]["start_tick"] == 1440

    def test_build_bar_grid_default_4_4(self) -> None:
        """Empty time_sigs defaults to 4/4."""
        bars = build_bar_grid([], total_ticks=1920, ticks_per_beat=480)
        assert len(bars) == 1
        assert bars[0]["time_signature"] == "4/4"


# ---------------------------------------------------------------------------
# Tick-to-seconds tests
# ---------------------------------------------------------------------------


class TestTicksToSeconds:
    def test_ticks_to_seconds_constant_tempo(self) -> None:
        """120 BPM (500000 usec/beat) at 480 tpb."""
        tempo_map = [{"tick": 0, "tempo": 500_000}]
        tpb = 480

        assert ticks_to_seconds(0, tempo_map, tpb) == pytest.approx(0.0)
        # 480 ticks = 1 beat = 0.5s at 120 BPM
        assert ticks_to_seconds(480, tempo_map, tpb) == pytest.approx(0.5)
        # 960 ticks = 2 beats = 1.0s
        assert ticks_to_seconds(960, tempo_map, tpb) == pytest.approx(1.0)

    def test_ticks_to_seconds_tempo_change(self) -> None:
        """120 BPM for first 960 ticks, then 60 BPM."""
        tempo_map = [
            {"tick": 0, "tempo": 500_000},  # 120 BPM
            {"tick": 960, "tempo": 1_000_000},  # 60 BPM
        ]
        tpb = 480

        # At tick 960: 2 beats at 120 BPM = 1.0s
        assert ticks_to_seconds(960, tempo_map, tpb) == pytest.approx(1.0)
        # At tick 1440: 1.0s + 480 ticks at 60 BPM (1s/beat) = 1.0 + 1.0 = 2.0s
        assert ticks_to_seconds(1440, tempo_map, tpb) == pytest.approx(2.0)

    def test_ticks_to_seconds_default_tempo(self) -> None:
        """Empty tempo map defaults to 120 BPM."""
        assert ticks_to_seconds(480, [], 480) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Note assignment tests
# ---------------------------------------------------------------------------


class TestAssignNotesToBars:
    def test_assign_notes_to_bars(self) -> None:
        """3 notes across 2 bars (1920 ticks each at 480 tpb, 4/4)."""
        bar_grid = [
            {"bar_number": 1, "start_tick": 0, "time_signature": "4/4"},
            {"bar_number": 2, "start_tick": 1920, "time_signature": "4/4"},
        ]
        notes = [
            {"onset_tick": 0, "pitch": 60},
            {"onset_tick": 960, "pitch": 62},
            {"onset_tick": 2400, "pitch": 64},
        ]

        result = assign_notes_to_bars(notes, bar_grid)

        assert len(result[1]) == 2
        assert len(result[2]) == 1
        assert result[2][0]["pitch"] == 64

    def test_assign_notes_before_first_bar(self) -> None:
        """Notes before bar 1 get assigned to bar 1."""
        bar_grid = [
            {"bar_number": 1, "start_tick": 100, "time_signature": "4/4"},
        ]
        notes = [{"onset_tick": 50, "pitch": 60}]

        result = assign_notes_to_bars(notes, bar_grid)
        assert len(result[1]) == 1

    def test_assign_notes_empty_bars(self) -> None:
        """Bars with no notes still appear in the result."""
        bar_grid = [
            {"bar_number": 1, "start_tick": 0, "time_signature": "4/4"},
            {"bar_number": 2, "start_tick": 1920, "time_signature": "4/4"},
        ]

        result = assign_notes_to_bars([], bar_grid)
        assert result[1] == []
        assert result[2] == []


# ---------------------------------------------------------------------------
# Real MIDI test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not ASAP_CHOPIN_ETUDE.exists(),
    reason="ASAP dataset not available",
)
class TestParseChopinEtude:
    def test_parse_chopin_etude(self) -> None:
        """Parse Chopin Etude Op. 10 No. 3, validate schema and bar structure."""
        result = parse_score_midi(
            midi_path=ASAP_CHOPIN_ETUDE,
            piece_id="Chopin/Etudes_op_10/3",
            composer="Chopin",
            title="Etude Op. 10 No. 3",
        )

        # Validate it's a proper ScoreData.
        assert isinstance(result, ScoreData)
        assert result.piece_id == "Chopin/Etudes_op_10/3"
        assert result.composer == "Chopin"

        # Should have a reasonable number of bars.
        assert result.total_bars > 10
        assert result.total_bars == len(result.bars)

        # First bar should start at tick 0 and time 0.
        first_bar = result.bars[0]
        assert first_bar.bar_number == 1
        assert first_bar.start_tick == 0
        assert first_bar.start_seconds == pytest.approx(0.0)

        # Should have notes.
        total_notes = sum(bar.note_count for bar in result.bars)
        assert total_notes > 100

        # Bar numbers should be sequential.
        bar_numbers = [b.bar_number for b in result.bars]
        assert bar_numbers == list(range(1, result.total_bars + 1))

        # Every note should have a valid pitch name.
        for bar in result.bars:
            for note in bar.notes:
                assert 0 <= note.pitch <= 127
                assert len(note.pitch_name) >= 2  # e.g. "C4"

        # Time signatures and tempo markings should be present.
        assert len(result.time_signatures) >= 1
        assert len(result.tempo_markings) >= 1

        # Bar start_seconds should be non-decreasing.
        for i in range(1, len(result.bars)):
            assert result.bars[i].start_seconds >= result.bars[i - 1].start_seconds
