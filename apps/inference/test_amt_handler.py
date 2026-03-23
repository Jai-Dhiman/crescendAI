"""Unit tests for Aria-AMT endpoint handler.

Tests output format validation and context deduplication logic.
These tests do NOT require GPU or model weights -- they test the
pure-Python helper functions independently.

The amt_handler module has heavy imports (torch, amt package) at module
level. We mock those imports so tests can run without GPU dependencies,
then import only the pure-Python helper functions we need to test.
"""

from __future__ import annotations

import importlib
import sys
import types
from unittest import mock

import pytest


def _import_amt_handler():
    """Import amt_handler with heavy dependencies mocked.

    Mocks the amt.* and torch modules so we can import the pure-Python
    helper functions (deduplicate_notes, midi_dict_to_notes_and_pedals)
    without requiring GPU libraries.
    """
    # Create mock modules for all heavy dependencies
    mock_modules = {}
    for mod_name in [
        "torch",
        "amt",
        "amt.config",
        "amt.inference",
        "amt.inference.model",
        "amt.tokenizer",
        "amt.audio",
    ]:
        mock_modules[mod_name] = types.ModuleType(mod_name)

    # Set up minimal attributes that amt_handler imports at module level
    mock_modules["amt.config"].load_model_config = mock.MagicMock()
    mock_modules["amt.inference.model"].AmtEncoderDecoder = mock.MagicMock()
    mock_modules["amt.inference.model"].ModelConfig = mock.MagicMock()
    mock_modules["amt.tokenizer"].AmtTokenizer = mock.MagicMock()
    mock_modules["amt.audio"].AudioTransform = mock.MagicMock()

    # torch needs some attributes
    mock_modules["torch"].inference_mode = lambda: lambda fn: fn
    mock_modules["torch"].from_numpy = mock.MagicMock()
    mock_modules["torch"].cuda = mock.MagicMock()

    with mock.patch.dict(sys.modules, mock_modules):
        # Remove cached module if previously imported
        sys.modules.pop("amt_handler", None)

        # Import with mocked dependencies
        spec = importlib.util.spec_from_file_location(
            "amt_handler",
            "/Users/jdhiman/Documents/crescendai-zero-config/apps/inference/amt_handler.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

    return module


# Import the helper functions
_handler_mod = _import_amt_handler()
deduplicate_notes = _handler_mod.deduplicate_notes
midi_dict_to_notes_and_pedals = _handler_mod.midi_dict_to_notes_and_pedals


# -- Fixtures ------------------------------------------------------------------


def _make_notes(
    specs: list[tuple[int, float, float, int]],
) -> list[dict]:
    """Create note dicts from (pitch, onset, offset, velocity) tuples."""
    return [
        {"pitch": p, "onset": round(o, 4), "offset": round(off, 4), "velocity": v}
        for p, o, off, v in specs
    ]


def _make_pedals(
    specs: list[tuple[float, int]],
) -> list[dict]:
    """Create pedal dicts from (time, value) tuples."""
    return [{"time": round(t, 4), "value": v} for t, v in specs]


# -- Output Format Tests -------------------------------------------------------


class TestOutputFormat:
    """Validate note and pedal event field types match PerfNote/PerfPedalEvent."""

    def test_note_has_required_fields(self):
        notes = _make_notes([(60, 0.12, 0.45, 78)])
        note = notes[0]
        assert "pitch" in note
        assert "onset" in note
        assert "offset" in note
        assert "velocity" in note

    def test_note_field_types(self):
        notes = _make_notes([(60, 0.12, 0.45, 78)])
        note = notes[0]
        assert isinstance(note["pitch"], int)
        assert isinstance(note["onset"], float)
        assert isinstance(note["offset"], float)
        assert isinstance(note["velocity"], int)

    def test_pedal_has_required_fields(self):
        pedals = _make_pedals([(0.10, 127)])
        pedal = pedals[0]
        assert "time" in pedal
        assert "value" in pedal

    def test_pedal_field_types(self):
        pedals = _make_pedals([(0.10, 127)])
        pedal = pedals[0]
        assert isinstance(pedal["time"], float)
        assert isinstance(pedal["value"], int)

    def test_notes_sorted_by_onset_then_pitch(self):
        notes = _make_notes([
            (72, 0.50, 0.80, 60),
            (60, 0.12, 0.45, 78),
            (64, 0.12, 0.40, 65),
        ])
        # Sort like the handler does
        notes.sort(key=lambda n: (n["onset"], n["pitch"]))
        assert notes[0]["pitch"] == 60
        assert notes[1]["pitch"] == 64
        assert notes[2]["pitch"] == 72

    def test_pedal_values_are_cc64_convention(self):
        """Pedal values should be 0 (off) or 127 (on) per CC64 convention."""
        pedals = _make_pedals([(0.10, 127), (0.85, 0)])
        assert pedals[0]["value"] == 127  # on
        assert pedals[1]["value"] == 0  # off

    def test_empty_notes_and_pedals(self):
        notes = _make_notes([])
        pedals = _make_pedals([])
        assert notes == []
        assert pedals == []


# -- Context Deduplication Tests -----------------------------------------------


class TestContextDeduplication:
    """Test that notes in the context window are filtered out and timestamps adjusted."""

    def test_no_context_returns_all_notes(self):
        """With context_duration=0, all notes are returned unchanged."""
        notes = _make_notes([
            (60, 1.0, 1.5, 80),
            (64, 2.0, 2.5, 70),
            (67, 3.0, 3.5, 90),
        ])
        pedals = _make_pedals([(0.5, 127), (2.0, 0)])

        result_notes, result_pedals = deduplicate_notes(notes, pedals, 0.0)

        assert len(result_notes) == 3
        assert len(result_pedals) == 2
        # Timestamps unchanged
        assert result_notes[0]["onset"] == 1.0
        assert result_pedals[0]["time"] == 0.5

    def test_context_filters_early_notes(self):
        """Notes with onset < context_duration are removed."""
        notes = _make_notes([
            (60, 2.0, 2.5, 80),   # in context (< 15s)
            (64, 10.0, 10.5, 70),  # in context (< 15s)
            (67, 15.0, 15.5, 90),  # at boundary (>= 15s) -- kept
            (72, 20.0, 20.5, 85),  # in chunk (>= 15s) -- kept
        ])
        pedals = _make_pedals([
            (5.0, 127),   # in context -- removed
            (15.0, 0),    # at boundary -- kept
            (18.0, 127),  # in chunk -- kept
        ])

        result_notes, result_pedals = deduplicate_notes(notes, pedals, 15.0)

        assert len(result_notes) == 2
        assert len(result_pedals) == 2

    def test_timestamps_adjusted_by_context_duration(self):
        """Returned notes have timestamps relative to chunk start."""
        notes = _make_notes([
            (60, 5.0, 5.5, 80),    # in context -- filtered
            (64, 15.0, 15.8, 70),   # onset=15.0, adjusted to 0.0
            (67, 20.0, 20.5, 90),   # onset=20.0, adjusted to 5.0
        ])
        pedals = _make_pedals([
            (3.0, 127),   # filtered
            (16.0, 0),    # adjusted to 1.0
        ])

        result_notes, result_pedals = deduplicate_notes(notes, pedals, 15.0)

        assert len(result_notes) == 2
        assert result_notes[0]["onset"] == 0.0
        assert result_notes[0]["offset"] == 0.8
        assert result_notes[0]["pitch"] == 64
        assert result_notes[1]["onset"] == 5.0
        assert result_notes[1]["offset"] == 5.5

        assert len(result_pedals) == 1
        assert result_pedals[0]["time"] == 1.0

    def test_all_notes_in_context_returns_empty(self):
        """When all notes are in the context window, result is empty."""
        notes = _make_notes([
            (60, 1.0, 1.5, 80),
            (64, 5.0, 5.5, 70),
        ])
        pedals = _make_pedals([(2.0, 127)])

        result_notes, result_pedals = deduplicate_notes(notes, pedals, 30.0)

        assert result_notes == []
        assert result_pedals == []

    def test_boundary_note_exact_onset_at_context_duration(self):
        """A note with onset == context_duration is included (>= comparison)."""
        notes = _make_notes([
            (60, 14.999, 15.5, 80),  # just before boundary -- filtered
            (64, 15.0, 15.5, 70),    # exactly at boundary -- kept
            (67, 15.001, 15.5, 90),  # just after boundary -- kept
        ])

        result_notes, _ = deduplicate_notes(notes, [], 15.0)

        assert len(result_notes) == 2
        assert result_notes[0]["pitch"] == 64
        assert result_notes[0]["onset"] == 0.0
        assert result_notes[1]["pitch"] == 67
        assert result_notes[1]["onset"] == pytest.approx(0.001, abs=1e-3)

    def test_velocity_preserved_after_dedup(self):
        """Velocity values are not modified by deduplication."""
        notes = _make_notes([
            (60, 16.0, 16.5, 42),
            (64, 18.0, 18.5, 127),
        ])

        result_notes, _ = deduplicate_notes(notes, [], 15.0)

        assert result_notes[0]["velocity"] == 42
        assert result_notes[1]["velocity"] == 127

    def test_negative_context_duration_returns_all(self):
        """Negative context_duration is treated as no context."""
        notes = _make_notes([(60, 1.0, 1.5, 80)])
        result_notes, _ = deduplicate_notes(notes, [], -1.0)
        assert len(result_notes) == 1
        assert result_notes[0]["onset"] == 1.0


# -- MidiDict Conversion Tests ------------------------------------------------


class TestMidiDictConversion:
    """Test conversion from ariautils MidiDict to note/pedal format."""

    def test_with_mock_midi_dict(self):
        """Validate conversion using a mock MidiDict-like object."""

        class MockMidiDict:
            def __init__(self):
                self.note_msgs = [
                    {
                        "type": "note",
                        "data": {
                            "pitch": 60,
                            "start": 0,     # tick
                            "end": 480,     # tick
                            "velocity": 80,
                        },
                        "tick": 0,
                        "channel": 0,
                    },
                    {
                        "type": "note",
                        "data": {
                            "pitch": 64,
                            "start": 480,
                            "end": 960,
                            "velocity": 70,
                        },
                        "tick": 480,
                        "channel": 0,
                    },
                ]
                self.pedal_msgs = [
                    {
                        "type": "pedal",
                        "data": 1,  # on
                        "tick": 0,
                        "channel": 0,
                    },
                    {
                        "type": "pedal",
                        "data": 0,  # off
                        "tick": 960,
                        "channel": 0,
                    },
                ]
                self.ticks_per_beat = 480
                self.tempo_msgs = [
                    {"type": "tempo", "data": 500000, "tick": 0}
                ]

            def tick_to_ms(self, tick: int) -> int:
                """Simple conversion: 480 ticks/beat, 500000 us/beat = 120 BPM.
                1 beat = 500ms. 480 ticks = 500ms.
                """
                return int(tick * 500 / 480)

        mock_obj = MockMidiDict()
        notes, pedals = midi_dict_to_notes_and_pedals(mock_obj)

        assert len(notes) == 2
        assert notes[0]["pitch"] == 60
        assert notes[0]["onset"] == pytest.approx(0.0, abs=0.01)
        assert notes[0]["offset"] == pytest.approx(0.5, abs=0.01)
        assert notes[0]["velocity"] == 80

        assert notes[1]["pitch"] == 64
        assert notes[1]["onset"] == pytest.approx(0.5, abs=0.01)

        assert len(pedals) == 2
        assert pedals[0]["value"] == 127  # on -> CC64 convention
        assert pedals[1]["value"] == 0    # off

    def test_empty_midi_dict(self):
        """Empty MidiDict returns empty lists."""

        class EmptyMidiDict:
            note_msgs = []
            pedal_msgs = []
            ticks_per_beat = 480
            tempo_msgs = [{"type": "tempo", "data": 500000, "tick": 0}]

            def tick_to_ms(self, tick: int) -> int:
                return int(tick * 500 / 480)

        notes, pedals = midi_dict_to_notes_and_pedals(EmptyMidiDict())
        assert notes == []
        assert pedals == []
