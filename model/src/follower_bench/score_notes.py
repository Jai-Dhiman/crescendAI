"""Score-note representation and loaders for the baseline monotonic
follower (issue #115). Two source formats collapse into one ScoreNote
shape: the WASM score-analysis crate's bar/tick-based fixture JSON (read
in place, not duplicated -- see docs/specs/2026-07-12-baseline-monotonic-
follower-design.md), and a raw score MIDI file via partitura.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from follower_bench.segments import PerfNote


@dataclass(frozen=True)
class ScoreNote:
    """One score note event: MIDI pitch and its position in the score's
    own timeline. `position` is an opaque monotonic label -- seconds for
    a fixed-tempo score render, beats for a partitura-loaded score MIDI --
    follow() never interprets its unit, only compares/reports it."""
    pitch: int
    position: float


def load_golden_fixture_notes(json_path: Path) -> tuple[list[PerfNote], list[ScoreNote]]:
    """Load the WASM score-analysis crate's bach_inv1_chunk0.json fixture,
    returning (perf_notes, score_notes) in follower_bench's own types.
    `perf_notes` entries already match PerfNote's fields exactly (pitch,
    onset, offset, velocity in seconds). `score_notes` are flattened from
    the fixture's per-bar `notes` lists, in bar order, position =
    onset_seconds.

    Raises:
        FileNotFoundError: json_path does not exist.
    """
    if not json_path.exists():
        raise FileNotFoundError(f"golden fixture not found: {json_path}")
    data = json.loads(json_path.read_text())

    perf_notes = [
        PerfNote(
            onset=float(n["onset"]),
            offset=float(n["offset"]),
            pitch=int(n["pitch"]),
            velocity=int(n["velocity"]),
        )
        for n in data["perf_notes"]
    ]

    score_notes = [
        ScoreNote(pitch=int(n["pitch"]), position=float(n["onset_seconds"]))
        for bar in data["score_bars"]
        for n in bar["notes"]
    ]

    return perf_notes, score_notes
