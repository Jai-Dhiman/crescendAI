# model/src/follower_bench/asap_alignment.py
"""Resolves an ASAP piece identifier to its performance/score MIDI paths
and beat-level alignment, and validates the alignment is usable as the
exact ground-truth substrate for a synthetic clip.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_MODULE_DIR = Path(__file__).resolve()
DEFAULT_ASAP_ROOT = _MODULE_DIR.parents[2] / "data/raw/asap-dataset"
DEFAULT_ANNOTATIONS_PATH = DEFAULT_ASAP_ROOT / "asap_annotations.json"

MIN_BEATS = 4


class AsapAlignmentMissingError(Exception):
    """Raised when an ASAP piece has no usable beat alignment: not present
    in the annotations file, not marked score_and_performance_aligned, or
    has fewer than MIN_BEATS beat anchors (or mismatched
    performance_beats/midi_score_beats lengths)."""


@dataclass(frozen=True)
class ClipAlignment:
    """The ground-truth substrate for one ASAP performance."""
    asap_piece: str
    performance_midi_path: Path
    score_midi_path: Path
    performance_beats: tuple[float, ...]
    midi_score_beats: tuple[float, ...]
    midi_score_downbeats: tuple[float, ...] = ()


def load_alignment(
    asap_piece: str,
    asap_root: Path = DEFAULT_ASAP_ROOT,
    annotations_path: Path = DEFAULT_ANNOTATIONS_PATH,
) -> ClipAlignment:
    """Load and validate the ASAP beat alignment for asap_piece.

    Raises:
        FileNotFoundError: annotations_path does not exist, or the
            resolved performance/score MIDI files do not exist.
        AsapAlignmentMissingError: asap_piece is not a key in the
            annotations file, is not marked score_and_performance_aligned,
            or has fewer than MIN_BEATS beat anchors.
    """
    if not annotations_path.exists():
        raise FileNotFoundError(f"ASAP annotations file not found: {annotations_path}")
    data = json.loads(annotations_path.read_text())
    entry = data.get(asap_piece)
    if entry is None:
        raise AsapAlignmentMissingError(
            f"{asap_piece!r} not found in ASAP annotations: {annotations_path}"
        )
    if not entry.get("score_and_performance_aligned", False):
        raise AsapAlignmentMissingError(
            f"{asap_piece!r} is not marked score_and_performance_aligned"
        )
    perf_beats = entry.get("performance_beats") or []
    score_beats = entry.get("midi_score_beats") or []
    if len(perf_beats) < MIN_BEATS or len(perf_beats) != len(score_beats):
        raise AsapAlignmentMissingError(
            f"{asap_piece!r} has an unusable beat alignment: "
            f"{len(perf_beats)} performance_beats vs {len(score_beats)} midi_score_beats "
            f"(need >= {MIN_BEATS} matched pairs)"
        )
    score_downbeats = entry.get("midi_score_downbeats") or []
    performance_midi_path = asap_root / asap_piece
    if not performance_midi_path.exists():
        raise FileNotFoundError(f"ASAP performance MIDI not found: {performance_midi_path}")
    score_midi_path = performance_midi_path.parent / "midi_score.mid"
    if not score_midi_path.exists():
        raise FileNotFoundError(f"ASAP score MIDI not found: {score_midi_path}")
    return ClipAlignment(
        asap_piece=asap_piece,
        performance_midi_path=performance_midi_path,
        score_midi_path=score_midi_path,
        performance_beats=tuple(float(b) for b in perf_beats),
        midi_score_beats=tuple(float(b) for b in score_beats),
        midi_score_downbeats=tuple(float(b) for b in score_downbeats),
    )
