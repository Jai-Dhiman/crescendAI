# model/src/piece_id_eval/notes.py
"""Symbolic note substrate and loaders.

Both AMT output (flat list of {onset,offset,pitch,velocity}) and score JSON
(bars[].notes[].{onset_seconds,duration_seconds,pitch,velocity}) are
normalised to the same Note NamedTuple, sorted ascending by onset.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple


class Note(NamedTuple):
    """A single symbolic note in seconds. Onset and offset are absolute seconds."""
    onset: float
    offset: float
    pitch: int
    velocity: int


def load_amt_notes(path: Path) -> list[Note]:
    """Load AMT notes from a flat JSON list of {onset,offset,pitch,velocity} dicts.

    Returns notes sorted ascending by onset.

    Raises:
        FileNotFoundError: if path does not exist.
        ValueError: if the JSON is not a list.
    """
    if not path.exists():
        raise FileNotFoundError(f"AMT notes file not found: {path}")
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON list, got {type(raw).__name__}: {path}")
    notes = [
        Note(
            onset=float(d["onset"]),
            offset=float(d["offset"]),
            pitch=int(d["pitch"]),
            velocity=int(d.get("velocity", 80)),
        )
        for d in raw
        if isinstance(d, dict) and "onset" in d and "pitch" in d
    ]
    notes.sort(key=lambda n: n.onset)
    return notes


def load_score_notes(path: Path) -> list[Note]:
    """Load score notes from a score JSON with bars[].notes[] structure.

    Each note has onset_seconds, duration_seconds, pitch, velocity.
    Returns notes sorted ascending by onset.

    Returns an empty list if the JSON has no 'bars' key.

    Raises:
        FileNotFoundError: if path does not exist.
        KeyError: if a note dict within bars lacks 'onset_seconds' or 'pitch'.
    """
    if not path.exists():
        raise FileNotFoundError(f"Score JSON not found: {path}")
    body = json.loads(path.read_text())
    notes: list[Note] = []
    for bar in body.get("bars") or []:
        for n in bar.get("notes") or []:
            onset = float(n["onset_seconds"])
            duration = float(n.get("duration_seconds", 0.25))
            notes.append(Note(
                onset=onset,
                offset=onset + duration,
                pitch=int(n["pitch"]),
                velocity=int(n.get("velocity", 80)),
            ))
    notes.sort(key=lambda n: n.onset)
    return notes
