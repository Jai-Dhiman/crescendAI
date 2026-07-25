"""Shared Transkun transcription helper.

Deliberately does NOT import `transkun`: it shells out to an isolated env
(`uv run --no-project --with transkun --python 3.11 transkun IN OUT --device cpu`),
so this module is import-safe from BOTH the service env and model/.venv (whose
torch deps conflict with Transkun). Parses the output MIDI with pretty_midi.

Returns the exact dict shapes both surfaces already expect:
  notes:  {"pitch": int, "onset": float, "offset": float, "velocity": int}
  pedals: {"time": float, "value": int}   (CC64 >= 64 -> value 127 "on", else 0)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pretty_midi


class TranskunError(RuntimeError):
    """Raised when Transkun transcription fails. Never return empty notes on error."""


def midi_to_notes_and_pedals(
    midi_path: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse a Transkun-produced MIDI file into note and pedal-event lists."""
    pm = pretty_midi.PrettyMIDI(str(midi_path))

    notes: list[dict[str, Any]] = []
    pedals: list[dict[str, Any]] = []  # CC64 parsing added in T2
    for inst in pm.instruments:
        for n in inst.notes:
            notes.append({
                "pitch": int(n.pitch),
                "onset": round(float(n.start), 4),
                "offset": round(float(n.end), 4),
                "velocity": int(n.velocity),
            })
        for cc in inst.control_changes:
            if int(cc.number) != 64:
                continue
            pedals.append({
                "time": round(float(cc.time), 4),
                "value": 127 if int(cc.value) >= 64 else 0,
            })

    notes.sort(key=lambda n: (n["onset"], n["pitch"]))
    pedals.sort(key=lambda e: e["time"])
    return notes, pedals
