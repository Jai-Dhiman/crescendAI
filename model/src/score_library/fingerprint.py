"""Score fingerprinting: the v2 piece-ID artifact (chroma + chord-events).

Per piece: a 12-bin L2-normalized velocity-weighted chroma vector (the certified
recall feature, mirrors piece_id_eval.note_chroma.chroma_vector) and a sequence of
12-bit pitch-class-set chord-event masks (onsets collapsed within 50ms, mirrors
piece_id_eval.stage0c_elastic_dtwgate._notes_to_events).
"""

from __future__ import annotations

import json
import math
from pathlib import Path


def _collect_all_notes(score_data: dict) -> list[dict]:
    """Return flat list of all note dicts across all bars."""
    notes = []
    for bar in score_data.get("bars", []):
        notes.extend(bar.get("notes", []))
    return notes


def _piece_chroma(notes: list[dict]) -> list[float]:
    """12-bin key-dependent velocity-weighted pitch-class histogram, L2-normalized.

    Mirrors piece_id_eval.note_chroma.chroma_vector (the certified recall feature).
    """
    cv = [0.0] * 12
    for n in notes:
        cv[int(n["pitch"]) % 12] += float(n.get("velocity", 80))
    norm = math.sqrt(sum(x * x for x in cv))
    if norm > 0:
        cv = [x / norm for x in cv]
    return cv


def _piece_events(notes: list[dict], onset_tol_s: float) -> list[int]:
    """Collapse onsets within onset_tol_s into chord-events; each = a 12-bit pc-set mask.

    Mirrors piece_id_eval.stage0c_elastic_dtwgate._notes_to_events (pitch-only).
    """
    if not notes:
        return []
    ordered = sorted(notes, key=lambda n: float(n["onset_seconds"]))
    events: list[int] = []
    anchor = float(ordered[0]["onset_seconds"])
    cur = 0
    for n in ordered:
        onset = float(n["onset_seconds"])
        if onset - anchor > onset_tol_s:
            events.append(cur)
            anchor = onset
            cur = 0
        cur |= 1 << (int(n["pitch"]) % 12)
    events.append(cur)
    return events


_WINDOW_NOTES = 400
_WINDOW_HOP = 200


def _piece_windows(notes: list[dict], window_notes: int = _WINDOW_NOTES,
                   window_hop: int = _WINDOW_HOP) -> list[list[float]]:
    """Per-piece overlapping note-WINDOW chroma vectors for the hybrid shortlist (#96).

    Mirrors score_library.pieceid_experimental.HybridShortlist EXACTLY: notes are
    onset-ordered, then a window_notes-wide window slides by window_hop and each
    window's chroma is emitted. Pieces with <= window_notes notes emit NO windows
    (the whole-piece chroma already covers them in the union). The trailing-window
    guard keeps only windows with >= max(2, window_notes//2) notes.
    """
    ordered = sorted(notes, key=lambda n: float(n["onset_seconds"]))
    n = len(ordered)
    if n <= window_notes:
        return []
    out: list[list[float]] = []
    start = 0
    min_len = max(2, window_notes // 2)
    while start < n:
        w = ordered[start:start + window_notes]
        if len(w) >= min_len:
            out.append(_piece_chroma(w))
        start += window_hop
    return out


def build_piece_index(scores_dir: Path, onset_tol_s: float = 0.05) -> dict:
    """Build the v2 piece-ID artifact (chroma vector + chord-event masks per piece)."""
    json_files = sorted(
        f for f in scores_dir.glob("*.json") if f.name not in ("titles.json", "seed.sql")
    )
    pieces: list[dict] = []
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        notes = _collect_all_notes(data)
        pieces.append({
            "piece_id": data["piece_id"],
            "composer": data["composer"],
            "title": data["title"],
            "chroma": _piece_chroma(notes),
            "events": _piece_events(notes, onset_tol_s),
            "windows": _piece_windows(notes),
        })
    pieces.sort(key=lambda p: p["piece_id"])
    return {"version": "v2", "onset_tol_ms": int(round(onset_tol_s * 1000)), "pieces": pieces}
