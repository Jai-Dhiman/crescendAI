"""Synthetic chroma fingerprint from catalog score JSON.

Mirrors the pitch-class accumulation in apps/api/src/wasm/score-analysis/src/chroma_dtw.rs::build_score_chroma:
  - One column per frame (1 / frame_rate_hz seconds wide)
  - Each note contributes its pitch-class (pitch % 12) to every frame it spans
  - 1e-3 floor applied before per-column L2 normalization
  - Output shape: (12, N), dtype float32
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def build_score_chroma(notes: list[dict], frame_rate_hz: float) -> np.ndarray:
    """Build a (12, N) float32 synthetic chroma from a list of note dicts.

    Each note dict must contain:
      - pitch (int): MIDI pitch number
      - onset_seconds (float): note start time in seconds
      - duration_seconds (float): note duration in seconds

    Raises:
        ValueError: if notes is empty or frame_rate_hz <= 0.
    """
    if frame_rate_hz <= 0:
        raise ValueError(f"frame_rate_hz must be positive, got {frame_rate_hz}")
    if not notes:
        raise ValueError("notes list is empty; cannot build score chroma")

    # Determine total duration from last note end
    end_sec = max(n["onset_seconds"] + n["duration_seconds"] for n in notes)
    n_frames = max(1, int(np.ceil(end_sec * frame_rate_hz)))

    chroma = np.zeros((12, n_frames), dtype=np.float32)
    for note in notes:
        pc = int(note["pitch"]) % 12
        onset_f = int(note["onset_seconds"] * frame_rate_hz)
        end_f = max(onset_f + 1, int((note["onset_seconds"] + note["duration_seconds"]) * frame_rate_hz))
        onset_f = max(0, min(onset_f, n_frames - 1))
        end_f = min(end_f, n_frames)
        chroma[pc, onset_f:end_f] += 1.0

    chroma += 1e-3
    norms = np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9
    chroma /= norms
    return chroma


def load_catalog_score_chroma(score_path: Path, frame_rate_hz: float) -> np.ndarray:
    """Load a catalog score JSON and return its synthetic chroma fingerprint.

    Raises:
        FileNotFoundError: if score_path does not exist.
        KeyError: if the JSON is missing required 'bars' key.
    """
    if not score_path.exists():
        raise FileNotFoundError(f"score JSON not found: {score_path}")
    data = json.loads(score_path.read_text())
    if "bars" not in data:
        raise KeyError(f"score JSON at {score_path} missing 'bars' key")
    notes: list[dict] = []
    for bar in data["bars"]:
        for note in bar.get("notes", []):
            notes.append({
                "pitch": note["pitch"],
                "onset_seconds": float(note["onset_seconds"]),
                "duration_seconds": float(note["duration_seconds"]),
            })
    if not notes:
        raise ValueError(f"score JSON at {score_path} contains no notes")
    return build_score_chroma(notes, frame_rate_hz)
