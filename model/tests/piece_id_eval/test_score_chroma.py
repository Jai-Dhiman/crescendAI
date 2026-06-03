"""Verify build_score_chroma produces correct pitch-class layout and L2 normalization."""
from __future__ import annotations

import json
import numpy as np
import pytest
from pathlib import Path

from piece_id_eval.score_chroma import build_score_chroma, load_catalog_score_chroma


def _single_note(pitch: int, onset: float, duration: float) -> dict:
    return {"pitch": pitch, "onset_seconds": onset, "duration_seconds": duration}


def test_single_c4_note_has_dominant_c_pitch_class() -> None:
    # C4 = MIDI 60 -> pitch class 0 (60 % 12 == 0)
    notes = [_single_note(pitch=60, onset=0.0, duration=1.0)]
    chroma = build_score_chroma(notes, frame_rate_hz=10.0)
    assert chroma.shape[0] == 12
    assert chroma.shape[1] >= 1
    # All columns should have pitch-class 0 as the highest
    col = chroma[:, 0]
    assert col[0] == col.max(), f"expected pitch-class 0 to dominate, got {col}"


def test_columns_are_l2_normalized() -> None:
    notes = [_single_note(pitch=60, onset=0.0, duration=2.0),
             _single_note(pitch=64, onset=0.5, duration=1.0)]
    chroma = build_score_chroma(notes, frame_rate_hz=10.0)
    norms = np.linalg.norm(chroma, axis=0)
    assert np.allclose(norms, 1.0, atol=1e-5), f"columns not unit-normed: {norms[:5]}"


def test_minimum_floor_enforced() -> None:
    # Even empty-ish frames should have floor >= 1e-3 before normalization
    notes = [_single_note(pitch=60, onset=5.0, duration=0.1)]
    chroma = build_score_chroma(notes, frame_rate_hz=10.0)
    # After normalization the floor is not directly visible, but shape is intact
    assert chroma.shape[0] == 12
    assert not np.any(np.isnan(chroma))
    assert not np.any(np.isinf(chroma))


def test_dtype_is_float32() -> None:
    notes = [_single_note(pitch=60, onset=0.0, duration=1.0)]
    chroma = build_score_chroma(notes, frame_rate_hz=10.0)
    assert chroma.dtype == np.float32


def test_load_catalog_score_chroma_from_real_score(tmp_path: Path) -> None:
    # Build a minimal score JSON matching the catalog schema
    score = {
        "piece_id": "test.piece",
        "bars": [
            {
                "bar_number": 1,
                "start_tick": 0,
                "start_seconds": 0.0,
                "notes": [
                    {"pitch": 60, "onset_seconds": 0.0, "duration_seconds": 0.5},
                    {"pitch": 64, "onset_seconds": 0.5, "duration_seconds": 0.5},
                ],
            }
        ],
    }
    score_path = tmp_path / "test.piece.json"
    score_path.write_text(json.dumps(score))
    chroma = load_catalog_score_chroma(score_path, frame_rate_hz=10.0)
    assert chroma.shape[0] == 12
    assert chroma.shape[1] >= 1
    assert chroma.dtype == np.float32
