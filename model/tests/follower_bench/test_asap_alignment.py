# model/tests/follower_bench/test_asap_alignment.py
"""Verify load_alignment resolves paths and validates the ASAP beat
alignment, through its public interface only, against real committed
ASAP fixtures (no synthetic annotation files -- the real dataset already
has both a clean-aligned and a not-aligned example)."""
from __future__ import annotations

from pathlib import Path

from follower_bench.asap_alignment import load_alignment

REPO_ROOT = Path(__file__).resolve().parents[3]
ASAP_ROOT = REPO_ROOT / "model/data/raw/asap-dataset"
ALIGNED_PIECE = "Liszt/Transcendental_Etudes/1/LuoJ05M.mid"


def test_load_alignment_resolves_real_aligned_piece() -> None:
    alignment = load_alignment(ALIGNED_PIECE)
    assert alignment.asap_piece == ALIGNED_PIECE
    assert alignment.performance_midi_path == ASAP_ROOT / ALIGNED_PIECE
    assert alignment.performance_midi_path.exists()
    assert alignment.score_midi_path == ASAP_ROOT / "Liszt/Transcendental_Etudes/1/midi_score.mid"
    assert alignment.score_midi_path.exists()
    assert len(alignment.performance_beats) == len(alignment.midi_score_beats)
    assert len(alignment.performance_beats) == 92
    assert alignment.performance_beats[0] < alignment.performance_beats[-1]
