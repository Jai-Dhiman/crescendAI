"""Run: cd model && uv run --with numpy --with pytest pytest \
        src/chroma_dtw_eval/test_dedup_amt_notes.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chroma_dtw_eval import amt_regen


def test_same_pitch_close_onsets_are_not_merged():
    notes = [
        {"pitch": 60, "onset": 1.00, "offset": 1.20, "velocity": 70},
        {"pitch": 60, "onset": 1.05, "offset": 1.25, "velocity": 72},  # 50ms later
    ]
    out = amt_regen._dedup_amt_notes(notes)
    assert len(out) == 2  # Transkun has no re-onset artifact; keep both
    assert sorted(round(n["onset"], 2) for n in out) == [1.00, 1.05]
