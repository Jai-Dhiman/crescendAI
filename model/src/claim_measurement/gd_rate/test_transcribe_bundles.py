"""Run: cd model && uv run --no-project --with numpy --with pytest pytest \
        src/claim_measurement/gd_rate/test_transcribe_bundles.py"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transcribe_bundles as tb


def test_transcribe_windows_offsets_and_pools_via_injected_callable():
    calls = []
    def fake_transcribe(pcm):
        calls.append(len(pcm))
        return ([{"pitch": 60, "onset": 1.0, "offset": 1.5, "velocity": 70}],
                [{"time": 0.5, "value": 127}])

    audio = np.zeros(int(200 * tb.SAMPLE_RATE), dtype=np.float32)
    notes, pedals = tb._transcribe_windows(fake_transcribe, audio, [0.0, 30.0])

    assert len(calls) == 2
    # window starts 0.0 and 30.0 -> note onsets 1.0 and 31.0
    assert sorted(round(n["onset"], 1) for n in notes) == [1.0, 31.0]
    assert sorted(round(p["time"], 1) for p in pedals) == [0.5, 30.5]


def test_bundle_records_transkun_substrate():
    b = tb._build_bundle("chopin_ballade_1", "rid", notes=[], pedal_events=[],
                        duration_sec=10.0, window_starts=[0.0])
    assert b["substrate_versions"]["amt"].startswith("transkun/")
