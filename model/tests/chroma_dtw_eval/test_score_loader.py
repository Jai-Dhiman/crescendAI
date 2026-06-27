"""Score loader generalization: variable-tempo / non-4/4 support (#98).

The loader consumes the score JSON's precomputed `start_seconds` (per bar) and
`onset_seconds` (per note) for the score-time axis, so it does not need a constant
tempo. The only tempo-dependent quantity (duration_beat) is derived from
`duration_ticks / ticks_per_quarter` (metric, tempo-independent). These tests pin
that the 4/4 constant-tempo path is unchanged and that variable-tempo / non-4/4
scores now load instead of raising.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from chroma_dtw_eval.amt_regen import _load_bach_json_score


def _note(pitch, onset_tick, onset_sec, dur_ticks=480, dur_sec=0.5):
    return {
        "pitch": pitch,
        "onset_tick": onset_tick,
        "onset_seconds": onset_sec,
        "duration_ticks": dur_ticks,
        "duration_seconds": dur_sec,
        "velocity": 80,
    }


def _score(tmp_path, tempos, tsigs, bars) -> Path:
    body = {"tempo_markings": tempos, "time_signatures": tsigs, "bars": bars}
    p = tmp_path / "score.json"
    p.write_text(json.dumps(body))
    return p


def test_constant_tempo_4_4_unchanged(tmp_path):
    # PPQ 480, 4/4 -> ticks_per_bar 1920; two bars at 120bpm (beat_sec 0.5).
    bars = [
        {"bar_number": 1, "start_tick": 0, "start_seconds": 0.0,
         "notes": [_note(60, 0, 0.0), _note(64, 480, 0.5)]},
        {"bar_number": 2, "start_tick": 1920, "start_seconds": 2.0,
         "notes": [_note(67, 1920, 2.0)]},
    ]
    path = _score(tmp_path,
                  [{"tick": 0, "tempo_usec": 500000, "bpm": 120.0}],
                  [{"tick": 0, "numerator": 4, "denominator": 4}],
                  bars)
    score_na, measure_table, _sha, beat_sec = _load_bach_json_score(path)

    assert beat_sec == pytest.approx(0.5)
    assert measure_table[0]["start_sec"] == 0.0
    assert measure_table[1]["start_sec"] == 2.0
    # onset_beat = onset_tick / 480; the note at tick 480 is beat 1.0.
    beats = {int(r["pitch"]): float(r["onset_beat"]) for r in score_na}
    assert beats[60] == pytest.approx(0.0)
    assert beats[64] == pytest.approx(1.0)
    # onset_sec axis = precomputed onset_seconds.
    secs = {int(r["pitch"]): float(r["onset_sec"]) for r in score_na}
    assert secs[64] == pytest.approx(0.5)


def test_variable_tempo_no_longer_rejected(tmp_path):
    # Two tempo markings: previously raised AmtRegenError.
    bars = [
        {"bar_number": 1, "start_tick": 0, "start_seconds": 0.0,
         "notes": [_note(60, 0, 0.0)]},
        {"bar_number": 2, "start_tick": 1920, "start_seconds": 2.0,
         "notes": [_note(62, 1920, 2.0)]},
        # Tempo halves at bar 3 -> bar takes 4s of audio, not 2 (non-uniform axis).
        {"bar_number": 3, "start_tick": 3840, "start_seconds": 6.0,
         "notes": [_note(64, 3840, 6.0)]},
    ]
    path = _score(tmp_path,
                  [{"tick": 0, "tempo_usec": 500000, "bpm": 120.0},
                   {"tick": 3840, "tempo_usec": 1000000, "bpm": 60.0}],
                  [{"tick": 0, "numerator": 4, "denominator": 4}],
                  bars)
    score_na, measure_table, _sha, _beat = _load_bach_json_score(path)
    # Non-uniform score-time axis preserved from precomputed start_seconds.
    assert [m["start_sec"] for m in measure_table] == [0.0, 2.0, 6.0]
    assert len(score_na) == 3


def test_non_4_4_supported(tmp_path):
    # 3/4 at PPQ 480 -> ticks_per_bar 1440. ticks_per_quarter must resolve to 480
    # so onset_beat of the 2nd-beat note (tick 480) is 1.0.
    bars = [
        {"bar_number": 1, "start_tick": 0, "start_seconds": 0.0,
         "notes": [_note(60, 0, 0.0), _note(62, 480, 0.5)]},
        {"bar_number": 2, "start_tick": 1440, "start_seconds": 1.5,
         "notes": [_note(64, 1440, 1.5)]},
    ]
    path = _score(tmp_path,
                  [{"tick": 0, "tempo_usec": 500000, "bpm": 120.0}],
                  [{"tick": 0, "numerator": 3, "denominator": 4}],
                  bars)
    score_na, measure_table, _sha, _beat = _load_bach_json_score(path)
    beats = {int(r["pitch"]): float(r["onset_beat"]) for r in score_na}
    assert beats[62] == pytest.approx(1.0)
    assert [m["start_sec"] for m in measure_table] == [0.0, 1.5]
