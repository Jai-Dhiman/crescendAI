# model/tests/follower_eval/test_asap_audio.py
"""Unit tests for the MAESTRO-audio source for Track A (issue #133).

The load-bearing piece is ``derive_shift``. ASAP performances are excerpts of
longer MAESTRO recordings, so a transcription is in MAESTRO's clock while the
beat truth is in ASAP's. The metadata ``start`` column looks like the offset but
is wrong by 0.5 s for at least one real row (Bach/Fugue/bwv_858/Zhang01M) --
enough to wreck a beat-level number while still looking plausible. These tests
pin that we derive the offset from the notes and refuse to guess when we can't.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from follower_eval import asap_audio as aa


def _na(onsets, pitches, durations=None):
    """A partitura-shaped note_array."""
    d = durations if durations is not None else [0.2] * len(onsets)
    return np.array(list(zip(onsets, pitches, d)),
                    dtype=[("onset_sec", "f8"), ("pitch", "i4"), ("duration_sec", "f8")])


def _patch_midis(monkeypatch, by_path):
    """Stub partitura so the tests exercise shift logic, not MIDI parsing."""
    class _P:
        def __init__(self, na): self._na = na
        def note_array(self): return self._na
    monkeypatch.setattr(aa.pa, "load_performance_midi",
                        lambda p: _P(by_path[Path(p).name]))


def test_derive_shift_ignores_a_wrong_metadata_start(monkeypatch):
    # the real Bach/Fugue/bwv_858 case: metadata says 90.33, truth is 89.83
    pitches = [60, 64, 67, 72, 65, 62, 59, 55, 60, 64]
    asap = _na([0.5 + i for i in range(10)], pitches)
    maestro = _na([89.83 + 0.5 + i for i in range(10)], pitches)
    _patch_midis(monkeypatch, {"a.mid": asap, "m.mid": maestro})

    s = aa.derive_shift(Path("a.mid"), Path("m.mid"), metadata_start=90.33)
    assert s.seconds == pytest.approx(89.83, abs=0.005)
    assert s.match_frac == 1.0
    assert s.metadata_start == 90.33     # recorded, but not used


def test_derive_shift_zero_offset_case(monkeypatch):
    pitches = [60, 62, 64, 65, 67, 69, 71, 72, 60, 62]
    na = _na([1.0 + i for i in range(10)], pitches)
    _patch_midis(monkeypatch, {"a.mid": na, "m.mid": na})
    assert aa.derive_shift(Path("a.mid"), Path("m.mid"), 0.0).seconds == pytest.approx(0.0)


def test_derive_shift_refuses_when_no_offset_explains_the_notes(monkeypatch):
    # same pitch material, incompatible timing (different performance): no single
    # shift lines the notes up -> must raise, never return a best-effort offset
    # that silently misplaces the truth
    asap = _na([0.5 * i for i in range(30)], [60 + (i % 12) for i in range(30)])
    maestro = _na([0.37 * i ** 1.3 for i in range(30)], [60 + (i % 12) for i in range(30)])
    _patch_midis(monkeypatch, {"a.mid": asap, "m.mid": maestro})
    with pytest.raises(aa.AsapAudioError, match="could not derive a time offset"):
        aa.derive_shift(Path("a.mid"), Path("m.mid"), 0.0)


def test_derive_shift_empty_is_loud(monkeypatch):
    _patch_midis(monkeypatch, {"a.mid": _na([], []), "m.mid": _na([1.0], [60])})
    with pytest.raises(aa.AsapAudioError, match="empty note array"):
        aa.derive_shift(Path("a.mid"), Path("m.mid"), 0.0)


def test_clip_bounds_pads_around_the_performance(monkeypatch):
    _patch_midis(monkeypatch, {"a.mid": _na([10.0, 20.0], [60, 62], [1.0, 1.0])})
    lo, hi = aa.clip_bounds(Path("a.mid"), shift=5.0)
    assert lo == pytest.approx(10.0 + 5.0 - aa.PAD_SEC)
    assert hi == pytest.approx(21.0 + 5.0 + aa.PAD_SEC)


def test_clip_bounds_never_negative(monkeypatch):
    _patch_midis(monkeypatch, {"a.mid": _na([0.1], [60], [0.1])})
    assert aa.clip_bounds(Path("a.mid"), shift=0.0)[0] == 0.0


def test_maestro_paths_missing_link_is_loud(tmp_path):
    with pytest.raises(aa.AsapAudioError, match="no maestro_audio_performance"):
        aa.maestro_paths({"midi_performance": "X/y.mid", "maestro_audio_performance": ""},
                         tmp_path, tmp_path)


def test_maestro_paths_missing_audio_is_loud(tmp_path):
    row = {"midi_performance": "X/y.mid",
           "maestro_audio_performance": "{maestro}/2006/rec.wav",
           "maestro_midi_performance": "{maestro}/2006/rec.midi"}
    with pytest.raises(aa.AsapAudioError, match="missing MAESTRO audio"):
        aa.maestro_paths(row, tmp_path, tmp_path)


def test_bundle_path_is_reversible():
    # asap_eval reconstructs piece keys from bundle filenames; the mapping must
    # survive the round trip or audio-mode silently evaluates nothing
    key = "Bach/Fugue/bwv_858/Zhang01M.mid"
    p = aa.bundle_path(Path("/c"), key)
    assert p.name == "Bach__Fugue__bwv_858__Zhang01M.json"
    assert p.stem.replace("__", "/") + ".mid" == key
