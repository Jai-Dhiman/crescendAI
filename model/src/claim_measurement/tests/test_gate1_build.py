"""Spec dispatch for the GATE 1 corrupt-bundle runner (pure, no AMT)."""
from __future__ import annotations

import numpy as np
import pytest

from claim_measurement.gate1.build_corrupt_bundles import (
    CorruptionSpec,
    build_corrupted_audio,
)
from claim_measurement.gate1.corruption import warp_time

SR = 16000


def _tone(dur: float = 8.0, sr: int = SR) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def test_clean_spec_is_identity():
    a = _tone()
    out, wm = build_corrupted_audio(a, SR, CorruptionSpec("clean", {}), np.random.default_rng(0))
    assert np.array_equal(out, a)
    assert warp_time(wm, 3.0) == pytest.approx(3.0, abs=1e-6)


def test_tempo_spec_shortens_and_warps():
    a = _tone()
    spec = CorruptionSpec("tempo", {"segments": [[2.0, 6.0, 2.0]]})
    out, wm = build_corrupted_audio(a, SR, spec, np.random.default_rng(0))
    assert len(out) < len(a)
    assert warp_time(wm, 4.0) == pytest.approx(3.0, abs=0.15)


def test_noise_spec_is_identity_length():
    a = _tone()
    spec = CorruptionSpec("noise", {"snr_db": 10.0})
    out, wm = build_corrupted_audio(a, SR, spec, np.random.default_rng(0))
    assert len(out) == len(a)
    assert not np.allclose(out, a)
    assert warp_time(wm, 5.0) == pytest.approx(5.0, abs=1e-6)


def test_dropout_spec_zeroes_window():
    a = _tone()
    spec = CorruptionSpec("dropout", {"start_sec": 2.0, "end_sec": 3.0})
    out, wm = build_corrupted_audio(a, SR, spec, np.random.default_rng(0))
    assert np.allclose(out[int(2.0 * SR):int(3.0 * SR)], 0.0)
    assert warp_time(wm, 5.0) == pytest.approx(5.0, abs=1e-6)


def test_pitch_spec_changes_region_only():
    a = _tone()
    spec = CorruptionSpec("pitch", {"start_sec": 2.0, "end_sec": 4.0, "semitones": 2.0})
    out, wm = build_corrupted_audio(a, SR, spec, np.random.default_rng(0))
    assert not np.allclose(out[int(2.0 * SR):int(4.0 * SR)], a[int(2.0 * SR):int(4.0 * SR)])
    assert np.allclose(out[: int(2.0 * SR)], a[: int(2.0 * SR)])


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        build_corrupted_audio(_tone(), SR, CorruptionSpec("bogus", {}), np.random.default_rng(0))
