"""GATE 1 corruption harness: construction-known audio degradation.

Each corruption returns the corrupted audio plus a `warp_map` describing the
exact clean_sec -> corrupt_sec mapping it induced. The map IS the ground truth:
it is derived from the actual produced sample counts, not from the requested
factors, so localization error is measured against a known transform.
"""
from __future__ import annotations

import numpy as np
import pytest

from claim_measurement.gate1.corruption import (
    add_noise,
    apply_piecewise_time_warp,
    pitch_shift_region,
    silence_region,
    warp_time,
)

SR = 16000


def _tone(dur_sec: float, freq: float = 440.0, sr: int = SR) -> np.ndarray:
    t = np.arange(int(dur_sec * sr)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_time_warp_map_is_monotonic_and_matches_output_duration():
    audio = _tone(10.0)
    segments = [(2.0, 6.0, 2.0)]  # speed up [2s,6s] by 2x -> region 4s becomes ~2s
    corrupt, warp_map = apply_piecewise_time_warp(audio, SR, segments)

    # Output is ~2s shorter (a 4s region compressed to ~2s).
    assert abs(len(corrupt) / SR - (len(audio) / SR - 2.0)) < 0.15

    # Identity before the region.
    assert warp_time(warp_map, 1.0) == pytest.approx(1.0, abs=0.05)

    # Middle of the sped-up region: clean 4.0s -> 2.0 + (4-2)/2 = 3.0s.
    assert warp_time(warp_map, 4.0) == pytest.approx(3.0, abs=0.15)

    # The map is strictly monotonic (a valid time reparameterization).
    ts = np.linspace(0.0, 9.9, 200)
    mapped = np.array([warp_time(warp_map, float(t)) for t in ts])
    assert np.all(np.diff(mapped) >= -1e-9)


def test_time_warp_map_matches_a_planted_landmark():
    # A single click at a known clean time should land at warp_time(click) in
    # the corrupted audio (within a phase-vocoder frame).
    audio = np.zeros(int(8.0 * SR), dtype=np.float32)
    click_clean = 5.0
    audio[int(click_clean * SR)] = 1.0
    segments = [(1.0, 4.0, 1.5)]  # compress [1,4] by 1.5x; click at 5.0 is after region
    corrupt, warp_map = apply_piecewise_time_warp(audio, SR, segments)

    expected = warp_time(warp_map, click_clean)
    detected = float(np.argmax(np.abs(corrupt))) / SR
    assert detected == pytest.approx(expected, abs=0.06)  # ~1 STFT hop tolerance


def test_add_noise_is_identity_in_time_and_hits_target_snr():
    audio = _tone(4.0)
    rng = np.random.default_rng(0)
    noisy = add_noise(audio, snr_db=10.0, rng=rng)

    assert len(noisy) == len(audio)  # W = identity (no time shift)
    sig_p = float(np.mean(audio**2))
    noise_p = float(np.mean((noisy - audio) ** 2))
    measured_snr = 10.0 * np.log10(sig_p / noise_p)
    assert measured_snr == pytest.approx(10.0, abs=1.0)


def test_silence_region_preserves_duration_and_zeroes_window():
    audio = _tone(6.0)
    out = silence_region(audio, SR, start_sec=2.0, end_sec=3.0)
    assert len(out) == len(audio)  # W = identity
    assert np.allclose(out[int(2.0 * SR):int(3.0 * SR)], 0.0)
    assert np.allclose(out[: int(2.0 * SR)], audio[: int(2.0 * SR)])


def test_pitch_shift_region_preserves_duration():
    audio = _tone(6.0, freq=440.0)
    out = pitch_shift_region(audio, SR, start_sec=2.0, end_sec=4.0, semitones=2.0)
    assert len(out) == len(audio)  # W = identity (duration preserved)
    # The shifted window changed; the untouched head is unchanged.
    assert not np.allclose(out[int(2.0 * SR):int(4.0 * SR)], audio[int(2.0 * SR):int(4.0 * SR)])
    assert np.allclose(out[: int(2.0 * SR)], audio[: int(2.0 * SR)])
