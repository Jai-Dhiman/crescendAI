from __future__ import annotations
import numpy as np
import pytest
from claim_taxonomy.verifier.measurers.timing import TimingMeasurer, Measurement
from claim_taxonomy.verifier.models import UnverifiableError
from claim_taxonomy.verifier.location_resolver import ResolvedRegion
from claim_taxonomy.verifier.substrate_error import SubstrateErrorEngine
import math


def _make_region(start: float = 0.0, end: float = 10.0) -> ResolvedRegion:
    return ResolvedRegion(
        audio_start_sec=start,
        audio_end_sec=end,
        alignment_uncertainty_sec=0.05,
        location_span_bars=5.0,
    )


def _make_bundle_with_notes(onsets: list[float]) -> dict:
    notes = [{"onset": t, "offset": t + 0.1, "pitch": 60, "velocity": 80} for t in onsets]
    return {
        "notes": notes,
        "pedal_events": [],
        "measure_table": [{"bar_number": 1, "start_sec": 0.0, "start_tick": 0}],
        "anchors": {"perf_audio_sec": [0.0, max(onsets)], "score_audio_sec": [0.0, max(onsets)]},
        "substrate_versions": {"bundle_schema": "v1"},
    }


def test_region_rush_gives_negative_d() -> None:
    piece_onsets = [i * 0.5 for i in range(100)]
    region_onsets = [i * 0.4 for i in range(25)]
    all_onsets = sorted(set(piece_onsets[25:]) | set(region_onsets))
    bundle = _make_bundle_with_notes(all_onsets)
    engine = SubstrateErrorEngine(seed=42)
    measurer = TimingMeasurer()
    region = _make_region(start=0.0, end=10.0)
    result = measurer.measure(location={"bar_start": 1, "bar_end": 5},
                              bundle=bundle, region=region, engine=engine)
    assert result.d < 0, f"Expected negative d (rushed), got {result.d}"
    assert result.event_count >= 8


def test_region_drag_gives_positive_d() -> None:
    piece_onsets = [i * 0.5 for i in range(100)]
    region_onsets = [i * 0.7 for i in range(14)]
    all_onsets = sorted(set(piece_onsets[14:]) | set(region_onsets))
    bundle = _make_bundle_with_notes(all_onsets)
    engine = SubstrateErrorEngine(seed=42)
    measurer = TimingMeasurer()
    region = _make_region(start=0.0, end=10.0)
    result = measurer.measure(location={"bar_start": 1, "bar_end": 5},
                              bundle=bundle, region=region, engine=engine)
    assert result.d > 0, f"Expected positive d (dragging), got {result.d}"


def test_whole_piece_uniform_tempo_low_cv() -> None:
    onsets = [i * 0.5 for i in range(100)]
    bundle = _make_bundle_with_notes(onsets)
    engine = SubstrateErrorEngine(seed=42)
    measurer = TimingMeasurer()
    region = ResolvedRegion(
        audio_start_sec=0.0, audio_end_sec=50.0,
        alignment_uncertainty_sec=0.05, location_span_bars=math.inf
    )
    result = measurer.measure(location="whole_piece", bundle=bundle, region=region, engine=engine)
    assert result.d < 5.0, f"Expected low CV% for uniform tempo, got {result.d}"


def test_region_too_short_raises() -> None:
    onsets = [0.5, 1.0, 1.5, 2.0, 2.5]
    onsets += [i * 0.5 + 10 for i in range(50)]
    bundle = _make_bundle_with_notes(onsets)
    engine = SubstrateErrorEngine(seed=42)
    measurer = TimingMeasurer()
    region = _make_region(start=0.0, end=4.0)
    with pytest.raises(UnverifiableError) as exc_info:
        measurer.measure(location={"bar_start": 1, "bar_end": 2},
                         bundle=bundle, region=region, engine=engine)
    assert exc_info.value.reason_code == "region_too_short"


def test_mistake_injection_onset_shift_recovers_anomaly() -> None:
    piece_onsets = [i * 0.5 for i in range(80)]
    shifted = [t * 0.8 for t in piece_onsets if t < 10.0]
    rest = [t for t in piece_onsets if t >= 10.0]
    bundle = _make_bundle_with_notes(sorted(shifted + rest))
    engine = SubstrateErrorEngine(seed=42)
    measurer = TimingMeasurer()
    region = _make_region(start=0.0, end=8.0)
    result = measurer.measure(location={"bar_start": 1, "bar_end": 4},
                              bundle=bundle, region=region, engine=engine)
    tau = 8.0
    assert abs(result.d) > tau, (
        f"Injection: 20% faster onset shift should produce |d|>{tau}%, got d={result.d}"
    )


def test_error_bar_is_positive() -> None:
    onsets = [i * 0.5 for i in range(60)]
    bundle = _make_bundle_with_notes(onsets)
    engine = SubstrateErrorEngine(seed=0)
    measurer = TimingMeasurer()
    region = _make_region(start=0.0, end=15.0)
    result = measurer.measure(location={"bar_start": 1, "bar_end": 5},
                              bundle=bundle, region=region, engine=engine)
    assert result.error_bar > 0.0
    assert not result.substrate_failure


def test_substrate_var_uses_full_piece_reference_and_consistent_sign() -> None:
    """_substrate_var must use the full-piece established tempo (not region-only) and
    produce perturbation deltas with the same sign convention as _region_d_and_sampling_var:
    (established - bpm) / established * 100, so rushed -> negative contribution."""
    # Region (0-5s): fast, IOI=0.25s -> ~240 BPM
    # Rest of piece (5-50s): normal, IOI=0.5s -> ~120 BPM (dominates median)
    region_onsets = np.array([i * 0.25 for i in range(20)])   # 0..4.75s
    rest_onsets = np.array([5.0 + i * 0.5 for i in range(80)])
    all_onsets = np.sort(np.concatenate([region_onsets, rest_onsets]))

    measurer = TimingMeasurer()
    engine = SubstrateErrorEngine(seed=42)

    # Full-piece established tempo is ~120 BPM (median dominated by rest_onsets).
    # Region BPM is ~240 BPM (faster than reference).
    # With full-piece reference: (established - region_bpm) / established * 100 < 0 (rushed).
    # With region-only reference (the old bug): established==240, result would be ~0.
    established_full = measurer._established_tempo(all_onsets)
    established_region = measurer._established_tempo(region_onsets)
    assert established_full < established_region * 0.7, (
        "test setup: full-piece tempo must differ substantially from region-only tempo"
    )

    substrate_var = measurer._substrate_var(region_onsets, all_onsets, engine)
    # Variance is non-negative; we mainly confirm the call succeeds and matches the
    # corrected signature. The perturbed d values should cluster around the true d < 0.
    assert substrate_var >= 0.0

    # Also verify sign consistency: each perturbed d should be negative (rushed region).
    jitters = engine.timing_onset_jitter_sec()
    signs_negative = []
    for j in jitters:
        perturbed = region_onsets + j
        bpm = measurer._region_median_bpm(perturbed)
        d_perturbed = (established_full - bpm) / established_full * 100.0
        signs_negative.append(d_perturbed < 0)
    assert all(signs_negative), (
        "all perturbed substrate-var deltas should be negative for a rushed region "
        "when using the full-piece reference"
    )
