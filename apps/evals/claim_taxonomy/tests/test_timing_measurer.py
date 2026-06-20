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
