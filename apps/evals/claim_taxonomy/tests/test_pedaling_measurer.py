from __future__ import annotations
import math
import numpy as np
import pytest
from claim_taxonomy.verifier.measurers.pedaling import PedalingMeasurer, Measurement
from claim_taxonomy.verifier.models import UnverifiableError
from claim_taxonomy.verifier.location_resolver import ResolvedRegion
from claim_taxonomy.verifier.substrate_error import SubstrateErrorEngine


def _make_region(start: float = 0.0, end: float = 20.0) -> ResolvedRegion:
    return ResolvedRegion(
        audio_start_sec=start,
        audio_end_sec=end,
        alignment_uncertainty_sec=0.05,
        location_span_bars=10.0,
    )


def _make_bundle(pedal_events: list[dict], n_bars: int = 20, bar_dur: float = 2.0) -> dict:
    measure_table = [
        {"bar_number": i + 1, "start_sec": i * bar_dur, "start_tick": i * 480}
        for i in range(n_bars)
    ]
    return {
        "notes": [{"onset": i * 0.5, "offset": i * 0.5 + 0.4, "pitch": 60, "velocity": 80}
                  for i in range(n_bars * 4)],
        "pedal_events": pedal_events,
        "measure_table": measure_table,
        "anchors": {
            "perf_audio_sec": [0.0, n_bars * bar_dur],
            "score_audio_sec": [0.0, n_bars * bar_dur],
        },
        "substrate_versions": {"bundle_schema": "v1"},
    }


def test_no_pedal_in_region_negative_d() -> None:
    pedal_events = [{"time": 20.0 + i * 1.0, "value": 127} for i in range(20)]
    bundle = _make_bundle(pedal_events)
    engine = SubstrateErrorEngine(seed=42)
    measurer = PedalingMeasurer()
    region = _make_region(start=0.0, end=20.0)
    result = measurer.measure(
        location={"bar_start": 1, "bar_end": 10},
        bundle=bundle, region=region, engine=engine,
    )
    assert result.d < 0, f"Expected negative d (sparse pedaling), got {result.d}"


def test_dense_pedal_in_region_near_zero_or_positive_d() -> None:
    pedal_events = [{"time": i * 2.0 + 0.1, "value": 127} for i in range(20)]
    bundle = _make_bundle(pedal_events)
    engine = SubstrateErrorEngine(seed=42)
    measurer = PedalingMeasurer()
    region = _make_region(start=0.0, end=20.0)
    result = measurer.measure(
        location={"bar_start": 1, "bar_end": 10},
        bundle=bundle, region=region, engine=engine,
    )
    assert abs(result.d) < 0.35, f"Expected near-zero d for uniform pedaling, got {result.d}"


def test_no_pedal_anywhere_region_too_short() -> None:
    bundle = _make_bundle(pedal_events=[])
    engine = SubstrateErrorEngine(seed=42)
    measurer = PedalingMeasurer()
    region = _make_region(start=0.0, end=20.0)
    with pytest.raises(UnverifiableError) as exc_info:
        measurer.measure(
            location={"bar_start": 1, "bar_end": 10},
            bundle=bundle, region=region, engine=engine,
        )
    assert exc_info.value.reason_code == "region_too_short"


def test_whole_piece_pedal_fraction() -> None:
    pedal_events = [{"time": i * 2.0 + 0.1, "value": 127} for i in range(20)]
    bundle = _make_bundle(pedal_events)
    engine = SubstrateErrorEngine(seed=42)
    measurer = PedalingMeasurer()
    region = ResolvedRegion(
        audio_start_sec=0.0, audio_end_sec=40.0,
        alignment_uncertainty_sec=0.05, location_span_bars=math.inf
    )
    result = measurer.measure(location="whole_piece", bundle=bundle, region=region, engine=engine)
    assert 0.0 <= result.d <= 1.0, f"Whole-piece d should be fraction 0-1, got {result.d}"


def test_cc_injection_recovers_sparse_anomaly() -> None:
    pedal_events = [{"time": 20.0 + i * 1.5, "value": 127} for i in range(13)]
    bundle = _make_bundle(pedal_events)
    engine = SubstrateErrorEngine(seed=42)
    measurer = PedalingMeasurer()
    region = _make_region(start=0.0, end=20.0)
    result = measurer.measure(
        location={"bar_start": 1, "bar_end": 10},
        bundle=bundle, region=region, engine=engine,
    )
    tau = 0.25
    assert abs(result.d) > tau, (
        f"CC injection (remove region pedal) should produce |d|>{tau}, got d={result.d}"
    )


def test_error_bar_positive() -> None:
    pedal_events = [{"time": i * 2.0 + 0.1, "value": 127} for i in range(20)]
    bundle = _make_bundle(pedal_events)
    engine = SubstrateErrorEngine(seed=42)
    measurer = PedalingMeasurer()
    region = _make_region()
    result = measurer.measure(
        location={"bar_start": 1, "bar_end": 10},
        bundle=bundle, region=region, engine=engine,
    )
    assert result.error_bar >= 0.0
