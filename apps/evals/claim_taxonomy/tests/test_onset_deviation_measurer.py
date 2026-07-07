"""FRONT 7b: signed onset-deviation-vs-score timing measurer (#101).

The score-RELATIVE statistic that replaces the degenerate self-relative IOI-CV
(GATE 3). Truth by construction: "rushing" is definitionally playing ahead of the
score. d = mean(perf_onset - score_onset) in ms:
  - d < 0  -> perf onsets are EARLY  -> rushing   (matches frozen router polarity "-")
  - d > 0  -> perf onsets are LATE   -> dragging  (polarity "+")
Unit is ms; the frozen route_verdict is unit-agnostic (reads d, tau, |d|>tau).

The measurer consumes a SCORE-ALIGNED bundle: each note carries ``score_onset`` (sec,
the aligned score time from the offline aria-AMT -> parangonar bar-map). Notes without a
score correspondence are excluded (they carry no directional signal).
"""
from __future__ import annotations

import math

import pytest

from claim_taxonomy.verifier.measurers.onset_deviation import OnsetDeviationMeasurer, Measurement
from claim_taxonomy.verifier.models import UnverifiableError
from claim_taxonomy.verifier.location_resolver import ResolvedRegion
from claim_taxonomy.verifier.substrate_error import SubstrateErrorEngine


def _region(start: float = 0.0, end: float = 10.0) -> ResolvedRegion:
    return ResolvedRegion(
        audio_start_sec=start, audio_end_sec=end,
        alignment_uncertainty_sec=0.02, location_span_bars=4.0,
    )


def _aligned_bundle(pairs: list[tuple[float, float]]) -> dict:
    """pairs = [(perf_onset_sec, score_onset_sec), ...] -> a score-aligned bundle."""
    notes = [
        {"onset": p, "offset": p + 0.1, "pitch": 60, "velocity": 80, "score_onset": s}
        for (p, s) in pairs
    ]
    dur = max((p for p, _ in pairs), default=1.0)
    return {
        "notes": notes,
        "pedal_events": [],
        "measure_table": [{"bar_number": 1, "start_sec": 0.0},
                          {"bar_number": 2, "start_sec": dur}],
        "anchors": {"perf_audio_sec": [0.0, dur], "score_audio_sec": [0.0, dur]},
        "substrate_versions": {"amt": "aria-amt", "align": "parangonar"},
    }


def test_whole_piece_rush_gives_negative_d() -> None:
    # every perf onset 40ms EARLIER than score -> rushing -> d ~ -40ms
    pairs = [(i * 0.5 - 0.04, i * 0.5) for i in range(20)]
    bundle = _aligned_bundle(pairs)
    m = OnsetDeviationMeasurer().measure(
        location="whole_piece", bundle=bundle, region=_region(), engine=SubstrateErrorEngine(seed=42))
    assert m.d < 0, f"rush should be negative, got {m.d}"
    assert m.d == pytest.approx(-40.0, abs=1.0)
    assert m.event_count == 20
    assert not m.substrate_failure


def test_whole_piece_drag_gives_positive_d() -> None:
    # every perf onset 30ms LATER than score -> dragging -> d ~ +30ms
    pairs = [(i * 0.5 + 0.03, i * 0.5) for i in range(20)]
    bundle = _aligned_bundle(pairs)
    m = OnsetDeviationMeasurer().measure(
        location="whole_piece", bundle=bundle, region=_region(), engine=SubstrateErrorEngine(seed=42))
    assert m.d > 0
    assert m.d == pytest.approx(30.0, abs=1.0)


def test_whole_piece_in_time_near_zero() -> None:
    pairs = [(i * 0.5, i * 0.5) for i in range(20)]
    bundle = _aligned_bundle(pairs)
    m = OnsetDeviationMeasurer().measure(
        location="whole_piece", bundle=bundle, region=_region(), engine=SubstrateErrorEngine(seed=42))
    assert m.d == pytest.approx(0.0, abs=1.0)


def test_bar_region_filters_to_local_deviation() -> None:
    # first half (perf < 5s) drags +50ms; second half is in time. A region over the
    # first half must report the local drag, not the whole-piece average.
    early = [(i * 0.5 + 0.05, i * 0.5) for i in range(10)]      # 0..4.5s, +50ms
    late = [(5.0 + i * 0.5, 5.0 + i * 0.5) for i in range(10)]  # 5..9.5s, in time
    bundle = _aligned_bundle(early + late)
    m = OnsetDeviationMeasurer().measure(
        location={"bar_start": 1, "bar_end": 4}, bundle=bundle,
        region=_region(start=0.0, end=4.9), engine=SubstrateErrorEngine(seed=42))
    assert m.d == pytest.approx(50.0, abs=1.5)
    assert m.event_count == 10


def test_insufficient_events_raises() -> None:
    pairs = [(i * 0.5, i * 0.5) for i in range(3)]  # < MINIMUM_EVENTS
    bundle = _aligned_bundle(pairs)
    with pytest.raises(UnverifiableError) as ei:
        OnsetDeviationMeasurer().measure(
            location="whole_piece", bundle=bundle, region=_region(),
            engine=SubstrateErrorEngine(seed=42))
    assert ei.value.reason_code in ("region_too_short", "substrate_failure")


def test_no_score_alignment_raises_substrate_failure() -> None:
    # notes present but NONE carry score_onset -> no directional signal
    notes = [{"onset": i * 0.5, "offset": i * 0.5 + 0.1, "pitch": 60, "velocity": 80}
             for i in range(20)]
    bundle = {"notes": notes, "pedal_events": [],
              "measure_table": [{"bar_number": 1, "start_sec": 0.0}],
              "anchors": {"perf_audio_sec": [0.0, 10.0], "score_audio_sec": [0.0, 10.0]},
              "substrate_versions": {}}
    with pytest.raises(UnverifiableError) as ei:
        OnsetDeviationMeasurer().measure(
            location="whole_piece", bundle=bundle, region=_region(),
            engine=SubstrateErrorEngine(seed=42))
    assert ei.value.reason_code == "substrate_failure"


def test_error_bar_grows_with_alignment_uncertainty() -> None:
    pairs = [(i * 0.5 - 0.04, i * 0.5) for i in range(20)]
    bundle = _aligned_bundle(pairs)
    lo = OnsetDeviationMeasurer().measure(
        location="whole_piece", bundle=bundle,
        region=_region(start=0.0, end=10.0), engine=SubstrateErrorEngine(seed=42))
    hi_region = ResolvedRegion(audio_start_sec=0.0, audio_end_sec=10.0,
                               alignment_uncertainty_sec=0.20, location_span_bars=4.0)
    hi = OnsetDeviationMeasurer().measure(
        location="whole_piece", bundle=bundle, region=hi_region,
        engine=SubstrateErrorEngine(seed=42))
    assert hi.error_bar > lo.error_bar
    assert math.isfinite(lo.error_bar)
