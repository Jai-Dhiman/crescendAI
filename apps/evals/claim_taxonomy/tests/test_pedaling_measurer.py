from __future__ import annotations
import math
import pytest
from claim_taxonomy.verifier.measurers.pedaling import (
    PedalingMeasurer,
    REFERENCE_FRACTION,
)
from claim_taxonomy.verifier.models import UnverifiableError
from claim_taxonomy.verifier.location_resolver import ResolvedRegion
from claim_taxonomy.verifier.substrate_error import SubstrateErrorEngine


def _region(start: float, end: float) -> ResolvedRegion:
    return ResolvedRegion(
        audio_start_sec=start,
        audio_end_sec=end,
        alignment_uncertainty_sec=0.05,
        location_span_bars=10.0,
    )


def _whole(total: float) -> ResolvedRegion:
    return ResolvedRegion(
        audio_start_sec=0.0, audio_end_sec=total,
        alignment_uncertainty_sec=0.05, location_span_bars=math.inf,
    )


def _events(spans: list[tuple[float, float]]) -> list[dict]:
    """CC64 on/off event stream from (on, off) spans."""
    evs = []
    for a, b in spans:
        evs.append({"time": a, "value": 127})
        evs.append({"time": b, "value": 0})
    return evs


def _bundle(spans: list[tuple[float, float]], total: float = 40.0) -> dict:
    # notes span [0, total] so _total_duration resolves to `total`
    n = max(int(total / 0.5), 1)
    notes = [
        {"onset": i * 0.5, "offset": min(i * 0.5 + 0.4, total), "pitch": 60, "velocity": 80}
        for i in range(n)
    ]
    # anchor the final note offset exactly at `total` so _total_duration == total
    notes.append({"onset": max(total - 0.1, 0.0), "offset": total, "pitch": 60, "velocity": 80})
    return {
        "notes": notes,
        "pedal_events": _events(spans),
        "substrate_versions": {"bundle_schema": "v1"},
    }


def _measure(location, bundle, region):
    return PedalingMeasurer().measure(
        location=location, bundle=bundle, region=region, engine=SubstrateErrorEngine(seed=42)
    )


# --- whole_piece: signed time-on-fraction vs REFERENCE_FRACTION ---

def test_dry_performance_measures_negative_d_not_abstention() -> None:
    """A genuinely dry (zero-pedal) performance is the signed under-pedal case:
    d = 0 - REFERENCE_FRACTION < 0, NOT an UnverifiableError. This is the core of
    front #3 -- it makes '-' (under-pedaled) claims adjudicable."""
    bundle = _bundle(spans=[], total=40.0)
    result = _measure("whole_piece", bundle, _whole(40.0))
    assert result.d == pytest.approx(-REFERENCE_FRACTION, abs=1e-9)
    assert result.d < 0
    assert result.event_count == 0


def test_heavy_pedal_positive_d() -> None:
    bundle = _bundle(spans=[(0.0, 38.0)], total=40.0)  # on_fraction = 0.95
    result = _measure("whole_piece", bundle, _whole(40.0))
    assert result.d == pytest.approx(0.95 - REFERENCE_FRACTION, abs=1e-6)
    assert result.d > 0


def test_on_fraction_equal_to_reference_near_zero_d() -> None:
    total = 100.0
    bundle = _bundle(spans=[(0.0, REFERENCE_FRACTION * total)], total=total)
    result = _measure("whole_piece", bundle, _whole(total))
    assert abs(result.d) < 1e-6


def test_whole_piece_d_in_fraction_units() -> None:
    bundle = _bundle(spans=[(0.0, 20.0)], total=40.0)  # on_fraction = 0.5
    result = _measure("whole_piece", bundle, _whole(40.0))
    assert -1.0 <= result.d <= 1.0


# --- region: signed vs whole-piece on-fraction (within-clip, same units) ---

def test_region_drier_than_piece_negative_d() -> None:
    # pedal everywhere except the first 10s region
    bundle = _bundle(spans=[(10.0, 40.0)], total=40.0)
    result = _measure({"bar_start": 1, "bar_end": 5}, bundle, _region(0.0, 10.0))
    assert result.d < 0


def test_region_wetter_than_piece_positive_d() -> None:
    # pedal only inside the first 10s region
    bundle = _bundle(spans=[(0.0, 10.0)], total=40.0)
    result = _measure({"bar_start": 1, "bar_end": 5}, bundle, _region(0.0, 10.0))
    assert result.d > 0


def test_region_zero_duration_raises_region_too_short() -> None:
    bundle = _bundle(spans=[(0.0, 20.0)], total=40.0)
    with pytest.raises(UnverifiableError) as exc:
        _measure({"bar_start": 5, "bar_end": 5}, bundle, _region(10.0, 10.0))
    assert exc.value.reason_code == "region_too_short"


def test_region_d_in_fraction_units() -> None:
    bundle = _bundle(spans=[(0.0, 10.0), (30.0, 40.0)], total=40.0)
    result = _measure({"bar_start": 1, "bar_end": 5}, bundle, _region(0.0, 10.0))
    assert -1.0 <= result.d <= 1.0


# --- substrate / robustness ---

def test_empty_bundle_substrate_failure() -> None:
    bundle = {"notes": [], "pedal_events": [], "substrate_versions": {"bundle_schema": "v1"}}
    with pytest.raises(UnverifiableError) as exc:
        _measure("whole_piece", bundle, _whole(40.0))
    assert exc.value.reason_code == "substrate_failure"


def test_error_bar_nonnegative() -> None:
    bundle = _bundle(spans=[(0.0, 10.0), (20.0, 30.0)], total=40.0)
    result = _measure("whole_piece", bundle, _whole(40.0))
    assert result.error_bar >= 0.0


def test_unterminated_pedal_closed_at_total_dur() -> None:
    # a lone 'on' with no matching 'off' is held to total_dur
    bundle = {
        "notes": [{"onset": i * 0.5, "offset": i * 0.5 + 0.4, "pitch": 60, "velocity": 80}
                  for i in range(80)],
        "pedal_events": [{"time": 0.0, "value": 127}],
        "substrate_versions": {"bundle_schema": "v1"},
    }
    result = _measure("whole_piece", bundle, _whole(40.0))
    # pedal down for the whole piece -> on_fraction ~ 1.0
    assert result.d == pytest.approx(1.0 - REFERENCE_FRACTION, abs=0.05)
