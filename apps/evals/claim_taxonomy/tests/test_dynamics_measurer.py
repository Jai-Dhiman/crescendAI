from __future__ import annotations
import numpy as np
import pytest
from claim_taxonomy.verifier.measurers.dynamics import (
    DynamicsMeasurer,
    REFERENCE_VELOCITY,
)
from claim_taxonomy.verifier.models import UnverifiableError
from claim_taxonomy.verifier.location_resolver import ResolvedRegion
from claim_taxonomy.verifier.substrate_error import SubstrateErrorEngine

SR = 16000

# Dynamics measures mean AMT note-velocity (perceived-loudness proxy), validated
# against PercePiano perceived dynamics at partial-rho 0.544 (n=180, #101 G-B).
# Units are MIDI velocity; the signed d feeds the frozen router. These tests pin the
# sign convention and the substrate (bundle `notes`, NOT librosa RMS).


def _make_region(start: float, end: float) -> ResolvedRegion:
    return ResolvedRegion(
        audio_start_sec=start,
        audio_end_sec=end,
        alignment_uncertainty_sec=0.05,
        location_span_bars=5.0,
    )


def _bundle(velocities, onsets=None) -> dict:
    """Bundle whose AMT `notes` carry the given per-note velocities (and onsets)."""
    if onsets is None:
        onsets = [0.1 * i for i in range(len(velocities))]
    notes = [
        {"onset": float(o), "offset": float(o) + 0.2, "pitch": 60, "velocity": int(v)}
        for o, v in zip(onsets, velocities)
    ]
    return {
        "notes": notes,
        "pedal_events": [],
        "measure_table": [{"bar_number": 1, "start_sec": 0.0, "start_tick": 0}],
        "substrate_versions": {"bundle_schema": "v1"},
    }


def test_whole_piece_loud_positive_d() -> None:
    # mean velocity well above the neutral reference -> d > 0 (loud/projected)
    bundle = _bundle([90] * 40)
    result = DynamicsMeasurer().measure(
        location="whole_piece", bundle=bundle,
        region=_make_region(0.0, 10.0), engine=SubstrateErrorEngine(seed=42),
    )
    assert result.d > 0, f"loud piece should have d>0, got {result.d}"
    assert result.d == pytest.approx(90.0 - REFERENCE_VELOCITY, abs=1e-6)


def test_whole_piece_soft_negative_d() -> None:
    # mean velocity well below the neutral reference -> d < 0 (soft/flat)
    bundle = _bundle([28] * 40)
    result = DynamicsMeasurer().measure(
        location="whole_piece", bundle=bundle,
        region=_make_region(0.0, 10.0), engine=SubstrateErrorEngine(seed=42),
    )
    assert result.d < 0, f"soft piece should have d<0, got {result.d}"


def test_whole_piece_signed_anomaly_is_reachable() -> None:
    # the OLD std/range statistic could never exceed tau=1.5dB; the velocity statistic
    # must be able to clear a velocity-unit tau in BOTH directions (non-degeneracy).
    tau = 8.0
    loud = DynamicsMeasurer().measure(
        location="whole_piece", bundle=_bundle([95] * 40),
        region=_make_region(0.0, 10.0), engine=SubstrateErrorEngine(seed=42),
    )
    soft = DynamicsMeasurer().measure(
        location="whole_piece", bundle=_bundle([25] * 40),
        region=_make_region(0.0, 10.0), engine=SubstrateErrorEngine(seed=42),
    )
    assert loud.d > tau, f"loud d should exceed +tau, got {loud.d}"
    assert soft.d < -tau, f"soft d should fall below -tau, got {soft.d}"


def test_region_louder_than_piece_positive_d() -> None:
    # first 20 notes (region) loud, rest soft -> region mean > piece mean -> d>0
    vel = [95] * 20 + [40] * 30
    onsets = [0.1 * i for i in range(20)] + [10.0 + 0.1 * i for i in range(30)]
    bundle = _bundle(vel, onsets)
    result = DynamicsMeasurer().measure(
        location={"bar_start": 1, "bar_end": 3}, bundle=bundle,
        region=_make_region(0.0, 5.0), engine=SubstrateErrorEngine(seed=42),
    )
    assert result.d > 0, f"loud region should have d>0, got {result.d}"


def test_region_softer_than_piece_negative_d() -> None:
    vel = [35] * 20 + [90] * 30
    onsets = [0.1 * i for i in range(20)] + [10.0 + 0.1 * i for i in range(30)]
    bundle = _bundle(vel, onsets)
    result = DynamicsMeasurer().measure(
        location={"bar_start": 1, "bar_end": 3}, bundle=bundle,
        region=_make_region(0.0, 5.0), engine=SubstrateErrorEngine(seed=42),
    )
    assert result.d < 0, f"soft region should have d<0, got {result.d}"


def test_whole_piece_too_few_notes_raises() -> None:
    with pytest.raises(UnverifiableError) as exc:
        DynamicsMeasurer().measure(
            location="whole_piece", bundle=_bundle([60] * 5),
            region=_make_region(0.0, 10.0), engine=SubstrateErrorEngine(seed=42),
        )
    assert exc.value.reason_code == "region_too_short"


def test_region_too_few_notes_raises() -> None:
    # 40 notes total but only a handful land in the region window
    vel = [60] * 40
    onsets = [0.05 * i for i in range(40)]  # all within first 2s
    bundle = _bundle(vel, onsets)
    with pytest.raises(UnverifiableError) as exc:
        DynamicsMeasurer().measure(
            location={"bar_start": 1, "bar_end": 3}, bundle=bundle,
            region=_make_region(5.0, 10.0), engine=SubstrateErrorEngine(seed=42),
        )
    assert exc.value.reason_code == "region_too_short"


def test_error_bar_positive_and_event_count() -> None:
    bundle = _bundle([70] * 40)
    result = DynamicsMeasurer().measure(
        location="whole_piece", bundle=bundle,
        region=_make_region(0.0, 10.0), engine=SubstrateErrorEngine(seed=42),
    )
    assert result.error_bar >= 0.0
    assert result.event_count == 40
