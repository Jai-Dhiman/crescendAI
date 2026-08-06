from __future__ import annotations

import pytest
from claim_taxonomy.verifier.location_resolver import ResolvedRegion
from claim_taxonomy.verifier.measurers.articulation import (
    IOI_FLOOR_SEC,
    MINIMUM_PAIRS,
    REFERENCE_RATIO,
    ArticulationMeasurer,
)
from claim_taxonomy.verifier.models import UnverifiableError
from claim_taxonomy.verifier.substrate_error import SubstrateErrorEngine

# Articulation measures median(note_duration / IOI) from Transkun note OFFSETS,
# validated on real MAESTRO audio against ground-truth MIDI at corr 0.930 (#101 FRONT
# 10). These tests pin the sign convention, the IOI floor that makes the statistic
# conditionable at all, and the abstentions.


def _make_region(start: float, end: float) -> ResolvedRegion:
    return ResolvedRegion(
        audio_start_sec=start,
        audio_end_sec=end,
        alignment_uncertainty_sec=0.05,
        location_span_bars=5.0,
    )


def _bundle(ratio: float, n: int = 40, ioi: float = 0.2, start: float = 0.0) -> dict:
    """n notes at a fixed IOI, each held for `ratio` x IOI, so the statistic
    is exactly `ratio`.
    """
    return _bundle_from_notes(_notes(ratio, n, ioi, start))


def _notes(ratio: float, n: int, ioi: float, start: float) -> list[dict]:
    return [
        {
            "onset": start + i * ioi,
            "offset": start + i * ioi + ratio * ioi,
            "pitch": 60,
            "velocity": 64,
        }
        for i in range(n)
    ]


def _bundle_from_notes(notes: list[dict]) -> dict:
    return {
        "notes": notes,
        "pedal_events": [],
        "measure_table": [{"bar_number": 1, "start_sec": 0.0, "start_tick": 0}],
        "substrate_versions": {"bundle_schema": "v1"},
    }


def _measure(bundle: dict, location="whole_piece", region=None):
    return ArticulationMeasurer().measure(
        location=location,
        bundle=bundle,
        region=region or _make_region(0.0, 100.0),
        engine=SubstrateErrorEngine(seed=42),
    )


def test_whole_piece_legato_positive_d() -> None:
    # notes held past their successor's onset -> overlapping -> more legato than neutral
    result = _measure(_bundle(ratio=1.6))
    assert result.d > 0, f"legato should have d>0, got {result.d}"
    assert result.d == pytest.approx(1.6 - REFERENCE_RATIO, abs=1e-6)


def test_whole_piece_detached_negative_d() -> None:
    # notes released well before the next onset -> staccato
    result = _measure(_bundle(ratio=0.3))
    assert result.d < 0, f"detached should have d<0, got {result.d}"
    assert result.d == pytest.approx(0.3 - REFERENCE_RATIO, abs=1e-6)


def test_both_directions_clear_the_shipped_tau() -> None:
    # non-degeneracy: the statistic must be able to exceed tau in BOTH directions, which
    # the unfloored FRONT 9 variant could not do reliably (its noise p90 was 6x larger
    # than tau).
    tau = 0.163
    assert _measure(_bundle(ratio=1.8)).d > tau
    assert _measure(_bundle(ratio=0.2)).d < -tau


def test_chord_notes_are_excluded_by_the_ioi_floor() -> None:
    """The floor is the whole reason this statistic is conditionable.

    A near-simultaneous chord note has IOI -> 0, so its duration/IOI ratio explodes.
    Without the floor a single 3-note chord drags the window median arbitrarily far;
    with it, the chord notes are dropped and the melodic notes decide the statistic. """
    melodic = _notes(ratio=0.5, n=MINIMUM_PAIRS + 5, ioi=0.2, start=0.0)
    # a dense chord: 10 notes 2ms apart, each held 0.5s -> unfloored ratios of ~250
    chord = [
        {
            "onset": 100.0 + 0.002 * i,
            "offset": 100.5 + 0.002 * i,
            "pitch": 60 + i,
            "velocity": 64,
        }
        for i in range(10)
    ]
    with_chord = _measure(_bundle_from_notes(melodic + chord))
    without_chord = _measure(_bundle_from_notes(melodic))
    assert with_chord.d == pytest.approx(without_chord.d, abs=1e-9)
    assert with_chord.d == pytest.approx(0.5 - REFERENCE_RATIO, abs=1e-6)


def test_notes_below_the_ioi_floor_do_not_count_as_events() -> None:
    below = IOI_FLOOR_SEC / 2
    bundle = _bundle_from_notes(
        _notes(ratio=0.5, n=MINIMUM_PAIRS + 20, ioi=below, start=0.0)
    )
    with pytest.raises(UnverifiableError) as exc:
        _measure(bundle)
    assert exc.value.reason_code == "region_too_short"


def test_region_more_legato_than_piece_positive_d() -> None:
    # region 0-5s is legato, the rest detached -> region ratio > whole-clip ratio
    notes = _notes(ratio=1.8, n=25, ioi=0.2, start=0.0) + _notes(
        ratio=0.3, n=40, ioi=0.2, start=20.0
    )
    result = _measure(
        _bundle_from_notes(notes),
        location={"bar_start": 1, "bar_end": 3},
        region=_make_region(0.0, 5.0),
    )
    assert result.d > 0, f"legato region should have d>0, got {result.d}"


def test_region_more_detached_than_piece_negative_d() -> None:
    notes = _notes(ratio=0.3, n=25, ioi=0.2, start=0.0) + _notes(
        ratio=1.8, n=40, ioi=0.2, start=20.0
    )
    result = _measure(
        _bundle_from_notes(notes),
        location={"bar_start": 1, "bar_end": 3},
        region=_make_region(0.0, 5.0),
    )
    assert result.d < 0, f"detached region should have d<0, got {result.d}"


def test_whole_piece_too_few_notes_raises() -> None:
    with pytest.raises(UnverifiableError) as exc:
        _measure(_bundle(ratio=1.0, n=5))
    assert exc.value.reason_code == "region_too_short"


def test_single_bar_is_too_fragile_to_measure() -> None:
    """The taxonomy's 'single-bar articulation claims are too fragile' rule, enforced by
    minimum_events rather than by prose: a one-bar region seldom holds 20 floored
    pairs."""
    notes = _notes(ratio=1.0, n=60, ioi=0.2, start=0.0)
    with pytest.raises(UnverifiableError) as exc:
        _measure(
            _bundle_from_notes(notes),
            location={"bar_start": 1, "bar_end": 1},
            region=_make_region(0.0, 2.0),
        )
    assert exc.value.reason_code == "region_too_short"


def test_error_bar_positive_and_event_count() -> None:
    result = _measure(_bundle(ratio=1.0, n=40))
    assert result.error_bar > 0.0
    # 40 notes yield 39 IOIs, all above the floor
    assert result.event_count == 39


def test_error_bar_never_drops_below_the_measured_substrate_floor() -> None:
    """The correlated floor is what keeps the near-threshold dead-band honest:
    Transkun's release bias is per-performance, so statistic error does NOT vanish as
    the note count grows."""
    from claim_taxonomy.verifier.measurers.articulation import SUBSTRATE_STATISTIC_FLOOR

    many = _measure(_bundle(ratio=1.0, n=400))
    assert many.error_bar >= SUBSTRATE_STATISTIC_FLOOR
