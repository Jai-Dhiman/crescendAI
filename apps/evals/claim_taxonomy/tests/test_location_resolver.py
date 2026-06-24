from __future__ import annotations
import math
import numpy as np
import pytest
from claim_taxonomy.verifier.location_resolver import LocationResolver, ResolvedRegion
from claim_taxonomy.verifier.models import UnverifiableError
from claim_taxonomy.verifier.substrate_error import SubstrateErrorEngine


def _make_bundle(n_bars: int = 10, bar_dur: float = 2.0) -> dict:
    """Synthetic bundle: n_bars of bar_dur seconds each, identity alignment."""
    measure_table = [
        {"bar_number": i + 1, "start_sec": i * bar_dur, "start_tick": i * 480}
        for i in range(n_bars)
    ]
    t = np.linspace(0.0, n_bars * bar_dur, 200)
    return {
        "measure_table": measure_table,
        "anchors": {
            "perf_audio_sec": t.tolist(),
            "score_audio_sec": t.tolist(),
        },
    }


def _low_coverage_bundle(n_bars: int = 10, bar_dur: float = 2.0, frac: float = 0.5) -> dict:
    """measure_table spans n_bars but anchors cover only `frac` of the score-time span."""
    measure_table = [
        {"bar_number": i + 1, "start_sec": i * bar_dur, "start_tick": i * 480}
        for i in range(n_bars)
    ]
    span = (n_bars - 1) * bar_dur
    t = np.linspace(0.0, frac * span, 100)  # anchors cover only the first `frac`
    return {
        "measure_table": measure_table,
        "anchors": {"perf_audio_sec": t.tolist(), "score_audio_sec": t.tolist()},
    }


def test_low_coverage_clip_gates_bar_claims() -> None:
    # Coverage 0.5 < threshold 0.9 -> even an in-span bar abstains as low_coverage.
    bundle = _low_coverage_bundle(frac=0.5)
    resolver = LocationResolver(bundle, SubstrateErrorEngine(seed=0), min_coverage=0.9)
    with pytest.raises(UnverifiableError) as exc:
        resolver.resolve({"bar_start": 2, "bar_end": 2})  # score 2s, inside the 0-9s anchors
    assert exc.value.reason_code == "low_coverage"


def test_high_coverage_clip_admits_bar_claims() -> None:
    bundle = _make_bundle()  # full coverage (1.0)
    resolver = LocationResolver(bundle, SubstrateErrorEngine(seed=0), min_coverage=0.9)
    region = resolver.resolve({"bar_start": 2, "bar_end": 4})
    assert region.audio_start_sec >= 0.0


def test_whole_piece_not_gated_by_coverage() -> None:
    bundle = _low_coverage_bundle(frac=0.3)
    resolver = LocationResolver(bundle, SubstrateErrorEngine(seed=0), min_coverage=0.9)
    region = resolver.resolve("whole_piece")  # must not raise
    assert region.location_span_bars == math.inf


def test_default_min_coverage_does_not_gate() -> None:
    # Backward compatible: default min_coverage=0 -> low-coverage bars still resolve.
    bundle = _low_coverage_bundle(frac=0.5)
    resolver = LocationResolver(bundle, SubstrateErrorEngine(seed=0))
    region = resolver.resolve({"bar_start": 2, "bar_end": 2})
    assert region.audio_start_sec >= 0.0


def test_bar_range_resolves_to_correct_audio_times() -> None:
    bundle = _make_bundle(n_bars=10, bar_dur=2.0)
    engine = SubstrateErrorEngine(seed=0)
    resolver = LocationResolver(bundle, engine)
    region = resolver.resolve({"bar_start": 3, "bar_end": 5})
    assert abs(region.audio_start_sec - 4.0) < 0.05
    assert abs(region.audio_end_sec - 10.0) < 0.05


def test_whole_piece_always_localizable() -> None:
    bundle = _make_bundle()
    engine = SubstrateErrorEngine(seed=0)
    resolver = LocationResolver(bundle, engine)
    region = resolver.resolve("whole_piece")
    assert region.location_span_bars == math.inf
    assert region.audio_start_sec >= 0.0


def test_missing_bar_raises_unlocalizable() -> None:
    bundle = _make_bundle(n_bars=5)
    engine = SubstrateErrorEngine(seed=0)
    resolver = LocationResolver(bundle, engine)
    with pytest.raises(UnverifiableError) as exc_info:
        resolver.resolve({"bar_start": 10, "bar_end": 12})
    assert exc_info.value.reason_code == "unlocalizable"


def test_too_few_anchors_raises_unlocalizable() -> None:
    bundle = {
        "measure_table": [
            {"bar_number": 1, "start_sec": 0.0, "start_tick": 0},
            {"bar_number": 2, "start_sec": 2.0, "start_tick": 480},
        ],
        "anchors": {"perf_audio_sec": [0.0], "score_audio_sec": [0.0]},
    }
    engine = SubstrateErrorEngine(seed=0)
    resolver = LocationResolver(bundle, engine)
    with pytest.raises(UnverifiableError) as exc_info:
        resolver.resolve({"bar_start": 1, "bar_end": 1})
    assert exc_info.value.reason_code == "unlocalizable"


def test_failsafe_triggers_when_uncertainty_exceeds_span() -> None:
    """A single-bar region finer than the alignment uncertainty must become unlocalizable.

    The SubstrateErrorEngine yields ~9.6ms alignment uncertainty (jitter sigma) for an
    identity alignment. A 4ms bar is finer than that uncertainty, so the failsafe
    (uncertainty_bars >= location_span_bars) must raise unlocalizable: the claim cannot
    be pinned to a bar narrower than the alignment precision.
    """
    bundle = {
        "measure_table": [
            {"bar_number": 1, "start_sec": 0.0, "start_tick": 0},
            {"bar_number": 2, "start_sec": 0.004, "start_tick": 480},  # 4ms bar < ~9.6ms uncertainty
        ],
        "anchors": {
            "perf_audio_sec": [0.0, 5.0],  # sparse identity alignment
            "score_audio_sec": [0.0, 5.0],
        },
    }
    engine = SubstrateErrorEngine(seed=42, n_samples=500)
    resolver = LocationResolver(bundle, engine)
    with pytest.raises(UnverifiableError) as exc_info:
        resolver.resolve({"bar_start": 1, "bar_end": 1})
    assert exc_info.value.reason_code == "unlocalizable"


def test_single_entry_measure_table_region_raises_substrate_failure() -> None:
    """A single-bar measure_table cannot infer bar duration; must raise substrate_failure."""
    bundle = {
        "measure_table": [{"bar_number": 1, "start_sec": 0.0, "start_tick": 0}],
        "anchors": {
            "perf_audio_sec": [0.0, 5.0],
            "score_audio_sec": [0.0, 5.0],
        },
    }
    engine = SubstrateErrorEngine(seed=0)
    resolver = LocationResolver(bundle, engine)
    with pytest.raises(UnverifiableError) as exc_info:
        resolver.resolve({"bar_start": 1, "bar_end": 1})
    assert exc_info.value.reason_code == "substrate_failure"


def test_whole_piece_single_entry_measure_table_does_not_call_bar_duration() -> None:
    """whole_piece resolution must NOT route through _bar_duration_sec."""
    bundle = {
        "measure_table": [{"bar_number": 1, "start_sec": 0.0, "start_tick": 0}],
        "anchors": {
            "perf_audio_sec": [0.0, 5.0],
            "score_audio_sec": [0.0, 5.0],
        },
    }
    engine = SubstrateErrorEngine(seed=0)
    resolver = LocationResolver(bundle, engine)
    # Must NOT raise; whole_piece does not need bar duration
    region = resolver.resolve("whole_piece")
    assert region.location_span_bars == float("inf")


def test_bar_outside_anchor_span_raises_unlocalizable() -> None:
    """A bar whose score-time lies beyond the matched-anchor span must abstain,
    not silently extrapolate (np.interp clamps to the last anchor -> a confident,
    wrong, and run-to-run-unstable audio time). Surfaced by GATE 1."""
    bundle = {
        "measure_table": [
            {"bar_number": i + 1, "start_sec": i * 2.0, "start_tick": i * 480}
            for i in range(12)  # bars 1..12, score starts 0..22s
        ],
        "anchors": {  # matched anchors only cover score-time 0..12s
            "perf_audio_sec": np.linspace(0.0, 12.0, 50).tolist(),
            "score_audio_sec": np.linspace(0.0, 12.0, 50).tolist(),
        },
    }
    engine = SubstrateErrorEngine(seed=0)
    resolver = LocationResolver(bundle, engine)
    # bar 9 starts at 16s, past the 12s anchor span -> extrapolation -> abstain.
    with pytest.raises(UnverifiableError) as exc_info:
        resolver.resolve({"bar_start": 9, "bar_end": 9})
    assert exc_info.value.reason_code == "unlocalizable"
    # A bar within the anchor span still resolves.
    region = resolver.resolve({"bar_start": 3, "bar_end": 3})
    assert region.audio_start_sec == pytest.approx(4.0, abs=0.1)


def test_resolved_region_has_uncertainty_field() -> None:
    bundle = _make_bundle()
    engine = SubstrateErrorEngine(seed=0)
    resolver = LocationResolver(bundle, engine)
    region = resolver.resolve({"bar_start": 2, "bar_end": 4})
    assert isinstance(region.alignment_uncertainty_sec, float)
    assert region.alignment_uncertainty_sec >= 0.0
    assert region.location_span_bars == 3  # bar 2, 3, 4
