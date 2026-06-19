# apps/evals/claim_taxonomy/tests/test_verdict_dispatch.py
"""Round-trip tests for verdict_dispatch.route_verdict.

Control-flow dispatch only — no real measurements. The registry entries
supply synthetic deviation/error_bar values directly so the dispatch logic
can be verified independently of measurement substrate.
"""
from __future__ import annotations

import copy

import pytest

from claim_taxonomy.verdict_dispatch import route_verdict


# ---------------------------------------------------------------------------
# Registry fixtures
# ---------------------------------------------------------------------------

ACTIVE_DIM = {
    "status": "active",
    "reference": "established_tempo",
    "check": "signed_tempo_deviation",
    "tolerance": {
        "name": "signed_tempo_deviation",
        "provisional": 8.0,
        "unit": "percent",
        "calibration_source": "#65/M1 error-bar study",
        "locked": False,
    },
    "reliability_tier": 1,
    "measurement": "amt_onsets_region_tempo_fit",
    "minimum_events": 8,
}

GATED_DIM = {
    "status": "gated_on_measurement",
    "gate_id": "rms_lufs_estimator",
    "reference": "within_region_range",
    "check": "dynamic_range_zscore_and_contour_slope_sign",
}

SCOPED_DIM = {
    "status": "scoped_out",
    "rationale": "perceptual",
}

REGISTRY = {
    "timing": ACTIVE_DIM,
    "dynamics": GATED_DIM,
    "phrasing": SCOPED_DIM,
}

# A base claim that is fully specified for an active dimension
BASE_TIMING_CLAIM = {
    "proposition": "You rushed in bars 10-14",
    "dimension": "timing",
    "location": {"bar_start": 10, "bar_end": 14},
    "polarity": "-",
    "magnitude": None,
    # Measurement context fields supplied by the caller (the verifier, issue #65)
    # For this stub: synthesized to test dispatch branches
    "_measurement": {
        "d": -12.0,          # signed deviation (negative = rushed)
        "tau": 8.0,          # tolerance threshold
        "error_bar": 2.0,    # measurement error bar
        "event_count": 20,   # events in region
        "localizable": True,
    },
}


def _make_claim(**overrides) -> dict:
    claim = copy.deepcopy(BASE_TIMING_CLAIM)
    claim.update(overrides)
    return claim


def _make_measurement(**overrides) -> dict:
    m = copy.deepcopy(BASE_TIMING_CLAIM["_measurement"])
    m.update(overrides)
    return m


# ---------------------------------------------------------------------------
# Dispatch step 1: scoped_out -> UNVERIFIABLE(out_of_scope_dim)
# ---------------------------------------------------------------------------

def test_scoped_out_dimension_returns_unverifiable_out_of_scope() -> None:
    claim = _make_claim(dimension="phrasing", _measurement=_make_measurement())
    verdict, reason = route_verdict(claim, REGISTRY)
    assert verdict == "UNVERIFIABLE"
    assert reason == "out_of_scope_dim"


# ---------------------------------------------------------------------------
# Dispatch step 2: gated_on_measurement -> UNVERIFIABLE(gated_dim)
# ---------------------------------------------------------------------------

def test_gated_dimension_returns_unverifiable_gated_dim() -> None:
    claim = _make_claim(dimension="dynamics", _measurement=_make_measurement())
    verdict, reason = route_verdict(claim, REGISTRY)
    assert verdict == "UNVERIFIABLE"
    assert reason == "gated_dim"


# ---------------------------------------------------------------------------
# Dispatch step 3: not localizable -> UNVERIFIABLE(unlocalizable)
# ---------------------------------------------------------------------------

def test_unlocalizable_claim_returns_unverifiable() -> None:
    claim = _make_claim(
        _measurement=_make_measurement(localizable=False)
    )
    verdict, reason = route_verdict(claim, REGISTRY)
    assert verdict == "UNVERIFIABLE"
    assert reason == "unlocalizable"


def test_missing_localizable_raises_type_error() -> None:
    claim = _make_claim(
        _measurement={
            "d": -12.0, "tau": 8.0, "error_bar": 2.0, "event_count": 20,
            # no localizable
        }
    )
    with pytest.raises(TypeError, match="localizable"):
        route_verdict(claim, REGISTRY)


# ---------------------------------------------------------------------------
# Dispatch step 4: substrate_failure -> UNVERIFIABLE(substrate_failure)
# ---------------------------------------------------------------------------

def test_substrate_failure_returns_unverifiable() -> None:
    claim = _make_claim(
        _measurement=_make_measurement(substrate_failure=True)
    )
    verdict, reason = route_verdict(claim, REGISTRY)
    assert verdict == "UNVERIFIABLE"
    assert reason == "substrate_failure"


# ---------------------------------------------------------------------------
# Dispatch step 5: region_too_short -> UNVERIFIABLE(region_too_short)
# ---------------------------------------------------------------------------

def test_region_too_short_returns_unverifiable() -> None:
    claim = _make_claim(
        _measurement=_make_measurement(event_count=3)  # < minimum_events=8
    )
    verdict, reason = route_verdict(claim, REGISTRY)
    assert verdict == "UNVERIFIABLE"
    assert reason == "region_too_short"


# ---------------------------------------------------------------------------
# Dispatch step 7: near_threshold -> UNVERIFIABLE(near_threshold)
# ---------------------------------------------------------------------------

def test_near_threshold_returns_unverifiable() -> None:
    # d = -9.0, tau = 8.0, error_bar = 2.0 -> |abs(d) - tau| = 1.0 <= 2.0
    claim = _make_claim(
        _measurement=_make_measurement(d=-9.0, tau=8.0, error_bar=2.0)
    )
    verdict, reason = route_verdict(claim, REGISTRY)
    assert verdict == "UNVERIFIABLE"
    assert reason == "near_threshold"


# ---------------------------------------------------------------------------
# Dispatch step 8: d confirms polarity -> SUPPORTED
# ---------------------------------------------------------------------------

def test_claim_confirming_polarity_returns_supported() -> None:
    # polarity="-", d=-12.0, tau=8.0, error_bar=2.0 -> |abs(d) - tau| = 4.0 > 2.0
    # d is negative, polarity is negative -> SUPPORTED
    claim = _make_claim(
        _measurement=_make_measurement(d=-12.0, tau=8.0, error_bar=2.0)
    )
    verdict, reason = route_verdict(claim, REGISTRY)
    assert verdict == "SUPPORTED"
    assert reason is None


def test_neutral_polarity_with_no_anomaly_returns_supported() -> None:
    # polarity="neutral" asserts absence-of-problem
    # d near zero (no anomaly detected) -> SUPPORTED
    claim = _make_claim(polarity="neutral",
                        _measurement=_make_measurement(d=1.0, tau=8.0, error_bar=2.0))
    verdict, reason = route_verdict(claim, REGISTRY)
    assert verdict == "SUPPORTED"
    assert reason is None


# ---------------------------------------------------------------------------
# Dispatch step 9: d does NOT confirm polarity -> REFUTED
# ---------------------------------------------------------------------------

def test_wrong_direction_claim_returns_refuted() -> None:
    # polarity="-" (claims rushed) but d=+12.0 (actually dragged) -> REFUTED
    claim = _make_claim(
        polarity="-",
        _measurement=_make_measurement(d=12.0, tau=8.0, error_bar=2.0)
    )
    verdict, reason = route_verdict(claim, REGISTRY)
    assert verdict == "REFUTED"
    assert reason is None


def test_fabricated_anomaly_where_none_exists_returns_refuted() -> None:
    # polarity="-", d=-3.0 (within tolerance, no real rush), error_bar=1.0
    # |abs(d) - tau| = |3 - 8| = 5 > 1.0 but d does not confirm polarity (|d|=3 < tau=8)
    claim = _make_claim(
        polarity="-",
        _measurement=_make_measurement(d=-3.0, tau=8.0, error_bar=1.0)
    )
    verdict, reason = route_verdict(claim, REGISTRY)
    assert verdict == "REFUTED"
    assert reason is None


# ---------------------------------------------------------------------------
# Error handling: invalid polarity raises TypeError
# ---------------------------------------------------------------------------

def test_invalid_polarity_raises_type_error() -> None:
    claim = _make_claim(polarity="bad_value")
    with pytest.raises(TypeError, match="polarity"):
        route_verdict(claim, REGISTRY)


# ---------------------------------------------------------------------------
# Error handling: unknown dimension raises TypeError
# ---------------------------------------------------------------------------

def test_unknown_dimension_raises_type_error() -> None:
    claim = _make_claim(dimension="nonexistent_dim")
    with pytest.raises(TypeError, match="Unknown dimension"):
        route_verdict(claim, REGISTRY)


def test_missing_measurement_context_raises_type_error() -> None:
    claim = {
        "proposition": "test",
        "dimension": "timing",
        "location": {"bar_start": 1, "bar_end": 4},
        "polarity": "-",
        "magnitude": None,
        # no _measurement key
    }
    with pytest.raises(TypeError, match="_measurement"):
        route_verdict(claim, REGISTRY)
