# apps/evals/claim_taxonomy/tests/test_verifier_models.py
from __future__ import annotations
import pytest
from claim_taxonomy.verifier.models import VerdictResult, UnverifiableError


def test_verdict_result_fields() -> None:
    r = VerdictResult(
        verdict="SUPPORTED",
        reason_code=None,
        measured_value=-12.0,
        tau=8.0,
        error_bar=1.5,
        event_count=20,
        units="percent",
        substrate_versions={"bundle_schema": "v1"},
        dimension="timing",
        location={"bar_start": 1, "bar_end": 4},
    )
    assert r.verdict == "SUPPORTED"
    assert r.reason_code is None
    assert r.measured_value == -12.0
    assert r.units == "percent"


def test_verdict_result_unverifiable_has_reason_code() -> None:
    r = VerdictResult(
        verdict="UNVERIFIABLE",
        reason_code="near_threshold",
        measured_value=-9.0,
        tau=8.0,
        error_bar=2.0,
        event_count=15,
        units="percent",
        substrate_versions={},
        dimension="timing",
        location="whole_piece",
    )
    assert r.reason_code == "near_threshold"


def test_unverifiable_error_has_reason_code_and_detail() -> None:
    err = UnverifiableError("unlocalizable", "bar 3 is within alignment uncertainty")
    assert err.reason_code == "unlocalizable"
    assert err.detail == "bar 3 is within alignment uncertainty"
    assert isinstance(err, Exception)


def test_unverifiable_error_reason_codes_are_strings() -> None:
    for code in ("out_of_scope_dim", "gated_dim", "unlocalizable",
                 "substrate_failure", "region_too_short", "near_threshold"):
        err = UnverifiableError(code, "test")
        assert err.reason_code == code
