# apps/evals/claim_taxonomy/tests/test_audit_integrity.py
"""Structural integrity test for the hand-decomposed baseline audit.

This test does not validate the correctness of the manual decomposition
decisions -- it verifies that the audit file has the required structure,
that all sample_claims have required fields, and that the scoped_out_fraction
is a valid fraction.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

AUDIT_PATH = (
    Path(__file__).resolve().parents[1] / "audit" / "baseline_v1_audit.json"
)

REQUIRED_CLAIM_FIELDS = {"proposition", "dimension", "location", "polarity"}
VALID_DIMENSIONS = {
    "timing", "pedaling", "dynamics", "articulation",
    "phrasing", "interpretation", "timbre"
}
VALID_POLARITIES = {"+", "-", "neutral"}


def _load_audit() -> dict:
    with open(AUDIT_PATH) as f:
        return json.load(f)


def test_audit_file_exists() -> None:
    assert AUDIT_PATH.exists(), f"Audit file not found at {AUDIT_PATH}"


def test_audit_has_required_top_level_keys() -> None:
    audit = _load_audit()
    required = {
        "total_claims",
        "dimension_distribution",
        "polarity_distribution",
        "location_distribution",
        "scoped_out_fraction",
        "sample_claims",
        "methodology",
    }
    missing = required - set(audit.keys())
    assert not missing, f"Audit missing keys: {missing}"


def test_scoped_out_fraction_is_valid() -> None:
    audit = _load_audit()
    f = audit["scoped_out_fraction"]
    assert isinstance(f, (int, float)), "scoped_out_fraction must be a number"
    assert 0.0 <= f <= 1.0, f"scoped_out_fraction must be in [0, 1]; got {f}"


def test_sample_claims_count_matches_total() -> None:
    audit = _load_audit()
    assert len(audit["sample_claims"]) == audit["total_claims"], (
        f"total_claims={audit['total_claims']} but "
        f"len(sample_claims)={len(audit['sample_claims'])}"
    )


def test_sample_claims_have_required_fields() -> None:
    audit = _load_audit()
    for i, claim in enumerate(audit["sample_claims"]):
        missing = REQUIRED_CLAIM_FIELDS - set(claim.keys())
        assert not missing, f"claim[{i}] missing fields: {missing}"


def test_sample_claims_have_valid_dimensions() -> None:
    audit = _load_audit()
    for i, claim in enumerate(audit["sample_claims"]):
        assert claim["dimension"] in VALID_DIMENSIONS, (
            f"claim[{i}] has unknown dimension: {claim['dimension']!r}"
        )


def test_sample_claims_have_valid_polarities() -> None:
    audit = _load_audit()
    for i, claim in enumerate(audit["sample_claims"]):
        assert claim["polarity"] in VALID_POLARITIES, (
            f"claim[{i}] has invalid polarity: {claim['polarity']!r}"
        )


def test_at_least_30_sample_claims() -> None:
    audit = _load_audit()
    assert audit["total_claims"] >= 30, (
        f"Expected >= 30 sample claims; got {audit['total_claims']}"
    )


def test_dimension_distribution_sums_to_total() -> None:
    audit = _load_audit()
    dist = audit["dimension_distribution"]
    total = sum(dist.values())
    assert total == audit["total_claims"], (
        f"dimension_distribution sums to {total} but total_claims={audit['total_claims']}"
    )


def test_scoped_out_fraction_matches_distribution() -> None:
    """scoped_out_fraction must match the count of scoped-out dimensions
    in dimension_distribution."""
    audit = _load_audit()
    dist = audit["dimension_distribution"]
    scoped_out_dims = {"phrasing", "interpretation", "timbre"}
    scoped_count = sum(dist.get(d, 0) for d in scoped_out_dims)
    expected_fraction = scoped_count / audit["total_claims"]
    assert abs(audit["scoped_out_fraction"] - expected_fraction) < 0.001, (
        f"scoped_out_fraction={audit['scoped_out_fraction']} does not match "
        f"computed {expected_fraction} from dimension_distribution"
    )


def test_header_aggregates_match_sample_claims_array() -> None:
    """Hard constraint: header aggregates must be derivable from the array,
    never trusted as hand-written values. Recompute every aggregate from
    sample_claims and assert the header matches."""
    from collections import Counter

    audit = _load_audit()
    claims = audit["sample_claims"]

    assert audit["total_claims"] == len(claims)

    dim = Counter(c["dimension"] for c in claims)
    for dimension, count in audit["dimension_distribution"].items():
        assert dim.get(dimension, 0) == count, (
            f"dimension_distribution[{dimension}]={count} but array has {dim.get(dimension, 0)}"
        )

    pol = Counter(c["polarity"] for c in claims)
    for polarity, count in audit["polarity_distribution"].items():
        assert pol.get(polarity, 0) == count, (
            f"polarity_distribution[{polarity}]={count} but array has {pol.get(polarity, 0)}"
        )

    scoped_out_dims = {"phrasing", "interpretation", "timbre"}
    scoped_count = sum(dim.get(d, 0) for d in scoped_out_dims)
    expected_fraction = scoped_count / len(claims)
    assert abs(audit["scoped_out_fraction"] - expected_fraction) < 0.001, (
        f"scoped_out_fraction={audit['scoped_out_fraction']} but array gives {expected_fraction}"
    )

    def _loc_bucket(loc) -> str:
        if loc == "whole_piece":
            return "whole_piece"
        if isinstance(loc, dict) and loc.get("bar_end") == loc.get("bar_start"):
            return "bar"
        return "region"

    loc = Counter(_loc_bucket(c["location"]) for c in claims)
    for bucket, count in audit["location_distribution"].items():
        assert loc.get(bucket, 0) == count, (
            f"location_distribution[{bucket}]={count} but array has {loc.get(bucket, 0)}"
        )
