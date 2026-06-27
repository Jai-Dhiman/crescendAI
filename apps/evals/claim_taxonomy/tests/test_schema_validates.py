# apps/evals/claim_taxonomy/tests/test_schema_validates.py
"""Verify that claim_taxonomy.schema.json is self-consistent and that a
minimal taxonomy skeleton validates against it without errors."""
from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

TAXONOMY_DIR = Path(__file__).resolve().parents[1]
SCHEMA_PATH = TAXONOMY_DIR / "claim_taxonomy.schema.json"


def _load_schema() -> dict:
    with open(SCHEMA_PATH) as f:
        return json.load(f)


def test_schema_file_exists() -> None:
    assert SCHEMA_PATH.exists(), f"Schema not found at {SCHEMA_PATH}"


def test_schema_has_required_top_level_keys() -> None:
    schema = _load_schema()
    for key in ("$schema", "$id", "type", "required", "properties"):
        assert key in schema, f"Schema missing top-level key: {key}"


def test_schema_draft_is_2020_12() -> None:
    schema = _load_schema()
    assert "2020-12" in schema.get("$schema", ""), (
        "Schema must use JSON Schema draft 2020-12"
    )


def test_minimal_valid_taxonomy_skeleton_passes_schema() -> None:
    """A skeleton with all required top-level fields passes validation."""
    schema = _load_schema()
    skeleton = {
        "taxonomy_version": "v0",
        "extractor_judge_boundary": {
            "description": "test",
            "llm_in_truth_label": False,
        },
        "claim_schema": {
            "type": "object",
            "required": ["proposition", "dimension", "location", "polarity"],
            "properties": {
                "proposition": {"type": "string"},
                "dimension": {"type": "string"},
                "location": {},
                "polarity": {"type": "string"},
                "magnitude": {},
            },
        },
        "dimensions": {
            "timing": {
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
            },
            "phrasing": {
                "status": "scoped_out",
                "rationale": "perceptual",
            },
        },
        "verdict_spec": {
            "labels": ["SUPPORTED", "REFUTED", "UNVERIFIABLE"],
            "unverifiable_reason_codes": [
                "out_of_scope_dim",
                "gated_dim",
                "unlocalizable",
                "substrate_failure",
                "region_too_short",
                "near_threshold",
            ],
            "faithfulness_formula": "SUPPORTED / (SUPPORTED + REFUTED)",
            "coverage_formula": "(SUPPORTED + REFUTED) / total_claims",
            "unverifiable_reporting": "histogram_by_reason_code",
        },
    }
    # Must not raise
    jsonschema.validate(instance=skeleton, schema=schema)


def test_llm_in_truth_label_true_fails_schema() -> None:
    """Non-circularity invariant: llm_in_truth_label=true must be rejected.

    The verifier's truth label may never invoke an LLM. The schema encodes
    this with const:false; if that constraint were removed, this test fails.
    """
    schema = _load_schema()
    bad = {
        "taxonomy_version": "v0",
        "extractor_judge_boundary": {
            "description": "x",
            "llm_in_truth_label": True,
        },
        "claim_schema": {},
        "dimensions": {},
        "verdict_spec": {
            "labels": [],
            "unverifiable_reason_codes": [],
            "faithfulness_formula": "x",
            "coverage_formula": "x",
            "unverifiable_reporting": "x",
        },
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=bad, schema=schema)


def test_taxonomy_missing_required_field_fails_schema() -> None:
    """A skeleton missing taxonomy_version must fail schema validation."""
    schema = _load_schema()
    bad = {
        # missing taxonomy_version
        "extractor_judge_boundary": {"description": "x", "llm_in_truth_label": False},
        "claim_schema": {},
        "dimensions": {},
        "verdict_spec": {},
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=bad, schema=schema)


TAXONOMY_PATH = TAXONOMY_DIR / "claim_taxonomy.json"


def test_claim_taxonomy_json_exists() -> None:
    assert TAXONOMY_PATH.exists(), f"Taxonomy not found at {TAXONOMY_PATH}"


def test_claim_taxonomy_json_validates_against_schema() -> None:
    """Full claim_taxonomy.json must pass JSON Schema validation."""
    schema = _load_schema()
    with open(TAXONOMY_PATH) as f:
        taxonomy = json.load(f)
    # Must not raise
    jsonschema.validate(instance=taxonomy, schema=schema)


def test_all_seven_dimensions_present() -> None:
    with open(TAXONOMY_PATH) as f:
        taxonomy = json.load(f)
    expected = {
        "timing", "pedaling", "dynamics", "articulation",
        "phrasing", "interpretation", "timbre"
    }
    actual = set(taxonomy["dimensions"].keys())
    assert actual == expected, f"Missing: {expected - actual}, Extra: {actual - expected}"


def test_active_dimensions_have_complete_registry() -> None:
    with open(TAXONOMY_PATH) as f:
        taxonomy = json.load(f)
    required_active_fields = {
        "status", "reference", "check", "tolerance",
        "reliability_tier", "measurement", "minimum_events"
    }
    for dim_name, dim in taxonomy["dimensions"].items():
        if dim["status"] == "active":
            missing = required_active_fields - set(dim.keys())
            assert not missing, (
                f"Active dimension '{dim_name}' missing fields: {missing}"
            )


def test_all_tolerances_cite_calibration() -> None:
    # #101 front-4: tolerances may now be locked (calibrated against human anomaly
    # labels) OR provisional (locked=false); either way the source must be documented.
    # A LOCKED tolerance must cite its calibration in calibration_source.
    with open(TAXONOMY_PATH) as f:
        taxonomy = json.load(f)
    for dim_name, dim in taxonomy["dimensions"].items():
        if dim.get("status") == "active":
            tol = dim["tolerance"]
            assert isinstance(tol["locked"], bool), (
                f"Dimension '{dim_name}': tolerance.locked must be a bool"
            )
            assert isinstance(tol["calibration_source"], str) and "#" in tol["calibration_source"], (
                f"Dimension '{dim_name}': calibration_source must cite an issue "
                f"(provisional tolerances stay documented)"
            )


def test_extractor_judge_boundary_llm_flag_is_false() -> None:
    with open(TAXONOMY_PATH) as f:
        taxonomy = json.load(f)
    assert taxonomy["extractor_judge_boundary"]["llm_in_truth_label"] is False
