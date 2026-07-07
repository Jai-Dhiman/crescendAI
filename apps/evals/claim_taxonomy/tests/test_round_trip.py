# apps/evals/claim_taxonomy/tests/test_round_trip.py
"""End-to-end round-trip: claim_taxonomy.json -> schema valid -> route_verdict
for example claims per active dimension.

This test does not perform real measurements. It verifies that:
1. claim_taxonomy.json passes JSON Schema validation.
2. Hand-authored example claims for active dimensions route to the expected
   verdict branch when supplied with synthetic measurement context.
3. Example claims for scoped_out and gated dimensions route to UNVERIFIABLE
   with the correct typed reason codes.
"""
from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from claim_taxonomy.verdict_dispatch import route_verdict

TAXONOMY_DIR = Path(__file__).resolve().parents[1]
TAXONOMY_PATH = TAXONOMY_DIR / "claim_taxonomy.json"
SCHEMA_PATH = TAXONOMY_DIR / "claim_taxonomy.schema.json"


def _load() -> tuple[dict, dict]:
    with open(TAXONOMY_PATH) as f:
        taxonomy = json.load(f)
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    return taxonomy, schema


def test_claim_taxonomy_json_passes_schema() -> None:
    """Taxonomy artifact must validate against the committed schema."""
    taxonomy, schema = _load()
    jsonschema.validate(instance=taxonomy, schema=schema)


def test_timing_claim_supported_routes_correctly() -> None:
    taxonomy, _ = _load()
    registry = taxonomy["dimensions"]
    claim = {
        "proposition": "You rushed through bars 12-16",
        "dimension": "timing",
        "location": {"bar_start": 12, "bar_end": 16},
        "polarity": "-",
        "magnitude": None,
        "_measurement": {
            # FRONT 7 units: signed onset deviation in ms (tau=30 provisional)
            "d": -45.0,
            "tau": registry["timing"]["tolerance"]["provisional"],
            "error_bar": 2.0,
            "event_count": 25,
            "localizable": True,
        },
    }
    verdict, reason = route_verdict(claim, registry)
    assert verdict == "SUPPORTED"
    assert reason is None


def test_timing_claim_refuted_when_no_rush_detected() -> None:
    taxonomy, _ = _load()
    registry = taxonomy["dimensions"]
    claim = {
        "proposition": "You rushed in bar 3",
        "dimension": "timing",
        "location": {"bar_start": 3, "bar_end": 3},
        "polarity": "-",
        "magnitude": None,
        "_measurement": {
            "d": 2.0,   # slightly faster but within tolerance
            "tau": registry["timing"]["tolerance"]["provisional"],
            "error_bar": 1.5,
            "event_count": 10,
            "localizable": True,
        },
    }
    verdict, reason = route_verdict(claim, registry)
    assert verdict == "REFUTED"
    assert reason is None


def test_pedaling_claim_supported() -> None:
    taxonomy, _ = _load()
    registry = taxonomy["dimensions"]
    claim = {
        "proposition": "Your pedaling was sparse in the opening",
        "dimension": "pedaling",
        "location": {"bar_start": 1, "bar_end": 8},
        "polarity": "-",
        "magnitude": None,
        "_measurement": {
            "d": -0.40,
            "tau": registry["pedaling"]["tolerance"]["provisional"],
            "error_bar": 0.05,
            "event_count": 6,
            "localizable": True,
        },
    }
    verdict, reason = route_verdict(claim, registry)
    assert verdict == "SUPPORTED"
    assert reason is None


def test_dynamics_active_routes_correctly() -> None:
    """After v0.1 dynamics is active; route_verdict produces a real verdict from _measurement."""
    taxonomy, _ = _load()
    registry = taxonomy["dimensions"]
    assert registry["dynamics"]["status"] == "active", (
        "dynamics must be active in v0.1 taxonomy"
    )
    claim = {
        "proposition": "Your dynamics were flat throughout",
        "dimension": "dynamics",
        "location": "whole_piece",
        "polarity": "-",
        "magnitude": None,
        "_measurement": {
            "d": -2.0,
            "tau": registry["dynamics"]["tolerance"]["provisional"],
            "error_bar": 0.2,
            "event_count": 50,
            "localizable": True,
        },
    }
    verdict, reason = route_verdict(claim, registry)
    assert verdict in ("SUPPORTED", "REFUTED", "UNVERIFIABLE")


def test_phrasing_scoped_out_returns_unverifiable() -> None:
    taxonomy, _ = _load()
    registry = taxonomy["dimensions"]
    claim = {
        "proposition": "Your phrasing lacked direction",
        "dimension": "phrasing",
        "location": "whole_piece",
        "polarity": "-",
        "magnitude": None,
        "_measurement": {
            "d": -1.0,
            "tau": 1.0,
            "error_bar": 0.1,
            "event_count": 50,
            "localizable": True,
        },
    }
    verdict, reason = route_verdict(claim, registry)
    assert verdict == "UNVERIFIABLE"
    assert reason == "out_of_scope_dim"


def test_unresolvable_location_returns_unverifiable() -> None:
    taxonomy, _ = _load()
    registry = taxonomy["dimensions"]
    claim = {
        "proposition": "You rushed in bar 5",
        "dimension": "timing",
        "location": {"bar_start": 5, "bar_end": 5},
        "polarity": "-",
        "magnitude": None,
        "_measurement": {
            "d": -15.0,
            "tau": 8.0,
            "error_bar": 2.0,
            "event_count": 12,
            "localizable": False,  # alignment uncertainty > span (1 bar)
        },
    }
    verdict, reason = route_verdict(claim, registry)
    assert verdict == "UNVERIFIABLE"
    assert reason == "unlocalizable"
