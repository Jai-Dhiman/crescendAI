"""Verify the committed baseline.json parses and has all required axes."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))

BASELINE_PATH = (
    Path(__file__).parents[3]
    / "results"
    / "exercise_routing"
    / "baseline.json"
)

REQUIRED_AXES = {
    "invocation_rate_floor",
    "kind_correctness_floor",
    "dimension_match_floor",
    "bar_range_grounding_floor",
    "tempo_sanity_floor",
}


def test_baseline_exists():
    assert BASELINE_PATH.exists(), f"baseline.json not found at {BASELINE_PATH}"


def test_baseline_parses():
    data = json.loads(BASELINE_PATH.read_text())
    assert isinstance(data, dict)


def test_baseline_has_all_axes():
    data = json.loads(BASELINE_PATH.read_text())
    missing = REQUIRED_AXES - set(data.keys())
    assert not missing, f"baseline.json missing axes: {missing}"


def test_baseline_floors_are_floats_in_0_1():
    data = json.loads(BASELINE_PATH.read_text())
    for axis in REQUIRED_AXES:
        val = data[axis]
        assert isinstance(val, (int, float)), f"{axis} must be numeric, got {type(val)}"
        assert 0.0 <= val <= 1.0, f"{axis}={val} must be in [0.0, 1.0]"
