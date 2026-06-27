"""Unit tests for selection.py (Python mirror of corpus-drill.ts selectPrimitive)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[4]))

from pipeline.exercise_routing.selection import (
    WIDEN_DEFAULT_PRIMITIVE,
    build_relevance_case,
    build_selector_case,
    select_primitive,
)
from pipeline.exercise_routing.score import SessionCapture

MANIFEST_PATH = (
    Path(__file__).parents[4]
    / "api" / "src" / "services" / "exercise_primitives_manifest.json"
)


def _manifest():
    return json.loads(MANIFEST_PATH.read_text())


def _fake_manifest():
    return {
        "hanon_001": {"dimensions": ["timing"], "techniques": ["evenness"],
                      "key": "C", "totalBars": 10, "title": "Hanon 1", "source": "hanon"},
        "hanon_002": {"dimensions": ["timing"], "techniques": ["evenness"],
                      "key": "C", "totalBars": 10, "title": "Hanon 2", "source": "hanon"},
        "czerny_010": {"dimensions": ["timing"], "techniques": ["velocity"],
                       "key": "C", "totalBars": 10, "title": "Czerny 10", "source": "czerny"},
    }


# --- select_primitive ---

def test_explicit_primitive_id_wins():
    m = _fake_manifest()
    pid, widened = select_primitive(
        {"kind": "corpus_drill", "target_dimension": "timing", "primitive_id": "czerny_010"}, m
    )
    assert (pid, widened) == ("czerny_010", False)


def test_explicit_primitive_absent_falls_through_to_dimension():
    m = _fake_manifest()
    pid, _ = select_primitive(
        {"kind": "corpus_drill", "target_dimension": "timing", "primitive_id": "nope"}, m
    )
    assert pid == "hanon_001"  # lowest suffix in timing bucket


def test_sort_is_by_suffix_then_id():
    m = _fake_manifest()
    pid, widened = select_primitive({"target_dimension": "timing"}, m)
    # suffix 001 < 002 < 010 -> hanon_001 first
    assert (pid, widened) == ("hanon_001", False)


def test_widen_to_default_on_empty_bucket():
    m = _fake_manifest()
    pid, widened = select_primitive({"target_dimension": "pedaling"}, m)
    assert (pid, widened) == (WIDEN_DEFAULT_PRIMITIVE, True)


def test_widen_default_absent_raises():
    m = {"czerny_010": {"dimensions": ["timing"], "techniques": [], "key": "C",
                        "totalBars": 1, "title": "t", "source": "czerny"}}
    with pytest.raises(KeyError, match="WIDEN_DEFAULT"):
        select_primitive({"target_dimension": "pedaling"}, m)


# --- parity with TS behavior on the real manifest ---

def test_parity_timing_routes_to_chopin_etude_001():
    """Pins the mirror to the documented TS selectPrimitive behavior."""
    pid, _ = select_primitive({"target_dimension": "timing"}, _manifest())
    assert pid == "chopin_etude_001"


def test_parity_pedaling_routes_to_real_pedaling_drill():
    m = _manifest()
    pid, widened = select_primitive({"target_dimension": "pedaling"}, m)
    assert widened is False
    assert "pedaling" in m[pid]["dimensions"]


# --- build_relevance_case ---

def _capture(prescribed_exercise, teaching_moments=None):
    return SessionCapture(
        session_id="s", recording=Path("r.wav"), piece_slug="p",
        teaching_moments=teaching_moments or [],
        baselines={}, piece_identification=None, piece_resolved=False,
        dominant_dimension="timing", prescribed_exercise=prescribed_exercise,
        synthesis_text="",
    )


def test_corpus_drill_builds_case_with_selected_drill():
    cap = _capture(
        {"kind": "corpus_drill", "target_dimension": "timing", "bar_range": [4, 8]},
        teaching_moments=[
            {"dimension": "timing", "is_positive": False, "reasoning": "rushed the run"}
        ],
    )
    case = build_relevance_case(cap, _manifest())
    assert case is not None
    assert case.weakness_dimension == "timing"
    assert case.drill.primitive_id == "chopin_etude_001"
    assert case.weakness_context == "rushed the run"
    assert case.bar_range == (4, 8)


def test_own_passage_loop_is_not_judged():
    cap = _capture({"kind": "own_passage_loop", "target_dimension": "timing", "bar_range": [4, 8]})
    assert build_relevance_case(cap, _manifest()) is None


def test_null_prescription_is_not_judged():
    assert build_relevance_case(_capture(None), _manifest()) is None


def test_selector_case_built_for_own_passage_session():
    """Counterfactual: even an own_passage_loop session yields a selector case."""
    cap = _capture(
        {"kind": "own_passage_loop", "target_dimension": "pedaling", "bar_range": [4, 8]},
        teaching_moments=[
            {"dimension": "timing", "is_positive": False, "reasoning": "rushed", "bar_range": [4, 8]}
        ],
    )
    case = build_selector_case(cap, _manifest())
    assert case is not None
    assert case.weakness_dimension == "timing"  # dominant_dimension, not routed target
    assert case.drill.primitive_id == "chopin_etude_001"
    assert case.bar_range == (4, 8)


def test_selector_case_none_without_dominant_dimension():
    cap = _capture(None)
    object.__setattr__(cap, "dominant_dimension", None)
    assert build_selector_case(cap, _manifest()) is None


def test_weakness_context_prefers_matching_negative_moment():
    cap = _capture(
        {"kind": "corpus_drill", "target_dimension": "pedaling", "bar_range": None},
        teaching_moments=[
            {"dimension": "timing", "is_positive": False, "reasoning": "timing issue"},
            {"dimension": "pedaling", "is_positive": False, "reasoning": "blurred pedal"},
        ],
    )
    case = build_relevance_case(cap, _manifest())
    assert case is not None
    assert case.weakness_context == "blurred pedal"
    assert case.bar_range is None
