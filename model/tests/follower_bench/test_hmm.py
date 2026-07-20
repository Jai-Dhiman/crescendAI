"""Behavioral tests for the opt-in Viterbi-HMM follower (issue #119)."""
from __future__ import annotations

import math

from follower_bench.follower import MatchedNote


def test_matched_note_confidence_defaults_none_and_accepts_a_value() -> None:
    # Existing positional construction (no confidence) is unchanged.
    m = MatchedNote(perf_index=0, score_index=0, perf_time=0.0, score_position=0.0)
    assert m.confidence is None
    # New: confidence can be supplied.
    m2 = MatchedNote(perf_index=1, score_index=2, perf_time=1.0, score_position=2.0, confidence=0.75)
    assert m2.confidence == 0.75
