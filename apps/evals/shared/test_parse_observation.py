"""Tests for _parse_observation (#143: the eval read fields the DO never emitted)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from shared.pipeline_client import _parse_observation


def _message(chunk_index: int = 4, **ctx_overrides) -> dict:
    ctx = {
        "chunk_index": chunk_index,
        "score": 0.31,
        "baseline": 0.62,
        "reasoning_trace": "rushed through the left-hand accompaniment",
        "predictions": [0.31, 0.5, 0.44, 0.7, 0.62, 0.55],
        "baselines": {"timing": 0.62},
        "analysis_facts": {"tier": 1},
        "bar_range": [9, 12],
        "analysis_tier": 1,
    }
    ctx.update(ctx_overrides)
    return {
        "type": "observation",
        "text": "I'm noticing something in your timing -- let's talk after.",
        "dimension": "timing",
        "framing": "correction",
        "eval_context": ctx,
    }


def test_reads_the_numeric_fields_from_eval_context():
    obs = _parse_observation(_message())

    # Before #143 these came back as 0, 0.0, 0.0, "" for every observation.
    assert obs.chunk_index == 4
    assert obs.score == 0.31
    assert obs.baseline == 0.62
    assert obs.reasoning_trace == "rushed through the left-hand accompaniment"


def test_keeps_the_student_facing_fields_at_the_top_level():
    obs = _parse_observation(_message())

    assert obs.dimension == "timing"
    assert obs.framing == "correction"
    assert "timing" in obs.text


def test_distinct_chunks_produce_distinct_observation_ids():
    # eval_observation_quality.py builds observation_id and the trace filename
    # from chunk_index. A constant 0 collapsed every trace onto one file.
    ids = {
        f"rec_chunk{_parse_observation(_message(chunk_index=i)).chunk_index}"
        for i in (0, 3, 7)
    }

    assert ids == {"rec_chunk0", "rec_chunk3", "rec_chunk7"}


def test_raises_when_eval_context_is_absent():
    # A non-eval session must fail loudly rather than yield zeros that look
    # like real measurements.
    message = _message()
    del message["eval_context"]

    with pytest.raises(KeyError, match="no eval_context"):
        _parse_observation(message)


def test_raises_when_eval_context_is_missing_a_field():
    message = _message()
    del message["eval_context"]["score"]

    with pytest.raises(KeyError):
        _parse_observation(message)


def test_preserves_the_raw_message_for_the_judge_context():
    obs = _parse_observation(_message())

    # eval_observation_quality.py reads the judge context off raw_message.
    ctx = obs.raw_message["eval_context"]
    assert ctx["predictions"] == [0.31, 0.5, 0.44, 0.7, 0.62, 0.55]
    assert ctx["analysis_facts"] == {"tier": 1}
    assert ctx["bar_range"] == [9, 12]
