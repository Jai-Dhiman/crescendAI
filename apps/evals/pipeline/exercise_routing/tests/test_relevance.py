"""Unit tests for relevance.py through its public interface (fake judge client)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[4]))

from pipeline.exercise_routing.relevance import (
    DrillInfo,
    RelevanceCase,
    aggregate_relevance,
    build_relevance_user,
    judge_relevance,
    load_drill_info,
)


class FakeClient:
    """Records the last prompt and returns a scripted response."""

    def __init__(self, response: str):
        self._response = response
        self.last_user: str | None = None
        self.last_system: str | None = None

    def complete(self, *, user: str, system: str, max_tokens: int) -> str:
        self.last_user = user
        self.last_system = system
        return self._response


def _case(dim="pedaling", techniques=("pedaling", "voicing"), title="Chopin Prelude (Op.28) 1"):
    return RelevanceCase(
        weakness_dimension=dim,
        weakness_context="pedal blurred the harmonies",
        bar_range=(12, 16),
        drill=DrillInfo(
            primitive_id="chopin_001",
            title=title,
            source="chopin",
            dimensions=[dim],
            techniques=list(techniques),
        ),
    )


def test_appropriate_drill_scores_high_and_is_appropriate():
    client = FakeClient('{"score": 3, "rationale": "pedal isolation targets pedaling"}')
    v = judge_relevance(_case(), client)
    assert v.score == 3
    assert v.appropriate is True


def test_off_target_drill_is_not_appropriate():
    # A finger warm-up for a pedaling weakness -> judge returns 0.
    client = FakeClient('{"score": 0, "rationale": "finger drill does not train pedaling"}')
    v = judge_relevance(
        _case(techniques=("finger_independence",), title="Hanon Exercise 1"), client
    )
    assert v.score == 0
    assert v.appropriate is False


def test_threshold_boundary_score_2_is_appropriate():
    client = FakeClient('{"score": 2, "rationale": "plausibly helps"}')
    assert judge_relevance(_case(), client).appropriate is True


def test_fenced_json_is_parsed():
    client = FakeClient('```json\n{"score": 3, "rationale": "ok"}\n```')
    assert judge_relevance(_case(), client).score == 3


def test_malformed_json_fails_loud():
    client = FakeClient("the drill seems fine to me")
    with pytest.raises(ValueError, match="non-JSON"):
        judge_relevance(_case(), client)


def test_out_of_range_score_fails_loud():
    client = FakeClient('{"score": 5, "rationale": "x"}')
    with pytest.raises(ValueError, match="out of range"):
        judge_relevance(_case(), client)


def test_missing_score_fails_loud():
    client = FakeClient('{"rationale": "forgot the score"}')
    with pytest.raises(ValueError, match="missing 'score'"):
        judge_relevance(_case(), client)


def test_prompt_includes_weakness_and_drill_details():
    client = FakeClient('{"score": 3, "rationale": "ok"}')
    judge_relevance(_case(), client)
    user = client.last_user
    assert "pedaling" in user
    assert "Chopin Prelude (Op.28) 1" in user
    assert "bars 12-16" in user


def test_session_scoped_bar_range_none_renders():
    case = RelevanceCase(
        weakness_dimension="dynamics",
        weakness_context="flat dynamics",
        bar_range=None,
        drill=_case().drill,
    )
    user = build_relevance_user(case)
    assert "session-scoped" in user


def test_aggregate_relevance_at_1_and_mean():
    client_hi = FakeClient('{"score": 3, "rationale": "a"}')
    client_lo = FakeClient('{"score": 0, "rationale": "b"}')
    verdicts = [
        judge_relevance(_case(), client_hi),
        judge_relevance(_case(), client_hi),
        judge_relevance(_case(), client_lo),
    ]
    agg = aggregate_relevance(verdicts)
    assert agg.n_judged == 3
    assert agg.relevance_at_1 == pytest.approx(2 / 3)
    assert agg.mean_score == pytest.approx(2.0)


def test_aggregate_empty_is_zero():
    agg = aggregate_relevance([])
    assert agg.n_judged == 0
    assert agg.relevance_at_1 == 0.0


def test_load_drill_info_from_manifest(tmp_path):
    manifest = {
        "hanon_001": {
            "dimensions": ["articulation", "timing"],
            "techniques": ["finger_independence", "evenness"],
            "key": "C",
            "totalBars": 29,
            "title": "Hanon Exercise 1",
            "source": "hanon",
        }
    }
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(manifest))
    info = load_drill_info("hanon_001", p)
    assert info.title == "Hanon Exercise 1"
    assert info.techniques == ["finger_independence", "evenness"]


def test_load_drill_info_missing_primitive_fails_loud(tmp_path):
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps({}))
    with pytest.raises(KeyError, match="not in manifest"):
        load_drill_info("nope_001", p)


def test_real_manifest_has_enriched_fields():
    """Guard: the shipped API manifest carries title+techniques (Unit 0)."""
    manifest_path = (
        Path(__file__).parents[4]
        / "api"
        / "src"
        / "services"
        / "exercise_primitives_manifest.json"
    )
    info = load_drill_info("hanon_001", manifest_path)
    assert info.title
    assert isinstance(info.techniques, list)
