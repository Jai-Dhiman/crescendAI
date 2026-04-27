# apps/evals/shared/test_judge_atomic.py
import pytest
from shared.judge_atomic import judge_atomic_matrix


class FakeJudge:
    def __init__(self, response: str) -> None:
        self.response = response
        self.last_user: str | None = None

    def complete(self, *, user: str, system: str, max_tokens: int) -> str:
        self.last_user = user
        return self.response


HAPPY_RESPONSE = """{
  "moves": [
    {"move_id": "voicing_diagnosis", "attempted": true, "criteria": [true, true, false, true, false]},
    {"move_id": "pedal_triage", "attempted": false, "criteria": null},
    {"move_id": "rubato_coaching", "attempted": false, "criteria": null},
    {"move_id": "phrasing_arc_analysis", "attempted": true, "criteria": [false, false, true, true, false]},
    {"move_id": "tempo_stability_triage", "attempted": false, "criteria": null},
    {"move_id": "dynamic_range_audit", "attempted": false, "criteria": null},
    {"move_id": "articulation_clarity_check", "attempted": false, "criteria": null},
    {"move_id": "exercise_proposal", "attempted": true, "criteria": [true, true, true, false, true]}
  ]
}"""


def test_judge_atomic_parses_happy_response():
    judge = FakeJudge(HAPPY_RESPONSE)
    result = judge_atomic_matrix(
        synthesis_text="Make the LH below mp in bars 3-6. Practice LH alone.",
        context={"piece_name": "Prelude", "composer": "Bach"},
        client=judge,
    )
    assert len(result.moves) == 8
    voicing = next(m for m in result.moves if m.move_id == "voicing_diagnosis")
    assert voicing.attempted is True
    assert voicing.criteria == [True, True, False, True, False]
    pedal = next(m for m in result.moves if m.move_id == "pedal_triage")
    assert pedal.attempted is False
    assert pedal.criteria is None


def test_judge_atomic_includes_synthesis_in_user_msg():
    judge = FakeJudge(HAPPY_RESPONSE)
    judge_atomic_matrix(synthesis_text="UNIQUE_MARKER_TEXT", context={"piece_name": "X", "composer": "Y"}, client=judge)
    assert "UNIQUE_MARKER_TEXT" in judge.last_user


class FakeJudge2:
    def __init__(self, response: str) -> None:
        self.response = response

    def complete(self, *, user: str, system: str, max_tokens: int) -> str:
        return self.response


def test_judge_atomic_raises_on_invalid_json():
    judge = FakeJudge2("this is not json")
    with pytest.raises(ValueError, match="atomic judge returned non-JSON"):
        judge_atomic_matrix(synthesis_text="x", context={}, client=judge)


def test_judge_atomic_raises_on_missing_moves_key():
    judge = FakeJudge2('{"foo": []}')
    with pytest.raises(ValueError, match="missing 'moves'"):
        judge_atomic_matrix(synthesis_text="x", context={}, client=judge)
