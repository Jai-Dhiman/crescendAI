# apps/evals/teaching_knowledge/test_run_eval_atomic_gate.py
from teaching_knowledge.run_eval import _maybe_atomic_judge


class FakeAtomicJudge:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0
    def complete(self, *, user: str, system: str, max_tokens: int) -> str:
        self.calls += 1
        return self.response


HAPPY = """{
  "moves": [
    {"move_id": "voicing_diagnosis", "attempted": false, "criteria": null},
    {"move_id": "pedal_triage", "attempted": false, "criteria": null},
    {"move_id": "rubato_coaching", "attempted": false, "criteria": null},
    {"move_id": "phrasing_arc_analysis", "attempted": false, "criteria": null},
    {"move_id": "tempo_stability_triage", "attempted": false, "criteria": null},
    {"move_id": "dynamic_range_audit", "attempted": false, "criteria": null},
    {"move_id": "articulation_clarity_check", "attempted": false, "criteria": null},
    {"move_id": "exercise_proposal", "attempted": false, "criteria": null}
  ]
}"""


def test_atomic_gate_fires_below_threshold():
    judge_dims = [{"score": 1.0}, {"score": 1.5}, {"score": 2.0}]  # mean = 1.5
    client = FakeAtomicJudge(HAPPY)
    result = _maybe_atomic_judge(
        synthesis_text="x", context={}, judge_dimensions=judge_dims,
        threshold=2.0, client=client,
    )
    assert result is not None
    assert len(result["moves"]) == 8
    assert client.calls == 1


def test_atomic_gate_skips_above_threshold():
    judge_dims = [{"score": 2.5}, {"score": 2.5}, {"score": 2.5}]
    client = FakeAtomicJudge(HAPPY)
    result = _maybe_atomic_judge(
        synthesis_text="x", context={}, judge_dimensions=judge_dims,
        threshold=2.0, client=client,
    )
    assert result is None
    assert client.calls == 0
