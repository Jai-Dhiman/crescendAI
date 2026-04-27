# apps/evals/teaching_knowledge/ablation/test_analyze.py
import pytest
from pathlib import Path
from teaching_knowledge.ablation.analyze import cosine_similarity


def test_identical_strings_cosine_one():
    assert cosine_similarity("Practice was good.", "Practice was good.") == pytest.approx(1.0, abs=1e-3)


def test_orthogonal_strings_cosine_lower():
    sim = cosine_similarity(
        "The pedaling needs work in bars 8-12 on the half-pedal change.",
        "Discrete logarithm cryptography over elliptic curves.",
    )
    assert sim < 0.7


def test_compute_deltas_returns_per_condition_delta(tmp_path: Path):
    import json as _json
    jsonl = tmp_path / "ablation.jsonl"
    rows = [
        {"recording_id": "r1", "condition": "real",     "judge_dimensions": [{"score": 2.5}]},
        {"recording_id": "r1", "condition": "flip",     "judge_dimensions": [{"score": 1.5}]},
        {"recording_id": "r1", "condition": "shuffle",  "judge_dimensions": [{"score": 2.0}]},
        {"recording_id": "r1", "condition": "marginal", "judge_dimensions": [{"score": 2.0}]},
    ]
    jsonl.write_text("\n".join(_json.dumps(r) for r in rows))
    from teaching_knowledge.ablation.analyze import compute_deltas
    deltas = compute_deltas(jsonl)
    assert abs(deltas["flip"] - 1.0) < 1e-6
    assert abs(deltas["shuffle"] - 0.5) < 1e-6
    assert abs(deltas["marginal"] - 0.5) < 1e-6


from teaching_knowledge.ablation.analyze import decide_verdict


def test_verdict_true_when_all_thresholds_met():
    v = decide_verdict(deltas={"flip": 0.4, "shuffle": 0.2, "marginal": 0.18}, mean_sim_flip=0.7)
    assert v == "true"


def test_verdict_false_when_flip_delta_low():
    v = decide_verdict(deltas={"flip": 0.1, "shuffle": 0.2, "marginal": 0.18}, mean_sim_flip=0.7)
    assert v == "false"


def test_verdict_false_when_high_similarity():
    v = decide_verdict(deltas={"flip": 0.4, "shuffle": 0.2, "marginal": 0.18}, mean_sim_flip=0.95)
    assert v == "false"


def test_verdict_equivocal_in_gap():
    v = decide_verdict(deltas={"flip": 0.2, "shuffle": 0.1, "marginal": 0.1}, mean_sim_flip=0.88)
    assert v == "equivocal"
