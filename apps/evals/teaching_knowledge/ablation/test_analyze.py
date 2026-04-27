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
