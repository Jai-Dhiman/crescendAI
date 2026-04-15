from __future__ import annotations

import json
from pathlib import Path

from teacher_model.eval_ab import ABReport, run_ab


def _write_run_jsonl(path: Path, dim_scores: list[tuple[int, int]], run_id: str, synthesis_ms: int) -> None:
    rows = []
    for i, (p, o) in enumerate(dim_scores):
        rows.append({
            "recording_id": f"r{i}",
            "run_id": run_id,
            "git_sha": "abc1234",
            "piece_slug": "p",
            "title": "t",
            "composer": "Bach",
            "skill_bucket": 3,
            "synthesis_latency_ms": synthesis_ms,
            "judge_latency_ms": 50,
            "muq_means": {},
            "synthesis_text": "",
            "judge_dimensions": [
                {
                    "criterion": "Specific Positive Praise",
                    "process": p,
                    "outcome": o,
                    "score": min(p, o),
                    "evidence": "",
                    "reason": "",
                }
            ],
            "judge_model": "gemma",
            "error": "",
        })
    path.write_text("\n".join(json.dumps(r) for r in rows))


def _write_index(path: Path, n: int) -> None:
    rows = [
        json.dumps({
            "recording_id": f"r{i}",
            "composer_era": "Baroque",
            "skill_bucket": 3,
            "duration_bucket": "30-60s",
        })
        for i in range(n)
    ]
    path.write_text("\n".join(rows))


def test_candidate_wins_on_clear_improvement(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    index = tmp_path / "dataset_index.jsonl"

    _write_run_jsonl(baseline, [(1, 1)] * 20, run_id="base", synthesis_ms=1000)
    _write_run_jsonl(candidate, [(3, 3)] * 20, run_id="cand", synthesis_ms=500)
    _write_index(index, 20)

    report = run_ab(baseline, candidate, index)
    assert isinstance(report, ABReport)
    assert report.verdict == "CANDIDATE_WINS"
    assert report.regression_report.composite_delta > 0
    assert report.efficiency_delta["synthesis_latency_ms"] == -500


def test_candidate_loses_on_clear_regression(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    index = tmp_path / "dataset_index.jsonl"

    _write_run_jsonl(baseline, [(3, 3)] * 20, run_id="base", synthesis_ms=500)
    _write_run_jsonl(candidate, [(1, 1)] * 20, run_id="cand", synthesis_ms=500)
    _write_index(index, 20)

    report = run_ab(baseline, candidate, index)
    assert report.verdict == "CANDIDATE_LOSES"
    assert report.regression_report.has_regression is True


def test_equivalent_when_no_significant_change(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    index = tmp_path / "dataset_index.jsonl"

    _write_run_jsonl(baseline, [(2, 2)] * 20, run_id="base", synthesis_ms=500)
    _write_run_jsonl(candidate, [(2, 2)] * 20, run_id="cand", synthesis_ms=500)
    _write_index(index, 20)

    report = run_ab(baseline, candidate, index)
    assert report.verdict == "EQUIVALENT"
