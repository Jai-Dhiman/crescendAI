# apps/evals/tests/test_aggregate.py
from __future__ import annotations

import json
from pathlib import Path

from teaching_knowledge.scripts.aggregate import (
    AggregateResult,
    aggregate_run,
)


def _fixture_row(
    recording_id: str,
    dim_scores: dict[str, tuple[int, int]],
    run_id: str = "test_run_id",
) -> dict:
    """Build a fixture JSONL row in the shape run_eval.py emits."""
    return {
        "recording_id": recording_id,
        "run_id": run_id,
        "git_sha": "abc1234",
        "judge_dimensions": [
            {
                "criterion": crit,
                "process": p,
                "outcome": o,
                "score": min(p, o),
                "evidence": "",
                "reason": "",
            }
            for crit, (p, o) in dim_scores.items()
        ],
        "error": "",
    }


def _fixture_index_row(recording_id: str, era: str, skill: int) -> dict:
    return {
        "recording_id": recording_id,
        "composer_era": era,
        "skill_bucket": skill,
        "duration_bucket": "30-60s",
    }


def test_aggregate_run_computes_per_dim_means(tmp_path: Path) -> None:
    jsonl = tmp_path / "baseline.jsonl"
    index = tmp_path / "dataset_index.jsonl"

    rows = [
        _fixture_row("r1", {"Audible-Specific Corrective Feedback": (3, 2), "Specific Positive Praise": (2, 2)}),
        _fixture_row("r2", {"Audible-Specific Corrective Feedback": (2, 1), "Specific Positive Praise": (3, 3)}),
        _fixture_row("r3", {"Audible-Specific Corrective Feedback": (1, 1), "Specific Positive Praise": (2, 2)}),
        _fixture_row("r4", {"Audible-Specific Corrective Feedback": (3, 3), "Specific Positive Praise": (3, 2)}),
        _fixture_row("r5", {"Audible-Specific Corrective Feedback": (2, 2), "Specific Positive Praise": (1, 1)}),
    ]
    jsonl.write_text("\n".join(json.dumps(r) for r in rows))

    index_rows = [
        _fixture_index_row("r1", "Baroque", 3),
        _fixture_index_row("r2", "Baroque", 3),
        _fixture_index_row("r3", "Romantic", 5),
        _fixture_index_row("r4", "Romantic", 5),
        _fixture_index_row("r5", "Romantic", 5),
    ]
    index.write_text("\n".join(json.dumps(r) for r in index_rows))

    result = aggregate_run(jsonl, index)

    assert isinstance(result, AggregateResult)
    assert result.total_rows == 5
    assert result.run_id == "test_run_id"

    dims_by_name = {d.name: d for d in result.dimensions}
    asc = dims_by_name["Audible-Specific Corrective Feedback"]
    # Process means: (3+2+1+3+2)/5 = 2.2
    assert abs(asc.mean_process - 2.2) < 0.001
    # Outcome means: (2+1+1+3+2)/5 = 1.8
    assert abs(asc.mean_outcome - 1.8) < 0.001
    assert asc.n == 5


def test_aggregate_run_computes_stratified_breakdowns(tmp_path: Path) -> None:
    jsonl = tmp_path / "baseline.jsonl"
    index = tmp_path / "dataset_index.jsonl"

    rows = [
        _fixture_row("r1", {"Specific Positive Praise": (3, 3)}),
        _fixture_row("r2", {"Specific Positive Praise": (3, 3)}),
        _fixture_row("r3", {"Specific Positive Praise": (1, 1)}),
        _fixture_row("r4", {"Specific Positive Praise": (1, 1)}),
        _fixture_row("r5", {"Specific Positive Praise": (2, 2)}),
    ]
    jsonl.write_text("\n".join(json.dumps(r) for r in rows))

    index_rows = [
        _fixture_index_row("r1", "Baroque", 3),
        _fixture_index_row("r2", "Baroque", 3),
        _fixture_index_row("r3", "Romantic", 5),
        _fixture_index_row("r4", "Romantic", 5),
        _fixture_index_row("r5", "Romantic", 5),
    ]
    index.write_text("\n".join(json.dumps(r) for r in index_rows))

    result = aggregate_run(jsonl, index)

    assert "Baroque" in result.by_era
    assert "Romantic" in result.by_era
    baroque_ppraise = result.by_era["Baroque"]["Specific Positive Praise"]
    romantic_ppraise = result.by_era["Romantic"]["Specific Positive Praise"]
    assert abs(baroque_ppraise - 3.0) < 0.001  # (3+3)/2
    # (1+1+2)/3 = 1.333
    assert abs(romantic_ppraise - (4 / 3)) < 0.001

    # by_skill must use string keys (matches dict[str, ...] type + JSON output)
    assert "3" in result.by_skill
    assert "5" in result.by_skill
    assert abs(result.by_skill["3"]["Specific Positive Praise"] - 3.0) < 0.001


def test_aggregate_skips_error_rows(tmp_path: Path) -> None:
    jsonl = tmp_path / "baseline.jsonl"
    index = tmp_path / "dataset_index.jsonl"

    rows = [
        _fixture_row("r1", {"Specific Positive Praise": (3, 3)}),
        {"recording_id": "r2", "run_id": "test_run_id", "git_sha": "x", "error": "boom", "judge_dimensions": []},
        _fixture_row("r3", {"Specific Positive Praise": (2, 2)}),
        _fixture_row("r4", {"Specific Positive Praise": (2, 2)}),
        _fixture_row("r5", {"Specific Positive Praise": (3, 3)}),
        _fixture_row("r6", {"Specific Positive Praise": (2, 2)}),
    ]
    jsonl.write_text("\n".join(json.dumps(r) for r in rows))
    index.write_text("\n".join(
        json.dumps(_fixture_index_row(f"r{i+1}", "Baroque", 3)) for i in range(6)
    ))

    result = aggregate_run(jsonl, index)
    assert result.total_rows == 5  # error row excluded


def test_aggregate_produces_bootstrap_ci_when_enough_samples(tmp_path: Path) -> None:
    jsonl = tmp_path / "baseline.jsonl"
    index = tmp_path / "dataset_index.jsonl"

    rows = [_fixture_row(f"r{i}", {"Specific Positive Praise": (2, 2)}) for i in range(10)]
    jsonl.write_text("\n".join(json.dumps(r) for r in rows))
    index.write_text("\n".join(
        json.dumps(_fixture_index_row(f"r{i}", "Baroque", 3)) for i in range(10)
    ))

    result = aggregate_run(jsonl, index)
    ppraise = next(d for d in result.dimensions if d.name == "Specific Positive Praise")
    assert ppraise.ci_process is not None
    low, high = ppraise.ci_process
    assert low <= 2.0 <= high


def test_aggregate_ci_is_none_for_tiny_samples(tmp_path: Path) -> None:
    jsonl = tmp_path / "baseline.jsonl"
    index = tmp_path / "dataset_index.jsonl"

    rows = [_fixture_row(f"r{i}", {"Specific Positive Praise": (2, 2)}) for i in range(3)]
    jsonl.write_text("\n".join(json.dumps(r) for r in rows))
    index.write_text("\n".join(
        json.dumps(_fixture_index_row(f"r{i}", "Baroque", 3)) for i in range(3)
    ))

    result = aggregate_run(jsonl, index)
    ppraise = next(d for d in result.dimensions if d.name == "Specific Positive Praise")
    assert ppraise.ci_process is None  # N < 5 triggers None from bootstrap_ci
