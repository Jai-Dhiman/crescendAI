from __future__ import annotations

import json
from pathlib import Path

import pytest

from teacher_model.calibration.judge_rerun import rerun_anchors


def _write_baseline(path: Path) -> None:
    rows = [
        {
            "piece_slug": "nocturne_op9no2",
            "recording_id": "rec_abc",
            "skill_bucket": 5,
            "composer": "Chopin",
            "title": "Chopin Nocturne",
            "synthesis_text": "Your pedaling sang. Try a gentle swell in mm. 3-4.",
            "muq_means": {"dynamics": 0.49},
            "judge_dimensions": [],
        },
        {
            "piece_slug": "wtc_book1_no1",
            "recording_id": "rec_xyz",
            "skill_bucket": 3,
            "composer": "Bach",
            "title": "Bach WTC Bk1 No1",
            "synthesis_text": "The voicing was clear in the opening.",
            "muq_means": {"dynamics": 0.55},
            "judge_dimensions": [],
        },
    ]
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _stub_judge(synthesis_text: str, context: dict) -> dict:
    return {
        "dimensions": [
            {"criterion": "Audible-Specific Corrective Feedback",
             "process": 2, "outcome": 1, "score": 1,
             "evidence": "mm. 3-4", "reason": "stub"},
        ],
        "model": "stub",
        "prompt_version": "synthesis_quality_judge_v2",
        "latency_ms": 1.0,
    }


def test_rerun_anchors_writes_one_jsonl_record_per_anchor(tmp_path: Path):
    baseline_path = tmp_path / "baseline.jsonl"
    output_path = tmp_path / "judge_runs.jsonl"
    _write_baseline(baseline_path)

    rerun_anchors(
        anchor_synth_ids=[
            "nocturne_op9no2__rec_abc__5",
            "wtc_book1_no1__rec_xyz__3",
        ],
        baseline_path=baseline_path,
        output_path=output_path,
        run_label="day1",
        judge_callable=_stub_judge,
    )

    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(records) == 2
    ids = {r["synth_id"] for r in records}
    assert ids == {"nocturne_op9no2__rec_abc__5", "wtc_book1_no1__rec_xyz__3"}
    assert all(r["run_label"] == "day1" for r in records)
    assert all("dimensions" in r and "ts" in r for r in records)
    assert records[0]["dimensions"][0]["criterion"] == "Audible-Specific Corrective Feedback"


def test_rerun_anchors_raises_when_judge_callable_is_none(tmp_path: Path):
    baseline_path = tmp_path / "baseline.jsonl"
    output_path = tmp_path / "judge_runs.jsonl"
    _write_baseline(baseline_path)

    with pytest.raises(ValueError, match="judge_callable must be provided"):
        rerun_anchors(
            anchor_synth_ids=["nocturne_op9no2__rec_abc__5"],
            baseline_path=baseline_path,
            output_path=output_path,
            run_label="day1",
            judge_callable=None,
        )


def test_rerun_anchors_raises_on_unknown_synth_id(tmp_path: Path):
    baseline_path = tmp_path / "baseline.jsonl"
    output_path = tmp_path / "judge_runs.jsonl"
    _write_baseline(baseline_path)

    with pytest.raises(KeyError, match="does_not_exist__nope__1"):
        rerun_anchors(
            anchor_synth_ids=["does_not_exist__nope__1"],
            baseline_path=baseline_path,
            output_path=output_path,
            run_label="day1",
            judge_callable=_stub_judge,
        )
