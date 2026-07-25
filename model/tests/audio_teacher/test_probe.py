"""Offline end-to-end probe runs through the CLI entrypoint."""
from __future__ import annotations

import json

import pytest

from audio_teacher.budget import BudgetExceededError
from audio_teacher.probe import main


def test_offline_probe_writes_population_partitioned_report(
    tmp_path, manifest_factory
):
    manifest_path = manifest_factory(
        [{"id": "p1", "axis": "pedaling", "population": "real", "degraded": "a"}]
    )
    recorded = tmp_path / "recorded.jsonl"
    recorded.write_text(
        json.dumps({"pair_id": "p1", "text": "blurred pedal.\nANSWER: A"}) + "\n"
    )
    run_dir = tmp_path / "run"

    rc = main(
        [
            "--manifest", str(manifest_path),
            "--repo-root", str(tmp_path),
            "--recorded", str(recorded),
            "--run-dir", str(run_dir),
        ]
    )

    report = json.loads((run_dir / "report.json").read_text())
    assert report["cells"]["pedaling/real"] == {
        "n": 1, "correct": 1, "unparseable": 0,
        "accuracy": 1.0, "unparseable_rate": 0.0,
    }
    # 1 pair < MIN_REAL_PAIRS_PER_AXIS: uncertain defaults to closed.
    assert report["verdict"] == "FAIL"
    assert rc == 1
    meta = json.loads((run_dir / "run_meta.json").read_text())
    assert meta["mode"] == "recorded"
    assert meta["spent_usd"] == 0.0
    assert (run_dir / "responses.jsonl").exists()


def test_resume_skips_pairs_already_answered(tmp_path, manifest_factory):
    manifest_path = manifest_factory(
        [
            {"id": "p1", "axis": "pedaling", "population": "real", "degraded": "a"},
            {"id": "p2", "axis": "pedaling", "population": "real", "degraded": "b"},
        ]
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "responses.jsonl").write_text(
        json.dumps({"pair_id": "p1", "text": "ANSWER: A", "cost_usd": 0.0}) + "\n"
    )
    # The recorded fixture deliberately LACKS p1: if the driver re-asked the
    # already-answered pair, RecordedResponseClient would raise KeyError.
    recorded = tmp_path / "recorded.jsonl"
    recorded.write_text(json.dumps({"pair_id": "p2", "text": "ANSWER: B"}) + "\n")

    rc = main(
        [
            "--manifest", str(manifest_path),
            "--repo-root", str(tmp_path),
            "--recorded", str(recorded),
            "--run-dir", str(run_dir),
        ]
    )

    report = json.loads((run_dir / "report.json").read_text())
    assert report["cells"]["pedaling/real"]["n"] == 2
    assert report["cells"]["pedaling/real"]["correct"] == 2
    assert rc == 1  # still under MIN_REAL_PAIRS_PER_AXIS: gate stays closed


def test_budget_guard_carries_forward_spend_across_resume(tmp_path, manifest_factory):
    manifest_path = manifest_factory(
        [
            {"id": "p1", "axis": "pedaling", "population": "real", "degraded": "a"},
            {"id": "p2", "axis": "pedaling", "population": "real", "degraded": "b"},
            {"id": "p3", "axis": "pedaling", "population": "real", "degraded": "a"},
        ]
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    # Two prior calls' actual costs (which can overshoot their pre-call
    # estimates) already sum to just over the $50 cap -- a live run that
    # crashed after this point must not get a fresh $50 allowance on resume.
    (run_dir / "responses.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"pair_id": "p1", "text": "ANSWER: A", "cost_usd": 30.0}),
                json.dumps({"pair_id": "p2", "text": "ANSWER: B", "cost_usd": 21.0}),
            ]
        )
        + "\n"
    )
    # p3 is the only pair left to answer; if the guard wrongly starts at
    # $0 it will sail through the precheck and call the client.
    recorded = tmp_path / "recorded.jsonl"
    recorded.write_text(json.dumps({"pair_id": "p3", "text": "ANSWER: A"}) + "\n")

    with pytest.raises(BudgetExceededError):
        main(
            [
                "--manifest", str(manifest_path),
                "--repo-root", str(tmp_path),
                "--recorded", str(recorded),
                "--run-dir", str(run_dir),
                "--max-spend", "50.0",
            ]
        )

    # Refused before the call: p3 must not have been answered or persisted.
    responses_text = (run_dir / "responses.jsonl").read_text()
    assert "p3" not in responses_text
    assert not (run_dir / "report.json").exists()
