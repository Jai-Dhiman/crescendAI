"""Offline end-to-end probe runs through the CLI entrypoint."""
from __future__ import annotations

import json

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
