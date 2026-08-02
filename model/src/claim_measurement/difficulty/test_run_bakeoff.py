"""Offline tests for run_bakeoff.py's CLI stage dispatch.

Run: cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
"""
import json
from pathlib import Path

from claim_measurement.difficulty.run_bakeoff import main


def _write_fixture_data(data_root: Path):
    manifest = [
        {"seg_id": "a", "key": "A.mid", "grade": 2, "video_id": "x", "midi_name": "mid/A.mid"},
        {"seg_id": "b", "key": "B.mid", "grade": 5, "video_id": "y", "midi_name": "mid/B.mid"},
    ]
    labels = {"A.mid": {"composer": "Bach"}, "B.mid": {"composer": "Czerny"}}
    (data_root / "results" / "amt_gap_curve").mkdir(parents=True)
    (data_root / "raw" / "psyllabus").mkdir(parents=True)
    (data_root / "results" / "amt_gap_curve" / "manifest.json").write_text(json.dumps(manifest))
    (data_root / "raw" / "psyllabus" / "new_clean_data.json").write_text(json.dumps(labels))
    mid_dir = data_root / "results" / "amt_gap_curve" / "transkun_mid"
    mid_dir.mkdir(parents=True)
    (mid_dir / "a.mid").write_bytes(b"")
    (mid_dir / "b.mid").write_bytes(b"")


def test_sample_stage_writes_sample_manifest(tmp_path):
    _write_fixture_data(tmp_path)

    exit_code = main(["--stage", "sample", "--data-root", str(tmp_path), "--target-n", "2"])

    assert exit_code == 0
    out = json.loads((tmp_path / "results" / "bakeoff" / "sample_manifest.json").read_text())
    assert {e["seg_id"] for e in out} == {"a", "b"}
