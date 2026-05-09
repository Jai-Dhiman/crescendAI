from __future__ import annotations

import json
from pathlib import Path

from teacher_model.calibration.analyze_drift import analyze_drift


def _write_ratings(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_intra_rater_kappa_is_one_when_ratings_agree(tmp_path: Path):
    ratings = []
    for synth_id in ["A1", "A2", "A3", "A4", "A5"]:
        for occurrence in (1, 2):
            anchor_id = f"anchor_{synth_id}" if occurrence == 2 else None
            for sub_score, value in [
                ("ascf_process", 2),
                ("tone_outcome", 3),
                ("autonomy_process", 1),
            ]:
                ratings.append({
                    "event_type": "rating",
                    "synth_id": synth_id if occurrence == 1 else f"{synth_id}_dup",
                    "anchor_origin_id": synth_id if occurrence == 2 else None,
                    "sub_score": sub_score,
                    "value": value,
                    "evidence": "",
                    "reason": "",
                    "session_id": "S001",
                    "session_idx": 1,
                    "ts": "2026-05-08T00:00:00Z",
                })
    ratings_path = tmp_path / "ratings.jsonl"
    _write_ratings(ratings_path, ratings)

    report = analyze_drift(ratings_path=ratings_path, judge_runs_path=None)

    intra = report["intra_rater_kappa"]
    assert intra["ascf_process"] == 1.0
    assert intra["tone_outcome"] == 1.0
    assert intra["autonomy_process"] == 1.0


def test_intra_rater_kappa_drops_below_one_when_ratings_disagree(tmp_path: Path):
    ratings = []
    for i, synth_id in enumerate(["A1", "A2", "A3", "A4", "A5"]):
        ratings.append({
            "event_type": "rating",
            "synth_id": synth_id,
            "anchor_origin_id": None,
            "sub_score": "ascf_process",
            "value": 2,
            "evidence": "", "reason": "",
            "session_id": "S001", "session_idx": i + 1,
            "ts": "2026-05-08T00:00:00Z",
        })
        ratings.append({
            "event_type": "rating",
            "synth_id": f"{synth_id}_dup",
            "anchor_origin_id": synth_id,
            "sub_score": "ascf_process",
            "value": 2 if i % 2 == 0 else 3,
            "evidence": "", "reason": "",
            "session_id": "S002", "session_idx": i + 1,
            "ts": "2026-05-08T01:00:00Z",
        })
    ratings_path = tmp_path / "ratings.jsonl"
    _write_ratings(ratings_path, ratings)

    report = analyze_drift(ratings_path=ratings_path, judge_runs_path=None)
    assert report["intra_rater_kappa"]["ascf_process"] < 1.0
    assert report["intra_rater_kappa"]["ascf_process"] >= -1.0
