from __future__ import annotations

import json
import random
from pathlib import Path

from teacher_model.calibration.analyze_calibration import calibrate


def _write_ratings(path: Path, pairs: list[tuple[int, int]], sub_score: str = "ascf_process") -> None:
    with path.open("w") as f:
        for i, (rv, _) in enumerate(pairs):
            rec = {
                "event_type": "rating",
                "synth_id": f"piece__rec_{i:04d}__0",
                "anchor_origin_id": None,
                "sub_score": sub_score,
                "value": rv,
                "evidence": "", "reason": "",
                "session_id": "S001", "session_idx": i + 1,
                "ts": "2026-05-08T00:00:00Z",
            }
            f.write(json.dumps(rec) + "\n")


_DIM_SLUG_TO_CRITERION = {
    "ascf": "Audible-Specific Corrective Feedback",
    "concrete_artifact": "Concrete Artifact Provision",
    "praise": "Specific Positive Praise",
    "autonomy": "Autonomy-Supporting Motivation",
    "scaffolded": "Scaffolded Guided Discovery",
    "style": "Style-Consistent Musical Language",
    "tone": "Appropriate Tone & Language",
}


def _write_baseline(path: Path, pairs: list[tuple[int, int]], sub_score: str = "ascf_process") -> None:
    slug, leg = sub_score.rsplit("_", 1)
    criterion = _DIM_SLUG_TO_CRITERION[slug]
    with path.open("w") as f:
        for i, (_, jv) in enumerate(pairs):
            row = {
                "piece_slug": "piece",
                "recording_id": f"rec_{i:04d}",
                "skill_bucket": 0,
                "composer": "Chopin",
                "title": f"Piece {i}",
                "synthesis_text": "x",
                "muq_means": {"dynamics": 0.5},
                "judge_dimensions": [{
                    "criterion": criterion,
                    "process": jv,
                    "outcome": jv,
                    "score": jv,
                    "evidence": "", "reason": "",
                }],
            }
            f.write(json.dumps(row) + "\n")


def test_perfect_agreement_yields_kappa_one(tmp_path: Path):
    pairs = [(0, 0)] * 50 + [(1, 1)] * 50 + [(2, 2)] * 50 + [(3, 3)] * 50
    ratings_path = tmp_path / "ratings.jsonl"
    baseline_path = tmp_path / "baseline.jsonl"
    _write_ratings(ratings_path, pairs)
    _write_baseline(baseline_path, pairs)

    report = calibrate(ratings_path=ratings_path, baseline_path=baseline_path)
    assert abs(report["per_sub_score_kappa"]["ascf_process"] - 1.0) < 0.01


def test_perfect_mirror_disagreement_yields_kappa_minus_one(tmp_path: Path):
    pairs = [(0, 3)] * 50 + [(1, 2)] * 50 + [(2, 1)] * 50 + [(3, 0)] * 50
    ratings_path = tmp_path / "ratings.jsonl"
    baseline_path = tmp_path / "baseline.jsonl"
    _write_ratings(ratings_path, pairs)
    _write_baseline(baseline_path, pairs)

    report = calibrate(ratings_path=ratings_path, baseline_path=baseline_path)
    assert report["per_sub_score_kappa"]["ascf_process"] <= -0.99


def test_independent_uniform_ratings_yield_kappa_near_zero(tmp_path: Path):
    rng = random.Random(123)
    pairs = [(rng.randint(0, 3), rng.randint(0, 3)) for _ in range(2000)]
    ratings_path = tmp_path / "ratings.jsonl"
    baseline_path = tmp_path / "baseline.jsonl"
    _write_ratings(ratings_path, pairs)
    _write_baseline(baseline_path, pairs)

    report = calibrate(ratings_path=ratings_path, baseline_path=baseline_path)
    k = report["per_sub_score_kappa"]["ascf_process"]
    assert abs(k) < 0.05, f"expected near-zero kappa for independent ratings, got {k}"
