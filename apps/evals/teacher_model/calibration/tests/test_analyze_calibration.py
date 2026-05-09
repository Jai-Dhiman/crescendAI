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


def _write_full_synth(ratings_path: Path, baseline_path: Path,
                     synths: list[tuple[str, dict[str, int], dict[str, int]]]) -> None:
    with ratings_path.open("w") as rf:
        for synth_id, rater_vals, _ in synths:
            for sub, v in rater_vals.items():
                rf.write(json.dumps({
                    "event_type": "rating",
                    "synth_id": synth_id,
                    "anchor_origin_id": None,
                    "sub_score": sub,
                    "value": v,
                    "evidence": "", "reason": "",
                    "session_id": "S001", "session_idx": 1,
                    "ts": "x",
                }) + "\n")

    slug_to_criterion = {
        "ascf": "Audible-Specific Corrective Feedback",
        "concrete_artifact": "Concrete Artifact Provision",
        "praise": "Specific Positive Praise",
        "autonomy": "Autonomy-Supporting Motivation",
        "scaffolded": "Scaffolded Guided Discovery",
        "style": "Style-Consistent Musical Language",
        "tone": "Appropriate Tone & Language",
    }
    with baseline_path.open("w") as bf:
        for synth_id, _, judge_vals in synths:
            piece_slug, recording_id, skill = synth_id.split("__")
            dims = []
            for slug, criterion in slug_to_criterion.items():
                v = judge_vals.get(slug, 3)
                dims.append({
                    "criterion": criterion,
                    "process": v, "outcome": v, "score": v,
                    "evidence": "", "reason": "",
                })
            bf.write(json.dumps({
                "piece_slug": piece_slug,
                "recording_id": recording_id,
                "skill_bucket": int(skill),
                "composer": "Chopin",
                "title": "x",
                "synthesis_text": "x",
                "muq_means": {},
                "judge_dimensions": dims,
            }) + "\n")


def test_threshold_agreement_high_when_pass_decisions_align(tmp_path: Path):
    synths = []
    for i in range(10):
        synth_id = f"piece__rec{i}__3"
        rv = {sub: 3 for sub in [
            "ascf_process", "concrete_artifact_process", "praise_process",
            "autonomy_process", "scaffolded_process", "style_process",
            "tone_process", "autonomy_outcome", "tone_outcome",
            "concrete_artifact_outcome", "praise_outcome",
        ]}
        jv = {s: 3 for s in ["ascf", "concrete_artifact", "praise", "autonomy",
                              "scaffolded", "style", "tone"]}
        synths.append((synth_id, rv, jv))
    ratings_path = tmp_path / "ratings.jsonl"
    baseline_path = tmp_path / "baseline.jsonl"
    _write_full_synth(ratings_path, baseline_path, synths)

    report = calibrate(ratings_path=ratings_path, baseline_path=baseline_path)
    assert report["threshold_decision_agreement"] == 1.0
    assert report["n_threshold_pairs"] == 10


def test_threshold_agreement_low_when_borderline_disagrees(tmp_path: Path):
    synths = []
    for i in range(10):
        synth_id = f"piece__rec{i}__3"
        rater_v = 3 if i % 2 == 0 else 2
        judge_v = 2 if i % 2 == 0 else 3
        rv = {sub: rater_v for sub in [
            "ascf_process", "concrete_artifact_process", "praise_process",
            "autonomy_process", "scaffolded_process", "style_process",
            "tone_process", "autonomy_outcome", "tone_outcome",
            "concrete_artifact_outcome", "praise_outcome",
        ]}
        jv = {s: judge_v for s in ["ascf", "concrete_artifact", "praise", "autonomy",
                                    "scaffolded", "style", "tone"]}
        synths.append((synth_id, rv, jv))
    ratings_path = tmp_path / "ratings.jsonl"
    baseline_path = tmp_path / "baseline.jsonl"
    _write_full_synth(ratings_path, baseline_path, synths)

    report = calibrate(ratings_path=ratings_path, baseline_path=baseline_path)
    assert report["threshold_decision_agreement"] == 0.0


def test_bucket_routing_perfect_agreement_yields_trusted(tmp_path: Path):
    synths = []
    for i in range(50):
        synth_id = f"piece__rec{i}__3"
        # rater values vary across 0..3 to give variance
        v = i % 4
        rv = {sub: v for sub in [
            "ascf_process", "concrete_artifact_process", "praise_process",
            "autonomy_process", "scaffolded_process", "style_process",
            "tone_process", "autonomy_outcome", "tone_outcome",
            "concrete_artifact_outcome", "praise_outcome",
        ]}
        jv = {s: v for s in ["ascf", "concrete_artifact", "praise", "autonomy",
                              "scaffolded", "style", "tone"]}
        synths.append((synth_id, rv, jv))
    ratings_path = tmp_path / "ratings.jsonl"
    baseline_path = tmp_path / "baseline.jsonl"
    _write_full_synth(ratings_path, baseline_path, synths)

    report = calibrate(ratings_path=ratings_path, baseline_path=baseline_path)

    for sub in [
        "ascf_process", "concrete_artifact_process", "praise_process",
        "autonomy_process", "scaffolded_process", "style_process",
        "tone_process", "autonomy_outcome", "tone_outcome",
        "concrete_artifact_outcome", "praise_outcome",
    ]:
        assert report["buckets"][sub] == "TRUSTED", (sub, report["buckets"])
    assert report["aggregate_gate_pass"] is True


def test_bucket_routing_saturated_dim_yields_ceiling_artifact(tmp_path: Path):
    # Both rater and judge always emit 3 for tone_outcome — variance is 0.
    synths = []
    for i in range(50):
        synth_id = f"piece__rec{i}__3"
        # Vary other dims, freeze tone outcome at 3
        v = i % 4
        rv = {
            "ascf_process": v, "concrete_artifact_process": v,
            "praise_process": v, "autonomy_process": v,
            "scaffolded_process": v, "style_process": v, "tone_process": v,
            "autonomy_outcome": v, "tone_outcome": 3,
            "concrete_artifact_outcome": v, "praise_outcome": v,
        }
        jv = {s: v for s in ["ascf", "concrete_artifact", "praise",
                              "autonomy", "scaffolded", "style"]}
        jv["tone"] = 3  # judge also saturated
        synths.append((synth_id, rv, jv))
    ratings_path = tmp_path / "ratings.jsonl"
    baseline_path = tmp_path / "baseline.jsonl"
    _write_full_synth(ratings_path, baseline_path, synths)

    report = calibrate(ratings_path=ratings_path, baseline_path=baseline_path)
    assert report["buckets"]["tone_outcome"] == "CEILING_ARTIFACT"


def test_bucket_routing_systematic_offset_yields_trusted_with_offset(tmp_path: Path):
    # Rater is consistently 1 less than judge for ascf_process; other dims aligned.
    synths = []
    for i in range(50):
        synth_id = f"piece__rec{i}__3"
        v = i % 4
        rater_ascf = max(0, v - 1)
        rv = {
            "ascf_process": rater_ascf,
            "concrete_artifact_process": v, "praise_process": v,
            "autonomy_process": v, "scaffolded_process": v, "style_process": v,
            "tone_process": v, "autonomy_outcome": v, "tone_outcome": v,
            "concrete_artifact_outcome": v, "praise_outcome": v,
        }
        jv = {s: v for s in ["ascf", "concrete_artifact", "praise", "autonomy",
                              "scaffolded", "style", "tone"]}
        synths.append((synth_id, rv, jv))
    ratings_path = tmp_path / "ratings.jsonl"
    baseline_path = tmp_path / "baseline.jsonl"
    _write_full_synth(ratings_path, baseline_path, synths)

    report = calibrate(ratings_path=ratings_path, baseline_path=baseline_path)
    assert report["buckets"]["ascf_process"] == "TRUSTED_WITH_OFFSET"
    assert abs(report["mean_offset"]["ascf_process"] - (-0.75)) < 0.05


def test_bucket_routing_independent_random_yields_untrusted(tmp_path: Path):
    rng = random.Random(7)
    synths = []
    for i in range(200):
        synth_id = f"piece__rec{i}__3"
        rv = {sub: rng.randint(0, 3) for sub in [
            "ascf_process", "concrete_artifact_process", "praise_process",
            "autonomy_process", "scaffolded_process", "style_process",
            "tone_process", "autonomy_outcome", "tone_outcome",
            "concrete_artifact_outcome", "praise_outcome",
        ]}
        jv = {s: rng.randint(0, 3) for s in ["ascf", "concrete_artifact", "praise",
                                              "autonomy", "scaffolded", "style", "tone"]}
        synths.append((synth_id, rv, jv))
    ratings_path = tmp_path / "ratings.jsonl"
    baseline_path = tmp_path / "baseline.jsonl"
    _write_full_synth(ratings_path, baseline_path, synths)

    report = calibrate(ratings_path=ratings_path, baseline_path=baseline_path)
    assert report["buckets"]["ascf_process"] == "UNTRUSTED"
    assert report["aggregate_gate_pass"] is False
