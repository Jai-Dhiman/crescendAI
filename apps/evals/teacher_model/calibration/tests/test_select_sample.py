from __future__ import annotations

import json
import random
from pathlib import Path

from teacher_model.calibration.select_sample import select_sample


def _write_synthetic_baseline(path: Path, n_per_band: int = 300) -> None:
    rng = random.Random(7)
    rows: list[dict] = []
    composers = ["Bach", "Beethoven", "Chopin", "Debussy"]
    skill_buckets = [1, 2, 3, 4, 5]

    def _make_row(scores: list[int], ascf_process: int, idx: int) -> dict:
        # scores is a list of 7 integer values (one per dimension)
        criteria = [
            "Audible-Specific Corrective Feedback",
            "Concrete Artifact Provision",
            "Specific Positive Praise",
            "Autonomy-Supporting Motivation",
            "Scaffolded Guided Discovery",
            "Style-Consistent Musical Language",
            "Appropriate Tone & Language",
        ]
        dims = []
        for i, crit in enumerate(criteria):
            s = scores[i]
            proc = ascf_process if i == 0 else s
            dims.append({
                "criterion": crit,
                "process": proc,
                "outcome": s,
                "score": min(proc, s),
                "evidence": "",
                "reason": "",
            })
        return {
            "piece_slug": f"piece_{idx}",
            "recording_id": f"rec_{idx}",
            "skill_bucket": rng.choice(skill_buckets),
            "composer": rng.choice(composers),
            "title": f"Piece {idx}",
            "synthesis_text": f"Synthesis text for piece {idx}.",
            "muq_means": {"dynamics": 0.5},
            "judge_dimensions": dims,
        }

    # high band: composite >= 2.7 (all 3s => composite 3.0)
    # threshold band: 2.3 <= composite < 2.7 (mix of 2s and 3s; 3x3 + 4x2 = 17/7 ~ 2.43)
    # low band: composite < 2.3, ascf_process >= 2 (all 2s => composite 2.0)
    # weak_dim band: ascf_process <= 1 regardless of composite
    idx = 0
    for _ in range(n_per_band):
        rows.append(_make_row(scores=[3, 3, 3, 3, 3, 3, 3], ascf_process=3, idx=idx)); idx += 1
    for _ in range(n_per_band):
        rows.append(_make_row(scores=[2, 3, 3, 2, 3, 2, 2], ascf_process=2, idx=idx)); idx += 1
    for _ in range(n_per_band):
        rows.append(_make_row(scores=[2, 2, 2, 2, 2, 2, 2], ascf_process=2, idx=idx)); idx += 1
    for _ in range(n_per_band):
        rows.append(_make_row(scores=[2, 2, 3, 2, 3, 2, 3], ascf_process=1, idx=idx)); idx += 1

    rng.shuffle(rows)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_select_sample_band_proportions_within_tolerance(tmp_path: Path):
    source = tmp_path / "baseline.jsonl"
    _write_synthetic_baseline(source)

    manifest = select_sample(
        source_path=source,
        target_n=200,
        holdout_n=30,
        anchor_n=20,
        seed=42,
    )

    band_counts = manifest["stats"]["band_counts"]
    assert abs(band_counts["threshold"] - 80) <= 2, band_counts
    assert abs(band_counts["high"] - 40) <= 2, band_counts
    assert abs(band_counts["low"] - 30) <= 2, band_counts
    assert abs(band_counts["weak_dim"] - 50) <= 2, band_counts
    assert sum(band_counts.values()) == 200


def test_select_sample_is_deterministic_for_fixed_seed(tmp_path: Path):
    source = tmp_path / "baseline.jsonl"
    _write_synthetic_baseline(source)

    m1 = select_sample(source_path=source, target_n=200, holdout_n=30, anchor_n=20, seed=42)
    m2 = select_sample(source_path=source, target_n=200, holdout_n=30, anchor_n=20, seed=42)

    ids1 = [e["synth_id"] for e in m1["main"]]
    ids2 = [e["synth_id"] for e in m2["main"]]
    assert ids1 == ids2


def test_era_min_quotas_satisfied(tmp_path: Path):
    source = tmp_path / "baseline.jsonl"
    _write_synthetic_baseline(source)

    manifest = select_sample(
        source_path=source, target_n=200, holdout_n=30, anchor_n=20, seed=42,
    )

    era_counts = manifest["stats"]["era_counts"]
    for era in ("Baroque", "Classical", "Romantic", "Impressionist"):
        assert era_counts.get(era, 0) >= 30, (era, era_counts)
    # Every main entry has an era field set
    assert all(e["era"] is not None for e in manifest["main"])


def test_holdout_is_disjoint_from_main(tmp_path: Path):
    source = tmp_path / "baseline.jsonl"
    _write_synthetic_baseline(source)

    manifest = select_sample(
        source_path=source, target_n=200, holdout_n=30, anchor_n=20, seed=42,
    )

    main_ids = {e["synth_id"] for e in manifest["main"]}
    holdout_ids = {e["synth_id"] for e in manifest["holdout"]}

    assert len(holdout_ids) == 30
    assert main_ids.isdisjoint(holdout_ids)
    assert manifest["stats"]["n_holdout"] == 30
