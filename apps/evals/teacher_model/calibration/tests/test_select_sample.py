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


def test_anchors_reference_main_and_have_scrambled_display_ids(tmp_path: Path):
    source = tmp_path / "baseline.jsonl"
    _write_synthetic_baseline(source)

    manifest = select_sample(
        source_path=source, target_n=200, holdout_n=30, anchor_n=20, seed=42,
    )

    main_ids = {e["synth_id"] for e in manifest["main"]}
    anchors = manifest["anchors"]
    assert len(anchors) == 20
    for a in anchors:
        assert a["synth_id"] in main_ids
        # The displayed id must differ from the original to avoid recognition.
        assert a["synth_id_displayed"] != a["synth_id"]
        assert a["display_position"] >= len(manifest["main"])
    # All scrambled display ids must be unique.
    displayed = [a["synth_id_displayed"] for a in anchors]
    assert len(set(displayed)) == len(displayed)
    assert manifest["stats"]["n_anchors_silent_dups"] == 20


def test_skill_group_min_quotas_satisfied(tmp_path: Path):
    source = tmp_path / "baseline.jsonl"
    _write_synthetic_baseline(source)

    manifest = select_sample(
        source_path=source, target_n=200, holdout_n=30, anchor_n=20, seed=42,
    )

    counts = manifest["stats"]["skill_group_counts"]
    assert counts["beginner"] >= 50, counts
    assert counts["intermediate"] >= 50, counts
    assert counts["advanced"] >= 50, counts
    assert counts["beginner"] + counts["intermediate"] + counts["advanced"] == 200


def test_skill_quota_forces_minority_group(tmp_path: Path):
    # Build a skewed source where intermediate (skill_bucket=3) rows are
    # scarce — only 56 total. The bulk of the pool (600+) is beginner
    # (bucket 1) and advanced (bucket 4). Without quota enforcement
    # (Pass 2), a proportional draw would pick ~10 intermediate rows at
    # most. With quota enforcement, intermediate must reach 50.
    #
    # Band classification rules (from select_sample.py):
    #   weak_dim: ascf_process <= 1
    #   high:     composite >= 2.7  (all scores 3 => 3.0)
    #   threshold: 2.3 <= composite < 2.7  (mix of 2s and 3s)
    #   low:      composite < 2.3  (all scores 2 => 2.0)
    #
    # Era quotas (>=30 each) are met by spreading rows across all four
    # composers: Bach=Baroque, Beethoven=Classical, Chopin=Romantic,
    # Debussy=Impressionist.
    criteria = [
        "Audible-Specific Corrective Feedback",
        "Concrete Artifact Provision",
        "Specific Positive Praise",
        "Autonomy-Supporting Motivation",
        "Scaffolded Guided Discovery",
        "Style-Consistent Musical Language",
        "Appropriate Tone & Language",
    ]

    def _make_row_with_band(
        composer: str, recording_id: str, skill_bucket: int, band: str
    ) -> dict:
        # Choose score values that produce the requested band classification.
        # weak_dim: ascf_process=1 (all other scores 2)
        # high:     all scores 3 (composite=3.0)
        # threshold: scores [3,3,3,2,2,2,2] (composite=17/7~2.43)
        # low:      all scores 2 (composite=2.0)
        if band == "weak_dim":
            scores = [2, 2, 2, 2, 2, 2, 2]
            ascf_proc = 1
        elif band == "high":
            scores = [3, 3, 3, 3, 3, 3, 3]
            ascf_proc = 3
        elif band == "threshold":
            scores = [3, 3, 3, 2, 2, 2, 2]
            ascf_proc = 3
        else:  # low
            scores = [2, 2, 2, 2, 2, 2, 2]
            ascf_proc = 2
        dims = []
        for i, c in enumerate(criteria):
            s = scores[i]
            proc = ascf_proc if i == 0 else s
            dims.append({
                "criterion": c,
                "process": proc,
                "outcome": s,
                "score": min(proc, s),
                "evidence": "",
                "reason": "",
            })
        return {
            "piece_slug": f"piece_{recording_id}",
            "recording_id": recording_id,
            "skill_bucket": skill_bucket,
            "composer": composer,
            "title": f"Piece {recording_id}",
            "synthesis_text": f"Synthesis text for {recording_id}.",
            "muq_means": {"dynamics": 0.5},
            "judge_dimensions": dims,
        }

    # Era mapping: Bach=Baroque, Beethoven=Classical, Chopin=Romantic, Debussy=Impressionist
    # Each composer gets 150 beginner + 150 advanced rows spread across all bands.
    # Intermediate rows: 14 per composer = 56 total (scarce minority), distributed
    # across all bands so band targets can be met.
    composers = ["Bach", "Beethoven", "Chopin", "Debussy"]
    bands = ["high", "threshold", "low", "weak_dim"]
    source = tmp_path / "skewed.jsonl"
    rows = []
    for comp in composers:
        tag = comp[:3].lower()
        # 150 beginner rows: 37-38 per band
        for b_idx, band in enumerate(bands):
            count = 38 if b_idx < 2 else 37
            for i in range(count):
                rows.append(_make_row_with_band(comp, f"rec_{tag}_beg_{band[:2]}{i}", 1, band))
        # 150 advanced rows: 37-38 per band
        for b_idx, band in enumerate(bands):
            count = 38 if b_idx < 2 else 37
            for i in range(count):
                rows.append(_make_row_with_band(comp, f"rec_{tag}_adv_{band[:2]}{i}", 4, band))
        # 14 intermediate rows: ~3-4 per band (scarce minority)
        for b_idx, band in enumerate(bands):
            count = 4 if b_idx < 2 else 3
            for i in range(count):
                rows.append(_make_row_with_band(comp, f"rec_{tag}_int_{band[:2]}{i}", 3, band))
    # Total: 4 * (150+150+14) = 1256 rows, only 56 intermediate

    with source.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    manifest = select_sample(
        source_path=source, target_n=200, holdout_n=30, anchor_n=20, seed=99,
    )

    counts = manifest["stats"]["skill_group_counts"]
    assert counts["intermediate"] >= 50, counts
