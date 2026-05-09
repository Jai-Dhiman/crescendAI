"""Calibration analysis: per-sub-score weighted kappa + (later) threshold agreement
+ (later) 4-bucket routing.

This file grows across T6 (kappa), T9 (threshold agreement), T12 (bucket routing).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

_CRITERION_TO_SLUG: dict[str, str] = {
    "Audible-Specific Corrective Feedback": "ascf",
    "Concrete Artifact Provision": "concrete_artifact",
    "Specific Positive Praise": "praise",
    "Autonomy-Supporting Motivation": "autonomy",
    "Scaffolded Guided Discovery": "scaffolded",
    "Style-Consistent Musical Language": "style",
    "Appropriate Tone & Language": "tone",
}


def _cohens_weighted_kappa(rater_a: list[int], rater_b: list[int], k: int = 4) -> float:
    if len(rater_a) != len(rater_b):
        raise ValueError(f"length mismatch: {len(rater_a)} vs {len(rater_b)}")
    n = len(rater_a)
    if n == 0:
        raise ValueError("empty input")
    confusion = [[0 for _ in range(k)] for _ in range(k)]
    for a, b in zip(rater_a, rater_b):
        if not (0 <= a < k and 0 <= b < k):
            raise ValueError(f"value out of range [0,{k}): a={a} b={b}")
        confusion[a][b] += 1
    marg_a = [sum(confusion[i][j] for j in range(k)) for i in range(k)]
    marg_b = [sum(confusion[i][j] for i in range(k)) for j in range(k)]
    denom = (k - 1) ** 2 if k > 1 else 1
    weights = [[((i - j) ** 2) / denom for j in range(k)] for i in range(k)]
    obs = sum(weights[i][j] * confusion[i][j] for i in range(k) for j in range(k)) / n
    exp = sum(
        weights[i][j] * marg_a[i] * marg_b[j] / (n * n)
        for i in range(k) for j in range(k)
    )
    if exp == 0:
        return 1.0 if obs == 0 else float("nan")
    return 1.0 - obs / exp


def _judge_value_for_sub_score(judge_dimensions: list[dict], sub_score: str) -> int | None:
    slug, leg = sub_score.rsplit("_", 1)
    criterion = next(
        (c for c, s in _CRITERION_TO_SLUG.items() if s == slug), None
    )
    if criterion is None:
        return None
    for d in judge_dimensions:
        if d.get("criterion") == criterion:
            v = d.get(leg)
            if v is None:
                return None
            return int(v)
    return None


def _build_baseline_index(baseline_path: Path) -> dict[str, dict]:
    index: dict[str, dict] = {}
    with baseline_path.open() as f:
        for line in f:
            row = json.loads(line)
            if row.get("error"):
                continue
            synth_id = f"{row['piece_slug']}__{row['recording_id']}__{row['skill_bucket']}"
            index[synth_id] = row
    return index


def calibrate(ratings_path: Path, baseline_path: Path) -> dict[str, Any]:
    baseline = _build_baseline_index(baseline_path)
    pairs_by_sub: dict[str, list[tuple[int, int]]] = defaultdict(list)

    with ratings_path.open() as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("event_type") != "rating":
                continue
            if rec.get("anchor_origin_id") is not None:
                continue  # anchor duplicates only feed analyze_drift
            row = baseline.get(rec["synth_id"])
            if row is None:
                continue
            jv = _judge_value_for_sub_score(row.get("judge_dimensions", []), rec["sub_score"])
            if jv is None:
                continue
            pairs_by_sub[rec["sub_score"]].append((rec["value"], jv))

    per_sub_score_kappa = {
        sub: _cohens_weighted_kappa([a for a, _ in pairs], [b for _, b in pairs])
        for sub, pairs in pairs_by_sub.items()
        if pairs
    }
    return {"per_sub_score_kappa": per_sub_score_kappa}
