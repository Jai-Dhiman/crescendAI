"""Calibration analysis: per-sub-score weighted kappa + (later) threshold agreement
+ (later) 4-bucket routing.

This file grows across T6 (kappa), T9 (threshold agreement), T12 (bucket routing).
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from teacher_model.calibration.rater_cli import PHASE_1_SUB_SCORES as _PHASE_1_SUB_SCORES

COMPOSITE_PASS_THRESHOLD: float = 2.5
KAPPA_TRUSTED_THRESHOLD: float = 0.6
SCORE_VARIANCE_CEILING_THRESHOLD: float = 0.2
OFFSET_TRUSTED_THRESHOLD: float = 0.3
AGGREGATE_GATE_MIN_TRUSTED: int = 7

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


def _rater_composite(values_by_sub: dict[str, int]) -> float | None:
    if not values_by_sub:
        return None
    return sum(values_by_sub.values()) / len(values_by_sub)


def _judge_composite(judge_dimensions: list[dict]) -> float | None:
    scores = [d["score"] for d in judge_dimensions if d.get("score") is not None]
    if not scores:
        return None
    return sum(scores) / len(scores)


def _variance(values: list[int]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def _route_bucket(kappa: float, mean_offset: float, rater_var: float, judge_var: float) -> str:
    if rater_var < SCORE_VARIANCE_CEILING_THRESHOLD or judge_var < SCORE_VARIANCE_CEILING_THRESHOLD:
        return "CEILING_ARTIFACT"
    if kappa < KAPPA_TRUSTED_THRESHOLD:
        return "UNTRUSTED"
    if abs(mean_offset) >= OFFSET_TRUSTED_THRESHOLD:
        return "TRUSTED_WITH_OFFSET"
    return "TRUSTED"


def calibrate(ratings_path: Path, baseline_path: Path) -> dict[str, Any]:
    baseline = _build_baseline_index(baseline_path)
    pairs_by_sub: dict[str, list[tuple[int, int]]] = defaultdict(list)
    rater_vals_per_synth: dict[str, dict[str, int]] = defaultdict(dict)

    with ratings_path.open() as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("event_type") != "rating":
                continue
            if rec.get("anchor_origin_id") is not None:
                continue
            row = baseline.get(rec["synth_id"])
            if row is None:
                continue
            jv = _judge_value_for_sub_score(row.get("judge_dimensions", []), rec["sub_score"])
            if jv is not None:
                pairs_by_sub[rec["sub_score"]].append((rec["value"], jv))
            rater_vals_per_synth[rec["synth_id"]][rec["sub_score"]] = rec["value"]

    per_sub_score_kappa = {
        sub: _cohens_weighted_kappa([a for a, _ in pairs], [b for _, b in pairs])
        for sub, pairs in pairs_by_sub.items()
        if pairs
    }

    threshold_pairs: list[tuple[int, int]] = []
    for synth_id, vals in rater_vals_per_synth.items():
        row = baseline.get(synth_id)
        if row is None:
            continue
        rc = _rater_composite(vals)
        jc = _judge_composite(row.get("judge_dimensions", []))
        if rc is None or jc is None:
            continue
        rater_pass = 1 if rc >= COMPOSITE_PASS_THRESHOLD else 0
        judge_pass = 1 if jc >= COMPOSITE_PASS_THRESHOLD else 0
        threshold_pairs.append((rater_pass, judge_pass))

    if threshold_pairs:
        agreement = sum(1 for a, b in threshold_pairs if a == b) / len(threshold_pairs)
        a_vals = [a for a, _ in threshold_pairs]
        b_vals = [b for _, b in threshold_pairs]
        kappa_thresh = _cohens_weighted_kappa(a_vals, b_vals, k=2)
    else:
        agreement = 0.0
        kappa_thresh = float("nan")

    buckets: dict[str, str] = {}
    mean_offset: dict[str, float] = {}
    rater_variance: dict[str, float] = {}
    judge_variance: dict[str, float] = {}

    for sub in _PHASE_1_SUB_SCORES:
        pairs = pairs_by_sub.get(sub, [])
        if not pairs:
            buckets[sub] = "UNTRUSTED"
            mean_offset[sub] = 0.0
            rater_variance[sub] = 0.0
            judge_variance[sub] = 0.0
            continue
        rater_vals = [a for a, _ in pairs]
        judge_vals = [b for _, b in pairs]
        rv = _variance(rater_vals)
        jv = _variance(judge_vals)
        offset = sum(a - b for a, b in pairs) / len(pairs)
        kappa = per_sub_score_kappa.get(sub, float("nan"))
        if math.isnan(kappa):
            kappa = -1.0
        buckets[sub] = _route_bucket(kappa, offset, rv, jv)
        mean_offset[sub] = offset
        rater_variance[sub] = rv
        judge_variance[sub] = jv

    n_trusted = sum(
        1 for s in _PHASE_1_SUB_SCORES
        if buckets[s] in ("TRUSTED", "TRUSTED_WITH_OFFSET")
    )
    aggregate_gate_pass = n_trusted >= AGGREGATE_GATE_MIN_TRUSTED

    return {
        "per_sub_score_kappa": per_sub_score_kappa,
        "threshold_decision_agreement": agreement,
        "threshold_decision_kappa": kappa_thresh,
        "n_threshold_pairs": len(threshold_pairs),
        "buckets": buckets,
        "mean_offset": mean_offset,
        "rater_variance": rater_variance,
        "judge_variance": judge_variance,
        "n_phase_1_trusted": n_trusted,
        "aggregate_gate_pass": aggregate_gate_pass,
    }
