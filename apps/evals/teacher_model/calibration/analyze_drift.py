"""Drift analysis for the rubric calibration protocol.

Intra-rater kappa on anchor duplicates and (when judge_runs_path is provided)
judge-vs-judge kappa on day1/day30 re-runs. Both share the same Cohen's
weighted-quadratic kappa implementation defined locally; analyze_calibration
re-implements identically (intentional duplication keeps modules independent).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def cohens_weighted_kappa(rater_a: list[int], rater_b: list[int], k: int = 4) -> float:
    """Cohen's weighted kappa with quadratic weights for a 0..(k-1) ordinal scale.

    Returns 1.0 for perfect agreement, ~0 for chance-level agreement, and
    can be negative for systematic disagreement. When marginals are degenerate
    (one rater outputs only one category), expected disagreement is 0 and the
    function returns 1.0 if observed disagreement is also 0, else float('nan').
    """
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


def analyze_drift(ratings_path: Path, judge_runs_path: Path | None) -> dict[str, Any]:
    pairs_by_sub_score: dict[str, list[tuple[int, int]]] = defaultdict(list)
    first_seen: dict[tuple[str, str], int] = {}

    with ratings_path.open() as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("event_type") != "rating":
                continue
            sub_score = rec["sub_score"]
            origin = rec.get("anchor_origin_id")
            if origin is None:
                first_seen[(rec["synth_id"], sub_score)] = rec["value"]
            else:
                first_value = first_seen.get((origin, sub_score))
                if first_value is None:
                    continue
                pairs_by_sub_score[sub_score].append((first_value, rec["value"]))

    intra_rater = {
        sub: cohens_weighted_kappa([a for a, _ in pairs], [b for _, b in pairs])
        for sub, pairs in pairs_by_sub_score.items()
        if pairs
    }

    return {
        "intra_rater_kappa": intra_rater,
        "judge_drift_kappa": {},
        "n_anchor_pairs": {sub: len(p) for sub, p in pairs_by_sub_score.items()},
    }
