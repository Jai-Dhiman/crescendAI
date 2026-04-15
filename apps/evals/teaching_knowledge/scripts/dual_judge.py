"""Dual-judge calibration harness.

Runs two judges over the same synthesis outputs (or compares existing
judge JSONLs) and reports per-dim Spearman agreement plus a trust-level
classification: high (>=0.7), uncertain (0.4-0.7), low (<0.4).

Usage (offline mode, against two existing judge JSONLs):
    cd apps/evals
    uv run python -m teaching_knowledge.scripts.dual_judge \\
        --judge-a results/judge_gemma.jsonl \\
        --judge-b results/judge_gpt.jsonl \\
        --out    results/dual_judge_calibration.md
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DimensionAgreement:
    name: str
    spearman: float
    trust_level: str
    n: int


@dataclass
class DualJudgeReport:
    dimensions: list[DimensionAgreement]
    n_compared: int


def _rank(values: list[float]) -> list[float]:
    """Dense rank with average-tie-breaking."""
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def _spearman(a: list[float], b: list[float]) -> float:
    """Pure-Python Spearman rank correlation."""
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    ra = _rank(a)
    rb = _rank(b)
    n = len(a)
    mean_a = sum(ra) / n
    mean_b = sum(rb) / n
    num = sum((ra[i] - mean_a) * (rb[i] - mean_b) for i in range(n))
    denom_a = sum((ra[i] - mean_a) ** 2 for i in range(n)) ** 0.5
    denom_b = sum((rb[i] - mean_b) ** 2 for i in range(n)) ** 0.5
    if denom_a == 0 or denom_b == 0:
        return 0.0
    return num / (denom_a * denom_b)


def _classify_trust(spearman: float) -> str:
    if spearman >= 0.7:
        return "high"
    if spearman >= 0.4:
        return "uncertain"
    return "low"


def _index_by_recording(rows: list[dict]) -> dict[str, dict[str, float]]:
    """rows -> {recording_id -> {criterion -> score}}."""
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        rid = row["recording_id"]
        crit_scores: dict[str, float] = {}
        for dim in row.get("judge_dimensions", []):
            score = dim.get("score")
            if score is not None:
                crit_scores[dim["criterion"]] = float(score)
        out[rid] = crit_scores
    return out


def compute_agreement(
    judge_a_rows: list[dict],
    judge_b_rows: list[dict],
) -> list[DimensionAgreement]:
    a_idx = _index_by_recording(judge_a_rows)
    b_idx = _index_by_recording(judge_b_rows)
    common_recs = sorted(set(a_idx) & set(b_idx))

    by_crit_a: dict[str, list[float]] = defaultdict(list)
    by_crit_b: dict[str, list[float]] = defaultdict(list)

    for rid in common_recs:
        for crit, a_score in a_idx[rid].items():
            if crit in b_idx[rid]:
                by_crit_a[crit].append(a_score)
                by_crit_b[crit].append(b_idx[rid][crit])

    agreements: list[DimensionAgreement] = []
    for crit in sorted(by_crit_a.keys()):
        a_vals = by_crit_a[crit]
        b_vals = by_crit_b[crit]
        rho = _spearman(a_vals, b_vals)
        agreements.append(
            DimensionAgreement(
                name=crit,
                spearman=rho,
                trust_level=_classify_trust(rho),
                n=len(a_vals),
            )
        )
    return agreements


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual-judge calibration")
    parser.add_argument("--judge-a", type=Path, required=True)
    parser.add_argument("--judge-b", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    agreements = compute_agreement(_load_jsonl(args.judge_a), _load_jsonl(args.judge_b))
    lines = ["# Dual-Judge Calibration", ""]
    lines.append(f"{'Dimension':<45} {'Spearman':>10} {'Trust':>10} {'N':>5}")
    lines.append("-" * 72)
    for ag in agreements:
        lines.append(f"{ag.name[:45]:<45} {ag.spearman:>10.3f} {ag.trust_level:>10} {ag.n:>5}")
    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
