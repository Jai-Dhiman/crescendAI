"""Cross-cutting analysis of E2E pipeline eval results.

Reads practice_eval.json and practice_eval_observations.json,
produces STOP generalization, observation quality, and piece ID reports.

No LLM calls -- pure computation on cached results.

Usage:
    cd apps/evals/
    uv run python -m pipeline.practice_eval.analyze_e2e --report reports/practice_eval.json
    uv run python -m pipeline.practice_eval.analyze_e2e --report reports/practice_eval.json --stop-only
    uv run python -m pipeline.practice_eval.analyze_e2e --report reports/practice_eval.json --piece-id-only
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def cohens_d(group1: list[float], group2: list[float]) -> float:
    """Compute Cohen's d effect size between two groups."""
    if len(group1) < 2 or len(group2) < 2:
        return 0.0
    m1, m2 = float(np.mean(group1)), float(np.mean(group2))
    s1, s2 = float(np.std(group1, ddof=1)), float(np.std(group2, ddof=1))
    n1, n2 = len(group1), len(group2)
    pooled = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0
    return float((m1 - m2) / pooled)


def bootstrap_ci(
    data: list[float],
    n_bootstrap: int = 5000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float] | None:
    """Compute bootstrap confidence interval. Returns None if N < 5."""
    if len(data) < 5:
        return None
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        means.append(float(np.mean(sample)))
    alpha = (1 - confidence) / 2
    low = float(np.quantile(means, alpha))
    high = float(np.quantile(means, 1 - alpha))
    return (round(low, 4), round(high, 4))


def build_confusion_matrix(results: list[dict]) -> dict[str, dict[str, int]]:
    """Build a confusion matrix from piece ID results."""
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in results:
        expected = r.get("expected", "unknown")
        actual = r.get("actual", "unidentified")
        matrix[expected][actual] += 1
    return {k: dict(v) for k, v in matrix.items()}


def print_stop_report(report: dict, observations: list[dict]) -> None:
    """Print STOP generalization analysis."""
    print("\n" + "=" * 60)
    print("STOP GENERALIZATION REPORT")
    print("=" * 60)

    meta = report.get("metadata", {})

    # Spearman correlation
    stop_corr = meta.get("stop_probability_skill_correlation", {})
    if stop_corr:
        rho = stop_corr.get("spearman_rho", "N/A")
        p = stop_corr.get("p_value", "N/A")
        n = stop_corr.get("n", 0)
        print(f"\nSpearman rho (STOP prob vs skill): {rho} (p={p}, n={n})")
        if isinstance(rho, (int, float)):
            if rho < -0.3:
                print("  -> Good: higher-skill students get lower STOP probability")
            elif rho > 0.1:
                print("  -> WARNING: STOP triggers MORE on skilled students (inverted)")
            else:
                print("  -> Weak/no correlation -- STOP may not generalize")

    # Trigger rate by bucket
    trigger_rates = meta.get("stop_trigger_rate_by_bucket", {})
    if trigger_rates:
        print(f"\n{'Bucket':<10} {'Trigger Rate':>15} {'N':>8} {'95% CI':>20}")
        print("-" * 55)
        for bucket in sorted(trigger_rates.keys(), key=lambda x: int(x)):
            info = trigger_rates[bucket]
            rate = info["rate"]
            n = info["n"]
            # Build binary trigger list for bootstrap
            triggers = [1.0] * int(round(rate * n)) + [0.0] * int(round((1 - rate) * n))
            ci = bootstrap_ci(triggers)
            ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "N<5"
            flag = " *" if n < 5 else ""
            print(f"  {bucket:<8} {rate:>13.3f} {n:>8}{flag} {ci_str:>20}")
        if any(trigger_rates[b]["n"] < 5 for b in trigger_rates):
            print("\n  * = N < 5, treat with caution")

    # Cohen's d between adjacent buckets
    bucket_probs = meta.get("stop_probabilities_by_bucket", {})
    if bucket_probs:
        buckets_sorted = sorted(bucket_probs.keys(), key=int)
        if len(buckets_sorted) > 1:
            print(f"\nCohen's d (STOP probability separation between adjacent buckets):")
            for i in range(len(buckets_sorted) - 1):
                b1, b2 = buckets_sorted[i], buckets_sorted[i + 1]
                d = cohens_d(bucket_probs[b1], bucket_probs[b2])
                effect = "large" if abs(d) > 0.8 else "medium" if abs(d) > 0.5 else "small"
                print(f"  Bucket {b1} vs {b2}: d={d:.3f} ({effect} effect)")


def print_observation_report(report: dict, observations: list[dict]) -> None:
    """Print observation quality dashboard by skill bucket."""
    print("\n" + "=" * 60)
    print("OBSERVATION QUALITY BY SKILL BUCKET")
    print("=" * 60)

    # Group by skill level
    by_bucket: dict[int, list[dict]] = defaultdict(list)
    for obs in observations:
        sl = obs.get("skill_level", 0)
        if sl > 0:
            by_bucket[sl].append(obs)

    if not by_bucket:
        print("\nNo skill-labeled observations found.")
        return

    # Collect all criteria
    all_criteria = set()
    for obs in observations:
        all_criteria.update(obs.get("judge_scores", {}).keys())
    criteria = sorted(all_criteria)

    if not criteria:
        print("\nNo judge scores found.")
        return

    # Header
    header = f"{'Criterion':<28}"
    for b in sorted(by_bucket.keys()):
        header += f" {'B' + str(b) + f'(n={len(by_bucket[b])})':>14}"
    print(f"\n{header}")
    print("-" * len(header))

    # Per-criterion, per-bucket pass rates
    for criterion in criteria:
        row = f"{criterion:<28}"
        for b in sorted(by_bucket.keys()):
            scores = [
                obs["judge_scores"].get(criterion)
                for obs in by_bucket[b]
                if obs["judge_scores"].get(criterion) is not None
            ]
            if scores:
                rate = sum(scores) / len(scores)
                n = len(scores)
                flag = "*" if n < 5 else " "
                row += f" {rate:>12.3f}{flag}"
            else:
                row += f" {'---':>13}"
        print(row)

    print("\n  * = N < 5, treat with caution")

    # Worst criterion per bucket
    print(f"\nWorst criterion per bucket:")
    for b in sorted(by_bucket.keys()):
        worst_name = None
        worst_rate = 1.0
        for criterion in criteria:
            scores = [
                obs["judge_scores"].get(criterion)
                for obs in by_bucket[b]
                if obs["judge_scores"].get(criterion) is not None
            ]
            if scores:
                rate = sum(scores) / len(scores)
                if rate < worst_rate:
                    worst_rate = rate
                    worst_name = criterion
        if worst_name:
            print(f"  Bucket {b}: {worst_name} ({worst_rate:.3f})")

    # Failure examples (worst-scoring observations)
    print(f"\nWorst observations (lowest judge pass rate):")
    scored_obs = []
    for obs in observations:
        scores = obs.get("judge_scores", {})
        passed = [v for v in scores.values() if v is not None]
        if passed:
            scored_obs.append((sum(passed) / len(passed), obs))
    scored_obs.sort(key=lambda x: x[0])
    for rate, obs in scored_obs[:3]:
        print(f"  [{rate:.0%}] Bucket {obs.get('skill_level', '?')} | {obs.get('piece', '?')}")
        print(f"    {obs.get('observation', '')[:120]}...")


def print_piece_id_report(report: dict) -> None:
    """Print piece ID accuracy analysis."""
    print("\n" + "=" * 60)
    print("PIECE IDENTIFICATION ACCURACY")
    print("=" * 60)

    meta = report.get("metadata", {})
    pid = meta.get("piece_id", {})
    if not pid:
        print("\nNo piece ID data (all scenarios used explicit piece_query)")
        return

    print(f"\nTop-1 accuracy: {pid.get('top1_accuracy', 'N/A')}")
    print(f"Total tested:   {pid.get('total', 0)}")
    print(f"Correct:        {pid.get('correct', 0)}")
    print(f"Mean notes to identify: {pid.get('mean_notes_to_identify', 'N/A')}")
    print(f"False positives (high-confidence wrong): {pid.get('false_positives', 0)}")


def main():
    parser = argparse.ArgumentParser(description="Analyze E2E pipeline eval results")
    parser.add_argument("--report", required=True, help="Path to practice_eval.json")
    parser.add_argument("--stop-only", action="store_true")
    parser.add_argument("--piece-id-only", action="store_true")
    args = parser.parse_args()

    report_path = Path(args.report)
    if not report_path.exists():
        print(f"Report not found: {report_path}")
        sys.exit(1)

    with open(report_path) as f:
        report = json.load(f)

    obs_path = report_path.parent / "practice_eval_observations.json"
    observations = []
    if obs_path.exists():
        with open(obs_path) as f:
            observations = json.load(f)

    if args.stop_only:
        print_stop_report(report, observations)
    elif args.piece_id_only:
        print_piece_id_report(report)
    else:
        print_stop_report(report, observations)
        print_observation_report(report, observations)
        print_piece_id_report(report)

    print("\n" + "=" * 60)
    print(f"Report:       {report_path}")
    print(f"Observations: {len(observations)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
