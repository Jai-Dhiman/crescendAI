"""A/B harness for comparing a candidate teacher (e.g., finetuned Qwen)
against a baseline (e.g., Sonnet 4.6) over the same dataset.

Usage:
    cd apps/evals
    uv run python -m teacher_model.eval_ab \\
        results/baseline_sonnet.jsonl \\
        results/candidate_qwen.jsonl \\
        --dataset-index teaching_knowledge/data/dataset_index.jsonl
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from teaching_knowledge.scripts.aggregate import aggregate_run
from teaching_knowledge.scripts.regression_check import (
    RegressionReport,
    check_regression,
    format_report,
)


@dataclass
class ABReport:
    baseline_run_id: str
    candidate_run_id: str
    regression_report: RegressionReport
    efficiency_delta: dict[str, float]
    verdict: str  # "CANDIDATE_WINS" | "CANDIDATE_LOSES" | "EQUIVALENT"


def _efficiency_delta(baseline_path: Path, candidate_path: Path) -> dict[str, float]:
    def avg(path: Path, key: str) -> float:
        vals: list[float] = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            v = row.get(key)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "synthesis_latency_ms": avg(candidate_path, "synthesis_latency_ms")
        - avg(baseline_path, "synthesis_latency_ms"),
        "judge_latency_ms": avg(candidate_path, "judge_latency_ms")
        - avg(baseline_path, "judge_latency_ms"),
    }


def _verdict(regression: RegressionReport) -> str:
    if regression.has_regression:
        return "CANDIDATE_LOSES"
    if regression.composite_significant and regression.composite_delta > 0:
        return "CANDIDATE_WINS"
    return "EQUIVALENT"


def run_ab(
    baseline_jsonl: Path,
    candidate_jsonl: Path,
    dataset_index: Path,
) -> ABReport:
    base_agg = aggregate_run(baseline_jsonl, dataset_index)
    cand_agg = aggregate_run(candidate_jsonl, dataset_index)
    regression = check_regression(base_agg, cand_agg)
    efficiency = _efficiency_delta(baseline_jsonl, candidate_jsonl)
    verdict = _verdict(regression)
    return ABReport(
        baseline_run_id=base_agg.run_id,
        candidate_run_id=cand_agg.run_id,
        regression_report=regression,
        efficiency_delta=efficiency,
        verdict=verdict,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Teacher A/B eval")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--dataset-index", type=Path, required=True)
    args = parser.parse_args()

    report = run_ab(args.baseline, args.candidate, args.dataset_index)
    print(format_report(report.regression_report))
    print()
    print(f"baseline run_id:  {report.baseline_run_id}")
    print(f"candidate run_id: {report.candidate_run_id}")
    print()
    print("efficiency deltas (candidate - baseline):")
    for k, v in report.efficiency_delta.items():
        print(f"  {k}: {v:+.1f}")
    print()
    print(f"VERDICT: {report.verdict}")


if __name__ == "__main__":
    main()
