"""Diff two aggregate results, flag dimensions that regressed.

Two aggregates are compared via CI overlap. A dimension is:
  - regressed if candidate mean < baseline mean AND CIs do not overlap
  - improved  if candidate mean > baseline mean AND CIs do not overlap
  - null      if CIs overlap

Usage:
    cd apps/evals
    uv run python -m teaching_knowledge.scripts.regression_check \\
        baseline_aggregate.json candidate_aggregate.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from teaching_knowledge.scripts.aggregate import AggregateResult, DimensionAggregate


@dataclass
class DimensionRegression:
    name: str
    baseline_mean: float | None
    candidate_mean: float | None
    delta: float | None
    significant: bool
    direction: str  # "regressed" | "improved" | "null"


@dataclass
class RegressionReport:
    dimensions: list[DimensionRegression]
    composite_delta: float
    composite_significant: bool
    has_regression: bool


def _ci_overlap(
    a: tuple[float, float] | None,
    b: tuple[float, float] | None,
) -> bool:
    if a is None or b is None:
        return True  # cannot determine significance -> treat as overlap
    return not (a[1] < b[0] or b[1] < a[0])


def _compare(
    name: str,
    baseline: DimensionAggregate,
    candidate: DimensionAggregate,
) -> DimensionRegression:
    b_mean = baseline.mean_process
    c_mean = candidate.mean_process
    delta = (c_mean - b_mean) if (b_mean is not None and c_mean is not None) else None
    overlaps = _ci_overlap(baseline.ci_process, candidate.ci_process)
    significant = not overlaps
    if not significant or delta is None:
        direction = "null"
    elif delta < 0:
        direction = "regressed"
    else:
        direction = "improved"
    return DimensionRegression(
        name=name,
        baseline_mean=b_mean,
        candidate_mean=c_mean,
        delta=delta,
        significant=significant,
        direction=direction,
    )


def check_regression(
    baseline: AggregateResult,
    candidate: AggregateResult,
) -> RegressionReport:
    baseline_by_name = {d.name: d for d in baseline.dimensions}
    candidate_by_name = {d.name: d for d in candidate.dimensions}
    names = sorted(set(baseline_by_name) | set(candidate_by_name))

    dims: list[DimensionRegression] = []
    for name in names:
        b = baseline_by_name.get(name)
        c = candidate_by_name.get(name)
        if b is None or c is None:
            continue
        dims.append(_compare(name, b, c))

    composite_delta = candidate.composite_mean - baseline.composite_mean
    composite_significant = not _ci_overlap(baseline.composite_ci, candidate.composite_ci)
    has_regression = any(d.direction == "regressed" for d in dims)

    return RegressionReport(
        dimensions=dims,
        composite_delta=composite_delta,
        composite_significant=composite_significant,
        has_regression=has_regression,
    )


def format_report(report: RegressionReport) -> str:
    lines: list[str] = []
    lines.append("=" * 64)
    lines.append("REGRESSION CHECK")
    lines.append("=" * 64)
    lines.append(
        f"composite delta: {report.composite_delta:+.3f}  "
        f"({'SIG' if report.composite_significant else 'null'})"
    )
    lines.append("")
    lines.append(f"{'Dimension':<45} {'delta':>8} {'direction':>12}")
    lines.append("-" * 68)
    for dim in report.dimensions:
        delta_str = f"{dim.delta:+8.3f}" if dim.delta is not None else "    n/a"
        lines.append(
            f"{dim.name[:45]:<45} {delta_str} {dim.direction:>12}"
            + ("  *" if dim.significant else "")
        )
    lines.append("")
    if report.has_regression:
        lines.append("!! REGRESSION DETECTED")
    else:
        lines.append("OK: no significant regressions")
    return "\n".join(lines)


def _load_aggregate(path: Path) -> AggregateResult:
    blob = json.loads(path.read_text())
    dims = [
        DimensionAggregate(
            name=d["name"],
            mean_process=d.get("mean_process"),
            ci_process=tuple(d["ci_process"]) if d.get("ci_process") else None,
            mean_outcome=d.get("mean_outcome"),
            ci_outcome=tuple(d["ci_outcome"]) if d.get("ci_outcome") else None,
            n=d.get("n", 0),
        )
        for d in blob["dimensions"]
    ]
    return AggregateResult(
        dimensions=dims,
        composite_mean=blob["composite_mean"],
        composite_ci=tuple(blob["composite_ci"]) if blob.get("composite_ci") else None,
        by_era=blob.get("by_era", {}),
        by_skill=blob.get("by_skill", {}),
        total_rows=blob.get("total_rows", 0),
        run_id=blob.get("run_id", ""),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Regression check between two runs")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    baseline = _load_aggregate(args.baseline)
    candidate = _load_aggregate(args.candidate)
    report = check_regression(baseline, candidate)
    print(format_report(report))


if __name__ == "__main__":
    main()
