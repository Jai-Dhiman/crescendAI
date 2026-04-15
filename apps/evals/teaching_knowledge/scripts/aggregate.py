"""Reduce a run JSONL to per-dim means + bootstrap CIs + stratified breakdowns.

Reads the output of run_eval.py, joins against dataset_index.jsonl for
composer_era and skill_bucket tags, and writes a single aggregate JSON.

Usage:
    cd apps/evals
    uv run python -m teaching_knowledge.scripts.aggregate \\
        results/baseline.jsonl --out results/baseline_aggregate.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from shared.stats import bootstrap_ci


@dataclass
class DimensionAggregate:
    name: str
    mean_process: float | None
    ci_process: tuple[float, float] | None
    mean_outcome: float | None
    ci_outcome: tuple[float, float] | None
    n: int


@dataclass
class AggregateResult:
    dimensions: list[DimensionAggregate]
    composite_mean: float
    composite_ci: tuple[float, float] | None
    by_era: dict[str, dict[str, float]] = field(default_factory=dict)
    by_skill: dict[str, dict[str, float]] = field(default_factory=dict)
    total_rows: int = 0
    run_id: str = ""


def _load_index(path: Path) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        index[row["recording_id"]] = row
    return index


def _iter_rows(jsonl_path: Path):
    for line in jsonl_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        yield json.loads(line)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def aggregate_run(jsonl_path: Path, dataset_index_path: Path) -> AggregateResult:
    index = _load_index(dataset_index_path)

    dim_process: dict[str, list[float]] = defaultdict(list)
    dim_outcome: dict[str, list[float]] = defaultdict(list)
    all_composite: list[float] = []

    by_era: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_skill: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    total_rows = 0
    run_id = ""

    for row in _iter_rows(jsonl_path):
        if row.get("error"):
            continue
        if not run_id:
            run_id = row.get("run_id", "")
        rec_id = row["recording_id"]
        tags = index.get(rec_id)

        dims = row.get("judge_dimensions", [])
        if not dims:
            continue
        total_rows += 1

        row_scores: list[float] = []
        for dim in dims:
            crit = dim.get("criterion", "unknown")
            # CRITICAL: skip parse_failure sentinel rows — T6 emits criterion="parse_failure"
            # with score=0, process=None, outcome=None to signal judge unparseable output.
            # Including these would drag composite means toward 0.
            if crit == "parse_failure":
                continue
            proc = dim.get("process")
            out = dim.get("outcome")
            if proc is not None:
                dim_process[crit].append(float(proc))
            if out is not None:
                dim_outcome[crit].append(float(out))
            if dim.get("score") is not None:
                row_scores.append(float(dim["score"]))
                if tags:
                    by_era[tags["composer_era"]][crit].append(float(dim["score"]))
                    by_skill[str(int(tags["skill_bucket"]))][crit].append(float(dim["score"]))

        if row_scores:
            all_composite.append(sum(row_scores) / len(row_scores))

    all_crits = sorted(set(dim_process.keys()) | set(dim_outcome.keys()))
    dimensions: list[DimensionAggregate] = []
    for crit in all_crits:
        procs = dim_process.get(crit, [])
        outs = dim_outcome.get(crit, [])
        dimensions.append(
            DimensionAggregate(
                name=crit,
                mean_process=_mean(procs),
                ci_process=bootstrap_ci(procs),
                mean_outcome=_mean(outs),
                ci_outcome=bootstrap_ci(outs),
                n=max(len(procs), len(outs)),
            )
        )

    composite_mean = _mean(all_composite) or 0.0
    composite_ci = bootstrap_ci(all_composite)

    by_era_final: dict[str, dict[str, float]] = {
        era: {crit: sum(vals) / len(vals) for crit, vals in crits.items() if vals}
        for era, crits in by_era.items()
    }
    by_skill_final: dict[str, dict[str, float]] = {
        skill: {crit: sum(vals) / len(vals) for crit, vals in crits.items() if vals}
        for skill, crits in by_skill.items()
    }

    return AggregateResult(
        dimensions=dimensions,
        composite_mean=composite_mean,
        composite_ci=composite_ci,
        by_era=by_era_final,
        by_skill=by_skill_final,
        total_rows=total_rows,
        run_id=run_id,
    )


def write_aggregate(result: AggregateResult, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(result), indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate a run JSONL")
    parser.add_argument("jsonl", type=Path, help="Path to run JSONL")
    parser.add_argument(
        "--dataset-index",
        type=Path,
        default=Path("teaching_knowledge/data/dataset_index.jsonl"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (default: <jsonl>_aggregate.json)",
    )
    args = parser.parse_args()

    out = args.out or args.jsonl.with_name(args.jsonl.stem + "_aggregate.json")
    result = aggregate_run(args.jsonl, args.dataset_index)
    write_aggregate(result, out)
    print(f"wrote {out}")
    print(f"  composite_mean: {result.composite_mean:.3f}")
    print(f"  total_rows:     {result.total_rows}")


if __name__ == "__main__":
    main()
