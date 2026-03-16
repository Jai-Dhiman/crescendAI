from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _get_git_info() -> tuple[str, bool]:
    """Return (sha, dirty) for the current git repo."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        sha = "unknown"

    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = len(status) > 0
    except (subprocess.CalledProcessError, FileNotFoundError):
        dirty = False

    return sha, dirty


@dataclass
class MetricResult:
    mean: float
    std: float
    n: int
    pass_threshold: float | None = None

    @property
    def passed(self) -> bool:
        if self.pass_threshold is None:
            return True
        return self.mean >= self.pass_threshold


@dataclass
class EvalReport:
    eval_name: str
    eval_version: str
    dataset: str
    metrics: dict[str, MetricResult]
    worst_cases: list[dict[str, Any]] = field(default_factory=list)
    cost: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        sha, dirty = _get_git_info()
        self.metadata.setdefault("git_sha", sha)
        self.metadata.setdefault("git_dirty", dirty)

    def to_json(self) -> dict[str, Any]:
        return {
            "eval_name": self.eval_name,
            "eval_version": self.eval_version,
            "dataset": self.dataset,
            "metrics": {
                name: {
                    "mean": m.mean,
                    "std": m.std,
                    "n": m.n,
                    "pass_threshold": m.pass_threshold,
                    "passed": m.passed,
                }
                for name, m in self.metrics.items()
            },
            "worst_cases": self.worst_cases,
            "cost": self.cost,
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2) + "\n")

    def print_summary(self) -> None:
        header = f"{'Metric':<30} {'Value':>10} {'Gate':>10} {'Status':>8}"
        separator = "-" * len(header)
        print()
        print(f"  {self.eval_name} v{self.eval_version} ({self.dataset})")
        print(f"  {separator}")
        print(f"  {header}")
        print(f"  {separator}")
        for name, m in self.metrics.items():
            value_str = f"{m.mean:.3f}"
            gate_str = f"{m.pass_threshold:.3f}" if m.pass_threshold is not None else "---"
            status_str = "PASS" if m.passed else "FAIL"
            print(f"  {name:<30} {value_str:>10} {gate_str:>10} {status_str:>8}")
        print(f"  {separator}")
        print()
