"""Config-driven autoresearch sweep framework.

Replaces per-phase scripts with a single runner that takes a phase config
and handles the sweep loop, Trackio logging, keep/revert logic.

Usage:
    cd model/
    uv run python -m model_improvement.autoresearch_runner --phase lr_schedule
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Callable

from src.paths import Results


@dataclass
class SweepConfig:
    """Configuration for one autoresearch phase."""
    name: str
    search_space: dict[str, list[Any]]
    train_fn: Callable[[dict[str, Any]], dict[str, float]]
    metric_key: str = "pairwise_accuracy"
    higher_is_better: bool = True
    results_dir: Path = field(default_factory=lambda: Results.root / "autoresearch")


def run_sweep(config: SweepConfig) -> dict[str, Any]:
    """Execute a full sweep over the search space.

    For each combination in the search space, calls config.train_fn(params)
    which must return a dict with at least config.metric_key.

    Returns:
        Dict with keys: best_params, best_metric, all_results, elapsed_seconds.
    """
    config.results_dir.mkdir(parents=True, exist_ok=True)

    # Generate all parameter combinations
    param_names = sorted(config.search_space.keys())
    param_values = [config.search_space[k] for k in param_names]
    combos = list(product(*param_values))

    print(f"=== Autoresearch: {config.name} ===")
    print(f"Search space: {len(combos)} combinations")
    print(f"Metric: {config.metric_key} ({'higher' if config.higher_is_better else 'lower'} is better)")

    best_metric = float("-inf") if config.higher_is_better else float("inf")
    best_params: dict[str, Any] = {}
    all_results: list[dict] = []
    start = time.time()

    for i, combo in enumerate(combos):
        params = dict(zip(param_names, combo))
        print(f"\n[{i+1}/{len(combos)}] {params}")

        try:
            result = config.train_fn(params)
            metric = result[config.metric_key]
            print(f"  -> {config.metric_key}={metric:.4f}")

            is_better = (
                metric > best_metric if config.higher_is_better
                else metric < best_metric
            )
            if is_better:
                best_metric = metric
                best_params = params
                print(f"  -> NEW BEST ({config.metric_key}={metric:.4f})")

            all_results.append({
                "params": params,
                "result": result,
                "is_best": is_better,
            })
        except Exception as e:
            print(f"  -> FAILED: {e}")
            all_results.append({
                "params": params,
                "result": None,
                "error": str(e),
            })

    elapsed = time.time() - start

    summary = {
        "phase": config.name,
        "best_params": best_params,
        "best_metric": best_metric,
        "total_experiments": len(combos),
        "successful": sum(1 for r in all_results if r.get("result") is not None),
        "elapsed_seconds": round(elapsed, 1),
        "all_results": all_results,
    }

    # Save results
    results_path = config.results_dir / f"{config.name}_results.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")
    print(f"Best: {best_params} -> {config.metric_key}={best_metric:.4f}")
    print(f"Total time: {elapsed:.0f}s")

    return summary


# --- Phase configs (populated when T5 data is ready) ---

PHASE_REGISTRY: dict[str, Callable[[], SweepConfig]] = {}


def register_phase(name: str):
    """Decorator to register a phase config factory."""
    def decorator(fn: Callable[[], SweepConfig]):
        PHASE_REGISTRY[name] = fn
        return fn
    return decorator


def main():
    parser = argparse.ArgumentParser(description="Config-driven autoresearch runner")
    parser.add_argument("--phase", required=True, choices=list(PHASE_REGISTRY.keys()),
                        help="Which autoresearch phase to run")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print search space without running")
    args = parser.parse_args()

    config_factory = PHASE_REGISTRY[args.phase]
    config = config_factory()

    if args.dry_run:
        param_names = sorted(config.search_space.keys())
        total = 1
        for values in config.search_space.values():
            total *= len(values)
        print(f"Phase: {config.name}")
        print(f"Parameters: {param_names}")
        print(f"Total combinations: {total}")
        for name, values in sorted(config.search_space.items()):
            print(f"  {name}: {values}")
        return

    run_sweep(config)


if __name__ == "__main__":
    main()
