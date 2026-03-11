"""Orchestrator: run all memory evaluation layers.

Usage:
    uv run python -m src.memory_eval.run_all
    uv run python -m src.memory_eval.run_all --layer retrieval
    uv run python -m src.memory_eval.run_all --layer synthesis --live
    uv run python -m src.memory_eval.run_all --layer temporal --live
    uv run python -m src.memory_eval.run_all --layer downstream --live
    uv run python -m src.memory_eval.run_all --layer chat_extraction --live
    uv run python -m src.memory_eval.run_all --layer locomo --live
    uv run python -m src.memory_eval.run_all --layer locomo --live --locomo-samples 5
    uv run python -m src.memory_eval.run_all --layer report
"""

from __future__ import annotations

import sys
from pathlib import Path

DATA_DIR = Path(__file__).parents[1] / "data"
SCENARIOS_PATH = DATA_DIR / "scenarios.jsonl"
CHAT_SCENARIOS_PATH = DATA_DIR / "chat_scenarios.jsonl"

LAYERS = ["retrieval", "synthesis", "temporal", "downstream", "chat_extraction", "locomo", "report"]


def ensure_scenarios() -> None:
    """Generate scenarios if they don't exist."""
    if not SCENARIOS_PATH.exists():
        print("Generating scenarios...")
        from .build_dataset import main as build_main
        build_main()


def ensure_chat_scenarios() -> None:
    """Generate chat extraction scenarios if they don't exist."""
    if not CHAT_SCENARIOS_PATH.exists():
        print("Generating chat extraction scenarios...")
        from .build_chat_scenarios import main as build_chat_main
        build_chat_main()


def run_retrieval() -> None:
    print("\n>>> Running retrieval assessment...")
    from .eval_retrieval import main as retrieval_main
    retrieval_main()


def run_synthesis(live: bool) -> None:
    print("\n>>> Running synthesis assessment...")
    if live:
        sys.argv = ["synthesis", "--live"]
    else:
        sys.argv = ["synthesis"]
    from .eval_synthesis import main as synthesis_main
    synthesis_main()


def run_temporal(live: bool) -> None:
    print("\n>>> Running temporal reasoning assessment...")
    if live:
        sys.argv = ["temporal", "--live"]
    else:
        sys.argv = ["temporal"]
    from .eval_temporal import main as temporal_main
    temporal_main()


def run_downstream(live: bool) -> None:
    print("\n>>> Running downstream impact assessment...")
    if live:
        sys.argv = ["downstream", "--live"]
    else:
        sys.argv = ["downstream"]
    from .eval_downstream import main as downstream_main
    downstream_main()


def run_chat_extraction(live: bool) -> None:
    print("\n>>> Running chat extraction assessment...")
    if live:
        sys.argv = ["chat_extraction", "--live"]
    else:
        sys.argv = ["chat_extraction"]
    from .eval_chat_extraction import main as chat_main
    chat_main()


def run_locomo(live: bool, max_samples: int = 2) -> None:
    print("\n>>> Running LoCoMo benchmark...")
    sys.argv = ["locomo"]
    if live:
        sys.argv.append("--live")
    sys.argv.extend(["--locomo-samples", str(max_samples)])
    from .locomo_adapter import main as locomo_main
    locomo_main()


def run_report() -> None:
    print("\n>>> Generating benchmark comparison report...")
    from .report import main as report_main
    report_main()


def main() -> None:
    args = sys.argv[1:]
    live = "--live" in args

    max_samples = 2
    if "--locomo-samples" in args:
        idx = args.index("--locomo-samples")
        if idx + 1 < len(args):
            max_samples = int(args[idx + 1])

    if "--layer" in args:
        idx = args.index("--layer")
        if idx + 1 >= len(args):
            print(f"Usage: --layer <{'|'.join(LAYERS)}>")
            sys.exit(1)
        layer = args[idx + 1]
        if layer not in LAYERS:
            print(f"Unknown layer: {layer}. Choose from: {', '.join(LAYERS)}")
            sys.exit(1)
        layers = [layer]
    else:
        layers = LAYERS

    ensure_scenarios()

    for layer in layers:
        if layer == "retrieval":
            run_retrieval()
        elif layer == "synthesis":
            run_synthesis(live)
        elif layer == "temporal":
            run_temporal(live)
        elif layer == "downstream":
            run_downstream(live)
        elif layer == "chat_extraction":
            ensure_chat_scenarios()
            run_chat_extraction(live)
        elif layer == "locomo":
            run_locomo(live, max_samples)
        elif layer == "report":
            run_report()

    if len(layers) > 1:
        print("\n" + "=" * 60)
        print("  All requested layers complete.")
        print("=" * 60)


if __name__ == "__main__":
    main()
