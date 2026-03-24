"""Orchestrator: run all memory evaluation layers.

Usage:
    uv run python -m src.memory_eval.run_all
    uv run python -m src.memory_eval.run_all --layer retrieval
    uv run python -m src.memory_eval.run_all --layer synthesis --live
    uv run python -m src.memory_eval.run_all --layer synthesis --layer temporal
    uv run python -m src.memory_eval.run_all --layer temporal --live
    uv run python -m src.memory_eval.run_all --layer downstream --live
    uv run python -m src.memory_eval.run_all --layer chat_extraction --live
    uv run python -m src.memory_eval.run_all --layer locomo --live
    uv run python -m src.memory_eval.run_all --layer locomo --live --locomo-samples 5
    uv run python -m src.memory_eval.run_all --layer report
    uv run python -m src.memory_eval.run_all --json-output
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parents[1] / "data"
SCENARIOS_PATH = DATA_DIR / "scenarios.jsonl"
CHAT_SCENARIOS_PATH = DATA_DIR / "chat_scenarios.jsonl"

LAYERS = ["retrieval", "synthesis", "temporal", "downstream", "chat_extraction", "locomo", "report"]

API_BASE = "http://localhost:8787"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments. Accepts argv for testability."""
    parser = argparse.ArgumentParser(
        description="Run memory evaluation layers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--layer",
        dest="layers",
        action="append",
        choices=LAYERS,
        metavar="LAYER",
        help=f"Layer(s) to run. Can be specified multiple times. Choices: {', '.join(LAYERS)}",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        default=False,
        help="Run against live APIs instead of cached responses.",
    )
    parser.add_argument(
        "--locomo-samples",
        type=int,
        default=2,
        dest="locomo_samples",
        help="Maximum LoCoMo samples to evaluate (default: 2).",
    )
    parser.add_argument(
        "--json-output",
        action="store_true",
        default=False,
        dest="json_output",
        help="Emit machine-readable JSON scores to stdout after evaluation.",
    )

    args = parser.parse_args(argv)

    # Default to all layers when none specified
    if not args.layers:
        args.layers = LAYERS[:]

    return args


def compute_composite(
    synthesis_recall: float,
    temporal_accuracy: float,
    chat_extraction_precision: float,
) -> float:
    """Compute composite memory eval score.

    Frozen formula: 0.4 * synthesis_recall + 0.3 * temporal_accuracy + 0.3 * chat_extraction_precision
    """
    return (
        0.4 * synthesis_recall
        + 0.3 * temporal_accuracy
        + 0.3 * chat_extraction_precision
    )


def compute_composite_without_chat(
    synthesis_recall: float,
    temporal_accuracy: float,
) -> float:
    """Compute composite when chat_extraction is unavailable.

    Reweighted formula: 0.55 * synthesis_recall + 0.45 * temporal_accuracy
    """
    return 0.55 * synthesis_recall + 0.45 * temporal_accuracy


def check_api_health() -> bool:
    """Check if the local dev API server is reachable.

    Returns True if healthy, False otherwise.
    """
    import requests
    try:
        resp = requests.get("http://localhost:8787/health", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


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
    args = parse_args()
    live = args.live
    max_samples = args.locomo_samples
    layers = args.layers
    json_output = args.json_output

    ensure_scenarios()

    if json_output:
        synthesis_metrics: dict = {}
        temporal_metrics: dict = {}
        chat_metrics: dict = {}

        if "synthesis" in layers:
            print("\n>>> Running synthesis assessment...")
            from .scenarios import load_scenarios
            from .eval_synthesis import run_synthesis_assessment, print_results as print_synthesis
            scenarios = load_scenarios(SCENARIOS_PATH)
            results = run_synthesis_assessment(scenarios, live=live)
            print_synthesis(results)
            valid = [r for r in results if r.raw_output != "[NOT CACHED]" and r.json_parsed]
            if valid:
                synthesis_metrics = {
                    "synthesis_recall": sum(r.new_fact_recall for r in valid) / len(valid),
                    "synthesis_precision": sum(r.new_fact_precision for r in valid) / len(valid),
                    "n": len(valid),
                }

        if "temporal" in layers:
            print("\n>>> Running temporal reasoning assessment...")
            from .scenarios import load_scenarios
            from .eval_temporal import run_temporal_assessment, print_results as print_temporal
            scenarios = load_scenarios(SCENARIOS_PATH)
            results = run_temporal_assessment(scenarios, live=live)
            print_temporal(results)
            if results:
                temporal_metrics = {
                    "temporal_assertion_accuracy": sum(1 for r in results if r.correct) / len(results),
                    "n": len(results),
                }

        if "chat_extraction" in layers:
            ensure_chat_scenarios()
            if live and not check_api_health():
                print(
                    f"\n  WARNING: Local dev API not reachable at {API_BASE}. "
                    "Skipping chat_extraction layer."
                )
            else:
                print("\n>>> Running chat extraction assessment...")
                from .scenarios import load_chat_scenarios
                from .eval_chat_extraction import run_chat_extraction_assessment, print_results as print_chat
                if CHAT_SCENARIOS_PATH.exists():
                    scenarios = load_chat_scenarios(CHAT_SCENARIOS_PATH)
                    results = run_chat_extraction_assessment(scenarios, live=live)
                    print_chat(results, scenarios=scenarios)
                    valid = [
                        r for r in results
                        if not (r.raw_outputs and r.raw_outputs[0].get("status") == "[NOT CACHED]")
                    ]
                    if valid:
                        chat_metrics = {
                            "chat_extraction_precision": sum(r.extraction_precision for r in valid) / len(valid),
                            "chat_extraction_recall": sum(r.extraction_recall for r in valid) / len(valid),
                            "n": len(valid),
                        }

        # Run non-composite layers in their original mode
        for layer in layers:
            if layer in ("synthesis", "temporal", "chat_extraction"):
                continue
            if layer == "retrieval":
                run_retrieval()
            elif layer == "downstream":
                run_downstream(live)
            elif layer == "locomo":
                run_locomo(live, max_samples)
            elif layer == "report":
                run_report()

        # Build and emit JSON
        output: dict = {}

        if synthesis_metrics:
            output["synthesis_recall"] = synthesis_metrics["synthesis_recall"]
            output["synthesis_precision"] = synthesis_metrics["synthesis_precision"]
            output["synthesis_n"] = synthesis_metrics["n"]

        if temporal_metrics:
            output["temporal_assertion_accuracy"] = temporal_metrics["temporal_assertion_accuracy"]
            output["temporal_n"] = temporal_metrics["n"]

        if chat_metrics:
            output["chat_extraction_precision"] = chat_metrics["chat_extraction_precision"]
            output["chat_extraction_recall"] = chat_metrics["chat_extraction_recall"]
            output["chat_extraction_n"] = chat_metrics["n"]

        synth_recall = synthesis_metrics.get("synthesis_recall")
        temp_acc = temporal_metrics.get("temporal_assertion_accuracy")
        chat_prec = chat_metrics.get("chat_extraction_precision")

        if synth_recall is not None and temp_acc is not None and chat_prec is not None:
            output["composite"] = compute_composite(synth_recall, temp_acc, chat_prec)
            output["composite_formula"] = (
                "0.4*synthesis_recall + 0.3*temporal_accuracy + 0.3*chat_extraction_precision"
            )
        elif synth_recall is not None and temp_acc is not None:
            output["composite"] = compute_composite_without_chat(synth_recall, temp_acc)
            output["composite_formula"] = (
                "0.55*synthesis_recall + 0.45*temporal_accuracy (chat unavailable)"
            )

        print("\n" + "=" * 60)
        print("  JSON OUTPUT")
        print("=" * 60)
        print(json.dumps(output, indent=2))

    else:
        # Non-JSON mode: original behavior
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
