"""CLI tool for rating benchmark outputs.

Presents each (scenario, claude_response) pair and prompts for ratings
on accuracy, actionability, and voice quality.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

def _print_separator() -> None:
    print("\n" + "=" * 70 + "\n")


def _get_rating(prompt: str) -> int | None:
    """Prompt for a 1-5 rating. Returns None to skip."""
    while True:
        val = input(f"  {prompt} (1-5, s=skip, q=quit): ").strip().lower()
        if val == "q":
            return None
        if val == "s":
            return -1  # sentinel for skip
        try:
            rating = int(val)
            if 1 <= rating <= 5:
                return rating
            print("    Enter 1-5")
        except ValueError:
            print("    Enter 1-5, s, or q")


def rate_results(results_path: Path) -> None:
    """Interactive rating loop for benchmark results."""
    if not results_path.exists():
        print(f"No results file at {results_path}")
        sys.exit(1)

    # Load all results
    results: list[dict] = []
    with open(results_path) as f:
        for line in f:
            results.append(json.loads(line))

    # Find unrated results
    unrated = [
        (i, r) for i, r in enumerate(results)
        if r.get("accuracy") is None
    ]

    if not unrated:
        print("All results are already rated.")
        return

    print(f"\n{len(unrated)} unrated results out of {len(results)} total.\n")
    print("Rating scale:")
    print("  1 = Poor / Wrong")
    print("  2 = Below average")
    print("  3 = Adequate")
    print("  4 = Good")
    print("  5 = Excellent")
    print("\nFor each response, rate on three axes:")
    print("  Accuracy: Does it correctly identify the musical issue?")
    print("  Actionability: Does it tell the student what to do?")
    print("  Voice: Does it sound like a real piano teacher?")

    rated_count = 0

    for idx, (orig_idx, result) in enumerate(unrated):
        _print_separator()
        print(f"[{idx + 1}/{len(unrated)}] Variant: {result['variant']}")
        print(f"Record: {result['record_id'][:12]}...")
        print(f"\n--- SCENARIO (what the LLM was told) ---\n")

        # Show a condensed version of the prompt
        prompt = result["prompt"]
        # Skip the "What to say" instruction part for readability
        display = prompt.split("## What to say")[0] if "## What to say" in prompt else prompt
        print(display.strip())

        print(f"\n--- CLAUDE'S RESPONSE ---\n")
        print(result["response"])
        print()

        accuracy = _get_rating("Accuracy (correct diagnosis?)")
        if accuracy is None:
            break
        if accuracy == -1:
            continue

        actionability = _get_rating("Actionability (tells student what to do?)")
        if actionability is None:
            break
        if actionability == -1:
            continue

        voice = _get_rating("Voice (sounds like a teacher?)")
        if voice is None:
            break
        if voice == -1:
            continue

        results[orig_idx]["accuracy"] = accuracy
        results[orig_idx]["actionability"] = actionability
        results[orig_idx]["voice"] = voice
        rated_count += 1

    # Save back
    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nRated {rated_count} results. Saved to {results_path}")


def print_summary(results_path: Path) -> None:
    """Print summary statistics of rated results."""
    results: list[dict] = []
    with open(results_path) as f:
        for line in f:
            results.append(json.loads(line))

    rated = [r for r in results if r.get("accuracy") is not None]
    if not rated:
        print("No rated results yet.")
        return

    print(f"\nRated: {len(rated)} / {len(results)} results\n")

    for variant in ["bare", "rich", "retrieved"]:
        variant_results = [r for r in rated if r["variant"] == variant]
        if not variant_results:
            continue

        n = len(variant_results)
        avg_acc = sum(r["accuracy"] for r in variant_results) / n
        avg_act = sum(r["actionability"] for r in variant_results) / n
        avg_voice = sum(r["voice"] for r in variant_results) / n

        print(f"  {variant} (n={n}):")
        print(f"    Accuracy:      {avg_acc:.2f}")
        print(f"    Actionability: {avg_act:.2f}")
        print(f"    Voice:         {avg_voice:.2f}")
        print()


if __name__ == "__main__":
    from src.paths import Results

    results_path = Results.root / "teacher_voice" / "benchmark_results.jsonl"

    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        print_summary(results_path)
    else:
        rate_results(results_path)
