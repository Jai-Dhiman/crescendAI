"""Human rating CLI for downstream A/B comparisons.

Follows the teacher_voice/rate.py pattern. Presents scenario + both responses
side by side (randomized order). Rates on 5 axes.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parents[2] / "data" / "memory_eval"

AXES = ["continuity", "specificity", "non_repetition", "approach_fit", "accuracy"]

AXIS_DESCRIPTIONS = {
    "continuity": "Awareness of student's history and patterns",
    "specificity": "References specific moments, dimensions, trends",
    "non_repetition": "Avoids repeating recent feedback",
    "approach_fit": "Framing matches student's engagement patterns",
    "accuracy": "Correct diagnosis given scores and context",
}


def _print_separator() -> None:
    print("\n" + "=" * 70 + "\n")


def _get_rating(prompt: str) -> int | None:
    while True:
        val = input(f"  {prompt} (1-5, s=skip, q=quit): ").strip().lower()
        if val == "q":
            return None
        if val == "s":
            return -1
        try:
            rating = int(val)
            if 1 <= rating <= 5:
                return rating
            print("    Enter 1-5")
        except ValueError:
            print("    Enter 1-5, s, or q")


def rate_downstream(
    downstream_path: Path | None = None,
    output_path: Path | None = None,
) -> None:
    if downstream_path is None:
        downstream_path = DATA_DIR / "downstream_cache.jsonl"
    if output_path is None:
        output_path = DATA_DIR / "human_ratings.jsonl"

    if not downstream_path.exists():
        print(f"No downstream results at {downstream_path}")
        print("Run: uv run python -m src.memory_eval.eval_downstream --live")
        sys.exit(1)

    # Load downstream responses, grouped by scenario
    responses: dict[str, dict] = {}
    with open(downstream_path) as f:
        for line in f:
            entry = json.loads(line)
            key = entry["key"]
            scenario_id = key.rsplit("_", 2)[0]  # strip _no_memory or _with_memory
            if scenario_id not in responses:
                responses[scenario_id] = {}
            if key.endswith("_no_memory"):
                responses[scenario_id]["no_memory"] = entry["response"]
            elif key.endswith("_with_memory"):
                responses[scenario_id]["with_memory"] = entry["response"]

    # Filter to complete pairs
    pairs = {sid: r for sid, r in responses.items() if "no_memory" in r and "with_memory" in r}

    if not pairs:
        print("No complete response pairs found.")
        sys.exit(1)

    # Load existing ratings
    existing_rated: set[str] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                entry = json.loads(line)
                existing_rated.add(entry["scenario_id"])

    unrated = [(sid, r) for sid, r in sorted(pairs.items()) if sid not in existing_rated]

    if not unrated:
        print("All pairs are already rated.")
        _print_summary(output_path)
        return

    print(f"\n{len(unrated)} unrated pairs out of {len(pairs)} total.\n")
    print("Rating scale: 1=Poor 2=Below avg 3=Adequate 4=Good 5=Excellent\n")
    print("For each response pair, rate on 5 axes:")
    for axis, desc in AXIS_DESCRIPTIONS.items():
        print(f"  {axis}: {desc}")

    rng = random.Random(42)
    rated_count = 0

    for idx, (scenario_id, resp) in enumerate(unrated):
        _print_separator()
        print(f"[{idx + 1}/{len(unrated)}] Scenario: {scenario_id}")

        # Randomize A/B order
        flipped = rng.random() < 0.5
        if flipped:
            label_a, resp_a = "with_memory", resp["with_memory"]
            label_b, resp_b = "no_memory", resp["no_memory"]
        else:
            label_a, resp_a = "no_memory", resp["no_memory"]
            label_b, resp_b = "with_memory", resp["with_memory"]

        print(f"\n--- Response A ---\n")
        print(resp_a.strip())
        print(f"\n--- Response B ---\n")
        print(resp_b.strip())
        print()

        # Rate A
        print("  Rating Response A:")
        a_ratings = {}
        quit_flag = False
        skip_flag = False
        for axis in AXES:
            r = _get_rating(f"  A - {axis}")
            if r is None:
                quit_flag = True
                break
            if r == -1:
                skip_flag = True
                break
            a_ratings[axis] = r

        if quit_flag:
            break
        if skip_flag:
            continue

        # Rate B
        print("  Rating Response B:")
        b_ratings = {}
        for axis in AXES:
            r = _get_rating(f"  B - {axis}")
            if r is None:
                quit_flag = True
                break
            if r == -1:
                skip_flag = True
                break
            b_ratings[axis] = r

        if quit_flag:
            break
        if skip_flag:
            continue

        # Preference
        pref = input("  Which is better? (a/b/tie): ").strip().lower()
        if pref not in ("a", "b", "tie"):
            pref = "tie"

        # Save
        if flipped:
            with_mem_ratings = a_ratings
            no_mem_ratings = b_ratings
            human_preferred = {"a": "with_memory", "b": "no_memory", "tie": "tie"}[pref]
        else:
            no_mem_ratings = a_ratings
            with_mem_ratings = b_ratings
            human_preferred = {"a": "no_memory", "b": "with_memory", "tie": "tie"}[pref]

        rating_entry = {
            "scenario_id": scenario_id,
            "flipped": flipped,
            "no_memory_ratings": no_mem_ratings,
            "with_memory_ratings": with_mem_ratings,
            "preferred": human_preferred,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "a") as f:
            f.write(json.dumps(rating_entry, ensure_ascii=False) + "\n")

        rated_count += 1

    print(f"\nRated {rated_count} pairs. Saved to {output_path}")
    _print_summary(output_path)


def _print_summary(output_path: Path) -> None:
    if not output_path.exists():
        return

    ratings = []
    with open(output_path) as f:
        for line in f:
            ratings.append(json.loads(line))

    if not ratings:
        return

    print(f"\n--- Human Rating Summary (n={len(ratings)}) ---\n")

    with_wins = sum(1 for r in ratings if r["preferred"] == "with_memory")
    no_wins = sum(1 for r in ratings if r["preferred"] == "no_memory")
    ties = sum(1 for r in ratings if r["preferred"] == "tie")

    print(f"  Preferences: with_memory={with_wins}, no_memory={no_wins}, tie={ties}")

    for axis in AXES:
        with_vals = [r["with_memory_ratings"][axis] for r in ratings if axis in r.get("with_memory_ratings", {})]
        no_vals = [r["no_memory_ratings"][axis] for r in ratings if axis in r.get("no_memory_ratings", {})]
        if with_vals and no_vals:
            with_avg = sum(with_vals) / len(with_vals)
            no_avg = sum(no_vals) / len(no_vals)
            delta = with_avg - no_avg
            print(f"  {axis}: with={with_avg:.2f} no={no_avg:.2f} delta={delta:+.2f}")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        _print_summary(DATA_DIR / "human_ratings.jsonl")
    else:
        rate_downstream()


if __name__ == "__main__":
    main()
