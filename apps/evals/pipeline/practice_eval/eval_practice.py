"""Practice recording evaluation.

Runs practice recordings through the full pipeline (wrangler dev),
judges each observation with derived criteria (v2 judge), produces
a segmented report by tier, framing, and skill level.

Requires wrangler dev running at localhost:8787.

Usage:
    cd apps/evals/
    uv run python -m pipeline.practice_eval.eval_practice
    uv run python -m pipeline.practice_eval.eval_practice --piece fur_elise
"""

from __future__ import annotations

import asyncio
import argparse
import json
import statistics
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parents[2]))

from paths import MODEL_DATA
from shared.judge import judge_observation
from shared.pipeline_client import run_recording, SessionResult
from shared.reporting import EvalReport, MetricResult

SCENARIOS_DIR = Path(__file__).parent / "scenarios"
REPORTS_DIR = Path(__file__).parents[2] / "reports"

# eval_runner.py caches to model/data/eval/inference_cache/ (singular "eval")
INFERENCE_CACHE_BASE = MODEL_DATA / "eval" / "inference_cache"

JUDGE_PROMPT = "observation_quality_judge_v2.txt"


def load_scenarios(piece_id: str) -> list[dict]:
    """Load included scenarios for a piece."""
    path = SCENARIOS_DIR / f"{piece_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No scenarios at {path}")
    with open(path) as f:
        data = yaml.safe_load(f)
    included = [c for c in data.get("candidates", []) if c.get("include")]
    for c in included:
        c["piece_query"] = data.get("piece_query", "")
    return included


def find_inference_cache() -> dict[str, dict]:
    """Find and load the most recent inference cache."""
    if not INFERENCE_CACHE_BASE.exists():
        raise FileNotFoundError(
            f"No inference cache at {INFERENCE_CACHE_BASE}. Run eval_runner.py first."
        )
    cache_dirs = sorted(d for d in INFERENCE_CACHE_BASE.iterdir() if d.is_dir())
    if not cache_dirs:
        raise FileNotFoundError("Inference cache directory is empty.")

    recordings: dict[str, dict] = {}
    for cache_dir in cache_dirs:
        for json_file in cache_dir.glob("*.json"):
            with open(json_file) as f:
                data = json.load(f)
            rec_id = data.get("recording_id", json_file.stem)
            recordings[rec_id] = data
    return recordings


def main():
    parser = argparse.ArgumentParser(description="Run practice recording eval")
    parser.add_argument("--wrangler-url", default="http://localhost:8787")
    parser.add_argument("--piece", default="all", choices=["fur_elise", "nocturne_op9no2", "all"])
    args = parser.parse_args()

    pieces = ["fur_elise", "nocturne_op9no2"] if args.piece == "all" else [args.piece]

    all_scores: dict[str, list[bool]] = {}
    all_observations: list[dict] = []
    total_judge_calls = 0
    total_recordings = 0
    recordings_with_obs = 0
    tier_counts: dict[str, int] = {}
    framing_counts: dict[str, int] = {}

    print("Loading inference cache...")
    cache = find_inference_cache()
    print(f"  {len(cache)} recordings in cache")

    for piece_id in pieces:
        scenarios = load_scenarios(piece_id)
        print(f"\n=== {piece_id}: {len(scenarios)} practice recordings ===")

        for i, scenario in enumerate(scenarios):
            video_id = scenario["video_id"]
            total_recordings += 1
            print(f"  [{i+1}/{len(scenarios)}] {video_id}...", end=" ", flush=True)

            if video_id not in cache:
                print("not in cache, skipping")
                continue

            recording = cache[video_id]
            result: SessionResult = asyncio.run(
                run_recording(
                    args.wrangler_url,
                    recording,
                    piece_query=scenario.get("piece_query"),
                )
            )

            if result.errors:
                print(f"ERRORS: {result.errors}")
                continue

            if not result.observations:
                print("no observations (STOP did not trigger)")
                continue

            recordings_with_obs += 1

            for obs in result.observations:
                eval_ctx = obs.raw_message.get("eval_context", {})
                tier = str(eval_ctx.get("tier", "3"))
                framing = obs.framing or "unknown"
                tier_counts[tier] = tier_counts.get(tier, 0) + 1
                framing_counts[framing] = framing_counts.get(framing, 0) + 1

                context = {
                    "predictions": eval_ctx.get("predictions", {}),
                    "baselines": eval_ctx.get("baselines", {}),
                    "recent_observations": eval_ctx.get("recent_observations", []),
                    "analysis_facts": eval_ctx.get("analysis_facts", {}),
                    "piece_name": eval_ctx.get("piece_name", scenario.get("title", video_id)),
                    "bar_range": eval_ctx.get("bar_range", "full recording"),
                    "teaching_moment": json.dumps(eval_ctx.get("teaching_moment", {})),
                    "subagent_output": json.dumps(eval_ctx.get("subagent_output", {})),
                    "scenario_notes": scenario.get("general_notes", ""),
                }

                judge_result = judge_observation(obs.text, context, prompt_file=JUDGE_PROMPT)
                total_judge_calls += 1

                skill_level = scenario.get("skill_level", 0)
                for score in judge_result.scores:
                    if score.passed is not None:
                        all_scores.setdefault(score.criterion, []).append(score.passed)

                all_observations.append({
                    "video_id": video_id,
                    "piece": piece_id,
                    "skill_level": skill_level,
                    "tier": tier,
                    "framing": framing,
                    "observation": obs.text[:300],
                    "judge_scores": {
                        s.criterion: s.passed for s in judge_result.scores
                    },
                    "judge_evidence": {
                        s.criterion: s.evidence for s in judge_result.scores
                    },
                })

            print(f"{len(result.observations)} obs")

    # Build report
    report = EvalReport(
        eval_name="practice_eval",
        eval_version="2.0",
        dataset=f"practice_{total_recordings}",
        metrics={},
    )
    for criterion, scores in all_scores.items():
        if scores:
            mean = sum(scores) / len(scores)
            std = statistics.stdev(scores) if len(scores) > 1 else 0.0
            report.metrics[criterion] = MetricResult(mean=mean, std=std, n=len(scores))

    report.metadata["total_recordings"] = total_recordings
    report.metadata["recordings_with_observations"] = recordings_with_obs
    report.metadata["stop_trigger_rate"] = (
        recordings_with_obs / total_recordings if total_recordings > 0 else 0
    )
    report.metadata["tier_distribution"] = tier_counts
    report.metadata["framing_distribution"] = framing_counts
    report.metadata["total_observations"] = len(all_observations)
    report.cost = {
        "judge_calls": total_judge_calls,
        "estimated_usd": round(total_judge_calls * 0.003, 2),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report.save(REPORTS_DIR / "practice_eval.json")
    report.print_summary()

    obs_path = REPORTS_DIR / "practice_eval_observations.json"
    with open(obs_path, "w") as f:
        json.dump(all_observations, f, indent=2)
    print(f"  Detailed observations: {obs_path}")


if __name__ == "__main__":
    main()
