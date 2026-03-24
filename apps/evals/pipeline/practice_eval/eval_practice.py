"""Practice recording evaluation.

Runs practice recordings through the full pipeline (wrangler dev),
judges each observation with derived criteria (v2/v3 judge), produces
a segmented report by tier, framing, and skill level.

Supports T5 scenarios (--scenarios t5), per-recording checkpointing,
STOP/piece-ID metric extraction, and judge v3 prompt selection.

Requires wrangler dev running at localhost:8787.

Usage:
    cd apps/evals/
    uv run python -m pipeline.practice_eval.eval_practice
    uv run python -m pipeline.practice_eval.eval_practice --piece fur_elise
    uv run python -m pipeline.practice_eval.eval_practice --scenarios t5
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

JUDGE_PROMPT_V2 = "observation_quality_judge_v2.txt"
JUDGE_PROMPT_V3 = "observation_quality_judge_v3.txt"


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


def _resolve_pieces(args) -> list[tuple[str, str]]:
    """Resolve piece list from args.

    Returns list of (scenario_name, piece_id) tuples.
    scenario_name is used to load the YAML file, piece_id is the canonical piece.
    """
    if args.scenarios:
        prefix = args.scenarios
        matches = sorted(SCENARIOS_DIR.glob(f"{prefix}_*.yaml"))
        if not matches:
            raise FileNotFoundError(
                f"No scenario files matching {prefix}_*.yaml in {SCENARIOS_DIR}"
            )
        results = []
        for m in matches:
            scenario_name = m.stem  # e.g. "t5_fur_elise"
            piece_id = scenario_name[len(prefix) + 1:]  # e.g. "fur_elise"
            results.append((scenario_name, piece_id))
        return results

    if args.piece != "all":
        return [(args.piece, args.piece)]

    return [("fur_elise", "fur_elise"), ("nocturne_op9no2", "nocturne_op9no2")]


def main():
    parser = argparse.ArgumentParser(description="Run practice recording eval")
    parser.add_argument("--wrangler-url", default="http://localhost:8787")
    parser.add_argument("--piece", default="all", choices=["fur_elise", "nocturne_op9no2", "all"])
    parser.add_argument("--scenarios", default=None,
                        help="Scenario prefix (e.g., 't5' matches t5_*.yaml)")
    args = parser.parse_args()

    pieces = _resolve_pieces(args)

    all_scores: dict[str, list[bool]] = {}
    all_observations: list[dict] = []
    total_judge_calls = 0
    total_recordings = 0
    recordings_with_obs = 0
    tier_counts: dict[str, int] = {}
    framing_counts: dict[str, int] = {}
    stop_probabilities: list[tuple[float, int]] = []
    piece_id_results: list[dict] = []

    # Per-recording checkpointing
    checkpoint_path = REPORTS_DIR / ".eval_checkpoint.json"
    completed_ids: set[str] = set()
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        completed_ids = set(ckpt.get("completed", []))
        all_observations = ckpt.get("observations", [])
        total_recordings = ckpt.get("total_recordings", 0)
        recordings_with_obs = ckpt.get("recordings_with_obs", 0)
        total_judge_calls = ckpt.get("total_judge_calls", 0)
        stop_probabilities = [(p, l) for p, l in ckpt.get("stop_probabilities", [])]
        piece_id_results = ckpt.get("piece_id_results", [])
        print(f"Resuming from checkpoint: {len(completed_ids)} recordings done")

    print("Loading inference cache...")
    cache = find_inference_cache()
    print(f"  {len(cache)} recordings in cache")

    for scenario_name, piece_id in pieces:
        scenarios = load_scenarios(scenario_name)
        print(f"\n=== {scenario_name}: {len(scenarios)} practice recordings ===")

        for i, scenario in enumerate(scenarios):
            video_id = scenario["video_id"]
            total_recordings += 1
            print(f"  [{i+1}/{len(scenarios)}] {video_id}...", end=" ", flush=True)

            if video_id in completed_ids:
                print("checkpointed, skipping")
                continue

            if video_id not in cache:
                print("not in cache, skipping")
                completed_ids.add(video_id)
                continue

            recording = cache[video_id]
            result: SessionResult = asyncio.run(
                run_recording(
                    args.wrangler_url,
                    recording,
                    piece_query=scenario.get("piece_query"),
                )
            )

            # Track piece identification (only when no piece_query, i.e. zero-config)
            if not scenario.get("piece_query") and result.piece_identification:
                piece_id_results.append({
                    "expected": piece_id,
                    "actual": result.piece_identification.piece_id,
                    "confidence": result.piece_identification.confidence,
                    "notes_consumed": result.piece_identification.notes_consumed,
                    "correct": piece_id == result.piece_identification.piece_id,
                })

            if result.errors:
                print(f"ERRORS: {result.errors}")
                completed_ids.add(video_id)
                continue

            if not result.observations:
                print("no observations (STOP did not trigger)")
                completed_ids.add(video_id)
                continue

            recordings_with_obs += 1
            skill_level = scenario.get("skill_level", 0)

            for obs in result.observations:
                eval_ctx = obs.raw_message.get("eval_context", {})
                tier = str(eval_ctx.get("tier", "3"))
                framing = obs.framing or "unknown"
                tier_counts[tier] = tier_counts.get(tier, 0) + 1
                framing_counts[framing] = framing_counts.get(framing, 0) + 1

                # Extract STOP data before building context dict
                teaching_moment = eval_ctx.get("teaching_moment", {})
                if isinstance(teaching_moment, str):
                    teaching_moment = json.loads(teaching_moment) if teaching_moment else {}
                stop_prob = teaching_moment.get("stop_probability")
                if stop_prob is not None:
                    stop_probabilities.append((stop_prob, skill_level))

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

                # Select judge prompt based on skill_level
                prompt = JUDGE_PROMPT_V3 if skill_level > 0 else JUDGE_PROMPT_V2
                context["skill_level"] = str(skill_level) if skill_level > 0 else "unknown"
                judge_result = judge_observation(obs.text, context, prompt_file=prompt)
                total_judge_calls += 1

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

            # Save checkpoint after each recording
            completed_ids.add(video_id)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with open(checkpoint_path, "w") as f:
                json.dump({
                    "completed": list(completed_ids),
                    "observations": all_observations,
                    "total_recordings": total_recordings,
                    "recordings_with_obs": recordings_with_obs,
                    "total_judge_calls": total_judge_calls,
                    "stop_probabilities": stop_probabilities,
                    "piece_id_results": piece_id_results,
                }, f)

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

    # STOP metrics
    if stop_probabilities:
        from scipy.stats import spearmanr
        from collections import defaultdict
        probs = [p for p, _ in stop_probabilities]
        levels = [l for _, l in stop_probabilities]
        rho, p_value = spearmanr(probs, levels)
        report.metadata["stop_probability_skill_correlation"] = {
            "spearman_rho": round(rho, 4),
            "p_value": round(p_value, 4),
            "n": len(stop_probabilities),
        }
        bucket_triggers = defaultdict(list)
        bucket_probs = defaultdict(list)
        for prob, level in stop_probabilities:
            bucket_triggers[level].append(prob >= 0.5)
            bucket_probs[level].append(prob)
        report.metadata["stop_trigger_rate_by_bucket"] = {
            str(k): {"rate": round(sum(v)/len(v), 3), "n": len(v)}
            for k, v in sorted(bucket_triggers.items())
        }
        report.metadata["stop_probabilities_by_bucket"] = {
            str(k): [round(p, 4) for p in v]
            for k, v in sorted(bucket_probs.items())
        }

    # Piece ID metrics
    if piece_id_results:
        correct = sum(1 for r in piece_id_results if r["correct"])
        report.metadata["piece_id"] = {
            "top1_accuracy": round(correct / len(piece_id_results), 3),
            "total": len(piece_id_results),
            "correct": correct,
            "mean_notes_to_identify": round(
                sum(r["notes_consumed"] for r in piece_id_results) / len(piece_id_results)
            ),
            "false_positives": sum(
                1 for r in piece_id_results
                if not r["correct"] and r["confidence"] > 0.8
            ),
        }

    # Inference cache fingerprints
    if INFERENCE_CACHE_BASE.exists():
        cache_subdirs = sorted(d.name for d in INFERENCE_CACHE_BASE.iterdir() if d.is_dir())
        report.metadata["inference_cache_fingerprints"] = cache_subdirs

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report.save(REPORTS_DIR / "practice_eval.json")
    report.print_summary()

    obs_path = REPORTS_DIR / "practice_eval_observations.json"
    with open(obs_path, "w") as f:
        json.dump(all_observations, f, indent=2)
    print(f"  Detailed observations: {obs_path}")

    # Clean up checkpoint after successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()


if __name__ == "__main__":
    main()
