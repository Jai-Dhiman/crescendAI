"""Observation quality evaluation.

Runs YouTube recordings through the full pipeline via wrangler dev,
then scores each observation with the LLM judge on 5 binary criteria.
"""

from __future__ import annotations

import asyncio
import statistics
from pathlib import Path

from shared.inference_cache import find_cache_dir, load_all_recordings, validate_cache
from shared.judge import judge_observation
from shared.pipeline_client import run_recording, SessionResult
from shared.reporting import EvalReport, MetricResult
from shared.traces import PipelineTrace

CRITERIA = [
    "Musical Accuracy",
    "Specificity",
    "Actionability",
    "Tone",
    "Dimension Appropriateness",
]

PASS_THRESHOLDS = {
    "Musical Accuracy": 0.70,
    "Specificity": 0.60,
    "Actionability": 0.60,
    "Tone": 0.80,
    "Dimension Appropriateness": 0.60,
}


def main(
    cache_dir: Path,
    traces_dir: Path,
    reports_dir: Path,
    wrangler_url: str,
) -> EvalReport:
    """Run the observation quality eval."""
    versioned_cache = find_cache_dir(cache_dir)
    if not versioned_cache:
        raise FileNotFoundError(f"No inference cache found in {cache_dir}")

    fingerprint = validate_cache(versioned_cache)
    recordings_map = load_all_recordings(versioned_cache)
    recordings = list(recordings_map.values())
    print(f"Loaded {len(recordings)} recordings from cache ({fingerprint})")

    all_scores: dict[str, list[bool]] = {c: [] for c in CRITERIA}
    worst_cases: list[dict] = []
    total_judge_calls = 0
    total_observations = 0
    no_observation_count = 0

    for i, recording in enumerate(recordings):
        recording_id = recording["recording_id"]
        print(f"  [{i+1}/{len(recordings)}] {recording_id}...", end=" ", flush=True)

        result: SessionResult = asyncio.run(run_recording(wrangler_url, recording))

        if result.errors:
            print(f"ERRORS: {result.errors}")
            continue

        if not result.observations:
            print("no observations")
            no_observation_count += 1
            continue

        total_observations += len(result.observations)

        for obs in result.observations:
            # Use eval_context echoed back by the Rust pipeline
            eval_ctx = obs.raw_message.get("eval_context", {})
            context = {
                "predictions": eval_ctx.get("predictions", {}),
                "baselines": eval_ctx.get("baselines", {}),
                "recent_observations": eval_ctx.get("recent_observations", []),
                "analysis_facts": eval_ctx.get("analysis_facts", {}),
                "piece_name": eval_ctx.get("piece_name", recording_id),
                "bar_range": eval_ctx.get("bar_range", "full recording"),
            }

            judge_result = judge_observation(obs.text, context)
            total_judge_calls += 1

            trace = PipelineTrace(
                observation_id=f"{recording_id}_chunk{obs.chunk_index}",
                recording_id=recording_id,
                chunk_index=obs.chunk_index,
                inference=recording["chunks"][obs.chunk_index]
                    if obs.chunk_index < len(recording["chunks"]) else {},
                teacher_observation=obs.text,
                judge_scores=[
                    {"criterion": s.criterion, "passed": s.passed, "evidence": s.evidence}
                    for s in judge_result.scores
                ],
            )
            trace.save(traces_dir)

            for score in judge_result.scores:
                if score.criterion in all_scores and score.passed is not None:
                    all_scores[score.criterion].append(score.passed)
                    if not score.passed:
                        worst_cases.append({
                            "recording_id": recording_id,
                            "chunk_index": obs.chunk_index,
                            "criterion": score.criterion,
                            "observation": obs.text[:200],
                            "evidence": score.evidence[:200],
                            "trace_file": f"{recording_id}_chunk{obs.chunk_index}.json",
                        })

        print(f"{len(result.observations)} obs, {result.duration_ms}ms")

    report = EvalReport(
        eval_name="observation_quality",
        eval_version="1.0",
        dataset=f"youtube_amt_{len(recordings)}",
        metrics={},
    )
    report.metadata["model_fingerprint"] = fingerprint
    report.metadata["total_observations"] = total_observations
    report.metadata["no_observation_recordings"] = no_observation_count

    for criterion in CRITERIA:
        scores = all_scores[criterion]
        if scores:
            mean = sum(scores) / len(scores)
            std = statistics.stdev(scores) if len(scores) > 1 else 0.0
            report.metrics[criterion] = MetricResult(
                mean=mean, std=std, n=len(scores),
                pass_threshold=PASS_THRESHOLDS.get(criterion),
            )

    report.worst_cases = sorted(worst_cases, key=lambda x: x["criterion"])[:20]
    report.cost = {
        "judge_calls": total_judge_calls,
        "estimated_usd": total_judge_calls * 0.003,
    }

    reports_dir.mkdir(parents=True, exist_ok=True)
    report.save(reports_dir / "observation_quality.json")
    report.print_summary()
    return report
