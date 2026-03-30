"""Synthesis pipeline evaluation.

Runs practice recordings through the full synthesis pipeline (wrangler dev),
captures synthesis output + accumulator state, evaluates all 7 capabilities:
piece ID, STOP, teaching moments, mode detection, synthesis quality,
exercises, score following.

Two-pass design: Pass A (with piece_query) tests synthesis with full context,
Pass B (without) tests zero-config piece ID and its impact on quality.

Requires wrangler dev running at localhost:8787.

Usage:
    cd apps/evals/
    uv run python -m pipeline.practice_eval.eval_practice --scenarios t5
    uv run python -m pipeline.practice_eval.eval_practice --scenarios t5 --capability stop,synthesis
"""

from __future__ import annotations

import asyncio
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parents[2]))

from paths import MODEL_DATA
from shared.judge import judge_synthesis, judge_teaching_moment, judge_differentiation
from shared.pipeline_client import run_recording, SessionResult
from shared.reporting import EvalReport, MetricResult

SCENARIOS_DIR = Path(__file__).parent / "scenarios"
REPORTS_DIR = Path(__file__).parents[2] / "reports"

INFERENCE_CACHE_BASE = MODEL_DATA / "eval" / "inference_cache"

# STOP classifier constants loaded from shared config
_STOP_CONFIG_PATH = Path(__file__).resolve().parents[4] / "apps" / "config" / "stop_config.json"
if not _STOP_CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"STOP config not found at {_STOP_CONFIG_PATH}. "
        "Run from the project root or check apps/config/stop_config.json exists."
    )
with open(_STOP_CONFIG_PATH) as _f:
    _STOP_CONFIG = json.load(_f)
STOP_SCALER_MEAN = _STOP_CONFIG["scaler_mean"]
STOP_SCALER_STD = _STOP_CONFIG["scaler_std"]
STOP_WEIGHTS = _STOP_CONFIG["weights"]
STOP_BIAS = _STOP_CONFIG["bias"]
STOP_THRESHOLD = _STOP_CONFIG["threshold"]
DIMS_6 = _STOP_CONFIG["dimensions"]

ALL_CAPABILITIES = {"stop", "piece_id", "teaching_moments", "mode_detection",
                    "synthesis", "exercises", "score_following", "differentiation"}


def stop_probability(scores: list[float]) -> float:
    """Compute STOP probability offline (mirrors Rust implementation)."""
    logit = sum(
        ((s - m) / std) * w
        for s, m, std, w in zip(scores, STOP_SCALER_MEAN, STOP_SCALER_STD, STOP_WEIGHTS)
    ) + STOP_BIAS
    return 1.0 / (1.0 + math.exp(-logit))


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
            if json_file.name.startswith("_"):
                continue
            with open(json_file) as f:
                data = json.load(f)
            rec_id = data.get("recording_id", json_file.stem)
            recordings[rec_id] = data
    return recordings


def _resolve_pieces(args) -> list[tuple[str, str]]:
    """Resolve piece list from args. Returns (scenario_name, piece_id) tuples."""
    if args.scenarios:
        prefix = args.scenarios
        matches = sorted(SCENARIOS_DIR.glob(f"{prefix}_*.yaml"))
        if not matches:
            raise FileNotFoundError(
                f"No scenario files matching {prefix}_*.yaml in {SCENARIOS_DIR}"
            )
        results = []
        for m in matches:
            scenario_name = m.stem
            piece_id = scenario_name[len(prefix) + 1:]
            results.append((scenario_name, piece_id))
        return results

    if args.piece != "all":
        return [(args.piece, args.piece)]

    return [("fur_elise", "fur_elise"), ("nocturne_op9no2", "nocturne_op9no2")]


# ---------------------------------------------------------------------------
# Statistical metric extraction (no LLM cost)
# ---------------------------------------------------------------------------

def extract_stop_metrics(eval_context: dict, skill_level: int) -> list[tuple[float, int]]:
    """Extract per-chunk STOP probabilities from scored_chunks in eval_context."""
    scored_chunks = eval_context.get("scored_chunks", [])
    results = []
    for chunk in scored_chunks:
        scores = chunk.get("scores", [0.0] * 6)
        if len(scores) == 6:
            prob = stop_probability(scores)
            results.append((prob, skill_level))
    return results


def extract_teaching_moment_metrics(eval_context: dict) -> dict:
    """Extract teaching moment selection metrics from accumulator state."""
    moments = eval_context.get("teaching_moments", [])
    baselines = eval_context.get("baselines", {})
    if not moments:
        return {}

    metrics: dict = {
        "moment_count": len(moments),
        "dimensions": [m.get("dimension", "") for m in moments],
        "deviations": [m.get("deviation", 0.0) for m in moments],
        "positive_count": sum(1 for m in moments if m.get("is_positive", False)),
    }

    # Blind-spot detection: are selected moments in the bottom 2 by deviation?
    if baselines and len(moments) > 0:
        positive_valid = sum(
            1 for m in moments
            if m.get("is_positive") and m.get("deviation", 0.0) > 0.15
        )
        positive_total = sum(1 for m in moments if m.get("is_positive"))
        metrics["positive_validity"] = (
            positive_valid / positive_total if positive_total > 0 else None
        )

    # Framing-deviation alignment
    framing_aligned = 0
    framing_total = 0
    for m in moments:
        dev = m.get("deviation", 0.0)
        is_positive = m.get("is_positive", False)
        if abs(dev) > 0.05:
            framing_total += 1
            if (is_positive and dev > 0) or (not is_positive and dev < 0):
                framing_aligned += 1
    metrics["framing_alignment"] = (
        framing_aligned / framing_total if framing_total > 0 else None
    )

    # Dedup: unique dimensions
    dims = [m.get("dimension", "") for m in moments]
    metrics["unique_dimensions"] = len(set(dims))
    metrics["dimension_diversity"] = len(set(dims)) / len(dims) if dims else 0

    return metrics


def extract_mode_detection_metrics(eval_context: dict) -> dict:
    """Extract practice mode detection metrics from accumulator state."""
    transitions = eval_context.get("mode_transitions", [])
    drilling_records = eval_context.get("drilling_records", [])
    timeline = eval_context.get("timeline", [])

    metrics: dict = {
        "transition_count": len(transitions),
        "drilling_episodes": len(drilling_records),
        "total_chunks": len(timeline),
        "chunks_with_audio": sum(1 for t in timeline if t.get("has_audio", False)),
    }

    # Spurious transitions: reversals within 2 chunks
    spurious = 0
    for i in range(len(transitions) - 1):
        if transitions[i].get("to") == transitions[i + 1].get("from"):
            chunk_gap = abs(
                transitions[i + 1].get("chunk_index", 0) - transitions[i].get("chunk_index", 0)
            )
            if chunk_gap <= 2:
                spurious += 1
    metrics["spurious_transitions"] = spurious

    return metrics


def extract_score_following_metrics(eval_context: dict) -> dict:
    """Extract score following and bar analysis metrics from accumulator state."""
    moments = eval_context.get("teaching_moments", [])
    if not moments:
        return {}

    bar_ranges = [m.get("bar_range") for m in moments if m.get("bar_range")]
    tiers = [m.get("analysis_tier", 3) for m in moments]

    metrics: dict = {
        "bar_coverage": len(bar_ranges) / len(moments) if moments else 0,
        "tier_distribution": {
            str(t): tiers.count(t) for t in set(tiers)
        },
        "tier_1_rate": tiers.count(1) / len(tiers) if tiers else 0,
    }

    # Bar range plausibility (2-16 bars per chunk is reasonable for 15s)
    plausible = 0
    for br in bar_ranges:
        if isinstance(br, (list, tuple)) and len(br) == 2:
            span = br[1] - br[0]
            if 1 <= span <= 20:
                plausible += 1
    metrics["bar_plausibility"] = plausible / len(bar_ranges) if bar_ranges else None

    return metrics


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run synthesis pipeline eval")
    parser.add_argument("--wrangler-url", default="http://localhost:8787")
    parser.add_argument("--piece", default="all", choices=["fur_elise", "nocturne_op9no2", "all"])
    parser.add_argument("--scenarios", default=None,
                        help="Scenario prefix (e.g., 't5' matches t5_*.yaml)")
    parser.add_argument("--capability", default=None,
                        help="Comma-separated capabilities to judge (e.g., 'stop,synthesis'). "
                             "Statistical metrics always computed. LLM judge only runs for selected.")
    parser.add_argument("--max-recordings", type=int, default=None,
                        help="Limit number of recordings (for testing)")
    args = parser.parse_args()

    # Parse capability filter
    judge_capabilities = ALL_CAPABILITIES
    if args.capability:
        judge_capabilities = set(args.capability.split(","))
        unknown = judge_capabilities - ALL_CAPABILITIES
        if unknown:
            print(f"Unknown capabilities: {unknown}. Valid: {sorted(ALL_CAPABILITIES)}")
            sys.exit(1)

    pieces = _resolve_pieces(args)

    # Accumulators
    stop_data: list[tuple[float, int]] = []           # (probability, skill_level)
    piece_id_data: list[dict] = []                     # per-recording piece ID results
    teaching_moment_data: list[dict] = []              # per-recording moment metrics
    mode_detection_data: list[dict] = []               # per-recording mode metrics
    score_following_data: list[dict] = []              # per-recording bar analysis
    synthesis_judge_results: list[dict] = []           # per-synthesis judge scores
    exercise_data: list[dict] = []                     # per-exercise judge scores
    pass_a_syntheses: list[dict] = []                  # for differentiation test
    efficiency_data: list[dict] = []                   # latency tracking
    total_judge_calls = 0
    total_recordings = 0

    # Per-recording checkpointing
    checkpoint_path = REPORTS_DIR / ".eval_checkpoint.json"
    completed_ids: set[str] = set()
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        completed_ids = set(ckpt.get("completed", []))
        stop_data = [(p, l) for p, l in ckpt.get("stop_data", [])]
        piece_id_data = ckpt.get("piece_id_data", [])
        teaching_moment_data = ckpt.get("teaching_moment_data", [])
        mode_detection_data = ckpt.get("mode_detection_data", [])
        score_following_data = ckpt.get("score_following_data", [])
        synthesis_judge_results = ckpt.get("synthesis_judge_results", [])
        exercise_data = ckpt.get("exercise_data", [])
        pass_a_syntheses = ckpt.get("pass_a_syntheses", [])
        efficiency_data = ckpt.get("efficiency_data", [])
        total_judge_calls = ckpt.get("total_judge_calls", 0)
        total_recordings = ckpt.get("total_recordings", 0)
        print(f"Resuming from checkpoint: {len(completed_ids)} passes done")

    print("Loading inference cache...")
    cache = find_inference_cache()
    print(f"  {len(cache)} recordings in cache")

    recording_count = 0

    for scenario_name, piece_id in pieces:
        scenarios = load_scenarios(scenario_name)
        print(f"\n=== {scenario_name}: {len(scenarios)} recordings ===")

        for i, scenario in enumerate(scenarios):
            if args.max_recordings and recording_count >= args.max_recordings:
                break

            video_id = scenario["video_id"]
            skill_level = scenario.get("skill_level", 0)
            piece_query = scenario.get("piece_query", "")

            # Check if both passes are done
            pass_a_key = f"{video_id}_A"
            pass_b_key = f"{video_id}_B"
            if pass_a_key in completed_ids and pass_b_key in completed_ids:
                continue

            if video_id not in cache:
                completed_ids.add(pass_a_key)
                completed_ids.add(pass_b_key)
                continue

            recording = cache[video_id]
            total_recordings += 1
            recording_count += 1
            student_id = f"eval-{video_id[:12]}"

            # --- Pass A: with piece_query (full context) ---
            if pass_a_key not in completed_ids:
                print(f"  [{i+1}/{len(scenarios)}] {video_id} Pass A (with context)...",
                      end=" ", flush=True)
                result_a: SessionResult = asyncio.run(
                    run_recording(args.wrangler_url, recording, student_id=student_id, piece_query=piece_query)
                )

                if result_a.errors:
                    print(f"ERRORS: {result_a.errors}")
                elif result_a.synthesis:
                    ec = result_a.synthesis.eval_context

                    # Statistical metrics (always computed, free)
                    stop_data.extend(extract_stop_metrics(ec, skill_level))
                    tm_metrics = extract_teaching_moment_metrics(ec)
                    if tm_metrics:
                        tm_metrics["video_id"] = video_id
                        tm_metrics["piece"] = piece_id
                        tm_metrics["skill_level"] = skill_level
                        teaching_moment_data.append(tm_metrics)
                    md_metrics = extract_mode_detection_metrics(ec)
                    md_metrics["video_id"] = video_id
                    mode_detection_data.append(md_metrics)
                    sf_metrics = extract_score_following_metrics(ec)
                    if sf_metrics:
                        sf_metrics["video_id"] = video_id
                        score_following_data.append(sf_metrics)

                    efficiency_data.append({
                        "video_id": video_id,
                        "pass": "A",
                        "duration_ms": result_a.duration_ms,
                        "chunk_send_ms": result_a.chunk_send_duration_ms,
                        "synthesis_latency_ms": result_a.synthesis_latency_ms,
                    })

                    # Store for differentiation test
                    pass_a_syntheses.append({
                        "video_id": video_id,
                        "piece": piece_id,
                        "skill_level": skill_level,
                        "text": result_a.synthesis.text,
                        "is_fallback": result_a.synthesis.is_fallback,
                    })

                    # LLM judge (only if capability selected)
                    if "synthesis" in judge_capabilities and not result_a.synthesis.is_fallback:
                        drilling_records = ec.get("drilling_records", [])
                        judge_ctx = {
                            "piece_name": ec.get("piece_context", {}).get("title", piece_id)
                                if ec.get("piece_context") else piece_id,
                            "composer": ec.get("piece_context", {}).get("composer", "Unknown")
                                if ec.get("piece_context") else "Unknown",
                            "skill_level": str(skill_level),
                            "drilling_detected": str(len(drilling_records) > 0).lower(),
                            "drilling_passage": json.dumps(drilling_records[0]) if drilling_records else "none",
                        }
                        jr = judge_synthesis(result_a.synthesis.text, judge_ctx)
                        total_judge_calls += 1
                        synthesis_judge_results.append({
                            "video_id": video_id,
                            "piece": piece_id,
                            "skill_level": skill_level,
                            "pass": "A",
                            "scores": {s.criterion: s.passed for s in jr.scores},
                            "evidence": {s.criterion: s.evidence for s in jr.scores},
                            "latency_ms": jr.latency_ms,
                        })

                    # Teaching moment judge
                    if "teaching_moments" in judge_capabilities and tm_metrics and tm_metrics.get("moment_count", 0) > 0:
                        moments = ec.get("teaching_moments", [])
                        all_moments_str = "\n".join(
                            f"- {m['dimension']}: score={m.get('score', 0):.2f}, "
                            f"deviation={m.get('deviation', 0):.2f}, "
                            f"positive={m.get('is_positive', False)}"
                            for m in moments
                        )
                        # Use the first moment as "selected" (top_moments[0])
                        selected = moments[0]
                        tm_judge_ctx = {
                            "piece_name": ec.get("piece_context", {}).get("title", piece_id)
                                if ec.get("piece_context") else piece_id,
                            "composer": ec.get("piece_context", {}).get("composer", "Unknown")
                                if ec.get("piece_context") else "Unknown",
                            "all_moments": all_moments_str,
                            "selected_dimension": selected.get("dimension", ""),
                            "deviation": str(round(selected.get("deviation", 0), 3)),
                            "score": str(round(selected.get("score", 0), 3)),
                        }
                        tm_jr = judge_teaching_moment(tm_judge_ctx)
                        total_judge_calls += 1
                        teaching_moment_data[-1]["judge_dimension_fit"] = {
                            s.criterion: s.passed for s in tm_jr.scores
                        }

                    print(f"synthesis ({len(result_a.synthesis.text)} chars)")
                else:
                    print("no synthesis")

                completed_ids.add(pass_a_key)

            # --- Pass B: without piece_query (zero-config) ---
            if pass_b_key not in completed_ids:
                print(f"  [{i+1}/{len(scenarios)}] {video_id} Pass B (zero-config)...",
                      end=" ", flush=True)
                result_b: SessionResult = asyncio.run(
                    run_recording(args.wrangler_url, recording, student_id=student_id, piece_query=None)
                )

                if result_b.errors:
                    print(f"ERRORS: {result_b.errors}")
                elif result_b.synthesis:
                    # Piece ID metrics (only in Pass B -- zero-config)
                    if result_b.piece_identification:
                        piece_id_data.append({
                            "video_id": video_id,
                            "expected": piece_id,
                            "actual": result_b.piece_identification.piece_id,
                            "confidence": result_b.piece_identification.confidence,
                            "notes_consumed": result_b.piece_identification.notes_consumed,
                            "correct": piece_id in result_b.piece_identification.piece_id,
                        })

                    efficiency_data.append({
                        "video_id": video_id,
                        "pass": "B",
                        "duration_ms": result_b.duration_ms,
                        "chunk_send_ms": result_b.chunk_send_duration_ms,
                        "synthesis_latency_ms": result_b.synthesis_latency_ms,
                    })

                    # Synthesis judge for Pass B (to compute delta)
                    if "synthesis" in judge_capabilities and not result_b.synthesis.is_fallback:
                        ec_b = result_b.synthesis.eval_context
                        drilling_b = ec_b.get("drilling_records", [])
                        judge_ctx_b = {
                            "piece_name": ec_b.get("piece_context", {}).get("title", piece_id)
                                if ec_b.get("piece_context") else "Unknown piece",
                            "composer": ec_b.get("piece_context", {}).get("composer", "Unknown")
                                if ec_b.get("piece_context") else "Unknown",
                            "skill_level": str(skill_level),
                            "drilling_detected": str(len(drilling_b) > 0).lower(),
                            "drilling_passage": json.dumps(drilling_b[0]) if drilling_b else "none",
                        }
                        jr_b = judge_synthesis(result_b.synthesis.text, judge_ctx_b)
                        total_judge_calls += 1
                        synthesis_judge_results.append({
                            "video_id": video_id,
                            "piece": piece_id,
                            "skill_level": skill_level,
                            "pass": "B",
                            "scores": {s.criterion: s.passed for s in jr_b.scores},
                            "evidence": {s.criterion: s.evidence for s in jr_b.scores},
                            "latency_ms": jr_b.latency_ms,
                        })

                    print(f"synthesis ({len(result_b.synthesis.text)} chars)")
                else:
                    print("no synthesis")

                completed_ids.add(pass_b_key)

            # Save checkpoint after each recording (both passes)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with open(checkpoint_path, "w") as f:
                json.dump({
                    "completed": list(completed_ids),
                    "stop_data": stop_data,
                    "piece_id_data": piece_id_data,
                    "teaching_moment_data": teaching_moment_data,
                    "mode_detection_data": mode_detection_data,
                    "score_following_data": score_following_data,
                    "synthesis_judge_results": synthesis_judge_results,
                    "exercise_data": exercise_data,
                    "pass_a_syntheses": pass_a_syntheses,
                    "efficiency_data": efficiency_data,
                    "total_judge_calls": total_judge_calls,
                    "total_recordings": total_recordings,
                }, f)

        if args.max_recordings and recording_count >= args.max_recordings:
            break

    # --- Differentiation test (after main loop) ---
    diff_results: list[dict] = []
    if "differentiation" in judge_capabilities:
        print("\n=== Differentiation test ===")
        # Group syntheses by piece, find triplets at skill levels 1, 3, 5
        by_piece: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
        for s in pass_a_syntheses:
            if not s.get("is_fallback"):
                by_piece[s["piece"]][s["skill_level"]].append(s)

        for piece_name, by_level in by_piece.items():
            if 1 in by_level and 3 in by_level and 5 in by_level:
                s1 = by_level[1][0]
                s3 = by_level[3][0]
                s5 = by_level[5][0]
                synth_triplet = [
                    (1, s1["text"]),
                    (3, s3["text"]),
                    (5, s5["text"]),
                ]
                jr = judge_differentiation(synth_triplet, piece_name, "Unknown")
                total_judge_calls += 1
                diff_results.append({
                    "piece": piece_name,
                    "scores": {sc.criterion: sc.passed for sc in jr.scores},
                    "evidence": {sc.criterion: sc.evidence for sc in jr.scores},
                })
                verdict = next((sc.evidence[:50] for sc in jr.scores), "")
                print(f"  {piece_name}: {verdict}")

    # --- Build report ---
    report = EvalReport(
        eval_name="synthesis_eval",
        eval_version="1.0",
        dataset=f"practice_{total_recordings}",
        metrics={},
    )

    report.metadata["total_recordings"] = total_recordings
    report.metadata["capabilities_judged"] = sorted(judge_capabilities)

    # STOP metrics
    if stop_data:
        _add_stop_metrics(report, stop_data)

    # Piece ID metrics
    if piece_id_data:
        _add_piece_id_metrics(report, piece_id_data)

    # Teaching moment metrics
    if teaching_moment_data:
        _add_teaching_moment_metrics(report, teaching_moment_data)

    # Mode detection metrics
    if mode_detection_data:
        _add_mode_detection_metrics(report, mode_detection_data)

    # Synthesis quality metrics
    if synthesis_judge_results:
        _add_synthesis_metrics(report, synthesis_judge_results)

    # Score following metrics
    if score_following_data:
        _add_score_following_metrics(report, score_following_data)

    # Differentiation
    if "differentiation" in judge_capabilities and diff_results:
        report.metadata["differentiation"] = diff_results

    # Efficiency
    if efficiency_data:
        synth_latencies = [e["synthesis_latency_ms"] for e in efficiency_data if e.get("synthesis_latency_ms")]
        if synth_latencies:
            report.metadata["efficiency"] = {
                "mean_synthesis_latency_ms": round(sum(synth_latencies) / len(synth_latencies)),
                "p90_synthesis_latency_ms": round(sorted(synth_latencies)[int(len(synth_latencies) * 0.9)]),
            }

    report.cost = {
        "judge_calls": total_judge_calls,
        "estimated_usd": round(total_judge_calls * 0.003, 2),
    }

    # Inference cache fingerprints
    if INFERENCE_CACHE_BASE.exists():
        cache_subdirs = sorted(d.name for d in INFERENCE_CACHE_BASE.iterdir() if d.is_dir())
        report.metadata["inference_cache_fingerprints"] = cache_subdirs

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report.save(REPORTS_DIR / "practice_eval.json")
    report.print_summary()

    # Save detailed results
    details = {
        "stop_data": stop_data,
        "piece_id_data": piece_id_data,
        "teaching_moment_data": teaching_moment_data,
        "mode_detection_data": mode_detection_data,
        "score_following_data": score_following_data,
        "synthesis_judge_results": synthesis_judge_results,
        "exercise_data": exercise_data,
        "pass_a_syntheses": [
            {k: v for k, v in s.items() if k != "text"}
            for s in pass_a_syntheses  # exclude full text to save space
        ],
        "efficiency_data": efficiency_data,
    }
    details_path = REPORTS_DIR / "practice_eval_details.json"
    with open(details_path, "w") as f:
        json.dump(details, f, indent=2)
    print(f"  Detailed results: {details_path}")

    # Clean up checkpoint after successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()


# ---------------------------------------------------------------------------
# Report builders
# ---------------------------------------------------------------------------

def _add_stop_metrics(report: EvalReport, stop_data: list[tuple[float, int]]) -> None:
    """Add STOP classification metrics to report."""
    from scipy.stats import spearmanr

    probs = [p for p, _ in stop_data]
    levels = [l for _, l in stop_data]

    rho, p_value = spearmanr(probs, levels)
    report.metrics["stop_spearman_rho"] = MetricResult(
        mean=rho, std=0, n=len(stop_data), pass_threshold=0.3,
    )

    bucket_triggers = defaultdict(list)
    bucket_probs = defaultdict(list)
    for prob, level in stop_data:
        bucket_triggers[level].append(prob >= STOP_THRESHOLD)
        bucket_probs[level].append(prob)

    report.metadata["stop"] = {
        "spearman_rho": round(rho, 4),
        "p_value": round(p_value, 4),
        "n": len(stop_data),
        "trigger_rate_by_bucket": {
            str(k): {"rate": round(sum(v) / len(v), 3), "n": len(v)}
            for k, v in sorted(bucket_triggers.items())
        },
        "dimension_distribution": _stop_dimension_distribution(stop_data),
    }


def _stop_dimension_distribution(stop_data: list[tuple[float, int]]) -> dict[str, int]:
    """Count which dimension contributes most to STOP across all chunks."""
    counts: dict[str, int] = {d: 0 for d in DIMS_6}
    for prob, _ in stop_data:
        if prob >= STOP_THRESHOLD:
            # Can't recover top dimension from just probability -- this would need
            # the raw scores. For now, track overall trigger count.
            pass
    return counts  # placeholder -- needs scored_chunks for per-dim analysis


def _add_piece_id_metrics(report: EvalReport, piece_id_data: list[dict]) -> None:
    """Add piece identification metrics to report."""
    correct = sum(1 for r in piece_id_data if r["correct"])
    total = len(piece_id_data)
    report.metrics["piece_id_top1_accuracy"] = MetricResult(
        mean=correct / total if total > 0 else 0,
        std=0, n=total, pass_threshold=0.8,
    )
    notes = [r["notes_consumed"] for r in piece_id_data if r["correct"]]
    report.metadata["piece_id"] = {
        "top1_accuracy": round(correct / total, 3) if total > 0 else 0,
        "total": total,
        "correct": correct,
        "mean_notes_to_identify": round(sum(notes) / len(notes)) if notes else 0,
        "false_positives": sum(
            1 for r in piece_id_data if not r["correct"] and r["confidence"] > 0.8
        ),
    }


def _add_teaching_moment_metrics(report: EvalReport, data: list[dict]) -> None:
    """Add teaching moment selection metrics to report."""
    valid = [d["positive_validity"] for d in data if d.get("positive_validity") is not None]
    aligned = [d["framing_alignment"] for d in data if d.get("framing_alignment") is not None]
    diversity = [d["dimension_diversity"] for d in data if d.get("dimension_diversity")]

    if valid:
        report.metrics["tm_positive_validity"] = MetricResult(
            mean=sum(valid) / len(valid), std=0, n=len(valid), pass_threshold=0.9,
        )
    if aligned:
        report.metrics["tm_framing_alignment"] = MetricResult(
            mean=sum(aligned) / len(aligned), std=0, n=len(aligned), pass_threshold=0.85,
        )

    # Dimension-piece fit from judge (if available)
    judged = [d for d in data if "judge_dimension_fit" in d]
    if judged:
        fits = [
            any(v for v in d["judge_dimension_fit"].values())
            for d in judged
        ]
        report.metrics["tm_dimension_piece_fit"] = MetricResult(
            mean=sum(fits) / len(fits), std=0, n=len(fits), pass_threshold=0.7,
        )

    report.metadata["teaching_moments"] = {
        "recordings_with_moments": len(data),
        "mean_moments_per_session": round(
            sum(d.get("moment_count", 0) for d in data) / len(data), 1
        ) if data else 0,
        "mean_dimension_diversity": round(sum(diversity) / len(diversity), 3) if diversity else 0,
    }


def _add_mode_detection_metrics(report: EvalReport, data: list[dict]) -> None:
    """Add practice mode detection metrics to report."""
    drilling_sessions = sum(1 for d in data if d.get("drilling_episodes", 0) > 0)
    spurious = [d.get("spurious_transitions", 0) for d in data]
    transitions = [d.get("transition_count", 0) for d in data]

    report.metadata["mode_detection"] = {
        "sessions_with_drilling": drilling_sessions,
        "total_sessions": len(data),
        "mean_transitions_per_session": round(sum(transitions) / len(transitions), 1) if transitions else 0,
        "mean_spurious_per_session": round(sum(spurious) / len(spurious), 2) if spurious else 0,
    }


def _add_synthesis_metrics(report: EvalReport, results: list[dict]) -> None:
    """Add synthesis quality metrics to report."""
    pass_a = [r for r in results if r["pass"] == "A"]
    pass_b = [r for r in results if r["pass"] == "B"]

    # Per-criterion pass rates
    criteria = set()
    for r in results:
        criteria.update(r["scores"].keys())

    for criterion in sorted(criteria):
        a_scores = [1 if r["scores"].get(criterion) else 0 for r in pass_a if r["scores"].get(criterion) is not None]
        if a_scores:
            report.metrics[f"synthesis_A_{criterion}"] = MetricResult(
                mean=sum(a_scores) / len(a_scores), std=0, n=len(a_scores),
            )
        b_scores = [1 if r["scores"].get(criterion) else 0 for r in pass_b if r["scores"].get(criterion) is not None]
        if b_scores:
            report.metrics[f"synthesis_B_{criterion}"] = MetricResult(
                mean=sum(b_scores) / len(b_scores), std=0, n=len(b_scores),
            )

    # Piece context delta
    delta = {}
    for criterion in sorted(criteria):
        a_vals = [1 if r["scores"].get(criterion) else 0 for r in pass_a if r["scores"].get(criterion) is not None]
        b_vals = [1 if r["scores"].get(criterion) else 0 for r in pass_b if r["scores"].get(criterion) is not None]
        if a_vals and b_vals:
            a_mean = sum(a_vals) / len(a_vals)
            b_mean = sum(b_vals) / len(b_vals)
            delta[criterion] = round(a_mean - b_mean, 3)

    report.metadata["synthesis"] = {
        "pass_a_count": len(pass_a),
        "pass_b_count": len(pass_b),
        "piece_context_delta": delta,
    }


def _add_score_following_metrics(report: EvalReport, data: list[dict]) -> None:
    """Add score following and bar analysis metrics to report."""
    coverage = [d.get("bar_coverage", 0) for d in data]
    tier1_rates = [d.get("tier_1_rate", 0) for d in data]
    plausible = [d.get("bar_plausibility") for d in data if d.get("bar_plausibility") is not None]

    if coverage:
        report.metrics["sf_bar_coverage"] = MetricResult(
            mean=sum(coverage) / len(coverage), std=0, n=len(coverage), pass_threshold=0.7,
        )
    if plausible:
        report.metrics["sf_bar_plausibility"] = MetricResult(
            mean=sum(plausible) / len(plausible), std=0, n=len(plausible), pass_threshold=0.8,
        )

    report.metadata["score_following"] = {
        "mean_bar_coverage": round(sum(coverage) / len(coverage), 3) if coverage else 0,
        "mean_tier_1_rate": round(sum(tier1_rates) / len(tier1_rates), 3) if tier1_rates else 0,
    }


if __name__ == "__main__":
    main()
