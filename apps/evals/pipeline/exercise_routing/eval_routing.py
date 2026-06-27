"""Thin orchestrator for the exercise-routing eval harness.

Entry point: python -m pipeline.exercise_routing.eval_routing [--skip-inference]

Requires `just dev` (MuQ:8000 + AMT:8001 + API:8787) and `just seed-fingerprint`
unless --skip-inference is passed (CI-safe smoke: validates wiring only).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

REPO_ROOT = Path(__file__).resolve().parents[4]
PRACTICE_EVAL_ROOT = REPO_ROOT / "model" / "data" / "evals" / "practice_eval"
RESULTS_DIR = Path(__file__).parents[2] / "results" / "exercise_routing"
BASELINE_PATH = RESULTS_DIR / "baseline.json"
LAST_RUN_PATH = RESULTS_DIR / "last_run.json"
MANIFEST_PATH = (
    REPO_ROOT / "apps" / "api" / "src" / "services" / "exercise_primitives_manifest.json"
)


def build_manifest() -> list[dict]:
    """Discover all practice_eval WAVs and their piece slugs from directory structure."""
    entries = []
    if not PRACTICE_EVAL_ROOT.exists():
        raise FileNotFoundError(
            f"practice_eval root not found: {PRACTICE_EVAL_ROOT}. "
            "Ensure model/data/evals/practice_eval/ is populated."
        )
    for piece_dir in sorted(PRACTICE_EVAL_ROOT.iterdir()):
        if not piece_dir.is_dir():
            continue
        audio_dir = piece_dir / "audio"
        if not audio_dir.exists():
            continue
        for wav in sorted(audio_dir.glob("*.wav")):
            entries.append({"recording": wav, "piece_slug": piece_dir.name})
    if not entries:
        raise FileNotFoundError(
            f"No WAV files found under {PRACTICE_EVAL_ROOT}. "
            "Run the audio-acquire recipe or provide recordings."
        )
    return entries


def _check_no_universal_piece_id_failure(scores: list) -> None:
    """Raise if ALL sessions returned piece_resolved=False (seed-fingerprint not run)."""
    invoked = [s for s in scores if s.invoked and s.error is None]
    if not invoked:
        return  # invocation itself is the issue; baseline gate catches it
    kind_correct_count = sum(1 for s in invoked if s.kind_correct)
    if kind_correct_count == 0 and len(invoked) >= 5:
        raise RuntimeError(
            "All invoked sessions produced corpus_drill with piece_resolved=False. "
            "This indicates `just seed-fingerprint` was not run before this eval. "
            "Run `just seed-fingerprint` and retry."
        )


def _check_baselines(axis_scores, baseline: dict) -> list[str]:
    """Return list of failure messages for axes below their floor thresholds."""
    failures = []
    checks = [
        ("invocation_rate", axis_scores.invocation_rate, baseline["invocation_rate_floor"]),
        ("kind_correctness_rate", axis_scores.kind_correctness_rate, baseline["kind_correctness_floor"]),
        ("dimension_match_rate", axis_scores.dimension_match_rate, baseline["dimension_match_floor"]),
        ("tempo_sanity_rate", axis_scores.tempo_sanity_rate, baseline["tempo_sanity_floor"]),
    ]
    for name, value, floor in checks:
        if value < floor:
            failures.append(f"  {name}: {value:.3f} < floor {floor:.3f}")

    # bar_range_grounding: only check if n > 0
    if axis_scores.bar_range_grounding_n > 0:
        floor = baseline["bar_range_grounding_floor"]
        if axis_scores.bar_range_grounding_rate < floor:
            failures.append(
                f"  bar_range_grounding_rate: {axis_scores.bar_range_grounding_rate:.3f} "
                f"< floor {floor:.3f} (n={axis_scores.bar_range_grounding_n})"
            )

    return failures


def run_relevance(captures: list, judge_provider: str, judge_model: str | None):
    """Judge selector relevance@1 over every invoked session.

    For each capture with a dominant weakness, ask the judge whether the drill the
    deterministic selector would pick is pedagogically appropriate. This is the
    selector-quality signal that cosine selection (Goal B) moves. The judge is a
    different model family from the V6 teacher (glm) per the same-family rule.

    Returns the RelevanceAggregate; raises if the judge client cannot be built
    (missing gateway creds) -- fail loud rather than silently skip the metric.
    """
    from teaching_knowledge.llm_client import LLMClient

    from pipeline.exercise_routing.relevance import aggregate_relevance, judge_relevance
    from pipeline.exercise_routing.selection import build_selector_case

    manifest = json.loads(MANIFEST_PATH.read_text())
    cases = [build_selector_case(c, manifest) for c in captures]
    cases = [c for c in cases if c is not None]

    if not cases:
        print("  [relevance] no judgeable sessions (no dominant dimensions)")
        from pipeline.exercise_routing.relevance import RelevanceAggregate
        return RelevanceAggregate(relevance_at_1=0.0, mean_score=0.0, n_judged=0)

    client = LLMClient(provider=judge_provider, model=judge_model, tier="judge")
    print(
        f"  [relevance] judging {len(cases)} selector choices "
        f"via {judge_provider}/{client.model} ..."
    )
    verdicts = []
    for i, case in enumerate(cases, 1):
        verdict = judge_relevance(case, client)
        verdicts.append(verdict)
        flag = "OK" if verdict.appropriate else "OFF"
        print(
            f"    [{i}/{len(cases)}] {case.weakness_dimension} -> "
            f"{case.drill.primitive_id} score={verdict.score} {flag}"
        )
    return aggregate_relevance(verdicts)


def run_skip_inference(baseline: dict) -> int:
    """Validate wiring without services: score.py imports, baseline has all axes."""
    from pipeline.exercise_routing.score import score_session, aggregate, SessionCapture, AxisScores

    print("[exercise-routing-eval] --skip-inference mode: validating wiring only")
    try:
        manifest = build_manifest()
        print(f"  manifest: {len(manifest)} recordings across practice_eval/")
    except FileNotFoundError as exc:
        print(f"  [warning] practice_eval audio not found: {exc}", file=sys.stderr)
        print("  manifest: skipped (no audio data — run audio-acquire to populate)")

    required = {"invocation_rate_floor", "kind_correctness_floor", "dimension_match_floor",
                "bar_range_grounding_floor", "tempo_sanity_floor"}
    missing = required - set(baseline.keys())
    if missing:
        raise RuntimeError(f"baseline.json is missing axes: {missing}")

    print("  score.py: OK")
    print("  baseline.json: OK")

    # Validate the relevance pipeline wiring with a fake client (no network):
    # selector-case build off the real manifest + judge parse path.
    from pipeline.exercise_routing.relevance import RelevanceCase, DrillInfo, judge_relevance
    from pipeline.exercise_routing.selection import select_primitive

    real_manifest = json.loads(MANIFEST_PATH.read_text())
    pid, _ = select_primitive({"target_dimension": "timing"}, real_manifest)

    class _FakeJudge:
        def complete(self, *, user, system, max_tokens):
            return '{"score": 3, "rationale": "smoke"}'

    entry = real_manifest[pid]
    case = RelevanceCase(
        weakness_dimension="timing", weakness_context="smoke", bar_range=(1, 2),
        drill=DrillInfo(pid, entry["title"], entry["source"],
                        entry["dimensions"], entry["techniques"]),
    )
    assert judge_relevance(case, _FakeJudge()).score == 3
    print("  relevance pipeline: OK")
    print("[exercise-routing-eval] smoke PASSED")
    return 0


def run_full(
    baseline: dict,
    wrangler_url: str,
    skip_relevance: bool = False,
    judge_provider: str = "workers-ai",
    judge_model: str | None = None,
) -> int:
    """Run full eval: drive all recordings, score, write last_run.json, diff baseline."""
    from shared.local_session import drive, check_services
    from pipeline.exercise_routing.score import score_session, aggregate, SessionScore

    print(f"[exercise-routing-eval] checking services at {wrangler_url} ...")
    check_services(wrangler_url)
    print("  services: OK")

    manifest = build_manifest()
    print(f"  manifest: {len(manifest)} recordings")

    session_scores = []
    captures = []
    for entry in manifest:
        recording: Path = entry["recording"]
        piece_slug: str = entry["piece_slug"]
        print(f"  driving {recording.name} ({piece_slug}) ...", end=" ", flush=True)
        try:
            capture = drive(recording=recording, piece_slug=piece_slug, wrangler_url=wrangler_url)
            captures.append(capture)
            score = score_session(capture)
            session_scores.append(score)
            status = "invoked" if score.invoked else "null"
            print(f"{status} kind={score.kind_correct} dim={score.dimension_match}")
        except (RuntimeError, TimeoutError, ConnectionError, OSError) as exc:
            err_score = SessionScore(
                session_id=str(recording),
                piece_slug=piece_slug,
                invoked=False,
                kind_correct=None,
                dimension_match=None,
                bar_range_grounded=None,
                tempo_in_bounds=None,
                tempo_weak_prior_flag=None,
                error=str(exc),
            )
            session_scores.append(err_score)
            print(f"ERROR: {exc}")
            print(f"  [error count so far: {sum(1 for s in session_scores if s.error is not None)}]")

    _check_no_universal_piece_id_failure(session_scores)

    axis_scores = aggregate(session_scores)

    if axis_scores.n_invoked == 0:
        raise RuntimeError(
            "0 invocations across all sessions — V6 gate is broken or "
            "prescribed_exercise is always null in the artifact. "
            "Check HARNESS_V6_ENABLED, seed-fingerprint, and the V6 gate condition."
        )

    last_run = {
        "n_sessions": axis_scores.n_sessions,
        "n_invoked": axis_scores.n_invoked,
        "n_errors": axis_scores.n_errors,
        "invocation_rate": round(axis_scores.invocation_rate, 4),
        "kind_correctness_rate": round(axis_scores.kind_correctness_rate, 4),
        "dimension_match_rate": round(axis_scores.dimension_match_rate, 4),
        "bar_range_grounding_rate": round(axis_scores.bar_range_grounding_rate, 4),
        "bar_range_grounding_n": axis_scores.bar_range_grounding_n,
        "tempo_sanity_rate": round(axis_scores.tempo_sanity_rate, 4),
        "tempo_weak_prior_flag_count": axis_scores.tempo_weak_prior_flag_count,
    }

    if not skip_relevance:
        rel = run_relevance(captures, judge_provider, judge_model)
        last_run["selector_relevance_at_1"] = round(rel.relevance_at_1, 4)
        last_run["selector_relevance_mean_score"] = round(rel.mean_score, 4)
        last_run["selector_relevance_n"] = rel.n_judged

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LAST_RUN_PATH.write_text(json.dumps(last_run, indent=2))
    print(f"\n[exercise-routing-eval] last_run.json written to {LAST_RUN_PATH}")

    print("\n--- Axis scores ---")
    for k, v in last_run.items():
        print(f"  {k}: {v}")

    failures = _check_baselines(axis_scores, baseline)

    # Relevance floor: only enforced when a floor is committed AND cases were judged.
    rel_floor = baseline.get("selector_relevance_at_1_floor")
    if (
        not skip_relevance
        and rel_floor is not None
        and last_run.get("selector_relevance_n", 0) > 0
    ):
        rel_value = last_run["selector_relevance_at_1"]
        if rel_value < rel_floor:
            failures.append(
                f"  selector_relevance_at_1: {rel_value:.3f} < floor {rel_floor:.3f} "
                f"(n={last_run['selector_relevance_n']})"
            )

    if failures:
        print("\nFAIL: axes below baseline floors:")
        for f in failures:
            print(f)
        return 1

    print("\nPASS: all axes above baseline floors")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Exercise-routing eval harness")
    parser.add_argument("--skip-inference", action="store_true",
                        help="Smoke mode: validate wiring without live services (CI-safe)")
    parser.add_argument("--wrangler-url", default="http://localhost:8787",
                        help="API base URL (default: http://localhost:8787)")
    parser.add_argument("--skip-relevance", action="store_true",
                        help="Skip the LLM-judge relevance@1 pass (stats axes only)")
    parser.add_argument("--judge-provider", default="workers-ai",
                        help="Relevance judge provider (workers-ai|anthropic|openrouter)")
    parser.add_argument("--judge-model", default=None,
                        help="Override the judge model id (defaults to provider's judge tier)")
    args = parser.parse_args()

    if not BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"baseline.json not found at {BASELINE_PATH}. "
            "Commit a baseline before running the harness."
        )
    baseline = json.loads(BASELINE_PATH.read_text())

    if args.skip_inference:
        return run_skip_inference(baseline)
    return run_full(
        baseline,
        args.wrangler_url,
        skip_relevance=args.skip_relevance,
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
    )


if __name__ == "__main__":
    sys.exit(main())
