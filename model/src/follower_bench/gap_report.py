# model/src/follower_bench/gap_report.py
"""Phase-0 gap report (issue #117): run the shipped baseline monotonic
follower (#115) over a stratified slice of the ASAP synthetic-clip
benchmark (#111), score every clip with the trajectory metric (#113),
and aggregate per pathology type. Locks a Trackio baseline and prints a
PASS/FAIL verdict against a per-pathology bar so #118 (jump-aware DP) has
a measured starting point.

The follower is run cold-start with the load-bearing continuity prior
(DEFAULT_SKIP_PENALTY, NOT NO_PRIOR -- day-0 showed skip_penalty=0 lets
the DP teleport). No silent caps: every skipped performance and every
failed run is recorded and reported.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from follower_bench.asap_alignment import DEFAULT_ANNOTATIONS_PATH, load_alignment
from follower_bench.calibration import calibration_stats
from follower_bench.clip_generator import generate
from follower_bench.follower import DEFAULT_SKIP_PENALTY, ContinuityPrior, bar_boundary_columns, follow
from follower_bench.hmm import TUNED_HMM_PARAMS, HmmParams, follow_hmm
from follower_bench.metric import aggregate_by_pathology, score_clip, trajectory_from_matches
from follower_bench.pathologies import PATHOLOGY_TYPES
from follower_bench.score_notes import load_score_notes_from_midi

# Per-pathology PASS bar (issue #117, user-approved 2026-07-13).
# jump is the KNOWN monotonic gap (#118) -- reported, never failed.
LOCK_BAR = {
    "clean": 0.90,
    "tempo_swing": 0.80,
    "wrong_note": 0.80,
    "hesitation": 0.80,
}
RELOCK_RECOVER = ("repeat", "restart")  # must recover: finite relock, median <= bar
RELOCK_MEDIAN_BAR_S = 8.0
KNOWN_GAP = ("jump",)  # inf relock expected; documents the gap #118 closes


@dataclass(frozen=True)
class RunOutcome:
    """One (performance, pathology, seed) benchmark cell result, or a
    recorded failure. Exactly one of `score` / `error` is set."""
    performance: str
    pathology: str
    seed: int
    score: object | None  # metric.TrajectoryScore
    error: str | None
    elapsed_s: float
    calibration: object | None = None  # calibration.CalibrationStats when --hmm; None on the additive path


def sample_performances(per_composer: int, annotations_path: Path = DEFAULT_ANNOTATIONS_PATH) -> list[str]:
    """Deterministically pick up to `per_composer` usable aligned
    performances per composer (sorted by key, first N), stratifying the
    239-piece / 1035-performance benchmark across its 16 composers. No
    RNG -- the sample is reproducible run to run."""
    data = json.loads(annotations_path.read_text())
    by_composer: dict[str, list[str]] = defaultdict(list)
    for key, entry in data.items():
        if not entry.get("score_and_performance_aligned"):
            continue
        pb = entry.get("performance_beats") or []
        sb = entry.get("midi_score_beats") or []
        if len(pb) < 4 or len(pb) != len(sb):
            continue
        by_composer[key.split("/")[0]].append(key)
    chosen: list[str] = []
    for composer in sorted(by_composer):
        chosen.extend(sorted(by_composer[composer])[:per_composer])
    return chosen


def _follow_for_cell(use_hmm, amt_notes, score_notes, prior, hmm_params, bar_boundaries, transpose_candidates):
    """Route one cell to the additive DP (default) or the #119 HMM decoder.
    transpose_candidates defaults to follow()/follow_hmm's own default when None."""
    if use_hmm:
        if transpose_candidates is None:
            return follow_hmm(amt_notes, score_notes, hmm_params, bar_boundaries=bar_boundaries)
        return follow_hmm(amt_notes, score_notes, hmm_params, bar_boundaries=bar_boundaries,
                          transpose_candidates=transpose_candidates)
    if transpose_candidates is None:
        return follow(amt_notes, score_notes, prior, bar_boundaries=bar_boundaries)
    return follow(amt_notes, score_notes, prior, bar_boundaries=bar_boundaries,
                  transpose_candidates=transpose_candidates)


def _run_cell(performance: str, pathology: str, seed: int, score_notes: list,
              bar_boundaries: tuple[int, ...], jump_back_penalty: float, jump_fwd_penalty: float,
              use_hmm: bool = False, hmm_params: HmmParams | None = None) -> RunOutcome:
    prior = ContinuityPrior(skip_penalty=DEFAULT_SKIP_PENALTY,
                            jump_back_penalty=jump_back_penalty,
                            jump_fwd_penalty=jump_fwd_penalty)
    if hmm_params is None:
        hmm_params = TUNED_HMM_PARAMS
    t0 = time.perf_counter()
    try:
        clip = generate(performance, pathology, seed)
        est = _follow_for_cell(use_hmm, list(clip.notes), score_notes, prior, hmm_params, bar_boundaries, None)
        est_traj = trajectory_from_matches(est.matches)
        score = score_clip(est_traj, clip)
        # HMM path only: measure confidence calibration for this cell (needs matches).
        calibration = calibration_stats(est.matches, clip) if (use_hmm and est.matches) else None
        return RunOutcome(performance, pathology, seed, score, None, time.perf_counter() - t0, calibration)
    except Exception as exc:  # loud: recorded, never silently dropped
        return RunOutcome(performance, pathology, seed, None, f"{type(exc).__name__}: {exc}",
                          time.perf_counter() - t0, None)


def _run_performance(task: tuple[str, list[int], int | None, float, float, bool]) -> tuple[list[RunOutcome], dict | None]:
    """Load one performance's score MIDI once, compute its bar-boundary
    columns from the ASAP downbeats, then run all its (pathology, seed)
    cells. Returns (outcomes, skip_record-or-None). Pickle-safe top-level
    function so multiprocessing.Pool can dispatch it. `clean` is
    RNG-invariant so it runs once regardless of seed count. When
    max_score_notes is set, a performance whose score MIDI exceeds it is
    recorded as an explicit skip (the #118 iteration-speed cap). jump
    penalties default to inf upstream, giving the monotonic baseline."""
    perf, seeds, max_score_notes, jump_back_penalty, jump_fwd_penalty, use_hmm = task
    try:
        alignment = load_alignment(perf)
        score_notes = load_score_notes_from_midi(alignment.score_midi_path)
    except Exception as exc:  # loud: recorded as a skip, never silently dropped
        return [], {"performance": perf, "reason": f"{type(exc).__name__}: {exc}"}
    if max_score_notes is not None and len(score_notes) > max_score_notes:
        return [], {"performance": perf,
                    "reason": f"excluded by --max-score-notes cap ({len(score_notes)} > {max_score_notes})"}
    bar_boundaries = bar_boundary_columns([n.position for n in score_notes], alignment.midi_score_downbeats)
    outcomes: list[RunOutcome] = []
    for pathology in PATHOLOGY_TYPES:
        cell_seeds = [seeds[0]] if pathology == "clean" else seeds
        for seed in cell_seeds:
            outcomes.append(_run_cell(perf, pathology, seed, score_notes,
                                      bar_boundaries, jump_back_penalty, jump_fwd_penalty,
                                      use_hmm=use_hmm))
    return outcomes, None


def run_gap_report(
    performances: list[str], seeds: list[int], workers: int = 1, max_score_notes: int | None = None,
    jump_back_penalty: float = math.inf, jump_fwd_penalty: float = math.inf,
    use_hmm: bool = False,
) -> dict:
    """Run every (performance, pathology, seed) cell, score it, and
    aggregate per pathology. Parallelizes over performances when
    workers > 1. `clean` is RNG-invariant so it runs once per performance
    regardless of seed count. `max_score_notes` excludes whole
    performances whose score MIDI exceeds the cap (recorded as skips).
    jump_back_penalty / jump_fwd_penalty (default inf = monotonic
    baseline) enable the #118 bar-boundary jumps."""
    outcomes: list[RunOutcome] = []
    skipped: list[dict] = []
    tasks = [(perf, seeds, max_score_notes, jump_back_penalty, jump_fwd_penalty, use_hmm) for perf in performances]
    if workers > 1:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            results = pool.map(_run_performance, tasks)
    else:
        results = [_run_performance(t) for t in tasks]
    for perf_outcomes, skip in results:
        outcomes.extend(perf_outcomes)
        if skip is not None:
            skipped.append(skip)

    ok = [o for o in outcomes if o.score is not None]
    failed = [o for o in outcomes if o.error is not None]
    aggregates = aggregate_by_pathology([o.score for o in ok])
    return {
        "aggregates": aggregates,
        "outcomes": outcomes,
        "ok": ok,
        "failed": failed,
        "skipped_performances": skipped,
        "n_performances": len(performances),
    }


def evaluate_bar(aggregates: dict) -> dict:
    """Apply the per-pathology PASS bar. Returns per-pathology verdicts +
    an overall pass flag. jump is reported as the known gap, never fails."""
    verdicts = {}
    overall_pass = True
    for pathology, agg in sorted(aggregates.items()):
        v = {"n_clips": agg.n_clips, "mean_lock_rate": agg.mean_lock_rate,
             "relock_success_rate": agg.relock_success_rate,
             "median_relock_latency_s": agg.median_relock_latency_s,
             "total_false_jumps": agg.total_false_jumps}
        if pathology in LOCK_BAR:
            passed = agg.mean_lock_rate >= LOCK_BAR[pathology]
            if pathology == "clean":
                passed = passed and agg.total_false_jumps == 0
            v["bar"] = f"lock_rate>={LOCK_BAR[pathology]}" + (" & 0 false_jumps" if pathology == "clean" else "")
            v["verdict"] = "PASS" if passed else "FAIL"
            overall_pass = overall_pass and passed
        elif pathology in RELOCK_RECOVER:
            passed = agg.relock_success_rate >= 0.999 and agg.median_relock_latency_s <= RELOCK_MEDIAN_BAR_S
            v["bar"] = f"relock_success=1.0 & median_relock<={RELOCK_MEDIAN_BAR_S}s"
            v["verdict"] = "PASS" if passed else "FAIL"
            overall_pass = overall_pass and passed
        elif pathology in KNOWN_GAP:
            v["bar"] = "known monotonic gap (#118) -- reported, not scored"
            v["verdict"] = "GAP"
        else:
            v["bar"] = "(no bar)"
            v["verdict"] = "INFO"
        verdicts[pathology] = v
    return {"per_pathology": verdicts, "overall_pass": overall_pass}


def _format_report(result: dict, evaluation: dict, wall_s: float) -> str:
    lines = []
    lines.append("=" * 74)
    lines.append("FOLLOWER PHASE-0 GAP REPORT (#117) -- baseline monotonic follower (#115)")
    lines.append("=" * 74)
    lines.append(f"performances sampled: {result['n_performances']}   "
                 f"clips scored: {len(result['ok'])}   failed: {len(result['failed'])}   "
                 f"skipped perfs: {len(result['skipped_performances'])}")
    lines.append(f"wall time: {wall_s:.1f}s")
    lines.append("")
    hdr = f"{'pathology':<13}{'n':>5}{'lock_rate':>11}{'relock_ok':>11}{'relock_med_s':>14}{'false_jmp':>10}  verdict"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for pathology, v in evaluation["per_pathology"].items():
        med = v["median_relock_latency_s"]
        med_s = "inf" if med == float("inf") else f"{med:.2f}"
        lines.append(f"{pathology:<13}{v['n_clips']:>5}{v['mean_lock_rate']:>11.3f}"
                     f"{v['relock_success_rate']:>11.3f}{med_s:>14}{v['total_false_jumps']:>10}  {v['verdict']}  [{v['bar']}]")
    lines.append("-" * len(hdr))
    lines.append(f"OVERALL: {'PASS' if evaluation['overall_pass'] else 'FAIL'} "
                 f"(jump = known gap #118, excluded from pass/fail)")
    calibrated = [o.calibration for o in result["ok"] if o.calibration is not None]
    if calibrated:
        import statistics as _st
        median_rho = _st.median(c.spearman_rho for c in calibrated)
        median_err = _st.median(c.overall_median_error for c in calibrated)
        lines.append("")
        lines.append(f"HMM CALIBRATION (n={len(calibrated)} cells): "
                     f"median spearman_rho={median_rho:.3f}  median overall_err={median_err:.3f} beats")
    if result["failed"]:
        lines.append("")
        lines.append(f"FAILED RUNS ({len(result['failed'])}):")
        for o in result["failed"][:20]:
            lines.append(f"  {o.performance} [{o.pathology}/{o.seed}] -> {o.error}")
    if result["skipped_performances"]:
        lines.append("")
        lines.append(f"SKIPPED PERFORMANCES ({len(result['skipped_performances'])}):")
        for s in result["skipped_performances"][:20]:
            lines.append(f"  {s['performance']} -> {s['reason']}")
    return "\n".join(lines)


def _add_cli_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register every gap_report CLI argument (shared by main() and the routing
    tests so the --hmm flag is introspectable without invoking a full run)."""
    ap.add_argument("--per-composer", type=int, default=5, help="max performances sampled per composer")
    ap.add_argument("--seeds", type=int, default=5, help="pathology seeds per performance (clean runs once)")
    ap.add_argument("--workers", type=int, default=8, help="parallel worker processes")
    ap.add_argument("--max-score-notes", type=int, default=None,
                    help="exclude performances whose score MIDI exceeds this many notes "
                         "(iteration-speed cap for #118; excluded perfs are reported, not silently dropped)")
    ap.add_argument("--jump-back-penalty", type=float, default=None,
                    help="enable #118 backward (repeat/restart) bar-boundary jumps at this penalty (default: off/inf)")
    ap.add_argument("--jump-fwd-penalty", type=float, default=None,
                    help="enable #118 forward (skip) bar-boundary jumps at this penalty (default: off/inf)")
    ap.add_argument("--hmm", action="store_true",
                    help="use the #119 Viterbi-HMM decoder instead of the additive DP")
    ap.add_argument("--trackio", action="store_true", help="log the baseline to Trackio")
    ap.add_argument("--out", type=Path, default=None, help="write the text report to this path")
    return ap


def main() -> None:
    ap = argparse.ArgumentParser(description="Follower Phase-0 gap report (#117)")
    _add_cli_args(ap)
    args = ap.parse_args()

    performances = sample_performances(args.per_composer)
    seeds = list(range(args.seeds))
    print(f"[gap-report] {len(performances)} performances x {len(PATHOLOGY_TYPES)} pathologies x {args.seeds} seeds ...")
    t0 = time.perf_counter()
    result = run_gap_report(
        performances, seeds, workers=args.workers, max_score_notes=args.max_score_notes,
        jump_back_penalty=args.jump_back_penalty if args.jump_back_penalty is not None else math.inf,
        jump_fwd_penalty=args.jump_fwd_penalty if args.jump_fwd_penalty is not None else math.inf,
        use_hmm=args.hmm,
    )
    wall = time.perf_counter() - t0
    evaluation = evaluate_bar(result["aggregates"])
    report = _format_report(result, evaluation, wall)
    print(report)
    if args.out:
        args.out.write_text(report + "\n")

    if args.trackio:
        import trackio
        trackio.init(project="score-follower-bench", name="gap-report-baseline-115")
        for pathology, v in evaluation["per_pathology"].items():
            trackio.log({f"{pathology}/lock_rate": v["mean_lock_rate"],
                         f"{pathology}/relock_success": v["relock_success_rate"],
                         f"{pathology}/false_jumps": v["total_false_jumps"],
                         f"{pathology}/n": v["n_clips"]})
        trackio.finish()
        print("[gap-report] logged Trackio baseline: score-follower-bench/gap-report-baseline-115")


if __name__ == "__main__":
    main()
