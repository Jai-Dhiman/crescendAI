"""Cross-cutting analysis of E2E pipeline eval results.

Reads practice_eval.json (EvalReport) and practice_eval_details.json,
produces formatted reports for all 7 capabilities plus efficiency and cost.

No LLM calls -- pure computation on cached results.

Usage:
    cd apps/evals/
    uv run python -m pipeline.practice_eval.analyze_e2e
    uv run python -m pipeline.practice_eval.analyze_e2e --capability stop
    uv run python -m pipeline.practice_eval.analyze_e2e --capability synthesis
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPORTS_DIR = Path(__file__).parents[2] / "reports"

CAPABILITIES = [
    "piece_id", "stop", "teaching_moments", "mode_detection",
    "synthesis", "score_following", "differentiation",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cohens_d(group1: list[float], group2: list[float]) -> float:
    """Compute Cohen's d effect size between two groups."""
    if len(group1) < 2 or len(group2) < 2:
        return 0.0
    m1, m2 = float(np.mean(group1)), float(np.mean(group2))
    s1, s2 = float(np.std(group1, ddof=1)), float(np.std(group2, ddof=1))
    n1, n2 = len(group1), len(group2)
    denom = n1 + n2 - 2
    if denom <= 0:
        return 0.0
    pooled = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / denom)
    if pooled == 0:
        return 0.0
    return float((m1 - m2) / pooled)


def _effect_label(d: float) -> str:
    ad = abs(d)
    if ad >= 0.8:
        return "large"
    if ad >= 0.5:
        return "medium"
    return "small"


def _pct(val: float | None, width: int = 7) -> str:
    if val is None:
        return "---".rjust(width)
    return f"{val:.1%}".rjust(width)


def _flt(val: float | None, decimals: int = 3, width: int = 8) -> str:
    if val is None:
        return "---".rjust(width)
    return f"{val:.{decimals}f}".rjust(width)


def _safe_mean(vals: list[float]) -> float | None:
    return sum(vals) / len(vals) if vals else None


def _percentile(vals: list[float], pct: float) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    idx = int(len(s) * pct)
    idx = min(idx, len(s) - 1)
    return s[idx]


# ---------------------------------------------------------------------------
# Section printers
# ---------------------------------------------------------------------------

def print_piece_id(report: dict, details: dict) -> None:
    """Piece ID: top-1 accuracy, notes consumed, false positives, per-piece breakdown."""
    print("\n" + "=" * 64)
    print("PIECE IDENTIFICATION")
    print("=" * 64)

    meta = report.get("metadata", {}).get("piece_id", {})
    if not meta:
        pid_data = details.get("piece_id_data", [])
        if not pid_data:
            print("\n  No piece ID data (all scenarios used explicit piece_query)")
            return
        # Reconstruct from details
        correct = sum(1 for r in pid_data if r.get("correct"))
        total = len(pid_data)
        notes = [r["notes_consumed"] for r in pid_data if r.get("correct")]
        false_pos = sum(
            1 for r in pid_data if not r.get("correct") and r.get("confidence", 0) > 0.8
        )
        meta = {
            "top1_accuracy": round(correct / total, 3) if total > 0 else 0,
            "total": total,
            "correct": correct,
            "mean_notes_to_identify": round(sum(notes) / len(notes)) if notes else 0,
            "false_positives": false_pos,
        }

    print(f"\n  Top-1 accuracy:       {meta.get('top1_accuracy', 'N/A')}")
    print(f"  Total tested:         {meta.get('total', 0)}")
    print(f"  Correct:              {meta.get('correct', 0)}")
    print(f"  Mean notes consumed:  {meta.get('mean_notes_to_identify', 'N/A')}")
    print(f"  False positives:      {meta.get('false_positives', 0)}")

    # Per-piece breakdown from details
    pid_data = details.get("piece_id_data", [])
    if pid_data:
        by_piece: dict[str, list[dict]] = defaultdict(list)
        for r in pid_data:
            by_piece[r.get("expected", "unknown")].append(r)

        print(f"\n  {'Piece':<30} {'Acc':>7} {'N':>5} {'Notes':>7} {'FP':>4}")
        print("  " + "-" * 55)
        for piece in sorted(by_piece.keys()):
            recs = by_piece[piece]
            n = len(recs)
            correct = sum(1 for r in recs if r.get("correct"))
            acc = correct / n if n > 0 else 0
            notes_list = [r["notes_consumed"] for r in recs if r.get("correct")]
            mean_notes = round(sum(notes_list) / len(notes_list)) if notes_list else 0
            fp = sum(
                1 for r in recs if not r.get("correct") and r.get("confidence", 0) > 0.8
            )
            flag = " *" if n < 5 else ""
            print(f"  {piece:<30} {acc:>6.1%} {n:>5}{flag} {mean_notes:>6} {fp:>4}")

        if any(len(by_piece[p]) < 5 for p in by_piece):
            print("\n  * = N < 5, treat with caution")


def print_stop(report: dict, details: dict) -> None:
    """STOP: Spearman rho, trigger rates by bucket, Cohen's d between adjacent buckets."""
    print("\n" + "=" * 64)
    print("STOP GENERALIZATION")
    print("=" * 64)

    meta_stop = report.get("metadata", {}).get("stop", {})
    stop_data = details.get("stop_data", [])

    if not meta_stop and not stop_data:
        print("\n  No STOP data.")
        return

    # Spearman correlation
    rho = meta_stop.get("spearman_rho", "N/A")
    p_val = meta_stop.get("p_value", "N/A")
    n = meta_stop.get("n", len(stop_data))
    print(f"\n  Spearman rho: {rho}  (p={p_val}, n={n})")

    if isinstance(rho, (int, float)):
        if rho < -0.3:
            print("    -> Good: higher-skill students get lower STOP probability")
        elif rho > 0.1:
            print("    -> WARNING: STOP triggers MORE on skilled students (inverted)")
        else:
            print("    -> Weak/no correlation")

    # Trigger rate by bucket
    trigger_rates = meta_stop.get("trigger_rate_by_bucket", {})
    if trigger_rates:
        print(f"\n  {'Bucket':<10} {'Rate':>8} {'N':>6}")
        print("  " + "-" * 26)
        for bucket in sorted(trigger_rates.keys(), key=lambda x: int(x)):
            info = trigger_rates[bucket]
            rate = info["rate"]
            bn = info["n"]
            flag = " *" if bn < 5 else ""
            print(f"  {bucket:<10} {rate:>7.3f} {bn:>5}{flag}")

        if any(trigger_rates[b]["n"] < 5 for b in trigger_rates):
            print("\n  * = N < 5, treat with caution")

    # Cohen's d between adjacent buckets (computed from raw stop_data)
    if stop_data:
        bucket_probs: dict[int, list[float]] = defaultdict(list)
        for prob, level in stop_data:
            bucket_probs[level].append(prob)

        buckets_sorted = sorted(bucket_probs.keys())
        if len(buckets_sorted) > 1:
            print(f"\n  Cohen's d (STOP probability, adjacent buckets):")
            for i in range(len(buckets_sorted) - 1):
                b1, b2 = buckets_sorted[i], buckets_sorted[i + 1]
                d = cohens_d(bucket_probs[b1], bucket_probs[b2])
                label = _effect_label(d)
                n1, n2 = len(bucket_probs[b1]), len(bucket_probs[b2])
                print(f"    Bucket {b1} vs {b2}: d={d:+.3f} ({label})"
                      f"  [n={n1},{n2}]")


def print_teaching_moments(report: dict, details: dict) -> None:
    """Teaching moments: positive validity, framing alignment, dimension-piece fit, diversity."""
    print("\n" + "=" * 64)
    print("TEACHING MOMENTS")
    print("=" * 64)

    tm_meta = report.get("metadata", {}).get("teaching_moments", {})
    tm_data = details.get("teaching_moment_data", [])
    metrics = report.get("metrics", {})

    if not tm_meta and not tm_data:
        print("\n  No teaching moment data.")
        return

    # Summary
    n_sessions = tm_meta.get("recordings_with_moments", len(tm_data))
    mean_moments = tm_meta.get("mean_moments_per_session", 0)
    mean_diversity = tm_meta.get("mean_dimension_diversity", 0)
    print(f"\n  Sessions with moments: {n_sessions}")
    print(f"  Mean moments/session:  {mean_moments}")
    print(f"  Mean dim diversity:    {mean_diversity:.3f}")

    # Metrics from report
    pv = metrics.get("tm_positive_validity", {})
    fa = metrics.get("tm_framing_alignment", {})
    dpf = metrics.get("tm_dimension_piece_fit", {})

    print(f"\n  {'Metric':<28} {'Value':>8} {'Gate':>8} {'N':>5} {'Status':>8}")
    print("  " + "-" * 59)

    for label, key in [
        ("Positive validity", "tm_positive_validity"),
        ("Framing alignment", "tm_framing_alignment"),
        ("Dimension-piece fit", "tm_dimension_piece_fit"),
    ]:
        m = metrics.get(key, {})
        if m:
            mean = m.get("mean", 0)
            gate = m.get("pass_threshold")
            n = m.get("n", 0)
            status = "PASS" if m.get("passed") else "FAIL"
            gate_str = f"{gate:.3f}" if gate is not None else "---"
            print(f"  {label:<28} {mean:>8.3f} {gate_str:>8} {n:>5} {status:>8}")
        else:
            print(f"  {label:<28} {'---':>8} {'---':>8} {'---':>5} {'---':>8}")

    # Per-recording breakdown
    if tm_data:
        print(f"\n  Per-recording detail:")
        print(f"  {'Video':<20} {'Piece':<16} {'Sk':>3} {'Mom':>4} {'PosV':>7} {'FrmA':>7} {'Div':>5}")
        print("  " + "-" * 64)
        for d in sorted(tm_data, key=lambda x: x.get("video_id", "")):
            vid = d.get("video_id", "?")[:18]
            piece = d.get("piece", "?")[:14]
            sk = d.get("skill_level", "?")
            mc = d.get("moment_count", 0)
            pv_val = d.get("positive_validity")
            fa_val = d.get("framing_alignment")
            div_val = d.get("dimension_diversity", 0)
            print(f"  {vid:<20} {piece:<16} {sk:>3} {mc:>4}"
                  f" {_pct(pv_val)} {_pct(fa_val)} {div_val:>5.2f}")


def print_mode_detection(report: dict, details: dict) -> None:
    """Mode detection: drilling episodes, transition counts, spurious transitions."""
    print("\n" + "=" * 64)
    print("MODE DETECTION")
    print("=" * 64)

    md_meta = report.get("metadata", {}).get("mode_detection", {})
    md_data = details.get("mode_detection_data", [])

    if not md_meta and not md_data:
        print("\n  No mode detection data.")
        return

    # Summary from metadata
    total = md_meta.get("total_sessions", len(md_data))
    drilling = md_meta.get("sessions_with_drilling", 0)
    mean_trans = md_meta.get("mean_transitions_per_session", 0)
    mean_spur = md_meta.get("mean_spurious_per_session", 0)

    print(f"\n  Total sessions:          {total}")
    print(f"  Sessions with drilling:  {drilling}")
    print(f"  Mean transitions/sess:   {mean_trans}")
    print(f"  Mean spurious/sess:      {mean_spur}")

    # Per-recording table
    if md_data:
        print(f"\n  {'Video':<24} {'Trans':>6} {'Drill':>6} {'Spur':>6} {'Chunks':>7}")
        print("  " + "-" * 51)
        for d in sorted(md_data, key=lambda x: x.get("video_id", "")):
            vid = d.get("video_id", "?")[:22]
            trans = d.get("transition_count", 0)
            drill = d.get("drilling_episodes", 0)
            spur = d.get("spurious_transitions", 0)
            chunks = d.get("total_chunks", 0)
            spur_flag = " !" if spur > 0 else ""
            print(f"  {vid:<24} {trans:>6} {drill:>6} {spur:>5}{spur_flag} {chunks:>7}")

        if any(d.get("spurious_transitions", 0) > 0 for d in md_data):
            print("\n  ! = spurious transition detected (reversal within 2 chunks)")


def print_synthesis(report: dict, details: dict) -> None:
    """Synthesis quality: per-criterion pass rates for Pass A and Pass B, piece context delta."""
    print("\n" + "=" * 64)
    print("SYNTHESIS QUALITY")
    print("=" * 64)

    metrics = report.get("metrics", {})
    synth_meta = report.get("metadata", {}).get("synthesis", {})
    synth_results = details.get("synthesis_judge_results", [])

    if not synth_results and not synth_meta:
        print("\n  No synthesis quality data.")
        return

    pass_a_count = synth_meta.get("pass_a_count", 0)
    pass_b_count = synth_meta.get("pass_b_count", 0)
    print(f"\n  Pass A (with context):  {pass_a_count} syntheses")
    print(f"  Pass B (zero-config):   {pass_b_count} syntheses")

    # Collect criteria from metrics keys
    criteria = set()
    for key in metrics:
        if key.startswith("synthesis_A_"):
            criteria.add(key[len("synthesis_A_"):])
        elif key.startswith("synthesis_B_"):
            criteria.add(key[len("synthesis_B_"):])

    if not criteria:
        # Try to collect from raw results
        for r in synth_results:
            criteria.update(r.get("scores", {}).keys())

    if not criteria:
        print("\n  No synthesis criteria found.")
        return

    delta = synth_meta.get("piece_context_delta", {})

    print(f"\n  {'Criterion':<28} {'Pass A':>8} {'Pass B':>8} {'Delta':>8}")
    print("  " + "-" * 54)

    for criterion in sorted(criteria):
        a_key = f"synthesis_A_{criterion}"
        b_key = f"synthesis_B_{criterion}"
        a_m = metrics.get(a_key, {})
        b_m = metrics.get(b_key, {})

        a_val = a_m.get("mean") if a_m else None
        b_val = b_m.get("mean") if b_m else None

        # Fall back to computing from details if metrics missing
        if a_val is None and synth_results:
            pass_a = [r for r in synth_results if r.get("pass") == "A"]
            a_scores = [1 if r["scores"].get(criterion) else 0
                        for r in pass_a if criterion in r.get("scores", {})]
            a_val = _safe_mean(a_scores)
        if b_val is None and synth_results:
            pass_b = [r for r in synth_results if r.get("pass") == "B"]
            b_scores = [1 if r["scores"].get(criterion) else 0
                        for r in pass_b if criterion in r.get("scores", {})]
            b_val = _safe_mean(b_scores)

        d_val = delta.get(criterion)
        if d_val is None and a_val is not None and b_val is not None:
            d_val = a_val - b_val

        print(f"  {criterion:<28} {_pct(a_val, 8)} {_pct(b_val, 8)} {_flt(d_val, 3, 8)}")


def print_score_following(report: dict, details: dict) -> None:
    """Score following: bar coverage, tier distribution, bar plausibility."""
    print("\n" + "=" * 64)
    print("SCORE FOLLOWING")
    print("=" * 64)

    sf_meta = report.get("metadata", {}).get("score_following", {})
    sf_data = details.get("score_following_data", [])
    metrics = report.get("metrics", {})

    if not sf_meta and not sf_data:
        print("\n  No score following data.")
        return

    # Aggregate metrics
    bar_cov = sf_meta.get("mean_bar_coverage", 0)
    tier1_rate = sf_meta.get("mean_tier_1_rate", 0)
    print(f"\n  Mean bar coverage:     {bar_cov:.3f}")
    print(f"  Mean Tier 1 rate:      {tier1_rate:.3f}")

    # Gate metrics
    cov_m = metrics.get("sf_bar_coverage", {})
    plaus_m = metrics.get("sf_bar_plausibility", {})

    print(f"\n  {'Metric':<24} {'Value':>8} {'Gate':>8} {'N':>5} {'Status':>8}")
    print("  " + "-" * 55)
    for label, m in [("Bar coverage", cov_m), ("Bar plausibility", plaus_m)]:
        if m:
            mean = m.get("mean", 0)
            gate = m.get("pass_threshold")
            n = m.get("n", 0)
            status = "PASS" if m.get("passed") else "FAIL"
            gate_str = f"{gate:.3f}" if gate is not None else "---"
            print(f"  {label:<24} {mean:>8.3f} {gate_str:>8} {n:>5} {status:>8}")

    # Tier distribution across all recordings
    if sf_data:
        tier_counts: dict[str, int] = defaultdict(int)
        for d in sf_data:
            for tier_key, count in d.get("tier_distribution", {}).items():
                tier_counts[tier_key] += count

        total_tiers = sum(tier_counts.values())
        if total_tiers > 0:
            print(f"\n  Tier distribution (across all recordings):")
            for tier in sorted(tier_counts.keys()):
                count = tier_counts[tier]
                pct = count / total_tiers
                bar = "#" * int(pct * 30)
                print(f"    Tier {tier}: {count:>5} ({pct:>6.1%})  {bar}")

        # Per-recording table
        print(f"\n  {'Video':<24} {'Cov':>6} {'T1%':>6} {'Plaus':>7}")
        print("  " + "-" * 45)
        for d in sorted(sf_data, key=lambda x: x.get("video_id", "")):
            vid = d.get("video_id", "?")[:22]
            cov = d.get("bar_coverage", 0)
            t1 = d.get("tier_1_rate", 0)
            pl = d.get("bar_plausibility")
            pl_str = f"{pl:.3f}" if pl is not None else "---"
            print(f"  {vid:<24} {cov:>6.3f} {t1:>5.1%} {pl_str:>7}")


def print_differentiation(report: dict, details: dict) -> None:
    """Differentiation: triplet results (skill levels 1 vs 3 vs 5)."""
    print("\n" + "=" * 64)
    print("DIFFERENTIATION")
    print("=" * 64)

    diff_data = report.get("metadata", {}).get("differentiation", [])

    if not diff_data:
        print("\n  No differentiation data (requires skill levels 1, 3, 5 per piece).")
        return

    # Collect all criteria across triplets
    all_criteria: set[str] = set()
    for d in diff_data:
        all_criteria.update(d.get("scores", {}).keys())

    print(f"\n  Triplets evaluated: {len(diff_data)}")

    # Per-piece results
    print(f"\n  {'Piece':<24}", end="")
    for c in sorted(all_criteria):
        print(f" {c[:10]:>10}", end="")
    print()
    print("  " + "-" * (24 + 11 * len(all_criteria)))

    for d in sorted(diff_data, key=lambda x: x.get("piece", "")):
        piece = d.get("piece", "?")[:22]
        scores = d.get("scores", {})
        print(f"  {piece:<24}", end="")
        for c in sorted(all_criteria):
            val = scores.get(c)
            if val is True:
                print(f"  {'PASS':>8}", end="")
            elif val is False:
                print(f"  {'FAIL':>8}", end="")
            else:
                print(f"  {'---':>8}", end="")
        print()

    # Aggregate pass rates
    if len(diff_data) > 1:
        print(f"\n  Aggregate pass rates:")
        for c in sorted(all_criteria):
            vals = [d["scores"].get(c) for d in diff_data if d["scores"].get(c) is not None]
            if vals:
                passed = sum(1 for v in vals if v)
                total = len(vals)
                print(f"    {c:<24} {passed}/{total} ({passed/total:.0%})")

    # Evidence snippets
    print(f"\n  Evidence:")
    for d in diff_data:
        piece = d.get("piece", "?")
        evidence = d.get("evidence", {})
        for c, ev in evidence.items():
            if ev:
                print(f"    [{piece}] {c}: {str(ev)[:80]}")


def print_efficiency(report: dict, details: dict) -> None:
    """Efficiency summary: synthesis latency p50/p90."""
    print("\n" + "=" * 64)
    print("EFFICIENCY")
    print("=" * 64)

    eff_meta = report.get("metadata", {}).get("efficiency", {})
    eff_data = details.get("efficiency_data", [])

    synth_latencies = [
        e["synthesis_latency_ms"] for e in eff_data
        if e.get("synthesis_latency_ms")
    ]

    if not eff_meta and not synth_latencies:
        print("\n  No efficiency data.")
        return

    if synth_latencies:
        p50 = _percentile(synth_latencies, 0.5)
        p90 = _percentile(synth_latencies, 0.9)
        mean_lat = _safe_mean(synth_latencies)
        print(f"\n  Synthesis latency (n={len(synth_latencies)}):")
        print(f"    Mean:  {mean_lat:>8.0f} ms")
        print(f"    P50:   {p50:>8.0f} ms")
        print(f"    P90:   {p90:>8.0f} ms")

        # Break down by pass
        for pass_label in ["A", "B"]:
            lats = [
                e["synthesis_latency_ms"] for e in eff_data
                if e.get("synthesis_latency_ms") and e.get("pass") == pass_label
            ]
            if lats:
                p50 = _percentile(lats, 0.5)
                p90 = _percentile(lats, 0.9)
                print(f"\n  Pass {pass_label} (n={len(lats)}):")
                print(f"    P50:   {p50:>8.0f} ms")
                print(f"    P90:   {p90:>8.0f} ms")
    elif eff_meta:
        print(f"\n  Mean synthesis latency: {eff_meta.get('mean_synthesis_latency_ms', 'N/A')} ms")
        print(f"  P90 synthesis latency:  {eff_meta.get('p90_synthesis_latency_ms', 'N/A')} ms")


def print_cost(report: dict) -> None:
    """Cost summary."""
    print("\n" + "=" * 64)
    print("COST")
    print("=" * 64)

    cost = report.get("cost", {})
    if not cost:
        print("\n  No cost data.")
        return

    print(f"\n  Judge calls:     {cost.get('judge_calls', 0)}")
    print(f"  Estimated cost:  ${cost.get('estimated_usd', 0):.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SECTION_MAP = {
    "piece_id": print_piece_id,
    "stop": print_stop,
    "teaching_moments": print_teaching_moments,
    "mode_detection": print_mode_detection,
    "synthesis": print_synthesis,
    "score_following": print_score_following,
    "differentiation": print_differentiation,
}


def main():
    parser = argparse.ArgumentParser(
        description="Analyze E2E pipeline eval results"
    )
    parser.add_argument(
        "--report",
        default=str(REPORTS_DIR / "practice_eval.json"),
        help="Path to practice_eval.json (default: reports/practice_eval.json)",
    )
    parser.add_argument(
        "--details",
        default=str(REPORTS_DIR / "practice_eval_details.json"),
        help="Path to practice_eval_details.json (default: reports/practice_eval_details.json)",
    )
    parser.add_argument(
        "--capability",
        default=None,
        choices=CAPABILITIES,
        help="Print only one capability section",
    )
    args = parser.parse_args()

    report_path = Path(args.report)
    details_path = Path(args.details)

    if not report_path.exists():
        print(f"Report not found: {report_path}")
        sys.exit(1)

    with open(report_path) as f:
        report = json.load(f)

    details: dict = {}
    if details_path.exists():
        with open(details_path) as f:
            details = json.load(f)
    else:
        print(f"Warning: details file not found: {details_path}")
        print("  Some per-recording breakdowns will be unavailable.\n")

    # Header
    meta = report.get("metadata", {})
    print("=" * 64)
    print(f"E2E PIPELINE EVAL ANALYSIS")
    print(f"  eval:       {report.get('eval_name', '?')} v{report.get('eval_version', '?')}")
    print(f"  dataset:    {report.get('dataset', '?')}")
    print(f"  recordings: {meta.get('total_recordings', '?')}")
    print(f"  judged:     {', '.join(meta.get('capabilities_judged', []))}")
    print(f"  git:        {meta.get('git_sha', '?')[:10]}"
          f"{'  (dirty)' if meta.get('git_dirty') else ''}")
    print("=" * 64)

    # Print sections
    if args.capability:
        printer = SECTION_MAP.get(args.capability)
        if printer:
            printer(report, details)
        else:
            print(f"Unknown capability: {args.capability}")
            sys.exit(1)
    else:
        for cap in CAPABILITIES:
            SECTION_MAP[cap](report, details)

    # Always print efficiency and cost
    print_efficiency(report, details)
    print_cost(report)

    # Footer
    print("\n" + "=" * 64)
    print(f"Report:   {report_path}")
    print(f"Details:  {details_path}")
    print("=" * 64)


if __name__ == "__main__":
    main()
