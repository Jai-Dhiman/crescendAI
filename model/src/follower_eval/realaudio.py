# model/src/follower_eval/realaudio.py
"""Proxy (scale) track of the real-audio score-follower eval (issue #133).

Runs the #119 Viterbi-HMM follower (``follow_hmm`` + ``TUNED_HMM_PARAMS``) on
real YouTube -> AMT transcribed practice bundles and reports anchor-free
structural proxies per clip and per piece. Generalizes the piece-locked,
gate1-specific ``follower_bench/realaudio_gate1.py`` into a corpus-wide runner
over ``DEFAULT_SCORE_BY_PIECE`` pieces, and drops the parangonar head-to-head
(parangonar is no longer treated as a reference -- see #133).

WHY THESE PROXIES (no synthetic ground truth exists on real audio):
  * coverage       -- fraction of transcribed notes the follower could place on
                      the score. Low coverage = the follower explained little of
                      what was actually played.
  * score_span_frac-- min->max score-seconds the alignment reaches, over the
                      score's total duration. ~1.0 = a full run-through was
                      tracked end to end; low = it stalled or collapsed early.
  * monotonicity   -- where an amateur plays straight, score position should be
                      non-decreasing; backward steps beyond a chord-noise
                      tolerance are either real repeats/restarts (expected on
                      practice audio) or alignment slips. REPORTED, not judged --
                      the gold track (S3) disambiguates which.
  * confidence     -- forward-backward posterior mass on each decoded column.
  * conf_vs_monotone -- does confidence DROP at backward steps? A follower that
                      is unsure exactly where it jumps is the "knows when it's
                      lost" property the cursor UI depends on.

RUNNING (data/ is gitignored -> absent in the worktree; run from the PRIMARY
checkout so CWD-relative data paths + the primary .venv resolve):

  cd /Users/jdhiman/Documents/crescendai/model
  .venv/bin/python -m follower_eval.realaudio \
    --bundles-root data/evals/claim_bundles \
    --scores-root data/scores \
    --out /tmp/realaudio_eval.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from chroma_dtw_eval.amt_regen import DEFAULT_SCORE_BY_PIECE, _load_bach_json_score
from follower_bench.follower import bar_boundary_columns
from follower_bench.hmm import TUNED_HMM_PARAMS, follow_hmm
from follower_bench.score_notes import ScoreNote
from follower_bench.segments import PerfNote

# Score-position backward step (seconds) below which a step is chord-internal
# ordering noise, not a genuine regression. Scores render in real seconds via
# per-note onset_seconds; 0.5s is well under a beat at typical tempi. Matches
# realaudio_gate1.MONO_TOL_SEC so the two runners' monotonicity is comparable.
MONO_TOL_SEC = 0.5


class RealAudioEvalError(RuntimeError):
    """Raised when a bundle or score cannot be loaded as required -- loud, never
    a silent skip that would inflate the corpus with empty cells."""


# Piece -> score FILENAME (not the module-anchored absolute path in
# DEFAULT_SCORE_BY_PIECE, which points into whichever checkout imported it).
# Joined with an explicit --scores-root so the runner works from any CWD.
SCORE_FILENAME_BY_PIECE = {piece: path.name for piece, path in DEFAULT_SCORE_BY_PIECE.items()}


@dataclass(frozen=True)
class ClipProxies:
    """One real transcribed clip's anchor-free proxy measurement."""
    piece: str
    bundle: str
    n_perf: int
    n_matched: int
    coverage: float
    score_span_sec: tuple[float, float]
    score_span_frac: float
    confidence_median: float | None
    conf_vs_monotone_spearman: float | None
    backward_steps: int
    backward_frac: float
    max_backstep_sec: float
    transpose_semitones: int
    wall_sec: float


def load_bundle_notes(bundle_path: Path) -> list[PerfNote]:
    """Transcribed-bundle note dicts -> PerfNote list sorted by onset (the HMM
    is monotone in perf order and assumes ascending onsets).

    Raises:
        RealAudioEvalError: the bundle has no non-empty ``notes`` array (an
            un-transcribed or empty clip is a data error, not a 0-coverage cell).
    """
    body = json.loads(bundle_path.read_text())
    notes = body.get("notes")
    if not notes:
        raise RealAudioEvalError(f"{bundle_path.name}: bundle has no 'notes' (not transcribed?)")
    pn = [
        PerfNote(
            onset=float(n["onset"]),
            offset=float(n.get("offset", n["onset"])),
            pitch=int(n["pitch"]),
            velocity=int(n.get("velocity", 0)),
        )
        for n in notes
    ]
    pn.sort(key=lambda p: p.onset)
    return pn


def load_score(score_path: Path) -> tuple[list[ScoreNote], tuple[int, ...], float]:
    """Load a rep score JSON into (score_notes, bar_boundaries, score_span_sec)
    via the production ``_load_bach_json_score`` (variable-tempo / non-4/4 safe,
    #98). score_notes.position is score-render SECONDS; bar_boundaries are the
    DP columns the HMM may jump to (bar downbeats)."""
    score_na, measure_table, _sha, _beat_sec = _load_bach_json_score(score_path)
    score_notes = [ScoreNote(pitch=int(s["pitch"]), position=float(s["onset_sec"])) for s in score_na]
    downbeats = sorted({float(r["start_sec"]) for r in measure_table})
    bar_boundaries = bar_boundary_columns([n.position for n in score_notes], downbeats)
    span = float(score_na["onset_sec"].max() - score_na["onset_sec"].min()) if len(score_na) else 0.0
    return score_notes, bar_boundaries, span


def _spearman(a: list[float], b: list[float]) -> float | None:
    """Rank correlation without scipy (one number, avoid the heavy import).
    Returns None when undefined (n<3 or zero variance)."""
    if len(a) < 3:
        return None
    if len(set(a)) < 2 or len(set(b)) < 2:  # constant input -> correlation undefined
        return None
    import numpy as np
    av, bv = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(av)).astype(float)
    rb = np.argsort(np.argsort(bv)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = float((ra ** 2).sum() * (rb ** 2).sum()) ** 0.5
    if denom == 0.0:
        return None
    return float((ra * rb).sum() / denom)


def _mono_stats(score_positions: list[float]) -> tuple[int, float, float]:
    """(score_sec sorted by perf order) -> (backward_steps, backward_frac,
    max_backstep_sec). A straight rendition is non-decreasing; steps below
    -MONO_TOL_SEC count as backward."""
    if len(score_positions) < 2:
        return 0, 0.0, 0.0
    deltas = [score_positions[i] - score_positions[i - 1] for i in range(1, len(score_positions))]
    back = [d for d in deltas if d < -MONO_TOL_SEC]
    return len(back), len(back) / len(deltas), (-min(back) if back else 0.0)


def run_clip(piece: str, bundle_path: Path, score_notes: list[ScoreNote],
             bar_boundaries: tuple[int, ...], score_span_sec: float) -> ClipProxies:
    """Follow one transcribed clip with the tuned HMM and compute its proxies."""
    perf = load_bundle_notes(bundle_path)
    t0 = time.perf_counter()
    est = follow_hmm(perf, score_notes, TUNED_HMM_PARAMS, bar_boundaries=bar_boundaries)
    wall = time.perf_counter() - t0

    ms = sorted(est.matches, key=lambda m: m.perf_time)
    positions = [m.score_position for m in ms]
    confs = [m.confidence for m in ms if m.confidence is not None]
    back_steps, back_frac, max_back = _mono_stats(positions)

    # calibration: is confidence LOWER at backward steps? step_ok[k]=1 if this
    # match steps forward, 0 if it steps back. Positive spearman => confident
    # where monotone (well-calibrated); we want conf to fall where it jumps.
    calib = None
    if len(ms) >= 4 and len(confs) == len(ms):
        step_ok = [1.0] + [1.0 if positions[k] - positions[k - 1] >= -MONO_TOL_SEC else 0.0
                           for k in range(1, len(positions))]
        calib = _spearman([m.confidence for m in ms], step_ok)

    smin = min(positions, default=0.0)
    smax = max(positions, default=0.0)
    return ClipProxies(
        piece=piece,
        bundle=bundle_path.stem,
        n_perf=len(perf),
        n_matched=len(ms),
        coverage=round(len(ms) / len(perf), 4) if perf else 0.0,
        score_span_sec=(round(smin, 2), round(smax, 2)),
        score_span_frac=round((smax - smin) / score_span_sec, 4) if score_span_sec else 0.0,
        confidence_median=round(statistics.median(confs), 4) if confs else None,
        conf_vs_monotone_spearman=round(calib, 4) if calib is not None else None,
        backward_steps=back_steps,
        backward_frac=round(back_frac, 4),
        max_backstep_sec=round(max_back, 2),
        transpose_semitones=est.transpose_semitones,
        wall_sec=round(wall, 2),
    )


def discover_bundles(bundles_root: Path) -> dict[str, list[Path]]:
    """Map piece_id -> sorted transcribed-bundle paths under bundles_root.
    Layout: ``<bundles_root>/<piece_id>/<video_id>.json`` (skips ``*.meta.json``
    and any ``_index.json``). Only pieces in SCORE_FILENAME_BY_PIECE are kept;
    an unknown piece dir is reported, never silently followed against no score."""
    out: dict[str, list[Path]] = {}
    for piece_dir in sorted(p for p in bundles_root.iterdir() if p.is_dir()):
        piece = piece_dir.name
        if piece not in SCORE_FILENAME_BY_PIECE:
            continue
        bundles = sorted(
            p for p in piece_dir.glob("*.json")
            if not p.name.endswith(".meta.json") and p.name != "_index.json"
        )
        if bundles:
            out[piece] = bundles
    return out


def run(bundles_root: Path, scores_root: Path,
        pieces: list[str] | None = None) -> dict:
    """Run the proxy eval over every transcribed bundle. Loads each piece's
    score once. Records per-clip proxies and per-clip/per-score failures loudly
    (never a silent drop)."""
    by_piece = discover_bundles(bundles_root)
    if pieces:
        by_piece = {p: v for p, v in by_piece.items() if p in pieces}
    if not by_piece:
        raise RealAudioEvalError(
            f"no transcribed bundles for known pieces under {bundles_root} "
            f"(pieces filter={pieces})")

    clips: list[ClipProxies] = []
    failures: list[dict] = []
    unknown_dirs = [p.name for p in bundles_root.iterdir()
                    if p.is_dir() and p.name not in SCORE_FILENAME_BY_PIECE]

    for piece, bundle_paths in by_piece.items():
        score_path = scores_root / SCORE_FILENAME_BY_PIECE[piece]
        try:
            score_notes, bar_boundaries, span = load_score(score_path)
        except Exception as exc:  # loud: whole piece recorded as failed, not skipped
            failures.append({"piece": piece, "bundle": None,
                             "error": f"score load: {type(exc).__name__}: {exc}"})
            continue
        for bp in bundle_paths:
            try:
                clips.append(run_clip(piece, bp, score_notes, bar_boundaries, span))
            except Exception as exc:
                failures.append({"piece": piece, "bundle": bp.stem,
                                 "error": f"{type(exc).__name__}: {exc}"})

    return {
        "clips": clips,
        "failures": failures,
        "unknown_piece_dirs": unknown_dirs,
        "per_piece": _aggregate(clips),
        "overall": _aggregate(clips, group=False).get("_all"),
        "n_clips": len(clips),
        "n_pieces": len({c.piece for c in clips}),
    }


def _aggregate(clips: list[ClipProxies], group: bool = True) -> dict:
    """Median proxies per piece (group=True) or over all clips (group=False,
    under key ``_all``). Medians are robust to the one weird clip."""
    buckets: dict[str, list[ClipProxies]] = {}
    for c in clips:
        buckets.setdefault(c.piece if group else "_all", []).append(c)
    out: dict[str, dict] = {}
    for key, group_clips in buckets.items():
        confs = [c.confidence_median for c in group_clips if c.confidence_median is not None]
        cals = [c.conf_vs_monotone_spearman for c in group_clips
                if c.conf_vs_monotone_spearman is not None]
        out[key] = {
            "n_clips": len(group_clips),
            "median_coverage": round(statistics.median(c.coverage for c in group_clips), 4),
            "median_span_frac": round(statistics.median(c.score_span_frac for c in group_clips), 4),
            "median_confidence": round(statistics.median(confs), 4) if confs else None,
            "median_calibration": round(statistics.median(cals), 4) if cals else None,
            "median_backward_frac": round(statistics.median(c.backward_frac for c in group_clips), 4),
            "total_backward_steps": sum(c.backward_steps for c in group_clips),
        }
    return out


def _format_report(result: dict) -> str:
    lines = ["=" * 78,
             "REAL-AUDIO FOLLOWER EVAL (#133) -- proxy track -- #119 HMM (TUNED_HMM_PARAMS)",
             "=" * 78,
             f"clips: {result['n_clips']}   pieces: {result['n_pieces']}   "
             f"failures: {len(result['failures'])}"]
    if result["unknown_piece_dirs"]:
        lines.append(f"unknown piece dirs (no score, skipped): {result['unknown_piece_dirs']}")
    lines.append("")
    hdr = (f"{'piece':<24}{'clip':<18}{'cov':>7}{'span':>7}{'conf':>7}"
           f"{'cal':>7}{'back':>7}{'bstep_s':>9}")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for c in result["clips"]:
        conf = f"{c.confidence_median:.2f}" if c.confidence_median is not None else "  n/a"
        cal = f"{c.conf_vs_monotone_spearman:+.2f}" if c.conf_vs_monotone_spearman is not None else " n/a"
        lines.append(f"{c.piece:<24}{c.bundle[:17]:<18}{c.coverage:>7.2f}"
                     f"{c.score_span_frac:>7.2f}{conf:>7}{cal:>7}"
                     f"{c.backward_frac:>7.2f}{c.max_backstep_sec:>9.1f}")
    lines.append("-" * len(hdr))
    lines.append("PER-PIECE MEDIANS")
    for piece, agg in sorted(result["per_piece"].items()):
        conf = f"{agg['median_confidence']:.2f}" if agg["median_confidence"] is not None else "n/a"
        cal = f"{agg['median_calibration']:+.2f}" if agg["median_calibration"] is not None else "n/a"
        lines.append(f"  {piece:<24} n={agg['n_clips']:<3} cov={agg['median_coverage']:.2f} "
                     f"span={agg['median_span_frac']:.2f} conf={conf} cal={cal} "
                     f"back={agg['median_backward_frac']:.2f}")
    if result["overall"]:
        o = result["overall"]
        lines.append("")
        lines.append(f"OVERALL MEDIANS  cov={o['median_coverage']:.2f} span={o['median_span_frac']:.2f} "
                     f"conf={o['median_confidence']} cal={o['median_calibration']} "
                     f"back={o['median_backward_frac']:.2f}")
    if result["failures"]:
        lines.append("")
        lines.append(f"FAILURES ({len(result['failures'])}):")
        for f in result["failures"][:20]:
            lines.append(f"  {f['piece']}/{f['bundle']} -> {f['error']}")
    return "\n".join(lines)


def _result_to_jsonable(result: dict) -> dict:
    out = dict(result)
    out["clips"] = [asdict(c) for c in result["clips"]]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Real-audio follower eval -- proxy track (#133)")
    ap.add_argument("--bundles-root", type=Path, default=Path("data/evals/claim_bundles"),
                    help="dir of <piece_id>/<video_id>.json transcribed bundles")
    ap.add_argument("--scores-root", type=Path, default=Path("data/scores"),
                    help="dir of rep score JSONs (joined with the per-piece filename)")
    ap.add_argument("--pieces", nargs="+", default=None,
                    help="restrict to these piece_ids (default: all discovered)")
    ap.add_argument("--out", type=Path, default=None, help="write the JSON report here")
    args = ap.parse_args()

    result = run(args.bundles_root, args.scores_root, pieces=args.pieces)
    print(_format_report(result))
    if args.out:
        args.out.write_text(json.dumps(_result_to_jsonable(result), indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
