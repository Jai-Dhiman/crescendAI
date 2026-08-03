# model/src/follower_eval/gold_report.py
# ruff: noqa: E501
"""Gold-track report + PASS/FAIL verdict for the real-audio follower eval (#133 S3).

Scans ``<bundles-root>/<piece>/<vid>.gold.json`` (written by
``follower_eval.tap_tool``), runs the accuracy metric per clip
(``follower_eval.accuracy``), and prints per-clip bar-localization distributions
plus a POOLED verdict. Pooling is over TAPS, not medians-of-medians: 72% of real
clips contain repeats, so a per-clip median hides exactly the localization we
care about (the proxy-track lesson, made concrete here).

RUNNING (from the PRIMARY checkout so data/ + the venv resolve):

  cd /Users/jdhiman/Documents/crescendai/model
  PYTHONPATH=<worktree>/model/src .venv/bin/python -m follower_eval.gold_report \
    --bundles-root data/evals/realaudio_bundles --scores-root data/scores \
    --out /tmp/gold_report.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from follower_eval.accuracy import (
    TOL_BARS_LENIENT,
    TOL_BARS_STRICT,
    ClipAccuracy,
    clip_to_jsonable,
    evaluate_clip,
)
from follower_eval.realaudio import (
    SCORE_FILENAME_BY_PIECE,
    RealAudioEvalError,
    load_score,
)

# --- PROVISIONAL PASS bars -------------------------------------------------
# These are PLACEHOLDERS, not the final gate. The #133 methodology is to set the
# gate from the OBSERVED gold distribution once ~20-30 clips are tapped (per-clip
# spread, not one median). Until then the verdict is advisory and clearly flagged
# PROVISIONAL in the printout. The provisional bar encodes the minimum a live
# score cursor needs: land on the right measure (within 1 bar) on the large
# majority of downbeats.
PROVISIONAL_PASS = {
    "min_within_1bar_frac": 0.85,  # >=85% of taps localized to within one bar
    "max_median_abs_err_bars": 0.50,  # typical error under half a bar
}


def _load_measure_table(score_path: Path) -> tuple[list, tuple, list[dict]]:
    """Score notes + bar boundary columns + measure_table (the bar->sec map the
    accuracy metric needs). Reuses the proxy runner's loader, then re-reads the
    measure_table (load_score drops it)."""
    from chroma_dtw_eval.amt_regen import _load_bach_json_score

    score_notes, bar_boundaries, _span = load_score(score_path)
    _na, measure_table, _sha, _beat = _load_bach_json_score(score_path)
    return score_notes, bar_boundaries, measure_table


def discover_gold(bundles_root: Path) -> dict[str, list[tuple[Path, Path]]]:
    """piece -> [(bundle_path, gold_path)] for every ``*.gold.json`` whose bundle
    exists. Loud if a gold file has no matching bundle (a stale/misplaced label,
    never silently skipped)."""
    out: dict[str, list[tuple[Path, Path]]] = {}
    for piece_dir in sorted(p for p in bundles_root.iterdir() if p.is_dir()):
        piece = piece_dir.name
        if piece not in SCORE_FILENAME_BY_PIECE:
            continue
        pairs: list[tuple[Path, Path]] = []
        for gold in sorted(piece_dir.glob("*.gold.json")):
            vid = gold.name[: -len(".gold.json")]
            bundle = piece_dir / f"{vid}.json"
            if not bundle.exists():
                raise RealAudioEvalError(
                    f"gold label {gold} has no bundle {bundle.name} (rebuild corpus?)"
                )
            pairs.append((bundle, gold))
        if pairs:
            out[piece] = pairs
    return out


def run(bundles_root: Path, scores_root: Path, pieces: list[str] | None = None) -> dict:
    """Evaluate every gold-labeled clip. Returns per-clip accuracies, the pooled
    tap-level distribution, and a PROVISIONAL verdict."""
    by_piece = discover_gold(bundles_root)
    if pieces:
        by_piece = {p: v for p, v in by_piece.items() if p in pieces}

    clips: list[ClipAccuracy] = []
    failures: list[dict] = []
    for piece, pairs in by_piece.items():
        score_path = scores_root / SCORE_FILENAME_BY_PIECE[piece]
        try:
            score_notes, bar_boundaries, measure_table = _load_measure_table(score_path)
        except Exception as exc:
            failures.append(
                {
                    "piece": piece,
                    "bundle": None,
                    "error": f"score load: {type(exc).__name__}: {exc}",
                }
            )
            continue
        for bundle_path, gold_path in pairs:
            try:
                clips.append(
                    evaluate_clip(
                        piece,
                        bundle_path,
                        gold_path,
                        score_notes,
                        bar_boundaries,
                        measure_table,
                    )
                )
            except Exception as exc:
                failures.append(
                    {
                        "piece": piece,
                        "bundle": bundle_path.stem,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    pooled = _pool(clips)
    verdict = _verdict(pooled)
    return {
        "clips": clips,
        "failures": failures,
        "pooled": pooled,
        "verdict": verdict,
        "n_clips": len(clips),
        "n_pieces": len({c.piece for c in clips}),
        "tolerances": {
            "lenient_bars": TOL_BARS_LENIENT,
            "strict_bars": TOL_BARS_STRICT,
        },
    }


def _pool(clips: list[ClipAccuracy]) -> dict | None:
    """Tap-level pooled distribution across all labeled clips (NOT a mean of
    per-clip medians). Returns None when nothing is labeled yet."""
    if not clips:
        return None
    errs_sec: list[float] = []
    errs_bars: list[float] = []
    relock: list[float] = []
    n_taps = n_restarts = n_no_relock = 0
    for c in clips:
        n_taps += c.n_taps
        n_restarts += c.n_restarts
        n_no_relock += c.n_restart_no_relock
        relock.extend(c.relock_latencies_sec)
        for te in c.tap_errors:
            if te.abs_err_sec is not None:
                errs_sec.append(te.abs_err_sec)
            if te.abs_err_bars is not None:
                errs_bars.append(te.abs_err_bars)
    if not errs_bars:
        return {"n_taps": n_taps, "n_decoded": 0, "note": "no taps could be decoded"}
    return {
        "n_taps": n_taps,
        "n_decoded": len(errs_bars),
        "median_abs_err_sec": round(statistics.median(errs_sec), 3),
        "median_abs_err_bars": round(statistics.median(errs_bars), 3),
        "p90_abs_err_bars": round(_pctl(errs_bars, 0.9), 3),
        "within_1bar_frac": round(
            sum(e <= TOL_BARS_LENIENT for e in errs_bars) / len(errs_bars), 4
        ),
        "within_half_bar_frac": round(
            sum(e <= TOL_BARS_STRICT for e in errs_bars) / len(errs_bars), 4
        ),
        "n_restarts": n_restarts,
        "n_restart_relocked": len(relock),
        "n_restart_no_relock": n_no_relock,
        "median_relock_sec": round(statistics.median(relock), 3) if relock else None,
        "p90_relock_sec": round(_pctl(relock, 0.9), 3) if relock else None,
    }


def _pctl(values: list[float], q: float) -> float:
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def _verdict(pooled: dict | None) -> dict:
    """PROVISIONAL PASS/FAIL against placeholder bars. reasons[] always lists
    every check so the printout is self-explanatory even on a partial label set."""
    if not pooled or not pooled.get("n_decoded"):
        return {
            "status": "NO_DATA",
            "provisional": True,
            "reasons": ["no gold-labeled clips decoded yet -- run the tap tool"],
        }
    w1 = pooled["within_1bar_frac"]
    med = pooled["median_abs_err_bars"]
    checks = [
        (
            w1 >= PROVISIONAL_PASS["min_within_1bar_frac"],
            f"within_1bar_frac {w1:.3f} vs >= {PROVISIONAL_PASS['min_within_1bar_frac']}",
        ),
        (
            med <= PROVISIONAL_PASS["max_median_abs_err_bars"],
            f"median_abs_err_bars {med:.3f} vs <= {PROVISIONAL_PASS['max_median_abs_err_bars']}",
        ),
    ]
    passed = all(ok for ok, _ in checks)
    return {
        "status": "PASS" if passed else "FAIL",
        "provisional": True,
        "reasons": [("PASS " if ok else "FAIL ") + msg for ok, msg in checks],
    }


def _format(result: dict) -> str:
    L = [
        "=" * 82,
        "REAL-AUDIO FOLLOWER EVAL (#133) -- GOLD track -- human bar-tap accuracy",
        "=" * 82,
        f"labeled clips: {result['n_clips']}   pieces: {result['n_pieces']}   "
        f"failures: {len(result['failures'])}",
    ]
    if result["n_clips"] == 0:
        L.append("")
        L.append("No *.gold.json labels found. Tap some clips first:")
        L.append(
            "  PYTHONPATH=<wt>/model/src .venv/bin/python -m follower_eval.tap_tool --serve"
        )
    else:
        hdr = (
            f"{'piece':<22}{'clip':<13}{'taps':>5}{'dec':>5}{'errsec':>8}"
            f"{'errbar':>8}{'<=1bar':>8}{'<=.5':>7}{'rst':>5}{'relok_s':>9}"
        )
        L.append("")
        L.append(hdr)
        L.append("-" * len(hdr))
        for c in sorted(result["clips"], key=lambda c: (c.piece, c.bundle)):
            eb = (
                f"{c.median_abs_err_bars:.2f}"
                if c.median_abs_err_bars is not None
                else " n/a"
            )
            es = (
                f"{c.median_abs_err_sec:.2f}"
                if c.median_abs_err_sec is not None
                else " n/a"
            )
            w1 = (
                f"{c.within_1bar_frac:.2f}"
                if c.within_1bar_frac is not None
                else " n/a"
            )
            wh = (
                f"{c.within_half_bar_frac:.2f}"
                if c.within_half_bar_frac is not None
                else " n/a"
            )
            rl = (
                f"{statistics.median(c.relock_latencies_sec):.1f}"
                if c.relock_latencies_sec
                else ("none" if c.n_restarts else "-")
            )
            L.append(
                f"{c.piece:<22}{c.bundle[:12]:<13}{c.n_taps:>5}{c.n_decoded:>5}"
                f"{es:>8}{eb:>8}{w1:>8}{wh:>7}{c.n_restarts:>5}{rl:>9}"
            )
        L.append("-" * len(hdr))
        p = result["pooled"]
        L.append("POOLED (tap-level, all clips):")
        L.append(
            f"  n_taps={p['n_taps']} decoded={p['n_decoded']}  "
            f"median_err={p.get('median_abs_err_sec')}s / {p.get('median_abs_err_bars')} bars  "
            f"p90={p.get('p90_abs_err_bars')} bars"
        )
        L.append(
            f"  within 1 bar={p.get('within_1bar_frac')}  within 0.5 bar={p.get('within_half_bar_frac')}"
        )
        mr = p.get("median_relock_sec")
        pr = p.get("p90_relock_sec")
        rl = f"median relock={mr}s p90={pr}s" if mr is not None else "no relock events"
        L.append(
            f"  restarts={p.get('n_restarts')} relocked={p.get('n_restart_relocked')} "
            f"never={p.get('n_restart_no_relock')}  {rl}"
        )
    v = result["verdict"]
    L.append("")
    L.append(
        f"VERDICT: {v['status']}"
        + ("  (PROVISIONAL bars -- see doc)" if v.get("provisional") else "")
    )
    for r in v["reasons"]:
        L.append(f"  - {r}")
    if result["failures"]:
        L.append("")
        L.append(f"FAILURES ({len(result['failures'])}):")
        for f in result["failures"][:20]:
            L.append(f"  {f['piece']}/{f['bundle']} -> {f['error']}")
    return "\n".join(L)


def _to_jsonable(result: dict) -> dict:
    out = dict(result)
    out["clips"] = [clip_to_jsonable(c) for c in result["clips"]]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Real-audio follower eval -- gold track report (#133)"
    )
    ap.add_argument(
        "--bundles-root", type=Path, default=Path("data/evals/realaudio_bundles")
    )
    ap.add_argument("--scores-root", type=Path, default=Path("data/scores"))
    ap.add_argument("--pieces", nargs="+", default=None)
    ap.add_argument("--out", type=Path, default=None, help="write the JSON report here")
    args = ap.parse_args()

    result = run(args.bundles_root, args.scores_root, pieces=args.pieces)
    print(_format(result))
    if args.out:
        args.out.write_text(json.dumps(_to_jsonable(result), indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
