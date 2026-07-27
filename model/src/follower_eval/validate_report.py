# model/src/follower_eval/validate_report.py
"""Aggregate the human validations (issue #133, Track B) into the answer the
proxy track can't give on its own: on the real amateur clips, how often does the
follower actually track, and are the low-confidence clips genuinely-hard
(right to abstain) or follower failures?

Reads ``<piece>/<vid>.validate.json`` (from ``validate_tool``) and reports the
verdict distribution, the fraction of playback flagged wrong, and -- the point --
a cross-tab of verdict against the v1 proxy confidence, so a low-confidence clip
that a human marked "junk / right to give up" counts as a correct abstention, not
a failure.

RUNNING (from the PRIMARY checkout):

  cd /Users/jdhiman/Documents/crescendai/model
  PYTHONPATH=<worktree>/model/src .venv/bin/python -m follower_eval.validate_report \
    --bundles-root data/evals/realaudio_bundles
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from follower_eval.realaudio import SCORE_FILENAME_BY_PIECE

SUBSET_JSON = Path(__file__).resolve().parent / "gold_subset.json"

# A follower "success" on a clip = it tracked, or it correctly abstained on a
# genuinely unfollowable clip. "recovered" is a partial success (relocked after a
# stop); "wrong" is the only outright failure.
GOOD_VERDICTS = {"tracked", "junk"}


def load_validations(bundles_root: Path) -> list[dict]:
    """All ``*.validate.json`` under known-piece dirs, loudest-fail on malformed."""
    out: list[dict] = []
    for piece_dir in sorted(p for p in bundles_root.iterdir() if p.is_dir()):
        if piece_dir.name not in SCORE_FILENAME_BY_PIECE:
            continue
        for v in sorted(piece_dir.glob("*.validate.json")):
            out.append(json.loads(v.read_text()))
    return out


def _confidence_map(subset_json: Path) -> dict[str, float]:
    """(piece/vid) -> v1 proxy confidence, for the adjudication cross-tab."""
    if not subset_json.exists():
        return {}
    return {f"{c['piece']}/{c['video_id']}": c["v1_confidence"]
            for c in json.loads(subset_json.read_text())["clips"]
            if c.get("v1_confidence") is not None}


def summarize(validations: list[dict], conf_map: dict[str, float]) -> dict:
    """Verdict counts, fraction-wrong stats, and the low/high-confidence
    adjudication split."""
    verdicts: dict[str, int] = {}
    fw: list[float] = []
    lowconf = {"good": 0, "bad": 0, "n": 0}   # clips with v1 conf < 0.5
    highconf = {"good": 0, "bad": 0, "n": 0}
    for v in validations:
        verdicts[v["verdict"]] = verdicts.get(v["verdict"], 0) + 1
        if v.get("fraction_wrong") is not None:
            fw.append(float(v["fraction_wrong"]))
        conf = conf_map.get(f"{v['piece']}/{v['video_id']}")
        if conf is not None:
            bucket = lowconf if conf < 0.5 else highconf
            bucket["n"] += 1
            bucket["good" if v["verdict"] in GOOD_VERDICTS else "bad"] += 1
    n = len(validations)
    good = sum(c for k, c in verdicts.items() if k in GOOD_VERDICTS)
    return {
        "n_validated": n,
        "verdicts": verdicts,
        "success_frac": round(good / n, 4) if n else None,
        "median_fraction_wrong": round(statistics.median(fw), 4) if fw else None,
        "p90_fraction_wrong": round(sorted(fw)[min(len(fw) - 1, int(0.9 * (len(fw) - 1)))], 4) if fw else None,
        "low_confidence": lowconf,   # does the follower correctly abstain when unsure?
        "high_confidence": highconf,
    }


def _format(summary: dict) -> str:
    L = ["=" * 76,
         "REAL-AUDIO FOLLOWER EVAL (#133) -- TRACK B -- human validation of amateur clips",
         "=" * 76,
         f"validated clips: {summary['n_validated']}"]
    if not summary["n_validated"]:
        L.append("")
        L.append("No *.validate.json yet. Label some clips:")
        L.append("  ...python -m follower_eval.validate_tool --precompute   # once")
        L.append("  ...python -m follower_eval.validate_tool --serve        # then label")
        return "\n".join(L)
    L.append("")
    L.append("verdicts:")
    for k in ("tracked", "recovered", "wrong", "junk"):
        L.append(f"  {k:<10} {summary['verdicts'].get(k, 0)}")
    L.append("")
    L.append(f"success (tracked or correctly-abstained): {summary['success_frac']}")
    L.append(f"fraction of playback flagged wrong: median {summary['median_fraction_wrong']} "
             f"p90 {summary['p90_fraction_wrong']}")
    L.append("")
    L.append("LOW-CONFIDENCE ADJUDICATION (proxy conf < 0.5 -- the 21% the proxy flagged):")
    lo, hi = summary["low_confidence"], summary["high_confidence"]
    L.append(f"  low-conf clips:  {lo['n']}  -> good {lo['good']} (tracked/abstained), bad {lo['bad']} (wrong)")
    L.append(f"  high-conf clips: {hi['n']}  -> good {hi['good']}, bad {hi['bad']}")
    L.append("  (low-conf clips marked 'junk' = the follower was RIGHT to be unsure;")
    L.append("   low-conf clips marked 'wrong' = genuine follower failures to fix.)")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description="Real-audio follower eval -- Track B validation report (#133)")
    ap.add_argument("--bundles-root", type=Path, default=Path("data/evals/realaudio_bundles"))
    ap.add_argument("--subset", type=Path, default=SUBSET_JSON)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    validations = load_validations(args.bundles_root)
    summary = summarize(validations, _confidence_map(args.subset))
    print(_format(summary))
    if args.out:
        args.out.write_text(json.dumps(summary, indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
