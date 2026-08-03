# model/src/follower_eval/validate_report.py
"""Aggregate the human validations (issue #133, Track B) into the answer the
proxy track can't give on its own: on the real amateur clips, how often does the
follower actually track, and are the low-confidence clips genuinely-hard
(right to abstain) or follower failures?

Reads ``<piece>/<vid>.validate.json`` (from ``validate_tool``) and reports the
verdict distribution, the fraction of playback flagged wrong, and -- the point --
a cross-tab of verdict against the confidence computed on the resolved score.
The report keeps tracked, recovered, wrong, and junk outcomes separate because
collapsing them into one success number hides calibration failures.

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

VERDICTS = ("tracked", "recovered", "wrong", "junk")
CONFIDENCE_THRESHOLD = 0.5


class ValidateReportError(RuntimeError):
    """A validation cannot support the report's claimed interpretation."""


def load_validations(bundles_root: Path) -> list[dict]:
    """All ``*.validate.json`` under known-piece dirs, loudest-fail on malformed."""
    out: list[dict] = []
    for piece_dir in sorted(p for p in bundles_root.iterdir() if p.is_dir()):
        if piece_dir.name not in SCORE_FILENAME_BY_PIECE:
            continue
        for v in sorted(piece_dir.glob("*.validate.json")):
            out.append(json.loads(v.read_text()))
    return out


def summarize(validations: list[dict]) -> dict:
    """Verdict counts, fraction-wrong stats, and confidence/source strata."""
    verdicts = {verdict: 0 for verdict in VERDICTS}
    fw: list[float] = []
    confidence_outcomes = {
        bucket: {verdict: 0 for verdict in VERDICTS}
        for bucket in ("low", "high", "unscored")
    }
    score_source_outcomes = {
        source: {verdict: 0 for verdict in VERDICTS}
        for source in ("verified", "unverified_label")
    }
    for v in validations:
        verdict = v["verdict"]
        if verdict not in verdicts:
            raise ValidateReportError(
                f"unknown verdict {verdict!r} in {v['piece']}/{v['video_id']}"
            )
        if "follower_confidence" not in v:
            raise ValidateReportError(
                f"{v['piece']}/{v['video_id']} predates resolved-score confidence; "
                "reload the current validator and re-save it"
            )
        score_source = v.get("score_source")
        if score_source not in {"piece_id", "label"}:
            raise ValidateReportError(
                f"unknown score source {score_source!r} in {v['piece']}/{v['video_id']}"
            )

        verdicts[verdict] += 1
        if v.get("fraction_wrong") is not None:
            fw.append(float(v["fraction_wrong"]))

        conf = v["follower_confidence"]
        confidence_bucket = (
            "unscored"
            if conf is None
            else "low"
            if conf < CONFIDENCE_THRESHOLD
            else "high"
        )
        confidence_outcomes[confidence_bucket][verdict] += 1

        source_bucket = "verified" if score_source == "piece_id" else "unverified_label"
        score_source_outcomes[source_bucket][verdict] += 1

    n = len(validations)
    return {
        "n_validated": n,
        "verdicts": verdicts,
        "median_fraction_wrong": round(statistics.median(fw), 4) if fw else None,
        "p90_fraction_wrong": round(
            sorted(fw)[min(len(fw) - 1, int(0.9 * (len(fw) - 1)))], 4
        )
        if fw
        else None,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "confidence_outcomes": confidence_outcomes,
        "score_source_outcomes": score_source_outcomes,
    }


def _format(summary: dict) -> str:
    L = [
        "=" * 76,
        "REAL-AUDIO FOLLOWER EVAL (#133) -- TRACK B -- "
        "human validation of amateur clips",
        "=" * 76,
        f"validated clips: {summary['n_validated']}",
    ]
    if not summary["n_validated"]:
        L.append("")
        L.append("No *.validate.json yet. Label some clips:")
        L.append("  ...python -m follower_eval.validate_tool --precompute   # once")
        L.append(
            "  ...python -m follower_eval.validate_tool --serve        # then label"
        )
        return "\n".join(L)

    L.append("")
    L.append("verdicts:")
    for verdict in VERDICTS:
        L.append(f"  {verdict:<10} {summary['verdicts'][verdict]}")

    L.append("")
    L.append(
        "fraction of playback flagged wrong: median "
        f"{summary['median_fraction_wrong']} "
        f"p90 {summary['p90_fraction_wrong']}"
    )

    L.append("")
    L.append(
        "CONFIDENCE x OUTCOME (resolved-score confidence; threshold "
        f"{summary['confidence_threshold']}):"
    )
    for bucket in ("low", "high", "unscored"):
        counts = summary["confidence_outcomes"][bucket]
        L.append(
            f"  {bucket:<8} "
            + " ".join(f"{verdict}={counts[verdict]}" for verdict in VERDICTS)
        )
    L.append("  high-confidence junk and wrong clips are calibration failures;")
    L.append(
        "  recovered clips remain separate because relocking is only partial evidence."
    )

    L.append("")
    L.append("SCORE SOURCE x OUTCOME:")
    for source in ("verified", "unverified_label"):
        counts = summary["score_source_outcomes"][source]
        L.append(
            f"  {source:<16} "
            + " ".join(f"{verdict}={counts[verdict]}" for verdict in VERDICTS)
        )
    L.append("  unverified_label rows cannot support a verified accuracy claim.")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Real-audio follower eval -- Track B validation report (#133)"
    )
    ap.add_argument(
        "--bundles-root", type=Path, default=Path("data/evals/realaudio_bundles")
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    validations = load_validations(args.bundles_root)
    summary = summarize(validations)
    print(_format(summary))
    if args.out:
        args.out.write_text(json.dumps(summary, indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
