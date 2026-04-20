#!/usr/bin/env python3
"""T5 single-ordinal labeling self-consistency instrumentation.

Adds lightweight drift tracking on top of the existing 1-5 single-ordinal
labeling workflow. Runs alongside your normal labeling so silent rater drift
("the 3 you give in April != the 3 you give in August") becomes visible.

Three commands:

    record_label <recording_id> <ordinal 1-5>
        Append a first-labeling event to label_log.jsonl.

    suggest_relabel
        Every 50 new labels, prompt for a relabel on 5 random recordings
        labeled more than 100 events ago. Writes the paired relabel to
        calibration_log.jsonl.

    kappa_report
        Compute quadratic-weighted Cohen's kappa on all paired labels.
        Also prints a rolling-100-pair kappa so drift is localized.
        Warns if rolling kappa < 0.6.

All data is stored under data/labels/t5/ alongside the existing composite
labels. JSONL is append-only and line-oriented so it survives crashes.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_ROOT = HERE.parent
T5_DIR = MODEL_ROOT / "data" / "labels" / "t5"
LABEL_LOG = T5_DIR / "label_log.jsonl"
CALIBRATION_LOG = T5_DIR / "calibration_log.jsonl"

RELABEL_INTERVAL = 50
RELABEL_SAMPLE_SIZE = 5
RELABEL_LOOKBACK = 100
KAPPA_WARN_THRESHOLD = 0.6
ROLLING_WINDOW = 100


def _ensure_dirs() -> None:
    T5_DIR.mkdir(parents=True, exist_ok=True)
    for path in (LABEL_LOG, CALIBRATION_LOG):
        if not path.exists():
            path.touch()


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _append_jsonl(path: Path, entry: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _validate_ordinal(value: int) -> int:
    if value < 1 or value > 5:
        raise ValueError(f"ordinal must be in 1..5, got {value}")
    return value


def cmd_record_label(args: argparse.Namespace) -> int:
    _ensure_dirs()
    ordinal = _validate_ordinal(int(args.ordinal))
    entry = {
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
        "recording_id": args.recording_id,
        "ordinal": ordinal,
    }
    _append_jsonl(LABEL_LOG, entry)
    total = len(_read_jsonl(LABEL_LOG))
    print(f"Recorded: {args.recording_id} -> {ordinal}  (total: {total})")
    if total > 0 and total % RELABEL_INTERVAL == 0:
        print(
            f"Every {RELABEL_INTERVAL} labels reached. "
            f"Run 'suggest_relabel' to check drift."
        )
    return 0


def cmd_suggest_relabel(args: argparse.Namespace) -> int:
    _ensure_dirs()
    labels = _read_jsonl(LABEL_LOG)
    if len(labels) < RELABEL_LOOKBACK:
        print(
            f"Need at least {RELABEL_LOOKBACK} labels before relabeling is "
            f"meaningful. Current: {len(labels)}."
        )
        return 0

    eligible = labels[:-RELABEL_LOOKBACK]
    if len(eligible) < RELABEL_SAMPLE_SIZE:
        print(
            f"Only {len(eligible)} eligible labels older than "
            f"{RELABEL_LOOKBACK} events ago. Skip for now."
        )
        return 0

    rng = random.Random(args.seed) if args.seed is not None else random
    sample = rng.sample(eligible, RELABEL_SAMPLE_SIZE)

    print("Relabel session — score each recording 1-5 without looking at the original.")
    print("(Ctrl-C to abort; partial progress is saved line-by-line.)\n")

    for entry in sample:
        rid = entry["recording_id"]
        original = entry["ordinal"]
        original_ts = entry["ts"]
        while True:
            try:
                raw = input(f"{rid}: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nAborted.")
                return 1
            if not raw:
                continue
            try:
                relabel = _validate_ordinal(int(raw))
                break
            except ValueError as exc:
                print(f"  invalid: {exc}. Try again (1-5).")

        now = dt.datetime.now(dt.timezone.utc)
        try:
            parsed = dt.datetime.fromisoformat(original_ts)
            days_between = (now - parsed).total_seconds() / 86400.0
        except (TypeError, ValueError):
            days_between = None

        _append_jsonl(
            CALIBRATION_LOG,
            {
                "ts": now.isoformat(),
                "recording_id": rid,
                "ordinal_original": original,
                "ordinal_relabel": relabel,
                "days_between": days_between,
            },
        )
        delta = relabel - original
        marker = "MATCH" if delta == 0 else f"DELTA {delta:+d}"
        print(f"  original={original}, relabel={relabel}  [{marker}]")

    print(f"\nRecorded {RELABEL_SAMPLE_SIZE} relabels.")
    return 0


def _quadratic_weighted_kappa(a: list[int], b: list[int], k: int = 5) -> float | None:
    if len(a) != len(b) or len(a) < 2:
        return None
    try:
        import numpy as np
    except ImportError:
        print("numpy required for kappa_report; run inside the project venv.")
        return None

    a_arr = np.asarray(a)
    b_arr = np.asarray(b)

    hist_a = np.bincount(a_arr - 1, minlength=k).astype(float)
    hist_b = np.bincount(b_arr - 1, minlength=k).astype(float)

    weights = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            weights[i, j] = ((i - j) ** 2) / ((k - 1) ** 2)

    observed = np.zeros((k, k))
    for x, y in zip(a_arr, b_arr):
        observed[x - 1, y - 1] += 1
    observed /= observed.sum() if observed.sum() else 1.0

    expected = np.outer(hist_a, hist_b)
    expected /= expected.sum() if expected.sum() else 1.0

    num = (weights * observed).sum()
    den = (weights * expected).sum()
    if den == 0:
        return None
    return float(1.0 - num / den)


def cmd_kappa_report(args: argparse.Namespace) -> int:
    _ensure_dirs()
    pairs = _read_jsonl(CALIBRATION_LOG)
    if not pairs:
        print("insufficient data: no calibration pairs yet.")
        return 0

    originals = [p["ordinal_original"] for p in pairs]
    relabels = [p["ordinal_relabel"] for p in pairs]

    overall = _quadratic_weighted_kappa(originals, relabels)
    if overall is None:
        print(f"insufficient data: need >= 2 pairs, have {len(pairs)}.")
        return 0

    print(f"Total paired relabels:     {len(pairs)}")
    print(f"Overall quadratic kappa:   {overall:.3f}")

    if len(pairs) >= ROLLING_WINDOW:
        rolling = _quadratic_weighted_kappa(
            originals[-ROLLING_WINDOW:], relabels[-ROLLING_WINDOW:]
        )
        if rolling is not None:
            print(f"Rolling {ROLLING_WINDOW}-pair kappa:     {rolling:.3f}")
            if rolling < KAPPA_WARN_THRESHOLD:
                print(
                    f"WARNING: rolling kappa below {KAPPA_WARN_THRESHOLD}. "
                    f"Recent labels may be drifting from your earlier calibration. "
                    f"Consider a calibration pass on 10 anchor recordings with known scores."
                )

    mean_delta = sum(r - o for o, r in zip(originals, relabels)) / len(pairs)
    print(f"Mean signed drift:         {mean_delta:+.2f}  (positive = you now score higher)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="t5_label_consistency",
        description="Track drift in T5 single-ordinal labeling.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_rec = sub.add_parser("record_label", help="Append a first-labeling event.")
    p_rec.add_argument("recording_id")
    p_rec.add_argument("ordinal", help="Integer 1-5")
    p_rec.set_defaults(func=cmd_record_label)

    p_sug = sub.add_parser(
        "suggest_relabel", help="Interactive relabel on random older recordings."
    )
    p_sug.add_argument("--seed", type=int, default=None)
    p_sug.set_defaults(func=cmd_suggest_relabel)

    p_kap = sub.add_parser(
        "kappa_report", help="Print drift stats from calibration_log.jsonl."
    )
    p_kap.set_defaults(func=cmd_kappa_report)

    args = parser.parse_args()
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
