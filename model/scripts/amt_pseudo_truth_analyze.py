"""Analyze parangonar-aligned pseudo-truth maps for noise-floor estimation.

Reads matched_pairs.json + score for each pilot clip under
model/data/evals/practice_eval_pseudo/<piece>/<video_id>/ and reports:
  - per-2s-window bar-rate distribution (median, IQR, max spike)
  - coverage gaps (audio ranges where no matched notes exist)
  - bootstrap-style jitter: drop 10%/20%/30% of matched pairs, recompute
    bar positions at fixed anchors, report the spread

Usage:
    uv run python scripts/amt_pseudo_truth_analyze.py --piece chopin_ballade_1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

MODEL_ROOT = Path(__file__).resolve().parents[1]
PSEUDO_ROOT = MODEL_ROOT / "data" / "evals" / "practice_eval_pseudo"


def _load_clip(piece_dir: Path, video_id: str) -> dict | None:
    clip_dir = piece_dir / video_id
    matched = clip_dir / "matched_pairs.json"
    report = clip_dir / "report.json"
    amt = clip_dir / "amt_notes.json"
    if not matched.exists() or not report.exists() or not amt.exists():
        return None
    return {
        "video_id": video_id,
        "report": json.loads(report.read_text()),
        "matches": json.loads(matched.read_text()),
        "amt_notes": json.loads(amt.read_text()),
        "clip_dir": clip_dir,
    }


def _build_anchor_pairs(matches: list[dict], score_id_to_div: dict, amt_id_to_sec: dict) -> np.ndarray:
    pairs: list[tuple[float, float]] = []
    for m in matches:
        if m.get("label") != "match":
            continue
        sid = m.get("score_id")
        pid = m.get("performance_id")
        if sid in score_id_to_div and pid in amt_id_to_sec:
            pairs.append((amt_id_to_sec[pid], score_id_to_div[sid]))
    if not pairs:
        return np.zeros((0, 2))
    arr = np.array(sorted(pairs), dtype=float)
    # Enforce monotonic score_div (running max) so np.interp is well-defined
    arr[:, 1] = np.maximum.accumulate(arr[:, 1])
    return arr


def _div_to_bar(div: float, measure_table: list[dict]) -> float | None:
    for m in measure_table:
        end = m["end_div"]
        if end is None:
            continue
        if m["start_div"] <= div < end:
            frac = (div - m["start_div"]) / max(end - m["start_div"], 1)
            return m["bar_number"] + frac
    return None


def _bar_at_audio_sec(t: float, pairs: np.ndarray, measure_table: list[dict]) -> float | None:
    if len(pairs) < 2:
        return None
    if t < pairs[0, 0] or t > pairs[-1, 0]:
        return None
    div = float(np.interp(t, pairs[:, 0], pairs[:, 1]))
    return _div_to_bar(div, measure_table)


def _measure_table_from_score(score_path: Path) -> list[dict]:
    import partitura as pt
    score = pt.load_score(str(score_path))
    part = score.parts[0]
    return [
        {"bar_number": m.number, "start_div": m.start.t, "end_div": (m.end.t if m.end else None)}
        for m in part.iter_all(pt.score.Measure)
    ]


def analyze_clip(clip: dict, measure_table: list[dict], score_id_to_div: dict) -> dict:
    amt_id_to_sec = {f"amt{i}": float(n["onset"]) for i, n in enumerate(clip["amt_notes"])}
    pairs = _build_anchor_pairs(clip["matches"], score_id_to_div, amt_id_to_sec)
    if len(pairs) < 10:
        return {"video_id": clip["video_id"], "error": "too few matched pairs", "n_pairs": int(len(pairs))}

    t0 = float(pairs[0, 0])
    t1 = float(pairs[-1, 0])

    # Per-30s-window bar-rate
    window = 30.0
    windows = []
    t = t0
    while t + window <= t1:
        b0 = _bar_at_audio_sec(t, pairs, measure_table)
        b1 = _bar_at_audio_sec(t + window, pairs, measure_table)
        if b0 is not None and b1 is not None:
            windows.append({"t_start": t, "bar_start": b0, "bar_end": b1, "bars_per_30s": b1 - b0})
        t += window

    rates = np.array([w["bars_per_30s"] for w in windows], dtype=float)
    rate_stats = {
        "count": int(len(rates)),
        "median": float(np.median(rates)) if len(rates) else None,
        "p10": float(np.percentile(rates, 10)) if len(rates) else None,
        "p90": float(np.percentile(rates, 90)) if len(rates) else None,
        "max": float(rates.max()) if len(rates) else None,
        "min": float(rates.min()) if len(rates) else None,
    }

    # Bootstrap jitter at fixed anchors: 30s, 60s, 90s, 120s
    rng = np.random.default_rng(42)
    anchors = [t for t in [30.0, 60.0, 90.0, 120.0, 150.0, 180.0] if t0 <= t <= t1]
    jitter: dict[str, dict[str, float]] = {}
    for frac_drop in (0.1, 0.2, 0.3):
        n_keep = int(len(pairs) * (1 - frac_drop))
        boot_results: dict[float, list[float]] = {t: [] for t in anchors}
        for _ in range(20):
            idx = rng.choice(len(pairs), size=n_keep, replace=False)
            idx.sort()
            sub = pairs[idx]
            for t in anchors:
                b = _bar_at_audio_sec(t, sub, measure_table)
                if b is not None:
                    boot_results[t].append(b)
        jitter[f"drop_{int(frac_drop*100)}pct"] = {
            f"t={t:.0f}s_bar_std": (float(np.std(v)) if v else None)
            for t, v in boot_results.items()
        }

    return {
        "video_id": clip["video_id"],
        "coverage": {"t_start": t0, "t_end": t1, "n_pairs": int(len(pairs))},
        "bar_rate_per_30s": rate_stats,
        "bootstrap_bar_std_by_anchor": jitter,
        "windows": windows,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--piece", required=True)
    p.add_argument("--score", required=True)
    args = p.parse_args()

    piece_dir = PSEUDO_ROOT / args.piece
    if not piece_dir.exists():
        print(f"no pseudo-truth output for piece: {piece_dir}", file=sys.stderr)
        return 1

    score_path = Path(args.score)
    if not score_path.is_absolute():
        score_path = MODEL_ROOT / args.score

    print(f"loading score {score_path.name}", flush=True)
    measure_table = _measure_table_from_score(score_path)
    import partitura as pt
    score_na = pt.load_score(str(score_path)).note_array()
    score_id_to_div = {n["id"]: float(n["onset_div"]) for n in score_na}
    print(f"  {len(measure_table)} measures, {len(score_id_to_div)} score notes", flush=True)

    out: dict = {"piece": args.piece, "clips": []}
    for clip_dir in sorted(piece_dir.iterdir()):
        if not clip_dir.is_dir():
            continue
        clip = _load_clip(piece_dir, clip_dir.name)
        if clip is None:
            print(f"  skip {clip_dir.name} (missing files)", flush=True)
            continue
        print(f"analyzing {clip_dir.name}", flush=True)
        result = analyze_clip(clip, measure_table, score_id_to_div)
        out["clips"].append(result)

    print()
    print(json.dumps(out, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
