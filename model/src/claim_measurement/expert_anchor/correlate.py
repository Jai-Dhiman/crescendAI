"""M2 expert-anchor correlation (#66 / GATE 2).

For each PercePiano segment, compute the MIDI-native deterministic measurements
(timing CV%, dynamics velocity dispersion, pedaling on-fraction) and Spearman-correlate
them against the PercePiano perceptual labels per dimension. GATE 2: a deterministic
dimension is admissible only if its measurement tracks expert perception.

No LLM anywhere in this path -- PercePiano labels are human perceptual ratings.

Usage:
    uv run python -m claim_measurement.expert_anchor.correlate \
        --midi-dir <percepiano midi> --labels <composite_labels.json> [--report out.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import partitura as pt
from scipy.stats import rankdata, spearmanr

from claim_measurement.expert_anchor.midi_measures import (
    dynamics_mean_velocity,
    pedaling_on_fraction,
    timing_ioi_cv,
)

# Validated GATE 2 proxies (measurement key -> perceptual dimension).
MEASURE_TO_DIM = {"ioi_cv": "timing", "mean_velocity": "dynamics", "pedal_frac": "pedaling"}
ALL_DIMS = ["timing", "dynamics", "pedaling", "articulation", "phrasing", "interpretation"]


def _partial_spearman(x: list[float], y: list[float], z: list[float]) -> float:
    """Spearman of x,y controlling for z: residualize rank(x), rank(y) on rank(z)."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    design = np.vstack([rz, np.ones_like(rz)]).T
    bx, *_ = np.linalg.lstsq(design, rx, rcond=None)
    by, *_ = np.linalg.lstsq(design, ry, rcond=None)
    ex, ey = rx - design @ bx, ry - design @ by
    return float(np.corrcoef(ex, ey)[0, 1])


def _cc64_events(perf) -> list[tuple[float, int]]:
    events: list[tuple[float, int]] = []
    for pp in perf.performedparts:
        for c in (pp.controls or []):
            if int(c.get("number", -1)) == 64:
                events.append((float(c["time"]), int(c["value"])))
    return events


def measure_segment(midi_path: Path) -> dict:
    """MIDI-native measurements for one segment. Per-measure failures -> None
    (e.g. too few onsets) so a segment never crashes the batch."""
    perf = pt.load_performance_midi(str(midi_path))
    na = perf.note_array()
    onsets = np.asarray(na["onset_sec"], dtype=np.float64)
    vels = np.asarray(na["velocity"], dtype=np.float64)
    total_dur = float(onsets.max() + na["duration_sec"].max()) if len(na) else 0.0

    out: dict = {}
    try:
        out["ioi_cv"] = timing_ioi_cv(onsets)
    except ValueError:
        out["ioi_cv"] = None
    try:
        out["mean_velocity"] = dynamics_mean_velocity(vels)
    except ValueError:
        out["mean_velocity"] = None
    try:
        out["pedal_frac"] = pedaling_on_fraction(_cc64_events(perf), total_dur) if total_dur > 0 else None
    except ValueError:
        out["pedal_frac"] = None
    return out


def run_correlation(midi_dir: Path, labels_path: Path) -> dict:
    """Per-dimension proxy<->perception correlation. Reports raw Spearman and the
    partial Spearman controlling for the mean of the OTHER 5 perceptual dims (halo
    control) -- the GATE 2 dimension-specific validity number."""
    labels = json.loads(labels_path.read_text())
    rows: list[tuple[dict, dict]] = []  # (measures, label) for fully-measured segments
    n_segments = 0
    for midi_path in sorted(midi_dir.glob("*.mid")):
        label = labels.get(midi_path.stem)
        if label is None:
            continue
        n_segments += 1
        rows.append((measure_segment(midi_path), label))

    results: dict = {"n_segments": n_segments, "dimensions": {}}
    for m, dim in MEASURE_TO_DIM.items():
        others = [d for d in ALL_DIMS if d != dim]
        x, y, ctrl = [], [], []
        for measures, label in rows:
            mv = measures.get(m)
            if mv is None or label.get(dim) is None:
                continue
            x.append(float(mv))
            y.append(float(label[dim]))
            ctrl.append(float(np.mean([label[d] for d in others])))
        if len(x) >= 10:
            rho, p = spearmanr(x, y)
            results["dimensions"][dim] = {
                "measure": m, "spearman_rho": float(rho), "p_value": float(p),
                "partial_rho_vs_halo": _partial_spearman(x, y, ctrl), "n": len(x),
            }
        else:
            results["dimensions"][dim] = {
                "measure": m, "spearman_rho": None, "p_value": None,
                "partial_rho_vs_halo": None, "n": len(x),
            }
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="claim_measurement.expert_anchor.correlate")
    parser.add_argument("--midi-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)

    results = run_correlation(args.midi_dir, args.labels)
    print(f"=== GATE 2: deterministic measurement vs PercePiano perception "
          f"(n_segments={results['n_segments']}) ===")
    print(f"{'dimension':12s} {'measure':14s} {'raw_rho':>8s} {'partial':>8s} {'p_value':>10s} {'n':>5s}")
    for dim, r in results["dimensions"].items():
        rho = "  -  " if r["spearman_rho"] is None else f"{r['spearman_rho']:+.3f}"
        part = "  -  " if r["partial_rho_vs_halo"] is None else f"{r['partial_rho_vs_halo']:+.3f}"
        p = "  -  " if r["p_value"] is None else f"{r['p_value']:.2e}"
        print(f"{dim:12s} {r['measure']:14s} {rho:>8s} {part:>8s} {p:>10s} {r['n']:>5d}")
    if args.report:
        args.report.write_text(json.dumps(results, indent=2))
        print(f"\nreport -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
