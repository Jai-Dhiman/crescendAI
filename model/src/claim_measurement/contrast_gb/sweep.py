# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     "numpy>=1.24.0",
#     "scipy>=1.10.0",
#     "partitura>=1.4.0",
# ]
# ///
"""Contrast-G-B feasibility sweep (#101 front-6): is there a dynamic-CONTRAST/shaping
statistic that tracks a PercePiano perceptual dimension, where global-spread statistics
failed?

Motivation: G-D (front-5) found the generator's dynamics feedback is ~90% CONTRAST/
shaping ("ebb and swell", "dynamic arc") and ~0% whole-piece absolute level -- so the
G-B-validated mean-velocity LEVEL statistic has no claims to score. The GATE-3 statistic
sweep already showed GLOBAL-SPREAD contrast statistics (std, IQR, range, crest) fail to
predict perceived "dynamics" (= PercePiano `dynamic_range`): std +0.04, range -0.01. But
"ebb and swell" is a TEMPORAL-ENVELOPE property (a low-frequency modulation over the
phrase), which a global std is blind to. This sweep tests whether a TEMPORAL-SHAPING
statistic clears the ~0.5 inter-rater ceiling against any of the 19 granular PercePiano
dims (esp. dynamic_range, drama, mood_energy, sophistication).

Method: per PercePiano segment (ground-truth MIDI, partitura), compute candidate stats
from the velocity-over-time signal; Spearman + halo-controlled partial Spearman (control
= mean of the OTHER 18 dims) vs each granular dim. PASS candidate = partial >= ~0.45,
p<1e-6. No LLM (PercePiano labels are human ratings). Built-in sanity: mean_velocity vs
dynamic_range partial should reproduce the GATE-2 ~0.56.

Run:
    uv run --script sweep.py \
        --midi-dir /ABS/model/data/midi/percepiano \
        --labels   /ABS/model/data/labels/percepiano/labels.json \
        --report   /ABS/model/data/results/contrast_gb_sweep.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import partitura as pt
from scipy.stats import rankdata, spearmanr

# PercePiano 19-dim order (model/src/audio_experiments/constants.py); raw label vectors
# are length 20 (19 dims + 1 trailing field we ignore).
PERCEPIANO_DIMENSIONS = [
    "timing", "articulation_length", "articulation_touch",
    "pedal_amount", "pedal_clarity",
    "timbre_variety", "timbre_depth", "timbre_brightness", "timbre_loudness",
    "dynamic_range", "tempo", "space", "balance", "drama",
    "mood_valence", "mood_energy", "mood_imagination",
    "sophistication", "interpretation",
]
# Dims a dynamic-shaping statistic might plausibly track (reported first).
SHAPING_DIMS = ["dynamic_range", "drama", "mood_energy", "sophistication",
                "timbre_loudness", "balance"]

N_BINS = 6  # time bins for the dynamic-envelope statistics (segments are 4-8 bars)


def _partial_spearman(x, y, z) -> float:
    """Spearman of x,y controlling for z (residualize ranks on rank(z))."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    design = np.vstack([rz, np.ones_like(rz)]).T
    bx, *_ = np.linalg.lstsq(design, rx, rcond=None)
    by, *_ = np.linalg.lstsq(design, ry, rcond=None)
    ex, ey = rx - design @ bx, ry - design @ by
    return float(np.corrcoef(ex, ey)[0, 1])


def _time_binned_means(t: np.ndarray, v: np.ndarray, k: int) -> np.ndarray | None:
    """Mean velocity in each of k equal-time bins over [t.min, t.max].

    The discrete dynamic envelope. Empty bins are linearly interpolated from neighbors
    (a gap in onsets is not a dynamic event). Returns None if <2 non-empty bins.
    """
    if t.size < k:
        return None
    t0, t1 = float(t.min()), float(t.max())
    if t1 <= t0:
        return None
    edges = np.linspace(t0, t1, k + 1)
    idx = np.clip(np.digitize(t, edges[1:-1]), 0, k - 1)
    means = np.full(k, np.nan)
    for b in range(k):
        sel = v[idx == b]
        if sel.size:
            means[b] = float(sel.mean())
    good = ~np.isnan(means)
    if good.sum() < 2:
        return None
    means[~good] = np.interp(np.flatnonzero(~good), np.flatnonzero(good), means[good])
    return means


def segment_stats(midi_path: Path) -> dict:
    """Candidate statistics from one segment's velocity-over-time signal."""
    perf = pt.load_performance_midi(str(midi_path))
    na = perf.note_array()
    if len(na) < 4:
        return {}
    order = np.argsort(na["onset_sec"])
    t = np.asarray(na["onset_sec"], dtype=np.float64)[order]
    v = np.asarray(na["velocity"], dtype=np.float64)[order]

    out: dict = {}
    # --- level baseline (sanity: reproduce GATE-2 ~0.56 vs dynamic_range) ---
    out["mean_velocity"] = float(v.mean())
    # --- global-spread baselines (doc says these FAIL vs dynamics) ---
    out["vel_std"] = float(v.std())
    out["vel_iqr"] = float(np.percentile(v, 75) - np.percentile(v, 25))
    out["vel_range"] = float(v.max() - v.min())
    # note-to-note jaggedness (accent density, not low-freq shape)
    out["local_contrast"] = float(np.mean(np.abs(np.diff(v)))) if v.size > 1 else 0.0

    # --- temporal-SHAPING family (the new bet: low-freq envelope over the phrase) ---
    env = _time_binned_means(t, v, N_BINS)
    if env is not None:
        # swell depth: peak-to-trough of the smoothed dynamic envelope
        out["swell_depth"] = float(env.max() - env.min())
        # envelope modulation: std of the bin-mean envelope (low-freq, not raw std)
        out["envelope_std"] = float(env.std())
        # arc curvature: |a| of a quadratic fit over normalized time (single arch/swell)
        xn = np.linspace(0.0, 1.0, env.size)
        a, _b, _c = np.polyfit(xn, env, 2)
        out["arc_curvature"] = float(abs(a))
        # terracing: count of dynamic direction-changes in the envelope (few = arched)
        d = np.diff(env)
        sign = np.sign(d)
        sign = sign[sign != 0]
        out["terracing"] = float(np.sum(sign[1:] != sign[:-1])) if sign.size > 1 else 0.0
        # normalized swell (gain-robust): peak-to-trough relative to the mean level
        out["swell_depth_norm"] = float((env.max() - env.min()) / env.mean()) if env.mean() > 0 else 0.0
    return out


STAT_KEYS = ["mean_velocity", "vel_std", "vel_iqr", "vel_range", "local_contrast",
             "swell_depth", "envelope_std", "arc_curvature", "terracing", "swell_depth_norm"]


def run(midi_dir: Path, labels_path: Path) -> dict:
    raw = json.loads(labels_path.read_text())
    rows: list[tuple[dict, list[float]]] = []
    for midi_path in sorted(midi_dir.glob("*.mid")):
        lab = raw.get(midi_path.stem)
        if lab is None:
            continue
        stats = segment_stats(midi_path)
        if stats:
            rows.append((stats, [float(x) for x in lab[:19]]))
    n = len(rows)

    # build per-dimension arrays once
    label_arr = np.array([r[1] for r in rows])  # n x 19
    dim_idx = {d: i for i, d in enumerate(PERCEPIANO_DIMENSIONS)}

    results: dict = {"n_segments": n, "n_bins": N_BINS, "stats": {}}
    for stat in STAT_KEYS:
        xs = np.array([r[0].get(stat, np.nan) for r in rows])
        valid = ~np.isnan(xs)
        per_dim = {}
        for dim in PERCEPIANO_DIMENSIONS:
            di = dim_idx[dim]
            y = label_arr[:, di]
            others = [j for j in range(19) if j != di]
            ctrl = label_arr[:, others].mean(axis=1)
            m = valid
            if m.sum() < 10:
                continue
            rho, p = spearmanr(xs[m], y[m])
            part = _partial_spearman(xs[m], y[m], ctrl[m])
            per_dim[dim] = {"raw_rho": float(rho), "partial_rho": part,
                           "p_value": float(p), "n": int(m.sum())}
        # best dim by |partial_rho|
        best = max(per_dim.items(), key=lambda kv: abs(kv[1]["partial_rho"])) if per_dim else None
        results["stats"][stat] = {
            "best_dim": best[0] if best else None,
            "best_partial_rho": best[1]["partial_rho"] if best else None,
            "best_p": best[1]["p_value"] if best else None,
            "shaping_dims": {d: per_dim[d] for d in SHAPING_DIMS if d in per_dim},
            "all_dims": per_dim,
        }
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="contrast_gb.sweep")
    ap.add_argument("--midi-dir", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args(argv)

    res = run(args.midi_dir, args.labels)
    print(f"=== contrast-G-B sweep (n={res['n_segments']}, {res['n_bins']} bins) ===")
    print(f"{'statistic':16s} {'best_dim':16s} {'best_partial':>12s} {'p':>10s}  "
         f"{'|':>1s} shaping-dim partials (dyn_range/drama/energy/soph)")
    for stat, r in res["stats"].items():
        bp = r["best_partial_rho"]
        bp_s = "  -  " if bp is None else f"{bp:+.3f}"
        p_s = "  -  " if r["best_p"] is None else f"{r['best_p']:.1e}"
        sd = r["shaping_dims"]
        sp = "  ".join(
            f"{d[:4]}={sd[d]['partial_rho']:+.3f}" for d in
            ["dynamic_range", "drama", "mood_energy", "sophistication"] if d in sd
        )
        # PASS requires the best dim to be a DYNAMIC-SHAPING dim (not a tempo/density
        # confound) AND clear the ~0.5 ceiling with p<1e-6.
        flag = ("  <== PASS?" if (bp is not None and r["best_dim"] in SHAPING_DIMS
                                  and abs(bp) >= 0.45 and (r["best_p"] or 1) < 1e-6) else "")
        print(f"{stat:16s} {str(r['best_dim']):16s} {bp_s:>12s} {p_s:>10s}  | {sp}{flag}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(res, indent=2))
        print(f"\nreport -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
