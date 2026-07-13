# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy>=1.24.0",
#     "scipy>=1.10.0",
#     "pretty_midi>=0.2.10",
#     "scikit-learn>=1.3.0",
#     "lightgbm>=4.3.0",
# ]
# ///
"""Phase 3c -- push tau-c past 0.723: candidate-feature superset + greedy selection.

Phase 3b hit tau-c 0.7229 (composer-disjoint 5-fold LGBMRegressor) on 7 features,
short of the ~0.77 audio-SOTA band. Hypothesis: we are FEATURE-limited -- we have only
4 real signal features (pitch_lz/entropy/range/note_count) + weak velocity/timing, while
RubricNet's ~0.77 uses ~19 including rhythm-complexity, interval/hand-span, pitch-class.

This harness extracts a ~23-feature SUPERSET (every candidate that is onset/pitch/velocity
based -- AMT-reliable, offsets still excluded) in ONE MIDI-parse pass, caches it, then runs
GREEDY FORWARD SELECTION on composer-disjoint 5-fold CV (add the feature that most raises
mean tau-c, stop when the gain drops below eps). Greedy-on-CV is how we find each feature's
HONEST marginal contribution and avoid the overfit that flattered raw importances in 3b.

Outputs: per-feature tau-c, the kitchen-sink (all-23) tau-c, and the greedy path. If the
greedy optimum ~= the current 7, features are SATURATED -> the wall is not feature design.

Run (data from + result to the primary checkout):
    uv run --script phase3c_explore.py [--limit N] [--refresh] [--eps 0.002]
"""
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from difficulty_features import pitch_lz_complexity  # noqa: E402  (reuse LZ76 on any int seq)
from psyllabus import load_records, notes_from_midi_bytes  # noqa: E402

PRIMARY = Path("/Users/jdhiman/Documents/crescendai")
PS_DIR = PRIMARY / "model/data/raw/psyllabus"
LABELS = PS_DIR / "new_clean_data.json"
MID_ZIP = PS_DIR / "mid.zip"
OUT = PRIMARY / "model/data/results/mirex_phase3c_explore.json"
FEAT_CACHE = PRIMARY / "model/data/results/mirex_phase3c_features.json"

N_FOLDS = 5
# IOI quantization edges (seconds) -> symbolic rhythm alphabet for entropy/LZ. Onset-only.
IOI_EDGES = np.array([0.04, 0.07, 0.10, 0.14, 0.20, 0.28, 0.40, 0.55, 0.75, 1.0, 1.4, 2.0])


def _entropy(symbols) -> float:
    s = np.asarray(symbols)
    if s.size < 1:
        return float("nan")
    _, counts = np.unique(s, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p)))


def candidate_features(notes) -> dict:
    """Superset of AMT-reliable (onset/pitch/velocity) difficulty candidates. Any feature
    that can't be computed on a degenerate piece is set to nan (LightGBM handles nan)."""
    ordered = sorted(notes, key=lambda n: (n["onset"], n["pitch"]))
    pitches = np.array([n["pitch"] for n in ordered], dtype=np.int64)
    onsets = np.array([n["onset"] for n in ordered], dtype=np.float64)
    vels = np.array([n["velocity"] for n in ordered], dtype=np.float64)
    n = len(ordered)
    span = float(onsets[-1] - onsets[0]) if n >= 2 else 0.0

    f = {}
    # --- existing 7 (recomputed here so the superset is self-contained) ---
    f["pitch_entropy"] = _entropy(pitches)
    f["pitch_lz_complexity"] = float(pitch_lz_complexity(pitches.tolist()))
    f["pitch_range"] = float(pitches.max() - pitches.min())
    f["note_count"] = float(n)
    f["vel_mean"] = float(np.mean(vels))
    f["vel_disp"] = float(np.std(vels)) if n >= 2 else float("nan")

    uniq_onsets = np.unique(onsets)
    iois = np.diff(uniq_onsets)
    iois = iois[iois > 1e-3]
    f["timing_ioi_cv"] = float(np.std(iois) / np.mean(iois)) if iois.size >= 2 and np.mean(iois) > 0 else float("nan")

    # --- rhythm / onset (onsets are AMT-reliable; offsets still excluded) ---
    f["notes_per_sec"] = float(n / span) if span > 0 else float("nan")
    if iois.size >= 2:
        sym = np.digitize(iois, IOI_EDGES)
        f["ioi_entropy"] = _entropy(sym)
        f["ioi_lz"] = float(pitch_lz_complexity(sym.tolist()))
        f["unique_ioi_ratio"] = float(len(np.unique(sym)) / sym.size)
        f["fast_note_fraction"] = float(np.mean(iois < 0.1))
        f["ioi_range_log"] = float(np.log2(iois.max() / iois.min())) if iois.min() > 0 else float("nan")
    else:
        f["ioi_entropy"] = f["ioi_lz"] = f["unique_ioi_ratio"] = f["fast_note_fraction"] = f["ioi_range_log"] = float("nan")

    # --- pitch-class / key complexity ---
    pc = pitches % 12
    f["n_pitch_classes"] = float(len(np.unique(pc)))
    f["pitch_class_entropy"] = _entropy(pc)
    f["pitch_range_p5_95"] = float(np.percentile(pitches, 95) - np.percentile(pitches, 5))

    # --- melodic interval / leaps (consecutive pitch differences in onset order) ---
    if n >= 2:
        intervals = np.abs(np.diff(pitches))
        f["interval_mean_abs"] = float(np.mean(intervals))
        f["interval_max_abs"] = float(np.max(intervals))
        f["interval_entropy"] = _entropy(np.clip(intervals, 0, 24))
        f["large_leap_fraction"] = float(np.mean(intervals > 12))
        f["vel_range"] = float(vels.max() - vels.min())
        # order-1 conditional pitch entropy (melodic predictability): H(next | current)
        f["pitch_bigram_cond_entropy"] = _cond_entropy(pitches)
    else:
        for k in ("interval_mean_abs", "interval_max_abs", "interval_entropy",
                  "large_leap_fraction", "vel_range", "pitch_bigram_cond_entropy"):
            f[k] = float("nan")

    # --- chord span / hand stretch (onset clusters within 30ms) ---
    # codex #2: chord WIDTH != chord density (polyphony). A close 4-note chord vs a tenth
    # differ radically; span percentiles + wide-stretch frequencies capture hand demand.
    cl_times, spans = _chord_clusters(onsets, pitches, tol=0.03)
    if spans:
        sp = np.asarray(spans, float)
        f["max_chord_span"] = float(sp.max())
        f["mean_chord_span"] = float(sp.mean())
        f["chord_span_p90"] = float(np.percentile(sp, 90))
        f["frac_chord_span_gt7"] = float(np.mean(sp > 7))
        f["frac_chord_span_gt12"] = float(np.mean(sp > 12))
        f["frac_chord_span_gt16"] = float(np.mean(sp > 16))
    else:
        for k in ("max_chord_span", "mean_chord_span", "chord_span_p90",
                  "frac_chord_span_gt7", "frac_chord_span_gt12", "frac_chord_span_gt16"):
            f[k] = float("nan")

    # --- density separated from length (codex #1: note_count conflates the two) ---
    f["onset_clusters_per_sec"] = float(len(cl_times) / span) if span > 0 else float("nan")
    f["median_ioi"] = float(np.median(iois)) if iois.size >= 1 else float("nan")
    if n >= 2:
        f["frac_interval_gt7"] = float(np.mean(np.abs(np.diff(pitches)) > 7))

    # --- local / sliding-window peaks (codex #6: the grade often lives in the hardest
    # passage, which whole-piece means wash out). 5s windows over onset time; aggregate the
    # PEAK local density/complexity, not just the average. ---
    win = _window_peaks(onsets, pitches, cl_times, spans, win_s=5.0)
    f.update(win)
    return f


def _cond_entropy(pitches: np.ndarray) -> float:
    """H(next pitch | current pitch), bits. Lower = more predictable melody."""
    from collections import defaultdict
    trans = defaultdict(lambda: defaultdict(int))
    for a, b in zip(pitches[:-1], pitches[1:]):
        trans[int(a)][int(b)] += 1
    total = len(pitches) - 1
    if total <= 0:
        return float("nan")
    h = 0.0
    for a, nexts in trans.items():
        na = sum(nexts.values())
        pa = na / total
        hb = 0.0
        for _, cnt in nexts.items():
            p = cnt / na
            hb -= p * np.log2(p)
        h += pa * hb
    return float(h)


def _chord_clusters(onsets: np.ndarray, pitches: np.ndarray, tol: float):
    """Onset clusters (notes within `tol` s). Returns (cluster_onset_times, cluster_spans),
    where span = max-min pitch in the cluster (monophonic cluster -> 0)."""
    order = np.argsort(onsets)
    o, p = onsets[order], pitches[order]
    times, spans, anchor, lo, hi = [], [], o[0], p[0], p[0]
    for i in range(1, len(o)):
        if o[i] - anchor <= tol:
            lo, hi = min(lo, p[i]), max(hi, p[i])
        else:
            times.append(anchor)
            spans.append(hi - lo)
            anchor, lo, hi = o[i], p[i], p[i]
    times.append(anchor)
    spans.append(hi - lo)
    return times, spans


def _window_peaks(onsets, pitches, cl_times, spans, win_s: float) -> dict:
    """Peak LOCAL difficulty over fixed-time windows. The hardest passage often sets the
    grade, so aggregate p90/max of per-window density and complexity, not just the mean."""
    keys = ("win_density_p90", "win_density_max", "win_pitch_entropy_p90",
            "win_interval_mean_p90", "win_chord_span_max_p90", "win_hard_fraction")
    span_total = float(onsets[-1] - onsets[0]) if len(onsets) >= 2 else 0.0
    if span_total < win_s:
        return {k: float("nan") for k in keys}
    edges = np.arange(onsets[0], onsets[-1] + win_s, win_s)
    bins = np.digitize(onsets, edges)
    cl_bins = np.digitize(np.asarray(cl_times), edges) if cl_times else np.array([], dtype=int)
    cl_spans = np.asarray(spans, float)
    dens, ents, ivs, cspans = [], [], [], []
    for b in np.unique(bins):
        wp = pitches[bins == b]
        if wp.size < 2:
            continue
        dens.append(wp.size / win_s)
        ents.append(_entropy(wp))
        ivs.append(float(np.mean(np.abs(np.diff(wp)))))
        wcs = cl_spans[cl_bins == b] if cl_bins.size else np.array([0.0])
        cspans.append(float(wcs.max()) if wcs.size else 0.0)
    if not dens:
        return {k: float("nan") for k in keys}
    dens = np.asarray(dens)
    # "hard window" = local density in the top-decile of this piece's own windows, as a
    # fraction of windows -> proxy for how much of the piece is sustained-hard.
    thresh = np.percentile(dens, 75)
    return {
        "win_density_p90": float(np.percentile(dens, 90)),
        "win_density_max": float(dens.max()),
        "win_pitch_entropy_p90": float(np.percentile(ents, 90)),
        "win_interval_mean_p90": float(np.percentile(ivs, 90)),
        "win_chord_span_max_p90": float(np.percentile(cspans, 90)),
        "win_hard_fraction": float(np.mean(dens >= thresh)),
    }


def tau_c(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return None
    t = stats.kendalltau(x, y, variant="c").statistic
    return None if np.isnan(t) else float(t)


def extract_rows(records):
    rows, parse_fail = [], 0
    with zipfile.ZipFile(MID_ZIP) as zf:
        for i, rec in enumerate(records):
            try:
                notes = notes_from_midi_bytes(zf.read(rec.midi_name))
                if len(notes) < 5:
                    raise ValueError(f"too few notes ({len(notes)})")
                rows.append({"grade": rec.grade, "composer": rec.composer, **candidate_features(notes)})
            except Exception as exc:  # noqa: BLE001
                parse_fail += 1
                if parse_fail <= 5:
                    print(f"  parse fail [{rec.key[:40]}]: {exc!r}", flush=True)
            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{len(records)} ({parse_fail} fails)", flush=True)
    return rows, parse_fail


def load_or_extract(records, use_cache):
    if use_cache and FEAT_CACHE.exists():
        cached = json.loads(FEAT_CACHE.read_text())
        if cached.get("n_records") == len(records):
            print(f"loaded {len(cached['rows'])} feature rows from cache", flush=True)
            return cached["rows"], cached.get("n_parse_fail", 0)
    rows, pf = extract_rows(records)
    FEAT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    FEAT_CACHE.write_text(json.dumps({"n_records": len(records), "n_parse_fail": pf, "rows": rows}))
    return rows, pf


def cv_tau(X, y, groups, cols, seed=2026, n_estimators=300):
    """Mean composer-disjoint 5-fold tau-c of an LGBMRegressor on the given columns."""
    import lightgbm as lgb
    Xs = X[:, cols]
    gkf = GroupKFold(n_splits=N_FOLDS)
    taus = []
    for tr, te in gkf.split(Xs, y, groups=groups):
        reg = lgb.LGBMRegressor(objective="regression", n_estimators=n_estimators,
                                learning_rate=0.03, num_leaves=31, min_child_samples=40,
                                subsample=0.8, subsample_freq=1, colsample_bytree=0.9,
                                reg_lambda=1.0, random_state=seed, n_jobs=-1, verbosity=-1)
        reg.fit(Xs[tr], y[tr])
        t = tau_c(reg.predict(Xs[te]), y[te])
        if t is not None:
            taus.append(t)
    return float(np.mean(taus)), float(np.std(taus))


def main():
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    eps = float(sys.argv[sys.argv.index("--eps") + 1]) if "--eps" in sys.argv else 0.002
    refresh = "--refresh" in sys.argv

    records, report = load_records(LABELS, MID_ZIP)
    if limit:
        records = records[:: max(1, len(records) // limit)][:limit]
    print(f"loaded {report}\nprocessing {len(records)} records\n", flush=True)

    rows, parse_fail = load_or_extract(records, use_cache=not refresh and limit is None)
    print(f"\nusable pieces: {len(rows)}  parse failures: {parse_fail}", flush=True)

    all_feats = [k for k in rows[0] if k not in ("grade", "composer")]
    X = np.array([[r[f] for f in all_feats] for r in rows], dtype=np.float64)
    y = np.array([r["grade"] for r in rows], dtype=int)
    groups = np.array([r["composer"] for r in rows])
    idx = {f: j for j, f in enumerate(all_feats)}

    # per-feature tau-c (signal each carries alone)
    per_feat = {}
    for f in all_feats:
        col = X[:, idx[f]]
        m = ~np.isnan(col)
        per_feat[f] = tau_c(col[m], y[m])

    current7 = ["pitch_lz_complexity", "pitch_entropy", "pitch_range", "note_count",
                "vel_mean", "vel_disp", "timing_ioi_cv"]
    base_mean, base_std = cv_tau(X, y, groups, [idx[f] for f in current7])
    kitchen_mean, kitchen_std = cv_tau(X, y, groups, list(range(len(all_feats))))

    print(f"\n  current-7 tau-c   = {base_mean:.4f} +/- {base_std:.4f}", flush=True)
    print(f"  kitchen-sink({len(all_feats)}) = {kitchen_mean:.4f} +/- {kitchen_std:.4f}", flush=True)

    # greedy forward selection on CV tau-c
    print("\n  greedy forward selection (composer-disjoint 5-fold tau-c):", flush=True)
    selected, remaining = [], list(all_feats)
    path, best_so_far = [], -1.0
    while remaining:
        scored = []
        for f in remaining:
            cols = [idx[c] for c in selected + [f]]
            m, _ = cv_tau(X, y, groups, cols)
            scored.append((m, f))
        scored.sort(reverse=True)
        best_gain = scored[0][0] - best_so_far
        if best_gain < eps and selected:
            print(f"    stop: best add {scored[0][1]} gains only {best_gain:+.4f} (<{eps})", flush=True)
            break
        best_mean, best_feat = scored[0]
        selected.append(best_feat)
        remaining.remove(best_feat)
        path.append({"step": len(selected), "added": best_feat, "tau_c": best_mean,
                     "gain": best_mean - best_so_far})
        print(f"    +{best_feat:24s} -> tau-c {best_mean:.4f}  (gain {best_mean - best_so_far:+.4f})", flush=True)
        best_so_far = best_mean

    summary = {
        "n_usable": len(rows), "n_parse_fail": parse_fail, "n_folds": N_FOLDS,
        "all_features": all_feats, "per_feature_tau_c": per_feat,
        "current7_tau_c": {"mean": base_mean, "std": base_std},
        "kitchen_sink_tau_c": {"mean": kitchen_mean, "std": kitchen_std},
        "greedy_path": path, "greedy_selected": selected,
        "greedy_best_tau_c": best_so_far,
        "improvement_over_current7": best_so_far - base_mean,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"summary": summary}, indent=2))

    print("\n=== PHASE 3c: FEATURE-CEILING PROBE ===")
    print("  per-feature tau-c (sorted):")
    for f, t in sorted(per_feat.items(), key=lambda kv: -(kv[1] or 0)):
        print(f"    {f:26s} {t:.4f}" if t is not None else f"    {f:26s}   None")
    print(f"\n  current-7      = {base_mean:.4f}")
    print(f"  kitchen-sink   = {kitchen_mean:.4f}  ({kitchen_mean - base_mean:+.4f} vs current-7)")
    print(f"  greedy-optimal = {best_so_far:.4f}  ({best_so_far - base_mean:+.4f} vs current-7)")
    print(f"  greedy set ({len(selected)}): {selected}")
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
