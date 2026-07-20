# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy>=1.24.0","scipy>=1.10.0","pretty_midi>=0.2.10",
#     "scikit-learn>=1.3.0","lightgbm>=4.3.0","partitura>=1.9.0",
# ]
# ///
"""Phase 5e -- cross-dataset test on Mikrokosmos (CIPI proper is Zenodo-restricted).

Phase 5c argued the 0.76 ceiling is a FEATURE wall, not a fixable label wall. The direct test is
an INDEPENDENT, cleaner-labeled corpus: if our 37 features rank it well ABOVE 0.76, cleaner labels
lift the ceiling (label wall); if ~0.76, the wall is the features. Mikrokosmos-difficulty (Bartok, 147
pieces, expert Henle grades 1-8, open MusicXML) is a MIREX-suggested set and the best open stand-in
for CIPI (whose Zenodo record is access-gated -- needs the owner's token).

Two experiments:
  1. INTERNAL: 37 features from the 147 scores -> Henle grade, grade-stratified 5-fold LGBM tau-c.
     (All Bartok, so NO composer-disjoint split is possible -- caveat: within-composer, easier than
     the PSyllabus cross-composer protocol.) Is it > 0.76 (label wall) or ~0.76 (feature wall)?
  2. TRANSFER: train on the full PSyllabus 7,899 (phase3c cache), predict Mikrokosmos. Does the
     feature->difficulty map generalize to an independent expert-graded corpus (MIREX-private-like)?
     Run full-37 AND a tempo-invariant PITCH-ONLY subset (scores have no tempo -> timing features are
     scale-mismatched across datasets; pitch/interval/complexity features transfer cleanly).

Caveats: MusicXML scores carry no performed velocity (default 64 -> velocity features degenerate) and
no tempo (onsets converted at a fixed 120bpm -> timing features are nominal, not performed).

Run from model/:  uv run --script src/claim_measurement/difficulty/phase5e_cross_dataset.py
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from phase3c_explore import candidate_features, load_or_extract  # noqa: E402
from psyllabus import load_records  # noqa: E402

PRIMARY = Path("/Users/jdhiman/Documents/crescendai")
PS_DIR = PRIMARY / "model/data/raw/psyllabus"
LABELS = PS_DIR / "new_clean_data.json"
MID_ZIP = PS_DIR / "mid.zip"
MK_DIR = PRIMARY / "model/data/raw/mikrokosmos"
MK_XML = MK_DIR / "musicxml"
MK_HENLE = MK_DIR / "metadata/henle_mikrokosmos.json"
MK_FEAT_CACHE = PRIMARY / "model/data/results/mirex_phase5e_mikrokosmos_features.json"
OUT = PRIMARY / "model/data/results/mirex_phase5e_cross_dataset.json"

SEC_PER_QUARTER = 0.5  # fixed 120bpm (scores carry no tempo)
N_FOLDS = 5


def tau_c(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return None
    t = stats.kendalltau(x, y, variant="c").statistic
    return None if np.isnan(t) else float(t)


def _mikrokosmos_notes(xml_path):
    import partitura
    na = partitura.utils.ensure_notearray(partitura.load_musicxml(str(xml_path)))
    notes = [{"pitch": int(p), "onset": float(oq * SEC_PER_QUARTER),
              "offset": float((oq + dq) * SEC_PER_QUARTER), "velocity": 64}
             for p, oq, dq in zip(na["pitch"], na["onset_quarter"], na["duration_quarter"])]
    notes.sort(key=lambda d: d["onset"])
    return notes


def build_mikrokosmos():
    if MK_FEAT_CACHE.exists():
        return json.loads(MK_FEAT_CACHE.read_text())
    henle = json.loads(MK_HENLE.read_text())
    rows, fail = [], 0
    for pid, grade in henle.items():
        xml = MK_XML / f"{pid}.xml"
        if not xml.exists():
            fail += 1
            continue
        try:
            notes = _mikrokosmos_notes(xml)
            if len(notes) < 5:
                raise ValueError("too few notes")
            rows.append({"pid": pid, "grade": int(grade), **candidate_features(notes)})
        except Exception as exc:  # noqa: BLE001
            fail += 1
            print(f"  mk parse fail [{pid}]: {exc!r}", flush=True)
    MK_FEAT_CACHE.write_text(json.dumps(rows))
    print(f"Mikrokosmos: {len(rows)} scores featurized ({fail} failed)", flush=True)
    return rows


def _lgbm(seed=2026):
    import lightgbm as lgb
    return lgb.LGBMRegressor(objective="regression", n_estimators=400, learning_rate=0.03,
                             num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.9, reg_lambda=1.0, random_state=seed,
                             n_jobs=-1, verbosity=-1)


def main():
    mk = build_mikrokosmos()
    feats = [k for k in mk[0] if k not in ("pid", "grade")]
    Xmk = np.array([[r[f] for f in feats] for r in mk], float)
    ymk = np.array([r["grade"] for r in mk], int)

    # tempo-invariant subset: pitch / interval / pitch-class complexity (no timing/density/velocity)
    TIMING_MARKERS = ("ioi", "per_sec", "density", "chord", "vel", "fast_note",
                      "onset_clusters", "timing", "win_hard", "duration")
    pitch_cols = [i for i, f in enumerate(feats)
                  if not any(m in f for m in TIMING_MARKERS)]
    print(f"features: {len(feats)} total, {len(pitch_cols)} tempo-invariant (pitch/interval)\n", flush=True)

    res = {}

    # 1. INTERNAL: grade-stratified 5-fold on Mikrokosmos (single composer -> no group split)
    from sklearn.model_selection import StratifiedKFold
    taus = []
    for seed in (2026, 2027, 2028):
        oof = np.full(len(ymk), np.nan)
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for tr, te in skf.split(Xmk, ymk):
            oof[te] = _lgbm(seed).fit(Xmk[tr], ymk[tr]).predict(Xmk[te])
        t = tau_c(oof, ymk)
        if t is not None:
            taus.append(t)
    res["mikrokosmos_internal_37"] = {"mean": float(np.mean(taus)), "std": float(np.std(taus)),
                                      "n": len(mk)}

    # 2. TRANSFER: PSyllabus-trained -> Mikrokosmos (full-37 and pitch-only)
    records, _ = load_records(LABELS, MID_ZIP)
    ps_rows, _ = load_or_extract(records, use_cache=True)
    ps_feats = [k for k in ps_rows[0] if k not in ("grade", "composer")]
    assert ps_feats == feats, "feature order mismatch between PSyllabus cache and Mikrokosmos"
    Xps = np.array([[r[f] for f in ps_feats] for r in ps_rows], float)
    yps = np.array([r["grade"] for r in ps_rows], int)

    reg_full = _lgbm().fit(Xps, yps)
    res["transfer_ps_to_mk_full37"] = {"tau_c": tau_c(reg_full.predict(Xmk), ymk)}
    reg_pitch = _lgbm().fit(Xps[:, pitch_cols], yps)
    res["transfer_ps_to_mk_pitchonly"] = {"tau_c": tau_c(reg_pitch.predict(Xmk[:, pitch_cols]), ymk),
                                          "n_features": len(pitch_cols)}
    # note_count baseline transfer (sanity floor)
    nc_idx = feats.index("note_count")
    res["transfer_note_count_baseline"] = {"tau_c": tau_c(Xmk[:, nc_idx], ymk)}

    internal = res["mikrokosmos_internal_37"]["mean"]
    verdict = {
        "mikrokosmos_internal_tau_c": internal,
        "vs_psyllabus_0764": internal - 0.764,
        "cleaner_labels_lift_ceiling": bool(internal > 0.764 + 0.02),
        "transfer_holds_full37": res["transfer_ps_to_mk_full37"]["tau_c"],
        "transfer_holds_pitchonly": res["transfer_ps_to_mk_pitchonly"]["tau_c"],
    }
    OUT.write_text(json.dumps({"results": res, "verdict": verdict}, indent=2, default=float))

    print("=== PHASE 5e: CROSS-DATASET (Mikrokosmos) ===")
    print(f"  1. INTERNAL 37-feat, grade-stratified 5-fold (n={len(mk)}, single composer):")
    print(f"       tau-c = {internal:.4f} +/- {res['mikrokosmos_internal_37']['std']:.4f}")
    print(f"       vs PSyllabus 0.764: {verdict['vs_psyllabus_0764']:+.4f}   "
          f"cleaner-labels-lift-ceiling: {verdict['cleaner_labels_lift_ceiling']}")
    print(f"  2. TRANSFER PSyllabus-trained -> Mikrokosmos:")
    print(f"       full-37     tau-c = {res['transfer_ps_to_mk_full37']['tau_c']}")
    print(f"       pitch-only  tau-c = {res['transfer_ps_to_mk_pitchonly']['tau_c']} "
          f"({res['transfer_ps_to_mk_pitchonly']['n_features']} feats)")
    print(f"       note_count  tau-c = {res['transfer_note_count_baseline']['tau_c']} (floor)")
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
