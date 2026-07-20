"""Phase 5d -- train-on-AMT augmentation: close the ~0.09 deployment gap at the source.

Every symbolic model so far trains on CLEAN MIDI and is tested on aria-amt-transcribed audio -- a
train/test distribution mismatch that costs ~0.09 tau-c (Phase 3e). This trains ON aria-amt features
so train matches test: render clean MIDI -> audio (fluidsynth) -> aria-amt -> features, train on
those, test on the SAME 43 real-audio aria-amt pieces from Phase 3e. If the amt-trained model beats
the clean-trained one (0.792) on that fixed test, the gap is (partly) closable in feature space --
and, bonus, rendered-audio training matches the MIREX synth-render anti-cheat half.

Pipeline (staged, resumable, disk-bounded -- full wavs deleted right after slicing):
  1. prep:  render+slice ~600 grade-stratified pieces (composer-disjoint from the 43 test) to 30s
            chunk wavs. uv run --script (fluidsynth + ffmpeg on PATH).
              uv run --script phase5d_train_on_amt.py --stage prep [--limit 600]
  2. transcribe (SEPARATE, the aria-amt env; ~16s/chunk on MPS):
              cd apps/inference && CRESCEND_DEVICE=auto uv run --python 3.11 --script \
                extract_amt_midi.py --wav-dir <chunks> --out-dir <amt_mid>
  3. eval:  stitch -> features -> train LGBM (full-37 / robust / parsimonious), test on the 43.
              uv run --script phase5d_train_on_amt.py --stage eval

Run from model/ (the paths are absolute to the primary checkout).
"""
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy>=1.24.0","scipy>=1.10.0","pretty_midi>=0.2.10",
#     "scikit-learn>=1.3.0","lightgbm>=4.3.0","soundfile>=0.12.0",
# ]
# ///
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from phase3c_explore import candidate_features, load_or_extract  # noqa: E402
from phase3e_amt_gap import _seg_id, _stitch_amt_notes as _stitch_test  # noqa: E402
from psyllabus import load_records, notes_from_midi_bytes  # noqa: E402

PRIMARY = Path("/Users/jdhiman/Documents/crescendai")
PS_DIR = PRIMARY / "model/data/raw/psyllabus"
LABELS = PS_DIR / "new_clean_data.json"
MID_ZIP = PS_DIR / "mid.zip"
SOUNDFONT = PRIMARY / "model/data/soundfonts/MuseScore_General.sf3"
TEST_MANIFEST = PRIMARY / "model/data/results/amt_gap/manifest.json"  # the 43-piece real-audio set

WORK = PRIMARY / "model/data/results/train_on_amt"
CHUNK_DIR = WORK / "chunks"      # {seg_id}__NNN.wav (deleted after transcription)
AMT_DIR = WORK / "amt_mid"        # {seg_id}__NNN.mid (kept; tiny)
MANIFEST = WORK / "manifest.json"
OUT = PRIMARY / "model/data/results/mirex_phase5d_train_on_amt.json"

CHUNK_S = 30.0
CLEAN_TRAINED_AMT_BASELINE = 0.792  # Phase 3e: full-clean-trained model on the amt-43 test


def tau_c(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return None
    t = stats.kendalltau(x, y, variant="c").statistic
    return None if np.isnan(t) else float(t)


def _test_composers() -> set[str]:
    """Composers of the 43-piece real-audio test set -> excluded from training (disjoint)."""
    test = json.loads(TEST_MANIFEST.read_text())
    test_keys = {m["key"] for m in test}
    records, _ = load_records(LABELS, MID_ZIP)
    return {r.composer for r in records if r.key in test_keys}


def _select(limit: int):
    """Grade-stratified stride sample, composer-disjoint from the test set, with a duration."""
    labels = json.loads(LABELS.read_text())
    records, _ = load_records(LABELS, MID_ZIP)
    excl = _test_composers()
    by_grade = {}
    for r in records:
        if r.composer in excl:
            continue
        dur = float(labels.get(r.key, {}).get("duration") or 0)
        if dur < 20:  # too short to matter / likely a fragment
            continue
        by_grade.setdefault(r.grade, []).append(r)
    per_grade = max(1, limit // len(by_grade))
    picked = []
    for g in sorted(by_grade):
        grp = by_grade[g]
        step = max(1, len(grp) // per_grade)
        picked.extend(grp[::step][:per_grade])
    return picked


def stage_prep(limit: int):
    if not SOUNDFONT.exists():
        raise FileNotFoundError(f"soundfont missing: {SOUNDFONT}")
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    picked = _select(limit)
    manifest = [{"seg_id": _seg_id(r.key), "key": r.key, "grade": r.grade,
                 "composer": r.composer, "midi_name": r.midi_name} for r in picked]
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"selected {len(manifest)} pieces (grades {sorted(set(m['grade'] for m in manifest))}), "
          f"composer-disjoint from the {len(_test_composers())} test composers", flush=True)

    done = ok = fail = 0
    with zipfile.ZipFile(MID_ZIP) as zf:
        for i, m in enumerate(manifest, 1):
            seg = m["seg_id"]
            # resumable: skip if any chunk or amt output already exists for this piece
            if list(CHUNK_DIR.glob(f"{seg}__*.wav")) or list(AMT_DIR.glob(f"{seg}__*.mid")):
                done += 1
                continue
            full = CHUNK_DIR / f"{seg}.full.wav"
            mid_tmp = CHUNK_DIR / f"{seg}.mid"
            try:
                mid_tmp.write_bytes(zf.read(m["midi_name"]))
                # render MIDI -> wav
                subprocess.run(["fluidsynth", "-ni", "-F", str(full), "-r", "24000",
                                str(SOUNDFONT), str(mid_tmp)],
                               check=True, capture_output=True)
                # slice -> 30s mono chunks {seg}__NNN.wav
                subprocess.run(["ffmpeg", "-y", "-i", str(full), "-f", "segment",
                                "-segment_time", "30", "-ar", "24000", "-ac", "1",
                                str(CHUNK_DIR / f"{seg}__%03d.wav")],
                               check=True, capture_output=True)
                mid_tmp.unlink(missing_ok=True)
                full.unlink(missing_ok=True)  # bound disk: keep only the 30s chunks
                ok += 1
            except Exception as exc:  # noqa: BLE001
                fail += 1
                full.unlink(missing_ok=True)
                mid_tmp.unlink(missing_ok=True)
                print(f"  render FAIL [{m['grade']}] {seg[:40]}: {exc!r}", flush=True)
            if i % 50 == 0:
                nchunks = len(list(CHUNK_DIR.glob("*__*.wav")))
                print(f"  {i}/{len(manifest)}  ok={ok} skip={done} fail={fail}  "
                      f"chunks={nchunks}", flush=True)
    nchunks = len(list(CHUNK_DIR.glob("*__*.wav")))
    print(f"\nprep done: {ok} rendered, {done} skipped, {fail} failed. {nchunks} chunk wavs.\n"
          f"Next: transcribe {CHUNK_DIR} -> {AMT_DIR} with the aria-amt env, then --stage eval.",
          flush=True)


def _stitch_train(seg_id: str):
    """Whole-piece AMT notes for a TRAIN piece, from 5d's amt_mid chunks (offset by NNN*30s)."""
    chunks = sorted(AMT_DIR.glob(f"{seg_id}__*.mid"))
    if not chunks:
        return None
    allnotes = []
    for cp in chunks:
        try:
            idx = int(cp.stem.rsplit("__", 1)[1])
        except (IndexError, ValueError):
            continue
        off = idx * CHUNK_S
        for n in notes_from_midi_bytes(cp.read_bytes()):
            allnotes.append({**n, "onset": n["onset"] + off, "offset": n["offset"] + off})
    allnotes.sort(key=lambda d: d["onset"])
    return allnotes


def _lgbm(seed=2026):
    import lightgbm as lgb
    return lgb.LGBMRegressor(objective="regression", n_estimators=400, learning_rate=0.03,
                             num_leaves=31, min_child_samples=30, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.9, reg_lambda=1.0, random_state=seed,
                             n_jobs=-1, verbosity=-1)


# robust (Phase 3e clean<->amt survival rho>0.8) + parsimonious (note_count + top robust pitch)
ROBUST = ["n_pitch_classes", "win_pitch_entropy_p90", "pitch_class_entropy", "pitch_entropy",
          "notes_per_sec", "onset_clusters_per_sec", "interval_entropy", "fast_note_fraction",
          "pitch_lz_complexity", "pitch_bigram_cond_entropy", "ioi_lz"]
PARSIMONIOUS = ["note_count", "pitch_lz_complexity", "pitch_entropy", "pitch_class_entropy",
                "ioi_lz", "notes_per_sec"]


def _feature_matrix(rows, feats, cols_names):
    idx = [feats.index(f) for f in cols_names]
    X = np.array([[r[f] for f in feats] for r in rows], float)
    return X[:, idx]


def _complete_segs() -> set[str]:
    """seg_ids whose EVERY 30s wav chunk has a transcribed .mid (partial pieces excluded --
    a truncated stitch would corrupt whole-piece features)."""
    import collections
    wav = collections.Counter(p.stem.rsplit("__", 1)[0] for p in CHUNK_DIR.glob("*__*.wav"))
    mid = collections.Counter(p.stem.rsplit("__", 1)[0] for p in AMT_DIR.glob("*__*.mid"))
    return {s for s, n in wav.items() if n > 0 and mid.get(s, 0) == n}


def stage_eval():
    manifest = json.loads(MANIFEST.read_text())
    complete = _complete_segs()

    # TRAIN: complete pieces only -> BOTH rendered-amt features and clean-MIDI features (matched set,
    # so amt-vs-clean isolates the train DISTRIBUTION, not the training-set SIZE).
    amt_rows, clean_match_rows = [], []
    with zipfile.ZipFile(MID_ZIP) as zf:
        for m in manifest:
            if m["seg_id"] not in complete:
                continue
            amt_notes = _stitch_train(m["seg_id"])
            clean_notes = notes_from_midi_bytes(zf.read(m["midi_name"]))
            if not (amt_notes and len(amt_notes) >= 5 and len(clean_notes) >= 5):
                continue
            amt_rows.append({"grade": m["grade"], **candidate_features(amt_notes)})
            clean_match_rows.append({"grade": m["grade"], **candidate_features(clean_notes)})
    # TEST: the 43 real-audio AMT pieces (Phase 3e)
    test = json.loads(TEST_MANIFEST.read_text())
    test_rows = []
    for m in test:
        notes = _stitch_test(m["seg_id"])
        if notes and len(notes) >= 5:
            test_rows.append({"grade": m["grade"], **candidate_features(notes)})

    feats = [k for k in amt_rows[0] if k != "grade"]
    y_train = np.array([r["grade"] for r in amt_rows], int)
    yte = np.array([r["grade"] for r in test_rows], int)
    n = len(amt_rows)
    is_pilot = n < len(manifest) * 0.9
    print(f"train pieces (COMPLETE): {n}/{len(manifest)}"
          f"{'  [PILOT -- transcription still running]' if is_pilot else ''}"
          f"   test (real-amt-43): {len(test_rows)}", flush=True)

    # clean-trained on the FULL 7,899 (the deployment baseline that gave 0.792)
    records, _ = load_records(LABELS, MID_ZIP)
    clean_full_rows, _ = load_or_extract(records, use_cache=True)
    ycl_full = np.array([r["grade"] for r in clean_full_rows], int)

    subsets = {"full_37": feats, "robust_11": ROBUST, "parsimonious_6": PARSIMONIOUS}
    res = {}
    for name, cols in subsets.items():
        Xamt = _feature_matrix(amt_rows, feats, cols)
        Xcl_m = _feature_matrix(clean_match_rows, feats, cols)
        Xte = _feature_matrix(test_rows, feats, cols)
        Xcl_f = _feature_matrix(clean_full_rows, feats, cols)
        amt_trained = tau_c(_lgbm().fit(Xamt, y_train).predict(Xte), yte)
        clean_matched = tau_c(_lgbm().fit(Xcl_m, y_train).predict(Xte), yte)   # same N, clean
        clean_full = tau_c(_lgbm().fit(Xcl_f, ycl_full).predict(Xte), yte)     # 7,899 clean
        res[name] = {"n_features": len(cols), "n_train": n,
                     "amt_trained_tau_c": amt_trained,
                     "clean_trained_matchedN_tau_c": clean_matched,
                     "clean_trained_full7899_tau_c": clean_full,
                     "distribution_gain_amt_vs_cleanMatched": (amt_trained - clean_matched)
                     if (amt_trained and clean_matched) else None}
        print(f"  [{name}] ({len(cols)}f, n_train={n})  amt={amt_trained}  "
              f"clean-matchedN={clean_matched}  clean-full7899={clean_full}  "
              f"dist-gain={res[name]['distribution_gain_amt_vs_cleanMatched']}", flush=True)

    best = max(res, key=lambda k: res[k]["amt_trained_tau_c"] or -1)
    verdict = {
        "is_pilot": is_pilot, "n_train": n,
        "clean_trained_full7899_baseline": CLEAN_TRAINED_AMT_BASELINE,
        "best_subset": best,
        "best_amt_trained_tau_c": res[best]["amt_trained_tau_c"],
        # does matching the train distribution (amt) beat clean at the SAME N?
        "train_on_amt_helps_distribution": bool(any(
            (r["distribution_gain_amt_vs_cleanMatched"] or 0) > 0.01 for r in res.values())),
        # does amt-trained (at this N) reach the full-clean deployment baseline?
        "amt_trained_reaches_full_baseline": bool(
            (res[best]["amt_trained_tau_c"] or 0) >= CLEAN_TRAINED_AMT_BASELINE),
    }
    OUT.write_text(json.dumps({"n_train": n, "n_test": len(test_rows), "is_pilot": is_pilot,
                               "results": res, "verdict": verdict}, indent=2, default=float))
    print("\n=== PHASE 5d: TRAIN-ON-AMT" + ("  [PILOT]" if is_pilot else "") + " ===")
    print(f"  test = 43 real-audio aria-amt pieces;  full-7899-clean baseline = {CLEAN_TRAINED_AMT_BASELINE}")
    for name, r in res.items():
        print(f"    {name:16s} amt(n={r['n_train']})={r['amt_trained_tau_c']}  "
              f"clean-matchedN={r['clean_trained_matchedN_tau_c']}  "
              f"clean-full={r['clean_trained_full7899_tau_c']}  "
              f"dist-gain={r['distribution_gain_amt_vs_cleanMatched']}")
    print(f"  train-on-AMT helps (distribution, matched N): {verdict['train_on_amt_helps_distribution']}")
    print(f"  amt-trained reaches full-clean baseline 0.792: {verdict['amt_trained_reaches_full_baseline']}")
    print(f"  wrote {OUT}")


def main():
    argv = sys.argv
    stage = argv[argv.index("--stage") + 1] if "--stage" in argv else "prep"
    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv else 600
    if stage == "prep":
        stage_prep(limit)
    elif stage == "eval":
        stage_eval()
    else:
        raise SystemExit(f"unknown --stage {stage!r} (prep|eval)")


if __name__ == "__main__":
    main()
