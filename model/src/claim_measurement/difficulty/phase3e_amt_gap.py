# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy>=1.24.0","scipy>=1.10.0","pretty_midi>=0.2.10",
#     "scikit-learn>=1.3.0","lightgbm>=4.3.0",
# ]
# ///
"""Phase 3e -- the AMT domain gap: does tau-c survive clean-MIDI -> aria-amt-on-audio?

Every result so far (0.764) is on CLEAN PSyllabus MIDI. The real MIREX pipeline transcribes
AUDIO with our aria-amt first, so the deployment number is unknown. This measures the drop on
a grade-stratified subset: each piece's YouTube audio -> our aria-amt -> the 37 features, vs the
same features from the clean PSyllabus MIDI. Reports:
  * tau-c(clean features, grade)   -- the reference on this subset
  * tau-c(amt features, grade)     -- the deployment estimate (train+test on AMT)
  * tau-c(clean-trained model on AMT features, grade) -- the REAL MIREX mismatch (train clean / test amt)
  * per-feature clean<->amt correlation -- which features survive transcription (Phase-1 style)

STAGES (resumable; the transcription stage runs in the aria-amt venv, not here):
  1. select+download:  uv run --script phase3e_amt_gap.py --stage prep [--per-grade 4]
     -> writes manifest.json + {seg_id}.wav (24k mono) into the work dir.
  2. transcribe (SEPARATE, aria-amt venv):
     CRESCEND_DEVICE=cpu <aria_env>/bin/python apps/inference/extract_amt_midi.py \
        --wav-dir <work>/wav --out-dir <work>/amt_mid
  3. compare:  uv run --script phase3e_amt_gap.py --stage compare
     -> reads clean MIDI (mid.zip) + amt_mid/{seg_id}.mid, extracts features, reports the gap.
"""
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase3c_explore import candidate_features  # noqa: E402  (the 37-feature superset)
from psyllabus import load_records, notes_from_midi_bytes  # noqa: E402

PRIMARY = Path("/Users/jdhiman/Documents/crescendai")
PS_DIR = PRIMARY / "model/data/raw/psyllabus"
LABELS = PS_DIR / "new_clean_data.json"
MID_ZIP = PS_DIR / "mid.zip"
WORK = PRIMARY / "model/data/results/amt_gap"
WAV_DIR = WORK / "wav"
AMT_DIR = WORK / "amt_mid"
MANIFEST = WORK / "manifest.json"
OUT = PRIMARY / "model/data/results/mirex_phase3e_amt_gap.json"


def tau_c(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return None
    t = stats.kendalltau(x, y, variant="c").statistic
    return None if np.isnan(t) else float(t)


def _seg_id(key: str) -> str:
    """Filesystem-safe id for a PSyllabus key."""
    return "".join(c if c.isalnum() else "_" for c in key)[:80]


def _video_id(youtube_link: str) -> str | None:
    if not youtube_link:
        return None
    vid = youtube_link.rstrip("/").split("/")[-1].split("?")[0]
    return vid or None


def select_records(per_grade: int):
    """Grade-stratified deterministic pick (~per_grade pieces per grade 0..10) with a resolvable
    video id and a present clean MIDI. Stride within each grade group (the label JSON is grouped)."""
    records, _ = load_records(LABELS, MID_ZIP)
    labels = json.loads(LABELS.read_text())
    by_grade = {}
    for rec in records:
        vid = _video_id(labels.get(rec.key, {}).get("youtube_link", ""))
        if vid is None:
            continue
        by_grade.setdefault(rec.grade, []).append((rec, vid))
    picked = []
    for grade in sorted(by_grade):
        group = by_grade[grade]
        step = max(1, len(group) // per_grade)
        picked.extend(group[::step][:per_grade])
    return picked


def stage_prep(per_grade: int):
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    picked = select_records(per_grade)
    manifest = [{"seg_id": _seg_id(r.key), "key": r.key, "grade": r.grade,
                 "video_id": vid, "midi_name": r.midi_name} for r, vid in picked]
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"selected {len(manifest)} pieces across grades "
          f"{sorted(set(m['grade'] for m in manifest))}", flush=True)

    ok, fail = 0, 0
    for m in manifest:
        wav = WAV_DIR / f"{m['seg_id']}.wav"
        if wav.exists():
            ok += 1
            continue
        url = f"https://www.youtube.com/watch?v={m['video_id']}"
        cmd = ["yt-dlp", "-q", "--no-warnings", "-f", "bestaudio", "-x",
               "--audio-format", "wav", "--postprocessor-args", "-ar 24000 -ac 1",
               "-o", str(WAV_DIR / f"{m['seg_id']}.%(ext)s"), url]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0 and wav.exists():
            ok += 1
            print(f"  dl ok   [{m['grade']}] {m['seg_id'][:40]}", flush=True)
        else:
            fail += 1
            print(f"  dl FAIL [{m['grade']}] {m['seg_id'][:40]}: {r.stderr.strip()[:80]}", flush=True)
    print(f"\ndownloaded {ok}/{len(manifest)} ({fail} failed). "
          f"Next: transcribe wav/ -> amt_mid/ with the aria-amt venv, then --stage compare.", flush=True)


def _feats_from_notes(notes):
    if len(notes) < 5:
        return None
    return candidate_features(notes)


def _stitch_amt_notes(seg_id: str) -> list | None:
    """Reconstruct whole-piece AMT notes from the 30s-chunk MIDIs (aria-amt caps at 30s/chunk).
    Chunk {seg_id}__NNN.mid holds notes relative to its own start; offset each by NNN*30s."""
    chunks = sorted(AMT_DIR.glob(f"{seg_id}__*.mid"))
    if not chunks:
        return None
    CHUNK_S = 30.0
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


def stage_compare():
    manifest = json.loads(MANIFEST.read_text())
    rows = []
    with zipfile.ZipFile(MID_ZIP) as zf:
        for m in manifest:
            amt_notes = _stitch_amt_notes(m["seg_id"])  # whole-piece, stitched from 30s chunks
            if not amt_notes:
                continue
            try:
                clean_notes = notes_from_midi_bytes(zf.read(m["midi_name"]))
                cf, af = _feats_from_notes(clean_notes), _feats_from_notes(amt_notes)
                if cf is None or af is None:
                    continue
                rows.append({"grade": m["grade"], "clean": cf, "amt": af})
            except Exception as exc:  # noqa: BLE001
                print(f"  compare skip {m['seg_id'][:40]}: {exc!r}", flush=True)
    n = len(rows)
    print(f"paired pieces (clean + amt both present): {n}", flush=True)
    if n < 10:
        print("  too few paired pieces for a stable tau-c; download+transcribe more.", flush=True)

    feats = list(rows[0]["clean"].keys())
    grades = np.array([r["grade"] for r in rows])
    clean_X = np.array([[r["clean"][f] for f in feats] for r in rows], float)
    amt_X = np.array([[r["amt"][f] for f in feats] for r in rows], float)

    # per-feature: tau-c vs grade (clean & amt) + clean<->amt Spearman (transcription survival)
    per_feat = {}
    for j, f in enumerate(feats):
        c, a = clean_X[:, j], amt_X[:, j]
        ok = ~(np.isnan(c) | np.isnan(a))
        rho = float(stats.spearmanr(c[ok], a[ok]).statistic) if ok.sum() >= 3 else None
        per_feat[f] = {"tau_c_clean": tau_c(c, grades), "tau_c_amt": tau_c(a, grades),
                       "clean_amt_spearman": rho}

    # note-count sanity: how badly does aria-amt over/under-transcribe note counts?
    nc_ratio = float(np.nanmedian(amt_X[:, feats.index("note_count")] /
                                   np.clip(clean_X[:, feats.index("note_count")], 1, None)))

    # model-level gap: LightGBM on the FULL clean set (train), tested on this subset's clean vs amt
    import lightgbm as lgb
    from phase3c_explore import load_or_extract
    full_records, _ = load_records(LABELS, MID_ZIP)
    full_rows, _ = load_or_extract(full_records, use_cache=True)  # phase3c cache
    fX = np.array([[r[f] for f in feats] for r in full_rows], float)
    fy = np.array([r["grade"] for r in full_rows], int)
    reg = lgb.LGBMRegressor(objective="regression", n_estimators=400, learning_rate=0.03,
                            num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
                            colsample_bytree=0.9, reg_lambda=1.0, random_state=2026, n_jobs=-1, verbosity=-1)
    reg.fit(fX, fy)
    tau_model_clean = tau_c(reg.predict(clean_X), grades)
    tau_model_amt = tau_c(reg.predict(amt_X), grades)

    summary = {
        "n_paired": n,
        "note_count_amt_over_clean_median_ratio": nc_ratio,
        "per_feature": per_feat,
        "model_full_clean_trained": {
            "tau_c_on_clean_subset": tau_model_clean,
            "tau_c_on_amt_subset": tau_model_amt,
            "gap": (tau_model_clean - tau_model_amt) if (tau_model_clean and tau_model_amt) else None,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"summary": summary}, indent=2))

    print("\n=== PHASE 3e: AMT DOMAIN GAP (clean MIDI -> aria-amt on audio) ===")
    print(f"  paired pieces: {n}   aria-amt/clean note-count median ratio: {nc_ratio:.2f}")
    print(f"\n  full-clean-trained model tau-c on this subset:")
    print(f"    on CLEAN features = {tau_model_clean}")
    print(f"    on AMT   features = {tau_model_amt}")
    print(f"    GAP (clean - amt) = {summary['model_full_clean_trained']['gap']}")
    print(f"\n  per-feature (tau-c clean | tau-c amt | clean<->amt survival rho), worst survival first:")
    for f, d in sorted(per_feat.items(), key=lambda kv: (kv[1]['clean_amt_spearman'] or 1)):
        print(f"    {f:26s} {str(round(d['tau_c_clean'],3) if d['tau_c_clean'] else None):>7s} "
              f"{str(round(d['tau_c_amt'],3) if d['tau_c_amt'] else None):>7s}  survive={d['clean_amt_spearman']}")
    print(f"\n  wrote {OUT}")


def main():
    stage = sys.argv[sys.argv.index("--stage") + 1] if "--stage" in sys.argv else "prep"
    per_grade = int(sys.argv[sys.argv.index("--per-grade") + 1]) if "--per-grade" in sys.argv else 4
    if stage == "prep":
        stage_prep(per_grade)
    elif stage == "compare":
        stage_compare()
    else:
        raise SystemExit(f"unknown --stage {stage!r} (prep|compare)")


if __name__ == "__main__":
    main()
