# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy>=1.24.0","scipy>=1.10.0","pretty_midi>=0.2.10",
#     "scikit-learn>=1.3.0","lightgbm>=4.3.0",
# ]
# ///
"""Stage 2 (#104/#135) -- Transkun difficulty-head re-fit: gate + full-scale data pipeline.

The n=604 learning-curve GATE was a strong green: a head trained on 604 transkun pieces
(arm B tau-c 0.807) already matched the deployed 7.9k-clean head reading transkun (0.804),
still climbing; matched-N B-A = +0.050 CI[+0.029,+0.071] SIG. So the full ~7.9k transkun
re-fit is justified (#135). This harness does both the gate (curve) and the full-scale
transcribe pipeline that feeds the re-fit.

GATE background: arm A = clean-train->transkun-test (deployed analog), arm B = transkun-train
->transkun-test (the re-fit), matched-N LOO CV so B-A isolates train-DISTRIBUTION from train-
SIZE. All data paths are ABSOLUTE -> run this from anywhere (lesson from the #125-worktree-
removed-mid-run incident: run long jobs from a STABLE cwd, never a prunable worktree).

STAGES:
  prep [--per-grade K | --all] [--workers W] : grade-stratified pick OR all pieces w/ a
        youtube_link + clean MIDI; additive (unions with existing manifest); parallel yt-dlp.
  transcribe [--workers W]                   : parallel whole-piece Transkun (isolated env w/
        setuptools per #132) -> transkun_mid/{seg_id}.mid; resumable (skips existing).
  curve                                      : matched-N LOO curve + paired-bootstrap CI on B-A.
"""
import json
import subprocess
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy import stats

ISSUE_104_DIFFICULTY = Path(
    "/Users/jdhiman/Documents/crescendai/.worktrees/issue-104-mirex-difficulty"
    "/model/src/claim_measurement/difficulty"
)
if not ISSUE_104_DIFFICULTY.is_dir():
    raise SystemExit(f"issue-104 difficulty dir not found: {ISSUE_104_DIFFICULTY}")
sys.path.insert(0, str(ISSUE_104_DIFFICULTY))

import phase3e_amt_gap as p3e  # noqa: E402
from phase3c_explore import load_or_extract  # noqa: E402
from psyllabus import load_records, notes_from_midi_bytes  # noqa: E402

PRIMARY = Path("/Users/jdhiman/Documents/crescendai")
LABELS = p3e.LABELS
MID_ZIP = p3e.MID_ZIP
WORK = PRIMARY / "model/data/results/amt_gap_curve"
WAV_DIR = WORK / "wav"
TK_DIR = WORK / "transkun_mid"
MANIFEST = WORK / "manifest.json"
OUT = PRIMARY / "model/data/results/mirex_stage2_refit_curve.json"

SMALL_N_PARAMS = dict(objective="regression", n_estimators=200, learning_rate=0.05, num_leaves=7,
                      min_child_samples=5, subsample=0.9, subsample_freq=1, colsample_bytree=0.9,
                      reg_lambda=1.0, random_state=2026, n_jobs=-1, verbosity=-1)
DEPLOYED_PARAMS = dict(objective="regression", n_estimators=400, learning_rate=0.03, num_leaves=31,
                       min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.9,
                       reg_lambda=1.0, random_state=2026, n_jobs=-1, verbosity=-1)


def _select_all():
    """Every piece with a resolvable youtube_link AND a clean MIDI -- the full ~7.9k train set."""
    records, _ = load_records(LABELS, MID_ZIP)
    labels = json.loads(LABELS.read_text())
    out = []
    for rec in records:
        vid = p3e._video_id(labels.get(rec.key, {}).get("youtube_link", ""))
        if vid is not None:
            out.append((rec, vid))
    return out


def _dl_one(m):
    wav = WAV_DIR / f"{m['seg_id']}.wav"
    if wav.exists():
        return m["seg_id"], True, "cached"
    url = f"https://www.youtube.com/watch?v={m['video_id']}"
    cmd = ["yt-dlp", "-q", "--no-warnings", "-f", "bestaudio", "-x", "--audio-format", "wav",
           "--postprocessor-args", "-ar 24000 -ac 1",
           "-o", str(WAV_DIR / f"{m['seg_id']}.%(ext)s"), url]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return m["seg_id"], (r.returncode == 0 and wav.exists()), (r.stderr or "").strip()[:70]


def _build_manifest(per_grade: int, all_mode: bool):
    """Additive: union the (all | grade-stratified) selection into any existing manifest."""
    picked = _select_all() if all_mode else p3e.select_records(per_grade)
    new = [{"seg_id": p3e._seg_id(r.key), "key": r.key, "grade": r.grade,
            "video_id": vid, "midi_name": r.midi_name} for r, vid in picked]
    existing = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else []
    by_id = {m["seg_id"]: m for m in existing}
    added = 0
    for m in new:
        if m["seg_id"] not in by_id:
            by_id[m["seg_id"]] = m
            added += 1
    manifest = list(by_id.values())
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"manifest now {len(manifest)} pieces ({added} newly added, {len(existing)} kept) across "
          f"grades {sorted(set(m['grade'] for m in manifest))}", flush=True)
    return manifest


def stage_prep(per_grade: int, all_mode: bool, workers: int):
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _build_manifest(per_grade, all_mode)
    todo = [m for m in manifest if not (WAV_DIR / f"{m['seg_id']}.wav").exists()]
    print(f"{len(todo)} to download ({workers} workers)", flush=True)
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for done, (seg, success, err) in enumerate(
                (f.result() for f in as_completed({ex.submit(_dl_one, m) for m in todo})), 1):
            if success:
                ok += 1
            else:
                fail += 1
            if done % 100 == 0:
                print(f"  dl {done}/{len(todo)} (ok={ok} fail={fail})", flush=True)
    print(f"\ndownload done ok={ok} fail={fail}. Next: --stage transcribe.", flush=True)


def _tk_one(wav):
    out = TK_DIR / f"{wav.stem}.mid"
    cmd = ["uv", "run", "--no-project", "--with", "transkun", "--with", "setuptools",
           "--python", "3.11", "transkun", str(wav), str(out), "--device", "cpu"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return wav.stem, (r.returncode == 0 and out.exists()), (r.stderr or "").strip()[-90:]


def _pipe_one(m):
    """Disk-safe per-piece unit for the full run: download -> transcribe -> DROP the wav, so peak
    disk is bounded to only the in-flight wavs (not ~79GB of the whole corpus at once)."""
    seg = m["seg_id"]
    if (TK_DIR / f"{seg}.mid").exists():
        return seg, "skip", ""
    wav = WAV_DIR / f"{seg}.wav"
    try:
        if not wav.exists():
            _, dok, derr = _dl_one(m)
            if not dok:
                return seg, "dlfail", derr
        _, tok, terr = _tk_one(wav)
        return seg, ("ok" if tok else "tkfail"), terr
    finally:
        wav.unlink(missing_ok=True)


def stage_pipeline(per_grade: int, all_mode: bool, workers: int):
    """Full-run driver: build (additive) manifest, then download+transcribe+drop-wav per piece."""
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    TK_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _build_manifest(per_grade, all_mode)
    todo = [m for m in manifest if not (TK_DIR / f"{m['seg_id']}.mid").exists()]
    print(f"{len(todo)}/{len(manifest)} pieces need a MIDI ({workers} workers, disk-safe: wav dropped "
          f"after each)", flush=True)
    tally = {"ok": 0, "skip": 0, "dlfail": 0, "tkfail": 0}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for done, (seg, status, err) in enumerate(
                (f.result() for f in as_completed({ex.submit(_pipe_one, m) for m in todo})), 1):
            tally[status] = tally.get(status, 0) + 1
            if status in ("dlfail", "tkfail"):
                print(f"  {status} {seg[:40]}: {err}", flush=True)
            if done % 50 == 0:
                print(f"  progress {done}/{len(todo)} {tally}", flush=True)
    print(f"\npipeline done {tally}. MIDIs in {TK_DIR}. Next: --stage curve.", flush=True)


def stage_transcribe(workers: int):
    TK_DIR.mkdir(parents=True, exist_ok=True)
    todo = [w for w in sorted(WAV_DIR.glob("*.wav")) if not (TK_DIR / f"{w.stem}.mid").exists()]
    print(f"{len(todo)} wavs to transcribe ({workers} workers)", flush=True)
    if not todo:
        print("nothing to do -> --stage curve.", flush=True)
        return
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for done, (stem, success, err) in enumerate(
                (f.result() for f in as_completed({ex.submit(_tk_one, w) for w in todo})), 1):
            if success:
                ok += 1
            else:
                fail += 1
                print(f"  FAIL {stem[:40]}: {err}", flush=True)
            if done % 50 == 0:
                print(f"  progress {done}/{len(todo)} (ok={ok} fail={fail})", flush=True)
    print(f"\ntranscribed ok={ok} fail={fail} -> {TK_DIR}. Next: --stage curve.", flush=True)


def _loo_predict(train_X, test_X, y, params):
    import lightgbm as lgb
    n = len(y)
    preds = np.empty(n)
    for i in range(n):
        tr = np.arange(n) != i
        preds[i] = lgb.LGBMRegressor(**params).fit(train_X[tr], y[tr]).predict(test_X[i:i + 1])[0]
    return preds


def _boot_tauc_diff(pred_a, pred_b, grades, n_boot=5000, seed=2026):
    n = len(grades)
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        d[b] = (stats.kendalltau(pred_a[idx], grades[idx], variant="c").statistic
                - stats.kendalltau(pred_b[idx], grades[idx], variant="c").statistic)
    lo, hi = (float(x) for x in np.percentile(d, [2.5, 97.5]))
    return lo, hi, float(np.mean(d > 0))


def _paired_rows():
    manifest = json.loads(MANIFEST.read_text())
    rows = []
    with zipfile.ZipFile(MID_ZIP) as zf:
        for m in manifest:
            tk = TK_DIR / f"{m['seg_id']}.mid"
            if not tk.exists():
                continue
            tk_notes = notes_from_midi_bytes(tk.read_bytes())
            clean_notes = notes_from_midi_bytes(zf.read(m["midi_name"]))
            cf, tf = p3e._feats_from_notes(clean_notes), p3e._feats_from_notes(tk_notes)
            if cf is None or tf is None:
                continue
            rows.append({"grade": m["grade"], "clean": cf, "transkun": tf})
    return rows


def stage_curve():
    import lightgbm as lgb
    rows = _paired_rows()
    n_all = len(rows)
    print(f"paired pieces (clean + transkun both present): {n_all}", flush=True)
    if n_all < 30:
        raise SystemExit(f"only {n_all} paired pieces")

    feats = list(rows[0]["clean"].keys())
    grades_all = np.array([r["grade"] for r in rows])
    clean_all = np.array([[r["clean"][f] for f in feats] for r in rows], float)
    tk_all = np.array([[r["transkun"][f] for f in feats] for r in rows], float)

    full_records, _ = load_records(LABELS, MID_ZIP)
    full_rows, _ = load_or_extract(full_records, use_cache=True)
    fX = np.array([[r[f] for f in feats] for r in full_rows], float)
    fy = np.array([r["grade"] for r in full_rows], int)
    deployed = lgb.LGBMRegressor(**DEPLOYED_PARAMS).fit(fX, fy)
    dep_clean = p3e.tau_c(deployed.predict(clean_all), grades_all)
    dep_tk = p3e.tau_c(deployed.predict(tk_all), grades_all)
    print(f"  deployed 7.9k head on full subset (n={n_all}): clean={dep_clean:.4f}  transkun={dep_tk:.4f}",
          flush=True)

    order = np.random.default_rng(2026).permutation(n_all)
    Ns = sorted(set(n for n in (38, 75, 150, 300, 600, 1000, n_all) if n <= n_all))

    curve = []
    for N in Ns:
        sel = order[:N]
        g, cX, tX = grades_all[sel], clean_all[sel], tk_all[sel]
        predA = _loo_predict(cX, tX, g, SMALL_N_PARAMS)
        predB = _loo_predict(tX, tX, g, SMALL_N_PARAMS)
        predC = _loo_predict(cX, cX, g, SMALL_N_PARAMS)
        tauA, tauB, tauC = (p3e.tau_c(predA, g), p3e.tau_c(predB, g), p3e.tau_c(predC, g))
        lo, hi, p = _boot_tauc_diff(predB, predA, g)
        curve.append({"N": int(N), "arm_A_clean_to_transkun": tauA, "arm_B_transkun_to_transkun": tauB,
                      "arm_C_clean_ceiling": tauC, "B_minus_A": tauB - tauA,
                      "B_minus_A_ci95": [lo, hi], "p_B_gt_A": p, "B_minus_A_significant": lo > 0.0,
                      "B_vs_deployed_transkun": tauB - dep_tk})
        print(f"  N={N:4d}  A={tauA:.3f}  B={tauB:.3f}  C={tauC:.3f}  "
              f"B-A={tauB - tauA:+.3f} CI[{lo:+.3f},{hi:+.3f}] P={p:.2f} "
              f"{'SIG' if lo > 0 else 'noise'}   B-vs-deployed={tauB - dep_tk:+.3f}", flush=True)

    top = curve[-1]
    crossed = top["arm_B_transkun_to_transkun"] >= dep_tk
    if top["B_minus_A_significant"] and crossed:
        verdict = (f"RETRAIN JUSTIFIED (strong): N={top['N']} arm B {top['arm_B_transkun_to_transkun']:.3f} "
                   f"beats matched-N clean-on-transkun by {top['B_minus_A']:+.3f} (CI excludes 0) AND has "
                   f"reached/passed the 7.9k-deployed transkun {dep_tk:.3f} with far less data.")
    elif top["B_minus_A_significant"]:
        verdict = (f"RETRAIN LIKELY: N={top['N']} B-A={top['B_minus_A']:+.3f} significant but arm B "
                   f"{top['arm_B_transkun_to_transkun']:.3f} not yet past deployed {dep_tk:.3f}; extrapolate trend.")
    else:
        verdict = (f"RETRAIN DEAD: N={top['N']} B-A={top['B_minus_A']:+.3f} within noise "
                   f"(CI [{top['B_minus_A_ci95'][0]:+.3f},{top['B_minus_A_ci95'][1]:+.3f}]).")

    summary = {"n_paired": n_all, "deployed_7900_head_on_subset": {"clean": dep_clean, "transkun": dep_tk},
               "learning_curve": curve, "verdict": verdict,
               "design_note": "matched-N LOO CV, identical SMALL_N_PARAMS; B-A isolates distribution from size."}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"summary": summary}, indent=2))
    print(f"\n  VERDICT: {verdict}\n  wrote {OUT}", flush=True)


def main():
    def _arg(flag, default):
        return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default
    stage = _arg("--stage", "curve")
    per_grade = int(_arg("--per-grade", 16))
    workers = int(_arg("--workers", 4))
    all_mode = "--all" in sys.argv
    if stage == "prep":
        stage_prep(per_grade, all_mode, workers)
    elif stage == "transcribe":
        stage_transcribe(workers)
    elif stage == "pipeline":
        stage_pipeline(per_grade, all_mode, workers)
    elif stage == "curve":
        stage_curve()
    else:
        raise SystemExit(f"unknown --stage {stage!r} (prep|transcribe|pipeline|curve)")


if __name__ == "__main__":
    main()
