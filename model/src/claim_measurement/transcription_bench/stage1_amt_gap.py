# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy>=1.24.0","scipy>=1.10.0","pretty_midi>=0.2.10",
#     "scikit-learn>=1.3.0","lightgbm>=4.3.0",
# ]
# ///
"""Stage 1 (#125) -- does Transkun beat aria-amt on the REAL-AUDIO difficulty gap?

Stage 0 showed Transkun > aria-amt on all 7 expressive metrics, but IN-DOMAIN (MAESTRO).
The deployment question is out-of-domain: #104's phase3e already measured aria-amt's deployed
tau-c on 43 grade-stratified YouTube pieces = 0.7924 (clean-MIDI-trained difficulty model,
tested on features re-extracted from transcribed audio; clean-subset reference = 0.8829).

This harness re-transcribes the SAME 43 pieces with Transkun and re-runs the identical
comparison, adding a Transkun arm beside aria-amt so both are measured on one clean baseline:
  * deployed tau-c (clean-trained model, tested on transkun features vs aria features)
  * per-feature clean<->transcribed survival Spearman -- aria vs transkun, side by side
  * note-count over/under-transcription ratio -- aria vs transkun

It reuses #104's exact loaders (candidate_features, psyllabus, phase3e helpers) by importing
phase3e_amt_gap from the issue-104 worktree, so the clean baseline and feature code are identical.

GATE: does Transkun's deployed tau-c exceed aria-amt's 0.7924 (and close the gap to 0.8829)?

Interpretation caveat baked into the report: candidate_features EXCLUDES note offsets/durations,
so Transkun's headline Stage-0 win (note-offset F1 0.79 vs 0.37) does NOT feed this pipeline.
The difficulty-relevant levers here are onset F1, velocity, chord-span, timing + OOD robustness.

STAGES (resumable):
  1. WAVs: produced by #104's phase3e_amt_gap.py --stage prep (writes amt_gap/wav/). Aria MIDIs
     (amt_gap/amt_mid/, 30s chunks) already exist from #104.
  2. transcribe:  uv run --script stage1_amt_gap.py --stage transcribe
     -> amt_gap/wav/{seg_id}.wav -> amt_gap/transkun_mid/{seg_id}.mid (whole-piece, no 30s cap).
     Shells out per file to the isolated transkun env (must NOT share this script's env).
  3. compare:  uv run --script stage1_amt_gap.py --stage compare
     -> reads clean MIDI + aria chunks + transkun MIDI, extracts features, reports the gate.
"""
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
from scipy import stats

# --- wire in #104's difficulty pipeline (exact same feature code + clean baseline) ---
ISSUE_104_DIFFICULTY = Path(
    "/Users/jdhiman/Documents/crescendai/.worktrees/issue-104-mirex-difficulty"
    "/model/src/claim_measurement/difficulty"
)
if not ISSUE_104_DIFFICULTY.is_dir():
    raise SystemExit(f"issue-104 difficulty dir not found: {ISSUE_104_DIFFICULTY}")
sys.path.insert(0, str(ISSUE_104_DIFFICULTY))

import phase3e_amt_gap as p3e  # noqa: E402  (brings candidate_features, psyllabus, paths)
from phase3c_explore import load_or_extract  # noqa: E402
from psyllabus import load_records, notes_from_midi_bytes  # noqa: E402

# reuse #104's paths verbatim so the clean baseline is byte-identical
WORK = p3e.WORK
WAV_DIR = p3e.WAV_DIR
ARIA_DIR = p3e.AMT_DIR
MANIFEST = p3e.MANIFEST
MID_ZIP = p3e.MID_ZIP
LABELS = p3e.LABELS
TRANSKUN_DIR = WORK / "transkun_mid"
OUT = Path("/Users/jdhiman/Documents/crescendai/model/data/results/mirex_stage1_transkun_gap.json")

ARIA_DEPLOYED_TAU_C = 0.7924283396430503  # from mirex_phase3e_amt_gap.json (the gate to beat)


def stage_transcribe():
    """Whole-piece Transkun transcription of each downloaded wav. No 30s stitching (unlike aria)."""
    TRANSKUN_DIR.mkdir(parents=True, exist_ok=True)
    wavs = sorted(WAV_DIR.glob("*.wav"))
    if not wavs:
        raise SystemExit(f"no wavs in {WAV_DIR} -- run #104 phase3e_amt_gap.py --stage prep first")
    ok, fail = 0, 0
    for i, wav in enumerate(wavs, 1):
        out = TRANSKUN_DIR / f"{wav.stem}.mid"
        if out.exists():
            ok += 1
            continue
        # isolated transkun env -- must NOT reuse this script's uv-run env (torch vs our deps)
        cmd = ["uv", "run", "--no-project", "--with", "transkun", "--with", "setuptools",
               "--python", "3.11", "transkun", str(wav), str(out)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0 and out.exists():
            ok += 1
            print(f"  [{i}/{len(wavs)}] ok   {wav.stem[:44]}", flush=True)
        else:
            fail += 1
            print(f"  [{i}/{len(wavs)}] FAIL {wav.stem[:44]}: {r.stderr.strip()[-120:]}", flush=True)
    print(f"\ntranscribed {ok}/{len(wavs)} ({fail} failed) -> {TRANSKUN_DIR}. Next: --stage compare.",
          flush=True)
    if fail:
        raise SystemExit(f"Transkun failed for {fail}/{len(wavs)} benchmark files")


def _transkun_notes(seg_id: str):
    """Whole-piece Transkun notes (single MIDI, no chunk offsetting)."""
    mid = TRANSKUN_DIR / f"{seg_id}.mid"
    if not mid.exists():
        return None
    return notes_from_midi_bytes(mid.read_bytes())


def _survival_rho(clean_col, other_col):
    ok = ~(np.isnan(clean_col) | np.isnan(other_col))
    return float(stats.spearmanr(clean_col[ok], other_col[ok]).statistic) if ok.sum() >= 3 else None


def stage_compare():
    manifest = json.loads(MANIFEST.read_text())
    rows = []
    with zipfile.ZipFile(MID_ZIP) as zf:
        for m in manifest:
            aria_notes = p3e._stitch_amt_notes(m["seg_id"])   # 30s chunks stitched
            tk_notes = _transkun_notes(m["seg_id"])           # whole piece
            if not aria_notes or not tk_notes:
                continue
            try:
                clean_notes = notes_from_midi_bytes(zf.read(m["midi_name"]))
                cf = p3e._feats_from_notes(clean_notes)
                af = p3e._feats_from_notes(aria_notes)
                tf = p3e._feats_from_notes(tk_notes)
                if cf is None or af is None or tf is None:
                    continue
                rows.append({"grade": m["grade"], "clean": cf, "aria": af, "transkun": tf})
            except Exception as exc:  # noqa: BLE001
                print(f"  compare skip {m['seg_id'][:40]}: {exc!r}", flush=True)

    n = len(rows)
    print(f"paired pieces (clean + aria + transkun all present): {n}", flush=True)
    if n < 10:
        print("  too few paired pieces for a stable tau-c; transcribe more.", flush=True)
        if n == 0:
            raise SystemExit("no paired pieces -- did --stage transcribe run?")

    feats = list(rows[0]["clean"].keys())
    grades = np.array([r["grade"] for r in rows])
    clean_X = np.array([[r["clean"][f] for f in feats] for r in rows], float)
    aria_X = np.array([[r["aria"][f] for f in feats] for r in rows], float)
    tk_X = np.array([[r["transkun"][f] for f in feats] for r in rows], float)

    # per-feature clean<->transcribed survival rho, aria vs transkun, on the SAME pieces
    per_feat = {}
    for j, f in enumerate(feats):
        per_feat[f] = {
            "tau_c_clean": p3e.tau_c(clean_X[:, j], grades),
            "survive_aria": _survival_rho(clean_X[:, j], aria_X[:, j]),
            "survive_transkun": _survival_rho(clean_X[:, j], tk_X[:, j]),
        }

    nc = feats.index("note_count")
    ratio_aria = float(np.nanmedian(aria_X[:, nc] / np.clip(clean_X[:, nc], 1, None)))
    ratio_tk = float(np.nanmedian(tk_X[:, nc] / np.clip(clean_X[:, nc], 1, None)))

    # deployed tau-c: LightGBM trained on the FULL clean set (identical to phase3e), tested on
    # this subset's clean / aria / transkun features
    import lightgbm as lgb
    full_records, _ = load_records(LABELS, MID_ZIP)
    full_rows, _ = load_or_extract(full_records, use_cache=True)
    fX = np.array([[r[f] for f in feats] for r in full_rows], float)
    fy = np.array([r["grade"] for r in full_rows], int)
    reg = lgb.LGBMRegressor(objective="regression", n_estimators=400, learning_rate=0.03,
                            num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
                            colsample_bytree=0.9, reg_lambda=1.0, random_state=2026, n_jobs=-1,
                            verbosity=-1)
    reg.fit(fX, fy)
    pred_clean, pred_aria, pred_tk = reg.predict(clean_X), reg.predict(aria_X), reg.predict(tk_X)
    tau_clean = p3e.tau_c(pred_clean, grades)
    tau_aria = p3e.tau_c(pred_aria, grades)
    tau_tk = p3e.tau_c(pred_tk, grades)

    # Is the transkun-vs-aria deployed gap real, or noise on ~39 pieces? Paired bootstrap over
    # pieces (same resample for both arms) -> CI on the tau-c difference. Seeded, no Math.random.
    rng = np.random.default_rng(2026)
    diffs = np.empty(5000)
    for b in range(5000):
        idx = rng.integers(0, n, n)
        diffs[b] = (stats.kendalltau(pred_tk[idx], grades[idx], variant="c").statistic
                    - stats.kendalltau(pred_aria[idx], grades[idx], variant="c").statistic)
    ci_lo, ci_hi = (float(x) for x in np.percentile(diffs, [2.5, 97.5]))
    p_tk_better = float(np.mean(diffs > 0))
    tie = ci_lo <= 0 <= ci_hi  # CI straddles zero -> statistically indistinguishable

    beats_aria = tau_tk is not None and tau_tk > tau_aria
    if tie:
        verdict = (f"TIE: deployed tau-c difference (transkun {tau_tk:.3f} vs aria {tau_aria:.3f}) is "
                   f"within noise (95% CI [{ci_lo:+.3f},{ci_hi:+.3f}], P(tk>aria)={p_tk_better:.2f}). "
                   f"Transcriber swap alone is tau-c-neutral for #104 difficulty. Per-feature survival "
                   f"favors transkun; the lever is Stage-2 (fine-tune transkun's ~7% note under-count, "
                   f"or re-fit the difficulty head on transkun features -- it overfit aria's biases).")
    elif beats_aria:
        verdict = (f"SHIP: transkun deployed tau-c {tau_tk:.3f} > aria {tau_aria:.3f} "
                   f"(95% CI [{ci_lo:+.3f},{ci_hi:+.3f}]).")
    else:
        verdict = (f"REGRESSION: transkun deployed tau-c {tau_tk:.3f} < aria {tau_aria:.3f}, CI excludes "
                   f"zero ([{ci_lo:+.3f},{ci_hi:+.3f}]). Do not swap without Stage-2 fine-tune.")

    summary = {
        "n_paired": n,
        "note_count_over_clean_median_ratio": {"aria": ratio_aria, "transkun": ratio_tk},
        "deployed_tau_c": {
            "clean_subset_reference": tau_clean,
            "aria_amt": tau_aria,
            "transkun": tau_tk,
            "aria_gate_from_phase3e": ARIA_DEPLOYED_TAU_C,
            "transkun_minus_aria": (tau_tk - tau_aria) if (tau_tk and tau_aria) else None,
            "transkun_beats_aria_this_run": beats_aria,
            "bootstrap_diff_ci95": [ci_lo, ci_hi],
            "bootstrap_p_transkun_better": p_tk_better,
            "statistical_tie": tie,
        },
        "verdict": verdict,
        "per_feature": per_feat,
        "note": ("candidate_features excludes note offsets/durations, so Transkun's Stage-0 "
                 "note-offset F1 win (0.79 vs 0.37) does not feed this pipeline; the levers here "
                 "are onset/velocity/chord-span/timing survival + OOD robustness."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"summary": summary}, indent=2))

    print("\n=== STAGE 1: TRANSKUN vs ARIA-AMT ON THE REAL-AUDIO DIFFICULTY GAP (#125) ===")
    print(f"  paired pieces: {n}")
    print(f"  note-count/clean median ratio   aria={ratio_aria:.2f}  transkun={ratio_tk:.2f}")
    print(f"\n  deployed tau-c (clean-trained difficulty model, tested on transcribed features):")
    print(f"    clean-subset reference = {tau_clean}")
    print(f"    aria-amt               = {tau_aria}")
    print(f"    transkun               = {tau_tk}")
    print(f"    gate (phase3e aria)    = {ARIA_DEPLOYED_TAU_C:.4f}")
    print(f"    transkun - aria        = {summary['deployed_tau_c']['transkun_minus_aria']}")
    print(f"    bootstrap diff 95% CI  = [{ci_lo:+.4f}, {ci_hi:+.4f}]   P(tk>aria)={p_tk_better:.2f}"
          f"   {'TIE (CI straddles 0)' if tie else 'significant'}")
    print(f"\n  VERDICT: {verdict}")
    print(f"\n  per-feature survival (clean<->transcribed Spearman), aria vs transkun, worst-aria first:")
    print(f"    {'feature':26s} {'tau_c_clean':>11s} {'survive_aria':>12s} {'survive_tk':>11s} {'Δ(tk-aria)':>11s}")
    def _sk(kv):
        return kv[1]["survive_aria"] if kv[1]["survive_aria"] is not None else 1.0
    for f, d in sorted(per_feat.items(), key=_sk):
        sa, st = d["survive_aria"], d["survive_transkun"]
        delta = (st - sa) if (sa is not None and st is not None) else None
        print(f"    {f:26s} {str(round(d['tau_c_clean'],3) if d['tau_c_clean'] else None):>11s} "
              f"{str(round(sa,3) if sa is not None else None):>12s} "
              f"{str(round(st,3) if st is not None else None):>11s} "
              f"{str(round(delta,3) if delta is not None else None):>11s}")
    print(f"\n  wrote {OUT}")


# ============================ Stage 2, LEVER 1 probe (#125) ============================
# Stage 1 found the deployed gap (transkun 0.790 vs aria 0.816) is a calibration artifact:
# the FROZEN LightGBM was trained on CLEAN score note-counts, and aria's ~4% over-count lands
# closer to that clean distribution than transkun's ~7% under-count. LEVER 1 asks: if we RE-FIT
# the head on transkun-transcribed features, does the gap close (and does it beat aria's 0.816)?
#
# The honest full test needs transkun transcriptions of all ~7.9k TRAIN pieces (a large GPU/yt-dlp
# job). This is the CHEAP proxy: re-fit on the ~39 transcriptions we already have, cross-validated.
#
# CONFOUND (why a naive "does the re-fit head beat 0.816?" is unfair): the deployed 0.816 head
# trained on 7,899 pieces; a re-fit head here can only train on ~38 (LOO within the subset). Phase-5d
# showed train-SET-SIZE dominates (clean-full-7899 ~0.79 >> clean-matched-75 ~0.70). So a 38-piece
# head loses to 0.816 on size grounds ALONE, independent of calibration. The confound-free signal is
# a MATCHED-N contrast -- both arms trained on the same ~38 pieces, differing only in feature source:
#   arm A  clean-train -> transkun-test  (the deployment mismatch, matched-N)
#   arm B  transkun-train -> transkun-test  (LEVER 1: calibration matched, matched-N)
# B - A isolates the calibration effect from the train-size effect. We ALSO report the literal bar
# (B vs the 7.9k deployed aria=0.816) so the underpowered comparison is on record, correctly framed.

# small-N head config, held IDENTICAL across arms A/B/C so the contrast is fair (the deployed
# min_child_samples=40 cannot split 38 rows -> would collapse to a constant; tune for N~38 instead).
SMALL_N_PARAMS = dict(objective="regression", n_estimators=200, learning_rate=0.05, num_leaves=7,
                      min_child_samples=5, subsample=0.9, subsample_freq=1, colsample_bytree=0.9,
                      reg_lambda=1.0, random_state=2026, n_jobs=-1, verbosity=-1)
DEPLOYED_PARAMS = dict(objective="regression", n_estimators=400, learning_rate=0.03, num_leaves=31,
                       min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.9,
                       reg_lambda=1.0, random_state=2026, n_jobs=-1, verbosity=-1)
OUT_REFIT = Path("/Users/jdhiman/Documents/crescendai/model/data/results/mirex_stage2_refit_probe.json")


def _paired_rows():
    """The 39 pieces with clean + aria + transkun features (same construction as stage_compare)."""
    manifest = json.loads(MANIFEST.read_text())
    rows = []
    with zipfile.ZipFile(MID_ZIP) as zf:
        for m in manifest:
            aria_notes = p3e._stitch_amt_notes(m["seg_id"])
            tk_notes = _transkun_notes(m["seg_id"])
            if not aria_notes or not tk_notes:
                continue
            clean_notes = notes_from_midi_bytes(zf.read(m["midi_name"]))
            cf, af, tf = (p3e._feats_from_notes(clean_notes),
                          p3e._feats_from_notes(aria_notes),
                          p3e._feats_from_notes(tk_notes))
            if cf is None or af is None or tf is None:
                continue
            rows.append({"grade": m["grade"], "clean": cf, "aria": af, "transkun": tf})
    return rows


def _loo_predict(train_X, test_X, y, params):
    """Leave-one-out CV predictions: fit on the other n-1 train rows, predict the held-out test row."""
    import lightgbm as lgb
    n = len(y)
    preds = np.empty(n)
    for i in range(n):
        tr = np.arange(n) != i
        preds[i] = lgb.LGBMRegressor(**params).fit(train_X[tr], y[tr]).predict(test_X[i:i + 1])[0]
    return preds


def _boot_tauc_diff(pred_a, pred_b, grades, n_boot=5000, seed=2026):
    """Paired bootstrap over pieces: 95% CI + P(a>b) on tau_c(a) - tau_c(b) (same resample both arms)."""
    n = len(grades)
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        d[b] = (stats.kendalltau(pred_a[idx], grades[idx], variant="c").statistic
                - stats.kendalltau(pred_b[idx], grades[idx], variant="c").statistic)
    lo, hi = (float(x) for x in np.percentile(d, [2.5, 97.5]))
    return lo, hi, float(np.mean(d > 0))


def stage_refit():
    import lightgbm as lgb
    rows = _paired_rows()
    n = len(rows)
    print(f"paired pieces (clean + aria + transkun all present): {n}", flush=True)
    if n < 10:
        raise SystemExit(f"too few paired pieces ({n}) for a stable tau-c")

    feats = list(rows[0]["clean"].keys())
    grades = np.array([r["grade"] for r in rows])
    clean_X = np.array([[r["clean"][f] for f in feats] for r in rows], float)
    aria_X = np.array([[r["aria"][f] for f in feats] for r in rows], float)
    tk_X = np.array([[r["transkun"][f] for f in feats] for r in rows], float)

    # --- reference: reconstruct the DEPLOYED 7.9k-clean-trained head (validate it reproduces
    #     Stage-1's 0.816/0.790 before trusting anything new -- wave-5 instrument discipline) ---
    full_records, _ = load_records(LABELS, MID_ZIP)
    full_rows, _ = load_or_extract(full_records, use_cache=True)
    fX = np.array([[r[f] for f in feats] for r in full_rows], float)
    fy = np.array([r["grade"] for r in full_rows], int)
    deployed = lgb.LGBMRegressor(**DEPLOYED_PARAMS).fit(fX, fy)
    dep_pred_aria, dep_pred_tk = deployed.predict(aria_X), deployed.predict(tk_X)
    tau_dep_aria, tau_dep_tk = p3e.tau_c(dep_pred_aria, grades), p3e.tau_c(dep_pred_tk, grades)
    print(f"  [instrument check] deployed 7.9k head: aria={tau_dep_aria:.4f} (exp ~0.8158)  "
          f"transkun={tau_dep_tk:.4f} (exp ~0.7897)", flush=True)

    # --- matched-N LOO arms (all train on ~38 pieces, IDENTICAL small-N config) ---
    predA = _loo_predict(clean_X, tk_X, grades, SMALL_N_PARAMS)   # clean-train -> transkun-test
    predB = _loo_predict(tk_X, tk_X, grades, SMALL_N_PARAMS)      # LEVER 1: transkun -> transkun
    predC = _loo_predict(clean_X, clean_X, grades, SMALL_N_PARAMS)  # matched-N clean ceiling
    tauA, tauB, tauC = (p3e.tau_c(predA, grades), p3e.tau_c(predB, grades), p3e.tau_c(predC, grades))

    # --- the two decisive contrasts ---
    cal_lo, cal_hi, cal_p = _boot_tauc_diff(predB, predA, grades)          # calibration delta (B-A)
    bar_lo, bar_hi, bar_p = _boot_tauc_diff(predB, dep_pred_aria, grades)  # literal bar (B vs 0.816)

    bar_pass = bar_lo > 0.0          # LEVER-1 head beats deployed aria, CI excludes zero
    cal_real = cal_lo > 0.0          # calibration matching helps, CI excludes zero (confound-free)

    if bar_pass:
        verdict = (f"PASS (unexpected): LEVER-1 re-fit head (transkun-trained, LOO) tau-c {tauB:.3f} "
                   f"beats deployed aria {tau_dep_aria:.3f}, 95% CI [{bar_lo:+.3f},{bar_hi:+.3f}] "
                   f"excludes zero. The gap WAS pure head calibration -- full 7.9k transkun re-fit warranted.")
    elif cal_real:
        verdict = (f"LITERAL BAR NOT MET, but CALIBRATION IS A REAL LEVER at matched-N: B-A = "
                   f"{tauB - tauA:+.3f}, 95% CI [{cal_lo:+.3f},{cal_hi:+.3f}] excludes zero. Re-fitting "
                   f"the head on transkun features helps it read transkun; the subset can't clear the "
                   f"absolute 0.816 (7.9k-trained) bar on train-size grounds (Phase-5d). A full-7.9k "
                   f"transkun re-fit is the only way to test the absolute bar -- borderline worth it.")
    else:
        verdict = (f"DEAD: neither test passes. LEVER-1 head tau-c {tauB:.3f} does NOT beat deployed "
                   f"aria {tau_dep_aria:.3f} (bar CI [{bar_lo:+.3f},{bar_hi:+.3f}]), AND the confound-free "
                   f"calibration delta B-A={tauB - tauA:+.3f} is within noise (CI [{cal_lo:+.3f},"
                   f"{cal_hi:+.3f}]). Re-fitting the head on transkun features does NOT close the gap even "
                   f"at matched-N. This probe does not prove a full-7.9k refit cannot help; it shows no "
                   f"reliable effect that would justify that expensive experiment. Under the original "
                   f"Gate-0 decision rule, #104 Stage-2 has no positive-EV lever left.")

    summary = {
        "n_paired": n,
        "deployed_7900_trained_reference": {
            "clean_subset": p3e.tau_c(deployed.predict(clean_X), grades),
            "aria_amt": tau_dep_aria, "transkun": tau_dep_tk,
        },
        "matched_N_loo_arms": {
            "A_clean_train_transkun_test": tauA,
            "B_transkun_train_transkun_test_LEVER1": tauB,
            "C_clean_train_clean_test_ceiling": tauC,
            "note": "all trained on ~38 pieces, identical SMALL_N_PARAMS; only feature source differs",
        },
        "calibration_delta_B_minus_A": {
            "point": tauB - tauA, "ci95": [cal_lo, cal_hi], "p_B_gt_A": cal_p,
            "is_real_confound_free": cal_real,
        },
        "literal_stage2_bar_B_vs_deployed_aria": {
            "point": tauB - tau_dep_aria, "ci95": [bar_lo, bar_hi], "p_B_gt_aria": bar_p,
            "passes": bar_pass,
            "caveat": "B trains on ~38 pieces, deployed aria on 7,899; this comparison is train-size "
                      "confounded (unfair to B). Use the matched-N calibration delta for the real signal.",
        },
        "verdict": verdict,
        "params": {"small_n": SMALL_N_PARAMS, "deployed": DEPLOYED_PARAMS},
    }
    OUT_REFIT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REFIT.write_text(json.dumps({"summary": summary}, indent=2))

    print("\n=== STAGE 2, LEVER 1 PROBE: re-fit difficulty head on transkun features (#125) ===")
    print(f"  paired pieces: {n}")
    print(f"\n  deployed 7.9k-clean-trained head (train-size {len(fy)}):")
    print(f"    clean-subset = {summary['deployed_7900_trained_reference']['clean_subset']:.4f}")
    print(f"    aria-amt     = {tau_dep_aria:.4f}   (the 0.816 bar)")
    print(f"    transkun     = {tau_dep_tk:.4f}   (the gap Lever 1 tries to close)")
    print(f"\n  matched-N LOO arms (all trained on ~{n - 1} pieces, identical config):")
    print(f"    A  clean-train  -> transkun-test  = {tauA:.4f}   (deployment mismatch)")
    print(f"    B  transkun-train-> transkun-test = {tauB:.4f}   (LEVER 1)")
    print(f"    C  clean-train  -> clean-test     = {tauC:.4f}   (matched-N clean ceiling)")
    print(f"\n  CONFOUND-FREE calibration delta  B - A = {tauB - tauA:+.4f}   "
          f"95% CI [{cal_lo:+.4f},{cal_hi:+.4f}]  P(B>A)={cal_p:.2f}  "
          f"{'REAL' if cal_real else 'within noise'}")
    print(f"  literal bar (train-size confounded) B vs deployed-aria = {tauB - tau_dep_aria:+.4f}  "
          f"95% CI [{bar_lo:+.4f},{bar_hi:+.4f}]  P(B>aria)={bar_p:.2f}  "
          f"{'PASS' if bar_pass else 'not met'}")
    print(f"\n  VERDICT: {verdict}")
    print(f"\n  wrote {OUT_REFIT}")


def main():
    stage = sys.argv[sys.argv.index("--stage") + 1] if "--stage" in sys.argv else "compare"
    if stage == "transcribe":
        stage_transcribe()
    elif stage == "compare":
        stage_compare()
    elif stage == "refit":
        stage_refit()
    else:
        raise SystemExit(f"unknown --stage {stage!r} (transcribe|compare|refit)")


if __name__ == "__main__":
    main()
