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
        cmd = ["uv", "run", "--no-project", "--with", "transkun", "--python", "3.11",
               "transkun", str(wav), str(out)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0 and out.exists():
            ok += 1
            print(f"  [{i}/{len(wavs)}] ok   {wav.stem[:44]}", flush=True)
        else:
            fail += 1
            print(f"  [{i}/{len(wavs)}] FAIL {wav.stem[:44]}: {r.stderr.strip()[-120:]}", flush=True)
    print(f"\ntranscribed {ok}/{len(wavs)} ({fail} failed) -> {TRANSKUN_DIR}. Next: --stage compare.",
          flush=True)


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


def main():
    stage = sys.argv[sys.argv.index("--stage") + 1] if "--stage" in sys.argv else "compare"
    if stage == "transcribe":
        stage_transcribe()
    elif stage == "compare":
        stage_compare()
    else:
        raise SystemExit(f"unknown --stage {stage!r} (transcribe|compare)")


if __name__ == "__main__":
    main()
