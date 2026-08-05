"""#138 Phase 1 real-audio second gate: 709 of 900 eval pieces have local
WAVs (re-fetched this session; see design spec). Resumable transcription
(`main`) plus MIDI drift (`midi_drift`) and per-fold audio scoring
(`score_audio_subset`) -- see this module's own docstring in the plan for the
deliberate scope split between what is CLI-wired here vs. what is a runbook
snippet over these tested primitives.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from claim_measurement.difficulty.bakeoff_cv import (
    composer_disjoint_folds,
    paired_boot,
    tau_c,
)

N_FOLDS, SEED = 5, 2026
ALPHAS = np.logspace(-1, 5, 25)


def midi_drift(
    reference_notes: list, candidate_notes: list, onset_tolerance: float
) -> dict:
    """note-count delta (candidate - reference) and onset F1: a candidate
    note matches a reference note when they share pitch and onsets differ by
    <= onset_tolerance seconds. Matching is greedy nearest-onset-first, and
    each reference/candidate note is used at most once."""
    pairs = []
    for ci, c in enumerate(candidate_notes):
        for ri, r in enumerate(reference_notes):
            if r["pitch"] != c["pitch"]:
                continue
            dt = abs(r["onset"] - c["onset"])
            if dt <= onset_tolerance:
                pairs.append((dt, ci, ri))
    pairs.sort(key=lambda p: p[0])

    matched_ref, matched_cand, tp = set(), set(), 0
    for _dt, ci, ri in pairs:
        if ci in matched_cand or ri in matched_ref:
            continue
        matched_cand.add(ci)
        matched_ref.add(ri)
        tp += 1

    precision = tp / len(candidate_notes) if candidate_notes else 0.0
    recall = tp / len(reference_notes) if reference_notes else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0 else 0.0)

    return {
        "note_count_delta": len(candidate_notes) - len(reference_notes),
        "onset_f1": f1}


def _import_transcribe_wav():
    """Locate apps/inference/amt (import-safe transkun_cli) from CWD-up or
    file-up and return its transcribe_wav. Mirrors follower_eval/build_corpus.py's
    locate-and-import pattern -- kept lazy so tests that inject a fake
    transcriber never need transkun_cli's own heavy deps on the import path."""
    for base in (Path.cwd(), Path(__file__).resolve()):
        for parent in [base, *base.parents]:
            cand = parent / "apps" / "inference" / "amt"
            if (cand / "transkun_cli.py").exists():
                sys.path.insert(0, str(cand))
                from transkun_cli import transcribe_wav  # type: ignore

                return transcribe_wav
    raise RuntimeError(
        "could not locate apps/inference/amt/transkun_cli.py from CWD or module path"
    )


def _write_cache_atomic(path: Path, notes: list, pedals: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump({"notes": notes, "pedals": pedals}, fh)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def main(argv=None, transcriber=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wav-manifest", type=Path, required=True,
                    help="JSON list of {seg_id, wav_path}")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)

    if transcriber is None:
        transcriber = _import_transcribe_wav()

    entries = json.loads(args.wav_manifest.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    done, skipped, failed = 0, 0, []
    for e in entries:
        out_path = Path(args.out_dir) / f"{e['seg_id']}.json"
        if out_path.exists():
            skipped += 1
            continue
        try:
            notes, pedals = transcriber(Path(e["wav_path"]))
            _write_cache_atomic(out_path, notes, pedals)
            done += 1
        except Exception as exc:  # noqa: BLE001 -- record and continue; the report is the source of truth
            failed.append(f"{e['seg_id']}: {exc!r}")
    print(f"transcribed={done} skipped={skipped} failed={len(failed)}")
    for f in failed[:10]:
        print(f"  FAIL {f}")
    return 0 if not failed else 1


def score_audio_subset(emb_by_fold: dict, audio_embeddings: dict,
                        features37_x: np.ndarray, y: np.ndarray,
                        composers: np.ndarray, seg_ids: list,
                        n_folds: int, seed: int) -> dict:
    """For every seg_id in audio_embeddings (a subset of seg_ids), fit a ridge
    model on that piece's OWN test fold's train rows of emb_by_fold[fold] and
    score the piece's audio-derived embedding through it. Also scores the
    SAME piece's original symbolic embedding through the SAME model, so any
    audio-vs-symbolic gap is attributable to audio provenance, not to the
    subset being easier or harder (design spec's real-audio second gate, item
    (b)). THE GATE (item (a)): features37_x is scored via ordinary
    composer-disjoint OOF over the FULL piece set (fit on each fold's train
    rows, predict that fold's own test rows) -- never refit on the audio
    subset alone -- and those OOF predictions are then restricted to the
    audio subset's rows and paired-bootstrapped against the audio-derived
    predictions on those same rows."""
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    idx_of = {s: i for i, s in enumerate(seg_ids)}
    test_folds = composer_disjoint_folds(composers, n_folds, seed)
    fold_of_idx = {i: f for f, idx in enumerate(test_folds) for i in idx}

    # features37 OOF over the full set, matching folds/seed exactly -- computed
    # once here, independent of which pieces have audio, then subset below.
    f37_oof = np.full(len(y), np.nan)
    for fold, test_idx in enumerate(test_folds):
        train_idx = np.setdiff1d(np.arange(len(seg_ids)), test_idx)
        f37_model = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
        f37_model.fit(features37_x[train_idx], y[train_idx])
        f37_oof[test_idx] = f37_model.predict(features37_x[test_idx])

    audio_pred, symbolic_pred, f37_pred, subset_y = [], [], [], []
    ridge_cache: dict = {}
    for seg_id, audio_embedding in audio_embeddings.items():
        i = idx_of[seg_id]
        fold = fold_of_idx[i]
        if fold not in ridge_cache:
            train_idx = np.setdiff1d(np.arange(len(seg_ids)), test_folds[fold])
            model = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
            model.fit(emb_by_fold[fold][train_idx], y[train_idx])
            ridge_cache[fold] = model
        model = ridge_cache[fold]
        audio_pred.append(model.predict(audio_embedding.reshape(1, -1))[0])
        symbolic_pred.append(model.predict(emb_by_fold[fold][i].reshape(1, -1))[0])
        f37_pred.append(f37_oof[i])
        subset_y.append(y[i])

    subset_y = np.array(subset_y)
    audio_pred = np.array(audio_pred)
    symbolic_pred = np.array(symbolic_pred)
    f37_pred = np.array(f37_pred)
    d_sym, lo_sym, hi_sym, p_sym = paired_boot(
        symbolic_pred, audio_pred, subset_y, seed=seed)
    d_f37, lo_f37, hi_f37, p_f37 = paired_boot(
        f37_pred, audio_pred, subset_y, seed=seed)
    return {
        "n": len(subset_y),
        "audio_tau_c": tau_c(audio_pred, subset_y),
        "symbolic_tau_c": tau_c(symbolic_pred, subset_y),
        "features37_tau_c": tau_c(f37_pred, subset_y),
        "delta_vs_symbolic": d_sym, "ci_lo_vs_symbolic": lo_sym,
        "ci_hi_vs_symbolic": hi_sym, "p_le_0_vs_symbolic": p_sym,
        "delta_vs_features37": d_f37, "ci_lo_vs_features37": lo_f37,
        "ci_hi_vs_features37": hi_f37, "p_le_0_vs_features37": p_f37,
    }


if __name__ == "__main__":
    sys.exit(main())
