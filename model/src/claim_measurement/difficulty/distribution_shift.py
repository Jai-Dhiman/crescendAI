"""#167 (#104 S2), parts 1-2: what synthesized audio costs, decomposed.

## The protocol is deployment-shaped, not symmetric

For each fold *f*, the ridge head is fit on **human** embeddings from that
fold's training rows and used to predict that fold's test rows from **the
arm's** embeddings. That asymmetry is the point: MIREX trains us on real
recordings and tests us on synthesis. Fitting the head on synthesized rows
would measure a system we cannot build, because we do not have the test set.

Across 5 folds each eval piece is scored exactly once, by the adapter that held
it out -- so an arm needs 900 renders and 900 transcriptions, not 4,500.

## Four arms, and what each delta is allowed to mean

    A   human       cached Transkun of the human recording   (the 0.8395 anchor)
    B0  source      mid.zip embedded directly
    B   expressive  mid.zip -> fluidsynth -> Transkun
    C   deadpan     deadpan(mid.zip) -> fluidsynth -> Transkun

    B0 - A   two transcription sources of one performance -- a confound
    B  - B0  the render/re-transcribe round trip -- the synthesis penalty
    C  - B   dynamics and pedal removed -- the score-rendering penalty

**B0 is not a result and must not be reported as one.** Arm A passes through
one transcription (of the human audio); B and C pass through two, since
`mid.zip` is itself a transcription. A bare B - A would bundle the synthesis
penalty with double-transcription loss and read worse than the truth.

**The two real deltas point at different fixes**, which is what #170's go/no-go
turns on. A multimodal audio branch bypasses Transkun, so it can rescue damage
that shows up in `B - B0`. It cannot rescue damage in `C - B`: a deadpan
rendering is genuinely different playing, and the fix there is training on
rendered audio, not adding a branch.

## Comparison discipline

Every arm is scored on the identical row set in the identical order, and the
evaluation refuses to run if any arm is missing a row -- `paired_boot` on
mismatched populations is worse than no comparison. `C - B` in particular is
paired piece-for-piece, which is where most of the power comes from.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from claim_measurement.difficulty.bakeoff_cv import (
    composer_disjoint_folds,
    paired_boot,
    tau_c,
)

N_FOLDS, SEED = 5, 2026
ALPHAS = np.logspace(-1, 5, 25)

HUMAN_ARM = "human"
SOURCE_ARM = "source"
ARM_ORDER = (HUMAN_ARM, SOURCE_ARM, "expressive", "deadpan")

# Only these deltas are reported, and each is listed with the one thing it is
# allowed to mean. Ordered so the confound is subtracted before either real
# effect is read.
DELTAS = (
    (HUMAN_ARM, SOURCE_ARM, "transcription source (a confound, not a result)"),
    (SOURCE_ARM, "expressive", "the synthesis penalty -- #170 can address this"),
    ("expressive", "deadpan", "the score-rendering penalty -- #170 cannot"),
    (HUMAN_ARM, "expressive",
     "end to end vs a human recording; BUNDLES the confound above, so it is an "
     "upper bound on the synthesis penalty, not the penalty"),
    (HUMAN_ARM, "deadpan", "end to end, human recording vs score rendering"),
)


def _fit_predict(X_train, y_train, X_test):
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
    model.fit(np.nan_to_num(X_train), y_train)
    return model.predict(np.nan_to_num(X_test))


def shift_oof(human_by_fold, arm_by_fold, y, composers,
              n_folds: int = N_FOLDS, seed: int = SEED) -> np.ndarray:
    """Out-of-fold predictions: head fit on HUMAN train rows, applied to the
    ARM's test rows.

    Pass `arm_by_fold is human_by_fold` for the baseline arm, which then
    reproduces `ft_eval.oof_tau_per_fold` exactly -- the anchor and the arms
    must come off the same code path or the delta measures the code.
    """
    oof = np.full(len(y), np.nan)
    for f, test_idx in enumerate(composer_disjoint_folds(composers, n_folds, seed)):
        for source, name in ((human_by_fold, "human"), (arm_by_fold, "arm")):
            if f not in source:
                raise KeyError(f"{name} embeddings are missing fold {f}")
        X_train_all, X_test_all = human_by_fold[f], arm_by_fold[f]
        if X_train_all.shape != X_test_all.shape:
            raise ValueError(
                f"fold {f}: human embeddings are {X_train_all.shape} but the "
                f"arm's are {X_test_all.shape}; the comparison would be unpaired")
        train_idx = np.setdiff1d(np.arange(len(y)), test_idx)
        if len(train_idx) < 3 or len(test_idx) == 0:
            continue
        oof[test_idx] = _fit_predict(X_train_all[train_idx], y[train_idx],
                                     X_test_all[test_idx])
    return oof


def stack_arm(vectors_by_seg, seg_ids, dim: int) -> np.ndarray:
    """Arm vectors in the canonical eval row order.

    Raises on any missing seg_id. An arm with holes still produces a tau-c, and
    it would be a tau-c over a different population than the one it is compared
    against -- so a short arm is refused rather than scored.
    """
    missing = [s for s in seg_ids if s not in vectors_by_seg]
    if missing:
        raise ValueError(
            f"arm is missing {len(missing)} of {len(seg_ids)} eval pieces "
            f"(e.g. {missing[:3]}); rerun the arm rather than scoring a "
            f"shorter population")
    out = np.stack([np.asarray(vectors_by_seg[s], dtype=np.float64).reshape(-1)
                    for s in seg_ids])
    if out.shape[1] != dim:
        raise ValueError(
            f"arm vectors are {out.shape[1]}-dim but the human embeddings are "
            f"{dim}-dim")
    if not np.all(np.isfinite(out)):
        bad = [s for s, row in zip(seg_ids, out) if not np.all(np.isfinite(row))]
        # Deliberately NOT nan_to_num'd, for the same reason score_wav refuses
        # to: a non-finite MoonBeam embedding means the backbone produced
        # garbage, and zeroing it emits a confident score from a broken pass.
        raise ValueError(
            f"{len(bad)} arm vectors are non-finite (e.g. {bad[:3]})")
    return out


def report(arms: dict, y: np.ndarray, composers: np.ndarray, log=print) -> dict:
    """tau-c per arm plus the paired deltas. `arms` maps arm name -> the
    per-fold embedding dict for that arm."""
    human = arms[HUMAN_ARM]
    oof = {name: shift_oof(human, by_fold, y, composers)
           for name, by_fold in arms.items()}

    scored = np.all([np.isfinite(v) for v in oof.values()], axis=0)
    y_s = y[scored]
    log(f"n={len(y_s)} pieces scored in every arm\n")
    taus = {}
    for name in ARM_ORDER:
        if name not in oof:
            continue
        taus[name] = float(tau_c(oof[name][scored], y_s))
        log(f"  {name:12s} tau-c {taus[name]:.4f}")

    log("")
    deltas = {}
    for a, b, meaning in DELTAS:
        if a not in oof or b not in oof:
            continue
        d, lo, hi, p = paired_boot(oof[a][scored], oof[b][scored], y_s, seed=SEED)
        verdict = "SIG" if (lo > 0 or hi < 0) else "noise"
        deltas[f"{b}-{a}"] = {"delta": float(d), "ci95": [float(lo), float(hi)],
                              "p": float(p), "verdict": verdict,
                              "meaning": meaning}
        log(f"  {b} - {a}: {d:+.4f} CI95[{lo:+.4f},{hi:+.4f}] "
            f"P(diff<=0)={p:.3f} {verdict}")
        log(f"      {meaning}")
    return {"n": int(len(y_s)), "tau_c": taus, "deltas": deltas}


# --------------------------------------------------------------------------
# The embedding pass
# --------------------------------------------------------------------------


def embed_arm(seg_ids, fold_by_seg, midi_for, embedder_for, log=print) -> dict:
    """seg_id -> embedding, grouping by fold so each adapter loads once.

    The 1.6GB backbone plus a LoRA adapter takes far longer to load than the
    ~1s an embedding costs, so 900 loads would dominate a pass that should take
    twenty minutes.
    """
    out = {}
    by_fold = {}
    for seg_id in seg_ids:
        by_fold.setdefault(fold_by_seg[seg_id], []).append(seg_id)
    for fold in sorted(by_fold):
        log(f"fold {fold}: embedding {len(by_fold[fold])} pieces")
        embed = embedder_for(fold)
        for seg_id in by_fold[fold]:
            out[seg_id] = np.asarray(embed(midi_for(seg_id)),
                                     dtype=np.float64).reshape(-1)
    return out


def write_arm_npz(path: Path, vectors_by_seg: dict) -> None:
    seg_ids = sorted(vectors_by_seg)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, seg_ids=np.array(seg_ids, dtype=object),
        embeddings=np.stack([vectors_by_seg[s] for s in seg_ids]))


def read_arm_npz(path: Path) -> dict:
    data = np.load(Path(path), allow_pickle=True)
    return {str(s): v for s, v in zip(data["seg_ids"], data["embeddings"])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--fold-emb-dir", type=Path, required=True)
    # NOT type=Path: these are NAME=PATH pairs, split below. Declaring Path
    # here parses cleanly and then dies on .partition at use time.
    ap.add_argument("--arm-npz", nargs="*", default=(), metavar="NAME=PATH",
                    help="e.g. expressive=/path/expressive.npz")
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args(argv)

    from claim_measurement.difficulty.bakeoff_npz import read_embedding_npz
    from claim_measurement.difficulty.bakeoff_paths import (
        features37_dir,
        features37_seg_ids,
        resolve_paths,
    )
    from claim_measurement.difficulty.train_fold import read_fold_embeddings

    paths = resolve_paths(args.data_root)
    f37 = features37_dir(paths.emb_root)
    seg_ids = features37_seg_ids(f37)
    records = [read_embedding_npz(f37 / f"{s}.npz") for s in seg_ids]
    y = np.array([r.grade for r in records])
    composers = np.array([r.composer_id for r in records])

    human = {}
    for f in range(N_FOLDS):
        fold = read_fold_embeddings(args.fold_emb_dir / f"emb_fold{f}.npz")
        if fold["seg_ids"] != seg_ids:
            raise SystemExit(
                f"emb_fold{f}.npz row order does not match features37's; the "
                f"comparison would be unpaired")
        human[f] = fold["embeddings"]

    arms = {HUMAN_ARM: human}
    dim = human[0].shape[1]
    for spec in args.arm_npz:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--arm-npz wants NAME=PATH, got {spec!r}")
        stacked = stack_arm(read_arm_npz(Path(path)), seg_ids, dim)
        # One arm, one matrix -- but shift_oof indexes per fold, and the arm's
        # rows are the SAME rows for every fold (a piece is only ever scored by
        # the adapter that held it out, so only that fold's rows are read).
        arms[name] = {f: stacked for f in range(N_FOLDS)}

    result = report(arms, y, composers)
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
