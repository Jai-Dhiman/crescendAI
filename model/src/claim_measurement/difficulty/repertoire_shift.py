"""#167 (#104 S2), part 3: what a genuine repertoire shift costs.

The contract says the private test repertoire is *"selected from sources
outside existing public piano difficulty datasets, including contemporary
works"*, and our composer-disjoint CV does not test repertoire shift at all --
worse, on the n=900 eval set composer-disjointness is **vacuous** (900
composers, 900 pieces), so those folds are effectively a random 5-way split.
Holding out an entire period is the first genuinely hard split we would have.

`period` from `new_clean_data.json` supplies it: 311 of the 900 eval pieces are
Modern (the eval set runs 71.4% C20+Modern, more contemporary than the 51.5% of
the training corpus). `syllabus` (17 systems here) supports the same treatment
for grading-system shift.

## Three arms, because a bare hold-out has a size confound

Dropping 311 Modern pieces shrinks each fold's ridge training pool by roughly a
third, and a smaller pool scores worse for reasons that have nothing to do with
repertoire. So the hold-out is measured against a **size-matched control** that
trains on a random subset of the same size drawn from the full pool:

    full          head fit on all of fold f's train rows          (reference)
    size_matched  fit on a random same-size subset of them        (control)
    holdout       fit on fold f's train rows with Modern removed  (the arm)

`holdout - size_matched` is the repertoire effect. `size_matched - full` is
what the missing rows alone cost, and reporting it separately is what stops the
two being read as one number.

## This is a floor, and the reason is not a footnote

Only the **ridge head** is refit here. The five LoRA adapters were fine-tuned on
pools that included Modern pieces, and re-running that fine-tune is GPU spend
(human-lit). So the measured drop understates a true never-saw-Modern system.
Report it as head-level repertoire shift, never as the system's repertoire
robustness.
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

ARMS = ("full", "size_matched", "holdout")


def _fit_predict(X_train, y_train, X_test):
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
    model.fit(np.nan_to_num(X_train), y_train)
    return model.predict(np.nan_to_num(X_test))


def holdout_oof(emb_by_fold, y, composers, holdout: np.ndarray, arm: str,
                n_folds: int = N_FOLDS, seed: int = SEED) -> np.ndarray:
    """Out-of-fold predictions for the held-out rows only; NaN elsewhere.

    `holdout` is a boolean mask over the canonical eval row order. Each fold's
    embeddings come from that fold's own adapter, exactly as in
    `ft_eval.oof_tau_per_fold` -- mixing rows across adapters would score a head
    fit on one encoder against another encoder's features.
    """
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    holdout = np.asarray(holdout, dtype=bool)
    if holdout.shape != y.shape:
        raise ValueError(
            f"holdout mask has shape {holdout.shape} but there are {y.shape} "
            f"eval rows; an off-by-one here silently scores the wrong pieces")
    if not holdout.any():
        raise ValueError("holdout mask selects no rows")

    rng = np.random.default_rng(seed)
    oof = np.full(len(y), np.nan)
    folds = composer_disjoint_folds(composers, n_folds, seed)
    for f, fold_test in enumerate(folds):
        if f not in emb_by_fold:
            raise KeyError(f"emb_by_fold is missing fold {f}")
        X = emb_by_fold[f]
        test_idx = np.array([i for i in fold_test if holdout[i]], dtype=int)
        if len(test_idx) == 0:
            continue

        pool = np.setdiff1d(np.arange(len(y)), np.asarray(fold_test))
        clean = pool[~holdout[pool]]
        if arm == "full":
            train_idx = pool
        elif arm == "holdout":
            train_idx = clean
        else:
            # Same n as the holdout arm, drawn from the full pool -- so the
            # only thing that differs between them is WHICH rows, not how many.
            train_idx = rng.choice(pool, size=len(clean), replace=False)

        if len(train_idx) < 3:
            raise ValueError(
                f"fold {f} arm {arm!r} has only {len(train_idx)} training rows")
        oof[test_idx] = _fit_predict(X[train_idx], y[train_idx], X[test_idx])
    return oof


def period_mask(seg_ids, manifest_rows, labels, period: str) -> np.ndarray:
    """Boolean mask over `seg_ids` for one `period` value.

    Raises on a seg_id whose period is missing or blank rather than treating it
    as not-Modern: 26 of the 900 eval pieces carry an empty or '-' period, and
    silently folding them into the training side would quietly contaminate the
    control with unlabelled repertoire.
    """
    key_by_seg = {row["seg_id"]: row["key"] for row in manifest_rows}
    mask, unlabelled = [], []
    for seg_id in seg_ids:
        key = key_by_seg.get(seg_id)
        value = (labels.get(key, {}) or {}).get("period", "")
        if not value or value == "-":
            unlabelled.append(seg_id)
        mask.append(value == period)
    if unlabelled:
        raise ValueError(
            f"{len(unlabelled)} eval pieces have no usable `period` "
            f"(e.g. {unlabelled[:3]}); pass --drop-unlabelled to exclude them "
            f"from BOTH sides rather than defaulting them into the train pool")
    return np.array(mask, dtype=bool)


def labelled_mask(seg_ids, manifest_rows, labels) -> np.ndarray:
    """Rows with a usable `period`. Everything else is dropped from both sides."""
    key_by_seg = {row["seg_id"]: row["key"] for row in manifest_rows}
    out = []
    for seg_id in seg_ids:
        value = (labels.get(key_by_seg.get(seg_id), {}) or {}).get("period", "")
        out.append(bool(value) and value != "-")
    return np.array(out, dtype=bool)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--fold-emb-dir", type=Path, required=True)
    ap.add_argument("--period", default="Modern")
    ap.add_argument("--drop-unlabelled", action="store_true")
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

    manifest_rows = json.loads(Path(paths.manifest).read_text())
    labels = json.loads(Path(paths.labels).read_text())

    keep = (labelled_mask(seg_ids, manifest_rows, labels) if args.drop_unlabelled
            else np.ones(len(seg_ids), dtype=bool))
    if not keep.all():
        print(f"dropping {int((~keep).sum())} eval pieces with no usable period")
        seg_ids = [s for s, k in zip(seg_ids, keep) if k]
        y, composers = y[keep], composers[keep]

    emb_by_fold = {}
    for f in range(N_FOLDS):
        fold = read_fold_embeddings(args.fold_emb_dir / f"emb_fold{f}.npz")
        index = {s: i for i, s in enumerate(fold["seg_ids"])}
        missing = [s for s in seg_ids if s not in index]
        if missing:
            raise SystemExit(
                f"emb_fold{f}.npz is missing {len(missing)} eval rows "
                f"(e.g. {missing[:3]}); the comparison would be unpaired")
        emb_by_fold[f] = fold["embeddings"][[index[s] for s in seg_ids]]

    holdout = period_mask(seg_ids, manifest_rows, labels, args.period)
    print(f"n={len(y)} eval pieces, {int(holdout.sum())} in period "
          f"{args.period!r} ({100 * holdout.mean():.1f}%)")

    oof = {arm: holdout_oof(emb_by_fold, y, composers, holdout, arm)
           for arm in ARMS}
    rows = holdout & ~np.isnan(oof["full"])
    y_h = y[rows]
    print(f"scored on {len(y_h)} held-out rows\n")
    for arm in ARMS:
        print(f"  {arm:13s} tau-c {tau_c(oof[arm][rows], y_h):.4f}")

    print()
    for a, b in (("full", "size_matched"), ("size_matched", "holdout"),
                 ("full", "holdout")):
        d, lo, hi, p = paired_boot(oof[a][rows], oof[b][rows], y_h, seed=SEED)
        verdict = "SIG" if (lo > 0 or hi < 0) else "noise"
        print(f"  {b} - {a}: {d:+.4f} CI95[{lo:+.4f},{hi:+.4f}] "
              f"P(diff<=0)={p:.3f} {verdict}")

    print("\nHEAD-LEVEL ONLY: the five LoRA adapters were fine-tuned on pools\n"
          "containing this period, so this is a floor on a never-saw-it system.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
