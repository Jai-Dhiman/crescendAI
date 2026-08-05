"""#138 Phase 1 gate (i): encoder-as-feature-extractor. Discards every fold's
trained head, scores fold f's mean-pooled emb_fold{f}.npz embeddings with
RidgeCV through bakeoff_cv.py's OWN composer-disjoint folds, and reports the
paired-bootstrap delta against features37|ridge on the SAME folds.

    cd model && uv run python -m claim_measurement.difficulty.ft_eval \\
        --data-root /path/to/model/data --fold-emb-dir /path/to/fold_embeddings

Per-fold X differs (each fold has its own adapter), which is why this needs
oof_tau_per_fold rather than bakeoff_cv.oof_tau_ridge -- see that function's
docstring. Seed is FIXED at 2026 (not averaged over multiple seeds like the
Phase 0 comparison): a set of per-fold adapters is welded to the (n_folds,
seed) pair that produced their training pools -- see the design spec.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from claim_measurement.difficulty.bakeoff_cv import composer_disjoint_folds

N_FOLDS, SEED = 5, 2026
ALPHAS = np.logspace(-1, 5, 25)


def oof_tau_per_fold(emb_by_fold: dict, y: np.ndarray, composers: np.ndarray,
                      n_folds: int, seed: int) -> np.ndarray:
    """OOF predictions where X differs per fold: for fold f, BOTH the ridge
    head's train rows and its test rows come from emb_by_fold[f] -- the
    embeddings extracted by fold f's own adapter. Mixing rows across adapters
    would score a head fit on one encoder against another encoder's
    features."""
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    oof = np.full(len(y), np.nan)
    for f, test_idx in enumerate(composer_disjoint_folds(composers, n_folds, seed)):
        if f not in emb_by_fold:
            raise KeyError(f"emb_by_fold is missing fold {f}")
        X = emb_by_fold[f]
        train_idx = np.setdiff1d(np.arange(len(y)), test_idx)
        if len(train_idx) < 3 or len(test_idx) == 0:
            continue
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
        model.fit(np.nan_to_num(X[train_idx]), y[train_idx])
        oof[test_idx] = model.predict(np.nan_to_num(X[test_idx]))
    return oof


from claim_measurement.difficulty.bakeoff_cv import paired_boot, tau_c  # noqa: E402
from claim_measurement.difficulty.bakeoff_npz import read_embedding_npz  # noqa: E402
from claim_measurement.difficulty.bakeoff_paths import resolve_paths  # noqa: E402
from claim_measurement.difficulty.train_fold import read_fold_embeddings  # noqa: E402


def _ridge_oof(X, y, composers, n_folds, seed):
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    oof = np.full(len(y), np.nan)
    for test_idx in composer_disjoint_folds(composers, n_folds, seed):
        train_idx = np.setdiff1d(np.arange(len(y)), test_idx)
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
        model.fit(np.nan_to_num(X[train_idx]), y[train_idx])
        oof[test_idx] = model.predict(np.nan_to_num(X[test_idx]))
    return oof


def _load_features37(emb_root: Path):
    paths = sorted((emb_root / "emb" / "features37").glob("*.npz"))
    if not paths:
        raise SystemExit(
            f"no features37 .npz files under {emb_root / 'emb' / 'features37'}")
    X, y, composers = [], [], []
    for path in paths:
        record = read_embedding_npz(path)
        X.append(record.embeddings["raw37"])
        y.append(record.grade)
        composers.append(record.composer_id)
    return np.stack(X), np.array(y), np.array(composers), [p.stem for p in paths]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument(
        "--fold-emb-dir", type=Path, required=True,
        help="dir containing emb_fold0.npz .. emb_fold{N_FOLDS-1}.npz from "
             "train_fold.py")
    args = ap.parse_args(argv)

    emb_root = resolve_paths(args.data_root).emb_root
    Xf, y, composers, seg_ids = _load_features37(emb_root)

    emb_by_fold = {}
    for f in range(N_FOLDS):
        fold_data = read_fold_embeddings(args.fold_emb_dir / f"emb_fold{f}.npz")
        if fold_data["seg_ids"] != seg_ids:
            raise SystemExit(
                f"emb_fold{f}.npz row order does not match features37's seg_id order; "
                f"the comparison would be unpaired")
        emb_by_fold[f] = fold_data["embeddings"]

    ft_oof = oof_tau_per_fold(emb_by_fold, y, composers, N_FOLDS, SEED)
    f37_oof = _ridge_oof(Xf, y, composers, N_FOLDS, SEED)

    print(f"n={len(y)} pieces, {len(set(composers))} composers")
    print(f"features37|ridge       tau-c {tau_c(f37_oof, y):.4f}")
    print(f"moonbeam_ft_mean|ridge tau-c {tau_c(ft_oof, y):.4f}")

    d, lo, hi, p = paired_boot(f37_oof, ft_oof, y, seed=SEED)
    verdict = "SIG" if lo > 0 else "noise"
    print(f"moonbeam_ft_mean|ridge - features37|ridge: {d:+.4f} "
          f"CI95[{lo:+.4f},{hi:+.4f}] P(diff<=0)={p:.3f} {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
