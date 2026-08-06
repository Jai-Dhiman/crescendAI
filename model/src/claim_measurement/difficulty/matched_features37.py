"""#149 matched-features37 arm: the supervision-matched baseline that decides
whether gate (i)'s +0.0357 is an encoder win or a supervision artifact.

Gate (i) (ft_eval.py) fits the features37 RidgeCV on the ~720 EVAL rows outside
each test fold, while the LoRA fine-tune trained on the ~3,800 NON-EVAL pieces
of that fold's option-D pool. So `fine-tuned - features37 = +0.0357` compares
two arms with different amounts of supervision, and part of the gap is the pool
size rather than the encoder. This module refits features37 per fold on the SAME
pool the LoRA trained on, scores the SAME 180 eval test rows, pools the OOF, and
paired-bootstraps against both the fine-tuned OOF and the eval-only features37
OOF.

    cd model && uv run python -m claim_measurement.difficulty.matched_features37 \\
        --data-root /path/to/model/data \\
        --fold-emb-dir /path/to/phase1_lora/fold_embeddings \\
        --fold-plans /path/to/phase1-lora-bundle/fold_plans.json

Everything except the pool refit is imported from ft_eval so the three arms are
the same protocol by construction: the same ALPHAS, the same
StandardScaler+RidgeCV, the same `composer_disjoint_folds(composers, 5, 2026)`
folds, the same `features37_seg_ids` row order, and the same paired_boot.

The folds are DERIVED, never read from the plans file -- the per-fold adapters
are welded to the (n_folds=5, seed=2026) pair that produced their training
pools, so the plans on disk are only allowed to CONFIRM the derived folds
(check_fold_identity). Reading fold membership from the file instead would let a
stale plans file silently score fold f's test pieces through the adapter that
trained on them.
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
from claim_measurement.difficulty.bakeoff_paths import resolve_paths
from claim_measurement.difficulty.bakeoff_sampling import load_bakeoff_manifest
from claim_measurement.difficulty.features37_backbone import load_feature37_cache
from claim_measurement.difficulty.ft_eval import (
    ALPHAS,
    N_FOLDS,
    SEED,
    _load_features37,
    _ridge_oof,
    oof_tau_per_fold,
)
from claim_measurement.difficulty.train_fold import read_fold_embeddings

_PLAN_KEYS = ("fold", "test_seg_ids", "train_seg_ids", "val_seg_ids")


def load_fold_plans(path: Path, n_folds: int = N_FOLDS) -> list[dict]:
    """The option-D per-fold pools as written by push_train_dataset.py.

    Validated rather than trusted: every consumer indexes plans[f]
    positionally against composer_disjoint_folds(...)[f], so a short file, a
    reordered file, or a plan missing a split would pair fold f's pool with
    another fold's test rows.
    """
    plans = json.loads(Path(path).read_text())
    if not isinstance(plans, list) or len(plans) != n_folds:
        raise ValueError(
            f"{path}: expected a list of {n_folds} fold plans, got "
            f"{len(plans) if isinstance(plans, list) else type(plans).__name__}")
    for i, plan in enumerate(plans):
        missing = [k for k in _PLAN_KEYS if k not in plan]
        if missing:
            raise ValueError(f"{path}: plan {i} is missing {', '.join(missing)}")
        if plan["fold"] != i:
            raise ValueError(
                f"{path}: plans are not in fold order -- position {i} holds "
                f"fold {plan['fold']}")
    return plans


def load_feature37_grades(cache_path: Path) -> dict[str, int]:
    """{piece key -> grade} from the same cache load_feature37_cache reads.

    The pool rows' labels have to come from the cache because the pool is not
    in the eval .npz set; the eval rows' grades still come from the .npz files,
    and check_eval_cache_agrees asserts the two agree on the 900 shared rows.
    """
    rows = json.loads(Path(cache_path).read_text()).get("rows") or []
    if not rows:
        raise ValueError(f"no rows in {cache_path}")
    return {r["key"]: int(r["grade"]) for r in rows}


def check_fold_identity(plans, seg_ids, composers, n_folds: int,
                        seed: int) -> list[str]:
    """Every violation of "the plans on disk ARE the derived folds", as
    human-readable strings. Empty list == the plans came from this exact
    (n_folds, seed) pair, which is the only condition under which fold f's
    adapter and fold f's pool describe the same experiment."""
    violations = []
    derived = composer_disjoint_folds(np.asarray(composers), n_folds, seed)
    for f, plan in enumerate(plans):
        expected = {seg_ids[i] for i in derived[f]}
        if set(plan["test_seg_ids"]) != expected:
            violations.append(
                f"fold {f}: plan test_seg_ids do not equal "
                f"composer_disjoint_folds(composers, {n_folds}, {seed})[{f}] "
                f"({len(set(plan['test_seg_ids']) & expected)} of "
                f"{len(expected)} shared)")
    return violations


def fold_train_keys(plan: dict, seg_id_to_key: dict[str, str]) -> list[str]:
    """Piece keys for fold f's matched training pool = train + val.

    The LoRA saw both splits; val was early-stopping only. RidgeCV picks its
    own alpha by internal CV, so there is nothing here for a held-out split to
    do, and withholding val would hand features37 a smaller pool than the
    encoder got -- the opposite of a matched arm.

    A seg_id absent from the manifest join raises: it means that pool row has
    no feature vector, and dropping it silently would shrink the matched pool
    below the one the LoRA trained on, biasing the comparison toward
    features37 looking weaker than it is.
    """
    return [seg_id_to_key[s]
            for s in list(plan["train_seg_ids"]) + list(plan["val_seg_ids"])]


def check_eval_cache_agrees(Xf, y, seg_ids, seg_id_to_key, by_key,
                            grade_by_key, atol: float = 1e-5) -> list[str]:
    """Assert the pool's feature source and the eval .npz files describe the
    same 37 columns in the same order, on the 900 rows they share.

    They are order-identical by construction -- the eval .npz files were
    written by CachedFeature37Backbone over load_feature37_cache's column
    order. This checks it anyway, because the failure mode of a drift is not a
    crash: the pool would be fit on permuted columns and predict the eval rows
    through the wrong coefficients, producing a plausible-looking wrong tau-c.
    """
    violations = []
    for i, seg_id in enumerate(seg_ids):
        key = seg_id_to_key[seg_id]
        cached = np.asarray(by_key[key], dtype=float)
        npz_row = np.asarray(Xf[i], dtype=float)
        if cached.shape != npz_row.shape:
            violations.append(f"{seg_id}: cache has {cached.shape} features, "
                              f"the .npz has {npz_row.shape}")
        elif not np.allclose(cached, npz_row, atol=atol, equal_nan=True):
            violations.append(
                f"{seg_id}: cached feature vector differs from the .npz row "
                f"(max abs diff {np.nanmax(np.abs(cached - npz_row)):.6g}) -- "
                f"feature ORDER or VALUES have drifted")
        if int(grade_by_key[key]) != int(y[i]):
            violations.append(f"{seg_id}: cache grade {grade_by_key[key]} != "
                              f".npz grade {int(y[i])}")
    return violations


def _ridge_fit_predict(X_train, y_train, X_test):
    # Same estimator, same ALPHAS, same nan_to_num as ft_eval._ridge_oof and
    # ft_eval.oof_tau_per_fold. The only difference between this arm and the
    # eval-only features37 arm must be WHICH ROWS train it.
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
    model.fit(np.nan_to_num(X_train), y_train)
    return model.predict(np.nan_to_num(X_test))


def matched_oof(plans, Xf, y, seg_ids, composers, seg_id_to_key, by_key,
                grade_by_key, n_folds: int = N_FOLDS, seed: int = SEED,
                fit_predict=_ridge_fit_predict) -> np.ndarray:
    """Pooled OOF predictions for features37 refit on the LoRA's own pools.

    For fold f: train rows are fold f's option-D pool (train + val, ~3,800
    non-eval pieces) with features and grades from the feature cache; test rows
    are fold f of composer_disjoint_folds(composers, n_folds, seed) taken from
    Xf, the canonical eval matrix. Raises unless the plans confirm the derived
    folds.
    """
    violations = check_fold_identity(plans, seg_ids, composers, n_folds, seed)
    if violations:
        raise ValueError("fold plans do not match the derived folds:\n  "
                         + "\n  ".join(violations))

    oof = np.full(len(y), np.nan)
    for f, test_idx in enumerate(composer_disjoint_folds(np.asarray(composers),
                                                         n_folds, seed)):
        keys = fold_train_keys(plans[f], seg_id_to_key)
        X_train = np.stack([np.asarray(by_key[k], dtype=float) for k in keys])
        y_train = np.array([grade_by_key[k] for k in keys], dtype=float)
        oof[test_idx] = fit_predict(X_train, y_train, Xf[test_idx])
    return oof


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--fold-emb-dir", type=Path, required=True,
                    help="dir containing emb_fold0.npz .. emb_fold{N_FOLDS-1}.npz")
    ap.add_argument("--fold-plans", type=Path, required=True,
                    help="fold_plans.json from push_train_dataset.py -- the "
                         "option-D pools the LoRA actually trained on")
    args = ap.parse_args(argv)

    paths = resolve_paths(args.data_root)
    Xf, y, composers, seg_ids = _load_features37(paths.emb_root)

    entries = load_bakeoff_manifest(paths.manifest, paths.labels,
                                    paths.transkun_mid_dir)
    seg_id_to_key = {e.seg_id: e.key for e in entries}
    by_key = load_feature37_cache(paths.feature37_cache)
    grade_by_key = load_feature37_grades(paths.feature37_cache)

    plans = load_fold_plans(args.fold_plans)
    violations = check_fold_identity(plans, seg_ids, composers, N_FOLDS, SEED)
    violations += check_eval_cache_agrees(Xf, y, seg_ids, seg_id_to_key, by_key,
                                          grade_by_key)
    if violations:
        raise SystemExit("REFUSING TO SCORE -- inputs are not the ones gate (i) "
                         "measured:\n  " + "\n  ".join(violations[:20]))

    emb_by_fold = {}
    for f in range(N_FOLDS):
        fold_data = read_fold_embeddings(args.fold_emb_dir / f"emb_fold{f}.npz")
        if fold_data["seg_ids"] != seg_ids:
            raise SystemExit(
                f"emb_fold{f}.npz row order does not match features37's seg_id "
                f"order; the comparison would be unpaired")
        emb_by_fold[f] = fold_data["embeddings"]

    ft_oof = oof_tau_per_fold(emb_by_fold, y, composers, N_FOLDS, SEED)
    f37_oof = _ridge_oof(Xf, y, composers, N_FOLDS, SEED)
    matched = matched_oof(plans, Xf, y, seg_ids, composers, seg_id_to_key,
                          by_key, grade_by_key, N_FOLDS, SEED)

    pool_sizes = [len(p["train_seg_ids"]) + len(p["val_seg_ids"]) for p in plans]
    print(f"n={len(y)} eval pieces, {len(set(composers))} composers")
    print(f"matched pool per fold (train+val): {pool_sizes}")
    print(f"features37|ridge (eval-only, ~720 rows) tau-c {tau_c(f37_oof, y):.4f}")
    print(f"features37|ridge (matched pool)         tau-c {tau_c(matched, y):.4f}")
    print(f"moonbeam_ft_mean|ridge                  tau-c {tau_c(ft_oof, y):.4f}")

    for label, a, b in (
        ("matched-features37 - features37(eval-only)", f37_oof, matched),
        ("moonbeam_ft_mean   - features37(eval-only)", f37_oof, ft_oof),
        ("moonbeam_ft_mean   - matched-features37   ", matched, ft_oof),
    ):
        d, lo, hi, p = paired_boot(a, b, y, seed=SEED)
        print(f"{label}: {d:+.4f} CI95[{lo:+.4f},{hi:+.4f}] "
              f"P(diff<=0)={p:.3f} {'SIG' if lo > 0 else 'noise'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
