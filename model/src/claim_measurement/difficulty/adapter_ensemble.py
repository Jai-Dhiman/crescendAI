"""#168 S3 lever 2: adapter ensemble at inference.

The five fold adapters are five fine-tunes of the same backbone that differ
only in their training pools; single-adapter eval (the 0.8395 anchor) uses
adapter f for fold f only. This scores whether combining all five at
inference beats that, reusing window_sweep's w1024 CPU arm (every adapter x
all 900 eval pieces already extracted) -- no retraining, no new extraction.

Pre-registered design (issue #168 comment, 2026-08-15, BEFORE any number was
computed):

- PRIMARY: prediction-mean, refit-clean. For fold f and adapter g, fit
  StandardScaler+RidgeCV on fold-f TRAIN rows of adapter-g embeddings,
  predict fold-f TEST rows; the ensemble OOF is the mean over g. The stored
  per-fold heads are deliberately NOT reused: head g (g != f) was fit on
  rows that include fold-f's test pieces, so reusing it would be ridge-level
  train-on-test.
- SECONDARY (exploratory, not the accept criterion): embedding-mean --
  average the five adapters' embedding matrices into one X, standard ridge
  OOF.
- Verdict: paired_boot (seed 2026) vs the single-adapter w1024 baseline;
  accepted gain = CI95 excluding zero. A noise result is a finding.
- Variance estimate: each adapter g scored alone across ALL folds (5 tau-c
  values). Adapter g's fold-g block coincides with the baseline's by
  construction.
- Contamination stance (recorded, accepted): piece-level clean everywhere
  (option D excluded all 900 eval pieces from every training pool);
  composer-level clean only on the (f, f) diagonal -- adapter g != f may
  have trained on non-eval pieces by a fold-f test composer.

    cd model && uv run python -m claim_measurement.difficulty.adapter_ensemble \\
        --data-root data \\
        --arm-dir data/results/phase1_lora/window_sweep/w1024
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from claim_measurement.difficulty.bakeoff_cv import paired_boot, tau_c
from claim_measurement.difficulty.ft_eval import (
    N_FOLDS,
    SEED,
    _load_features37,
    _ridge_oof,
    oof_tau_per_fold,
)
from claim_measurement.difficulty.window_sweep import load_arm


def single_adapter_oofs(emb_by_adapter: dict, y: np.ndarray,
                        composers: np.ndarray, n_folds: int = N_FOLDS,
                        seed: int = SEED) -> dict:
    """One clean OOF vector per adapter: adapter g's embeddings used for
    EVERY fold, ridge refit per fold on that fold's train rows only."""
    return {g: _ridge_oof(emb_by_adapter[g], y, composers, n_folds, seed)
            for g in sorted(emb_by_adapter)}


def prediction_mean_oof(single_oofs: dict) -> np.ndarray:
    """Primary ensemble: fold f's prediction = the mean of the adapters'
    fold-f OOF predictions. Each single OOF's fold-f block comes from a
    ridge fit only on fold-f train rows, so the mean stays OOF-clean."""
    return np.mean(np.stack([single_oofs[g] for g in sorted(single_oofs)]),
                   axis=0)


def embedding_mean_oof(emb_by_adapter: dict, y: np.ndarray,
                       composers: np.ndarray, n_folds: int = N_FOLDS,
                       seed: int = SEED) -> np.ndarray:
    """Secondary ensemble: one X = the element-wise mean of the adapters'
    embedding matrices, then the standard ridge OOF. Feature-space averaging
    across adapters is a DIFFERENT experiment from prediction averaging;
    reported alongside, never the accept criterion."""
    X = np.mean(np.stack([emb_by_adapter[g] for g in sorted(emb_by_adapter)]),
                axis=0)
    return _ridge_oof(X, y, composers, n_folds, seed)


def _report(name: str, oof: np.ndarray, base_oof: np.ndarray,
            y: np.ndarray, seed: int) -> None:
    d, lo, hi, p = paired_boot(base_oof, oof, y, seed=seed)
    verdict = "SIG" if (lo > 0 or hi < 0) else "noise"
    print(f"{name:<20} tau-c {tau_c(oof, y):.4f}  vs baseline: {d:+.4f} "
          f"CI95[{lo:+.4f},{hi:+.4f}] P(diff<=0)={p:.3f} {verdict}")


def score(arm_dir: Path, data_root: Path | None, n_folds: int = N_FOLDS,
          seed: int = SEED) -> int:
    from claim_measurement.difficulty.bakeoff_paths import resolve_paths

    emb_root = resolve_paths(data_root).emb_root
    _, y, composers, seg_ids = _load_features37(emb_root)
    emb_by_adapter = load_arm(Path(arm_dir), seg_ids, n_folds)

    base_oof = oof_tau_per_fold(emb_by_adapter, y, composers, n_folds, seed)
    print(f"n={len(y)} pieces, {len(set(composers))} composers, "
          f"seed {seed}, {n_folds} folds")
    print(f"baseline (adapter f on fold f)  tau-c {tau_c(base_oof, y):.4f}")

    singles = single_adapter_oofs(emb_by_adapter, y, composers, n_folds, seed)
    taus = {g: tau_c(oof, y) for g, oof in singles.items()}
    for g, t in taus.items():
        print(f"adapter {g} alone (all folds)   tau-c {t:.4f}")
    values = np.array(list(taus.values()))
    print(f"single-adapter spread: min {values.min():.4f} "
          f"max {values.max():.4f} range {values.max() - values.min():.4f} "
          f"std {values.std(ddof=1):.4f}")

    _report("pred-mean (PRIMARY)", prediction_mean_oof(singles),
            base_oof, y, seed)
    _report("emb-mean (secondary)",
            embedding_mean_oof(emb_by_adapter, y, composers, n_folds, seed),
            base_oof, y, seed)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm-dir", type=Path, required=True,
                    help="w1024 arm dir: emb_fold{F}.npz = adapter F's "
                         "embeddings for ALL eval pieces")
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--n-folds", type=int, default=N_FOLDS)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)
    return score(args.arm_dir, args.data_root, args.n_folds, args.seed)


if __name__ == "__main__":
    sys.exit(main())
