"""#168 final-model retrain: paired comparison of two adapter arms trained on
DIFFERENT pools (5,798 pre-#169 vs 7,665 post-#169), scored on identical folds.

ft_eval.py compares one arm against features37; adapter_ensemble.py compares
ensembles against their own arm's diagonal. Neither compares two arms to each
other, which is the retrain's pre-registered criterion (#168 comment,
2026-08-23, posted before any retrain number existed):

- PRIMARY: fold-diagonal single-adapter OOF, retrain vs baseline (the 0.8395
  anchor). Accepted change = CI95 excluding zero.
- SECONDARY: pred-mean 5-adapter ensemble, retrain vs baseline (the 0.8445
  anchor), ridge refit per (fold, adapter) exactly as lever 2 registered.

The comparison is legitimate only because the seed-2026 fold TEST sets are
identical across the two pools: test membership is a function of the eval
composers and the seed, never of the training pool, so enlarging the pool
cannot move a test row. `load_arm` re-checks each arm's row order against
features37's seg_ids, so a silently permuted arm raises instead of scoring.

    cd model && uv run python -m claim_measurement.difficulty.retrain_compare \\
        --baseline-arm-dir data/results/phase1_lora/fold_embeddings \\
        --retrain-arm-dir data/results/phase1_lora/fold_embeddings_7665
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from claim_measurement.difficulty.adapter_ensemble import (
    prediction_mean_oof,
    single_adapter_oofs,
)
from claim_measurement.difficulty.bakeoff_cv import paired_boot, tau_c
from claim_measurement.difficulty.ft_eval import (
    N_FOLDS,
    SEED,
    _load_features37,
    oof_tau_per_fold,
)
from claim_measurement.difficulty.window_sweep import load_arm


def _report(label: str, base_oof: np.ndarray, new_oof: np.ndarray,
            y: np.ndarray, seed: int) -> None:
    d, lo, hi, p = paired_boot(base_oof, new_oof, y, seed=seed)
    verdict = "SIG" if (lo > 0 or hi < 0) else "noise"
    print(f"{label:<22} baseline {tau_c(base_oof, y):.4f}  "
          f"retrain {tau_c(new_oof, y):.4f}  "
          f"delta {d:+.4f} CI95[{lo:+.4f},{hi:+.4f}] "
          f"P(diff<=0)={p:.3f} {verdict}")


def compare(baseline_arm_dir: Path, retrain_arm_dir: Path,
            data_root: Path | None, n_folds: int = N_FOLDS,
            seed: int = SEED) -> int:
    from claim_measurement.difficulty.bakeoff_paths import resolve_paths

    emb_root = resolve_paths(data_root).emb_root
    _, y, composers, seg_ids = _load_features37(emb_root)
    base = load_arm(Path(baseline_arm_dir), seg_ids, n_folds)
    new = load_arm(Path(retrain_arm_dir), seg_ids, n_folds)

    print(f"n={len(y)} pieces, {len(set(composers))} composers, "
          f"seed {seed}, {n_folds} folds")
    _report("diagonal (PRIMARY)",
            oof_tau_per_fold(base, y, composers, n_folds, seed),
            oof_tau_per_fold(new, y, composers, n_folds, seed), y, seed)
    _report("pred-mean (SECONDARY)",
            prediction_mean_oof(
                single_adapter_oofs(base, y, composers, n_folds, seed)),
            prediction_mean_oof(
                single_adapter_oofs(new, y, composers, n_folds, seed)),
            y, seed)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline-arm-dir", type=Path, required=True,
                    help="dir of emb_fold{0..4}.npz from the 5,798-pool run")
    ap.add_argument("--retrain-arm-dir", type=Path, required=True,
                    help="dir of emb_fold{0..4}.npz from the 7,665-pool run")
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--n-folds", type=int, default=N_FOLDS)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)
    return compare(args.baseline_arm_dir, args.retrain_arm_dir,
                   args.data_root, args.n_folds, args.seed)


if __name__ == "__main__":
    sys.exit(main())
