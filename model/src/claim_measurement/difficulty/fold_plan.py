"""Option-D per-fold training-set construction + leakage invariants for #138
Phase 1 (LoRA fine-tune of MoonBeam). This module IS the entire leakage
argument -- get build_fold_plans and check_fold_plans right and the fine-tune
cannot see an eval piece or an eval-fold composer during training.

Composer-disjointness is a PER-FOLD constraint (see the design spec): fold f's
train pool excludes composers that appear in fold f's test set, not composers
appearing in ANY fold's test set. A set of per-fold adapters is therefore
welded to the (n_folds, seed) pair that produced them -- a different seed's
test fold can contain composers these adapters trained on.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from claim_measurement.difficulty.bakeoff_cv import composer_disjoint_folds


@dataclass(frozen=True)
class FoldPlan:
    fold: int
    test_seg_ids: tuple
    train_seg_ids: tuple
    val_seg_ids: tuple


def build_fold_plans(eval_entries, pool_entries, n_folds: int, seed: int,
                      val_frac: float) -> list:
    """eval_entries: the 900-piece eval sample. pool_entries: the full eligible
    pool (superset of eval_entries). For fold f: test = fold f of
    composer_disjoint_folds(eval composers, n_folds, seed); train pool =
    pool_entries minus every eval piece and minus every piece whose composer
    appears in fold f's test set (option D); val is a composer-disjoint
    ~val_frac slice carved out of that train pool for early stopping."""
    eval_composers = np.array([e.composer for e in eval_entries])
    eval_seg_ids = [e.seg_id for e in eval_entries]
    eval_seg_id_set = set(eval_seg_ids)
    test_folds = composer_disjoint_folds(eval_composers, n_folds, seed)

    plans = []
    for f, test_idx in enumerate(test_folds):
        test_composers = set(eval_composers[test_idx])
        test_seg_ids = tuple(eval_seg_ids[i] for i in test_idx)
        train_pool = [e for e in pool_entries
                      if e.seg_id not in eval_seg_id_set
                      and e.composer not in test_composers]
        train_seg_ids, val_seg_ids = _carve_val(
            train_pool, val_frac, seed=seed * 100 + f)
        plans.append(FoldPlan(fold=f, test_seg_ids=test_seg_ids,
                               train_seg_ids=train_seg_ids, val_seg_ids=val_seg_ids))
    return plans


def _carve_val(train_pool, val_frac: float, seed: int):
    """Deterministically carve a ~val_frac slice of train_pool into val, whole
    composers only, so val is composer-disjoint from the remaining train."""
    by_composer: dict = {}
    for e in train_pool:
        by_composer.setdefault(e.composer, []).append(e)
    composers = sorted(by_composer)
    order = np.random.default_rng(seed).permutation(len(composers))
    target = int(round(val_frac * len(train_pool)))

    val_composers, val_count = set(), 0
    for idx in order:
        if val_count >= target:
            break
        c = composers[idx]
        val_composers.add(c)
        val_count += len(by_composer[c])

    train_seg_ids, val_seg_ids = [], []
    for e in train_pool:
        (val_seg_ids if e.composer in val_composers else train_seg_ids).append(e.seg_id)
    return tuple(train_seg_ids), tuple(val_seg_ids)
