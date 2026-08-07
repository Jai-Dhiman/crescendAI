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


# The submission plan's `fold` index. Deliberately not 5: that would read as a
# sixth CV fold, and this plan is the opposite of a fold -- it holds nothing
# out. train_fold.py selects a plan by `--fold`, so it needs some integer.
ALL_DATA_FOLD = 99


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


def build_all_data_plan(pool_entries, val_frac: float = 0.0,
                        seed: int = 2026) -> FoldPlan:
    """The SUBMISSION plan: train on every eligible piece, hold nothing out.

    This is the inverse of `build_fold_plans` and the reason is in #104: the
    per-fold composer exclusions exist only to keep OUR cross-validation
    honest, and the MIREX test set is disjoint from PSyllabus by construction.
    Fold 0 trains on 3,815 pieces; this trains on all ~5,798. That is the
    largest free lever in the campaign and it falls out of work that has to
    happen anyway.

    `val_frac` defaults to 0.0 -- carving a validation slice would defeat the
    entire point by withholding pieces from the model we ship. Nothing depends
    on it: `train_fold.py` runs a fixed number of epochs with no early stopping,
    and its per-step `loss` line is the divergence signal. A non-zero value is
    available for a diagnostic run.

    **A model trained from this plan can never be evaluated.** Every piece we
    have a label for is in its training set, so any tau-c measured on it is
    train-on-test -- the exact contamination #135's 0.824 anchor died of.
    Validate the recipe on folds; deploy it once.
    """
    if not 0.0 <= val_frac < 1.0:
        raise ValueError(f"val_frac must be in [0, 1), got {val_frac}")
    if not pool_entries:
        raise ValueError("pool_entries is empty")

    if val_frac == 0.0:
        train_seg_ids = tuple(sorted(e.seg_id for e in pool_entries))
        val_seg_ids: tuple = ()
    else:
        train_seg_ids, val_seg_ids = _carve_val(pool_entries, val_frac, seed=seed)
    return FoldPlan(fold=ALL_DATA_FOLD, test_seg_ids=(),
                     train_seg_ids=train_seg_ids, val_seg_ids=val_seg_ids)


def check_all_data_plan(plan: FoldPlan, pool_entries) -> list:
    """Return every violation of the submission plan's invariants, as
    human-readable strings. Empty list == clean.

    The load-bearing one is COVERAGE: train + val must be exactly the pool,
    with nothing dropped and nothing duplicated. "We trained on all 5,798
    pieces" is a claim that goes in the technical report, so it is asserted
    here rather than assumed from the absence of an exclusion step.
    """
    violations: list = []
    pool_seg_ids = {e.seg_id for e in pool_entries}
    train_set = set(plan.train_seg_ids)
    val_set = set(plan.val_seg_ids)

    if plan.fold != ALL_DATA_FOLD:
        violations.append(f"fold is {plan.fold}, expected {ALL_DATA_FOLD}")
    if plan.test_seg_ids:
        violations.append(
            f"the submission plan holds {len(plan.test_seg_ids)} pieces out; it "
            f"must hold nothing out")
    if train_set & val_set:
        violations.append("train/val seg_id overlap")
    if len(train_set) != len(plan.train_seg_ids):
        violations.append("train_seg_ids contains duplicates")

    covered = train_set | val_set
    missing = pool_seg_ids - covered
    extra = covered - pool_seg_ids
    if missing:
        violations.append(
            f"{len(missing)} pool piece(s) are in neither train nor val, so this "
            f"is not an all-data plan: {sorted(missing)[:5]}")
    if extra:
        violations.append(
            f"{len(extra)} piece(s) are not in the pool: {sorted(extra)[:5]}")
    return violations


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


def check_fold_plans(
    plans, eval_entries, pool_entries, n_folds: int, seed: int
) -> list:
    """Re-derive the expected test folds and return every leakage/consistency
    violation found, as human-readable strings. Empty list == clean."""
    violations: list = []
    eval_composers = np.array([e.composer for e in eval_entries])
    eval_seg_ids = [e.seg_id for e in eval_entries]
    eval_seg_id_set = set(eval_seg_ids)
    composer_of = {e.seg_id: e.composer for e in pool_entries}
    composer_of.update({e.seg_id: e.composer for e in eval_entries})
    expected_test_folds = composer_disjoint_folds(eval_composers, n_folds, seed)

    if len(plans) != n_folds:
        violations.append(f"expected {n_folds} plans, got {len(plans)}")

    for plan in plans:
        expected_test = {eval_seg_ids[i] for i in expected_test_folds[plan.fold]}
        if set(plan.test_seg_ids) != expected_test:
            violations.append(
                f"fold {plan.fold}: test_seg_ids do not equal "
                f"composer_disjoint_folds(eval composers, {n_folds}, "
                f"{seed})[{plan.fold}]")

        train_set = set(plan.train_seg_ids)
        val_set = set(plan.val_seg_ids)
        test_set = set(plan.test_seg_ids)
        if train_set & test_set:
            violations.append(f"fold {plan.fold}: train/test seg_id overlap")
        if val_set & test_set:
            violations.append(f"fold {plan.fold}: val/test seg_id overlap")
        if train_set & val_set:
            violations.append(f"fold {plan.fold}: train/val seg_id overlap")
        if (train_set | val_set) & eval_seg_id_set:
            violations.append(
                f"fold {plan.fold}: an eval piece leaked into train or val")

        test_composers = {composer_of[s] for s in plan.test_seg_ids}
        train_composers = {composer_of[s] for s in plan.train_seg_ids}
        val_composers = {composer_of[s] for s in plan.val_seg_ids}
        if test_composers & train_composers:
            violations.append(f"fold {plan.fold}: a test composer appears in train")
        if val_composers & train_composers:
            violations.append(f"fold {plan.fold}: a val composer appears in train")

    return violations
