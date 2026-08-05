"""Tests for fold_plan (#149 / #138 Phase 1) -- the option-D per-fold training
set construction + leakage invariants.

Run: cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
"""
import pytest

from claim_measurement.difficulty.bakeoff_sampling import ManifestEntry
from claim_measurement.difficulty.fold_plan import (
    FoldPlan, build_fold_plans, check_fold_plans,
)


def _entries(n_composers: int, pieces_per_composer: int, prefix: str) -> list[ManifestEntry]:
    return [
        ManifestEntry(
            seg_id=f"{prefix}c{c}_p{p}", key=f"{prefix}c{c}_p{p}.mid",
            grade=p % 11, composer=f"composer_{c}",
        )
        for c in range(n_composers) for p in range(pieces_per_composer)
    ]


def test_build_fold_plans_excludes_eval_pieces_and_test_fold_composers_from_train():
    eval_entries = _entries(n_composers=20, pieces_per_composer=1, prefix="eval_")
    pool_entries = eval_entries + _entries(
        n_composers=20, pieces_per_composer=5, prefix="pool_")

    plans = build_fold_plans(eval_entries, pool_entries, n_folds=5, seed=2026, val_frac=0.12)

    assert len(plans) == 5
    eval_seg_ids = {e.seg_id for e in eval_entries}
    pool_composer_of = {e.seg_id: e.composer for e in pool_entries}
    for plan in plans:
        train_and_val = set(plan.train_seg_ids) | set(plan.val_seg_ids)
        assert not (train_and_val & eval_seg_ids), "an eval piece leaked into train/val"
        test_composers = {e.composer for e in eval_entries if e.seg_id in plan.test_seg_ids}
        train_composers = {pool_composer_of[s] for s in plan.train_seg_ids}
        assert not (test_composers & train_composers), "a test composer leaked into train"


def test_val_carve_is_composer_disjoint_from_train_and_near_target_fraction():
    eval_entries = _entries(n_composers=5, pieces_per_composer=1, prefix="eval_")
    pool_entries = eval_entries + _entries(
        n_composers=100, pieces_per_composer=4, prefix="pool_")

    plans = build_fold_plans(eval_entries, pool_entries, n_folds=5, seed=2026, val_frac=0.12)

    pool_composer_of = {e.seg_id: e.composer for e in pool_entries}
    for plan in plans:
        train_composers = {pool_composer_of[s] for s in plan.train_seg_ids}
        val_composers = {pool_composer_of[s] for s in plan.val_seg_ids}
        assert not (train_composers & val_composers)
        total = len(plan.train_seg_ids) + len(plan.val_seg_ids)
        frac = len(plan.val_seg_ids) / total
        assert 0.05 < frac < 0.20


def test_check_fold_plans_flags_a_composer_that_straddles_test_and_train():
    eval_entries = _entries(n_composers=5, pieces_per_composer=1, prefix="eval_")
    pool_entries = eval_entries + _entries(
        n_composers=5, pieces_per_composer=4, prefix="pool_")
    plans = build_fold_plans(eval_entries, pool_entries, n_folds=5, seed=2026, val_frac=0.12)

    fold0 = plans[0]
    test_composer = next(e.composer for e in eval_entries if e.seg_id in fold0.test_seg_ids)
    leaking_seg_id = next(e.seg_id for e in pool_entries if e.composer == test_composer)
    tampered = FoldPlan(
        fold=0, test_seg_ids=fold0.test_seg_ids,
        train_seg_ids=fold0.train_seg_ids + (leaking_seg_id,),
        val_seg_ids=fold0.val_seg_ids,
    )
    tampered_plans = [tampered if p.fold == 0 else p for p in plans]

    violations = check_fold_plans(
        tampered_plans, eval_entries, pool_entries, n_folds=5, seed=2026)

    assert any("fold 0" in v for v in violations)
