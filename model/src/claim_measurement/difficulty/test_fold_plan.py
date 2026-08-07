"""Tests for fold_plan (#149 / #138 Phase 1) -- the option-D per-fold training
set construction + leakage invariants.

Run: cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
"""
import pytest

from claim_measurement.difficulty.bakeoff_sampling import ManifestEntry
from claim_measurement.difficulty.fold_plan import (
    ALL_DATA_FOLD,
    FoldPlan,
    build_all_data_plan,
    build_fold_plans,
    check_all_data_plan,
    check_fold_plans,
)


def _entries(
    n_composers: int, pieces_per_composer: int, prefix: str
) -> list[ManifestEntry]:
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

    plans = build_fold_plans(
        eval_entries, pool_entries, n_folds=5, seed=2026, val_frac=0.12)

    assert len(plans) == 5
    eval_seg_ids = {e.seg_id for e in eval_entries}
    pool_composer_of = {e.seg_id: e.composer for e in pool_entries}
    for plan in plans:
        train_and_val = set(plan.train_seg_ids) | set(plan.val_seg_ids)
        assert not (train_and_val & eval_seg_ids), "an eval piece leaked into train/val"
        test_composers = {
            e.composer for e in eval_entries if e.seg_id in plan.test_seg_ids}
        train_composers = {pool_composer_of[s] for s in plan.train_seg_ids}
        assert not (
            test_composers & train_composers), "a test composer leaked into train"


def test_val_carve_is_composer_disjoint_from_train_and_near_target_fraction():
    eval_entries = _entries(n_composers=5, pieces_per_composer=1, prefix="eval_")
    pool_entries = eval_entries + _entries(
        n_composers=100, pieces_per_composer=4, prefix="pool_")

    plans = build_fold_plans(
        eval_entries, pool_entries, n_folds=5, seed=2026, val_frac=0.12)

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
    plans = build_fold_plans(
        eval_entries, pool_entries, n_folds=5, seed=2026, val_frac=0.12)

    fold0 = plans[0]
    test_composer = next(
        e.composer for e in eval_entries if e.seg_id in fold0.test_seg_ids)
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


def test_check_fold_plans_returns_empty_for_plans_build_fold_plans_produced():
    eval_entries = _entries(n_composers=10, pieces_per_composer=1, prefix="eval_")
    pool_entries = eval_entries + _entries(
        n_composers=30, pieces_per_composer=4, prefix="pool_")
    plans = build_fold_plans(
        eval_entries, pool_entries, n_folds=5, seed=2026, val_frac=0.12)

    violations = check_fold_plans(
        plans, eval_entries, pool_entries, n_folds=5, seed=2026)

    assert violations == []


# --------------------------------------------------------------------------
# The submission plan (#166 / #104 S1): the inverse of a fold. Trains on every
# eligible piece because the MIREX test set is disjoint from PSyllabus by
# construction -- fold 0 sees 3,815 pieces, this sees all ~5,798.
# --------------------------------------------------------------------------


def test_the_submission_plan_trains_on_every_pool_piece_and_holds_nothing_out():
    """The whole reason this plan exists. If coverage is ever less than the
    full pool, 'trained on all 5,798 pieces' becomes a false statement in the
    technical report, which MIREX 2026 requires us to make."""
    pool = _entries(n_composers=20, pieces_per_composer=5, prefix="pool_")

    plan = build_all_data_plan(pool)

    assert plan.fold == ALL_DATA_FOLD
    assert plan.test_seg_ids == ()
    assert plan.val_seg_ids == ()
    assert set(plan.train_seg_ids) == {e.seg_id for e in pool}
    assert len(plan.train_seg_ids) == len(pool)
    assert check_all_data_plan(plan, pool) == []


def test_a_validation_slice_still_covers_the_pool_between_train_and_val():
    """val_frac defaults to 0 because withholding pieces defeats the point,
    but a diagnostic run may want one -- and it must move pieces from train to
    val, never drop them."""
    pool = _entries(n_composers=20, pieces_per_composer=5, prefix="pool_")

    plan = build_all_data_plan(pool, val_frac=0.2)

    assert plan.val_seg_ids, "val_frac=0.2 should carve a non-empty val slice"
    assert not (set(plan.train_seg_ids) & set(plan.val_seg_ids))
    assert (set(plan.train_seg_ids) | set(plan.val_seg_ids)) == {e.seg_id for e in pool}
    assert check_all_data_plan(plan, pool) == []


def test_check_catches_a_plan_that_silently_dropped_pieces():
    """The failure this guards against is silent: a plan missing 200 pieces
    trains fine, produces an adapter, and reads as a success."""
    pool = _entries(n_composers=20, pieces_per_composer=5, prefix="pool_")
    plan = build_all_data_plan(pool)
    truncated = FoldPlan(fold=ALL_DATA_FOLD, test_seg_ids=(),
                          train_seg_ids=plan.train_seg_ids[:-7], val_seg_ids=())

    violations = check_all_data_plan(truncated, pool)

    assert len(violations) == 1
    assert "7 pool piece(s) are in neither train nor val" in violations[0]


def test_check_rejects_a_plan_that_holds_anything_out():
    pool = _entries(n_composers=20, pieces_per_composer=5, prefix="pool_")
    plan = build_all_data_plan(pool)
    with_test = FoldPlan(fold=ALL_DATA_FOLD, test_seg_ids=(pool[0].seg_id,),
                          train_seg_ids=plan.train_seg_ids, val_seg_ids=())

    violations = check_all_data_plan(with_test, pool)

    assert any("must hold nothing out" in v for v in violations)


def test_check_rejects_a_cv_fold_plan_passed_in_by_mistake():
    """build_fold_plans and build_all_data_plan return the same type, so the
    only thing stopping a fold-0 plan from being staged as the submission plan
    is this check."""
    eval_entries = _entries(n_composers=20, pieces_per_composer=1, prefix="eval_")
    pool = eval_entries + _entries(
        n_composers=20, pieces_per_composer=5, prefix="pool_")
    fold_zero = build_fold_plans(eval_entries, pool, 5, 2026, 0.12)[0]

    violations = check_all_data_plan(fold_zero, pool)

    assert any("expected 99" in v for v in violations)
    assert any("must hold nothing out" in v for v in violations)


def test_an_out_of_range_val_frac_is_refused():
    pool = _entries(n_composers=4, pieces_per_composer=2, prefix="pool_")

    with pytest.raises(ValueError, match="val_frac"):
        build_all_data_plan(pool, val_frac=1.0)
