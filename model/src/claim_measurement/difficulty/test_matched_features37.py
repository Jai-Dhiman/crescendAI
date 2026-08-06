"""Tests for matched_features37 (#149) -- the supervision-matched features37
arm that decides whether gate (i)'s +0.0357 is an encoder win or a supervision
artifact.

Run: cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
"""
import json

import numpy as np
import pytest

from claim_measurement.difficulty.bakeoff_cv import composer_disjoint_folds, tau_c
from claim_measurement.difficulty.matched_features37 import (
    check_eval_cache_agrees,
    check_fold_identity,
    fold_train_keys,
    load_feature37_grades,
    load_fold_plans,
    matched_oof,
)


def _plan(fold, test, train, val):
    return {"fold": fold, "test_seg_ids": list(test),
            "train_seg_ids": list(train), "val_seg_ids": list(val)}


def _write_plans(path, plans):
    path.write_text(json.dumps(plans))
    return path


# --- load_fold_plans ------------------------------------------------------

def test_load_fold_plans_reads_the_five_option_d_plans(tmp_path):
    plans = [_plan(f, [f"t{f}"], [f"a{f}"], [f"v{f}"]) for f in range(5)]
    got = load_fold_plans(_write_plans(tmp_path / "p.json", plans), n_folds=5)
    assert [p["fold"] for p in got] == [0, 1, 2, 3, 4]


def test_load_fold_plans_rejects_a_wrong_fold_count(tmp_path):
    # A 4-plan file would silently drop a fifth of the eval set from the OOF
    # vector, leaving 180 NaNs that a downstream nan-tolerant tau would hide.
    plans = [_plan(f, [f"t{f}"], [f"a{f}"], [f"v{f}"]) for f in range(4)]
    with pytest.raises(ValueError, match="5"):
        load_fold_plans(_write_plans(tmp_path / "p.json", plans), n_folds=5)


def test_load_fold_plans_rejects_plans_out_of_fold_order(tmp_path):
    # Every consumer indexes plans[f] positionally against
    # composer_disjoint_folds(...)[f]; a reordered file pairs fold f's pool
    # with another fold's test rows, which is exactly the leak this arm must
    # not introduce.
    plans = [_plan(f, [f"t{f}"], [f"a{f}"], [f"v{f}"]) for f in [0, 2, 1, 3, 4]]
    with pytest.raises(ValueError, match="order"):
        load_fold_plans(_write_plans(tmp_path / "p.json", plans), n_folds=5)


def test_load_fold_plans_rejects_a_missing_key(tmp_path):
    plans = [_plan(f, [f"t{f}"], [f"a{f}"], [f"v{f}"]) for f in range(5)]
    del plans[3]["val_seg_ids"]
    with pytest.raises(ValueError, match="val_seg_ids"):
        load_fold_plans(_write_plans(tmp_path / "p.json", plans), n_folds=5)


# --- check_fold_identity --------------------------------------------------

def test_check_fold_identity_is_clean_when_the_plans_match_the_derived_folds():
    seg_ids = [f"p{i:03d}" for i in range(50)]
    composers = np.arange(50)
    folds = composer_disjoint_folds(composers, 5, 2026)
    plans = [_plan(f, [seg_ids[i] for i in folds[f]], [], []) for f in range(5)]
    assert check_fold_identity(plans, seg_ids, composers, 5, 2026) == []


def test_check_fold_identity_flags_a_seed_mismatch():
    # The load-bearing guard of the whole arm: the per-fold adapters are welded
    # to (n_folds=5, seed=2026). If the plans on disk came from any other
    # (n_folds, seed) pair, fold f's adapter trained on pieces this code would
    # then score as fold f's test set.
    seg_ids = [f"p{i:03d}" for i in range(50)]
    composers = np.arange(50)
    folds = composer_disjoint_folds(composers, 5, 1234)
    plans = [_plan(f, [seg_ids[i] for i in folds[f]], [], []) for f in range(5)]
    violations = check_fold_identity(plans, seg_ids, composers, 5, 2026)
    assert violations and any("fold 0" in v for v in violations)


def test_check_fold_identity_flags_a_single_swapped_piece():
    seg_ids = [f"p{i:03d}" for i in range(50)]
    composers = np.arange(50)
    folds = composer_disjoint_folds(composers, 5, 2026)
    plans = [_plan(f, [seg_ids[i] for i in folds[f]], [], []) for f in range(5)]
    plans[2]["test_seg_ids"][0] = "p999"
    violations = check_fold_identity(plans, seg_ids, composers, 5, 2026)
    assert violations and any("fold 2" in v for v in violations)


# --- fold_train_keys ------------------------------------------------------

def test_fold_train_keys_pools_train_and_val():
    # The LoRA saw both splits (val only for early stopping). RidgeCV selects
    # its own alpha by internal CV, so there is no held-out set to preserve
    # and withholding val would hand features37 a smaller pool than the
    # encoder got -- the opposite of a matched arm.
    plan = _plan(0, ["t0"], ["a", "b"], ["c"])
    seg_id_to_key = {s: f"KEY_{s}" for s in ["t0", "a", "b", "c"]}
    assert fold_train_keys(plan, seg_id_to_key) == ["KEY_a", "KEY_b", "KEY_c"]


def test_fold_train_keys_never_returns_an_eval_piece():
    plan = _plan(0, ["t0", "t1"], ["a"], ["c"])
    seg_id_to_key = {s: f"KEY_{s}" for s in ["t0", "t1", "a", "c"]}
    keys = fold_train_keys(plan, seg_id_to_key)
    assert not ({"KEY_t0", "KEY_t1"} & set(keys))


def test_fold_train_keys_raises_on_an_unmapped_seg_id():
    # A seg_id absent from the manifest join means the pool row has no feature
    # vector. Dropping it silently would shrink the matched pool below the one
    # the LoRA trained on and bias the comparison in features37's favour.
    plan = _plan(0, ["t0"], ["a", "ghost"], [])
    with pytest.raises(KeyError):
        fold_train_keys(plan, {"t0": "KEY_t0", "a": "KEY_a"})


# --- check_eval_cache_agrees ---------------------------------------------

def test_check_eval_cache_agrees_is_clean_on_consistent_inputs():
    seg_ids = ["p0", "p1"]
    Xf = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y = np.array([3.0, 7.0])
    seg_id_to_key = {"p0": "K0", "p1": "K1"}
    by_key = {"K0": Xf[0].copy(), "K1": Xf[1].copy()}
    grade_by_key = {"K0": 3, "K1": 7}
    assert check_eval_cache_agrees(Xf, y, seg_ids, seg_id_to_key, by_key,
                                   grade_by_key) == []


def test_check_eval_cache_agrees_flags_a_feature_order_drift():
    # If load_feature37_cache's column order ever diverged from the order
    # frozen into the eval .npz files, the pool would be fit on permuted
    # columns and predict the eval rows through the wrong coefficients -- a
    # silently wrong, plausible-looking tau-c.
    seg_ids = ["p0"]
    Xf = np.array([[1.0, 2.0]], dtype=np.float32)
    by_key = {"K0": np.array([2.0, 1.0], dtype=np.float32)}  # swapped columns
    violations = check_eval_cache_agrees(Xf, np.array([3.0]), seg_ids,
                                         {"p0": "K0"}, by_key, {"K0": 3})
    assert violations and any("p0" in v for v in violations)


def test_check_eval_cache_agrees_flags_a_grade_mismatch():
    seg_ids = ["p0"]
    Xf = np.array([[1.0, 2.0]], dtype=np.float32)
    by_key = {"K0": Xf[0].copy()}
    violations = check_eval_cache_agrees(Xf, np.array([3.0]), seg_ids,
                                         {"p0": "K0"}, by_key, {"K0": 9})
    assert violations and any("grade" in v for v in violations)


# --- matched_oof ----------------------------------------------------------

def _synthetic_matched_setup(n_eval=60, n_pool=200, n_features=4, seed=2026):
    """An eval set and a disjoint pool that share one linear signal in column
    0, so a ridge fit on the POOL can score the eval rows."""
    rng = np.random.default_rng(seed)
    seg_ids = [f"e{i:03d}" for i in range(n_eval)]
    composers = np.arange(n_eval)
    y = rng.integers(0, 11, size=n_eval).astype(float)
    Xf = np.column_stack([y, rng.normal(size=(n_eval, n_features - 1))]
                          ).astype(np.float32)

    pool_seg_ids = [f"q{i:04d}" for i in range(n_pool)]
    pool_y = rng.integers(0, 11, size=n_pool).astype(float)
    pool_X = np.column_stack([pool_y, rng.normal(size=(n_pool, n_features - 1))]
                              ).astype(np.float32)

    seg_id_to_key = {s: f"K_{s}" for s in seg_ids + pool_seg_ids}
    by_key = {f"K_{s}": pool_X[i] for i, s in enumerate(pool_seg_ids)}
    by_key.update({f"K_{s}": Xf[i] for i, s in enumerate(seg_ids)})
    grade_by_key = {f"K_{s}": int(pool_y[i]) for i, s in enumerate(pool_seg_ids)}
    grade_by_key.update({f"K_{s}": int(y[i]) for i, s in enumerate(seg_ids)})

    folds = composer_disjoint_folds(composers, 5, 2026)
    plans = []
    for f in range(5):
        test = [seg_ids[i] for i in folds[f]]
        cut = n_pool * 4 // 5
        plans.append(_plan(f, test, pool_seg_ids[:cut], pool_seg_ids[cut:]))
    return plans, Xf, y, seg_ids, composers, seg_id_to_key, by_key, grade_by_key


def test_matched_oof_recovers_a_pool_trained_linear_signal():
    args = _synthetic_matched_setup()
    oof = matched_oof(*args, n_folds=5, seed=2026)
    assert not np.isnan(oof).any()
    assert tau_c(oof, args[2]) > 0.9


def test_matched_oof_fits_only_on_pool_rows_never_on_eval_rows():
    """The entire point of the arm. A ridge whose train rows leaked eval rows
    would beat the honest matched arm and could make the fine-tune look worse
    than it is, so this asserts the recorded train matrix against the pool.
    """
    (plans, Xf, y, seg_ids, composers, seg_id_to_key, by_key,
     grade_by_key) = _synthetic_matched_setup()
    seen = []

    def recording_fit_predict(X_train, y_train, X_test):
        seen.append((np.asarray(X_train).copy(), np.asarray(y_train).copy()))
        return np.zeros(len(X_test))

    matched_oof(plans, Xf, y, seg_ids, composers, seg_id_to_key, by_key,
                grade_by_key, n_folds=5, seed=2026,
                fit_predict=recording_fit_predict)

    eval_rows = {tuple(np.round(row, 6)) for row in Xf}
    assert len(seen) == 5
    for f, (X_train, y_train) in enumerate(seen):
        assert len(X_train) == len(plans[f]["train_seg_ids"]) + len(
            plans[f]["val_seg_ids"])
        assert len(y_train) == len(X_train)
        train_rows = {tuple(np.round(row, 6)) for row in X_train}
        assert not (train_rows & eval_rows), f"fold {f} trained on an eval row"


def test_matched_oof_scores_each_fold_exactly_once():
    """Pooled OOF only pairs against the fine-tuned OOF if every eval row is
    predicted by exactly the fold that holds it."""
    (plans, Xf, y, seg_ids, composers, seg_id_to_key, by_key,
     grade_by_key) = _synthetic_matched_setup()
    calls = []

    def marking_fit_predict(X_train, y_train, X_test):
        calls.append(len(X_test))
        return np.full(len(X_test), float(len(calls)))

    oof = matched_oof(plans, Xf, y, seg_ids, composers, seg_id_to_key, by_key,
                      grade_by_key, n_folds=5, seed=2026,
                      fit_predict=marking_fit_predict)
    assert sum(calls) == len(y)
    assert not np.isnan(oof).any()
    assert sorted(np.unique(oof).tolist()) == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_matched_oof_raises_when_the_plans_are_not_the_derived_folds():
    (plans, Xf, y, seg_ids, composers, seg_id_to_key, by_key,
     grade_by_key) = _synthetic_matched_setup()
    plans[1]["test_seg_ids"] = list(plans[2]["test_seg_ids"])
    with pytest.raises(ValueError, match="fold"):
        matched_oof(plans, Xf, y, seg_ids, composers, seg_id_to_key, by_key,
                    grade_by_key, n_folds=5, seed=2026)


# --- load_feature37_grades ----------------------------------------------

def test_load_feature37_grades_reads_the_cache_grades(tmp_path):
    cache = tmp_path / "f.json"
    cache.write_text(json.dumps({"rows": [{"key": "A", "grade": 4, "composer": "X"},
                                           {"key": "B", "grade": 7, "composer": "Y"}]}))
    assert load_feature37_grades(cache) == {"A": 4, "B": 7}


def test_load_feature37_grades_raises_on_an_empty_cache(tmp_path):
    cache = tmp_path / "f.json"
    cache.write_text(json.dumps({"rows": []}))
    with pytest.raises(ValueError):
        load_feature37_grades(cache)
