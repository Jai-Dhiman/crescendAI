"""Offline tests for the #167 shift evaluation. The properties under test are
about which rows train, which rows are scored, and what the harness refuses --
not about the encoder, so the embeddings are synthetic."""
from __future__ import annotations

import numpy as np
import pytest

from claim_measurement.difficulty.bakeoff_cv import composer_disjoint_folds
from claim_measurement.difficulty.distribution_shift import (
    HUMAN_ARM,
    N_FOLDS,
    SEED,
    embed_arm,
    read_arm_npz,
    report,
    shift_oof,
    stack_arm,
    write_arm_npz,
)

N, DIM = 120, 8


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 11, size=N)
    composers = np.arange(N)
    human = {f: rng.normal(size=(N, DIM)) for f in range(N_FOLDS)}
    arm = {f: rng.normal(size=(N, DIM)) for f in range(N_FOLDS)}
    return human, arm, y, composers


# --------------------------------------------------------------------------
# shift_oof -- the asymmetric protocol
# --------------------------------------------------------------------------


def test_every_row_is_scored_exactly_once(toy):
    human, arm, y, composers = toy
    assert np.all(np.isfinite(shift_oof(human, arm, y, composers)))


def test_the_head_trains_on_human_rows_and_predicts_arm_rows(toy, monkeypatch):
    """The whole protocol in one assertion: MIREX trains us on real recordings
    and tests us on synthesis, so the fit and the prediction must come from
    different populations."""
    human, arm, y, composers = toy
    for f in range(N_FOLDS):
        human[f][:, 0] = 100.0    # tag: human rows
        arm[f][:, 0] = -100.0     # tag: arm rows
    seen = []

    def spy(X_train, y_train, X_test):
        seen.append((set(X_train[:, 0].tolist()), set(X_test[:, 0].tolist())))
        return np.zeros(len(X_test))

    monkeypatch.setattr(
        "claim_measurement.difficulty.distribution_shift._fit_predict", spy)
    shift_oof(human, arm, y, composers)

    assert seen
    for train_tags, test_tags in seen:
        assert train_tags == {100.0}
        assert test_tags == {-100.0}


def test_train_and_test_rows_never_overlap(toy, monkeypatch):
    human, arm, y, composers = toy
    captured = []

    def spy(X_train, y_train, X_test):
        captured.append(set(y_train.tolist()))
        return np.zeros(len(X_test))

    monkeypatch.setattr(
        "claim_measurement.difficulty.distribution_shift._fit_predict", spy)
    shift_oof(human, arm, np.arange(N), composers)

    folds = composer_disjoint_folds(composers, N_FOLDS, SEED)
    for train_rows, test_idx in zip(captured, folds):
        assert not (train_rows & set(np.asarray(test_idx).tolist()))


def test_passing_human_as_the_arm_reproduces_ft_evals_baseline(toy):
    """The anchor and the arms must come off the same code path, or the delta
    measures the code rather than the shift."""
    from claim_measurement.difficulty.ft_eval import oof_tau_per_fold

    human, _arm, y, composers = toy
    np.testing.assert_allclose(
        shift_oof(human, human, y, composers),
        oof_tau_per_fold(human, y, composers, N_FOLDS, SEED))


def test_each_fold_uses_its_own_adapters_embeddings(toy, monkeypatch):
    human, arm, y, composers = toy
    for f in range(N_FOLDS):
        human[f][:, 1] = f
        arm[f][:, 1] = f
    seen = []

    def spy(X_train, y_train, X_test):
        seen.append((set(X_train[:, 1].tolist()), set(X_test[:, 1].tolist())))
        return np.zeros(len(X_test))

    monkeypatch.setattr(
        "claim_measurement.difficulty.distribution_shift._fit_predict", spy)
    shift_oof(human, arm, y, composers)
    for f, (train_tags, test_tags) in enumerate(seen):
        assert train_tags == {f} and test_tags == {f}


def test_shift_oof_rejects_a_missing_arm_fold(toy):
    human, arm, y, composers = toy
    del arm[2]
    with pytest.raises(KeyError, match="arm embeddings are missing fold 2"):
        shift_oof(human, arm, y, composers)


def test_shift_oof_rejects_mismatched_shapes(toy):
    human, arm, y, composers = toy
    arm[0] = arm[0][:-1]
    with pytest.raises(ValueError, match="unpaired"):
        shift_oof(human, arm, y, composers)


# --------------------------------------------------------------------------
# stack_arm -- what the harness refuses
# --------------------------------------------------------------------------


def test_stack_arm_returns_rows_in_the_canonical_order():
    vectors = {"b": np.arange(DIM) + 1.0, "a": np.zeros(DIM)}
    out = stack_arm(vectors, ["a", "b"], DIM)
    assert out.shape == (2, DIM)
    assert out[0].tolist() == [0.0] * DIM


def test_stack_arm_refuses_an_arm_with_holes():
    """A short arm still produces a tau-c, over a different population than the
    one it is compared against."""
    with pytest.raises(ValueError, match="missing 1 of 2"):
        stack_arm({"a": np.zeros(DIM)}, ["a", "b"], DIM)


def test_stack_arm_refuses_a_non_finite_vector():
    """Not nan_to_num'd, for the same reason score_wav refuses to: a non-finite
    embedding means the backbone produced garbage, and zeroing it emits a
    confident score from a broken forward pass."""
    bad = np.zeros(DIM)
    bad[3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        stack_arm({"a": np.zeros(DIM), "b": bad}, ["a", "b"], DIM)


def test_stack_arm_refuses_a_dimension_mismatch():
    with pytest.raises(ValueError, match="dim"):
        stack_arm({"a": np.zeros(DIM + 1)}, ["a"], DIM)


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def test_report_scores_every_arm_on_the_identical_rows(toy):
    human, arm, y, composers = toy
    out = report({HUMAN_ARM: human, "expressive": arm}, y, composers,
                 log=lambda *a: None)
    assert out["n"] == N
    assert set(out["tau_c"]) == {HUMAN_ARM, "expressive"}


def test_report_names_what_each_delta_is_allowed_to_mean(toy):
    human, arm, y, composers = toy
    out = report({HUMAN_ARM: human, "source": arm, "expressive": arm,
                  "deadpan": arm}, y, composers, log=lambda *a: None)
    assert "#170 can address this" in out["deltas"]["expressive-source"]["meaning"]
    assert "#170 cannot" in out["deltas"]["deadpan-expressive"]["meaning"]
    assert "confound" in out["deltas"]["source-human"]["meaning"]


def test_report_omits_deltas_for_arms_that_were_not_run(toy):
    human, arm, y, composers = toy
    out = report({HUMAN_ARM: human, "expressive": arm}, y, composers,
                 log=lambda *a: None)
    assert "deadpan-expressive" not in out["deltas"]


def test_a_baseline_arm_reads_as_a_zero_delta_against_itself(toy):
    human, _arm, y, composers = toy
    out = report({HUMAN_ARM: human, "deadpan": human}, y, composers,
                 log=lambda *a: None)
    assert out["tau_c"][HUMAN_ARM] == pytest.approx(out["tau_c"]["deadpan"])
    assert out["deltas"]["deadpan-human"]["delta"] == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------
# embed_arm and the npz round trip
# --------------------------------------------------------------------------


def test_embed_arm_loads_each_adapter_once(toy):
    """900 model loads would dominate a pass that should take twenty minutes."""
    loads = []

    def embedder_for(fold):
        loads.append(fold)
        return lambda midi: np.full(DIM, float(fold))

    seg_ids = [f"s{i}" for i in range(10)]
    fold_by_seg = {s: i % N_FOLDS for i, s in enumerate(seg_ids)}
    out = embed_arm(seg_ids, fold_by_seg, lambda s: s, embedder_for,
                    log=lambda *a: None)

    assert sorted(loads) == list(range(N_FOLDS))
    assert len(loads) == N_FOLDS
    assert out["s0"].tolist() == [0.0] * DIM
    assert out["s1"].tolist() == [1.0] * DIM


def test_embed_arm_covers_every_seg_id(toy):
    seg_ids = [f"s{i}" for i in range(10)]
    fold_by_seg = {s: i % N_FOLDS for i, s in enumerate(seg_ids)}
    out = embed_arm(seg_ids, fold_by_seg, lambda s: s,
                    lambda f: (lambda m: np.zeros(DIM)), log=lambda *a: None)
    assert set(out) == set(seg_ids)


def test_arm_npz_round_trips(tmp_path):
    vectors = {"b": np.arange(DIM, dtype=float), "a": np.ones(DIM)}
    write_arm_npz(tmp_path / "arm.npz", vectors)
    back = read_arm_npz(tmp_path / "arm.npz")
    assert set(back) == {"a", "b"}
    np.testing.assert_array_equal(back["b"], vectors["b"])
