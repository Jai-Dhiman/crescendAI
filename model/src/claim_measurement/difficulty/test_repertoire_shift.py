"""Offline tests for the #167 repertoire hold-out. Synthetic embeddings and
grades throughout -- the properties under test are about which rows train and
which rows are scored, not about the encoder."""
from __future__ import annotations

import numpy as np
import pytest

from claim_measurement.difficulty.bakeoff_cv import composer_disjoint_folds
from claim_measurement.difficulty.repertoire_shift import (
    N_FOLDS,
    SEED,
    holdout_oof,
    labelled_mask,
    period_mask,
)

N = 120


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 11, size=N)
    composers = np.arange(N)          # one piece per composer, as in the real eval set
    emb_by_fold = {f: rng.normal(size=(N, 8)) for f in range(N_FOLDS)}
    holdout = np.zeros(N, dtype=bool)
    holdout[::3] = True               # 40 of 120
    return emb_by_fold, y, composers, holdout


MANIFEST = [{"seg_id": "a", "key": "A"}, {"seg_id": "b", "key": "B"},
            {"seg_id": "c", "key": "C"}]
LABELS = {"A": {"period": "Modern"}, "B": {"period": "Romantic"},
          "C": {"period": "Modern"}}


# --------------------------------------------------------------------------
# period_mask
# --------------------------------------------------------------------------


def test_period_mask_selects_the_named_period_in_seg_id_order():
    assert period_mask(["a", "b", "c"], MANIFEST, LABELS, "Modern").tolist() == \
           [True, False, True]


def test_period_mask_raises_rather_than_defaulting_unlabelled_pieces_to_train():
    """26 of the 900 eval pieces carry an empty or '-' period. Folding them
    into the training side would contaminate the control with unlabelled
    repertoire and never show up in the output."""
    labels = {**LABELS, "B": {"period": ""}}
    with pytest.raises(ValueError, match="no usable `period`"):
        period_mask(["a", "b", "c"], MANIFEST, labels, "Modern")


def test_period_mask_treats_a_dash_as_unlabelled():
    labels = {**LABELS, "B": {"period": "-"}}
    with pytest.raises(ValueError, match="no usable `period`"):
        period_mask(["a", "b", "c"], MANIFEST, labels, "Modern")


def test_labelled_mask_marks_blank_and_dash_and_missing_rows():
    labels = {"A": {"period": "Modern"}, "B": {"period": "-"}}
    assert labelled_mask(["a", "b", "c"], MANIFEST, labels).tolist() == \
           [True, False, False]


# --------------------------------------------------------------------------
# holdout_oof -- which rows get scored
# --------------------------------------------------------------------------


@pytest.mark.parametrize("arm", ["full", "size_matched", "holdout"])
def test_only_held_out_rows_are_scored(toy, arm):
    emb_by_fold, y, composers, holdout = toy
    oof = holdout_oof(emb_by_fold, y, composers, holdout, arm)
    assert np.all(np.isfinite(oof[holdout]))
    assert np.all(np.isnan(oof[~holdout]))


def test_every_held_out_row_is_scored_exactly_once_across_the_folds(toy):
    emb_by_fold, y, composers, holdout = toy
    oof = holdout_oof(emb_by_fold, y, composers, holdout, "holdout")
    assert int(np.isfinite(oof).sum()) == int(holdout.sum())


def test_the_holdout_arm_never_trains_on_a_held_out_row(toy, monkeypatch):
    """The property the whole measurement rests on. Captured at the fit
    boundary rather than inferred from the score."""
    emb_by_fold, _y, composers, holdout = toy
    captured = []

    def spy(X_train, y_train, X_test):
        captured.append(set(y_train.tolist()))
        return np.zeros(len(X_test))

    monkeypatch.setattr(
        "claim_measurement.difficulty.repertoire_shift._fit_predict", spy)
    # Each row's "grade" is its own index, so a leaked row is visible by name.
    holdout_oof(emb_by_fold, np.arange(N), composers, holdout, "holdout")

    held = set(np.flatnonzero(holdout).tolist())
    for train_rows in captured:
        assert not (train_rows & held)


def test_the_full_arm_does_train_on_held_out_rows_from_other_folds(toy, monkeypatch):
    emb_by_fold, y, composers, holdout = toy
    captured = []

    def spy(X_train, y_train, X_test):
        captured.append(set(y_train.tolist()))
        return np.zeros(len(X_test))

    monkeypatch.setattr(
        "claim_measurement.difficulty.repertoire_shift._fit_predict", spy)
    holdout_oof(emb_by_fold, np.arange(N), composers, holdout, "full")

    held = set(np.flatnonzero(holdout).tolist())
    assert any(train_rows & held for train_rows in captured)


def test_size_matched_trains_on_exactly_as_many_rows_as_holdout(toy, monkeypatch):
    """Without this the hold-out drop would bundle the repertoire effect with
    the cost of a third fewer training rows."""
    emb_by_fold, y, composers, holdout = toy
    sizes = {}

    for arm in ("holdout", "size_matched"):
        captured = []

        def spy(X_train, y_train, X_test, _c=captured):
            _c.append(len(y_train))
            return np.zeros(len(X_test))

        monkeypatch.setattr(
            "claim_measurement.difficulty.repertoire_shift._fit_predict", spy)
        holdout_oof(emb_by_fold, y, composers, holdout, arm)
        sizes[arm] = captured

    assert sizes["holdout"] == sizes["size_matched"]


def test_size_matched_draws_different_rows_than_holdout(toy, monkeypatch):
    """Same n, different membership -- otherwise the control is the arm."""
    emb_by_fold, y, composers, holdout = toy
    rows = {}
    for arm in ("holdout", "size_matched"):
        captured = []

        def spy(X_train, y_train, X_test, _c=captured):
            _c.append(set(y_train.tolist()))
            return np.zeros(len(X_test))

        monkeypatch.setattr(
            "claim_measurement.difficulty.repertoire_shift._fit_predict", spy)
        holdout_oof(emb_by_fold, np.arange(N), composers, holdout, arm)
        rows[arm] = captured
    assert rows["holdout"] != rows["size_matched"]


def test_the_full_arm_trains_on_the_whole_pool(toy, monkeypatch):
    emb_by_fold, y, composers, holdout = toy
    captured = []

    def spy(X_train, y_train, X_test):
        captured.append(len(y_train))
        return np.zeros(len(X_test))

    monkeypatch.setattr(
        "claim_measurement.difficulty.repertoire_shift._fit_predict", spy)
    holdout_oof(emb_by_fold, y, composers, holdout, "full")

    folds = composer_disjoint_folds(composers, N_FOLDS, SEED)
    assert captured == [N - len(test_idx) for test_idx in folds]


def test_each_fold_is_scored_with_its_own_adapters_embeddings(toy, monkeypatch):
    """Mixing rows across adapters would score a head fit on one encoder
    against another encoder's features."""
    emb_by_fold, y, composers, holdout = toy
    # Tag each fold's embedding block with its fold index in column 0.
    for f in range(N_FOLDS):
        emb_by_fold[f][:, 0] = f
    seen = []

    def spy(X_train, y_train, X_test):
        seen.append((set(X_train[:, 0].tolist()), set(X_test[:, 0].tolist())))
        return np.zeros(len(X_test))

    monkeypatch.setattr(
        "claim_measurement.difficulty.repertoire_shift._fit_predict", spy)
    holdout_oof(emb_by_fold, y, composers, holdout, "holdout")

    for f, (train_tags, test_tags) in enumerate(seen):
        assert train_tags == {f} and test_tags == {f}


# --------------------------------------------------------------------------
# holdout_oof -- guards
# --------------------------------------------------------------------------


def test_rejects_an_unknown_arm(toy):
    emb_by_fold, y, composers, holdout = toy
    with pytest.raises(ValueError, match="arm must be one of"):
        holdout_oof(emb_by_fold, y, composers, holdout, "nope")


def test_rejects_a_mask_of_the_wrong_length(toy):
    emb_by_fold, y, composers, _ = toy
    with pytest.raises(ValueError, match="off-by-one"):
        holdout_oof(emb_by_fold, y, composers, np.ones(N - 1, dtype=bool), "full")


def test_rejects_an_empty_holdout(toy):
    emb_by_fold, y, composers, _ = toy
    with pytest.raises(ValueError, match="no rows"):
        holdout_oof(emb_by_fold, y, composers, np.zeros(N, dtype=bool), "full")


def test_rejects_a_missing_fold(toy):
    emb_by_fold, y, composers, holdout = toy
    del emb_by_fold[3]
    with pytest.raises(KeyError, match="missing fold 3"):
        holdout_oof(emb_by_fold, y, composers, holdout, "full")


def test_is_deterministic_across_calls(toy):
    """size_matched draws a random subset; a fresh draw per call would make the
    control's number unreproducible."""
    emb_by_fold, y, composers, holdout = toy
    a = holdout_oof(emb_by_fold, y, composers, holdout, "size_matched")
    b = holdout_oof(emb_by_fold, y, composers, holdout, "size_matched")
    np.testing.assert_array_equal(np.nan_to_num(a), np.nan_to_num(b))
