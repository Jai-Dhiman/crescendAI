"""Tests for retrain_compare (#168 final-model retrain) -- paired comparison
of two adapter arms trained on different pools.

Run: cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
"""
import numpy as np

from claim_measurement.difficulty.bakeoff_cv import paired_boot, tau_c
from claim_measurement.difficulty.ft_eval import oof_tau_per_fold
from claim_measurement.difficulty.retrain_compare import compare
from claim_measurement.difficulty.train_fold import write_fold_embeddings

N, N_FOLDS, SEED = 60, 5, 2026


def _arm(y, rng, noise):
    """Five adapter matrices carrying the grade signal at a given noise level.
    Lower noise stands in for the better-trained arm."""
    return {g: np.column_stack([
        y.astype(np.float32) + rng.normal(scale=noise, size=len(y)).astype(np.float32),
        rng.normal(size=(len(y), 2)).astype(np.float32),
    ]) for g in range(N_FOLDS)}


def _write_arm(dir_path, emb_by_fold, seg_ids, y, composers):
    dir_path.mkdir(parents=True, exist_ok=True)
    for f, X in emb_by_fold.items():
        write_fold_embeddings(dir_path / f"emb_fold{f}.npz", seg_ids, X,
                              y, composers)


def test_identical_arms_report_zero_delta(tmp_path, monkeypatch, capsys):
    """Two arms with byte-identical embeddings must produce delta exactly
    0.0000 on both criteria -- any nonzero value means the two arms are not
    being scored through the same folds."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 11, size=N)
    composers = np.arange(N)
    seg_ids = [f"seg{i:04d}" for i in range(N)]
    arm = _arm(y, rng, noise=2.0)

    _write_arm(tmp_path / "base", arm, seg_ids, y, composers)
    _write_arm(tmp_path / "new", arm, seg_ids, y, composers)
    _patch_features37(monkeypatch, seg_ids, y, composers)

    compare(tmp_path / "base", tmp_path / "new", data_root=None,
            n_folds=N_FOLDS, seed=SEED)
    out = capsys.readouterr().out
    assert "diagonal (PRIMARY)" in out and "pred-mean (SECONDARY)" in out
    assert out.count("delta +0.0000") == 2, out


def test_delta_sign_follows_the_retrain_arm(tmp_path, monkeypatch, capsys):
    """A retrain arm with a CLEANER signal must report a POSITIVE delta.

    This is the test that matters: a swapped argument order would still print
    a plausible number, with the sign -- and therefore the verdict -- inverted.
    """
    rng = np.random.default_rng(1)
    y = rng.integers(0, 11, size=N)
    composers = np.arange(N)
    seg_ids = [f"seg{i:04d}" for i in range(N)]

    _write_arm(tmp_path / "base", _arm(y, rng, noise=6.0), seg_ids, y, composers)
    _write_arm(tmp_path / "new", _arm(y, rng, noise=0.5), seg_ids, y, composers)
    _patch_features37(monkeypatch, seg_ids, y, composers)

    compare(tmp_path / "base", tmp_path / "new", data_root=None,
            n_folds=N_FOLDS, seed=SEED)
    out = capsys.readouterr().out
    primary = next(ln for ln in out.splitlines() if "PRIMARY" in ln)
    assert "delta +" in primary, primary
    base_tau = float(primary.split("baseline")[1].split()[0])
    retrain_tau = float(primary.split("retrain")[1].split()[0])
    assert retrain_tau > base_tau, primary


def test_matches_a_hand_rolled_paired_boot(tmp_path, monkeypatch, capsys):
    """The printed primary delta must equal paired_boot over the two arms'
    diagonal OOF vectors -- pins the reported number to the pre-registered
    statistic rather than to whatever the helper happens to compute."""
    rng = np.random.default_rng(2)
    y = rng.integers(0, 11, size=N)
    composers = np.arange(N)
    seg_ids = [f"seg{i:04d}" for i in range(N)]
    base_arm, new_arm = _arm(y, rng, noise=4.0), _arm(y, rng, noise=1.0)

    _write_arm(tmp_path / "base", base_arm, seg_ids, y, composers)
    _write_arm(tmp_path / "new", new_arm, seg_ids, y, composers)
    _patch_features37(monkeypatch, seg_ids, y, composers)

    compare(tmp_path / "base", tmp_path / "new", data_root=None,
            n_folds=N_FOLDS, seed=SEED)
    primary = next(ln for ln in capsys.readouterr().out.splitlines()
                   if "PRIMARY" in ln)

    base_oof = oof_tau_per_fold(base_arm, y, composers, N_FOLDS, SEED)
    new_oof = oof_tau_per_fold(new_arm, y, composers, N_FOLDS, SEED)
    d, lo, hi, _ = paired_boot(base_oof, new_oof, y, seed=SEED)
    assert f"delta {d:+.4f} CI95[{lo:+.4f},{hi:+.4f}]" in primary, primary
    assert f"retrain {tau_c(new_oof, y):.4f}" in primary, primary


def _patch_features37(monkeypatch, seg_ids, y, composers):
    """features37 supplies only the labels and the canonical row order here;
    the real .npz corpus is not needed to exercise the comparison."""
    import claim_measurement.difficulty.retrain_compare as rc

    monkeypatch.setattr(rc, "_load_features37",
                        lambda _root: (None, y, composers, seg_ids))
    monkeypatch.setattr(
        "claim_measurement.difficulty.bakeoff_paths.resolve_paths",
        lambda _root: type("P", (), {"emb_root": None})())
