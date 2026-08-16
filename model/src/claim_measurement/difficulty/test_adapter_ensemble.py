"""Tests for adapter_ensemble (#168 S3 lever 2) -- adapter ensemble at
inference.

Run: cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
"""
import numpy as np

from claim_measurement.difficulty.adapter_ensemble import (
    embedding_mean_oof,
    main,
    prediction_mean_oof,
    single_adapter_oofs,
)
from claim_measurement.difficulty.bakeoff_cv import (
    composer_disjoint_folds,
    tau_c,
)
from claim_measurement.difficulty.bakeoff_npz import write_embedding_npz
from claim_measurement.difficulty.ft_eval import oof_tau_per_fold
from claim_measurement.difficulty.train_fold import write_fold_embeddings

N, N_FOLDS, SEED = 60, 5, 2026


def _population(n=N, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 11, size=n)
    composers = np.arange(n)
    return y, composers, rng


def _adapter_matrices(y, rng, n_adapters=N_FOLDS, noise=2.0):
    """Each adapter sees the same grade signal plus INDEPENDENT noise -- the
    regime where averaging should help."""
    return {g: np.column_stack([
        y.astype(np.float32) + rng.normal(scale=noise, size=len(y)).astype(np.float32),
        rng.normal(size=(len(y), 2)).astype(np.float32),
    ]) for g in range(n_adapters)}


def test_identical_adapters_make_ensemble_equal_baseline():
    """With all five adapters identical, both ensembles and every single
    reduce to the same model as the fold-diagonal baseline -- any deviation
    is a protocol bug, not an ensemble effect."""
    y, composers, rng = _population()
    X = _adapter_matrices(y, rng, n_adapters=1)[0]
    emb_by_adapter = {g: X for g in range(N_FOLDS)}

    base = oof_tau_per_fold(emb_by_adapter, y, composers, N_FOLDS, SEED)
    singles = single_adapter_oofs(emb_by_adapter, y, composers, N_FOLDS, SEED)

    np.testing.assert_allclose(prediction_mean_oof(singles), base)
    # atol: the float32 mean of five identical matrices is not bit-identical
    # to any one of them, and the ridge fit magnifies that to ~1e-8.
    np.testing.assert_allclose(
        embedding_mean_oof(emb_by_adapter, y, composers, N_FOLDS, SEED),
        base, atol=1e-5)


def test_single_adapter_diagonal_matches_baseline_block():
    """Adapter g alone, restricted to fold g's test rows, must equal the
    baseline exactly: same embeddings, same train rows, same ridge. This is
    the guard that single_adapter_oofs refits per fold instead of reusing
    the stored (leaky for g != f) heads."""
    y, composers, rng = _population()
    emb_by_adapter = _adapter_matrices(y, rng)

    base = oof_tau_per_fold(emb_by_adapter, y, composers, N_FOLDS, SEED)
    singles = single_adapter_oofs(emb_by_adapter, y, composers, N_FOLDS, SEED)

    for g, test_idx in enumerate(
            composer_disjoint_folds(composers, N_FOLDS, SEED)):
        np.testing.assert_allclose(singles[g][test_idx], base[test_idx])


def test_prediction_mean_beats_each_single_on_independent_noise():
    """The mechanism the lever bets on: with a shared signal and independent
    per-adapter noise, averaging predictions must rank better than any one
    adapter. Deterministic under the fixed rng."""
    y, composers, rng = _population()
    emb_by_adapter = _adapter_matrices(y, rng, noise=4.0)

    singles = single_adapter_oofs(emb_by_adapter, y, composers, N_FOLDS, SEED)
    ensemble_tau = tau_c(prediction_mean_oof(singles), y)

    assert all(ensemble_tau > tau_c(oof, y) for oof in singles.values())


def _write_cli_fixture(tmp_path, emb_by_adapter, y, composers):
    data_root = tmp_path / "data"
    emb_dir = data_root / "results" / "bakeoff" / "emb" / "features37"
    rng = np.random.default_rng(1)
    seg_ids = [f"p{i:03d}" for i in range(len(y))]
    for i, seg_id in enumerate(seg_ids):
        write_embedding_npz(emb_dir / f"{seg_id}.npz",
                             {"raw37": rng.normal(size=5).astype(np.float32)},
                             grade=int(y[i]), composer_id=int(composers[i]))
    arm_dir = tmp_path / "w1024"
    for g, X in emb_by_adapter.items():
        write_fold_embeddings(arm_dir / f"emb_fold{g}.npz", seg_ids, X,
                               y, composers)
    return data_root, arm_dir


def test_main_reports_baseline_spread_and_both_ensembles(tmp_path, capsys):
    y, composers, rng = _population()
    emb_by_adapter = _adapter_matrices(y, rng)
    data_root, arm_dir = _write_cli_fixture(tmp_path, emb_by_adapter,
                                             y, composers)

    exit_code = main(["--arm-dir", str(arm_dir),
                      "--data-root", str(data_root)])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "baseline (adapter f on fold f)" in out
    assert sum("adapter" in ln and "alone" in ln
               for ln in out.splitlines()) == N_FOLDS
    assert "single-adapter spread" in out
    pred_line = next(ln for ln in out.splitlines()
                     if ln.startswith("pred-mean"))
    emb_line = next(ln for ln in out.splitlines()
                    if ln.startswith("emb-mean"))
    for line in (pred_line, emb_line):
        assert "vs baseline" in line
        assert ("SIG" in line) or ("noise" in line)


def test_main_identical_adapters_report_zero_delta_noise(tmp_path, capsys):
    y, composers, rng = _population()
    X = _adapter_matrices(y, rng, n_adapters=1)[0]
    data_root, arm_dir = _write_cli_fixture(
        tmp_path, {g: X for g in range(N_FOLDS)}, y, composers)

    main(["--arm-dir", str(arm_dir), "--data-root", str(data_root)])

    out = capsys.readouterr().out
    pred_line = next(ln for ln in out.splitlines()
                     if ln.startswith("pred-mean"))
    assert "+0.0000" in pred_line
    assert "noise" in pred_line
    assert "range 0.0000" in out
