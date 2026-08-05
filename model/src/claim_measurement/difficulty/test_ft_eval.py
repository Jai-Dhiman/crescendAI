"""Tests for ft_eval (#149 / #138 Phase 1) -- the gate: OOF where X differs
per fold, plus the CLI wiring against features37 + emb_fold{F}.npz files.

Run: cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
"""
import numpy as np
import pytest

from claim_measurement.difficulty.bakeoff_cv import tau_c
from claim_measurement.difficulty.ft_eval import oof_tau_per_fold


def test_oof_tau_per_fold_recovers_a_strong_per_fold_linear_signal():
    rng = np.random.default_rng(2026)
    n = 200
    # all distinct -> vacuous disjointness
    composers = np.array([f"composer_{i}" for i in range(n)])
    y = rng.integers(0, 11, size=n).astype(float)

    emb_by_fold = {}
    for f in range(5):
        rng_f = np.random.default_rng(1000 + f)
        noise = rng_f.normal(size=(n, 3)) * 0.01
        emb_by_fold[f] = np.column_stack([y * (f + 1), noise])

    oof = oof_tau_per_fold(emb_by_fold, y, composers, n_folds=5, seed=2026)

    assert not np.isnan(oof).any()
    assert tau_c(oof, y) > 0.9


def test_oof_tau_per_fold_raises_on_missing_fold_embeddings():
    composers = np.array([f"composer_{i}" for i in range(50)])
    y = np.arange(50, dtype=float) % 11
    emb_by_fold = {0: np.random.default_rng(0).normal(size=(50, 2))}  # folds 1-4 gone

    with pytest.raises(KeyError):
        oof_tau_per_fold(emb_by_fold, y, composers, n_folds=5, seed=2026)


from claim_measurement.difficulty.bakeoff_npz import write_embedding_npz
from claim_measurement.difficulty.ft_eval import main
from claim_measurement.difficulty.train_fold import write_fold_embeddings


def test_main_prints_the_gate_comparison_against_features37(tmp_path, capsys):
    data_root = tmp_path / "data"
    emb_dir = data_root / "results" / "bakeoff" / "emb" / "features37"
    rng = np.random.default_rng(0)
    n = 60
    seg_ids = [f"p{i:03d}" for i in range(n)]  # zero-padded -> lexical == list order
    grades = rng.integers(0, 11, size=n)
    composers = np.arange(n)  # all distinct -> vacuous disjointness, like the real 900

    for i, seg_id in enumerate(seg_ids):
        write_embedding_npz(emb_dir / f"{seg_id}.npz",
                             {"raw37": rng.normal(size=5).astype(np.float32)},
                             grade=int(grades[i]), composer_id=int(composers[i]))

    fold_emb_dir = tmp_path / "fold_embeddings"
    for f in range(5):
        # feature 0 is a strong linear signal so the gate reports SIG, not noise
        embeddings = np.column_stack([grades.astype(np.float32) * (f + 1),
                                       rng.normal(size=(n, 2)).astype(np.float32)])
        write_fold_embeddings(fold_emb_dir / f"emb_fold{f}.npz", seg_ids=seg_ids,
                               embeddings=embeddings, grades=grades,
                               composer_ids=composers)

    exit_code = main(
        ["--data-root", str(data_root), "--fold-emb-dir", str(fold_emb_dir)])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "moonbeam_ft_mean|ridge - features37|ridge:" in out
