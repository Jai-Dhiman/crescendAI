"""Tests for window_sweep (#168 S3 lever 1) -- eval-time chunk-length sweep.

Run: cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
"""
import json

import numpy as np
import pytest
import torch

from claim_measurement.difficulty.audio_emb_extract import build_fold_embedder
from claim_measurement.difficulty.bakeoff_npz import write_embedding_npz
from claim_measurement.difficulty.test_train_fold import _FakeOuter
from claim_measurement.difficulty.train_fold import (
    read_fold_embeddings,
    write_fold_embeddings,
)
from claim_measurement.difficulty.window_sweep import (
    extract_arm,
    load_eval_manifest,
    main,
)


def _write_bundle(bundle_dir, n=20):
    seg_ids = [f"p{i:03d}" for i in range(n)]
    manifest = [{"seg_id": s, "grade": i % 11, "composer_id": i}
                for i, s in enumerate(seg_ids)]
    bundle_dir.mkdir(parents=True)
    (bundle_dir / "eval_manifest.json").write_text(json.dumps(manifest))
    (bundle_dir / "midi").mkdir()
    for s in seg_ids:
        (bundle_dir / "midi" / f"{s}.mid").write_bytes(b"")
    return seg_ids


def test_extract_arm_writes_fold_npz_in_manifest_order(tmp_path):
    bundle = tmp_path / "bundle"
    seg_ids = _write_bundle(bundle)
    out_dir = tmp_path / "w512"

    def embedder_for_fold(fold):
        return lambda midi_path: np.full(4, float(fold), dtype=np.float32)

    extract_arm(bundle, out_dir, embedder_for_fold, n_folds=2)

    for f in range(2):
        data = read_fold_embeddings(out_dir / f"emb_fold{f}.npz")
        assert data["seg_ids"] == seg_ids
        assert data["embeddings"].shape == (len(seg_ids), 4)
        assert (data["embeddings"] == float(f)).all()
        assert data["grades"].dtype == np.int32


def test_extract_arm_resumes_per_fold_without_rebuilding_that_adapter(tmp_path):
    """One arm is ~3h of CPU; an interrupt must not cost completed folds, and
    a completed fold's 839M adapter load must not happen at all on re-run."""
    bundle = tmp_path / "bundle"
    seg_ids = _write_bundle(bundle)
    out_dir = tmp_path / "w512"
    write_fold_embeddings(out_dir / "emb_fold0.npz", seg_ids,
                           np.zeros((len(seg_ids), 4), dtype=np.float32),
                           np.zeros(len(seg_ids), dtype=np.int32),
                           np.arange(len(seg_ids), dtype=np.int32))
    built = []

    def embedder_for_fold(fold):
        built.append(fold)
        return lambda midi_path: np.zeros(4, dtype=np.float32)

    extract_arm(bundle, out_dir, embedder_for_fold, n_folds=2)

    assert built == [1]


def test_extract_arm_raises_on_nonfinite_embedding(tmp_path):
    """Research-harness rule: a NaN embedding must be loud. Written silently it
    would flow through np.nan_to_num in ft_eval and score as a zero vector."""
    bundle = tmp_path / "bundle"
    _write_bundle(bundle)

    def embedder_for_fold(fold):
        return lambda midi_path: np.full(4, np.nan, dtype=np.float32)

    with pytest.raises(ValueError, match="non-finite"):
        extract_arm(bundle, tmp_path / "w512",
                    embedder_for_fold, n_folds=1)


def _fake_loader_factory(record):
    def loader(checkpoint_path, repo_root, model_config):
        return (_FakeOuter(hidden=4, n_layers=1, vocab=16, record=record),
                lambda path: torch.arange(6) % 16,
                4)
    return loader


def _save_fake_adapter(adapter_dir):
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.05,
        target_modules=["self_attn.q_proj", "self_attn.k_proj",
                        "self_attn.v_proj", "self_attn.o_proj",
                        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"])
    peft_model = get_peft_model(_FakeOuter(hidden=4, n_layers=1, vocab=16),
                                lora_config)
    peft_model.save_pretrained(str(adapter_dir))


@pytest.mark.parametrize("window_len,expected_chunks", [
    (None, 2),  # falls back to the loader's max_len=4 -> ceil(6/4) chunks
    (2, 3),
    (6, 1),
])
def test_build_fold_embedder_window_len_overrides_chunking(
    tmp_path, window_len, expected_chunks
):
    """The sweep's whole mechanism: window_len must reach _extract_full_piece
    as the chunk length. The fake inner model records one entry per forward
    pass, and _extract_full_piece forwards once per chunk."""
    adapter_dir = tmp_path / "adapter"
    _save_fake_adapter(adapter_dir)
    record = []

    embed = build_fold_embedder(adapter_dir, checkpoint=None, repo_root=None,
                                model_config=None,
                                loader_factory=_fake_loader_factory(record),
                                window_len=window_len)
    embed(tmp_path / "piece.mid")

    assert len(record) == expected_chunks


def _write_score_fixture(tmp_path, arm_names, n=60):
    """features37 dir + one sweep root where every arm carries the SAME
    embeddings, so every paired delta is exactly zero."""
    data_root = tmp_path / "data"
    emb_dir = data_root / "results" / "bakeoff" / "emb" / "features37"
    rng = np.random.default_rng(0)
    seg_ids = [f"p{i:03d}" for i in range(n)]
    grades = rng.integers(0, 11, size=n)
    composers = np.arange(n)
    for i, seg_id in enumerate(seg_ids):
        write_embedding_npz(emb_dir / f"{seg_id}.npz",
                             {"raw37": rng.normal(size=5).astype(np.float32)},
                             grade=int(grades[i]), composer_id=int(composers[i]))

    sweep_root = tmp_path / "sweep"
    embeddings = np.column_stack([grades.astype(np.float32),
                                   rng.normal(size=(n, 2)).astype(np.float32)])
    for arm in arm_names:
        for f in range(5):
            write_fold_embeddings(sweep_root / arm / f"emb_fold{f}.npz",
                                   seg_ids, embeddings, grades, composers)
    return data_root, sweep_root, seg_ids


def test_score_identical_arms_report_zero_delta_noise(tmp_path, capsys):
    data_root, sweep_root, _ = _write_score_fixture(
        tmp_path, ["w1024", "w256"])

    exit_code = main(["score", "--sweep-root", str(sweep_root),
                      "--data-root", str(data_root)])

    assert exit_code == 0
    out = capsys.readouterr().out
    arm_line = next(ln for ln in out.splitlines() if ln.startswith("w256"))
    assert "+0.0000" in arm_line
    assert "noise" in arm_line


def test_score_rejects_missing_baseline_arm(tmp_path):
    data_root, sweep_root, _ = _write_score_fixture(tmp_path, ["w256"])

    with pytest.raises(SystemExit, match="baseline"):
        main(["score", "--sweep-root", str(sweep_root),
              "--data-root", str(data_root)])


def test_arm_oof_rejects_row_order_mismatch(tmp_path):
    """The pairing guard: permuted rows would not crash downstream, they would
    silently score wrong embeddings against the canonical y."""
    data_root, sweep_root, seg_ids = _write_score_fixture(tmp_path, ["w1024"])
    fold0 = read_fold_embeddings(sweep_root / "w1024" / "emb_fold0.npz")
    write_fold_embeddings(sweep_root / "w1024" / "emb_fold0.npz",
                           list(reversed(fold0["seg_ids"])),
                           fold0["embeddings"], fold0["grades"],
                           fold0["composer_ids"])

    with pytest.raises(SystemExit, match="row order"):
        main(["score", "--sweep-root", str(sweep_root),
              "--data-root", str(data_root)])


def test_load_eval_manifest_preserves_order(tmp_path):
    bundle = tmp_path / "bundle"
    expected = _write_bundle(bundle, n=7)

    seg_ids, grades, composer_ids = load_eval_manifest(
        bundle / "eval_manifest.json")

    assert seg_ids == expected
    assert grades.dtype == np.int32 and len(grades) == 7
    assert composer_ids.dtype == np.int32


def test_script_header_matches_audio_emb_extract():
    """window_sweep imports the same local chain (train_fold, bakeoff_cv,
    audio_emb_extract...) so its isolated-env header must not drift from
    audio_emb_extract.py's, whose own header is guarded dep-by-dep."""
    from pathlib import Path

    from claim_measurement.difficulty import audio_emb_extract, window_sweep

    def header(module):
        lines = Path(module.__file__).read_text().splitlines()
        end = lines.index("# ///")
        return "\n".join(lines[:end + 1])

    assert header(window_sweep) == header(audio_emb_extract)
