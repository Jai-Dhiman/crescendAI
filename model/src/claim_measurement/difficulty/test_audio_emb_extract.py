"""Tests for audio_emb_extract (#149 / #138 Phase 1 Stage 5b) -- audio-derived
MoonBeam embeddings, each piece run through its OWN fold's adapter. The
backbone is injected (the loader_factory pattern train_fold.py establishes),
so every test here is CPU-only, offline, and needs no checkpoint.

Run: cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
"""
import json

import numpy as np
import torch

from claim_measurement.difficulty.audio_emb_extract import (
    build_fold_embedder,
    extract_audio_embeddings,
    fold_of_seg_ids,
    write_notes_midi,
)
from claim_measurement.difficulty.bakeoff_cv import composer_disjoint_folds
from claim_measurement.difficulty.bakeoff_npz import read_embedding_npz
from claim_measurement.difficulty.psyllabus import notes_from_midi_bytes

# The fake MoonBeam-shaped model train_fold's own tests already exercise --
# reused rather than duplicated so both suites test the same model shape.
from claim_measurement.difficulty.test_train_fold import _FakeOuter

_NOTES = [
    {"pitch": 60, "onset": 0.0, "offset": 0.5, "velocity": 80},
    {"pitch": 64, "onset": 0.5, "offset": 1.0, "velocity": 90},
    {"pitch": 67, "onset": 1.0, "offset": 1.75, "velocity": 70},
]


def test_write_notes_midi_round_trips_the_cached_note_dicts(tmp_path):
    """MoonBeam's tokenizer consumes a MIDI file, but realaudio_check.py's
    cache holds note dicts -- so they have to be materialised back losslessly
    enough that the encoder sees the transcription, not an approximation."""
    midi_path = tmp_path / "piece.mid"

    write_notes_midi(_NOTES, midi_path)
    round_tripped = notes_from_midi_bytes(midi_path.read_bytes())

    assert [n["pitch"] for n in round_tripped] == [60, 64, 67]
    assert [n["velocity"] for n in round_tripped] == [80, 90, 70]
    np.testing.assert_allclose([n["onset"] for n in round_tripped],
                               [0.0, 0.5, 1.0], atol=1e-3)
    np.testing.assert_allclose([n["offset"] for n in round_tripped],
                               [0.5, 1.0, 1.75], atol=1e-3)


def _eval_index(n=40):
    seg_ids = [f"p{i:03d}" for i in range(n)]
    composer_ids = {s: i % 13 for i, s in enumerate(seg_ids)}
    grades = {s: i % 11 for i, s in enumerate(seg_ids)}
    return seg_ids, grades, composer_ids


def _write_cache(cache_dir, seg_ids):
    cache_dir.mkdir(parents=True, exist_ok=True)
    for seg_id in seg_ids:
        (cache_dir / f"{seg_id}.json").write_text(
            json.dumps({"notes": _NOTES, "pedals": []}))


def test_each_piece_is_embedded_through_its_own_composer_disjoint_fold_adapter(
    tmp_path,
):
    """Scoring a piece through an adapter that trained on it is train-on-test
    -- the exact contamination #135's 0.824 anchor died of. The fold must come
    from bakeoff_cv.composer_disjoint_folds at (5, 2026), the same folds
    ft_eval.py and score_audio_subset use."""
    seg_ids, grades, composer_ids = _eval_index()
    composers = np.array([composer_ids[s] for s in seg_ids])
    cache_dir = tmp_path / "cache"
    subset = seg_ids[:12]
    _write_cache(cache_dir, subset)
    out_dir = tmp_path / "audio_emb"

    fold_of = fold_of_seg_ids(seg_ids, composers, n_folds=5, seed=2026)

    def embedder_for_fold(fold):
        # a vector that identifies which fold's adapter produced it
        return lambda midi_path: np.full(4, float(fold), dtype=np.float32)

    report = extract_audio_embeddings(cache_dir, out_dir, fold_of, grades,
                                      composer_ids, embedder_for_fold)

    assert (report.ok, report.skipped, report.failed) == (12, 0, [])
    expected_folds = {seg_ids[i]: f for f, idx in enumerate(
        composer_disjoint_folds(composers, 5, 2026)) for i in idx}
    for seg_id in subset:
        record = read_embedding_npz(out_dir / f"{seg_id}.npz")
        assert record.embeddings["mean_pool"][0] == expected_folds[seg_id]
        assert record.grade == grades[seg_id]
        assert record.composer_id == composer_ids[seg_id]


def test_extraction_is_resumable_and_never_loads_an_already_done_folds_adapter(
    tmp_path,
):
    """This is a long local run over ~709 pieces; an interrupt must not cost
    the work already on disk, and re-running must not reload an 839M backbone
    for a fold whose pieces are all extracted."""
    seg_ids, grades, composer_ids = _eval_index()
    composers = np.array([composer_ids[s] for s in seg_ids])
    fold_of = fold_of_seg_ids(seg_ids, composers, n_folds=5, seed=2026)
    cache_dir = tmp_path / "cache"
    subset = seg_ids[:12]
    _write_cache(cache_dir, subset)
    out_dir = tmp_path / "audio_emb"
    loaded = []

    def embedder_for_fold(fold):
        loaded.append(fold)
        return lambda midi_path: np.full(4, float(fold), dtype=np.float32)

    first = extract_audio_embeddings(cache_dir, out_dir, fold_of, grades,
                                     composer_ids, embedder_for_fold)
    loaded_first = list(loaded)
    loaded.clear()
    second = extract_audio_embeddings(cache_dir, out_dir, fold_of, grades,
                                      composer_ids, embedder_for_fold)

    assert first.ok == 12 and first.skipped == 0
    assert second.ok == 0 and second.skipped == 12
    assert loaded_first  # the first pass loaded at least one fold's adapter
    assert loaded == []  # the second loaded none


def test_a_cached_piece_that_is_not_an_eval_piece_is_reported_not_dropped(tmp_path):
    seg_ids, grades, composer_ids = _eval_index()
    composers = np.array([composer_ids[s] for s in seg_ids])
    fold_of = fold_of_seg_ids(seg_ids, composers, n_folds=5, seed=2026)
    cache_dir = tmp_path / "cache"
    _write_cache(cache_dir, ["not_an_eval_piece"])

    report = extract_audio_embeddings(
        cache_dir, tmp_path / "audio_emb", fold_of, grades, composer_ids,
        lambda fold: (lambda midi_path: np.zeros(4, dtype=np.float32)))

    assert report.ok == 0
    assert len(report.failed) == 1
    assert "not an eval piece" in report.failed[0]


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


def test_build_fold_embedder_extraction_is_byte_identical_across_repeated_calls(
    tmp_path,
):
    """lora_dropout=0.05 is active in train mode and torch.no_grad() does NOT
    disable dropout, so without the .eval() in build_fold_embedder every
    extraction would be a different random draw and the real-audio gate would
    be measuring noise. This was a P0 in review; keep it nailed down."""
    adapter_dir = tmp_path / "adapter"
    _save_fake_adapter(adapter_dir)
    midi_path = tmp_path / "piece.mid"
    write_notes_midi(_NOTES, midi_path)

    def fake_loader_factory(checkpoint_path, repo_root, model_config):
        return (_FakeOuter(hidden=4, n_layers=1, vocab=16),
                lambda path: torch.arange(6) % 16,
                4)

    embed = build_fold_embedder(adapter_dir, checkpoint=None, repo_root=None,
                                model_config=None,
                                loader_factory=fake_loader_factory)

    first, second, third = embed(midi_path), embed(midi_path), embed(midi_path)

    assert first.shape == (4,)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first, third)


def test_peft_is_imported_only_after_the_loader_puts_the_fork_on_sys_path():
    """Ordering contract, asserted on the source because the order IS the
    contract: loader_factory installs the fork's PARTIAL vendored transformers
    on sys.path, _stub_absent_transformers_models then supplies the
    models.bloom that peft/utils/constants feature-probes for, and only then
    may peft be imported. The original code imported peft FIRST, which binds
    whatever transformers is already installed and skips the stub entirely.
    """
    from pathlib import Path

    from claim_measurement.difficulty import audio_emb_extract

    src = Path(audio_emb_extract.__file__).read_text()
    loader_call = src.index("    base_model, tokenize, max_len = loader_factory(")
    stub_call = src.index("    _stub_absent_transformers_models()\n")
    peft_import = src.index("    from peft import PeftModel")
    assert loader_call < stub_call < peft_import, (
        "build_fold_embedder must call loader_factory, then the stub, then "
        "import peft -- in that order")


def test_script_header_declares_every_dep_this_module_and_its_imports_need():
    """`uv run --script` builds an isolated env from this file's `# /// script`
    header ALONE. A module-scope import anywhere in the local import chain that
    the header does not declare fails at startup; a LAZY import that it does not
    declare fails later, mid-run, after real work. That is how the first pilot
    job died (scipy, via bakeoff_cv's `from scipy import stats`).
    """
    import ast
    import re
    import sys
    from pathlib import Path

    from claim_measurement.difficulty import audio_emb_extract

    path = Path(audio_emb_extract.__file__).resolve()
    module_dir = path.parent
    block = re.search(r"# /// script\n(.*?)# ///", path.read_text(), re.DOTALL).group(1)
    declared = {re.split(r"[<>=!\[]", d)[0].strip().lower().replace("-", "_")
                for d in re.findall(r'"([^"]+)"', block)}
    # distribution name -> importable top-level module, where they differ
    declared |= {"sklearn" if d == "scikit_learn" else d for d in declared}

    # This file plus every local module it imports at module scope.
    chain = ["audio_emb_extract.py", "bakeoff_cv.py", "bakeoff_npz.py",
             "train_fold.py"]
    for filename in chain:
        tree = ast.parse((module_dir / filename).read_text())
        # Lazy imports count too: pretty_midi and peft are function-scope here
        # and are still fatal when the function runs inside the isolated env.
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots = [node.module.split(".")[0]]
            else:
                continue
            for root in roots:
                if (root in ("__future__", "claim_measurement", "transformers",
                             "bakeoff_cv", "ranking_loss", "trackio")
                        or root in sys.stdlib_module_names):
                    continue  # local, vendored-by-the-fork, or bundle-supplied
                assert root in declared, (
                    f"{filename} imports {root!r}, but audio_emb_extract.py's "
                    f"`# /// script` header does not declare it. The isolated "
                    f"env would die on it. Declared: {sorted(declared)}")
