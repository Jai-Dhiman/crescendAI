"""Tests for train_fold (#149 / #138 Phase 1) -- LoRA target modules and the
CLI wiring of a full fine-tune epoch, via an injected fake loader_factory
(the pattern moonbeam_extract_script.py already establishes). No mocks of
internal collaborators; real torch and real peft (both in model/.venv).

Run: cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
"""
from claim_measurement.difficulty.train_fold import lora_target_modules


def test_lora_target_modules_targets_top_5_of_15_layers_35_modules():
    modules = lora_target_modules(n_layers=15, n_top=5)

    assert len(modules) == 35
    assert modules[:7] == [
        "model.layers.10.self_attn.q_proj", "model.layers.10.self_attn.k_proj",
        "model.layers.10.self_attn.v_proj", "model.layers.10.self_attn.o_proj",
        "model.layers.10.mlp.gate_proj", "model.layers.10.mlp.up_proj",
        "model.layers.10.mlp.down_proj",
    ]
    assert {int(m.split(".")[2]) for m in modules} == {10, 11, 12, 13, 14}
    excluded = {"decoder_embedding", "summary_projection", "lm_head", "fc_out"}
    assert not any(any(x in m for x in excluded) for m in modules)


import json
from pathlib import Path

import numpy as np
import pytest
import torch

from claim_measurement.difficulty.train_fold import (
    _extract_full_piece,
    main,
    read_fold_embeddings,
)


class _FakeLayer(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.self_attn = torch.nn.Module()
        self.self_attn.q_proj = torch.nn.Linear(hidden, hidden, bias=False)
        self.self_attn.k_proj = torch.nn.Linear(hidden, hidden, bias=False)
        self.self_attn.v_proj = torch.nn.Linear(hidden, hidden, bias=False)
        self.self_attn.o_proj = torch.nn.Linear(hidden, hidden, bias=False)
        self.mlp = torch.nn.Module()
        self.mlp.gate_proj = torch.nn.Linear(hidden, hidden, bias=False)
        self.mlp.up_proj = torch.nn.Linear(hidden, hidden, bias=False)
        self.mlp.down_proj = torch.nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        h = x + self.self_attn.o_proj(
            self.self_attn.q_proj(x) + self.self_attn.k_proj(x)
            + self.self_attn.v_proj(x))
        h = h + self.mlp.down_proj(
            torch.relu(self.mlp.gate_proj(h)) * self.mlp.up_proj(h))
        return h


class _FakeInner(torch.nn.Module):
    def __init__(self, hidden, n_layers, vocab):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab, hidden)
        self.layers = torch.nn.ModuleList([_FakeLayer(hidden) for _ in range(n_layers)])

    def forward(self, input_ids, position_ids=None, use_cache=False, return_dict=True):
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h)

        class Out:
            pass

        out = Out()
        out.last_hidden_state = h
        return out


class _FakeOuter(torch.nn.Module):
    """Mimics LlamaForCausalLM: a .model attribute (inner transformer) plus an
    lm_head this design never calls and never LoRA-targets."""

    def __init__(self, hidden=4, n_layers=1, vocab=16):
        super().__init__()
        self.model = _FakeInner(hidden, n_layers, vocab)
        self.lm_head = torch.nn.Linear(hidden, vocab, bias=False)


_TOKEN_LENGTHS = {"t0": 6, "t1": 7, "t2": 8, "t3": 6, "v0": 7, "e0": 5, "e1": 9}


def _stage_fake_bundle_dir(tmp_path):
    """A minimal already-downloaded push_train_dataset.py bundle: just the
    code/ subdir main() needs combined_loss/tau_c from, staged from the SAME
    files push_train_dataset.stage_training_bundle copies verbatim (never a
    second, vendored implementation)."""
    import shutil

    bundle_dir = tmp_path / "bundle"
    code_dir = bundle_dir / "code"
    code_dir.mkdir(parents=True)
    module_dir = Path(__file__).resolve().parent
    shutil.copy2(module_dir / "ranking_loss.py", code_dir / "ranking_loss.py")
    shutil.copy2(module_dir / "bakeoff_cv.py", code_dir / "bakeoff_cv.py")
    return bundle_dir


def test_main_trains_a_lora_adapter_and_writes_emb_fold_for_all_eval_pieces(tmp_path):
    fold_plan = [{
        "fold": 0,
        "test_seg_ids": ["e0", "e1"],
        "train_seg_ids": ["t0", "t1", "t2", "t3"],
        "val_seg_ids": ["v0"],
    }]
    (tmp_path / "fold_plan.json").write_text(json.dumps(fold_plan))
    (tmp_path / "pool_grades.json").write_text(json.dumps(
        {"t0": 1, "t1": 5, "t2": 8, "t3": 3, "v0": 4}))
    (tmp_path / "eval_manifest.json").write_text(json.dumps([
        {"seg_id": "e0", "grade": 2, "composer_id": 0},
        {"seg_id": "e1", "grade": 9, "composer_id": 1},
    ]))
    out_dir = tmp_path / "fold0"

    def fake_loader_factory(checkpoint_path, repo_root, model_config):
        outer = _FakeOuter(hidden=4, n_layers=1, vocab=16)

        def tokenize(midi_path):
            n = _TOKEN_LENGTHS[midi_path.stem]
            return torch.arange(n) % 16

        return outer, tokenize, 4  # matches --max-len below

    exit_code = main(
        [
            "--fold", "0",
            "--checkpoint", str(tmp_path / "fake.pt"),
            "--repo-root", str(tmp_path / "repo"),
            "--model-config", str(tmp_path / "repo" / "model_config.json"),
            "--fold-plan", str(tmp_path / "fold_plan.json"),
            "--pool-grades", str(tmp_path / "pool_grades.json"),
            "--eval-manifest", str(tmp_path / "eval_manifest.json"),
            "--midi-dir", str(tmp_path / "mid"),
            "--out-dir", str(out_dir),
            "--bundle-dir", str(_stage_fake_bundle_dir(tmp_path)),
            "--hidden-size", "4",
            "--n-layers", "1",
            "--n-top-layers", "1",
            "--max-len", "4",
            "--epochs", "1",
            "--micro-batch", "2",
        ],
        loader_factory=fake_loader_factory,
    )

    assert exit_code == 0
    assert (out_dir / "adapter" / "adapter_config.json").exists()
    fold_data = read_fold_embeddings(out_dir / "emb_fold0.npz")
    assert fold_data["seg_ids"] == ["e0", "e1"]
    assert fold_data["embeddings"].shape == (2, 4)
    assert list(fold_data["grades"]) == [2, 9]
    assert list(fold_data["composer_ids"]) == [0, 1]


def test_extraction_is_byte_identical_across_repeated_calls_in_eval_mode():
    """LoRA's lora_dropout=0.05 is active in train mode -- torch.no_grad()
    does NOT disable dropout -- so extraction is only deterministic once the
    model has been put in eval mode. Regression test for the P0 finding: the
    graded emb_fold{F}.npz must not be stochastic."""
    from peft import LoraConfig, get_peft_model

    outer = _FakeOuter(hidden=4, n_layers=1, vocab=16)
    lora_config = LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.05,
        target_modules=["self_attn.q_proj", "self_attn.k_proj",
                         "self_attn.v_proj", "self_attn.o_proj",
                         "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"])
    peft_model = get_peft_model(outer, lora_config)
    transformer = peft_model.model.model
    tokens = torch.arange(6) % 16

    peft_model.eval()
    first = _extract_full_piece(transformer, tokens, max_len=4)
    second = _extract_full_piece(transformer, tokens, max_len=4)
    third = _extract_full_piece(transformer, tokens, max_len=4)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first, third)


def test_main_actually_updates_lora_weights_not_just_the_score_head(
    tmp_path, monkeypatch
):
    """A stale or un-injected `transformer` reference would still pass every
    assertion in test_main_trains_a_lora_adapter_and_writes_emb_fold_for_all_eval_pieces
    (the forward pass, save_pretrained, and extraction all succeed with only
    the head training) and land indistinguishably near frozen 0.8257 -- a
    false negative that costs $13 and a wrong MIREX conclusion. Guard against
    a detached LoRA by asserting lora_A AND lora_B actually move."""
    fold_plan = [{
        "fold": 0,
        "test_seg_ids": ["e0", "e1"],
        "train_seg_ids": ["t0", "t1", "t2", "t3"],
        "val_seg_ids": ["v0"],
    }]
    (tmp_path / "fold_plan.json").write_text(json.dumps(fold_plan))
    (tmp_path / "pool_grades.json").write_text(json.dumps(
        {"t0": 1, "t1": 5, "t2": 8, "t3": 3, "v0": 4}))
    (tmp_path / "eval_manifest.json").write_text(json.dumps([
        {"seg_id": "e0", "grade": 2, "composer_id": 0},
        {"seg_id": "e1", "grade": 9, "composer_id": 1},
    ]))
    out_dir = tmp_path / "fold0_lora_moves"
    captured = {}

    def fake_loader_factory(checkpoint_path, repo_root, model_config):
        outer = _FakeOuter(hidden=4, n_layers=1, vocab=16)
        captured["outer"] = outer

        def tokenize(midi_path):
            n = _TOKEN_LENGTHS[midi_path.stem]
            return torch.arange(n) % 16

        return outer, tokenize, 4  # matches --max-len below

    import peft

    real_get_peft_model = peft.get_peft_model
    initial = {}

    def spying_get_peft_model(base_model, lora_config):
        peft_model = real_get_peft_model(base_model, lora_config)
        proj = peft_model.model.model.layers[-1].self_attn.q_proj
        initial["lora_A"] = proj.lora_A["default"].weight.detach().clone()
        initial["lora_B"] = proj.lora_B["default"].weight.detach().clone()
        return peft_model

    monkeypatch.setattr(peft, "get_peft_model", spying_get_peft_model)

    exit_code = main(
        [
            "--fold", "0",
            "--checkpoint", str(tmp_path / "fake.pt"),
            "--repo-root", str(tmp_path / "repo"),
            "--model-config", str(tmp_path / "repo" / "model_config.json"),
            "--fold-plan", str(tmp_path / "fold_plan.json"),
            "--pool-grades", str(tmp_path / "pool_grades.json"),
            "--eval-manifest", str(tmp_path / "eval_manifest.json"),
            "--midi-dir", str(tmp_path / "mid"),
            "--out-dir", str(out_dir),
            "--bundle-dir", str(_stage_fake_bundle_dir(tmp_path)),
            "--hidden-size", "4",
            "--n-layers", "1",
            "--n-top-layers", "1",
            "--max-len", "4",
            "--epochs", "1",
            "--micro-batch", "2",
        ],
        loader_factory=fake_loader_factory,
    )

    assert exit_code == 0
    proj = captured["outer"].model.layers[-1].self_attn.q_proj
    assert not torch.equal(proj.lora_A["default"].weight.detach(), initial["lora_A"])
    assert not torch.equal(proj.lora_B["default"].weight.detach(), initial["lora_B"])


def _fold0_args(tmp_path, out_dir, extra=()):
    fold_plan = [{
        "fold": 0,
        "test_seg_ids": ["e0", "e1"],
        "train_seg_ids": ["t0", "t1", "t2", "t3"],
        "val_seg_ids": ["v0"],
    }]
    (tmp_path / "fold_plan.json").write_text(json.dumps(fold_plan))
    (tmp_path / "pool_grades.json").write_text(json.dumps(
        {"t0": 1, "t1": 5, "t2": 8, "t3": 3, "v0": 4}))
    (tmp_path / "eval_manifest.json").write_text(json.dumps([
        {"seg_id": "e0", "grade": 2, "composer_id": 0},
        {"seg_id": "e1", "grade": 9, "composer_id": 1},
    ]))
    return [
        "--fold", "0",
        "--checkpoint", str(tmp_path / "fake.pt"),
        "--repo-root", str(tmp_path / "repo"),
        "--model-config", str(tmp_path / "repo" / "model_config.json"),
        "--fold-plan", str(tmp_path / "fold_plan.json"),
        "--pool-grades", str(tmp_path / "pool_grades.json"),
        "--eval-manifest", str(tmp_path / "eval_manifest.json"),
        "--midi-dir", str(tmp_path / "mid"),
        "--out-dir", str(out_dir),
        "--bundle-dir", str(_stage_fake_bundle_dir(tmp_path)),
        "--hidden-size", "4",
        "--n-layers", "1",
        "--n-top-layers", "1",
        "--max-len", "4",
        "--epochs", "1",
        "--micro-batch", "2",
        *extra,
    ]


def _fake_loader_factory(checkpoint_path, repo_root, model_config):
    outer = _FakeOuter(hidden=4, n_layers=1, vocab=16)

    def tokenize(midi_path):
        n = _TOKEN_LENGTHS[midi_path.stem]
        return torch.arange(n) % 16

    return outer, tokenize, 4  # matches --max-len in _fold0_args


def test_main_requires_bundle_dir_or_bundle_repo(tmp_path):
    """Neither flag given means main() has no way to reach combined_loss/
    tau_c (train_fold.py is uploaded alone by `hf jobs uv run`, without the
    rest of this package) -- refuse loudly rather than crashing deep inside
    the training loop."""
    args = _fold0_args(tmp_path, tmp_path / "fold0")
    # Drop the --bundle-dir this helper adds, to exercise the missing-both case.
    bundle_idx = args.index("--bundle-dir")
    del args[bundle_idx:bundle_idx + 2]

    with pytest.raises(ValueError, match="--bundle-dir or --bundle-repo"):
        main(args, loader_factory=_fake_loader_factory)


def test_main_uploads_adapter_and_embeddings_when_output_repo_is_set(tmp_path):
    """--output-repo must trigger the injected uploader with the out-dir and
    repo id -- without this, an HF Jobs container's local disk (and the $13
    of GPU time that filled it) is discarded when the container exits."""
    out_dir = tmp_path / "fold0"
    calls = []

    def fake_uploader(uploaded_out_dir, repo_id):
        calls.append((uploaded_out_dir, repo_id))

    exit_code = main(
        _fold0_args(tmp_path, out_dir, extra=(
            "--output-repo", "jaidhiman/phase1-lora-fold0",
        )),
        loader_factory=_fake_loader_factory,
        uploader=fake_uploader,
    )

    assert exit_code == 0
    assert calls == [(out_dir, "jaidhiman/phase1-lora-fold0")]


def test_main_does_not_upload_when_output_repo_is_not_set(tmp_path):
    calls = []

    def fake_uploader(uploaded_out_dir, repo_id):
        calls.append((uploaded_out_dir, repo_id))

    exit_code = main(
        _fold0_args(tmp_path, tmp_path / "fold0"),
        loader_factory=_fake_loader_factory,
        uploader=fake_uploader,
    )

    assert exit_code == 0
    assert not calls


def _stage_full_fake_bundle(tmp_path):
    """A bundle staged the way push_train_dataset.py stages one: code/ plus
    every input path train_fold.py now DEFAULTS into, at the same relative
    locations. The submit line inside a job container passes none of them,
    because snapshot_download's cache path is unknowable at submit time."""
    bundle_dir = _stage_fake_bundle_dir(tmp_path)
    (bundle_dir / "fold_plans.json").write_text(json.dumps([{
        "fold": 0,
        "test_seg_ids": ["e0", "e1"],
        "train_seg_ids": ["t0", "t1", "t2", "t3"],
        "val_seg_ids": ["v0"],
    }]))
    (bundle_dir / "grades.json").write_text(json.dumps(
        {"t0": 1, "t1": 5, "t2": 8, "t3": 3, "v0": 4}))
    (bundle_dir / "eval_manifest.json").write_text(json.dumps([
        {"seg_id": "e0", "grade": 2, "composer_id": 0},
        {"seg_id": "e1", "grade": 9, "composer_id": 1},
    ]))
    (bundle_dir / "midi").mkdir()
    config_dir = bundle_dir / "moonbeam_repo" / "src" / "llama_recipes" / "configs"
    config_dir.mkdir(parents=True)
    (config_dir / "model_config.json").write_text("{}")
    return bundle_dir


def _bundle_only_args(bundle_dir, out_dir, extra=()):
    return [
        "--fold", "0",
        "--out-dir", str(out_dir),
        "--bundle-dir", str(bundle_dir),
        "--checkpoint", str(bundle_dir / "fake.pt"),
        "--hidden-size", "4",
        "--n-layers", "1",
        "--n-top-layers", "1",
        "--max-len", "4",
        "--epochs", "1",
        "--micro-batch", "2",
        *extra,
    ]


def test_main_defaults_every_input_path_into_the_downloaded_bundle(tmp_path):
    """The whole point of #149's FIX 1: inside `hf jobs uv run` the bundle
    lands at snapshot_download's cache path, so a submit line can only name
    --fold/--bundle-repo/--output-repo/--out-dir. Everything else must resolve
    itself out of the downloaded bundle."""
    bundle_dir = _stage_full_fake_bundle(tmp_path)
    out_dir = tmp_path / "fold0"
    seen = {}

    def recording_loader_factory(checkpoint_path, repo_root, model_config):
        seen["repo_root"] = Path(repo_root)
        seen["model_config"] = Path(model_config)
        return _fake_loader_factory(checkpoint_path, repo_root, model_config)

    exit_code = main(_bundle_only_args(bundle_dir, out_dir),
                     loader_factory=recording_loader_factory)

    assert exit_code == 0
    assert seen["repo_root"] == bundle_dir / "moonbeam_repo"
    assert seen["model_config"] == (
        bundle_dir / "moonbeam_repo" / "src" / "llama_recipes" / "configs"
        / "model_config.json")
    fold_data = read_fold_embeddings(out_dir / "emb_fold0.npz")
    assert fold_data["seg_ids"] == ["e0", "e1"]


def test_an_explicitly_passed_path_still_wins_over_the_bundle_default(tmp_path):
    """Local runs against loose files depend on this; a default that silently
    overrode an explicit flag would also be a very quiet way to train on the
    wrong fold plan."""
    bundle_dir = _stage_full_fake_bundle(tmp_path)
    other_plan = tmp_path / "other_fold_plans.json"
    other_plan.write_text(json.dumps([{
        "fold": 0,
        "test_seg_ids": ["e0"],
        "train_seg_ids": ["t0", "t1"],
        "val_seg_ids": [],
    }]))
    (bundle_dir / "eval_manifest.json").write_text(json.dumps([
        {"seg_id": "e0", "grade": 2, "composer_id": 0},
    ]))
    out_dir = tmp_path / "fold0_explicit"

    exit_code = main(
        _bundle_only_args(bundle_dir, out_dir,
                          extra=("--fold-plan", str(other_plan))),
        loader_factory=_fake_loader_factory)

    assert exit_code == 0
    assert read_fold_embeddings(out_dir / "emb_fold0.npz")["seg_ids"] == ["e0"]


def test_main_names_the_missing_bundle_file_when_a_default_does_not_exist(tmp_path):
    """A bundle staged by an older push_train_dataset.py has no
    eval_manifest.json. Dying here, naming the resolved path, beats dying a
    hundred lines later inside an unrelated read_text()."""
    bundle_dir = _stage_full_fake_bundle(tmp_path)
    (bundle_dir / "eval_manifest.json").unlink()

    with pytest.raises(FileNotFoundError, match="--eval-manifest"):
        main(_bundle_only_args(bundle_dir, tmp_path / "fold0"),
             loader_factory=_fake_loader_factory)


def test_main_downloads_the_checkpoint_when_none_is_given(tmp_path):
    """moonbeam_839M.pt is 1.6 GB, public, and deliberately not in the bundle
    -- a fresh job container has no local copy, so --checkpoint must be
    optional and fall back to a Hub download."""
    bundle_dir = _stage_full_fake_bundle(tmp_path)
    args = _bundle_only_args(bundle_dir, tmp_path / "fold0")
    checkpoint_idx = args.index("--checkpoint")
    del args[checkpoint_idx:checkpoint_idx + 2]
    downloads, seen = [], {}

    def fake_downloader(repo_id, filename):
        downloads.append((repo_id, filename))
        return tmp_path / "downloaded_moonbeam_839M.pt"

    def recording_loader_factory(checkpoint_path, repo_root, model_config):
        seen["checkpoint"] = Path(checkpoint_path)
        return _fake_loader_factory(checkpoint_path, repo_root, model_config)

    exit_code = main(args, loader_factory=recording_loader_factory,
                     checkpoint_downloader=fake_downloader)

    assert exit_code == 0
    assert downloads == [
        ("guozixunnicolas/moonbeam-midi-foundation-model", "moonbeam_839M.pt")]
    assert seen["checkpoint"] == tmp_path / "downloaded_moonbeam_839M.pt"


def test_main_does_not_download_a_checkpoint_that_was_passed_explicitly(tmp_path):
    bundle_dir = _stage_full_fake_bundle(tmp_path)
    downloads = []

    def fake_downloader(repo_id, filename):
        downloads.append((repo_id, filename))
        return tmp_path / "never_used.pt"

    exit_code = main(_bundle_only_args(bundle_dir, tmp_path / "fold0"),
                     loader_factory=_fake_loader_factory,
                     checkpoint_downloader=fake_downloader)

    assert exit_code == 0
    assert not downloads


def test_script_header_declares_every_dep_the_staged_bundle_code_imports():
    """`hf jobs uv run` builds the container from train_fold.py's `# /// script`
    header alone, but the code it imports at runtime comes from the BUNDLE's
    code/ dir (push_train_dataset._CODE_FILES). A module-scope import in one of
    those files that the header does not declare fails only inside a paid
    container, ~1 minute in, after the 6260-file bundle download.

    That is exactly how the first real pilot job died: bakeoff_cv.py does
    `from scipy import stats`, and the header listed numpy but not scipy.
    """
    import ast
    import re
    import sys
    from pathlib import Path

    from claim_measurement.difficulty import train_fold as train_fold_module
    from claim_measurement.difficulty.push_train_dataset import _CODE_FILES

    train_fold_path = Path(train_fold_module.__file__).resolve()
    module_dir = train_fold_path.parent
    block = re.search(r"# /// script\n(.*?)# ///",
                      train_fold_path.read_text(), re.DOTALL).group(1)
    declared = {re.split(r"[<>=!\[]", d)[0].strip().lower().replace("-", "_")
                for d in re.findall(r'"([^"]+)"', block)}
    # distribution name -> importable top-level module, where they differ
    declared |= {"sklearn" if d == "scikit_learn" else d for d in declared}

    for filename in _CODE_FILES:
        tree = ast.parse((module_dir / filename).read_text())
        # module scope only -- lazy imports inside functions are not container-fatal
        for node in tree.body:
            if isinstance(node, ast.Import):
                roots = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots = [node.module.split(".")[0]]
            else:
                continue
            for root in roots:
                if root in ("__future__",) or root in sys.stdlib_module_names:
                    continue
                assert root in declared, (
                    f"{filename} imports {root!r} at module scope, but train_fold.py's "
                    f"`# /// script` header does not declare it. The HF Jobs container "
                    f"would die on import. Declared: {sorted(declared)}")
