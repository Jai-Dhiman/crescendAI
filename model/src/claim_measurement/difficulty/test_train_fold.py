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

import numpy as np
import torch

from claim_measurement.difficulty.train_fold import main, read_fold_embeddings


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

        return outer, tokenize

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
