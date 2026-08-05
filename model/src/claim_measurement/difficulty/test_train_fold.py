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
