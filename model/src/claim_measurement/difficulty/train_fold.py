# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "numpy>=1.24.0", "torch>=2.0.0", "peft>=0.11.0", "trackio",
#     "mido", "music21", "pandas", "tqdm", "regex", "requests",
#     "filelock", "pyyaml", "safetensors", "tokenizers==0.19.1",
#     "huggingface_hub",
# ]
# ///
"""#138 Phase 1 LoRA fine-tune of MoonBeam-839M, one fold at a time. HF Jobs
entry point -- run under the SAME isolated uv-managed Python 3.12 venv as
moonbeam_extract_script.py (see that file's module docstring for the fork
clone/checkpoint setup). This file's own `# /// script` header restates torch
+ peft (already pinned in model/.venv, restated here because HF Jobs builds a
FRESH environment from this header, never model/.venv) plus the same
transformers_minimal-fork transitive deps moonbeam_extract_script.py needs,
plus trackio for telemetry.

    hf jobs uv run --flavor a100-large train_fold.py \\
        --fold 0 --checkpoint .../moonbeam_839M.pt --repo-root .../repo \\
        --model-config .../model_config.json \\
        --fold-plan .../fold_plans.json --pool-grades .../grades.json \\
        --eval-manifest .../eval_manifest.json \\
        --midi-dir .../transkun_mid --out-dir .../fold0

Only the encoder weights are graded (design spec's gate (i)): the score head
trained here is DISCARDED after training. `emb_fold{F}.npz` -- the only
artifact ft_eval.py reads -- holds MEAN-POOLED embeddings for ALL 900 eval
pieces (not just this fold's 180), extracted with the SAME full-piece,
no-window forward pass moonbeam_extract_script.py uses, so the gate stays
paired against frozen 0.8257. Training itself samples one random 1024-token
window per piece per step (a deliberate crop augmentation -- see the design
spec's "Train-time vs extract-time windowing"); only extraction is
window-free.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECTIONS = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
)


def lora_target_modules(n_layers: int, n_top: int) -> list[str]:
    """The LoRA target module names for the top n_top of n_layers MoonBeam
    decoder layers: self_attn.{q,k,v,o}_proj and mlp.{gate,up,down}_proj per
    layer, on checkpoint-matching names `model.layers.{L}.{...}`. Explicitly
    excludes decoder_embedding/summary_projection/lm_head/fc_out -- the
    fork's DEFAULT target_modules (src/llama_recipes/configs/peft.py:11),
    which target the generative decoder heads this design never invokes."""
    if n_top > n_layers:
        raise ValueError(f"n_top ({n_top}) cannot exceed n_layers ({n_layers})")
    return [f"model.layers.{layer}.{proj}"
            for layer in range(n_layers - n_top, n_layers) for proj in PROJECTIONS]


if __name__ == "__main__":
    sys.exit(0)  # placeholder exit; main() is added in Task 12
