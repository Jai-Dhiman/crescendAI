# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "numpy>=1.24.0",
#     "torch>=2.0.0",
#     # Transitive deps of the fork's VENDORED transformers + MusicTokenizer.
#     # The fork itself is NOT a pip dependency: it has no [project] table and
#     # is put on sys.path from --repo-root instead (see _real_loader).
#     "mido",
#     "music21",          # imported at module scope by the fork's modeling_llama
#     "pandas",           # imported at module scope by the fork's music_tokenizer
#     "tqdm",
#     "regex",
#     "requests",
#     "filelock",
#     "pyyaml",
#     "safetensors",
#     "tokenizers==0.19.1",
#     "huggingface_hub",
# ]
# ///
"""MoonBeam-839M extraction, run under an ISOLATED uv-managed Python 3.12 venv
-- NEVER the shared model/.venv (this repo has twice polluted that shared venv
with a competing pretraining stack's pinned deps; see project memory
project_uv_run_mutates_model_venv.md: "uv run --with X --python N" from
inside model/ rebuilds the shared .venv).

SETUP (run once). Fetch the fork at its pinned commit and the checkpoint:
    W=model/data/weights/moonbeam
    git clone https://github.com/guozixunnicolas/moonbeam-midi-foundation-model $W/repo
    git -C $W/repo checkout 4e2c015c89ae44a9542a7e9a67f9d7098f487ef1
    hf download guozixunnicolas/moonbeam-midi-foundation-model moonbeam_839M.pt --local-dir $W
    # moonbeam_839M.pt is 1.6GB (bf16), not the 3.3GB the Phase 0 design assumed.

`uv run --script` resolves THIS file's own `# /// script` metadata block into
its own cached, ephemeral env keyed to python==3.12.* + the deps above --
never the project's model/.venv. That is different from a bare `uv run`
invoked from inside model/, or `uv run --with X`, both of which DO sync the
shared project venv (the known gotcha above).

RUN (CPU is enough -- one 839M forward pass over a ~1000-event piece is ~2s):
    cd model/src/claim_measurement/difficulty
    uv run --script moonbeam_extract_script.py \
        --checkpoint /path/to/model/data/weights/moonbeam/moonbeam_839M.pt \
        --repo-root /path/to/model/data/weights/moonbeam/repo \
        --model-config /path/to/model/data/weights/moonbeam/repo/src/llama_recipes/configs/model_config.json \
        --sample-manifest /path/to/model/data/results/bakeoff/sample_manifest.json \
        --midi-dir /path/to/model/data/results/amt_gap_curve/transkun_mid \
        --out-dir /path/to/model/data/results/bakeoff/emb/moonbeam \
        --composer-index /path/to/model/data/results/bakeoff/composer_index.json

model_config.json is the 839M config (hidden 1920, 15 layers); the checkpoint
loads against it with zero missing and zero unexpected keys. model_config_small.json
is the 309M model -- using it would silently mismatch, which _real_loader refuses.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np


def _real_loader(checkpoint_path: Path, repo_root: Path, model_config: Path) -> Callable[[Path], np.ndarray]:
    """Build the real per-token-hidden-state loader against MoonBeam's fork.

    Verified against moonbeam_839M.pt at fork commit 4e2c015 (see the module
    docstring). Imports of the fork are deferred to inside this function so
    importing this MODULE never requires the isolated venv.

    Three things about the fork that are not in its README and cost a spike to
    find, so do not re-derive them:

    1. The fork VENDORS its transformers under
       src/llama_recipes/transformers_minimal/src. It has no [project] table,
       so it cannot be pip-installed -- it must be put on sys.path FIRST so
       `import transformers` resolves to the fork, not upstream transformers.
    2. Tokens are 6-dim compounds (onset, dur, octave, pitch_class,
       instrument, velocity), and for attn_implementation == "sdpa" the fork's
       own LlamaForCausalLM.forward sets `position_ids = input_ids` -- the
       compound token IS the positional signal (FME). We replicate that line
       when calling the inner LlamaModel directly.
    3. MusicTokenizer.midi_to_compound goes straight from a MIDI path to
       compounds, so data_preprocess.py and its .npy intermediate are not
       needed at all.
    """
    import importlib.util

    import torch

    repo_root = Path(repo_root)
    sys.path.insert(0, str(repo_root / "src" / "llama_recipes" / "transformers_minimal" / "src"))
    from transformers import LlamaConfig, LlamaForCausalLM

    # Load music_tokenizer by file, NOT via `llama_recipes.datasets`: that
    # package's __init__ imports HuggingFace `datasets`, which this extraction
    # does not need and which the isolated venv does not install.
    spec = importlib.util.spec_from_file_location(
        "moonbeam_music_tokenizer",
        repo_root / "src" / "llama_recipes" / "datasets" / "music_tokenizer.py")
    music_tokenizer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(music_tokenizer)

    config = LlamaConfig.from_pretrained(model_config)
    if config._attn_implementation != "sdpa":
        raise ValueError(
            f"expected attn_implementation 'sdpa' (the position_ids = input_ids "
            f"contract this loader replicates), got {config._attn_implementation!r}")
    model = LlamaForCausalLM(config)

    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = {(k[7:] if k.startswith("module.") else k): v
             for k, v in raw["model_state_dict"].items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        # The fork loads with strict=False, which would silently leave layers
        # randomly initialized. A bake-off measured on a half-loaded backbone
        # is worse than no bake-off, so refuse instead.
        raise ValueError(
            f"checkpoint does not match {model_config}: "
            f"{len(missing)} missing / {len(unexpected)} unexpected keys "
            f"(first missing: {missing[:5]}, first unexpected: {unexpected[:5]})")
    model.eval()

    tokenizer = music_tokenizer.MusicTokenizer(
        timeshift_vocab_size=config.onset_vocab_size, dur_vocab_size=config.dur_vocab_size,
        octave_vocab_size=config.octave_vocab_size,
        pitch_class_vocab_size=config.pitch_class_vocab_size,
        instrument_vocab_size=config.instrument_vocab_size,
        velocity_vocab_size=config.velocity_vocab_size)
    max_len = int(config.max_len)

    def load_hidden_states(midi_path: Path) -> np.ndarray:
        compounds = tokenizer.midi_to_compound(str(midi_path))
        tokens = tokenizer.encode_series(compounds, if_add_sos=True, if_add_eos=True)
        x = torch.tensor(tokens, dtype=torch.long)
        # Chunk to max_len and concatenate, rather than truncating to the first
        # window: MoonBeam must see the whole piece because the Aria arm it is
        # being compared against averages over chunks covering the whole piece.
        # Truncation would have handed Aria a coverage advantage that has
        # nothing to do with backbone quality.
        chunks = [x[i:i + max_len] for i in range(0, len(x), max_len)]
        with torch.no_grad():
            hidden = [model.model(input_ids=c.unsqueeze(0), position_ids=c.unsqueeze(0),
                                   use_cache=False, return_dict=True).last_hidden_state.squeeze(0)
                      for c in chunks]
        return torch.cat(hidden, dim=0).float().numpy()

    return load_hidden_states


def main(argv: list[str] | None = None, loader_factory=_real_loader) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # -> model/src
    from claim_measurement.difficulty.bakeoff_sampling import ManifestEntry
    from claim_measurement.difficulty.extract import extract_embeddings
    from claim_measurement.difficulty.moonbeam_backbone import MoonBeamBackbone

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--sample-manifest", type=Path, required=True)
    ap.add_argument("--midi-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--composer-index", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, required=True,
                    help="clone of guozixunnicolas/moonbeam-midi-foundation-model (pin 4e2c015)")
    ap.add_argument("--model-config", type=Path, required=True,
                    help="the fork's src/llama_recipes/configs/model_config.json (the 839M config)")
    args = ap.parse_args(argv)

    entries = [ManifestEntry(**e) for e in json.loads(args.sample_manifest.read_text())]
    loader = loader_factory(args.checkpoint, repo_root=args.repo_root, model_config=args.model_config)
    backbone = MoonBeamBackbone(loader=loader)
    report = extract_embeddings(backbone, entries, midi_dir=args.midi_dir,
                                 out_dir=args.out_dir, composer_index_path=args.composer_index)
    print(f"ok={report.ok} skipped={report.skipped} failed={len(report.failed)}")
    for f in report.failed[:10]:
        print(f"  FAIL {f}")
    return 0 if not report.failed else 1


if __name__ == "__main__":
    sys.exit(main())
