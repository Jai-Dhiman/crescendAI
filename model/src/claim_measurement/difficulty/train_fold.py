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

    hf jobs uv run --flavor a100-large --timeout 3h --secrets HF_TOKEN \\
        train_fold.py --fold 0 \\
        --bundle-repo Jai-D/phase1-lora-bundle \\
        --output-repo Jai-D/phase1-lora-fold0 --out-dir /data/fold0

`hf jobs uv run` uploads ONLY this file -- not the rest of this package, so
`--bundle-repo` (a `push_train_dataset.py`-staged HF dataset repo) is
snapshot_download'd at startup and its `code/` subdir is where `combined_loss`
and `tau_c` are imported from (never a second, vendored copy of either). Pass
`--bundle-dir` instead of `--bundle-repo` for a local/offline run against an
already-downloaded bundle.

The bundle's download path inside a job container is snapshot_download's cache
path, which cannot be known at submit time -- so `--fold-plan`,
`--pool-grades`, `--eval-manifest`, `--midi-dir`, `--repo-root` and
`--model-config` all DEFAULT to their location inside the downloaded bundle
(see `_bundle_default`). An explicitly-passed path still wins, which is what a
local run against loose files relies on. The 1.6 GB checkpoint is not in the
bundle at all: with no `--checkpoint`, it is fetched from `--checkpoint-repo` /
`--checkpoint-filename` on the Hub.

`--output-repo` uploads `adapter/` and `emb_fold{F}.npz` to a Hub model repo
after training -- without it, a job container's local disk (and the $13 of GPU
time that filled it) is discarded when the container exits.

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

# Hub coordinates, not filesystem paths: the checkpoint is public and is
# deliberately kept out of the training bundle (1.6 GB, bf16).
CHECKPOINT_REPO = "guozixunnicolas/moonbeam-midi-foundation-model"
CHECKPOINT_FILENAME = "moonbeam_839M.pt"

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


def _real_loader(checkpoint_path: Path, repo_root: Path, model_config: Path):
    """Build the real trainable MoonBeam model + tokenizer against the fork.
    Mirrors moonbeam_extract_script.py::_real_loader's checkpoint/tokenizer
    setup exactly (see that file for the three undocumented fork facts), but
    returns the OUTER LlamaForCausalLM itself (gradients flow; never called
    under torch.no_grad) plus a `tokenize(midi_path) -> LongTensor` callable,
    rather than a numpy-returning inference closure. Also returns
    config.max_len -- the SAME source moonbeam_extract_script.py reads its
    max_len from -- so main() can catch a --max-len that has drifted out of
    sync with the config instead of silently unpairing from frozen 0.8257."""
    import importlib.util

    repo_root = Path(repo_root)
    sys.path.insert(
        0, str(repo_root / "src" / "llama_recipes" / "transformers_minimal" / "src"))
    from transformers import LlamaConfig, LlamaForCausalLM

    spec = importlib.util.spec_from_file_location(
        "moonbeam_music_tokenizer",
        repo_root / "src" / "llama_recipes" / "datasets" / "music_tokenizer.py")
    music_tokenizer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(music_tokenizer)

    config = LlamaConfig.from_pretrained(model_config)
    if config._attn_implementation != "sdpa":
        raise ValueError(
            f"expected attn_implementation 'sdpa', got {config._attn_implementation!r}")
    model = LlamaForCausalLM(config)

    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = {(k[7:] if k.startswith("module.") else k): v
             for k, v in raw["model_state_dict"].items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"checkpoint does not match {model_config}: {len(missing)} missing / "
            f"{len(unexpected)} unexpected keys "
            f"(first missing: {missing[:5]}, first unexpected: {unexpected[:5]})")

    tokenizer = music_tokenizer.MusicTokenizer(
        timeshift_vocab_size=config.onset_vocab_size,
        dur_vocab_size=config.dur_vocab_size,
        octave_vocab_size=config.octave_vocab_size,
        pitch_class_vocab_size=config.pitch_class_vocab_size,
        instrument_vocab_size=config.instrument_vocab_size,
        velocity_vocab_size=config.velocity_vocab_size)

    def tokenize(midi_path: Path) -> torch.Tensor:
        compounds = tokenizer.midi_to_compound(str(midi_path))
        tokens = tokenizer.encode_series(compounds, if_add_sos=True, if_add_eos=True)
        return torch.tensor(tokens, dtype=torch.long)

    return model, tokenize, int(config.max_len)


def _score_head(hidden_size: int, n_levels: int) -> torch.nn.Module:
    """The trained-then-DISCARDED head: one linear layer producing a scalar
    ranking score plus n_levels-1 ordinal logits from a mean-pooled embedding."""
    return torch.nn.Linear(hidden_size, 1 + (n_levels - 1))


def _random_window(
    tokens: torch.Tensor, max_len: int, rng: np.random.Generator
) -> torch.Tensor:
    """One random contiguous max_len-token window (the whole sequence if it
    is already <= max_len). A deliberate crop augmentation at train time --
    see the design spec's "Train-time vs extract-time windowing"."""
    if len(tokens) <= max_len:
        return tokens
    start = int(rng.integers(0, len(tokens) - max_len + 1))
    return tokens[start:start + max_len]


def _mean_pool_window(
    transformer: torch.nn.Module, tokens: torch.Tensor
) -> torch.Tensor:
    x = tokens.unsqueeze(0)
    hidden = transformer(input_ids=x, position_ids=x, use_cache=False,
                          return_dict=True).last_hidden_state.squeeze(0)
    return hidden.mean(dim=0)


def _extract_full_piece(transformer: torch.nn.Module, tokens: torch.Tensor,
                         max_len: int) -> np.ndarray:
    """Extraction shaped like moonbeam_extract_script.py's forward pass
    (chunk to max_len, forward every chunk, concatenate, mean over ALL
    tokens), so the gate stays paired against frozen 0.8257. Deterministic
    ONLY if the caller has already put the model in eval mode -- LoRA
    dropout is active in train mode and would make this stochastic."""
    chunks = [tokens[i:i + max_len] for i in range(0, len(tokens), max_len)]
    with torch.no_grad():
        hidden = [
            transformer(input_ids=c.unsqueeze(0), position_ids=c.unsqueeze(0),
                        use_cache=False, return_dict=True).last_hidden_state.squeeze(0)
            for c in chunks]
    return torch.cat(hidden, dim=0).mean(dim=0).float().numpy()


def write_fold_embeddings(path: Path, seg_ids: list[str], embeddings: np.ndarray,
                           grades: np.ndarray, composer_ids: np.ndarray) -> None:
    """emb_fold{F}.npz: one bulk array file for ALL eval pieces (NOT the
    per-piece bakeoff_npz contract -- ft_eval.py needs one (900, hidden)
    matrix per fold, not 900 files per fold)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, seg_ids=np.array(seg_ids), embeddings=embeddings.astype(np.float32),
             grades=np.asarray(grades, dtype=np.int32),
             composer_ids=np.asarray(composer_ids, dtype=np.int32))


def read_fold_embeddings(path: Path) -> dict:
    with np.load(path) as z:
        return {
            "seg_ids": [str(s) for s in z["seg_ids"]],
            "embeddings": z["embeddings"],
            "grades": z["grades"],
            "composer_ids": z["composer_ids"],
        }


def _real_checkpoint_download(repo_id: str, filename: str) -> Path:
    """Fetch the MoonBeam checkpoint from the Hub. The 1.6 GB moonbeam_839M.pt
    is deliberately NOT staged into the training bundle (it is public, and
    re-uploading it per bundle would triple the bundle's size); a fresh HF Jobs
    container has no local copy of it either, so it is downloaded here."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=repo_id, filename=filename))


def _bundle_default(explicit: Path | None, bundle_dir: Path, relative: str,
                     flag: str) -> Path:
    """The value of a path flag that defaults into the downloaded bundle. An
    explicitly-passed path always wins (a local run against loose files needs
    that); otherwise the bundle-relative location must actually exist -- a
    missing default means the bundle was staged by an older
    push_train_dataset.py, and dying here with the resolved path named beats
    dying in an unrelated read_text() a hundred lines later."""
    if explicit is not None:
        return Path(explicit)
    resolved = bundle_dir / relative
    if not resolved.exists():
        raise FileNotFoundError(
            f"{flag} was not passed, so it defaults to {relative!r} inside the "
            f"bundle, but {resolved} does not exist -- re-stage and re-upload "
            f"the bundle with push_train_dataset.py, or pass {flag} explicitly")
    return resolved


def _real_uploader(out_dir: Path, repo_id: str) -> None:
    """Upload the trained adapter/ directory and emb_fold{F}.npz to a Hub
    model repo, so the artifacts survive the HF Jobs container's exit --
    without this, the $13 of GPU time that produced them is discarded."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(folder_path=str(out_dir), repo_id=repo_id, repo_type="model")


def main(argv: list[str] | None = None, loader_factory=_real_loader,
         uploader=_real_uploader,
         checkpoint_downloader=_real_checkpoint_download) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument(
        "--checkpoint", type=Path, default=None,
        help="local moonbeam_839M.pt. Omit inside a job container (there is no "
             "local copy there): it is then downloaded from --checkpoint-repo.")
    ap.add_argument("--checkpoint-repo", default=CHECKPOINT_REPO)
    ap.add_argument("--checkpoint-filename", default=CHECKPOINT_FILENAME)
    ap.add_argument("--repo-root", type=Path, default=None,
                    help="defaults to moonbeam_repo/ inside the bundle")
    ap.add_argument(
        "--model-config", type=Path, default=None,
        help="defaults to the bundle's "
             "moonbeam_repo/src/llama_recipes/configs/model_config.json "
             "(the 839M config; model_config_small.json is the 309M model)")
    ap.add_argument("--fold-plan", type=Path, default=None,
                    help="defaults to fold_plans.json inside the bundle")
    ap.add_argument(
        "--pool-grades", type=Path, default=None,
        help="JSON {seg_id: grade} covering every train/val seg_id in "
             "--fold-plan; defaults to grades.json inside the bundle")
    ap.add_argument(
        "--eval-manifest", type=Path, default=None,
        help="JSON list of {seg_id, grade, composer_id} for all 900 eval pieces, "
             "in the SAME seg_id-sorted order ft_eval.py reads from "
             "emb/features37/; defaults to eval_manifest.json inside the bundle")
    ap.add_argument("--midi-dir", type=Path, default=None,
                    help="defaults to midi/ inside the bundle")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--bundle-dir", type=Path, default=None,
        help="an already-downloaded push_train_dataset.py bundle (must contain a "
             "code/ subdir with ranking_loss.py + bakeoff_cv.py) -- skips the "
             "snapshot_download below. Required when --bundle-repo is not given "
             "(e.g. a local/offline run).")
    ap.add_argument(
        "--bundle-repo", default=None,
        help="the HF dataset repo id push_train_dataset.py uploaded the training "
             "bundle to; snapshot_download'd at startup when --bundle-dir is not "
             "given (this is how the code/ dir reaches the HF Jobs container, "
             "which `hf jobs uv run` never uploads on its own)")
    ap.add_argument(
        "--output-repo", default=None,
        help="HF Hub model repo id to upload adapter/ + emb_fold{F}.npz to after "
             "training. If omitted, artifacts are left on local/container disk "
             "only -- fine for a local run, but a job container discards them "
             "on exit, so this MUST be set for an `hf jobs uv run` invocation.")
    ap.add_argument("--hidden-size", type=int, default=1920)
    ap.add_argument("--n-layers", type=int, default=15)
    ap.add_argument("--n-top-layers", type=int, default=5)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ordinal-weight", type=float, default=0.1)
    ap.add_argument("--n-levels", type=int, default=11)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args(argv)
    torch.manual_seed(args.seed)

    if args.bundle_dir is not None:
        bundle_dir = Path(args.bundle_dir)
    elif args.bundle_repo:
        from huggingface_hub import snapshot_download
        bundle_dir = Path(
            snapshot_download(repo_id=args.bundle_repo, repo_type="dataset"))
    else:
        raise ValueError("either --bundle-dir or --bundle-repo is required, so "
                          "combined_loss/tau_c can be imported from the bundle's "
                          "code/ subdir (train_fold.py is uploaded alone by "
                          "`hf jobs uv run`, without the rest of this package)")
    fold_plan_path = _bundle_default(
        args.fold_plan, bundle_dir, "fold_plans.json", "--fold-plan")
    pool_grades_path = _bundle_default(
        args.pool_grades, bundle_dir, "grades.json", "--pool-grades")
    eval_manifest_path = _bundle_default(
        args.eval_manifest, bundle_dir, "eval_manifest.json", "--eval-manifest")
    midi_dir = _bundle_default(args.midi_dir, bundle_dir, "midi", "--midi-dir")
    repo_root = _bundle_default(
        args.repo_root, bundle_dir, "moonbeam_repo", "--repo-root")
    model_config = _bundle_default(
        args.model_config, bundle_dir,
        "moonbeam_repo/src/llama_recipes/configs/model_config.json",
        "--model-config")

    sys.path.insert(0, str(bundle_dir / "code"))
    import bakeoff_cv as _bakeoff_cv
    import ranking_loss as _ranking_loss
    tau_c = _bakeoff_cv.tau_c
    combined_loss = _ranking_loss.combined_loss

    plans = json.loads(fold_plan_path.read_text())
    plan = next(p for p in plans if p["fold"] == args.fold)
    pool_grades = json.loads(pool_grades_path.read_text())
    eval_pieces = json.loads(eval_manifest_path.read_text())

    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = Path(checkpoint_downloader(
            args.checkpoint_repo, args.checkpoint_filename))
        print(f"downloaded checkpoint "
              f"{args.checkpoint_repo}/{args.checkpoint_filename} -> {checkpoint}")

    base_model, tokenize, config_max_len = loader_factory(
        checkpoint, repo_root=repo_root, model_config=model_config)
    if config_max_len != args.max_len:
        raise ValueError(
            f"--max-len {args.max_len} does not match model_config.json's max_len "
            f"{config_max_len} -- this would silently unpair the fine-tuned arm's "
            f"extraction from frozen 0.8257")

    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=lora_target_modules(args.n_layers, args.n_top_layers))
    peft_model = get_peft_model(base_model, lora_config)
    transformer = peft_model.model.model  # inner transformer, LoRA-injected in place

    score_head = _score_head(args.hidden_size, args.n_levels)
    trainable_params = ([p for p in peft_model.parameters() if p.requires_grad]
                         + list(score_head.parameters()))
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    rng = np.random.default_rng(args.seed)
    train_seg_ids = list(plan["train_seg_ids"])
    val_seg_ids = list(plan["val_seg_ids"])

    for epoch in range(args.epochs):
        peft_model.train()
        order = rng.permutation(len(train_seg_ids))
        for start in range(0, len(order), args.micro_batch):
            batch_ids = [
                train_seg_ids[i] for i in order[start:start + args.micro_batch]]
            scores, ordinal_logits, grades = [], [], []
            for seg_id in batch_ids:
                tokens = tokenize(midi_dir / f"{seg_id}.mid")
                window = _random_window(tokens, args.max_len, rng)
                pooled = _mean_pool_window(transformer, window)
                head_out = score_head(pooled)
                scores.append(head_out[0])
                ordinal_logits.append(head_out[1:])
                grades.append(pool_grades[seg_id])
            scores_t = torch.stack(scores)
            ordinal_t = torch.stack(ordinal_logits)
            grades_t = torch.tensor(grades, dtype=torch.long)

            loss = combined_loss(
                scores_t, ordinal_t, grades_t, args.n_levels, args.ordinal_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        peft_model.eval()
        with torch.no_grad():
            val_scores = []
            for seg_id in val_seg_ids:
                tokens = tokenize(midi_dir / f"{seg_id}.mid")
                window = _random_window(tokens, args.max_len, rng)
                pooled = _mean_pool_window(transformer, window)
                val_scores.append(score_head(pooled)[0].item())
            val_grades = [pool_grades[seg_id] for seg_id in val_seg_ids]
            val_tau = tau_c(val_scores, val_grades) if val_seg_ids else None
        print(f"epoch {epoch}: val_ranking_tau={val_tau}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(out_dir / "adapter"))

    peft_model.eval()
    with torch.no_grad():
        embeddings = np.stack([
            _extract_full_piece(
                transformer, tokenize(midi_dir / f"{p['seg_id']}.mid"),
                args.max_len)
            for p in eval_pieces
        ])
    write_fold_embeddings(
        out_dir / f"emb_fold{args.fold}.npz",
        seg_ids=[p["seg_id"] for p in eval_pieces],
        embeddings=embeddings,
        grades=np.array([p["grade"] for p in eval_pieces]),
        composer_ids=np.array([p["composer_id"] for p in eval_pieces]),
    )
    print(f"fold {args.fold}: wrote adapter + emb_fold{args.fold}.npz for "
          f"{len(eval_pieces)} eval pieces")

    if args.output_repo is not None:
        uploader(out_dir, args.output_repo)
        print(f"fold {args.fold}: uploaded adapter + emb_fold{args.fold}.npz to "
              f"{args.output_repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
