# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub>=0.25.0"]
# ///
"""Aria 650M LoRA fine-tune as an HF Jobs A100 run (issue #74 runbook / issue #78 work).

This is the real-training counterpart to jobs/smoke_train.py. It is a self-bootstrapping
cloud entrypoint: `hf jobs uv run` hands it a bare env, so it clones the repo, installs
it, stages data from R2, then assembles and trains using the EXISTING repo building
blocks (it intentionally mirrors scripts/smoke_test_aria.py, the proven assembly):

    load_composite_labels -> AriaMidiPairDataset(midi_dir=<AMT MIDI>, ...) ->
    DataLoader(collate=aria_midi_collate_fn) -> AriaLoRAModel -> train_model(...)

Two persistence contracts, identical to the smoke job:
  - Trackio metrics sync to a persistent HF Space (TRACKIO_SPACE_ID).
  - The best checkpoint is uploaded to a HF Hub repo (CKPT_REPO).

GATING: requires #72 AMT MIDI (data/midi/amt/t1) and #73 score-MIDI coverage to exist
in R2, and the per-stream training head wiring finalized under #78. Launch via:
    uv run model/jobs/hf_launch.py aria --timeout 6h --detach \
        --trackio-space <user>/crescendai-trackio --ckpt-repo <user>/crescendai-aria-t1

Env (HF_TOKEN injected by the launcher's `--secrets HF_TOKEN`):
  REPO_REF        git ref to train from              (default main)
  AMT_MIDI_PREFIX R2 prefix of #72 AMT MIDI           (default midi/amt/t1)
  TRACKIO_SPACE_ID / CKPT_REPO / MAX_EPOCHS / FOLD_IDX / LORA_RANK
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/Jai-Dhiman/crescendAI.git"


def sh(cmd: list[str], cwd: str | None = None) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def bootstrap_repo() -> Path:
    """Clone + editable-install the repo so its training code is importable."""
    ref = os.environ.get("REPO_REF", "main")
    workdir = Path("/tmp/crescendai")
    if not workdir.exists():
        sh(["git", "clone", "--depth", "1", "--branch", ref, REPO_URL, str(workdir)])
    sh(["uv", "pip", "install", "--system", "-e", "."], cwd=str(workdir / "model"))
    sys.path.insert(0, str(workdir / "model" / "src"))
    return workdir / "model"


def stage_amt_midi(model_dir: Path) -> Path:
    """Pull #72 AMT MIDI from R2 into the expected local path (explicit, no fallback)."""
    prefix = os.environ.get("AMT_MIDI_PREFIX", "midi/amt/t1")
    dst = model_dir / "data" / "midi" / "amt" / "t1"
    dst.mkdir(parents=True, exist_ok=True)
    sh(["rclone", "copy", f"r2:crescendai-bucket/{prefix}", str(dst)])
    n = len(list(dst.glob("*.mid")))
    if n == 0:
        raise FileNotFoundError(
            f"No AMT MIDI staged from r2:crescendai-bucket/{prefix} -- run #72 to completion first."
        )
    print(f"[aria] staged {n} AMT MIDI files -> {dst}", flush=True)
    return dst


def main() -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    user = api.whoami()["name"]
    trackio_space = os.environ.get("TRACKIO_SPACE_ID", f"{user}/crescendai-trackio")
    ckpt_repo = os.environ.get("CKPT_REPO", f"{user}/crescendai-aria-t1")
    max_epochs = int(os.environ.get("MAX_EPOCHS", "10"))
    fold_idx = int(os.environ.get("FOLD_IDX", "0"))
    lora_rank = int(os.environ.get("LORA_RANK", "32"))

    model_dir = bootstrap_repo()
    amt_midi_dir = stage_amt_midi(model_dir)

    import json

    import torch

    from model_improvement.aria_encoder import (
        AriaLoRAModel,
        AriaMidiPairDataset,
        aria_midi_collate_fn,
    )
    from model_improvement.taxonomy import load_composite_labels
    from model_improvement.training import train_model
    from paths import Labels

    labels_raw = load_composite_labels(Labels.composite / "composite_labels.json")
    labels = {k: v.tolist() for k, v in labels_raw.items()}
    piece_mapping = json.loads((Labels.percepiano / "piece_mapping.json").read_text())
    folds = json.loads((Labels.percepiano / "folds.json").read_text())

    # Piece-stratified train/val from the verified clean folds (#73 leak fix).
    val_pieces = set(folds[fold_idx]["val"]) if isinstance(folds, list) else set(folds[str(fold_idx)]["val"])
    train_keys, val_keys = [], []
    for piece, segs in piece_mapping.items():
        (val_keys if piece in val_pieces else train_keys).extend(segs)

    def make_loader(keys: list[str], shuffle: bool):
        ds = AriaMidiPairDataset(
            midi_dir=amt_midi_dir, labels=labels, piece_to_keys=piece_mapping,
            keys=[k for k in keys if k in labels], max_seq_len=512,
        )
        if len(ds) == 0:
            raise RuntimeError(f"0 pairs from {amt_midi_dir} -- AMT MIDI keys must match label keys.")
        return torch.utils.data.DataLoader(
            ds, batch_size=8, shuffle=shuffle, collate_fn=aria_midi_collate_fn, num_workers=2,
        )

    train_loader, val_loader = make_loader(train_keys, True), make_loader(val_keys, False)
    model = AriaLoRAModel(learning_rate=1e-5, lora_rank=lora_rank, max_epochs=max_epochs, warmup_epochs=1)

    trainer = train_model(
        model=model, train_loader=train_loader, val_loader=val_loader,
        model_name="aria_lora_t1", fold_idx=fold_idx,
        checkpoint_dir=str(model_dir / "data" / "checkpoints"),
        max_epochs=max_epochs, trackio_experiment_id=f"aria-t1-fold{fold_idx}",
    )

    # Persist the best checkpoint to the Hub (the job filesystem is ephemeral).
    best = trainer.checkpoint_callback.best_model_path
    if best and Path(best).exists():
        api.create_repo(ckpt_repo, repo_type="model", exist_ok=True, private=True)
        api.upload_file(path_or_fileobj=best, path_in_repo=Path(best).name,
                        repo_id=ckpt_repo, repo_type="model")
        print(f"[aria] checkpoint -> https://huggingface.co/{ckpt_repo}/{Path(best).name}", flush=True)
    print(f"[aria] DONE. Trackio -> https://huggingface.co/spaces/{trackio_space}", flush=True)


if __name__ == "__main__":
    main()
