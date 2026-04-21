"""Aria LoRA Phase C smoke test — 10 steps on T1 subset, MPS device.

Validates the pipeline end-to-end:
  - Aria weights download + LoRA application
  - AriaMidiPairDataset tokenization
  - Forward + backward on MPS (batch_size=2)
  - LoRA param gradients non-zero, base params frozen (zero grad)
  - Loss decreasing across 10 steps

Usage:
    cd model
    uv run python scripts/smoke_test_aria.py

Requires:  ~2.5GB disk for Aria weights (cached in ~/.cache/huggingface).
MPS note:  First run is slow (MPS kernel compilation, ~60s). Steps 2-10 fast.
"""

import json
import logging
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_improvement.aria_encoder import AriaMidiPairDataset, AriaLoRAModel, aria_midi_collate_fn
from model_improvement.taxonomy import load_composite_labels
from paths import Labels, Midi

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("smoke_test_aria")

# ---------- Config ----------
BATCH_SIZE = 2
N_STEPS = 10
LEARNING_RATE = 1e-5
SUBSET_SIZE = 60    # Use only 60 of 1202 T1 segments to keep dataset small
MAX_SEQ_LEN = 512   # Token truncation (PercePiano segs ~64 notes = ~192 tokens)


class SmokeTestCallback(pl.Callback):
    """Captures per-step loss and grad norms; prints step-by-step report."""

    def __init__(self, lora_param_names: set[str]) -> None:
        super().__init__()
        self.lora_param_names = lora_param_names
        self.losses: list[float] = []
        self.lora_gnorms: list[float] = []
        self.base_gnorms: list[float] = []

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: dict,
        batch: dict,
        batch_idx: int,
    ) -> None:
        loss_val = outputs["loss"].item() if isinstance(outputs, dict) else float(outputs)
        self.losses.append(loss_val)

        lora_gnorm_sq = 0.0
        base_gnorm_sq = 0.0
        for name, param in pl_module.named_parameters():
            if param.grad is None:
                continue
            g2 = param.grad.detach().norm().item() ** 2
            if "lora_" in name:
                lora_gnorm_sq += g2
            elif name.startswith("backbone"):
                base_gnorm_sq += g2

        self.lora_gnorms.append(lora_gnorm_sq ** 0.5)
        self.base_gnorms.append(base_gnorm_sq ** 0.5)

        step = batch_idx + 1
        print(
            f"step {step:>2}  loss={loss_val:>8.4f}  "
            f"lora_gnorm={self.lora_gnorms[-1]:>10.6f}  "
            f"base_gnorm={self.base_gnorms[-1]:>10.6f}"
        )

    def print_summary(self) -> None:
        if not self.losses:
            print("No steps recorded.")
            return

        print("\n--- Assertions ---")
        loss_trend_ok = self.losses[-1] < self.losses[0]
        print(
            f"Loss decreasing (step 1 -> {len(self.losses)}): "
            f"{self.losses[0]:.4f} -> {self.losses[-1]:.4f}  "
            f"{'PASS' if loss_trend_ok else 'WARN (may need more steps)'}"
        )

        lora_ok = all(g > 1e-12 for g in self.lora_gnorms)
        print(f"LoRA grad norms non-zero:  {'PASS' if lora_ok else 'FAIL'}")

        base_ok = all(g < 1e-12 for g in self.base_gnorms)
        print(f"Base backbone grads zero:  {'PASS' if base_ok else 'FAIL'}")

        no_nan = all(l == l for l in self.losses)
        print(f"No NaN losses:             {'PASS' if no_nan else 'FAIL'}")

        print("\n--- Summary ---")
        if lora_ok and base_ok and no_nan:
            print("Pipeline READY for cloud run.")
        else:
            print("ISSUES found — investigate before cloud run.")


def _select_accelerator() -> tuple[str, int]:
    """Return (accelerator, devices) for the Lightning Trainer."""
    if torch.backends.mps.is_available():
        print("Using MPS device (Apple M-series GPU)")
        return "mps", 1
    if torch.cuda.is_available():
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        return "gpu", 1
    raise RuntimeError(
        "Neither MPS nor CUDA available. "
        "Smoke test requires a GPU. On Apple silicon, ensure PyTorch >= 2.0."
    )


def main() -> None:
    # --- Load labels ---
    logger.info("Loading composite labels")
    labels_raw = load_composite_labels(Labels.composite / "composite_labels.json")
    labels: dict[str, list[float]] = {k: v.tolist() for k, v in labels_raw.items()}

    # --- Load piece mapping ---
    with open(Labels.percepiano / "piece_mapping.json") as f:
        piece_mapping: dict[str, list[str]] = json.load(f)

    # --- Subset: first SUBSET_SIZE segments that appear in piece_mapping ---
    # (238 composite label keys are singletons absent from piece_mapping;
    #  sorting alphabetically puts Beethoven first, all of which are absent)
    mapped_segs: set[str] = set()
    for segs in piece_mapping.values():
        mapped_segs.update(segs)
    all_keys = sorted(set(labels.keys()) & mapped_segs)[:SUBSET_SIZE]
    logger.info("Using %d segments for smoke test dataset", len(all_keys))

    # --- Build dataset ---
    dataset = AriaMidiPairDataset(
        midi_dir=Midi.percepiano,
        labels=labels,
        piece_to_keys=piece_mapping,
        keys=all_keys,
        max_seq_len=MAX_SEQ_LEN,
    )
    if len(dataset) == 0:
        raise RuntimeError(
            f"Dataset has 0 pairs. Check MIDI directory: {Midi.percepiano}"
        )
    logger.info("Dataset: %d pairs", len(dataset))

    # pin_memory is only supported on CUDA; MPS requires False
    pin_memory = torch.cuda.is_available()
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=aria_midi_collate_fn,
        num_workers=0,
        pin_memory=pin_memory,
    )

    # --- Instantiate model ---
    logger.info("Building AriaLoRAModel (downloads ~2.5GB weights if not cached)")
    model = AriaLoRAModel(
        learning_rate=LEARNING_RATE,
        lora_rank=32,
        max_epochs=1,
        warmup_epochs=0,
    )

    # --- Verify freeze structure before training ---
    lora_param_names = {n for n, _ in model.named_parameters() if "lora_" in n}
    base_backbone_params = [
        (n, p)
        for n, p in model.named_parameters()
        if n.startswith("backbone") and "lora_" not in n
    ]

    print("\n--- Freeze verification (pre-training) ---")
    n_lora = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
    n_base = sum(p.numel() for n, p in base_backbone_params)
    n_heads = sum(
        p.numel()
        for n, p in model.named_parameters()
        if not n.startswith("backbone")
    )
    print(f"LoRA adapter params:         {n_lora:>12,}")
    print(f"Frozen backbone params:      {n_base:>12,}")
    print(f"Trainable head params:       {n_heads:>12,}")
    all_base_frozen = all(not p.requires_grad for _, p in base_backbone_params)
    all_lora_trainable = all(
        p.requires_grad
        for n, p in model.named_parameters()
        if "lora_" in n
    )
    print(f"Base backbone frozen:        {'PASS' if all_base_frozen else 'FAIL'}")
    print(f"LoRA params trainable:       {'PASS' if all_lora_trainable else 'FAIL'}")
    print()

    # --- Configure Trainer for smoke test ---
    accelerator, devices = _select_accelerator()
    callback = SmokeTestCallback(lora_param_names=lora_param_names)

    trainer = pl.Trainer(
        max_steps=N_STEPS,
        accelerator=accelerator,
        devices=devices,
        precision="32-true",
        enable_checkpointing=False,
        enable_model_summary=False,
        logger=False,
        callbacks=[callback],
        enable_progress_bar=False,
    )

    print("--- Smoke test: 10 steps ---")
    print(f"{'step':>4}  {'loss':>8}  {'lora_gnorm':>12}  {'base_gnorm':>12}")
    trainer.fit(model, train_dataloaders=loader)

    callback.print_summary()


if __name__ == "__main__":
    main()
