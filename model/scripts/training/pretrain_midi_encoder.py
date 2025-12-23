#!/usr/bin/env python3
"""
MIDI Encoder Pre-training on GiantMIDI-Piano.

Pre-trains MIDIBertEncoder using masked MIDI prediction task:
- 15% random masking per OctupleMIDI dimension
- Separate reconstruction loss per dimension
- Weighted by vocabulary size (pitch gets higher weight)

Supports both GiantMIDI-Piano and MAESTRO MIDI directories.

GiantMIDI-Piano (recommended):
    - Download from: https://github.com/bytedance/GiantMIDI-Piano
    - Size: 193MB (10,855 MIDI files)
    - Directory structure: GiantMIDI-Piano/midis_v1.2/

MAESTRO (alternative):
    - Download from: https://magenta.tensorflow.org/datasets/maestro
    - Smaller dataset, already have if using MAESTRO for training

Usage:
    # Pre-train on GiantMIDI-Piano (recommended)
    python scripts/pretrain_midi_encoder.py \
        --midi_dir data/GiantMIDI-Piano/midis_v1.2 \
        --output_dir checkpoints/midi_pretrain \
        --epochs 30 \
        --batch_size 64

    # Resume from checkpoint
    python scripts/pretrain_midi_encoder.py \
        --midi_dir data/GiantMIDI-Piano/midis_v1.2 \
        --output_dir checkpoints/midi_pretrain \
        --resume checkpoints/midi_pretrain/latest.pt

Expected training time:
    - GiantMIDI-Piano (10,855 files): ~8-12 GPU hours on A100
    - MAESTRO (~1,200 files): ~2-4 GPU hours on A100
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.crescendai.data.midi_processing import OctupleMIDITokenizer, load_midi
from src.crescendai.models.midi_encoder import MIDIBertEncoder

# Mask token ID (use vocab_size for each dimension as mask token)
MASK_TOKEN_OFFSET = 1  # Add to vocab_size to get mask token ID


class MIDIPretrainDataset(Dataset):
    """
    Dataset for MIDI pre-training using masked prediction.

    Loads MIDI files, tokenizes with OctupleMIDI, and creates
    masked sequences for self-supervised learning.
    """

    def __init__(
        self,
        midi_paths: List[Path],
        max_seq_length: int = 512,
        mask_prob: float = 0.15,
        tokenizer: Optional[OctupleMIDITokenizer] = None,
    ):
        """
        Initialize pre-training dataset.

        Args:
            midi_paths: List of paths to MIDI files
            max_seq_length: Maximum sequence length
            mask_prob: Probability of masking each token (default 15%)
            tokenizer: OctupleMIDI tokenizer instance
        """
        self.midi_paths = midi_paths
        self.max_seq_length = max_seq_length
        self.mask_prob = mask_prob
        self.tokenizer = tokenizer or OctupleMIDITokenizer()

        # Vocabulary sizes for masking
        self.vocab_sizes = self.tokenizer.vocab_sizes

    def __len__(self) -> int:
        return len(self.midi_paths)

    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        Load and tokenize a MIDI file, then apply masking.

        Returns:
            Dict with:
                - input_tokens: Masked tokens [seq_len, 8]
                - target_tokens: Original tokens [seq_len, 8]
                - mask: Boolean mask indicating masked positions [seq_len, 8]
        """
        midi_path = self.midi_paths[idx]

        try:
            # Load MIDI
            midi = load_midi(str(midi_path))

            # Tokenize
            tokens = self.tokenizer.encode(midi)

            if len(tokens) == 0:
                return None

            # Truncate or pad to max_seq_length
            if len(tokens) > self.max_seq_length:
                # Random crop for variety during training
                start_idx = random.randint(0, len(tokens) - self.max_seq_length)
                tokens = tokens[start_idx : start_idx + self.max_seq_length]
            elif len(tokens) < self.max_seq_length:
                # Pad with zeros
                padding = np.zeros(
                    (self.max_seq_length - len(tokens), 8), dtype=np.int64
                )
                tokens = np.concatenate([tokens, padding], axis=0)

            # Create mask (True = masked)
            mask = np.random.random((self.max_seq_length, 8)) < self.mask_prob

            # Don't mask padding tokens (where all dimensions are 0)
            is_padding = (tokens == 0).all(axis=1)
            mask[is_padding] = False

            # Store original tokens as targets
            target_tokens = tokens.copy()

            # Apply masking: replace masked tokens with mask token ID
            input_tokens = tokens.copy()
            for dim_idx, dim_name in enumerate(
                [
                    "event_type",
                    "beat",
                    "position",
                    "pitch",
                    "duration",
                    "velocity",
                    "instrument",
                    "bar",
                ]
            ):
                vocab_size = self.vocab_sizes[dim_name]
                # Use 0 as mask token (will be ignored in loss for padding)
                # Actually, we'll use a special strategy: replace with random token 80%,
                # keep original 10%, use mask token 10% (BERT-style)
                dim_mask = mask[:, dim_idx]

                # 80% random replacement
                random_replace_mask = dim_mask & (
                    np.random.random(self.max_seq_length) < 0.8
                )
                input_tokens[random_replace_mask, dim_idx] = np.random.randint(
                    0, vocab_size, size=random_replace_mask.sum()
                )

                # 10% keep original (already done by copy)
                # 10% use special mask token (vocab_size, will be handled in model)

            return {
                "input_tokens": torch.from_numpy(input_tokens).long(),
                "target_tokens": torch.from_numpy(target_tokens).long(),
                "mask": torch.from_numpy(mask).bool(),
            }

        except Exception as e:
            print(f"Error loading {midi_path}: {e}")
            return None


def collate_pretrain_batch(
    batch: List[Optional[Dict[str, torch.Tensor]]],
) -> Optional[Dict[str, torch.Tensor]]:
    """Collate function that filters out None samples."""
    # Filter out None samples
    batch = [b for b in batch if b is not None]

    if len(batch) == 0:
        return None

    return {
        "input_tokens": torch.stack([b["input_tokens"] for b in batch]),
        "target_tokens": torch.stack([b["target_tokens"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
    }


class MIDIPretrainModel(nn.Module):
    """
    MIDI pre-training model with masked prediction heads.

    Wraps MIDIBertEncoder and adds prediction heads for each
    OctupleMIDI dimension.
    """

    def __init__(
        self,
        encoder: MIDIBertEncoder,
        vocab_sizes: Dict[str, int],
    ):
        """
        Initialize pre-training model.

        Args:
            encoder: MIDIBertEncoder to pre-train
            vocab_sizes: Vocabulary sizes for each dimension
        """
        super().__init__()

        self.encoder = encoder
        self.vocab_sizes = vocab_sizes
        self.hidden_size = encoder.hidden_size

        # Prediction heads for each dimension
        self.prediction_heads = nn.ModuleDict(
            {
                "event_type": nn.Linear(self.hidden_size, vocab_sizes["event_type"]),
                "beat": nn.Linear(self.hidden_size, vocab_sizes["beat"]),
                "position": nn.Linear(self.hidden_size, vocab_sizes["position"]),
                "pitch": nn.Linear(self.hidden_size, vocab_sizes["pitch"]),
                "duration": nn.Linear(self.hidden_size, vocab_sizes["duration"]),
                "velocity": nn.Linear(self.hidden_size, vocab_sizes["velocity"]),
                "instrument": nn.Linear(self.hidden_size, vocab_sizes["instrument"]),
                "bar": nn.Linear(self.hidden_size, vocab_sizes["bar"]),
            }
        )

        # Loss weights based on vocabulary size (larger vocab = harder task)
        total_vocab = sum(vocab_sizes.values())
        self.loss_weights = {
            dim: vocab_sizes[dim] / total_vocab * len(vocab_sizes)
            for dim in vocab_sizes
        }
        # Give extra weight to pitch (most musically important)
        self.loss_weights["pitch"] *= 2.0

    def forward(
        self,
        input_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with loss computation.

        Args:
            input_tokens: Masked input tokens [batch, seq_len, 8]
            target_tokens: Original tokens [batch, seq_len, 8]
            mask: Boolean mask [batch, seq_len, 8]

        Returns:
            Dict with total_loss and per-dimension losses
        """
        # Encode
        hidden = self.encoder(input_tokens)  # [batch, seq_len, hidden_size]

        # Compute losses for each dimension
        losses = {}
        total_loss = 0.0

        dim_names = [
            "event_type",
            "beat",
            "position",
            "pitch",
            "duration",
            "velocity",
            "instrument",
            "bar",
        ]

        for dim_idx, dim_name in enumerate(dim_names):
            # Get predictions for this dimension
            logits = self.prediction_heads[dim_name](
                hidden
            )  # [batch, seq_len, vocab_size]

            # Get targets and mask for this dimension
            targets = target_tokens[:, :, dim_idx]  # [batch, seq_len]
            dim_mask = mask[:, :, dim_idx]  # [batch, seq_len]

            # Only compute loss on masked positions
            if dim_mask.any():
                # Flatten for cross entropy
                masked_logits = logits[dim_mask]  # [num_masked, vocab_size]
                masked_targets = targets[dim_mask]  # [num_masked]

                # Cross entropy loss
                loss = F.cross_entropy(masked_logits, masked_targets, reduction="mean")
                losses[dim_name] = loss

                # Weighted contribution to total loss
                total_loss = total_loss + self.loss_weights[dim_name] * loss
            else:
                losses[dim_name] = torch.tensor(0.0, device=input_tokens.device)

        losses["total"] = total_loss

        return losses


def find_midi_files(midi_dir: Path, limit: Optional[int] = None) -> List[Path]:
    """
    Find all MIDI files in directory.

    Args:
        midi_dir: Directory containing MIDI files
        limit: Optional limit on number of files

    Returns:
        List of MIDI file paths
    """
    midi_files = []

    # Search for .mid and .midi files
    for pattern in ["**/*.mid", "**/*.midi", "**/*.MID", "**/*.MIDI"]:
        midi_files.extend(midi_dir.glob(pattern))

    # Remove duplicates and sort
    midi_files = sorted(set(midi_files))

    if limit is not None and len(midi_files) > limit:
        # Random sample for faster iteration
        random.shuffle(midi_files)
        midi_files = midi_files[:limit]

    return midi_files


def train_epoch(
    model: MIDIPretrainModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_losses = {
        dim: 0.0
        for dim in [
            "event_type",
            "beat",
            "position",
            "pitch",
            "duration",
            "velocity",
            "instrument",
            "bar",
            "total",
        ]
    }
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        if batch is None:
            continue

        # Move to device
        input_tokens = batch["input_tokens"].to(device)
        target_tokens = batch["target_tokens"].to(device)
        mask = batch["mask"].to(device)

        # Forward pass
        optimizer.zero_grad()
        losses = model(input_tokens, target_tokens, mask)

        # Backward pass
        losses["total"].backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Accumulate losses
        for dim in total_losses:
            total_losses[dim] += losses[dim].item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix(
            {
                "loss": f"{losses['total'].item():.4f}",
                "pitch": f"{losses['pitch'].item():.4f}",
            }
        )

    # Average losses
    avg_losses = {
        dim: total / max(num_batches, 1) for dim, total in total_losses.items()
    }
    return avg_losses


def validate(
    model: MIDIPretrainModel,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Validate model."""
    model.eval()

    total_losses = {
        dim: 0.0
        for dim in [
            "event_type",
            "beat",
            "position",
            "pitch",
            "duration",
            "velocity",
            "instrument",
            "bar",
            "total",
        ]
    }
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            if batch is None:
                continue

            input_tokens = batch["input_tokens"].to(device)
            target_tokens = batch["target_tokens"].to(device)
            mask = batch["mask"].to(device)

            losses = model(input_tokens, target_tokens, mask)

            for dim in total_losses:
                total_losses[dim] += losses[dim].item()
            num_batches += 1

    avg_losses = {
        dim: total / max(num_batches, 1) for dim, total in total_losses.items()
    }
    return avg_losses


def main():
    parser = argparse.ArgumentParser(
        description="Pre-train MIDI encoder on GiantMIDI-Piano or MAESTRO"
    )
    parser.add_argument(
        "--midi_dir", type=str, required=True, help="Directory containing MIDI files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/midi_pretrain",
        help="Output directory",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size (reduced for larger model)",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps")
    parser.add_argument(
        "--max_seq_length", type=int, default=512, help="Maximum sequence length"
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=768,
        help="Hidden size (768 to match MidiBERT)",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=12,
        help="Number of transformer layers (12 to match MidiBERT)",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=12,
        help="Number of attention heads (12 to match MidiBERT)",
    )
    parser.add_argument(
        "--mask_prob", type=float, default=0.15, help="Masking probability"
    )
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split")
    parser.add_argument(
        "--limit_files",
        type=int,
        default=None,
        help="Limit number of files (for debugging)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data workers"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find MIDI files
    midi_dir = Path(args.midi_dir)
    midi_files = find_midi_files(midi_dir, limit=args.limit_files)
    print(f"Found {len(midi_files)} MIDI files")

    if len(midi_files) == 0:
        print("Error: No MIDI files found!")
        return

    # Split into train/val
    random.shuffle(midi_files)
    val_size = int(len(midi_files) * args.val_split)
    val_files = midi_files[:val_size]
    train_files = midi_files[val_size:]
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")

    # Create tokenizer
    tokenizer = OctupleMIDITokenizer()

    # Create datasets
    train_dataset = MIDIPretrainDataset(
        midi_paths=train_files,
        max_seq_length=args.max_seq_length,
        mask_prob=args.mask_prob,
        tokenizer=tokenizer,
    )

    val_dataset = MIDIPretrainDataset(
        midi_paths=val_files,
        max_seq_length=args.max_seq_length,
        mask_prob=args.mask_prob,
        tokenizer=tokenizer,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_pretrain_batch,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_pretrain_batch,
        pin_memory=True,
    )

    # Create encoder (using args for all dimensions)
    encoder = MIDIBertEncoder(
        vocab_sizes=tokenizer.vocab_sizes,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=0.1,
        max_seq_length=args.max_seq_length,
    )

    # Create pre-training model
    model = MIDIPretrainModel(
        encoder=encoder,
        vocab_sizes=tokenizer.vocab_sizes,
    ).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    # Create scheduler (linear warmup + cosine decay)
    num_training_steps = len(train_loader) * args.epochs

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / args.warmup_steps
        else:
            progress = (step - args.warmup_steps) / (
                num_training_steps - args.warmup_steps
            )
            return max(0.1, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume from checkpoint if specified
    start_epoch = 1
    best_val_loss = float("inf")
    training_log = []

    if args.resume is not None:
        resume_path = Path(args.resume)
        if resume_path.exists():
            print(f"Resuming from checkpoint: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_loss = checkpoint.get("val_losses", {}).get("total", float("inf"))
            print(f"  Resuming from epoch {start_epoch}")
            print(f"  Previous best val loss: {best_val_loss:.4f}")
        else:
            print(f"Warning: Resume checkpoint not found: {resume_path}")

    # Training loop

    print("\nStarting pre-training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Mask probability: {args.mask_prob}")
    print()

    for epoch in range(start_epoch, args.epochs + 1):
        # Train
        train_losses = train_epoch(model, train_loader, optimizer, device, epoch)

        # Validate
        val_losses = validate(model, val_loader, device)

        # Update scheduler
        scheduler.step()

        # Log
        log_entry = {
            "epoch": epoch,
            "train_loss": train_losses["total"],
            "val_loss": val_losses["total"],
            "train_losses": train_losses,
            "val_losses": val_losses,
            "lr": optimizer.param_groups[0]["lr"],
        }
        training_log.append(log_entry)

        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train loss: {train_losses['total']:.4f}")
        print(f"  Val loss: {val_losses['total']:.4f}")
        print(
            f"  Pitch loss: train={train_losses['pitch']:.4f}, val={val_losses['pitch']:.4f}"
        )
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "encoder_state_dict": encoder.state_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "args": vars(args),
        }

        # Save latest
        torch.save(checkpoint, output_dir / "latest.pt")

        # Save best
        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            torch.save(checkpoint, output_dir / "best.pt")
            print(f"  New best model saved! (val_loss={val_losses['total']:.4f})")

        # Save periodic checkpoints
        if epoch % 5 == 0:
            torch.save(checkpoint, output_dir / f"epoch_{epoch:03d}.pt")

    # Save final encoder weights only (for fine-tuning)
    torch.save(encoder.state_dict(), output_dir / "encoder_pretrained.pt")

    # Save training log
    with open(output_dir / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)

    print("\nPre-training complete!")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Checkpoints saved to: {output_dir}")
    print(f"  Encoder weights: {output_dir / 'encoder_pretrained.pt'}")


if __name__ == "__main__":
    main()
