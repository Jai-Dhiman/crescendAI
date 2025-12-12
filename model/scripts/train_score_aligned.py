#!/usr/bin/env python3
"""
Train the score-aligned piano performance evaluation model.

This model incorporates score alignment features to compare
performance MIDI against reference MusicXML scores.

Supports two encoder modes:
- Flat encoder: Standard transformer-based score encoder
- Hierarchical encoder: HAN-style note->beat->measure hierarchy (recommended)

Expected improvement: R-squared from 0.18 to 0.30-0.40 (based on PercePiano paper)

Usage:
    # Basic training with hierarchical encoder (recommended)
    python scripts/train_score_aligned.py --data-dir data/processed --score-dir data/scores --use-hierarchical

    # Training with flat encoder
    python scripts/train_score_aligned.py --data-dir data/processed --score-dir data/scores

For Thunder Compute:
    python scripts/train_score_aligned.py --data-dir data/processed --score-dir data/scores --gpus 1 --precision 16 --use-hierarchical
"""

import argparse
import sys
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.percepiano_score_dataset import create_score_dataloaders
from src.models.score_aligned_module import ScoreAlignedModule, ScoreAlignedModuleWithFallback


def main():
    parser = argparse.ArgumentParser(
        description="Train score-aligned piano performance evaluation model"
    )

    # Data arguments
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing processed PercePiano JSON files",
    )
    parser.add_argument(
        "--score-dir",
        type=Path,
        default=None,
        help="Directory containing MusicXML score files",
    )

    # Model arguments
    parser.add_argument(
        "--midi-hidden-dim",
        type=int,
        default=768,
        help="MIDI encoder hidden dimension (768 for MidiBERT)",
    )
    parser.add_argument(
        "--midi-num-layers",
        type=int,
        default=12,
        help="Number of MIDI encoder layers",
    )
    parser.add_argument(
        "--score-hidden-dim",
        type=int,
        default=256,
        help="Score encoder hidden dimension",
    )
    parser.add_argument(
        "--fusion-type",
        type=str,
        default="gated",
        choices=["concat", "crossattn", "gated"],
        help="Fusion method for combining MIDI and score features",
    )
    parser.add_argument(
        "--use-fallback",
        action="store_true",
        help="Train with fallback mode (handles missing scores)",
    )
    parser.add_argument(
        "--fallback-probability",
        type=float,
        default=0.2,
        help="Probability of training without score features",
    )
    parser.add_argument(
        "--freeze-midi-encoder",
        action="store_true",
        help="Freeze MIDI encoder initially (for faster convergence)",
    )
    parser.add_argument(
        "--use-hierarchical",
        action="store_true",
        help="Use hierarchical HAN encoder (recommended for better performance)",
    )
    parser.add_argument(
        "--score-note-features",
        type=int,
        default=20,
        help="Number of per-note score features (20 for expanded features)",
    )

    # Training arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=50,
        help="Maximum number of epochs",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="32",
        choices=["16", "32", "bf16"],
        help="Training precision",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loader workers",
    )
    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=1.0,
        help="Gradient clipping value",
    )
    parser.add_argument(
        "--accumulate-grad-batches",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )

    # Logging arguments
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="score_aligned",
        help="Experiment name for logging",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Use Weights & Biases logging",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory for saving checkpoints",
    )

    # Sequence length arguments
    parser.add_argument(
        "--max-midi-seq-length",
        type=int,
        default=1024,
        help="Maximum MIDI sequence length",
    )
    parser.add_argument(
        "--max-score-notes",
        type=int,
        default=1024,
        help="Maximum number of score notes",
    )

    args = parser.parse_args()

    # Validate paths
    if not args.data_dir.exists():
        print(f"Error: Data directory not found: {args.data_dir}")
        sys.exit(1)

    if args.score_dir and not args.score_dir.exists():
        print(f"Warning: Score directory not found: {args.score_dir}")
        print("Training will proceed without score alignment features")
        args.score_dir = None

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_score_dataloaders(
        data_dir=args.data_dir,
        score_dir=args.score_dir,
        batch_size=args.batch_size,
        max_midi_seq_length=args.max_midi_seq_length,
        max_score_notes=args.max_score_notes,
        num_workers=args.num_workers,
    )
    print(f"Train: {len(train_loader.dataset)} samples")
    print(f"Val: {len(val_loader.dataset)} samples")
    print(f"Test: {len(test_loader.dataset)} samples")

    # Create model
    print("Creating model...")
    ModelClass = ScoreAlignedModuleWithFallback if args.use_fallback else ScoreAlignedModule

    model_kwargs = {
        "midi_hidden_dim": args.midi_hidden_dim,
        "midi_num_layers": args.midi_num_layers,
        "score_hidden_dim": args.score_hidden_dim,
        "score_note_features": args.score_note_features,
        "fusion_type": args.fusion_type,
        "learning_rate": args.learning_rate,
        "freeze_midi_encoder": args.freeze_midi_encoder,
        "max_seq_length": args.max_midi_seq_length,
        "use_hierarchical_encoder": args.use_hierarchical,
    }

    if args.use_fallback:
        model_kwargs["fallback_probability"] = args.fallback_probability

    model = ModelClass(**model_kwargs)

    # Log encoder type
    encoder_type = "Hierarchical (HAN)" if args.use_hierarchical else "Flat (Transformer)"
    print(f"Score encoder: {encoder_type}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=args.checkpoint_dir / args.experiment_name,
            filename="best-{epoch:02d}-{val/mean_r2:.4f}",
            monitor="val/mean_r2",
            mode="max",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val/mean_r2",
            mode="max",
            patience=args.patience,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Create logger
    if args.use_wandb:
        logger = WandbLogger(
            project="crescendai-score-aligned",
            name=args.experiment_name,
            log_model=True,
        )
    else:
        logger = TensorBoardLogger(
            save_dir="logs",
            name=args.experiment_name,
        )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus if args.gpus > 0 else 1,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=args.gradient_clip,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=10,
        val_check_interval=0.5,  # Validate twice per epoch
    )

    # Train
    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)

    # Test
    print("Running final test...")
    test_results = trainer.test(model, test_loader, ckpt_path="best")

    # Print results
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    print(f"Best val R-squared: {trainer.checkpoint_callback.best_model_score:.4f}")

    if test_results:
        print("\nTest Results:")
        for key, value in test_results[0].items():
            print(f"  {key}: {value:.4f}")

    # Check tempo improvement specifically
    print("\n" + "-" * 60)
    print("Key metric to watch: val/tempo_r2")
    print("Target: Improve from R2=-0.15 to R2>0.10")
    print("-" * 60)


if __name__ == "__main__":
    main()
