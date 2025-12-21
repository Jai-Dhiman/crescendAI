#!/usr/bin/env python3
"""
Contrastive pre-training script for MERT-MIDIBert alignment.

This script trains projection heads to align MERT audio embeddings with
MIDIBert MIDI embeddings using InfoNCE contrastive loss with hard negative mining.

Purpose (from TRAINING_PLAN_v2.md Phase 3):
- Align MERT and MIDIBert representation spaces
- Train projection heads with InfoNCE loss on paired audio-MIDI samples
- Success criterion: Cross-modal alignment score > 0.6

Usage:
    python scripts/pretrain_contrastive.py \
        --train_path /tmp/maestro_data/annotations/train.jsonl \
        --val_path /tmp/maestro_data/annotations/val.jsonl \
        --output_dir /tmp/checkpoints/contrastive \
        --epochs 15 \
        --batch_size 64

Reference: CLaMP3, MuLan, CLIP for cross-modal contrastive learning.
"""

import argparse
import os
import gc
import json
from pathlib import Path
from datetime import datetime

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.crescendai.models.lightning_module import PerformanceEvaluationModel
from src.crescendai.data.dataset import create_contrastive_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description='Contrastive pre-training for MERT-MIDIBert alignment')

    # Data paths
    parser.add_argument('--train_path', type=str, required=True,
                       help='Path to training annotations JSONL')
    parser.add_argument('--val_path', type=str, required=True,
                       help='Path to validation annotations JSONL')

    # Output
    parser.add_argument('--output_dir', type=str, default='/tmp/checkpoints/contrastive',
                       help='Directory to save checkpoints')
    parser.add_argument('--gdrive_output', type=str, default=None,
                       help='Google Drive path for syncing (e.g., gdrive:crescendai_checkpoints/contrastive)')

    # Training
    parser.add_argument('--epochs', type=int, default=15,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size (larger is better for contrastive learning)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate for projection heads')
    parser.add_argument('--warmup_epochs', type=int, default=2,
                       help='Number of warmup epochs')
    parser.add_argument('--temperature', type=float, default=0.07,
                       help='Temperature for InfoNCE loss')

    # Hard negative mining
    parser.add_argument('--use_hard_negatives', action='store_true', default=True,
                       help='Use hard negative mining (same piece, different degradation)')
    parser.add_argument('--hard_neg_ratio', type=float, default=0.25,
                       help='Fraction of batch that should be hard negatives')

    # Model
    parser.add_argument('--audio_dim', type=int, default=768,
                       help='Audio encoder dimension (768 for MERT-95M)')
    parser.add_argument('--midi_dim', type=int, default=256,
                       help='MIDI encoder dimension (256 for MIDIBert)')
    parser.add_argument('--shared_dim', type=int, default=512,
                       help='Shared projection space dimension')
    parser.add_argument('--freeze_encoders', action='store_true', default=True,
                       help='Freeze encoders and only train projection heads')
    parser.add_argument('--midi_pretrained', type=str, default=None,
                       help='Path to pretrained MIDIBert checkpoint')

    # Hardware
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--precision', type=str, default='16-mixed',
                       choices=['32', '16-mixed', 'bf16-mixed'],
                       help='Training precision')

    # Dimensions (needed for dataset but not used in contrastive training)
    parser.add_argument('--dimensions', type=str, nargs='+',
                       default=['note_accuracy', 'rhythmic_stability', 'articulation_clarity',
                               'pedal_technique', 'tone_quality', 'dynamic_range',
                               'musical_expression', 'overall_interpretation'],
                       help='Evaluation dimension names')

    # Success criteria
    parser.add_argument('--alignment_target', type=float, default=0.6,
                       help='Target alignment score for Phase 3 gate check')

    return parser.parse_args()


def main():
    args = parse_args()

    print("="*70)
    print("CONTRASTIVE PRE-TRAINING: MERT-MIDIBert Alignment")
    print("="*70)
    print(f"Train path: {args.train_path}")
    print(f"Val path: {args.val_path}")
    print(f"Output dir: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Temperature: {args.temperature}")
    print(f"Hard negatives: {args.use_hard_negatives} (ratio: {args.hard_neg_ratio})")
    print(f"Freeze encoders: {args.freeze_encoders}")
    print(f"Alignment target: {args.alignment_target}")
    print("="*70)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create dataloaders with hard negative mining
    print("\nCreating dataloaders...")
    train_loader = create_contrastive_dataloader(
        annotation_path=args.train_path,
        dimension_names=args.dimensions,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_hard_negatives=args.use_hard_negatives,
        hard_neg_ratio=args.hard_neg_ratio,
    )

    val_loader = create_contrastive_dataloader(
        annotation_path=args.val_path,
        dimension_names=args.dimensions,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_hard_negatives=False,  # No hard negatives for validation
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Create model in contrastive training mode
    print("\nCreating model in contrastive mode...")
    model = PerformanceEvaluationModel(
        audio_dim=args.audio_dim,
        midi_dim=args.midi_dim,
        shared_dim=args.shared_dim,
        training_mode="contrastive",
        modality="fusion",
        fusion_type="gated",  # Fusion type doesn't matter in contrastive mode
        use_projection=True,
        freeze_audio_encoder=args.freeze_encoders,
        gradient_checkpointing=True,
        midi_pretrained_checkpoint=args.midi_pretrained,
        contrastive_temperature=args.temperature,
        contrastive_weight=1.0,  # Only contrastive loss in this mode
        learning_rate=args.learning_rate,
        backbone_lr=args.learning_rate * 0.1 if not args.freeze_encoders else 0,
        heads_lr=args.learning_rate,
        warmup_steps=len(train_loader) * args.warmup_epochs,
        max_epochs=args.epochs,
    )

    if args.freeze_encoders:
        print("Freezing encoders, training only projection heads...")
        # Freeze MERT
        if model.audio_encoder is not None:
            for param in model.audio_encoder.parameters():
                param.requires_grad = False
        # Freeze MIDIBert
        if model.midi_encoder is not None:
            for param in model.midi_encoder.parameters():
                param.requires_grad = False

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=str(output_dir),
            filename='contrastive-{epoch:02d}-{val_alignment_score:.4f}',
            monitor='val_alignment_score',
            mode='max',  # Higher alignment is better
            save_top_k=1,
            save_last=True,
        ),
        EarlyStopping(
            monitor='val_alignment_score',
            patience=5,
            mode='max',
        ),
        LearningRateMonitor(logging_interval='step'),
    ]

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        precision=args.precision,
        accelerator='auto',
        devices='auto',
        callbacks=callbacks,
        logger=TensorBoardLogger(save_dir=str(output_dir), name='logs'),
        log_every_n_steps=50,
        gradient_clip_val=1.0,
        val_check_interval=0.5,
    )

    # Train
    print("\nStarting contrastive pre-training...")
    trainer.fit(model, train_loader, val_loader)

    # Get best results
    best_alignment = callbacks[0].best_model_score
    best_checkpoint = callbacks[0].best_model_path

    print("\n" + "="*70)
    print("CONTRASTIVE PRE-TRAINING COMPLETE")
    print("="*70)
    print(f"Best alignment score: {best_alignment:.4f}")
    print(f"Best checkpoint: {best_checkpoint}")

    # Phase 3 gate check
    print("\n" + "="*70)
    print("PHASE 3 GATE CHECK (TRAINING_PLAN_v2.md)")
    print("="*70)
    print(f"Target alignment: >= {args.alignment_target}")
    print(f"Achieved alignment: {best_alignment:.4f}")

    if best_alignment >= args.alignment_target:
        print("\nPASS: Cross-modal alignment achieved target")
        print("-> GO TO PHASE 4: Full training with aligned encoders")
        gate_passed = True
    else:
        print("\nFAIL: Cross-modal alignment below target")
        print("-> Consider: More epochs, different temperature, or architecture changes")
        gate_passed = False

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'args': vars(args),
        'best_alignment_score': float(best_alignment) if best_alignment is not None else None,
        'best_checkpoint': best_checkpoint,
        'gate_passed': gate_passed,
        'trainable_params': trainable_params,
        'total_params': total_params,
    }

    results_path = output_dir / 'contrastive_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Sync to Google Drive if specified
    if args.gdrive_output:
        print(f"\nSyncing to Google Drive: {args.gdrive_output}")
        os.system(f"rclone copy {output_dir} {args.gdrive_output} --progress")
        print("Sync complete!")

    # Cleanup
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    return gate_passed


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
