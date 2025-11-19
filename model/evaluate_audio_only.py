#!/usr/bin/env python3
"""
Evaluate the audio-only model checkpoint.

Usage:
    python evaluate_audio_only.py --checkpoint /path/to/audio-epoch=02-val_loss=7.9935.ckpt
"""

import argparse
from pathlib import Path
import pytorch_lightning as pl
import torch

from src.models.lightning_module import PerformanceEvaluationModel
from src.data.dataset import create_dataloaders


def evaluate_audio_only_model(
    checkpoint_path: str,
    test_annotation_path: str = "/tmp/crescendai_data/data/annotations/synthetic_test.jsonl",
    batch_size: int = 16,
    num_workers: int = 4,
):
    """
    Evaluate the audio-only model on the test set.

    Args:
        checkpoint_path: Path to the model checkpoint
        test_annotation_path: Path to test annotations
        batch_size: Batch size for evaluation
        num_workers: Number of dataloader workers
    """
    print("="*70)
    print("AUDIO-ONLY MODEL EVALUATION")
    print("="*70)

    # Check checkpoint exists
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"\nCheckpoint: {checkpoint_path.name}")
    print(f"Test data: {test_annotation_path}")

    # Load model from checkpoint
    print("\nLoading model from checkpoint...")
    model = PerformanceEvaluationModel.load_from_checkpoint(str(checkpoint_path))
    model.eval()

    print(f"Model loaded successfully")
    print(f"  Audio dim: {model.hparams.audio_dim}")
    print(f"  MIDI dim: {model.hparams.midi_dim}")
    print(f"  Dimensions: {model.dimension_names}")

    # Create test dataloader
    print("\nCreating test dataloader...")
    _, _, test_loader = create_dataloaders(
        train_annotation_path=test_annotation_path,  # Dummy, not used
        val_annotation_path=test_annotation_path,    # Dummy, not used
        test_annotation_path=test_annotation_path,
        dimension_names=model.dimension_names,
        batch_size=batch_size,
        num_workers=num_workers,
        augmentation_config=None,  # No augmentation for testing
        audio_sample_rate=24000,
        max_audio_length=240000,
        max_midi_events=512,
    )

    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Test batches: {len(test_loader)}")

    # Create trainer for evaluation
    print("\nRunning evaluation...")
    trainer = pl.Trainer(
        accelerator='auto',
        devices='auto',
        precision=16,
        logger=False,
    )

    # Run test
    test_results = trainer.test(model, dataloaders=test_loader, verbose=True)

    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    results = test_results[0]

    # Print overall metrics
    print(f"\nOverall:")
    print(f"  Test Loss: {results.get('test_loss', 'N/A'):.4f}")

    # Print per-dimension metrics
    for dim in model.dimension_names:
        print(f"\n{dim}:")
        print(f"  MAE:              {results.get(f'test_mae_{dim}', 'N/A'):.4f}")
        print(f"  Pearson r:        {results.get(f'test_pearson_{dim}', 'N/A'):.4f}")
        print(f"  Spearman rho:     {results.get(f'test_spearman_{dim}', 'N/A'):.4f}")
        print(f"  Task Loss:        {results.get(f'test_task_loss_{dim}', 'N/A'):.4f}")

    # Calculate average correlations
    avg_pearson = sum(results.get(f'test_pearson_{dim}', 0) for dim in model.dimension_names) / len(model.dimension_names)
    avg_spearman = sum(results.get(f'test_spearman_{dim}', 0) for dim in model.dimension_names) / len(model.dimension_names)

    print(f"\nAverages:")
    print(f"  Avg Pearson:      {avg_pearson:.4f}")
    print(f"  Avg Spearman:     {avg_spearman:.4f}")

    print("\n" + "="*70)

    # Interpretation
    print("\nInterpretation:")
    if avg_pearson > 0.5:
        print("  ✓ Strong correlation - model performs well")
    elif avg_pearson > 0.3:
        print("  ~ Moderate correlation - model shows promise")
    else:
        print("  ✗ Weak correlation - model needs improvement")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate audio-only model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.ckpt file)",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="/tmp/crescendai_data/data/annotations/synthetic_test.jsonl",
        help="Path to test annotations (default: /tmp/crescendai_data/data/annotations/synthetic_test.jsonl)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for evaluation (default: 16)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers (default: 4)",
    )

    args = parser.parse_args()

    evaluate_audio_only_model(
        checkpoint_path=args.checkpoint,
        test_annotation_path=args.test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
