"""
Diagnostic experiments to validate model architecture and training setup.

These quick tests help identify fundamental issues before full training:
1. Single-batch overfitting - verify model has capacity to learn
2. MERT layer ablation - confirm layer 5-7 > layer 12
3. Label quality baseline - check if labels are informative
4. Per-dimension analysis - identify which dimensions are learnable
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import sys
import numpy as np
from scipy.stats import pearsonr

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import PerformanceDataset, collate_fn
from src.models.audio_encoder import MERTEncoder
from src.models.aggregation import HierarchicalAggregator
from src.models.mtl_head import MultiTaskHead


def single_batch_overfit_test(
    train_dataset,
    dimension_names,
    device='cuda',
    max_iterations=1000,
    target_loss=0.1
):
    """
    Test if model can overfit a single batch.

    If model can't achieve near-zero loss on 1 batch, architecture has issues.

    Args:
        train_dataset: Training dataset
        dimension_names: List of dimension names
        device: Device to use
        max_iterations: Maximum training iterations
        target_loss: Target loss to achieve

    Returns:
        Dict with results
    """
    print("\n" + "="*70)
    print("DIAGNOSTIC 1: Single-Batch Overfitting Test")
    print("="*70)
    print("Testing if model has capacity to learn...")
    print(f"Target: Loss < {target_loss}, Correlation > 0.95")

    # Create single-batch dataloader
    single_batch = Subset(train_dataset, range(min(16, len(train_dataset))))
    loader = DataLoader(single_batch, batch_size=16, collate_fn=collate_fn)
    batch = next(iter(loader))

    # Create simple model
    encoder = MERTEncoder(
        use_layer_selection=True,
        selected_layers=(5, 6, 7),
        freeze_bottom_layers=False
    ).to(device)

    aggregator = HierarchicalAggregator(
        input_dim=768,
        lstm_hidden=256,
        lstm_layers=2,
        attention_heads=4,
        dropout=0.2,
        output_dim=512
    ).to(device)

    head = MultiTaskHead(
        input_dim=512,
        shared_hidden=256,
        task_hidden=128,
        dimensions=dimension_names,
        dropout=0.1
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) +
        list(aggregator.parameters()) +
        list(head.parameters()),
        lr=1e-3  # Higher LR for faster overfitting
    )

    criterion = nn.MSELoss()

    # Move batch to device
    audio = batch['audio_waveform'].to(device)
    labels = batch['labels'].to(device)

    # Training loop
    losses = []
    correlations = []

    for iteration in range(max_iterations):
        optimizer.zero_grad()

        # Forward pass
        embeddings, _ = encoder(audio)
        aggregated, _ = aggregator(embeddings)  # aggregator returns (output, attention)
        predictions, _ = head(aggregated)

        # Loss
        loss = criterion(predictions, labels)

        # Backward
        loss.backward()
        optimizer.step()

        # Track metrics
        losses.append(loss.item())

        # Compute correlation every 100 iterations
        if iteration % 100 == 0:
            with torch.no_grad():
                pred_np = predictions.cpu().numpy()
                label_np = labels.cpu().numpy()

                # Average correlation across dimensions
                corrs = []
                for dim_idx in range(len(dimension_names)):
                    if pred_np[:, dim_idx].std() > 1e-6:  # Check for variance
                        corr, _ = pearsonr(pred_np[:, dim_idx], label_np[:, dim_idx])
                        corrs.append(corr)

                avg_corr = np.mean(corrs) if corrs else 0.0
                correlations.append(avg_corr)

                print(f"Iteration {iteration:4d} | Loss: {loss.item():.4f} | Avg Correlation: {avg_corr:.4f}")

        # Early stopping if target reached
        if loss.item() < target_loss:
            print(f"\nTarget loss reached at iteration {iteration}!")
            break

    # Final evaluation
    with torch.no_grad():
        embeddings, _ = encoder(audio)
        aggregated, _ = aggregator(embeddings)  # aggregator returns (output, attention)
        predictions, _ = head(aggregated)
        final_loss = criterion(predictions, labels).item()

        pred_np = predictions.cpu().numpy()
        label_np = labels.cpu().numpy()

        final_corrs = []
        for dim_idx, dim_name in enumerate(dimension_names):
            if pred_np[:, dim_idx].std() > 1e-6:
                corr, _ = pearsonr(pred_np[:, dim_idx], label_np[:, dim_idx])
                final_corrs.append(corr)
                print(f"{dim_name}: {corr:.4f}")

        avg_final_corr = np.mean(final_corrs) if final_corrs else 0.0

    print(f"\nFinal Loss: {final_loss:.4f}")
    print(f"Final Avg Correlation: {avg_final_corr:.4f}")

    # Verdict
    success = final_loss < target_loss and avg_final_corr > 0.95
    print("\nVERDICT:", "PASS" if success else "FAIL")
    if not success:
        print("Model cannot overfit single batch - architecture issue likely!")

    return {
        'success': success,
        'final_loss': final_loss,
        'final_correlation': avg_final_corr,
        'losses': losses,
        'correlations': correlations
    }


def layer_ablation_test(
    train_dataset,
    dimension_names,
    device='cuda',
    num_epochs=1,
    batch_size=16
):
    """
    Compare different MERT layer selections.

    Tests:
    - Layer 12 (last) - current default
    - Layers 5-7 (middle) - research recommendation
    - Layer 4, 6, 8 (multi-scale) - advanced option

    Args:
        train_dataset: Training dataset
        dimension_names: List of dimension names
        device: Device to use
        num_epochs: Number of epochs to train
        batch_size: Batch size

    Returns:
        Dict with results for each configuration
    """
    print("\n" + "="*70)
    print("DIAGNOSTIC 2: MERT Layer Ablation Test")
    print("="*70)
    print("Comparing layer selections...")

    # Test configurations
    configs = [
        ('Layer 12 (last)', False, (12,)),
        ('Layers 5-7 (middle)', True, (5, 6, 7)),
        ('Layers 4,6,8 (multi-scale)', True, (4, 6, 8)),
    ]

    # Create small subset for quick test
    subset_size = min(500, len(train_dataset))
    subset = Subset(train_dataset, range(subset_size))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    results = {}

    for config_name, use_selection, layers in configs:
        print(f"\nTesting: {config_name}")
        print("-" * 50)

        # Create model
        encoder = MERTEncoder(
            use_layer_selection=use_selection,
            selected_layers=layers if use_selection else None,
            freeze_bottom_layers=False
        ).to(device)

        aggregator = HierarchicalAggregator(
            input_dim=768,
            lstm_hidden=256,
            lstm_layers=2,
            attention_heads=4,
            dropout=0.2,
            output_dim=512
        ).to(device)

        head = MultiTaskHead(
            input_dim=512,
            shared_hidden=256,
            task_hidden=128,
            dimensions=dimension_names
        ).to(device)

        optimizer = torch.optim.AdamW(
            list(encoder.parameters()) +
            list(aggregator.parameters()) +
            list(head.parameters()),
            lr=3e-5
        )

        criterion = nn.MSELoss()

        # Train for 1 epoch
        total_loss = 0
        num_batches = 0

        encoder.train()
        aggregator.train()
        head.train()

        for batch in loader:
            audio = batch['audio_waveform'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            embeddings, _ = encoder(audio)
            aggregated, _ = aggregator(embeddings)  # aggregator returns (output, attention)
            predictions, _ = head(aggregated)

            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Average Loss: {avg_loss:.4f}")

        results[config_name] = {
            'avg_loss': avg_loss,
            'layers': layers
        }

    # Compare results
    print("\n" + "="*70)
    print("LAYER ABLATION RESULTS")
    print("="*70)

    best_config = min(results.items(), key=lambda x: x[1]['avg_loss'])

    for config_name, metrics in results.items():
        print(f"{config_name}: Loss = {metrics['avg_loss']:.4f}")

    print(f"\nBest Configuration: {best_config[0]}")
    print(f"Expected: Layers 5-7 should outperform Layer 12 by 15-20%")

    return results


def label_quality_baseline(train_dataset, dimension_names):
    """
    Check if labels have signal (not just noise).

    Computes correlation of predicting mean vs actual labels.
    If correlation > 0.2, labels may be too noisy.

    Args:
        train_dataset: Training dataset
        dimension_names: List of dimension names

    Returns:
        Dict with baseline correlations
    """
    print("\n" + "="*70)
    print("DIAGNOSTIC 3: Label Quality Baseline")
    print("="*70)
    print("Testing if labels are informative...")

    # Collect all labels
    loader = DataLoader(train_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

    all_labels = []
    for batch in loader:
        all_labels.append(batch['labels'])

    all_labels = torch.cat(all_labels, dim=0).numpy()

    # Compute mean for each dimension
    means = all_labels.mean(axis=0)
    stds = all_labels.std(axis=0)

    print("\nLabel Statistics:")
    for dim_idx, dim_name in enumerate(dimension_names):
        print(f"{dim_name}: mean={means[dim_idx]:.2f}, std={stds[dim_idx]:.2f}")

    # Predict mean for all samples
    predictions = np.tile(means, (len(all_labels), 1))

    # Compute correlation
    baseline_corrs = []
    print("\nBaseline Correlations (predicting mean):")
    for dim_idx, dim_name in enumerate(dimension_names):
        if stds[dim_idx] > 1e-6:
            corr, _ = pearsonr(predictions[:, dim_idx], all_labels[:, dim_idx])
            baseline_corrs.append(corr)
            print(f"{dim_name}: {corr:.4f}")
        else:
            print(f"{dim_name}: No variance (constant)")

    avg_baseline = np.mean(baseline_corrs) if baseline_corrs else 0.0
    print(f"\nAverage Baseline Correlation: {avg_baseline:.4f}")

    # Verdict
    if avg_baseline > 0.2:
        print("\nWARNING: High baseline correlation suggests labels may be noisy!")
        print("Consider reviewing label generation process.")
    elif avg_baseline < 0.05:
        print("\nGOOD: Low baseline correlation - labels have signal!")

    return {
        'baseline_correlations': baseline_corrs,
        'avg_baseline': avg_baseline,
        'means': means,
        'stds': stds
    }


def run_all_diagnostics(
    annotation_path: str,
    dimension_names: list,
    device: str = 'cuda'
):
    """
    Run all diagnostic experiments.

    Args:
        annotation_path: Path to training annotations
        dimension_names: List of dimension names
        device: Device to use

    Returns:
        Dict with all results
    """
    print("\n" + "="*70)
    print("RUNNING DIAGNOSTIC EXPERIMENTS")
    print("="*70)

    # Create dataset
    dataset = PerformanceDataset(
        annotation_path=annotation_path,
        dimension_names=dimension_names,
        audio_sample_rate=24000,
        max_audio_length=240000,
        apply_augmentation=False
    )

    print(f"\nDataset: {len(dataset)} samples")
    print(f"Dimensions: {dimension_names}")

    results = {}

    # Run diagnostics
    results['label_quality'] = label_quality_baseline(dataset, dimension_names)
    results['single_batch_overfit'] = single_batch_overfit_test(dataset, dimension_names, device)
    results['layer_ablation'] = layer_ablation_test(dataset, dimension_names, device)

    # Summary
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)

    print(f"\n1. Label Quality: {'GOOD' if results['label_quality']['avg_baseline'] < 0.2 else 'CONCERNING'}")
    print(f"   Baseline correlation: {results['label_quality']['avg_baseline']:.4f}")

    print(f"\n2. Model Capacity: {'PASS' if results['single_batch_overfit']['success'] else 'FAIL'}")
    print(f"   Can overfit single batch: {results['single_batch_overfit']['success']}")

    print(f"\n3. Layer Selection: See detailed results above")

    return results


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description='Run diagnostic experiments')
    parser.add_argument('--train-path', type=str, required=True,
                       help='Path to training annotations')
    parser.add_argument('--dimensions', type=str, nargs='+',
                       default=['note_accuracy', 'rhythmic_precision', 'tone_quality'],
                       help='Dimension names')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')

    args = parser.parse_args()

    results = run_all_diagnostics(
        annotation_path=args.train_path,
        dimension_names=args.dimensions,
        device=args.device
    )

    print("\nDiagnostics complete!")
