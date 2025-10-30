#!/usr/bin/env python3
"""
Evaluate a trained checkpoint on validation set.

Computes:
- Per-dimension Pearson/Spearman correlations
- Per-dimension MAE
- Prediction statistics (mean, std, range)
- Prediction diversity analysis
"""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import pytorch_lightning as pl

from src.models.lightning_module import PerformanceEvaluationModel
from src.data.dataset import create_dataloaders


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def evaluate_model(model, dataloader, device, dimension_names):
    """Evaluate model on a dataloader."""
    model.eval()

    all_predictions = []
    all_targets = []

    print(f"\nEvaluating on {len(dataloader)} batches...")

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(dataloader)} batches...")

            audio_waveform = batch['audio_waveform'].to(device)
            midi_tokens = batch.get('midi_tokens', None)
            if midi_tokens is not None:
                midi_tokens = midi_tokens.to(device)
            labels = batch['labels'].to(device)

            output = model.forward(
                audio_waveform=audio_waveform,
                midi_tokens=midi_tokens,
            )

            predictions = output['scores']

            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    return all_predictions, all_targets


def compute_metrics(predictions, targets, dimension_names):
    """Compute per-dimension metrics."""
    num_dims = predictions.shape[1]

    results = {}

    for i, dim_name in enumerate(dimension_names):
        pred = predictions[:, i]
        target = targets[:, i]

        # Remove any NaN or inf values
        valid_mask = np.isfinite(pred) & np.isfinite(target)
        pred = pred[valid_mask]
        target = target[valid_mask]

        # Compute metrics
        mae = np.abs(pred - target).mean()

        # Correlations (only if there's variance)
        if pred.std() > 0.01 and target.std() > 0.01:
            pearson_r, pearson_p = pearsonr(pred, target)
            spearman_r, spearman_p = spearmanr(pred, target)
        else:
            pearson_r, pearson_p = 0.0, 1.0
            spearman_r, spearman_p = 0.0, 1.0

        results[dim_name] = {
            'mae': mae,
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'pred_mean': pred.mean(),
            'pred_std': pred.std(),
            'target_mean': target.mean(),
            'target_std': target.std(),
            'pred_min': pred.min(),
            'pred_max': pred.max(),
        }

    return results


def analyze_prediction_diversity(predictions, dimension_names):
    """Analyze whether predictions are diverse or near-constant."""
    print("\n" + "=" * 80)
    print("PREDICTION DIVERSITY ANALYSIS")
    print("=" * 80)

    for i, dim_name in enumerate(dimension_names):
        pred = predictions[:, i]

        # Compute coefficient of variation (std/mean)
        cv = pred.std() / pred.mean() if pred.mean() > 0 else 0

        # Compute range
        range_val = pred.max() - pred.min()

        # Compute unique values (rounded to 2 decimals)
        unique_values = len(np.unique(np.round(pred, 2)))

        print(f"\n{dim_name}:")
        print(f"  Mean: {pred.mean():.2f}")
        print(f"  Std: {pred.std():.2f}")
        print(f"  CV: {cv:.3f}")
        print(f"  Range: [{pred.min():.2f}, {pred.max():.2f}] (span={range_val:.2f})")
        print(f"  Unique values: {unique_values}/{len(pred)}")

        # Flag issues
        if pred.std() < 1.0:
            print(f"  WARNING: Very low std ({pred.std():.3f}) - predictions near-constant!")
        if range_val < 5.0:
            print(f"  WARNING: Very small range ({range_val:.2f}) - model may have collapsed!")
        if unique_values < 10:
            print(f"  WARNING: Very few unique values ({unique_values}) - model output may be stuck!")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")

    # Set seed
    pl.seed_everything(config.get("seed", 42))

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_annotation_path=config["data"]["train_path"],
        val_annotation_path=config["data"]["val_path"],
        test_annotation_path=config["data"].get("test_path", None),
        dimension_names=config["data"]["dimensions"],
        batch_size=config["data"]["batch_size"],
        num_workers=0,
        augmentation_config=None,  # No augmentation for evaluation
        audio_sample_rate=config["data"]["audio_sample_rate"],
        max_audio_length=config["data"]["max_audio_length"],
        max_midi_events=config["data"]["max_midi_events"],
    )

    # Load model
    print(f"\nLoading model from checkpoint: {args.checkpoint}")
    model = PerformanceEvaluationModel.load_from_checkpoint(args.checkpoint)

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(f"Device: {device}")

    # Evaluate on validation set
    print("\n" + "=" * 80)
    print("VALIDATION SET EVALUATION")
    print("=" * 80)

    predictions, targets = evaluate_model(model, val_loader, device, config["data"]["dimensions"])

    print(f"\nTotal samples: {len(predictions)}")

    # Compute metrics
    results = compute_metrics(predictions, targets, config["data"]["dimensions"])

    # Print results
    print("\n" + "=" * 80)
    print("PER-DIMENSION RESULTS")
    print("=" * 80)

    print(f"\n{'Dimension':<25} {'MAE':<10} {'Pearson r':<12} {'Spearman r':<12}")
    print("-" * 80)
    for dim_name in config["data"]["dimensions"]:
        r = results[dim_name]
        print(f"{dim_name:<25} {r['mae']:<10.2f} {r['pearson_r']:<12.3f} {r['spearman_r']:<12.3f}")

    # Detailed results
    print("\n" + "=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)

    for dim_name in config["data"]["dimensions"]:
        r = results[dim_name]
        print(f"\n{dim_name}:")
        print(f"  MAE: {r['mae']:.2f}")
        print(f"  Pearson r: {r['pearson_r']:.3f} (p={r['pearson_p']:.4f})")
        print(f"  Spearman r: {r['spearman_r']:.3f} (p={r['spearman_p']:.4f})")
        print(f"  Predictions: mean={r['pred_mean']:.2f}, std={r['pred_std']:.2f}, range=[{r['pred_min']:.2f}, {r['pred_max']:.2f}]")
        print(f"  Targets: mean={r['target_mean']:.2f}, std={r['target_std']:.2f}")

    # Analyze prediction diversity
    analyze_prediction_diversity(predictions, config["data"]["dimensions"])

    # Overall assessment
    print("\n" + "=" * 80)
    print("OVERALL ASSESSMENT")
    print("=" * 80)

    pearson_rs = [results[dim]['pearson_r'] for dim in config["data"]["dimensions"]]
    mean_pearson = np.mean(pearson_rs)

    print(f"\nMean Pearson r across all dimensions: {mean_pearson:.3f}")

    # Count dimensions with good correlation
    good_dims = [dim for dim in config["data"]["dimensions"] if results[dim]['pearson_r'] > 0.3]
    print(f"Dimensions with r > 0.3: {len(good_dims)}/{len(config['data']['dimensions'])}")
    if good_dims:
        print(f"  {', '.join(good_dims)}")

    # Count dimensions with low variance
    low_var_dims = [dim for dim in config["data"]["dimensions"] if results[dim]['pred_std'] < 1.0]
    print(f"\nDimensions with low prediction variance (std < 1.0): {len(low_var_dims)}/{len(config['data']['dimensions'])}")
    if low_var_dims:
        print(f"  {', '.join(low_var_dims)}")

    # Recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    if mean_pearson > 0.2 and len(low_var_dims) < len(config["data"]["dimensions"]) / 2:
        print("\nPROCEED to Stage 3 (expert fine-tuning)")
        print("Model has learned useful patterns and shows reasonable diversity.")
    elif mean_pearson > 0.1:
        print("\nCAUTION - Consider proceeding but watch for issues")
        print("Model has learned some patterns but may have problems.")
    else:
        print("\nDO NOT PROCEED to Stage 3")
        print("Model has not learned useful patterns. Fix issues first.")
        print("\nSuggested fixes:")
        print("- Check pseudo-label quality")
        print("- Increase training epochs")
        print("- Adjust learning rate or architecture")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
