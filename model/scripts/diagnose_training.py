#!/usr/bin/env python3
"""
Training diagnostics script.

Checks:
- Model forward pass and output ranges
- Gradient flow and norms
- Parameter updates
- Loss components and breakdown
- Data statistics (labels, audio, MIDI)
- Learning rate schedule

Usage:
    python scripts/diagnose_training.py --config configs/pseudo_pretrain.yaml
    python scripts/diagnose_training.py --config configs/pseudo_pretrain.yaml --checkpoint checkpoints/model.ckpt
"""

import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Any
import pytorch_lightning as pl

from src.models.lightning_module import PerformanceEvaluationModel
from src.data.dataset import create_dataloaders


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def analyze_data_statistics(train_loader, val_loader, dimension_names):
    """Analyze data statistics from dataloaders."""
    print("\n" + "=" * 80)
    print("DATA STATISTICS")
    print("=" * 80)

    # Collect all labels from training set
    all_train_labels = []
    all_val_labels = []

    print("\nCollecting training data statistics...")
    for batch in train_loader:
        all_train_labels.append(batch['labels'].numpy())
    all_train_labels = np.concatenate(all_train_labels, axis=0)

    print("Collecting validation data statistics...")
    for batch in val_loader:
        all_val_labels.append(batch['labels'].numpy())
    all_val_labels = np.concatenate(all_val_labels, axis=0)

    # Per-dimension statistics
    print(f"\nTraining set: {len(all_train_labels)} samples")
    print(f"Validation set: {len(all_val_labels)} samples")

    print(f"\n{'Dimension':<25} {'Train Mean':<12} {'Train Std':<12} {'Val Mean':<12} {'Val Std':<12}")
    print("-" * 80)
    for i, dim_name in enumerate(dimension_names):
        train_mean = all_train_labels[:, i].mean()
        train_std = all_train_labels[:, i].std()
        val_mean = all_val_labels[:, i].mean()
        val_std = all_val_labels[:, i].std()
        print(f"{dim_name:<25} {train_mean:<12.2f} {train_std:<12.2f} {val_mean:<12.2f} {val_std:<12.2f}")

    # Check for potential issues
    print("\nPotential issues:")
    issues_found = False
    for i, dim_name in enumerate(dimension_names):
        train_std = all_train_labels[:, i].std()
        val_std = all_val_labels[:, i].std()

        if train_std < 1.0:
            print(f"  WARNING: {dim_name} has very low training variance ({train_std:.3f})")
            issues_found = True
        if val_std < 1.0:
            print(f"  WARNING: {dim_name} has very low validation variance ({val_std:.3f})")
            issues_found = True

    if not issues_found:
        print("  None detected")

    return all_train_labels, all_val_labels


def analyze_model_outputs(model, batch, device):
    """Analyze model outputs on a single batch."""
    print("\n" + "=" * 80)
    print("MODEL OUTPUT ANALYSIS")
    print("=" * 80)

    model.eval()
    with torch.no_grad():
        # Move batch to device
        audio_waveform = batch['audio_waveform'].to(device)
        midi_tokens = batch.get('midi_tokens', None)
        if midi_tokens is not None:
            midi_tokens = midi_tokens.to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        output = model.forward(
            audio_waveform=audio_waveform,
            midi_tokens=midi_tokens,
        )

        predictions = output['scores']
        uncertainties = output['uncertainties']

        # Statistics
        print(f"\nBatch size: {predictions.shape[0]}")
        print(f"Number of dimensions: {predictions.shape[1]}")

        print(f"\nPredictions:")
        print(f"  Range: [{predictions.min().item():.2f}, {predictions.max().item():.2f}]")
        print(f"  Mean: {predictions.mean().item():.2f}")
        print(f"  Std: {predictions.std().item():.2f}")

        print(f"\nTargets:")
        print(f"  Range: [{labels.min().item():.2f}, {labels.max().item():.2f}]")
        print(f"  Mean: {labels.mean().item():.2f}")
        print(f"  Std: {labels.std().item():.2f}")

        print(f"\nPer-dimension breakdown:")
        print(f"{'Dimension':<25} {'Pred Mean':<12} {'Pred Std':<12} {'Target Mean':<12} {'Target Std':<12}")
        print("-" * 80)
        for i, dim_name in enumerate(model.dimension_names):
            pred_mean = predictions[:, i].mean().item()
            pred_std = predictions[:, i].std().item()
            target_mean = labels[:, i].mean().item()
            target_std = labels[:, i].std().item()
            print(f"{dim_name:<25} {pred_mean:<12.2f} {pred_std:<12.2f} {target_mean:<12.2f} {target_std:<12.2f}")

        print(f"\nUncertainties (σ):")
        for i, dim_name in enumerate(model.dimension_names):
            print(f"  {dim_name:<25} {uncertainties[i].item():.4f}")

        # Check for issues
        print("\nPotential issues:")
        issues_found = False

        if predictions.std().item() < 1.0:
            print(f"  WARNING: Predictions have very low variance ({predictions.std().item():.3f})")
            print(f"           Model may be outputting near-constant values")
            issues_found = True

        if predictions.min().item() < -10 or predictions.max().item() > 110:
            print(f"  WARNING: Predictions outside expected [0, 100] range")
            issues_found = True

        if not issues_found:
            print("  None detected")


def analyze_gradients(model, batch, device):
    """Analyze gradient flow through the model."""
    print("\n" + "=" * 80)
    print("GRADIENT ANALYSIS")
    print("=" * 80)

    model.train()
    model.zero_grad()

    # Move batch to device
    audio_waveform = batch['audio_waveform'].to(device)
    midi_tokens = batch.get('midi_tokens', None)
    if midi_tokens is not None:
        midi_tokens = midi_tokens.to(device)
    labels = batch['labels'].to(device)

    # Forward pass
    output = model.forward(
        audio_waveform=audio_waveform,
        midi_tokens=midi_tokens,
    )

    predictions = output['scores']

    # Compute loss
    loss_output = model.loss_fn(
        predictions=predictions,
        targets=labels,
        log_vars=model.mtl_head.log_vars,
    )

    loss = loss_output['loss']
    task_losses = loss_output['task_losses']

    # Backward pass
    loss.backward()

    # Analyze gradients
    print(f"\nLoss breakdown:")
    print(f"  Total loss: {loss.item():.4f}")
    for i, dim_name in enumerate(model.dimension_names):
        print(f"  {dim_name:<25} {task_losses[i].item():.4f}")

    # Gradient norms by module
    print(f"\nGradient norms by module:")
    gradient_stats = {}
    for name, module in model.named_children():
        total_norm = 0.0
        num_params = 0
        for param in module.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                num_params += 1
        total_norm = total_norm ** 0.5
        gradient_stats[name] = total_norm
        print(f"  {name:<30} {total_norm:<12.6f} ({num_params} params)")

    # Check for issues
    print("\nPotential issues:")
    issues_found = False

    if loss.item() > 1000:
        print(f"  WARNING: Very high loss ({loss.item():.2f})")
        print(f"           May indicate scaling issues or initialization problems")
        issues_found = True

    for name, norm in gradient_stats.items():
        if norm < 1e-6:
            print(f"  WARNING: Very small gradients in {name} ({norm:.2e})")
            print(f"           Module may not be learning")
            issues_found = True
        elif norm > 100:
            print(f"  WARNING: Very large gradients in {name} ({norm:.2e})")
            print(f"           May need gradient clipping")
            issues_found = True

    if not issues_found:
        print("  None detected")


def analyze_parameter_updates(model, batch, device, learning_rate=1e-4):
    """Analyze parameter updates after one step."""
    print("\n" + "=" * 80)
    print("PARAMETER UPDATE ANALYSIS")
    print("=" * 80)

    model.train()

    # Save initial parameters
    initial_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            initial_params[name] = param.data.clone()

    # Create simple optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    optimizer.zero_grad()

    # Move batch to device
    audio_waveform = batch['audio_waveform'].to(device)
    midi_tokens = batch.get('midi_tokens', None)
    if midi_tokens is not None:
        midi_tokens = midi_tokens.to(device)
    labels = batch['labels'].to(device)

    # Forward + backward
    output = model.forward(
        audio_waveform=audio_waveform,
        midi_tokens=midi_tokens,
    )

    predictions = output['scores']
    loss_output = model.loss_fn(
        predictions=predictions,
        targets=labels,
        log_vars=model.mtl_head.log_vars,
    )

    loss = loss_output['loss']
    loss.backward()

    # Take optimizer step
    optimizer.step()

    # Analyze parameter changes
    print(f"\nParameter updates (sample of 10 layers):")
    print(f"{'Layer':<50} {'Max Change':<15} {'Mean Change':<15}")
    print("-" * 80)

    param_changes = {}
    for name, param in model.named_parameters():
        if param.requires_grad and name in initial_params:
            change = (param.data - initial_params[name]).abs()
            max_change = change.max().item()
            mean_change = change.mean().item()
            param_changes[name] = (max_change, mean_change)

    # Show first 10 and last 10
    items = list(param_changes.items())
    for name, (max_change, mean_change) in items[:10]:
        print(f"{name:<50} {max_change:<15.2e} {mean_change:<15.2e}")

    # Check for issues
    print("\nPotential issues:")
    issues_found = False

    no_updates = [name for name, (max_change, _) in param_changes.items() if max_change < 1e-10]
    if no_updates:
        print(f"  WARNING: {len(no_updates)} parameters not updating")
        print(f"           First few: {no_updates[:3]}")
        issues_found = True

    if not issues_found:
        print("  None detected")


def main():
    parser = argparse.ArgumentParser(description="Diagnose training issues")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--num-batches", type=int, default=1, help="Number of batches to analyze")
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
        num_workers=0,  # Single-threaded for diagnostics
        augmentation_config=config["data"].get("augmentation", None),
        audio_sample_rate=config["data"]["audio_sample_rate"],
        max_audio_length=config["data"]["max_audio_length"],
        max_midi_events=config["data"]["max_midi_events"],
    )

    # Analyze data statistics
    analyze_data_statistics(train_loader, val_loader, config["data"]["dimensions"])

    # Initialize model
    print("\n" + "=" * 80)
    print("MODEL INITIALIZATION")
    print("=" * 80)

    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"\nLoading model from checkpoint: {args.checkpoint}")
        model = PerformanceEvaluationModel.load_from_checkpoint(args.checkpoint)
    else:
        print("\nCreating new model from scratch")
        model = PerformanceEvaluationModel(
            audio_dim=config["model"]["audio_dim"],
            midi_dim=config["model"]["midi_dim"],
            fusion_dim=config["model"]["fusion_dim"],
            aggregator_dim=config["model"]["aggregator_dim"],
            num_dimensions=config["model"]["num_dimensions"],
            dimension_names=config["data"]["dimensions"],
            mert_model_name=config["model"]["mert_model_name"],
            freeze_audio_encoder=config["model"]["freeze_audio_encoder"],
            gradient_checkpointing=config["model"]["gradient_checkpointing"],
            learning_rate=config["training"]["learning_rate"],
            backbone_lr=config["training"]["backbone_lr"],
            heads_lr=config["training"]["heads_lr"],
            weight_decay=config["training"]["weight_decay"],
            warmup_steps=config["training"]["warmup_steps"],
            max_epochs=config["training"]["max_epochs"],
        )

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(f"\nDevice: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Get a batch
    print("\nGetting sample batch...")
    batch = next(iter(train_loader))

    # Run diagnostics
    analyze_model_outputs(model, batch, device)
    analyze_gradients(model, batch, device)
    analyze_parameter_updates(model, batch, device, learning_rate=config["training"]["heads_lr"])

    print("\n" + "=" * 80)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
