#!/usr/bin/env python3
"""
Diagnostic script for PercePiano Replica model.

This script helps identify why the model achieves R2 ~0.08-0.10 instead of 0.397:
1. Checks hierarchical aggregation output shapes
2. Analyzes attention weight distributions
3. Identifies collapsed output dimensions
4. Compares feature statistics with expected values

Usage:
    python scripts/diagnostics/diagnose_percepiano.py
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from percepiano.data.percepiano_vnet_dataset import PercePianoVNetDataset
from percepiano.models.hierarchy_utils import find_boundaries_batch
from percepiano.models.percepiano_replica import (
    PERCEPIANO_DIMENSIONS,
    PercePianoVNetModule,
)


def check_data_format(data_dir: Path):
    """Check if data is in correct format."""
    print("\n" + "=" * 70)
    print("1. DATA FORMAT CHECK")
    print("=" * 70)

    train_dir = data_dir / "train"
    if not train_dir.exists():
        print(f"ERROR: {train_dir} does not exist!")
        return None

    pkl_files = list(train_dir.glob("*.pkl"))
    print(f"Found {len(pkl_files)} training samples")

    if not pkl_files:
        print("ERROR: No .pkl files found!")
        return None

    # Load a sample
    sample_path = pkl_files[0]
    with open(sample_path, "rb") as f:
        sample = pickle.load(f)

    print(f"\nSample file: {sample_path.name}")
    print(f"Keys: {list(sample.keys())}")

    # Check required keys
    required_keys = ["input", "labels", "note_location", "num_notes"]
    missing = [k for k in required_keys if k not in sample]
    if missing:
        print(f"ERROR: Missing required keys: {missing}")
        return None

    # Check shapes
    inp = np.array(sample["input"])
    labels = np.array(sample["labels"])
    print(f"\nInput shape: {inp.shape} (expected: (num_notes, 84))")
    print(f"Labels shape: {labels.shape} (expected: (19,))")
    print(f"num_notes: {sample['num_notes']}")

    # Check note_location
    nl = sample["note_location"]
    print(f"\nNote location keys: {list(nl.keys())}")
    for key in ["beat", "measure", "voice"]:
        if key in nl:
            arr = np.array(nl[key])
            print(f"  {key}: shape={arr.shape}, range=[{arr.min()}, {arr.max()}]")

    return sample


def check_hierarchical_aggregation(model, dataloader, device):
    """Check if hierarchical aggregation is working correctly."""
    print("\n" + "=" * 70)
    print("2. HIERARCHICAL AGGREGATION CHECK")
    print("=" * 70)

    model.eval()
    batch = next(iter(dataloader))

    input_features = batch["input_features"].to(device)
    note_locations = {
        "beat": batch["note_locations_beat"].to(device),
        "measure": batch["note_locations_measure"].to(device),
        "voice": batch["note_locations_voice"].to(device),
    }

    print(f"\nInput shape: {input_features.shape}")
    print(f"Beat indices shape: {note_locations['beat'].shape}")
    print(
        f"Beat range: [{note_locations['beat'].min().item()}, {note_locations['beat'].max().item()}]"
    )
    print(
        f"Measure range: [{note_locations['measure'].min().item()}, {note_locations['measure'].max().item()}]"
    )

    # Check boundary detection
    boundaries = find_boundaries_batch(note_locations["beat"])
    num_beats = [len(b) - 1 for b in boundaries]
    print(
        f"\nBeats per sample: mean={np.mean(num_beats):.1f}, min={min(num_beats)}, max={max(num_beats)}"
    )
    print(f"First sample boundaries: {boundaries[0][:10]}...")

    # Run forward pass and inspect intermediate outputs
    with torch.no_grad():
        han_outputs = model.han_encoder(input_features, note_locations)

    print(f"\nHAN output shapes:")
    for key, val in han_outputs.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: {val.shape}")
            if val.dim() == 3:  # [B, T, C]
                # Check for zeros (collapsed representations)
                non_zero_frac = (val.abs() > 1e-6).float().mean().item()
                print(f"    Non-zero fraction: {non_zero_frac:.4f}")

    # Check total_note_cat statistics
    total_note_cat = han_outputs["total_note_cat"]
    print(f"\nTotal_note_cat statistics:")
    print(f"  Mean: {total_note_cat.mean().item():.4f}")
    print(f"  Std: {total_note_cat.std().item():.4f}")
    print(f"  Min: {total_note_cat.min().item():.4f}")
    print(f"  Max: {total_note_cat.max().item():.4f}")

    return han_outputs


def check_attention_collapse(model, dataloader, device):
    """Check if attention weights are collapsing."""
    print("\n" + "=" * 70)
    print("3. ATTENTION WEIGHT ANALYSIS")
    print("=" * 70)

    model.eval()
    batch = next(iter(dataloader))

    input_features = batch["input_features"].to(device)
    note_locations = {
        "beat": batch["note_locations_beat"].to(device),
        "measure": batch["note_locations_measure"].to(device),
        "voice": batch["note_locations_voice"].to(device),
    }

    with torch.no_grad():
        # Run full HAN forward to get properly shaped intermediate outputs
        han_outputs = model.han_encoder(input_features, note_locations)

        # Check final attention on contracted features
        contracted = model.performance_contractor(han_outputs["total_note_cat"])
        final_attn_scores = model.final_attention.get_attention(contracted)
        print(f"\nFinal attention scores shape: {final_attn_scores.shape}")
        print(f"  Mean: {final_attn_scores.mean().item():.4f}")
        print(f"  Std: {final_attn_scores.std().item():.4f}")
        final_softmax = torch.softmax(final_attn_scores, dim=1)
        print(f"  After softmax std: {final_softmax.std().item():.4f}")

        # Check if attention is uniform (bad sign)
        entropy = (
            -(final_softmax * (final_softmax + 1e-10).log()).sum(dim=1).mean().item()
        )
        max_entropy = np.log(final_softmax.shape[1])
        print(f"  Attention entropy: {entropy:.4f} / {max_entropy:.4f} (max)")
        if entropy / max_entropy > 0.9:
            print("  WARNING: Attention is nearly uniform - may cause averaging!")


def check_prediction_variance(model, dataloader, device):
    """Check prediction variance per dimension."""
    print("\n" + "=" * 70)
    print("4. PREDICTION VARIANCE ANALYSIS")
    print("=" * 70)

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            input_features = batch["input_features"].to(device)
            note_locations = {
                "beat": batch["note_locations_beat"].to(device),
                "measure": batch["note_locations_measure"].to(device),
                "voice": batch["note_locations_voice"].to(device),
            }

            outputs = model(input_features, note_locations)
            all_preds.append(outputs["predictions"].cpu())
            all_targets.append(batch["scores"])

    preds = torch.cat(all_preds, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()

    print(f"\nPrediction statistics:")
    print(f"  Overall range: [{preds.min():.4f}, {preds.max():.4f}]")
    print(f"  Overall mean: {preds.mean():.4f}")
    print(f"  Overall std: {preds.std():.4f}")

    print(f"\nTarget statistics:")
    print(f"  Overall range: [{targets.min():.4f}, {targets.max():.4f}]")
    print(f"  Overall mean: {targets.mean():.4f}")
    print(f"  Overall std: {targets.std():.4f}")

    # Per-dimension analysis
    print(f"\nPer-dimension analysis:")
    print(f"{'Dimension':<25} {'Pred Std':>10} {'Target Std':>10} {'Status':<15}")
    print(f"{'-' * 25} {'-' * 10} {'-' * 10} {'-' * 15}")

    collapsed = []
    for i, dim in enumerate(PERCEPIANO_DIMENSIONS):
        pred_std = preds[:, i].std()
        target_std = targets[:, i].std()

        if pred_std < 0.03:
            status = "COLLAPSED"
            collapsed.append(dim)
        elif pred_std < target_std * 0.5:
            status = "LOW VARIANCE"
        else:
            status = "OK"

        print(f"{dim:<25} {pred_std:>10.4f} {target_std:>10.4f} {status:<15}")

    print(f"\nCollapsed dimensions ({len(collapsed)}/19): {collapsed}")

    return preds, targets


def check_gradient_flow(model, dataloader, device):
    """Check gradient flow through the network."""
    print("\n" + "=" * 70)
    print("5. GRADIENT FLOW CHECK")
    print("=" * 70)

    model.train()
    batch = next(iter(dataloader))

    input_features = batch["input_features"].to(device)
    note_locations = {
        "beat": batch["note_locations_beat"].to(device),
        "measure": batch["note_locations_measure"].to(device),
        "voice": batch["note_locations_voice"].to(device),
    }
    targets = batch["scores"].to(device)

    # Forward pass
    outputs = model(input_features, note_locations)
    loss = torch.nn.functional.mse_loss(outputs["predictions"], targets)

    # Backward pass
    loss.backward()

    print(f"\nGradient norms by layer:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if "weight" in name and grad_norm < 1e-6:
                print(f"  {name}: {grad_norm:.2e} [VANISHING!]")
            elif "weight" in name:
                print(f"  {name}: {grad_norm:.2e}")

    # Zero gradients
    model.zero_grad()


def main():
    print("=" * 70)
    print("PERCEPIANO REPLICA DIAGNOSTIC REPORT")
    print("=" * 70)

    # Configuration
    data_dir = PROJECT_ROOT / "data" / "percepiano_vnet_split"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Data directory: {data_dir}")

    # 1. Check data format
    sample = check_data_format(data_dir)
    if sample is None:
        return

    # 2. Create dataset and dataloader
    print("\n" + "=" * 70)
    print("LOADING DATA AND MODEL")
    print("=" * 70)

    dataset = PercePianoVNetDataset(
        data_dir=data_dir / "train",
        max_notes=1024,
        augment=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )

    print(f"Dataset size: {len(dataset)}")

    # 3. Create model (with input_size=79 as in notebook)
    model = PercePianoVNetModule(
        input_size=79,  # Match notebook CONFIG
        hidden_size=256,
        note_layers=2,
        voice_layers=2,
        beat_layers=2,
        measure_layers=1,
        num_attention_heads=8,
        dropout=0.2,
    )
    model = model.to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    # 4. Run diagnostics
    check_hierarchical_aggregation(model, dataloader, device)
    check_attention_collapse(model, dataloader, device)
    preds, targets = check_prediction_variance(model, dataloader, device)
    check_gradient_flow(model, dataloader, device)

    # 5. Summary
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)

    pred_std = preds.std()
    target_std = targets.std()

    print(f"\nPrediction variance ratio: {pred_std / target_std:.2f}")
    print("  (Should be close to 1.0 for good learning)")

    if pred_std / target_std < 0.5:
        print("\nDIAGNOSIS: Model predictions are too narrow (variance collapse)")
        print("Likely causes:")
        print("  1. Final attention is averaging everything")
        print("  2. Hierarchical aggregation loses information")
        print("  3. MSE loss with sigmoid causes mode collapse")
        print("\nRecommended fixes:")
        print("  1. Check attention weights for uniformity")
        print("  2. Try per-dimension loss weighting")
        print("  3. Consider removing sigmoid or using different loss")


if __name__ == "__main__":
    main()
