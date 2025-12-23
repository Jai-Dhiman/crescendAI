#!/usr/bin/env python3
"""
Quick single-batch overfit test to verify PercePiano architecture works.

This test:
1. Creates synthetic VirtuosoNet-style features (79-dim)
2. Creates synthetic hierarchical indices (beat, measure, voice)
3. Attempts to overfit a single batch for 100 steps
4. Reports if the model can learn (loss -> 0, R2 -> 1)

If this passes: Architecture is correct, problem is likely in data
If this fails: There's a bug in the architecture

Usage:
    cd model
    uv run python scripts/diagnostics/quick_overfit_test.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from percepiano.models.percepiano_replica import (
    PERCEPIANO_DIMENSIONS,
    PercePianoVNetModule,
)


def create_synthetic_batch(
    batch_size: int = 4,
    num_notes: int = 256,
    feature_dim: int = 79,
    num_dimensions: int = 19,
    device: str = "cpu",
):
    """
    Create synthetic VirtuosoNet-style data for testing.

    Features are created to have reasonable statistics:
    - Base features: z-score normalized (mean ~0, std ~1)
    - Labels: uniform in [0, 1]
    - Beat/measure indices: sequential, starting from 1
    """
    # Create normalized features (like z-score normalized VirtuosoNet features)
    features = torch.randn(batch_size, num_notes, feature_dim)

    # Create labels in [0, 1] range (sigmoid output)
    labels = torch.rand(batch_size, num_dimensions)

    # Create hierarchical indices
    # Beat indices: 1, 1, 1, 1, 2, 2, 2, 2, ... (4 notes per beat, starting from 1)
    notes_per_beat = 4
    beat_indices = torch.zeros(batch_size, num_notes, dtype=torch.long)
    for b in range(batch_size):
        for n in range(num_notes):
            beat_indices[b, n] = (n // notes_per_beat) + 1  # Start from 1, not 0

    # Measure indices: 1, 1, ..., 2, 2, ... (16 notes per measure, starting from 1)
    notes_per_measure = 16
    measure_indices = torch.zeros(batch_size, num_notes, dtype=torch.long)
    for b in range(batch_size):
        for n in range(num_notes):
            measure_indices[b, n] = (n // notes_per_measure) + 1  # Start from 1, not 0

    # Voice indices: all 1 (single voice, starting from 1)
    voice_indices = torch.ones(batch_size, num_notes, dtype=torch.long)

    # Attention mask: all valid
    attention_mask = torch.ones(batch_size, num_notes, dtype=torch.bool)

    # Num notes
    num_notes_tensor = torch.full((batch_size,), num_notes, dtype=torch.long)

    return {
        "input_features": features.to(device),
        "labels": labels.to(device),
        "note_locations": {
            "beat": beat_indices.to(device),
            "measure": measure_indices.to(device),
            "voice": voice_indices.to(device),
        },
        "attention_mask": attention_mask.to(device),
        "num_notes": num_notes_tensor.to(device),
    }


def run_overfit_test(
    num_steps: int = 100,
    batch_size: int = 4,
    num_notes: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
) -> dict:
    """
    Run single-batch overfit test.

    Returns:
        dict with test results
    """
    print("=" * 60)
    print("SINGLE-BATCH OVERFIT TEST")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num notes: {num_notes}")
    print(f"  Feature dim: 79")
    print(f"  Num steps: {num_steps}")
    print(f"  Learning rate: {lr}")
    print(f"  Device: {device}")

    # Create model with input_size=79 (stripped VirtuosoNet features)
    print("\nCreating PercePianoVNetModule with input_size=79...")
    model = PercePianoVNetModule(
        input_size=79,  # Stripped features (not 84)
        hidden_size=256,
        learning_rate=lr,
    )
    model = model.to(device)
    model.train()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Create synthetic batch
    print("\nCreating synthetic batch...")
    batch = create_synthetic_batch(
        batch_size=batch_size,
        num_notes=num_notes,
        feature_dim=79,
        num_dimensions=len(PERCEPIANO_DIMENSIONS),
        device=device,
    )
    print(f"  Input shape: {batch['input_features'].shape}")
    print(f"  Labels shape: {batch['labels'].shape}")
    print(
        f"  Beat indices range: [{batch['note_locations']['beat'].min()}, {batch['note_locations']['beat'].max()}]"
    )
    print(
        f"  Measure indices range: [{batch['note_locations']['measure'].min()}, {batch['note_locations']['measure'].max()}]"
    )

    # Set up optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Training loop
    print(f"\nTraining for {num_steps} steps...")
    losses = []
    r2_scores = []

    for step in range(num_steps):
        optimizer.zero_grad()

        # Forward pass
        outputs = model(
            batch["input_features"],
            batch["note_locations"],
            batch.get("attention_mask"),
        )
        predictions = outputs["predictions"]
        targets = batch["labels"]

        # Compute loss
        loss = criterion(predictions, targets)

        # Check for NaN
        if torch.isnan(loss):
            print(f"\n[CRITICAL] NaN loss at step {step}!")
            return {
                "success": False,
                "error": "NaN loss",
                "final_loss": float("nan"),
                "final_r2": float("nan"),
            }

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

        optimizer.step()

        # Compute R2
        with torch.no_grad():
            pred_np = predictions.cpu().numpy().flatten()
            target_np = targets.cpu().numpy().flatten()
            r2 = r2_score(target_np, pred_np)

        losses.append(loss.item())
        r2_scores.append(r2)

        if step % 20 == 0:
            print(f"  Step {step:3d} | Loss: {loss.item():.6f} | R2: {r2:.4f}")

    # Final results
    final_loss = losses[-1]
    final_r2 = r2_scores[-1]

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nFinal loss: {final_loss:.6f}")
    print(f"Final R2: {final_r2:.4f}")
    print(
        f"Loss reduction: {losses[0]:.6f} -> {final_loss:.6f} ({(1 - final_loss / losses[0]) * 100:.1f}%)"
    )
    print(f"R2 improvement: {r2_scores[0]:.4f} -> {final_r2:.4f}")

    # Check success criteria
    success = final_loss < 0.01 and final_r2 > 0.95

    if success:
        print("\n[SUCCESS] Model can overfit single batch!")
        print("  Architecture is correct.")
        print("  If training fails on real data, the issue is in the data pipeline.")
    else:
        print("\n[WARNING] Model did not fully overfit!")
        if final_loss >= 0.01:
            print(f"  Loss {final_loss:.6f} >= 0.01 threshold")
        if final_r2 <= 0.95:
            print(f"  R2 {final_r2:.4f} <= 0.95 threshold")
        print("\n  Possible issues:")
        print("    - Learning rate too low (try 1e-2)")
        print("    - Model architecture issue")
        print("    - Need more training steps")

    return {
        "success": success,
        "final_loss": final_loss,
        "final_r2": final_r2,
        "losses": losses,
        "r2_scores": r2_scores,
    }


def main():
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Run test
    results = run_overfit_test(
        num_steps=100,
        batch_size=4,
        num_notes=256,
        lr=1e-3,
        device=device,
    )

    # Return exit code
    return 0 if results["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
