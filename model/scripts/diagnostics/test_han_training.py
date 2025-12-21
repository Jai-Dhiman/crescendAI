#!/usr/bin/env python
"""
Quick HAN Training Test Script

This script runs a quick test of the full PercePiano HAN architecture to verify:
1. Data can be loaded and passed through the model
2. Model can compute gradients
3. Training shows learning signal (loss decreases)

Run this BEFORE the full notebook to catch issues early.

Usage:
    python scripts/test_han_training.py --data_dir /path/to/percepiano_vnet
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
from rich.console import Console
from sklearn.metrics import r2_score

console = Console()


def run_quick_training_test(data_dir: str, num_epochs: int = 5, batch_size: int = 8):
    """Run a quick training test with the full HAN architecture."""

    console.print("[bold cyan]=" * 70)
    console.print("[bold cyan]PERCEPIANO HAN QUICK TRAINING TEST")
    console.print("[bold cyan]=" * 70)

    # Check CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    console.print(f"\nDevice: {device}")

    # Import modules
    console.print("\n[cyan]Loading modules...[/cyan]")
    from src.percepiano.data.percepiano_vnet_dataset import create_vnet_dataloaders
    from src.percepiano.models.percepiano_replica import PercePianoVNetModule

    # Create dataloaders
    console.print(f"\n[cyan]Creating dataloaders from {data_dir}...[/cyan]")
    try:
        train_loader, val_loader, _ = create_vnet_dataloaders(
            data_dir=data_dir,
            batch_size=batch_size,
            max_notes=1024,
            num_workers=0,
        )
        console.print(f"  Train samples: {len(train_loader.dataset)}")
        console.print(f"  Val samples: {len(val_loader.dataset)}")
    except Exception as e:
        console.print(f"[red]Failed to create dataloaders: {e}[/red]")
        return False

    # Create model
    console.print("\n[cyan]Creating PercePiano HAN model...[/cyan]")
    try:
        model = PercePianoVNetModule(
            input_size=79,  # Only normalized features
            hidden_size=256,
            note_layers=2,
            voice_layers=2,
            beat_layers=2,
            measure_layers=1,
            num_attention_heads=8,
            final_hidden=128,
            learning_rate=1e-4,
            weight_decay=1e-5,
            dropout=0.2,
        )
        model = model.to(device)
        console.print(f"  Parameters: {model.count_parameters():,}")
    except Exception as e:
        console.print(f"[red]Failed to create model: {e}[/red]")
        return False

    # Verify first batch
    console.print("\n[cyan]Verifying first batch...[/cyan]")
    batch = next(iter(train_loader))
    console.print(f"  input_features: {batch['input_features'].shape}")
    console.print(f"  attention_mask: {batch['attention_mask'].shape}")
    console.print(f"  scores: {batch['scores'].shape}")
    console.print(f"  beat indices range: [{batch['note_locations_beat'].min()}, {batch['note_locations_beat'].max()}]")

    # Check for NaN in input
    if torch.isnan(batch['input_features']).any():
        console.print("[red]  CRITICAL: NaN values in input features![/red]")
        return False

    # Check label range
    label_min, label_max = batch['scores'].min().item(), batch['scores'].max().item()
    console.print(f"  Label range: [{label_min:.3f}, {label_max:.3f}]")
    if label_min < 0 or label_max > 1:
        console.print("[red]  CRITICAL: Labels outside [0, 1] range![/red]")

    # Training loop
    console.print(f"\n[cyan]Running {num_epochs} training epochs...[/cyan]")
    console.print("[yellow]Looking for: loss decrease, R2 improvement[/yellow]\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.MSELoss()

    train_losses = []
    val_r2s = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_losses = []

        for batch_idx, batch in enumerate(train_loader):
            input_features = batch['input_features'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            scores = batch['scores'].to(device)

            note_locations = {
                'beat': batch['note_locations_beat'].to(device),
                'measure': batch['note_locations_measure'].to(device),
                'voice': batch['note_locations_voice'].to(device),
            }

            optimizer.zero_grad()

            try:
                outputs = model(
                    input_features=input_features,
                    note_locations=note_locations,
                    attention_mask=attention_mask,
                )
                predictions = outputs['predictions']
                loss = criterion(predictions, scores)

                # Check for NaN loss
                if torch.isnan(loss):
                    console.print(f"[red]  Epoch {epoch+1}: NaN loss at batch {batch_idx}![/red]")
                    console.print(f"  Predictions: mean={predictions.mean():.4f}, std={predictions.std():.4f}")
                    return False

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()

                epoch_losses.append(loss.item())

            except Exception as e:
                console.print(f"[red]  Epoch {epoch+1}, Batch {batch_idx}: Forward pass failed: {e}[/red]")
                import traceback
                traceback.print_exc()
                return False

        train_loss = np.mean(epoch_losses)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        all_preds, all_targets = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_features = batch['input_features'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                scores = batch['scores'].to(device)

                note_locations = {
                    'beat': batch['note_locations_beat'].to(device),
                    'measure': batch['note_locations_measure'].to(device),
                    'voice': batch['note_locations_voice'].to(device),
                }

                outputs = model(
                    input_features=input_features,
                    note_locations=note_locations,
                    attention_mask=attention_mask,
                )

                all_preds.append(outputs['predictions'].cpu())
                all_targets.append(scores.cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()

        # Compute R2
        try:
            val_r2 = r2_score(all_targets, all_preds)
        except Exception:
            val_r2 = float('-inf')

        val_r2s.append(val_r2)

        # Print progress
        r2_color = "green" if val_r2 > 0 else "red"
        loss_trend = "v" if len(train_losses) > 1 and train_loss < train_losses[-2] else "^" if len(train_losses) > 1 else "-"

        console.print(
            f"  Epoch {epoch+1}/{num_epochs} | "
            f"Loss: {train_loss:.4f} {loss_trend} | "
            f"Val R2: [{r2_color}]{val_r2:+.4f}[/{r2_color}]"
        )

        # Check prediction statistics
        pred_std = all_preds.std()
        if pred_std < 0.01:
            console.print(f"    [yellow]Warning: Prediction std={pred_std:.4f} (collapsed?)[/yellow]")

    # Summary
    console.print("\n" + "=" * 70)
    console.print("[bold cyan]TEST RESULTS[/bold cyan]")
    console.print("=" * 70)

    # Check for learning signal
    loss_decreased = train_losses[-1] < train_losses[0]
    r2_improved = val_r2s[-1] > val_r2s[0]
    final_r2_positive = val_r2s[-1] > 0

    console.print(f"\n1. LOSS TREND")
    console.print(f"   Initial: {train_losses[0]:.4f}")
    console.print(f"   Final:   {train_losses[-1]:.4f}")
    console.print(f"   Decreased: {'[green]YES[/green]' if loss_decreased else '[red]NO[/red]'}")

    console.print(f"\n2. R2 TREND")
    console.print(f"   Initial: {val_r2s[0]:+.4f}")
    console.print(f"   Final:   {val_r2s[-1]:+.4f}")
    console.print(f"   Improved: {'[green]YES[/green]' if r2_improved else '[red]NO[/red]'}")
    console.print(f"   Positive: {'[green]YES[/green]' if final_r2_positive else '[red]NO[/red]'}")

    console.print(f"\n3. DIAGNOSIS")
    if loss_decreased and (r2_improved or final_r2_positive):
        console.print("[green]Model is learning - full training should work![/green]")
        console.print("Run the full notebook for complete training.")
        return True
    elif loss_decreased:
        console.print("[yellow]Loss decreasing but R2 not improving.[/yellow]")
        console.print("Possible causes:")
        console.print("  - May need more epochs")
        console.print("  - Hierarchical attention may need time to learn structure")
        console.print("Recommendation: Try full training for 20+ epochs")
        return True
    else:
        console.print("[red]Model not learning - check data pipeline![/red]")
        console.print("Possible causes:")
        console.print("  1. Feature-label misalignment")
        console.print("  2. Label scale issue")
        console.print("  3. Beat/measure index format")
        console.print("Run Data Quality Diagnostics in the notebook.")
        return False


def main():
    parser = argparse.ArgumentParser(description='Quick HAN training test')
    parser.add_argument('--data_dir', type=str,
                        default='/tmp/percepiano_data/percepiano_vnet',
                        help='Path to preprocessed VirtuosoNet data')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of test epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for testing')

    args = parser.parse_args()

    # Check if data exists
    data_path = Path(args.data_dir)
    if not data_path.exists():
        console.print(f"[red]Data directory not found: {args.data_dir}[/red]")
        console.print("Run preprocessing first or provide correct path.")
        sys.exit(1)

    train_dir = data_path / 'train'
    if not train_dir.exists() or not list(train_dir.glob('*.pkl')):
        console.print(f"[red]No training data found in {train_dir}[/red]")
        sys.exit(1)

    success = run_quick_training_test(
        data_dir=args.data_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
