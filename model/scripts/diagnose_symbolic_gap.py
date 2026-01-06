#!/usr/bin/env python3
"""
Diagnose the R2 gap between our PercePiano training (0.350) and published SOTA (0.397).

This script:
1. Loads checkpoint metadata to check at which epoch each fold stopped
2. Verifies data dimensions and label normalization
3. Reports per-fold R2 breakdown

Usage:
    python scripts/diagnose_symbolic_gap.py [--checkpoint-dir PATH] [--data-dir PATH]
"""

import argparse
import json
import pickle
import subprocess
from pathlib import Path

import torch


def download_checkpoints(checkpoint_dir: Path, gdrive_path: str) -> None:
    """Download checkpoints from Google Drive if not present locally."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Check if checkpoints exist locally
    local_checkpoints = list(checkpoint_dir.glob("fold*_best.pt"))
    if len(local_checkpoints) >= 4:
        print(f"Found {len(local_checkpoints)} checkpoints locally")
        return

    print(f"Downloading checkpoints from {gdrive_path}...")
    result = subprocess.run(
        ["rclone", "copy", gdrive_path, str(checkpoint_dir), "--progress"],
        capture_output=False,
    )
    if result.returncode != 0:
        raise RuntimeError("Failed to download checkpoints from Google Drive")


def analyze_checkpoints(checkpoint_dir: Path) -> dict:
    """Analyze checkpoint metadata for all folds."""
    results = {}

    for fold in range(4):
        checkpoint_path = checkpoint_dir / f"fold{fold}_best.pt"
        if not checkpoint_path.exists():
            print(f"Warning: {checkpoint_path} not found")
            results[fold] = {"error": "checkpoint not found"}
            continue

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        results[fold] = {
            "epoch": checkpoint.get("epoch", "unknown"),
            "r2": checkpoint.get("r2", "unknown"),
            "fold": checkpoint.get("fold", fold),
        }

    return results


def analyze_data(data_dir: Path) -> dict:
    """Analyze preprocessed data for dimensions and label ranges."""
    results = {"folds": {}}

    for fold in range(4):
        fold_path = data_dir / f"fold{fold}"
        if not fold_path.exists():
            results["folds"][fold] = {"error": "fold directory not found"}
            continue

        fold_info = {"train": {}, "valid": {}, "test": {}}

        for split in ["train", "valid", "test"]:
            split_path = fold_path / split
            if not split_path.exists():
                fold_info[split] = {"error": "split not found"}
                continue

            # Count samples
            pkl_files = [f for f in split_path.glob("*.pkl") if f.name != "stat.pkl"]
            fold_info[split]["num_samples"] = len(pkl_files)

            # Check first sample dimensions
            if pkl_files:
                with open(pkl_files[0], "rb") as f:
                    sample = pickle.load(f)
                fold_info[split]["input_dim"] = (
                    sample["input"].shape[1] if len(sample["input"]) > 0 else 0
                )
                fold_info[split]["sample_file"] = pkl_files[0].name

        # Load stats
        stat_path = fold_path / "train" / "stat.pkl"
        if stat_path.exists():
            with open(stat_path, "rb") as f:
                stats = pickle.load(f)
            fold_info["input_keys"] = len(stats.get("input_keys", []))
            fold_info["key_to_dim_input"] = len(
                stats.get("key_to_dim", {}).get("input", {})
            )

        results["folds"][fold] = fold_info

    return results


def analyze_labels(label_path: Path) -> dict:
    """Analyze label file for normalization range."""
    if not label_path.exists():
        return {"error": "label file not found"}

    with open(label_path) as f:
        labels = json.load(f)

    # Collect all label values
    all_values = []
    num_dims = None
    for key, values in labels.items():
        if num_dims is None:
            num_dims = len(values)
        all_values.extend(values[:19])  # First 19 are perceptual dims

    return {
        "num_samples": len(labels),
        "num_dimensions": num_dims,
        "min_value": min(all_values),
        "max_value": max(all_values),
        "mean_value": sum(all_values) / len(all_values),
    }


def main():
    parser = argparse.ArgumentParser(description="Diagnose PercePiano R2 gap")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("/tmp/checkpoints/percepiano_original"),
        help="Directory containing checkpoint files",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/tmp/percepiano_original"),
        help="Directory containing preprocessed data",
    )
    parser.add_argument(
        "--label-path",
        type=Path,
        default=Path(
            "/tmp/percepiano_labels/label_2round_mean_reg_19_with0_rm_highstd0.json"
        ),
        help="Path to label JSON file",
    )
    parser.add_argument(
        "--gdrive-checkpoint-path",
        type=str,
        default="gdrive:crescendai_data/checkpoints/percepiano_original",
        help="Google Drive path for checkpoints",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("PERCEPIANO SYMBOLIC MODEL GAP DIAGNOSIS")
    print("=" * 70)
    print(f"Target R2: 0.397 (published SOTA)")
    print(f"Current R2: 0.350")
    print(f"Gap: -0.047 (11.8% relative)")
    print("=" * 70)

    # Download checkpoints if needed
    print("\n[1] CHECKPOINT ANALYSIS")
    print("-" * 70)
    try:
        download_checkpoints(args.checkpoint_dir, args.gdrive_checkpoint_path)
        checkpoint_results = analyze_checkpoints(args.checkpoint_dir)

        print("\nPer-Fold Results:")
        print(f"{'Fold':<6} {'Epoch':<8} {'R2':>10}")
        print("-" * 30)

        total_r2 = 0
        valid_folds = 0
        early_stopped = False

        for fold in range(4):
            info = checkpoint_results[fold]
            if "error" in info:
                print(f"{fold:<6} {info['error']}")
            else:
                epoch = info["epoch"]
                r2 = info["r2"]
                total_r2 += r2
                valid_folds += 1

                # Check if early stopping triggered before epoch 80
                flag = ""
                if isinstance(epoch, int) and epoch < 80:
                    flag = " [EARLY STOP?]"
                    early_stopped = True

                print(f"{fold:<6} {epoch:<8} {r2:>+.4f}{flag}")

        if valid_folds > 0:
            avg_r2 = total_r2 / valid_folds
            print("-" * 30)
            print(f"{'Avg':<6} {'':<8} {avg_r2:>+.4f}")

        if early_stopped:
            print("\n*** FINDING: Some folds stopped before epoch 80!")
            print("*** This suggests early stopping may be cutting training too early.")
            print(
                "*** Original PercePiano trains for 100 epochs with NO early stopping."
            )
    except Exception as e:
        print(f"Error analyzing checkpoints: {e}")

    # Analyze data
    print("\n[2] DATA ANALYSIS")
    print("-" * 70)
    if args.data_dir.exists():
        data_results = analyze_data(args.data_dir)

        print("\nPer-Fold Data Summary:")
        for fold in range(4):
            fold_info = data_results["folds"].get(fold, {})
            if "error" in fold_info:
                print(f"  Fold {fold}: {fold_info['error']}")
            else:
                train_info = fold_info.get("train", {})
                valid_info = fold_info.get("valid", {})
                print(f"  Fold {fold}:")
                print(
                    f"    Train: {train_info.get('num_samples', '?')} samples, input_dim={train_info.get('input_dim', '?')}"
                )
                print(f"    Valid: {valid_info.get('num_samples', '?')} samples")
    else:
        print(f"Data directory not found: {args.data_dir}")

    # Analyze labels
    print("\n[3] LABEL ANALYSIS")
    print("-" * 70)
    if args.label_path.exists():
        label_results = analyze_labels(args.label_path)
        if "error" not in label_results:
            print(f"  Num samples: {label_results['num_samples']}")
            print(f"  Num dimensions: {label_results['num_dimensions']}")
            print(
                f"  Value range: [{label_results['min_value']:.4f}, {label_results['max_value']:.4f}]"
            )
            print(f"  Mean value: {label_results['mean_value']:.4f}")

            if label_results["min_value"] >= 0 and label_results["max_value"] <= 1:
                print("  Normalization: [0, 1] range confirmed")
            else:
                print("  WARNING: Values outside [0, 1] range!")
        else:
            print(f"  {label_results['error']}")
    else:
        print(f"Label file not found: {args.label_path}")

    # Summary and recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print("""
1. If early stopping triggered before epoch 80-100:
   -> Re-train WITHOUT early stopping (run full 100 epochs)
   -> Match original: num_epochs=100, no patience logic

2. If training ran full epochs but R2 still ~0.35:
   -> Investigate data preprocessing differences
   -> Compare with original PercePiano data from Zenodo

3. Verify LR scheduler is StepLR(step_size=3000, gamma=0.98)
""")


if __name__ == "__main__":
    main()
