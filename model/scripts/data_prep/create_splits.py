#!/usr/bin/env python3
"""
Create train/val/test splits for PercePiano data.

Uses performer-based splitting to prevent data leakage between sets.

Usage:
    cd model
    uv run python scripts/data_prep/create_splits.py
"""

import random
import shutil
from collections import defaultdict
from pathlib import Path

# Paths
INPUT_DIR = Path("data/percepiano_vnet_converted")
OUTPUT_DIR = Path("data/percepiano_vnet_split")

# Split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Random seed for reproducibility
SEED = 42


def get_performer_id(filename: str) -> str:
    """Extract performer ID from filename."""
    # Format: Composer_Piece_bars_PerformerID_Segment
    parts = filename.split("_")
    if len(parts) >= 5:
        return parts[-2]
    return "unknown"


def main():
    print("=" * 60)
    print("Creating Train/Val/Test Splits")
    print("=" * 60)

    # Get all pickle files (excluding stat.pkl)
    pkl_files = sorted([f for f in INPUT_DIR.glob("*.pkl") if f.stem != "stat"])
    print(f"\nFound {len(pkl_files)} samples")

    # Group by performer
    performer_files = defaultdict(list)
    for f in pkl_files:
        performer = get_performer_id(f.stem)
        performer_files[performer].append(f)

    # Filter out "Score" (these are score MIDIs, not performances)
    if "Score" in performer_files:
        print(f"  Excluding {len(performer_files['Score'])} 'Score' files")
        del performer_files["Score"]

    performers = sorted(performer_files.keys())
    print(f"  Performers: {len(performers)}")
    for p in performers:
        print(f"    {p}: {len(performer_files[p])} samples")

    # Shuffle performers and split
    random.seed(SEED)
    random.shuffle(performers)

    n_train = int(len(performers) * TRAIN_RATIO)
    n_val = int(len(performers) * VAL_RATIO)

    train_performers = performers[:n_train]
    val_performers = performers[n_train : n_train + n_val]
    test_performers = performers[n_train + n_val :]

    print(f"\nSplit by performers:")
    print(f"  Train: {len(train_performers)} performers")
    print(f"  Val: {len(val_performers)} performers")
    print(f"  Test: {len(test_performers)} performers")

    # Create output directories
    train_dir = OUTPUT_DIR / "train"
    val_dir = OUTPUT_DIR / "val"
    test_dir = OUTPUT_DIR / "test"

    for d in [train_dir, val_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Copy files
    def copy_files(performer_list, target_dir):
        count = 0
        for p in performer_list:
            for f in performer_files[p]:
                shutil.copy(f, target_dir / f.name)
                count += 1
        return count

    train_count = copy_files(train_performers, train_dir)
    val_count = copy_files(val_performers, val_dir)
    test_count = copy_files(test_performers, test_dir)

    # Copy stat.pkl to output root
    stat_file = INPUT_DIR / "stat.pkl"
    if stat_file.exists():
        shutil.copy(stat_file, OUTPUT_DIR / "stat.pkl")

    print(f"\nSamples:")
    print(f"  Train: {train_count}")
    print(f"  Val: {val_count}")
    print(f"  Test: {test_count}")

    print(f"\nOutput: {OUTPUT_DIR}")
    print(f"\nTrain performers: {train_performers}")
    print(f"Val performers: {val_performers}")
    print(f"Test performers: {test_performers}")


if __name__ == "__main__":
    main()
