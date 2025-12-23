#!/usr/bin/env python3
"""
Create K-Fold Split Assignments for PercePiano Dataset

This script generates piece-based fold assignments following the PercePiano paper
methodology. All samples from the same piece are assigned to the same fold.

Usage:
    python create_kfold_splits.py --data-dir /path/to/percepiano_vnet_split

Output:
    Creates fold_assignments.json in the data directory
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from percepiano.data.kfold_split import (
    create_piece_based_folds,
    print_fold_statistics,
    save_fold_assignments,
)


def main():
    parser = argparse.ArgumentParser(
        description="Create piece-based k-fold split assignments for PercePiano"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/Users/jdhiman/Documents/crescendai/model/data/percepiano_vnet_split",
        help="Path to data directory containing train/val/test subdirs",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=4,
        help="Number of cross-validation folds (default: 4)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Fraction of pieces for test set (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for fold assignments (default: data_dir/fold_assignments.json)",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory does not exist: {data_dir}")
        sys.exit(1)

    output_path = args.output or data_dir / "fold_assignments.json"

    print(f"Creating {args.n_folds}-fold piece-based splits...")
    print(f"Data directory: {data_dir}")
    print(f"Test ratio: {args.test_ratio}")
    print(f"Random seed: {args.seed}")

    # Create fold assignments
    assignments = create_piece_based_folds(
        data_dir=data_dir,
        n_folds=args.n_folds,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    # Print statistics
    print_fold_statistics(assignments, n_folds=args.n_folds)

    # Save to file
    save_fold_assignments(assignments, output_path)

    print(f"\nFold assignments saved to: {output_path}")
    print("Done!")


if __name__ == "__main__":
    main()
