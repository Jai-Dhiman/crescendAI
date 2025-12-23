#!/usr/bin/env python3
"""
Convert existing 84-dim PercePiano data to 83-dim (SOTA configuration).

The SOTA configuration uses 78 base features (excludes section_tempo).
This script removes section_tempo from existing processed data.

Old layout (84-dim = 79 base + 5 unnorm):
- Indices 0-78: Base features (includes section_tempo at index 5)
- Indices 79-83: Unnorm features

New layout (83-dim = 78 base + 5 unnorm):
- Indices 0-77: Base features (section_tempo removed)
- Indices 78-82: Unnorm features

Usage:
    python scripts/data_prep/convert_to_78feat.py \
        --input_dir data/percepiano_vnet \
        --output_dir data/percepiano_vnet_split
"""

import argparse
import pickle
import shutil
from pathlib import Path
from tqdm import tqdm
import numpy as np


# Old feature indices (79 base features)
OLD_BASE_DIM = 79
OLD_TOTAL_DIM = 84  # 79 base + 5 unnorm

# New feature indices (78 base features, section_tempo removed)
NEW_BASE_DIM = 78
NEW_TOTAL_DIM = 83  # 78 base + 5 unnorm

# section_tempo was at index 5 in the old layout
SECTION_TEMPO_IDX = 5


def convert_sample(input_file: Path, output_file: Path) -> bool:
    """
    Convert a single sample from 84-dim to 83-dim.

    Returns True if successful, False otherwise.
    """
    try:
        with open(input_file, 'rb') as f:
            data = pickle.load(f)

        input_features = data['input']
        num_notes = input_features.shape[0]

        # Verify input dimensions
        if input_features.shape[1] != OLD_TOTAL_DIM:
            print(f"Warning: {input_file.name} has {input_features.shape[1]} dims, expected {OLD_TOTAL_DIM}")
            return False

        # Create new feature array
        new_features = np.zeros((num_notes, NEW_TOTAL_DIM), dtype=np.float32)

        # Copy base features, skipping section_tempo at index 5
        # Old: 0,1,2,3,4,5,6,7,...,78  (79 features)
        # New: 0,1,2,3,4,  5,6,...,77  (78 features, section_tempo removed)
        new_features[:, :SECTION_TEMPO_IDX] = input_features[:, :SECTION_TEMPO_IDX]  # 0-4
        new_features[:, SECTION_TEMPO_IDX:NEW_BASE_DIM] = input_features[:, SECTION_TEMPO_IDX+1:OLD_BASE_DIM]  # 6-78 -> 5-77

        # Copy unnorm features (shift from 79-83 to 78-82)
        new_features[:, NEW_BASE_DIM:] = input_features[:, OLD_BASE_DIM:]

        # Update data
        data['input'] = new_features

        # Save
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'wb') as f:
            pickle.dump(data, f)

        return True

    except Exception as e:
        print(f"Error converting {input_file.name}: {e}")
        return False


def convert_stats(input_file: Path, output_file: Path) -> bool:
    """
    Convert normalization stats, removing section_tempo.
    """
    try:
        with open(input_file, 'rb') as f:
            stats = pickle.load(f)

        # Remove section_tempo from stats
        if 'section_tempo' in stats.get('mean', {}):
            del stats['mean']['section_tempo']
        if 'section_tempo' in stats.get('std', {}):
            del stats['std']['section_tempo']

        with open(output_file, 'wb') as f:
            pickle.dump(stats, f)

        return True

    except Exception as e:
        print(f"Error converting stats: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Convert 84-dim data to 83-dim (SOTA)')
    parser.add_argument('--input_dir', type=Path, required=True,
                       help='Input directory with 84-dim data')
    parser.add_argument('--output_dir', type=Path, required=True,
                       help='Output directory for 83-dim data')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    print("="*60)
    print("CONVERTING 84-DIM TO 83-DIM (SOTA CONFIGURATION)")
    print("="*60)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print()
    print("Changes:")
    print("  - Removing section_tempo (index 5)")
    print("  - Base features: 79 -> 78")
    print("  - Total features: 84 -> 83")
    print("="*60)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert each split
    total_converted = 0
    total_failed = 0

    for split in ['train', 'val', 'test']:
        split_input = input_dir / split
        split_output = output_dir / split

        if not split_input.exists():
            print(f"\nSkipping {split} (not found)")
            continue

        pkl_files = list(split_input.glob('*.pkl'))
        print(f"\nConverting {split}: {len(pkl_files)} files")

        for pkl_file in tqdm(pkl_files, desc=split):
            output_file = split_output / pkl_file.name
            if convert_sample(pkl_file, output_file):
                total_converted += 1
            else:
                total_failed += 1

    # Convert stats file
    stats_input = input_dir / 'stat.pkl'
    stats_output = output_dir / 'stat.pkl'
    if stats_input.exists():
        print(f"\nConverting stat.pkl...")
        if convert_stats(stats_input, stats_output):
            print("  stat.pkl converted successfully")
        else:
            print("  Failed to convert stat.pkl")

    # Copy fold_assignments.json if it exists
    fold_input = input_dir / 'fold_assignments.json'
    fold_output = output_dir / 'fold_assignments.json'
    if fold_input.exists() and not fold_output.exists():
        shutil.copy(fold_input, fold_output)
        print(f"  Copied fold_assignments.json")

    # Verify
    print()
    print("="*60)
    print("CONVERSION COMPLETE")
    print("="*60)
    print(f"Converted: {total_converted} files")
    print(f"Failed:    {total_failed} files")

    # Verify output dimensions
    for split in ['train', 'val', 'test']:
        split_output = output_dir / split
        if split_output.exists():
            sample_files = list(split_output.glob('*.pkl'))
            if sample_files:
                with open(sample_files[0], 'rb') as f:
                    data = pickle.load(f)
                print(f"  {split}: {len(sample_files)} files, shape={data['input'].shape}")


if __name__ == '__main__':
    main()
