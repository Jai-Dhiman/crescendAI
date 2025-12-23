#!/usr/bin/env python3
"""
Split and convert PercePiano data to 83-dim (SOTA configuration).

This script:
1. Reads the train/val/test split JSON files
2. Copies samples to the appropriate split directories
3. Converts from 84-dim to 83-dim (removes section_tempo at index 5)

Usage:
    python scripts/data_prep/split_and_convert.py
"""

import json
import pickle
import shutil
from pathlib import Path
from tqdm import tqdm
import numpy as np


# Paths
MODEL_ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = MODEL_ROOT / "data"
INPUT_DIR = DATA_ROOT / "percepiano_vnet_converted"
OUTPUT_DIR = DATA_ROOT / "percepiano_vnet_split"
SPLITS_DIR = DATA_ROOT / "processed"

# Feature dimensions
OLD_BASE_DIM = 79
OLD_TOTAL_DIM = 84
NEW_BASE_DIM = 78
NEW_TOTAL_DIM = 83
SECTION_TEMPO_IDX = 5


def load_split_samples(split_file: Path) -> set:
    """Load sample names from a split JSON file."""
    with open(split_file, 'r') as f:
        samples = json.load(f)
    return {s['name'] for s in samples}


def convert_sample(data: dict) -> dict:
    """Convert a sample from 84-dim to 83-dim."""
    input_features = data['input']
    num_notes = input_features.shape[0]

    if input_features.shape[1] != OLD_TOTAL_DIM:
        raise ValueError(f"Expected {OLD_TOTAL_DIM} dims, got {input_features.shape[1]}")

    # Create new feature array
    new_features = np.zeros((num_notes, NEW_TOTAL_DIM), dtype=np.float32)

    # Copy base features, skipping section_tempo at index 5
    new_features[:, :SECTION_TEMPO_IDX] = input_features[:, :SECTION_TEMPO_IDX]
    new_features[:, SECTION_TEMPO_IDX:NEW_BASE_DIM] = input_features[:, SECTION_TEMPO_IDX+1:OLD_BASE_DIM]

    # Copy unnorm features (shift from 79-83 to 78-82)
    new_features[:, NEW_BASE_DIM:] = input_features[:, OLD_BASE_DIM:]

    result = data.copy()
    result['input'] = new_features
    return result


def main():
    print("="*60)
    print("SPLIT AND CONVERT TO 83-DIM (SOTA)")
    print("="*60)

    # Load split assignments
    splits = {}
    for split in ['train', 'val', 'test']:
        split_file = SPLITS_DIR / f"percepiano_{split}.json"
        if split_file.exists():
            splits[split] = load_split_samples(split_file)
            print(f"  {split}: {len(splits[split])} samples in JSON")
        else:
            print(f"  {split}: JSON file not found")
            splits[split] = set()

    # Get all input files
    input_files = list(INPUT_DIR.glob('*.pkl'))
    print(f"\nInput directory: {INPUT_DIR}")
    print(f"Found {len(input_files)} pkl files")

    # Create output directories
    for split in ['train', 'val', 'test']:
        (OUTPUT_DIR / split).mkdir(parents=True, exist_ok=True)

    # Process each file
    counts = {'train': 0, 'val': 0, 'test': 0, 'skipped': 0}

    print("\nProcessing files...")
    for pkl_file in tqdm(input_files):
        sample_name = pkl_file.stem

        # Determine split
        target_split = None
        for split in ['train', 'val', 'test']:
            if sample_name in splits[split]:
                target_split = split
                break

        if target_split is None:
            counts['skipped'] += 1
            continue

        # Load, convert, and save
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)

            converted = convert_sample(data)

            output_file = OUTPUT_DIR / target_split / pkl_file.name
            with open(output_file, 'wb') as f:
                pickle.dump(converted, f)

            counts[target_split] += 1

        except Exception as e:
            print(f"\nError processing {sample_name}: {e}")
            counts['skipped'] += 1

    # Create stat.pkl (copy from existing or create new)
    old_stat = INPUT_DIR.parent / "percepiano_vnet_split" / "stat.pkl"
    if old_stat.exists():
        # Convert existing stats
        with open(old_stat, 'rb') as f:
            stats = pickle.load(f)
        if 'section_tempo' in stats.get('mean', {}):
            del stats['mean']['section_tempo']
        if 'section_tempo' in stats.get('std', {}):
            del stats['std']['section_tempo']
        with open(OUTPUT_DIR / "stat.pkl", 'wb') as f:
            pickle.dump(stats, f)
        print("\nConverted stat.pkl")

    # Summary
    print("\n" + "="*60)
    print("CONVERSION COMPLETE")
    print("="*60)
    print(f"  train: {counts['train']} files")
    print(f"  val:   {counts['val']} files")
    print(f"  test:  {counts['test']} files")
    print(f"  skipped: {counts['skipped']} files")

    # Verify dimensions
    print("\nVerifying output dimensions...")
    for split in ['train', 'val', 'test']:
        split_dir = OUTPUT_DIR / split
        files = list(split_dir.glob('*.pkl'))
        if files:
            with open(files[0], 'rb') as f:
                data = pickle.load(f)
            print(f"  {split}: {len(files)} files, input shape = {data['input'].shape}")


if __name__ == '__main__':
    main()
