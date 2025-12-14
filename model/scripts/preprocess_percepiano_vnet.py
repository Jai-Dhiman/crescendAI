#!/usr/bin/env python3
"""
Preprocess PercePiano data using VirtuosoNet feature extraction.

This script extracts 78-dimensional VirtuosoNet features from the PercePiano dataset
and saves them as pickle files for training.

Usage:
    python scripts/preprocess_percepiano_vnet.py

Output:
    data/processed/percepiano_vnet/
        train/*.pkl
        val/*.pkl
        test/*.pkl
        stat.pkl (normalization statistics)
"""

import sys
import json
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from tqdm import tqdm
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.virtuosonet_feature_extractor import (
    VirtuosoNetFeatureExtractor,
    FeatureStats,
    VNET_INPUT_KEYS,
    FEATURE_DIMS,
    NORM_FEAT_KEYS,
    TOTAL_FEATURE_DIM,
)


def load_split_data(split_file: Path) -> List[Dict[str, Any]]:
    """Load a split JSON file."""
    with open(split_file, 'r') as f:
        return json.load(f)


def get_composer_from_name(name: str) -> str:
    """Extract composer name from file name."""
    name_lower = name.lower()
    if 'beethoven' in name_lower:
        return 'Beethoven'
    elif 'schubert' in name_lower:
        return 'Schubert'
    elif 'bach' in name_lower:
        return 'Bach'
    elif 'chopin' in name_lower:
        return 'Chopin'
    elif 'mozart' in name_lower:
        return 'Mozart'
    elif 'liszt' in name_lower:
        return 'Liszt'
    elif 'brahms' in name_lower:
        return 'Brahms'
    elif 'debussy' in name_lower:
        return 'Debussy'
    elif 'ravel' in name_lower:
        return 'Ravel'
    else:
        return 'unknown'


def process_sample(
    sample: Dict[str, Any],
    data_root: Path,
    score_xml_dir: Path,
    extractor: VirtuosoNetFeatureExtractor,
) -> Optional[Dict[str, Any]]:
    """
    Process a single sample and extract features.

    Args:
        sample: Sample dict with 'name', 'midi_path', 'score_path', 'percepiano_scores'
        data_root: Root data directory
        score_xml_dir: Directory containing score XML files
        extractor: VirtuosoNet feature extractor

    Returns:
        Feature dictionary or None if extraction fails
    """
    try:
        # Get paths
        midi_path = data_root / sample['midi_path']
        score_path = score_xml_dir / sample['score_path']

        if not midi_path.exists():
            print(f"MIDI file not found: {midi_path}")
            return None

        if not score_path.exists():
            print(f"Score XML not found: {score_path}")
            return None

        # Extract features
        features = extractor.extract_features(score_path, midi_path)

        # Add labels
        scores = sample.get('percepiano_scores', [])
        if len(scores) >= 19:
            features['labels'] = np.array(scores[:19], dtype=np.float32)
        else:
            print(f"Warning: {sample['name']} has only {len(scores)} scores, expected 19")
            features['labels'] = np.zeros(19, dtype=np.float32)

        # Add metadata
        features['name'] = sample['name']

        return features

    except Exception as e:
        print(f"Error processing {sample.get('name', 'unknown')}: {e}")
        return None


def compute_normalization_stats(features_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Compute z-score normalization statistics from training features.

    Args:
        features_list: List of feature dictionaries

    Returns:
        Dict with 'mean' and 'std' for each normalizable feature
    """
    # Concatenate all features
    all_features = np.concatenate([f['input'] for f in features_list], axis=0)

    stats = {'mean': {}, 'std': {}}
    feature_idx = 0

    for key in VNET_INPUT_KEYS:
        dim = FEATURE_DIMS[key]

        if key in NORM_FEAT_KEYS:
            feat_slice = all_features[:, feature_idx:feature_idx + dim]
            stats['mean'][key] = float(np.mean(feat_slice))
            std = float(np.std(feat_slice))
            stats['std'][key] = std if std > 0 else 1.0
        else:
            stats['mean'][key] = 0.0
            stats['std'][key] = 1.0

        feature_idx += dim

    return stats


def apply_normalization(features: Dict[str, Any], stats: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """Apply z-score normalization to features."""
    normalized = features.copy()
    input_features = features['input'].copy()

    feature_idx = 0
    for key in VNET_INPUT_KEYS:
        dim = FEATURE_DIMS[key]

        if key in NORM_FEAT_KEYS:
            mean = stats['mean'][key]
            std = stats['std'][key]
            input_features[:, feature_idx:feature_idx + dim] = (
                input_features[:, feature_idx:feature_idx + dim] - mean
            ) / std

        feature_idx += dim

    normalized['input'] = input_features
    return normalized


def main():
    parser = argparse.ArgumentParser(description='Preprocess PercePiano data with VirtuosoNet features')
    parser.add_argument('--data_root', type=Path, default=PROJECT_ROOT / 'data',
                       help='Root data directory')
    parser.add_argument('--output_dir', type=Path, default=None,
                       help='Output directory for processed data')
    parser.add_argument('--skip_normalization', action='store_true',
                       help='Skip z-score normalization')
    args = parser.parse_args()

    # Set up paths
    data_root = args.data_root
    processed_dir = data_root / 'processed'
    raw_dir = data_root / 'raw' / 'PercePiano'
    score_xml_dir = raw_dir / 'virtuoso' / 'data' / 'score_xml'

    output_dir = args.output_dir or (processed_dir / 'percepiano_vnet')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load split files
    splits = {
        'train': processed_dir / 'percepiano_train.json',
        'val': processed_dir / 'percepiano_val.json',
        'test': processed_dir / 'percepiano_test.json',
    }

    # Process each split
    all_features = {'train': [], 'val': [], 'test': []}

    for split_name, split_file in splits.items():
        print(f"\nProcessing {split_name} split...")

        if not split_file.exists():
            print(f"Warning: {split_file} not found, skipping")
            continue

        samples = load_split_data(split_file)
        print(f"Found {len(samples)} samples")

        # Create output directory for this split
        split_output_dir = output_dir / split_name
        split_output_dir.mkdir(parents=True, exist_ok=True)

        # Process samples
        for sample in tqdm(samples, desc=split_name):
            composer = get_composer_from_name(sample.get('name', ''))
            extractor = VirtuosoNetFeatureExtractor(composer=composer)

            features = process_sample(sample, data_root, score_xml_dir, extractor)

            if features is not None:
                all_features[split_name].append(features)

    # Compute normalization statistics from training set
    if not args.skip_normalization and all_features['train']:
        print("\nComputing normalization statistics from training set...")
        stats = compute_normalization_stats(all_features['train'])

        # Save statistics
        stats_file = output_dir / 'stat.pkl'
        with open(stats_file, 'wb') as f:
            pickle.dump(stats, f)
        print(f"Saved normalization statistics to {stats_file}")

        # Apply normalization to all splits
        for split_name in all_features:
            print(f"Applying normalization to {split_name}...")
            all_features[split_name] = [
                apply_normalization(f, stats) for f in all_features[split_name]
            ]

    # Save processed features
    for split_name, features_list in all_features.items():
        if not features_list:
            continue

        split_output_dir = output_dir / split_name
        print(f"\nSaving {len(features_list)} samples to {split_output_dir}...")

        for features in tqdm(features_list, desc=f"Saving {split_name}"):
            name = features.get('name', 'unknown')
            output_file = split_output_dir / f"{name}.pkl"

            with open(output_file, 'wb') as f:
                pickle.dump(features, f)

    # Print summary
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print(f"Output directory: {output_dir}")
    for split_name, features_list in all_features.items():
        print(f"  {split_name}: {len(features_list)} samples")

    if all_features['train']:
        sample = all_features['train'][0]
        print(f"\nFeature dimensions:")
        print(f"  Input shape: {sample['input'].shape}")
        print(f"  Labels shape: {sample['labels'].shape}")
        print(f"  Num notes: {sample['num_notes']}")


if __name__ == '__main__':
    main()
