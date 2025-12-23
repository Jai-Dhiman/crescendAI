#!/usr/bin/env python3
"""
Preprocess PercePiano data using VirtuosoNet feature extraction.

This script extracts 83-dimensional VirtuosoNet features from the PercePiano dataset
(SOTA configuration, R2 = 0.397):
- 78 base features (normalized where applicable, excludes section_tempo)
- 5 preserved unnormalized features (midi_pitch_unnorm, duration_unnorm, etc.)

The unnormalized features are critical for key augmentation (pitch shifting).

Usage:
    python scripts/data_prep/preprocess_percepiano_vnet.py

Output:
    data/processed/percepiano_vnet/
        train/*.pkl
        val/*.pkl
        test/*.pkl
        stat.pkl (normalization statistics)
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

# Add project root to path (model/)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from percepiano.data.virtuosonet_feature_extractor import (
    BASE_FEATURE_DIM,
    FEATURE_DIMS,
    NORM_FEAT_KEYS,
    PRESERVE_FEAT_KEYS,
    TOTAL_FEATURE_DIM,
    VNET_INPUT_KEYS,
    FeatureStats,
    VirtuosoNetFeatureExtractor,
)


def load_split_data(split_file: Path) -> List[Dict[str, Any]]:
    """Load a split JSON file."""
    with open(split_file, "r") as f:
        return json.load(f)


def add_unnorm_features(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add preserved unnormalized features to the input array.

    This copies certain features BEFORE normalization and appends them
    to the end of the feature vector. This matches the original PercePiano
    behavior in data_for_training.py:412-419.

    The unnorm features are critical for:
    - Key augmentation (midi_pitch_unnorm provides raw MIDI pitch 21-108)
    - Preserving original scale information

    Args:
        features: Feature dict with 'input' array of shape (num_notes, 78) - SOTA config

    Returns:
        Updated features with 'input' array of shape (num_notes, 83)
    """
    input_features = features["input"]
    num_notes = input_features.shape[0]

    # Create expanded array with space for unnorm features
    expanded = np.zeros((num_notes, TOTAL_FEATURE_DIM), dtype=np.float32)
    expanded[:, :BASE_FEATURE_DIM] = input_features

    # Copy preserved features to unnorm slots
    # Feature indices in VNET_INPUT_KEYS order (SOTA 78-feature config):
    # midi_pitch=0, duration=1, beat_importance=2, measure_length=3, qpm_primo=4, following_rest=5
    # (section_tempo was removed, so following_rest shifted from 6 to 5)
    preserve_indices = {
        "midi_pitch": 0,
        "duration": 1,
        "beat_importance": 2,
        "measure_length": 3,
        "following_rest": 5,  # Was 6 before section_tempo removal
    }

    for i, key in enumerate(PRESERVE_FEAT_KEYS):
        src_idx = preserve_indices[key]
        dst_idx = BASE_FEATURE_DIM + i
        expanded[:, dst_idx] = input_features[:, src_idx]

    result = features.copy()
    result["input"] = expanded
    return result


def get_composer_from_name(name: str) -> str:
    """Extract composer name from file name."""
    name_lower = name.lower()
    if "beethoven" in name_lower:
        return "Beethoven"
    elif "schubert" in name_lower:
        return "Schubert"
    elif "bach" in name_lower:
        return "Bach"
    elif "chopin" in name_lower:
        return "Chopin"
    elif "mozart" in name_lower:
        return "Mozart"
    elif "liszt" in name_lower:
        return "Liszt"
    elif "brahms" in name_lower:
        return "Brahms"
    elif "debussy" in name_lower:
        return "Debussy"
    elif "ravel" in name_lower:
        return "Ravel"
    else:
        return "unknown"


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
        # Get MIDI path - handle both absolute and relative paths
        midi_path_str = sample["midi_path"]
        midi_path = Path(midi_path_str)
        if not midi_path.is_absolute():
            midi_path = data_root / midi_path_str

        # Get score path - handle both absolute and relative paths
        score_path_str = sample["score_path"]
        score_path = Path(score_path_str)
        if not score_path.is_absolute():
            # Try as relative to score_xml_dir first
            score_path = score_xml_dir / score_path_str
            if not score_path.exists():
                # Try just the filename in score_xml_dir
                score_path = score_xml_dir / Path(score_path_str).name

        if not midi_path.exists():
            print(f"MIDI file not found: {midi_path}")
            return None

        if not score_path.exists():
            print(f"Score XML not found: {score_path}")
            return None

        # Extract features
        features = extractor.extract_features(score_path, midi_path)

        # Add labels
        scores = sample.get("percepiano_scores", [])
        if len(scores) >= 19:
            features["labels"] = np.array(scores[:19], dtype=np.float32)
        else:
            print(
                f"Warning: {sample['name']} has only {len(scores)} scores, expected 19"
            )
            features["labels"] = np.zeros(19, dtype=np.float32)

        # Add metadata
        features["name"] = sample["name"]

        return features

    except Exception as e:
        import traceback

        sample_name = sample.get("name", "unknown")
        print(f"\nError processing {sample_name}:")
        print(f"  Type: {type(e).__name__}")
        print(f"  Message: {e}")
        print(f"  MIDI: {sample.get('midi_path', 'N/A')}")
        print(f"  Score: {sample.get('score_path', 'N/A')}")
        print(f"  Traceback:\n{traceback.format_exc()}")
        return None


def compute_normalization_stats(
    features_list: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """
    Compute z-score normalization statistics from training features.

    Only computes stats for the base 78 features (not the unnorm features).

    Args:
        features_list: List of feature dictionaries (with 83-dim input)

    Returns:
        Dict with 'mean' and 'std' for each normalizable feature
    """
    # Concatenate all features (only use base features for stats)
    all_features = np.concatenate(
        [f["input"][:, :BASE_FEATURE_DIM] for f in features_list], axis=0
    )

    stats = {"mean": {}, "std": {}}
    feature_idx = 0

    for key in VNET_INPUT_KEYS:
        dim = FEATURE_DIMS[key]

        if key in NORM_FEAT_KEYS:
            feat_slice = all_features[:, feature_idx : feature_idx + dim]
            stats["mean"][key] = float(np.mean(feat_slice))
            std = float(np.std(feat_slice))
            stats["std"][key] = std if std > 0 else 1.0
        else:
            stats["mean"][key] = 0.0
            stats["std"][key] = 1.0

        feature_idx += dim

    return stats


def apply_normalization(
    features: Dict[str, Any], stats: Dict[str, Dict[str, float]]
) -> Dict[str, Any]:
    """
    Apply z-score normalization to base features only.

    The unnorm features (indices 78-82) are left untouched.
    """
    normalized = features.copy()
    input_features = features["input"].copy()

    # Only normalize the base features (first 78)
    feature_idx = 0
    for key in VNET_INPUT_KEYS:
        dim = FEATURE_DIMS[key]

        if key in NORM_FEAT_KEYS:
            mean = stats["mean"][key]
            std = stats["std"][key]
            input_features[:, feature_idx : feature_idx + dim] = (
                input_features[:, feature_idx : feature_idx + dim] - mean
            ) / std

        feature_idx += dim

    # Unnorm features (indices BASE_FEATURE_DIM to TOTAL_FEATURE_DIM) remain unchanged
    normalized["input"] = input_features
    return normalized


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess PercePiano data with VirtuosoNet features"
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Root data directory",
    )
    parser.add_argument(
        "--json_dir",
        type=Path,
        default=None,
        help="Directory containing percepiano_*.json split files (defaults to data_root)",
    )
    parser.add_argument(
        "--score_xml_dir",
        type=Path,
        default=None,
        help="Directory containing score XML files",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--skip_normalization", action="store_true", help="Skip z-score normalization"
    )
    args = parser.parse_args()

    # Set up paths
    data_root = args.data_root

    # JSON directory: explicit arg, or data_root itself, or data_root/processed
    if args.json_dir:
        json_dir = args.json_dir
    elif (data_root / "percepiano_train.json").exists():
        json_dir = data_root
    else:
        json_dir = data_root / "processed"

    # Score XML directory: explicit arg, or search common locations
    if args.score_xml_dir:
        score_xml_dir = args.score_xml_dir
    elif (data_root / "PercePiano" / "virtuoso" / "data" / "score_xml").exists():
        score_xml_dir = data_root / "PercePiano" / "virtuoso" / "data" / "score_xml"
    else:
        score_xml_dir = (
            data_root / "raw" / "PercePiano" / "virtuoso" / "data" / "score_xml"
        )

    output_dir = args.output_dir or (data_root / "percepiano_vnet")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"JSON directory: {json_dir}")
    print(f"Score XML directory: {score_xml_dir}")
    print(f"Output directory: {output_dir}")

    # Load split files
    splits = {
        "train": json_dir / "percepiano_train.json",
        "val": json_dir / "percepiano_val.json",
        "test": json_dir / "percepiano_test.json",
    }

    # Process each split
    all_features = {"train": [], "val": [], "test": []}

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
            composer = get_composer_from_name(sample.get("name", ""))
            extractor = VirtuosoNetFeatureExtractor(composer=composer)

            features = process_sample(sample, data_root, score_xml_dir, extractor)

            if features is not None:
                all_features[split_name].append(features)

    # Add unnorm features BEFORE normalization
    # This preserves raw values for key augmentation (midi_pitch_unnorm = raw MIDI 21-108)
    print("\nAdding unnorm features (78-dim -> 83-dim)...")
    for split_name in all_features:
        all_features[split_name] = [
            add_unnorm_features(f) for f in all_features[split_name]
        ]

    # Compute normalization statistics from training set
    if not args.skip_normalization and all_features["train"]:
        print("\nComputing normalization statistics from training set...")
        stats = compute_normalization_stats(all_features["train"])

        # Save statistics
        stats_file = output_dir / "stat.pkl"
        with open(stats_file, "wb") as f:
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
            name = features.get("name", "unknown")
            output_file = split_output_dir / f"{name}.pkl"

            with open(output_file, "wb") as f:
                pickle.dump(features, f)

    # Print summary
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    for split_name, features_list in all_features.items():
        print(f"  {split_name}: {len(features_list)} samples")

    if all_features["train"]:
        sample = all_features["train"][0]
        print(f"\nFeature dimensions:")
        print(f"  Input shape: {sample['input'].shape}")
        print(f"  Labels shape: {sample['labels'].shape}")
        print(f"  Num notes: {sample['num_notes']}")


if __name__ == "__main__":
    main()
