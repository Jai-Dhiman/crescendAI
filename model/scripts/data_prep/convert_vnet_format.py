#!/usr/bin/env python3
"""
Convert VirtuosoNet preprocessing output to the format expected by PercePianoVNetDataset.

The preprocessing saves individual feature columns, but the dataset expects:
- input: (num_notes, 84) stacked feature array (79 base + 5 unnorm)
- note_location: dict with beat/measure/voice
- labels: (19,) PercePiano scores

This script also applies z-score normalization to the appropriate features.

Usage:
    cd model
    uv run python scripts/data_prep/convert_vnet_format.py
"""

import json
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Feature configuration (from virtuosonet_feature_extractor.py)
VNET_INPUT_KEYS = (
    "midi_pitch",
    "duration",
    "beat_importance",
    "measure_length",
    "qpm_primo",
    "section_tempo",
    "following_rest",
    "distance_from_abs_dynamic",
    "distance_from_recent_tempo",
    "beat_position",
    "xml_position",
    "grace_order",
    "preceded_by_grace_note",
    "followed_by_fermata_rest",
    "pitch",
    "tempo",
    "dynamic",
    "time_sig_vec",
    "slur_beam_vec",
    "composer_vec",
    "notation",
    "tempo_primo",
)

FEATURE_DIMS = {
    "midi_pitch": 1,
    "duration": 1,
    "beat_importance": 1,
    "measure_length": 1,
    "qpm_primo": 1,
    "section_tempo": 1,
    "following_rest": 1,
    "distance_from_abs_dynamic": 1,
    "distance_from_recent_tempo": 1,
    "beat_position": 1,
    "xml_position": 1,
    "grace_order": 1,
    "preceded_by_grace_note": 1,
    "followed_by_fermata_rest": 1,
    "pitch": 13,
    "tempo": 5,
    "dynamic": 4,
    "time_sig_vec": 9,
    "slur_beam_vec": 6,
    "composer_vec": 17,
    "notation": 9,
    "tempo_primo": 2,
}

BASE_FEATURE_DIM = sum(FEATURE_DIMS.values())  # 79

# Features that need z-score normalization (first 9 scalar features)
NORM_FEAT_KEYS = (
    "midi_pitch",
    "duration",
    "beat_importance",
    "measure_length",
    "qpm_primo",
    "section_tempo",
    "following_rest",
    "distance_from_abs_dynamic",
    "distance_from_recent_tempo",
)

# Features to preserve as unnormalized (appended after base features)
PRESERVE_FEAT_KEYS = (
    "midi_pitch",
    "duration",
    "beat_importance",
    "measure_length",
    "following_rest",
)


def stack_features(data: dict) -> tuple:
    """Stack features into (num_notes, 79) array without normalization."""
    num_notes = len(data["midi_pitch"])
    input_features = np.zeros((num_notes, BASE_FEATURE_DIM), dtype=np.float32)

    feature_idx = 0
    for key in VNET_INPUT_KEYS:
        dim = FEATURE_DIMS[key]
        feat_values = data.get(key)

        if feat_values is None:
            feature_idx += dim
            continue

        if dim == 1:
            input_features[:, feature_idx] = np.array(feat_values, dtype=np.float32)
        else:
            for i, vec in enumerate(feat_values):
                input_features[i, feature_idx : feature_idx + dim] = np.array(
                    vec[:dim], dtype=np.float32
                )

        feature_idx += dim

    # Add unnormalized features (5 dims) -> total 84
    unnorm_features = np.zeros((num_notes, len(PRESERVE_FEAT_KEYS)), dtype=np.float32)
    for i, key in enumerate(PRESERVE_FEAT_KEYS):
        unnorm_features[:, i] = np.array(data[key], dtype=np.float32)

    full_features = np.concatenate([input_features, unnorm_features], axis=1)
    return full_features, num_notes


def compute_normalization_stats(all_features: list) -> dict:
    """Compute mean and std for features that need normalization."""
    # Concatenate all samples
    all_data = np.concatenate(all_features, axis=0)
    print(
        f"  Computing stats from {all_data.shape[0]} notes across {len(all_features)} samples"
    )

    stats = {"mean": {}, "std": {}}

    # Compute stats for first 9 features (NORM_FEAT_KEYS)
    feature_idx = 0
    for key in VNET_INPUT_KEYS:
        dim = FEATURE_DIMS[key]
        if key in NORM_FEAT_KEYS:
            feat_slice = all_data[:, feature_idx : feature_idx + dim]
            mean_val = float(np.mean(feat_slice))
            std_val = float(np.std(feat_slice))
            stats["mean"][key] = mean_val
            stats["std"][key] = max(std_val, 1e-6)  # Avoid division by zero
            print(f"    {key}: mean={mean_val:.4f}, std={std_val:.4f}")
        feature_idx += dim

    return stats


def apply_normalization(features: np.ndarray, stats: dict) -> np.ndarray:
    """Apply z-score normalization to the first 9 features."""
    features = features.copy()
    feature_idx = 0

    for key in VNET_INPUT_KEYS:
        dim = FEATURE_DIMS[key]
        if key in NORM_FEAT_KEYS:
            mean = stats["mean"][key]
            std = stats["std"][key]
            features[:, feature_idx : feature_idx + dim] = (
                features[:, feature_idx : feature_idx + dim] - mean
            ) / std
        feature_idx += dim

    return features


def convert_sample(data: dict, labels: list, stats: dict) -> dict:
    """Convert a single sample to the expected format with normalization."""
    # Stack features
    full_features, num_notes = stack_features(data)

    # Apply normalization to base features (first 79 dims)
    full_features[:, :BASE_FEATURE_DIM] = apply_normalization(
        full_features[:, :BASE_FEATURE_DIM], stats
    )

    # Get note_location
    note_location = data.get("note_location", {})

    # Labels (first 19 values)
    labels_arr = np.array(labels[:19], dtype=np.float32)

    return {
        "input": full_features,
        "note_location": note_location,
        "labels": labels_arr,
        "num_notes": num_notes,
    }


def main():
    print("=" * 60)
    print("Converting VirtuosoNet Features to Dataset Format")
    print("=" * 60)

    # Paths
    input_dir = Path("data/percepiano_vnet")
    output_dir = Path("data/percepiano_vnet_converted")
    label_file = Path(
        "data/raw/PercePiano/label_2round_mean_reg_19_with0_rm_highstd0.json"
    )

    # Load labels
    print(f"\nLoading labels from {label_file}...")
    with open(label_file) as f:
        labels_dict = json.load(f)
    print(f"  Loaded {len(labels_dict)} labels")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get pickle files
    pkl_files = sorted(input_dir.glob("*.pkl"))
    print(f"\nFound {len(pkl_files)} samples")

    # PASS 1: Load all samples and compute normalization stats
    print("\n--- Pass 1: Computing normalization statistics ---")
    all_features = []
    valid_files = []

    for pkl_path in tqdm(pkl_files, desc="Loading"):
        sample_id = pkl_path.stem
        if sample_id not in labels_dict:
            continue

        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            features, _ = stack_features(data)
            all_features.append(features)
            valid_files.append(pkl_path)
        except Exception as e:
            print(f"\nError loading {sample_id}: {e}")

    print(f"\n  Valid samples: {len(valid_files)}")
    stats = compute_normalization_stats(all_features)

    # Save stats
    stats_path = output_dir / "stat.pkl"
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)
    print(f"\n  Saved normalization stats to {stats_path}")

    # PASS 2: Apply normalization and save
    print("\n--- Pass 2: Applying normalization and saving ---")
    success_count = 0
    errors = 0

    for i, pkl_path in enumerate(tqdm(valid_files, desc="Converting")):
        sample_id = pkl_path.stem

        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)

            labels = labels_dict[sample_id]
            converted = convert_sample(data, labels, stats)

            output_path = output_dir / pkl_path.name
            with open(output_path, "wb") as f:
                pickle.dump(converted, f)

            success_count += 1

        except Exception as e:
            print(f"\nError processing {sample_id}: {e}")
            errors += 1

    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"  Success: {success_count}")
    print(f"  Errors: {errors}")
    print(f"\nOutput: {output_dir}")
    print(f"Stats: {stats_path}")
    print(f"\nNext: Run diagnostics on converted data:")
    print(f"  uv run python scripts/diagnostics/diagnose_features.py {output_dir}")


if __name__ == "__main__":
    main()
