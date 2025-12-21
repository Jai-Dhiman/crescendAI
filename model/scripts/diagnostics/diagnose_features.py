#!/usr/bin/env python3
"""
Quick diagnostic script to examine preprocessed VirtuosoNet features.

Run this in the Thunder Compute runtime to check what the model actually receives.

Usage:
    python scripts/diagnose_features.py /tmp/percepiano_data/percepiano_vnet/train
"""

import sys
import pickle
import numpy as np
from pathlib import Path


def diagnose_sample(pkl_path: Path) -> dict:
    """Analyze a single preprocessed sample."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    input_features = data['input']
    labels = data['labels']
    num_notes = data.get('num_notes', input_features.shape[0])

    print(f"\n{'='*60}")
    print(f"Sample: {pkl_path.stem}")
    print(f"{'='*60}")

    print(f"\nFull features shape: {input_features.shape}")
    print(f"Num notes: {num_notes}")

    # Analyze first 79 features (what model receives)
    base_features = input_features[:num_notes, :79]
    print(f"\n--- First 79 features (MODEL INPUT) ---")
    print(f"  Shape: {base_features.shape}")
    print(f"  Range: [{base_features.min():.4f}, {base_features.max():.4f}]")
    print(f"  Mean: {base_features.mean():.4f}, Std: {base_features.std():.4f}")

    # Check first few scalar features (should be normalized)
    print(f"\n  Per-feature breakdown (first 14 scalar features):")
    feature_names = [
        'midi_pitch', 'duration', 'beat_importance', 'measure_length',
        'qpm_primo', 'section_tempo', 'following_rest', 'distance_from_abs_dynamic',
        'distance_from_recent_tempo', 'beat_position', 'xml_position', 'grace_order',
        'preceded_by_grace_note', 'followed_by_fermata_rest'
    ]
    for i, name in enumerate(feature_names):
        feat = base_features[:, i]
        print(f"    [{i:2d}] {name:25s}: range=[{feat.min():8.3f}, {feat.max():8.3f}], mean={feat.mean():7.3f}")

    # Check unnorm features (indices 79-83)
    if input_features.shape[1] > 79:
        unnorm_features = input_features[:num_notes, 79:]
        print(f"\n--- Last {unnorm_features.shape[1]} features (UNNORM, should NOT go to model) ---")
        print(f"  Range: [{unnorm_features.min():.4f}, {unnorm_features.max():.4f}]")
        print(f"  Mean: {unnorm_features.mean():.4f}")

        unnorm_names = ['midi_pitch_unnorm', 'duration_unnorm', 'beat_importance_unnorm',
                       'measure_length_unnorm', 'following_rest_unnorm']
        for i, name in enumerate(unnorm_names):
            if i < unnorm_features.shape[1]:
                feat = unnorm_features[:, i]
                print(f"    [{79+i:2d}] {name:25s}: range=[{feat.min():8.3f}, {feat.max():8.3f}]")

    # Check labels
    print(f"\n--- Labels ---")
    print(f"  Shape: {labels.shape}")
    print(f"  Range: [{labels.min():.4f}, {labels.max():.4f}]")
    print(f"  Mean: {labels.mean():.4f}")

    return {
        'base_max': float(base_features.max()),
        'base_min': float(base_features.min()),
        'midi_pitch_max': float(base_features[:, 0].max()),
        'midi_pitch_min': float(base_features[:, 0].min()),
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/diagnose_features.py <vnet_dir>")
        print("  e.g.: python scripts/diagnose_features.py /tmp/percepiano_data/percepiano_vnet/train")
        sys.exit(1)

    vnet_dir = Path(sys.argv[1])
    pkl_files = list(vnet_dir.glob('*.pkl'))

    if not pkl_files:
        print(f"No .pkl files found in {vnet_dir}")
        sys.exit(1)

    print(f"Found {len(pkl_files)} samples in {vnet_dir}")

    # Analyze first 3 samples
    results = []
    for pkl_path in pkl_files[:3]:
        r = diagnose_sample(pkl_path)
        results.append(r)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    max_base_max = max(r['base_max'] for r in results)
    max_midi_pitch = max(r['midi_pitch_max'] for r in results)

    print(f"\nMax value in first 79 features: {max_base_max:.4f}")
    print(f"Max value in midi_pitch (idx 0): {max_midi_pitch:.4f}")

    if max_midi_pitch > 10:
        print(f"\n[CRITICAL] midi_pitch feature has max={max_midi_pitch:.1f}")
        print("  This should be z-score normalized (typical range: -2 to +2)")
        print("  Raw MIDI pitch values (21-108) mean NORMALIZATION WAS NOT APPLIED!")
        print("\n  FIX: Delete percepiano_vnet directory and re-run preprocessing")
    else:
        print(f"\n[OK] Features appear to be normalized correctly")


if __name__ == '__main__':
    main()
