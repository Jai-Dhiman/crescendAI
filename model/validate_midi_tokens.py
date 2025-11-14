#!/usr/bin/env python3
"""
MIDI Token Validation Script

Validates that all MIDI tokens in the dataset are within vocabulary bounds.
Run this BEFORE training to catch data corruption issues early.

Usage:
    python validate_midi_tokens.py --annotation-file path/to/annotations.jsonl
"""

import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from src.data.midi_processing import load_midi, encode_octuple_midi


# Vocabulary sizes (must match MIDIBertEncoder)
VOCAB_SIZES = {
    'type': 5,
    'beat': 16,
    'position': 16,
    'pitch': 88,
    'duration': 128,
    'velocity': 128,
    'instrument': 1,
    'bar': 512,
}


def validate_tokens(tokens: np.ndarray, sample_id: str) -> dict:
    """
    Validate token array is within vocabulary bounds.

    Args:
        tokens: Array of shape [events, 8]
        sample_id: Identifier for this sample (for error reporting)

    Returns:
        Dictionary with validation results
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }

    if len(tokens) == 0:
        results['warnings'].append(f"{sample_id}: Empty token sequence")
        return results

    # Check dimensions
    if tokens.shape[1] != 8:
        results['valid'] = False
        results['errors'].append(
            f"{sample_id}: Invalid shape {tokens.shape}, expected [N, 8]"
        )
        return results

    # Validate each dimension
    dim_names = ['type', 'beat', 'position', 'pitch', 'duration', 'velocity', 'instrument', 'bar']

    for dim_idx, dim_name in enumerate(dim_names):
        values = tokens[:, dim_idx]
        vocab_size = VOCAB_SIZES[dim_name]

        # Check for out-of-bounds values
        min_val = values.min()
        max_val = values.max()

        if min_val < 0:
            results['valid'] = False
            results['errors'].append(
                f"{sample_id}: {dim_name} has negative values (min={min_val})"
            )

        if max_val >= vocab_size:
            results['valid'] = False
            results['errors'].append(
                f"{sample_id}: {dim_name} exceeds vocab size "
                f"(max={max_val}, vocab_size={vocab_size})"
            )

        # Store stats
        results['stats'][dim_name] = {
            'min': int(min_val),
            'max': int(max_val),
            'mean': float(values.mean()),
            'vocab_size': vocab_size
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Validate MIDI tokens in dataset")
    parser.add_argument(
        '--annotation-file',
        type=str,
        required=True,
        help='Path to annotation JSONL file'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to validate (default: all)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed validation results'
    )
    args = parser.parse_args()

    annotation_path = Path(args.annotation_file)

    if not annotation_path.exists():
        print(f"Error: Annotation file not found: {annotation_path}")
        return 1

    # Load annotations
    print(f"Loading annotations from {annotation_path}...")
    annotations = []
    with open(annotation_path, 'r') as f:
        for line in f:
            if line.strip():
                annotations.append(json.loads(line))

    if args.max_samples is not None:
        annotations = annotations[:args.max_samples]

    print(f"Validating {len(annotations)} samples...\n")

    # Validation results
    total_samples = 0
    valid_samples = 0
    failed_samples = 0
    empty_midi = 0
    errors_by_dimension = {dim: 0 for dim in VOCAB_SIZES.keys()}

    # Global statistics
    global_stats = {dim: {'min': float('inf'), 'max': float('-inf')}
                   for dim in VOCAB_SIZES.keys()}

    # Validate each sample
    for idx, annotation in enumerate(tqdm(annotations, desc="Validating")):
        total_samples += 1

        # Skip if no MIDI
        if 'midi_path' not in annotation or not annotation['midi_path']:
            empty_midi += 1
            continue

        midi_path = Path(annotation['midi_path'])

        try:
            # Load and tokenize MIDI
            midi = load_midi(str(midi_path))
            tokens = encode_octuple_midi(midi)

            # Validate tokens
            results = validate_tokens(tokens, f"sample_{idx}")

            if results['valid']:
                valid_samples += 1

                # Update global statistics
                for dim_name, stats in results['stats'].items():
                    global_stats[dim_name]['min'] = min(
                        global_stats[dim_name]['min'],
                        stats['min']
                    )
                    global_stats[dim_name]['max'] = max(
                        global_stats[dim_name]['max'],
                        stats['max']
                    )
            else:
                failed_samples += 1

                # Count errors by dimension
                for error in results['errors']:
                    for dim_name in VOCAB_SIZES.keys():
                        if dim_name in error:
                            errors_by_dimension[dim_name] += 1

                if args.verbose:
                    print(f"\nValidation failed for sample {idx}:")
                    print(f"  MIDI: {midi_path}")
                    for error in results['errors']:
                        print(f"  - {error}")

        except Exception as e:
            failed_samples += 1
            if args.verbose:
                print(f"\nFailed to process sample {idx}: {e}")
                print(f"  MIDI: {midi_path}")

    # Print summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    print(f"Total samples: {total_samples}")
    print(f"Valid samples: {valid_samples} ({100*valid_samples/total_samples:.1f}%)")
    print(f"Failed samples: {failed_samples} ({100*failed_samples/total_samples:.1f}%)")
    print(f"Empty MIDI: {empty_midi}")
    print()

    if failed_samples > 0:
        print("Errors by dimension:")
        for dim_name, count in errors_by_dimension.items():
            if count > 0:
                print(f"  {dim_name}: {count} samples")
        print()

    print("Global token statistics:")
    print(f"{'Dimension':<15} {'Min':>8} {'Max':>8} {'Vocab Size':>12} {'Status':>10}")
    print("-" * 60)
    for dim_name in VOCAB_SIZES.keys():
        min_val = global_stats[dim_name]['min']
        max_val = global_stats[dim_name]['max']
        vocab_size = VOCAB_SIZES[dim_name]

        # Determine status
        if min_val == float('inf'):
            status = 'NO DATA'
        elif min_val < 0 or max_val >= vocab_size:
            status = 'ERROR'
        else:
            status = 'OK'

        # Format values
        min_str = str(int(min_val)) if min_val != float('inf') else 'N/A'
        max_str = str(int(max_val)) if max_val != float('-inf') else 'N/A'

        print(f"{dim_name:<15} {min_str:>8} {max_str:>8} {vocab_size:>12} {status:>10}")

    print("="*80)

    if failed_samples == 0:
        print("\n✓ All samples passed validation!")
        print("Dataset is ready for training.")
        return 0
    else:
        print(f"\n✗ {failed_samples} samples failed validation!")
        print("Please fix data issues before training.")
        return 1


if __name__ == '__main__':
    exit(main())
