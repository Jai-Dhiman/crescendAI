#!/usr/bin/env python3
"""
Split Annotations into Train/Val/Test Sets

Splits JSONL annotation files while preserving piece-level grouping.
Ensures segments from the same piece stay in the same split.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def load_annotations(annotation_path: Path) -> List[Dict]:
    """Load annotations from JSONL file."""
    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

    annotations = []
    with open(annotation_path, 'r') as f:
        for line in f:
            if line.strip():
                annotations.append(json.loads(line))

    return annotations


def group_by_piece(annotations: List[Dict]) -> Dict[str, List[Dict]]:
    """Group annotations by piece_id to keep segments together."""
    grouped = defaultdict(list)

    for ann in annotations:
        piece_id = ann.get('metadata', {}).get('piece_id', 'unknown')
        grouped[piece_id].append(ann)

    return dict(grouped)


def split_pieces(
    grouped_annotations: Dict[str, List[Dict]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split annotations by piece into train/val/test sets.

    Args:
        grouped_annotations: Dict mapping piece_id -> list of segments
        train_ratio: Fraction of pieces for training
        val_ratio: Fraction of pieces for validation
        test_ratio: Fraction of pieces for testing
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_annotations, val_annotations, test_annotations)
    """
    random.seed(seed)

    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    # Get piece IDs and shuffle
    piece_ids = list(grouped_annotations.keys())
    random.shuffle(piece_ids)

    # Calculate split indices
    num_pieces = len(piece_ids)
    train_idx = int(num_pieces * train_ratio)
    val_idx = train_idx + int(num_pieces * val_ratio)

    # Split piece IDs
    train_pieces = piece_ids[:train_idx]
    val_pieces = piece_ids[train_idx:val_idx]
    test_pieces = piece_ids[val_idx:]

    # Collect annotations for each split
    train_annotations = []
    for piece_id in train_pieces:
        train_annotations.extend(grouped_annotations[piece_id])

    val_annotations = []
    for piece_id in val_pieces:
        val_annotations.extend(grouped_annotations[piece_id])

    test_annotations = []
    for piece_id in test_pieces:
        test_annotations.extend(grouped_annotations[piece_id])

    return train_annotations, val_annotations, test_annotations


def save_annotations(annotations: List[Dict], output_path: Path) -> None:
    """Save annotations to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for ann in annotations:
            f.write(json.dumps(ann) + '\n')

    print(f"Saved {len(annotations)} annotations to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Split annotations into train/val/test sets')
    parser.add_argument('--input', type=str, required=True,
                        help='Input JSONL annotation file')
    parser.add_argument('--train-output', type=str, required=True,
                        help='Output path for training annotations')
    parser.add_argument('--val-output', type=str, required=True,
                        help='Output path for validation annotations')
    parser.add_argument('--test-output', type=str, default=None,
                        help='Output path for test annotations (optional)')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Fraction of pieces for training (default: 0.8)')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                        help='Fraction of pieces for validation (default: 0.2)')
    parser.add_argument('--test-ratio', type=float, default=0.0,
                        help='Fraction of pieces for testing (default: 0.0)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    # Load annotations
    input_path = Path(args.input)
    print(f"Loading annotations from {input_path}")
    annotations = load_annotations(input_path)
    print(f"Loaded {len(annotations)} total annotations")

    # Group by piece
    grouped = group_by_piece(annotations)
    print(f"Grouped into {len(grouped)} pieces")

    # Split annotations
    if args.test_output is None:
        # Two-way split (train/val only)
        test_ratio = 0.0
        val_ratio = args.val_ratio
        train_ratio = 1.0 - val_ratio
    else:
        # Three-way split (train/val/test)
        train_ratio = args.train_ratio
        val_ratio = args.val_ratio
        test_ratio = args.test_ratio

    print(f"\nSplitting with ratios: train={train_ratio:.2f}, val={val_ratio:.2f}, test={test_ratio:.2f}")

    train_annotations, val_annotations, test_annotations = split_pieces(
        grouped,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=args.seed,
    )

    # Save splits
    train_output_path = Path(args.train_output)
    val_output_path = Path(args.val_output)

    save_annotations(train_annotations, train_output_path)
    save_annotations(val_annotations, val_output_path)

    if args.test_output is not None and test_ratio > 0:
        test_output_path = Path(args.test_output)
        save_annotations(test_annotations, test_output_path)

    # Print summary
    print("\n" + "="*60)
    print("Split complete!")
    print(f"Training:   {len(train_annotations)} segments from {sum(1 for p in grouped if any(a in train_annotations for a in grouped[p]))} pieces")
    print(f"Validation: {len(val_annotations)} segments from {sum(1 for p in grouped if any(a in val_annotations for a in grouped[p]))} pieces")
    if test_ratio > 0:
        print(f"Testing:    {len(test_annotations)} segments from {sum(1 for p in grouped if any(a in test_annotations for a in grouped[p]))} pieces")
    print("="*60)


if __name__ == '__main__':
    main()
