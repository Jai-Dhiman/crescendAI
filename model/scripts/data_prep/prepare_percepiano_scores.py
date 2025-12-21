#!/usr/bin/env python3
"""
Prepare PercePiano dataset with score alignment information.

Downloads/processes:
1. PercePiano MIDI files and annotations
2. Reference MusicXML scores
3. Creates mapping between performances and scores

Usage:
    python scripts/prepare_percepiano_scores.py --percepiano-dir /path/to/PercePiano --output-dir data/processed
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def find_score_for_performance(
    midi_filename: str,
    score_dir: Path,
    piece_mapping: Dict[str, str],
) -> Optional[str]:
    """
    Find the matching score file for a performance MIDI.

    The PercePiano dataset has a naming convention that links performances to pieces.
    This function attempts to match a performance MIDI to its reference score.

    Args:
        midi_filename: Name of the performance MIDI file
        score_dir: Directory containing MusicXML scores
        piece_mapping: Dictionary mapping performance patterns to piece names

    Returns:
        Relative path to score file, or None if not found
    """
    # Try to extract piece identifier from MIDI filename
    # PercePiano naming: {piece}_{performer}_{segment}.mid or similar

    midi_stem = Path(midi_filename).stem

    # Try direct mapping first
    if midi_stem in piece_mapping:
        score_name = piece_mapping[midi_stem]
        score_path = score_dir / f"{score_name}.musicxml"
        if score_path.exists():
            return f"{score_name}.musicxml"

    # Try pattern matching
    for pattern, score_name in piece_mapping.items():
        if pattern in midi_stem:
            score_path = score_dir / f"{score_name}.musicxml"
            if score_path.exists():
                return f"{score_name}.musicxml"

    # Try to find a score with similar name
    for score_file in score_dir.glob("*.musicxml"):
        score_stem = score_file.stem.lower()
        midi_lower = midi_stem.lower()

        # Check for common substrings
        if len(score_stem) > 3 and score_stem in midi_lower:
            return score_file.name
        if len(midi_lower) > 3 and midi_lower[:10] in score_stem:
            return score_file.name

    return None


def create_piece_mapping(score_dir: Path) -> Dict[str, str]:
    """
    Create a mapping from performance identifiers to piece names.

    This is based on the PercePiano dataset structure where scores
    are in virtuoso/data/all_2rounds/ or similar directories.

    Args:
        score_dir: Directory containing score files

    Returns:
        Dictionary mapping patterns to score names
    """
    mapping = {}

    for score_file in score_dir.glob("*.musicxml"):
        score_name = score_file.stem

        # Create multiple lookup patterns from the score name
        # Score names often contain composer and piece info
        patterns = [
            score_name,
            score_name.lower(),
            score_name.replace("_", " "),
            score_name.split("_")[0] if "_" in score_name else score_name,
        ]

        for pattern in patterns:
            if pattern not in mapping:
                mapping[pattern] = score_name

    return mapping


def load_percepiano_annotations(annotations_path: Path) -> Dict[str, Dict]:
    """
    Load PercePiano annotations file.

    The annotations contain perceptual ratings for each segment.

    Args:
        annotations_path: Path to annotations JSON file

    Returns:
        Dictionary mapping segment names to their annotations
    """
    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_path}")

    with open(annotations_path, "r") as f:
        annotations = json.load(f)

    return annotations


def process_annotations_to_scores(
    annotations: Dict[str, Dict],
    dimensions: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Process raw annotations into normalized scores.

    PercePiano uses a 1-7 Likert scale for most dimensions.
    This normalizes to 0-1 range.

    Args:
        annotations: Raw annotation dictionary
        dimensions: List of dimension names to extract

    Returns:
        Dictionary mapping segment names to dimension scores
    """
    scores = {}

    for segment_name, segment_data in annotations.items():
        segment_scores = {}

        for dim in dimensions:
            if dim in segment_data:
                # Get raw value (typically 1-7 scale)
                raw_value = segment_data[dim]

                if isinstance(raw_value, (int, float)):
                    # Normalize from 1-7 to 0-1
                    normalized = (raw_value - 1) / 6.0
                    normalized = max(0.0, min(1.0, normalized))
                elif isinstance(raw_value, list):
                    # Average if multiple ratings
                    avg_value = np.mean(raw_value)
                    normalized = (avg_value - 1) / 6.0
                    normalized = max(0.0, min(1.0, normalized))
                else:
                    normalized = 0.5  # Default to middle

                segment_scores[dim] = normalized
            else:
                segment_scores[dim] = 0.5  # Default

        scores[segment_name] = segment_scores

    return scores


def create_dataset_json(
    midi_dir: Path,
    score_dir: Optional[Path],
    annotations: Dict[str, Dict],
    dimensions: List[str],
    output_path: Path,
    split: str = "all",
) -> List[Dict]:
    """
    Create JSON dataset file linking MIDIs to scores and annotations.

    Args:
        midi_dir: Directory containing performance MIDI files
        score_dir: Directory containing MusicXML scores (optional)
        annotations: Annotation dictionary
        dimensions: List of dimensions
        output_path: Path to output JSON file
        split: Dataset split name

    Returns:
        List of sample dictionaries
    """
    samples = []
    piece_mapping = create_piece_mapping(score_dir) if score_dir else {}

    # Process scores
    processed_scores = process_annotations_to_scores(annotations, dimensions)

    for midi_file in sorted(midi_dir.glob("*.mid*")):
        midi_name = midi_file.stem

        # Find matching annotation
        if midi_name not in processed_scores:
            # Try variations
            found = False
            for ann_name in processed_scores.keys():
                if midi_name in ann_name or ann_name in midi_name:
                    midi_name_for_scores = ann_name
                    found = True
                    break
            if not found:
                print(f"Warning: No annotation found for {midi_name}, skipping")
                continue
        else:
            midi_name_for_scores = midi_name

        # Find matching score
        score_path = None
        if score_dir:
            score_path = find_score_for_performance(
                midi_file.name,
                score_dir,
                piece_mapping,
            )

        sample = {
            "name": midi_name,
            "midi_path": str(midi_file),
            "score_path": score_path,
            "scores": processed_scores[midi_name_for_scores],
            "split": split,
        }

        samples.append(sample)

    # Save to JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"Created {output_path} with {len(samples)} samples")
    return samples


def split_dataset(
    samples: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split samples into train/val/test sets.

    Uses piece-based splitting to avoid data leakage (same piece
    in train and test with different performers).

    Args:
        samples: List of sample dictionaries
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed

    Returns:
        Tuple of (train_samples, val_samples, test_samples)
    """
    np.random.seed(seed)

    # Group by piece (inferred from MIDI name)
    piece_groups = {}
    for sample in samples:
        # Extract piece identifier (first part of name before performer/segment)
        name = sample["name"]
        parts = name.split("_")
        piece_id = parts[0] if len(parts) > 1 else name

        if piece_id not in piece_groups:
            piece_groups[piece_id] = []
        piece_groups[piece_id].append(sample)

    # Shuffle pieces
    piece_ids = list(piece_groups.keys())
    np.random.shuffle(piece_ids)

    # Calculate split points
    n_pieces = len(piece_ids)
    n_train = int(n_pieces * train_ratio)
    n_val = int(n_pieces * val_ratio)

    train_pieces = piece_ids[:n_train]
    val_pieces = piece_ids[n_train:n_train + n_val]
    test_pieces = piece_ids[n_train + n_val:]

    # Collect samples
    train_samples = []
    val_samples = []
    test_samples = []

    for piece_id in train_pieces:
        train_samples.extend(piece_groups[piece_id])
    for piece_id in val_pieces:
        val_samples.extend(piece_groups[piece_id])
    for piece_id in test_pieces:
        test_samples.extend(piece_groups[piece_id])

    # Update split labels
    for s in train_samples:
        s["split"] = "train"
    for s in val_samples:
        s["split"] = "val"
    for s in test_samples:
        s["split"] = "test"

    return train_samples, val_samples, test_samples


def main():
    parser = argparse.ArgumentParser(
        description="Prepare PercePiano dataset with score alignment"
    )
    parser.add_argument(
        "--percepiano-dir",
        type=Path,
        required=True,
        help="Path to PercePiano repository root",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting",
    )
    args = parser.parse_args()

    # Validate paths
    percepiano_dir = args.percepiano_dir
    if not percepiano_dir.exists():
        print(f"Error: PercePiano directory not found: {percepiano_dir}")
        sys.exit(1)

    # Find subdirectories
    # PercePiano structure:
    # - virtuoso/data/all_2rounds/*.mid (performance MIDI)
    # - virtuoso/data/score_xml/*.musicxml (scores) or all_2rounds/*.musicxml
    # - data/annotations.json (annotations)

    midi_dir = percepiano_dir / "virtuoso" / "data" / "all_2rounds"
    if not midi_dir.exists():
        midi_dir = percepiano_dir / "data" / "midi"
    if not midi_dir.exists():
        # Try to find MIDI files anywhere
        midi_files = list(percepiano_dir.rglob("*.mid"))
        if midi_files:
            midi_dir = midi_files[0].parent
        else:
            print(f"Error: Could not find MIDI directory in {percepiano_dir}")
            sys.exit(1)

    print(f"Using MIDI directory: {midi_dir}")

    # Find score directory
    score_dir = percepiano_dir / "virtuoso" / "data" / "all_2rounds"
    if not score_dir.exists() or not list(score_dir.glob("*.musicxml")):
        score_dir = percepiano_dir / "virtuoso" / "data" / "score_xml"
    if not score_dir.exists() or not list(score_dir.glob("*.musicxml")):
        score_dir = percepiano_dir / "data" / "scores"
    if not score_dir.exists() or not list(score_dir.glob("*.musicxml")):
        print("Warning: Score directory not found, proceeding without scores")
        score_dir = None
    else:
        print(f"Using score directory: {score_dir}")

    # Find annotations
    annotations_path = percepiano_dir / "data" / "annotations.json"
    if not annotations_path.exists():
        annotations_path = percepiano_dir / "annotations.json"
    if not annotations_path.exists():
        # Try to find annotations file
        ann_files = list(percepiano_dir.rglob("*annotation*.json"))
        if ann_files:
            annotations_path = ann_files[0]
        else:
            print(f"Error: Annotations file not found in {percepiano_dir}")
            sys.exit(1)

    print(f"Using annotations: {annotations_path}")

    # Load annotations
    annotations = load_percepiano_annotations(annotations_path)
    print(f"Loaded {len(annotations)} annotations")

    # Define dimensions (all 19 PercePiano dimensions)
    dimensions = [
        "timing",
        "articulation_length",
        "articulation_touch",
        "pedal_amount",
        "pedal_clarity",
        "timbre_variety",
        "timbre_depth",
        "timbre_brightness",
        "timbre_loudness",
        "dynamic_range",
        "tempo",
        "space",
        "balance",
        "drama",
        "mood_valence",
        "mood_energy",
        "mood_imagination",
        "sophistication",
        "interpretation",
    ]

    # Create full dataset JSON
    output_dir = args.output_dir
    all_samples = create_dataset_json(
        midi_dir=midi_dir,
        score_dir=score_dir,
        annotations=annotations,
        dimensions=dimensions,
        output_path=output_dir / "percepiano_all.json",
        split="all",
    )

    if not all_samples:
        print("Error: No samples created. Check MIDI and annotation matching.")
        sys.exit(1)

    # Split dataset
    train_samples, val_samples, test_samples = split_dataset(
        all_samples,
        seed=args.seed,
    )

    # Save splits
    with open(output_dir / "percepiano_train.json", "w") as f:
        json.dump(train_samples, f, indent=2)
    print(f"Train: {len(train_samples)} samples")

    with open(output_dir / "percepiano_val.json", "w") as f:
        json.dump(val_samples, f, indent=2)
    print(f"Val: {len(val_samples)} samples")

    with open(output_dir / "percepiano_test.json", "w") as f:
        json.dump(test_samples, f, indent=2)
    print(f"Test: {len(test_samples)} samples")

    # Print statistics
    print("\n--- Dataset Statistics ---")
    print(f"Total samples: {len(all_samples)}")
    print(f"Samples with scores: {sum(1 for s in all_samples if s.get('score_path'))}")

    # Score distribution by dimension
    print("\nScore distributions (mean +/- std):")
    for dim in dimensions:
        values = [s["scores"].get(dim, 0.5) for s in all_samples]
        print(f"  {dim}: {np.mean(values):.3f} +/- {np.std(values):.3f}")

    print(f"\nDataset files saved to: {output_dir}")


if __name__ == "__main__":
    main()
