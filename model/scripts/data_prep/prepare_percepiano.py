#!/usr/bin/env python3
"""
Prepare PercePiano dataset for training.

Downloads the dataset from GitHub and processes annotations into our format.
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import random


# All 19 PercePiano dimensions (matching reference implementation exactly)
# Labels are kept in original 0-1 scale (no aggregation, no scaling)
PERCEPIANO_DIMENSIONS = [
    "timing",              # 0: Stable <-> Unstable
    "articulation_length", # 1: Short <-> Long
    "articulation_touch",  # 2: Soft/Cushioned <-> Hard/Solid
    "pedal_amount",        # 3: Sparse/Dry <-> Saturated/Wet
    "pedal_clarity",       # 4: Clean <-> Blurred
    "timbre_variety",      # 5: Even <-> Colorful
    "timbre_depth",        # 6: Shallow <-> Rich
    "timbre_brightness",   # 7: Bright <-> Dark
    "timbre_loudness",     # 8: Soft <-> Loud
    "dynamic_range",       # 9: Little Range <-> Large Range
    "tempo",               # 10: Fast-paced <-> Slow-paced
    "space",               # 11: Flat <-> Spacious
    "balance",             # 12: Disproportioned <-> Balanced
    "drama",               # 13: Pure <-> Dramatic
    "mood_valence",        # 14: Optimistic <-> Dark
    "mood_energy",         # 15: Low Energy <-> High Energy
    "mood_imagination",    # 16: Honest <-> Imaginative
    "sophistication",      # 17: Sophisticated/Mellow <-> Raw/Crude
    "interpretation",      # 18: Unsatisfactory <-> Convincing
]


def clone_percepiano(data_dir: Path) -> Path:
    """Clone PercePiano repository if not exists."""
    repo_dir = data_dir / "PercePiano"

    if repo_dir.exists():
        print(f"PercePiano already exists at {repo_dir}")
        return repo_dir

    print("Cloning PercePiano repository...")
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/JonghoKimSNU/PercePiano.git", str(repo_dir)],
        check=True,
    )
    print(f"Cloned to {repo_dir}")
    return repo_dir


def load_annotations(repo_dir: Path) -> Dict[str, List[float]]:
    """Load mean annotations from PercePiano."""
    labels_file = repo_dir / "labels" / "label_2round_mean_reg_19_with0_rm_highstd0.json"

    if not labels_file.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_file}")

    with open(labels_file, "r") as f:
        annotations = json.load(f)

    print(f"Loaded {len(annotations)} annotated performances")
    return annotations


def find_midi_files(repo_dir: Path) -> Dict[str, Path]:
    """Find all performance MIDI files."""
    midi_dir = repo_dir / "virtuoso" / "data" / "all_2rounds"

    if not midi_dir.exists():
        raise FileNotFoundError(f"MIDI directory not found: {midi_dir}")

    midi_files = {}
    for midi_file in midi_dir.glob("*.mid"):
        # Extract performance name from filename
        # Format: [piece]_[bars]bars_[segment]_[player].mid
        name = midi_file.stem
        midi_files[name] = midi_file

    print(f"Found {len(midi_files)} MIDI files")
    return midi_files


def map_dimensions(percepiano_scores: List[float]) -> Dict[str, float]:
    """Return all 19 PercePiano dimensions in 0-1 scale (no aggregation).

    This matches the reference implementation exactly.
    percepiano_scores has 20 values, last one is performer ID (ignored).
    """
    return {
        dim: percepiano_scores[i]
        for i, dim in enumerate(PERCEPIANO_DIMENSIONS)
    }


def extract_piece_name(name: str) -> str:
    """
    Extract piece name from PercePiano sample name.

    Format: [composer]_[piece]_[section]_[bars]bars_[segment]_[player]
    e.g. Schubert_D935_no.3_4bars_1_25 -> Schubert_D935_no.3_4bars
    """
    parts = name.split("_")
    # Find the part containing 'bars' and include everything up to it
    for i, part in enumerate(parts):
        if "bars" in part.lower():
            return "_".join(parts[: i + 1])
    # Fallback: return first two parts (composer_piece)
    return "_".join(parts[:2]) if len(parts) >= 2 else parts[0]


def create_splits(
    samples: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Create train/val/test splits based on piece (not performer).

    This ensures the model generalizes to new pieces, not just new performers
    of the same piece.
    """
    # Group by piece
    pieces = {}
    for sample in samples:
        # Extract piece name (everything up to and including 'bars')
        name = sample["name"]
        piece = extract_piece_name(name)
        if piece not in pieces:
            pieces[piece] = []
        pieces[piece].append(sample)

    # Shuffle pieces
    piece_names = list(pieces.keys())
    random.seed(seed)
    random.shuffle(piece_names)

    # Split pieces
    n_pieces = len(piece_names)
    n_train = int(n_pieces * train_ratio)
    n_val = int(n_pieces * val_ratio)

    train_pieces = piece_names[:n_train]
    val_pieces = piece_names[n_train : n_train + n_val]
    test_pieces = piece_names[n_train + n_val :]

    # Collect samples
    train_samples = [s for p in train_pieces for s in pieces[p]]
    val_samples = [s for p in val_pieces for s in pieces[p]]
    test_samples = [s for p in test_pieces for s in pieces[p]]

    print(f"Split: {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test")
    print(f"Pieces: {len(train_pieces)} train, {len(val_pieces)} val, {len(test_pieces)} test")

    return train_samples, val_samples, test_samples


def prepare_dataset(data_dir: Path, output_dir: Path):
    """Main function to prepare the dataset."""
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clone repository
    repo_dir = clone_percepiano(data_dir)

    # Load annotations and MIDI files
    annotations = load_annotations(repo_dir)
    midi_files = find_midi_files(repo_dir)

    # Match annotations to MIDI files
    samples = []
    missing = 0
    for name, scores in annotations.items():
        if name in midi_files:
            sample = {
                "name": name,
                "midi_path": str(midi_files[name]),
                "percepiano_scores": scores,
                "scores": map_dimensions(scores),
            }
            samples.append(sample)
        else:
            missing += 1

    print(f"Matched {len(samples)} samples, {missing} missing MIDI files")

    # Create splits
    train, val, test = create_splits(samples)

    # Save splits
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        output_file = output_dir / f"percepiano_{split_name}.json"
        with open(output_file, "w") as f:
            json.dump(split_data, f, indent=2)
        print(f"Saved {output_file}")

    # Save dimension info for reference
    dimension_file = output_dir / "dimensions.json"
    with open(dimension_file, "w") as f:
        json.dump(
            {
                "dimensions": PERCEPIANO_DIMENSIONS,
                "num_dimensions": len(PERCEPIANO_DIMENSIONS),
                "scale": "0-1",
            },
            f,
            indent=2,
        )
    print(f"Saved {dimension_file}")

    # Print statistics
    print("\nDataset Statistics:")
    print(f"  Total samples: {len(samples)}")
    print(f"  Train: {len(train)} ({len(train)/len(samples)*100:.1f}%)")
    print(f"  Val: {len(val)} ({len(val)/len(samples)*100:.1f}%)")
    print(f"  Test: {len(test)} ({len(test)/len(samples)*100:.1f}%)")

    # Print score distribution for each dimension
    print("\nScore Distribution (mean +/- std):")
    for dim in PERCEPIANO_DIMENSIONS:
        values = [s["scores"][dim] for s in samples]
        mean = sum(values) / len(values)
        std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
        print(f"  {dim}: {mean:.3f} +/- {std:.3f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare PercePiano dataset")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory to store raw data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory to store processed data",
    )
    args = parser.parse_args()

    prepare_dataset(args.data_dir, args.output_dir)
