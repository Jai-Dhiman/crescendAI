"""
K-Fold Cross-Validation Split Utilities for PercePiano

Implements piece-based k-fold splitting following the PercePiano paper methodology:
- All performances of the same piece stay in the same fold
- ~15% held out as test set
- Remaining ~85% split into 4 folds for cross-validation
"""

import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


def extract_piece_id(filename: str) -> str:
    """
    Extract piece identifier from sample filename.

    Filename format: Composer_Piece_Movement_Length_PerformerID_SegmentID.pkl
    Example: Beethoven_WoO80_thema_8bars_1_1.pkl -> Beethoven_WoO80_thema_8bars

    The piece ID is the first 4 underscore-separated parts, which uniquely
    identifies a musical excerpt (same piece/movement/length).

    Args:
        filename: Sample filename (with or without .pkl extension)

    Returns:
        Piece identifier string
    """
    # Remove extension if present
    name = filename.replace(".pkl", "")

    # Split by underscore and take first 4 parts
    parts = name.split("_")
    if len(parts) < 4:
        raise ValueError(
            f"Invalid filename format: {filename}. Expected at least 4 underscore-separated parts."
        )

    return "_".join(parts[:4])


def get_all_samples(data_dir: Union[str, Path]) -> List[str]:
    """
    Get all sample filenames from the data directory.

    Searches train/, val/, and test/ subdirectories for .pkl files.

    Args:
        data_dir: Path to data directory containing train/val/test subdirs

    Returns:
        List of sample filenames (without path, with .pkl extension)
    """
    data_dir = Path(data_dir)
    samples = []

    for subdir in ["train", "val", "test"]:
        subdir_path = data_dir / subdir
        if subdir_path.exists():
            pkl_files = list(subdir_path.glob("*.pkl"))
            samples.extend([f.name for f in pkl_files])

    if not samples:
        raise ValueError(f"No .pkl files found in {data_dir}/[train|val|test]/")

    return samples


def group_samples_by_piece(samples: List[str]) -> Dict[str, List[str]]:
    """
    Group samples by their piece ID.

    Args:
        samples: List of sample filenames

    Returns:
        Dictionary mapping piece_id -> list of sample filenames
    """
    piece_groups = defaultdict(list)

    for sample in samples:
        piece_id = extract_piece_id(sample)
        piece_groups[piece_id].append(sample)

    return dict(piece_groups)


def create_piece_based_folds(
    data_dir: Union[str, Path],
    n_folds: int = 4,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, Dict[str, Union[int, str]]]:
    """
    Create piece-based k-fold assignments.

    All samples from the same piece are assigned to the same fold.
    Test set is held out first, then remaining pieces are distributed across folds.

    Args:
        data_dir: Path to data directory
        n_folds: Number of CV folds (default: 4)
        test_ratio: Fraction of pieces for test set (default: 0.15)
        seed: Random seed for reproducibility

    Returns:
        Dictionary mapping sample_name -> {'fold': 0-3 or 'test', 'piece_id': str}
    """
    random.seed(seed)

    # Get all samples and group by piece
    samples = get_all_samples(data_dir)
    piece_groups = group_samples_by_piece(samples)

    # Get list of unique pieces
    pieces = list(piece_groups.keys())
    random.shuffle(pieces)

    # Calculate number of test pieces
    n_test = max(1, int(len(pieces) * test_ratio))

    # Assign pieces to test set
    test_pieces = set(pieces[:n_test])
    cv_pieces = pieces[n_test:]

    # Distribute CV pieces across folds
    fold_assignments = {}

    for i, piece in enumerate(cv_pieces):
        fold_id = i % n_folds
        for sample in piece_groups[piece]:
            fold_assignments[sample] = {"fold": fold_id, "piece_id": piece}

    # Assign test pieces
    for piece in test_pieces:
        for sample in piece_groups[piece]:
            fold_assignments[sample] = {"fold": "test", "piece_id": piece}

    return fold_assignments


def save_fold_assignments(
    assignments: Dict[str, Dict[str, Union[int, str]]], output_path: Union[str, Path]
) -> None:
    """
    Save fold assignments to JSON file.

    Args:
        assignments: Fold assignments dictionary
        output_path: Path to output JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(assignments, f, indent=2)

    print(f"Saved fold assignments to {output_path}")


def load_fold_assignments(
    path: Union[str, Path],
) -> Dict[str, Dict[str, Union[int, str]]]:
    """
    Load fold assignments from JSON file.

    Args:
        path: Path to JSON file

    Returns:
        Fold assignments dictionary
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Fold assignments file not found: {path}")

    with open(path, "r") as f:
        assignments = json.load(f)

    return assignments


def get_fold_samples(
    assignments: Dict[str, Dict[str, Union[int, str]]], fold_id: int, mode: str
) -> List[str]:
    """
    Get samples for a specific fold and mode.

    Args:
        assignments: Fold assignments dictionary
        fold_id: Fold ID (0 to n_folds-1)
        mode: 'train' or 'val'
            - 'train': Returns samples from all folds EXCEPT fold_id
            - 'val': Returns samples from fold_id only

    Returns:
        List of sample filenames for this fold/mode
    """
    if mode not in ("train", "val"):
        raise ValueError(f"mode must be 'train' or 'val', got: {mode}")

    samples = []

    for sample, info in assignments.items():
        sample_fold = info["fold"]

        # Skip test samples
        if sample_fold == "test":
            continue

        if mode == "val":
            # Validation: use only samples from this fold
            if sample_fold == fold_id:
                samples.append(sample)
        else:
            # Training: use samples from all OTHER folds
            if sample_fold != fold_id:
                samples.append(sample)

    return samples


def get_test_samples(assignments: Dict[str, Dict[str, Union[int, str]]]) -> List[str]:
    """
    Get test set samples.

    Args:
        assignments: Fold assignments dictionary

    Returns:
        List of test sample filenames
    """
    return [sample for sample, info in assignments.items() if info["fold"] == "test"]


def print_fold_statistics(
    assignments: Dict[str, Dict[str, Union[int, str]]], n_folds: int = 4
) -> None:
    """
    Print statistics about fold assignments.

    Args:
        assignments: Fold assignments dictionary
        n_folds: Number of CV folds
    """
    # Count samples per fold
    fold_counts = defaultdict(int)
    piece_counts = defaultdict(set)

    for sample, info in assignments.items():
        fold = info["fold"]
        piece_id = info["piece_id"]
        fold_counts[fold] += 1
        piece_counts[fold].add(piece_id)

    print("\n=== Fold Assignment Statistics ===")
    print(f"Total samples: {len(assignments)}")
    print(
        f"Total pieces: {len(set(info['piece_id'] for info in assignments.values()))}"
    )
    print()

    # Print test set info
    print(
        f"Test set: {fold_counts['test']} samples, {len(piece_counts['test'])} pieces"
    )

    # Print CV fold info
    print("\nCross-validation folds:")
    for fold_id in range(n_folds):
        n_samples = fold_counts[fold_id]
        n_pieces = len(piece_counts[fold_id])
        print(f"  Fold {fold_id}: {n_samples} samples, {n_pieces} pieces")

    # Print train/val splits for each fold
    print("\nTrain/Val distribution per fold:")
    for fold_id in range(n_folds):
        val_samples = get_fold_samples(assignments, fold_id, "val")
        train_samples = get_fold_samples(assignments, fold_id, "train")
        print(f"  Fold {fold_id}: train={len(train_samples)}, val={len(val_samples)}")

    print()
