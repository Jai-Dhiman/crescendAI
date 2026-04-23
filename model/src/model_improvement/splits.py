"""Three-way train/val/test split generation with stratification.

Replaces 4-fold CV. T5 stratifies by piece+bucket, T1 by piece,
T2 by competition+round.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Dict, List


def generate_t5_splits(
    recordings: list[dict[str, Any]],
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Split T5 recordings stratified by piece + skill_bucket.

    Each piece+bucket group is split proportionally so every group
    appears in every split.

    Args:
        recordings: List of dicts with keys: video_id, piece, skill_bucket.
        train: Fraction for training.
        val: Fraction for validation.
        test: Fraction for test.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys "train", "val", "test", each a list of recording dicts.

    Raises:
        ValueError: If any piece+bucket group has fewer than 3 recordings.
    """
    rng = random.Random(seed)

    # Group by (piece, bucket)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for rec in recordings:
        key = (rec["piece"], rec["skill_bucket"])
        groups[key].append(rec)

    # Validate minimum size
    for (piece, bucket), group_recs in groups.items():
        if len(group_recs) < 3:
            raise ValueError(
                f"piece={piece}, bucket={bucket} has {len(group_recs)} recordings, "
                f"fewer than 3 required for 3-way split"
            )

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

    for _key, group_recs in sorted(groups.items()):
        shuffled = list(group_recs)
        rng.shuffle(shuffled)
        n = len(shuffled)

        # Ensure at least 1 in each split
        n_test = max(1, round(n * test))
        n_val = max(1, round(n * val))
        n_train = n - n_val - n_test

        if n_train < 1:
            n_train = 1
            n_val = max(1, (n - 1) // 2)
            n_test = n - n_train - n_val

        splits["train"].extend(shuffled[:n_train])
        splits["val"].extend(shuffled[n_train:n_train + n_val])
        splits["test"].extend(shuffled[n_train + n_val:])

    return splits


def generate_t1_splits(
    records: list[dict[str, Any]],
    train: float = 0.8,
    test: float = 0.2,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Split T1 (PercePiano) records stratified by piece. No val split."""
    rng = random.Random(seed)

    groups: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        groups[rec["piece"]].append(rec)

    splits: dict[str, list[dict]] = {"train": [], "test": []}

    for _piece, group_recs in sorted(groups.items()):
        shuffled = list(group_recs)
        rng.shuffle(shuffled)
        n_test = max(1, round(len(shuffled) * test))
        splits["test"].extend(shuffled[:n_test])
        splits["train"].extend(shuffled[n_test:])

    return splits


def generate_t2_splits(
    records: list[dict[str, Any]],
    train: float = 0.85,
    test: float = 0.15,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Split T2 (competition) by holding out entire rounds.

    Holdout unit is (competition, round) to prevent same-performer
    leakage across splits.
    """
    rng = random.Random(seed)

    # Collect unique (competition, round) keys
    round_keys = sorted({(r["competition"], r["round"]) for r in records})
    rng.shuffle(round_keys)

    # Map rounds to their recordings
    round_to_recs: dict[tuple, list[dict]] = defaultdict(list)
    for rec in records:
        round_to_recs[(rec["competition"], rec["round"])].append(rec)

    # Greedily assign rounds to test until we hit the target fraction
    total = len(records)
    target_test = round(total * test)
    test_count = 0
    test_rounds: set[tuple] = set()

    for rk in round_keys:
        if test_count >= target_test:
            break
        test_rounds.add(rk)
        test_count += len(round_to_recs[rk])

    splits: dict[str, list[dict]] = {"train": [], "test": []}
    for rk, recs in round_to_recs.items():
        if rk in test_rounds:
            splits["test"].extend(recs)
        else:
            splits["train"].extend(recs)

    return splits


def create_piece_stratified_folds(
    multi_performer_pieces: Dict[str, List[str]],
    n_folds: int = 4,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Create stratified folds ensuring no piece leakage.

    All recordings of a piece go into the same fold. Pieces are greedily
    assigned to the smallest current fold (by total recording count) to
    balance fold sizes.

    Args:
        multi_performer_pieces: Dict mapping piece_id to recording keys.
        n_folds: Number of folds.
        seed: Random seed for reproducibility.

    Returns:
        Dict with fold_0, fold_1, etc. containing recording keys.
    """
    import numpy as np

    np.random.seed(seed)

    piece_ids = list(multi_performer_pieces.keys())
    np.random.shuffle(piece_ids)

    fold_sizes = [0] * n_folds
    fold_pieces: List[List[str]] = [[] for _ in range(n_folds)]

    sorted_pieces = sorted(
        piece_ids, key=lambda p: len(multi_performer_pieces[p]), reverse=True
    )

    for pid in sorted_pieces:
        smallest_fold = min(range(n_folds), key=lambda i: fold_sizes[i])
        fold_pieces[smallest_fold].append(pid)
        fold_sizes[smallest_fold] += len(multi_performer_pieces[pid])

    fold_assignments = {}
    for i, pieces in enumerate(fold_pieces):
        keys = []
        for pid in pieces:
            keys.extend(multi_performer_pieces[pid])
        fold_assignments[f"fold_{i}"] = keys

    return fold_assignments
