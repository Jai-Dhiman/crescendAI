"""Pair sampling utilities for disentanglement experiments.

Provides efficient sampling strategies for pairwise comparisons.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


def sample_pairs_same_piece(
    piece_to_keys: Dict[str, List[str]],
    n_pairs: int,
    seed: int = 42,
) -> List[Tuple[str, str, str]]:
    """Sample random pairs from same-piece recordings.

    Args:
        piece_to_keys: Dict mapping piece_id to list of recording keys.
        n_pairs: Number of pairs to sample.
        seed: Random seed.

    Returns:
        List of (piece_id, key_a, key_b) tuples.
    """
    rng = np.random.RandomState(seed)

    # Filter to pieces with 2+ recordings
    valid_pieces = [
        (pid, keys) for pid, keys in piece_to_keys.items() if len(keys) >= 2
    ]

    if not valid_pieces:
        return []

    # Compute sampling weights (proportional to number of possible pairs)
    weights = []
    for pid, keys in valid_pieces:
        n_possible = len(keys) * (len(keys) - 1) // 2
        weights.append(n_possible)
    weights = np.array(weights) / sum(weights)

    pairs = []
    for _ in range(n_pairs):
        # Sample piece
        idx = rng.choice(len(valid_pieces), p=weights)
        pid, keys = valid_pieces[idx]

        # Sample two distinct recordings
        k_a, k_b = rng.choice(keys, size=2, replace=False)
        pairs.append((pid, k_a, k_b))

    return pairs


def sample_hard_pairs(
    piece_to_keys: Dict[str, List[str]],
    labels: Dict,
    n_pairs: int,
    min_diff: float = 0.1,
    max_diff: float = 0.3,
    seed: int = 42,
) -> List[Tuple[str, str, str]]:
    """Sample hard pairs where performances have moderate score differences.

    Hard pairs have differences large enough to have clear rankings but
    small enough to be challenging.

    Args:
        piece_to_keys: Dict mapping piece_id to list of recording keys.
        labels: Dict mapping keys to label arrays.
        n_pairs: Number of pairs to sample.
        min_diff: Minimum mean absolute difference.
        max_diff: Maximum mean absolute difference.
        seed: Random seed.

    Returns:
        List of (piece_id, key_a, key_b) tuples.
    """
    rng = np.random.RandomState(seed)

    # Build list of all valid pairs with their difficulty
    candidate_pairs = []
    for pid, keys in piece_to_keys.items():
        if len(keys) < 2:
            continue

        for i, k_a in enumerate(keys):
            for k_b in keys[i + 1 :]:
                if k_a not in labels or k_b not in labels:
                    continue

                # Compute mean absolute difference
                diff = np.abs(
                    np.array(labels[k_a][:19]) - np.array(labels[k_b][:19])
                ).mean()

                if min_diff <= diff <= max_diff:
                    candidate_pairs.append((pid, k_a, k_b, diff))

    if not candidate_pairs:
        # Fall back to regular sampling
        return sample_pairs_same_piece(piece_to_keys, n_pairs, seed)

    # Sample from candidates
    indices = rng.choice(len(candidate_pairs), size=min(n_pairs, len(candidate_pairs)))
    return [(p[0], p[1], p[2]) for p in [candidate_pairs[i] for i in indices]]


def compute_pairwise_statistics(
    piece_to_keys: Dict[str, List[str]],
    labels: Dict,
) -> Dict:
    """Compute statistics about pairwise comparisons.

    Args:
        piece_to_keys: Dict mapping piece_id to list of recording keys.
        labels: Dict mapping keys to label arrays.

    Returns:
        Dict with statistics.
    """
    n_pieces = len(piece_to_keys)
    n_recordings = sum(len(keys) for keys in piece_to_keys.values())
    n_pairs = 0
    diffs = []

    for keys in piece_to_keys.values():
        n_pairs += len(keys) * (len(keys) - 1) // 2

        for i, k_a in enumerate(keys):
            for k_b in keys[i + 1 :]:
                if k_a in labels and k_b in labels:
                    diff = np.abs(
                        np.array(labels[k_a][:19]) - np.array(labels[k_b][:19])
                    )
                    diffs.append(diff.mean())

    diffs = np.array(diffs) if diffs else np.array([0.0])

    return {
        "n_pieces": n_pieces,
        "n_recordings": n_recordings,
        "n_possible_pairs": n_pairs,
        "mean_diff": float(diffs.mean()),
        "std_diff": float(diffs.std()),
        "min_diff": float(diffs.min()),
        "max_diff": float(diffs.max()),
        "median_diff": float(np.median(diffs)),
    }


def get_fold_piece_mapping(
    piece_to_keys: Dict[str, List[str]],
    fold_assignments: Dict[str, List[str]],
    fold_id: int,
    mode: str,
) -> Tuple[Dict[str, List[str]], List[str]]:
    """Get piece mapping and keys for a specific fold.

    Args:
        piece_to_keys: Full piece to keys mapping.
        fold_assignments: Dict with fold_0, fold_1, etc.
        fold_id: Which fold to use for validation.
        mode: "train" or "val".

    Returns:
        Tuple of (filtered piece_to_keys, list of valid keys).
    """
    if mode == "val":
        valid_keys = set(fold_assignments.get(f"fold_{fold_id}", []))
    else:  # train
        valid_keys = set()
        for i in range(4):
            if i != fold_id:
                valid_keys.update(fold_assignments.get(f"fold_{i}", []))

    # Filter piece mapping
    filtered = {}
    for pid, keys in piece_to_keys.items():
        filtered_keys = [k for k in keys if k in valid_keys]
        if len(filtered_keys) >= 2:
            filtered[pid] = filtered_keys

    all_keys = []
    for keys in filtered.values():
        all_keys.extend(keys)

    return filtered, all_keys
