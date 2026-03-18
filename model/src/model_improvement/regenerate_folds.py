"""Regenerate piece-stratified CV folds with no piece leaking.

Uses create_piece_stratified_folds() to assign pieces to folds (greedy
bin-packing, no piece appears in multiple folds). Then converts to the
train/val format expected by the training code.

Backs up the old folds.json before overwriting.

Usage:
    cd model/
    uv run python -m model_improvement.regenerate_folds
    uv run python -m model_improvement.regenerate_folds --verify-only  # just check for leaking
"""

from __future__ import annotations

import argparse
import json
import shutil
from src.paths import Labels
from disentanglement.data.pairwise_dataset import create_piece_stratified_folds


FOLDS_PATH = Labels.percepiano / "folds.json"
PIECE_MAPPING_PATH = Labels.percepiano / "piece_mapping.json"


def extract_piece_id(segment_key: str) -> str:
    """Extract piece ID from a segment key.

    Segment keys look like: Schubert_D960_mv2_8bars_2_07
    Piece IDs look like: Schubert_D960_mv2_8bars
    The last two parts are performer_id and segment_number.
    """
    parts = segment_key.rsplit("_", 2)
    return parts[0] if len(parts) >= 3 else segment_key


def verify_folds(folds: list[dict], piece_mapping: dict) -> bool:
    """Check if folds have piece-level leaking."""
    # Build reverse mapping: segment_key -> piece_id
    key_to_piece = {}
    for piece_id, keys in piece_mapping.items():
        for key in keys:
            key_to_piece[key] = piece_id

    has_leak = False
    for fold_idx, fold in enumerate(folds):
        train_keys = set(fold["train"])
        val_keys = set(fold["val"])

        train_pieces = {key_to_piece.get(k, extract_piece_id(k)) for k in train_keys}
        val_pieces = {key_to_piece.get(k, extract_piece_id(k)) for k in val_keys}

        overlap = train_pieces & val_pieces
        if overlap:
            has_leak = True
            print(f"  Fold {fold_idx}: LEAK -- {len(overlap)} pieces in both train and val")
            for p in sorted(overlap)[:5]:
                train_count = sum(1 for k in train_keys if key_to_piece.get(k) == p)
                val_count = sum(1 for k in val_keys if key_to_piece.get(k) == p)
                print(f"    {p}: {train_count} in train, {val_count} in val")
            if len(overlap) > 5:
                print(f"    ... and {len(overlap) - 5} more")
        else:
            print(f"  Fold {fold_idx}: CLEAN -- {len(val_pieces)} val pieces, "
                  f"{len(train_pieces)} train pieces, no overlap")

        print(f"    train={len(train_keys)} segments, val={len(val_keys)} segments")

    return not has_leak


def regenerate():
    """Generate clean piece-stratified folds and save."""
    # Load piece mapping
    with open(PIECE_MAPPING_PATH) as f:
        piece_mapping = json.load(f)

    print(f"Loaded {len(piece_mapping)} pieces from {PIECE_MAPPING_PATH}")
    total_segments = sum(len(v) for v in piece_mapping.values())
    print(f"Total segments: {total_segments}")

    # Generate fold assignments (piece-stratified, no leaking)
    fold_assignments = create_piece_stratified_folds(
        piece_mapping, n_folds=4, seed=42
    )

    print(f"\nFold assignments:")
    for fold_name, keys in fold_assignments.items():
        pieces_in_fold = set()
        for key in keys:
            for piece_id, piece_keys in piece_mapping.items():
                if key in piece_keys:
                    pieces_in_fold.add(piece_id)
                    break
        print(f"  {fold_name}: {len(keys)} segments, {len(pieces_in_fold)} pieces")

    # Convert to train/val format expected by training code
    # Each fold takes turns being the validation set
    fold_names = sorted(fold_assignments.keys())
    folds_for_training = []

    for val_fold_name in fold_names:
        val_keys = fold_assignments[val_fold_name]
        train_keys = []
        for other_fold_name in fold_names:
            if other_fold_name != val_fold_name:
                train_keys.extend(fold_assignments[other_fold_name])

        folds_for_training.append({
            "train": train_keys,
            "val": val_keys,
        })

    print(f"\nGenerated {len(folds_for_training)} train/val folds:")
    for i, fold in enumerate(folds_for_training):
        print(f"  Fold {i}: train={len(fold['train'])}, val={len(fold['val'])}")

    # Verify no leaking
    print("\nVerifying clean folds:")
    is_clean = verify_folds(folds_for_training, piece_mapping)

    if not is_clean:
        raise RuntimeError("Generated folds still have leaking -- this should not happen")

    # Backup old folds
    if FOLDS_PATH.exists():
        backup_path = FOLDS_PATH.with_suffix(".json.leaked_backup")
        shutil.copy2(FOLDS_PATH, backup_path)
        print(f"\nBacked up old folds to {backup_path}")

    # Save new folds
    with open(FOLDS_PATH, "w") as f:
        json.dump(folds_for_training, f, indent=2)

    print(f"Saved clean folds to {FOLDS_PATH}")


def verify_only():
    """Just check existing folds for leaking."""
    with open(FOLDS_PATH) as f:
        folds = json.load(f)
    with open(PIECE_MAPPING_PATH) as f:
        piece_mapping = json.load(f)

    print(f"Checking {len(folds)} folds for piece-level leaking:\n")
    is_clean = verify_folds(folds, piece_mapping)

    if is_clean:
        print("\nAll folds are clean.")
    else:
        print("\nLEAKING DETECTED. Run without --verify-only to regenerate.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate piece-stratified CV folds")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify existing folds, don't regenerate")
    args = parser.parse_args()

    if args.verify_only:
        verify_only()
    else:
        regenerate()
