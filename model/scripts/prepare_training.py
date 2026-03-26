"""One-command pipeline: T5 labeling done -> training ready.

Steps:
1. Load all T5 manifests
2. Run data integrity checks
3. Generate train/val/test splits for T5
4. Save splits to model/data/splits/
5. Print summary

Embedding extraction and HF Bucket upload are separate steps
(require GPU and network respectively).

Usage:
    cd model/
    uv run python scripts/prepare_training.py
    uv run python scripts/prepare_training.py --check-audio
    uv run python scripts/prepare_training.py --check-embeddings
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.paths import Evals, Splits, Embeddings

from src.model_improvement.data_integrity import load_all_t5_manifests, check_integrity
from src.model_improvement.splits import generate_t5_splits


def main():
    parser = argparse.ArgumentParser(description="Prepare training data from T5 manifests")
    parser.add_argument("--check-audio", action="store_true",
                        help="Also check that audio files exist")
    parser.add_argument("--check-embeddings", action="store_true",
                        help="Also check that embedding files exist")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    skill_eval_dir = Evals.skill_eval
    print(f"Loading T5 manifests from {skill_eval_dir}...")
    recordings = load_all_t5_manifests(skill_eval_dir)
    # Filter to downloaded only
    recordings = [r for r in recordings if r.get("downloaded", False)]
    print(f"  {len(recordings)} downloaded recordings across {len({r['piece'] for r in recordings})} pieces")

    # Data integrity checks
    print("\nRunning integrity checks...")
    audio_dir = skill_eval_dir if args.check_audio else None
    emb_dir = Embeddings.t5_muq if args.check_embeddings else None
    errors = check_integrity(recordings, audio_dir=audio_dir, embedding_dir=emb_dir)

    if errors:
        print(f"\nFOUND {len(errors)} INTEGRITY ERRORS:")
        for err in errors:
            print(f"  - {err}")
        raise SystemExit(1)
    print("  All checks passed")

    # Generate splits
    print(f"\nGenerating T5 splits (seed={args.seed})...")
    t5_splits = generate_t5_splits(recordings, train=0.8, val=0.1, test=0.1, seed=args.seed)
    print(f"  train={len(t5_splits['train'])}, val={len(t5_splits['val'])}, test={len(t5_splits['test'])}")

    # Save splits
    Splits.root.mkdir(parents=True, exist_ok=True)
    splits_path = Splits.root / "t5_splits.json"
    serializable = {
        split_name: [
            {"video_id": r["video_id"], "piece": r["piece"], "skill_bucket": r["skill_bucket"]}
            for r in recs
        ]
        for split_name, recs in t5_splits.items()
    }
    with open(splits_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"  Saved to {splits_path}")

    # Summary
    print("\n=== SUMMARY ===")
    print(f"T5: {len(recordings)} recordings -> train/val/test split")
    pieces = sorted({r["piece"] for r in recordings})
    print(f"Pieces: {len(pieces)}")
    for piece in pieces:
        piece_recs = [r for r in recordings if r["piece"] == piece]
        buckets = sorted({r["skill_bucket"] for r in piece_recs})
        print(f"  {piece}: {len(piece_recs)} recordings, buckets {buckets}")

    print(f"\nSplits saved to {splits_path}")
    print("Next steps:")
    print("  1. Extract MuQ embeddings: uv run python scripts/extract_t5_muq.py")
    print("  2. Extract Aria embeddings: uv run python scripts/extract_t5_aria.py")
    print("  3. Upload to HF Bucket: hf buckets sync model/data/embeddings/ hf://buckets/crescendai/training-data/embeddings/")


if __name__ == "__main__":
    main()
