"""
MinHash + LSH near-duplicate detection for CPT corpus documents.

Runs after all collection is complete, before assembling the final CPT corpus.
YouTube re-uploads, quoted passages in different sources, and similar textbook
excerpts can appear multiple times -- this module finds and removes them.

Default Jaccard threshold of 0.8 catches near-duplicates while allowing
legitimately similar pedagogical content.

Usage:
    uv run python -m teacher_model.dedup --corpus-dir data/corpus --threshold 0.8
    uv run python -m teacher_model.dedup --remove  # actually delete duplicates (default: dry run)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from datasketch import MinHash, MinHashLSH

DEFAULT_CORPUS_DIR = Path(__file__).parent / "data" / "corpus"
MIN_FILE_CHARS = 100


def text_to_shingles(text: str, k: int = 5) -> set[str]:
    """Convert text to character k-shingles (lowercase, stripped)."""
    text = text.strip().lower()
    if len(text) < k:
        return set()
    return {text[i : i + k] for i in range(len(text) - k + 1)}


def build_minhash(shingles: set[str], num_perm: int = 128) -> MinHash:
    """Build a MinHash from a set of shingles."""
    mh = MinHash(num_perm=num_perm)
    for shingle in shingles:
        mh.update(shingle.encode("utf-8"))
    return mh


def find_duplicates(
    corpus_dir: Path | str,
    threshold: float = 0.8,
    num_perm: int = 128,
) -> list[tuple[str, str, float]]:
    """Find near-duplicate pairs in corpus_dir.

    Returns a list of (file1, file2, jaccard_similarity) tuples where
    jaccard_similarity >= threshold. Files shorter than MIN_FILE_CHARS are skipped.
    Uses MinHashLSH for efficient O(n) approximate lookup.
    """
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    txt_files = sorted(corpus_dir.glob("*.txt"))
    if not txt_files:
        return []

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes: dict[str, MinHash] = {}

    for path in txt_files:
        content = path.read_text(encoding="utf-8", errors="replace")
        if len(content) < MIN_FILE_CHARS:
            continue
        shingles = text_to_shingles(content)
        if not shingles:
            continue
        mh = build_minhash(shingles, num_perm=num_perm)
        minhashes[path.name] = mh
        lsh.insert(path.name, mh)

    duplicates: list[tuple[str, str, float]] = []
    seen: set[frozenset[str]] = set()

    for name, mh in minhashes.items():
        candidates = lsh.query(mh)
        for candidate in candidates:
            if candidate == name:
                continue
            pair = frozenset({name, candidate})
            if pair in seen:
                continue
            seen.add(pair)
            jaccard = mh.jaccard(minhashes[candidate])
            if jaccard >= threshold:
                file1, file2 = sorted([name, candidate])
                duplicates.append((file1, file2, jaccard))

    duplicates.sort(key=lambda t: (-t[2], t[0], t[1]))
    return duplicates


def remove_duplicates(
    corpus_dir: Path | str,
    threshold: float = 0.8,
    dry_run: bool = True,
) -> int:
    """Find and optionally remove near-duplicate files from corpus_dir.

    Keeps the first file alphabetically in each duplicate pair. When dry_run
    is True, only reports what would be removed without touching the filesystem.

    Returns the count of files removed (or that would be removed).
    """
    corpus_dir = Path(corpus_dir)
    pairs = find_duplicates(corpus_dir, threshold=threshold)

    to_remove: set[str] = set()
    for file1, file2, similarity in pairs:
        # file1 < file2 alphabetically -- keep file1, remove file2
        keeper = file1
        duplicate = file2
        if duplicate not in to_remove:
            if keeper not in to_remove:
                to_remove.add(duplicate)
                action = "would remove" if dry_run else "removing"
                print(
                    f"{action}: {duplicate} (duplicate of {keeper}, "
                    f"jaccard={similarity:.3f})"
                )

    if not dry_run:
        for name in to_remove:
            (corpus_dir / name).unlink()

    return len(to_remove)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MinHash near-duplicate detection for the CPT corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=DEFAULT_CORPUS_DIR,
        help="Directory containing .txt corpus files",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Jaccard similarity threshold for near-duplicate detection",
    )
    parser.add_argument(
        "--num-perm",
        type=int,
        default=128,
        help="Number of permutations for MinHash (higher = more accurate)",
    )
    parser.add_argument(
        "--remove",
        action="store_true",
        default=False,
        help="Actually delete duplicate files (default: dry run only)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    dry_run = not args.remove

    if dry_run:
        print(f"Dry run: scanning {args.corpus_dir} for near-duplicates...")
    else:
        print(f"Scanning {args.corpus_dir} for near-duplicates (will delete)...")

    pairs = find_duplicates(
        args.corpus_dir,
        threshold=args.threshold,
        num_perm=args.num_perm,
    )

    if not pairs:
        print("No near-duplicates found.")
        return

    print(f"\nFound {len(pairs)} near-duplicate pair(s):\n")
    for file1, file2, similarity in pairs:
        print(f"  {file1} <-> {file2}  jaccard={similarity:.3f}")

    print()
    count = remove_duplicates(
        args.corpus_dir,
        threshold=args.threshold,
        dry_run=dry_run,
    )

    if dry_run:
        print(f"\n{count} file(s) would be removed. Pass --remove to delete them.")
    else:
        print(f"\n{count} file(s) removed.")


if __name__ == "__main__":
    main(sys.argv[1:])
