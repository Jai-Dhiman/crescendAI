"""
Corpus assembly status dashboard and dedup orchestrator.

Reports how close the CPT training corpus is to the 100M/200M token targets,
and wraps the dedup pipeline with a simple CLI.

Usage:
    uv run python -m teacher_model.corpus_builder status   - print status report
    uv run python -m teacher_model.corpus_builder stats    - print JSON stats
    uv run python -m teacher_model.corpus_builder dedup    - dry-run dedup scan
    uv run python -m teacher_model.corpus_builder dedup --remove  - actually remove dups
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import tiktoken

from teacher_model.provenance import ProvenanceManifest
from teacher_model.dedup import find_duplicates, remove_duplicates

CORPUS_DIR = Path(__file__).parent / "data" / "corpus"

TARGET_100M = 100_000_000
TARGET_200M = 200_000_000

_ENCODING = tiktoken.get_encoding("cl100k_base")


def corpus_stats() -> dict:
    """Compute stats for the assembled corpus.

    Returns a dict with keys:
        documents          - number of .txt files in CORPUS_DIR
        provenance_records - total records in the provenance manifest
        total_words        - sum of word counts across all .txt files
        total_tokens       - sum of tiktoken cl100k_base token counts
        tokens_by_tier     - dict mapping source_tier -> token count (from provenance)
        target_100m        - bool, whether total_tokens >= 100M
        target_200m        - bool, whether total_tokens >= 200M
    """
    corpus_dir = CORPUS_DIR
    txt_files = sorted(corpus_dir.glob("*.txt")) if corpus_dir.is_dir() else []

    total_words = 0
    total_tokens = 0
    for path in txt_files:
        text = path.read_text(encoding="utf-8", errors="replace")
        total_words += len(text.split())
        total_tokens += len(_ENCODING.encode(text))

    manifest = ProvenanceManifest()
    provenance_records = manifest.count()
    tier_counts = manifest.by_tier()  # tier -> record count

    # Scale per-tier token counts proportionally from the total.
    # Provenance word counts give us the ratio; scale to observed token total.
    provenance_words = manifest.total_words()
    if provenance_words > 0 and total_tokens > 0:
        tokens_per_word = total_tokens / provenance_words
    else:
        tokens_per_word = 1.3  # reasonable default for English prose

    # Build tokens_by_tier from per-tier word sums in the manifest.
    records = manifest._records()
    words_by_tier: dict[str, int] = {}
    for record in records:
        words_by_tier[record.source_tier] = (
            words_by_tier.get(record.source_tier, 0) + record.word_count
        )

    tokens_by_tier: dict[str, int] = {
        tier: int(words * tokens_per_word)
        for tier, words in words_by_tier.items()
    }

    return {
        "documents": len(txt_files),
        "provenance_records": provenance_records,
        "total_words": total_words,
        "total_tokens": total_tokens,
        "tokens_by_tier": tokens_by_tier,
        "target_100m": total_tokens >= TARGET_100M,
        "target_200m": total_tokens >= TARGET_200M,
    }


def print_status() -> None:
    """Print a formatted corpus assembly status report."""
    stats = corpus_stats()

    total_tokens = stats["total_tokens"]
    pct_100m = total_tokens / TARGET_100M * 100
    pct_200m = total_tokens / TARGET_200M * 100

    separator = "=" * 60
    print(separator)
    print("CORPUS ASSEMBLY STATUS")
    print(separator)
    print(f"Documents:  {stats['documents']:,}")
    print(f"Words:      {stats['total_words']:,}")
    print(f"Tokens:     {total_tokens:,}")
    print(
        f"Progress:   {pct_100m:.1f}% of 100M target"
        f" / {pct_200m:.1f}% of 200M target"
    )

    print()
    print("By source tier:")
    tokens_by_tier = stats["tokens_by_tier"]
    if tokens_by_tier:
        for tier, tok_count in sorted(tokens_by_tier.items()):
            print(f"  {tier}: {tok_count:,} tokens")
    else:
        print("  (no provenance records)")

    print()
    corpus_dir = CORPUS_DIR
    if corpus_dir.is_dir() and stats["documents"] > 0:
        pairs = find_duplicates(corpus_dir)
        if pairs:
            print(f"WARNING: {len(pairs)} near-duplicate pair(s) detected.")
            print("Run `corpus_builder dedup` for details.")
        else:
            print("No duplicates detected.")
    else:
        print("No duplicates detected.")

    print(separator)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Corpus assembly status and dedup orchestrator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("status", help="Print formatted status report")
    subparsers.add_parser("stats", help="Print JSON stats")

    dedup_parser = subparsers.add_parser("dedup", help="Run near-duplicate detection")
    dedup_parser.add_argument(
        "--remove",
        action="store_true",
        default=False,
        help="Actually delete duplicate files (default: dry run)",
    )
    dedup_parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Jaccard similarity threshold",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "status":
        print_status()

    elif args.command == "stats":
        stats = corpus_stats()
        print(json.dumps(stats, indent=2))

    elif args.command == "dedup":
        dry_run = not args.remove
        corpus_dir = CORPUS_DIR

        if not corpus_dir.is_dir():
            print(f"Corpus directory not found: {corpus_dir}", file=sys.stderr)
            sys.exit(1)

        mode = "Dry run" if dry_run else "LIVE RUN"
        print(f"{mode}: scanning {corpus_dir} for near-duplicates...")

        pairs = find_duplicates(corpus_dir, threshold=args.threshold)

        if not pairs:
            print("No near-duplicates found.")
            return

        print(f"\nFound {len(pairs)} near-duplicate pair(s):\n")
        for file1, file2, similarity in pairs:
            print(f"  {file1} <-> {file2}  jaccard={similarity:.3f}")

        print()
        count = remove_duplicates(corpus_dir, threshold=args.threshold, dry_run=dry_run)

        if dry_run:
            print(f"\n{count} file(s) would be removed. Pass --remove to delete them.")
        else:
            print(f"\n{count} file(s) removed.")


if __name__ == "__main__":
    main(sys.argv[1:])
