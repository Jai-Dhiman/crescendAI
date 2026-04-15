"""CLI entry point for the score library pipeline.

Usage:
    uv run python -m score_library.cli parse --asap-dir data/raw/asap
    uv run python -m score_library.cli stats --source data/scores
    uv run python -m score_library.cli upload --source data/scores
    uv run python -m score_library.cli build --asap-dir data/raw/asap
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

from src.paths import Scores

from score_library.discover import discover_pieces
from score_library.parse import parse_score_midi
from score_library.schema import PieceCatalogEntry, ScoreData

logger = logging.getLogger(__name__)


def cmd_parse(args):
    asap_dir = Path(args.asap_dir)
    output_dir = Path(args.output) if args.output else Scores.root
    titles_path = output_dir / "titles.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    pieces = discover_pieces(asap_dir, titles_path if titles_path.exists() else None)
    print(f"Discovered {len(pieces)} pieces")

    successes = 0
    failures = []
    for entry in pieces:
        try:
            score_data = parse_score_midi(entry.score_midi_path, entry.piece_id, entry.composer, entry.title)
            out_path = output_dir / f"{entry.piece_id}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(score_data.model_dump(), f, indent=2)
            successes += 1
        except Exception as e:
            failures.append((entry.piece_id, str(e)))
            logger.error("Failed to parse %s: %s", entry.piece_id, e)

    print(f"\nParsed: {successes}/{len(pieces)}")
    if failures:
        print(f"Failures ({len(failures)}):")
        for pid, err in failures:
            print(f"  {pid}: {err}")


def cmd_stats(args):
    source_dir = Path(args.source)
    json_files = sorted(f for f in source_dir.glob("*.json") if f.name not in ("titles.json", "seed.sql"))
    if not json_files:
        print(f"No score JSON files found in {source_dir}")
        return

    composers = Counter()
    bar_counts = []
    note_counts = []
    time_sig_changes = 0
    tempo_changes = 0

    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        composers[data["composer"]] += 1
        bar_counts.append(data["total_bars"])
        total_notes = sum(b["note_count"] for b in data["bars"])
        note_counts.append(total_notes)
        if len(data.get("time_signatures", [])) > 1:
            time_sig_changes += 1
        if len(data.get("tempo_markings", [])) > 1:
            tempo_changes += 1

    bar_counts.sort()
    note_counts.sort()
    n = len(json_files)

    print(f"Pieces: {n}")
    print(f"\nComposer distribution:")
    for composer, count in sorted(composers.items()):
        print(f"  {composer}: {count}")
    print(f"\nBar count: min={bar_counts[0]}, median={bar_counts[n//2]}, max={bar_counts[-1]}")
    print(f"Note count: min={note_counts[0]}, median={note_counts[n//2]}, max={note_counts[-1]}")
    print(f"Pieces with time sig changes: {time_sig_changes}")
    print(f"Pieces with tempo changes: {tempo_changes}")


def cmd_upload(args):
    from score_library.upload import upload_to_r2, generate_d1_seed
    source_dir = Path(args.source)
    seed_path = Path(args.seed_output) if args.seed_output else Scores.root / "seed.sql"
    generate_d1_seed(source_dir, output_path=seed_path)
    if args.skip_r2:
        print("Skipping R2 upload (--skip-r2 flag)")
    else:
        upload_to_r2(source_dir, version=args.version)


def cmd_build(args):
    cmd_parse(args)
    # Set defaults for upload args
    args.source = args.output or str(Scores.root)
    args.version = getattr(args, "version", "v1")
    args.seed_output = getattr(args, "seed_output", None)
    args.skip_r2 = getattr(args, "skip_r2", False)
    cmd_upload(args)


def cmd_upload_mxl(args):
    from score_library.upload import upload_mxl_to_r2

    asap_dir = Path(args.asap_dir)
    upload_mxl_to_r2(asap_dir, version=args.version, dry_run=args.dry_run)


def cmd_fingerprint(args):
    from score_library.fingerprint import build_ngram_index, build_rerank_features

    scores_dir = Path(args.scores_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = [f for f in scores_dir.glob("*.json") if f.name not in ("titles.json", "seed.sql")]
    print(f"Building fingerprints for {len(json_files)} scores in {scores_dir}")

    print(f"Building N-gram index (max_freq={args.max_freq})...")
    ngram_index = build_ngram_index(scores_dir, max_freq=args.max_freq)
    ngram_path = output_dir / "ngram_index.json"
    with open(ngram_path, "w") as f:
        json.dump(ngram_index, f)
    ngram_size_kb = ngram_path.stat().st_size / 1024
    print(f"  N-gram index: {len(ngram_index)} trigrams -> {ngram_path} ({ngram_size_kb:.1f} KB)")
    if ngram_size_kb > 5120:
        logger.warning("N-gram index exceeds 5MB (%.1f KB) -- lower --max-freq to reduce size", ngram_size_kb)

    print("Building rerank features...")
    rerank_features = build_rerank_features(scores_dir)
    rerank_path = output_dir / "rerank_features.json"
    with open(rerank_path, "w") as f:
        json.dump(rerank_features, f)
    rerank_size_kb = rerank_path.stat().st_size / 1024
    print(f"  Rerank features: {len(rerank_features)} pieces -> {rerank_path} ({rerank_size_kb:.1f} KB)")

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(description="Score MIDI Library pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_parse = sub.add_parser("parse", help="Parse ASAP score MIDIs to local JSON")
    p_parse.add_argument("--asap-dir", required=True)
    p_parse.add_argument("--output", help=f"Output directory (default: {Scores.root})")

    p_stats = sub.add_parser("stats", help="Print score library statistics")
    p_stats.add_argument("--source", required=True)

    p_upload = sub.add_parser("upload", help="Upload to R2 and seed D1")
    p_upload.add_argument("--source", required=True)
    p_upload.add_argument("--version", default="v1")
    p_upload.add_argument("--seed-output", help="Output path for D1 seed SQL")
    p_upload.add_argument("--skip-r2", action="store_true", help="Skip R2 upload")

    p_build = sub.add_parser("build", help="Full pipeline: parse + upload")
    p_build.add_argument("--asap-dir", required=True)
    p_build.add_argument("--output", help=f"Output directory (default: {Scores.root})")
    p_build.add_argument("--version", default="v1")
    p_build.add_argument("--seed-output")
    p_build.add_argument("--skip-r2", action="store_true")

    p_upload_mxl = sub.add_parser(
        "upload-mxl",
        help="Upload score.musicxml files from the ASAP dataset to R2 for OSMD rendering",
    )
    p_upload_mxl.add_argument("--asap-dir", required=True, help="Root of the cloned ASAP dataset (e.g. data/raw/asap)")
    p_upload_mxl.add_argument("--version", default="v1", help="R2 key version prefix (default: v1)")
    p_upload_mxl.add_argument("--dry-run", action="store_true", help="Print what would be uploaded without uploading")

    p_fingerprint = sub.add_parser("fingerprint", help="Build N-gram index and rerank features")
    p_fingerprint.add_argument("--scores-dir", required=True, help="Directory containing score JSON files")
    p_fingerprint.add_argument("--output-dir", required=True, help="Output directory for fingerprint artifacts")
    p_fingerprint.add_argument(
        "--max-freq",
        type=int,
        default=3,
        help="Prune trigrams appearing in more than N locations (default: 3, keeps index under 5MB)",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    {
        "parse": cmd_parse,
        "stats": cmd_stats,
        "upload": cmd_upload,
        "build": cmd_build,
        "upload-mxl": cmd_upload_mxl,
        "fingerprint": cmd_fingerprint,
    }[args.command](args)


if __name__ == "__main__":
    main()
