"""CLI orchestrator for the exercise corpus embedding validation pipeline.

Usage (full pipeline -- requires acquired MusicXML files and Aria weights):
    python -m exercise_corpus.run \\
        --sources model/src/exercise_corpus/sources.toml \\
        --output-dir model/data

Usage (validate only -- reads existing catalog):
    python -m exercise_corpus.run --validate-only --db model/data/exercise_primitives.db \\
        --output-dir model/data/results
"""

import argparse
import logging
import tomllib
from pathlib import Path

from exercise_corpus.catalog import write_primitives
from exercise_corpus.embed import embed_primitives
from exercise_corpus.segment import segment_source
from exercise_corpus.validate import ValidationResult, run_validation

logger = logging.getLogger(__name__)


def run_pipeline(
    sources_path: Path | None = None,
    output_dir: Path | None = None,
    dry_run: bool = False,
    validate_only: bool = False,
    db_path: Path | None = None,
) -> ValidationResult | None:
    """Run the full exercise corpus pipeline or a subset.

    Args:
        sources_path: path to sources.toml. Required unless validate_only=True.
        output_dir: root output directory. Scores go to output_dir/scores/exercise_primitives/,
            MIDI to output_dir/midi/exercise_primitives/, catalog to
            output_dir/exercise_primitives.db, results to output_dir/results/.
        dry_run: if True, check that all source MusicXML files exist and return None
            (no segmentation or embedding). Raises FileNotFoundError for any missing file.
        validate_only: if True, skip segment/embed and run only validate against
            an existing db_path.
        db_path: required when validate_only=True.

    Returns:
        ValidationResult from validate.run_validation, or None when dry_run=True.

    Raises:
        FileNotFoundError: if any source MusicXML is missing (dry_run or normal run).
        ValueError: if required arguments are missing.
    """
    if validate_only:
        if db_path is None:
            raise ValueError("db_path is required when validate_only=True")
        if output_dir is None:
            raise ValueError("output_dir is required when validate_only=True")
        results_dir = Path(output_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        return run_validation(Path(db_path), results_dir)

    if sources_path is None:
        raise ValueError("sources_path is required when validate_only=False")
    if output_dir is None:
        raise ValueError("output_dir is required when validate_only=False")

    output_dir = Path(output_dir)
    score_dir = output_dir / "scores" / "exercise_primitives"
    midi_dir = output_dir / "midi" / "exercise_primitives"
    catalog_path = output_dir / "exercise_primitives.db"
    results_dir = output_dir / "results"

    with open(Path(sources_path), "rb") as f:
        manifest = tomllib.load(f)
    sources = manifest["sources"]

    # Validate all MusicXML paths exist before doing any work
    for source in sources:
        xml_path = Path(source["musicxml_path"])
        if not xml_path.exists():
            raise FileNotFoundError(
                f"Source MusicXML not found for {source['name']!r}: {xml_path}\n"
                "Acquire from IMSLP and place at the path listed in sources.toml."
            )

    if dry_run:
        logger.info("dry_run=True: all %d source files present. OK.", len(sources))
        print(f"dry_run OK: all {len(sources)} source MusicXML files found.")
        return None

    all_primitives = []
    for source in sources:
        xml_path = Path(source["musicxml_path"])
        logger.info("Segmenting source %r from %s", source["name"], xml_path)
        primitives = segment_source(xml_path, source["name"], score_dir, midi_dir)
        all_primitives.extend(primitives)
        logger.info("  -> %d primitives", len(primitives))

    logger.info("Total primitives: %d. Extracting embeddings...", len(all_primitives))
    embeddings = embed_primitives(midi_dir)

    logger.info("Writing catalog to %s", catalog_path)
    write_primitives(all_primitives, embeddings, catalog_path)

    logger.info("Running validation...")
    result = run_validation(catalog_path, results_dir)
    print(
        f"\nValidation complete: purity={result.purity:.4f} -- {result.verdict} "
        f"(threshold 0.70)\n"
        f"UMAP: {result.umap_path}\n"
        f"Review artifact (15 pairs): {result.pairs_path}\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exercise corpus embedding validation pipeline."
    )
    parser.add_argument(
        "--sources",
        type=Path,
        default=None,
        help="Path to sources.toml manifest.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Root output directory (default: data/).",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Skip segment/embed; run validate against an existing catalog.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Path to existing SQLite catalog (required with --validate-only).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check that all source MusicXML files exist and exit cleanly (no segmentation or embedding).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    run_pipeline(
        sources_path=args.sources,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        validate_only=args.validate_only,
        db_path=args.db,
    )


if __name__ == "__main__":
    main()
