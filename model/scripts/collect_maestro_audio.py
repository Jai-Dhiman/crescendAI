"""Segment MAESTRO v3 audio and extract MuQ embeddings for T3 contrastive training.

Run from model/ directory:
    python scripts/collect_maestro_audio.py --maestro-dir data/maestro-v3.0.0
    python scripts/collect_maestro_audio.py --maestro-dir data/maestro-v3.0.0 --skip-embeddings

Expects MAESTRO v3 audio to be downloaded already (200GB WAV).
Download from: https://magenta.tensorflow.org/datasets/maestro

Each step is idempotent. Running again skips completed work.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MODEL_ROOT / "src"))

from model_improvement.maestro import (
    build_piece_performer_mapping,
    parse_maestro_audio_metadata,
    segment_and_embed_maestro,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MAESTRO audio segmentation and MuQ embedding extraction"
    )
    parser.add_argument(
        "--maestro-dir",
        type=Path,
        default=MODEL_ROOT / "data" / "maestro-v3.0.0",
        help="Path to MAESTRO v3 root directory",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=MODEL_ROOT / "data" / "maestro_cache",
        help="Output directory for cached segments and embeddings",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Only parse metadata, skip segmentation and embedding",
    )
    parser.add_argument(
        "--segment-duration",
        type=float,
        default=30.0,
        help="Segment duration in seconds (default: 30)",
    )
    args = parser.parse_args()

    logger.info("MAESTRO audio pipeline")
    logger.info("MAESTRO dir: %s", args.maestro_dir)
    logger.info("Cache dir: %s", args.cache_dir)
    t_start = time.time()

    # Step 1: Parse metadata
    logger.info("=" * 60)
    logger.info("Step 1: Parsing MAESTRO metadata...")
    records = parse_maestro_audio_metadata(args.maestro_dir)
    logger.info("Found %d audio records", len(records))

    # Count audio files that exist on disk
    n_exists = sum(
        1 for r in records
        if (args.maestro_dir / r["audio_filename"]).exists()
    )
    logger.info("Audio files on disk: %d / %d", n_exists, len(records))

    if n_exists == 0:
        logger.error(
            "No audio files found. Download MAESTRO v3 audio from: "
            "https://magenta.tensorflow.org/datasets/maestro"
        )
        sys.exit(1)

    # Step 2: Segment and extract embeddings
    if args.skip_embeddings:
        logger.info("=" * 60)
        logger.info("Step 2: SKIPPED (--skip-embeddings)")
    else:
        logger.info("=" * 60)
        logger.info("Step 2: Segmenting audio and extracting MuQ embeddings...")
        n_segments = segment_and_embed_maestro(
            args.maestro_dir, args.cache_dir,
            segment_duration=args.segment_duration,
        )
        logger.info("Processed %d new segments", n_segments)

    # Step 3: Build contrastive mapping
    logger.info("=" * 60)
    logger.info("Step 3: Building piece-performer contrastive mapping...")
    mapping = build_piece_performer_mapping(args.cache_dir)
    logger.info(
        "Contrastive pairs: %d pieces with 2+ recordings",
        len(mapping),
    )

    # Save mapping for training
    mapping_path = args.cache_dir / "contrastive_mapping.json"
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)
    logger.info("Saved mapping to %s", mapping_path)

    # Summary
    logger.info("=" * 60)
    metadata_path = args.cache_dir / "metadata.jsonl"
    emb_dir = args.cache_dir / "muq_embeddings"
    n_segments_total = 0
    if metadata_path.exists():
        import jsonlines
        with jsonlines.open(metadata_path) as reader:
            n_segments_total = sum(1 for _ in reader)
    n_emb = len(list(emb_dir.glob("*.pt"))) if emb_dir.exists() else 0

    logger.info("Total segments: %d", n_segments_total)
    logger.info("Total embeddings: %d", n_emb)
    logger.info("Contrastive pieces: %d", len(mapping))
    logger.info(
        "Contrastive segments: %d",
        sum(len(v) for v in mapping.values()),
    )

    elapsed = time.time() - t_start
    logger.info("Pipeline complete in %.1f minutes", elapsed / 60)


if __name__ == "__main__":
    main()
