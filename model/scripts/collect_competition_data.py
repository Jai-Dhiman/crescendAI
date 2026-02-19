"""Collect Chopin 2021 competition data: scrape, download, embed.

Run from model/ directory:
    python scripts/collect_competition_data.py
    python scripts/collect_competition_data.py --skip-download
    python scripts/collect_competition_data.py --skip-embeddings
    python scripts/collect_competition_data.py --edition 2021

Each step is idempotent. Running again skips completed work.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MODEL_ROOT / "src"))

from model_improvement.competition import (
    download_competition_audio,
    discover_youtube_urls,
    extract_competition_embeddings,
    load_competition_metadata,
    scrape_chopin_results,
    segment_and_embed_competition,
)
from model_improvement.data import CompetitionPairSampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect Chopin competition data pipeline"
    )
    parser.add_argument(
        "--edition", type=int, default=2021,
        help="Competition edition year (default: 2021)",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip audio download step",
    )
    parser.add_argument(
        "--skip-embeddings", action="store_true",
        help="Skip MuQ embedding extraction step",
    )
    args = parser.parse_args()

    cache_dir = MODEL_ROOT / "data" / "competition_cache" / f"chopin{args.edition}"
    metadata_path = cache_dir / "metadata.jsonl"

    logger.info("Competition data pipeline: Chopin %d", args.edition)
    logger.info("Cache directory: %s", cache_dir)
    t_start = time.time()

    # Step 1: Scrape results
    logger.info("=" * 60)
    logger.info("Step 1: Scraping competition results...")
    results = scrape_chopin_results(cache_dir)
    logger.info("Found %d performers", len(results))

    # Step 2: Discover YouTube URLs
    logger.info("=" * 60)
    logger.info("Step 2: Discovering YouTube URLs...")
    url_mapping = discover_youtube_urls(cache_dir, results)
    total_videos = sum(
        len(vids)
        for rounds in url_mapping.values()
        for vids in rounds.values()
    )
    logger.info(
        "Found %d videos for %d performers",
        total_videos, len(url_mapping),
    )

    # Step 3: Download audio
    if args.skip_download:
        logger.info("=" * 60)
        logger.info("Step 3: SKIPPED (--skip-download)")
    else:
        logger.info("=" * 60)
        logger.info("Step 3: Downloading audio...")
        records = download_competition_audio(
            url_mapping, results, cache_dir, metadata_path,
        )
        logger.info("Downloaded %d new recordings", len(records))

    # Step 3b: Rename metadata.jsonl to recordings.jsonl if needed
    # (backward compat: old pipeline wrote metadata.jsonl at recording level)
    recordings_path = cache_dir / "recordings.jsonl"
    old_metadata = cache_dir / "metadata.jsonl"
    if old_metadata.exists() and not recordings_path.exists():
        old_metadata.rename(recordings_path)

    # Step 4: Segment and extract per-segment embeddings
    if args.skip_embeddings:
        logger.info("=" * 60)
        logger.info("Step 4: SKIPPED (--skip-embeddings)")
    else:
        logger.info("=" * 60)
        logger.info("Step 4: Segmenting audio and extracting MuQ embeddings...")
        n_segments = segment_and_embed_competition(cache_dir)
        logger.info("Processed %d new segments", n_segments)

    # Step 5: Summary statistics
    logger.info("=" * 60)
    logger.info("Step 5: Summary statistics")

    all_records = load_competition_metadata(cache_dir)
    audio_dir = cache_dir / "audio"
    emb_dir = cache_dir / "muq_embeddings"

    n_audio = len(list(audio_dir.glob("*.wav"))) if audio_dir.exists() else 0
    n_emb = len(list(emb_dir.glob("*.pt"))) if emb_dir.exists() else 0

    logger.info("  Metadata records: %d", len(all_records))
    logger.info("  Audio files: %d", n_audio)
    logger.info("  MuQ embeddings: %d", n_emb)

    # Pair generation summary
    if all_records:
        sampler = CompetitionPairSampler(all_records)
        logger.info("  Within-piece pairs: %d", sampler.n_within_piece_pairs)
        logger.info("  Cross-round pairs: %d", sampler.n_cross_round_pairs)

    # Per-round breakdown
    round_counts: dict[str, int] = {}
    for r in all_records:
        rnd = r.get("round", "unknown")
        round_counts[rnd] = round_counts.get(rnd, 0) + 1
    for rnd in ["preliminary", "stage1", "stage2", "stage3", "final"]:
        if rnd in round_counts:
            logger.info("  %s: %d recordings", rnd, round_counts[rnd])

    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info("Pipeline complete in %.1f minutes", elapsed / 60)


if __name__ == "__main__":
    main()
