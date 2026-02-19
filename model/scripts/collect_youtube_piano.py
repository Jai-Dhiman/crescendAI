"""Collect unlabeled piano audio from YouTube for T4 augmentation invariance training.

Run from model/ directory:
    python scripts/collect_youtube_piano.py
    python scripts/collect_youtube_piano.py --max-videos-per-channel 50
    python scripts/collect_youtube_piano.py --skip-download --skip-augmentation

Each step is idempotent. Running again skips completed work.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MODEL_ROOT / "src"))

from model_improvement.youtube_piano import (
    discover_channel_videos,
    download_piano_audio,
    load_channel_list,
    segment_and_embed_piano,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="YouTube piano audio collection for T4 invariance training"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=MODEL_ROOT / "data" / "youtube_piano_cache",
        help="Output directory",
    )
    parser.add_argument(
        "--channels-file",
        type=Path,
        default=MODEL_ROOT / "data" / "youtube_piano_cache" / "channels.yaml",
        help="Path to channels YAML file",
    )
    parser.add_argument(
        "--max-videos-per-channel",
        type=int,
        default=100,
        help="Max videos to discover per channel",
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--skip-augmentation", action="store_true")
    args = parser.parse_args()

    logger.info("YouTube piano collection pipeline")
    logger.info("Cache dir: %s", args.cache_dir)
    t_start = time.time()

    # Step 1: Load channel list
    logger.info("=" * 60)
    logger.info("Step 1: Loading channel list...")
    channels = load_channel_list(args.channels_file)
    logger.info("Found %d channels", len(channels))

    # Step 2: Discover videos
    logger.info("=" * 60)
    logger.info("Step 2: Discovering videos...")
    all_videos = []
    for ch in channels:
        logger.info("  Discovering: %s", ch["name"])
        videos = discover_channel_videos(
            ch["url"], max_videos=args.max_videos_per_channel,
        )
        for v in videos:
            v["channel"] = ch["name"]
        all_videos.extend(videos)
        logger.info("  Found %d videos", len(videos))
    logger.info("Total videos discovered: %d", len(all_videos))

    # Step 3: Download audio
    if args.skip_download:
        logger.info("=" * 60)
        logger.info("Step 3: SKIPPED (--skip-download)")
    else:
        logger.info("=" * 60)
        logger.info("Step 3: Downloading audio...")
        records = download_piano_audio(all_videos, args.cache_dir)
        logger.info("Downloaded %d new recordings", len(records))

    # Step 4: Segment and extract clean embeddings
    if args.skip_embeddings:
        logger.info("=" * 60)
        logger.info("Step 4: SKIPPED (--skip-embeddings)")
    else:
        logger.info("=" * 60)
        logger.info("Step 4: Segmenting and extracting clean MuQ embeddings...")
        n = segment_and_embed_piano(args.cache_dir)
        logger.info("Processed %d new segments", n)

    # Step 5: Augment and extract augmented embeddings
    if args.skip_augmentation:
        logger.info("=" * 60)
        logger.info("Step 5: SKIPPED (--skip-augmentation)")
    else:
        logger.info("=" * 60)
        logger.info("Step 5: Generating augmented embeddings...")
        from model_improvement.augmentation import augment_and_embed_piano
        n_aug = augment_and_embed_piano(args.cache_dir)
        logger.info("Generated %d augmented embeddings", n_aug)

    # Summary
    logger.info("=" * 60)
    import jsonlines
    metadata_path = args.cache_dir / "metadata.jsonl"
    emb_dir = args.cache_dir / "muq_embeddings"
    aug_dir = args.cache_dir / "muq_embeddings_augmented"

    n_segments = 0
    if metadata_path.exists():
        with jsonlines.open(metadata_path) as reader:
            n_segments = sum(1 for _ in reader)
    n_emb = len(list(emb_dir.glob("*.pt"))) if emb_dir.exists() else 0
    n_aug_emb = len(list(aug_dir.glob("*.pt"))) if aug_dir.exists() else 0

    logger.info("Total segments: %d", n_segments)
    logger.info("Clean embeddings: %d", n_emb)
    logger.info("Augmented embeddings: %d", n_aug_emb)

    elapsed = time.time() - t_start
    logger.info("Pipeline complete in %.1f minutes", elapsed / 60)


if __name__ == "__main__":
    main()
