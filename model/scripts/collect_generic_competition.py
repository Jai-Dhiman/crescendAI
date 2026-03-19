"""Generic competition data pipeline: download audio from recordings.jsonl, segment, embed.

Works with any competition that has a pre-built recordings.jsonl manifest
(Chopin 2015, Cliburn 2022, Queen Elisabeth 2025, etc.).

Usage (from model/ directory):
    python scripts/collect_generic_competition.py cliburn_2022
    python scripts/collect_generic_competition.py chopin_2015 --skip-embeddings
    python scripts/collect_generic_competition.py cliburn_2022 --dry-run

The manifest file is read from:
    data/manifests/competition/{competition_key}_recordings.jsonl

Downloaded audio and embeddings go to:
    data/manifests/competition/{competition_key}/audio/
    data/manifests/competition/{competition_key}/muq_embeddings/
    data/manifests/competition/{competition_key}/recordings.jsonl  (with durations filled)
    data/manifests/competition/{competition_key}/metadata.jsonl    (segment-level)
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

import jsonlines

MODEL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MODEL_ROOT / "src"))

from model_improvement.competition import segment_and_embed_competition

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _download_audio(
    url: str,
    output_path: Path,
) -> None:
    """Download audio from YouTube using yt-dlp, output as 24kHz mono WAV."""
    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "wav",
        "--postprocessor-args", "ffmpeg:-ar 24000 -ac 1",
        "--output", str(output_path.with_suffix(".%(ext)s")),
        "--no-playlist",
        "--quiet",
    ]
    cmd.append(url)

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(
            f"yt-dlp failed (code {result.returncode}): {result.stderr[:500]}"
        )


def _get_wav_duration(wav_path: Path) -> float:
    """Get duration of a WAV file in seconds."""
    try:
        import soundfile as sf
        info = sf.info(str(wav_path))
        return info.duration
    except Exception:
        return 0.0


def download_from_manifest(
    manifest_path: Path,
    cache_dir: Path,
    dry_run: bool = False,
) -> int:
    """Download audio for all entries in a recordings.jsonl manifest.

    Reads the source manifest, downloads audio via yt-dlp for entries
    with a source_url, and writes a cache-local recordings.jsonl with
    audio_path and duration_seconds filled in.

    Returns count of newly downloaded recordings.
    """
    audio_dir = cache_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    recordings_path = cache_dir / "recordings.jsonl"

    # Load source manifest
    with jsonlines.open(manifest_path) as reader:
        source_records = list(reader)

    # Load existing downloads for idempotency
    existing_ids: set[str] = set()
    if recordings_path.exists():
        with jsonlines.open(recordings_path) as reader:
            for record in reader:
                existing_ids.add(record["recording_id"])

    new_count = 0
    skipped_no_url = 0
    failed = 0

    for entry in source_records:
        recording_id = entry["recording_id"]
        source_url = entry.get("source_url")
        wav_path = audio_dir / f"{recording_id}.wav"

        # Skip if already downloaded
        if recording_id in existing_ids and wav_path.exists() and wav_path.stat().st_size > 0:
            logger.debug("Skipping %s (already exists)", recording_id)
            continue

        if not source_url:
            skipped_no_url += 1
            logger.debug("No URL for %s, skipping", recording_id)
            continue

        if dry_run:
            logger.info("[DRY RUN] Would download %s from %s", recording_id, source_url)
            new_count += 1
            continue

        logger.info("Downloading %s ...", recording_id)

        try:
            _download_audio(source_url, wav_path)
        except Exception as e:
            logger.error("Failed to download %s: %s", recording_id, e)
            failed += 1
            continue

        if not wav_path.exists() or wav_path.stat().st_size == 0:
            logger.error("Download produced empty file: %s", recording_id)
            failed += 1
            continue

        duration = _get_wav_duration(wav_path)

        # Write to cache-local recordings.jsonl (append for resumability)
        cache_record = {
            **entry,
            "audio_path": f"audio/{recording_id}.wav",
            "duration_seconds": duration,
        }
        with jsonlines.open(recordings_path, mode="a") as writer:
            writer.write(cache_record)

        existing_ids.add(recording_id)
        new_count += 1

    logger.info(
        "Download complete: %d new, %d skipped (no URL), %d failed",
        new_count, skipped_no_url, failed,
    )
    return new_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generic competition data collection pipeline"
    )
    parser.add_argument(
        "competition_key",
        help="Competition key matching manifest filename, e.g. 'cliburn_2022'",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip audio download step",
    )
    parser.add_argument(
        "--skip-embeddings", action="store_true",
        help="Skip MuQ embedding extraction step",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be downloaded without actually downloading",
    )
    args = parser.parse_args()

    manifests_dir = MODEL_ROOT / "data" / "manifests" / "competition"
    manifest_path = manifests_dir / f"{args.competition_key}_recordings.jsonl"

    if not manifest_path.exists():
        logger.error("Manifest not found: %s", manifest_path)
        logger.info(
            "Available manifests: %s",
            [p.stem for p in manifests_dir.glob("*_recordings.jsonl")],
        )
        sys.exit(1)

    cache_dir = manifests_dir / args.competition_key
    logger.info("Competition: %s", args.competition_key)
    logger.info("Manifest: %s", manifest_path)
    logger.info("Cache directory: %s", cache_dir)

    t_start = time.time()

    # Count entries
    with jsonlines.open(manifest_path) as reader:
        entries = list(reader)
    has_url = sum(1 for e in entries if e.get("source_url"))
    logger.info(
        "Manifest: %d entries (%d with URLs, %d without)",
        len(entries), has_url, len(entries) - has_url,
    )

    # Step 1: Download audio
    if args.skip_download:
        logger.info("Step 1: SKIPPED (--skip-download)")
    else:
        logger.info("=" * 60)
        logger.info("Step 1: Downloading audio...")
        n_downloaded = download_from_manifest(
            manifest_path, cache_dir, dry_run=args.dry_run,
        )
        logger.info("Downloaded %d recordings", n_downloaded)

    if args.dry_run:
        logger.info("Dry run complete.")
        return

    # Step 2: Segment and extract MuQ embeddings
    if args.skip_embeddings:
        logger.info("Step 2: SKIPPED (--skip-embeddings)")
    else:
        logger.info("=" * 60)
        logger.info("Step 2: Segmenting audio and extracting MuQ embeddings...")
        n_segments = segment_and_embed_competition(cache_dir)
        logger.info("Processed %d segments", n_segments)

    # Step 3: Summary
    logger.info("=" * 60)
    logger.info("Summary:")

    recordings_path = cache_dir / "recordings.jsonl"
    metadata_path = cache_dir / "metadata.jsonl"
    audio_dir = cache_dir / "audio"
    emb_dir = cache_dir / "muq_embeddings"

    n_recordings = 0
    if recordings_path.exists():
        with jsonlines.open(recordings_path) as reader:
            n_recordings = sum(1 for _ in reader)

    n_segments = 0
    if metadata_path.exists():
        with jsonlines.open(metadata_path) as reader:
            n_segments = sum(1 for _ in reader)

    n_audio = len(list(audio_dir.glob("*.wav"))) if audio_dir.exists() else 0
    n_emb = len(list(emb_dir.glob("*.pt"))) if emb_dir.exists() else 0

    logger.info("  Recordings downloaded: %d", n_recordings)
    logger.info("  Audio files on disk: %d", n_audio)
    logger.info("  Segments: %d", n_segments)
    logger.info("  MuQ embeddings: %d", n_emb)

    elapsed = time.time() - t_start
    logger.info("Pipeline complete in %.1f minutes", elapsed / 60)


if __name__ == "__main__":
    main()
