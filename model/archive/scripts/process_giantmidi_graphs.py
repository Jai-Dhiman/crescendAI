"""Convert GIANTMIDI-Piano MIDI files into PyG graph shards for S2 pretraining.

Memory-safe processing for M4 (32 GB RAM):
- Single MIDI parse per file (no double-parse)
- Homogeneous graphs only (no hetero -- S2H lost)
- 50-graph shards with RSS monitoring
- Resumable via manifest file

Usage:
    cd model
    uv run python scripts/process_giantmidi_graphs.py
"""

from __future__ import annotations

import gc
import json
import logging
import time
from pathlib import Path

import psutil
import pretty_midi
import torch

from model_improvement.graph import count_midi_notes, parsed_midi_to_graph
from src.paths import Raw, Pretraining

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Configuration ---
MIDI_DIR = Raw.giantmidi / "GiantMIDI-PIano" / "midis"
SHARD_DIR = Pretraining.graphs / "shards"
MANIFEST_PATH = Raw.giantmidi / "giantmidi_manifest.json"

MAX_NOTES = 5_000
SHARD_SIZE = 50
RSS_FLUSH_THRESHOLD_GB = 20.0
FIRST_SHARD_ID = 75  # existing shards are 0000-0074


def _extract_youtube_id(filename: str) -> str:
    """Extract YouTube ID from GIANTMIDI filename.

    Filenames follow: "Composer, Title, YouTubeID.mid"
    The YouTube ID is the last comma-separated field before .mid.

    Args:
        filename: MIDI filename (with .mid extension).

    Returns:
        YouTube ID string.

    Raises:
        ValueError: If filename doesn't contain a comma separator.
    """
    stem = filename.removesuffix(".mid")
    parts = stem.rsplit(", ", 1)
    if len(parts) < 2:
        # Fallback: comma without trailing space
        parts = stem.rsplit(",", 1)
    if len(parts) < 2:
        raise ValueError(f"Cannot extract YouTube ID from filename: {filename}")
    return parts[1].strip()


def _get_rss_gb() -> float:
    """Get current process RSS in GB."""
    return psutil.Process().memory_info().rss / (1024 ** 3)


def _load_manifest() -> dict:
    """Load existing manifest for resumption, or return empty template."""
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return {
        "processed_keys": [],
        "skipped": [],
        "failed": [],
        "shard_range": [FIRST_SHARD_ID, FIRST_SHARD_ID],
        "note_count_buckets": {
            "0-500": 0, "500-1000": 0, "1000-2000": 0,
            "2000-3000": 0, "3000-5000": 0, "5000+_skipped": 0,
        },
        "total_processing_time_seconds": 0.0,
    }


def _save_manifest(manifest: dict) -> None:
    """Save manifest to disk."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


def _bucket_note_count(n: int) -> str:
    """Return histogram bucket key for a note count."""
    if n < 500:
        return "0-500"
    elif n < 1000:
        return "500-1000"
    elif n < 2000:
        return "1000-2000"
    elif n < 3000:
        return "2000-3000"
    elif n <= MAX_NOTES:
        return "3000-5000"
    else:
        return "5000+_skipped"


def _save_shard(shard_id: int, graphs: dict[str, object]) -> None:
    """Save a graph shard to disk."""
    path = SHARD_DIR / f"graphs_{shard_id:04d}.pt"
    torch.save(graphs, path)
    logger.info(
        "  Shard %04d saved (%d graphs, RSS=%.1f GB)",
        shard_id, len(graphs), _get_rss_gb(),
    )


def main() -> None:
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest()
    processed_set = set(manifest["processed_keys"])
    skipped_set = {entry["key"] for entry in manifest["skipped"]}
    failed_set = {entry["key"] for entry in manifest["failed"]}
    already_handled = processed_set | skipped_set | failed_set

    # Discover MIDI files
    midi_files = sorted(f.name for f in MIDI_DIR.iterdir() if f.suffix == ".mid")
    logger.info("Found %d MIDI files in %s", len(midi_files), MIDI_DIR)

    # Filter to unprocessed files
    remaining = []
    for filename in midi_files:
        try:
            yt_id = _extract_youtube_id(filename)
        except ValueError as e:
            logger.warning("Skipping %s: %s", filename, e)
            continue
        key = f"giantmidi/{yt_id}"
        if key not in already_handled:
            remaining.append((filename, key))

    logger.info(
        "%d already processed, %d skipped, %d failed, %d remaining",
        len(processed_set), len(skipped_set), len(failed_set), len(remaining),
    )

    if not remaining:
        logger.info("Nothing to process.")
        return

    # Determine starting shard ID from manifest (authoritative source)
    next_shard_id = manifest["shard_range"][1] if processed_set else FIRST_SHARD_ID

    # Process files
    shard_buffer: dict[str, object] = {}
    t_start = time.time()
    processed_this_run = 0
    t_last_log = t_start

    for i, (filename, key) in enumerate(remaining):
        midi_path = MIDI_DIR / filename

        try:
            midi = pretty_midi.PrettyMIDI(str(midi_path))
        except Exception as e:
            manifest["failed"].append({"key": key, "reason": str(e)})
            logger.warning("Parse error for %s: %s", filename, e)
            continue

        note_count = count_midi_notes(midi)

        # Update histogram
        bucket = _bucket_note_count(note_count)
        manifest["note_count_buckets"][bucket] = (
            manifest["note_count_buckets"].get(bucket, 0) + 1
        )

        # Skip if too many or zero notes
        if note_count > MAX_NOTES:
            manifest["skipped"].append({
                "key": key, "reason": f"too many notes ({note_count})",
            })
            continue

        if note_count == 0:
            manifest["skipped"].append({"key": key, "reason": "no notes"})
            continue

        # Build graph
        try:
            graph = parsed_midi_to_graph(midi)
        except Exception as e:
            manifest["failed"].append({"key": key, "reason": str(e)})
            logger.warning("Graph build error for %s: %s", filename, e)
            continue

        shard_buffer[key] = graph
        manifest["processed_keys"].append(key)
        processed_this_run += 1

        # Flush shard if buffer full or RSS too high
        rss_gb = _get_rss_gb()
        if len(shard_buffer) >= SHARD_SIZE or rss_gb > RSS_FLUSH_THRESHOLD_GB:
            if rss_gb > RSS_FLUSH_THRESHOLD_GB:
                logger.warning(
                    "RSS=%.1f GB exceeds threshold, force-flushing shard", rss_gb
                )
            _save_shard(next_shard_id, shard_buffer)
            manifest["shard_range"][1] = next_shard_id + 1
            _save_manifest(manifest)
            next_shard_id += 1
            shard_buffer = {}
            del graph, midi
            gc.collect()

        # Progress log every 30 seconds
        now = time.time()
        if now - t_last_log > 30:
            elapsed = now - t_start
            rate = processed_this_run / elapsed if elapsed > 0 else 0
            eta_remaining = (len(remaining) - i - 1) / rate if rate > 0 else 0
            logger.info(
                "  [%d/%d] %d processed, RSS=%.1f GB, "
                "%.1f files/s, ETA %.0f min",
                i + 1, len(remaining), processed_this_run,
                _get_rss_gb(), rate, eta_remaining / 60,
            )
            t_last_log = now

    # Flush remaining buffer
    if shard_buffer:
        _save_shard(next_shard_id, shard_buffer)
        manifest["shard_range"][1] = next_shard_id + 1
        _save_manifest(manifest)
        del shard_buffer
        gc.collect()

    # Finalize manifest
    elapsed = time.time() - t_start
    manifest["total_processing_time_seconds"] += elapsed

    _save_manifest(manifest)

    # Summary
    logger.info("\n=== GIANTMIDI Processing Complete ===")
    logger.info("Processed: %d", len(manifest["processed_keys"]))
    logger.info("Skipped: %d", len(manifest["skipped"]))
    logger.info("Failed: %d", len(manifest["failed"]))
    logger.info("Shards: %04d - %04d", FIRST_SHARD_ID, manifest["shard_range"][1] - 1)
    logger.info("Note count distribution: %s", json.dumps(manifest["note_count_buckets"]))
    logger.info("Time: %.0f seconds (%.1f min)", elapsed, elapsed / 60)
    logger.info("Manifest: %s", MANIFEST_PATH)


if __name__ == "__main__":
    main()
