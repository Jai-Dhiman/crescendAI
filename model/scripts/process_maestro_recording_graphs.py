"""Convert MAESTRO MIDI files into per-recording PyG graph shards for S2-max pretraining.

Memory-safe processing for M4 (32 GB RAM):
- Single MIDI parse per file (no double-parse)
- Homogeneous graphs only
- 50-graph shards with RSS monitoring
- Resumable via manifest file

Keys use the scheme `maestro_rec/{midi_filename_stem}`, which is distinct from
the existing `maestro__` prefix used for per-title graphs.

Usage:
    cd model
    uv run python scripts/process_maestro_recording_graphs.py
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Configuration ---
MAESTRO_JSON = Path(__file__).parent.parent / "data/maestro_cache/maestro-v3.0.0.json"
MIDI_BASE_DIR = Path(__file__).parent.parent / "data/maestro_cache"
SHARD_DIR = Path(__file__).parent.parent / "data/pretrain_cache/graphs/shards"
MANIFEST_PATH = (
    Path(__file__).parent.parent / "data/maestro_cache/recording_graphs_manifest.json"
)

MAX_NOTES = 10_000  # MAESTRO pieces are longer than GIANTMIDI
SHARD_SIZE = 50
RSS_FLUSH_THRESHOLD_GB = 20.0
FIRST_SHARD_ID = 241  # GIANTMIDI shards end at 0240


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
            "0-500": 0,
            "500-1000": 0,
            "1000-2000": 0,
            "2000-5000": 0,
            "5000-10000": 0,
            "10000+_skipped": 0,
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
    elif n < 5000:
        return "2000-5000"
    elif n <= MAX_NOTES:
        return "5000-10000"
    else:
        return "10000+_skipped"


def _save_shard(shard_id: int, graphs: dict[str, object]) -> None:
    """Save a graph shard to disk."""
    path = SHARD_DIR / f"graphs_{shard_id:04d}.pt"
    torch.save(graphs, path)
    logger.info(
        "  Shard %04d saved (%d graphs, RSS=%.1f GB)",
        shard_id,
        len(graphs),
        _get_rss_gb(),
    )


def _load_maestro_entries() -> list[tuple[str, str]]:
    """Load (midi_filename, key) pairs from maestro-v3.0.0.json.

    Returns:
        List of (midi_filename, key) tuples sorted by midi_filename.

    Raises:
        FileNotFoundError: If the MAESTRO JSON is not found.
        KeyError: If expected fields are missing from the JSON.
    """
    if not MAESTRO_JSON.exists():
        raise FileNotFoundError(f"MAESTRO JSON not found: {MAESTRO_JSON}")

    with open(MAESTRO_JSON) as f:
        d = json.load(f)

    if "midi_filename" not in d:
        raise KeyError("Expected 'midi_filename' column in maestro-v3.0.0.json")

    entries = []
    for _idx, midi_filename in d["midi_filename"].items():
        stem = Path(midi_filename).stem
        key = f"maestro_rec/{stem}"
        entries.append((midi_filename, key))

    # Sort for deterministic processing order
    entries.sort(key=lambda x: x[0])
    return entries


def main() -> None:
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest()
    processed_set = set(manifest["processed_keys"])
    skipped_set = {entry["key"] for entry in manifest["skipped"]}
    failed_set = {entry["key"] for entry in manifest["failed"]}
    already_handled = processed_set | skipped_set | failed_set

    # Load all MAESTRO entries
    all_entries = _load_maestro_entries()
    logger.info("Found %d entries in %s", len(all_entries), MAESTRO_JSON)

    # Filter to unprocessed entries
    remaining = [
        (midi_filename, key)
        for midi_filename, key in all_entries
        if key not in already_handled
    ]

    logger.info(
        "%d already processed, %d skipped, %d failed, %d remaining",
        len(processed_set),
        len(skipped_set),
        len(failed_set),
        len(remaining),
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

    for i, (midi_filename, key) in enumerate(remaining):
        midi_path = MIDI_BASE_DIR / midi_filename

        if not midi_path.exists():
            manifest["failed"].append({"key": key, "reason": f"file not found: {midi_path}"})
            logger.warning("File not found: %s", midi_path)
            continue

        try:
            midi = pretty_midi.PrettyMIDI(str(midi_path))
        except Exception as e:
            manifest["failed"].append({"key": key, "reason": str(e)})
            logger.warning("Parse error for %s: %s", midi_filename, e)
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
                "key": key,
                "reason": f"too many notes ({note_count})",
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
            logger.warning("Graph build error for %s: %s", midi_filename, e)
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
                i + 1,
                len(remaining),
                processed_this_run,
                _get_rss_gb(),
                rate,
                eta_remaining / 60,
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
    logger.info("\n=== MAESTRO Recording Graphs Processing Complete ===")
    logger.info("Processed: %d", len(manifest["processed_keys"]))
    logger.info("Skipped: %d", len(manifest["skipped"]))
    logger.info("Failed: %d", len(manifest["failed"]))
    logger.info("Shards: %04d - %04d", FIRST_SHARD_ID, manifest["shard_range"][1] - 1)
    logger.info("Note count distribution: %s", json.dumps(manifest["note_count_buckets"]))
    logger.info("Time: %.0f seconds (%.1f min)", elapsed, elapsed / 60)
    logger.info("Manifest: %s", MANIFEST_PATH)


if __name__ == "__main__":
    main()
