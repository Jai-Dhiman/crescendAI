"""Transcribe Chopin 2021 competition audio to MIDI and convert to graphs.

Runs on Thunder Compute (needs GPU for ByteDance piano_transcription).

Pipeline:
  1. Download 66 recordings from YouTube (yt-dlp)
  2. Transcribe each to MIDI (ByteDance piano_transcription)
  3. Convert MIDI to PyG graphs for S2 ordinal training

Usage:
    cd model
    uv run python scripts/transcribe_competition_midi.py
"""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path

import pretty_midi
import torch

from model_improvement.competition import (
    CompetitionRecord,
    _download_audio,
    load_competition_metadata,
)
from model_improvement.graph import count_midi_notes, parsed_midi_to_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Configuration ---
CACHE_DIR = Path(__file__).parent.parent / "data/competition_cache/chopin2021"
AUDIO_DIR = CACHE_DIR / "audio"
MIDI_DIR = CACHE_DIR / "transcribed_midi"
SHARD_DIR = Path(__file__).parent.parent / "data/pretrain_cache/graphs/shards"
RECORDINGS_PATH = CACHE_DIR / "recordings.jsonl"

SHARD_SIZE = 50
MAX_NOTES = 10_000
# Competition shards start after MAESTRO shards (~0241-0266)
FIRST_SHARD_ID = 267

MANIFEST_PATH = CACHE_DIR / "transcription_manifest.json"


# ---------------------------------------------------------------------------
# Phase 1: Download
# ---------------------------------------------------------------------------

def phase1_download() -> list[CompetitionRecord]:
    """Download competition recordings from YouTube.

    Reads source URLs from recordings.jsonl, downloads each to audio/.
    Skips already-downloaded files (resumable).

    Returns:
        List of CompetitionRecord for all recordings (downloaded + pre-existing).

    Raises:
        FileNotFoundError: If recordings.jsonl does not exist.
    """
    if not RECORDINGS_PATH.exists():
        raise FileNotFoundError(
            f"recordings.jsonl not found at {RECORDINGS_PATH}. "
            "Run collect_competition_data.py first."
        )

    records = load_competition_metadata(CACHE_DIR)
    if not records:
        raise RuntimeError(f"No records found in {RECORDINGS_PATH}")

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    skipped = 0

    for rec in records:
        recording_id = rec["recording_id"]
        source_url = rec.get("source_url", "")
        wav_path = AUDIO_DIR / f"{recording_id}.wav"

        if wav_path.exists() and wav_path.stat().st_size > 0:
            logger.debug("Skipping %s (already downloaded)", recording_id)
            skipped += 1
            continue

        if not source_url:
            logger.warning("No source_url for %s, skipping", recording_id)
            continue

        logger.info("Downloading %s ...", recording_id)
        try:
            _download_audio(
                url=source_url,
                output_path=wav_path,
                start_sec=None,
                end_sec=None,
            )
        except Exception as e:
            logger.error("Failed to download %s: %s", recording_id, e)
            continue

        if not wav_path.exists() or wav_path.stat().st_size == 0:
            logger.error("Download produced empty file for %s", recording_id)
            continue

        downloaded += 1

    logger.info(
        "Phase 1 complete: %d downloaded, %d skipped (already present)",
        downloaded, skipped,
    )

    # Rebuild as CompetitionRecord dataclasses
    competition_records = [
        CompetitionRecord(
            recording_id=r["recording_id"],
            competition=r["competition"],
            edition=r["edition"],
            round=r["round"],
            placement=r["placement"],
            performer=r["performer"],
            piece=r["piece"],
            audio_path=r["audio_path"],
            duration_seconds=r["duration_seconds"],
            source_url=r.get("source_url", ""),
            country=r.get("country", ""),
        )
        for r in records
    ]
    return competition_records


# ---------------------------------------------------------------------------
# Phase 2: Transcribe (GPU required)
# ---------------------------------------------------------------------------

def phase2_transcribe(records: list[CompetitionRecord]) -> None:
    """Transcribe competition WAV files to MIDI using ByteDance piano_transcription.

    Requires CUDA GPU.  # TODO(thunder): requires GPU

    Args:
        records: List of CompetitionRecord to transcribe.

    Raises:
        ImportError: If piano_transcription_inference is not installed.
        RuntimeError: If CUDA is not available.
    """
    # TODO(thunder): requires GPU
    try:
        from piano_transcription_inference import PianoTranscription  # type: ignore[import]
    except ImportError as e:
        raise ImportError(
            "piano_transcription_inference is not installed. "
            "Install it on Thunder Compute: uv pip install piano-transcription-inference"
        ) from e

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU required for ByteDance piano_transcription. "
            "Run this phase on Thunder Compute."
        )

    MIDI_DIR.mkdir(parents=True, exist_ok=True)

    # TODO(thunder): requires GPU
    transcriber = PianoTranscription(device="cuda")

    transcribed = 0
    skipped = 0
    failed = 0

    for rec in records:
        wav_path = AUDIO_DIR / f"{rec.recording_id}.wav"
        midi_path = MIDI_DIR / f"{rec.recording_id}.mid"

        if midi_path.exists() and midi_path.stat().st_size > 0:
            logger.debug("Skipping %s (already transcribed)", rec.recording_id)
            skipped += 1
            continue

        if not wav_path.exists():
            logger.warning("WAV not found for %s, skipping", rec.recording_id)
            continue

        logger.info("Transcribing %s ...", rec.recording_id)
        try:
            # TODO(thunder): requires GPU
            transcriber.transcribe(str(wav_path), str(midi_path))
        except Exception as e:
            logger.error("Transcription failed for %s: %s", rec.recording_id, e)
            failed += 1
            continue

        if not midi_path.exists() or midi_path.stat().st_size == 0:
            logger.error("Transcription produced empty MIDI for %s", rec.recording_id)
            failed += 1
            continue

        # Log note count to verify transcription quality
        try:
            midi = pretty_midi.PrettyMIDI(str(midi_path))
            n_notes = count_midi_notes(midi)
            logger.info("  %s: %d notes", rec.recording_id, n_notes)
        except Exception as e:
            logger.warning("Could not count notes for %s: %s", rec.recording_id, e)

        transcribed += 1

    logger.info(
        "Phase 2 complete: %d transcribed, %d skipped, %d failed",
        transcribed, skipped, failed,
    )


# ---------------------------------------------------------------------------
# Phase 3: Graph conversion
# ---------------------------------------------------------------------------

def _load_manifest() -> dict:
    """Load existing graph conversion manifest, or return empty template."""
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return {
        "processed_keys": [],
        "skipped": [],
        "failed": [],
        "shard_range": [FIRST_SHARD_ID, FIRST_SHARD_ID],
    }


def _save_manifest(manifest: dict) -> None:
    """Save graph conversion manifest to disk."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


def _save_shard(shard_id: int, graphs: dict) -> None:
    """Save a graph shard to disk."""
    path = SHARD_DIR / f"graphs_{shard_id:04d}.pt"
    torch.save(graphs, path)
    logger.info("Shard %04d saved (%d graphs)", shard_id, len(graphs))


def phase3_convert_graphs() -> None:
    """Convert transcribed MIDI files to PyG graph shards.

    Key scheme: competition/{recording_id}
    Shards start at FIRST_SHARD_ID=267 (after MAESTRO shards ~0241-0266).
    Resumable via manifest file.
    """
    SHARD_DIR.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest()
    processed_set = set(manifest["processed_keys"])
    skipped_set = {entry["key"] for entry in manifest["skipped"]}
    failed_set = {entry["key"] for entry in manifest["failed"]}
    already_handled = processed_set | skipped_set | failed_set

    midi_files = sorted(MIDI_DIR.glob("*.mid"))
    if not midi_files:
        logger.warning("No MIDI files found in %s. Run phase 2 first.", MIDI_DIR)
        return

    logger.info("Found %d transcribed MIDI files", len(midi_files))

    # Filter to unprocessed files
    remaining = []
    for midi_path in midi_files:
        recording_id = midi_path.stem
        key = f"competition/{recording_id}"
        if key not in already_handled:
            remaining.append((midi_path, key))

    logger.info(
        "%d already processed, %d remaining",
        len(already_handled), len(remaining),
    )

    if not remaining:
        logger.info("Nothing to process.")
        return

    next_shard_id = manifest["shard_range"][1] if processed_set else FIRST_SHARD_ID
    shard_buffer: dict[str, object] = {}

    for midi_path, key in remaining:
        try:
            midi = pretty_midi.PrettyMIDI(str(midi_path))
        except Exception as e:
            manifest["failed"].append({"key": key, "reason": str(e)})
            logger.warning("Parse error for %s: %s", midi_path.name, e)
            continue

        note_count = count_midi_notes(midi)

        if note_count == 0:
            manifest["skipped"].append({"key": key, "reason": "no notes"})
            logger.debug("Skipping %s: no notes", midi_path.name)
            continue

        if note_count > MAX_NOTES:
            manifest["skipped"].append({
                "key": key,
                "reason": f"too many notes ({note_count})",
            })
            logger.debug("Skipping %s: %d notes (> %d)", midi_path.name, note_count, MAX_NOTES)
            continue

        try:
            graph = parsed_midi_to_graph(midi)
        except Exception as e:
            manifest["failed"].append({"key": key, "reason": str(e)})
            logger.warning("Graph build error for %s: %s", midi_path.name, e)
            continue

        shard_buffer[key] = graph
        manifest["processed_keys"].append(key)

        if len(shard_buffer) >= SHARD_SIZE:
            _save_shard(next_shard_id, shard_buffer)
            manifest["shard_range"][1] = next_shard_id + 1
            _save_manifest(manifest)
            next_shard_id += 1
            shard_buffer = {}
            del graph, midi
            gc.collect()

    # Flush remaining buffer
    if shard_buffer:
        _save_shard(next_shard_id, shard_buffer)
        manifest["shard_range"][1] = next_shard_id + 1
        _save_manifest(manifest)
        del shard_buffer
        gc.collect()

    _save_manifest(manifest)

    logger.info("Phase 3 complete.")
    logger.info("Processed: %d", len(manifest["processed_keys"]))
    logger.info("Skipped: %d", len(manifest["skipped"]))
    logger.info("Failed: %d", len(manifest["failed"]))
    logger.info(
        "Shards: %04d - %04d",
        FIRST_SHARD_ID, manifest["shard_range"][1] - 1,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=== Competition MIDI Transcription Pipeline ===")
    logger.info("Cache dir: %s", CACHE_DIR)

    # Phase 1: Download audio
    logger.info("\n--- Phase 1: Download audio ---")
    records = phase1_download()
    logger.info("Total records: %d", len(records))

    # Phase 2: Transcribe to MIDI (GPU required on Thunder Compute)
    logger.info("\n--- Phase 2: Transcribe to MIDI ---")
    logger.info("NOTE: Phase 2 requires CUDA GPU.  # TODO(thunder): requires GPU")
    phase2_transcribe(records)

    # Phase 3: Convert to graphs
    logger.info("\n--- Phase 3: Convert MIDI to graphs ---")
    phase3_convert_graphs()

    logger.info("\n=== Pipeline complete ===")


if __name__ == "__main__":
    main()
