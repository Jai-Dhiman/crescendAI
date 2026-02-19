"""MAESTRO v3 audio pipeline: metadata, segmentation, MuQ embedding extraction.

Processes MAESTRO v3 audio recordings into 30s segments with MuQ embeddings
for cross-performer contrastive training (T3 data tier).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import jsonlines

logger = logging.getLogger(__name__)


@dataclass
class MaestroSegment:
    segment_id: str
    audio_filename: str
    canonical_title: str
    canonical_composer: str
    split: str
    segment_start: float
    segment_end: float
    duration_seconds: float


def parse_maestro_audio_metadata(maestro_dir: Path) -> list[dict]:
    """Parse MAESTRO v3 metadata JSON for audio file entries.

    Args:
        maestro_dir: Path to MAESTRO root (contains maestro-v3.0.0.json).

    Returns:
        List of dicts with keys: audio_filename, canonical_title,
        canonical_composer, split, duration, midi_filename.

    Raises:
        FileNotFoundError: If maestro_dir or metadata JSON does not exist.
    """
    maestro_dir = Path(maestro_dir)
    if not maestro_dir.exists():
        raise FileNotFoundError(f"MAESTRO directory not found: {maestro_dir}")

    metadata_path = maestro_dir / "maestro-v3.0.0.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"MAESTRO metadata not found: {metadata_path}")

    with open(metadata_path) as f:
        raw = json.load(f)

    # MAESTRO JSON is column-oriented: {col_name: {row_idx: value}}
    if isinstance(raw, dict) and "audio_filename" in raw:
        row_keys = list(raw["audio_filename"].keys())
        records = [
            {col: raw[col][k] for col in raw}
            for k in row_keys
        ]
    elif isinstance(raw, list):
        records = raw
    else:
        raise ValueError(
            f"Unexpected MAESTRO JSON format: top-level type={type(raw).__name__}"
        )

    # Filter to records that have audio_filename
    records = [r for r in records if r.get("audio_filename")]

    logger.info("Parsed %d MAESTRO audio records", len(records))
    return records


def segment_and_embed_maestro(
    maestro_dir: Path,
    cache_dir: Path,
    segment_duration: float = 30.0,
    min_segment_duration: float = 5.0,
) -> int:
    """Segment all MAESTRO audio and extract per-segment MuQ embeddings.

    For each audio file in MAESTRO, loads audio via Pedalboard, segments
    into 30s clips, extracts MuQ embeddings per segment, and writes:
    - cache_dir/metadata.jsonl with per-segment metadata
    - cache_dir/muq_embeddings/{segment_id}.pt per segment

    Args:
        maestro_dir: Path to MAESTRO root with audio files.
        cache_dir: Output directory for cached segments and embeddings.
        segment_duration: Duration of each segment in seconds.
        min_segment_duration: Minimum duration for last segment.

    Returns:
        Count of newly processed segments.
    """
    import torch
    from audio_experiments.extractors.muq import MuQExtractor
    from model_improvement.audio_utils import load_audio, segment_audio

    metadata_path = cache_dir / "metadata.jsonl"
    emb_dir = cache_dir / "muq_embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    records = parse_maestro_audio_metadata(maestro_dir)

    # Load existing segment IDs for idempotency
    existing_segments: set[str] = set()
    if metadata_path.exists():
        with jsonlines.open(metadata_path) as reader:
            for seg in reader:
                existing_segments.add(seg["segment_id"])

    extractor = MuQExtractor(cache_dir=emb_dir)
    new_count = 0

    for i, record in enumerate(records):
        audio_filename = record["audio_filename"]
        audio_path = maestro_dir / audio_filename

        if not audio_path.exists():
            logger.warning("Audio file not found: %s", audio_path)
            continue

        # Create a stable base ID from the audio filename
        base_id = audio_filename.replace("/", "_").replace(".", "_")
        base_id = f"maestro_{base_id}"

        # Skip if any segments for this recording already exist
        if any(sid.startswith(base_id) for sid in existing_segments):
            if i % 100 == 0:
                logger.debug("[%d/%d] Already processed: %s", i, len(records), base_id)
            continue

        logger.info("[%d/%d] Processing: %s", i + 1, len(records), audio_filename)

        audio, sr = load_audio(audio_path, target_sr=24000)
        segments = segment_audio(
            audio, sr=sr,
            segment_duration=segment_duration,
            min_duration=min_segment_duration,
        )

        for j, seg in enumerate(segments):
            segment_id = f"{base_id}_seg{j:03d}"

            if segment_id in existing_segments:
                continue

            audio_tensor = torch.from_numpy(seg["audio"]).float()
            embedding = extractor.extract_from_audio(audio_tensor)
            torch.save(embedding, emb_dir / f"{segment_id}.pt")

            seg_record = MaestroSegment(
                segment_id=segment_id,
                audio_filename=audio_filename,
                canonical_title=record.get("canonical_title", "Unknown"),
                canonical_composer=record.get("canonical_composer", "Unknown"),
                split=record.get("split", "train"),
                segment_start=seg["start_sec"],
                segment_end=seg["end_sec"],
                duration_seconds=seg["end_sec"] - seg["start_sec"],
            )

            with jsonlines.open(metadata_path, mode="a") as writer:
                writer.write(asdict(seg_record))

            existing_segments.add(segment_id)
            new_count += 1

    del extractor
    logger.info("Processed %d new MAESTRO segments", new_count)
    return new_count


def build_piece_performer_mapping(cache_dir: Path) -> dict:
    """Build piece-to-segment mapping for contrastive pair generation.

    Groups segments by canonical_title. Pieces with 2+ distinct source
    recordings enable contrastive learning (same piece, different performer).

    Args:
        cache_dir: Directory containing metadata.jsonl.

    Returns:
        Dict mapping canonical_title -> list of segment_ids.
        Only includes pieces with 2+ distinct source recordings.
    """
    metadata_path = cache_dir / "metadata.jsonl"
    if not metadata_path.exists():
        return {}

    # Group segments by piece, tracking which source recordings they come from
    piece_to_recordings: dict[str, set[str]] = {}
    piece_to_segments: dict[str, list[str]] = {}

    with jsonlines.open(metadata_path) as reader:
        for seg in reader:
            title = seg["canonical_title"]
            audio_file = seg["audio_filename"]
            segment_id = seg["segment_id"]

            if title not in piece_to_recordings:
                piece_to_recordings[title] = set()
                piece_to_segments[title] = []

            piece_to_recordings[title].add(audio_file)
            piece_to_segments[title].append(segment_id)

    # Filter to pieces with 2+ distinct recordings
    contrastive_mapping = {
        title: segments
        for title, segments in piece_to_segments.items()
        if len(piece_to_recordings[title]) >= 2
    }

    logger.info(
        "Contrastive mapping: %d pieces with 2+ recordings, %d total segments",
        len(contrastive_mapping),
        sum(len(s) for s in contrastive_mapping.values()),
    )

    return contrastive_mapping
