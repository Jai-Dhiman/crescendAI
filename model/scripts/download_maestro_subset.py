"""Download MAESTRO v3 WAV files for the 50 selected AMT recordings.

Maps recording IDs from select_maestro_subset() back to audio filenames,
then extracts individual files from the remote MAESTRO zip via HTTP range
requests (avoids downloading the full 108GB archive).

Requires: uv pip install remotezip

Run from model/ directory:
    python scripts/download_maestro_subset.py
    python scripts/download_maestro_subset.py --n-recordings 20
"""

import argparse
import json
import logging
import sys
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MODEL_ROOT / "src"))

from src.paths import Embeddings
from model_improvement.layer1_validation import select_maestro_subset
from model_improvement.maestro import parse_maestro_audio_metadata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MAESTRO_ZIP_URL = (
    "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/"
    "maestro-v3.0.0.zip"
)
# Files inside the zip are prefixed with this directory
ZIP_PREFIX = "maestro-v3.0.0/"


def recording_id_to_audio_filename(
    recording_id: str, maestro_records: list[dict]
) -> str | None:
    """Map a recording ID back to a MAESTRO audio_filename.

    Recording IDs have format: maestro_{year}_{filename_with_underscores}
    Audio filenames have format: {year}/{filename}.wav
    """
    body = recording_id
    if body.startswith("maestro_"):
        body = body[len("maestro_"):]
    # Also strip _segNNN if present (backwards compat)
    seg_idx = body.rfind("_seg")
    if seg_idx >= 0 and body[seg_idx + 4:].isdigit():
        body = body[:seg_idx]

    for record in maestro_records:
        af = record["audio_filename"]
        normalized = af.replace("/", "_").replace(".", "_")
        if normalized == body:
            return af

    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download MAESTRO v3 audio for AMT subset"
    )
    parser.add_argument(
        "--maestro-dir",
        type=Path,
        default=Embeddings.maestro,
        help="MAESTRO cache directory (contains maestro-v3.0.0.json)",
    )
    parser.add_argument(
        "--n-recordings",
        type=int,
        default=50,
        help="Number of recordings to select (default: 50)",
    )
    args = parser.parse_args()

    maestro_dir = args.maestro_dir

    # Load contrastive mapping
    mapping_path = maestro_dir / "contrastive_mapping.json"
    if not mapping_path.exists():
        logger.error("Contrastive mapping not found: %s", mapping_path)
        sys.exit(1)

    with open(mapping_path) as f:
        contrastive_mapping = json.load(f)

    # Select subset (returns unique recording IDs)
    selected_recordings = select_maestro_subset(contrastive_mapping, args.n_recordings)
    logger.info("Selected %d recordings", len(selected_recordings))

    # Map recording IDs to audio filenames
    maestro_records = parse_maestro_audio_metadata(maestro_dir)

    audio_filenames = set()
    unmapped = []
    for rec_id in selected_recordings:
        af = recording_id_to_audio_filename(rec_id, maestro_records)
        if af:
            audio_filenames.add(af)
        else:
            unmapped.append(rec_id)

    if unmapped:
        logger.warning("Could not map %d recording IDs to audio files", len(unmapped))
        for s in unmapped[:5]:
            logger.warning("  %s", s)

    # Filter to files not already on disk
    to_download = []
    skipped = 0
    for af in sorted(audio_filenames):
        local_path = maestro_dir / af
        if local_path.exists():
            skipped += 1
        else:
            to_download.append(af)

    logger.info(
        "Audio files: %d total, %d already on disk, %d to download",
        len(audio_filenames), skipped, len(to_download),
    )

    if not to_download:
        logger.info("All audio files already present. Nothing to download.")
    else:
        # Extract files from remote zip using HTTP range requests
        from remotezip import RemoteZip

        logger.info("Opening remote MAESTRO zip (reading central directory)...")
        downloaded = 0
        failed = 0

        with RemoteZip(MAESTRO_ZIP_URL) as rz:
            zip_names = set(rz.namelist())

            for af in to_download:
                zip_path = ZIP_PREFIX + af
                if zip_path not in zip_names:
                    logger.error("Not found in zip: %s", zip_path)
                    failed += 1
                    continue

                local_path = maestro_dir / af
                local_path.parent.mkdir(parents=True, exist_ok=True)

                logger.info("Extracting: %s", af)
                try:
                    data = rz.read(zip_path)
                    local_path.write_bytes(data)
                    downloaded += 1
                except Exception as e:
                    logger.error("Failed to extract %s: %s", af, e)
                    if local_path.exists():
                        local_path.unlink()
                    failed += 1

        logger.info(
            "Done: %d downloaded, %d skipped (exist), %d failed",
            downloaded, skipped, failed,
        )

    # Report ground-truth MIDI availability
    midi_count = 0
    for af in sorted(audio_filenames):
        for record in maestro_records:
            if record["audio_filename"] == af:
                midi_file = record.get("midi_filename", "")
                midi_path = maestro_dir / midi_file
                if midi_path.exists():
                    midi_count += 1
                else:
                    logger.warning("Ground-truth MIDI missing: %s", midi_file)
                break

    logger.info("Ground-truth MIDI files found: %d / %d", midi_count, len(audio_filenames))


if __name__ == "__main__":
    main()
