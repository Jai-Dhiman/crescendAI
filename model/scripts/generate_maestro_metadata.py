"""
Generate metadata.jsonl for MAESTRO MuQ embeddings.

Maps each of the 24,321 .pt embedding files in maestro_cache/muq_embeddings/
to piece/performer info from maestro-v3.0.0.json.

Output: model/data/maestro_cache/metadata.jsonl
"""

import json
import logging
import os
import re
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)


def load_maestro_json(json_path: Path) -> list[dict]:
    """Load column-oriented MAESTRO JSON and return list of row dicts."""
    with open(json_path) as f:
        col_data = json.load(f)

    # Determine number of entries from first column
    first_col = next(iter(col_data.values()))
    n = len(first_col)

    rows = []
    for i in range(n):
        key = str(i)
        row = {col: col_data[col][key] for col in col_data}
        rows.append(row)

    return rows


def build_prefix_lookup(rows: list[dict]) -> dict[str, dict]:
    """Build a mapping from embedding prefix to MAESTRO row."""
    lookup: dict[str, dict] = {}

    for row in rows:
        audio_filename = row["audio_filename"]
        year = int(row["year"])

        basename = os.path.basename(audio_filename)
        basename_no_dots = basename.replace(".", "_")
        prefix = f"maestro_{year}_{basename_no_dots}"

        if prefix in lookup:
            raise ValueError(
                f"Duplicate prefix detected: {prefix!r}. "
                f"Colliding entries: {lookup[prefix]['audio_filename']!r} vs {audio_filename!r}"
            )

        lookup[prefix] = row

    return lookup


def extract_prefix(stem: str) -> str:
    """
    Extract the embedding prefix from a filename stem (without .pt extension).

    Expects format: <prefix>_seg<NNN>
    Example: maestro_2018_MIDI-..._wav_seg000 -> maestro_2018_MIDI-..._wav
    """
    match = re.match(r"^(.+)_seg\d+$", stem)
    if match is None:
        raise ValueError(f"Filename stem does not match expected pattern: {stem!r}")
    return match.group(1)


def derive_recording_id(audio_filename: str) -> str:
    """
    Derive recording_id from audio_filename.

    Example: '2018/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1.wav'
          -> 'MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1'
    """
    basename = os.path.basename(audio_filename)
    # Strip the .wav extension
    if basename.endswith(".wav"):
        return basename[:-4]
    # Fallback: strip last extension
    stem, _ = os.path.splitext(basename)
    return stem


def main() -> None:
    script_dir = Path(__file__).parent
    model_dir = script_dir.parent
    cache_dir = model_dir / "data" / "maestro_cache"
    embeddings_dir = cache_dir / "muq_embeddings"
    json_path = cache_dir / "maestro-v3.0.0.json"
    output_path = cache_dir / "metadata.jsonl"

    if not json_path.exists():
        raise FileNotFoundError(f"MAESTRO JSON not found: {json_path}")

    if not embeddings_dir.exists():
        raise FileNotFoundError(f"Embeddings directory not found: {embeddings_dir}")

    log.info("Loading MAESTRO metadata from %s", json_path)
    rows = load_maestro_json(json_path)
    log.info("Loaded %d MAESTRO entries", len(rows))

    log.info("Building prefix lookup")
    prefix_lookup = build_prefix_lookup(rows)
    log.info("Built lookup with %d prefixes", len(prefix_lookup))

    log.info("Scanning embedding files in %s", embeddings_dir)
    pt_files = sorted(embeddings_dir.glob("*.pt"))
    log.info("Found %d .pt files", len(pt_files))

    unmatched: list[str] = []
    records: list[dict] = []

    for pt_file in pt_files:
        stem = pt_file.stem  # filename without .pt
        try:
            prefix = extract_prefix(stem)
        except ValueError as exc:
            log.warning("Skipping malformed filename %s: %s", pt_file.name, exc)
            unmatched.append(pt_file.name)
            continue

        row = prefix_lookup.get(prefix)
        if row is None:
            log.warning("No MAESTRO entry for prefix %r (file: %s)", prefix, pt_file.name)
            unmatched.append(pt_file.name)
            continue

        segment_id = stem
        recording_id = derive_recording_id(row["audio_filename"])

        record = {
            "segment_id": segment_id,
            "recording_id": recording_id,
            "canonical_title": row["canonical_title"],
            "canonical_composer": row["canonical_composer"],
            "split": row["split"],
            "year": int(row["year"]),
            "midi_filename": row["midi_filename"],
        }
        records.append(record)

    log.info("Writing %d records to %s", len(records), output_path)
    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    log.info("Done. Wrote %d records.", len(records))

    if unmatched:
        log.warning("%d unmatched files:", len(unmatched))
        for name in unmatched[:20]:
            log.warning("  %s", name)
        if len(unmatched) > 20:
            log.warning("  ... and %d more", len(unmatched) - 20)
        sys.exit(1)
    else:
        log.info("All %d embedding files matched successfully.", len(pt_files))


if __name__ == "__main__":
    main()
