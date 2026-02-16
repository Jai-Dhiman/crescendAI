"""MIDI file discovery for multi-dataset pretraining.

Discovers MIDI files from ASAP, MAESTRO, ATEPP, and PercePiano datasets.
No downloads -- just file discovery from already-downloaded directories.
"""

from __future__ import annotations

import csv
import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MIDIFileEntry:
    """A discovered MIDI file with source metadata."""

    source: str
    source_key: str
    midi_path: Path
    composer: str
    piece: str
    performer: str


def load_asap_midi_files(asap_cache_dir: Path) -> list[MIDIFileEntry]:
    """Discover MIDI files from the ASAP dataset.

    Expects directory structure: asap_cache/{Composer}/{Piece}/{performance_*.mid}

    Args:
        asap_cache_dir: Path to the ASAP cache directory.

    Returns:
        List of MIDIFileEntry objects for each discovered performance MIDI.

    Raises:
        FileNotFoundError: If asap_cache_dir does not exist.
    """
    asap_cache_dir = Path(asap_cache_dir)
    if not asap_cache_dir.exists():
        raise FileNotFoundError(f"ASAP cache directory not found: {asap_cache_dir}")

    entries = []
    for midi_path in sorted(asap_cache_dir.glob("**/performance_*.mid")):
        # Path structure: asap_cache/{Composer}/{Piece}/performance_{performer}.mid
        rel = midi_path.relative_to(asap_cache_dir)
        parts = rel.parts
        if len(parts) < 3:
            logger.warning("Unexpected ASAP path structure: %s", midi_path)
            continue

        composer = parts[0]
        piece = parts[1] if len(parts) == 3 else "/".join(parts[1:-1])
        performer = midi_path.stem.replace("performance_", "")

        source_key = f"asap__{composer}/{piece}/{performer}"
        entries.append(
            MIDIFileEntry(
                source="asap",
                source_key=source_key,
                midi_path=midi_path,
                composer=composer,
                piece=piece,
                performer=performer,
            )
        )

    return entries


def load_maestro_midi_files(maestro_dir: Path) -> list[MIDIFileEntry]:
    """Discover MIDI files from the MAESTRO v3 dataset.

    Parses maestro-v3.0.0.json for metadata and matches with .midi files on disk.

    Args:
        maestro_dir: Path to the MAESTRO directory (contains maestro-v3.0.0.json).

    Returns:
        List of MIDIFileEntry objects for each discovered MIDI.

    Raises:
        FileNotFoundError: If maestro_dir or metadata JSON does not exist.
    """
    maestro_dir = Path(maestro_dir)
    if not maestro_dir.exists():
        raise FileNotFoundError(f"MAESTRO directory not found: {maestro_dir}")

    metadata_path = maestro_dir / "maestro-v3.0.0.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"MAESTRO metadata not found: {metadata_path}. "
            "Expected maestro-v3.0.0.json in the MAESTRO directory."
        )

    with open(metadata_path) as f:
        raw = json.load(f)

    # MAESTRO JSON is column-oriented: {col_name: {row_idx: value}}
    # Convert to list of row dicts.
    if isinstance(raw, dict) and "midi_filename" in raw:
        row_keys = list(raw["midi_filename"].keys())
        metadata = [
            {col: raw[col][k] for col in raw}
            for k in row_keys
        ]
    elif isinstance(raw, list):
        metadata = raw
    else:
        raise ValueError(
            f"Unexpected MAESTRO JSON format: top-level type={type(raw).__name__}"
        )

    entries = []
    for record in metadata:
        midi_filename = record.get("midi_filename")
        if not midi_filename:
            continue

        midi_path = maestro_dir / midi_filename
        if not midi_path.exists():
            logger.warning("MAESTRO MIDI not found: %s", midi_path)
            continue

        canonical_title = record.get("canonical_title", midi_path.stem)
        canonical_composer = record.get("canonical_composer", "Unknown")

        source_key = f"maestro__{canonical_title}"
        entries.append(
            MIDIFileEntry(
                source="maestro",
                source_key=source_key,
                midi_path=midi_path,
                composer=canonical_composer,
                piece=canonical_title,
                performer="Unknown",
            )
        )

    return entries


def load_atepp_midi_files(atepp_dir: Path) -> list[MIDIFileEntry]:
    """Discover MIDI files from the ATEPP dataset.

    Parses ATEPP_metadata.csv for metadata and discovers .mid files.

    Args:
        atepp_dir: Path to the ATEPP directory (contains ATEPP_metadata.csv).

    Returns:
        List of MIDIFileEntry objects for each discovered MIDI.

    Raises:
        FileNotFoundError: If atepp_dir or metadata CSV does not exist.
    """
    atepp_dir = Path(atepp_dir)
    if not atepp_dir.exists():
        raise FileNotFoundError(f"ATEPP directory not found: {atepp_dir}")

    metadata_path = atepp_dir / "ATEPP_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"ATEPP metadata not found: {metadata_path}. "
            "Expected ATEPP_metadata.csv in the ATEPP directory."
        )

    # Build metadata lookup from CSV
    meta_lookup: dict[str, dict] = {}
    with open(metadata_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            midi_path_str = row.get("midi_path") or row.get("path", "")
            if midi_path_str:
                meta_lookup[midi_path_str] = row

    entries = []
    for midi_path in sorted(atepp_dir.glob("**/*.mid")):
        rel = midi_path.relative_to(atepp_dir)
        rel_stem = str(rel.with_suffix(""))
        rel_str = str(rel)

        # Try to find metadata by relative path
        meta = meta_lookup.get(rel_str) or meta_lookup.get(rel_stem)

        if meta:
            composer = meta.get("composer", "Unknown")
            piece = meta.get("track", "") or meta.get("work", "") or meta.get("piece", "Unknown")
            performer = meta.get("artist", "") or meta.get("performer", "Unknown")
        else:
            # Infer from directory structure
            parts = rel.parts
            composer = parts[0] if len(parts) > 1 else "Unknown"
            piece = parts[1] if len(parts) > 2 else "Unknown"
            performer = midi_path.stem

        source_key = f"atepp__{rel_stem}"
        entries.append(
            MIDIFileEntry(
                source="atepp",
                source_key=source_key,
                midi_path=midi_path,
                composer=composer,
                piece=piece,
                performer=performer,
            )
        )

    return entries


def load_percepiano_midi_files(midi_dir: Path) -> list[MIDIFileEntry]:
    """Discover MIDI files from the PercePiano dataset.

    Simple glob of *.mid files in the directory.

    Args:
        midi_dir: Path to the PercePiano MIDI directory.

    Returns:
        List of MIDIFileEntry objects for each discovered MIDI.

    Raises:
        FileNotFoundError: If midi_dir does not exist.
    """
    midi_dir = Path(midi_dir)
    if not midi_dir.exists():
        raise FileNotFoundError(f"PercePiano MIDI directory not found: {midi_dir}")

    entries = []
    for midi_path in sorted(midi_dir.glob("*.mid")):
        stem = midi_path.stem
        source_key = f"percepiano__{stem}"
        entries.append(
            MIDIFileEntry(
                source="percepiano",
                source_key=source_key,
                midi_path=midi_path,
                composer="Unknown",
                piece="Unknown",
                performer=stem,
            )
        )

    return entries


def load_all_midi_files(data_dir: Path) -> list[MIDIFileEntry]:
    """Discover MIDI files from all available datasets.

    Calls all individual loaders, skipping sources whose directories
    don't exist (with a warning). Raises errors for missing expected
    metadata files within existing directories.

    Args:
        data_dir: Base data directory containing subdirectories for each source.

    Returns:
        Combined list of MIDIFileEntry objects from all available sources.
    """
    data_dir = Path(data_dir)
    all_entries: list[MIDIFileEntry] = []
    source_counts: dict[str, int] = {}

    loaders = [
        ("asap", data_dir / "asap_cache", load_asap_midi_files),
        ("maestro", data_dir / "maestro_cache", load_maestro_midi_files),
        ("atepp", data_dir / "atepp_cache", load_atepp_midi_files),
        ("percepiano", data_dir / "percepiano_midi", load_percepiano_midi_files),
    ]

    for source_name, source_dir, loader_fn in loaders:
        if not source_dir.exists():
            warnings.warn(
                f"{source_name} directory not found at {source_dir}, skipping",
                stacklevel=2,
            )
            source_counts[source_name] = 0
            continue

        entries = loader_fn(source_dir)
        all_entries.extend(entries)
        source_counts[source_name] = len(entries)

    print("MIDI file discovery summary:")
    for source, count in source_counts.items():
        print(f"  {source}: {count} files")
    print(f"  Total: {len(all_entries)} files")

    return all_entries
