"""ASAP score MIDI discovery -- find all pieces and their score MIDI files."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from score_library.titles import clean_title_from_path, load_titles

logger = logging.getLogger(__name__)


@dataclass
class PieceEntry:
    """A discovered piece with its metadata and score MIDI path."""

    piece_id: str
    composer: str
    title: str
    score_midi_path: Path


def derive_piece_id(piece_dir: Path, base_dir: Path) -> str:
    """Derive a canonical piece ID from a directory path.

    The piece ID is the relative path from base_dir, lowercased, with dots
    as separators.

    Example: base_dir/Chopin/Etudes_op_10/3 -> "chopin.etudes_op_10.3"
    """
    rel = piece_dir.relative_to(base_dir)
    return ".".join(p.lower() for p in rel.parts)


def discover_pieces(
    asap_dir: Path,
    titles_path: Path | None = None,
) -> list[PieceEntry]:
    """Recursively discover all pieces in an ASAP dataset directory.

    A "piece directory" is any directory that directly contains one or more
    files matching the glob ``score_*.mid``. For each such directory, the first
    matching file is selected (they are typically identical).

    Args:
        asap_dir: Root of the ASAP dataset.
        titles_path: Optional path to a JSON file mapping piece_id -> title.

    Returns:
        Sorted list of PieceEntry objects.
    """
    titles = load_titles(titles_path)
    entries: list[PieceEntry] = []

    if not asap_dir.exists():
        logger.warning("ASAP directory does not exist: %s", asap_dir)
        return entries

    # Walk all subdirectories looking for score_*.mid files
    for score_midi in sorted(asap_dir.rglob("score_*.mid")):
        piece_dir = score_midi.parent

        # Skip if we already processed this directory
        if any(e.score_midi_path.parent == piece_dir for e in entries):
            continue

        piece_id = derive_piece_id(piece_dir, asap_dir)
        rel = piece_dir.relative_to(asap_dir)
        composer = rel.parts[0] if rel.parts else "Unknown"
        title = titles.get(piece_id, clean_title_from_path(piece_dir, asap_dir))

        entries.append(
            PieceEntry(
                piece_id=piece_id,
                composer=composer,
                title=title,
                score_midi_path=score_midi,
            )
        )

    logger.info("Discovered %d pieces in %s", len(entries), asap_dir)
    return sorted(entries, key=lambda e: e.piece_id)
