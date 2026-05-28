"""Segmentation of multi-exercise MusicXML sources into individual primitives.

Each source (Hanon, Czerny, Burgmuller) encodes a collection of exercises as
separate <part> elements in a single MusicXML file. This module parses each
source, extracts one Primitive per part, and exports per-primitive MusicXML
and MIDI files.

Supported source names: "hanon", "czerny", "burgmuller"
"""

import logging
from pathlib import Path

import partitura
import partitura.utils.music

from exercise_corpus import Primitive

logger = logging.getLogger(__name__)


class SegmentationError(Exception):
    """Raised when a source MusicXML does not match the expected boundary pattern."""


_SOURCE_CONFIGS: dict[str, dict] = {
    "hanon": {
        "title_prefix": "Hanon Exercise",
        "id_prefix": "hanon",
    },
    "czerny": {
        "title_prefix": "Czerny Etude",
        "id_prefix": "czerny",
    },
    "burgmuller": {
        "title_prefix": "Burgmuller Piece",
        "id_prefix": "burgmuller",
    },
}


def segment_source(
    musicxml_path: Path,
    source_name: str,
    output_score_dir: Path,
    output_midi_dir: Path,
) -> list[Primitive]:
    """Segment a multi-exercise MusicXML file into individual Primitive instances.

    Each <part> in the MusicXML is treated as one exercise primitive. Exports
    per-primitive MusicXML and MIDI files to the given output directories.

    Args:
        musicxml_path: path to the source MusicXML file.
        source_name: one of "hanon", "czerny", "burgmuller".
        output_score_dir: directory to write per-primitive MusicXML files.
        output_midi_dir: directory to write per-primitive MIDI files.

    Returns:
        List of Primitive dataclasses, one per exercise, ordered by
        source_exercise_number ascending.

    Raises:
        SegmentationError: if source_name is not recognized, or if the parsed
            score contains zero parts, or if a part contains zero notes.
        FileNotFoundError: if musicxml_path does not exist.
    """
    musicxml_path = Path(musicxml_path)
    if not musicxml_path.exists():
        raise FileNotFoundError(f"MusicXML not found: {musicxml_path}")

    if source_name not in _SOURCE_CONFIGS:
        raise SegmentationError(
            f"unknown source {source_name!r}; supported: {sorted(_SOURCE_CONFIGS)}"
        )

    config = _SOURCE_CONFIGS[source_name]
    output_score_dir = Path(output_score_dir)
    output_midi_dir = Path(output_midi_dir)
    output_score_dir.mkdir(parents=True, exist_ok=True)
    output_midi_dir.mkdir(parents=True, exist_ok=True)

    score = partitura.load_musicxml(str(musicxml_path))
    parts = list(score.parts)

    if len(parts) == 0:
        raise SegmentationError(
            f"{source_name}: expected at least 1 part in {musicxml_path}, found 0"
        )

    primitives: list[Primitive] = []
    for idx, part in enumerate(parts, start=1):
        note_array = partitura.utils.music.ensure_notearray(part)
        n_notes = len(note_array)

        if n_notes == 0:
            raise SegmentationError(
                f"{source_name}: part {idx} contains 0 notes in {musicxml_path}"
            )

        primitive_id = f"{config['id_prefix']}_{idx:03d}"
        title = f"{config['title_prefix']} {idx}"
        xml_out = output_score_dir / f"{primitive_id}.xml"
        mid_out = output_midi_dir / f"{primitive_id}.mid"

        partitura.save_musicxml(part, str(xml_out))
        partitura.save_score_midi(part, str(mid_out))

        primitives.append(
            Primitive(
                primitive_id=primitive_id,
                source=source_name,
                source_exercise_number=idx,
                title=title,
                musicxml_path=xml_out,
                midi_path=mid_out,
                n_notes=n_notes,
            )
        )
        logger.info("Segmented %s (n_notes=%d) -> %s", primitive_id, n_notes, xml_out)

    return primitives
