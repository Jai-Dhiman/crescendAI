"""Segmentation of public-domain piano pedagogy MusicXML into individual primitives.

Real method-book MusicXML encodes an entire collection as a SINGLE part (one
piano instrument), not one part per exercise. Boundaries between exercises are
expressed as structural markers inside the part. This module segments per source
according to that source's actual structure:

- "hanon":      one part containing N exercises, each wrapped in a Repeat
                barline. Split on Repeat spans -> one primitive per exercise.
- "czerny":     one part = one etude -> one primitive (whole part).
- "burgmuller": one part = one study -> one primitive (whole part).

Each primitive is exported as its own MusicXML + MIDI file. The MIDI is what
feeds Aria embedding; the MusicXML is retained for later rendering/transform.

Supported source names: "hanon", "czerny", "burgmuller"
"""

import logging
from pathlib import Path

import partitura
import partitura.utils.music
from partitura.score import Note, Part, Repeat, TimeSignature

from exercise_corpus import Primitive

logger = logging.getLogger(__name__)


class SegmentationError(Exception):
    """Raised when a source MusicXML does not match the expected boundary pattern."""


# Per-source segmentation strategy.
# boundary="repeat": split the single part on Repeat-barline spans.
# boundary="whole":  the whole part is one primitive.
_SOURCE_CONFIGS: dict[str, dict] = {
    "hanon": {
        "title_prefix": "Hanon Exercise",
        "id_prefix": "hanon",
        "boundary": "repeat",
    },
    "czerny": {
        "title_prefix": "Czerny Etude",
        "id_prefix": "czerny",
        "boundary": "whole",
    },
    "burgmuller": {
        "title_prefix": "Burgmuller Piece",
        "id_prefix": "burgmuller",
        "boundary": "whole",
    },
}


def _active_time_signature(part: Part, t: int) -> TimeSignature | None:
    """Return the TimeSignature in effect at or before time t, if any."""
    ts_list = sorted(part.iter_all(TimeSignature), key=lambda x: x.start.t)
    before = [x for x in ts_list if x.start.t <= t]
    if before:
        return before[-1]
    return ts_list[0] if ts_list else None


def _build_subpart(
    part: Part, start_t: int, end_t: int, part_id: str, part_name: str
) -> Part:
    """Construct a new Part containing the notes of `part` in [start_t, end_t),
    rebased so the segment begins at time 0. Copies the active time signature and
    lays out measures so the result is valid, exportable MusicXML.
    """
    quarter = int(part.quarter_duration_map(start_t))
    sub = Part(part_id, part_name, quarter_duration=quarter)

    ts = _active_time_signature(part, start_t)
    if ts is not None:
        sub.add(TimeSignature(ts.beats, ts.beat_type), 0)

    for note in part.iter_all(Note):
        if start_t <= note.start.t < end_t:
            sub.add(
                Note(step=note.step, octave=note.octave, alter=note.alter, voice=note.voice),
                int(note.start.t - start_t),
                int(note.end.t - start_t),
            )

    partitura.score.add_measures(sub)
    return sub


def _export_primitive(
    part: Part,
    source_name: str,
    idx: int,
    config: dict,
    output_score_dir: Path,
    output_midi_dir: Path,
) -> Primitive:
    """Export a (whole or sub) part as one primitive's MusicXML + MIDI and return
    its Primitive record."""
    n_notes = len(partitura.utils.music.ensure_notearray(part))
    if n_notes == 0:
        raise SegmentationError(
            f"{source_name}: exercise {idx} contains 0 notes"
        )

    primitive_id = f"{config['id_prefix']}_{idx:03d}"
    title = f"{config['title_prefix']} {idx}"
    xml_out = output_score_dir / f"{primitive_id}.xml"
    mid_out = output_midi_dir / f"{primitive_id}.mid"

    partitura.save_musicxml(part, str(xml_out))
    partitura.save_score_midi(part, str(mid_out))

    logger.info("Segmented %s (n_notes=%d) -> %s", primitive_id, n_notes, xml_out)
    return Primitive(
        primitive_id=primitive_id,
        source=source_name,
        source_exercise_number=idx,
        title=title,
        musicxml_path=xml_out,
        midi_path=mid_out,
        n_notes=n_notes,
    )


def segment_source(
    musicxml_path: Path,
    source_name: str,
    output_score_dir: Path,
    output_midi_dir: Path,
) -> list[Primitive]:
    """Segment a source MusicXML file into individual Primitive instances.

    Segmentation strategy is per-source (see module docstring):
    - hanon: split the single part on Repeat-barline spans (one primitive per
      exercise).
    - czerny / burgmuller: the whole part is one primitive.

    Exports per-primitive MusicXML and MIDI to the given output directories.

    Args:
        musicxml_path: path to the source MusicXML (.xml or compressed .mxl).
        source_name: one of "hanon", "czerny", "burgmuller".
        output_score_dir: directory to write per-primitive MusicXML files.
        output_midi_dir: directory to write per-primitive MIDI files.

    Returns:
        List of Primitive dataclasses ordered by source_exercise_number ascending.

    Raises:
        SegmentationError: if source_name is unrecognized; if the score has zero
            parts; if a "repeat" source has no Repeat markers; or if a resulting
            segment has zero notes.
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

    score = partitura.load_score(str(musicxml_path))
    parts = list(score.parts)
    if len(parts) == 0:
        raise SegmentationError(
            f"{source_name}: expected at least 1 part in {musicxml_path}, found 0"
        )

    primitives: list[Primitive] = []

    if config["boundary"] == "repeat":
        part = parts[0]
        repeats = sorted(part.iter_all(Repeat), key=lambda r: r.start.t)
        if not repeats:
            raise SegmentationError(
                f"{source_name}: boundary='repeat' but no Repeat markers found "
                f"in {musicxml_path}; cannot split into exercises"
            )
        for idx, rep in enumerate(repeats, start=1):
            if rep.end is None:
                raise SegmentationError(
                    f"{source_name}: repeat {idx} has no end in {musicxml_path}"
                )
            sub = _build_subpart(
                part,
                rep.start.t,
                rep.end.t,
                part_id=f"{config['id_prefix']}_{idx:03d}",
                part_name=f"{config['title_prefix']} {idx}",
            )
            primitives.append(
                _export_primitive(
                    sub, source_name, idx, config, output_score_dir, output_midi_dir
                )
            )
    else:  # "whole": each part is one primitive (real files have a single part)
        for idx, part in enumerate(parts, start=1):
            primitives.append(
                _export_primitive(
                    part, source_name, idx, config, output_score_dir, output_midi_dir
                )
            )

    return primitives
