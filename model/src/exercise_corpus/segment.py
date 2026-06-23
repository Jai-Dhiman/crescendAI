"""Segmentation of public-domain piano pedagogy MusicXML into individual primitives.

Real method-book MusicXML encodes an entire collection as a SINGLE part (one
piano instrument), not one part per exercise. Boundaries between exercises are
expressed as structural markers inside the part. This module segments per source
according to that source's actual structure:

- "hanon":      one part containing N exercises, each wrapped in a Repeat
                barline. Split on Repeat spans -> one primitive per exercise.
- per-file MIDI sources ("bach", "czerny", "burgmuller", "chopin", "satie"):
                a DIRECTORY of per-piece .mid files (one piece = one file,
                acquired from the Mutopia Project). Each file is one primitive;
                no boundary detection is needed.

For the repeat-boundary source each primitive is exported as its own MusicXML +
MIDI. For per-file MIDI sources there is no MusicXML (Mutopia ships LilyPond +
MIDI, and there is no robust .ly -> MusicXML converter); the raw .mid is copied
verbatim into the MIDI output dir and `musicxml_path` is set to that MIDI path.
The MIDI is what feeds Aria embedding; nothing downstream of slice A reads
`musicxml_path` yet.

Supported source names: "hanon", "bach", "czerny", "burgmuller", "chopin",
"satie" (#17 Mutopia core) and "beethoven_sonatas", "mozart_sonatas",
"scarlatti_sonatas", "haydn_sonatas", "joplin_rags", "chopin_mazurkas"
(#49 KernScores expansion -- all per_file).
"""

import logging
import shutil
from pathlib import Path

import partitura
import partitura.utils.music
from partitura.score import Note, Part, Repeat, TimeSignature

from exercise_corpus import Primitive

logger = logging.getLogger(__name__)


class SegmentationError(Exception):
    """Raised when a source MusicXML does not match the expected boundary pattern."""


# Per-source segmentation strategy.
# boundary="repeat":   split the single MusicXML part on Repeat-barline spans.
# boundary="per_file": the source path is a DIRECTORY of per-piece .mid files;
#                      each file is one primitive (Mutopia per-piece MIDI).
_SOURCE_CONFIGS: dict[str, dict] = {
    "hanon": {
        "title_prefix": "Hanon Exercise",
        "id_prefix": "hanon",
        "boundary": "repeat",
    },
    "bach": {
        "title_prefix": "Bach Invention",
        "id_prefix": "bach",
        "boundary": "per_file",
    },
    "czerny": {
        "title_prefix": "Czerny Study (Op.840)",
        "id_prefix": "czerny",
        "boundary": "per_file",
    },
    "burgmuller": {
        "title_prefix": "Burgmuller Study (Op.100)",
        "id_prefix": "burgmuller",
        "boundary": "per_file",
    },
    "chopin": {
        "title_prefix": "Chopin Prelude (Op.28)",
        "id_prefix": "chopin",
        "boundary": "per_file",
    },
    "satie": {
        "title_prefix": "Satie",
        "id_prefix": "satie",
        "boundary": "per_file",
    },
    # --- #49 corpus expansion: KernScores craigsapp repos (kern -> MIDI via
    # verovio; all public domain, pre-1928 composers). Per-piece MIDI, so the
    # same boundary='per_file' shape as the Mutopia sources above. id_prefix is
    # source-distinct so ids never collide with the #17 sources (e.g.
    # chopin_mazurka_NNN vs the existing chopin_NNN Op.28 preludes).
    "beethoven_sonatas": {
        "title_prefix": "Beethoven Sonata Movement",
        "id_prefix": "beethoven_sonata",
        "boundary": "per_file",
    },
    "mozart_sonatas": {
        "title_prefix": "Mozart Sonata Movement",
        "id_prefix": "mozart_sonata",
        "boundary": "per_file",
    },
    "scarlatti_sonatas": {
        "title_prefix": "Scarlatti Sonata",
        "id_prefix": "scarlatti",
        "boundary": "per_file",
    },
    "haydn_sonatas": {
        "title_prefix": "Haydn Sonata Movement",
        "id_prefix": "haydn_sonata",
        "boundary": "per_file",
    },
    "joplin_rags": {
        "title_prefix": "Joplin Rag",
        "id_prefix": "joplin",
        "boundary": "per_file",
    },
    "chopin_mazurkas": {
        "title_prefix": "Chopin Mazurka",
        "id_prefix": "chopin_mazurka",
        "boundary": "per_file",
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


def _export_midi_primitive(
    midi_in: Path,
    source_name: str,
    idx: int,
    config: dict,
    output_midi_dir: Path,
) -> Primitive:
    """Copy a per-piece MIDI file verbatim into the MIDI output dir as one
    primitive and return its Primitive record.

    No MusicXML exists for these sources (Mutopia ships LilyPond + MIDI), so
    `musicxml_path` is set to the copied MIDI path. partitura is used only to
    count notes (the 0-note guard); the MIDI is copied, not re-emitted, so the
    embedding input stays byte-faithful to the source engraving.
    """
    score = partitura.load_score_midi(str(midi_in))
    n_notes = len(partitura.utils.music.ensure_notearray(score))
    if n_notes == 0:
        raise SegmentationError(
            f"{source_name}: file {midi_in.name} contains 0 notes"
        )

    primitive_id = f"{config['id_prefix']}_{idx:03d}"
    title = f"{config['title_prefix']} {idx}"
    mid_out = output_midi_dir / f"{primitive_id}.mid"
    shutil.copyfile(midi_in, mid_out)

    logger.info("Segmented %s (n_notes=%d) <- %s", primitive_id, n_notes, midi_in.name)
    return Primitive(
        primitive_id=primitive_id,
        source=source_name,
        source_exercise_number=idx,
        title=title,
        musicxml_path=mid_out,
        midi_path=mid_out,
        n_notes=n_notes,
    )


def segment_source(
    musicxml_path: Path,
    source_name: str,
    output_score_dir: Path,
    output_midi_dir: Path,
) -> list[Primitive]:
    """Segment a source into individual Primitive instances.

    Segmentation strategy is per-source (see module docstring):
    - hanon: split the single MusicXML part on Repeat-barline spans (one
      primitive per exercise).
    - per-file MIDI sources (bach, czerny, burgmuller, chopin, satie): the path
      is a DIRECTORY of per-piece .mid files; each file is one primitive,
      numbered by sorted filename.

    Exports per-primitive MIDI (and MusicXML for the repeat source) to the given
    output directories.

    Args:
        musicxml_path: for "hanon", path to the source MusicXML (.xml/.mxl); for
            per-file MIDI sources, path to a directory of .mid files.
        source_name: one of the keys of _SOURCE_CONFIGS.
        output_score_dir: directory to write per-primitive MusicXML files
            (used by the repeat source only).
        output_midi_dir: directory to write per-primitive MIDI files.

    Returns:
        List of Primitive dataclasses ordered by source_exercise_number ascending.

    Raises:
        SegmentationError: if source_name is unrecognized; if a per_file source
            directory contains no .mid files; if the score has zero parts; if a
            "repeat" source has no Repeat markers; or if a segment has zero notes.
        FileNotFoundError: if musicxml_path does not exist.
    """
    musicxml_path = Path(musicxml_path)
    if not musicxml_path.exists():
        raise FileNotFoundError(f"source path not found: {musicxml_path}")

    if source_name not in _SOURCE_CONFIGS:
        raise SegmentationError(
            f"unknown source {source_name!r}; supported: {sorted(_SOURCE_CONFIGS)}"
        )

    config = _SOURCE_CONFIGS[source_name]
    output_score_dir = Path(output_score_dir)
    output_midi_dir = Path(output_midi_dir)
    output_score_dir.mkdir(parents=True, exist_ok=True)
    output_midi_dir.mkdir(parents=True, exist_ok=True)

    if config["boundary"] == "per_file":
        midi_files = sorted(musicxml_path.glob("*.mid"))
        if not midi_files:
            raise SegmentationError(
                f"{source_name}: boundary='per_file' but no .mid files found "
                f"in directory {musicxml_path}"
            )
        return [
            _export_midi_primitive(
                midi_in, source_name, idx, config, output_midi_dir
            )
            for idx, midi_in in enumerate(midi_files, start=1)
        ]

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
    else:
        raise SegmentationError(
            f"{source_name}: unsupported boundary {config['boundary']!r}"
        )

    return primitives
