"""Deterministic score-domain transforms over exercise primitives.

Three transforms, each producing a fresh, exportable partitura Part:
  - transpose:   shift every note by N semitones (key practice).
  - scale_tempo: scale the tempo marking (slow practice).
  - excerpt:     keep only a bar span, rebased to time 0 (segment loop).

"Deterministic" means same input -> identical output, no randomness, and every
transform is checkable in the symbolic domain (note count, pitch, bpm, bar span)
rather than by ear. Out-of-domain requests (pitch off the keyboard, non-positive
tempo, inverted/out-of-range bars) raise rather than silently clamping.
"""

from dataclasses import dataclass
from pathlib import Path

import partitura
import partitura.utils.music as pm
from partitura.score import Measure, Note, Part, Tempo, TimeSignature

# Standard 88-key piano range: A0 (21) .. C8 (108).
PIANO_MIDI_LOW = 21
PIANO_MIDI_HIGH = 108

DEFAULT_BPM = 120.0


@dataclass
class Variant:
    """A transformed primitive plus the transform that produced it."""

    part: Part
    transform: str  # "transpose" | "tempo" | "excerpt"
    params: dict
    source_primitive_id: str | None = None


def load_primitive(midi_path: Path) -> Part:
    """Load a primitive MIDI as a single partitura Part.

    Raises:
        FileNotFoundError: if midi_path does not exist.
        ValueError: if the score has no parts.
    """
    midi_path = Path(midi_path)
    if not midi_path.exists():
        raise FileNotFoundError(f"Primitive MIDI not found: {midi_path}")
    score = partitura.load_score(str(midi_path))
    parts = list(score.parts)
    if not parts:
        raise ValueError(f"No parts found in {midi_path}")
    return parts[0]


def _quarter(part: Part) -> int:
    return int(part.quarter_duration_map(0))


def _active_time_signature(part: Part) -> TimeSignature | None:
    ts_list = sorted(part.iter_all(TimeSignature), key=lambda x: x.start.t)
    return ts_list[0] if ts_list else None


def _base_bpm(part: Part) -> float:
    tempos = sorted(part.iter_all(Tempo), key=lambda x: x.start.t)
    return tempos[0].bpm if tempos else DEFAULT_BPM


def _build(
    quarter: int,
    notes: list[tuple[str, int, int | None, int, int, int]],
    ts: TimeSignature | None,
    bpm: float,
    part_id: str = "v",
    part_name: str = "variant",
) -> Part:
    """Build a fresh Part from (step, octave, alter, voice, start_t, end_t) note
    tuples, laying out measures so the result is valid, exportable MusicXML/MIDI.
    """
    out = Part(part_id, part_name, quarter_duration=quarter)
    if ts is not None:
        out.add(TimeSignature(ts.beats, ts.beat_type), 0)
    out.add(Tempo(bpm, unit="q"), 0)
    for step, octave, alter, voice, start_t, end_t in notes:
        out.add(
            Note(step=step, octave=octave, alter=alter, voice=voice),
            int(start_t),
            int(end_t),
        )
    partitura.score.add_measures(out)
    return out


def transpose(part: Part, semitones: int) -> Variant:
    """Shift every note by `semitones`, preserving timing, voice, and count.

    Raises:
        ValueError: if any resulting pitch falls outside the 88-key piano range.
    """
    notes = sorted(part.iter_all(Note), key=lambda n: (n.start.t, n.midi_pitch))
    rebuilt: list[tuple[str, int, int | None, int, int, int]] = []
    for n in notes:
        new_midi = n.midi_pitch + semitones
        if not (PIANO_MIDI_LOW <= new_midi <= PIANO_MIDI_HIGH):
            raise ValueError(
                f"transpose by {semitones} puts pitch {new_midi} outside piano "
                f"range [{PIANO_MIDI_LOW}, {PIANO_MIDI_HIGH}]"
            )
        step, alter, octave = pm.midi_pitch_to_pitch_spelling(new_midi)
        rebuilt.append((step, octave, alter, n.voice, n.start.t, n.end.t))
    out = _build(_quarter(part), rebuilt, _active_time_signature(part), _base_bpm(part))
    return Variant(part=out, transform="transpose", params={"semitones": semitones})


def scale_tempo(part: Part, factor: float) -> Variant:
    """Scale the tempo by `factor` (e.g. 0.5 = half speed for slow practice).

    Notes are untouched; only the Tempo marking changes.

    Raises:
        ValueError: if factor is not strictly positive.
    """
    if factor <= 0:
        raise ValueError(f"tempo factor must be > 0, got {factor}")
    new_bpm = _base_bpm(part) * factor
    notes = sorted(part.iter_all(Note), key=lambda n: (n.start.t, n.midi_pitch))
    rebuilt = [
        (n.step, n.octave, n.alter, n.voice, n.start.t, n.end.t) for n in notes
    ]
    out = _build(_quarter(part), rebuilt, _active_time_signature(part), new_bpm)
    return Variant(
        part=out,
        transform="tempo",
        params={"factor": factor, "bpm": new_bpm},
    )


def excerpt(part: Part, start_bar: int, end_bar: int) -> Variant:
    """Keep only measures [start_bar, end_bar] (inclusive), rebased to time 0.

    Bar numbers refer to partitura Measure.number. The kept span is loop material
    for a segment_loop exercise.

    Raises:
        ValueError: if start_bar > end_bar, the range is out of range, or the
            resulting span contains no notes.
    """
    if start_bar > end_bar:
        raise ValueError(
            f"start_bar ({start_bar}) must be <= end_bar ({end_bar})"
        )
    measures = {m.number: m for m in part.iter_all(Measure)}
    if start_bar not in measures or end_bar not in measures:
        raise ValueError(
            f"bar range [{start_bar}, {end_bar}] out of range; "
            f"available bars {min(measures)}..{max(measures)}"
        )
    span_start = measures[start_bar].start.t
    span_end = measures[end_bar].end.t

    notes = sorted(
        (n for n in part.iter_all(Note) if span_start <= n.start.t < span_end),
        key=lambda n: (n.start.t, n.midi_pitch),
    )
    if not notes:
        raise ValueError(
            f"bar range [{start_bar}, {end_bar}] contains no notes"
        )
    rebuilt = [
        (n.step, n.octave, n.alter, n.voice, n.start.t - span_start, n.end.t - span_start)
        for n in notes
    ]
    out = _build(_quarter(part), rebuilt, _active_time_signature(part), _base_bpm(part))
    return Variant(
        part=out,
        transform="excerpt",
        params={"start_bar": start_bar, "end_bar": end_bar},
    )


def materialize(variant: Variant, out_path: Path) -> Path:
    """Write a variant's Part to a MIDI file.

    Raises:
        ValueError: if the variant Part has no notes.
    """
    out_path = Path(out_path)
    if not any(True for _ in variant.part.iter_all(Note)):
        raise ValueError("Cannot materialize a variant with no notes")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partitura.save_score_midi(variant.part, str(out_path))
    return out_path
