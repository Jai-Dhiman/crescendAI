"""#167 (#104 S2): turn a PSyllabus MIDI into the kind of audio MIREX will
actually hand us.

The contract says *"each test piece is represented by at least two recordings
-- one human performance and one synthesized score rendering from a fixed
soundfont -- scored independently."* Half the test audio is therefore synthetic,
and our pipeline transcribes first, having only ever been fed real recordings.
This module builds the synthetic half so that shift can be measured instead of
feared.

## `mid.zip` is a transcription, not a score

PSyllabus ships a MIDI for all 7,901 pieces, and it is tempting to read them as
score renderings. They are not. Sampling 33 of them against the label JSON's
`duration` field gives a median relative difference of 0.0002, a median of 64
distinct velocities, CC64 pedal, and off-grid onsets: they are expressive
transcriptions of the same YouTube performances the audio comes from.

So rendering them as-is reproduces the **timbre** shift and nothing else, while
MIREX's "score rendering" also carries a **performance** shift -- rigid timing,
flat dynamics, markings-only pedal. `deadpan` exists to separate those two,
because they have different fixes: a multimodal audio branch (#170) can rescue
timbre damage and cannot rescue deadpan damage, which needs render-augmented
training instead.

## What `deadpan` does and does not remove

It flattens velocity and drops the sustain pedal. **It does not quantize
timing**, and that limitation is deliberate rather than an omission. Deadpan
timing needs a metrical grid; a transcription has none. These files carry a
nominal constant 120bpm whose beats correspond to nothing in the audio (sampled
onsets from one: 0.4336, 1.3828, 1.4167, 1.4258), so quantizing to it would
fabricate structure rather than remove it. **A deadpan-vs-expressive delta
measured through this module is a lower bound on the real shift.**

Dropping the pedal still reaches the encoder even though MoonBeam's 6-dim
compound tokens carry no pedal channel and `write_notes_midi` drops pedals
anyway. The route is indirect: pedal changes what fluidsynth *sounds*, which
changes where Transkun places note offsets, which changes the durations that do
reach the encoder. Same for velocity -- a uniformly-struck render is
re-transcribed with uniform velocity estimates.
"""
from __future__ import annotations

import io
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

# 44.1 kHz because the contract specifies it for the input WAV, and a rate
# mismatch would be measuring resampling rather than synthesis.
SAMPLE_RATE = 44100

# fluidsynth's default gain (0.2) renders quietly enough that soft passages sit
# near the noise floor after 16-bit quantization; 0.5 is the conventional
# headroom-preserving value. Clipping is checked for by the caller, not assumed
# away here.
GAIN = 0.5

DEADPAN_VELOCITY = 64

SUSTAIN_CC = 64


@dataclass(frozen=True)
class Performance:
    """Notes plus sustain-pedal events. `psyllabus.notes_from_midi_bytes`
    already reads note dicts and is reused everywhere else, but it discards
    control changes -- correct for its callers, wrong here, since the pedal is
    exactly one of the things the expressive and deadpan arms differ in."""

    notes: list[dict]
    pedals: list[dict]


def read_performance(raw: bytes) -> Performance:
    """Notes (`{pitch,onset,offset,velocity}`, onset-ordered) and sustain-pedal
    events (`{time,value}`, time-ordered) from raw MIDI bytes."""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(io.BytesIO(raw))
    notes, pedals = [], []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            notes.append({"pitch": int(n.pitch), "onset": float(n.start),
                          "offset": float(n.end), "velocity": int(n.velocity)})
        for cc in inst.control_changes:
            if cc.number == SUSTAIN_CC:
                pedals.append({"time": float(cc.time), "value": int(cc.value)})
    notes.sort(key=lambda d: d["onset"])
    pedals.sort(key=lambda d: d["time"])
    return Performance(notes=notes, pedals=pedals)


def deadpan(perf: Performance, velocity: int = DEADPAN_VELOCITY) -> Performance:
    """Flatten every velocity and drop the sustain pedal, preserving timing.

    The proxy for MIREX's "synthesized score rendering". See the module
    docstring for why timing survives and why that makes any delta measured
    against this a lower bound.
    """
    if not 1 <= velocity <= 127:
        raise ValueError(
            f"velocity must be a valid MIDI velocity in 1..127, got {velocity} "
            f"-- 0 is a note-off, which would render silence")
    return Performance(
        notes=[{**n, "velocity": velocity} for n in perf.notes],
        pedals=[],
    )


def write_performance_midi(perf: Performance, path: Path) -> None:
    """One-instrument piano MIDI carrying notes and sustain pedal.

    Distinct from `audio_emb_extract.write_notes_midi`, which drops pedals
    because they cannot reach MoonBeam. This file is going to fluidsynth, where
    the pedal is audible and therefore load-bearing.
    """
    import pretty_midi

    if not perf.notes:
        raise ValueError(
            f"refusing to write {path} from a performance with no notes -- "
            f"fluidsynth would emit a valid but silent WAV, and a silent WAV "
            f"fails later as an unexplained empty transcription")

    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    for note in perf.notes:
        instrument.notes.append(pretty_midi.Note(
            velocity=int(note["velocity"]), pitch=int(note["pitch"]),
            start=float(note["onset"]), end=float(note["offset"])))
    for pedal in perf.pedals:
        instrument.control_changes.append(pretty_midi.ControlChange(
            number=SUSTAIN_CC, value=int(pedal["value"]),
            time=float(pedal["time"])))
    pm.instruments.append(instrument)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(path))


def render_wav(midi_path: Path, soundfont: Path, out_wav: Path,
               sample_rate: int = SAMPLE_RATE, gain: float = GAIN,
               run=subprocess.run) -> Path:
    """Render a MIDI to a 44.1 kHz WAV through fluidsynth and a fixed soundfont.

    Raises on a missing binary, a missing soundfont, a non-zero exit, or an
    empty output. This is the research harness, so it fails loudly per
    `model/CLAUDE.md`; the container's inverted failure policy stops at
    `score_wav_or_fallback` and must not reach here.
    """
    midi_path, soundfont, out_wav = Path(midi_path), Path(soundfont), Path(out_wav)
    binary = shutil.which("fluidsynth")
    if binary is None:
        raise FileNotFoundError(
            "fluidsynth is not on PATH; `brew install fluidsynth`. The "
            "synthesized-audio arm cannot be silently skipped -- an unrendered "
            "arm would report as a missing population, not a passing one")
    if not soundfont.is_file():
        raise FileNotFoundError(f"soundfont not found: {soundfont}")
    if not midi_path.is_file():
        raise FileNotFoundError(f"midi not found: {midi_path}")

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    # -ni: no interactive shell, no MIDI input device. -F: render to file
    # (offline, faster than realtime and deterministic).
    result = run(
        [binary, "-ni", "-F", str(out_wav), "-r", str(sample_rate),
         "-g", str(gain), str(soundfont), str(midi_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"fluidsynth failed on {midi_path} (exit {result.returncode}): "
            f"{(result.stderr or '').strip()[:400]}")
    if not out_wav.is_file() or out_wav.stat().st_size == 0:
        raise RuntimeError(
            f"fluidsynth exited 0 but wrote no audio to {out_wav} for "
            f"{midi_path}")
    return out_wav
