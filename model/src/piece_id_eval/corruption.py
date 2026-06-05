# model/src/piece_id_eval/corruption.py
"""Synthetic note degradation for corruption-ablation experiments.

Three independent corruption modes applied in order:
1. Deletion: each note dropped independently with probability deletion_rate.
2. Jitter: onset/offset shifted by Uniform(-jitter_seconds, +jitter_seconds).
3. Insertion: for each surviving note, with probability insertion_rate, insert
   a random note nearby (pitch = original_pitch + Uniform(-12, 12), clamped
   to MIDI range 21-108; onset within +/-1.0s of the original).

Output is sorted ascending by onset.
"""
from __future__ import annotations

import random

from piece_id_eval.notes import Note

_MIDI_MIN = 21
_MIDI_MAX = 108


def corrupt_notes(
    notes: list[Note],
    deletion_rate: float,
    insertion_rate: float,
    jitter_seconds: float,
    seed: int,
) -> list[Note]:
    """Apply deletion, jitter, and insertion corruption to notes.

    Args:
        notes: input note list (not modified in place).
        deletion_rate: probability in [0, 1] each note is dropped.
        insertion_rate: probability in [0, 1] a spurious note is inserted
            adjacent to each surviving note.
        jitter_seconds: max absolute onset/offset shift in seconds (Uniform).
        seed: RNG seed for full determinism.

    Returns:
        Corrupted list of Note, sorted ascending by onset.
    """
    rng = random.Random(seed)
    result: list[Note] = []

    for n in notes:
        # Deletion
        if deletion_rate > 0.0 and rng.random() < deletion_rate:
            continue

        # Jitter
        if jitter_seconds > 0.0:
            shift = rng.uniform(-jitter_seconds, jitter_seconds)
            onset = max(0.0, n.onset + shift)
            offset = max(onset + 0.01, n.offset + shift)
            n = Note(onset=onset, offset=offset, pitch=n.pitch, velocity=n.velocity)

        result.append(n)

        # Insertion
        if insertion_rate > 0.0 and rng.random() < insertion_rate:
            pitch_shift = rng.randint(-12, 12)
            new_pitch = max(_MIDI_MIN, min(_MIDI_MAX, n.pitch + pitch_shift))
            onset_shift = rng.uniform(-1.0, 1.0)
            new_onset = max(0.0, n.onset + onset_shift)
            new_offset = new_onset + rng.uniform(0.05, 0.3)
            new_velocity = max(1, min(127, n.velocity + rng.randint(-20, 20)))
            result.append(Note(onset=new_onset, offset=new_offset, pitch=new_pitch, velocity=new_velocity))

    result.sort(key=lambda n: n.onset)
    return result
