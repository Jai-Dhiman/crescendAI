"""MIDI corruption primitives for practice-distribution augmentation.

Transforms professional MAESTRO MIDI into plausible student-practice MIDI via
five primitives. Each returns a deep-copied PrettyMIDI so the original is
never mutated. Compose with `apply_practice_corruptions` for a random subset.

Primitive summary:
    drop_notes           -- note omissions (student misses a key)
    substitute_wrong_notes -- pitch substitutions ±1-3 semitones
    jitter_tempo         -- smooth cumulative timing drift
    compress_velocity    -- dynamic range compression toward mf
    insert_pauses        -- stop-and-restart hesitations
"""

from __future__ import annotations

import copy
import random
from typing import Optional

import numpy as np
import pretty_midi


# ---------------------------------------------------------------------------
# Core primitives
# ---------------------------------------------------------------------------


def drop_notes(
    midi: pretty_midi.PrettyMIDI,
    rate: float = 0.05,
    rng: Optional[random.Random] = None,
) -> pretty_midi.PrettyMIDI:
    """Randomly drop notes to simulate a student missing keys.

    Args:
        midi: Source MIDI object.
        rate: Probability of dropping any individual note.
        rng: Optional seeded random.Random instance for reproducibility.

    Returns:
        New PrettyMIDI with a random ~rate fraction of notes removed.
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"rate must be in [0, 1], got {rate}")

    _rng = rng or random.Random()
    result = copy.deepcopy(midi)

    for instrument in result.instruments:
        instrument.notes = [
            n for n in instrument.notes if _rng.random() >= rate
        ]

    return result


def substitute_wrong_notes(
    midi: pretty_midi.PrettyMIDI,
    rate: float = 0.03,
    semitone_range: int = 3,
    rng: Optional[random.Random] = None,
) -> pretty_midi.PrettyMIDI:
    """Shift a random fraction of notes by ±1–semitone_range semitones.

    Simulates a student playing wrong notes — neighboring keys, not random
    pitches. Pitch is clamped to [0, 127] (valid MIDI range).

    Args:
        midi: Source MIDI object.
        rate: Probability of substituting any individual note.
        semitone_range: Maximum absolute pitch shift in semitones.
        rng: Optional seeded random.Random instance.

    Returns:
        New PrettyMIDI with random note pitches shifted.
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"rate must be in [0, 1], got {rate}")
    if semitone_range < 1:
        raise ValueError(f"semitone_range must be >= 1, got {semitone_range}")

    _rng = rng or random.Random()
    result = copy.deepcopy(midi)

    offsets = list(range(-semitone_range, 0)) + list(range(1, semitone_range + 1))

    for instrument in result.instruments:
        for note in instrument.notes:
            if _rng.random() < rate:
                shift = _rng.choice(offsets)
                note.pitch = int(np.clip(note.pitch + shift, 0, 127))

    return result


def jitter_tempo(
    midi: pretty_midi.PrettyMIDI,
    std: float = 0.05,
    control_interval: float = 0.5,
    rng: Optional[random.Random] = None,
) -> pretty_midi.PrettyMIDI:
    """Apply smooth cumulative tempo drift to simulate timing instabilities.

    Samples random acceleration/deceleration impulses at `control_interval`
    second boundaries, integrates them into a cumulative time warp, then
    applies the warp to all note onsets/offsets. Notes within the same chord
    maintain their relative timing so the result is musically coherent.

    The warp is monotonically increasing by construction: drift is clipped so
    the warped time never decreases or exceeds 3x the original duration.

    Args:
        midi: Source MIDI object.
        std: Standard deviation of the per-control-point timing jitter
            as a fraction of control_interval.
        control_interval: Spacing between control points in seconds.
        rng: Optional seeded random.Random instance.

    Returns:
        New PrettyMIDI with all note times warped by the jitter curve.
    """
    if std < 0.0:
        raise ValueError(f"std must be >= 0, got {std}")

    _rng = rng or random.Random()
    result = copy.deepcopy(midi)

    total_duration = midi.get_end_time()
    if total_duration <= 0.0:
        return result

    # Build original control points
    n_points = max(2, int(np.ceil(total_duration / control_interval)) + 1)
    t_orig = np.linspace(0.0, total_duration, n_points)

    # Sample jitter deltas (Gaussian) and integrate into cumulative offset
    np_rng = np.random.RandomState(_rng.randint(0, 2**31 - 1))
    deltas = np_rng.randn(n_points) * std * control_interval
    deltas[0] = 0.0  # anchor: first control point stays at 0

    cum_drift = np.cumsum(deltas)

    # Clamp so warped times are monotonically increasing and stay in [0, 3*total]
    t_warped = t_orig + cum_drift
    t_warped = np.clip(t_warped, 0.0, total_duration * 3.0)

    # Enforce monotonicity via cumulative maximum
    t_warped = np.maximum.accumulate(t_warped)

    # Warp function via linear interpolation
    def _warp(t: float) -> float:
        return float(np.interp(t, t_orig, t_warped))

    for instrument in result.instruments:
        for note in instrument.notes:
            note.start = _warp(note.start)
            note.end = max(note.start + 0.01, _warp(note.end))

    # Warp control changes and pitch bends too
    for instrument in result.instruments:
        for cc in instrument.control_changes:
            cc.time = _warp(cc.time)
        for pb in instrument.pitch_bends:
            pb.time = _warp(pb.time)

    return result


def compress_velocity(
    midi: pretty_midi.PrettyMIDI,
    factor: float = 0.4,
    midpoint: int = 64,
    rng: Optional[random.Random] = None,
) -> pretty_midi.PrettyMIDI:
    """Compress dynamic range toward mezzo-forte to simulate a less expressive player.

    Applies: v_new = midpoint + (v_orig - midpoint) * factor

    A factor of 0 collapses everything to midpoint; 1.0 leaves velocities
    unchanged. Values < 0 invert dynamics (not a useful practice simulation,
    but valid).

    Args:
        midi: Source MIDI object.
        factor: Compression factor in [0, 1].
        midpoint: Target velocity (1-127) to compress toward.
        rng: Unused; kept for API symmetry with other primitives.

    Returns:
        New PrettyMIDI with compressed velocity values clamped to [1, 127].
    """
    if not 0.0 <= factor <= 1.0:
        raise ValueError(f"factor must be in [0, 1], got {factor}")
    if not 1 <= midpoint <= 127:
        raise ValueError(f"midpoint must be in [1, 127], got {midpoint}")

    del rng  # compression is deterministic; parameter kept for API symmetry
    result = copy.deepcopy(midi)

    for instrument in result.instruments:
        for note in instrument.notes:
            compressed = midpoint + (note.velocity - midpoint) * factor
            note.velocity = int(np.clip(round(compressed), 1, 127))

    return result


def insert_pauses(
    midi: pretty_midi.PrettyMIDI,
    rate: float = 0.02,
    max_dur: float = 0.8,
    min_gap: float = 0.3,
    rng: Optional[random.Random] = None,
) -> pretty_midi.PrettyMIDI:
    """Insert stop-and-restart pauses at phrase-boundary-like positions.

    Finds positions in the MIDI where all instruments are simultaneously
    silent for at least `min_gap` seconds, then extends a random subset of
    those silences by a random amount in (0, max_dur].

    If no natural silences are found, inserts pauses at random positions
    between consecutive notes (fallback).

    Args:
        midi: Source MIDI object.
        rate: Probability of extending any discovered silence.
        max_dur: Maximum extra pause duration in seconds.
        min_gap: Minimum existing silence to qualify as a phrase boundary.
        rng: Optional seeded random.Random instance.

    Returns:
        New PrettyMIDI with extended silences at a random subset of positions.
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"rate must be in [0, 1], got {rate}")
    if max_dur <= 0.0:
        raise ValueError(f"max_dur must be > 0, got {max_dur}")

    _rng = rng or random.Random()
    result = copy.deepcopy(midi)

    total_duration = midi.get_end_time()
    if total_duration <= 0.0:
        return result

    # Collect all note events across all instruments
    all_events: list[tuple[float, float]] = []  # (onset, offset)
    for instrument in result.instruments:
        for note in instrument.notes:
            all_events.append((note.start, note.end))

    if not all_events:
        return result

    all_events.sort()

    # Find silence gaps between consecutive note-end and next note-start
    gap_positions: list[tuple[float, float]] = []  # (gap_start, gap_end)
    prev_end = 0.0
    for onset, offset in all_events:
        if onset - prev_end >= min_gap:
            gap_positions.append((prev_end, onset))
        prev_end = max(prev_end, offset)

    # Fallback: if no natural gaps, sample random insertion points
    if not gap_positions:
        n_fallback = max(1, int(len(all_events) * rate * 5))
        for _ in range(n_fallback):
            t = _rng.uniform(0.0, total_duration * 0.9)
            gap_positions.append((t, t))

    # For each gap, optionally extend it
    insertions: list[tuple[float, float]] = []  # (position, extra_duration)
    for _gap_start, gap_end in gap_positions:
        if _rng.random() < rate:
            extra = _rng.uniform(0.0, max_dur)
            insertions.append((gap_end, extra))

    if not insertions:
        return result

    # Sort by position descending so each insertion doesn't shift earlier ones
    insertions.sort(key=lambda x: x[0], reverse=True)

    for pos, extra in insertions:
        for instrument in result.instruments:
            for note in instrument.notes:
                if note.start >= pos:
                    note.start += extra
                    note.end += extra
        for instrument in result.instruments:
            for cc in instrument.control_changes:
                if cc.time >= pos:
                    cc.time += extra
            for pb in instrument.pitch_bends:
                if pb.time >= pos:
                    pb.time += extra

    return result


# ---------------------------------------------------------------------------
# Composite entry point
# ---------------------------------------------------------------------------


def apply_practice_corruptions(
    midi: pretty_midi.PrettyMIDI,
    rng: Optional[random.Random] = None,
    drop_rate: float = 0.05,
    sub_rate: float = 0.03,
    tempo_std: float = 0.05,
    velocity_factor: float = 0.4,
    pause_rate: float = 0.02,
    max_pause: float = 0.8,
    min_primitives: int = 2,
    max_primitives: int = 5,
) -> pretty_midi.PrettyMIDI:
    """Apply a random subset of MIDI corruption primitives.

    Randomly selects between min_primitives and max_primitives of the five
    primitives, applies them in a fixed order (drop → substitute → tempo →
    velocity → pause) so that the subset is consistent regardless of which
    primitives were sampled.

    Args:
        midi: Source MIDI object.
        rng: Optional seeded random.Random instance.
        drop_rate: Passed to drop_notes.
        sub_rate: Passed to substitute_wrong_notes.
        tempo_std: Passed to jitter_tempo.
        velocity_factor: Passed to compress_velocity.
        pause_rate: Passed to insert_pauses.
        max_pause: Passed to insert_pauses.
        min_primitives: Minimum number of primitives to apply (inclusive).
        max_primitives: Maximum number of primitives to apply (inclusive).

    Returns:
        Corrupted PrettyMIDI object.
    """
    _rng = rng or random.Random()

    all_primitives = [
        ("drop",       lambda m: drop_notes(m, rate=drop_rate, rng=_rng)),
        ("substitute", lambda m: substitute_wrong_notes(m, rate=sub_rate, rng=_rng)),
        ("tempo",      lambda m: jitter_tempo(m, std=tempo_std, rng=_rng)),
        ("velocity",   lambda m: compress_velocity(m, factor=velocity_factor, rng=_rng)),
        ("pause",      lambda m: insert_pauses(m, rate=pause_rate, max_dur=max_pause, rng=_rng)),
    ]

    n_to_apply = _rng.randint(min_primitives, max_primitives)
    chosen_names = set(_rng.sample([name for name, _ in all_primitives], n_to_apply))

    result = midi
    for name, fn in all_primitives:
        if name in chosen_names:
            result = fn(result)

    return result
