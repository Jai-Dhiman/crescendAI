# model/src/piece_id_eval/windowing.py
"""Arbitrary-start subsequence window sampler.

Given a list of Notes and a window length in seconds, samples n_starts
uniformly random start offsets and returns subsets of notes falling within
each window. Deterministic via seed.

window_seconds=None -> single full-piece window (no start sampling).
If the recording duration < window_seconds, falls back to the full recording
as a single window.
"""
from __future__ import annotations

import random

from piece_id_eval.notes import Note


def sample_windows(
    notes: list[Note],
    window_seconds: float | None,
    n_starts: int,
    seed: int,
) -> list[list[Note]]:
    """Sample up to n_starts windows of window_seconds duration from notes.

    Args:
        notes: sorted list of Note (ascending onset).
        window_seconds: window duration in seconds. None -> full piece.
        n_starts: number of random start offsets to sample.
        seed: RNG seed for determinism.

    Returns:
        List of note sublists, each covering [start, start + window_seconds).
        If window_seconds is None or recording is shorter than window_seconds,
        returns [notes] (the full recording as a single window).
    """
    if not notes:
        return [[]]

    if window_seconds is None:
        return [notes]

    recording_duration = notes[-1].onset - notes[0].onset
    if recording_duration <= window_seconds:
        return [notes]

    rng = random.Random(seed)
    first_onset = notes[0].onset
    max_start = notes[-1].onset - window_seconds

    windows: list[list[Note]] = []
    for _ in range(n_starts):
        start = rng.uniform(first_onset, max_start)
        end = start + window_seconds
        window = [n for n in notes if start <= n.onset < end]
        windows.append(window)
    return windows
