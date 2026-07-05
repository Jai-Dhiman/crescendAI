"""Deterministic difficulty features from AMT/MIDI note events (MIREX 2026, #104).

Four features, all computed from ONLY pitch and onset -- the signals aria-amt
recovers reliably (0.97 note F1). Note *durations/offsets* are excluded on purpose:
AMT offsets are fragile (offset F1 0.85; duration ratio up to 3.2x on pedaled
textures), so a difficulty feature built on them would not survive the MIREX
anti-cheat design (human vs synth rendering scored independently). See
docs/competitions/research-learnings.md rows A5/A6.

Features:
- pitch_entropy         Shannon entropy (bits) of the MIDI-pitch histogram.
- pitch_lz_complexity   Lempel-Ziv (LZ76) production complexity of the onset-ordered
                        pitch sequence -- an integer count of distinct phrases.
- polyphony_per_onset   Mean notes per onset cluster (chord density). Onset-only:
                        avoids offsets, unlike a true concurrent-voice count.
- pitch_range           max - min MIDI pitch (semitones).

A note is a dict {"pitch": int, "onset": float, "offset": float, "velocity": int},
matching the amt_fidelity schema. Every function raises loudly on empty input
rather than returning a silent sentinel.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

Note = dict


def pitch_entropy(pitches: Sequence[int]) -> float:
    """Shannon entropy (base-2, bits) of the MIDI-pitch histogram.

    A single repeated pitch -> 0; k equiprobable pitches -> log2(k). Captures
    pitch-content diversity independent of order or rhythm.
    """
    p = np.asarray(pitches, dtype=np.int64)
    if p.size < 1:
        raise ValueError("need >=1 note for pitch entropy")
    _, counts = np.unique(p, return_counts=True)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log2(probs)))


def pitch_lz_complexity(sequence: Sequence[int]) -> int:
    """Lempel-Ziv (LZ76) production complexity of a symbol sequence.

    The classic Kaspar-Schuster parse: scan left to right, counting the number of
    distinct phrases needed to reconstruct the string from its own prefix history.
    Returns the raw integer phrase count (higher = less compressible = harder pattern).
    Examples: [] -> 0; [a] -> 1; [a,a,a,a] -> 2 (a|aaa); all-distinct length-n -> n.

    Deliberately un-normalized so tiny inputs have exact, hand-checkable values; any
    normalization (e.g. n/log2 n) is an asymptotic transform applied downstream.
    """
    s = list(sequence)
    n = len(s)
    if n <= 1:
        return n   # [] -> 0 phrases, [a] -> 1 phrase
    i = 0          # start of the current comparison prefix
    l = 1          # start of the substring being parsed
    k = 1          # current match length
    k_max = 1      # longest match found from any i for this substring
    c = 1          # phrase count
    while True:
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            if k > k_max:
                k_max = k
            i += 1
            if i == l:            # exhausted all prefixes -> new phrase
                c += 1
                l += k_max
                if l + 1 > n:
                    break
                i = 0
                k = 1
                k_max = 1
            else:
                k = 1
    return int(c)


def polyphony_per_onset(onsets: Sequence[float], onset_tol_s: float = 0.03) -> float:
    """Mean number of notes per onset cluster (chord density).

    Notes whose onsets fall within `onset_tol_s` of the cluster's first onset are
    treated as struck together. Monophonic melody -> 1.0; a single n-note chord -> n.
    Onset-only by design (no offsets), so it survives AMT's unreliable durations.
    The default tolerance (30 ms) matches AMT onset noise so near-simultaneous
    transcribed onsets still collapse into one chord.
    """
    o = np.sort(np.asarray(onsets, dtype=np.float64))
    if o.size < 1:
        raise ValueError("need >=1 note for polyphony")
    cluster_sizes = []
    anchor = o[0]
    size = 1
    for t in o[1:]:
        if t - anchor <= onset_tol_s:
            size += 1
        else:
            cluster_sizes.append(size)
            anchor = t
            size = 1
    cluster_sizes.append(size)
    return float(np.mean(cluster_sizes))


def pitch_range(pitches: Sequence[int]) -> int:
    """Span between the highest and lowest MIDI pitch (semitones)."""
    p = np.asarray(pitches, dtype=np.int64)
    if p.size < 1:
        raise ValueError("need >=1 note for pitch range")
    return int(p.max() - p.min())


def notes_to_pitch_sequence(notes: Sequence[Note]) -> list[int]:
    """Deterministic pitch sequence for LZ: sort by (onset, pitch), take pitches.

    Ordering chords low->high keeps the sequence reproducible across the oracle,
    AMT-on-real, and AMT-on-synth passes (aria-amt does not guarantee note order).
    """
    ordered = sorted(notes, key=lambda note: (note["onset"], note["pitch"]))
    return [int(note["pitch"]) for note in ordered]


def extract_difficulty_features(notes: Sequence[Note]) -> dict[str, float]:
    """All four difficulty features from a list of note dicts.

    The Phase-1 entry point: called three ways (oracle MIDI, AMT-on-real-audio,
    AMT-on-synth-render) to measure cross-render stability.
    """
    if len(notes) < 1:
        raise ValueError("need >=1 note to extract difficulty features")
    pitches = [int(note["pitch"]) for note in notes]
    onsets = [float(note["onset"]) for note in notes]
    return {
        "pitch_entropy": pitch_entropy(pitches),
        "pitch_lz_complexity": float(pitch_lz_complexity(notes_to_pitch_sequence(notes))),
        "polyphony": polyphony_per_onset(onsets),
        "pitch_range": float(pitch_range(pitches)),
    }
