"""Score fingerprinting: N-gram index and rerank feature vectors.

128-dim feature vector layout (reference implementation for Rust Task 7):
  [0:12]   Pitch class histogram (normalized, 12 pitch classes C..B)
  [12:37]  Interval histogram (-12 to +12 semitones = 25 bins, normalized)
  [37:41]  Pitch stats (min, max, mean, std; each scaled to [0,1] over MIDI range 0-127)
  [41:66]  IOI histogram (25 bins at 50ms each, i.e. 0-50ms, 50-100ms, ..., up to 1200ms+; normalized)
  [66:78]  Velocity histogram (12 equal-width bins over 0-127, normalized)
  [78:82]  Velocity stats (min, max, mean, std; each scaled to [0,1] over 0-127)
  [82:128] Reserved / zero-padded
"""

from __future__ import annotations

import json
import math
from pathlib import Path


def extract_pitch_trigrams(pitches: list[int]) -> list[tuple]:
    """Extract consecutive pitch trigrams from a sequence of MIDI pitches.

    Args:
        pitches: List of MIDI pitch values (0-127).

    Returns:
        List of (p0, p1, p2) tuples. Empty if fewer than 3 pitches.
    """
    if len(pitches) < 3:
        return []
    return [(pitches[i], pitches[i + 1], pitches[i + 2]) for i in range(len(pitches) - 2)]


def compute_rerank_features(notes: list[dict]) -> list[float]:
    """Compute the 128-dim rerank feature vector for a list of note dicts.

    Each note dict must have:
        pitch: int (MIDI pitch 0-127)
        onset_seconds: float
        velocity: int (0-127)

    Returns:
        List of 128 floats following the layout documented at the module level.
    """
    features: list[float] = [0.0] * 128

    if not notes:
        return features

    pitches = [n["pitch"] for n in notes]
    velocities = [n["velocity"] for n in notes]
    onsets = [n["onset_seconds"] for n in notes]

    # [0:12] Pitch class histogram (normalized)
    pc_counts = [0] * 12
    for p in pitches:
        pc_counts[p % 12] += 1
    total = len(pitches)
    for i in range(12):
        features[i] = pc_counts[i] / total

    # [12:37] Interval histogram (-12 to +12 semitones, 25 bins, normalized)
    # Bin index = interval + 12 (so interval -12 -> bin 0, interval 0 -> bin 12, interval +12 -> bin 24)
    interval_counts = [0] * 25
    intervals = [pitches[i + 1] - pitches[i] for i in range(len(pitches) - 1)]
    for iv in intervals:
        bin_idx = max(0, min(24, iv + 12))
        interval_counts[bin_idx] += 1
    n_intervals = len(intervals)
    if n_intervals > 0:
        for i in range(25):
            features[12 + i] = interval_counts[i] / n_intervals

    # [37:41] Pitch stats (min, max, mean, std; scaled to [0,1] over MIDI range 0-127)
    p_min = min(pitches)
    p_max = max(pitches)
    p_mean = sum(pitches) / len(pitches)
    p_var = sum((p - p_mean) ** 2 for p in pitches) / len(pitches)
    p_std = math.sqrt(p_var)
    features[37] = p_min / 127.0
    features[38] = p_max / 127.0
    features[39] = p_mean / 127.0
    features[40] = p_std / 127.0  # std can theoretically reach 63.5, but we keep scale consistent

    # [41:66] IOI histogram (25 bins, each 50ms wide; bins 0..23 cover 0-1200ms, bin 24 is overflow)
    ioi_counts = [0] * 25
    sorted_onsets = sorted(onsets)
    iois = [sorted_onsets[i + 1] - sorted_onsets[i] for i in range(len(sorted_onsets) - 1)]
    for ioi in iois:
        bin_idx = min(24, int(ioi / 0.05))
        ioi_counts[bin_idx] += 1
    n_iois = len(iois)
    if n_iois > 0:
        for i in range(25):
            features[41 + i] = ioi_counts[i] / n_iois

    # [66:78] Velocity histogram (12 equal-width bins over 0-127, normalized)
    vel_counts = [0] * 12
    for v in velocities:
        bin_idx = min(11, int(v / 128.0 * 12))
        vel_counts[bin_idx] += 1
    for i in range(12):
        features[66 + i] = vel_counts[i] / len(velocities)

    # [78:82] Velocity stats (min, max, mean, std; scaled to [0,1] over 0-127)
    v_min = min(velocities)
    v_max = max(velocities)
    v_mean = sum(velocities) / len(velocities)
    v_var = sum((v - v_mean) ** 2 for v in velocities) / len(velocities)
    v_std = math.sqrt(v_var)
    features[78] = v_min / 127.0
    features[79] = v_max / 127.0
    features[80] = v_mean / 127.0
    features[81] = v_std / 127.0

    # [82:128] Reserved / zero-padded (already initialized to 0.0)

    return features


def _collect_bar_pitches(score_data: dict) -> list[tuple[int, list[int]]]:
    """Return list of (bar_number, pitches) for all bars in a score."""
    result = []
    for bar in score_data.get("bars", []):
        pitches = [n["pitch"] for n in bar.get("notes", [])]
        if pitches:
            result.append((bar["bar_number"], pitches))
    return result


def _collect_all_notes(score_data: dict) -> list[dict]:
    """Return flat list of all note dicts across all bars."""
    notes = []
    for bar in score_data.get("bars", []):
        notes.extend(bar.get("notes", []))
    return notes


def build_ngram_index(scores_dir: Path, max_freq: int = 3) -> dict:
    """Build an inverted N-gram index over all scores.

    For each score JSON in scores_dir, extract consecutive pitch trigrams from
    each bar. Build an inverted index mapping trigram -> list of
    (piece_id, bar_number) locations.

    Trigrams that appear in more than max_freq locations are pruned: they are
    too common to be useful for piece identification and would inflate index size.
    The default max_freq=3 keeps the index under 5MB for the 242-piece library.

    Args:
        scores_dir: Directory containing score JSON files.
        max_freq: Maximum number of (piece_id, bar_number) entries per trigram.
            Trigrams with more entries are discarded. Default: 3.

    Returns:
        Dict mapping "p0,p1,p2" string keys to list of [piece_id, bar_number] pairs.
        Only trigrams with <= max_freq entries are included.
    """
    index: dict[str, list[list]] = {}
    json_files = sorted(
        f for f in scores_dir.glob("*.json") if f.name not in ("titles.json", "seed.sql")
    )

    for jf in json_files:
        with open(jf) as f:
            score_data = json.load(f)
        piece_id = score_data["piece_id"]

        for bar_number, pitches in _collect_bar_pitches(score_data):
            trigrams = extract_pitch_trigrams(pitches)
            for trigram in trigrams:
                key = f"{trigram[0]},{trigram[1]},{trigram[2]}"
                if key not in index:
                    index[key] = []
                index[key].append([piece_id, bar_number])

    # Prune overly common trigrams
    return {k: v for k, v in index.items() if len(v) <= max_freq}


def build_rerank_features(scores_dir: Path) -> dict[str, list[float]]:
    """Compute 128-dim rerank feature vectors for all scores.

    Args:
        scores_dir: Directory containing score JSON files.

    Returns:
        Dict mapping piece_id -> 128-dim float list.
    """
    result: dict[str, list[float]] = {}
    json_files = sorted(
        f for f in scores_dir.glob("*.json") if f.name not in ("titles.json", "seed.sql")
    )

    for jf in json_files:
        with open(jf) as f:
            score_data = json.load(f)
        piece_id = score_data["piece_id"]
        notes = _collect_all_notes(score_data)
        result[piece_id] = compute_rerank_features(notes)

    return result
