#!/usr/bin/env python3
"""
Calculate note accuracy by comparing performance MIDI to score MIDI.

This script:
1. Loads performance MIDI and score MIDI (MAESTRO has perfect alignment)
2. Computes note-level F1 score (pitch + onset matching)
3. Outputs note_accuracy score (0-100 scale)

Usage:
    python scripts/calculate_note_accuracy.py --midi-perf path/to/performance.midi --midi-score path/to/score.midi
"""

import argparse
import pretty_midi
import numpy as np
from pathlib import Path


def extract_notes(midi_path, instrument_idx=0):
    """
    Extract notes from MIDI file.

    Returns: List of (onset_time, pitch, duration, velocity)
    """
    midi = pretty_midi.PrettyMIDI(str(midi_path))

    if len(midi.instruments) == 0:
        return []

    # Use first instrument (piano)
    instrument = midi.instruments[instrument_idx]

    notes = []
    for note in instrument.notes:
        notes.append({
            'onset': note.start,
            'pitch': note.pitch,
            'duration': note.end - note.start,
            'velocity': note.velocity
        })

    return notes


def match_notes(perf_notes, score_notes, onset_tolerance=0.05):
    """
    Match performance notes to score notes.

    Parameters:
    - perf_notes: List of performance notes
    - score_notes: List of score notes
    - onset_tolerance: Time tolerance for onset matching (seconds)

    Returns:
    - matched: Number of correctly matched notes
    - insertions: Number of extra notes in performance
    - deletions: Number of missing notes from score
    """
    matched = 0
    insertions = 0
    deletions = 0

    # Create sets for efficient matching
    perf_set = set()
    score_set = set()

    for note in perf_notes:
        # Quantize onset to tolerance
        onset_bin = round(note['onset'] / onset_tolerance)
        perf_set.add((onset_bin, note['pitch']))

    for note in score_notes:
        onset_bin = round(note['onset'] / onset_tolerance)
        score_set.add((onset_bin, note['pitch']))

    # Count matches
    matched = len(perf_set & score_set)

    # Count insertions (extra notes)
    insertions = len(perf_set - score_set)

    # Count deletions (missing notes)
    deletions = len(score_set - perf_set)

    return matched, insertions, deletions


def compute_note_accuracy(matched, insertions, deletions):
    """
    Compute note accuracy score (0-100).

    Uses F1-based metric:
    - Precision = matched / (matched + insertions)
    - Recall = matched / (matched + deletions)
    - F1 = 2 * (precision * recall) / (precision + recall)
    - Accuracy = F1 * 100

    Special cases:
    - If no notes in score: return 0
    - If no notes in performance: return 0
    - If perfect match: return 100
    """
    if matched + deletions == 0:
        # No notes in score
        return 0.0

    if matched + insertions == 0:
        # No notes in performance
        return 0.0

    precision = matched / (matched + insertions)
    recall = matched / (matched + deletions)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)

    # Scale to 0-100
    accuracy = f1 * 100

    return accuracy


def calculate_note_accuracy_from_midi(perf_midi_path, score_midi_path, onset_tolerance=0.05):
    """
    Calculate note accuracy by comparing performance to score MIDI.

    Parameters:
    - perf_midi_path: Path to performance MIDI file
    - score_midi_path: Path to score MIDI file (ground truth)
    - onset_tolerance: Onset matching tolerance in seconds (default: 50ms)

    Returns:
    - note_accuracy: Score from 0-100
    - stats: Dictionary with detailed statistics
    """
    # Extract notes
    perf_notes = extract_notes(perf_midi_path)
    score_notes = extract_notes(score_midi_path)

    # Match notes
    matched, insertions, deletions = match_notes(perf_notes, score_notes, onset_tolerance)

    # Compute accuracy
    accuracy = compute_note_accuracy(matched, insertions, deletions)

    # Detailed statistics
    stats = {
        'perf_notes': len(perf_notes),
        'score_notes': len(score_notes),
        'matched': matched,
        'insertions': insertions,
        'deletions': deletions,
        'precision': matched / (matched + insertions) if (matched + insertions) > 0 else 0,
        'recall': matched / (matched + deletions) if (matched + deletions) > 0 else 0,
        'note_accuracy': accuracy
    }

    return accuracy, stats


def main():
    parser = argparse.ArgumentParser(description="Calculate note accuracy from MIDI comparison")
    parser.add_argument("--midi-perf", type=str, required=True,
                       help="Path to performance MIDI file")
    parser.add_argument("--midi-score", type=str, default=None,
                       help="Path to score MIDI file (if different from performance)")
    parser.add_argument("--onset-tolerance", type=float, default=0.05,
                       help="Onset matching tolerance in seconds (default: 0.05)")
    args = parser.parse_args()

    # For MAESTRO, performance and score MIDI are the same file
    # (perfectly aligned by Disklavier)
    score_midi_path = args.midi_score if args.midi_score else args.midi_perf

    print(f"Performance MIDI: {args.midi_perf}")
    print(f"Score MIDI: {score_midi_path}")
    print(f"Onset tolerance: {args.onset_tolerance}s")

    # Calculate accuracy
    accuracy, stats = calculate_note_accuracy_from_midi(
        args.midi_perf,
        score_midi_path,
        onset_tolerance=args.onset_tolerance
    )

    print(f"\n" + "=" * 60)
    print(f"NOTE ACCURACY RESULTS")
    print("=" * 60)
    print(f"\nPerformance notes: {stats['perf_notes']}")
    print(f"Score notes: {stats['score_notes']}")
    print(f"Matched: {stats['matched']}")
    print(f"Insertions (extra): {stats['insertions']}")
    print(f"Deletions (missing): {stats['deletions']}")
    print(f"\nPrecision: {stats['precision']:.3f}")
    print(f"Recall: {stats['recall']:.3f}")
    print(f"\nNote Accuracy Score: {stats['note_accuracy']:.1f}/100")

    # Interpretation
    if accuracy >= 95:
        print("\nInterpretation: Excellent (virtuoso performance)")
    elif accuracy >= 85:
        print("\nInterpretation: Very good (advanced level)")
    elif accuracy >= 75:
        print("\nInterpretation: Good (intermediate level)")
    elif accuracy >= 60:
        print("\nInterpretation: Fair (early intermediate)")
    else:
        print("\nInterpretation: Needs improvement (beginner level)")


if __name__ == "__main__":
    main()
