#!/usr/bin/env python3
"""
Generate Pseudo-Labels for MAESTRO Dataset

Creates heuristic labels for Stage 2 pre-training using MIDI-audio alignment.

Pseudo-label heuristics:
- Note Accuracy: MIDI note density vs. audio onset detection
- Rhythmic Precision: Onset timing variance from grid
- Dynamics Control: Velocity variation (std dev)
- Articulation: Note duration consistency
- Pedaling: Sustain pedal usage patterns
- Tone Quality: Spectral centroid consistency

Generated annotations in JSONL format for training.
"""

import argparse
import csv
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import pretty_midi
from tqdm import tqdm

warnings.filterwarnings('ignore', category=FutureWarning)


def load_maestro_metadata(metadata_csv: Path) -> List[Dict]:
    """Load MAESTRO metadata from CSV."""
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_csv}")

    pieces = []
    with open(metadata_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pieces.append(row)

    return pieces


def filter_pieces(
    pieces: List[Dict],
    composers: Optional[List[str]] = None,
    min_duration: Optional[float] = None,
    max_duration: Optional[float] = None,
    years: Optional[List[int]] = None,
    piece_ids: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> List[Dict]:
    """
    Filter MAESTRO pieces based on criteria.

    Args:
        pieces: List of piece metadata dicts
        composers: List of composer names to include (e.g., ['Beethoven', 'Chopin'])
        min_duration: Minimum duration in seconds
        max_duration: Maximum duration in seconds
        years: List of recording years to include
        piece_ids: Specific piece IDs to include
        limit: Maximum number of pieces to return

    Returns:
        Filtered list of pieces
    """
    filtered = pieces

    if composers is not None:
        filtered = [p for p in filtered if any(c.lower() in p['canonical_composer'].lower() for c in composers)]

    if min_duration is not None:
        filtered = [p for p in filtered if float(p['duration']) >= min_duration]

    if max_duration is not None:
        filtered = [p for p in filtered if float(p['duration']) <= max_duration]

    if years is not None:
        filtered = [p for p in filtered if int(p['year']) in years]

    if piece_ids is not None:
        filtered = [p for p in filtered if p['piece_id'] in piece_ids]

    if limit is not None:
        filtered = filtered[:limit]

    return filtered


def compute_note_accuracy_score(midi: pretty_midi.PrettyMIDI, audio_path: Path, sr: int = 24000) -> float:
    """
    Pseudo-label for note accuracy based on MIDI density vs. audio onset detection.

    Higher score if MIDI notes match detected onsets in audio.
    """
    try:
        # Load audio
        y, _ = librosa.load(audio_path, sr=sr)

        # Detect onsets in audio
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)

        # Get MIDI note onsets
        midi_notes = []
        for instrument in midi.instruments:
            if not instrument.is_drum:
                midi_notes.extend(instrument.notes)

        if len(midi_notes) == 0:
            return 50.0

        midi_onset_times = np.array([note.start for note in midi_notes])

        # Match MIDI onsets to detected onsets (within 50ms tolerance)
        tolerance = 0.05
        matches = 0
        for midi_time in midi_onset_times:
            if np.any(np.abs(onset_times - midi_time) < tolerance):
                matches += 1

        # Score: percentage of MIDI notes that have detected onsets
        accuracy = (matches / len(midi_onset_times)) * 100

        # Clip to 40-95 range (avoid extremes for pseudo-labels)
        return np.clip(accuracy, 40, 95)

    except Exception as e:
        print(f"Warning: Failed to compute note accuracy: {e}")
        return 70.0  # Neutral default


def compute_rhythmic_precision_score(midi: pretty_midi.PrettyMIDI) -> float:
    """
    Pseudo-label for rhythmic precision based on timing grid alignment.

    Higher score if notes align to metrical grid.
    """
    try:
        # Extract all note onsets
        onsets = []
        for instrument in midi.instruments:
            if not instrument.is_drum:
                onsets.extend([note.start for note in instrument.notes])

        if len(onsets) == 0:
            return 50.0

        onsets = np.array(sorted(onsets))

        # Estimate tempo and beat times
        tempo = midi.estimate_tempo()
        beat_duration = 60.0 / tempo

        # Compute deviation from nearest beat
        deviations = []
        for onset in onsets:
            nearest_beat = round(onset / beat_duration) * beat_duration
            deviation = abs(onset - nearest_beat)
            deviations.append(deviation)

        # Score: inverse of mean deviation (normalized)
        mean_deviation = np.mean(deviations)

        # Convert to 0-100 scale (smaller deviation = higher score)
        # Assume 0ms deviation = 95, 100ms deviation = 50
        precision = 95 - (mean_deviation * 1000) * 0.45

        return np.clip(precision, 40, 95)

    except Exception as e:
        print(f"Warning: Failed to compute rhythmic precision: {e}")
        return 70.0


def compute_dynamics_control_score(midi: pretty_midi.PrettyMIDI) -> float:
    """
    Pseudo-label for dynamics control based on velocity variation.

    Higher score if velocity changes are smooth and controlled.
    """
    try:
        velocities = []
        for instrument in midi.instruments:
            if not instrument.is_drum:
                velocities.extend([note.velocity for note in instrument.notes])

        if len(velocities) == 0:
            return 50.0

        velocities = np.array(velocities)

        # Compute velocity statistics
        velocity_std = np.std(velocities)
        velocity_range = np.max(velocities) - np.min(velocities)

        # Good dynamics: moderate variation (not too flat, not too erratic)
        # Penalize both very low std (no dynamics) and very high std (erratic)
        ideal_std = 20.0
        std_penalty = abs(velocity_std - ideal_std) / 10.0

        # Score based on range and smoothness
        dynamics_score = 80 - std_penalty

        return np.clip(dynamics_score, 40, 95)

    except Exception as e:
        print(f"Warning: Failed to compute dynamics control: {e}")
        return 70.0


def compute_articulation_score(midi: pretty_midi.PrettyMIDI) -> float:
    """
    Pseudo-label for articulation based on note duration consistency.

    Higher score if note durations are appropriate and consistent.
    """
    try:
        durations = []
        for instrument in midi.instruments:
            if not instrument.is_drum:
                durations.extend([note.end - note.start for note in instrument.notes])

        if len(durations) == 0:
            return 50.0

        durations = np.array(durations)

        # Filter out very short/long notes (grace notes, fermatas)
        reasonable_durations = durations[(durations > 0.05) & (durations < 5.0)]

        if len(reasonable_durations) == 0:
            return 50.0

        # Score based on coefficient of variation (std/mean)
        cv = np.std(reasonable_durations) / np.mean(reasonable_durations)

        # Lower CV = more consistent = better articulation
        articulation_score = 90 - (cv * 30)

        return np.clip(articulation_score, 40, 95)

    except Exception as e:
        print(f"Warning: Failed to compute articulation: {e}")
        return 70.0


def compute_pedaling_score(midi: pretty_midi.PrettyMIDI) -> float:
    """
    Pseudo-label for pedaling based on sustain pedal usage patterns.

    Higher score if pedal changes are appropriate and musical.
    """
    try:
        pedal_changes = []
        for instrument in midi.instruments:
            if not instrument.is_drum:
                # Control change 64 is sustain pedal
                pedal_events = [cc for cc in instrument.control_changes if cc.number == 64]
                pedal_changes.extend(pedal_events)

        if len(pedal_changes) == 0:
            # No pedal data, use neutral score
            return 70.0

        # Compute pedal change rate
        duration = midi.get_end_time()
        pedal_rate = len(pedal_changes) / duration if duration > 0 else 0

        # Ideal rate: 1-3 changes per second (neither too sparse nor too frequent)
        ideal_rate = 2.0
        rate_penalty = abs(pedal_rate - ideal_rate) * 10

        pedaling_score = 85 - rate_penalty

        return np.clip(pedaling_score, 40, 95)

    except Exception as e:
        print(f"Warning: Failed to compute pedaling: {e}")
        return 70.0


def compute_tone_quality_score(audio_path: Path, sr: int = 24000) -> float:
    """
    Pseudo-label for tone quality based on spectral centroid consistency.

    Higher score if spectral characteristics are consistent and pleasant.
    """
    try:
        # Load audio
        y, _ = librosa.load(audio_path, sr=sr)

        # Compute spectral centroid
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

        # Score based on centroid consistency (std dev)
        centroid_std = np.std(centroid)

        # Lower std = more consistent tone
        tone_score = 85 - (centroid_std / 100)

        return np.clip(tone_score, 40, 95)

    except Exception as e:
        print(f"Warning: Failed to compute tone quality: {e}")
        return 70.0


def generate_pseudo_labels_for_piece(
    piece: Dict,
    maestro_root: Path,
    segment_duration: float = 20.0,
    segment_overlap: float = 5.0,
) -> List[Dict]:
    """
    Generate pseudo-labels for a single piece.

    Args:
        piece: Piece metadata dict
        maestro_root: Root directory of MAESTRO dataset
        segment_duration: Duration of each segment in seconds
        segment_overlap: Overlap between segments in seconds

    Returns:
        List of annotation dicts (one per segment)
    """
    audio_path = maestro_root / piece['audio_path']
    midi_path = maestro_root / piece['midi_path']

    if not audio_path.exists() or not midi_path.exists():
        raise FileNotFoundError(f"Missing files for {piece['piece_id']}")

    # Load MIDI
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    duration = float(piece['duration'])

    # Generate segments
    stride = segment_duration - segment_overlap
    num_segments = int((duration - segment_duration) / stride) + 1

    annotations = []

    for i in range(num_segments):
        start_time = i * stride
        end_time = min(start_time + segment_duration, duration)

        # Create segment annotation
        annotation = {
            'audio_path': str(audio_path),
            'midi_path': str(midi_path),
            'start_time': start_time,
            'end_time': end_time,
            'labels': {
                'note_accuracy': compute_note_accuracy_score(midi, audio_path),
                'rhythmic_precision': compute_rhythmic_precision_score(midi),
                'dynamics_control': compute_dynamics_control_score(midi),
                'articulation': compute_articulation_score(midi),
                'pedaling': compute_pedaling_score(midi),
                'tone_quality': compute_tone_quality_score(audio_path),
            },
            'metadata': {
                'piece_id': piece['piece_id'],
                'composer': piece['canonical_composer'],
                'title': piece['canonical_title'],
                'year': piece['year'],
                'segment_index': i,
                'is_pseudo_label': True,
            }
        }

        annotations.append(annotation)

    return annotations


def save_annotations(annotations: List[Dict], output_file: Path) -> None:
    """Save annotations to JSONL file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        for ann in annotations:
            f.write(json.dumps(ann) + '\n')

    print(f"Saved {len(annotations)} annotations to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Generate pseudo-labels for MAESTRO dataset')
    parser.add_argument('--maestro-root', type=str, default='data/maestro',
                        help='Root directory of MAESTRO dataset')
    parser.add_argument('--output', type=str, default='data/annotations/maestro_pseudo_labels.jsonl',
                        help='Output JSONL file for annotations')
    parser.add_argument('--metadata-csv', type=str, default=None,
                        help='Path to metadata.csv (default: maestro-root/metadata.csv)')

    # Filtering options
    parser.add_argument('--composers', nargs='+', default=None,
                        help='Filter by composers (e.g., --composers Beethoven Chopin)')
    parser.add_argument('--min-duration', type=float, default=300,
                        help='Minimum piece duration in seconds (default: 300 = 5 min)')
    parser.add_argument('--max-duration', type=float, default=900,
                        help='Maximum piece duration in seconds (default: 900 = 15 min)')
    parser.add_argument('--years', nargs='+', type=int, default=None,
                        help='Filter by recording years (e.g., --years 2004 2006 2008)')
    parser.add_argument('--piece-ids', nargs='+', default=None,
                        help='Specific piece IDs to process')
    parser.add_argument('--limit', type=int, default=50,
                        help='Maximum number of pieces to process (default: 50)')

    # Segmentation options
    parser.add_argument('--segment-duration', type=float, default=20.0,
                        help='Duration of each segment in seconds (default: 20)')
    parser.add_argument('--segment-overlap', type=float, default=5.0,
                        help='Overlap between segments in seconds (default: 5)')

    args = parser.parse_args()

    maestro_root = Path(args.maestro_root)

    # Load metadata
    if args.metadata_csv is None:
        metadata_csv = maestro_root / 'metadata.csv'
    else:
        metadata_csv = Path(args.metadata_csv)

    print(f"Loading metadata from {metadata_csv}")
    pieces = load_maestro_metadata(metadata_csv)
    print(f"Loaded {len(pieces)} pieces")

    # Filter pieces
    filtered_pieces = filter_pieces(
        pieces,
        composers=args.composers,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        years=args.years,
        piece_ids=args.piece_ids,
        limit=args.limit,
    )

    print(f"\nFiltered to {len(filtered_pieces)} pieces:")
    if args.composers:
        print(f"  Composers: {', '.join(args.composers)}")
    print(f"  Duration: {args.min_duration}-{args.max_duration}s")
    if args.years:
        print(f"  Years: {', '.join(map(str, args.years))}")
    if args.limit:
        print(f"  Limit: {args.limit}")

    # Generate pseudo-labels
    all_annotations = []

    print("\nGenerating pseudo-labels...")
    for piece in tqdm(filtered_pieces, desc="Processing pieces"):
        try:
            annotations = generate_pseudo_labels_for_piece(
                piece,
                maestro_root,
                segment_duration=args.segment_duration,
                segment_overlap=args.segment_overlap,
            )
            all_annotations.extend(annotations)
        except Exception as e:
            print(f"Error processing {piece['piece_id']}: {e}")
            continue

    # Save annotations
    output_file = Path(args.output)
    save_annotations(all_annotations, output_file)

    print("\n" + "="*60)
    print(f"Pseudo-label generation complete!")
    print(f"Pieces processed: {len(filtered_pieces)}")
    print(f"Total segments: {len(all_annotations)}")
    print(f"Output file: {output_file}")
    print("="*60)


if __name__ == '__main__':
    main()
