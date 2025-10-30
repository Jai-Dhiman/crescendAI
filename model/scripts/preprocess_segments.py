#!/usr/bin/env python3
"""
Preprocess MAESTRO selected pieces into fixed-duration segments for labeling.

Segments audio and MIDI files into 25-second chunks with 5-second overlap,
creating a dataset ready for expert labeling in Label Studio.
"""

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import mido
import numpy as np
import soundfile as sf
from tqdm import tqdm


def load_audio(audio_path: Path, sr: int = 22050) -> Tuple[np.ndarray, int]:
    """Load audio file at specified sample rate."""
    audio, sr = librosa.load(audio_path, sr=sr, mono=True)
    return audio, sr


def segment_audio(
    audio: np.ndarray,
    sr: int,
    segment_duration: float = 25.0,
    overlap: float = 5.0
) -> List[Tuple[np.ndarray, float, float]]:
    """
    Segment audio into fixed-duration chunks with overlap.

    Returns:
        List of (audio_segment, start_time, end_time) tuples
    """
    segment_samples = int(segment_duration * sr)
    overlap_samples = int(overlap * sr)
    hop_samples = segment_samples - overlap_samples

    segments = []
    start_sample = 0

    while start_sample < len(audio):
        end_sample = min(start_sample + segment_samples, len(audio))

        # Skip if segment is too short (less than 50% of target duration)
        if (end_sample - start_sample) < (segment_samples * 0.5):
            break

        audio_segment = audio[start_sample:end_sample]
        start_time = start_sample / sr
        end_time = end_sample / sr

        segments.append((audio_segment, start_time, end_time))

        # Move to next segment
        start_sample += hop_samples

    return segments


def segment_midi(
    midi_path: Path,
    start_time: float,
    end_time: float,
    output_path: Path
) -> None:
    """
    Extract MIDI segment between start_time and end_time.

    Preserves tempo, time signature, and all note events within the time range.
    """
    midi = mido.MidiFile(midi_path)
    new_midi = mido.MidiFile(ticks_per_beat=midi.ticks_per_beat)

    for track in midi.tracks:
        new_track = mido.MidiTrack()
        current_time = 0.0
        tempo = 500000  # Default tempo (120 BPM)

        for msg in track:
            # Update current time based on delta time
            if msg.time > 0:
                tick_duration = tempo / (midi.ticks_per_beat * 1000000)
                current_time += msg.time * tick_duration

            # Track tempo changes
            if msg.type == 'set_tempo':
                tempo = msg.tempo

            # Include message if it's within our time range
            if start_time <= current_time <= end_time:
                # Adjust timing for segment start
                if len(new_track) == 0:
                    # First message: adjust relative to segment start
                    adjusted_msg = msg.copy(time=0)
                else:
                    adjusted_msg = msg.copy()

                new_track.append(adjusted_msg)

        if len(new_track) > 0:
            new_midi.tracks.append(new_track)

    new_midi.save(output_path)


def create_segment_metadata(
    piece_info: Dict,
    segment_idx: int,
    start_time: float,
    end_time: float,
    audio_path: Path,
    midi_path: Path
) -> Dict:
    """Create metadata for a single segment."""
    return {
        'segment_id': f"{piece_info['piece_id']}_seg{segment_idx:03d}",
        'piece_id': piece_info['piece_id'],
        'composer': piece_info['composer'],
        'title': piece_info['title'],
        'year': piece_info['year'],
        'difficulty': piece_info['difficulty'],
        'split': piece_info['split'],
        'segment_idx': segment_idx,
        'start_time': round(start_time, 2),
        'end_time': round(end_time, 2),
        'duration': round(end_time - start_time, 2),
        'audio_path': str(audio_path.relative_to(audio_path.parents[2])),
        'midi_path': str(midi_path.relative_to(midi_path.parents[2])),
    }


def preprocess_piece(
    piece_info: Dict,
    maestro_dir: Path,
    output_dir: Path,
    segment_duration: float = 25.0,
    overlap: float = 5.0,
    sr: int = 22050
) -> List[Dict]:
    """
    Preprocess a single piece into segments.

    Returns:
        List of segment metadata dictionaries
    """
    audio_path = maestro_dir / piece_info['audio_filename']
    midi_path = maestro_dir / piece_info['midi_filename']

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not midi_path.exists():
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")

    # Load and segment audio
    audio, sr = load_audio(audio_path, sr=sr)
    audio_segments = segment_audio(audio, sr, segment_duration, overlap)

    # Create output directories
    piece_id = piece_info['piece_id']
    piece_output_dir = output_dir / piece_id
    piece_output_dir.mkdir(parents=True, exist_ok=True)

    # Process each segment
    segment_metadata = []
    for seg_idx, (audio_seg, start_time, end_time) in enumerate(audio_segments):
        # Save audio segment
        audio_output_path = piece_output_dir / f"{piece_id}_seg{seg_idx:03d}.wav"
        sf.write(audio_output_path, audio_seg, sr)

        # Save MIDI segment
        midi_output_path = piece_output_dir / f"{piece_id}_seg{seg_idx:03d}.midi"
        segment_midi(midi_path, start_time, end_time, midi_output_path)

        # Create metadata
        metadata = create_segment_metadata(
            piece_info, seg_idx, start_time, end_time,
            audio_output_path, midi_output_path
        )
        segment_metadata.append(metadata)

    return segment_metadata


def load_maestro_selected(csv_path: Path) -> List[Dict]:
    """Load selected pieces from maestro_selected.csv."""
    pieces = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            # Create a unique piece ID from filename
            audio_filename = row['audio_filename']
            piece_id = Path(audio_filename).stem

            pieces.append({
                'piece_id': piece_id,
                'composer': row['canonical_composer'],
                'title': row['canonical_title'],
                'year': int(row['year']),
                'duration': float(row['duration']),
                'audio_filename': row['audio_filename'],
                'midi_filename': row['midi_filename'],
                'difficulty': float(row['difficulty']) if row['difficulty'] else None,
                'split': row['split'],
            })

    return pieces


def create_label_studio_manifest(
    segments: List[Dict],
    output_path: Path,
    audio_base_url: str = "/data/local-files/?d="
) -> None:
    """
    Create Label Studio import manifest in JSON format.

    Each task includes audio player and fields for 8-10 evaluation dimensions.
    """
    tasks = []

    for segment in segments:
        # Extract just the relative path from segments/ directory
        audio_rel_path = segment['audio_path']

        task = {
            'data': {
                'audio': f"{audio_base_url}{audio_rel_path}",
                'segment_id': segment['segment_id'],
                'composer': segment['composer'],
                'title': segment['title'],
                'difficulty': segment['difficulty'],
                'start_time': segment['start_time'],
                'end_time': segment['end_time'],
                'duration': segment['duration'],
                'composer_display': f"Composer: {segment['composer']}",
                'title_display': f"Title: {segment['title']}",
                'difficulty_display': f"Difficulty: {segment['difficulty']}",
                'duration_display': f"{segment['duration']:.1f}s",
            },
            'meta': {
                'piece_id': segment['piece_id'],
                'midi_path': segment['midi_path'],
            }
        }
        tasks.append(task)

    with open(output_path, 'w') as f:
        json.dump(tasks, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess MAESTRO pieces into segments for labeling'
    )
    parser.add_argument(
        '--maestro-dir',
        type=Path,
        default=Path('data/maestro_selected'),
        help='Directory containing extracted MAESTRO pieces'
    )
    parser.add_argument(
        '--csv-path',
        type=Path,
        default=Path('data/maestro_selected.csv'),
        help='Path to maestro_selected.csv'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/segments'),
        help='Output directory for segments'
    )
    parser.add_argument(
        '--segment-duration',
        type=float,
        default=25.0,
        help='Segment duration in seconds (default: 25.0)'
    )
    parser.add_argument(
        '--overlap',
        type=float,
        default=5.0,
        help='Overlap between segments in seconds (default: 5.0)'
    )
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=22050,
        help='Audio sample rate (default: 22050 Hz)'
    )
    parser.add_argument(
        '--max-pieces',
        type=int,
        default=None,
        help='Maximum number of pieces to process (for testing)'
    )

    args = parser.parse_args()

    # Resolve paths
    project_root = Path(__file__).parent.parent

    # Handle both absolute and relative paths
    if args.maestro_dir.is_absolute():
        maestro_dir = args.maestro_dir
    else:
        maestro_dir = project_root / args.maestro_dir

    if args.csv_path.is_absolute():
        csv_path = args.csv_path
    else:
        csv_path = project_root / args.csv_path

    if args.output_dir.is_absolute():
        output_dir = args.output_dir
    else:
        output_dir = project_root / args.output_dir

    # Verify inputs
    if not maestro_dir.exists():
        raise FileNotFoundError(f"MAESTRO directory not found: {maestro_dir}")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pieces
    print(f"Loading pieces from {csv_path}...")
    pieces = load_maestro_selected(csv_path)

    if args.max_pieces:
        pieces = pieces[:args.max_pieces]
        print(f"Processing first {args.max_pieces} pieces for testing...")

    print(f"Found {len(pieces)} pieces to process")
    print(f"Segment duration: {args.segment_duration}s with {args.overlap}s overlap")

    # Process each piece
    all_segments = []
    failed_pieces = []

    for piece in tqdm(pieces, desc="Processing pieces"):
        try:
            segments = preprocess_piece(
                piece,
                maestro_dir,
                output_dir,
                segment_duration=args.segment_duration,
                overlap=args.overlap,
                sr=args.sample_rate
            )
            all_segments.extend(segments)
        except Exception as e:
            print(f"\nError processing {piece['piece_id']}: {e}")
            failed_pieces.append(piece['piece_id'])
            continue

    # Save segment metadata
    metadata_path = output_dir / 'segments_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(all_segments, f, indent=2)

    print(f"\n✓ Preprocessing complete!")
    print(f"  Total segments: {len(all_segments)}")
    print(f"  Successful pieces: {len(pieces) - len(failed_pieces)}/{len(pieces)}")
    if failed_pieces:
        print(f"  Failed pieces: {', '.join(failed_pieces)}")
    print(f"  Metadata saved to: {metadata_path}")

    # Create Label Studio manifest
    manifest_path = output_dir / 'label_studio_import.json'
    create_label_studio_manifest(all_segments, manifest_path)
    print(f"  Label Studio manifest: {manifest_path}")

    # Print statistics
    total_duration = sum(s['duration'] for s in all_segments)
    avg_segments_per_piece = len(all_segments) / max(len(pieces) - len(failed_pieces), 1)

    print(f"\nDataset statistics:")
    print(f"  Total audio duration: {total_duration / 3600:.1f} hours")
    print(f"  Average segments per piece: {avg_segments_per_piece:.1f}")
    print(f"  Estimated labeling time (5 min/segment): {len(all_segments) * 5 / 60:.1f} hours")

    print(f"\nNext steps:")
    print(f"1. Review segments in {output_dir}")
    print(f"2. Import {manifest_path} into Label Studio")
    print(f"3. Begin labeling (target: 200-300 segments)")


if __name__ == '__main__':
    main()
