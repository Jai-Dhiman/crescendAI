#!/usr/bin/env python3
"""
Segment MAESTRO dataset into fixed-length clips.

Extracts MAESTRO zip and creates 10-second audio/MIDI segments.
This is the first step in creating the training dataset.

Space-efficient approach:
- Only stores pristine segments (no degraded copies)
- Degradation applied at runtime in dataloader

Usage:
    python scripts/segment_maestro.py \
        --maestro_zip data/maestro-v3.0.0.zip \
        --output_dir /tmp/maestro_segments \
        --segment_length 10 \
        --max_pieces 100  # Optional: limit for testing
"""

import argparse
import json
import zipfile
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import shutil
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm


def extract_maestro_metadata(zip_path: Path) -> pd.DataFrame:
    """Extract metadata CSV from MAESTRO zip without full extraction."""
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Find the CSV file
        csv_files = [f for f in zf.namelist() if f.endswith('.csv')]
        if not csv_files:
            raise ValueError("No CSV file found in MAESTRO zip")

        csv_name = csv_files[0]
        print(f"Reading metadata from: {csv_name}")

        with zf.open(csv_name) as f:
            df = pd.read_csv(f)

    return df


def segment_audio_file(
    audio_path: Path,
    output_dir: Path,
    segment_length: float,
    sample_rate: int,
    piece_id: str,
) -> List[Dict]:
    """
    Segment a single audio file into fixed-length clips.

    Returns list of segment metadata dictionaries.
    """
    try:
        import librosa
        import soundfile as sf
    except ImportError:
        print("ERROR: librosa and soundfile required. Install with: uv pip install librosa soundfile")
        sys.exit(1)

    segments = []

    try:
        # Load audio
        audio, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
        duration = len(audio) / sr

        # Calculate number of segments
        num_segments = int(duration // segment_length)

        if num_segments == 0:
            return segments

        samples_per_segment = int(segment_length * sr)

        for i in range(num_segments):
            start_sample = i * samples_per_segment
            end_sample = start_sample + samples_per_segment

            segment_audio = audio[start_sample:end_sample]

            # Skip silent segments
            if np.max(np.abs(segment_audio)) < 0.01:
                continue

            # Save segment
            segment_name = f"{piece_id}_seg{i:04d}.wav"
            segment_path = output_dir / "audio" / segment_name

            sf.write(str(segment_path), segment_audio, sr)

            segments.append({
                'segment_id': f"{piece_id}_seg{i:04d}",
                'piece_id': piece_id,
                'segment_idx': i,
                'start_time': i * segment_length,
                'end_time': (i + 1) * segment_length,
                'audio_path': str(segment_path),
                'duration': segment_length,
            })

    except Exception as e:
        print(f"  Warning: Failed to process {audio_path}: {e}")

    return segments


def segment_midi_file(
    midi_path: Path,
    output_dir: Path,
    segment_length: float,
    piece_id: str,
    audio_segments: List[Dict],
) -> None:
    """
    Segment MIDI file to match audio segments.
    Updates audio_segments list with MIDI paths.
    """
    try:
        import pretty_midi
    except ImportError:
        print("ERROR: pretty_midi required. Install with: uv pip install pretty_midi")
        sys.exit(1)

    try:
        midi = pretty_midi.PrettyMIDI(str(midi_path))

        for seg in audio_segments:
            start_time = seg['start_time']
            end_time = seg['end_time']

            # Create new MIDI with only notes in this time range
            segment_midi = pretty_midi.PrettyMIDI()

            for instrument in midi.instruments:
                new_instrument = pretty_midi.Instrument(
                    program=instrument.program,
                    is_drum=instrument.is_drum,
                    name=instrument.name
                )

                for note in instrument.notes:
                    # Check if note overlaps with segment
                    if note.end > start_time and note.start < end_time:
                        # Clip note to segment boundaries
                        new_start = max(0, note.start - start_time)
                        new_end = min(segment_length, note.end - start_time)

                        if new_end > new_start:
                            new_note = pretty_midi.Note(
                                velocity=note.velocity,
                                pitch=note.pitch,
                                start=new_start,
                                end=new_end
                            )
                            new_instrument.notes.append(new_note)

                # Copy control changes in time range
                for cc in instrument.control_changes:
                    if start_time <= cc.time < end_time:
                        new_cc = pretty_midi.ControlChange(
                            number=cc.number,
                            value=cc.value,
                            time=cc.time - start_time
                        )
                        new_instrument.control_changes.append(new_cc)

                if new_instrument.notes:
                    segment_midi.instruments.append(new_instrument)

            # Save MIDI segment
            midi_name = f"{seg['segment_id']}.mid"
            midi_path = output_dir / "midi" / midi_name

            if segment_midi.instruments:
                segment_midi.write(str(midi_path))
                seg['midi_path'] = str(midi_path)
            else:
                seg['midi_path'] = None

    except Exception as e:
        print(f"  Warning: Failed to process MIDI {midi_path}: {e}")
        # Set MIDI paths to None for all segments
        for seg in audio_segments:
            seg['midi_path'] = None


def process_maestro(
    zip_path: Path,
    output_dir: Path,
    segment_length: float = 10.0,
    sample_rate: int = 24000,
    max_pieces: Optional[int] = None,
    splits: List[str] = ['train', 'validation', 'test'],
) -> Dict[str, List[Dict]]:
    """
    Process MAESTRO dataset: extract, segment, and create metadata.

    Returns dictionary of split -> list of segment metadata.
    """
    print("="*70)
    print("MAESTRO SEGMENTATION")
    print("="*70)

    # Create output directories
    (output_dir / "audio").mkdir(parents=True, exist_ok=True)
    (output_dir / "midi").mkdir(parents=True, exist_ok=True)

    # Read metadata
    metadata = extract_maestro_metadata(zip_path)
    print(f"\nTotal pieces in MAESTRO: {len(metadata)}")

    # Create temp extraction directory
    temp_dir = output_dir / "_temp_extract"
    temp_dir.mkdir(exist_ok=True)

    all_segments = {split: [] for split in splits}

    # Process by split
    for split in splits:
        split_df = metadata[metadata['split'] == split]

        if max_pieces:
            split_df = split_df.head(max_pieces // len(splits))

        print(f"\n{'='*50}")
        print(f"Processing {split}: {len(split_df)} pieces")
        print(f"{'='*50}")

        with zipfile.ZipFile(zip_path, 'r') as zf:
            for idx, row in tqdm(split_df.iterrows(), total=len(split_df), desc=split):
                piece_id = Path(row['audio_filename']).stem

                # Extract audio
                audio_zip_path = f"maestro-v3.0.0/{row['audio_filename']}"
                midi_zip_path = f"maestro-v3.0.0/{row['midi_filename']}"

                try:
                    # Extract to temp
                    zf.extract(audio_zip_path, temp_dir)
                    zf.extract(midi_zip_path, temp_dir)

                    audio_path = temp_dir / audio_zip_path
                    midi_path = temp_dir / midi_zip_path

                    # Segment audio
                    segments = segment_audio_file(
                        audio_path, output_dir, segment_length, sample_rate, piece_id
                    )

                    # Segment MIDI (updates segments in-place)
                    if segments:
                        segment_midi_file(
                            midi_path, output_dir, segment_length, piece_id, segments
                        )

                    # Add metadata
                    for seg in segments:
                        seg['split'] = split
                        seg['canonical_composer'] = row.get('canonical_composer', 'Unknown')
                        seg['canonical_title'] = row.get('canonical_title', 'Unknown')

                    all_segments[split].extend(segments)

                    # Clean up temp files immediately to save space
                    audio_path.unlink(missing_ok=True)
                    midi_path.unlink(missing_ok=True)

                except Exception as e:
                    print(f"  Error processing {piece_id}: {e}")
                    continue

    # Clean up temp directory
    shutil.rmtree(temp_dir, ignore_errors=True)

    # Print summary
    print("\n" + "="*70)
    print("SEGMENTATION COMPLETE")
    print("="*70)

    total_segments = 0
    for split, segments in all_segments.items():
        print(f"{split}: {len(segments)} segments")
        total_segments += len(segments)

    print(f"\nTotal: {total_segments} segments")

    # Calculate disk usage
    audio_size = sum(f.stat().st_size for f in (output_dir / "audio").glob("*.wav"))
    midi_size = sum(f.stat().st_size for f in (output_dir / "midi").glob("*.mid"))

    print(f"\nDisk usage:")
    print(f"  Audio: {audio_size / 1e9:.2f} GB")
    print(f"  MIDI: {midi_size / 1e6:.2f} MB")
    print(f"  Total: {(audio_size + midi_size) / 1e9:.2f} GB")

    return all_segments


def main():
    parser = argparse.ArgumentParser(description="Segment MAESTRO dataset")
    parser.add_argument('--maestro_zip', type=str, required=True,
                        help='Path to maestro-v3.0.0.zip')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for segments')
    parser.add_argument('--segment_length', type=float, default=10.0,
                        help='Segment length in seconds (default: 10)')
    parser.add_argument('--sample_rate', type=int, default=24000,
                        help='Audio sample rate (default: 24000)')
    parser.add_argument('--max_pieces', type=int, default=None,
                        help='Maximum pieces to process (for testing)')

    args = parser.parse_args()

    zip_path = Path(args.maestro_zip)
    output_dir = Path(args.output_dir)

    if not zip_path.exists():
        print(f"ERROR: {zip_path} not found")
        sys.exit(1)

    # Process
    all_segments = process_maestro(
        zip_path=zip_path,
        output_dir=output_dir,
        segment_length=args.segment_length,
        sample_rate=args.sample_rate,
        max_pieces=args.max_pieces,
    )

    # Save segment metadata
    metadata_path = output_dir / "segments_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(all_segments, f, indent=2)

    print(f"\nMetadata saved to: {metadata_path}")
    print("\nNext step: Run generate_annotations_with_tiers.py")


if __name__ == "__main__":
    main()
