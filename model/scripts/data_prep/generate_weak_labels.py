"""
Generate weak supervision labels for ATEPP and MAESTRO datasets.

Applies labeling functions to create pseudo-labels for training.
Outputs JSONL annotation files compatible with PerformanceDataset.

Usage:
    python scripts/generate_weak_labels.py \
        --atepp_dir /tmp/atepp_data \
        --maestro_dir /tmp/maestro_data \
        --output_dir /tmp/weak_labels \
        --max_samples 10000
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import pretty_midi
import librosa
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from labeling_functions import get_all_labeling_functions
from weak_supervision import apply_labeling_functions


def process_midi_file(
    midi_path: Path,
    labeling_functions: Dict,
    audio_path: Optional[Path] = None,
    segment_duration: float = 10.0,
    sample_rate: int = 24000,
    skip_audio: bool = False,
) -> List[Dict]:
    """
    Process a single MIDI file and generate weak labels.

    Args:
        midi_path: Path to MIDI file
        labeling_functions: Dictionary of labeling functions by dimension
        audio_path: Optional path to audio file (for audio-based functions)
        segment_duration: Duration of each segment in seconds
        sample_rate: Audio sample rate

    Returns:
        List of annotation dictionaries
    """
    try:
        # Load MIDI
        midi_data = pretty_midi.PrettyMIDI(str(midi_path))

        # Get MIDI duration
        midi_duration = midi_data.get_end_time()

        if midi_duration < 1.0:
            # Skip very short files
            return []

        # Load audio if available (unless skipped)
        audio_data = None
        if not skip_audio and audio_path and audio_path.exists():
            try:
                audio_data, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
            except Exception as e:
                print(f"Warning: Could not load audio {audio_path}: {e}")

        # Segment into chunks
        annotations = []
        num_segments = max(1, int(midi_duration / segment_duration))

        for seg_idx in range(num_segments):
            start_time = seg_idx * segment_duration
            end_time = min(start_time + segment_duration, midi_duration)

            # Extract segment MIDI (for segment-specific labeling)
            # For now, use full MIDI (segment extraction is complex)
            # In production, implement proper MIDI segmentation

            # Prepare data for labeling functions
            sample_data = {
                'midi_data': midi_data,
                'audio_data': audio_data,
                'sr': sample_rate,
            }

            # Apply labeling functions
            labels = apply_labeling_functions(
                sample_data,
                labeling_functions,
                use_adaptive_weights=False
            )

            # Create annotation entry
            annotation = {
                'audio_path': str(audio_path) if audio_path else None,
                'midi_path': str(midi_path),
                'start_time': start_time,
                'end_time': end_time,
                'labels': labels,
                'dataset': 'atepp' if 'atepp' in str(midi_path).lower() else 'maestro',
            }

            annotations.append(annotation)

    except Exception as e:
        print(f"Error processing {midi_path}: {e}")
        return []

    return annotations


def process_dataset(
    dataset_dir: Path,
    dataset_name: str,
    labeling_functions: Dict,
    max_samples: Optional[int] = None,
    audio_extensions: List[str] = ['.wav', '.mp3', '.flac'],
    skip_audio: bool = False,
) -> List[Dict]:
    """
    Process an entire dataset directory.

    Args:
        dataset_dir: Root directory of dataset
        dataset_name: Name of dataset (for logging)
        labeling_functions: Dictionary of labeling functions
        max_samples: Maximum number of samples to process
        audio_extensions: List of audio file extensions to look for

    Returns:
        List of annotations
    """
    print(f"\n{'='*80}")
    print(f"Processing {dataset_name} dataset")
    print(f"{'='*80}")
    print(f"Dataset directory: {dataset_dir}")

    # Find all MIDI files (excluding macOS metadata)
    midi_files = []
    for ext in ['.mid', '.midi']:
        all_files = list(dataset_dir.rglob(f'*{ext}'))
        # Filter out __MACOSX metadata files
        midi_files.extend([f for f in all_files if '__MACOSX' not in str(f)])

    print(f"Found {len(midi_files)} MIDI files")

    if max_samples:
        midi_files = midi_files[:max_samples]
        print(f"Limiting to {max_samples} samples")

    # Process each MIDI file
    all_annotations = []

    for midi_path in tqdm(midi_files, desc=f"Processing {dataset_name}"):
        # Look for corresponding audio file
        audio_path = None
        for ext in audio_extensions:
            potential_audio = midi_path.with_suffix(ext)
            if potential_audio.exists():
                audio_path = potential_audio
                break

        # Process file
        annotations = process_midi_file(
            midi_path,
            labeling_functions,
            audio_path=audio_path,
            skip_audio=skip_audio
        )

        all_annotations.extend(annotations)

    print(f"Generated {len(all_annotations)} annotations from {dataset_name}")
    return all_annotations


def save_annotations(
    annotations: List[Dict],
    output_path: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
):
    """
    Save annotations to JSONL files (train/val/test split).

    Args:
        annotations: List of annotation dictionaries
        output_path: Base output path (will create train/val/test files)
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
    """
    print(f"\n{'='*80}")
    print("Saving annotations")
    print(f"{'='*80}")

    # Shuffle annotations
    np.random.shuffle(annotations)

    # Split into train/val/test
    n = len(annotations)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_annotations = annotations[:n_train]
    val_annotations = annotations[n_train:n_train + n_val]
    test_annotations = annotations[n_train + n_val:]

    print(f"Train: {len(train_annotations)}")
    print(f"Val: {len(val_annotations)}")
    print(f"Test: {len(test_annotations)}")

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save splits
    splits = {
        'train': train_annotations,
        'val': val_annotations,
        'test': test_annotations,
    }

    for split_name, split_annotations in splits.items():
        split_path = output_path.parent / f"{output_path.stem}_{split_name}.jsonl"

        with open(split_path, 'w') as f:
            for annotation in split_annotations:
                f.write(json.dumps(annotation) + '\n')

        print(f"Saved {split_name}: {split_path}")

    # Save statistics
    stats = {
        'total_samples': len(annotations),
        'train_samples': len(train_annotations),
        'val_samples': len(val_annotations),
        'test_samples': len(test_annotations),
        'dimensions': list(annotations[0]['labels'].keys()) if annotations else [],
    }

    # Compute label statistics
    if annotations:
        for dim in stats['dimensions']:
            values = [a['labels'][dim] for a in annotations if dim in a['labels']]
            stats[f'{dim}_mean'] = float(np.mean(values))
            stats[f'{dim}_std'] = float(np.std(values))
            stats[f'{dim}_min'] = float(np.min(values))
            stats[f'{dim}_max'] = float(np.max(values))

    stats_path = output_path.parent / f"{output_path.stem}_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"Saved statistics: {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate weak supervision labels")
    parser.add_argument(
        '--atepp_dir',
        type=str,
        help='Path to ATEPP dataset directory'
    )
    parser.add_argument(
        '--maestro_dir',
        type=str,
        help='Path to MAESTRO dataset directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/tmp/weak_labels',
        help='Output directory for annotations'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples per dataset (for testing, e.g., 100)'
    )
    parser.add_argument(
        '--skip_audio',
        action='store_true',
        help='Skip audio loading (MIDI-only labeling functions, faster)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    print("=" * 80)
    print("WEAK SUPERVISION LABEL GENERATION")
    print("=" * 80)

    # Load labeling functions
    print("\nLoading labeling functions...")
    labeling_functions = get_all_labeling_functions()

    print("Labeling functions loaded:")
    for dim, funcs in labeling_functions.items():
        print(f"  {dim}: {len(funcs)} functions")

    # Process datasets
    all_annotations = []

    if args.atepp_dir:
        atepp_dir = Path(args.atepp_dir)
        if not atepp_dir.exists():
            print(f"Warning: ATEPP directory not found: {atepp_dir}")
        else:
            atepp_annotations = process_dataset(
                atepp_dir,
                "ATEPP",
                labeling_functions,
                max_samples=args.max_samples,
                skip_audio=args.skip_audio
            )
            all_annotations.extend(atepp_annotations)

    if args.maestro_dir:
        maestro_dir = Path(args.maestro_dir)
        if not maestro_dir.exists():
            print(f"Warning: MAESTRO directory not found: {maestro_dir}")
        else:
            maestro_annotations = process_dataset(
                maestro_dir,
                "MAESTRO",
                labeling_functions,
                max_samples=args.max_samples,
                skip_audio=args.skip_audio
            )
            all_annotations.extend(maestro_annotations)

    if not all_annotations:
        print("\nError: No annotations generated. Please check dataset paths.")
        return

    # Save annotations
    output_path = Path(args.output_dir) / "weak_labels.jsonl"
    save_annotations(all_annotations, output_path)

    print("\n" + "=" * 80)
    print("LABEL GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nGenerated {len(all_annotations)} total annotations")
    print(f"Output directory: {output_path.parent}")
    print("\nNext steps:")
    print("1. Update experiment_full.yaml to point to these annotation files")
    print("2. Run training with train_full_model.ipynb")
    print("3. Evaluate results to determine which dimensions need expert labels")


if __name__ == "__main__":
    main()
