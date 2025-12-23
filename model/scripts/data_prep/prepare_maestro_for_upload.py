"""
Prepare MAESTRO dataset for remote training with quality variance.

Efficiently processes maestro-v3.0.0.zip to create maestro_with_variance.tar.gz
with 4 quality tiers, 8-dimension annotations, ready for upload.

Disk-space optimized:
- Processes in chunks
- Cleans up intermediates
- Only keeps final tar.gz

Usage:
    python scripts/prepare_maestro_for_upload.py \
        --maestro_zip ~/Downloads/maestro-v3.0.0.zip \
        --output_dir /tmp/maestro_upload \
        --max_samples 1000  # Optional: limit for testing
"""

import argparse
import json
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.degradation import QualityTier, apply_quality_tier, sample_quality_tier
from labeling_functions import get_all_labeling_functions
from weak_supervision import apply_labeling_functions

try:
    import librosa
    import pretty_midi
except ImportError:
    print("Installing required packages...")
    import subprocess

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "pretty_midi", "librosa", "scipy"]
    )
    import librosa
    import pretty_midi


def extract_maestro_efficiently(
    maestro_zip_path: Path, extract_dir: Path, max_pieces: int = None
):
    """
    Extract MAESTRO zip efficiently (only MIDI + WAV, skip unnecessary files).

    Args:
        maestro_zip_path: Path to maestro-v3.0.0.zip
        extract_dir: Where to extract
        max_pieces: Maximum number of pieces to process (None = all)

    Returns:
        List of (audio_path, midi_path) tuples
    """
    print("\n" + "=" * 80)
    print("STEP 1: EXTRACTING MAESTRO")
    print("=" * 80)

    extract_dir.mkdir(parents=True, exist_ok=True)

    pairs = []

    with zipfile.ZipFile(maestro_zip_path, "r") as zf:
        # Get all WAV and MIDI files
        all_files = [f for f in zf.namelist() if f.endswith((".wav", ".midi", ".mid"))]

        print(f"Found {len([f for f in all_files if f.endswith('.wav')])} audio files")
        print(
            f"Found {len([f for f in all_files if f.endswith(('.midi', '.mid'))])} MIDI files"
        )

        # Extract files
        print("Extracting files...")
        for file in tqdm(all_files, desc="Extracting"):
            if "__MACOSX" in file:
                continue
            zf.extract(file, extract_dir)

        print(f"Extracted to: {extract_dir}")

    # Find audio-MIDI pairs
    audio_files = list(extract_dir.rglob("*.wav"))

    if max_pieces:
        audio_files = audio_files[:max_pieces]

    print(f"\nPairing audio and MIDI files...")
    for audio_path in tqdm(audio_files, desc="Finding pairs"):
        # Try both .midi and .mid extensions
        midi_path = audio_path.with_suffix(".midi")
        if not midi_path.exists():
            midi_path = audio_path.with_suffix(".mid")

        if midi_path.exists():
            pairs.append((audio_path, midi_path))

    print(f"Found {len(pairs)} audio-MIDI pairs")

    return pairs


def create_degraded_segments(
    audio_midi_pairs: list,
    output_dir: Path,
    segment_duration: float = 10.0,
    sample_rate: int = 24000,
    quality_tier_distribution: dict = None,
):
    """
    Create degraded segments with quality tiers.

    Args:
        audio_midi_pairs: List of (audio_path, midi_path) tuples
        output_dir: Output directory for segments
        segment_duration: Segment duration in seconds
        sample_rate: Audio sample rate
        quality_tier_distribution: Custom tier distribution (default: from TRAINING_PLAN_v2.md)

    Returns:
        List of annotation dictionaries
    """
    print("\n" + "=" * 80)
    print("STEP 2: CREATING DEGRADED SEGMENTS")
    print("=" * 80)

    audio_dir = output_dir / "audio"
    midi_dir = output_dir / "midi"
    audio_dir.mkdir(parents=True, exist_ok=True)
    midi_dir.mkdir(parents=True, exist_ok=True)

    # Get labeling functions
    labeling_functions = get_all_labeling_functions()

    annotations = []
    segment_idx = 0

    for audio_path, midi_path in tqdm(audio_midi_pairs, desc="Processing pieces"):
        try:
            # Load MIDI
            midi_data = pretty_midi.PrettyMIDI(str(midi_path))
            midi_duration = midi_data.get_end_time()

            # Load audio
            audio_data, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
            audio_duration = len(audio_data) / sample_rate

            # Use shorter duration
            duration = min(midi_duration, audio_duration)

            if duration < segment_duration:
                continue

            # Create segments
            num_segments = int(duration / segment_duration)

            for seg_idx in range(num_segments):
                start_time = seg_idx * segment_duration
                end_time = min(start_time + segment_duration, duration)

                # Extract segment
                start_sample = int(start_time * sample_rate)
                end_sample = int(end_time * sample_rate)
                audio_segment = audio_data[start_sample:end_sample]

                # For each segment, create 4 quality tiers
                for quality_tier in QualityTier:
                    # Apply degradation
                    degraded_midi, degraded_audio, metadata = apply_quality_tier(
                        midi_data,
                        audio_segment,
                        sample_rate,
                        quality_tier,
                        seed=segment_idx,
                    )

                    # Save files
                    tier_name = quality_tier.value
                    audio_filename = f"{segment_idx:08d}_{tier_name}.wav"
                    midi_filename = f"{segment_idx:08d}_{tier_name}.mid"

                    audio_output_path = audio_dir / audio_filename
                    midi_output_path = midi_dir / midi_filename

                    librosa.output.write_wav(
                        str(audio_output_path), degraded_audio, sample_rate
                    )
                    degraded_midi.write(str(midi_output_path))

                    # Generate weak labels
                    sample_data = {
                        "midi_data": degraded_midi,
                        "audio_data": degraded_audio,
                        "sr": sample_rate,
                    }

                    labels = apply_labeling_functions(
                        sample_data, labeling_functions, use_adaptive_weights=False
                    )

                    # Scale labels by quality tier
                    score_range = metadata["degradations_applied"]
                    scale_factor = metadata["quality_score"] / 100.0

                    scaled_labels = {
                        dim: val * scale_factor for dim, val in labels.items()
                    }

                    # Create annotation
                    annotation = {
                        "audio_path": str(audio_output_path),
                        "midi_path": str(midi_output_path),
                        "start_time": start_time,
                        "end_time": end_time,
                        "labels": scaled_labels,
                        "quality_tier": quality_tier.value,
                        "quality_score": metadata["quality_score"],
                        "degradations": metadata["degradations_applied"],
                        "dataset": "maestro",
                        "original_piece": audio_path.stem,
                    }

                    annotations.append(annotation)
                    segment_idx += 1

        except Exception as e:
            print(f"Error processing {audio_path.name}: {e}")
            continue

    print(f"\nCreated {len(annotations)} degraded segments")

    return annotations


def create_train_val_test_split(
    annotations: list,
    output_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
):
    """
    Split annotations into train/val/test and save.

    Args:
        annotations: List of annotation dicts
        output_dir: Output directory
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
    """
    print("\n" + "=" * 80)
    print("STEP 3: CREATING TRAIN/VAL/TEST SPLIT")
    print("=" * 80)

    # Shuffle
    np.random.shuffle(annotations)

    # Split
    n = len(annotations)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_annotations = annotations[:n_train]
    val_annotations = annotations[n_train : n_train + n_val]
    test_annotations = annotations[n_train + n_val :]

    print(f"Train: {len(train_annotations)}")
    print(f"Val: {len(val_annotations)}")
    print(f"Test: {len(test_annotations)}")

    # Save
    annotations_dir = output_dir / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": train_annotations,
        "val": val_annotations,
        "test": test_annotations,
    }

    for split_name, split_data in splits.items():
        output_path = annotations_dir / f"{split_name}.jsonl"
        with open(output_path, "w") as f:
            for annotation in split_data:
                f.write(json.dumps(annotation) + "\n")
        print(f"Saved {split_name}: {output_path}")

    # Print quality tier distribution
    print("\nQuality tier distribution:")
    for tier in QualityTier:
        count = sum(1 for a in annotations if a["quality_tier"] == tier.value)
        pct = count / len(annotations) * 100
        print(f"  {tier.value}: {count} ({pct:.1f}%)")


def create_final_tarball(data_dir: Path, output_path: Path):
    """
    Create final tar.gz for upload.

    Args:
        data_dir: Directory containing audio/, midi/, annotations/
        output_path: Output tar.gz path
    """
    print("\n" + "=" * 80)
    print("STEP 4: CREATING TAR.GZ FOR UPLOAD")
    print("=" * 80)

    print(f"Creating {output_path}...")

    with tarfile.open(output_path, "w:gz") as tar:
        for item in ["audio", "midi", "annotations"]:
            item_path = data_dir / item
            if item_path.exists():
                print(f"Adding {item}/...")
                tar.add(item_path, arcname=item)

    # Get size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n✓ Created: {output_path}")
    print(f"  Size: {size_mb:.1f} MB")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Prepare MAESTRO with variance for upload"
    )
    parser.add_argument(
        "--maestro_zip", type=str, required=True, help="Path to maestro-v3.0.0.zip"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/tmp/maestro_upload",
        help="Working directory (will be cleaned up)",
    )
    parser.add_argument(
        "--output_tarball",
        type=str,
        default="maestro_with_variance.tar.gz",
        help="Output tar.gz filename",
    )
    parser.add_argument(
        "--max_pieces",
        type=int,
        default=None,
        help="Max pieces to process (for testing, e.g., 10)",
    )
    parser.add_argument(
        "--segment_duration",
        type=float,
        default=10.0,
        help="Segment duration in seconds",
    )
    parser.add_argument(
        "--sample_rate", type=int, default=24000, help="Audio sample rate"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    np.random.seed(args.seed)

    maestro_zip = Path(args.maestro_zip)
    output_dir = Path(args.output_dir)

    if not maestro_zip.exists():
        print(f"Error: {maestro_zip} not found!")
        print("Please download MAESTRO v3.0.0 first:")
        print(
            "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip"
        )
        return

    print("=" * 80)
    print("MAESTRO WITH VARIANCE - DATA PREPARATION")
    print("=" * 80)
    print(f"Input: {maestro_zip}")
    print(f"Output: {args.output_tarball}")
    print(f"Working dir: {output_dir}")
    if args.max_pieces:
        print(f"Max pieces: {args.max_pieces} (TESTING MODE)")
    print("")

    # Step 1: Extract MAESTRO
    extract_dir = output_dir / "maestro_extracted"
    audio_midi_pairs = extract_maestro_efficiently(
        maestro_zip, extract_dir, max_pieces=args.max_pieces
    )

    # Step 2: Create degraded segments
    segments_dir = output_dir / "segments"
    annotations = create_degraded_segments(
        audio_midi_pairs,
        segments_dir,
        segment_duration=args.segment_duration,
        sample_rate=args.sample_rate,
    )

    # Step 3: Train/val/test split
    create_train_val_test_split(annotations, segments_dir)

    # Step 4: Create tar.gz
    output_tarball = Path.cwd() / args.output_tarball
    create_final_tarball(segments_dir, output_tarball)

    # Cleanup
    print("\n" + "=" * 80)
    print("STEP 5: CLEANUP")
    print("=" * 80)
    print(f"Removing temporary directory: {output_dir}")
    shutil.rmtree(output_dir)
    print("✓ Cleanup complete")

    # Final instructions
    print("\n" + "=" * 80)
    print("SUCCESS - READY FOR UPLOAD")
    print("=" * 80)
    print(f"\n✓ Created: {output_tarball}")
    print(f"  Size: {output_tarball.stat().st_size / (1024**3):.2f} GB")
    print("\nNext steps:")
    print("1. Upload this file to Thunder Compute")
    print("2. Upload train_full_model.ipynb")
    print("3. Run all cells in notebook")
    print("4. Get 3 trained models + diagnostics!")


if __name__ == "__main__":
    main()
