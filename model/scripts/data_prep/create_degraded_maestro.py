"""
Create degraded MAESTRO dataset with 4 quality tiers.

Processes original MAESTRO segments and creates 4 versions of each:
- Pristine (30%): Original
- Good (30%): Light degradation
- Moderate (25%): Moderate degradation
- Poor (15%): Heavy degradation

Input: Original MAESTRO annotation files (synthetic_train/val/test.jsonl)
Output: New annotation files with 4x samples and quality tier labels

Usage:
    python scripts/create_degraded_maestro.py \
        --input_dir /tmp/crescendai_data/data/annotations \
        --output_dir /tmp/crescendai_data/data/annotations_degraded \
        --audio_dir /tmp/crescendai_data/data/all_segments \
        --midi_dir /tmp/crescendai_data/data/all_segments \
        --max_samples 1000  # For testing, omit for full dataset
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

import librosa
import numpy as np
import pretty_midi
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.degradation import (
    QUALITY_TIER_PARAMS,
    QualityTier,
    apply_quality_tier,
    sample_quality_tier,
)


def process_single_sample(
    annotation: Dict,
    audio_dir: Path,
    midi_dir: Path,
    output_audio_dir: Path,
    output_midi_dir: Path,
    quality_tier: QualityTier,
    sample_idx: int,
    sample_rate: int = 24000,
) -> Optional[Dict]:
    """
    Process a single sample and create degraded version.

    Args:
        annotation: Original annotation dictionary
        audio_dir: Directory containing original audio files
        midi_dir: Directory containing original MIDI files
        output_audio_dir: Directory for output audio files
        output_midi_dir: Directory for output MIDI files
        quality_tier: Quality tier to apply
        sample_idx: Unique sample index for naming
        sample_rate: Audio sample rate

    Returns:
        New annotation dictionary with degraded paths, or None if processing failed
    """
    try:
        # Load original files
        audio_path = Path(annotation.get("audio_path", ""))
        midi_path = Path(annotation.get("midi_path", ""))

        # Load MIDI
        midi_data = None
        if midi_path.exists():
            try:
                midi_data = pretty_midi.PrettyMIDI(str(midi_path))
            except Exception as e:
                print(f"Warning: Could not load MIDI {midi_path}: {e}")
                return None

        # Load audio
        audio_data = None
        if audio_path.exists():
            try:
                audio_data, sr = librosa.load(
                    str(audio_path), sr=sample_rate, mono=True
                )
            except Exception as e:
                print(f"Warning: Could not load audio {audio_path}: {e}")
                return None

        # Apply degradation
        degraded_midi, degraded_audio, metadata = apply_quality_tier(
            midi_data,
            audio_data,
            sample_rate,
            quality_tier,
            seed=sample_idx,  # Use sample index as seed for reproducibility
        )

        # Generate output filenames
        tier_name = quality_tier.value
        audio_output_name = f"{sample_idx:08d}_{tier_name}.wav"
        midi_output_name = f"{sample_idx:08d}_{tier_name}.mid"

        audio_output_path = output_audio_dir / audio_output_name
        midi_output_path = output_midi_dir / midi_output_name

        # Save degraded files
        if degraded_audio is not None:
            librosa.output.write_wav(
                str(audio_output_path), degraded_audio, sample_rate
            )
        else:
            audio_output_path = None

        if degraded_midi is not None:
            degraded_midi.write(str(midi_output_path))
        else:
            midi_output_path = None

        # Create new annotation
        new_annotation = {
            "audio_path": str(audio_output_path) if audio_output_path else None,
            "midi_path": str(midi_output_path) if midi_output_path else None,
            "start_time": annotation.get("start_time", 0),
            "end_time": annotation.get("end_time", 10),
            "labels": annotation["labels"].copy(),
            "dataset": annotation.get("dataset", "maestro"),
            "quality_tier": metadata["quality_tier"],
            "quality_score": metadata["quality_score"],
            "degradations_applied": metadata["degradations_applied"],
            "original_audio_path": str(audio_path),
            "original_midi_path": str(midi_path),
        }

        # Adjust labels based on quality tier
        # Scale original labels by quality score (0-100 scale)
        # Pristine: multiply by ~0.97 (average of 95-100 range)
        # Good: multiply by ~0.87 (average of 80-95 range)
        # Moderate: multiply by ~0.72 (average of 65-80 range)
        # Poor: multiply by ~0.57 (average of 50-65 range)
        score_range = QUALITY_TIER_PARAMS[quality_tier]["score_range"]
        scale_factor = np.mean(score_range) / 100.0

        for dim, original_value in annotation["labels"].items():
            # Scale down based on degradation level
            degraded_value = original_value * scale_factor
            new_annotation["labels"][dim] = degraded_value

        return new_annotation

    except Exception as e:
        print(f"Error processing sample {sample_idx}: {e}")
        return None


def process_annotation_file(
    input_file: Path,
    output_file: Path,
    audio_dir: Path,
    midi_dir: Path,
    output_audio_dir: Path,
    output_midi_dir: Path,
    max_samples: Optional[int] = None,
    sample_rate: int = 24000,
):
    """
    Process an entire annotation file.

    Creates 4 degraded versions of each sample (one per quality tier).

    Args:
        input_file: Input JSONL annotation file
        output_file: Output JSONL annotation file
        audio_dir: Directory containing original audio files
        midi_dir: Directory containing original MIDI files
        output_audio_dir: Directory for output audio files
        output_midi_dir: Directory for output MIDI files
        max_samples: Maximum number of original samples to process (None = all)
        sample_rate: Audio sample rate
    """
    print(f"\n{'=' * 80}")
    print(f"Processing {input_file.name}")
    print(f"{'=' * 80}")

    # Read annotations
    annotations = []
    with open(input_file, "r") as f:
        for line in f:
            annotations.append(json.loads(line))

    if max_samples:
        annotations = annotations[:max_samples]

    print(f"Loaded {len(annotations)} annotations")
    print(f"Will generate {len(annotations) * 4} degraded samples (4 tiers per sample)")

    # Process each annotation with all 4 quality tiers
    all_degraded = []
    sample_idx = 0

    for orig_annotation in tqdm(annotations, desc="Processing samples"):
        # Create one version per quality tier
        for quality_tier in QualityTier:
            degraded_annotation = process_single_sample(
                orig_annotation,
                audio_dir,
                midi_dir,
                output_audio_dir,
                output_midi_dir,
                quality_tier,
                sample_idx,
                sample_rate,
            )

            if degraded_annotation:
                all_degraded.append(degraded_annotation)

            sample_idx += 1

    print(f"Successfully created {len(all_degraded)} degraded samples")

    # Save output annotations
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        for annotation in all_degraded:
            f.write(json.dumps(annotation) + "\n")

    print(f"Saved to: {output_file}")

    # Print statistics
    tier_counts = {}
    for tier in QualityTier:
        count = sum(1 for a in all_degraded if a["quality_tier"] == tier.value)
        tier_counts[tier.value] = count

    print(f"\nQuality tier distribution:")
    for tier_name, count in tier_counts.items():
        percentage = count / len(all_degraded) * 100 if all_degraded else 0
        print(f"  {tier_name}: {count} ({percentage:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Create degraded MAESTRO dataset")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing original annotation files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for degraded annotations",
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        required=True,
        help="Directory containing original audio files",
    )
    parser.add_argument(
        "--midi_dir",
        type=str,
        required=True,
        help="Directory containing original MIDI files",
    )
    parser.add_argument(
        "--output_audio_dir",
        type=str,
        default=None,
        help="Output directory for degraded audio files (default: output_dir/audio)",
    )
    parser.add_argument(
        "--output_midi_dir",
        type=str,
        default=None,
        help="Output directory for degraded MIDI files (default: output_dir/midi)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of original samples per split (for testing, e.g., 100)",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=24000,
        help="Audio sample rate (default: 24000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    print("=" * 80)
    print("DEGRADED MAESTRO DATASET GENERATION")
    print("=" * 80)
    print(f"\nInput directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Audio directory: {args.audio_dir}")
    print(f"MIDI directory: {args.midi_dir}")

    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    audio_dir = Path(args.audio_dir)
    midi_dir = Path(args.midi_dir)

    # Setup output directories
    if args.output_audio_dir:
        output_audio_dir = Path(args.output_audio_dir)
    else:
        output_audio_dir = output_dir / "audio"

    if args.output_midi_dir:
        output_midi_dir = Path(args.output_midi_dir)
    else:
        output_midi_dir = output_dir / "midi"

    output_audio_dir.mkdir(parents=True, exist_ok=True)
    output_midi_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output audio directory: {output_audio_dir}")
    print(f"Output MIDI directory: {output_midi_dir}")

    # Process each split
    splits = ["train", "val", "test"]

    for split in splits:
        input_file = input_dir / f"synthetic_{split}.jsonl"

        if not input_file.exists():
            print(f"\nWarning: {input_file} not found, skipping...")
            continue

        output_file = output_dir / f"degraded_{split}.jsonl"

        process_annotation_file(
            input_file,
            output_file,
            audio_dir,
            midi_dir,
            output_audio_dir,
            output_midi_dir,
            max_samples=args.max_samples,
            sample_rate=args.sample_rate,
        )

    print("\n" + "=" * 80)
    print("DEGRADED DATASET GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")
    print("\nNext steps:")
    print("1. Update experiment_full.yaml to point to degraded annotations")
    print("2. Run diagnostic training (Phase 2)")
    print("3. Analyze whether model learns quality vs complexity")


if __name__ == "__main__":
    main()
