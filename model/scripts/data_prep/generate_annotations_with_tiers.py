#!/usr/bin/env python3
"""
Generate annotations with quality tier assignments.

Creates JSONL annotation files where each pristine segment gets
assigned to one of 4 quality tiers (with degradation parameters).
Degradation is applied at runtime by the dataloader.

This approach:
- Saves 4x disk space (no degraded file copies)
- Allows flexible degradation parameters
- Supports augmentation variation during training

Usage:
    python scripts/generate_annotations_with_tiers.py \
        --segments_dir /tmp/maestro_segments \
        --output_dir /tmp/maestro_segments/annotations
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional
import sys

import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.crescendai.data.degradation import QualityTier, QUALITY_TIER_PARAMS


# 8 dimensions from TRAINING_PLAN_v2.md
DIMENSIONS = [
    'note_accuracy',
    'rhythmic_stability',
    'articulation_clarity',
    'pedal_technique',
    'tone_quality',
    'dynamic_range',
    'musical_expression',
    'overall_interpretation',
]


def assign_quality_tier(seed: int) -> QualityTier:
    """
    Assign quality tier based on probability distribution.

    Distribution:
    - Pristine: 30%
    - Good: 30%
    - Moderate: 25%
    - Poor: 15%
    """
    random.seed(seed)
    r = random.random()

    cumulative = 0
    for tier in QualityTier:
        cumulative += QUALITY_TIER_PARAMS[tier]['probability']
        if r < cumulative:
            return tier

    return QualityTier.POOR  # Fallback


def compute_base_labels(segment: Dict, seed: int) -> Dict[str, float]:
    """
    Compute base labels for a pristine segment.

    For MAESTRO (virtuoso performances), base scores are high (85-100).
    These will be scaled down based on quality tier.
    """
    random.seed(seed)
    np.random.seed(seed)

    labels = {}

    # All MAESTRO performances are high quality (virtuoso)
    # Base scores between 85-100 with slight variation
    for dim in DIMENSIONS:
        base = np.random.uniform(85, 100)
        labels[dim] = round(base, 2)

    return labels


def create_annotation(
    segment: Dict,
    quality_tier: QualityTier,
    segment_idx: int,
) -> Dict:
    """
    Create a single annotation with quality tier info.
    """
    params = QUALITY_TIER_PARAMS[quality_tier]

    # Compute base labels (for pristine quality)
    base_labels = compute_base_labels(segment, seed=segment_idx)

    # Scale labels based on quality tier
    score_range = params['score_range']
    scale_factor = np.mean(score_range) / 100.0

    scaled_labels = {}
    for dim, base_value in base_labels.items():
        scaled_labels[dim] = round(base_value * scale_factor, 2)

    # Generate quality score within tier range
    quality_score = np.random.uniform(score_range[0], score_range[1])

    # Note: start_time/end_time are set to 0/duration because the audio files
    # are already segmented. The original times from the full piece are stored
    # in original_start_time/original_end_time for reference.
    return {
        'segment_id': segment['segment_id'],
        'audio_path': segment['audio_path'],
        'midi_path': segment.get('midi_path'),
        'duration': segment['duration'],
        'original_start_time': segment['start_time'],
        'original_end_time': segment['end_time'],
        'labels': scaled_labels,
        # Quality tier info (used by dataloader for runtime degradation)
        'quality_tier': quality_tier.value,
        'quality_score': round(quality_score, 2),
        'degradation_params': {
            'midi_jitter_ms': params['midi_jitter_ms'],
            'wrong_note_rate': params['wrong_note_rate'],
            'dynamics_compression': params['dynamics_compression'],
            'audio_noise_snr_db': params['audio_noise_snr_db'],
            'audio_filter_enabled': params['audio_filter_enabled'],
        },
        # Original metadata
        'split': segment.get('split', 'train'),
        'piece_id': segment.get('piece_id', ''),
        'composer': segment.get('canonical_composer', 'Unknown'),
        'title': segment.get('canonical_title', 'Unknown'),
    }


def generate_annotations(
    segments_metadata_path: Path,
    output_dir: Path,
    samples_per_segment: int = 1,
    seed: int = 42,
) -> Dict[str, int]:
    """
    Generate annotations for all segments with quality tier assignments.

    Args:
        segments_metadata_path: Path to segments_metadata.json from segmentation
        output_dir: Output directory for JSONL files
        samples_per_segment: How many annotations per segment (1 = each segment gets one tier)
        seed: Random seed

    Returns:
        Dictionary with counts per split
    """
    print("="*70)
    print("ANNOTATION GENERATION")
    print("="*70)

    random.seed(seed)
    np.random.seed(seed)

    # Load segments metadata
    with open(segments_metadata_path) as f:
        all_segments = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)

    counts = {}
    tier_counts = {tier.value: 0 for tier in QualityTier}

    for split, segments in all_segments.items():
        print(f"\nProcessing {split}: {len(segments)} segments")

        annotations = []
        segment_idx = 0

        for segment in tqdm(segments, desc=split):
            for _ in range(samples_per_segment):
                # Assign quality tier
                tier = assign_quality_tier(seed=seed + segment_idx)
                tier_counts[tier.value] += 1

                # Create annotation
                annotation = create_annotation(segment, tier, segment_idx)
                annotations.append(annotation)

                segment_idx += 1

        # Shuffle annotations
        random.shuffle(annotations)

        # Save to JSONL
        output_path = output_dir / f"{split}.jsonl"
        with open(output_path, 'w') as f:
            for ann in annotations:
                f.write(json.dumps(ann) + '\n')

        counts[split] = len(annotations)
        print(f"  Saved {len(annotations)} annotations to {output_path}")

    # Print summary
    print("\n" + "="*70)
    print("ANNOTATION SUMMARY")
    print("="*70)

    total = sum(counts.values())
    print(f"\nTotal annotations: {total}")

    for split, count in counts.items():
        print(f"  {split}: {count}")

    print(f"\nQuality tier distribution:")
    for tier, count in tier_counts.items():
        pct = count / total * 100 if total > 0 else 0
        print(f"  {tier}: {count} ({pct:.1f}%)")

    return counts


def main():
    parser = argparse.ArgumentParser(description="Generate annotations with quality tiers")
    parser.add_argument('--segments_dir', type=str, required=True,
                        help='Directory containing segmented data (from segment_maestro.py)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for annotations (default: segments_dir/annotations)')
    parser.add_argument('--samples_per_segment', type=int, default=1,
                        help='Annotations per segment (default: 1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    segments_dir = Path(args.segments_dir)
    segments_metadata_path = segments_dir / "segments_metadata.json"

    if not segments_metadata_path.exists():
        print(f"ERROR: {segments_metadata_path} not found")
        print("Run segment_maestro.py first")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else segments_dir / "annotations"

    generate_annotations(
        segments_metadata_path=segments_metadata_path,
        output_dir=output_dir,
        samples_per_segment=args.samples_per_segment,
        seed=args.seed,
    )

    print(f"\nAnnotations saved to: {output_dir}")
    print("\nNext step: Package with package_dataset.py")


if __name__ == "__main__":
    main()
