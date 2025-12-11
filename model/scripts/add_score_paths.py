#!/usr/bin/env python3
"""
Add score_path mappings to existing PercePiano JSON files.

This script matches performance MIDI files to their reference MusicXML scores
based on the PercePiano naming convention.

Usage:
    python scripts/add_score_paths.py --data-dir data/processed --score-dir data/raw/PercePiano/virtuoso/data/score_xml
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Optional


def extract_piece_info(filename: str) -> Dict[str, str]:
    """
    Extract piece information from PercePiano filename.

    Naming convention: {Composer}_{Piece}_{bars}bars_{segment}_{performer}.mid
    Score naming: {Composer}_{Piece}_{bars}bars_Score_{segment}.musicxml

    Examples:
        Schubert_D960_mv3_8bars_9_20.mid -> Schubert_D960_mv3_8bars
        Beethoven_WoO80_var26_8bars_1_15.mid -> Beethoven_WoO80_var26_8bars
    """
    stem = Path(filename).stem

    # Pattern: Composer_Piece_Nbars_segment_performer
    # We want: Composer_Piece_Nbars
    pattern = r'^(.+?)_(\d+bars)_\d+_\d+$'
    match = re.match(pattern, stem)

    if match:
        piece_prefix = match.group(1)  # e.g., "Schubert_D960_mv3"
        bars = match.group(2)  # e.g., "8bars"
        return {
            'piece_prefix': piece_prefix,
            'bars': bars,
            'full_prefix': f"{piece_prefix}_{bars}",
        }

    return {'piece_prefix': stem, 'bars': '', 'full_prefix': stem}


def find_matching_score(
    midi_filename: str,
    score_files: Dict[str, Path],
) -> Optional[str]:
    """
    Find matching score file for a performance MIDI.

    Args:
        midi_filename: Name of MIDI file
        score_files: Dict mapping score prefixes to paths

    Returns:
        Relative path to score file or None
    """
    info = extract_piece_info(midi_filename)

    # Try exact prefix match
    prefix = info['full_prefix']

    # Score files use "_Score_" or "_Score2_" in their names
    for score_prefix, score_path in score_files.items():
        # Check if this score is for the same piece
        if prefix in score_prefix or score_prefix.startswith(prefix):
            return score_path.name

    # Try matching just the piece name (without segment)
    piece_prefix = info['piece_prefix']
    for score_prefix, score_path in score_files.items():
        if piece_prefix in score_prefix:
            return score_path.name

    return None


def index_score_files(score_dir: Path) -> Dict[str, Path]:
    """Create index of score files by their prefix."""
    scores = {}
    for score_file in score_dir.glob("*.musicxml"):
        # Remove "_Score_N" or "_Score2_N" suffix to get piece prefix
        stem = score_file.stem
        # Pattern: {piece}_Score{optional 2}_{segment}
        pattern = r'^(.+?)_Score2?_\d+$'
        match = re.match(pattern, stem)
        if match:
            prefix = match.group(1)
        else:
            prefix = stem
        scores[stem] = score_file
    return scores


def add_score_paths(data_dir: Path, score_dir: Path) -> Dict[str, int]:
    """
    Add score_path to all JSON files in data_dir.

    Returns:
        Statistics dict with counts
    """
    stats = {'total': 0, 'matched': 0, 'unmatched': 0}

    # Index score files
    print(f"Indexing score files from {score_dir}...")
    score_files = index_score_files(score_dir)
    print(f"Found {len(score_files)} score files")

    # Process each split
    for split in ['train', 'val', 'test']:
        json_path = data_dir / f'percepiano_{split}.json'
        if not json_path.exists():
            print(f"Skipping {split} (file not found)")
            continue

        with open(json_path) as f:
            samples = json.load(f)

        matched = 0
        for sample in samples:
            midi_filename = Path(sample['midi_path']).name
            score_path = find_matching_score(midi_filename, score_files)

            sample['score_path'] = score_path
            if score_path:
                matched += 1

        # Save updated file
        with open(json_path, 'w') as f:
            json.dump(samples, f, indent=2)

        print(f"{split}: {matched}/{len(samples)} samples matched with scores")
        stats['total'] += len(samples)
        stats['matched'] += matched
        stats['unmatched'] += len(samples) - matched

    return stats


def main():
    parser = argparse.ArgumentParser(description="Add score paths to PercePiano data")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing processed JSON files",
    )
    parser.add_argument(
        "--score-dir",
        type=Path,
        default=Path("data/raw/PercePiano/virtuoso/data/score_xml"),
        help="Directory containing MusicXML score files",
    )
    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"Error: Data directory not found: {args.data_dir}")
        return 1

    if not args.score_dir.exists():
        print(f"Error: Score directory not found: {args.score_dir}")
        return 1

    stats = add_score_paths(args.data_dir, args.score_dir)

    print("\n" + "="*50)
    print("Summary:")
    print(f"  Total samples: {stats['total']}")
    print(f"  Matched with scores: {stats['matched']} ({100*stats['matched']/stats['total']:.1f}%)")
    print(f"  Without scores: {stats['unmatched']}")
    print("="*50)

    return 0


if __name__ == "__main__":
    exit(main())
