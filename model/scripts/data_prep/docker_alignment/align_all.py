#!/usr/bin/env python3
"""
Align all performance MIDIs with their corresponding score MIDIs.
This runs inside the Docker container with the Linux alignment tool.
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from tqdm import tqdm

# Paths inside container
ALIGN_TOOL = "/alignment"
DATA_DIR = "/data"
INPUT_DIR = f"{DATA_DIR}/all_2rounds"
SCORE_MIDI_DIR = f"{DATA_DIR}/score_midi"
# Labels file is in parent of data directory, mounted at /labels
LABELS_FILE = "/labels/label_2round_mean_reg_19_with0_rm_highstd0.json"


def get_score_midi_path(perform_path: str) -> str:
    """Get the score MIDI path for a performance MIDI."""
    basename = os.path.basename(perform_path)
    parts = basename.split("_")
    # Convert: Composer_Piece_bars_performer_segment.mid -> Composer_Piece_bars_Score_segment.mid
    score_name = "_".join(parts[:-2]) + "_Score_" + parts[-1]
    return os.path.join(SCORE_MIDI_DIR, score_name)


def align_midi_pair(perform_path: str, score_midi_path: str) -> bool:
    """
    Align a performance MIDI with its score MIDI using Nakamura's tool.

    Returns True if successful, False otherwise.
    """
    try:
        # Copy files to alignment tool directory
        shutil.copy(perform_path, os.path.join(ALIGN_TOOL, 'infer.mid'))
        shutil.copy(score_midi_path, os.path.join(ALIGN_TOOL, 'score.mid'))

        # Change to alignment tool directory
        original_dir = os.getcwd()
        os.chdir(ALIGN_TOOL)

        # Run alignment
        result = subprocess.run(
            ["bash", "MIDIToMIDIAlign.sh", "score", "infer"],
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout per file
        )

        os.chdir(original_dir)

        if result.returncode != 0:
            print(f"  Alignment failed: {result.stderr[:200]}")
            return False

        # Move result files to input directory
        corresp_file = os.path.join(ALIGN_TOOL, 'infer_corresp.txt')
        match_file = os.path.join(ALIGN_TOOL, 'infer_match.txt')
        spr_file = os.path.join(ALIGN_TOOL, 'infer_spr.txt')

        if os.path.exists(corresp_file):
            dest = perform_path.replace('.mid', '_infer_corresp.txt')
            shutil.move(corresp_file, dest)
        else:
            print("  No corresp file generated")
            return False

        if os.path.exists(match_file):
            dest = perform_path.replace('.mid', '_infer_match.txt')
            shutil.move(match_file, dest)

        if os.path.exists(spr_file):
            dest = perform_path.replace('.mid', '_infer_spr.txt')
            shutil.move(spr_file, dest)

        return True

    except subprocess.TimeoutExpired:
        print("  Alignment timed out")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


def main():
    print("=" * 60)
    print("MIDI-to-MIDI Alignment (Nakamura Tool)")
    print("=" * 60)

    # Load labels to filter valid files
    if not os.path.exists(LABELS_FILE):
        print(f"Error: Labels file not found: {LABELS_FILE}")
        sys.exit(1)

    with open(LABELS_FILE) as f:
        labels = json.load(f)
    label_keys = set(labels.keys())

    # Find all performance MIDIs
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory not found: {INPUT_DIR}")
        sys.exit(1)

    midi_files = sorted(Path(INPUT_DIR).glob("*.mid"))
    print(f"Found {len(midi_files)} MIDI files")

    # Filter to files with labels
    valid_files = []
    for f in midi_files:
        file_id = f.stem  # filename without extension
        if file_id in label_keys:
            valid_files.append(str(f))
    print(f"Files with labels: {len(valid_files)}")

    # Process each file
    success_count = 0
    skip_count = 0
    fail_count = 0

    for perform_path in tqdm(valid_files, desc="Aligning"):
        # Check if already aligned
        corresp_path = perform_path.replace('.mid', '_infer_corresp.txt')
        if os.path.exists(corresp_path):
            skip_count += 1
            continue

        # Get score MIDI path
        score_midi_path = get_score_midi_path(perform_path)
        if not os.path.exists(score_midi_path):
            print(f"\nScore MIDI not found: {score_midi_path}")
            fail_count += 1
            continue

        # Run alignment
        if align_midi_pair(perform_path, score_midi_path):
            success_count += 1
        else:
            fail_count += 1

    print("\n" + "=" * 60)
    print("ALIGNMENT COMPLETE")
    print("=" * 60)
    print(f"  Success: {success_count}")
    print(f"  Skipped (already aligned): {skip_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Total: {success_count + skip_count + fail_count}")


if __name__ == "__main__":
    main()
