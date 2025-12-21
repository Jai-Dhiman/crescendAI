#!/usr/bin/env python3
"""
Prepare PercePiano data for Google Drive upload.

Creates a directory structure ready for upload to Thunder Compute.

Usage:
    python scripts/prepare_gdrive_upload.py

Output:
    Creates 'gdrive_upload/percepiano_data/' containing everything needed.
"""

import shutil
import json
from pathlib import Path


def main():
    # Paths
    project_root = Path(__file__).parent.parent
    processed_dir = project_root / "data" / "processed"
    percepiano_repo = project_root / "data" / "raw" / "PercePiano"
    output_dir = project_root / "gdrive_upload" / "percepiano_data"

    print("=" * 60)
    print("Preparing PercePiano data for Google Drive upload")
    print("=" * 60)

    # Verify source files exist
    required_files = [
        processed_dir / "percepiano_train.json",
        processed_dir / "percepiano_val.json",
        processed_dir / "percepiano_test.json",
        percepiano_repo / "virtuoso" / "data" / "all_2rounds",
    ]

    for f in required_files:
        if not f.exists():
            print(f"ERROR: Required file/directory not found: {f}")
            print("Run 'python scripts/prepare_percepiano.py' first")
            return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Copy JSON files
    print("\nCopying JSON files...")
    for split in ["train", "val", "test"]:
        src = processed_dir / f"percepiano_{split}.json"
        dst = output_dir / f"percepiano_{split}.json"
        shutil.copy2(src, dst)
        print(f"  {src.name} -> {dst}")

    # Copy PercePiano repository (just the MIDI files we need)
    midi_src = percepiano_repo / "virtuoso" / "data" / "all_2rounds"
    midi_dst = output_dir / "PercePiano" / "virtuoso" / "data" / "all_2rounds"

    print(f"\nCopying MIDI files from {midi_src}...")
    if midi_dst.exists():
        shutil.rmtree(midi_dst)
    shutil.copytree(midi_src, midi_dst)

    midi_count = len(list(midi_dst.glob("*.mid")))
    print(f"  Copied {midi_count} MIDI files")

    # Calculate total size
    total_size = 0
    for f in output_dir.rglob("*"):
        if f.is_file():
            total_size += f.stat().st_size

    print(f"\n{'=' * 60}")
    print("UPLOAD INSTRUCTIONS")
    print(f"{'=' * 60}")
    print(f"\nTotal size: {total_size / (1024*1024):.1f} MB")
    print(f"\nUpload this folder to your Google Drive root:")
    print(f"  {output_dir}")
    print(f"\nThe folder should appear as:")
    print(f"  gdrive:percepiano_data/")
    print(f"    percepiano_train.json")
    print(f"    percepiano_val.json")
    print(f"    percepiano_test.json")
    print(f"    PercePiano/")
    print(f"      virtuoso/")
    print(f"        data/")
    print(f"          all_2rounds/")
    print(f"            *.mid ({midi_count} files)")

    print(f"\n{'=' * 60}")
    print("After upload, run the notebook on Thunder Compute:")
    print("  notebooks/train_midi_only_percepiano.ipynb")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
