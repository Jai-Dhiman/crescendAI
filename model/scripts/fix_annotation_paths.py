#!/usr/bin/env python3
"""
Fix annotation file paths after downloading from Hugging Face Hub.

The annotation files contain hardcoded Google Drive paths like:
  /content/drive/MyDrive/crescendai_data/all_segments/...

This script updates them to local SSD paths:
  /tmp/crescendai_data/data/all_segments/...

Run this immediately after extracting the tar.gz archive in Colab.
"""

import json
import sys
from pathlib import Path
from typing import Optional


def fix_annotation_paths(
    annotation_path: str,
    old_prefix: str = "/content/drive/MyDrive/crescendai_data",
    new_prefix: str = "/tmp/crescendai_data/data",
    backup: bool = True,
) -> int:
    """
    Fix paths in a single annotation file.

    Args:
        annotation_path: Path to JSONL annotation file
        old_prefix: Old path prefix to replace (Google Drive)
        new_prefix: New path prefix (local SSD)
        backup: Create .bak backup before modifying

    Returns:
        Number of paths fixed
    """
    annotation_path = Path(annotation_path)

    if not annotation_path.exists():
        print(f"Error: {annotation_path} does not exist")
        return 0

    # Read all annotations
    print(f"Reading {annotation_path}...")
    annotations = []
    with open(annotation_path) as f:
        for line in f:
            if line.strip():
                annotations.append(json.loads(line))

    # Fix paths
    fixed_count = 0
    for ann in annotations:
        # Fix audio path
        if 'audio_path' in ann and old_prefix in ann['audio_path']:
            ann['audio_path'] = ann['audio_path'].replace(old_prefix, new_prefix)
            fixed_count += 1

        # Fix MIDI path
        if 'midi_path' in ann and ann['midi_path'] and old_prefix in ann['midi_path']:
            ann['midi_path'] = ann['midi_path'].replace(old_prefix, new_prefix)

    if fixed_count == 0:
        print(f"  No paths needed fixing (already correct)")
        return 0

    # Create backup
    if backup:
        backup_path = annotation_path.with_suffix('.jsonl.bak')
        if not backup_path.exists():
            annotation_path.rename(backup_path)
            print(f"  Created backup: {backup_path.name}")
        else:
            print(f"  Backup already exists, skipping")

    # Write fixed annotations
    with open(annotation_path, 'w') as f:
        for ann in annotations:
            f.write(json.dumps(ann) + '\n')

    print(f"  Fixed {fixed_count:,} paths")
    return fixed_count


def fix_all_annotations(
    annotations_dir: str = "/tmp/crescendai_data/data/annotations",
    old_prefix: str = "/content/drive/MyDrive/crescendai_data",
    new_prefix: str = "/tmp/crescendai_data/data",
) -> None:
    """Fix paths in all annotation files in a directory."""
    annotations_dir = Path(annotations_dir)

    if not annotations_dir.exists():
        print(f"Error: {annotations_dir} does not exist")
        print("\nMake sure you've downloaded and extracted the data first:")
        print("  1. Run the Hugging Face Hub download cell")
        print("  2. Extract the archive to /tmp/crescendai_data/")
        print("  3. Then run this script")
        sys.exit(1)

    print("="*70)
    print("FIXING ANNOTATION PATHS")
    print("="*70)
    print(f"\nOld prefix: {old_prefix}")
    print(f"New prefix: {new_prefix}\n")

    # Find all JSONL files (skip macOS metadata files starting with ._)
    jsonl_files = [
        f for f in annotations_dir.glob("*.jsonl")
        if not f.name.startswith("._")
    ]

    if not jsonl_files:
        print(f"No .jsonl files found in {annotations_dir}")
        sys.exit(1)

    print(f"Found {len(jsonl_files)} annotation files:\n")

    total_fixed = 0
    for jsonl_file in sorted(jsonl_files):
        fixed = fix_annotation_paths(
            str(jsonl_file),
            old_prefix=old_prefix,
            new_prefix=new_prefix,
        )
        total_fixed += fixed

    print("\n" + "="*70)
    if total_fixed > 0:
        print(f"✓ FIXED {total_fixed:,} PATHS")
        print("="*70)
        print("\nYou can now run the preflight check and training!")
    else:
        print("✓ ALL PATHS ALREADY CORRECT")
        print("="*70)
        print("\nNo changes needed. Ready to train!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fix annotation paths after downloading data"
    )
    parser.add_argument(
        '--annotations-dir',
        default='/tmp/crescendai_data/data/annotations',
        help='Directory containing annotation files'
    )
    parser.add_argument(
        '--old-prefix',
        default='/content/drive/MyDrive/crescendai_data',
        help='Old path prefix to replace'
    )
    parser.add_argument(
        '--new-prefix',
        default='/tmp/crescendai_data/data',
        help='New path prefix'
    )

    args = parser.parse_args()

    fix_all_annotations(
        annotations_dir=args.annotations_dir,
        old_prefix=args.old_prefix,
        new_prefix=args.new_prefix,
    )
