#!/usr/bin/env python3
"""
Copy training data from Google Drive to local Colab SSD.

This is the SINGLE BIGGEST optimization for Colab training:
- 10-50x faster I/O
- Can use num_workers > 0
- More reliable (no Drive flakiness)

Run this ONCE before training. Data persists for the Colab session.
"""

import json
import shutil
from pathlib import Path
from tqdm import tqdm
import sys


def get_file_size_mb(path):
    """Get file size in MB."""
    return path.stat().st_size / 1024 / 1024


def copy_data_to_local(
    drive_root="/content/drive/MyDrive/crescendai_data",
    local_root="/tmp/crescendai_data",
    create_subset=False,
    subset_size=10000,
):
    """
    Copy audio/MIDI data from Google Drive to local SSD.

    Args:
        drive_root: Root directory on Google Drive
        local_root: Destination on local SSD (/tmp/ or /content/)
        create_subset: Whether to create a random subset (for testing)
        subset_size: Number of training samples in subset
    """
    drive_root = Path(drive_root)
    local_root = Path(local_root)

    print("="*70)
    print("COPYING DATA FROM GOOGLE DRIVE TO LOCAL SSD")
    print("="*70)
    print(f"\nSource:      {drive_root}")
    print(f"Destination: {local_root}")
    print(f"Subset mode: {create_subset} ({subset_size:,} samples)" if create_subset else "")
    print()

    # Check if Drive is mounted
    if not drive_root.exists():
        print(f"✗ Google Drive not found at {drive_root}")
        print("  Please mount Drive first:")
        print("  from google.colab import drive")
        print("  drive.mount('/content/drive')")
        return False

    # Create local directory structure
    print("1. Creating local directory structure...")
    (local_root / 'all_segments').mkdir(parents=True, exist_ok=True)
    (local_root / 'all_segments/midi_segments').mkdir(parents=True, exist_ok=True)
    (local_root / 'annotations').mkdir(parents=True, exist_ok=True)
    print("  ✓ Directories created")

    # Copy annotation files
    print("\n2. Copying annotation files...")
    annotation_files = {}

    for split in ['train', 'val', 'test']:
        src = drive_root / 'annotations' / f'synthetic_{split}_filtered.jsonl'
        dst = local_root / 'annotations' / f'synthetic_{split}_filtered.jsonl'

        if not src.exists():
            print(f"  ⚠ {split}: Not found at {src}")
            continue

        shutil.copy2(src, dst)
        annotation_files[split] = dst
        print(f"  ✓ {split}: {get_file_size_mb(dst):.1f} MB")

    if not annotation_files:
        print("  ✗ No annotation files found!")
        return False

    # Create subset if requested
    if create_subset and 'train' in annotation_files:
        print(f"\n3. Creating {subset_size:,}-sample training subset...")
        import random
        random.seed(42)

        # Load full training set
        with open(annotation_files['train']) as f:
            train_data = [json.loads(line) for line in f if line.strip()]

        original_count = len(train_data)
        print(f"  Original: {original_count:,} samples")

        # Random sample
        if subset_size < len(train_data):
            train_data = random.sample(train_data, subset_size)
            print(f"  Subset:   {len(train_data):,} samples ({len(train_data)/original_count*100:.1f}%)")

            # Save subset
            with open(annotation_files['train'], 'w') as f:
                for item in train_data:
                    f.write(json.dumps(item) + '\n')
        else:
            print(f"  Using all {len(train_data):,} samples (requested size >= available)")

    # Collect unique file paths from all annotations
    print(f"\n4. Collecting file list from annotations...")
    audio_files = set()
    midi_files = set()

    for split, ann_file in annotation_files.items():
        with open(ann_file) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    # Resolve paths relative to Drive root
                    audio_path = Path(data['audio_path'])
                    if not audio_path.is_absolute():
                        audio_path = drive_root / audio_path
                    audio_files.add(audio_path)

                    if 'midi_path' in data and data['midi_path']:
                        midi_path = Path(data['midi_path'])
                        if not midi_path.is_absolute():
                            midi_path = drive_root / midi_path
                        midi_files.add(midi_path)

    print(f"  Found {len(audio_files):,} unique audio files")
    print(f"  Found {len(midi_files):,} unique MIDI files")

    # Estimate sizes (skip if Drive is having issues)
    try:
        sample_audio = list(audio_files)[0] if audio_files else None
        if sample_audio and sample_audio.exists():
            avg_audio_size = get_file_size_mb(sample_audio)
            total_audio_mb = avg_audio_size * len(audio_files)
            print(f"  Estimated audio size: ~{total_audio_mb:.0f} MB ({avg_audio_size:.1f} MB/file)")
    except (OSError, IOError) as e:
        print(f"  ⚠ Could not estimate size (Drive I/O issue): {e}")
        print(f"  Proceeding with copy anyway...")

    # Copy audio files with retry logic for Drive flakiness
    print(f"\n5. Copying {len(audio_files):,} audio files...")
    copied_audio = 0
    skipped_audio = 0
    failed_audio = 0

    for audio_path in tqdm(list(audio_files), desc="  Audio", unit="files"):
        max_retries = 3
        retry_delay = 0.5

        for attempt in range(max_retries):
            try:
                # Calculate relative path from Drive root
                if drive_root in audio_path.parents:
                    rel_path = audio_path.relative_to(drive_root)
                else:
                    # Path might already be relative
                    rel_path = audio_path

                local_path = local_root / rel_path
                local_path.parent.mkdir(parents=True, exist_ok=True)

                # Check if already exists (skip)
                if local_path.exists():
                    skipped_audio += 1
                    break

                # Try to copy (may fail due to Drive I/O)
                if audio_path.exists():
                    shutil.copy2(audio_path, local_path)
                    copied_audio += 1
                    break
                else:
                    # File doesn't exist, don't retry
                    if attempt == 0:  # Only warn once
                        print(f"\n  ⚠ Audio not found: {audio_path}")
                    failed_audio += 1
                    break

            except (OSError, IOError) as e:
                # I/O error (Drive flakiness), retry
                if attempt < max_retries - 1:
                    import time
                    time.sleep(retry_delay)
                    continue
                else:
                    # All retries failed
                    print(f"\n  ✗ Failed to copy after {max_retries} attempts: {audio_path.name[:60]}...")
                    failed_audio += 1
                    break
            except Exception as e:
                # Other errors, don't retry
                print(f"\n  ✗ Error copying {audio_path.name[:60]}...: {e}")
                failed_audio += 1
                break

    print(f"  ✓ Copied: {copied_audio:,}, Skipped: {skipped_audio:,}, Failed: {failed_audio:,}")

    # Copy MIDI files with retry logic
    print(f"\n6. Copying {len(midi_files):,} MIDI files...")
    copied_midi = 0
    skipped_midi = 0
    failed_midi = 0

    for midi_path in tqdm(list(midi_files), desc="  MIDI", unit="files"):
        max_retries = 3
        retry_delay = 0.5

        for attempt in range(max_retries):
            try:
                # Calculate relative path
                if drive_root in midi_path.parents:
                    rel_path = midi_path.relative_to(drive_root)
                else:
                    rel_path = midi_path

                local_path = local_root / rel_path
                local_path.parent.mkdir(parents=True, exist_ok=True)

                # Check if already exists
                if local_path.exists():
                    skipped_midi += 1
                    break

                # Try to copy
                if midi_path.exists():
                    shutil.copy2(midi_path, local_path)
                    copied_midi += 1
                    break
                else:
                    if attempt == 0:
                        print(f"\n  ⚠ MIDI not found: {midi_path}")
                    failed_midi += 1
                    break

            except (OSError, IOError) as e:
                # I/O error, retry
                if attempt < max_retries - 1:
                    import time
                    time.sleep(retry_delay)
                    continue
                else:
                    print(f"\n  ✗ Failed to copy after {max_retries} attempts: {midi_path.name[:60]}...")
                    failed_midi += 1
                    break
            except Exception as e:
                print(f"\n  ✗ Error copying {midi_path.name[:60]}...: {e}")
                failed_midi += 1
                break

    print(f"  ✓ Copied: {copied_midi:,}, Skipped: {skipped_midi:,}, Failed: {failed_midi:,}")

    # Update annotation paths to point to local files
    print("\n7. Updating annotation paths to local files...")
    for split, ann_file in annotation_files.items():
        # Read annotations
        with open(ann_file) as f:
            annotations = [json.loads(line) for line in f if line.strip()]

        # Update paths
        updated = 0
        for ann in annotations:
            # Update audio path
            old_audio = Path(ann['audio_path'])
            if drive_root in old_audio.parents:
                rel_path = old_audio.relative_to(drive_root)
            else:
                rel_path = old_audio
            new_audio = local_root / rel_path
            ann['audio_path'] = str(new_audio)

            # Update MIDI path
            if 'midi_path' in ann and ann['midi_path']:
                old_midi = Path(ann['midi_path'])
                if drive_root in old_midi.parents:
                    rel_path = old_midi.relative_to(drive_root)
                else:
                    rel_path = old_midi
                new_midi = local_root / rel_path
                ann['midi_path'] = str(new_midi)

            updated += 1

        # Write updated annotations
        with open(ann_file, 'w') as f:
            for ann in annotations:
                f.write(json.dumps(ann) + '\n')

        print(f"  ✓ {split}: Updated {updated:,} paths")

    # Calculate total size
    print("\n8. Verifying copied data...")
    total_size = 0
    file_count = 0

    for root, dirs, files in (local_root / 'all_segments').walk():
        for file in files:
            file_path = root / file
            total_size += file_path.stat().st_size
            file_count += 1

    print(f"  ✓ Total files: {file_count:,}")
    print(f"  ✓ Total size:  {total_size / 1024 / 1024 / 1024:.2f} GB")
    print(f"  ✓ Location:    {local_root}")

    # Final summary
    print("\n" + "="*70)

    # Check if copy was successful
    total_files = len(audio_files) + len(midi_files)
    copied_files = copied_audio + copied_midi
    failed_files = failed_audio + failed_midi
    success_rate = copied_files / total_files if total_files > 0 else 0

    if success_rate >= 0.95:
        print("✓ DATA COPY COMPLETE")
        success = True
    elif success_rate >= 0.80:
        print("⚠ DATA COPY PARTIALLY COMPLETE")
        print(f"  {failed_files:,}/{total_files:,} files failed ({failed_files/total_files*100:.1f}%)")
        print(f"  Training may still work, but some samples may fail to load")
        success = True  # Still allow training
    else:
        print("✗ DATA COPY FAILED")
        print(f"  Only {copied_files:,}/{total_files:,} files copied ({success_rate*100:.1f}%)")
        print(f"  Too many failures - Google Drive may be having issues")
        success = False

    print("="*70)
    print(f"\nLocal annotation files:")
    for split, ann_file in annotation_files.items():
        print(f"  {split:5s}: {ann_file}")

    print(f"\nCopy statistics:")
    print(f"  Audio: {copied_audio:,} copied, {skipped_audio:,} skipped, {failed_audio:,} failed")
    print(f"  MIDI:  {copied_midi:,} copied, {skipped_midi:,} skipped, {failed_midi:,} failed")
    print(f"  Total: {copied_files:,}/{total_files:,} files ({success_rate*100:.1f}%)")

    if success:
        print(f"\nNext steps:")
        print(f"  1. Run preflight check:")
        print(f"     python scripts/preflight_check.py --config configs/experiment_10k.yaml")
        print(f"  2. Train models (now 6x faster!):")
        print(f"     python train.py --config configs/experiment_10k.yaml --mode audio")
    else:
        print(f"\nTroubleshooting:")
        print(f"  1. Check Google Drive connection")
        print(f"  2. Try restarting Colab runtime")
        print(f"  3. Re-run this script (it will resume where it left off)")

    return success


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Copy training data from Drive to local SSD')
    parser.add_argument(
        '--drive-root',
        type=str,
        default='/content/drive/MyDrive/crescendai_data',
        help='Root directory on Google Drive'
    )
    parser.add_argument(
        '--local-root',
        type=str,
        default='/tmp/crescendai_data',
        help='Destination directory on local SSD'
    )
    parser.add_argument(
        '--subset',
        action='store_true',
        help='Create training subset (for testing)'
    )
    parser.add_argument(
        '--subset-size',
        type=int,
        default=10000,
        help='Number of training samples in subset (default: 10000)'
    )

    args = parser.parse_args()

    success = copy_data_to_local(
        drive_root=args.drive_root,
        local_root=args.local_root,
        create_subset=args.subset,
        subset_size=args.subset_size,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
