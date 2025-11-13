#!/usr/bin/env python3
"""
Google Drive Mount Diagnostic Script

Run this in Colab to diagnose file access issues before training.
"""

import os
import time
from pathlib import Path
import json


def check_drive_mount():
    """Check if Google Drive is properly mounted."""
    drive_path = Path("/content/drive")
    mydrive_path = Path("/content/drive/MyDrive")

    print("="*80)
    print("GOOGLE DRIVE MOUNT CHECK")
    print("="*80)

    if not drive_path.exists():
        print("✗ Google Drive not mounted at /content/drive")
        print("\nRun this in your Colab notebook:")
        print("  from google.colab import drive")
        print("  drive.mount('/content/drive')")
        return False

    print("✓ Google Drive mounted at /content/drive")

    if not mydrive_path.exists():
        print("✗ MyDrive folder not found")
        return False

    print("✓ MyDrive folder accessible")
    return True


def check_data_directory():
    """Check if data directory exists and is accessible."""
    data_path = Path("/content/drive/MyDrive/crescendai_data")
    segments_path = data_path / "all_segments"
    annotations_path = data_path / "annotations"

    print("\n" + "="*80)
    print("DATA DIRECTORY CHECK")
    print("="*80)

    if not data_path.exists():
        print(f"✗ Data directory not found: {data_path}")
        return False

    print(f"✓ Data directory exists: {data_path}")

    if not segments_path.exists():
        print(f"✗ Segments directory not found: {segments_path}")
        return False

    print(f"✓ Segments directory exists: {segments_path}")

    if not annotations_path.exists():
        print(f"✗ Annotations directory not found: {annotations_path}")
        return False

    print(f"✓ Annotations directory exists: {annotations_path}")

    return True


def test_file_access_speed():
    """Test file access speed and detect throttling."""
    segments_path = Path("/content/drive/MyDrive/crescendai_data/all_segments")

    print("\n" + "="*80)
    print("FILE ACCESS SPEED TEST")
    print("="*80)

    try:
        files = list(segments_path.glob("*.wav"))[:20]
        print(f"Found {len(files)} WAV files to test")

        if not files:
            print("✗ No WAV files found in segments directory")
            return False

        access_times = []
        errors = 0

        for i, file_path in enumerate(files):
            start = time.time()
            try:
                exists = file_path.exists()
                elapsed = time.time() - start
                access_times.append(elapsed)

                if elapsed > 1.0:
                    print(f"  ⚠ Slow access ({elapsed:.2f}s): {file_path.name}")
            except OSError as e:
                errors += 1
                print(f"  ✗ I/O error: {file_path.name}: {e}")

        if access_times:
            avg_time = sum(access_times) / len(access_times)
            max_time = max(access_times)

            print(f"\nAccess Statistics:")
            print(f"  Average time: {avg_time*1000:.1f}ms")
            print(f"  Max time: {max_time*1000:.1f}ms")
            print(f"  Errors: {errors}/{len(files)}")

            if avg_time > 0.5:
                print("\n  ⚠ WARNING: Slow file access detected")
                print("     This may indicate Google Drive throttling or poor connection")
            elif errors > 0:
                print("\n  ⚠ WARNING: I/O errors detected")
                print("     Files may be corrupted or Drive sync is incomplete")
            else:
                print("\n  ✓ File access speed is good")

        return errors == 0

    except Exception as e:
        print(f"✗ Error during test: {e}")
        return False


def test_annotation_file():
    """Test reading annotation file and verify paths."""
    train_ann = Path("/content/drive/MyDrive/crescendai_data/annotations/synthetic_train.jsonl")

    print("\n" + "="*80)
    print("ANNOTATION FILE CHECK")
    print("="*80)

    if not train_ann.exists():
        print(f"✗ Training annotation file not found: {train_ann}")
        return False

    print(f"✓ Training annotation file exists")

    try:
        with open(train_ann, 'r') as f:
            lines = [line for line in f if line.strip()]
            print(f"✓ {len(lines)} annotations loaded")

            # Check first 5 samples
            print("\nChecking first 5 samples:")
            for i, line in enumerate(lines[:5]):
                ann = json.loads(line)
                audio_path = Path(ann['audio_path'])
                midi_path = Path(ann.get('midi_path', ''))

                audio_exists = audio_path.exists() if audio_path else False
                midi_exists = midi_path.exists() if midi_path else False

                status = "✓" if audio_exists else "✗"
                print(f"  {status} Sample {i+1}: audio={audio_exists}, midi={midi_exists}")

                if not audio_exists:
                    print(f"     Missing: {audio_path}")

        return True

    except Exception as e:
        print(f"✗ Error reading annotation file: {e}")
        return False


def provide_recommendations():
    """Provide recommendations based on test results."""
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    print("""
If you encountered I/O errors or slow access:

1. REMOUNT GOOGLE DRIVE (Most Common Fix)
   from google.colab import drive
   drive.mount('/content/drive', force_remount=True)

2. RESTART RUNTIME
   Runtime → Restart runtime
   Then remount Drive

3. REDUCE SAMPLE SIZE IN PREFLIGHT CHECK
   python preflight_check.py --config config.yaml --skip-files

   Or modify check_annotation_file() to use sample_size=10 instead of 100

4. CHECK GOOGLE DRIVE QUOTA
   Make sure you haven't exceeded your Drive storage or API limits

5. COPY FILES TO COLAB LOCAL STORAGE (Fastest Option)
   !mkdir -p /content/data
   !cp -r /content/drive/MyDrive/crescendai_data /content/data/

   Then update annotation paths to use /content/data instead of /content/drive

6. USE FEWER WORKERS IN DATALOADER
   Set num_workers: 0 or 1 in your config to reduce concurrent file access
""")


def main():
    """Run all diagnostic checks."""
    print("\n" + "="*80)
    print("CRESCENDAI GOOGLE DRIVE DIAGNOSTIC TOOL")
    print("="*80 + "\n")

    all_passed = True

    all_passed &= check_drive_mount()
    all_passed &= check_data_directory()
    all_passed &= test_file_access_speed()
    all_passed &= test_annotation_file()

    provide_recommendations()

    print("\n" + "="*80)
    if all_passed:
        print("✓ ALL CHECKS PASSED - You should be able to train")
    else:
        print("✗ SOME CHECKS FAILED - Review recommendations above")
    print("="*80 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
