#!/usr/bin/env python3
"""
Diagnostic Script for I/O Errors

Investigates the root cause of file access issues during training.

Usage:
    python diagnose_io_errors.py --annotation-file path/to/annotations.jsonl
"""

import argparse
import json
import time
from pathlib import Path
from tqdm import tqdm


def check_file_accessibility(file_path: Path, num_retries: int = 3) -> dict:
    """
    Check if a file is accessible and determine the nature of any errors.

    Returns:
        dict with keys: 'accessible', 'error_type', 'error_message', 'retry_success'
    """
    result = {
        'accessible': False,
        'error_type': None,
        'error_message': None,
        'retry_success': False,
        'attempts_needed': 0
    }

    for attempt in range(num_retries):
        try:
            # Try to open and read a small chunk
            with open(file_path, 'rb') as f:
                f.read(1024)  # Read first 1KB

            result['accessible'] = True
            result['attempts_needed'] = attempt + 1

            if attempt > 0:
                result['retry_success'] = True

            return result

        except FileNotFoundError as e:
            result['error_type'] = 'FILE_NOT_FOUND'
            result['error_message'] = str(e)
            return result  # Don't retry file not found

        except PermissionError as e:
            result['error_type'] = 'PERMISSION_DENIED'
            result['error_message'] = str(e)
            return result  # Don't retry permission errors

        except OSError as e:
            result['error_type'] = 'IO_ERROR'
            result['error_message'] = str(e)

            if attempt < num_retries - 1:
                time.sleep(0.5)  # Wait before retry
                continue
            else:
                return result  # All retries exhausted

        except Exception as e:
            result['error_type'] = 'UNKNOWN'
            result['error_message'] = str(e)
            return result

    return result


def main():
    parser = argparse.ArgumentParser(description="Diagnose I/O errors in dataset")
    parser.add_argument(
        '--annotation-file',
        type=str,
        required=True,
        help='Path to annotation JSONL file'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum samples to check (default: all)'
    )
    parser.add_argument(
        '--check-midi',
        action='store_true',
        help='Also check MIDI files'
    )
    args = parser.parse_args()

    annotation_path = Path(args.annotation_file)

    if not annotation_path.exists():
        print(f"ERROR: Annotation file not found: {annotation_path}")
        return 1

    print("="*80)
    print("I/O ERROR DIAGNOSTIC")
    print("="*80)
    print(f"Annotation file: {annotation_path}")
    print(f"File size: {annotation_path.stat().st_size / 1e6:.2f} MB")
    print()

    # Load annotations
    print("Loading annotations...")
    annotations = []
    with open(annotation_path, 'r') as f:
        for line in f:
            if line.strip():
                annotations.append(json.loads(line))

    total_samples = len(annotations)
    print(f"Total samples in annotation file: {total_samples}")

    if args.max_samples:
        annotations = annotations[:args.max_samples]
        print(f"Testing first {len(annotations)} samples")

    print()

    # Statistics
    stats = {
        'total': 0,
        'audio_accessible': 0,
        'audio_not_found': 0,
        'audio_io_error': 0,
        'audio_permission_error': 0,
        'audio_other_error': 0,
        'audio_retry_success': 0,
        'midi_accessible': 0,
        'midi_not_found': 0,
        'midi_io_error': 0,
    }

    failed_files = []
    intermittent_files = []

    # Check each sample
    print("Checking file accessibility...")
    for idx, annotation in enumerate(tqdm(annotations, desc="Checking files")):
        stats['total'] += 1

        # Check audio file
        audio_path = Path(annotation['audio_path'])
        audio_result = check_file_accessibility(audio_path, num_retries=3)

        if audio_result['accessible']:
            stats['audio_accessible'] += 1
            if audio_result['retry_success']:
                stats['audio_retry_success'] += 1
                intermittent_files.append({
                    'path': str(audio_path),
                    'type': 'audio',
                    'attempts_needed': audio_result['attempts_needed']
                })
        else:
            if audio_result['error_type'] == 'FILE_NOT_FOUND':
                stats['audio_not_found'] += 1
            elif audio_result['error_type'] == 'IO_ERROR':
                stats['audio_io_error'] += 1
            elif audio_result['error_type'] == 'PERMISSION_DENIED':
                stats['audio_permission_error'] += 1
            else:
                stats['audio_other_error'] += 1

            failed_files.append({
                'sample_idx': idx,
                'path': str(audio_path),
                'type': 'audio',
                'error_type': audio_result['error_type'],
                'error_message': audio_result['error_message']
            })

        # Check MIDI file if requested
        if args.check_midi and 'midi_path' in annotation and annotation['midi_path']:
            midi_path = Path(annotation['midi_path'])
            midi_result = check_file_accessibility(midi_path, num_retries=3)

            if midi_result['accessible']:
                stats['midi_accessible'] += 1
            else:
                if midi_result['error_type'] == 'FILE_NOT_FOUND':
                    stats['midi_not_found'] += 1
                elif midi_result['error_type'] == 'IO_ERROR':
                    stats['midi_io_error'] += 1

                failed_files.append({
                    'sample_idx': idx,
                    'path': str(midi_path),
                    'type': 'midi',
                    'error_type': midi_result['error_type'],
                    'error_message': midi_result['error_message']
                })

    # Print results
    print("\n" + "="*80)
    print("DIAGNOSTIC RESULTS")
    print("="*80)
    print(f"\nSamples checked: {stats['total']}")
    print()

    print("AUDIO FILES:")
    print(f"  Accessible: {stats['audio_accessible']} ({100*stats['audio_accessible']/stats['total']:.1f}%)")
    print(f"  Not found: {stats['audio_not_found']}")
    print(f"  I/O errors: {stats['audio_io_error']}")
    print(f"  Permission errors: {stats['audio_permission_error']}")
    print(f"  Other errors: {stats['audio_other_error']}")
    print(f"  Intermittent (succeeded after retry): {stats['audio_retry_success']}")

    if args.check_midi:
        print()
        print("MIDI FILES:")
        print(f"  Accessible: {stats['midi_accessible']}")
        print(f"  Not found: {stats['midi_not_found']}")
        print(f"  I/O errors: {stats['midi_io_error']}")

    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)

    total_failed = len(failed_files)
    total_intermittent = len(intermittent_files)

    if total_failed == 0:
        print("\n✓ All files are accessible!")
        print("The I/O errors during training may be intermittent Google Drive issues.")
    else:
        print(f"\n✗ {total_failed} files have persistent access issues ({100*total_failed/stats['total']:.2f}%)")

        # Categorize issues
        if stats['audio_not_found'] > 0:
            print(f"\n⚠ {stats['audio_not_found']} files are MISSING (not uploaded to Google Drive)")
            print("  → These should have been filtered out by the filtering script")

        if stats['audio_io_error'] > 0:
            print(f"\n⚠ {stats['audio_io_error']} files have PERSISTENT I/O errors")
            print("  → This could indicate:")
            print("     - Corrupted files on Google Drive")
            print("     - Google Drive sync issues")
            print("     - Drive mount problems")

        if stats['audio_permission_error'] > 0:
            print(f"\n⚠ {stats['audio_permission_error']} files have PERMISSION errors")
            print("  → Check file permissions on Google Drive")

    if total_intermittent > 0:
        print(f"\n⚠ {total_intermittent} files are INTERMITTENT (accessible after retry)")
        print("  → This indicates Google Drive mount instability")
        print("  → Solutions: Increase retry attempts, reduce num_workers, or copy data locally")

    # Show failed files
    if total_failed > 0 and total_failed <= 20:
        print("\n" + "="*80)
        print("FAILED FILES")
        print("="*80)
        for failure in failed_files[:20]:
            print(f"\nSample {failure['sample_idx']}:")
            print(f"  Type: {failure['type']}")
            print(f"  Path: {failure['path']}")
            print(f"  Error: {failure['error_type']}")
            print(f"  Message: {failure['error_message']}")
    elif total_failed > 20:
        print(f"\n({total_failed} failed files - too many to display)")
        print(f"First 5 failures:")
        for failure in failed_files[:5]:
            print(f"  - {Path(failure['path']).name}: {failure['error_type']}")

    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    if total_failed > stats['total'] * 0.01:  # > 1% failure rate
        print("\n1. RE-RUN FILTERING SCRIPT")
        print("   The filtered annotation file still contains corrupted files.")
        print("   Run the filtering script again with more thorough checks.")

    if total_intermittent > 0:
        print("\n2. ADD RETRY LOGIC")
        print("   Google Drive mount has intermittent issues.")
        print("   Solutions:")
        print("   - Increase retry attempts in data loader")
        print("   - Reduce num_workers to decrease concurrent access")
        print("   - Add delays between retry attempts")

    if stats['audio_io_error'] > 0:
        print("\n3. CONSIDER ALTERNATIVE APPROACH")
        print("   If I/O errors persist:")
        print("   - Copy dataset to local Colab storage (/content/)")
        print("   - Use rsync to sync only valid files")
        print("   - Check Google Drive web interface for corrupted files")

    print("="*80)

    return 0 if total_failed == 0 else 1


if __name__ == '__main__':
    exit(main())
