#!/usr/bin/env python3
"""
Diagnostic script to identify MIDI loading failures.

This script tests MIDI loading with different approaches to find the root cause.
"""

import sys
import traceback
import numpy as np
import pretty_midi
from pathlib import Path


def test_midi_file(midi_path: str):
    """Test loading a single MIDI file with detailed diagnostics."""
    print(f"\n{'='*80}")
    print(f"Testing MIDI file: {midi_path}")
    print(f"{'='*80}")

    path = Path(midi_path)
    if not path.exists():
        print(f"❌ File does not exist!")
        return False

    print(f"✓ File exists ({path.stat().st_size} bytes)")

    # Test 1: Load with pretty_midi (default)
    print("\n[Test 1] Loading with pretty_midi.PrettyMIDI()...")
    try:
        midi = pretty_midi.PrettyMIDI(str(path))
        print(f"✓ Successfully loaded MIDI file")
        print(f"  - Duration: {midi.get_end_time():.2f} seconds")
        print(f"  - Instruments: {len(midi.instruments)}")
        print(f"  - Total notes: {sum(len(inst.notes) for inst in midi.instruments)}")

        # Test 2: Get tempo changes (this is where the error occurs)
        print("\n[Test 2] Getting tempo changes...")
        try:
            tempo_changes = midi.get_tempo_changes()
            tempo_times, tempos = tempo_changes
            print(f"✓ Tempo changes retrieved")
            print(f"  - tempo_times type: {type(tempo_times)}, shape: {tempo_times.shape if hasattr(tempo_times, 'shape') else 'N/A'}")
            print(f"  - tempos type: {type(tempos)}, shape: {tempos.shape if hasattr(tempos, 'shape') else 'N/A'}")
            print(f"  - tempo_times: {tempo_times}")
            print(f"  - tempos: {tempos}")

            # Check dimensions
            if hasattr(tempo_times, 'ndim') and hasattr(tempos, 'ndim'):
                if tempo_times.ndim != tempos.ndim:
                    print(f"⚠️  Dimension mismatch detected!")
                    print(f"  - tempo_times.ndim: {tempo_times.ndim}")
                    print(f"  - tempos.ndim: {tempos.ndim}")
                else:
                    print(f"✓ Dimensions match: both are {tempo_times.ndim}D arrays")

        except Exception as e:
            print(f"❌ Failed to get tempo changes: {e}")
            print(f"\nFull traceback:")
            traceback.print_exc()
            return False

        # Test 3: Get piano roll
        print("\n[Test 3] Getting piano roll...")
        try:
            piano_roll = midi.get_piano_roll(fs=100)
            print(f"✓ Piano roll generated: shape {piano_roll.shape}")
        except Exception as e:
            print(f"❌ Failed to get piano roll: {e}")

        # Test 4: Iterate through notes
        print("\n[Test 4] Iterating through notes...")
        try:
            for i, instrument in enumerate(midi.instruments):
                print(f"  Instrument {i}: {instrument.name or 'Unnamed'}")
                print(f"    - Program: {instrument.program}")
                print(f"    - Is drum: {instrument.is_drum}")
                print(f"    - Notes: {len(instrument.notes)}")
                if instrument.notes:
                    first_note = instrument.notes[0]
                    print(f"    - First note: pitch={first_note.pitch}, start={first_note.start:.2f}s, end={first_note.end:.2f}s, velocity={first_note.velocity}")
        except Exception as e:
            print(f"❌ Failed to iterate notes: {e}")

        return True

    except Exception as e:
        print(f"❌ Failed to load MIDI file: {e}")
        print(f"\nFull traceback:")
        traceback.print_exc()
        return False


def test_multiple_files(annotation_path: str, num_files: int = 10):
    """Test multiple MIDI files from annotation file."""
    import json

    print(f"\n{'='*80}")
    print(f"Testing {num_files} MIDI files from annotations")
    print(f"{'='*80}")

    with open(annotation_path, 'r') as f:
        annotations = [json.loads(line) for line in f if line.strip()]

    print(f"Total annotations: {len(annotations)}")

    # Test first N files
    success_count = 0
    fail_count = 0
    failed_files = []

    for i, ann in enumerate(annotations[:num_files]):
        if 'midi_path' in ann and ann['midi_path']:
            print(f"\n--- File {i+1}/{num_files} ---")
            success = test_midi_file(ann['midi_path'])
            if success:
                success_count += 1
            else:
                fail_count += 1
                failed_files.append(ann['midi_path'])

    print(f"\n{'='*80}")
    print(f"Summary: {success_count} succeeded, {fail_count} failed")
    print(f"{'='*80}")

    if failed_files:
        print("\nFailed files:")
        for f in failed_files:
            print(f"  - {f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Test single MIDI file:")
        print("    python diagnose_midi.py /path/to/file.midi")
        print("  Test multiple files from annotations:")
        print("    python diagnose_midi.py /path/to/annotations.jsonl --num 10")
        sys.exit(1)

    path = sys.argv[1]

    if path.endswith('.jsonl'):
        num = 10
        if '--num' in sys.argv:
            num = int(sys.argv[sys.argv.index('--num') + 1])
        test_multiple_files(path, num)
    else:
        test_midi_file(path)
