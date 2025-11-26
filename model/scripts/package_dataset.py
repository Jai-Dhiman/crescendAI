#!/usr/bin/env python3
"""
Package segmented MAESTRO dataset into a tar.gz for upload.

Creates a compressed archive containing:
- audio/ - 10-second WAV segments (pristine only)
- midi/ - MIDI segments
- annotations/ - JSONL files with quality tier assignments

The dataloader applies degradation at runtime based on annotation params,
so we only need to store pristine audio (saves 4x space).

Usage:
    python scripts/package_dataset.py \
        --segments_dir /tmp/maestro_segments \
        --output_path /tmp/maestro_with_variance.tar.gz
"""

import argparse
import json
import os
import sys
import tarfile
from pathlib import Path
from typing import Optional

from tqdm import tqdm


def verify_dataset(segments_dir: Path) -> bool:
    """Verify that all necessary files exist."""
    print("Verifying dataset...")

    # Check directories
    audio_dir = segments_dir / "audio"
    midi_dir = segments_dir / "midi"
    annotations_dir = segments_dir / "annotations"

    if not audio_dir.exists():
        print(f"ERROR: {audio_dir} not found")
        return False

    if not annotations_dir.exists():
        print(f"ERROR: {annotations_dir} not found")
        print("Run generate_annotations_with_tiers.py first")
        return False

    # Check files
    audio_files = list(audio_dir.glob("*.wav"))
    midi_files = list(midi_dir.glob("*.mid")) if midi_dir.exists() else []
    annotation_files = list(annotations_dir.glob("*.jsonl"))

    print(f"  Audio files: {len(audio_files)}")
    print(f"  MIDI files: {len(midi_files)}")
    print(f"  Annotation files: {len(annotation_files)}")

    if len(audio_files) == 0:
        print("ERROR: No audio files found")
        return False

    if len(annotation_files) == 0:
        print("ERROR: No annotation files found")
        return False

    # Verify annotation files reference existing audio
    print("\nVerifying annotations...")
    missing_audio = 0
    missing_midi = 0
    total_samples = 0

    for ann_file in annotation_files:
        with open(ann_file) as f:
            for line in f:
                if not line.strip():
                    continue
                ann = json.loads(line)
                total_samples += 1

                # Check audio path
                audio_path = Path(ann['audio_path'])
                if not audio_path.exists():
                    missing_audio += 1
                    if missing_audio <= 3:
                        print(f"  Missing audio: {audio_path}")

                # Check MIDI path (optional)
                if ann.get('midi_path'):
                    midi_path = Path(ann['midi_path'])
                    if not midi_path.exists():
                        missing_midi += 1

    print(f"  Total samples: {total_samples}")
    if missing_audio > 0:
        print(f"  Missing audio files: {missing_audio}")
        return False
    if missing_midi > 0:
        print(f"  Missing MIDI files: {missing_midi} (warning only)")

    print("Verification passed!")
    return True


def update_annotation_paths(
    annotations_dir: Path,
    output_annotations_dir: Path,
    base_name: str = ""
) -> None:
    """
    Update annotation paths to be relative for portability.

    Changes absolute paths like /tmp/maestro_segments/audio/file.wav
    to relative paths like audio/file.wav
    """
    output_annotations_dir.mkdir(parents=True, exist_ok=True)

    for ann_file in annotations_dir.glob("*.jsonl"):
        output_file = output_annotations_dir / ann_file.name
        updated_annotations = []

        with open(ann_file) as f:
            for line in f:
                if not line.strip():
                    continue
                ann = json.loads(line)

                # Update audio path to relative
                if ann.get('audio_path'):
                    audio_path = Path(ann['audio_path'])
                    ann['audio_path'] = f"audio/{audio_path.name}"

                # Update MIDI path to relative
                if ann.get('midi_path'):
                    midi_path = Path(ann['midi_path'])
                    ann['midi_path'] = f"midi/{midi_path.name}"

                updated_annotations.append(ann)

        with open(output_file, 'w') as f:
            for ann in updated_annotations:
                f.write(json.dumps(ann) + '\n')


def calculate_sizes(segments_dir: Path) -> dict:
    """Calculate sizes of different components."""
    audio_dir = segments_dir / "audio"
    midi_dir = segments_dir / "midi"
    annotations_dir = segments_dir / "annotations"

    sizes = {}

    if audio_dir.exists():
        sizes['audio'] = sum(f.stat().st_size for f in audio_dir.glob("*.wav"))
        sizes['audio_count'] = len(list(audio_dir.glob("*.wav")))

    if midi_dir.exists():
        sizes['midi'] = sum(f.stat().st_size for f in midi_dir.glob("*.mid"))
        sizes['midi_count'] = len(list(midi_dir.glob("*.mid")))

    if annotations_dir.exists():
        sizes['annotations'] = sum(f.stat().st_size for f in annotations_dir.glob("*.jsonl"))
        sizes['annotation_count'] = len(list(annotations_dir.glob("*.jsonl")))

    return sizes


def package_dataset(
    segments_dir: Path,
    output_path: Path,
    compression: str = "gz",
    delete_after: bool = False,
) -> None:
    """
    Package dataset into tar.gz archive.

    Args:
        segments_dir: Directory containing audio/, midi/, annotations/
        output_path: Output tar.gz path
        compression: Compression type (gz, bz2, xz)
        delete_after: Delete source files after packaging
    """
    print("="*70)
    print("PACKAGING DATASET")
    print("="*70)

    # Verify
    if not verify_dataset(segments_dir):
        print("\nERROR: Dataset verification failed")
        sys.exit(1)

    # Calculate sizes before packaging
    sizes = calculate_sizes(segments_dir)
    print(f"\nSource sizes:")
    print(f"  Audio: {sizes.get('audio', 0) / 1e9:.2f} GB ({sizes.get('audio_count', 0)} files)")
    print(f"  MIDI: {sizes.get('midi', 0) / 1e6:.2f} MB ({sizes.get('midi_count', 0)} files)")
    print(f"  Annotations: {sizes.get('annotations', 0) / 1e3:.1f} KB ({sizes.get('annotation_count', 0)} files)")

    total_source = sum(v for k, v in sizes.items() if not k.endswith('_count'))
    print(f"  Total: {total_source / 1e9:.2f} GB")

    # Create temp directory for relative-path annotations
    import tempfile
    import shutil

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        tmp_annotations = tmp_path / "annotations"

        print("\nUpdating annotation paths for portability...")
        update_annotation_paths(
            segments_dir / "annotations",
            tmp_annotations
        )

        # Create tar archive
        print(f"\nCreating archive: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        mode = f"w:{compression}" if compression else "w"

        with tarfile.open(output_path, mode) as tar:
            # Add audio files
            audio_dir = segments_dir / "audio"
            audio_files = sorted(audio_dir.glob("*.wav"))
            print(f"Adding {len(audio_files)} audio files...")
            for audio_file in tqdm(audio_files, desc="Audio"):
                tar.add(audio_file, arcname=f"audio/{audio_file.name}")

            # Add MIDI files
            midi_dir = segments_dir / "midi"
            if midi_dir.exists():
                midi_files = sorted(midi_dir.glob("*.mid"))
                print(f"Adding {len(midi_files)} MIDI files...")
                for midi_file in tqdm(midi_files, desc="MIDI"):
                    tar.add(midi_file, arcname=f"midi/{midi_file.name}")

            # Add annotations (with relative paths)
            annotation_files = sorted(tmp_annotations.glob("*.jsonl"))
            print(f"Adding {len(annotation_files)} annotation files...")
            for ann_file in annotation_files:
                tar.add(ann_file, arcname=f"annotations/{ann_file.name}")

    # Report final size
    final_size = output_path.stat().st_size
    compression_ratio = total_source / final_size if final_size > 0 else 0

    print(f"\nArchive created: {output_path}")
    print(f"  Compressed size: {final_size / 1e9:.2f} GB")
    print(f"  Compression ratio: {compression_ratio:.1f}x")

    # Delete source files if requested
    if delete_after:
        print("\nDeleting source files to free space...")
        shutil.rmtree(segments_dir / "audio", ignore_errors=True)
        shutil.rmtree(segments_dir / "midi", ignore_errors=True)
        print("Source files deleted.")

    print("\n" + "="*70)
    print("PACKAGING COMPLETE")
    print("="*70)
    print(f"\nTo extract on remote runtime:")
    print(f"  tar -xzf {output_path.name} -C /tmp/maestro_data")


def main():
    parser = argparse.ArgumentParser(description="Package MAESTRO segments for upload")
    parser.add_argument('--segments_dir', type=str, required=True,
                        help='Directory containing audio/, midi/, annotations/')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Output tar.gz path (default: segments_dir/maestro_with_variance.tar.gz)')
    parser.add_argument('--compression', type=str, default='gz',
                        choices=['gz', 'bz2', 'xz', ''],
                        help='Compression type (default: gz)')
    parser.add_argument('--delete-after', action='store_true',
                        help='Delete source files after packaging (to free disk space)')

    args = parser.parse_args()

    segments_dir = Path(args.segments_dir)
    if not segments_dir.exists():
        print(f"ERROR: {segments_dir} not found")
        sys.exit(1)

    output_path = Path(args.output_path) if args.output_path else \
        segments_dir / "maestro_with_variance.tar.gz"

    package_dataset(
        segments_dir=segments_dir,
        output_path=output_path,
        compression=args.compression,
        delete_after=args.delete_after,
    )


if __name__ == "__main__":
    main()
