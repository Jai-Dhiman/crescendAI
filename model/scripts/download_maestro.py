#!/usr/bin/env python3
"""
MAESTRO Dataset Download Script

Downloads MAESTRO v3.0.0 dataset with support for subset selection.
Organizes files into audio/ and midi/ directories by year.
"""

import argparse
import csv
import hashlib
import json
import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

MAESTRO_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip"
MAESTRO_JSON_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.json"


def download_file(url: str, dest: Path, show_progress: bool = True) -> None:
    """Download file with progress reporting."""
    print(f"Downloading {url} to {dest}")

    def report_progress(block_num, block_size, total_size):
        if show_progress and total_size > 0:
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 / total_size)
            print(f"\rProgress: {percent:.1f}% ({downloaded / 1e9:.2f}GB / {total_size / 1e9:.2f}GB)", end="")

    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest, reporthook=report_progress if show_progress else None)
    if show_progress:
        print()  # New line after progress


def verify_checksum(file_path: Path, expected_md5: Optional[str] = None) -> bool:
    """Verify file integrity with MD5 checksum."""
    if expected_md5 is None:
        return True

    print(f"Verifying checksum for {file_path.name}...")
    md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            md5.update(chunk)

    actual_md5 = md5.hexdigest()
    if actual_md5 == expected_md5:
        print(f"Checksum verified: {actual_md5}")
        return True
    else:
        print(f"Checksum mismatch! Expected {expected_md5}, got {actual_md5}")
        return False


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract zip file."""
    print(f"Extracting {zip_path} to {extract_to}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete")


def organize_files(maestro_dir: Path, output_dir: Path, subset: Optional[int] = None) -> list:
    """
    Organize MAESTRO files into year/audio and year/midi directories.

    Args:
        maestro_dir: Path to extracted MAESTRO directory
        output_dir: Output directory for organized files
        subset: Number of pieces to include (None = all)

    Returns:
        List of metadata dictionaries for each piece
    """
    # Load MAESTRO metadata
    metadata_file = maestro_dir / "maestro-v3.0.0.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    with open(metadata_file, 'r') as f:
        maestro_data = json.load(f)

    # Filter to training set for pseudo-labeling
    pieces = [p for p in maestro_data if p.get('split') == 'train']

    if subset is not None:
        pieces = pieces[:subset]
        print(f"Using subset of {subset} pieces (out of {len(maestro_data)} total)")

    metadata = []

    for i, piece in enumerate(pieces, 1):
        # Parse paths
        audio_rel_path = piece['audio_filename']
        midi_rel_path = piece['midi_filename']

        audio_src = maestro_dir / audio_rel_path
        midi_src = maestro_dir / midi_rel_path

        if not audio_src.exists() or not midi_src.exists():
            print(f"Warning: Missing files for {piece.get('canonical_title', 'unknown')}, skipping")
            continue

        # Determine year and piece ID
        year = piece.get('year', 'unknown')
        canonical_title = piece.get('canonical_title', f'piece_{i}')
        piece_id = f"{year}_{canonical_title.replace(' ', '_').replace('/', '_')}"

        # Create output directories
        audio_dir = output_dir / str(year) / "audio"
        midi_dir = output_dir / str(year) / "midi"
        audio_dir.mkdir(parents=True, exist_ok=True)
        midi_dir.mkdir(parents=True, exist_ok=True)

        # Copy files
        audio_dest = audio_dir / audio_src.name
        midi_dest = midi_dir / midi_src.name

        if not audio_dest.exists():
            import shutil
            shutil.copy2(audio_src, audio_dest)

        if not midi_dest.exists():
            import shutil
            shutil.copy2(midi_src, midi_dest)

        # Add to metadata
        metadata.append({
            'piece_id': piece_id,
            'audio_path': str(audio_dest.relative_to(output_dir)),
            'midi_path': str(midi_dest.relative_to(output_dir)),
            'duration': piece.get('duration', 0.0),
            'canonical_composer': piece.get('canonical_composer', 'Unknown'),
            'canonical_title': canonical_title,
            'year': year,
            'split': piece.get('split', 'unknown')
        })

        print(f"Processed {i}/{len(pieces)}: {canonical_title}")

    return metadata


def save_metadata_csv(metadata: list, output_file: Path) -> None:
    """Save metadata to CSV file."""
    if not metadata:
        print("No metadata to save")
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', newline='') as f:
        fieldnames = ['piece_id', 'audio_path', 'midi_path', 'duration',
                      'canonical_composer', 'canonical_title', 'year', 'split']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata)

    print(f"Metadata saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Download and organize MAESTRO dataset')
    parser.add_argument('--output', type=str, default='data/maestro',
                        help='Output directory (default: data/maestro)')
    parser.add_argument('--subset', type=int, default=None,
                        help='Download subset of N pieces (default: all)')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip download if files already exist')
    parser.add_argument('--keep-zip', action='store_true',
                        help='Keep zip file after extraction')

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download paths
    download_dir = output_dir / 'downloads'
    download_dir.mkdir(parents=True, exist_ok=True)
    zip_path = download_dir / 'maestro-v3.0.0.zip'
    extract_dir = download_dir / 'extracted'

    # Step 1: Download
    if not args.skip_download or not zip_path.exists():
        download_file(MAESTRO_URL, zip_path)
    else:
        print(f"Using existing download: {zip_path}")

    # Step 2: Extract
    if not extract_dir.exists():
        extract_zip(zip_path, extract_dir)
    else:
        print(f"Using existing extraction: {extract_dir}")

    # Find maestro directory
    maestro_dir = extract_dir / 'maestro-v3.0.0'
    if not maestro_dir.exists():
        raise FileNotFoundError(f"MAESTRO directory not found: {maestro_dir}")

    # Step 3: Organize files
    metadata = organize_files(maestro_dir, output_dir, subset=args.subset)

    # Step 4: Save metadata
    metadata_csv = output_dir / 'metadata.csv'
    save_metadata_csv(metadata, metadata_csv)

    # Step 5: Cleanup
    if not args.keep_zip:
        print(f"Removing zip file: {zip_path}")
        zip_path.unlink()

    print("\n" + "="*60)
    print(f"MAESTRO download complete!")
    print(f"Pieces downloaded: {len(metadata)}")
    print(f"Output directory: {output_dir}")
    print(f"Metadata file: {metadata_csv}")
    print("="*60)


if __name__ == '__main__':
    main()
