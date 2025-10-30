#!/usr/bin/env python3
"""
Extract only the selected MAESTRO pieces from the full dataset zip.

This script reads maestro_selected.csv and extracts only those audio/MIDI files
from the maestro-v3.0.0.zip, saving significant disk space.
"""

import csv
import zipfile
from pathlib import Path
from typing import List, Tuple

def load_selected_files(csv_path: Path) -> Tuple[List[str], List[str]]:
    """Load audio and MIDI filenames from maestro_selected.csv."""
    audio_files = []
    midi_files = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            audio_files.append(row['audio_filename'])
            midi_files.append(row['midi_filename'])

    return audio_files, midi_files

def extract_selected_files(
    zip_path: Path,
    output_dir: Path,
    audio_files: List[str],
    midi_files: List[str]
) -> None:
    """Extract only the selected files from the MAESTRO zip."""

    all_files = set(audio_files + midi_files)
    extracted_count = 0
    total_files = len(all_files)

    print(f"Opening {zip_path.name}...")
    print(f"Will extract {total_files} files ({len(audio_files)} audio + {len(midi_files)} MIDI)")

    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Get all files in the zip
        zip_members = {member.filename: member for member in zf.infolist()}

        # Extract each selected file
        for target_file in sorted(all_files):
            # Try different possible paths in the zip
            possible_paths = [
                f"maestro-v3.0.0/{target_file}",
                target_file,
            ]

            extracted = False
            for zip_path_variant in possible_paths:
                if zip_path_variant in zip_members:
                    member = zip_members[zip_path_variant]

                    # Extract to output directory, preserving structure
                    # Remove the maestro-v3.0.0/ prefix if present
                    relative_path = target_file
                    output_path = output_dir / relative_path

                    # Create parent directories
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                    # Extract the file
                    with zf.open(member) as source, open(output_path, 'wb') as target:
                        target.write(source.read())

                    extracted_count += 1
                    file_type = "audio" if target_file in audio_files else "MIDI"
                    print(f"[{extracted_count}/{total_files}] Extracted {file_type}: {target_file}")

                    extracted = True
                    break

            if not extracted:
                print(f"WARNING: Could not find {target_file} in zip")

    print(f"\n✓ Extraction complete: {extracted_count}/{total_files} files extracted")

def main():
    # Setup paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"

    csv_path = data_dir / "maestro_selected.csv"
    zip_path = data_dir / "maestro-v3.0.0.zip"
    output_dir = data_dir / "maestro_selected"

    # Verify files exist
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip not found: {zip_path}")

    # Load selected files
    print(f"Reading selected files from {csv_path.name}...")
    audio_files, midi_files = load_selected_files(csv_path)
    print(f"Found {len(audio_files)} pieces to extract")

    # Extract files
    extract_selected_files(zip_path, output_dir, audio_files, midi_files)

    # Print final directory info
    print(f"\nExtracted files are in: {output_dir}")
    print("\nNext steps:")
    print("1. Verify extraction completed successfully")
    print("2. Delete maestro-v3.0.0.zip to free up 101 GB")
    print("3. Begin labeling with Label Studio")

if __name__ == "__main__":
    main()
