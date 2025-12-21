#!/usr/bin/env python3
"""
Run original PercePiano VirtuosoNet preprocessing.

This script:
1. Sets up paths to match the original PercePiano expectations
2. Runs the feature extraction from the original codebase
3. Saves output to data/percepiano_vnet/

Usage:
    cd model
    uv run python scripts/data_prep/run_percepiano_preprocessing.py
"""

import os
import sys
import json
import pickle
import shutil
from pathlib import Path
from tqdm import tqdm

# Paths
MODEL_ROOT = Path(__file__).parent.parent.parent
PERCEPIANO_ROOT = MODEL_ROOT / "data" / "raw" / "PercePiano"
VIRTUOSO_ROOT = PERCEPIANO_ROOT / "virtuoso" / "virtuoso"
PYSCORE_ROOT = VIRTUOSO_ROOT / "pyScoreParser"

# Add pyScoreParser to Python path
sys.path.insert(0, str(PYSCORE_ROOT))

# Data paths
DATA_PATH = PERCEPIANO_ROOT / "virtuoso" / "data"
LABEL_FILE = PERCEPIANO_ROOT / "label_2round_mean_reg_19_with0_rm_highstd0.json"
OUTPUT_DIR = MODEL_ROOT / "data" / "percepiano_vnet"


def check_dependencies():
    """Check if required dependencies are available."""
    print("Checking dependencies...")

    try:
        import mido
        print(f"  [OK] mido")
    except ImportError:
        print("  [MISSING] mido - run: uv add mido")
        return False

    try:
        import pretty_midi
        print(f"  [OK] pretty_midi")
    except ImportError:
        print("  [MISSING] pretty_midi - run: uv add pretty_midi")
        return False

    try:
        import pandas
        print(f"  [OK] pandas")
    except ImportError:
        print("  [MISSING] pandas - run: uv add pandas")
        return False

    return True


def check_data_files():
    """Check if required data files exist."""
    print("\nChecking data files...")

    # Check label file
    if not LABEL_FILE.exists():
        print(f"  [MISSING] Label file: {LABEL_FILE}")
        return False
    print(f"  [OK] Label file: {LABEL_FILE}")

    # Check MIDI directory
    midi_dir = DATA_PATH / "all_2rounds"
    if not midi_dir.exists():
        print(f"  [MISSING] MIDI directory: {midi_dir}")
        return False
    midi_files = list(midi_dir.glob("*.mid"))
    print(f"  [OK] MIDI directory: {len(midi_files)} files")

    # Check score MIDI directory
    score_midi_dir = DATA_PATH / "score_midi"
    if not score_midi_dir.exists():
        print(f"  [MISSING] Score MIDI directory: {score_midi_dir}")
        return False
    score_files = list(score_midi_dir.glob("*.mid"))
    print(f"  [OK] Score MIDI directory: {len(score_files)} files")

    # Check score XML directory
    score_xml_dir = DATA_PATH / "score_xml"
    if not score_xml_dir.exists():
        print(f"  [MISSING] Score XML directory: {score_xml_dir}")
        return False
    xml_files = list(score_xml_dir.glob("*.musicxml"))
    print(f"  [OK] Score XML directory: {len(xml_files)} files")

    return True


def run_preprocessing():
    """Run the original PercePiano preprocessing."""
    print("\n" + "=" * 60)
    print("Running PercePiano VirtuosoNet Preprocessing")
    print("=" * 60)

    # Import PercePiano modules (now that path is set up)
    try:
        from data_class import (
            DataSet,
            DEFAULT_SCORE_FEATURES,
            DEFAULT_PERFORM_FEATURES,
            DEFAULT_PURE_PERFORM_FEATURES,
            PieceData,
        )
        from data_for_training import PairDataset
        from m2pf_dataset_performerfold import M2PFSet
        print("\n[OK] Successfully imported PercePiano modules")
    except ImportError as e:
        print(f"\n[ERROR] Failed to import PercePiano modules: {e}")
        print("\nTrying to diagnose the issue...")

        # Try importing individual modules
        try:
            from musicxml_parser import MusicXMLDocument
            print("  [OK] musicxml_parser")
        except ImportError as e2:
            print(f"  [FAIL] musicxml_parser: {e2}")

        try:
            from midi_utils import midi_utils
            print("  [OK] midi_utils")
        except ImportError as e2:
            print(f"  [FAIL] midi_utils: {e2}")

        return False

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run preprocessing
    print(f"\nData path: {DATA_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")

    try:
        # Load dataset using M2PFSet
        print("\nLoading dataset...")
        dataset = M2PFSet(
            path=str(DATA_PATH),
            split="all_2rounds",
            save=True,  # Must be True to create .dat cache files on first run
        )
        print(f"Loaded {len(dataset.pieces)} pieces")

        # Extract features for each piece
        print("\nExtracting features...")
        for piece in tqdm(dataset.pieces, desc="Processing pieces"):
            try:
                piece.extract_perform_features(DEFAULT_PERFORM_FEATURES)
                piece.extract_perform_features(DEFAULT_PURE_PERFORM_FEATURES)
                piece.extract_score_features(DEFAULT_SCORE_FEATURES)
            except Exception as e:
                print(f"\nWarning: Error extracting features for piece: {e}")
                piece.performances = []

        # Create pair dataset
        print("\nCreating pair dataset...")
        pair_data = PairDataset(dataset)
        print(f"Created {len(pair_data.data_pairs)} pairs")

        # Save features
        print(f"\nSaving features to {OUTPUT_DIR}...")
        saved_count = 0
        for pair in tqdm(pair_data.data_pairs, desc="Saving features"):
            try:
                perform_path = pair.perform_path
                feature_data = pair.features

                # Extract note_location before processing
                note_location = feature_data.pop("note_location", {})
                feature_data.pop("labels", None)
                num_notes = feature_data.pop("num_notes", 0)

                # Filter zero notes
                index_to_be_deleted = []
                for i, mmidi in enumerate(
                    zip(feature_data.get("mmidi_pitch", []),
                        feature_data.get("mmidi_velocity", []))
                ):
                    if mmidi[0] == 0 and mmidi[1] == 0:
                        index_to_be_deleted.append(i)

                # Expand global features and filter
                for key, value in feature_data.items():
                    if not isinstance(value, list) or len(value) != num_notes:
                        value = [value] * num_notes
                        feature_data[key] = value
                    filtered_value = [
                        v for i, v in enumerate(value) if i not in index_to_be_deleted
                    ]
                    feature_data[key] = filtered_value

                # Filter note_location
                if note_location:
                    note_location = {
                        'beat': [elem for i, elem in enumerate(note_location.get('beat', [])) if i not in index_to_be_deleted],
                        'voice': [elem for i, elem in enumerate(note_location.get('voice', [])) if i not in index_to_be_deleted],
                        'measure': [elem for i, elem in enumerate(note_location.get('measure', [])) if i not in index_to_be_deleted],
                        'section': [elem for i, elem in enumerate(note_location.get('section', [])) if i not in index_to_be_deleted],
                    }
                    feature_data["note_location"] = note_location

                # Save pickle file
                output_name = Path(perform_path).stem + ".pkl"
                output_file = OUTPUT_DIR / output_name

                with open(output_file, 'wb') as f:
                    pickle.dump(feature_data, f)

                saved_count += 1

            except Exception as e:
                print(f"\nWarning: Error saving {perform_path}: {e}")

        print(f"\n[SUCCESS] Saved {saved_count} feature files to {OUTPUT_DIR}")
        return True

    except Exception as e:
        import traceback
        print(f"\n[ERROR] Preprocessing failed: {e}")
        print(traceback.format_exc())
        return False


def main():
    print("=" * 60)
    print("PercePiano VirtuosoNet Preprocessing Setup")
    print("=" * 60)

    # Check dependencies
    if not check_dependencies():
        print("\n[ERROR] Missing dependencies. Please install them first.")
        return 1

    # Check data files
    if not check_data_files():
        print("\n[ERROR] Missing data files. Please ensure PercePiano data is complete.")
        return 1

    # Run preprocessing
    if not run_preprocessing():
        print("\n[ERROR] Preprocessing failed.")
        return 1

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"\nOutput: {OUTPUT_DIR}")
    print("\nNext steps:")
    print("  1. Run feature diagnostics:")
    print("     uv run python scripts/diagnostics/diagnose_features.py data/percepiano_vnet")
    print("  2. Create train/val/test splits")
    print("  3. Compute normalization statistics")

    return 0


if __name__ == "__main__":
    sys.exit(main())
