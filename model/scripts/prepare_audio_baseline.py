#!/usr/bin/env python3
"""
Local preprocessing script for Audio Baseline.

Run this locally (M4 Mac) to prepare data before uploading to Thunder Compute.
This handles CPU-intensive tasks that don't need GPU:
  1. Download Salamander Grand Piano soundfont (~400MB)
  2. Render all MIDI files to WAV using FluidSynth
  3. Create piece-based fold assignments

Usage:
    cd model
    uv run python scripts/prepare_audio_baseline.py

After running, sync to Google Drive:
    rclone copy data/audio gdrive:crescendai_data/audio_baseline/audio --progress
    rclone copy data/soundfonts gdrive:crescendai_data/audio_baseline/soundfonts --progress
    rclone copy data/cache/audio_fold_assignments.json gdrive:crescendai_data/audio_baseline/ --progress
"""

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path


# Direct file imports to bypass package __init__.py chain
def import_from_file(module_name: str, file_path: Path):
    """Import a module directly from file path, bypassing package __init__.py."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


PROJECT_ROOT = Path(__file__).parent.parent
SRC_ROOT = PROJECT_ROOT / "src"

# Import render_midi module directly
render_midi = import_from_file(
    "render_midi", SRC_ROOT / "percepiano" / "audio" / "render_midi.py"
)
batch_render_midi = render_midi.batch_render_midi
check_fluidsynth_installed = render_midi.check_fluidsynth_installed
download_salamander_soundfont = render_midi.download_salamander_soundfont


def get_name_wo_performer(filename: str) -> str:
    """
    Extract composition group name (without performer ID).

    Matches PercePiano's m2pf_dataset_compositionfold.py exactly.

    Example: Beethoven_WoO80_thema_8bars_1_1 -> Beethoven_WoO80_thema_8bars_1
    (removes the performer ID which is second-to-last)
    """
    parts = filename.split("_")
    prefix = "_".join(parts[:-2])  # Everything except last 2 parts
    suffix = "_".join(parts[-1:])  # Last part (segment ID)
    return prefix + "_" + suffix


def create_audio_fold_assignments(
    label_file: Path,
    output_file: Path,
    n_folds: int = 4,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict:
    """
    Create composition-based fold assignments matching PercePiano exactly.

    Replicates m2pf_dataset_compositionfold.py:
    1. Group samples by composition (piece+segment, removing performer ID)
    2. Randomly select compositions for test set until reaching 15% of samples
       (only compositions with >1 sample are selected)
    3. Assign remaining compositions to folds via round-robin

    Args:
        label_file: Path to PercePiano label JSON
        output_file: Path to save fold assignments
        n_folds: Number of CV folds
        test_ratio: Fraction of samples for test set
        seed: Random seed

    Returns:
        Fold assignments dictionary
    """
    import random
    from collections import defaultdict

    random.seed(seed)

    with open(label_file) as f:
        labels = json.load(f)

    all_keys = list(labels.keys())
    print(f"Total samples: {len(all_keys)}")

    # Shuffle all keys first (matching PercePiano)
    random.shuffle(all_keys)

    # Group by composition (piece+segment without performer ID)
    composition_groups: dict[str, list[str]] = defaultdict(list)
    for key in all_keys:
        comp_name = get_name_wo_performer(key)
        composition_groups[comp_name].append(key)

    print(f"Found {len(composition_groups)} unique compositions")

    # Select test set: randomly pick compositions until reaching test_ratio
    # Only select compositions with more than 1 sample (matching PercePiano)
    test_keys: list[str] = []
    test_compositions: set[str] = set()
    available_compositions = list(composition_groups.keys())

    while len(test_keys) < test_ratio * len(all_keys) and available_compositions:
        # Randomly select a composition
        comp = random.choice(available_compositions)
        available_compositions.remove(comp)

        # Only add if it has more than 1 sample (matching PercePiano line 133)
        if len(composition_groups[comp]) > 1:
            test_keys.extend(composition_groups[comp])
            test_compositions.add(comp)

    print(
        f"Test set: {len(test_keys)} samples from {len(test_compositions)} compositions"
    )

    # Get remaining compositions (not in test)
    cv_compositions = [
        c for c in composition_groups.keys() if c not in test_compositions
    ]

    # Assign to folds via round-robin (matching PercePiano line 149)
    fold_map: dict[str, int] = {}
    for i, comp in enumerate(cv_compositions):
        fold_idx = i % n_folds
        for key in composition_groups[comp]:
            fold_map[key] = fold_idx

    # Build output structure
    fold_assignments: dict[str, list[str]] = {
        "test": test_keys,
        "fold_0": [],
        "fold_1": [],
        "fold_2": [],
        "fold_3": [],
    }

    for key, fold_idx in fold_map.items():
        fold_assignments[f"fold_{fold_idx}"].append(key)

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(fold_assignments, f, indent=2)

    print(f"Saved fold assignments to {output_file}")

    # Print statistics
    print("\nFold statistics:")
    for fold_name, keys in fold_assignments.items():
        print(f"  {fold_name}: {len(keys)} samples")

    return fold_assignments


def main():
    print("=" * 70)
    print("AUDIO BASELINE - LOCAL PREPROCESSING")
    print("=" * 70)
    print("This script prepares data locally before uploading to Thunder Compute.")
    print("Tasks: soundfont download, MIDI rendering, fold assignments")
    print("=" * 70 + "\n")

    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    PERCEPIANO_ROOT = PROJECT_ROOT / "data" / "raw" / "PercePiano"
    MIDI_DIR = PERCEPIANO_ROOT / "virtuoso" / "data" / "all_2rounds"
    LABEL_FILE = PERCEPIANO_ROOT / "label_2round_mean_reg_19_with0_rm_highstd0.json"

    AUDIO_DIR = PROJECT_ROOT / "data" / "audio" / "percepiano_rendered"
    SOUNDFONT_PATH = PROJECT_ROOT / "data" / "soundfonts" / "SalamanderGrandPiano.sf2"
    FOLD_FILE = PROJECT_ROOT / "data" / "cache" / "audio_fold_assignments.json"

    # Create directories
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    SOUNDFONT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FOLD_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Verify paths
    print("Checking paths...")
    if not MIDI_DIR.exists():
        print(f"[ERROR] MIDI directory not found: {MIDI_DIR}")
        print("Make sure PercePiano data is downloaded.")
        sys.exit(1)

    if not LABEL_FILE.exists():
        print(f"[ERROR] Label file not found: {LABEL_FILE}")
        sys.exit(1)

    midi_count = len(list(MIDI_DIR.glob("*.mid")))
    print(f"  MIDI files: {midi_count}")

    with open(LABEL_FILE) as f:
        labels = json.load(f)
    print(f"  Labels: {len(labels)} segments")
    print()

    # Step 1: Check FluidSynth
    print("-" * 70)
    print("Step 1: Check FluidSynth")
    print("-" * 70)

    if not check_fluidsynth_installed():
        print("[ERROR] FluidSynth not installed!")
        print("Install with:")
        print("  macOS: brew install fluidsynth")
        print("  Linux: apt-get install fluidsynth")
        sys.exit(1)

    # Get FluidSynth version
    result = subprocess.run(["fluidsynth", "--version"], capture_output=True, text=True)
    version_line = result.stdout.strip().split("\n")[0] if result.stdout else "unknown"
    print(f"  FluidSynth: {version_line}")
    print()

    # Step 2: Download Soundfont
    print("-" * 70)
    print("Step 2: Download Salamander Grand Piano Soundfont")
    print("-" * 70)

    if SOUNDFONT_PATH.exists():
        size_mb = SOUNDFONT_PATH.stat().st_size / 1e6
        print(f"  Already exists: {SOUNDFONT_PATH}")
        print(f"  Size: {size_mb:.1f} MB")
    else:
        print("  Downloading (~400MB, may take a few minutes)...")
        start = time.time()
        download_salamander_soundfont(SOUNDFONT_PATH)
        elapsed = time.time() - start
        print(f"  Downloaded in {elapsed:.1f}s")
    print()

    # Step 3: Render MIDI to WAV
    print("-" * 70)
    print("Step 3: Render MIDI to WAV")
    print("-" * 70)

    label_keys = list(labels.keys())
    existing_wavs = len(list(AUDIO_DIR.glob("*.wav")))

    if existing_wavs == len(label_keys):
        print(f"  All {existing_wavs} WAV files already rendered!")
        print("  Skipping rendering step.")
    else:
        print(f"  Target: {len(label_keys)} files")
        print(f"  Existing: {existing_wavs} files")
        print(f"  To render: {len(label_keys) - existing_wavs} files")
        print()
        print("  Starting rendering (this may take 30-60 minutes)...")
        print()

        start = time.time()
        successful, failed = batch_render_midi(
            midi_dir=MIDI_DIR,
            output_dir=AUDIO_DIR,
            soundfont_path=SOUNDFONT_PATH,
            label_keys=label_keys,
            max_workers=4,
            skip_existing=True,
            sample_rate=44100,
            gain=0.8,
        )
        elapsed = time.time() - start

        print()
        print(f"  Rendering complete in {elapsed / 60:.1f} minutes")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")

        if failed > 0:
            print(f"  [WARNING] {failed} files failed to render")
    print()

    # Step 4: Create Fold Assignments
    print("-" * 70)
    print("Step 4: Create Fold Assignments")
    print("-" * 70)

    if FOLD_FILE.exists():
        print(f"  Already exists: {FOLD_FILE}")
        with open(FOLD_FILE) as f:
            fold_assignments = json.load(f)
        print("  Fold statistics:")
        for fold_name, keys in fold_assignments.items():
            print(f"    {fold_name}: {len(keys)} samples")
    else:
        print("  Creating piece-based 4-fold assignments...")
        fold_assignments = create_audio_fold_assignments(
            label_file=LABEL_FILE,
            output_file=FOLD_FILE,
            n_folds=4,
            test_ratio=0.15,
            seed=42,
        )
    print()

    # Summary
    print("=" * 70)
    print("LOCAL PREPROCESSING COMPLETE")
    print("=" * 70)

    wav_count = len(list(AUDIO_DIR.glob("*.wav")))
    soundfont_size = (
        SOUNDFONT_PATH.stat().st_size / 1e6 if SOUNDFONT_PATH.exists() else 0
    )

    print(f"  Soundfont: {soundfont_size:.1f} MB")
    print(f"  WAV files: {wav_count}")
    print(f"  Fold assignments: {FOLD_FILE.name}")

    # Calculate total size
    wav_size = sum(f.stat().st_size for f in AUDIO_DIR.glob("*.wav")) / 1e9
    print(f"\n  Total audio size: {wav_size:.2f} GB")

    print("\n" + "-" * 70)
    print("NEXT STEPS: Upload to Google Drive")
    print("-" * 70)
    print("Run these commands to sync data for Thunder Compute:\n")
    print("  # Sync rendered audio (~5-10 GB)")
    print("  rclone copy data/audio/percepiano_rendered \\")
    print("    gdrive:crescendai_data/audio_baseline/percepiano_rendered --progress")
    print()
    print("  # Sync soundfont (~400 MB)")
    print("  rclone copy data/soundfonts \\")
    print("    gdrive:crescendai_data/audio_baseline/soundfonts --progress")
    print()
    print("  # Sync fold assignments")
    print("  rclone copy data/cache/audio_fold_assignments.json \\")
    print("    gdrive:crescendai_data/audio_baseline/ --progress")
    print()
    print("Then run the notebook on Thunder Compute for MERT extraction + training.")
    print("=" * 70)


if __name__ == "__main__":
    main()
