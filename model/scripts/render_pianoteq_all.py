#!/usr/bin/env python3
"""
Render all PercePiano MIDI files with Pianoteq presets.

Renders the full PercePiano dataset using 6 Pianoteq demo presets for
timbre-invariant training data augmentation.

Output structure:
    data/rendered/pianoteq/<preset_name>/<midi_stem>.wav

Usage:
    python scripts/render_pianoteq_all.py
    python scripts/render_pianoteq_all.py --preset "NY Steinway Model D"  # Single preset
    python scripts/render_pianoteq_all.py --dry-run  # Preview without rendering
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_ROOT = PROJECT_ROOT / "model"

CONFIG = {
    "pianoteq_path": "/Applications/Pianoteq 9/Pianoteq 9.app/Contents/MacOS/Pianoteq 9",
    "midi_dir": MODEL_ROOT / "data" / "raw" / "PercePiano" / "virtuoso" / "data" / "all_2rounds",
    "output_dir": MODEL_ROOT / "data" / "rendered" / "pianoteq",
    "sample_rate": 24000,
    "presets": [
        "YC5 Vintage",
        "HB Steinway Model D",
        "NY Steinway D Honky Tonk",
        "NY Steinway D Worn Out",
        "U4 Small",
        "K2 Basic",
    ],
}


def get_preset_dir_name(preset: str) -> str:
    """Convert preset name to safe directory name."""
    return preset.replace(" ", "_").replace(".", "")


def render_midi(
    midi_path: Path,
    output_path: Path,
    preset: str,
    sample_rate: int,
    pianoteq_path: str,
) -> bool:
    """Render a single MIDI file with Pianoteq."""
    if output_path.exists():
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        pianoteq_path,
        "--headless",
        "--preset", preset,
        "--midi", str(midi_path),
        "--wav", str(output_path),
        "--rate", str(sample_rate),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return output_path.exists()
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def render_preset(
    midi_files: List[Path],
    preset: str,
    output_dir: Path,
    sample_rate: int,
    pianoteq_path: str,
    dry_run: bool = False,
) -> dict:
    """Render all MIDI files for a single preset."""
    preset_dir = output_dir / get_preset_dir_name(preset)

    stats = {"total": len(midi_files), "skipped": 0, "rendered": 0, "failed": 0}

    for midi_path in tqdm(midi_files, desc=preset, leave=False):
        output_path = preset_dir / f"{midi_path.stem}.wav"

        if output_path.exists():
            stats["skipped"] += 1
            continue

        if dry_run:
            stats["rendered"] += 1
            continue

        success = render_midi(
            midi_path, output_path, preset, sample_rate, pianoteq_path
        )

        if success:
            stats["rendered"] += 1
        else:
            stats["failed"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="Render PercePiano with Pianoteq")
    parser.add_argument(
        "--preset",
        type=str,
        help="Render only this preset (default: all presets)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview rendering without actually rendering",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override output directory",
    )
    args = parser.parse_args()

    pianoteq_path = CONFIG["pianoteq_path"]
    if not Path(pianoteq_path).exists():
        print(f"Error: Pianoteq not found at {pianoteq_path}")
        sys.exit(1)

    midi_dir = CONFIG["midi_dir"]
    midi_files = sorted(midi_dir.glob("*.mid"))
    if not midi_files:
        print(f"Error: No MIDI files found in {midi_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else CONFIG["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    presets = [args.preset] if args.preset else CONFIG["presets"]

    # Validate preset names
    for preset in presets:
        if preset not in CONFIG["presets"]:
            print(f"Warning: '{preset}' not in default presets, proceeding anyway")

    total_renders = len(midi_files) * len(presets)
    print(f"Rendering {len(midi_files)} MIDIs x {len(presets)} presets = {total_renders} files")
    print(f"Output directory: {output_dir}")
    if args.dry_run:
        print("DRY RUN - no files will be rendered")
    print()

    all_stats = {}

    with tqdm(total=len(presets), desc="Presets") as pbar:
        for preset in presets:
            stats = render_preset(
                midi_files=midi_files,
                preset=preset,
                output_dir=output_dir,
                sample_rate=CONFIG["sample_rate"],
                pianoteq_path=pianoteq_path,
                dry_run=args.dry_run,
            )
            all_stats[preset] = stats
            pbar.update(1)

    print("\nRendering Complete")
    print("=" * 60)

    total_rendered = 0
    total_skipped = 0
    total_failed = 0

    for preset, stats in all_stats.items():
        print(f"{preset}:")
        print(f"  Rendered: {stats['rendered']}, Skipped: {stats['skipped']}, Failed: {stats['failed']}")
        total_rendered += stats["rendered"]
        total_skipped += stats["skipped"]
        total_failed += stats["failed"]

    print()
    print(f"Total: {total_rendered} rendered, {total_skipped} skipped, {total_failed} failed")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
