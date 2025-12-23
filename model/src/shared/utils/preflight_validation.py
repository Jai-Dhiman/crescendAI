#!/usr/bin/env python3
"""
Pre-flight validation for score-aligned training.

Validates all requirements before training starts:
1. Score files are accessible
2. Pre-trained encoder weights exist
3. MIDI files exist for all samples
4. Data files have required fields

This module implements FAIL-FAST validation - if any critical requirement
is not met, training will not start. This prevents wasting compute time
on runs that will produce R^2 = 0.

Usage:
    # As a module:
    from src.utils.preflight_validation import run_preflight_validation
    run_preflight_validation(data_dir, score_dir, pretrained_checkpoint)

    # As a script:
    python -m src.utils.preflight_validation --data-dir data/processed --score-dir data/scores
"""

import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch


class PreflightValidationError(Exception):
    """
    Raised when pre-flight validation fails.

    This exception indicates that training should NOT proceed because
    critical requirements are not met.
    """

    pass


class ScoreValidationError(PreflightValidationError):
    """Raised when score file validation fails."""

    pass


class MIDIValidationError(PreflightValidationError):
    """Raised when MIDI file validation fails."""

    pass


class EncoderValidationError(PreflightValidationError):
    """Raised when pre-trained encoder validation fails."""

    pass


class DataValidationError(PreflightValidationError):
    """Raised when data file validation fails."""

    pass


def validate_data_files(data_dir: Path) -> Tuple[int, int, int]:
    """
    Validate that data files exist and are readable.

    Args:
        data_dir: Directory containing percepiano_*.json files

    Returns:
        Tuple of (train_count, val_count, test_count)

    Raises:
        DataValidationError: If data files are missing or invalid
    """
    counts = {}

    for split in ["train", "val", "test"]:
        data_file = data_dir / f"percepiano_{split}.json"

        if not data_file.exists():
            raise DataValidationError(
                f"Data file not found: {data_file}\n"
                "\n"
                "ACTION REQUIRED:\n"
                "1. Run data preparation: python scripts/prepare_percepiano.py\n"
                "2. Or download from GDrive: rclone copy gdrive:percepiano_data/percepiano_{split}.json {data_dir}/"
            )

        try:
            with open(data_file) as f:
                samples = json.load(f)
        except json.JSONDecodeError as e:
            raise DataValidationError(
                f"Invalid JSON in {data_file}: {e}\n"
                "The data file is corrupted. Re-download or re-generate it."
            )

        if not isinstance(samples, list):
            raise DataValidationError(
                f"Expected list in {data_file}, got {type(samples).__name__}"
            )

        if len(samples) == 0:
            raise DataValidationError(f"Empty data file: {data_file}")

        # Validate first sample has required fields
        required_fields = ["name", "midi_path", "percepiano_scores"]
        sample = samples[0]
        missing_fields = [f for f in required_fields if f not in sample]
        if missing_fields:
            raise DataValidationError(
                f"Data file {data_file} is missing required fields: {missing_fields}\n"
                "Expected fields: {required_fields}"
            )

        counts[split] = len(samples)

    print(
        f"[OK] Data files: train={counts['train']}, val={counts['val']}, test={counts['test']}"
    )
    return counts["train"], counts["val"], counts["test"]


def validate_midi_files(data_dir: Path) -> int:
    """
    Validate MIDI files exist for all samples.

    Args:
        data_dir: Directory containing percepiano_*.json files

    Returns:
        Total number of MIDI files validated

    Raises:
        MIDIValidationError: If MIDI files are missing
    """
    missing_midi = []
    total = 0

    for split in ["train", "val", "test"]:
        data_file = data_dir / f"percepiano_{split}.json"

        with open(data_file) as f:
            samples = json.load(f)

        for sample in samples:
            total += 1
            midi_path = Path(sample["midi_path"])

            if not midi_path.exists():
                missing_midi.append((split, sample["name"], str(midi_path)))

    if missing_midi:
        error_lines = [
            f"Missing {len(missing_midi)} MIDI files out of {total}",
            "",
            "First 10 missing:",
        ]
        for split, name, path in missing_midi[:10]:
            error_lines.append(f"  [{split}] {name}: {path}")

        if len(missing_midi) > 10:
            error_lines.append(f"  ... and {len(missing_midi) - 10} more")

        error_lines.extend(
            [
                "",
                "ACTION REQUIRED:",
                "1. Download MIDI files from GDrive:",
                "   rclone copy gdrive:percepiano_data/PercePiano/virtuoso/data/all_2rounds/ /tmp/midi_files/",
                "2. Or update paths in data files to match actual file locations",
            ]
        )

        raise MIDIValidationError("\n".join(error_lines))

    print(f"[OK] MIDI files: {total}/{total} exist")
    return total


def validate_score_files(
    data_dir: Path,
    score_dir: Path,
    min_coverage: float = 0.95,
) -> Tuple[int, int]:
    """
    Validate that score files exist for training samples.

    Args:
        data_dir: Directory containing percepiano_*.json files
        score_dir: Directory containing MusicXML score files
        min_coverage: Minimum fraction of samples that must have scores (default: 95%)

    Returns:
        Tuple of (found_count, total_count)

    Raises:
        ScoreValidationError: If score coverage is below threshold
    """
    if not score_dir.exists():
        raise ScoreValidationError(
            f"Score directory not found: {score_dir}\n"
            "\n"
            "ACTION REQUIRED:\n"
            "1. Upload scores to GDrive: python scripts/upload_score_files.py\n"
            "2. Download scores: rclone copy gdrive:percepiano_data/PercePiano/virtuoso/data/score_xml/ {score_dir}/\n"
            "3. Or copy local scores: cp -r data/raw/PercePiano/virtuoso/data/score_xml/ {score_dir}/"
        )

    missing_scores = []
    no_score_path = []
    total = 0
    found = 0

    for split in ["train", "val", "test"]:
        data_file = data_dir / f"percepiano_{split}.json"

        with open(data_file) as f:
            samples = json.load(f)

        for sample in samples:
            total += 1
            score_path = sample.get("score_path")

            if not score_path:
                no_score_path.append((split, sample["name"]))
                continue

            full_path = score_dir / score_path

            if full_path.exists():
                found += 1
            else:
                missing_scores.append((split, sample["name"], str(full_path)))

    coverage = found / total if total > 0 else 0

    if coverage < min_coverage:
        error_lines = [
            f"Score file coverage ({coverage:.1%}) below threshold ({min_coverage:.1%})",
            f"Found {found} of {total} score files",
            "",
        ]

        if no_score_path:
            error_lines.append(
                f"Samples without score_path field: {len(no_score_path)}"
            )

        if missing_scores:
            error_lines.append(f"Missing score files: {len(missing_scores)}")
            error_lines.append("")
            error_lines.append("First 10 missing:")
            for split, name, path in missing_scores[:10]:
                error_lines.append(f"  [{split}] {name}: {path}")

        error_lines.extend(
            [
                "",
                "ACTION REQUIRED:",
                "1. Upload scores: python scripts/upload_score_files.py",
                f"2. Download scores: rclone copy gdrive:percepiano_data/PercePiano/virtuoso/data/score_xml/ {score_dir}/",
                f"3. Or copy local: cp -r data/raw/PercePiano/virtuoso/data/score_xml/* {score_dir}/",
            ]
        )

        raise ScoreValidationError("\n".join(error_lines))

    print(f"[OK] Score files: {found}/{total} ({coverage:.1%})")
    return found, total


def validate_pretrained_encoder(
    checkpoint_path: Optional[Path],
    require_pretrained: bool = False,
) -> bool:
    """
    Validate that pre-trained encoder weights exist and are valid.

    Args:
        checkpoint_path: Path to pre-trained encoder checkpoint
        require_pretrained: If True, raise error when checkpoint is None

    Returns:
        True if checkpoint was loaded, False if skipped

    Raises:
        EncoderValidationError: If checkpoint is required but missing/invalid
    """
    if checkpoint_path is None:
        if require_pretrained:
            raise EncoderValidationError(
                "Pre-trained encoder checkpoint is required but not provided.\n"
                "\n"
                "ACTION REQUIRED:\n"
                "1. Download: rclone copy gdrive:crescendai_checkpoints/midi_pretrain/encoder_pretrained.pt /tmp/checkpoints/\n"
                "2. Pass to training: --midi-pretrained-checkpoint /tmp/checkpoints/encoder_pretrained.pt\n"
                "\n"
                "Or run pre-training first:\n"
                "  python scripts/pretrain_midi_encoder.py --midi_dir <path>"
            )
        else:
            print("[WARN] No pre-trained encoder specified - training from scratch")
            print(
                "       For better results, download encoder_pretrained.pt from GDrive"
            )
            return False

    if not checkpoint_path.exists():
        raise EncoderValidationError(
            f"Pre-trained encoder not found: {checkpoint_path}\n"
            "\n"
            "ACTION REQUIRED:\n"
            "1. Download: rclone copy gdrive:crescendai_checkpoints/midi_pretrain/encoder_pretrained.pt {checkpoint_path.parent}/\n"
            "2. Or check the path is correct"
        )

    # Verify it's a valid checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as e:
        raise EncoderValidationError(
            f"Failed to load pre-trained encoder: {e}\n"
            f"File: {checkpoint_path}\n"
            "The checkpoint file may be corrupted. Re-download it."
        )

    # Check for expected structure
    if isinstance(checkpoint, dict):
        if "encoder_state_dict" in checkpoint:
            print(
                f"[OK] Pre-trained encoder: {checkpoint_path} (full checkpoint format)"
            )
        else:
            print(f"[OK] Pre-trained encoder: {checkpoint_path} (state_dict format)")
    else:
        print(f"[WARN] Unexpected checkpoint format: {type(checkpoint)}")

    return True


def run_preflight_validation(
    data_dir: Path,
    score_dir: Path,
    pretrained_checkpoint: Optional[Path] = None,
    require_pretrained: bool = False,
    min_score_coverage: float = 0.95,
) -> None:
    """
    Run all pre-flight validation checks.

    This function should be called at the start of training to ensure
    all requirements are met. If any validation fails, training should
    NOT proceed.

    Args:
        data_dir: Directory containing percepiano_*.json files
        score_dir: Directory containing MusicXML score files
        pretrained_checkpoint: Path to pre-trained encoder (optional)
        require_pretrained: If True, require pre-trained encoder
        min_score_coverage: Minimum score file coverage (default: 95%)

    Raises:
        PreflightValidationError: If any validation fails
    """
    print("=" * 60)
    print("PRE-FLIGHT VALIDATION")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Score directory: {score_dir}")
    print(f"Pre-trained encoder: {pretrained_checkpoint or 'None'}")
    print("=" * 60)

    # Run all validations
    validate_data_files(data_dir)
    validate_midi_files(data_dir)
    validate_score_files(data_dir, score_dir, min_score_coverage)
    validate_pretrained_encoder(pretrained_checkpoint, require_pretrained)

    print("=" * 60)
    print("ALL VALIDATIONS PASSED - Ready to train")
    print("=" * 60)


def main():
    """CLI entry point for running validation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Pre-flight validation for score-aligned training"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing percepiano_*.json files",
    )
    parser.add_argument(
        "--score-dir",
        type=Path,
        default=Path("data/scores"),
        help="Directory containing MusicXML score files",
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        type=Path,
        default=None,
        help="Path to pre-trained encoder checkpoint",
    )
    parser.add_argument(
        "--require-pretrained",
        action="store_true",
        help="Require pre-trained encoder (fail if not provided)",
    )
    parser.add_argument(
        "--min-score-coverage",
        type=float,
        default=0.95,
        help="Minimum score file coverage (default: 0.95)",
    )

    args = parser.parse_args()

    try:
        run_preflight_validation(
            data_dir=args.data_dir,
            score_dir=args.score_dir,
            pretrained_checkpoint=args.pretrained_checkpoint,
            require_pretrained=args.require_pretrained,
            min_score_coverage=args.min_score_coverage,
        )
    except PreflightValidationError as e:
        print(f"\n[VALIDATION FAILED]\n{e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
