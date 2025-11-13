#!/usr/bin/env python3
"""
Pre-flight Check Script for Training

Verifies that all requirements are met before starting training:
- Annotation files exist and are valid
- Audio/MIDI files are accessible
- Config files are correct
- Model can be initialized
- Dataloaders work

Usage:
    python preflight_check.py --config configs/baseline_audioonly.yaml
"""

import argparse
import json
import yaml
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
import time

# Import model components
from src.models.lightning_module import PerformanceEvaluationModel
from src.data.dataset import create_dataloaders


def check_file_exists_with_retry(file_path: Path, max_retries: int = 3, delay: float = 0.5) -> bool:
    """
    Check if file exists with retry logic to handle Google Drive I/O issues.

    Args:
        file_path: Path to check
        max_retries: Number of retry attempts
        delay: Delay between retries in seconds

    Returns:
        True if file exists, False otherwise

    Raises:
        OSError: If I/O error persists after all retries
    """
    for attempt in range(max_retries):
        try:
            return file_path.exists()
        except OSError as e:
            if attempt < max_retries - 1:
                time.sleep(delay)
                continue
            else:
                raise OSError(f"Failed to access {file_path} after {max_retries} attempts: {e}")
    return False


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def check_annotation_file(annotation_path: Path, check_files: bool = True, sample_size: int = 100) -> dict:
    """
    Check annotation file validity.

    Args:
        annotation_path: Path to annotation JSONL file
        check_files: Whether to verify audio/MIDI files exist
        sample_size: Number of samples to check for file existence

    Returns:
        Dictionary with check results
    """
    results = {
        "exists": False,
        "valid_format": False,
        "num_samples": 0,
        "missing_audio": 0,
        "missing_midi": 0,
        "io_errors": 0,
        "dimensions": set(),
        "errors": []
    }

    # Check file exists
    if not annotation_path.exists():
        results["errors"].append(f"File not found: {annotation_path}")
        return results

    results["exists"] = True

    # Parse annotations
    try:
        annotations = []
        with open(annotation_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        ann = json.loads(line)
                        annotations.append(ann)

                        # Collect dimension names
                        if 'labels' in ann:
                            results["dimensions"].update(ann['labels'].keys())

                    except json.JSONDecodeError as e:
                        results["errors"].append(f"Line {line_num}: Invalid JSON - {e}")
                        if len(results["errors"]) > 10:
                            results["errors"].append("... (more errors truncated)")
                            break

        results["num_samples"] = len(annotations)
        results["valid_format"] = len(annotations) > 0

    except Exception as e:
        results["errors"].append(f"Error reading file: {e}")
        return results

    # Check file paths (sample only for speed)
    if check_files and annotations:
        sample_indices = np.random.choice(
            len(annotations),
            min(sample_size, len(annotations)),
            replace=False
        )

        for idx in sample_indices:
            ann = annotations[idx]

            # Check audio file
            if 'audio_path' in ann:
                audio_path = Path(ann['audio_path'])
                try:
                    if not check_file_exists_with_retry(audio_path, max_retries=3, delay=0.5):
                        results["missing_audio"] += 1
                except OSError as e:
                    results["io_errors"] += 1
                    if results["io_errors"] <= 3:
                        results["errors"].append(f"I/O error accessing {audio_path.name}: {e}")

            # Check MIDI file (optional)
            if 'midi_path' in ann and ann['midi_path']:
                midi_path = Path(ann['midi_path'])
                try:
                    if not check_file_exists_with_retry(midi_path, max_retries=3, delay=0.5):
                        results["missing_midi"] += 1
                except OSError as e:
                    results["io_errors"] += 1
                    if results["io_errors"] <= 3:
                        results["errors"].append(f"I/O error accessing {midi_path.name}: {e}")

            # Add small delay between files to avoid Google Drive throttling
            time.sleep(0.05)

        if results["io_errors"] > 0:
            results["errors"].append(
                f"Total I/O errors: {results['io_errors']} (Google Drive mount issues or sync delays)"
            )

    return results


def check_config(config: dict) -> dict:
    """Check configuration validity."""
    results = {
        "valid": True,
        "errors": [],
        "warnings": []
    }

    # Check required sections
    required_sections = ['data', 'model', 'training', 'callbacks', 'logging']
    for section in required_sections:
        if section not in config:
            results["errors"].append(f"Missing required section: {section}")
            results["valid"] = False

    # Check data paths
    if 'data' in config:
        for path_key in ['train_path', 'val_path']:
            if path_key not in config['data']:
                results["errors"].append(f"Missing {path_key} in data config")
                results["valid"] = False

    # Check model dimensions
    if 'model' in config:
        if config['model'].get('midi_dim', 0) == 0:
            results["warnings"].append("MIDI encoder disabled (audio-only mode)")

        if config['model'].get('num_dimensions', 0) == 0:
            results["errors"].append("num_dimensions must be > 0")
            results["valid"] = False

    # Check training settings
    if 'training' in config:
        if config['training'].get('max_epochs', 0) == 0:
            results["errors"].append("max_epochs must be > 0")
            results["valid"] = False

        batch_size = config['data'].get('batch_size', 0)
        grad_accum = config['training'].get('accumulate_grad_batches', 1)
        effective_batch = batch_size * grad_accum

        if effective_batch < 16:
            results["warnings"].append(
                f"Small effective batch size ({effective_batch}). Consider increasing."
            )

    return results


def test_model_initialization(config: dict) -> dict:
    """Test that model can be initialized."""
    results = {
        "success": False,
        "error": None,
        "num_parameters": 0
    }

    try:
        model = PerformanceEvaluationModel(
            audio_dim=config["model"]["audio_dim"],
            midi_dim=config["model"]["midi_dim"],
            fusion_dim=config["model"]["fusion_dim"],
            aggregator_dim=config["model"]["aggregator_dim"],
            num_dimensions=config["model"]["num_dimensions"],
            dimension_names=config["data"]["dimensions"],
            mert_model_name=config["model"]["mert_model_name"],
            freeze_audio_encoder=config["model"]["freeze_audio_encoder"],
            gradient_checkpointing=config["model"]["gradient_checkpointing"],
            learning_rate=config["training"]["learning_rate"],
            backbone_lr=config["training"]["backbone_lr"],
            heads_lr=config["training"]["heads_lr"],
            weight_decay=config["training"]["weight_decay"],
            warmup_steps=config["training"]["warmup_steps"],
            max_epochs=config["training"]["max_epochs"],
        )

        # Count parameters
        results["num_parameters"] = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        results["trainable_parameters"] = trainable

        results["success"] = True

        # Clean up
        del model
        torch.cuda.empty_cache()

    except Exception as e:
        results["error"] = str(e)

    return results


def test_dataloaders(config: dict, max_batches: int = 2) -> dict:
    """Test that dataloaders work correctly."""
    results = {
        "success": False,
        "error": None,
        "train_samples": 0,
        "val_samples": 0,
        "test_samples": 0
    }

    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            train_annotation_path=config["data"]["train_path"],
            val_annotation_path=config["data"]["val_path"],
            test_annotation_path=config["data"].get("test_path", None),
            dimension_names=config["data"]["dimensions"],
            batch_size=config["data"]["batch_size"],
            num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
            augmentation_config=None,  # Disable augmentation for testing
            audio_sample_rate=config["data"]["audio_sample_rate"],
            max_audio_length=config["data"]["max_audio_length"],
            max_midi_events=config["data"]["max_midi_events"],
        )

        results["train_samples"] = len(train_loader.dataset)
        results["val_samples"] = len(val_loader.dataset)
        if test_loader is not None:
            results["test_samples"] = len(test_loader.dataset)

        # Test loading a few batches
        print("  Testing train dataloader...")
        for i, batch in enumerate(train_loader):
            if i >= max_batches:
                break
            assert 'audio_waveform' in batch
            assert 'labels' in batch
            assert batch['audio_waveform'].shape[0] <= config["data"]["batch_size"]

        print("  Testing val dataloader...")
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            assert 'audio_waveform' in batch
            assert 'labels' in batch

        results["success"] = True

    except Exception as e:
        results["error"] = str(e)

    return results


def main():
    parser = argparse.ArgumentParser(description="Pre-flight check for training")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--skip-files", action="store_true", help="Skip checking if files exist (faster)")
    parser.add_argument("--skip-model", action="store_true", help="Skip model initialization test")
    parser.add_argument("--skip-dataloader", action="store_true", help="Skip dataloader test")
    args = parser.parse_args()

    print("="*80)
    print("PRE-FLIGHT CHECK FOR TRAINING")
    print("="*80)
    print(f"Config: {args.config}\n")

    # Load config
    try:
        config = load_config(args.config)
        print("✓ Config file loaded successfully\n")
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
        return 1

    # Check config validity
    print("Checking configuration...")
    config_results = check_config(config)
    if config_results["valid"]:
        print("✓ Configuration is valid")
    else:
        print("✗ Configuration has errors:")
        for error in config_results["errors"]:
            print(f"  - {error}")

    if config_results["warnings"]:
        print("⚠ Warnings:")
        for warning in config_results["warnings"]:
            print(f"  - {warning}")
    print()

    # Check annotation files
    print("Checking annotation files...")
    for split_name, path_key in [("Train", "train_path"), ("Val", "val_path"), ("Test", "test_path")]:
        if path_key not in config["data"]:
            continue

        ann_path = Path(config["data"][path_key])
        print(f"\n  {split_name}: {ann_path}")

        results = check_annotation_file(ann_path, check_files=not args.skip_files)

        if results["exists"]:
            print(f"    ✓ File exists")
        else:
            print(f"    ✗ File not found")
            continue

        if results["valid_format"]:
            print(f"    ✓ Valid format ({results['num_samples']} samples)")
        else:
            print(f"    ✗ Invalid format")

        if results["dimensions"]:
            print(f"    ✓ Dimensions: {', '.join(sorted(results['dimensions']))}")

        if not args.skip_files:
            if results["io_errors"] > 0:
                print(f"    ⚠ I/O errors encountered: {results['io_errors']} files (Google Drive sync issues)")
                print(f"      This is common with Google Drive mounts. Files may still be accessible.")

            if results["missing_audio"] > 0:
                print(f"    ⚠ Missing audio files: {results['missing_audio']}/100 samples checked")
            elif results["io_errors"] == 0:
                print(f"    ✓ All audio files exist (sampled)")

            if results["missing_midi"] > 0 and config["model"]["midi_dim"] > 0:
                print(f"    ⚠ Missing MIDI files: {results['missing_midi']}/100 samples checked")
            elif config["model"]["midi_dim"] > 0 and results["io_errors"] == 0:
                print(f"    ✓ All MIDI files exist (sampled)")

        if results["errors"]:
            print(f"    ✗ Errors found:")
            for error in results["errors"][:5]:
                print(f"      - {error}")

    print()

    # Test model initialization
    if not args.skip_model:
        print("Testing model initialization...")
        model_results = test_model_initialization(config)

        if model_results["success"]:
            print(f"✓ Model initialized successfully")
            print(f"  Total parameters: {model_results['num_parameters']:,}")
            print(f"  Trainable parameters: {model_results['trainable_parameters']:,}")
        else:
            print(f"✗ Model initialization failed: {model_results['error']}")
        print()

    # Test dataloaders
    if not args.skip_dataloader:
        print("Testing dataloaders...")
        dataloader_results = test_dataloaders(config)

        if dataloader_results["success"]:
            print(f"✓ Dataloaders working correctly")
            print(f"  Train samples: {dataloader_results['train_samples']}")
            print(f"  Val samples: {dataloader_results['val_samples']}")
            if dataloader_results["test_samples"] > 0:
                print(f"  Test samples: {dataloader_results['test_samples']}")
        else:
            print(f"✗ Dataloader test failed: {dataloader_results['error']}")
        print()

    # Final summary
    print("="*80)
    print("SUMMARY")
    print("="*80)

    all_good = (
        config_results["valid"] and
        (args.skip_model or model_results.get("success", False)) and
        (args.skip_dataloader or dataloader_results.get("success", False))
    )

    if all_good:
        print("✓ All checks passed! Ready for training.")
        print(f"\nTo start training, run:")
        print(f"  python train.py --config {args.config}")
        return 0
    else:
        print("✗ Some checks failed. Please fix issues before training.")
        return 1


if __name__ == "__main__":
    exit(main())
