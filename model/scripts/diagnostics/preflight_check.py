#!/usr/bin/env python3
"""
Comprehensive preflight check before training.

Verifies:
1. Data files exist and are readable
2. Audio/MIDI loading works correctly
3. Data is on local SSD (not Google Drive)
4. Loading speed is acceptable
5. Model can be instantiated
6. GPU is available with sufficient memory

Run this BEFORE starting any training to catch issues early.
"""

import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple


def check_annotation_files(
    train_path: str,
    val_path: str,
    test_path: Optional[str] = None,
) -> Tuple[bool, dict]:
    """Check that annotation files exist and are readable."""
    print("\n1. Checking annotation files...")

    files = {
        "train": train_path,
        "val": val_path,
    }
    if test_path:
        files["test"] = test_path

    results = {}
    all_ok = True

    for split, path in files.items():
        path_obj = Path(path)

        if not path_obj.exists():
            print(f"  ✗ {split}: NOT FOUND at {path}")
            all_ok = False
            results[split] = {"exists": False, "count": 0, "size_mb": 0}
            continue

        # Count samples
        with open(path_obj) as f:
            samples = [line for line in f if line.strip()]
            count = len(samples)

        size_mb = path_obj.stat().st_size / 1024 / 1024

        # Check it's on local disk (not Drive)
        is_local = (
            "/tmp/" in str(path_obj)
            or "/content/" in str(path_obj)
            and "/drive/" not in str(path_obj)
        )

        if is_local:
            location = "LOCAL SSD ✓"
        else:
            location = "GOOGLE DRIVE ⚠"
            print(f"  ⚠ {split}: On Google Drive (will be SLOW!)")

        print(f"  ✓ {split}: {count:,} samples, {size_mb:.1f} MB ({location})")

        results[split] = {
            "exists": True,
            "count": count,
            "size_mb": size_mb,
            "is_local": is_local,
            "path": str(path_obj),
        }

    return all_ok, results


def check_data_loading(
    annotation_path: str,
    num_samples: int = 5,
    timeout_per_sample: float = 10.0,
) -> Tuple[bool, dict]:
    """Test actual audio/MIDI loading from data files."""
    print(f"\n2. Testing data loading ({num_samples} samples)...")

    from src.crescendai.data.audio_processing import (
        TORCHAUDIO_AVAILABLE,
        load_audio,
        normalize_audio,
    )
    from src.crescendai.data.midi_processing import (
        align_midi_to_audio,
        encode_octuple_midi,
        load_midi,
    )

    # Report which audio backend is available
    if TORCHAUDIO_AVAILABLE:
        print("  Using: torchaudio (FAST ✓)")
    else:
        print("  Using: librosa (slower, consider installing torchaudio)")

    # Load sample annotations
    with open(annotation_path) as f:
        annotations = [json.loads(line) for line in f if line.strip()]

    if len(annotations) == 0:
        print("  ✗ No annotations found!")
        return False, {}

    # Test loading
    num_to_test = min(num_samples, len(annotations))
    results = {
        "audio_success": 0,
        "midi_success": 0,
        "failures": [],
        "load_times": [],
    }

    for i in range(num_to_test):
        ann = annotations[i]
        audio_path = ann["audio_path"]
        midi_path = ann.get("midi_path")

        try:
            # Time the loading
            start_time = time.time()

            # Load audio
            audio, sr = load_audio(audio_path, sr=24000)
            audio = normalize_audio(audio)

            # Load MIDI if available
            if midi_path:
                midi = load_midi(midi_path)
                audio_duration = len(audio) / sr
                midi = align_midi_to_audio(midi, audio_duration)
                tokens = encode_octuple_midi(midi)
                results["midi_success"] += 1

            load_time = time.time() - start_time
            results["load_times"].append(load_time)
            results["audio_success"] += 1

            # Check for slow loading (probable Google Drive)
            if load_time > 2.0:
                print(
                    f"  ⚠ Sample {i + 1}: SLOW ({load_time:.2f}s) - check data location!"
                )
            else:
                print(
                    f"  ✓ Sample {i + 1}: {load_time:.3f}s ({len(audio):,} samples, {len(tokens) if midi_path else 0} MIDI tokens)"
                )

        except Exception as e:
            results["failures"].append({"index": i, "error": str(e)})
            print(f"  ✗ Sample {i + 1}: {str(e)[:80]}...")

            # Don't fail immediately, try more samples
            if len(results["failures"]) >= 3:
                print(f"  ✗ Too many failures ({len(results['failures'])}), stopping")
                break

    # Summary
    success_rate = results["audio_success"] / num_to_test
    if results["load_times"]:
        avg_time = sum(results["load_times"]) / len(results["load_times"])
        max_time = max(results["load_times"])
    else:
        avg_time = 0
        max_time = 0

    print(f"\n  Summary:")
    print(
        f"    Audio loaded: {results['audio_success']}/{num_to_test} ({success_rate * 100:.0f}%)"
    )
    print(f"    MIDI loaded:  {results['midi_success']}/{num_to_test}")
    print(f"    Avg time:     {avg_time:.3f}s")
    print(f"    Max time:     {max_time:.3f}s")

    if avg_time > 1.0:
        print(f"    ⚠ Average load time > 1s suggests data is on Google Drive!")
        print(f"      Expected: <0.1s on local SSD, <0.5s with torchaudio")

    all_ok = success_rate >= 0.8 and len(results["failures"]) == 0

    return all_ok, results


def check_dataloader_speed(
    train_path: str,
    val_path: str,
    batch_size: int = 8,
    num_workers: int = 0,
    num_batches: int = 3,
) -> Tuple[bool, dict]:
    """Test DataLoader speed with actual configuration."""
    print(f"\n3. Testing DataLoader speed...")
    print(f"  Config: batch_size={batch_size}, num_workers={num_workers}")

    from src.crescendai.data.dataset import create_dataloaders

    # Create dataloaders
    train_loader, val_loader, _ = create_dataloaders(
        train_annotation_path=train_path,
        val_annotation_path=val_path,
        test_annotation_path=None,
        dimension_names=["note_accuracy", "rhythmic_precision", "tone_quality"],
        batch_size=batch_size,
        num_workers=num_workers,
        augmentation_config=None,
        audio_sample_rate=24000,
        max_audio_length=240000,
        max_midi_events=512,
    )

    print(
        f"  Dataset: {len(train_loader.dataset):,} train, {len(val_loader.dataset):,} val"
    )

    # Time a few batches
    times = []
    num_to_test = min(num_batches, len(train_loader))

    print(f"  Loading {num_to_test} batches...")
    for i, batch in enumerate(train_loader):
        if i >= num_to_test:
            break

        start = time.time()
        # Access batch data to force loading
        _ = batch["audio_waveform"].shape
        if "midi_tokens" in batch:
            _ = batch["midi_tokens"].shape
        elapsed = time.time() - start

        times.append(elapsed)
        print(f"    Batch {i + 1}: {elapsed:.3f}s")

    avg_time = sum(times) / len(times) if times else 0
    print(f"\n  Average batch time: {avg_time:.3f}s")

    # Estimate epoch time
    batches_per_epoch = len(train_loader)
    estimated_epoch_time = avg_time * batches_per_epoch
    print(
        f"  Estimated epoch time: {estimated_epoch_time / 60:.1f} min ({batches_per_epoch} batches)"
    )

    # Check if speed is acceptable
    results = {
        "avg_batch_time": avg_time,
        "estimated_epoch_min": estimated_epoch_time / 60,
        "batches_per_epoch": batches_per_epoch,
    }

    # Flag if too slow
    if avg_time > 2.0:
        print(f"  ⚠ VERY SLOW! (>2s/batch) - likely reading from Google Drive")
        print(f"     Copy data to local SSD for 10-30x speedup")
        all_ok = False
    elif avg_time > 0.5:
        print(f"  ⚠ Slow (>0.5s/batch) - consider optimizations")
        all_ok = True
    else:
        print(f"  ✓ Good speed (<0.5s/batch)")
        all_ok = True

    return all_ok, results


def check_model_instantiation(config_path: str) -> Tuple[bool, dict]:
    """Test that model can be instantiated with config."""
    print(f"\n4. Testing model instantiation...")

    try:
        import torch
        import yaml

        from src.crescendai.models.lightning_module import PerformanceEvaluationModel

        # Load config
        with open(config_path) as f:
            config = yaml.safe_load(f)

        results = {}

        # Test each mode
        for mode in ["audio", "midi", "fusion"]:
            print(f"  Testing {mode} mode...", end=" ")

            # Apply mode overrides
            model_config = config["model"].copy()
            mode_overrides = config.get("modes", {}).get(mode, {})
            model_config.update(mode_overrides)

            # Instantiate model
            model = PerformanceEvaluationModel(
                dimension_names=config["data"]["dimensions"],
                audio_dim=model_config["audio_dim"],
                midi_dim=model_config["midi_dim"],
                fusion_dim=model_config["fusion_dim"],
                aggregator_dim=model_config["aggregator_dim"],
                num_dimensions=len(config["data"]["dimensions"]),
                mert_model_name=model_config["mert_model_name"],
                freeze_audio_encoder=model_config["freeze_audio_encoder"],
                gradient_checkpointing=model_config["gradient_checkpointing"],
                learning_rate=config["training"]["learning_rate"],
                backbone_lr=config["training"]["backbone_lr"],
                heads_lr=config["training"]["heads_lr"],
                weight_decay=config["training"]["weight_decay"],
                warmup_steps=config["training"]["warmup_steps"],
                max_epochs=config["training"]["max_epochs"],
            )

            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )

            results[mode] = {
                "total_params_m": total_params / 1e6,
                "trainable_params_m": trainable_params / 1e6,
            }

            print(f"✓ ({trainable_params / 1e6:.1f}M params)")

            # Clean up
            del model
            torch.cuda.empty_cache()

        return True, results

    except Exception as e:
        print(f"✗ Failed: {e}")
        return False, {"error": str(e)}


def check_gpu() -> Tuple[bool, dict]:
    """Check GPU availability and memory."""
    print(f"\n5. Checking GPU...")

    try:
        import torch

        if not torch.cuda.is_available():
            print("  ✗ NO GPU DETECTED!")
            print("    Enable GPU: Runtime → Change runtime type → T4 GPU")
            return False, {}

        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

        print(f"  ✓ GPU: {gpu_name}")
        print(f"  ✓ Memory: {gpu_memory_gb:.1f} GB")

        # Check if memory is sufficient
        required_memory = 12.0  # GB for MERT-95M + batch_size=16
        if gpu_memory_gb < required_memory:
            print(
                f"  ⚠ Warning: {gpu_memory_gb:.1f} GB < {required_memory} GB recommended"
            )
            print(f"    May need to reduce batch_size or use gradient checkpointing")

        results = {
            "gpu_name": gpu_name,
            "memory_gb": gpu_memory_gb,
            "sufficient": gpu_memory_gb >= required_memory,
        }

        return True, results

    except Exception as e:
        print(f"  ✗ GPU check failed: {e}")
        return False, {}


def main():
    """Run all preflight checks."""
    import argparse

    parser = argparse.ArgumentParser(description="Preflight check before training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment_10k.yaml",
        help="Path to config file",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("PREFLIGHT CHECK")
    print("=" * 70)
    print("\nVerifying training environment and data...")

    # Load config
    try:
        import yaml

        with open(args.config) as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"\n✗ Failed to load config: {e}")
        return 1

    # Run checks
    checks = {}

    checks["annotations"] = check_annotation_files(
        train_path=config["data"]["train_path"],
        val_path=config["data"]["val_path"],
        test_path=config["data"].get("test_path"),
    )

    checks["data_loading"] = check_data_loading(
        annotation_path=config["data"]["train_path"],
        num_samples=5,
    )

    checks["dataloader"] = check_dataloader_speed(
        train_path=config["data"]["train_path"],
        val_path=config["data"]["val_path"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        num_batches=3,
    )

    checks["model"] = check_model_instantiation(args.config)

    checks["gpu"] = check_gpu()

    # Summary
    print("\n" + "=" * 70)
    print("PREFLIGHT SUMMARY")
    print("=" * 70)

    all_passed = True
    for check_name, (passed, _) in checks.items():
        symbol = "✓" if passed else "✗"
        print(f"  {symbol} {check_name}")
        if not passed:
            all_passed = False

    print("=" * 70)

    if all_passed:
        print("\n✓ ALL CHECKS PASSED - Ready for training!")
        return 0
    else:
        print("\n⚠ SOME CHECKS FAILED - Review issues above")
        print("\nCommon fixes:")
        print("  1. Data on Drive → Run: python scripts/copy_data_to_local.py")
        print("  2. Slow loading → Install torchaudio, use local SSD")
        print("  3. No GPU → Runtime → Change runtime type → T4 GPU")
        return 1


if __name__ == "__main__":
    sys.exit(main())
