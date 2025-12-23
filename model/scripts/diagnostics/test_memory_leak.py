#!/usr/bin/env python3
"""
Memory leak detection script for training pipeline.

Run this before full training to verify memory doesn't accumulate.
Simulates 100 training iterations and monitors CPU/GPU memory.

Usage:
    python scripts/test_memory_leak.py --annotation-path /path/to/train.jsonl
"""

import argparse
import gc
import os
import sys
import time

import psutil
import torch

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.crescendai.data.dataset import create_dataloaders
from src.crescendai.models.audio_encoder import MERTEncoder


def get_memory_mb():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    cpu_mb = process.memory_info().rss / 1024 / 1024

    gpu_mb = 0
    if torch.cuda.is_available():
        gpu_mb = torch.cuda.memory_allocated() / 1024 / 1024

    return cpu_mb, gpu_mb


def test_dataloader_memory(annotation_path: str, num_batches: int = 50):
    """Test DataLoader for memory leaks."""
    print("=" * 70)
    print("TEST 1: DataLoader Memory Test")
    print("=" * 70)

    dimensions = [
        "note_accuracy",
        "rhythmic_stability",
        "articulation_clarity",
        "pedal_technique",
        "tone_quality",
        "dynamic_range",
        "musical_expression",
        "overall_interpretation",
    ]

    train_loader, _, _ = create_dataloaders(
        train_annotation_path=annotation_path,
        val_annotation_path=annotation_path,  # Use same for test
        test_annotation_path=None,
        dimension_names=dimensions,
        batch_size=8,
        num_workers=1,
        augmentation_config=None,
        audio_sample_rate=24000,
        max_audio_length=240000,
        max_midi_events=512,
    )

    initial_cpu, initial_gpu = get_memory_mb()
    print(f"Initial memory - CPU: {initial_cpu:.1f} MB, GPU: {initial_gpu:.1f} MB")

    memory_samples = []

    for i, batch in enumerate(train_loader):
        if i >= num_batches:
            break

        # Just access the data (simulates what would happen in training)
        _ = batch["audio_waveform"]
        _ = batch["labels"]

        if i % 10 == 0:
            cpu_mb, gpu_mb = get_memory_mb()
            memory_samples.append((i, cpu_mb, gpu_mb))
            print(
                f"  Batch {i:3d}: CPU {cpu_mb:.1f} MB (+{cpu_mb - initial_cpu:.1f}), GPU {gpu_mb:.1f} MB"
            )

    final_cpu, final_gpu = get_memory_mb()
    cpu_growth = final_cpu - initial_cpu

    print(f"\nFinal memory - CPU: {final_cpu:.1f} MB, GPU: {final_gpu:.1f} MB")
    print(f"CPU memory growth: {cpu_growth:.1f} MB over {num_batches} batches")

    if cpu_growth > 500:  # More than 500MB growth is suspicious
        print("WARNING: Significant memory growth detected in DataLoader!")
        return False
    else:
        print("PASSED: DataLoader memory is stable")
        return True


def test_encoder_memory(annotation_path: str, num_batches: int = 50):
    """Test MERT encoder for memory leaks."""
    print("\n" + "=" * 70)
    print("TEST 2: MERT Encoder Memory Test")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load encoder
    print("Loading MERT encoder...")
    encoder = MERTEncoder(
        model_name="m-a-p/MERT-v1-95M",
        freeze_backbone=True,
        gradient_checkpointing=False,
    ).to(device)
    encoder.eval()

    dimensions = [
        "note_accuracy",
        "rhythmic_stability",
        "articulation_clarity",
        "pedal_technique",
        "tone_quality",
        "dynamic_range",
        "musical_expression",
        "overall_interpretation",
    ]

    train_loader, _, _ = create_dataloaders(
        train_annotation_path=annotation_path,
        val_annotation_path=annotation_path,
        test_annotation_path=None,
        dimension_names=dimensions,
        batch_size=8,
        num_workers=1,
        augmentation_config=None,
        audio_sample_rate=24000,
        max_audio_length=240000,
        max_midi_events=512,
    )

    # Warm up
    print("Warming up...")
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i >= 3:
                break
            audio = batch["audio_waveform"].to(device)
            _ = encoder(audio)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    initial_cpu, initial_gpu = get_memory_mb()
    print(
        f"Initial memory (after warmup) - CPU: {initial_cpu:.1f} MB, GPU: {initial_gpu:.1f} MB"
    )

    memory_samples = []

    print("\nRunning encoder forward passes...")
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i >= num_batches:
                break

            audio = batch["audio_waveform"].to(device)
            embeddings, _ = encoder(audio)

            # Simulate what happens in training - we use the output
            _ = embeddings.mean()

            if i % 10 == 0:
                cpu_mb, gpu_mb = get_memory_mb()
                memory_samples.append((i, cpu_mb, gpu_mb))
                print(
                    f"  Batch {i:3d}: CPU {cpu_mb:.1f} MB (+{cpu_mb - initial_cpu:.1f}), GPU {gpu_mb:.1f} MB (+{gpu_mb - initial_gpu:.1f})"
                )

    final_cpu, final_gpu = get_memory_mb()
    cpu_growth = final_cpu - initial_cpu
    gpu_growth = final_gpu - initial_gpu

    print(f"\nFinal memory - CPU: {final_cpu:.1f} MB, GPU: {final_gpu:.1f} MB")
    print(f"CPU memory growth: {cpu_growth:.1f} MB over {num_batches} batches")
    print(f"GPU memory growth: {gpu_growth:.1f} MB over {num_batches} batches")

    # Check for linear growth pattern (leak indicator)
    if len(memory_samples) >= 3:
        first_growth = memory_samples[1][1] - memory_samples[0][1]
        last_growth = memory_samples[-1][1] - memory_samples[-2][1]

        if cpu_growth > 1000:  # More than 1GB growth
            print("WARNING: Significant CPU memory growth detected!")
            return False
        elif cpu_growth > 500 and last_growth > first_growth:
            print("WARNING: Memory appears to be growing linearly (leak pattern)!")
            return False

    print("PASSED: Encoder memory is stable")
    return True


def test_training_simulation(annotation_path: str, num_batches: int = 100):
    """Simulate actual training loop to check for leaks."""
    print("\n" + "=" * 70)
    print("TEST 3: Training Loop Simulation")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Import the full model
    from src.crescendai.models.lightning_module import PerformanceEvaluationModel

    dimensions = [
        "note_accuracy",
        "rhythmic_stability",
        "articulation_clarity",
        "pedal_technique",
        "tone_quality",
        "dynamic_range",
        "musical_expression",
        "overall_interpretation",
    ]

    print("Creating model...")
    model = PerformanceEvaluationModel(
        audio_dim=768,
        midi_dim=None,  # Audio-only
        shared_dim=512,
        aggregator_dim=512,
        num_dimensions=len(dimensions),
        dimension_names=dimensions,
        modality="audio",
        fusion_type="gated",
        use_projection=True,
        freeze_audio_encoder=True,
    ).to(device)

    model.train()

    train_loader, _, _ = create_dataloaders(
        train_annotation_path=annotation_path,
        val_annotation_path=annotation_path,
        test_annotation_path=None,
        dimension_names=dimensions,
        batch_size=8,
        num_workers=1,
        augmentation_config=None,
        audio_sample_rate=24000,
        max_audio_length=240000,
        max_midi_events=512,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Warm up
    print("Warming up...")
    for i, batch in enumerate(train_loader):
        if i >= 3:
            break
        audio = batch["audio_waveform"].to(device)
        labels = batch["labels"].to(device)

        output = model(audio_waveform=audio)
        if output is not None:
            loss = torch.nn.functional.mse_loss(output["scores"], labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    initial_cpu, initial_gpu = get_memory_mb()
    print(
        f"Initial memory (after warmup) - CPU: {initial_cpu:.1f} MB, GPU: {initial_gpu:.1f} MB"
    )

    memory_samples = []
    start_time = time.time()

    print(f"\nSimulating {num_batches} training iterations...")
    for i, batch in enumerate(train_loader):
        if i >= num_batches:
            break

        audio = batch["audio_waveform"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        output = model(audio_waveform=audio)
        if output is not None:
            loss = torch.nn.functional.mse_loss(output["scores"], labels)

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if i % 20 == 0:
            cpu_mb, gpu_mb = get_memory_mb()
            memory_samples.append((i, cpu_mb, gpu_mb))
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(
                f"  Batch {i:3d}: CPU {cpu_mb:.1f} MB (+{cpu_mb - initial_cpu:.1f}), "
                f"GPU {gpu_mb:.1f} MB (+{gpu_mb - initial_gpu:.1f}), "
                f"Rate: {rate:.2f} it/s"
            )

    final_cpu, final_gpu = get_memory_mb()
    cpu_growth = final_cpu - initial_cpu
    gpu_growth = final_gpu - initial_gpu
    elapsed = time.time() - start_time

    print(f"\n{'=' * 50}")
    print(f"RESULTS after {num_batches} iterations ({elapsed:.1f}s)")
    print(f"{'=' * 50}")
    print(f"Final memory - CPU: {final_cpu:.1f} MB, GPU: {final_gpu:.1f} MB")
    print(f"CPU memory growth: {cpu_growth:.1f} MB")
    print(f"GPU memory growth: {gpu_growth:.1f} MB")
    print(f"Average rate: {num_batches / elapsed:.2f} it/s")

    # Extrapolate to full epoch (7119 batches)
    full_epoch_batches = 7119
    extrapolated_cpu = cpu_growth * (full_epoch_batches / num_batches)
    print(
        f"\nExtrapolated CPU growth for full epoch ({full_epoch_batches} batches): {extrapolated_cpu:.1f} MB"
    )

    if cpu_growth > 500:
        print("\nWARNING: Significant memory growth detected!")
        print("Memory leak likely still present.")
        return False
    elif extrapolated_cpu > 2000:
        print(
            "\nWARNING: Extrapolated growth exceeds 2GB - may still OOM on full epoch"
        )
        return False
    else:
        print("\nPASSED: Memory growth is acceptable")
        print("Training should complete without OOM.")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Test for memory leaks in training pipeline"
    )
    parser.add_argument(
        "--annotation-path",
        type=str,
        required=True,
        help="Path to training annotation JSONL file",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=100,
        help="Number of batches to test (default: 100)",
    )
    parser.add_argument(
        "--test",
        type=str,
        choices=["dataloader", "encoder", "training", "all"],
        default="all",
        help="Which test to run",
    )
    args = parser.parse_args()

    print("Memory Leak Detection Script")
    print("=" * 70)
    print(f"Annotation path: {args.annotation_path}")
    print(f"Test batches: {args.num_batches}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    results = {}

    if args.test in ["dataloader", "all"]:
        results["dataloader"] = test_dataloader_memory(
            args.annotation_path, args.num_batches
        )

    if args.test in ["encoder", "all"]:
        results["encoder"] = test_encoder_memory(args.annotation_path, args.num_batches)

    if args.test in ["training", "all"]:
        results["training"] = test_training_simulation(
            args.annotation_path, args.num_batches
        )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_passed = all(results.values())
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {test_name}: {status}")

    print()
    if all_passed:
        print("All tests passed! Training should complete without OOM.")
    else:
        print("Some tests failed. Review warnings above before running full training.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
