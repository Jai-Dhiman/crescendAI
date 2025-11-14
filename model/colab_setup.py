#!/usr/bin/env python3
"""
Complete Colab Setup Script for Training

Run this ONCE before training to:
1. Filter annotation files (remove missing/corrupted files)
2. Generate clean training config in /content/training/

Usage in Colab:
    !python model/colab_setup.py

Then train:
    !python model/train.py --config /content/training/training_config.yaml
"""

import json
import time
import yaml
from pathlib import Path
from tqdm import tqdm


# ============================================================================
# Configuration
# ============================================================================

DRIVE_BASE = Path("/content/drive/MyDrive/crescendai_data")
ANNOTATIONS_DIR = DRIVE_BASE / "annotations"
OUTPUT_DIR = Path("/content/training")
CONFIG_OUTPUT = OUTPUT_DIR / "training_config.yaml"

# Input annotation files (with _colab suffix)
INPUT_FILES = {
    "train": ANNOTATIONS_DIR / "synthetic_train_colab.jsonl",
    "val": ANNOTATIONS_DIR / "synthetic_val_colab.jsonl",
    "test": ANNOTATIONS_DIR / "synthetic_test_colab.jsonl",
}

# Output filtered annotation files
OUTPUT_FILES = {
    "train": OUTPUT_DIR / "synthetic_train_filtered.jsonl",
    "val": OUTPUT_DIR / "synthetic_val_filtered.jsonl",
    "test": OUTPUT_DIR / "synthetic_test_filtered.jsonl",
}


# ============================================================================
# Helper Functions
# ============================================================================

def check_file_exists_with_retry(file_path: Path, max_retries: int = 3, delay: float = 0.5) -> bool:
    """Check if file exists with retry logic for Google Drive."""
    for attempt in range(max_retries):
        try:
            return file_path.exists()
        except OSError:
            if attempt < max_retries - 1:
                time.sleep(delay)
                continue
            else:
                return False
    return False


def filter_annotation_file(input_path: Path, output_path: Path, require_midi: bool = True):
    """Filter annotation file to remove samples with missing files."""
    print(f"\nFiltering: {input_path.name}")

    # Read annotations
    annotations = []
    with open(input_path, 'r') as f:
        for line in f:
            if line.strip():
                annotations.append(json.loads(line))

    print(f"  Total samples: {len(annotations)}")

    # Filter annotations
    valid_annotations = []
    missing_audio = 0
    missing_midi = 0

    for ann in tqdm(annotations, desc="  Checking files"):
        # Check audio file
        audio_path = Path(ann['audio_path'])
        if not check_file_exists_with_retry(audio_path, max_retries=3, delay=0.3):
            missing_audio += 1
            continue

        # Check MIDI file if required
        if require_midi:
            midi_path = Path(ann.get('midi_path', ''))
            if not midi_path or not ann.get('midi_path'):
                missing_midi += 1
                continue

            if not check_file_exists_with_retry(midi_path, max_retries=3, delay=0.3):
                missing_midi += 1
                continue

        valid_annotations.append(ann)
        time.sleep(0.02)  # Avoid Drive throttling

    # Write filtered annotations
    with open(output_path, 'w') as f:
        for ann in valid_annotations:
            f.write(json.dumps(ann) + '\n')

    # Print summary
    filtered = len(annotations) - len(valid_annotations)
    print(f"  ✓ Kept: {len(valid_annotations)} ({100 * len(valid_annotations) / len(annotations):.1f}%)")
    print(f"  ✗ Filtered: {filtered} ({100 * filtered / len(annotations):.1f}%)")
    if missing_audio > 0:
        print(f"    - Missing audio: {missing_audio}")
    if missing_midi > 0:
        print(f"    - Missing MIDI: {missing_midi}")


def create_training_config():
    """Create clean training configuration."""
    config = {
        "data": {
            "train_path": str(OUTPUT_FILES["train"]),
            "val_path": str(OUTPUT_FILES["val"]),
            "test_path": str(OUTPUT_FILES["test"]),
            "dimensions": [
                "note_accuracy",
                "rhythmic_precision",
                "dynamics_control",
                "articulation",
                "pedaling",
                "tone_quality"
            ],
            "audio_sample_rate": 24000,
            "max_audio_length": 240000,
            "max_midi_events": 512,
            "batch_size": 8,
            "num_workers": 2,  # Reduced to avoid Drive throttling
            "pin_memory": True,
            "augmentation": {
                "enabled": True,
                "pitch_shift": {
                    "enabled": True,
                    "probability": 0.3,
                    "min_semitones": -2,
                    "max_semitones": 2
                },
                "time_stretch": {
                    "enabled": True,
                    "probability": 0.3,
                    "min_rate": 0.85,
                    "max_rate": 1.15
                },
                "add_noise": {
                    "enabled": True,
                    "probability": 0.2,
                    "min_snr_db": 25,
                    "max_snr_db": 40
                },
                "room_acoustics": {
                    "enabled": True,
                    "probability": 0.2,
                    "num_room_types": 5
                },
                "compress_audio": {
                    "enabled": True,
                    "probability": 0.15,
                    "bitrates": [128, 192, 256, 320]
                },
                "gain_variation": {
                    "enabled": True,
                    "probability": 0.3,
                    "min_db": -6,
                    "max_db": 6
                },
                "max_transforms": 3
            }
        },
        "model": {
            "audio_dim": 768,
            "midi_dim": 256,
            "fusion_dim": 1024,
            "aggregator_dim": 512,
            "num_dimensions": 6,
            "mert_model_name": "m-a-p/MERT-v1-95M",
            "freeze_audio_encoder": False,
            "gradient_checkpointing": True,
            "midi_hidden_size": 256,
            "midi_num_layers": 6,
            "midi_num_heads": 4,
            "fusion_num_heads": 8,
            "fusion_dropout": 0.1,
            "lstm_hidden": 256,
            "lstm_layers": 2,
            "attention_heads": 4,
            "aggregator_dropout": 0.2,
            "shared_hidden": 256,
            "task_hidden": 128,
            "mtl_dropout": 0.1
        },
        "training": {
            "max_epochs": 18,
            "precision": 16,
            "optimizer": "AdamW",
            "learning_rate": 1e-5,
            "backbone_lr": 1e-5,
            "heads_lr": 1e-4,
            "weight_decay": 0.01,
            "scheduler": "cosine",
            "warmup_steps": 500,
            "min_lr": 1e-6,
            "gradient_clip_val": 1.0,
            "accumulate_grad_batches": 4,
            "val_check_interval": 1.0,
            "limit_val_batches": 1.0,
            "track_correlations": True
        },
        "callbacks": {
            "checkpoint": {
                "monitor": "val_loss",
                "mode": "min",
                "save_top_k": 3,
                "save_last": True,
                "dirpath": "/content/checkpoints",
                "filename": "baseline-{epoch:02d}-{val_loss:.4f}"
            },
            "early_stopping": {
                "monitor": "val_loss",
                "mode": "min",
                "patience": 5,
                "min_delta": 0.001
            },
            "lr_monitor": {
                "logging_interval": "step"
            }
        },
        "logging": {
            "log_every_n_steps": 50,
            "use_wandb": False,
            "wandb_project": "piano-eval-baseline",
            "wandb_entity": None,
            "wandb_run_name": "baseline-synthetic",
            "use_tensorboard": True,
            "tensorboard_logdir": "/content/logs"
        },
        "seed": 42
    }

    return config


# ============================================================================
# Main Setup
# ============================================================================

def main():
    print("="*80)
    print("CRESCENDAI COLAB TRAINING SETUP")
    print("="*80)
    print()

    # Check Google Drive mount
    print("Step 1: Checking Google Drive mount...")
    if not DRIVE_BASE.exists():
        print("✗ Google Drive not mounted or data directory not found")
        print(f"  Expected: {DRIVE_BASE}")
        print("\nPlease run:")
        print("  from google.colab import drive")
        print("  drive.mount('/content/drive')")
        return 1
    print(f"✓ Data directory found: {DRIVE_BASE}")
    print()

    # Create output directory
    print("Step 2: Creating output directories...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Path("/content/checkpoints").mkdir(parents=True, exist_ok=True)
    Path("/content/logs").mkdir(parents=True, exist_ok=True)
    print("✓ Output directories created")
    print()

    # Filter annotation files
    print("Step 3: Filtering annotation files...")
    print("This will remove samples with missing/corrupted audio or MIDI files")
    print("(This may take 5-10 minutes due to Google Drive throttling)")
    print()

    total_original = 0
    total_filtered = 0

    for split, input_path in INPUT_FILES.items():
        if not input_path.exists():
            print(f"⚠ Skipping {split}: File not found")
            continue

        output_path = OUTPUT_FILES[split]
        filter_annotation_file(input_path, output_path, require_midi=True)

        # Count samples
        with open(output_path, 'r') as f:
            filtered_count = sum(1 for _ in f)
        total_filtered += filtered_count

    print()
    print(f"Total samples after filtering: {total_filtered:,}")
    print()

    # Create training config
    print("Step 4: Creating training configuration...")
    config = create_training_config()

    with open(CONFIG_OUTPUT, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"✓ Config saved to: {CONFIG_OUTPUT}")
    print()

    # Print summary
    print("="*80)
    print("SETUP COMPLETE!")
    print("="*80)
    print()
    print("Next steps:")
    print()
    print("1. (Optional) Run preflight check:")
    print(f"   !python model/preflight_check.py --config {CONFIG_OUTPUT}")
    print()
    print("2. Start training:")
    print(f"   !python model/train.py --config {CONFIG_OUTPUT}")
    print()
    print("3. After training, copy checkpoints to Drive:")
    print("   !mkdir -p /content/drive/MyDrive/crescendai_checkpoints")
    print("   !cp -r /content/checkpoints/* /content/drive/MyDrive/crescendai_checkpoints/")
    print()
    print("Configuration summary:")
    print(f"  - Training samples: {total_filtered:,}")
    print(f"  - Model: Multi-modal (Audio + MIDI)")
    print(f"  - Batch size: 8 (effective: 32 with grad accum)")
    print(f"  - Max epochs: 18")
    print(f"  - Checkpoints: /content/checkpoints")
    print(f"  - Logs: /content/logs")
    print()

    return 0


if __name__ == "__main__":
    exit(main())
