#!/usr/bin/env python3
"""
Colab Setup for VS Code Integration

Run this ONCE when starting a new Colab session from VS Code:
    python colab_setup.py

This script:
1. Mounts Google Drive (for training data only)
2. Installs dependencies via uv
3. Verifies GPU access
4. Creates training config pointing to Drive data

No git clone needed - you're already working with local files via VS Code!

After running this, you can run train.py directly from VS Code.
"""

import os
import sys
from pathlib import Path


# ============================================================================
# Configuration
# ============================================================================

DRIVE_BASE = Path("/content/drive/MyDrive/crescendai_data")
CHECKPOINT_DIR = Path("/content/drive/MyDrive/crescendai_checkpoints")


# ============================================================================
# Helper Functions
# ============================================================================

def mount_google_drive():
    """Mount Google Drive for data access."""
    print("="*70)
    print("MOUNTING GOOGLE DRIVE (for training data)")
    print("="*70)

    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        print("✓ Google Drive mounted")
    except ImportError:
        print("✗ Not running in Colab - skipping Drive mount")
        return False
    except Exception as e:
        print(f"✗ Failed to mount Drive: {e}")
        return False

    # Verify data exists
    if not DRIVE_BASE.exists():
        print(f"\n✗ Data directory not found: {DRIVE_BASE}")
        print("\nExpected structure:")
        print("  MyDrive/crescendai_data/")
        print("    all_segments/")
        print("    annotations/")
        return False

    print(f"✓ Data directory: {DRIVE_BASE}")

    # Create checkpoint directory
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Checkpoint directory: {CHECKPOINT_DIR}")

    existing_ckpts = list(CHECKPOINT_DIR.glob('*.ckpt'))
    if existing_ckpts:
        print(f"✓ Found {len(existing_ckpts)} existing checkpoint(s)")

    print("="*70)
    return True


def verify_gpu():
    """Verify GPU is available."""
    print("\n" + "="*70)
    print("GPU VERIFICATION")
    print("="*70)

    import torch

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU memory: {memory_gb:.1f} GB")
        print("\n✓ GPU READY")
    else:
        print("\n⚠️  WARNING: No GPU detected")
        print("Training will be extremely slow")

    print("="*70)
    return torch.cuda.is_available()


def install_dependencies():
    """Install Python dependencies using uv."""
    print("\n" + "="*70)
    print("INSTALLING DEPENDENCIES")
    print("="*70)

    # Check if uv is installed
    uv_path = Path.home() / ".cargo" / "bin" / "uv"

    if not uv_path.exists():
        print("Installing uv...")
        os.system("curl -LsSf https://astral.sh/uv/install.sh | sh")

        # Add to PATH
        cargo_bin = str(Path.home() / ".cargo" / "bin")
        os.environ['PATH'] = f"{cargo_bin}:{os.environ.get('PATH', '')}"
        print("✓ uv installed")
    else:
        print("✓ uv already installed")

    # Install dependencies (from local pyproject.toml)
    print("\nInstalling packages...")
    result = os.system("uv pip install --system -e .")

    if result != 0:
        print("\n✗ Failed to install dependencies")
        return False

    print("✓ Dependencies installed")
    print("="*70)
    return True


def setup_environment():
    """Set up environment variables."""
    print("\n" + "="*70)
    print("ENVIRONMENT SETUP")
    print("="*70)

    # Non-interactive matplotlib
    os.environ['MPLBACKEND'] = 'Agg'
    print("✓ Matplotlib backend: Agg")

    # Suppress warnings
    import warnings
    warnings.filterwarnings('ignore', message='divide by zero encountered in scalar divide')
    print("✓ Suppressed MIDI warnings")

    print("="*70)


def print_next_steps():
    """Print what to do after setup."""
    print("\n" + "="*80)
    print(" "*25 + "SETUP COMPLETE!")
    print("="*80)
    print("\nYour local code is ready to run on Colab GPU!")
    print("\nNext steps:")
    print("\n1. (Optional) Run preflight check:")
    print("   python preflight_check.py")
    print("\n2. Start training:")
    print("   python train.py --config /tmp/training_config.yaml")
    print("\n   Or create a custom config first:")
    print("   python create_config.py")
    print("\n3. Monitor with TensorBoard (in VS Code or Colab):")
    print("   tensorboard --logdir /content/logs")
    print("\nCheckpoints save to: /content/drive/MyDrive/crescendai_checkpoints")
    print("="*80)


# ============================================================================
# Main Setup
# ============================================================================

def main():
    """Run Colab setup for VS Code workflow."""
    print("\n" + "="*80)
    print(" "*20 + "COLAB SETUP FOR VS CODE")
    print("="*80)
    print("\nSetting up Colab environment for local code execution...")
    print()

    # Step 1: Mount Google Drive (for data only)
    if not mount_google_drive():
        print("\n✗ Setup failed at Drive mount")
        return 1

    # Step 2: Install dependencies
    if not install_dependencies():
        print("\n✗ Setup failed at dependency installation")
        return 1

    # Step 3: Verify GPU
    has_gpu = verify_gpu()
    if not has_gpu:
        print("\n⚠️  No GPU - training will be slow")

    # Step 4: Setup environment
    setup_environment()

    # Success!
    print_next_steps()

    return 0


if __name__ == "__main__":
    sys.exit(main())
