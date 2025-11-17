#!/usr/bin/env python3
"""
Setup Colab environment for optimal audio training performance.

Fixes:
1. Audio backend issues (PySoundFile/librosa warnings)
2. Install required system dependencies
3. Verify GPU and memory
4. Configure optimal settings

Run this FIRST before any training.
"""

import subprocess
import sys
import warnings


def run_command(cmd, description, show_output=False):
    """Run shell command with error handling."""
    print(f"  {description}...", end=" ", flush=True)
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=not show_output,
            text=True,
            check=False
        )
        if result.returncode == 0:
            print("✓")
            return True
        else:
            print(f"✗ (exit code {result.returncode})")
            if result.stderr and not show_output:
                print(f"    Error: {result.stderr[:200]}")
            return False
    except Exception as e:
        print(f"✗ ({str(e)})")
        return False


def install_audio_backends():
    """Install proper audio backends for fast, reliable loading."""
    print("\n1. Installing audio backends...")

    # Install system dependencies
    run_command(
        "apt-get update -qq && apt-get install -y -qq libsndfile1 ffmpeg libavcodec-extra",
        "Installing libsndfile1 + ffmpeg"
    )

    # Verify Python packages
    run_command(
        "pip install -q --upgrade soundfile audioread",
        "Updating soundfile + audioread"
    )


def verify_installations():
    """Verify all required packages are installed correctly."""
    print("\n2. Verifying installations...")

    packages = {
        'soundfile': 'Audio I/O (primary backend)',
        'librosa': 'Audio processing',
        'torch': 'PyTorch',
        'torchaudio': 'PyTorch audio',
        'pytorch_lightning': 'Training framework',
        'transformers': 'MERT model',
    }

    all_good = True
    for package, description in packages.items():
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"  ✓ {package:20s} {version:15s} ({description})")
        except ImportError:
            print(f"  ✗ {package:20s} MISSING ({description})")
            all_good = False

    return all_good


def check_gpu():
    """Check GPU availability and memory."""
    print("\n3. Checking GPU...")

    try:
        import torch

        if not torch.cuda.is_available():
            print("  ✗ NO GPU DETECTED!")
            print("    Enable GPU: Runtime → Change runtime type → T4 GPU")
            return False

        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

        print(f"  ✓ GPU: {gpu_name}")
        print(f"  ✓ Memory: {gpu_memory:.1f} GB")

        # Check if sufficient memory
        if gpu_memory < 10:
            print(f"  ⚠ Warning: {gpu_memory:.1f} GB may be insufficient (need ~12GB for MERT-95M)")

        return True

    except Exception as e:
        print(f"  ✗ GPU check failed: {e}")
        return False


def test_audio_loading():
    """Test that audio loading works without falling back to audioread."""
    print("\n4. Testing audio backend...")

    try:
        import soundfile as sf
        import numpy as np
        import tempfile
        import os

        # Create a test audio file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            test_file = f.name

        # Write test audio (1 second at 24kHz)
        test_audio = np.random.randn(24000).astype(np.float32) * 0.1
        sf.write(test_file, test_audio, 24000)

        # Try loading with soundfile directly
        audio, sr = sf.read(test_file)
        print(f"  ✓ soundfile: Loaded {len(audio)} samples at {sr} Hz")

        # Try loading with librosa (should use soundfile backend)
        import librosa

        # Suppress warnings temporarily to check backend
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            audio_librosa, sr_librosa = librosa.load(test_file, sr=24000)

            # Check if PySoundFile warning appeared
            pysoundfile_warnings = [
                warning for warning in w
                if 'PySoundFile failed' in str(warning.message)
            ]

            if pysoundfile_warnings:
                print("  ✗ librosa falling back to audioread (SLOW!)")
                print("     This may cause training slowdowns")
                success = False
            else:
                print(f"  ✓ librosa: Loaded {len(audio_librosa)} samples (using soundfile backend)")
                success = True

        # Clean up
        os.remove(test_file)

        return success

    except Exception as e:
        print(f"  ✗ Audio loading test failed: {e}")
        return False


def configure_warnings():
    """Configure warning filters for cleaner output."""
    print("\n5. Configuring warning filters...")

    # Filter common but harmless warnings
    warnings.filterwarnings('ignore', message='divide by zero')
    warnings.filterwarnings('ignore', category=SyntaxWarning)
    warnings.filterwarnings('ignore', message='PySoundFile failed')
    warnings.filterwarnings('ignore', category=FutureWarning, module='librosa')

    print("  ✓ Suppressed harmless warnings")


def set_pytorch_optimizations():
    """Set PyTorch optimizations for better performance."""
    print("\n6. Configuring PyTorch optimizations...")

    try:
        import torch

        # Enable Tensor Cores for better performance
        torch.set_float32_matmul_precision('high')
        print("  ✓ Enabled high-precision matmul (Tensor Cores)")

        # Enable cudNN benchmarking for faster convolutions
        torch.backends.cudnn.benchmark = True
        print("  ✓ Enabled cudNN auto-tuner")

        # Disable gradient for inference
        torch.set_grad_enabled(True)
        print("  ✓ PyTorch optimizations applied")

        return True

    except Exception as e:
        print(f"  ✗ PyTorch optimization failed: {e}")
        return False


def main():
    """Main setup routine."""
    print("="*70)
    print("COLAB ENVIRONMENT SETUP")
    print("="*70)
    print("\nOptimizing environment for audio model training...")

    # Run all setup steps
    install_audio_backends()
    packages_ok = verify_installations()
    gpu_ok = check_gpu()
    audio_ok = test_audio_loading()
    configure_warnings()
    pytorch_ok = set_pytorch_optimizations()

    # Summary
    print("\n" + "="*70)
    print("SETUP SUMMARY")
    print("="*70)

    status = {
        "Python packages": packages_ok,
        "GPU": gpu_ok,
        "Audio backend": audio_ok,
        "PyTorch optimizations": pytorch_ok,
    }

    for item, ok in status.items():
        symbol = "✓" if ok else "✗"
        print(f"  {symbol} {item}")

    if all(status.values()):
        print("\n✓ Environment is ready for training!")
        return 0
    else:
        print("\n⚠ Some issues detected - training may be slow or fail")
        print("   Review errors above and fix before training")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
