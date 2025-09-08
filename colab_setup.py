#!/usr/bin/env python3
"""
Google Colab Setup Script for Piano Perception Transformer
Run this first in Colab to set up the environment
"""

import os
import subprocess
import sys
from pathlib import Path


def setup_colab_environment():
    """Setup complete environment for training in Colab"""

    print("🚀 Setting up Piano Perception Transformer in Colab")
    print("=" * 60)

    # 1. Install JAX with GPU support
    print("📦 Installing JAX with GPU support...")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--upgrade",
            "jax[cuda12_pip]==0.4.28",
            "-f",
            "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html",
        ]
    )

    # 2. Install other ML dependencies
    print("📦 Installing ML frameworks...")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "flax==0.8.4",
            "optax==0.1.9",
            "orbax-checkpoint==0.4.4",
        ]
    )

    # 3. Install audio processing
    print("🎵 Installing audio processing libraries...")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "librosa==0.10.1",
            "soundfile==0.12.1",
            "audioread==3.0.1",
        ]
    )

    # 4. Install scientific computing
    print("🔬 Installing scientific computing...")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "numpy>=1.24.0",
            "scipy>=1.10.0",
            "pandas>=2.0.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "tqdm>=4.65.0",
            "scikit-learn>=1.3.0",
        ]
    )

    # 5. Verify JAX GPU setup
    print("✅ Verifying JAX GPU setup...")
    import jax

    print(f"   JAX version: {jax.__version__}")
    print(f"   JAX devices: {jax.devices()}")
    print(f"   JAX backend: {jax.default_backend()}")

    # 6. Create directory structure
    print("📁 Creating directory structure...")
    dirs = ["checkpoints", "results", "data", "models", "logs"]

    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"   ✓ {dir_name}/")

    print("\n🎉 Setup complete! Ready for training.")
    print("=" * 60)


def check_gpu_memory():
    """Check available GPU memory"""
    try:
        import jax

        if len(jax.devices("gpu")) > 0:
            # Get memory info
            from jax.lib import xla_bridge

            backend = xla_bridge.get_backend()
            print(f"🔧 GPU Backend: {backend.platform}")

            # Simple memory test
            x = jax.random.normal(jax.random.PRNGKey(0), (1000, 1000))
            x = jax.device_put(x, jax.devices("gpu")[0])
            print(f"✓ GPU memory test passed")
            print(f"✓ Tensor shape: {x.shape}")
            del x
        else:
            print("⚠️  No GPU detected, using CPU")
    except Exception as e:
        print(f"❌ GPU check failed: {e}")


def download_sample_data():
    """Download a small sample of data for testing"""
    print("💾 Setting up sample data...")

    # Create synthetic data for immediate testing
    import numpy as np

    sample_spectrograms = np.random.randn(10, 128, 128).astype(np.float32)
    sample_labels = np.random.rand(10, 19).astype(np.float32)

    np.save("data/sample_spectrograms.npy", sample_spectrograms)
    np.save("data/sample_labels.npy", sample_labels)

    print("✓ Sample synthetic data created")
    print(f"  Spectrograms: {sample_spectrograms.shape}")
    print(f"  Labels: {sample_labels.shape}")


if __name__ == "__main__":
    try:
        setup_colab_environment()
        check_gpu_memory()
        download_sample_data()
        print("\n🚀 Ready to start training!")

    except Exception as e:
        print(f"❌ Setup failed: {e}")
        raise
