#!/usr/bin/env python3
"""
CrescendAI Local Deployment Script
Deploy and test the piano performance analysis service locally with JAX[cpu]
"""

import subprocess
import sys
import os
from pathlib import Path
import json
import time


def check_local_environment():
    """Check if local environment is properly set up"""
    try:
        # Check if we're in a virtual environment
        if not (
            hasattr(sys, "real_prefix")
            or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
        ):
            print("⚠️  Not in a virtual environment. Consider activating venv.")
        # Check Python version
        if sys.version_info < (3, 9):
            print("❌ Python 3.9+ required")
            return False

        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")

        # Check JAX[cpu] installation
        try:
            import jax

            devices = jax.devices()
            print(f"✅ JAX devices: {[str(d) for d in devices]}")
            if any("gpu" in str(d).lower() for d in devices):
                print("⚠️  GPU devices detected - using JAX[cpu] for local deployment")
        except ImportError:
            print("❌ JAX not found. Run: uv add jax[cpu]")
            return False

        return True

    except Exception as e:
        print(f"❌ Environment check failed: {e}")
        return False


def install_local_dependencies():
    """Install dependencies optimized for local CPU development"""
    print("📦 Installing local dependencies with JAX[cpu]...")

    try:
        # Install CPU-optimized dependencies
        result = subprocess.run(
            ["uv", "add", "jax[cpu]>=0.4.13,<0.4.30"],
            capture_output=True,
            text=True,
            check=True,
        )

        print("✅ Local dependencies installed successfully!")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except FileNotFoundError:
        print("❌ uv not found. Install with: pip install uv")
        return False


def test_local_service():
    """Test the service locally with CPU backend"""
    print("🧪 Testing service locally with JAX[cpu]...")

    try:
        # Set JAX to CPU-only mode
        env = os.environ.copy()
        env["JAX_PLATFORMS"] = "cpu"
        env["CUDA_VISIBLE_DEVICES"] = ""

        # Test the preprocessing pipeline first
        from crescendai_model.utils.preprocessing import create_preprocessing_pipeline

        processor = create_preprocessing_pipeline()

        print("✅ Preprocessing pipeline loaded successfully!")

        # Test model loading (without inference to save time)
        model_path = Path("results/final_finetuned_model.pkl")
        if model_path.exists():
            print(f"✅ Model file found: {model_path}")
        else:
            print("⚠️  Model file not found - training required for full testing")

        return True

    except Exception as e:
        print(f"❌ Local test failed: {e}")
        return False


def run_local_server():
    """Run the FastAPI server locally"""
    print("🚀 Starting local FastAPI server...")

    try:
        # Set JAX to CPU-only mode
        env = os.environ.copy()
        env["JAX_PLATFORMS"] = "cpu"
        env["CUDA_VISIBLE_DEVICES"] = ""

        # Start the server with asyncio loop to avoid uvloop/orbax conflict
        cmd = [
            "uvicorn",
            "crescendai_model.service.api:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8001",
            "--reload",
            "--loop",
            "asyncio",  # Use asyncio instead of uvloop
        ]

        print("🌐 Server will be available at: http://localhost:8001")
        print("📚 API documentation at: http://localhost:8001/docs")
        print("\n🔄 Starting server (Ctrl+C to stop)...")

        # Run the server (this will block)
        subprocess.run(cmd, env=env, check=True)

    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Server failed: {e}")
        return False
    except FileNotFoundError:
        print("❌ uvicorn not found. Run: uv add uvicorn[standard]")
        return False


def create_test_audio():
    """Create a test audio file for local validation"""
    print("🎵 Creating test audio for local validation...")

    try:
        import numpy as np
        import librosa
        import soundfile as sf

        # Generate a simple test audio (piano-like tones)
        sr = 22050
        duration = 10  # 10 seconds
        t = np.linspace(0, duration, int(sr * duration))

        # Create some piano-like frequencies
        frequencies = [261.63, 329.63, 392.00, 523.25]  # C4, E4, G4, C5
        audio = np.zeros_like(t)

        for i, freq in enumerate(frequencies):
            # Simple envelope
            envelope = np.exp(-t * 0.5) * (1 + 0.3 * np.sin(2 * np.pi * 2 * t))
            tone = 0.25 * envelope * np.sin(2 * np.pi * freq * t)
            audio += tone

        # Normalize
        audio = audio / np.max(np.abs(audio))

        # Save test file
        test_path = Path("test_audio_local.wav")
        sf.write(test_path, audio, sr)

        print(f"✅ Test audio created: {test_path} ({duration}s)")
        return test_path

    except ImportError as e:
        print(f"❌ Failed to create test audio: {e}")
        print("Install required packages: uv add librosa soundfile")
        return None


def validate_local_preprocessing():
    """Validate the preprocessing pipeline locally"""
    print("🔄 Validating local preprocessing pipeline...")

    try:
        # Force CPU mode
        os.environ["JAX_PLATFORMS"] = "cpu"

        # Test the preprocessing helpers
        from crescendai_model.utils.preprocessing import create_preprocessing_pipeline

        processor = create_preprocessing_pipeline()

        # Create test audio if needed
        test_audio = create_test_audio()
        if test_audio is None:
            return False

        # Test preprocessing
        result = processor.process_audio_file(str(test_audio))

        if result["status"] == "success":
            shape = result["metadata"]["shape"]
            print(f"✅ Local preprocessing successful: mel-spec shape {shape}")

            # Cleanup
            test_audio.unlink()
            return True
        else:
            print(f"❌ Local preprocessing failed: {result['error']}")
            return False

    except Exception as e:
        print(f"❌ Local preprocessing validation failed: {e}")
        return False


def check_model_file():
    """Check if the trained model file exists"""
    model_path = Path("results/final_finetuned_model.pkl")
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✅ Model file found: {model_path} ({size_mb:.1f}MB)")
        return True
    else:
        print(f"❌ Model file not found: {model_path}")
        print("Run training first: uv run python3 -m crescendai_model.train_ast")
        return False


def main():
    """Main local deployment workflow"""
    print("🎹 CrescendAI Local Deployment (JAX[cpu])")
    print("=" * 50)

    # Pre-flight checks
    print("\n📋 Running pre-flight checks:")

    if not check_local_environment():
        return 1

    print("\n🔄 Running full validation:")
    success = True
    success &= validate_local_preprocessing()
    success &= check_model_file()

    if not success:
        print("\n❌ Validation failed - please fix issues above before deployment")
        return 1

    print("\n✅ All checks passed - starting local FastAPI server...")
    return 0 if run_local_server() else 1


if __name__ == "__main__":
    sys.exit(main())
