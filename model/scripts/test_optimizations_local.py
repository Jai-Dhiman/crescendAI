#!/usr/bin/env python3
"""
Local test script to verify optimizations work before deploying to Colab.

Run this locally to catch any issues before uploading to Colab.
"""

import sys
from pathlib import Path


def test_imports():
    """Test that all required modules can be imported."""
    print("\n1. Testing imports...")

    required_modules = [
        'torch',
        'torchaudio',
        'pytorch_lightning',
        'librosa',
        'soundfile',
        'yaml',
        'transformers',
    ]

    all_ok = True
    for module_name in required_modules:
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"  ✓ {module_name:20s} {version}")
        except ImportError as e:
            print(f"  ✗ {module_name:20s} MISSING: {e}")
            all_ok = False

    return all_ok


def test_audio_processing():
    """Test audio loading with torchaudio."""
    print("\n2. Testing audio processing...")

    try:
        from src.data.audio_processing import load_audio, TORCHAUDIO_AVAILABLE

        if TORCHAUDIO_AVAILABLE:
            print("  ✓ torchaudio available (FAST)")
        else:
            print("  ⚠ torchaudio not available, will use librosa (slower)")

        # Test that function exists and has correct signature
        import inspect
        sig = inspect.signature(load_audio)
        params = list(sig.parameters.keys())

        required_params = ['path', 'sr', 'mono', 'prefer_torchaudio']
        missing = [p for p in required_params if p not in params]

        if missing:
            print(f"  ✗ load_audio missing parameters: {missing}")
            return False

        print("  ✓ load_audio function signature correct")
        return True

    except Exception as e:
        print(f"  ✗ Audio processing test failed: {e}")
        return False


def test_scripts_exist():
    """Test that all optimization scripts exist."""
    print("\n3. Checking optimization scripts...")

    scripts = [
        'scripts/setup_colab_environment.py',
        'scripts/copy_data_to_local.py',
        'scripts/preflight_check.py',
    ]

    all_ok = True
    for script in scripts:
        path = Path(script)
        if path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"  ✓ {script:45s} ({size_kb:.1f} KB)")
        else:
            print(f"  ✗ {script:45s} MISSING")
            all_ok = False

    return all_ok


def test_config():
    """Test that config file is valid and uses local paths."""
    print("\n4. Testing config file...")

    try:
        import yaml

        config_path = Path('configs/experiment_10k.yaml')
        if not config_path.exists():
            print(f"  ✗ Config not found: {config_path}")
            return False

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Check paths
        train_path = config['data']['train_path']
        num_workers = config['data']['num_workers']

        print(f"  Train path: {train_path}")
        print(f"  Num workers: {num_workers}")

        # Verify paths are local (not Drive)
        if '/tmp/crescendai_data/' in train_path:
            print("  ✓ Config uses local SSD paths")
        else:
            print("  ⚠ Config may not be using optimal paths")

        # Verify num_workers
        if num_workers > 0:
            print(f"  ✓ Parallel loading enabled (num_workers={num_workers})")
        else:
            print("  ⚠ Parallel loading disabled (num_workers=0)")

        # Check all required fields
        required = ['dimensions', 'batch_size', 'audio_sample_rate']
        missing = [f for f in required if f not in config['data']]

        if missing:
            print(f"  ✗ Config missing fields: {missing}")
            return False

        print(f"  ✓ Config valid with {len(config['data']['dimensions'])} dimensions")
        return True

    except Exception as e:
        print(f"  ✗ Config test failed: {e}")
        return False


def test_model_import():
    """Test that model can be imported."""
    print("\n5. Testing model import...")

    try:
        from src.models.lightning_module import PerformanceEvaluationModel

        # Check that class has required methods
        required_methods = ['forward', 'training_step', 'validation_step']
        missing = [m for m in required_methods if not hasattr(PerformanceEvaluationModel, m)]

        if missing:
            print(f"  ✗ Model missing methods: {missing}")
            return False

        print("  ✓ Model class imported successfully")
        return True

    except Exception as e:
        print(f"  ✗ Model import failed: {e}")
        return False


def test_notebook_exists():
    """Test that notebook file exists and is valid."""
    print("\n6. Checking notebook...")

    try:
        import json

        nb_path = Path('notebooks/train_full_model.ipynb')
        if not nb_path.exists():
            print(f"  ✗ Notebook not found: {nb_path}")
            return False

        # Parse notebook
        with open(nb_path) as f:
            nb = json.load(f)

        num_cells = len(nb.get('cells', []))
        print(f"  ✓ Notebook has {num_cells} cells")

        # Check for optimization cells
        cell_sources = [
            ''.join(cell.get('source', []))
            for cell in nb.get('cells', [])
        ]

        has_setup = any('setup_colab_environment' in src for src in cell_sources)
        has_copy = any('copy_data_to_local' in src for src in cell_sources)
        has_preflight = any('preflight_check' in src for src in cell_sources)

        if has_setup and has_copy and has_preflight:
            print("  ✓ Notebook has all optimization cells")
        else:
            print("  ⚠ Notebook may be missing optimization cells:")
            if not has_setup:
                print("     - setup_colab_environment")
            if not has_copy:
                print("     - copy_data_to_local")
            if not has_preflight:
                print("     - preflight_check")

        return True

    except Exception as e:
        print(f"  ✗ Notebook check failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*70)
    print("LOCAL OPTIMIZATION TESTS")
    print("="*70)
    print("\nTesting optimizations before deploying to Colab...\n")

    # Change to model directory if needed
    if Path('model').exists():
        import os
        os.chdir('model')

    tests = {
        'Imports': test_imports(),
        'Audio processing': test_audio_processing(),
        'Scripts': test_scripts_exist(),
        'Config': test_config(),
        'Model': test_model_import(),
        'Notebook': test_notebook_exists(),
    }

    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    for test_name, passed in tests.items():
        symbol = "✓" if passed else "✗"
        print(f"  {symbol} {test_name}")

    all_passed = all(tests.values())

    print("="*70)

    if all_passed:
        print("\n✓ ALL TESTS PASSED!")
        print("\nReady to deploy to Colab:")
        print("  1. Commit and push changes")
        print("  2. In Colab: git pull")
        print("  3. Run optimization workflow in notebook")
        return 0
    else:
        print("\n⚠ SOME TESTS FAILED")
        print("\nFix issues above before deploying to Colab")
        return 1


if __name__ == "__main__":
    sys.exit(main())
