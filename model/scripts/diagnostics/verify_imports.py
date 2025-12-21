#!/usr/bin/env python3
"""
Verify imports work after repository reorganization (commit 4c2c5e2).

Tests all major import paths for:
- percepiano (SOTA replica)
- crescendai (custom multi-modal)
- shared (common utilities)
- backward compatibility shims

Usage:
    cd model
    uv run python scripts/diagnostics/verify_imports.py
"""

import sys
from pathlib import Path


def test_percepiano_imports():
    """Test PercePiano replica imports."""
    errors = []

    # Data
    try:
        from src.percepiano.data import (
            PercePianoVNetDataset,
            PercePianoVNetDataModule,
        )
        print("[OK] percepiano.data.percepiano_vnet_dataset")
    except Exception as e:
        errors.append(f"percepiano.data.percepiano_vnet_dataset: {e}")

    try:
        from src.percepiano.data import PercePianoDataset
        print("[OK] percepiano.data.percepiano_dataset")
    except Exception as e:
        errors.append(f"percepiano.data.percepiano_dataset: {e}")

    # Models
    try:
        from src.percepiano.models import (
            PercePianoVNetModule,
            PercePianoHAN,
            PERCEPIANO_DIMENSIONS,
        )
        print("[OK] percepiano.models.percepiano_replica")
    except Exception as e:
        errors.append(f"percepiano.models.percepiano_replica: {e}")

    try:
        from src.percepiano.models import (
            make_higher_node,
            span_beat_to_note_num,
            find_boundaries_batch,
            compute_actual_lengths,
        )
        print("[OK] percepiano.models.hierarchy_utils")
    except Exception as e:
        errors.append(f"percepiano.models.hierarchy_utils: {e}")

    try:
        from src.percepiano.models import HanEncoder
        print("[OK] percepiano.models.han_encoder")
    except Exception as e:
        errors.append(f"percepiano.models.han_encoder: {e}")

    return errors


def test_crescendai_imports():
    """Test CrescendAI custom imports."""
    errors = []

    # Data
    try:
        from src.crescendai.data import PerformanceDataset, create_dataloaders
        print("[OK] crescendai.data.dataset")
    except Exception as e:
        errors.append(f"crescendai.data.dataset: {e}")

    try:
        from src.crescendai.data import OctupleMIDITokenizer
        print("[OK] crescendai.data.midi_processing")
    except Exception as e:
        errors.append(f"crescendai.data.midi_processing: {e}")

    # Models
    try:
        from src.crescendai.models import PerformanceEvaluationModel
        print("[OK] crescendai.models.lightning_module")
    except Exception as e:
        errors.append(f"crescendai.models.lightning_module: {e}")

    # Losses
    try:
        from src.crescendai.losses import UncertaintyWeightedLoss
        print("[OK] crescendai.losses.uncertainty_loss")
    except Exception as e:
        errors.append(f"crescendai.losses.uncertainty_loss: {e}")

    return errors


def test_shared_imports():
    """Test shared component imports."""
    errors = []

    # Models
    try:
        from src.shared.models import MultiTaskHead, HierarchicalAggregator
        print("[OK] shared.models")
    except Exception as e:
        errors.append(f"shared.models: {e}")

    # Evaluation
    try:
        from src.shared.evaluation import compute_all_metrics
        print("[OK] shared.evaluation.metrics")
    except Exception as e:
        errors.append(f"shared.evaluation.metrics: {e}")

    # Callbacks
    try:
        from src.shared.callbacks import StagedUnfreezingCallback
        print("[OK] shared.callbacks.unfreezing")
    except Exception as e:
        errors.append(f"shared.callbacks.unfreezing: {e}")

    # Utils
    try:
        from src.shared.utils import validate_data_files
        print("[OK] shared.utils.preflight_validation")
    except Exception as e:
        errors.append(f"shared.utils.preflight_validation: {e}")

    return errors


def test_backward_compat_imports():
    """Test backward compatibility shim imports."""
    errors = []

    # These should route to the correct subpackages
    try:
        from src.models import PercePianoVNetModule
        print("[OK] src.models (backward compat)")
    except Exception as e:
        errors.append(f"src.models backward compat: {e}")

    try:
        from src.data import PercePianoVNetDataset
        print("[OK] src.data (backward compat)")
    except Exception as e:
        errors.append(f"src.data backward compat: {e}")

    return errors


def main():
    print("=" * 60)
    print("Import Verification After Reorganization")
    print("=" * 60)
    print()

    all_errors = []

    print("--- PercePiano Imports ---")
    all_errors.extend(test_percepiano_imports())
    print()

    print("--- CrescendAI Imports ---")
    all_errors.extend(test_crescendai_imports())
    print()

    print("--- Shared Imports ---")
    all_errors.extend(test_shared_imports())
    print()

    print("--- Backward Compatibility ---")
    all_errors.extend(test_backward_compat_imports())
    print()

    print("=" * 60)
    if all_errors:
        print(f"FAILED: {len(all_errors)} import errors")
        print("=" * 60)
        for error in all_errors:
            print(f"  - {error}")
        return 1
    else:
        print("SUCCESS: All imports verified")
        print("=" * 60)
        return 0


if __name__ == "__main__":
    sys.exit(main())
