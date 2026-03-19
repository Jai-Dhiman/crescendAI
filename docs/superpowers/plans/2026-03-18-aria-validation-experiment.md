# Aria Symbolic Encoder Validation - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate whether Aria's frozen representations capture piano performance quality signal, compare two Aria variants and MuQ via linear probes on 4-fold CV, and compare ByteDance vs Aria AMT on YouTube recordings.

**Architecture:** Extract frozen embeddings from two Aria model variants (embedding 512-dim, base 1536-dim with last-token pooling) for 1,202 PercePiano MIDI files. Train identical linear probes (Linear -> 6 scores) on each embedding set plus mean-pooled MuQ, evaluate pairwise accuracy on clean piece-stratified folds. Compute error correlation between best Aria and MuQ. Separately, compare ByteDance and Aria-AMT on 51 YouTube audio files.

**Tech Stack:** PyTorch, transformers (AutoModel), Aria (EleutherAI -- AbsTokenizer + model), mido, scipy, existing MetricsSuite from `model/src/model_improvement/metrics.py`

**Spec:** `docs/superpowers/specs/2026-03-18-aria-validation-experiment-design.md`

---

## File Structure

```
model/src/model_improvement/
  aria_embeddings.py       # Tokenize MIDI + extract embeddings (both variants)
  aria_linear_probe.py     # Linear probe training, evaluation, error correlation, results table
  aria_amt_compare.py      # AMT comparison on YouTube audio (Stage 0)

model/tests/model_improvement/
  test_aria_embeddings.py   # Tests for tokenization + extraction
  test_aria_linear_probe.py # Tests for probe training + pairwise computation
```

**Key existing files (read-only except `paths.py`):**
- `model/src/model_improvement/metrics.py` -- `MetricsSuite.pairwise_accuracy()`, `regression_r2()`, `format_comparison_table()`
- `model/src/model_improvement/evaluation.py` -- `aggregate_folds()`
- `model/src/model_improvement/taxonomy.py` -- `DIMENSIONS`, `NUM_DIMS`, `load_composite_labels()`
- `model/src/paths.py` -- `Embeddings.percepiano`, `Midi.percepiano`, `Labels.composite`, `Labels.percepiano`, `Evals.youtube_amt` (add `Weights` class)
- `model/data/labels/percepiano/folds.json` -- 4 clean folds, each `{"train": [...], "val": [...]}`
- `model/data/labels/composite/composite_labels.json` -- `{segment_id: {dynamics: float, ...}}`
- `model/data/embeddings/percepiano/muq_embeddings.pt` -- `{segment_id: tensor[frames, 1024]}`, 1201 keys

**Data notes:**
- MuQ embeddings are variable-length frame-level: shape `[N_frames, 1024]` per segment (e.g., `[426, 1024]`). Mean-pool across dim 0 to get `[1024]`.
- Labels have 1,202 keys. MuQ has 1,201 keys (one missing). The probe must handle missing embeddings by skipping those segments.
- MIDI filenames: `{segment_id}.mid` (e.g., `Beethoven_WoO80_thema_8bars_1_1.mid`). Strip `.mid` to get the label key.

---

## Task 1: Setup -- Install Dependencies and Download Weights

**Files:**
- Modify: `model/pyproject.toml` (add aria dependency)
- Modify: `model/src/paths.py` (add `Weights` class)

- [ ] **Step 0: Add `Weights` class to `paths.py`**

In `model/src/paths.py`, add after the `Calibration` class:

```python
class Weights:
    root = DATA_ROOT / "weights"
```

This follows the project convention that all data paths are defined centrally in `paths.py`.

- [ ] **Step 1: Add safetensors to project dependencies**

In `model/pyproject.toml`, add to the `dependencies` list:

```toml
"safetensors>=0.4.0",
```

- [ ] **Step 2: Install aria from git**

Run:
```bash
cd /Users/jdhiman/Documents/crescendai/model
uv add git+https://github.com/EleutherAI/aria.git
```

If this fails due to dependency conflicts, try:
```bash
uv pip install git+https://github.com/EleutherAI/aria.git
```

Record the installed commit hash for reproducibility.

- [ ] **Step 3: Install safetensors**

Run:
```bash
cd /Users/jdhiman/Documents/crescendai/model
uv add safetensors
```

- [ ] **Step 4: Download Aria model weights**

Run:
```bash
cd /Users/jdhiman/Documents/crescendai/model
huggingface-cli download loubb/aria-medium-embedding --local-dir data/weights/aria-medium-embedding
huggingface-cli download loubb/aria-medium-base --local-dir data/weights/aria-medium-base
huggingface-cli download loubb/aria-amt --local-dir data/weights/aria-amt
```

Expected: ~2.5GB per model, 3 directories created under `data/weights/`.

- [ ] **Step 5: Verify imports work**

Run:
```bash
cd /Users/jdhiman/Documents/crescendai/model
uv run python3 -c "
from aria.tokenizer import AbsTokenizer
from transformers import AutoModel
print('AbsTokenizer imported successfully')
print('AutoModel imported successfully')
"
```

Expected: Both imports succeed with no errors.

- [ ] **Step 6: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai/model
git add pyproject.toml uv.lock
git commit -m "deps: add aria and safetensors for symbolic encoder validation"
```

---

## Task 2: Embedding Extraction Script -- `aria_embeddings.py`

**Files:**
- Create: `model/src/model_improvement/aria_embeddings.py`
- Test: `model/tests/model_improvement/test_aria_embeddings.py`

- [ ] **Step 1: Write failing test for MIDI tokenization**

Create `model/tests/model_improvement/test_aria_embeddings.py`:

```python
"""Tests for Aria embedding extraction."""

import tempfile
from pathlib import Path

import mido
import pytest
import torch


def _create_test_midi(path: Path) -> None:
    """Create a minimal valid MIDI file with a few piano notes."""
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=500000))
    # C4 quarter note
    track.append(mido.Message("note_on", note=60, velocity=80, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, time=480))
    # E4 quarter note
    track.append(mido.Message("note_on", note=64, velocity=90, time=0))
    track.append(mido.Message("note_off", note=64, velocity=0, time=480))
    # G4 quarter note
    track.append(mido.Message("note_on", note=67, velocity=70, time=0))
    track.append(mido.Message("note_off", note=67, velocity=0, time=480))
    track.append(mido.MetaMessage("end_of_track", time=0))
    mid.save(str(path))


class TestTokenizeMidi:
    def test_tokenize_returns_list_of_tokens(self):
        from model_improvement.aria_embeddings import tokenize_midi

        with tempfile.TemporaryDirectory() as tmpdir:
            midi_path = Path(tmpdir) / "test.mid"
            _create_test_midi(midi_path)
            tokens = tokenize_midi(midi_path)
            assert isinstance(tokens, list)
            assert len(tokens) > 0
            # Each token should be a tuple (Aria AbsTokenizer format)
            assert all(isinstance(t, (tuple, str)) for t in tokens)

    def test_tokenize_nonexistent_file_raises(self):
        from model_improvement.aria_embeddings import tokenize_midi

        with pytest.raises(FileNotFoundError):
            tokenize_midi(Path("/nonexistent/path.mid"))


class TestExtractEmbeddings:
    def test_extract_single_midi_embedding_variant(self):
        """Test embedding extraction returns correct shape for embedding variant."""
        from model_improvement.aria_embeddings import extract_embedding

        with tempfile.TemporaryDirectory() as tmpdir:
            midi_path = Path(tmpdir) / "test.mid"
            _create_test_midi(midi_path)
            emb = extract_embedding(midi_path, variant="embedding")
            assert isinstance(emb, torch.Tensor)
            assert emb.shape == (512,)
            assert emb.dtype == torch.float32

    def test_extract_single_midi_base_variant(self):
        """Test embedding extraction returns correct shape for base variant."""
        from model_improvement.aria_embeddings import extract_embedding

        with tempfile.TemporaryDirectory() as tmpdir:
            midi_path = Path(tmpdir) / "test.mid"
            _create_test_midi(midi_path)
            emb = extract_embedding(midi_path, variant="base")
            assert isinstance(emb, torch.Tensor)
            assert emb.shape == (1536,)
            assert emb.dtype == torch.float32

    def test_invalid_variant_raises(self):
        from model_improvement.aria_embeddings import extract_embedding

        with tempfile.TemporaryDirectory() as tmpdir:
            midi_path = Path(tmpdir) / "test.mid"
            _create_test_midi(midi_path)
            with pytest.raises(ValueError, match="variant"):
                extract_embedding(midi_path, variant="invalid")


class TestExtractAll:
    def test_extract_all_returns_dict(self):
        """Test batch extraction returns dict with correct keys and shapes."""
        from model_improvement.aria_embeddings import extract_all_embeddings

        with tempfile.TemporaryDirectory() as tmpdir:
            midi_dir = Path(tmpdir) / "midi"
            midi_dir.mkdir()
            _create_test_midi(midi_dir / "segment_a.mid")
            _create_test_midi(midi_dir / "segment_b.mid")

            result = extract_all_embeddings(midi_dir, variant="embedding")
            assert isinstance(result, dict)
            assert "segment_a" in result
            assert "segment_b" in result
            assert result["segment_a"].shape == (512,)
            assert result["segment_b"].shape == (512,)

    def test_extract_all_skips_non_midi(self):
        """Non-.mid files should be ignored."""
        from model_improvement.aria_embeddings import extract_all_embeddings

        with tempfile.TemporaryDirectory() as tmpdir:
            midi_dir = Path(tmpdir) / "midi"
            midi_dir.mkdir()
            _create_test_midi(midi_dir / "segment_a.mid")
            (midi_dir / "readme.txt").write_text("not a midi")

            result = extract_all_embeddings(midi_dir, variant="embedding")
            assert len(result) == 1
            assert "segment_a" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /Users/jdhiman/Documents/crescendai/model
uv run pytest tests/model_improvement/test_aria_embeddings.py -v --no-header 2>&1 | head -30
```

Expected: FAIL with `ModuleNotFoundError: No module named 'model_improvement.aria_embeddings'`

- [ ] **Step 3: Write `aria_embeddings.py` implementation**

Create `model/src/model_improvement/aria_embeddings.py`:

```python
"""Extract frozen Aria embeddings from MIDI files.

Supports two Aria variants:
- "embedding": aria-medium-embedding (512-dim, EOS token)
- "base": aria-medium-base (1536-dim, last-token pooling)

Usage:
    python -m model_improvement.aria_embeddings \
      --variant embedding \
      --midi-dir data/midi/percepiano \
      --output data/embeddings/percepiano/aria_embedding.pt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from transformers import AutoModel

from aria.tokenizer import AbsTokenizer

logger = logging.getLogger(__name__)

# Lazy-loaded model cache (avoids reloading per call)
_MODEL_CACHE: dict[str, AutoModel] = {}
_TOKENIZER_CACHE: dict[str, AbsTokenizer] = {}

VARIANT_CONFIG = {
    "embedding": {
        "model_name": "loubb/aria-medium-embedding",
        "output_dim": 512,
        "max_seq_len": 2048,
    },
    "base": {
        "model_name": "loubb/aria-medium-base",
        "output_dim": 1536,
        "max_seq_len": 8192,
    },
}

# Use centralized path from paths.py
from paths import Weights
_WEIGHTS_ROOT = Weights.root


def _get_tokenizer() -> AbsTokenizer:
    """Get or create a cached AbsTokenizer instance."""
    if "abs" not in _TOKENIZER_CACHE:
        _TOKENIZER_CACHE["abs"] = AbsTokenizer()
    return _TOKENIZER_CACHE["abs"]


def _get_model(variant: str) -> AutoModel:
    """Get or create a cached Aria model for the given variant."""
    if variant not in VARIANT_CONFIG:
        raise ValueError(
            f"Unknown variant '{variant}'. "
            f"Must be one of: {list(VARIANT_CONFIG.keys())}"
        )
    if variant not in _MODEL_CACHE:
        config = VARIANT_CONFIG[variant]
        local_path = _WEIGHTS_ROOT / f"aria-medium-{variant}"
        if local_path.exists():
            model_path = str(local_path)
        else:
            model_path = config["model_name"]
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        model.eval()
        _MODEL_CACHE[variant] = model
    return _MODEL_CACHE[variant]


def tokenize_midi(midi_path: Path) -> list:
    """Tokenize a MIDI file using Aria's AbsTokenizer.

    Args:
        midi_path: Path to a .mid file.

    Returns:
        List of tokens in AbsTokenizer format.

    Raises:
        FileNotFoundError: If midi_path does not exist.
    """
    if not midi_path.exists():
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")

    tokenizer = _get_tokenizer()
    midi_dict = tokenizer.tokenize_midi_file(midi_path)
    tokens = tokenizer.tokenize(midi_dict)
    return tokens


def extract_embedding(
    midi_path: Path,
    variant: str = "embedding",
) -> torch.Tensor:
    """Extract a frozen Aria embedding from a single MIDI file.

    Args:
        midi_path: Path to a .mid file.
        variant: "embedding" (512-dim) or "base" (1536-dim, last-token).

    Returns:
        1-D tensor of shape (output_dim,).

    Raises:
        FileNotFoundError: If midi_path does not exist.
        ValueError: If variant is unknown.
    """
    if variant not in VARIANT_CONFIG:
        raise ValueError(
            f"Unknown variant '{variant}'. "
            f"Must be one of: {list(VARIANT_CONFIG.keys())}"
        )

    tokens = tokenize_midi(midi_path)
    model = _get_model(variant)
    tokenizer = _get_tokenizer()

    config = VARIANT_CONFIG[variant]
    max_seq_len = config["max_seq_len"]

    # Convert tokens to input IDs
    token_ids = tokenizer.encode(tokens)
    if len(token_ids) > max_seq_len:
        token_ids = token_ids[:max_seq_len]

    input_ids = torch.tensor([token_ids], dtype=torch.long)

    with torch.no_grad():
        outputs = model(input_ids)
        hidden_states = outputs.last_hidden_state  # [1, seq_len, hidden_dim]
        # Both variants use last-token pooling
        embedding = hidden_states[0, -1, :]

    return embedding.float().cpu()


def extract_all_embeddings(
    midi_dir: Path,
    variant: str = "embedding",
) -> dict[str, torch.Tensor]:
    """Extract embeddings for all .mid files in a directory.

    Args:
        midi_dir: Directory containing .mid files.
        variant: "embedding" (512-dim) or "base" (1536-dim).

    Returns:
        Dict mapping segment_id (filename without .mid) to embedding tensor.
    """
    midi_files = sorted(midi_dir.glob("*.mid"))
    if not midi_files:
        raise FileNotFoundError(f"No .mid files found in {midi_dir}")

    embeddings: dict[str, torch.Tensor] = {}
    total = len(midi_files)

    for idx, midi_path in enumerate(midi_files):
        segment_id = midi_path.stem  # strip .mid extension
        if (idx + 1) % 50 == 0 or idx == 0:
            logger.info(
                "Extracting %s: %d/%d (%s)",
                variant, idx + 1, total, segment_id,
            )
        try:
            emb = extract_embedding(midi_path, variant=variant)
            embeddings[segment_id] = emb
        except Exception as exc:
            logger.error("Failed to extract %s: %s", segment_id, exc)
            raise

    logger.info(
        "Extraction complete: %d/%d segments, variant=%s, dim=%d",
        len(embeddings), total, variant, VARIANT_CONFIG[variant]["output_dim"],
    )
    return embeddings


def main() -> None:
    """CLI entry point for embedding extraction."""
    parser = argparse.ArgumentParser(
        description="Extract Aria embeddings from MIDI"
    )
    parser.add_argument(
        "--variant", choices=["embedding", "base"],
        required=True, help="Aria model variant",
    )
    parser.add_argument(
        "--midi-dir", type=Path,
        required=True, help="Directory containing .mid files",
    )
    parser.add_argument(
        "--output", type=Path,
        required=True, help="Output .pt file path",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s"
    )

    embeddings = extract_all_embeddings(args.midi_dir, variant=args.variant)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, args.output)
    logger.info("Saved %d embeddings to %s", len(embeddings), args.output)


if __name__ == "__main__":
    main()
```

**Important:** The tokenization API (`tokenize_midi_file`, `tokenize`, `encode`) depends on the actual Aria library interface. After installing aria in Task 1, verify the exact method names by running:
```bash
uv run python3 -c "from aria.tokenizer import AbsTokenizer; help(AbsTokenizer)" 2>&1 | head -50
```

Adjust `tokenize_midi()` and `extract_embedding()` if the API differs. The core pattern is: MIDI file -> tokenizer -> token IDs -> model forward pass -> hidden states at last position.

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
cd /Users/jdhiman/Documents/crescendai/model
uv run pytest tests/model_improvement/test_aria_embeddings.py -v --no-header 2>&1 | tail -20
```

Expected: All 6 tests pass. Note: First run will be slow (~30s) as Aria models load.

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai/model
git add src/model_improvement/aria_embeddings.py tests/model_improvement/test_aria_embeddings.py
git commit -m "feat: add Aria embedding extraction from MIDI files

Supports two variants (embedding 512-dim, base 1536-dim).
Tokenizes MIDI via AbsTokenizer, extracts frozen representations."
```

---

## Task 3: Run Embedding Extraction on PercePiano

**Files:**
- Output: `model/data/embeddings/percepiano/aria_embedding.pt`
- Output: `model/data/embeddings/percepiano/aria_base.pt`

- [ ] **Step 1: Extract embedding variant (Set A)**

Run:
```bash
cd /Users/jdhiman/Documents/crescendai/model
uv run python3 -m model_improvement.aria_embeddings \
  --variant embedding \
  --midi-dir data/midi/percepiano \
  --output data/embeddings/percepiano/aria_embedding.pt
```

Expected: 1,202 embeddings extracted, each 512-dim. Takes ~10 min on CPU.

- [ ] **Step 2: Verify Set A**

Run:
```bash
cd /Users/jdhiman/Documents/crescendai/model
uv run python3 -c "
import torch
emb = torch.load('data/embeddings/percepiano/aria_embedding.pt', map_location='cpu', weights_only=True)
print(f'Keys: {len(emb)}')
k = list(emb.keys())[0]
print(f'First key: {k}, shape: {emb[k].shape}')
assert len(emb) == 1202, f'Expected 1202, got {len(emb)}'
assert emb[k].shape == (512,), f'Expected (512,), got {emb[k].shape}'
print('Set A OK')
"
```

- [ ] **Step 3: Extract base variant (Set B)**

Run:
```bash
cd /Users/jdhiman/Documents/crescendai/model
uv run python3 -m model_improvement.aria_embeddings \
  --variant base \
  --midi-dir data/midi/percepiano \
  --output data/embeddings/percepiano/aria_base.pt
```

Expected: 1,202 embeddings extracted, each 1536-dim. Takes ~20 min on CPU.

- [ ] **Step 4: Verify Set B**

Run:
```bash
cd /Users/jdhiman/Documents/crescendai/model
uv run python3 -c "
import torch
emb = torch.load('data/embeddings/percepiano/aria_base.pt', map_location='cpu', weights_only=True)
print(f'Keys: {len(emb)}')
k = list(emb.keys())[0]
print(f'First key: {k}, shape: {emb[k].shape}')
assert len(emb) == 1202, f'Expected 1202, got {len(emb)}'
assert emb[k].shape == (1536,), f'Expected (1536,), got {emb[k].shape}'
print('Set B OK')
"
```

---

## Task 4: Linear Probe Script -- `aria_linear_probe.py`

**Files:**
- Create: `model/src/model_improvement/aria_linear_probe.py`
- Test: `model/tests/model_improvement/test_aria_linear_probe.py`

- [ ] **Step 1: Write failing tests for the linear probe**

Create `model/tests/model_improvement/test_aria_linear_probe.py`:

```python
"""Tests for Aria linear probe evaluation."""

import numpy as np
import torch
import pytest


class TestPairwiseFromRegression:
    """Test converting pointwise regression to pairwise accuracy."""

    def test_perfect_predictions_give_100_pct(self):
        from model_improvement.aria_linear_probe import (
            compute_pairwise_from_regression,
        )

        # 4 samples with clear ordering
        predictions = torch.tensor([
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],  # best
            [0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
            [0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
            [0.3, 0.2, 0.1, 0.0, 0.0, 0.0],  # worst
        ])
        labels = {
            "a": np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4]),
            "b": np.array([0.7, 0.6, 0.5, 0.4, 0.3, 0.2]),
            "c": np.array([0.5, 0.4, 0.3, 0.2, 0.1, 0.0]),
            "d": np.array([0.3, 0.2, 0.1, 0.0, 0.0, 0.0]),
        }
        keys = ["a", "b", "c", "d"]
        result = compute_pairwise_from_regression(predictions, keys, labels)
        assert result["overall"] > 0.95

    def test_random_predictions_near_50_pct(self):
        from model_improvement.aria_linear_probe import (
            compute_pairwise_from_regression,
        )

        torch.manual_seed(123)
        n = 50
        predictions = torch.rand(n, 6)
        labels = {
            f"s{i}": np.random.rand(6).astype(np.float32)
            for i in range(n)
        }
        keys = [f"s{i}" for i in range(n)]
        result = compute_pairwise_from_regression(predictions, keys, labels)
        assert 0.35 < result["overall"] < 0.65

    def test_per_dimension_breakdown_returned(self):
        from model_improvement.aria_linear_probe import (
            compute_pairwise_from_regression,
        )

        predictions = torch.tensor([
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            [0.3, 0.2, 0.1, 0.0, 0.0, 0.0],
        ])
        labels = {
            "a": np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4]),
            "b": np.array([0.3, 0.2, 0.1, 0.0, 0.0, 0.0]),
        }
        keys = ["a", "b"]
        result = compute_pairwise_from_regression(predictions, keys, labels)
        assert "per_dimension" in result
        assert len(result["per_dimension"]) == 6


class TestTrainLinearProbe:
    def test_probe_trains_and_returns_predictions(self):
        from model_improvement.aria_linear_probe import train_linear_probe

        torch.manual_seed(42)
        n_train, n_val = 20, 5
        dim = 32
        train_emb = torch.randn(n_train, dim)
        train_labels = torch.rand(n_train, 6)
        val_emb = torch.randn(n_val, dim)
        val_labels = torch.rand(n_val, 6)

        val_preds, train_preds = train_linear_probe(
            train_emb, train_labels, val_emb, val_labels,
            lr=1e-2, weight_decay=0.0, max_epochs=50, patience=10,
        )
        assert val_preds.shape == (n_val, 6)
        assert train_preds.shape == (n_train, 6)

    def test_probe_on_perfect_data_fits_well(self):
        from model_improvement.aria_linear_probe import train_linear_probe

        torch.manual_seed(42)
        dim = 16
        W = torch.randn(dim, 6)
        train_emb = torch.randn(100, dim)
        train_labels = train_emb @ W  # perfectly linear
        val_emb = torch.randn(20, dim)
        val_labels = val_emb @ W

        val_preds, _ = train_linear_probe(
            train_emb, train_labels, val_emb, val_labels,
            lr=1e-2, weight_decay=0.0, max_epochs=200, patience=20,
        )
        from model_improvement.metrics import MetricsSuite
        suite = MetricsSuite()
        r2 = suite.regression_r2(val_preds, val_labels)
        assert r2 > 0.90


class TestErrorCorrelation:
    def test_identical_errors_give_high_phi(self):
        from model_improvement.aria_linear_probe import (
            compute_error_correlation,
        )

        # Both models get same pairs right/wrong
        correct_a = torch.tensor([True, True, False, False, True])
        correct_b = torch.tensor([True, True, False, False, True])
        phi = compute_error_correlation(correct_a, correct_b)
        assert phi > 0.95

    def test_independent_errors_give_low_phi(self):
        from model_improvement.aria_linear_probe import (
            compute_error_correlation,
        )

        torch.manual_seed(42)
        n = 1000
        correct_a = torch.randint(0, 2, (n,), dtype=torch.bool)
        correct_b = torch.randint(0, 2, (n,), dtype=torch.bool)
        phi = compute_error_correlation(correct_a, correct_b)
        assert -0.15 < phi < 0.15
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cd /Users/jdhiman/Documents/crescendai/model
uv run pytest tests/model_improvement/test_aria_linear_probe.py -v --no-header 2>&1 | head -20
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `aria_linear_probe.py` implementation**

Create `model/src/model_improvement/aria_linear_probe.py`:

```python
"""Linear probe evaluation for Aria frozen embeddings.

Trains a simple Linear(dim, 6) on frozen embeddings across 4-fold CV.
Computes pairwise accuracy, R2, per-dimension breakdown, and error
correlation with MuQ.

Usage:
    python -m model_improvement.aria_linear_probe
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr

from model_improvement.evaluation import aggregate_folds
from model_improvement.metrics import MetricsSuite, format_comparison_table
from model_improvement.taxonomy import DIMENSIONS, NUM_DIMS, load_composite_labels
from paths import Embeddings, Labels

logger = logging.getLogger(__name__)


def compute_pairwise_from_regression(
    predictions: torch.Tensor,
    keys: list[str],
    labels: dict[str, np.ndarray],
    ambiguous_threshold: float = 0.05,
) -> dict:
    """Compute pairwise accuracy from pointwise regression predictions.

    Uses scalar-logit semantics consistently (as per spec): for each pair
    (i, j), average pred_diff across dims to get a scalar logit, average
    label_diff across dims to get a scalar target. A pair is correct if
    sign(scalar_logit) == sign(scalar_target). Ambiguous pairs where
    mean |label_diff| < threshold are excluded.

    Also computes per-dimension pairwise accuracy (per-dim, not averaged).

    Args:
        predictions: Tensor of shape (n_samples, 6).
        keys: Ordered list of segment IDs matching prediction rows.
        labels: Dict mapping segment_id to numpy array of 6 scores.
        ambiguous_threshold: Min mean |label_diff| to include a pair.

    Returns:
        Dict with "overall", "per_dimension", "n_comparisons",
        "correct_mask", "non_ambiguous_mask".
    """
    n = len(keys)

    pred_diffs = []   # [n_pairs, 6]
    label_diffs = []  # [n_pairs, 6]

    for i in range(n):
        for j in range(i + 1, n):
            pred_diff = predictions[i] - predictions[j]
            lab_a = torch.tensor(
                labels[keys[i]][:NUM_DIMS], dtype=torch.float32,
            )
            lab_b = torch.tensor(
                labels[keys[j]][:NUM_DIMS], dtype=torch.float32,
            )
            pred_diffs.append(pred_diff)
            label_diffs.append(lab_a - lab_b)

    pred_diffs_t = torch.stack(pred_diffs)   # [n_pairs, 6]
    label_diffs_t = torch.stack(label_diffs)  # [n_pairs, 6]

    # Scalar logit: mean across dimensions
    scalar_logit = pred_diffs_t.mean(dim=1)    # [n_pairs]
    scalar_target = label_diffs_t.mean(dim=1)  # [n_pairs]

    # Non-ambiguous mask: mean |label_diff| >= threshold
    non_ambiguous = label_diffs_t.abs().mean(dim=1) >= ambiguous_threshold

    # Overall accuracy (scalar-logit, matches spec)
    correct = (scalar_logit > 0) == (scalar_target > 0)
    n_non_ambig = non_ambiguous.sum().item()
    if n_non_ambig > 0:
        overall = float((correct & non_ambiguous).sum().item() / n_non_ambig)
    else:
        overall = 0.5

    # Per-dimension accuracy
    per_dim: dict[int, float] = {}
    for d in range(NUM_DIMS):
        dim_non_ambig = label_diffs_t[:, d].abs() >= ambiguous_threshold
        if dim_non_ambig.sum() > 0:
            dim_correct = (pred_diffs_t[:, d] > 0) == (label_diffs_t[:, d] > 0)
            per_dim[d] = float(
                (dim_correct & dim_non_ambig).sum().item()
                / dim_non_ambig.sum().item()
            )
        else:
            per_dim[d] = 0.5

    return {
        "overall": overall,
        "per_dimension": per_dim,
        "n_comparisons": int(n_non_ambig),
        "correct_mask": correct,
        "non_ambiguous_mask": non_ambiguous,
    }


def train_linear_probe(
    train_emb: torch.Tensor,
    train_labels: torch.Tensor,
    val_emb: torch.Tensor,
    val_labels: torch.Tensor,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 100,
    patience: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Train a linear probe with early stopping on val MSE.

    Returns:
        Tuple of (val_predictions, train_predictions).
    """
    dim_in = train_emb.shape[1]
    probe = nn.Linear(dim_in, NUM_DIMS)
    nn.init.xavier_uniform_(probe.weight)
    nn.init.zeros_(probe.bias)

    optimizer = torch.optim.Adam(
        probe.parameters(), lr=lr, weight_decay=weight_decay,
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        probe.train()
        optimizer.zero_grad()
        pred = probe(train_emb)
        loss = criterion(pred, train_labels)
        loss.backward()
        optimizer.step()

        probe.eval()
        with torch.no_grad():
            val_pred = probe(val_emb)
            val_loss = criterion(val_pred, val_labels).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                k: v.clone() for k, v in probe.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is not None:
        probe.load_state_dict(best_state)

    probe.eval()
    with torch.no_grad():
        val_preds = probe(val_emb)
        train_preds = probe(train_emb)

    return val_preds, train_preds


def compute_error_correlation(
    correct_a: torch.Tensor,
    correct_b: torch.Tensor,
) -> float:
    """Compute phi coefficient between two binary correct/incorrect vectors.

    Returns:
        Phi coefficient (float). 1.0 = identical errors, 0.0 = independent.
    """
    a = correct_a.float().numpy()
    b = correct_b.float().numpy()
    r, _ = pearsonr(a, b)
    return float(r)


def load_embeddings_as_matrix(
    emb_dict: dict[str, torch.Tensor],
    keys: list[str],
) -> tuple[torch.Tensor, list[str]]:
    """Convert embedding dict to matrix, filtering to available keys.

    Returns:
        Tuple of (embedding_matrix [n, dim], valid_keys [n]).
    """
    valid_keys = [k for k in keys if k in emb_dict]
    if not valid_keys:
        raise ValueError("No keys found in embedding dict")
    matrix = torch.stack([emb_dict[k] for k in valid_keys])
    return matrix, valid_keys


def run_fold_evaluation(
    emb_dict: dict[str, torch.Tensor],
    labels: dict[str, np.ndarray],
    fold: dict[str, list[str]],
    fold_idx: int,
    model_name: str,
    restrict_val_keys: list[str] | None = None,
) -> dict:
    """Run linear probe evaluation for a single fold.

    Args:
        restrict_val_keys: If provided, only use these val keys (for
            ensuring consistent keys across models in error correlation).
    """
    train_keys = fold["train"]
    val_keys = restrict_val_keys if restrict_val_keys else fold["val"]

    train_emb, train_valid = load_embeddings_as_matrix(emb_dict, train_keys)
    val_emb, val_valid = load_embeddings_as_matrix(emb_dict, val_keys)

    train_labels_t = torch.tensor(
        np.array([labels[k] for k in train_valid]), dtype=torch.float32,
    )
    val_labels_t = torch.tensor(
        np.array([labels[k] for k in val_valid]), dtype=torch.float32,
    )

    logger.info(
        "Fold %d [%s]: train=%d, val=%d",
        fold_idx, model_name, len(train_valid), len(val_valid),
    )

    val_preds, _ = train_linear_probe(
        train_emb, train_labels_t, val_emb, val_labels_t,
    )

    pw = compute_pairwise_from_regression(val_preds, val_valid, labels)

    suite = MetricsSuite()
    r2 = suite.regression_r2(val_preds, val_labels_t)

    result = {
        "pairwise": pw["overall"],
        "r2": r2,
        "pairwise_detail": pw,
        "n_comparisons": pw["n_comparisons"],
    }

    for d_idx, d_name in enumerate(DIMENSIONS):
        if d_idx in pw["per_dimension"]:
            result[f"pw_{d_name}"] = pw["per_dimension"][d_idx]

    return result


def run_full_evaluation(
    emb_dict: dict[str, torch.Tensor],
    labels: dict[str, np.ndarray],
    folds: list[dict],
    model_name: str,
    restrict_val_keys_per_fold: list[list[str]] | None = None,
) -> dict:
    """Run linear probe across all folds and aggregate results.

    Args:
        restrict_val_keys_per_fold: If provided, a list of key lists
            (one per fold) to ensure consistent val keys across models.
    """
    torch.manual_seed(42)
    fold_results = []
    all_correct_masks = []
    all_non_ambiguous_masks = []

    for fold_idx, fold in enumerate(folds):
        restrict_keys = (
            restrict_val_keys_per_fold[fold_idx]
            if restrict_val_keys_per_fold
            else None
        )
        result = run_fold_evaluation(
            emb_dict, labels, fold, fold_idx, model_name,
            restrict_val_keys=restrict_keys,
        )
        fold_results.append(result)
        all_correct_masks.append(
            result["pairwise_detail"]["correct_mask"]
        )
        all_non_ambiguous_masks.append(
            result["pairwise_detail"]["non_ambiguous_mask"]
        )

    aggregated = aggregate_folds(fold_results)
    aggregated["fold_results"] = fold_results
    aggregated["all_correct_masks"] = all_correct_masks
    aggregated["all_non_ambiguous_masks"] = all_non_ambiguous_masks
    return aggregated


def mean_pool_muq(
    muq_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Mean-pool MuQ frame-level embeddings to fixed-dim vectors.

    Args:
        muq_dict: {segment_id: tensor[n_frames, 1024]}

    Returns:
        {segment_id: tensor[1024]}
    """
    return {k: v.mean(dim=0) for k, v in muq_dict.items()}


def main() -> None:
    """Run the full Aria validation experiment."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s",
    )

    # Load data
    labels = load_composite_labels(
        Labels.composite / "composite_labels.json",
    )
    with open(Labels.percepiano / "folds.json") as f:
        folds = json.load(f)

    print(f"Loaded {len(labels)} labels, {len(folds)} folds")
    print(
        f"Fold sizes: "
        f"{[(len(f['train']), len(f['val'])) for f in folds]}"
    )
    print()

    results_all = {}

    # Load all embedding dicts to compute shared keys
    emb_dicts: dict[str, dict[str, torch.Tensor]] = {}

    aria_emb_path = Embeddings.percepiano / "aria_embedding.pt"
    if aria_emb_path.exists():
        emb_dicts["Aria-Embedding"] = torch.load(
            aria_emb_path, map_location="cpu", weights_only=True,
        )

    aria_base_path = Embeddings.percepiano / "aria_base.pt"
    if aria_base_path.exists():
        emb_dicts["Aria-Base"] = torch.load(
            aria_base_path, map_location="cpu", weights_only=True,
        )

    muq_path = Embeddings.percepiano / "muq_embeddings.pt"
    if muq_path.exists():
        muq_raw = torch.load(
            muq_path, map_location="cpu", weights_only=False,
        )
        emb_dicts["MuQ"] = mean_pool_muq(muq_raw)

    # Compute shared val keys per fold (intersection of all models)
    # This ensures error correlation compares the same pairs
    shared_val_keys_per_fold: list[list[str]] = []
    for fold in folds:
        shared = set(fold["val"])
        for emb_dict in emb_dicts.values():
            shared &= set(emb_dict.keys())
        shared &= set(labels.keys())
        shared_val_keys_per_fold.append(sorted(shared))

    print(
        f"Shared val keys per fold: "
        f"{[len(k) for k in shared_val_keys_per_fold]}"
    )
    print()

    # --- Set A: Aria embedding variant ---
    if "Aria-Embedding" in emb_dicts:
        print("=== Set A: Aria Embedding (512-dim) ===")
        aria_emb = emb_dicts["Aria-Embedding"]
        results_a = run_full_evaluation(
            aria_emb, labels, folds, "Aria-Embedding",
            restrict_val_keys_per_fold=shared_val_keys_per_fold,
        )
        results_all["Aria-Embedding"] = results_a
        print(
            f"  Pairwise: {results_a['pairwise_mean']:.4f} "
            f"+/- {results_a['pairwise_std']:.4f}"
        )
        print(
            f"  R2:       {results_a['r2_mean']:.4f} "
            f"+/- {results_a['r2_std']:.4f}"
        )
        for d in DIMENSIONS:
            key = f"pw_{d}_mean"
            if key in results_a:
                print(f"  {d:>15s}: {results_a[key]:.4f}")
        print()
    else:
        print("Skipping Set A: aria_embedding.pt not found")

    # --- Set B: Aria base variant ---
    if "Aria-Base" in emb_dicts:
        print("=== Set B: Aria Base (1536-dim, last-token) ===")
        aria_base = emb_dicts["Aria-Base"]
        results_b = run_full_evaluation(
            aria_base, labels, folds, "Aria-Base",
            restrict_val_keys_per_fold=shared_val_keys_per_fold,
        )
        results_all["Aria-Base"] = results_b
        print(
            f"  Pairwise: {results_b['pairwise_mean']:.4f} "
            f"+/- {results_b['pairwise_std']:.4f}"
        )
        print(
            f"  R2:       {results_b['r2_mean']:.4f} "
            f"+/- {results_b['r2_std']:.4f}"
        )
        for d in DIMENSIONS:
            key = f"pw_{d}_mean"
            if key in results_b:
                print(f"  {d:>15s}: {results_b[key]:.4f}")
        print()
    else:
        print("Skipping Set B: aria_base.pt not found")

    # --- MuQ baseline ---
    if "MuQ" in emb_dicts:
        print("=== MuQ Baseline (1024-dim, mean-pooled) ===")
        muq_pooled = emb_dicts["MuQ"]
        results_muq = run_full_evaluation(
            muq_pooled, labels, folds, "MuQ",
            restrict_val_keys_per_fold=shared_val_keys_per_fold,
        )
        results_all["MuQ"] = results_muq
        print(
            f"  Pairwise: {results_muq['pairwise_mean']:.4f} "
            f"+/- {results_muq['pairwise_std']:.4f}"
        )
        print(
            f"  R2:       {results_muq['r2_mean']:.4f} "
            f"+/- {results_muq['r2_std']:.4f}"
        )
        for d in DIMENSIONS:
            key = f"pw_{d}_mean"
            if key in results_muq:
                print(f"  {d:>15s}: {results_muq[key]:.4f}")
        print()
    else:
        print("Skipping MuQ: muq_embeddings.pt not found")

    # --- Comparison Table ---
    if len(results_all) > 1:
        print("=== Comparison Table ===")
        table_data = {}
        for name, res in results_all.items():
            table_data[name] = {
                "pairwise": res.get("pairwise_mean", 0.0),
                "r2": res.get("r2_mean", 0.0),
            }
            for d in DIMENSIONS:
                key = f"pw_{d}_mean"
                if key in res:
                    table_data[name][f"pw_{d}"] = res[key]
        print(format_comparison_table(table_data))
        print()

    # --- Error Correlation ---
    if len(results_all) >= 2 and "MuQ" in results_all:
        print("=== Error Correlation (MuQ vs best Aria) ===")
        aria_names = [n for n in results_all if n != "MuQ"]
        best_aria_name = max(
            aria_names,
            key=lambda n: results_all[n].get("pairwise_mean", 0.0),
        )
        best_aria = results_all[best_aria_name]
        muq_res = results_all["MuQ"]

        all_phi = []
        for fold_idx in range(len(folds)):
            aria_correct = best_aria["all_correct_masks"][fold_idx]
            muq_correct = muq_res["all_correct_masks"][fold_idx]
            aria_na = best_aria["all_non_ambiguous_masks"][fold_idx]
            muq_na = muq_res["all_non_ambiguous_masks"][fold_idx]

            shared_mask = aria_na & muq_na
            if shared_mask.sum() > 0:
                phi = compute_error_correlation(
                    aria_correct[shared_mask],
                    muq_correct[shared_mask],
                )
                all_phi.append(phi)

        if all_phi:
            mean_phi = float(np.mean(all_phi))
            std_phi = float(np.std(all_phi))
            print(f"  Best Aria: {best_aria_name}")
            print(f"  Phi coefficient: {mean_phi:.4f} +/- {std_phi:.4f}")
            if mean_phi < 0.50:
                print(
                    "  -> Models make DIFFERENT mistakes "
                    "-> fusion promising"
                )
            elif mean_phi < 0.70:
                print(
                    "  -> MODERATE overlap "
                    "-> fusion may help"
                )
            else:
                print(
                    "  -> Models are REDUNDANT "
                    "-> fusion unlikely to help"
                )
        print()

    # --- Decision ---
    print("=== Decision ===")
    for name, res in results_all.items():
        pw = res.get("pairwise_mean", 0.0)
        if pw > 0.60:
            print(
                f"  {name}: {pw:.4f} "
                f"-- QUALITY SIGNAL CONFIRMED (>60%)"
            )
        elif pw > 0.55:
            print(
                f"  {name}: {pw:.4f} "
                f"-- MARGINAL (55-60%), try LoRA fine-tuning"
            )
        else:
            print(
                f"  {name}: {pw:.4f} "
                f"-- NO QUALITY SIGNAL (~50%)"
            )

    # Save results
    from paths import Results
    results_path = Results.root / "aria_validation.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    save_data = {}
    for name, res in results_all.items():
        save_data[name] = {
            k: v for k, v in res.items()
            if k not in (
                "fold_results",
                "all_correct_masks",
                "all_non_ambiguous_masks",
            )
        }
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
cd /Users/jdhiman/Documents/crescendai/model
uv run pytest tests/model_improvement/test_aria_linear_probe.py -v --no-header 2>&1 | tail -20
```

Expected: All 7 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai/model
git add src/model_improvement/aria_linear_probe.py tests/model_improvement/test_aria_linear_probe.py
git commit -m "feat: add linear probe evaluation for Aria frozen embeddings

4-fold CV, pairwise from regression, R2, per-dimension breakdown,
error correlation with MuQ, comparison table, decision output."
```

---

## Task 5: Run Linear Probe Evaluation

**Files:**
- Output: `model/data/results/aria_validation.json`

- [ ] **Step 1: Run the full evaluation**

Run:
```bash
cd /Users/jdhiman/Documents/crescendai/model
uv run python3 -m model_improvement.aria_linear_probe
```

Expected output: pairwise accuracy, R2, per-dimension breakdown for each of 3 models, comparison table, error correlation, and decision recommendation. Save the full console output.

- [ ] **Step 2: Verify results file was saved**

Run:
```bash
cd /Users/jdhiman/Documents/crescendai/model
uv run python3 -m json.tool data/results/aria_validation.json | head -30
```

- [ ] **Step 3: Commit results**

```bash
cd /Users/jdhiman/Documents/crescendai/model
git add data/results/aria_validation.json
git commit -m "results: Aria validation experiment -- linear probe on frozen embeddings"
```

---

## Task 6: AMT Comparison Script -- `aria_amt_compare.py`

**Files:**
- Create: `model/src/model_improvement/aria_amt_compare.py`

This task is independent of Tasks 2-5 and can run in parallel.

- [ ] **Step 1: Write `aria_amt_compare.py`**

Create `model/src/model_improvement/aria_amt_compare.py`:

```python
"""Compare ByteDance AMT vs Aria-AMT on YouTube audio recordings.

Runs both AMT systems on 51 YouTube wav files, compares MIDI statistics,
and computes Aria embedding cosine similarity between the two outputs.

Usage:
    python -m model_improvement.aria_amt_compare
"""

from __future__ import annotations

import logging
from pathlib import Path

import mido
import numpy as np
import torch

from paths import Evals, Midi

logger = logging.getLogger(__name__)

YOUTUBE_AMT_DIR = Evals.youtube_amt
BYTEDANCE_OUT = Midi.amt / "youtube_bytedance"
ARIA_AMT_OUT = Midi.amt / "youtube_aria"


def compute_midi_stats(midi_path: Path) -> dict:
    """Compute descriptive statistics from a MIDI file.

    Returns dict with: note_count, mean_velocity, velocity_std,
    onset_density (notes/sec), has_pedal.
    """
    mid = mido.MidiFile(str(midi_path))
    velocities = []
    has_pedal = False

    for track in mid.tracks:
        for msg in track:
            if msg.type == "note_on" and msg.velocity > 0:
                velocities.append(msg.velocity)
            elif msg.type == "control_change" and msg.control == 64:
                has_pedal = True

    duration_sec = mid.length if mid.length > 0 else 1.0

    return {
        "note_count": len(velocities),
        "mean_velocity": (
            float(np.mean(velocities)) if velocities else 0.0
        ),
        "velocity_std": (
            float(np.std(velocities)) if velocities else 0.0
        ),
        "onset_density": len(velocities) / duration_sec,
        "has_pedal": has_pedal,
    }


def run_bytedance_amt(wav_path: Path, output_path: Path) -> None:
    """Run ByteDance piano transcription on a wav file."""
    from piano_transcription_inference import PianoTranscription

    transcriptor = PianoTranscription(device="cpu")
    transcriptor.transcribe(str(wav_path), str(output_path))


def run_aria_amt(wav_path: Path, output_path: Path) -> None:
    """Run Aria-AMT on a wav file.

    Note: The API depends on the aria-amt package. Verify after
    installing loubb/aria-amt weights. Check the model card at:
    https://huggingface.co/loubb/aria-amt
    """
    raise NotImplementedError(
        "Aria-AMT API needs verification after installing "
        "loubb/aria-amt. Check the HF model card for usage."
    )


def compare_amt_systems() -> None:
    """Run both AMT systems and compare results."""
    wav_files = sorted(YOUTUBE_AMT_DIR.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(
            f"No wav files in {YOUTUBE_AMT_DIR}"
        )

    print(f"Found {len(wav_files)} YouTube audio files")

    BYTEDANCE_OUT.mkdir(parents=True, exist_ok=True)
    ARIA_AMT_OUT.mkdir(parents=True, exist_ok=True)

    # Run transcriptions
    for wav_path in wav_files:
        video_id = wav_path.stem
        bd_midi = BYTEDANCE_OUT / f"{video_id}.mid"
        aria_midi = ARIA_AMT_OUT / f"{video_id}.mid"

        if not bd_midi.exists():
            logger.info("ByteDance AMT: %s", video_id)
            try:
                run_bytedance_amt(wav_path, bd_midi)
            except Exception as exc:
                logger.error(
                    "ByteDance AMT failed for %s: %s", video_id, exc,
                )

        if not aria_midi.exists():
            logger.info("Aria AMT: %s", video_id)
            try:
                run_aria_amt(wav_path, aria_midi)
            except Exception as exc:
                logger.error(
                    "Aria AMT failed for %s: %s", video_id, exc,
                )

    # Compare MIDI statistics
    print("\n=== MIDI Statistics Comparison ===")
    header = (
        f"{'Video ID':<15} {'System':<12} {'Notes':>6} "
        f"{'MeanVel':>8} {'VelStd':>7} {'Density':>8} {'Pedal':>6}"
    )
    print(header)
    print("-" * len(header))

    cosine_sims = []
    video_ids_compared = []

    for wav_path in wav_files:
        video_id = wav_path.stem
        bd_midi = BYTEDANCE_OUT / f"{video_id}.mid"
        aria_midi = ARIA_AMT_OUT / f"{video_id}.mid"

        if bd_midi.exists():
            stats = compute_midi_stats(bd_midi)
            print(
                f"{video_id:<15} {'ByteDance':<12} "
                f"{stats['note_count']:>6d} "
                f"{stats['mean_velocity']:>8.1f} "
                f"{stats['velocity_std']:>7.1f} "
                f"{stats['onset_density']:>8.2f} "
                f"{str(stats['has_pedal']):>6}"
            )

        if aria_midi.exists():
            stats = compute_midi_stats(aria_midi)
            print(
                f"{video_id:<15} {'Aria-AMT':<12} "
                f"{stats['note_count']:>6d} "
                f"{stats['mean_velocity']:>8.1f} "
                f"{stats['velocity_std']:>7.1f} "
                f"{stats['onset_density']:>8.2f} "
                f"{str(stats['has_pedal']):>6}"
            )

        # Cosine similarity of Aria embeddings from both AMTs
        if bd_midi.exists() and aria_midi.exists():
            try:
                from model_improvement.aria_embeddings import (
                    extract_embedding,
                )
                emb_bd = extract_embedding(
                    bd_midi, variant="embedding",
                )
                emb_aria = extract_embedding(
                    aria_midi, variant="embedding",
                )
                cos_sim = torch.nn.functional.cosine_similarity(
                    emb_bd.unsqueeze(0), emb_aria.unsqueeze(0),
                ).item()
                cosine_sims.append(cos_sim)
                video_ids_compared.append(video_id)
                print(
                    f"{video_id:<15} {'CosSim':<12} "
                    f"{cos_sim:>8.4f}"
                )
            except Exception as exc:
                logger.error(
                    "Embedding comparison failed for %s: %s",
                    video_id, exc,
                )

        print()

    # Summary
    if cosine_sims:
        sims = np.array(cosine_sims)
        print("=== Cosine Similarity Distribution ===")
        print(f"  Mean: {sims.mean():.4f}")
        print(f"  Std:  {sims.std():.4f}")
        print(f"  Min:  {sims.min():.4f}")
        print(f"  Max:  {sims.max():.4f}")

        # Flag outliers (>2 std below mean)
        threshold = sims.mean() - 2 * sims.std()
        outlier_indices = np.where(sims < threshold)[0]
        if len(outlier_indices) > 0:
            print(f"\n  Low outliers (>2 std below mean):")
            for idx in outlier_indices:
                print(
                    f"    {video_ids_compared[idx]}: "
                    f"{cosine_sims[idx]:.4f}"
                )


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s",
    )
    compare_amt_systems()


if __name__ == "__main__":
    main()
```

**Important note on Aria-AMT:** The `run_aria_amt()` function is a placeholder. After downloading `loubb/aria-amt` weights in Task 1, verify the actual inference API:
```bash
uv run python3 -c "
from transformers import AutoModel
model = AutoModel.from_pretrained('data/weights/aria-amt', trust_remote_code=True)
print(type(model))
"
```
Then update `run_aria_amt()` accordingly. The ByteDance AMT uses `piano_transcription_inference` which is already a dependency.

- [ ] **Step 2: Run ByteDance AMT on YouTube audio**

Run:
```bash
cd /Users/jdhiman/Documents/crescendai/model
uv run python3 -m model_improvement.aria_amt_compare
```

Note: Aria-AMT will fail with `NotImplementedError` until its API is verified. ByteDance AMT should run. The comparison will be partial (ByteDance stats only) until Aria-AMT is implemented.

- [ ] **Step 3: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai/model
git add src/model_improvement/aria_amt_compare.py
git commit -m "feat: add AMT comparison script (ByteDance vs Aria-AMT on YouTube)

ByteDance AMT functional. Aria-AMT placeholder pending API verification."
```

---

## Task 7: Final Verification and Results Summary

- [ ] **Step 1: Run full test suite**

Run:
```bash
cd /Users/jdhiman/Documents/crescendai/model
uv run pytest tests/model_improvement/test_aria_embeddings.py tests/model_improvement/test_aria_linear_probe.py -v
```

Expected: All tests pass.

- [ ] **Step 2: Verify all deliverables exist**

Run:
```bash
cd /Users/jdhiman/Documents/crescendai/model
echo "=== Scripts ==="
ls -la src/model_improvement/aria_*.py
echo "=== Tests ==="
ls -la tests/model_improvement/test_aria_*.py
echo "=== Embeddings ==="
ls -la data/embeddings/percepiano/aria_*.pt
echo "=== Results ==="
ls -la data/results/aria_validation.json
```

- [ ] **Step 3: Record final console output**

Save the complete output from Task 5 Step 1 as the experiment record. Key numbers:
- Aria-Embedding pairwise accuracy (mean +/- std)
- Aria-Base pairwise accuracy (mean +/- std)
- MuQ baseline pairwise accuracy (mean +/- std)
- Per-dimension pairwise for all three
- Error correlation phi
- Decision recommendation

- [ ] **Step 4: Final commit**

```bash
cd /Users/jdhiman/Documents/crescendai/model
git add -A data/results/
git commit -m "results: complete Aria validation experiment deliverables"
```
