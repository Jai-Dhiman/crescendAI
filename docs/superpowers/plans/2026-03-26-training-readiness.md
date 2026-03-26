# Training Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare the entire ML training pipeline so that when T5 labeling completes, training begins with one command.

**Architecture:** Extend existing PyTorch Lightning pipeline with: (1) three-way data splits replacing 4-fold CV, (2) T5 skill discrimination eval metric, (3) Trackio experiment tracking callback, (4) config-driven autoresearch runner, (5) HF Bucket storage integration, (6) doc updates.

**Tech Stack:** PyTorch Lightning, trackio, huggingface_hub, numpy, scipy, pytest

**Spec:** `docs/superpowers/specs/2026-03-26-training-readiness-design.md`

---

## File Map

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `model/src/model_improvement/splits.py` | Three-way train/val/test split generation with stratification |
| Create | `model/src/model_improvement/skill_discrimination.py` | T5 skill discrimination pairwise metric + bootstrap CIs |
| Create | `model/src/model_improvement/trackio_callback.py` | PyTorch Lightning callback for Trackio logging |
| Create | `model/src/model_improvement/autoresearch_runner.py` | Config-driven autoresearch sweep framework |
| Create | `model/src/model_improvement/data_integrity.py` | T5 data integrity checks |
| Create | `model/scripts/prepare_training.py` | One-command pipeline: integrity -> splits -> embeddings -> upload |
| Create | `model/tests/model_improvement/test_splits.py` | 5 critical split tests |
| Create | `model/tests/model_improvement/test_skill_discrimination.py` | 3 metric tests |
| Modify | `model/src/paths.py` | Add T5 embedding and split paths |
| Modify | `docs/model/01-data.md` | Replace Thunder Compute, add HF Buckets, update eval strategy |
| Modify | `model/CLAUDE.md` | Replace Thunder Compute in Stack, add Trackio |
| Modify | `docs/model/04-north-star.md` | Update eval tiers, T5 as primary metric |
| Modify | `docs/model/03-encoders.md` | Add Aria confound check |

---

### Task 1: Add T5 Paths to paths.py

**Files:**
- Modify: `model/src/paths.py:25-31`

- [ ] **Step 1: Add T5 paths to Embeddings and new Splits class**

```python
# In model/src/paths.py, add to Embeddings class (after competition line):
#   t5 = root / "t5"
# Add new Splits class after Evals class:
# class Splits:
#     root = DATA_ROOT / "splits"
```

Add these to `model/src/paths.py`:

In the `Embeddings` class, after `competition = root / "competition"`:

```python
    t5_muq = root / "t5_muq"
    t5_aria = root / "t5_aria"
```

After the `Evals` class (line ~83), add:

```python
class Splits:
    root = DATA_ROOT / "splits"
```

- [ ] **Step 2: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai
git add model/src/paths.py
git commit -m "feat: add T5 embedding and split paths"
```

---

### Task 2: Implement Split Generation (TDD)

**Files:**
- Create: `model/src/model_improvement/splits.py`
- Create: `model/tests/model_improvement/test_splits.py`

- [ ] **Step 1: Write the 5 failing split tests**

Create `model/tests/model_improvement/test_splits.py`:

```python
"""Tests for three-way train/val/test split generation."""

import pytest

from model_improvement.splits import (
    generate_t5_splits,
    generate_t1_splits,
    generate_t2_splits,
)


def _make_t5_recordings(pieces=3, buckets=5, per_bucket=6):
    """Helper: create synthetic T5 recording list."""
    recordings = []
    for p in range(pieces):
        for b in range(1, buckets + 1):
            for i in range(per_bucket):
                recordings.append({
                    "video_id": f"piece{p}_bucket{b}_rec{i}",
                    "piece": f"piece_{p}",
                    "skill_bucket": b,
                })
    return recordings


class TestT5Splits:
    def test_stratification_by_piece_and_bucket(self):
        """Every piece+bucket combo appears in every split."""
        recs = _make_t5_recordings(pieces=3, buckets=5, per_bucket=6)
        splits = generate_t5_splits(recs, train=0.8, val=0.1, test=0.1, seed=42)

        for split_name in ("train", "val", "test"):
            split_recs = splits[split_name]
            combos = {(r["piece"], r["skill_bucket"]) for r in split_recs}
            expected = {(f"piece_{p}", b) for p in range(3) for b in range(1, 6)}
            assert combos == expected, f"{split_name} missing combos: {expected - combos}"

    def test_no_recording_leak_across_splits(self):
        """No recording appears in more than one split."""
        recs = _make_t5_recordings()
        splits = generate_t5_splits(recs, train=0.8, val=0.1, test=0.1, seed=42)

        train_ids = {r["video_id"] for r in splits["train"]}
        val_ids = {r["video_id"] for r in splits["val"]}
        test_ids = {r["video_id"] for r in splits["test"]}

        assert train_ids.isdisjoint(val_ids), "train/val overlap"
        assert train_ids.isdisjoint(test_ids), "train/test overlap"
        assert val_ids.isdisjoint(test_ids), "val/test overlap"
        assert len(train_ids) + len(val_ids) + len(test_ids) == len(recs)

    def test_sparse_bucket_raises(self):
        """Piece+bucket with <3 recordings raises ValueError."""
        recs = _make_t5_recordings(pieces=1, buckets=5, per_bucket=6)
        # Remove all but 2 from bucket 3
        recs = [r for r in recs if not (r["skill_bucket"] == 3 and r["video_id"].endswith(("_rec2", "_rec3", "_rec4", "_rec5")))]
        with pytest.raises(ValueError, match="bucket.*fewer than 3"):
            generate_t5_splits(recs, train=0.8, val=0.1, test=0.1, seed=42)


class TestT1Splits:
    def test_stratification_by_piece(self):
        """T1 split is stratified by piece (each piece in both train and test)."""
        records = [
            {"key": f"piece{p}_rec{i}", "piece": f"piece_{p}"}
            for p in range(3) for i in range(10)
        ]
        splits = generate_t1_splits(records, train=0.8, test=0.2, seed=42)

        train_pieces = {r["piece"] for r in splits["train"]}
        test_pieces = {r["piece"] for r in splits["test"]}
        assert train_pieces == test_pieces == {f"piece_{p}" for p in range(3)}


class TestT2Splits:
    def test_holdout_by_round(self):
        """T2 holdout uses entire rounds -- no performer appears in both train and test."""
        records = []
        for comp in ("chopin", "cliburn"):
            for round_name in ("prelim", "semifinal", "final"):
                for performer in range(5):
                    records.append({
                        "recording_id": f"{comp}_{round_name}_p{performer}",
                        "competition": comp,
                        "round": round_name,
                        "performer_id": f"{comp}_p{performer}",
                    })
        splits = generate_t2_splits(records, train=0.85, test=0.15, seed=42)

        train_performers = {r["performer_id"] for r in splits["train"]}
        test_performers = {r["performer_id"] for r in splits["test"]}
        # Within a competition, test rounds should not share performers with train
        # (all performers appear in all rounds, so holdout by round means all performers
        # appear in both -- but entire rounds are held out, not individual recordings)
        train_rounds = {(r["competition"], r["round"]) for r in splits["train"]}
        test_rounds = {(r["competition"], r["round"]) for r in splits["test"]}
        assert train_rounds.isdisjoint(test_rounds), "same round in train and test"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/jdhiman/Documents/crescendai/model
uv run python -m pytest tests/model_improvement/test_splits.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'model_improvement.splits'`

- [ ] **Step 3: Implement splits.py**

Create `model/src/model_improvement/splits.py`:

```python
"""Three-way train/val/test split generation with stratification.

Replaces 4-fold CV. T5 stratifies by piece+bucket, T1 by piece,
T2 by competition+round.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any


def generate_t5_splits(
    recordings: list[dict[str, Any]],
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Split T5 recordings stratified by piece + skill_bucket.

    Each piece+bucket group is split proportionally so every group
    appears in every split.

    Args:
        recordings: List of dicts with keys: video_id, piece, skill_bucket.
        train: Fraction for training.
        val: Fraction for validation.
        test: Fraction for test.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys "train", "val", "test", each a list of recording dicts.

    Raises:
        ValueError: If any piece+bucket group has fewer than 3 recordings.
    """
    rng = random.Random(seed)

    # Group by (piece, bucket)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for rec in recordings:
        key = (rec["piece"], rec["skill_bucket"])
        groups[key].append(rec)

    # Validate minimum size
    for (piece, bucket), group_recs in groups.items():
        if len(group_recs) < 3:
            raise ValueError(
                f"piece={piece}, bucket={bucket} has {len(group_recs)} recordings, "
                f"fewer than 3 required for 3-way split"
            )

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

    for _key, group_recs in sorted(groups.items()):
        shuffled = list(group_recs)
        rng.shuffle(shuffled)
        n = len(shuffled)

        # Ensure at least 1 in each split
        n_test = max(1, round(n * test))
        n_val = max(1, round(n * val))
        n_train = n - n_val - n_test

        if n_train < 1:
            n_train = 1
            n_val = max(1, (n - 1) // 2)
            n_test = n - n_train - n_val

        splits["train"].extend(shuffled[:n_train])
        splits["val"].extend(shuffled[n_train:n_train + n_val])
        splits["test"].extend(shuffled[n_train + n_val:])

    return splits


def generate_t1_splits(
    records: list[dict[str, Any]],
    train: float = 0.8,
    test: float = 0.2,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Split T1 (PercePiano) records stratified by piece. No val split."""
    rng = random.Random(seed)

    groups: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        groups[rec["piece"]].append(rec)

    splits: dict[str, list[dict]] = {"train": [], "test": []}

    for _piece, group_recs in sorted(groups.items()):
        shuffled = list(group_recs)
        rng.shuffle(shuffled)
        n_test = max(1, round(len(shuffled) * test))
        splits["test"].extend(shuffled[:n_test])
        splits["train"].extend(shuffled[n_test:])

    return splits


def generate_t2_splits(
    records: list[dict[str, Any]],
    train: float = 0.85,
    test: float = 0.15,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Split T2 (competition) by holding out entire rounds.

    Holdout unit is (competition, round) to prevent same-performer
    leakage across splits.
    """
    rng = random.Random(seed)

    # Collect unique (competition, round) keys
    round_keys = sorted({(r["competition"], r["round"]) for r in records})
    rng.shuffle(round_keys)

    # Map rounds to their recordings
    round_to_recs: dict[tuple, list[dict]] = defaultdict(list)
    for rec in records:
        round_to_recs[(rec["competition"], rec["round"])].append(rec)

    # Greedily assign rounds to test until we hit the target fraction
    total = len(records)
    target_test = round(total * test)
    test_count = 0
    test_rounds: set[tuple] = set()

    for rk in round_keys:
        if test_count >= target_test:
            break
        test_rounds.add(rk)
        test_count += len(round_to_recs[rk])

    splits: dict[str, list[dict]] = {"train": [], "test": []}
    for rk, recs in round_to_recs.items():
        if rk in test_rounds:
            splits["test"].extend(recs)
        else:
            splits["train"].extend(recs)

    return splits
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/jdhiman/Documents/crescendai/model
uv run python -m pytest tests/model_improvement/test_splits.py -v
```

Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai
git add model/src/model_improvement/splits.py model/tests/model_improvement/test_splits.py
git commit -m "feat: three-way split generation with stratification (T1/T2/T5)"
```

---

### Task 3: Implement Skill Discrimination Metric (TDD)

**Files:**
- Create: `model/src/model_improvement/skill_discrimination.py`
- Create: `model/tests/model_improvement/test_skill_discrimination.py`

- [ ] **Step 1: Write the 3 failing metric tests**

Create `model/tests/model_improvement/test_skill_discrimination.py`:

```python
"""Tests for T5 skill discrimination pairwise metric."""

import numpy as np
import pytest

from model_improvement.skill_discrimination import (
    skill_discrimination_pairwise,
)


class TestSkillDiscrimination:
    def test_perfect_discrimination(self):
        """Model scores perfectly correlate with skill buckets -> 100% accuracy."""
        scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])  # 5 recordings
        buckets = np.array([1, 2, 3, 4, 5])
        result = skill_discrimination_pairwise(scores, buckets)
        assert result["pairwise_accuracy"] == 1.0
        assert result["n_pairs"] == 10  # C(5,2) = 10

    def test_single_recording_bucket(self):
        """Bucket with 1 recording still generates cross-bucket pairs."""
        scores = np.array([0.1, 0.3, 0.5])
        buckets = np.array([1, 1, 5])  # bucket 5 has only 1 recording
        result = skill_discrimination_pairwise(scores, buckets)
        # Pairs: (rec0,rec2) and (rec1,rec2) are cross-bucket. (rec0,rec1) is same-bucket (skipped).
        assert result["n_pairs"] == 2
        assert result["pairwise_accuracy"] == 1.0  # 0.1<0.5 and 0.3<0.5, both correct

    def test_per_dimension_shape(self):
        """Per-dimension breakdown returns one entry per dimension."""
        n_dims = 6
        scores = np.random.rand(10, n_dims)
        buckets = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
        result = skill_discrimination_pairwise(scores, buckets)
        assert len(result["per_dimension"]) == n_dims
        for d in range(n_dims):
            assert 0.0 <= result["per_dimension"][d] <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/jdhiman/Documents/crescendai/model
uv run python -m pytest tests/model_improvement/test_skill_discrimination.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement skill_discrimination.py**

Create `model/src/model_improvement/skill_discrimination.py`:

```python
"""T5 skill discrimination metric: can the model rank skill levels correctly?

Given model scores and ordinal skill buckets (1-5), compute pairwise
accuracy across all cross-bucket pairs. Higher bucket = higher expected score.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np


def skill_discrimination_pairwise(
    scores: np.ndarray,
    buckets: np.ndarray,
    n_bootstrap: int = 0,
    seed: int = 42,
) -> dict:
    """Pairwise accuracy: does the model score higher-bucket recordings higher?

    Only cross-bucket pairs are counted (same-bucket pairs are ambiguous).
    For multi-dimensional scores, computes per-dimension and overall (mean score).

    Args:
        scores: Model predictions, shape (n_recordings,) or (n_recordings, n_dims).
        buckets: Ordinal skill bucket per recording, shape (n_recordings,). Values 1-5.
        n_bootstrap: If >0, compute bootstrap 95% CI with this many resamples.
        seed: Random seed for bootstrap.

    Returns:
        Dict with keys:
        - pairwise_accuracy: float (overall, using mean across dims if multi-dim)
        - n_pairs: int
        - per_dimension: dict[int, float] (only if scores is 2D)
        - ci_lower, ci_upper: float (only if n_bootstrap > 0)
    """
    scores = np.asarray(scores)
    buckets = np.asarray(buckets)
    n = len(scores)

    is_multidim = scores.ndim == 2
    if not is_multidim:
        scores_1d = scores
    else:
        scores_1d = scores.mean(axis=1)

    # Generate all cross-bucket pairs
    correct = 0
    total = 0
    per_dim_correct: dict[int, int] = {}
    per_dim_total: dict[int, int] = {}

    if is_multidim:
        n_dims = scores.shape[1]
        for d in range(n_dims):
            per_dim_correct[d] = 0
            per_dim_total[d] = 0

    for i, j in combinations(range(n), 2):
        if buckets[i] == buckets[j]:
            continue  # same bucket, skip

        total += 1
        # Convention: higher bucket = higher expected score
        if buckets[i] < buckets[j]:
            low_idx, high_idx = i, j
        else:
            low_idx, high_idx = j, i

        if scores_1d[high_idx] > scores_1d[low_idx]:
            correct += 1

        if is_multidim:
            for d in range(n_dims):
                per_dim_total[d] += 1
                if scores[high_idx, d] > scores[low_idx, d]:
                    per_dim_correct[d] += 1

    result: dict = {
        "pairwise_accuracy": correct / total if total > 0 else 0.5,
        "n_pairs": total,
    }

    if is_multidim:
        result["per_dimension"] = {
            d: per_dim_correct[d] / per_dim_total[d] if per_dim_total[d] > 0 else 0.5
            for d in range(n_dims)
        }

    # Bootstrap CI
    if n_bootstrap > 0 and total > 0:
        rng = np.random.RandomState(seed)
        boot_accs = []
        indices = np.arange(n)
        for _ in range(n_bootstrap):
            sample = rng.choice(indices, size=n, replace=True)
            s_scores = scores_1d[sample]
            s_buckets = buckets[sample]
            bc, bt = 0, 0
            for ii, jj in combinations(range(n), 2):
                if s_buckets[ii] == s_buckets[jj]:
                    continue
                bt += 1
                if s_buckets[ii] < s_buckets[jj]:
                    lo, hi = ii, jj
                else:
                    lo, hi = jj, ii
                if s_scores[hi] > s_scores[lo]:
                    bc += 1
            boot_accs.append(bc / bt if bt > 0 else 0.5)
        result["ci_lower"] = float(np.percentile(boot_accs, 2.5))
        result["ci_upper"] = float(np.percentile(boot_accs, 97.5))

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/jdhiman/Documents/crescendai/model
uv run python -m pytest tests/model_improvement/test_skill_discrimination.py -v
```

Expected: all 3 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai
git add model/src/model_improvement/skill_discrimination.py model/tests/model_improvement/test_skill_discrimination.py
git commit -m "feat: T5 skill discrimination pairwise metric with bootstrap CIs"
```

---

### Task 4: Implement Data Integrity Checks

**Files:**
- Create: `model/src/model_improvement/data_integrity.py`

- [ ] **Step 1: Implement data_integrity.py**

Create `model/src/model_improvement/data_integrity.py`:

```python
"""T5 data integrity checks. Run before split generation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


def load_all_t5_manifests(skill_eval_dir: Path) -> list[dict[str, Any]]:
    """Load all T5 recordings from manifest.yaml files."""
    all_recordings = []
    for manifest_path in sorted(skill_eval_dir.glob("*/manifest.yaml")):
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)
        piece = manifest["piece"]
        for rec in manifest.get("recordings", []):
            rec["piece"] = piece
            all_recordings.append(rec)
    return all_recordings


def check_integrity(
    recordings: list[dict[str, Any]],
    audio_dir: Path | None = None,
    embedding_dir: Path | None = None,
) -> list[str]:
    """Run all integrity checks on T5 recordings.

    Returns:
        List of error strings. Empty list means all checks pass.
    """
    errors: list[str] = []

    # Check 1: duplicate video_ids
    seen_ids: dict[str, str] = {}
    for rec in recordings:
        vid = rec["video_id"]
        piece = rec["piece"]
        if vid in seen_ids:
            errors.append(
                f"Duplicate video_id {vid}: in {seen_ids[vid]} and {piece}"
            )
        seen_ids[vid] = piece

    # Check 2: bucket balance (min 3 per piece+bucket)
    groups: dict[tuple, int] = defaultdict(int)
    for rec in recordings:
        groups[(rec["piece"], rec["skill_bucket"])] += 1
    for (piece, bucket), count in sorted(groups.items()):
        if count < 3:
            errors.append(
                f"piece={piece}, bucket={bucket}: only {count} recordings (need >=3)"
            )

    # Check 3: audio files exist (if audio_dir provided)
    if audio_dir is not None:
        for rec in recordings:
            piece = rec["piece"]
            vid = rec["video_id"]
            audio_path = audio_dir / piece / f"{vid}.wav"
            if not audio_path.exists():
                # Also check .opus and .webm
                alt_paths = [
                    audio_dir / piece / f"{vid}.opus",
                    audio_dir / piece / f"{vid}.webm",
                ]
                if not any(p.exists() for p in alt_paths):
                    errors.append(f"Missing audio: {piece}/{vid}")

    # Check 4: embeddings exist and are non-empty (if embedding_dir provided)
    if embedding_dir is not None:
        for rec in recordings:
            vid = rec["video_id"]
            emb_path = embedding_dir / f"{vid}.pt"
            if not emb_path.exists():
                errors.append(f"Missing embedding: {vid}.pt")
            elif emb_path.stat().st_size == 0:
                errors.append(f"Empty embedding: {vid}.pt")

    return errors
```

- [ ] **Step 2: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai
git add model/src/model_improvement/data_integrity.py
git commit -m "feat: T5 data integrity checks (duplicates, balance, files)"
```

---

### Task 5: Implement Trackio Callback

**Files:**
- Create: `model/src/model_improvement/trackio_callback.py`

- [ ] **Step 1: Implement trackio_callback.py**

Create `model/src/model_improvement/trackio_callback.py`:

```python
"""PyTorch Lightning callback for Trackio experiment tracking."""

from __future__ import annotations

import subprocess
from typing import Any

import pytorch_lightning as pl
import torch


class TrackioCallback(pl.Callback):
    """Log training metrics to Trackio per epoch.

    Logs: train_loss, val_skill_discrimination, learning_rate.
    Requires trackio to be installed: uv pip install trackio
    """

    def __init__(
        self,
        experiment_id: str,
        project: str = "crescendai-training",
        config: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.experiment_id = experiment_id
        self.project = project
        self.config = config or {}
        self._run = None

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        try:
            import trackio
            self._run = trackio.Run(
                name=self.experiment_id,
                project=self.project,
                config=self.config,
            )
            # Log git commit hash
            try:
                commit = subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"],
                    text=True, stderr=subprocess.DEVNULL,
                ).strip()
                self._run.log({"git_commit": commit}, step=0)
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass
        except ImportError:
            print("WARNING: trackio not installed, skipping experiment tracking")
            self._run = None

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._run is None:
            return

        metrics: dict[str, float] = {}
        for key, val in trainer.callback_metrics.items():
            if isinstance(val, torch.Tensor):
                metrics[key] = val.item()
            elif isinstance(val, (int, float)):
                metrics[key] = float(val)

        # Extract learning rate from optimizer
        if trainer.optimizers:
            opt = trainer.optimizers[0]
            metrics["learning_rate"] = opt.param_groups[0]["lr"]

        self._run.log(metrics, step=trainer.current_epoch)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._run is not None:
            self._run.finish()
```

- [ ] **Step 2: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai
git add model/src/model_improvement/trackio_callback.py
git commit -m "feat: Trackio PL callback for experiment tracking"
```

---

### Task 6: Implement Config-Driven Autoresearch Runner

**Files:**
- Create: `model/src/model_improvement/autoresearch_runner.py`

- [ ] **Step 1: Implement autoresearch_runner.py**

Create `model/src/model_improvement/autoresearch_runner.py`:

```python
"""Config-driven autoresearch sweep framework.

Replaces per-phase scripts with a single runner that takes a phase config
and handles the sweep loop, Trackio logging, keep/revert logic.

Usage:
    cd model/
    uv run python -m model_improvement.autoresearch_runner --phase lr_schedule
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Callable

from src.paths import Results


@dataclass
class SweepConfig:
    """Configuration for one autoresearch phase."""
    name: str
    search_space: dict[str, list[Any]]
    train_fn: Callable[[dict[str, Any]], dict[str, float]]
    metric_key: str = "pairwise_accuracy"
    higher_is_better: bool = True
    results_dir: Path = field(default_factory=lambda: Results.root / "autoresearch")


def run_sweep(config: SweepConfig) -> dict[str, Any]:
    """Execute a full sweep over the search space.

    For each combination in the search space, calls config.train_fn(params)
    which must return a dict with at least config.metric_key.

    Returns:
        Dict with keys: best_params, best_metric, all_results, elapsed_seconds.
    """
    config.results_dir.mkdir(parents=True, exist_ok=True)

    # Generate all parameter combinations
    param_names = sorted(config.search_space.keys())
    param_values = [config.search_space[k] for k in param_names]
    combos = list(product(*param_values))

    print(f"=== Autoresearch: {config.name} ===")
    print(f"Search space: {len(combos)} combinations")
    print(f"Metric: {config.metric_key} ({'higher' if config.higher_is_better else 'lower'} is better)")

    best_metric = float("-inf") if config.higher_is_better else float("inf")
    best_params: dict[str, Any] = {}
    all_results: list[dict] = []
    start = time.time()

    for i, combo in enumerate(combos):
        params = dict(zip(param_names, combo))
        print(f"\n[{i+1}/{len(combos)}] {params}")

        try:
            result = config.train_fn(params)
            metric = result[config.metric_key]
            print(f"  -> {config.metric_key}={metric:.4f}")

            is_better = (
                metric > best_metric if config.higher_is_better
                else metric < best_metric
            )
            if is_better:
                best_metric = metric
                best_params = params
                print(f"  -> NEW BEST ({config.metric_key}={metric:.4f})")

            all_results.append({
                "params": params,
                "result": result,
                "is_best": is_better,
            })
        except Exception as e:
            print(f"  -> FAILED: {e}")
            all_results.append({
                "params": params,
                "result": None,
                "error": str(e),
            })

    elapsed = time.time() - start

    summary = {
        "phase": config.name,
        "best_params": best_params,
        "best_metric": best_metric,
        "total_experiments": len(combos),
        "successful": sum(1 for r in all_results if r.get("result") is not None),
        "elapsed_seconds": round(elapsed, 1),
        "all_results": all_results,
    }

    # Save results
    results_path = config.results_dir / f"{config.name}_results.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")
    print(f"Best: {best_params} -> {config.metric_key}={best_metric:.4f}")
    print(f"Total time: {elapsed:.0f}s")

    return summary


# --- Phase configs (populated when T5 data is ready) ---

PHASE_REGISTRY: dict[str, Callable[[], SweepConfig]] = {}


def register_phase(name: str):
    """Decorator to register a phase config factory."""
    def decorator(fn: Callable[[], SweepConfig]):
        PHASE_REGISTRY[name] = fn
        return fn
    return decorator


def main():
    parser = argparse.ArgumentParser(description="Config-driven autoresearch runner")
    parser.add_argument("--phase", required=True, choices=list(PHASE_REGISTRY.keys()),
                        help="Which autoresearch phase to run")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print search space without running")
    args = parser.parse_args()

    config_factory = PHASE_REGISTRY[args.phase]
    config = config_factory()

    if args.dry_run:
        param_names = sorted(config.search_space.keys())
        total = 1
        for values in config.search_space.values():
            total *= len(values)
        print(f"Phase: {config.name}")
        print(f"Parameters: {param_names}")
        print(f"Total combinations: {total}")
        for name, values in sorted(config.search_space.items()):
            print(f"  {name}: {values}")
        return

    run_sweep(config)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai
git add model/src/model_improvement/autoresearch_runner.py
git commit -m "feat: config-driven autoresearch sweep framework"
```

---

### Task 7: Implement prepare_training.py Pipeline Script

**Files:**
- Create: `model/scripts/prepare_training.py`

- [ ] **Step 1: Implement prepare_training.py**

Create `model/scripts/prepare_training.py`:

```python
"""One-command pipeline: T5 labeling done -> training ready.

Steps:
1. Load all T5 manifests
2. Run data integrity checks
3. Generate train/val/test splits for T1, T2, T5
4. Save splits to model/data/splits/
5. Print summary

Embedding extraction and HF Bucket upload are separate steps
(require GPU and network respectively).

Usage:
    cd model/
    uv run python scripts/prepare_training.py
    uv run python scripts/prepare_training.py --check-audio
    uv run python scripts/prepare_training.py --check-embeddings
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.paths import Evals, Splits, Embeddings

from model_improvement.data_integrity import load_all_t5_manifests, check_integrity
from model_improvement.splits import (
    generate_t5_splits,
    generate_t1_splits,
    generate_t2_splits,
)


def main():
    parser = argparse.ArgumentParser(description="Prepare training data from T5 manifests")
    parser.add_argument("--check-audio", action="store_true",
                        help="Also check that audio files exist")
    parser.add_argument("--check-embeddings", action="store_true",
                        help="Also check that embedding files exist")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    skill_eval_dir = Evals.skill_eval
    print(f"Loading T5 manifests from {skill_eval_dir}...")
    recordings = load_all_t5_manifests(skill_eval_dir)
    # Filter to downloaded only
    recordings = [r for r in recordings if r.get("downloaded", False)]
    print(f"  {len(recordings)} downloaded recordings across {len({r['piece'] for r in recordings})} pieces")

    # Data integrity checks
    print("\nRunning integrity checks...")
    audio_dir = skill_eval_dir if args.check_audio else None
    emb_dir = Embeddings.t5_muq if args.check_embeddings else None
    errors = check_integrity(recordings, audio_dir=audio_dir, embedding_dir=emb_dir)

    if errors:
        print(f"\nFOUND {len(errors)} INTEGRITY ERRORS:")
        for err in errors:
            print(f"  - {err}")
        raise SystemExit(1)
    print("  All checks passed")

    # Generate splits
    print(f"\nGenerating T5 splits (seed={args.seed})...")
    t5_splits = generate_t5_splits(recordings, train=0.8, val=0.1, test=0.1, seed=args.seed)
    print(f"  train={len(t5_splits['train'])}, val={len(t5_splits['val'])}, test={len(t5_splits['test'])}")

    # Save splits
    Splits.root.mkdir(parents=True, exist_ok=True)
    splits_path = Splits.root / "t5_splits.json"
    serializable = {
        split_name: [
            {"video_id": r["video_id"], "piece": r["piece"], "skill_bucket": r["skill_bucket"]}
            for r in recs
        ]
        for split_name, recs in t5_splits.items()
    }
    with open(splits_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"  Saved to {splits_path}")

    # Summary
    print("\n=== SUMMARY ===")
    print(f"T5: {len(recordings)} recordings -> train/val/test split")
    pieces = sorted({r["piece"] for r in recordings})
    print(f"Pieces: {len(pieces)}")
    for piece in pieces:
        piece_recs = [r for r in recordings if r["piece"] == piece]
        buckets = sorted({r["skill_bucket"] for r in piece_recs})
        print(f"  {piece}: {len(piece_recs)} recordings, buckets {buckets}")

    print(f"\nSplits saved to {splits_path}")
    print("Next steps:")
    print("  1. Extract MuQ embeddings: uv run python scripts/extract_t5_muq.py")
    print("  2. Extract Aria embeddings: uv run python scripts/extract_t5_aria.py")
    print("  3. Upload to HF Bucket: hf buckets sync model/data/embeddings/ hf://buckets/crescendai/training-data/embeddings/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai
git add model/scripts/prepare_training.py
git commit -m "feat: one-command prepare_training.py pipeline script"
```

---

### Task 8: Doc Updates -- Replace Thunder Compute with HF Jobs

**Files:**
- Modify: `docs/model/01-data.md`
- Modify: `model/CLAUDE.md`
- Modify: `docs/model/04-north-star.md`
- Modify: `docs/model/03-encoders.md`

- [ ] **Step 1: Update docs/model/01-data.md**

Replace the Thunder Compute references. Find the storage table (~line 200-204) and update:

```
| Local (Mac) | 68GB available (12 GB cleaned) | Code, caches, composite labels |
| GDrive | 80GB total | Checkpoints, results, final embeddings |
| Thunder Compute | 500GB | Raw audio download, processing, MuQ extraction |
```

Replace with:

```
| Local (Mac) | 68GB available | Active working set (current experiment's embeddings + T5 data) |
| HF Bucket (private) | ~92GB | Embeddings, manifests, checkpoints, T5 audio |
| GDrive (via rclone) | 80GB total | Archival backup: results, labels, final weights |
| HF Jobs (cloud) | On-demand | Full training runs, Aria fine-tuning, validation |
```

Also replace the processing principle (~line 205):

```
**Principle:** Raw audio lives and dies on the remote. Only `.pt` embeddings and `.jsonl` metadata return.
```

Replace with:

```
**Principle:** Training data lives on HF Bucket. Local disk holds only the active experiment's data. GDrive is archival backup only.
```

Also find the "All heavy data processing" sentence (~line 3) referencing Thunder Compute:

```
Dataset inventory for piano performance evaluation training. All heavy data processing (audio download, segmentation, MuQ extraction) runs on Thunder Compute (A100). Only embeddings and metadata come back.
```

Replace with:

```
Dataset inventory for piano performance evaluation training. Heavy data processing (embedding extraction, full training runs) runs on HF Jobs (L4 $0.80/hr default, A100 $2.50/hr for Aria). Training data stored on HF Bucket.
```

- [ ] **Step 2: Update model/CLAUDE.md**

Find the Stack section and replace:

```
- **Training**: Thunder Compute (A100 GPU/80GB VRAM with 8 vCPU/64GB RAM)
```

with:

```
- **Training**: HF Jobs (L4 $0.80/hr default, A100 $2.50/hr for Aria fine-tuning)
- **Experiment tracking**: Trackio (syncs to HF Space dashboard)
- **Training data**: HF Bucket (private, ~92GB, mounted in HF Jobs)
```

- [ ] **Step 3: Update docs/model/04-north-star.md**

Find the Training Evaluation table (~line 159-163) and update:

```
| **E1** | Pairwise accuracy (clean folds) | > 75% (audio), > 70% (symbolic), > 80% (fused) | Core ranking quality. ...
```

Replace with:

```
| **E1** | T5 skill discrimination (pairwise, val split) | > 70% cross-bucket accuracy | Primary metric: can model rank skill levels? Single train/val/test split. |
```

Also find "All evaluation on clean piece-stratified folds" (~line 173) and replace with:

```
Evaluation uses single train/val/test split. T5 val (10%) for autoresearch optimization. T5 test (10%) + T1 test (20%) + T2 test (15%) for final reporting only (never seen during optimization). Bootstrap 95% CIs on all reported metrics.
```

- [ ] **Step 4: Update docs/model/03-encoders.md**

Add to the Aria section (after the existing validation results):

```
**Confound check (required on every experiment):** Aria-only skill discrimination on T5 val. If Aria discriminates skill buckets (above 50% chance), the signal is musical. If MuQ discriminates but Aria doesn't, MuQ may be exploiting audio quality as a shortcut.
```

- [ ] **Step 5: Commit all doc updates**

```bash
cd /Users/jdhiman/Documents/crescendai
git add docs/model/01-data.md model/CLAUDE.md docs/model/04-north-star.md docs/model/03-encoders.md
git commit -m "docs: replace Thunder Compute with HF Jobs, update eval strategy"
```

---

### Task 9: Update Project Memory

**Files:**
- Modify: `~/.claude/projects/-Users-jdhiman-Documents-crescendai/memory/MEMORY.md`

- [ ] **Step 1: Update memory with training readiness decisions**

Update the MEMORY.md index to reflect the key decisions from this spec:

- Eval strategy changed from 4-fold CV to single train/val/test split
- Thunder Compute replaced by HF Jobs (with Thunder as documented fallback)
- HF Buckets for training data storage
- Trackio for experiment tracking
- T5 skill discrimination as primary metric

- [ ] **Step 2: Commit** (memory files are not committed to repo)

No git commit needed for memory files.

---

## Self-Review

**Spec coverage check:**
- Section 1 (Eval strategy): Task 2 (splits), Task 3 (metric) -- covered
- Section 2 (Compute): Task 8 (doc updates) -- covered
- Section 3 (Storage): Task 7 (prepare_training.py mentions HF Bucket upload), Task 8 (docs) -- covered
- Section 4 (Trackio): Task 5 -- covered
- Section 5 (Autoresearch): Task 6 -- covered (framework only; phase configs are registered when T5 data arrives)
- Section 6 (Aria research): Not in this plan -- spec says "separate research task, not blocking"
- Section 7 (Doc updates): Task 8 -- covered
- Section 8 (Pipeline): Task 7 (prepare_training.py) + Task 4 (integrity) -- covered
- Data integrity checks: Task 4 -- covered
- Unit tests: Task 2 (5 split tests) + Task 3 (3 metric tests) -- covered
- Audio quality annotation: Labeling workflow, not code -- noted in spec, no code task needed
- Bootstrap CIs: Built into Task 3's `skill_discrimination_pairwise(n_bootstrap=...)` -- covered
- GDrive cleanup: Manual operational task, not code -- mentioned in spec Section 3

**Placeholder scan:** No TBDs, TODOs, or "fill in later" found.

**Type consistency:** `generate_t5_splits`, `generate_t1_splits`, `generate_t2_splits` names match between tests and implementation. `skill_discrimination_pairwise` name matches. `SweepConfig` and `run_sweep` are consistent.
