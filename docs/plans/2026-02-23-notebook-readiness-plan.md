# Notebook Readiness Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Update all audio and symbolic training/comparison notebooks and their backing modules to use the 6-dim composite taxonomy, complete evaluation logic, and reflect the distillation NO-GO decision.

**Architecture:** Layered -- shared Python modules first (`taxonomy.py`, `evaluation.py`, updates to `metrics.py`/encoders), then notebooks (audio path, symbolic path), then docs. All logic in `model/src/model_improvement/`; notebooks import and orchestrate.

**Tech Stack:** PyTorch Lightning, pytest, rclone, scipy, numpy, torch-geometric

---

## Task 1: Create `taxonomy.py` module

**Files:**
- Create: `model/src/model_improvement/taxonomy.py`
- Test: `model/tests/model_improvement/test_taxonomy.py`

**Step 1: Write the failing test**

```python
# model/tests/model_improvement/test_taxonomy.py
import json
import numpy as np
import pytest
from model_improvement.taxonomy import (
    DIMENSIONS,
    NUM_DIMS,
    load_composite_labels,
    load_dimension_definitions,
    get_dimension_names,
)


def test_dimensions_constant():
    assert DIMENSIONS == [
        "dynamics", "timing", "pedaling",
        "articulation", "phrasing", "interpretation",
    ]
    assert NUM_DIMS == 6


def test_load_composite_labels(tmp_path):
    data = {
        "seg_a": {"dynamics": 0.5, "timing": 0.6, "pedaling": 0.7,
                  "articulation": 0.4, "phrasing": 0.3, "interpretation": 0.8},
        "seg_b": {"dynamics": 0.1, "timing": 0.2, "pedaling": 0.3,
                  "articulation": 0.4, "phrasing": 0.5, "interpretation": 0.6},
    }
    path = tmp_path / "composite_labels.json"
    path.write_text(json.dumps(data))

    result = load_composite_labels(path)
    assert set(result.keys()) == {"seg_a", "seg_b"}
    assert isinstance(result["seg_a"], np.ndarray)
    assert result["seg_a"].shape == (6,)
    np.testing.assert_allclose(result["seg_a"], [0.5, 0.6, 0.7, 0.4, 0.3, 0.8])


def test_load_dimension_definitions(tmp_path):
    data = {"dimensions": {"dynamics": {"description": "test"}}}
    path = tmp_path / "dimension_definitions.json"
    path.write_text(json.dumps(data))
    result = load_dimension_definitions(path)
    assert "dimensions" in result


def test_get_dimension_names():
    names = get_dimension_names()
    assert len(names) == 6
    assert names[0] == "dynamics"
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/jdhiman/Documents/crescendai/model && python -m pytest tests/model_improvement/test_taxonomy.py -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError`

**Step 3: Write minimal implementation**

```python
# model/src/model_improvement/taxonomy.py
"""Teacher-grounded taxonomy: 6-dim composite label system.

Single source of truth for dimension names, count, and label loading.
Replaces the hardcoded 19-dim PercePiano labels throughout the pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DIMENSIONS: list[str] = [
    "dynamics",
    "timing",
    "pedaling",
    "articulation",
    "phrasing",
    "interpretation",
]

NUM_DIMS: int = len(DIMENSIONS)


def load_composite_labels(path: str | Path) -> dict[str, np.ndarray]:
    """Load composite labels JSON, returning segment_id -> [6] numpy array.

    The JSON has structure: {segment_id: {dim_name: score, ...}}.
    Returns arrays with dimensions ordered per DIMENSIONS constant.
    """
    with open(path) as f:
        raw = json.load(f)

    result = {}
    for seg_id, dim_scores in raw.items():
        vec = np.array([dim_scores[d] for d in DIMENSIONS], dtype=np.float32)
        result[seg_id] = vec
    return result


def load_dimension_definitions(path: str | Path) -> dict:
    """Load dimension_definitions.json (full taxonomy metadata)."""
    with open(path) as f:
        return json.load(f)


def get_dimension_names() -> list[str]:
    """Return ordered dimension names."""
    return list(DIMENSIONS)
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/jdhiman/Documents/crescendai/model && python -m pytest tests/model_improvement/test_taxonomy.py -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add model/src/model_improvement/taxonomy.py model/tests/model_improvement/test_taxonomy.py
git commit -m "feat(model): add taxonomy module for 6-dim composite label system"
```

---

## Task 2: Update `metrics.py` with new metric functions

**Files:**
- Modify: `model/src/model_improvement/metrics.py`
- Modify: `model/tests/model_improvement/test_metrics.py`

**Step 1: Write failing tests**

Append to `model/tests/model_improvement/test_metrics.py`:

```python
from model_improvement.metrics import (
    stop_auc,
    competition_spearman,
    per_dimension_breakdown,
)


def test_stop_auc():
    embeddings = torch.randn(50, 1024)
    is_stop = torch.tensor([1] * 25 + [0] * 25)
    video_ids = [f"v{i % 5}" for i in range(50)]
    result = stop_auc(embeddings, is_stop, video_ids)
    assert "auc" in result
    assert 0.0 <= result["auc"] <= 1.0


def test_competition_spearman():
    predictions = torch.randn(20, 6)
    placements = torch.arange(1, 21, dtype=torch.float32)
    result = competition_spearman(predictions, placements)
    assert "rho" in result
    assert "p_value" in result


def test_competition_spearman_empty():
    result = competition_spearman(None, None)
    assert result is None


def test_per_dimension_breakdown():
    suite = MetricsSuite()
    logits = torch.randn(100, 6)
    labels_a = torch.rand(100, 6)
    labels_b = torch.rand(100, 6)
    result = per_dimension_breakdown(suite, logits, labels_a, labels_b)
    assert len(result) == 6
    assert "dynamics" in result
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/jdhiman/Documents/crescendai/model && python -m pytest tests/model_improvement/test_metrics.py::test_stop_auc -v`
Expected: FAIL with `ImportError`

**Step 3: Implement**

Add to end of `model/src/model_improvement/metrics.py`:

```python
def stop_auc(
    embeddings: torch.Tensor,
    is_stop: torch.Tensor,
    video_ids: list[str],
) -> dict:
    """LOVO-CV AUC for STOP prediction using logistic regression.

    Args:
        embeddings: Feature vectors, shape (n_samples, dim).
        is_stop: Binary labels (1=STOP, 0=CONTINUE), shape (n_samples,).
        video_ids: Video ID per sample for leave-one-video-out CV.

    Returns:
        Dict with "auc" (float) and "per_video_auc" (dict).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    X = embeddings.detach().cpu().numpy()
    y = is_stop.detach().cpu().numpy()
    unique_videos = sorted(set(video_ids))

    all_probs = np.zeros(len(y))
    per_video: Dict[str, float] = {}

    for held_out in unique_videos:
        test_mask = np.array([v == held_out for v in video_ids])
        train_mask = ~test_mask

        if y[train_mask].sum() == 0 or y[train_mask].sum() == train_mask.sum():
            continue
        if y[test_mask].sum() == 0 or y[test_mask].sum() == test_mask.sum():
            continue

        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        clf.fit(X[train_mask], y[train_mask])
        probs = clf.predict_proba(X[test_mask])[:, 1]
        all_probs[test_mask] = probs

        try:
            per_video[held_out] = float(roc_auc_score(y[test_mask], probs))
        except ValueError:
            pass

    try:
        overall_auc = float(roc_auc_score(y, all_probs))
    except ValueError:
        overall_auc = 0.5

    return {"auc": overall_auc, "per_video_auc": per_video}


def competition_spearman(
    predictions: torch.Tensor | None,
    placements: torch.Tensor | None,
) -> dict | None:
    """Spearman rho of mean prediction vs competition placement.

    Returns None if inputs are None (graceful skip when no T2 data).
    """
    if predictions is None or placements is None:
        return None

    pred_np = predictions.detach().cpu().numpy()
    place_np = placements.detach().cpu().numpy()
    mean_pred = pred_np.mean(axis=1)
    rho, p_value = stats.spearmanr(mean_pred, place_np)
    return {"rho": float(rho), "p_value": float(p_value)}


def per_dimension_breakdown(
    suite: MetricsSuite,
    logits: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
) -> dict[str, float]:
    """Per-dimension pairwise accuracy using taxonomy dimension names.

    Returns dict mapping dimension name to pairwise accuracy.
    """
    from model_improvement.taxonomy import DIMENSIONS

    pw = suite.pairwise_accuracy(logits, labels_a, labels_b)
    per_dim = pw["per_dimension"]
    return {DIMENSIONS[d]: acc for d, acc in per_dim.items() if d < len(DIMENSIONS)}
```

Note: `sklearn` imports are deferred inside `stop_auc` to avoid mandatory dependency at module load time.

**Step 4: Run tests**

Run: `cd /Users/jdhiman/Documents/crescendai/model && python -m pytest tests/model_improvement/test_metrics.py -v`
Expected: PASS (all 9 tests)

**Step 5: Commit**

```bash
git add model/src/model_improvement/metrics.py model/tests/model_improvement/test_metrics.py
git commit -m "feat(model): add stop_auc, competition_spearman, per_dimension_breakdown to metrics"
```

---

## Task 3: Update encoder defaults from `num_labels=19` to `NUM_DIMS`

**Files:**
- Modify: `model/src/model_improvement/audio_encoders.py` (lines 32, 320, 647)
- Modify: `model/src/model_improvement/symbolic_encoders.py` (lines 346, 661, 1041)
- Modify: `model/tests/model_improvement/test_audio_encoders.py`
- Modify: `model/tests/model_improvement/test_symbolic_encoders.py`

**Step 1: Write failing test**

Append to `model/tests/model_improvement/test_audio_encoders.py`:

```python
def test_default_num_labels_is_taxonomy():
    from model_improvement.taxonomy import NUM_DIMS
    model = MuQLoRAModel()
    assert model.num_labels == NUM_DIMS
```

Append to `model/tests/model_improvement/test_symbolic_encoders.py`:

```python
def test_gnn_default_num_labels_is_taxonomy():
    from model_improvement.taxonomy import NUM_DIMS
    model = GNNSymbolicEncoder()
    assert model.num_labels == NUM_DIMS
```

**Step 2: Run to verify failure**

Run: `cd /Users/jdhiman/Documents/crescendai/model && python -m pytest tests/model_improvement/test_audio_encoders.py::test_default_num_labels_is_taxonomy tests/model_improvement/test_symbolic_encoders.py::test_gnn_default_num_labels_is_taxonomy -v`
Expected: FAIL (num_labels=19 != 6)

**Step 3: Make changes**

In `model/src/model_improvement/audio_encoders.py`:
- Add import at top: `from model_improvement.taxonomy import NUM_DIMS`
- Line 32: change `num_labels: int = 19` to `num_labels: int = NUM_DIMS`
- Line 320: change `num_labels: int = 19` to `num_labels: int = NUM_DIMS`
- Line 647: change `num_labels: int = 19` to `num_labels: int = NUM_DIMS`

In `model/src/model_improvement/symbolic_encoders.py`:
- Add import at top: `from model_improvement.taxonomy import NUM_DIMS`
- Line 346: change `num_labels: int = 19` to `num_labels: int = NUM_DIMS`
- Line 661: change `num_labels: int = 19` to `num_labels: int = NUM_DIMS`
- Line 1041: change `num_labels: int = 19` to `num_labels: int = NUM_DIMS`

Note: `TransformerSymbolicEncoder` at line 32 has `num_labels: int` (no default) -- leave it; callers already pass explicitly.

**Step 4: Fix existing tests that rely on old default**

Existing tests create models without specifying `num_labels` and expect 19-dim behavior. Add `num_labels=19` explicitly to those test constructors. Search for model instantiations in the test files and add the parameter where the test's tensor shapes assume 19 dimensions.

**Step 5: Run full encoder test suites**

Run: `cd /Users/jdhiman/Documents/crescendai/model && python -m pytest tests/model_improvement/test_audio_encoders.py tests/model_improvement/test_symbolic_encoders.py -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add model/src/model_improvement/audio_encoders.py model/src/model_improvement/symbolic_encoders.py model/tests/model_improvement/test_audio_encoders.py model/tests/model_improvement/test_symbolic_encoders.py
git commit -m "refactor(model): change default num_labels from 19 to NUM_DIMS (6)"
```

---

## Task 4: Create `evaluation.py` module

**Files:**
- Create: `model/src/model_improvement/evaluation.py`
- Test: `model/tests/model_improvement/test_evaluation.py`

**Step 1: Write failing tests**

```python
# model/tests/model_improvement/test_evaluation.py
import torch
import pytest
from model_improvement.evaluation import (
    aggregate_folds,
    select_winner,
)


def test_aggregate_folds():
    folds = [
        {"pairwise": 0.80, "r2": 0.50},
        {"pairwise": 0.85, "r2": 0.55},
        {"pairwise": 0.82, "r2": 0.52},
        {"pairwise": 0.78, "r2": 0.48},
    ]
    result = aggregate_folds(folds)
    assert "pairwise_mean" in result
    assert "pairwise_std" in result
    assert abs(result["pairwise_mean"] - 0.8125) < 0.001


def test_aggregate_folds_handles_missing_keys():
    folds = [
        {"pairwise": 0.80},
        {"pairwise": 0.85, "r2": 0.55},
    ]
    result = aggregate_folds(folds)
    assert "pairwise_mean" in result
    assert "r2_mean" in result


def test_select_winner():
    results = {
        "A1": {"pairwise_mean": 0.85, "r2_mean": 0.50, "score_drop_pct": 5.0},
        "A2": {"pairwise_mean": 0.87, "r2_mean": 0.55, "score_drop_pct": 8.0},
        "A3": {"pairwise_mean": 0.83, "r2_mean": 0.60, "score_drop_pct": 20.0},
    }
    winner = select_winner(results)
    assert winner == "A2"  # A3 vetoed (20% > 15%), A2 beats A1 on pairwise


def test_select_winner_all_vetoed():
    results = {
        "A1": {"pairwise_mean": 0.85, "score_drop_pct": 20.0},
    }
    winner = select_winner(results)
    assert winner is None
```

**Step 2: Run to verify failure**

Run: `cd /Users/jdhiman/Documents/crescendai/model && python -m pytest tests/model_improvement/test_evaluation.py -v`
Expected: FAIL with `ImportError`

**Step 3: Implement**

```python
# model/src/model_improvement/evaluation.py
"""Shared evaluation logic for audio and symbolic comparison notebooks."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from model_improvement.metrics import (
    MetricsSuite,
    compute_robustness_metrics,
    competition_spearman,
)
from model_improvement.taxonomy import NUM_DIMS

ROBUSTNESS_VETO_THRESHOLD = 15.0


def evaluate_model(
    model: torch.nn.Module,
    val_keys: list[str],
    labels: dict[str, list[float] | np.ndarray],
    get_input_fn: Callable[[str], tuple[torch.Tensor, torch.Tensor]],
    encode_fn: Callable,
    compare_fn: Callable,
    predict_fn: Callable,
    num_dims: int = NUM_DIMS,
) -> dict:
    """Evaluate a model on one fold's validation keys.

    Args:
        model: The encoder model (in .eval() mode).
        val_keys: Segment keys in this fold's validation set.
        labels: segment_key -> label array (at least num_dims long).
        get_input_fn: key -> (input_tensor[1,...], mask[1,...]).
        encode_fn: (model, input, mask) -> z embedding [1, D].
        compare_fn: (model, z_a, z_b) -> ranking logits [1, num_dims].
        predict_fn: (model, input, mask) -> scores [1, num_dims].
        num_dims: Number of label dimensions.

    Returns:
        Dict with "pairwise", "pairwise_detail", "r2" keys.
    """
    suite = MetricsSuite(ambiguous_threshold=0.05)
    results = {}
    valid_keys = [k for k in val_keys if k in labels]
    model.eval()

    with torch.no_grad():
        all_logits, all_la, all_lb = [], [], []
        for i, key_a in enumerate(valid_keys):
            for key_b in valid_keys[i + 1:]:
                try:
                    inp_a, mask_a = get_input_fn(key_a)
                    inp_b, mask_b = get_input_fn(key_b)
                    z_a = encode_fn(model, inp_a, mask_a)
                    z_b = encode_fn(model, inp_b, mask_b)
                    logits = compare_fn(model, z_a, z_b)
                    lab_a = torch.tensor(
                        labels[key_a][:num_dims], dtype=torch.float32
                    )
                    lab_b = torch.tensor(
                        labels[key_b][:num_dims], dtype=torch.float32
                    )
                    all_logits.append(logits)
                    all_la.append(lab_a.unsqueeze(0))
                    all_lb.append(lab_b.unsqueeze(0))
                except Exception:
                    continue

        if all_logits:
            pw = suite.pairwise_accuracy(
                torch.cat(all_logits), torch.cat(all_la), torch.cat(all_lb)
            )
            results["pairwise"] = pw["overall"]
            results["pairwise_detail"] = pw

        all_preds, all_targets = [], []
        for key in valid_keys:
            try:
                inp, mask = get_input_fn(key)
                pred = predict_fn(model, inp, mask)
                target = torch.tensor(
                    labels[key][:num_dims], dtype=torch.float32
                ).unsqueeze(0)
                all_preds.append(pred)
                all_targets.append(target)
            except Exception:
                continue

        if all_preds:
            results["r2"] = suite.regression_r2(
                torch.cat(all_preds), torch.cat(all_targets)
            )

    return results


def aggregate_folds(fold_results: list[dict]) -> dict:
    """Compute mean and std across folds for all numeric metrics.

    Args:
        fold_results: List of per-fold result dicts.

    Returns:
        Dict with "{metric}_mean" and "{metric}_std" for each numeric key.
    """
    all_keys: set[str] = set()
    for fr in fold_results:
        for k, v in fr.items():
            if isinstance(v, (int, float)):
                all_keys.add(k)

    result = {}
    for key in sorted(all_keys):
        values = [
            fr[key]
            for fr in fold_results
            if key in fr and isinstance(fr[key], (int, float))
        ]
        if values:
            result[f"{key}_mean"] = float(np.mean(values))
            result[f"{key}_std"] = float(np.std(values))
    return result


def run_robustness_eval(
    model: torch.nn.Module,
    val_keys: list[str],
    get_input_fn: Callable,
    predict_fn: Callable,
    noise_std: float = 0.05,
) -> dict:
    """Evaluate robustness via clean vs noisy predictions.

    Adds Gaussian noise to inputs as a proxy when real augmented data
    is unavailable.
    """
    clean_scores, aug_scores = [], []
    model.eval()

    with torch.no_grad():
        for key in val_keys:
            try:
                inp, mask = get_input_fn(key)
                pred_clean = predict_fn(model, inp, mask)
                clean_scores.append(pred_clean)

                inp_aug = inp + torch.randn_like(inp) * noise_std
                pred_aug = predict_fn(model, inp_aug, mask)
                aug_scores.append(pred_aug)
            except Exception:
                continue

    if not clean_scores:
        return {"pearson_r": 0.0, "score_drop_pct": 100.0}

    return compute_robustness_metrics(
        torch.cat(clean_scores), torch.cat(aug_scores)
    )


def select_winner(
    results: dict[str, dict],
    veto_threshold: float = ROBUSTNESS_VETO_THRESHOLD,
) -> str | None:
    """Select best model: primary pairwise, tiebreak R2, veto robustness.

    Args:
        results: {model_name: {pairwise_mean, r2_mean, score_drop_pct}}.
        veto_threshold: Max allowed score_drop_pct.

    Returns:
        Winning model name, or None if all vetoed.
    """
    candidates = []
    for name, metrics in results.items():
        drop = metrics.get("score_drop_pct", 0.0)
        if drop > veto_threshold:
            continue
        candidates.append((
            name,
            metrics.get("pairwise_mean", 0.0),
            metrics.get("r2_mean", 0.0),
        ))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return candidates[0][0]
```

**Step 4: Run tests**

Run: `cd /Users/jdhiman/Documents/crescendai/model && python -m pytest tests/model_improvement/test_evaluation.py -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add model/src/model_improvement/evaluation.py model/tests/model_improvement/test_evaluation.py
git commit -m "feat(model): add shared evaluation module for comparison notebooks"
```

---

## Task 5: Update `01_audio_training.ipynb`

**Files:**
- Modify: `model/notebooks/model_improvement/01_audio_training.ipynb`

**Step 1: Update imports cell (cell-4)**

Add after existing imports:
```python
from model_improvement.taxonomy import load_composite_labels, NUM_DIMS, DIMENSIONS
```

**Step 2: Update data loading cell (cell-6)**

Replace the labels loading block (the `with open(cache_dir / 'labels.json')` block) with:
```python
# Load 6-dim composite labels (teacher-grounded taxonomy)
composite_path = DATA_DIR / 'composite_labels' / 'composite_labels.json'
labels_raw = load_composite_labels(composite_path)
labels = {k: v.tolist() for k, v in labels_raw.items()}
print(f'Loaded {len(labels)} composite labels ({NUM_DIMS} dims: {DIMENSIONS})')
```

Keep the embeddings loading code unchanged.

**Step 3: Update model configs**

In cell-12 (A1_CONFIG): change `'num_labels': 19` to `'num_labels': NUM_DIMS`
In cell-15 (A2_CONFIG): change `'num_labels': 19` to `'num_labels': NUM_DIMS`
In cell-17 (A3_CONFIG): change `'num_labels': 19` to `'num_labels': NUM_DIMS`

**Step 4: Add composite_labels rclone sync (cell-3)**

After the existing rclone sync lines for percepiano/competition/maestro, add:
```python
!rclone copy gdrive:crescendai_data/model_improvement/data/composite_labels ../data/composite_labels --progress 2>/dev/null || echo 'composite_labels: using local copy'
```

**Step 5: Verify notebook is valid JSON**

Run: `cd /Users/jdhiman/Documents/crescendai/model && python -c "import json; json.load(open('notebooks/model_improvement/01_audio_training.ipynb'))"`
Expected: No error

**Step 6: Commit**

```bash
git add model/notebooks/model_improvement/01_audio_training.ipynb
git commit -m "feat(model): update audio training notebook to use 6-dim composite labels"
```

---

## Task 6: Update `03_audio_comparison.ipynb`

**Files:**
- Modify: `model/notebooks/model_improvement/03_audio_comparison.ipynb`

**Step 1: Update imports (cell-4)**

Replace existing imports with:
```python
from model_improvement.audio_encoders import MuQLoRAModel, MuQStagedModel, MuQFullUnfreezeModel
from model_improvement.metrics import MetricsSuite, compute_robustness_metrics, format_comparison_table, stop_auc, competition_spearman, per_dimension_breakdown
from model_improvement.evaluation import aggregate_folds, run_robustness_eval, select_winner, ROBUSTNESS_VETO_THRESHOLD
from model_improvement.taxonomy import load_composite_labels, NUM_DIMS, DIMENSIONS
```

**Step 2: Update data loading (cell-6)**

Replace `labels.json` loading with:
```python
composite_path = DATA_DIR / 'composite_labels' / 'composite_labels.json'
labels_raw = load_composite_labels(composite_path)
labels = {k: v.tolist() for k, v in labels_raw.items()}
print(f'Loaded {len(labels)} composite labels ({NUM_DIMS} dims: {DIMENSIONS})')
```

**Step 3: Update evaluate_model_on_fold (cell-11)**

Change all `[:19]` label slicing to `[:NUM_DIMS]` in `build_val_pairs` and `evaluate_model_on_fold`.

Specifically in `build_val_pairs`:
```python
lab_a = torch.tensor(labels_dict[key_a][:NUM_DIMS], dtype=torch.float32)
lab_b = torch.tensor(labels_dict[key_b][:NUM_DIMS], dtype=torch.float32)
```

And in the regression section:
```python
target = torch.tensor(labels_dict[key][:NUM_DIMS], dtype=torch.float32).unsqueeze(0)
```

**Step 4: Update per-dimension names (cell-19)**

Replace the 19-element `DIMENSION_NAMES` list with:
```python
DIMENSION_NAMES = DIMENSIONS
```

In `plot_per_dimension_comparison`, change `n_dims = 19` to `n_dims = NUM_DIMS`.

**Step 5: Replace local select_winner (cell-21)**

Replace the local `select_winner` function and call with:
```python
if comparison:
    selection_input = {}
    for name, metrics in comparison.items():
        selection_input[name] = {
            'pairwise_mean': metrics.get('pairwise', 0.0),
            'r2_mean': metrics.get('r2', 0.0),
            'score_drop_pct': metrics.get('score_drop_pct', 0.0),
        }
    winner = select_winner(selection_input)
    print(f'\nWINNER: {winner}')
    for name, metrics in comparison.items():
        marker = ' <-- WINNER' if name == winner else ''
        print(f'  {name}: pairwise={metrics.get("pairwise", 0):.4f}, r2={metrics.get("r2", 0):.4f}{marker}')
else:
    print('Run training and evaluation first.')
```

**Step 6: Fix upload_checkpoint rclone path (cell-22)**

Change `gdrive:crescendai/model/checkpoints/model_improvement/` to `gdrive:crescendai_data/model_improvement/checkpoints/`. Or remove cell-22 entirely since `upload_checkpoint` is already imported from `model_improvement.training`.

**Step 7: Verify and commit**

Run: `cd /Users/jdhiman/Documents/crescendai/model && python -c "import json; json.load(open('notebooks/model_improvement/03_audio_comparison.ipynb'))"`

```bash
git add model/notebooks/model_improvement/03_audio_comparison.ipynb
git commit -m "feat(model): update audio comparison notebook with 6-dim eval and shared evaluation module"
```

---

## Task 7: Update `02_symbolic_training.ipynb`

**Files:**
- Modify: `model/notebooks/model_improvement/02_symbolic_training.ipynb`

**Step 1: Update imports (cell-4)**

Add after existing imports:
```python
from model_improvement.taxonomy import load_composite_labels, NUM_DIMS, DIMENSIONS
```

**Step 2: Update data loading (cell-6)**

Replace `labels.json` loading with composite labels:
```python
composite_path = DATA_DIR / 'composite_labels' / 'composite_labels.json'
labels_raw = load_composite_labels(composite_path)
labels = {k: v.tolist() for k, v in labels_raw.items()}
print(f'Loaded {len(labels)} composite labels ({NUM_DIMS} dims: {DIMENSIONS})')
```

Keep `folds.json` and `piece_mapping.json` loading unchanged.

**Step 3: Update model configs**

- Cell-14 (S1_CONFIG): `'num_labels': 19` -> `'num_labels': NUM_DIMS`
- Cell-16 (S2_CONFIG): `'num_labels': 19` -> `'num_labels': NUM_DIMS`
- Cell-18 (S2H_CONFIG): `'num_labels': 19` -> `'num_labels': NUM_DIMS`
- Cell-20 (S3_CONFIG): `'num_labels': 19` -> `'num_labels': NUM_DIMS`

**Step 4: Verify and commit**

Run: `cd /Users/jdhiman/Documents/crescendai/model && python -c "import json; json.load(open('notebooks/model_improvement/02_symbolic_training.ipynb'))"`

```bash
git add model/notebooks/model_improvement/02_symbolic_training.ipynb
git commit -m "feat(model): update symbolic training notebook to use 6-dim composite labels"
```

---

## Task 8: Update `04_symbolic_comparison.ipynb`

**Files:**
- Modify: `model/notebooks/model_improvement/04_symbolic_comparison.ipynb`

**Step 1: Update imports (cell-4)**

Replace imports with:
```python
from model_improvement.symbolic_encoders import (
    TransformerSymbolicEncoder,
    GNNSymbolicEncoder,
    GNNHeteroSymbolicEncoder,
    ContinuousSymbolicEncoder,
)
from model_improvement.tokenizer import PianoTokenizer, extract_continuous_features
from model_improvement.metrics import MetricsSuite, compute_robustness_metrics, format_comparison_table, stop_auc, competition_spearman, per_dimension_breakdown
from model_improvement.evaluation import aggregate_folds, run_robustness_eval, select_winner, ROBUSTNESS_VETO_THRESHOLD
from model_improvement.taxonomy import load_composite_labels, NUM_DIMS, DIMENSIONS
```

Note: Add `GNNHeteroSymbolicEncoder` import (missing from current notebook).

**Step 2: Update data loading (cell-6)**

Same composite labels pattern as other notebooks.

**Step 3: Update model loading (cell-13)**

Add S2H to the model loading loop:
```python
for name, cls, kwargs in [
    ('S1', TransformerSymbolicEncoder, {'stage': 'finetune'}),
    ('S2', GNNSymbolicEncoder, {'stage': 'finetune'}),
    ('S2H', GNNHeteroSymbolicEncoder, {'stage': 'finetune'}),
    ('S3', ContinuousSymbolicEncoder, {'stage': 'finetune'}),
]:
```

**Step 4: Update evaluate_s1/evaluate_s3 label slicing (cell-15)**

Change all `[:19]` to `[:NUM_DIMS]` in both functions.

**Step 5: Add evaluate_s2 function (cell-15)**

After `evaluate_s3`, add:
```python
def evaluate_s2(model, val_keys, labels_dict, graphs_dict):
    results = {}
    model.eval()
    valid_keys = [k for k in val_keys if k in graphs_dict and k in labels_dict]
    suite = MetricsSuite(ambiguous_threshold=0.05)

    with torch.no_grad():
        all_logits, all_la, all_lb = [], [], []
        for i, key_a in enumerate(valid_keys):
            for key_b in valid_keys[i + 1:]:
                try:
                    ga, gb = graphs_dict[key_a], graphs_dict[key_b]
                    z_a = model.encode_graph(ga.x, ga.edge_index, torch.zeros(ga.x.size(0), dtype=torch.long))
                    z_b = model.encode_graph(gb.x, gb.edge_index, torch.zeros(gb.x.size(0), dtype=torch.long))
                    logits = model.compare(z_a, z_b)
                    lab_a = torch.tensor(labels_dict[key_a][:NUM_DIMS], dtype=torch.float32)
                    lab_b = torch.tensor(labels_dict[key_b][:NUM_DIMS], dtype=torch.float32)
                    all_logits.append(logits)
                    all_la.append(lab_a.unsqueeze(0))
                    all_lb.append(lab_b.unsqueeze(0))
                except Exception:
                    continue

        if all_logits:
            pw = suite.pairwise_accuracy(torch.cat(all_logits), torch.cat(all_la), torch.cat(all_lb))
            results['pairwise'] = pw['overall']
            results['pairwise_detail'] = pw
    return results
```

Note: Check the actual GNN encoder API for `encode_graph` and `compare` methods. If the API differs, adapt accordingly. The GNNSymbolicEncoder uses `_pool_graph(x, edge_index, batch)` internally -- check forward signature.

**Step 6: Update evaluation loop (cell-16)**

Replace the `S2` skip with actual evaluation:
```python
elif name == 'S2':
    fold_res = evaluate_s2(model, val_keys, labels, score_graphs)
elif name == 'S2H':
    fold_res = evaluate_s2(model, val_keys, labels, hetero_graphs)
```

This requires loading score graphs. Add graph loading after cell-7 (MIDI paths):
```python
# Load pre-computed score graphs for S2 evaluation
pretrain_dir = DATA_DIR / 'pretrain_cache'
all_graphs_path = pretrain_dir / 'graphs' / 'all_graphs.pt'
if all_graphs_path.exists():
    all_graphs = torch.load(all_graphs_path, map_location='cpu', weights_only=False)
    score_graphs = {k.replace('percepiano__', ''): v for k, v in all_graphs.items() if k.startswith('percepiano__')}
    print(f'Loaded {len(score_graphs)} score graphs for S2 evaluation')
else:
    score_graphs = {}
    print('Score graphs not found -- S2 evaluation will be skipped')

all_hetero_path = pretrain_dir / 'graphs' / 'all_hetero_graphs.pt'
if all_hetero_path.exists():
    all_hetero = torch.load(all_hetero_path, map_location='cpu', weights_only=False)
    hetero_graphs = {k.replace('percepiano__', ''): v for k, v in all_hetero.items() if k.startswith('percepiano__')}
    print(f'Loaded {len(hetero_graphs)} hetero graphs for S2H evaluation')
else:
    hetero_graphs = {}
    print('Hetero graphs not found -- S2H evaluation will be skipped')
```

**Step 7: Add robustness evaluation**

Add a new cell between current cells 16 and 17 with robustness evaluation for S3 (Gaussian noise on features) and placeholder for S1/S2/S2H:

```python
robustness_results = {}
if folds and models:
    val_keys = folds[-1]['val']
    for name, model in models.items():
        print(f'Robustness check for {name}...')
        model.eval()
        clean, aug = [], []
        with torch.no_grad():
            for key in val_keys:
                try:
                    if name == 'S1' and key in s1_tokens:
                        ids = torch.tensor(s1_tokens[key]).unsqueeze(0)
                        mask = torch.ones(1, ids.size(1), dtype=torch.bool)
                        pred_c = model(ids, mask)['scores']
                        # Add noise to embedding layer output is not feasible;
                        # use token dropout as proxy: mask 5% of tokens
                        noisy_ids = ids.clone()
                        drop_mask = torch.rand_like(ids.float()) < 0.05
                        noisy_ids[drop_mask] = 0
                        pred_a = model(noisy_ids, mask)['scores']
                    elif name == 'S3' and key in s3_features:
                        feat = s3_features[key].unsqueeze(0)
                        mask = torch.ones(1, feat.size(1), dtype=torch.bool)
                        pred_c = model(feat, mask)['scores']
                        feat_aug = feat + torch.randn_like(feat) * 0.05
                        pred_a = model(feat_aug, mask)['scores']
                    else:
                        continue
                    clean.append(pred_c)
                    aug.append(pred_a)
                except Exception:
                    continue
        if clean:
            rob = compute_robustness_metrics(torch.cat(clean), torch.cat(aug))
        else:
            rob = {"pearson_r": 1.0, "score_drop_pct": 0.0}
        robustness_results[name] = rob
        print(f'  pearson_r={rob["pearson_r"]:.4f}, score_drop_pct={rob["score_drop_pct"]:.1f}%')
```

**Step 8: Update dimension names (cell-22)**

Replace `DIMENSION_NAMES` (19-element list) with `DIMENSIONS`, change `n_dims = 19` to `n_dims = NUM_DIMS`.

**Step 9: Update comparison table (cell-21)**

Add robustness data to comparison dict:
```python
if name in robustness_results:
    comparison[name]['robustness'] = robustness_results[name]['pearson_r']
    comparison[name]['score_drop_pct'] = robustness_results[name]['score_drop_pct']
```

**Step 10: Replace select_winner (cell-24)**

Same pattern as audio comparison: use imported `select_winner`.

**Step 11: Fix upload_checkpoint rclone path (cell-25)**

Same fix: `gdrive:crescendai_data/model_improvement/checkpoints/` or remove duplicate.

**Step 12: Verify and commit**

```bash
git add model/notebooks/model_improvement/04_symbolic_comparison.ipynb
git commit -m "feat(model): update symbolic comparison with full eval, S2/S2H support, robustness"
```

---

## Task 9: Surgical doc updates

**Files:**
- Modify: `docs/03-model-improvement.md`
- Modify: `docs/01-data-collection.md`
- Modify: `docs/02-teacher-grounded-taxonomy.md`

**Step 1: Update `docs/03-model-improvement.md`**

After line 15 (the "Repo cleanup..." prerequisite), insert:

```
> **Status (2026-02-23):** Taxonomy COMPLETE (all 5 gates pass, 6 dimensions). Distillation pilot NO-GO (0/6 dims pass r > 0.5). Data collection COMPLETE (T1-T3 in GDrive, T4 deferred). Training proceeds on baseline path with composite labels only.
```

At line 219, change:
```
### Round 0: Distillation A/B (conditional, only if pilot passed)
```
To:
```
### Round 0: Distillation A/B -- SKIPPED

> **Decision (2026-02-23):** Distillation pilot returned NO-GO (per-dimension Pearson r all near zero, 0/6 pass r > 0.5). L_regression uses T1 composite labels only (1,202 segments, 6 dims). T2/T3/T4 contribute via ranking/contrastive/invariance without regression labels. Round 1 is the effective starting point.
```

After line 345 ("All model configs change..."), add:
```
> **Resolved:** N = 6. Labels loaded from `composite_labels/composite_labels.json`. Default in all encoders updated via `taxonomy.NUM_DIMS`.
```

**Step 2: Update `docs/01-data-collection.md`**

After line 3 ("Complete inventory of data..."), insert:
```

> **Status (2026-02-23):** T1 COMPLETE (4.4 GB), T2 COMPLETE (3.3 GB), T3 COMPLETE (34.2 GB), T4 DEFERRED. Composite labels COMPLETE (`model/data/composite_labels/`). All data cached in `gdrive:crescendai_data/model_improvement/data/`.
```

**Step 3: Update `docs/02-teacher-grounded-taxonomy.md`**

After line 1 ("# Teacher-Grounded Feedback Taxonomy Design"), insert:
```

> **Status (2026-02-23):** COMPLETE. All deliverables produced: 6 dimensions, composite labels (1,202 segments), quote bank (60 quotes), validation report (all 5 gates PASS). Distillation pilot NO-GO. Outputs in `model/data/composite_labels/`.
```

**Step 4: Commit**

```bash
git add docs/01-data-collection.md docs/02-teacher-grounded-taxonomy.md docs/03-model-improvement.md
git commit -m "docs: add status annotations for taxonomy completion and distillation NO-GO"
```

---

## Task 10: Add verification cell to `00_data_collection.ipynb`

**Files:**
- Modify: `model/notebooks/model_improvement/00_data_collection.ipynb`

**Step 1: Add verification cell**

Insert a new code cell after the setup cells (after cell-3, before the T2 section). The cell runs `rclone size --json` for each cache and prints a status table:

```python
import subprocess as _sp
import json as _json

GDRIVE_BASE = 'gdrive:crescendai_data/model_improvement/data'
EXPECTED = {
    'percepiano_cache': 1200,
    'competition_cache': 2000,
    'maestro_cache':     20000,
    'pretrain_cache':    100,
    'asap_cache':        2000,
    'percepiano_midi':   1200,
}

print(f'{"Cache":<22} {"Status":<10} {"Objects":>10} {"Size":>12}')
print('-' * 58)

for name, min_objects in EXPECTED.items():
    try:
        result = _sp.run(
            ['rclone', 'size', f'{GDRIVE_BASE}/{name}/', '--json'],
            capture_output=True, text=True, timeout=30,
        )
        info = _json.loads(result.stdout)
        count = info.get('count', 0)
        size_gb = info.get('bytes', 0) / (1024**3)
        status = 'OK' if count >= min_objects else 'LOW'
        print(f'{name:<22} {status:<10} {count:>10,} {size_gb:>10.2f} GB')
    except Exception as e:
        print(f'{name:<22} {"ERROR":<10} {"--":>10} {"--":>12}  ({e})')
```

**Step 2: Verify and commit**

```bash
git add model/notebooks/model_improvement/00_data_collection.ipynb
git commit -m "feat(model): add GDrive verification cell to data collection notebook"
```

---

## Task 11: Run full test suite and final verification

**Step 1: Run all model_improvement tests**

Run: `cd /Users/jdhiman/Documents/crescendai/model && python -m pytest tests/model_improvement/ -v --tb=short`

Expected: All tests pass. If any break due to the `num_labels` default change, fix by passing `num_labels=19` explicitly in those test constructors where tests assume 19-dim behavior.

**Step 2: Verify all notebooks are valid JSON**

Run:
```bash
cd /Users/jdhiman/Documents/crescendai/model
for nb in notebooks/model_improvement/*.ipynb; do
    python -c "import json; json.load(open('$nb'))" && echo "OK: $nb" || echo "FAIL: $nb"
done
```

**Step 3: Final commit if any fixups needed**

```bash
git add -u
git commit -m "fix(model): test and notebook fixups for 6-dim label migration"
```
