# Layer 1 Validation Experiments -- Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement four validation experiments that test whether the trained audio/symbolic models generalize beyond PercePiano, survive AMT transcription, differentiate intermediate players, and improve teacher feedback with MIDI context.

**Architecture:** Single notebook (`04_layer1_validation.ipynb`) for Experiments 1-3 (model assessment), two new Python modules (`feedback_assessment.py`, `midi_comparison.py`) for Experiment 4. All code follows the existing pattern: notebooks are minimal orchestration, logic lives in `src/model_improvement/`. All experiments run locally on M4.

**Tech Stack:** PyTorch, PyTorch Lightning, pretty_midi, scipy.stats, MuQ extractor, A1/S2 checkpoints, yt-dlp, matplotlib/seaborn.

---

## Data Inventory (verified 2026-03-09)

**Local (`model/data/`):**

- `checkpoints/model_improvement/A1/fold_3/epoch=5-val_loss=0.5450.ckpt` -- A1 best checkpoint
- `checkpoints/model_improvement/S2/fold_3/` -- S2 best checkpoint
- `maestro_cache/` -- 1,276 ground-truth MIDI files + `contrastive_mapping.json` + `maestro-v3.0.0.json`
- `percepiano_cache/` -- `muq_embeddings.pt`, `labels.json`, `piece_mapping.json`, `folds.json`, `_muq_file_cache/`
- `composite_labels/` -- `composite_labels.json`, `dimension_definitions.json`, `quote_bank.json`

**GDrive (`gdrive:crescendai_data/model_improvement/data/`):**

- `maestro_cache/muq_embeddings/` -- 24,321 pre-extracted segment embeddings (.pt)
- `competition_cache/chopin2021/muq_embeddings/` -- 2,293 segment embeddings (.pt)
- `competition_cache/chopin2021/metadata.jsonl` -- segment-level metadata
- `competition_cache/chopin2021/recordings.jsonl` -- recording-level metadata
- `percepiano_cache/_muq_file_cache/` -- per-file MuQ embeddings

**Not yet available (must be created):**

- Competition audio files (download via yt-dlp; ~2,293 segments already embedded on GDrive)
- Intermediate pianist audio (Experiment 3; download from YouTube)
- AMT transcriptions (Experiment 2; run YourMT3+ and ByteDance locally)

---

## Task 1: Sync Competition Data from GDrive

**Files:**

- Download: `data/competition_cache/chopin2021/metadata.jsonl`
- Download: `data/competition_cache/chopin2021/recordings.jsonl`
- Download: `data/competition_cache/chopin2021/muq_embeddings/*.pt` (2,293 files)

**Step 1: Create local competition_cache directory and sync**

```bash
cd model
mkdir -p data/competition_cache/chopin2021
rclone copy gdrive:crescendai_data/model_improvement/data/competition_cache/chopin2021/metadata.jsonl data/competition_cache/chopin2021/ -P
rclone copy gdrive:crescendai_data/model_improvement/data/competition_cache/chopin2021/recordings.jsonl data/competition_cache/chopin2021/ -P
rclone copy gdrive:crescendai_data/model_improvement/data/competition_cache/chopin2021/muq_embeddings/ data/competition_cache/chopin2021/muq_embeddings/ -P
```

**Step 2: Verify file counts**

```bash
wc -l data/competition_cache/chopin2021/metadata.jsonl
# Expected: ~2293 lines
ls data/competition_cache/chopin2021/muq_embeddings/*.pt | wc -l
# Expected: 2293
```

**Step 3: Commit**

```bash
git add -n data/competition_cache/  # dry-run to verify .gitignore handles it
git status
# Expected: competition_cache/ is .gitignored (data files should NOT be committed)
```

---

## Task 2: Experiment 1 -- Competition Correlation Scoring Function

**Files:**

- Create: `model/src/model_improvement/layer1_validation.py`
- Test: `model/tests/test_layer1_validation.py`

This task builds the scoring and correlation logic. The notebook orchestration comes in Task 5.

**Step 1: Write the failing test for `score_competition_segments`**

```python
# model/tests/test_layer1_validation.py
"""Tests for Layer 1 validation experiment helpers."""

import numpy as np
import pytest
import torch

from model_improvement.layer1_validation import (
    score_competition_segments,
    competition_correlation,
)


def test_score_competition_segments_returns_dict():
    """score_competition_segments returns {segment_id: scores_array}."""

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1024, 6)

        def predict_scores(self, x, mask=None):
            # Return fixed scores for deterministic test
            return torch.tensor([[0.5, 0.6, 0.7, 0.4, 0.3, 0.8]])

    model = FakeModel()
    embeddings = {
        "seg_001": torch.randn(10, 1024),
        "seg_002": torch.randn(15, 1024),
    }
    result = score_competition_segments(model, embeddings)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"seg_001", "seg_002"}
    for scores in result.values():
        assert isinstance(scores, np.ndarray)
        assert scores.shape == (6,)


def test_competition_correlation_spearman():
    """competition_correlation computes Spearman rho per aggregation method."""
    segment_scores = {
        "perf1_seg000": np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.9]),
        "perf1_seg001": np.array([0.7, 0.8, 0.5, 0.6, 0.5, 0.8]),
        "perf2_seg000": np.array([0.3, 0.4, 0.3, 0.2, 0.3, 0.4]),
        "perf2_seg001": np.array([0.4, 0.3, 0.4, 0.3, 0.2, 0.3]),
    }
    metadata = [
        {"segment_id": "perf1_seg000", "performer": "Alice", "placement": 1, "round": "final"},
        {"segment_id": "perf1_seg001", "performer": "Alice", "placement": 1, "round": "final"},
        {"segment_id": "perf2_seg000", "performer": "Bob", "placement": 5, "round": "stage2"},
        {"segment_id": "perf2_seg001", "performer": "Bob", "placement": 5, "round": "stage2"},
    ]
    result = competition_correlation(segment_scores, metadata)
    assert "mean" in result
    assert "median" in result
    assert "min" in result
    for agg_name, corr in result.items():
        assert "rho" in corr
        assert "p_value" in corr
        assert "per_dimension" in corr
        assert isinstance(corr["per_dimension"], dict)
        assert len(corr["per_dimension"]) == 6
```

**Step 2: Run test to verify it fails**

Run: `cd model && uv run pytest tests/test_layer1_validation.py -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError`

**Step 3: Write minimal implementation**

```python
# model/src/model_improvement/layer1_validation.py
"""Layer 1 validation experiment helpers.

Experiment 1: Competition correlation (does A1 quality signal predict placement?)
Experiment 2: AMT degradation (does S2 survive transcribed MIDI?)
Experiment 3: Dynamic range (does A1 differentiate intermediate players?)
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import torch
from scipy import stats

from model_improvement.taxonomy import DIMENSIONS

logger = logging.getLogger(__name__)


def score_competition_segments(
    model: torch.nn.Module,
    embeddings: dict[str, torch.Tensor],
    max_frames: int = 1000,
) -> dict[str, np.ndarray]:
    """Score competition segments using A1 model's predict_scores.

    Args:
        model: A1 model with predict_scores(x, mask) -> [1, 6].
        embeddings: {segment_id: tensor [T, 1024]}.
        max_frames: Truncate embeddings longer than this.

    Returns:
        {segment_id: np.ndarray [6]} with per-dimension scores.
    """
    model.eval()
    device = next(model.parameters()).device
    results = {}

    with torch.no_grad():
        for seg_id, emb in embeddings.items():
            if emb.shape[0] > max_frames:
                emb = emb[:max_frames]
            x = emb.unsqueeze(0).to(device)  # [1, T, 1024]
            mask = torch.ones(1, x.shape[1], dtype=torch.bool, device=device)
            scores = model.predict_scores(x, mask)  # [1, 6]
            results[seg_id] = scores.squeeze(0).cpu().numpy()

    return results


def competition_correlation(
    segment_scores: dict[str, np.ndarray],
    metadata: list[dict],
    aggregations: tuple[str, ...] = ("mean", "median", "min"),
) -> dict[str, dict]:
    """Compute Spearman rho of aggregated scores vs competition placement.

    Groups segments by performer, aggregates per-dimension scores using each
    method, then correlates the mean-across-dimensions aggregate with placement.
    Also computes per-dimension correlations.

    Args:
        segment_scores: {segment_id: np.ndarray [6]}.
        metadata: List of dicts with segment_id, performer, placement.
        aggregations: Aggregation methods to try.

    Returns:
        {agg_name: {rho, p_value, per_dimension: {dim_name: {rho, p_value}}}}.
    """
    # Group scores by performer
    performer_segments: dict[str, list[np.ndarray]] = defaultdict(list)
    performer_placement: dict[str, int] = {}

    for meta in metadata:
        seg_id = meta["segment_id"]
        if seg_id not in segment_scores:
            continue
        performer = meta["performer"]
        performer_segments[performer].append(segment_scores[seg_id])
        performer_placement[performer] = meta["placement"]

    agg_fns = {
        "mean": lambda arrs: np.mean(arrs, axis=0),
        "median": lambda arrs: np.median(arrs, axis=0),
        "min": lambda arrs: np.min(arrs, axis=0),
    }

    results = {}
    performers = sorted(performer_segments.keys())
    placements = np.array([performer_placement[p] for p in performers])

    for agg_name in aggregations:
        fn = agg_fns[agg_name]
        agg_scores = np.array([fn(performer_segments[p]) for p in performers])  # [P, 6]

        # Overall: mean across dimensions vs placement
        overall = agg_scores.mean(axis=1)
        # Negate placement because lower placement = better, higher score = better
        rho, p_value = stats.spearmanr(overall, -placements)

        # Per-dimension
        per_dim = {}
        for d, dim_name in enumerate(DIMENSIONS):
            dim_rho, dim_p = stats.spearmanr(agg_scores[:, d], -placements)
            per_dim[dim_name] = {"rho": float(dim_rho), "p_value": float(dim_p)}

        results[agg_name] = {
            "rho": float(rho),
            "p_value": float(p_value),
            "per_dimension": per_dim,
            "n_performers": len(performers),
        }

    return results
```

**Step 4: Run test to verify it passes**

Run: `cd model && uv run pytest tests/test_layer1_validation.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add model/src/model_improvement/layer1_validation.py model/tests/test_layer1_validation.py
git commit -m "feat(model): add competition correlation scoring for Layer 1 Experiment 1"
```

---

## Task 3: Experiment 2 -- AMT Degradation Helpers

**Files:**

- Modify: `model/src/model_improvement/layer1_validation.py`
- Modify: `model/tests/test_layer1_validation.py`

**Step 1: Write the failing test for `amt_degradation_comparison`**

Add to `model/tests/test_layer1_validation.py`:

```python
from model_improvement.layer1_validation import amt_degradation_comparison


def test_amt_degradation_comparison():
    """amt_degradation_comparison returns per-dimension pairwise accuracy drop."""
    # Simulate: 3 sources (gt, amt_a, amt_b) with pairwise results
    pairwise_results = {
        "ground_truth": {
            "overall": 0.72,
            "per_dimension": {0: 0.77, 1: 0.65, 2: 0.72, 3: 0.70, 4: 0.63, 5: 0.77},
        },
        "yourmt3": {
            "overall": 0.68,
            "per_dimension": {0: 0.73, 1: 0.62, 2: 0.68, 3: 0.65, 4: 0.60, 5: 0.72},
        },
        "bytedance": {
            "overall": 0.60,
            "per_dimension": {0: 0.65, 1: 0.55, 2: 0.60, 3: 0.58, 4: 0.52, 5: 0.64},
        },
    }
    result = amt_degradation_comparison(pairwise_results, baseline="ground_truth")
    assert "yourmt3" in result
    assert "bytedance" in result
    for source, drops in result.items():
        assert "overall_drop_pct" in drops
        assert "per_dimension_drop_pct" in drops
        assert len(drops["per_dimension_drop_pct"]) == 6
        # All drops should be non-negative (AMT should be worse)
        assert drops["overall_drop_pct"] >= 0


def test_select_maestro_subset():
    """select_maestro_subset picks pieces with multiple performers."""
    from model_improvement.layer1_validation import select_maestro_subset

    contrastive_mapping = {
        "piece_a": ["perf1", "perf2", "perf3"],
        "piece_b": ["perf4", "perf5"],
        "piece_c": ["perf6"],  # single performer, should be excluded
    }
    result = select_maestro_subset(contrastive_mapping, n_recordings=4)
    assert len(result) == 4
    # All selected recordings should come from multi-performer pieces
    for key in result:
        found = False
        for piece, perfs in contrastive_mapping.items():
            if key in perfs and len(perfs) >= 2:
                found = True
                break
        assert found, f"{key} not from multi-performer piece"
```

**Step 2: Run test to verify it fails**

Run: `cd model && uv run pytest tests/test_layer1_validation.py::test_amt_degradation_comparison -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

Append to `model/src/model_improvement/layer1_validation.py`:

```python
def amt_degradation_comparison(
    pairwise_results: dict[str, dict],
    baseline: str = "ground_truth",
) -> dict[str, dict]:
    """Compute per-dimension pairwise accuracy drop relative to baseline.

    Args:
        pairwise_results: {source_name: {overall: float, per_dimension: {int: float}}}.
        baseline: Key in pairwise_results to use as reference.

    Returns:
        {source_name: {overall_drop_pct, per_dimension_drop_pct: {dim_name: float}}}.
        Drop is expressed as percentage points (baseline - source).
    """
    base = pairwise_results[baseline]
    results = {}

    for source, pw in pairwise_results.items():
        if source == baseline:
            continue
        overall_drop = (base["overall"] - pw["overall"]) * 100
        per_dim_drop = {}
        for d, dim_name in enumerate(DIMENSIONS):
            base_acc = base["per_dimension"][d]
            source_acc = pw["per_dimension"][d]
            per_dim_drop[dim_name] = round((base_acc - source_acc) * 100, 2)

        results[source] = {
            "overall_drop_pct": round(overall_drop, 2),
            "per_dimension_drop_pct": per_dim_drop,
            "viable": overall_drop < 10.0,  # < 10% drop = viable
        }

    return results


def select_maestro_subset(
    contrastive_mapping: dict[str, list[str]],
    n_recordings: int = 50,
) -> list[str]:
    """Select MAESTRO recordings from pieces with multiple performers.

    Prioritizes pieces with the most performers to maximize contrastive pairs.

    Args:
        contrastive_mapping: {piece_name: [recording_key, ...]}.
        n_recordings: Target number of recordings.

    Returns:
        List of recording keys.
    """
    # Filter to multi-performer pieces, sort by performer count descending
    multi = {
        piece: perfs
        for piece, perfs in contrastive_mapping.items()
        if len(perfs) >= 2
    }
    sorted_pieces = sorted(multi.keys(), key=lambda p: len(multi[p]), reverse=True)

    selected = []
    for piece in sorted_pieces:
        if len(selected) >= n_recordings:
            break
        for perf in multi[piece]:
            if len(selected) >= n_recordings:
                break
            selected.append(perf)

    return selected
```

**Step 4: Run test to verify it passes**

Run: `cd model && uv run pytest tests/test_layer1_validation.py -v`
Expected: PASS (all tests)

**Step 5: Commit**

```bash
git add model/src/model_improvement/layer1_validation.py model/tests/test_layer1_validation.py
git commit -m "feat(model): add AMT degradation comparison helpers for Experiment 2"
```

---

## Task 4: Experiment 3 -- Dynamic Range Analysis Helper

**Files:**

- Modify: `model/src/model_improvement/layer1_validation.py`
- Modify: `model/tests/test_layer1_validation.py`

**Step 1: Write the failing test**

Add to `model/tests/test_layer1_validation.py`:

```python
from model_improvement.layer1_validation import dynamic_range_analysis


def test_dynamic_range_analysis():
    """dynamic_range_analysis returns separation and variance stats."""
    scores_by_group = {
        "intermediate": {
            "player1_seg0": np.array([0.4, 0.5, 0.3, 0.4, 0.3, 0.4]),
            "player1_seg1": np.array([0.5, 0.4, 0.4, 0.5, 0.4, 0.3]),
            "player2_seg0": np.array([0.3, 0.3, 0.2, 0.3, 0.2, 0.3]),
        },
        "advanced": {
            "adv1_seg0": np.array([0.7, 0.8, 0.6, 0.7, 0.6, 0.8]),
            "adv1_seg1": np.array([0.8, 0.7, 0.7, 0.8, 0.7, 0.7]),
        },
    }
    result = dynamic_range_analysis(scores_by_group)
    assert "separation" in result
    assert "within_group_variance" in result
    assert "per_dimension" in result
    # Separation should be positive (advanced > intermediate)
    assert result["separation"]["overall"] > 0
    # Per-dimension should have 6 entries
    assert len(result["per_dimension"]) == 6
```

**Step 2: Run test to verify it fails**

Run: `cd model && uv run pytest tests/test_layer1_validation.py::test_dynamic_range_analysis -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Append to `model/src/model_improvement/layer1_validation.py`:

```python
def dynamic_range_analysis(
    scores_by_group: dict[str, dict[str, np.ndarray]],
) -> dict:
    """Analyze score distributions across skill groups.

    Args:
        scores_by_group: {group_name: {segment_id: np.ndarray [6]}}.

    Returns:
        Dict with separation metrics, within-group variance, per-dimension breakdown.
    """
    group_stats = {}
    for group, segments in scores_by_group.items():
        all_scores = np.array(list(segments.values()))  # [N, 6]
        group_stats[group] = {
            "mean": all_scores.mean(axis=0),      # [6]
            "std": all_scores.std(axis=0),         # [6]
            "overall_mean": float(all_scores.mean()),
            "overall_std": float(all_scores.std()),
            "n_segments": len(segments),
        }

    # Separation between groups (if exactly 2 groups, compute Cohen's d)
    groups = sorted(scores_by_group.keys())
    separation = {}
    if len(groups) >= 2:
        for i, g1 in enumerate(groups):
            for g2 in groups[i + 1:]:
                key = f"{g1}_vs_{g2}"
                diff = group_stats[g2]["overall_mean"] - group_stats[g1]["overall_mean"]
                pooled_std = np.sqrt(
                    (group_stats[g1]["overall_std"] ** 2 + group_stats[g2]["overall_std"] ** 2) / 2
                )
                cohens_d = diff / pooled_std if pooled_std > 0 else 0.0
                separation[key] = {"mean_diff": float(diff), "cohens_d": float(cohens_d)}
        separation["overall"] = float(
            group_stats[groups[-1]]["overall_mean"] - group_stats[groups[0]]["overall_mean"]
        )
    else:
        separation["overall"] = 0.0

    # Within-group variance
    within_var = {}
    for group in groups:
        within_var[group] = {
            "overall": float(group_stats[group]["overall_std"]),
            "per_dimension": {
                DIMENSIONS[d]: float(group_stats[group]["std"][d])
                for d in range(len(DIMENSIONS))
            },
        }

    # Per-dimension separation
    per_dim = {}
    if len(groups) >= 2:
        g1, g2 = groups[0], groups[-1]
        for d, dim_name in enumerate(DIMENSIONS):
            diff = float(group_stats[g2]["mean"][d] - group_stats[g1]["mean"][d])
            per_dim[dim_name] = {"mean_diff": diff}
    else:
        for dim_name in DIMENSIONS:
            per_dim[dim_name] = {"mean_diff": 0.0}

    return {
        "group_stats": group_stats,
        "separation": separation,
        "within_group_variance": within_var,
        "per_dimension": per_dim,
    }
```

**Step 4: Run test to verify it passes**

Run: `cd model && uv run pytest tests/test_layer1_validation.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add model/src/model_improvement/layer1_validation.py model/tests/test_layer1_validation.py
git commit -m "feat(model): add dynamic range analysis helper for Experiment 3"
```

---

## Task 5: Create the Notebook (Experiments 1-3)

**Files:**

- Create: `model/notebooks/model_improvement/04_layer1_validation.ipynb`

This notebook is orchestration only -- all logic lives in `src/`. Follow existing notebook conventions from `01_audio_training.ipynb`.

**Step 1: Create notebook with setup cell**

Cell 1 (markdown):

```markdown
# Layer 1 Validation Experiments

Validates four assumptions before investing in Layer 2 model improvements.
See `docs/plans/2026-03-09-layer1-validation-design.md` for the full design.

## Setup
```

Cell 2 (code -- setup):

```python
import sys
sys.path.insert(0, "../../src") if "../../src" not in sys.path else None

import json
import logging
from pathlib import Path

import jsonlines
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import pytorch_lightning as pl

pl.seed_everything(42, workers=True)
torch.set_float32_matmul_precision("medium")
logging.basicConfig(level=logging.INFO)

from model_improvement.audio_encoders import MuQLoRAModel
from model_improvement.taxonomy import DIMENSIONS, load_composite_labels
from model_improvement.layer1_validation import (
    score_competition_segments,
    competition_correlation,
    dynamic_range_analysis,
)

DATA_DIR = Path("../../data")
CHECKPOINT_DIR = DATA_DIR / "checkpoints/model_improvement"
COMPETITION_DIR = DATA_DIR / "competition_cache/chopin2021"
PERCEPIANO_DIR = DATA_DIR / "percepiano_cache"
```

**Step 2: Add Experiment 1 cells**

Cell 3 (markdown):

```markdown
## Experiment 1: Competition Correlation

**Question:** Does A1's quality signal predict expert competition placement?

**Data:** 2,293 segments from Chopin 2021 (already on GDrive, synced locally).

**Decision gate:** rho > 0.3 on at least one aggregation = model signal is real.
```

Cell 4 (code -- load model and data):

```python
# Load A1 fold 3 checkpoint
a1_ckpt = sorted(CHECKPOINT_DIR.glob("A1/fold_3/*.ckpt"))[0]
print(f"Loading A1 from {a1_ckpt.name}")

a1_model = MuQLoRAModel.load_from_checkpoint(
    str(a1_ckpt),
    use_pretrained_muq=False,  # embeddings are pre-extracted
)
a1_model.eval()
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
a1_model = a1_model.to(device)

# Load competition embeddings
emb_dir = COMPETITION_DIR / "muq_embeddings"
embeddings = {}
for pt_file in sorted(emb_dir.glob("*.pt")):
    embeddings[pt_file.stem] = torch.load(pt_file, map_location="cpu", weights_only=True)
print(f"Loaded {len(embeddings)} competition segment embeddings")

# Load metadata
with jsonlines.open(COMPETITION_DIR / "metadata.jsonl") as reader:
    metadata = list(reader)
print(f"Loaded {len(metadata)} metadata records")
print(f"Performers: {len(set(m['performer'] for m in metadata))}")
print(f"Rounds: {sorted(set(m['round'] for m in metadata))}")
```

Cell 5 (code -- score and correlate):

```python
# Score all segments
segment_scores = score_competition_segments(a1_model, embeddings)
print(f"Scored {len(segment_scores)} segments")

# Compute correlations
corr_results = competition_correlation(segment_scores, metadata)

# Display results
print("\n=== Competition Correlation Results ===\n")
for agg_name, corr in corr_results.items():
    gate = "PASS" if abs(corr["rho"]) > 0.3 else "INVESTIGATE" if abs(corr["rho"]) > 0.2 else "FAIL"
    print(f"{agg_name:>8s}: rho={corr['rho']:+.3f}  p={corr['p_value']:.4f}  n={corr['n_performers']}  [{gate}]")
    for dim_name, dim_corr in corr["per_dimension"].items():
        print(f"           {dim_name:>14s}: rho={dim_corr['rho']:+.3f}  p={dim_corr['p_value']:.4f}")
    print()
```

Cell 6 (code -- scatter plots):

```python
# Scatter plots: per-aggregation, overall score vs placement
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, agg_name in zip(axes, ["mean", "median", "min"]):
    # Rebuild per-performer aggregates for plotting
    from collections import defaultdict
    perf_segs = defaultdict(list)
    perf_place = {}
    for m in metadata:
        if m["segment_id"] in segment_scores:
            perf_segs[m["performer"]].append(segment_scores[m["segment_id"]])
            perf_place[m["performer"]] = m["placement"]

    agg_fn = {"mean": np.mean, "median": np.median, "min": np.min}[agg_name]
    performers = sorted(perf_segs.keys())
    agg_overall = [agg_fn(perf_segs[p], axis=0).mean() for p in performers]
    placements = [perf_place[p] for p in performers]

    ax.scatter(placements, agg_overall, alpha=0.7)
    ax.set_xlabel("Competition Placement (lower = better)")
    ax.set_ylabel(f"A1 Score ({agg_name})")
    ax.set_title(f"{agg_name}: rho={corr_results[agg_name]['rho']:+.3f}")
    ax.invert_xaxis()

plt.tight_layout()
plt.savefig(COMPETITION_DIR / "correlation_scatter.png", dpi=150)
plt.show()
```

Cell 7 (code -- per-dimension heatmap):

```python
# Per-dimension correlation heatmap
fig, ax = plt.subplots(figsize=(10, 4))
data = []
for agg_name in ["mean", "median", "min"]:
    row = [corr_results[agg_name]["per_dimension"][d]["rho"] for d in DIMENSIONS]
    data.append(row)

sns.heatmap(
    np.array(data), annot=True, fmt=".2f", cmap="RdYlGn",
    xticklabels=DIMENSIONS, yticklabels=["mean", "median", "min"],
    ax=ax, center=0, vmin=-0.6, vmax=0.6,
)
ax.set_title("Per-Dimension Spearman rho vs Competition Placement")
plt.tight_layout()
plt.savefig(COMPETITION_DIR / "correlation_heatmap.png", dpi=150)
plt.show()
```

**Step 3: Add Experiment 3 cells**

Cell 8 (markdown):

```markdown
## Experiment 3: Dynamic Range at Intermediate Level

**Question:** Does A1 produce meaningful score variance on intermediate-level recordings?

**Data:** YouTube recordings of intermediate pianists.
Must be collected first using yt-dlp (manual search for student recitals).

**No hard gate.** Diagnostic only -- informs Layer 3 data priorities.
```

Cell 9 (code -- intermediate data collection):

```python
# This cell handles downloading and embedding intermediate recordings.
# Run once, then skip on subsequent notebook runs.

from model_improvement.audio_utils import load_audio, segment_audio
from audio_experiments.extractors.muq import MuQExtractor

INTERMEDIATE_DIR = DATA_DIR / "intermediate_cache"
INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
INTERMEDIATE_AUDIO = INTERMEDIATE_DIR / "audio"
INTERMEDIATE_AUDIO.mkdir(parents=True, exist_ok=True)
INTERMEDIATE_EMB = INTERMEDIATE_DIR / "muq_embeddings"
INTERMEDIATE_EMB.mkdir(parents=True, exist_ok=True)

# YouTube URLs for intermediate performances
# Manually curated: student recitals, diploma exams, amateur performances
# of pieces in PercePiano/MAESTRO repertoire.
# Format: (url, performer_label, piece_label)
INTERMEDIATE_URLS = [
    # TODO: Fill with actual URLs after manual YouTube search
    # Example: ("https://www.youtube.com/watch?v=XXXXX", "student_01", "chopin_nocturne_op9_2"),
]

# Download, segment, and extract embeddings
# (Skip if already done)
if INTERMEDIATE_URLS and not list(INTERMEDIATE_EMB.glob("*.pt")):
    import subprocess

    for url, performer, piece in INTERMEDIATE_URLS:
        out_path = INTERMEDIATE_AUDIO / f"{performer}_{piece}.wav"
        if out_path.exists():
            continue
        subprocess.run([
            "yt-dlp", "--extract-audio", "--audio-format", "wav",
            "--postprocessor-args", "ffmpeg:-ar 24000 -ac 1",
            "--output", str(out_path.with_suffix(".%(ext)s")),
            "--no-playlist", "--quiet", url,
        ], timeout=300)

    extractor = MuQExtractor(cache_dir=INTERMEDIATE_EMB)
    for wav_file in sorted(INTERMEDIATE_AUDIO.glob("*.wav")):
        audio, sr = load_audio(wav_file, target_sr=24000)
        segments = segment_audio(audio, sr=sr, segment_duration=30.0)
        for i, seg in enumerate(segments):
            seg_id = f"{wav_file.stem}_seg{i:03d}"
            if (INTERMEDIATE_EMB / f"{seg_id}.pt").exists():
                continue
            audio_tensor = torch.from_numpy(seg["audio"]).float()
            embedding = extractor.extract_from_audio(audio_tensor)
            torch.save(embedding, INTERMEDIATE_EMB / f"{seg_id}.pt")
    del extractor
    print(f"Extracted {len(list(INTERMEDIATE_EMB.glob('*.pt')))} intermediate embeddings")
else:
    print(f"Intermediate embeddings: {len(list(INTERMEDIATE_EMB.glob('*.pt')))} files")
```

Cell 10 (code -- score intermediate and compare):

```python
# Score intermediate segments with A1
int_embeddings = {}
for pt_file in sorted(INTERMEDIATE_EMB.glob("*.pt")):
    int_embeddings[pt_file.stem] = torch.load(pt_file, map_location="cpu", weights_only=True)

if int_embeddings:
    int_scores = score_competition_segments(a1_model, int_embeddings)
else:
    int_scores = {}
    print("No intermediate data yet. Populate INTERMEDIATE_URLS and re-run cell 9.")

# Load PercePiano scores for comparison (advanced reference)
labels = load_composite_labels(DATA_DIR / "composite_labels/composite_labels.json")
percepiano_scores = {k: np.array(v[:6]) for k, v in labels.items()}

# Compare distributions
if int_scores:
    groups = {
        "intermediate": int_scores,
        "advanced (PercePiano)": percepiano_scores,
    }
    # Add competition data as "professional" tier
    groups["professional (Chopin 2021)"] = segment_scores

    range_results = dynamic_range_analysis(groups)

    print("\n=== Dynamic Range Analysis ===\n")
    for group, stats in range_results["group_stats"].items():
        print(f"{group}: mean={stats['overall_mean']:.3f}, std={stats['overall_std']:.3f}, n={stats['n_segments']}")
    print()
    for key, sep in range_results["separation"].items():
        if isinstance(sep, dict):
            print(f"  {key}: diff={sep['mean_diff']:+.3f}, Cohen's d={sep['cohens_d']:.2f}")
```

Cell 11 (code -- box plots):

```python
if int_scores:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for d, (ax, dim_name) in enumerate(zip(axes, DIMENSIONS)):
        plot_data = []
        group_labels = []
        for group_name, scores_dict in groups.items():
            vals = [s[d] for s in scores_dict.values()]
            plot_data.extend(vals)
            group_labels.extend([group_name] * len(vals))

        import pandas as pd
        df = pd.DataFrame({"Score": plot_data, "Group": group_labels})
        sns.boxplot(data=df, x="Group", y="Score", ax=ax)
        ax.set_title(dim_name)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")

    plt.suptitle("Per-Dimension Score Distributions by Skill Level")
    plt.tight_layout()
    plt.savefig(INTERMEDIATE_DIR / "dynamic_range_boxplots.png", dpi=150)
    plt.show()
else:
    print("Skipping box plots -- no intermediate data available yet.")
```

**Step 4: Add Experiment 2 placeholder cells**

Cell 12 (markdown):

```markdown
## Experiment 2: AMT Degradation Test

**Question:** How much does S2's pairwise accuracy drop with transcribed MIDI vs ground-truth?

**Data:** 50 MAESTRO recordings with ground-truth MIDI. Transcribed with YourMT3+ (ceiling) and ByteDance (floor).

**Decision gate:** Per-dimension pairwise drop < 10% = symbolic path viable.

**Note:** This experiment requires installing YourMT3+ and ByteDance Piano Transcription.
AMT transcription takes ~4-8 hours on M4. Run the transcription cells once,
then the assessment cells reuse cached transcriptions.
```

Cell 13 (code -- select MAESTRO subset):

```python
import json

with open(DATA_DIR / "maestro_cache/contrastive_mapping.json") as f:
    contrastive_mapping = json.load(f)

from model_improvement.layer1_validation import select_maestro_subset
selected_keys = select_maestro_subset(contrastive_mapping, n_recordings=50)
print(f"Selected {len(selected_keys)} recordings from {len(contrastive_mapping)} pieces")
print(f"Sample keys: {selected_keys[:3]}")
```

Cell 14 (code -- transcription with YourMT3+ and ByteDance):

```python
# AMT transcription -- run once, results are cached.
# Requires: uv pip install piano-transcription-inference
# YourMT3+ must be installed from GitHub: pip install git+https://github.com/mimbres/YourMT3.git
#
# This cell is expensive (~4-8 hours on M4). Skip if transcriptions already exist.

AMT_DIR = DATA_DIR / "amt_cache"
AMT_DIR.mkdir(parents=True, exist_ok=True)
YOURMT3_DIR = AMT_DIR / "yourmt3"
YOURMT3_DIR.mkdir(parents=True, exist_ok=True)
BYTEDANCE_DIR = AMT_DIR / "bytedance"
BYTEDANCE_DIR.mkdir(parents=True, exist_ok=True)

# Locate MAESTRO audio files for selected keys
# MAESTRO audio must be downloaded separately (~120GB total, or ~5GB for 50 files)
MAESTRO_AUDIO_DIR = DATA_DIR / "maestro_cache/audio"

if not MAESTRO_AUDIO_DIR.exists():
    print("MAESTRO audio not found. Download the 50 selected recordings first:")
    print("  See maestro-v3.0.0.json for download URLs")
    print("  Place WAV/FLAC files in data/maestro_cache/audio/")
else:
    # TODO: Run YourMT3+ transcription
    # TODO: Run ByteDance transcription
    # TODO: Save MIDI outputs to YOURMT3_DIR and BYTEDANCE_DIR
    print("AMT transcription code to be implemented when audio is available")
```

Cell 15 (code -- assess S2 on GT vs AMT MIDI):

```python
# Load S2 model
from model_improvement.symbolic_encoders import GraphNeuralNetworkEncoder
from model_improvement.graph import midi_to_graph
from model_improvement.layer1_validation import amt_degradation_comparison, select_maestro_subset

s2_ckpt = sorted(CHECKPOINT_DIR.glob("S2/fold_3/*.ckpt"))[0]
print(f"Loading S2 from {s2_ckpt.name}")

s2_model = GraphNeuralNetworkEncoder.load_from_checkpoint(str(s2_ckpt))
s2_model.eval()
s2_model = s2_model.to(device)

# Build graphs from ground-truth, YourMT3+, and ByteDance MIDI
# Then run pairwise assessment for each source
# Compare per-dimension accuracy drops
# (Detailed implementation depends on AMT output format)

print("S2 assessment on AMT MIDI: awaiting transcription output from cell 14")
```

**Step 5: Verify notebook creates without errors**

Run: `cd model && python -c "import json; json.load(open('notebooks/model_improvement/04_layer1_validation.ipynb')); print('Valid notebook')"`

**Step 6: Commit**

```bash
git add model/notebooks/model_improvement/04_layer1_validation.ipynb
git commit -m "feat(model): add Layer 1 validation notebook for Experiments 1-3"
```

---

## Task 6: Experiment 4 -- MIDI Comparison Module

**Files:**

- Create: `model/src/model_improvement/midi_comparison.py`
- Test: `model/tests/test_midi_comparison.py`

**Step 1: Write the failing test**

```python
# model/tests/test_midi_comparison.py
"""Tests for structured MIDI comparison."""

import numpy as np
import pytest

from model_improvement.midi_comparison import (
    compare_velocity_curves,
    compare_onset_timing,
    compare_note_accuracy,
    structured_midi_comparison,
)


def _make_notes(pitches, velocities, onsets, durations):
    """Helper to create fake note lists as dicts."""
    return [
        {"pitch": p, "velocity": v, "onset": o, "duration": d}
        for p, v, o, d in zip(pitches, velocities, onsets, durations)
    ]


def test_compare_velocity_curves():
    """Velocity comparison returns MAE and correlation."""
    perf_notes = _make_notes(
        [60, 62, 64], [80, 90, 70], [0.0, 0.5, 1.0], [0.4, 0.4, 0.4]
    )
    score_notes = _make_notes(
        [60, 62, 64], [64, 64, 64], [0.0, 0.5, 1.0], [0.4, 0.4, 0.4]
    )
    result = compare_velocity_curves(perf_notes, score_notes)
    assert "velocity_mae" in result
    assert "velocity_correlation" in result
    assert result["velocity_mae"] > 0  # Different velocities


def test_compare_onset_timing():
    """Onset comparison returns mean and max deviation."""
    perf_notes = _make_notes(
        [60, 62, 64], [80, 80, 80], [0.02, 0.48, 1.05], [0.4, 0.4, 0.4]
    )
    score_notes = _make_notes(
        [60, 62, 64], [64, 64, 64], [0.0, 0.5, 1.0], [0.4, 0.4, 0.4]
    )
    result = compare_onset_timing(perf_notes, score_notes)
    assert "mean_deviation_ms" in result
    assert "max_deviation_ms" in result
    assert result["mean_deviation_ms"] > 0


def test_compare_note_accuracy():
    """Note accuracy reports missed and extra notes."""
    perf_notes = _make_notes(
        [60, 62, 65], [80, 80, 80], [0.0, 0.5, 1.0], [0.4, 0.4, 0.4]
    )
    score_notes = _make_notes(
        [60, 62, 64], [64, 64, 64], [0.0, 0.5, 1.0], [0.4, 0.4, 0.4]
    )
    result = compare_note_accuracy(perf_notes, score_notes)
    assert "note_f1" in result
    assert "missed_notes" in result
    assert "extra_notes" in result
    assert result["missed_notes"] == 1  # pitch 64 missing
    assert result["extra_notes"] == 1  # pitch 65 extra


def test_structured_midi_comparison_full():
    """structured_midi_comparison returns all comparison features."""
    perf_notes = _make_notes(
        [60, 62, 64], [80, 90, 70], [0.02, 0.48, 1.05], [0.4, 0.4, 0.4]
    )
    score_notes = _make_notes(
        [60, 62, 64], [64, 64, 64], [0.0, 0.5, 1.0], [0.4, 0.4, 0.4]
    )
    result = structured_midi_comparison(perf_notes, score_notes)
    assert "velocity" in result
    assert "timing" in result
    assert "accuracy" in result
```

**Step 2: Run test to verify it fails**

Run: `cd model && uv run pytest tests/test_midi_comparison.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# model/src/model_improvement/midi_comparison.py
"""Structured MIDI comparison for Experiment 4 (MIDI-as-context feedback test).

Compares a performance MIDI against a score reference MIDI to produce
structured features that can augment teacher LLM feedback.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def _match_notes(
    perf_notes: list[dict],
    score_notes: list[dict],
    onset_tolerance: float = 0.15,
    pitch_tolerance: int = 0,
) -> tuple[list[tuple[dict, dict]], list[dict], list[dict]]:
    """Match performance notes to score notes by pitch + onset proximity.

    Returns:
        (matched_pairs, missed_score_notes, extra_perf_notes)
    """
    used_score = set()
    matched = []
    extra = []

    for pn in perf_notes:
        best_idx = -1
        best_dist = float("inf")
        for i, sn in enumerate(score_notes):
            if i in used_score:
                continue
            if abs(pn["pitch"] - sn["pitch"]) > pitch_tolerance:
                continue
            dist = abs(pn["onset"] - sn["onset"])
            if dist < onset_tolerance and dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_idx >= 0:
            matched.append((pn, score_notes[best_idx]))
            used_score.add(best_idx)
        else:
            extra.append(pn)

    missed = [sn for i, sn in enumerate(score_notes) if i not in used_score]
    return matched, missed, extra


def compare_velocity_curves(
    perf_notes: list[dict],
    score_notes: list[dict],
) -> dict:
    """Compare velocity profiles between performance and score.

    Args:
        perf_notes: [{pitch, velocity, onset, duration}, ...].
        score_notes: [{pitch, velocity, onset, duration}, ...].

    Returns:
        {velocity_mae, velocity_correlation, velocity_profile_summary}.
    """
    matched, _, _ = _match_notes(perf_notes, score_notes)
    if not matched:
        return {"velocity_mae": 0.0, "velocity_correlation": 0.0, "n_matched": 0}

    perf_vel = np.array([m[0]["velocity"] for m in matched], dtype=float)
    score_vel = np.array([m[1]["velocity"] for m in matched], dtype=float)

    mae = float(np.mean(np.abs(perf_vel - score_vel)))

    if len(matched) >= 3 and np.std(perf_vel) > 0 and np.std(score_vel) > 0:
        corr, _ = stats.pearsonr(perf_vel, score_vel)
    else:
        corr = 0.0

    return {
        "velocity_mae": mae,
        "velocity_correlation": float(corr),
        "n_matched": len(matched),
        "perf_velocity_mean": float(perf_vel.mean()),
        "perf_velocity_std": float(perf_vel.std()),
        "score_velocity_mean": float(score_vel.mean()),
    }


def compare_onset_timing(
    perf_notes: list[dict],
    score_notes: list[dict],
) -> dict:
    """Compare onset timing between performance and score.

    Returns:
        {mean_deviation_ms, max_deviation_ms, std_deviation_ms, timing_profile}.
    """
    matched, _, _ = _match_notes(perf_notes, score_notes)
    if not matched:
        return {"mean_deviation_ms": 0.0, "max_deviation_ms": 0.0, "std_deviation_ms": 0.0}

    deviations_ms = np.array([
        (m[0]["onset"] - m[1]["onset"]) * 1000.0 for m in matched
    ])

    return {
        "mean_deviation_ms": float(np.mean(np.abs(deviations_ms))),
        "max_deviation_ms": float(np.max(np.abs(deviations_ms))),
        "std_deviation_ms": float(np.std(deviations_ms)),
        "mean_signed_deviation_ms": float(np.mean(deviations_ms)),
        "n_matched": len(matched),
    }


def compare_note_accuracy(
    perf_notes: list[dict],
    score_notes: list[dict],
) -> dict:
    """Compare note-level accuracy between performance and score.

    Returns:
        {note_f1, precision, recall, missed_notes, extra_notes}.
    """
    matched, missed, extra = _match_notes(perf_notes, score_notes)

    n_matched = len(matched)
    n_missed = len(missed)
    n_extra = len(extra)

    precision = n_matched / (n_matched + n_extra) if (n_matched + n_extra) > 0 else 0.0
    recall = n_matched / (n_matched + n_missed) if (n_matched + n_missed) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "note_f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "missed_notes": n_missed,
        "extra_notes": n_extra,
        "matched_notes": n_matched,
    }


def structured_midi_comparison(
    perf_notes: list[dict],
    score_notes: list[dict],
) -> dict:
    """Full structured comparison of performance vs score MIDI.

    Combines velocity, timing, and accuracy comparisons into a single
    structured output suitable for inclusion in teacher LLM prompts.

    Args:
        perf_notes: Performance note list [{pitch, velocity, onset, duration}].
        score_notes: Score reference note list.

    Returns:
        {velocity: {...}, timing: {...}, accuracy: {...}, summary: str}.
    """
    velocity = compare_velocity_curves(perf_notes, score_notes)
    timing = compare_onset_timing(perf_notes, score_notes)
    accuracy = compare_note_accuracy(perf_notes, score_notes)

    # Build human-readable summary for LLM context
    summary_parts = []
    summary_parts.append(
        f"Note accuracy: F1={accuracy['note_f1']:.2f} "
        f"({accuracy['missed_notes']} missed, {accuracy['extra_notes']} extra)"
    )
    summary_parts.append(
        f"Timing: mean deviation {timing['mean_deviation_ms']:.0f}ms, "
        f"max {timing['max_deviation_ms']:.0f}ms"
    )
    summary_parts.append(
        f"Dynamics: velocity MAE={velocity['velocity_mae']:.0f}, "
        f"correlation={velocity['velocity_correlation']:.2f}"
    )

    return {
        "velocity": velocity,
        "timing": timing,
        "accuracy": accuracy,
        "summary": "; ".join(summary_parts),
    }
```

**Step 4: Run test to verify it passes**

Run: `cd model && uv run pytest tests/test_midi_comparison.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add model/src/model_improvement/midi_comparison.py model/tests/test_midi_comparison.py
git commit -m "feat(model): add structured MIDI comparison module for Experiment 4"
```

---

## Task 7: Experiment 4 -- Feedback Assessment Script

**Files:**

- Create: `model/src/model_improvement/feedback_assessment.py`
- Test: `model/tests/test_feedback_assessment.py`

**Step 1: Write the failing test**

```python
# model/tests/test_feedback_assessment.py
"""Tests for LLM feedback assessment (Experiment 4)."""

import pytest

from model_improvement.feedback_assessment import (
    build_condition_a_prompt,
    build_condition_b_prompt,
    build_judge_prompt,
    parse_judge_response,
)


def test_build_condition_a_prompt():
    """Condition A includes scores but no MIDI comparison."""
    scores = {"dynamics": 0.65, "timing": 0.71, "pedaling": 0.35,
              "articulation": 0.58, "phrasing": 0.62, "interpretation": 0.54}
    student_context = {"level": "intermediate", "session_count": 12}
    prompt = build_condition_a_prompt(scores, student_context)
    assert "0.35" in prompt  # pedaling score appears
    assert "comparison" not in prompt.lower()


def test_build_condition_b_prompt():
    """Condition B includes scores AND MIDI comparison."""
    scores = {"dynamics": 0.65, "timing": 0.71, "pedaling": 0.35,
              "articulation": 0.58, "phrasing": 0.62, "interpretation": 0.54}
    student_context = {"level": "intermediate", "session_count": 12}
    midi_comparison = {
        "velocity": {"velocity_mae": 15.0, "velocity_correlation": 0.8},
        "timing": {"mean_deviation_ms": 45.0, "max_deviation_ms": 120.0},
        "accuracy": {"note_f1": 0.95, "missed_notes": 2, "extra_notes": 1},
        "summary": "Note accuracy: F1=0.95; Timing: mean deviation 45ms; Dynamics: MAE=15",
    }
    prompt = build_condition_b_prompt(scores, student_context, midi_comparison)
    assert "0.35" in prompt
    assert "45" in prompt  # timing deviation
    assert "F1" in prompt or "accuracy" in prompt.lower()


def test_build_judge_prompt():
    """Judge prompt presents two observations in randomized order."""
    obs_a = "Your pedaling could use some work in this passage."
    obs_b = "In bars 5-8, the sustain pedal is held through the harmonic change at beat 3."
    prompt = build_judge_prompt(obs_a, obs_b)
    # Both observations must appear
    assert obs_a in prompt or obs_b in prompt
    assert "specificity" in prompt.lower() or "actionability" in prompt.lower()


def test_parse_judge_response():
    """parse_judge_response extracts winner and reasoning."""
    response = '{"winner": "B", "specificity": "B references specific bars", "actionability": "B tells what to do", "accuracy": "Both reasonable"}'
    result = parse_judge_response(response)
    assert result["winner"] in ("A", "B")
    assert "specificity" in result
```

**Step 2: Run test to verify it fails**

Run: `cd model && uv run pytest tests/test_feedback_assessment.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# model/src/model_improvement/feedback_assessment.py
"""LLM feedback assessment for Experiment 4 (MIDI-as-context test).

Generates teacher observations under two conditions:
  A: A1 dimension scores + student context only
  B: A1 dimension scores + student context + structured MIDI comparison

Then uses an LLM judge to determine which produces more specific, actionable feedback.
"""

from __future__ import annotations

import json
import logging
import random

logger = logging.getLogger(__name__)

# Subagent prompt template adapted from docs/apps/06a-subagent-architecture.md
_SUBAGENT_SYSTEM = """You are a piano teaching assistant analyzing a practice moment.
Given the student's context and performance data, generate a brief, specific
teaching observation (2-3 sentences) that a piano teacher would say.

Focus on one dimension. Be specific about what you hear and what to try next.
Reference passage location if possible. Speak warmly but directly."""


def build_condition_a_prompt(
    scores: dict[str, float],
    student_context: dict,
) -> str:
    """Build teacher prompt with A1 scores only (no MIDI data).

    Args:
        scores: {dimension_name: float} for 6 dimensions.
        student_context: {level, session_count, goals, ...}.

    Returns:
        Complete prompt string for the teacher LLM.
    """
    scores_text = "\n".join(f"  {dim}: {val:.2f}" for dim, val in scores.items())
    weakest_dim = min(scores, key=scores.get)

    return f"""{_SUBAGENT_SYSTEM}

## Student Context
- Level: {student_context.get('level', 'unknown')}
- Session count: {student_context.get('session_count', 0)}
- Goals: {student_context.get('goals', 'none specified')}

## Performance Scores (0-1 scale, higher = better)
{scores_text}

The weakest dimension is **{weakest_dim}** ({scores[weakest_dim]:.2f}).

Generate a teaching observation focused on {weakest_dim}."""


def build_condition_b_prompt(
    scores: dict[str, float],
    student_context: dict,
    midi_data: dict,
) -> str:
    """Build teacher prompt with A1 scores AND structured MIDI data.

    Args:
        scores: {dimension_name: float} for 6 dimensions.
        student_context: {level, session_count, goals, ...}.
        midi_data: Output of structured_midi_comparison().

    Returns:
        Complete prompt string for the teacher LLM.
    """
    scores_text = "\n".join(f"  {dim}: {val:.2f}" for dim, val in scores.items())
    weakest_dim = min(scores, key=scores.get)

    midi_summary = midi_data.get("summary", "")
    velocity_info = midi_data.get("velocity", {})
    timing_info = midi_data.get("timing", {})
    accuracy_info = midi_data.get("accuracy", {})

    midi_detail = f"""## MIDI Analysis (performance vs score reference)
{midi_summary}

### Velocity (dynamics)
- MAE: {velocity_info.get('velocity_mae', 0):.0f}
- Correlation with score dynamics: {velocity_info.get('velocity_correlation', 0):.2f}
- Performance mean velocity: {velocity_info.get('perf_velocity_mean', 0):.0f}

### Timing
- Mean onset deviation: {timing_info.get('mean_deviation_ms', 0):.0f}ms
- Max onset deviation: {timing_info.get('max_deviation_ms', 0):.0f}ms
- Systematic tendency: {timing_info.get('mean_signed_deviation_ms', 0):+.0f}ms (positive = rushing)

### Note Accuracy
- F1: {accuracy_info.get('note_f1', 0):.2f}
- Missed notes: {accuracy_info.get('missed_notes', 0)}
- Extra notes: {accuracy_info.get('extra_notes', 0)}"""

    return f"""{_SUBAGENT_SYSTEM}

## Student Context
- Level: {student_context.get('level', 'unknown')}
- Session count: {student_context.get('session_count', 0)}
- Goals: {student_context.get('goals', 'none specified')}

## Performance Scores (0-1 scale, higher = better)
{scores_text}

The weakest dimension is **{weakest_dim}** ({scores[weakest_dim]:.2f}).

{midi_detail}

Use the MIDI analysis data to make your observation more specific.
Reference concrete details (e.g., velocity differences, timing deviations).
Generate a teaching observation focused on {weakest_dim}."""


def build_judge_prompt(
    observation_a: str,
    observation_b: str,
    randomize: bool = True,
) -> str:
    """Build LLM judge prompt comparing two observations.

    Presents observations in randomized order to avoid position bias.

    Args:
        observation_a: Teacher observation from Condition A.
        observation_b: Teacher observation from Condition B.
        randomize: Whether to randomize presentation order.

    Returns:
        Judge prompt string. The caller tracks which is which.
    """
    if randomize and random.random() < 0.5:
        first, second = observation_b, observation_a
        first_label, second_label = "B", "A"
    else:
        first, second = observation_a, observation_b
        first_label, second_label = "A", "B"

    return f"""You are judging two piano teaching observations for the same practice moment.
Rate which observation is better on three criteria:

1. **Specificity**: Does it reference particular passages, bars, or musical details?
2. **Actionability**: Does it tell the student what to do differently?
3. **Accuracy**: Does the observation sound musically plausible and precise?

## Observation X
{first}

## Observation Y
{second}

Respond with JSON only:
{{
    "winner": "X" or "Y",
    "specificity": "Which is more specific and why (1 sentence)",
    "actionability": "Which is more actionable and why (1 sentence)",
    "accuracy": "Which sounds more accurate and why (1 sentence)",
    "confidence": "high" or "medium" or "low"
}}"""


def parse_judge_response(response: str) -> dict:
    """Parse judge LLM response into structured result.

    Args:
        response: Raw LLM response (should be JSON).

    Returns:
        Parsed dict with winner, specificity, actionability, accuracy.

    Raises:
        ValueError: If response cannot be parsed as JSON.
    """
    # Try to extract JSON from response (handle markdown code blocks)
    text = response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])

    parsed = json.loads(text)

    required = {"winner", "specificity", "actionability", "accuracy"}
    missing = required - set(parsed.keys())
    if missing:
        raise ValueError(f"Judge response missing keys: {missing}")

    return parsed
```

**Step 4: Run test to verify it passes**

Run: `cd model && uv run pytest tests/test_feedback_assessment.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add model/src/model_improvement/feedback_assessment.py model/tests/test_feedback_assessment.py
git commit -m "feat(model): add LLM feedback assessment module for Experiment 4"
```

---

## Task 8: Experiment 4 -- Runner Script

**Files:**

- Create: `model/src/model_improvement/run_feedback_assessment.py`

This is the standalone script referenced in the design doc. It orchestrates the full
Experiment 4 pipeline: select segments, score with A1, build MIDI comparisons,
generate observations under both conditions, run LLM judge.

**Step 1: Write the runner script**

```python
# model/src/model_improvement/run_feedback_assessment.py
"""Run Experiment 4: MIDI-as-context feedback assessment.

Usage:
    cd model
    uv run python -m model_improvement.run_feedback_assessment

Requires ANTHROPIC_API_KEY environment variable.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

from model_improvement.audio_encoders import MuQLoRAModel
from model_improvement.feedback_assessment import (
    build_condition_a_prompt,
    build_condition_b_prompt,
    build_judge_prompt,
    parse_judge_response,
)
from model_improvement.layer1_validation import score_competition_segments
from model_improvement.midi_comparison import structured_midi_comparison
from model_improvement.taxonomy import DIMENSIONS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

DATA_DIR = Path(__file__).parent.parent.parent / "data"
CHECKPOINT_DIR = DATA_DIR / "checkpoints/model_improvement"
PERCEPIANO_DIR = DATA_DIR / "percepiano_cache"
RESULTS_DIR = DATA_DIR / "experiment4_results"


def _load_anthropic_client():
    """Load Anthropic client. Raises if API key not set."""
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable not set")
    return anthropic.Anthropic(api_key=api_key)


def _call_llm(client, prompt: str, model: str = "claude-sonnet-4-6") -> str:
    """Call Claude API and return text response."""
    response = client.messages.create(
        model=model,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def _select_segments(
    labels: dict,
    n_top: int = 5,
    n_middle: int = 10,
    n_bottom: int = 5,
    piece_prefix: str = "Schubert_D960_mv3",
) -> list[str]:
    """Select PercePiano segments spanning the quality range for one piece."""
    # Filter to the target piece
    piece_keys = [k for k in labels if k.startswith(piece_prefix)]
    if not piece_keys:
        logger.warning("No segments found for %s, using all keys", piece_prefix)
        piece_keys = list(labels.keys())

    # Sort by mean score
    scored = [(k, np.mean(labels[k][:6])) for k in piece_keys]
    scored.sort(key=lambda x: x[1], reverse=True)

    selected = []
    selected.extend([k for k, _ in scored[:n_top]])
    mid_start = len(scored) // 2 - n_middle // 2
    selected.extend([k for k, _ in scored[mid_start:mid_start + n_middle]])
    selected.extend([k for k, _ in scored[-n_bottom:]])

    # Deduplicate while preserving order
    seen = set()
    result = []
    for k in selected:
        if k not in seen:
            seen.add(k)
            result.append(k)

    return result[:n_top + n_middle + n_bottom]


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load A1 model
    a1_ckpt = sorted(CHECKPOINT_DIR.glob("A1/fold_3/*.ckpt"))[0]
    logger.info("Loading A1 from %s", a1_ckpt.name)
    a1_model = MuQLoRAModel.load_from_checkpoint(str(a1_ckpt), use_pretrained_muq=False)
    a1_model.eval()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    a1_model = a1_model.to(device)

    # Load labels and embeddings
    from model_improvement.taxonomy import load_composite_labels
    labels = load_composite_labels(DATA_DIR / "composite_labels/composite_labels.json")

    # Select 20 segments
    selected = _select_segments(labels)
    logger.info("Selected %d segments for assessment", len(selected))

    # Load pre-extracted embeddings
    emb_cache = PERCEPIANO_DIR / "_muq_file_cache"
    embeddings = {}
    for key in selected:
        pt_path = emb_cache / f"{key}.pt"
        if pt_path.exists():
            embeddings[key] = torch.load(pt_path, map_location="cpu", weights_only=True)
        else:
            logger.warning("Embedding not found for %s", key)

    # Score with A1
    segment_scores = score_competition_segments(a1_model, embeddings)

    # Build MIDI comparisons (requires Score MIDI)
    # For segments with corresponding MIDI, build structured comparison.
    # Score MIDI path convention: same key with _Score suffix in percepiano_midi/
    midi_comparisons = {}
    midi_dir = DATA_DIR / "percepiano_midi"
    if midi_dir.exists():
        import pretty_midi
        for key in selected:
            perf_midi_path = midi_dir / f"{key}.mid"
            # Find score MIDI (look for Score version)
            score_key = key.rsplit("_", 1)[0] + "_Score"
            score_midi_path = midi_dir / f"{score_key}.mid"

            if perf_midi_path.exists() and score_midi_path.exists():
                perf_pm = pretty_midi.PrettyMIDI(str(perf_midi_path))
                score_pm = pretty_midi.PrettyMIDI(str(score_midi_path))

                perf_notes = [
                    {"pitch": n.pitch, "velocity": n.velocity,
                     "onset": n.start, "duration": n.end - n.start}
                    for inst in perf_pm.instruments for n in inst.notes
                ]
                score_notes = [
                    {"pitch": n.pitch, "velocity": n.velocity,
                     "onset": n.start, "duration": n.end - n.start}
                    for inst in score_pm.instruments for n in inst.notes
                ]

                midi_comparisons[key] = structured_midi_comparison(perf_notes, score_notes)
    else:
        logger.warning("No MIDI directory at %s -- Condition B will be limited", midi_dir)

    # Generate observations and judge
    client = _load_anthropic_client()
    results = []
    student_context = {"level": "intermediate", "session_count": 12}

    for key in selected:
        if key not in segment_scores:
            continue

        scores = {dim: float(segment_scores[key][d]) for d, dim in enumerate(DIMENSIONS)}

        # Condition A: scores only
        prompt_a = build_condition_a_prompt(scores, student_context)
        obs_a = _call_llm(client, prompt_a)

        # Condition B: scores + MIDI data
        midi_comp = midi_comparisons.get(key)
        if midi_comp:
            prompt_b = build_condition_b_prompt(scores, student_context, midi_comp)
        else:
            # Fall back to condition A if no MIDI available
            logger.warning("No MIDI data for %s, skipping", key)
            continue

        obs_b = _call_llm(client, prompt_b)

        # Judge
        judge_prompt = build_judge_prompt(obs_a, obs_b)
        judge_response = _call_llm(client, judge_prompt)

        try:
            judgment = parse_judge_response(judge_response)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Failed to parse judge response for %s: %s", key, e)
            judgment = {"winner": "unknown", "error": str(e)}

        result = {
            "segment_id": key,
            "scores": scores,
            "observation_a": obs_a,
            "observation_b": obs_b,
            "judgment": judgment,
            "has_midi": midi_comp is not None,
        }
        results.append(result)
        logger.info(
            "  %s: winner=%s", key, judgment.get("winner", "unknown")
        )

    # Save results
    output_path = RESULTS_DIR / "feedback_assessment_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    wins_b = sum(1 for r in results if r["judgment"].get("winner") == "B")
    wins_a = sum(1 for r in results if r["judgment"].get("winner") == "A")
    total = wins_a + wins_b
    if total > 0:
        win_rate_b = wins_b / total * 100
        gate = "BUILD" if win_rate_b > 65 else "SKIP" if win_rate_b < 55 else "BORDERLINE"
        logger.info("\n=== Experiment 4 Results ===")
        logger.info("Condition B (MIDI context) wins: %d/%d (%.0f%%)", wins_b, total, win_rate_b)
        logger.info("Condition A (scores only) wins: %d/%d (%.0f%%)", wins_a, total, 100 - win_rate_b)
        logger.info("Decision: %s score comparison pipeline", gate)
    else:
        logger.warning("No valid comparisons completed")

    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
```

**Step 2: Verify the script parses without syntax errors**

Run: `cd model && uv run python -c "import ast; ast.parse(open('src/model_improvement/run_feedback_assessment.py').read()); print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add model/src/model_improvement/run_feedback_assessment.py
git commit -m "feat(model): add Experiment 4 feedback assessment runner script"
```

---

## Task 9: Update pyproject.toml and hatch build config

**Files:**

- Modify: `model/pyproject.toml`

**Step 1: Verify current state**

The design doc mentions adding `yourmt3` and `piano-transcription-inference`. However, these are heavy dependencies only needed for Experiment 2's AMT transcription. Add them as optional dependencies rather than core deps.

**Step 2: Add optional dependency group for Layer 1 experiments**

Add after the existing `[project.optional-dependencies]` section:

```toml
# Layer 1 validation experiments (Experiment 2 AMT transcription)
layer1 = [
    "piano-transcription-inference>=0.1.0",
]
```

Note: `yourmt3` is not on PyPI -- it must be installed from GitHub. Add a comment in the notebook (cell 14) instead.

Also verify the hatch build includes the new module paths (it should since `model_improvement` is already listed).

**Step 3: Commit**

```bash
git add model/pyproject.toml
git commit -m "chore(model): add optional layer1 dependency group for AMT transcription"
```

---

## Task 10: Run All Tests and Verify

**Step 1: Run all new tests**

```bash
cd model && uv run pytest tests/test_layer1_validation.py tests/test_midi_comparison.py tests/test_feedback_assessment.py -v
```

Expected: All tests PASS.

**Step 2: Run existing tests to check for regressions**

```bash
cd model && uv run pytest tests/ -v --timeout=60
```

Expected: No regressions.

**Step 3: Verify notebook JSON is valid**

```bash
cd model && python -c "import json; json.load(open('notebooks/model_improvement/04_layer1_validation.ipynb')); print('Valid notebook')"
```

Expected: `Valid notebook`

**Step 4: Commit if any fixes were needed**

```bash
git add -A
git status
# Only commit if there are changes from fixes
```

---

## Task 11: Update Training Results Doc

**Files:**

- Modify: `docs/model/04-training-results.md`

**Step 1: Add Layer 1 validation section stub**

Add at the end of the document:

```markdown
## Layer 1 Validation Results

*Status: EXPERIMENTS READY, awaiting execution*

See `docs/plans/2026-03-09-layer1-validation-design.md` for experiment design.
Code: `model/src/model_improvement/layer1_validation.py`, `midi_comparison.py`, `feedback_assessment.py`.
Notebook: `model/notebooks/model_improvement/04_layer1_validation.ipynb`.

### Experiment 1: Competition Correlation
- Data: 2,293 segments from Chopin 2021 (synced from GDrive)
- Gate: rho > 0.3 = signal is real

### Experiment 2: AMT Degradation
- Data: 50 MAESTRO recordings, GT vs YourMT3+ vs ByteDance MIDI
- Gate: per-dimension pairwise drop < 10% = symbolic viable

### Experiment 3: Dynamic Range
- Data: intermediate YouTube recordings (to be collected)
- Diagnostic only, no hard gate

### Experiment 4: MIDI-as-Context Feedback
- Data: 20 PercePiano Schubert D960 segments
- Gate: MIDI-context wins > 65% of LLM judge pairs
```

**Step 2: Commit**

```bash
git add docs/model/04-training-results.md
git commit -m "docs(model): add Layer 1 validation results stub to training results"
```
