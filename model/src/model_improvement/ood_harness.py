"""Out-of-distribution testing harness.

Keeps OOD data strictly outside piece-stratified CV (folds.json) so practice
distribution measurements don't leak into the fold leaderboard. Emits the same
JSON structure as evaluate_model() so downstream dashboards treat fold and OOD
results identically.

Typical usage:

    from model_improvement.ood_harness import OODDataset, run_ood_test
    ds = OODDataset(
        cache_dir=Evals.root / "ood_practice" / "embeddings",
        labels_path=Evals.root / "ood_practice" / "labels.json",
    )
    result = run_ood_test(
        model, ds,
        encode_fn=lambda m, inp, mask: m.encode(inp, mask),
        compare_fn=lambda m, z_a, z_b: m.compare(z_a, z_b),
        predict_fn=lambda m, inp, mask: m.predict_scores(inp, mask),
    )
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from model_improvement.evaluation import evaluate_model
from model_improvement.taxonomy import NUM_DIMS


class OODDataset:
    """Load OOD practice recordings from an arbitrary directory outside folds.json.

    Expected on-disk layout (cache_dir + labels_path may be siblings or anywhere):
        labels_path: JSON mapping segment_id -> {
            "ordinal": int (1-5 single-rater bucket, optional),
            "scores":  list[float] length NUM_DIMS (derived composite, required),
        }
        cache_dir:   {segment_id}.pt torch tensors, [T, D] frame embeddings.

    An empty or missing labels_path yields an empty dataset (len 0); the harness
    handles that cleanly so CI smoke tests can run without real data.
    """

    def __init__(self, cache_dir: str | Path, labels_path: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.labels_path = Path(labels_path)

        self.labels: dict[str, list[float]] = {}
        self.skill_tiers: dict[str, int | float] = {}

        if not self.labels_path.exists():
            self.keys: list[str] = []
            return

        with open(self.labels_path) as f:
            raw = json.load(f)

        for seg_id, entry in raw.items():
            scores = entry.get("scores")
            if scores is None or len(scores) < NUM_DIMS:
                continue
            self.labels[seg_id] = list(np.asarray(scores, dtype=np.float32).tolist())
            if "ordinal" in entry and entry["ordinal"] is not None:
                self.skill_tiers[seg_id] = int(entry["ordinal"])

        self.keys = sorted(self.labels.keys())

    def load_embedding(self, key: str) -> torch.Tensor:
        """Read the [T, D] frame embedding for a segment."""
        path = self.cache_dir / f"{key}.pt"
        if not path.exists():
            raise FileNotFoundError(f"No embedding for OOD key {key} at {path}")
        return torch.load(path, map_location="cpu", weights_only=True)

    def __len__(self) -> int:
        return len(self.keys)


def _default_get_input_fn(
    dataset: OODDataset,
) -> Callable[[str], tuple[torch.Tensor, torch.Tensor]]:
    """Build a get_input_fn that reads frame tensors from the dataset's cache_dir."""

    def _inner(key: str) -> tuple[torch.Tensor, torch.Tensor]:
        emb = dataset.load_embedding(key)
        if emb.dim() == 2:
            emb = emb.unsqueeze(0)  # [1, T, D]
        mask = torch.ones(emb.shape[:2], dtype=torch.bool)
        return emb, mask

    return _inner


def run_ood_test(
    model: torch.nn.Module,
    ood_dataset: OODDataset,
    encode_fn: Callable,
    compare_fn: Callable,
    predict_fn: Callable,
    get_input_fn: Optional[Callable[[str], tuple[torch.Tensor, torch.Tensor]]] = None,
    num_dims: int = NUM_DIMS,
) -> dict:
    """Score a model on an OODDataset; returns the same shape as evaluate_model().

    If get_input_fn is None, a default loader reads tensors from
    ood_dataset.cache_dir. Supply your own when embeddings live in memory
    already (e.g. pre-extracted during a sweep).

    Returns:
        Dict with all fold-scoring fields (pairwise, r2, dimension_collapse_score,
        etc.) plus:
          - "n_samples": number of OOD clips scored
          - "skipped": sentinel when the dataset is empty (no other fields)
    """
    if len(ood_dataset) == 0:
        return {"skipped": "empty_ood_dataset", "n_samples": 0}

    if get_input_fn is None:
        get_input_fn = _default_get_input_fn(ood_dataset)

    skill_tiers = ood_dataset.skill_tiers if ood_dataset.skill_tiers else None

    result = evaluate_model(
        model=model,
        val_keys=ood_dataset.keys,
        labels=ood_dataset.labels,
        get_input_fn=get_input_fn,
        encode_fn=encode_fn,
        compare_fn=compare_fn,
        predict_fn=predict_fn,
        num_dims=num_dims,
        skill_tiers=skill_tiers,
    )
    result["n_samples"] = len(ood_dataset)
    result["ood_source"] = str(ood_dataset.labels_path)
    return result
