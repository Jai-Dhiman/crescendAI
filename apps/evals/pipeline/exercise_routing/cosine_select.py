"""Python mirror of cosine-select.ts for the eval's cosine A/B re-measure.

The eval compares selector strategies on the SAME sessions: the deterministic
top-1 (selection.select_primitive) vs cosine-within-dimension (here). Both read
the same committed catalog asset (exercise_embeddings.json, L2-normalized) so the
eval's cosine choice matches what the Worker's cosine-select.ts would pick for the
same query -- a faithful counterfactual, not a re-implementation that could drift.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_DEFAULT_ASSET = (
    Path(__file__).resolve().parents[3]
    / "api" / "src" / "services" / "exercise_embeddings.json"
)


class CatalogEmbeddings:
    """Preloaded L2-normalized catalog matrix + id list + manifest dimensions."""

    def __init__(self, asset_path: Path, manifest: dict):
        data = json.loads(Path(asset_path).read_text())
        self.dim: int = data["dim"]
        self.ids: list[str] = data["ids"]
        self.matrix = np.asarray(data["vectors"], dtype=np.float32)  # (n, dim), normalized
        if self.matrix.shape != (len(self.ids), self.dim):
            raise ValueError(
                f"asset matrix {self.matrix.shape} != ({len(self.ids)}, {self.dim})"
            )
        self.manifest = manifest


def load_catalog(manifest: dict, asset_path: Path = _DEFAULT_ASSET) -> CatalogEmbeddings:
    return CatalogEmbeddings(asset_path, manifest)


def cosine_select_within_dimension(
    query: np.ndarray, target_dimension: str, catalog: CatalogEmbeddings
) -> str | None:
    """Return the primitive_id best matching `query` among drills tagged for the
    dimension, or None if the bucket is empty (caller widens). Ties break by id.
    """
    q = np.asarray(query, dtype=np.float32)
    if q.shape != (catalog.dim,):
        raise ValueError(f"query dim {q.shape} != catalog dim ({catalog.dim},)")
    norm = float(np.linalg.norm(q))
    if norm == 0:
        raise ValueError("query embedding has zero magnitude")
    q = q / norm

    sims = catalog.matrix @ q  # (n,) cosine (matrix pre-normalized)
    best_id: str | None = None
    best_score = -2.0
    for i, pid in enumerate(catalog.ids):
        entry = catalog.manifest.get(pid)
        if entry is None or target_dimension not in entry.get("dimensions", []):
            continue
        s = float(sims[i])
        if s > best_score or (s == best_score and (best_id is None or pid < best_id)):
            best_score = s
            best_id = pid
    return best_id
