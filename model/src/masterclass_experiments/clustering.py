"""Sentence-transformer embedding and HDBSCAN clustering of open descriptions."""

from __future__ import annotations

import json
from pathlib import Path

import hdbscan
import numpy as np
from sentence_transformers import SentenceTransformer


def load_open_descriptions(jsonl_path: Path) -> tuple[list[str], list[str]]:
    """Load moment IDs and open descriptions from moments JSONL.

    Falls back to feedback_summary when open_description is missing
    (backward compat with moments extracted before the A+C merge).

    Returns:
        (moment_ids, descriptions) -- parallel lists.
    """
    ids: list[str] = []
    descriptions: list[str] = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            desc = record.get("open_description") or record.get("feedback_summary")
            if not desc:
                continue
            ids.append(record["moment_id"])
            descriptions.append(desc)
    return ids, descriptions


def embed_descriptions(
    descriptions: list[str],
    model_name: str = "all-MiniLM-L6-v2",
) -> np.ndarray:
    """Embed descriptions using a sentence transformer.

    Returns:
        np.ndarray of shape [N, D] (float32).
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(descriptions, show_progress_bar=False)
    return np.asarray(embeddings, dtype=np.float32)


def cluster_descriptions(
    embeddings: np.ndarray,
    min_cluster_size: int = 15,
    min_samples: int | None = None,
) -> tuple[np.ndarray, hdbscan.HDBSCAN]:
    """Cluster embeddings with HDBSCAN.

    Returns:
        (labels, clusterer) -- labels array of shape [N], -1 for noise.
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    clusterer.fit(embeddings)
    return clusterer.labels_, clusterer


def summarize_clusters(
    descriptions: list[str],
    labels: np.ndarray,
) -> list[dict]:
    """Produce a summary of each cluster.

    Returns:
        List of dicts with cluster_id, size, examples, and frequency.
    """
    total = (labels >= 0).sum()
    cluster_ids = sorted(set(labels[labels >= 0]))
    summaries = []
    for cid in cluster_ids:
        mask = labels == cid
        examples = [d for d, m in zip(descriptions, mask) if m]
        summaries.append(
            {
                "cluster_id": int(cid),
                "size": int(mask.sum()),
                "frequency": float(mask.sum() / total) if total > 0 else 0.0,
                "examples": examples[:10],
            }
        )
    return summaries
