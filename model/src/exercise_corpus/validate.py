"""k-NN source purity validation, UMAP visualization, and review artifact generation."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless runs
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
import umap

from exercise_corpus.catalog import read_primitives

logger = logging.getLogger(__name__)

PURITY_THRESHOLD = 0.70
PAIRS_PER_SOURCE = 5
K = 5


@dataclass
class ValidationResult:
    purity: float
    verdict: str
    pairs: list[dict]
    umap_path: Path
    pairs_path: Path


def source_purity(
    embeddings: np.ndarray,
    labels: list[str],
    k: int = K,
) -> float:
    """Compute k-NN source purity.

    For each point, find its k nearest neighbors (excluding itself) and
    compute the fraction whose label matches the query point's label.
    Average across all points.

    Args:
        embeddings: float32 array of shape (n, dim).
        labels: list of n source strings.
        k: number of neighbors.

    Returns:
        Float in [0, 1]. 1.0 = all neighbors same source.
    """
    n = len(labels)
    if n <= k:
        raise ValueError(
            f"Need more than k={k} points to compute purity, got {n}"
        )
    # k+1 because NearestNeighbors includes the point itself
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn.fit(embeddings)
    _, indices = nn.kneighbors(embeddings)
    # indices[:, 0] is the point itself; skip it
    neighbor_indices = indices[:, 1:]
    labels_arr = np.array(labels)
    same = (labels_arr[neighbor_indices] == labels_arr[:, None]).sum(axis=1)
    return float(same.mean() / k)


def run_validation(db_path: Path, output_dir: Path) -> ValidationResult:
    """Run full validation: purity metric, UMAP plot, and 15-pair review artifact.

    Does NOT raise on a failing metric. Reports PASS/FAIL verdict with numbers.

    Args:
        db_path: path to the SQLite catalog populated by catalog.write_primitives.
        output_dir: directory to write exercise_primitives_umap.png and
            exercise_primitives_neighbors.json.

    Returns:
        ValidationResult with purity, verdict, pairs, and file paths.

    Raises:
        FileNotFoundError: if db_path does not exist.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_primitives(db_path)
    if len(rows) == 0:
        raise ValueError(f"Catalog at {db_path} contains no primitives")

    embeddings = np.stack([r.embedding for r in rows], axis=0)
    labels = [r.source for r in rows]
    primitive_ids = [r.primitive_id for r in rows]

    purity = source_purity(embeddings, labels, k=K)
    verdict = "PASS" if purity >= PURITY_THRESHOLD else "FAIL"
    logger.info(
        "k-NN source purity (k=%d): %.4f -- %s (threshold %.2f)",
        K, purity, verdict, PURITY_THRESHOLD,
    )

    # UMAP
    reducer = umap.UMAP(n_components=2, random_state=42, metric="cosine")
    embedding_2d = reducer.fit_transform(embeddings)
    unique_sources = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 0.5, len(unique_sources)))
    source_to_color = dict(zip(unique_sources, colors))
    fig, ax = plt.subplots(figsize=(8, 6))
    for source in unique_sources:
        mask = np.array([l == source for l in labels])
        ax.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            label=source,
            color=source_to_color[source],
            alpha=0.7,
            s=20,
        )
    ax.set_title(f"Exercise Primitives UMAP (purity={purity:.3f}, {verdict})")
    ax.legend()
    umap_path = output_dir / "exercise_primitives_umap.png"
    fig.savefig(str(umap_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("UMAP plot saved to %s", umap_path)

    # 15 within-source nearest-neighbor pairs (5 per source)
    nn = NearestNeighbors(n_neighbors=K + 1, metric="cosine")
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)

    pairs: list[dict] = []
    seen_sources: dict[str, int] = {}
    for i, (pid, source) in enumerate(zip(primitive_ids, labels)):
        if seen_sources.get(source, 0) >= PAIRS_PER_SOURCE:
            continue
        for rank in range(1, K + 1):
            j = indices[i, rank]
            neighbor_source = labels[j]
            if neighbor_source == source:
                pairs.append(
                    {
                        "query_id": pid,
                        "neighbor_id": primitive_ids[j],
                        "source_a": source,
                        "source_b": neighbor_source,
                        "cosine_distance": float(distances[i, rank]),
                    }
                )
                seen_sources[source] = seen_sources.get(source, 0) + 1
                break

    pairs_path = output_dir / "exercise_primitives_neighbors.json"
    with open(pairs_path, "w") as f:
        json.dump(pairs, f, indent=2)
    logger.info("Review artifact saved to %s (%d pairs)", pairs_path, len(pairs))

    return ValidationResult(
        purity=purity,
        verdict=verdict,
        pairs=pairs,
        umap_path=umap_path,
        pairs_path=pairs_path,
    )
