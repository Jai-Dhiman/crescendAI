"""Aria-embedding matcher: cosine-ranked retrieval over the exercise catalog.

The catalog (catalog.py) already stores a 512-dim Aria embedding per primitive.
Retrieval is therefore pure linear algebra: L2-normalize the catalog matrix and
the query, take dot products, and rank. No pgvector here -- that production index
is the api-side concern (slice B / issue #29). This module is the model-side
reference implementation, validated on the small local corpus.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from exercise_corpus.catalog import CatalogRow, read_primitives

EMBED_DIM = 512


@dataclass
class Match:
    primitive_id: str
    source: str
    title: str
    midi_path: Path
    score: float  # cosine similarity in [-1, 1]


@dataclass
class CatalogIndex:
    """Preloaded catalog matrix for repeated queries without re-reading SQLite."""

    rows: list[CatalogRow]
    matrix: np.ndarray  # (n, EMBED_DIM), L2-normalized float32


def _l2_normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    # A zero vector has no direction; refuse it rather than dividing by zero.
    if np.any(norms == 0):
        raise ValueError("Cannot normalize a zero-magnitude embedding")
    return m / norms


def load_index(db_path: Path) -> CatalogIndex:
    """Read the catalog and build an L2-normalized embedding matrix.

    Args:
        db_path: path to the SQLite catalog written by catalog.write_primitives.

    Returns:
        CatalogIndex holding rows (catalog order) and the normalized matrix.

    Raises:
        FileNotFoundError: if db_path does not exist.
        ValueError: if the catalog is empty or any embedding is not EMBED_DIM.
    """
    rows = read_primitives(db_path)
    if not rows:
        raise ValueError(f"Catalog at {db_path} contains no primitives")
    matrix = np.stack([r.embedding for r in rows], axis=0).astype(np.float32)
    if matrix.shape[1] != EMBED_DIM:
        raise ValueError(
            f"Catalog embeddings must be {EMBED_DIM}-dim, got {matrix.shape[1]}"
        )
    return CatalogIndex(rows=rows, matrix=_l2_normalize(matrix))


def match_exercises(
    query: np.ndarray,
    db_path: Path | None = None,
    index: CatalogIndex | None = None,
    top_k: int = 5,
    sources: list[str] | None = None,
) -> list[Match]:
    """Return the top_k catalog exercises most similar to a query embedding.

    Args:
        query: a 512-dim embedding (e.g. an Aria embedding of a student's weak
            segment), in the same space as the catalog.
        db_path: path to the SQLite catalog. Mutually sufficient with index.
        index: a preloaded CatalogIndex (avoids re-reading SQLite per query).
        top_k: maximum number of matches to return.
        sources: optional whitelist of source names to restrict candidates to.

    Returns:
        List of Match ordered by descending cosine similarity, ties broken by
        primitive_id ascending for determinism. Length <= top_k.

    Raises:
        ValueError: if neither db_path nor index is given, if query is not
            512-dim, or (via load_index) if the catalog is empty.
    """
    if index is None:
        if db_path is None:
            raise ValueError("match_exercises requires db_path or index")
        index = load_index(db_path)

    q = np.asarray(query, dtype=np.float32)
    if q.shape != (EMBED_DIM,):
        raise ValueError(
            f"query must be a {EMBED_DIM}-dim vector, got shape {q.shape}"
        )
    q = _l2_normalize(q)

    rows = index.rows
    sims = index.matrix @ q  # (n,) cosine similarities

    candidates = list(range(len(rows)))
    if sources is not None:
        allowed = set(sources)
        candidates = [i for i in candidates if rows[i].source in allowed]

    # Sort by (-score, primitive_id): score descending, deterministic tie-break.
    candidates.sort(key=lambda i: (-float(sims[i]), rows[i].primitive_id))

    return [
        Match(
            primitive_id=rows[i].primitive_id,
            source=rows[i].source,
            title=rows[i].title,
            midi_path=rows[i].midi_path,
            score=float(sims[i]),
        )
        for i in candidates[:top_k]
    ]
