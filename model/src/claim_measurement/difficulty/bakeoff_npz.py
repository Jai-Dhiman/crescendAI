"""Per-piece .npz embedding contract shared by every backbone extractor.

Numeric-only arrays (float32 vectors, int32 scalars) so np.load never needs
pickle=True. A backbone may produce more than one pooled vector per piece
(MoonBeam: mean_pool + last_token); each is stored as its own array keyed
"emb__{pooling_name}" so an arbitrary number of poolings round-trip through
one file without colliding with the reserved "grade"/"composer_id" keys.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_EMB_PREFIX = "emb__"


@dataclass(frozen=True)
class EmbeddingRecord:
    embeddings: dict[str, np.ndarray]
    grade: int
    composer_id: int


def write_embedding_npz(path: Path, embeddings: dict[str, np.ndarray], grade: int, composer_id: int) -> None:
    if not embeddings:
        raise ValueError("embeddings must contain at least one pooling vector")
    arrays = {f"{_EMB_PREFIX}{name}": np.asarray(vec, dtype=np.float32)
              for name, vec in embeddings.items()}
    arrays["grade"] = np.array(int(grade), dtype=np.int32)
    arrays["composer_id"] = np.array(int(composer_id), dtype=np.int32)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def read_embedding_npz(path: Path) -> EmbeddingRecord:
    with np.load(path) as z:
        embeddings = {k[len(_EMB_PREFIX):]: z[k] for k in z.files if k.startswith(_EMB_PREFIX)}
        grade = int(z["grade"])
        composer_id = int(z["composer_id"])
    return EmbeddingRecord(embeddings=embeddings, grade=grade, composer_id=composer_id)
