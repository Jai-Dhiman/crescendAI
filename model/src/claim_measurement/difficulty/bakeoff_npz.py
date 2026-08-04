"""Per-piece .npz embedding contract shared by every backbone extractor.

Numeric-only arrays (float32 vectors, int32 scalars) so np.load never needs
pickle=True. A backbone may produce more than one pooled vector per piece
(MoonBeam: mean_pool + last_token); each is stored as its own array keyed
"emb__{pooling_name}" so an arbitrary number of poolings round-trip through
one file without colliding with the reserved "grade"/"composer_id" keys.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_EMB_PREFIX = "emb__"
_TMP_SUFFIX = ".npz.tmp"


@dataclass(frozen=True)
class EmbeddingRecord:
    embeddings: dict[str, np.ndarray]
    grade: int
    composer_id: int


def write_embedding_npz(path: Path, embeddings: dict[str, np.ndarray], grade: int, composer_id: int) -> None:
    """Write atomically: temp file in the destination directory, then
    os.replace. A crash mid-write during a long GPU extraction must never leave
    a truncated .npz behind, because extract_embeddings' resume path treats
    "file exists" as "already done"."""
    if not embeddings:
        raise ValueError("embeddings must contain at least one pooling vector")
    arrays = {f"{_EMB_PREFIX}{name}": np.asarray(vec, dtype=np.float32)
              for name, vec in embeddings.items()}
    arrays["grade"] = np.array(int(grade), dtype=np.int32)
    arrays["composer_id"] = np.array(int(composer_id), dtype=np.int32)
    parent = Path(path).parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=parent, suffix=_TMP_SUFFIX)
    try:
        with os.fdopen(fd, "wb") as fh:
            np.savez(fh, **arrays)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def read_embedding_npz(path: Path) -> EmbeddingRecord:
    with np.load(path) as z:
        embeddings = {k[len(_EMB_PREFIX):]: z[k] for k in z.files if k.startswith(_EMB_PREFIX)}
        grade = int(z["grade"])
        composer_id = int(z["composer_id"])
    return EmbeddingRecord(embeddings=embeddings, grade=grade, composer_id=composer_id)
