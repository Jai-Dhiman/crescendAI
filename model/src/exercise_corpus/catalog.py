"""SQLite catalog for exercise primitives with embedding storage."""

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


@dataclass
class CatalogRow:
    primitive_id: str
    source: str
    source_exercise_number: int
    title: str
    musicxml_path: Path
    midi_path: Path
    embedding: np.ndarray
    n_notes: int
    created_at: str


_DDL = """
CREATE TABLE IF NOT EXISTS primitives (
    primitive_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    source_exercise_number INTEGER NOT NULL,
    title TEXT NOT NULL,
    musicxml_path TEXT NOT NULL,
    midi_path TEXT NOT NULL,
    embedding BLOB NOT NULL,
    n_notes INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
"""


def write_primitives(
    primitives: list,
    embeddings: dict[str, torch.Tensor],
    db_path: Path,
) -> None:
    """Write primitives and their embeddings to the SQLite catalog.

    Args:
        primitives: list of Primitive dataclass instances.
        embeddings: dict mapping primitive_id to 512-dim torch.Tensor.
        db_path: path to the SQLite database file (created if absent).

    Raises:
        KeyError: if a primitive_id has no entry in embeddings.
        ValueError: if an embedding tensor is not 1-D.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(_DDL)
        for p in primitives:
            if p.primitive_id not in embeddings:
                raise KeyError(
                    f"No embedding found for primitive_id={p.primitive_id!r}"
                )
            emb = embeddings[p.primitive_id]
            if emb.ndim != 1:
                raise ValueError(
                    f"Embedding for {p.primitive_id!r} must be 1-D, got shape {emb.shape}"
                )
            emb_blob = emb.numpy().astype(np.float32).tobytes()
            conn.execute(
                """
                INSERT OR REPLACE INTO primitives
                (primitive_id, source, source_exercise_number, title,
                 musicxml_path, midi_path, embedding, n_notes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    p.primitive_id,
                    p.source,
                    p.source_exercise_number,
                    p.title,
                    str(p.musicxml_path),
                    str(p.midi_path),
                    emb_blob,
                    p.n_notes,
                    now,
                ),
            )
        conn.commit()
    finally:
        conn.close()


def read_primitives(db_path: Path) -> list[CatalogRow]:
    """Read all primitives from the SQLite catalog.

    Args:
        db_path: path to the SQLite database file.

    Returns:
        List of CatalogRow instances ordered by source, source_exercise_number.

    Raises:
        FileNotFoundError: if db_path does not exist.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Catalog database not found: {db_path}")
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.execute(
            """
            SELECT primitive_id, source, source_exercise_number, title,
                   musicxml_path, midi_path, embedding, n_notes, created_at
            FROM primitives
            ORDER BY source, source_exercise_number
            """
        )
        rows = []
        for row in cursor.fetchall():
            emb = np.frombuffer(row[6], dtype=np.float32).copy()
            rows.append(
                CatalogRow(
                    primitive_id=row[0],
                    source=row[1],
                    source_exercise_number=row[2],
                    title=row[3],
                    musicxml_path=Path(row[4]),
                    midi_path=Path(row[5]),
                    embedding=emb,
                    n_notes=row[7],
                    created_at=row[8],
                )
            )
        return rows
    finally:
        conn.close()
