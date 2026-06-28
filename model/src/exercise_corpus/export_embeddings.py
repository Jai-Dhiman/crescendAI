"""Export the catalog's Aria embedding matrix as a Worker-importable JSON asset.

The cosine selector (corpus-drill.ts) needs the 154-drill catalog embeddings at
serve time to rank against a query embedding of the student's weak passage. The
Worker has no torch and cannot read the SQLite catalog, so the matrix is exported
once -- L2-normalized, aligned to the manifest's primitive ids -- into a JSON
asset bundled with the API.

The matrix is L2-normalized here so the Worker's cosine reduces to a dot product.
Rows are ordered to MATCH the catalog read order; the `ids` array is the alignment
key. Validates that the catalog ids exactly equal the manifest ids (a drill the
selector can pick but cannot embed -- or vice versa -- is a build error).

Run: just export-exercise-embeddings
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

_MODEL_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DB = _MODEL_ROOT / "model" / "data" / "exercise_primitives.db"
_DEFAULT_MANIFEST = (
    _MODEL_ROOT / "apps" / "api" / "src" / "services" / "exercise_primitives_manifest.json"
)
_DEFAULT_OUT = (
    _MODEL_ROOT / "apps" / "api" / "src" / "services" / "exercise_embeddings.json"
)
EMBED_DIM = 512


def export(
    db_path: Path = _DEFAULT_DB,
    manifest_path: Path = _DEFAULT_MANIFEST,
    out_path: Path = _DEFAULT_OUT,
) -> Path:
    """Write the L2-normalized catalog matrix + id list as JSON. Returns out_path."""
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(
            f"catalog db not found: {db_path}. Regenerate via the corpus embed pipeline."
        )
    manifest = json.loads(Path(manifest_path).read_text())

    con = sqlite3.connect(str(db_path))
    try:
        rows = con.execute(
            "SELECT primitive_id, embedding FROM primitives ORDER BY primitive_id"
        ).fetchall()
    finally:
        con.close()

    ids = [r[0] for r in rows]
    if set(ids) != set(manifest):
        only_db = sorted(set(ids) - set(manifest))
        only_man = sorted(set(manifest) - set(ids))
        raise ValueError(
            "catalog/manifest id mismatch -- "
            f"in db not manifest: {only_db[:5]}; in manifest not db: {only_man[:5]}"
        )

    mat = np.stack(
        [np.frombuffer(r[1], dtype=np.float32) for r in rows], axis=0
    ).astype(np.float32)
    if mat.shape != (len(ids), EMBED_DIM):
        raise ValueError(f"expected ({len(ids)}, {EMBED_DIM}) matrix, got {mat.shape}")

    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("a catalog embedding has zero magnitude; cannot normalize")
    mat = mat / norms

    payload = {
        "dim": EMBED_DIM,
        "ids": ids,
        # round to 6 dp: keeps the asset compact without perceptibly changing cosine.
        "vectors": [[round(float(x), 6) for x in row] for row in mat],
    }
    out_path = Path(out_path)
    out_path.write_text(json.dumps(payload) + "\n")
    return out_path


if __name__ == "__main__":
    p = export()
    print(f"wrote {p} ({p.stat().st_size // 1024} KiB)")
