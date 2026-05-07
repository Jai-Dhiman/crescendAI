"""Stage 4: stratified 1% per-source held-out split."""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

SMALL_SOURCE_THRESHOLD = 100


def run_split(manifest_in: Path, out_dir: Path, seed: int = 42) -> tuple[Path, Path]:
    """Stratified 1%-per-source split. Sources <100 docs -> all to train.

    Returns (train_path, validation_path).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with Path(manifest_in).open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]

    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)

    train_rows: list[dict] = []
    val_rows: list[dict] = []
    for source in sorted(by_source):
        group = by_source[source]
        group_sorted = sorted(group, key=lambda r: r["doc_id"])
        if len(group_sorted) < SMALL_SOURCE_THRESHOLD:
            train_rows.extend(group_sorted)
            continue
        n_val = len(group_sorted) // 100
        source_hash = int(hashlib.md5(source.encode()).hexdigest()[:8], 16) % 10000
        rng = random.Random(seed + source_hash)
        idxs = list(range(len(group_sorted)))
        rng.shuffle(idxs)
        val_idx_set = set(idxs[:n_val])
        for i, row in enumerate(group_sorted):
            (val_rows if i in val_idx_set else train_rows).append(row)

    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "validation.jsonl"
    with train_path.open("w", encoding="utf-8") as fh:
        for row in train_rows:
            fh.write(json.dumps(row) + "\n")
    with val_path.open("w", encoding="utf-8") as fh:
        for row in val_rows:
            fh.write(json.dumps(row) + "\n")
    return train_path, val_path
