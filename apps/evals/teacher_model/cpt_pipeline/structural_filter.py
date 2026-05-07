"""Stage 2: structural filter for cpt_pipeline."""
from __future__ import annotations

import json
from pathlib import Path

MIN_CHARS = 100


def run_filter(manifest_in: Path, out_dir: Path) -> Path:
    """Drop docs failing structural gates; emit surviving docs and drops sidecar.

    Returns path to the output manifest.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_out = out_dir / "manifest.jsonl"
    drops_path = out_dir / "drops.jsonl"

    with Path(manifest_in).open(encoding="utf-8") as fh, \
         manifest_out.open("w", encoding="utf-8") as out_fh, \
         drops_path.open("w", encoding="utf-8") as drops_fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get("text", "")
            if len(text) < MIN_CHARS:
                drops_fh.write(json.dumps({"doc_id": row["doc_id"], "drop_reason": "too_short"}) + "\n")
                continue
            out_fh.write(json.dumps(row) + "\n")
    return manifest_out
