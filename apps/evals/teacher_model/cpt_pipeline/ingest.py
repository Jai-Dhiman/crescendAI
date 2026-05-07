"""Stage 1: ingest corpus + provenance into unified manifest."""
from __future__ import annotations

import json
from pathlib import Path

from teacher_model.cpt_pipeline.source_resolver import build_provenance_index, resolve_source


def run_ingest(corpus_dir: Path, provenance_dir: Path, out_dir: Path) -> Path:
    """Walk corpus_dir for .txt files, join to provenance_dir JSONLs, emit manifest."""
    corpus_dir = Path(corpus_dir)
    provenance_dir = Path(provenance_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_out = out_dir / "manifest.jsonl"
    drops_path = out_dir / "drops.jsonl"

    index = build_provenance_index(provenance_dir)

    with manifest_out.open("w", encoding="utf-8") as out_fh, \
         drops_path.open("w", encoding="utf-8") as drops_fh:
        for path in sorted(corpus_dir.glob("*.txt")):
            doc_id = path.stem
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                drops_fh.write(json.dumps({"doc_id": doc_id, "drop_reason": "decode_error"}) + "\n")
                continue
            source = resolve_source(path.name, index)
            row = {
                "doc_id": doc_id,
                "source": source,
                "text": text,
                "word_count": len(text.split()),
            }
            out_fh.write(json.dumps(row) + "\n")
    return manifest_out
