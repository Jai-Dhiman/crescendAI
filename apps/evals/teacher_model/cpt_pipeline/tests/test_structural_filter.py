"""Structural filter behavior tests."""
import json
from pathlib import Path

from teacher_model.cpt_pipeline.structural_filter import run_filter


def _write_manifest(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_drops_doc_below_min_chars(tmp_path):
    manifest_in = tmp_path / "in.jsonl"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _write_manifest(manifest_in, [
        {"doc_id": "long", "source": "youtube:tonebase", "text": "x" * 200, "word_count": 1},
        {"doc_id": "short", "source": "youtube:tonebase", "text": "Too short.", "word_count": 2},
    ])

    manifest_out = run_filter(manifest_in, out_dir)

    surviving = _read_jsonl(manifest_out)
    assert [r["doc_id"] for r in surviving] == ["long"]
    drops = _read_jsonl(out_dir / "drops.jsonl")
    assert any(d["doc_id"] == "short" and d["drop_reason"] == "too_short" for d in drops), \
        f"expected short doc dropped with reason too_short, got {drops}"
