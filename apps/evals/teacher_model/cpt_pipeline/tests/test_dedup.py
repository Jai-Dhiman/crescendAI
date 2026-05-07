"""Dedup behavior tests."""
import json
from pathlib import Path

from teacher_model.cpt_pipeline.dedup import run_dedup


def _write_manifest(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_3a_collapses_doc_level_near_dups(tmp_path):
    manifest_in = tmp_path / "in.jsonl"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    base = (
        "Slow practice is the foundation of all technique work. Begin every "
        "session with five minutes of metronome scales at quarter equals 60. The "
        "objective is not speed but evenness, articulation, and tonal control. "
        "Listen for unevenness in the weaker fingers and isolate problem groups."
    )
    _write_manifest(manifest_in, [
        {"doc_id": "DUP1234567x", "source": "youtube:tonebase", "text": base, "word_count": 60},
        {"doc_id": "DUP1234567y", "source": "youtube:tonebase", "text": base + " Additional tail.", "word_count": 62},
        {"doc_id": "OTHER12345A", "source": "youtube:tonebase", "text": "Wholly different content here about voicing in chamber music with eight to ten distinct sentences.", "word_count": 20},
    ])

    manifest_out = run_dedup(manifest_in, out_dir)

    surviving = _read_jsonl(manifest_out)
    surviving_ids = sorted(r["doc_id"] for r in surviving)
    assert "DUP1234567x" in surviving_ids, "alphabetically-first dup should survive"
    assert "DUP1234567y" not in surviving_ids, "alphabetically-later dup should be removed"
    assert "OTHER12345A" in surviving_ids, "non-dup unrelated doc must remain"
