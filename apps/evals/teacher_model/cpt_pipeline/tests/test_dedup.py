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


def test_3b_strips_within_doc_repeated_lines(tmp_path):
    manifest_in = tmp_path / "in.jsonl"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    repeat_line = "These are some lengthy disclaimer words that repeat across pages."
    text = (
        "Real article content begins here in the first paragraph. "
        "Discussing pedagogy in detail.\n"
        f"{repeat_line}\n"
        "Section 1 body content goes here with several distinct sentences.\n"
        f"{repeat_line}\n"
        "Section 2 body content with distinct words.\n"
        f"{repeat_line}\n"
        "Section 3 body content with distinct words.\n"
        f"{repeat_line}\n"
        "Conclusion of the article in this final paragraph.\n"
    )
    _write_manifest(manifest_in, [
        {"doc_id": "REPEATSABCDX", "source": "youtube:tonebase", "text": text, "word_count": 80},
    ])

    manifest_out = run_dedup(manifest_in, out_dir)

    surviving = _read_jsonl(manifest_out)
    assert len(surviving) == 1
    out_text = surviving[0]["text"]
    assert out_text.count(repeat_line) == 1, \
        f"expected exactly 1 occurrence of repeated line, got {out_text.count(repeat_line)}: {out_text!r}"
    assert "Section 1 body content" in out_text and "Conclusion" in out_text, \
        "surrounding text was incorrectly stripped"


def test_3c_drops_lines_in_more_than_20_docs(tmp_path):
    manifest_in = tmp_path / "in.jsonl"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    boilerplate = "This is a sufficiently long boilerplate disclaimer line for stripping."
    rows = []
    for i in range(25):
        text = (
            f"Doc {i} unique body content here with distinct words and pedagogy detail. "
            f"More distinct content for doc {i}.\n"
            f"{boilerplate}\n"
            f"Final unique body content for doc {i} with more distinct words.\n"
        )
        rows.append({"doc_id": f"D{i:010d}A", "source": "youtube:tonebase", "text": text, "word_count": 30})
    _write_manifest(manifest_in, rows)

    manifest_out = run_dedup(manifest_in, out_dir)

    surviving = _read_jsonl(manifest_out)
    assert len(surviving) == 25, "no doc-level dups expected; all 25 should survive"
    assert all(boilerplate not in r["text"] for r in surviving), \
        "boilerplate line should be stripped from every doc"
