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
    long_text = (
        "Slow practice is the foundation of all piano technique. "
        "Begin every session with scales at a comfortable tempo before increasing speed. "
        "The objective is evenness across all fingers and consistent tone production."
    )
    _write_manifest(manifest_in, [
        {"doc_id": "long", "source": "youtube:tonebase", "text": long_text, "word_count": 30},
        {"doc_id": "short", "source": "youtube:tonebase", "text": "Too short.", "word_count": 2},
    ])

    manifest_out = run_filter(manifest_in, out_dir)

    surviving = _read_jsonl(manifest_out)
    assert [r["doc_id"] for r in surviving] == ["long"]
    drops = _read_jsonl(out_dir / "drops.jsonl")
    assert any(d["doc_id"] == "short" and d["drop_reason"] == "too_short" for d in drops), \
        f"expected short doc dropped with reason too_short, got {drops}"


def test_drops_doc_above_non_ascii_ratio(tmp_path):
    manifest_in = tmp_path / "in.jsonl"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    text_high_nonascii = "ascii prefix " + ("中文" * 100)  # heavily Chinese
    ok_text = (
        "Slow practice is the foundation of all piano technique. "
        "Begin every session with scales at a comfortable tempo before increasing speed. "
        "The objective is evenness across all fingers and consistent tone production."
    )
    _write_manifest(manifest_in, [
        {"doc_id": "ok", "source": "youtube:tonebase", "text": ok_text, "word_count": 30},
        {"doc_id": "nonascii", "source": "youtube:tonebase", "text": text_high_nonascii, "word_count": 100},
    ])

    manifest_out = run_filter(manifest_in, out_dir)

    surviving = _read_jsonl(manifest_out)
    assert [r["doc_id"] for r in surviving] == ["ok"]
    drops = _read_jsonl(out_dir / "drops.jsonl")
    assert any(d["doc_id"] == "nonascii" and d["drop_reason"] == "non_ascii_ratio" for d in drops)


def test_drops_non_english_doc(tmp_path):
    manifest_in = tmp_path / "in.jsonl"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    french_text = (
        "La pratique du piano est un art exigeant qui demande de la patience. "
        "Les exercices de Hanon developpent la force des doigts. "
        "Chaque journee de pratique doit commencer par un echauffement progressif."
    )
    english_text = (
        "Slow practice is the foundation of all technique work. Begin every "
        "session with five minutes of metronome scales. The objective is evenness."
    )
    _write_manifest(manifest_in, [
        {"doc_id": "en", "source": "youtube:tonebase", "text": english_text, "word_count": 25},
        {"doc_id": "fr", "source": "youtube:tonebase", "text": french_text, "word_count": 30},
    ])

    manifest_out = run_filter(manifest_in, out_dir)

    surviving = _read_jsonl(manifest_out)
    assert [r["doc_id"] for r in surviving] == ["en"]
    drops = _read_jsonl(out_dir / "drops.jsonl")
    assert any(d["doc_id"] == "fr" and d["drop_reason"] == "non_english" for d in drops)
