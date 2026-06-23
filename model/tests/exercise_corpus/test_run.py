# model/tests/exercise_corpus/test_run.py
import json
import subprocess
import sys
import pytest
import torch
import partitura
from partitura.score import Note, Part, TimeSignature
from pathlib import Path

from exercise_corpus import Primitive
from exercise_corpus.run import run_pipeline
from exercise_corpus.catalog import write_primitives


def _write_midi(path: Path, n_notes: int = 4) -> None:
    """Write a minimal valid MIDI of n_notes quarter notes (per_file source input)."""
    part = Part("P0", "test", quarter_duration=1)
    part.add(TimeSignature(4, 4), 0)
    for i in range(n_notes):
        part.add(Note(step="C", octave=4, voice=1), start=i, end=i + 1)
    partitura.score.add_measures(part)
    path.parent.mkdir(parents=True, exist_ok=True)
    partitura.save_score_midi(part, str(path))


def _make_primitive(source: str, n: int) -> Primitive:
    return Primitive(
        primitive_id=f"{source}_{n:03d}",
        source=source,
        source_exercise_number=n,
        title=f"{source} {n}",
        musicxml_path=Path(f"/fake/{source}_{n:03d}.xml"),
        midi_path=Path(f"/fake/{source}_{n:03d}.mid"),
        n_notes=8,
    )


def test_run_pipeline_validate_only_requires_db_path():
    with pytest.raises(ValueError, match="db_path is required"):
        run_pipeline(validate_only=True, db_path=None, output_dir=Path("/tmp/out"))


def test_run_pipeline_validate_only_requires_output_dir(tmp_path: Path):
    db = tmp_path / "exercise_primitives.db"
    with pytest.raises(ValueError, match="output_dir is required"):
        run_pipeline(validate_only=True, db_path=db, output_dir=None)


def test_run_pipeline_requires_sources_path():
    with pytest.raises(ValueError, match="sources_path is required"):
        run_pipeline(validate_only=False, sources_path=None, output_dir=Path("/tmp/out"))


def test_run_pipeline_requires_output_dir(tmp_path: Path):
    sources = tmp_path / "sources.toml"
    sources.write_text('[[sources]]\n')
    with pytest.raises(ValueError, match="output_dir is required"):
        run_pipeline(validate_only=False, sources_path=sources, output_dir=None)


def test_cli_help_exits_zero():
    result = subprocess.run(
        [sys.executable, "-m", "exercise_corpus.run", "--help"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parents[2]),  # model/
    )
    assert result.returncode == 0


def test_run_pipeline_dry_run_raises_on_missing_musicxml(tmp_path: Path):
    # sources.toml points to musicxml_path that does not exist
    sources_toml = tmp_path / "sources.toml"
    sources_toml.write_text(
        '[[sources]]\n'
        'name = "hanon"\n'
        'title = "Hanon"\n'
        'composer = "Hanon"\n'
        'opus = ""\n'
        'license = "public_domain"\n'
        f'musicxml_path = "{tmp_path / "missing.xml"}"\n'
    )
    with pytest.raises(FileNotFoundError, match="missing.xml"):
        run_pipeline(sources_toml, tmp_path, dry_run=True)


def test_run_pipeline_validate_only_reads_existing_catalog(tmp_path: Path):
    # Populate a catalog directly (bypasses segment + embed)
    db_path = tmp_path / "exercise_primitives.db"
    counts = {"hanon": 60, "czerny": 40, "burgmuller": 25}
    primitives = []
    embeddings = {}
    import numpy as np
    source_centers = {
        "hanon":      np.array([1.0] + [0.0] * 511, dtype=np.float32),
        "czerny":     np.array([0.0, 1.0] + [0.0] * 510, dtype=np.float32),
        "burgmuller": np.array([0.0, 0.0, 1.0] + [0.0] * 509, dtype=np.float32),
    }
    idx = 0
    for source, count in counts.items():
        for i in range(1, count + 1):
            p = _make_primitive(source, i)
            noise = np.random.default_rng(idx).normal(0, 0.01, 512).astype(np.float32)
            emb = torch.tensor(source_centers[source] + noise)
            primitives.append(p)
            embeddings[p.primitive_id] = emb
            idx += 1
    write_primitives(primitives, embeddings, db_path)

    result = run_pipeline(
        validate_only=True,
        db_path=db_path,
        output_dir=tmp_path,
    )
    assert result.purity >= 0.70
    assert result.verdict == "PASS"


def test_segment_only_writes_manifest_and_skips_embed(tmp_path: Path):
    # #49 boundary: segment_only stages per-primitive MIDI + the embed-ready
    # manifest (with source-level coarse tags), then STOPS before the Aria embed.
    src = tmp_path / "raw" / "joplin_rags"
    _write_midi(src / "rag_a.mid")
    _write_midi(src / "rag_b.mid")
    sources_toml = tmp_path / "sources.toml"
    sources_toml.write_text(
        '[[sources]]\n'
        'name = "joplin_rags"\n'
        'license = "public_domain"\n'
        f'musicxml_path = "{src}"\n'
        'dimensions = ["timing", "articulation"]\n'
        'techniques = ["stride-bass", "syncopation"]\n'
    )

    result = run_pipeline(
        sources_path=sources_toml,
        output_dir=tmp_path,
        segment_only=True,
    )
    # segment_only returns None (no validation) and writes NO catalog db.
    assert result is None
    assert not (tmp_path / "exercise_primitives.db").exists()

    manifest_path = tmp_path / "embed_ready_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema"] == "exercise_corpus.embed_ready.v1"
    assert manifest["n_primitives"] == 2
    rows = {r["primitive_id"]: r for r in manifest["primitives"]}
    assert set(rows) == {"joplin_001", "joplin_002"}
    row = rows["joplin_001"]
    assert row["source"] == "joplin_rags"
    assert row["source_dimensions"] == ["timing", "articulation"]
    assert row["source_techniques"] == ["stride-bass", "syncopation"]
    # The staged MIDI the embed job will read must actually exist on disk.
    assert Path(row["midi_path"]).exists()
