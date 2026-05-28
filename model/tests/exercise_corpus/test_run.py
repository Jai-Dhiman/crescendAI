# model/tests/exercise_corpus/test_run.py
import subprocess
import sys
import pytest
import torch
from pathlib import Path
from unittest.mock import patch

from exercise_corpus import Primitive
from exercise_corpus.run import run_pipeline
from exercise_corpus.catalog import write_primitives


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
