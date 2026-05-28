# model/tests/exercise_corpus/test_validate.py
import json
import numpy as np
import pytest
import torch
from pathlib import Path

from exercise_corpus import Primitive
from exercise_corpus.validate import source_purity, run_validation, ValidationResult
from exercise_corpus.catalog import write_primitives


def _make_synthetic_primitives_and_embeddings(
    counts: dict[str, int], dim: int = 512
) -> tuple[list[Primitive], dict[str, torch.Tensor]]:
    """Build primitives and tight Gaussian cluster embeddings per source."""
    primitives = []
    embeddings = {}
    source_centers = {
        "hanon":      np.array([1.0] + [0.0] * (dim - 1), dtype=np.float32),
        "czerny":     np.array([0.0, 1.0] + [0.0] * (dim - 2), dtype=np.float32),
        "burgmuller": np.array([0.0, 0.0, 1.0] + [0.0] * (dim - 3), dtype=np.float32),
    }
    idx = 0
    for source, count in counts.items():
        center = source_centers[source]
        for i in range(1, count + 1):
            pid = f"{source}_{i:03d}"
            noise = np.random.default_rng(idx).normal(0, 0.01, dim).astype(np.float32)
            emb = torch.tensor(center + noise)
            primitives.append(
                Primitive(
                    primitive_id=pid,
                    source=source,
                    source_exercise_number=i,
                    title=f"{source} {i}",
                    musicxml_path=Path(f"/fake/{pid}.xml"),
                    midi_path=Path(f"/fake/{pid}.mid"),
                    n_notes=8,
                )
            )
            embeddings[pid] = emb
            idx += 1
    return primitives, embeddings


def test_purity_perfect_separation():
    # Each class has at least k+1=3 members so all k=2 neighbors are within-class
    embeddings_array = np.array([
        [1.0, 0.0, 0.0],
        [1.1, 0.0, 0.0],
        [1.2, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.1, 0.0],
        [0.0, 1.2, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.1],
        [0.0, 0.0, 1.2],
    ], dtype=np.float32)
    labels = ["a", "a", "a", "b", "b", "b", "c", "c", "c"]
    purity = source_purity(embeddings_array, labels, k=2)
    assert purity == pytest.approx(1.0)


def test_purity_shuffled_near_random_floor():
    rng = np.random.default_rng(42)
    # 60 + 40 + 25 = 125 total -- same distribution as production spec
    labels = ["hanon"] * 60 + ["czerny"] * 40 + ["burgmuller"] * 25
    # Random embeddings -> neighbors are random -> purity approaches class-size-weighted random
    embeddings_array = rng.standard_normal((125, 512)).astype(np.float32)
    purity = source_purity(embeddings_array, labels, k=5)
    # Random floor for 60/40/25 split is ~0.43; allow generous range
    assert 0.20 <= purity <= 0.70, f"Expected near-random purity, got {purity:.3f}"


def test_run_validation_emits_15_pairs(tmp_path: Path):
    # 60 hanon + 40 czerny + 25 burgmuller
    counts = {"hanon": 60, "czerny": 40, "burgmuller": 25}
    primitives, embeddings = _make_synthetic_primitives_and_embeddings(counts)

    db_path = tmp_path / "catalog.db"
    write_primitives(primitives, embeddings, db_path)

    result = run_validation(db_path, tmp_path)

    assert isinstance(result, ValidationResult)
    assert len(result.pairs) == 15
    # All 15 pairs must be within-source
    for pair in result.pairs:
        assert pair["source_a"] == pair["source_b"], (
            f"Cross-source pair found: {pair['source_a']} vs {pair['source_b']}"
        )


def test_run_validation_umap_file_created(tmp_path: Path):
    counts = {"hanon": 60, "czerny": 40, "burgmuller": 25}
    primitives, embeddings = _make_synthetic_primitives_and_embeddings(counts)
    db_path = tmp_path / "catalog.db"
    write_primitives(primitives, embeddings, db_path)

    result = run_validation(db_path, tmp_path)
    assert result.umap_path.exists()


def test_run_validation_pairs_json_is_valid(tmp_path: Path):
    counts = {"hanon": 60, "czerny": 40, "burgmuller": 25}
    primitives, embeddings = _make_synthetic_primitives_and_embeddings(counts)
    db_path = tmp_path / "catalog.db"
    write_primitives(primitives, embeddings, db_path)

    result = run_validation(db_path, tmp_path)
    assert result.pairs_path.exists()
    with open(result.pairs_path) as f:
        parsed = json.load(f)
    assert isinstance(parsed, list)
    assert len(parsed) == 15


def test_run_validation_verdict_pass_for_tight_clusters(tmp_path: Path):
    counts = {"hanon": 60, "czerny": 40, "burgmuller": 25}
    primitives, embeddings = _make_synthetic_primitives_and_embeddings(counts)
    db_path = tmp_path / "catalog.db"
    write_primitives(primitives, embeddings, db_path)

    result = run_validation(db_path, tmp_path)
    assert result.verdict == "PASS"
    assert result.purity >= 0.70
