"""Offline tests for the 37-feature bake-off arm.

Run: cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
"""
import json
from pathlib import Path

import numpy as np
import pytest

from claim_measurement.difficulty.features37_backbone import (
    N_BASE_FEATURES,
    CachedFeature37Backbone,
    extractor_fingerprint,
    load_feature37_cache,
)


def _cache_row(key: str, grade: int, composer: str, base_values: list[float]) -> dict:
    row = {"key": key, "grade": grade, "composer": composer}
    row.update({f"f{i}": v for i, v in enumerate(base_values)})
    row["tk_something"] = 99.0   # a Transkun-family column the 37 arm must exclude
    return row


def _write_cache(path: Path, rows: list[dict], sha: str) -> Path:
    path.write_text(json.dumps({"n_rows": len(rows), "extractor_sha": sha,
                                "rows": rows}))
    return path


def test_load_cache_returns_37_features_per_key_excluding_tk_columns(tmp_path):
    rows = [_cache_row("A", 3, "Bach", [float(i) for i in range(N_BASE_FEATURES)]),
            _cache_row("B", 7, "Czerny",
                       [float(i) * 2 for i in range(N_BASE_FEATURES)])]
    cache = _write_cache(tmp_path / "c.json", rows, sha="abc123")

    by_key = load_feature37_cache(cache, expected_sha="abc123")

    assert set(by_key) == {"A", "B"}
    assert by_key["A"].shape == (N_BASE_FEATURES,)
    # Cache insertion order preserved, tk_ column dropped -- same matrix as arm BASE_37.
    assert by_key["B"][0] == 0.0 and by_key["B"][-1] == float(N_BASE_FEATURES - 1) * 2


def test_load_cache_raises_when_extractor_fingerprint_moved(tmp_path):
    rows = [_cache_row("A", 3, "Bach", [1.0] * N_BASE_FEATURES)]
    cache = _write_cache(tmp_path / "c.json", rows, sha="stale00")

    with pytest.raises(ValueError, match="STALE FEATURE CACHE"):
        load_feature37_cache(cache, expected_sha="fresh11")


def test_load_cache_raises_on_empty_rows(tmp_path):
    cache = _write_cache(tmp_path / "c.json", [], sha="abc123")

    with pytest.raises(ValueError, match="no rows"):
        load_feature37_cache(cache, expected_sha="abc123")


def test_load_cache_raises_when_base_feature_count_is_not_37(tmp_path):
    rows = [_cache_row("A", 3, "Bach", [1.0] * (N_BASE_FEATURES - 1))]
    cache = _write_cache(tmp_path / "c.json", rows, sha="abc123")

    with pytest.raises(ValueError, match=f"expected {N_BASE_FEATURES} base features"):
        load_feature37_cache(cache, expected_sha="abc123")


def test_extractor_fingerprint_is_stable_and_short():
    assert extractor_fingerprint() == extractor_fingerprint()
    assert len(extractor_fingerprint()) == 16


def test_backbone_maps_seg_id_to_the_cached_vector():
    vec = np.arange(N_BASE_FEATURES, dtype=np.float32)
    backbone = CachedFeature37Backbone(by_key={"A key": vec},
                                       seg_id_to_key={"seg_a": "A key"})

    out = backbone.embed(Path("/wherever/seg_a.mid"))

    assert list(out) == ["raw37"]
    assert np.array_equal(out["raw37"], vec)


def test_backbone_raises_on_a_seg_id_missing_from_the_manifest_map():
    backbone = CachedFeature37Backbone(by_key={}, seg_id_to_key={})

    # Loud KeyError, not a zero vector: a silently-imputed row would corrupt the
    # baseline this arm exists to establish.
    with pytest.raises(KeyError):
        backbone.embed(Path("/wherever/unknown.mid"))


def test_features37_stage_writes_one_npz_per_sampled_piece(tmp_path, capsys):
    from claim_measurement.difficulty.bakeoff_npz import read_embedding_npz
    from claim_measurement.difficulty.run_bakeoff import main

    sample = [{"seg_id": "seg_a", "key": "A", "grade": 3, "composer": "Bach"},
              {"seg_id": "seg_b", "key": "B", "grade": 7, "composer": "Czerny"}]
    bakeoff = tmp_path / "results" / "bakeoff"
    bakeoff.mkdir(parents=True)
    (bakeoff / "sample_manifest.json").write_text(json.dumps(sample))
    rows = [_cache_row("A", 3, "Bach", [1.0] * N_BASE_FEATURES),
            _cache_row("B", 7, "Czerny", [2.0] * N_BASE_FEATURES)]
    (tmp_path / "results").mkdir(exist_ok=True)
    _write_cache(tmp_path / "results" / "mirex_137_tk_features.json", rows,
                 sha=extractor_fingerprint())

    exit_code = main(["--stage", "features37", "--data-root", str(tmp_path)])

    assert exit_code == 0
    assert "ok=2" in capsys.readouterr().out
    record = read_embedding_npz(bakeoff / "emb" / "features37" / "seg_a.npz")
    assert record.grade == 3
    assert record.embeddings["raw37"].shape == (N_BASE_FEATURES,)
