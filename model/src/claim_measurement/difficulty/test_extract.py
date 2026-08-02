"""Tests for the backbone-agnostic extraction orchestrator.

Run: cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
"""
from claim_measurement.difficulty.backbone import FakeBackbone
from claim_measurement.difficulty.bakeoff_npz import read_embedding_npz
from claim_measurement.difficulty.bakeoff_sampling import ManifestEntry
from claim_measurement.difficulty.extract import extract_embeddings


def test_extract_embeddings_with_fake_backbone_produces_conformant_npz(tmp_path):
    entries = [
        ManifestEntry(seg_id="a", key="A.mid", grade=3, composer="Czerny"),
        ManifestEntry(seg_id="b", key="B.mid", grade=7, composer="Bach"),
    ]
    backbone = FakeBackbone(pooling_names=("mean_pool", "last_token"), dim=4)
    out_dir = tmp_path / "emb"
    index_path = tmp_path / "composer_index.json"

    report = extract_embeddings(backbone, entries, midi_dir=tmp_path / "mid",
                                 out_dir=out_dir, composer_index_path=index_path)

    assert report.ok == 2
    assert report.failed == []
    rec_a = read_embedding_npz(out_dir / "a.npz")
    assert set(rec_a.embeddings) == {"mean_pool", "last_token"}
    assert rec_a.embeddings["mean_pool"].shape == (4,)
    assert rec_a.grade == 3
    rec_b = read_embedding_npz(out_dir / "b.npz")
    assert rec_a.composer_id != rec_b.composer_id


def test_extract_embeddings_records_failures_and_continues(tmp_path):
    class BrokenOnB:
        def embed(self, midi_path):
            if midi_path.stem == "b":
                raise RuntimeError("simulated corrupt MIDI")
            return {"embedding": __import__("numpy").zeros(3, dtype="float32")}

    entries = [
        ManifestEntry(seg_id="a", key="A.mid", grade=1, composer="Liszt"),
        ManifestEntry(seg_id="b", key="B.mid", grade=2, composer="Chopin"),
    ]
    out_dir = tmp_path / "emb"

    report = extract_embeddings(BrokenOnB(), entries, midi_dir=tmp_path / "mid",
                                 out_dir=out_dir, composer_index_path=tmp_path / "idx.json")

    assert report.ok == 1
    assert len(report.failed) == 1
    assert "b" in report.failed[0]
    assert (out_dir / "a.npz").exists()
    assert not (out_dir / "b.npz").exists()
