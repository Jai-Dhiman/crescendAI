# model/tests/exercise_corpus/test_build_render_assets.py
"""build() materializes one renderable asset per drill (tiered) + the API manifest.

Tier A (committed MusicXML) is exercised hermetically -- the .xml fixtures are
committed. Tier B (Chopin etude **kern) and Tier C (Mutopia MIDI) depend on
gitignored/regenerable corpus assets (data/midi/..., ~/crescendai_corpus_staging),
so those tests SKIP when the source isn't present rather than fail in a bare CI.
"""

import io
import json
import zipfile
from pathlib import Path

import partitura
import pytest
from partitura.score import Note

from exercise_corpus.build_render_assets import (
    _build_tier_a,
    _build_tier_c,
    _etude_krn_index,
    _DEFAULT_KERN_REPO,
    _DEFAULT_MIDI_DIR,
    build,
)

_MODEL_ROOT = Path(__file__).resolve().parents[2]
_XML_DIR = _MODEL_ROOT / "data" / "scores" / "exercise_primitives"
_EMBED_MANIFEST = _MODEL_ROOT / "data" / "embed_ready_manifest.json"
_ZIP_MAGIC = b"PK\x03\x04"


def _note_count_xml(xml_path: Path) -> int:
    score = partitura.load_score(str(xml_path))
    return sum(1 for _ in list(score.parts)[0].iter_all(Note))


def _inner_xml_bytes(mxl_bytes: bytes) -> bytes:
    with zipfile.ZipFile(io.BytesIO(mxl_bytes)) as zf:
        for name in zf.namelist():
            if not name.startswith("META-INF") and name.endswith(".xml"):
                return zf.read(name)
    raise AssertionError("no MusicXML entry inside MXL ZIP")


def _subset_embed_manifest(tmp_path: Path, ids: set[str]) -> Path:
    """A small embed_ready_manifest containing only `ids`, for fast hermetic build()
    orchestration tests that don't need the whole 154-drill corpus."""
    rows = [
        p
        for p in json.loads(_EMBED_MANIFEST.read_text())["primitives"]
        if p["primitive_id"] in ids
    ]
    path = tmp_path / "embed.json"
    path.write_text(json.dumps({"schema": "test", "n_primitives": len(rows), "primitives": rows}))
    return path


# ---- Tier A: committed MusicXML -> MXL (hermetic) --------------------------


def test_tier_a_emits_valid_mxl_with_matching_note_count(tmp_path: Path):
    xml_path = _XML_DIR / "hanon_001.xml"
    assert xml_path.exists(), "tier-A .xml fixtures are committed"

    asset, bars = _build_tier_a("hanon_001", xml_path, tmp_path)
    assert asset.name == "hanon_001.mxl"
    assert bars == 29

    mxl_bytes = asset.read_bytes()
    assert mxl_bytes[:4] == _ZIP_MAGIC
    inner = _inner_xml_bytes(mxl_bytes)
    inner_path = tmp_path / "hanon_001_inner.musicxml"
    inner_path.write_bytes(inner)
    assert _note_count_xml(inner_path) == _note_count_xml(xml_path)
    assert b"<!DOCTYPE" not in inner  # Verovio WASM parser invariant


# ---- Tier C: score-derived MIDI -> partitura MusicXML -> MXL (render-gated) -


def test_tier_c_emits_render_gated_mxl(tmp_path: Path):
    midi = _DEFAULT_MIDI_DIR / "satie_001.mid"
    if not midi.exists():
        pytest.skip(f"tier-C MIDI corpus absent ({midi}); run corpus acquire recipes")

    asset, bars = _build_tier_c("satie_001", midi, tmp_path)
    assert asset.name == "satie_001.mxl"
    assert bars >= 1
    mxl_bytes = asset.read_bytes()
    assert mxl_bytes[:4] == _ZIP_MAGIC
    inner = _inner_xml_bytes(mxl_bytes)
    assert b"<!DOCTYPE" not in inner


# ---- Tier B: etude .krn index mapping (skips without staging repo) ---------


def test_etude_krn_index_maps_position_to_exercise_number():
    if not _DEFAULT_KERN_REPO.exists():
        pytest.skip(f"etude kern repo absent ({_DEFAULT_KERN_REPO})")
    files = _etude_krn_index(_DEFAULT_KERN_REPO)
    assert len(files) == 24
    # chopin_etude_001 (source_exercise_number 1) maps to Op.10 No.1's first edition.
    assert files[0].name == "010-1b-Sm-001.krn"
    # chopin_etude_013 (Op.25 No.1) is the 13th in sort order.
    assert files[12].name == "025-1b-LE-001.krn"


# ---- build() orchestration + manifest emission ----------------------------


def test_build_subset_emits_manifest_for_built_primitives(tmp_path: Path):
    # hanon_001 (tier A) + satie_001 (tier C) -- no kern repo needed.
    if not (_DEFAULT_MIDI_DIR / "satie_001.mid").exists():
        pytest.skip("tier-C MIDI corpus absent; run corpus acquire recipes")

    embed = _subset_embed_manifest(tmp_path, {"hanon_001", "satie_001"})
    manifest_path = tmp_path / "manifest.json"
    produced = build(
        out_dir=tmp_path / "assets",
        embed_manifest=embed,
        manifest_path=manifest_path,
    )
    assert len(produced) == 2

    manifest = json.loads(manifest_path.read_text())
    assert set(manifest.keys()) == {"hanon_001", "satie_001"}
    # Tier-A bar count + verbatim tags (regression vs the old 22-primitive manifest).
    assert manifest["hanon_001"] == {
        "dimensions": ["articulation", "timing"],
        "key": "C",
        "totalBars": 29,
    }
    # Tier-C drill carries a PEDALING dimension the old corpus could never route to.
    assert "pedaling" in manifest["satie_001"]["dimensions"]
    assert manifest["satie_001"]["totalBars"] >= 1


def test_build_subset_is_idempotent(tmp_path: Path):
    if not (_DEFAULT_MIDI_DIR / "satie_001.mid").exists():
        pytest.skip("tier-C MIDI corpus absent; run corpus acquire recipes")

    embed = _subset_embed_manifest(tmp_path, {"hanon_001", "satie_001"})
    out = tmp_path / "assets"
    build(out_dir=out, embed_manifest=embed)
    first = {p.name: p.read_bytes() for p in out.glob("*.mxl")}

    build(out_dir=out, embed_manifest=embed)
    for name, original in first.items():
        assert (out / name).read_bytes() == original, f"{name} changed on rebuild"


def test_build_raises_on_primitive_missing_technique_tag(tmp_path: Path):
    embed = tmp_path / "embed.json"
    embed.write_text(
        json.dumps(
            {
                "schema": "test",
                "n_primitives": 1,
                "primitives": [
                    {"primitive_id": "ghost_999", "source": "ghost", "source_exercise_number": 1}
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="ghost_999"):
        build(out_dir=tmp_path / "assets", embed_manifest=embed)
