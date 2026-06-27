# model/tests/exercise_corpus/test_build_render_assets.py
"""build() must materialize every committed primitive .xml as a valid MXL ZIP
whose inner MusicXML preserves the source note count."""

import glob
import io
import zipfile
from pathlib import Path

import partitura
from partitura.score import Note

from exercise_corpus.build_render_assets import build

_XML_DIR = Path(__file__).resolve().parents[2] / "data" / "scores" / "exercise_primitives"
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


def test_build_emits_valid_mxl_with_matching_note_count(tmp_path: Path):
    out_dir = tmp_path / "mxl"
    produced = build(xml_dir=_XML_DIR, out_dir=out_dir)

    xml_files = sorted(glob.glob(str(_XML_DIR / "*.xml")))
    assert len(produced) == len(xml_files) == 22

    for xml_path_str in xml_files:
        xml_path = Path(xml_path_str)
        mxl_path = out_dir / f"{xml_path.stem}.mxl"
        assert mxl_path.exists(), f"missing asset for {xml_path.stem}"

        mxl_bytes = mxl_path.read_bytes()
        assert mxl_bytes[:4] == _ZIP_MAGIC, f"{mxl_path.name} is not a ZIP"

        # Inner MusicXML must parse and preserve note count.
        inner = _inner_xml_bytes(mxl_bytes)
        inner_path = tmp_path / f"{xml_path.stem}_inner.musicxml"
        inner_path.write_bytes(inner)
        assert _note_count_xml(inner_path) == _note_count_xml(xml_path)

        # DOCTYPE must be stripped (Verovio WASM parser invariant).
        assert b"<!DOCTYPE" not in inner


def test_build_is_idempotent(tmp_path: Path):
    out_dir = tmp_path / "mxl"
    build(xml_dir=_XML_DIR, out_dir=out_dir)
    first = {p: (out_dir / f"{Path(p).stem}.mxl").read_bytes()
             for p in glob.glob(str(_XML_DIR / "*.xml"))}

    # Second run must not change any asset's bytes.
    build(xml_dir=_XML_DIR, out_dir=out_dir)
    for src, original_bytes in first.items():
        again = (out_dir / f"{Path(src).stem}.mxl").read_bytes()
        assert again == original_bytes, f"{Path(src).stem}.mxl changed on rebuild"


def test_build_raises_naming_bad_xml(tmp_path: Path):
    import pytest

    bad_dir = tmp_path / "src"
    bad_dir.mkdir()
    bad = bad_dir / "broken_001.xml"
    bad.write_text("<not-musicxml>this will not parse</not-musicxml>")

    with pytest.raises(ValueError, match="broken_001"):
        build(xml_dir=bad_dir, out_dir=tmp_path / "out")


# ---- appended: manifest-emission behavior (Task 1) -------------------------
# build() emits the manifest ONLY when an explicit manifest_path is provided
# (default None => no write), so the three regression tests above stay
# manifest-side-effect-free. Bar counts (29/22/23) were verified empirically
# via partitura; assert against the manifest build() actually wrote.

# The 22 primitives that have committed .mxl assets on main.
_EXPECTED_IDS = {
    *(f"hanon_{i:03d}" for i in range(1, 21)),
    "czerny_001",
    "burgmuller_001",
}


def test_build_emits_manifest_for_exactly_the_built_primitives(tmp_path: Path):
    import json

    manifest_path = tmp_path / "manifest.json"
    # Explicit manifest_path + tmp out_dir: writes ONLY to tmp, never to the
    # committed apps/api manifest or the real model/data mxl dir.
    build(xml_dir=_XML_DIR, out_dir=tmp_path / "mxl", manifest_path=manifest_path)

    assert manifest_path.exists(), f"manifest not written at {manifest_path}"
    manifest = json.loads(manifest_path.read_text())

    assert set(manifest.keys()) == _EXPECTED_IDS

    # Spot-check the three distinct sources with verbatim keys and real bar counts.
    # Bar counts verified empirically via partitura at plan time (29/22/23).
    assert manifest["hanon_001"] == {
        "dimensions": ["articulation", "timing"],
        "key": "C",
        "totalBars": 29,
    }
    assert manifest["czerny_001"] == {
        "dimensions": ["timing", "articulation"],
        "key": "c",  # VERBATIM lowercase from technique_tags.toml (not normalized)
        "totalBars": 22,
    }
    assert manifest["burgmuller_001"] == {
        "dimensions": ["phrasing", "dynamics", "interpretation"],
        "key": "C",
        "totalBars": 23,
    }
