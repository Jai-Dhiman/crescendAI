# model/src/exercise_corpus/build_render_assets.py
"""Build committed renderable assets for EVERY exercise-corpus drill, plus the
API manifest that the teacher routes over.

The corpus is enumerated from ``data/embed_ready_manifest.json`` (the authoritative
154-drill roster) cross-referenced with ``technique_tags.toml`` (the authoritative
per-primitive dimensions + key) -- NOT from the handful of committed ``*.xml`` (only
the 22 #17 slice-A primitives ever had MusicXML). Each drill gets ONE renderable
asset, tiered by available source fidelity:

  Tier A -- committed MusicXML (``data/scores/exercise_primitives/{id}.xml``):
            wrap as an MXL ZIP. The 20 Hanon + czerny_001 + burgmuller_001.
  Tier B -- Chopin etudes (``chopin_etude_*``): render the ORIGINAL public-domain
            **kern (humdrum-chopin-first-editions) DIRECTLY to Verovio-native MEI.
            Cleanest engraving; ``getPieceData`` prefers ``.mei``. 24 drills.
  Tier C -- everything else (Mutopia prebuilt MIDI, no clean MusicXML source):
            partitura ``load_score_midi`` -> ``save_musicxml`` -> MXL ZIP. The
            source MIDI is score-derived (quantized, metered), so the engraving is
            legible though not publication-grade. ~108 drills.

Every asset passes a Verovio render gate before it counts (tier B inherently;
tier C via an explicit ``loadData`` + ``getPageCount`` check) -- a manifest entry
without a renderable asset is the "blank score card" failure mode, so we fail loud
rather than emit one. ``totalBars`` is the measure count of the ENGRAVED asset
(MEI for tier B, the partitura score for A/C), so the clip span ``[1, totalBars]``
matches what the student sees.

Deterministic and idempotent: an MXL whose inner XML already equals the freshly
produced XML is left untouched; an existing ``.mei`` is reused (Verovio output is
not byte-stable across versions, so we skip-if-exists rather than rewrite). Any
source that fails to load/render RAISES naming the offending primitive -- explicit
exceptions over silent fallbacks (CLAUDE.md).

Run:  just build-exercise-assets   (cd model && build(manifest_path=<api manifest>))
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import tomllib
import warnings
import zipfile
from pathlib import Path

import partitura

from score_library.upload import _strip_doctype, wrap_as_mxl_zip

_logger = logging.getLogger(__name__)

# Anchor to this module, never CWD (CLAUDE.md: just recipes shift CWD).
_MODEL_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_XML_DIR = _MODEL_ROOT / "data" / "scores" / "exercise_primitives"
_DEFAULT_MIDI_DIR = _MODEL_ROOT / "data" / "midi" / "exercise_primitives"
_DEFAULT_OUT_DIR = _MODEL_ROOT / "data" / "exercise_primitives" / "assets"
_DEFAULT_EMBED_MANIFEST = _MODEL_ROOT / "data" / "embed_ready_manifest.json"
_TECHNIQUE_TAGS = Path(__file__).resolve().parent / "technique_tags.toml"

# Tier B source: the public-domain Chopin first-editions **kern, sorted exactly as
# acquire_chopin_etudes.sh sorts them so chopin_etude_{i:03d} maps to the same .krn
# the drill's MIDI was rendered from (source_exercise_number == i).
_STAGING_ROOT = Path(
    os.environ.get("CRESCENDAI_CORPUS_STAGING", str(Path.home() / "crescendai_corpus_staging"))
)
_DEFAULT_KERN_REPO = _STAGING_ROOT / "humdrum-chopin-first-editions" / "kern"
_ETUDE_PREFIX = "chopin_etude_"
_ETUDE_KRN_RE = re.compile(r"0(10|25)-1b-")
_MEI_MEASURE_RE = re.compile(r"<measure\b")


def _existing_inner_xml(mxl_path: Path) -> bytes | None:
    """Return the inner MusicXML bytes of an existing .mxl, or None if absent or
    unreadable -- used for the idempotent skip-if-unchanged check."""
    if not mxl_path.exists():
        return None
    try:
        with zipfile.ZipFile(io.BytesIO(mxl_path.read_bytes())) as zf:
            for name in zf.namelist():
                if not name.startswith("META-INF") and name.endswith(".xml"):
                    return zf.read(name)
    except zipfile.BadZipFile:
        _logger.warning(f"corrupted existing .mxl, will rebuild: {mxl_path}")
        return None
    return None


def _measures_in_score(score: "partitura.score.Score") -> int:
    """Measure count of a partitura score's first part."""
    part = score.parts[0]
    return len(list(part.iter_all(partitura.score.Measure)))


def _write_mxl(raw_xml: bytes, primitive_id: str, out_dir: Path) -> Path:
    """Idempotently write {id}.mxl from MusicXML bytes. wrap_as_mxl_zip owns
    DOCTYPE stripping (sole owner), so we compare against the singly-stripped
    source it would produce."""
    mxl_path = out_dir / f"{primitive_id}.mxl"
    if _existing_inner_xml(mxl_path) != _strip_doctype(raw_xml):
        mxl_path.write_bytes(wrap_as_mxl_zip(raw_xml, primitive_id))
    return mxl_path


def _render_gate_musicxml(xml: bytes, primitive_id: str) -> None:
    """Fail loud if Verovio's MusicXML importer cannot render this XML. Tier C's
    partitura->MusicXML intermediate is the lossy path, so we gate it explicitly:
    a drill in the manifest with an unrenderable asset surfaces as a blank card."""
    import verovio

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tk = verovio.toolkit()
        tk.setInputFrom("musicxml")
        loaded = tk.loadData(xml.decode("utf-8", "replace"))
        pages = tk.getPageCount() if loaded else 0
    if not loaded or pages < 1:
        raise ValueError(
            f"tier-C MusicXML failed Verovio render gate: {primitive_id} "
            f"(loaded={loaded}, pages={pages})"
        )


def _build_tier_a(primitive_id: str, xml_path: Path, out_dir: Path) -> tuple[Path, int]:
    """Committed MusicXML -> MXL ZIP. Bars from the source score."""
    try:
        score = partitura.load_score(str(xml_path))
    except Exception as e:  # noqa: BLE001 -- re-raise naming the offending file
        raise ValueError(
            f"primitive .xml failed partitura load: {xml_path} ({e})"
        ) from e
    bars = _measures_in_score(score)
    return _write_mxl(xml_path.read_bytes(), primitive_id, out_dir), bars


def _build_tier_c(primitive_id: str, midi_path: Path, out_dir: Path) -> tuple[Path, int]:
    """Score-derived MIDI -> partitura MusicXML -> MXL ZIP (render-gated)."""
    try:
        score = partitura.load_score_midi(str(midi_path))
    except Exception as e:  # noqa: BLE001 -- re-raise naming the offending file
        raise ValueError(
            f"primitive .mid failed partitura load: {midi_path} ({e})"
        ) from e
    bars = _measures_in_score(score)
    buf = io.BytesIO()
    partitura.save_musicxml(score, buf)
    xml = buf.getvalue()
    _render_gate_musicxml(xml, primitive_id)
    return _write_mxl(xml, primitive_id, out_dir), bars


def _etude_krn_index(kern_repo: Path) -> list[Path]:
    """The 24 complete-edition Chopin etude .krn, sorted exactly as
    acquire_chopin_etudes.sh sorts them (Op.10 010-1b-Sm then Op.25 025-1b-LE).
    Position i (1-based) == chopin_etude's source_exercise_number."""
    if not kern_repo.exists():
        raise FileNotFoundError(
            f"Chopin etude kern repo missing: {kern_repo}\n"
            "Run: bash model/src/exercise_corpus/acquire_chopin_etudes.sh"
        )
    files = sorted(f for f in kern_repo.glob("*.krn") if _ETUDE_KRN_RE.search(f.name))
    if len(files) != 24:
        raise ValueError(
            f"expected 24 complete-edition etude .krn in {kern_repo}, found {len(files)} "
            "-- repo layout may have changed"
        )
    return files


def _build_tier_b(primitive_id: str, krn_path: Path, out_dir: Path) -> tuple[Path, int]:
    """Original **kern -> Verovio-native MEI (the cleanest engraving). Skip-if-exists
    for idempotency (Verovio MEI is not byte-stable across versions); bars are the
    MEI measure count either way."""
    import verovio

    mei_path = out_dir / f"{primitive_id}.mei"
    if mei_path.exists():
        mei = mei_path.read_text()
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tk = verovio.toolkit()
            tk.setInputFrom("humdrum")
            if not tk.loadFile(str(krn_path)):
                raise ValueError(
                    f"verovio loadFile failed for etude {primitive_id}: {krn_path}"
                )
            pages = tk.getPageCount()
            if pages < 1:
                raise ValueError(
                    f"verovio getPageCount={pages} for etude {primitive_id}: {krn_path}"
                )
            mei = tk.getMEI()
        if not mei or len(mei) < 500:
            raise ValueError(
                f"empty/short MEI for etude {primitive_id} ({len(mei)}B): {krn_path}"
            )
        mei_path.write_text(mei)
    bars = len(_MEI_MEASURE_RE.findall(mei))
    if bars < 1:
        raise ValueError(f"MEI for etude {primitive_id} has no <measure> elements")
    return mei_path, bars


def build(
    *,
    out_dir: Path = _DEFAULT_OUT_DIR,
    xml_dir: Path = _DEFAULT_XML_DIR,
    midi_dir: Path = _DEFAULT_MIDI_DIR,
    embed_manifest: Path = _DEFAULT_EMBED_MANIFEST,
    kern_repo: Path = _DEFAULT_KERN_REPO,
    manifest_path: Path | None = None,
) -> list[Path]:
    """Materialize one renderable asset per drill (tiered) + optionally the API
    manifest. Returns produced asset paths sorted by primitive id.

    The roster is ``embed_manifest`` (primitive_id + source_exercise_number);
    dimensions + key come from ``technique_tags.toml``; bars from the engraved asset.

    If ``manifest_path`` is provided, writes the JSON manifest the API routes over
    (``{id: {dimensions, key, totalBars}}``). Default None => no write.

    Raises:
        FileNotFoundError: if the embed manifest, a referenced MIDI, or the etude
            kern repo (when any etude is built) is missing.
        ValueError: if a primitive lacks a technique_tags entry, an etude's
            source_exercise_number is out of range, or any source fails to
            load/render (message names the primitive).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    embed_manifest = Path(embed_manifest)
    if not embed_manifest.exists():
        raise FileNotFoundError(f"embed_ready_manifest not found: {embed_manifest}")
    primitives = json.loads(embed_manifest.read_text())["primitives"]

    with open(_TECHNIQUE_TAGS, "rb") as f:
        tags = tomllib.load(f)

    etude_index: list[Path] | None = None  # built lazily on first etude
    produced: list[Path] = []
    manifest: dict[str, dict] = {}

    for prim in sorted(primitives, key=lambda r: r["primitive_id"]):
        primitive_id = prim["primitive_id"]
        if primitive_id not in tags:
            raise ValueError(
                f"primitive {primitive_id!r} is in {embed_manifest.name} but has no "
                "technique_tags.toml entry (per-primitive dimensions are authoritative)"
            )

        xml_path = Path(xml_dir) / f"{primitive_id}.xml"
        if xml_path.exists():
            asset, bars = _build_tier_a(primitive_id, xml_path, out_dir)
        elif primitive_id.startswith(_ETUDE_PREFIX):
            if etude_index is None:
                etude_index = _etude_krn_index(Path(kern_repo))
            i = prim["source_exercise_number"]
            if not (1 <= i <= len(etude_index)):
                raise ValueError(
                    f"etude {primitive_id} has source_exercise_number={i} out of range "
                    f"[1, {len(etude_index)}]"
                )
            asset, bars = _build_tier_b(primitive_id, etude_index[i - 1], out_dir)
        else:
            midi_path = Path(midi_dir) / f"{primitive_id}.mid"
            if not midi_path.exists():
                raise FileNotFoundError(
                    f"primitive {primitive_id!r} has no committed .xml and no MIDI at "
                    f"{midi_path} -- run the corpus acquire recipes first"
                )
            asset, bars = _build_tier_c(primitive_id, midi_path, out_dir)

        produced.append(asset)
        entry = tags[primitive_id]
        manifest[primitive_id] = {
            "dimensions": entry["dimensions"],
            "key": entry["key"],
            "totalBars": bars,
        }

    if manifest_path is not None:
        manifest_path = Path(manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    return produced
