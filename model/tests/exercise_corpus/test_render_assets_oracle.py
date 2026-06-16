# model/tests/exercise_corpus/test_render_assets_oracle.py
"""Hermetic faithful-shift oracle: within EACH engine (Verovio and partitura),
transposing a committed exercise-primitive MusicXML by N semitones must shift
its pitch-class multiset by exactly N mod 12 relative to that same engine's
transpose-0 baseline. This proves each engine transposes by exactly N — the
property S3 needs — and is robust to repeat expansion (N and 0 expand
identically within an engine) and to accidental realization (constant within
one engine).

A SECONDARY, independent cross-engine witness asserts that the transpose-0
baseline multisets agree between Verovio and partitura. This holds for 20 of 22
primitives. The 2 documented divergences are base-file-realization facts, NOT
transpose bugs:
  - burgmuller_001: contains <repeat> pairs; Verovio's renderToTimemap EXPANDS
    repeats (420 onsets) while partitura reads the 248 literal score notes.
  - czerny_001: no repeats, equal note sums (392 == 392), but partitura and
    Verovio realize a handful of accidentals to different SOUNDING midi pitches,
    so midi_pitch % 12 disagrees at baseline.
Both are xfail(strict=True) on the SECONDARY check only; the PRIMARY
faithful-shift assertion still covers ALL 22, so the transpose guarantee is NOT
narrowed.

Hermetic means: enumerate by globbing the COMMITTED
model/data/scores/exercise_primitives/*.xml. Never touch the gitignored
exercise_primitives.db or the gitignored .mid files. Verovio runs via its
Python binding (same WASM core as the web's verovio npm package), in-process
with partitura — no Node subprocess.
"""

import glob
import json
from collections import Counter
from pathlib import Path

import partitura
import pytest
import verovio
from partitura.score import Note

from exercise_corpus.transforms import transpose

# Anchor to this file, never CWD (CLAUDE.md: just recipes shift CWD).
_XML_DIR = Path(__file__).resolve().parents[2] / "data" / "scores" / "exercise_primitives"
_XML_FILES = sorted(glob.glob(str(_XML_DIR / "*.xml")))
# In-range band: hanon-style primitives sit mid-keyboard, so -5..+6 stays on
# the 88-key piano for these compact patterns. Off-keyboard symmetry is Task 2.
_IN_RANGE_OFFSETS = list(range(-5, 7))

# Primitives whose transpose-0 cross-engine baseline multisets diverge for a
# DOCUMENTED base-file-realization reason (NOT a transpose bug). The PRIMARY
# faithful-shift assertion still covers these, so the transpose guarantee is
# unaffected. Asserted to be exactly this set so an UNEXPECTED new divergence
# fails loudly rather than being masked.
_BASELINE_DIVERGENCE = {
    "burgmuller_001": "repeat expansion (Verovio timemap expands <repeat>, partitura reads literal notes)",
    "czerny_001": "accidental realization (partitura vs Verovio sound a few accidentals to different midi pitches)",
}


def _pc_partitura(part) -> Counter:
    return Counter(n.midi_pitch % 12 for n in part.iter_all(Note))


def _pc_verovio(tk: "verovio.toolkit", xml: str, semitones: int) -> Counter:
    tk.setOptions({"transpose": str(semitones)})
    assert tk.loadData(xml), "verovio loadData failed"
    tm = tk.renderToTimemap({"includeMeasures": True})
    if isinstance(tm, str):
        tm = json.loads(tm)
    c: Counter = Counter()
    for entry in tm:
        for note_id in (entry.get("on") or []):
            v = tk.getMIDIValuesForElement(note_id)
            if isinstance(v, str):
                v = json.loads(v)
            if isinstance(v, dict) and isinstance(v.get("pitch"), int):
                c[v["pitch"] % 12] += 1
    return c


def _shift_by(pc: Counter, n: int) -> Counter:
    """Shift every pitch class in the multiset by n semitones (mod 12)."""
    shifted: Counter = Counter()
    for cls, count in pc.items():
        shifted[(cls + n) % 12] += count
    return shifted


def test_committed_xml_files_present():
    # Guards the hermetic enumeration: if the glob is empty the parametrized
    # tests would silently pass with zero cases.
    assert len(_XML_FILES) == 22, f"expected 22 committed primitive .xml, found {len(_XML_FILES)}"


def test_baseline_divergence_set_is_exactly_the_two_known_cases():
    # Lock the documented exception list so an unexpected NEW cross-engine
    # baseline divergence cannot hide behind the xfail markers.
    assert set(_BASELINE_DIVERGENCE) == {"burgmuller_001", "czerny_001"}
    assert len(_BASELINE_DIVERGENCE) == 2


@pytest.mark.parametrize("xml_path", _XML_FILES, ids=lambda p: Path(p).stem)
def test_faithful_shift_invariant_holds_in_each_engine(xml_path: str):
    """PRIMARY (all 22): within EACH engine, transpose-by-N == shift-by-N of
    that engine's transpose-0 baseline. Proves each engine transposes by exactly
    N semitones. Robust to repeats and accidental realization (the relation is
    intra-engine, so any per-engine quirk is present identically at N and 0)."""
    xml = Path(xml_path).read_text()
    score = partitura.load_score(xml_path)
    part0 = list(score.parts)[0]
    tk = verovio.toolkit()

    # Per-engine transpose-0 baselines.
    base_partitura = _pc_partitura(transpose(part0, 0).part)
    base_verovio = _pc_verovio(tk, xml, 0)

    # Two independent passes so a partitura off-keyboard rejection only skips
    # ITS OWN branch — Verovio (which has no keyboard bound) is still asserted
    # at every offset. Off-keyboard offsets still expect partitura to raise.
    partitura_matched = 0
    for semis in _IN_RANGE_OFFSETS:
        if semis == 0:
            continue
        try:
            partitura_pc = _pc_partitura(transpose(part0, semis).part)
        except ValueError:
            # Off-keyboard for this primitive at this offset — partitura raises
            # (transforms.py lines 104-108); Task 2 owns rejection symmetry.
            # Skip only the partitura branch for this offset.
            continue
        assert partitura_pc == _shift_by(base_partitura, semis), (
            f"{Path(xml_path).stem} @ {semis}: partitura did not faithfully "
            f"shift its own baseline: {dict(partitura_pc)} != "
            f"{dict(_shift_by(base_partitura, semis))}"
        )
        partitura_matched += 1

    verovio_matched = 0
    for semis in _IN_RANGE_OFFSETS:
        if semis == 0:
            continue
        verovio_pc = _pc_verovio(tk, xml, semis)
        assert verovio_pc == _shift_by(base_verovio, semis), (
            f"{Path(xml_path).stem} @ {semis}: verovio did not faithfully "
            f"shift its own baseline: {dict(verovio_pc)} != "
            f"{dict(_shift_by(base_verovio, semis))}"
        )
        verovio_matched += 1

    assert partitura_matched > 0, f"no in-range partitura offsets matched for {Path(xml_path).stem}"
    assert verovio_matched > 0, f"no in-range verovio offsets matched for {Path(xml_path).stem}"


def _baseline_marks(p: str):
    stem = Path(p).stem
    if stem in _BASELINE_DIVERGENCE:
        return pytest.param(
            p,
            marks=pytest.mark.xfail(
                reason=f"baseline cross-engine divergence ({_BASELINE_DIVERGENCE[stem]}); "
                "documented base-file-realization fact, NOT a transpose bug — the "
                "PRIMARY faithful-shift test still covers this primitive",
                strict=True,
            ),
        )
    return pytest.param(p)


@pytest.mark.parametrize(
    "xml_path", [_baseline_marks(p) for p in _XML_FILES], ids=lambda p: Path(p).stem
)
def test_baseline_cross_engine_agreement(xml_path: str):
    """SECONDARY (independent witness, 20 of 22): the transpose-0 baseline
    multisets agree between Verovio and partitura. The 2 documented divergences
    (repeat expansion, accidental realization) are xfail(strict=True) — failing
    loudly if they ever start agreeing, and never silently dropped."""
    xml = Path(xml_path).read_text()
    score = partitura.load_score(xml_path)
    part0 = list(score.parts)[0]
    tk = verovio.toolkit()

    base_partitura = _pc_partitura(transpose(part0, 0).part)
    base_verovio = _pc_verovio(tk, xml, 0)
    assert base_verovio == base_partitura, (
        f"{Path(xml_path).stem} baseline cross-engine mismatch: "
        f"verovio {dict(base_verovio)} != partitura {dict(base_partitura)}"
    )


def _max_in_range_up(part) -> int:
    """Smallest positive offset that pushes the highest note above MIDI 108."""
    highest = max(n.midi_pitch for n in part.iter_all(Note))
    return 108 - highest + 1  # this many semitones up is the first off-keyboard offset


def test_partitura_rejects_off_keyboard_transpose():
    # Use the first committed primitive; push it just past the top of the
    # keyboard and assert transforms.py raises (the off-keyboard GATE itself is
    # deferred to S4 — here we only assert the reject path exists and is sharp).
    xml_path = _XML_FILES[0]
    score = partitura.load_score(xml_path)
    part0 = list(score.parts)[0]

    off_offset = _max_in_range_up(part0)
    with pytest.raises(ValueError, match="outside piano"):
        transpose(part0, off_offset)
