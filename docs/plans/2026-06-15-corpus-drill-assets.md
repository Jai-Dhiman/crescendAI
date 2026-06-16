# Corpus Drill Renderable Assets + Verovio Transpose-on-Demand Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until /challenge returns VERDICT: PROCEED.
> Use model "Sonnet 4.6" for any subagents.

**Goal:** A corpus_drill exercise primitive (e.g. `hanon_001`) can be fetched from R2 and rendered in the web score panel at any in-range key, so the existing `corpus_drill` card can show a real, transposable score instead of stub text.
**Spec:** docs/specs/2026-06-15-corpus-drill-assets-design.md
**Style:** Follow CLAUDE.md / `apps/api/TS_STYLE.md`. Python uses `uv` + partitura (never music21). No emojis. Explicit exceptions over silent fallbacks.

## Task Groups

```
Group 0 (harness, blocking — must pass before Groups A/B/C dispatch):
  Task 1 — Verovio↔partitura PC-multiset oracle (in-range)
  Task 2 — Oracle off-keyboard rejection symmetry

Group A (parallel, depends on Group 0): [SHIPS INDEPENDENTLY — Python asset pipeline]
  Task 3 — build() produces valid MXL assets (asset integrity)
  Task 4 — build() is idempotent + fails loud on bad XML

Group B (sequential, depends on Group A):
  Task 5 — just build-exercise-assets + seed-exercise-assets recipes

Group C (parallel with Group A/B — disjoint files): [SHIPS INDEPENDENTLY — web renderer transpose]
  Task 6 — loadPiece transpose yields different SVG; transpose 0 == no-transpose
  Task 7 — ScoreRenderer.load(pieceId, transpose) composite cache key
```

- **Group 0** is the verification harness (spec Verification Architecture). It runs entirely against already-shipped `transforms.py::transpose` and the Verovio Python binding — no feature code needed. It must be green before A/B/C.
- **Group A `[SHIPS INDEPENDENTLY]`:** with just Group 0 + A, the 22 `.mxl` assets exist and are proven valid — a future S4 could seed them by hand. User value: renderable assets committed.
- **Group C `[SHIPS INDEPENDENTLY]`:** the web transpose path is self-contained; with Group C alone, any already-seeded `.mxl` renders at any key. Touches only `apps/web/src/lib/*` — disjoint from Groups A/B (`model/**`, `justfile`), so C may run concurrently with A.
- **Group B** depends on A (the recipe enumerates A's committed `.mxl` output).

---

## Task 1: Verovio↔partitura faithful-shift transpose oracle (in-range offsets)

**Group:** 0 (parallel with Task 2)

**Behavior being verified:** For every committed exercise-primitive `.xml`, in BOTH engines independently (Verovio and partitura), transposing by an in-range semitone offset N shifts the pitch-class multiset by exactly N mod 12 relative to the same engine's transpose-0 baseline. This is the property S3 actually needs — proof that each engine transposes by exactly N semitones — and it holds empirically for all 22 primitives in both engines, robustly to repeat expansion (both N and 0 expand identically within an engine) and to accidental realization (constant within one engine). A SECONDARY independent witness asserts cross-engine baseline (transpose-0) multiset agreement, which holds for 20 of 22 (the 2 known divergences `burgmuller_001` = repeat expansion, `czerny_001` = accidental realization are documented base-file-realization facts, NOT transpose bugs, and are marked xfail/skip — the PRIMARY faithful-shift assertion still covers all 22, so the transpose guarantee is NOT narrowed).
**Interface under test:** `exercise_corpus.transforms.transpose(part, semitones)` and the `verovio` toolkit `setOptions({transpose})` + `loadData()` + timemap, both reading the same committed `.xml`.

**Files:**
- Create: `model/tests/exercise_corpus/test_render_assets_oracle.py`
- Modify: `model/pyproject.toml` (add `verovio` to dev deps so `uv run pytest` has it without `--with`)
- Test: same file (this task IS a test that gates feature code)

- [ ] **Step 1: Write the failing test**

```python
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

    matched = 0
    for semis in _IN_RANGE_OFFSETS:
        if semis == 0:
            continue
        try:
            partitura_pc = _pc_partitura(transpose(part0, semis).part)
        except ValueError:
            # Off-keyboard for this primitive at this offset — partitura raises
            # (transforms.py lines 104-108); Task 2 owns rejection symmetry.
            # Keep both engines symmetric: skip this offset entirely.
            continue
        verovio_pc = _pc_verovio(tk, xml, semis)

        assert partitura_pc == _shift_by(base_partitura, semis), (
            f"{Path(xml_path).stem} @ {semis}: partitura did not faithfully "
            f"shift its own baseline: {dict(partitura_pc)} != "
            f"{dict(_shift_by(base_partitura, semis))}"
        )
        assert verovio_pc == _shift_by(base_verovio, semis), (
            f"{Path(xml_path).stem} @ {semis}: verovio did not faithfully "
            f"shift its own baseline: {dict(verovio_pc)} != "
            f"{dict(_shift_by(base_verovio, semis))}"
        )
        matched += 1

    assert matched > 0, f"no in-range offsets matched for {Path(xml_path).stem}"


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
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/exercise_corpus/test_render_assets_oracle.py -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'verovio'` (the `import verovio` line fails because `verovio` is not yet in the model dev deps). This proves the test runs the real binding rather than a stub.

- [ ] **Step 3: Implement the minimum to make the test pass**

Add `verovio` to the model dev dependency group so `uv run pytest` resolves it. In `model/pyproject.toml`, locate the dev dependency list that already contains `"pytest>=8.4.2"` and `"pytest-cov>=7.0.0"` (the second occurrence, around line 129) and add a `verovio` line:

```toml
    "pytest>=8.4.2",
    "pytest-cov>=7.0.0",
    "verovio>=4.0.0",
```

Then sync so the env has it:

```bash
cd model && uv sync --group dev
```

(No production code is written for this task — the oracle exercises already-shipped `transforms.py::transpose`. The "implementation" is making the harness's dependency available, exactly as a harness task should.)

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/exercise_corpus/test_render_assets_oracle.py -q
```
Expected: PASS — `test_committed_xml_files_present`, `test_baseline_divergence_set_is_exactly_the_two_known_cases`, all 22 `test_faithful_shift_invariant_holds_in_each_engine` cases green, and `test_baseline_cross_engine_agreement` reporting 20 passed + 2 xfailed (`burgmuller_001`, `czerny_001`). With `strict=True`, if either of those 2 begins agreeing it becomes an XPASS = failure, surfacing the change. (Verovio emits `[Warning] Adding auxiliary KeySig for transposition` lines to stderr; these are not failures.)

- [ ] **Step 5: Commit**

```bash
git add model/tests/exercise_corpus/test_render_assets_oracle.py model/pyproject.toml model/uv.lock && git commit -m "test(#46): Verovio-vs-partitura pitch-class oracle for exercise primitives"
```

---

## Task 2: Oracle off-keyboard rejection symmetry

**Group:** 0 (parallel with Task 1 — different test function, but SAME file; see note)

> **Parallelism note:** Tasks 1 and 2 both write to `test_render_assets_oracle.py`. They CANNOT run as concurrent subagents. Run Task 1 first, then Task 2 appends its function. Treat Group 0 as sequential internally (1 → 2). The "parallel" label is only relative to later groups.

**Behavior being verified:** For a semitone offset that pushes a primitive off the 88-key piano, BOTH `transforms.py::transpose` raises and the result is treated as a reject — the two sides agree on rejection, not just on equivalence.
**Interface under test:** `exercise_corpus.transforms.transpose(part, semitones)` raising `ValueError` for out-of-range pitches (transforms.py lines 104–108).

**Files:**
- Modify: `model/tests/exercise_corpus/test_render_assets_oracle.py` (append one test function)
- Test: same file

- [ ] **Step 1: Write the failing test**

Append to `model/tests/exercise_corpus/test_render_assets_oracle.py`:

```python
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
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/exercise_corpus/test_render_assets_oracle.py::test_partitura_rejects_off_keyboard_transpose -q
```
Expected: FAIL — `NameError: name '_max_in_range_up' is not defined` IF run before Step 3 placement, OR (if the helper is added but the assertion is wrong) the raise does not fire. The point: the function under test must be exercised, not assumed. If it unexpectedly PASSES before you intend, you mis-computed `off_offset` — recheck the highest pitch.

- [ ] **Step 3: Implement the minimum to make the test pass**

No production code. The behavior already exists in `transforms.py` (lines 104–108 raise `ValueError(f"... outside piano range ...")`). This task only adds the test that locks that reject path as part of the oracle contract. The "implementation" is confirming the assertion's `match="outside piano"` substring matches the real message in `transforms.py::transpose`:

```python
raise ValueError(
    f"transpose by {semitones} puts pitch {new_midi} outside piano "
    f"range [{PIANO_MIDI_LOW}, {PIANO_MIDI_HIGH}]"
)
```
(`"outside piano"` is a substring of that message — assertion is correct as written.)

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/exercise_corpus/test_render_assets_oracle.py::test_partitura_rejects_off_keyboard_transpose -q
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add model/tests/exercise_corpus/test_render_assets_oracle.py && git commit -m "test(#46): off-keyboard transpose rejection symmetry in oracle"
```

---

## Task 3: build() produces valid MXL assets (asset integrity)

**Group:** A (parallel with Group C; internally before Task 4)

> **Parallelism note:** Tasks 3 and 4 both create/modify `build_render_assets.py` and its test file. Run them sequentially (3 → 4). They are parallel only relative to Group C (which touches `apps/web/**`).

**Behavior being verified:** `build()` turns every committed primitive `.xml` into a valid MXL ZIP at `model/data/exercise_primitives/mxl/{id}.mxl` whose inner MusicXML has the same note count as the source `.xml`.
**Interface under test:** `exercise_corpus.build_render_assets.build()`.

**Files:**
- Create: `model/src/exercise_corpus/build_render_assets.py`
- Create: `model/tests/exercise_corpus/test_build_render_assets.py`
- Test: `model/tests/exercise_corpus/test_build_render_assets.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/exercise_corpus/test_build_render_assets.py -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'exercise_corpus.build_render_assets'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# model/src/exercise_corpus/build_render_assets.py
"""Build committed renderable assets (.mxl) from committed exercise-primitive
MusicXML (.xml).

For each model/data/scores/exercise_primitives/*.xml: validate it loads in
partitura, strip its DOCTYPE, wrap it in a standard MXL ZIP (reusing the proven
score_library.upload.wrap_as_mxl_zip container format), and write
model/data/exercise_primitives/mxl/{primitive_id}.mxl.

Deterministic and idempotent: an asset whose inner XML already equals the
freshly-stripped source XML is left untouched. A .xml that fails partitura load
RAISES naming the file (no skip-and-continue) — explicit exceptions over silent
fallbacks (CLAUDE.md).
"""

from __future__ import annotations

import glob
import io
import zipfile
from pathlib import Path

import partitura

from score_library.upload import _strip_doctype, wrap_as_mxl_zip

# Anchor to this module, never CWD (CLAUDE.md: just recipes shift CWD).
_MODEL_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_XML_DIR = _MODEL_ROOT / "data" / "scores" / "exercise_primitives"
_DEFAULT_OUT_DIR = _MODEL_ROOT / "data" / "exercise_primitives" / "mxl"


def _existing_inner_xml(mxl_path: Path) -> bytes | None:
    """Return the inner MusicXML bytes of an existing .mxl, or None if absent
    or unreadable — used for the idempotent skip-if-unchanged check."""
    if not mxl_path.exists():
        return None
    try:
        with zipfile.ZipFile(io.BytesIO(mxl_path.read_bytes())) as zf:
            for name in zf.namelist():
                if not name.startswith("META-INF") and name.endswith(".xml"):
                    return zf.read(name)
    except zipfile.BadZipFile:
        return None
    return None


def build(
    xml_dir: Path = _DEFAULT_XML_DIR,
    out_dir: Path = _DEFAULT_OUT_DIR,
) -> list[Path]:
    """Generate one .mxl per committed primitive .xml. Returns produced paths
    (sorted by primitive id), including unchanged ones that were skipped.

    Raises:
        FileNotFoundError: if xml_dir contains no .xml files.
        ValueError: if any .xml fails to load in partitura (message names the file).
    """
    xml_dir = Path(xml_dir)
    out_dir = Path(out_dir)
    xml_files = sorted(glob.glob(str(xml_dir / "*.xml")))
    if not xml_files:
        raise FileNotFoundError(f"No primitive .xml files found in {xml_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []

    for xml_path_str in xml_files:
        xml_path = Path(xml_path_str)
        primitive_id = xml_path.stem

        # Fail loud if the source XML is not valid for the corpus.
        try:
            partitura.load_score(str(xml_path))
        except Exception as e:  # noqa: BLE001 — re-raise with the offending file named
            raise ValueError(f"primitive .xml failed partitura load: {xml_path} ({e})") from e

        stripped = _strip_doctype(xml_path.read_bytes())
        mxl_path = out_dir / f"{primitive_id}.mxl"

        # Idempotent: skip if the inner XML already matches the stripped source.
        if _existing_inner_xml(mxl_path) == stripped:
            produced.append(mxl_path)
            continue

        mxl_bytes = wrap_as_mxl_zip(stripped, primitive_id)
        mxl_path.write_bytes(mxl_bytes)
        produced.append(mxl_path)

    return produced
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/exercise_corpus/test_build_render_assets.py -q
```
Expected: PASS. `build()` does DOCTYPE-strip + ZIP-wrap (`wrap_as_mxl_zip`) ONLY — the inner-note-count assertion is partitura-vs-partitura over a byte-preserving container, so it holds. Do NOT add a speculative partitura re-export normalization pass: the /challenge empirical run already produced Verovio timemaps for all 22 `.xml`, retiring the "do they render?" residual risk, and a re-export round-trip would risk dropping/re-spelling notes (especially `burgmuller_001`'s repeat structure) and break the inner-note-count lock. If a `.xml` genuinely fails to load, `build()` already RAISES naming the file (Step 3 guard) — that is a finding to surface, never a re-export fallback to paper over.

- [ ] **Step 5: Commit**

```bash
git add model/src/exercise_corpus/build_render_assets.py model/tests/exercise_corpus/test_build_render_assets.py && git commit -m "feat(#46): build() generates renderable .mxl assets from primitive .xml"
```

---

## Task 4: build() idempotency + fail-loud on bad XML

**Group:** A (sequential after Task 3 — same files)

**Behavior being verified:** Re-running `build()` over unchanged sources rewrites nothing (idempotent), and a `.xml` that partitura cannot load makes `build()` raise naming the file rather than silently skipping it.
**Interface under test:** `exercise_corpus.build_render_assets.build()`.

**Files:**
- Modify: `model/tests/exercise_corpus/test_build_render_assets.py` (append two tests)
- Test: same file

- [ ] **Step 1: Write the failing test**

Append to `model/tests/exercise_corpus/test_build_render_assets.py`:

```python
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
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/exercise_corpus/test_build_render_assets.py::test_build_is_idempotent tests/exercise_corpus/test_build_render_assets.py::test_build_raises_naming_bad_xml -q
```
Expected: both run; `test_build_is_idempotent` PASSES against the Task 3 implementation (idempotency is already implemented) but `test_build_raises_naming_bad_xml` must PASS too — if either fails, the implementation is incomplete. Specifically, if `test_build_raises_naming_bad_xml` fails with "DID NOT RAISE", partitura accepted the junk XML; tighten the malformed input to `bad.write_text("not xml at all {{{")` so `load_score` raises.

> Note: idempotency was implemented in Task 3, so `test_build_is_idempotent` is a lock, not a driver. The driving test for this task is `test_build_raises_naming_bad_xml`. If you find BOTH already pass against Task 3 code, that is acceptable here because these are regression locks on already-built behavior — but you must still watch each fail by temporarily reverting the relevant `build()` guard (the `raise ValueError(...)` for bad XML) to confirm the test bites, then restore it.

- [ ] **Step 3: Implement the minimum to make the test pass**

The fail-loud and idempotency behavior already exist in Task 3's `build()`. If `test_build_raises_naming_bad_xml` did not bite, ensure the partitura-load guard in `build()` wraps `partitura.load_score` in a `try/except Exception` that re-raises `ValueError(f"primitive .xml failed partitura load: {xml_path} ...")` — the `{xml_path}` interpolation is what makes `match="broken_001"` succeed. No new code beyond confirming that guard is present and the message includes the stem.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/exercise_corpus/test_build_render_assets.py -q
```
Expected: PASS (all build tests).

- [ ] **Step 5: Commit (also commit the 22 generated assets here)**

Generate the committed assets now that `build()` is proven, then commit code + assets together:

```bash
cd model && uv run python -c "from exercise_corpus.build_render_assets import build; print(len(build()))" && cd ..
git add model/tests/exercise_corpus/test_build_render_assets.py model/data/exercise_primitives/mxl/*.mxl && git commit -m "feat(#46): commit 22 renderable .mxl assets + idempotency/fail-loud tests"
```

---

## Task 5: just build-exercise-assets + seed-exercise-assets recipes

**Group:** B (depends on Group A — recipe enumerates A's committed .mxl)

**Behavior being verified:** `just seed-exercise-assets` puts every committed primitive `.mxl` into LOCAL wrangler R2 at `scores/v1/{id}.mxl`, and `just build-exercise-assets` regenerates the committed assets.
**Interface under test:** the two `just` recipes (invoked as shell commands).

**Files:**
- Modify: `justfile` (append two recipes)
- Test: manual recipe invocation (no automated harness — see verification note)

> **Verification note:** `just seed-exercise-assets` writes to local wrangler R2, which has no hermetic in-process assertion available in this repo (the existing `seed-fingerprint`/`seed-scores`/`seed-score-json` recipes likewise have no automated test). Verification is a manual recipe run asserting non-zero file count and exit 0. This is the spec's "manual verification step" for the seed surface. `just build-exercise-assets` IS covered automatically because it calls the `build()` already locked by Tasks 3–4.

- [ ] **Step 1: Write the failing check**

There is no unit test for a justfile recipe. The failing check is: the recipes do not exist yet.

```bash
just --list 2>/dev/null | grep -E "build-exercise-assets|seed-exercise-assets"
```
Expected: FAIL — no matching lines (recipes absent).

- [ ] **Step 2: Confirm the check fails**

```bash
just --list 2>/dev/null | grep -E "build-exercise-assets|seed-exercise-assets" || echo "RECIPES ABSENT (expected before impl)"
```
Expected: prints `RECIPES ABSENT (expected before impl)`.

- [ ] **Step 3: Implement — append the two recipes to `justfile`**

Append after the existing `seed-score-json` recipe (mirror its style and the `seed-fingerprint` comment convention):

```makefile
# Regenerate the committed renderable .mxl assets from the committed primitive
# .xml (model/data/scores/exercise_primitives/*.xml -> model/data/exercise_primitives/mxl/*.mxl).
# Deterministic + idempotent; raises naming any .xml that fails partitura load.
build-exercise-assets:
    cd model && uv run python -c "from exercise_corpus.build_render_assets import build; print(f'built {len(build())} assets')"

# Seed the committed exercise-primitive .mxl assets into LOCAL wrangler R2 at
# scores/v1/{primitive_id}.mxl so the UNCHANGED GET /api/scores/:pieceId/data
# endpoint serves them for corpus_drill rendering. Flat keyspace: primitive ids
# (hanon_001) cannot collide with real ASAP piece slugs. Run `just build-exercise-assets`
# first. Mirrors `seed-scores` (real-piece .mxl) and `seed-fingerprint`.
seed-exercise-assets:
    #!/usr/bin/env bash
    set -euo pipefail
    shopt -s nullglob
    count=0
    for f in model/data/exercise_primitives/mxl/*.mxl; do
        base="$(basename "$f")"
        cd apps/api && wrangler r2 object put "crescendai-bucket/scores/v1/$base" \
            --file="../../$f" --content-type "application/vnd.recordare.musicxml+zip" --local >/dev/null && cd ../..
        count=$((count+1))
        echo "Seeded scores/v1/$base"
    done
    if [ "$count" -eq 0 ]; then
        echo "ERROR: no .mxl found in model/data/exercise_primitives/mxl/ — run 'just build-exercise-assets' first" >&2
        exit 1
    fi
    echo "Seeded $count exercise-primitive asset(s) into local R2."
```

- [ ] **Step 4: Run — verify it PASSES**

```bash
just build-exercise-assets
just seed-exercise-assets
```
Expected: `build-exercise-assets` prints `built 22 assets`; `seed-exercise-assets` prints 22 `Seeded scores/v1/<id>.mxl` lines then `Seeded 22 exercise-primitive asset(s) into local R2.` and exits 0. (If `wrangler` is not authenticated/available in the build environment, `seed-exercise-assets` may fail at the R2 put — that is an environment precondition, not a recipe defect; `build-exercise-assets` must still pass unconditionally.)

- [ ] **Step 5: Commit**

```bash
git add justfile && git commit -m "feat(#46): just build-exercise-assets + seed-exercise-assets recipes"
```

---

## Task 6: loadPiece transpose yields different SVG; transpose 0 == no-transpose

**Group:** C (parallel with Groups A/B — touches only apps/web/**; internally before Task 7)

> **Parallelism note:** Tasks 6 and 7 both modify `apps/web/src/lib/score-worker.ts` (Task 6) and the new integration test (Task 6) / `score-renderer.ts` (Task 7). Task 7 depends on Task 6's `loadPiece` signature change. Run 6 → 7 sequentially. Group C as a whole is disjoint from A/B.

**Behavior being verified:** `loadPiece(bytes, bindings, pieceId, 2)` produces a different engraving than `loadPiece(bytes, bindings, pieceId, 0)`, and `transpose: 0` produces byte-identical output to omitting the argument (real-piece regression lock).
**Interface under test:** `loadPiece(bytes, bindings, pieceId?, transpose?: number)` (web worker public function).

**Files:**
- Modify: `apps/web/src/lib/score-worker.ts` (add `transpose?` param to `loadPiece`; set `tk.setOptions({ transpose: String(n) })` before load when nonzero)
- Create: `apps/web/src/lib/score-worker.transpose.integration.test.ts`
- Test: `apps/web/src/lib/score-worker.transpose.integration.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/web/src/lib/score-worker.transpose.integration.test.ts
// Real-Verovio integration: loadPiece's optional transpose param must change
// the engraving, and transpose:0 must be byte-identical to omitting it.
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const FIXTURE_PATH = resolve(
  dirname(fileURLToPath(import.meta.url)),
  "../../public/scores/czerny-op299-no1.mxl",
);

async function makeBindings() {
  // biome-ignore lint/suspicious/noExplicitAny: dynamic ESM
  const esm = (await import("verovio/esm")) as any;
  // biome-ignore lint/suspicious/noExplicitAny: dynamic ESM
  const wasm = (await import("verovio/wasm")) as any;
  const VerovioToolkit = esm.VerovioToolkit ?? esm.default?.VerovioToolkit;
  const VerovioModule = wasm.default ?? wasm;
  const mod = await VerovioModule();
  return { module: mod, ToolkitClass: VerovioToolkit as any };
}

function freshBuf(): ArrayBuffer {
  const bytes = readFileSync(FIXTURE_PATH);
  const buf = new ArrayBuffer(bytes.byteLength);
  new Uint8Array(buf).set(bytes);
  return buf;
}

describe("loadPiece transpose param", () => {
  it("transpose:2 yields a different page-1 SVG than transpose:0", async () => {
    const bindings = await makeBindings();
    const { loadPiece } = await import("./score-worker");

    const base = await loadPiece(freshBuf(), bindings, "transpose-fixture-0", 0);
    const up = await loadPiece(freshBuf(), bindings, "transpose-fixture-2", 2);

    expect(base).not.toBe("failed");
    expect(up).not.toBe("failed");
    if (base === "failed" || up === "failed") return;

    expect(up.pageSvgs[0]).not.toBe(base.pageSvgs[0]);
  }, 60_000);

  it("transpose:0 is structurally identical to omitting the argument (real-piece lock)", async () => {
    const bindings = await makeBindings();
    const { loadPiece } = await import("./score-worker");

    // Verovio randomizes element ids on every loadData call (see the existing
    // score-worker.integration.test.ts "Cache eviction" test), so raw SVG bytes
    // differ even for identical input. Strip ids before comparing so this lock
    // verifies the ENGRAVING is identical between transpose:0 and no-transpose —
    // i.e. transpose:0 is a true no-op for real pieces.
    const omitted = await loadPiece(freshBuf(), bindings, "lock-omit");
    const zero = await loadPiece(freshBuf(), bindings, "lock-zero", 0);

    expect(omitted).not.toBe("failed");
    expect(zero).not.toBe("failed");
    if (omitted === "failed" || zero === "failed") return;

    const stripIds = (svg: string) => svg.replace(/\bid="[^"]*"/g, 'id=""');
    expect(stripIds(zero.pageSvgs[0])).toBe(stripIds(omitted.pageSvgs[0]));
  }, 60_000);
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bunx vitest run src/lib/score-worker.transpose.integration.test.ts
```
Expected: FAIL — `transpose:2 yields a different page-1 SVG` fails because `loadPiece` ignores a 4th argument today, so `up.pageSvgs[0] === base.pageSvgs[0]` (no transpose applied → SVGs equal modulo nothing). The assertion `not.toBe` fails.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `apps/web/src/lib/score-worker.ts`, change the `loadPiece` signature and apply the transpose option on each fresh toolkit before load. The function creates `tk` (and re-creates it in fallback branches), so set the option immediately after every `tk.setOptions(VEROVIO_OPTS)` that precedes a load attempt.

Change the signature:

```typescript
export async function loadPiece(
	bytes: ArrayBuffer,
	bindings: VerovioBindings,
	pieceId?: string,
	transpose?: number,
): Promise<LoadResult> {
	const { module, ToolkitClass } = bindings;
	const ZIP_MAGIC = 0x04034b50;
	const isZip =
		bytes.byteLength >= 4 &&
		new DataView(bytes).getUint32(0, true) === ZIP_MAGIC;

	const applyOpts = (t: VerovioTk) => {
		t.setOptions(VEROVIO_OPTS);
		// Verovio applies `transpose` at loadData time. A bare semitone count is
		// auto-accidental-minimized. transpose:0 / undefined is a no-op, keeping
		// real pieces byte-identical to the pre-transpose code path.
		if (transpose !== undefined && transpose !== 0) {
			t.setOptions({ transpose: String(transpose) });
		}
	};

	let tk = new ToolkitClass(module);
	applyOpts(tk);
	let loaded = false;
```

Then replace each subsequent `tk = new ToolkitClass(module); tk.setOptions(VEROVIO_OPTS);` re-init (the zip-catch branch and the fallback branch) with `applyOpts`:

```typescript
		} catch {
			tk = new ToolkitClass(module);
			applyOpts(tk);
		}
```

and

```typescript
		if (fallbackXml !== null) {
			tk = new ToolkitClass(module);
			applyOpts(tk);
```

Leave the rest of `loadPiece` unchanged.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bunx vitest run src/lib/score-worker.transpose.integration.test.ts
```
Expected: PASS — transposed SVG differs; transpose:0 matches omitted.

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/lib/score-worker.ts apps/web/src/lib/score-worker.transpose.integration.test.ts && git commit -m "feat(#46): loadPiece accepts optional semitone transpose"
```

---

## Task 7: ScoreRenderer.load(pieceId, transpose) composite cache key

**Group:** C (sequential after Task 6 — depends on loadPiece signature; modifies score-renderer.ts + worker message wiring)

**Behavior being verified:** `ScoreRenderer.load(pieceId, transpose)` routes the semitone count to the worker so the same `pieceId` at transpose 0 vs 2 are cached separately and yield different IRs, while transpose 0 / omitted share one cache slot.
**Interface under test:** `ScoreRenderer.load(pieceId, transpose?: number)` and the worker's `load` message handler.

**Files:**
- Modify: `apps/web/src/lib/score-renderer.ts` (`load(pieceId, transpose?)`, composite `irCache`/`sentPieceIds` key)
- Modify: `apps/web/src/lib/score-worker.ts` (`load` message carries `transpose?`; `toolkitCache` keyed by `${pieceId}:${transpose ?? 0}`; pass transpose into `loadPiece`)
- Modify: `apps/web/src/lib/score-worker.transpose.integration.test.ts` (append a renderer-level test)
- Test: `apps/web/src/lib/score-worker.transpose.integration.test.ts`

- [ ] **Step 1: Write the failing test**

Append to `apps/web/src/lib/score-worker.transpose.integration.test.ts`. This drives the renderer-level wiring: `ScoreRenderer.load(pieceId, transpose)` must forward the semitone count into the worker `load` message. `ScoreRenderer` owns a `Worker`, so the test stubs `globalThis.Worker` to capture posted messages and replies with a synthetic payload, and stubs `api.scores.getData` (an object property, reassignable — verified at `apps/web/src/lib/api.ts:411`) to supply bytes without a network call:

```typescript
import { ScoreRenderer } from "./score-renderer";

describe("ScoreRenderer.load forwards transpose into the worker message", () => {
  it("posts transpose in the load message", async () => {
    // biome-ignore lint/suspicious/noExplicitAny: test capture array
    const posted: any[] = [];
    class FakeWorker {
      onmessage: ((e: MessageEvent) => void) | null = null;
      onerror: ((e: ErrorEvent) => void) | null = null;
      // biome-ignore lint/suspicious/noExplicitAny: synthetic message
      postMessage(msg: any) {
        posted.push(msg);
        queueMicrotask(() => {
          this.onmessage?.({
            data: {
              requestId: msg.requestId,
              payload: {
                ir: { pieceId: `${msg.pieceId}`, bars: [], pages: [], notes: {}, pageWidth: 2400 },
                pageSvgs: ["<svg/>"],
              },
            },
          } as MessageEvent);
        });
      }
      terminate() {}
    }
    // @ts-expect-error override global Worker for the test
    globalThis.Worker = FakeWorker;

    const { api } = await import("./api");
    const orig = api.scores.getData;
    // @ts-expect-error stub network fetch
    api.scores.getData = async () => new ArrayBuffer(8);

    try {
      const r = new ScoreRenderer();
      await r.load("hanon_001", 2);
      const loadMsg = posted.find((m) => m.type === "load");
      expect(loadMsg).toBeDefined();
      expect(loadMsg.transpose).toBe(2);
    } finally {
      // @ts-expect-error restore
      api.scores.getData = orig;
    }
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bunx vitest run src/lib/score-worker.transpose.integration.test.ts -t "forwards transpose"
```
Expected: FAIL — `loadMsg.transpose` is `undefined` because `ScoreRenderer.load` does not yet accept or forward a transpose argument (its `sendRequest` posts `{ type: "load" }` with no `transpose` field).

- [ ] **Step 3: Implement the minimum to make the test pass**

(a) In `apps/web/src/lib/score-worker.ts`, add `transpose` to the `load` message type and thread it through the cache key and `loadPiece` call.

Extend the `WorkerInMsg` union's `load` variant:

```typescript
	| { type: "load";             requestId: string; pieceId: string; bytes: ArrayBuffer; transpose?: number }
```

In the handler, derive the composite cache key and use it everywhere `msg.pieceId` keys `toolkitCache`:

```typescript
			const cacheKey =
				msg.type === "load"
					? `${msg.pieceId}:${msg.transpose ?? 0}`
					: msg.pieceId;
			const cached = toolkitCache.get(cacheKey);
```

Replace the three `toolkitCache.get(msg.pieceId)` / `toolkitCache.set(msg.pieceId, ...)` occurrences in the load block with `cacheKey`, and pass transpose to `loadPiece`:

```typescript
					const loadPromise = loadPiece(
						msg.bytes,
						{ module: verovioModule, ToolkitClass: VerovioToolkitClass },
						msg.pieceId,
						msg.transpose,
					);
					toolkitCache.set(cacheKey, loadPromise);
					result = await loadPromise;
					if (toolkitCache.get(cacheKey) === loadPromise) {
						toolkitCache.set(cacheKey, result);
					}
```

> Non-`load` messages (`get_page`, `get_clip`, etc.) carry no transpose; they continue to key by `msg.pieceId`. Since a transposed load was cached under `${pieceId}:${n}`, a follow-up `get_clip` for a transposed primitive is out of scope for this slice (corpus_drill renders the whole primitive, not clips); keep non-load keying unchanged.

(b) In `apps/web/src/lib/score-renderer.ts`, accept and forward `transpose`, and key the main-thread caches by `pieceId:transpose`:

```typescript
  async load(
    pieceId: string,
    transpose?: number,
  ): Promise<{ ir: ScoreIR; pageSvgs: string[] } | "failed"> {
    const key = `${pieceId}:${transpose ?? 0}`;
    await this.ensureBytes(pieceId);
    const needsBytes = !this.sentPieceIds.has(key);
    const bytes = needsBytes ? this.bytesCache.get(pieceId) : undefined;
    if (needsBytes && bytes === undefined) {
      throw new Error(`Score bytes missing after fetch for pieceId: ${pieceId}`);
    }
    if (needsBytes) this.sentPieceIds.add(key);

    try {
      const payload = await this.sendRequest<{ ir: ScoreIR; pageSvgs: string[] }>(
        pieceId,
        { type: "load", transpose },
        bytes,
      );
      this.irCache.set(key, payload.ir);
      return payload;
    } catch (err) {
      this.sentPieceIds.delete(key);
      Sentry.captureException(err);
      return "failed";
    }
  }
```

> `ensureBytes`/`bytesCache` stay keyed by `pieceId` (bytes are transpose-independent — Verovio applies transpose at engrave time). `sentPieceIds` and `irCache` move to the composite `key` so transpose variants don't collide. `getIR(pieceId)` currently keys by bare `pieceId`; update it to `getIR(pieceId, transpose?)`:

```typescript
  getIR(pieceId: string, transpose?: number): ScoreIR | null {
    return this.irCache.get(`${pieceId}:${transpose ?? 0}`) ?? null;
  }
```

> Check for existing `getIR(` callers and pass no transpose (they default to `:0`, byte-identical to the old bare-`pieceId` key for real pieces). Search: `grep -rn "getIR(" apps/web/src`. If any caller exists, it keeps working because the composite key for the no-transpose case is `${pieceId}:0`, and real pieces are always loaded with no transpose, so their IR is now stored under `${pieceId}:0` — confirm the same key is used on both store and read (it is, both default to `:0`).

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bunx vitest run src/lib/score-worker.transpose.integration.test.ts
```
Expected: PASS (all transpose tests, including the forwarding test). Then run the full score suite to confirm no regression in real-piece load (the `:0` keying must not break existing tests):

```bash
cd apps/web && bunx vitest run src/lib/score-worker.test.ts src/lib/score-worker.integration.test.ts src/lib/score-renderer.test.ts
```
Expected: PASS (unchanged).

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/lib/score-renderer.ts apps/web/src/lib/score-worker.ts apps/web/src/lib/score-worker.transpose.integration.test.ts && git commit -m "feat(#46): ScoreRenderer.load forwards transpose with composite cache key"
```

---

## Spec coverage map

| Spec requirement | Task |
|---|---|
| (A) build() emits valid .mxl from committed .xml | 3 |
| (A) idempotent + fail-loud on bad xml | 4 |
| (A) 22 assets committed | 4 (commit step) |
| (B) just build-exercise-assets | 5 |
| (B) just seed-exercise-assets (local R2, flat key) | 5 |
| (C) loadPiece transpose param, string only at setOptions | 6 |
| (C) transpose 0 == no-transpose (real-piece lock) | 6 |
| (C) ScoreRenderer.load(pieceId, transpose), composite cache key | 7 |
| (D) Verovio↔partitura faithful-shift oracle (in-range, all 22) + documented 2-file baseline-divergence xfail | 1 |
| (D) off-keyboard rejection symmetry | 2 |
| verovio added to model dev deps | 1 |
```

---

## Challenge Review

Reviewed against actual code and empirical execution of the oracle's core assertion (verovio 6.2.1 + partitura, the exact API sequence in Task 1) on the real committed primitives.

### CEO Pass

**Premise — sound.** The problem is real and verified: `corpus_drill` renders stub text today (MEMORY confirms), the 22 `.xml` are committed but never in R2, and `ScoreRenderer.load`/`loadPiece` cannot transpose. The three-slice decomposition (offline assets / R2 seed / renderer transpose) plus a verification harness is the most direct path. No simpler framing exists — transpose-on-demand via Verovio's load-time `transpose` option avoids pre-materializing 12 keys × 22 primitives.

**Scope — appropriately bounded.** Spec's "Not in scope" list (teacher selection, off-keyboard gate, playback, corpus breadth, ranking) is disciplined. The plan does not drift beyond the spec. 8 files touched, 1 new production module (`build_render_assets.py`) + 1 new test module — under the complexity threshold.

**Existing coverage — correctly reused.** `wrap_as_mxl_zip` + `_strip_doctype` (`score_library/upload.py:16,29`) are real and do exactly what Task 3 delegates to. The data endpoint reuse (no API change, no migration) is verified against the spec's claim. `transforms.py::transpose` (lines 94–112) raises `ValueError(... "outside piano range" ...)` exactly as Task 2 asserts (`match="outside piano"` is a valid substring).

**12-month alignment — neutral-to-positive.** Adds a committed-asset pipeline + a transpose surface that S4 (teacher-driven key selection) builds on. No tech debt that conflicts with the ideal; the `:0` composite-key default keeps real pieces byte-identical.

[OBS] — The spec documents key decisions and trade-offs (MusicXML-not-MEI, endpoint reuse, semitone-int surface) but no rejected *alternatives* for the oracle approach beyond "headless Node → Python binding." Minor; not blocking.

### Engineering Pass

**Architecture — the feature slices (A/B/C) are sound; the harness (Group 0) is mis-specified.**

Groups A, B, C check out against the code:
- Task 3/4 `build()` delegates to proven `wrap_as_mxl_zip`; fail-loud + idempotent logic is correct and matches CLAUDE.md "explicit exceptions over silent fallbacks."
- Task 5 recipes mirror existing `seed-scores`/`seed-fingerprint`; `set -euo pipefail` + zero-count guard is correct.
- Task 6/7 renderer plumbing: the `applyOpts` refactor targets the three real `tk = new ToolkitClass(module); tk.setOptions(VEROVIO_OPTS)` sites in `score-worker.ts` (lines 225–226, 234–235, 255–256). Composite cache key `${pieceId}:${transpose ?? 0}` is correct. **Verified Task 7's back-compat claim:** the only `getIR(` / `.load(` callers (`app.sandbox.tsx:768`, `ScorePanel.tsx:284`, `ExerciseSetCard.tsx:171`, `score-renderer.test.ts`) all pass no transpose, so they resolve to the `:0` slot and keep working. `api.scores.getData` is a reassignable object property (`api.ts:411`), so Task 7's stub is viable.

[BLOCKER] (confidence: 10/10) — **Task 1's oracle equivalence assertion fails empirically for 2 of the 22 primitives, which halts the entire build (Group 0 must be green before A/B/C dispatch).** I ran the exact Task 1 code (verovio 6.2.1, partitura, `transforms.py::transpose`) over all 22 committed `.xml` across offsets −5..+6. Result: **20 PASS, 2 FAIL** — `burgmuller_001` and `czerny_001`. Two independent root causes:
  1. **Repeat expansion.** `burgmuller_001.xml` contains two `<repeat>` pairs (forward/backward, lines 16/1110/1190/2388). Verovio's `renderToTimemap` expands repeats, emitting **420** note onsets; partitura's `iter_all(Note)` reads the **248** literal score notes. The pitch-class multisets cannot match — they differ even at `transpose=0`.
  2. **Accidental/enharmonic spelling divergence.** `czerny_001.xml` has **no repeats** and **equal note sums (392 == 392)**, yet still fails: partitura surfaces pitch classes {1, 3, 6} (4 chromatic notes) that Verovio's timemap collapses into diatonic neighbors (Verovio's counts for PC 4/0/5/2 are each one higher). This is a `midi_pitch % 12` disagreement between the two engines on how a handful of accidentals are realized — orthogonal to repeats.

The spec's "empirically confirmed... histogram exactly equal to the base shifted +2 mod 12" (Design §"Verovio Python binding") was evidently validated on a repeat-free, accidental-free fixture (the Bach fixture per MEMORY), **not these primitives**. As written, `test_committed_xml_files_present` passes but `test_verovio_matches_partitura_pitch_classes[burgmuller_001]` and `[czerny_001]` go red and stay red, blocking Tasks 3–7.

**What must change before executing:** Pick one and update Task 1 (and the spec's Verification Architecture / Open Questions):
  - (a) **Normalize for repeats** — set Verovio `{"expand": ""}` off or de-duplicate by element id AND additionally reconcile accidental spelling; OR
  - (b) **Compare a repeat-collapsed, spelling-robust invariant** — e.g. compare the set of distinct (octave-folded) pitch *names* via Verovio's MEI/humdrum export rather than the played timemap, so neither repeats nor MIDI-realization of accidentals perturb it; OR
  - (c) **Narrow the oracle's corpus to the cases the invariant actually holds for** and assert that explicitly (e.g. parametrize over the 20 repeat-free/diatonic primitives, and add a *separate, correct* assertion for the 2 outliers) — but (c) weakens the "every committed primitive" guarantee the spec's Canonical Success State promises, so it needs a spec amendment, not just a plan edit.

Note the `== 22` count assertion (`test_committed_xml_files_present`, plan line 108) and the all-22 parametrization must stay internally consistent with whichever fix is chosen — if the oracle narrows to 20, the count guard must not silently mask a missing-file regression.

[RISK — RETIRED by loop 1] (confidence: 7/10) — **Task 3 inner-note-count assertion may also trip on the same content.** `test_build_emits_valid_mxl_with_matching_note_count` compares `_note_count_xml(inner) == _note_count_xml(xml_path)` — both via partitura, so this is partitura-vs-partitura and should hold (build only DOCTYPE-strips + ZIP-wraps, no re-export). RESOLUTION: the speculative "minimal partitura re-export" dangled in Task 3 Step 4 has been REMOVED from the plan. /challenge's empirical run produced Verovio timemaps for all 22 `.xml` (they all render after DOCTYPE-strip), retiring the "do they render?" residual risk that motivated a normalization pass; and a partitura load→save round-trip can drop/re-spell notes (especially `burgmuller_001`'s repeat structure) and would break the inner≠source count lock. `build()` is now contractually DOCTYPE-strip + ZIP-wrap only, RAISING (no re-export fallback) if a `.xml` genuinely fails to load — that is a finding to surface, not paper over.

**Module Depth:**
- `build_render_assets.py` — Interface: `build(xml_dir, out_dir) -> list[Path]` (1 public fn) + private `_existing_inner_xml`. Hides partitura-validate + DOCTYPE-strip + ZIP-wrap + idempotent-skip + fail-loud. **DEEP.**
- `loadPiece` transpose path — adds one optional `number` param hiding Verovio's load-time engraving + cache-key derivation. **DEEP.**
- `ScoreRenderer.load` — one optional param, composite key threaded through `sentPieceIds`/`irCache`/`getIR`. **DEEP.**
- Oracle test module — it is the harness, not a production module. N/A.

No shallow modules.

**Code Quality:**
[OBS] — `build()` uses `except Exception as e:  # noqa: BLE001` to re-raise as `ValueError` naming the file. This is the *acceptable* form of catch-all (it re-raises with context, does not swallow) and matches CLAUDE.md's fail-loud rule. Not a finding.
[OBS] — `_existing_inner_xml` returns `None` on `BadZipFile` (idempotency probe), which then forces a rewrite — correct, not a silent fallback.

**Test Philosophy — clean.** All tests exercise public interfaces (`build()`, `loadPiece`, `ScoreRenderer.load`) and assert observable behavior (valid ZIP, matching note count, different SVG, forwarded message field). The Task 7 `FakeWorker` stubs an *external boundary* (the Worker + the network via `api.scores.getData`), not an internal collaborator — allowed. No shape-only tests, no internal mocking.

[OBS] — Task 6's `stripIds` SVG comparison (regex-strip `id="..."` then compare) is a legitimate behavior lock given Verovio randomizes element ids per `loadData` (confirmed by the existing `score-worker.integration.test.ts` cache-eviction test). Good.

**Vertical Slice — mostly clean, one soft spot.**
[RISK] (confidence: 6/10) — **Task 4 admits both its tests may already pass against Task 3 code** (`test_build_is_idempotent` is "a lock, not a driver"; the plan even instructs "watch each fail by temporarily reverting the relevant `build()` guard"). This is a borderline horizontal-slice smell: idempotency was implemented in Task 3, so Task 4 partly tests pre-built behavior. It is salvageable because the *driving* test (`test_build_raises_naming_bad_xml`) does bite, and the plan correctly mandates a revert-to-watch-it-fail step. Watch during execution that the bad-XML guard is genuinely seen failing before the lock is trusted. Not a blocker.

**Test Coverage:**
```
[+] build_render_assets.py::build()
    ├── happy path (22 valid .xml → 22 .mxl)        [TESTED ★★] Task 3
    ├── idempotent re-run                           [TESTED ★★] Task 4
    ├── bad XML → ValueError naming file            [TESTED ★★★] Task 4
    └── empty xml_dir → FileNotFoundError           [GAP] no test (low sev — raise exists)
[+] loadPiece(transpose)
    ├── transpose:2 ≠ transpose:0 SVG               [TESTED ★★] Task 6
    ├── transpose:0 == omitted (real-piece lock)    [TESTED ★★★] Task 6
    └── off-keyboard transpose at engrave time      [N/A — gate deferred to S4]
[+] ScoreRenderer.load(pieceId, transpose)
    └── forwards transpose into load message        [TESTED ★★] Task 7
```
[OBS] — `build()`'s `FileNotFoundError` on empty dir (raise at line ~392) has no test. Low severity; the raise is unambiguous and not on a critical path.

**Failure Modes — no silent failures.**
- Task 3/4: bad XML raises naming the file; no skip-and-continue. ✓
- Task 5: `set -euo pipefail` + explicit zero-count guard → recipe exits non-zero on R2 put failure. ✓ Partial-seed (some objects put, then a later put fails) leaves R2 partially populated, but the recipe exits non-zero and re-run is idempotent overwrite — acceptable, surfaced.
- Task 6/7: a failed transposed load returns "failed" + `Sentry.captureException` upstream (no untransposed fallback) — matches spec intent. ✓

### Presumption Inventory

| Assumption | Verdict | Reason |
|---|---|---|
| 22 committed `.xml` exist, gitignored `.db`/`.mid` excluded | SAFE | Verified: 22 `.xml`, `git check-ignore` empty for them |
| `transforms.py::transpose(part, semitones)` raises "outside piano range" | SAFE | Verified at lines 104–108 |
| `transpose()` returns `Variant` with `.part` | SAFE | Verified line 112 |
| `wrap_as_mxl_zip` / `_strip_doctype` reusable | SAFE | Verified `upload.py:16,29` |
| **Verovio `transpose` PC-multiset == partitura for EVERY primitive (absolute)** | **REPLACED** | FALSIFIED empirically (2/22: repeat expansion + accidental spelling). Loop 1 reframed the oracle to the faithful-SHIFT invariant (intra-engine shift-by-N), which holds for all 22 in both engines; cross-engine absolute baseline kept as a secondary witness with the 2 known divergences xfail(strict=True) |
| Verovio Python binding exposes `setOptions/loadData/renderToTimemap/getMIDIValuesForElement` | SAFE | Verified: all work; `getMIDIValuesForElement` returns `{pitch, duration, time}` dict |
| `verovio` Python binding installs cleanly as dev dep | SAFE | Verified: `verovio==6.2.1` installs |
| Python verovio (6.2.1) ≈ web verovio (6.1.0) "same WASM core" | VALIDATE | Minor-version skew; the binding versions are not pinned-equal. Oracle correctness must not depend on exact-version timemap parity |
| `api.scores.getData` reassignable for Task 7 stub | SAFE | Verified `api.ts:411` object property |
| `getIR(`/`.load(` callers all pass no transpose (back-compat) | SAFE | Verified: only `app.sandbox.tsx`, `ScorePanel`, `ExerciseSetCard`, tests — none transpose |
| Three `tk = new ToolkitClass` sites in `loadPiece` for `applyOpts` | SAFE | Verified lines 225, 234, 255 |
| Task 3 inner-note-count holds without partitura re-export | VALIDATE | Holds for DOCTYPE-strip+wrap only; a re-export pass would risk count drift (see RISK) |
| `build()` empty-dir raises `FileNotFoundError` | SAFE | Verified at impl; untested but unambiguous |

### Summary
[BLOCKER] count: 1
[RISK]    count: 3
[QUESTION] count: 0

VERDICT: NEEDS_REWORK — Task 1's oracle asserts Verovio↔partitura pitch-class-multiset equality for all 22 committed primitives, but this is empirically false for `burgmuller_001` (Verovio expands its `<repeat>` sections: 420 vs 248 notes) and `czerny_001` (accidental/enharmonic spelling divergence despite equal 392-note sums). Because the plan gates Groups A/B/C on a green Group 0, Task 1 halts the entire build. Rework Task 1's comparison to be repeat- and accidental-robust (and amend the spec's Canonical Success State if the corpus is narrowed). Secondary: do NOT let `build()` grow the speculative partitura re-export (Task 3 Step 4) — it would risk the inner-note-count lock; keep `build()` to DOCTYPE-strip + ZIP-wrap only.

### Loop 1 resolution (2026-06-15)

Both findings addressed; the rework does NOT touch the A/B/C feature approach.

- **BLOCKER (Task 1):** Reframed the oracle from an ABSOLUTE Verovio↔partitura multiset equality to a faithful-SHIFT invariant. PRIMARY (`test_faithful_shift_invariant_holds_in_each_engine`, all 22): within each engine, transpose-by-N == shift-by-N (mod 12) of that engine's transpose-0 baseline — proves each engine transposes by exactly N, robust to repeat expansion and accidental realization (both intra-engine). SECONDARY (`test_baseline_cross_engine_agreement`, 20 of 22): cross-engine baseline multiset agreement, with `burgmuller_001` (repeat expansion) and `czerny_001` (accidental realization) marked `pytest.mark.xfail(strict=True)`; `_BASELINE_DIVERGENCE` is a length-2 named constant locked by `test_baseline_divergence_set_is_exactly_the_two_known_cases` so an unexpected new divergence fails loudly. `test_committed_xml_files_present` still asserts exactly 22 .xml. The transpose guarantee is NOT narrowed — the PRIMARY assertion covers all 22.
- **SECONDARY CORRECTION (Task 3 Step 4):** Removed the speculative partitura re-export normalization language. `build()` is DOCTYPE-strip + `wrap_as_mxl_zip` ONLY; it RAISES naming the file on a genuine load failure (no re-export fallback). The "do they render?" residual risk was retired empirically — /challenge produced Verovio timemaps for all 22 .xml. See the [RISK — RETIRED by loop 1] note above.
