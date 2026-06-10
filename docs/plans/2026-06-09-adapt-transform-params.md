# ADAPT Layer: Passage-Aware Transform Parameters Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** `build_briefing` computes transpose interval and excerpt span from the student's actual diagnosed passage instead of hardcoded constants.
**Spec:** docs/specs/2026-06-09-adapt-transform-params-design.md
**Style:** Follow the project's coding standards (CLAUDE.md / AGENTS.md)

---

## Pre-flight: Known Pre-existing Test Failures

`model/tests/exercise_corpus/test_transforms.py` has **8 pre-existing failures** because `model/data/midi/` is gitignored and absent in this worktree. These are NOT regressions. Success criteria for this plan: all tests in `test_keys.py`, `test_briefing.py`, `test_tags.py` pass; `test_transforms.py` failure count does not increase beyond 8.

Run baseline: `cd model && uv run pytest tests/exercise_corpus/ -q 2>&1 | tail -5`

---

## Task Groups

Group A (independent): Task 1
Group B (sequential, depends on A): Task 2, Task 3  [Task 2 and 3 touch different files; safe to parallelize within B]
Group C (sequential, depends on B): Task 4
Group D (sequential, depends on C): Task 5, Task 6

---

### Task 1: `parse_key_to_pc` and `transpose_interval` pure helpers

**Group:** A (parallel with Task 2)

**Behavior being verified:** `parse_key_to_pc` maps key signature strings to pitch-class integers 0–11 with enharmonic equivalence and mode stripping; raises `ValueError` on unrecognizable input. `transpose_interval` returns the nearest-octave semitone shift in [-5, +6]; same pitch returns 0; tritone resolves to +6.

**Interface under test:**
```python
from exercise_corpus.keys import parse_key_to_pc, transpose_interval
```

**Files:**
- Create: `model/src/exercise_corpus/keys.py`
- Create: `model/tests/exercise_corpus/test_keys.py`

---

- [ ] **Step 1: Write the failing test**

Create `model/tests/exercise_corpus/test_keys.py`:

```python
"""Tests for keys.py -- pure key-resolution helpers."""

import pytest

from exercise_corpus.keys import parse_key_to_pc, transpose_interval


# --- parse_key_to_pc ---

def test_parse_c_major():
    assert parse_key_to_pc("C major") == 0

def test_parse_c_bare():
    assert parse_key_to_pc("C") == 0

def test_parse_a_minor():
    assert parse_key_to_pc("Am") == 9

def test_parse_a_minor_space():
    assert parse_key_to_pc("A minor") == 9

def test_parse_eb():
    assert parse_key_to_pc("Eb") == 3

def test_parse_eb_minor():
    assert parse_key_to_pc("Ebm") == 3

def test_parse_cs():
    assert parse_key_to_pc("C#") == 1

def test_parse_db():
    assert parse_key_to_pc("Db") == 1

def test_parse_gb():
    assert parse_key_to_pc("Gb") == 6

def test_parse_fs():
    assert parse_key_to_pc("F#") == 6

def test_parse_bb():
    assert parse_key_to_pc("Bb") == 10

def test_parse_g_major():
    assert parse_key_to_pc("G major") == 7

def test_parse_csm():
    assert parse_key_to_pc("C#m") == 1

def test_parse_unknown_raises():
    with pytest.raises(ValueError, match="unparseable"):
        parse_key_to_pc("Q major")

def test_parse_empty_raises():
    with pytest.raises(ValueError, match="unparseable"):
        parse_key_to_pc("")

def test_parse_garbage_raises():
    with pytest.raises(ValueError, match="unparseable"):
        parse_key_to_pc("not a key")


# --- transpose_interval ---

def test_same_key_is_zero():
    assert transpose_interval(0, 0) == 0

def test_same_key_any_pc_is_zero():
    assert transpose_interval(9, 9) == 0

def test_c_to_eb_is_plus_3():
    # C=0 -> Eb=3
    assert transpose_interval(0, 3) == 3

def test_c_to_a_is_minus_3_not_plus_9():
    # C=0 -> A=9; d=9 > 6 so d -= 12 -> -3
    assert transpose_interval(0, 9) == -3

def test_c_to_g_is_plus_7_reduced():
    # C=0 -> G=7; d=7 > 6 so d -= 12 -> -5
    assert transpose_interval(0, 7) == -5

def test_c_to_f_is_plus_5():
    # C=0 -> F=5; d=5 <= 6
    assert transpose_interval(0, 5) == 5

def test_tritone_is_plus_6():
    # C=0 -> F#=6; d=6, convention: +6
    assert transpose_interval(0, 6) == 6

def test_eb_to_c_is_minus_3():
    # Eb=3 -> C=0; d = (0-3)%12 = 9 > 6 -> 9-12 = -3
    assert transpose_interval(3, 0) == -3

def test_range_is_bounded():
    for from_pc in range(12):
        for to_pc in range(12):
            result = transpose_interval(from_pc, to_pc)
            assert -5 <= result <= 6, f"out of range for {from_pc}->{to_pc}: {result}"
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai-wt-issue41/model && uv run pytest tests/exercise_corpus/test_keys.py -q
```

Expected: FAIL — `ModuleNotFoundError: No module named 'exercise_corpus.keys'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `model/src/exercise_corpus/keys.py`:

```python
"""Key-resolution helpers for the ADAPT layer.

Three pure/near-pure helpers:
  parse_key_to_pc   -- key signature string -> pitch class integer 0-11
  transpose_interval -- nearest-octave semitone shift from one pc to another
  load_passage_key  -- resolve key_signature from a piece JSON on disk

All enharmonic normalization and mode stripping live here, hidden from callers.
Default scores_dir is anchored to __file__, not CWD, so it survives `just`
recipe cwd shifts.
"""

import json
from pathlib import Path

# Pitch class lookup. Enharmonic pairs share a value.
# Keys are canonical tonic names (uppercase root + optional accidental).
_PC: dict[str, int] = {
    "C": 0,
    "C#": 1, "Db": 1,
    "D": 2,
    "D#": 3, "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6, "Gb": 6,
    "G": 7,
    "G#": 8, "Ab": 8,
    "A": 9,
    "A#": 10, "Bb": 10,
    "B": 11,
}

# Anchored default: keys.py lives at model/src/exercise_corpus/keys.py,
# so parents[2] is model/src/exercise_corpus/../../../ -> model/
_DEFAULT_SCORES_DIR = Path(__file__).resolve().parents[2] / "data" / "scores"


def parse_key_to_pc(key_signature: str) -> int:
    """Map a key signature string to a pitch class integer 0-11.

    Strips trailing mode tokens (" major", " minor", "m") so only the tonic
    matters. Raises ValueError on unrecognizable input.

    Examples:
        "C major" -> 0
        "Am" -> 9
        "Eb" -> 3
        "C#m" -> 1
        "F#" -> 6
    """
    s = key_signature.strip()
    # Strip trailing " major" / " minor" (case-insensitive)
    for suffix in (" major", " minor"):
        if s.lower().endswith(suffix):
            s = s[: -len(suffix)].strip()
            break
    # Strip trailing "m" (minor shorthand) — only if not the entire string
    if s.endswith("m") and len(s) > 1:
        s = s[:-1]
    if s in _PC:
        return _PC[s]
    raise ValueError(f"unparseable key signature: {key_signature!r}")


def transpose_interval(from_pc: int, to_pc: int) -> int:
    """Nearest-octave semitone shift from from_pc to to_pc.

    Returns an integer in [-5, +6]. Tritone (d=6) resolves to +6 by convention.

    Examples:
        (0, 0) -> 0     (same key, no shift)
        (0, 3) -> +3    (C -> Eb)
        (0, 9) -> -3    (C -> A, nearest is down 3, not up 9)
        (0, 6) -> +6    (tritone, +6 by convention)
    """
    d = (to_pc - from_pc) % 12
    if d > 6:
        d -= 12
    return d


def load_passage_key(piece_id: str, scores_dir: Path | None = None) -> str | None:
    """Resolve the key signature string for a piece from its score JSON.

    Args:
        piece_id: identifier used as the JSON filename stem.
        scores_dir: directory containing <piece_id>.json files. Defaults to
            model/data/scores/ anchored to this file's location.

    Returns:
        The `key_signature` string from the JSON, or None if the field is null.

    Raises:
        FileNotFoundError: if <scores_dir>/<piece_id>.json does not exist.
    """
    if scores_dir is None:
        scores_dir = _DEFAULT_SCORES_DIR
    path = Path(scores_dir) / f"{piece_id}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"score JSON not found for piece_id {piece_id!r}: {path}"
        )
    with open(path) as f:
        data = json.load(f)
    return data.get("key_signature")
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai-wt-issue41/model && uv run pytest tests/exercise_corpus/test_keys.py -q
```

Expected: PASS (all tests green)

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai-wt-issue41 && git add model/src/exercise_corpus/keys.py model/tests/exercise_corpus/test_keys.py && git commit -m "feat(#41): add keys.py -- parse_key_to_pc, transpose_interval pure helpers"
```

---

### Task 2: `load_passage_key` with fixture and missing-file error

**Group:** B (depends on Group A — Task 1 must be committed first since this extends test_keys.py)

**Behavior being verified:** `load_passage_key` returns the `key_signature` field from a committed JSON fixture on the happy path; raises `FileNotFoundError` for a nonexistent piece ID; returns `None` when the fixture's `key_signature` field is null.

**Interface under test:**
```python
from exercise_corpus.keys import load_passage_key
```

**Files:**
- Modify: `model/tests/exercise_corpus/test_keys.py` (add `load_passage_key` tests)

**Files:**
- Modify: `model/tests/exercise_corpus/test_keys.py`

---

- [ ] **Step 1: Write the failing test**

Append to `model/tests/exercise_corpus/test_keys.py`:

```python
# --- load_passage_key ---

from pathlib import Path


def test_load_passage_key_returns_key_from_committed_fixture():
    # Uses the git-committed bach.prelude.bwv_846.json which has key_signature "C major"
    scores_dir = Path(__file__).resolve().parents[3] / "model" / "data" / "scores"
    result = load_passage_key("bach.prelude.bwv_846", scores_dir=scores_dir)
    assert result == "C major"


def test_load_passage_key_raises_for_missing_piece(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="nonexistent_piece"):
        load_passage_key("nonexistent_piece", scores_dir=tmp_path)


def test_load_passage_key_returns_none_when_key_signature_is_null(tmp_path: Path):
    fixture = tmp_path / "no_key_piece.json"
    fixture.write_text('{"piece_id": "no_key_piece", "key_signature": null}')
    result = load_passage_key("no_key_piece", scores_dir=tmp_path)
    assert result is None
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai-wt-issue41/model && uv run pytest tests/exercise_corpus/test_keys.py -q -k "load_passage_key"
```

Expected: FAIL — `ImportError` or `NameError: load_passage_key` (Task 1 not yet committed if running truly in parallel; if Task 1 is done, FAIL because `load_passage_key` isn't imported in the test yet).

- [ ] **Step 3: Implement — add import to test file**

Add `load_passage_key` to the import at the top of `model/tests/exercise_corpus/test_keys.py`:

```python
from exercise_corpus.keys import parse_key_to_pc, transpose_interval, load_passage_key
```

The implementation of `load_passage_key` itself is already in `keys.py` from Task 1. No production code changes needed in this task.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai-wt-issue41/model && uv run pytest tests/exercise_corpus/test_keys.py -q
```

Expected: PASS (all test_keys.py tests green including load_passage_key tests)

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai-wt-issue41 && git add model/tests/exercise_corpus/test_keys.py && git commit -m "test(#41): add load_passage_key tests to test_keys.py"
```

---

### Task 3: `TagSet.key` field + `technique_tags.toml` update + existing test updates

**Group:** B (parallel with Task 2 — touches different files: tags.py, technique_tags.toml, test_tags.py, test_briefing.py)

**Behavior being verified:** `TagSet` requires a `key` field; `load_tags` raises `ValueError` if any TOML entry lacks `key`; all 22 technique_tags.toml entries declare `key = "C"`; existing `test_tags.py` tests pass with updated TOML fixtures; existing `test_briefing.py` `_tags` helper updated to pass `key=`.

**Interface under test:**
```python
from exercise_corpus.tags import TagSet, load_tags
```

**Files:**
- Modify: `model/src/exercise_corpus/tags.py`
- Modify: `model/src/exercise_corpus/technique_tags.toml`
- Modify: `model/tests/exercise_corpus/test_tags.py`
- Modify: `model/tests/exercise_corpus/test_briefing.py`

---

- [ ] **Step 1: Write the failing test**

In `model/tests/exercise_corpus/test_tags.py`, add two tests and update the existing TOML fixture strings:

The three existing tests use TOML fixture strings that lack `key`. They will fail once `load_tags` enforces the field. Update those test fixtures AND add new tests:

```python
# Replace the existing test_load_tags_reads_valid_table to include key:
def test_load_tags_reads_valid_table(tmp_path: Path):
    toml = _write_toml(
        tmp_path,
        """
[hanon_001]
dimensions = ["articulation", "timing"]
techniques = ["finger_independence", "evenness"]
key = "C"

[burgmuller_001]
dimensions = ["phrasing", "interpretation"]
key = "C"
""",
    )
    tags = load_tags(toml, known_primitive_ids={"hanon_001", "burgmuller_001"})

    assert set(tags) == {"hanon_001", "burgmuller_001"}
    assert isinstance(tags["hanon_001"], TagSet)
    assert tags["hanon_001"].dimensions == frozenset({"articulation", "timing"})
    assert tags["hanon_001"].techniques == frozenset({"finger_independence", "evenness"})
    assert tags["hanon_001"].key == "C"
    assert tags["burgmuller_001"].key == "C"
    assert tags["burgmuller_001"].techniques == frozenset()


def test_load_tags_rejects_entry_missing_key(tmp_path: Path):
    toml = _write_toml(
        tmp_path,
        """
[hanon_001]
dimensions = ["timing"]
""",
    )
    with pytest.raises(ValueError, match="missing required 'key'"):
        load_tags(toml, known_primitive_ids={"hanon_001"})


# Update test_load_tags_rejects_unknown_dimension to add key = "C":
def test_load_tags_rejects_unknown_dimension(tmp_path: Path):
    toml = _write_toml(
        tmp_path,
        """
[hanon_001]
dimensions = ["timing", "tempo"]
key = "C"
""",
    )
    with pytest.raises(ValueError, match="unknown dimension"):
        load_tags(toml, known_primitive_ids={"hanon_001"})


# Update test_load_tags_rejects_unknown_primitive to add key = "C":
def test_load_tags_rejects_unknown_primitive(tmp_path: Path):
    toml = _write_toml(
        tmp_path,
        """
[ghost_999]
dimensions = ["timing"]
key = "C"
""",
    )
    with pytest.raises(ValueError, match="unknown primitive_id"):
        load_tags(toml, known_primitive_ids={"hanon_001"})
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai-wt-issue41/model && uv run pytest tests/exercise_corpus/test_tags.py -q
```

Expected: FAIL — `TypeError: TagSet.__init__() missing required argument 'key'` or similar (TagSet doesn't have `key` yet).

- [ ] **Step 3: Implement**

**3a. Update `model/src/exercise_corpus/tags.py`** — add `key: str` to TagSet and enforce in load_tags:

```python
@dataclass(frozen=True)
class TagSet:
    """The dimensions an exercise can address plus free technique labels."""

    dimensions: frozenset[str]
    techniques: frozenset[str]
    key: str
```

In `load_tags`, after the dims validation loop, add key enforcement:

```python
        raw_key = entry.get("key")
        if raw_key is None:
            raise ValueError(
                f"missing required 'key' field for {primitive_id!r} in {path}"
            )
        tags[primitive_id] = TagSet(
            dimensions=frozenset(dims), techniques=frozenset(techs), key=raw_key
        )
```

(Remove the old `tags[primitive_id] = TagSet(...)` line and replace with the above.)

**3b. Update `model/src/exercise_corpus/technique_tags.toml`** — add `key = "C"` to all 22 entries. Each entry currently looks like:

```toml
[hanon_001]
dimensions = ["articulation", "timing"]
techniques = ["finger_independence", "evenness"]
```

Add `key = "C"` as a third line to every entry. All 22 entries (hanon_001 through hanon_020, czerny_001, burgmuller_001) are C major.

**3c. Update `model/tests/exercise_corpus/test_briefing.py`** — two changes:

First, update the `_tags` helper which constructs `TagSet` directly:

```python
def _tags(mapping: dict[str, list[str]]) -> dict[str, TagSet]:
    return {pid: TagSet(frozenset(dims), frozenset()) for pid, dims in mapping.items()}
```

Change to:

```python
def _tags(mapping: dict[str, list[str]]) -> dict[str, TagSet]:
    return {pid: TagSet(frozenset(dims), frozenset(), key="C") for pid, dims in mapping.items()}
```

Second, update the `_diagnosis()` helper's default `piece_id` from `"fur_elise"` to `"bach.prelude.bwv_846"`:

```python
# Before:
def _diagnosis(**kw) -> Diagnosis:
    return Diagnosis(
        dimension=kw.get("dimension", "timing"),
        severity=kw.get("severity", "moderate"),
        bar_range=kw.get("bar_range", (3, 6)),
        piece_id=kw.get("piece_id", "fur_elise"),
    )
```

Change to:

```python
def _diagnosis(**kw) -> Diagnosis:
    return Diagnosis(
        dimension=kw.get("dimension", "timing"),
        severity=kw.get("severity", "moderate"),
        bar_range=kw.get("bar_range", (3, 6)),
        piece_id=kw.get("piece_id", "bach.prelude.bwv_846"),
    )
```

Rationale: `model/data/scores/bach.prelude.bwv_846.json` is git-committed and has `key_signature: "C major"`. When `build_briefing` is wired in Task 5 to call `load_passage_key(diagnosis.piece_id, scores_dir)`, the pre-existing tests will use the anchored default `scores_dir` and resolve this real file — so they will NOT raise `FileNotFoundError`. The resolved key is C major, exercise key is C major, so `transpose_semitones=0` and `target_key="C major"`. Pre-existing tests do not assert on `transpose_semitones` or `target_key` (those fields do not yet exist), so their assertions are unaffected.

Note: `piece_id="fur_elise"` would resolve to `model/data/scores/fur_elise.json` which does NOT exist, causing `FileNotFoundError` in every pre-existing test after Task 5's changes. Using `bach.prelude.bwv_846` is the fix.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai-wt-issue41/model && uv run pytest tests/exercise_corpus/test_tags.py tests/exercise_corpus/test_briefing.py -q
```

Expected: PASS (all test_tags.py and test_briefing.py tests green)

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai-wt-issue41 && git add model/src/exercise_corpus/tags.py model/src/exercise_corpus/technique_tags.toml model/tests/exercise_corpus/test_tags.py model/tests/exercise_corpus/test_briefing.py && git commit -m "feat(#41): add TagSet.key field; enforce key in load_tags; update toml + tests"
```

---

### Task 4: E2E fixture JSON for `build_briefing` transpose test

**Group:** C (depends on Group B)

**Behavior being verified:** A committed fixture score JSON with `key_signature: "Eb"` exists at `model/tests/exercise_corpus/fixtures/scores/test_piece_eb.json`; `load_passage_key` can resolve it when `scores_dir` points at the fixtures directory.

**Interface under test:**
```python
from exercise_corpus.keys import load_passage_key
```

**Files:**
- Create: `model/tests/exercise_corpus/fixtures/scores/test_piece_eb.json`
- Modify: `model/tests/exercise_corpus/test_keys.py`

---

- [ ] **Step 1: Write the failing test**

Append to `model/tests/exercise_corpus/test_keys.py`:

```python
def test_load_passage_key_from_test_fixture(tmp_path: Path):
    # Uses a committed fixture at tests/exercise_corpus/fixtures/scores/
    fixtures_scores = Path(__file__).resolve().parent / "fixtures" / "scores"
    result = load_passage_key("test_piece_eb", scores_dir=fixtures_scores)
    assert result == "Eb"
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai-wt-issue41/model && uv run pytest tests/exercise_corpus/test_keys.py::test_load_passage_key_from_test_fixture -q
```

Expected: FAIL — `FileNotFoundError: score JSON not found for piece_id 'test_piece_eb'`

- [ ] **Step 3: Implement — create the fixture JSON**

Create `model/tests/exercise_corpus/fixtures/scores/test_piece_eb.json`:

```json
{
  "piece_id": "test_piece_eb",
  "key_signature": "Eb"
}
```

No other fields are required — `load_passage_key` only reads `key_signature` from the top-level object.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai-wt-issue41/model && uv run pytest tests/exercise_corpus/test_keys.py -q
```

Expected: PASS (all test_keys.py tests green)

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai-wt-issue41 && git add model/tests/exercise_corpus/fixtures/scores/test_piece_eb.json model/tests/exercise_corpus/test_keys.py && git commit -m "test(#41): add test_piece_eb.json fixture for build_briefing transpose E2E"
```

---

### Task 5: `ExerciseBriefing` transpose fields + excerpt-from-bar_range in `build_briefing`

**Group:** D (depends on Group C)

**Behavior being verified:** `build_briefing` with a fixture `scores_dir` pointing at `test_piece_eb.json` emits `transpose_semitones=3` and `target_key="Eb"` (C-major exercise transposed +3 to Eb). With `bar_range=(5, 12)`, the excerpt transform_params has `end_bar=8` (length = 12-5+1). With `bar_range=(5, 8)`, `end_bar=4`. With unknown `scores_dir` / no key, `transpose_semitones=None`. Untagged dimension still raises `NoPrimitiveForDimensionError` before any key resolution.

**Interface under test:**
```python
from exercise_corpus.briefing import build_briefing, ExerciseBriefing, Diagnosis
```

**Files:**
- Modify: `model/src/exercise_corpus/briefing.py`
- Modify: `model/tests/exercise_corpus/test_briefing.py`

---

- [ ] **Step 1: Write the failing tests**

Append to `model/tests/exercise_corpus/test_briefing.py`:

```python
FIXTURES_SCORES = Path(__file__).resolve().parent / "fixtures" / "scores"


def test_build_briefing_transpose_semitones_eb(tmp_path: Path):
    """C-major drill transposed +3 when passage is in Eb."""
    db = _make_catalog(tmp_path, ["hanon_001", "hanon_002"])
    tags = _tags({pid: ["timing", "articulation"] for pid in ["hanon_001", "hanon_002"]})
    diag = Diagnosis(
        dimension="timing",
        severity="moderate",
        bar_range=(5, 8),
        piece_id="test_piece_eb",
    )
    briefing = build_briefing(
        diag, tags, history=[], now=0, db_path=db, scores_dir=FIXTURES_SCORES
    )
    assert briefing.transpose_semitones == 3
    assert briefing.target_key == "Eb"


def test_build_briefing_transpose_none_when_key_absent(tmp_path: Path):
    """transpose_semitones and target_key are None when piece JSON has null key_signature."""
    db = _make_catalog(tmp_path, ["hanon_001"])
    tags = _tags({"hanon_001": ["timing", "articulation"]})
    # Write a fixture with null key_signature into tmp_path
    null_key_json = tmp_path / "no_key_piece.json"
    null_key_json.write_text('{"piece_id": "no_key_piece", "key_signature": null}')
    diag = Diagnosis(
        dimension="timing",
        severity="moderate",
        bar_range=(3, 6),
        piece_id="no_key_piece",
    )
    briefing = build_briefing(
        diag, tags, history=[], now=0, db_path=db, scores_dir=tmp_path
    )
    assert briefing.transpose_semitones is None
    assert briefing.target_key is None


def test_build_briefing_excerpt_end_bar_from_bar_range_8_bars(tmp_path: Path):
    """excerpt transform_params["end_bar"] equals bar_range length (8 bars)."""
    db = _make_catalog(tmp_path, ["hanon_001"])
    tags = _tags({"hanon_001": ["timing", "articulation"]})
    null_key_json = tmp_path / "no_key_piece.json"
    null_key_json.write_text('{"piece_id": "no_key_piece", "key_signature": null}')
    diag = Diagnosis(
        dimension="timing",
        severity="moderate",
        bar_range=(5, 12),
        piece_id="no_key_piece",
    )
    briefing = build_briefing(
        diag, tags, history=[], now=0, db_path=db, scores_dir=tmp_path
    )
    assert briefing.transform == "excerpt"
    assert briefing.transform_params["start_bar"] == 1
    assert briefing.transform_params["end_bar"] == 8  # 12 - 5 + 1


def test_build_briefing_excerpt_end_bar_from_bar_range_4_bars(tmp_path: Path):
    """excerpt transform_params["end_bar"] equals bar_range length (4 bars)."""
    db = _make_catalog(tmp_path, ["hanon_001"])
    tags = _tags({"hanon_001": ["timing", "articulation"]})
    null_key_json = tmp_path / "no_key_piece.json"
    null_key_json.write_text('{"piece_id": "no_key_piece", "key_signature": null}')
    diag = Diagnosis(
        dimension="timing",
        severity="moderate",
        bar_range=(5, 8),
        piece_id="no_key_piece",
    )
    briefing = build_briefing(
        diag, tags, history=[], now=0, db_path=db, scores_dir=tmp_path
    )
    assert briefing.transform == "excerpt"
    assert briefing.transform_params["end_bar"] == 4  # 8 - 5 + 1


def test_build_briefing_target_key_in_instruction(tmp_path: Path):
    """When target_key is set, instruction text contains the key name."""
    db = _make_catalog(tmp_path, ["hanon_001"])
    tags = _tags({"hanon_001": ["timing", "articulation"]})
    diag = Diagnosis(
        dimension="timing",
        severity="moderate",
        bar_range=(5, 8),
        piece_id="test_piece_eb",
    )
    briefing = build_briefing(
        diag, tags, history=[], now=0, db_path=db, scores_dir=FIXTURES_SCORES
    )
    assert "Eb" in briefing.instruction


def test_untagged_dimension_raises_before_key_resolution(tmp_path: Path):
    """NoPrimitiveForDimensionError is raised before any key resolution attempt."""
    db = _make_catalog(tmp_path, ["hanon_001"])
    tags = _tags({"hanon_001": ["timing", "articulation"]})
    diag = Diagnosis(
        dimension="dynamics",
        severity="moderate",
        bar_range=(5, 8),
        piece_id="test_piece_eb",
    )
    with pytest.raises(NoPrimitiveForDimensionError):
        build_briefing(
            diag, tags, history=[], now=0, db_path=db, scores_dir=FIXTURES_SCORES
        )
```

Also add `NoPrimitiveForDimensionError` to existing import in test_briefing.py:
```python
from exercise_corpus.match import NoPrimitiveForDimensionError, load_index
```
(This import already exists — no change needed.)

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai-wt-issue41/model && uv run pytest tests/exercise_corpus/test_briefing.py -q -k "transpose or excerpt_end_bar or target_key or key_resolution"
```

Expected: FAIL — `TypeError: build_briefing() got an unexpected keyword argument 'scores_dir'` or `AttributeError: 'ExerciseBriefing' object has no attribute 'transpose_semitones'`

- [ ] **Step 3: Implement**

**3a. Update `ExerciseBriefing` dataclass** in `model/src/exercise_corpus/briefing.py` — add two new fields after `transform_params`:

```python
@dataclass
class ExerciseBriefing:
    target_dimension: str
    severity: str
    exercise_type: str
    matched_primitive_id: str
    matched_source: str
    matched_title: str
    match_score: float
    transform: str | None  # "tempo" | "excerpt" | None
    transform_params: dict | None
    transpose_semitones: int | None  # key shift; None if passage key unknown
    target_key: str | None           # resolved passage key string e.g. "Eb"
    bar_range: tuple[int, int]
    estimated_minutes: int
    instruction: str
    success_criterion: str
    action_binding: str | None
    candidates: list[Match] = field(default_factory=list)
```

**3b. Update `_transform_params`** to derive excerpt span from bar_range:

```python
def _transform_params(
    transform: str | None, severity: str, bar_range: tuple[int, int]
) -> dict | None:
    if transform == "tempo":
        return {"factor": 0.5 if severity == "significant" else 0.66}
    if transform == "excerpt":
        length = bar_range[1] - bar_range[0] + 1
        return {"start_bar": 1, "end_bar": length}
    return None
```

**3c. Add import at top of `briefing.py`:**

```python
from exercise_corpus.keys import load_passage_key, parse_key_to_pc, transpose_interval
```

**3d. Update `build_briefing` signature and body:**

Add `scores_dir: Path | None = None` parameter:

```python
def build_briefing(
    diagnosis: Diagnosis,
    tags: dict[str, TagSet],
    history: list[PrescriptionRecord],
    now: float,
    db_path=None,
    index: CatalogIndex | None = None,
    top_k: int = 5,
    scores_dir: Path | None = None,
) -> ExerciseBriefing:
```

After `top = matches[0]`, add key resolution:

```python
    top = matches[0]

    # Key resolution: transpose C-major exercise into the student's passage key.
    exercise_pc = parse_key_to_pc(tags[top.primitive_id].key)
    passage_key_str = load_passage_key(diagnosis.piece_id, scores_dir)
    if passage_key_str is None:
        transpose_semitones = None
        target_key = None
    else:
        passage_pc = parse_key_to_pc(passage_key_str)
        transpose_semitones = transpose_interval(exercise_pc, passage_pc)
        target_key = passage_key_str
```

Update the `_transform_params` call (now takes `bar_range`):

```python
    exercise_type, transform, action_binding = _DIMENSION_PLAN[diagnosis.dimension]
    start, end = diagnosis.bar_range
    t_params = _transform_params(transform, diagnosis.severity, diagnosis.bar_range)
```

Build the instruction, adding key text when target_key is set:

```python
    base_instruction = _INSTRUCTION[exercise_type].format(
        start=start, end=end, title=top.title
    )
    if target_key is not None:
        instruction = base_instruction + f" Transpose into the key of {target_key}."
    else:
        instruction = base_instruction
```

Update the `ExerciseBriefing(...)` constructor call to pass the new fields and use `t_params` / `instruction`:

```python
    return ExerciseBriefing(
        target_dimension=diagnosis.dimension,
        severity=diagnosis.severity,
        exercise_type=exercise_type,
        matched_primitive_id=top.primitive_id,
        matched_source=top.source,
        matched_title=top.title,
        match_score=top.score,
        transform=transform,
        transform_params=t_params,
        transpose_semitones=transpose_semitones,
        target_key=target_key,
        bar_range=diagnosis.bar_range,
        estimated_minutes=_MINUTES[diagnosis.severity],
        instruction=instruction,
        success_criterion=_SUCCESS[exercise_type],
        action_binding=action_binding,
        candidates=matches,
    )
```

Also remove the now-unused `_EXCERPT_BARS` constant and the `Path` import if not already present (add `from pathlib import Path` if needed).

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai-wt-issue41/model && uv run pytest tests/exercise_corpus/test_briefing.py -q
```

Expected: PASS (all test_briefing.py tests green, including updated and new ones)

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai-wt-issue41 && git add model/src/exercise_corpus/briefing.py model/tests/exercise_corpus/test_briefing.py && git commit -m "feat(#41): add transpose_semitones/target_key to ExerciseBriefing; excerpt from bar_range"
```

---

### Task 6: Full suite green — verify no regressions

**Group:** D (depends on Group C; run after Task 5)

**Behavior being verified:** All exercise corpus tests pass except the 8 pre-existing `test_transforms.py` failures. No new failures introduced by the changes in this plan.

**Interface under test:** Full test suite.

**Files:**
- No new files — verification only.

---

- [ ] **Step 1: Run the full suite**

```bash
cd /Users/jdhiman/Documents/crescendai-wt-issue41/model && uv run pytest tests/exercise_corpus/ -q 2>&1 | tail -15
```

Expected output pattern:
```
X passed, 8 failed, 1 skipped
```
where X >= 24 (all pre-plan tests plus new tests from this plan) and the 8 failures are exactly in `test_transforms.py` (gitignored MIDI data absent).

- [ ] **Step 2: Confirm transform failures are pre-existing**

```bash
cd /Users/jdhiman/Documents/crescendai-wt-issue41/model && uv run pytest tests/exercise_corpus/test_transforms.py -q 2>&1 | tail -5
```

Expected: 8 failures, all `FileNotFoundError` for missing MIDI files. If any failure message is different (e.g., `AttributeError`, `TypeError`), that is a regression introduced by this plan and must be fixed before shipping.

- [ ] **Step 3: Commit verification note to issue**

```bash
gh issue comment 41 --body "STATE: ADAPT layer impl complete; all exercise_corpus tests pass except 8 pre-existing test_transforms.py failures (gitignored MIDI). Next: /challenge then /build."
```

---

## Success Criteria

```bash
cd /Users/jdhiman/Documents/crescendai-wt-issue41/model && uv run pytest tests/exercise_corpus/test_keys.py tests/exercise_corpus/test_briefing.py tests/exercise_corpus/test_tags.py -q
```

All green — including all pre-existing `test_briefing.py` tests (which now use `piece_id="bach.prelude.bwv_846"` via the updated `_diagnosis()` helper). `test_transforms.py` 8 pre-existing failures (gitignored MIDI data) are excluded from this criteria and must not increase.

---

## Challenge Review

### CEO Pass

#### Premise Challenge

Right problem. `build_briefing` genuinely emits useless C-major drills for students working in other keys, and the 8-bar hardcoded excerpt is pedagogically absurd for a student who struggled over 2 bars. Without this change, the exercise prescription layer is a facade — it shows the right drill type but sizes and keys it wrong 100% of the time for non-C students.

Direct path: yes. The three-helper decomposition in `keys.py` is the minimum to make key resolution testable in isolation. No proxy problem.

No simpler alternative exists. The user explicitly settled Option A (separate `transpose_semitones` field) in brainstorm — not re-litigating this.

Existing coverage: `briefing.py` already has `_transform_params`, `_EXCERPT_BARS`, `_INSTRUCTION`. The plan extends these correctly rather than inventing new patterns.

#### Scope Check

The 8 files in the plan's File Structure are the minimum. No new services or classes beyond the new `keys.py` module. Nothing can be cut without losing core behavior. Scope matches spec exactly.

#### Twelve-Month Alignment

```
CURRENT STATE                     THIS PLAN                         12-MONTH IDEAL
build_briefing emits C-major  →   passage-key transpose +        →  full ADAPT layer: key, excerpt,
drills with hardcoded 4-bar        excerpt-from-bar_range;             difficulty levels, technique
excerpt regardless of student      adds transpose_semitones/           discrimination (issues #42/#43),
key or bar range                   target_key to ExerciseBriefing      API wiring (#29)
```

This plan moves directly toward the ideal. The `transpose_semitones` field design is explicitly forward-compatible with the downstream API wiring (#29) — it is already the right contract.

#### Alternatives Check

Spec documents Option A vs option of encoding transpose inside `transform_params`. Trade-offs are present (Option A was chosen for orthogonality). Documented. No action needed.

---

### Engineering Pass

#### Architecture

Data flow after Task 5:
```
build_briefing(diagnosis, tags, ..., scores_dir)
    │
    ├── severity/dimension guard
    ├── cooldown check
    ├── match_by_dimension() -> top match
    ├── tags[top.primitive_id].key -> exercise_pc via parse_key_to_pc()
    ├── load_passage_key(diagnosis.piece_id, scores_dir)
    │       ├── None (key_signature: null in JSON) -> transpose=None, target_key=None
    │       ├── str -> parse_key_to_pc() -> transpose_interval() -> semitones
    │       └── FileNotFoundError (JSON absent) -> PROPAGATES UNCAUGHT
    ├── _transform_params(transform, severity, bar_range)
    └── ExerciseBriefing(...)
```

The `FileNotFoundError` propagation is the critical architectural issue — see BLOCKER below.

Component boundaries are clean. `keys.py` is purely functional except for `load_passage_key`'s I/O, which is correctly isolated.

No security issues (no user input flows to SQL or shell; all I/O is read-only from committed JSON files).

No N+1 or fan-out issues.

#### Module Depth Audit

- **`keys.py` (NEW):** Interface = 3 functions. Hides enharmonic table, path anchoring, JSON parsing, nearest-octave arithmetic. **DEEP.**
- **`tags.py` (MODIFY):** Adding one required field to `TagSet` dataclass. Interface remains 2 exports. **DEEP.**
- **`briefing.py` (MODIFY):** Adding 2 fields to dataclass and one optional param to `build_briefing`. Interface stays narrow. **DEEP.**

#### Code Quality

Minor comment error: `keys.py` line `# so parents[2] is model/src/exercise_corpus/../../../ -> model/` is wrong — `parents[2]` from `model/src/exercise_corpus/keys.py` is `model/src/`, not `model/`. The code is correct (`parents[2] / "data" / "scores"` resolves to `model/src/data/scores`, NOT `model/data/scores`).

Wait — let me recalculate: `Path("model/src/exercise_corpus/keys.py").resolve().parents[0]` = `model/src/exercise_corpus`, `parents[1]` = `model/src`, `parents[2]` = `model`. The code IS correct because Python's `Path.parents` is 0-indexed from the leaf. The comment just misstates the intermediate path, not the final result. This is a documentation nit, not a code bug.

DRY: no violations introduced.

Error handling: `parse_key_to_pc` raises `ValueError` on unknown input — explicit, not a catch-all. `load_passage_key` raises `FileNotFoundError` — explicit. Consistent with project standard of explicit exceptions over fallbacks.

#### Test Philosophy Audit

All new tests verify behavior through the public interface — no internal state assertions, no mocking of internal collaborators. External boundaries (filesystem via `tmp_path`) are appropriately used as seams.

`test_range_is_bounded` in Task 1 is a property test over all 144 (from_pc, to_pc) pairs — this is strong behavioral coverage.

`test_target_key_in_instruction` verifies text content of the briefing instruction — this is behavior (what the student sees), not shape.

No forbidden patterns detected.

#### Vertical Slice Audit

Each task is one test → one implementation → one commit. Tasks 1–5 are properly sequenced. Task 6 is verification only (no commit needed for a green suite run — the plan correctly has no commit step for Task 6, just a GitHub issue comment).

[OBS] — Task 2's Step 3 says "The implementation of `load_passage_key` itself is already in `keys.py` from Task 1. No production code changes needed in this task." This is correct — but the stated "failing" reason in Step 2 ("FAIL — `ImportError` or `NameError: load_passage_key`") is slightly wrong: if Task 1 is done, the function exists and is importable; the test fails because the import in the test file hasn't been updated yet AND the fixture path may not exist. This is a documentation inconsistency, not a blocker.

#### Test Coverage Gaps

```
[+] model/src/exercise_corpus/keys.py
    ├── parse_key_to_pc()
    │   ├── [TESTED ★★★]  happy path: C, Am, Eb, C#, Db, Gb, F#, Bb, G, C#m — Task 1
    │   ├── [TESTED ★★★]  enharmonic equivalence (C#==Db, Gb==F#) — Task 1
    │   └── [TESTED ★★★]  ValueError on unknown/empty/garbage — Task 1
    ├── transpose_interval()
    │   ├── [TESTED ★★★]  same key (0), up (+3, +5, +6 tritone), down (-3, -5) — Task 1
    │   └── [TESTED ★★★]  all 144 pairs in [-5,+6] — Task 1
    └── load_passage_key()
        ├── [TESTED ★★★]  happy path from committed fixture — Task 2
        ├── [TESTED ★★]   FileNotFoundError for nonexistent piece — Task 2
        └── [TESTED ★★]   null key_signature -> None — Task 2

[+] model/src/exercise_corpus/briefing.py (new behavior)
    ├── build_briefing() with scores_dir
    │   ├── [TESTED ★★]   transpose_semitones=3, target_key="Eb" — Task 5
    │   ├── [TESTED ★★]   transpose_semitones=None when key_signature=null — Task 5
    │   ├── [TESTED ★★]   excerpt end_bar from bar_range (8 bars and 4 bars) — Task 5
    │   ├── [TESTED ★★]   target_key in instruction text — Task 5
    │   ├── [TESTED ★★]   NoPrimitiveForDimensionError before key resolution — Task 5
    │   └── [GAP]         FileNotFoundError when piece_id has no score JSON — not directly
    │                     tested through build_briefing (load_passage_key tests cover it
    │                     directly in test_keys.py); acceptable gap — caller error path.
    └── _transform_params() signature change (now takes bar_range)
        └── [TESTED ★★]   indirectly via excerpt tests

[+] model/src/exercise_corpus/tags.py (TagSet.key field)
    ├── [TESTED ★★★]  load_tags with key field present — Task 3
    └── [TESTED ★★★]  ValueError on missing key field — Task 3
```

#### Failure Modes

Task 3 (TagSet.key field addition) has a failure mode where `technique_tags.toml` is updated but `tags.py` is not, or vice versa. The vertical-slice structure sequences these correctly in the same commit — not a runtime issue.

All file writes are to committed test fixtures — no partial-state risk.

---

### Presumption Inventory

| Assumption | Verdict | Reason |
|---|---|---|
| `bach.prelude.bwv_846.json` is tracked in git and has `key_signature: "C major"` | SAFE | Verified: `git ls-files` confirms it is tracked; `json.load()` confirms `key_signature == "C major"`. |
| `model/data/scores/` contains only committed, non-gitignored files | SAFE | `git check-ignore` confirms the directory is not gitignored; the bach file is committed. |
| All 22 `technique_tags.toml` entries are C major and need `key = "C"` | SAFE | Read the TOML — Hanon 1-20, Czerny op.299 no.1, Burgmuller op.100 no.1 are all C major fingering studies. |
| `parents[2]` from `model/src/exercise_corpus/keys.py` is `model/` | SAFE | Verified: `parents[0]` = `model/src/exercise_corpus`, `parents[1]` = `model/src`, `parents[2]` = `model`. Code is correct; comment is imprecise but harmless. |
| Existing `test_briefing.py` tests will still pass after Task 5 | SAFE (fixed in loop 1) | Task 3 step 3c updates `_diagnosis()` default to `piece_id="bach.prelude.bwv_846"`, which resolves to a git-committed score JSON with `key_signature: "C major"`. Pre-existing tests get `transpose_semitones=0` and do not assert on the new fields, so their assertions remain valid. |
| `test_load_passage_key_returns_key_from_committed_fixture` will pass in CI | VALIDATE | The test uses `parents[3]` to reach `model/data/scores/`; that file is committed. Should pass anywhere the repo is cloned, but `parents[3]` is fragile if the test file is ever moved. |
| `fixtures/scores/` directory does not already exist | SAFE | Confirmed: `model/tests/exercise_corpus/fixtures/` contains only `.xml` files, no `scores/` subdirectory. The Task 4 step that creates `fixtures/scores/test_piece_eb.json` will implicitly create the subdirectory. |

---

### Blockers and Risks

**[BLOCKER — RESOLVED]** — The `build_briefing` implementation in Task 5 calls `load_passage_key(diagnosis.piece_id, scores_dir)` unconditionally, where `load_passage_key` raises `FileNotFoundError` when the score JSON is absent. All pre-existing `test_briefing.py` tests used `_diagnosis()` which hardcoded `piece_id="fur_elise"` — a file that does not exist — which would raise `FileNotFoundError` and break every pre-existing test after Task 5's changes.

**Resolution (loop 1):** The plan preserves `FileNotFoundError` propagation (no catch-and-degrade; explicit exceptions per CLAUDE.md). The spec now states two distinct conditions: missing JSON file = `FileNotFoundError` (real error, propagates); `key_signature: null` inside an existing JSON = `None` return (legitimate musical absence, no error). The pre-existing test fix is: Task 3 step 3c updates `_diagnosis()`'s default `piece_id` from `"fur_elise"` to `"bach.prelude.bwv_846"` — `model/data/scores/bach.prelude.bwv_846.json` is git-committed with `key_signature: "C major"`. Pre-existing tests then resolve a real C-major piece, get `transpose_semitones=0`, and their existing assertions (which don't touch the new fields) remain valid. Tests that specifically exercise the null-key path use `tmp_path` with an explicit null-key fixture JSON and pass `scores_dir=tmp_path`.

---

**[RISK] (confidence: 7/10)** — Task 5's `test_untagged_dimension_raises_before_key_resolution` asserts that `NoPrimitiveForDimensionError` is raised before any key resolution. The current `build_briefing` implementation calls `match_by_dimension` before the key resolution block — so the error would already propagate before `load_passage_key` is called. The test verifies the right behavior, but the name "before_key_resolution" is only guaranteed if the `match_by_dimension` call remains above the key resolution block in the implementation. If a refactor reorders those steps, the test would still pass (the error is still raised), but the ordering guarantee would be silently lost. Low severity since the test behavior is correct; worth a comment in the implementation. No change needed.

**[RISK] (confidence: 6/10)** — Task 2's `test_load_passage_key_returns_key_from_committed_fixture` uses `parents[3]` to reach the `model/data/scores/` path from `model/tests/exercise_corpus/test_keys.py`. This is correct today but fragile: if the test file moves to a different nesting depth, the path breaks silently with `FileNotFoundError`. Consider replacing with a `tmp_path`-based fixture that writes the same data, or anchoring via `exercise_corpus.__file__` (which is stable). Non-blocking since the file is committed and the nesting is unlikely to change.

**[OBS]** — Task 6 Step 3 posts a GitHub issue comment with a stale message: "Next: /challenge then /build" — but this plan is already past /challenge when Task 6 runs. The comment body should read "Next: /ship" or "Next: /review" depending on workflow.

**[OBS]** — The `_EXCERPT_BARS` constant removal mentioned at the end of Task 5 Step 3 is the right call (it becomes dead code after the change), consistent with the project's "remove orphans your changes create" coding standard. The plan correctly flags it.

---

### Summary

[BLOCKER] count: 1 (resolved in loop 1 — see BLOCKER section above)
[RISK]    count: 2 (non-blocking, no plan changes required)
[QUESTION] count: 0

VERDICT: PROCEED — Blocker resolved. `_diagnosis()` default updated to a real committed score JSON; spec error table clarified with explicit two-condition distinction (missing file vs null key); success criteria updated. Plan is ready for /build.
