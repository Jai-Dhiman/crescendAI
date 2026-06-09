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

**3c. Update `model/tests/exercise_corpus/test_briefing.py`** — the `_tags` helper constructs `TagSet` directly:

```python
def _tags(mapping: dict[str, list[str]]) -> dict[str, TagSet]:
    return {pid: TagSet(frozenset(dims), frozenset()) for pid, dims in mapping.items()}
```

Change to:

```python
def _tags(mapping: dict[str, list[str]]) -> dict[str, TagSet]:
    return {pid: TagSet(frozenset(dims), frozenset(), key="C") for pid, dims in mapping.items()}
```

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

All green. `test_transforms.py` failure count does not exceed 8.
