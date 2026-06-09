# Dimension-Tag Retrieval — Implementation Plan

> **For the build agent:** Dispatch each task group in dependency order. Within a group, tasks touch the same file and are SEQUENTIAL. Do NOT start execution until /challenge returns VERDICT: PROCEED.
> **Branch:** `issue-36-exercise-matcher-transforms-briefing` (worktree at `/Users/jdhiman/Documents/crescendai-wt-issue36`). All paths below are relative to the repo root; run pytest from `model/`.

**Goal:** Make the exercise matcher retrieve by diagnosed dimension via curated technique tags; untagged dimensions raise `NoPrimitiveForDimensionError`.
**Spec:** docs/specs/2026-06-08-dimension-tag-retrieval-design.md
**Style:** model/ only. uv. Explicit exceptions over fallbacks. No emojis. Follow CLAUDE.md.

## Task Groups
- **Group A (sequential):** Task 1 → Task 2 → Task 3 — all touch `tags.py` + `test_tags.py`.
- **Group B (sequential, depends on A):** Task 4 → Task 5 — all touch `match.py` + `test_match.py`.
- **Group C (sequential, depends on A+B):** Task 6 — `technique_tags.toml` + integration test in `test_match.py`.
- **Group D (sequential, depends on B+C):** Task 7 → Task 8 — `briefing.py` + `test_briefing.py`.

Groups A and B are mostly sequential because each module stacks on the prior. No two tasks in different groups may run in parallel here (the briefing rewire imports both `tags` and `match`).

`[SHIPS INDEPENDENTLY]` — Group A (the tag layer) is a standalone, importable, tested module even before retrieval is rewired.

---

### Task 1: `load_tags` reads a valid tag TOML into TagSets
**Group:** A

**Behavior being verified:** A well-formed `technique_tags.toml` loads into a `{primitive_id: TagSet}` map with the declared dimensions and techniques.
**Interface under test:** `exercise_corpus.tags.load_tags`

**Files:**
- Create: `model/src/exercise_corpus/tags.py`
- Test: `model/tests/exercise_corpus/test_tags.py`

- [ ] **Step 1: Write the failing test**

```python
# model/tests/exercise_corpus/test_tags.py
"""Tests for tags.py -- the editorial technique-tag layer.

load_tags reads a version-controlled technique_tags.toml and validates it against
the catalog: every dimension label must be one of the canonical 6, and every
tagged primitive_id must exist in the catalog. Tests use tmp_path TOML fixtures
and an explicit known_primitive_ids set, so no catalog DB or Aria weights are
required.
"""

from pathlib import Path

import pytest

from exercise_corpus.tags import TagSet, load_tags


def _write_toml(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "technique_tags.toml"
    p.write_text(body)
    return p


def test_load_tags_reads_valid_table(tmp_path: Path):
    toml = _write_toml(
        tmp_path,
        """
[hanon_001]
dimensions = ["articulation", "timing"]
techniques = ["finger_independence", "evenness"]

[burgmuller_001]
dimensions = ["phrasing", "interpretation"]
""",
    )
    tags = load_tags(toml, known_primitive_ids={"hanon_001", "burgmuller_001"})

    assert set(tags) == {"hanon_001", "burgmuller_001"}
    assert isinstance(tags["hanon_001"], TagSet)
    assert tags["hanon_001"].dimensions == frozenset({"articulation", "timing"})
    assert tags["hanon_001"].techniques == frozenset({"finger_independence", "evenness"})
    # techniques key may be omitted -> empty frozenset
    assert tags["burgmuller_001"].dimensions == frozenset({"phrasing", "interpretation"})
    assert tags["burgmuller_001"].techniques == frozenset()
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/exercise_corpus/test_tags.py::test_load_tags_reads_valid_table -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'exercise_corpus.tags'` (module does not exist yet).

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# model/src/exercise_corpus/tags.py
"""Editorial technique-tag layer for the exercise corpus.

Each exercise primitive is tagged with the teacher DIMENSIONS it can address
plus free-vocabulary technique labels. Retrieval (match.match_by_dimension)
filters candidates by these dimensions. Tags are hand-authored editorial data,
so they live in a version-controlled technique_tags.toml -- never baked into the
machine-regenerated SQLite catalog, which would silently drop them on rebuild.

load_tags validates the table against the catalog so authoring drift (a typo in
a dimension, or a tag for a primitive that no longer exists) fails loudly.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path

# The canonical 6 teacher dimensions (mirrors
# apps/api/src/harness/artifacts/diagnosis.ts DIMENSIONS).
DIMENSIONS = (
    "dynamics",
    "timing",
    "pedaling",
    "articulation",
    "phrasing",
    "interpretation",
)


@dataclass(frozen=True)
class TagSet:
    """The dimensions an exercise can address plus free technique labels."""

    dimensions: frozenset[str]
    techniques: frozenset[str]


def load_tags(path: Path, known_primitive_ids: set[str]) -> dict[str, TagSet]:
    """Read and validate technique_tags.toml into a {primitive_id: TagSet} map.

    Args:
        path: path to the technique_tags.toml file.
        known_primitive_ids: the primitive_ids present in the catalog; every
            tagged primitive must be one of these.

    Returns:
        Dict mapping primitive_id to its TagSet.

    Raises:
        FileNotFoundError: if path does not exist.
        ValueError: if a tag references a primitive_id absent from the catalog,
            or declares a dimension outside the canonical 6.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"technique tags file not found: {path}")
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    tags: dict[str, TagSet] = {}
    for primitive_id, entry in raw.items():
        if primitive_id not in known_primitive_ids:
            raise ValueError(
                f"tag references unknown primitive_id {primitive_id!r} "
                f"(not in catalog)"
            )
        dims = tuple(entry.get("dimensions", ()))
        for d in dims:
            if d not in DIMENSIONS:
                raise ValueError(
                    f"unknown dimension {d!r} for {primitive_id!r}; "
                    f"valid dimensions are {DIMENSIONS}"
                )
        techs = tuple(entry.get("techniques", ()))
        tags[primitive_id] = TagSet(
            dimensions=frozenset(dims), techniques=frozenset(techs)
        )
    return tags
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/exercise_corpus/test_tags.py::test_load_tags_reads_valid_table -q
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add model/src/exercise_corpus/tags.py model/tests/exercise_corpus/test_tags.py
git commit -m "feat(exercise-corpus): add technique-tag layer (load_tags, #36)"
```

---

### Task 2: `load_tags` rejects an unknown dimension label
**Group:** A (depends on Task 1; same files)

**Behavior being verified:** A tag declaring a dimension outside the canonical 6 fails loudly.
**Interface under test:** `exercise_corpus.tags.load_tags`

**Files:**
- Modify: `model/src/exercise_corpus/tags.py` (no change expected — validation already present; this task proves it via test)
- Test: `model/tests/exercise_corpus/test_tags.py`

- [ ] **Step 1: Write the failing test**

```python
def test_load_tags_rejects_unknown_dimension(tmp_path: Path):
    toml = _write_toml(
        tmp_path,
        """
[hanon_001]
dimensions = ["timing", "tempo"]
""",
    )
    with pytest.raises(ValueError, match="unknown dimension"):
        load_tags(toml, known_primitive_ids={"hanon_001"})
```

- [ ] **Step 2: Run test — verify it FAILS or PASSES**

```bash
cd model && uv run pytest tests/exercise_corpus/test_tags.py::test_load_tags_rejects_unknown_dimension -q
```
Expected: PASS immediately — the dimension-validation branch was implemented in Task 1. This test is a guard that locks the behavior. If it FAILS, the Task 1 implementation is wrong (the `for d in dims` validation is missing or the message differs); fix `load_tags` so `"tempo"` raises `ValueError` matching `"unknown dimension"`.

> Note: this is a behavior already covered by the Task 1 implementation; the test is added as a separate vertical slice to pin it. No new implementation code is expected. If it passes on first run, proceed to commit.

- [ ] **Step 3: Implement (only if Step 2 failed)** — ensure the dimension-validation loop in `load_tags` raises `ValueError(f"unknown dimension {d!r} ...")`. (Already present from Task 1.)

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/exercise_corpus/test_tags.py -q
```
Expected: PASS (both tags tests).

- [ ] **Step 5: Commit**

```bash
git add model/tests/exercise_corpus/test_tags.py model/src/exercise_corpus/tags.py
git commit -m "test(exercise-corpus): lock unknown-dimension rejection in load_tags (#36)"
```

---

### Task 3: `load_tags` rejects a tag for an unknown primitive_id
**Group:** A (depends on Task 2; same files)

**Behavior being verified:** A tag for a primitive_id not present in the catalog fails loudly (tags ↔ catalog lockstep).
**Interface under test:** `exercise_corpus.tags.load_tags`

**Files:**
- Modify: `model/src/exercise_corpus/tags.py` (no change expected — validation already present)
- Test: `model/tests/exercise_corpus/test_tags.py`

- [ ] **Step 1: Write the failing test**

```python
def test_load_tags_rejects_unknown_primitive(tmp_path: Path):
    toml = _write_toml(
        tmp_path,
        """
[ghost_999]
dimensions = ["timing"]
""",
    )
    with pytest.raises(ValueError, match="unknown primitive_id"):
        load_tags(toml, known_primitive_ids={"hanon_001"})
```

- [ ] **Step 2: Run test — verify it PASSES (guard)**

```bash
cd model && uv run pytest tests/exercise_corpus/test_tags.py::test_load_tags_rejects_unknown_primitive -q
```
Expected: PASS — the `if primitive_id not in known_primitive_ids` branch was implemented in Task 1. If it FAILS, add that branch to `load_tags` raising `ValueError` matching `"unknown primitive_id"`.

- [ ] **Step 3: Implement (only if Step 2 failed)** — ensure the lockstep check is present in `load_tags`. (Already present from Task 1.)

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/exercise_corpus/test_tags.py -q
```
Expected: PASS (all three tags tests).

- [ ] **Step 5: Commit**

```bash
git add model/tests/exercise_corpus/test_tags.py model/src/exercise_corpus/tags.py
git commit -m "test(exercise-corpus): lock tags<->catalog lockstep in load_tags (#36)"
```

---

### Task 4: `match_by_dimension` returns dimension-filtered, deterministically ranked matches
**Group:** B (depends on Group A)

**Behavior being verified:** Retrieval restricted to a dimension returns only primitives tagged for it, in deterministic order, bounded by top_k.
**Interface under test:** `exercise_corpus.match.match_by_dimension`

**Files:**
- Modify: `model/src/exercise_corpus/match.py`
- Test: `model/tests/exercise_corpus/test_match.py`

- [ ] **Step 1: Write the failing test**

Add to `model/tests/exercise_corpus/test_match.py` (imports at top of file gain `match_by_dimension`, `NoPrimitiveForDimensionError`, and `TagSet`):

```python
# add to the existing imports near the top of test_match.py:
import math
from exercise_corpus.match import (
    Match,
    NoPrimitiveForDimensionError,
    load_index,
    match_by_dimension,
    match_exercises,
)
from exercise_corpus.tags import TagSet


def _dummy_catalog(tmp_path: Path, primitive_ids: list[str]) -> Path:
    """Synthetic catalog with arbitrary (unused-by-tag-retrieval) embeddings."""
    rng = np.random.default_rng(0)
    vectors = {pid: rng.standard_normal(512) for pid in primitive_ids}
    return _make_catalog(tmp_path, vectors)


def test_match_by_dimension_filters_to_tagged_primitives(tmp_path: Path):
    db = _dummy_catalog(
        tmp_path, ["hanon_001", "hanon_002", "hanon_003", "burgmuller_001"]
    )
    tags = {
        "hanon_001": TagSet(frozenset({"timing", "articulation"}), frozenset()),
        "hanon_002": TagSet(frozenset({"timing", "articulation"}), frozenset()),
        "hanon_003": TagSet(frozenset({"timing", "articulation"}), frozenset()),
        "burgmuller_001": TagSet(frozenset({"phrasing", "interpretation"}), frozenset()),
    }

    results = match_by_dimension("timing", tags, db_path=db, top_k=5)

    ids = [m.primitive_id for m in results]
    assert ids == ["hanon_001", "hanon_002", "hanon_003"]  # burgmuller excluded
    assert "burgmuller_001" not in ids
    assert all(isinstance(m, Match) for m in results)
    # No cosine query in tag mode -> score is the nan sentinel.
    assert all(math.isnan(m.score) for m in results)
    # Deterministic across repeated calls.
    again = [m.primitive_id for m in match_by_dimension("timing", tags, db_path=db, top_k=5)]
    assert again == ids
    # top_k bounds the result count.
    assert len(match_by_dimension("timing", tags, db_path=db, top_k=2)) == 2
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/exercise_corpus/test_match.py::test_match_by_dimension_filters_to_tagged_primitives -q
```
Expected: FAIL — `ImportError: cannot import name 'NoPrimitiveForDimensionError' from 'exercise_corpus.match'` (and `match_by_dimension` undefined).

- [ ] **Step 3: Implement the minimum to make the test pass**

Add to `model/src/exercise_corpus/match.py`. Add `from exercise_corpus.tags import TagSet` to the imports, then append:

```python
class NoPrimitiveForDimensionError(Exception):
    """Raised when no catalog primitive is tagged for the requested dimension."""


def match_by_dimension(
    dimension: str,
    tags: dict[str, TagSet],
    db_path: Path | None = None,
    index: CatalogIndex | None = None,
    top_k: int = 5,
) -> list[Match]:
    """Return catalog exercises tagged for `dimension`, deterministically ranked.

    Unlike match_exercises, this does NOT rank by cosine similarity to a query --
    it filters the catalog to primitives whose technique tags include `dimension`
    and ranks them deterministically by (source_exercise_number, primitive_id).
    Match.score is therefore not a cosine; it is nan (no query was scored).

    Args:
        dimension: the diagnosed weakness dimension (one of tags.DIMENSIONS).
        tags: {primitive_id: TagSet} from tags.load_tags.
        db_path: path to the SQLite catalog. Mutually sufficient with index.
        index: a preloaded CatalogIndex (avoids re-reading SQLite).
        top_k: maximum number of matches to return.

    Returns:
        List of Match (score=nan) ordered by (source_exercise_number,
        primitive_id), length <= top_k.

    Raises:
        ValueError: if neither db_path nor index is given.
        NoPrimitiveForDimensionError: if no primitive is tagged for `dimension`
            (no off-dimension fallback).
    """
    if index is None:
        if db_path is None:
            raise ValueError("match_by_dimension requires db_path or index")
        index = load_index(db_path)

    bucket = [
        r
        for r in index.rows
        if (ts := tags.get(r.primitive_id)) is not None and dimension in ts.dimensions
    ]
    if not bucket:
        raise NoPrimitiveForDimensionError(
            f"no catalog primitive is tagged for dimension {dimension!r}"
        )
    bucket.sort(key=lambda r: (r.source_exercise_number, r.primitive_id))
    return [
        Match(
            primitive_id=r.primitive_id,
            source=r.source,
            title=r.title,
            midi_path=r.midi_path,
            score=float("nan"),
        )
        for r in bucket[:top_k]
    ]
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/exercise_corpus/test_match.py::test_match_by_dimension_filters_to_tagged_primitives -q
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add model/src/exercise_corpus/match.py model/tests/exercise_corpus/test_match.py
git commit -m "feat(exercise-corpus): add match_by_dimension dimension-filtered retrieval (#36)"
```

---

### Task 5: `match_by_dimension` raises on an untagged dimension
**Group:** B (depends on Task 4; same files)

**Behavior being verified:** A dimension no primitive is tagged for raises `NoPrimitiveForDimensionError` — no off-dimension fallback.
**Interface under test:** `exercise_corpus.match.match_by_dimension`

**Files:**
- Test: `model/tests/exercise_corpus/test_match.py`
- Modify: `model/src/exercise_corpus/match.py` (no change expected — raise implemented in Task 4)

- [ ] **Step 1: Write the failing test**

```python
def test_match_by_dimension_raises_for_untagged_dimension(tmp_path: Path):
    db = _dummy_catalog(tmp_path, ["hanon_001", "hanon_002"])
    tags = {
        "hanon_001": TagSet(frozenset({"timing", "articulation"}), frozenset()),
        "hanon_002": TagSet(frozenset({"timing", "articulation"}), frozenset()),
    }
    with pytest.raises(NoPrimitiveForDimensionError, match="pedaling"):
        match_by_dimension("pedaling", tags, db_path=db, top_k=5)
```

- [ ] **Step 2: Run test — verify it PASSES (guard)**

```bash
cd model && uv run pytest tests/exercise_corpus/test_match.py::test_match_by_dimension_raises_for_untagged_dimension -q
```
Expected: PASS — the empty-bucket raise was implemented in Task 4. If it FAILS, ensure `match_by_dimension` raises `NoPrimitiveForDimensionError` (with the dimension in the message) when `bucket` is empty.

- [ ] **Step 3: Implement (only if Step 2 failed)** — ensure the `if not bucket: raise NoPrimitiveForDimensionError(...)` branch is present. (Already from Task 4.)

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/exercise_corpus/test_match.py -q
```
Expected: PASS (all match tests, including the untouched `match_exercises` tests).

- [ ] **Step 5: Commit**

```bash
git add model/tests/exercise_corpus/test_match.py model/src/exercise_corpus/match.py
git commit -m "test(exercise-corpus): lock no-fallback raise in match_by_dimension (#36)"
```

---

### Task 6: ship `technique_tags.toml` and validate its real editorial buckets offline
**Group:** C (depends on Groups A + B)

**Behavior being verified:** The *shipped* `technique_tags.toml`, loaded against the 22 real primitive_ids, yields the intended buckets: `timing` = Hanon ∪ Czerny (excludes Burgmüller); `phrasing` = Burgmüller only; `pedaling` and `dynamics` raise.
**Interface under test:** `load_tags` + `match_by_dimension` over the real shipped tag file.

**Files:**
- Create: `model/src/exercise_corpus/technique_tags.toml`
- Test: `model/tests/exercise_corpus/test_match.py`

- [ ] **Step 1: Write the failing test**

```python
# Resolve the shipped tag file relative to the installed package, not CWD.
import exercise_corpus
_PKG_DIR = Path(exercise_corpus.__file__).resolve().parent
SHIPPED_TAGS = _PKG_DIR / "technique_tags.toml"

# The 22 real primitive_ids in the corpus (Hanon 1-20 + Czerny no.1 + Burgmuller no.1).
_REAL_IDS = [f"hanon_{i:03d}" for i in range(1, 21)] + ["czerny_001", "burgmuller_001"]


def test_shipped_tags_yield_expected_dimension_buckets(tmp_path: Path):
    # Synthetic catalog over the REAL ids -> no real DB / Aria weights needed.
    db = _dummy_catalog(tmp_path, _REAL_IDS)
    tags = load_tags(SHIPPED_TAGS, known_primitive_ids=set(_REAL_IDS))

    timing = {m.primitive_id for m in match_by_dimension("timing", tags, db_path=db, top_k=50)}
    assert "hanon_001" in timing
    assert "czerny_001" in timing
    assert "burgmuller_001" not in timing  # Hanon/Czerny are not phrasing studies

    phrasing = {m.primitive_id for m in match_by_dimension("phrasing", tags, db_path=db, top_k=50)}
    assert phrasing == {"burgmuller_001"}

    # Conservative authoring: nothing in this corpus teaches pedaling or dynamics.
    with pytest.raises(NoPrimitiveForDimensionError):
        match_by_dimension("pedaling", tags, db_path=db, top_k=50)
    with pytest.raises(NoPrimitiveForDimensionError):
        match_by_dimension("dynamics", tags, db_path=db, top_k=50)
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/exercise_corpus/test_match.py::test_shipped_tags_yield_expected_dimension_buckets -q
```
Expected: FAIL — `FileNotFoundError: technique tags file not found: .../exercise_corpus/technique_tags.toml` (file not created yet).

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `model/src/exercise_corpus/technique_tags.toml`:

```toml
# Editorial technique tags per exercise primitive (issue #36).
#
# Retrieval (match.match_by_dimension) filters candidate exercises by `dimensions`.
# An empty bucket for a requested dimension raises NoPrimitiveForDimensionError --
# there is NO off-dimension fallback. Conservative authoring: only high-confidence
# dimension claims are made.
#
# This corpus (Hanon premiere partie 1-20 + Czerny op.299 no.1 + Burgmuller op.100
# no.1) contains no pedaling or dynamics study, so `pedaling` and `dynamics` are
# intentionally untagged and will raise -- that error is the issue #17 corpus-
# breadth gap made legible, not a bug.
#
# `techniques` is free vocabulary used only as human-readable annotation today.

[hanon_001]
dimensions = ["articulation", "timing"]
techniques = ["finger_independence", "evenness"]

[hanon_002]
dimensions = ["articulation", "timing"]
techniques = ["finger_independence", "evenness"]

[hanon_003]
dimensions = ["articulation", "timing"]
techniques = ["finger_independence", "evenness"]

[hanon_004]
dimensions = ["articulation", "timing"]
techniques = ["finger_independence", "evenness"]

[hanon_005]
dimensions = ["articulation", "timing"]
techniques = ["finger_independence", "evenness"]

[hanon_006]
dimensions = ["articulation", "timing"]
techniques = ["finger_independence", "evenness"]

[hanon_007]
dimensions = ["articulation", "timing"]
techniques = ["finger_independence", "evenness"]

[hanon_008]
dimensions = ["articulation", "timing"]
techniques = ["finger_independence", "evenness"]

[hanon_009]
dimensions = ["articulation", "timing"]
techniques = ["finger_independence", "evenness"]

[hanon_010]
dimensions = ["articulation", "timing"]
techniques = ["finger_independence", "evenness"]

[hanon_011]
dimensions = ["articulation", "timing"]
techniques = ["finger_independence", "evenness"]

[hanon_012]
dimensions = ["articulation", "timing"]
techniques = ["finger_independence", "evenness"]

[hanon_013]
dimensions = ["articulation", "timing"]
techniques = ["finger_independence", "evenness"]

[hanon_014]
dimensions = ["articulation", "timing"]
techniques = ["finger_independence", "evenness"]

[hanon_015]
dimensions = ["articulation", "timing"]
techniques = ["finger_independence", "evenness"]

[hanon_016]
dimensions = ["articulation", "timing"]
techniques = ["finger_independence", "evenness"]

[hanon_017]
dimensions = ["articulation", "timing"]
techniques = ["finger_independence", "evenness"]

[hanon_018]
dimensions = ["articulation", "timing"]
techniques = ["finger_independence", "evenness"]

[hanon_019]
dimensions = ["articulation", "timing"]
techniques = ["finger_independence", "evenness"]

[hanon_020]
dimensions = ["articulation", "timing"]
techniques = ["finger_independence", "evenness"]

[czerny_001]
dimensions = ["timing", "articulation"]
techniques = ["velocity", "finger_facility"]

[burgmuller_001]
dimensions = ["phrasing", "interpretation"]
techniques = ["cantabile", "legato"]
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/exercise_corpus/test_match.py::test_shipped_tags_yield_expected_dimension_buckets -q
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add model/src/exercise_corpus/technique_tags.toml model/tests/exercise_corpus/test_match.py
git commit -m "feat(exercise-corpus): ship conservative technique_tags.toml + bucket test (#36)"
```

---

### Task 7: rewire `build_briefing` to dimension-tag retrieval
**Group:** D (depends on Groups B + C)

**Behavior being verified:** `build_briefing` emits a briefing whose matched primitive is dimension-relevant, retrieved via tags rather than a query embedding. (This is a signature change: `query_embedding` is removed, `tags` is added — all existing `test_briefing.py` call sites migrate in the same commit so the suite stays green.)
**Interface under test:** `exercise_corpus.briefing.build_briefing`

**Files:**
- Modify: `model/src/exercise_corpus/briefing.py`
- Modify: `model/tests/exercise_corpus/test_briefing.py`

- [ ] **Step 1: Write the failing test**

Replace the imports/helpers and the existing tests in `test_briefing.py` so they use the new signature. The full migrated file:

```python
"""Tests for briefing.py -- the matcher + transform + memory loop.

build_briefing ties dimension-tag retrieval (slice B, match_by_dimension) and
slice C (transforms) together: given a diagnosed weakness and a tag map, it emits
an ExerciseBriefing that maps onto the api-side ExerciseArtifact contract,
choosing a dimension-appropriate exercise type + deterministic transform, and
respecting a 3-day cooldown so the same weakness is not re-prescribed back-to-back.

Synthetic-catalog tests cover the planning logic without Aria weights; one
end-to-end test runs the whole loop against the real catalog (skipped when the
gitignored catalog DB is absent) and executes the planned transform.
"""

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from partitura.score import Note

import exercise_corpus
from exercise_corpus import Primitive
from exercise_corpus.briefing import (
    CooldownError,
    Diagnosis,
    ExerciseBriefing,
    PrescriptionRecord,
    build_briefing,
    should_prescribe,
)
from exercise_corpus.catalog import write_primitives
from exercise_corpus.match import NoPrimitiveForDimensionError, load_index
from exercise_corpus.tags import TagSet, load_tags
from exercise_corpus.transforms import load_primitive, scale_tempo

DAY = 86_400
REAL_DB = Path("data/exercise_primitives.db")
SHIPPED_TAGS = Path(exercise_corpus.__file__).resolve().parent / "technique_tags.toml"


def _make_catalog(tmp_path: Path, primitive_ids: list[str]) -> Path:
    rng = np.random.default_rng(0)
    primitives, embeddings = [], {}
    for i, pid in enumerate(primitive_ids, start=1):
        source = pid.split("_")[0]
        primitives.append(
            Primitive(
                primitive_id=pid,
                source=source,
                source_exercise_number=i,
                title=f"{source} {i}",
                musicxml_path=tmp_path / f"{pid}.xml",
                midi_path=tmp_path / f"{pid}.mid",
                n_notes=100 + i,
            )
        )
        embeddings[pid] = torch.from_numpy(rng.standard_normal(512).astype(np.float32))
    db = tmp_path / "cat.db"
    write_primitives(primitives, embeddings, db)
    return db


def _tags(mapping: dict[str, list[str]]) -> dict[str, TagSet]:
    return {pid: TagSet(frozenset(dims), frozenset()) for pid, dims in mapping.items()}


def _diagnosis(dimension="timing", severity="moderate", bars=(5, 8)) -> Diagnosis:
    return Diagnosis(
        dimension=dimension,
        severity=severity,
        bar_range=bars,
        piece_id="fur_elise",
    )


def test_build_briefing_emits_briefing_for_a_weakness(tmp_path: Path):
    db = _make_catalog(tmp_path, ["hanon_001", "hanon_002", "hanon_003"])
    tags = _tags({pid: ["timing", "articulation"] for pid in ["hanon_001", "hanon_002", "hanon_003"]})

    briefing = build_briefing(_diagnosis(), tags, history=[], now=0, db_path=db)

    assert isinstance(briefing, ExerciseBriefing)
    # Deterministic ranking -> lowest (source_exercise_number, primitive_id) first.
    assert briefing.matched_primitive_id == "hanon_001"
    assert briefing.matched_primitive_id in {"hanon_001", "hanon_002", "hanon_003"}
    assert briefing.target_dimension == "timing"
    assert briefing.exercise_type == "segment_loop"
    assert briefing.bar_range == (5, 8)
    assert "5" in briefing.instruction and "8" in briefing.instruction
    assert briefing.estimated_minutes == 5  # moderate
    assert len(briefing.candidates) >= 1


def test_phrasing_selects_slow_practice_tempo(tmp_path: Path):
    db = _make_catalog(tmp_path, ["burgmuller_001"])
    tags = _tags({"burgmuller_001": ["phrasing", "interpretation"]})

    phrasing = build_briefing(
        _diagnosis("phrasing", "significant"), tags, history=[], now=0, db_path=db
    )
    assert phrasing.exercise_type == "slow_practice"
    assert phrasing.transform == "tempo"
    assert phrasing.estimated_minutes == 8  # significant


def test_minor_severity_rejected(tmp_path: Path):
    db = _make_catalog(tmp_path, ["hanon_001"])
    tags = _tags({"hanon_001": ["timing", "articulation"]})
    with pytest.raises(ValueError, match="severity"):
        build_briefing(_diagnosis(severity="minor"), tags, history=[], now=0, db_path=db)


def test_cooldown_blocks_recent_repeat(tmp_path: Path):
    db = _make_catalog(tmp_path, ["hanon_001", "hanon_002"])
    tags = _tags({pid: ["timing", "articulation"] for pid in ["hanon_001", "hanon_002"]})
    now = 10 * DAY
    history = [
        PrescriptionRecord(
            primitive_id="hanon_002",
            dimension="timing",
            bar_range=(5, 8),
            prescribed_at=now - 1 * DAY,
        )
    ]
    with pytest.raises(CooldownError):
        build_briefing(_diagnosis(bars=(6, 9)), tags, history=history, now=now, db_path=db)


def test_cooldown_expired_allows_represcribe(tmp_path: Path):
    db = _make_catalog(tmp_path, ["hanon_001", "hanon_002"])
    tags = _tags({pid: ["timing", "articulation"] for pid in ["hanon_001", "hanon_002"]})
    now = 10 * DAY
    history = [
        PrescriptionRecord(
            primitive_id="hanon_002",
            dimension="timing",
            bar_range=(5, 8),
            prescribed_at=now - 5 * DAY,
        )
    ]
    briefing = build_briefing(_diagnosis(bars=(5, 8)), tags, history=history, now=now, db_path=db)
    assert isinstance(briefing, ExerciseBriefing)


def test_should_prescribe_predicate():
    now = 10 * DAY
    rec = PrescriptionRecord("hanon_001", "timing", (5, 8), now - 1 * DAY)
    assert should_prescribe("timing", (6, 9), [rec], now) is False
    assert should_prescribe("dynamics", (6, 9), [rec], now) is True
    assert should_prescribe("timing", (20, 24), [rec], now) is True
    assert should_prescribe("timing", (6, 9), [rec], now + 5 * DAY) is True


def test_end_to_end_briefing_transform_is_realizable():
    """Full loop on the real catalog: tag-match -> plan -> execute the transform."""
    if not REAL_DB.exists():
        pytest.skip("real catalog DB not present")
    idx = load_index(REAL_DB)
    tags = load_tags(SHIPPED_TAGS, known_primitive_ids={r.primitive_id for r in idx.rows})

    briefing = build_briefing(
        _diagnosis("phrasing", "significant"), tags, history=[], now=0, index=idx
    )
    part = load_primitive(
        Path("data/midi/exercise_primitives") / f"{briefing.matched_primitive_id}.mid"
    )
    assert briefing.transform == "tempo"
    variant = scale_tempo(part, briefing.transform_params["factor"])
    assert len(list(variant.part.iter_all(Note))) == len(list(part.iter_all(Note)))
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/exercise_corpus/test_briefing.py -q
```
Expected: FAIL — `TypeError: build_briefing() got an unexpected keyword argument 'tags'` / missing required positional, because `build_briefing` still has the old `query_embedding` signature.

- [ ] **Step 3: Implement the minimum to make the test pass**

Edit `model/src/exercise_corpus/briefing.py`:

(a) Replace the match import:
```python
from exercise_corpus.match import CatalogIndex, Match, match_by_dimension
from exercise_corpus.tags import TagSet
```

(b) Replace the `build_briefing` signature and the matcher call. New signature + body head (everything from `top = matches[0]` onward is unchanged):
```python
def build_briefing(
    diagnosis: Diagnosis,
    tags: dict[str, TagSet],
    history: list[PrescriptionRecord],
    now: float,
    db_path=None,
    index: CatalogIndex | None = None,
    top_k: int = 5,
) -> ExerciseBriefing:
    """Match a diagnosed weakness to an exercise and emit a prescription briefing.

    Retrieval is by diagnosed dimension via curated technique tags
    (match.match_by_dimension), not by a query embedding.

    Raises:
        ValueError: if severity is not in {moderate, significant} or the
            dimension is unknown.
        CooldownError: if the diagnosis was addressed within the cooldown window.
        NoPrimitiveForDimensionError: if no catalog primitive is tagged for the
            diagnosis dimension (e.g. pedaling or dynamics on the current corpus).
    """
    if diagnosis.severity not in VALID_SEVERITIES:
        raise ValueError(
            f"severity must be one of {VALID_SEVERITIES}, got {diagnosis.severity!r}"
        )
    if diagnosis.dimension not in _DIMENSION_PLAN:
        raise ValueError(
            f"unknown dimension {diagnosis.dimension!r}; "
            f"supported: {sorted(_DIMENSION_PLAN)}"
        )
    if not should_prescribe(diagnosis.dimension, diagnosis.bar_range, history, now):
        raise CooldownError(
            f"{diagnosis.dimension} at bars {diagnosis.bar_range} was addressed "
            f"within the last {COOLDOWN_DAYS} days"
        )

    matches = match_by_dimension(
        diagnosis.dimension, tags, db_path=db_path, index=index, top_k=top_k
    )
    top = matches[0]
    # ... (unchanged: exercise_type/transform/action_binding lookup and the
    #      ExerciseBriefing(...) construction stay exactly as they are today)
```

Remove the now-dead `if not matches: raise ValueError("matcher returned no candidates")` guard (an empty bucket now raises inside `match_by_dimension`). Leave the rest of the function body (the `ExerciseBriefing(...)` return) untouched.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/exercise_corpus/test_briefing.py -q
```
Expected: PASS (the real-catalog E2E test SKIPS — `REAL_DB` absent — which is expected).

- [ ] **Step 5: Commit**

```bash
git add model/src/exercise_corpus/briefing.py model/tests/exercise_corpus/test_briefing.py
git commit -m "feat(exercise-corpus): retrieve briefings by dimension tag, not query embedding (#36)"
```

---

### Task 8: `build_briefing` propagates the no-primitive error for an untagged dimension
**Group:** D (depends on Task 7; same files)

**Behavior being verified:** A diagnosis for a dimension the corpus cannot serve (e.g. `dynamics`) raises `NoPrimitiveForDimensionError` from `build_briefing` — passing severity/dimension/cooldown but failing honestly at retrieval.
**Interface under test:** `exercise_corpus.briefing.build_briefing`

**Files:**
- Test: `model/tests/exercise_corpus/test_briefing.py`
- Modify: `model/src/exercise_corpus/briefing.py` (no change expected — error propagates from match_by_dimension)

- [ ] **Step 1: Write the failing test**

```python
def test_untagged_dimension_raises(tmp_path: Path):
    db = _make_catalog(tmp_path, ["hanon_001", "hanon_002"])
    tags = _tags({pid: ["timing", "articulation"] for pid in ["hanon_001", "hanon_002"]})
    # dynamics is a valid teacher dimension (passes the _DIMENSION_PLAN check) but
    # nothing in this catalog is tagged for it -> honest failure at retrieval.
    with pytest.raises(NoPrimitiveForDimensionError):
        build_briefing(_diagnosis("dynamics"), tags, history=[], now=0, db_path=db)
```

- [ ] **Step 2: Run test — verify it PASSES (guard)**

```bash
cd model && uv run pytest tests/exercise_corpus/test_briefing.py::test_untagged_dimension_raises -q
```
Expected: PASS — the error propagates from `match_by_dimension` (Task 4). If it FAILS with a different error (e.g. `dynamics` short-circuits earlier), confirm `dynamics` is in `_DIMENSION_PLAN` (it is) so the code reaches `match_by_dimension`.

- [ ] **Step 3: Implement (only if Step 2 failed)** — none expected; the behavior is inherited from `match_by_dimension`.

- [ ] **Step 4: Run the full exercise_corpus suite**

```bash
cd model && uv run pytest tests/exercise_corpus/ -q
```
Expected: PASS (all tests; the real-catalog E2E test skips).

- [ ] **Step 5: Commit**

```bash
git add model/tests/exercise_corpus/test_briefing.py model/src/exercise_corpus/briefing.py
git commit -m "test(exercise-corpus): build_briefing raises on untagged dimension (#36)"
```

---

## Coverage check (plan ↔ spec)
- `tags.py` / `load_tags` happy + 2 rejection paths → Tasks 1, 2, 3.
- `match_by_dimension` filter/rank/top_k/nan → Task 4; no-fallback raise → Task 5.
- shipped `technique_tags.toml` + real-bucket validation → Task 6.
- `build_briefing` rewire (signature, dimension retrieval, downstream unchanged) → Task 7; untagged-dimension propagation → Task 8.
- `match_exercises` kept intact (deferred tiebreaker) → no task touches it; its existing tests must stay green (verified in Tasks 5 and 8 full-suite runs).

## Notes for the build agent
- Tasks 2, 3, 5, 8 are **guard tests** for behavior implemented in an earlier task in the same module. They are expected to PASS on first run. This is deliberate vertical-slice pinning of distinct behaviors, not horizontal test-batching — each still gets its own commit. If a guard test FAILS, the earlier task's implementation is wrong; fix it before committing.
- The real-catalog E2E test (`test_end_to_end_briefing_transform_is_realizable`) SKIPS in this worktree because `data/exercise_primitives.db` is gitignored/absent. The offline `test_shipped_tags_yield_expected_dimension_buckets` (Task 6) is the binding verification of the shipped tags.
- Do not modify `match_exercises`, `transforms.py`, `catalog.py`, or any file outside `model/src/exercise_corpus/` and its tests.
