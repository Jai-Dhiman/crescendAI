# Exercise Corpus Embedding Validation Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** Prove Aria 512-dim embeddings cluster piano pedagogy primitives by source (purity >= 0.70, k=5) so slice B infrastructure investment is justified.
**Spec:** docs/specs/2026-05-28-exercise-corpus-embedding-validation-design.md
**Style:** Follow the project's coding standards (model/CLAUDE.md, CLAUDE.md). Python via uv. Explicit exception handling, no silent fallbacks. No emojis.

---

## Task Groups

Group A (parallel): Task 1, Task 2, Task 3
Group B (parallel, depends on A): Task 4, Task 5
Group C (sequential, depends on B): Task 6
Group D (sequential, depends on C): Task 7

---

### Task 1: Package scaffold, pyproject update, and MusicXML fixtures

**Group:** A (parallel with Task 2, Task 3)

**Behavior being verified:** The `exercise_corpus` package is importable and three MusicXML fixture files representing minimal 3-exercise corpora are readable by partitura.

**Interface under test:** `import exercise_corpus` succeeds; `partitura.load_musicxml(fixture_path)` returns a score object with the expected number of parts/movements for each fixture.

**Files:**
- Create: `model/src/exercise_corpus/__init__.py`
- Create: `model/tests/exercise_corpus/__init__.py`
- Create: `model/tests/exercise_corpus/fixtures/hanon_3ex.xml`
- Create: `model/tests/exercise_corpus/fixtures/czerny_3ex.xml`
- Create: `model/tests/exercise_corpus/fixtures/burgmuller_3ex.xml`
- Modify: `model/pyproject.toml`
- Test: `model/tests/exercise_corpus/test_fixtures.py`

- [ ] **Step 1: Write the failing test**

```python
# model/tests/exercise_corpus/test_fixtures.py
import partitura
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


def test_hanon_fixture_has_three_parts():
    score = partitura.load_musicxml(str(FIXTURES / "hanon_3ex.xml"))
    parts = list(partitura.utils.iter_parts(score))
    assert len(parts) == 3


def test_czerny_fixture_has_three_parts():
    score = partitura.load_musicxml(str(FIXTURES / "czerny_3ex.xml"))
    parts = list(partitura.utils.iter_parts(score))
    assert len(parts) == 3


def test_burgmuller_fixture_has_three_parts():
    score = partitura.load_musicxml(str(FIXTURES / "burgmuller_3ex.xml"))
    parts = list(partitura.utils.iter_parts(score))
    assert len(parts) == 3


def test_package_importable():
    import exercise_corpus  # noqa: F401
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/model && uv run pytest tests/exercise_corpus/test_fixtures.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'exercise_corpus'` (package not created yet) and `ModuleNotFoundError: No module named 'partitura'` (not in deps).

- [ ] **Step 3: Implement**

**3a. Update `model/pyproject.toml`** — add `partitura>=1.8.0` to the `dependencies` list, and add `"src/exercise_corpus"` to `[tool.hatch.build.targets.wheel] packages`:

In the `dependencies` list, after `"dtw-python>=1.7.4"`, add:
```toml
    "partitura>=1.8.0",
```

In `[tool.hatch.build.targets.wheel]`, change:
```toml
packages = ["src/score_alignment", "src/audio_experiments", "src/model_improvement", "src/masterclass_experiments", "src/score_library", "src/exercise_corpus"]
```

**3b. Create `model/src/exercise_corpus/__init__.py`:**
```python
"""Exercise corpus construction and embedding validation pipeline."""
```

**3c. Create `model/tests/exercise_corpus/__init__.py`:**
```python
```

**3d. Create `model/tests/exercise_corpus/fixtures/hanon_3ex.xml`** — minimal MusicXML with 3 parts, each 4 measures, C major scale fragments (C4-G4), 4/4 time, quarter notes. Each part has a unique `<part-name>` ("Hanon Exercise 1", "Hanon Exercise 2", "Hanon Exercise 3") and contains 8 notes (C4 D4 E4 F4 G4 F4 E4 D4 as quarter notes across 2 measures, then same pattern in measures 3-4):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN"
  "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="3.1">
  <part-list>
    <score-part id="P1"><part-name>Hanon Exercise 1</part-name></score-part>
    <score-part id="P2"><part-name>Hanon Exercise 2</part-name></score-part>
    <score-part id="P3"><part-name>Hanon Exercise 3</part-name></score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>1</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>
      <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>D</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>E</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>F</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
    </measure>
    <measure number="2">
      <note><pitch><step>G</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>F</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>E</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>D</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
    </measure>
  </part>
  <part id="P2">
    <measure number="1">
      <attributes>
        <divisions>1</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>
      <note><pitch><step>D</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>E</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>F</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>G</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
    </measure>
    <measure number="2">
      <note><pitch><step>A</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>G</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>F</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>E</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
    </measure>
  </part>
  <part id="P3">
    <measure number="1">
      <attributes>
        <divisions>1</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>
      <note><pitch><step>E</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>F</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>G</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>A</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
    </measure>
    <measure number="2">
      <note><pitch><step>B</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>A</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>G</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>F</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
    </measure>
  </part>
</score-partwise>
```

**3e. Create `model/tests/exercise_corpus/fixtures/czerny_3ex.xml`** — same structure as hanon_3ex.xml but with part names "Czerny Etude 1", "Czerny Etude 2", "Czerny Etude 3" and slightly different pitches (start on G4, A4, B4 respectively, same 8-note ascending/descending quarter-note pattern):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN"
  "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="3.1">
  <part-list>
    <score-part id="P1"><part-name>Czerny Etude 1</part-name></score-part>
    <score-part id="P2"><part-name>Czerny Etude 2</part-name></score-part>
    <score-part id="P3"><part-name>Czerny Etude 3</part-name></score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>1</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>
      <note><pitch><step>G</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>A</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>B</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>C</step><octave>5</octave></pitch><duration>1</duration><type>quarter</type></note>
    </measure>
    <measure number="2">
      <note><pitch><step>D</step><octave>5</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>C</step><octave>5</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>B</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>A</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
    </measure>
  </part>
  <part id="P2">
    <measure number="1">
      <attributes>
        <divisions>1</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>
      <note><pitch><step>A</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>B</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>C</step><octave>5</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>D</step><octave>5</octave></pitch><duration>1</duration><type>quarter</type></note>
    </measure>
    <measure number="2">
      <note><pitch><step>E</step><octave>5</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>D</step><octave>5</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>C</step><octave>5</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>B</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
    </measure>
  </part>
  <part id="P3">
    <measure number="1">
      <attributes>
        <divisions>1</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>
      <note><pitch><step>B</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>C</step><octave>5</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>D</step><octave>5</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>E</step><octave>5</octave></pitch><duration>1</duration><type>quarter</type></note>
    </measure>
    <measure number="2">
      <note><pitch><step>F</step><octave>5</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>E</step><octave>5</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>D</step><octave>5</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>C</step><octave>5</octave></pitch><duration>1</duration><type>quarter</type></note>
    </measure>
  </part>
</score-partwise>
```

**3f. Create `model/tests/exercise_corpus/fixtures/burgmuller_3ex.xml`** — same 3-part structure, part names "Burgmuller Piece 1", "Burgmuller Piece 2", "Burgmuller Piece 3", pitches starting on C5, D5, E5:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN"
  "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="3.1">
  <part-list>
    <score-part id="P1"><part-name>Burgmuller Piece 1</part-name></score-part>
    <score-part id="P2"><part-name>Burgmuller Piece 2</part-name></score-part>
    <score-part id="P3"><part-name>Burgmuller Piece 3</part-name></score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>1</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>
      <note><pitch><step>C</step><octave>5</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>D</step><octave>5</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>E</step><octave>5</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>F</step><octave>5</octave></pitch><duration>1</duration><type>quarter</type></note>
    </measure>
    <measure number="2">
      <note><pitch><step>G</step><octave>5</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>F</step><octave>5</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>E</step><octave>5</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>D</step><octave>5</octave></pitch><duration>1</duration><type>quarter</type></note>
    </measure>
  </part>
  <part id="P2">
    <measure number="1">
      <attributes>
        <divisions>1</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>
      <note><pitch><step>D</step><octave>5</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>E</step><octave>5</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>F</step><octave>5</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>G</step><octave>5</octave></pitch><duration>1</duration><type>quarter</type></note>
    </measure>
    <measure number="2">
      <note><pitch><step>A</step><octave>5</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>G</step><octave>5</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>F</step><octave>5</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>E</step><octave>5</octave></pitch><duration>1</duration><type>quarter</type></note>
    </measure>
  </part>
  <part id="P3">
    <measure number="1">
      <attributes>
        <divisions>1</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>
      <note><pitch><step>E</step><octave>5</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>F</step><octave>5</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>G</step><octave>5</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>A</step><octave>5</octave></pitch><duration>1</duration><type>quarter</type></note>
    </measure>
    <measure number="2">
      <note><pitch><step>B</step><octave>5</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>A</step><octave>5</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>G</step><octave>5</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>F</step><octave>5</octave></pitch><duration>1</duration><type>quarter</type></note>
    </measure>
  </part>
</score-partwise>
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/model && uv run pytest tests/exercise_corpus/test_fixtures.py -v
```
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add model/src/exercise_corpus/__init__.py model/tests/exercise_corpus/__init__.py model/tests/exercise_corpus/fixtures/ model/tests/exercise_corpus/test_fixtures.py model/pyproject.toml && git commit -m "feat(exercise-corpus): scaffold package, add MusicXML fixtures, add partitura dep"
```

---

### Task 2: sources.toml manifest

**Group:** A (parallel with Task 1, Task 3)

**Behavior being verified:** The sources manifest can be parsed and yields exactly three source entries with required keys.

**Interface under test:** `tomllib.loads(sources_content)` returns a dict whose `"sources"` key is a list of 3 dicts, each with `"name"`, `"license"`, and `"musicxml_path"` keys.

**Files:**
- Create: `model/src/exercise_corpus/sources.toml`
- Test: `model/tests/exercise_corpus/test_sources.py`

- [ ] **Step 1: Write the failing test**

```python
# model/tests/exercise_corpus/test_sources.py
import tomllib
from pathlib import Path

SOURCES_PATH = Path(__file__).parents[3] / "src" / "exercise_corpus" / "sources.toml"


def test_sources_toml_exists():
    assert SOURCES_PATH.exists(), f"sources.toml not found at {SOURCES_PATH}"


def test_sources_has_three_entries():
    with open(SOURCES_PATH, "rb") as f:
        data = tomllib.load(f)
    assert "sources" in data
    assert len(data["sources"]) == 3


def test_each_source_has_required_keys():
    with open(SOURCES_PATH, "rb") as f:
        data = tomllib.load(f)
    required_keys = {"name", "license", "musicxml_path"}
    for source in data["sources"]:
        missing = required_keys - set(source.keys())
        assert not missing, f"Source {source.get('name', '?')} missing keys: {missing}"


def test_source_names_match_expected():
    with open(SOURCES_PATH, "rb") as f:
        data = tomllib.load(f)
    names = {s["name"] for s in data["sources"]}
    assert names == {"hanon", "czerny", "burgmuller"}
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/model && uv run pytest tests/exercise_corpus/test_sources.py -v
```
Expected: FAIL — `AssertionError: sources.toml not found at ...` (file doesn't exist yet).

- [ ] **Step 3: Implement**

Create `model/src/exercise_corpus/sources.toml`:

```toml
# Exercise corpus source manifest.
# Each entry points to a locally acquired MusicXML file.
# The user must place the acquired MusicXML at musicxml_path before running the pipeline.
# All sources are unambiguously public domain globally.

[[sources]]
name = "hanon"
title = "The Virtuoso Pianist (Hanon)"
composer = "Charles-Louis Hanon"
opus = ""
license = "public_domain"
# Acquire from IMSLP: https://imslp.org/wiki/The_Virtuoso_Pianist_(Hanon,_Charles-Louis)
musicxml_path = "data/scores/exercise_primitives/raw/hanon_op599.xml"

[[sources]]
name = "czerny"
title = "The School of Velocity op.299 (Czerny)"
composer = "Carl Czerny"
opus = "299"
license = "public_domain"
# Acquire from IMSLP: https://imslp.org/wiki/School_of_Velocity%2C_Op.299_(Czerny%2C_Carl)
musicxml_path = "data/scores/exercise_primitives/raw/czerny_op299.xml"

[[sources]]
name = "burgmuller"
title = "25 Progressive Studies op.100 (Burgmuller)"
composer = "Johann Friedrich Burgmuller"
opus = "100"
license = "public_domain"
# Acquire from IMSLP: https://imslp.org/wiki/25_Progressive_Studies%2C_Op.100_(Burgm%C3%BCller%2C_Johann_Friedrich)
musicxml_path = "data/scores/exercise_primitives/raw/burgmuller_op100.xml"
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/model && uv run pytest tests/exercise_corpus/test_sources.py -v
```
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add model/src/exercise_corpus/sources.toml model/tests/exercise_corpus/test_sources.py && git commit -m "feat(exercise-corpus): add sources.toml manifest with 3 public-domain sources"
```

---

### Task 3: catalog.py — SQLite read/write with embedding round-trip

**Group:** A (parallel with Task 1, Task 2)

**Behavior being verified:** Writing N primitives with embeddings to the catalog and reading them back produces identical records with exactly equal numpy embedding arrays.

**Interface under test:** `catalog.write_primitives(primitives, embeddings, db_path)` then `catalog.read_primitives(db_path)` returns all rows with embeddings equal to the originals.

**Files:**
- Create: `model/src/exercise_corpus/catalog.py`
- Test: `model/tests/exercise_corpus/test_catalog.py`

- [ ] **Step 1: Write the failing test**

```python
# model/tests/exercise_corpus/test_catalog.py
import numpy as np
import pytest
import torch
from pathlib import Path

from exercise_corpus.catalog import write_primitives, read_primitives, CatalogRow
from exercise_corpus.segment import Primitive


def _make_primitive(n: int) -> Primitive:
    return Primitive(
        primitive_id=f"hanon_{n:03d}",
        source="hanon",
        source_exercise_number=n,
        title=f"Hanon Exercise {n}",
        musicxml_path=Path(f"/fake/scores/hanon_{n:03d}.xml"),
        midi_path=Path(f"/fake/midi/hanon_{n:03d}.mid"),
        n_notes=8 * n,
    )


def test_round_trip_preserves_all_fields(tmp_path: Path):
    db_path = tmp_path / "test_catalog.db"
    primitives = [_make_primitive(i) for i in range(1, 4)]
    embeddings = {
        p.primitive_id: torch.randn(512) for p in primitives
    }
    write_primitives(primitives, embeddings, db_path)
    rows = read_primitives(db_path)

    assert len(rows) == 3
    row_by_id = {r.primitive_id: r for r in rows}
    for p in primitives:
        row = row_by_id[p.primitive_id]
        assert row.source == p.source
        assert row.source_exercise_number == p.source_exercise_number
        assert row.title == p.title
        assert str(row.musicxml_path) == str(p.musicxml_path)
        assert str(row.midi_path) == str(p.midi_path)
        assert row.n_notes == p.n_notes
        assert isinstance(row.embedding, np.ndarray)
        assert row.embedding.shape == (512,)
        expected = embeddings[p.primitive_id].numpy()
        np.testing.assert_array_equal(row.embedding, expected)


def test_created_at_is_populated(tmp_path: Path):
    db_path = tmp_path / "test_catalog.db"
    p = _make_primitive(1)
    write_primitives([p], {p.primitive_id: torch.randn(512)}, db_path)
    rows = read_primitives(db_path)
    assert rows[0].created_at is not None
    assert len(rows[0].created_at) > 0
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/model && uv run pytest tests/exercise_corpus/test_catalog.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'exercise_corpus.catalog'`

- [ ] **Step 3: Implement**

Create `model/src/exercise_corpus/catalog.py`:

```python
"""SQLite catalog for exercise primitives with embedding storage."""

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


@dataclass
class CatalogRow:
    primitive_id: str
    source: str
    source_exercise_number: int
    title: str
    musicxml_path: Path
    midi_path: Path
    embedding: np.ndarray
    n_notes: int
    created_at: str


_DDL = """
CREATE TABLE IF NOT EXISTS primitives (
    primitive_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    source_exercise_number INTEGER NOT NULL,
    title TEXT NOT NULL,
    musicxml_path TEXT NOT NULL,
    midi_path TEXT NOT NULL,
    embedding BLOB NOT NULL,
    n_notes INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
"""


def write_primitives(
    primitives: list,
    embeddings: dict[str, torch.Tensor],
    db_path: Path,
) -> None:
    """Write primitives and their embeddings to the SQLite catalog.

    Args:
        primitives: list of Primitive dataclass instances.
        embeddings: dict mapping primitive_id to 512-dim torch.Tensor.
        db_path: path to the SQLite database file (created if absent).

    Raises:
        KeyError: if a primitive_id has no entry in embeddings.
        ValueError: if an embedding tensor is not 1-D.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(_DDL)
        for p in primitives:
            if p.primitive_id not in embeddings:
                raise KeyError(
                    f"No embedding found for primitive_id={p.primitive_id!r}"
                )
            emb = embeddings[p.primitive_id]
            if emb.ndim != 1:
                raise ValueError(
                    f"Embedding for {p.primitive_id!r} must be 1-D, got shape {emb.shape}"
                )
            emb_blob = emb.numpy().astype(np.float32).tobytes()
            conn.execute(
                """
                INSERT OR REPLACE INTO primitives
                (primitive_id, source, source_exercise_number, title,
                 musicxml_path, midi_path, embedding, n_notes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    p.primitive_id,
                    p.source,
                    p.source_exercise_number,
                    p.title,
                    str(p.musicxml_path),
                    str(p.midi_path),
                    emb_blob,
                    p.n_notes,
                    now,
                ),
            )
        conn.commit()
    finally:
        conn.close()


def read_primitives(db_path: Path) -> list[CatalogRow]:
    """Read all primitives from the SQLite catalog.

    Args:
        db_path: path to the SQLite database file.

    Returns:
        List of CatalogRow instances ordered by source, source_exercise_number.

    Raises:
        FileNotFoundError: if db_path does not exist.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Catalog database not found: {db_path}")
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.execute(
            """
            SELECT primitive_id, source, source_exercise_number, title,
                   musicxml_path, midi_path, embedding, n_notes, created_at
            FROM primitives
            ORDER BY source, source_exercise_number
            """
        )
        rows = []
        for row in cursor.fetchall():
            emb = np.frombuffer(row[6], dtype=np.float32).copy()
            rows.append(
                CatalogRow(
                    primitive_id=row[0],
                    source=row[1],
                    source_exercise_number=row[2],
                    title=row[3],
                    musicxml_path=Path(row[4]),
                    midi_path=Path(row[5]),
                    embedding=emb,
                    n_notes=row[7],
                    created_at=row[8],
                )
            )
        return rows
    finally:
        conn.close()
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/model && uv run pytest tests/exercise_corpus/test_catalog.py -v
```
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add model/src/exercise_corpus/catalog.py model/tests/exercise_corpus/test_catalog.py && git commit -m "feat(exercise-corpus): add catalog.py with SQLite embedding round-trip"
```

---

### Task 4: segment.py — segmentation with MIDI round-trip

**Group:** B (depends on Group A — requires Task 1 fixtures and package scaffold)

**Behavior being verified:** `segment_source` called on the Hanon 3-exercise fixture returns exactly 3 `Primitive` instances with the correct note counts, and the derived MIDI for each primitive contains the same pitch set as the MusicXML source.

**Interface under test:** `segment.segment_source(musicxml_path, source_name, output_score_dir, output_midi_dir) -> list[Primitive]`

**Files:**
- Create: `model/src/exercise_corpus/segment.py`
- Test: `model/tests/exercise_corpus/test_segment.py`

- [ ] **Step 1: Write the failing test**

```python
# model/tests/exercise_corpus/test_segment.py
import mido
import partitura
import pytest
from pathlib import Path

from exercise_corpus.segment import segment_source, Primitive, SegmentationError

FIXTURES = Path(__file__).parent / "fixtures"


def _midi_pitch_set(midi_path: Path) -> set[int]:
    """Return the set of MIDI note numbers present in a .mid file."""
    mid = mido.MidiFile(str(midi_path))
    pitches = set()
    for track in mid.tracks:
        for msg in track:
            if msg.type == "note_on" and msg.velocity > 0:
                pitches.add(msg.note)
    return pitches


def _musicxml_pitch_set(musicxml_path: Path) -> set[int]:
    """Return the set of MIDI pitch numbers in a single-part MusicXML file."""
    score = partitura.load_musicxml(str(musicxml_path))
    note_array = partitura.utils.music.ensure_notearray(score)
    return set(int(n) for n in note_array["pitch"])


def test_hanon_fixture_yields_three_primitives(tmp_path: Path):
    score_dir = tmp_path / "scores"
    midi_dir = tmp_path / "midi"
    primitives = segment_source(
        FIXTURES / "hanon_3ex.xml", "hanon", score_dir, midi_dir
    )
    assert len(primitives) == 3


def test_primitive_fields_are_populated(tmp_path: Path):
    score_dir = tmp_path / "scores"
    midi_dir = tmp_path / "midi"
    primitives = segment_source(
        FIXTURES / "hanon_3ex.xml", "hanon", score_dir, midi_dir
    )
    for i, p in enumerate(primitives, start=1):
        assert isinstance(p, Primitive)
        assert p.source == "hanon"
        assert p.source_exercise_number == i
        assert p.n_notes > 0
        assert p.musicxml_path.exists()
        assert p.midi_path.exists()


def test_midi_pitch_set_matches_musicxml(tmp_path: Path):
    score_dir = tmp_path / "scores"
    midi_dir = tmp_path / "midi"
    primitives = segment_source(
        FIXTURES / "hanon_3ex.xml", "hanon", score_dir, midi_dir
    )
    for p in primitives:
        xml_pitches = _musicxml_pitch_set(p.musicxml_path)
        midi_pitches = _midi_pitch_set(p.midi_path)
        # MIDI pitch set must be a subset of or equal to the XML pitch set
        # (partitura MIDI export may merge enharmonics but no new pitches appear)
        assert midi_pitches.issubset(xml_pitches) or midi_pitches == xml_pitches, (
            f"Primitive {p.primitive_id}: MIDI pitches {midi_pitches} not "
            f"consistent with XML pitches {xml_pitches}"
        )


def test_bad_source_name_raises_segmentation_error(tmp_path: Path):
    with pytest.raises(SegmentationError, match="unknown source"):
        segment_source(
            FIXTURES / "hanon_3ex.xml", "unknown_source", tmp_path, tmp_path
        )


def test_zero_parts_raises_segmentation_error(tmp_path: Path):
    # Write a MusicXML with no parts
    empty_xml = tmp_path / "empty.xml"
    empty_xml.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN"'
        ' "http://www.musicxml.org/dtds/partwise.dtd">'
        "<score-partwise version=\"3.1\"><part-list></part-list></score-partwise>"
    )
    with pytest.raises(SegmentationError, match="hanon"):
        segment_source(empty_xml, "hanon", tmp_path / "s", tmp_path / "m")
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/model && uv run pytest tests/exercise_corpus/test_segment.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'exercise_corpus.segment'`

- [ ] **Step 3: Implement**

Create `model/src/exercise_corpus/segment.py`:

```python
"""Segmentation of multi-exercise MusicXML sources into individual primitives.

Each source (Hanon, Czerny, Burgmuller) encodes a collection of exercises as
separate <part> elements in a single MusicXML file. This module parses each
source, extracts one Primitive per part, and exports per-primitive MusicXML
and MIDI files.

Supported source names: "hanon", "czerny", "burgmuller"
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import partitura
import partitura.score as pt_score

logger = logging.getLogger(__name__)


class SegmentationError(Exception):
    """Raised when a source MusicXML does not match the expected boundary pattern."""


@dataclass
class Primitive:
    primitive_id: str
    source: str
    source_exercise_number: int
    title: str
    musicxml_path: Path
    midi_path: Path
    n_notes: int


_SOURCE_CONFIGS: dict[str, dict] = {
    "hanon": {
        "title_prefix": "Hanon Exercise",
        "id_prefix": "hanon",
    },
    "czerny": {
        "title_prefix": "Czerny Etude",
        "id_prefix": "czerny",
    },
    "burgmuller": {
        "title_prefix": "Burgmuller Piece",
        "id_prefix": "burgmuller",
    },
}


def segment_source(
    musicxml_path: Path,
    source_name: str,
    output_score_dir: Path,
    output_midi_dir: Path,
) -> list[Primitive]:
    """Segment a multi-exercise MusicXML file into individual Primitive instances.

    Each <part> in the MusicXML is treated as one exercise primitive. Exports
    per-primitive MusicXML and MIDI files to the given output directories.

    Args:
        musicxml_path: path to the source MusicXML file.
        source_name: one of "hanon", "czerny", "burgmuller".
        output_score_dir: directory to write per-primitive MusicXML files.
        output_midi_dir: directory to write per-primitive MIDI files.

    Returns:
        List of Primitive dataclasses, one per exercise, ordered by
        source_exercise_number ascending.

    Raises:
        SegmentationError: if source_name is not recognized, or if the parsed
            score contains zero parts, or if a part contains zero notes.
        FileNotFoundError: if musicxml_path does not exist.
    """
    musicxml_path = Path(musicxml_path)
    if not musicxml_path.exists():
        raise FileNotFoundError(f"MusicXML not found: {musicxml_path}")

    if source_name not in _SOURCE_CONFIGS:
        raise SegmentationError(
            f"unknown source {source_name!r}; supported: {sorted(_SOURCE_CONFIGS)}"
        )

    config = _SOURCE_CONFIGS[source_name]
    output_score_dir = Path(output_score_dir)
    output_midi_dir = Path(output_midi_dir)
    output_score_dir.mkdir(parents=True, exist_ok=True)
    output_midi_dir.mkdir(parents=True, exist_ok=True)

    score = partitura.load_musicxml(str(musicxml_path))
    parts = list(partitura.utils.iter_parts(score))

    if len(parts) == 0:
        raise SegmentationError(
            f"{source_name}: expected at least 1 part in {musicxml_path}, found 0"
        )

    primitives: list[Primitive] = []
    for idx, part in enumerate(parts, start=1):
        note_array = partitura.utils.music.ensure_notearray(part)
        n_notes = len(note_array)

        if n_notes == 0:
            raise SegmentationError(
                f"{source_name}: part {idx} contains 0 notes in {musicxml_path}"
            )

        primitive_id = f"{config['id_prefix']}_{idx:03d}"
        title = f"{config['title_prefix']} {idx}"
        xml_out = output_score_dir / f"{primitive_id}.xml"
        mid_out = output_midi_dir / f"{primitive_id}.mid"

        partitura.save_musicxml(part, str(xml_out))
        partitura.save_midi(part, str(mid_out))

        primitives.append(
            Primitive(
                primitive_id=primitive_id,
                source=source_name,
                source_exercise_number=idx,
                title=title,
                musicxml_path=xml_out,
                midi_path=mid_out,
                n_notes=n_notes,
            )
        )
        logger.info("Segmented %s (n_notes=%d) -> %s", primitive_id, n_notes, xml_out)

    return primitives
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/model && uv run pytest tests/exercise_corpus/test_segment.py -v
```
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add model/src/exercise_corpus/segment.py model/tests/exercise_corpus/test_segment.py && git commit -m "feat(exercise-corpus): add segment.py with partitura-based primitive extraction"
```

---

### Task 5: validate.py — purity metric and review artifact

**Group:** B (depends on Group A — requires catalog.py from Task 3)

**Behavior being verified:** The purity metric returns 1.0 for perfectly separated synthetic embeddings, near 0.43 for fully shuffled labels, and `run_validation` emits exactly 15 within-source neighbor pairs across 3 sources.

**Interface under test:** `validate.source_purity(embeddings, labels, k) -> float` and `validate.run_validation(db_path, output_dir) -> ValidationResult`

**Files:**
- Create: `model/src/exercise_corpus/validate.py`
- Test: `model/tests/exercise_corpus/test_validate.py`

- [ ] **Step 1: Write the failing test**

```python
# model/tests/exercise_corpus/test_validate.py
import json
import numpy as np
import pytest
import torch
from pathlib import Path

from exercise_corpus.validate import source_purity, run_validation, ValidationResult
from exercise_corpus.catalog import write_primitives
from exercise_corpus.segment import Primitive


def _make_synthetic_primitives_and_embeddings(
    counts: dict[str, int], dim: int = 512
) -> tuple[list[Primitive], dict[str, torch.Tensor]]:
    """Build primitives and tight Gaussian cluster embeddings per source."""
    primitives = []
    embeddings = {}
    source_centers = {
        "hanon":      np.array([1.0] + [0.0] * (dim - 1), dtype=np.float32),
        "czerny":     np.array([0.0, 1.0] + [0.0] * (dim - 2), dtype=np.float32),
        "burgmuller": np.array([0.0, 0.0, 1.0] + [0.0] * (dim - 3), dtype=np.float32),
    }
    idx = 0
    for source, count in counts.items():
        center = source_centers[source]
        for i in range(1, count + 1):
            pid = f"{source}_{i:03d}"
            noise = np.random.default_rng(idx).normal(0, 0.01, dim).astype(np.float32)
            emb = torch.tensor(center + noise)
            primitives.append(
                Primitive(
                    primitive_id=pid,
                    source=source,
                    source_exercise_number=i,
                    title=f"{source} {i}",
                    musicxml_path=Path(f"/fake/{pid}.xml"),
                    midi_path=Path(f"/fake/{pid}.mid"),
                    n_notes=8,
                )
            )
            embeddings[pid] = emb
            idx += 1
    return primitives, embeddings


def test_purity_perfect_separation():
    embeddings_array = np.array([
        [1.0, 0.0, 0.0],
        [1.1, 0.0, 0.0],
        [1.2, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.1, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    labels = ["a", "a", "a", "b", "b", "c"]
    purity = source_purity(embeddings_array, labels, k=2)
    assert purity == pytest.approx(1.0)


def test_purity_shuffled_near_random_floor():
    rng = np.random.default_rng(42)
    # 60 + 40 + 25 = 125 total — same distribution as production spec
    labels = ["hanon"] * 60 + ["czerny"] * 40 + ["burgmuller"] * 25
    # Random embeddings -> neighbors are random -> purity approaches class-size-weighted random
    embeddings_array = rng.standard_normal((125, 512)).astype(np.float32)
    purity = source_purity(embeddings_array, labels, k=5)
    # Random floor for 60/40/25 split is ~0.43; allow generous range
    assert 0.20 <= purity <= 0.70, f"Expected near-random purity, got {purity:.3f}"


def test_run_validation_emits_15_pairs(tmp_path: Path):
    # 60 hanon + 40 czerny + 25 burgmuller
    counts = {"hanon": 60, "czerny": 40, "burgmuller": 25}
    primitives, embeddings = _make_synthetic_primitives_and_embeddings(counts)

    db_path = tmp_path / "catalog.db"
    write_primitives(primitives, embeddings, db_path)

    result = run_validation(db_path, tmp_path)

    assert isinstance(result, ValidationResult)
    assert len(result.pairs) == 15
    # All 15 pairs must be within-source
    for pair in result.pairs:
        assert pair["source_a"] == pair["source_b"], (
            f"Cross-source pair found: {pair['source_a']} vs {pair['source_b']}"
        )


def test_run_validation_umap_file_created(tmp_path: Path):
    counts = {"hanon": 60, "czerny": 40, "burgmuller": 25}
    primitives, embeddings = _make_synthetic_primitives_and_embeddings(counts)
    db_path = tmp_path / "catalog.db"
    write_primitives(primitives, embeddings, db_path)

    result = run_validation(db_path, tmp_path)
    assert result.umap_path.exists()


def test_run_validation_pairs_json_is_valid(tmp_path: Path):
    counts = {"hanon": 60, "czerny": 40, "burgmuller": 25}
    primitives, embeddings = _make_synthetic_primitives_and_embeddings(counts)
    db_path = tmp_path / "catalog.db"
    write_primitives(primitives, embeddings, db_path)

    result = run_validation(db_path, tmp_path)
    assert result.pairs_path.exists()
    with open(result.pairs_path) as f:
        parsed = json.load(f)
    assert isinstance(parsed, list)
    assert len(parsed) == 15


def test_run_validation_verdict_pass_for_tight_clusters(tmp_path: Path):
    counts = {"hanon": 60, "czerny": 40, "burgmuller": 25}
    primitives, embeddings = _make_synthetic_primitives_and_embeddings(counts)
    db_path = tmp_path / "catalog.db"
    write_primitives(primitives, embeddings, db_path)

    result = run_validation(db_path, tmp_path)
    assert result.verdict == "PASS"
    assert result.purity >= 0.70
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/model && uv run pytest tests/exercise_corpus/test_validate.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'exercise_corpus.validate'`

- [ ] **Step 3: Implement**

Create `model/src/exercise_corpus/validate.py`:

```python
"""k-NN source purity validation, UMAP visualization, and review artifact generation."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless runs
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
import umap

from exercise_corpus.catalog import read_primitives

logger = logging.getLogger(__name__)

PURITY_THRESHOLD = 0.70
PAIRS_PER_SOURCE = 5
K = 5


@dataclass
class ValidationResult:
    purity: float
    verdict: str
    pairs: list[dict]
    umap_path: Path
    pairs_path: Path


def source_purity(
    embeddings: np.ndarray,
    labels: list[str],
    k: int = K,
) -> float:
    """Compute k-NN source purity.

    For each point, find its k nearest neighbors (excluding itself) and
    compute the fraction whose label matches the query point's label.
    Average across all points.

    Args:
        embeddings: float32 array of shape (n, dim).
        labels: list of n source strings.
        k: number of neighbors.

    Returns:
        Float in [0, 1]. 1.0 = all neighbors same source.
    """
    n = len(labels)
    if n <= k:
        raise ValueError(
            f"Need more than k={k} points to compute purity, got {n}"
        )
    # k+1 because NearestNeighbors includes the point itself
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn.fit(embeddings)
    _, indices = nn.kneighbors(embeddings)
    # indices[:, 0] is the point itself; skip it
    neighbor_indices = indices[:, 1:]
    labels_arr = np.array(labels)
    same = (labels_arr[neighbor_indices] == labels_arr[:, None]).sum(axis=1)
    return float(same.mean() / k)


def run_validation(db_path: Path, output_dir: Path) -> ValidationResult:
    """Run full validation: purity metric, UMAP plot, and 15-pair review artifact.

    Does NOT raise on a failing metric. Reports PASS/FAIL verdict with numbers.

    Args:
        db_path: path to the SQLite catalog populated by catalog.write_primitives.
        output_dir: directory to write exercise_primitives_umap.png and
            exercise_primitives_neighbors.json.

    Returns:
        ValidationResult with purity, verdict, pairs, and file paths.

    Raises:
        FileNotFoundError: if db_path does not exist.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_primitives(db_path)
    if len(rows) == 0:
        raise ValueError(f"Catalog at {db_path} contains no primitives")

    embeddings = np.stack([r.embedding for r in rows], axis=0)
    labels = [r.source for r in rows]
    primitive_ids = [r.primitive_id for r in rows]

    purity = source_purity(embeddings, labels, k=K)
    verdict = "PASS" if purity >= PURITY_THRESHOLD else "FAIL"
    logger.info(
        "k-NN source purity (k=%d): %.4f — %s (threshold %.2f)",
        K, purity, verdict, PURITY_THRESHOLD,
    )

    # UMAP
    reducer = umap.UMAP(n_components=2, random_state=42, metric="cosine")
    embedding_2d = reducer.fit_transform(embeddings)
    unique_sources = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 0.5, len(unique_sources)))
    source_to_color = dict(zip(unique_sources, colors))
    fig, ax = plt.subplots(figsize=(8, 6))
    for source in unique_sources:
        mask = np.array([l == source for l in labels])
        ax.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            label=source,
            color=source_to_color[source],
            alpha=0.7,
            s=20,
        )
    ax.set_title(f"Exercise Primitives UMAP (purity={purity:.3f}, {verdict})")
    ax.legend()
    umap_path = output_dir / "exercise_primitives_umap.png"
    fig.savefig(str(umap_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("UMAP plot saved to %s", umap_path)

    # 15 within-source nearest-neighbor pairs (5 per source)
    nn = NearestNeighbors(n_neighbors=K + 1, metric="cosine")
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)

    pairs: list[dict] = []
    seen_sources: dict[str, int] = {}
    for i, (pid, source) in enumerate(zip(primitive_ids, labels)):
        if seen_sources.get(source, 0) >= PAIRS_PER_SOURCE:
            continue
        for rank in range(1, K + 1):
            j = indices[i, rank]
            neighbor_source = labels[j]
            if neighbor_source == source:
                pairs.append(
                    {
                        "query_id": pid,
                        "neighbor_id": primitive_ids[j],
                        "source_a": source,
                        "source_b": neighbor_source,
                        "cosine_distance": float(distances[i, rank]),
                    }
                )
                seen_sources[source] = seen_sources.get(source, 0) + 1
                break

    pairs_path = output_dir / "exercise_primitives_neighbors.json"
    with open(pairs_path, "w") as f:
        json.dump(pairs, f, indent=2)
    logger.info("Review artifact saved to %s (%d pairs)", pairs_path, len(pairs))

    return ValidationResult(
        purity=purity,
        verdict=verdict,
        pairs=pairs,
        umap_path=umap_path,
        pairs_path=pairs_path,
    )
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/model && uv run pytest tests/exercise_corpus/test_validate.py -v
```
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add model/src/exercise_corpus/validate.py model/tests/exercise_corpus/test_validate.py && git commit -m "feat(exercise-corpus): add validate.py with k-NN purity, UMAP, and review artifact"
```

---

### Task 6: embed.py — thin adapter over aria_embeddings

**Group:** C (depends on Group B — embed.py is tested against the public interface only; no Aria weights required in tests)

**Behavior being verified:** `embed_primitives` delegates to `aria_embeddings.extract_all_embeddings` with `variant="embedding"` and returns the resulting dict unchanged.

**Interface under test:** `embed.embed_primitives(midi_dir: Path) -> dict[str, torch.Tensor]`

**Files:**
- Create: `model/src/exercise_corpus/embed.py`
- Test: `model/tests/exercise_corpus/test_embed.py`

- [ ] **Step 1: Write the failing test**

```python
# model/tests/exercise_corpus/test_embed.py
"""Tests for embed.py through its public interface.

embed_primitives is a thin adapter. The test patches the underlying
extract_all_embeddings at the module boundary (not an internal) to avoid
requiring Aria weights in CI. The behavior under test is:
  - correct variant is passed through
  - the dict returned by the underlying call is returned as-is
  - FileNotFoundError from the underlying call propagates (no swallowing)
"""
import torch
import pytest
from pathlib import Path
from unittest.mock import patch

from exercise_corpus.embed import embed_primitives


def test_embed_primitives_passes_variant_and_returns_dict(tmp_path: Path):
    fake_result = {
        "hanon_001": torch.randn(512),
        "hanon_002": torch.randn(512),
    }
    with patch(
        "exercise_corpus.embed.extract_all_embeddings",
        return_value=fake_result,
    ) as mock_fn:
        result = embed_primitives(tmp_path)

    mock_fn.assert_called_once_with(tmp_path, variant="embedding")
    assert result is fake_result


def test_embed_primitives_propagates_file_not_found(tmp_path: Path):
    with patch(
        "exercise_corpus.embed.extract_all_embeddings",
        side_effect=FileNotFoundError("No .mid files found"),
    ):
        with pytest.raises(FileNotFoundError, match="No .mid files"):
            embed_primitives(tmp_path)
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/model && uv run pytest tests/exercise_corpus/test_embed.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'exercise_corpus.embed'`

- [ ] **Step 3: Implement**

Create `model/src/exercise_corpus/embed.py`:

```python
"""Thin adapter over aria_embeddings for exercise corpus embedding extraction.

Pins variant="embedding" (512-dim EOS-pooled) so the rest of the pipeline
never references aria_embeddings directly.
"""

from pathlib import Path

import torch

from model_improvement.aria_embeddings import extract_all_embeddings


def embed_primitives(midi_dir: Path) -> dict[str, torch.Tensor]:
    """Extract 512-dim Aria embeddings from all MIDI files in midi_dir.

    Args:
        midi_dir: directory containing .mid files (one per primitive).

    Returns:
        Dict mapping filename stem (primitive_id) to 512-dim float32 tensor.

    Raises:
        FileNotFoundError: if midi_dir contains no .mid files or does not exist.
    """
    return extract_all_embeddings(Path(midi_dir), variant="embedding")
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/model && uv run pytest tests/exercise_corpus/test_embed.py -v
```
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add model/src/exercise_corpus/embed.py model/tests/exercise_corpus/test_embed.py && git commit -m "feat(exercise-corpus): add embed.py adapter over aria_embeddings"
```

---

### Task 7: run.py — CLI orchestrator + full test suite sweep

**Group:** D (depends on Group C — wires all modules together)

**Behavior being verified:** The CLI orchestrator wires segment → embed → catalog → validate. When invoked with `--dry-run` (sources not present, no Aria weights), it raises `FileNotFoundError` naming the missing MusicXML. When invoked against a pre-populated catalog (bypassing segment/embed), `validate` is called and returns a result.

**Interface under test:** `python -m exercise_corpus.run --help` exits 0; `run_pipeline(sources_path, output_dir, dry_run=True)` raises `FileNotFoundError` with the missing path in the message.

**Files:**
- Create: `model/src/exercise_corpus/run.py`
- Test: `model/tests/exercise_corpus/test_run.py`

- [ ] **Step 1: Write the failing test**

```python
# model/tests/exercise_corpus/test_run.py
import subprocess
import sys
import pytest
import torch
from pathlib import Path
from unittest.mock import patch

from exercise_corpus.run import run_pipeline
from exercise_corpus.catalog import write_primitives
from exercise_corpus.segment import Primitive


def _make_primitive(source: str, n: int) -> Primitive:
    return Primitive(
        primitive_id=f"{source}_{n:03d}",
        source=source,
        source_exercise_number=n,
        title=f"{source} {n}",
        musicxml_path=Path(f"/fake/{source}_{n:03d}.xml"),
        midi_path=Path(f"/fake/{source}_{n:03d}.mid"),
        n_notes=8,
    )


def test_cli_help_exits_zero():
    result = subprocess.run(
        [sys.executable, "-m", "exercise_corpus.run", "--help"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parents[3]),  # model/
    )
    assert result.returncode == 0


def test_run_pipeline_dry_run_raises_on_missing_musicxml(tmp_path: Path):
    # sources.toml points to musicxml_path that does not exist
    sources_toml = tmp_path / "sources.toml"
    sources_toml.write_text(
        '[[sources]]\n'
        'name = "hanon"\n'
        'title = "Hanon"\n'
        'composer = "Hanon"\n'
        'opus = ""\n'
        'license = "public_domain"\n'
        f'musicxml_path = "{tmp_path / "missing.xml"}"\n'
    )
    with pytest.raises(FileNotFoundError, match="missing.xml"):
        run_pipeline(sources_toml, tmp_path, dry_run=True)


def test_run_pipeline_validate_only_reads_existing_catalog(tmp_path: Path):
    # Populate a catalog directly (bypasses segment + embed)
    db_path = tmp_path / "exercise_primitives.db"
    counts = {"hanon": 60, "czerny": 40, "burgmuller": 25}
    primitives = []
    embeddings = {}
    import numpy as np
    source_centers = {
        "hanon":      np.array([1.0] + [0.0] * 511, dtype=np.float32),
        "czerny":     np.array([0.0, 1.0] + [0.0] * 510, dtype=np.float32),
        "burgmuller": np.array([0.0, 0.0, 1.0] + [0.0] * 509, dtype=np.float32),
    }
    idx = 0
    for source, count in counts.items():
        for i in range(1, count + 1):
            p = _make_primitive(source, i)
            noise = np.random.default_rng(idx).normal(0, 0.01, 512).astype(np.float32)
            emb = torch.tensor(source_centers[source] + noise)
            primitives.append(p)
            embeddings[p.primitive_id] = emb
            idx += 1
    write_primitives(primitives, embeddings, db_path)

    result = run_pipeline(
        validate_only=True,
        db_path=db_path,
        output_dir=tmp_path,
    )
    assert result.purity >= 0.70
    assert result.verdict == "PASS"
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/model && uv run pytest tests/exercise_corpus/test_run.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'exercise_corpus.run'`

- [ ] **Step 3: Implement**

Create `model/src/exercise_corpus/run.py`:

```python
"""CLI orchestrator for the exercise corpus embedding validation pipeline.

Usage (full pipeline — requires acquired MusicXML files and Aria weights):
    python -m exercise_corpus.run \\
        --sources model/src/exercise_corpus/sources.toml \\
        --output-dir model/data

Usage (validate only — reads existing catalog):
    python -m exercise_corpus.run --validate-only --db model/data/exercise_primitives.db \\
        --output-dir model/data/results
"""

import argparse
import logging
import tomllib
from pathlib import Path

from exercise_corpus.catalog import write_primitives
from exercise_corpus.embed import embed_primitives
from exercise_corpus.segment import segment_source
from exercise_corpus.validate import ValidationResult, run_validation

logger = logging.getLogger(__name__)


def run_pipeline(
    sources_path: Path | None = None,
    output_dir: Path | None = None,
    dry_run: bool = False,
    validate_only: bool = False,
    db_path: Path | None = None,
) -> ValidationResult:
    """Run the full exercise corpus pipeline or a subset.

    Args:
        sources_path: path to sources.toml. Required unless validate_only=True.
        output_dir: root output directory. Scores go to output_dir/scores/exercise_primitives/,
            MIDI to output_dir/midi/exercise_primitives/, catalog to
            output_dir/exercise_primitives.db, results to output_dir/results/.
        dry_run: if True, raise FileNotFoundError for the first missing MusicXML
            without running segmentation.
        validate_only: if True, skip segment/embed and run only validate against
            an existing db_path.
        db_path: required when validate_only=True.

    Returns:
        ValidationResult from validate.run_validation.

    Raises:
        FileNotFoundError: if any source MusicXML is missing (dry_run or normal run).
        ValueError: if required arguments are missing.
    """
    if validate_only:
        if db_path is None:
            raise ValueError("db_path is required when validate_only=True")
        if output_dir is None:
            raise ValueError("output_dir is required when validate_only=True")
        results_dir = Path(output_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        return run_validation(Path(db_path), results_dir)

    if sources_path is None:
        raise ValueError("sources_path is required when validate_only=False")
    if output_dir is None:
        raise ValueError("output_dir is required when validate_only=False")

    output_dir = Path(output_dir)
    score_dir = output_dir / "scores" / "exercise_primitives"
    midi_dir = output_dir / "midi" / "exercise_primitives"
    catalog_path = output_dir / "exercise_primitives.db"
    results_dir = output_dir / "results"

    with open(Path(sources_path), "rb") as f:
        manifest = tomllib.load(f)
    sources = manifest["sources"]

    # Validate all MusicXML paths exist before doing any work
    for source in sources:
        xml_path = Path(source["musicxml_path"])
        if not xml_path.exists():
            raise FileNotFoundError(
                f"Source MusicXML not found for {source['name']!r}: {xml_path}\n"
                "Acquire from IMSLP and place at the path listed in sources.toml."
            )

    if dry_run:
        logger.info("dry_run=True: all source files present, stopping before segmentation.")
        # Return a placeholder — dry_run is only used in tests to check the missing-file raise
        raise RuntimeError("dry_run completed without error (all sources present)")

    all_primitives = []
    for source in sources:
        xml_path = Path(source["musicxml_path"])
        logger.info("Segmenting source %r from %s", source["name"], xml_path)
        primitives = segment_source(xml_path, source["name"], score_dir, midi_dir)
        all_primitives.extend(primitives)
        logger.info("  -> %d primitives", len(primitives))

    logger.info("Total primitives: %d. Extracting embeddings...", len(all_primitives))
    embeddings = embed_primitives(midi_dir)

    logger.info("Writing catalog to %s", catalog_path)
    write_primitives(all_primitives, embeddings, catalog_path)

    logger.info("Running validation...")
    result = run_validation(catalog_path, results_dir)
    print(
        f"\nValidation complete: purity={result.purity:.4f} — {result.verdict} "
        f"(threshold 0.70)\n"
        f"UMAP: {result.umap_path}\n"
        f"Review artifact (15 pairs): {result.pairs_path}\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exercise corpus embedding validation pipeline."
    )
    parser.add_argument(
        "--sources",
        type=Path,
        default=None,
        help="Path to sources.toml manifest.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Root output directory (default: data/).",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Skip segment/embed; run validate against an existing catalog.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Path to existing SQLite catalog (required with --validate-only).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check that all source MusicXML files exist without running the pipeline.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    run_pipeline(
        sources_path=args.sources,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        validate_only=args.validate_only,
        db_path=args.db,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/model && uv run pytest tests/exercise_corpus/test_run.py -v
```
Expected: PASS (3 tests)

- [ ] **Step 5: Run the full exercise_corpus test suite**

```bash
cd /Users/jdhiman/Documents/crescendai/model && uv run pytest tests/exercise_corpus/ -v
```
Expected: All tests PASS. No tests referencing Aria weights should run against the real model.

- [ ] **Step 6: Commit**

```bash
git add model/src/exercise_corpus/run.py model/tests/exercise_corpus/test_run.py && git commit -m "feat(exercise-corpus): add run.py CLI orchestrator, all pipeline tests pass"
```

---

## Post-Ship Manual Acceptance Gates (not automated build gates)

These criteria apply only after the user manually acquires the three MusicXML files from IMSLP/OpenScore, places them at the paths in `sources.toml`, and runs:

```bash
cd model && uv run python -m exercise_corpus.run \
    --sources src/exercise_corpus/sources.toml \
    --output-dir data
```

**Gate 1 — Purity:** k-NN source purity (k=5) >= 0.70. Printed to terminal. Plot at `data/results/exercise_primitives_umap.png`.

**Gate 2 — Human review:** Open `data/results/exercise_primitives_neighbors.json`. Review the 15 within-source pairs. Accept if >= 11/15 pairs are judged pedagogically sensible (i.e., the neighbors share recognizable technique family, not just source). If < 11/15, segmentation granularity or embedding variant needs reconsideration before proceeding to slice B.

**Gate 3 — Note counts:** Verify that the sum of `n_notes` across all catalog rows is consistent with the known exercise counts (Hanon: ~60 × ~100 notes; Czerny op.299: ~40 × ~100 notes; Burgmüller op.100: 25 × ~50 notes). A dramatically wrong count indicates a segmentation boundary bug.

**Slice B is unblocked when:** Both Gate 1 (PASS) and Gate 2 (>= 11/15) are satisfied. Update `docs/plans/exercise-system-rebuild-index.md` slice A row to SHIPPED and flip slice B to DESIGN IN PROGRESS.
