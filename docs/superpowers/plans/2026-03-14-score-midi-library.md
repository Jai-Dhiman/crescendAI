# Score MIDI Library Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parse 242 ASAP score MIDIs into structured JSON, upload to R2 + D1, and serve via API endpoints.

**Architecture:** Python batch CLI pipeline (discover -> parse -> upload) produces per-piece JSON artifacts. D1 holds the searchable piece catalog, R2 holds full bar-centric score data. Rust API worker serves two GET endpoints with Cloudflare Cache API caching.

**Tech Stack:** Python (mido, pydantic, boto3), Rust/WASM (Cloudflare Workers, D1, R2), uv, wrangler

**Design spec:** `docs/superpowers/specs/2026-03-14-score-midi-library-design.md`

---

## File Structure

```
model/src/score_library/           # NEW -- Python batch pipeline
  __init__.py                      # Package init
  schema.py                        # Pydantic models for JSON output
  discover.py                      # Stage 1: Walk ASAP dirs, derive piece_id
  parse.py                         # Stage 2: MIDI -> bar-centric JSON
  upload.py                        # Stage 3: R2 + D1 upload
  cli.py                           # CLI entry point (build/parse/upload/stats)
  __main__.py                      # python -m score_library.cli support
  titles.py                        # Title generation + titles.json loader

model/data/score_library/          # OUTPUT -- generated JSON artifacts
  titles.json                      # Static title mapping (242 entries)
  {piece_id}.json                  # Per-piece score data (generated)

model/tests/score_library/         # NEW -- tests
  __init__.py
  test_schema.py                   # Pydantic model validation
  test_discover.py                 # Discovery + piece_id derivation
  test_parse.py                    # MIDI parsing, bar grid, edge cases
  test_golden.py                   # Golden file tests for bar grid accuracy
  test_integration.py              # Full pipeline on actual ASAP cache

apps/api/migrations/0003_pieces.sql  # NEW -- D1 pieces table
apps/api/src/services/scores.rs      # NEW -- score lookup handlers
apps/api/src/services/mod.rs         # MODIFY -- add scores module
apps/api/src/server.rs               # MODIFY -- add score routes
apps/api/wrangler.toml               # MODIFY -- add SCORES R2 binding
```

---

## Chunk 1: Pydantic Schema + Discovery

### Task 1: Pydantic Schema

**Files:**
- Create: `model/src/score_library/__init__.py`
- Create: `model/src/score_library/schema.py`
- Create: `model/tests/score_library/__init__.py`
- Create: `model/tests/score_library/test_schema.py`

- [ ] **Step 1: Write failing test for Pydantic models**

```python
# model/tests/score_library/test_schema.py
"""Tests for score library Pydantic schema."""

from score_library.schema import ScoreNote, PedalEvent, Bar, ScoreData, PieceCatalogEntry


def test_score_note_valid():
    note = ScoreNote(
        pitch=64, pitch_name="E4", velocity=49,
        onset_tick=0, onset_seconds=0.0,
        duration_ticks=240, duration_seconds=0.42, track=0,
    )
    assert note.pitch == 64
    assert note.pitch_name == "E4"


def test_bar_with_notes():
    note = ScoreNote(
        pitch=64, pitch_name="E4", velocity=49,
        onset_tick=0, onset_seconds=0.0,
        duration_ticks=240, duration_seconds=0.42, track=0,
    )
    bar = Bar(
        bar_number=1, start_tick=0, start_seconds=0.0,
        time_signature="2/4", notes=[note], pedal_events=[],
        note_count=1, pitch_range=[64, 64], mean_velocity=49,
    )
    assert bar.bar_number == 1
    assert len(bar.notes) == 1


def test_empty_bar():
    bar = Bar(
        bar_number=5, start_tick=1920, start_seconds=3.43,
        time_signature="4/4", notes=[], pedal_events=[],
        note_count=0, pitch_range=[0, 0], mean_velocity=0,
    )
    assert bar.note_count == 0


def test_score_data_serialization():
    note = ScoreNote(
        pitch=64, pitch_name="E4", velocity=49,
        onset_tick=0, onset_seconds=0.0,
        duration_ticks=240, duration_seconds=0.42, track=0,
    )
    bar = Bar(
        bar_number=1, start_tick=0, start_seconds=0.0,
        time_signature="2/4", notes=[note], pedal_events=[],
        note_count=1, pitch_range=[64, 64], mean_velocity=49,
    )
    score = ScoreData(
        piece_id="chopin.etudes_op_10.3",
        composer="Chopin",
        title="Etude Op. 10 No. 3",
        key_signature="E",
        time_signatures=[{"bar": 1, "numerator": 2, "denominator": 4}],
        tempo_markings=[{"bar": 1, "bpm": 100}],
        total_bars=1,
        bars=[bar],
    )
    d = score.model_dump()
    assert d["piece_id"] == "chopin.etudes_op_10.3"
    assert len(d["bars"]) == 1
    assert d["bars"][0]["notes"][0]["pitch_name"] == "E4"


def test_piece_catalog_entry():
    entry = PieceCatalogEntry(
        piece_id="chopin.etudes_op_10.3",
        composer="Chopin",
        title="Etude Op. 10 No. 3",
        key_signature="E",
        time_signature="2/4",
        tempo_bpm=100,
        bar_count=77,
        duration_seconds=231.0,
        note_count=1500,
        pitch_range_low=40,
        pitch_range_high=88,
        has_time_sig_changes=False,
        has_tempo_changes=False,
        source="asap",
    )
    assert entry.piece_id == "chopin.etudes_op_10.3"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd model && uv run pytest tests/score_library/test_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'score_library'`

- [ ] **Step 3: Register score_library package in pyproject.toml**

Add `"src/score_library"` to the packages list in `model/pyproject.toml`:

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/score_alignment", "src/audio_experiments", "src/disentanglement", "src/model_improvement", "src/masterclass_experiments", "src/score_library"]
```

Then run: `cd model && uv sync`

- [ ] **Step 4: Write Pydantic schema**

```python
# model/src/score_library/__init__.py
"""Score MIDI Library -- batch pipeline for parsing ASAP score MIDIs."""

# model/src/score_library/schema.py
"""Pydantic models for score library JSON output.

Defines the bar-centric data structure that downstream consumers
(score following, bar-aligned analysis, teacher feedback) depend on.
"""

from pydantic import BaseModel


class ScoreNote(BaseModel):
    pitch: int
    pitch_name: str
    velocity: int
    onset_tick: int
    onset_seconds: float
    duration_ticks: int
    duration_seconds: float
    track: int


class PedalEvent(BaseModel):
    type: str  # "on" or "off"
    tick: int
    seconds: float


class Bar(BaseModel):
    bar_number: int
    start_tick: int
    start_seconds: float
    time_signature: str
    notes: list[ScoreNote]
    pedal_events: list[PedalEvent]
    note_count: int
    pitch_range: list[int]  # [low, high]
    mean_velocity: int


class ScoreData(BaseModel):
    """Full parsed score -- stored in R2 as JSON."""
    piece_id: str
    composer: str
    title: str
    key_signature: str | None
    time_signatures: list[dict]
    tempo_markings: list[dict]
    total_bars: int
    bars: list[Bar]


class PieceCatalogEntry(BaseModel):
    """Row in the D1 pieces table."""
    piece_id: str
    composer: str
    title: str
    key_signature: str | None
    time_signature: str | None
    tempo_bpm: int | None
    bar_count: int
    duration_seconds: float | None
    note_count: int
    pitch_range_low: int | None
    pitch_range_high: int | None
    has_time_sig_changes: bool
    has_tempo_changes: bool
    source: str = "asap"
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd model && uv run pytest tests/score_library/test_schema.py -v`
Expected: All 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add model/src/score_library/ model/tests/score_library/ model/pyproject.toml
git commit -m "feat(score-library): add Pydantic schema for score data models"
```

---

### Task 2: Discovery Module

**Files:**
- Create: `model/src/score_library/discover.py`
- Create: `model/src/score_library/titles.py`
- Create: `model/tests/score_library/test_discover.py`

- [ ] **Step 1: Write failing test for discovery**

```python
# model/tests/score_library/test_discover.py
"""Tests for ASAP score MIDI discovery."""

from pathlib import Path

import pytest

from score_library.discover import discover_pieces, derive_piece_id, PieceEntry


def test_derive_piece_id_3_level(tmp_path):
    """3-level path: Chopin/Etudes_op_10/3 -> chopin.etudes_op_10.3"""
    base = tmp_path / "asap_cache"
    piece_dir = base / "Chopin" / "Etudes_op_10" / "3"
    assert derive_piece_id(piece_dir, base) == "chopin.etudes_op_10.3"


def test_derive_piece_id_2_level(tmp_path):
    """2-level path: Balakirev/Islamey -> balakirev.islamey"""
    base = tmp_path / "asap_cache"
    piece_dir = base / "Balakirev" / "Islamey"
    assert derive_piece_id(piece_dir, base) == "balakirev.islamey"


def test_discover_pieces_finds_both_depths(tmp_path):
    """Discovers pieces at both 2-level and 3-level depths."""
    base = tmp_path / "asap_cache"

    # 3-level piece with 2 score files (should deduplicate)
    d3 = base / "Chopin" / "Etudes_op_10" / "3"
    d3.mkdir(parents=True)
    (d3 / "score_Performer1.mid").write_bytes(b"fake")
    (d3 / "score_Performer2.mid").write_bytes(b"fake")
    (d3 / "performance_Performer1.mid").write_bytes(b"fake")

    # 2-level piece
    d2 = base / "Balakirev" / "Islamey"
    d2.mkdir(parents=True)
    (d2 / "score_SomePerf.mid").write_bytes(b"fake")
    (d2 / "performance_SomePerf.mid").write_bytes(b"fake")

    pieces = discover_pieces(base)
    assert len(pieces) == 2

    ids = {p.piece_id for p in pieces}
    assert "chopin.etudes_op_10.3" in ids
    assert "balakirev.islamey" in ids

    # Verify composer extracted correctly
    by_id = {p.piece_id: p for p in pieces}
    assert by_id["chopin.etudes_op_10.3"].composer == "Chopin"
    assert by_id["balakirev.islamey"].composer == "Balakirev"


def test_discover_pieces_skips_dirs_without_scores(tmp_path):
    """Directories with only performance files are skipped."""
    base = tmp_path / "asap_cache"
    d = base / "Chopin" / "Etudes_op_10" / "3"
    d.mkdir(parents=True)
    (d / "performance_Perf1.mid").write_bytes(b"fake")

    pieces = discover_pieces(base)
    assert len(pieces) == 0


def test_discover_pieces_empty_dir(tmp_path):
    """Empty ASAP dir returns empty list."""
    base = tmp_path / "asap_cache"
    base.mkdir(parents=True)
    pieces = discover_pieces(base)
    assert len(pieces) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd model && uv run pytest tests/score_library/test_discover.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write discovery module**

```python
# model/src/score_library/discover.py
"""Stage 1: Discover ASAP score MIDIs and derive piece metadata.

Walks the ASAP cache directory, finds all directories containing score_*.mid
files, deduplicates to one score per piece, and derives piece_id + composer.

    ASAP directory structure (variable depth):
      2-level: {Composer}/{Piece}/score_{performer}.mid
      3-level: {Composer}/{Collection}/{Number}/score_{performer}.mid

    A "piece directory" is any dir that directly contains score_*.mid files.
    piece_id = relative path under asap_cache, lowercased, dot-separated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from score_library.titles import load_titles, clean_title_from_path

logger = logging.getLogger(__name__)


@dataclass
class PieceEntry:
    piece_id: str
    composer: str
    title: str
    score_midi_path: Path


def derive_piece_id(piece_dir: Path, base_dir: Path) -> str:
    """Derive dot-separated piece_id from directory path."""
    rel = piece_dir.relative_to(base_dir)
    return ".".join(part.lower() for part in rel.parts)


def discover_pieces(
    asap_dir: Path,
    titles_path: Path | None = None,
) -> list[PieceEntry]:
    """Discover all unique pieces in the ASAP cache.

    Returns one PieceEntry per piece directory, using the first
    score_*.mid file found (all are identical in note content).
    """
    asap_dir = Path(asap_dir)
    if not asap_dir.exists():
        raise FileNotFoundError(f"ASAP cache directory not found: {asap_dir}")

    titles = load_titles(titles_path) if titles_path else {}

    # Find all directories that directly contain score_*.mid files
    seen_dirs: set[Path] = set()
    for score_path in sorted(asap_dir.rglob("score_*.mid")):
        seen_dirs.add(score_path.parent)

    entries = []
    for piece_dir in sorted(seen_dirs):
        piece_id = derive_piece_id(piece_dir, asap_dir)
        composer = piece_dir.relative_to(asap_dir).parts[0]

        # Pick first score file (all are identical)
        score_files = sorted(piece_dir.glob("score_*.mid"))
        if not score_files:
            continue

        title = titles.get(piece_id, clean_title_from_path(piece_dir, asap_dir))

        entries.append(PieceEntry(
            piece_id=piece_id,
            composer=composer,
            title=title,
            score_midi_path=score_files[0],
        ))

    logger.info("Discovered %d pieces from %s", len(entries), asap_dir)
    return entries
```

```python
# model/src/score_library/titles.py
"""Title generation and lookup for score library pieces.

Bootstrapping strategy: auto-generate titles algorithmically, then
manually review and fix. The static titles.json mapping overrides
algorithmic generation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


def load_titles(titles_path: Path | None) -> dict[str, str]:
    """Load title mapping from titles.json. Returns empty dict if not found."""
    if titles_path is None or not titles_path.exists():
        return {}
    with open(titles_path) as f:
        return json.load(f)


def clean_title_from_path(piece_dir: Path, base_dir: Path) -> str:
    """Generate a human-readable title from ASAP directory names.

    Applies common transformations:
      - Underscores to spaces
      - op_N -> Op. N
      - bwv_N -> BWV N
      - no_N -> No. N
      - Capitalize words
      - Join collection + number with appropriate separator
    """
    rel = piece_dir.relative_to(base_dir)
    parts = list(rel.parts)

    # Skip composer (first part)
    name_parts = parts[1:]

    # Process each part
    cleaned = []
    for part in name_parts:
        s = part.replace("_", " ")
        # op N -> Op. N
        s = re.sub(r"\bop (\d+)", r"Op. \1", s, flags=re.IGNORECASE)
        # bwv N -> BWV N
        s = re.sub(r"\bbwv (\d+)", r"BWV \1", s, flags=re.IGNORECASE)
        # no N -> No. N (but not at the start of a bare number)
        s = re.sub(r"\bno (\d+)", r"No. \1", s, flags=re.IGNORECASE)
        # Capitalize first letter of each word
        s = " ".join(w.capitalize() if not w[0].isdigit() else w for w in s.split())
        cleaned.append(s)

    if len(cleaned) == 1:
        return cleaned[0]
    elif len(cleaned) == 2:
        # Collection + number: "Etudes Op. 10" + "3" -> "Etude Op. 10 No. 3"
        collection, number = cleaned
        if number.isdigit():
            return f"{collection} No. {number}"
        return f"{collection} - {number}"
    else:
        return " - ".join(cleaned)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd model && uv run pytest tests/score_library/test_discover.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add model/src/score_library/discover.py model/src/score_library/titles.py model/tests/score_library/test_discover.py
git commit -m "feat(score-library): add ASAP discovery with variable-depth path handling"
```

---

### Task 3: Generate titles.json

**Files:**
- Create: `model/data/score_library/titles.json`

- [ ] **Step 1: Write a one-off script to generate initial titles**

```python
# Run interactively: cd model && uv run python -c "
from pathlib import Path
from score_library.discover import discover_pieces
import json

pieces = discover_pieces(Path('data/asap_cache'))
titles = {p.piece_id: p.title for p in pieces}

# Write to data/score_library/titles.json
Path('data/score_library').mkdir(parents=True, exist_ok=True)
with open('data/score_library/titles.json', 'w') as f:
    json.dump(titles, f, indent=2, sort_keys=True)

print(f'Generated {len(titles)} titles')
# Print a sample for review
for pid, title in sorted(titles.items())[:20]:
    print(f'  {pid}: {title}')
# "
```

- [ ] **Step 2: Manually review titles.json, fix the ~20-30 that look wrong**

Open `model/data/score_library/titles.json` and fix entries like:
- `Annees De Pelerinage 2` -> `Annees de Pelerinage, Book 2`
- `Sonata 2 - 1st No Repeat` -> `Sonata No. 2 - 1st Movement`
- Any plural/singular issues
- Any BWV/Op. number formatting issues

- [ ] **Step 3: Commit**

```bash
git add model/data/score_library/titles.json
git commit -m "feat(score-library): add curated titles.json for 242 ASAP pieces"
```

---

## Chunk 2: MIDI Parser (Core)

### Task 4: Bar Grid Builder + Note Parser

**Files:**
- Create: `model/src/score_library/parse.py`
- Create: `model/tests/score_library/test_parse.py`

- [ ] **Step 1: Write failing tests for MIDI parsing**

```python
# model/tests/score_library/test_parse.py
"""Tests for MIDI score parsing.

Tests the bar grid builder, note-to-bar assignment, tick-to-seconds
conversion, pedal extraction, and per-bar summary computation.

    BAR GRID ALGORITHM:
    ┌─────────────────────────────────────────────────────┐
    │  Time sig events -> ticks_per_bar at each change    │
    │  Accumulate bar boundaries: [0, 960, 1920, ...]     │
    │  Assign notes to bars by onset_tick (bisect)        │
    │  Convert ticks to seconds via tempo map             │
    └─────────────────────────────────────────────────────┘
"""

from pathlib import Path

import pytest

from score_library.parse import (
    build_bar_grid,
    assign_notes_to_bars,
    parse_score_midi,
    ticks_to_seconds,
)
from score_library.schema import ScoreData


# --- Bar grid tests ---

def test_build_bar_grid_simple_4_4():
    """4/4 time, 480 ticks/beat -> 1920 ticks/bar."""
    time_sigs = [{"tick": 0, "numerator": 4, "denominator": 4}]
    total_ticks = 1920 * 4  # 4 bars
    ticks_per_beat = 480

    bars = build_bar_grid(time_sigs, total_ticks, ticks_per_beat)
    assert len(bars) == 4
    assert bars[0] == {"bar_number": 1, "start_tick": 0, "ticks_per_bar": 1920, "time_sig": "4/4"}
    assert bars[1]["start_tick"] == 1920
    assert bars[3]["start_tick"] == 5760


def test_build_bar_grid_3_4():
    """3/4 time, 480 ticks/beat -> 1440 ticks/bar."""
    time_sigs = [{"tick": 0, "numerator": 3, "denominator": 4}]
    total_ticks = 1440 * 3
    ticks_per_beat = 480

    bars = build_bar_grid(time_sigs, total_ticks, ticks_per_beat)
    assert len(bars) == 3
    assert bars[0]["ticks_per_bar"] == 1440


def test_build_bar_grid_time_sig_change():
    """Time sig changes mid-piece: 4/4 for 2 bars, then 3/4 for 2 bars."""
    time_sigs = [
        {"tick": 0, "numerator": 4, "denominator": 4},
        {"tick": 3840, "numerator": 3, "denominator": 4},  # After 2 bars of 4/4
    ]
    total_ticks = 3840 + 1440 * 2  # 2 bars of 4/4 + 2 bars of 3/4
    ticks_per_beat = 480

    bars = build_bar_grid(time_sigs, total_ticks, ticks_per_beat)
    assert len(bars) == 4
    assert bars[0]["time_sig"] == "4/4"
    assert bars[1]["time_sig"] == "4/4"
    assert bars[2]["time_sig"] == "3/4"
    assert bars[2]["start_tick"] == 3840
    assert bars[3]["start_tick"] == 3840 + 1440


def test_build_bar_grid_6_8():
    """6/8 time: 6 eighth notes per bar = 3 quarter-note beats."""
    time_sigs = [{"tick": 0, "numerator": 6, "denominator": 8}]
    total_ticks = 1440 * 2  # 2 bars (6 * 240 = 1440 ticks/bar at 480 tpb)
    ticks_per_beat = 480

    bars = build_bar_grid(time_sigs, total_ticks, ticks_per_beat)
    assert len(bars) == 2
    assert bars[0]["ticks_per_bar"] == 1440  # 6 * (480/2)


# --- Tick-to-seconds tests ---

def test_ticks_to_seconds_constant_tempo():
    """120 BPM, 480 ticks/beat -> 0.5s per beat."""
    tempo_map = [{"tick": 0, "tempo": 500000}]  # 120 BPM
    ticks_per_beat = 480
    assert ticks_to_seconds(0, tempo_map, ticks_per_beat) == pytest.approx(0.0)
    assert ticks_to_seconds(480, tempo_map, ticks_per_beat) == pytest.approx(0.5)
    assert ticks_to_seconds(960, tempo_map, ticks_per_beat) == pytest.approx(1.0)


def test_ticks_to_seconds_tempo_change():
    """Tempo changes mid-piece: 120 BPM for 2 beats, then 60 BPM."""
    tempo_map = [
        {"tick": 0, "tempo": 500000},     # 120 BPM
        {"tick": 960, "tempo": 1000000},   # 60 BPM, starts at tick 960
    ]
    ticks_per_beat = 480
    # First 960 ticks at 120 BPM = 1.0s
    assert ticks_to_seconds(960, tempo_map, ticks_per_beat) == pytest.approx(1.0)
    # Next 480 ticks at 60 BPM (1s per beat) = 1.0s
    assert ticks_to_seconds(1440, tempo_map, ticks_per_beat) == pytest.approx(2.0)


# --- Note assignment tests ---

def test_assign_notes_to_bars():
    """Notes are assigned to correct bars by onset tick."""
    bar_grid = [
        {"bar_number": 1, "start_tick": 0, "ticks_per_bar": 1920, "time_sig": "4/4"},
        {"bar_number": 2, "start_tick": 1920, "ticks_per_bar": 1920, "time_sig": "4/4"},
    ]
    notes = [
        {"pitch": 60, "velocity": 80, "onset_tick": 0, "duration_ticks": 480, "track": 0},
        {"pitch": 64, "velocity": 70, "onset_tick": 960, "duration_ticks": 480, "track": 0},
        {"pitch": 67, "velocity": 90, "onset_tick": 1920, "duration_ticks": 480, "track": 1},
    ]
    assigned = assign_notes_to_bars(notes, bar_grid)
    assert len(assigned[1]) == 2  # bar 1 has 2 notes
    assert len(assigned[2]) == 1  # bar 2 has 1 note
    assert assigned[2][0]["pitch"] == 67


# --- Full parse test (requires real MIDI) ---

@pytest.mark.skipif(
    not Path("data/asap_cache/Chopin/Etudes_op_10/3/score_SunMeiting08.mid").exists(),
    reason="ASAP cache not available",
)
def test_parse_chopin_etude():
    """Parse a real score MIDI and validate output schema."""
    midi_path = Path("data/asap_cache/Chopin/Etudes_op_10/3/score_SunMeiting08.mid")
    result = parse_score_midi(
        midi_path,
        piece_id="chopin.etudes_op_10.3",
        composer="Chopin",
        title="Etude Op. 10 No. 3",
    )
    assert isinstance(result, ScoreData)
    assert result.piece_id == "chopin.etudes_op_10.3"
    assert result.total_bars > 0
    assert len(result.bars) == result.total_bars
    assert result.bars[0].bar_number in (0, 1)  # 0 if pickup, 1 otherwise
    # Verify all bars have valid structure
    for bar in result.bars:
        assert bar.note_count == len(bar.notes)
        if bar.notes:
            assert bar.pitch_range[0] <= bar.pitch_range[1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd model && uv run pytest tests/score_library/test_parse.py -v -k "not test_parse_chopin"`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write the MIDI parser**

```python
# model/src/score_library/parse.py
"""Stage 2: Parse score MIDI into bar-centric JSON structure.

    PARSING PIPELINE:
    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
    │  Read MIDI   │ -> │  Build bar   │ -> │ Assign notes │
    │  (mido)      │    │  grid from   │    │ + pedal to   │
    │              │    │  time sigs   │    │ bars         │
    └──────────────┘    └──────────────┘    └──────────────┘
           │                                       │
           v                                       v
    ┌──────────────┐                      ┌──────────────┐
    │ Build tempo  │                      │ Compute bar  │
    │ map for      │ -------------------> │ summaries +  │
    │ tick->secs   │                      │ ScoreData    │
    └──────────────┘                      └──────────────┘
"""

from __future__ import annotations

import bisect
import logging
from pathlib import Path

import mido

from score_library.schema import Bar, PedalEvent, ScoreData, ScoreNote

logger = logging.getLogger(__name__)

# MIDI note number to name mapping
_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _pitch_name(midi_note: int) -> str:
    """Convert MIDI note number to name (e.g., 60 -> 'C4')."""
    octave = (midi_note // 12) - 1
    name = _NOTE_NAMES[midi_note % 12]
    return f"{name}{octave}"


def build_bar_grid(
    time_sigs: list[dict],
    total_ticks: int,
    ticks_per_beat: int,
) -> list[dict]:
    """Build bar boundaries from time signature events.

    Args:
        time_sigs: List of {"tick": int, "numerator": int, "denominator": int}
        total_ticks: Total ticks in the MIDI file
        ticks_per_beat: MIDI ticks per quarter note

    Returns:
        List of {"bar_number": int, "start_tick": int, "ticks_per_bar": int, "time_sig": str}
    """
    if not time_sigs:
        time_sigs = [{"tick": 0, "numerator": 4, "denominator": 4}]

    bars = []
    bar_number = 1
    current_tick = 0
    sig_idx = 0

    while current_tick < total_ticks:
        # Advance to next time sig if we've reached it
        while sig_idx + 1 < len(time_sigs) and time_sigs[sig_idx + 1]["tick"] <= current_tick:
            sig_idx += 1

        sig = time_sigs[sig_idx]
        # ticks_per_bar = numerator * (ticks_per_beat * 4 / denominator)
        ticks_per_bar = int(sig["numerator"] * ticks_per_beat * 4 / sig["denominator"])
        time_sig_str = f"{sig['numerator']}/{sig['denominator']}"

        bars.append({
            "bar_number": bar_number,
            "start_tick": current_tick,
            "ticks_per_bar": ticks_per_bar,
            "time_sig": time_sig_str,
        })
        current_tick += ticks_per_bar
        bar_number += 1

    return bars


def ticks_to_seconds(
    tick: int,
    tempo_map: list[dict],
    ticks_per_beat: int,
) -> float:
    """Convert MIDI tick to seconds using tempo map.

    Args:
        tick: The tick position to convert
        tempo_map: List of {"tick": int, "tempo": int} (microseconds per beat)
        ticks_per_beat: MIDI ticks per quarter note
    """
    if not tempo_map:
        tempo_map = [{"tick": 0, "tempo": 500000}]  # 120 BPM default

    seconds = 0.0
    prev_tick = 0
    prev_tempo = tempo_map[0]["tempo"]

    for entry in tempo_map:
        if entry["tick"] >= tick:
            break
        # Accumulate time from prev_tick to this tempo change
        delta_ticks = entry["tick"] - prev_tick
        seconds += delta_ticks * prev_tempo / (ticks_per_beat * 1_000_000)
        prev_tick = entry["tick"]
        prev_tempo = entry["tempo"]

    # Remaining ticks from last tempo change to target tick
    delta_ticks = tick - prev_tick
    seconds += delta_ticks * prev_tempo / (ticks_per_beat * 1_000_000)
    return seconds


def assign_notes_to_bars(
    notes: list[dict],
    bar_grid: list[dict],
) -> dict[int, list[dict]]:
    """Assign notes to bars by onset tick using binary search.

    Returns: dict mapping bar_number -> list of notes in that bar.
    """
    bar_starts = [b["start_tick"] for b in bar_grid]
    bar_numbers = [b["bar_number"] for b in bar_grid]

    result: dict[int, list[dict]] = {b["bar_number"]: [] for b in bar_grid}

    for note in notes:
        # Find which bar this note belongs to
        idx = bisect.bisect_right(bar_starts, note["onset_tick"]) - 1
        if idx < 0:
            idx = 0
        bar_num = bar_numbers[idx]
        result[bar_num].append(note)

    return result


def parse_score_midi(
    midi_path: Path,
    piece_id: str,
    composer: str,
    title: str,
) -> ScoreData:
    """Parse a score MIDI file into bar-centric ScoreData.

    Args:
        midi_path: Path to the score MIDI file
        piece_id: Canonical piece identifier
        composer: Composer name
        title: Human-readable title
    """
    mid = mido.MidiFile(str(midi_path))
    tpb = mid.ticks_per_beat

    # 1. Extract meta events (time sigs, key sigs, tempo)
    time_sigs = []
    tempo_map = []
    key_signature = None

    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == "time_signature":
                time_sigs.append({
                    "tick": abs_tick,
                    "numerator": msg.numerator,
                    "denominator": msg.denominator,
                })
            elif msg.type == "set_tempo":
                tempo_map.append({"tick": abs_tick, "tempo": msg.tempo})
            elif msg.type == "key_signature" and key_signature is None:
                key_signature = msg.key

    # Sort by tick (events may come from different tracks)
    time_sigs.sort(key=lambda x: x["tick"])
    tempo_map.sort(key=lambda x: x["tick"])

    # 2. Extract notes and pedal events from all tracks
    raw_notes = []
    raw_pedal = []

    for track_idx, track in enumerate(mid.tracks):
        abs_tick = 0
        active_notes: dict[int, dict] = {}  # pitch -> note data

        for msg in track:
            abs_tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                active_notes[msg.note] = {
                    "pitch": msg.note,
                    "velocity": msg.velocity,
                    "onset_tick": abs_tick,
                    "track": track_idx,
                }
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                if msg.note in active_notes:
                    note = active_notes.pop(msg.note)
                    note["duration_ticks"] = abs_tick - note["onset_tick"]
                    raw_notes.append(note)
            elif msg.type == "control_change" and msg.control == 64:
                raw_pedal.append({
                    "type": "on" if msg.value >= 64 else "off",
                    "tick": abs_tick,
                    "track": track_idx,
                })

    # 3. Compute total ticks
    total_ticks = max(
        (n["onset_tick"] + n["duration_ticks"] for n in raw_notes),
        default=0,
    )

    # 4. Build bar grid
    bar_grid = build_bar_grid(time_sigs, total_ticks, tpb)

    # 5. Assign notes to bars
    notes_by_bar = assign_notes_to_bars(raw_notes, bar_grid)

    # 6. Assign pedal events to bars
    pedal_by_bar: dict[int, list] = {b["bar_number"]: [] for b in bar_grid}
    bar_starts = [b["start_tick"] for b in bar_grid]
    bar_numbers = [b["bar_number"] for b in bar_grid]
    for pe in raw_pedal:
        idx = bisect.bisect_right(bar_starts, pe["tick"]) - 1
        if idx < 0:
            idx = 0
        pedal_by_bar[bar_numbers[idx]].append(pe)

    # 7. Build ScoreData
    bars = []
    for bg in bar_grid:
        bn = bg["bar_number"]
        bar_notes = notes_by_bar.get(bn, [])

        score_notes = [
            ScoreNote(
                pitch=n["pitch"],
                pitch_name=_pitch_name(n["pitch"]),
                velocity=n["velocity"],
                onset_tick=n["onset_tick"],
                onset_seconds=round(ticks_to_seconds(n["onset_tick"], tempo_map, tpb), 4),
                duration_ticks=n["duration_ticks"],
                duration_seconds=round(
                    ticks_to_seconds(n["onset_tick"] + n["duration_ticks"], tempo_map, tpb)
                    - ticks_to_seconds(n["onset_tick"], tempo_map, tpb),
                    4,
                ),
                track=n["track"],
            )
            for n in sorted(bar_notes, key=lambda x: (x["onset_tick"], x["pitch"]))
        ]

        pedal_events = [
            PedalEvent(
                type=pe["type"],
                tick=pe["tick"],
                seconds=round(ticks_to_seconds(pe["tick"], tempo_map, tpb), 4),
            )
            for pe in pedal_by_bar.get(bn, [])
        ]

        pitches = [n.pitch for n in score_notes]
        velocities = [n.velocity for n in score_notes]

        bars.append(Bar(
            bar_number=bn,
            start_tick=bg["start_tick"],
            start_seconds=round(ticks_to_seconds(bg["start_tick"], tempo_map, tpb), 4),
            time_signature=bg["time_sig"],
            notes=score_notes,
            pedal_events=pedal_events,
            note_count=len(score_notes),
            pitch_range=[min(pitches, default=0), max(pitches, default=0)],
            mean_velocity=int(sum(velocities) / len(velocities)) if velocities else 0,
        ))

    # Build time_signatures and tempo_markings lists with bar numbers
    ts_with_bars = []
    for ts in time_sigs:
        bar_idx = bisect.bisect_right(bar_starts, ts["tick"]) - 1
        if bar_idx < 0:
            bar_idx = 0
        ts_with_bars.append({
            "bar": bar_numbers[bar_idx],
            "numerator": ts["numerator"],
            "denominator": ts["denominator"],
        })

    tempo_with_bars = []
    for tm in tempo_map:
        bar_idx = bisect.bisect_right(bar_starts, tm["tick"]) - 1
        if bar_idx < 0:
            bar_idx = 0
        tempo_with_bars.append({
            "bar": bar_numbers[bar_idx],
            "bpm": int(round(60_000_000 / tm["tempo"])),
        })

    return ScoreData(
        piece_id=piece_id,
        composer=composer,
        title=title,
        key_signature=key_signature,
        time_signatures=ts_with_bars,
        tempo_markings=tempo_with_bars,
        total_bars=len(bars),
        bars=bars,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd model && uv run pytest tests/score_library/test_parse.py -v -k "not test_parse_chopin"`
Expected: All unit tests PASS

- [ ] **Step 5: Run the real MIDI test**

Run: `cd model && uv run pytest tests/score_library/test_parse.py::test_parse_chopin_etude -v`
Expected: PASS (validates against Pydantic schema)

- [ ] **Step 6: Commit**

```bash
git add model/src/score_library/parse.py model/tests/score_library/test_parse.py
git commit -m "feat(score-library): add MIDI parser with bar grid builder"
```

---

### Task 5: Golden File Tests for Bar Grid Accuracy

**Files:**
- Create: `model/tests/score_library/test_golden.py`
- Create: `model/tests/score_library/golden/` (test fixtures)

- [ ] **Step 1: Identify 5 pieces with known time signature changes**

Run this to find pieces with time sig changes:

```bash
cd model && uv run python -c "
import mido
from pathlib import Path

pieces_with_changes = []
for score_dir in sorted(Path('data/asap_cache').rglob('score_*.mid')):
    # Deduplicate per piece directory
    if any(p[0] == score_dir.parent for p in pieces_with_changes):
        continue
    mid = mido.MidiFile(str(score_dir))
    time_sigs = []
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == 'time_signature':
                time_sigs.append((abs_tick, msg.numerator, msg.denominator))
    unique_sigs = set((n, d) for _, n, d in time_sigs)
    if len(unique_sigs) > 1:
        pieces_with_changes.append((score_dir.parent, time_sigs))
        rel = score_dir.parent.relative_to(Path('data/asap_cache'))
        print(f'{rel}: {time_sigs[:5]}')

print(f'\nTotal pieces with time sig changes: {len(pieces_with_changes)}')
"
```

- [ ] **Step 2: Pick 5 diverse pieces, manually verify bar counts**

For each of the 5 chosen pieces, manually compute the expected bar count from the MIDI time signatures and total ticks. Store as golden test data.

- [ ] **Step 3: Write golden file tests**

```python
# model/tests/score_library/test_golden.py
"""Golden file tests for bar grid accuracy.

These tests verify that the bar grid builder produces exactly correct
bar counts and boundaries for specific pieces with known characteristics.
This is the highest-risk codepath -- wrong bar numbers silently propagate
to all downstream consumers.

Golden data was manually verified against the MIDI files.
"""

from pathlib import Path

import pytest

from score_library.parse import parse_score_midi

ASAP_DIR = Path("data/asap_cache")

# Golden data: (piece_dir_relative, piece_id, expected_bar_count, expected_time_sigs_count)
# Each entry was manually verified.
GOLDEN_PIECES: list[tuple[str, str, int, int]] = [
    # TODO: Fill in after Step 2 identifies pieces.
    # Example format:
    # ("Beethoven/Piano_Sonatas/8-1", "beethoven.piano_sonatas.8-1", 312, 2),
]


@pytest.mark.skipif(not ASAP_DIR.exists(), reason="ASAP cache not available")
@pytest.mark.parametrize("rel_path,piece_id,expected_bars,expected_ts_count", GOLDEN_PIECES)
def test_golden_bar_count(rel_path, piece_id, expected_bars, expected_ts_count):
    """Bar count must match manually verified value exactly."""
    piece_dir = ASAP_DIR / rel_path
    score_files = sorted(piece_dir.glob("score_*.mid"))
    assert score_files, f"No score files in {piece_dir}"

    result = parse_score_midi(score_files[0], piece_id, "Test", "Test Piece")
    assert result.total_bars == expected_bars, (
        f"{piece_id}: expected {expected_bars} bars, got {result.total_bars}"
    )
    assert len(result.time_signatures) == expected_ts_count, (
        f"{piece_id}: expected {expected_ts_count} time sig entries, got {len(result.time_signatures)}"
    )
```

- [ ] **Step 4: Fill in GOLDEN_PIECES with verified data from Step 2 and run**

Run: `cd model && uv run pytest tests/score_library/test_golden.py -v`
Expected: All golden tests PASS

- [ ] **Step 5: Commit**

```bash
git add model/tests/score_library/test_golden.py
git commit -m "test(score-library): add golden file tests for bar grid accuracy"
```

---

## Chunk 3: CLI + Upload + Integration

### Task 6: CLI Entry Point + Stats Command

**Files:**
- Create: `model/src/score_library/cli.py`

- [ ] **Step 1: Write CLI with parse and stats commands**

```python
# model/src/score_library/cli.py
"""CLI entry point for the score library pipeline.

Usage:
    uv run python -m score_library.cli build --asap-dir data/asap_cache
    uv run python -m score_library.cli parse --asap-dir data/asap_cache
    uv run python -m score_library.cli upload --source data/score_library
    uv run python -m score_library.cli stats --source data/score_library
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

from score_library.discover import discover_pieces
from score_library.parse import parse_score_midi
from score_library.schema import PieceCatalogEntry, ScoreData

logger = logging.getLogger(__name__)


def cmd_parse(args: argparse.Namespace) -> None:
    """Discover and parse all ASAP score MIDIs to local JSON."""
    asap_dir = Path(args.asap_dir)
    output_dir = Path(args.output or "data/score_library")
    titles_path = output_dir / "titles.json"

    output_dir.mkdir(parents=True, exist_ok=True)

    pieces = discover_pieces(asap_dir, titles_path if titles_path.exists() else None)
    print(f"Discovered {len(pieces)} pieces")

    successes = 0
    failures = []

    for entry in pieces:
        try:
            score_data = parse_score_midi(
                entry.score_midi_path,
                entry.piece_id,
                entry.composer,
                entry.title,
            )
            out_path = output_dir / f"{entry.piece_id}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(score_data.model_dump(), f, indent=2)
            successes += 1
        except Exception as e:
            failures.append((entry.piece_id, str(e)))
            logger.error("Failed to parse %s: %s", entry.piece_id, e)

    print(f"\nParsed: {successes}/{len(pieces)}")
    if failures:
        print(f"Failures ({len(failures)}):")
        for pid, err in failures:
            print(f"  {pid}: {err}")


def cmd_stats(args: argparse.Namespace) -> None:
    """Print statistics about the parsed score library."""
    source_dir = Path(args.source)
    json_files = sorted(source_dir.glob("*.json"))
    json_files = [f for f in json_files if f.name != "titles.json"]

    if not json_files:
        print(f"No score JSON files found in {source_dir}")
        return

    composers: Counter[str] = Counter()
    bar_counts = []
    note_counts = []
    time_sig_changes = 0
    tempo_changes = 0

    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        composers[data["composer"]] += 1
        bar_counts.append(data["total_bars"])
        total_notes = sum(b["note_count"] for b in data["bars"])
        note_counts.append(total_notes)
        if len(data["time_signatures"]) > 1:
            time_sig_changes += 1
        if len(data["tempo_markings"]) > 1:
            tempo_changes += 1

    bar_counts.sort()
    note_counts.sort()

    print(f"Pieces: {len(json_files)}")
    print(f"\nComposer distribution:")
    for composer, count in sorted(composers.items()):
        print(f"  {composer}: {count}")
    print(f"\nBar count: min={bar_counts[0]}, median={bar_counts[len(bar_counts)//2]}, max={bar_counts[-1]}")
    print(f"Note count: min={note_counts[0]}, median={note_counts[len(note_counts)//2]}, max={note_counts[-1]}")
    print(f"Pieces with time sig changes: {time_sig_changes}")
    print(f"Pieces with tempo changes: {tempo_changes}")


def cmd_upload(args: argparse.Namespace) -> None:
    """Upload parsed JSONs to R2 and seed D1."""
    # Implemented in Task 7
    from score_library.upload import upload_to_r2, generate_d1_seed
    source_dir = Path(args.source)
    upload_to_r2(source_dir, version=args.version)
    generate_d1_seed(source_dir, output_path=Path(args.seed_output or "data/score_library/seed.sql"))


def cmd_build(args: argparse.Namespace) -> None:
    """Full pipeline: parse + upload."""
    cmd_parse(args)
    cmd_upload(args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score MIDI Library pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_parse = sub.add_parser("parse", help="Parse ASAP score MIDIs to local JSON")
    p_parse.add_argument("--asap-dir", required=True, help="Path to ASAP cache directory")
    p_parse.add_argument("--output", help="Output directory (default: data/score_library)")

    p_stats = sub.add_parser("stats", help="Print score library statistics")
    p_stats.add_argument("--source", required=True, help="Path to parsed score library")

    p_upload = sub.add_parser("upload", help="Upload to R2 and seed D1")
    p_upload.add_argument("--source", required=True, help="Path to parsed score library")
    p_upload.add_argument("--version", default="v1", help="Version prefix for R2 path")
    p_upload.add_argument("--seed-output", help="Output path for D1 seed SQL")

    p_build = sub.add_parser("build", help="Full pipeline: parse + upload")
    p_build.add_argument("--asap-dir", required=True, help="Path to ASAP cache directory")
    p_build.add_argument("--output", help="Output directory (default: data/score_library)")
    p_build.add_argument("--version", default="v1", help="Version prefix for R2 path")
    p_build.add_argument("--seed-output", help="Output path for D1 seed SQL")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    {"parse": cmd_parse, "stats": cmd_stats, "upload": cmd_upload, "build": cmd_build}[args.command](args)


if __name__ == "__main__":
    main()
```

Also add `__main__.py` for `python -m score_library.cli`:

```python
# model/src/score_library/__main__.py
from score_library.cli import main

main()
```

- [ ] **Step 2: Smoke test the parse command on real data**

Run: `cd model && uv run python -m score_library.cli parse --asap-dir data/asap_cache`
Expected: `Parsed: 242/242` (or close, with any failures listed)

- [ ] **Step 3: Smoke test the stats command**

Run: `cd model && uv run python -m score_library.cli stats --source data/score_library`
Expected: Composer distribution, bar/note count ranges, time sig/tempo change counts

- [ ] **Step 4: Commit**

```bash
git add model/src/score_library/cli.py model/src/score_library/__main__.py
git commit -m "feat(score-library): add CLI with parse and stats commands"
```

---

### Task 7: Upload Module (R2 + D1 Seed)

**Files:**
- Create: `model/src/score_library/upload.py`
- Modify: `model/pyproject.toml` (add boto3)

- [ ] **Step 1: Add boto3 dependency**

Add `"boto3>=1.28.0"` to the `dependencies` list in `model/pyproject.toml`.

Run: `cd model && uv sync`

- [ ] **Step 2: Write the upload module**

```python
# model/src/score_library/upload.py
"""Stage 3: Upload parsed scores to R2 and generate D1 seed SQL.

R2 upload uses the S3-compatible API via boto3.
D1 seeding generates a SQL file to be run via wrangler.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_d1_seed(source_dir: Path, output_path: Path) -> None:
    """Generate D1 seed SQL from parsed score library JSON files."""
    json_files = sorted(source_dir.glob("*.json"))
    json_files = [f for f in json_files if f.name != "titles.json" and f.name != "seed.sql"]

    rows = []
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)

        all_notes = sum(b["note_count"] for b in data["bars"])
        all_pitches = [n["pitch"] for b in data["bars"] for n in b["notes"]]

        has_ts_changes = 1 if len(data["time_signatures"]) > 1 else 0
        has_tempo_changes = 1 if len(data["tempo_markings"]) > 1 else 0

        ts = data["time_signatures"][0] if data["time_signatures"] else None
        tempo = data["tempo_markings"][0] if data["tempo_markings"] else None

        # Escape single quotes in title
        title = data["title"].replace("'", "''")
        key_sig = data.get("key_signature") or ""
        time_sig = f"{ts['numerator']}/{ts['denominator']}" if ts else ""
        tempo_bpm = str(tempo["bpm"]) if tempo else "NULL"
        pitch_low = str(min(all_pitches)) if all_pitches else "NULL"
        pitch_high = str(max(all_pitches)) if all_pitches else "NULL"

        # Compute duration as last note end time, not last bar start
        last_notes = [n for b in data["bars"] for n in b["notes"]]
        if last_notes:
            last = max(last_notes, key=lambda n: n["onset_seconds"] + n["duration_seconds"])
            duration = last["onset_seconds"] + last["duration_seconds"]
        else:
            duration = 0.0

        rows.append(
            f"('{data['piece_id']}', '{data['composer']}', '{title}', "
            f"'{key_sig}', '{time_sig}', {tempo_bpm}, "
            f"{data['total_bars']}, {duration:.1f}, {all_notes}, "
            f"{pitch_low}, {pitch_high}, "
            f"{has_ts_changes}, {has_tempo_changes}, 'asap')"
        )

    sql = "INSERT OR REPLACE INTO pieces (piece_id, composer, title, key_signature, time_signature, tempo_bpm, bar_count, duration_seconds, note_count, pitch_range_low, pitch_range_high, has_time_sig_changes, has_tempo_changes, source) VALUES\n"
    sql += ",\n".join(rows) + ";\n"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(sql)

    logger.info("Generated D1 seed SQL with %d rows at %s", len(rows), output_path)
    print(f"Generated seed SQL: {len(rows)} rows -> {output_path}")


def upload_to_r2(source_dir: Path, version: str = "v1") -> None:
    """Upload parsed score JSONs to R2 via S3-compatible API.

    Requires environment variables:
        R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY
    """
    import boto3

    account_id = os.environ.get("R2_ACCOUNT_ID")
    access_key = os.environ.get("R2_ACCESS_KEY_ID")
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")

    if not all([account_id, access_key, secret_key]):
        raise RuntimeError(
            "R2 credentials not set. Export R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY."
        )

    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
    )

    bucket = "crescendai-bucket"
    json_files = sorted(source_dir.glob("*.json"))
    json_files = [f for f in json_files if f.name != "titles.json" and f.name != "seed.sql"]

    uploaded = 0
    for jf in json_files:
        piece_id = jf.stem
        key = f"scores/{version}/{piece_id}.json"
        s3.upload_file(str(jf), bucket, key, ExtraArgs={"ContentType": "application/json"})
        uploaded += 1

    print(f"Uploaded {uploaded} score files to R2 (scores/{version}/)")

    if uploaded != len(json_files):
        raise RuntimeError(f"Upload count mismatch: {uploaded} uploaded vs {len(json_files)} expected")
```

- [ ] **Step 3: Test D1 seed generation locally**

Run: `cd model && uv run python -m score_library.cli upload --source data/score_library --seed-output data/score_library/seed.sql`
Expected: `Generated seed SQL: 242 rows` (R2 upload will fail without credentials -- that's expected)

Note: The R2 upload requires credentials. Test locally with just the D1 seed generation. R2 upload is tested during deployment.

- [ ] **Step 4: Commit**

```bash
git add model/src/score_library/upload.py model/pyproject.toml
git commit -m "feat(score-library): add R2 upload and D1 seed SQL generation"
```

---

### Task 8: Integration Test

**Files:**
- Create: `model/tests/score_library/test_integration.py`

- [ ] **Step 1: Write integration test**

```python
# model/tests/score_library/test_integration.py
"""Integration test: run full parse pipeline on actual ASAP cache.

Validates that all 242 pieces parse successfully and output conforms
to the Pydantic schema.
"""

import json
from pathlib import Path

import pytest

from score_library.discover import discover_pieces
from score_library.parse import parse_score_midi
from score_library.schema import ScoreData

ASAP_DIR = Path("data/asap_cache")


@pytest.mark.skipif(not ASAP_DIR.exists(), reason="ASAP cache not available")
def test_full_pipeline_all_pieces():
    """All 242 ASAP pieces parse without errors."""
    pieces = discover_pieces(ASAP_DIR)
    assert len(pieces) >= 240, f"Expected ~242 pieces, got {len(pieces)}"

    failures = []
    for entry in pieces:
        try:
            result = parse_score_midi(
                entry.score_midi_path, entry.piece_id, entry.composer, entry.title,
            )
            # Validate schema by serializing
            data = result.model_dump()
            assert data["total_bars"] > 0
            assert len(data["bars"]) == data["total_bars"]

            # Validate JSON size is reasonable
            json_str = json.dumps(data)
            size_kb = len(json_str) / 1024
            assert size_kb < 500, f"{entry.piece_id}: JSON too large ({size_kb:.0f}KB)"

        except Exception as e:
            failures.append((entry.piece_id, str(e)))

    if failures:
        msg = f"{len(failures)} pieces failed:\n"
        for pid, err in failures:
            msg += f"  {pid}: {err}\n"
        pytest.fail(msg)


@pytest.mark.skipif(not ASAP_DIR.exists(), reason="ASAP cache not available")
def test_spot_check_bar_consistency():
    """Spot-check: every bar has a valid bar_number and non-negative start_tick."""
    pieces = discover_pieces(ASAP_DIR)

    for entry in pieces[:20]:  # Spot-check first 20
        result = parse_score_midi(
            entry.score_midi_path, entry.piece_id, entry.composer, entry.title,
        )
        prev_tick = -1
        for bar in result.bars:
            assert bar.start_tick >= prev_tick, (
                f"{entry.piece_id} bar {bar.bar_number}: start_tick {bar.start_tick} < prev {prev_tick}"
            )
            assert bar.note_count == len(bar.notes)
            prev_tick = bar.start_tick
```

- [ ] **Step 2: Run integration test**

Run: `cd model && uv run pytest tests/score_library/test_integration.py -v --timeout=120`
Expected: Both tests PASS, all 242 pieces parse

- [ ] **Step 3: Commit**

```bash
git add model/tests/score_library/test_integration.py
git commit -m "test(score-library): add integration test for full ASAP pipeline"
```

---

## Chunk 4: API Worker (Rust)

### Task 9: D1 Migration + Wrangler Config

**Files:**
- Create: `apps/api/migrations/0003_pieces.sql`
- Modify: `apps/api/wrangler.toml`

- [ ] **Step 1: Write D1 migration**

```sql
-- apps/api/migrations/0003_pieces.sql
-- Score MIDI Library: piece catalog table
-- Design spec: docs/superpowers/specs/2026-03-14-score-midi-library-design.md

CREATE TABLE IF NOT EXISTS pieces (
  piece_id TEXT PRIMARY KEY,
  composer TEXT NOT NULL,
  title TEXT NOT NULL,
  key_signature TEXT,
  time_signature TEXT,
  tempo_bpm INTEGER,
  bar_count INTEGER NOT NULL,
  duration_seconds REAL,
  note_count INTEGER NOT NULL,
  pitch_range_low INTEGER,
  pitch_range_high INTEGER,
  has_time_sig_changes INTEGER NOT NULL DEFAULT 0,
  has_tempo_changes INTEGER NOT NULL DEFAULT 0,
  source TEXT NOT NULL DEFAULT 'asap',
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_pieces_composer ON pieces(composer);
```

- [ ] **Step 2: Add SCORES R2 binding to wrangler.toml**

Add after the existing `[[r2_buckets]]` block:

```toml
[[r2_buckets]]
binding = "SCORES"
bucket_name = "crescendai-bucket"
```

- [ ] **Step 3: Apply migration locally**

Run: `cd apps/api && npx wrangler d1 migrations apply crescendai-db --local`
Expected: Migration 0003 applied successfully

- [ ] **Step 4: Commit**

```bash
git add apps/api/migrations/0003_pieces.sql apps/api/wrangler.toml
git commit -m "feat(api): add pieces table migration and SCORES R2 binding"
```

---

### Task 10: Score API Endpoints (Rust)

**Files:**
- Create: `apps/api/src/services/scores.rs`
- Modify: `apps/api/src/services/mod.rs`
- Modify: `apps/api/src/server.rs`

- [ ] **Step 1: Write the scores service**

```rust
// apps/api/src/services/scores.rs
//! Score library API handlers.
//!
//! GET /api/scores/:piece_id      -> D1 piece catalog lookup
//! GET /api/scores/:piece_id/data -> R2 full score data (cached)
//! GET /api/scores?composer=X     -> D1 filtered list

use worker::Env;

/// Fetch piece catalog entry from D1.
pub async fn handle_get_piece(
    env: &Env,
    piece_id: &str,
) -> http::Response<axum::body::Body> {
    let db = match env.d1("DB") {
        Ok(db) => db,
        Err(_) => return json_error(500, "Database unavailable"),
    };

    let stmt = db
        .prepare("SELECT piece_id, composer, title, key_signature, time_signature, tempo_bpm, bar_count, duration_seconds, note_count, pitch_range_low, pitch_range_high, has_time_sig_changes, has_tempo_changes, source FROM pieces WHERE piece_id = ?1")
        .bind(&[piece_id.into()])
        .unwrap();

    match stmt.first::<worker::d1::D1Result>(None).await {
        Ok(Some(row)) => {
            let json = row.to_json().unwrap_or_default();
            http::Response::builder()
                .status(200)
                .header("Content-Type", "application/json")
                .body(axum::body::Body::from(json))
                .unwrap()
        }
        Ok(None) => json_error(404, "piece_not_found"),
        Err(e) => {
            worker::console_error!("D1 query failed for piece {}: {:?}", piece_id, e);
            json_error(500, "database_error")
        }
    }
}

/// Fetch full score data from R2 with caching.
pub async fn handle_get_piece_data(
    env: &Env,
    piece_id: &str,
) -> http::Response<axum::body::Body> {
    let bucket = match env.bucket("SCORES") {
        Ok(b) => b,
        Err(_) => return json_error(500, "Storage unavailable"),
    };

    let key = format!("scores/v1/{}.json", piece_id);
    match bucket.get(&key).execute().await {
        Ok(Some(obj)) => {
            let bytes = obj.body().unwrap().bytes().await.unwrap_or_default();
            http::Response::builder()
                .status(200)
                .header("Content-Type", "application/json")
                .header("Cache-Control", "public, max-age=31536000, immutable")
                .body(axum::body::Body::from(bytes))
                .unwrap()
        }
        Ok(None) => json_error(404, "piece_not_found"),
        Err(e) => {
            worker::console_error!("R2 fetch failed for {}: {:?}", key, e);
            http::Response::builder()
                .status(503)
                .header("Content-Type", "application/json")
                .header("Retry-After", "5")
                .body(axum::body::Body::from(r#"{"error":"storage_unavailable"}"#))
                .unwrap()
        }
    }
}

/// List pieces, optionally filtered by composer.
pub async fn handle_list_pieces(
    env: &Env,
    composer: Option<&str>,
) -> http::Response<axum::body::Body> {
    let db = match env.d1("DB") {
        Ok(db) => db,
        Err(_) => return json_error(500, "Database unavailable"),
    };

    let result = if let Some(c) = composer {
        db.prepare("SELECT piece_id, composer, title, key_signature, bar_count, note_count FROM pieces WHERE composer = ?1 ORDER BY title")
            .bind(&[c.into()])
            .unwrap()
            .all()
            .await
    } else {
        db.prepare("SELECT piece_id, composer, title, key_signature, bar_count, note_count FROM pieces ORDER BY composer, title")
            .all()
            .await
    };

    match result {
        Ok(rows) => {
            let json = rows.to_json().unwrap_or_default();
            http::Response::builder()
                .status(200)
                .header("Content-Type", "application/json")
                .body(axum::body::Body::from(json))
                .unwrap()
        }
        Err(e) => {
            worker::console_error!("D1 list query failed: {:?}", e);
            json_error(500, "database_error")
        }
    }
}

fn json_error(status: u16, message: &str) -> http::Response<axum::body::Body> {
    http::Response::builder()
        .status(status)
        .header("Content-Type", "application/json")
        .body(axum::body::Body::from(format!(r#"{{"error":"{}"}}"#, message)))
        .unwrap()
}
```

- [ ] **Step 2: Register module in mod.rs**

Add `pub mod scores;` to `apps/api/src/services/mod.rs`.

- [ ] **Step 3: Add routes to server.rs**

Add these route handlers in `server.rs` before the final 404 fallback. Follow the existing pattern of `path.starts_with()` + `trim_start_matches()`:

```rust
// Score library: GET /api/scores/:piece_id/data (must come before /api/scores/:piece_id)
if path.starts_with("/api/scores/") && path.ends_with("/data") && method == http::Method::GET {
    let piece_id = path
        .trim_start_matches("/api/scores/")
        .trim_end_matches("/data");
    if !piece_id.is_empty() && !piece_id.contains('/') {
        return into_worker_response(with_cors(
            crate::services::scores::handle_get_piece_data(&env, piece_id).await,
            origin.as_deref(),
        )).await;
    }
}

// Score library: GET /api/scores/:piece_id
if path.starts_with("/api/scores/") && method == http::Method::GET {
    let piece_id = path.trim_start_matches("/api/scores/");
    if !piece_id.is_empty() && !piece_id.contains('/') {
        return into_worker_response(with_cors(
            crate::services::scores::handle_get_piece(&env, piece_id).await,
            origin.as_deref(),
        )).await;
    }
}

// Score library: GET /api/scores?composer=X
if path == "/api/scores" && method == http::Method::GET {
    let query = req.uri().query().unwrap_or("");
    let composer = query
        .split('&')
        .find_map(|pair| {
            let mut parts = pair.splitn(2, '=');
            if parts.next() == Some("composer") {
                parts.next().map(|v| urlencoding::decode(v).unwrap_or_default().into_owned())
            } else {
                None
            }
        });
    return into_worker_response(with_cors(
        crate::services::scores::handle_list_pieces(&env, composer.as_deref()).await,
        origin.as_deref(),
    )).await;
}
```

- [ ] **Step 4: Verify it compiles**

Run: `cd apps/api && cargo check`
Expected: Compiles without errors

Note: The D1 and R2 API usage may need adaptation to the exact `worker` crate version used in the project. Check existing handlers (e.g., `ask.rs`, `sync.rs`) for the exact D1/R2 patterns and match them.

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/services/scores.rs apps/api/src/services/mod.rs apps/api/src/server.rs
git commit -m "feat(api): add score library GET endpoints (D1 catalog + R2 data)"
```

---

### Task 11: Deploy and Seed

- [ ] **Step 1: Apply D1 migration to production**

Run: `cd apps/api && npx wrangler d1 migrations apply crescendai-db --remote`

- [ ] **Step 2: Run the parse pipeline locally**

Run: `cd model && uv run python -m score_library.cli parse --asap-dir data/asap_cache`
Expected: 242/242 parsed

- [ ] **Step 3: Upload score JSONs to R2**

Set R2 credentials, then run:
```bash
cd model && uv run python -m score_library.cli upload --source data/score_library --seed-output data/score_library/seed.sql
```

- [ ] **Step 4: Seed D1 with piece catalog**

Run: `cd apps/api && npx wrangler d1 execute crescendai-db --remote --file=../../model/data/score_library/seed.sql`

- [ ] **Step 5: Deploy the worker**

Run: `cd apps/api && npx wrangler deploy`

- [ ] **Step 6: Verify endpoints**

```bash
# Piece lookup
curl https://api.crescend.ai/api/scores/chopin.etudes_op_10.3

# Full score data
curl https://api.crescend.ai/api/scores/chopin.etudes_op_10.3/data | head -c 500

# List by composer
curl "https://api.crescend.ai/api/scores?composer=Chopin"

# 404 for unknown piece
curl https://api.crescend.ai/api/scores/nonexistent.piece
```

- [ ] **Step 7: Commit any deployment fixes**

```bash
git commit -m "deploy: seed D1 and upload score library to R2"
```
