# Catalog-Expansion Implementation Plan

**Goal:** Add schema-valid score JSONs for the 11 missing `practice_eval` pieces so all 16 labeled evaluation pieces have a catalog entry, regenerate the fingerprint index (244 -> 255), and commit a `slug -> piece_id` map (`eval_piece_map.json`) the #21 chroma harness consumes. New machinery: an independent, chroma-free validation gate (`validate.py`), a ranked-URL + lockfile ingestion driver (`manual.py`), a coverage/quality acceptance harness (`catalog_coverage.py`), and a `parse-manual` CLI subcommand.

**Spec path:** `docs/specs/2026-06-02-catalog-expansion-design.md`

**Style / Constraints:**
- Python via `uv` (never pip). Run tests with `cd model && uv run python -m pytest tests/score_library/test_X.py -v`.
- `partitura` (CPJKU) is the symbolic standard in this repo; never `music21`. (Not needed in this feature — all MIDI work uses `mido`, which is already a dependency and used by `parse.py`.)
- `mido` is fine for building MIDI fixtures and reading MIDI.
- Fetch uses stdlib `urllib.request` ONLY (`requests`/`httpx` are not dependencies).
- Explicit exceptions over silent fallbacks: an all-sources-fail piece raises `SourceResolutionError` and HALTs the run; no partial-catalog commit.
- No emojis anywhere. No backup files.
- Tests verify behavior through public interfaces only (`validate_score`, `ingest_manifest`, `check_coverage`, `cmd_parse_manual`). No mocking of internal collaborators. `fetch_fn` is dependency-injected at the public boundary because fetch is a genuine external dependency, not an internal collaborator.
- **NO test asserts chroma rank.** The gate is chroma-independent by design (gating on chroma self-recognition would rig #21).
- The package is installed editable, so imports are `from score_library.X import ...` and `from src.paths import Scores` (matches existing `cli.py`).
- `parse.py` and the JSON schema are UNCHANGED. The quantization check recovers the 16th-note grid from `bar.start_tick` deltas + `time_signature` (no `ticks_per_beat` field exists in the output, and none is added).

---

## File Structure

| File | Responsibility (deep module: narrow interface, hidden complexity) | New/Mod |
|------|-------------------------------------------------------------------|---------|
| `model/src/score_library/validate.py` | **DEEP, pure, no I/O.** Public: `validate_score(score, expected) -> list[Violation]`, `ExpectedMeta`, `Violation`. Hides: DoD-minimum + pitch-range logic, bar-count tolerance band, 16th-grid recovery + median-deviation, Krumhansl-Schmuckler key-profile correlation + enharmonic tonic mapping. | New |
| `model/src/score_library/manual.py` | **DEEP.** Public: `ingest_manifest(manifest_path, scores_dir, lock_path, fetch_fn=_http_fetch) -> IngestReport`, `IngestReport`, `SourceResolutionError`, `_http_fetch`. Hides: urllib fetch, sha256, ranked-source iteration, lockfile read/write, temp-file lifecycle, wiring `parse_score_midi` -> `validate_score`, HALT-on-all-fail with per-candidate failure table. | New |
| `model/src/score_library/catalog_coverage.py` | **DEEP-ish (acceptance harness).** Public: `check_coverage(scores_dir, mapping) -> list[str]`, `CANONICAL_MAP`. Hides: per-piece existence + DoD-minimum (>=20 notes, total_bars>=1, monotonic onsets) checks. Returns `[]` when all pass. | New |
| `model/src/score_library/cli.py` | Shallow glue: `cmd_parse_manual(args)` -> `ingest_manifest`; `parse-manual` subparser; dispatch registration. `cmd_fingerprint` UNCHANGED. | Modify |
| `model/data/manifests/manual_scores.json` | 11-piece ranked-URL manifest (intent). | New (Group D) |
| `model/data/manifests/manual_scores.lock.json` | Build-written resolution (`{piece_id: {resolved_url, sha256}}`). | New (Group D, generated) |
| `model/data/evals/piece_id/eval_piece_map.json` | 16-entry `slug -> piece_id` (the #21 contract). | New (Group D) |
| `model/data/scores/{11 piece_id}.json` | Ingested score JSONs. | New (Group D, generated) |
| `model/data/fingerprints/ngram_index.json` + `rerank_features.json` | Regenerated over 255 pieces. | Modify (Group D, generated) |
| `model/tests/score_library/test_validate.py` | Tests for the gate (Group A). | New |
| `model/tests/score_library/test_manual.py` | Tests for the driver (Group B). | New |
| `model/tests/score_library/test_catalog_coverage.py` | Tests for the harness (Group 0). | New |
| `justfile` | `catalog-verify` recipe. | Modify (Group D) |

---

## Task Groups

- **Group 0 — `catalog_coverage.py` harness.** One task (G0-1). Independent file. **[SHIPS INDEPENDENTLY]** as tested machinery. Parallel-eligible with Group A.
- **Group A — `validate.py` gate.** Sequential A1->A5 (same file). **[SHIPS INDEPENDENTLY]** as tested machinery. Parallel-eligible with Group 0.
  - A1 DoD-minimums, A2 pitch-range, A3 bar-count, A4 quantization, A5 key-agreement.
- **Group B — `manual.py` driver.** Depends on Group A. Sequential B1->B4 (same file).
  - B1 happy-path, B2 hash-mismatch, B3 ranked-fallback, B4 all-fail HALT.
- **Group C — `cli.py` subcommand.** Depends on Group B. One task (C1).
- **Group D — execution / data tasks.** Depends on Groups A/B/C. **NOT red-green TDD** — these are data-authoring + run-the-machinery tasks verified by the Group-0 harness (`just catalog-verify` GREEN) and the unit suites. D1 author manifest, D2 run `parse-manual` (writes 11 scores + lockfile), D3 fingerprint + `eval_piece_map.json` + `catalog-verify` recipe + commit.
  - **KNOWN AUTOMATION RISK:** if a piece cannot be auto-sourced from any candidate URL with metric (non-performance) timing in the correct key, the run HALTs loudly with a per-candidate failure table (`SourceResolutionError`). The executor must complete all machinery and all resolvable pieces, then surface unresolved pieces with their failure table — DO NOT fabricate URLs, DO NOT commit partial coverage, DO NOT weaken the gate to force a pass.

---

## Group 0

### Task G0-1 — `check_coverage` + `CANONICAL_MAP` (catalog_coverage.py) [SHIPS INDEPENDENTLY]

**Step 1 — Write the failing test.**
Create `model/tests/score_library/test_catalog_coverage.py`:

```python
"""Tests for the catalog coverage / quality acceptance harness."""

from __future__ import annotations

import json
from pathlib import Path

from score_library.catalog_coverage import CANONICAL_MAP, check_coverage


def _write_score(path: Path, piece_id: str, bars: list[list[float]]) -> None:
    """Write a minimal schema-shaped score JSON.

    bars: list of bars, each a list of onset_seconds for that bar's notes.
    """
    bar_models = []
    for i, onsets in enumerate(bars, start=1):
        notes = [
            {
                "pitch": 60,
                "pitch_name": "C4",
                "velocity": 80,
                "onset_tick": int(o * 1000),
                "onset_seconds": o,
                "duration_ticks": 240,
                "duration_seconds": 0.25,
                "track": 0,
            }
            for o in onsets
        ]
        bar_models.append(
            {
                "bar_number": i,
                "start_tick": (i - 1) * 1920,
                "start_seconds": float(i - 1),
                "time_signature": "4/4",
                "notes": notes,
                "pedal_events": [],
                "note_count": len(notes),
                "pitch_range": [60, 60] if notes else [],
                "mean_velocity": 80 if notes else 0,
            }
        )
    data = {
        "piece_id": piece_id,
        "composer": "Test",
        "title": "Test",
        "key_signature": None,
        "time_signatures": [{"tick": 0, "numerator": 4, "denominator": 4}],
        "tempo_markings": [],
        "total_bars": len(bar_models),
        "bars": bar_models,
    }
    path.write_text(json.dumps(data))


def _good_onsets() -> list[list[float]]:
    """3 bars, 24 monotonic onsets total (>= 20)."""
    return [
        [i * 0.25 for i in range(8)],
        [2.0 + i * 0.25 for i in range(8)],
        [4.0 + i * 0.25 for i in range(8)],
    ]


class TestCheckCoverage:
    def test_canonical_map_has_16_entries(self) -> None:
        assert len(CANONICAL_MAP) == 16

    def test_all_present_and_good_returns_empty(self, tmp_path: Path) -> None:
        mapping = {"slug_a": "good.piece"}
        _write_score(tmp_path / "good.piece.json", "good.piece", _good_onsets())
        assert check_coverage(tmp_path, mapping) == []

    def test_missing_file_reported(self, tmp_path: Path) -> None:
        mapping = {"slug_a": "absent.piece"}
        result = check_coverage(tmp_path, mapping)
        assert result == ["slug_a: MISSING absent.piece.json"]

    def test_too_few_notes_reported(self, tmp_path: Path) -> None:
        mapping = {"slug_a": "thin.piece"}
        _write_score(tmp_path / "thin.piece.json", "thin.piece", [[0.0, 0.25, 0.5]])
        result = check_coverage(tmp_path, mapping)
        assert len(result) == 1
        assert "thin.piece" in result[0]
        assert "20" in result[0]

    def test_zero_bars_reported(self, tmp_path: Path) -> None:
        mapping = {"slug_a": "empty.piece"}
        _write_score(tmp_path / "empty.piece.json", "empty.piece", [])
        result = check_coverage(tmp_path, mapping)
        assert any("total_bars" in r for r in result)

    def test_non_monotonic_onsets_reported(self, tmp_path: Path) -> None:
        mapping = {"slug_a": "jumbled.piece"}
        # 24 notes but bar 2 onsets dip below bar 1's last -> non-monotonic flat list
        bars = [
            [i * 0.25 for i in range(8)],   # 0.0 .. 1.75
            [0.5 + i * 0.25 for i in range(8)],  # restarts at 0.5 < 1.75
            [4.0 + i * 0.25 for i in range(8)],
        ]
        _write_score(tmp_path / "jumbled.piece.json", "jumbled.piece", bars)
        result = check_coverage(tmp_path, mapping)
        assert any("monotonic" in r for r in result)
```

**Step 2 — Run, verify FAIL.**
```
cd model && uv run python -m pytest tests/score_library/test_catalog_coverage.py -v
```
Expected: collection error / `ModuleNotFoundError: No module named 'score_library.catalog_coverage'` (the module does not exist yet).

**Step 3 — Implement.**
Create `model/src/score_library/catalog_coverage.py`:

```python
"""Catalog coverage and quality acceptance harness.

check_coverage verifies that every (slug -> piece_id) in a mapping resolves to
an existing, non-trivial, monotonic score JSON. It is TDD'd against tmp_path
fixtures and RUN against the real catalog as the feature's acceptance gate.
No chroma logic anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path

#: Canonical 16-entry slug -> piece_id map (the #21 chroma-harness contract).
CANONICAL_MAP: dict[str, str] = {
    "bach_invention_1": "bach.inventions.1",
    "bach_prelude_c_wtc1": "bach.prelude.bwv_846",
    "chopin_ballade_1": "chopin.ballades.1",
    "chopin_etude_op10no4": "chopin.etudes_op_10.4",
    "chopin_waltz_csm": "chopin.waltzes.64-2",
    "clair_de_lune": "debussy.suite_bergamasque.3_clair_de_lune",
    "debussy_arabesque_1": "debussy.deux_arabesques.1",
    "fantaisie_impromptu": "chopin.fantaisie_impromptu",
    "fur_elise": "beethoven.fur_elise",
    "liszt_liebestraum_3": "liszt.liebestraume.3",
    "moonlight_sonata_mvt1": "beethoven.piano_sonatas.14-1",
    "mozart_k545_mvt1": "mozart.piano_sonatas.16-1",
    "nocturne_op9no2": "chopin.nocturnes.9-2",
    "pathetique_mvt2": "beethoven.piano_sonatas.8-2",
    "rachmaninoff_prelude_csm": "rachmaninoff.preludes_op_3.2",
    "schumann_traumerei": "schumann.kinderszenen.7",
}


def check_coverage(scores_dir: Path, mapping: dict[str, str]) -> list[str]:
    """Verify each (slug, piece_id) resolves to a non-trivial, monotonic score.

    For each entry: if {piece_id}.json is missing, append a MISSING line.
    Otherwise load it, flatten all note onsets across bars, and append failure
    strings for < 20 notes, total_bars < 1, or non-monotonic onsets.

    Returns an empty list when every entry passes.
    """
    failures: list[str] = []
    for slug, piece_id in mapping.items():
        score_path = scores_dir / f"{piece_id}.json"
        if not score_path.exists():
            failures.append(f"{slug}: MISSING {piece_id}.json")
            continue

        with open(score_path) as f:
            data = json.load(f)

        onsets: list[float] = []
        for bar in data.get("bars", []):
            for note in bar.get("notes", []):
                onsets.append(note["onset_seconds"])

        if len(onsets) < 20:
            failures.append(f"{slug}: {piece_id} has {len(onsets)} notes (< 20 minimum)")
        if data.get("total_bars", 0) < 1:
            failures.append(f"{slug}: {piece_id} has total_bars < 1")
        if any(onsets[i] < onsets[i - 1] for i in range(1, len(onsets))):
            failures.append(f"{slug}: {piece_id} onsets are not monotonic")

    return failures
```

**Step 4 — Run, verify PASS.**
```
cd model && uv run python -m pytest tests/score_library/test_catalog_coverage.py -v
```
Expected: all 6 tests pass.

**Step 5 — Commit.**
```
git add model/src/score_library/catalog_coverage.py model/tests/score_library/test_catalog_coverage.py && git commit -m "feat(score-library): add catalog coverage acceptance harness"
```

---

## Group A — validate.py (sequential A1->A5)

> Each A-task adds ONE check to the SAME file `validate.py`. The dataclasses `ExpectedMeta` and `Violation` are introduced in A1 and unchanged thereafter. The fixture builder `_make_score(...)` lives in `test_validate.py` and is introduced in A1, reused (already present) by A2-A5; each task ADDS its own test methods to the existing test file.

### Task A1 — DoD-minimums (validate.py) [SHIPS INDEPENDENTLY]

`validate_score` flattens notes across bars and emits: `Violation("min_notes", ...)` if total notes < `expected.min_notes`; `Violation("total_bars", ...)` if `score.total_bars < 1`; `Violation("monotonic_onsets", ...)` if the flat `onset_seconds` list is not non-decreasing.

**Step 1 — Write the failing test.**
Create `model/tests/score_library/test_validate.py`:

```python
"""Tests for the chroma-independent validation gate."""

from __future__ import annotations

from score_library.schema import Bar, ScoreData, ScoreNote
from score_library.validate import ExpectedMeta, Violation, validate_score


def _note(pitch: int, onset_seconds: float, onset_tick: int) -> ScoreNote:
    return ScoreNote(
        pitch=pitch,
        pitch_name="X",
        velocity=80,
        onset_tick=onset_tick,
        onset_seconds=onset_seconds,
        duration_ticks=240,
        duration_seconds=0.25,
        track=0,
    )


def _bar(
    bar_number: int,
    start_tick: int,
    notes: list[ScoreNote],
    time_signature: str = "4/4",
) -> Bar:
    pitches = [n.pitch for n in notes]
    return Bar(
        bar_number=bar_number,
        start_tick=start_tick,
        start_seconds=float(bar_number - 1),
        time_signature=time_signature,
        notes=notes,
        pedal_events=[],
        note_count=len(notes),
        pitch_range=[min(pitches), max(pitches)] if pitches else [],
        mean_velocity=80 if notes else 0,
    )


def _make_score(bars: list[Bar], piece_id: str = "test.piece") -> ScoreData:
    return ScoreData(
        piece_id=piece_id,
        composer="Test",
        title="Test",
        key_signature=None,
        time_signatures=[{"tick": 0, "numerator": 4, "denominator": 4}],
        tempo_markings=[],
        total_bars=len(bars),
        bars=bars,
    )


def _c_major_clean_bars(n_bars: int = 3, ppb: int = 480) -> list[Bar]:
    """n_bars of clean 4/4 16th-grid C-major scale notes, monotonic, >= 20 notes.

    16th = ppb/4 ticks. Bar length = ppb*4 ticks. 8 sixteenths per bar (every
    other 16th) keeps onsets exactly on grid.
    """
    bar_ticks = ppb * 4
    sixteenth = ppb // 4
    c_major = [60, 62, 64, 65, 67, 69, 71, 72]
    bars: list[Bar] = []
    for b in range(n_bars):
        notes = []
        for i, pitch in enumerate(c_major):
            tick = b * bar_ticks + i * (2 * sixteenth)
            notes.append(_note(pitch, tick / 1000.0, tick))
        bars.append(_bar(b + 1, b * bar_ticks, notes))
    return bars


def _expected(**overrides) -> ExpectedMeta:
    base = dict(piece_id="test.piece", expected_key="C major", expected_bars=3)
    base.update(overrides)
    return ExpectedMeta(**base)


class TestDoDMinimums:
    def test_clean_score_has_no_min_violations(self) -> None:
        score = _make_score(_c_major_clean_bars())
        violations = validate_score(score, _expected())
        assert not any(v.check == "min_notes" for v in violations)
        assert not any(v.check == "total_bars" for v in violations)
        assert not any(v.check == "monotonic_onsets" for v in violations)

    def test_too_few_notes_flagged(self) -> None:
        bars = [_bar(1, 0, [_note(60, 0.0, 0), _note(62, 0.25, 240)])]
        score = _make_score(bars)
        violations = validate_score(score, _expected(expected_bars=1))
        assert any(v.check == "min_notes" for v in violations)

    def test_zero_bars_flagged(self) -> None:
        score = _make_score([])
        violations = validate_score(score, _expected())
        assert any(v.check == "total_bars" for v in violations)

    def test_non_monotonic_onsets_flagged(self) -> None:
        bars = _c_major_clean_bars()
        # Corrupt: make the last note of bar 1 precede an earlier note.
        bars[0].notes[-1] = _note(72, -1.0, -1000)
        score = _make_score(bars)
        violations = validate_score(score, _expected())
        assert any(v.check == "monotonic_onsets" for v in violations)

    def test_violation_is_frozen_dataclass(self) -> None:
        v = Violation(check="min_notes", detail="x")
        assert v.check == "min_notes"
        assert v.detail == "x"
```

**Step 2 — Run, verify FAIL.**
```
cd model && uv run python -m pytest tests/score_library/test_validate.py -v
```
Expected: `ModuleNotFoundError: No module named 'score_library.validate'`.

**Step 3 — Implement.**
Create `model/src/score_library/validate.py`:

```python
"""Chroma-independent validation gate for ingested score MIDIs.

validate_score returns a list of Violations (empty == pass). It uses only
chroma-independent signals: DoD minimums, pitch range, bar-count plausibility,
16th-grid quantization (recovered from bar.start_tick deltas + time signature),
and Krumhansl-Schmuckler key agreement. It NEVER uses chroma self-recognition;
gating on the chroma matcher would rig the #21 feasibility harness.
"""

from __future__ import annotations

from dataclasses import dataclass

from score_library.schema import ScoreData


@dataclass(frozen=True)
class ExpectedMeta:
    """Per-piece expected metadata and gate thresholds."""

    piece_id: str
    expected_key: str
    expected_bars: int
    min_notes: int = 20
    bar_tol_low: float = 0.7
    bar_tol_high: float = 2.2
    quant_max_median_dev_beats: float = 0.18
    key_min_correlation: float = 0.6
    pitch_low: int = 21
    pitch_high: int = 108


@dataclass(frozen=True)
class Violation:
    """A single failed gate check."""

    check: str
    detail: str


def _flatten_notes(score: ScoreData) -> list:
    notes = []
    for bar in score.bars:
        notes.extend(bar.notes)
    return notes


def validate_score(score: ScoreData, expected: ExpectedMeta) -> list[Violation]:
    """Validate a parsed score against expected metadata. Empty list == pass."""
    violations: list[Violation] = []

    notes = _flatten_notes(score)

    # (a) DoD minimums.
    if len(notes) < expected.min_notes:
        violations.append(
            Violation("min_notes", f"{len(notes)} notes < {expected.min_notes} minimum")
        )
    if score.total_bars < 1:
        violations.append(Violation("total_bars", f"total_bars={score.total_bars} < 1"))
    onsets = [n.onset_seconds for n in notes]
    if any(onsets[i] < onsets[i - 1] for i in range(1, len(onsets))):
        violations.append(Violation("monotonic_onsets", "onset_seconds not non-decreasing"))

    return violations
```

**Step 4 — Run, verify PASS.**
```
cd model && uv run python -m pytest tests/score_library/test_validate.py -v
```
Expected: all 5 `TestDoDMinimums` tests pass.

**Step 5 — Commit.**
```
git add model/src/score_library/validate.py model/tests/score_library/test_validate.py && git commit -m "feat(score-library): add DoD-minimum checks to validation gate"
```

---

### Task A2 — pitch-range (validate.py)

Add `Violation("pitch_range", ...)` if any `note.pitch < expected.pitch_low` or `> expected.pitch_high`.

**Step 1 — Write the failing test.**
Append to `model/tests/score_library/test_validate.py`:

```python
class TestPitchRange:
    def test_in_range_score_has_no_pitch_violation(self) -> None:
        score = _make_score(_c_major_clean_bars())
        violations = validate_score(score, _expected())
        assert not any(v.check == "pitch_range" for v in violations)

    def test_pitch_below_low_flagged(self) -> None:
        bars = _c_major_clean_bars()
        bars[0].notes[0] = _note(20, bars[0].notes[0].onset_seconds, bars[0].notes[0].onset_tick)
        score = _make_score(bars)
        violations = validate_score(score, _expected())
        assert any(v.check == "pitch_range" for v in violations)

    def test_pitch_above_high_flagged(self) -> None:
        bars = _c_major_clean_bars()
        bars[0].notes[0] = _note(109, bars[0].notes[0].onset_seconds, bars[0].notes[0].onset_tick)
        score = _make_score(bars)
        violations = validate_score(score, _expected())
        assert any(v.check == "pitch_range" for v in violations)
```

**Step 2 — Run, verify FAIL.**
```
cd model && uv run python -m pytest tests/score_library/test_validate.py::TestPitchRange -v
```
Expected: `test_pitch_below_low_flagged` and `test_pitch_above_high_flagged` FAIL (no `pitch_range` violation produced yet); `AssertionError`.

**Step 3 — Implement.**
In `model/src/score_library/validate.py`, inside `validate_score`, after the DoD-minimums block and before `return violations`, add:

```python
    # (b) Pitch range.
    out_of_range = [n.pitch for n in notes if n.pitch < expected.pitch_low or n.pitch > expected.pitch_high]
    if out_of_range:
        violations.append(
            Violation(
                "pitch_range",
                f"{len(out_of_range)} notes outside [{expected.pitch_low}, {expected.pitch_high}]",
            )
        )
```

**Step 4 — Run, verify PASS.**
```
cd model && uv run python -m pytest tests/score_library/test_validate.py -v
```
Expected: all `TestDoDMinimums` and `TestPitchRange` tests pass.

**Step 5 — Commit.**
```
git add model/src/score_library/validate.py model/tests/score_library/test_validate.py && git commit -m "feat(score-library): add pitch-range check to validation gate"
```

---

### Task A3 — bar-count plausibility (validate.py)

Add `Violation("bar_count", ...)` if `score.total_bars` is NOT in `[expected.bar_tol_low * expected.expected_bars, expected.bar_tol_high * expected.expected_bars]` (inclusive).

**Step 1 — Write the failing test.**
Append to `model/tests/score_library/test_validate.py`:

```python
class TestBarCount:
    def test_plausible_bar_count_no_violation(self) -> None:
        # 3 actual bars; expected 3 -> well within [0.7*3, 2.2*3] = [2.1, 6.6].
        score = _make_score(_c_major_clean_bars(n_bars=3))
        violations = validate_score(score, _expected(expected_bars=3))
        assert not any(v.check == "bar_count" for v in violations)

    def test_too_few_bars_flagged(self) -> None:
        # 3 actual bars; expected 10 -> 0.7*10 = 7.0 > 3 -> violation.
        score = _make_score(_c_major_clean_bars(n_bars=3))
        violations = validate_score(score, _expected(expected_bars=10))
        assert any(v.check == "bar_count" for v in violations)

    def test_too_many_bars_flagged(self) -> None:
        # 3 actual bars; expected 1 -> 2.2*1 = 2.2 < 3 -> violation.
        score = _make_score(_c_major_clean_bars(n_bars=3))
        violations = validate_score(score, _expected(expected_bars=1))
        assert any(v.check == "bar_count" for v in violations)

    def test_repeat_unfold_tolerated(self) -> None:
        # 6 actual bars; expected 3 -> 2.2*3 = 6.6 >= 6 -> no violation (repeat unfold).
        score = _make_score(_c_major_clean_bars(n_bars=6))
        violations = validate_score(score, _expected(expected_bars=3))
        assert not any(v.check == "bar_count" for v in violations)
```

**Step 2 — Run, verify FAIL.**
```
cd model && uv run python -m pytest tests/score_library/test_validate.py::TestBarCount -v
```
Expected: `test_too_few_bars_flagged` and `test_too_many_bars_flagged` FAIL (no `bar_count` violation produced yet); `AssertionError`.

**Step 3 — Implement.**
In `validate_score`, after the pitch-range block and before `return violations`, add:

```python
    # (c) Bar-count plausibility.
    low = expected.bar_tol_low * expected.expected_bars
    high = expected.bar_tol_high * expected.expected_bars
    if not (low <= score.total_bars <= high):
        violations.append(
            Violation(
                "bar_count",
                f"total_bars={score.total_bars} outside plausible [{low:.1f}, {high:.1f}] for expected {expected.expected_bars}",
            )
        )
```

**Step 4 — Run, verify PASS.**
```
cd model && uv run python -m pytest tests/score_library/test_validate.py -v
```
Expected: all tests through `TestBarCount` pass.

**Step 5 — Commit.**
```
git add model/src/score_library/validate.py model/tests/score_library/test_validate.py && git commit -m "feat(score-library): add bar-count plausibility check to validation gate"
```

---

### Task A4 — quantization (validate.py)

Recover the 16th-note grid and flag rubato/performance-timed MIDI. Algorithm (kept simple and robust):
- If fewer than 2 bars: SKIP this check (return no `quantization` violation).
- For each note, find its bar (the last bar whose `start_tick <= note.onset_tick`; notes before bar 1 belong to bar 1). Compute `bar_ticks` = (next bar `start_tick` - this bar `start_tick`); for the LAST bar reuse the previous bar's `bar_ticks`.
- Parse the bar's `time_signature` "num/den". `subdivisions_per_bar = numerator * 16 / denominator` (number of 16th-grid lines per bar). One beat = `16/denominator` sixteenths.
- `fraction = (note.onset_tick - bar.start_tick) / bar_ticks` (position within bar in [0,1)).
- `grid_pos = fraction * subdivisions_per_bar`; deviation from the nearest grid line (in 16th units) = `abs(grid_pos - round(grid_pos))`. Convert to beats: `dev_beats = dev_sixteenths / (16 / denominator)`.
- Take the **median** `dev_beats` over all notes (median makes triplets/grace notes tolerable; a rubato MIDI pushes the median over threshold). `Violation("quantization", ...)` if median > `expected.quant_max_median_dev_beats`.

**Step 1 — Write the failing test.**
Append to `model/tests/score_library/test_validate.py`:

```python
class TestQuantization:
    def test_clean_grid_passes(self) -> None:
        # Notes exactly on the 16th grid -> median deviation ~ 0.
        score = _make_score(_c_major_clean_bars(n_bars=3))
        violations = validate_score(score, _expected(expected_bars=3))
        assert not any(v.check == "quantization" for v in violations)

    def test_single_bar_skips_check(self) -> None:
        # Only 1 bar -> quantization check skipped even with off-grid notes.
        ppb = 480
        bar_ticks = ppb * 4
        notes = [_note(60 + i, (i * 137) / 1000.0, i * 137) for i in range(20)]
        score = _make_score([_bar(1, 0, notes)])
        violations = validate_score(score, _expected(expected_bars=1))
        assert not any(v.check == "quantization" for v in violations)

    def test_performance_timed_flagged(self) -> None:
        # Jitter every note far off the 16th grid -> median deviation exceeds 0.18 beats.
        ppb = 480
        bar_ticks = ppb * 4
        sixteenth = ppb // 4  # 120 ticks
        bars = []
        for b in range(3):
            notes = []
            for i in range(8):
                # On-grid tick + ~half-a-16th jitter (60 ticks = 0.5 sixteenth = 0.125 beat off,
                # but alternating sign keeps it consistently far from grid lines).
                base = b * bar_ticks + i * (2 * sixteenth)
                jitter = 60 if i % 2 == 0 else -60
                tick = base + jitter
                notes.append(_note(60 + i, tick / 1000.0, tick))
            bars.append(_bar(b + 1, b * bar_ticks, notes))
        score = _make_score(bars)
        violations = validate_score(score, _expected(expected_bars=3))
        assert any(v.check == "quantization" for v in violations)
```

**Step 2 — Run, verify FAIL.**
```
cd model && uv run python -m pytest tests/score_library/test_validate.py::TestQuantization -v
```
Expected: `test_performance_timed_flagged` FAILs (no `quantization` violation produced yet); `AssertionError`.

**Step 3 — Implement.**
In `validate.py`, add a module-level import at the top (after `from dataclasses import dataclass`):

```python
from statistics import median
```

In `validate_score`, after the bar-count block and before `return violations`, add:

```python
    # (d) Quantization: recover the 16th grid from bar.start_tick deltas + time sig.
    if len(score.bars) >= 2:
        bar_ticks_list: list[int] = []
        for i in range(len(score.bars) - 1):
            bar_ticks_list.append(score.bars[i + 1].start_tick - score.bars[i].start_tick)
        # Last bar reuses the previous bar's tick span.
        bar_ticks_list.append(bar_ticks_list[-1] if bar_ticks_list else 0)

        devs_beats: list[float] = []
        for bi, bar in enumerate(score.bars):
            bar_ticks = bar_ticks_list[bi]
            if bar_ticks <= 0:
                continue
            try:
                num_str, den_str = bar.time_signature.split("/")
                denominator = int(den_str)
            except (ValueError, AttributeError):
                continue
            numerator = int(num_str)
            subdivisions_per_bar = numerator * 16 / denominator
            sixteenths_per_beat = 16 / denominator
            for note in bar.notes:
                fraction = (note.onset_tick - bar.start_tick) / bar_ticks
                grid_pos = fraction * subdivisions_per_bar
                dev_sixteenths = abs(grid_pos - round(grid_pos))
                devs_beats.append(dev_sixteenths / sixteenths_per_beat)

        if devs_beats:
            med = median(devs_beats)
            if med > expected.quant_max_median_dev_beats:
                violations.append(
                    Violation(
                        "quantization",
                        f"median grid deviation {med:.3f} beats > {expected.quant_max_median_dev_beats} (performance-timed?)",
                    )
                )
```

**Step 4 — Run, verify PASS.**
```
cd model && uv run python -m pytest tests/score_library/test_validate.py -v
```
Expected: all tests through `TestQuantization` pass.

**Step 5 — Commit.**
```
git add model/src/score_library/validate.py model/tests/score_library/test_validate.py && git commit -m "feat(score-library): add 16th-grid quantization check to validation gate"
```

---

### Task A5 — key-agreement (Krumhansl-Schmuckler) (validate.py)

Build a 12-bin pitch-class histogram (count per `pitch % 12`, normalized to sum 1). Parse `expected_key` as "TONIC MODE": TONIC in the chromatic set (map enharmonics to pitch class 0..11; `Db`->1, `Ab`->8, `C#`->1, etc.), MODE in {`major`, `minor`}. Rotate the Krumhansl major/minor profile so its tonic aligns to the expected tonic pc, Pearson-correlate histogram vs rotated profile, and flag `Violation("key_agreement", ...)` if correlation < `expected.key_min_correlation`. **Do NOT trust `score.key_signature`** (defaulted/unreliable). **Do NOT use chroma.**

**Step 1 — Write the failing test.**
Append to `model/tests/score_library/test_validate.py`:

```python
class TestKeyAgreement:
    def test_c_major_score_agrees_with_c_major(self) -> None:
        score = _make_score(_c_major_clean_bars(n_bars=3))
        violations = validate_score(score, _expected(expected_key="C major", expected_bars=3))
        assert not any(v.check == "key_agreement" for v in violations)

    def test_c_major_score_disagrees_with_fsharp_major(self) -> None:
        # Same C-major notes, but expecting F# major -> low correlation -> violation.
        score = _make_score(_c_major_clean_bars(n_bars=3))
        violations = validate_score(score, _expected(expected_key="F# major", expected_bars=3))
        assert any(v.check == "key_agreement" for v in violations)

    def test_a_minor_score_agrees_with_a_minor(self) -> None:
        # A natural-minor scale notes.
        ppb = 480
        bar_ticks = ppb * 4
        sixteenth = ppb // 4
        a_minor = [69, 71, 72, 74, 76, 77, 79, 81]
        bars = []
        for b in range(3):
            notes = []
            for i, pitch in enumerate(a_minor):
                tick = b * bar_ticks + i * (2 * sixteenth)
                notes.append(_note(pitch, tick / 1000.0, tick))
            bars.append(_bar(b + 1, b * bar_ticks, notes))
        score = _make_score(bars)
        violations = validate_score(score, _expected(expected_key="A minor", expected_bars=3))
        assert not any(v.check == "key_agreement" for v in violations)

    def test_enharmonic_db_major_parsed(self) -> None:
        # A Db-major scale should agree with expected "Db major".
        ppb = 480
        bar_ticks = ppb * 4
        sixteenth = ppb // 4
        db_major = [61, 63, 65, 66, 68, 70, 72, 73]  # Db Eb F Gb Ab Bb C Db
        bars = []
        for b in range(3):
            notes = []
            for i, pitch in enumerate(db_major):
                tick = b * bar_ticks + i * (2 * sixteenth)
                notes.append(_note(pitch, tick / 1000.0, tick))
            bars.append(_bar(b + 1, b * bar_ticks, notes))
        score = _make_score(bars)
        violations = validate_score(score, _expected(expected_key="Db major", expected_bars=3))
        assert not any(v.check == "key_agreement" for v in violations)
```

**Step 2 — Run, verify FAIL.**
```
cd model && uv run python -m pytest tests/score_library/test_validate.py::TestKeyAgreement -v
```
Expected: `test_c_major_score_disagrees_with_fsharp_major` FAILs (no `key_agreement` violation produced yet); `AssertionError`.

**Step 3 — Implement.**
In `validate.py`, add module-level constants and a tonic map (after the `from statistics import median` import):

```python
KRUMHANSL_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
KRUMHANSL_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

_TONIC_TO_PC = {
    "C": 0, "B#": 0,
    "C#": 1, "Db": 1,
    "D": 2,
    "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4,
    "F": 5, "E#": 5,
    "F#": 6, "Gb": 6,
    "G": 7,
    "G#": 8, "Ab": 8,
    "A": 9,
    "A#": 10, "Bb": 10,
    "B": 11, "Cb": 11,
}


def _pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    cov = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(n))
    var_a = sum((a[i] - mean_a) ** 2 for i in range(n))
    var_b = sum((b[i] - mean_b) ** 2 for i in range(n))
    denom = (var_a * var_b) ** 0.5
    if denom == 0:
        return 0.0
    return cov / denom
```

In `validate_score`, after the quantization block and before `return violations`, add:

```python
    # (e) Key agreement (Krumhansl-Schmuckler) -- chroma-independent.
    parts = expected.expected_key.split()
    if len(parts) == 2 and parts[0] in _TONIC_TO_PC and parts[1] in ("major", "minor"):
        tonic_pc = _TONIC_TO_PC[parts[0]]
        profile = KRUMHANSL_MAJOR if parts[1] == "major" else KRUMHANSL_MINOR
        rotated = [profile[(pc - tonic_pc) % 12] for pc in range(12)]

        pc_counts = [0] * 12
        for n in notes:
            pc_counts[n.pitch % 12] += 1
        total = sum(pc_counts)
        if total > 0:
            histogram = [c / total for c in pc_counts]
            corr = _pearson(histogram, rotated)
            if corr < expected.key_min_correlation:
                violations.append(
                    Violation(
                        "key_agreement",
                        f"key correlation {corr:.3f} < {expected.key_min_correlation} for expected {expected.expected_key}",
                    )
                )
```

**Step 4 — Run, verify PASS.**
```
cd model && uv run python -m pytest tests/score_library/test_validate.py -v
```
Expected: all `test_validate.py` tests pass.

**Step 5 — Commit.**
```
git add model/src/score_library/validate.py model/tests/score_library/test_validate.py && git commit -m "feat(score-library): add Krumhansl key-agreement check to validation gate"
```

---

## Group B — manual.py (depends A, sequential B1->B4)

> All B-tasks share `test_manual.py`. B1 introduces three inline `mido`-built fixture builders (`build_clean_c_major_bytes`, `build_performance_timed_bytes`, `build_transposed_bytes`) and a dict-backed `fetch_fn` fake; B2-B4 reuse them (already present) and ADD test methods.

### Task B1 — happy-path ingest (manual.py)

`ingest_manifest(manifest_path, scores_dir, lock_path, fetch_fn=_http_fetch) -> IngestReport`. Reads manifest JSON (list of `{slug, piece_id, composer, title, expected_key, expected_bars, license, sources:[url,...]}`). For each piece, walk `sources` in order: `bytes=fetch_fn(url)`; `sha=sha256(bytes)`; write bytes to a temp file; `score=parse_score_midi(tmp, piece_id, composer, title)`; `violations=validate_score(score, ExpectedMeta(...from entry...))`; if empty -> WIN: write `scores_dir/{piece_id}.json` (`json.dump(score.model_dump(), indent=2)`), record `{resolved_url, sha256}`, break to next piece. Write the lockfile as JSON `{piece_id: {resolved_url, sha256}}`. `IngestReport` is a frozen dataclass summarizing resolved pieces.

**Step 1 — Write the failing test.**
Create `model/tests/score_library/test_manual.py`:

```python
"""Tests for the ranked-source manual ingestion driver."""

from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path

import mido
import pytest

from score_library.manual import (
    IngestReport,
    SourceResolutionError,
    ingest_manifest,
)

PPB = 480
BAR_TICKS = PPB * 4
SIXTEENTH = PPB // 4


def _scale_track(pitches: list[int], n_bars: int, jitter: int = 0) -> mido.MidiTrack:
    """Build a track: each bar plays `pitches` on every other 16th (8 notes/bar)."""
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    track.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
    # Build absolute-tick (note_on, note_off) events, then delta-encode.
    events: list[tuple[int, str, int]] = []
    for b in range(n_bars):
        for i, pitch in enumerate(pitches):
            base = b * BAR_TICKS + i * (2 * SIXTEENTH)
            jt = (jitter if i % 2 == 0 else -jitter)
            on = base + jt
            off = on + SIXTEENTH
            events.append((on, "on", pitch))
            events.append((off, "off", pitch))
    events.sort(key=lambda e: e[0])
    prev = 0
    for abs_tick, kind, pitch in events:
        delta = abs_tick - prev
        prev = abs_tick
        if kind == "on":
            track.append(mido.Message("note_on", note=pitch, velocity=80, time=delta))
        else:
            track.append(mido.Message("note_off", note=pitch, velocity=0, time=delta))
    return track


def _midi_bytes(track: mido.MidiTrack) -> bytes:
    mid = mido.MidiFile(ticks_per_beat=PPB)
    mid.tracks.append(track)
    buf = BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


def build_clean_c_major_bytes() -> bytes:
    """>= 20 notes on a clean 4/4 16th grid in C major, several bars, monotonic."""
    return _midi_bytes(_scale_track([60, 62, 64, 65, 67, 69, 71, 72], n_bars=3))


def build_performance_timed_bytes() -> bytes:
    """Same C-major notes but onsets jittered far off-grid (quantization fails)."""
    return _midi_bytes(_scale_track([60, 62, 64, 65, 67, 69, 71, 72], n_bars=3, jitter=60))


def build_transposed_bytes() -> bytes:
    """Clean grid but transposed to F# major (key-agreement vs C major fails)."""
    return _midi_bytes(_scale_track([66, 68, 70, 71, 73, 75, 77, 78], n_bars=3))


def _make_fetch(mapping: dict[str, bytes]):
    def fetch(url: str) -> bytes:
        if url not in mapping:
            raise KeyError(f"no fixture for {url}")
        return mapping[url]
    return fetch


def _write_manifest(path: Path, entries: list[dict]) -> None:
    path.write_text(json.dumps(entries))


class TestIngestHappyPath:
    def test_clean_source_writes_json_and_lockfile(self, tmp_path: Path) -> None:
        scores_dir = tmp_path / "scores"
        scores_dir.mkdir()
        manifest_path = tmp_path / "manifest.json"
        lock_path = tmp_path / "lock.json"

        clean = build_clean_c_major_bytes()
        url = "https://example.org/cmaj.mid"
        _write_manifest(
            manifest_path,
            [
                {
                    "slug": "test_slug",
                    "piece_id": "test.cmajor",
                    "composer": "Test",
                    "title": "C Major",
                    "expected_key": "C major",
                    "expected_bars": 3,
                    "license": "PD",
                    "sources": [url],
                }
            ],
        )

        report = ingest_manifest(
            manifest_path, scores_dir, lock_path, fetch_fn=_make_fetch({url: clean})
        )

        assert isinstance(report, IngestReport)
        # Score JSON written.
        out = scores_dir / "test.cmajor.json"
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["piece_id"] == "test.cmajor"
        # Lockfile written with correct sha256.
        lock = json.loads(lock_path.read_text())
        assert lock["test.cmajor"]["resolved_url"] == url
        assert lock["test.cmajor"]["sha256"] == hashlib.sha256(clean).hexdigest()
```

**Step 2 — Run, verify FAIL.**
```
cd model && uv run python -m pytest tests/score_library/test_manual.py -v
```
Expected: `ModuleNotFoundError: No module named 'score_library.manual'`.

**Step 3 — Implement.**
Create `model/src/score_library/manual.py`:

```python
"""Ranked-source manual ingestion driver.

ingest_manifest fetches each piece's MIDI from a ranked list of public-domain
URLs, parses it via parse_score_midi, validates via validate_score, and pins the
winning (url + sha256) to a build-written lockfile. The first source that passes
the gate wins. If every source for a piece fails, it raises SourceResolutionError
with a per-candidate failure table -- no silent skip, no partial commit beyond the
per-piece JSONs already written for pieces that DID pass.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from score_library.parse import parse_score_midi
from score_library.schema import ScoreData
from score_library.validate import ExpectedMeta, validate_score


class SourceResolutionError(Exception):
    """Raised when no candidate source for a piece passes the validation gate."""


@dataclass(frozen=True)
class IngestReport:
    """Summary of a successful ingest run: piece_id -> {resolved_url, sha256}."""

    resolved: dict[str, dict[str, str]]


def _http_fetch(url: str) -> bytes:
    """Fetch raw bytes from a URL using stdlib urllib (no third-party deps)."""
    with urllib.request.urlopen(url) as resp:  # noqa: S310 (PD URLs from a pinned manifest)
        return resp.read()


def _expected_from_entry(entry: dict) -> ExpectedMeta:
    return ExpectedMeta(
        piece_id=entry["piece_id"],
        expected_key=entry["expected_key"],
        expected_bars=entry["expected_bars"],
    )


def ingest_manifest(
    manifest_path: Path,
    scores_dir: Path,
    lock_path: Path,
    fetch_fn=_http_fetch,
) -> IngestReport:
    """Resolve every piece in the manifest, writing score JSONs + a lockfile."""
    manifest_path = Path(manifest_path)
    scores_dir = Path(scores_dir)
    lock_path = Path(lock_path)
    scores_dir.mkdir(parents=True, exist_ok=True)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    existing_lock: dict[str, dict[str, str]] = {}
    if lock_path.exists():
        existing_lock = json.loads(lock_path.read_text())

    resolved: dict[str, dict[str, str]] = {}
    entries = json.loads(manifest_path.read_text())

    for entry in entries:
        piece_id = entry["piece_id"]
        expected = _expected_from_entry(entry)
        pinned_sha = existing_lock.get(piece_id, {}).get("sha256")
        candidate_failures: list[str] = []
        won = False

        for url in entry["sources"]:
            raw = fetch_fn(url)
            sha = hashlib.sha256(raw).hexdigest()

            if pinned_sha is not None and sha != pinned_sha:
                candidate_failures.append(f"{url}: hash_mismatch (got {sha[:12]}, pinned {pinned_sha[:12]})")
                continue

            with tempfile.NamedTemporaryFile(suffix=".mid", delete=True) as tmp:
                tmp.write(raw)
                tmp.flush()
                score: ScoreData = parse_score_midi(
                    tmp.name, piece_id, entry["composer"], entry["title"]
                )

            violations = validate_score(score, expected)
            if violations:
                detail = "; ".join(f"{v.check}:{v.detail}" for v in violations)
                candidate_failures.append(f"{url}: {detail}")
                continue

            out_path = scores_dir / f"{piece_id}.json"
            with open(out_path, "w") as f:
                json.dump(score.model_dump(), f, indent=2)
            resolved[piece_id] = {"resolved_url": url, "sha256": sha}
            won = True
            break

        if not won:
            table = "\n".join(f"  - {line}" for line in candidate_failures)
            raise SourceResolutionError(
                f"No source resolved for {piece_id}. Candidate failures:\n{table}"
            )

    with open(lock_path, "w") as f:
        json.dump(resolved, f, indent=2)

    return IngestReport(resolved=resolved)
```

**Step 4 — Run, verify PASS.**
```
cd model && uv run python -m pytest tests/score_library/test_manual.py -v
```
Expected: `TestIngestHappyPath::test_clean_source_writes_json_and_lockfile` passes.

**Step 5 — Commit.**
```
git add model/src/score_library/manual.py model/tests/score_library/test_manual.py && git commit -m "feat(score-library): add ranked-source manual ingest driver (happy path)"
```

---

### Task B2 — hash-mismatch rejection (manual.py)

If the lockfile pins this piece to a sha and a candidate's sha differs, that candidate is rejected with a `hash_mismatch` failure and the loop continues. (Implemented in B1; B2 adds the test that nails the behavior through the public interface.)

**Step 1 — Write the failing test.**
Append to `model/tests/score_library/test_manual.py`:

```python
class TestIngestHashMismatch:
    def test_pinned_wrong_sha_rejects_only_candidate_then_halts(self, tmp_path: Path) -> None:
        scores_dir = tmp_path / "scores"
        scores_dir.mkdir()
        manifest_path = tmp_path / "manifest.json"
        lock_path = tmp_path / "lock.json"

        clean = build_clean_c_major_bytes()
        url = "https://example.org/cmaj.mid"
        # Pre-seed the lockfile pinning this piece to a DIFFERENT sha.
        lock_path.write_text(json.dumps({"test.cmajor": {"resolved_url": url, "sha256": "deadbeef"}}))
        _write_manifest(
            manifest_path,
            [
                {
                    "slug": "test_slug",
                    "piece_id": "test.cmajor",
                    "composer": "Test",
                    "title": "C Major",
                    "expected_key": "C major",
                    "expected_bars": 3,
                    "license": "PD",
                    "sources": [url],
                }
            ],
        )

        with pytest.raises(SourceResolutionError) as exc:
            ingest_manifest(
                manifest_path, scores_dir, lock_path, fetch_fn=_make_fetch({url: clean})
            )
        assert "hash_mismatch" in str(exc.value)
        # The only candidate was rejected by the pin, so no JSON was written.
        assert not (scores_dir / "test.cmajor.json").exists()
```

**Step 2 — Run, verify FAIL.**

> NOTE TO EXECUTOR: the hash-mismatch path is already implemented in B1, so this test may PASS immediately. If it passes on first run, that is acceptable for this task (the behavior is verified through the public interface); proceed to Step 5. The watch-it-fail discipline is satisfied across the group because B1's test failed before the module existed. Run:

```
cd model && uv run python -m pytest tests/score_library/test_manual.py::TestIngestHashMismatch -v
```
Expected: PASS (behavior already implemented). If it had failed, the implementation in Step 3 would close the gap.

**Step 3 — Implement.**
No new code required — the `pinned_sha` check in `ingest_manifest` (B1) already records `hash_mismatch` and continues. (If the executor's B1 diverged and the test fails, add the `if pinned_sha is not None and sha != pinned_sha` guard exactly as written in B1.)

**Step 4 — Run, verify PASS.**
```
cd model && uv run python -m pytest tests/score_library/test_manual.py -v
```
Expected: all `test_manual.py` tests pass.

**Step 5 — Commit.**
```
git add model/tests/score_library/test_manual.py && git commit -m "test(score-library): verify hash-mismatch rejects pinned candidate"
```

---

### Task B3 — ranked fallback (manual.py)

When the first source fails the gate (e.g. performance-timed) and a later source passes, the later source wins and the lockfile records the later URL.

**Step 1 — Write the failing test.**
Append to `model/tests/score_library/test_manual.py`:

```python
class TestIngestRankedFallback:
    def test_first_fails_second_wins(self, tmp_path: Path) -> None:
        scores_dir = tmp_path / "scores"
        scores_dir.mkdir()
        manifest_path = tmp_path / "manifest.json"
        lock_path = tmp_path / "lock.json"

        bad = build_performance_timed_bytes()
        good = build_clean_c_major_bytes()
        url_bad = "https://example.org/perf.mid"
        url_good = "https://example.org/clean.mid"
        _write_manifest(
            manifest_path,
            [
                {
                    "slug": "test_slug",
                    "piece_id": "test.cmajor",
                    "composer": "Test",
                    "title": "C Major",
                    "expected_key": "C major",
                    "expected_bars": 3,
                    "license": "PD",
                    "sources": [url_bad, url_good],
                }
            ],
        )

        report = ingest_manifest(
            manifest_path,
            scores_dir,
            lock_path,
            fetch_fn=_make_fetch({url_bad: bad, url_good: good}),
        )

        assert (scores_dir / "test.cmajor.json").exists()
        lock = json.loads(lock_path.read_text())
        # Second (clean) source won; lockfile records its URL.
        assert lock["test.cmajor"]["resolved_url"] == url_good
        assert report.resolved["test.cmajor"]["resolved_url"] == url_good
```

**Step 2 — Run, verify FAIL.**

> NOTE TO EXECUTOR: ranked fallback is implemented in B1's loop, so this test likely PASSES immediately. Same rationale as B2 — acceptable. Run:

```
cd model && uv run python -m pytest tests/score_library/test_manual.py::TestIngestRankedFallback -v
```
Expected: PASS (behavior already implemented).

**Step 3 — Implement.**
No new code required — the `for url in entry["sources"]` loop with `continue`-on-violation and `break`-on-win already implements ranked fallback. (Precondition for this test passing: the A4 quantization check correctly flags `build_performance_timed_bytes`. If it does not, revisit A4's threshold — do NOT weaken it; instead increase the fixture jitter.)

**Step 4 — Run, verify PASS.**
```
cd model && uv run python -m pytest tests/score_library/test_manual.py -v
```
Expected: all `test_manual.py` tests pass.

**Step 5 — Commit.**
```
git add model/tests/score_library/test_manual.py && git commit -m "test(score-library): verify ranked-source fallback picks clean source"
```

---

### Task B4 — all-fail HALT (manual.py)

When every source for a piece fails the gate, `ingest_manifest` raises `SourceResolutionError` carrying a per-candidate failure table, and writes no JSON for that piece.

**Step 1 — Write the failing test.**
Append to `model/tests/score_library/test_manual.py`:

```python
class TestIngestAllFailHalt:
    def test_all_sources_fail_raises_with_table(self, tmp_path: Path) -> None:
        scores_dir = tmp_path / "scores"
        scores_dir.mkdir()
        manifest_path = tmp_path / "manifest.json"
        lock_path = tmp_path / "lock.json"

        perf = build_performance_timed_bytes()
        transposed = build_transposed_bytes()
        url_a = "https://example.org/perf.mid"
        url_b = "https://example.org/transposed.mid"
        _write_manifest(
            manifest_path,
            [
                {
                    "slug": "test_slug",
                    "piece_id": "test.cmajor",
                    "composer": "Test",
                    "title": "C Major",
                    "expected_key": "C major",
                    "expected_bars": 3,
                    "license": "PD",
                    "sources": [url_a, url_b],
                }
            ],
        )

        with pytest.raises(SourceResolutionError) as exc:
            ingest_manifest(
                manifest_path,
                scores_dir,
                lock_path,
                fetch_fn=_make_fetch({url_a: perf, url_b: transposed}),
            )
        msg = str(exc.value)
        assert "test.cmajor" in msg
        assert url_a in msg and url_b in msg
        # No JSON written for the unresolved piece.
        assert not (scores_dir / "test.cmajor.json").exists()
```

**Step 2 — Run, verify FAIL.**

> NOTE TO EXECUTOR: HALT-on-all-fail is implemented in B1, so this test likely PASSES immediately. Same rationale. Run:

```
cd model && uv run python -m pytest tests/score_library/test_manual.py::TestIngestAllFailHalt -v
```
Expected: PASS (behavior already implemented).

**Step 3 — Implement.**
No new code required — the `if not won: raise SourceResolutionError(...)` block already builds the failure table from `candidate_failures`. (Precondition: A4 flags `perf` on quantization and A5 flags `transposed` on key_agreement; if either passes the gate this test will not raise — fix the gate sensitivity or fixture, never silence the test.)

**Step 4 — Run, verify PASS.**
```
cd model && uv run python -m pytest tests/score_library/test_manual.py -v
```
Expected: all `test_manual.py` tests pass.

**Step 5 — Commit.**
```
git add model/tests/score_library/test_manual.py && git commit -m "test(score-library): verify all-sources-fail HALTs with failure table"
```

---

## Group C — cli.py (depends B)

### Task C1 — `parse-manual` subcommand (cli.py)

Add `cmd_parse_manual(args)` calling `ingest_manifest(Path(args.manifest), Scores.root, Path(args.lock))`, a `parse-manual` subparser with `--manifest` (required) and `--lock` (default `model/data/manifests/manual_scores.lock.json`), and register it in the dispatch dict. `cmd_fingerprint` UNCHANGED.

**Step 1 — Write the failing test.**
Create `model/tests/score_library/test_cli_parse_manual.py`:

```python
"""Test that the parse-manual CLI wiring ingests a tiny manifest end to end."""

from __future__ import annotations

import argparse
import hashlib
import json
from io import BytesIO
from pathlib import Path

import mido

from score_library.cli import cmd_parse_manual

PPB = 480
BAR_TICKS = PPB * 4
SIXTEENTH = PPB // 4


def _clean_c_major_bytes() -> bytes:
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    track.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
    pitches = [60, 62, 64, 65, 67, 69, 71, 72]
    events: list[tuple[int, str, int]] = []
    for b in range(3):
        for i, pitch in enumerate(pitches):
            on = b * BAR_TICKS + i * (2 * SIXTEENTH)
            events.append((on, "on", pitch))
            events.append((on + SIXTEENTH, "off", pitch))
    events.sort(key=lambda e: e[0])
    prev = 0
    for abs_tick, kind, pitch in events:
        delta = abs_tick - prev
        prev = abs_tick
        msg_type = "note_on" if kind == "on" else "note_off"
        vel = 80 if kind == "on" else 0
        track.append(mido.Message(msg_type, note=pitch, velocity=vel, time=delta))
    mid = mido.MidiFile(ticks_per_beat=PPB)
    mid.tracks.append(track)
    buf = BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


def test_cmd_parse_manual_writes_score(tmp_path: Path, monkeypatch) -> None:
    scores_dir = tmp_path / "scores"
    manifest_path = tmp_path / "manifest.json"
    lock_path = tmp_path / "lock.json"

    clean = _clean_c_major_bytes()
    url = "https://example.org/cmaj.mid"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "slug": "test_slug",
                    "piece_id": "test.cmajor",
                    "composer": "Test",
                    "title": "C Major",
                    "expected_key": "C major",
                    "expected_bars": 3,
                    "license": "PD",
                    "sources": [url],
                }
            ]
        )
    )

    # Redirect Scores.root to the temp dir and inject a fake fetch.
    import score_library.cli as cli_mod
    monkeypatch.setattr(cli_mod.Scores, "root", scores_dir)

    import score_library.manual as manual_mod
    monkeypatch.setattr(manual_mod, "_http_fetch", lambda u: clean)

    args = argparse.Namespace(manifest=str(manifest_path), lock=str(lock_path))
    cmd_parse_manual(args)

    out = scores_dir / "test.cmajor.json"
    assert out.exists()
    lock = json.loads(lock_path.read_text())
    assert lock["test.cmajor"]["sha256"] == hashlib.sha256(clean).hexdigest()
```

**Step 2 — Run, verify FAIL.**
```
cd model && uv run python -m pytest tests/score_library/test_cli_parse_manual.py -v
```
Expected: `ImportError: cannot import name 'cmd_parse_manual' from 'score_library.cli'`.

**Step 3 — Implement.**
In `model/src/score_library/cli.py`, add `cmd_parse_manual` after `cmd_fingerprint` (around line 158):

```python
def cmd_parse_manual(args):
    from score_library.manual import ingest_manifest

    manifest_path = Path(args.manifest)
    default_lock = REPO_ROOT_DATA / "manifests" / "manual_scores.lock.json"
    lock_path = Path(args.lock) if args.lock else default_lock
    report = ingest_manifest(manifest_path, Scores.root, lock_path)
    print(f"Resolved {len(report.resolved)} pieces -> {lock_path}")
    for piece_id, info in report.resolved.items():
        print(f"  {piece_id}: {info['resolved_url']}")
```

At the top of `cli.py`, after `from src.paths import Scores`, add an import for the data root used to build the default lock path:

```python
from src.paths import DATA_ROOT as REPO_ROOT_DATA
```

Register the subparser in `main()` (after the `fingerprint` subparser block, before `args = parser.parse_args()`):

```python
    p_parse_manual = sub.add_parser(
        "parse-manual", help="Ingest manual score MIDIs from a ranked-URL manifest"
    )
    p_parse_manual.add_argument("--manifest", required=True, help="Path to manual_scores.json")
    p_parse_manual.add_argument(
        "--lock",
        default=None,
        help="Lockfile path (default: data/manifests/manual_scores.lock.json)",
    )
```

Add to the dispatch dict:

```python
        "parse-manual": cmd_parse_manual,
```

**Step 4 — Run, verify PASS.**
```
cd model && uv run python -m pytest tests/score_library/test_cli_parse_manual.py -v
```
Expected: `test_cmd_parse_manual_writes_score` passes.

**Step 5 — Commit.**
```
git add model/src/score_library/cli.py model/tests/score_library/test_cli_parse_manual.py && git commit -m "feat(score-library): add parse-manual CLI subcommand"
```

---

## Group D — execution / data tasks (depends A/B/C; verified by Group-0 harness, NOT red-green TDD)

> These tasks author real data and run the machinery. They are verified by `just catalog-verify` (Group-0 harness) GREEN and by the existing unit suites, NOT by new red-green tests. **KNOWN AUTOMATION RISK** (restate at execution time): each of the 11 pieces must auto-source from a candidate URL with metric (non-performance) timing in the correct key; if a piece cannot resolve, `parse-manual` HALTs with a `SourceResolutionError` failure table. Complete all machinery and all resolvable pieces; surface unresolved pieces with their failure table. DO NOT fabricate URLs, DO NOT commit partial coverage, DO NOT weaken the gate.

### Task D1 — author `manual_scores.json`

Create `model/data/manifests/manual_scores.json`: a JSON list of 11 entries (one per missing piece). Each entry: `{slug, piece_id, composer, title, expected_key, expected_bars, license, sources:[url, ...]}`. Source URLs must be **real public-domain engraved MIDIs**, Mutopia-first, with MuseScore-PD as fallback. **Avoid kunstderfuge** (not reliably PD / not stable). Rank sources best-first (cleanest/most-authoritative engraving first). Use these expected-metadata tuples (`expected_bars` approximate):

| piece_id | expected_key | expected_bars |
|----------|--------------|---------------|
| bach.inventions.1 | C major | 22 |
| chopin.waltzes.64-2 | C# minor | 129 |
| debussy.suite_bergamasque.3_clair_de_lune | Db major | 72 |
| debussy.deux_arabesques.1 | E major | 107 |
| chopin.fantaisie_impromptu | C# minor | 138 |
| beethoven.fur_elise | A minor | 124 |
| liszt.liebestraume.3 | Ab major | 85 |
| beethoven.piano_sonatas.14-1 | C# minor | 69 |
| mozart.piano_sonatas.16-1 | C major | 73 |
| rachmaninoff.preludes_op_3.2 | C# minor | 62 |
| schumann.kinderszenen.7 | F major | 24 |

(The other 5 of the 16 — `bach.prelude.bwv_846`, `chopin.ballades.1`, `chopin.etudes_op_10.4`, `beethoven.piano_sonatas.8-2`, `chopin.nocturnes.9-2` — are already in the ASAP catalog and are NOT ingested here; they appear only in `eval_piece_map.json`.)

`slug` per entry must match the `CANONICAL_MAP` key for that `piece_id` (e.g. `fur_elise`, `clair_de_lune`, `schumann_traumerei`, etc.).

**Verify:** the file is valid JSON with 11 entries, each having a non-empty `sources` list:
```
cd model && uv run python -c "import json; d=json.load(open('data/manifests/manual_scores.json')); assert len(d)==11; assert all(e['sources'] for e in d); print('OK', len(d))"
```

### Task D2 — run `parse-manual` (writes 11 scores + lockfile)

```
cd model && uv run python -m score_library.cli parse-manual --manifest data/manifests/manual_scores.json
```
Expected: writes `data/scores/{piece_id}.json` for all 11 pieces and `data/manifests/manual_scores.lock.json` with 11 `{resolved_url, sha256}` entries. The driver auto-falls-back through ranked sources; if any piece exhausts all sources it HALTs with a `SourceResolutionError` table — at that point source a replacement PD URL, add it to that entry's `sources`, and re-run. DO NOT proceed past a HALT with partial coverage.

**Verify:**
```
cd model && ls data/scores | grep -E "bach.inventions.1|beethoven.fur_elise|schumann.kinderszenen.7" && uv run python -c "import json; l=json.load(open('data/manifests/manual_scores.lock.json')); assert len(l)==11; print('locked', len(l))"
```

### Task D3 — fingerprint + `eval_piece_map.json` + `catalog-verify` recipe + commit

1. Add the `catalog-verify` recipe to `justfile` after the `fingerprint` recipe (around line 88):

```
# Verify all 16 labeled eval pieces have a non-trivial, monotonic catalog entry
catalog-verify:
    cd model && uv run python -c "from score_library.catalog_coverage import check_coverage, CANONICAL_MAP; from src.paths import Scores; import sys; f=check_coverage(Scores.root, CANONICAL_MAP); print(chr(10).join(f) if f else 'PASS'); sys.exit(1 if f else 0)"
```

2. Write `model/data/evals/piece_id/eval_piece_map.json` = the 16-entry `slug -> piece_id` map (identical content to `CANONICAL_MAP`):
```
cd model && mkdir -p data/evals/piece_id && uv run python -c "import json; from score_library.catalog_coverage import CANONICAL_MAP; json.dump(CANONICAL_MAP, open('data/evals/piece_id/eval_piece_map.json','w'), indent=2)"
```

3. Regenerate the fingerprint index over 255 pieces:
```
just fingerprint
```
Expected: the `fingerprint` output reports building over **255** scores (244 + 11).

4. Run the acceptance gate (must be GREEN):
```
just catalog-verify
```
Expected: prints `PASS` and exits 0. (RED before the 11 scores landed; GREEN now.)

5. Sanity: full score-library suite still green:
```
cd model && uv run python -m pytest tests/score_library/ -v
```

6. Commit the generated artifacts + recipe + manifest/lockfile + the 11 score JSONs:
```
git add justfile model/data/manifests/manual_scores.json model/data/manifests/manual_scores.lock.json model/data/evals/piece_id/eval_piece_map.json model/data/scores/ model/data/fingerprints/ngram_index.json model/data/fingerprints/rerank_features.json && git commit -m "feat(score-library): ingest 11 eval pieces, regenerate fingerprints over 255, add catalog-verify"
```

---

## Notes on dependency ordering for `/build`

- Group 0 and Group A may run in parallel (different files).
- Group A is strictly sequential A1->A5 (one file, cumulative `validate_score`).
- Group B requires `validate.py` (A5 done) AND `parse.py` (existing); strictly sequential B1->B4 (one file). B2-B4 are behavior-locking tests over B1's implementation — their "watch it fail" is satisfied at B1 (module absent); B2-B4 may pass on first run.
- Group C requires `manual.py` (B done) and edits `cli.py`.
- Group D requires C done and is data/execution, gated by `just catalog-verify`.
