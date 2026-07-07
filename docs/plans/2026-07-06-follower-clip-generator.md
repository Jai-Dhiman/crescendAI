# Synthetic Clip Generator for the Score-Follower Benchmark Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until `/challenge` returns VERDICT: PROCEED.

**Goal:** Given an ASAP piece and a practice-pathology type, deterministically produce a spliced performance note stream with an exact, ground-truth score-position-over-time trajectory, so every other Phase-0 component of EPIC #108 has labeled data to consume.
**Spec:** `docs/specs/2026-07-06-follower-clip-generator-design.md`
**Style:** Follow `CLAUDE.md` (explicit exceptions over silent fallbacks, `uv` not `pip`, no emojis) and the conventions already in `model/src/piece_id_eval/` (frozen dataclasses/NamedTuples for symbolic data, `Path(__file__).resolve()` + `.parents[N]` for default paths, docstrings stating `Raises:`).

All work happens inside `model/`. All commands below assume `cd model` first (paths are relative to `model/`). Branch: `issue-111-clip-generator` (this plan already runs inside the `.worktrees/issue-111-clip-generator` worktree — do not create a new one).

## Verified facts this plan depends on (do not re-derive, just use)

- `model/data/raw/asap-dataset/asap_annotations.json` is keyed by the performance MIDI's path relative to the asap-dataset root (e.g. `"Bach/Fugue/bwv_846/Shi05M.mid"`). Each entry has (among others) `performance_beats: list[float]` (seconds) and `midi_score_beats: list[float]` (score-beat position), same length, index-aligned, plus `score_and_performance_aligned: bool`.
- The score MIDI for a performance is always its sibling `midi_score.mid` (same directory).
- `partitura.load_performance_midi(path).note_array()` returns a structured numpy array with fields `onset_sec, duration_sec, onset_tick, duration_tick, pitch, velocity, track, channel, id`.
- Real fixture piece, aligned, short (40.7s, 92 beats): `Liszt/Transcendental_Etudes/1/LuoJ05M.mid` — referred to below as `ALIGNED_PIECE`.
- Real fixture piece, NOT aligned (`score_and_performance_aligned: false`): `Beethoven/Piano_Sonatas/16-1/LuoJ03M.mid` — referred to below as `UNALIGNED_PIECE`.
- `model/pyproject.toml` has `[tool.hatch.build.targets.wheel]` with a `packages = [...]` list of `src/<name>` entries; a new package must be added there to be importable via `uv run`.
- Test command pattern already used in this repo: `cd model && uv run pytest tests/<pkg>/test_x.py -q`.

## Task Groups

```
Task 0 (solo, blocks everything else): environment setup — ASAP dataset symlink
Group A (solo):                    Task 1
Group B (parallel with D, depends on A):  Task 2, Task 3, Task 4      (asap_alignment.py, sequential within group)
Group D (parallel with B, depends on A):  Task 5, Task 6, Task 7, Task 8   (segments.py, sequential within group)
Group C (depends on B and D):      Task 9, Task 10, Task 11, Task 12, Task 13   (trajectory.py, sequential)
Group E (depends on B, C, D):      Task 14 .. Task 22                  (pathologies.py, sequential)
Group F (depends on E):            Task 23 .. Task 26                  (clip_generator.py, sequential)
```

Tasks within the same file are sequential (same-file edits cannot run as parallel subagents). Group B and Group D touch disjoint files (`asap_alignment.py` vs `segments.py`) and have no import dependency on each other, so they run in parallel. Group C imports from both B's and D's modules, so it waits for both.

---

### Task 0: Environment setup — make the ASAP dataset available in this worktree

**Group:** solo, blocks everything else (runs before Task 1 / Group A)

**This is environment setup, not a code artifact.** `model/data/raw` is gitignored (`model/.gitignore` contains `data/raw`), and git worktrees do not inherit untracked/ignored files. Every task from Task 2 onward reads real fixtures from `model/data/raw/asap-dataset/` (`asap_annotations.json` plus the `ALIGNED_PIECE`/`UNALIGNED_PIECE` MIDIs). Without this step those tasks fail with `FileNotFoundError` instead of exercising the behavior under test. There is no TDD test for this step — it is a prerequisite, verified by direct inspection, not a unit test.

The dataset already exists in the primary checkout (the repo root you cloned from, not this worktree) at `model/data/raw/asap-dataset/`. Symlink it into this worktree rather than copying it (ASAP is large; a copy would duplicate that data and drift from the primary copy). The symlink target lives under `model/data/raw/`, which is gitignored, so the symlink itself is never committed.

Resolve the primary checkout's path **dynamically** — do not hardcode `.worktrees/issue-111-clip-generator`, since `/build` may run this plan from a different worktree than the one it was drafted in. `git rev-parse --git-common-dir` always resolves to the `.git` directory inside the primary checkout root, from any worktree, so its parent directory is the primary checkout root:

- [ ] **Step 1: Create the symlink**

```bash
cd model
PRIMARY_ROOT="$(dirname "$(git rev-parse --git-common-dir)")"
mkdir -p data/raw
ln -s "$PRIMARY_ROOT/model/data/raw/asap-dataset" data/raw/asap-dataset
```

- [ ] **Step 2: Verify — no test, direct inspection only**

```bash
cd model
test -f data/raw/asap-dataset/asap_annotations.json && echo "annotations OK"
test -f "data/raw/asap-dataset/Liszt/Transcendental_Etudes/1/LuoJ05M.mid" && echo "ALIGNED_PIECE OK"
test -f "data/raw/asap-dataset/Beethoven/Piano_Sonatas/16-1/LuoJ03M.mid" && echo "UNALIGNED_PIECE OK"
```
Expected: all three lines print. If `PRIMARY_ROOT` resolution finds a worktree checkout instead of the primary one (e.g. this plan is being run nested inside another worktree), re-derive it from `git worktree list --porcelain`'s first `worktree <path>` entry (the primary checkout is always listed first and is the only one without a `branch refs/heads/issue-...` line for this feature) and repeat Step 1.

- [ ] **Step 3: No commit** — `data/raw/` is gitignored, so `git status` should show no new tracked files from this step. Do not `git add` anything here.

---

### Task 1: Package scaffold is importable

**Group:** A (solo, blocks everything else)

**Behavior being verified:** `import follower_bench` succeeds after the package is registered.
**Interface under test:** the package itself.

**Files:**
- Create: `model/src/follower_bench/__init__.py`
- Modify: `model/pyproject.toml`
- Test: `model/tests/follower_bench/test_package.py`
- Create: `model/tests/follower_bench/__init__.py` (empty — matches sibling test packages `tests/piece_id_eval/__init__.py` and `tests/chroma_dtw_eval/__init__.py`, which are also empty)

- [ ] **Step 1: Write the failing test**

```python
# model/tests/follower_bench/test_package.py
"""Verify the follower_bench package is importable (editable-install wiring)."""
from __future__ import annotations

import follower_bench


def test_package_is_importable() -> None:
    assert follower_bench is not None
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/follower_bench/test_package.py -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'follower_bench'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `model/src/follower_bench/__init__.py`:

```python
"""Synthetic clip generator for the symbolic score-follower benchmark (issue #111).

Splices ASAP performances (already beat-aligned to their score MIDI) into
practice-pathology clips with exact ground-truth score-position trajectories.
See docs/specs/2026-07-06-follower-clip-generator-design.md.
"""
```

Edit `model/pyproject.toml`: in `[tool.hatch.build.targets.wheel]`, add `"src/follower_bench"` to the `packages` list (append it to the existing comma-separated list, do not remove any existing entries):

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/score_alignment", "src/audio_experiments", "src/model_improvement", "src/masterclass_experiments", "src/score_library", "src/exercise_corpus", "src/chroma_dtw_eval", "src/piece_id_eval", "src/follower_bench"]
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/follower_bench/test_package.py -q
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd model && git add src/follower_bench/__init__.py tests/follower_bench/__init__.py tests/follower_bench/test_package.py pyproject.toml && git commit -m "feat(follower-bench): scaffold follower_bench package (#111)"
```

---

### Task 2: `load_alignment` returns a valid ClipAlignment for a real aligned ASAP piece

**Group:** B (sequential: Task 2 → 3 → 4; depends on Group A; runs in parallel with Group D)

**Behavior being verified:** given the real, aligned ASAP piece `Liszt/Transcendental_Etudes/1/LuoJ05M.mid`, `load_alignment` resolves both MIDI paths and returns the beat arrays from the annotation.
**Interface under test:** `follower_bench.asap_alignment.load_alignment`

**Files:**
- Create: `model/src/follower_bench/asap_alignment.py`
- Test: `model/tests/follower_bench/test_asap_alignment.py`

- [ ] **Step 1: Write the failing test**

```python
# model/tests/follower_bench/test_asap_alignment.py
"""Verify load_alignment resolves paths and validates the ASAP beat
alignment, through its public interface only, against real committed
ASAP fixtures (no synthetic annotation files -- the real dataset already
has both a clean-aligned and a not-aligned example)."""
from __future__ import annotations

from pathlib import Path

from follower_bench.asap_alignment import load_alignment

REPO_ROOT = Path(__file__).resolve().parents[3]
ASAP_ROOT = REPO_ROOT / "model/data/raw/asap-dataset"
ALIGNED_PIECE = "Liszt/Transcendental_Etudes/1/LuoJ05M.mid"


def test_load_alignment_resolves_real_aligned_piece() -> None:
    alignment = load_alignment(ALIGNED_PIECE)
    assert alignment.asap_piece == ALIGNED_PIECE
    assert alignment.performance_midi_path == ASAP_ROOT / ALIGNED_PIECE
    assert alignment.performance_midi_path.exists()
    assert alignment.score_midi_path == ASAP_ROOT / "Liszt/Transcendental_Etudes/1/midi_score.mid"
    assert alignment.score_midi_path.exists()
    assert len(alignment.performance_beats) == len(alignment.midi_score_beats)
    assert len(alignment.performance_beats) == 92
    assert alignment.performance_beats[0] < alignment.performance_beats[-1]
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/follower_bench/test_asap_alignment.py -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'follower_bench.asap_alignment'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# model/src/follower_bench/asap_alignment.py
"""Resolves an ASAP piece identifier to its performance/score MIDI paths
and beat-level alignment, and validates the alignment is usable as the
exact ground-truth substrate for a synthetic clip.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_MODULE_DIR = Path(__file__).resolve()
DEFAULT_ASAP_ROOT = _MODULE_DIR.parents[2] / "data/raw/asap-dataset"
DEFAULT_ANNOTATIONS_PATH = DEFAULT_ASAP_ROOT / "asap_annotations.json"

MIN_BEATS = 4


class AsapAlignmentMissingError(Exception):
    """Raised when an ASAP piece has no usable beat alignment: not present
    in the annotations file, not marked score_and_performance_aligned, or
    has fewer than MIN_BEATS beat anchors (or mismatched
    performance_beats/midi_score_beats lengths)."""


@dataclass(frozen=True)
class ClipAlignment:
    """The ground-truth substrate for one ASAP performance."""
    asap_piece: str
    performance_midi_path: Path
    score_midi_path: Path
    performance_beats: tuple[float, ...]
    midi_score_beats: tuple[float, ...]


def load_alignment(
    asap_piece: str,
    asap_root: Path = DEFAULT_ASAP_ROOT,
    annotations_path: Path = DEFAULT_ANNOTATIONS_PATH,
) -> ClipAlignment:
    """Load and validate the ASAP beat alignment for asap_piece.

    Raises:
        FileNotFoundError: annotations_path does not exist, or the
            resolved performance/score MIDI files do not exist.
        AsapAlignmentMissingError: asap_piece is not a key in the
            annotations file, is not marked score_and_performance_aligned,
            or has fewer than MIN_BEATS beat anchors.
    """
    if not annotations_path.exists():
        raise FileNotFoundError(f"ASAP annotations file not found: {annotations_path}")
    data = json.loads(annotations_path.read_text())
    entry = data.get(asap_piece)
    if entry is None:
        raise AsapAlignmentMissingError(
            f"{asap_piece!r} not found in ASAP annotations: {annotations_path}"
        )
    if not entry.get("score_and_performance_aligned", False):
        raise AsapAlignmentMissingError(
            f"{asap_piece!r} is not marked score_and_performance_aligned"
        )
    perf_beats = entry.get("performance_beats") or []
    score_beats = entry.get("midi_score_beats") or []
    if len(perf_beats) < MIN_BEATS or len(perf_beats) != len(score_beats):
        raise AsapAlignmentMissingError(
            f"{asap_piece!r} has an unusable beat alignment: "
            f"{len(perf_beats)} performance_beats vs {len(score_beats)} midi_score_beats "
            f"(need >= {MIN_BEATS} matched pairs)"
        )
    performance_midi_path = asap_root / asap_piece
    if not performance_midi_path.exists():
        raise FileNotFoundError(f"ASAP performance MIDI not found: {performance_midi_path}")
    score_midi_path = performance_midi_path.parent / "midi_score.mid"
    if not score_midi_path.exists():
        raise FileNotFoundError(f"ASAP score MIDI not found: {score_midi_path}")
    return ClipAlignment(
        asap_piece=asap_piece,
        performance_midi_path=performance_midi_path,
        score_midi_path=score_midi_path,
        performance_beats=tuple(float(b) for b in perf_beats),
        midi_score_beats=tuple(float(b) for b in score_beats),
    )
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/follower_bench/test_asap_alignment.py -q
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd model && git add src/follower_bench/asap_alignment.py tests/follower_bench/test_asap_alignment.py && git commit -m "feat(follower-bench): load_alignment resolves a real aligned ASAP piece (#111)"
```

---

### Task 3: `load_alignment` raises `AsapAlignmentMissingError` for a real not-aligned piece

**Group:** B (sequential, after Task 2)

**Behavior being verified:** a real ASAP entry with `score_and_performance_aligned: false` is rejected, not silently used.
**Interface under test:** `follower_bench.asap_alignment.load_alignment`

**Files:**
- Modify: `model/src/follower_bench/asap_alignment.py` (no code change needed — this test exercises existing Step-3 logic; if the test fails, fix the validation branch)
- Modify: `model/tests/follower_bench/test_asap_alignment.py`

- [ ] **Step 1: Write the failing test**

Add to `model/tests/follower_bench/test_asap_alignment.py`:

```python
import pytest

from follower_bench.asap_alignment import AsapAlignmentMissingError

UNALIGNED_PIECE = "Beethoven/Piano_Sonatas/16-1/LuoJ03M.mid"


def test_load_alignment_rejects_real_unaligned_piece() -> None:
    with pytest.raises(AsapAlignmentMissingError, match="score_and_performance_aligned"):
        load_alignment(UNALIGNED_PIECE)
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/follower_bench/test_asap_alignment.py::test_load_alignment_rejects_real_unaligned_piece -q
```
Expected: this specific test should already PASS given Task 2's implementation (the `score_and_performance_aligned` check already exists). If it unexpectedly FAILS, that means `Beethoven/Piano_Sonatas/16-1/LuoJ03M.mid`'s annotation entry does not have `score_and_performance_aligned: false` any more (dataset drift) — re-pick an unaligned piece via `uv run python3 -c "import json; d=json.load(open('data/raw/asap-dataset/asap_annotations.json')); print(next(k for k,v in d.items() if not v.get('score_and_performance_aligned')))"` and use that key instead, then re-run.

- [ ] **Step 3: Implement the minimum to make the test pass**

No production code change required (Task 2 already implemented the check). If Step 2 showed a FAIL, this step is: update `UNALIGNED_PIECE` in the test to whatever the re-run of the lookup command above prints, then move to Step 4.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/follower_bench/test_asap_alignment.py -q
```
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
cd model && git add tests/follower_bench/test_asap_alignment.py && git commit -m "test(follower-bench): load_alignment rejects a real not-aligned ASAP piece (#111)"
```

---

### Task 4: `load_alignment` raises `AsapAlignmentMissingError` for an unknown piece key

**Group:** B (sequential, after Task 3)

**Behavior being verified:** an `asap_piece` string that is not a key in the annotations file is rejected with the specific skip-worthy exception (not `KeyError`, not silently returning `None`).
**Interface under test:** `follower_bench.asap_alignment.load_alignment`

**Files:**
- Modify: `model/tests/follower_bench/test_asap_alignment.py`

- [ ] **Step 1: Write the failing test**

Add to `model/tests/follower_bench/test_asap_alignment.py`:

```python
def test_load_alignment_rejects_unknown_piece_key() -> None:
    with pytest.raises(AsapAlignmentMissingError, match="not found"):
        load_alignment("Nonexistent/Composer/piece/Nobody99X.mid")
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/follower_bench/test_asap_alignment.py::test_load_alignment_rejects_unknown_piece_key -q
```
Expected: this test should already PASS given Task 2's implementation (the `entry is None` branch). Confirm it passes; this task exists to make the "unknown key" behavior an explicit, permanent regression test per the spec's success criterion ("A piece missing ASAP alignment is SKIPPED with a logged explicit reason ... never fabricated").

- [ ] **Step 3: Implement the minimum to make the test pass**

No production code change required.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/follower_bench/test_asap_alignment.py -q
```
Expected: PASS (all three tests)

- [ ] **Step 5: Commit**

```bash
cd model && git add tests/follower_bench/test_asap_alignment.py && git commit -m "test(follower-bench): load_alignment rejects an unknown ASAP piece key (#111)"
```

---

### Task 5: `apply_segments` passes notes through unchanged for a single identity segment

**Group:** D (sequential: Task 5 → 6 → 7 → 8; depends on Group A; runs in parallel with Group B)

**Behavior being verified:** replaying the whole clip as one `Segment(src_start, src_end, src_start, 1.0)` reproduces the input note stream, sorted by onset.
**Interface under test:** `follower_bench.segments.apply_segments`

**Files:**
- Create: `model/src/follower_bench/segments.py`
- Test: `model/tests/follower_bench/test_segments.py`

- [ ] **Step 1: Write the failing test**

```python
# model/tests/follower_bench/test_segments.py
"""Verify the hard-splice engine (apply_segments / apply_note_mutations)
through its public interface only, on synthetic PerfNote lists."""
from __future__ import annotations

import pytest

from follower_bench.segments import PerfNote, Segment, apply_segments


def test_apply_segments_identity_reproduces_input_notes() -> None:
    notes = [
        PerfNote(onset=0.0, offset=0.5, pitch=60, velocity=80),
        PerfNote(onset=1.0, offset=1.4, pitch=62, velocity=70),
        PerfNote(onset=2.0, offset=2.6, pitch=64, velocity=90),
    ]
    identity = [Segment(src_start=0.0, src_end=3.0, dst_start=0.0, time_scale=1.0)]
    result = apply_segments(notes, identity)
    assert result == notes
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/follower_bench/test_segments.py -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'follower_bench.segments'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# model/src/follower_bench/segments.py
"""Hard-splice engine: rearranges spans of a clean performance's note
stream into a new timeline, and applies note-level pitch mutations. Every
timeline-rearranging pathology (repeat/jump/restart/hesitation/
tempo_swing) is expressed as a list of Segments; wrong_note is a
NoteMutation applied on top of an identity segment.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PerfNote:
    """A single performance note in absolute seconds."""
    onset: float
    offset: float
    pitch: int
    velocity: int


@dataclass(frozen=True)
class Segment:
    """One contiguous span of source (clean) performance time
    [src_start, src_end), replayed starting at dst_start in the new clip,
    optionally time-scaled (dst_duration = (src_end - src_start) *
    time_scale)."""
    src_start: float
    src_end: float
    dst_start: float
    time_scale: float = 1.0

    @property
    def dst_end(self) -> float:
        return self.dst_start + (self.src_end - self.src_start) * self.time_scale


def apply_segments(notes: list[PerfNote], segments: list[Segment]) -> list[PerfNote]:
    """Rebuild a note stream by replaying each segment's source notes
    (onset in the half-open range [src_start, src_end)) at its
    destination time. Segments are applied independently and
    concatenated, then the result is sorted by onset ascending --
    segments may duplicate or omit source notes (repeat/jump), and are
    expected to be given in destination order for a coherent timeline.
    """
    out: list[PerfNote] = []
    for seg in segments:
        for n in notes:
            if seg.src_start <= n.onset < seg.src_end:
                new_onset = seg.dst_start + (n.onset - seg.src_start) * seg.time_scale
                new_duration = (n.offset - n.onset) * seg.time_scale
                out.append(
                    PerfNote(
                        onset=new_onset,
                        offset=new_onset + new_duration,
                        pitch=n.pitch,
                        velocity=n.velocity,
                    )
                )
    out.sort(key=lambda n: n.onset)
    return out
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/follower_bench/test_segments.py -q
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd model && git add src/follower_bench/segments.py tests/follower_bench/test_segments.py && git commit -m "feat(follower-bench): apply_segments identity passthrough (#111)"
```

---

### Task 6: `apply_segments` duplicates a sub-range of notes for a repeat-like splice

**Group:** D (sequential, after Task 5)

**Behavior being verified:** a 3-segment plan (play 0..Y, repeat X..Y, continue Y..end) duplicates the notes in [X,Y) with correctly time-shifted onsets, and shifts the tail notes forward by the repeated span's duration.
**Interface under test:** `follower_bench.segments.apply_segments`

**Files:**
- Modify: `model/tests/follower_bench/test_segments.py`

- [ ] **Step 1: Write the failing test**

Add to `model/tests/follower_bench/test_segments.py`. `Segment` arithmetic used to derive the expected onsets below: `seg1 = Segment(0.0, 2.0, 0.0, 1.0)` so `seg1.dst_end == 2.0`; `seg2 = Segment(1.0, 2.0, 2.0, 1.0)` so a source note at onset 1.0 maps to `new_onset = 2.0 + (1.0 - 1.0) * 1.0 = 2.0`, and one at onset 1.6 maps to `2.0 + (1.6 - 1.0) = 2.6`, and `seg2.dst_end == 3.0`; `seg3 = Segment(2.0, 3.0, 3.0, 1.0)` so the tail note at onset 2.5 maps to `3.0 + (2.5 - 2.0) = 3.5`:

```python
def test_apply_segments_repeat_splice_duplicates_and_shifts_notes() -> None:
    notes = [
        PerfNote(onset=0.0, offset=0.4, pitch=60, velocity=80),   # before X
        PerfNote(onset=1.0, offset=1.4, pitch=62, velocity=80),   # inside [X=1, Y=2)
        PerfNote(onset=1.6, offset=1.9, pitch=64, velocity=80),   # inside [X=1, Y=2)
        PerfNote(onset=2.5, offset=2.9, pitch=65, velocity=80),   # after Y, the tail
    ]
    x, y, t_min, t_max = 1.0, 2.0, 0.0, 3.0
    seg1 = Segment(t_min, y, t_min, 1.0)
    seg2 = Segment(x, y, seg1.dst_end, 1.0)
    seg3 = Segment(y, t_max, seg2.dst_end, 1.0)
    result = apply_segments(notes, [seg1, seg2, seg3])

    onset_pitch_pairs = [(round(n.onset, 6), n.pitch) for n in result]
    assert len(result) == 6  # 4 original notes + 2 duplicated from the repeated span [1.0, 2.0)
    assert (0.0, 60) in onset_pitch_pairs   # seg1: before X, unchanged
    assert (1.0, 62) in onset_pitch_pairs   # seg1: first pass through X..Y
    assert (1.6, 64) in onset_pitch_pairs   # seg1: first pass through X..Y
    assert (2.0, 62) in onset_pitch_pairs   # seg2: repeat pass, X replayed at seg2.dst_start=2.0
    assert (2.6, 64) in onset_pitch_pairs   # seg2: repeat pass
    assert (3.5, 65) in onset_pitch_pairs   # seg3: tail note shifted forward by the repeat's duration
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest "tests/follower_bench/test_segments.py::test_apply_segments_repeat_splice_duplicates_and_shifts_notes" -q
```
Expected: this test should already PASS given Task 5's `apply_segments` implementation, since the algorithm is generic over any segment list. If it fails, the bug is in the test's hand-computed expected onsets, not the implementation — recompute them from the `Segment` arithmetic shown above and fix the assertions. Do not change `apply_segments` unless the failure is a real logic bug (e.g. off-by-one on the half-open boundary).

- [ ] **Step 3: Implement the minimum to make the test pass**

No production code change expected. If Step 2 revealed a genuine `apply_segments` bug (not a test arithmetic error), fix it in `model/src/follower_bench/segments.py` and re-run.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/follower_bench/test_segments.py -q
```
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
cd model && git add tests/follower_bench/test_segments.py && git commit -m "test(follower-bench): apply_segments duplicates a repeat-span with shifted onsets (#111)"
```

---

### Task 7: `apply_note_mutations` substitutes the nearest note's pitch, clamped

**Group:** D (sequential, after Task 6)

**Behavior being verified:** given a `NoteMutation(target_onset, pitch_delta)`, the note whose onset is nearest `target_onset` has its pitch shifted by `pitch_delta` and clamped to `[0, 127]`; all other notes are untouched.
**Interface under test:** `follower_bench.segments.apply_note_mutations`

**Files:**
- Modify: `model/src/follower_bench/segments.py`
- Modify: `model/tests/follower_bench/test_segments.py`

- [ ] **Step 1: Write the failing test**

Add to `model/tests/follower_bench/test_segments.py`:

```python
from follower_bench.segments import NoteMutation, apply_note_mutations


def test_apply_note_mutations_shifts_nearest_note_pitch_clamped() -> None:
    notes = [
        PerfNote(onset=0.0, offset=0.4, pitch=60, velocity=80),
        PerfNote(onset=1.0, offset=1.4, pitch=126, velocity=80),
        PerfNote(onset=2.0, offset=2.4, pitch=64, velocity=80),
    ]
    mutations = [NoteMutation(target_onset=1.05, pitch_delta=5)]
    result = apply_note_mutations(notes, mutations)

    assert result[0].pitch == 60
    assert result[1].pitch == 127  # 126 + 5 clamped to 127
    assert result[2].pitch == 64
    assert result[1].onset == 1.0 and result[1].offset == 1.4 and result[1].velocity == 80
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest "tests/follower_bench/test_segments.py::test_apply_note_mutations_shifts_nearest_note_pitch_clamped" -q
```
Expected: FAIL — `ImportError: cannot import name 'NoteMutation' from 'follower_bench.segments'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Add to `model/src/follower_bench/segments.py`:

```python
@dataclass(frozen=True)
class NoteMutation:
    """Substitute the pitch of the note whose onset is nearest to
    target_onset by pitch_delta semitones, clamped to the MIDI range
    [0, 127]."""
    target_onset: float
    pitch_delta: int


def apply_note_mutations(notes: list[PerfNote], mutations: list[NoteMutation]) -> list[PerfNote]:
    """Apply pitch mutations to the note(s) nearest each mutation's
    target_onset.

    Raises:
        ValueError: notes is empty (nothing to mutate).
    """
    result = list(notes)
    for mut in mutations:
        if not result:
            raise ValueError("cannot apply NoteMutation: note list is empty")
        idx = min(range(len(result)), key=lambda i: abs(result[i].onset - mut.target_onset))
        target = result[idx]
        new_pitch = max(0, min(127, target.pitch + mut.pitch_delta))
        result[idx] = PerfNote(
            onset=target.onset, offset=target.offset, pitch=new_pitch, velocity=target.velocity
        )
    return result
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/follower_bench/test_segments.py -q
```
Expected: PASS (all three tests)

- [ ] **Step 5: Commit**

```bash
cd model && git add src/follower_bench/segments.py tests/follower_bench/test_segments.py && git commit -m "feat(follower-bench): apply_note_mutations substitutes nearest note's pitch (#111)"
```

---

### Task 8: `apply_note_mutations` raises `ValueError` on an empty note list

**Group:** D (sequential, after Task 7)

**Behavior being verified:** mutating an empty note list is an explicit error, not a silent no-op.
**Interface under test:** `follower_bench.segments.apply_note_mutations`

**Files:**
- Modify: `model/tests/follower_bench/test_segments.py`

- [ ] **Step 1: Write the failing test**

Add to `model/tests/follower_bench/test_segments.py`:

```python
def test_apply_note_mutations_raises_on_empty_notes() -> None:
    with pytest.raises(ValueError, match="empty"):
        apply_note_mutations([], [NoteMutation(target_onset=1.0, pitch_delta=1)])
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest "tests/follower_bench/test_segments.py::test_apply_note_mutations_raises_on_empty_notes" -q
```
Expected: this test should already PASS given Task 7's implementation (the `if not result: raise ValueError(...)` guard). Confirm it passes; this task exists to make the empty-input guard a permanent regression test.

- [ ] **Step 3: Implement the minimum to make the test pass**

No production code change required.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/follower_bench/test_segments.py -q
```
Expected: PASS (all four tests)

- [ ] **Step 5: Commit**

```bash
cd model && git add tests/follower_bench/test_segments.py && git commit -m "test(follower-bench): apply_note_mutations rejects an empty note list (#111)"
```

---

### Task 9: `TrueTrajectory.score_position_at` interpolates linearly and clamps

**Group:** C (sequential: Task 9 → 10 → 11 → 12 → 13; depends on Group B AND Group D both being complete)

**Behavior being verified:** given synthetic anchors, `score_position_at` linearly interpolates between the two bracketing anchors and clamps to the first/last anchor's value outside the anchor range.
**Interface under test:** `follower_bench.trajectory.TrueTrajectory.score_position_at`

**Files:**
- Create: `model/src/follower_bench/trajectory.py`
- Test: `model/tests/follower_bench/test_trajectory.py`

- [ ] **Step 1: Write the failing test**

```python
# model/tests/follower_bench/test_trajectory.py
"""Verify TrueTrajectory / from_alignment / build_trajectory_from_segments
through their public interface only."""
from __future__ import annotations

import pytest

from follower_bench.trajectory import TrueTrajectory


def test_score_position_at_interpolates_and_clamps() -> None:
    traj = TrueTrajectory(anchors=((0.0, 0.0), (1.0, 0.5), (2.0, 1.0)))
    assert traj.score_position_at(0.5) == pytest.approx(0.25)
    assert traj.score_position_at(1.5) == pytest.approx(0.75)
    assert traj.score_position_at(-1.0) == pytest.approx(0.0)   # clamp below range
    assert traj.score_position_at(5.0) == pytest.approx(1.0)    # clamp above range
    assert traj.score_position_at(1.0) == pytest.approx(0.5)    # exactly on an anchor
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/follower_bench/test_trajectory.py -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'follower_bench.trajectory'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# model/src/follower_bench/trajectory.py
"""Exact ground-truth mapping from performance time (seconds) to score
position (score beats). Piecewise-linear between explicit anchor points.
Anchors are (perf_time_seconds, score_beat_position) pairs, sorted
ascending by perf_time. A hard-splice discontinuity (repeat/jump/restart)
is represented as two anchors separated by a fixed, tiny time epsilon
with different score positions -- a near-instant transition rather than a
gradual ramp -- so a discontinuity always resolves within
DISCONTINUITY_EPS_S seconds of the injected event's perf_time.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass

from follower_bench.asap_alignment import ClipAlignment
from follower_bench.segments import Segment

DISCONTINUITY_EPS_S = 1e-3


@dataclass(frozen=True)
class TrueTrajectory:
    """Piecewise-linear score-position(perf_time) ground truth."""
    anchors: tuple[tuple[float, float], ...]

    def score_position_at(self, t: float) -> float:
        """Interpolate score position at perf-time t seconds. Clamps to
        the first/last anchor's score position outside the anchors' time
        range."""
        times = [a[0] for a in self.anchors]
        positions = [a[1] for a in self.anchors]
        if t <= times[0]:
            return positions[0]
        if t >= times[-1]:
            return positions[-1]
        i = bisect.bisect_right(times, t) - 1
        t0, p0 = times[i], positions[i]
        t1, p1 = times[i + 1], positions[i + 1]
        if t1 == t0:
            return p0
        frac = (t - t0) / (t1 - t0)
        return p0 + frac * (p1 - p0)

    def is_monotonic_non_decreasing(self) -> bool:
        """True iff score position never decreases as perf_time
        advances."""
        positions = [a[1] for a in self.anchors]
        return all(b >= a for a, b in zip(positions, positions[1:]))


def from_alignment(alignment: ClipAlignment) -> TrueTrajectory:
    """Build the clean (unmodified) trajectory directly from an ASAP
    beat alignment: performance_beats <-> midi_score_beats, zipped
    index-for-index."""
    anchors = tuple(zip(alignment.performance_beats, alignment.midi_score_beats))
    return TrueTrajectory(anchors=anchors)


def build_trajectory_from_segments(
    clean_traj: TrueTrajectory, segments: list[Segment]
) -> TrueTrajectory:
    """Build a spliced trajectory by replaying clean_traj's own anchors
    through each Segment's affine (perf_time -> dst_time) map, in
    destination order. Where consecutive segments are NOT contiguous in
    source time (a hard-splice jump), the later segment's start anchor is
    offset by DISCONTINUITY_EPS_S so the transition resolves as a sharp
    but well-defined ramp rather than an undefined vertical step.
    """
    anchors: list[tuple[float, float]] = []
    prev_seg: Segment | None = None
    for seg in segments:
        contiguous = prev_seg is None or abs(seg.src_start - prev_seg.src_end) < 1e-9
        start_dst = seg.dst_start if contiguous else seg.dst_start + DISCONTINUITY_EPS_S
        anchors.append((start_dst, clean_traj.score_position_at(seg.src_start)))
        for src_t, pos in clean_traj.anchors:
            if seg.src_start < src_t < seg.src_end:
                dst_t = seg.dst_start + (src_t - seg.src_start) * seg.time_scale
                anchors.append((dst_t, pos))
        anchors.append((seg.dst_end, clean_traj.score_position_at(seg.src_end)))
        prev_seg = seg
    anchors.sort(key=lambda a: a[0])
    return TrueTrajectory(anchors=tuple(anchors))
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/follower_bench/test_trajectory.py -q
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd model && git add src/follower_bench/trajectory.py tests/follower_bench/test_trajectory.py && git commit -m "feat(follower-bench): TrueTrajectory.score_position_at interpolates and clamps (#111)"
```

---

### Task 10: `TrueTrajectory.is_monotonic_non_decreasing` detects ordering violations

**Group:** C (sequential, after Task 9)

**Behavior being verified:** returns `True` for ascending score positions, `False` when any later position is smaller than an earlier one.
**Interface under test:** `follower_bench.trajectory.TrueTrajectory.is_monotonic_non_decreasing`

**Files:**
- Modify: `model/tests/follower_bench/test_trajectory.py`

- [ ] **Step 1: Write the failing test**

Add to `model/tests/follower_bench/test_trajectory.py`:

```python
def test_is_monotonic_non_decreasing_true_for_ascending() -> None:
    traj = TrueTrajectory(anchors=((0.0, 0.0), (1.0, 0.5), (2.0, 0.5), (3.0, 1.0)))
    assert traj.is_monotonic_non_decreasing() is True


def test_is_monotonic_non_decreasing_false_for_a_regression() -> None:
    traj = TrueTrajectory(anchors=((0.0, 0.0), (1.0, 1.5), (2.0, 0.5), (3.0, 1.0)))
    assert traj.is_monotonic_non_decreasing() is False
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest "tests/follower_bench/test_trajectory.py::test_is_monotonic_non_decreasing_true_for_ascending" "tests/follower_bench/test_trajectory.py::test_is_monotonic_non_decreasing_false_for_a_regression" -q
```
Expected: these should already PASS given Task 9's implementation. Confirm; this task exists to make the monotonicity check (used directly by the spec's clean-control success criterion) a permanent regression test.

- [ ] **Step 3: Implement the minimum to make the test pass**

No production code change required.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/follower_bench/test_trajectory.py -q
```
Expected: PASS (all tests so far)

- [ ] **Step 5: Commit**

```bash
cd model && git add tests/follower_bench/test_trajectory.py && git commit -m "test(follower-bench): is_monotonic_non_decreasing detects ordering violations (#111)"
```

---

### Task 11: `from_alignment` builds a trajectory matching a real ClipAlignment exactly

**Group:** C (sequential, after Task 10)

**Behavior being verified:** `from_alignment` on the real `Liszt/Transcendental_Etudes/1/LuoJ05M.mid` `ClipAlignment` produces anchors that are exactly the zipped `performance_beats`/`midi_score_beats` arrays.
**Interface under test:** `follower_bench.trajectory.from_alignment`

**Files:**
- Modify: `model/tests/follower_bench/test_trajectory.py`

- [ ] **Step 1: Write the failing test**

Add to `model/tests/follower_bench/test_trajectory.py`:

```python
from follower_bench.asap_alignment import load_alignment
from follower_bench.trajectory import from_alignment

ALIGNED_PIECE = "Liszt/Transcendental_Etudes/1/LuoJ05M.mid"


def test_from_alignment_matches_real_asap_beat_arrays_exactly() -> None:
    alignment = load_alignment(ALIGNED_PIECE)
    traj = from_alignment(alignment)
    assert len(traj.anchors) == len(alignment.performance_beats) == 92
    assert traj.anchors[0] == (alignment.performance_beats[0], alignment.midi_score_beats[0])
    assert traj.anchors[-1] == (alignment.performance_beats[-1], alignment.midi_score_beats[-1])
    assert traj.is_monotonic_non_decreasing() is True
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest "tests/follower_bench/test_trajectory.py::test_from_alignment_matches_real_asap_beat_arrays_exactly" -q
```
Expected: FAIL — `ImportError: cannot import name 'from_alignment' from 'follower_bench.trajectory'` (Task 9's Step 3 already wrote `from_alignment` into `trajectory.py`, so if Task 9 was completed as specified this import succeeds — in that case this specific test should go straight to PASS in Step 2; if it instead fails on the import, `from_alignment` was not yet added and must be added now).

- [ ] **Step 3: Implement the minimum to make the test pass**

`from_alignment` was already implemented in Task 9's Step 3 (`trajectory.py` already contains it). No further code change is needed; this task exists to add the real-fixture regression test.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/follower_bench/test_trajectory.py -q
```
Expected: PASS (all tests so far)

- [ ] **Step 5: Commit**

```bash
cd model && git add tests/follower_bench/test_trajectory.py && git commit -m "test(follower-bench): from_alignment matches real ASAP beat arrays exactly (#111)"
```

---

### Task 12: `build_trajectory_from_segments` reproduces the clean trajectory for one identity segment

**Group:** C (sequential, after Task 11)

**Behavior being verified:** splicing with a single `Segment(src_start, src_end, src_start, 1.0)` (no rearrangement) reproduces `clean_traj`'s own values exactly at every original anchor time — the "clean" pathology's correctness criterion.
**Interface under test:** `follower_bench.trajectory.build_trajectory_from_segments`

**Files:**
- Modify: `model/tests/follower_bench/test_trajectory.py`

- [ ] **Step 1: Write the failing test**

Add to `model/tests/follower_bench/test_trajectory.py`:

```python
from follower_bench.segments import Segment
from follower_bench.trajectory import build_trajectory_from_segments


def test_build_trajectory_from_segments_identity_matches_clean_exactly() -> None:
    clean = TrueTrajectory(anchors=((0.0, 0.0), (1.0, 0.5), (2.0, 1.0), (3.0, 1.5)))
    identity = [Segment(src_start=0.0, src_end=3.0, dst_start=0.0, time_scale=1.0)]
    spliced = build_trajectory_from_segments(clean, identity)
    for t, expected_pos in clean.anchors:
        assert spliced.score_position_at(t) == pytest.approx(expected_pos)
    assert spliced.is_monotonic_non_decreasing() is True
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest "tests/follower_bench/test_trajectory.py::test_build_trajectory_from_segments_identity_matches_clean_exactly" -q
```
Expected: this should already PASS given Task 9's implementation of `build_trajectory_from_segments`. Confirm; this task exists to add the identity-segment regression test explicitly (the "clean" pathology depends on this exact behavior).

- [ ] **Step 3: Implement the minimum to make the test pass**

No production code change required.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/follower_bench/test_trajectory.py -q
```
Expected: PASS (all tests so far)

- [ ] **Step 5: Commit**

```bash
cd model && git add tests/follower_bench/test_trajectory.py && git commit -m "test(follower-bench): build_trajectory_from_segments identity matches clean exactly (#111)"
```

---

### Task 13: `build_trajectory_from_segments` inserts a sharp discontinuity for a non-contiguous splice

**Group:** C (sequential, after Task 12)

**Behavior being verified:** for a two-segment plan where the second segment's `src_start` does NOT equal the first segment's `src_end` (a hard-splice jump, e.g. a `jump` pathology), the resulting trajectory (a) matches the clean correspondence exactly away from the splice, and (b) transitions from the pre-jump score position to the post-jump score position within `DISCONTINUITY_EPS_S` seconds of the injected perf_time.
**Interface under test:** `follower_bench.trajectory.build_trajectory_from_segments`

**Files:**
- Modify: `model/tests/follower_bench/test_trajectory.py`

- [ ] **Step 1: Write the failing test**

Add to `model/tests/follower_bench/test_trajectory.py`:

```python
from follower_bench.trajectory import DISCONTINUITY_EPS_S


def test_build_trajectory_from_segments_jump_is_a_sharp_discontinuity() -> None:
    # clean: perf_time 0..4 <-> score position 0..2 (linear, slope 0.5)
    clean = TrueTrajectory(anchors=((0.0, 0.0), (1.0, 0.5), (2.0, 1.0), (3.0, 1.5), (4.0, 2.0)))
    x, z, t_min, t_max = 1.0, 3.0, 0.0, 4.0  # skip the middle [1.0, 3.0)
    seg1 = Segment(t_min, x, t_min, 1.0)
    seg2 = Segment(z, t_max, seg1.dst_end, 1.0)
    spliced = build_trajectory_from_segments(clean, [seg1, seg2])

    jump_perf_time = seg1.dst_end  # == x == 1.0
    # away from the splice: exact clean correspondence
    assert spliced.score_position_at(0.5) == pytest.approx(clean.score_position_at(0.5))
    # just before the jump: still the pre-jump value (score position at x)
    assert spliced.score_position_at(jump_perf_time) == pytest.approx(clean.score_position_at(x))
    # just after DISCONTINUITY_EPS_S: the post-jump value (score position at z)
    assert spliced.score_position_at(jump_perf_time + DISCONTINUITY_EPS_S) == pytest.approx(
        clean.score_position_at(z)
    )
    # well after the jump, mapped back through seg2's affine map, matches clean exactly
    later_dst_t = seg2.dst_start + 0.5
    later_src_t = z + (later_dst_t - seg2.dst_start)
    assert spliced.score_position_at(later_dst_t) == pytest.approx(clean.score_position_at(later_src_t))
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest "tests/follower_bench/test_trajectory.py::test_build_trajectory_from_segments_jump_is_a_sharp_discontinuity" -q
```
Expected: this should already PASS given Task 9's implementation. Confirm; this task exists to lock in the exact-discontinuity-width behavior as a permanent regression test, since it is the mechanism every non-clean pathology's ground truth depends on.

- [ ] **Step 3: Implement the minimum to make the test pass**

No production code change required.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/follower_bench/test_trajectory.py -q
```
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
cd model && git add tests/follower_bench/test_trajectory.py && git commit -m "test(follower-bench): build_trajectory_from_segments jump is a sharp discontinuity (#111)"
```

---

### Task 14: `build_plan("clean", ...)` returns one identity segment and no events

**Group:** E (sequential: Task 14 → 15 → ... → 22; depends on Groups B, C, D all complete)

**Behavior being verified:** the control pathology produces exactly one segment spanning the whole aligned range with no rearrangement, and zero pathology events.
**Interface under test:** `follower_bench.pathologies.build_plan`

**Files:**
- Create: `model/src/follower_bench/pathologies.py`
- Test: `model/tests/follower_bench/test_pathologies.py`

- [ ] **Step 1: Write the failing test**

```python
# model/tests/follower_bench/test_pathologies.py
"""Verify build_plan's per-pathology-type Segment/PathologyEvent
construction through its public interface only, on a real ClipAlignment."""
from __future__ import annotations

import random

import pytest

from follower_bench.asap_alignment import load_alignment
from follower_bench.pathologies import PATHOLOGY_TYPES, build_plan

ALIGNED_PIECE = "Liszt/Transcendental_Etudes/1/LuoJ05M.mid"


def _alignment():
    return load_alignment(ALIGNED_PIECE)


def test_build_plan_clean_is_one_identity_segment_no_events() -> None:
    alignment = _alignment()
    plan = build_plan(alignment, "clean", random.Random(0))
    t_min, t_max = alignment.performance_beats[0], alignment.performance_beats[-1]

    assert len(plan.segments) == 1
    seg = plan.segments[0]
    assert seg.src_start == pytest.approx(t_min)
    assert seg.src_end == pytest.approx(t_max)
    assert seg.dst_start == pytest.approx(t_min)
    assert seg.time_scale == pytest.approx(1.0)
    assert plan.events == ()
    assert plan.note_mutations == ()
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/follower_bench/test_pathologies.py -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'follower_bench.pathologies'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# model/src/follower_bench/pathologies.py
"""Per-pathology-type construction of a ClipPlan: a deterministic (given
rng) Segment list that rearranges a clean ASAP performance's timeline,
plus the PathologyEvent labels describing what was injected and where.
wrong_note additionally carries a NoteMutation (pitch substitution, no
timeline rearrangement) instead of a rearranging Segment list.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from follower_bench.asap_alignment import ClipAlignment
from follower_bench.segments import NoteMutation, Segment
from follower_bench.trajectory import from_alignment

PATHOLOGY_TYPES = (
    "clean",
    "repeat",
    "jump",
    "restart",
    "hesitation",
    "wrong_note",
    "tempo_swing",
)


@dataclass(frozen=True)
class PathologyEvent:
    """One injected pathology event. For pure timeline jumps (repeat,
    jump, restart), from_score_position != to_score_position. For
    pathologies that do not change score position (hesitation,
    wrong_note, tempo_swing), from_score_position == to_score_position."""
    type: str
    perf_time: float
    from_score_position: float
    to_score_position: float


@dataclass(frozen=True)
class ClipPlan:
    segments: tuple[Segment, ...]
    events: tuple[PathologyEvent, ...]
    note_mutations: tuple[NoteMutation, ...] = ()


def _bounds(alignment: ClipAlignment) -> tuple[float, float]:
    t_min = alignment.performance_beats[0]
    t_max = alignment.performance_beats[-1]
    if t_max <= t_min:
        raise ValueError(
            f"{alignment.asap_piece}: zero-duration performance ({t_min}..{t_max}), cannot splice"
        )
    return t_min, t_max


def build_plan(alignment: ClipAlignment, pathology_type: str, rng: random.Random) -> ClipPlan:
    """Build the ClipPlan for pathology_type.

    Raises:
        ValueError: pathology_type is not one of PATHOLOGY_TYPES, or the
            alignment's beat range is zero-duration.
    """
    if pathology_type not in PATHOLOGY_TYPES:
        raise ValueError(f"Unknown pathology_type {pathology_type!r}; must be one of {PATHOLOGY_TYPES}")

    t_min, t_max = _bounds(alignment)

    if pathology_type == "clean":
        return ClipPlan(segments=(Segment(t_min, t_max, t_min, 1.0),), events=())

    raise NotImplementedError(f"pathology_type {pathology_type!r} not yet implemented")
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/follower_bench/test_pathologies.py -q
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd model && git add src/follower_bench/pathologies.py tests/follower_bench/test_pathologies.py && git commit -m "feat(follower-bench): build_plan clean control is one identity segment (#111)"
```

---

### Task 15: `build_plan("repeat", ...)` describes the back-jump correctly

**Group:** E (sequential, after Task 14)

**Behavior being verified:** the repeat pathology returns 3 segments (forward through Y, repeat X..Y, continue past Y) and exactly one `PathologyEvent` whose `perf_time` is where the repeat begins, and whose `from_score_position`/`to_score_position` describe jumping from Y's score position back to X's.
**Interface under test:** `follower_bench.pathologies.build_plan`

**Files:**
- Modify: `model/src/follower_bench/pathologies.py`
- Modify: `model/tests/follower_bench/test_pathologies.py`

- [ ] **Step 1: Write the failing test**

Add to `model/tests/follower_bench/test_pathologies.py`:

```python
from follower_bench.trajectory import from_alignment


def test_build_plan_repeat_describes_the_back_jump() -> None:
    alignment = _alignment()
    plan = build_plan(alignment, "repeat", random.Random(0))
    clean_traj = from_alignment(alignment)

    assert len(plan.segments) == 3
    seg1, seg2, seg3 = plan.segments
    t_min, t_max = alignment.performance_beats[0], alignment.performance_beats[-1]
    assert seg1.src_start == pytest.approx(t_min)
    assert seg1.dst_start == pytest.approx(t_min)
    assert seg2.src_start < seg1.src_end  # X < Y
    assert seg2.src_end == pytest.approx(seg1.src_end)  # both end at Y
    assert seg2.dst_start == pytest.approx(seg1.dst_end)  # repeat pass starts right after the first pass
    assert seg3.src_start == pytest.approx(seg1.src_end)  # continue from Y
    assert seg3.src_end == pytest.approx(t_max)

    assert len(plan.events) == 1
    event = plan.events[0]
    assert event.type == "repeat"
    assert event.perf_time == pytest.approx(seg1.dst_end)
    assert event.from_score_position == pytest.approx(clean_traj.score_position_at(seg1.src_end))
    assert event.to_score_position == pytest.approx(clean_traj.score_position_at(seg2.src_start))
    assert event.to_score_position < event.from_score_position  # jumped backward in score position
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest "tests/follower_bench/test_pathologies.py::test_build_plan_repeat_describes_the_back_jump" -q
```
Expected: FAIL — `NotImplementedError: pathology_type 'repeat' not yet implemented`

- [ ] **Step 3: Implement the minimum to make the test pass**

Add to `model/src/follower_bench/pathologies.py`, above the `_bounds` docstring-adjacent constants near the top (module-level, after `PATHOLOGY_TYPES`):

```python
def _pick_two_points(alignment: ClipAlignment, rng: random.Random) -> tuple[float, float]:
    """Pick two ordered perf-time points inside the piece's beat-anchored
    range: a in [15%, 35%) of duration, b in [55%, 75%) of duration. The
    fixed, non-overlapping bands guarantee t_min < a < b < t_max for any
    duration > 0."""
    t_min, t_max = _bounds(alignment)
    duration = t_max - t_min
    a = t_min + rng.uniform(0.15, 0.35) * duration
    b = t_min + rng.uniform(0.55, 0.75) * duration
    return a, b
```

Replace the `raise NotImplementedError(...)` line in `build_plan` with:

```python
    clean_traj = from_alignment(alignment)

    if pathology_type == "repeat":
        x, y = _pick_two_points(alignment, rng)
        seg1 = Segment(t_min, y, t_min, 1.0)
        seg2 = Segment(x, y, seg1.dst_end, 1.0)
        seg3 = Segment(y, t_max, seg2.dst_end, 1.0)
        event = PathologyEvent(
            type="repeat",
            perf_time=seg1.dst_end,
            from_score_position=clean_traj.score_position_at(y),
            to_score_position=clean_traj.score_position_at(x),
        )
        return ClipPlan(segments=(seg1, seg2, seg3), events=(event,))

    raise NotImplementedError(f"pathology_type {pathology_type!r} not yet implemented")
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/follower_bench/test_pathologies.py -q
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd model && git add src/follower_bench/pathologies.py tests/follower_bench/test_pathologies.py && git commit -m "feat(follower-bench): build_plan repeat describes the back-jump (#111)"
```

---

### Task 16: `build_plan("jump", ...)` describes the forward skip correctly

**Group:** E (sequential, after Task 15)

**Behavior being verified:** the jump pathology returns 2 segments (play up to X, skip ahead to Z and continue) and one event describing a FORWARD score-position jump.
**Interface under test:** `follower_bench.pathologies.build_plan`

**Files:**
- Modify: `model/src/follower_bench/pathologies.py`
- Modify: `model/tests/follower_bench/test_pathologies.py`

- [ ] **Step 1: Write the failing test**

Add to `model/tests/follower_bench/test_pathologies.py`:

```python
def test_build_plan_jump_describes_the_forward_skip() -> None:
    alignment = _alignment()
    plan = build_plan(alignment, "jump", random.Random(0))
    clean_traj = from_alignment(alignment)
    t_min, t_max = alignment.performance_beats[0], alignment.performance_beats[-1]

    assert len(plan.segments) == 2
    seg1, seg2 = plan.segments
    assert seg1.src_start == pytest.approx(t_min)
    assert seg1.dst_start == pytest.approx(t_min)
    assert seg2.src_start > seg1.src_end  # Z > X: the omitted middle
    assert seg2.dst_start == pytest.approx(seg1.dst_end)
    assert seg2.src_end == pytest.approx(t_max)

    assert len(plan.events) == 1
    event = plan.events[0]
    assert event.type == "jump"
    assert event.perf_time == pytest.approx(seg1.dst_end)
    assert event.from_score_position == pytest.approx(clean_traj.score_position_at(seg1.src_end))
    assert event.to_score_position == pytest.approx(clean_traj.score_position_at(seg2.src_start))
    assert event.to_score_position > event.from_score_position  # jumped forward, omitting the middle
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest "tests/follower_bench/test_pathologies.py::test_build_plan_jump_describes_the_forward_skip" -q
```
Expected: FAIL — `NotImplementedError: pathology_type 'jump' not yet implemented`

- [ ] **Step 3: Implement the minimum to make the test pass**

In `model/src/follower_bench/pathologies.py`, add this branch above the final `raise NotImplementedError`:

```python
    if pathology_type == "jump":
        x, z = _pick_two_points(alignment, rng)
        seg1 = Segment(t_min, x, t_min, 1.0)
        seg2 = Segment(z, t_max, seg1.dst_end, 1.0)
        event = PathologyEvent(
            type="jump",
            perf_time=seg1.dst_end,
            from_score_position=clean_traj.score_position_at(x),
            to_score_position=clean_traj.score_position_at(z),
        )
        return ClipPlan(segments=(seg1, seg2), events=(event,))
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/follower_bench/test_pathologies.py -q
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd model && git add src/follower_bench/pathologies.py tests/follower_bench/test_pathologies.py && git commit -m "feat(follower-bench): build_plan jump describes the forward skip (#111)"
```

---

### Task 17: `build_plan("restart", ...)` jumps back to an arbitrary earlier point

**Group:** E (sequential, after Task 16)

**Behavior being verified:** the restart pathology returns 2 segments (play up to Y, jump back to an earlier point R and continue to the end) and one event describing a backward jump to R (which may be much earlier than a typical repeat span).
**Interface under test:** `follower_bench.pathologies.build_plan`

**Files:**
- Modify: `model/src/follower_bench/pathologies.py`
- Modify: `model/tests/follower_bench/test_pathologies.py`

- [ ] **Step 1: Write the failing test**

Add to `model/tests/follower_bench/test_pathologies.py`:

```python
def test_build_plan_restart_jumps_back_to_an_earlier_point() -> None:
    alignment = _alignment()
    plan = build_plan(alignment, "restart", random.Random(0))
    clean_traj = from_alignment(alignment)
    t_min, t_max = alignment.performance_beats[0], alignment.performance_beats[-1]

    assert len(plan.segments) == 2
    seg1, seg2 = plan.segments
    assert seg1.src_start == pytest.approx(t_min)
    assert seg1.dst_start == pytest.approx(t_min)
    assert seg2.src_start < seg1.src_end  # R < Y
    assert seg2.dst_start == pytest.approx(seg1.dst_end)
    assert seg2.src_end == pytest.approx(t_max)  # restart plays through to the end, not back to Y

    assert len(plan.events) == 1
    event = plan.events[0]
    assert event.type == "restart"
    assert event.perf_time == pytest.approx(seg1.dst_end)
    assert event.from_score_position == pytest.approx(clean_traj.score_position_at(seg1.src_end))
    assert event.to_score_position == pytest.approx(clean_traj.score_position_at(seg2.src_start))
    assert event.to_score_position < event.from_score_position
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest "tests/follower_bench/test_pathologies.py::test_build_plan_restart_jumps_back_to_an_earlier_point" -q
```
Expected: FAIL — `NotImplementedError: pathology_type 'restart' not yet implemented`

- [ ] **Step 3: Implement the minimum to make the test pass**

In `model/src/follower_bench/pathologies.py`, add this branch above the final `raise NotImplementedError`:

```python
    if pathology_type == "restart":
        r, y = _pick_two_points(alignment, rng)
        seg1 = Segment(t_min, y, t_min, 1.0)
        seg2 = Segment(r, t_max, seg1.dst_end, 1.0)
        event = PathologyEvent(
            type="restart",
            perf_time=seg1.dst_end,
            from_score_position=clean_traj.score_position_at(y),
            to_score_position=clean_traj.score_position_at(r),
        )
        return ClipPlan(segments=(seg1, seg2), events=(event,))
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/follower_bench/test_pathologies.py -q
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd model && git add src/follower_bench/pathologies.py tests/follower_bench/test_pathologies.py && git commit -m "feat(follower-bench): build_plan restart jumps back to an earlier point (#111)"
```

---

### Task 18: `build_plan("hesitation", ...)` inserts a same-position pause

**Group:** E (sequential, after Task 17)

**Behavior being verified:** the hesitation pathology returns 2 segments with a destination-time GAP equal to the inserted pause (no segment covers that gap), and one event whose `from_score_position == to_score_position` (score position does not change, only time passes).
**Interface under test:** `follower_bench.pathologies.build_plan`

**Files:**
- Modify: `model/src/follower_bench/pathologies.py`
- Modify: `model/tests/follower_bench/test_pathologies.py`

- [ ] **Step 1: Write the failing test**

Add to `model/tests/follower_bench/test_pathologies.py`:

```python
def test_build_plan_hesitation_inserts_a_same_position_pause() -> None:
    alignment = _alignment()
    plan = build_plan(alignment, "hesitation", random.Random(0))
    clean_traj = from_alignment(alignment)
    t_min, t_max = alignment.performance_beats[0], alignment.performance_beats[-1]

    assert len(plan.segments) == 2
    seg1, seg2 = plan.segments
    assert seg1.src_start == pytest.approx(t_min)
    assert seg2.src_start == pytest.approx(seg1.src_end)  # SAME score position resumed, no src jump
    assert seg2.src_end == pytest.approx(t_max)
    pause = seg2.dst_start - seg1.dst_end
    assert 1.0 <= pause <= 3.0  # a real pause, not instantaneous and not absurdly long

    assert len(plan.events) == 1
    event = plan.events[0]
    assert event.type == "hesitation"
    assert event.perf_time == pytest.approx(seg1.dst_end)
    assert event.from_score_position == pytest.approx(event.to_score_position)
    assert event.from_score_position == pytest.approx(clean_traj.score_position_at(seg1.src_end))
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest "tests/follower_bench/test_pathologies.py::test_build_plan_hesitation_inserts_a_same_position_pause" -q
```
Expected: FAIL — `NotImplementedError: pathology_type 'hesitation' not yet implemented`

- [ ] **Step 3: Implement the minimum to make the test pass**

Add these module-level constants near `PATHOLOGY_TYPES` in `model/src/follower_bench/pathologies.py`:

```python
_HESITATION_PAUSE_MIN_S = 1.0
_HESITATION_PAUSE_MAX_S = 3.0
```

Add this branch above the final `raise NotImplementedError`:

```python
    if pathology_type == "hesitation":
        duration = t_max - t_min
        p = t_min + rng.uniform(0.3, 0.7) * duration
        pause = rng.uniform(_HESITATION_PAUSE_MIN_S, _HESITATION_PAUSE_MAX_S)
        seg1 = Segment(t_min, p, t_min, 1.0)
        seg2 = Segment(p, t_max, seg1.dst_end + pause, 1.0)
        pos = clean_traj.score_position_at(p)
        event = PathologyEvent(
            type="hesitation", perf_time=seg1.dst_end, from_score_position=pos, to_score_position=pos
        )
        return ClipPlan(segments=(seg1, seg2), events=(event,))
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/follower_bench/test_pathologies.py -q
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd model && git add src/follower_bench/pathologies.py tests/follower_bench/test_pathologies.py && git commit -m "feat(follower-bench): build_plan hesitation inserts a same-position pause (#111)"
```

---

### Task 19: `build_plan("wrong_note", ...)` carries a NoteMutation with no timeline change

**Group:** E (sequential, after Task 18)

**Behavior being verified:** the wrong_note pathology returns a single identity segment (score position never changes) plus exactly one `NoteMutation`, and one event with `from_score_position == to_score_position`.
**Interface under test:** `follower_bench.pathologies.build_plan`

**Files:**
- Modify: `model/src/follower_bench/pathologies.py`
- Modify: `model/tests/follower_bench/test_pathologies.py`

- [ ] **Step 1: Write the failing test**

Add to `model/tests/follower_bench/test_pathologies.py`:

```python
def test_build_plan_wrong_note_is_a_pitch_mutation_with_no_timeline_change() -> None:
    alignment = _alignment()
    plan = build_plan(alignment, "wrong_note", random.Random(0))
    clean_traj = from_alignment(alignment)
    t_min, t_max = alignment.performance_beats[0], alignment.performance_beats[-1]

    assert len(plan.segments) == 1
    seg = plan.segments[0]
    assert seg.src_start == pytest.approx(t_min)
    assert seg.src_end == pytest.approx(t_max)
    assert seg.time_scale == pytest.approx(1.0)

    assert len(plan.note_mutations) == 1
    mutation = plan.note_mutations[0]
    assert t_min <= mutation.target_onset <= t_max
    assert mutation.pitch_delta != 0

    assert len(plan.events) == 1
    event = plan.events[0]
    assert event.type == "wrong_note"
    assert event.from_score_position == pytest.approx(event.to_score_position)
    assert event.from_score_position == pytest.approx(clean_traj.score_position_at(mutation.target_onset))
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest "tests/follower_bench/test_pathologies.py::test_build_plan_wrong_note_is_a_pitch_mutation_with_no_timeline_change" -q
```
Expected: FAIL — `NotImplementedError: pathology_type 'wrong_note' not yet implemented`

- [ ] **Step 3: Implement the minimum to make the test pass**

Add this module-level constant near `PATHOLOGY_TYPES` in `model/src/follower_bench/pathologies.py`:

```python
_WRONG_NOTE_PITCH_DELTAS = (-2, -1, 1, 2)
```

Add this branch above the final `raise NotImplementedError`:

```python
    if pathology_type == "wrong_note":
        duration = t_max - t_min
        p = t_min + rng.uniform(0.3, 0.7) * duration
        delta = rng.choice(_WRONG_NOTE_PITCH_DELTAS)
        pos = clean_traj.score_position_at(p)
        event = PathologyEvent(
            type="wrong_note", perf_time=p, from_score_position=pos, to_score_position=pos
        )
        mutation = NoteMutation(target_onset=p, pitch_delta=delta)
        return ClipPlan(
            segments=(Segment(t_min, t_max, t_min, 1.0),),
            events=(event,),
            note_mutations=(mutation,),
        )
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/follower_bench/test_pathologies.py -q
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd model && git add src/follower_bench/pathologies.py tests/follower_bench/test_pathologies.py && git commit -m "feat(follower-bench): build_plan wrong_note is a pitch mutation, no timeline change (#111)"
```

---

### Task 20: `build_plan("tempo_swing", ...)` produces a contiguous piecewise tempo ramp

**Group:** E (sequential, after Task 19)

**Behavior being verified:** the tempo_swing pathology returns N+2 segments (lead-in, N sub-segments spanning the swing window with linearly varying `time_scale`, tail) that are fully contiguous in source time (no score-position jump — a nonlinear TIME warp only), and one event with `from_score_position == to_score_position`.
**Interface under test:** `follower_bench.pathologies.build_plan`

**Files:**
- Modify: `model/src/follower_bench/pathologies.py`
- Modify: `model/tests/follower_bench/test_pathologies.py`

- [ ] **Step 1: Write the failing test**

Add to `model/tests/follower_bench/test_pathologies.py`:

```python
def test_build_plan_tempo_swing_is_a_contiguous_piecewise_time_ramp() -> None:
    alignment = _alignment()
    plan = build_plan(alignment, "tempo_swing", random.Random(0))
    clean_traj = from_alignment(alignment)
    t_min, t_max = alignment.performance_beats[0], alignment.performance_beats[-1]

    # lead-in + N sub-segments + tail; N is an implementation constant >= 2
    assert len(plan.segments) >= 4
    segs = plan.segments
    assert segs[0].src_start == pytest.approx(t_min)
    assert segs[0].time_scale == pytest.approx(1.0)
    assert segs[-1].src_end == pytest.approx(t_max)
    assert segs[-1].time_scale == pytest.approx(1.0)

    # fully contiguous in SOURCE time: no segment introduces a score-position jump
    for prev, curr in zip(segs, segs[1:]):
        assert curr.src_start == pytest.approx(prev.src_end)
        assert curr.dst_start == pytest.approx(prev.dst_end)

    # at least one interior sub-segment has a non-1.0 time_scale (the actual swing)
    assert any(s.time_scale != pytest.approx(1.0) for s in segs[1:-1])

    assert len(plan.events) == 1
    event = plan.events[0]
    assert event.type == "tempo_swing"
    assert event.from_score_position == pytest.approx(event.to_score_position)
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest "tests/follower_bench/test_pathologies.py::test_build_plan_tempo_swing_is_a_contiguous_piecewise_time_ramp" -q
```
Expected: FAIL — `NotImplementedError: pathology_type 'tempo_swing' not yet implemented`

- [ ] **Step 3: Implement the minimum to make the test pass**

Add these module-level constants near `PATHOLOGY_TYPES` in `model/src/follower_bench/pathologies.py`:

```python
_TEMPO_SWING_SUBSEGMENTS = 4
_TEMPO_SWING_SCALE_START = 1.3
_TEMPO_SWING_SCALE_END = 0.8
```

Replace the final `raise NotImplementedError(...)` line in `build_plan` with:

```python
    # tempo_swing
    x, y = _pick_two_points(alignment, rng)
    pos_x = clean_traj.score_position_at(x)
    n = _TEMPO_SWING_SUBSEGMENTS
    scales = [
        _TEMPO_SWING_SCALE_START + (_TEMPO_SWING_SCALE_END - _TEMPO_SWING_SCALE_START) * i / (n - 1)
        for i in range(n)
    ]
    sub_bounds = [x + (y - x) * i / n for i in range(n + 1)]
    segments = [Segment(t_min, x, t_min, 1.0)]
    for i in range(n):
        prev = segments[-1]
        segments.append(Segment(sub_bounds[i], sub_bounds[i + 1], prev.dst_end, scales[i]))
    tail_start = segments[-1].dst_end
    segments.append(Segment(y, t_max, tail_start, 1.0))
    event = PathologyEvent(
        type="tempo_swing", perf_time=x, from_score_position=pos_x, to_score_position=pos_x
    )
    return ClipPlan(segments=tuple(segments), events=(event,))
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/follower_bench/test_pathologies.py -q
```
Expected: PASS (all pathology-type tests)

- [ ] **Step 5: Commit**

```bash
cd model && git add src/follower_bench/pathologies.py tests/follower_bench/test_pathologies.py && git commit -m "feat(follower-bench): build_plan tempo_swing is a contiguous piecewise time ramp (#111)"
```

---

### Task 21: `build_plan` raises `ValueError` for an unknown pathology_type

**Group:** E (sequential, after Task 20)

**Behavior being verified:** an invalid `pathology_type` string is rejected immediately, listing the valid options, rather than falling through silently.
**Interface under test:** `follower_bench.pathologies.build_plan`

**Files:**
- Modify: `model/tests/follower_bench/test_pathologies.py`

- [ ] **Step 1: Write the failing test**

Add to `model/tests/follower_bench/test_pathologies.py`:

```python
def test_build_plan_rejects_unknown_pathology_type() -> None:
    alignment = _alignment()
    with pytest.raises(ValueError, match="Unknown pathology_type"):
        build_plan(alignment, "does_not_exist", random.Random(0))
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest "tests/follower_bench/test_pathologies.py::test_build_plan_rejects_unknown_pathology_type" -q
```
Expected: this should already PASS given the `if pathology_type not in PATHOLOGY_TYPES: raise ValueError(...)` guard written in Task 14's Step 3. Confirm; this task exists to make it a permanent regression test.

- [ ] **Step 3: Implement the minimum to make the test pass**

No production code change required.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/follower_bench/test_pathologies.py -q
```
Expected: PASS (all tests so far)

- [ ] **Step 5: Commit**

```bash
cd model && git add tests/follower_bench/test_pathologies.py && git commit -m "test(follower-bench): build_plan rejects an unknown pathology_type (#111)"
```

---

### Task 22: `build_plan` raises `ValueError` for a zero-duration alignment

**Group:** E (sequential, after Task 21)

**Behavior being verified:** an alignment whose first and last `performance_beats` are equal (degenerate, zero-duration) is rejected explicitly rather than producing a division-by-zero or a nonsensical clip.
**Interface under test:** `follower_bench.pathologies.build_plan`

**Files:**
- Modify: `model/tests/follower_bench/test_pathologies.py`

- [ ] **Step 1: Write the failing test**

Add to `model/tests/follower_bench/test_pathologies.py`:

```python
from follower_bench.asap_alignment import ClipAlignment


def test_build_plan_rejects_zero_duration_alignment() -> None:
    degenerate = ClipAlignment(
        asap_piece="fake/degenerate.mid",
        performance_midi_path=_alignment().performance_midi_path,
        score_midi_path=_alignment().score_midi_path,
        performance_beats=(1.0, 1.0, 1.0, 1.0),
        midi_score_beats=(0.0, 0.5, 1.0, 1.5),
    )
    with pytest.raises(ValueError, match="zero-duration"):
        build_plan(degenerate, "repeat", random.Random(0))
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest "tests/follower_bench/test_pathologies.py::test_build_plan_rejects_zero_duration_alignment" -q
```
Expected: this should already PASS given `_bounds`'s `if t_max <= t_min: raise ValueError(...)` guard written in Task 14's Step 3. Confirm; this task exists to make it a permanent regression test with a directly-constructed degenerate `ClipAlignment` (no real ASAP fixture is degenerate, so this must be synthesized).

- [ ] **Step 3: Implement the minimum to make the test pass**

No production code change required.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/follower_bench/test_pathologies.py -q
```
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
cd model && git add tests/follower_bench/test_pathologies.py && git commit -m "test(follower-bench): build_plan rejects a zero-duration alignment (#111)"
```

---

### Task 23: `generate("clean", ...)` produces a monotonic trajectory matching ASAP exactly

**Group:** F (sequential: Task 23 → 24 → 25 → 26; depends on Group E complete)

**Behavior being verified:** end-to-end, `generate()` on the real aligned fixture piece with `pathology_type="clean"` returns a `SynthClip` whose `true_trajectory` is monotonic non-decreasing and matches the ASAP beat alignment exactly at every annotated beat — the spec's clean-control success criterion.
**Interface under test:** `follower_bench.clip_generator.generate`

**Files:**
- Create: `model/src/follower_bench/clip_generator.py`
- Test: `model/tests/follower_bench/test_clip_generator.py`

- [ ] **Step 1: Write the failing test**

```python
# model/tests/follower_bench/test_clip_generator.py
"""End-to-end verification of generate() through its public interface
only, on real ASAP fixtures."""
from __future__ import annotations

import pytest

from follower_bench.asap_alignment import AsapAlignmentMissingError, load_alignment
from follower_bench.clip_generator import generate

ALIGNED_PIECE = "Liszt/Transcendental_Etudes/1/LuoJ05M.mid"
UNALIGNED_PIECE = "Beethoven/Piano_Sonatas/16-1/LuoJ03M.mid"


def test_generate_clean_is_monotonic_and_matches_asap_exactly() -> None:
    clip = generate(ALIGNED_PIECE, "clean", seed=1)
    alignment = load_alignment(ALIGNED_PIECE)

    assert clip.asap_piece == ALIGNED_PIECE
    assert clip.pathology_type == "clean"
    assert clip.seed == 1
    assert len(clip.notes) > 0
    assert clip.event_labels == ()
    assert clip.true_trajectory.is_monotonic_non_decreasing() is True

    for perf_beat, score_beat in zip(alignment.performance_beats, alignment.midi_score_beats):
        assert clip.true_trajectory.score_position_at(perf_beat) == pytest.approx(score_beat)
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/follower_bench/test_clip_generator.py -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'follower_bench.clip_generator'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# model/src/follower_bench/clip_generator.py
"""Public entrypoint for the synthetic score-follower benchmark (issue
#111): given an ASAP piece identifier and a pathology type, produce a
pathology-injected performance note stream together with its exact
ground-truth score-position trajectory and the labels of what was
injected. Composes asap_alignment (truth substrate) + pathologies
(splice plan) + segments (splice engine) + trajectory (ground truth)
behind one call.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import partitura as pa

from follower_bench.asap_alignment import load_alignment
from follower_bench.pathologies import PathologyEvent, build_plan
from follower_bench.segments import PerfNote, apply_note_mutations, apply_segments
from follower_bench.trajectory import TrueTrajectory, build_trajectory_from_segments, from_alignment


@dataclass(frozen=True)
class SynthClip:
    """One generated benchmark clip: the pathology-injected note stream,
    its exact ground-truth score-position trajectory, and the injected
    pathology event labels. `notes` is an in-memory note stream
    (onset/offset/pitch/velocity in seconds); MIDI-file serialization is
    out of scope for #111 and is added by #112 when needed."""
    asap_piece: str
    pathology_type: str
    seed: int
    notes: tuple[PerfNote, ...]
    true_trajectory: TrueTrajectory
    event_labels: tuple[PathologyEvent, ...]


def _load_perf_notes(path: Path) -> list[PerfNote]:
    ppart = pa.load_performance_midi(str(path))
    note_array = ppart.note_array()
    return [
        PerfNote(
            onset=float(row["onset_sec"]),
            offset=float(row["onset_sec"] + row["duration_sec"]),
            pitch=int(row["pitch"]),
            velocity=int(row["velocity"]),
        )
        for row in note_array
    ]


def generate(asap_piece: str, pathology_type: str, seed: int) -> SynthClip:
    """Generate one pathology-injected clip for asap_piece.

    Raises:
        AsapAlignmentMissingError: asap_piece has no usable ASAP beat
            alignment (propagated from asap_alignment.load_alignment) --
            the caller (a batch driver) is expected to catch this and
            skip the piece with a logged reason, never fabricate a
            trajectory.
        FileNotFoundError: the resolved MIDI files are missing on disk.
        ValueError: pathology_type is not a known PATHOLOGY_TYPES member,
            or the piece's beat range is zero-duration.
    """
    alignment = load_alignment(asap_piece)
    rng = random.Random(seed)
    plan = build_plan(alignment, pathology_type, rng)
    clean_traj = from_alignment(alignment)

    notes = _load_perf_notes(alignment.performance_midi_path)
    spliced = apply_segments(notes, list(plan.segments))
    spliced = apply_note_mutations(spliced, list(plan.note_mutations))

    trajectory = build_trajectory_from_segments(clean_traj, list(plan.segments))

    return SynthClip(
        asap_piece=asap_piece,
        pathology_type=pathology_type,
        seed=seed,
        notes=tuple(spliced),
        true_trajectory=trajectory,
        event_labels=plan.events,
    )
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/follower_bench/test_clip_generator.py -q
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd model && git add src/follower_bench/clip_generator.py tests/follower_bench/test_clip_generator.py && git commit -m "feat(follower-bench): generate() clean control matches ASAP exactly (#111)"
```

---

### Task 24: `generate("repeat", ...)` is deterministic and has the back-jump at the right time

**Group:** F (sequential, after Task 23)

**Behavior being verified:** two calls to `generate()` with the same `(asap_piece, pathology_type, seed)` produce byte-identical notes and trajectories (determinism), and the repeat's back-jump appears in `true_trajectory` at the event's `perf_time`.
**Interface under test:** `follower_bench.clip_generator.generate`

**Files:**
- Modify: `model/tests/follower_bench/test_clip_generator.py`

- [ ] **Step 1: Write the failing test**

Add to `model/tests/follower_bench/test_clip_generator.py`:

```python
from follower_bench.trajectory import DISCONTINUITY_EPS_S


def test_generate_repeat_is_deterministic_and_has_the_back_jump() -> None:
    clip_a = generate(ALIGNED_PIECE, "repeat", seed=7)
    clip_b = generate(ALIGNED_PIECE, "repeat", seed=7)

    assert clip_a.notes == clip_b.notes
    assert clip_a.true_trajectory.anchors == clip_b.true_trajectory.anchors
    assert clip_a.event_labels == clip_b.event_labels

    assert len(clip_a.event_labels) == 1
    event = clip_a.event_labels[0]
    assert event.type == "repeat"

    before = clip_a.true_trajectory.score_position_at(event.perf_time)
    after = clip_a.true_trajectory.score_position_at(event.perf_time + DISCONTINUITY_EPS_S)
    assert before == pytest.approx(event.from_score_position)
    assert after == pytest.approx(event.to_score_position)
    assert after < before  # confirmed backward jump
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest "tests/follower_bench/test_clip_generator.py::test_generate_repeat_is_deterministic_and_has_the_back_jump" -q
```
Expected: this should already PASS given Task 23's `generate()` implementation composed with the already-verified `build_plan("repeat", ...)` and `build_trajectory_from_segments`. Confirm; this task exists to lock the end-to-end determinism + discontinuity-timing contract as a permanent regression test.

- [ ] **Step 3: Implement the minimum to make the test pass**

No production code change required.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/follower_bench/test_clip_generator.py -q
```
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
cd model && git add tests/follower_bench/test_clip_generator.py && git commit -m "test(follower-bench): generate() repeat is deterministic with the back-jump at perf_time (#111)"
```

---

### Task 25: `generate` propagates `AsapAlignmentMissingError` for a real unaligned piece

**Group:** F (sequential, after Task 24)

**Behavior being verified:** requesting a real ASAP piece with no usable alignment raises `AsapAlignmentMissingError` all the way through `generate()` — the exact "SKIPPED, never fabricated" contract from the spec.
**Interface under test:** `follower_bench.clip_generator.generate`

**Files:**
- Modify: `model/tests/follower_bench/test_clip_generator.py`

- [ ] **Step 1: Write the failing test**

Add to `model/tests/follower_bench/test_clip_generator.py`:

```python
def test_generate_propagates_missing_alignment_for_real_unaligned_piece() -> None:
    with pytest.raises(AsapAlignmentMissingError):
        generate(UNALIGNED_PIECE, "clean", seed=1)
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest "tests/follower_bench/test_clip_generator.py::test_generate_propagates_missing_alignment_for_real_unaligned_piece" -q
```
Expected: this should already PASS given Task 23's `generate()` calling `load_alignment` first with no `try/except` swallowing the exception. Confirm; this task exists to make the spec's central "never fabricate, always raise/skip" contract a permanent regression test at the public `generate()` boundary (not just inside `asap_alignment.py`).

- [ ] **Step 3: Implement the minimum to make the test pass**

No production code change required.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/follower_bench/test_clip_generator.py -q
```
Expected: PASS (all tests so far)

- [ ] **Step 5: Commit**

```bash
cd model && git add tests/follower_bench/test_clip_generator.py && git commit -m "test(follower-bench): generate() propagates AsapAlignmentMissingError, never fabricates (#111)"
```

---

### Task 26: Property test — non-injected regions preserve the ASAP correspondence, across all 6 pathologies

**Group:** F (sequential, after Task 25)

**Behavior being verified:** for every non-clean pathology type, generated on the real aligned fixture piece, the trajectory strictly BEFORE the first injected event's `perf_time` matches the clean ASAP correspondence exactly (not an approximation) — the spec's dedicated property-test success criterion, run across all 6 pathology types in one parametrized test.
**Interface under test:** `follower_bench.clip_generator.generate`

**Files:**
- Modify: `model/tests/follower_bench/test_clip_generator.py`

- [ ] **Step 1: Write the failing test**

Add to `model/tests/follower_bench/test_clip_generator.py`:

```python
from follower_bench.trajectory import from_alignment

NON_CLEAN_PATHOLOGY_TYPES = ("repeat", "jump", "restart", "hesitation", "wrong_note", "tempo_swing")


@pytest.mark.parametrize("pathology_type", NON_CLEAN_PATHOLOGY_TYPES)
def test_generate_preserves_asap_correspondence_before_the_injected_event(pathology_type: str) -> None:
    clip = generate(ALIGNED_PIECE, pathology_type, seed=3)
    alignment = load_alignment(ALIGNED_PIECE)
    clean_traj = from_alignment(alignment)

    assert len(clip.event_labels) == 1
    first_event_time = clip.event_labels[0].perf_time

    # Sample perf_time strictly before the injected event: for every
    # pathology type, the region BEFORE the first event replays the clean
    # performance's own opening span unchanged (dst_start == src_start,
    # time_scale == 1.0 for the lead-in segment), so the trajectory must
    # match the clean ASAP correspondence exactly there.
    sample_time = first_event_time / 2.0
    assert sample_time > alignment.performance_beats[0]
    assert clip.true_trajectory.score_position_at(sample_time) == pytest.approx(
        clean_traj.score_position_at(sample_time)
    )
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest "tests/follower_bench/test_clip_generator.py::test_generate_preserves_asap_correspondence_before_the_injected_event" -q
```
Expected: this should already PASS for all 6 parametrized cases given the existing `build_plan`/`build_trajectory_from_segments` implementations (every pathology's first segment is `Segment(t_min, ..., t_min, 1.0)` — an identity lead-in). Confirm all 6 pass; this task exists to make the spec's dedicated "Property test: in non-injected regions of any clip, the trajectory preserves the ASAP correspondence" success criterion an explicit, permanent, parametrized regression test spanning every pathology type in one place.

- [ ] **Step 3: Implement the minimum to make the test pass**

No production code change required. If any of the 6 parametrized cases fails, that is a real bug in the corresponding `build_plan` branch (not a test issue) — fix the branch in `model/src/follower_bench/pathologies.py` so its lead-in segment is a true identity replay of `[t_min, first_split_point)`, then re-run.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/follower_bench/ -q
```
Expected: PASS — the full `follower_bench` test suite (all tasks) is green.

- [ ] **Step 5: Commit**

```bash
cd model && git add tests/follower_bench/test_clip_generator.py && git commit -m "test(follower-bench): property test -- non-injected regions preserve ASAP correspondence across all pathologies (#111)"
```

---

## Post-plan note for the build agent

After Task 26, run the full suite once more from `model/` to confirm nothing regressed:

```bash
cd model && uv run pytest tests/follower_bench/ -v
```

Do not touch `model/data/`, `model/src/piece_id_eval/`, or any other existing package — this plan's diff is scoped entirely to `model/src/follower_bench/`, `model/tests/follower_bench/`, and the single `pyproject.toml` line from Task 1.

---

## Challenge Review

### CEO Pass

**Premise Challenge.** Right problem, direct path. No labeled score-follower data exists; manufacturing it from ASAP's already-verified beat alignment via mechanical splicing (rather than a bespoke per-pathology trajectory model) is the simplest approach that stays exact by construction. Confirmed against the actual repo: `model/data/raw/asap-dataset/asap_annotations.json` exists in the primary checkout, the `Liszt/Transcendental_Etudes/1/LuoJ05M.mid` entry has 92 index-aligned `performance_beats`/`midi_score_beats` pairs with `score_and_performance_aligned: true`, and `Beethoven/Piano_Sonatas/16-1/LuoJ03M.mid` has `score_and_performance_aligned: false` — both "verified facts" in the plan check out exactly (down to the 40.7s / 92-beat claim: `41.742708 - 0.99375 ≈ 40.75s`). No existing code in `model/src/` already does this (checked `piece_id_eval`, `chroma_dtw_eval` — neither builds pathology-injected clips), so this is genuinely new, not a duplicate of existing coverage.

**Scope Check.** Scope matches the spec exactly: MIDI-space only, no audio/AMT/metric/follower/harness, no disk serialization, no batch driver — all correctly deferred to #112-#116 per the design doc's explicit "Not in scope" list and "Alternatives considered" section. 5 new files, 1 modified line — well under the 8-file/2-service complexity-smell threshold. The hardest problem (keeping the spliced note stream and the spliced ground-truth trajectory from drifting apart) is not avoided — it's solved head-on by driving both off the identical `Segment` list, which is the plan's central design decision and is followed through consistently in every task.

**Twelve-Month Alignment.**
```
CURRENT STATE                          THIS PLAN                        12-MONTH IDEAL
No labeled pathology data;         →   follower_bench: deterministic  →  Full EPIC #108 pipeline:
score-follower work (#108) has         MIDI-space clip generator with     synthetic + eventually real
no ground truth to build against       exact splice-derived ground         pathology-injected clips,
                                        truth (7 pathology types)          audio rendering, AMT, metric,
                                                                            follower, harness — all
                                                                            consuming this substrate
```
Moves cleanly toward the ideal; no tech debt identified that conflicts with it.

**Alternatives Check.** Spec documents 3 alternatives with rejection reasons (synthetic hesitation ramps / probabilistic tempo curves; disk serialization inside `generate()`; a batch corpus-generation script). Satisfies this requirement — no `[QUESTION]` needed.

### Engineering Pass

**Architecture / data flow.**
```
generate(asap_piece, pathology_type, seed)
   ├── load_alignment(asap_piece)         → ClipAlignment (raises AsapAlignmentMissingError / FileNotFoundError)
   ├── build_plan(alignment, type, rng)   → ClipPlan (Segments + PathologyEvents [+ NoteMutation])
   ├── _load_perf_notes(path)             → partitura.load_performance_midi(...).note_array()
   ├── apply_segments + apply_note_mutations → spliced notes
   └── build_trajectory_from_segments(from_alignment(alignment), plan.segments) → TrueTrajectory
                                          ↓
                                    error path: exceptions propagate uncaught to the caller
                                    (by design — spec's "never fabricate, always raise/skip" contract)
```
Verified `partitura.load_performance_midi(...).note_array()` against the real fixture MIDI: field names are exactly `onset_sec, duration_sec, onset_tick, duration_tick, pitch, velocity, track, channel, id` as the plan's "Verified facts" section and Task 23's `_load_perf_notes` claim. Also verified the two time bases the design depends on are actually consistent: the fixture's note onsets (0.87s–42.86s) and its `performance_beats` (0.99s–41.74s) sit in the same absolute-second coordinate system (beats span a subset of the note range, as expected for beat-tracked annotations vs. raw note onsets) — this is the one presumption that could have silently broken the whole design and it holds.

**Module Depth Audit.**
- `asap_alignment.py` — interface: `load_alignment()` + `ClipAlignment` + `AsapAlignmentMissingError`. Hides JSON parsing, path resolution, 4-way validation. DEEP.
- `trajectory.py` — interface: `TrueTrajectory` (2 methods) + `from_alignment()` + `build_trajectory_from_segments()`. Hides piecewise-linear interpolation, clamping, and the anchor-carryover/discontinuity-epsilon arithmetic that keeps the trajectory exact through splices. DEEP.
- `segments.py` — interface: `PerfNote`, `Segment` (+`.dst_end`), `NoteMutation`, `apply_segments()`, `apply_note_mutations()` — 5 exports over ~50 LOC of fairly direct list-comprehension arithmetic. Interface size is close to implementation size. Borderline SHALLOW, but justified: these are the foundational data types every other module imports, and splitting them further would just move the same arithmetic around. Not blocking.
- `pathologies.py` — interface: `PATHOLOGY_TYPES`, `PathologyEvent`, `ClipPlan`, `build_plan()`. Hides the 7-branch dispatch and seeded splice-point selection (~150 LOC). DEEP.
- `clip_generator.py` — interface: `SynthClip`, `generate()`. Hides MIDI loading and composition of the other four modules. DEEP.

**Code Quality / Failure Modes.** No catch-all exception handling anywhere — every raised exception is specific (`FileNotFoundError`, `AsapAlignmentMissingError`, `ValueError`) and propagates uncaught, matching CLAUDE.md's "explicit exception handling over silent fallbacks." One silent-failure risk found in `build_plan` (Task 20): the final branch is an un-guarded `else`-by-omission — the code drops straight from the `hesitation`/`wrong_note` checks into tempo_swing-building code with no `if pathology_type == "tempo_swing":` guard, relying on the fact that `PATHOLOGY_TYPES` currently has exactly 7 members and 6 are explicitly matched above it. If an 8th pathology type is ever appended to `PATHOLOGY_TYPES` without adding a matching `if` branch, `build_plan` would silently build a tempo_swing-shaped plan for it instead of raising — a real silent-failure path, mitigated only by the fact that adding a type would presumably come with its own task/test. See `[RISK]` below.

**Test Philosophy.** All tests exercise public interfaces only (`load_alignment`, `apply_segments`/`apply_note_mutations`, `TrueTrajectory` methods, `build_plan`, `generate`) against real committed ASAP fixtures or synthetic-but-direct dataclass construction — no mocking of internal collaborators anywhere, and no shape-only tests (every test asserts on computed values, not just presence/type). This is a real strength and matches the repo's existing `piece_id_eval`/`chroma_dtw_eval` conventions.

**Vertical Slice Audit.** Most tasks are clean one-test/one-implementation/one-commit slices. However, three tasks bundle more implementation than their own task's test covers, with the remaining behavior only locked in as tests several tasks later:
- Task 2 implements `load_alignment` with 4 validation branches (entry-not-found, not-aligned, insufficient-beats/length-mismatch, missing MIDI files) but its own test only exercises the happy path; the other branches ship with zero test coverage until Tasks 3 and 4.
- Task 9 implements `score_position_at`, `is_monotonic_non_decreasing`, `from_alignment`, AND `build_trajectory_from_segments` (including the discontinuity-epsilon logic) in one commit, but its own test only exercises `score_position_at`; the other three are untested until Tasks 10-13.
- Task 14 implements the `PATHOLOGY_TYPES`-membership `ValueError` guard and the zero-duration `ValueError` guard in the same commit as the "clean" branch, but those guards are untested until Tasks 21 and 22.

This is the mirror image of the named "missing implementation" anti-pattern (test now, implementation later) — here it's implementation now, dedicated test later. It doesn't fit either forbidden pattern in the letter (no task defers its own test to a future task; each task's own behavior is tested in that task), but it does mean each of these three commits temporarily ships untested branches, which cuts against CLAUDE.md's "strict TDD, watch-it-fail discipline." Given these are small, deterministic, pure-function guard clauses that are naturally written together and get test coverage within the same task group before Group boundaries close, I'm rating this `[RISK]` rather than `[BLOCKER]` — but flagging it because a stricter reading of the plan skill's vertical-slice rule would require it.

**Test Coverage Gaps.**
```
[+] segments.py
    │
    ├── apply_segments()
    │   ├── [TESTED] ★★  identity passthrough — Task 5
    │   ├── [TESTED] ★★  duplicate a repeated span — Task 6
    │   └── [GAP]        a segment list with an OMITTED source range (jump/restart's
    │                     dropped middle) is never directly asserted at the note level —
    │                     only indirectly implied by the jump/restart trajectory tests
    │                     and the Task 26 property test, neither of which counts notes
    │
    └── apply_note_mutations()
        ├── [TESTED] ★★  nearest-note substitution, clamped — Task 7
        └── [TESTED] ★★  empty-notes ValueError — Task 8

[+] trajectory.py
    └── build_trajectory_from_segments()
        ├── [TESTED] ★★  identity (clean) — Task 12
        ├── [TESTED] ★★★ src-discontinuous jump, sharp transition — Task 13
        └── [GAP]        a DESTINATION-time gap with CONTIGUOUS source (the
                          hesitation shape) is never directly tested; reading the
                          `contiguous` check (compares src_start/src_end only) shows
                          it correctly falls into the "no discontinuity offset" branch,
                          so the trajectory stays flat through the pause — but this
                          isn't asserted anywhere, only Task 18's from==to score
                          position at the event boundary, not mid-pause sampling
```
Neither gap is on a critical path (auth/payments/irreversible-operation equivalent) — this is offline synthetic-data generation — so both are `[RISK]`, not `[BLOCKER]`.

**Failure Modes.** Every raised exception is specific and uncaught; no transaction boundaries apply (pure in-memory computation, no writes). Task 22's synthetic degenerate `ClipAlignment` is the only way to hit the zero-duration guard since no real ASAP entry is degenerate — correctly identified in the plan as needing direct construction rather than a real fixture.

**Operational blocker (found by direct verification, not in the plan's own claims).** `model/data/raw` is entirely gitignored in this repo (`model/.gitignore` contains `data/raw`, with only `data/raw/room_irs/README.md` tracked). Git worktrees do not share untracked/ignored files with the primary checkout. I verified directly: the primary checkout at `/Users/jdhiman/Documents/crescendai/model/data/raw/asap-dataset/` has the full ASAP dataset (annotations file + both fixture pieces' MIDI files), but `/Users/jdhiman/Documents/crescendai/.worktrees/issue-111-clip-generator/model/data/raw/` has no `asap-dataset` directory at all. Every task from Task 2 onward depends on real files under this path (`asap_annotations.json`, the fixture MIDIs) — none of them will find it. This is exactly the gotcha already recorded in this project's own memory (`project_worktree_fullstack_gotchas.md`: "copy WASM pkg/ + gitignored data from primary" when running a worktree's full stack) — it was known, and the plan doesn't carry it forward. As written, Task 2's test will fail with `FileNotFoundError: ASAP annotations file not found`, not the plan's stated expected `ModuleNotFoundError`, and every subsequent task fails the same way. `[BLOCKER]`.

### Presumption Inventory

| ASSUMPTION | VERDICT | REASON |
|---|---|---|
| ASAP annotations JSON schema (`performance_beats`, `midi_score_beats`, `score_and_performance_aligned`) matches the plan's description | SAFE | Verified directly against the real file for both fixture pieces. |
| `partitura.load_performance_midi(...).note_array()` field names (`onset_sec`, `duration_sec`, `pitch`, `velocity`, ...) | SAFE | Verified by running it against the real fixture MIDI in this repo's `uv` environment. |
| Note-stream absolute time base and annotation `performance_beats` absolute time base are the same coordinate system | SAFE | Verified: note onsets (0.87s-42.86s) and performance_beats (0.99s-41.74s) overlap consistently, beats nested inside the note range as expected. |
| The isolated worktree has the ASAP dataset available under `model/data/raw/asap-dataset/` | RISKY | Verified FALSE — `data/raw` is gitignored and the worktree's copy is empty; no plan step copies it in. |
| `model/pyproject.toml`'s current `packages` list is exactly what Task 1 shows before the edit | SAFE | Verified byte-for-byte against the real file. |
| `build_plan`'s final branch will only ever be reached by `tempo_swing` | RISKY | True today (7 types, 6 explicit `if`s), but structurally an un-guarded fallthrough rather than an explicit check — silent misclassification risk if `PATHOLOGY_TYPES` grows without a matching branch. |

### Summary
[BLOCKER] count: 1
[RISK]    count: 4
[QUESTION] count: 0

VERDICT: NEEDS_REWORK — Add an explicit setup step (before Task 2, likely folded into Task 1 or a new "Task 0") that copies or symlinks `model/data/raw/asap-dataset/` from the primary checkout into this worktree, since `data/raw` is gitignored and worktrees do not inherit untracked files — without it every task from Task 2 onward fails on `FileNotFoundError` instead of the plan's expected outcomes. Everything else surfaced ([RISK] items: the implementation-before-dedicated-test pattern in Tasks 2/9/14, the untested note-level omission behavior for jump/restart splices, the untested mid-pause trajectory flatness for hesitation, and the un-guarded tempo_swing fallthrough) is real but non-blocking — safe to proceed once the data-availability fix is made.

---

## Re-Challenge Review (2026-07-06)

**Scope of this pass:** verify the newly added Task 0 (ASAP dataset symlink, dynamically resolved via `git rev-parse --git-common-dir`) actually resolves the sole prior `[BLOCKER]`. Direct verification performed against the real worktree and primary checkout (not re-derived from reading the plan text alone):

1. `cd .worktrees/issue-111-clip-generator && git rev-parse --git-common-dir` → `/Users/jdhiman/Documents/crescendai/.git`; `dirname` of that → `/Users/jdhiman/Documents/crescendai`, the correct primary checkout root, confirmed independently against `git worktree list --porcelain` (primary listed first, no `branch refs/heads/issue-111-clip-generator` line).
2. The primary checkout has the full dataset at `model/data/raw/asap-dataset/` (`asap_annotations.json`, 44.8MB; both fixture MIDIs present and non-empty).
3. Ran Task 0's exact Step 1 and Step 2 commands live in the actual worktree (`.worktrees/issue-111-clip-generator/model`): `data/raw` did not yet contain `asap-dataset`; after `mkdir -p data/raw && ln -s "$PRIMARY_ROOT/model/data/raw/asap-dataset" data/raw/asap-dataset`, all three Step-2 existence checks printed the expected `OK` lines (`annotations OK`, `ALIGNED_PIECE OK`, `UNALIGNED_PIECE OK`).
4. `git status --short data/raw` after creating the symlink produced empty output — confirms `data/raw` is genuinely gitignored in this worktree (the repo's `.gitignore` covers it via a top-level `data/` rule plus a `raw/` rule, not verbatim the "`data/raw`" line the plan's prose paraphrases — a wording nuance, not a functional gap) and the symlink will not get committed.
5. Removed the test symlink afterward to leave the worktree exactly as `/build` will find it (no residue from this verification).
6. Re-verified `model/pyproject.toml`'s current `packages` list is still byte-for-byte what Task 1's Step 3 shows before the edit — unchanged since the prior pass.

**Conclusion:** the operational blocker is genuinely resolved. Task 0 is solo, runs before Group A, uses a dynamically-resolved path (not a hardcoded worktree name), fails loudly and gives a recovery path (`git worktree list --porcelain`) if `PRIMARY_ROOT` resolution ever lands on a non-primary checkout, and correctly avoids committing anything. No new blockers introduced by Task 0 itself — it is inspection-only, not TDD (correctly, per the plan's own admission: "There is no TDD test for this step").

No new issues found beyond the four `[RISK]`s already on record from the prior pass (implementation-before-dedicated-test pattern in Tasks 2/9/14; untested note-level omission for jump/restart; untested mid-pause trajectory flatness for hesitation; un-guarded tempo_swing fallthrough) — all still stand as non-blocking, previously assessed in detail above.

### Summary (re-challenge)
[BLOCKER] count: 0
[RISK]    count: 4 (carried over, unchanged)
[QUESTION] count: 0

VERDICT: PROCEED_WITH_CAUTION — Monitor during `/build`: (1) Tasks 2/9/14 ship a few validation branches ahead of their dedicated regression test (resolved 1-2 tasks later, not a correctness risk, just a stricter-TDD nit); (2) no test directly asserts note-level omission at the `apply_segments` level for jump/restart's dropped middle span (only indirectly covered via trajectory/property tests); (3) hesitation's mid-pause trajectory flatness is untested (only boundary values are asserted); (4) `build_plan`'s final `tempo_swing` branch is an unguarded fallthrough rather than an explicit `if` — safe today (7 types, 6 explicit checks) but would silently misclassify an 8th pathology type added without a matching branch.
