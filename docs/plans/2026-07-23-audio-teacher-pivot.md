# Audio-Native Teacher Pivot (Wave 1) Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** Stand up the gated audio-native-teacher program — epic + Gate 0 issue + contingency issue + old-plan migration — and ship an offline-testable Gate 0 probe harness (`model/src/audio_teacher/`) that the user can later run against Tinker/Inkling under a $50 hard cap.
**Spec:** docs/specs/2026-07-23-audio-teacher-pivot-design.md
**Style:** Follow the project's coding standards (CLAUDE.md, model/CLAUDE.md). No emojis. Explicit exceptions, no silent fallbacks. `__file__`-anchored default paths, never CWD-relative.

**Branch / issue:** `issue-127-audio-teacher-pivot`; the eventual merge closes #127.

**Environment notes for every code task:**
- All commands run from the worktree's `model/` directory: `cd model && ...`.
- First run in a fresh worktree: `cd model && uv sync --dev` (creates the worktree's own venv; mostly hardlinked from the uv cache). Do NOT point `UV_PROJECT_ENVIRONMENT` at the primary checkout's venv — its editable install resolves packages from the primary `src/`, which lacks `audio_teacher`.
- Test command shape: `uv run python -m pytest tests/audio_teacher/<file> -v` (pytest addopts add coverage; that is fine).
- No test may import `tinker`, `tinker_cookbook`, or `tml_renderers`, and no test may perform network I/O.

**Out of scope (do not add):** Gate 1+ issues or training code, contrastive rendering engine, grounding/verifier code, pedagogy training, the Bradley-Terry comparator baseline, production deploys, running the probe against Tinker.

## Task Groups

```
Group A (sequential, bootstrap):        Task 1
Group B (parallel, depends on A):       Task 2, Task 3, Task 4
Group C (parallel, depends on B):       Task 5, Task 6, Task 7
Group D (parallel, depends on C):       Task 8, Task 10, Task 12
Group E (parallel, depends on D):       Task 9, Task 11
Group F (parallel, depends on E):       Task 13, Task 14
Group G (sequential, depends on F):     Task 15
Group H (sequential, depends on G):     Task 16
Group I (sequential, independent of code groups; may run concurrently with A-H):
                                        Task 17 -> Task 18 -> Task 19   [SHIPS INDEPENDENTLY]
```

Group I ships standalone value: with only Group I done, the issue graph already reflects the pivot (epic live, old plans closed, Gate 0 criteria locked). Groups A–H are one vertical deliverable (the harness); Group H's offline CLI e2e is the integration proof.

No two tasks in the same parallel group touch the same file (checked in self-review): `prompts.py`/`test_prompts.py` — Task 4 (B) then Task 5 (C); `manifest.py`/`test_manifest.py` — Task 6 (C), Task 8 (D), Task 9 (E); `scorer.py`/`test_scorer.py` — Task 10 (D), Task 11 (E), Task 13 (F); `tinker_client.py` — Task 12 (D) only; `probe.py`/`test_probe.py` — Task 15 (G) then Task 16 (H). Within Group D, Task 8 (manifest), Task 10 (scorer), and Task 12 (tinker_client) are disjoint; within Group E, Task 9 (manifest) and Task 11 (scorer) are disjoint.

---

### Task 1: Bootstrap package + valid-WAV validation
**Group:** A (runs alone; everything else depends on it)

**Behavior being verified:** A well-formed mono 16 kHz PCM WAV passes `validate_wav` and reports correct rate/frames/duration.
**Interface under test:** `audio_teacher.audio.validate_wav(path, expected_sample_rate) -> WavInfo`

**Files:**
- Modify: `model/pyproject.toml`
- Create: `model/src/audio_teacher/__init__.py`
- Create: `model/src/audio_teacher/audio.py`
- Create: `model/tests/audio_teacher/__init__.py`
- Create: `model/tests/audio_teacher/conftest.py`
- Test: `model/tests/audio_teacher/test_audio.py`

- [ ] **Step 1: Write the failing test**

Create `model/tests/audio_teacher/__init__.py` (empty file) and `model/tests/audio_teacher/conftest.py`:

```python
"""Shared generated fixtures for audio_teacher tests.

Everything is generated under tmp_path -- no binary fixture files are
committed to the repo.
"""
from __future__ import annotations

import struct
import wave
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def wav_factory(tmp_path):
    """Write a PCM-16 silence WAV under tmp_path and return its path."""

    def _write(
        rel: str,
        *,
        sample_rate: int = 16000,
        channels: int = 1,
        seconds: float = 1.0,
    ) -> Path:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        nframes = int(sample_rate * seconds)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(
                struct.pack(f"<{nframes * channels}h", *([0] * (nframes * channels)))
            )
        return path

    return _write


@pytest.fixture
def manifest_factory(tmp_path, wav_factory):
    """Build a loadable probe manifest YAML rooted at tmp_path.

    Each entry in `pairs` is a dict with keys id / axis / population /
    degraded (all optional except id); the two clip WAVs are generated
    automatically. Returns the manifest path. Load with
    load_manifest(path, repo_root=tmp_path).
    """

    def _build(pairs: list[dict], *, sample_rate: int = 16000) -> Path:
        entries = []
        for p in pairs:
            a = wav_factory(f"clips/{p['id']}_a.wav", sample_rate=sample_rate)
            b = wav_factory(f"clips/{p['id']}_b.wav", sample_rate=sample_rate)
            entries.append(
                {
                    "id": p["id"],
                    "axis": p.get("axis", "pedaling"),
                    "population": p.get("population", "real"),
                    "clip_a": str(a.relative_to(tmp_path)),
                    "clip_b": str(b.relative_to(tmp_path)),
                    "degraded": p.get("degraded", "a"),
                    "description": "test contrast",
                }
            )
        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text(
            yaml.safe_dump(
                {"schema_version": 1, "sample_rate": sample_rate, "pairs": entries},
                sort_keys=False,
            )
        )
        return manifest_path

    return _build
```

Create `model/tests/audio_teacher/test_audio.py`:

```python
"""WAV validation behavior through validate_wav's public interface."""
from __future__ import annotations

import pytest

from audio_teacher.audio import validate_wav


def test_valid_mono_wav_passes_validation(wav_factory):
    path = wav_factory("clips/ok.wav", sample_rate=16000, seconds=2.0)
    info = validate_wav(path, expected_sample_rate=16000)
    assert info.sample_rate == 16000
    assert info.num_frames == 32000
    assert info.duration_seconds == pytest.approx(2.0)
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run python -m pytest tests/audio_teacher/test_audio.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'audio_teacher'`

- [ ] **Step 3: Implement the minimum to make the test pass**

In `model/pyproject.toml`, change:

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/score_alignment", "src/audio_experiments", "src/model_improvement", "src/masterclass_experiments", "src/score_library", "src/exercise_corpus", "src/chroma_dtw_eval", "src/piece_id_eval", "src/follower_bench"]
```

to:

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/score_alignment", "src/audio_experiments", "src/model_improvement", "src/masterclass_experiments", "src/score_library", "src/exercise_corpus", "src/chroma_dtw_eval", "src/piece_id_eval", "src/follower_bench", "src/audio_teacher"]
```

Create `model/src/audio_teacher/__init__.py`:

```python
"""Gate 0 audio-native-teacher probe harness (issue #127 pivot).

Contrast-pair manifest in -> Inkling A/B judgments -> deterministic
population-partitioned report. Offline-testable end to end; the Tinker
client is isolated behind the ProbeClient protocol.
"""
```

Create `model/src/audio_teacher/audio.py`:

```python
"""WAV header validation for probe clips."""
from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path


class MalformedClipError(Exception):
    """A probe clip failed WAV validation. The message always names the file."""


@dataclass(frozen=True)
class WavInfo:
    path: Path
    sample_rate: int
    num_frames: int
    duration_seconds: float


def validate_wav(path: Path | str, expected_sample_rate: int) -> WavInfo:
    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate()
        nframes = wf.getnframes()
    return WavInfo(
        path=Path(path),
        sample_rate=rate,
        num_frames=nframes,
        duration_seconds=nframes / rate,
    )
```

Then `cd model && uv sync --dev` so the editable install picks up the new package.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run python -m pytest tests/audio_teacher/test_audio.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add model/pyproject.toml model/uv.lock model/src/audio_teacher/__init__.py model/src/audio_teacher/audio.py model/tests/audio_teacher/ && git commit -m "feat(audio-teacher): bootstrap Gate 0 probe package + WAV info parsing (#127)"
```

---

### Task 2: Malformed WAVs abort naming the file
**Group:** B (parallel with Task 3, Task 4)

**Behavior being verified:** Stereo, wrong-sample-rate, and truncated clips each raise `MalformedClipError` whose message names the offending file.
**Interface under test:** `audio_teacher.audio.validate_wav`

**Files:**
- Modify: `model/src/audio_teacher/audio.py`
- Test: `model/tests/audio_teacher/test_audio.py`

- [ ] **Step 1: Write the failing test**

Append to `model/tests/audio_teacher/test_audio.py`:

```python
def _make_stereo(wav_factory):
    return wav_factory("clips/stereo.wav", channels=2)


def _make_wrong_rate(wav_factory):
    return wav_factory("clips/rate44k.wav", sample_rate=44100)


def _make_truncated(wav_factory):
    path = wav_factory("clips/trunc.wav", seconds=1.0)
    data = path.read_bytes()
    path.write_bytes(data[: len(data) - 1000])
    return path


@pytest.mark.parametrize(
    "make_bad", [_make_stereo, _make_wrong_rate, _make_truncated],
    ids=["stereo", "wrong_rate", "truncated"],
)
def test_malformed_wav_aborts_naming_the_file(wav_factory, make_bad):
    from audio_teacher.audio import MalformedClipError

    path = make_bad(wav_factory)
    with pytest.raises(MalformedClipError) as excinfo:
        validate_wav(path, expected_sample_rate=16000)
    assert path.name in str(excinfo.value)
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run python -m pytest tests/audio_teacher/test_audio.py -v
```
Expected: FAIL — the three new parametrized cases raise nothing (or a raw `wave.Error`), so `pytest.raises(MalformedClipError)` reports `DID NOT RAISE` / wrong exception type. `test_valid_mono_wav_passes_validation` still passes.

- [ ] **Step 3: Implement the minimum to make the test pass**

Replace `validate_wav` in `model/src/audio_teacher/audio.py` with:

```python
def validate_wav(path: Path | str, expected_sample_rate: int) -> WavInfo:
    """Parse and validate a probe clip: readable RIFF/WAVE, mono, expected
    sample rate, non-empty, not truncated (declared frames all present)."""
    path = Path(path)
    try:
        with wave.open(str(path), "rb") as wf:
            channels = wf.getnchannels()
            rate = wf.getframerate()
            nframes = wf.getnframes()
            sampwidth = wf.getsampwidth()
            payload = wf.readframes(nframes)
    except (wave.Error, EOFError) as exc:
        raise MalformedClipError(f"{path}: not a readable WAV file ({exc})") from exc

    if channels != 1:
        raise MalformedClipError(f"{path}: expected mono, got {channels} channels")
    if rate != expected_sample_rate:
        raise MalformedClipError(
            f"{path}: expected sample rate {expected_sample_rate}, got {rate}"
        )
    if nframes == 0:
        raise MalformedClipError(f"{path}: zero-length audio")
    expected_bytes = nframes * channels * sampwidth
    if len(payload) != expected_bytes:
        raise MalformedClipError(
            f"{path}: truncated audio data "
            f"(header declares {expected_bytes} bytes, read {len(payload)})"
        )
    return WavInfo(
        path=path,
        sample_rate=rate,
        num_frames=nframes,
        duration_seconds=nframes / rate,
    )
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run python -m pytest tests/audio_teacher/test_audio.py -v
```
Expected: PASS (all 4 tests)

- [ ] **Step 5: Commit**

```bash
git add model/src/audio_teacher/audio.py model/tests/audio_teacher/test_audio.py && git commit -m "feat(audio-teacher): malformed-clip validation aborts naming the file (#127)"
```

---

### Task 3: Budget guard raises before the overshooting call
**Group:** B (parallel with Task 2, Task 4)

**Behavior being verified:** `BudgetGuard.precheck` raises `BudgetExceededError` BEFORE a call whose estimate would exceed the cap, and refused calls do not change spend.
**Interface under test:** `audio_teacher.budget.BudgetGuard`

**Files:**
- Create: `model/src/audio_teacher/budget.py`
- Test: `model/tests/audio_teacher/test_budget.py`

- [ ] **Step 1: Write the failing test**

Create `model/tests/audio_teacher/test_budget.py`:

```python
"""Hard spend-cap behavior: the guard refuses BEFORE the overshooting call."""
from __future__ import annotations

import pytest

from audio_teacher.budget import BudgetExceededError, BudgetGuard


def test_precheck_raises_before_the_overshooting_call():
    guard = BudgetGuard(max_spend_usd=1.0)
    guard.precheck(0.4)
    guard.record(0.4)
    guard.precheck(0.5)
    guard.record(0.5)  # spent 0.9 of 1.0
    with pytest.raises(BudgetExceededError) as excinfo:
        guard.precheck(0.2)  # would project 1.1 -- must refuse BEFORE the call
    assert "cap" in str(excinfo.value)
    assert guard.spent_usd == pytest.approx(0.9)  # the refused call charged nothing
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run python -m pytest tests/audio_teacher/test_budget.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'audio_teacher.budget'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `model/src/audio_teacher/budget.py`:

```python
"""Hard pre-call spend cap for the Gate 0 probe run.

precheck() raises BEFORE the call that would exceed the cap. A
warn-and-continue mode deliberately does not exist ($50 hard stop is a
standing program policy).
"""
from __future__ import annotations


class BudgetExceededError(Exception):
    """The next call's estimated cost would exceed the spend cap."""


class BudgetGuard:
    def __init__(self, max_spend_usd: float):
        if max_spend_usd <= 0:
            raise ValueError(f"max_spend_usd must be positive, got {max_spend_usd}")
        self.max_spend_usd = max_spend_usd
        self.spent_usd = 0.0

    def precheck(self, estimated_next_cost_usd: float) -> None:
        """Raise BudgetExceededError if the next call would exceed the cap.

        Call this BEFORE every sampling call. Refusal charges nothing.
        """
        projected = self.spent_usd + estimated_next_cost_usd
        if projected > self.max_spend_usd:
            raise BudgetExceededError(
                f"next call estimated at ${estimated_next_cost_usd:.4f} would take "
                f"spend to ${projected:.4f}, over the ${self.max_spend_usd:.2f} cap "
                f"(spent so far ${self.spent_usd:.4f}). Responses so far are saved; "
                f"a re-run resumes from the manifest."
            )

    def record(self, actual_cost_usd: float) -> None:
        self.spent_usd += actual_cost_usd
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run python -m pytest tests/audio_teacher/test_budget.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add model/src/audio_teacher/budget.py model/tests/audio_teacher/test_budget.py && git commit -m "feat(audio-teacher): budget guard raises before the overshooting call (#127)"
```

---

### Task 4: Per-axis elicitation questions
**Group:** B (parallel with Task 2, Task 3)

**Behavior being verified:** `build_question` returns an axis-specific degradation question that carries the strict `ANSWER: A|B` instruction; unknown axes raise.
**Interface under test:** `audio_teacher.prompts.build_question`

**Files:**
- Create: `model/src/audio_teacher/prompts.py`
- Test: `model/tests/audio_teacher/test_prompts.py`

- [ ] **Step 1: Write the failing test**

Create `model/tests/audio_teacher/test_prompts.py`:

```python
"""Elicitation contract: axis-specific question + strict answer format."""
from __future__ import annotations

import pytest

from audio_teacher.prompts import build_question


@pytest.mark.parametrize(
    "axis,keyword",
    [("pedaling", "pedal"), ("dynamics", "dynamic"), ("phrasing", "phras")],
)
def test_question_names_the_axis_contrast_and_forces_ab_answer(axis, keyword):
    question = build_question(axis)
    assert keyword in question.lower()
    assert 'ANSWER: A' in question and 'ANSWER: B' in question
    with pytest.raises(KeyError):
        build_question("rubato")
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run python -m pytest tests/audio_teacher/test_prompts.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'audio_teacher.prompts'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `model/src/audio_teacher/prompts.py`:

```python
"""Structured elicitation for contrast axes.

The question always asks which clip exhibits the DEGRADATION on the
axis, so downstream a response is correct iff the parsed choice equals
ContrastPair.degraded.
"""
from __future__ import annotations

AXIS_QUESTIONS = {
    "pedaling": (
        "One of these two piano recordings is over-pedaled: excessive sustain "
        "pedal blurs harmonies and note attacks together. Which one?"
    ),
    "dynamics": (
        "One of these two piano recordings has flat, unshaped dynamics: little "
        "contrast between loud and soft, and no dynamic direction across the "
        "phrase. Which one?"
    ),
    "phrasing": (
        "One of these two piano recordings has weak phrasing: no breathing "
        "between phrases, uniform note weight, and no sense of line. Which one?"
    ),
}

ANSWER_INSTRUCTION = (
    "Listen to Clip A, then Clip B. Explain briefly, then end with a final "
    'line of exactly "ANSWER: A" or "ANSWER: B".'
)


def build_question(axis: str) -> str:
    if axis not in AXIS_QUESTIONS:
        raise KeyError(
            f"no elicitation question for axis {axis!r}; known: {sorted(AXIS_QUESTIONS)}"
        )
    return f"{AXIS_QUESTIONS[axis]}\n\n{ANSWER_INSTRUCTION}"
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run python -m pytest tests/audio_teacher/test_prompts.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add model/src/audio_teacher/prompts.py model/tests/audio_teacher/test_prompts.py && git commit -m "feat(audio-teacher): per-axis elicitation questions with forced A/B answer (#127)"
```

---

### Task 5: Forced-choice answer parsing
**Group:** C (parallel with Task 6, Task 7)

**Behavior being verified:** `parse_choice` extracts the A/B choice from model text (last `ANSWER:` line wins, case-insensitive) and returns `None` when no valid answer exists.
**Interface under test:** `audio_teacher.prompts.parse_choice`

**Files:**
- Modify: `model/src/audio_teacher/prompts.py`
- Test: `model/tests/audio_teacher/test_prompts.py`

- [ ] **Step 1: Write the failing test**

Append to `model/tests/audio_teacher/test_prompts.py`:

```python
@pytest.mark.parametrize(
    "text,expected",
    [
        ("The first clip blurs badly.\nANSWER: A", "a"),
        ("answer: b", "b"),
        ("ANSWER: A\nOn reflection...\nANSWER: B", "b"),  # last answer wins
        ("Both sound similar to me.", None),
        ("ANSWER: C", None),
        ("", None),
    ],
    ids=["plain", "lowercase", "last_wins", "no_answer", "invalid_letter", "empty"],
)
def test_parse_choice_extracts_forced_ab_or_none(text, expected):
    from audio_teacher.prompts import parse_choice

    assert parse_choice(text) == expected
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run python -m pytest tests/audio_teacher/test_prompts.py -v
```
Expected: FAIL — `ImportError: cannot import name 'parse_choice' from 'audio_teacher.prompts'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Append to `model/src/audio_teacher/prompts.py` (add `import re` below `from __future__ import annotations`):

```python
_ANSWER_RE = re.compile(r"ANSWER:\s*([AB])\b", re.IGNORECASE)


def parse_choice(text: str) -> str | None:
    """Extract the forced A/B choice from a model response.

    The last ANSWER: line wins (models sometimes revise). Returns "a",
    "b", or None when no well-formed answer exists -- the scorer counts
    None as unparseable, which pushes the gate toward FAIL (closed).
    """
    matches = _ANSWER_RE.findall(text)
    if not matches:
        return None
    return matches[-1].lower()
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run python -m pytest tests/audio_teacher/test_prompts.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add model/src/audio_teacher/prompts.py model/tests/audio_teacher/test_prompts.py && git commit -m "feat(audio-teacher): tolerant last-answer-wins choice parsing (#127)"
```

---

### Task 6: Manifest loader — valid manifest loads with every clip validated
**Group:** C (parallel with Task 5, Task 7)

**Behavior being verified:** A schema-v1 YAML manifest with existing valid clips loads into a `ProbeManifest` of `ContrastPair`s in file order, with clip paths resolved against `repo_root`.
**Interface under test:** `audio_teacher.manifest.load_manifest`

**Files:**
- Create: `model/src/audio_teacher/manifest.py`
- Test: `model/tests/audio_teacher/test_manifest.py`

- [ ] **Step 1: Write the failing test**

Create `model/tests/audio_teacher/test_manifest.py`:

```python
"""Manifest loader behavior: full validation on load, loud failures."""
from __future__ import annotations

import pytest

from audio_teacher.manifest import load_manifest


def test_valid_manifest_loads_pairs_in_order_with_resolved_clips(
    tmp_path, manifest_factory
):
    manifest_path = manifest_factory(
        [
            {"id": "p1", "axis": "pedaling", "population": "real", "degraded": "a"},
            {"id": "p2", "axis": "dynamics", "population": "synthetic", "degraded": "b"},
        ]
    )
    manifest = load_manifest(manifest_path, repo_root=tmp_path)
    assert manifest.sample_rate == 16000
    assert [p.pair_id for p in manifest.pairs] == ["p1", "p2"]
    p1, p2 = manifest.pairs
    assert p1.axis == "pedaling" and p1.population == "real" and p1.degraded == "a"
    assert p2.axis == "dynamics" and p2.population == "synthetic" and p2.degraded == "b"
    assert p1.clip_a.is_absolute() and p1.clip_a.exists()
    assert p2.clip_b == tmp_path / "clips" / "p2_b.wav"
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run python -m pytest tests/audio_teacher/test_manifest.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'audio_teacher.manifest'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `model/src/audio_teacher/manifest.py`:

```python
"""Contrast-pair manifest: YAML schema v1 + loader for the Gate 0 probe.

Schema (YAML):
    schema_version: 1
    sample_rate: 16000            # every clip header must match
    pairs:
      - id: unique-string
        axis: pedaling            # pedaling | dynamics | phrasing
        population: real          # real | synthetic  (NEVER pooled downstream)
        clip_a: relative/path.wav # relative to repo_root (model/)
        clip_b: relative/path.wav
        degraded: a               # which clip exhibits the degradation on axis
        description: free text

Loading validates EVERY clip: existence (with the exact rehydrate command
in the error when the path is R2-offloaded per data/manifests/r2_offload.json)
and WAV header (mono, expected sample rate, not truncated). A manifest
that cannot fully load raises -- probing a silently filtered subsample
is impossible.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from audio_teacher.audio import validate_wav

MODEL_ROOT = Path(__file__).resolve().parents[2]  # model/
DEFAULT_OFFLOAD_REGISTRY = MODEL_ROOT / "data" / "manifests" / "r2_offload.json"

AXES = ("pedaling", "dynamics", "phrasing")
POPULATIONS = ("real", "synthetic")
_REQUIRED_PAIR_KEYS = (
    "id", "axis", "population", "clip_a", "clip_b", "degraded", "description",
)


class ManifestError(Exception):
    """The manifest file itself violates the schema."""


@dataclass(frozen=True)
class ContrastPair:
    pair_id: str
    axis: str
    population: str
    clip_a: Path
    clip_b: Path
    degraded: str  # "a" | "b"
    description: str


@dataclass(frozen=True)
class ProbeManifest:
    sample_rate: int
    pairs: tuple[ContrastPair, ...]


def _ensure_clip_local(path: Path, repo_root: Path, offload_registry: Path) -> None:
    """Raise FileNotFoundError with the exact rehydrate command for a missing
    clip. Mirrors src/paths.py ensure_local (not importable from installed
    packages), extended with prefix matching so clip FILES under an offloaded
    DIRECTORY (e.g. data/raw/competition) resolve to that directory's entry.
    """
    if path.exists():
        return
    hint = ""
    if offload_registry.exists():
        with offload_registry.open() as f:
            registry = json.load(f)
        try:
            rel = str(path.relative_to(repo_root))
        except ValueError:
            rel = str(path)
        for registered, entry in registry.get("entries", {}).items():
            if rel == registered or rel.startswith(registered + "/"):
                if "r2_prefix" in entry:
                    cmd = (
                        f"rclone copy {registry['remote_name']}:{registry['bucket']}"
                        f"/{entry['r2_prefix']} {registered}"
                    )
                else:
                    cmd = entry.get("regen_command", "")
                hint = (
                    f" Clip is under R2-offloaded path ({entry.get('reason', '')})."
                    f" Rehydrate with:\n    {cmd}"
                )
                break
    raise FileNotFoundError(f"probe clip missing: {path}.{hint}")


def load_manifest(
    manifest_path: Path | str,
    repo_root: Path = MODEL_ROOT,
    offload_registry: Path = DEFAULT_OFFLOAD_REGISTRY,
) -> ProbeManifest:
    manifest_path = Path(manifest_path)
    raw = yaml.safe_load(manifest_path.read_text())
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ManifestError(f"{manifest_path}: expected a mapping with schema_version: 1")
    sample_rate = raw.get("sample_rate")
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ManifestError(f"{manifest_path}: sample_rate must be a positive integer")
    raw_pairs = raw.get("pairs")
    if not isinstance(raw_pairs, list) or not raw_pairs:
        raise ManifestError(f"{manifest_path}: pairs must be a non-empty list")

    pairs: list[ContrastPair] = []
    seen: set[str] = set()
    for i, rp in enumerate(raw_pairs):
        missing = [k for k in _REQUIRED_PAIR_KEYS if k not in rp]
        if missing:
            raise ManifestError(f"{manifest_path}: pair[{i}] missing keys {missing}")
        pair_id = rp["id"]
        if pair_id in seen:
            raise ManifestError(f"{manifest_path}: duplicate pair id {pair_id!r}")
        seen.add(pair_id)
        if rp["axis"] not in AXES:
            raise ManifestError(
                f"{manifest_path}: pair {pair_id!r} axis {rp['axis']!r} not in {AXES}"
            )
        if rp["population"] not in POPULATIONS:
            raise ManifestError(
                f"{manifest_path}: pair {pair_id!r} population {rp['population']!r} "
                f"not in {POPULATIONS}"
            )
        if rp["degraded"] not in ("a", "b"):
            raise ManifestError(
                f"{manifest_path}: pair {pair_id!r} degraded must be 'a' or 'b'"
            )
        clip_a = repo_root / rp["clip_a"]
        clip_b = repo_root / rp["clip_b"]
        for clip in (clip_a, clip_b):
            _ensure_clip_local(clip, repo_root, offload_registry)
            validate_wav(clip, expected_sample_rate=sample_rate)
        pairs.append(
            ContrastPair(
                pair_id=pair_id,
                axis=rp["axis"],
                population=rp["population"],
                clip_a=clip_a,
                clip_b=clip_b,
                degraded=rp["degraded"],
                description=rp["description"],
            )
        )
    return ProbeManifest(sample_rate=sample_rate, pairs=tuple(pairs))
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run python -m pytest tests/audio_teacher/test_manifest.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add model/src/audio_teacher/manifest.py model/tests/audio_teacher/test_manifest.py && git commit -m "feat(audio-teacher): contrast-pair manifest loader validates every clip (#127)"
```

---

### Task 7: Recorded-response client (the offline fake)
**Group:** C (parallel with Task 5, Task 6)

**Behavior being verified:** `RecordedResponseClient` replays canned JSONL responses by pair id at zero cost and raises a loud `KeyError` for a pair with no recording (incomplete fixture).
**Interface under test:** `audio_teacher.client.RecordedResponseClient` (via the `ProbeClient` contract)

**Files:**
- Create: `model/src/audio_teacher/client.py`
- Test: `model/tests/audio_teacher/test_client.py`

- [ ] **Step 1: Write the failing test**

Create `model/tests/audio_teacher/test_client.py`:

```python
"""RecordedResponseClient: offline replay through the ProbeClient contract."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from audio_teacher.client import RecordedResponseClient
from audio_teacher.manifest import ContrastPair


def _pair(pair_id: str) -> ContrastPair:
    return ContrastPair(
        pair_id=pair_id,
        axis="pedaling",
        population="real",
        clip_a=Path("clips/a.wav"),
        clip_b=Path("clips/b.wav"),
        degraded="a",
        description="",
    )


def test_replays_recorded_response_and_errors_on_missing_pair(tmp_path):
    recorded = tmp_path / "recorded.jsonl"
    recorded.write_text(json.dumps({"pair_id": "p1", "text": "ANSWER: A"}) + "\n")
    client = RecordedResponseClient(recorded)

    assert client.estimate_cost_usd(_pair("p1")) == 0.0
    resp = client.ask(_pair("p1"))
    assert resp.pair_id == "p1"
    assert resp.text == "ANSWER: A"
    assert resp.cost_usd == 0.0

    with pytest.raises(KeyError) as excinfo:
        client.ask(_pair("p9"))
    assert "p9" in str(excinfo.value)
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run python -m pytest tests/audio_teacher/test_client.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'audio_teacher.client'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `model/src/audio_teacher/client.py`:

```python
"""Sampling-client boundary for the probe.

ProbeClient is the mockable seam: the probe driver only ever sees this
protocol. RecordedResponseClient is the offline implementation used by
every test and by re-scoring saved runs; the real Tinker implementation
lives in audio_teacher.tinker_client and is never imported by tests.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from audio_teacher.manifest import ContrastPair


@dataclass(frozen=True)
class ProbeResponse:
    pair_id: str
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


class ProbeClient(Protocol):
    def estimate_cost_usd(self, pair: ContrastPair) -> float: ...

    def ask(self, pair: ContrastPair) -> ProbeResponse: ...


class RecordedResponseClient:
    """Replays canned responses from a JSONL file keyed by pair_id.

    Record shape per line: {"pair_id": str, "text": str} (extra keys are
    ignored). A pair without a recording raises KeyError -- an incomplete
    fixture must fail loudly, never be skipped.
    """

    def __init__(self, responses_path: Path | str):
        self._responses: dict[str, str] = {}
        with Path(responses_path).open() as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                self._responses[rec["pair_id"]] = rec["text"]

    def estimate_cost_usd(self, pair: ContrastPair) -> float:
        return 0.0

    def ask(self, pair: ContrastPair) -> ProbeResponse:
        if pair.pair_id not in self._responses:
            raise KeyError(
                f"no recorded response for pair {pair.pair_id!r} -- the recorded "
                f"fixture is incomplete"
            )
        return ProbeResponse(
            pair_id=pair.pair_id,
            text=self._responses[pair.pair_id],
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
        )
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run python -m pytest tests/audio_teacher/test_client.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add model/src/audio_teacher/client.py model/tests/audio_teacher/test_client.py && git commit -m "feat(audio-teacher): ProbeClient protocol + recorded-response fake (#127)"
```

---

### Task 8: Missing/offloaded clip fails with the exact rehydrate command
**Group:** D (parallel with Task 10, Task 12)

**Behavior being verified:** Loading a manifest whose clip is missing but registered (by directory prefix) in the R2 offload registry raises `FileNotFoundError` containing the exact `rclone copy` command; a missing unregistered clip still raises naming the path.
**Interface under test:** `audio_teacher.manifest.load_manifest`

**Files:**
- Modify: `model/src/audio_teacher/manifest.py` (only if the test exposes a gap — the Task 6 implementation already carries `_ensure_clip_local`; this task locks the behavior with tests)
- Test: `model/tests/audio_teacher/test_manifest.py`

- [ ] **Step 1: Write the failing test**

Append to `model/tests/audio_teacher/test_manifest.py` (add `import json` to the imports):

```python
def test_missing_offloaded_clip_error_contains_rehydrate_command(
    tmp_path, manifest_factory
):
    manifest_path = manifest_factory([{"id": "p1"}])
    missing = tmp_path / "clips" / "p1_a.wav"
    missing.unlink()
    registry = tmp_path / "r2_offload.json"
    registry.write_text(
        json.dumps(
            {
                "bucket": "crescendai-bucket",
                "remote_name": "r2",
                "entries": {
                    "clips": {"r2_prefix": "mirex-probe/clips", "reason": "test offload"}
                },
            }
        )
    )
    with pytest.raises(FileNotFoundError) as excinfo:
        load_manifest(manifest_path, repo_root=tmp_path, offload_registry=registry)
    message = str(excinfo.value)
    assert "p1_a.wav" in message
    assert "rclone copy r2:crescendai-bucket/mirex-probe/clips clips" in message


def test_missing_unregistered_clip_still_fails_naming_the_path(
    tmp_path, manifest_factory
):
    manifest_path = manifest_factory([{"id": "p1"}])
    (tmp_path / "clips" / "p1_b.wav").unlink()
    with pytest.raises(FileNotFoundError) as excinfo:
        load_manifest(
            manifest_path,
            repo_root=tmp_path,
            offload_registry=tmp_path / "no_registry.json",
        )
    assert "p1_b.wav" in str(excinfo.value)
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run python -m pytest tests/audio_teacher/test_manifest.py -v
```
Expected: with the Task 6 implementation in place these tests may already PASS. If they pass, that is acceptable for this task ONLY because the implementation predates the test by design (Task 6 shipped `_ensure_clip_local` as a unit); to keep the failing-first discipline honest, first run with a deliberate breakage check: temporarily change the assertion string to `"rclone WRONG"` and confirm the test FAILS, then restore it. If instead the real assertions fail (e.g., prefix matching bug), fix `_ensure_clip_local` minimally in Step 3.

- [ ] **Step 3: Implement the minimum to make the test pass**

No change expected — `_ensure_clip_local` from Task 6 already implements exact-and-prefix registry matching and the rclone hint. If Step 2's real assertions failed, fix only the matching/hint construction in `_ensure_clip_local` (do not touch schema validation).

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run python -m pytest tests/audio_teacher/test_manifest.py -v
```
Expected: PASS (all manifest tests)

- [ ] **Step 5: Commit**

```bash
git add model/tests/audio_teacher/test_manifest.py model/src/audio_teacher/manifest.py && git commit -m "test(audio-teacher): lock rehydrate-hint behavior for offloaded probe clips (#127)"
```

---

### Task 9: Manifest schema violations raise ManifestError
**Group:** E (parallel with Task 11; runs after Group D because Task 8 also touches `manifest.py` + `test_manifest.py`)

**Behavior being verified:** Structurally invalid manifests (wrong schema_version, bad axis, bad population, bad degraded, duplicate id, missing key) each raise `ManifestError` before any clip I/O result is returned.
**Interface under test:** `audio_teacher.manifest.load_manifest`

**Files:**
- Modify: `model/src/audio_teacher/manifest.py` (only if a case is not already rejected)
- Test: `model/tests/audio_teacher/test_manifest.py`

- [ ] **Step 1: Write the failing test**

Append to `model/tests/audio_teacher/test_manifest.py` (add `import yaml` to the imports):

```python
def _mutate_schema_version(doc):
    doc["schema_version"] = 2


def _mutate_axis(doc):
    doc["pairs"][0]["axis"] = "rubato"


def _mutate_population(doc):
    doc["pairs"][0]["population"] = "studio"


def _mutate_degraded(doc):
    doc["pairs"][0]["degraded"] = "c"


def _mutate_duplicate_id(doc):
    doc["pairs"].append(dict(doc["pairs"][0]))


def _mutate_missing_key(doc):
    del doc["pairs"][0]["description"]


@pytest.mark.parametrize(
    "mutate",
    [
        _mutate_schema_version,
        _mutate_axis,
        _mutate_population,
        _mutate_degraded,
        _mutate_duplicate_id,
        _mutate_missing_key,
    ],
    ids=[
        "schema_version", "axis", "population", "degraded",
        "duplicate_id", "missing_key",
    ],
)
def test_schema_violations_raise_manifest_error(tmp_path, manifest_factory, mutate):
    from audio_teacher.manifest import ManifestError

    manifest_path = manifest_factory([{"id": "p1"}])
    doc = yaml.safe_load(manifest_path.read_text())
    mutate(doc)
    manifest_path.write_text(yaml.safe_dump(doc, sort_keys=False))
    with pytest.raises(ManifestError):
        load_manifest(manifest_path, repo_root=tmp_path)
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run python -m pytest tests/audio_teacher/test_manifest.py -v
```
Expected: the Task 6 implementation already rejects all six cases, so these may PASS immediately. Apply the same honesty check as Task 8 Step 2: temporarily weaken one assertion (`pytest.raises(ValueError)`) to confirm the test is live, then restore. If any real case does NOT raise `ManifestError`, proceed to Step 3.

- [ ] **Step 3: Implement the minimum to make the test pass**

No change expected. If a case slipped through, add the missing validation branch in `load_manifest` raising `ManifestError` with the manifest path and pair id in the message (mirror the existing branches — do not restructure).

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run python -m pytest tests/audio_teacher/test_manifest.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add model/tests/audio_teacher/test_manifest.py model/src/audio_teacher/manifest.py && git commit -m "test(audio-teacher): lock manifest schema rejection cases (#127)"
```

---

### Task 10: Scorer — population-partitioned cells, no pooled number anywhere
**Group:** D (parallel with Task 8, Task 12)

**Behavior being verified:** `score_responses` produces per-`axis/population` cells (n, correct, unparseable, accuracy, unparseable_rate); real and synthetic are NEVER pooled — no combined cell or overall accuracy exists in the report; mismatched response sets raise `ProbeIncompleteError`.
**Interface under test:** `audio_teacher.scorer.score_responses`

**Files:**
- Create: `model/src/audio_teacher/scorer.py`
- Test: `model/tests/audio_teacher/test_scorer.py`

- [ ] **Step 1: Write the failing test**

Create `model/tests/audio_teacher/test_scorer.py`:

```python
"""Scorer behavior: partitioned cells, ex-ante verdicts, determinism.

The #21 scar: synthetic-vs-real pooling produced a fatally misleading
number once. Any pooled figure in this report is a test failure.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from audio_teacher.manifest import ContrastPair, ProbeManifest
from audio_teacher.scorer import ProbeIncompleteError, score_responses


def _pair(pair_id: str, axis: str = "pedaling", population: str = "real",
          degraded: str = "a") -> ContrastPair:
    return ContrastPair(
        pair_id=pair_id,
        axis=axis,
        population=population,
        clip_a=Path(f"clips/{pair_id}_a.wav"),
        clip_b=Path(f"clips/{pair_id}_b.wav"),
        degraded=degraded,
        description="",
    )


def _manifest(pairs) -> ProbeManifest:
    return ProbeManifest(sample_rate=16000, pairs=tuple(pairs))


def test_cells_are_partitioned_by_population_and_never_pooled():
    manifest = _manifest(
        [
            _pair("r1", population="real", degraded="a"),
            _pair("r2", population="real", degraded="b"),
            _pair("s1", population="synthetic", degraded="a"),
        ]
    )
    responses = {
        "r1": "blurred.\nANSWER: A",   # correct
        "r2": "muddy.\nANSWER: A",     # wrong (degraded is b)
        "s1": "ANSWER: A",             # correct
    }
    report = score_responses(manifest, responses)
    assert report["cells"]["pedaling/real"] == {
        "n": 2, "correct": 1, "unparseable": 0,
        "accuracy": 0.5, "unparseable_rate": 0.0,
    }
    assert report["cells"]["pedaling/synthetic"] == {
        "n": 1, "correct": 1, "unparseable": 0,
        "accuracy": 1.0, "unparseable_rate": 0.0,
    }
    # No pooled number anywhere: only axis/population cell keys, no
    # bare-axis cell, no overall/pooled key in the report.
    assert set(report["cells"]) == {"pedaling/real", "pedaling/synthetic"}
    assert "overall" not in report and "accuracy" not in report

    with pytest.raises(ProbeIncompleteError):
        score_responses(manifest, {"r1": "ANSWER: A"})  # missing r2, s1
    with pytest.raises(ProbeIncompleteError):
        score_responses(manifest, {**responses, "ghost": "ANSWER: B"})  # extra
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run python -m pytest tests/audio_teacher/test_scorer.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'audio_teacher.scorer'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `model/src/audio_teacher/scorer.py`:

```python
"""Probe scoring: per-(axis, population) accuracy + ex-ante gate verdict.

Populations are NEVER pooled (issue #21 synthetic-gap scar): every number
in the report is keyed "axis/population". The verdict reads ONLY real-
population cells; synthetic cells are informative. Uncertainty (too few
real pairs, too many unparseable responses) FAILS -- the gate defaults
to closed. Thresholds are ex-ante constants; there is deliberately no
override knob.
"""
from __future__ import annotations

import json
from typing import Mapping

from audio_teacher.manifest import ProbeManifest
from audio_teacher.prompts import parse_choice

KILL_THRESHOLD = 0.70
MIN_REAL_PAIRS_PER_AXIS = 20
MAX_UNPARSEABLE_RATE = 0.10


class ProbeIncompleteError(Exception):
    """Responses do not cover the manifest exactly (missing or extra pairs)."""


def score_responses(manifest: ProbeManifest, responses: Mapping[str, str]) -> dict:
    manifest_ids = {p.pair_id for p in manifest.pairs}
    missing = sorted(manifest_ids - set(responses))
    extra = sorted(set(responses) - manifest_ids)
    if missing or extra:
        raise ProbeIncompleteError(
            f"responses do not match manifest: missing={missing} extra={extra}"
        )

    cells: dict[str, dict] = {}
    for pair in manifest.pairs:
        key = f"{pair.axis}/{pair.population}"
        cell = cells.setdefault(key, {"n": 0, "correct": 0, "unparseable": 0})
        cell["n"] += 1
        choice = parse_choice(responses[pair.pair_id])
        if choice is None:
            cell["unparseable"] += 1
        elif choice == pair.degraded:
            cell["correct"] += 1
    for cell in cells.values():
        cell["accuracy"] = cell["correct"] / cell["n"]
        cell["unparseable_rate"] = cell["unparseable"] / cell["n"]

    reasons = _verdict_reasons(manifest, cells)
    return {
        "schema_version": 1,
        "thresholds": {
            "kill_threshold": KILL_THRESHOLD,
            "min_real_pairs_per_axis": MIN_REAL_PAIRS_PER_AXIS,
            "max_unparseable_rate": MAX_UNPARSEABLE_RATE,
        },
        "cells": cells,
        "verdict": "PASS" if not reasons else "FAIL",
        "verdict_reasons": reasons,
    }


def _verdict_reasons(manifest: ProbeManifest, cells: dict[str, dict]) -> list[str]:
    reasons: list[str] = []
    for axis in sorted({p.axis for p in manifest.pairs}):
        real = cells.get(f"{axis}/real")
        if real is None:
            reasons.append(
                f"{axis}: no real-population pairs; synthetic alone never opens "
                f"the gate (issue #21 synthetic-gap)"
            )
            continue
        if real["n"] < MIN_REAL_PAIRS_PER_AXIS:
            reasons.append(
                f"{axis}/real: only {real['n']} pairs, need >= {MIN_REAL_PAIRS_PER_AXIS}"
            )
        if real["unparseable_rate"] > MAX_UNPARSEABLE_RATE:
            reasons.append(
                f"{axis}/real: unparseable rate {real['unparseable_rate']:.2f} "
                f"above {MAX_UNPARSEABLE_RATE:.2f} (ambiguous -> gate stays closed)"
            )
        if real["accuracy"] < KILL_THRESHOLD:
            reasons.append(
                f"{axis}/real: accuracy {real['accuracy']:.2f} below "
                f"{KILL_THRESHOLD:.2f} kill threshold"
            )
    return reasons
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run python -m pytest tests/audio_teacher/test_scorer.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add model/src/audio_teacher/scorer.py model/tests/audio_teacher/test_scorer.py && git commit -m "feat(audio-teacher): population-partitioned probe scoring, no pooling (#127)"
```

---

### Task 11: Scorer — ex-ante verdict rules (gate defaults to closed)
**Group:** E (parallel with Task 9)

**Behavior being verified:** The verdict is PASS only when every axis clears all real-population criteria; low accuracy, insufficient real pairs, synthetic-only axes, and high unparseable rates each FAIL with a naming reason; a failing synthetic cell alone never fails a passing axis.
**Interface under test:** `audio_teacher.scorer.score_responses`

**Files:**
- Modify: `model/src/audio_teacher/scorer.py` (only if a rule is missing — Task 10 shipped `_verdict_reasons`)
- Test: `model/tests/audio_teacher/test_scorer.py`

- [ ] **Step 1: Write the failing test**

Append to `model/tests/audio_teacher/test_scorer.py`:

```python
def _bulk(axis: str, population: str, n_correct: int, n_wrong: int,
          n_unparseable: int, start: int = 0):
    """Build (pairs, responses) for one cell. degraded='a'; correct answers
    say A, wrong say B, unparseable give no ANSWER line."""
    pairs, responses = [], {}
    i = start
    for count, text in (
        (n_correct, "ANSWER: A"),
        (n_wrong, "ANSWER: B"),
        (n_unparseable, "cannot tell"),
    ):
        for _ in range(count):
            pid = f"{axis[:3]}_{population[:3]}_{i}"
            pairs.append(_pair(pid, axis=axis, population=population, degraded="a"))
            responses[pid] = text
            i += 1
    return pairs, responses


@pytest.mark.parametrize(
    "cell_specs,expected_verdict,expected_reason_fragment",
    [
        # 20/20 real correct + synthetic all wrong: synthetic never gates.
        ([("pedaling", "real", 20, 0, 0), ("pedaling", "synthetic", 0, 5, 0)],
         "PASS", None),
        # 10/20 real correct: below the 0.70 kill threshold.
        ([("pedaling", "real", 10, 10, 0)], "FAIL", "below"),
        # Only 5 real pairs: insufficient evidence, gate stays closed.
        ([("pedaling", "real", 5, 0, 0)], "FAIL", "only 5 pairs"),
        # Synthetic-only axis: never opens the gate.
        ([("dynamics", "synthetic", 20, 0, 0)], "FAIL", "no real-population"),
        # 17 correct + 3 unparseable of 20: accuracy 0.85 but ambiguity 0.15.
        ([("pedaling", "real", 17, 0, 3)], "FAIL", "unparseable"),
    ],
    ids=["pass_synthetic_never_gates", "fail_accuracy", "fail_insufficient",
         "fail_synthetic_only", "fail_ambiguous"],
)
def test_verdict_applies_ex_ante_kill_rules(
    cell_specs, expected_verdict, expected_reason_fragment
):
    pairs, responses = [], {}
    start = 0
    for axis, population, n_correct, n_wrong, n_unparseable in cell_specs:
        p, r = _bulk(axis, population, n_correct, n_wrong, n_unparseable, start)
        pairs += p
        responses.update(r)
        start += len(p)
    report = score_responses(_manifest(pairs), responses)
    assert report["verdict"] == expected_verdict
    if expected_reason_fragment is None:
        assert report["verdict_reasons"] == []
    else:
        assert any(expected_reason_fragment in r for r in report["verdict_reasons"])
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run python -m pytest tests/audio_teacher/test_scorer.py -v
```
Expected: the Task 10 implementation should already satisfy these rules, so they may PASS immediately. Honesty check as in Task 8/9: temporarily set `expected_verdict` of the first case to `"FAIL"`, confirm the test fails, restore. If a real case fails, proceed to Step 3.

- [ ] **Step 3: Implement the minimum to make the test pass**

No change expected. If a rule is wrong, fix only `_verdict_reasons` in `model/src/audio_teacher/scorer.py` to match the parametrized cases exactly (constants stay `0.70` / `20` / `0.10`).

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run python -m pytest tests/audio_teacher/test_scorer.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add model/tests/audio_teacher/test_scorer.py model/src/audio_teacher/scorer.py && git commit -m "test(audio-teacher): lock ex-ante verdict rules -- uncertain defaults to closed (#127)"
```

---

### Task 12: Tinker client — real implementation behind the seam, loud when SDK absent
**Group:** D (parallel with Task 8, Task 10)

**Behavior being verified:** Constructing `TinkerProbeClient` without the Tinker SDK installed raises `TinkerNotInstalledError` whose message contains the install command. (This is the ONLY tested path; live sampling is exercised by the user's funded Gate 0 run, never by tests.)
**Interface under test:** `audio_teacher.tinker_client.TinkerProbeClient`

**Files:**
- Create: `model/src/audio_teacher/tinker_client.py`
- Test: `model/tests/audio_teacher/test_tinker_client.py`

- [ ] **Step 1: Write the failing test**

Create `model/tests/audio_teacher/test_tinker_client.py`:

```python
"""The not-installed error path -- the only Tinker surface tests touch."""
from __future__ import annotations

import importlib.util

import pytest

_TINKER_MISSING = importlib.util.find_spec("tinker") is None


@pytest.mark.skipif(
    not _TINKER_MISSING,
    reason="tinker SDK installed in this env; the not-installed path is unreachable",
)
def test_missing_sdk_raises_with_install_instructions():
    from audio_teacher.tinker_client import TinkerNotInstalledError, TinkerProbeClient

    with pytest.raises(TinkerNotInstalledError) as excinfo:
        TinkerProbeClient(
            sample_rate=16000,
            usd_per_1m_input_tokens=1.0,
            usd_per_1m_output_tokens=3.0,
        )
    assert "uv add tinker" in str(excinfo.value)
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run python -m pytest tests/audio_teacher/test_tinker_client.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'audio_teacher.tinker_client'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `model/src/audio_teacher/tinker_client.py`:

```python
"""Real Tinker sampling client for the Gate 0 Inkling probe.

Wraps the Inkling audio flow
(https://tinker-docs.thinkingmachines.ai/cookbook/inkling/audio): local
WAV files are referenced with tml_renderers.chat.AudioPointer and encoded
client-side into DMel tokens; remote URLs are unsupported, which is why
the manifest loader guarantees clips exist locally before any call.

Never imported by tests except the not-installed error path; all scoring
tests use audio_teacher.client.RecordedResponseClient. The SDK trio is
deliberately NOT in pyproject dependencies -- verify current package
names on tinker-docs.thinkingmachines.ai when funding the probe run.
"""
from __future__ import annotations

from audio_teacher.audio import validate_wav
from audio_teacher.client import ProbeResponse
from audio_teacher.manifest import ContrastPair
from audio_teacher.prompts import build_question

INKLING_MODEL = "thinkingmachines/Inkling"

# Conservative DMel-tokens-per-second estimate for PRE-CALL cost projection.
# Overestimating trips the budget cap earlier, never later -- the safe
# direction. Actual recorded cost uses real token counts from the response.
AUDIO_TOKENS_PER_SECOND_ESTIMATE = 100

# Text scaffold (question + chat template) allowance for pre-call estimates.
TEXT_TOKENS_ESTIMATE = 500


class TinkerNotInstalledError(RuntimeError):
    """The Tinker SDK trio is not installed in this environment."""


class TinkerProbeClient:
    """ProbeClient implementation that samples thinkingmachines/Inkling."""

    def __init__(
        self,
        sample_rate: int,
        usd_per_1m_input_tokens: float,
        usd_per_1m_output_tokens: float,
        max_tokens: int = 256,
    ):
        try:
            import tinker
            from tinker_cookbook import model_info
            from tinker_cookbook.renderers import get_renderer
            from tinker_cookbook.tokenizer_utils import get_tokenizer
            from tml_renderers import chat
        except ImportError as exc:
            raise TinkerNotInstalledError(
                "Tinker SDK not installed; the live probe needs it. Install with:\n"
                "    cd model && uv add tinker tinker-cookbook\n"
                "(verify current package names on tinker-docs.thinkingmachines.ai; "
                "tests never need this -- offline scoring uses RecordedResponseClient)"
            ) from exc
        self._tinker = tinker
        self._chat = chat
        self._sample_rate = sample_rate
        self._in_rate = usd_per_1m_input_tokens
        self._out_rate = usd_per_1m_output_tokens
        self._max_tokens = max_tokens
        self._renderer = get_renderer(
            model_info.get_recommended_renderer_name(INKLING_MODEL),
            get_tokenizer(INKLING_MODEL),
        )
        self._sampling = tinker.ServiceClient().create_sampling_client(
            base_model=INKLING_MODEL
        )

    def _messages(self, pair: ContrastPair):
        chat = self._chat
        user = chat.Author(chat.AuthorKind.User)

        def clip_message(path):
            info = validate_wav(path, expected_sample_rate=self._sample_rate)
            return chat.Message(
                content=chat.AudioPointer(
                    location=str(path),
                    format=chat.AudioFormat.Wav,
                    num_frames=info.num_frames,
                    sample_rate=info.sample_rate,
                ),
                author=user,
            )

        return [
            chat.Message(content=chat.Text(build_question(pair.axis)), author=user),
            chat.Message(content=chat.Text("Clip A:"), author=user),
            clip_message(pair.clip_a),
            chat.Message(content=chat.Text("Clip B:"), author=user),
            clip_message(pair.clip_b),
        ]

    def estimate_cost_usd(self, pair: ContrastPair) -> float:
        seconds = 0.0
        for path in (pair.clip_a, pair.clip_b):
            info = validate_wav(path, expected_sample_rate=self._sample_rate)
            seconds += info.duration_seconds
        est_input = seconds * AUDIO_TOKENS_PER_SECOND_ESTIMATE + TEXT_TOKENS_ESTIMATE
        return (est_input / 1e6) * self._in_rate + (
            self._max_tokens / 1e6
        ) * self._out_rate

    def ask(self, pair: ContrastPair) -> ProbeResponse:
        prompt = self._renderer.build_generation_prompt(self._messages(pair))
        result = self._sampling.sample(
            prompt=prompt,
            num_samples=1,
            sampling_params=self._tinker.SamplingParams(
                max_tokens=self._max_tokens,
                stop=self._renderer.get_stop_sequences(),
            ),
        ).result()
        tokens = result.sequences[0].tokens
        message, _termination = self._renderer.parse_response(tokens)
        content = getattr(message, "content", message)
        text = content.text if hasattr(content, "text") else str(content)
        # Loud on API drift: if ModelInput stops exposing length, this raises.
        input_tokens = prompt.length
        output_tokens = len(tokens)
        cost = (input_tokens / 1e6) * self._in_rate + (
            output_tokens / 1e6
        ) * self._out_rate
        return ProbeResponse(
            pair_id=pair.pair_id,
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run python -m pytest tests/audio_teacher/test_tinker_client.py -v
```
Expected: PASS (1 passed, or 1 skipped iff the SDK happens to be installed)

- [ ] **Step 5: Commit**

```bash
git add model/src/audio_teacher/tinker_client.py model/tests/audio_teacher/test_tinker_client.py && git commit -m "feat(audio-teacher): Inkling Tinker client behind the ProbeClient seam (#127)"
```

---

### Task 13: Deterministic report rendering
**Group:** F (parallel with Task 14)

**Behavior being verified:** `render_report(score_responses(...))` is byte-identical for the same manifest + responses, regardless of response dict insertion order, and contains no timestamp.
**Interface under test:** `audio_teacher.scorer.render_report`

**Files:**
- Modify: `model/src/audio_teacher/scorer.py`
- Test: `model/tests/audio_teacher/test_scorer.py`

- [ ] **Step 1: Write the failing test**

Append to `model/tests/audio_teacher/test_scorer.py`:

```python
def test_same_inputs_render_byte_identical_reports():
    from audio_teacher.scorer import render_report

    pairs, responses = _bulk("pedaling", "real", 20, 0, 0)
    manifest = _manifest(pairs)
    reversed_responses = dict(reversed(list(responses.items())))
    r1 = render_report(score_responses(manifest, responses))
    r2 = render_report(score_responses(manifest, reversed_responses))
    assert r1 == r2
    assert r1.endswith("\n")
    assert "generated_at" not in r1  # volatile metadata lives in run_meta.json
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run python -m pytest tests/audio_teacher/test_scorer.py -v
```
Expected: FAIL — `ImportError: cannot import name 'render_report' from 'audio_teacher.scorer'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Append to `model/src/audio_teacher/scorer.py`:

```python
def render_report(report: dict) -> str:
    """Deterministic serialization: the same report dict always renders to
    byte-identical text (sorted keys, fixed indent, trailing newline). The
    report carries no timestamps -- volatile run metadata belongs in
    run_meta.json, written by the probe driver."""
    return json.dumps(report, indent=2, sort_keys=True) + "\n"
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run python -m pytest tests/audio_teacher/test_scorer.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add model/src/audio_teacher/scorer.py model/tests/audio_teacher/test_scorer.py && git commit -m "feat(audio-teacher): byte-deterministic report rendering (#127)"
```

---

### Task 14: Manifest curation script (CSV -> YAML, deliberately shallow)
**Group:** F (parallel with Task 13)

**Behavior being verified:** `build_manifest` converts a pairs CSV into a schema-v1 YAML manifest that round-trips through `load_manifest`; an invalid row aborts before anything is written.
**Interface under test:** `audio_teacher.build_manifest.main`

**Files:**
- Create: `model/src/audio_teacher/build_manifest.py`
- Test: `model/tests/audio_teacher/test_build_manifest.py`

- [ ] **Step 1: Write the failing test**

Create `model/tests/audio_teacher/test_build_manifest.py`:

```python
"""CSV -> YAML curation: output is loadable by construction."""
from __future__ import annotations

import pytest

from audio_teacher.build_manifest import CurationError, main
from audio_teacher.manifest import load_manifest

_HEADER = "id,axis,population,clip_a,clip_b,degraded,description\n"


def test_csv_curation_round_trips_and_rejects_bad_rows(tmp_path, wav_factory):
    wav_factory("clips/x_a.wav")
    wav_factory("clips/x_b.wav")
    csv_path = tmp_path / "pairs.csv"
    csv_path.write_text(
        _HEADER + "x,pedaling,real,clips/x_a.wav,clips/x_b.wav,a,over-pedaled take\n"
    )
    out = tmp_path / "probe_manifest.yaml"
    rc = main(
        [
            "--pairs-csv", str(csv_path),
            "--sample-rate", "16000",
            "--out", str(out),
            "--repo-root", str(tmp_path),
        ]
    )
    assert rc == 0
    manifest = load_manifest(out, repo_root=tmp_path)
    assert [p.pair_id for p in manifest.pairs] == ["x"]
    assert manifest.pairs[0].degraded == "a"
    assert manifest.pairs[0].description == "over-pedaled take"

    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text(
        _HEADER + "y,rubato,real,clips/x_a.wav,clips/x_b.wav,a,bad axis\n"
    )
    bad_out = tmp_path / "bad_manifest.yaml"
    with pytest.raises(CurationError) as excinfo:
        main(
            [
                "--pairs-csv", str(bad_csv),
                "--sample-rate", "16000",
                "--out", str(bad_out),
                "--repo-root", str(tmp_path),
            ]
        )
    assert "rubato" in str(excinfo.value)
    assert not bad_out.exists()  # nothing written on invalid input
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run python -m pytest tests/audio_teacher/test_build_manifest.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'audio_teacher.build_manifest'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `model/src/audio_teacher/build_manifest.py`:

```python
"""SHALLOW curation script: contrast-pair CSV -> YAML probe manifest.

Deliberately not a rendering framework (Gate 0 scope decision). CSV
columns: id,axis,population,clip_a,clip_b,degraded,description with clip
paths relative to --repo-root (default: model/). The generated YAML is
round-tripped through load_manifest before this script reports success,
so a written manifest is loadable by construction (clips exist locally,
headers valid, schema well-formed).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml

from audio_teacher.manifest import AXES, MODEL_ROOT, POPULATIONS, load_manifest

_COLUMNS = ["id", "axis", "population", "clip_a", "clip_b", "degraded", "description"]


class CurationError(Exception):
    """A CSV row is invalid; nothing is written."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-csv", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=MODEL_ROOT)
    args = parser.parse_args(argv)

    with args.pairs_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != _COLUMNS:
            raise CurationError(
                f"{args.pairs_csv}: header must be {','.join(_COLUMNS)}, "
                f"got {reader.fieldnames}"
            )
        rows = list(reader)
    if not rows:
        raise CurationError(f"{args.pairs_csv}: no pairs")

    entries = []
    for i, row in enumerate(rows):
        if row["axis"] not in AXES:
            raise CurationError(
                f"{args.pairs_csv} row {i}: axis {row['axis']!r} not in {AXES}"
            )
        if row["population"] not in POPULATIONS:
            raise CurationError(
                f"{args.pairs_csv} row {i}: population {row['population']!r} "
                f"not in {POPULATIONS}"
            )
        if row["degraded"] not in ("a", "b"):
            raise CurationError(
                f"{args.pairs_csv} row {i}: degraded must be 'a' or 'b', "
                f"got {row['degraded']!r}"
            )
        entries.append({k: row[k] for k in _COLUMNS})

    args.out.write_text(
        yaml.safe_dump(
            {"schema_version": 1, "sample_rate": args.sample_rate, "pairs": entries},
            sort_keys=False,
        )
    )
    manifest = load_manifest(args.out, repo_root=args.repo_root)
    print(f"wrote {args.out} ({len(manifest.pairs)} pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Note: the YAML pair mapping uses key `id` (CSV column name), matching the loader's schema. `entries` keeps CSV column names, which are exactly the schema keys.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run python -m pytest tests/audio_teacher/test_build_manifest.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add model/src/audio_teacher/build_manifest.py model/tests/audio_teacher/test_build_manifest.py && git commit -m "feat(audio-teacher): shallow CSV->YAML manifest curation script (#127)"
```

---

### Task 15: Probe CLI — offline end-to-end run writes the report
**Group:** G (alone; depends on all of A–F)

**Behavior being verified:** `python -m audio_teacher.probe --manifest M --recorded R` runs fully offline: loads the manifest, replays responses, writes a population-partitioned `report.json` plus `run_meta.json`, prints the verdict, and returns exit code 1 for FAIL (gate closed).
**Interface under test:** `audio_teacher.probe.main`

**Files:**
- Create: `model/src/audio_teacher/probe.py`
- Test: `model/tests/audio_teacher/test_probe.py`

- [ ] **Step 1: Write the failing test**

Create `model/tests/audio_teacher/test_probe.py`:

```python
"""Offline end-to-end probe runs through the CLI entrypoint."""
from __future__ import annotations

import json

from audio_teacher.probe import main


def test_offline_probe_writes_population_partitioned_report(
    tmp_path, manifest_factory
):
    manifest_path = manifest_factory(
        [{"id": "p1", "axis": "pedaling", "population": "real", "degraded": "a"}]
    )
    recorded = tmp_path / "recorded.jsonl"
    recorded.write_text(
        json.dumps({"pair_id": "p1", "text": "blurred pedal.\nANSWER: A"}) + "\n"
    )
    run_dir = tmp_path / "run"

    rc = main(
        [
            "--manifest", str(manifest_path),
            "--repo-root", str(tmp_path),
            "--recorded", str(recorded),
            "--run-dir", str(run_dir),
        ]
    )

    report = json.loads((run_dir / "report.json").read_text())
    assert report["cells"]["pedaling/real"] == {
        "n": 1, "correct": 1, "unparseable": 0,
        "accuracy": 1.0, "unparseable_rate": 0.0,
    }
    # 1 pair < MIN_REAL_PAIRS_PER_AXIS: uncertain defaults to closed.
    assert report["verdict"] == "FAIL"
    assert rc == 1
    meta = json.loads((run_dir / "run_meta.json").read_text())
    assert meta["mode"] == "recorded"
    assert meta["spent_usd"] == 0.0
    assert (run_dir / "responses.jsonl").exists()
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run python -m pytest tests/audio_teacher/test_probe.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'audio_teacher.probe'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `model/src/audio_teacher/probe.py`:

```python
"""Gate 0 probe CLI: contrast-pair manifest -> per-pair Inkling judgments
-> deterministic report.json + run_meta.json.

Offline mode (--recorded) replays canned responses and never touches the
network; live mode requires the Tinker SDK plus explicit USD token rates
(there is no silent default rate). Responses append to
<run-dir>/responses.jsonl after every call, so a crashed or
budget-stopped run resumes by skipping answered pairs. Tinker API errors
and BudgetExceededError propagate as-is -- saved responses ARE the run
state. The report is only written once every manifest pair has a
response. Exit code: 0 on PASS, 1 on FAIL.

Usage:
    uv run python -m audio_teacher.probe --manifest data/manifests/gate0.yaml \
        --recorded runs/recorded.jsonl                      # offline re-score
    uv run python -m audio_teacher.probe --manifest data/manifests/gate0.yaml \
        --max-spend 50.0 --usd-per-1m-input-tokens X \
        --usd-per-1m-output-tokens Y                        # live (user-run)
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

from audio_teacher.budget import BudgetGuard
from audio_teacher.client import ProbeClient, RecordedResponseClient
from audio_teacher.manifest import load_manifest
from audio_teacher.scorer import render_report, score_responses

MODEL_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_ROOT = MODEL_ROOT / "data" / "results" / "audio_teacher"


def _read_responses(path: Path) -> dict[str, str]:
    responses: dict[str, str] = {}
    if path.exists():
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                responses[rec["pair_id"]] = rec["text"]
    return responses


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--repo-root", type=Path, default=None,
        help="root that manifest clip paths are relative to (default: model/)",
    )
    parser.add_argument(
        "--run-dir", type=Path, default=None,
        help="default: model/data/results/audio_teacher/<manifest stem>",
    )
    parser.add_argument(
        "--recorded", type=Path, default=None,
        help="offline mode: replay responses from this JSONL instead of Tinker",
    )
    parser.add_argument("--max-spend", type=float, default=50.0)
    parser.add_argument("--usd-per-1m-input-tokens", type=float, default=None)
    parser.add_argument("--usd-per-1m-output-tokens", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args(argv)

    load_kwargs = {}
    if args.repo_root is not None:
        load_kwargs["repo_root"] = args.repo_root
    manifest = load_manifest(args.manifest, **load_kwargs)

    if args.recorded is not None:
        client: ProbeClient = RecordedResponseClient(args.recorded)
    else:
        if args.usd_per_1m_input_tokens is None or args.usd_per_1m_output_tokens is None:
            parser.error(
                "live mode requires --usd-per-1m-input-tokens and "
                "--usd-per-1m-output-tokens (no silent default rate exists; "
                "current rates: see the Gate 0 issue)"
            )
        from audio_teacher.tinker_client import TinkerProbeClient

        client = TinkerProbeClient(
            sample_rate=manifest.sample_rate,
            usd_per_1m_input_tokens=args.usd_per_1m_input_tokens,
            usd_per_1m_output_tokens=args.usd_per_1m_output_tokens,
            max_tokens=args.max_tokens,
        )

    run_dir = args.run_dir if args.run_dir is not None else (
        DEFAULT_RUNS_ROOT / args.manifest.stem
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    responses_path = run_dir / "responses.jsonl"
    responses = _read_responses(responses_path)

    guard = BudgetGuard(max_spend_usd=args.max_spend)
    with responses_path.open("a") as out:
        for pair in manifest.pairs:
            if pair.pair_id in responses:
                continue  # resume: answered in a previous run
            guard.precheck(client.estimate_cost_usd(pair))
            resp = client.ask(pair)
            guard.record(resp.cost_usd)
            out.write(
                json.dumps(
                    {
                        "pair_id": resp.pair_id,
                        "text": resp.text,
                        "input_tokens": resp.input_tokens,
                        "output_tokens": resp.output_tokens,
                        "cost_usd": resp.cost_usd,
                    }
                )
                + "\n"
            )
            out.flush()
            responses[resp.pair_id] = resp.text

    report = score_responses(manifest, responses)
    (run_dir / "report.json").write_text(render_report(report))
    (run_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(),
                "manifest": str(args.manifest),
                "mode": "recorded" if args.recorded is not None else "tinker",
                "spent_usd": guard.spent_usd,
                "max_spend_usd": args.max_spend,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"verdict: {report['verdict']}")
    for reason in report["verdict_reasons"]:
        print(f"  - {reason}")
    print(f"report: {run_dir / 'report.json'}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run python -m pytest tests/audio_teacher/test_probe.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add model/src/audio_teacher/probe.py model/tests/audio_teacher/test_probe.py && git commit -m "feat(audio-teacher): probe CLI -- manifest to deterministic report, offline-capable (#127)"
```

---

### Task 16: Probe resume — answered pairs are never re-asked
**Group:** H (alone; depends on G)

**Behavior being verified:** With a pre-existing `responses.jsonl`, a re-run only queries unanswered pairs (proved by giving the client no recording for the answered pair — re-asking would raise) and the final report covers all pairs.
**Interface under test:** `audio_teacher.probe.main`

**Files:**
- Modify: `model/src/audio_teacher/probe.py` (only if the test exposes a gap — Task 15 shipped the resume loop)
- Test: `model/tests/audio_teacher/test_probe.py`

- [ ] **Step 1: Write the failing test**

Append to `model/tests/audio_teacher/test_probe.py`:

```python
def test_resume_skips_pairs_already_answered(tmp_path, manifest_factory):
    manifest_path = manifest_factory(
        [
            {"id": "p1", "axis": "pedaling", "population": "real", "degraded": "a"},
            {"id": "p2", "axis": "pedaling", "population": "real", "degraded": "b"},
        ]
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "responses.jsonl").write_text(
        json.dumps({"pair_id": "p1", "text": "ANSWER: A"}) + "\n"
    )
    # The recorded fixture deliberately LACKS p1: if the driver re-asked the
    # already-answered pair, RecordedResponseClient would raise KeyError.
    recorded = tmp_path / "recorded.jsonl"
    recorded.write_text(json.dumps({"pair_id": "p2", "text": "ANSWER: B"}) + "\n")

    rc = main(
        [
            "--manifest", str(manifest_path),
            "--repo-root", str(tmp_path),
            "--recorded", str(recorded),
            "--run-dir", str(run_dir),
        ]
    )

    report = json.loads((run_dir / "report.json").read_text())
    assert report["cells"]["pedaling/real"]["n"] == 2
    assert report["cells"]["pedaling/real"]["correct"] == 2
    assert rc == 1  # still under MIN_REAL_PAIRS_PER_AXIS: gate stays closed
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run python -m pytest tests/audio_teacher/test_probe.py -v
```
Expected: the Task 15 implementation already resumes, so this may PASS immediately. Honesty check as in Tasks 8/9/11: temporarily seed `responses.jsonl` with pair id `"px"` instead of `"p1"` and confirm the test FAILS with KeyError, then restore. If the real test fails, proceed to Step 3.

- [ ] **Step 3: Implement the minimum to make the test pass**

No change expected. If resume is broken, fix only the `if pair.pair_id in responses: continue` path and `_read_responses` in `model/src/audio_teacher/probe.py`.

- [ ] **Step 4: Run test — verify it PASSES, then run the whole suite**

```bash
cd model && uv run python -m pytest tests/audio_teacher/test_probe.py -v
cd model && uv run python -m pytest tests/audio_teacher -v
```
Expected: PASS (all audio_teacher tests green)

- [ ] **Step 5: Commit**

```bash
git add model/tests/audio_teacher/test_probe.py model/src/audio_teacher/probe.py && git commit -m "test(audio-teacher): lock resume-from-responses behavior (#127)"
```

---

### Task 17: Create label + epic issue
**Group:** I (sequential; first of the GitHub tasks; may run concurrently with code groups)

**Behavior being verified:** Command-work; verified by `gh` output (design: "Migration is NOT code-tested"). No repo diff, no commit.

- [ ] **Step 1: Create the label**

```bash
gh label create "epic:audio-teacher" --color 5319e7 --description "Audio-native teacher gated program (Inkling/Tinker era)" || gh label list | grep audio-teacher
```

- [ ] **Step 2: Create the epic issue**

```bash
gh issue create \
  --title "EPIC: Audio-native teacher — gated program (Inkling/Tinker era)" \
  --label "epic:audio-teacher" \
  --body "$(cat <<'EOF'
Durable roadmap for the audio-native teacher program. Supersedes the Qwen 5-stage teacher-finetune plan (`epic:teacher-finetune`) and the MuQ/Aria encoder-v2 training plan (`epic:model-v2`). Pivot tracking issue: #127.

## (a) Architecture decision

One audio-native LoRA-tuned open multimodal model (Inkling-on-Tinker class) collapses ear + teacher into a single model that hears the performance and reasons about it.

- The harness claim-verifier stays EXTERNAL to the model and gates every claim the model makes. The model never grades its own homework.
- MuQ retires as a graded 6-dim stream; it survives only as frozen instrumentation (probes, sanity baselines, drift checks).
- The transcription stream (Transkun, per #125) is untouched by this program.

## (b) 4-layer data plan

- **Layer 1 — perception:** T2 competition ordinal data: 9,059 segments COMPLETE (Chopin 2021 + Cliburn 2022; see docs/model/01-data.md). T5 YouTube skill: 2/16 pieces — scale-up re-targeted onto #33. MAESTRO T3 is the source for future contrastive renderings (wave 2).
- **Layer 2 — grounding:** auto-generated from the score-follower lane (#108 epic; #119 HMM shipped 2026-07; #126 timing-aware HMM open) + the verifier lanes (#101/#67). Explicitly NOT this program's code — this program consumes their outputs.
- **Layer 3 — judgment:** scarce human/expert data — masterclass/jury commentary, rubric-filtered distillation. The scarcity is the moat and the bottleneck.
- **Layer 4 — pedagogy/voice:** model-agnostic; #32 (rubric calibration r>=0.7 gate) and #40 (corpus composition) port unchanged.

## (c) Gate ladder (ex-ante kill criteria; uncertain defaults to CLOSED)

- **Gate 0 — Inkling piano-perception probe.** $50 hard cap. Kill criteria live in the Gate 0 issue body, fixed before any run. FAIL => activate the DIY contingency issue; the Qwen/MuQ-finetune plans never resurrect.
- **Gate 1 — ordinal RL on T2** vs a frozen-encoder Bradley-Terry comparator. Separate funding decision (~$500). Opens only on Gate 0 PASS.
- **Gate 2 — grounded-claim SFT.** Blocked on the follower/verifier grounding lanes (#108/#126, #101/#67).
- **Gate 3+ — pedagogy.** Blocked on #32 (rubric calibration r>=0.7).

## (d) Wave-2 issue drafts (checklist — promote to a real issue ONLY when its gate opens)

- [ ] Gate 1: ordinal RL on T2 vs frozen-encoder Bradley-Terry comparator
- [ ] Contrastive rendering engine (MAESTRO T3 -> controlled degradations)
- [ ] Grounding SFT factory (follower/verifier outputs -> grounded-claim training pairs)
- [ ] Judgment-data corpus (masterclass/jury commentary curation)
- [ ] Pedagogy port (#32 gate + #40 corpus into the audio-native stack)
- [ ] Serving decision (incl. Inkling-Small eval; production stays glm V6 until pedagogy gates pass AND a serveable variant exists)

## (e) Standing policies

- PercePiano is eval-only: NEVER gating, NEVER trained on.
- $50 hard stop before Gate 1; the probe harness enforces it pre-call (BudgetGuard).
- Production serving stays the glm V6 harness until pedagogy gates pass AND a serveable variant exists.
- Split manifests are committed artifacts created BEFORE any training and shared by all contenders (the fold-leak lesson).
- Synthetic-rendered and real clip populations are always reported separately, never pooled (the #21 synthetic-gap scar).
- A failed gate activates the DIY contingency issue — old plans close permanently and never resurrect.
EOF
)"
```

- [ ] **Step 3: Verify**

```bash
gh issue list --label epic:audio-teacher --state open --json number,title
```
Expected: the epic issue is listed. Note its number for Tasks 18/19 (recoverable any time via the same command).

---

### Task 18: Create Gate 0 + contingency issues
**Group:** I (after Task 17)

- [ ] **Step 1: Resolve the epic number**

```bash
EPIC=$(gh issue list --label epic:audio-teacher --state open --search "EPIC: Audio-native teacher" --json number --jq '.[0].number'); echo "EPIC=$EPIC"
```

- [ ] **Step 2: Create the Gate 0 issue**

```bash
gh issue create \
  --title "Gate 0: Inkling piano-perception probe" \
  --label "epic:audio-teacher" \
  --body "$(cat <<EOF
Part of the audio-native teacher gated program (#$EPIC). The probe harness (\`model/src/audio_teacher/\`, built on #127) is the deliverable of wave 1; THIS issue tracks the funded probe run, executed manually.

## Question

Can base Inkling (no fine-tune) hear piano-performance contrasts at all? If not, LoRA on top of it is dead on arrival and the DIY contingency activates.

## Ex-ante kill criteria (FIXED BEFORE ANY RUN; mirrored as constants in \`model/src/audio_teacher/scorer.py\` — no override knob exists)

- PASS requires, for EVERY probed axis (pedaling, dynamics, phrasing), on the REAL clip population:
  - pairwise accuracy >= 0.70 (KILL_THRESHOLD; chance is 0.50 on forced A/B choice)
  - >= 20 real contrast pairs (MIN_REAL_PAIRS_PER_AXIS)
  - unparseable-response rate <= 0.10 (MAX_UNPARSEABLE_RATE)
- FAIL on any violation. Ambiguous/insufficient data => FAIL. Uncertain defaults to CLOSED.
- Synthetic-rendered and real populations are ALWAYS reported separately and NEVER pooled; synthetic cells never open the gate (the #21 synthetic-gap scar). Example kill: base Inkling cannot distinguish over-pedaled vs clean on >= 70% of real contrast pairs => FAIL.
- Budget: \$50.00 hard cap, enforced pre-call by BudgetGuard (raises BEFORE the overshooting call). A budget stop preserves run state; a re-run resumes.

## Run protocol

1. Curate contrast pairs: \`uv run python -m audio_teacher.build_manifest --pairs-csv pairs.csv --sample-rate 16000 --out data/manifests/gate0_probe.yaml\` (T2 raw audio is R2-offloaded; the loader error names the exact rclone rehydrate command).
2. Install the SDK: \`cd model && uv add tinker tinker-cookbook\` — verify current package names + token pricing on tinker-docs.thinkingmachines.ai first (deliberately not in pyproject).
3. Run: \`uv run python -m audio_teacher.probe --manifest data/manifests/gate0_probe.yaml --max-spend 50.0 --usd-per-1m-input-tokens <rate> --usd-per-1m-output-tokens <rate>\`
4. Post report.json + run_meta.json here.

## Decision rule

- PASS => open the Gate 1 funding decision (ordinal RL vs BT comparator, ~\$500; wave-2 draft on #$EPIC).
- FAIL => activate the DIY contingency issue; Qwen/MuQ-finetune plans stay closed permanently.
EOF
)"
```

- [ ] **Step 3: Create the parked contingency issue**

```bash
gh issue create \
  --title "Contingency: DIY audio-native path (activates only on a failed gate)" \
  --label "epic:audio-teacher,blocked" \
  --body "$(cat <<EOF
PARKED. Part of the audio-native teacher gated program (#$EPIC). This issue activates ONLY when an Inkling gate FAILS — it is the designated fallback; the closed Qwen/MuQ-finetune plans never resurrect.

## Scope on activation (survey first, no training)

- MuQ / MuQ-MuLan encoder + trained projector into an open LLM (the LLaVA recipe, for audio)
- Qwen3-Omni lineage
- SALMONN / Audio-Flamingo lineage

## Blocker

No work of any kind before a gate on #$EPIC formally FAILS. Blocked-on: Gate 0 verdict.
EOF
)"
```

- [ ] **Step 4: Verify**

```bash
gh issue list --label epic:audio-teacher --state open --json number,title,labels
```
Expected: three issues (epic, Gate 0, contingency — contingency also carries `blocked`).

---

### Task 19: Migration batch — close dead plans, relabel survivors
**Group:** I (after Task 18)

- [ ] **Step 1: Capture the BEFORE snapshot (goes into ship notes)**

```bash
EPIC=$(gh issue list --label epic:audio-teacher --state open --search "EPIC: Audio-native teacher" --json number --jq '.[0].number')
gh issue list --state open --limit 100 --json number,title,labels --jq '.[] | "\(.number)\t\(.title)\t\([.labels[].name] | join(","))"' | sort -n > /tmp/issue-migration-before.txt
cat /tmp/issue-migration-before.txt
```

- [ ] **Step 2: Close the nine dead issues, each linking the epic as successor**

```bash
for N in 71 79 80 81 82 83 84 16 55; do
  gh issue close "$N" --reason "not planned" --comment "Closed by the audio-native teacher pivot (#127): superseded by the gated program epic #$EPIC. Old plans close permanently; a failed Inkling gate activates the DIY contingency issue, never this plan."
done
```

- [ ] **Step 3: Relabel #32 #33 #40 (keep `deferred`), with a retarget comment each**

```bash
gh issue edit 32 --remove-label "epic:teacher-finetune" --add-label "epic:audio-teacher"
gh issue comment 32 --body "Retargeted by the audio-native teacher pivot (#127 -> epic #$EPIC): this is the Layer-4 pedagogy gate, unchanged — rubric calibration r>=0.7 still gates Gate 3+. Stays deferred."

gh issue edit 33 --remove-label "epic:teacher-finetune" --add-label "epic:audio-teacher"
gh issue comment 33 --body "Retargeted by the audio-native teacher pivot (#127 -> epic #$EPIC): now the Layer-1 T5 skill-data scale-up (2/16 pieces done) feeding the audio-native perception layer. Stays deferred."

gh issue edit 40 --remove-label "epic:teacher-finetune" --add-label "epic:audio-teacher"
gh issue comment 40 --body "Retargeted by the audio-native teacher pivot (#127 -> epic #$EPIC): Layer-4 corpus-composition work, model-agnostic, ports unchanged. Stays deferred."
```

- [ ] **Step 4: Capture the AFTER snapshot and post the verification checklist to #127**

```bash
gh issue list --state open --limit 100 --json number,title,labels --jq '.[] | "\(.number)\t\(.title)\t\([.labels[].name] | join(","))"' | sort -n > /tmp/issue-migration-after.txt
diff /tmp/issue-migration-before.txt /tmp/issue-migration-after.txt || true
gh issue list --label "epic:teacher-finetune" --state open --json number
gh issue list --label "epic:model-v2" --state open --json number
```
Expected: both label queries return `[]` (no open issues left under either dead epic label); the diff shows exactly: 9 issues gone, 3 new `epic:audio-teacher` issues, #32/#33/#40 relabeled.

```bash
gh issue comment 127 --body "STATE: wave-1 issue migration executed — epic #$EPIC + Gate 0 + contingency created; #71 #79 #80 #81 #82 #83 #84 #16 #55 closed not-planned with successor link; #32 #33 #40 relabeled epic:audio-teacher (deferred kept). Verified: epic:teacher-finetune and epic:model-v2 have zero open issues. Next: merge the probe harness branch (issue-127-audio-teacher-pivot)."
```

---

## Plan Self-Review (completed)

1. **Spec coverage:** epic/Gate 0/contingency/migration -> Tasks 17–19; harness modules audio/manifest/prompts/client/tinker_client/budget/scorer/probe/build_manifest -> Tasks 1–16; every spec test bullet (loader missing/malformed, scorer partition + verdicts, budget pre-call raise, determinism, offline e2e, resume) has a named task. Migration is deliberately not code-tested (spec + design).
2. **Placeholder scan:** no TBD/TODO; all steps carry exact code or exact commands.
3. **Type consistency:** `ContrastPair(pair_id, axis, population, clip_a, clip_b, degraded, description)`, `ProbeManifest(sample_rate, pairs)`, `ProbeResponse(pair_id, text, input_tokens, output_tokens, cost_usd)`, `validate_wav(path, expected_sample_rate)`, `load_manifest(manifest_path, repo_root, offload_registry)`, `score_responses(manifest, responses)`, `render_report(report)`, `BudgetGuard(max_spend_usd).precheck/record/spent_usd`, `main(argv) -> int` — used identically across all tasks.
4. **Group correctness:** A:1; B:2,3,4; C:5,6,7; D:8,10,12; E:9,11; F:13,14; G:15; H:16; I:17→18→19 (matches the Task Groups table and every task header). No parallel group shares a file.
5. **Vertical slices:** every code task = one test (single function, parametrization allowed) + one implementation + one commit. Tasks 8/9/11/16 are lock-in tests over behavior shipped by an earlier vertical slice in the same file lineage; each carries an explicit mutation-based honesty check so a test that can never fail is caught.
6. **Behavior tests:** all tests go through public interfaces (`validate_wav`, `load_manifest`, `build_question`/`parse_choice`, `RecordedResponseClient`, `BudgetGuard`, `score_responses`/`render_report`, `probe.main`, `build_manifest.main`); no internal mocking, no private-method calls, no network.

## Challenge Review

### CEO Pass

**Premise Challenge.** Right problem: yes. Two dead issue graphs (`epic:teacher-finetune`, `epic:model-v2`) currently masquerade as the live roadmap, and nothing tests whether base Inkling can hear piano contrasts before money is committed to Gate 1. Real pain without this: a future session (or the user, mid-funding-decision) could act on #71/#79/#80/#81/#82/#83/#84/#16/#55 as if they were current, or fund Gate 1 without an ex-ante-gated Gate 0 result. Direct path: yes — the plan does exactly the two things named in the spec goal (issue-graph migration, offline probe harness) and nothing else. Existing coverage: `model/src/follower_bench/` is correctly named and followed as the structural precedent (flat modules, `tests/<pkg>/`, `__file__`-anchored paths, manifest-in/report-out); `model/src/paths.py::ensure_local` is correctly identified as the un-importable precedent the manifest loader re-implements (verified: `paths.py` is genuinely absent from the hatch `packages` list, so the design's claim is accurate, not just asserted).

**Scope Check.** MVP-cuttable: `build_manifest.py` (Task 14) is the most deferrable piece — Gate 0 curation could start from a hand-written YAML manifest without the CSV adapter, and the design already labels it SHALLOW/deliberate. Everything else (audio validation, manifest loader, budget guard, scorer, probe CLI) is load-bearing for the harness's one claim ("offline-testable end to end"). Hardest problem (population-partitioned, non-poolable scoring with an ex-ante-closed verdict) is being solved directly, not avoided — this is the one part of the design with a named prior failure (#21 synthetic-gap) driving it, and it gets the most test coverage (Tasks 10/11/13). Plan matches spec goal with no drift I can find: every spec bullet in "Solution (from the user's perspective)" traces to a task; the Out-of-Scope list in both plan and spec is identical and is respected in every task (no Gate 1 code, no BT comparator, no rendering engine appear anywhere in Tasks 1–19). File-count complexity smell: 9 new `src/` modules + 10 new test files exceeds the "8 files" trigger, but each module maps 1:1 to a named responsibility in the spec's Modules section and mirrors `follower_bench`'s existing flat layout — this is breadth inherent to a small harness with clean seams, not invented complexity. Not a blocker.

**Twelve-Month Alignment.**
```
CURRENT STATE                          THIS PLAN                              12-MONTH IDEAL
Two dead issue epics                → epic:audio-teacher (roadmap +      →   Gate 0 run executed, Gate 1
(#16/#55 Qwen finetune,                gate ladder) + Gate 0 issue +          funding decision made on real
#71/#79-84 MuQ/Aria v2)                contingency issue; 9 issues            evidence, wave-2 issues
                                        closed, 3 relabeled                   promoted from the checklist

No offline-testable way to           → model/src/audio_teacher/ (offline-  →   Harness reused for Gate 1's
probe "can Inkling hear this"          testable manifest→judgment→          ordinal-RL eval and Gate 2's
before spending money                  report pipeline, Tinker behind a     grounded-claim SFT eval
                                        Protocol seam)
```
Moves toward the ideal; no tech debt identified that conflicts with it. The Tinker-behind-a-Protocol seam is specifically what lets Gate 1+ reuse this harness rather than rebuilding it.

**Alternatives Check.**
```
[QUESTION] — The design doc doesn't restate the alternatives considered during the
             /brainstorm session (issue #127's body references "Option C" of a
             two-wave hybrid issue-graph approach, implying B and possibly A were
             discussed and rejected). If that reasoning lives only in the brainstorm
             transcript and not in a durable doc, a future engineer reading
             docs/specs/2026-07-23-audio-teacher-pivot-design.md alone won't know why
             the two-wave hybrid won over a single flat issue dump or a fully-deferred
             wave-2. Low severity since #127's issue body captures the "hardest
             decision resolved" (fallback posture) already.
```

### Engineering Pass

**Architecture.** Data flow is linear and matches the stated design:
```
manifest.yaml → load_manifest (schema + WAV validation + offload check)
                      │
                      ▼
              ProbeManifest.pairs ──► probe.main loop ──► ProbeClient.ask (recorded|tinker)
                      │                     │                       │
                      │              BudgetGuard.precheck ──raises──► BudgetExceededError
                      │                     │                     (propagates; responses.jsonl
                      │                     ▼                      so far preserved)
                      │            responses.jsonl (append+flush per pair; resume source)
                      │                     │
                      ▼                     ▼
              score_responses(manifest, responses) ──► report.json (render_report, deterministic)
                                                    └──► run_meta.json (mode, spend, timestamp)
```
Verified against actual code, not assumed: `model/src/paths.py` confirmed absent from `model/pyproject.toml`'s `[tool.hatch.build.targets.wheel] packages` list (line 104), so the design's claim that `ensure_local` isn't importable from the installed package — and that `manifest.py` must re-implement it — is accurate. The re-implementation (`_ensure_clip_local`) also verified to genuinely extend `ensure_local`'s behavior: production `ensure_local` does an **exact-key** lookup (`manifest.get("entries", {}).get(rel)`, `model/src/paths.py:40`) with no directory-prefix matching, so a per-file lookup under an offloaded directory like `data/raw/competition/foo.wav` would silently miss the registry entry keyed `data/raw/competition` today. The plan's `_ensure_clip_local` adds `rel == registered or rel.startswith(registered + "/")`, which is a real, deliberate fix for T2 curation (files, not directories, get validated per-clip) — correctly called out in the design as an extension, not a copy-paste bug.

No security-relevant input flows found: no SQL, no shell execution (rclone/regen commands are only ever printed in exception messages, never invoked), no unsanitized user input reaches an interpreter. Budget/scoring/manifest logic is pure Python over local files.

**Module Depth Audit** (reading each file's planned interface vs. hidden implementation):
- `audio.py` — Interface: `validate_wav(path, rate) -> WavInfo`, `MalformedClipError`. Hidden: RIFF parsing, 4 distinct failure modes, dual truncation detection (EOFError catch + explicit byte-count check). **DEEP.**
- `manifest.py` — Interface: `load_manifest(path, repo_root, offload_registry) -> ProbeManifest`. Hidden: schema validation (6 branches), duplicate-id tracking, offload-registry prefix matching, per-clip WAV validation fan-out (~150 LOC). **DEEP.**
- `prompts.py` — Interface: `build_question(axis)`, `parse_choice(text)`. Hidden: per-axis phrasing table, tolerant regex parsing, last-match-wins semantics. **DEEP**, verging on thin (two ~10-line functions) but each hides a genuine contract (the axis→question table, the parsing tolerance rules) that would be duplicated at every call site otherwise.
- `client.py` — Interface: `ProbeClient` protocol, `RecordedResponseClient`. Hidden: JSONL bookkeeping, loud-on-incomplete-fixture semantics. **DEEP** for a 2-method fake; correctly minimal.
- `budget.py` — Interface: `BudgetGuard(cap)`, `.precheck`, `.record`, `.spent_usd`. Hidden: the projection-before-charge invariant, the deliberate absence of a warn-mode. **DEEP** — small but the single invariant it enforces is exactly the one the program depends on (never overshoot $50).
- `scorer.py` — Interface: `score_responses`, `render_report`, constants. Hidden: per-cell aggregation, ex-ante verdict rules, deterministic serialization. **DEEP** — the most load-bearing module, appropriately the most tested.
- `tinker_client.py` — Interface: `TinkerProbeClient(...)`. Hidden: renderer/tokenizer wiring, `AudioPointer` message assembly, cost accounting. **UNCLEAR** — see Presumption Inventory; the hidden implementation is built against an SDK surface (`tinker_cookbook.model_info.get_recommended_renderer_name`, `tml_renderers.chat.AudioPointer`, `prompt.length`, `renderer.parse_response`) that is never exercised by any test and whose exact shape is unverified against current Tinker/Inkling docs. Depth can't be assessed for correctness, only for intent; the module is honestly labeled "never imported by tests except the not-installed path," which is the right mitigation given the run itself is out of scope.
- `probe.py` — Interface: `main(argv) -> int`. Hidden: client selection, resume logic, budget wiring, report/meta writing, exit-code mapping. **DEEP** — correctly the composition root.
- `build_manifest.py` — Interface: `main(argv) -> int`. Declared and confirmed **SHALLOW by design** (thin CSV→YAML adapter delegating all real validation to `load_manifest`); acceptable per the plan's own stated Gate-0-is-a-probe-not-a-framework rationale.

**Code Quality.**
- `MODEL_ROOT = Path(__file__).resolve().parents[2]` is defined independently in both `manifest.py` (Task 6) and `probe.py` (Task 15) rather than imported from one place. Minor DRY violation — 2 occurrences, below the 3+ threshold that would make this a hard flag, but worth a one-line note since `probe.py` already imports from `manifest.py` and could reuse its `MODEL_ROOT`.
- No catch-all exception handling anywhere in the plan's code (`except (wave.Error, EOFError)`, `except ImportError` are both narrow and intentional) — matches CLAUDE.md's "explicit exception handling over silent fallbacks."
- Edge cases: empty manifest pairs list rejected (`ManifestError`), zero-length WAV rejected, empty `parse_choice("")` returns `None` (tested), missing/extra response sets rejected (`ProbeIncompleteError`). No gaps found in the edge cases the tests actually exercise.
- Follows `__file__`-anchored default paths per CLAUDE.md/MEMORY.md's explicit gotcha (`feedback_anchor_default_paths_to_module.md`) — verified in `manifest.py`, `probe.py`, `build_manifest.py` (`--repo-root` defaults to `MODEL_ROOT`, never CWD).

**Test Philosophy Audit.** All tests call public module-level functions/classes (`validate_wav`, `load_manifest`, `build_question`/`parse_choice`, `RecordedResponseClient(...).ask`, `BudgetGuard(...).precheck`, `score_responses`/`render_report`, `probe.main`, `build_manifest.main`). No internal mocking of collaborators — `RecordedResponseClient` is a real fake at the *external* boundary (the sampling call), which is the correct thing to fake, not an internal collaborator of the module under test. No shape-only tests: every assertion checks a specific value (accuracy numbers, exact error substrings, exit codes), not just "field exists."

**Vertical Slice Audit.**
```
[RISK] (confidence: 7/10) — Task 6 ships a single vertical-slice commit whose
       implementation (schema_version/sample_rate/pairs/required-keys/duplicate-id/
       axis/population/degraded validation + offload-registry prefix matching +
       per-clip WAV validation, ~150 LOC) is driven by exactly one test (the valid-
       manifest happy path). The six schema-violation branches and the offload-hint
       branch are *implemented* in Task 6 but only *tested* three tasks later, in
       Tasks 8 and 9. This is "Bundled commits" scope (section 9's RISK category,
       not its BLOCKER category — no test is written without an implementation, and
       no test is written that would pass without the code existing), but it does
       mean Task 6's commit, on its own, ships materially untested code for two
       task-cycles. Same pattern, smaller degree, in Task 10 (scorer verdict logic
       tested fully only in Task 11) and Task 15 (resume loop tested fully only in
       Task 16). Mitigation already in the plan: Tasks 8/9/11/16 each carry an
       explicit mutation-based "honesty check" (temporarily break the assertion,
       confirm the test fails, restore) specifically to guard against a lock-in
       test that could never fail — this is a genuine safeguard against the
       classic "test that tests nothing" failure mode, and it's disclosed
       explicitly in the plan's own self-review (item 5). Watch during execution:
       if a build-agent skips the honesty-check step under time pressure (it reads
       as optional busywork once the assertions already pass), the safeguard is
       lost silently. Fallback: require the honesty-check commands actually be run
       and their output captured, not just the step checkbox ticked.
```
No task defers implementation past its own step (the inverse anti-pattern) — checked every task's Step 3 against its Step 1 test.

**Test Coverage Gaps.**
```
[+] model/src/audio_teacher/manifest.py
    │
    ├── load_manifest()
    │   ├── [TESTED] ★★  valid manifest, ordered pairs, resolved clips — Task 6
    │   ├── [TESTED] ★★★ missing+offloaded clip → rclone hint — Task 8
    │   ├── [TESTED] ★★  missing+unregistered clip → path named — Task 8
    │   ├── [TESTED] ★★★ 6 schema-violation branches — Task 9
    │   └── [GAP]        clip_a == clip_b (degenerate pair, same file both sides) —
    │                    no test either way; not a correctness bug (scorer would
    │                    still work) but worth a one-line curation-time guard note
    │                    if build_manifest.py is ever pointed at bad CSV data.
    │
[+] model/src/audio_teacher/scorer.py
    ├── score_responses()
    │   ├── [TESTED] ★★★ partitioned cells, never pooled — Task 10
    │   ├── [TESTED] ★★★ 5 ex-ante verdict rule branches — Task 11
    │   └── [TESTED] ★★  ProbeIncompleteError on missing/extra — Task 10
    └── render_report()
        └── [TESTED] ★★★ order-independence + no-timestamp — Task 13

[+] model/src/audio_teacher/probe.py
    └── main()
        ├── [TESTED] ★★  offline e2e, report+meta written, exit code — Task 15
        ├── [TESTED] ★★★ resume skips answered pairs — Task 16
        └── [GAP]        live-mode arg validation (missing --usd-per-1m-*-tokens
                         triggers parser.error) has no test. Non-critical: it's a
                         CLI usage guard, not a data-mutation path, and the plan's
                         own scope excludes running the probe. RISK, not BLOCKER.
```

**Failure Modes.** Every I/O and async-adjacent operation reviewed:
- Budget overshoot: raises before the call, responses saved so far — recoverable, loud (BudgetExceededError propagates to caller; no catch anywhere silences it).
- Process killed mid-run: `responses.jsonl` is appended+flushed per pair, so a resumed run picks up where it left off (Task 16 proves this). `out.flush()` doesn't `fsync`, so in the worst case (OS crash, not just process kill) the very last written line could be lost from the page cache — extremely low-severity for a local research tool with a $50 budget and no concurrent writers; not worth a blocker.
- Malformed/missing clip: raises `MalformedClipError`/`FileNotFoundError` at manifest-load time, before any spend — correct fail-fast posture.
- Tinker SDK absent: raises `TinkerNotInstalledError` with install instructions, not a bare `ImportError` — good.
- Tinker API errors during a live run: explicitly stated to "propagate as-is" — no catch-all swallows them; matches CLAUDE.md's explicit-exceptions rule. Not tested (correctly out of scope), but the code path (Task 12) contains no exception handling around `self._sampling.sample(...)`, so this is true by construction, not just by claim.

### Presumption Inventory

| Assumption | Verdict | Reason |
|---|---|---|
| `model/src/paths.py` is not in the wheel `packages` list, so `manifest.py` must re-implement `ensure_local` | SAFE | Verified directly: `packages` list in `model/pyproject.toml:104` has no `src` entry for the top-level package containing `paths.py` (it's `model/src/paths.py`, a bare module, not a sub-package — confirmed not among the 8 listed packages). |
| `_ensure_clip_local`'s prefix matching is a deliberate extension over `ensure_local`'s exact-match lookup, not an accidental behavior divergence | SAFE | Verified by reading `model/src/paths.py:40` (`manifest.get("entries", {}).get(rel)` — exact key only) against the plan's `rel == registered or rel.startswith(registered + "/")`; design doc explicitly calls this an extension. |
| `pyyaml` is already a dependency, so `manifest.py`/`conftest.py`'s `import yaml` needs no new pyproject entry | SAFE | Verified: `model/pyproject.toml:34` lists `"pyyaml>=6.0"`. |
| Python's `wave.readframes` reliably raises on truncated payloads | RISKY (but harmless) | Behavior varies across implementations/versions; the plan doesn't rely on this alone — the explicit `len(payload) != expected_bytes` check after `readframes` catches truncation regardless of whether `wave` itself raises. Belt-and-suspenders design already covers the risk. |
| `tinker`, `tinker_cookbook`, `tml_renderers` expose the exact API surface used in Task 12 (`model_info.get_recommended_renderer_name`, `chat.AudioPointer`, `chat.Author`/`AuthorKind`, `renderer.parse_response`, `prompt.length`, `tinker.ServiceClient().create_sampling_client`) | VALIDATE | Never exercised by any test in this plan (by design — running the probe is out of scope); the design doc's own Open Questions section already flags SDK package-name/version uncertainty but does not flag the deeper API-shape uncertainty. Low risk to Wave 1 (nothing here blocks the offline harness or the issue migration) but will cost real debugging time whenever the user actually funds and runs Gate 0. Worth a one-line addition to the Gate 0 issue body: "verify `tinker_client.py`'s API calls against current cookbook examples before running — they were written without SDK access." |
| The nine issue numbers (#71 #79 #80 #81 #82 #83 #84 #16 #55) and the three relabel targets (#32 #33 #40) are still open, correctly labeled, and are the complete set of dead-plan issues | VALIDATE | Not independently re-verified in this review (would require live `gh issue list` against the real repo state at execution time); Task 19 Step 1 already captures a BEFORE snapshot for exactly this reason, which is the right verification point — flagging here only so the build agent doesn't skip reading that snapshot's output before closing anything. |
| Group I (GitHub migration) can run fully concurrently with Groups A–H (code) with no shared state | SAFE | Confirmed: Group I touches no files in the worktree at all (pure `gh` CLI calls against the remote issue tracker); no file-level or state-level collision with the code groups. |

### Summary
[BLOCKER] count: 0
[RISK]    count: 3
[QUESTION] count: 1

VERDICT: PROCEED_WITH_CAUTION — (1) Task 6/10/15's implementation-ahead-of-test bundling (schema validation, verdict logic, resume loop shipped before their locking tests in Tasks 8/9/11/16) relies on the plan's own mutation-based "honesty check" steps actually being executed, not just checkbox-ticked — watch that the build agent runs them for real; (2) `probe.py`'s live-mode argument-validation branch (missing token-rate flags) has no test — low severity, CLI-usage-only, and out of scope's "don't run the probe" already limits exposure; (3) `tinker_client.py`'s API surface (renderer/tokenizer/AudioPointer calls) is unverified against the actual Tinker/Inkling SDK and untested by design — acceptable for Wave 1 since running the probe is explicitly out of scope, but flag it in the Gate 0 issue so the user re-verifies the API shape before funding a live run.


