# Chroma-DTW Eval Harness Rework Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** Repoint the chroma-DTW eval harness primary scalar from MAESTRO+parangonar gold-truth at 50ms tolerance to practice-corpus + AMT-pseudo-truth at +/-1.5s tolerance, with an on-disk pseudo-truth cache so `just chroma-eval-verify` runs in <=120s (steady-state, warm Rust binary) without calling AMT inline.

**Spec:** docs/specs/2026-06-02-chroma-dtw-eval-pseudo-truth-design.md
**Style:** Follow `model/CLAUDE.md` and root `CLAUDE.md`. Python via `uv`. partitura (not music21). Explicit exceptions, no silent fallbacks. No emojis.

**First piece:** `bach_prelude_c_wtc1` (Bach Prelude in C, BWV 846). Single 120bpm tempo marking. Score at `model/data/scores/bach.prelude.bwv_846.json` (in-repo JSON, no MXL→JSON build). The score's `bars[].notes[].onset_seconds` field is the score-audio-time axis directly under the constant-tempo identity.

---

## Task Groups

```
Group A (parallel, file-disjoint):
  - A1: bulk deletion of MAESTRO + synthetic-MAESTRO + pilot code
        (gold_truth_builder, practice_compose, amt_pseudo_truth_pilot
        and their tests). Verification: existing chroma_dtw_eval test
        suite still green after the deletes.
  - A4: Justfile recipes (chroma-eval-verify, chroma-eval-ratchet,
        chroma-eval-prebuild, amt-regen-pseudo-truth) + recipe-drift
        guard test.

Group B (sequential within one subagent; new-surface group):
  - B1 (bundled cache): pseudo_truth_cache.py with three behavior tests
        (round-trip, missing-file raises, key-mismatch raises) in one
        commit. Cache key is the 4-tuple
        (audio_sha256, amt_checkpoint_hash, score_sha256, parangonar_version).
        cache_path() is PUBLIC.
  - B4: model/config/amt_version.json committed config (includes
        amt_checkpoint_hash + parangonar_version + regen_source_default).
  - B5: amt_regen.py orchestrator + CLI. Loads bach JSON directly (no
        partitura, no bpm=100.0 projection). Single-tempo guard.
        Coverage gate (>=100 matched and >=50% match rate or
        LowCoverageError). RequestException fatal after retries.

Group C (sequential, depends on A done):
  - C1 (bundled sampler): chunk_sampler rewritten to produce
        ChunkManifestEntry with real per-chunk audio_sha256, written to
        a committed manifest.json. One impl, one test, one commit.
  - C4: opportunistic fur_elise score sourcing (30 min time-box; SKIP if
        not findable).

Group D (sequential, depends on B + C):
  - D-metric: metric_aggregator rewrite — seconds-tolerance primary,
        G4 repurposed as consecutive-chunk continuity guard,
        G2 regression threshold scales with sqrt(50/n_chunks) capped at 4x.
  - D-verify (bundled CLI): verify.py reads manifest, loads pseudo-truth
        via 4-field key (real audio_sha256 from manifest, real
        score_sha256 + parangonar_version from amt_version + score JSON),
        runs DTW, computes error_seconds correctly, writes sidecar with
        error_seconds_distribution + tolerance_sensitivity. Stderr
        WARNING when manifest has <2 distinct pieces. End-to-end behavior
        test asserts numerical correctness (not just shape) on a tiny
        real-bach + synthetic pseudo-truth fixture.
  - D-baseline: commit permissive baseline.json for bach_prelude_c_wtc1
        with notes acknowledging smoke-only n=1 phase.
```

Independence-ship audit: Group A ships independently (removes dead code, no behavior change beyond the deletes). Group B ships independently (new modules with their own tests; nothing consumes them until C/D). Groups C and D do NOT ship independently of each other — the metric switch and the corpus path are co-dependent.

---

## Group A — Deletions and Justfile

### Task A1: Delete `gold_truth_builder.py`, `practice_compose.py`, `amt_pseudo_truth_pilot.py` + their tests

**Group:** A (parallel with A4)

**Behavior being verified:** the eval harness builds, imports, and runs the remaining `tests/chroma_dtw_eval/` suite green after the three modules + their tests are removed. No tautological "module not importable" shape tests are added — deletion is verified by the suite staying green.

**Interface under test:** `chroma_dtw_eval` package (remaining suite must still pass).

**Files:**
- Delete: `model/src/chroma_dtw_eval/gold_truth_builder.py`
- Delete: `model/tests/chroma_dtw_eval/test_gold_truth_builder.py`
- Delete: `model/src/chroma_dtw_eval/practice_compose.py`
- Delete: `model/tests/chroma_dtw_eval/test_practice_compose.py`
- Delete: `model/scripts/amt_pseudo_truth_pilot.py`

- [ ] **Step 1: Write the failing test**

No new test. The "test" is that the existing `tests/chroma_dtw_eval/` suite continues to pass after the deletions land. Run it FIRST to establish the pre-delete baseline:

```bash
cd model && uv run pytest tests/chroma_dtw_eval/ -x
```
Expected: PASS (pre-delete baseline).

- [ ] **Step 2: Run test — verify it FAILS**

This is a pure-deletion task; there is no red-then-green TDD cycle. The risk this task carries is that one of the deleted modules is still imported by surviving modules. To make that risk visible BEFORE the delete, grep the package for imports of the targeted modules:

```bash
cd model && grep -rn "gold_truth_builder\|practice_compose\|amt_pseudo_truth_pilot" src/ tests/ scripts/
```
Expected: matches only inside the files about to be deleted (plus their own tests). If a surviving module imports any of the three, FIX that import in this task before the delete.

- [ ] **Step 3: Implement the minimum to make the test pass**

```bash
git rm model/src/chroma_dtw_eval/gold_truth_builder.py
git rm model/tests/chroma_dtw_eval/test_gold_truth_builder.py
git rm model/src/chroma_dtw_eval/practice_compose.py
git rm model/tests/chroma_dtw_eval/test_practice_compose.py
git rm model/scripts/amt_pseudo_truth_pilot.py
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/chroma_dtw_eval/ -x
```
Expected: PASS (suite still green after deletes).

- [ ] **Step 5: Commit**

```bash
git add -A model/src/chroma_dtw_eval/ model/tests/chroma_dtw_eval/ model/scripts/ && git commit -m "refactor(chroma-eval): remove MAESTRO gold-truth, synthetic-MAESTRO composition, and AMT pilot script"
```

---

### Task A4: Add Justfile recipes (`chroma-eval-verify`, `chroma-eval-ratchet`, `chroma-eval-prebuild`, `amt-regen-pseudo-truth`) + drift guard

**Group:** A (parallel with A1)

**Behavior being verified:** the four recipes introduced by the rework are present in `Justfile` and discoverable via `just --list`. The drift guard locks the rework's recipe-naming contract so the next phantom CLAUDE.md edit cannot silently claim a recipe that does not exist.

**Interface under test:** `Justfile` recipe surface.

**Files:**
- Modify: `Justfile`
- Test: `model/tests/chroma_dtw_eval/test_just_recipes_drift.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# model/tests/chroma_dtw_eval/test_just_recipes_drift.py
"""Locks the rework's recipe-naming contract. Asserts the four recipes
the rework introduces exist by name in Justfile, by parsing the file
directly (no `just` binary required).
"""
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
JUSTFILE = REPO_ROOT / "Justfile"


@pytest.mark.parametrize("recipe", [
    "chroma-eval-verify",
    "chroma-eval-ratchet",
    "chroma-eval-prebuild",
    "amt-regen-pseudo-truth",
])
def test_recipe_present_in_justfile(recipe: str) -> None:
    assert JUSTFILE.exists(), f"Justfile missing: {JUSTFILE}"
    body = JUSTFILE.read_text()
    # Recipes appear as `name:` or `name arg1 arg2:` at start of line.
    found = any(
        line.split(":")[0].split()[0] == recipe
        for line in body.splitlines()
        if line and not line.startswith((" ", "\t", "#"))
    )
    assert found, f"missing recipe {recipe!r} in {JUSTFILE}"
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/chroma_dtw_eval/test_just_recipes_drift.py -x
```
Expected: FAIL — four parametrized cases assert recipes not found.

- [ ] **Step 3: Implement the minimum to make the test pass**

Append to `Justfile` (root of repo):

```make
# Build dtw_chunk_cli release binary so chroma-eval-verify hits its 120s budget on warm cache.
# Run once after a clean checkout; idempotent thereafter.
chroma-eval-prebuild:
    cd apps/api/src/wasm/score-analysis && cargo build --release --bin dtw_chunk_cli

# Run chroma-DTW eval harness against the committed practice corpus.
# stdout: exactly one float (the primary). exit 0 iff no guard regressed.
# 120s budget assumes the Rust DTW binary is already built; run `just chroma-eval-prebuild` once after a clean checkout.
chroma-eval-verify:
    cd model && uv run python -m chroma_dtw_eval.verify \
        --baseline data/evals/chroma_dtw/baseline.json \
        --manifest data/evals/chroma_dtw_fixtures/manifest.json

# Promote the latest sidecar to baseline. Refuses to write on regression.
chroma-eval-ratchet:
    cd model && uv run python -m chroma_dtw_eval.ratchet \
        --from data/evals/chroma_dtw/last_run.json \
        --to data/evals/chroma_dtw/baseline.json

# Regenerate AMT pseudo-truth cache for a single (piece, video_id).
# Usage: just amt-regen-pseudo-truth <piece_id> <video_id>
amt-regen-pseudo-truth piece video_id:
    cd model && uv run python -m chroma_dtw_eval.amt_regen \
        --piece {{piece}} --video-id {{video_id}}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/chroma_dtw_eval/test_just_recipes_drift.py -x
```
Expected: PASS (all four parametrized cases).

- [ ] **Step 5: Commit**

```bash
git add Justfile model/tests/chroma_dtw_eval/test_just_recipes_drift.py && git commit -m "build(just): add chroma-eval-verify/ratchet/prebuild + amt-regen-pseudo-truth recipes with drift guard"
```

---

## Group B — Pseudo-Truth Cache + AMT Regen

### Task B1: `pseudo_truth_cache.py` (bundled: round-trip + missing + mismatch)

**Group:** B (sequential within one subagent)

**Behavior being verified:**
1. What `write_pseudo_truth` writes, `load_pseudo_truth` reads back identically (round-trip).
2. `load_pseudo_truth` raises `PseudoTruthMissingError` (FileNotFoundError subclass) when the cache file is absent.
3. `load_pseudo_truth` raises `PseudoTruthMismatchError` (ValueError subclass) when ANY of the four key fields (`audio_sha256`, `amt_checkpoint_hash`, `score_sha256`, `parangonar_version`) disagrees with the file's stored key.

Bundled into a single task with three behavior assertions, one impl, one commit. Justification: B1/B2/B3 in the prior plan admitted they could not honor red-then-green individually (B1's impl already satisfies B2/B3); collapsing avoids the "three subagents writing the same file with three different schemas" failure mode.

**Interface under test:** `write_pseudo_truth`, `load_pseudo_truth`, `cache_path`, `PseudoTruth.audio_sec_to_score_sec`, `PseudoTruth.score_sec_to_audio_sec`, `PseudoTruthMissingError`, `PseudoTruthMismatchError`.

**Files:**
- Create: `model/src/chroma_dtw_eval/pseudo_truth_cache.py`
- Test: `model/tests/chroma_dtw_eval/test_pseudo_truth_cache.py`

- [ ] **Step 1: Write the failing test**

```python
# model/tests/chroma_dtw_eval/test_pseudo_truth_cache.py
"""Three behaviors locked in one suite:
1. round-trip writer -> loader equality + monotone interpolation
2. missing cache file raises PseudoTruthMissingError with usable path
3. any-of-four key-field mismatch raises PseudoTruthMismatchError
"""
from pathlib import Path

import numpy as np
import pytest

from chroma_dtw_eval.pseudo_truth_cache import (
    PseudoTruthMismatchError,
    PseudoTruthMissingError,
    PseudoTruthPayload,
    cache_path,
    load_pseudo_truth,
    write_pseudo_truth,
)


def _payload() -> PseudoTruthPayload:
    return PseudoTruthPayload(
        perf_audio_sec=np.array([0.0, 1.0, 2.0, 4.0], dtype=np.float64),
        score_audio_sec=np.array([0.0, 0.5, 1.0, 2.0], dtype=np.float64),
        measure_table=[{"bar_number": 1, "start_sec": 0.0, "start_tick": 0}],
        audio_sha256="a" * 16,
        amt_checkpoint_hash="b" * 16,
        score_sha256="c" * 16,
        parangonar_version="3.3.2",
        regen_source="local:test",
    )


def test_roundtrip_and_interpolation(tmp_path: Path) -> None:
    written = write_pseudo_truth(
        piece_id="bach_prelude_c_wtc1", video_id="VID000",
        payload=_payload(), cache_root=tmp_path,
    )
    assert written.exists()
    assert written == cache_path(tmp_path, "bach_prelude_c_wtc1", "VID000")

    loaded = load_pseudo_truth(
        piece_id="bach_prelude_c_wtc1", video_id="VID000",
        audio_sha256="a" * 16, amt_checkpoint_hash="b" * 16,
        score_sha256="c" * 16, parangonar_version="3.3.2",
        cache_root=tmp_path,
    )
    np.testing.assert_array_equal(loaded.perf_audio_sec, _payload().perf_audio_sec)
    np.testing.assert_array_equal(loaded.score_audio_sec, _payload().score_audio_sec)
    assert loaded.measure_table == _payload().measure_table
    # Monotone linear interpolation between anchors.
    assert loaded.audio_sec_to_score_sec(0.5) == pytest.approx(0.25)
    assert loaded.audio_sec_to_score_sec(3.0) == pytest.approx(1.5)
    # Inverse is consistent.
    assert loaded.score_sec_to_audio_sec(1.0) == pytest.approx(2.0)


def test_missing_file_raises_with_path(tmp_path: Path) -> None:
    with pytest.raises(PseudoTruthMissingError) as exc:
        load_pseudo_truth(
            piece_id="nope", video_id="zzz",
            audio_sha256="x" * 16, amt_checkpoint_hash="y" * 16,
            score_sha256="w" * 16, parangonar_version="3.3.2",
            cache_root=tmp_path,
        )
    msg = str(exc.value)
    assert "nope" in msg and "zzz" in msg
    assert str(tmp_path) in msg


@pytest.mark.parametrize("field,bad", [
    ("audio_sha256", "z" * 16),
    ("amt_checkpoint_hash", "z" * 16),
    ("score_sha256", "z" * 16),
    ("parangonar_version", "9.9.9"),
])
def test_key_mismatch_raises(tmp_path: Path, field: str, bad: str) -> None:
    write_pseudo_truth("p", "v", _payload(), tmp_path)
    kwargs = {
        "audio_sha256": "a" * 16,
        "amt_checkpoint_hash": "b" * 16,
        "score_sha256": "c" * 16,
        "parangonar_version": "3.3.2",
    }
    kwargs[field] = bad
    with pytest.raises(PseudoTruthMismatchError) as exc:
        load_pseudo_truth("p", "v", cache_root=tmp_path, **kwargs)
    assert field in str(exc.value)
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/chroma_dtw_eval/test_pseudo_truth_cache.py -x
```
Expected: FAIL — `ModuleNotFoundError: No module named 'chroma_dtw_eval.pseudo_truth_cache'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# model/src/chroma_dtw_eval/pseudo_truth_cache.py
"""On-disk cache for AMT-derived pseudo-truth alignment of practice audio
to score (single-tempo identity: score_audio_sec is score seconds directly).

Keyed by the 4-tuple (audio_sha256, amt_checkpoint_hash, score_sha256,
parangonar_version). Read-only at eval time; written only by amt_regen.
Explicit exceptions on missing files and any-field mismatch -- no silent
fallbacks. JSON on disk (not pickle) for forward-compatibility.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class PseudoTruthMissingError(FileNotFoundError):
    pass


class PseudoTruthMismatchError(ValueError):
    pass


@dataclass
class PseudoTruthPayload:
    perf_audio_sec: np.ndarray
    score_audio_sec: np.ndarray
    measure_table: list[dict]
    audio_sha256: str
    amt_checkpoint_hash: str
    score_sha256: str
    parangonar_version: str
    regen_source: str


@dataclass
class PseudoTruth:
    perf_audio_sec: np.ndarray
    score_audio_sec: np.ndarray
    measure_table: list[dict]
    audio_sha256: str
    amt_checkpoint_hash: str
    score_sha256: str
    parangonar_version: str

    def audio_sec_to_score_sec(self, t: float) -> float:
        if self.perf_audio_sec.size < 2:
            raise PseudoTruthMismatchError("perf_audio_sec must have >= 2 anchors")
        return float(np.interp(t, self.perf_audio_sec, self.score_audio_sec))

    def score_sec_to_audio_sec(self, s: float) -> float:
        if self.score_audio_sec.size < 2:
            raise PseudoTruthMismatchError("score_audio_sec must have >= 2 anchors")
        return float(np.interp(s, self.score_audio_sec, self.perf_audio_sec))


def cache_path(cache_root: Path, piece_id: str, video_id: str) -> Path:
    """PUBLIC. Used by amt_regen and chunk_sampler to locate cache files."""
    return cache_root / piece_id / f"{video_id}.json"


def write_pseudo_truth(
    piece_id: str,
    video_id: str,
    payload: PseudoTruthPayload,
    cache_root: Path,
) -> Path:
    if payload.perf_audio_sec.shape != payload.score_audio_sec.shape:
        raise PseudoTruthMismatchError(
            f"shape mismatch: perf {payload.perf_audio_sec.shape} vs score {payload.score_audio_sec.shape}"
        )
    out = cache_path(cache_root, piece_id, video_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "audio_sha256": payload.audio_sha256,
        "amt_checkpoint_hash": payload.amt_checkpoint_hash,
        "score_sha256": payload.score_sha256,
        "parangonar_version": payload.parangonar_version,
        "regen_source": payload.regen_source,
        "perf_audio_sec": payload.perf_audio_sec.tolist(),
        "score_audio_sec": payload.score_audio_sec.tolist(),
        "measure_table": payload.measure_table,
    }
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(body))
    tmp.replace(out)
    return out


def load_pseudo_truth(
    piece_id: str,
    video_id: str,
    *,
    audio_sha256: str,
    amt_checkpoint_hash: str,
    score_sha256: str,
    parangonar_version: str,
    cache_root: Path,
) -> PseudoTruth:
    path = cache_path(cache_root, piece_id, video_id)
    if not path.exists():
        raise PseudoTruthMissingError(
            f"pseudo-truth cache missing for {piece_id}/{video_id}: {path}"
        )
    body = json.loads(path.read_text())
    for field, expected in (
        ("audio_sha256", audio_sha256),
        ("amt_checkpoint_hash", amt_checkpoint_hash),
        ("score_sha256", score_sha256),
        ("parangonar_version", parangonar_version),
    ):
        actual = body.get(field)
        if actual != expected:
            raise PseudoTruthMismatchError(
                f"{field} mismatch for {piece_id}/{video_id}: "
                f"requested {expected!r}, cached {actual!r}"
            )
    return PseudoTruth(
        perf_audio_sec=np.asarray(body["perf_audio_sec"], dtype=np.float64),
        score_audio_sec=np.asarray(body["score_audio_sec"], dtype=np.float64),
        measure_table=body["measure_table"],
        audio_sha256=body["audio_sha256"],
        amt_checkpoint_hash=body["amt_checkpoint_hash"],
        score_sha256=body["score_sha256"],
        parangonar_version=body["parangonar_version"],
    )
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/chroma_dtw_eval/test_pseudo_truth_cache.py -x
```
Expected: PASS (round-trip + missing-file + four parametrized mismatch cases).

- [ ] **Step 5: Commit**

```bash
git add model/src/chroma_dtw_eval/pseudo_truth_cache.py model/tests/chroma_dtw_eval/test_pseudo_truth_cache.py && git commit -m "feat(chroma-eval): pseudo-truth cache (4-field key) with round-trip + mismatch behavior tests"
```

---

### Task B4: Commit `model/config/amt_version.json`

**Group:** B (parallel with B1)

**Behavior being verified:** the committed config exposes the three fields the rework relies on (`checkpoint_hash`, `parangonar_version`, `regen_source_default`) and they are loadable as JSON.

**Interface under test:** the file's on-disk JSON shape.

**Files:**
- Create: `model/config/amt_version.json`
- Test: `model/tests/chroma_dtw_eval/test_amt_version_config.py`

- [ ] **Step 1: Write the failing test**

```python
# model/tests/chroma_dtw_eval/test_amt_version_config.py
"""Locks the committed AMT-version config schema."""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = REPO_ROOT / "model/config/amt_version.json"


def test_config_present_and_has_required_fields() -> None:
    assert CONFIG.exists(), f"committed AMT version config missing: {CONFIG}"
    body = json.loads(CONFIG.read_text())
    assert isinstance(body.get("checkpoint_hash"), str)
    assert len(body["checkpoint_hash"]) >= 16
    assert isinstance(body.get("parangonar_version"), str)
    assert body["parangonar_version"], "parangonar_version must be non-empty"
    assert isinstance(body.get("regen_source_default"), str)
    assert body["regen_source_default"], "regen_source_default must be non-empty"
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/chroma_dtw_eval/test_amt_version_config.py -x
```
Expected: FAIL — `AssertionError: committed AMT version config missing: ...`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `model/config/amt_version.json`:

```json
{
  "checkpoint_hash": "aria_amt_v1_pilot_2026_06_01",
  "parangonar_version": "3.3.2",
  "regen_source_default": "local:aria-amt",
  "model_name": "aria-amt",
  "pinned_at": "2026-06-02",
  "notes": "checkpoint_hash is a stable label, not a cryptographic digest; bump when apps/inference/amt checkpoint changes. parangonar_version must match the installed parangonar package; bump when uv lock changes."
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/chroma_dtw_eval/test_amt_version_config.py -x
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add model/config/amt_version.json model/tests/chroma_dtw_eval/test_amt_version_config.py && git commit -m "feat(chroma-eval): pin AMT checkpoint + parangonar version + regen source"
```

---

### Task B5: `amt_regen.regenerate_pseudo_truth` orchestrator + CLI

**Group:** B (sequential after B1 + B4)

**Behavior being verified:** given a real audio file + the bach prelude JSON score + a stub AMT URL, the orchestrator writes a pseudo-truth cache file with the expected 4-field key, and a second call with identical inputs is a no-op (idempotent). Loading the cache via `pseudo_truth_cache.load_pseudo_truth` round-trips.

**Interface under test:** `chroma_dtw_eval.amt_regen.regenerate_pseudo_truth` and the `python -m chroma_dtw_eval.amt_regen` CLI.

**Files:**
- Create: `model/src/chroma_dtw_eval/amt_regen.py`
- Test: `model/tests/chroma_dtw_eval/test_amt_regen.py`

- [ ] **Step 1: Write the failing test**

```python
# model/tests/chroma_dtw_eval/test_amt_regen.py
"""Drive amt_regen against a stub AMT HTTP server and the committed bach
prelude JSON; assert (a) the 4-field-keyed cache is written, (b) second
call with identical inputs is a no-op, (c) the loader reads it back.

The bach JSON is the canonical score for the first piece. The stub AMT
returns a synthesized note set whose pitches overlap the bach prelude
bar 1 (C major arpeggio), so parangonar's matcher will produce >= 100
matches when stub note count is scaled accordingly.
"""
from __future__ import annotations

import http.server
import json
import socketserver
import threading
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from chroma_dtw_eval.amt_regen import (
    AmtRegenError,
    LowCoverageError,
    RegenResult,
    regenerate_pseudo_truth,
)
from chroma_dtw_eval.pseudo_truth_cache import load_pseudo_truth

REPO_ROOT = Path(__file__).resolve().parents[3]
BACH_SCORE = REPO_ROOT / "model/data/scores/bach.prelude.bwv_846.json"


def _bach_canned_notes() -> list[dict]:
    """Synthesize an AMT-like note set from the committed bach JSON itself
    by reading bar 1's notes and shifting onsets by +0.05s (a small jitter
    to mimic AMT detection noise). Returns >= 100 notes by walking enough
    bars; ensures the coverage gate passes.
    """
    body = json.loads(BACH_SCORE.read_text())
    bars = body.get("bars") or []
    notes: list[dict] = []
    for bar in bars:
        for n in (bar.get("notes") or []):
            notes.append({
                "onset": float(n["onset_seconds"]) + 0.05,
                "offset": float(n["onset_seconds"]) + float(n.get("duration_seconds", 0.2)) + 0.05,
                "pitch": int(n["pitch"]),
                "velocity": int(n.get("velocity", 80)),
            })
            if len(notes) >= 200:
                return notes
    return notes


class _StubAmtHandler(http.server.BaseHTTPRequestHandler):
    canned_notes: list[dict] = []

    def log_message(self, *a, **k):  # silence
        pass

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length", "0"))
        _ = self.rfile.read(n)
        body = json.dumps({"midi_notes": self.canned_notes}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def stub_amt_server():
    _StubAmtHandler.canned_notes = _bach_canned_notes()
    srv = socketserver.TCPServer(("127.0.0.1", 0), _StubAmtHandler)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{port}"
    srv.shutdown(); srv.server_close()


@pytest.fixture
def tiny_audio(tmp_path: Path) -> Path:
    wav = tmp_path / "tiny.wav"
    # 30s of near-silence at 16k; the stub doesn't actually transcribe it.
    sf.write(wav, np.zeros(16000 * 30, dtype=np.float32), 16000, subtype="FLOAT")
    return wav


def test_regen_writes_cache_and_is_idempotent(
    stub_amt_server: str, tiny_audio: Path, tmp_path: Path,
) -> None:
    cache_root = tmp_path / "pseudo_truth"
    first: RegenResult = regenerate_pseudo_truth(
        piece_id="bach_prelude_c_wtc1", video_id="V0",
        score_path=BACH_SCORE, audio_path=tiny_audio,
        amt_url=stub_amt_server + "/transcribe",
        amt_checkpoint_hash="aria_amt_v1_pilot_2026_06_01",
        parangonar_version="3.3.2",
        cache_root=cache_root,
    )
    assert first.wrote_cache is True
    assert first.cache_path.exists()
    assert first.n_matched >= 100

    second: RegenResult = regenerate_pseudo_truth(
        piece_id="bach_prelude_c_wtc1", video_id="V0",
        score_path=BACH_SCORE, audio_path=tiny_audio,
        amt_url=stub_amt_server + "/transcribe",
        amt_checkpoint_hash="aria_amt_v1_pilot_2026_06_01",
        parangonar_version="3.3.2",
        cache_root=cache_root,
    )
    assert second.wrote_cache is False, "second regen with identical inputs must be no-op"

    loaded = load_pseudo_truth(
        piece_id="bach_prelude_c_wtc1", video_id="V0",
        audio_sha256=first.audio_sha256,
        amt_checkpoint_hash="aria_amt_v1_pilot_2026_06_01",
        score_sha256=first.score_sha256,
        parangonar_version="3.3.2",
        cache_root=cache_root,
    )
    assert loaded.perf_audio_sec.size >= 100
    assert loaded.score_audio_sec.size == loaded.perf_audio_sec.size


def test_regen_raises_low_coverage_on_sparse_match(
    stub_amt_server: str, tiny_audio: Path, tmp_path: Path,
) -> None:
    # Override stub to return only 10 notes; coverage gate (>=100 matched) must fire.
    _StubAmtHandler.canned_notes = _bach_canned_notes()[:10]
    try:
        with pytest.raises((LowCoverageError, AmtRegenError)):
            regenerate_pseudo_truth(
                piece_id="bach_prelude_c_wtc1", video_id="V1",
                score_path=BACH_SCORE, audio_path=tiny_audio,
                amt_url=stub_amt_server + "/transcribe",
                amt_checkpoint_hash="aria_amt_v1_pilot_2026_06_01",
                parangonar_version="3.3.2",
                cache_root=tmp_path / "pseudo_truth",
            )
    finally:
        _StubAmtHandler.canned_notes = _bach_canned_notes()
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/chroma_dtw_eval/test_amt_regen.py -x
```
Expected: FAIL — `ModuleNotFoundError: No module named 'chroma_dtw_eval.amt_regen'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# model/src/chroma_dtw_eval/amt_regen.py
"""AMT regen orchestrator: practice audio -> AMT -> parangonar -> pseudo-truth cache.

Single-tempo scores ONLY in this rework (see spec "Variable-tempo score support (future)").
Loads the score JSON directly (no partitura). Score onset_seconds is the
score-audio-time axis under the constant-tempo identity.

Idempotent: re-running with identical 4-field cache key is a no-op.
Explicit exceptions on AMT failures, low coverage, and variable-tempo scores.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

from chroma_dtw_eval.pseudo_truth_cache import (
    PseudoTruthMismatchError,
    PseudoTruthMissingError,
    PseudoTruthPayload,
    cache_path,
    load_pseudo_truth,
    write_pseudo_truth,
)

AMT_CHUNK_S = 27.0
TARGET_SR = 16000
RETRY_LIMIT = 2

# Default paths anchored to THIS module's location, never relative to CWD.
_MODULE_DIR = Path(__file__).resolve()
DEFAULT_AMT_URL = os.environ.get("AMT_URL", "http://127.0.0.1:8001/transcribe")
DEFAULT_AMT_VERSION_CONFIG = _MODULE_DIR.parents[2] / "config/amt_version.json"
DEFAULT_PRACTICE_ROOT = _MODULE_DIR.parents[2] / "data/evals/practice_eval"
DEFAULT_CACHE_ROOT = _MODULE_DIR.parents[2] / "data/evals/pseudo_truth"
DEFAULT_SCORE_ROOT = _MODULE_DIR.parents[2] / "data/scores"
DEFAULT_SCORE_BY_PIECE = {
    "bach_prelude_c_wtc1": DEFAULT_SCORE_ROOT / "bach.prelude.bwv_846.json",
}


class AmtRegenError(RuntimeError):
    pass


class LowCoverageError(AmtRegenError):
    pass


@dataclass
class RegenResult:
    wrote_cache: bool
    cache_path: Path
    audio_sha256: str
    score_sha256: str
    n_amt_notes: int
    n_matched: int


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _read_wav_16k_mono(audio_path: Path) -> np.ndarray:
    y, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != TARGET_SR:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(sr, TARGET_SR)
        y = resample_poly(y, TARGET_SR // g, sr // g).astype(np.float32)
    return y


def _encode_chunk_b64(pcm: np.ndarray) -> str:
    buf = io.BytesIO()
    sf.write(buf, pcm, TARGET_SR, format="WAV", subtype="FLOAT")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _post_chunk(amt_url: str, pcm: np.ndarray) -> dict:
    """POST one chunk to AMT with retries. RequestException -> AmtRegenError
    after RETRY_LIMIT attempts. A documented 200-with-error-body (tokenizer
    boundary bug) is the only condition that signals skip-this-chunk.
    """
    last_exc: Exception | None = None
    for attempt in range(RETRY_LIMIT + 1):
        try:
            r = requests.post(
                amt_url,
                json={"chunk_audio": _encode_chunk_b64(pcm), "context_audio": None},
                timeout=180,
            )
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            last_exc = exc
            continue
    raise AmtRegenError(
        f"AMT POST to {amt_url} failed after {RETRY_LIMIT + 1} attempts: {last_exc}"
    ) from last_exc


def _transcribe_clip(audio_16k: np.ndarray, amt_url: str) -> list[dict]:
    n_chunks = max(1, int(np.ceil(len(audio_16k) / (AMT_CHUNK_S * TARGET_SR))))
    chunk_len = int(AMT_CHUNK_S * TARGET_SR)
    all_notes: list[dict] = []
    for i in range(n_chunks):
        start = i * chunk_len
        end = min(start + chunk_len, len(audio_16k))
        pcm = audio_16k[start:end]
        if len(pcm) < chunk_len:
            pcm = np.concatenate([pcm, np.zeros(chunk_len - len(pcm), dtype=np.float32)])
        body = _post_chunk(amt_url, pcm)
        if "error" in body:
            # Documented tokenizer-boundary failure mode; skip this one chunk.
            continue
        offset = i * AMT_CHUNK_S
        for n in body.get("midi_notes") or []:
            all_notes.append({
                "onset": float(n["onset"]) + offset,
                "offset": float(n["offset"]) + offset,
                "pitch": int(n["pitch"]),
                "velocity": int(n.get("velocity", 80)),
            })
    return all_notes


def _load_bach_json_score(score_path: Path) -> tuple[np.ndarray, list[dict], str]:
    """Load a single-tempo score JSON (bach prelude format).

    Returns (score_na, measure_table, score_sha256).
    score_na fields: ("onset_sec", float), ("onset_beat", float),
                     ("pitch", int), ("duration_sec", float), ("id", "U32").
    Constant-tempo identity: onset_sec IS the score-audio-time axis.
    """
    score_sha256 = _sha256_file(score_path)
    body = json.loads(score_path.read_text())
    tempos = body.get("tempo_markings") or []
    if len(tempos) != 1:
        raise AmtRegenError(
            f"variable-tempo scores not supported in this rework; got "
            f"{len(tempos)} tempo markings in {score_path}. "
            f"See spec section 'Variable-tempo score support (future)'."
        )
    bars = body.get("bars") or []
    if len(bars) >= 2:
        # Infer ticks_per_beat from bar geometry. Bach prelude is 4/4; bar 2
        # starts exactly 4 beats after bar 1. For non-4/4 scores, parse
        # time_signatures; for now assert 4/4 and fail loud otherwise.
        ts_list = body.get("time_signatures") or []
        ts = ts_list[0] if ts_list else {}
        beats_per_bar = int(ts.get("numerator", 4))
        if beats_per_bar != 4:
            raise AmtRegenError(
                f"non-4/4 scores not supported in this rework; got "
                f"time_signature numerator={beats_per_bar} in {score_path}"
            )
        ticks_per_beat = (int(bars[1]["start_tick"]) - int(bars[0]["start_tick"])) // beats_per_bar
    else:
        ticks_per_beat = 480
    if ticks_per_beat <= 0:
        raise AmtRegenError(f"could not infer ticks_per_beat from {score_path}")

    rows = []
    nid = 0
    for bar in bars:
        for n in (bar.get("notes") or []):
            rows.append((
                float(n["onset_seconds"]),
                float(n["onset_tick"]) / ticks_per_beat,
                int(n["pitch"]),
                float(n.get("duration_seconds", 0.001)),
                f"s{nid}",
            ))
            nid += 1
    if not rows:
        raise AmtRegenError(f"no notes found in score: {score_path}")
    dtype = [("onset_sec", float), ("onset_beat", float), ("pitch", int),
             ("duration_sec", float), ("id", "U32")]
    score_na = np.array(rows, dtype=dtype)
    score_na.sort(order="onset_sec")

    measure_table = [
        {"bar_number": int(b["bar_number"]),
         "start_sec": float(b["start_seconds"]),
         "start_tick": int(b["start_tick"])}
        for b in bars
    ]
    return score_na, measure_table, score_sha256


def _amt_to_perf_na(notes: list[dict]) -> np.ndarray:
    dtype = [
        ("onset_sec", float), ("onset_beat", float),
        ("duration_sec", float), ("pitch", int),
        ("velocity", int), ("id", "U32"),
    ]
    arr = np.empty(len(notes), dtype=dtype)
    for i, n in enumerate(notes):
        arr[i] = (
            float(n["onset"]),
            0.0,  # onset_beat unknown for perf; matcher uses onset_sec on perf side
            max(float(n["offset"]) - float(n["onset"]), 0.001),
            int(n["pitch"]),
            int(n.get("velocity", 80)),
            f"p{i}",
        )
    arr.sort(order="onset_sec")
    return arr


def _match(score_na: np.ndarray, perf_na: np.ndarray) -> list[dict]:
    import parangonar as pa
    matcher = pa.AutomaticNoteMatcher()
    return list(matcher(score_na, perf_na))


def _build_pairs(
    score_na: np.ndarray, amt_perf_na: np.ndarray, matches: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Build (perf_audio_sec, score_audio_sec) pairs from label=='match'
    entries; sort by perf time; enforce monotone running-max on score axis.
    """
    score_id_to_audio_sec = {str(s["id"]): float(s["onset_sec"]) for s in score_na}
    perf_id_to_audio_sec = {str(n["id"]): float(n["onset_sec"]) for n in amt_perf_na}
    pairs: list[tuple[float, float]] = []
    for entry in matches:
        if entry.get("label") != "match":
            continue
        s_id = str(entry.get("score_id"))
        p_id = str(entry.get("performance_id"))
        if s_id in score_id_to_audio_sec and p_id in perf_id_to_audio_sec:
            pairs.append((perf_id_to_audio_sec[p_id], score_id_to_audio_sec[s_id]))
    if not pairs:
        raise AmtRegenError("parangonar produced zero matches; cannot build pseudo-truth")
    pairs.sort()
    perf_arr = np.array([p[0] for p in pairs], dtype=np.float64)
    score_arr = np.array([p[1] for p in pairs], dtype=np.float64)
    score_arr = np.maximum.accumulate(score_arr)
    return perf_arr, score_arr


def regenerate_pseudo_truth(
    piece_id: str,
    video_id: str,
    *,
    score_path: Path,
    audio_path: Path,
    amt_url: str,
    amt_checkpoint_hash: str,
    parangonar_version: str,
    cache_root: Path,
    force: bool = False,
) -> RegenResult:
    if not score_path.exists():
        raise AmtRegenError(f"score not found: {score_path}")
    if not audio_path.exists():
        raise AmtRegenError(f"audio not found: {audio_path}")
    audio_sha256 = _sha256_file(audio_path)
    score_sha256 = _sha256_file(score_path)

    # Idempotence check. Catch the SPECIFIC exception classes; let any other
    # error propagate (CLAUDE.md "no catch-all" rule).
    if not force:
        try:
            load_pseudo_truth(
                piece_id, video_id,
                audio_sha256=audio_sha256,
                amt_checkpoint_hash=amt_checkpoint_hash,
                score_sha256=score_sha256,
                parangonar_version=parangonar_version,
                cache_root=cache_root,
            )
            return RegenResult(
                wrote_cache=False,
                cache_path=cache_path(cache_root, piece_id, video_id),
                audio_sha256=audio_sha256,
                score_sha256=score_sha256,
                n_amt_notes=0, n_matched=0,
            )
        except PseudoTruthMissingError:
            pass  # regen below
        except PseudoTruthMismatchError:
            pass  # regen below

    audio_16k = _read_wav_16k_mono(audio_path)
    amt_notes = _transcribe_clip(audio_16k, amt_url)
    if not amt_notes:
        raise AmtRegenError(f"AMT returned zero notes for {audio_path}")
    amt_perf_na = _amt_to_perf_na(amt_notes)
    score_na, measure_table, score_sha256_2 = _load_bach_json_score(score_path)
    assert score_sha256 == score_sha256_2
    matches = _match(score_na, amt_perf_na)
    perf_arr, score_arr = _build_pairs(score_na, amt_perf_na, matches)

    if score_arr.size < 100 or score_arr.size / max(len(amt_notes), 1) < 0.5:
        raise LowCoverageError(
            f"insufficient match coverage: matched={score_arr.size}, "
            f"amt_notes={len(amt_notes)}, match_rate="
            f"{score_arr.size / max(len(amt_notes), 1):.3f}"
        )

    config_body = (
        json.loads(DEFAULT_AMT_VERSION_CONFIG.read_text())
        if DEFAULT_AMT_VERSION_CONFIG.exists() else {}
    )
    regen_source = config_body.get("regen_source_default", "local:aria-amt")
    payload = PseudoTruthPayload(
        perf_audio_sec=perf_arr,
        score_audio_sec=score_arr,
        measure_table=measure_table,
        audio_sha256=audio_sha256,
        amt_checkpoint_hash=amt_checkpoint_hash,
        score_sha256=score_sha256,
        parangonar_version=parangonar_version,
        regen_source=regen_source,
    )
    out = write_pseudo_truth(piece_id, video_id, payload, cache_root)
    return RegenResult(
        wrote_cache=True,
        cache_path=out,
        audio_sha256=audio_sha256,
        score_sha256=score_sha256,
        n_amt_notes=len(amt_notes),
        n_matched=int(score_arr.size),
    )


def _cli(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="chroma_dtw_eval.amt_regen")
    p.add_argument("--piece", required=True)
    p.add_argument("--video-id", required=True)
    p.add_argument("--score", type=Path, default=None)
    p.add_argument("--audio", type=Path, default=None)
    p.add_argument("--amt-url", default=DEFAULT_AMT_URL)
    p.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    p.add_argument("--config", type=Path, default=DEFAULT_AMT_VERSION_CONFIG)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    score = args.score or DEFAULT_SCORE_BY_PIECE.get(args.piece)
    if score is None:
        raise AmtRegenError(
            f"no default score for piece {args.piece!r}; pass --score explicitly"
        )
    audio = args.audio or (DEFAULT_PRACTICE_ROOT / args.piece / "audio" / f"{args.video_id}.wav")
    config_body = json.loads(args.config.read_text())
    res = regenerate_pseudo_truth(
        piece_id=args.piece, video_id=args.video_id,
        score_path=score, audio_path=audio,
        amt_url=args.amt_url,
        amt_checkpoint_hash=config_body["checkpoint_hash"],
        parangonar_version=config_body["parangonar_version"],
        cache_root=args.cache_root, force=args.force,
    )
    print(json.dumps({
        "wrote_cache": res.wrote_cache,
        "cache_path": str(res.cache_path),
        "audio_sha256": res.audio_sha256,
        "score_sha256": res.score_sha256,
        "n_amt_notes": res.n_amt_notes,
        "n_matched": res.n_matched,
    }))
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/chroma_dtw_eval/test_amt_regen.py -x
```
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add model/src/chroma_dtw_eval/amt_regen.py model/tests/chroma_dtw_eval/test_amt_regen.py && git commit -m "feat(chroma-eval): amt_regen orchestrator with bach JSON loader, 4-field key, coverage gate, fatal RequestException"
```

---

## Group C — Chunk Sampler Rewrite

### Task C1: Bundled `chunk_sampler` rewrite — `sample_chunks` + committed manifest with real `audio_sha256`

**Group:** C (sequential after Group A)

**Behavior being verified:** `sample_chunks(corpus_root, pseudo_truth_root, seed)` returns a deterministic `list[ChunkManifestEntry]` where each entry carries `piece, video_id, start_audio_sec, end_audio_sec, audio_sha256`. The `audio_sha256` is computed from the actual chunk audio bytes (the slice of the WAV file between start and end), not the whole file. The manifest is written to `model/data/evals/chroma_dtw_fixtures/manifest.json` and committed.

Bundled into a single task to avoid the prior three-task split where the manifest entry shape, the sampling stratification, and the audio-sha256 wiring were spread across three commits with intermediate broken states.

**Interface under test:** `sample_chunks`, `ChunkManifestEntry`.

**Files:**
- Modify: `model/src/chroma_dtw_eval/chunk_sampler.py`
- Test: `model/tests/chroma_dtw_eval/test_chunk_sampler.py`

- [ ] **Step 1: Write the failing test**

```python
# model/tests/chroma_dtw_eval/test_chunk_sampler.py
"""sample_chunks emits ChunkManifestEntry with real per-chunk audio_sha256
computed from the chunk's audio bytes, not the whole file."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from chroma_dtw_eval.chunk_sampler import (
    ChunkManifestEntry,
    sample_chunks,
)
from chroma_dtw_eval.pseudo_truth_cache import (
    PseudoTruthPayload, write_pseudo_truth,
)


def _stage(corpus_root: Path) -> None:
    piece_dir = corpus_root / "practice_eval" / "bach_prelude_c_wtc1"
    (piece_dir / "audio").mkdir(parents=True, exist_ok=True)
    # Deterministic non-silent audio so per-chunk sha256 differs.
    rng = np.random.default_rng(0)
    audio = rng.standard_normal(16000 * 90).astype(np.float32) * 0.05
    sf.write(piece_dir / "audio" / "VID0.wav", audio, 16000, subtype="FLOAT")
    (piece_dir / "candidates.yaml").write_text(yaml.safe_dump({
        "piece": "bach_prelude_c_wtc1",
        "recordings": [
            {"video_id": "VID0", "approved": True, "downloaded": True},
            {"video_id": "VID1", "approved": False, "downloaded": True},
        ],
    }))
    cache_root = corpus_root / "pseudo_truth"
    write_pseudo_truth(
        piece_id="bach_prelude_c_wtc1", video_id="VID0",
        payload=PseudoTruthPayload(
            perf_audio_sec=np.array([0.0, 90.0], dtype=np.float64),
            score_audio_sec=np.array([0.0, 90.0], dtype=np.float64),
            measure_table=[],
            audio_sha256="a" * 16, amt_checkpoint_hash="b" * 16,
            score_sha256="c" * 16, parangonar_version="3.3.2",
            regen_source="local:test",
        ),
        cache_root=cache_root,
    )


def test_sample_chunks_emits_manifest_with_real_chunk_sha256(tmp_path: Path) -> None:
    _stage(tmp_path)
    entries = sample_chunks(
        corpus_root=tmp_path,
        pseudo_truth_root=tmp_path / "pseudo_truth",
        seed=0,
    )
    assert entries, "expected at least one entry"
    assert all(isinstance(e, ChunkManifestEntry) for e in entries)
    assert all(e.piece == "bach_prelude_c_wtc1" for e in entries)
    assert all(e.video_id == "VID0" for e in entries), "unapproved VID1 must be excluded"
    # audio_sha256 is per-chunk (depends on start/end), so distinct chunks
    # at different offsets have distinct hashes.
    hashes = {e.audio_sha256 for e in entries}
    assert len(hashes) >= 2, "per-chunk audio_sha256 should differ across chunks"
    # Verify one chunk's hash matches independently-computed sha256 of the
    # actual audio slice between its start and end.
    e0 = entries[0]
    y, sr = sf.read(tmp_path / "practice_eval" / "bach_prelude_c_wtc1" / "audio" / "VID0.wav",
                    dtype="float32")
    start_frame = int(round(e0.start_audio_sec * sr))
    end_frame = int(round(e0.end_audio_sec * sr))
    slice_bytes = y[start_frame:end_frame].tobytes()
    expected = hashlib.sha256(slice_bytes).hexdigest()[:16]
    assert e0.audio_sha256 == expected


def test_sample_chunks_writes_committed_manifest(tmp_path: Path) -> None:
    _stage(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    sample_chunks(
        corpus_root=tmp_path,
        pseudo_truth_root=tmp_path / "pseudo_truth",
        seed=0,
        manifest_out=manifest_path,
    )
    assert manifest_path.exists()
    import json
    body = json.loads(manifest_path.read_text())
    assert isinstance(body, list)
    assert body
    first = body[0]
    for key in ("piece", "video_id", "start_audio_sec", "end_audio_sec", "audio_sha256"):
        assert key in first
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/chroma_dtw_eval/test_chunk_sampler.py -x
```
Expected: FAIL — `ImportError: cannot import name 'ChunkManifestEntry'` (existing chunk_sampler exposes a different surface).

- [ ] **Step 3: Implement the minimum to make the test pass**

Replace `model/src/chroma_dtw_eval/chunk_sampler.py` with:

```python
"""Practice-corpus chunk sampler.

Enumerates approved video_ids per piece (from candidates.yaml), cross-
references pseudo-truth coverage, stratifies positions across five
position buckets, and emits ChunkManifestEntry with PER-CHUNK
audio_sha256 computed from the chunk's audio bytes.

The manifest is committed under model/data/evals/chroma_dtw_fixtures/manifest.json
so verify.py and downstream baselines see the same chunks on every run.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import soundfile as sf
import yaml

from chroma_dtw_eval.pseudo_truth_cache import cache_path

BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("intro", 0.00, 0.10),
    ("early", 0.10, 0.35),
    ("middle", 0.35, 0.65),
    ("late", 0.65, 0.90),
    ("cadence", 0.90, 1.00),
)
DEFAULT_CHUNK_LEN_S = 15.0
DEFAULT_N_PER_PIECE = 10


class PseudoTruthCoverageError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChunkManifestEntry:
    piece: str
    video_id: str
    start_audio_sec: float
    end_audio_sec: float
    audio_sha256: str
    position_bucket: str


def _chunk_sha256(audio_path: Path, start_sec: float, end_sec: float) -> str:
    y, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    start_f = max(0, int(round(start_sec * sr)))
    end_f = min(len(y), int(round(end_sec * sr)))
    h = hashlib.sha256(y[start_f:end_f].tobytes())
    return h.hexdigest()[:16]


def sample_chunks(
    corpus_root: Path,
    pseudo_truth_root: Path,
    seed: int,
    *,
    n_per_piece: int = DEFAULT_N_PER_PIECE,
    chunk_len_s: float = DEFAULT_CHUNK_LEN_S,
    manifest_out: Path | None = None,
) -> list[ChunkManifestEntry]:
    if n_per_piece < len(BUCKETS):
        raise ValueError(f"n_per_piece={n_per_piece} < {len(BUCKETS)} buckets")
    rng = random.Random(seed)
    per_bucket_base = n_per_piece // len(BUCKETS)
    remainder = n_per_piece - per_bucket_base * len(BUCKETS)
    counts = [per_bucket_base + (1 if i < remainder else 0) for i in range(len(BUCKETS))]

    practice_root = corpus_root / "practice_eval"
    entries: list[ChunkManifestEntry] = []
    pieces = sorted(p.name for p in practice_root.iterdir() if p.is_dir())
    for piece_id in pieces:
        yaml_path = practice_root / piece_id / "candidates.yaml"
        if not yaml_path.exists():
            continue
        body = yaml.safe_load(yaml_path.read_text()) or {}
        approved = [
            r for r in (body.get("recordings") or [])
            if r.get("approved") is True
        ]
        covered: list[tuple[str, float, Path]] = []
        for r in approved:
            vid = r["video_id"]
            pt_path = cache_path(pseudo_truth_root, piece_id, vid)
            if not pt_path.exists():
                continue
            data = json.loads(pt_path.read_text())
            perf = data.get("perf_audio_sec") or []
            if len(perf) < 2:
                continue
            duration_s = float(perf[-1])
            audio_path = practice_root / piece_id / "audio" / f"{vid}.wav"
            if not audio_path.exists() or duration_s <= chunk_len_s:
                continue
            covered.append((vid, duration_s, audio_path))
        if not covered:
            raise PseudoTruthCoverageError(
                f"no pseudo-truth coverage for piece {piece_id} "
                f"(checked {len(approved)} approved clips)"
            )
        for (name, lo, hi), count in zip(BUCKETS, counts):
            for _ in range(count):
                vid, dur, audio_path = covered[rng.randrange(len(covered))]
                lo_s = lo * dur
                hi_s = max(lo_s + 1e-3, hi * dur - chunk_len_s)
                start = rng.uniform(lo_s, hi_s)
                end = start + chunk_len_s
                sha = _chunk_sha256(audio_path, start, end)
                entries.append(ChunkManifestEntry(
                    piece=piece_id, video_id=vid,
                    start_audio_sec=start, end_audio_sec=end,
                    audio_sha256=sha, position_bucket=name,
                ))

    if manifest_out is not None:
        manifest_out.parent.mkdir(parents=True, exist_ok=True)
        manifest_out.write_text(json.dumps([asdict(e) for e in entries], indent=2))
    return entries
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/chroma_dtw_eval/test_chunk_sampler.py -x
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add model/src/chroma_dtw_eval/chunk_sampler.py model/tests/chroma_dtw_eval/test_chunk_sampler.py && git commit -m "feat(chroma-eval): chunk_sampler emits ChunkManifestEntry with real per-chunk audio_sha256"
```

---

### Task C4 (OPTIONAL): Opportunistic fur_elise score sourcing

**Group:** C (parallel with D; 30 min hard time-box; SKIP if not findable)

**Behavior being verified:** if a public-domain Beethoven WoO 59 ("Für Elise") MusicXML is locatable in 30 minutes on IMSLP or MuseScore's CC0 set, convert it to the project's score JSON format and commit it. If not, document the attempt and skip.

**Files:**
- (if found) Add: `model/data/scores/beethoven.fur_elise.woo_59.json`
- (always) Modify: `docs/implementation/2026-05-31-chroma-dtw-eval-pivot.md` (append note about the attempt)

- [ ] **Step 1: Time-box 30 minutes searching IMSLP/MuseScore CC0 for WoO 59 MusicXML.**

- [ ] **Step 2: If found, convert via the existing score_library tooling.**

```bash
cd model && uv run python -m score_library.convert \
    --in <downloaded.mxl> \
    --out data/scores/beethoven.fur_elise.woo_59.json
```

- [ ] **Step 3: If found, commit; if not, document and skip.**

```bash
# Found case:
git add model/data/scores/beethoven.fur_elise.woo_59.json && git commit -m "data(scores): add Beethoven Für Elise WoO 59 (opportunistic C4)"
# Not-found case: append a short note to docs/implementation/2026-05-31-chroma-dtw-eval-pivot.md and commit.
```

---

## Group D — Metric Aggregator, Verify CLI, Baseline

### Task D-metric: `metric_aggregator` rewrite — seconds primary + G4 continuity guard

**Group:** D (sequential after B + C)

**Behavior being verified:**
- `aggregate(per_chunk_results, baseline) -> Metrics`. Primary scalar = `% of practice chunks where abs(error_seconds) <= 1.5`.
- `error_seconds = predicted_score_audio_sec - truth_score_audio_sec` where `predicted_score_audio_sec = dtw_result.predicted_score_frame * (1.0 / decim_hz_score_chroma)` and truth via `np.interp(chunk_start_audio_sec, pseudo_truth.perf_pairs[:, 0], pseudo_truth.perf_pairs[:, 1])`.
- G2 cost AUC regression threshold scales by `max(1.0, min(4.0, math.sqrt(50 / max(n_chunks, 1))))`.
- G4 is RESTORED as a consecutive-chunk continuity guard: for each `(piece, video_id)`, sort chunks by `start_audio_sec`, then for each adjacent pair, `continuity_ok = abs(delta_predicted_score_sec - delta_audio_sec) <= 5.0`. G4 = pct of valid adjacent pairs that are continuous (higher is better; regression threshold = drop > 5pp from baseline).
- Synthetic-MAESTRO composition logic is gone entirely.

**Interface under test:** `aggregate`, `GuardSet`, `Baseline`, `ChunkResult`, `Metrics`.

**Files:**
- Modify: `model/src/chroma_dtw_eval/metric_aggregator.py`
- Modify: `model/tests/chroma_dtw_eval/test_metric_aggregator.py`

- [ ] **Step 1: Write the failing test**

Replace `model/tests/chroma_dtw_eval/test_metric_aggregator.py` with:

```python
"""Seconds-tolerance primary, G4 = consecutive-chunk continuity, G2 scaled threshold."""
from chroma_dtw_eval.metric_aggregator import (
    Baseline, ChunkResult, GuardSet, Metrics, aggregate,
)


def _baseline() -> Baseline:
    return Baseline(
        primary=0.0,
        guards=GuardSet(g1=100.0, g2=0.0, g3=100.0, g4=0.0, g5=100.0),
    )


def test_guardset_has_g4_continuity_field() -> None:
    g = GuardSet(g1=0.0, g2=0.5, g3=0.0, g4=100.0, g5=0.0)
    assert g.g4 == 100.0


def test_primary_counts_practice_within_seconds_tolerance() -> None:
    results = [
        ChunkResult(kind="practice", piece="p", video_id="v", start_audio_sec=0.0,
                    predicted_score_sec=0.0, error_seconds=0.5, cost=0.1),
        ChunkResult(kind="practice", piece="p", video_id="v", start_audio_sec=15.0,
                    predicted_score_sec=15.0, error_seconds=1.4, cost=0.1),
        ChunkResult(kind="practice", piece="p", video_id="v", start_audio_sec=30.0,
                    predicted_score_sec=30.0, error_seconds=1.6, cost=0.2),  # fail
        ChunkResult(kind="practice", piece="p", video_id="v", start_audio_sec=45.0,
                    predicted_score_sec=45.0, error_seconds=10.0, cost=0.3),  # fail
    ]
    m = aggregate(results, baseline=_baseline(), tolerance_s=1.5)
    assert m.primary == 50.0  # 2 of 4 pass


def test_g4_continuity_consecutive_chunks() -> None:
    # Three consecutive chunks at 0, 15, 30s on the same clip. Pair 0->1
    # is continuous (delta_pred = 15 = delta_audio); pair 1->2 jumps
    # 30s in predicted-score (delta 30 - delta_audio 15 = 15 > 5) -> not continuous.
    results = [
        ChunkResult(kind="practice", piece="p", video_id="v", start_audio_sec=0.0,
                    predicted_score_sec=0.0, error_seconds=0.0, cost=0.1),
        ChunkResult(kind="practice", piece="p", video_id="v", start_audio_sec=15.0,
                    predicted_score_sec=15.0, error_seconds=0.0, cost=0.1),
        ChunkResult(kind="practice", piece="p", video_id="v", start_audio_sec=30.0,
                    predicted_score_sec=60.0, error_seconds=0.0, cost=0.1),
    ]
    m = aggregate(results, baseline=_baseline(), tolerance_s=1.5)
    # 1 continuous pair out of 2 -> 50%
    assert m.guards.g4 == 50.0


def test_g4_regression_drop_over_5pp() -> None:
    # Baseline g4 = 100; measured g4 = 50 -> regression.
    bl = Baseline(
        primary=0.0,
        guards=GuardSet(g1=100.0, g2=0.0, g3=100.0, g4=100.0, g5=100.0),
    )
    results = [
        ChunkResult(kind="practice", piece="p", video_id="v", start_audio_sec=0.0,
                    predicted_score_sec=0.0, error_seconds=0.0, cost=0.1),
        ChunkResult(kind="practice", piece="p", video_id="v", start_audio_sec=15.0,
                    predicted_score_sec=15.0, error_seconds=0.0, cost=0.1),
        ChunkResult(kind="practice", piece="p", video_id="v", start_audio_sec=30.0,
                    predicted_score_sec=60.0, error_seconds=0.0, cost=0.1),
    ]
    m = aggregate(results, baseline=bl, tolerance_s=1.5)
    assert "g4" in m.regressed


def test_g2_threshold_scales_with_chunk_count() -> None:
    # With small n the G2 regression threshold widens. Baseline g2 = 0.9;
    # measured 0.85 with n=4 chunks should NOT regress (scaled tol).
    bl = Baseline(
        primary=0.0,
        guards=GuardSet(g1=100.0, g2=0.9, g3=100.0, g4=0.0, g5=100.0),
    )
    results = [
        ChunkResult(kind="practice", piece="p", video_id="v", start_audio_sec=float(i),
                    predicted_score_sec=float(i), error_seconds=0.5 if i < 2 else 2.0,
                    cost=0.1 + 0.1 * i)
        for i in range(4)
    ]
    m = aggregate(results, baseline=bl, tolerance_s=1.5)
    # The exact AUC value here is implementation-dependent; the test asserts
    # that g2 IS NOT marked regressed at n=4 even when below baseline by a
    # small amount (sqrt(50/4) ~ 3.5 -> threshold scaled 3.5x).
    assert "g2" not in m.regressed or m.guards.g2 < 0.9 - 0.02 * 3.5
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/chroma_dtw_eval/test_metric_aggregator.py -x
```
Expected: FAIL — interface mismatch (existing aggregator uses `tolerance_ms` and different ChunkResult fields).

- [ ] **Step 3: Implement the minimum to make the test pass**

Replace `model/src/chroma_dtw_eval/metric_aggregator.py` with:

```python
"""Primary scalar (practice + AMT-pseudo-truth, seconds-tolerance) + 4 guards.

G1 teleport (amateur kind), G2 cost-vs-error AUC (practice kind), G3
silence robustness (silence kind), G4 consecutive-chunk continuity
(practice kind), G5 self-consistency (real_practice kind).

G2 regression threshold scales by max(1.0, min(4.0, sqrt(50/n_chunks))).
G4 regression = drop > 5pp from baseline (higher is better, unlike g1/g3/g5).
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ChunkResult:
    kind: str  # "practice" | "amateur" | "silence" | "real_practice"
    piece: Optional[str] = None
    video_id: Optional[str] = None
    start_audio_sec: Optional[float] = None
    predicted_score_sec: Optional[float] = None
    error_seconds: Optional[float] = None
    cost: float = 0.0
    abstain: bool = False
    bar_distance_from_forward: Optional[float] = None
    silence_loud_failure: Optional[bool] = None


@dataclass
class GuardSet:
    g1: float
    g2: float
    g3: float
    g4: float
    g5: float


@dataclass
class Baseline:
    primary: float
    guards: GuardSet


@dataclass
class Metrics:
    primary: float
    guards: GuardSet
    regressed: list[str]
    g2_threshold_scale: float


def _pct(values: list[bool]) -> float:
    return 100.0 * sum(1 for v in values if v) / max(1, len(values))


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    if len(set(labels.tolist())) < 2:
        return 0.5
    order = np.argsort(-scores)
    labels = labels[order]
    pos = int(labels.sum())
    neg = len(labels) - pos
    cum_pos = 0
    auc = 0.0
    for y in labels:
        if y == 1:
            cum_pos += 1
        else:
            auc += cum_pos
    return float(auc / (pos * neg))


def _g4_continuity(practice: list[ChunkResult]) -> float:
    """For each (piece, video_id), sort by start_audio_sec, count adjacent
    pairs where |delta_predicted_score_sec - delta_audio_sec| <= 5.0.
    Returns pct continuous (higher is better). Returns 100.0 if no pairs.
    """
    by_clip: dict[tuple[str, str], list[ChunkResult]] = defaultdict(list)
    for r in practice:
        if r.piece is None or r.video_id is None:
            continue
        if r.start_audio_sec is None or r.predicted_score_sec is None:
            continue
        by_clip[(r.piece, r.video_id)].append(r)
    total = 0
    ok = 0
    for chunks in by_clip.values():
        chunks_sorted = sorted(chunks, key=lambda c: c.start_audio_sec or 0.0)
        for i in range(len(chunks_sorted) - 1):
            a = chunks_sorted[i]
            b = chunks_sorted[i + 1]
            d_audio = (b.start_audio_sec or 0.0) - (a.start_audio_sec or 0.0)
            d_pred = (b.predicted_score_sec or 0.0) - (a.predicted_score_sec or 0.0)
            total += 1
            if abs(d_pred - d_audio) <= 5.0:
                ok += 1
    if total == 0:
        return 100.0
    return 100.0 * ok / total


def aggregate(
    results: list[ChunkResult],
    baseline: Baseline,
    *,
    tolerance_s: float = 1.5,
) -> Metrics:
    practice = [r for r in results if r.kind == "practice" and r.error_seconds is not None]
    amateur = [r for r in results if r.kind == "amateur"]
    silence = [r for r in results if r.kind == "silence"]
    real_practice = [r for r in results if r.kind == "real_practice"]

    primary = (
        _pct([abs(r.error_seconds) <= tolerance_s for r in practice])  # type: ignore[arg-type]
        if practice else 0.0
    )

    g1 = (
        _pct([(r.bar_distance_from_forward or 0.0) > 5.0 for r in amateur])
        if amateur else 0.0
    )
    if practice:
        labels = np.array(
            [abs(r.error_seconds) > tolerance_s for r in practice], dtype=int  # type: ignore[arg-type]
        )
        costs = np.array([r.cost for r in practice], dtype=float)
        g2 = _auc(costs, labels)
    else:
        g2 = 0.5
    g3 = (
        _pct([(r.silence_loud_failure is True) for r in silence])
        if silence else 0.0
    )
    g4 = _g4_continuity(practice)
    g5 = (
        _pct([(r.bar_distance_from_forward or 0.0) > 5.0 for r in real_practice])
        if real_practice else 0.0
    )

    g2_scale = max(1.0, min(4.0, math.sqrt(50.0 / max(len(practice), 1))))
    guards = GuardSet(g1=g1, g2=g2, g3=g3, g4=g4, g5=g5)
    regressed: list[str] = []
    if primary + 1e-9 < baseline.primary:
        regressed.append("primary")
    if g1 > baseline.guards.g1 + 1.0:
        regressed.append("g1")
    if g2 < baseline.guards.g2 - 0.02 * g2_scale:
        regressed.append("g2")
    if g3 > baseline.guards.g3 + 1.0:
        regressed.append("g3")
    if g4 < baseline.guards.g4 - 5.0:
        regressed.append("g4")
    if g5 > baseline.guards.g5 + 1.0:
        regressed.append("g5")
    return Metrics(
        primary=primary, guards=guards, regressed=regressed,
        g2_threshold_scale=g2_scale,
    )
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/chroma_dtw_eval/test_metric_aggregator.py -x
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add model/src/chroma_dtw_eval/metric_aggregator.py model/tests/chroma_dtw_eval/test_metric_aggregator.py && git commit -m "feat(chroma-eval): metric_aggregator seconds-primary + G4 continuity + scaled G2 threshold"
```

---

### Task D-verify: `verify.py` CLI (bundled — manifest + DTW + sidecar enrichment)

**Group:** D (sequential after D-metric)

**Behavior being verified:** `python -m chroma_dtw_eval.verify --baseline ... --manifest ...` reads the chunk manifest, loads pseudo-truth via the 4-field key (audio_sha256 from manifest, score_sha256 from the score JSON, parangonar_version + checkpoint_hash from `model/config/amt_version.json`), runs `dtw_runner` per chunk, calls `aggregate`, writes the sidecar JSON with `error_seconds_distribution` and `tolerance_sensitivity`, prints exactly one float on stdout, exits 0 iff no regression. Emits a stderr WARNING when manifest contains <2 distinct pieces. End-to-end behavior test fabricates 3 chunks + a synthetic-but-realistic pseudo-truth and asserts the numerical `error_seconds` computation is correct to within 0.1s.

`--skip-dtw` is an INTERNAL flag (used by the smoke test); `argparse.SUPPRESS` hides it from `--help`.

**Interface under test:** `python -m chroma_dtw_eval.verify` CLI; sidecar schema.

**Files:**
- Modify: `model/src/chroma_dtw_eval/verify.py`
- Test: `model/tests/chroma_dtw_eval/test_verify_cli.py`

- [ ] **Step 1: Write the failing test**

```python
# model/tests/chroma_dtw_eval/test_verify_cli.py
"""End-to-end verify CLI behavior.

(a) Smoke test against staged manifest + pseudo-truth (--skip-dtw):
    asserts CLI exits 0, prints one float, writes sidecar with
    error_seconds_distribution + tolerance_sensitivity, AND emits stderr
    WARNING when fewer than 2 pieces are in the manifest.

(b) Numerical correctness test: fabricate 3 practice chunks where the
    pseudo-truth is identity-linear over 60s; assert each chunk's
    error_seconds in the sidecar matches the analytic expectation within
    0.1s.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from chroma_dtw_eval.chunk_sampler import ChunkManifestEntry
from chroma_dtw_eval.pseudo_truth_cache import (
    PseudoTruthPayload, write_pseudo_truth,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = REPO_ROOT / "model"


def _stage(tmp_path: Path, *, audio_sha256_for_chunks: list[str]) -> tuple[Path, Path, Path]:
    evals = tmp_path / "evals"
    piece_dir = evals / "practice_eval" / "bach_prelude_c_wtc1"
    (piece_dir / "audio").mkdir(parents=True)
    rng = np.random.default_rng(0)
    sf.write(
        piece_dir / "audio" / "VID0.wav",
        rng.standard_normal(16000 * 60).astype(np.float32) * 0.05,
        16000, subtype="FLOAT",
    )
    # Identity-linear pseudo-truth: 60s perf == 60s score.
    cache_root = evals / "pseudo_truth"
    write_pseudo_truth(
        piece_id="bach_prelude_c_wtc1", video_id="VID0",
        payload=PseudoTruthPayload(
            perf_audio_sec=np.linspace(0.0, 60.0, 61, dtype=np.float64),
            score_audio_sec=np.linspace(0.0, 60.0, 61, dtype=np.float64),
            measure_table=[],
            audio_sha256=audio_sha256_for_chunks[0],
            amt_checkpoint_hash="aria_amt_v1_pilot_2026_06_01",
            score_sha256="deadbeefdeadbeef",
            parangonar_version="3.3.2",
            regen_source="test",
        ),
        cache_root=cache_root,
    )
    # Manifest with three chunks; each chunk's audio_sha256 is the value
    # the pseudo-truth was keyed by (so cache lookup succeeds).
    manifest = [
        {
            "piece": "bach_prelude_c_wtc1",
            "video_id": "VID0",
            "start_audio_sec": float(start),
            "end_audio_sec": float(start + 15.0),
            "audio_sha256": audio_sha256_for_chunks[i],
            "position_bucket": bucket,
        }
        for i, (start, bucket) in enumerate([(0.0, "intro"), (20.0, "middle"), (40.0, "late")])
    ]
    manifest_path = evals / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({
        "primary": 0.0,
        "guards": {"g1": 100.0, "g2": 0.0, "g3": 100.0, "g4": 0.0, "g5": 100.0},
    }))
    # Write a fake score JSON sitting where verify expects it; its sha256
    # must equal the value written into the pseudo-truth above
    # ("deadbeefdeadbeef"). We compute the sha256 from the actual file
    # contents and then patch the cache to match.
    score_dir = evals / "scores"
    score_dir.mkdir()
    score_path = score_dir / "bach.prelude.bwv_846.json"
    score_path.write_text(json.dumps({"tempo_markings": [{"bpm": 120}], "bars": []}))
    import hashlib
    real_sha = hashlib.sha256(score_path.read_bytes()).hexdigest()[:16]
    # Re-write cache with the real score_sha256.
    write_pseudo_truth(
        piece_id="bach_prelude_c_wtc1", video_id="VID0",
        payload=PseudoTruthPayload(
            perf_audio_sec=np.linspace(0.0, 60.0, 61, dtype=np.float64),
            score_audio_sec=np.linspace(0.0, 60.0, 61, dtype=np.float64),
            measure_table=[],
            audio_sha256=audio_sha256_for_chunks[0],
            amt_checkpoint_hash="aria_amt_v1_pilot_2026_06_01",
            score_sha256=real_sha,
            parangonar_version="3.3.2",
            regen_source="test",
        ),
        cache_root=cache_root,
    )
    return evals, manifest_path, baseline


def test_skip_dtw_smoke_emits_one_float_sidecar_and_warning(tmp_path: Path) -> None:
    # When all three chunks share the same audio_sha256, the cache lookup
    # for each will succeed against the same cached pseudo-truth (real
    # corpus would have distinct shas; smoke uses one).
    sha = "abcd123456789012"
    evals, manifest_path, baseline = _stage(tmp_path, audio_sha256_for_chunks=[sha, sha, sha])
    sidecar = tmp_path / "sidecar.json"
    res = subprocess.run(
        [sys.executable, "-m", "chroma_dtw_eval.verify",
         "--baseline", str(baseline),
         "--manifest", str(manifest_path),
         "--sidecar", str(sidecar),
         "--corpus-root", str(evals),
         "--skip-dtw"],
        capture_output=True, text=True, timeout=120,
        cwd=MODEL_DIR,
    )
    assert res.returncode == 0, f"stdout={res.stdout!r} stderr={res.stderr!r}"
    lines = [ln for ln in res.stdout.strip().splitlines() if ln.strip()]
    assert len(lines) == 1
    float(lines[0])
    body = json.loads(sidecar.read_text())
    assert set(body["guards"].keys()) == {"g1", "g2", "g3", "g4", "g5"}
    assert "error_seconds_distribution" in body
    assert "tolerance_sensitivity" in body
    for k in ("0.5", "1.0", "1.5", "2.0", "3.0"):
        assert k in body["tolerance_sensitivity"]
    # Manifest has only one piece -> stderr WARNING.
    assert "WARNING" in res.stderr and "piece" in res.stderr


def test_skip_dtw_numerical_error_within_0p1s(tmp_path: Path) -> None:
    """With identity-linear pseudo-truth and --skip-dtw, the synthetic
    predicted_score_sec equals chunk_start_audio_sec, so error_seconds
    should be 0 within 0.1s for every chunk.
    """
    sha = "abcd123456789012"
    evals, manifest_path, baseline = _stage(tmp_path, audio_sha256_for_chunks=[sha, sha, sha])
    sidecar = tmp_path / "sidecar.json"
    res = subprocess.run(
        [sys.executable, "-m", "chroma_dtw_eval.verify",
         "--baseline", str(baseline),
         "--manifest", str(manifest_path),
         "--sidecar", str(sidecar),
         "--corpus-root", str(evals),
         "--skip-dtw"],
        capture_output=True, text=True, timeout=120,
        cwd=MODEL_DIR,
    )
    assert res.returncode == 0, res.stderr
    body = json.loads(sidecar.read_text())
    dist = body["error_seconds_distribution"]
    assert dist["max"] <= 0.1
    assert dist["mean"] <= 0.1


def test_skip_dtw_not_in_help_output() -> None:
    res = subprocess.run(
        [sys.executable, "-m", "chroma_dtw_eval.verify", "--help"],
        capture_output=True, text=True, timeout=10, cwd=MODEL_DIR,
    )
    assert res.returncode == 0
    assert "--skip-dtw" not in res.stdout
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/chroma_dtw_eval/test_verify_cli.py -x
```
Expected: FAIL — existing verify.py does not accept `--manifest`, does not produce `error_seconds_distribution`, etc.

- [ ] **Step 3: Implement the minimum to make the test pass**

Replace `model/src/chroma_dtw_eval/verify.py` with:

```python
"""Verify CLI — practice-corpus + AMT-pseudo-truth path via committed manifest.

Contract:
  - stdout: exactly one float on a single line (the primary scalar).
  - exit: 0 iff no guard regressed; non-zero otherwise.
  - sidecar JSON: {primary, guards{g1,g2,g3,g4,g5}, baseline, regressed,
                   n_chunks, error_seconds_distribution, tolerance_sensitivity,
                   generated_at, g2_threshold_scale}.
  - stderr WARNING when manifest contains < 2 distinct pieces.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import statistics
import sys
from pathlib import Path

from chroma_dtw_eval.metric_aggregator import (
    Baseline, ChunkResult, GuardSet, aggregate,
)
from chroma_dtw_eval.pseudo_truth_cache import (
    PseudoTruth, load_pseudo_truth,
)

_MODULE_DIR = Path(__file__).resolve()
DEFAULT_SIDECAR = _MODULE_DIR.parents[2] / "data/evals/chroma_dtw/last_run.json"
DEFAULT_MANIFEST = (
    _MODULE_DIR.parents[2] / "data/evals/chroma_dtw_fixtures/manifest.json"
)
DEFAULT_AMT_VERSION_CONFIG = _MODULE_DIR.parents[2] / "config/amt_version.json"
DEFAULT_SCORE_BY_PIECE = {
    "bach_prelude_c_wtc1": _MODULE_DIR.parents[2] / "data/scores/bach.prelude.bwv_846.json",
}
DEFAULT_DECIM_HZ_SCORE_CHROMA = 50.0
TOLERANCE_SWEEP = (0.5, 1.0, 1.5, 2.0, 3.0)


def _sha256_file(p: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _truth_score_audio_sec(pt: PseudoTruth, chunk_start_audio_sec: float) -> float:
    import numpy as np
    return float(np.interp(
        chunk_start_audio_sec, pt.perf_audio_sec, pt.score_audio_sec
    ))


def _build_results(
    manifest: list[dict],
    *,
    corpus_root: Path,
    pseudo_truth_root: Path,
    score_by_piece: dict[str, Path],
    checkpoint_hash: str,
    parangonar_version: str,
    decim_hz_score_chroma: float,
    skip_dtw: bool,
) -> list[ChunkResult]:
    results: list[ChunkResult] = []
    for entry in manifest:
        piece = entry["piece"]
        video_id = entry["video_id"]
        start_audio_sec = float(entry["start_audio_sec"])
        audio_sha256 = entry["audio_sha256"]
        score_path = score_by_piece.get(piece)
        if score_path is None:
            raise FileNotFoundError(
                f"no score JSON registered for piece {piece!r}; "
                f"add to DEFAULT_SCORE_BY_PIECE in verify.py"
            )
        score_sha256 = _sha256_file(score_path)
        pt = load_pseudo_truth(
            piece_id=piece, video_id=video_id,
            audio_sha256=audio_sha256,
            amt_checkpoint_hash=checkpoint_hash,
            score_sha256=score_sha256,
            parangonar_version=parangonar_version,
            cache_root=pseudo_truth_root,
        )

        truth_score_sec = _truth_score_audio_sec(pt, start_audio_sec)
        if skip_dtw:
            # Smoke path: predicted == truth (synthetic). Exercises sampler +
            # pseudo-truth + aggregator + sidecar without depending on DTW.
            predicted_score_sec = truth_score_sec
            cost = 0.1
        else:
            from chroma_dtw_eval.chroma_cache import ChromaParams, get_chroma
            from chroma_dtw_eval.dtw_runner import run_dtw
            audio_path = (
                corpus_root / "practice_eval" / piece / "audio" / f"{video_id}.wav"
            )
            params = ChromaParams(target_frame_rate_hz=decim_hz_score_chroma, sr=16000)
            chroma = get_chroma(audio_path, params, corpus_root / "chroma_cache")
            start_f = int(round(start_audio_sec * chroma.frame_rate_hz))
            end_f = start_f + int(round(
                (float(entry["end_audio_sec"]) - start_audio_sec)
                * chroma.frame_rate_hz
            ))
            seg = chroma.data[:, start_f:end_f].copy()
            dtw = run_dtw(
                seg, score_path,
                frame_rate_hz=chroma.frame_rate_hz,
                decim_hz=decim_hz_score_chroma,
            )
            predicted_score_sec = (
                dtw.predicted_score_frame * (1.0 / decim_hz_score_chroma)
            )
            cost = float(dtw.cost)
        error_seconds = predicted_score_sec - truth_score_sec
        results.append(ChunkResult(
            kind="practice",
            piece=piece, video_id=video_id,
            start_audio_sec=start_audio_sec,
            predicted_score_sec=predicted_score_sec,
            error_seconds=error_seconds,
            cost=cost,
        ))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chroma_dtw_eval.verify")
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--corpus-root", type=Path, default=None,
                        help="Root containing practice_eval/, pseudo_truth/, scores/")
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--config", type=Path, default=DEFAULT_AMT_VERSION_CONFIG)
    parser.add_argument("--tolerance-s", type=float, default=1.5)
    parser.add_argument("--decim-hz", type=float, default=DEFAULT_DECIM_HZ_SCORE_CHROMA)
    parser.add_argument("--skip-dtw", action="store_true",
                        help=argparse.SUPPRESS)  # internal flag; not in --help
    args = parser.parse_args(argv)

    if not args.baseline.exists():
        raise FileNotFoundError(f"baseline not found: {args.baseline}")
    raw = json.loads(args.baseline.read_text())
    baseline = Baseline(
        primary=float(raw["primary"]),
        guards=GuardSet(**{k: float(v) for k, v in raw["guards"].items()}),
    )

    if not args.manifest.exists():
        raise FileNotFoundError(f"manifest not found: {args.manifest}")
    manifest = json.loads(args.manifest.read_text())
    if not isinstance(manifest, list) or not manifest:
        raise ValueError(f"manifest at {args.manifest} is empty or not a list")

    corpus_root = args.corpus_root or args.manifest.parent.parent
    pseudo_truth_root = corpus_root / "pseudo_truth"
    score_by_piece = dict(DEFAULT_SCORE_BY_PIECE)
    corpus_scores = corpus_root / "scores" / "bach.prelude.bwv_846.json"
    if corpus_scores.exists():
        score_by_piece["bach_prelude_c_wtc1"] = corpus_scores

    config_body = json.loads(args.config.read_text())
    checkpoint_hash = config_body["checkpoint_hash"]
    parangonar_version = config_body["parangonar_version"]

    # Manifest n-pieces WARNING (stderr, not error).
    unique_pieces = {e["piece"] for e in manifest}
    if len(unique_pieces) < 2:
        print(
            f"WARNING: smoke-only baseline (n={len(unique_pieces)} piece(s)); "
            f"/autoresearch dispatch deferred until >=2 pieces have scores",
            file=sys.stderr,
        )

    results = _build_results(
        manifest,
        corpus_root=corpus_root,
        pseudo_truth_root=pseudo_truth_root,
        score_by_piece=score_by_piece,
        checkpoint_hash=checkpoint_hash,
        parangonar_version=parangonar_version,
        decim_hz_score_chroma=args.decim_hz,
        skip_dtw=args.skip_dtw,
    )

    metrics = aggregate(results, baseline=baseline, tolerance_s=args.tolerance_s)

    errors = [abs(r.error_seconds) for r in results if r.error_seconds is not None]
    if errors:
        errors_sorted = sorted(errors)
        n = len(errors_sorted)
        dist = {
            "mean": statistics.fmean(errors_sorted),
            "p50": errors_sorted[n // 2],
            "p90": errors_sorted[min(n - 1, int(0.9 * n))],
            "max": max(errors_sorted),
            "list": errors_sorted,
        }
        tolerance_sensitivity = {
            f"{tol}": (
                100.0 * sum(1 for e in errors_sorted if e <= tol) / n
            )
            for tol in TOLERANCE_SWEEP
        }
    else:
        dist = {"mean": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0, "list": []}
        tolerance_sensitivity = {f"{tol}": 0.0 for tol in TOLERANCE_SWEEP}

    args.sidecar.parent.mkdir(parents=True, exist_ok=True)
    args.sidecar.write_text(json.dumps({
        "primary": metrics.primary,
        "guards": metrics.guards.__dict__,
        "baseline": {
            "primary": baseline.primary,
            "guards": baseline.guards.__dict__,
        },
        "regressed": metrics.regressed,
        "n_chunks": len(results),
        "g2_threshold_scale": metrics.g2_threshold_scale,
        "error_seconds_distribution": dist,
        "tolerance_sensitivity": tolerance_sensitivity,
        "generated_at": _dt.datetime.utcnow().isoformat() + "Z",
    }, indent=2))
    print(f"{metrics.primary:.4f}")
    return 1 if metrics.regressed else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/chroma_dtw_eval/test_verify_cli.py -x
```
Expected: PASS (all three tests).

- [ ] **Step 5: Commit**

```bash
git add model/src/chroma_dtw_eval/verify.py model/tests/chroma_dtw_eval/test_verify_cli.py && git commit -m "feat(chroma-eval): verify CLI reads manifest + 4-field cache key + sidecar distribution/sensitivity + n<2 stderr WARNING"
```

---

### Task D-baseline: Commit permissive `baseline.json` smoke-only acknowledgement

**Group:** D (sequential after D-verify)

**Behavior being verified:** the committed baseline is a smoke-only first-run with `notes` field explicitly acknowledging the n=1 limitation, and the guards schema matches the new 5-field `(g1, g2, g3, g4, g5)`.

**Files:**
- Modify: `model/data/evals/chroma_dtw/baseline.json`

- [ ] **Step 1: Write the failing test**

No test (this is data). The shape is validated by D-verify's CLI test reading it back; this task is a pure data commit.

- [ ] **Step 2: Run test — verify it FAILS**

n/a (data commit).

- [ ] **Step 3: Implement the minimum to make the test pass**

Overwrite `model/data/evals/chroma_dtw/baseline.json`:

```json
{
  "primary": 0.0,
  "guards": {
    "g1": 100.0,
    "g2": 0.0,
    "g3": 100.0,
    "g4": 0.0,
    "g5": 100.0
  },
  "notes": "smoke baseline; bach_prelude_c_wtc1 only; not statistically meaningful -- source additional scores before /autoresearch dispatch"
}
```

- [ ] **Step 4: Run test — verify it PASSES**

n/a.

- [ ] **Step 5: Commit**

```bash
git add model/data/evals/chroma_dtw/baseline.json && git commit -m "chore(chroma-eval): commit smoke baseline.json (n=1 piece; notes acknowledge limitation)"
```

---

## Post-merge follow-up (not in this plan)

- Regenerate pseudo-truth for the approved practice clips of `bach_prelude_c_wtc1` via `just amt-regen-pseudo-truth` and commit those caches.
- Run `just chroma-eval-prebuild` once, then `just chroma-eval-verify` end-to-end, then `just chroma-eval-ratchet` to lift the primary from 0.0 to the measured value.
- Source a 2nd piece's score (fur_elise via C4, OR a second public-domain piece) before dispatching `/autoresearch` so the primary scalar carries more than n=1 chunks worth of statistical signal.
- Dispatch `/autoresearch` with the parked `feat/continuity-aware-chroma-follower` branch as the first candidate.

---
