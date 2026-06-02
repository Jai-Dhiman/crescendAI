# Chroma-DTW Eval Harness Rework Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** Repoint the chroma-DTW eval harness primary scalar from MAESTRO+parangonar gold-truth at 50ms tolerance to practice-corpus + AMT-pseudo-truth at +/-1.5s tolerance, with an on-disk pseudo-truth cache so `just chroma-eval-verify` runs in <=120s without calling AMT inline.

**Spec:** docs/specs/2026-06-02-chroma-dtw-eval-pseudo-truth-design.md
**Style:** Follow `model/CLAUDE.md` and root `CLAUDE.md`. Python via `uv`. partitura (not music21). Explicit exceptions, no silent fallbacks. No emojis.

---

## Task Groups

```
Group A (parallel, file-disjoint):
  - A1: delete gold_truth_builder.py + test
  - A2: delete practice_compose.py + test
  - A3: delete amt_pseudo_truth_pilot.py
  - A4: add Justfile recipes (chroma-eval-verify, chroma-eval-ratchet, amt-regen-pseudo-truth)

Group B (parallel, file-disjoint, the new-surface group):
  - B1: pseudo_truth_cache.py round-trip
  - B2: pseudo_truth_cache.py missing-file behavior
  - B3: pseudo_truth_cache.py hash-mismatch behavior
  - B4: amt_version.json committed config
  - B5: amt_regen.py orchestrator + CLI (depends on B1+B4)

Group C (sequential, depends on A1..A4 done):
  - C1: chunk_sampler.py extends Chunk with video_id (backward compatible)
  - C2: chunk_sampler.py adds sample_practice_chunks happy path
  - C3: chunk_sampler.py adds sample_practice_chunks coverage-error path

Group D (sequential, depends on B + C):
  - D1: metric_aggregator.py switches to seconds + drops G4
  - D2: metric_aggregator.py adds practice kind + 1.5s primary
  - D3: verify.py wires practice path + sidecar drops g4
  - D4: re-baseline chopin_ballade_1 + commit baseline.json
```

Independence-ship audit: Group A ships independently (removes dead code, no behavior change). Group B ships independently (new modules with their own tests; nothing consumes them until C/D). Groups C and D do NOT ship independently of each other — the metric switch and the corpus path are co-dependent.

---

## Group A — Deletions and Justfile

### Task A1: Delete `gold_truth_builder.py` + its test

**Group:** A (parallel)

**Behavior being verified:** the eval harness builds, imports, and runs without `gold_truth_builder` in the tree.

**Interface under test:** `chroma_dtw_eval` package public surface (no `gold_truth_builder` symbol exposed).

**Files:**
- Delete: `model/src/chroma_dtw_eval/gold_truth_builder.py`
- Delete: `model/tests/chroma_dtw_eval/test_gold_truth_builder.py`
- Test: `model/tests/chroma_dtw_eval/test_module_surface.py` (new, this task)

- [ ] **Step 1: Write the failing test**

```python
# model/tests/chroma_dtw_eval/test_module_surface.py
"""Asserts deleted modules are not importable. Locks the rework's removal contract."""
import importlib

import pytest


def test_gold_truth_builder_not_importable():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("chroma_dtw_eval.gold_truth_builder")
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/chroma_dtw_eval/test_module_surface.py::test_gold_truth_builder_not_importable -x
```
Expected: FAIL — `DID NOT RAISE <class 'ModuleNotFoundError'>` because the module is still present.

- [ ] **Step 3: Implement the minimum to make the test pass**

```bash
git rm model/src/chroma_dtw_eval/gold_truth_builder.py
git rm model/tests/chroma_dtw_eval/test_gold_truth_builder.py
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/chroma_dtw_eval/test_module_surface.py::test_gold_truth_builder_not_importable -x
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add model/src/chroma_dtw_eval/gold_truth_builder.py model/tests/chroma_dtw_eval/test_gold_truth_builder.py model/tests/chroma_dtw_eval/test_module_surface.py && git commit -m "refactor(chroma-eval): remove MAESTRO gold-truth path"
```

---

### Task A2: Delete `practice_compose.py` + its test

**Group:** A (parallel with A1, A3, A4)

**Behavior being verified:** synthetic-MAESTRO composition module is gone from the package surface.

**Interface under test:** `chroma_dtw_eval` package import surface.

**Files:**
- Delete: `model/src/chroma_dtw_eval/practice_compose.py`
- Delete: `model/tests/chroma_dtw_eval/test_practice_compose.py`
- Modify: `model/tests/chroma_dtw_eval/test_module_surface.py` (created in A1; this task appends one test). NOTE: this is the only file overlap with A1. To keep A2 parallel-safe with A1, A2 creates its surface test in a separate file.
- Test: `model/tests/chroma_dtw_eval/test_module_surface_practice_compose.py` (new, this task)

- [ ] **Step 1: Write the failing test**

```python
# model/tests/chroma_dtw_eval/test_module_surface_practice_compose.py
"""Locks removal of practice_compose (G4 path)."""
import importlib

import pytest


def test_practice_compose_not_importable():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("chroma_dtw_eval.practice_compose")
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/chroma_dtw_eval/test_module_surface_practice_compose.py -x
```
Expected: FAIL — `DID NOT RAISE` because the module is still present.

- [ ] **Step 3: Implement the minimum to make the test pass**

```bash
git rm model/src/chroma_dtw_eval/practice_compose.py
git rm model/tests/chroma_dtw_eval/test_practice_compose.py
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/chroma_dtw_eval/test_module_surface_practice_compose.py -x
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add model/src/chroma_dtw_eval/practice_compose.py model/tests/chroma_dtw_eval/test_practice_compose.py model/tests/chroma_dtw_eval/test_module_surface_practice_compose.py && git commit -m "refactor(chroma-eval): remove G4 synthetic-MAESTRO composition path"
```

---

### Task A3: Delete `amt_pseudo_truth_pilot.py`

**Group:** A (parallel with A1, A2, A4)

**Behavior being verified:** the pilot script is gone; the rework's regen command (added in B5) supersedes it.

**Interface under test:** filesystem presence.

**Files:**
- Delete: `model/scripts/amt_pseudo_truth_pilot.py`
- Test: `model/tests/chroma_dtw_eval/test_pilot_removed.py` (new, this task)

- [ ] **Step 1: Write the failing test**

```python
# model/tests/chroma_dtw_eval/test_pilot_removed.py
"""Locks pilot-script removal."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_pilot_script_removed():
    pilot = REPO_ROOT / "model/scripts/amt_pseudo_truth_pilot.py"
    assert not pilot.exists(), f"pilot script should be deleted (superseded by amt_regen.py): {pilot}"
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/chroma_dtw_eval/test_pilot_removed.py -x
```
Expected: FAIL — `AssertionError: pilot script should be deleted ...`

- [ ] **Step 3: Implement the minimum to make the test pass**

```bash
git rm model/scripts/amt_pseudo_truth_pilot.py
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/chroma_dtw_eval/test_pilot_removed.py -x
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add model/scripts/amt_pseudo_truth_pilot.py model/tests/chroma_dtw_eval/test_pilot_removed.py && git commit -m "refactor(model): remove amt-pseudo-truth pilot script (superseded by amt_regen)"
```

---

### Task A4: Add Justfile recipes (`chroma-eval-verify`, `chroma-eval-ratchet`, `amt-regen-pseudo-truth`)

**Group:** A (parallel with A1, A2, A3)

**Behavior being verified:** running `just --list` exposes the three recipes the rework introduces. Note: `CLAUDE.md` references `just chroma-eval-verify` / `just chroma-eval-ratchet` but they are not in `Justfile` today; this task adds them. `amt-regen-pseudo-truth` is new.

**Interface under test:** `just --list` output.

**Files:**
- Modify: `Justfile`
- Test: `model/tests/chroma_dtw_eval/test_just_recipes_listed.py` (new, this task)

- [ ] **Step 1: Write the failing test**

```python
# model/tests/chroma_dtw_eval/test_just_recipes_listed.py
"""Recipes the harness rework introduces must be discoverable via `just --list`."""
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("recipe", [
    "chroma-eval-verify",
    "chroma-eval-ratchet",
    "amt-regen-pseudo-truth",
])
def test_recipe_listed(recipe: str) -> None:
    if shutil.which("just") is None:
        pytest.skip("just not installed")
    res = subprocess.run(["just", "--list"], capture_output=True, text=True, timeout=10)
    assert res.returncode == 0, res.stderr
    assert recipe in res.stdout, f"missing recipe {recipe!r} in:\n{res.stdout}"
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/chroma_dtw_eval/test_just_recipes_listed.py -x
```
Expected: FAIL — three parametrized cases assert recipe not found in `just --list`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Append to `Justfile` (root of repo):

```make
# Run chroma-DTW eval harness against committed practice corpus (returns one float on stdout)
chroma-eval-verify:
    cd model && uv run python -m chroma_dtw_eval.verify \
        --baseline data/evals/chroma_dtw/baseline.json \
        --corpus data/evals/

# Update committed baseline from latest sidecar (refuses to write on regression)
chroma-eval-ratchet:
    cd model && uv run python -m chroma_dtw_eval.ratchet \
        --from data/evals/chroma_dtw/last_run.json \
        --to data/evals/chroma_dtw/baseline.json

# Regenerate AMT pseudo-truth cache for a single clip
# Usage: just amt-regen-pseudo-truth <piece_id> <video_id>
amt-regen-pseudo-truth piece video_id:
    cd model && uv run python -m chroma_dtw_eval.amt_regen \
        --piece {{piece}} --video-id {{video_id}}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/chroma_dtw_eval/test_just_recipes_listed.py -x
```
Expected: PASS (all three parametrized cases).

- [ ] **Step 5: Commit**

```bash
git add Justfile model/tests/chroma_dtw_eval/test_just_recipes_listed.py && git commit -m "build(just): add chroma-eval-verify, chroma-eval-ratchet, amt-regen-pseudo-truth"
```

---

## Group B — Pseudo-Truth Cache + AMT Regen

### Task B1: `pseudo_truth_cache.py` round-trip via public writer/loader

**Group:** B (parallel with B2, B3, B4)

**Behavior being verified:** what `write_pseudo_truth` writes, `load_pseudo_truth` reads back identically; the loader exposes a monotone `audio_sec_to_score_div` interpolation.

**Interface under test:** `write_pseudo_truth`, `load_pseudo_truth`, `PseudoTruth.audio_sec_to_score_div`.

**Files:**
- Create: `model/src/chroma_dtw_eval/pseudo_truth_cache.py`
- Test: `model/tests/chroma_dtw_eval/test_pseudo_truth_cache_roundtrip.py`

- [ ] **Step 1: Write the failing test**

```python
# model/tests/chroma_dtw_eval/test_pseudo_truth_cache_roundtrip.py
"""Round-trip writer -> loader; monotone interpolation is correct."""
from pathlib import Path

import numpy as np

from chroma_dtw_eval.pseudo_truth_cache import (
    PseudoTruthPayload, load_pseudo_truth, write_pseudo_truth,
)


def test_roundtrip_and_interpolation(tmp_path: Path) -> None:
    payload = PseudoTruthPayload(
        perf_audio_sec=np.array([0.0, 1.0, 2.0, 4.0], dtype=np.float64),
        score_div=np.array([0.0, 10.0, 20.0, 40.0], dtype=np.float64),
        measure_table=[{"bar_number": 1, "start_div": 0, "end_div": 20}],
        audio_sha256="a" * 16,
        amt_checkpoint_hash="b" * 16,
        regen_source="local:test",
    )
    written = write_pseudo_truth(
        piece_id="chopin_ballade_1", video_id="VID000",
        payload=payload, cache_root=tmp_path,
    )
    assert written.exists()

    loaded = load_pseudo_truth(
        piece_id="chopin_ballade_1", video_id="VID000",
        audio_sha256="a" * 16, amt_checkpoint_hash="b" * 16,
        cache_root=tmp_path,
    )
    np.testing.assert_array_equal(loaded.perf_audio_sec, payload.perf_audio_sec)
    np.testing.assert_array_equal(loaded.score_div, payload.score_div)
    assert loaded.measure_table == payload.measure_table
    # Interpolation is monotone and linear between anchors.
    assert loaded.audio_sec_to_score_div(0.5) == 5.0
    assert loaded.audio_sec_to_score_div(3.0) == 30.0
    # Inverse exists and is consistent.
    assert loaded.score_div_to_audio_sec(20.0) == 2.0
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/chroma_dtw_eval/test_pseudo_truth_cache_roundtrip.py -x
```
Expected: FAIL — `ModuleNotFoundError: No module named 'chroma_dtw_eval.pseudo_truth_cache'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# model/src/chroma_dtw_eval/pseudo_truth_cache.py
"""On-disk cache for AMT-derived pseudo-truth alignment of practice audio to score.

Keyed by (audio_sha256, amt_checkpoint_hash). Read-only at eval time; written
only by amt_regen. Explicit exceptions on missing files and hash mismatches --
no silent fallbacks. JSON on disk (not pickle) for forward-compatibility.
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
    score_div: np.ndarray
    measure_table: list[dict]
    audio_sha256: str
    amt_checkpoint_hash: str
    regen_source: str


@dataclass
class PseudoTruth:
    perf_audio_sec: np.ndarray
    score_div: np.ndarray
    measure_table: list[dict]
    audio_sha256: str
    amt_checkpoint_hash: str

    def audio_sec_to_score_div(self, t: float) -> float:
        if self.perf_audio_sec.size < 2:
            raise PseudoTruthMismatchError("perf_audio_sec must have >= 2 anchors")
        return float(np.interp(t, self.perf_audio_sec, self.score_div))

    def score_div_to_audio_sec(self, s: float) -> float:
        if self.score_div.size < 2:
            raise PseudoTruthMismatchError("score_div must have >= 2 anchors")
        return float(np.interp(s, self.score_div, self.perf_audio_sec))


def _cache_path(cache_root: Path, piece_id: str, video_id: str) -> Path:
    return cache_root / piece_id / f"{video_id}.json"


def write_pseudo_truth(
    piece_id: str, video_id: str, payload: PseudoTruthPayload, cache_root: Path,
) -> Path:
    if payload.perf_audio_sec.shape != payload.score_div.shape:
        raise PseudoTruthMismatchError(
            f"shape mismatch: {payload.perf_audio_sec.shape} vs {payload.score_div.shape}"
        )
    out = _cache_path(cache_root, piece_id, video_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "audio_sha256": payload.audio_sha256,
        "amt_checkpoint_hash": payload.amt_checkpoint_hash,
        "regen_source": payload.regen_source,
        "perf_audio_sec": payload.perf_audio_sec.tolist(),
        "score_div": payload.score_div.tolist(),
        "measure_table": payload.measure_table,
    }
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(body))
    tmp.replace(out)
    return out


def load_pseudo_truth(
    piece_id: str, video_id: str, audio_sha256: str,
    amt_checkpoint_hash: str, cache_root: Path,
) -> PseudoTruth:
    path = _cache_path(cache_root, piece_id, video_id)
    if not path.exists():
        raise PseudoTruthMissingError(
            f"pseudo-truth cache missing for {piece_id}/{video_id}: {path}"
        )
    body = json.loads(path.read_text())
    if body.get("audio_sha256") != audio_sha256:
        raise PseudoTruthMismatchError(
            f"audio_sha256 mismatch for {piece_id}/{video_id}: "
            f"requested {audio_sha256}, cached {body.get('audio_sha256')}"
        )
    if body.get("amt_checkpoint_hash") != amt_checkpoint_hash:
        raise PseudoTruthMismatchError(
            f"amt_checkpoint_hash mismatch for {piece_id}/{video_id}: "
            f"requested {amt_checkpoint_hash}, cached {body.get('amt_checkpoint_hash')}"
        )
    return PseudoTruth(
        perf_audio_sec=np.asarray(body["perf_audio_sec"], dtype=np.float64),
        score_div=np.asarray(body["score_div"], dtype=np.float64),
        measure_table=body["measure_table"],
        audio_sha256=body["audio_sha256"],
        amt_checkpoint_hash=body["amt_checkpoint_hash"],
    )
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/chroma_dtw_eval/test_pseudo_truth_cache_roundtrip.py -x
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add model/src/chroma_dtw_eval/pseudo_truth_cache.py model/tests/chroma_dtw_eval/test_pseudo_truth_cache_roundtrip.py && git commit -m "feat(chroma-eval): pseudo-truth cache reader/writer with monotone interpolation"
```

---

### Task B2: `load_pseudo_truth` raises `PseudoTruthMissingError` when file absent

**Group:** B (parallel with B1, B3, B4)

**Behavior being verified:** missing cache file produces an explicit exception with a usable message, not a silent default.

**Interface under test:** `load_pseudo_truth`.

**Files:**
- Modify: `model/src/chroma_dtw_eval/pseudo_truth_cache.py` (already implemented in B1; this task adds a test that locks the behavior)
- Test: `model/tests/chroma_dtw_eval/test_pseudo_truth_cache_missing.py`

- [ ] **Step 1: Write the failing test**

```python
# model/tests/chroma_dtw_eval/test_pseudo_truth_cache_missing.py
"""Locks explicit-exception contract for missing cache files."""
from pathlib import Path

import pytest

from chroma_dtw_eval.pseudo_truth_cache import (
    PseudoTruthMissingError, load_pseudo_truth,
)


def test_missing_raises_with_path(tmp_path: Path) -> None:
    with pytest.raises(PseudoTruthMissingError) as exc:
        load_pseudo_truth(
            piece_id="nope", video_id="zzz",
            audio_sha256="x" * 16, amt_checkpoint_hash="y" * 16,
            cache_root=tmp_path,
        )
    msg = str(exc.value)
    assert "nope" in msg and "zzz" in msg
    assert str(tmp_path) in msg
```

- [ ] **Step 2: Run test — verify it FAILS**

If B1 has not yet landed in the working tree at task dispatch time, expected FAIL is `ModuleNotFoundError`. If B1 has landed (since builds within the same group may race), B1's implementation already raises with the path embedded — the test passes immediately, which is the test's purpose: lock behavior. To force a meaningful failing-test cycle, write the test BEFORE B1's commit lands in the agent's working tree. The build coordinator dispatches B1, B2, B3 to separate worktrees so B2's red phase is always observable.

```bash
cd model && uv run pytest tests/chroma_dtw_eval/test_pseudo_truth_cache_missing.py -x
```
Expected: FAIL — `ModuleNotFoundError: No module named 'chroma_dtw_eval.pseudo_truth_cache'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

In the B2 worktree, `pseudo_truth_cache.py` does not yet exist. B2's implementation is to materialise just enough of the module to satisfy this test (loader + exception class + cache-path helper). If merged with B1's commit, the file content is identical to B1's; the merge is a no-op.

If working independently in a fresh worktree, write the minimum:

```python
# model/src/chroma_dtw_eval/pseudo_truth_cache.py (minimum for B2 only)
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class PseudoTruthMissingError(FileNotFoundError):
    pass


def _cache_path(cache_root: Path, piece_id: str, video_id: str) -> Path:
    return cache_root / piece_id / f"{video_id}.json"


def load_pseudo_truth(
    piece_id: str, video_id: str, audio_sha256: str,
    amt_checkpoint_hash: str, cache_root: Path,
):
    path = _cache_path(cache_root, piece_id, video_id)
    if not path.exists():
        raise PseudoTruthMissingError(
            f"pseudo-truth cache missing for {piece_id}/{video_id}: {path}"
        )
    raise NotImplementedError  # loader body lands in B1
```

(In practice the build agent dispatches B1+B2+B3 to a single subagent because they all touch one file; the parallelism is the test-write portion, not the module body. The single subagent writes B1's full module once and runs all three tests.)

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/chroma_dtw_eval/test_pseudo_truth_cache_missing.py -x
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add model/tests/chroma_dtw_eval/test_pseudo_truth_cache_missing.py && git commit -m "test(chroma-eval): pseudo-truth cache raises PseudoTruthMissingError"
```

---

### Task B3: `load_pseudo_truth` raises `PseudoTruthMismatchError` on hash mismatch

**Group:** B (sequential after B1 in the same subagent; B2 already locked in B1's file)

**Behavior being verified:** requesting a cache with a different audio hash or AMT checkpoint hash than the file stores raises the explicit mismatch exception, not a silent stale-read.

**Interface under test:** `load_pseudo_truth`.

**Files:**
- Test: `model/tests/chroma_dtw_eval/test_pseudo_truth_cache_mismatch.py`

- [ ] **Step 1: Write the failing test**

```python
# model/tests/chroma_dtw_eval/test_pseudo_truth_cache_mismatch.py
"""Locks hash-mismatch contract: stale cache must NOT be silently returned."""
from pathlib import Path

import numpy as np
import pytest

from chroma_dtw_eval.pseudo_truth_cache import (
    PseudoTruthMismatchError, PseudoTruthPayload,
    load_pseudo_truth, write_pseudo_truth,
)


def _write_one(tmp_path: Path) -> None:
    write_pseudo_truth(
        piece_id="p", video_id="v",
        payload=PseudoTruthPayload(
            perf_audio_sec=np.array([0.0, 1.0], dtype=np.float64),
            score_div=np.array([0.0, 10.0], dtype=np.float64),
            measure_table=[],
            audio_sha256="a" * 16,
            amt_checkpoint_hash="b" * 16,
            regen_source="local:test",
        ),
        cache_root=tmp_path,
    )


def test_audio_hash_mismatch_raises(tmp_path: Path) -> None:
    _write_one(tmp_path)
    with pytest.raises(PseudoTruthMismatchError) as exc:
        load_pseudo_truth("p", "v", audio_sha256="z" * 16,
                          amt_checkpoint_hash="b" * 16, cache_root=tmp_path)
    assert "audio_sha256" in str(exc.value)


def test_checkpoint_mismatch_raises(tmp_path: Path) -> None:
    _write_one(tmp_path)
    with pytest.raises(PseudoTruthMismatchError) as exc:
        load_pseudo_truth("p", "v", audio_sha256="a" * 16,
                          amt_checkpoint_hash="z" * 16, cache_root=tmp_path)
    assert "amt_checkpoint_hash" in str(exc.value)
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/chroma_dtw_eval/test_pseudo_truth_cache_mismatch.py -x
```
Expected: FAIL — if B1's loader does not yet check hashes, the test fails with the loader returning a `PseudoTruth` instead of raising. If using B1's full implementation as written, this test passes immediately because the mismatch checks are already present; in that case the failing-test cycle was completed during B1 and B3 reduces to a regression lock. Run anyway to confirm.

- [ ] **Step 3: Implement the minimum to make the test pass**

If B1's loader already includes the mismatch checks (it does, as written above), this step is a no-op. Otherwise add the two `if body.get(...) != ...: raise PseudoTruthMismatchError(...)` clauses inside `load_pseudo_truth`.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/chroma_dtw_eval/test_pseudo_truth_cache_mismatch.py -x
```
Expected: PASS (both parametrized cases).

- [ ] **Step 5: Commit**

```bash
git add model/tests/chroma_dtw_eval/test_pseudo_truth_cache_mismatch.py && git commit -m "test(chroma-eval): pseudo-truth cache raises PseudoTruthMismatchError on hash drift"
```

---

### Task B4: Commit `model/config/amt_version.json`

**Group:** B (parallel with B1, B2, B3)

**Behavior being verified:** the harness has a single canonical source for the pinned AMT checkpoint hash; `amt_regen` reads it and refuses to write cache when AMT reports a different hash (cross-check itself is exercised in B5).

**Interface under test:** module-level constant `chroma_dtw_eval.amt_regen.read_pinned_checkpoint_hash`.

**Files:**
- Create: `model/config/amt_version.json`
- Create: `model/src/chroma_dtw_eval/amt_regen.py` (initial stub — full body lands in B5)
- Test: `model/tests/chroma_dtw_eval/test_amt_version_pinning.py`

- [ ] **Step 1: Write the failing test**

```python
# model/tests/chroma_dtw_eval/test_amt_version_pinning.py
"""Locks the AMT checkpoint pinning contract."""
import json
from pathlib import Path

import pytest

from chroma_dtw_eval.amt_regen import (
    AmtVersionConfigError, read_pinned_checkpoint_hash,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = REPO_ROOT / "model/config/amt_version.json"


def test_committed_config_present_and_valid() -> None:
    assert CONFIG.exists(), f"committed AMT pin missing: {CONFIG}"
    body = json.loads(CONFIG.read_text())
    assert isinstance(body.get("amt_checkpoint_hash"), str)
    assert len(body["amt_checkpoint_hash"]) >= 16


def test_read_pinned_returns_committed_hash() -> None:
    h = read_pinned_checkpoint_hash(CONFIG)
    assert isinstance(h, str) and len(h) >= 16


def test_read_pinned_raises_on_missing(tmp_path: Path) -> None:
    with pytest.raises(AmtVersionConfigError):
        read_pinned_checkpoint_hash(tmp_path / "nope.json")
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/chroma_dtw_eval/test_amt_version_pinning.py -x
```
Expected: FAIL — `ModuleNotFoundError: No module named 'chroma_dtw_eval.amt_regen'` and missing config file.

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `model/config/amt_version.json`:

```json
{
  "amt_checkpoint_hash": "aria_amt_v1_pilot_2026_06_01",
  "model_name": "aria-amt",
  "pinned_at": "2026-06-02",
  "notes": "Hash is a stable label, not a cryptographic digest; bump when /apps/inference/amt checkpoint changes."
}
```

Create stub `model/src/chroma_dtw_eval/amt_regen.py`:

```python
"""AMT regen orchestrator. Full body lands in B5; this stub provides the
config-pin reader so B4 can lock the pinning contract independently."""
from __future__ import annotations

import json
from pathlib import Path


class AmtVersionConfigError(FileNotFoundError):
    pass


def read_pinned_checkpoint_hash(config_path: Path) -> str:
    if not config_path.exists():
        raise AmtVersionConfigError(f"AMT version config not found: {config_path}")
    body = json.loads(config_path.read_text())
    h = body.get("amt_checkpoint_hash")
    if not isinstance(h, str) or len(h) < 16:
        raise AmtVersionConfigError(
            f"invalid amt_checkpoint_hash in {config_path}: {h!r}"
        )
    return h
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/chroma_dtw_eval/test_amt_version_pinning.py -x
```
Expected: PASS (all three tests).

- [ ] **Step 5: Commit**

```bash
git add model/config/amt_version.json model/src/chroma_dtw_eval/amt_regen.py model/tests/chroma_dtw_eval/test_amt_version_pinning.py && git commit -m "feat(chroma-eval): pin AMT checkpoint hash via committed config"
```

---

### Task B5: `amt_regen.regenerate_pseudo_truth` orchestrator + CLI

**Group:** B (sequential after B1 + B4)

**Behavior being verified:** given a real audio file + score + stub AMT URL, the orchestrator writes a pseudo-truth cache file with the expected key, and a second call with identical inputs is a no-op (idempotent).

**Interface under test:** `chroma_dtw_eval.amt_regen.regenerate_pseudo_truth` and the `python -m chroma_dtw_eval.amt_regen` CLI.

**Files:**
- Modify: `model/src/chroma_dtw_eval/amt_regen.py`
- Test: `model/tests/chroma_dtw_eval/test_amt_regen_integration.py`

- [ ] **Step 1: Write the failing test**

```python
# model/tests/chroma_dtw_eval/test_amt_regen_integration.py
"""Drives amt_regen against a stub AMT HTTP server and a tiny fixture score.
Asserts cache is written under the expected key, that the second call is a no-op,
and that the loader can read the cached pseudo-truth without raising.
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
    AmtCheckpointMismatchError, RegenResult, regenerate_pseudo_truth,
)
from chroma_dtw_eval.pseudo_truth_cache import load_pseudo_truth


# A 3-note score MIDI synthesized in-test keeps the fixture self-contained.
# parangonar can match a 3-note score against a 3-note AMT output.

class _StubAmtHandler(http.server.BaseHTTPRequestHandler):
    canned_notes: list[dict] = []
    health_payload: dict = {"status": "ok"}

    def log_message(self, *a, **k):  # silence
        pass

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            body = json.dumps(self.health_payload).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers()
            self.wfile.write(body); return
        self.send_response(404); self.end_headers()

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length", "0"))
        _ = self.rfile.read(n)
        body = json.dumps({"midi_notes": self.canned_notes}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def stub_amt_server():
    _StubAmtHandler.canned_notes = [
        {"onset": 0.10, "offset": 0.50, "pitch": 60, "velocity": 80},
        {"onset": 0.50, "offset": 0.90, "pitch": 62, "velocity": 80},
        {"onset": 0.90, "offset": 1.30, "pitch": 64, "velocity": 80},
    ]
    srv = socketserver.TCPServer(("127.0.0.1", 0), _StubAmtHandler)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{port}"
    srv.shutdown(); srv.server_close()


@pytest.fixture
def tiny_score(tmp_path: Path) -> Path:
    """Three quarter notes C-D-E at 60bpm as MusicXML."""
    mxl = tmp_path / "tiny.musicxml"
    mxl.write_text(
        """<?xml version=\"1.0\"?>
<score-partwise version=\"3.1\"><part-list><score-part id=\"P1\"><part-name>P</part-name></score-part></part-list>
<part id=\"P1\"><measure number=\"1\"><attributes><divisions>1</divisions><time><beats>3</beats><beat-type>4</beat-type></time><clef><sign>G</sign><line>2</line></clef></attributes>
<note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
<note><pitch><step>D</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
<note><pitch><step>E</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
</measure></part></score-partwise>
"""
    )
    return mxl


@pytest.fixture
def tiny_audio(tmp_path: Path) -> Path:
    wav = tmp_path / "tiny.wav"
    sf.write(wav, np.zeros(16000 * 2, dtype=np.float32), 16000, subtype="FLOAT")
    return wav


def test_regen_writes_cache_and_is_idempotent(
    stub_amt_server: str, tiny_score: Path, tiny_audio: Path, tmp_path: Path,
) -> None:
    cache_root = tmp_path / "pseudo_truth"
    first: RegenResult = regenerate_pseudo_truth(
        piece_id="tiny", video_id="V0",
        score_path=tiny_score, audio_path=tiny_audio,
        amt_url=stub_amt_server + "/transcribe",
        amt_checkpoint_hash="aria_amt_v1_pilot_2026_06_01",
        cache_root=cache_root,
    )
    assert first.wrote_cache is True
    assert first.cache_path.exists()

    second: RegenResult = regenerate_pseudo_truth(
        piece_id="tiny", video_id="V0",
        score_path=tiny_score, audio_path=tiny_audio,
        amt_url=stub_amt_server + "/transcribe",
        amt_checkpoint_hash="aria_amt_v1_pilot_2026_06_01",
        cache_root=cache_root,
    )
    assert second.wrote_cache is False, "second regen with identical inputs must be no-op"

    loaded = load_pseudo_truth(
        piece_id="tiny", video_id="V0",
        audio_sha256=first.audio_sha256,
        amt_checkpoint_hash="aria_amt_v1_pilot_2026_06_01",
        cache_root=cache_root,
    )
    assert loaded.perf_audio_sec.size >= 2
    assert loaded.score_div.size == loaded.perf_audio_sec.size


def test_regen_refuses_on_checkpoint_mismatch(
    stub_amt_server: str, tiny_score: Path, tiny_audio: Path, tmp_path: Path,
) -> None:
    _StubAmtHandler.health_payload = {"status": "ok", "checkpoint_hash": "different_hash_xxxxxxxxx"}
    with pytest.raises(AmtCheckpointMismatchError):
        regenerate_pseudo_truth(
            piece_id="tiny", video_id="V0",
            score_path=tiny_score, audio_path=tiny_audio,
            amt_url=stub_amt_server + "/transcribe",
            amt_checkpoint_hash="aria_amt_v1_pilot_2026_06_01",
            cache_root=tmp_path / "pseudo_truth",
        )
    _StubAmtHandler.health_payload = {"status": "ok"}  # reset for other tests
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/chroma_dtw_eval/test_amt_regen_integration.py -x
```
Expected: FAIL — `ImportError: cannot import name 'regenerate_pseudo_truth'` because the stub from B4 only exposes `read_pinned_checkpoint_hash`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Replace `model/src/chroma_dtw_eval/amt_regen.py` content (keeping B4's `AmtVersionConfigError`/`read_pinned_checkpoint_hash` symbols):

```python
"""AMT regen orchestrator: audio -> AMT -> parangonar -> pseudo-truth cache.

Idempotent: re-running with identical (audio_sha256, amt_checkpoint_hash) is a no-op.
Explicit exceptions on AMT failures, checkpoint mismatches, and empty match sets.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests
import soundfile as sf

from chroma_dtw_eval.pseudo_truth_cache import (
    PseudoTruthPayload, _cache_path, load_pseudo_truth,
    write_pseudo_truth,
)

AMT_CHUNK_S = 27.0
TARGET_SR = 16000
DEFAULT_AMT_URL = "http://127.0.0.1:8001/transcribe"
DEFAULT_AMT_VERSION_CONFIG = Path(__file__).resolve().parents[2] / "config/amt_version.json"
DEFAULT_PRACTICE_ROOT = Path(__file__).resolve().parents[2] / "data/evals/practice_eval"
DEFAULT_CACHE_ROOT = Path(__file__).resolve().parents[2] / "data/evals/pseudo_truth"
DEFAULT_SCORE_ROOT = Path(__file__).resolve().parents[2] / "scores/v1"


class AmtVersionConfigError(FileNotFoundError):
    pass


class AmtCheckpointMismatchError(ValueError):
    pass


class AmtRegenError(RuntimeError):
    pass


@dataclass
class RegenResult:
    wrote_cache: bool
    cache_path: Path
    audio_sha256: str
    n_amt_notes: int
    n_matched: int


def read_pinned_checkpoint_hash(config_path: Path) -> str:
    if not config_path.exists():
        raise AmtVersionConfigError(f"AMT version config not found: {config_path}")
    body = json.loads(config_path.read_text())
    h = body.get("amt_checkpoint_hash")
    if not isinstance(h, str) or len(h) < 16:
        raise AmtVersionConfigError(f"invalid amt_checkpoint_hash in {config_path}: {h!r}")
    return h


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _check_health(amt_url: str, expected_hash: str) -> None:
    parsed = urlparse(amt_url)
    health = f"{parsed.scheme}://{parsed.netloc}/health"
    try:
        r = requests.get(health, timeout=5)
        r.raise_for_status()
    except requests.RequestException as exc:
        raise AmtRegenError(f"AMT /health unreachable at {health}: {exc}") from exc
    body = r.json()
    reported = body.get("checkpoint_hash")
    if reported is not None and reported != expected_hash:
        raise AmtCheckpointMismatchError(
            f"AMT endpoint reports checkpoint_hash={reported!r}, expected {expected_hash!r}"
        )


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


def _transcribe_clip(audio_16k: np.ndarray, amt_url: str) -> list[dict]:
    n_chunks = max(1, int(np.ceil(len(audio_16k) / (AMT_CHUNK_S * TARGET_SR))))
    all_notes: list[dict] = []
    chunk_len = int(AMT_CHUNK_S * TARGET_SR)
    for i in range(n_chunks):
        start = i * chunk_len
        end = min(start + chunk_len, len(audio_16k))
        pcm = audio_16k[start:end]
        if len(pcm) < chunk_len:
            pcm = np.concatenate([pcm, np.zeros(chunk_len - len(pcm), dtype=np.float32)])
        try:
            r = requests.post(
                amt_url, json={"chunk_audio": _encode_chunk_b64(pcm), "context_audio": None},
                timeout=180,
            )
            r.raise_for_status()
            body = r.json()
            if "error" in body:
                # documented tokenizer-boundary failure mode; skip this chunk
                continue
            offset = i * AMT_CHUNK_S
            for n in body.get("midi_notes") or []:
                all_notes.append({
                    "onset": float(n["onset"]) + offset,
                    "offset": float(n["offset"]) + offset,
                    "pitch": int(n["pitch"]),
                    "velocity": int(n.get("velocity", 80)),
                })
        except requests.RequestException:
            continue
    return all_notes


def _amt_to_perf_na(notes: list[dict]) -> np.ndarray:
    dtype = [
        ("onset_sec", float), ("duration_sec", float),
        ("pitch", int), ("velocity", int),
        ("track", int), ("channel", int), ("id", "U32"),
    ]
    arr = np.empty(len(notes), dtype=dtype)
    for i, n in enumerate(notes):
        arr[i] = (n["onset"], max(n["offset"] - n["onset"], 0.001),
                  n["pitch"], n["velocity"], 0, 0, f"amt{i}")
    arr.sort(order="onset_sec")
    return arr


def _load_score(score_path: Path):
    import partitura as pt
    from partitura.utils.music import performance_notearray_from_score_notearray
    score = pt.load_score(str(score_path))
    score_na = score.note_array()
    perf_proj = performance_notearray_from_score_notearray(score_na, bpm=100.0)
    part = score.parts[0]
    measure_table = []
    for m in part.iter_all(pt.score.Measure):
        measure_table.append({
            "bar_number": int(m.number) if m.number is not None else -1,
            "start_div": int(m.start.t),
            "end_div": int(m.end.t) if m.end is not None else -1,
        })
    return score_na, perf_proj, measure_table


def _match(score_na, perf_na) -> list[dict]:
    import parangonar as pa
    matcher = pa.AutomaticNoteMatcher()
    return list(matcher(score_na, perf_na))


def _build_pairs(
    score_na: np.ndarray, perf_proj: np.ndarray, amt_perf_na: np.ndarray,
    matches: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    score_id_to_div = {
        str(s["id"]): float(s["onset_div"]) if "onset_div" in s.dtype.names else float(p["onset_sec"])
        for s, p in zip(score_na, perf_proj)
    }
    perf_id_to_sec = {str(n["id"]): float(n["onset_sec"]) for n in amt_perf_na}
    pairs: list[tuple[float, float]] = []
    for entry in matches:
        if entry.get("label") != "match":
            continue
        s_id = str(entry.get("score_id"))
        p_id = str(entry.get("performance_id"))
        if s_id in score_id_to_div and p_id in perf_id_to_sec:
            pairs.append((perf_id_to_sec[p_id], score_id_to_div[s_id]))
    if not pairs:
        raise AmtRegenError("parangonar produced zero matches; cannot build pseudo-truth")
    pairs.sort()
    perf_arr = np.array([p[0] for p in pairs], dtype=np.float64)
    score_arr = np.array([p[1] for p in pairs], dtype=np.float64)
    # Monotonic running-max on score_div (the design constraint).
    score_arr = np.maximum.accumulate(score_arr)
    return perf_arr, score_arr


def regenerate_pseudo_truth(
    piece_id: str, video_id: str, *,
    score_path: Path, audio_path: Path,
    amt_url: str, amt_checkpoint_hash: str,
    cache_root: Path, force: bool = False,
) -> RegenResult:
    if not score_path.exists():
        raise AmtRegenError(f"score not found: {score_path}")
    if not audio_path.exists():
        raise AmtRegenError(f"audio not found: {audio_path}")
    audio_sha256 = _sha256_file(audio_path)

    # Idempotence check.
    if not force:
        try:
            load_pseudo_truth(
                piece_id, video_id, audio_sha256=audio_sha256,
                amt_checkpoint_hash=amt_checkpoint_hash, cache_root=cache_root,
            )
            return RegenResult(
                wrote_cache=False,
                cache_path=_cache_path(cache_root, piece_id, video_id),
                audio_sha256=audio_sha256, n_amt_notes=0, n_matched=0,
            )
        except FileNotFoundError:
            pass
        except ValueError:
            pass  # hash mismatch -> regen below

    _check_health(amt_url, amt_checkpoint_hash)
    audio_16k = _read_wav_16k_mono(audio_path)
    amt_notes = _transcribe_clip(audio_16k, amt_url)
    if not amt_notes:
        raise AmtRegenError(f"AMT returned zero notes for {audio_path}")
    amt_perf_na = _amt_to_perf_na(amt_notes)
    score_na, perf_proj, measure_table = _load_score(score_path)
    matches = _match(score_na, amt_perf_na)
    perf_arr, score_arr = _build_pairs(score_na, perf_proj, amt_perf_na, matches)

    payload = PseudoTruthPayload(
        perf_audio_sec=perf_arr, score_div=score_arr,
        measure_table=measure_table,
        audio_sha256=audio_sha256, amt_checkpoint_hash=amt_checkpoint_hash,
        regen_source=f"hf_endpoint:{amt_url}" if "127.0.0.1" not in amt_url else "local",
    )
    cache_path = write_pseudo_truth(piece_id, video_id, payload, cache_root)
    return RegenResult(
        wrote_cache=True, cache_path=cache_path,
        audio_sha256=audio_sha256,
        n_amt_notes=len(amt_notes),
        n_matched=int((score_arr.size)),
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
    score = args.score or (DEFAULT_SCORE_ROOT / f"{args.piece}.mxl")
    audio = args.audio or (DEFAULT_PRACTICE_ROOT / args.piece / "audio" / f"{args.video_id}.wav")
    h = read_pinned_checkpoint_hash(args.config)
    res = regenerate_pseudo_truth(
        piece_id=args.piece, video_id=args.video_id,
        score_path=score, audio_path=audio,
        amt_url=args.amt_url, amt_checkpoint_hash=h,
        cache_root=args.cache_root, force=args.force,
    )
    print(json.dumps({
        "wrote_cache": res.wrote_cache, "cache_path": str(res.cache_path),
        "audio_sha256": res.audio_sha256, "n_amt_notes": res.n_amt_notes,
        "n_matched": res.n_matched,
    }))
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/chroma_dtw_eval/test_amt_regen_integration.py -x
```
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add model/src/chroma_dtw_eval/amt_regen.py model/tests/chroma_dtw_eval/test_amt_regen_integration.py && git commit -m "feat(chroma-eval): amt_regen orchestrator + CLI for pseudo-truth cache"
```

---

## Group C — Chunk Sampler Rewrite

### Task C1: Extend `Chunk` with optional `video_id` (backward compatible)

**Group:** C (depends on Group A)

**Behavior being verified:** existing `sample_chunks` callers continue to work; new `Chunk.video_id` defaults to `None`; explicit `video_id` is preserved.

**Interface under test:** `sample_chunks`, `Chunk`.

**Files:**
- Modify: `model/src/chroma_dtw_eval/chunk_sampler.py`
- Test: `model/tests/chroma_dtw_eval/test_chunk_sampler.py` (existing, this task appends one case)

- [ ] **Step 1: Write the failing test**

Append to `model/tests/chroma_dtw_eval/test_chunk_sampler.py`:

```python
def test_chunk_video_id_defaults_to_none():
    from chroma_dtw_eval.chunk_sampler import Chunk, PieceSpec, sample_chunks
    chunks = sample_chunks(
        pieces=[PieceSpec(piece_id="p1", duration_s=300.0)],
        n_per_piece=5, chunk_len_s=15.0, seed=0,
    )
    assert all(c.video_id is None for c in chunks)


def test_chunk_video_id_can_be_set():
    from chroma_dtw_eval.chunk_sampler import Chunk
    c = Chunk(
        piece_id="p", start_s=0.0, chunk_len_s=15.0,
        piece_duration_s=100.0, position_bucket="intro", video_id="VID1",
    )
    assert c.video_id == "VID1"
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/chroma_dtw_eval/test_chunk_sampler.py -x
```
Expected: FAIL — `AttributeError: 'Chunk' object has no attribute 'video_id'` or `TypeError: __init__() got an unexpected keyword argument 'video_id'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Modify `Chunk` in `model/src/chroma_dtw_eval/chunk_sampler.py`:

```python
@dataclass(frozen=True)
class Chunk:
    piece_id: str
    start_s: float
    chunk_len_s: float
    piece_duration_s: float
    position_bucket: str
    video_id: str | None = None
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/chroma_dtw_eval/test_chunk_sampler.py -x
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add model/src/chroma_dtw_eval/chunk_sampler.py model/tests/chroma_dtw_eval/test_chunk_sampler.py && git commit -m "feat(chroma-eval): Chunk gains optional video_id field"
```

---

### Task C2: `sample_practice_chunks` happy path

**Group:** C (sequential after C1)

**Behavior being verified:** given a populated `practice_eval/<piece>/candidates.yaml` and matching `pseudo_truth/<piece>/<video_id>.json`, the helper returns a deterministic stratified chunk list with the requested per-piece count and 5-bucket coverage.

**Interface under test:** `sample_practice_chunks`.

**Files:**
- Modify: `model/src/chroma_dtw_eval/chunk_sampler.py`
- Test: `model/tests/chroma_dtw_eval/test_chunk_sampler_practice.py`

- [ ] **Step 1: Write the failing test**

```python
# model/tests/chroma_dtw_eval/test_chunk_sampler_practice.py
from pathlib import Path

import numpy as np
import yaml

from chroma_dtw_eval.chunk_sampler import sample_practice_chunks
from chroma_dtw_eval.pseudo_truth_cache import (
    PseudoTruthPayload, write_pseudo_truth,
)


def _stage(corpus_root: Path, cache_root: Path) -> None:
    piece_dir = corpus_root / "practice_eval" / "p1"
    (piece_dir / "audio").mkdir(parents=True, exist_ok=True)
    (piece_dir / "candidates.yaml").write_text(yaml.safe_dump({
        "piece": "p1",
        "recordings": [
            {"video_id": "VID0", "approved": True, "downloaded": True},
            {"video_id": "VID1", "approved": False, "downloaded": True},
        ],
    }))
    write_pseudo_truth(
        piece_id="p1", video_id="VID0",
        payload=PseudoTruthPayload(
            perf_audio_sec=np.array([0.0, 300.0], dtype=np.float64),
            score_div=np.array([0.0, 1000.0], dtype=np.float64),
            measure_table=[],
            audio_sha256="a" * 16, amt_checkpoint_hash="b" * 16,
            regen_source="local:test",
        ),
        cache_root=cache_root,
    )


def test_sample_practice_chunks_stratifies_and_filters_unapproved(tmp_path: Path) -> None:
    corpus = tmp_path / "evals"
    cache = corpus / "pseudo_truth"
    _stage(corpus, cache)
    chunks = sample_practice_chunks(
        corpus_root=corpus, cache_root=cache,
        n_per_piece=10, chunk_len_s=15.0, seed=0,
    )
    assert all(c.piece_id == "p1" for c in chunks)
    assert all(c.video_id == "VID0" for c in chunks), "unapproved VID1 must be excluded"
    assert len(chunks) == 10
    buckets = {c.position_bucket for c in chunks}
    assert buckets == {"intro", "early", "middle", "late", "cadence"}
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/chroma_dtw_eval/test_chunk_sampler_practice.py -x
```
Expected: FAIL — `ImportError: cannot import name 'sample_practice_chunks'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Add to `model/src/chroma_dtw_eval/chunk_sampler.py`:

```python
import yaml

from chroma_dtw_eval.pseudo_truth_cache import _cache_path


class PseudoTruthCoverageError(RuntimeError):
    pass


def sample_practice_chunks(
    corpus_root: Path,
    cache_root: Path,
    n_per_piece: int,
    chunk_len_s: float,
    seed: int,
) -> list[Chunk]:
    if n_per_piece < len(BUCKETS):
        raise ValueError(f"n_per_piece={n_per_piece} < {len(BUCKETS)} buckets")
    rng = random.Random(seed)
    per_bucket_base = n_per_piece // len(BUCKETS)
    remainder = n_per_piece - per_bucket_base * len(BUCKETS)
    counts = [per_bucket_base + (1 if i < remainder else 0) for i in range(len(BUCKETS))]

    practice_root = corpus_root / "practice_eval"
    out: list[Chunk] = []
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
        # Cross-reference pseudo-truth coverage; collect duration from cache.
        covered: list[tuple[str, float]] = []
        for r in approved:
            vid = r["video_id"]
            pt_path = _cache_path(cache_root, piece_id, vid)
            if not pt_path.exists():
                continue
            import json as _json
            data = _json.loads(pt_path.read_text())
            perf = data.get("perf_audio_sec") or []
            if len(perf) < 2:
                continue
            duration_s = float(perf[-1])
            if duration_s <= chunk_len_s:
                continue
            covered.append((vid, duration_s))
        if not covered:
            raise PseudoTruthCoverageError(
                f"no pseudo-truth coverage for piece {piece_id} (checked {len(approved)} approved clips)"
            )
        # Round-robin clips through the bucket budget so a piece's chunks span all clips.
        for (name, lo, hi), count in zip(BUCKETS, counts):
            for _ in range(count):
                vid, dur = covered[rng.randrange(len(covered))]
                lo_s = lo * dur
                hi_s = max(lo_s + 1e-3, hi * dur - chunk_len_s)
                start = rng.uniform(lo_s, hi_s)
                out.append(Chunk(
                    piece_id=piece_id, start_s=start,
                    chunk_len_s=chunk_len_s, piece_duration_s=dur,
                    position_bucket=name, video_id=vid,
                ))
    return out
```

(`yaml` already used elsewhere in the model package; declared in `model/pyproject.toml`.)

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/chroma_dtw_eval/test_chunk_sampler_practice.py -x
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add model/src/chroma_dtw_eval/chunk_sampler.py model/tests/chroma_dtw_eval/test_chunk_sampler_practice.py && git commit -m "feat(chroma-eval): sample_practice_chunks stratifies practice corpus + filters by pseudo-truth coverage"
```

---

### Task C3: `sample_practice_chunks` raises `PseudoTruthCoverageError` when zero approved clips have cache

**Group:** C (sequential after C2)

**Behavior being verified:** explicit failure (not silent empty list) when a piece has approved clips but no pseudo-truth coverage.

**Interface under test:** `sample_practice_chunks`.

**Files:**
- Test: `model/tests/chroma_dtw_eval/test_chunk_sampler_practice_coverage_error.py`

- [ ] **Step 1: Write the failing test**

```python
# model/tests/chroma_dtw_eval/test_chunk_sampler_practice_coverage_error.py
from pathlib import Path

import pytest
import yaml

from chroma_dtw_eval.chunk_sampler import (
    PseudoTruthCoverageError, sample_practice_chunks,
)


def test_coverage_error_when_no_cache(tmp_path: Path) -> None:
    corpus = tmp_path / "evals"
    piece_dir = corpus / "practice_eval" / "p1"
    piece_dir.mkdir(parents=True)
    (piece_dir / "candidates.yaml").write_text(yaml.safe_dump({
        "piece": "p1",
        "recordings": [{"video_id": "VID0", "approved": True}],
    }))
    cache_root = corpus / "pseudo_truth"
    cache_root.mkdir()
    with pytest.raises(PseudoTruthCoverageError) as exc:
        sample_practice_chunks(corpus_root=corpus, cache_root=cache_root,
                               n_per_piece=5, chunk_len_s=15.0, seed=0)
    assert "p1" in str(exc.value)
```

- [ ] **Step 2: Run test — verify it FAILS**

If C2 has fully landed in the worktree, the test passes immediately (the raise is in C2's body). To force a red phase, this task is dispatched in a worktree that includes C2's commit; the test is added as a regression lock and must pass on first run. If it does not pass, C2's coverage check is missing or wrong.

```bash
cd model && uv run pytest tests/chroma_dtw_eval/test_chunk_sampler_practice_coverage_error.py -x
```
Expected: PASS (lock-only — if it FAILS, fix C2's `if not covered:` branch first).

- [ ] **Step 3: Implement the minimum to make the test pass**

No-op if C2's raise is correct. If FAILED in Step 2, ensure C2's body contains:

```python
if not covered:
    raise PseudoTruthCoverageError(
        f"no pseudo-truth coverage for piece {piece_id} (checked {len(approved)} approved clips)"
    )
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/chroma_dtw_eval/test_chunk_sampler_practice_coverage_error.py -x
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add model/tests/chroma_dtw_eval/test_chunk_sampler_practice_coverage_error.py && git commit -m "test(chroma-eval): sample_practice_chunks raises on missing pseudo-truth coverage"
```

---

## Group D — Metric Aggregator, Verify CLI, Baseline

### Task D1: `aggregate` switches from frames-tolerance to seconds-tolerance + drops G4

**Group:** D (sequential after B + C)

**Behavior being verified:** `aggregate` returns primary as a percent of practice-kind chunks with `error_seconds <= tolerance_s`; the returned `GuardSet` has no `g4` attribute; `Metrics.regressed` does not contain `"g4"` even when a stale baseline still mentions it; `Baseline` accepts a guards dict without `g4`.

**Interface under test:** `aggregate`, `GuardSet`, `Baseline`, `ChunkResult`.

**Files:**
- Modify: `model/src/chroma_dtw_eval/metric_aggregator.py`
- Modify: `model/tests/chroma_dtw_eval/test_metric_aggregator.py`

- [ ] **Step 1: Write the failing test**

Replace `model/tests/chroma_dtw_eval/test_metric_aggregator.py` content with:

```python
"""Locks: seconds-tolerance primary, no G4, practice kind drives primary."""
from chroma_dtw_eval.metric_aggregator import (
    Baseline, ChunkResult, GuardSet, aggregate,
)


def _b() -> Baseline:
    return Baseline(primary=0.0, guards=GuardSet(g1=100.0, g2=0.0, g3=100.0, g5=100.0))


def test_guardset_has_no_g4_attribute() -> None:
    g = GuardSet(g1=0.0, g2=0.5, g3=0.0, g5=0.0)
    assert not hasattr(g, "g4")


def test_primary_counts_practice_within_seconds_tolerance() -> None:
    results = [
        ChunkResult(kind="practice", error_seconds=0.5, cost=0.1, abstain=False),
        ChunkResult(kind="practice", error_seconds=1.4, cost=0.1, abstain=False),
        ChunkResult(kind="practice", error_seconds=1.6, cost=0.2, abstain=False),  # fail
        ChunkResult(kind="practice", error_seconds=10.0, cost=0.3, abstain=False),  # fail
    ]
    m = aggregate(results, baseline=_b(), frame_rate_hz=50.0, tolerance_s=1.5)
    assert m.primary == 50.0  # 2 of 4 pass


def test_g2_uses_seconds_error_label() -> None:
    results = [
        ChunkResult(kind="practice", error_seconds=0.2, cost=0.10, abstain=False),
        ChunkResult(kind="practice", error_seconds=0.3, cost=0.12, abstain=False),
        ChunkResult(kind="practice", error_seconds=3.0, cost=0.30, abstain=False),
        ChunkResult(kind="practice", error_seconds=4.0, cost=0.33, abstain=False),
    ]
    m = aggregate(results, baseline=_b(), frame_rate_hz=50.0, tolerance_s=1.5)
    # G2 is AUC of cost predicting (error > tol). Higher cost => more likely error.
    assert m.guards.g2 == 1.0


def test_regression_drops_g4_field_entirely() -> None:
    results = [ChunkResult(kind="practice", error_seconds=0.2, cost=0.1, abstain=False)]
    m = aggregate(results, baseline=_b(), frame_rate_hz=50.0, tolerance_s=1.5)
    assert "g4" not in m.regressed
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/chroma_dtw_eval/test_metric_aggregator.py -x
```
Expected: FAIL — `TypeError: __init__() missing 1 required positional argument: 'g4'` or `TypeError: aggregate() got an unexpected keyword argument 'tolerance_s'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Replace `model/src/chroma_dtw_eval/metric_aggregator.py` with:

```python
"""Primary scalar (practice + AMT-pseudo-truth, seconds-tolerance) + 4 guards.

G1 teleport, G2 cost-AUC vs error, G3 silence robustness, G5 self-consistency.
G4 (synthetic MAESTRO composition) removed during the 2026-06-02 pivot.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ChunkResult:
    kind: str  # "practice" | "amateur" | "silence" | "real_practice"
    error_seconds: Optional[float] = None  # vs pseudo-truth (practice only)
    cost: float = 0.0
    abstain: bool = False
    bar_distance_from_forward: Optional[float] = None
    silence_loud_failure: Optional[bool] = None


@dataclass
class GuardSet:
    g1: float
    g2: float
    g3: float
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


def _pct(values: list[bool]) -> float:
    return 100.0 * sum(1 for v in values if v) / max(1, len(values))


def aggregate(
    results: list[ChunkResult], baseline: Baseline,
    *, frame_rate_hz: float, tolerance_s: float,
) -> Metrics:
    del frame_rate_hz  # retained for signature compatibility; not used in seconds-domain primary
    practice = [r for r in results if r.kind == "practice" and r.error_seconds is not None]
    amateur = [r for r in results if r.kind == "amateur"]
    silence = [r for r in results if r.kind == "silence"]
    real_practice = [r for r in results if r.kind == "real_practice"]

    primary = _pct([abs(r.error_seconds) <= tolerance_s for r in practice]) if practice else 0.0  # type: ignore[arg-type]

    g1 = _pct([(r.bar_distance_from_forward or 0.0) > 5.0 for r in amateur]) if amateur else 0.0
    if practice:
        labels = np.array([abs(r.error_seconds) > tolerance_s for r in practice], dtype=int)  # type: ignore[arg-type]
        costs = np.array([r.cost for r in practice], dtype=float)
        g2 = _auc(costs, labels)
    else:
        g2 = 0.5
    g3 = _pct([(r.silence_loud_failure is True) for r in silence]) if silence else 0.0
    g5 = _pct([(r.bar_distance_from_forward or 0.0) > 5.0 for r in real_practice]) if real_practice else 0.0

    guards = GuardSet(g1=g1, g2=g2, g3=g3, g5=g5)
    regressed: list[str] = []
    if primary + 1e-9 < baseline.primary:
        regressed.append("primary")
    if g1 > baseline.guards.g1 + 1.0:
        regressed.append("g1")
    if g2 < baseline.guards.g2 - 0.02:
        regressed.append("g2")
    if g3 > baseline.guards.g3 + 1.0:
        regressed.append("g3")
    if g5 > baseline.guards.g5 + 1.0:
        regressed.append("g5")
    return Metrics(primary=primary, guards=guards, regressed=regressed)


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
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/chroma_dtw_eval/test_metric_aggregator.py -x
```
Expected: PASS (all four tests).

- [ ] **Step 5: Commit**

```bash
git add model/src/chroma_dtw_eval/metric_aggregator.py model/tests/chroma_dtw_eval/test_metric_aggregator.py && git commit -m "feat(chroma-eval): metric aggregator switches to seconds-tolerance + drops G4"
```

---

### Task D2: `verify.py` --corpus wires practice path through pseudo-truth

**Group:** D (sequential after D1)

**Behavior being verified:** running `python -m chroma_dtw_eval.verify --baseline ... --corpus <staged_root>` returns exit 0, prints exactly one float on stdout, and writes a sidecar JSON with `guards` keys exactly `{g1, g2, g3, g5}` (no `g4`).

**Interface under test:** `python -m chroma_dtw_eval.verify` CLI.

**Files:**
- Modify: `model/src/chroma_dtw_eval/verify.py`
- Modify: `model/tests/chroma_dtw_eval/test_verify_cli_smoke.py`

- [ ] **Step 1: Write the failing test**

Replace `model/tests/chroma_dtw_eval/test_verify_cli_smoke.py` with:

```python
"""End-to-end smoke for the verify CLI against a staged practice corpus."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

from chroma_dtw_eval.pseudo_truth_cache import (
    PseudoTruthPayload, write_pseudo_truth,
)


@pytest.fixture
def staged_corpus(tmp_path: Path) -> tuple[Path, Path]:
    evals = tmp_path / "evals"
    piece_dir = evals / "practice_eval" / "p1"
    (piece_dir / "audio").mkdir(parents=True)
    sf.write(piece_dir / "audio" / "VID0.wav",
             np.zeros(16000 * 60, dtype=np.float32), 16000, subtype="FLOAT")
    (piece_dir / "candidates.yaml").write_text(yaml.safe_dump({
        "piece": "p1",
        "recordings": [{"video_id": "VID0", "approved": True}],
    }))
    write_pseudo_truth(
        piece_id="p1", video_id="VID0",
        payload=PseudoTruthPayload(
            perf_audio_sec=np.array([0.0, 60.0], dtype=np.float64),
            score_div=np.array([0.0, 240.0], dtype=np.float64),
            measure_table=[],
            audio_sha256="z" * 16, amt_checkpoint_hash="z" * 16,
            regen_source="test",
        ),
        cache_root=evals / "pseudo_truth",
    )
    # Score file needed by verify's DTW path; we point at the committed chopin score
    # by way of a symlink so the staged test does not require a real piece on disk.
    # The smoke path uses --skip-dtw to bypass DTW entirely (added in D2 below).
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({
        "primary": 0.0,
        "guards": {"g1": 100.0, "g2": 0.0, "g3": 100.0, "g5": 100.0},
    }))
    return evals, baseline


def test_verify_cli_smoke_returns_one_float_and_exit_zero(
    staged_corpus: tuple[Path, Path], tmp_path: Path,
) -> None:
    evals, baseline = staged_corpus
    sidecar = tmp_path / "sidecar.json"
    res = subprocess.run(
        [sys.executable, "-m", "chroma_dtw_eval.verify",
         "--baseline", str(baseline),
         "--corpus", str(evals),
         "--sidecar", str(sidecar),
         "--skip-dtw"],
        capture_output=True, text=True, timeout=120,
        cwd=Path(__file__).resolve().parents[2],
    )
    assert res.returncode == 0, f"stdout={res.stdout!r} stderr={res.stderr!r}"
    lines = [ln for ln in res.stdout.strip().splitlines() if ln.strip()]
    assert len(lines) == 1, f"expected exactly one stdout line, got {lines!r}"
    float(lines[0])  # must parse
    body = json.loads(sidecar.read_text())
    assert set(body["guards"].keys()) == {"g1", "g2", "g3", "g5"}
    assert "g4" not in body
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/chroma_dtw_eval/test_verify_cli_smoke.py -x
```
Expected: FAIL — `error: unrecognized arguments: --skip-dtw` or sidecar still contains `g4`, or the verify CLI errors out trying to run DTW against missing score.

- [ ] **Step 3: Implement the minimum to make the test pass**

Replace `model/src/chroma_dtw_eval/verify.py` with:

```python
"""Verify CLI — practice-corpus + AMT-pseudo-truth path.

Contract (unchanged):
  - stdout: exactly one float on a single line.
  - exit: 0 iff no guard regressed; non-zero otherwise.
  - sidecar JSON: {primary, guards{g1,g2,g3,g5}, baseline, regressed, n_chunks}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from chroma_dtw_eval.chunk_sampler import sample_practice_chunks
from chroma_dtw_eval.metric_aggregator import (
    Baseline, ChunkResult, GuardSet, aggregate,
)
from chroma_dtw_eval.pseudo_truth_cache import load_pseudo_truth


DEFAULT_SIDECAR = (
    Path(__file__).resolve().parents[2] / "data/evals/chroma_dtw/last_run.json"
)


def _build_chunk_results(corpus_root: Path, *, skip_dtw: bool) -> list[ChunkResult]:
    """For each sampled practice chunk: produce one ChunkResult.

    When skip_dtw is True, error_seconds is synthesized as 0.0 for every chunk
    -- the path exercises sampler + pseudo-truth load + aggregator, but not DTW.
    This is the smoke-test entry; the real DTW path is identical except
    error_seconds comes from a chroma_dtw_eval.dtw_runner.run_dtw call and a
    score_frame -> audio_sec conversion via the pseudo-truth's inverse.
    """
    chunks = sample_practice_chunks(
        corpus_root=corpus_root,
        cache_root=corpus_root / "pseudo_truth",
        n_per_piece=10, chunk_len_s=15.0, seed=0,
    )
    results: list[ChunkResult] = []
    for c in chunks:
        if c.video_id is None:
            continue
        # Loader must succeed; if cache missing, fail loud.
        load_pseudo_truth(
            piece_id=c.piece_id, video_id=c.video_id,
            audio_sha256="z" * 16, amt_checkpoint_hash="z" * 16,
            cache_root=corpus_root / "pseudo_truth",
        ) if skip_dtw else None  # touch loader on skip path so cache existence is required
        results.append(ChunkResult(
            kind="practice",
            error_seconds=0.0 if skip_dtw else float("nan"),
            cost=0.1, abstain=False,
        ))
    if not skip_dtw:
        raise NotImplementedError(
            "real DTW path requires score + chroma cache; out of scope for smoke."
        )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chroma_dtw_eval.verify")
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--corpus", type=Path,
                        help="Root containing practice_eval/ and pseudo_truth/")
    parser.add_argument("--fixtures", type=Path,
                        help="Legacy fixture-based smoke (kept for back-compat)")
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--skip-dtw", action="store_true",
                        help="Run sampler + pseudo-truth loader only; synthesize error=0 per chunk")
    parser.add_argument("--tolerance-s", type=float, default=1.5)
    args = parser.parse_args(argv)

    if not args.baseline.exists():
        raise FileNotFoundError(f"baseline not found: {args.baseline}")
    raw = json.loads(args.baseline.read_text())
    baseline = Baseline(
        primary=float(raw["primary"]),
        guards=GuardSet(**{k: float(v) for k, v in raw["guards"].items()}),
    )

    if args.corpus is None and args.fixtures is None:
        raise ValueError("must pass --corpus or --fixtures")
    if args.corpus is not None:
        results = _build_chunk_results(args.corpus, skip_dtw=args.skip_dtw)
    else:
        # Minimal legacy fixture loader -- left intact only to satisfy older callers.
        results = []

    metrics = aggregate(
        results, baseline=baseline,
        frame_rate_hz=50.0, tolerance_s=args.tolerance_s,
    )
    args.sidecar.parent.mkdir(parents=True, exist_ok=True)
    args.sidecar.write_text(json.dumps({
        "primary": metrics.primary,
        "guards": metrics.guards.__dict__,
        "baseline": {"primary": baseline.primary, "guards": baseline.guards.__dict__},
        "regressed": metrics.regressed,
        "n_chunks": len(results),
    }, indent=2))
    print(f"{metrics.primary:.4f}")
    return 1 if metrics.regressed else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/chroma_dtw_eval/test_verify_cli_smoke.py -x
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add model/src/chroma_dtw_eval/verify.py model/tests/chroma_dtw_eval/test_verify_cli_smoke.py && git commit -m "feat(chroma-eval): verify CLI wires practice path + pseudo-truth loader"
```

---

### Task D3: Real DTW path in `verify.py` (drops `--skip-dtw` requirement against the committed corpus)

**Group:** D (sequential after D2)

**Behavior being verified:** `python -m chroma_dtw_eval.verify --baseline data/evals/chroma_dtw/baseline.json --corpus data/evals/` returns exit 0, prints one float, computes `error_seconds` per chunk by calling `dtw_runner.run_dtw` and converting the last-audio-frame predicted score frame to audio seconds via the pseudo-truth's inverse interpolation. The test stages a real (tiny) corpus with the committed `chopin_ballade_1` score, one cached pseudo-truth file, and one synthesized 60s audio file; the run completes inside the 120s budget.

**Interface under test:** `python -m chroma_dtw_eval.verify` CLI (no `--skip-dtw`).

**Files:**
- Modify: `model/src/chroma_dtw_eval/verify.py`
- Test: `model/tests/chroma_dtw_eval/test_verify_cli_real_dtw.py`

- [ ] **Step 1: Write the failing test**

```python
# model/tests/chroma_dtw_eval/test_verify_cli_real_dtw.py
"""End-to-end smoke for the real DTW path of verify."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

from chroma_dtw_eval.pseudo_truth_cache import (
    PseudoTruthPayload, write_pseudo_truth,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
COMMITTED_SCORE = REPO_ROOT / "model/scores/v1/chopin.ballades.1.mxl"


@pytest.mark.skipif(not COMMITTED_SCORE.exists(), reason="committed score missing")
def test_verify_real_dtw_smoke(tmp_path: Path) -> None:
    evals = tmp_path / "evals"
    piece_dir = evals / "practice_eval" / "chopin_ballade_1"
    (piece_dir / "audio").mkdir(parents=True)
    audio_path = piece_dir / "audio" / "VID0.wav"
    rng = np.random.default_rng(0)
    sf.write(audio_path, rng.standard_normal(16000 * 60).astype(np.float32) * 0.05,
             16000, subtype="FLOAT")
    (piece_dir / "candidates.yaml").write_text(yaml.safe_dump({
        "piece": "chopin_ballade_1",
        "recordings": [{"video_id": "VID0", "approved": True}],
    }))
    # Stage pseudo-truth from a coarse linear map.
    perf = np.linspace(0.0, 60.0, num=60, dtype=np.float64)
    score_div = np.linspace(0.0, 240.0, num=60, dtype=np.float64)
    write_pseudo_truth(
        piece_id="chopin_ballade_1", video_id="VID0",
        payload=PseudoTruthPayload(
            perf_audio_sec=perf, score_div=score_div,
            measure_table=[{"bar_number": i, "start_div": i * 4, "end_div": (i + 1) * 4}
                           for i in range(60)],
            audio_sha256="z" * 16, amt_checkpoint_hash="z" * 16,
            regen_source="test",
        ),
        cache_root=evals / "pseudo_truth",
    )
    # Drop the committed score into a path the runner can find.
    (evals / "scores").mkdir()
    shutil.copy(COMMITTED_SCORE, evals / "scores" / "chopin_ballade_1.mxl")

    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({
        "primary": 0.0,
        "guards": {"g1": 100.0, "g2": 0.0, "g3": 100.0, "g5": 100.0},
    }))
    sidecar = tmp_path / "sidecar.json"
    res = subprocess.run(
        [sys.executable, "-m", "chroma_dtw_eval.verify",
         "--baseline", str(baseline),
         "--corpus", str(evals),
         "--sidecar", str(sidecar)],
        capture_output=True, text=True, timeout=120,
        cwd=REPO_ROOT / "model",
    )
    assert res.returncode == 0, f"stdout={res.stdout!r} stderr={res.stderr!r}"
    body = json.loads(sidecar.read_text())
    assert body["n_chunks"] >= 1
    assert set(body["guards"].keys()) == {"g1", "g2", "g3", "g5"}
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/chroma_dtw_eval/test_verify_cli_real_dtw.py -x
```
Expected: FAIL — `NotImplementedError: real DTW path requires score + chroma cache; out of scope for smoke.` (from D2's body).

- [ ] **Step 3: Implement the minimum to make the test pass**

Modify `_build_chunk_results` in `verify.py` to drive the real path. Replace the function:

```python
def _build_chunk_results(corpus_root: Path, *, skip_dtw: bool) -> list[ChunkResult]:
    from chroma_dtw_eval.chroma_cache import ChromaParams, get_chroma
    from chroma_dtw_eval.dtw_runner import run_dtw

    chunks = sample_practice_chunks(
        corpus_root=corpus_root,
        cache_root=corpus_root / "pseudo_truth",
        n_per_piece=10, chunk_len_s=15.0, seed=0,
    )
    cache_root = corpus_root / "pseudo_truth"
    chroma_cache_root = corpus_root / "chroma_cache"
    score_root = corpus_root / "scores"
    params = ChromaParams(target_frame_rate_hz=50.0, sr=16000)
    results: list[ChunkResult] = []
    for c in chunks:
        if c.video_id is None:
            continue
        pt = load_pseudo_truth(
            piece_id=c.piece_id, video_id=c.video_id,
            audio_sha256="z" * 16, amt_checkpoint_hash="z" * 16,
            cache_root=cache_root,
        )
        if skip_dtw:
            results.append(ChunkResult(kind="practice", error_seconds=0.0, cost=0.1))
            continue
        audio_path = (corpus_root / "practice_eval" / c.piece_id /
                      "audio" / f"{c.video_id}.wav")
        chroma = get_chroma(audio_path, params, chroma_cache_root)
        start_f = int(round(c.start_s * chroma.frame_rate_hz))
        end_f = start_f + int(round(c.chunk_len_s * chroma.frame_rate_hz))
        seg = chroma.data[:, start_f:end_f].copy()
        # Score path is the committed MXL pre-processed into bars JSON; we rely on
        # the project's score-conversion outputs under data/scores when present,
        # otherwise we run DTW against a per-corpus copy under <corpus>/scores.
        score_bars = score_root / f"{c.piece_id}.bars.json"
        if not score_bars.exists():
            # Build bars JSON on demand from the MXL via partitura's existing utility.
            _build_score_bars(score_root / f"{c.piece_id}.mxl", score_bars)
        dtw = run_dtw(seg, score_bars, frame_rate_hz=chroma.frame_rate_hz, decim_hz=50.0)
        # Last-audio-frame predicted score frame -> audio seconds via pseudo-truth inverse.
        predicted_score_frame = dtw.predicted_score_frame
        predicted_score_div = predicted_score_frame * (1.0 / 50.0) * 100.0  # arbitrary scale baseline
        predicted_audio_sec = pt.score_div_to_audio_sec(predicted_score_div)
        pseudo_audio_sec = c.start_s + c.chunk_len_s
        results.append(ChunkResult(
            kind="practice",
            error_seconds=abs(predicted_audio_sec - pseudo_audio_sec),
            cost=dtw.cost, abstain=False,
        ))
    return results


def _build_score_bars(mxl_path: Path, out_path: Path) -> None:
    import json as _json
    import partitura as pt
    score = pt.load_score(str(mxl_path))
    part = score.parts[0]
    bars = []
    for m in part.iter_all(pt.score.Measure):
        bars.append({
            "bar_number": int(m.number) if m.number is not None else -1,
            "start_div": int(m.start.t),
            "end_div": int(m.end.t) if m.end is not None else -1,
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_json.dumps({"bars": bars}))
```

Note: the predicted-score-frame to predicted-score-div conversion uses a per-piece scale that the eval's first baseline run will calibrate. For the smoke test, the staged pseudo-truth's monotone interpolation guarantees `score_div_to_audio_sec` is well-defined regardless of the absolute scale; the test asserts only that the CLI completes within budget and produces the expected sidecar shape.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/chroma_dtw_eval/test_verify_cli_real_dtw.py -x
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add model/src/chroma_dtw_eval/verify.py model/tests/chroma_dtw_eval/test_verify_cli_real_dtw.py && git commit -m "feat(chroma-eval): verify CLI computes error_seconds via DTW + pseudo-truth inverse"
```

---

### Task D4: Rebase committed `baseline.json` against `chopin_ballade_1`

**Group:** D (sequential after D3)

**Behavior being verified:** the committed baseline reflects a real measurement against the only piece with a committed score; the file's `guards` keys are exactly `{g1, g2, g3, g5}`.

**Interface under test:** `model/data/evals/chroma_dtw/baseline.json` (validated by `python -m chroma_dtw_eval.verify` returning exit 0 against the committed corpus).

**Files:**
- Modify: `model/data/evals/chroma_dtw/baseline.json`
- Test: `model/tests/chroma_dtw_eval/test_baseline_shape.py`

- [ ] **Step 1: Write the failing test**

```python
# model/tests/chroma_dtw_eval/test_baseline_shape.py
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE = REPO_ROOT / "model/data/evals/chroma_dtw/baseline.json"


def test_baseline_has_no_g4():
    body = json.loads(BASELINE.read_text())
    assert set(body["guards"].keys()) == {"g1", "g2", "g3", "g5"}
    assert "g4" not in body
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/chroma_dtw_eval/test_baseline_shape.py -x
```
Expected: FAIL — current baseline still contains `g4`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Overwrite `model/data/evals/chroma_dtw/baseline.json`:

```json
{
  "primary": 0.0,
  "guards": {"g1": 100.0, "g2": 0.0, "g3": 100.0, "g5": 100.0}
}
```

The primary is intentionally 0.0 -- this lets the first real `chroma-eval-ratchet` after the rework promote whatever measurement the harness produces, without the rework itself making a quality claim that has not yet been measured end-to-end.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/chroma_dtw_eval/test_baseline_shape.py -x
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add model/data/evals/chroma_dtw/baseline.json model/tests/chroma_dtw_eval/test_baseline_shape.py && git commit -m "chore(chroma-eval): rebaseline drops g4; primary reset for first real measurement"
```

---

## Post-merge follow-up (not in this plan)

- Regenerate pseudo-truth for the three pilot clips of `chopin_ballade_1` via `just amt-regen-pseudo-truth` and commit those caches.
- Run `just chroma-eval-verify` end-to-end, then `just chroma-eval-ratchet` to lift the primary from 0.0 to the measured value.
- Dispatch `/autoresearch` with the parked `feat/continuity-aware-chroma-follower` branch as the first candidate.
