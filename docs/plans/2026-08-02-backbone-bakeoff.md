# Frozen Backbone Bake-Off (#138 Phase 0) Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent
> per task within a group). Do NOT start execution until `/challenge` returns
> `VERDICT: PROCEED`.

**Goal:** Give a human a composer-disjoint, tau-c-scored comparison of frozen
Aria-medium vs. frozen MoonBeam-839M embeddings on Transkun-domain difficulty
MIDIs, fully offline-testable, so they can pick a backbone for the #138
Phase-1 LoRA fine-tune without guessing.
**Spec:** `docs/specs/2026-08-02-backbone-bakeoff-design.md`
**Style:** Follow `CLAUDE.md` / `model/CLAUDE.md` if present. Python via `uv`,
never `pip`. Every new file lives under
`model/src/claim_measurement/difficulty/` (created fresh by this plan — it
does not exist on this branch). Tests are colocated (`test_*.py` beside the
module under test), run via
`cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov`
(this repo's established pattern — verified working against
`src/claim_measurement/score_align/`, 19/19 passed; the `gc_error_bars`
sibling module's colocated test is a broken precedent — flat `from x import y`
— and must NOT be copied. Every test in this plan uses the fully-qualified
`from claim_measurement.difficulty.X import Y` form instead).

**Work location:** `/Users/jdhiman/Documents/crescendai/.worktrees/issue-138-encoder-finetune`,
branch `issue-138-encoder-finetune`. This worktree already exists — do not
create a new one. All commands below assume `cd` into this worktree's
`model/` directory unless stated otherwise.

**Verified preconditions (checked during planning, do not re-verify):**
- `model/src/model_improvement/aria_embeddings.py` exists on this branch with
  `extract_embedding(midi_path, variant="embedding") -> torch.Tensor`.
  `aria = { git = "https://github.com/EleutherAI/aria.git" }` is already a
  `model/pyproject.toml` dependency (line 126) — the shared `model/.venv`
  already has `aria`, `ariautils`, `torch`, `safetensors`.
- `scikit-learn>=1.3.0` and `scipy` are already `model/pyproject.toml`
  dependencies.
- This worktree's `model/data/` does **not** contain
  `raw/psyllabus/new_clean_data.json`, `results/amt_gap_curve/`, or
  `weights/aria-medium-embedding` — those exist only under the main
  checkout's `model/data/` (confirmed via filesystem check). This is why
  every path in this plan is `__file__`-anchored with an override, never a
  hardcoded absolute path — the human running the real bake-off passes
  `--data-root /Users/jdhiman/Documents/crescendai/model/data`.
- No `moonbeam` references exist anywhere in this repo (confirmed via
  repo-wide grep) — this is a from-scratch integration.

## Task Groups

```
Group A (parallel): Task 1
Group B (sequential, same file): Task 2 -> Task 3 -> Task 4
Group C (sequential, same file, parallel with A/B): Task 5 -> Task 6
Group D (sequential, same file, parallel with A/B/C): Task 7 -> Task 8
Group E (parallel with A/B/C/D): Task 9
Group F (depends on C, D, E): Task 10
Group G (parallel with F, depends on nothing new): Task 11
Group H (sequential, same file, parallel with F/G): Task 12a -> Task 12b
Group I (depends on F, H, C, D): Task 13
Group J (sequential, same file, depends on A, B, C, D): Task 14 -> Task 15
```

None of these groups ship independently to an end user (there is no user
until the human runs the bake-off) — the whole plan is one `[SHIPS
INDEPENDENTLY]` unit whose deliverable is "harness ready to run on GPU."

---

### Task 1: `__file__`-anchored path resolution
**Group:** A (parallel)

**Behavior being verified:** `resolve_paths` builds the four bake-off data
paths under a given root, so the harness works identically in this worktree
(no local data) and against the main checkout (real data) via one override
argument.
**Interface under test:** `resolve_paths(data_root: Path | None) -> BakeoffPaths`

**Files:**
- Create: `model/src/claim_measurement/difficulty/__init__.py`
- Create: `model/src/claim_measurement/difficulty/bakeoff_paths.py`
- Test: `model/src/claim_measurement/difficulty/test_bakeoff_paths.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for bakeoff_paths.

Run: cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
"""
from pathlib import Path

from claim_measurement.difficulty.bakeoff_paths import resolve_paths


def test_resolve_paths_uses_override_root():
    paths = resolve_paths(data_root=Path("/tmp/fake_data_root"))
    assert paths.manifest == Path("/tmp/fake_data_root/results/amt_gap_curve/manifest.json")
    assert paths.labels == Path("/tmp/fake_data_root/raw/psyllabus/new_clean_data.json")
    assert paths.transkun_mid_dir == Path("/tmp/fake_data_root/results/amt_gap_curve/transkun_mid")
    assert paths.emb_root == Path("/tmp/fake_data_root/results/bakeoff")
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_bakeoff_paths.py -q --no-cov
```
Expected: FAIL — `ModuleNotFoundError: No module named 'claim_measurement.difficulty'` (package/module do not exist yet).

- [ ] **Step 3: Implement the minimum to make the test pass**

`model/src/claim_measurement/difficulty/__init__.py`: empty file.

`model/src/claim_measurement/difficulty/bakeoff_paths.py`:
```python
"""__file__-anchored default data paths for the backbone bake-off (#138 Phase 0).

Worktrees each have their own (independently populated) model/data/ directory.
The 5798 Transkun MIDIs, manifest.json, and psyllabus labels currently exist
only under the main checkout's model/data/ -- so every path here is overridable
via `resolve_paths(data_root=...)` (and run_bakeoff.py's --data-root), never
hardcoded to an absolute machine path.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[3] / "data"


@dataclass(frozen=True)
class BakeoffPaths:
    manifest: Path
    labels: Path
    transkun_mid_dir: Path
    emb_root: Path


def resolve_paths(data_root: Path | None = None) -> BakeoffPaths:
    root = Path(data_root) if data_root is not None else DEFAULT_DATA_ROOT
    return BakeoffPaths(
        manifest=root / "results" / "amt_gap_curve" / "manifest.json",
        labels=root / "raw" / "psyllabus" / "new_clean_data.json",
        transkun_mid_dir=root / "results" / "amt_gap_curve" / "transkun_mid",
        emb_root=root / "results" / "bakeoff",
    )
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_bakeoff_paths.py -q --no-cov
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add model/src/claim_measurement/difficulty/__init__.py \
        model/src/claim_measurement/difficulty/bakeoff_paths.py \
        model/src/claim_measurement/difficulty/test_bakeoff_paths.py
git commit -m "feat(mirex-difficulty): bake-off __file__-anchored path resolution (#138)"
```

---

### Task 2: Kendall tau-c, ported
**Group:** B (sequential — first of 3 tasks touching `bakeoff_cv.py`)

**Behavior being verified:** `tau_c` returns `None` (never `0.0` or `NaN`) on
degenerate input — fewer than 3 points, or a constant side — and returns the
correct Kendall tau-c otherwise, matching the exact behavior ported from
`phase5b_aria_probe.py` (commit `7976b5e6`, unmerged `issue-104-mirex-difficulty`).
**Interface under test:** `tau_c(x, y) -> float | None`

**Files:**
- Create: `model/src/claim_measurement/difficulty/bakeoff_cv.py`
- Create: `model/src/claim_measurement/difficulty/test_bakeoff_cv.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for bakeoff_cv (ported from the unmerged issue-104-mirex-difficulty
branch's phase5b_aria_probe.py, commit 7976b5e6 -- see the design spec for why
this is a port, not a cross-branch import).

Run: cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
"""
import math

from claim_measurement.difficulty.bakeoff_cv import tau_c


def test_tau_c_perfect_agreement_is_one():
    assert tau_c([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0


def test_tau_c_perfect_disagreement_is_minus_one():
    assert tau_c([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0


def test_tau_c_none_for_constant_y():
    assert tau_c([1, 2, 3, 4], [5, 5, 5, 5]) is None


def test_tau_c_none_for_fewer_than_three_points():
    assert tau_c([1, 2], [1, 2]) is None


def test_tau_c_handles_ties_without_raising():
    result = tau_c([1, 1, 2, 3], [1, 2, 2, 3])
    assert result is not None
    assert not math.isnan(result)
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_bakeoff_cv.py -q --no-cov
```
Expected: FAIL — `ModuleNotFoundError: No module named 'claim_measurement.difficulty.bakeoff_cv'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
"""Composer-disjoint CV + Kendall tau-c, ported from the unmerged
issue-104-mirex-difficulty branch's phase5b_aria_probe.py (commit 7976b5e6).

Ported, not imported cross-branch: worktrees are separate checkouts and this
file does not exist on issue-138-encoder-finetune or on main. The RidgeCV
pipeline, alpha grid, and tau-c convention are unchanged from the original;
the private `_folds` closure inside the original `_oof_tau` is promoted to
the public `composer_disjoint_folds` (added in Task 3) so it is independently
testable per the #138 Phase 0 design's TDD targets.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def tau_c(x, y) -> float | None:
    """Kendall tau-c, nan-safe. None (never 0.0) when the input cannot
    support a rank correlation -- fewer than 3 points, or either side
    constant."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return None
    t = stats.kendalltau(x, y, variant="c").statistic
    return None if np.isnan(t) else float(t)
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_bakeoff_cv.py -q --no-cov
```
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add model/src/claim_measurement/difficulty/bakeoff_cv.py \
        model/src/claim_measurement/difficulty/test_bakeoff_cv.py
git commit -m "feat(mirex-difficulty): port tau_c into bake-off harness (#138)"
```

---

### Task 3: Composer-disjoint fold splitter
**Group:** B (sequential, depends on Task 2, same file)

**Behavior being verified:** No composer's rows straddle two folds — the
safety property the whole bake-off protocol depends on (a random split would
let a model memorize "Czerny pieces are grade 4").
**Interface under test:** `composer_disjoint_folds(composers, n_folds, seed) -> list[np.ndarray]`

**Files:**
- Modify: `model/src/claim_measurement/difficulty/bakeoff_cv.py`
- Modify: `model/src/claim_measurement/difficulty/test_bakeoff_cv.py`

- [ ] **Step 1: Write the failing test**

Append to `test_bakeoff_cv.py`:
```python
import numpy as np

from claim_measurement.difficulty.bakeoff_cv import composer_disjoint_folds


def test_composer_disjoint_folds_no_composer_straddles_a_fold():
    rng = np.random.default_rng(0)
    composers = np.array(
        [f"composer_{i % 30}" for i in range(300)]
    )
    rng.shuffle(composers)

    folds = composer_disjoint_folds(composers, n_folds=5, seed=2026)

    assert len(folds) == 5
    all_indices = np.concatenate(folds)
    assert sorted(all_indices) == list(range(300))  # every row covered exactly once
    fold_composer_sets = [set(composers[f]) for f in folds]
    for i in range(5):
        for j in range(i + 1, 5):
            assert not (fold_composer_sets[i] & fold_composer_sets[j]), (
                f"fold {i} and fold {j} share a composer"
            )
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_bakeoff_cv.py::test_composer_disjoint_folds_no_composer_straddles_a_fold -q --no-cov
```
Expected: FAIL — `ImportError: cannot import name 'composer_disjoint_folds'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Append to `bakeoff_cv.py`:
```python
def composer_disjoint_folds(composers: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    """Split row indices into n_folds folds such that no composer's rows
    straddle two folds. Greedy bin-packing: composers are shuffled
    deterministically by seed, then each composer's whole row group is
    assigned to the fold with the fewest rows so far."""
    composers = np.asarray(composers)
    uniq = sorted(set(composers))
    sizes = {c: int(np.sum(composers == c)) for c in uniq}
    order = np.random.default_rng(seed).permutation(len(uniq))
    counts = [0] * n_folds
    fold_of = {}
    for idx in order:
        c = uniq[idx]
        f = int(np.argmin(counts))
        fold_of[c] = f
        counts[f] += sizes[c]
    return [np.array([i for i, c in enumerate(composers) if fold_of[c] == f])
            for f in range(n_folds)]
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_bakeoff_cv.py -q --no-cov
```
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add model/src/claim_measurement/difficulty/bakeoff_cv.py \
        model/src/claim_measurement/difficulty/test_bakeoff_cv.py
git commit -m "feat(mirex-difficulty): composer-disjoint fold splitter (#138)"
```

---

### Task 4: RidgeCV out-of-fold tau-c over seeds
**Group:** B (sequential, depends on Task 3, same file)

**Behavior being verified:** `oof_tau_ridge` fits a `StandardScaler ->
RidgeCV` pipeline per composer-disjoint fold, predicts out-of-fold, and
reports mean/std tau-c over the given seeds — the exact protocol the design
requires ("composer-disjoint grouped 5-fold x 5-seed RidgeCV").
**Interface under test:** `oof_tau_ridge(X, y, composers, n_folds, seeds) -> dict`

**Files:**
- Modify: `model/src/claim_measurement/difficulty/bakeoff_cv.py`
- Modify: `model/src/claim_measurement/difficulty/test_bakeoff_cv.py`

- [ ] **Step 1: Write the failing test**

Append to `test_bakeoff_cv.py`:
```python
from claim_measurement.difficulty.bakeoff_cv import oof_tau_ridge


def test_oof_tau_ridge_recovers_a_strong_linear_signal():
    rng = np.random.default_rng(2026)
    n = 200
    composers = np.array([f"composer_{i % 20}" for i in range(n)])
    X = rng.normal(size=(n, 5))
    y = X[:, 0] * 10  # near-perfectly linearly predictable from feature 0

    result = oof_tau_ridge(X, y, composers, n_folds=5, seeds=[2026, 2027])

    assert result["n_seeds"] == 2
    assert result["mean"] > 0.5  # a strong linear signal should rank well OOF
    assert result["std"] >= 0.0


def test_oof_tau_ridge_reports_zero_seeds_when_target_is_constant():
    rng = np.random.default_rng(0)
    n = 60
    composers = np.array([f"composer_{i % 10}" for i in range(n)])
    X = rng.normal(size=(n, 3))
    y = np.zeros(n)  # constant target -> tau_c is always None

    result = oof_tau_ridge(X, y, composers, n_folds=5, seeds=[2026])

    assert result == {"mean": None, "std": None, "n_seeds": 0}
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_bakeoff_cv.py -k oof_tau_ridge -q --no-cov
```
Expected: FAIL — `ImportError: cannot import name 'oof_tau_ridge'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Append to `bakeoff_cv.py`:
```python
def oof_tau_ridge(X: np.ndarray, y: np.ndarray, composers: np.ndarray,
                   n_folds: int, seeds: list[int]) -> dict:
    """Composer-disjoint grouped n_folds-fold RidgeCV, repeated per seed
    (each seed re-draws the fold assignment). Returns mean/std tau-c over
    seeds where a fold produced a valid tau-c, and n_seeds actually used."""
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    alphas = np.logspace(-1, 5, 25)
    taus = []
    for seed in seeds:
        oof = np.full(len(y), np.nan)
        for te in composer_disjoint_folds(composers, n_folds, seed):
            tr = np.setdiff1d(np.arange(len(y)), te)
            if len(tr) < 3 or len(te) == 0:
                continue
            model = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))
            model.fit(np.nan_to_num(X[tr]), y[tr])
            oof[te] = model.predict(np.nan_to_num(X[te]))
        t = tau_c(oof, y)
        if t is not None:
            taus.append(t)
    return {"mean": float(np.mean(taus)) if taus else None,
            "std": float(np.std(taus)) if taus else None,
            "n_seeds": len(taus)}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_bakeoff_cv.py -q --no-cov
```
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add model/src/claim_measurement/difficulty/bakeoff_cv.py \
        model/src/claim_measurement/difficulty/test_bakeoff_cv.py
git commit -m "feat(mirex-difficulty): RidgeCV OOF tau-c over composer-disjoint folds (#138)"
```

---

### Task 5: `.npz` embedding contract — single pooling round trip
**Group:** C (sequential, parallel with A/B/D/E)

**Behavior being verified:** Writing then reading one piece's embedding
preserves the embedding vector, grade, and composer id exactly — the shared
output contract both backbone extractors write into.
**Interface under test:** `write_embedding_npz`, `read_embedding_npz`

**Files:**
- Create: `model/src/claim_measurement/difficulty/bakeoff_npz.py`
- Create: `model/src/claim_measurement/difficulty/test_bakeoff_npz.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for the shared .npz embedding contract.

Run: cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
"""
import numpy as np

from claim_measurement.difficulty.bakeoff_npz import read_embedding_npz, write_embedding_npz


def test_round_trip_preserves_embedding_grade_composer(tmp_path):
    path = tmp_path / "piece_001.npz"
    write_embedding_npz(path, {"embedding": np.array([1.0, 2.0, 3.0], dtype=np.float32)},
                         grade=7, composer_id=42)

    record = read_embedding_npz(path)

    np.testing.assert_array_equal(record.embeddings["embedding"], [1.0, 2.0, 3.0])
    assert record.grade == 7
    assert record.composer_id == 42
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_bakeoff_npz.py -q --no-cov
```
Expected: FAIL — `ModuleNotFoundError: No module named 'claim_measurement.difficulty.bakeoff_npz'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
"""Per-piece .npz embedding contract shared by every backbone extractor.

Numeric-only arrays (float32 vectors, int32 scalars) so np.load never needs
pickle=True. A backbone may produce more than one pooled vector per piece
(MoonBeam: mean_pool + last_token); each is stored as its own array keyed
"emb__{pooling_name}" so an arbitrary number of poolings round-trip through
one file without colliding with the reserved "grade"/"composer_id" keys.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_EMB_PREFIX = "emb__"


@dataclass(frozen=True)
class EmbeddingRecord:
    embeddings: dict[str, np.ndarray]
    grade: int
    composer_id: int


def write_embedding_npz(path: Path, embeddings: dict[str, np.ndarray], grade: int, composer_id: int) -> None:
    if not embeddings:
        raise ValueError("embeddings must contain at least one pooling vector")
    arrays = {f"{_EMB_PREFIX}{name}": np.asarray(vec, dtype=np.float32)
              for name, vec in embeddings.items()}
    arrays["grade"] = np.array(int(grade), dtype=np.int32)
    arrays["composer_id"] = np.array(int(composer_id), dtype=np.int32)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def read_embedding_npz(path: Path) -> EmbeddingRecord:
    with np.load(path) as z:
        embeddings = {k[len(_EMB_PREFIX):]: z[k] for k in z.files if k.startswith(_EMB_PREFIX)}
        grade = int(z["grade"])
        composer_id = int(z["composer_id"])
    return EmbeddingRecord(embeddings=embeddings, grade=grade, composer_id=composer_id)
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_bakeoff_npz.py -q --no-cov
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add model/src/claim_measurement/difficulty/bakeoff_npz.py \
        model/src/claim_measurement/difficulty/test_bakeoff_npz.py
git commit -m "feat(mirex-difficulty): .npz embedding contract, single pooling (#138)"
```

---

### Task 6: `.npz` embedding contract — multi-pooling round trip
**Group:** C (sequential, depends on Task 5, same file)

**Behavior being verified:** A piece with TWO pooling vectors (MoonBeam's
mean-over-tokens + last-token) round-trips both without collision — the
property that makes one contract serve both backbones.
**Interface under test:** `write_embedding_npz`, `read_embedding_npz`

**Files:**
- Modify: `model/src/claim_measurement/difficulty/test_bakeoff_npz.py`

- [ ] **Step 1: Write the failing test**

Append to `test_bakeoff_npz.py`:
```python
def test_round_trip_preserves_multiple_poolings(tmp_path):
    path = tmp_path / "piece_002.npz"
    write_embedding_npz(
        path,
        {"mean_pool": np.array([0.1, 0.2], dtype=np.float32),
         "last_token": np.array([0.9, 0.8], dtype=np.float32)},
        grade=3, composer_id=1,
    )

    record = read_embedding_npz(path)

    assert set(record.embeddings) == {"mean_pool", "last_token"}
    np.testing.assert_allclose(record.embeddings["mean_pool"], [0.1, 0.2])
    np.testing.assert_allclose(record.embeddings["last_token"], [0.9, 0.8])
```

- [ ] **Step 2: Run test — verify it FAILS**

Actually run this before assuming: given Task 5's implementation already
supports an arbitrary number of embeddings, this MAY pass immediately. Run it
first to confirm which case applies:

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_bakeoff_npz.py::test_round_trip_preserves_multiple_poolings -q --no-cov
```
If it FAILS, proceed to Step 3. If it already PASSES, this confirms the
Task-5 implementation already generalizes — skip Step 3, note this in the
commit message, and proceed to Step 5.

- [ ] **Step 3: Implement the minimum to make the test pass** (only if Step 2 failed)

No implementation change is expected — `write_embedding_npz`/`read_embedding_npz`
already iterate `embeddings` as a dict of arbitrary size. If it does fail,
the most likely cause is a key-prefix collision; fix by confirming the
`_EMB_PREFIX` stripping in `read_embedding_npz` only strips the prefix
(`k[len(_EMB_PREFIX):]`), not the whole key.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_bakeoff_npz.py -q --no-cov
```
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add model/src/claim_measurement/difficulty/test_bakeoff_npz.py
git commit -m "test(mirex-difficulty): npz contract covers multi-pooling backbones (#138)"
```

---

### Task 7: Manifest + composer-labels join
**Group:** D (sequential, parallel with A/B/C/E)

**Behavior being verified:** Given `amt_gap_curve/manifest.json` (seg_id,
key, grade — no composer field) and `new_clean_data.json` (key -> composer),
`load_bakeoff_manifest` joins them and drops any entry missing a composer or
missing an on-disk Transkun MIDI — mirroring the exact filter
`tk_ablation.py` uses (`issue-137-transkun-features`, unmerged) for the same
manifest.
**Interface under test:** `load_bakeoff_manifest(manifest_path, labels_path, transkun_mid_dir) -> list[ManifestEntry]`

**Files:**
- Create: `model/src/claim_measurement/difficulty/bakeoff_sampling.py`
- Create: `model/src/claim_measurement/difficulty/test_bakeoff_sampling.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for bakeoff_sampling.

Run: cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
"""
import json

from claim_measurement.difficulty.bakeoff_sampling import (
    ManifestEntry,
    load_bakeoff_manifest,
)


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_load_bakeoff_manifest_joins_and_filters(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    labels_path = tmp_path / "new_clean_data.json"
    mid_dir = tmp_path / "transkun_mid"
    mid_dir.mkdir()

    _write_json(manifest_path, [
        {"seg_id": "has_composer_has_midi", "key": "A.mid", "grade": 3,
         "video_id": "x", "midi_name": "mid/A.mid"},
        {"seg_id": "no_composer", "key": "B.mid", "grade": 5,
         "video_id": "y", "midi_name": "mid/B.mid"},
        {"seg_id": "no_midi_on_disk", "key": "C.mid", "grade": 1,
         "video_id": "z", "midi_name": "mid/C.mid"},
    ])
    _write_json(labels_path, {
        "A.mid": {"composer": "Bach"},
        "B.mid": {"composer": ""},
        "C.mid": {"composer": "Czerny"},
    })
    (mid_dir / "has_composer_has_midi.mid").write_bytes(b"")
    # no_midi_on_disk.mid deliberately absent

    entries = load_bakeoff_manifest(manifest_path, labels_path, mid_dir)

    assert entries == [ManifestEntry(seg_id="has_composer_has_midi", key="A.mid",
                                      grade=3, composer="Bach")]
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_bakeoff_sampling.py -q --no-cov
```
Expected: FAIL — `ModuleNotFoundError: No module named 'claim_measurement.difficulty.bakeoff_sampling'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
"""Manifest+labels join and composer-stratified sampling for the bake-off.

Pure functions: no MIDI parsing here (that's each Backbone adapter's job at
extraction time) -- this module only decides WHICH pieces make the sample.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ManifestEntry:
    seg_id: str
    key: str
    grade: int
    composer: str


def load_bakeoff_manifest(manifest_path: Path, labels_path: Path, transkun_mid_dir: Path) -> list[ManifestEntry]:
    """Join amt_gap_curve/manifest.json (seg_id/key/grade) against
    new_clean_data.json (key -> composer), keeping only entries that have
    BOTH a non-empty composer AND an on-disk Transkun MIDI -- a composer-less
    entry cannot be placed in a composer-disjoint fold, and a piece without a
    Transkun MIDI cannot be embedded."""
    manifest = json.loads(Path(manifest_path).read_text())
    labels = json.loads(Path(labels_path).read_text())
    entries = []
    for m in manifest:
        composer = str(labels.get(m["key"], {}).get("composer", "")).strip()
        if not composer:
            continue
        if not (Path(transkun_mid_dir) / f"{m['seg_id']}.mid").exists():
            continue
        entries.append(ManifestEntry(seg_id=m["seg_id"], key=m["key"],
                                      grade=int(m["grade"]), composer=composer))
    return entries
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_bakeoff_sampling.py -q --no-cov
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add model/src/claim_measurement/difficulty/bakeoff_sampling.py \
        model/src/claim_measurement/difficulty/test_bakeoff_sampling.py
git commit -m "feat(mirex-difficulty): manifest+composer-labels join (#138)"
```

---

### Task 8: Composer-stratified sampling
**Group:** D (sequential, depends on Task 7, same file)

**Behavior being verified:** `composer_stratified_sample` returns exactly
`target_n` entries (when enough exist) and every composer present in the
input contributes at least one piece — the property "composer-stratified"
actually means, and the one the ~800-1000-piece sample in the design depends
on for composer-disjoint CV to have enough groups.
**Interface under test:** `composer_stratified_sample(entries, target_n, seed) -> list[ManifestEntry]`

**Files:**
- Modify: `model/src/claim_measurement/difficulty/bakeoff_sampling.py`
- Modify: `model/src/claim_measurement/difficulty/test_bakeoff_sampling.py`

- [ ] **Step 1: Write the failing test**

Append to `test_bakeoff_sampling.py`:
```python
from claim_measurement.difficulty.bakeoff_sampling import composer_stratified_sample


def _make_entries(n_composers: int, pieces_per_composer: int) -> list[ManifestEntry]:
    entries = []
    for c in range(n_composers):
        for p in range(pieces_per_composer):
            entries.append(ManifestEntry(
                seg_id=f"c{c}_p{p}", key=f"c{c}_p{p}.mid",
                grade=p % 11, composer=f"composer_{c}",
            ))
    return entries


def test_composer_stratified_sample_covers_every_composer_and_hits_target_n():
    entries = _make_entries(n_composers=50, pieces_per_composer=20)  # 1000 entries

    sample = composer_stratified_sample(entries, target_n=200, seed=2026)

    assert len(sample) == 200
    assert len({e.seg_id for e in sample}) == 200  # no duplicates
    sampled_composers = {e.composer for e in sample}
    assert sampled_composers == {f"composer_{c}" for c in range(50)}  # every composer represented


def test_composer_stratified_sample_returns_everything_when_target_exceeds_pool():
    entries = _make_entries(n_composers=5, pieces_per_composer=3)  # 15 entries

    sample = composer_stratified_sample(entries, target_n=100, seed=2026)

    assert len(sample) == 15
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_bakeoff_sampling.py -k stratified -q --no-cov
```
Expected: FAIL — `ImportError: cannot import name 'composer_stratified_sample'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Append to `bakeoff_sampling.py`:
```python
def composer_stratified_sample(entries: list[ManifestEntry], target_n: int, seed: int) -> list[ManifestEntry]:
    """Draw up to target_n entries so every composer present in `entries`
    contributes at least one piece to the output (when target_n >= the
    number of distinct composers), proportional to each composer's share of
    `entries` beyond that floor, deterministic given `seed`."""
    if target_n <= 0:
        raise ValueError("target_n must be positive")
    if target_n >= len(entries):
        return list(entries)

    by_composer: dict[str, list[ManifestEntry]] = {}
    for e in entries:
        by_composer.setdefault(e.composer, []).append(e)
    composers = sorted(by_composer)
    rng = np.random.default_rng(seed)
    for c in composers:
        rng.shuffle(by_composer[c])  # deterministic per-composer order

    if len(composers) > target_n:
        keep = sorted(rng.permutation(composers)[:target_n].tolist())
        return [by_composer[c][0] for c in keep]

    quotas = {c: 1 for c in composers}
    remaining = target_n - len(composers)
    total = len(entries)
    for c in composers:
        share = len(by_composer[c]) / total
        quotas[c] += int(round(share * remaining))

    sample: list[ManifestEntry] = []
    for c in composers:
        take = min(quotas[c], len(by_composer[c]))
        sample.extend(by_composer[c][:take])

    if len(sample) > target_n:
        rng.shuffle(sample)
        sample = sample[:target_n]
    elif len(sample) < target_n:
        taken_ids = {e.seg_id for e in sample}
        leftover = [e for e in entries if e.seg_id not in taken_ids]
        rng.shuffle(leftover)
        sample.extend(leftover[: target_n - len(sample)])

    return sample
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_bakeoff_sampling.py -q --no-cov
```
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add model/src/claim_measurement/difficulty/bakeoff_sampling.py \
        model/src/claim_measurement/difficulty/test_bakeoff_sampling.py
git commit -m "feat(mirex-difficulty): composer-stratified sampling (#138)"
```

---

### Task 9: `Backbone` protocol + `FakeBackbone`
**Group:** E (parallel with A/B/C/D)

**Behavior being verified:** `FakeBackbone.embed()` returns one vector per
declared pooling name, of the declared dimension, deterministic given the
same MIDI path — the test double every other module's tests depend on to
never touch real model weights.
**Interface under test:** `FakeBackbone.embed(midi_path) -> dict[str, np.ndarray]`

**Files:**
- Create: `model/src/claim_measurement/difficulty/backbone.py`
- Create: `model/src/claim_measurement/difficulty/test_backbone.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for the Backbone protocol's test double.

Run: cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
"""
from pathlib import Path

import numpy as np

from claim_measurement.difficulty.backbone import FakeBackbone


def test_fake_backbone_returns_declared_poolings_and_dim():
    backbone = FakeBackbone(pooling_names=("mean_pool", "last_token"), dim=6)

    result = backbone.embed(Path("/fake/piece.mid"))

    assert set(result) == {"mean_pool", "last_token"}
    assert result["mean_pool"].shape == (6,)
    assert result["mean_pool"].dtype == np.float32


def test_fake_backbone_is_deterministic_per_path_but_differs_across_paths():
    backbone = FakeBackbone()

    a1 = backbone.embed(Path("/fake/a.mid"))["embedding"]
    a2 = backbone.embed(Path("/fake/a.mid"))["embedding"]
    b = backbone.embed(Path("/fake/b.mid"))["embedding"]

    np.testing.assert_array_equal(a1, a2)
    assert not np.array_equal(a1, b)
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_backbone.py -q --no-cov
```
Expected: FAIL — `ModuleNotFoundError: No module named 'claim_measurement.difficulty.backbone'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
"""The seam: a narrow interface both real backbones and test fakes implement,
so extraction and its tests never depend on Aria or MoonBeam internals."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol


class Backbone(Protocol):
    def embed(self, midi_path: Path) -> dict:
        """Return one or more named pooled embedding vectors for one MIDI file."""
        ...


class FakeBackbone:
    """Deterministic, weight-free stand-in: each pooling name maps to a
    fixed-length vector derived from a hash of the MIDI path, so different
    paths get different (but reproducible) vectors and tests never touch a
    real model."""

    def __init__(self, pooling_names: tuple[str, ...] = ("embedding",), dim: int = 8):
        self.pooling_names = pooling_names
        self.dim = dim

    def embed(self, midi_path: Path) -> dict:
        import numpy as np
        seed = int(hashlib.sha256(str(midi_path).encode()).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        return {name: rng.random(self.dim).astype(np.float32) for name in self.pooling_names}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_backbone.py -q --no-cov
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add model/src/claim_measurement/difficulty/backbone.py \
        model/src/claim_measurement/difficulty/test_backbone.py
git commit -m "feat(mirex-difficulty): Backbone protocol + FakeBackbone test double (#138)"
```

---

### Task 10: Extraction orchestrator
**Group:** F (depends on Task 6 [npz], Task 8 [sampling/ManifestEntry], Task 9 [backbone])

**Behavior being verified:** Given any `Backbone` and a list of
`ManifestEntry`, `extract_embeddings` writes one conformant `.npz` per piece
and records failures loudly without stopping the run — the exact TDD target
"an extractor built on a fake backbone produces conformant .npz without any
real model weights."
**Interface under test:** `extract_embeddings(backbone, entries, midi_dir, out_dir, composer_index_path) -> ExtractionReport`

**Files:**
- Create: `model/src/claim_measurement/difficulty/extract.py`
- Create: `model/src/claim_measurement/difficulty/test_extract.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for the backbone-agnostic extraction orchestrator.

Run: cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
"""
from claim_measurement.difficulty.backbone import FakeBackbone
from claim_measurement.difficulty.bakeoff_npz import read_embedding_npz
from claim_measurement.difficulty.bakeoff_sampling import ManifestEntry
from claim_measurement.difficulty.extract import extract_embeddings


def test_extract_embeddings_with_fake_backbone_produces_conformant_npz(tmp_path):
    entries = [
        ManifestEntry(seg_id="a", key="A.mid", grade=3, composer="Czerny"),
        ManifestEntry(seg_id="b", key="B.mid", grade=7, composer="Bach"),
    ]
    backbone = FakeBackbone(pooling_names=("mean_pool", "last_token"), dim=4)
    out_dir = tmp_path / "emb"
    index_path = tmp_path / "composer_index.json"

    report = extract_embeddings(backbone, entries, midi_dir=tmp_path / "mid",
                                 out_dir=out_dir, composer_index_path=index_path)

    assert report.ok == 2
    assert report.failed == []
    rec_a = read_embedding_npz(out_dir / "a.npz")
    assert set(rec_a.embeddings) == {"mean_pool", "last_token"}
    assert rec_a.embeddings["mean_pool"].shape == (4,)
    assert rec_a.grade == 3
    rec_b = read_embedding_npz(out_dir / "b.npz")
    assert rec_a.composer_id != rec_b.composer_id


def test_extract_embeddings_records_failures_and_continues(tmp_path):
    class BrokenOnB:
        def embed(self, midi_path):
            if "b" in str(midi_path):
                raise RuntimeError("simulated corrupt MIDI")
            return {"embedding": __import__("numpy").zeros(3, dtype="float32")}

    entries = [
        ManifestEntry(seg_id="a", key="A.mid", grade=1, composer="Liszt"),
        ManifestEntry(seg_id="b", key="B.mid", grade=2, composer="Chopin"),
    ]
    out_dir = tmp_path / "emb"

    report = extract_embeddings(BrokenOnB(), entries, midi_dir=tmp_path / "mid",
                                 out_dir=out_dir, composer_index_path=tmp_path / "idx.json")

    assert report.ok == 1
    assert len(report.failed) == 1
    assert "b" in report.failed[0]
    assert (out_dir / "a.npz").exists()
    assert not (out_dir / "b.npz").exists()
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_extract.py -q --no-cov
```
Expected: FAIL — `ModuleNotFoundError: No module named 'claim_measurement.difficulty.extract'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
"""Backbone-agnostic per-piece extraction: iterate manifest entries, call the
backbone, write the shared .npz contract. Failures are recorded loudly, never
silently dropped (a bad MIDI or an OOM on one piece must not corrupt the run
or vanish from the report)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from claim_measurement.difficulty.bakeoff_npz import write_embedding_npz
from claim_measurement.difficulty.bakeoff_sampling import ManifestEntry
from claim_measurement.difficulty.backbone import Backbone


@dataclass
class ExtractionReport:
    ok: int = 0
    failed: list = field(default_factory=list)


def _composer_id(composer: str, index_path: Path) -> int:
    """Look up (or append) composer in the shared composer_index.json so
    every backbone's npz files reference the same numeric ids."""
    if index_path.exists():
        index = json.loads(index_path.read_text())
    else:
        index = []
    if composer not in index:
        index.append(composer)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(json.dumps(index))
    return index.index(composer)


def extract_embeddings(backbone: Backbone, entries: list[ManifestEntry],
                        midi_dir: Path, out_dir: Path, composer_index_path: Path) -> ExtractionReport:
    report = ExtractionReport()
    for entry in entries:
        midi_path = Path(midi_dir) / f"{entry.seg_id}.mid"
        try:
            embeddings = backbone.embed(midi_path)
            composer_id = _composer_id(entry.composer, composer_index_path)
            write_embedding_npz(Path(out_dir) / f"{entry.seg_id}.npz",
                                 embeddings, grade=entry.grade, composer_id=composer_id)
            report.ok += 1
        except Exception as exc:  # noqa: BLE001 -- record and continue; the run report is the source of truth
            report.failed.append(f"{entry.seg_id}: {exc!r}")
    return report
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_extract.py -q --no-cov
```
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add model/src/claim_measurement/difficulty/extract.py \
        model/src/claim_measurement/difficulty/test_extract.py
git commit -m "feat(mirex-difficulty): backbone-agnostic extraction orchestrator (#138)"
```

---

### Task 11: Aria backbone adapter
**Group:** G (parallel with F — no dependency on this plan's own modules beyond nothing)

**Behavior being verified:** `AriaBackbone.embed()` calls
`model_improvement.aria_embeddings.extract_embedding(path, variant="embedding")`
and returns `{"embedding": <float32 ndarray>}` — verified with the real
Aria call monkeypatched, since loading the real 512-dim weights is the
human-lit GPU boundary, not this adapter's job.
**Interface under test:** `AriaBackbone.embed(midi_path) -> dict[str, np.ndarray]`

**Files:**
- Create: `model/src/claim_measurement/difficulty/aria_backbone.py`
- Create: `model/src/claim_measurement/difficulty/test_aria_backbone.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for the Aria backbone adapter (no real weights loaded).

Run: cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
"""
from pathlib import Path

import numpy as np
import torch

import model_improvement.aria_embeddings as aria_embeddings
from claim_measurement.difficulty.aria_backbone import AriaBackbone


def test_embed_wraps_extract_embedding_as_numpy(monkeypatch):
    def fake_extract_embedding(midi_path, variant="embedding"):
        assert variant == "embedding"
        return torch.tensor([1.0, 2.0, 3.0])

    monkeypatch.setattr(aria_embeddings, "extract_embedding", fake_extract_embedding)

    result = AriaBackbone().embed(Path("/fake/piece.mid"))

    assert set(result) == {"embedding"}
    assert isinstance(result["embedding"], np.ndarray)
    assert result["embedding"].dtype == np.float32
    np.testing.assert_allclose(result["embedding"], [1.0, 2.0, 3.0])
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_aria_backbone.py -q --no-cov
```
Expected: FAIL — `ModuleNotFoundError: No module named 'claim_measurement.difficulty.aria_backbone'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
"""Aria-medium adapter: the thin seam between the Backbone protocol and the
real Aria weight-loading/inference call in model_improvement.aria_embeddings.
Loading real weights is the human-lit GPU boundary -- this class is tested by
monkeypatching extract_embedding, never by loading a real checkpoint."""
from __future__ import annotations

from pathlib import Path

import numpy as np


class AriaBackbone:
    """Backbone protocol implementation over the existing 512-dim
    TransformerEMB embedding path."""

    def embed(self, midi_path: Path) -> dict:
        from model_improvement.aria_embeddings import extract_embedding
        vec = extract_embedding(midi_path, variant="embedding")
        return {"embedding": vec.detach().cpu().numpy().astype(np.float32)}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_aria_backbone.py -q --no-cov
```
Expected: PASS. If it fails instead with an import error unrelated to
`claim_measurement` (e.g. `aria`/`ariautils`/`torch`/`safetensors` missing),
this contradicts the plan's verified precondition — stop and report rather
than reinterpreting the failure.

- [ ] **Step 5: Commit**

```bash
git add model/src/claim_measurement/difficulty/aria_backbone.py \
        model/src/claim_measurement/difficulty/test_aria_backbone.py
git commit -m "feat(mirex-difficulty): Aria backbone adapter (#138)"
```

---

### Task 12a: MoonBeam pooling math
**Group:** H (sequential — first of 2 tasks touching `moonbeam_backbone.py`, parallel with F/G)

**Behavior being verified:** Given raw per-token hidden states, `MoonBeamBackbone.embed()`
computes BOTH candidate poolings the design calls for — mean-over-tokens and
last-token — correctly, entirely offline, via an injected fake `loader`
(never the real `transformers_minimal` fork or `moonbeam_839M.pt`).
**Interface under test:** `MoonBeamBackbone(loader=...).embed(midi_path) -> dict[str, np.ndarray]`

**Files:**
- Create: `model/src/claim_measurement/difficulty/moonbeam_backbone.py`
- Create: `model/src/claim_measurement/difficulty/test_moonbeam_backbone.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for the MoonBeam pooling math, against an injected fake loader --
no transformers_minimal fork, no moonbeam_839M.pt, no isolated venv needed.

Run: cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
"""
from pathlib import Path

import numpy as np

from claim_measurement.difficulty.moonbeam_backbone import MoonBeamBackbone


def test_embed_computes_mean_pool_and_last_token_from_injected_loader():
    hidden_states = np.array([[1.0, 0.0], [3.0, 0.0], [5.0, 6.0]], dtype=np.float32)
    backbone = MoonBeamBackbone(loader=lambda midi_path: hidden_states)

    result = backbone.embed(Path("/fake/piece.mid"))

    assert set(result) == {"mean_pool", "last_token"}
    np.testing.assert_allclose(result["mean_pool"], [3.0, 2.0])
    np.testing.assert_allclose(result["last_token"], [5.0, 6.0])
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_moonbeam_backbone.py -q --no-cov
```
Expected: FAIL — `ModuleNotFoundError: No module named 'claim_measurement.difficulty.moonbeam_backbone'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
"""MoonBeam-839M adapter -- an INTEGRATION SPIKE (issue #138 design). MoonBeam's
pooling API is undocumented and untested until the human GPU run against the
real 839M checkpoint; this class only owns the pooling MATH (mean-over-tokens
vs. last-token), injected behind a `loader` callable so the class is fully
testable without the transformers_minimal fork or moonbeam_839M.pt installed.

The real loader (checkpoint + tokenizer + forward pass) lives in
moonbeam_extract_script.py, which runs under an ISOLATED uv-managed Python
3.12 venv -- see that file's module docstring for the setup recipe. Importing
THIS module never requires that venv; only calling MoonBeamBackbone with a
real loader does.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np


class MoonBeamBackbone:
    """Backbone protocol implementation. `loader(midi_path) -> np.ndarray`
    must return raw per-token hidden states, shape (seq_len, hidden_dim);
    this class only does the pooling, never the checkpoint/tokenizer call."""

    def __init__(self, loader: Callable[[Path], np.ndarray] | None = None):
        if loader is None:
            raise ValueError(
                "MoonBeamBackbone requires an explicit `loader` (real checkpoint "
                "inference lives in moonbeam_extract_script.py, run under the "
                "isolated MoonBeam venv -- see that file's docstring for setup)."
            )
        self._loader = loader

    def embed(self, midi_path: Path) -> dict:
        hidden_states = np.asarray(self._loader(midi_path), dtype=np.float32)
        if hidden_states.ndim != 2:
            raise ValueError(
                f"loader must return (seq_len, hidden_dim) hidden states, "
                f"got shape {hidden_states.shape}"
            )
        return {
            "mean_pool": hidden_states.mean(axis=0),
            "last_token": hidden_states[-1],
        }
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_moonbeam_backbone.py -q --no-cov
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add model/src/claim_measurement/difficulty/moonbeam_backbone.py \
        model/src/claim_measurement/difficulty/test_moonbeam_backbone.py
git commit -m "feat(mirex-difficulty): MoonBeam backbone pooling math (#138)"
```

---

### Task 12b: MoonBeam construction without a loader fails loudly
**Group:** H (sequential, depends on Task 12a, same file)

**Behavior being verified:** Constructing `MoonBeamBackbone()` with no loader
raises immediately with a message naming the isolated-venv requirement,
rather than deferring to a confusing failure deep inside `embed()` — explicit
failure over silent fallback, per this repo's coding rules.
**Interface under test:** `MoonBeamBackbone(loader=None)`

**Files:**
- Modify: `model/src/claim_measurement/difficulty/test_moonbeam_backbone.py`

- [ ] **Step 1: Write the failing test**

Append to `test_moonbeam_backbone.py`:
```python
import pytest


def test_construction_without_loader_fails_loudly():
    with pytest.raises(ValueError, match="isolated MoonBeam venv"):
        MoonBeamBackbone(loader=None)
```

- [ ] **Step 2: Run test — verify it FAILS**

This is expected to already PASS given Task 12a's implementation (the
`ValueError` is already raised in `__init__`). Run it to confirm:

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_moonbeam_backbone.py -k without_loader -q --no-cov
```
If it PASSES immediately, this documents/locks the existing behavior with an
explicit regression test (acceptable — the TDD requirement is that the test
is written and verified meaningful, not that every test must start red when
the behavior it documents was a deliberate side effect of a prior task). If
it FAILS, implement the missing `raise ValueError(...)` in `__init__` per
Task 12a's code before proceeding.

- [ ] **Step 3: Implement the minimum to make the test pass** (only if Step 2 failed)

No change expected.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_moonbeam_backbone.py -q --no-cov
```
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add model/src/claim_measurement/difficulty/test_moonbeam_backbone.py
git commit -m "test(mirex-difficulty): MoonBeamBackbone fails loudly without a loader (#138)"
```

---

### Task 13: MoonBeam isolated-venv extraction script
**Group:** I (depends on Task 10 [extract], Task 12b [moonbeam_backbone], Task 6 [npz], Task 8 [sampling])

**Behavior being verified:** `moonbeam_extract_script.py`'s CLI wires a
loader factory + `MoonBeamBackbone` + `extract_embeddings` together
correctly — verified fully offline via an injected fake `loader_factory`,
never the real MoonBeam fork. This file also carries the isolated-venv setup
documentation (module docstring), satisfying the "MoonBeam deps in an
isolated venv, documented" hard constraint without a new standalone doc file.
**Interface under test:** `main(argv, loader_factory) -> int`

**Files:**
- Create: `model/src/claim_measurement/difficulty/moonbeam_extract_script.py`
- Create: `model/src/claim_measurement/difficulty/test_moonbeam_extract_script.py`

- [ ] **Step 1: Write the failing test**

```python
"""Offline test of moonbeam_extract_script's CLI wiring, via an injected fake
loader_factory -- never touches the real moonbeam fork or checkpoint.

Run: cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
"""
import json

import numpy as np

from claim_measurement.difficulty.bakeoff_npz import read_embedding_npz
from claim_measurement.difficulty.moonbeam_extract_script import main


def test_main_wires_injected_loader_into_extraction(tmp_path):
    sample_manifest = tmp_path / "sample_manifest.json"
    sample_manifest.write_text(json.dumps([
        {"seg_id": "a", "key": "A.mid", "grade": 2, "composer": "Bach"},
    ]))
    out_dir = tmp_path / "emb"
    composer_index = tmp_path / "composer_index.json"

    def fake_loader_factory(checkpoint_path):
        return lambda midi_path: np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    exit_code = main(
        [
            "--checkpoint", str(tmp_path / "fake.pt"),
            "--sample-manifest", str(sample_manifest),
            "--midi-dir", str(tmp_path / "mid"),
            "--out-dir", str(out_dir),
            "--composer-index", str(composer_index),
        ],
        loader_factory=fake_loader_factory,
    )

    assert exit_code == 0
    record = read_embedding_npz(out_dir / "a.npz")
    np.testing.assert_allclose(record.embeddings["mean_pool"], [2.0, 3.0])
    np.testing.assert_allclose(record.embeddings["last_token"], [3.0, 4.0])
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_moonbeam_extract_script.py -q --no-cov
```
Expected: FAIL — `ModuleNotFoundError: No module named 'claim_measurement.difficulty.moonbeam_extract_script'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "numpy>=1.24.0",
#     "torch>=2.0.0",
#     # MoonBeam's own fork -- NOT on PyPI, pinned commit chosen by the human
#     # running the real bake-off (guozixunnicolas/moonbeam-midi-foundation-model,
#     # its bundled transformers_minimal fork + custom tokenizer). Uncomment and
#     # pin once that commit is chosen:
#     # "moonbeam @ git+https://github.com/guozixunnicolas/moonbeam-midi-foundation-model",
# ]
# ///
"""MoonBeam-839M extraction, run under an ISOLATED uv-managed Python 3.12 venv
-- NEVER the shared model/.venv (this repo has twice polluted that shared venv
with a competing pretraining stack's pinned deps; see project memory
project_uv_run_mutates_model_venv.md: "uv run --with X --python N" from
inside model/ rebuilds the shared .venv).

SETUP (human-lit, run once, from this file's own directory):
    cd model/src/claim_measurement/difficulty
    uv run --script moonbeam_extract_script.py --help
    # `uv run --script` resolves THIS file's own `# /// script` metadata block
    # into its own cached, ephemeral env keyed to python==3.12.* + the deps
    # above -- never the project's model/.venv. That is different from a bare
    # `uv run` invoked from inside model/, or `uv run --with X`, both of which
    # DO sync the shared project venv (the known gotcha above). Before the
    # real run: uncomment the moonbeam git dependency once its exact
    # commit/tag is chosen, and implement `_real_loader` below against the
    # fork's actual checkpoint-loading API -- undocumented as of #138 Phase 0,
    # this is the human-lit GPU validation step, not this build's.

RUN (human-lit, needs moonbeam_839M.pt, ~3.3GB, and a GPU):
    uv run --script moonbeam_extract_script.py \
        --checkpoint /path/to/moonbeam_839M.pt \
        --sample-manifest /path/to/model/data/results/bakeoff/sample_manifest.json \
        --midi-dir /path/to/model/data/results/amt_gap_curve/transkun_mid \
        --out-dir /path/to/model/data/results/bakeoff/emb/moonbeam \
        --composer-index /path/to/model/data/results/bakeoff/composer_index.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np


def _real_loader(checkpoint_path: Path) -> Callable[[Path], np.ndarray]:
    """Build the real per-token-hidden-state loader against MoonBeam's fork.

    NOT implemented in this build: the fork's exact checkpoint/tokenizer API
    is undocumented and can only be nailed down by the human running this
    under the isolated venv against the real 3.3GB checkpoint (#138 Phase 0
    design: "the real GPU forward-pass validation is NOT part of this
    build"). Imports of the fork's packages are deferred to inside this
    function so importing this MODULE never requires them.
    """
    raise NotImplementedError(
        "wire this against the transformers_minimal fork's real checkpoint/"
        "tokenizer API once the isolated venv is set up; see this file's "
        "module docstring"
    )


def main(argv: list[str] | None = None, loader_factory=_real_loader) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # -> model/src
    from claim_measurement.difficulty.bakeoff_sampling import ManifestEntry
    from claim_measurement.difficulty.extract import extract_embeddings
    from claim_measurement.difficulty.moonbeam_backbone import MoonBeamBackbone

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--sample-manifest", type=Path, required=True)
    ap.add_argument("--midi-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--composer-index", type=Path, required=True)
    args = ap.parse_args(argv)

    entries = [ManifestEntry(**e) for e in json.loads(args.sample_manifest.read_text())]
    backbone = MoonBeamBackbone(loader=loader_factory(args.checkpoint))
    report = extract_embeddings(backbone, entries, midi_dir=args.midi_dir,
                                 out_dir=args.out_dir, composer_index_path=args.composer_index)
    print(f"ok={report.ok} failed={len(report.failed)}")
    for f in report.failed[:10]:
        print(f"  FAIL {f}")
    return 0 if not report.failed else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_moonbeam_extract_script.py -q --no-cov
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add model/src/claim_measurement/difficulty/moonbeam_extract_script.py \
        model/src/claim_measurement/difficulty/test_moonbeam_extract_script.py
git commit -m "feat(mirex-difficulty): MoonBeam isolated-venv extraction script (#138)"
```

---

### Task 14: `run_bakeoff.py` — sample stage
**Group:** J (sequential, depends on Task 1 [paths], Task 8 [sampling])

**Behavior being verified:** `run_bakeoff.py --stage sample` reads the
manifest+labels+MIDI-dir under `--data-root`, draws the composer-stratified
sample, and writes it to `{data_root}/results/bakeoff/sample_manifest.json`
— the first of the four human-run stages, exercised end to end through the
public CLI.
**Interface under test:** `main(["--stage", "sample", ...]) -> int`

**Files:**
- Create: `model/src/claim_measurement/difficulty/run_bakeoff.py`
- Create: `model/src/claim_measurement/difficulty/test_run_bakeoff.py`

- [ ] **Step 1: Write the failing test**

```python
"""Offline tests for run_bakeoff.py's CLI stage dispatch.

Run: cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
"""
import json
from pathlib import Path

from claim_measurement.difficulty.run_bakeoff import main


def _write_fixture_data(data_root: Path):
    manifest = [
        {"seg_id": "a", "key": "A.mid", "grade": 2, "video_id": "x", "midi_name": "mid/A.mid"},
        {"seg_id": "b", "key": "B.mid", "grade": 5, "video_id": "y", "midi_name": "mid/B.mid"},
    ]
    labels = {"A.mid": {"composer": "Bach"}, "B.mid": {"composer": "Czerny"}}
    (data_root / "results" / "amt_gap_curve").mkdir(parents=True)
    (data_root / "raw" / "psyllabus").mkdir(parents=True)
    (data_root / "results" / "amt_gap_curve" / "manifest.json").write_text(json.dumps(manifest))
    (data_root / "raw" / "psyllabus" / "new_clean_data.json").write_text(json.dumps(labels))
    mid_dir = data_root / "results" / "amt_gap_curve" / "transkun_mid"
    mid_dir.mkdir(parents=True)
    (mid_dir / "a.mid").write_bytes(b"")
    (mid_dir / "b.mid").write_bytes(b"")


def test_sample_stage_writes_sample_manifest(tmp_path):
    _write_fixture_data(tmp_path)

    exit_code = main(["--stage", "sample", "--data-root", str(tmp_path), "--target-n", "2"])

    assert exit_code == 0
    out = json.loads((tmp_path / "results" / "bakeoff" / "sample_manifest.json").read_text())
    assert {e["seg_id"] for e in out} == {"a", "b"}
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_run_bakeoff.py -q --no-cov
```
Expected: FAIL — `ModuleNotFoundError: No module named 'claim_measurement.difficulty.run_bakeoff'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
"""CLI stage dispatch for the frozen backbone bake-off (#138 Phase 0).

Stages:
    sample           -- draw the composer-stratified Transkun sample
    extract-aria      -- human-lit GPU stage (needs real Aria weights); not
                          wired into this offline CLI, run interactively
                          against AriaBackbone (see docs/specs/2026-08-02-
                          backbone-bakeoff-design.md)
    extract-moonbeam  -- points at moonbeam_extract_script.py, which must run
                          under the isolated MoonBeam venv (see that file's
                          docstring)
    eval              -- composer-disjoint tau-c for whichever backbone(s)
                          have extracted embeddings under
                          --data-root/results/bakeoff/emb/{backbone}/

Usage:
    cd model && uv run python -m claim_measurement.difficulty.run_bakeoff \
        --stage sample [--target-n 900] [--data-root PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from claim_measurement.difficulty.bakeoff_cv import oof_tau_ridge
from claim_measurement.difficulty.bakeoff_npz import read_embedding_npz
from claim_measurement.difficulty.bakeoff_paths import resolve_paths
from claim_measurement.difficulty.bakeoff_sampling import (
    composer_stratified_sample,
    load_bakeoff_manifest,
)

N_FOLDS = 5
SEEDS = list(range(2026, 2031))


def _stage_sample(paths, target_n: int) -> None:
    entries = load_bakeoff_manifest(paths.manifest, paths.labels, paths.transkun_mid_dir)
    sample = composer_stratified_sample(entries, target_n, seed=2026)
    paths.emb_root.mkdir(parents=True, exist_ok=True)
    out = paths.emb_root / "sample_manifest.json"
    out.write_text(json.dumps([e.__dict__ for e in sample], indent=2))
    print(f"sampled {len(sample)}/{len(entries)} eligible pieces -> {out}")


def _stage_eval(paths) -> dict:
    """Per-backbone, per-pooling composer-disjoint tau-c, from whatever
    backbone_dir/*.npz files exist under paths.emb_root/emb/."""
    results = {}
    emb_dir = paths.emb_root / "emb"
    if not emb_dir.exists():
        return results
    for backbone_dir in sorted(p for p in emb_dir.glob("*") if p.is_dir()):
        by_pooling: dict = {}
        grades, composer_ids = [], []
        for npz_path in sorted(backbone_dir.glob("*.npz")):
            record = read_embedding_npz(npz_path)
            for pooling_name, vec in record.embeddings.items():
                by_pooling.setdefault(pooling_name, []).append(vec)
            grades.append(record.grade)
            composer_ids.append(record.composer_id)
        y = np.array(grades)
        composers = np.array(composer_ids)
        results[backbone_dir.name] = {
            pooling_name: oof_tau_ridge(np.stack(vecs), y, composers, N_FOLDS, SEEDS)
            for pooling_name, vecs in by_pooling.items()
        }
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["sample", "extract-aria", "extract-moonbeam", "eval"])
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--target-n", type=int, default=900)
    args = ap.parse_args(argv)

    paths = resolve_paths(args.data_root)

    if args.stage == "sample":
        _stage_sample(paths, args.target_n)
    elif args.stage == "extract-aria":
        print("extract-aria: human-lit GPU stage, use claim_measurement.difficulty.aria_backbone.AriaBackbone "
              "+ claim_measurement.difficulty.extract.extract_embeddings directly (see design spec)")
    elif args.stage == "extract-moonbeam":
        print("Run under the isolated MoonBeam venv: see moonbeam_extract_script.py's module docstring")
    elif args.stage == "eval":
        print(json.dumps(_stage_eval(paths), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_run_bakeoff.py -q --no-cov
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add model/src/claim_measurement/difficulty/run_bakeoff.py \
        model/src/claim_measurement/difficulty/test_run_bakeoff.py
git commit -m "feat(mirex-difficulty): run_bakeoff CLI, sample stage (#138)"
```

---

### Task 15: `run_bakeoff.py` — eval stage
**Group:** J (sequential, depends on Task 14, same file)

**Behavior being verified:** `run_bakeoff.py --stage eval` reads every
backbone's `.npz` directory under `{data_root}/results/bakeoff/emb/`, and
prints a JSON summary keyed by backbone name then pooling name, each holding
`{mean, std, n_seeds}` — the decision-rule-ready output the design's item 3
requires ("Outputs per-backbone frozen tau-c (mean/std over seeds)").
**Interface under test:** `main(["--stage", "eval", ...]) -> int` (stdout captured)

**Files:**
- Modify: `model/src/claim_measurement/difficulty/test_run_bakeoff.py`

- [ ] **Step 1: Write the failing test**

Append to `test_run_bakeoff.py`:
```python
import numpy as np

from claim_measurement.difficulty.bakeoff_npz import write_embedding_npz


def test_eval_stage_reports_per_backbone_per_pooling_tau_c(tmp_path, capsys):
    rng = np.random.default_rng(0)
    emb_dir = tmp_path / "results" / "bakeoff" / "emb" / "aria"
    for i in range(12):
        write_embedding_npz(
            emb_dir / f"piece_{i}.npz",
            {"embedding": rng.random(4).astype(np.float32)},
            grade=i % 6,
            composer_id=i % 6,
        )

    exit_code = main(["--stage", "eval", "--data-root", str(tmp_path)])

    assert exit_code == 0
    printed = json.loads(capsys.readouterr().out)
    assert "aria" in printed
    assert set(printed["aria"]["embedding"]) == {"mean", "std", "n_seeds"}
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_run_bakeoff.py -k eval_stage -q --no-cov
```
Expected: FAIL if `_stage_eval`/eval wiring has a bug not caught by Task 14
(Task 14's implementation already includes `_stage_eval` and the `eval`
dispatch branch, so this may already pass). Run it first to confirm which
case applies, same as Task 6's protocol: if it already PASSES, this
documents/locks existing behavior with an explicit test (acceptable per Task
12b's precedent) — skip Step 3.

- [ ] **Step 3: Implement the minimum to make the test pass** (only if Step 2 failed)

No implementation change expected — `_stage_eval` and the `eval` CLI branch
already exist from Task 14. If Step 2 fails, the most likely causes are (a)
`by_pooling` typed as `dict` without a default-factory pattern breaking
`setdefault`, or (b) `results[backbone_dir.name]` computed before `y`/`composers`
are populated for an empty directory — fix by ensuring `_stage_eval` skips a
`backbone_dir` with zero `.npz` files rather than calling `np.stack([])`.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_run_bakeoff.py -q --no-cov
```
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add model/src/claim_measurement/difficulty/test_run_bakeoff.py
git commit -m "test(mirex-difficulty): run_bakeoff eval stage reports per-backbone tau-c (#138)"
```

---

## Final verification (after all 15 tasks)

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
```
Expected: all tests across all 14 files pass, offline, no GPU, no MoonBeam
deps, no real Aria weights.

```bash
cd model && uv run python -m claim_measurement.difficulty.run_bakeoff --stage sample --data-root /Users/jdhiman/Documents/crescendai/model/data --target-n 900
```
This line is documentation of the human-lit next step (real sample draw
against the main checkout's real data) — **not** part of this plan's
executed tasks; do not run it as part of `/build`.

---

## Challenge Review

Verification method: read the plan and spec in full, read every real source
file the plan cites (`aria_embeddings.py`, `pyproject.toml`, `paths.py`,
`score_align/` precedent, `gc_error_bars/` broken precedent), ran the
`score_align` and `gc_error_bars` pytest precondition claims live, diffed
`bakeoff_cv.py`'s ported functions against the actual unmerged
`issue-104-mirex-difficulty` commit `7976b5e6` (`git show`), diffed
`load_bakeoff_manifest`'s filter against the actual unmerged
`issue-137-transkun-features` `tk_ablation.py`, checked the real
`manifest.json`/`new_clean_data.json` field shapes and `transkun_mid/`
naming under the main checkout, and executed every non-trivial algorithm in
the plan (`tau_c`, `composer_disjoint_folds`, `oof_tau_ridge`,
`composer_stratified_sample`) verbatim against the plan's own test fixtures
in a live `uv run python3` REPL. All passed exactly as the plan claims.

### CEO Pass

**Premise Challenge.** Right problem, real pain: memory records #124 (LoRA
single-split fine-tune) already burned a GPU budget on the wrong bet once
("frozen-on-H 0.6785 -> lora-on-H 0.6583" — the fine-tune made things worse
under the honest OOD gate). A cheap frozen-embedding decider before the next
expensive fine-tune is the direct, proportionate fix, not a proxy problem.
No simpler framing was found — a bake-off needs both backbones on the same
protocol, which is exactly what this plan builds.

**Scope Check.** Correctly matches the spec's stated goal and "Not in
scope" boundaries (no Phase-1 trainer code, no re-litigating the deployed
0.824 baseline, no symbolic-feature comparison). Task count (15 tasks, ~20
files including tests) exceeds the ">8 files" complexity-smell threshold,
but every file maps to a genuinely separate, independently-testable concern
(path resolution, CV math, npz I/O, sampling, a protocol+fake, an
orchestrator, two backbone adapters, one CLI) — this granularity matches the
existing `score_align/` precedent (3 files for one concern) scaled to two
backbones plus shared eval math. Not flagged as a real complexity problem.

**Twelve-Month Alignment.**
```
CURRENT STATE                  THIS PLAN                       12-MONTH IDEAL
Aria integrated,           ->  offline-testable decider    ->  LoRA fine-tune on
MoonBeam absent,               harness; human runs it            the winning backbone,
no bake-off protocol            on GPU, picks a winner            shipped to MIREX
exists on this branch
```
Moves toward the ideal; no tech debt created that conflicts with it (frozen
extraction code is throwaway-after-decision by design, not meant to survive
into Phase 1).

**Alternatives Check.**
```
[QUESTION] — The spec's "Not in scope" section implicitly rules out
             alternatives (e.g., skip the bake-off and fine-tune both
             backbones directly, or use a cheaper single-fold proxy instead
             of 5-fold x 5-seed) but does not name them or say why they were
             rejected. Given #124's memory lesson that a single-split
             evaluation already produced a misleading result once, the
             5-fold x 5-seed choice deserves one sentence of "why not
             cheaper" in the spec for a future reader.
```

### Engineering Pass

**Architecture.** Data flow is clean and matches how the code actually
works today:
```
run_bakeoff.py --stage sample
    -> bakeoff_paths.resolve_paths (verified: parents[3] == model/data,
       matches DEFAULT_DATA_ROOT)
    -> bakeoff_sampling.load_bakeoff_manifest (verified: filter logic is an
       exact match of the real, unmerged tk_ablation.py's
       "no midi -> skip; no composer -> skip" — confirmed against
       issue-137-transkun-features's actual file)
    -> bakeoff_sampling.composer_stratified_sample -> sample_manifest.json

[human, GPU] extract-aria / extract-moonbeam
    -> AriaBackbone.embed / MoonBeamBackbone.embed
    -> extract.extract_embeddings (per-piece try/except, loud failure list)
    -> bakeoff_npz.write_embedding_npz -> {seg_id}.npz

run_bakeoff.py --stage eval
    -> bakeoff_npz.read_embedding_npz (per backbone dir, per pooling)
    -> bakeoff_cv.oof_tau_ridge (verified: composer_disjoint_folds,
       RidgeCV/StandardScaler pipeline, and the alphas grid are a verbatim
       port of phase5b_aria_probe.py's _oof_tau/_folds, confirmed via
       `git show 7976b5e6`)
    -> stdout JSON {backbone: {pooling: {mean, std, n_seeds}}}
```
No security surface (no SQL, shell, or LLM-prompt injection paths — all
inputs are local JSON/MIDI files under a human-supplied `--data-root`).

**Module Depth Audit.**
- `bakeoff_paths.py`: 1 function, hides worktree-vs-main-checkout path
  layout. DEEP.
- `bakeoff_cv.py`: 3 functions, hides RidgeCV pipeline + composer
  bin-packing + seed-repeated OOF bookkeeping (verified non-trivial and
  correct by execution). DEEP.
- `bakeoff_npz.py`: 2 functions, hides the pickle-free multi-pooling
  key-prefix scheme. DEEP.
- `bakeoff_sampling.py`: 2 functions, hides join/filter + quota-with-cap
  sampling (verified correct by execution, including the >target_n-composers
  branch). DEEP.
- `backbone.py`: Protocol + `FakeBackbone`. SHALLOW by explicit design (the
  spec says so: "Hides: nothing by design — this is the seam"). Correctly
  not flagged as a problem since it's declared intentional, not accidental.
- `extract.py`: 1 function + report dataclass, hides per-entry
  try/except-and-record, npz writing, composer-id bookkeeping. DEEP.
- `aria_backbone.py` / `moonbeam_backbone.py`: thin adapters, appropriately
  thin given they wrap either an existing verified function
  (`extract_embedding`) or injected test seams. Acceptable, not shallow in
  the harmful sense — they carry real logic (tensor->ndarray conversion;
  mean/last-token pooling math, verified by execution).
- `run_bakeoff.py`: CLI dispatch, appropriately thin — it is intentionally a
  wiring layer over already-deep modules.

**Code Quality.**
```
[RISK] (confidence: 6/10) — extract.py's `except Exception as exc:  # noqa:
       BLE001` is a catch-all, flagged by this repo's own test-philosophy
       standard as a smell. It is explicitly justified in-comment ("record
       and continue; the run report is the source of truth") and matches
       CLAUDE.md's "failures should be loud" rule (failures ARE surfaced —
       via report.failed, printed by run_bakeoff.py, and a non-zero exit
       code from moonbeam_extract_script.py) rather than silently
       swallowed. Acceptable as designed; flagged for awareness, not as a
       blocker.
```

**Test Philosophy / Vertical Slice Audit.** All 15 tasks are one
test(-group)-then-implementation-then-commit; no task writes bulk tests
before any implementation exists. Tasks 6, 12b, and 15 use an explicit
"run first, it may already pass" pattern for behavior that a prior task's
implementation already generalizes to — this is documented and justified
(locks existing behavior with an explicit regression test) rather than
horizontal slicing. All tests exercise public module interfaces; no test
mocks an internal collaborator of the module under test (Task 11's
monkeypatch of `aria_embeddings.extract_embedding` is an *external*
boundary — the real ML weight load — not an internal collaborator of
`AriaBackbone`). No shape-only tests: every test asserts on computed values,
not just presence/type of fields.

**Test Coverage Gaps.**
```
[+] bakeoff_cv.py
    │
    ├── tau_c()
    │   ├── [TESTED] ★★  perfect agreement / disagreement — Task 2
    │   ├── [TESTED] ★★  constant side -> None — Task 2
    │   ├── [TESTED] ★★  <3 points -> None — Task 2
    │   └── [TESTED] ★    ties don't raise — Task 2 (doesn't assert the
    │                     actual tau value, only non-nan)
    ├── composer_disjoint_folds()
    │   ├── [TESTED] ★★★ no-straddle + full coverage — Task 3
    │   └── [GAP]         n_folds > n_composers (some folds necessarily
    │                     empty) — untested; oof_tau_ridge's `len(te)==0:
    │                     continue` guard exists but is never exercised by
    │                     name
    └── oof_tau_ridge()
        ├── [TESTED] ★★  strong linear signal recovered — Task 4 (verified
        │                by live execution: mean=1.0)
        ├── [TESTED] ★★  constant target -> zero seeds — Task 4 (verified:
        │                {"mean": None, "std": None, "n_seeds": 0})
        └── [GAP]         a fold with <3 train rows (the `len(tr) < 3:
                          continue` branch) — untested
```
None of these gaps sit on a critical/irreversible path (no auth, payments,
or data mutation) — this is an offline decision-support harness for one
human's own next GPU spend, not user-facing production code. Not blockers
given the pre-beta/zero-user stage, but worth a follow-up if the eval stage
is ever reused past Phase 0.

**Failure Modes.**
```
[RISK] (confidence: 7/10) — extract_embeddings has no resume/skip-existing
       logic: it does not check whether {seg_id}.npz already exists before
       calling backbone.embed() again. The code being ported
       (phase5b_aria_probe.py's stage_embed) explicitly HAD this
       ("todo = [m for m in manifest if not (EMB_DIR / f'{seg_id}.npz')
       .exists()]", docstring: "Stages (resumable)"), and this property was
       dropped without discussion in the spec's "Key decisions" section.
       For a ~900-piece MoonBeam-839M GPU extraction (the expensive human-
       lit step this harness exists to gate), a crash or interrupt partway
       through means re-extracting everything already done, not just the
       remainder — costly in the exact resource (GPU time) this Phase-0
       gate is designed to conserve. Fallback: the human running the real
       extraction can work around this by pre-filtering `entries` to
       exclude already-written seg_ids before calling extract_embeddings,
       but nothing in run_bakeoff.py or extract.py does this automatically,
       and it isn't mentioned as a known limitation anywhere in the plan or
       spec.
```
```
[OBS] — write_embedding_npz writes directly to the target path via
        np.savez (no write-to-temp-then-rename). A crash mid-write during
        the real GPU run would leave a truncated/corrupt .npz for that one
        piece. Given the RISK above (no resume logic reads existing files
        before re-extracting anyway), this compounds it: even if
        resumability is added later, a corrupt half-written file could
        silently poison a resume-check's "already exists" test unless that
        future fix also validates readability, not just existence.
```

### Presumption Inventory

| ASSUMPTION | VERDICT | REASON |
|---|---|---|
| `phase5b_aria_probe.py`'s tau_c/`_folds`/`_oof_tau` port is behavior-identical | SAFE | Diffed verbatim via `git show 7976b5e6`; matches exactly except the intentional `_folds` -> `composer_disjoint_folds` promotion |
| `tk_ablation.py`'s manifest+composer filter is mirrored exactly | SAFE | Diffed via `git show issue-137-transkun-features`; identical two-condition skip logic |
| `model_improvement.aria_embeddings.extract_embedding(midi_path, variant="embedding") -> torch.Tensor` signature | SAFE | Read the real function; signature matches exactly |
| `claim_measurement.difficulty.*` is importable the same way `claim_measurement.score_align.*` is, despite not being in `pyproject.toml`'s `[tool.hatch.build.targets.wheel] packages` list | SAFE | Ran the real `score_align` test suite live (19/19 passed) proving this import pattern already works for a sibling, unlisted subpackage under the same top-level `claim_measurement` package |
| `gc_error_bars`'s colocated test is a broken flat-import precedent, not to be copied | SAFE | Ran it live; confirmed `ModuleNotFoundError: No module named 'gc_churn_metrics'` exactly as claimed |
| Real `manifest.json`/`new_clean_data.json` field names (`seg_id`, `key`, `grade`, `composer`) match the plan's fixtures | SAFE | Read the real files under the main checkout's `model/data/`; fields match exactly |
| `MoonBeamBackbone`'s real constructor signature (per spec: `checkpoint_path: Path, loader=None`) vs. the plan's actual implementation (`loader` only, no `checkpoint_path` param) | RISKY | See below — genuine spec/plan drift |
| Composer-stratified sampling's quota-with-rounding algorithm hits exactly `target_n` with no duplicates across the tested scales | SAFE | Executed live against the plan's own test fixtures (50x20 and 5x3 composer/piece grids); both assertions passed |
| No resume/skip-existing logic is an acceptable regression from the ported code's resumability | VALIDATE | Real functional gap for the expensive human-lit GPU step this harness exists to protect; not exercised by any test since it's cross-cutting CLI behavior, not a unit |

```
[QUESTION] — Spec section "Modules" (line ~195) documents
             `MoonBeamBackbone.__init__(self, checkpoint_path: Path,
             loader=None)`, but Task 12a's actual implementation is
             `MoonBeamBackbone.__init__(self, loader=None)` with no
             `checkpoint_path` parameter at all — the checkpoint path is
             instead threaded through `moonbeam_extract_script.py`'s
             `loader_factory(checkpoint_path)` closure (Task 13). The
             actual design is arguably cleaner (checkpoint concerns stay
             out of the pooling-only class), but the spec should be
             corrected to match what was actually built, since a future
             reader diffing spec against code will see a false
             discrepancy.
```

### Summary
[BLOCKER] count: 0
[RISK]    count: 3
[QUESTION] count: 2

### VERDICT: PROCEED_WITH_CAUTION — monitor: (1) extract_embeddings' missing resume/skip-existing logic before the human runs the real, expensive MoonBeam-839M GPU extraction — worth a one-line pre-filter fix or an explicit documented limitation before that run, not before this build; (2) the catch-all `except Exception` in extract.py, acceptable as designed but worth keeping an eye on if failure modes widen later; (3) untested `n_folds > n_composers` and `<3-train-rows` fold-guard branches in bakeoff_cv.py, low severity given the non-production, single-human-user stage of this harness.
