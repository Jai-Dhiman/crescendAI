# MIREX Track B Submission System Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** An original music-preference reward model (frozen-encoder + Bradley-Terry head) submittable to MIREX 2026 Track B, plus the generator-identity shortcut probe — built as a nested standalone repo with a synthetic smoke harness proving the pipeline before any real data flows.
**Spec:** docs/specs/2026-07-04-mirex-trackb-build-design.md
**Style:** Python via `uv`, explicit exceptions over fallbacks, no emojis, no backup files (CLAUDE.md).

## Execution environment (READ FIRST — this plan spans two repos)

- **Nested repo (all code):** `/Users/jdhiman/Documents/crescendai/mirex-trackb/` — created by Task 1 as its OWN git repo (not part of crescendai). All Task 1-20 work happens there **on branch `build/initial-system`** (the crescendai PreToolUse hook judges files by their nearest enclosing repo and blocks primary-checkout edits on `main` — including the nested repo's own `main`; the feature branch avoids this). Do NOT create a crescendai worktree for Tasks 1-20; `cd /Users/jdhiman/Documents/crescendai/mirex-trackb` is the working directory.
- **Crescendai side (Task 21 only):** the existing worktree `/Users/jdhiman/Documents/crescendai/.worktrees/issue-106-mirex-trackb` (branch `issue-106-mirex-trackb`).
- **Test command everywhere:** `uv run pytest <file> -x -q` from the nested repo root.
- **Calendar note:** after Task 19 lands, the user can start real overnight CLAP extraction of CMI-Pref — prioritize the path Task 1 → 2 → 8 → 16 → 19 if wall-clock gets tight before the ~Jul 24 fork decision.

## Shared contracts (single source of truth — every task conforms to these signatures)

- `PairRecord(pair_id, source, prompt, lyrics, ref_audio, clip_a, clip_b, label, generator_a, generator_b, confidence, modality)` — frozen dataclass; `label` in `{"A","B"}`; `clip_a`/`clip_b`/`ref_audio` are corpus-scoped string clip_refs.
- `FeatureStore(root)` with `put(clip_key, encoder, vec) -> bool`, `get(clip_key, encoder) -> np.ndarray`, `write_manifest(encoder, corpus, entries: dict[clip_ref, clip_key], dim: int)`, `verify(encoder, corpus) -> dict[clip_ref, clip_key]`.
- Text context convention: prompt embeddings are stored in the SAME store under manifest corpus name `f"{corpus}__text"`, clip_ref = the pair's `pair_id`, clip_key = `sha256("text:" + prompt).hexdigest()`.
- `BTHead(audio_dim, ctx_dim, width, depth, joint)` with `margin(feat_a, feat_b, ctx) -> Tensor` (antisymmetric by construction in both modes).
- `TrainConfig(encoder, head_width, head_depth, joint, lr, epochs, batch_size, sampler, adversarial_lambda, seed, k_folds, final_generators)`; `train(config, pairs, store, corpus) -> TrainResult(checkpoint_path, metrics)`.
- Checkpoint file = `torch.save({"state_dict", "config" (asdict), "audio_dim", "ctx_dim", "encoder"})`.
- Toy encoder: name `"toy"`, dim 16, deterministic (content-hash-seeded); exists so CLI/e2e tests run with zero GPU/downloads.

## Task Groups

```
Group 0 (sequential):             Task 1 (scaffold)
Group A (parallel, needs 0):      Task 2 (schema), Task 3 (store), Task 4 (budget), Task 5 (ratchet)
Group B (parallel, needs A):      Task 6 (folds), Task 7 (fixtures), Task 8 (encoders)
Group C (sequential, needs B):    Task 9 (pointwise head), Task 10 (joint head), Task 11 (trainer), Task 12 (smoke)
Group D (sequential, needs C):    Task 13 (generator-ID probe), Task 14 (decompose), Task 15 (mitigation)
Group E (parallel, needs A only): Task 16 (cmi_pref), Task 17 (aime_survey), Task 18 (music_arena)
Group F (sequential, needs C+E):  Task 19 (extract CLI), Task 20 (main.py contract)
Group G (sequential, needs F):    Task 21 (crescendai gitignore + living doc + real config + README)
```

Group E may be dispatched concurrently with Groups C/D (it depends only on Group A and touches disjoint files).

**Decouple check:** Group 0-C is the smoke-validated training spine `[SHIPS INDEPENDENTLY]` — with it the user can already point /autoresearch at fixture sweeps to validate the loop mechanics. Groups E+F unlock the real wave-1 extraction (`just extract`) — the artifact the ~Jul 24 fork decision needs.

---

### Task 1: Repo scaffold
**Group:** 0

**Behavior being verified:** the nested repo exists as its own git repo, `uv run pytest` collects and passes a trivial import test.
**Interface under test:** `import trackb`.

**Files:**
- Create: `mirex-trackb/pyproject.toml`, `mirex-trackb/justfile`, `mirex-trackb/.gitignore`, `mirex-trackb/LICENSE`, `mirex-trackb/README.md`, `mirex-trackb/src/trackb/__init__.py`
- Test: `mirex-trackb/tests/test_scaffold.py`

- [x] **Step 1: Create the repo and write the failing test**

```bash
mkdir -p /Users/jdhiman/Documents/crescendai/mirex-trackb
cd /Users/jdhiman/Documents/crescendai/mirex-trackb
git init -b main
git checkout -b build/initial-system
mkdir -p src/trackb tests
touch tests/__init__.py   # tests is a package: later tasks import tests.fixtures.*
```

```python
# tests/test_scaffold.py
def test_package_imports():
    import trackb
    assert trackb.__version__ == "0.1.0"
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/mirex-trackb && uv run pytest tests/test_scaffold.py -x -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'trackb'` (or uv errors on missing pyproject; create pyproject first in that case and re-run to see the import failure).

- [x] **Step 3: Implement the minimum to make the test pass**

```toml
# pyproject.toml
[project]
name = "trackb"
version = "0.1.0"
description = "MIREX 2026 Track B: frozen-encoder Bradley-Terry music preference model + generator-shortcut study"
requires-python = ">=3.11"
license = {text = "Apache-2.0"}
dependencies = [
    "numpy>=1.26",
    "torch>=2.2",
    "soundfile>=0.12",
    "soxr>=0.4",
    "datasets>=2.19",
    "transformers>=4.40",
    "scikit-learn>=1.4",
]

[dependency-groups]
dev = ["pytest>=8.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/trackb"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

```python
# src/trackb/__init__.py
__version__ = "0.1.0"
```

```gitignore
# .gitignore
.venv/
__pycache__/
*.pyc
data/
checkpoints/
results/
tests/fixtures/generated/
.worktrees/
```

```justfile
# justfile
test:
    uv run pytest -x -q
```

```markdown
# README.md
# trackb

MIREX 2026 Track B (CMI-RewardBench) submission: frozen-encoder + Bradley-Terry
preference head, with a generator-identity shortcut study. Apache-2.0 code;
training data licenses documented per-corpus in src/trackb/corpora/.

Run `python main.py --path input.jsonl` for the MIREX contract (available after
the submit module lands). `just test` runs the behavior suite (no GPU needed).
```

```bash
curl -s https://www.apache.org/licenses/LICENSE-2.0.txt -o LICENSE
```

- [x] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/mirex-trackb && uv run pytest tests/test_scaffold.py -x -q
```
Expected: PASS

- [x] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai/mirex-trackb && git add -A && git commit -m "feat(scaffold): uv project skeleton with trackb package"
```

---

### Task 2: PairRecord schema + validation
**Group:** A (parallel with Tasks 3, 4, 5)

**Behavior being verified:** `validate_pairs` accepts valid records and raises `SchemaError` naming the field and source on each violation class (missing generator, bad label, duplicate pair_id, empty clip ref).
**Interface under test:** `trackb.schema.validate_pairs`, `trackb.schema.PairRecord`.

**Files:**
- Create: `src/trackb/schema.py`
- Test: `tests/test_schema.py`

- [x] **Step 1: Write the failing test**

```python
# tests/test_schema.py
import dataclasses
import pytest
from trackb.schema import PairRecord, SchemaError, validate_pairs


def make(**overrides):
    base = dict(
        pair_id="p1", source="fixture", prompt="a calm piano piece",
        lyrics=None, ref_audio=None, clip_a="fx0a", clip_b="fx0b",
        label="A", generator_a="g0", generator_b="g1",
        confidence=1.0, modality=None,
    )
    base.update(overrides)
    return PairRecord(**base)


def test_valid_pairs_pass_through():
    records = [make(), make(pair_id="p2", label="B")]
    assert validate_pairs(records) == records


def test_missing_generator_raises_naming_field_and_source():
    with pytest.raises(SchemaError, match=r"generator_b.*fixture"):
        validate_pairs([make(pair_id="p3", generator_b="")])


def test_bad_label_raises():
    with pytest.raises(SchemaError, match=r"label"):
        validate_pairs([make(label="model_a")])


def test_duplicate_pair_id_raises():
    with pytest.raises(SchemaError, match=r"duplicate.*p1"):
        validate_pairs([make(), make(clip_a="fx9a")])


def test_empty_clip_ref_raises():
    with pytest.raises(SchemaError, match=r"clip_a"):
        validate_pairs([make(clip_a="")])


def test_record_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        make().label = "B"
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
uv run pytest tests/test_schema.py -x -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'trackb.schema'`

- [x] **Step 3: Implement the minimum to make the test pass**

```python
# src/trackb/schema.py
"""Canonical pair schema. Every corpus adapter yields PairRecord; every
invariant lives HERE so adapters stay dumb translation layers."""
from dataclasses import dataclass
from typing import Iterable

VALID_LABELS = {"A", "B"}


class SchemaError(ValueError):
    pass


@dataclass(frozen=True)
class PairRecord:
    pair_id: str
    source: str
    prompt: str
    lyrics: str | None
    ref_audio: str | None
    clip_a: str
    clip_b: str
    label: str
    generator_a: str
    generator_b: str
    confidence: float | None
    modality: str | None


_REQUIRED_NONEMPTY = ("pair_id", "source", "prompt", "clip_a", "clip_b",
                      "generator_a", "generator_b")


def validate_pairs(records: Iterable[PairRecord]) -> list[PairRecord]:
    out: list[PairRecord] = []
    seen: set[str] = set()
    for r in records:
        for field in _REQUIRED_NONEMPTY:
            if not getattr(r, field):
                raise SchemaError(
                    f"empty required field '{field}' in pair '{r.pair_id}' "
                    f"from source '{r.source}'"
                )
        if r.label not in VALID_LABELS:
            raise SchemaError(
                f"label must be one of {sorted(VALID_LABELS)}, got '{r.label}' "
                f"in pair '{r.pair_id}' from source '{r.source}'"
            )
        if r.pair_id in seen:
            raise SchemaError(f"duplicate pair_id '{r.pair_id}'")
        seen.add(r.pair_id)
        out.append(r)
    return out
```

- [x] **Step 4: Run test — verify it PASSES**

```bash
uv run pytest tests/test_schema.py -x -q
```
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add src/trackb/schema.py tests/test_schema.py && git commit -m "feat(schema): PairRecord + validate_pairs with loud per-field failures"
```

---

### Task 3: FeatureStore
**Group:** A (parallel with Tasks 2, 4, 5)

**Behavior being verified:** put/get roundtrip; manifest verify passes when complete and raises `ManifestError` naming the missing clip_ref when a feature file is gone; re-put of an existing key is skipped (idempotent resume).
**Interface under test:** `trackb.extract.store.FeatureStore`.

**Files:**
- Create: `src/trackb/extract/__init__.py`, `src/trackb/extract/store.py`
- Test: `tests/test_store.py`

- [x] **Step 1: Write the failing test**

```python
# tests/test_store.py
import numpy as np
import pytest
from trackb.extract.store import FeatureStore, ManifestError


def test_put_get_roundtrip(tmp_path):
    store = FeatureStore(tmp_path)
    vec = np.arange(8, dtype=np.float32)
    assert store.put("abc123", "toy", vec) is True
    got = store.get("abc123", "toy")
    np.testing.assert_array_equal(got, vec)
    assert got.dtype == np.float32


def test_re_put_is_skipped(tmp_path):
    store = FeatureStore(tmp_path)
    store.put("abc123", "toy", np.zeros(8, dtype=np.float32))
    assert store.put("abc123", "toy", np.ones(8, dtype=np.float32)) is False
    np.testing.assert_array_equal(store.get("abc123", "toy"), np.zeros(8))


def test_verify_returns_entries_when_complete(tmp_path):
    store = FeatureStore(tmp_path)
    store.put("k1", "toy", np.zeros(4, dtype=np.float32))
    store.put("k2", "toy", np.zeros(4, dtype=np.float32))
    store.write_manifest("toy", "fixture", {"clipA": "k1", "clipB": "k2"}, dim=4)
    assert store.verify("toy", "fixture") == {"clipA": "k1", "clipB": "k2"}


def test_verify_raises_naming_missing_clip(tmp_path):
    store = FeatureStore(tmp_path)
    store.put("k1", "toy", np.zeros(4, dtype=np.float32))
    store.write_manifest("toy", "fixture", {"clipA": "k1", "clipB": "kGONE"}, dim=4)
    with pytest.raises(ManifestError, match=r"clipB"):
        store.verify("toy", "fixture")


def test_verify_raises_on_missing_manifest(tmp_path):
    with pytest.raises(ManifestError, match=r"toy.*fixture"):
        FeatureStore(tmp_path).verify("toy", "fixture")
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
uv run pytest tests/test_store.py -x -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'trackb.extract'`

- [x] **Step 3: Implement the minimum to make the test pass**

```python
# src/trackb/extract/__init__.py
```

```python
# src/trackb/extract/store.py
"""Content-hash-keyed feature store with verified manifests.

Layout:  root/features/{encoder}/{key[:2]}/{key}.npy
         root/manifests/{encoder}__{corpus}.json
Reads must go through verify(): an unverified or incomplete manifest refuses
assembly, naming the missing clip. put() is idempotent (resume-safe)."""
import json
from pathlib import Path

import numpy as np


class ManifestError(RuntimeError):
    pass


class FeatureStore:
    def __init__(self, root: Path | str):
        self.root = Path(root)

    def _vec_path(self, clip_key: str, encoder: str) -> Path:
        return self.root / "features" / encoder / clip_key[:2] / f"{clip_key}.npy"

    def _manifest_path(self, encoder: str, corpus: str) -> Path:
        return self.root / "manifests" / f"{encoder}__{corpus}.json"

    def put(self, clip_key: str, encoder: str, vec: np.ndarray) -> bool:
        path = self._vec_path(clip_key, encoder)
        if path.exists():
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vec.astype(np.float32))
        return True

    def get(self, clip_key: str, encoder: str) -> np.ndarray:
        return np.load(self._vec_path(clip_key, encoder))

    def write_manifest(self, encoder: str, corpus: str,
                       entries: dict[str, str], dim: int) -> None:
        path = self._manifest_path(encoder, corpus)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {"encoder": encoder, "corpus": corpus, "dim": dim,
             "count": len(entries), "entries": entries}, indent=1))

    def verify(self, encoder: str, corpus: str) -> dict[str, str]:
        path = self._manifest_path(encoder, corpus)
        if not path.exists():
            raise ManifestError(
                f"no manifest for encoder '{encoder}' corpus '{corpus}' "
                f"at {path} - run extraction first")
        manifest = json.loads(path.read_text())
        missing = [ref for ref, key in manifest["entries"].items()
                   if not self._vec_path(key, encoder).exists()]
        if missing:
            raise ManifestError(
                f"manifest {encoder}__{corpus}: {len(missing)} feature file(s) "
                f"missing, first: '{missing[0]}' - re-run extraction")
        return manifest["entries"]
```

- [x] **Step 4: Run test — verify it PASSES**

```bash
uv run pytest tests/test_store.py -x -q
```
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add src/trackb/extract tests/test_store.py && git commit -m "feat(extract): content-hash FeatureStore with verified manifests"
```

---

### Task 4: Public-eval budget guard
**Group:** A (parallel with Tasks 2, 3, 5)

**Behavior being verified:** the 6th spend raises `BudgetExhausted`; spends persist across instances (a new process cannot reset the counter).
**Interface under test:** `trackb.evalx.budget.PublicEvalBudget`.

**Files:**
- Create: `src/trackb/evalx/__init__.py`, `src/trackb/evalx/budget.py`
- Test: `tests/test_budget.py`

- [x] **Step 1: Write the failing test**

```python
# tests/test_budget.py
import pytest
from trackb.evalx.budget import BudgetExhausted, PublicEvalBudget


def test_five_spends_then_refusal(tmp_path):
    ledger = tmp_path / "budget.json"
    budget = PublicEvalBudget(ledger, limit=5)
    for i in range(5):
        assert budget.spend(f"reason {i}") == i + 1
    with pytest.raises(BudgetExhausted, match=r"5"):
        budget.spend("one too many")


def test_spends_persist_across_instances(tmp_path):
    ledger = tmp_path / "budget.json"
    PublicEvalBudget(ledger, limit=2).spend("first")
    second = PublicEvalBudget(ledger, limit=2)
    assert second.remaining == 1
    second.spend("second")
    with pytest.raises(BudgetExhausted):
        PublicEvalBudget(ledger, limit=2).spend("third")


def test_reason_is_required(tmp_path):
    with pytest.raises(ValueError, match=r"reason"):
        PublicEvalBudget(tmp_path / "b.json", limit=5).spend("")
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
uv run pytest tests/test_budget.py -x -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'trackb.evalx'`

- [x] **Step 3: Implement the minimum to make the test pass**

```python
# src/trackb/evalx/__init__.py
```

```python
# src/trackb/evalx/budget.py
"""Hard counter on public CMI-Pref-500 evaluations. The public split is spent,
not used: every evaluation is logged with a reason and the limit is refused
loudly. Guards the campaign against silent test-set overfitting."""
import json
from pathlib import Path


class BudgetExhausted(RuntimeError):
    pass


class PublicEvalBudget:
    def __init__(self, ledger_path: Path | str, limit: int = 5):
        self.path = Path(ledger_path)
        self.limit = limit

    def _spends(self) -> list[str]:
        if not self.path.exists():
            return []
        return json.loads(self.path.read_text())["spends"]

    @property
    def remaining(self) -> int:
        return self.limit - len(self._spends())

    def spend(self, reason: str) -> int:
        if not reason:
            raise ValueError("a non-empty reason is required to spend a public eval")
        spends = self._spends()
        if len(spends) >= self.limit:
            raise BudgetExhausted(
                f"public-eval budget exhausted: {self.limit}/{self.limit} used. "
                f"Reasons so far: {spends}")
        spends.append(reason)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"spends": spends}, indent=1))
        return len(spends)
```

- [x] **Step 4: Run test — verify it PASSES**

```bash
uv run pytest tests/test_budget.py -x -q
```
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add src/trackb/evalx tests/test_budget.py && git commit -m "feat(evalx): public-eval budget guard refusing spend six"
```

---

### Task 5: Ratchet
**Group:** A (parallel with Tasks 2, 3, 4)

**Behavior being verified:** `check` passes on improvement and raises `RatchetError` on regression beyond epsilon; `promote` copies last_run to baseline; a missing baseline passes (first run bootstraps).
**Interface under test:** `trackb.evalx.ratchet.check`, `trackb.evalx.ratchet.promote`.

**Files:**
- Create: `src/trackb/evalx/ratchet.py`
- Test: `tests/test_ratchet.py`

- [x] **Step 1: Write the failing test**

```python
# tests/test_ratchet.py
import json
import pytest
from trackb.evalx.ratchet import RatchetError, check, promote


def write(path, acc):
    path.write_text(json.dumps({"holdout_acc": acc}))


def test_improvement_passes(tmp_path):
    write(tmp_path / "baseline.json", 0.70)
    write(tmp_path / "last_run.json", 0.75)
    result = check(tmp_path / "last_run.json", tmp_path / "baseline.json")
    assert result["delta"] == pytest.approx(0.05)


def test_regression_raises(tmp_path):
    write(tmp_path / "baseline.json", 0.75)
    write(tmp_path / "last_run.json", 0.70)
    with pytest.raises(RatchetError, match=r"regressed"):
        check(tmp_path / "last_run.json", tmp_path / "baseline.json")


def test_regression_within_epsilon_passes(tmp_path):
    write(tmp_path / "baseline.json", 0.750)
    write(tmp_path / "last_run.json", 0.748)
    check(tmp_path / "last_run.json", tmp_path / "baseline.json", epsilon=0.005)


def test_missing_baseline_bootstraps(tmp_path):
    write(tmp_path / "last_run.json", 0.70)
    result = check(tmp_path / "last_run.json", tmp_path / "baseline.json")
    assert result["baseline"] is None


def test_promote_copies(tmp_path):
    write(tmp_path / "last_run.json", 0.77)
    promote(tmp_path / "last_run.json", tmp_path / "baseline.json")
    assert json.loads((tmp_path / "baseline.json").read_text())["holdout_acc"] == 0.77
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
uv run pytest tests/test_ratchet.py -x -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'trackb.evalx.ratchet'`

- [x] **Step 3: Implement the minimum to make the test pass**

```python
# src/trackb/evalx/ratchet.py
"""baseline.json / last_run.json compare-and-promote (the crescendai
chroma-eval idiom). Promotion is a deliberate act, never automatic."""
import json
import shutil
from pathlib import Path


class RatchetError(RuntimeError):
    pass


def check(last_run: Path | str, baseline: Path | str,
          metric: str = "holdout_acc", epsilon: float = 0.0) -> dict:
    last = json.loads(Path(last_run).read_text())[metric]
    baseline_path = Path(baseline)
    if not baseline_path.exists():
        return {"metric": metric, "last_run": last, "baseline": None, "delta": None}
    base = json.loads(baseline_path.read_text())[metric]
    delta = last - base
    if delta < -epsilon:
        raise RatchetError(
            f"{metric} regressed: {last:.4f} vs baseline {base:.4f} "
            f"(delta {delta:+.4f}, epsilon {epsilon})")
    return {"metric": metric, "last_run": last, "baseline": base, "delta": delta}


def promote(last_run: Path | str, baseline: Path | str) -> None:
    shutil.copyfile(last_run, baseline)
```

- [x] **Step 4: Run test — verify it PASSES**

```bash
uv run pytest tests/test_ratchet.py -x -q
```
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add src/trackb/evalx/ratchet.py tests/test_ratchet.py && git commit -m "feat(evalx): baseline/last_run ratchet with epsilon"
```

---

### Task 6: Generator-holdout folds
**Group:** B (parallel with Tasks 7, 8)

**Behavior being verified:** fold purity — no rotating fold's training pair touches that fold's held-out generators; final-fold generators appear in NO rotating fold at all; every non-final pair appears in at least one validation fold.
**Interface under test:** `trackb.evalx.folds.make_folds`.

**Files:**
- Create: `src/trackb/evalx/folds.py`
- Test: `tests/test_folds.py`

- [x] **Step 1: Write the failing test**

```python
# tests/test_folds.py
import pytest
from trackb.schema import PairRecord
from trackb.evalx.folds import make_folds


def pair(i, ga, gb):
    return PairRecord(
        pair_id=f"p{i}", source="fixture", prompt="x", lyrics=None,
        ref_audio=None, clip_a=f"c{i}a", clip_b=f"c{i}b", label="A",
        generator_a=ga, generator_b=gb, confidence=None, modality=None)


def build_pairs():
    gens = ["g0", "g1", "g2", "g3"]
    pairs, i = [], 0
    for a in gens:
        for b in gens:
            if a != b:
                for _ in range(4):
                    pairs.append(pair(i, a, b)); i += 1
    return pairs


def test_rotating_fold_purity():
    folds = make_folds(build_pairs(), k=3, final_generators={"g3"}, seed=0)
    assert len(folds.rotating) == 3
    for train, val in folds.rotating:
        val_gens = {g for p in val for g in (p.generator_a, p.generator_b)}
        held = val_gens - {g for p in train for g in (p.generator_a, p.generator_b)}
        assert held, "each fold must hold out at least one generator entirely"
        for p in train:
            assert p.generator_a not in held and p.generator_b not in held


def test_final_generators_never_in_rotating_folds():
    folds = make_folds(build_pairs(), k=3, final_generators={"g3"}, seed=0)
    for train, val in folds.rotating:
        for p in train + val:
            assert "g3" not in (p.generator_a, p.generator_b)
    assert all("g3" in (p.generator_a, p.generator_b) for p in folds.final_holdout)
    for p in folds.final_train:
        assert "g3" not in (p.generator_a, p.generator_b)


def test_unknown_final_generator_raises():
    with pytest.raises(ValueError, match=r"gX"):
        make_folds(build_pairs(), k=2, final_generators={"gX"}, seed=0)
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
uv run pytest tests/test_folds.py -x -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'trackb.evalx.folds'`

- [x] **Step 3: Implement the minimum to make the test pass**

```python
# src/trackb/evalx/folds.py
"""Generator-holdout fold construction. A pair is held out if EITHER clip's
generator is held out. Rotating folds are what sweeps/autoresearch optimize;
the final fold's generators are excluded from ALL rotating folds and spent
only on submission decisions."""
import random
from dataclasses import dataclass

from trackb.schema import PairRecord


@dataclass
class Folds:
    rotating: list[tuple[list[PairRecord], list[PairRecord]]]
    final_train: list[PairRecord]
    final_holdout: list[PairRecord]


def _generators(pairs: list[PairRecord]) -> set[str]:
    return {g for p in pairs for g in (p.generator_a, p.generator_b)}


def make_folds(pairs: list[PairRecord], k: int,
               final_generators: set[str], seed: int = 0) -> Folds:
    all_gens = _generators(pairs)
    unknown = final_generators - all_gens
    if unknown:
        raise ValueError(f"final_generators not present in pairs: {sorted(unknown)}")

    touches_final = [p for p in pairs
                     if p.generator_a in final_generators
                     or p.generator_b in final_generators]
    rest = [p for p in pairs if p not in touches_final]
    rest_gens = sorted(_generators(rest) - final_generators)

    rng = random.Random(seed)
    rng.shuffle(rest_gens)
    groups = [set(rest_gens[i::k]) for i in range(k)]

    rotating = []
    for held in groups:
        val = [p for p in rest
               if p.generator_a in held or p.generator_b in held]
        train = [p for p in rest
                 if p.generator_a not in held and p.generator_b not in held]
        rotating.append((train, val))
    return Folds(rotating=rotating, final_train=rest, final_holdout=touches_final)
```

- [x] **Step 4: Run test — verify it PASSES**

```bash
uv run pytest tests/test_folds.py -x -q
```
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add src/trackb/evalx/folds.py tests/test_folds.py && git commit -m "feat(evalx): rotating generator-holdout folds with excluded final fold"
```

---

### Task 7: Synthetic fixture corpus with planted signal
**Group:** B (parallel with Tasks 6, 8)

**Behavior being verified:** the fixture generator produces a schema-valid mini-corpus whose labels follow a planted quality ordering, with generator identity also encoded in the features (so the shortcut probe has signal), plus a populated toy FeatureStore including prompt-text context vectors.
**Interface under test:** `tests.fixtures.make_fixtures.build_fixture_corpus`.

**Files:**
- Create: `tests/fixtures/__init__.py`, `tests/fixtures/make_fixtures.py`
- Test: `tests/test_fixtures.py`

- [x] **Step 1: Write the failing test**

```python
# tests/test_fixtures.py
import numpy as np
from trackb.schema import validate_pairs
from trackb.extract.store import FeatureStore
from tests.fixtures.make_fixtures import QUALITY, build_fixture_corpus


def test_fixture_corpus_is_schema_valid_and_planted(tmp_path):
    pairs, store = build_fixture_corpus(tmp_path, n_pairs=48, seed=7)
    validate_pairs(pairs)
    assert len(pairs) == 48
    # planted signal: label always matches the quality ordering
    for p in pairs:
        expected = "A" if QUALITY[p.generator_a] > QUALITY[p.generator_b] else "B"
        assert p.label == expected
    # features exist for every clip and every prompt context
    entries = store.verify("toy", "fixture")
    ctx_entries = store.verify("toy", "fixture__text")
    clip_refs = {p.clip_a for p in pairs} | {p.clip_b for p in pairs}
    assert clip_refs == set(entries.keys())
    assert {p.pair_id for p in pairs} == set(ctx_entries.keys())
    vec = store.get(entries[pairs[0].clip_a], "toy")
    assert vec.shape == (16,) and vec.dtype == np.float32


def test_fixture_is_deterministic(tmp_path):
    pairs1, _ = build_fixture_corpus(tmp_path / "a", n_pairs=48, seed=7)
    pairs2, _ = build_fixture_corpus(tmp_path / "b", n_pairs=48, seed=7)
    assert [p.pair_id for p in pairs1] == [p.pair_id for p in pairs2]
    assert [p.label for p in pairs1] == [p.label for p in pairs2]
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
uv run pytest tests/test_fixtures.py -x -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'tests.fixtures.make_fixtures'`

- [x] **Step 3: Implement the minimum to make the test pass**

```python
# tests/fixtures/__init__.py
```

```python
# tests/fixtures/make_fixtures.py
"""Synthetic mini-corpus with a PLANTED preference signal.

4 fake generators g0..g3 with fixed quality ranks. A clip's feature vector is
quality * u_quality + one_hot(generator) * 0.5 + noise, so:
  - a head can learn preference from features (smoke pipeline target > 0.9)
  - a generator-ID probe can recover generator identity (shortcut probe target)
Labels follow quality exactly. Deterministic for a given seed."""
from pathlib import Path

import numpy as np

from trackb.extract.store import FeatureStore
from trackb.schema import PairRecord

DIM = 16
QUALITY = {"g0": 3.0, "g1": 2.0, "g2": 1.0, "g3": 0.0}
GENERATORS = list(QUALITY)


def _clip_vec(rng, generator: str) -> np.ndarray:
    u_quality = np.zeros(DIM, dtype=np.float32)
    u_quality[:4] = 0.5
    one_hot = np.zeros(DIM, dtype=np.float32)
    one_hot[4 + GENERATORS.index(generator)] = 0.5
    noise = rng.standard_normal(DIM).astype(np.float32) * 0.1
    return QUALITY[generator] * u_quality + one_hot + noise


def build_fixture_corpus(root: Path, n_pairs: int = 48,
                         seed: int = 7) -> tuple[list[PairRecord], FeatureStore]:
    rng = np.random.default_rng(seed)
    store = FeatureStore(Path(root))
    pairs: list[PairRecord] = []
    clip_entries: dict[str, str] = {}
    ctx_entries: dict[str, str] = {}
    for i in range(n_pairs):
        ga, gb = rng.choice(GENERATORS, size=2, replace=False)
        ga, gb = str(ga), str(gb)
        ref_a, ref_b = f"fx{i}a", f"fx{i}b"
        key_a, key_b = f"key_{ref_a}", f"key_{ref_b}"
        store.put(key_a, "toy", _clip_vec(rng, ga))
        store.put(key_b, "toy", _clip_vec(rng, gb))
        clip_entries[ref_a], clip_entries[ref_b] = key_a, key_b
        pair_id = f"fx{i}"
        ctx_key = f"ctx_{pair_id}"
        store.put(ctx_key, "toy", rng.standard_normal(DIM).astype(np.float32))
        ctx_entries[pair_id] = ctx_key
        pairs.append(PairRecord(
            pair_id=pair_id, source="fixture", prompt=f"prompt {i}",
            lyrics=None, ref_audio=None, clip_a=ref_a, clip_b=ref_b,
            label="A" if QUALITY[ga] > QUALITY[gb] else "B",
            generator_a=ga, generator_b=gb, confidence=1.0, modality=None))
    store.write_manifest("toy", "fixture", clip_entries, dim=DIM)
    store.write_manifest("toy", "fixture__text", ctx_entries, dim=DIM)
    return pairs, store
```

- [x] **Step 4: Run test — verify it PASSES**

```bash
uv run pytest tests/test_fixtures.py -x -q
```
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add tests/fixtures tests/test_fixtures.py && git commit -m "feat(fixtures): synthetic planted-signal mini-corpus + toy store"
```

---

### Task 8: Encoder registry + audio loading
**Group:** B (parallel with Tasks 6, 7)

**Behavior being verified:** unknown encoder names are rejected; the toy encoder is deterministic per content; `load_audio` decodes and resamples a real wav file via soundfile (never torchaudio decode — the P0 torchcodec ABI gotcha).
**Interface under test:** `trackb.extract.encoders.embed`, `text_embed`, `load_audio`.

**Files:**
- Create: `src/trackb/extract/encoders.py`
- Test: `tests/test_encoders.py`

- [x] **Step 1: Write the failing test**

```python
# tests/test_encoders.py
import numpy as np
import pytest
import soundfile as sf
from trackb.extract.encoders import UnknownEncoderError, embed, load_audio, text_embed


def test_unknown_encoder_raises():
    with pytest.raises(UnknownEncoderError, match=r"nope"):
        embed("nope", np.zeros(100, dtype=np.float32), 16000)


def test_toy_embed_is_deterministic_and_content_sensitive():
    wav1 = np.linspace(-1, 1, 1000, dtype=np.float32)
    wav2 = np.linspace(-1, 0.5, 1000, dtype=np.float32)
    a = embed("toy", wav1, 16000)
    b = embed("toy", wav1, 16000)
    c = embed("toy", wav2, 16000)
    assert a.shape == (16,) and a.dtype == np.float32
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_toy_text_embed_deterministic():
    a = text_embed("toy", "a calm piano piece")
    b = text_embed("toy", "a calm piano piece")
    c = text_embed("toy", "different prompt")
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_load_audio_decodes_and_resamples(tmp_path):
    sr = 22050
    t = np.linspace(0, 1, sr, dtype=np.float32)
    stereo = np.stack([np.sin(2 * np.pi * 440 * t)] * 2, axis=1)
    path = tmp_path / "tone.wav"
    sf.write(path, stereo, sr)
    wav, out_sr = load_audio(path, target_sr=16000)
    assert out_sr == 16000
    assert wav.ndim == 1
    assert abs(len(wav) - 16000) <= 2
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
uv run pytest tests/test_encoders.py -x -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'trackb.extract.encoders'`

- [x] **Step 3: Implement the minimum to make the test pass**

```python
# src/trackb/extract/encoders.py
"""Encoder registry. All audio decode goes through soundfile (the torchcodec
0.14 / torch 2.10 macOS ABI break makes torchaudio decode unusable - P0
gotcha #3). Real encoders lazy-import their heavy deps so the test suite and
toy pipeline never download models.

Registered encoders:
  toy       - 16-d deterministic content-hash vectors (tests / smoke / CI)
  clap      - laion/larger_clap_music, 512-d audio + text     (Apache-2.0)
  mert      - m-a-p/MERT-v1-330M, 1024-d audio only           (CC-BY-NC)
  muq_mulan - OpenMuQ/MuQ-MuLan-large, 512-d audio + text     (CC-BY-NC)
"""
import hashlib

import numpy as np
import soundfile as sf
import soxr

TOY_DIM = 16
CHUNK_SECONDS = 30.0
MAX_SECONDS = 120.0


class UnknownEncoderError(KeyError):
    pass


def load_audio(path, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if target_sr is not None and sr != target_sr:
        wav = soxr.resample(wav, sr, target_sr).astype(np.float32)
        sr = target_sr
    return wav, sr


def _hash_vec(payload: bytes, dim: int) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim).astype(np.float32)


def _toy_embed(wav: np.ndarray, sr: int, device: str) -> np.ndarray:
    return _hash_vec(wav.tobytes(), TOY_DIM)


def _toy_text_embed(text: str, device: str) -> np.ndarray:
    return _hash_vec(("text:" + text).encode("utf-8"), TOY_DIM)


def _chunks(wav: np.ndarray, sr: int) -> list[np.ndarray]:
    wav = wav[: int(MAX_SECONDS * sr)]
    size = int(CHUNK_SECONDS * sr)
    return [wav[i:i + size] for i in range(0, len(wav), size)] or [wav]


def _clap_embed(wav: np.ndarray, sr: int, device: str) -> np.ndarray:
    import torch
    from transformers import ClapModel, ClapProcessor
    model, processor = _clap_cached(device)
    if sr != 48000:
        wav = soxr.resample(wav, sr, 48000).astype(np.float32)
    feats = []
    with torch.no_grad():
        for chunk in _chunks(wav, 48000):
            inputs = processor(audios=chunk, sampling_rate=48000,
                               return_tensors="pt").to(device)
            feats.append(model.get_audio_features(**inputs)[0].cpu().numpy())
    return np.mean(feats, axis=0).astype(np.float32)


def _clap_text_embed(text: str, device: str) -> np.ndarray:
    import torch
    model, processor = _clap_cached(device)
    with torch.no_grad():
        inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
        return model.get_text_features(**inputs)[0].cpu().numpy().astype(np.float32)


_CLAP = {}


def _clap_cached(device: str):
    if "model" not in _CLAP:
        from transformers import ClapModel, ClapProcessor
        _CLAP["model"] = ClapModel.from_pretrained(
            "laion/larger_clap_music").to(device).eval()
        _CLAP["processor"] = ClapProcessor.from_pretrained("laion/larger_clap_music")
    return _CLAP["model"], _CLAP["processor"]


def _mert_embed(wav: np.ndarray, sr: int, device: str) -> np.ndarray:
    import torch
    from transformers import AutoModel, Wav2Vec2FeatureExtractor
    if "model" not in _MERT:
        _MERT["model"] = AutoModel.from_pretrained(
            "m-a-p/MERT-v1-330M", trust_remote_code=True).to(device).eval()
        _MERT["fe"] = Wav2Vec2FeatureExtractor.from_pretrained(
            "m-a-p/MERT-v1-330M", trust_remote_code=True)
    if sr != 24000:
        wav = soxr.resample(wav, sr, 24000).astype(np.float32)
    feats = []
    with torch.no_grad():
        for chunk in _chunks(wav, 24000):
            inputs = _MERT["fe"](chunk, sampling_rate=24000,
                                 return_tensors="pt").to(device)
            hidden = _MERT["model"](**inputs).last_hidden_state
            feats.append(hidden.mean(dim=1)[0].cpu().numpy())
    return np.mean(feats, axis=0).astype(np.float32)


_MERT = {}


def _mert_text_embed(text: str, device: str) -> np.ndarray:
    raise UnknownEncoderError("mert has no text tower - use clap or muq_mulan for text")


def _muq_mulan_embed(wav: np.ndarray, sr: int, device: str) -> np.ndarray:
    import torch
    if "model" not in _MULAN:
        from muq import MuQMuLan  # pip install muq (or vendored per CMI-RewardBench)
        _MULAN["model"] = MuQMuLan.from_pretrained(
            "OpenMuQ/MuQ-MuLan-large").to(device).eval()
    if sr != 24000:
        wav = soxr.resample(wav, sr, 24000).astype(np.float32)
    feats = []
    with torch.no_grad():
        for chunk in _chunks(wav, 24000):
            tensor = torch.from_numpy(chunk).unsqueeze(0).to(device)
            feats.append(_MULAN["model"](wavs=tensor)[0].cpu().numpy())
    return np.mean(feats, axis=0).astype(np.float32)


_MULAN = {}


def _muq_mulan_text_embed(text: str, device: str) -> np.ndarray:
    import torch
    if "model" not in _MULAN:
        from muq import MuQMuLan
        _MULAN["model"] = MuQMuLan.from_pretrained(
            "OpenMuQ/MuQ-MuLan-large").to(device).eval()
    with torch.no_grad():
        return _MULAN["model"](texts=[text])[0].cpu().numpy().astype(np.float32)


_REGISTRY = {
    "toy": (_toy_embed, _toy_text_embed),
    "clap": (_clap_embed, _clap_text_embed),
    "mert": (_mert_embed, _mert_text_embed),
    "muq_mulan": (_muq_mulan_embed, _muq_mulan_text_embed),
}


def embed(name: str, wav: np.ndarray, sr: int, device: str = "cpu") -> np.ndarray:
    if name not in _REGISTRY:
        raise UnknownEncoderError(
            f"unknown encoder '{name}', available: {sorted(_REGISTRY)}")
    return _REGISTRY[name][0](wav, sr, device)


def text_embed(name: str, text: str, device: str = "cpu") -> np.ndarray:
    if name not in _REGISTRY:
        raise UnknownEncoderError(
            f"unknown encoder '{name}', available: {sorted(_REGISTRY)}")
    return _REGISTRY[name][1](text, device)
```

- [x] **Step 4: Run test — verify it PASSES**

```bash
uv run pytest tests/test_encoders.py -x -q
```
Expected: PASS (real encoder paths are exercised later by the extraction CLI on real data, deliberately not in unit tests — no GPU/downloads in CI)

- [x] **Step 5: Commit**

```bash
git add src/trackb/extract/encoders.py tests/test_encoders.py && git commit -m "feat(extract): encoder registry (toy/clap/mert/muq_mulan) + soundfile decode"
```

---

### Task 9: Pointwise Bradley-Terry head — symmetry
**Group:** C (sequential; after Group B)

**Behavior being verified:** swapping clip A and clip B exactly negates the margin (the entire position-bias failure class dies here).
**Interface under test:** `trackb.train.head.BTHead.margin`.

**Files:**
- Create: `src/trackb/train/__init__.py`, `src/trackb/train/head.py`
- Test: `tests/test_head.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_head.py
import torch
from trackb.train.head import BTHead


def test_pointwise_margin_is_antisymmetric():
    torch.manual_seed(0)
    head = BTHead(audio_dim=16, ctx_dim=16, width=32, depth=2, joint=0)
    a, b = torch.randn(5, 16), torch.randn(5, 16)
    ctx = torch.randn(5, 16)
    m_ab = head.margin(a, b, ctx)
    m_ba = head.margin(b, a, ctx)
    assert m_ab.shape == (5,)
    torch.testing.assert_close(m_ab, -m_ba)
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
uv run pytest tests/test_head.py -x -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'trackb.train'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# src/trackb/train/__init__.py
```

```python
# src/trackb/train/head.py
"""Bradley-Terry preference head over frozen encoder features.

joint=0 (pointwise): margin = f(a, ctx) - f(b, ctx). Position-bias-immune by
construction (each clip scored independently).
joint=1 (light cross-encoder, the CMI-RM-style interaction): margin =
(g(a,b,ctx) - g(b,a,ctx)) / 2 - antisymmetrized so both orderings are baked in.
"""
import torch
import torch.nn as nn


def _mlp(in_dim: int, width: int, depth: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = in_dim
    for _ in range(depth):
        layers += [nn.Linear(dim, width), nn.ReLU()]
        dim = width
    layers.append(nn.Linear(dim, 1))
    return nn.Sequential(*layers)


class BTHead(nn.Module):
    def __init__(self, audio_dim: int, ctx_dim: int,
                 width: int = 256, depth: int = 2, joint: int = 0):
        super().__init__()
        self.joint = joint
        self.audio_dim = audio_dim
        self.ctx_dim = ctx_dim
        if joint == 0:
            self.scorer = _mlp(audio_dim + ctx_dim, width, depth)
        else:
            self.scorer = _mlp(2 * audio_dim + ctx_dim, width, depth)

    def penultimate(self, feat: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """Pointwise clip representation before the final linear (for the
        adversarial probe). Only defined for joint=0."""
        x = torch.cat([feat, ctx], dim=-1)
        return self.scorer[:-1](x)

    def margin(self, feat_a: torch.Tensor, feat_b: torch.Tensor,
               ctx: torch.Tensor) -> torch.Tensor:
        if self.joint == 0:
            score_a = self.scorer(torch.cat([feat_a, ctx], dim=-1)).squeeze(-1)
            score_b = self.scorer(torch.cat([feat_b, ctx], dim=-1)).squeeze(-1)
            return score_a - score_b
        g_ab = self.scorer(torch.cat([feat_a, feat_b, ctx], dim=-1)).squeeze(-1)
        g_ba = self.scorer(torch.cat([feat_b, feat_a, ctx], dim=-1)).squeeze(-1)
        return (g_ab - g_ba) / 2
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
uv run pytest tests/test_head.py -x -q
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/trackb/train tests/test_head.py && git commit -m "feat(train): pointwise BT head with antisymmetric margin"
```

---

### Task 10: Joint (cross-encoder) head — antisymmetry
**Group:** C (sequential; after Task 9 — same files)

**Behavior being verified:** the joint head's margin is exactly antisymmetric despite being pair-input, and actually uses cross-clip interaction (its margin differs from any pointwise decomposition on crafted inputs).
**Interface under test:** `trackb.train.head.BTHead.margin` with `joint=1`.

**Files:**
- Modify: `src/trackb/train/head.py` (already implemented in Task 9 — this task adds the behavioral test that pins it)
- Test: `tests/test_head.py`

- [x] **Step 1: Write the failing test** (append to `tests/test_head.py`)

```python
def test_joint_margin_is_antisymmetric():
    torch.manual_seed(0)
    head = BTHead(audio_dim=16, ctx_dim=16, width=32, depth=2, joint=1)
    a, b = torch.randn(5, 16), torch.randn(5, 16)
    ctx = torch.randn(5, 16)
    torch.testing.assert_close(head.margin(a, b, ctx), -head.margin(b, a, ctx))


def test_joint_uses_cross_clip_interaction():
    torch.manual_seed(0)
    head = BTHead(audio_dim=4, ctx_dim=4, width=32, depth=2, joint=1)
    ctx = torch.zeros(1, 4)
    a = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    b1 = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    b2 = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    # a pointwise decomposition would satisfy m(a,b1)-m(a,b2) == s(b2)-s(b1)
    # independent of a; verify the joint head's margin depends on the pairing
    a2 = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    d1 = (head.margin(a, b1, ctx) - head.margin(a, b2, ctx)).item()
    d2 = (head.margin(a2, b1, ctx) - head.margin(a2, b2, ctx)).item()
    assert abs(d1 - d2) > 1e-6
```

- [x] **Step 2: Run test — verify it FAILS or PASSES**

```bash
uv run pytest tests/test_head.py -x -q
```
Expected: PASS (Task 9 implemented both modes). If `test_joint_uses_cross_clip_interaction` FAILS, the joint scorer is degenerate — fix `head.py` so the joint branch concatenates `[feat_a, feat_b, ctx]` through a shared MLP as written in Task 9. The value of this task is pinning the behavior; a pass without code change is acceptable ONLY because Task 9's diff already contains the implementation and this test would fail against a pointwise-only head (verifiable: temporarily setting `joint=0` in the test makes `test_joint_uses_cross_clip_interaction` fail).

- [x] **Step 3: Implement** — only if Step 2 failed; the Task 9 implementation is the reference.

- [x] **Step 4: Run test — verify it PASSES**

```bash
uv run pytest tests/test_head.py -x -q
```
Expected: PASS (all four head tests)

- [x] **Step 5: Commit**

```bash
git add tests/test_head.py && git commit -m "test(train): pin joint-head antisymmetry + cross-clip interaction"
```

---

### Task 11: Trainer — determinism
**Group:** C (sequential; after Task 10)

**Behavior being verified:** two `train()` runs with the same config, pairs, and store produce identical metrics (bit-for-bit) — the precondition for autoresearch keep-or-revert decisions.
**Interface under test:** `trackb.train.trainer.train`.

**Files:**
- Create: `src/trackb/train/trainer.py`
- Test: `tests/test_trainer.py`

- [x] **Step 1: Write the failing test**

```python
# tests/test_trainer.py
from trackb.train.trainer import TrainConfig, train
from tests.fixtures.make_fixtures import build_fixture_corpus


def config(**overrides):
    base = dict(encoder="toy", head_width=32, head_depth=2, joint=0,
                lr=0.01, epochs=30, batch_size=16, sampler="uniform",
                adversarial_lambda=0.0, seed=3, k_folds=2,
                final_generators=("g3",))
    base.update(overrides)
    return TrainConfig(**base)


def test_train_is_deterministic(tmp_path):
    pairs, store = build_fixture_corpus(tmp_path / "fx", n_pairs=48, seed=7)
    r1 = train(config(), pairs, store, corpus="fixture",
               out_dir=tmp_path / "run1")
    r2 = train(config(), pairs, store, corpus="fixture",
               out_dir=tmp_path / "run2")
    assert r1.metrics == r2.metrics
    assert r1.metrics["holdout_acc"] > 0.0
    assert (tmp_path / "run1" / "checkpoint.pt").exists()


def test_metrics_contain_per_fold_accuracies(tmp_path):
    pairs, store = build_fixture_corpus(tmp_path / "fx", n_pairs=48, seed=7)
    result = train(config(), pairs, store, corpus="fixture",
                   out_dir=tmp_path / "run")
    assert len(result.metrics["fold_accs"]) == 2
    assert result.metrics["holdout_acc"] == sum(
        result.metrics["fold_accs"]) / 2
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
uv run pytest tests/test_trainer.py -x -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'trackb.train.trainer'`

- [x] **Step 3: Implement the minimum to make the test pass**

```python
# src/trackb/train/trainer.py
"""Config-in, checkpoint+metrics-out trainer. Deliberately thin: /autoresearch
mutates TrainConfig fields, never this file. CPU-only on purpose - the head is
tiny and CPU is the deterministic path (MPS nondeterminism would poison the
keep-or-revert loop)."""
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from trackb.evalx.folds import make_folds
from trackb.extract.store import FeatureStore
from trackb.schema import PairRecord
from trackb.train.head import BTHead


@dataclass(frozen=True)
class TrainConfig:
    encoder: str
    head_width: int
    head_depth: int
    joint: int
    lr: float
    epochs: int
    batch_size: int
    sampler: str            # "uniform" | "generator_balanced"
    adversarial_lambda: float
    seed: int
    k_folds: int
    final_generators: tuple[str, ...]


@dataclass
class TrainResult:
    checkpoint_path: Path
    metrics: dict


def assemble(pairs: list[PairRecord], store: FeatureStore, encoder: str,
             corpus: str) -> dict:
    """Join pairs to cached features. Refuses unverified manifests (store.verify
    raises ManifestError naming the missing clip)."""
    clip_entries = store.verify(encoder, corpus)
    ctx_entries = store.verify(encoder, f"{corpus}__text")
    feat_a = np.stack([store.get(clip_entries[p.clip_a], encoder) for p in pairs])
    feat_b = np.stack([store.get(clip_entries[p.clip_b], encoder) for p in pairs])
    ctx = np.stack([store.get(ctx_entries[p.pair_id], encoder) for p in pairs])
    y = np.array([1.0 if p.label == "A" else 0.0 for p in pairs], dtype=np.float32)
    return {"feat_a": torch.from_numpy(feat_a), "feat_b": torch.from_numpy(feat_b),
            "ctx": torch.from_numpy(ctx), "y": torch.from_numpy(y)}


def _fit(config: TrainConfig, data: dict) -> BTHead:
    torch.manual_seed(config.seed)
    head = BTHead(audio_dim=data["feat_a"].shape[1], ctx_dim=data["ctx"].shape[1],
                  width=config.head_width, depth=config.head_depth,
                  joint=config.joint)
    opt = torch.optim.Adam(head.parameters(), lr=config.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    n = len(data["y"])
    generator = torch.Generator().manual_seed(config.seed)
    for _ in range(config.epochs):
        perm = torch.randperm(n, generator=generator)
        for start in range(0, n, config.batch_size):
            idx = perm[start:start + config.batch_size]
            margin = head.margin(data["feat_a"][idx], data["feat_b"][idx],
                                 data["ctx"][idx])
            loss = loss_fn(margin, data["y"][idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return head


def _accuracy(head: BTHead, data: dict) -> float:
    with torch.no_grad():
        margin = head.margin(data["feat_a"], data["feat_b"], data["ctx"])
        pred = (margin > 0).float()
    return float((pred == data["y"]).float().mean())


def train(config: TrainConfig, pairs: list[PairRecord], store: FeatureStore,
          corpus: str, out_dir: Path | str) -> TrainResult:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = make_folds(pairs, k=config.k_folds,
                       final_generators=set(config.final_generators),
                       seed=config.seed)
    fold_accs = []
    for train_pairs, val_pairs in folds.rotating:
        head = _fit(config, assemble(train_pairs, store, config.encoder, corpus))
        fold_accs.append(_accuracy(
            head, assemble(val_pairs, store, config.encoder, corpus)))
    final_head = _fit(config, assemble(folds.final_train, store,
                                       config.encoder, corpus))
    checkpoint_path = out_dir / "checkpoint.pt"
    torch.save({"state_dict": final_head.state_dict(),
                "config": asdict(config),
                "audio_dim": final_head.audio_dim,
                "ctx_dim": final_head.ctx_dim,
                "encoder": config.encoder}, checkpoint_path)
    metrics = {"fold_accs": fold_accs,
               "holdout_acc": sum(fold_accs) / len(fold_accs)}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    return TrainResult(checkpoint_path=checkpoint_path, metrics=metrics)
```

- [x] **Step 4: Run test — verify it PASSES**

```bash
uv run pytest tests/test_trainer.py -x -q
```
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add src/trackb/train/trainer.py tests/test_trainer.py && git commit -m "feat(train): deterministic config-driven k-fold trainer"
```

---

### Task 12: Smoke — the pipeline learns the planted signal
**Group:** C (sequential; after Task 11)

**Behavior being verified:** the canonical success state from the spec — on the synthetic fixture corpus, generator-holdout accuracy exceeds 0.9. This is the harness every later change is judged against.
**Interface under test:** `train()` end-to-end on fixtures, plus `trackb.evalx.score.score_checkpoint`.

**Files:**
- Create: `src/trackb/evalx/score.py`
- Modify: `justfile` (add `smoke` recipe)
- Test: `tests/test_smoke.py`

- [x] **Step 1: Write the failing test**

```python
# tests/test_smoke.py
from trackb.train.trainer import TrainConfig, train
from trackb.evalx.score import score_checkpoint
from tests.fixtures.make_fixtures import build_fixture_corpus


def test_pipeline_learns_planted_signal(tmp_path):
    pairs, store = build_fixture_corpus(tmp_path / "fx", n_pairs=96, seed=7)
    config = TrainConfig(encoder="toy", head_width=64, head_depth=2, joint=0,
                         lr=0.01, epochs=150, batch_size=16, sampler="uniform",
                         adversarial_lambda=0.0, seed=3, k_folds=2,
                         final_generators=("g3",))
    result = train(config, pairs, store, corpus="fixture",
                   out_dir=tmp_path / "run")
    assert result.metrics["holdout_acc"] > 0.9, result.metrics


def test_score_checkpoint_on_final_fold(tmp_path):
    from trackb.evalx.folds import make_folds
    pairs, store = build_fixture_corpus(tmp_path / "fx", n_pairs=96, seed=7)
    config = TrainConfig(encoder="toy", head_width=64, head_depth=2, joint=0,
                         lr=0.01, epochs=150, batch_size=16, sampler="uniform",
                         adversarial_lambda=0.0, seed=3, k_folds=2,
                         final_generators=("g3",))
    result = train(config, pairs, store, corpus="fixture",
                   out_dir=tmp_path / "run")
    folds = make_folds(pairs, k=2, final_generators={"g3"}, seed=3)
    acc = score_checkpoint(result.checkpoint_path, folds.final_holdout,
                           store, corpus="fixture")
    assert acc > 0.8
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
uv run pytest tests/test_smoke.py -x -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'trackb.evalx.score'`

- [x] **Step 3: Implement the minimum to make the test pass**

```python
# src/trackb/evalx/score.py
"""Score a saved checkpoint on a set of pairs through cached features."""
from pathlib import Path

import torch

from trackb.extract.store import FeatureStore
from trackb.schema import PairRecord
from trackb.train.head import BTHead
from trackb.train.trainer import assemble


def load_head(checkpoint_path: Path | str) -> tuple[BTHead, dict]:
    # weights_only=True: the checkpoint is tensors + primitives only; never
    # unpickle arbitrary objects from a file we might receive/move around
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    head = BTHead(audio_dim=ckpt["audio_dim"], ctx_dim=ckpt["ctx_dim"],
                  width=cfg["head_width"], depth=cfg["head_depth"],
                  joint=cfg["joint"])
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    return head, ckpt


def score_checkpoint(checkpoint_path: Path | str, pairs: list[PairRecord],
                     store: FeatureStore, corpus: str) -> float:
    head, ckpt = load_head(checkpoint_path)
    data = assemble(pairs, store, ckpt["encoder"], corpus)
    with torch.no_grad():
        margin = head.margin(data["feat_a"], data["feat_b"], data["ctx"])
        pred = (margin > 0).float()
    return float((pred == data["y"]).float().mean())
```

Append to `justfile`:

```justfile
smoke:
    uv run pytest tests/test_smoke.py -x -q
```

- [x] **Step 4: Run test — verify it PASSES**

```bash
uv run pytest tests/test_smoke.py -x -q && just smoke
```
Expected: PASS (if `holdout_acc` lands below 0.9, raise `epochs` to 300 in BOTH tests — the planted signal is linearly separable; failure at 300 epochs indicates a real bug in assemble/folds, not tuning)

- [x] **Step 5: Commit**

```bash
git add src/trackb/evalx/score.py tests/test_smoke.py justfile && git commit -m "feat(evalx): checkpoint scoring + planted-signal smoke gate"
```

---

### Task 13: Generator-ID probe
**Group:** D (sequential; after Group C)

**Behavior being verified:** on features that encode generator identity the probe recovers it (accuracy near 1.0); on generator-independent features it sits at chance — the calibration pair that makes the real-data probe result interpretable.
**Interface under test:** `trackb.probe.generator_id.generator_id_probe`.

**Files:**
- Create: `src/trackb/probe/__init__.py`, `src/trackb/probe/generator_id.py`
- Test: `tests/test_probe.py`

- [x] **Step 1: Write the failing test**

```python
# tests/test_probe.py
import numpy as np
from trackb.probe.generator_id import generator_id_probe


def test_probe_recovers_identity_encoding_features():
    rng = np.random.default_rng(0)
    labels = [f"g{i % 4}" for i in range(400)]
    X = np.stack([np.eye(4)[int(g[1])].repeat(4) + rng.normal(0, 0.05, 16)
                  for g in labels]).astype(np.float32)
    report = generator_id_probe(X, labels, seed=0)
    assert report["accuracy"] > 0.95
    assert report["chance"] == 0.25
    assert report["n_classes"] == 4


def test_probe_sits_at_chance_on_independent_features():
    rng = np.random.default_rng(0)
    labels = [f"g{i % 4}" for i in range(400)]
    X = rng.standard_normal((400, 16)).astype(np.float32)
    report = generator_id_probe(X, labels, seed=0)
    assert abs(report["accuracy"] - 0.25) < 0.12
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
uv run pytest tests/test_probe.py -x -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'trackb.probe'`

- [x] **Step 3: Implement the minimum to make the test pass**

```python
# src/trackb/probe/__init__.py
```

```python
# src/trackb/probe/generator_id.py
"""The shortcut probe: can generator identity be read off frozen features?
High probe accuracy + a large in-dist-vs-holdout gap = the reward model can
lean on WHICH generator instead of WHAT quality - the LBD headline claim."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def generator_id_probe(X: np.ndarray, labels: list[str], seed: int = 0) -> dict:
    y = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y)
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(X_train, y_train)
    classes = sorted(set(labels))
    return {"accuracy": float(clf.score(X_test, y_test)),
            "chance": 1.0 / len(classes),
            "n_classes": len(classes)}
```

- [x] **Step 4: Run test — verify it PASSES**

```bash
uv run pytest tests/test_probe.py -x -q
```
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add src/trackb/probe tests/test_probe.py && git commit -m "feat(probe): generator-ID probe with chance calibration"
```

---

### Task 14: In-distribution vs holdout decomposition
**Group:** D (sequential; after Task 13)

**Behavior being verified:** `decompose` trains the same config under a random pair split and under generator-holdout folds and reports both accuracies plus the gap — on the fixture corpus (where features carry generator identity) the in-distribution accuracy is at least the holdout accuracy.
**Interface under test:** `trackb.probe.decompose.decompose`.

**Files:**
- Create: `src/trackb/probe/decompose.py`
- Test: `tests/test_decompose.py`

- [x] **Step 1: Write the failing test**

```python
# tests/test_decompose.py
from trackb.probe.decompose import decompose
from trackb.train.trainer import TrainConfig
from tests.fixtures.make_fixtures import build_fixture_corpus


def test_decompose_reports_both_regimes(tmp_path):
    pairs, store = build_fixture_corpus(tmp_path / "fx", n_pairs=96, seed=7)
    config = TrainConfig(encoder="toy", head_width=64, head_depth=2, joint=0,
                         lr=0.01, epochs=150, batch_size=16, sampler="uniform",
                         adversarial_lambda=0.0, seed=3, k_folds=2,
                         final_generators=("g3",))
    report = decompose(config, pairs, store, corpus="fixture",
                       out_dir=tmp_path / "runs")
    assert set(report) == {"in_dist_acc", "holdout_acc", "gap"}
    assert report["gap"] == report["in_dist_acc"] - report["holdout_acc"]
    assert report["in_dist_acc"] >= report["holdout_acc"] - 0.05
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
uv run pytest tests/test_decompose.py -x -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'trackb.probe.decompose'`

- [x] **Step 3: Implement the minimum to make the test pass**

```python
# src/trackb/probe/decompose.py
"""ID-vs-OOD decomposition: the same config trained under (a) a random pair
split (generators seen in training) and (b) generator-holdout folds. The gap
is the shortcut's contribution to apparent accuracy."""
import random
from pathlib import Path

from trackb.extract.store import FeatureStore
from trackb.schema import PairRecord
from trackb.train.trainer import TrainConfig, _accuracy, _fit, assemble, train


def decompose(config: TrainConfig, pairs: list[PairRecord], store: FeatureStore,
              corpus: str, out_dir: Path | str) -> dict:
    out_dir = Path(out_dir)
    # (a) in-distribution: random 80/20 pair split, all generators seen
    rng = random.Random(config.seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    cut = int(0.8 * len(shuffled))
    head = _fit(config, assemble(shuffled[:cut], store, config.encoder, corpus))
    in_dist = _accuracy(head, assemble(shuffled[cut:], store,
                                       config.encoder, corpus))
    # (b) generator-holdout: the trainer's own rotating-fold metric
    holdout = train(config, pairs, store, corpus,
                    out_dir=out_dir / "holdout").metrics["holdout_acc"]
    return {"in_dist_acc": in_dist, "holdout_acc": holdout,
            "gap": in_dist - holdout}
```

- [x] **Step 4: Run test — verify it PASSES**

```bash
uv run pytest tests/test_decompose.py -x -q
```
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add src/trackb/probe/decompose.py tests/test_decompose.py && git commit -m "feat(probe): in-dist vs generator-holdout decomposition"
```

---

### Task 15: Mitigation — generator-balanced sampling + adversarial term
**Group:** D (sequential; after Task 14 — modifies trainer.py)

**Behavior being verified:** with `sampler="generator_balanced"` the training loop samples generators near-uniformly (exposed via `TrainResult.metrics["sample_stats"]`); with `adversarial_lambda > 0` training runs and reports `adv_loss` (its OOD effect is a research measurement made by the probe on real data, not a unit assertion).
**Interface under test:** `trackb.train.trainer.train` config fields `sampler`, `adversarial_lambda`.

**Files:**
- Modify: `src/trackb/train/trainer.py`, `src/trackb/train/head.py`
- Test: `tests/test_mitigation.py`

- [x] **Step 1: Write the failing test**

```python
# tests/test_mitigation.py
from trackb.train.trainer import TrainConfig, train
from tests.fixtures.make_fixtures import build_fixture_corpus


def config(**overrides):
    base = dict(encoder="toy", head_width=32, head_depth=2, joint=0,
                lr=0.01, epochs=20, batch_size=16, sampler="uniform",
                adversarial_lambda=0.0, seed=3, k_folds=2,
                final_generators=("g3",))
    base.update(overrides)
    return TrainConfig(**base)


def test_generator_balanced_sampler_equalizes_exposure(tmp_path):
    pairs, store = build_fixture_corpus(tmp_path / "fx", n_pairs=96, seed=7)
    result = train(config(sampler="generator_balanced"), pairs, store,
                   corpus="fixture", out_dir=tmp_path / "run")
    stats = result.metrics["sample_stats"]
    counts = list(stats.values())
    assert max(counts) / max(min(counts), 1) < 1.5


def test_adversarial_lambda_reports_adv_loss(tmp_path):
    pairs, store = build_fixture_corpus(tmp_path / "fx", n_pairs=96, seed=7)
    result = train(config(adversarial_lambda=0.5), pairs, store,
                   corpus="fixture", out_dir=tmp_path / "run")
    assert result.metrics["adv_loss"] > 0.0


def test_unknown_sampler_raises(tmp_path):
    import pytest
    pairs, store = build_fixture_corpus(tmp_path / "fx", n_pairs=48, seed=7)
    with pytest.raises(ValueError, match=r"sampler"):
        train(config(sampler="magic"), pairs, store, corpus="fixture",
              out_dir=tmp_path / "run")
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
uv run pytest tests/test_mitigation.py -x -q
```
Expected: FAIL — `KeyError: 'sample_stats'` (or equivalent: current trainer ignores sampler/adversarial fields)

- [x] **Step 3: Implement the minimum to make the test pass**

In `src/trackb/train/head.py`, append:

```python
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


def grad_reverse(x: torch.Tensor, lam: float) -> torch.Tensor:
    return GradReverse.apply(x, lam)
```

In `src/trackb/train/trainer.py`, replace `_fit` and the metrics assembly in `train` with:

```python
def _pair_weights(config: TrainConfig, pairs: list[PairRecord]) -> torch.Tensor:
    if config.sampler == "uniform":
        return torch.ones(len(pairs))
    if config.sampler == "generator_balanced":
        counts: dict[str, int] = {}
        for p in pairs:
            for g in (p.generator_a, p.generator_b):
                counts[g] = counts.get(g, 0) + 1
        weights = [2.0 / (counts[p.generator_a] + counts[p.generator_b])
                   for p in pairs]
        return torch.tensor(weights)
    raise ValueError(
        f"unknown sampler '{config.sampler}', expected 'uniform' or "
        f"'generator_balanced'")


def _fit(config: TrainConfig, data: dict,
         pairs: list[PairRecord] | None = None,
         stats: dict | None = None) -> tuple["BTHead", float]:
    import torch.nn.functional as F
    from trackb.train.head import grad_reverse
    torch.manual_seed(config.seed)
    head = BTHead(audio_dim=data["feat_a"].shape[1], ctx_dim=data["ctx"].shape[1],
                  width=config.head_width, depth=config.head_depth,
                  joint=config.joint)
    adv_head = None
    gen_index: dict[str, int] = {}
    gen_labels = None
    if config.adversarial_lambda > 0 and pairs is not None and config.joint == 0:
        gens = sorted({g for p in pairs for g in (p.generator_a, p.generator_b)})
        gen_index = {g: i for i, g in enumerate(gens)}
        gen_labels = torch.tensor([gen_index[p.generator_a] for p in pairs])
        adv_head = nn.Linear(config.head_width, len(gens))
    params = list(head.parameters()) + (
        list(adv_head.parameters()) if adv_head is not None else [])
    opt = torch.optim.Adam(params, lr=config.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    n = len(data["y"])
    weights = _pair_weights(config, pairs) if pairs is not None else torch.ones(n)
    generator = torch.Generator().manual_seed(config.seed)
    adv_loss_total, adv_steps = 0.0, 0
    for _ in range(config.epochs):
        idx_all = torch.multinomial(weights, n, replacement=True,
                                    generator=generator)
        if stats is not None and pairs is not None:
            for i in idx_all.tolist():
                for g in (pairs[i].generator_a, pairs[i].generator_b):
                    stats[g] = stats.get(g, 0) + 1
        for start in range(0, n, config.batch_size):
            idx = idx_all[start:start + config.batch_size]
            margin = head.margin(data["feat_a"][idx], data["feat_b"][idx],
                                 data["ctx"][idx])
            loss = loss_fn(margin, data["y"][idx])
            if adv_head is not None and gen_labels is not None:
                rep = head.penultimate(data["feat_a"][idx], data["ctx"][idx])
                logits = adv_head(grad_reverse(rep, config.adversarial_lambda))
                adv = F.cross_entropy(logits, gen_labels[idx])
                loss = loss + config.adversarial_lambda * adv
                adv_loss_total += float(adv)
                adv_steps += 1
            opt.zero_grad()
            loss.backward()
            opt.step()
    return head, (adv_loss_total / adv_steps if adv_steps else 0.0)
```

And update `train()` to thread pairs/stats through and report them:

```python
def train(config: TrainConfig, pairs: list[PairRecord], store: FeatureStore,
          corpus: str, out_dir: Path | str) -> TrainResult:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = make_folds(pairs, k=config.k_folds,
                       final_generators=set(config.final_generators),
                       seed=config.seed)
    fold_accs = []
    sample_stats: dict[str, int] = {}
    adv_loss = 0.0
    for train_pairs, val_pairs in folds.rotating:
        head, adv_loss = _fit(config,
                              assemble(train_pairs, store, config.encoder, corpus),
                              pairs=train_pairs, stats=sample_stats)
        fold_accs.append(_accuracy(
            head, assemble(val_pairs, store, config.encoder, corpus)))
    final_head, _ = _fit(config, assemble(folds.final_train, store,
                                          config.encoder, corpus),
                         pairs=folds.final_train)
    checkpoint_path = out_dir / "checkpoint.pt"
    torch.save({"state_dict": final_head.state_dict(),
                "config": asdict(config),
                "audio_dim": final_head.audio_dim,
                "ctx_dim": final_head.ctx_dim,
                "encoder": config.encoder}, checkpoint_path)
    metrics = {"fold_accs": fold_accs,
               "holdout_acc": sum(fold_accs) / len(fold_accs),
               "sample_stats": sample_stats,
               "adv_loss": adv_loss}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    return TrainResult(checkpoint_path=checkpoint_path, metrics=metrics)
```

Note: `decompose` (Task 14) calls `_fit(config, data)` and uses only the head — update its call site to unpack: `head, _ = _fit(config, assemble(...))`.

- [x] **Step 4: Run test — verify it PASSES (and no regressions)**

```bash
uv run pytest tests/test_mitigation.py tests/test_trainer.py tests/test_smoke.py tests/test_decompose.py -x -q
```
Expected: PASS (determinism and smoke still hold — sampling uses the seeded generator)

- [x] **Step 5: Commit**

```bash
git add src/trackb/train tests/test_mitigation.py && git commit -m "feat(train): generator-balanced sampler + gradient-reversal adversarial term"
```

---

### Task 16: CMI-Pref adapter
**Group:** E (parallel with Tasks 17, 18; needs Group A only)

**Behavior being verified:** raw CMI-Pref rows (verified real schema: `audio-path`, `audio2`, `model_a`, `model_b`, `preference-musicality`, `confidence_preference-musicality`, `ref-audio-path`, `lyrics`, `prompt`) map to schema-valid PairRecords; rows with empty preference are skipped; generator names are populated.
**Interface under test:** `trackb.corpora.cmi_pref.rows_to_pairs`.

**Files:**
- Create: `src/trackb/corpora/__init__.py`, `src/trackb/corpora/cmi_pref.py`, `tests/fixtures/cmi_pref_sample.jsonl`
- Test: `tests/test_cmi_pref.py`

- [x] **Step 1: Write the failing test**

Create `tests/fixtures/cmi_pref_sample.jsonl` (3 rows in the REAL field layout — verified against `all_test.jsonl` from the P0 scratch env on 2026-07-04):

```json
{"audio-path": "cmi-pref/gen-audio/aaa.mp3", "audio2": "cmi-pref/gen-audio/bbb.mp3", "prompt": "anime rock, female vocal", "lyrics": "[Verse1] la la", "prompt id": "90502", "ref-audio-path": "", "preference-musicality": "model_b", "confidence_preference-musicality": "2.0", "preference-alignment": "model_a", "confidence_preference-alignment": "1.0", "user id": "604", "model_a": "songgen", "model_b": "yue", "source": "cmi-arena-annotation", "split": "train"}
{"audio-path": "cmi-pref/gen-audio/ccc.mp3", "audio2": "cmi-pref/gen-audio/ddd.mp3", "prompt": "lofi hip hop beat", "lyrics": "", "prompt id": "90503", "ref-audio-path": "cmi-pref/ref-audio/ref1.mp3", "preference-musicality": "model_a", "confidence_preference-musicality": "3.0", "preference-alignment": "model_b", "confidence_preference-alignment": "2.0", "user id": "605", "model_a": "musicgen", "model_b": "suno", "source": "cmi-arena-annotation", "split": "train"}
{"audio-path": "cmi-pref/gen-audio/eee.mp3", "audio2": "cmi-pref/gen-audio/fff.mp3", "prompt": "orchestral trailer", "lyrics": "", "prompt id": "90504", "ref-audio-path": "", "preference-musicality": "", "confidence_preference-musicality": "", "preference-alignment": "model_a", "confidence_preference-alignment": "1.0", "user id": "606", "model_a": "udio", "model_b": "yue", "source": "cmi-arena-annotation", "split": "train"}
```

```python
# tests/test_cmi_pref.py
import json
from pathlib import Path
from trackb.corpora.cmi_pref import rows_to_pairs
from trackb.schema import validate_pairs

FIXTURE = Path(__file__).parent / "fixtures" / "cmi_pref_sample.jsonl"


def load_rows():
    return [json.loads(line) for line in FIXTURE.read_text().splitlines()]


def test_musicality_target_maps_and_skips_unlabeled():
    pairs = validate_pairs(rows_to_pairs(load_rows(), target="musicality"))
    assert len(pairs) == 2  # third row has empty preference-musicality
    first = pairs[0]
    assert first.label == "B"           # preference-musicality == model_b
    assert first.generator_a == "songgen" and first.generator_b == "yue"
    assert first.clip_a == "cmi-pref/gen-audio/aaa.mp3"
    assert first.lyrics == "[Verse1] la la"
    assert first.ref_audio is None      # empty string -> None
    assert first.confidence == 2.0
    assert pairs[1].ref_audio == "cmi-pref/ref-audio/ref1.mp3"
    assert pairs[1].label == "A"


def test_alignment_target_uses_alignment_preference():
    pairs = validate_pairs(rows_to_pairs(load_rows(), target="alignment"))
    assert len(pairs) == 3
    assert [p.label for p in pairs] == ["A", "B", "A"]


def test_unknown_target_raises():
    import pytest
    with pytest.raises(ValueError, match=r"target"):
        rows_to_pairs(load_rows(), target="vibes")
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
uv run pytest tests/test_cmi_pref.py -x -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'trackb.corpora'`

- [x] **Step 3: Implement the minimum to make the test pass**

```python
# src/trackb/corpora/__init__.py
```

```python
# src/trackb/corpora/cmi_pref.py
"""CMI-Pref adapter (HaiwenXia/cmi-pref, CC-BY-NC-SA - research use only).

Real row schema (verified 2026-07-04 against the P0 scratch all_test.jsonl):
audio-path (clip A), audio2 (clip B), model_a/model_b (GENERATOR names),
preference-musicality / preference-alignment in {model_a, model_b, ""},
confidence_* (stringly floats), ref-audio-path ("" when absent), lyrics,
prompt, prompt id, user id, split."""
from trackb.schema import PairRecord

_TARGET_FIELDS = {
    "musicality": ("preference-musicality", "confidence_preference-musicality"),
    "alignment": ("preference-alignment", "confidence_preference-alignment"),
}


def rows_to_pairs(rows: list[dict], target: str = "musicality") -> list[PairRecord]:
    if target not in _TARGET_FIELDS:
        raise ValueError(
            f"target must be one of {sorted(_TARGET_FIELDS)}, got '{target}'")
    pref_field, conf_field = _TARGET_FIELDS[target]
    pairs = []
    for i, row in enumerate(rows):
        pref = row.get(pref_field, "")
        if pref not in ("model_a", "model_b"):
            continue  # unlabeled for this target - documented skip, not an error
        conf_raw = row.get(conf_field, "")
        pairs.append(PairRecord(
            pair_id=f"cmipref-{target}-{row['prompt id']}-{row['user id']}-{i}",
            source="cmi_pref",
            prompt=row["prompt"],
            lyrics=row["lyrics"] or None,
            ref_audio=row["ref-audio-path"] or None,
            clip_a=row["audio-path"],
            clip_b=row["audio2"],
            label="A" if pref == "model_a" else "B",
            generator_a=row["model_a"],
            generator_b=row["model_b"],
            confidence=float(conf_raw) if conf_raw else None,
            modality=None))
    return pairs


def load_split(split: str, target: str = "musicality") -> list[PairRecord]:
    """Load from HF hub. Network path - used by extraction/training CLIs,
    exercised on real data, not in unit tests."""
    from datasets import load_dataset
    ds = load_dataset("HaiwenXia/cmi-pref", split=split)
    return rows_to_pairs(list(ds), target=target)
```

- [x] **Step 4: Run test — verify it PASSES**

```bash
uv run pytest tests/test_cmi_pref.py -x -q
```
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add src/trackb/corpora tests/fixtures/cmi_pref_sample.jsonl tests/test_cmi_pref.py && git commit -m "feat(corpora): cmi_pref adapter with musicality/alignment targets"
```

---

### Task 17: AIME-survey adapter
**Group:** E (parallel with Tasks 16, 18)

**Behavior being verified:** AIME-survey rows (verified real schema: `question-type`, `description`, `model-1`, `track-1-id`, `track-1-begin`, `track-1-end`, `model-2`, `track-2-*`, `answer` int) map to PairRecords with segment-carrying clip_refs; non-{1,2} answers raise loudly (never silently dropped — we have not verified tie codes exist, so an unknown code must surface).
**Interface under test:** `trackb.corpora.aime_survey.rows_to_pairs`.

**Files:**
- Create: `src/trackb/corpora/aime_survey.py`
- Test: `tests/test_aime_survey.py`

- [x] **Step 1: Write the failing test**

```python
# tests/test_aime_survey.py
import pytest
from trackb.corpora.aime_survey import AimeAnswerError, rows_to_pairs
from trackb.schema import validate_pairs

ROWS = [
    {"question-type": "Music Quality", "description": "newage, world, celtic",
     "model-1": "Suno v3", "track-1-id": 5331, "track-1-begin": "00:00:53",
     "track-1-end": "00:01:03", "model-2": "MusicGen Large", "track-2-id": 1226,
     "track-2-begin": "00:00:00", "track-2-end": "00:00:10", "answer": 1},
    {"question-type": "Music Quality", "description": "jazz, upbeat",
     "model-1": "Udio", "track-1-id": 42, "track-1-begin": "00:00:00",
     "track-1-end": "00:00:10", "model-2": "Suno v3", "track-2-id": 77,
     "track-2-begin": "00:00:05", "track-2-end": "00:00:15", "answer": 2},
    {"question-type": "Text Alignment", "description": "rock",
     "model-1": "Udio", "track-1-id": 1, "track-1-begin": "00:00:00",
     "track-1-end": "00:00:10", "model-2": "Suno v3", "track-2-id": 2,
     "track-2-begin": "00:00:00", "track-2-end": "00:00:10", "answer": 1},
]


def test_music_quality_rows_map_with_segment_refs():
    pairs = validate_pairs(rows_to_pairs(ROWS, question_types={"Music Quality"}))
    assert len(pairs) == 2  # Text Alignment row filtered out
    first = pairs[0]
    assert first.label == "A"
    assert first.generator_a == "Suno v3" and first.generator_b == "MusicGen Large"
    assert first.clip_a == "aime:5331:00:00:53-00:01:03"
    assert first.clip_b == "aime:1226:00:00:00-00:00:10"
    assert first.prompt == "newage, world, celtic"
    assert pairs[1].label == "B"


def test_unknown_answer_code_raises():
    bad = dict(ROWS[0])
    bad["answer"] = 0
    with pytest.raises(AimeAnswerError, match=r"answer.*0"):
        rows_to_pairs([bad], question_types={"Music Quality"})
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
uv run pytest tests/test_aime_survey.py -x -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'trackb.corpora.aime_survey'`

- [x] **Step 3: Implement the minimum to make the test pass**

```python
# src/trackb/corpora/aime_survey.py
"""AIME-survey adapter (disco-eth/AIME-survey, CC-BY-4.0 - commercial-clean).

Real schema (verified 2026-07-04 via HF datasets-server): question-type,
description (tag prompt), model-1/model-2 (generators), track-{1,2}-id
(joins to the disco-eth/AIME audio corpus), track-*-begin/end (HH:MM:SS
segment spans), answer (int64; observed values 1 and 2). Unknown answer codes
raise - if the dataset contains tie codes we want to FIND OUT, not drop them."""
from trackb.schema import PairRecord


class AimeAnswerError(ValueError):
    pass


def clip_ref(track_id, begin: str, end: str) -> str:
    return f"aime:{track_id}:{begin}-{end}"


def rows_to_pairs(rows: list[dict],
                  question_types: set[str] = frozenset({"Music Quality"}),
                  ) -> list[PairRecord]:
    pairs = []
    for i, row in enumerate(rows):
        if row["question-type"] not in question_types:
            continue
        answer = row["answer"]
        if answer not in (1, 2):
            raise AimeAnswerError(
                f"unexpected answer code {answer!r} at row {i} - only 1/2 are "
                f"verified; inspect the dataset card before mapping this code")
        pairs.append(PairRecord(
            pair_id=f"aime-{i}",
            source="aime_survey",
            prompt=row["description"],
            lyrics=None,
            ref_audio=None,
            clip_a=clip_ref(row["track-1-id"], row["track-1-begin"],
                            row["track-1-end"]),
            clip_b=clip_ref(row["track-2-id"], row["track-2-begin"],
                            row["track-2-end"]),
            label="A" if answer == 1 else "B",
            generator_a=row["model-1"],
            generator_b=row["model-2"],
            confidence=None,
            modality="audio"))
    return pairs


def load(question_types: set[str] = frozenset({"Music Quality"})) -> list[PairRecord]:
    """Network path - real data, used by CLIs, not unit-tested."""
    from datasets import load_dataset
    ds = load_dataset("disco-eth/AIME-survey", split="train")
    return rows_to_pairs(list(ds), question_types=question_types)
```

- [x] **Step 4: Run test — verify it PASSES**

```bash
uv run pytest tests/test_aime_survey.py -x -q
```
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add src/trackb/corpora/aime_survey.py tests/test_aime_survey.py && git commit -m "feat(corpora): aime_survey adapter with segment clip_refs"
```

---

### Task 18: Music Arena adapter
**Group:** E (parallel with Tasks 16, 17)

**Behavior being verified:** Music Arena rows (verified real schema: `battle_uuid`, `prompt`, `lyrics`, `audio_a`/`audio_b`, `system_a`/`system_b`, `preference`) map to PairRecords keeping only decisive A/B outcomes; unknown preference values raise loudly (the decisive-value vocabulary is pinned by the first real extraction, not guessed).
**Interface under test:** `trackb.corpora.music_arena.rows_to_pairs`.

**Files:**
- Create: `src/trackb/corpora/music_arena.py`
- Test: `tests/test_music_arena.py`

- [x] **Step 1: Write the failing test**

```python
# tests/test_music_arena.py
import pytest
from trackb.corpora.music_arena import PreferenceValueError, rows_to_pairs
from trackb.schema import validate_pairs


def row(uuid, pref, **overrides):
    base = {"battle_uuid": uuid, "prompt": "synthwave chase scene",
            "lyrics": "", "system_a": "sysA-v1", "system_b": "sysB-v2",
            "preference": pref, "is_instrumental": True}
    base.update(overrides)
    return base


def test_decisive_rows_map_and_ties_are_skipped():
    rows = [row("u1", "A"), row("u2", "B", lyrics="some words"),
            row("u3", "TIE"), row("u4", "BOTH_BAD")]
    pairs = validate_pairs(rows_to_pairs(rows, config_name="2025_12"))
    assert [p.label for p in pairs] == ["A", "B"]
    assert pairs[0].clip_a == "musicarena:2025_12:u1:a"
    assert pairs[0].clip_b == "musicarena:2025_12:u1:b"
    assert pairs[0].generator_a == "sysA-v1"
    assert pairs[0].lyrics is None
    assert pairs[1].lyrics == "some words"


def test_unknown_preference_value_raises():
    with pytest.raises(PreferenceValueError, match=r"MAYBE"):
        rows_to_pairs([row("u9", "MAYBE")], config_name="2025_12")
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
uv run pytest tests/test_music_arena.py -x -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'trackb.corpora.music_arena'`

- [x] **Step 3: Implement the minimum to make the test pass**

```python
# src/trackb/corpora/music_arena.py
"""Music Arena adapter (music-arena/music-arena-dataset, CC-BY-4.0).

Real schema (verified 2026-07-04 via HF datasets-server, config 2025_12):
battle_uuid, prompt, lyrics, audio_a/audio_b (HF Audio), system_a/system_b
(generators), preference, is_instrumental, monthly configs (2025_07-08 ...).
The preference vocabulary is pinned optimistically to {A, B, TIE, BOTH_BAD}
per the published dataset description; any OTHER value raises so the real
vocabulary surfaces at first extraction instead of being silently dropped."""
from trackb.schema import PairRecord

DECISIVE = {"A": "A", "B": "B"}
KNOWN_INDECISIVE = {"TIE", "BOTH_BAD"}


class PreferenceValueError(ValueError):
    pass


def rows_to_pairs(rows: list[dict], config_name: str) -> list[PairRecord]:
    pairs = []
    for row in rows:
        pref = row["preference"]
        if pref in KNOWN_INDECISIVE:
            continue
        if pref not in DECISIVE:
            raise PreferenceValueError(
                f"unknown preference value {pref!r} in battle "
                f"'{row['battle_uuid']}' (config {config_name}) - extend the "
                f"vocabulary deliberately, do not drop")
        uuid = row["battle_uuid"]
        pairs.append(PairRecord(
            pair_id=f"musicarena-{config_name}-{uuid}",
            source="music_arena",
            prompt=row["prompt"],
            lyrics=row["lyrics"] or None,
            ref_audio=None,
            clip_a=f"musicarena:{config_name}:{uuid}:a",
            clip_b=f"musicarena:{config_name}:{uuid}:b",
            label=DECISIVE[pref],
            generator_a=row["system_a"],
            generator_b=row["system_b"],
            confidence=None,
            modality=None))
    return pairs


def load(config_name: str) -> list[PairRecord]:
    """Network path - real data, used by CLIs, not unit-tested."""
    from datasets import load_dataset
    ds = load_dataset("music-arena/music-arena-dataset", config_name,
                      split="train")
    return rows_to_pairs(list(ds), config_name=config_name)
```

- [x] **Step 4: Run test — verify it PASSES**

```bash
uv run pytest tests/test_music_arena.py -x -q
```
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add src/trackb/corpora/music_arena.py tests/test_music_arena.py && git commit -m "feat(corpora): music_arena adapter with pinned preference vocabulary"
```

---

### Task 19: Extraction CLI
**Group:** F (sequential; needs Groups C and E)

**Behavior being verified:** `python -m trackb.extract.run --encoder toy --corpus <fixture-dir>` walks a directory of audio files referenced by a pairs jsonl, embeds each clip AND each prompt, writes the feature store + manifests, and a second run skips existing work (resume). This is the command the user runs overnight for real CMI-Pref extraction.
**Interface under test:** `trackb.extract.run.extract_corpus` and its CLI.

**Files:**
- Create: `src/trackb/extract/run.py`
- Modify: `justfile` (add `extract` recipe)
- Test: `tests/test_extract_run.py`

- [x] **Step 1: Write the failing test**

```python
# tests/test_extract_run.py
import json
import numpy as np
import soundfile as sf
from trackb.extract.run import extract_corpus
from trackb.extract.store import FeatureStore
from trackb.schema import PairRecord


def make_audio_corpus(root):
    (root / "audio").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for name in ("a1.wav", "b1.wav", "a2.wav", "b2.wav"):
        sf.write(root / "audio" / name,
                 rng.standard_normal(16000).astype(np.float32), 16000)
    pairs = [
        PairRecord(pair_id="p1", source="local", prompt="first prompt",
                   lyrics=None, ref_audio=None, clip_a="audio/a1.wav",
                   clip_b="audio/b1.wav", label="A", generator_a="g0",
                   generator_b="g1", confidence=None, modality=None),
        PairRecord(pair_id="p2", source="local", prompt="second prompt",
                   lyrics=None, ref_audio=None, clip_a="audio/a2.wav",
                   clip_b="audio/b2.wav", label="B", generator_a="g0",
                   generator_b="g1", confidence=None, modality=None),
    ]
    return pairs


def test_extract_writes_verified_store_and_resumes(tmp_path):
    pairs = make_audio_corpus(tmp_path)
    stats = extract_corpus(pairs, corpus="local", encoder="toy",
                           audio_root=tmp_path, store_root=tmp_path / "data",
                           device="cpu")
    assert stats["clips_embedded"] == 4 and stats["prompts_embedded"] == 2
    store = FeatureStore(tmp_path / "data")
    entries = store.verify("toy", "local")
    assert set(entries) == {"audio/a1.wav", "audio/b1.wav",
                            "audio/a2.wav", "audio/b2.wav"}
    store.verify("toy", "local__text")
    # resume: nothing re-embedded
    stats2 = extract_corpus(pairs, corpus="local", encoder="toy",
                            audio_root=tmp_path, store_root=tmp_path / "data",
                            device="cpu")
    assert stats2["clips_embedded"] == 0 and stats2["prompts_embedded"] == 0


def test_missing_audio_file_fails_loudly(tmp_path):
    import pytest
    pairs = make_audio_corpus(tmp_path)
    (tmp_path / "audio" / "b2.wav").unlink()
    with pytest.raises(FileNotFoundError, match=r"b2\.wav"):
        extract_corpus(pairs, corpus="local", encoder="toy",
                       audio_root=tmp_path, store_root=tmp_path / "data",
                       device="cpu")
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
uv run pytest tests/test_extract_run.py -x -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'trackb.extract.run'`

- [x] **Step 3: Implement the minimum to make the test pass**

```python
# src/trackb/extract/run.py
"""Extraction CLI: pairs + audio files -> verified feature store.

Same entrypoint locally (Mac overnight, --device mps --batch note: encoder
forward is per-clip here; MPS wired-memory only bites at batch>2 which we do
not use) and in a cloud job (--device cuda). Resume-safe: existing keys skip.

AIME segment refs ("aime:{track_id}:{begin}-{end}") are sliced after decode.
clip_key = sha256(file bytes) for whole files, sha256(file bytes + span) for
segments; text keys = sha256("text:" + prompt)."""
import argparse
import hashlib
import json
from pathlib import Path

from trackb.extract.encoders import embed, load_audio, text_embed
from trackb.extract.store import FeatureStore
from trackb.schema import PairRecord


def _hms_to_seconds(hms: str) -> float:
    h, m, s = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _resolve(clip_ref: str, audio_root: Path) -> tuple[Path, tuple[float, float] | None]:
    if clip_ref.startswith("aime:"):
        _, track_id, span = clip_ref.split(":", 2)
        begin, end = span.rsplit("-", 1)
        matches = sorted(audio_root.glob(f"{track_id}.*"))
        if not matches:
            raise FileNotFoundError(
                f"no audio file for AIME track {track_id} under {audio_root}")
        return matches[0], (_hms_to_seconds(begin), _hms_to_seconds(end))
    path = audio_root / clip_ref
    if not path.exists():
        raise FileNotFoundError(f"audio file missing: {path}")
    return path, None


def _clip_key(path: Path, span: tuple[float, float] | None) -> str:
    h = hashlib.sha256(path.read_bytes())
    if span is not None:
        h.update(f"span:{span[0]}-{span[1]}".encode())
    return h.hexdigest()


def extract_corpus(pairs: list[PairRecord], corpus: str, encoder: str,
                   audio_root: Path | str, store_root: Path | str,
                   device: str = "cpu") -> dict:
    audio_root, store = Path(audio_root), FeatureStore(store_root)
    clip_refs = sorted({ref for p in pairs for ref in
                        (p.clip_a, p.clip_b) + ((p.ref_audio,) if p.ref_audio else ())})
    entries: dict[str, str] = {}
    clips_embedded = 0
    dim = None
    for ref in clip_refs:
        path, span = _resolve(ref, audio_root)
        key = _clip_key(path, span)
        entries[ref] = key
        if store.put_needed(key, encoder):
            wav, sr = load_audio(path)
            if span is not None:
                wav = wav[int(span[0] * sr):int(span[1] * sr)]
            vec = embed(encoder, wav, sr, device=device)
            store.put(key, encoder, vec)
            clips_embedded += 1
            dim = len(vec)
        elif dim is None:
            dim = len(store.get(key, encoder))
    store.write_manifest(encoder, corpus, entries, dim=dim or 0)

    ctx_entries: dict[str, str] = {}
    prompts_embedded = 0
    ctx_dim = None
    for p in pairs:
        key = hashlib.sha256(("text:" + p.prompt).encode("utf-8")).hexdigest()
        ctx_entries[p.pair_id] = key
        if store.put_needed(key, encoder):
            vec = text_embed(encoder, p.prompt, device=device)
            store.put(key, encoder, vec)
            prompts_embedded += 1
            ctx_dim = len(vec)
        elif ctx_dim is None:
            ctx_dim = len(store.get(key, encoder))
    store.write_manifest(encoder, f"{corpus}__text", ctx_entries, dim=ctx_dim or 0)
    return {"clips_embedded": clips_embedded, "prompts_embedded": prompts_embedded,
            "clips_total": len(clip_refs), "pairs_total": len(pairs)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--corpus", required=True,
                        choices=["cmi_pref_train", "cmi_pref_test",
                                 "aime_survey", "music_arena"])
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--store-root", default="data")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--music-arena-config", default="2025_12")
    args = parser.parse_args()
    if args.corpus == "cmi_pref_train":
        from trackb.corpora.cmi_pref import load_split
        pairs, corpus = load_split("train"), "cmi_pref_train"
    elif args.corpus == "cmi_pref_test":
        from trackb.corpora.cmi_pref import load_split
        pairs, corpus = load_split("test"), "cmi_pref_test"
    elif args.corpus == "aime_survey":
        from trackb.corpora.aime_survey import load
        pairs, corpus = load(), "aime_survey"
    else:
        from trackb.corpora.music_arena import load
        pairs, corpus = load(args.music_arena_config), "music_arena"
    stats = extract_corpus(pairs, corpus=corpus, encoder=args.encoder,
                           audio_root=args.audio_root,
                           store_root=args.store_root, device=args.device)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
```

Add to `src/trackb/extract/store.py` (one method on `FeatureStore`):

```python
    def put_needed(self, clip_key: str, encoder: str) -> bool:
        return not self._vec_path(clip_key, encoder).exists()
```

Append to `justfile`:

```justfile
extract encoder corpus audio_root device="cpu":
    uv run python -m trackb.extract.run --encoder {{encoder}} --corpus {{corpus}} --audio-root {{audio_root}} --device {{device}}
```

- [x] **Step 4: Run test — verify it PASSES**

```bash
uv run pytest tests/test_extract_run.py tests/test_store.py -x -q
```
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add src/trackb/extract tests/test_extract_run.py justfile && git commit -m "feat(extract): resume-safe extraction CLI with AIME segment slicing"
```

---

### Task 20: MIREX contract entrypoint
**Group:** F (sequential; after Task 19)

**Behavior being verified:** `main.py` reads MIREX input rows `{sample_id, prompt, audio_A, audio_B}` (+ optional `lyrics`, `ref_audio` — ignored gracefully when absent), scores them with a saved checkpoint, and writes `{sample_id, preferred_candidate}` rows; malformed rows fail loudly naming the row; exact ties emit "A" with a warning; output is deterministic.
**Interface under test:** `main.run` (and the `python main.py --path input.jsonl` CLI it wraps).

**Files:**
- Create: `main.py`, `tests/fixtures/toy_input.jsonl`
- Test: `tests/test_main_contract.py`

- [x] **Step 1: Write the failing test**

Create `tests/fixtures/toy_input.jsonl` (the P0 toy contract shape):

```json
{"sample_id": "toy-001", "prompt": "an upbeat synthwave track with driving bass", "audio_A": "audio/a1.wav", "audio_B": "audio/b1.wav"}
{"sample_id": "toy-002", "prompt": "a slow melancholic piano ballad", "audio_A": "audio/a2.wav", "audio_B": "audio/b2.wav", "lyrics": "some words"}
```

```python
# tests/test_main_contract.py
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from main import run
from trackb.train.trainer import TrainConfig, train
from tests.fixtures.make_fixtures import build_fixture_corpus

FIXTURE_INPUT = Path(__file__).parent / "fixtures" / "toy_input.jsonl"


def build_checkpoint(tmp_path):
    pairs, store = build_fixture_corpus(tmp_path / "fx", n_pairs=48, seed=7)
    config = TrainConfig(encoder="toy", head_width=32, head_depth=2, joint=0,
                         lr=0.01, epochs=50, batch_size=16, sampler="uniform",
                         adversarial_lambda=0.0, seed=3, k_folds=2,
                         final_generators=("g3",))
    return train(config, pairs, store, corpus="fixture",
                 out_dir=tmp_path / "run").checkpoint_path


def make_eval_audio(root):
    (root / "audio").mkdir(parents=True)
    rng = np.random.default_rng(1)
    for name in ("a1.wav", "b1.wav", "a2.wav", "b2.wav"):
        sf.write(root / "audio" / name,
                 rng.standard_normal(16000).astype(np.float32), 16000)


def test_contract_end_to_end_and_deterministic(tmp_path):
    ckpt = build_checkpoint(tmp_path)
    make_eval_audio(tmp_path)
    out1, out2 = tmp_path / "pred1.jsonl", tmp_path / "pred2.jsonl"
    n = run(FIXTURE_INPUT, out1, ckpt, audio_root=tmp_path)
    run(FIXTURE_INPUT, out2, ckpt, audio_root=tmp_path)
    assert n == 2
    rows = [json.loads(line) for line in out1.read_text().splitlines()]
    assert [r["sample_id"] for r in rows] == ["toy-001", "toy-002"]
    assert all(r["preferred_candidate"] in ("A", "B") for r in rows)
    assert out1.read_text() == out2.read_text()


def test_malformed_row_fails_naming_row(tmp_path):
    ckpt = build_checkpoint(tmp_path)
    make_eval_audio(tmp_path)
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"sample_id": "x1", "prompt": "no audio fields"}\n')
    with pytest.raises(ValueError, match=r"x1.*audio_A"):
        run(bad, tmp_path / "out.jsonl", ckpt, audio_root=tmp_path)


def test_missing_checkpoint_fails_loudly(tmp_path):
    make_eval_audio(tmp_path)
    with pytest.raises(FileNotFoundError, match=r"checkpoint"):
        run(FIXTURE_INPUT, tmp_path / "out.jsonl",
            tmp_path / "nope" / "checkpoint.pt", audio_root=tmp_path)
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
uv run pytest tests/test_main_contract.py -x -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'main'`

- [x] **Step 3: Implement the minimum to make the test pass**

```python
# main.py
"""MIREX 2026 Track B submission entrypoint.

Contract: python main.py --path input.jsonl
Input rows:  {sample_id, prompt, audio_A, audio_B[, lyrics, ref_audio]}
Output rows: {sample_id, preferred_candidate: "A"|"B"}  ->  predictions.jsonl

Graceful degrade: optional fields absent -> scored without them. Loud failure:
malformed rows, missing audio, missing checkpoint. Exact ties -> "A" plus a
warning (deterministic; never random). Eval-day note: --device mps on the
32 GB Mac, sequential per-clip encoding (P0 wired-memory lesson)."""
import argparse
import json
import logging
import sys
from pathlib import Path

from trackb.evalx.score import load_head
from trackb.extract.encoders import embed, load_audio, text_embed

logger = logging.getLogger("trackb.submit")

REQUIRED_FIELDS = ("sample_id", "prompt", "audio_A", "audio_B")


def run(input_path: Path | str, output_path: Path | str,
        checkpoint_path: Path | str, audio_root: Path | str = ".",
        device: str = "cpu") -> int:
    import torch
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    head, ckpt = load_head(checkpoint_path)
    encoder = ckpt["encoder"]
    audio_root = Path(audio_root)
    n = 0
    with open(output_path, "w") as out:
        for i, line in enumerate(Path(input_path).read_text().splitlines()):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = [f for f in REQUIRED_FIELDS if f not in row]
            if missing:
                raise ValueError(
                    f"row {row.get('sample_id', f'#{i}')}: missing required "
                    f"field(s) {missing} - refusing to guess (need "
                    f"{list(REQUIRED_FIELDS)})")
            feats = []
            for field in ("audio_A", "audio_B"):
                wav, sr = load_audio(audio_root / row[field])
                feats.append(torch.from_numpy(
                    embed(encoder, wav, sr, device=device)).unsqueeze(0))
            ctx = torch.from_numpy(
                text_embed(encoder, row["prompt"], device=device)).unsqueeze(0)
            with torch.no_grad():
                margin = float(head.margin(feats[0], feats[1], ctx))
            if margin == 0.0:
                logger.warning("exact tie on %s - emitting 'A' deterministically",
                               row["sample_id"])
            preferred = "A" if margin >= 0 else "B"
            out.write(json.dumps({"sample_id": row["sample_id"],
                                  "preferred_candidate": preferred}) + "\n")
            n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="input JSONL")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--output", default="predictions.jsonl")
    parser.add_argument("--audio-root", default=".")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    n = run(args.path, args.output, args.checkpoint,
            audio_root=args.audio_root, device=args.device)
    print(f"wrote {n} predictions to {args.output}")


if __name__ == "__main__":
    main()
```

- [x] **Step 4: Run test — verify it PASSES**

```bash
uv run pytest tests/test_main_contract.py -x -q && uv run pytest -x -q
```
Expected: PASS (full suite green)

- [x] **Step 5: Commit**

```bash
git add main.py tests/fixtures/toy_input.jsonl tests/test_main_contract.py && git commit -m "feat(submit): MIREX contract entrypoint with loud failures + deterministic ties"
```

---

### Task 21: Crescendai-side wiring + real first config
**Group:** G (sequential; after Group F). Runs in the CRESCENDAI worktree `/Users/jdhiman/Documents/crescendai/.worktrees/issue-106-mirex-trackb` for the two crescendai files, and in the nested repo for the config/README/justfile.

**Behavior being verified:** the parent repo ignores the nested repo (`git check-ignore` exits 0); the living doc records the build; the nested repo carries the real wave-1 config and complete just recipes.

**Files:**
- Modify: `crescendai/.gitignore` (worktree copy), `crescendai/docs/mirex/track-b-cmi-rewardbench.md` (worktree copy)
- Create (nested repo): `configs/first.json`
- Modify (nested repo): `justfile`, `README.md`

- [x] **Step 1: Write the failing check**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-106-mirex-trackb && git check-ignore mirex-trackb; echo "exit=$?"
```
Expected: `exit=1` (not yet ignored — the "failing test")

- [x] **Step 2: Implement**

Append to `/Users/jdhiman/Documents/crescendai/.worktrees/issue-106-mirex-trackb/.gitignore`:

```gitignore
# nested standalone repo (MIREX Track B, issue #106) - own git history
mirex-trackb/
```

Append to the decision log of `/Users/jdhiman/Documents/crescendai/.worktrees/issue-106-mirex-trackb/docs/mirex/track-b-cmi-rewardbench.md` — note: this worktree branched from main, which does NOT contain the doc (it lives on `issue-105-mirex-rewardbench`, unmerged). First copy it over, then append:

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-106-mirex-trackb
git checkout issue-105-mirex-rewardbench -- docs/mirex/
```

```markdown
- **2026-07-04 (Option C approved — build started, issue #106)** — Brainstorm + /plan complete. Win condition re-scoped: LBD credential (finding > rank); license fork dissolved (NC data fine for research submission); LLM-judge stream killed at <=$100 budget. System = extract-once feature cache (CLAP/MERT/MuQ-MuLan; CMI-Pref local overnight, AIME-survey + Music Arena cloud ~$25-40) -> BT-head sweeps on cached features (Mac, autoresearch, nested generator-holdout validation, public-500 budget <=5 evals) -> generator-identity shortcut probe + mitigation as the LBD headline. Code: NESTED STANDALONE REPO at `crescendai/mirex-trackb/` (own git history, Apache-2.0, private->public). Spec: docs/specs/2026-07-04-mirex-trackb-build-design.md; plan: docs/plans/2026-07-04-mirex-trackb-build.md (both on branch issue-106-mirex-trackb). Corpus schemas verified against real data (CMI-Pref via P0 scratch; AIME-survey + Music Arena via HF datasets-server): generator identity present in ALL three (model_a/b, model-1/2, system_a/b).
```

Create `configs/first.json` in the nested repo (the first real experiment — CLAP over CMI-Pref train):

```json
{
  "encoder": "clap",
  "head_width": 256,
  "head_depth": 2,
  "joint": 0,
  "lr": 0.001,
  "epochs": 100,
  "batch_size": 64,
  "sampler": "uniform",
  "adversarial_lambda": 0.0,
  "seed": 3,
  "k_folds": 4,
  "final_generators": []
}
```

(`final_generators` is intentionally empty in the committed config: the real generator list is only known after the first extraction; the training CLI below requires it to be non-empty and refuses otherwise, so the user consciously picks the final fold from the real generator inventory — never a silent default.)

Replace the nested repo `justfile` with the complete recipe set:

```justfile
test:
    uv run pytest -x -q

smoke:
    uv run pytest tests/test_smoke.py -x -q

extract encoder corpus audio_root device="cpu":
    uv run python -m trackb.extract.run --encoder {{encoder}} --corpus {{corpus}} --audio-root {{audio_root}} --device {{device}}

train config="configs/first.json":
    uv run python -m trackb.train.cli --config {{config}}

eval-ratchet:
    uv run python -c "from trackb.evalx.ratchet import check; import json; print(json.dumps(check('data/evals/last_run.json', 'data/evals/baseline.json')))"

eval-promote:
    uv run python -c "from trackb.evalx.ratchet import promote; promote('data/evals/last_run.json', 'data/evals/baseline.json'); print('promoted')"
```

Create `src/trackb/train/cli.py` in the nested repo (the thin CLI the `train` recipe calls — glue only, all behavior already tested through `train()`):

```python
# src/trackb/train/cli.py
"""CLI glue: config JSON -> train() -> data/evals/last_run.json."""
import argparse
import json
from pathlib import Path

from trackb.corpora.cmi_pref import load_split
from trackb.extract.store import FeatureStore
from trackb.train.trainer import TrainConfig, train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--store-root", default="data")
    parser.add_argument("--corpus", default="cmi_pref_train")
    parser.add_argument("--out-dir", default="checkpoints/latest")
    args = parser.parse_args()
    raw = json.loads(Path(args.config).read_text())
    raw["final_generators"] = tuple(raw["final_generators"])
    config = TrainConfig(**raw)
    if not config.final_generators:
        raise ValueError(
            "final_generators is empty - pick the final holdout generators "
            "from the real generator inventory before training (see README)")
    pairs = load_split("train")
    result = train(config, pairs, FeatureStore(args.store_root),
                   corpus=args.corpus, out_dir=args.out_dir)
    evals = Path("data/evals")
    evals.mkdir(parents=True, exist_ok=True)
    (evals / "last_run.json").write_text(json.dumps(result.metrics, indent=1))
    print(json.dumps(result.metrics))


if __name__ == "__main__":
    main()
```

Update the nested repo `README.md` — replace the body with:

```markdown
# trackb

MIREX 2026 Track B (CMI-RewardBench) submission: frozen-encoder + Bradley-Terry
preference head, with a generator-identity shortcut study. Apache-2.0 code;
training-data licenses documented per-corpus in src/trackb/corpora/ (CMI-Pref
is CC-BY-NC-SA: research use, disclosed per MIREX rules).

## Commands
- `just test` - behavior suite (no GPU, no downloads)
- `just smoke` - planted-signal end-to-end gate
- `just extract clap cmi_pref_train <audio_root> [device]` - feature extraction (resume-safe)
- `just train configs/first.json` - train + write data/evals/last_run.json
- `just eval-ratchet` / `just eval-promote` - compare/promote against baseline
- `python main.py --path input.jsonl` - MIREX contract inference

## Wave-1 recipe
1. Download CMI-Pref audio (see the crescendai living doc P0 recipe - curl
   /resolve fallback, ffprobe-verify) to a local audio root.
2. Overnight: `just extract clap cmi_pref_train <root> mps` (also run
   `cmi_pref_test` once - it feeds the budgeted public evals only).
3. Inspect the generator inventory, set final_generators in configs/first.json.
4. `just train` -> `just eval-ratchet` -> iterate (or point /autoresearch at it).
```

- [x] **Step 3: Run checks — verify they PASS**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-106-mirex-trackb && git check-ignore mirex-trackb && echo IGNORED
cd /Users/jdhiman/Documents/crescendai/mirex-trackb && uv run pytest -x -q
```
Expected: `IGNORED` printed; full nested suite PASS.

- [x] **Step 4: Commit (both repos)**

```bash
cd /Users/jdhiman/Documents/crescendai/mirex-trackb && git add -A && git commit -m "feat(ops): train CLI, wave-1 config, complete just recipes"
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-106-mirex-trackb && git add .gitignore docs/mirex/ && git commit -m "docs(#106): ignore nested mirex-trackb repo + living-doc build entry"
```

---

## Plan self-review notes

- **Spec coverage:** schema→T2; store→T3; budget→T4; ratchet→T5; folds→T6; fixtures/harness→T7 (Task Group 0 role is split across T1+T7+T12: scaffold, fixtures, smoke gate); encoders→T8; heads→T9/T10; trainer→T11; smoke canonical state→T12; probe→T13; decomposition→T14; mitigation→T15; three adapters→T16-18; extraction CLI→T19; MIREX contract→T20; crescendai wiring + first real config→T21. Spec's "both-orderings averaging iff joint" is satisfied structurally: the joint head is antisymmetrized inside `margin` (T9/T10), so both orderings are always averaged — no separate submit-time branch needed.
- **Type consistency:** `TrainConfig` fields identical in T11/T12/T14/T15/T21 config JSON; `_fit` signature change in T15 is propagated to its only other caller (`decompose`, noted in T15); `FeatureStore.put_needed` added in T19 is additive.
- **Group correctness:** Group A tasks touch disjoint files; Group B disjoint; C and D sequential (shared `head.py`/`trainer.py`); E disjoint; F sequential (T19 touches `store.py` — no Group A task is still running by then).
- **Known deliberate limits:** real-encoder paths (clap/mert/muq_mulan) and HF `load()`/`load_split()` network paths are exercised by the first real extraction, not unit tests; Music Arena's preference vocabulary and AIME's answer codes are pinned pessimistically (unknown values raise). MuQ-MuLan requires the `muq` package (vendored in the P0 scratch CMI-RewardBench repo if the PyPI package is unavailable) — cloud-wave concern, not wave-1-local.

## Challenge Review

### CEO Pass

**Premise Challenge.** The problem is real and specific: issue #105's P0 proved the CMI-RewardBench harness works but the reproduced 77.8% belongs to QMUL's checkpoint, not to CrescendAI — no submission is possible without an owned system. The plan is a direct path to that goal (extract-once feature cache → sweepable BT head → generator-holdout validation → shortcut probe), not a proxy. No existing crescendai code partially solves this (this is a new research artifact, correctly isolated as a nested repo so it doesn't entangle with the product monorepo). Verdict: right problem, direct path.

**Scope Check.** The plan matches the spec's Modules section field-for-field (verified by re-reading `docs/specs/2026-07-04-mirex-trackb-build-design.md` Modules and File Changes tables against the plan's 21 tasks — every module/adapter/file in the spec maps to exactly one task, see the plan's own "Plan self-review notes: Spec coverage" line, which I independently confirmed). The hardest problem (generator-holdout fold purity, position-bias-immune head, content-hash resume-safe extraction) is being solved, not avoided — Tasks 6, 9/10, and 3 are exactly the guts of this. 20 files is a lot, but this is infra for a multi-corpus, multi-encoder research pipeline; the module count is justified by the spec's own DEEP/thin-by-design rationale, not accidental complexity. Minimum viable version: Group 0-C alone (scaffold → schema → store → budget → ratchet → folds → fixtures → encoders → heads → trainer → smoke) already proves the full loop mechanics on synthetic data — the plan itself flags this as "[SHIPS INDEPENDENTLY]," which is the right MVP framing.

**Twelve-Month Alignment.**
```
CURRENT STATE                         THIS PLAN                              12-MONTH IDEAL
No owned preference model;       →    Frozen-encoder + BT head,         →    Submitted MIREX entry + LBD
CMI-RM's checkpoint borrowed           nested repo, generator-holdout          finding; reusable pattern for
for P0 reproduction only.              validation, shortcut-probe study.       any future preference-model work.
```
This moves toward the ideal without creating monorepo tech debt — the nested-repo isolation is the correct call given this is a one-off competition submission, not a product feature.

**Alternatives Check.** The spec's "Not in scope" section and the living doc's "Approach ranking" (LLM-judge ~60-70% ceiling; fusion meta-judge gains inside seed noise; no proprietary encoder) function as the alternatives analysis, with rejection reasoning recorded. This satisfies the alternatives check — no `[QUESTION]` needed here.

### Engineering Pass

**Architecture.** Data flow is clean and matches the "compute cliff" framing: `corpora/*.py` → `PairRecord` → `extract/run.py` (embeds once, content-hash keyed) → `FeatureStore` → `train/trainer.py` (assembles cached tensors, never re-embeds) → `evalx/{folds,score,ratchet,budget}` → `probe/*` and `main.py` as pure consumers. Traced end-to-end for both the smoke path (Tasks 7→11→12) and the real path (Tasks 16-18→19→21); no divergence found between the two.

```
corpora adapter → PairRecord → validate_pairs → extract.run (embed once)
                                                        ↓
                                                  FeatureStore (content-hash)
                                                        ↓
                                    trainer.assemble → BTHead.margin → BCE loss
                                                        ↓
                                    evalx.folds (generator-holdout) → score/ratchet/budget
                                                        ↓
                                              probe.generator_id / decompose
                                                        ↓
                                         main.py (checkpoint + encoder → predictions.jsonl)
```
Error path: `FeatureStore.verify` and corpus adapters raise loudly and are not caught anywhere upstream (no blanket `except`), so failures surface to the CLI/test runner — matches the project's explicit-exception-over-fallback preference. No SQL/shell/prompt-injection surface; `audio_root / clip_ref` path joins use corpus-provided strings without traversal sanitization, but the corpora are trusted HF datasets, not untrusted user input, so this is `[OBS]` not a finding.

**Module Depth Audit** (spot-check on the modules with the most exported surface):
- `schema.py`: interface = `PairRecord`, `validate_pairs`, `SchemaError` (3 symbols); hides per-field/dedup invariants across all corpora. DEEP.
- `extract/store.py`: interface = `put/get/write_manifest/verify` (+ `put_needed` added in T19) — 5 methods; hides on-disk layout, resume semantics, manifest integrity. DEEP.
- `extract/encoders.py`: interface = `embed/text_embed/load_audio` (3 functions) fronting 4 registered encoders each with device placement, chunking, HF loading. DEEP.
- `train/head.py` + `train/trainer.py`: interface = `BTHead.margin/penultimate`, `TrainConfig`, `train()` — hides antisymmetrization, k-fold orchestration, adversarial gradient-reversal, sampler weighting. DEEP, though `trainer.py`'s `_fit` (Task 15 version) has grown to ~50 lines with three responsibilities (fitting, sampling-stat collection, adversarial term) threaded through optional params (`pairs=None, stats=None`) — borderline SHALLOW-by-accretion. Not a blocker; flagged as a `[RISK]` below (Code Quality).
- `corpora/*.py` (cmi_pref, aime_survey, music_arena): interface = one `rows_to_pairs` + one `load`/`load_split` per module; genuinely shallow but explicitly justified in the spec ("depth here would mean duplicating validation three times") — this is the correct call, not a smell.

**Code Quality.**
- `[RISK]` (confidence: 6/10) — `trainer._fit` (post-Task-15) takes on three concerns (model fit, sample-stat bookkeeping, adversarial loss) via optional `pairs`/`stats` params with `if adv_head is not None and pairs is not None and config.joint == 0` gating. Adversarial mitigation silently has no effect when `joint=1` (the cross-encoder head) — there is no test combining `adversarial_lambda > 0` with `joint=1`, and no error is raised to say "adversarial mitigation is pointwise-only." Watch during Task 15 execution: either add a loud `ValueError` when `adversarial_lambda > 0 and joint == 1`, or add a test/comment documenting the restriction is intentional. Fallback: leave as-is for wave 1 (pointwise head is the default anyway) but note it in the nested repo's README so a future config sweep doesn't waste a cloud run on a no-op combination.
- `[OBS]` — `Folds` construction in Task 6 (`rest = [p for p in pairs if p not in touches_final]`) is O(n·m) via dataclass equality rather than an id/pair_id-keyed set; fine for the 48-pair test fixture and probably fine for wave-1's ~4-20k real pairs, but worth a comment if a corpus grows past ~100k pairs.
- `[RISK]` (confidence: 7/10) — no bare `except Exception` patterns found anywhere in the plan's code — good adherence to the project's explicit-exception preference. No further code-quality violations found on inspection of all 21 tasks' listed source.

**Test Philosophy Audit.** All tests exercise public module interfaces (`validate_pairs`, `FeatureStore.put/get/verify`, `BTHead.margin`, `train()`, `generator_id_probe`, `run()` for `main.py`) with real in-memory/tmp_path fixtures — no internal collaborator mocking anywhere in the 21 tasks. This is a clean, behavior-first test suite. No `[BLOCKER]`s here.

**Vertical Slice Audit.** 19 of 21 tasks are clean one-test→one-impl→one-commit slices, verified by reading each task's Steps 1-5. The two flagged exceptions:
- Task 10 (joint-head antisymmetry test may pass without new code): justification given is that Task 9's diff already contains the joint-mode implementation, and the plan explicitly instructs verifying the test *would* fail against a pointwise-only head (temporarily setting `joint=0`) to prove it's a real behavioral pin, not a tautology. This is an adequate justification — the test is still checking real behavior (cross-clip interaction), just landing one task after the code that implements it. Not flagged.
- Task 21 (git-check-ignore "test"): adequate for the `.gitignore` line itself. However, Task 21's Step 2 also creates `src/trackb/train/cli.py` — genuinely new application logic (loads a JSON config, constructs `TrainConfig`, and raises `ValueError` if `final_generators` is empty) — under the umbrella of an "ops" task whose only verification is `git check-ignore` + `uv run pytest -x -q` (the *existing* suite, which does not import or exercise `cli.py` at all). See Test Coverage Gaps below — flagged as `[RISK]`, not `[BLOCKER]`, since the guard clause is simple and every function `cli.py` calls (`train`, `TrainConfig`, `FeatureStore`) is already tested elsewhere; the untested surface is a single one-line invariant check.

**Test Coverage Gaps.**
```
[+] src/trackb/extract/run.py
    │
    ├── extract_corpus() — plain clip_ref path ("audio/a1.wav")
    │   ├── [TESTED] ★★  embed + write manifest — Task 19 test
    │   ├── [TESTED] ★★  resume (second run re-embeds 0) — Task 19 test
    │   └── [TESTED] ★★  missing audio file raises FileNotFoundError — Task 19 test
    │
    └── extract_corpus() / _resolve() — "aime:{track_id}:{begin}-{end}" path
        ├── [GAP] _hms_to_seconds() — no test at all (any format, boundary values)
        ├── [GAP] audio_root.glob(f"{track_id}.*") file-matching — no test; assumes
        │         one file per track_id named exactly "{track_id}.<ext>", unverified
        │         against the real disco-eth/AIME audio corpus's actual file-naming
        │         convention
        └── [GAP] wav[int(span[0]*sr):int(span[1]*sr)] slicing — no test for the
                  aime clip_ref branch at all; test_extract_run.py only builds
                  plain-named wavs, never an "aime:" ref

[+] src/trackb/train/cli.py (Task 21)
    └── main() / final_generators empty guard
        └── [GAP] no test imports or calls this module at all
```
`[RISK]` (confidence: 8/10) — the AIME segment-resolution branch of `extract/run.py` (`_hms_to_seconds`, the `aime:` half of `_resolve`, and the slice-by-span logic) is completely unexercised by any test in the plan. This is not a peripheral path — AIME-survey is one of three wave-1 corpora and Task 17 explicitly encodes segment spans into `clip_a`/`clip_b` (`"aime:5331:00:00:53-00:01:03"`) specifically so extraction can slice them, but Task 19's test corpus never constructs an "aime:" ref. Since this is a real-data-only path (per the plan's own "known deliberate limits" note that real-encoder/network paths are validated at first real extraction, not in CI), this may be an intentional deferral consistent with that pattern — but unlike the encoder-registry deferral (which only skips GPU-dependent embedding, with the surrounding logic tested via the "toy" encoder), here the entire glob-matching and time-slicing logic is untested by *anything*, including the toy encoder. Recommend adding one `test_extract_run.py` case with a synthetic `"aime:track1:00:00:00-00:00:01"` ref and a `track1.wav` fixture file before the real Task 17→19 AIME extraction run — cheap insurance against a silent wrong-file-match or off-by-one slice during the actual overnight/cloud extraction (which is expensive to re-run and burns calendar time before the ~Jul 24 fork decision).

**Failure Modes.** Reviewed per task: `FeatureStore.verify` (Task 3) refuses assembly loudly on incomplete manifests — no silent partial state. `PublicEvalBudget` (Task 4) persists to disk before allowing a 6th spend — a crash after `spend()` writes but before the caller uses the result would just look like one used spend, not a corruption. `train()` (Task 11/15) writes `checkpoint.pt` and `metrics.json` only after all folds complete; a mid-training crash leaves `out_dir` empty or partially populated but never a checkpoint that silently doesn't match its metrics (both write in the same function, back to back, no separate transaction boundary needed since there's no concurrent writer). `main.py::run()` (Task 20) raises loudly on missing checkpoint, malformed rows, and missing audio; writes rows to `output_path` incrementally inside the loop, so a mid-run crash leaves a truncated-but-valid-prefix `predictions.jsonl` rather than a corrupt file — acceptable for a non-interactive batch CLI. No silent failures identified in any of the 21 tasks.

### Presumption Inventory

| ASSUMPTION | VERDICT | REASON |
|---|---|---|
| CMI-Pref real row schema (`audio-path`, `model_a`, `preference-musicality`, etc.) matches Task 16's fixture | SAFE | Explicitly verified 2026-07-04 against the P0 scratch `all_test.jsonl`, per the plan's own comment and the living doc's P0 section. |
| AIME-survey schema (`question-type`, `track-1-id`, `answer` ∈ {1,2}) | VALIDATE | Verified via HF datasets-server per the plan's comment, but the plan itself flags that tie/other answer codes are unconfirmed — hence the loud-raise design (correct mitigation), but the *audio file naming convention* for `track-{1,2}-id` is NOT verified anywhere (see Test Coverage Gaps `[RISK]` above). |
| Music Arena preference vocabulary is exactly `{A, B, TIE, BOTH_BAD}` | VALIDATE | Plan's own docstring says this is "pinned optimistically... per the published dataset description" — the loud-raise-on-unknown design correctly surfaces a wrong guess at first real extraction rather than silently misclassifying, so the risk is contained even if the assumption is wrong. |
| `guard-primary-tree-edits.py` judges files by nearest enclosing repo, so nested-repo edits on `build/initial-system` are not blocked by the crescendai hook | VALIDATE | Plan states this was "audited" this session; I did not independently re-verify the hook's source in this review (out of scope of the plan's own file list) — worth a quick sanity check (`echo test >> mirex-trackb/README.md` then attempt a Write) before Task 1 begins, since if wrong it blocks the entire Group 0-F execution. |
| Adversarial mitigation (Task 15) is meaningful only for `joint=0` and this restriction needs no user-facing signal | RISKY | No error raised when `adversarial_lambda > 0` is combined with `joint=1`; silently a no-op. See Code Quality `[RISK]` above. |
| `torch.load(..., weights_only=True)` (Task 12's `score.py`) can deserialize the Task 11/15 checkpoint dict (nested dataclass-derived dict, tensors, str/float primitives) | SAFE | `weights_only=True` in modern torch supports dicts of tensors + Python primitives (the checkpoint's `asdict(config)` is exactly that); no custom classes are pickled into the checkpoint. |
| `mirex-trackb`'s own `.gitignore` (Task 1) correctly excludes `data/`, `checkpoints/`, `results/` so multi-GB feature caches and audio never get committed to the nested repo (which will go private→public on GitHub) | SAFE | Explicitly listed in Task 1's `.gitignore` content; consistent with the spec's "Raw bulk audio never lands on the Mac... ship home only feature tensors" stance, which still must not be *committed* even though it's on disk — worth a final `git status` sanity check before the private→public flip, but the ignore rules as written are correct. |

### Summary
[BLOCKER] count: 0
[RISK]    count: 4
[QUESTION] count: 0

### Cautions
No blockers. The plan is well-scoped, correctly isolates a one-off research artifact from the product monorepo, and its test suite is genuinely behavior-first with no internal-mocking or shape-testing violations. The four risks to watch during execution: (1) the AIME segment-extraction code path (`_hms_to_seconds`, glob-by-track-id, time-slicing) has zero test coverage anywhere in the plan and should get one cheap synthetic-fixture test added to Task 19 before the real AIME extraction burns calendar time on a possible silent wrong-file-match or off-by-one slice; (2) Task 21's `train/cli.py` ships a untested (if simple) `final_generators`-empty guard under an "ops" justification that technically covers only the `.gitignore` line, not this new file; (3) the adversarial-mitigation term silently no-ops when combined with the joint/cross-encoder head, with no error or test pinning that restriction; (4) the "guard-primary-tree-edits.py judges by nearest enclosing repo" claim that justifies skipping a crescendai worktree for Tasks 1-20 was asserted as "audited this session" but not independently re-verified in this review — worth one quick sanity check before Task 1's first commit.

VERDICT: PROCEED_WITH_CAUTION — monitor: (1) AIME segment-slicing path untested, add a fixture before real extraction; (2) Task 21's cli.py final_generators guard is untested; (3) adversarial_lambda silently no-ops under joint=1; (4) re-verify the guard-primary-tree-edits.py nested-repo behavior before Task 1's first edit.
