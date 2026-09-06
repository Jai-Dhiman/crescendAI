# #175: 5-Adapter Pred-Mean Ensemble in the Submission Container — Implementation Plan

> **Revision 2** — reworked after `/challenge` returned `NEEDS_REWORK` on
> revision 1 with five blockers. What changed is recorded under
> *Round-1 blockers and how this revision resolves them*; the full round-1
> review is preserved as an appendix.

> **For the build agent:** one task = **one failing test** = one implementation
> = one commit. No task writes a second test. Dispatch groups per the dependency
> map; tasks within a group touch the same file and run in file order. Do NOT
> start execution until `/challenge` returns `VERDICT: PROCEED`.

**Goal:** `predict.sh <wav>` inside the submission image prints one float that is
the mean of five members' predictions, with exactly one transcription, one
tokenization and one 839M base load per run — and a pre-existing single-adapter
model directory still scores unchanged.

**Spec:** `docs/specs/2026-09-05-ensemble-container-design.md` (APPROVED)
**Issue:** [#175](https://github.com/Jai-Dhiman/crescendAI/issues/175), sub-issue of #104 (S5). Deadline 2026-10-01.
**Style:** follow the conventions already in `model/src/claim_measurement/difficulty/`
— plain pytest, no mocks of internal collaborators, fakes injected only at
process boundaries (`loader_factory`, `transcribe`), and a docstring on every
test saying *why the behaviour matters*.

## The step cycle (every task, no exceptions)

1. Write **one** failing test.
2. Run it and **verify it fails for the stated reason** — not on an import error
   from a typo, and not passing by accident. If it passes before the
   implementation, the test is pinning shape rather than behaviour: fix the test.
3. Implement the minimum that makes it pass.
4. Run the full suite: `cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov`
5. Commit with the message given in the task.

Baseline on this branch is **246 passed / 6 skipped**, measured 2026-09-05 by
running the command above in this worktree. (Issue #175's STATE and revision 1
of this plan both said 172/4; that number was stale. If your run does not start
from 246/6, stop and say so rather than adjusting the target.) Every task grows
the passed count and shrinks it never.

## Hard constraints (do not violate in any task)

- **No task may report a tau-c from these artifacts.** Every labelled piece is
  in every Experiment B run's training set. The honest headroom number is the
  label ceiling's `+0.0239` (#168).
- **This worktree's `model/data/` is EMPTY** (gitignored; the real tree is
  `/Users/jdhiman/Documents/crescendai/model/data`). Every automated test builds
  synthetic fixtures under `tmp_path`. The opt-in suites that need real
  artifacts are run with `CRESCENDAI_DATA_ROOT` pointed at the primary checkout.
- **Never `uv run --with` from inside `model/`** — it mutates the shared `.venv`.
  A new dependency must be added to both `score_wav.py`'s and
  `audio_emb_extract.py`'s `# /// script` headers (the header tests pin this).
- **Member order is correctness.** Float addition is not associative;
  `discover_members` sorts by name and nothing downstream re-orders.
- **Transcription and tokenization happen once per WAV, K times never.**
  Transcription owns the linear runtime term (0.036x realtime).
- **No silent coercion.** An unexpected embedding shape raises; it is never
  reshaped into a plausible one. `model/CLAUDE.md`: loud failure over silent
  fallback, and no error handling for impossible scenarios.
- **The container's inverted failure policy stays in the container.**
- **Do not change `predict.sh`, and change the `Dockerfile` only to correct the
  now-stale comment** naming the model layout (`adapter/ + ridge_head.npz`).
  A comment is not a build change; `COPY model/ /app/model/` is wholesale and
  needs no edit (verified by reading both files).
- Do not delete `model/data/results/phase1_lora/fold_embeddings_7665` — 5 GPU-hours.
- **HF Jobs is PRE-PAID and must be serialised.** Budget ONE five-member runtime
  probe run, not five. A staggered fleet dying together is balance exhaustion,
  not a code bug.

## Round-1 blockers and how this revision resolves them

| # | Round-1 blocker | Resolution in revision 2 |
|---|---|---|
| 1 | Task 12 unexecutable: `runtime_probe.py` hardcodes one adapter (`assemble_model_dir`, line 125-142) and `push_runtime_probe.py` stages one head, but the `4.93s + 0.0360x` baseline is a **cuda/HF-Jobs** number with no local GPU equivalent | Both files added to File Changes; new Tasks 20-22 teach them the `members/` layout **before** the probe run |
| 2 | Task 4 silently breaks `test_load_scorer_binds_the_device_onto_the_transcriber` (`test_score_wav.py:341`) — it passes an empty `tmp_path`, and `discover_members` now raises first | Task 12 is dedicated to repairing that test, and runs **before** the `load_scorer` rewire lands |
| 3 | No fast determinism test on `build_ensemble_embedder`, the newly-shipped path | New Task 11 mirrors the existing `build_fold_embedder` determinism test |
| 4 | The `vectors.ndim == 1` reshape is a silent fallback | Deleted from the plan; `EnsembleHead.predict`'s shape check is the only handler |
| 5 | Tasks 2/3/5/6/7 wrote 2-4 tests each before any implementation | Split into one behaviour per task: 22 tasks, each a single test |

Round-1 RISKs also folded in: symmetric legacy/`members` clearing (Task 17);
peft adapter names carry the member basenames, not `m0..m4` (Task 9); the
unreachable empty-input guards are **not written** rather than written-and-untested.

## File Changes

| File | Change | Type |
|---|---|---|
| `model/src/claim_measurement/difficulty/conftest.py` | `make_model_dir` fixture | **Create** |
| `model/src/claim_measurement/difficulty/test_conftest_fixtures.py` | fixture's own tests | **Create** |
| `model/src/claim_measurement/difficulty/score_wav.py` | `EnsembleHead`, `MemberPaths`, `discover_members`; `load_scorer` uses them | Modify |
| `model/src/claim_measurement/difficulty/test_score_wav.py` | the above, both layouts, + repair one existing test | Modify |
| `model/src/claim_measurement/difficulty/audio_emb_extract.py` | `build_ensemble_embedder` | Modify |
| `model/src/claim_measurement/difficulty/test_audio_emb_extract.py` | one-base-load, shared-tokenize, real adapter switch, determinism | Modify |
| `model/src/claim_measurement/difficulty/build_model_dir.py` | `--from-ensemble`, `build_ensemble_model_dir`, symmetric clearing | Modify |
| `model/src/claim_measurement/difficulty/test_build_model_dir.py` | ensemble build tests | Modify |
| `model/src/claim_measurement/difficulty/runtime_probe.py` | `assemble_model_dir` learns the `members/` layout | **Modify (added in rev 2)** |
| `model/src/claim_measurement/difficulty/test_runtime_probe.py` | multi-member assembly test | **Modify (added in rev 2)** |
| `model/src/claim_measurement/difficulty/push_runtime_probe.py` | stage K heads into the probe bundle | **Modify (added in rev 2)** |
| `model/src/claim_measurement/difficulty/test_score_wav_integration.py` | real-backbone ensemble mean (opt-in) | Modify |
| `apps/inference/mirex-difficulty/build_context.py` | validate either layout | Modify |
| `apps/inference/mirex-difficulty/test_build_context.py` | layout validation | **Create** |
| `apps/inference/mirex-difficulty/test_container.py` | five-member scoring (opt-in) | Modify |
| `apps/inference/mirex-difficulty/Dockerfile` | one stale comment | Modify (comment only) |
| `docs/mirex/track-a-difficulty-prediction.md` | the re-measured runtime | Modify |

Two files are new that the spec's table did not list, both flagged in round 1
and retained: `conftest.py` (the shared fixture seam — and, unlike a package
module, it can never ship, because `build_context.CODE_FILES` is an explicit
list) and `test_build_context.py` (`build_context.py` has no test file today and
Task 18 changes its validation).

## One decision the spec left open

**A model directory holding BOTH layouts is refused**, not silently resolved.
Preferring one would ship a score from a directory nobody chose — the same
artifact-mixing failure the `fallback_score` check exists to catch. Both
builders therefore clear the other layout when they write (Task 17), in both
directions.

## Task Groups

```
Group 0 (solo, first):          Tasks 1-2      [conftest.py]
Group A (sequential-internal):  Tasks 3-8      [score_wav.py]           parallel with B
Group B (sequential-internal):  Tasks 9-11     [audio_emb_extract.py]   parallel with A
Group C (solo):                 Tasks 12-13    [score_wav.py]           needs A + B
Group D (sequential-internal):  Tasks 14-17    [build_model_dir.py]     needs A
Group E (sequential-internal):  Tasks 18-19    [build_context.py, test_container.py] needs D
Group F (sequential-internal):  Tasks 20-22    [runtime_probe.py, push_runtime_probe.py] needs C
Group G (solo, last):           Tasks 23-24    [integration, decision log] needs all
```

- Group 0 first: every later group consumes its fixture.
- A and B share no file and no import; they run concurrently.
- **Group C is the join**: it repairs the existing device test and then rewires
  `load_scorer`, which is the only place A and B meet. Doing the repair first
  (Task 12) means the rewire never lands on a red suite.
- D needs `MEMBERS_SUBDIR` and `EnsembleHead` from A. E stages what D builds.
- F is the runtime-measurement path and needs a working `load_scorer` (C).
- G is opt-in verification against real artifacts, and is **human-lit**.

**Ships-independently checkpoint:** after Group C the ensemble scores from
Python and the whole value of the issue is exercisable. D-F are packaging and
measurement. Nothing before Group C is independently useful.

---

## Group 0 — The Harness

### Task 1: The fixture writes a readable two-member directory

**Behavior:** `make_model_dir({"s2026": head, "s2027": head})` produces
`members/<name>/adapter/adapter_config.json` + `ridge_head.npz` per member, and
the heads round-trip through `read_ridge_head`. A fixture that silently writes
the wrong layout makes every downstream assertion vacuous, so it is tested like
production code.

**Files:** Create `conftest.py`, create `test_conftest_fixtures.py`.

**The test:**
```python
"""#175: the fixture builder every ensemble test consumes, tested itself."""
import numpy as np

from claim_measurement.difficulty.score_wav import RidgeHead, read_ridge_head


def _head(dim=4, coef=1.0, fallback=5.0):
    return RidgeHead(mean=np.zeros(dim), scale=np.ones(dim),
                     coef=np.full(dim, coef), intercept=0.0,
                     fallback_score=fallback)


def test_the_fixture_writes_a_readable_two_member_directory(make_model_dir):
    model_dir = make_model_dir({"s2026": _head(coef=1.0),
                                "s2027": _head(coef=2.0)})

    for name, expected in (("s2026", 1.0), ("s2027", 2.0)):
        member = model_dir / "members" / name
        assert (member / "adapter" / "adapter_config.json").exists()
        np.testing.assert_array_equal(
            read_ridge_head(member / "ridge_head.npz").coef, np.full(4, expected))
```
Expected failure: `fixture 'make_model_dir' not found`.

**The implementation** (`conftest.py`) — note the docstring is load-bearing
context, not decoration:
```python
"""#175: shared fixtures for the ensemble model-directory tests.

Lives in conftest.py rather than in the package so it can be shared across test
files WITHOUT being importable production code -- build_context.CODE_FILES
enumerates what ships in the image, and test scaffolding must never be on it.

The adapters written here are placeholder JSON, not real LoRA weights: the fast
suite exercises layout discovery and head arithmetic, and the embedder is tested
separately with an injected loader (test_audio_emb_extract.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from claim_measurement.difficulty.score_wav import (
    ADAPTER_SUBDIR, HEAD_FILENAME, write_ridge_head)


@pytest.fixture
def make_model_dir(tmp_path):
    def build(members: dict, layout: str = "members", root=None) -> Path:
        model_dir = Path(root) if root is not None else tmp_path / "model_dir"
        if layout == "legacy":
            if len(members) != 1:
                raise ValueError(
                    f"the legacy layout holds exactly one adapter, got "
                    f"{len(members)}")
            (head,) = members.values()
            _write_adapter(model_dir / ADAPTER_SUBDIR)
            write_ridge_head(model_dir / HEAD_FILENAME, head)
        elif layout == "members":
            for name, head in members.items():
                member = model_dir / "members" / name
                _write_adapter(member / ADAPTER_SUBDIR)
                write_ridge_head(member / HEAD_FILENAME, head)
        else:
            raise ValueError(f"unknown layout {layout!r}")
        return model_dir

    return build


def _write_adapter(adapter_dir: Path) -> None:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_config.json").write_text(json.dumps(
        {"peft_type": "LORA", "r": 4, "note": "placeholder -- no weights"}))
```

**Commit:** `test(#175): add a shared fixture for synthetic ensemble model directories`

### Task 2: The fixture can also write the legacy layout

**Behavior:** `layout="legacy"` writes `adapter/` + `ridge_head.npz` at the root
and no `members/`. The merged container, `runtime_probe.py` and the staged Hub
bundles are all legacy-layout; the tests that prove it still loads need it built.

**The test:**
```python
def test_the_fixture_can_also_write_the_legacy_single_adapter_layout(make_model_dir):
    model_dir = make_model_dir({"only": _head()}, layout="legacy")

    assert (model_dir / "adapter" / "adapter_config.json").exists()
    assert (model_dir / "ridge_head.npz").exists()
    assert not (model_dir / "members").exists()
```
If this passes immediately off Task 1's implementation, say so in the commit and
keep it — it pins a property that is easy to regress.

**Commit:** `test(#175): pin the fixture's legacy-layout output`

---

## Group A — `score_wav.py` (parallel with Group B)

Tasks 3-6 build `EnsembleHead`; Tasks 7-8 build `discover_members`. Add the
shared `_head` helper (as in Task 1) beside the existing `_training_data` in
`test_score_wav.py` during Task 3, and extend that file's top-of-file import
from `score_wav` as each symbol appears.

### Task 3: The ensemble score is the mean of its members' own predictions

**Behavior:** the pre-registered submission model (#168) is the **prediction**
mean over five heads. If this ever drifts to, say, a coefficient average, we
would ship a model nobody measured — the two differ as soon as the members'
scalers differ.

**The test:**
```python
def test_the_ensemble_score_is_the_mean_of_its_members_own_predictions():
    heads = [_head(coef=1.0), _head(coef=2.0), _head(coef=3.0)]
    vectors = np.array([[1.0, 0.0, 0.0, 0.0]] * 3)

    ensemble = EnsembleHead.from_members(heads)

    individually = [h.predict(v) for h, v in zip(heads, vectors)]
    assert ensemble.predict(vectors) == pytest.approx(float(np.mean(individually)))
    assert ensemble.n_members == 3
```

**The implementation** — `EnsembleHead` in `score_wav.py`, immediately after
`RidgeHead` (it is the same section of the file and reads as its plural). Write
only what this test needs; Tasks 4-6 add the guards.
```python
@dataclass(frozen=True)
class EnsembleHead:
    """K RidgeHeads, scored as the mean of their predictions.

    K=1 is an EnsembleHead with one member, so `score_wav` has no
    single-vs-ensemble branch -- the only code that interprets layout is
    `discover_members`. Two scoring paths would mean the single-adapter path is
    the tested one and the ensemble path is the one that ships, which is how a
    submission-day divergence happens.
    """

    members: tuple

    @classmethod
    def from_members(cls, heads) -> "EnsembleHead":
        return cls(members=tuple(heads))

    @property
    def n_members(self) -> int:
        return len(self.members)

    @property
    def n_features(self) -> int:
        return self.members[0].n_features

    @property
    def fallback_score(self) -> float:
        return float(self.members[0].fallback_score)

    def predict(self, vectors) -> float:
        vectors = np.asarray(vectors, dtype=np.float64)
        # A list comprehension in member order, then one np.mean. Float addition
        # is not associative, so this order is part of #166's determinism
        # criterion -- do not reorder or vectorise into a different summation.
        return float(np.mean([h.predict(v)
                              for h, v in zip(self.members, vectors)]))
```

**Commit:** `feat(#175): add EnsembleHead, the pred-mean over K ridge heads`

### Task 4: One member is an ensemble of one, not a separate code path

**Behavior:** K=1 goes through the same object as K=5 and agrees with the bare
`RidgeHead` it wraps, including `fallback_score`. This is the property that lets
the legacy container path and the ensemble path share one tested implementation.

**The test:**
```python
def test_one_member_is_an_ensemble_of_one_not_a_separate_code_path():
    head = _head(coef=2.0)
    vector = np.array([[1.0, 1.0, 0.0, 0.0]])

    ensemble = EnsembleHead.from_members([head])

    assert ensemble.n_members == 1
    assert ensemble.predict(vector) == pytest.approx(head.predict(vector[0]))
    assert ensemble.fallback_score == head.fallback_score
```
Likely passes off Task 3. Keep it: it is the load-bearing claim of the whole
design, and a future refactor that special-cases K=1 must break something.

**Commit:** `test(#175): pin K=1 as an ensemble of one`

### Task 5: Members whose fallback disagrees are refused

**Behavior:** every member head is fit on the same 7,665 grades, so
`np.median(y)` is identical **by construction** (#168 verified identical seg_id
order across all five runs). A mismatch therefore means adapters from two
different training pools were staged into one directory — a silent
artifact-mixing bug that would otherwise ship as a plausible-looking average.
Fatal, not averageable, not a warning.

**The test:**
```python
def test_members_whose_fallback_disagrees_are_refused():
    with pytest.raises(ValueError, match="fallback"):
        EnsembleHead.from_members([_head(fallback=5.0), _head(fallback=6.0)])
```

**The implementation** — add to `from_members`:
```python
        heads = tuple(heads)
        fallbacks = {float(h.fallback_score) for h in heads}
        if len(fallbacks) != 1:
            raise ValueError(
                f"member heads disagree on fallback_score ({sorted(fallbacks)}) "
                f"-- they were fit on different training pools, so this "
                f"directory mixes artifacts and must not be scored")
        return cls(members=heads)
```
Do **not** add an empty-list guard: `discover_members` raises on a directory
with no members, so `from_members([])` is unreachable from any real caller, and
CLAUDE.md forbids error handling for impossible scenarios.

**Commit:** `feat(#175): refuse member heads fit on different training pools`

### Task 6: A vector matrix of the wrong shape is refused

**Behavior:** a `(K, D)` matrix whose K disagrees with the member count means
the embedder and the heads were built from different member lists. It raises;
it is never reshaped. A silent coercion here would emit a confident score from
one member while claiming five.

**The test:**
```python
def test_a_vector_matrix_of_the_wrong_shape_is_refused():
    ensemble = EnsembleHead.from_members([_head(), _head()])

    with pytest.raises(ValueError, match="2 members"):
        ensemble.predict(np.zeros((3, 4)))
```

**The implementation** — the first lines of `predict`:
```python
        if vectors.ndim != 2 or vectors.shape[0] != self.n_members:
            raise ValueError(
                f"expected a ({self.n_members}, {self.n_features}) matrix for "
                f"{self.n_members} members, got shape {vectors.shape} -- the "
                f"embedder and the heads were built from different member lists")
```

**Commit:** `feat(#175): refuse an embedding matrix that does not match the member count`

### Task 7: Members are discovered in sorted order

**Behavior:** `members/` yields one `MemberPaths` per subdirectory, **sorted by
name**. Float addition is not associative, so the mean depends on summation
order; sorting is what makes two independent container runs reproduce, which is
a stated exit criterion of #166.

**The test:**
```python
def test_members_are_discovered_in_sorted_order(make_model_dir):
    model_dir = make_model_dir({"s2030": _head(), "s2026": _head(),
                                "s2028": _head()})

    members = discover_members(model_dir)

    assert [m.name for m in members] == ["s2026", "s2028", "s2030"]
    assert members[0].adapter_dir == model_dir / "members" / "s2026" / "adapter"
    assert members[0].head_path == model_dir / "members" / "s2026" / "ridge_head.npz"
```

**The implementation** — `MemberPaths`, `MEMBERS_SUBDIR = "members"` beside the
existing `ADAPTER_SUBDIR`, and the `members/` arm of `discover_members`. Tasks 8
and its siblings add the legacy and failure arms.

**Commit:** `feat(#175): discover ensemble members in a stable sorted order`

### Task 8: The remaining `discover_members` arms

Three more single-test tasks, in this order — split for the same reason as
Tasks 3-6, and each drives a distinct branch:

**8a — a legacy directory is a one-member ensemble.** The merged container,
`runtime_probe.py` and the staged Hub bundles are all legacy; regressing that
path 26 days from the deadline to gain uniformity is a bad trade.
```python
def test_a_legacy_single_adapter_directory_is_a_one_member_ensemble(make_model_dir):
    model_dir = make_model_dir({"only": _head()}, layout="legacy")

    members = discover_members(model_dir)

    assert len(members) == 1
    assert members[0].adapter_dir == model_dir / "adapter"
    assert members[0].head_path == model_dir / "ridge_head.npz"
```
Commit: `feat(#175): load a legacy single-adapter directory as one member`

**8b — neither layout fails loudly.**
```python
def test_a_directory_with_neither_layout_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="no adapter"):
        discover_members(tmp_path)
```
Commit: `feat(#175): fail loudly on a model directory with no adapter`

**8c — both layouts present is refused.** Silently preferring one would score
through a model nobody chose.
```python
def test_a_directory_holding_both_layouts_is_refused(make_model_dir, tmp_path):
    model_dir = make_model_dir({"s2026": _head()}, root=tmp_path / "mixed")
    make_model_dir({"stale": _head()}, layout="legacy", root=tmp_path / "mixed")

    with pytest.raises(ValueError, match="both"):
        discover_members(model_dir)
```
Commit: `feat(#175): refuse a model directory holding both layouts`

The finished function:
```python
def discover_members(model_dir: Path) -> list:
    """The ONE function that knows there are two on-disk layouts.

        <model-dir>/members/<name>/adapter + ridge_head.npz   (K members)
        <model-dir>/adapter + ridge_head.npz                  (legacy, K=1)

    Sorted by name, because float addition is not associative and the mean over
    members therefore depends on summation order -- two container runs must
    reproduce bit-identically (#166).
    """
    model_dir = Path(model_dir)
    members_root = model_dir / MEMBERS_SUBDIR
    has_members = members_root.is_dir() and any(members_root.iterdir())
    has_legacy = (model_dir / ADAPTER_SUBDIR).is_dir()

    if has_members and has_legacy:
        raise ValueError(
            f"{model_dir} holds both {MEMBERS_SUBDIR}/ and a legacy "
            f"{ADAPTER_SUBDIR}/ -- preferring one silently would score through "
            f"a model nobody chose. Remove the stale one.")
    if has_members:
        return [MemberPaths(name=d.name, adapter_dir=d / ADAPTER_SUBDIR,
                            head_path=d / HEAD_FILENAME)
                for d in sorted(members_root.iterdir(), key=lambda p: p.name)
                if d.is_dir()]
    if has_legacy:
        return [MemberPaths(name=ADAPTER_SUBDIR,
                            adapter_dir=model_dir / ADAPTER_SUBDIR,
                            head_path=model_dir / HEAD_FILENAME)]
    raise FileNotFoundError(
        f"{model_dir} holds no adapter: expected {MEMBERS_SUBDIR}/<name>/ "
        f"members or a legacy {ADAPTER_SUBDIR}/")
```

---

## Group B — `audio_emb_extract.py` (parallel with Group A)

Promote the inline `fake_loader_factory` at `test_audio_emb_extract.py:171` to a
module-level `_fake_loader_factory` in Task 9 so every task here uses one.

### Task 9: The base backbone is loaded once for all members

**Behavior:** K adapters produce a `(K, D)` matrix from **one** `loader_factory`
call. `build_fold_embedder` loads the full 839M base on every call
(`audio_emb_extract.py:110-153`, verified), so K calls would load 1.6GB K times
— a memory problem and a startup-cost problem the 24h/1-GPU budget cannot absorb.
Five 3.2M-parameter LoRA deltas share one resident base.

**This task also resolves the spec's open question** — whether `peft==0.11.1`
supports `load_adapter`/`set_adapter` — for the *API*. Whether it works on the
vendored fork's model class is settled by Task 23.
**Fallback ladder, only after measuring, never as a first choice:**
`load_adapter`/`set_adapter` → `add_adapter` against a shared base → K base
loads with the memory cost recorded in the decision log. If rung 1 raises,
record the exact traceback in the commit body before moving down.

**The test:**
```python
def test_the_base_backbone_is_loaded_once_for_all_members(tmp_path):
    dirs = [tmp_path / "s2026" / "adapter", tmp_path / "s2027" / "adapter"]
    for d in dirs:
        _save_fake_adapter(d)
    midi_path = tmp_path / "piece.mid"
    write_notes_midi(_NOTES, midi_path)
    loads = []

    def counting_loader(checkpoint_path, repo_root, model_config):
        loads.append(checkpoint_path)
        return (_FakeOuter(hidden=4, n_layers=1, vocab=16),
                lambda path: torch.arange(6) % 16, 4)

    embed = build_ensemble_embedder(dirs, checkpoint=None, repo_root=None,
                                    model_config=None,
                                    loader_factory=counting_loader)
    vectors = embed(midi_path)

    assert len(loads) == 1, f"the 839M base must load once, loaded {len(loads)}x"
    assert vectors.shape == (2, 4)
```

**The implementation** — after `build_fold_embedder`. The loader → stub →
`import peft` order is a contract asserted on the source
(`test_peft_is_imported_only_after_the_loader_puts_the_fork_on_sys_path`);
preserve it exactly, and extend that test to cover the new function if the
assertion's anchor lines move.
```python
def build_ensemble_embedder(adapter_dirs, checkpoint: Path, repo_root: Path,
                            model_config: Path, loader_factory=_real_loader,
                            device=None, names=None):
    """A `embed(midi_path) -> (K, D)` closure over ONE resident base backbone
    with K LoRA adapters attached by name.

    The base is 839M parameters (1.6GB); the adapters are 3.2M each. Calling
    build_fold_embedder K times would load the base K times, which is both a
    memory problem and a startup-cost problem the 24h/1-GPU budget cannot
    absorb.

    `names` carries the member basenames from discover_members so a peft error
    names `alldata_s2029` rather than `m3`. Adapters are attached in the order
    given -- discover_members sorts, and that order is part of the determinism
    contract, so this function must not re-order.
    """
    adapter_dirs = [Path(d) for d in adapter_dirs]
    names = list(names) if names is not None else [d.parent.name or f"m{i}"
                                                   for i, d in enumerate(adapter_dirs)]

    base_model, tokenize, max_len = loader_factory(
        checkpoint, repo_root=repo_root, model_config=model_config)
    _stub_absent_transformers_models()
    from peft import PeftModel

    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dirs[0]),
                                           adapter_name=names[0])
    for name, adapter_dir in zip(names[1:], adapter_dirs[1:]):
        peft_model.load_adapter(str(adapter_dir), adapter_name=name)
    # eval() is load-bearing: lora_dropout=0.05 is active in train mode and
    # torch.no_grad() does NOT disable dropout, so without it every extraction
    # is a fresh random draw.
    peft_model.eval()
    if device is not None:
        peft_model.to(device)
    transformer = peft_model.model.model  # inner LoRA-injected transformer

    def embed(midi_path: Path) -> np.ndarray:
        # Tokenised ONCE and reused across members: _extract_full_piece takes
        # tokens, not a path, and does not mutate them (verified,
        # train_fold.py:272-280), so the same tensor is safe to share.
        tokens = tokenize(Path(midi_path))
        rows = []
        for name in names:
            peft_model.set_adapter(name)
            rows.append(_extract_full_piece(transformer, tokens, max_len,
                                            device=device))
        return np.stack(rows)

    return embed
```

**Commit:** `feat(#175): attach K LoRA adapters to one resident MoonBeam base`

### Task 10: Switching adapters actually changes the embedding

**Behavior:** if `set_adapter` silently no-ops, every row is the same member's
embedding and the "ensemble" is one adapter wearing five names — a
plausible-looking score from a model we never pre-registered, invisible in the
output. Two adapters saved from independently-initialised LoRA layers must
disagree.

**The test:**
```python
def test_switching_adapters_actually_changes_the_embedding(tmp_path):
    dirs = [tmp_path / "a" / "adapter", tmp_path / "b" / "adapter"]
    for seed, d in enumerate(dirs):
        torch.manual_seed(seed)
        _save_fake_adapter(d, init_lora_weights=False)  # non-zero B matrices

    midi_path = tmp_path / "piece.mid"
    write_notes_midi(_NOTES, midi_path)
    embed = build_ensemble_embedder(dirs, checkpoint=None, repo_root=None,
                                    model_config=None,
                                    loader_factory=_fake_loader_factory)

    vectors = embed(midi_path)

    assert not np.allclose(vectors[0], vectors[1]), (
        "both members produced the same embedding -- set_adapter did not switch")
```
`_save_fake_adapter` gains an `init_lora_weights` pass-through: **PEFT
initialises LoRA's B matrix to zeros by default**, which makes every adapter a
mathematical identity and would render this assertion unable to fail. That
default is precisely why the test needs `init_lora_weights=False`.

**Commit:** `test(#175): pin that switching adapters changes the embedding`

### Task 11: Repeated extraction is byte-identical (NEW in rev 2)

**Behavior:** three `embed(midi)` calls return bit-identical `(K, D)` matrices.
Determinism is a stated exit criterion of #166, and
`test_build_fold_embedder_extraction_is_byte_identical_across_repeated_calls`
(`test_audio_emb_extract.py:159`) guards `build_fold_embedder`, which after
Group C is no longer what the container calls. `set_adapter` is invoked *after*
`eval()` on every extraction; if it ever perturbs module training state,
`lora_dropout=0.05` makes every score a fresh random draw — and without this
test the only remaining guard is a 21GB opt-in container run.

**The test:**
```python
def test_ensemble_extraction_is_byte_identical_across_repeated_calls(tmp_path):
    dirs = [tmp_path / "a" / "adapter", tmp_path / "b" / "adapter"]
    for d in dirs:
        _save_fake_adapter(d, init_lora_weights=False)
    midi_path = tmp_path / "piece.mid"
    write_notes_midi(_NOTES, midi_path)
    embed = build_ensemble_embedder(dirs, checkpoint=None, repo_root=None,
                                    model_config=None,
                                    loader_factory=_fake_loader_factory)

    first, second, third = embed(midi_path), embed(midi_path), embed(midi_path)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first, third)
```
If it fails, the cause is almost certainly `set_adapter` flipping training mode:
call `peft_model.eval()` inside the member loop rather than once, and say so in
the commit body.

**Commit:** `test(#175): pin ensemble extraction determinism, the shipped path`

---

## Group C — The join (`score_wav.py`, needs A and B)

### Task 12: Repair the existing device-binding test (NEW in rev 2 — do this FIRST)

**Behavior:** `--device` still reaches Transkun after `load_scorer` is rewired.
`test_load_scorer_binds_the_device_onto_the_transcriber`
(`test_score_wav.py:341-372`) monkeypatches `sw.read_ridge_head` and
`sw.build_fold_embedder` and calls `load_scorer(tmp_path, ...)` on an **empty**
`tmp_path`. After Task 13, `discover_members(tmp_path)` raises
`FileNotFoundError` before either patch is consulted, and the test dies — taking
with it the guard on the difference between the 24h clause fitting and not
(#166). Repair it **before** the rewire, so the rewire never lands on a red suite.

**The change, and why it is split across two commits.** Task 12 does ONLY the
half that is green both before and after the rewire: build a legacy directory
with `make_model_dir` instead of passing a bare `tmp_path`, and drop the
`read_ridge_head` patch (the fixture writes a real head). It keeps patching
`sw.build_fold_embedder`, which is still what `load_scorer` calls at this point.

The patch target must NOT be flipped here. `monkeypatch.setattr` raises
`AttributeError` on an attribute that does not exist, so patching
`sw.build_ensemble_embedder` before Task 13 imports it would leave the suite RED
between two commits. **Task 13 flips the patch target to
`sw.build_ensemble_embedder` (returning `np.zeros((1, 4))`) in the same commit
as the rewire** — the test change and the code change are one atomic unit.

The assertion `seen["device"] == "cuda"` is unchanged in both commits: that is
the behaviour under test and it must not weaken.

Also confirm `test_load_scorer_rejects_an_unknown_failure_mode`
(`test_score_wav.py:330`) still passes: it also passes a bare `tmp_path`, and it
survives **only because the `on_failure` validation is the first statement in
`load_scorer`**. Keep it first.

**Commit:** `test(#175): repoint the device-binding test at a real legacy model dir`

### Task 13: `score_wav` and `load_scorer` score through K members

**Behavior:** one WAV produces one transcription and one `embed` call returning
a `(K, D)` matrix that reaches `EnsembleHead.predict` unflattened.

**The test:**
```python
def test_a_wav_is_transcribed_once_no_matter_how_many_members(tmp_path):
    """Transcription owns the linear runtime term (0.036x realtime).
    Transcribing per member would be a ~5x regression on the dominant cost and
    put the 24h/1-GPU exclusion clause in play."""
    wav = tmp_path / "piece.wav"
    wav.write_bytes(b"")
    calls, embed_calls = [], []

    def embed(midi_path):
        embed_calls.append(midi_path)
        return np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

    head = EnsembleHead.from_members([_head(coef=1.0), _head(coef=3.0)])

    score = score_wav(wav, _fake_transcribe(calls), embed, head)

    assert len(calls) == 1, "one WAV, one transcription"
    assert len(embed_calls) == 1, "one WAV, one embed call for all members"
    assert score == pytest.approx((1.0 + 3.0) / 2)
```

**The implementation.** In `score_wav`, rename `vector` to `vectors`, keep the
finite check over the whole matrix, and **add no shape coercion** — an embedder
returning the wrong rank raises in `EnsembleHead.predict`:
```python
    vectors = np.asarray(vectors, dtype=np.float64)
    if not np.all(np.isfinite(vectors)):
        raise ValueError(
            f"{wav_path} produced a non-finite embedding "
            f"({int((~np.isfinite(vectors)).sum())} of {vectors.size} entries)")
    score = head.predict(vectors)
```
In `load_scorer`:
```python
    members = discover_members(model_dir)
    head = EnsembleHead.from_members(
        [read_ridge_head(m.head_path) for m in members])
    embed = build_ensemble_embedder(
        [m.adapter_dir for m in members], checkpoint, repo_root, model_config,
        device=device, names=[m.name for m in members])
```
Update the `load_scorer` docstring (the base loads once and all K adapters
attach to it), the `head:` annotations on `score_wav` /
`score_wav_or_fallback`, and the module docstring's **Model directory layout**
block to show both layouts.

**In this same commit**, flip Task 12's repaired test from patching
`sw.build_fold_embedder` to patching `sw.build_ensemble_embedder`. This is the
one place a test edit and an implementation edit must ship together: splitting
them leaves the suite red in between, because `monkeypatch.setattr` raises on an
attribute that does not exist yet.

**Commit:** `feat(#175): score a WAV through every ensemble member from one transcription`

Then one more single-test task:

**13b — a non-finite entry in any member row is a failure.** A non-finite
embedding means the backbone produced garbage for that member; averaging it away
would emit a confident score from a broken forward pass.
```python
def test_a_non_finite_entry_in_any_member_row_is_a_failure(tmp_path):
    wav = tmp_path / "piece.wav"
    wav.write_bytes(b"")
    bad = np.array([[1.0, 0.0, 0.0, 0.0], [np.nan, 0.0, 0.0, 0.0]])
    head = EnsembleHead.from_members([_head(), _head()])

    with pytest.raises(ValueError, match="non-finite"):
        score_wav(wav, _fake_transcribe(), lambda p: bad, head)
```
Commit: `test(#175): fail on a non-finite entry in any member row`

**13c — a legacy directory still loads end to end.** The regression guard on not
breaking a working submission path: `discover_members` returns one member, and
`build_ensemble_embedder` is called with exactly `[model_dir / "adapter"]`.
Commit: `test(#175): pin that load_scorer still reads a legacy model directory`

After Group C, re-run the full suite and confirm
`test_scoring_the_same_wav_twice_gives_a_bit_identical_score` still passes.

---

## Group D — `build_model_dir.py` (needs Group A)

### Task 14: Each member is staged under its source directory basename

**Behavior:** `.../alldata_s2026` becomes `members/alldata_s2026/`, so
provenance is readable off the layout and the sort order `EnsembleHead` depends
on is stable **and** meaningful rather than an accident of copy order. Each
head is fit from that member's own `emb_fold99.npz` with **no exclusions**, via
the existing `fit_head_from_fold_embeddings(emb, exclude_seg_ids=None)` — never
re-deriving folds. `_build_all_data`'s no-tau-c rationale applies unchanged.

**The test:** build two synthetic artifact dirs with `_fold_npz`, call
`build_ensemble_model_dir`, assert both member directories exist with an adapter
and a head. **Commit:** `feat(#175): stage K all-data artifacts under their source basenames`

### Task 15: Duplicate basenames are refused
Silently overwriting would ship a four-member ensemble labelled five.
**Commit:** `feat(#175): refuse two ensemble members with the same basename`

### Task 16: The staged members load as one EnsembleHead
The end of the vertical slice: what the builder writes is exactly what
`load_scorer` reads, and the `fallback_score` agreement holds for heads fit from
one grade distribution.
```python
    members = discover_members(out_dir)
    head = EnsembleHead.from_members([read_ridge_head(m.head_path) for m in members])
    assert head.n_members == 2
```
**Commit:** `test(#175): pin that a staged ensemble directory loads as one head`

### Task 17: Clearing the other layout is symmetric (RISK from round 1)

**Behavior:** writing `members/` removes a stale legacy `adapter/` +
`ridge_head.npz`, **and** writing the legacy layout removes a stale `members/`.
`discover_members` refuses a directory holding both, so an asymmetric builder is
a trap for whoever rebuilds a single-adapter directory during submission week.
Two tests, one per direction — so two tasks (17a, 17b) by the one-test rule.
**Commits:** `fix(#175): clear a stale legacy adapter when writing members/` and
`fix(#175): clear a stale members/ when writing the legacy layout`

### Task 17c: `--from-ensemble` on the CLI, with a manifest that discloses K

**Behavior:** `--from-ensemble <dir>` (repeatable, `action="append"`) builds the
directory; the manifest records member names, per-member head rows, `n_members`,
and carries the same no-tau-c warning `--from-all-data` does; and it is mutually
exclusive with the other two sources. **Why the manifest matters:** MIREX 2026
makes training-data size, model size and compute a *mandatory disclosure*, and a
five-member ensemble is five times the inference compute. Test through
`main(argv=[...])`, matching the existing CLI tests. Update the module
docstring's layout block.
**Commit:** `feat(#175): add --from-ensemble to build_model_dir`

---

## Group E — Container staging (needs Group D)

### Task 18: `build_context.py` accepts either layout

**Behavior:** staging a `members/`-layout directory succeeds, a legacy one still
succeeds, and one with neither fails in seconds — before 1.6GB is copied — with
a message naming both layouts.

**Files:** modify `build_context.py`, create `test_build_context.py`, and fix the
`Dockerfile`'s now-stale `adapter/ + ridge_head.npz + manifest.json` comment.

`build_context`'s other inputs are all `path.exists()` checks, so the test builds
a minimal fake tree for them under `tmp_path`. Replace:
```python
    _require(model_dir / "ridge_head.npz", "the ridge head")
    _require(model_dir / "adapter" / "adapter_config.json", "the LoRA adapter")
```
with a check accepting either layout. **Duplicating the rule rather than
importing `discover_members` is deliberate:** `build_context.py` runs OUTSIDE
the isolated script env, and importing `score_wav` would drag in peft and torch.
The new test is what keeps the duplicate honest.
Run it explicitly:
`uv run python -m pytest apps/inference/mirex-difficulty/test_build_context.py -q --no-cov`
**Commits:** one per direction (accepts `members/`; rejects neither-layout).

### Task 19: The image scores through five members (opt-in, human-lit)

**Behavior:** a five-member image prints exactly one clean line per WAV (stdout
must stay clean — the fork's `MusicTokenizer` prints ~96KB of vocabulary on
construction), two runs of the same WAV agree bit-identically, and the baked
manifest reports five members.

**Deliberately not verified here:** that the container's score equals the mean of
five individually-run scores. That needs five images or five mounted model dirs
at ~21GB and minutes per run. Task 23 proves the same property at the Python
level with K=2 against the real backbone. **Sequence Task 19 after Task 23** so a
real-backbone failure is found without a 21GB build. If time runs short before
2026-10-01, this is the first task to defer.
**Commit:** `test(#175): assert the image scores through five members deterministically`

---

## Group F — The runtime path (NEW in rev 2, needs Group C)

Round-1 blocker 1. Without these three tasks, Task 24 cannot run: the
`4.93s + 0.0360x` baseline is a **cuda/HF-Jobs** number, there is no local GPU,
and Docker-on-macOS inflates perf ~2.5x — so no local measurement is comparable.

### Task 20: `assemble_model_dir` builds the `members/` layout

**Behavior:** given K adapter dirs and K head dirs, `runtime_probe.assemble_model_dir`
writes `members/<name>/adapter` + `ridge_head.npz`. Today it hardcodes one
(`runtime_probe.py:125-142`):
```python
shutil.copytree(adapter_dir / "adapter", dest / "adapter")
for name in ("ridge_head.npz", "manifest.json"):
    shutil.copy2(head_dir / name, dest / name)
```
Keep the single-adapter call shape working — the K=1 probe is the baseline
this measurement is compared against, and it must stay reproducible.
Test in `test_runtime_probe.py` with `tmp_path`, no GPU, no network.
**Commit:** `feat(#175): assemble a multi-member probe model directory`

### Task 21: `runtime_probe.main` resolves K adapter repos

**Behavior:** `--adapter-dir` becomes repeatable and the Hub path snapshots the
five `Jai-D/phase1-lora-7665-alldata-s{2026..2030}` repos. Preserve the offline
trio (`--adapter-dir`, `--repo-root`, `--checkpoint`): three #149 launches died
on environment plumbing rather than on what they measured, and the whole path
must stay rehearsable on CPU against local files first.
**Commit:** `feat(#175): let the runtime probe resolve K adapters`

### Task 22: `push_runtime_probe.py` stages K heads

**Behavior:** the probe bundle's `head/` (one `ridge_head.npz` + `manifest.json`
today, `push_runtime_probe.py:113`) carries one head per member.
**Gotcha to respect:** `snapshot_download` materialises files as symlinks into
`<repo>/blobs/<sha256>`, which is why `build_transcriber` exists at all — do not
introduce a new `.resolve()`-based path walk into the staged bundle.
**Commit:** `feat(#175): stage K member heads into the runtime probe bundle`

---

## Group G — Real-artifact verification (human-lit, last)

### Task 23: Opt-in integration — the real backbone, two real adapters

**Behavior, against the real fork model class:** `load_adapter`/`set_adapter`
work on the vendored MoonBeam fork's `LlamaForCausalLM` (the spec's open
question, which Task 9 can only answer for `_FakeOuter`), the base loads once,
and a two-member ensemble's score equals the mean of the two single-member
directories' scores on the same WAV.

Run from the primary checkout, or from the worktree with `CRESCENDAI_DATA_ROOT`
pointed at it — `model/data/` is empty here.
**Watch for:** K identical rows, which means `set_adapter` did not switch on the
real class. Fallback ladder from Task 9, in order, measured.
**Commit:** `test(#175): verify the ensemble mean end-to-end on the real backbone`

### Task 24: Re-measure runtime and record it

The spec *predicts* `~5.7s + 0.0360x` from `4.93s + 0.0360x` (+0.19s/member on
the fixed term only). That is a prediction to be measured, not an assumption to
ship on; the 24h/1-GPU clause is an exclusion criterion.

- [ ] Run the probe on cuda against the five-member directory. **One serialised
      HF Jobs run** — it is pre-paid, and a staggered fleet dying together is
      balance exhaustion, not a code bug.
- [ ] Record the measured fixed and linear terms in
      `docs/mirex/track-a-difficulty-prediction.md` as a dated entry beside the
      single-adapter `4.93s + 0.0360x`, with the implied items-per-24h at the
      103s corpus median.
- [ ] **If the linear term moved at all, stop.** Something is transcribing or
      tokenizing per member, and the fix is in Group B, not in the log.
**Commit:** `docs(#175): record the measured five-member container runtime`

---

## Definition of done

- `cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov`
  is green, materially above the 246-passing baseline.
- `discover_members` is the only function in the tree that interprets layout.
- A five-member image prints one deterministic float per WAV, and a legacy
  single-adapter directory still scores unchanged.
- The measured runtime is in the decision log, with the linear term unmoved.
- `STATE:` posted on #175. No tau-c reported anywhere.

---

## Challenge Review — Round 2 (revision 2)

Re-reviewed against revision 2. Round-1 blockers 1-5 are verified resolved:

| # | Round-1 blocker | Status |
|---|---|---|
| 1 | `runtime_probe.py` / `push_runtime_probe.py` missing, Task 12 unexecutable | **RESOLVED** — both in the File Changes table; Group F (Tasks 20-22) precedes the probe run; the pre-paid/serialised HF Jobs constraint is stated |
| 2 | Task 4 breaks the device-binding test | **RESOLVED** — dedicated Task 12, sequenced before the rewire, with the `on_failure`-validation-stays-first note preserved |
| 3 | No fast determinism test on the shipped path | **RESOLVED** — Task 11, mirroring the existing `build_fold_embedder` test, with the likely cause named if it fails |
| 4 | `ndim == 1` silent reshape | **RESOLVED** — deleted; "No silent coercion" is now a hard constraint |
| 5 | Five horizontally-sliced tasks | **RESOLVED** — 25 single-test tasks; the step cycle is stated once and binds every task |

Round-1 RISKs folded in and verified present: symmetric layout clearing (Task
17a/17b), peft adapter names carry member basenames (Task 9's `names=`
parameter), Dockerfile comment-only edit explicitly permitted, unreachable
empty-input guards not written rather than written-and-untested.

### New findings in revision 2

`[BLOCKER — FIXED IN THIS PASS]` **(confidence: 9/10) — Task 12 as first written
would have left the suite RED between two commits.** It instructed the repair to
patch `sw.build_ensemble_embedder`, which does not exist until Task 13.
`monkeypatch.setattr` raises `AttributeError` on a missing attribute, so the
"repair" would fail from the moment it landed until the rewire. This is the same
class of error as round-1 blocker 2, reintroduced by its own fix.
**Applied:** Task 12 now does only the fixture half (green before *and* after)
and keeps patching `build_fold_embedder`; Task 13 flips the patch target **in
the same commit as the rewire**, which is the one place a test edit and an
implementation edit must ship atomically. Both task bodies state why.

`[RISK]` **(confidence: 7/10) — the lettered sub-tasks (8a-8c, 13b-13c, 17a-17c)
reintroduce blocker 5's ambiguity in the numbering, not the content.** Each
lettered item has its own single test and its own commit message, so the content
is correct, but a `/build` subagent handed "Task 8" could reasonably take all
three. **Mitigation for the build agent:** treat every lettered item as its own
task. The file has 25 `### Task` headings and should produce ~25 commits; if a
subagent proposes fewer, it has bundled.

`[RISK]` **(confidence: 6/10) — Group D (Tasks 14-17) specifies behaviour and
commit messages but not test code**, where Groups A-C give it verbatim. The
existing `test_build_model_dir.py` conventions (`_fold_npz`, `main(argv=[...])`)
are close enough to follow, and the behaviours are unambiguous, but expect more
variance in what the subagents write there than elsewhere. Acceptable: these are
the least subtle tasks in the plan.

`[OBS]` — The default `names=[d.parent.name or f"m{i}" ...]` in
`build_ensemble_embedder` yields the model-dir name for a legacy adapter
(`.../model_dir/adapter` → `"model_dir"`). Harmless, and `load_scorer` always
passes `names` explicitly. Avoid only the literal `"default"`, which peft
reserves.

`[OBS]` — Verified again and unchanged: the `Dockerfile` needs no build change
(`COPY model/ /app/model/` is wholesale, `MIREX_MODEL_DIR=/app/model`),
`predict.sh` passes `--model-dir` through without enumerating `adapter/`, and
sharing one `tokens` tensor across members is safe (`_extract_full_piece` does a
non-mutating `.to(device)` then slices, `train_fold.py:272-280`).

### Summary

```
[BLOCKER]  count: 1 (fixed in this pass; no open blockers)
[RISK]     count: 2
[QUESTION] count: 0
```

VERDICT: PROCEED_WITH_CAUTION — two risks to monitor during /build:
(1) the lettered sub-tasks must each be dispatched as their own task — expect
~25 commits, and treat a smaller number as bundling;
(2) Group D's tasks carry behaviour statements rather than verbatim tests, so
review those subagents' test code against `test_build_model_dir.py`'s existing
conventions more closely than the rest.

Two things remain unresolvable before execution and are correctly deferred to
tasks that measure rather than assume: whether `peft==0.11.1`'s named-adapter
API works on the vendored fork's model class (Task 9 for the API, Task 23 for
the fork), and the real five-member runtime (Task 24, one serialised pre-paid
HF Jobs run).

---

## Appendix: Round-1 Challenge Review (superseded)

Preserved because its verified code readings — the `runtime_probe.py` layout
hardcoding, the breaking device test, and the `tokens`-sharing safety proof —
are the evidence behind revision 2 and should not have to be re-derived.

## Challenge Review

Reviewed 2026-09-05 against the spec and the actual source of `score_wav.py`,
`audio_emb_extract.py`, `train_fold.py`, `build_model_dir.py`, `build_context.py`,
`runtime_probe.py`, `push_runtime_probe.py`, `predict.sh`, `Dockerfile`, and the
four existing test files.

### CEO Pass

**Premise — SOUND.** The problem is real and verified: `load_scorer`
(`score_wav.py:280-284`) reads exactly `model_dir/adapter` and one
`ridge_head.npz`, so the container ships one-fifth of the pre-registered model.
The ensemble is pre-registered on #168 and is not being re-litigated here. No
alternative framing is cheaper — the members must all run, and they must run
against one resident base.

`[OBS]` — Existing coverage is being extended, not reinvented:
`fit_head_from_fold_embeddings`, `build_model_dir`, `write_manifest`,
`_extract_full_piece` and the `loader_factory` injection pattern are all reused.
No new service, no new class beyond two dataclasses.

**Scope — one item of undeclared scope, one item that should be cut.**

`[BLOCKER]` **(confidence: 9/10) — Task 12 is not executable as written, and
`runtime_probe.py` is missing from the File Changes table.**
`runtime_probe.assemble_model_dir` (`runtime_probe.py:125-142`) hardcodes the
legacy layout:
```python
shutil.copytree(adapter_dir / "adapter", dest / "adapter")
for name in ("ridge_head.npz", "manifest.json"):
    shutil.copy2(head_dir / name, dest / name)
```
and `main` downloads exactly one adapter (`snapshot_download(ADAPTER_REPO,
allow_patterns=["adapter/**"])`) while `push_runtime_probe.py:113` stages exactly
one `head/ridge_head.npz`. A five-member probe needs five adapter repos
(`Jai-D/phase1-lora-7665-alldata-s{2026..2030}`) and five heads staged into the
bundle. This is not optional: the `4.93s + 0.0360x` baseline was measured on
**cuda via HF Jobs**, there is no local GPU, and `project_measurement_environment_traps`
records that Docker-on-macOS inflates perf ~2.5x — so a CPU or local-Docker number
is not comparable to the baseline it must be compared against.
**Fix:** add a Group E task pair — `runtime_probe.assemble_model_dir` learns the
`members/` layout (one test, tmp_path, no GPU), and `push_runtime_probe.py`
stages K heads — before Task 12. Note that HF Jobs is PRE-PAID and serialised:
budget one probe run, not five.

`[RISK]` **(confidence: 8/10) — Task 10 is expensive and proves little the fast
suite does not.** It needs a ~21GB image build plus minutes per scoring run, and
its three assertions (one clean line, determinism, manifest says five) are all
obtainable more cheaply — the first two from Task 11, the third from Task 8.
**Recommendation:** keep it, because the container is the only artifact MIREX
actually runs and #166's suite already exists to be extended, but sequence it
after Task 11 so a real-backbone failure is found without a 21GB build. If time
runs short before 2026-10-01, this is the first task to defer.

`[OBS]` — Minimum viable version, if half the time were available: Groups 0, A,
B, C (Tasks 1-8) plus Task 12. That ships the ensemble and proves the runtime.
Groups D and E are staging and confirmation. Nothing in the plan is speculative
scope.

**Twelve-Month Alignment**

```
CURRENT STATE                      THIS PLAN                        12-MONTH IDEAL
one adapter + one head,       →  K adapters on one resident    →  a submitted, ranked
legacy layout only,              base; layout resolved in one     MIREX entry, and a
scoring path hardcoded to K=1     function; K=1 a special case     research harness whose
                                  of K                             shipped path IS the
                                                                   measured path
```
Moves toward it. The one debt it creates — two on-disk layouts — is deliberate,
time-boxed by the 2026-10-01 deadline, and confined to `discover_members`.

`[QUESTION]` — The spec documents rejected alternatives for all three design
decisions (separate `score_wav_ensemble`; full migration to `members/`; unsorted
members). Alternatives coverage is adequate; no action.

### Engineering Pass

**Architecture**

```
predict.sh → score_wav.py main → load_scorer
                                   ├── discover_members(model_dir) ──┐ sorted by name
                                   ├── EnsembleHead.from_members ←──┤ K heads
                                   └── build_ensemble_embedder ←────┘ K adapter dirs
                                            │
                                            └── loader_factory ×1 → PeftModel + K adapters

score(wav) → transcribe ×1 → temp MIDI → embed ──→ tokenize ×1
                                                    └── per member: set_adapter → forward
                                                        → np.stack → (K,D)
                                         → finite check → EnsembleHead.predict → float
             error path → score_wav_or_fallback → SCORE_FAILURE on stderr → median
```
No user input reaches SQL, a shell, or an LLM prompt. `predict.sh` passes WAV
paths through `"${args[@]}"`, already quoted. No new fan-out: K is 5, fixed, and
the loop is over adapters, not over data. Deployment is a container rebuild with
no migration and no partial state; rollback is rebuilding from the legacy model
directory, which this plan explicitly keeps working.

`[BLOCKER]` **(confidence: 9/10) — Task 4 breaks an existing passing test and
the plan does not mention it.** `test_load_scorer_binds_the_device_onto_the_transcriber`
(`test_score_wav.py:341-372`) monkeypatches `sw.read_ridge_head` and
`sw.build_fold_embedder`, then calls `load_scorer(tmp_path, ...)` with an EMPTY
`tmp_path`. After Task 4, `load_scorer` calls `discover_members(tmp_path)` first,
which raises `FileNotFoundError` before either patch is consulted — the test
fails, and the `--device` binding it protects (the difference between the 24h
clause fitting and not, per #166) loses its guard.
**Fix:** Task 4 must update that test to build a legacy directory via
`make_model_dir` and patch `sw.build_ensemble_embedder`. State it as an explicit
step, and re-assert `seen["device"] == "cuda"` unchanged. Verify
`test_load_scorer_rejects_an_unknown_failure_mode` (line 330) still passes — it
does, because the `on_failure` validation is the first statement in `load_scorer`
and must STAY first.

`[BLOCKER]` **(confidence: 8/10) — the `vectors.ndim == 1` reshape in Task 4's
implementation is a silent fallback and must be deleted.**
```python
if vectors.ndim == 1:                      # a one-member embedder may hand
    vectors = vectors.reshape(1, -1)       # back a bare vector
```
No production embedder returns 1D: `build_ensemble_embedder` always `np.stack`s.
This branch exists only to accommodate hand-written test fakes, and in exchange
it silently coerces an unexpected shape into a plausible one — exactly the
silent fallback `model/CLAUDE.md` and the user's stated style forbid ("no error
handling for impossible scenarios"; "prefer explicit failures"). If a future
embedder regresses to returning one vector, this converts a loud shape error
into a confident wrong score from one member.
**Fix:** delete the branch; let `EnsembleHead.predict`'s existing shape check
raise. Update the Task 4 fakes to return `(K, D)`.

**Module Depth Audit**

| Module | Interface | Hidden impl | Verdict |
|---|---|---|---|
| `EnsembleHead` | 1 constructor + `predict` + 3 properties | ~45 LOC: pred-mean, fallback agreement, width agreement, shape check | **ADEQUATE, not deep.** Its value is *branch elimination* in the scoring path, not information hiding. That is a legitimate reason and the spec argues it; do not add methods to it to make it feel deeper. |
| `discover_members` + `MemberPaths` | 1 function → list of a 3-field record | ~30 LOC: two layouts, sort order, two distinct failures | **DEEP** |
| `build_ensemble_embedder` | 1 function → 1 closure | ~40 LOC: single base load, peft import ordering, named-adapter attach/select, shared tokenization, stacking, eval() | **DEEP** |
| `build_ensemble_model_dir` | 1 function | ~35 LOC: per-member head fitting, copying, basename dedup, legacy clearing, manifest | **DEEP** |

No shallow modules. Four new symbols across four files is proportionate.

**Code Quality**

`[RISK]` **(confidence: 8/10) — `build_model_dir`'s legacy path does not clear a
stale `members/`.** Task 7 makes the ensemble builder clear a legacy `adapter/`,
but not the reverse: running `--from-all-data` into a directory that already
holds `members/` produces exactly the both-layouts state `discover_members`
refuses. The asymmetry is a trap for whoever rebuilds a single-adapter directory
during submission week. **Fix:** make the clearing symmetric in Task 7, with a
test for each direction.

`[RISK]` **(confidence: 7/10) — Task 5 names the PEFT adapters `m0..m4`,
discarding the provenance the layout was designed to carry.** The spec's whole
argument for basename naming is that provenance is readable off the artifact; a
peft error then reads `adapter m3 not found` instead of naming
`alldata_s2029`. **Fix:** use the member names from `discover_members` as the
peft adapter names, passed in alongside the dirs. Avoid the literal `"default"`,
which peft reserves. Verify basenames are valid peft adapter names before
relying on this (they are plain `[A-Za-z0-9_]`, so this should be safe).

`[OBS]` — No catch-all handlers are introduced. `score_wav_or_fallback`'s
`except Exception` is pre-existing, documented as a deliberate inversion, and
confined to the container.

`[OBS]` — The `Dockerfile`'s model comment ("adapter/ + ridge_head.npz +
manifest.json") goes stale. The plan forbids Dockerfile changes to protect the
build; a comment-only edit is not a build change and should be explicitly
permitted in Task 9.

**Test Philosophy Audit**

`[OBS]` — Tests exercise public interfaces (`predict`, `discover_members`,
`build_ensemble_embedder`, `main(argv=...)`, `docker run`) against real temp
directories and real `peft` objects. No internal collaborator is mocked except
at genuine process boundaries: `loader_factory` (loads 1.6GB), `transcribe`
(shells out to another env), and `build_ensemble_embedder` in one `load_scorer`
test. The last is the weakest and is a pre-existing convention
(`test_score_wav.py:362`), not a new one — acceptable.

`[OBS]` — Every test asserts on behaviour, not shape. The nearest thing to a
shape test — Task 1's fixture test — is testing the harness itself, which is
appropriate: a fixture that writes the wrong layout makes every downstream
assertion vacuous.

**Vertical Slice Audit**

`[BLOCKER]` **(confidence: 10/10) — five tasks write multiple tests before any
implementation.** The plan's own header says "one task = one failing test = one
implementation = one commit" and then Tasks 2 (4 tests), 3 (4), 5 (2), 6 (2) and
7 (4) violate it. This is not pedantry: Task 2's
`test_a_vector_matrix_of_the_wrong_shape_is_refused` asserts `match="2 members"`
— an error string that does not exist yet. Tests written in bulk pin imagined
behaviour, and `/build` dispatches one subagent per task, so a bundled task
gives one agent four imagined contracts to satisfy at once.
**Fix — split into single-behaviour tasks (same code, more commits):**

| Was | Becomes |
|---|---|
| Task 2 | 2a pred-mean equals members' own predictions · 2b K=1 goes through the same object · 2c mismatched `fallback_score` refused · 2d wrong-shape matrix refused |
| Task 3 | 3a `members/` discovered in sorted order · 3b legacy dir is a one-member ensemble · 3c neither layout fails loudly · 3d both layouts refused |
| Task 5 | 5a base loads once for K members, returns (K,D) · 5b K=1 uses the same path |
| Task 6 | 6a one tokenization per WAV · 6b switching adapters changes the embedding |
| Task 7 | 7a members staged under source basenames · 7b duplicate basenames refused · 7c stale legacy adapter cleared (and its converse) · 7d staged members load as one EnsembleHead |

Task 4 already bundles three tests and splits the same way (4a transcribe-once,
4b non-finite entry, 4c legacy regression). Task numbering downstream shifts;
the group dependency map is unchanged.

**Test Coverage Gaps**

```
[+] score_wav.py
    ├── EnsembleHead.from_members
    │   ├── [TESTED] ★★★ mismatched fallback, mismatched width, empty
    │   └── [GAP]    empty member list -- ValueError branch has no test
    ├── EnsembleHead.predict
    │   ├── [TESTED] ★★★ pred-mean, K=1, wrong shape
    │   └── [GAP]    determinism across repeated calls (see BLOCKER below)
    ├── discover_members
    │   └── [TESTED] ★★★ all four branches
    └── score_wav (modified)
        ├── [TESTED] ★★  one transcription, one embed call
        ├── [TESTED] ★★★ non-finite entry in any member row
        └── [GAP]    existing device-binding test BREAKS (see BLOCKER above)

[+] audio_emb_extract.py
    └── build_ensemble_embedder
        ├── [TESTED] ★★  one base load, (K,D) shape
        ├── [TESTED] ★★★ one tokenization; adapters actually switch
        ├── [GAP]    determinism across repeated embed() calls
        └── [GAP]    empty adapter_dirs -- ValueError branch has no test

[+] build_model_dir.py / build_context.py
    └── [TESTED] ★★  per the plan; symmetry gap noted as RISK above
```

`[BLOCKER]` **(confidence: 9/10) — no fast test pins determinism on the shipped
path.** `test_build_fold_embedder_extraction_is_byte_identical_across_repeated_calls`
(`test_audio_emb_extract.py:159`) guards `build_fold_embedder`, which after Task
4 is no longer what the container calls. Determinism is a stated exit criterion
of #166, and `peft.set_adapter` is called *after* `eval()` on every extraction —
if it ever perturbs module training state, `lora_dropout=0.05` makes every score
a fresh random draw and the only guard left is a 21GB opt-in container test.
**Fix:** add a task mirroring the existing test for `build_ensemble_embedder`:
`embed(midi)` three times, `assert_array_equal` on all K rows. Cheap, fast, and
it directly protects the property the container test asserts expensively.

`[RISK]` **(confidence: 6/10) — the two `ValueError` guards on empty inputs
(`from_members([])`, `build_ensemble_embedder([])`) have no tests.** Verify
whether they are reachable at all: `discover_members` already raises on an empty
directory, so an empty list may be unreachable from any real caller — in which
case CLAUDE.md's "no error handling for impossible scenarios" says delete the
guards rather than test them. Decide one way or the other; do not leave untested
defensive code.

**Failure Modes**

`[OBS]` — No task mutates shared state, writes to a database, or runs async.
Mid-execution failure leaves a half-written model directory under `tmp_path` or
an `--out-dir`, and both builders `rmtree` before writing, so a re-run is clean.
`main`'s per-item flush (`score_wav.py:415-421`) already survives an OOM kill and
is untouched.

`[OBS]` — Silent-failure audit: every new failure path raises. The one silent
coercion introduced is the `ndim == 1` reshape, filed as a BLOCKER above. The
`fallback_score` disagreement check converts what would be a silent
artifact-mixing bug into a loud refusal — the right direction.

`[RISK]` **(confidence: 7/10) — the peft fallback ladder is only exercisable
against `_FakeOuter` until Task 11.** Task 5 answers "does peft 0.11.1 have this
API", not "does it work on the vendored fork's `LlamaForCausalLM`". Reading
`_extract_full_piece` (`train_fold.py:264-280`) confirms the shape of the risk
is narrow: it takes the inner transformer directly (`peft_model.model.model`),
and LoRA layers are injected *inside* that module tree, so `set_adapter` on the
wrapper must propagate to them for the switch to be visible. Test 6b is the
guard, but it guards the fake. **Watch for:** Task 11 producing K identical rows
on the real backbone. **Fallback:** the ladder in the plan, in order, measured.

`[OBS]` — Verified safe: sharing one `tokens` tensor across K members is sound.
`_extract_full_piece` does `tokens.to(device)` (non-mutating, returns self when
already resident) and slices into chunks; nothing writes in place. Confirmed by
reading `train_fold.py:272-280`.

### Presumption Inventory

| Assumption | Verdict | Reason |
|---|---|---|
| The Dockerfile needs no change | **SAFE** | Read it: `COPY model/ /app/model/` is wholesale and `MIREX_MODEL_DIR=/app/model`; only a stale comment is affected |
| `predict.sh` needs no change | **SAFE** | Read it: passes `--model-dir "$MIREX_MODEL_DIR"` through and never enumerates `adapter/` |
| `_extract_full_piece` takes tokens, so tokenization can be shared | **SAFE** | Verified at `train_fold.py:264`; no in-place mutation of `tokens` |
| Member `fallback_score` is identical by construction | **SAFE** | #168 STATE verified identical seg_id order and one shared 7,665-grade pool across all five runs |
| `peft==0.11.1` supports `load_adapter`/`set_adapter` on the fork's class | **VALIDATE** | Task 5 answers it for `_FakeOuter`; only Task 11 answers it for the fork. Ladder documented |
| `conftest.py` in the difficulty dir is collected and never ships | **SAFE** | pytest collects conftest from the target dir; `build_context.CODE_FILES` is an explicit list that excludes it |
| `build_fold_embedder` survives Task 4 with live callers | **SAFE** | `audio_emb_extract.py:261` and `synth_embed.py:130` still call it; no dead code created |
| Task 12 can re-measure runtime with `runtime_probe.py` as-is | **RISKY** | False. `assemble_model_dir` hardcodes one adapter and `push_runtime_probe.py` stages one head — filed as BLOCKER |
| The existing `score_wav` suite passes unchanged after Task 4 | **RISKY** | False for `test_load_scorer_binds_the_device_onto_the_transcriber` — filed as BLOCKER |
| A five-member runtime can be measured without HF Jobs GPU time | **RISKY** | No local GPU; Docker-on-macOS inflates perf ~2.5x; the baseline is a cuda number. Budget one pre-paid serialised probe run |

### Summary

```
[BLOCKER]  count: 5
[RISK]     count: 6
[QUESTION] count: 1
```

Blockers, in the order they should be fixed:
1. Task 12 unexecutable — `runtime_probe.py` and `push_runtime_probe.py` are missing from the File Changes table and only support K=1.
2. Task 4 silently breaks `test_load_scorer_binds_the_device_onto_the_transcriber`.
3. No fast determinism test on `build_ensemble_embedder`, the shipped path.
4. The `ndim == 1` reshape is a silent fallback and must be deleted.
5. Five tasks are horizontally sliced; split into one behaviour per task.

VERDICT: NEEDS_REWORK — resolve blockers 1-5 in the plan, then re-run /challenge.
