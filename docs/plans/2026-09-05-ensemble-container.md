# #175: 5-Adapter Pred-Mean Ensemble in the Submission Container — Implementation Plan

> **For the build agent:** one task = one failing test = one implementation =
> one commit. Dispatch groups per the dependency map below; tasks *within* a
> group touch the same file and must run in file order. Do NOT start execution
> until `/challenge` returns `VERDICT: PROCEED`.

**Goal:** `predict.sh <wav>` inside the submission image prints one float that is
the mean of five members' predictions, with exactly one transcription, one
tokenization and one 839M base load per run — and a pre-existing single-adapter
model directory still scores unchanged.

**Spec:** `docs/specs/2026-09-05-ensemble-container-design.md` (APPROVED)
**Issue:** [#175](https://github.com/Jai-Dhiman/crescendAI/issues/175), sub-issue of #104 (S5). Deadline 2026-10-01.
**Style:** follow the conventions already in `model/src/claim_measurement/difficulty/`
— plain pytest, no mocks of internal collaborators, fakes injected only at
process boundaries (`loader_factory`, `transcribe`, `embed`), and a docstring on
every test saying *why the behaviour matters*, not what the code does.

**Test command (every task):**
```
cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
```
Baseline on this branch is **172 passed / 4 skipped**. Every task must grow the
passed count and never shrink it. Single-file form:
```
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_score_wav.py -q --no-cov
```

## Hard constraints (do not violate in any task)

- **No task may report a tau-c from these artifacts.** Every labelled piece is
  in every Experiment B run's training set. If a number is wanted, the honest
  one is the label ceiling's `+0.0239` headroom (#168). This is the same rule
  `build_model_dir.py`'s `--from-all-data` manifest already carries.
- **This worktree's `model/data/` is EMPTY** (gitignored; the real tree is
  `/Users/jdhiman/Documents/crescendai/model/data`). Every automated test in
  this plan therefore builds synthetic fixtures under `tmp_path`. The two
  opt-in suites that need real artifacts (`test_score_wav_integration.py`,
  `test_container.py`) are run manually with `CRESCENDAI_DATA_ROOT` pointed at
  the primary checkout, and the plan says so where it matters.
- **Never `uv run --with` from inside `model/`** — it mutates the shared
  `.venv`. `score_wav.py` is a `# /// script` for the same reason, and any new
  dependency must be added to *both* its header and
  `audio_emb_extract.py`'s (the header tests pin this).
- **Member order is correctness, not tidiness.** Float addition is not
  associative; `discover_members` returns members sorted by name and nothing
  downstream may re-order them. #166's determinism exit criterion (two runs
  reproduce bit-identically) must still pass at the end.
- **Transcription and tokenization happen once per WAV, K times never.**
  Transcription owns the linear runtime term (0.036x realtime). Doing it per
  member is a ~5x regression on the dominant cost and risks the 24h/1-GPU
  exclusion clause.
- **The container's inverted failure policy stays in the container.**
  `score_wav_or_fallback` must not leak back into the research harness.
- Do not change the `Dockerfile` or `predict.sh`. `COPY model/ /app/model/` is
  wholesale, so a new subdirectory ships without edits (verified).
- Do not delete `model/data/results/phase1_lora/fold_embeddings_7665` — 5 GPU-hours.

## Two deviations from the spec's File Changes table (flag for /challenge)

1. **`model/src/claim_measurement/difficulty/conftest.py` is NEW.** The spec's
   Task Group 0 calls for a fixture builder shared by the `score_wav` and
   `build_model_dir` suites. The existing convention is a private `_helper` per
   test file (`_fold_npz`, `_save_fake_adapter`), which cannot be shared across
   files. A `conftest.py` is the standard seam and — unlike a helper module in
   the package — is **not** in `build_context.CODE_FILES`, so test scaffolding
   can never ship inside the image.
2. **`apps/inference/mirex-difficulty/test_build_context.py` is NEW.**
   `build_context.py` has no test file today, and Task 9 changes its validation
   logic. TDD needs somewhere to put the failing test. It is fast, needs no
   Docker and no checkpoint, and is run explicitly:
   `uv run python -m pytest apps/inference/mirex-difficulty/test_build_context.py -q --no-cov`.

## One decision the spec left open (flag for /challenge)

**What if BOTH layouts are present in one model directory?** The spec defines
`members/` as additive and legacy as still-loading, but not the collision.
This plan makes `discover_members` **raise** on ambiguity rather than silently
preferring one, because silently ignoring a stale legacy `adapter/` beside a
fresh `members/` is precisely the artifact-mixing failure the `fallback_score`
assertion exists to catch — it would ship a plausible-looking score from a
directory nobody chose. `build_ensemble_model_dir` therefore removes a legacy
`adapter/` + `ridge_head.npz` when it writes `members/`, exactly as
`build_model_dir` already `rmtree`s a stale adapter.

## Task Groups

```
Group 0 (solo, first):         Task 1        [conftest.py]
Group A (sequential-internal): Tasks 2-4     [score_wav.py]          parallel with B
Group B (sequential-internal): Tasks 5-6     [audio_emb_extract.py]  parallel with A
Group C (sequential-internal): Tasks 7-8     [build_model_dir.py]    depends on A
Group D (sequential-internal): Tasks 9-10    [build_context.py, test_container.py] depends on C
Group E (solo, last):          Tasks 11-12   [integration + runtime]  depends on all
```

- Group 0 first: every later group's tests consume its fixture.
- A and B share no file and no import (A imports `build_ensemble_embedder` only
  in Task 4, which is A's last task — if B is not done when A reaches Task 4,
  A waits).
- C imports `MEMBERS_SUBDIR` / `EnsembleHead` from `score_wav.py` (Group A).
- D stages what C builds.
- E is opt-in verification against real artifacts and the decision-log entry;
  it is human-run, not part of the fast suite.

**Ships-independently checkpoint:** after Group B, the ensemble embedder is
real and measurable in isolation; after Group A+B, `load_scorer` scores through
K adapters and the whole value of the issue is exercisable from Python. Groups
C-D are packaging. Nothing before Group A is independently useful.

---

## Group 0 — The Harness

### Task 1: A fixture that writes a synthetic two-member model directory

**Behavior being verified:** the fixture produces a directory whose members'
heads round-trip through `read_ridge_head`, in both the `members/` and the
legacy layout — so every later task asserts against a real on-disk directory
instead of a mock.

**Interface under test:**
`make_model_dir(members: dict[str, RidgeHead], layout: str = "members") -> Path`
(a factory fixture; `layout="legacy"` requires exactly one member)

**Files:**
- Create: `model/src/claim_measurement/difficulty/conftest.py`
- Create: `model/src/claim_measurement/difficulty/test_conftest_fixtures.py`

- [ ] **Step 1: Write the failing test** (`test_conftest_fixtures.py`)

```python
"""#175: the fixture builder every ensemble test consumes, tested itself.

A fixture that silently writes the wrong layout would make every downstream
assertion vacuous, so it gets the same treatment as production code.
"""
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
        head = read_ridge_head(member / "ridge_head.npz")
        np.testing.assert_array_equal(head.coef, np.full(4, expected))


def test_the_fixture_can_also_write_the_legacy_single_adapter_layout(make_model_dir):
    """The merged container, runtime_probe.py and the staged Hub bundles all
    use the legacy layout; tests that it still loads need it built here."""
    model_dir = make_model_dir({"only": _head()}, layout="legacy")

    assert (model_dir / "adapter" / "adapter_config.json").exists()
    assert (model_dir / "ridge_head.npz").exists()
    assert not (model_dir / "members").exists()
```

- [ ] **Step 2: Run test — verify it FAILS**
```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_conftest_fixtures.py -q --no-cov
```
Expected: FAIL — `fixture 'make_model_dir' not found`.

- [ ] **Step 3: Implement `conftest.py`**

```python
"""#175: shared fixtures for the ensemble model-directory tests.

Lives in conftest.py rather than in the package so it can be shared across
test files WITHOUT being importable production code -- `build_context.CODE_FILES`
enumerates what ships in the image, and test scaffolding must never be on that
list.

The adapters written here are placeholder JSON, not real LoRA weights: the fast
suite exercises layout discovery and head arithmetic, and the embedder is tested
separately with an injected loader (test_audio_emb_extract.py). Anything that
needs real adapter weights needs the 1.6GB checkpoint and is opt-in.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from claim_measurement.difficulty.score_wav import (
    ADAPTER_SUBDIR,
    HEAD_FILENAME,
    write_ridge_head,
)


@pytest.fixture
def make_model_dir(tmp_path):
    def build(members: dict, layout: str = "members", root: Path | None = None) -> Path:
        model_dir = Path(root) if root is not None else tmp_path / "model_dir"
        if layout == "legacy":
            if len(members) != 1:
                raise ValueError(
                    f"the legacy layout holds exactly one adapter, got "
                    f"{len(members)} -- a multi-member legacy dir cannot exist")
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

- [ ] **Step 4: Run test — verify it PASSES**, then run the full suite.
- [ ] **Step 5: Commit** — `test(#175): add a shared fixture for synthetic ensemble model directories`

---

## Group A — `score_wav.py`

### Task 2: `EnsembleHead` averages member predictions and refuses mismatched fallbacks

**Behavior being verified:** the mean over K members equals the mean of what
those K heads predict individually; K=1 is not a special case; and members whose
`fallback_score` disagrees are refused at construction, because differing
fallbacks mean adapters from different training pools were staged into one
directory — a silent artifact-mixing bug that would otherwise ship as a
plausible-looking average.

**Interface under test:**
`EnsembleHead.from_members(heads) -> EnsembleHead`;
`.predict(vectors: (K, D)) -> float`; `.fallback_score`; `.n_members`; `.n_features`

**Files:**
- Modify: `model/src/claim_measurement/difficulty/score_wav.py`
- Modify: `model/src/claim_measurement/difficulty/test_score_wav.py`

- [ ] **Step 1: Write the failing tests** (append to `test_score_wav.py`)

```python
def test_the_ensemble_score_is_the_mean_of_its_members_own_predictions():
    """The pre-registered submission model (#168) is the prediction mean over
    five heads. If this ever drifts to, say, a coefficient average, we would be
    shipping a model nobody measured."""
    heads = [_head(coef=1.0), _head(coef=2.0), _head(coef=3.0)]
    vectors = np.array([[1.0, 0.0, 0.0, 0.0]] * 3)

    ensemble = EnsembleHead.from_members(heads)

    individually = [h.predict(v) for h, v in zip(heads, vectors)]
    assert ensemble.predict(vectors) == pytest.approx(float(np.mean(individually)))
    assert ensemble.n_members == 3


def test_one_member_is_an_ensemble_of_one_not_a_separate_code_path():
    """K=1 must go through the same object as K=5. Two scoring paths means the
    single-adapter path is the tested one and the ensemble path is the one that
    ships, which is how a submission-day divergence happens."""
    head = _head(coef=2.0)
    vector = np.array([[1.0, 1.0, 0.0, 0.0]])

    ensemble = EnsembleHead.from_members([head])

    assert ensemble.n_members == 1
    assert ensemble.predict(vector) == pytest.approx(head.predict(vector[0]))
    assert ensemble.fallback_score == head.fallback_score


def test_members_whose_fallback_disagrees_are_refused():
    """Every member head is fit on the same 7,665 grades, so np.median(y) is
    identical by construction. A mismatch means adapters from two different
    training pools were staged into one directory -- fatal, not averageable."""
    with pytest.raises(ValueError, match="fallback"):
        EnsembleHead.from_members([_head(fallback=5.0), _head(fallback=6.0)])


def test_a_vector_matrix_of_the_wrong_shape_is_refused():
    """A (K,D) matrix whose K disagrees with the member count means the
    embedder and the heads were built from different member lists."""
    ensemble = EnsembleHead.from_members([_head(), _head()])

    with pytest.raises(ValueError, match="2 members"):
        ensemble.predict(np.zeros((3, 4)))
```

Add `_head` next to the existing `_training_data` helper, and extend the
top-of-file import with `EnsembleHead`:

```python
def _head(dim=4, coef=1.0, intercept=0.0, fallback=5.0) -> RidgeHead:
    return RidgeHead(mean=np.zeros(dim), scale=np.ones(dim),
                     coef=np.full(dim, coef), intercept=intercept,
                     fallback_score=fallback)
```

- [ ] **Step 2: Run test — verify it FAILS** (`ImportError: cannot import name 'EnsembleHead'`)

- [ ] **Step 3: Implement `EnsembleHead`** in `score_wav.py`, immediately after
  `RidgeHead` (it is the same section of the file and reads as its plural).

```python
@dataclass(frozen=True)
class EnsembleHead:
    """K RidgeHeads, scored as the mean of their predictions.

    K=1 is an EnsembleHead with one member, so `score_wav` has no
    single-vs-ensemble branch -- the only code that interprets layout is
    `discover_members`. The pre-registered #168 submission model is the
    prediction mean, not a coefficient mean; the two differ once the members'
    scalers differ, and only one of them was measured.
    """

    members: tuple

    @classmethod
    def from_members(cls, heads) -> "EnsembleHead":
        heads = tuple(heads)
        if not heads:
            raise ValueError("an ensemble needs at least one member head")
        fallbacks = {float(h.fallback_score) for h in heads}
        if len(fallbacks) != 1:
            # Not averaged, not warned about: every member is fit on the same
            # 7,665 grades, so np.median(y) is identical BY CONSTRUCTION. A
            # disagreement means artifacts from two training pools were staged
            # into one directory, and the mean of them is meaningless.
            raise ValueError(
                f"member heads disagree on fallback_score ({sorted(fallbacks)}) "
                f"-- they were fit on different training pools, so this "
                f"directory mixes artifacts and must not be scored")
        widths = {h.n_features for h in heads}
        if len(widths) != 1:
            raise ValueError(
                f"member heads expect different embedding widths ({sorted(widths)})")
        return cls(members=heads)

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
        if vectors.ndim != 2 or vectors.shape[0] != self.n_members:
            raise ValueError(
                f"expected a ({self.n_members}, {self.n_features}) matrix for "
                f"{self.n_members} members, got shape {vectors.shape} -- the "
                f"embedder and the heads were built from different member lists")
        # float(np.mean(...)) over a list built in member order. Float addition
        # is not associative, so this order is part of #166's determinism
        # criterion; do not reorder or vectorise into a different summation.
        return float(np.mean([h.predict(v) for h, v in zip(self.members, vectors)]))
```

- [ ] **Step 4: Run test — verify it PASSES**, then the full suite.
- [ ] **Step 5: Commit** — `feat(#175): add EnsembleHead, the pred-mean over K ridge heads`

---

### Task 3: `discover_members` resolves both layouts, in a stable order

**Behavior being verified:** a `members/` directory yields one `MemberPaths` per
subdirectory **sorted by name**; a legacy directory yields exactly one member; a
directory with neither fails loudly; and a directory with BOTH is refused rather
than silently preferring one.

**Interface under test:**
`discover_members(model_dir) -> list[MemberPaths]` with `.name`, `.adapter_dir`, `.head_path`

**Files:**
- Modify: `model/src/claim_measurement/difficulty/score_wav.py`
- Modify: `model/src/claim_measurement/difficulty/test_score_wav.py`

- [ ] **Step 1: Write the failing tests** (append to `test_score_wav.py`)

```python
def test_members_are_discovered_in_sorted_order(make_model_dir):
    """Float addition is not associative, so the mean depends on summation
    order. Sorting by name is what makes two independent container runs
    reproduce -- a stated exit criterion of #166."""
    model_dir = make_model_dir({"s2030": _head(), "s2026": _head(),
                                "s2028": _head()})

    members = discover_members(model_dir)

    assert [m.name for m in members] == ["s2026", "s2028", "s2030"]
    assert members[0].adapter_dir == model_dir / "members" / "s2026" / "adapter"
    assert members[0].head_path == model_dir / "members" / "s2026" / "ridge_head.npz"


def test_a_legacy_single_adapter_directory_is_a_one_member_ensemble(make_model_dir):
    """The merged container, runtime_probe.py and the staged Hub bundles all
    use the legacy layout. Regressing that path 26 days from the deadline to
    gain uniformity would be a bad trade, so it keeps loading."""
    model_dir = make_model_dir({"only": _head()}, layout="legacy")

    members = discover_members(model_dir)

    assert len(members) == 1
    assert members[0].adapter_dir == model_dir / "adapter"
    assert members[0].head_path == model_dir / "ridge_head.npz"


def test_a_directory_with_neither_layout_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="no adapter"):
        discover_members(tmp_path)


def test_a_directory_holding_both_layouts_is_refused(make_model_dir, tmp_path):
    """Silently preferring one would ship a score from a directory nobody
    chose -- the same artifact-mixing failure EnsembleHead's fallback check
    exists to catch."""
    model_dir = make_model_dir({"s2026": _head()}, root=tmp_path / "mixed")
    make_model_dir({"stale": _head()}, layout="legacy", root=tmp_path / "mixed")

    with pytest.raises(ValueError, match="both"):
        discover_members(model_dir)
```

- [ ] **Step 2: Run test — verify it FAILS.**

- [ ] **Step 3: Implement** in `score_wav.py`, and add `MEMBERS_SUBDIR = "members"`
  beside the existing `ADAPTER_SUBDIR` constant.

```python
@dataclass(frozen=True)
class MemberPaths:
    name: str
    adapter_dir: Path
    head_path: Path


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
        return [MemberPaths(name=d.name,
                            adapter_dir=d / ADAPTER_SUBDIR,
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

- [ ] **Step 4: Run test — verify it PASSES**, then the full suite.
- [ ] **Step 5: Commit** — `feat(#175): discover ensemble members from either model-dir layout`

---

### Task 4: `score_wav` and `load_scorer` score through K members

**Behavior being verified:** one WAV produces one transcription and one embed
call returning a `(K, D)` matrix, which reaches `EnsembleHead.predict`
unflattened; a non-finite entry anywhere in the matrix is still a loud failure;
and `load_scorer` on a legacy directory behaves exactly as it did before.

**Interface under test:** `score_wav(wav, transcribe, embed, head: EnsembleHead) -> float`;
`load_scorer(model_dir, ...)` unchanged in signature.

**Depends on:** Task 3, and Task 5 (`build_ensemble_embedder`) for the
`load_scorer` half. If Group B is not finished, do Steps 1-3 against the
embedder's agreed signature and block on B before Step 4.

**Files:**
- Modify: `model/src/claim_measurement/difficulty/score_wav.py`
- Modify: `model/src/claim_measurement/difficulty/test_score_wav.py`

- [ ] **Step 1: Write the failing tests** (append to `test_score_wav.py`)

```python
def test_a_wav_is_transcribed_once_no_matter_how_many_members(tmp_path):
    """Transcription owns the linear runtime term (0.036x realtime).
    Transcribing per member would be a ~5x regression on the dominant cost and
    put the 24h/1-GPU exclusion clause in play."""
    wav = tmp_path / "piece.wav"
    wav.write_bytes(b"")
    calls = []
    embed_calls = []

    def embed(midi_path):
        embed_calls.append(midi_path)
        return np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

    head = EnsembleHead.from_members([_head(coef=1.0), _head(coef=3.0)])

    score = score_wav(wav, _fake_transcribe(calls), embed, head)

    assert len(calls) == 1, "one WAV, one transcription"
    assert len(embed_calls) == 1, "one WAV, one embed call for all members"
    assert score == pytest.approx((1.0 + 3.0) / 2)


def test_a_non_finite_entry_in_any_member_row_is_a_failure(tmp_path):
    """A non-finite embedding means the backbone produced garbage for that
    member. Averaging it away -- or nan_to_num'ing it -- would emit a confident
    score from a broken forward pass."""
    wav = tmp_path / "piece.wav"
    wav.write_bytes(b"")
    bad = np.array([[1.0, 0.0, 0.0, 0.0], [np.nan, 0.0, 0.0, 0.0]])
    head = EnsembleHead.from_members([_head(), _head()])

    with pytest.raises(ValueError, match="non-finite"):
        score_wav(wav, _fake_transcribe(), lambda p: bad, head)


def test_load_scorer_still_reads_a_legacy_single_adapter_directory(
        make_model_dir, monkeypatch):
    """The merged container and the staged Hub bundles are all legacy-layout.
    This is the regression guard on not breaking a working submission path."""
    model_dir = make_model_dir({"only": _head()}, layout="legacy")
    seen = {}

    def fake_build(adapter_dirs, *a, **kw):
        seen["adapter_dirs"] = list(adapter_dirs)
        return lambda midi_path: np.zeros((1, 4))

    monkeypatch.setattr(score_wav_module, "build_ensemble_embedder", fake_build)
    score = score_wav_module.load_scorer(
        model_dir, checkpoint=None, repo_root=None, model_config=None,
        transcribe=_fake_transcribe())

    assert seen["adapter_dirs"] == [model_dir / "adapter"]
    assert isinstance(score(Path("x.wav"))[0], float)
```

- [ ] **Step 2: Run test — verify it FAILS.**

- [ ] **Step 3: Implement.** In `score_wav`, replace the single-vector handling:

```python
        vectors = embed(midi_path)

    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim == 1:                      # a one-member embedder may hand
        vectors = vectors.reshape(1, -1)       # back a bare vector
    if not np.all(np.isfinite(vectors)):
        raise ValueError(
            f"{wav_path} produced a non-finite embedding "
            f"({int((~np.isfinite(vectors)).sum())} of {vectors.size} entries)")

    score = head.predict(vectors)
```

  and in `load_scorer`, replace the head/embedder construction:

```python
    members = discover_members(model_dir)
    head = EnsembleHead.from_members(
        [read_ridge_head(m.head_path) for m in members])
    embed = build_ensemble_embedder(
        [m.adapter_dir for m in members], checkpoint, repo_root, model_config,
        device=device)
```

  Update the `load_scorer` docstring: the backbone is loaded ONCE and all K
  adapters are attached to it; the type annotation on `score_wav` /
  `score_wav_or_fallback` becomes `head: EnsembleHead`.

- [ ] **Step 4: Run test — verify it PASSES**, then the full suite. Confirm
  `test_scoring_the_same_wav_twice_gives_a_bit_identical_score` still passes.
- [ ] **Step 5: Commit** — `feat(#175): score a WAV through every ensemble member from one transcription`

---

## Group B — `audio_emb_extract.py`

### Task 5: `build_ensemble_embedder` loads the 839M base exactly once

**Behavior being verified:** K adapters produce a `(K, D)` matrix from one
`loader_factory` call. **This is the task that empirically resolves the spec's
open question** — whether `peft==0.11.1` supports `load_adapter`/`set_adapter`
on the fork's model class. The existing `_FakeOuter` + `_save_fake_adapter`
machinery in `test_audio_emb_extract.py` exercises the real `peft` API against a
fake base, which answers the API question; the fork-class question is settled
separately by Task 11's opt-in run against the real checkpoint.

**Fallback ladder (only after measuring, never as a first choice):**
`load_adapter`/`set_adapter` → `add_adapter` against a shared base → K base
loads with the memory cost recorded in the decision log.

**Interface under test:**
`build_ensemble_embedder(adapter_dirs, checkpoint, repo_root, model_config, loader_factory=_real_loader, device=None) -> embed(midi_path) -> (K, D)`

**Files:**
- Modify: `model/src/claim_measurement/difficulty/audio_emb_extract.py`
- Modify: `model/src/claim_measurement/difficulty/test_audio_emb_extract.py`

- [ ] **Step 1: Write the failing tests** (append to `test_audio_emb_extract.py`)

```python
def test_the_base_backbone_is_loaded_once_for_all_members(tmp_path):
    """build_fold_embedder's loader_factory loads the full 839M base on every
    call, so K calls would load 1.6GB K times -- a memory problem and a
    startup-cost problem. Five 3.2M-parameter LoRA deltas share one base."""
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


def test_one_member_goes_through_the_same_path_as_five(tmp_path):
    """K=1 must not be a separate branch; the container's legacy-layout path
    is this call with a one-element list."""
    adapter_dir = tmp_path / "adapter"
    _save_fake_adapter(adapter_dir)
    midi_path = tmp_path / "piece.mid"
    write_notes_midi(_NOTES, midi_path)

    embed = build_ensemble_embedder([adapter_dir], checkpoint=None,
                                    repo_root=None, model_config=None,
                                    loader_factory=_fake_loader_factory)

    assert embed(midi_path).shape == (1, 4)
```

Promote the existing inline `fake_loader_factory` to a module-level
`_fake_loader_factory` so both tests use it.

- [ ] **Step 2: Run test — verify it FAILS.**

- [ ] **Step 3: Implement** `build_ensemble_embedder` in `audio_emb_extract.py`,
  directly after `build_fold_embedder`. Preserve the loader → stub → `import peft`
  order exactly (`test_peft_is_imported_only_after_the_loader_puts_the_fork_on_sys_path`
  asserts it on the source of `build_fold_embedder`; add the same guard for the
  new function in Task 6 if the ordering line moves).

```python
def build_ensemble_embedder(adapter_dirs, checkpoint: Path, repo_root: Path,
                            model_config: Path, loader_factory=_real_loader,
                            device=None):
    """A `embed(midi_path) -> (K, D)` closure over ONE resident base backbone
    with K LoRA adapters attached by name.

    The base is 839M parameters (1.6GB); the adapters are 3.2M each. Calling
    build_fold_embedder K times would load the base K times, which is both a
    memory problem and a startup-cost problem the 24h/1-GPU budget cannot
    absorb.

    Adapters are attached in the order given -- `discover_members` sorts, and
    that order is part of the determinism contract, so this function must not
    re-order.
    """
    adapter_dirs = [Path(d) for d in adapter_dirs]
    if not adapter_dirs:
        raise ValueError("an ensemble embedder needs at least one adapter dir")

    base_model, tokenize, max_len = loader_factory(
        checkpoint, repo_root=repo_root, model_config=model_config)
    _stub_absent_transformers_models()
    from peft import PeftModel

    names = [f"m{i}" for i in range(len(adapter_dirs))]
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

    def embed(midi_path: Path) -> np.ndarray:
        # Tokenised ONCE and reused across members: _extract_full_piece takes
        # tokens, not a path, and the tokenizer is member-independent.
        tokens = tokenize(Path(midi_path))
        rows = []
        for name in names:
            peft_model.set_adapter(name)
            transformer = peft_model.model.model  # inner LoRA-injected transformer
            rows.append(_extract_full_piece(transformer, tokens, max_len,
                                            device=device))
        return np.stack(rows)

    return embed
```

  **If `load_adapter`/`set_adapter` raises on this peft version:** stop, record
  the exact traceback in the task notes, and take the next rung of the fallback
  ladder above. Do not silently degrade to K base loads.

- [ ] **Step 4: Run test — verify it PASSES**, then the full suite.
- [ ] **Step 5: Commit** — `feat(#175): attach K LoRA adapters to one resident MoonBeam base`

---

### Task 6: One tokenization per WAV, and the members must actually differ

**Behavior being verified:** `tokenize` is called once per `embed` call
regardless of K, **and** the K rows are not all identical — because a
`set_adapter` that silently no-ops would produce K copies of one member's
embedding and an "ensemble" that is one adapter wearing five names. That failure
is invisible in the score: it is a plausible number from the wrong model.

**Files:**
- Modify: `model/src/claim_measurement/difficulty/audio_emb_extract.py` (only if the tests fail)
- Modify: `model/src/claim_measurement/difficulty/test_audio_emb_extract.py`

- [ ] **Step 1: Write the failing tests** (append to `test_audio_emb_extract.py`)

```python
def test_a_wav_is_tokenized_once_no_matter_how_many_members(tmp_path):
    """_extract_full_piece takes TOKENS, not a path, so tokenization is
    member-independent and belongs outside the member loop. Doing it K times
    is pure waste on the fixed runtime term."""
    dirs = [tmp_path / f"s{i}" / "adapter" for i in (2026, 2027, 2028)]
    for d in dirs:
        _save_fake_adapter(d)
    midi_path = tmp_path / "piece.mid"
    write_notes_midi(_NOTES, midi_path)
    tokenize_calls = []

    def counting_loader(checkpoint_path, repo_root, model_config):
        def tokenize(path):
            tokenize_calls.append(path)
            return torch.arange(6) % 16
        return (_FakeOuter(hidden=4, n_layers=1, vocab=16), tokenize, 4)

    embed = build_ensemble_embedder(dirs, checkpoint=None, repo_root=None,
                                    model_config=None,
                                    loader_factory=counting_loader)
    embed(midi_path)

    assert len(tokenize_calls) == 1, (
        f"one MIDI, one tokenization; got {len(tokenize_calls)}")


def test_switching_adapters_actually_changes_the_embedding(tmp_path):
    """If set_adapter silently no-ops, every row is the same member's
    embedding and the 'ensemble' is one adapter wearing five names -- a
    plausible-looking score from a model we never pre-registered. Two adapters
    saved from independently-initialised LoRA layers must not agree."""
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

  `_save_fake_adapter` gains an `init_lora_weights` pass-through: PEFT
  initialises LoRA's `B` matrix to zeros by default, which makes every adapter
  a mathematical identity and this test vacuously unable to fail. That default
  is exactly why the assertion needs `init_lora_weights=False`.

- [ ] **Step 2: Run test.** The tokenize test should already PASS from Task 5's
  implementation — if it does, say so explicitly in the commit and keep the test
  (it pins a property that is easy to regress). The adapter-switch test is the
  one that may fail and drive implementation.
- [ ] **Step 3: Implement** only what the failures demand.
- [ ] **Step 4: Run the full suite.**
- [ ] **Step 5: Commit** — `test(#175): pin one tokenization per WAV and a real adapter switch`

---

## Group C — `build_model_dir.py`

### Task 7: `build_ensemble_model_dir` stages K members under their source basenames

**Behavior being verified:** K artifact directories become
`members/<basename>/adapter` + `ridge_head.npz`, each head fit on its own run's
`emb_fold99.npz` with no exclusions; duplicate basenames are refused rather than
silently overwriting a member; a stale legacy `adapter/` in the output directory
is removed so `discover_members` never sees both layouts.

**Interface under test:**
`build_ensemble_model_dir(out_dir, artifact_dirs, **provenance) -> Path`

**Files:**
- Modify: `model/src/claim_measurement/difficulty/build_model_dir.py`
- Modify: `model/src/claim_measurement/difficulty/test_build_model_dir.py`

- [ ] **Step 1: Write the failing tests** (append to `test_build_model_dir.py`)

```python
def test_each_member_is_staged_under_its_source_directory_basename(tmp_path):
    """Provenance readable off the layout: `s2026` comes from
    `.../alldata_s2026`. The sort order EnsembleHead depends on is then stable
    AND meaningful, rather than an accident of copy order."""
    artifacts = []
    for seed in (2026, 2027):
        d = tmp_path / f"alldata_s{seed}"
        _fold_npz(d / "emb_fold99.npz", n=60, dim=8, seed=seed)
        (d / "adapter").mkdir(parents=True)
        (d / "adapter" / "adapter_config.json").write_text("{}")
        artifacts.append(d)
    out_dir = tmp_path / "ensemble"

    build_ensemble_model_dir(out_dir, artifacts)

    assert sorted(p.name for p in (out_dir / "members").iterdir()) == [
        "alldata_s2026", "alldata_s2027"]
    for name in ("alldata_s2026", "alldata_s2027"):
        assert (out_dir / "members" / name / "adapter" / "adapter_config.json").exists()
        assert (out_dir / "members" / name / "ridge_head.npz").exists()


def test_two_artifact_dirs_with_the_same_basename_are_refused(tmp_path):
    """Silently overwriting would ship a four-member ensemble labelled five."""
    ...


def test_a_stale_legacy_adapter_is_cleared_when_members_are_written(tmp_path):
    """discover_members refuses a directory holding both layouts, so the
    builder must not create one by writing members/ into an old model dir."""
    ...


def test_the_staged_members_load_as_one_ensemble_head(tmp_path):
    """The end of the vertical slice: what the builder writes is exactly what
    load_scorer reads. Also proves the fallback_score agreement holds for heads
    fit from the same grade distribution."""
    ...
    members = discover_members(out_dir)
    head = EnsembleHead.from_members([read_ridge_head(m.head_path) for m in members])
    assert head.n_members == 2
```

- [ ] **Step 2: Run test — verify it FAILS.**
- [ ] **Step 3: Implement** `build_ensemble_model_dir`, reusing
  `fit_head_from_fold_embeddings(emb, exclude_seg_ids=None)` per member —
  *never* re-deriving folds. The all-data rationale in `_build_all_data`'s
  docstring applies unchanged and must not be restated in a way that drifts.
- [ ] **Step 4: Run test — verify it PASSES**, then the full suite.
- [ ] **Step 5: Commit** — `feat(#175): stage K all-data artifacts into one ensemble model directory`

---

### Task 8: `--from-ensemble` on the CLI, with a manifest that discloses K

**Behavior being verified:** `--from-ensemble <dir> --from-ensemble <dir> ...
--out-dir <out>` builds the directory; the manifest records the member names,
per-member head rows, `n_members`, and carries the same no-tau-c warning
`--from-all-data` does; and `--from-ensemble` is mutually exclusive with the
other two sources.

**Why the manifest matters:** MIREX 2026 makes training-data size, model size
and compute a *mandatory disclosure*. A five-member ensemble is five times the
inference compute and that has to be readable off the artifact.

**Files:**
- Modify: `model/src/claim_measurement/difficulty/build_model_dir.py`
- Modify: `model/src/claim_measurement/difficulty/test_build_model_dir.py`

- [ ] **Step 1: Write the failing tests** — CLI invocation through `main(argv=[...])`,
  matching the existing `test_the_submission_manifest_carries_the_no_tau_c_warning`
  and `test_a_fold_artifact_dir_passed_to_the_submission_path_is_refused` style.
- [ ] **Step 2: Run test — verify it FAILS.**
- [ ] **Step 3: Implement** the `--from-ensemble` `action="append"` argument, the
  three-way mutual exclusion, and the manifest. Update the module docstring's
  layout block to show `members/`.
- [ ] **Step 4: Run test — verify it PASSES**, then the full suite.
- [ ] **Step 5: Commit** — `feat(#175): add --from-ensemble to build_model_dir`

---

## Group D — Container staging

### Task 9: `build_context.py` accepts either layout

**Behavior being verified:** staging a `members/`-layout model directory
succeeds; staging a legacy one still succeeds; staging a directory with neither
fails in seconds with a message naming both layouts — before 1.6GB is copied.

**Files:**
- Modify: `apps/inference/mirex-difficulty/build_context.py`
- Create: `apps/inference/mirex-difficulty/test_build_context.py`

- [ ] **Step 1: Write the failing tests.** `build_context`'s other inputs
  (checkpoint, fork repo, Transkun shim, code files) are `_require`d too, so the
  test builds a minimal fake tree for all of them under `tmp_path` — cheap, since
  every check is `path.exists()`. Assert the failure message names both layouts.
- [ ] **Step 2: Run test — verify it FAILS.**
- [ ] **Step 3: Implement.** Replace the two hardcoded `_require` lines:
```python
    _require(model_dir / "ridge_head.npz", "the ridge head")
    _require(model_dir / "adapter" / "adapter_config.json", "the LoRA adapter")
```
  with a layout check accepting `members/<name>/adapter/adapter_config.json` +
  `ridge_head.npz` for at least one member, or the legacy pair. Duplicating the
  rule rather than importing `discover_members` is deliberate: `build_context.py`
  runs OUTSIDE the isolated script env and importing `score_wav` would drag in
  peft/torch. The test above is what keeps the duplicate honest.
- [ ] **Step 4: Run test — verify it PASSES**, plus the model suite (unchanged).
- [ ] **Step 5: Commit** — `feat(#175): stage either model-dir layout into the container context`

---

### Task 10: The image scores through five members

**Behavior being verified:** a five-member image prints exactly one clean line
per WAV (stdout must stay clean — the fork's `MusicTokenizer` prints ~96KB of
vocabulary on construction), two runs of the same WAV agree bit-identically, and
the baked manifest reports five members.

**Not verified here:** that the container's score equals the mean of five
individually-run scores. Proving that needs five separate images or five mounted
model dirs, at ~21GB and minutes per run each. Task 11 proves it at the Python
level against the real backbone with K=2, which is the same property at a
tractable cost.

**Files:**
- Modify: `apps/inference/mirex-difficulty/test_container.py`

- [ ] **Step 1: Write the failing tests** — extend the existing opt-in suite.
- [ ] **Step 2..4:** requires a built image; run per the file's header with
  `CRESCENDAI_RUN_CONTAINER_TESTS=1`. This is a **human-lit** step: it needs the
  primary checkout's artifacts and a ~21GB build.
- [ ] **Step 5: Commit** — `test(#175): assert the image scores through five members deterministically`

---

## Group E — Real-artifact verification and the runtime number

### Task 11: Opt-in integration — the real backbone, two real adapters

**Behavior being verified, against the real fork model class:** that
`load_adapter`/`set_adapter` work on the vendored MoonBeam fork's class (the
spec's open question, which Task 5 can only answer for `_FakeOuter`), that the
base loads once, and that a two-member ensemble's score equals the mean of the
two single-member directories' scores on the same WAV.

Run from the primary checkout, or from the worktree with
`CRESCENDAI_DATA_ROOT` pointed at it — `model/data/` is empty here.

**Files:** Modify `model/src/claim_measurement/difficulty/test_score_wav_integration.py`

- [ ] Build two single-member dirs and one two-member dir from the Experiment B
  artifacts, score the same WAV through all three, assert the mean.
- [ ] **Commit** — `test(#175): verify the ensemble mean end-to-end on the real backbone`

### Task 12: Re-measure runtime and record it

**Why:** the spec *predicts* `~5.7s + 0.0360x` from `4.93s + 0.0360x`
(+0.19s/member on the fixed term only). That is a prediction to be measured, not
an assumption to ship on. The 24h/1-GPU clause is an exclusion criterion.

- [ ] Run `runtime_probe.py` on cuda against the five-member directory.
- [ ] Record the measured fixed and linear terms in
  `docs/mirex/track-a-difficulty-prediction.md` as a dated entry, alongside the
  single-adapter `4.93s + 0.0360x` so the delta is readable. State the implied
  items-per-24h at the 103s corpus median.
- [ ] If the linear term moved at all, stop: something is transcribing or
  tokenizing per member, and the fix is in Group B, not in the log.
- [ ] **Commit** — `docs(#175): record the measured five-member container runtime`

---

## Definition of done

- `cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov`
  is green with materially more than 172 passing tests.
- `discover_members` is the only function in the tree that knows there are two
  layouts.
- A five-member image prints one deterministic float per WAV, and a legacy
  single-adapter directory still scores unchanged.
- The measured runtime is in the decision log, with the linear term unmoved.
- `STATE:` posted on #175. No tau-c reported anywhere.
