# #138 Phase 1: LoRA Fine-Tune of MoonBeam-839M for Difficulty Ranking — Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per
> task) except where a group is explicitly marked sequential-internally (tasks in
> that group touch the same file and must run in file order). Do NOT start
> execution until `/challenge` returns `VERDICT: PROCEED`.

**Goal:** Produce a LoRA fine-tune of MoonBeam-839M whose mean-pooled embeddings,
scored through `bakeoff_cv.py`'s own composer-disjoint folds, beat the 37 hand
features (tau-c 0.8048) by a paired-bootstrap-significant margin at n=900, with
the same delta holding on the 709-piece real-audio subset.

**Spec:** `docs/specs/2026-08-04-phase1-lora-moonbeam-difficulty-design.md`
**Style:** Follow `CLAUDE.md` and the established conventions in
`model/src/claim_measurement/difficulty/` (plain pytest, no mocks of internal
collaborators, injected fakes only at process boundaries — see
`moonbeam_extract_script.py` / `test_moonbeam_extract_script.py` and
`moonbeam_backbone.py` / `test_moonbeam_backbone.py`).

**Test command (every task):**
```
cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
```
41 tests are green today; this plan must grow that number. Single-file form:
```
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_fold_plan.py -q --no-cov
```

**Hard constraints (do not violate in any task):**
- Never `uv run --with` from inside `model/` — it mutates the shared `.venv`.
  `train_fold.py` is a standalone `# /// script` (like `moonbeam_extract_script.py`)
  for the same reason.
- `composer_index.json` is READ-ONLY on this path. `train_fold.py` never calls
  `extract_embeddings`; it writes `emb_fold{F}.npz` directly via
  `write_fold_embeddings`. `ft_eval.py` reads grades/composer ids from the
  EXISTING `emb/features37/*.npz`.
- `ft_eval.py` imports `composer_disjoint_folds` from `bakeoff_cv.py`, never
  reimplements it.
- No task launches an HF job, spends money, hits the network, or requires a GPU.
  Every task is verified by the CPU pytest command above (Task Group 0 additionally
  by one real-data script re-run, itself GPU-free and network-free).
- Never delete or modify `model/data/results/amt_gap_curve/wav/`.
- This worktree's `model/data/` is EMPTY (verified this session: no
  `results/amt_gap_curve/manifest.json`, no `results/bakeoff/`). Every automated
  test in this plan therefore uses synthetic `tmp_path` fixtures, exactly like the
  existing suite (`test_bakeoff_sampling.py`, `test_run_bakeoff.py`, etc.) — none
  of them depend on real data being present. Where the spec's verified real-data
  facts (fold sizes, composer counts) are relevant, the task says so explicitly
  and states that they are confirmed by a documented manual re-run against the
  main checkout's data, not by the automated suite.

## Task Groups

```
Group 0 (solo, first):        Task 1                              [bakeoff_cv.py, features37_compare.py]
Group A (sequential-internal): Tasks 2-5   (parallel with Group B) [fold_plan.py]
Group B (sequential-internal): Tasks 6-10  (parallel with Group A) [ranking_loss.py]
Group C (sequential-internal): Tasks 11-12 (depends on Group B)    [train_fold.py]
Group D (sequential-internal): Tasks 13-15 (depends on Group C)    [ft_eval.py]
Group E (sequential-internal): Tasks 16-19 (depends on Group 0; parallel with C/D/F) [realaudio_check.py]
Group F (sequential-internal): Tasks 20-23 (depends on Group A; parallel with C/D/E) [push_train_dataset.py]
Group G (solo, last):          Task 24     (depends on C, D, E, F) [docs/mirex/phase1-lora-runbook.md]
```

Group 0 must complete before any other group starts (bakeoff_cv.py is the shared
import surface). After Group 0: Groups A and B have no file or import overlap and
run concurrently. Group C imports `ranking_loss.combined_loss` and
`bakeoff_cv.tau_c`, so it depends on B (and transitively 0), not on A directly —
but do not start C before A is also done, since a human is meant to read the fold
plan story in order. Group D imports `train_fold.read_fold_embeddings`, so it
depends on C. Group E only imports from `bakeoff_cv.py` (Group 0), so it can run
concurrently with C/D. Group F only imports `fold_plan.FoldPlan`/`build_fold_plans`
and `bakeoff_sampling.load_bakeoff_manifest` (Group A), so it can also run
concurrently with C/D/E. Group G is documentation referencing every CLI built in
C/D/E/F, so it is last.

None of these modules are `[SHIPS INDEPENDENTLY]` in the sense of independent user
value: Phase 1 is one measurement pipeline, and no partial subset of it answers
the gate question. The nearest thing to an early-value checkpoint is Group 0 (the
harness), which by itself re-confirms the Phase 0 numbers this whole phase is
measured against.

---

## Group 0 — The Harness (must come first)

### Task 1: Promote `paired_boot` into `bakeoff_cv.py`, rewire `features37_compare.py`, re-verify the reference numbers

**Group:** 0 (solo, blocks every other group)

**Behavior being verified:** `paired_boot`'s bootstrap CI is strictly positive
when arm B is uniformly less noisy than arm A, and straddles zero when the two
arms are identical — the exact statistical contract both `ft_eval.py` (Group D)
and `realaudio_check.py` (Group E) will rely on. Then, moving it does not change
`features37_compare.py`'s printed reference numbers.

**Interface under test:** `bakeoff_cv.paired_boot(oof_a, oof_b, y, seed=2026, n_boot=2000) -> tuple[float, float, float, float]`

**Files:**
- Modify: `model/src/claim_measurement/difficulty/bakeoff_cv.py`
- Modify: `model/src/claim_measurement/difficulty/features37_compare.py`
- Modify: `model/src/claim_measurement/difficulty/test_bakeoff_cv.py`

- [ ] **Step 1: Write the failing tests** (append to `test_bakeoff_cv.py`)

```python
def test_paired_boot_ci_is_strictly_positive_when_b_is_uniformly_better():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 11, size=200).astype(float)
    oof_a = y + rng.normal(scale=3.0, size=200)  # noisy
    oof_b = y + rng.normal(scale=0.2, size=200)  # much less noisy -> higher tau-c

    mean_diff, lo, hi, p_le_0 = paired_boot(oof_a, oof_b, y, seed=2026, n_boot=500)

    assert mean_diff > 0
    assert lo > 0
    assert p_le_0 < 0.05


def test_paired_boot_ci_straddles_zero_when_arms_are_identical():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 11, size=200).astype(float)
    oof_a = y + rng.normal(scale=1.0, size=200)
    oof_b = oof_a.copy()  # identical arm -> diff is exactly zero every resample

    mean_diff, lo, hi, p_le_0 = paired_boot(oof_a, oof_b, y, seed=2026, n_boot=200)

    assert abs(mean_diff) < 1e-9
    assert lo <= 0 <= hi
```

Also add `paired_boot` to the existing top-of-file import in `test_bakeoff_cv.py`:

```python
from claim_measurement.difficulty.bakeoff_cv import (
    composer_disjoint_folds,
    oof_tau_ridge,
    paired_boot,
    tau_c,
)
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_bakeoff_cv.py -q --no-cov
```
Expected: FAIL — `ImportError: cannot import name 'paired_boot' from 'claim_measurement.difficulty.bakeoff_cv'`

- [ ] **Step 3: Implement — add `paired_boot` to `bakeoff_cv.py`** (append to the
  file, after `oof_tau_ridge`)

```python
def paired_boot(oof_a: np.ndarray, oof_b: np.ndarray, y: np.ndarray,
                 seed: int = 2026, n_boot: int = 2000) -> tuple[float, float, float, float]:
    """Bootstrap the tau-c(b) - tau-c(a) difference over PIECES, resampling the
    SAME indices for both arms so the fold noise they share cancels. Promoted
    from features37_compare.py (a standalone `# /// script` that ft_eval.py
    cannot import -- lightgbm is not in model/.venv) so the gate (ft_eval.py,
    realaudio_check.py) and the Phase 0 baseline share one bootstrap
    implementation. Returns (mean_diff, ci_lo, ci_hi, P(diff <= 0))."""
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        diffs[b] = (stats.kendalltau(oof_b[i], y[i], variant="c").statistic
                    - stats.kendalltau(oof_a[i], y[i], variant="c").statistic)
    lo, hi = (float(v) for v in np.percentile(diffs, [2.5, 97.5]))
    return float(np.mean(diffs)), lo, hi, float(np.mean(diffs <= 0))
```

  Then rewire `features37_compare.py`: delete its local `def paired_boot(...):`
  block (the ten lines currently between `def oof_lgbm(...)` and `def main(...)`),
  add `paired_boot` to its existing `bakeoff_cv` import so the import block reads:

```python
from claim_measurement.difficulty.bakeoff_cv import (  # noqa: E402
    composer_disjoint_folds,
    paired_boot,
    tau_c,
)
```

  and remove the now-unused `N_BOOT` from the module-level tuple assignment
  (it was only read inside the deleted local `paired_boot`):

```python
N_FOLDS, SEEDS = 5, list(range(2026, 2031))
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
```
Expected: PASS, 43 passed (41 existing + 2 new).

- [ ] **Step 5: Re-run the harness against real data — verify the reference numbers are unchanged**

```bash
cd model/src/claim_measurement/difficulty && uv run --no-project --script \
    features37_compare.py --data-root /Users/jdhiman/Documents/crescendai/model/data
```
Expected: prints `features37|ridge` tau-c `0.8048` and `moonbeam_mean|ridge` tau-c
`0.8257` (matching the numbers already recorded in the design spec). This is the
harness: it locks the reference values the rest of Phase 1 is measured against,
confirming the promotion moved code without moving behavior. (Uses
`--data-root` pointing at the main checkout because this worktree's `model/data`
is empty — no GPU, no network, no HF job; pure local file read + RidgeCV.)

- [ ] **Step 6: Commit**

```bash
git add model/src/claim_measurement/difficulty/bakeoff_cv.py \
        model/src/claim_measurement/difficulty/features37_compare.py \
        model/src/claim_measurement/difficulty/test_bakeoff_cv.py \
    && git commit -m "refactor(#149): promote paired_boot into bakeoff_cv.py"
```

---

## Group A — `fold_plan.py` (sequential internally; parallel with Group B)

This module is the entire leakage argument (see spec). All four tasks build one
file, so a build agent must run them in order 2 → 3 → 4 → 5. Note on the
verified real-data facts (exact pool sizes 3815/4082/4283/4028/4149,
510/511/510/508/515 composers): this worktree's `model/data` is empty, so none
of the tests below assert those exact numbers — they assert the INVARIANTS
(option-D exclusion, composer-disjoint val carve, leakage detection) on
synthetic fixtures, exactly like the rest of this test suite. Task 3's Step 5
records a documented manual re-run against the main checkout that DOES confirm
the exact counts; that re-run is not part of the automated gate.

### Task 2: `build_fold_plans` excludes eval pieces and test-fold composers from train (option D)

**Group:** A (first; depends on Group 0's `composer_disjoint_folds` import, no new code there)

**Behavior being verified:** For every fold, no eval piece and no eval-fold-composer's
pool piece ever appears in that fold's train or val set.

**Interface under test:** `fold_plan.build_fold_plans(eval_entries, pool_entries, n_folds, seed, val_frac) -> list[FoldPlan]`

**Files:**
- Create: `model/src/claim_measurement/difficulty/fold_plan.py`
- Create: `model/src/claim_measurement/difficulty/test_fold_plan.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for fold_plan (#149 / #138 Phase 1) -- the option-D per-fold training
set construction + leakage invariants.

Run: cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
"""
import pytest

from claim_measurement.difficulty.bakeoff_sampling import ManifestEntry
from claim_measurement.difficulty.fold_plan import FoldPlan, build_fold_plans, check_fold_plans


def _entries(n_composers: int, pieces_per_composer: int, prefix: str) -> list[ManifestEntry]:
    return [
        ManifestEntry(seg_id=f"{prefix}c{c}_p{p}", key=f"{prefix}c{c}_p{p}.mid",
                      grade=p % 11, composer=f"composer_{c}")
        for c in range(n_composers) for p in range(pieces_per_composer)
    ]


def test_build_fold_plans_excludes_eval_pieces_and_test_fold_composers_from_train():
    eval_entries = _entries(n_composers=20, pieces_per_composer=1, prefix="eval_")
    pool_entries = eval_entries + _entries(n_composers=20, pieces_per_composer=5, prefix="pool_")

    plans = build_fold_plans(eval_entries, pool_entries, n_folds=5, seed=2026, val_frac=0.12)

    assert len(plans) == 5
    eval_seg_ids = {e.seg_id for e in eval_entries}
    pool_composer_of = {e.seg_id: e.composer for e in pool_entries}
    for plan in plans:
        train_and_val = set(plan.train_seg_ids) | set(plan.val_seg_ids)
        assert not (train_and_val & eval_seg_ids), "an eval piece leaked into train/val"
        test_composers = {e.composer for e in eval_entries if e.seg_id in plan.test_seg_ids}
        train_composers = {pool_composer_of[s] for s in plan.train_seg_ids}
        assert not (test_composers & train_composers), "a test composer leaked into train"
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_fold_plan.py -q --no-cov
```
Expected: FAIL — `ModuleNotFoundError: No module named 'claim_measurement.difficulty.fold_plan'`

- [ ] **Step 3: Implement**

```python
"""Option-D per-fold training-set construction + leakage invariants for #138
Phase 1 (LoRA fine-tune of MoonBeam). This module IS the entire leakage
argument -- get build_fold_plans and check_fold_plans right and the fine-tune
cannot see an eval piece or an eval-fold composer during training.

Composer-disjointness is a PER-FOLD constraint (see the design spec): fold f's
train pool excludes composers that appear in fold f's test set, not composers
appearing in ANY fold's test set. A set of per-fold adapters is therefore
welded to the (n_folds, seed) pair that produced them -- a different seed's
test fold can contain composers these adapters trained on.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from claim_measurement.difficulty.bakeoff_cv import composer_disjoint_folds


@dataclass(frozen=True)
class FoldPlan:
    fold: int
    test_seg_ids: tuple
    train_seg_ids: tuple
    val_seg_ids: tuple


def build_fold_plans(eval_entries, pool_entries, n_folds: int, seed: int,
                      val_frac: float) -> list:
    """eval_entries: the 900-piece eval sample. pool_entries: the full eligible
    pool (superset of eval_entries). For fold f: test = fold f of
    composer_disjoint_folds(eval composers, n_folds, seed); train pool =
    pool_entries minus every eval piece and minus every piece whose composer
    appears in fold f's test set (option D); val is a composer-disjoint
    ~val_frac slice carved out of that train pool for early stopping."""
    eval_composers = np.array([e.composer for e in eval_entries])
    eval_seg_ids = [e.seg_id for e in eval_entries]
    eval_seg_id_set = set(eval_seg_ids)
    test_folds = composer_disjoint_folds(eval_composers, n_folds, seed)

    plans = []
    for f, test_idx in enumerate(test_folds):
        test_composers = set(eval_composers[test_idx])
        test_seg_ids = tuple(eval_seg_ids[i] for i in test_idx)
        train_pool = [e for e in pool_entries
                      if e.seg_id not in eval_seg_id_set and e.composer not in test_composers]
        train_seg_ids, val_seg_ids = _carve_val(train_pool, val_frac, seed=seed * 100 + f)
        plans.append(FoldPlan(fold=f, test_seg_ids=test_seg_ids,
                               train_seg_ids=train_seg_ids, val_seg_ids=val_seg_ids))
    return plans


def _carve_val(train_pool, val_frac: float, seed: int):
    """Deterministically carve a ~val_frac slice of train_pool into val, whole
    composers only, so val is composer-disjoint from the remaining train."""
    by_composer: dict = {}
    for e in train_pool:
        by_composer.setdefault(e.composer, []).append(e)
    composers = sorted(by_composer)
    order = np.random.default_rng(seed).permutation(len(composers))
    target = int(round(val_frac * len(train_pool)))

    val_composers, val_count = set(), 0
    for idx in order:
        if val_count >= target:
            break
        c = composers[idx]
        val_composers.add(c)
        val_count += len(by_composer[c])

    train_seg_ids, val_seg_ids = [], []
    for e in train_pool:
        (val_seg_ids if e.composer in val_composers else train_seg_ids).append(e.seg_id)
    return tuple(train_seg_ids), tuple(val_seg_ids)
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_fold_plan.py -q --no-cov
```
Expected: PASS

- [ ] **Step 5: Document the real-data verification (no code; informational only)**

Record in this task's commit message body (not a new doc file) that the exact
production counts are re-derived by running, against the main checkout (which
has the real data; this worktree does not):

```bash
cd model/src/claim_measurement/difficulty && uv run --no-project --script \
    -c "
import sys, json
sys.path.insert(0, '.')
from claim_measurement.difficulty.bakeoff_paths import resolve_paths
from claim_measurement.difficulty.bakeoff_sampling import load_bakeoff_manifest
from claim_measurement.difficulty.fold_plan import build_fold_plans
paths = resolve_paths()
pool = load_bakeoff_manifest(paths.manifest, paths.labels, paths.transkun_mid_dir)
sample = json.loads((paths.emb_root / 'sample_manifest.json').read_text())
sample_ids = {e['seg_id'] for e in sample}
eval_entries = sorted((e for e in pool if e.seg_id in sample_ids), key=lambda e: e.seg_id)
plans = build_fold_plans(eval_entries, pool, 5, 2026, 0.12)
print([len(p.train_seg_ids) + len(p.val_seg_ids) for p in plans])
"
```
Expected (per the design spec's verified facts, not re-derived by pytest):
`[3815, 4082, 4283, 4028, 4149]`. This step is a documented manual check, not
part of the CPU pytest gate (this worktree's `model/data` has no manifest to
run it against).

- [ ] **Step 6: Commit**

```bash
git add model/src/claim_measurement/difficulty/fold_plan.py \
        model/src/claim_measurement/difficulty/test_fold_plan.py \
    && git commit -m "feat(#149): build_fold_plans -- option-D per-fold train/val construction"
```

### Task 3: Val carve is composer-disjoint from train and near the target fraction

**Group:** A (depends on Task 2)

**Behavior being verified:** The ~12% validation slice never shares a composer
with the remaining train pieces, and its size is in a sane range around
`val_frac` (whole-composer carving can't hit the target exactly).

**Interface under test:** `fold_plan.build_fold_plans` (same public function; this
test targets the val-carve behavior specifically)

**Files:**
- Modify: `model/src/claim_measurement/difficulty/test_fold_plan.py`
- (no change to `fold_plan.py` — Task 2's `_carve_val` already implements this;
  this task is the vertical slice that proves it)

- [ ] **Step 1: Write the failing test** (append to `test_fold_plan.py`)

```python
def test_val_carve_is_composer_disjoint_from_train_and_near_target_fraction():
    eval_entries = _entries(n_composers=5, pieces_per_composer=1, prefix="eval_")
    pool_entries = eval_entries + _entries(n_composers=100, pieces_per_composer=4, prefix="pool_")

    plans = build_fold_plans(eval_entries, pool_entries, n_folds=5, seed=2026, val_frac=0.12)

    pool_composer_of = {e.seg_id: e.composer for e in pool_entries}
    for plan in plans:
        train_composers = {pool_composer_of[s] for s in plan.train_seg_ids}
        val_composers = {pool_composer_of[s] for s in plan.val_seg_ids}
        assert not (train_composers & val_composers)
        total = len(plan.train_seg_ids) + len(plan.val_seg_ids)
        frac = len(plan.val_seg_ids) / total
        assert 0.05 < frac < 0.20
```

Given this behavior is already implemented by Task 2's `_carve_val`, this test
is EXPECTED to pass immediately once collected — but must first be verified to
fail against a stub. To keep this a genuine red step, temporarily comment out
the `_carve_val` body's composer-disjoint carving and replace it with a naive
`train_seg_ids, val_seg_ids = tuple(e.seg_id for e in train_pool), ()` (an
obviously wrong stub, val always empty) BEFORE running Step 2, confirm the test
fails against the stub (`0.0 > 0.05` assertion fails), then revert to the real
`_carve_val` from Task 2 and re-run.

- [ ] **Step 2: Run test against the naive stub — verify it FAILS**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_fold_plan.py -q --no-cov -k val_carve
```
Expected: FAIL — `assert 0.0 > 0.05` (empty val set from the stub)

- [ ] **Step 3: Revert to the real `_carve_val`** (already written in Task 2 — no new production code this task)

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_fold_plan.py -q --no-cov
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add model/src/claim_measurement/difficulty/test_fold_plan.py \
    && git commit -m "test(#149): prove build_fold_plans' val carve is composer-disjoint from train"
```

### Task 4: `check_fold_plans` flags a composer that straddles test and train

**Group:** A (depends on Task 3)

**Behavior being verified:** Given a hand-tampered `FoldPlan` where a test-fold
composer's piece was pushed into train, `check_fold_plans` returns a non-empty
violation list naming that fold.

**Interface under test:** `fold_plan.check_fold_plans(plans, eval_entries, pool_entries, n_folds, seed) -> list[str]`

**Files:**
- Modify: `model/src/claim_measurement/difficulty/fold_plan.py`
- Modify: `model/src/claim_measurement/difficulty/test_fold_plan.py`

- [ ] **Step 1: Write the failing test** (append to `test_fold_plan.py`)

```python
def test_check_fold_plans_flags_a_composer_that_straddles_test_and_train():
    eval_entries = _entries(n_composers=5, pieces_per_composer=1, prefix="eval_")
    pool_entries = eval_entries + _entries(n_composers=5, pieces_per_composer=4, prefix="pool_")
    plans = build_fold_plans(eval_entries, pool_entries, n_folds=5, seed=2026, val_frac=0.12)

    fold0 = plans[0]
    test_composer = next(e.composer for e in eval_entries if e.seg_id in fold0.test_seg_ids)
    leaking_seg_id = next(e.seg_id for e in pool_entries if e.composer == test_composer)
    tampered = FoldPlan(fold=0, test_seg_ids=fold0.test_seg_ids,
                        train_seg_ids=fold0.train_seg_ids + (leaking_seg_id,),
                        val_seg_ids=fold0.val_seg_ids)
    tampered_plans = [tampered if p.fold == 0 else p for p in plans]

    violations = check_fold_plans(tampered_plans, eval_entries, pool_entries, n_folds=5, seed=2026)

    assert any("fold 0" in v for v in violations)
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_fold_plan.py -q --no-cov -k check_fold_plans_flags
```
Expected: FAIL — `ImportError: cannot import name 'check_fold_plans'`

- [ ] **Step 3: Implement** (append to `fold_plan.py`)

```python
def check_fold_plans(plans, eval_entries, pool_entries, n_folds: int, seed: int) -> list:
    """Re-derive the expected test folds and return every leakage/consistency
    violation found, as human-readable strings. Empty list == clean."""
    violations: list = []
    eval_composers = np.array([e.composer for e in eval_entries])
    eval_seg_ids = [e.seg_id for e in eval_entries]
    eval_seg_id_set = set(eval_seg_ids)
    composer_of = {e.seg_id: e.composer for e in pool_entries}
    composer_of.update({e.seg_id: e.composer for e in eval_entries})
    expected_test_folds = composer_disjoint_folds(eval_composers, n_folds, seed)

    if len(plans) != n_folds:
        violations.append(f"expected {n_folds} plans, got {len(plans)}")

    for plan in plans:
        expected_test = {eval_seg_ids[i] for i in expected_test_folds[plan.fold]}
        if set(plan.test_seg_ids) != expected_test:
            violations.append(
                f"fold {plan.fold}: test_seg_ids do not equal "
                f"composer_disjoint_folds(eval composers, {n_folds}, {seed})[{plan.fold}]")

        train_set = set(plan.train_seg_ids)
        val_set = set(plan.val_seg_ids)
        test_set = set(plan.test_seg_ids)
        if train_set & test_set:
            violations.append(f"fold {plan.fold}: train/test seg_id overlap")
        if val_set & test_set:
            violations.append(f"fold {plan.fold}: val/test seg_id overlap")
        if train_set & val_set:
            violations.append(f"fold {plan.fold}: train/val seg_id overlap")
        if (train_set | val_set) & eval_seg_id_set:
            violations.append(f"fold {plan.fold}: an eval piece leaked into train or val")

        test_composers = {composer_of[s] for s in plan.test_seg_ids}
        train_composers = {composer_of[s] for s in plan.train_seg_ids}
        val_composers = {composer_of[s] for s in plan.val_seg_ids}
        if test_composers & train_composers:
            violations.append(f"fold {plan.fold}: a test composer appears in train")
        if val_composers & train_composers:
            violations.append(f"fold {plan.fold}: a val composer appears in train")

    return violations
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_fold_plan.py -q --no-cov
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add model/src/claim_measurement/difficulty/fold_plan.py \
        model/src/claim_measurement/difficulty/test_fold_plan.py \
    && git commit -m "feat(#149): check_fold_plans -- detect composer/seg_id leakage across folds"
```

### Task 5: `check_fold_plans` returns empty for plans `build_fold_plans` itself produced

**Group:** A (depends on Task 4, last task in this group)

**Behavior being verified:** No false positives — a clean set of plans passes
its own checker.

**Interface under test:** `fold_plan.check_fold_plans`

**Files:**
- Modify: `model/src/claim_measurement/difficulty/test_fold_plan.py`

- [ ] **Step 1: Write the failing test** (append to `test_fold_plan.py`)

```python
def test_check_fold_plans_returns_empty_for_plans_build_fold_plans_produced():
    eval_entries = _entries(n_composers=10, pieces_per_composer=1, prefix="eval_")
    pool_entries = eval_entries + _entries(n_composers=30, pieces_per_composer=4, prefix="pool_")
    plans = build_fold_plans(eval_entries, pool_entries, n_folds=5, seed=2026, val_frac=0.12)

    violations = check_fold_plans(plans, eval_entries, pool_entries, n_folds=5, seed=2026)

    assert violations == []
```

This test is not expected to require new production code (Task 4's
`check_fold_plans` and Task 2/3's `build_fold_plans` already satisfy it) — its
purpose is to catch a false-positive regression in `check_fold_plans` before
`train_fold.py` (Group C) starts trusting it.

- [ ] **Step 2: Run test — verify it currently PASSES** (this IS the expected
  behavior of already-written code; run it to confirm no false positive exists)

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_fold_plan.py -q --no-cov
```
Expected: PASS, all fold_plan tests green (4 tests total for this module).

- [ ] **Step 3: No implementation change needed.**

- [ ] **Step 4: Re-run the full suite**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
```
Expected: PASS, 47 passed (43 after Task 1 + 4 from fold_plan.py).

- [ ] **Step 5: Commit**

```bash
git add model/src/claim_measurement/difficulty/test_fold_plan.py \
    && git commit -m "test(#149): check_fold_plans has no false positives on build_fold_plans' own output"
```

---

## Group B — `ranking_loss.py` (sequential internally; parallel with Group A)

Pure `torch` (real, CPU, already in `model/.venv` — verified this session).
Factored out of the GPU training script precisely so this suite can reach it.
Five tasks, one file, run in order 6 → 7 → 8 → 9 → 10.

### Task 6: `ordered_pairs` finds every strictly grade-ordered index pair

**Group:** B (first)

**Behavior being verified:** All `(i, j)` pairs where `grades[i] > grades[j]`
are found, none where grades tie.

**Interface under test:** `ranking_loss.ordered_pairs(grades: torch.Tensor) -> torch.Tensor`

**Files:**
- Create: `model/src/claim_measurement/difficulty/ranking_loss.py`
- Create: `model/src/claim_measurement/difficulty/test_ranking_loss.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for ranking_loss (#149 / #138 Phase 1) -- pairwise ranking + ordinal
auxiliary loss, real torch on CPU, no mocks.

Run: cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
"""
import torch

from claim_measurement.difficulty.ranking_loss import ordered_pairs


def test_ordered_pairs_finds_all_strictly_grade_ordered_index_pairs():
    grades = torch.tensor([3, 1, 3, 2])

    pairs = ordered_pairs(grades)

    pair_set = {tuple(p.tolist()) for p in pairs}
    assert pair_set == {(0, 1), (0, 3), (2, 1), (2, 3), (3, 1)}
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_ranking_loss.py -q --no-cov
```
Expected: FAIL — `ModuleNotFoundError: No module named 'claim_measurement.difficulty.ranking_loss'`

- [ ] **Step 3: Implement**

```python
"""Pairwise ranking + cumulative-link ordinal auxiliary loss for #138 Phase 1
LoRA training. Pure torch, CPU-testable -- factored out of train_fold.py
precisely so this offline suite can reach it without a GPU (see design spec's
Modules section for the rationale: the gate metric is Kendall tau-c, a rank
correlation, so pairwise ranking is the primary objective; a low-weight
ordinal auxiliary only pins the score scale, which pure pairwise loss does
not constrain).
"""
from __future__ import annotations

import torch


def ordered_pairs(grades: torch.Tensor) -> torch.Tensor:
    """All (i, j) index pairs within one batch where grades[i] > grades[j].
    Returns an (n_pairs, 2) int64 tensor; shape (0, 2) when no such pair
    exists (e.g. every piece in the batch shares one grade), never raises."""
    n = grades.shape[0]
    gi = grades.unsqueeze(1).expand(n, n)
    gj = grades.unsqueeze(0).expand(n, n)
    mask = gi > gj
    return mask.nonzero(as_tuple=False)
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_ranking_loss.py -q --no-cov
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add model/src/claim_measurement/difficulty/ranking_loss.py \
        model/src/claim_measurement/difficulty/test_ranking_loss.py \
    && git commit -m "feat(#149): ranking_loss.ordered_pairs -- strictly grade-ordered index pairs"
```

### Task 7: `pairwise_ranking_loss` is lower for correctly-ranked scores than reverse-ranked

**Group:** B (depends on Task 6)

**Behavior being verified:** The logistic ranking loss rewards scores that
agree with the grade order and penalizes scores that disagree.

**Interface under test:** `ranking_loss.pairwise_ranking_loss(scores, grades) -> torch.Tensor`

**Files:**
- Modify: `model/src/claim_measurement/difficulty/ranking_loss.py`
- Modify: `model/src/claim_measurement/difficulty/test_ranking_loss.py`

- [ ] **Step 1: Write the failing test** (append to `test_ranking_loss.py`)

```python
from claim_measurement.difficulty.ranking_loss import pairwise_ranking_loss


def test_pairwise_ranking_loss_is_lower_for_correctly_ranked_scores():
    grades = torch.tensor([1, 2, 3])
    correct_scores = torch.tensor([0.1, 0.5, 0.9], requires_grad=True)
    reversed_scores = torch.tensor([0.9, 0.5, 0.1], requires_grad=True)

    correct_loss = pairwise_ranking_loss(correct_scores, grades)
    reversed_loss = pairwise_ranking_loss(reversed_scores, grades)

    assert correct_loss.item() < reversed_loss.item()
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_ranking_loss.py -q --no-cov -k pairwise_ranking_loss_is_lower
```
Expected: FAIL — `ImportError: cannot import name 'pairwise_ranking_loss'`

- [ ] **Step 3: Implement** (append to `ranking_loss.py`)

```python
def pairwise_ranking_loss(scores: torch.Tensor, grades: torch.Tensor) -> torch.Tensor:
    """Pairwise logistic ranking loss: -log(sigmoid(score_i - score_j)) for
    every strictly grade-ordered pair (i higher-graded than j). Returns a
    finite 0.0 still attached to `scores`' autograd graph when the batch has
    zero ordered pairs (e.g. every piece shares one grade) rather than NaN
    from averaging an empty tensor -- see Task 8."""
    pairs = ordered_pairs(grades)
    if pairs.shape[0] == 0:
        return scores.sum() * 0.0
    hi, lo = pairs[:, 0], pairs[:, 1]
    return torch.nn.functional.softplus(-(scores[hi] - scores[lo])).mean()
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_ranking_loss.py -q --no-cov
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add model/src/claim_measurement/difficulty/ranking_loss.py \
        model/src/claim_measurement/difficulty/test_ranking_loss.py \
    && git commit -m "feat(#149): ranking_loss.pairwise_ranking_loss -- logistic ranking loss"
```

### Task 8: `pairwise_ranking_loss` is a finite zero, still attached to the autograd graph, for a degenerate batch

**Group:** B (depends on Task 7)

**Behavior being verified:** A micro-batch whose pieces all share one grade
must not NaN or crash `.backward()` — this is a real training case, not a
hypothetical (see design spec).

**Interface under test:** `ranking_loss.pairwise_ranking_loss`

**Files:**
- Modify: `model/src/claim_measurement/difficulty/test_ranking_loss.py`

- [ ] **Step 1: Write the failing test** (append to `test_ranking_loss.py`)

```python
def test_pairwise_ranking_loss_is_a_finite_zero_for_a_degenerate_batch():
    grades = torch.tensor([4, 4, 4])  # every piece shares one grade -> zero ordered pairs
    scores = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)

    loss = pairwise_ranking_loss(scores, grades)

    assert loss.item() == 0.0
    loss.backward()  # must not raise -- the zero is still attached to the graph
    assert scores.grad is not None
```

Given Task 7's implementation already contains the degenerate-batch guard,
this test is expected to already pass on the current code. To keep this a
genuine red step, temporarily replace the guard's `return scores.sum() * 0.0`
with `return torch.tensor(float("nan"))` (a detached, ungraphed stand-in)
before Step 2, confirm the test fails, then restore Task 7's real
implementation.

- [ ] **Step 2: Run test against the broken stand-in — verify it FAILS**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_ranking_loss.py -q --no-cov -k degenerate
```
Expected: FAIL — `RuntimeError: element 0 of tensors does not require grad` (the
detached NaN stand-in breaks `.backward()`)

- [ ] **Step 3: Restore the real implementation** (Task 7's `return scores.sum() * 0.0` — no new code)

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_ranking_loss.py -q --no-cov
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add model/src/claim_measurement/difficulty/test_ranking_loss.py \
    && git commit -m "test(#149): pairwise_ranking_loss is a finite, graph-attached zero on a degenerate batch"
```

### Task 9: `ordinal_loss` penalizes wrong cumulative-link threshold predictions

**Group:** B (depends on Task 8)

**Behavior being verified:** The 10 binary "grade > k" logits (11-level scale)
score lower loss when they match the true grade's cumulative-link pattern than
when they are inverted.

**Interface under test:** `ranking_loss.ordinal_loss(logits, grades, n_levels) -> torch.Tensor`

**Files:**
- Modify: `model/src/claim_measurement/difficulty/ranking_loss.py`
- Modify: `model/src/claim_measurement/difficulty/test_ranking_loss.py`

- [ ] **Step 1: Write the failing test** (append to `test_ranking_loss.py`)

```python
from claim_measurement.difficulty.ranking_loss import ordinal_loss


def test_ordinal_loss_penalizes_wrong_threshold_predictions():
    grades = torch.tensor([0, 10])  # 11-level scale: min and max grade
    n_levels = 11
    correct_logits = torch.stack([torch.full((n_levels - 1,), -10.0),
                                   torch.full((n_levels - 1,), 10.0)])
    wrong_logits = torch.stack([torch.full((n_levels - 1,), 10.0),
                                 torch.full((n_levels - 1,), -10.0)])

    correct_loss = ordinal_loss(correct_logits, grades, n_levels)
    wrong_loss = ordinal_loss(wrong_logits, grades, n_levels)

    assert correct_loss.item() < wrong_loss.item()
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_ranking_loss.py -q --no-cov -k ordinal_loss_penalizes
```
Expected: FAIL — `ImportError: cannot import name 'ordinal_loss'`

- [ ] **Step 3: Implement** (append to `ranking_loss.py`)

```python
def ordinal_loss(logits: torch.Tensor, grades: torch.Tensor, n_levels: int) -> torch.Tensor:
    """Cumulative-link ordinal loss: n_levels - 1 binary "grade > k" targets
    per row, BCE-with-logits against `logits` (shape (batch, n_levels - 1))."""
    thresholds = torch.arange(n_levels - 1, device=grades.device)
    targets = (grades.unsqueeze(1) > thresholds.unsqueeze(0)).float()
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_ranking_loss.py -q --no-cov
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add model/src/claim_measurement/difficulty/ranking_loss.py \
        model/src/claim_measurement/difficulty/test_ranking_loss.py \
    && git commit -m "feat(#149): ranking_loss.ordinal_loss -- cumulative-link auxiliary"
```

### Task 10: `combined_loss` sums pairwise ranking loss and weighted ordinal loss

**Group:** B (depends on Task 9, last task in this group)

**Behavior being verified:** `combined_loss` wires the two losses together with
the stated weight — this is the exact loss `train_fold.py` (Group C) will call.

**Interface under test:** `ranking_loss.combined_loss(scores, ordinal_logits, grades, n_levels, ordinal_weight) -> torch.Tensor`

**Files:**
- Modify: `model/src/claim_measurement/difficulty/ranking_loss.py`
- Modify: `model/src/claim_measurement/difficulty/test_ranking_loss.py`

- [ ] **Step 1: Write the failing test** (append to `test_ranking_loss.py`)

```python
import pytest

from claim_measurement.difficulty.ranking_loss import combined_loss


def test_combined_loss_equals_pairwise_plus_weighted_ordinal():
    grades = torch.tensor([1, 3])
    scores = torch.tensor([0.2, 0.8])
    n_levels = 11
    ordinal_logits = torch.zeros((2, n_levels - 1))
    weight = 0.1

    combined = combined_loss(scores, ordinal_logits, grades, n_levels, ordinal_weight=weight)
    expected = (pairwise_ranking_loss(scores, grades)
                + weight * ordinal_loss(ordinal_logits, grades, n_levels))

    assert combined.item() == pytest.approx(expected.item())
```

Add `pairwise_ranking_loss` and `ordinal_loss` to this test file's existing
import lines if not already present (they were added in Tasks 7 and 9).

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_ranking_loss.py -q --no-cov -k combined_loss
```
Expected: FAIL — `ImportError: cannot import name 'combined_loss'`

- [ ] **Step 3: Implement** (append to `ranking_loss.py`)

```python
def combined_loss(scores: torch.Tensor, ordinal_logits: torch.Tensor, grades: torch.Tensor,
                   n_levels: int, ordinal_weight: float) -> torch.Tensor:
    """The training objective train_fold.py optimizes: pairwise ranking loss
    (primary, matches the tau-c gate metric) plus a low-weight ordinal
    auxiliary (keeps the score scale from drifting freely)."""
    return (pairwise_ranking_loss(scores, grades)
            + ordinal_weight * ordinal_loss(ordinal_logits, grades, n_levels))
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
```
Expected: PASS, 52 passed (47 after Group A + 5 from ranking_loss.py).

- [ ] **Step 5: Commit**

```bash
git add model/src/claim_measurement/difficulty/ranking_loss.py \
        model/src/claim_measurement/difficulty/test_ranking_loss.py \
    && git commit -m "feat(#149): ranking_loss.combined_loss -- the #138 Phase 1 training objective"
```

---

## Group C — `train_fold.py` (sequential internally; depends on Groups 0 and B)

Standalone `# /// script` HF Jobs entry point, mirroring `moonbeam_extract_script.py`
exactly: heavy fork-specific imports (mido, music21, the vendored transformers)
are deferred inside `_real_loader`, never at module scope, so this file imports
cleanly under `model/.venv` (which already has torch 2.9.0 and peft 0.18.1) without
needing the isolated MoonBeam venv. `_real_loader` itself is never exercised by
the pytest suite — like `moonbeam_extract_script.py`'s own `_real_loader`, it is
verified by documentation-level correctness and the pilot HF Job, not CPU tests.
Both tasks build one file; run in order 11 → 12.

### Task 11: `lora_target_modules` returns the exact 35 target modules for the top 5 of 15 layers

**Group:** C (first)

**Behavior being verified:** Given `n_layers=15, n_top=5`, the function returns
`self_attn.{q,k,v,o}_proj` and `mlp.{gate,up,down}_proj` for layers 10-14 only
(35 modules total), and never touches `lm_head`/`decoder_embedding`/
`summary_projection`/`fc_out` (the fork's DEFAULT target modules, explicitly
excluded per the design spec).

**Interface under test:** `train_fold.lora_target_modules(n_layers: int, n_top: int) -> list[str]`

**Files:**
- Create: `model/src/claim_measurement/difficulty/train_fold.py`
- Create: `model/src/claim_measurement/difficulty/test_train_fold.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for train_fold (#149 / #138 Phase 1) -- LoRA target modules and the
CLI wiring of a full fine-tune epoch, via an injected fake loader_factory
(the pattern moonbeam_extract_script.py already establishes). No mocks of
internal collaborators; real torch and real peft (both in model/.venv).

Run: cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
"""
from claim_measurement.difficulty.train_fold import lora_target_modules


def test_lora_target_modules_targets_top_5_of_15_layers_35_modules():
    modules = lora_target_modules(n_layers=15, n_top=5)

    assert len(modules) == 35
    assert modules[:7] == [
        "model.layers.10.self_attn.q_proj", "model.layers.10.self_attn.k_proj",
        "model.layers.10.self_attn.v_proj", "model.layers.10.self_attn.o_proj",
        "model.layers.10.mlp.gate_proj", "model.layers.10.mlp.up_proj",
        "model.layers.10.mlp.down_proj",
    ]
    assert {int(m.split(".")[2]) for m in modules} == {10, 11, 12, 13, 14}
    excluded = {"decoder_embedding", "summary_projection", "lm_head", "fc_out"}
    assert not any(any(x in m for x in excluded) for m in modules)
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_train_fold.py -q --no-cov
```
Expected: FAIL — `ModuleNotFoundError: No module named 'claim_measurement.difficulty.train_fold'`

- [ ] **Step 3: Implement** (this is the start of `train_fold.py` — only the
  header, `lora_target_modules`, and the module docstring; the rest of the file
  is built in Task 12)

```python
# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "numpy>=1.24.0", "torch>=2.0.0", "peft>=0.11.0", "trackio",
#     "mido", "music21", "pandas", "tqdm", "regex", "requests",
#     "filelock", "pyyaml", "safetensors", "tokenizers==0.19.1",
#     "huggingface_hub",
# ]
# ///
"""#138 Phase 1 LoRA fine-tune of MoonBeam-839M, one fold at a time. HF Jobs
entry point -- run under the SAME isolated uv-managed Python 3.12 venv as
moonbeam_extract_script.py (see that file's module docstring for the fork
clone/checkpoint setup). This file's own `# /// script` header restates torch
+ peft (already pinned in model/.venv, restated here because HF Jobs builds a
FRESH environment from this header, never model/.venv) plus the same
transformers_minimal-fork transitive deps moonbeam_extract_script.py needs,
plus trackio for telemetry.

    hf jobs uv run --flavor a100-large train_fold.py \\
        --fold 0 --checkpoint .../moonbeam_839M.pt --repo-root .../repo \\
        --model-config .../model_config.json \\
        --fold-plan .../fold_plans.json --pool-grades .../grades.json \\
        --eval-manifest .../eval_manifest.json \\
        --midi-dir .../transkun_mid --out-dir .../fold0

Only the encoder weights are graded (design spec's gate (i)): the score head
trained here is DISCARDED after training. `emb_fold{F}.npz` -- the only
artifact ft_eval.py reads -- holds MEAN-POOLED embeddings for ALL 900 eval
pieces (not just this fold's 180), extracted with the SAME full-piece,
no-window forward pass moonbeam_extract_script.py uses, so the gate stays
paired against frozen 0.8257. Training itself samples one random 1024-token
window per piece per step (a deliberate crop augmentation -- see the design
spec's "Train-time vs extract-time windowing"); only extraction is
window-free.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECTIONS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
               "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def lora_target_modules(n_layers: int, n_top: int) -> list[str]:
    """The LoRA target module names for the top n_top of n_layers MoonBeam
    decoder layers: self_attn.{q,k,v,o}_proj and mlp.{gate,up,down}_proj per
    layer, on checkpoint-matching names `model.layers.{L}.{...}`. Explicitly
    excludes decoder_embedding/summary_projection/lm_head/fc_out -- the
    fork's DEFAULT target_modules (src/llama_recipes/configs/peft.py:11),
    which target the generative decoder heads this design never invokes."""
    if n_top > n_layers:
        raise ValueError(f"n_top ({n_top}) cannot exceed n_layers ({n_layers})")
    return [f"model.layers.{layer}.{proj}"
            for layer in range(n_layers - n_top, n_layers) for proj in PROJECTIONS]


if __name__ == "__main__":
    sys.exit(0)  # placeholder exit; main() is added in Task 12
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_train_fold.py -q --no-cov
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add model/src/claim_measurement/difficulty/train_fold.py \
        model/src/claim_measurement/difficulty/test_train_fold.py \
    && git commit -m "feat(#149): train_fold.lora_target_modules -- top-5-of-15-layer LoRA targets"
```

### Task 12: `main()` trains a LoRA adapter via an injected fake loader and writes `emb_fold{F}.npz` for all eval pieces

**Group:** C (depends on Task 11, last task in this group)

**Behavior being verified:** Given an injected fake `loader_factory` (never the
real MoonBeam fork or checkpoint) returning a tiny CPU model whose submodule
names match `lora_target_modules`'s convention, `main()` PEFT-wraps it, runs
one epoch of pairwise-ranking + ordinal training on the fold's train pieces,
saves an adapter, and writes `emb_fold{F}.npz` holding full-piece mean-pooled
embeddings for every eval piece (not just this fold's test pieces).

This plumbing (peft's `get_peft_model` mutating an outer wrapper object
in-place so `peft_model.model.model(...)` still calls through the injected
LoRA layers) was validated directly against the real `torch`/`peft` in
`model/.venv` before writing this task — the exact fake-model shape below
was run end-to-end (forward, backward, `save_pretrained`) and confirmed
working.

**Interface under test:** `train_fold.main(argv, loader_factory=...) -> int`,
`train_fold.write_fold_embeddings`, `train_fold.read_fold_embeddings`

**Files:**
- Modify: `model/src/claim_measurement/difficulty/train_fold.py`
- Modify: `model/src/claim_measurement/difficulty/test_train_fold.py`

- [ ] **Step 1: Write the failing test** (append to `test_train_fold.py`)

```python
import json

import numpy as np
import torch

from claim_measurement.difficulty.train_fold import main, read_fold_embeddings


class _FakeLayer(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.self_attn = torch.nn.Module()
        self.self_attn.q_proj = torch.nn.Linear(hidden, hidden, bias=False)
        self.self_attn.k_proj = torch.nn.Linear(hidden, hidden, bias=False)
        self.self_attn.v_proj = torch.nn.Linear(hidden, hidden, bias=False)
        self.self_attn.o_proj = torch.nn.Linear(hidden, hidden, bias=False)
        self.mlp = torch.nn.Module()
        self.mlp.gate_proj = torch.nn.Linear(hidden, hidden, bias=False)
        self.mlp.up_proj = torch.nn.Linear(hidden, hidden, bias=False)
        self.mlp.down_proj = torch.nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        h = x + self.self_attn.o_proj(
            self.self_attn.q_proj(x) + self.self_attn.k_proj(x) + self.self_attn.v_proj(x))
        h = h + self.mlp.down_proj(torch.relu(self.mlp.gate_proj(h)) * self.mlp.up_proj(h))
        return h


class _FakeInner(torch.nn.Module):
    def __init__(self, hidden, n_layers, vocab):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab, hidden)
        self.layers = torch.nn.ModuleList([_FakeLayer(hidden) for _ in range(n_layers)])

    def forward(self, input_ids, position_ids=None, use_cache=False, return_dict=True):
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h)

        class Out:
            pass

        out = Out()
        out.last_hidden_state = h
        return out


class _FakeOuter(torch.nn.Module):
    """Mimics LlamaForCausalLM: a .model attribute (inner transformer) plus an
    lm_head this design never calls and never LoRA-targets."""

    def __init__(self, hidden=4, n_layers=1, vocab=16):
        super().__init__()
        self.model = _FakeInner(hidden, n_layers, vocab)
        self.lm_head = torch.nn.Linear(hidden, vocab, bias=False)


_TOKEN_LENGTHS = {"t0": 6, "t1": 7, "t2": 8, "t3": 6, "v0": 7, "e0": 5, "e1": 9}


def test_main_trains_a_lora_adapter_and_writes_emb_fold_for_all_eval_pieces(tmp_path):
    fold_plan = [{
        "fold": 0,
        "test_seg_ids": ["e0", "e1"],
        "train_seg_ids": ["t0", "t1", "t2", "t3"],
        "val_seg_ids": ["v0"],
    }]
    (tmp_path / "fold_plan.json").write_text(json.dumps(fold_plan))
    (tmp_path / "pool_grades.json").write_text(json.dumps(
        {"t0": 1, "t1": 5, "t2": 8, "t3": 3, "v0": 4}))
    (tmp_path / "eval_manifest.json").write_text(json.dumps([
        {"seg_id": "e0", "grade": 2, "composer_id": 0},
        {"seg_id": "e1", "grade": 9, "composer_id": 1},
    ]))
    out_dir = tmp_path / "fold0"

    def fake_loader_factory(checkpoint_path, repo_root, model_config):
        outer = _FakeOuter(hidden=4, n_layers=1, vocab=16)

        def tokenize(midi_path):
            n = _TOKEN_LENGTHS[midi_path.stem]
            return torch.arange(n) % 16

        return outer, tokenize

    exit_code = main(
        [
            "--fold", "0",
            "--checkpoint", str(tmp_path / "fake.pt"),
            "--repo-root", str(tmp_path / "repo"),
            "--model-config", str(tmp_path / "repo" / "model_config.json"),
            "--fold-plan", str(tmp_path / "fold_plan.json"),
            "--pool-grades", str(tmp_path / "pool_grades.json"),
            "--eval-manifest", str(tmp_path / "eval_manifest.json"),
            "--midi-dir", str(tmp_path / "mid"),
            "--out-dir", str(out_dir),
            "--hidden-size", "4",
            "--n-layers", "1",
            "--n-top-layers", "1",
            "--max-len", "4",
            "--epochs", "1",
            "--micro-batch", "2",
        ],
        loader_factory=fake_loader_factory,
    )

    assert exit_code == 0
    assert (out_dir / "adapter" / "adapter_config.json").exists()
    fold_data = read_fold_embeddings(out_dir / "emb_fold0.npz")
    assert fold_data["seg_ids"] == ["e0", "e1"]
    assert fold_data["embeddings"].shape == (2, 4)
    assert list(fold_data["grades"]) == [2, 9]
    assert list(fold_data["composer_ids"]) == [0, 1]
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_train_fold.py -q --no-cov -k main_trains
```
Expected: FAIL — `TypeError: main() got an unexpected keyword argument 'loader_factory'`
(current `main` is just the Task-11 placeholder `if __name__ == "__main__": sys.exit(0)`,
no `main()` function exists yet)

- [ ] **Step 3: Implement** (replace the `if __name__ == "__main__":` placeholder
  block at the bottom of `train_fold.py` with the full implementation below)

```python
def _real_loader(checkpoint_path: Path, repo_root: Path, model_config: Path):
    """Build the real trainable MoonBeam model + tokenizer against the fork.
    Mirrors moonbeam_extract_script.py::_real_loader's checkpoint/tokenizer
    setup exactly (see that file for the three undocumented fork facts), but
    returns the OUTER LlamaForCausalLM itself (gradients flow; never called
    under torch.no_grad) plus a `tokenize(midi_path) -> LongTensor` callable,
    rather than a numpy-returning inference closure."""
    import importlib.util

    repo_root = Path(repo_root)
    sys.path.insert(0, str(repo_root / "src" / "llama_recipes" / "transformers_minimal" / "src"))
    from transformers import LlamaConfig, LlamaForCausalLM

    spec = importlib.util.spec_from_file_location(
        "moonbeam_music_tokenizer",
        repo_root / "src" / "llama_recipes" / "datasets" / "music_tokenizer.py")
    music_tokenizer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(music_tokenizer)

    config = LlamaConfig.from_pretrained(model_config)
    if config._attn_implementation != "sdpa":
        raise ValueError(
            f"expected attn_implementation 'sdpa', got {config._attn_implementation!r}")
    model = LlamaForCausalLM(config)

    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = {(k[7:] if k.startswith("module.") else k): v
             for k, v in raw["model_state_dict"].items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"checkpoint does not match {model_config}: {len(missing)} missing / "
            f"{len(unexpected)} unexpected keys "
            f"(first missing: {missing[:5]}, first unexpected: {unexpected[:5]})")

    tokenizer = music_tokenizer.MusicTokenizer(
        timeshift_vocab_size=config.onset_vocab_size, dur_vocab_size=config.dur_vocab_size,
        octave_vocab_size=config.octave_vocab_size,
        pitch_class_vocab_size=config.pitch_class_vocab_size,
        instrument_vocab_size=config.instrument_vocab_size,
        velocity_vocab_size=config.velocity_vocab_size)

    def tokenize(midi_path: Path) -> torch.Tensor:
        compounds = tokenizer.midi_to_compound(str(midi_path))
        tokens = tokenizer.encode_series(compounds, if_add_sos=True, if_add_eos=True)
        return torch.tensor(tokens, dtype=torch.long)

    return model, tokenize


def _score_head(hidden_size: int, n_levels: int) -> torch.nn.Module:
    """The trained-then-DISCARDED head: one linear layer producing a scalar
    ranking score plus n_levels-1 ordinal logits from a mean-pooled embedding."""
    return torch.nn.Linear(hidden_size, 1 + (n_levels - 1))


def _random_window(tokens: torch.Tensor, max_len: int, rng: np.random.Generator) -> torch.Tensor:
    """One random contiguous max_len-token window (the whole sequence if it
    is already <= max_len). A deliberate crop augmentation at train time --
    see the design spec's "Train-time vs extract-time windowing"."""
    if len(tokens) <= max_len:
        return tokens
    start = int(rng.integers(0, len(tokens) - max_len + 1))
    return tokens[start:start + max_len]


def _mean_pool_window(transformer: torch.nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    x = tokens.unsqueeze(0)
    hidden = transformer(input_ids=x, position_ids=x, use_cache=False,
                          return_dict=True).last_hidden_state.squeeze(0)
    return hidden.mean(dim=0)


def _extract_full_piece(transformer: torch.nn.Module, tokens: torch.Tensor, max_len: int) -> np.ndarray:
    """Byte-identical extraction to moonbeam_extract_script.py: chunk to
    max_len, forward every chunk, concatenate, mean over ALL tokens -- so the
    gate stays paired against frozen 0.8257."""
    chunks = [tokens[i:i + max_len] for i in range(0, len(tokens), max_len)]
    with torch.no_grad():
        hidden = [transformer(input_ids=c.unsqueeze(0), position_ids=c.unsqueeze(0),
                               use_cache=False, return_dict=True).last_hidden_state.squeeze(0)
                  for c in chunks]
    return torch.cat(hidden, dim=0).mean(dim=0).float().numpy()


def write_fold_embeddings(path: Path, seg_ids: list[str], embeddings: np.ndarray,
                           grades: np.ndarray, composer_ids: np.ndarray) -> None:
    """emb_fold{F}.npz: one bulk array file for ALL eval pieces (NOT the
    per-piece bakeoff_npz contract -- ft_eval.py needs one (900, hidden)
    matrix per fold, not 900 files per fold)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, seg_ids=np.array(seg_ids), embeddings=embeddings.astype(np.float32),
              grades=np.asarray(grades, dtype=np.int32),
              composer_ids=np.asarray(composer_ids, dtype=np.int32))


def read_fold_embeddings(path: Path) -> dict:
    with np.load(path) as z:
        return {
            "seg_ids": [str(s) for s in z["seg_ids"]],
            "embeddings": z["embeddings"],
            "grades": z["grades"],
            "composer_ids": z["composer_ids"],
        }


def main(argv: list[str] | None = None, loader_factory=_real_loader) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # -> model/src
    from claim_measurement.difficulty.bakeoff_cv import tau_c
    from claim_measurement.difficulty.ranking_loss import combined_loss

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--model-config", type=Path, required=True)
    ap.add_argument("--fold-plan", type=Path, required=True)
    ap.add_argument("--pool-grades", type=Path, required=True,
                    help="JSON {seg_id: grade} covering every train/val seg_id in --fold-plan")
    ap.add_argument("--eval-manifest", type=Path, required=True,
                    help="JSON list of {seg_id, grade, composer_id} for all 900 eval pieces, "
                         "in the SAME seg_id-sorted order ft_eval.py reads from emb/features37/")
    ap.add_argument("--midi-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--hidden-size", type=int, default=1920)
    ap.add_argument("--n-layers", type=int, default=15)
    ap.add_argument("--n-top-layers", type=int, default=5)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ordinal-weight", type=float, default=0.1)
    ap.add_argument("--n-levels", type=int, default=11)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args(argv)

    plans = json.loads(args.fold_plan.read_text())
    plan = next(p for p in plans if p["fold"] == args.fold)
    pool_grades = json.loads(args.pool_grades.read_text())
    eval_pieces = json.loads(args.eval_manifest.read_text())

    base_model, tokenize = loader_factory(args.checkpoint, repo_root=args.repo_root,
                                           model_config=args.model_config)

    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                              target_modules=lora_target_modules(args.n_layers, args.n_top_layers))
    peft_model = get_peft_model(base_model, lora_config)
    transformer = peft_model.model.model  # inner transformer, LoRA-injected in place

    score_head = _score_head(args.hidden_size, args.n_levels)
    trainable_params = [p for p in peft_model.parameters() if p.requires_grad] + list(score_head.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    rng = np.random.default_rng(args.seed)
    train_seg_ids = list(plan["train_seg_ids"])
    val_seg_ids = list(plan["val_seg_ids"])

    for epoch in range(args.epochs):
        order = rng.permutation(len(train_seg_ids))
        for start in range(0, len(order), args.micro_batch):
            batch_ids = [train_seg_ids[i] for i in order[start:start + args.micro_batch]]
            scores, ordinal_logits, grades = [], [], []
            for seg_id in batch_ids:
                tokens = tokenize(Path(args.midi_dir) / f"{seg_id}.mid")
                window = _random_window(tokens, args.max_len, rng)
                pooled = _mean_pool_window(transformer, window)
                head_out = score_head(pooled)
                scores.append(head_out[0])
                ordinal_logits.append(head_out[1:])
                grades.append(pool_grades[seg_id])
            scores_t = torch.stack(scores)
            ordinal_t = torch.stack(ordinal_logits)
            grades_t = torch.tensor(grades, dtype=torch.long)

            loss = combined_loss(scores_t, ordinal_t, grades_t, args.n_levels, args.ordinal_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            val_scores = []
            for seg_id in val_seg_ids:
                tokens = tokenize(Path(args.midi_dir) / f"{seg_id}.mid")
                window = _random_window(tokens, args.max_len, rng)
                pooled = _mean_pool_window(transformer, window)
                val_scores.append(score_head(pooled)[0].item())
            val_grades = [pool_grades[seg_id] for seg_id in val_seg_ids]
            val_tau = tau_c(val_scores, val_grades) if val_seg_ids else None
        print(f"epoch {epoch}: val_ranking_tau={val_tau}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(out_dir / "adapter"))

    with torch.no_grad():
        embeddings = np.stack([
            _extract_full_piece(transformer, tokenize(Path(args.midi_dir) / f"{p['seg_id']}.mid"),
                                 args.max_len)
            for p in eval_pieces
        ])
    write_fold_embeddings(
        out_dir / f"emb_fold{args.fold}.npz",
        seg_ids=[p["seg_id"] for p in eval_pieces],
        embeddings=embeddings,
        grades=np.array([p["grade"] for p in eval_pieces]),
        composer_ids=np.array([p["composer_id"] for p in eval_pieces]),
    )
    print(f"fold {args.fold}: wrote adapter + emb_fold{args.fold}.npz for {len(eval_pieces)} eval pieces")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
```
Expected: PASS, 54 passed (52 after Group B + 2 from train_fold.py).

- [ ] **Step 5: Commit**

```bash
git add model/src/claim_measurement/difficulty/train_fold.py \
        model/src/claim_measurement/difficulty/test_train_fold.py \
    && git commit -m "feat(#149): train_fold.main -- LoRA fine-tune loop + emb_fold{F}.npz emission"
```

---

## Group D — `ft_eval.py` (sequential internally; depends on Group C)

Gate (i): discard every fold's trained head, score `emb_fold{F}.npz`'s
mean-pooled embeddings with RidgeCV through `bakeoff_cv.py`'s own
composer-disjoint folds. The one new statistical primitive is
`oof_tau_per_fold`: ordinary OOF holds one `X` and varies the fold, but here
`X` itself differs per fold, since each fold has its own adapter. Three tasks,
one file, run in order 13 → 14 → 15.

### Task 13: `oof_tau_per_fold` recovers a strong signal that is only visible when each fold uses ITS OWN embedding matrix

**Group:** D (first)

**Behavior being verified:** For fold `f`, both the ridge head's train rows and
test rows come from `emb_by_fold[f]` — never mixed across folds. The test
constructs per-fold matrices where the linear signal's SCALE differs by fold
(`X = y * (f + 1)`), so a correct implementation (using fold f's own matrix for
fold f) recovers tau-c near 1, while an implementation that mixed rows across
adapters would not.

**Interface under test:** `ft_eval.oof_tau_per_fold(emb_by_fold, y, composers, n_folds, seed) -> np.ndarray`

**Files:**
- Create: `model/src/claim_measurement/difficulty/ft_eval.py`
- Create: `model/src/claim_measurement/difficulty/test_ft_eval.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for ft_eval (#149 / #138 Phase 1) -- the gate: OOF where X differs
per fold, plus the CLI wiring against features37 + emb_fold{F}.npz files.

Run: cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
"""
import numpy as np
import pytest

from claim_measurement.difficulty.bakeoff_cv import tau_c
from claim_measurement.difficulty.ft_eval import oof_tau_per_fold


def test_oof_tau_per_fold_recovers_a_strong_per_fold_linear_signal():
    rng = np.random.default_rng(2026)
    n = 200
    composers = np.array([f"composer_{i}" for i in range(n)])  # all distinct -> vacuous disjointness
    y = rng.integers(0, 11, size=n).astype(float)

    emb_by_fold = {}
    for f in range(5):
        rng_f = np.random.default_rng(1000 + f)
        noise = rng_f.normal(size=(n, 3)) * 0.01
        emb_by_fold[f] = np.column_stack([y * (f + 1), noise])

    oof = oof_tau_per_fold(emb_by_fold, y, composers, n_folds=5, seed=2026)

    assert not np.isnan(oof).any()
    assert tau_c(oof, y) > 0.9
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_ft_eval.py -q --no-cov
```
Expected: FAIL — `ModuleNotFoundError: No module named 'claim_measurement.difficulty.ft_eval'`

- [ ] **Step 3: Implement** (this is the start of `ft_eval.py`)

```python
"""#138 Phase 1 gate (i): encoder-as-feature-extractor. Discards every fold's
trained head, scores fold f's mean-pooled emb_fold{f}.npz embeddings with
RidgeCV through bakeoff_cv.py's OWN composer-disjoint folds, and reports the
paired-bootstrap delta against features37|ridge on the SAME folds.

    cd model && uv run python -m claim_measurement.difficulty.ft_eval \\
        --data-root /path/to/model/data --fold-emb-dir /path/to/fold_embeddings

Per-fold X differs (each fold has its own adapter), which is why this needs
oof_tau_per_fold rather than bakeoff_cv.oof_tau_ridge -- see that function's
docstring. Seed is FIXED at 2026 (not averaged over multiple seeds like the
Phase 0 comparison): a set of per-fold adapters is welded to the (n_folds,
seed) pair that produced their training pools -- see the design spec.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from claim_measurement.difficulty.bakeoff_cv import composer_disjoint_folds

N_FOLDS, SEED = 5, 2026
ALPHAS = np.logspace(-1, 5, 25)


def oof_tau_per_fold(emb_by_fold: dict, y: np.ndarray, composers: np.ndarray,
                      n_folds: int, seed: int) -> np.ndarray:
    """OOF predictions where X differs per fold: for fold f, BOTH the ridge
    head's train rows and its test rows come from emb_by_fold[f] -- the
    embeddings extracted by fold f's own adapter. Mixing rows across adapters
    would score a head fit on one encoder against another encoder's
    features."""
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    oof = np.full(len(y), np.nan)
    for f, test_idx in enumerate(composer_disjoint_folds(composers, n_folds, seed)):
        if f not in emb_by_fold:
            raise KeyError(f"emb_by_fold is missing fold {f}")
        X = emb_by_fold[f]
        train_idx = np.setdiff1d(np.arange(len(y)), test_idx)
        if len(train_idx) < 3 or len(test_idx) == 0:
            continue
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
        model.fit(np.nan_to_num(X[train_idx]), y[train_idx])
        oof[test_idx] = model.predict(np.nan_to_num(X[test_idx]))
    return oof


if __name__ == "__main__":
    sys.exit(0)  # placeholder exit; main() is added in Task 15
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_ft_eval.py -q --no-cov
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add model/src/claim_measurement/difficulty/ft_eval.py \
        model/src/claim_measurement/difficulty/test_ft_eval.py \
    && git commit -m "feat(#149): ft_eval.oof_tau_per_fold -- OOF where X differs per fold"
```

### Task 14: `oof_tau_per_fold` raises when a fold's embeddings are missing

**Group:** D (depends on Task 13)

**Behavior being verified:** A missing `emb_by_fold[f]` key must raise loudly,
never silently produce partial/NaN OOF rows that a downstream `tau_c` call
would quietly drop.

**Interface under test:** `ft_eval.oof_tau_per_fold`

**Files:**
- Modify: `model/src/claim_measurement/difficulty/test_ft_eval.py`

- [ ] **Step 1: Write the failing test** (append to `test_ft_eval.py`)

```python
def test_oof_tau_per_fold_raises_on_missing_fold_embeddings():
    composers = np.array([f"composer_{i}" for i in range(50)])
    y = np.arange(50, dtype=float) % 11
    emb_by_fold = {0: np.random.default_rng(0).normal(size=(50, 2))}  # folds 1-4 missing

    with pytest.raises(KeyError):
        oof_tau_per_fold(emb_by_fold, y, composers, n_folds=5, seed=2026)
```

- [ ] **Step 2: Run test — verify it currently PASSES** (Task 13's `if f not in
  emb_by_fold: raise KeyError(...)` guard already implements this; run to
  confirm no regression risk before Group C/D wiring proceeds)

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_ft_eval.py -q --no-cov
```
Expected: PASS (2 tests)

- [ ] **Step 3: No implementation change needed.**

- [ ] **Step 4: Re-run to confirm**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
```
Expected: PASS, 56 passed.

- [ ] **Step 5: Commit**

```bash
git add model/src/claim_measurement/difficulty/test_ft_eval.py \
    && git commit -m "test(#149): oof_tau_per_fold raises loudly on a missing fold's embeddings"
```

### Task 15: `main()` prints the gate comparison against features37 on the same folds

**Group:** D (depends on Task 14, last task in this group)

**Behavior being verified:** Given features37 `.npz` files (the existing
per-piece contract) and `emb_fold{F}.npz` files (Group C's output, row-aligned
by seg_id), `main()` prints `moonbeam_ft_mean|ridge - features37|ridge: ...`
with a `SIG`/`noise` verdict, using the SAME `paired_boot` promoted in Task 1.

**Interface under test:** `ft_eval.main(argv) -> int`

**Files:**
- Modify: `model/src/claim_measurement/difficulty/ft_eval.py`
- Modify: `model/src/claim_measurement/difficulty/test_ft_eval.py`

- [ ] **Step 1: Write the failing test** (append to `test_ft_eval.py`)

```python
from claim_measurement.difficulty.bakeoff_npz import write_embedding_npz
from claim_measurement.difficulty.ft_eval import main
from claim_measurement.difficulty.train_fold import write_fold_embeddings


def test_main_prints_the_gate_comparison_against_features37(tmp_path, capsys):
    data_root = tmp_path / "data"
    emb_dir = data_root / "results" / "bakeoff" / "emb" / "features37"
    rng = np.random.default_rng(0)
    n = 60
    seg_ids = [f"p{i:03d}" for i in range(n)]  # zero-padded -> lexical sort == list order
    grades = rng.integers(0, 11, size=n)
    composers = np.arange(n)  # all distinct -> vacuous disjointness, like the real 900

    for i, seg_id in enumerate(seg_ids):
        write_embedding_npz(emb_dir / f"{seg_id}.npz",
                             {"raw37": rng.normal(size=5).astype(np.float32)},
                             grade=int(grades[i]), composer_id=int(composers[i]))

    fold_emb_dir = tmp_path / "fold_embeddings"
    for f in range(5):
        # feature 0 is a strong linear signal so the gate reports SIG, not noise
        embeddings = np.column_stack([grades.astype(np.float32) * (f + 1),
                                       rng.normal(size=(n, 2)).astype(np.float32)])
        write_fold_embeddings(fold_emb_dir / f"emb_fold{f}.npz", seg_ids=seg_ids,
                               embeddings=embeddings, grades=grades, composer_ids=composers)

    exit_code = main(["--data-root", str(data_root), "--fold-emb-dir", str(fold_emb_dir)])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "moonbeam_ft_mean|ridge - features37|ridge:" in out
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_ft_eval.py -q --no-cov -k main_prints
```
Expected: FAIL — `TypeError: main() takes 0 positional arguments but 1 was given`
(current `main` is the Task-13 placeholder)

- [ ] **Step 3: Implement** (replace the placeholder `if __name__ == "__main__":`
  block at the bottom of `ft_eval.py`)

```python
from claim_measurement.difficulty.bakeoff_cv import paired_boot, tau_c  # noqa: E402
from claim_measurement.difficulty.bakeoff_npz import read_embedding_npz  # noqa: E402
from claim_measurement.difficulty.bakeoff_paths import resolve_paths  # noqa: E402
from claim_measurement.difficulty.train_fold import read_fold_embeddings  # noqa: E402


def _ridge_oof(X, y, composers, n_folds, seed):
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    oof = np.full(len(y), np.nan)
    for test_idx in composer_disjoint_folds(composers, n_folds, seed):
        train_idx = np.setdiff1d(np.arange(len(y)), test_idx)
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
        model.fit(np.nan_to_num(X[train_idx]), y[train_idx])
        oof[test_idx] = model.predict(np.nan_to_num(X[test_idx]))
    return oof


def _load_features37(emb_root: Path):
    paths = sorted((emb_root / "emb" / "features37").glob("*.npz"))
    if not paths:
        raise SystemExit(f"no features37 .npz files under {emb_root / 'emb' / 'features37'}")
    X, y, composers = [], [], []
    for path in paths:
        record = read_embedding_npz(path)
        X.append(record.embeddings["raw37"])
        y.append(record.grade)
        composers.append(record.composer_id)
    return np.stack(X), np.array(y), np.array(composers), [p.stem for p in paths]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--fold-emb-dir", type=Path, required=True,
                    help="dir containing emb_fold0.npz .. emb_fold{N_FOLDS-1}.npz from train_fold.py")
    args = ap.parse_args(argv)

    emb_root = resolve_paths(args.data_root).emb_root
    Xf, y, composers, seg_ids = _load_features37(emb_root)

    emb_by_fold = {}
    for f in range(N_FOLDS):
        fold_data = read_fold_embeddings(args.fold_emb_dir / f"emb_fold{f}.npz")
        if fold_data["seg_ids"] != seg_ids:
            raise SystemExit(
                f"emb_fold{f}.npz row order does not match features37's seg_id order; "
                f"the comparison would be unpaired")
        emb_by_fold[f] = fold_data["embeddings"]

    ft_oof = oof_tau_per_fold(emb_by_fold, y, composers, N_FOLDS, SEED)
    f37_oof = _ridge_oof(Xf, y, composers, N_FOLDS, SEED)

    print(f"n={len(y)} pieces, {len(set(composers))} composers")
    print(f"features37|ridge       tau-c {tau_c(f37_oof, y):.4f}")
    print(f"moonbeam_ft_mean|ridge tau-c {tau_c(ft_oof, y):.4f}")

    d, lo, hi, p = paired_boot(f37_oof, ft_oof, y, seed=SEED)
    print(f"moonbeam_ft_mean|ridge - features37|ridge: {d:+.4f} CI95[{lo:+.4f},{hi:+.4f}] "
          f"P(diff<=0)={p:.3f} {'SIG' if lo > 0 else 'noise'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

  Note: this replaces the Task-13 placeholder's trailing block; the module-level
  imports of `paired_boot`, `read_embedding_npz`, `resolve_paths`, and
  `read_fold_embeddings` are inserted directly below the existing
  `from claim_measurement.difficulty.bakeoff_cv import composer_disjoint_folds`
  import line (not deferred inside `main`), matching this module's plain-import
  style (it is not a `# /// script`).

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
```
Expected: PASS, 57 passed.

- [ ] **Step 5: Commit**

```bash
git add model/src/claim_measurement/difficulty/ft_eval.py \
        model/src/claim_measurement/difficulty/test_ft_eval.py \
    && git commit -m "feat(#149): ft_eval.main -- the #138 Phase 1 gate (i) report"
```

---

## Group E — `realaudio_check.py` (sequential internally; depends on Group 0 only; parallel with C/D/F)

Second gate, on the 709-piece real-audio subset. Scope note (deliberate, not an
oversight): this module's `main()` implements exactly the resumable-transcription
stage the spec's "Tested through" line commits to testing
(`main --stage transcribe` with an injected fake transcriber). The scoring
primitive (`score_audio_subset`) is built and unit-tested as a pure function
across Tasks 18-19; wiring it against real per-piece audio embeddings (themselves
produced by a separate `moonbeam_extract_script.py`-style run against each fold's
saved adapter — a GPU step, out of this module's scope) is a short driver snippet
documented in the runbook (Group G), not additional untested CLI code. This
mirrors `features37_compare.py`'s own `main()`, which likewise has no dedicated
pytest coverage — verified by a real-data script re-run instead (Task 1).

The design spec's "Real-audio second gate" section requires `realaudio_check.py`
to report three things: (a) tau-c on the audio subset, paired-bootstrapped
against **features37 on the same pieces** — this is the actual gate; (b) the
same subset's symbolic tau-c, so any gap is attributable to audio provenance
rather than the subset being easier or harder; (c) MIDI drift vs the stored
Transkun MIDIs. Task 16 covers (c). Task 18 builds the (b) scaffolding —
matched audio-vs-symbolic scoring per fold. Task 19 adds (a) — the
features37-paired gate — by scoring features37 through the SAME
composer-disjoint folds (`bakeoff_cv.composer_disjoint_folds` at the same
seed) as ordinary OOF (fit on each fold's train rows, predict that fold's
test rows), then restricting those OOF predictions to the audio subset's
rows before pairing against the audio-derived predictions. features37 is
never refit on only the audio subset — that would change its training set
and break the pairing.

### Task 16: `midi_drift` computes note-count delta and onset F1 with greedy pitch+tolerance matching

**Group:** E (first)

**Behavior being verified:** Identical note lists give zero delta and perfect
F1; a shifted note (beyond tolerance) plus an extra note degrade F1 and report
the correct count delta.

**Interface under test:** `realaudio_check.midi_drift(reference_notes, candidate_notes, onset_tolerance) -> dict`

**Files:**
- Create: `model/src/claim_measurement/difficulty/realaudio_check.py`
- Create: `model/src/claim_measurement/difficulty/test_realaudio_check.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for realaudio_check (#149 / #138 Phase 1) -- the real-audio second
gate: MIDI drift, resumable transcription, and per-fold audio scoring.

Run: cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
"""
import json

import numpy as np
import pytest

from claim_measurement.difficulty.realaudio_check import midi_drift


def test_midi_drift_computes_note_count_delta_and_onset_f1_with_tolerance_matching():
    reference = [{"pitch": 60, "onset": 0.0, "offset": 0.5, "velocity": 80},
                 {"pitch": 64, "onset": 0.5, "offset": 1.0, "velocity": 80}]

    identical = midi_drift(reference, reference, onset_tolerance=0.05)
    assert identical == {"note_count_delta": 0, "onset_f1": 1.0}

    candidate = [{"pitch": 60, "onset": 0.20, "offset": 0.5, "velocity": 80},  # onset shifted past tolerance
                 {"pitch": 64, "onset": 0.5, "offset": 1.0, "velocity": 80},
                 {"pitch": 67, "onset": 2.0, "offset": 2.5, "velocity": 80}]  # extra note
    degraded = midi_drift(reference, candidate, onset_tolerance=0.05)
    assert degraded["note_count_delta"] == 1
    assert degraded["onset_f1"] == pytest.approx(2 / 5)  # tp=1, precision=1/3, recall=1/2
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_realaudio_check.py -q --no-cov
```
Expected: FAIL — `ModuleNotFoundError: No module named 'claim_measurement.difficulty.realaudio_check'`

- [ ] **Step 3: Implement** (this is the start of `realaudio_check.py`)

```python
"""#138 Phase 1 real-audio second gate: 709 of 900 eval pieces have local
WAVs (re-fetched this session; see design spec). Resumable transcription
(`main`) plus MIDI drift (`midi_drift`) and per-fold audio scoring
(`score_audio_subset`) -- see this module's own docstring in the plan for the
deliberate scope split between what is CLI-wired here vs. what is a runbook
snippet over these tested primitives.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np

from claim_measurement.difficulty.bakeoff_cv import composer_disjoint_folds, paired_boot, tau_c

N_FOLDS, SEED = 5, 2026
ALPHAS = np.logspace(-1, 5, 25)


def midi_drift(reference_notes: list, candidate_notes: list, onset_tolerance: float) -> dict:
    """note-count delta (candidate - reference) and onset F1: a candidate
    note matches a reference note when they share pitch and onsets differ by
    <= onset_tolerance seconds. Matching is greedy nearest-onset-first, and
    each reference/candidate note is used at most once."""
    pairs = []
    for ci, c in enumerate(candidate_notes):
        for ri, r in enumerate(reference_notes):
            if r["pitch"] != c["pitch"]:
                continue
            dt = abs(r["onset"] - c["onset"])
            if dt <= onset_tolerance:
                pairs.append((dt, ci, ri))
    pairs.sort(key=lambda p: p[0])

    matched_ref, matched_cand, tp = set(), set(), 0
    for _dt, ci, ri in pairs:
        if ci in matched_cand or ri in matched_ref:
            continue
        matched_cand.add(ci)
        matched_ref.add(ri)
        tp += 1

    precision = tp / len(candidate_notes) if candidate_notes else 0.0
    recall = tp / len(reference_notes) if reference_notes else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"note_count_delta": len(candidate_notes) - len(reference_notes), "onset_f1": f1}


if __name__ == "__main__":
    sys.exit(0)  # placeholder exit; main() is added in Task 17
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_realaudio_check.py -q --no-cov
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add model/src/claim_measurement/difficulty/realaudio_check.py \
        model/src/claim_measurement/difficulty/test_realaudio_check.py \
    && git commit -m "feat(#149): realaudio_check.midi_drift -- note-count delta + onset F1"
```

### Task 17: `main --stage transcribe` skips pieces whose cache file already exists (resumable)

**Group:** E (depends on Task 16)

**Behavior being verified:** Given an injected fake transcriber, `main()`
transcribes only the pieces without an existing cache file, writes each
result atomically as JSON, and reports counts.

**Interface under test:** `realaudio_check.main(argv, transcriber=...) -> int`

**Files:**
- Modify: `model/src/claim_measurement/difficulty/realaudio_check.py`
- Modify: `model/src/claim_measurement/difficulty/test_realaudio_check.py`

- [ ] **Step 1: Write the failing test** (append to `test_realaudio_check.py`)

```python
from claim_measurement.difficulty.realaudio_check import main


def test_transcribe_stage_skips_pieces_whose_cache_file_already_exists(tmp_path):
    wav_manifest = tmp_path / "wav_manifest.json"
    wav_manifest.write_text(json.dumps([
        {"seg_id": "already_done", "wav_path": str(tmp_path / "a.wav")},
        {"seg_id": "new_piece", "wav_path": str(tmp_path / "b.wav")},
    ]))
    out_dir = tmp_path / "cache"
    out_dir.mkdir()
    (out_dir / "already_done.json").write_text(json.dumps({"notes": [], "pedals": []}))

    calls = []

    def fake_transcriber(wav_path):
        calls.append(wav_path)
        return ([{"pitch": 60, "onset": 0.0, "offset": 0.5, "velocity": 80}], [])

    exit_code = main(["--wav-manifest", str(wav_manifest), "--out-dir", str(out_dir)],
                      transcriber=fake_transcriber)

    assert exit_code == 0
    assert calls == [tmp_path / "b.wav"]  # only the not-yet-cached piece was transcribed
    cached = json.loads((out_dir / "new_piece.json").read_text())
    assert cached["notes"][0]["pitch"] == 60
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_realaudio_check.py -q --no-cov -k transcribe_stage
```
Expected: FAIL — `ImportError: cannot import name 'main'`

- [ ] **Step 3: Implement** (replace the placeholder `if __name__ == "__main__":`
  block at the bottom of `realaudio_check.py`)

```python
def _import_transcribe_wav():
    """Locate apps/inference/amt (import-safe transkun_cli) from CWD-up or
    file-up and return its transcribe_wav. Mirrors follower_eval/build_corpus.py's
    locate-and-import pattern -- kept lazy so tests that inject a fake
    transcriber never need transkun_cli's own heavy deps on the import path."""
    for base in (Path.cwd(), Path(__file__).resolve()):
        for parent in [base, *base.parents]:
            cand = parent / "apps" / "inference" / "amt"
            if (cand / "transkun_cli.py").exists():
                sys.path.insert(0, str(cand))
                from transkun_cli import transcribe_wav  # type: ignore

                return transcribe_wav
    raise RuntimeError(
        "could not locate apps/inference/amt/transkun_cli.py from CWD or module path"
    )


def _write_cache_atomic(path: Path, notes: list, pedals: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump({"notes": notes, "pedals": pedals}, fh)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def main(argv=None, transcriber=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wav-manifest", type=Path, required=True,
                    help="JSON list of {seg_id, wav_path}")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)

    if transcriber is None:
        transcriber = _import_transcribe_wav()

    entries = json.loads(args.wav_manifest.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    done, skipped, failed = 0, 0, []
    for e in entries:
        out_path = Path(args.out_dir) / f"{e['seg_id']}.json"
        if out_path.exists():
            skipped += 1
            continue
        try:
            notes, pedals = transcriber(Path(e["wav_path"]))
            _write_cache_atomic(out_path, notes, pedals)
            done += 1
        except Exception as exc:  # noqa: BLE001 -- record and continue; the report is the source of truth
            failed.append(f"{e['seg_id']}: {exc!r}")
    print(f"transcribed={done} skipped={skipped} failed={len(failed)}")
    for f in failed[:10]:
        print(f"  FAIL {f}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
```
Expected: PASS, 59 passed.

- [ ] **Step 5: Commit**

```bash
git add model/src/claim_measurement/difficulty/realaudio_check.py \
        model/src/claim_measurement/difficulty/test_realaudio_check.py \
    && git commit -m "feat(#149): realaudio_check.main -- resumable audio transcription cache"
```

### Task 18: `score_audio_subset` reports matched audio and symbolic tau-c on the same piece subset

**Group:** E (depends on Task 17)

**Behavior being verified:** For each audio piece, fit a ridge model on its OWN
fold's train rows and score BOTH its audio-derived embedding and its original
symbolic embedding through that same model — so any audio-vs-symbolic gap is
attributable to audio provenance, not to the subset being easier/harder (design
spec, "Real-audio second gate", item (b)). This task builds that scaffolding
plus the (c) MIDI-drift context; Task 19 adds the (a) features37-paired gate,
which is the actual pass/fail criterion the spec requires.

**Interface under test:** `realaudio_check.score_audio_subset(emb_by_fold, audio_embeddings, y, composers, seg_ids, n_folds, seed) -> dict`

**Files:**
- Modify: `model/src/claim_measurement/difficulty/realaudio_check.py`
- Modify: `model/src/claim_measurement/difficulty/test_realaudio_check.py`

- [ ] **Step 1: Write the failing test** (append to `test_realaudio_check.py`)

```python
from claim_measurement.difficulty.realaudio_check import score_audio_subset


def test_score_audio_subset_reports_matched_symbolic_and_audio_tau_c():
    rng = np.random.default_rng(2026)
    n = 60
    composers = np.array([f"composer_{i}" for i in range(n)])  # distinct -> vacuous disjointness
    y = rng.integers(0, 11, size=n).astype(float)
    seg_ids = [f"p{i:03d}" for i in range(n)]

    emb_by_fold = {
        f: np.column_stack([y, rng.normal(size=(n, 2)) * 0.01]).astype(np.float32)
        for f in range(5)
    }
    audio_subset = set(seg_ids[:20])
    audio_embeddings = {
        # 3 columns to match emb_by_fold's 3-column shape (y + 2 noise cols) --
        # score_audio_subset scores this row through the SAME ridge model fit
        # on emb_by_fold[fold], so the feature count must match exactly.
        seg_id: np.array([y[i] + 0.05, 0.0, 0.0], dtype=np.float32)
        for i, seg_id in enumerate(seg_ids) if seg_id in audio_subset
    }

    result = score_audio_subset(emb_by_fold, audio_embeddings, y, composers, seg_ids,
                                 n_folds=5, seed=2026)

    assert result["n"] == 20
    assert result["audio_tau_c"] > 0.9
    assert result["symbolic_tau_c"] > 0.9
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_realaudio_check.py -q --no-cov -k score_audio_subset
```
Expected: FAIL — `ImportError: cannot import name 'score_audio_subset'`

- [ ] **Step 3: Implement** (append to `realaudio_check.py`, after `midi_drift`)

```python
def score_audio_subset(emb_by_fold: dict, audio_embeddings: dict, y: np.ndarray,
                        composers: np.ndarray, seg_ids: list, n_folds: int, seed: int) -> dict:
    """For every seg_id in audio_embeddings (a subset of seg_ids), fit a ridge
    model on that piece's OWN test fold's train rows of emb_by_fold[fold] and
    score the piece's audio-derived embedding through it. Also scores the
    SAME piece's original symbolic embedding through the SAME model, so any
    audio-vs-symbolic gap is attributable to audio provenance, not to the
    subset being easier or harder (design spec's real-audio second gate)."""
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    idx_of = {s: i for i, s in enumerate(seg_ids)}
    test_folds = composer_disjoint_folds(composers, n_folds, seed)
    fold_of_idx = {i: f for f, idx in enumerate(test_folds) for i in idx}

    audio_pred, symbolic_pred, subset_y = [], [], []
    ridge_cache: dict = {}
    for seg_id, audio_embedding in audio_embeddings.items():
        i = idx_of[seg_id]
        fold = fold_of_idx[i]
        if fold not in ridge_cache:
            train_idx = np.setdiff1d(np.arange(len(seg_ids)), test_folds[fold])
            model = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
            model.fit(emb_by_fold[fold][train_idx], y[train_idx])
            ridge_cache[fold] = model
        model = ridge_cache[fold]
        audio_pred.append(model.predict(audio_embedding.reshape(1, -1))[0])
        symbolic_pred.append(model.predict(emb_by_fold[fold][i].reshape(1, -1))[0])
        subset_y.append(y[i])

    subset_y = np.array(subset_y)
    audio_pred, symbolic_pred = np.array(audio_pred), np.array(symbolic_pred)
    d, lo, hi, p = paired_boot(symbolic_pred, audio_pred, subset_y, seed=seed)
    return {
        "n": len(subset_y),
        "audio_tau_c": tau_c(audio_pred, subset_y),
        "symbolic_tau_c": tau_c(symbolic_pred, subset_y),
        "delta_vs_symbolic": d, "ci_lo_vs_symbolic": lo, "ci_hi_vs_symbolic": hi, "p_le_0_vs_symbolic": p,
    }
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
```
Expected: PASS, 60 passed.

- [ ] **Step 5: Commit**

```bash
git add model/src/claim_measurement/difficulty/realaudio_check.py \
        model/src/claim_measurement/difficulty/test_realaudio_check.py \
    && git commit -m "feat(#149): realaudio_check.score_audio_subset -- matched audio-vs-symbolic tau-c"
```

### Task 19: `score_audio_subset` adds the features37-paired gate on the same audio subset

**Group:** E (depends on Task 18, last task in this group)

**Behavior being verified:** features37 is scored through the SAME
composer-disjoint folds (`bakeoff_cv.composer_disjoint_folds` at the same
seed) as ordinary out-of-fold prediction (fit on each fold's train rows,
predict that fold's test rows) over the FULL piece set, then those OOF
predictions are restricted to the audio subset's rows and paired-bootstrapped
against the audio-derived predictions on those same rows. This is the actual
gate the design spec's "Real-audio second gate" section requires (item (a)):
"tau-c on the audio subset, paired-bootstrapped against features37 on the
same pieces." features37 is never refit on only the 20-piece audio subset —
only its already-computed, full-set OOF predictions are subset-restricted —
so the comparison stays paired against the same features37 fit the 0.8048
reference number itself rests on.

**Interface under test:** `realaudio_check.score_audio_subset(emb_by_fold, audio_embeddings, features37_x, y, composers, seg_ids, n_folds, seed) -> dict`
(now takes `features37_x`, the full-piece-set 37-feature matrix, as a new
required argument)

**Files:**
- Modify: `model/src/claim_measurement/difficulty/realaudio_check.py`
- Modify: `model/src/claim_measurement/difficulty/test_realaudio_check.py`

- [ ] **Step 1: Write the failing test** (append to `test_realaudio_check.py`;
  also update Task 18's test call site, since the signature gains a required
  argument)

```python
def test_score_audio_subset_reports_features37_gate_paired_against_audio():
    rng = np.random.default_rng(2026)
    n = 60
    composers = np.array([f"composer_{i}" for i in range(n)])  # distinct -> vacuous disjointness
    y = rng.integers(0, 11, size=n).astype(float)
    seg_ids = [f"p{i:03d}" for i in range(n)]

    emb_by_fold = {
        f: np.column_stack([y, rng.normal(size=(n, 2)) * 0.01]).astype(np.float32)
        for f in range(5)
    }
    # A deliberately weak features37 stand-in (heavy noise on top of y) so the
    # near-perfect audio arm clearly beats it -- this fixture only needs to
    # prove the gate computes and pairs correctly, not that any real numbers hold.
    features37_x = np.column_stack([y + rng.normal(scale=4.0, size=n),
                                     rng.normal(size=(n, 4))]).astype(np.float32)
    audio_subset = set(seg_ids[:20])
    audio_embeddings = {
        seg_id: np.array([y[i] + 0.05, 0.0, 0.0], dtype=np.float32)
        for i, seg_id in enumerate(seg_ids) if seg_id in audio_subset
    }

    result = score_audio_subset(emb_by_fold, audio_embeddings, features37_x, y, composers, seg_ids,
                                 n_folds=5, seed=2026)

    assert result["n"] == 20
    assert result["audio_tau_c"] > result["features37_tau_c"]
    assert result["delta_vs_features37"] > 0
    assert result["ci_lo_vs_features37"] > 0  # SIG on this fixture
    assert result["ci_lo_vs_features37"] <= result["delta_vs_features37"] <= result["ci_hi_vs_features37"]
```

Also update the earlier test's call site (Task 18's
`test_score_audio_subset_reports_matched_symbolic_and_audio_tau_c`) to pass a
`features37_x` matching `emb_by_fold`'s column count, since the signature is
now shared:

```python
    features37_x = rng.normal(size=(n, 5)).astype(np.float32)  # unused by this test's assertions

    result = score_audio_subset(emb_by_fold, audio_embeddings, features37_x, y, composers, seg_ids,
                                 n_folds=5, seed=2026)
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_realaudio_check.py -q --no-cov
```
Expected: FAIL — `TypeError` from `score_audio_subset()` on both tests: Task
18's `score_audio_subset` (still 7 params: `emb_by_fold, audio_embeddings, y,
composers, seg_ids, n_folds, seed`) does not accept the `features37_x`
argument both updated call sites now pass.

- [ ] **Step 3: Implement** (replace `score_audio_subset` in `realaudio_check.py`)

```python
def score_audio_subset(emb_by_fold: dict, audio_embeddings: dict, features37_x: np.ndarray,
                        y: np.ndarray, composers: np.ndarray, seg_ids: list,
                        n_folds: int, seed: int) -> dict:
    """For every seg_id in audio_embeddings (a subset of seg_ids), fit a ridge
    model on that piece's OWN test fold's train rows of emb_by_fold[fold] and
    score the piece's audio-derived embedding through it. Also scores the
    SAME piece's original symbolic embedding through the SAME model, so any
    audio-vs-symbolic gap is attributable to audio provenance, not to the
    subset being easier or harder (design spec's real-audio second gate, item
    (b)). THE GATE (item (a)): features37_x is scored via ordinary
    composer-disjoint OOF over the FULL piece set (fit on each fold's train
    rows, predict that fold's own test rows) -- never refit on the audio
    subset alone -- and those OOF predictions are then restricted to the
    audio subset's rows and paired-bootstrapped against the audio-derived
    predictions on those same rows."""
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    idx_of = {s: i for i, s in enumerate(seg_ids)}
    test_folds = composer_disjoint_folds(composers, n_folds, seed)
    fold_of_idx = {i: f for f, idx in enumerate(test_folds) for i in idx}

    # features37 OOF over the full set, matching folds/seed exactly -- computed
    # once here, independent of which pieces have audio, then subset below.
    f37_oof = np.full(len(y), np.nan)
    for fold, test_idx in enumerate(test_folds):
        train_idx = np.setdiff1d(np.arange(len(seg_ids)), test_idx)
        f37_model = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
        f37_model.fit(features37_x[train_idx], y[train_idx])
        f37_oof[test_idx] = f37_model.predict(features37_x[test_idx])

    audio_pred, symbolic_pred, f37_pred, subset_y = [], [], [], []
    ridge_cache: dict = {}
    for seg_id, audio_embedding in audio_embeddings.items():
        i = idx_of[seg_id]
        fold = fold_of_idx[i]
        if fold not in ridge_cache:
            train_idx = np.setdiff1d(np.arange(len(seg_ids)), test_folds[fold])
            model = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
            model.fit(emb_by_fold[fold][train_idx], y[train_idx])
            ridge_cache[fold] = model
        model = ridge_cache[fold]
        audio_pred.append(model.predict(audio_embedding.reshape(1, -1))[0])
        symbolic_pred.append(model.predict(emb_by_fold[fold][i].reshape(1, -1))[0])
        f37_pred.append(f37_oof[i])
        subset_y.append(y[i])

    subset_y = np.array(subset_y)
    audio_pred, symbolic_pred, f37_pred = np.array(audio_pred), np.array(symbolic_pred), np.array(f37_pred)
    d_sym, lo_sym, hi_sym, p_sym = paired_boot(symbolic_pred, audio_pred, subset_y, seed=seed)
    d_f37, lo_f37, hi_f37, p_f37 = paired_boot(f37_pred, audio_pred, subset_y, seed=seed)
    return {
        "n": len(subset_y),
        "audio_tau_c": tau_c(audio_pred, subset_y),
        "symbolic_tau_c": tau_c(symbolic_pred, subset_y),
        "features37_tau_c": tau_c(f37_pred, subset_y),
        "delta_vs_symbolic": d_sym, "ci_lo_vs_symbolic": lo_sym, "ci_hi_vs_symbolic": hi_sym, "p_le_0_vs_symbolic": p_sym,
        "delta_vs_features37": d_f37, "ci_lo_vs_features37": lo_f37, "ci_hi_vs_features37": hi_f37, "p_le_0_vs_features37": p_f37,
    }
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
```
Expected: PASS, 61 passed.

- [ ] **Step 5: Commit**

```bash
git add model/src/claim_measurement/difficulty/realaudio_check.py \
        model/src/claim_measurement/difficulty/test_realaudio_check.py \
    && git commit -m "feat(#149): realaudio_check.score_audio_subset -- add the features37-paired gate"
```

---

## Group F — `push_train_dataset.py` (sequential internally; depends on Group A; parallel with C/D/E)

The judgment is entirely in WHAT gets staged; the upload itself is three lines
behind an injected `uploader` so no test ever touches the network. Staging
itself never touches the network either — the MoonBeam fork snapshot and the
Transkun MIDIs must already exist locally (see `moonbeam_extract_script.py`'s
SETUP section). Four tasks, one file, run in order 20 → 21 → 22 → 23.

### Task 20: `stage_training_bundle` copies every piece referenced by any fold plan and reports counts

**Group:** F (first)

**Behavior being verified:** The union of all folds' train+val+test seg_ids is
copied into the staging tree exactly once each, grades and fold plans are
written alongside, the repo snapshot is copied wholesale, and a report gives
accurate counts plus a checksum.

**Interface under test:** `push_train_dataset.stage_training_bundle(paths: BundleSources, plans: list[FoldPlan], staging_dir: Path) -> BundleReport`

**Files:**
- Create: `model/src/claim_measurement/difficulty/push_train_dataset.py`
- Create: `model/src/claim_measurement/difficulty/test_push_train_dataset.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for push_train_dataset (#149 / #138 Phase 1) -- hermetic HF Jobs
training-bundle staging + upload, uploader injected so no test hits the network.

Run: cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
"""
import json

import pytest

from claim_measurement.difficulty.fold_plan import FoldPlan
from claim_measurement.difficulty.push_train_dataset import BundleSources, stage_training_bundle


def _write_fake_repo(repo_dir):
    (repo_dir / "src").mkdir(parents=True)
    (repo_dir / "src" / "model_config.json").write_text("{}")
    (repo_dir / "README.md").write_text("moonbeam fork snapshot")


def test_stage_training_bundle_copies_every_referenced_piece_and_reports_counts(tmp_path):
    midi_dir = tmp_path / "transkun_mid"
    midi_dir.mkdir()
    seg_ids = ["a", "b", "c", "d"]
    for seg_id in seg_ids:
        (midi_dir / f"{seg_id}.mid").write_bytes(b"midi-bytes")
    repo_snapshot_dir = tmp_path / "repo"
    _write_fake_repo(repo_snapshot_dir)

    plans = [
        FoldPlan(fold=0, test_seg_ids=("a",), train_seg_ids=("b", "c"), val_seg_ids=("d",)),
        FoldPlan(fold=1, test_seg_ids=("b",), train_seg_ids=("a", "c"), val_seg_ids=("d",)),
    ]
    paths = BundleSources(midi_dir=midi_dir, grades={s: i for i, s in enumerate(seg_ids)},
                           repo_snapshot_dir=repo_snapshot_dir)
    staging_dir = tmp_path / "staging"

    report = stage_training_bundle(paths, plans, staging_dir)

    assert report.n_midis == 4  # {a,b,c,d}, deduplicated across both plans
    assert report.n_fold_plans == 2
    assert report.repo_snapshot_files == 2
    assert len(report.checksum) == 64  # sha256 hex digest
    for seg_id in seg_ids:
        assert (staging_dir / "midi" / f"{seg_id}.mid").exists()
    assert json.loads((staging_dir / "grades.json").read_text()) == {s: i for i, s in enumerate(seg_ids)}
    staged_plans = json.loads((staging_dir / "fold_plans.json").read_text())
    assert len(staged_plans) == 2
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_push_train_dataset.py -q --no-cov
```
Expected: FAIL — `ModuleNotFoundError: No module named 'claim_measurement.difficulty.push_train_dataset'`

- [ ] **Step 3: Implement** (this is the start of `push_train_dataset.py`)

```python
"""Hermetic HF Jobs training-bundle staging + upload for #138 Phase 1.

Judgment lives entirely in WHAT gets staged (stage_training_bundle); the
upload itself is three lines behind an injected `uploader` so tests never
touch the network. Staging never fetches anything over the network either --
the MoonBeam fork snapshot and the Transkun MIDIs must already exist locally
(see moonbeam_extract_script.py's SETUP section for the fork clone recipe).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from claim_measurement.difficulty.fold_plan import FoldPlan


@dataclass(frozen=True)
class BundleSources:
    midi_dir: Path
    grades: dict
    repo_snapshot_dir: Path


@dataclass(frozen=True)
class BundleReport:
    n_midis: int
    n_fold_plans: int
    repo_snapshot_files: int
    checksum: str


def _referenced_seg_ids(plans: list) -> list:
    seg_ids: set = set()
    for plan in plans:
        seg_ids.update(plan.train_seg_ids)
        seg_ids.update(plan.val_seg_ids)
        seg_ids.update(plan.test_seg_ids)
    return sorted(seg_ids)


def stage_training_bundle(paths: BundleSources, plans: list, staging_dir: Path) -> BundleReport:
    staging_dir = Path(staging_dir)
    midi_out = staging_dir / "midi"
    midi_out.mkdir(parents=True, exist_ok=True)

    seg_ids = _referenced_seg_ids(plans)
    missing_grades = [s for s in seg_ids if s not in paths.grades]
    if missing_grades:
        raise ValueError(
            f"{len(missing_grades)} piece(s) referenced by a fold plan have no grade: "
            f"{missing_grades[:5]}")

    for seg_id in seg_ids:
        src = Path(paths.midi_dir) / f"{seg_id}.mid"
        if not src.exists():
            raise FileNotFoundError(f"fold plan references {seg_id}, but {src} does not exist")
        shutil.copy2(src, midi_out / f"{seg_id}.mid")

    (staging_dir / "grades.json").write_text(json.dumps(
        {seg_id: paths.grades[seg_id] for seg_id in seg_ids}))
    (staging_dir / "fold_plans.json").write_text(
        json.dumps([dataclasses.asdict(p) for p in plans]))

    repo_out = staging_dir / "moonbeam_repo"
    if repo_out.exists():
        shutil.rmtree(repo_out)
    shutil.copytree(paths.repo_snapshot_dir, repo_out)
    repo_files = sum(1 for p in repo_out.rglob("*") if p.is_file())

    hasher = hashlib.sha256()
    for p in sorted(staging_dir.rglob("*")):
        if p.is_file():
            hasher.update(str(p.relative_to(staging_dir)).encode())
            hasher.update(str(p.stat().st_size).encode())

    return BundleReport(n_midis=len(seg_ids), n_fold_plans=len(plans),
                         repo_snapshot_files=repo_files, checksum=hasher.hexdigest())


if __name__ == "__main__":
    sys.exit(0)  # placeholder exit; main() is added in Task 23
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_push_train_dataset.py -q --no-cov
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add model/src/claim_measurement/difficulty/push_train_dataset.py \
        model/src/claim_measurement/difficulty/test_push_train_dataset.py \
    && git commit -m "feat(#149): push_train_dataset.stage_training_bundle -- hermetic bundle staging"
```

### Task 21: `stage_training_bundle` raises loudly when a referenced piece has no grade

**Group:** F (depends on Task 20)

**Behavior being verified:** A missing grade must abort staging, not silently
omit the piece from `grades.json` (a truncated bundle is worse than no bundle —
the training job would either crash on a KeyError mid-run or, worse, silently
train on fewer pieces than the fold plan says).

**Interface under test:** `push_train_dataset.stage_training_bundle`

**Files:**
- Modify: `model/src/claim_measurement/difficulty/test_push_train_dataset.py`

- [ ] **Step 1: Write the failing test** (append to `test_push_train_dataset.py`)

```python
def test_stage_training_bundle_raises_when_a_referenced_piece_has_no_grade(tmp_path):
    midi_dir = tmp_path / "transkun_mid"
    midi_dir.mkdir()
    (midi_dir / "a.mid").write_bytes(b"x")
    repo_snapshot_dir = tmp_path / "repo"
    _write_fake_repo(repo_snapshot_dir)
    plans = [FoldPlan(fold=0, test_seg_ids=("a",), train_seg_ids=(), val_seg_ids=())]
    paths = BundleSources(midi_dir=midi_dir, grades={}, repo_snapshot_dir=repo_snapshot_dir)

    with pytest.raises(ValueError, match="no grade"):
        stage_training_bundle(paths, plans, tmp_path / "staging")
```

- [ ] **Step 2: Run test — verify it currently PASSES** (Task 20's
  `missing_grades` guard already implements this; run to confirm no
  false-negative risk)

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_push_train_dataset.py -q --no-cov
```
Expected: PASS (2 tests)

- [ ] **Step 3: No implementation change needed.**

- [ ] **Step 4: Re-run to confirm**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
```
Expected: PASS, 63 passed.

- [ ] **Step 5: Commit**

```bash
git add model/src/claim_measurement/difficulty/test_push_train_dataset.py \
    && git commit -m "test(#149): stage_training_bundle raises loudly on a missing grade"
```

### Task 22: `stage_training_bundle` raises loudly when a referenced piece has no MIDI on disk

**Group:** F (depends on Task 21)

**Behavior being verified:** A missing MIDI file must abort staging with a
clear message naming the piece — never silently produce a bundle short of a
fold plan's referenced pieces.

**Interface under test:** `push_train_dataset.stage_training_bundle`

**Files:**
- Modify: `model/src/claim_measurement/difficulty/test_push_train_dataset.py`

- [ ] **Step 1: Write the failing test** (append to `test_push_train_dataset.py`)

```python
def test_stage_training_bundle_raises_when_a_referenced_piece_has_no_midi_on_disk(tmp_path):
    midi_dir = tmp_path / "transkun_mid"
    midi_dir.mkdir()  # empty -- "a.mid" is never written
    repo_snapshot_dir = tmp_path / "repo"
    _write_fake_repo(repo_snapshot_dir)
    plans = [FoldPlan(fold=0, test_seg_ids=("a",), train_seg_ids=(), val_seg_ids=())]
    paths = BundleSources(midi_dir=midi_dir, grades={"a": 3}, repo_snapshot_dir=repo_snapshot_dir)

    with pytest.raises(FileNotFoundError, match="a.mid"):
        stage_training_bundle(paths, plans, tmp_path / "staging")
```

- [ ] **Step 2: Run test — verify it currently PASSES** (Task 20's
  `src.exists()` guard already implements this)

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_push_train_dataset.py -q --no-cov
```
Expected: PASS (3 tests)

- [ ] **Step 3: No implementation change needed.**

- [ ] **Step 4: Re-run to confirm**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
```
Expected: PASS, 64 passed.

- [ ] **Step 5: Commit**

```bash
git add model/src/claim_measurement/difficulty/test_push_train_dataset.py \
    && git commit -m "test(#149): stage_training_bundle raises loudly on a missing MIDI file"
```

### Task 23: `main()` builds fold plans from the manifest, stages the bundle, and calls the injected uploader

**Group:** F (depends on Task 22, last task in this group)

**Behavior being verified:** `main()` joins the manifest/labels (via
`load_bakeoff_manifest`), restricts to the eval sample, calls
`build_fold_plans`, stages the bundle, and calls the injected `uploader` with
the staging dir and repo id — no network touched.

**Interface under test:** `push_train_dataset.main(argv, uploader=...) -> int`

**Files:**
- Modify: `model/src/claim_measurement/difficulty/push_train_dataset.py`
- Modify: `model/src/claim_measurement/difficulty/test_push_train_dataset.py`

- [ ] **Step 1: Write the failing test** (append to `test_push_train_dataset.py`)

```python
from claim_measurement.difficulty.push_train_dataset import main


def test_main_builds_fold_plans_stages_and_calls_the_injected_uploader(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    labels_path = tmp_path / "labels.json"
    midi_dir = tmp_path / "transkun_mid"
    midi_dir.mkdir()

    seg_ids = [f"p{i:02d}" for i in range(10)]
    manifest = [{"seg_id": s, "key": f"{s}.mid", "grade": i % 11, "video_id": "x",
                 "midi_name": f"mid/{s}.mid"} for i, s in enumerate(seg_ids)]
    labels = {f"{s}.mid": {"composer": f"composer_{i}"} for i, s in enumerate(seg_ids)}
    manifest_path.write_text(json.dumps(manifest))
    labels_path.write_text(json.dumps(labels))
    for s in seg_ids:
        (midi_dir / f"{s}.mid").write_bytes(b"x")

    sample_manifest_path = tmp_path / "sample_manifest.json"
    sample_manifest_path.write_text(json.dumps([{"seg_id": s} for s in seg_ids[:6]]))

    repo_snapshot_dir = tmp_path / "repo"
    _write_fake_repo(repo_snapshot_dir)
    staging_dir = tmp_path / "staging"

    calls = []

    def fake_uploader(staged_dir, repo_id):
        calls.append((staged_dir, repo_id))

    exit_code = main(
        [
            "--manifest", str(manifest_path),
            "--labels", str(labels_path),
            "--sample-manifest", str(sample_manifest_path),
            "--midi-dir", str(midi_dir),
            "--repo-snapshot-dir", str(repo_snapshot_dir),
            "--staging-dir", str(staging_dir),
            "--repo-id", "jaidhiman/phase1-lora-bundle",
            "--n-folds", "2",
        ],
        uploader=fake_uploader,
    )

    assert exit_code == 0
    assert calls == [(staging_dir, "jaidhiman/phase1-lora-bundle")]
    staged_plans = json.loads((staging_dir / "fold_plans.json").read_text())
    assert len(staged_plans) == 2
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/test_push_train_dataset.py -q --no-cov -k main_builds
```
Expected: FAIL — `ImportError: cannot import name 'main'`

- [ ] **Step 3: Implement** (replace the placeholder `if __name__ == "__main__":`
  block at the bottom of `push_train_dataset.py`)

```python
def main(argv=None, uploader=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--sample-manifest", type=Path, required=True,
                    help="the 900-piece eval sample_manifest.json (run_bakeoff.py --stage sample's output)")
    ap.add_argument("--midi-dir", type=Path, required=True)
    ap.add_argument("--repo-snapshot-dir", type=Path, required=True)
    ap.add_argument("--staging-dir", type=Path, required=True)
    ap.add_argument("--repo-id", required=True,
                    help="private HF dataset repo id, e.g. jaidhiman/phase1-lora-bundle")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--val-frac", type=float, default=0.12)
    args = ap.parse_args(argv)

    from claim_measurement.difficulty.bakeoff_sampling import load_bakeoff_manifest
    from claim_measurement.difficulty.fold_plan import build_fold_plans

    pool_entries = load_bakeoff_manifest(args.manifest, args.labels, args.midi_dir)
    sample_seg_ids = {e["seg_id"] for e in json.loads(args.sample_manifest.read_text())}
    eval_entries = sorted((e for e in pool_entries if e.seg_id in sample_seg_ids),
                          key=lambda e: e.seg_id)

    plans = build_fold_plans(eval_entries, pool_entries, args.n_folds, args.seed, args.val_frac)

    grades = {e.seg_id: e.grade for e in pool_entries}
    paths = BundleSources(midi_dir=args.midi_dir, grades=grades,
                          repo_snapshot_dir=args.repo_snapshot_dir)
    report = stage_training_bundle(paths, plans, args.staging_dir)
    print(f"staged {report.n_midis} MIDIs, {report.n_fold_plans} fold plans, "
          f"{report.repo_snapshot_files} repo files, checksum {report.checksum}")

    if uploader is None:
        from huggingface_hub import HfApi

        def uploader(staged_dir: Path, repo_id: str) -> None:
            api = HfApi()
            api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
            api.upload_folder(folder_path=str(staged_dir), repo_id=repo_id, repo_type="dataset")

    uploader(args.staging_dir, args.repo_id)
    print(f"uploaded to {args.repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
```
Expected: PASS, 65 passed (24 net-new tests over the 41 baseline).

- [ ] **Step 5: Commit**

```bash
git add model/src/claim_measurement/difficulty/push_train_dataset.py \
        model/src/claim_measurement/difficulty/test_push_train_dataset.py \
    && git commit -m "feat(#149): push_train_dataset.main -- fold plans + staging + injected upload"
```

---

## Group G — Runbook (solo, last; depends on Groups C, D, E, F)

### Task 24: Write `docs/mirex/phase1-lora-runbook.md`

**Group:** G (depends on Tasks 12, 15, 19, 23 — every CLI this doc references
must exist and be tested first)

**Behavior being verified:** N/A (documentation, not code) — there is no pytest
step for this task. Its "verification" is that every command it shows uses
flags that actually exist on the CLIs built in Groups C/D/E/F (cross-checked
against those tasks' `argparse` definitions above) and that the abort criteria
match the design spec's Open Questions exactly.

**Files:**
- Create: `docs/mirex/phase1-lora-runbook.md`

- [ ] **Step 1: Write the file**

```markdown
# #138 Phase 1 LoRA Fine-Tune Runbook

Operator sequence for the MoonBeam-839M LoRA fine-tune gate (#149). Every step
below that spends money or touches a GPU is **human-lit**: the operator runs it,
reads the printed numbers, and decides whether to continue. Nothing in this
repo launches an HF Job automatically.

## Stage 0 — one-time setup (already covered by moonbeam_extract_script.py)

Clone the MoonBeam fork at the pinned commit and fetch the checkpoint (see
`model/src/claim_measurement/difficulty/moonbeam_extract_script.py`'s module
docstring for the exact commands). Confirm the checkpoint loads against
`model_config.json` with zero missing/unexpected keys — `_real_loader`'s
strict check in both `moonbeam_extract_script.py` and `train_fold.py` refuses
to proceed otherwise.

Before staging anything, sanity-check the fold-plan sizes against the design
spec's verified facts (this worktree's `model/data` is empty; point
`--data-root` at the main checkout):

```bash
cd model/src/claim_measurement/difficulty && uv run --no-project --script \
    features37_compare.py --data-root /Users/jdhiman/Documents/crescendai/model/data
```
Expected: `features37|ridge` tau-c `0.8048`, `moonbeam_mean|ridge` tau-c `0.8257`
(the Phase 0 numbers this whole phase is measured against — Task Group 0's
harness). If either number has drifted, STOP: something about the data or the
fold seed has changed and the gate threshold below is no longer valid.

## Stage 1 — stage and upload the training bundle (once)

```bash
cd model/src/claim_measurement/difficulty && uv run python -m \
    claim_measurement.difficulty.push_train_dataset \
    --manifest /path/to/model/data/results/amt_gap_curve/manifest.json \
    --labels /path/to/model/data/raw/psyllabus/new_clean_data.json \
    --sample-manifest /path/to/model/data/results/bakeoff/sample_manifest.json \
    --midi-dir /path/to/model/data/results/amt_gap_curve/transkun_mid \
    --repo-snapshot-dir /path/to/model/data/weights/moonbeam/repo \
    --staging-dir /path/to/staging/phase1-lora-bundle \
    --repo-id <your-hf-username>/phase1-lora-bundle
```
Read the printed `staged N MIDIs, 5 fold plans, ...` report before it uploads.
Abort criterion: if `n_midis` is far from the expected ~4000-4300 per fold
(sum across all 5 folds' train+val, deduplicated union will be close to the
full 5798-piece pool minus per-fold exclusions), STOP and re-check the sample
manifest and labels join.

## Stage 2 — the pilot fold

```bash
hf jobs uv run --flavor a100-large --timeout 3h \
    model/src/claim_measurement/difficulty/train_fold.py \
    --fold 0 \
    --checkpoint /path/to/moonbeam_839M.pt \
    --repo-root /path/to/moonbeam/repo \
    --model-config /path/to/moonbeam/repo/src/llama_recipes/configs/model_config.json \
    --fold-plan /path/to/staging/phase1-lora-bundle/fold_plans.json \
    --pool-grades /path/to/staging/phase1-lora-bundle/grades.json \
    --eval-manifest /path/to/eval_manifest.json \
    --midi-dir /path/to/staging/phase1-lora-bundle/midi \
    --out-dir /path/to/fold_embeddings/fold0
```

Monitor with `hf jobs ps`, `hf jobs logs <job-id>`, `hf jobs inspect <job-id>`.
Abort criteria (design spec's Open Questions):
- **Val ranking tau is flat or diverging** across the printed `epoch N:
  val_ranking_tau=...` lines — stop, the objective/LR is not working, do not
  spend money on folds 1-4.
- **Peak memory does not fit `a100-large` at `--micro-batch 8`** — drop to
  `--micro-batch 4` and retry the pilot before scaling to the remaining folds
  (do not switch GPU flavor first).
- **Measured wall-clock is wildly off from the ~1 GPU-hr/fold estimate** — use
  the MEASURED number, not the estimate, to budget folds 1-4 (`hf jobs stats`
  after completion).

If the pilot's `emb_fold0.npz` looks reasonable (900 rows, finite values), proceed.

## Stage 3 — folds 1-4 (same seed, ~$13 total for all 5)

Repeat Stage 2 with `--fold 1`, `--fold 2`, `--fold 3`, `--fold 4`, same
`--fold-plan`/`--pool-grades`/`--eval-manifest`, different `--out-dir` per fold
(e.g. `fold_embeddings/fold{N}`).

## Stage 4 — gate (i): encoder-as-feature-extractor (local, CPU, free)

```bash
cd model && uv run python -m claim_measurement.difficulty.ft_eval \
    --data-root /path/to/model/data --fold-emb-dir /path/to/fold_embeddings
```
Expected output: `moonbeam_ft_mean|ridge - features37|ridge: +0.0XXX
CI95[+a,+b] P(diff<=0)=p SIG|noise`. **The gate passes only if `a > 0`
(`SIG`).** If `noise`, STOP — do not proceed to the real-audio gate or report
an end-to-end number; the fine-tune did not clear 0.8048.

## Stage 5 — gate (ii): real-audio second gate (local, resumable)

Transcribe the 709 available WAVs (resumable — safe to interrupt and re-run):

```bash
cd model/src/claim_measurement/difficulty && uv run python -m \
    claim_measurement.difficulty.realaudio_check \
    --wav-manifest /path/to/audio_wav_manifest.json \
    --out-dir /path/to/audio_midi_cache
```

Extract MoonBeam embeddings for each transcribed piece using ITS OWN fold's
saved adapter (a `moonbeam_extract_script.py`-style run per fold, pointed at
`--repo-root`/`--model-config` as before and the fold's `adapter/` directory
loaded via `peft`'s `PeftModel.from_pretrained`), writing one `.npz` per piece
into `audio_emb/` via the standard `bakeoff_npz.write_embedding_npz` contract
(key `"mean_pool"`). This step is a GPU-optional but compute-bearing step
outside `realaudio_check.py`'s tested scope; wire it as a short local script.

Then compute the real-audio gate — audio vs. features37 on the SAME subset,
scored through the SAME composer-disjoint folds/seed — plus the matched
symbolic comparison that makes it interpretable:

```python
import json
import numpy as np
from pathlib import Path
from claim_measurement.difficulty.bakeoff_npz import read_embedding_npz
from claim_measurement.difficulty.ft_eval import _load_features37
from claim_measurement.difficulty.train_fold import read_fold_embeddings
from claim_measurement.difficulty.realaudio_check import score_audio_subset
from claim_measurement.difficulty.bakeoff_paths import resolve_paths

emb_root = resolve_paths(Path("/path/to/model/data")).emb_root
Xf, y, composers, seg_ids = _load_features37(emb_root)
emb_by_fold = {f: read_fold_embeddings(f"/path/to/fold_embeddings/fold{f}/emb_fold{f}.npz")["embeddings"]
               for f in range(5)}
audio_dir = Path("/path/to/audio_emb")
audio_embeddings = {p.stem: read_embedding_npz(p).embeddings["mean_pool"]
                    for p in sorted(audio_dir.glob("*.npz"))}

result = score_audio_subset(emb_by_fold, audio_embeddings, Xf, y, composers, seg_ids,
                             n_folds=5, seed=2026)
print(result)
```
Expected: `audio_tau_c`, `symbolic_tau_c`, and `features37_tau_c` all reported.
`delta_vs_features37`/`ci_lo_vs_features37`/`ci_hi_vs_features37` are **THE
GATE** (item (a) of the design spec's "Real-audio second gate": tau-c on the
audio subset, paired-bootstrapped against features37 on the same pieces).
**The gate passes only if `ci_lo_vs_features37 > 0`** on this n=709(-ish)
subset (half-width ≈ ±0.017 per the design spec, enough to resolve the
+0.024 margin). `delta_vs_symbolic`/`ci_lo_vs_symbolic`/`ci_hi_vs_symbolic`
are item (b) — context, not the gate — showing whether any audio-vs-symbolic
gap is attributable to audio provenance rather than the subset being
easier or harder.

Also compute MIDI drift per piece (`realaudio_check.midi_drift`) against the
stored Transkun MIDIs at `model/data/results/amt_gap_curve/transkun_mid/` to
confirm any audio-vs-symbolic gap is attributable to audio provenance, not
transcription failure on this subset specifically.

## If both gates pass

Report the measured deltas (not the FLOP-derived estimates) in
`docs/mirex/track-a-difficulty-prediction.md`'s decision log, per the design
spec's File Changes table. That edit is out of this plan's scope (it happens
at ship time, once real numbers exist).

## If either gate fails

Report the negative result plainly. A `noise` verdict on gate (i) or a
`ci_lo_vs_features37 <= 0` on gate (ii) is a real finding — #137's own history
is seven converging nulls; an eighth is not a failure of this plan, it is data.
```

- [ ] **Step 2: No test to run — this is documentation.** Sanity-check by grep
  that every CLI flag mentioned resolves to a real `argparse.add_argument` call
  in the corresponding module (manual review against Tasks 12, 15, 17, 23 above).

- [ ] **Step 3: N/A**

- [ ] **Step 4: Final full-suite confirmation**

```bash
cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
```
Expected: PASS, 65 passed (unchanged from Task 23 — this task adds no code).

- [ ] **Step 5: Commit**

```bash
git add docs/mirex/phase1-lora-runbook.md \
    && git commit -m "docs(#149): #138 Phase 1 LoRA runbook -- staged hf jobs sequence + abort criteria"
```

---

## Plan Self-Review

1. **Spec coverage.** Every module in the spec's Modules section has at least
   one task group: `fold_plan.py` (A), `ranking_loss.py` (B), `train_fold.py`
   (C), `ft_eval.py` (D), `realaudio_check.py` (E), `push_train_dataset.py` (F).
   Task Group 0 (paired_boot promotion) is Task 1. The runbook is Task 24. The
   spec's File Changes table is covered file-for-file.
2. **Placeholder scan.** No task contains "TBD"/"TODO"/"implement later"; every
   step has literal code or an exact command. The two module-start tasks (11,
   13, 16, 20 for train_fold/ft_eval/realaudio_check/push_train_dataset) each
   end their file with an explicit `if __name__ == "__main__": sys.exit(0)  #
   placeholder exit; main() is added in Task N` comment — this is intentional
   scaffolding within a single vertical slice sequence (the file is genuinely
   incomplete until its `main()` task lands later in the SAME group), not an
   unfinished deliverable left dangling across groups.
3. **Type/signature consistency.** `FoldPlan(fold, test_seg_ids, train_seg_ids,
   val_seg_ids)` is used identically in Tasks 2-5, 12 (JSON round-trip), 20-23.
   `lora_target_modules(n_layers, n_top)` in Task 11 is called with the same
   argument names in Task 12's CLI (`--n-layers`, `--n-top-layers`). `paired_boot`'s
   signature `(oof_a, oof_b, y, seed=2026, n_boot=2000)` is identical across
   Task 1, Task 15, Task 18, and Task 19 — Task 19 also adds the features37-paired
   arm required by the design spec's "Real-audio second gate", scored through
   `bakeoff_cv.composer_disjoint_folds` at the same seed the moonbeam arm uses.
   `read_fold_embeddings`/`write_fold_embeddings` in `train_fold.py` (Task 12)
   are imported unchanged by `ft_eval.py` (Task 15).
4. **Group correctness.** No two tasks in the same group touch the same file
   concurrently — each group is stated as sequential-internally precisely
   because its tasks share one file; groups that DO run concurrently (0 vs
   nothing before it; A vs B; {C,D} vs E vs F) touch disjoint files, confirmed
   against the File Changes table.
5. **Vertical slice check.** Every task is one test (or one already-passing
   test added for regression-proofing, explicitly called out as such in Tasks
   5, 14, 17, 21, 22) + one implementation + one commit. Task 1 is the sole
   deliberate exception (bundles two tests plus a real-data script re-run) —
   this matches the spec's own explicit framing of "Task Group 0" as one
   harness deliverable, not an oversight.
6. **Behavior test check.** No test mocks an internal collaborator: every
   injection point (`loader_factory`, `transcriber`, `uploader`) is a fake at
   the PROCESS BOUNDARY (a fork's checkpoint loader, `transkun_cli.transcribe_wav`,
   `huggingface_hub`'s upload), exactly matching `moonbeam_extract_script.py`'s
   and `moonbeam_backbone.py`'s established pattern. No test asserts on private
   state or calls a private method directly (`_carve_val`, `_real_loader`,
   `_ridge_oof`, `_write_cache_atomic`, etc. are all exercised only through
   their public callers).
