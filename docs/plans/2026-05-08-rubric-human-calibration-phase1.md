# Rubric Human Calibration Protocol — Phase 1 Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task). Within a group, tasks touching different files run in parallel; tasks touching the same file are explicitly chained via "depends on Task N." Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** Produce a frozen `filter_recipe.py` artifact that tells the Stage 2 SFT data pipeline which of the 14 v2-rubric sub-scores are trustworthy, at what weight, and with what bias correction — derived from founder ratings on 200 stratified syntheses against the LLM judge.
**Spec:** `docs/specs/2026-05-08-rubric-human-calibration-phase1-design.md`
**Style:** Follow `CLAUDE.md` (project-root) and user preferences: explicit exception handling, no emojis, no fallback mechanisms, `uv` for Python package management.

## Task Groups

```
Group A (parallel, no shared files):
  T1 era_lookup        T2 judge_rerun         T3 select_sample.band
  T4 rater_cli.blinding  T5 analyze_drift     T6 analyze_calibration.kappa

Group B (depends on A; parallel across separate files):
  T7 select_sample.era_quotas (depends T3)
  T8 rater_cli.capture (depends T4)
  T9 analyze_calibration.threshold_agree (depends T6)

Group C (depends on B; parallel across separate files):
  T10 select_sample.holdout (depends T7)
  T11 rater_cli.session_cap (depends T8)
  T12 analyze_calibration.bucket_routing (depends T9)

Group D (depends on C; parallel across separate files):
  T13 select_sample.anchor_injection (depends T10)
  T14 rater_cli.resume (depends T11)
  T15 emit_recipe.base (depends T12)

Group E (depends on D):
  T16 emit_recipe.bias_correction (depends T15)
  T17 select_sample.skill_quotas    (depends T13)
  T18 analyze_drift.judge_drift     (depends T5; consumes T2 output)
```

> **Blocker resolution amendment (2026-05-08):** Validated against real
> `apps/evals/results/baseline_v1.jsonl` (920 valid rows). Changes:
>
> - **Band redefinition.** `low` is now `composite < 2.3` (was `≤ 2.0`).
>   The (2.0, 2.3) gap that excluded 188 rows is folded into `low`, growing
>   the low pool from 40 → ~228. Rows in (2.0, 2.3) are still below the
>   2.5 pass threshold and useful as low-anchor examples; the gap added no
>   information.
> - **Band targets rebalanced** to `{threshold:80, high:40, low:30, weak_dim:50}`
>   (from 80/40/40/40). The 10 slots shifted from `low` to `weak_dim` go to
>   the band that already covers the baseline's weakest dimension (ASCF,
>   per project memory) and that has 95 candidates of slack.
> - **Era quotas unchanged at 30/era**, feasible after band redefinition
>   (Impressionist 42 total → ~40+ after holdout draw).
> - **T17 (new) — skill-bucket quotas:** beginner(1-2)≥50, intermediate(3)≥50,
>   advanced(4-5)≥50. Restores the spec requirement the original plan dropped.
> - **T18 (new) — judge drift κ:** `analyze_drift` consumes `judge_runs_path`
>   and emits `judge_drift_kappa` per sub-score. Restores the day-1-vs-day-30
>   gate the original plan stubbed as `{}`.

**Test runner (used in every task):** `cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest -xvs teacher_model/calibration/tests/<file>::<test_name>`

**Common scaffolding required before any task:** create `apps/evals/teacher_model/calibration/__init__.py`, `apps/evals/teacher_model/calibration/tests/__init__.py`, and `apps/evals/teacher_model/calibration/artifacts/.gitkeep` as empty files. The first task to run creates these; subsequent tasks assume their existence. T1 owns this scaffolding.

---

## Sub-score ID convention (used across all tasks)

The 14 sub-scores use these stable string IDs:

```
ascf_process,                concrete_artifact_process,    praise_process,
autonomy_process,            scaffolded_process,           style_process,
tone_process,
ascf_outcome,                concrete_artifact_outcome,    praise_outcome,
autonomy_outcome,            scaffolded_outcome,           style_outcome,
tone_outcome
```

Phase 1 calibrates 11: all 7 `*_process` + `autonomy_outcome` + `tone_outcome` + `concrete_artifact_outcome` + `praise_outcome`.
Phase 2 (deferred): `ascf_outcome`, `scaffolded_outcome`, `style_outcome`.

The mapping from the `criterion` field in `baseline_v1.jsonl` to these IDs:

```
"Audible-Specific Corrective Feedback" -> ascf
"Concrete Artifact Provision"          -> concrete_artifact
"Specific Positive Praise"             -> praise
"Autonomy-Supporting Motivation"       -> autonomy
"Scaffolded Guided Discovery"          -> scaffolded
"Style-Consistent Musical Language"    -> style
"Appropriate Tone & Language"          -> tone
```

---

## Synth ID convention

Each row in `apps/evals/results/baseline_v1.jsonl` is uniquely keyed by `(piece_slug, recording_id, skill_bucket)`. The string `synth_id` used throughout the protocol is:

```
synth_id = f"{piece_slug}__{recording_id}__{skill_bucket}"
```

---

## Task 1: era_lookup classifies known composers and unknowns
**Group:** A (parallel; no dependencies)

**Behavior being verified:** Given a composer string, the function returns the correct musical era. Bach→Baroque, Beethoven→Classical, Chopin→Romantic, Debussy→Impressionist, anything else→Other.

**Interface under test:** `composer_to_era(composer: str) -> str`

**Files:**
- Create: `apps/evals/teacher_model/calibration/__init__.py`
- Create: `apps/evals/teacher_model/calibration/tests/__init__.py`
- Create: `apps/evals/teacher_model/calibration/artifacts/.gitkeep`
- Create: `apps/evals/teacher_model/calibration/era_lookup.py`
- Test: `apps/evals/teacher_model/calibration/tests/test_era_lookup.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/teacher_model/calibration/tests/test_era_lookup.py
from teacher_model.calibration.era_lookup import composer_to_era


def test_known_composers_map_to_expected_eras():
    assert composer_to_era("Bach") == "Baroque"
    assert composer_to_era("Beethoven") == "Classical"
    assert composer_to_era("Chopin") == "Romantic"
    assert composer_to_era("Debussy") == "Impressionist"


def test_unknown_composer_returns_other():
    assert composer_to_era("Stravinsky") == "Other"
    assert composer_to_era("") == "Other"


def test_known_composers_are_case_sensitive_match():
    # Spec choice: exact-case match only, since baseline_v1.jsonl uses canonical
    # capitalization. Lowercased input is treated as unknown rather than silently
    # normalized — surfaces data quality issues instead of hiding them.
    assert composer_to_era("bach") == "Other"
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest -xvs teacher_model/calibration/tests/test_era_lookup.py
```
Expected: FAIL — `ModuleNotFoundError: No module named 'teacher_model.calibration'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# apps/evals/teacher_model/calibration/__init__.py
# (empty file)
```

```python
# apps/evals/teacher_model/calibration/tests/__init__.py
# (empty file)
```

```text
# apps/evals/teacher_model/calibration/artifacts/.gitkeep
```

```python
# apps/evals/teacher_model/calibration/era_lookup.py
"""Map composer string to musical era for sample-stratification quotas.

Only the 4 composers present in apps/evals/results/baseline_v1.jsonl are
recognized. Unknown composers map to "Other" and do not contribute to era
quotas in select_sample.
"""
from __future__ import annotations

_COMPOSER_TO_ERA: dict[str, str] = {
    "Bach": "Baroque",
    "Beethoven": "Classical",
    "Chopin": "Romantic",
    "Debussy": "Impressionist",
}


def composer_to_era(composer: str) -> str:
    return _COMPOSER_TO_ERA.get(composer, "Other")
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest -xvs teacher_model/calibration/tests/test_era_lookup.py
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/calibration/__init__.py apps/evals/teacher_model/calibration/tests/__init__.py apps/evals/teacher_model/calibration/artifacts/.gitkeep apps/evals/teacher_model/calibration/era_lookup.py apps/evals/teacher_model/calibration/tests/test_era_lookup.py && git commit -m "feat(calibration): add composer-to-era lookup"
```

---

## Task 2: judge_rerun emits jsonl matching the v2 judge output shape
**Group:** A (parallel with T1, T3, T4, T5, T6; no dependencies)

**Behavior being verified:** Given a list of `synth_id`s drawn from `baseline_v1.jsonl`, the rerun harness invokes `judge_synthesis_v2` for each and writes one jsonl record per synthesis. Each record contains `synth_id`, `run_label`, `dimensions` (list of {criterion, process, outcome, score}), `ts`. A pluggable `judge_callable` parameter allows tests to inject a stub instead of hitting the live API.

**Interface under test:** `rerun_anchors(anchor_synth_ids: list[str], baseline_path: Path, output_path: Path, run_label: str, judge_callable: Callable | None = None) -> None`

**Files:**
- Create: `apps/evals/teacher_model/calibration/judge_rerun.py`
- Test: `apps/evals/teacher_model/calibration/tests/test_judge_rerun.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/teacher_model/calibration/tests/test_judge_rerun.py
from __future__ import annotations

import json
from pathlib import Path

from teacher_model.calibration.judge_rerun import rerun_anchors


def _write_baseline(path: Path) -> None:
    rows = [
        {
            "piece_slug": "nocturne_op9no2",
            "recording_id": "rec_abc",
            "skill_bucket": 5,
            "composer": "Chopin",
            "title": "Chopin Nocturne",
            "synthesis_text": "Your pedaling sang. Try a gentle swell in mm. 3-4.",
            "muq_means": {"dynamics": 0.49},
            "judge_dimensions": [],
        },
        {
            "piece_slug": "wtc_book1_no1",
            "recording_id": "rec_xyz",
            "skill_bucket": 3,
            "composer": "Bach",
            "title": "Bach WTC Bk1 No1",
            "synthesis_text": "The voicing was clear in the opening.",
            "muq_means": {"dynamics": 0.55},
            "judge_dimensions": [],
        },
    ]
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _stub_judge(synthesis_text: str, context: dict) -> dict:
    # Returns the same shape `judge_synthesis_v2` returns: a JudgeResultV2-like
    # dict containing list of dimension dicts. We use a dict here so the test
    # has no dependency on the JudgeResultV2 dataclass.
    return {
        "dimensions": [
            {"criterion": "Audible-Specific Corrective Feedback",
             "process": 2, "outcome": 1, "score": 1,
             "evidence": "mm. 3-4", "reason": "stub"},
        ],
        "model": "stub",
        "prompt_version": "synthesis_quality_judge_v2",
        "latency_ms": 1.0,
    }


def test_rerun_anchors_writes_one_jsonl_record_per_anchor(tmp_path: Path):
    baseline_path = tmp_path / "baseline.jsonl"
    output_path = tmp_path / "judge_runs.jsonl"
    _write_baseline(baseline_path)

    rerun_anchors(
        anchor_synth_ids=[
            "nocturne_op9no2__rec_abc__5",
            "wtc_book1_no1__rec_xyz__3",
        ],
        baseline_path=baseline_path,
        output_path=output_path,
        run_label="day1",
        judge_callable=_stub_judge,
    )

    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(records) == 2
    ids = {r["synth_id"] for r in records}
    assert ids == {"nocturne_op9no2__rec_abc__5", "wtc_book1_no1__rec_xyz__3"}
    assert all(r["run_label"] == "day1" for r in records)
    assert all("dimensions" in r and "ts" in r for r in records)
    assert records[0]["dimensions"][0]["criterion"] == "Audible-Specific Corrective Feedback"


def test_rerun_anchors_raises_on_unknown_synth_id(tmp_path: Path):
    baseline_path = tmp_path / "baseline.jsonl"
    output_path = tmp_path / "judge_runs.jsonl"
    _write_baseline(baseline_path)

    try:
        rerun_anchors(
            anchor_synth_ids=["does_not_exist__nope__1"],
            baseline_path=baseline_path,
            output_path=output_path,
            run_label="day1",
            judge_callable=_stub_judge,
        )
    except KeyError as e:
        assert "does_not_exist__nope__1" in str(e)
        return
    raise AssertionError("Expected KeyError for unknown synth_id")
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest -xvs teacher_model/calibration/tests/test_judge_rerun.py
```
Expected: FAIL — `ModuleNotFoundError: No module named 'teacher_model.calibration.judge_rerun'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# apps/evals/teacher_model/calibration/judge_rerun.py
"""Re-run the v2 judge on a fixed set of anchor syntheses at a labeled time point.

Used to compute judge-vs-judge κ (day1 vs day30) for the protocol's drift gate.
The judge_callable parameter is injectable so tests can stub the network call;
in production, callers pass shared.judge.judge_synthesis_v2 (wrapped) here.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


def _build_baseline_index(baseline_path: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    with baseline_path.open() as f:
        for line in f:
            row = json.loads(line)
            if row.get("error"):
                continue
            synth_id = f"{row['piece_slug']}__{row['recording_id']}__{row['skill_bucket']}"
            index[synth_id] = row
    return index


def rerun_anchors(
    anchor_synth_ids: list[str],
    baseline_path: Path,
    output_path: Path,
    run_label: str,
    judge_callable: Callable[[str, dict], dict] | None = None,
) -> None:
    if judge_callable is None:
        raise ValueError(
            "judge_callable must be provided. In production pass a wrapper around "
            "shared.judge.judge_synthesis_v2 that returns a dict; tests pass a stub."
        )

    index = _build_baseline_index(baseline_path)

    with output_path.open("w") as out:
        for synth_id in anchor_synth_ids:
            if synth_id not in index:
                raise KeyError(f"synth_id not found in baseline: {synth_id}")
            row = index[synth_id]
            context = {
                "piece_name": row.get("title", "Unknown"),
                "composer": row.get("composer", "Unknown"),
                "skill_level": row.get("skill_bucket", "Unknown"),
            }
            result = judge_callable(row["synthesis_text"], context)
            record = {
                "synth_id": synth_id,
                "run_label": run_label,
                "dimensions": result["dimensions"],
                "model": result.get("model", ""),
                "prompt_version": result.get("prompt_version", ""),
                "latency_ms": result.get("latency_ms", 0.0),
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            out.write(json.dumps(record) + "\n")
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest -xvs teacher_model/calibration/tests/test_judge_rerun.py
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/calibration/judge_rerun.py apps/evals/teacher_model/calibration/tests/test_judge_rerun.py && git commit -m "feat(calibration): add judge_rerun harness with injectable judge callable"
```

---

## Task 3: select_sample produces correct band proportions
**Group:** A (parallel with T1, T2, T4, T5, T6; no dependencies)

**Behavior being verified:** Given a `baseline_v1.jsonl`-shaped source, `select_sample` returns a manifest where the main 200 has 80 in the threshold band (composite 2.3–2.7), 40 in high (≥2.7), 30 in low (composite < 2.3), 50 in weak-dim band (ASCF process ≤ 1), each within ±2 entries of target counts.

**Interface under test:** `select_sample(source_path: Path, target_n: int, holdout_n: int, anchor_n: int, seed: int) -> dict`

**Files:**
- Create: `apps/evals/teacher_model/calibration/select_sample.py`
- Test: `apps/evals/teacher_model/calibration/tests/test_select_sample.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/teacher_model/calibration/tests/test_select_sample.py
from __future__ import annotations

import json
import random
from pathlib import Path

from teacher_model.calibration.select_sample import select_sample


def _write_synthetic_baseline(path: Path, n_per_band: int = 300) -> None:
    """Generate 4*n_per_band syntheses with a controlled composite-score distribution.

    Bands engineered so each downstream band has well over the 40/20/20/20 target
    even after the era and skill quota constraints kick in.
    """
    rng = random.Random(7)
    rows: list[dict] = []
    composers = ["Bach", "Beethoven", "Chopin", "Debussy"]
    skill_buckets = [1, 2, 3, 4, 5]

    def _make_row(composite: float, ascf_process: int, idx: int) -> dict:
        # Build judge_dimensions with all 7 dims; only ASCF process and the
        # composite (mean of process+outcome across dims) matter for selection.
        # Use the composite to set every dim's process and outcome to that score.
        score = max(0, min(3, round(composite)))
        dims = [
            {"criterion": "Audible-Specific Corrective Feedback",
             "process": ascf_process, "outcome": score, "score": min(ascf_process, score),
             "evidence": "", "reason": ""},
            {"criterion": "Concrete Artifact Provision",
             "process": score, "outcome": score, "score": score,
             "evidence": "", "reason": ""},
            {"criterion": "Specific Positive Praise",
             "process": score, "outcome": score, "score": score,
             "evidence": "", "reason": ""},
            {"criterion": "Autonomy-Supporting Motivation",
             "process": score, "outcome": score, "score": score,
             "evidence": "", "reason": ""},
            {"criterion": "Scaffolded Guided Discovery",
             "process": score, "outcome": score, "score": score,
             "evidence": "", "reason": ""},
            {"criterion": "Style-Consistent Musical Language",
             "process": score, "outcome": score, "score": score,
             "evidence": "", "reason": ""},
            {"criterion": "Appropriate Tone & Language",
             "process": score, "outcome": score, "score": score,
             "evidence": "", "reason": ""},
        ]
        return {
            "piece_slug": f"piece_{idx}",
            "recording_id": f"rec_{idx}",
            "skill_bucket": rng.choice(skill_buckets),
            "composer": rng.choice(composers),
            "title": f"Piece {idx}",
            "synthesis_text": f"Synthesis text for piece {idx}.",
            "muq_means": {"dynamics": 0.5},
            "judge_dimensions": dims,
        }

    idx = 0
    # High band: composite >= 2.7
    for _ in range(n_per_band):
        rows.append(_make_row(composite=2.85, ascf_process=3, idx=idx)); idx += 1
    # Threshold band: 2.3 <= composite <= 2.7
    for _ in range(n_per_band):
        rows.append(_make_row(composite=2.5, ascf_process=2, idx=idx)); idx += 1
    # Low band: composite <= 2.0
    for _ in range(n_per_band):
        rows.append(_make_row(composite=1.8, ascf_process=2, idx=idx)); idx += 1
    # Weak-dim band: ASCF process <= 1.5 (composite is anywhere)
    for _ in range(n_per_band):
        rows.append(_make_row(composite=2.4, ascf_process=1, idx=idx)); idx += 1

    rng.shuffle(rows)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_select_sample_band_proportions_within_tolerance(tmp_path: Path):
    source = tmp_path / "baseline.jsonl"
    _write_synthetic_baseline(source)

    manifest = select_sample(
        source_path=source,
        target_n=200,
        holdout_n=30,
        anchor_n=20,
        seed=42,
    )

    band_counts = manifest["stats"]["band_counts"]
    # Targets: {threshold:80, high:40, low:30, weak_dim:50} = 200
    assert abs(band_counts["threshold"] - 80) <= 2, band_counts
    assert abs(band_counts["high"] - 40) <= 2, band_counts
    assert abs(band_counts["low"] - 30) <= 2, band_counts
    assert abs(band_counts["weak_dim"] - 50) <= 2, band_counts
    assert sum(band_counts.values()) == 200


def test_select_sample_is_deterministic_for_fixed_seed(tmp_path: Path):
    source = tmp_path / "baseline.jsonl"
    _write_synthetic_baseline(source)

    m1 = select_sample(source_path=source, target_n=200, holdout_n=30, anchor_n=20, seed=42)
    m2 = select_sample(source_path=source, target_n=200, holdout_n=30, anchor_n=20, seed=42)

    ids1 = [e["synth_id"] for e in m1["main"]]
    ids2 = [e["synth_id"] for e in m2["main"]]
    assert ids1 == ids2
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest -xvs teacher_model/calibration/tests/test_select_sample.py
```
Expected: FAIL — `ModuleNotFoundError: No module named 'teacher_model.calibration.select_sample'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# apps/evals/teacher_model/calibration/select_sample.py
"""Stratified sample selector for the rubric calibration protocol.

Initial implementation: band stratification only. Later tasks layer on era
quotas (T7), holdout reservation (T10), and anchor injection (T13).
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

_BAND_TARGETS: dict[str, int] = {
    "threshold": 80,
    "high": 40,
    "low": 30,
    "weak_dim": 50,
}


def _row_synth_id(row: dict[str, Any]) -> str:
    return f"{row['piece_slug']}__{row['recording_id']}__{row['skill_bucket']}"


def _row_composite(row: dict[str, Any]) -> float | None:
    scores = [
        d["score"] for d in row.get("judge_dimensions", [])
        if d.get("score") is not None
    ]
    if not scores:
        return None
    return sum(scores) / len(scores)


def _row_ascf_process(row: dict[str, Any]) -> int | None:
    for d in row.get("judge_dimensions", []):
        if d.get("criterion") == "Audible-Specific Corrective Feedback":
            return d.get("process")
    return None


def _classify_band(row: dict[str, Any]) -> str | None:
    """Return the band membership for the row, or None if it has no judge data.

    Order matters: weak_dim takes priority over composite-band when ASCF
    process <= 1 because the protocol explicitly oversamples weak-ASCF cases
    regardless of where their composite lands.

    Bands partition the composite range with no gap: high ≥ 2.7,
    threshold ∈ [2.3, 2.7), low < 2.3.
    """
    ascf_p = _row_ascf_process(row)
    if ascf_p is not None and ascf_p <= 1:
        return "weak_dim"
    composite = _row_composite(row)
    if composite is None:
        return None
    if composite >= 2.7:
        return "high"
    if composite >= 2.3:
        return "threshold"
    return "low"


def _load_valid_rows(source_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with source_path.open() as f:
        for line in f:
            row = json.loads(line)
            if row.get("error"):
                continue
            if not row.get("judge_dimensions"):
                continue
            rows.append(row)
    return rows


def select_sample(
    source_path: Path,
    target_n: int,
    holdout_n: int,
    anchor_n: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    rows = _load_valid_rows(source_path)

    by_band: dict[str, list[dict[str, Any]]] = {b: [] for b in _BAND_TARGETS}
    for r in rows:
        band = _classify_band(r)
        if band is None:
            continue
        by_band[band].append(r)

    if sum(_BAND_TARGETS.values()) != target_n:
        raise ValueError(
            f"Band targets sum to {sum(_BAND_TARGETS.values())} but target_n={target_n}. "
            f"Update _BAND_TARGETS or pass matching target_n."
        )
    band_targets = dict(_BAND_TARGETS)

    main: list[dict[str, Any]] = []
    band_counts: dict[str, int] = {}
    for band, target in band_targets.items():
        pool = by_band[band]
        if len(pool) < target:
            raise ValueError(
                f"Band '{band}' has {len(pool)} rows but needs {target}. "
                f"Source pool too small or distribution too skewed."
            )
        rng.shuffle(pool)
        chosen = pool[:target]
        main.extend(chosen)
        band_counts[band] = len(chosen)

    rng.shuffle(main)

    return {
        "version": 1,
        "seed": seed,
        "source_path": str(source_path),
        "main": [
            {
                "synth_id": _row_synth_id(r),
                "band": _classify_band(r),
                "era": None,
                "skill_bucket": r["skill_bucket"],
                "is_anchor_seed": False,
                "anchor_position": None,
            }
            for r in main
        ],
        "anchors": [],
        "holdout": [],
        "stats": {
            "n_main": len(main),
            "n_anchors_silent_dups": 0,
            "n_holdout": 0,
            "band_counts": band_counts,
        },
    }
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest -xvs teacher_model/calibration/tests/test_select_sample.py
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/calibration/select_sample.py apps/evals/teacher_model/calibration/tests/test_select_sample.py && git commit -m "feat(calibration): select_sample band stratification"
```

---

## Task 4: rater_cli does not leak judge scores into output streams (security boundary)
**Group:** A (parallel with T1, T2, T3, T5, T6; no dependencies)

**Behavior being verified:** Under no execution path of the rating presentation flow does any `judge_composite`, `judge_per_dim`, `process`, `outcome`, or `score` value from the source data appear in the rater's output stream. This is the security boundary that protects the protocol from rater anchoring bias.

**Interface under test:** `redact_for_rater(row: dict) -> dict` — pure function used by the CLI to produce the redacted view shown to the rater. The CLI's display loop is built on top of this function in T8; testing the redactor in isolation is the surgical way to verify the boundary.

**Files:**
- Create: `apps/evals/teacher_model/calibration/rater_cli.py`
- Test: `apps/evals/teacher_model/calibration/tests/test_rater_cli.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/teacher_model/calibration/tests/test_rater_cli.py
from __future__ import annotations

import json
import re

from teacher_model.calibration.rater_cli import redact_for_rater


def test_redacted_view_strips_all_judge_score_fields():
    leak_sentinel = "LEAK_DETECTOR_42"
    row = {
        "synth_id": "p__r__3",
        "piece_slug": "p",
        "recording_id": "r",
        "skill_bucket": 3,
        "composer": "Chopin",
        "title": "Test piece",
        "synthesis_text": "Some teacher feedback here.",
        "muq_means": {"dynamics": 0.5},
        "judge_dimensions": [
            {"criterion": "Audible-Specific Corrective Feedback",
             "process": 2, "outcome": 1, "score": 1,
             "evidence": leak_sentinel, "reason": leak_sentinel},
        ],
        "judge_latency_ms": 1234.5,
        "judge_model": "claude-sonnet-4-6",
    }

    redacted = redact_for_rater(row)
    serialized = json.dumps(redacted)

    # No score values, no evidence, no reasons from the judge.
    assert leak_sentinel not in serialized
    assert "process" not in redacted.get("judge_dimensions", [{}])[0] if redacted.get("judge_dimensions") else True
    assert "judge_dimensions" not in redacted
    assert "judge_latency_ms" not in redacted
    assert "judge_model" not in redacted
    # Numeric fields that could anchor
    assert "score" not in serialized.lower() or "skill_bucket" in serialized  # skill_bucket allowed
    # Any field name containing "judge" must be absent
    assert not any("judge" in k.lower() for k in redacted.keys())


def test_redacted_view_keeps_what_rater_needs():
    row = {
        "synth_id": "p__r__3",
        "piece_slug": "p",
        "recording_id": "r",
        "skill_bucket": 3,
        "composer": "Chopin",
        "title": "Test piece",
        "synthesis_text": "Some teacher feedback here.",
        "muq_means": {"dynamics": 0.5},
        "judge_dimensions": [],
    }
    redacted = redact_for_rater(row)
    assert redacted["synth_id"] == "p__r__3"
    assert redacted["composer"] == "Chopin"
    assert redacted["title"] == "Test piece"
    assert redacted["skill_bucket"] == 3
    assert redacted["synthesis_text"] == "Some teacher feedback here."
    # muq_means is allowed because the constrained-rubric dims (Praise outcome,
    # Concrete Artifact outcome) use it as ground truth.
    assert redacted["muq_means"] == {"dynamics": 0.5}
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest -xvs teacher_model/calibration/tests/test_rater_cli.py
```
Expected: FAIL — `ModuleNotFoundError: No module named 'teacher_model.calibration.rater_cli'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# apps/evals/teacher_model/calibration/rater_cli.py
"""Founder rating CLI for the rubric calibration protocol.

The CLI presents one synthesis at a time to the founder, captures 11 sub-score
ratings + evidence quote + reason per synthesis, and writes append-only jsonl.

This file starts with the security-critical redaction function. Later tasks
add: T8 rating capture loop, T11 session cap, T14 resume-from-crash.
"""
from __future__ import annotations

from typing import Any

# Fields from the source row that the rater is allowed to see. Anything not
# in this allow-list is stripped. Allow-list (not deny-list) is the safer
# discipline: a future schema change in baseline_v1.jsonl that adds a new
# judge field stays redacted by default.
_RATER_VISIBLE_FIELDS: frozenset[str] = frozenset({
    "synth_id",
    "piece_slug",
    "recording_id",
    "skill_bucket",
    "composer",
    "title",
    "synthesis_text",
    "muq_means",
})


def redact_for_rater(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k in _RATER_VISIBLE_FIELDS}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest -xvs teacher_model/calibration/tests/test_rater_cli.py
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/calibration/rater_cli.py apps/evals/teacher_model/calibration/tests/test_rater_cli.py && git commit -m "feat(calibration): rater_cli judge-score redaction (security boundary)"
```

---

## Task 5: analyze_drift computes intra-rater κ on duplicate ratings
**Group:** A (parallel with T1, T2, T3, T4, T6; no dependencies)

**Behavior being verified:** Given a ratings.jsonl containing some synth_ids rated twice (anchor duplicates), `analyze_drift` returns an `intra_rater_kappa` per sub-score that exactly equals 1.0 when the two ratings agree on every sub-score, and is below 1.0 when they disagree.

**Interface under test:** `analyze_drift(ratings_path: Path, judge_runs_path: Path | None) -> dict`

**Files:**
- Create: `apps/evals/teacher_model/calibration/analyze_drift.py`
- Test: `apps/evals/teacher_model/calibration/tests/test_analyze_drift.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/teacher_model/calibration/tests/test_analyze_drift.py
from __future__ import annotations

import json
from pathlib import Path

from teacher_model.calibration.analyze_drift import analyze_drift


def _write_ratings(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_intra_rater_kappa_is_one_when_ratings_agree(tmp_path: Path):
    # Two anchored synth_ids, each rated twice with identical values.
    ratings = []
    for synth_id in ["A1", "A2", "A3", "A4", "A5"]:
        for occurrence in (1, 2):
            anchor_id = f"anchor_{synth_id}" if occurrence == 2 else None
            for sub_score, value in [
                ("ascf_process", 2),
                ("tone_outcome", 3),
                ("autonomy_process", 1),
            ]:
                ratings.append({
                    "event_type": "rating",
                    "synth_id": synth_id if occurrence == 1 else f"{synth_id}_dup",
                    "anchor_origin_id": synth_id if occurrence == 2 else None,
                    "sub_score": sub_score,
                    "value": value,
                    "evidence": "",
                    "reason": "",
                    "session_id": "S001",
                    "session_idx": 1,
                    "ts": "2026-05-08T00:00:00Z",
                })
    ratings_path = tmp_path / "ratings.jsonl"
    _write_ratings(ratings_path, ratings)

    report = analyze_drift(ratings_path=ratings_path, judge_runs_path=None)

    intra = report["intra_rater_kappa"]
    assert intra["ascf_process"] == 1.0
    assert intra["tone_outcome"] == 1.0
    assert intra["autonomy_process"] == 1.0


def test_intra_rater_kappa_drops_below_one_when_ratings_disagree(tmp_path: Path):
    ratings = []
    for i, synth_id in enumerate(["A1", "A2", "A3", "A4", "A5"]):
        # First occurrence
        ratings.append({
            "event_type": "rating",
            "synth_id": synth_id,
            "anchor_origin_id": None,
            "sub_score": "ascf_process",
            "value": 2,
            "evidence": "", "reason": "",
            "session_id": "S001", "session_idx": i + 1,
            "ts": "2026-05-08T00:00:00Z",
        })
        # Second (duplicate) occurrence — every other one disagrees by 1
        ratings.append({
            "event_type": "rating",
            "synth_id": f"{synth_id}_dup",
            "anchor_origin_id": synth_id,
            "sub_score": "ascf_process",
            "value": 2 if i % 2 == 0 else 3,
            "evidence": "", "reason": "",
            "session_id": "S002", "session_idx": i + 1,
            "ts": "2026-05-08T01:00:00Z",
        })
    ratings_path = tmp_path / "ratings.jsonl"
    _write_ratings(ratings_path, ratings)

    report = analyze_drift(ratings_path=ratings_path, judge_runs_path=None)
    assert report["intra_rater_kappa"]["ascf_process"] < 1.0
    assert report["intra_rater_kappa"]["ascf_process"] >= -1.0
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest -xvs teacher_model/calibration/tests/test_analyze_drift.py
```
Expected: FAIL — `ModuleNotFoundError: No module named 'teacher_model.calibration.analyze_drift'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# apps/evals/teacher_model/calibration/analyze_drift.py
"""Drift analysis for the rubric calibration protocol.

Intra-rater κ on anchor duplicates and (when judge_runs_path is provided)
judge-vs-judge κ on day1/day30 re-runs. Both share the same Cohen's
weighted-quadratic κ implementation defined locally; analyze_calibration
re-implements identically (intentional duplication keeps modules independent).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def cohens_weighted_kappa(rater_a: list[int], rater_b: list[int], k: int = 4) -> float:
    """Cohen's weighted κ with quadratic weights for a 0..(k-1) ordinal scale.

    Returns 1.0 for perfect agreement, ~0 for chance-level agreement, and
    can be negative for systematic disagreement. When marginals are degenerate
    (one rater outputs only one category), expected disagreement is 0 and the
    function returns 1.0 if observed disagreement is also 0, else float('nan').
    """
    if len(rater_a) != len(rater_b):
        raise ValueError(f"length mismatch: {len(rater_a)} vs {len(rater_b)}")
    n = len(rater_a)
    if n == 0:
        raise ValueError("empty input")

    confusion = [[0 for _ in range(k)] for _ in range(k)]
    for a, b in zip(rater_a, rater_b):
        if not (0 <= a < k and 0 <= b < k):
            raise ValueError(f"value out of range [0,{k}): a={a} b={b}")
        confusion[a][b] += 1

    marg_a = [sum(confusion[i][j] for j in range(k)) for i in range(k)]
    marg_b = [sum(confusion[i][j] for i in range(k)) for j in range(k)]

    denom = (k - 1) ** 2 if k > 1 else 1
    weights = [[((i - j) ** 2) / denom for j in range(k)] for i in range(k)]

    obs = sum(weights[i][j] * confusion[i][j] for i in range(k) for j in range(k)) / n
    exp = sum(
        weights[i][j] * marg_a[i] * marg_b[j] / (n * n)
        for i in range(k) for j in range(k)
    )
    if exp == 0:
        return 1.0 if obs == 0 else float("nan")
    return 1.0 - obs / exp


def analyze_drift(ratings_path: Path, judge_runs_path: Path | None) -> dict[str, Any]:
    pairs_by_sub_score: dict[str, list[tuple[int, int]]] = defaultdict(list)
    first_seen: dict[tuple[str, str], int] = {}

    with ratings_path.open() as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("event_type") != "rating":
                continue
            sub_score = rec["sub_score"]
            origin = rec.get("anchor_origin_id")
            if origin is None:
                first_seen[(rec["synth_id"], sub_score)] = rec["value"]
            else:
                first_value = first_seen.get((origin, sub_score))
                if first_value is None:
                    continue
                pairs_by_sub_score[sub_score].append((first_value, rec["value"]))

    intra_rater = {
        sub: cohens_weighted_kappa([a for a, _ in pairs], [b for _, b in pairs])
        for sub, pairs in pairs_by_sub_score.items()
        if pairs
    }

    return {
        "intra_rater_kappa": intra_rater,
        "judge_drift_kappa": {},
        "n_anchor_pairs": {sub: len(p) for sub, p in pairs_by_sub_score.items()},
    }
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest -xvs teacher_model/calibration/tests/test_analyze_drift.py
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/calibration/analyze_drift.py apps/evals/teacher_model/calibration/tests/test_analyze_drift.py && git commit -m "feat(calibration): analyze_drift intra-rater kappa"
```

---

## Task 6: analyze_calibration computes weighted κ correctly on known ground truth
**Group:** A (parallel with T1, T2, T3, T4, T5; no dependencies)

**Behavior being verified:** Given paired ratings with constructed κ ground truth, `calibrate` returns a per-sub-score κ matching the construction within ±0.01 for perfect agreement, ≤ -0.99 for perfect mirror disagreement on a symmetric distribution, and within ±0.05 of zero for independent uniform ratings at n=2000.

**Interface under test:** `calibrate(ratings_path: Path, baseline_path: Path) -> dict` — first iteration returns `{"per_sub_score_kappa": {...}}` only; threshold-decision agreement and bucket routing arrive in T9 and T12.

**Files:**
- Create: `apps/evals/teacher_model/calibration/analyze_calibration.py`
- Test: `apps/evals/teacher_model/calibration/tests/test_analyze_calibration.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/teacher_model/calibration/tests/test_analyze_calibration.py
from __future__ import annotations

import json
import random
from pathlib import Path

from teacher_model.calibration.analyze_calibration import calibrate


def _write_ratings(path: Path, pairs: list[tuple[int, int]], sub_score: str = "ascf_process") -> None:
    """Write ratings.jsonl + a matched baseline.jsonl such that calibrate can pair them.

    Each (rater_value, judge_value) pair becomes one synth_id with one
    rating event in ratings.jsonl and one judge_dimensions entry in baseline.
    """
    with path.open("w") as f:
        for i, (rv, _) in enumerate(pairs):
            rec = {
                "event_type": "rating",
                "synth_id": f"S{i:04d}",
                "anchor_origin_id": None,
                "sub_score": sub_score,
                "value": rv,
                "evidence": "", "reason": "",
                "session_id": "S001", "session_idx": i + 1,
                "ts": "2026-05-08T00:00:00Z",
            }
            f.write(json.dumps(rec) + "\n")


_DIM_SLUG_TO_CRITERION = {
    "ascf": "Audible-Specific Corrective Feedback",
    "concrete_artifact": "Concrete Artifact Provision",
    "praise": "Specific Positive Praise",
    "autonomy": "Autonomy-Supporting Motivation",
    "scaffolded": "Scaffolded Guided Discovery",
    "style": "Style-Consistent Musical Language",
    "tone": "Appropriate Tone & Language",
}


def _write_baseline(path: Path, pairs: list[tuple[int, int]], sub_score: str = "ascf_process") -> None:
    slug, leg = sub_score.rsplit("_", 1)
    criterion = _DIM_SLUG_TO_CRITERION[slug]
    with path.open("w") as f:
        for i, (_, jv) in enumerate(pairs):
            row = {
                "piece_slug": f"piece_{i}",
                "recording_id": f"rec_{i}",
                "skill_bucket": 3,
                "composer": "Chopin",
                "title": f"Piece {i}",
                "synthesis_text": "x",
                "muq_means": {"dynamics": 0.5},
                "judge_dimensions": [{
                    "criterion": criterion,
                    "process": jv if leg == "process" else jv,
                    "outcome": jv if leg == "outcome" else jv,
                    "score": jv,
                    "evidence": "", "reason": "",
                }],
            }
            # Fix synth_id alignment: rater wrote S{i:04d}; map by reconstructing
            # the same synth_id in baseline.
            row["piece_slug"] = "piece"
            row["recording_id"] = f"rec_{i:04d}"
            row["skill_bucket"] = 0  # so synth_id = "piece__rec_0001__0"
            f.write(json.dumps(row) + "\n")


def _rewrite_ratings_to_match(path: Path, pairs: list[tuple[int, int]], sub_score: str = "ascf_process") -> None:
    with path.open("w") as f:
        for i, (rv, _) in enumerate(pairs):
            rec = {
                "event_type": "rating",
                "synth_id": f"piece__rec_{i:04d}__0",
                "anchor_origin_id": None,
                "sub_score": sub_score,
                "value": rv,
                "evidence": "", "reason": "",
                "session_id": "S001", "session_idx": i + 1,
                "ts": "2026-05-08T00:00:00Z",
            }
            f.write(json.dumps(rec) + "\n")


def test_perfect_agreement_yields_kappa_one(tmp_path: Path):
    pairs = [(0, 0)] * 50 + [(1, 1)] * 50 + [(2, 2)] * 50 + [(3, 3)] * 50
    ratings_path = tmp_path / "ratings.jsonl"
    baseline_path = tmp_path / "baseline.jsonl"
    _rewrite_ratings_to_match(ratings_path, pairs)
    _write_baseline(baseline_path, pairs)

    report = calibrate(ratings_path=ratings_path, baseline_path=baseline_path)
    assert abs(report["per_sub_score_kappa"]["ascf_process"] - 1.0) < 0.01


def test_perfect_mirror_disagreement_yields_kappa_minus_one(tmp_path: Path):
    pairs = [(0, 3)] * 50 + [(1, 2)] * 50 + [(2, 1)] * 50 + [(3, 0)] * 50
    ratings_path = tmp_path / "ratings.jsonl"
    baseline_path = tmp_path / "baseline.jsonl"
    _rewrite_ratings_to_match(ratings_path, pairs)
    _write_baseline(baseline_path, pairs)

    report = calibrate(ratings_path=ratings_path, baseline_path=baseline_path)
    assert report["per_sub_score_kappa"]["ascf_process"] <= -0.99


def test_independent_uniform_ratings_yield_kappa_near_zero(tmp_path: Path):
    rng = random.Random(123)
    pairs = [(rng.randint(0, 3), rng.randint(0, 3)) for _ in range(2000)]
    ratings_path = tmp_path / "ratings.jsonl"
    baseline_path = tmp_path / "baseline.jsonl"
    _rewrite_ratings_to_match(ratings_path, pairs)
    _write_baseline(baseline_path, pairs)

    report = calibrate(ratings_path=ratings_path, baseline_path=baseline_path)
    k = report["per_sub_score_kappa"]["ascf_process"]
    assert abs(k) < 0.05, f"expected near-zero κ for independent ratings, got {k}"
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest -xvs teacher_model/calibration/tests/test_analyze_calibration.py
```
Expected: FAIL — `ModuleNotFoundError: No module named 'teacher_model.calibration.analyze_calibration'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# apps/evals/teacher_model/calibration/analyze_calibration.py
"""Calibration analysis: per-sub-score weighted κ + (later) threshold agreement
+ (later) 4-bucket routing.

This file grows across T6 (κ), T9 (threshold agreement), T12 (bucket routing).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

_CRITERION_TO_SLUG: dict[str, str] = {
    "Audible-Specific Corrective Feedback": "ascf",
    "Concrete Artifact Provision": "concrete_artifact",
    "Specific Positive Praise": "praise",
    "Autonomy-Supporting Motivation": "autonomy",
    "Scaffolded Guided Discovery": "scaffolded",
    "Style-Consistent Musical Language": "style",
    "Appropriate Tone & Language": "tone",
}


def _cohens_weighted_kappa(rater_a: list[int], rater_b: list[int], k: int = 4) -> float:
    if len(rater_a) != len(rater_b):
        raise ValueError(f"length mismatch: {len(rater_a)} vs {len(rater_b)}")
    n = len(rater_a)
    if n == 0:
        raise ValueError("empty input")
    confusion = [[0 for _ in range(k)] for _ in range(k)]
    for a, b in zip(rater_a, rater_b):
        if not (0 <= a < k and 0 <= b < k):
            raise ValueError(f"value out of range [0,{k}): a={a} b={b}")
        confusion[a][b] += 1
    marg_a = [sum(confusion[i][j] for j in range(k)) for i in range(k)]
    marg_b = [sum(confusion[i][j] for i in range(k)) for j in range(k)]
    denom = (k - 1) ** 2 if k > 1 else 1
    weights = [[((i - j) ** 2) / denom for j in range(k)] for i in range(k)]
    obs = sum(weights[i][j] * confusion[i][j] for i in range(k) for j in range(k)) / n
    exp = sum(
        weights[i][j] * marg_a[i] * marg_b[j] / (n * n)
        for i in range(k) for j in range(k)
    )
    if exp == 0:
        return 1.0 if obs == 0 else float("nan")
    return 1.0 - obs / exp


def _judge_value_for_sub_score(judge_dimensions: list[dict], sub_score: str) -> int | None:
    slug, leg = sub_score.rsplit("_", 1)
    criterion = next(
        (c for c, s in _CRITERION_TO_SLUG.items() if s == slug), None
    )
    if criterion is None:
        return None
    for d in judge_dimensions:
        if d.get("criterion") == criterion:
            v = d.get(leg)
            if v is None:
                return None
            return int(v)
    return None


def _build_baseline_index(baseline_path: Path) -> dict[str, dict]:
    index: dict[str, dict] = {}
    with baseline_path.open() as f:
        for line in f:
            row = json.loads(line)
            if row.get("error"):
                continue
            synth_id = f"{row['piece_slug']}__{row['recording_id']}__{row['skill_bucket']}"
            index[synth_id] = row
    return index


def calibrate(ratings_path: Path, baseline_path: Path) -> dict[str, Any]:
    baseline = _build_baseline_index(baseline_path)
    pairs_by_sub: dict[str, list[tuple[int, int]]] = defaultdict(list)

    with ratings_path.open() as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("event_type") != "rating":
                continue
            if rec.get("anchor_origin_id") is not None:
                continue  # anchor duplicates only feed analyze_drift
            row = baseline.get(rec["synth_id"])
            if row is None:
                continue
            jv = _judge_value_for_sub_score(row.get("judge_dimensions", []), rec["sub_score"])
            if jv is None:
                continue
            pairs_by_sub[rec["sub_score"]].append((rec["value"], jv))

    per_sub_score_kappa = {
        sub: _cohens_weighted_kappa([a for a, _ in pairs], [b for _, b in pairs])
        for sub, pairs in pairs_by_sub.items()
        if pairs
    }
    return {"per_sub_score_kappa": per_sub_score_kappa}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest -xvs teacher_model/calibration/tests/test_analyze_calibration.py
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/calibration/analyze_calibration.py apps/evals/teacher_model/calibration/tests/test_analyze_calibration.py && git commit -m "feat(calibration): per-sub-score weighted kappa"
```

---

## Task 7: select_sample enforces era min-quotas via era_lookup
**Group:** B (depends on T3 — same file `select_sample.py`)

**Behavior being verified:** When generating the main sample, every era present in the source pool with ≥30 candidates is represented by ≥30 manifest entries. `composer_to_era` from T1 is the era assignment.

**Interface under test:** `select_sample(...) -> dict` — same function as T3; the `manifest["main"][i]["era"]` field is now populated, and `manifest["stats"]["era_counts"]` exists.

**Files:**
- Modify: `apps/evals/teacher_model/calibration/select_sample.py`
- Modify: `apps/evals/teacher_model/calibration/tests/test_select_sample.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to: apps/evals/teacher_model/calibration/tests/test_select_sample.py
def test_era_min_quotas_satisfied(tmp_path: Path):
    source = tmp_path / "baseline.jsonl"
    _write_synthetic_baseline(source)

    manifest = select_sample(
        source_path=source, target_n=200, holdout_n=30, anchor_n=20, seed=42,
    )

    era_counts = manifest["stats"]["era_counts"]
    for era in ("Baroque", "Classical", "Romantic", "Impressionist"):
        assert era_counts.get(era, 0) >= 30, (era, era_counts)
    # Every main entry has an era field set
    assert all(e["era"] is not None for e in manifest["main"])
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest -xvs teacher_model/calibration/tests/test_select_sample.py::test_era_min_quotas_satisfied
```
Expected: FAIL — `KeyError: 'era_counts'` (the stats dict from T3 has no `era_counts` key)

- [ ] **Step 3: Implement the minimum to make the test pass**

Replace the body of `select_sample` in `apps/evals/teacher_model/calibration/select_sample.py` with the following (add the `composer_to_era` import; replace the per-band selection block with quota-aware selection; populate `era` and `era_counts`):

```python
# apps/evals/teacher_model/calibration/select_sample.py
"""Stratified sample selector for the rubric calibration protocol.

Band stratification (T3) + era min-quotas (this task). Holdout reservation
arrives in T10, anchor injection in T13.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from teacher_model.calibration.era_lookup import composer_to_era

_BAND_TARGETS: dict[str, int] = {
    "threshold": 80,
    "high": 40,
    "low": 30,
    "weak_dim": 50,
}

_ERA_MIN_QUOTAS: dict[str, int] = {
    "Baroque": 30,
    "Classical": 30,
    "Romantic": 30,
    "Impressionist": 30,
}


def _row_synth_id(row: dict[str, Any]) -> str:
    return f"{row['piece_slug']}__{row['recording_id']}__{row['skill_bucket']}"


def _row_composite(row: dict[str, Any]) -> float | None:
    scores = [
        d["score"] for d in row.get("judge_dimensions", [])
        if d.get("score") is not None
    ]
    if not scores:
        return None
    return sum(scores) / len(scores)


def _row_ascf_process(row: dict[str, Any]) -> int | None:
    for d in row.get("judge_dimensions", []):
        if d.get("criterion") == "Audible-Specific Corrective Feedback":
            return d.get("process")
    return None


def _classify_band(row: dict[str, Any]) -> str | None:
    ascf_p = _row_ascf_process(row)
    if ascf_p is not None and ascf_p <= 1:
        return "weak_dim"
    composite = _row_composite(row)
    if composite is None:
        return None
    if composite >= 2.7:
        return "high"
    if composite >= 2.3:
        return "threshold"
    return "low"


def _load_valid_rows(source_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with source_path.open() as f:
        for line in f:
            row = json.loads(line)
            if row.get("error"):
                continue
            if not row.get("judge_dimensions"):
                continue
            rows.append(row)
    return rows


def _select_with_quotas(
    by_band: dict[str, list[dict[str, Any]]],
    band_targets: dict[str, int],
    era_min_quotas: dict[str, int],
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Two-pass selection:
    1. Reserve era min-quotas first by drawing era-pure picks across bands.
    2. Fill remaining band slots from the unreserved pool.
    Raises ValueError if either constraint cannot be satisfied.
    """
    chosen: list[dict[str, Any]] = []
    chosen_ids: set[str] = set()
    era_counter: Counter[str] = Counter()
    band_counter: Counter[str] = Counter()

    # Pass 1: era reservation
    for era, min_q in era_min_quotas.items():
        # Pull candidates of this era from any band, preferring bands that are
        # furthest from their target so the reservation also helps band balance.
        candidates: list[tuple[str, dict[str, Any]]] = []
        for band, pool in by_band.items():
            for r in pool:
                if composer_to_era(r["composer"]) == era and _row_synth_id(r) not in chosen_ids:
                    candidates.append((band, r))
        rng.shuffle(candidates)
        if len(candidates) < min_q:
            raise ValueError(
                f"Era '{era}' has only {len(candidates)} candidates but needs {min_q}."
            )
        for band, r in candidates[:min_q]:
            if band_counter[band] >= band_targets[band]:
                continue  # band full, skip and revisit with broader pool below
            chosen.append(r)
            chosen_ids.add(_row_synth_id(r))
            era_counter[era] += 1
            band_counter[band] += 1
            if era_counter[era] >= min_q:
                break
        # If we did not hit min_q because some bands were full, fall back to
        # any band (allows era quota to push past per-band frac slightly):
        if era_counter[era] < min_q:
            for band, r in candidates:
                if _row_synth_id(r) in chosen_ids:
                    continue
                chosen.append(r)
                chosen_ids.add(_row_synth_id(r))
                era_counter[era] += 1
                band_counter[band] += 1
                if era_counter[era] >= min_q:
                    break

    # Pass 2: fill remaining band slots
    for band, target in band_targets.items():
        remaining = target - band_counter[band]
        if remaining <= 0:
            continue
        pool = [r for r in by_band[band] if _row_synth_id(r) not in chosen_ids]
        rng.shuffle(pool)
        if len(pool) < remaining:
            raise ValueError(
                f"Band '{band}' has only {len(pool)} unreserved rows but needs {remaining} more."
            )
        for r in pool[:remaining]:
            chosen.append(r)
            chosen_ids.add(_row_synth_id(r))
            band_counter[band] += 1
            era_counter[composer_to_era(r["composer"])] += 1

    return chosen


def select_sample(
    source_path: Path,
    target_n: int,
    holdout_n: int,
    anchor_n: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    rows = _load_valid_rows(source_path)

    by_band: dict[str, list[dict[str, Any]]] = {b: [] for b in _BAND_TARGETS}
    for r in rows:
        band = _classify_band(r)
        if band is None:
            continue
        by_band[band].append(r)

    if sum(_BAND_TARGETS.values()) != target_n:
        raise ValueError(
            f"Band targets sum to {sum(_BAND_TARGETS.values())} but target_n={target_n}."
        )
    band_targets = dict(_BAND_TARGETS)

    main = _select_with_quotas(by_band, band_targets, _ERA_MIN_QUOTAS, rng)
    rng.shuffle(main)

    band_counts = Counter(_classify_band(r) for r in main)
    era_counts = Counter(composer_to_era(r["composer"]) for r in main)

    return {
        "version": 1,
        "seed": seed,
        "source_path": str(source_path),
        "main": [
            {
                "synth_id": _row_synth_id(r),
                "band": _classify_band(r),
                "era": composer_to_era(r["composer"]),
                "skill_bucket": r["skill_bucket"],
                "is_anchor_seed": False,
                "anchor_position": None,
            }
            for r in main
        ],
        "anchors": [],
        "holdout": [],
        "stats": {
            "n_main": len(main),
            "n_anchors_silent_dups": 0,
            "n_holdout": 0,
            "band_counts": dict(band_counts),
            "era_counts": dict(era_counts),
        },
    }
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest -xvs teacher_model/calibration/tests/test_select_sample.py
```
Expected: PASS (all three test_select_sample tests, including pre-existing T3 ones)

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/calibration/select_sample.py apps/evals/teacher_model/calibration/tests/test_select_sample.py && git commit -m "feat(calibration): select_sample era min-quotas"
```

---

## Task 8: rater_cli captures one rating event per sub-score and appends to jsonl
**Group:** B (depends on T4 — same file `rater_cli.py`)

**Behavior being verified:** Given a manifest with one synthesis and a stub input provider, the rater capture loop produces 11 rating events (one per Phase-1 sub-score) appended to the output jsonl, each with `event_type="rating"`, the correct `synth_id`, the rater's value, and a session_id.

**Interface under test:** `capture_synthesis_ratings(redacted_row: dict, sub_scores: list[str], session_id: str, session_idx_start: int, output_path: Path, input_provider: Callable) -> int` — returns count of rating events written.

**Files:**
- Modify: `apps/evals/teacher_model/calibration/rater_cli.py`
- Modify: `apps/evals/teacher_model/calibration/tests/test_rater_cli.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to: apps/evals/teacher_model/calibration/tests/test_rater_cli.py
import json
from pathlib import Path
from teacher_model.calibration.rater_cli import (
    capture_synthesis_ratings,
    PHASE_1_SUB_SCORES,
)


def test_capture_writes_one_event_per_sub_score(tmp_path: Path):
    redacted = {
        "synth_id": "p__r__3",
        "synthesis_text": "x",
        "title": "Piece",
        "composer": "Chopin",
        "skill_bucket": 3,
    }
    output = tmp_path / "ratings.jsonl"

    inputs = iter([
        # Each sub-score: (value, evidence, reason)
        (2, "ev_ascf_p", "rs_ascf_p"),
        (3, "ev_ca_p",   "rs_ca_p"),
        (3, "ev_pr_p",   "rs_pr_p"),
        (3, "ev_au_p",   "rs_au_p"),
        (2, "ev_sc_p",   "rs_sc_p"),
        (3, "ev_st_p",   "rs_st_p"),
        (3, "ev_to_p",   "rs_to_p"),
        (3, "ev_au_o",   "rs_au_o"),
        (3, "ev_to_o",   "rs_to_o"),
        (2, "ev_ca_o",   "rs_ca_o"),
        (2, "ev_pr_o",   "rs_pr_o"),
    ])

    def provider(_redacted: dict, sub_score: str) -> tuple[int, str, str]:
        return next(inputs)

    n = capture_synthesis_ratings(
        redacted_row=redacted,
        sub_scores=PHASE_1_SUB_SCORES,
        session_id="S001",
        session_idx_start=1,
        output_path=output,
        input_provider=provider,
    )
    assert n == 11
    events = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(events) == 11
    assert all(e["event_type"] == "rating" for e in events)
    assert all(e["synth_id"] == "p__r__3" for e in events)
    assert all(e["session_id"] == "S001" for e in events)
    assert {e["sub_score"] for e in events} == set(PHASE_1_SUB_SCORES)
    assert events[0]["value"] == 2 and events[0]["evidence"] == "ev_ascf_p"


def test_phase_1_sub_scores_has_exactly_eleven():
    assert len(PHASE_1_SUB_SCORES) == 11
    # Exclude Phase-2 expert dims
    assert "ascf_outcome" not in PHASE_1_SUB_SCORES
    assert "scaffolded_outcome" not in PHASE_1_SUB_SCORES
    assert "style_outcome" not in PHASE_1_SUB_SCORES
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest -xvs teacher_model/calibration/tests/test_rater_cli.py::test_capture_writes_one_event_per_sub_score
```
Expected: FAIL — `ImportError: cannot import name 'capture_synthesis_ratings' from 'teacher_model.calibration.rater_cli'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Append to `apps/evals/teacher_model/calibration/rater_cli.py`:

```python
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

PHASE_1_SUB_SCORES: list[str] = [
    "ascf_process",
    "concrete_artifact_process",
    "praise_process",
    "autonomy_process",
    "scaffolded_process",
    "style_process",
    "tone_process",
    "autonomy_outcome",
    "tone_outcome",
    "concrete_artifact_outcome",
    "praise_outcome",
]


def capture_synthesis_ratings(
    redacted_row: dict,
    sub_scores: list[str],
    session_id: str,
    session_idx_start: int,
    output_path: Path,
    input_provider: Callable[[dict, str], tuple[int, str, str]],
) -> int:
    n_written = 0
    with output_path.open("a") as out:
        for sub_score in sub_scores:
            value, evidence, reason = input_provider(redacted_row, sub_score)
            event = {
                "event_type": "rating",
                "synth_id": redacted_row["synth_id"],
                "anchor_origin_id": redacted_row.get("anchor_origin_id"),
                "sub_score": sub_score,
                "value": value,
                "evidence": evidence,
                "reason": reason,
                "session_id": session_id,
                "session_idx": session_idx_start,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            out.write(json.dumps(event) + "\n")
            n_written += 1
    return n_written
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest -xvs teacher_model/calibration/tests/test_rater_cli.py
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/calibration/rater_cli.py apps/evals/teacher_model/calibration/tests/test_rater_cli.py && git commit -m "feat(calibration): rater_cli capture loop with phase-1 sub-scores"
```

---

## Task 9: analyze_calibration computes threshold-decision agreement
**Group:** B (depends on T6 — same file `analyze_calibration.py`)

**Behavior being verified:** When rater and judge agree on whether a synthesis's composite passes the 2.5 cutoff for ≥80% of syntheses, threshold agreement ≥ 0.80; when they disagree on every borderline case, threshold agreement < 0.80.

**Interface under test:** `calibrate(...)` now returns `{"per_sub_score_kappa": ..., "threshold_decision_agreement": float, "threshold_decision_kappa": float, "n_threshold_pairs": int}`.

The composite for the rater is computed as the mean of all Phase-1 sub-score values for that synth_id; the composite for the judge is the row's `score` mean over judge_dimensions (to mirror `apps/evals/results/baseline_v1_aggregate.json`).

**Files:**
- Modify: `apps/evals/teacher_model/calibration/analyze_calibration.py`
- Modify: `apps/evals/teacher_model/calibration/tests/test_analyze_calibration.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to: apps/evals/teacher_model/calibration/tests/test_analyze_calibration.py


def _write_full_synth(ratings_path: Path, baseline_path: Path,
                     synths: list[tuple[str, dict[str, int], dict[str, int]]]) -> None:
    """For each (synth_id, rater_values_per_sub, judge_values_per_dim), write
    the full rater + baseline records. rater_values_per_sub maps sub_score id
    -> rater 0..3. judge_values_per_dim maps the 7 dim slugs -> 0..3 used as
    both process and outcome.
    """
    with ratings_path.open("w") as rf:
        for synth_id, rater_vals, _ in synths:
            for sub, v in rater_vals.items():
                rf.write(json.dumps({
                    "event_type": "rating",
                    "synth_id": synth_id,
                    "anchor_origin_id": None,
                    "sub_score": sub,
                    "value": v,
                    "evidence": "", "reason": "",
                    "session_id": "S001", "session_idx": 1,
                    "ts": "x",
                }) + "\n")

    slug_to_criterion = {
        "ascf": "Audible-Specific Corrective Feedback",
        "concrete_artifact": "Concrete Artifact Provision",
        "praise": "Specific Positive Praise",
        "autonomy": "Autonomy-Supporting Motivation",
        "scaffolded": "Scaffolded Guided Discovery",
        "style": "Style-Consistent Musical Language",
        "tone": "Appropriate Tone & Language",
    }
    with baseline_path.open("w") as bf:
        for synth_id, _, judge_vals in synths:
            piece_slug, recording_id, skill = synth_id.split("__")
            dims = []
            for slug, criterion in slug_to_criterion.items():
                v = judge_vals.get(slug, 3)
                dims.append({
                    "criterion": criterion,
                    "process": v, "outcome": v, "score": v,
                    "evidence": "", "reason": "",
                })
            bf.write(json.dumps({
                "piece_slug": piece_slug,
                "recording_id": recording_id,
                "skill_bucket": int(skill),
                "composer": "Chopin",
                "title": "x",
                "synthesis_text": "x",
                "muq_means": {},
                "judge_dimensions": dims,
            }) + "\n")


def test_threshold_agreement_high_when_pass_decisions_align(tmp_path: Path):
    # 10 syntheses, all clearly passing (rater 3 across, judge 3 across) — agreement = 1.0
    synths = []
    for i in range(10):
        synth_id = f"piece__rec{i}__3"
        rv = {sub: 3 for sub in [
            "ascf_process", "concrete_artifact_process", "praise_process",
            "autonomy_process", "scaffolded_process", "style_process",
            "tone_process", "autonomy_outcome", "tone_outcome",
            "concrete_artifact_outcome", "praise_outcome",
        ]}
        jv = {s: 3 for s in ["ascf", "concrete_artifact", "praise", "autonomy",
                              "scaffolded", "style", "tone"]}
        synths.append((synth_id, rv, jv))
    ratings_path = tmp_path / "ratings.jsonl"
    baseline_path = tmp_path / "baseline.jsonl"
    _write_full_synth(ratings_path, baseline_path, synths)

    report = calibrate(ratings_path=ratings_path, baseline_path=baseline_path)
    assert report["threshold_decision_agreement"] == 1.0
    assert report["n_threshold_pairs"] == 10


def test_threshold_agreement_low_when_borderline_disagrees(tmp_path: Path):
    # 10 syntheses on rater side: alternating 3 (pass) and 2 (fail).
    # Judge side: opposite parity (alternating 2 and 3) — 100% disagreement.
    synths = []
    for i in range(10):
        synth_id = f"piece__rec{i}__3"
        rater_v = 3 if i % 2 == 0 else 2
        judge_v = 2 if i % 2 == 0 else 3
        rv = {sub: rater_v for sub in [
            "ascf_process", "concrete_artifact_process", "praise_process",
            "autonomy_process", "scaffolded_process", "style_process",
            "tone_process", "autonomy_outcome", "tone_outcome",
            "concrete_artifact_outcome", "praise_outcome",
        ]}
        jv = {s: judge_v for s in ["ascf", "concrete_artifact", "praise", "autonomy",
                                    "scaffolded", "style", "tone"]}
        synths.append((synth_id, rv, jv))
    ratings_path = tmp_path / "ratings.jsonl"
    baseline_path = tmp_path / "baseline.jsonl"
    _write_full_synth(ratings_path, baseline_path, synths)

    report = calibrate(ratings_path=ratings_path, baseline_path=baseline_path)
    assert report["threshold_decision_agreement"] == 0.0
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest -xvs teacher_model/calibration/tests/test_analyze_calibration.py::test_threshold_agreement_high_when_pass_decisions_align
```
Expected: FAIL — `KeyError: 'threshold_decision_agreement'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Replace the body of `calibrate` in `apps/evals/teacher_model/calibration/analyze_calibration.py` with:

```python
COMPOSITE_PASS_THRESHOLD: float = 2.5


def _rater_composite(values_by_sub: dict[str, int]) -> float | None:
    if not values_by_sub:
        return None
    return sum(values_by_sub.values()) / len(values_by_sub)


def _judge_composite(judge_dimensions: list[dict]) -> float | None:
    scores = [d["score"] for d in judge_dimensions if d.get("score") is not None]
    if not scores:
        return None
    return sum(scores) / len(scores)


def calibrate(ratings_path: Path, baseline_path: Path) -> dict[str, Any]:
    baseline = _build_baseline_index(baseline_path)
    pairs_by_sub: dict[str, list[tuple[int, int]]] = defaultdict(list)
    rater_vals_per_synth: dict[str, dict[str, int]] = defaultdict(dict)

    with ratings_path.open() as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("event_type") != "rating":
                continue
            if rec.get("anchor_origin_id") is not None:
                continue
            row = baseline.get(rec["synth_id"])
            if row is None:
                continue
            jv = _judge_value_for_sub_score(row.get("judge_dimensions", []), rec["sub_score"])
            if jv is not None:
                pairs_by_sub[rec["sub_score"]].append((rec["value"], jv))
            rater_vals_per_synth[rec["synth_id"]][rec["sub_score"]] = rec["value"]

    per_sub_score_kappa = {
        sub: _cohens_weighted_kappa([a for a, _ in pairs], [b for _, b in pairs])
        for sub, pairs in pairs_by_sub.items()
        if pairs
    }

    threshold_pairs: list[tuple[int, int]] = []
    for synth_id, vals in rater_vals_per_synth.items():
        row = baseline.get(synth_id)
        if row is None:
            continue
        rc = _rater_composite(vals)
        jc = _judge_composite(row.get("judge_dimensions", []))
        if rc is None or jc is None:
            continue
        rater_pass = 1 if rc >= COMPOSITE_PASS_THRESHOLD else 0
        judge_pass = 1 if jc >= COMPOSITE_PASS_THRESHOLD else 0
        threshold_pairs.append((rater_pass, judge_pass))

    if threshold_pairs:
        agreement = sum(1 for a, b in threshold_pairs if a == b) / len(threshold_pairs)
        a_vals = [a for a, _ in threshold_pairs]
        b_vals = [b for _, b in threshold_pairs]
        kappa = _cohens_weighted_kappa(a_vals, b_vals, k=2)
    else:
        agreement = 0.0
        kappa = float("nan")

    return {
        "per_sub_score_kappa": per_sub_score_kappa,
        "threshold_decision_agreement": agreement,
        "threshold_decision_kappa": kappa,
        "n_threshold_pairs": len(threshold_pairs),
    }
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest -xvs teacher_model/calibration/tests/test_analyze_calibration.py
```
Expected: PASS (all five tests in the file)

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/calibration/analyze_calibration.py apps/evals/teacher_model/calibration/tests/test_analyze_calibration.py && git commit -m "feat(calibration): threshold-decision agreement"
```

---

## Task 10: select_sample reserves the holdout-30 disjoint from main 200
**Group:** C (depends on T7 — same file `select_sample.py`)

**Behavior being verified:** `manifest["holdout"]` contains exactly `holdout_n` entries; no synth_id appears in both `main` and `holdout`; holdout is drawn from rows that classify into a band (so they are valid OFFSET re-validation candidates).

**Files:**
- Modify: `apps/evals/teacher_model/calibration/select_sample.py`
- Modify: `apps/evals/teacher_model/calibration/tests/test_select_sample.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to: apps/evals/teacher_model/calibration/tests/test_select_sample.py
def test_holdout_is_disjoint_from_main(tmp_path: Path):
    source = tmp_path / "baseline.jsonl"
    _write_synthetic_baseline(source)

    manifest = select_sample(
        source_path=source, target_n=200, holdout_n=30, anchor_n=20, seed=42,
    )

    main_ids = {e["synth_id"] for e in manifest["main"]}
    holdout_ids = {e["synth_id"] for e in manifest["holdout"]}

    assert len(holdout_ids) == 30
    assert main_ids.isdisjoint(holdout_ids)
    assert manifest["stats"]["n_holdout"] == 30
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest -xvs teacher_model/calibration/tests/test_select_sample.py::test_holdout_is_disjoint_from_main
```
Expected: FAIL — `assert len(holdout_ids) == 30` fails (holdout is empty in T7's manifest).

- [ ] **Step 3: Implement the minimum to make the test pass**

Replace the `select_sample` function body in `apps/evals/teacher_model/calibration/select_sample.py` with this version (key change: reserve holdout from valid-band rows BEFORE running `_select_with_quotas`):

```python
def select_sample(
    source_path: Path,
    target_n: int,
    holdout_n: int,
    anchor_n: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    rows = _load_valid_rows(source_path)

    valid_with_band: list[tuple[dict[str, Any], str]] = []
    for r in rows:
        band = _classify_band(r)
        if band is not None:
            valid_with_band.append((r, band))

    rng.shuffle(valid_with_band)
    if len(valid_with_band) < holdout_n + target_n:
        raise ValueError(
            f"Source has only {len(valid_with_band)} band-classified rows "
            f"but needs {holdout_n + target_n}."
        )

    holdout_rows = [r for r, _ in valid_with_band[:holdout_n]]
    remaining = valid_with_band[holdout_n:]

    by_band: dict[str, list[dict[str, Any]]] = {b: [] for b in _BAND_TARGETS}
    for r, band in remaining:
        by_band[band].append(r)

    if sum(_BAND_TARGETS.values()) != target_n:
        raise ValueError(
            f"Band targets sum to {sum(_BAND_TARGETS.values())} but target_n={target_n}."
        )
    band_targets = dict(_BAND_TARGETS)

    main = _select_with_quotas(by_band, band_targets, _ERA_MIN_QUOTAS, rng)
    rng.shuffle(main)

    band_counts = Counter(_classify_band(r) for r in main)
    era_counts = Counter(composer_to_era(r["composer"]) for r in main)

    return {
        "version": 1,
        "seed": seed,
        "source_path": str(source_path),
        "main": [
            {
                "synth_id": _row_synth_id(r),
                "band": _classify_band(r),
                "era": composer_to_era(r["composer"]),
                "skill_bucket": r["skill_bucket"],
                "is_anchor_seed": False,
                "anchor_position": None,
            }
            for r in main
        ],
        "anchors": [],
        "holdout": [
            {
                "synth_id": _row_synth_id(r),
                "band": _classify_band(r),
                "era": composer_to_era(r["composer"]),
                "skill_bucket": r["skill_bucket"],
            }
            for r in holdout_rows
        ],
        "stats": {
            "n_main": len(main),
            "n_anchors_silent_dups": 0,
            "n_holdout": len(holdout_rows),
            "band_counts": dict(band_counts),
            "era_counts": dict(era_counts),
        },
    }
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest -xvs teacher_model/calibration/tests/test_select_sample.py
```
Expected: PASS (all four select_sample tests)

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/calibration/select_sample.py apps/evals/teacher_model/calibration/tests/test_select_sample.py && git commit -m "feat(calibration): reserve holdout disjoint from main"
```

---

## Task 11: rater_cli enforces session cap of 15 ratings per session
**Group:** C (depends on T8 — same file `rater_cli.py`)

**Behavior being verified:** When `capture_synthesis_ratings` is called within a session that has already used `session_idx_start` slots, and emitting 11 more would exceed 15, the function raises `SessionCapExceeded` BEFORE writing any partial events — and the output file remains in a consistent state (no orphan rows for the rejected synthesis).

**Interface under test:** `capture_synthesis_ratings(...)` raises `SessionCapExceeded` exception type.

**Files:**
- Modify: `apps/evals/teacher_model/calibration/rater_cli.py`
- Modify: `apps/evals/teacher_model/calibration/tests/test_rater_cli.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to: apps/evals/teacher_model/calibration/tests/test_rater_cli.py
import pytest
from teacher_model.calibration.rater_cli import (
    capture_synthesis_ratings,
    SessionCapExceeded,
    PHASE_1_SUB_SCORES,
)


def test_session_cap_blocks_when_remaining_slots_insufficient(tmp_path: Path):
    # Cap is 15. Start at session_idx_start=10 — only 6 slots remain, but
    # capturing one synthesis writes 11 rating events, so this must raise
    # BEFORE writing anything.
    redacted = {"synth_id": "p__r__3", "synthesis_text": "x"}
    output = tmp_path / "ratings.jsonl"

    def provider(_, __):
        return (3, "e", "r")

    with pytest.raises(SessionCapExceeded):
        capture_synthesis_ratings(
            redacted_row=redacted,
            sub_scores=PHASE_1_SUB_SCORES,
            session_id="S001",
            session_idx_start=10,
            output_path=output,
            input_provider=provider,
        )

    # Output must be empty or non-existent — no partial writes
    if output.exists():
        assert output.read_text() == ""


def test_session_cap_allows_when_slots_sufficient(tmp_path: Path):
    redacted = {"synth_id": "p__r__3", "synthesis_text": "x"}
    output = tmp_path / "ratings.jsonl"

    def provider(_, __):
        return (3, "e", "r")

    n = capture_synthesis_ratings(
        redacted_row=redacted,
        sub_scores=PHASE_1_SUB_SCORES,
        session_id="S001",
        session_idx_start=1,
        output_path=output,
        input_provider=provider,
    )
    assert n == 11
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest -xvs teacher_model/calibration/tests/test_rater_cli.py::test_session_cap_blocks_when_remaining_slots_insufficient
```
Expected: FAIL — `ImportError: cannot import name 'SessionCapExceeded'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Edit `apps/evals/teacher_model/calibration/rater_cli.py`: add the `SessionCapExceeded` class, the `MAX_RATINGS_PER_SESSION` constant, and a pre-flight check at the top of `capture_synthesis_ratings`.

```python
MAX_RATINGS_PER_SESSION: int = 15


class SessionCapExceeded(Exception):
    """Raised when a synthesis would push a session past 15 rating events."""


def capture_synthesis_ratings(
    redacted_row: dict,
    sub_scores: list[str],
    session_id: str,
    session_idx_start: int,
    output_path: Path,
    input_provider: Callable[[dict, str], tuple[int, str, str]],
) -> int:
    needed = len(sub_scores)
    last_idx = session_idx_start + needed - 1
    if last_idx > MAX_RATINGS_PER_SESSION:
        raise SessionCapExceeded(
            f"Session {session_id} would reach idx {last_idx}, "
            f"exceeding cap of {MAX_RATINGS_PER_SESSION}. "
            f"Start a new session and continue."
        )

    n_written = 0
    with output_path.open("a") as out:
        for i, sub_score in enumerate(sub_scores):
            value, evidence, reason = input_provider(redacted_row, sub_score)
            event = {
                "event_type": "rating",
                "synth_id": redacted_row["synth_id"],
                "anchor_origin_id": redacted_row.get("anchor_origin_id"),
                "sub_score": sub_score,
                "value": value,
                "evidence": evidence,
                "reason": reason,
                "session_id": session_id,
                "session_idx": session_idx_start + i,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            out.write(json.dumps(event) + "\n")
            n_written += 1
    return n_written
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest -xvs teacher_model/calibration/tests/test_rater_cli.py
```
Expected: PASS (all rater_cli tests)

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/calibration/rater_cli.py apps/evals/teacher_model/calibration/tests/test_rater_cli.py && git commit -m "feat(calibration): rater_cli session cap enforcement"
```

---

## Task 12: analyze_calibration assigns each Phase-1 sub-score to a bucket
**Group:** C (depends on T9 — same file `analyze_calibration.py`)

**Behavior being verified:** `calibrate(...)` returns `{"buckets": {sub_score: bucket_name}, ...}` where bucket assignment follows: TRUSTED if κ ≥ 0.6 AND threshold per-sub-score variance ≥ 0.2 AND offset < 0.3; CEILING_ARTIFACT if rater variance < 0.2 OR judge variance < 0.2; TRUSTED_WITH_OFFSET if κ ≥ 0.6 AND mean offset ≥ 0.3 AND offset variance is low; UNTRUSTED otherwise. Aggregate gate field `aggregate_gate_pass` is True iff ≥7 of 11 Phase-1 sub-scores in TRUSTED or TRUSTED_WITH_OFFSET.

**Files:**
- Modify: `apps/evals/teacher_model/calibration/analyze_calibration.py`
- Modify: `apps/evals/teacher_model/calibration/tests/test_analyze_calibration.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to: apps/evals/teacher_model/calibration/tests/test_analyze_calibration.py
def test_bucket_routing_perfect_agreement_yields_trusted(tmp_path: Path):
    synths = []
    for i in range(50):
        synth_id = f"piece__rec{i}__3"
        # rater values vary across 0..3 to give variance
        v = i % 4
        rv = {sub: v for sub in [
            "ascf_process", "concrete_artifact_process", "praise_process",
            "autonomy_process", "scaffolded_process", "style_process",
            "tone_process", "autonomy_outcome", "tone_outcome",
            "concrete_artifact_outcome", "praise_outcome",
        ]}
        jv = {s: v for s in ["ascf", "concrete_artifact", "praise", "autonomy",
                              "scaffolded", "style", "tone"]}
        synths.append((synth_id, rv, jv))
    ratings_path = tmp_path / "ratings.jsonl"
    baseline_path = tmp_path / "baseline.jsonl"
    _write_full_synth(ratings_path, baseline_path, synths)

    report = calibrate(ratings_path=ratings_path, baseline_path=baseline_path)

    for sub in [
        "ascf_process", "concrete_artifact_process", "praise_process",
        "autonomy_process", "scaffolded_process", "style_process",
        "tone_process", "autonomy_outcome", "tone_outcome",
        "concrete_artifact_outcome", "praise_outcome",
    ]:
        assert report["buckets"][sub] == "TRUSTED", (sub, report["buckets"])
    assert report["aggregate_gate_pass"] is True


def test_bucket_routing_saturated_dim_yields_ceiling_artifact(tmp_path: Path):
    # Both rater and judge always emit 3 for tone_outcome — variance is 0.
    synths = []
    for i in range(50):
        synth_id = f"piece__rec{i}__3"
        # Vary other dims, freeze tone outcome at 3
        v = i % 4
        rv = {
            "ascf_process": v, "concrete_artifact_process": v,
            "praise_process": v, "autonomy_process": v,
            "scaffolded_process": v, "style_process": v, "tone_process": v,
            "autonomy_outcome": v, "tone_outcome": 3,
            "concrete_artifact_outcome": v, "praise_outcome": v,
        }
        jv = {s: v for s in ["ascf", "concrete_artifact", "praise",
                              "autonomy", "scaffolded", "style"]}
        jv["tone"] = 3  # judge also saturated
        synths.append((synth_id, rv, jv))
    ratings_path = tmp_path / "ratings.jsonl"
    baseline_path = tmp_path / "baseline.jsonl"
    _write_full_synth(ratings_path, baseline_path, synths)

    report = calibrate(ratings_path=ratings_path, baseline_path=baseline_path)
    assert report["buckets"]["tone_outcome"] == "CEILING_ARTIFACT"


def test_bucket_routing_systematic_offset_yields_trusted_with_offset(tmp_path: Path):
    # Rater is consistently 1 less than judge for ascf_process; other dims aligned.
    synths = []
    for i in range(50):
        synth_id = f"piece__rec{i}__3"
        v = i % 4
        rater_ascf = max(0, v - 1)
        rv = {
            "ascf_process": rater_ascf,
            "concrete_artifact_process": v, "praise_process": v,
            "autonomy_process": v, "scaffolded_process": v, "style_process": v,
            "tone_process": v, "autonomy_outcome": v, "tone_outcome": v,
            "concrete_artifact_outcome": v, "praise_outcome": v,
        }
        jv = {s: v for s in ["ascf", "concrete_artifact", "praise", "autonomy",
                              "scaffolded", "style", "tone"]}
        synths.append((synth_id, rv, jv))
    ratings_path = tmp_path / "ratings.jsonl"
    baseline_path = tmp_path / "baseline.jsonl"
    _write_full_synth(ratings_path, baseline_path, synths)

    report = calibrate(ratings_path=ratings_path, baseline_path=baseline_path)
    assert report["buckets"]["ascf_process"] == "TRUSTED_WITH_OFFSET"
    assert abs(report["mean_offset"]["ascf_process"] - (-0.75)) < 0.05  # E[v - (v-1)] over 0..3 = 0.75 → judge minus rater = 0.75; rater minus judge = -0.75


def test_bucket_routing_independent_random_yields_untrusted(tmp_path: Path):
    rng = random.Random(7)
    synths = []
    for i in range(200):
        synth_id = f"piece__rec{i}__3"
        rv = {sub: rng.randint(0, 3) for sub in [
            "ascf_process", "concrete_artifact_process", "praise_process",
            "autonomy_process", "scaffolded_process", "style_process",
            "tone_process", "autonomy_outcome", "tone_outcome",
            "concrete_artifact_outcome", "praise_outcome",
        ]}
        jv = {s: rng.randint(0, 3) for s in ["ascf", "concrete_artifact", "praise",
                                              "autonomy", "scaffolded", "style", "tone"]}
        synths.append((synth_id, rv, jv))
    ratings_path = tmp_path / "ratings.jsonl"
    baseline_path = tmp_path / "baseline.jsonl"
    _write_full_synth(ratings_path, baseline_path, synths)

    report = calibrate(ratings_path=ratings_path, baseline_path=baseline_path)
    assert report["buckets"]["ascf_process"] == "UNTRUSTED"
    assert report["aggregate_gate_pass"] is False
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest -xvs teacher_model/calibration/tests/test_analyze_calibration.py::test_bucket_routing_perfect_agreement_yields_trusted
```
Expected: FAIL — `KeyError: 'buckets'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Append to `apps/evals/teacher_model/calibration/analyze_calibration.py` and update `calibrate` to compute buckets and the aggregate gate:

```python
KAPPA_TRUSTED_THRESHOLD: float = 0.6
SCORE_VARIANCE_CEILING_THRESHOLD: float = 0.2
OFFSET_TRUSTED_THRESHOLD: float = 0.3
AGGREGATE_GATE_MIN_TRUSTED: int = 7


def _variance(values: list[int]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def _route_bucket(
    kappa: float, mean_offset: float, rater_var: float, judge_var: float
) -> str:
    if rater_var < SCORE_VARIANCE_CEILING_THRESHOLD or judge_var < SCORE_VARIANCE_CEILING_THRESHOLD:
        return "CEILING_ARTIFACT"
    if kappa < KAPPA_TRUSTED_THRESHOLD:
        return "UNTRUSTED"
    if abs(mean_offset) >= OFFSET_TRUSTED_THRESHOLD:
        return "TRUSTED_WITH_OFFSET"
    return "TRUSTED"
```

Then replace the final `return` block of `calibrate` with the version below (everything else above the `return` stays). The replacement adds bucket assignment and aggregate-gate computation:

```python
    buckets: dict[str, str] = {}
    mean_offset: dict[str, float] = {}
    rater_variance: dict[str, float] = {}
    judge_variance: dict[str, float] = {}

    phase_1_sub_scores = [
        "ascf_process", "concrete_artifact_process", "praise_process",
        "autonomy_process", "scaffolded_process", "style_process",
        "tone_process", "autonomy_outcome", "tone_outcome",
        "concrete_artifact_outcome", "praise_outcome",
    ]

    for sub in phase_1_sub_scores:
        pairs = pairs_by_sub.get(sub, [])
        if not pairs:
            buckets[sub] = "UNTRUSTED"
            mean_offset[sub] = 0.0
            rater_variance[sub] = 0.0
            judge_variance[sub] = 0.0
            continue
        rater_vals = [a for a, _ in pairs]
        judge_vals = [b for _, b in pairs]
        rv = _variance(rater_vals)
        jv = _variance(judge_vals)
        offset = sum(a - b for a, b in pairs) / len(pairs)
        kappa = per_sub_score_kappa.get(sub, float("nan"))
        if kappa != kappa:  # NaN
            kappa = -1.0
        buckets[sub] = _route_bucket(kappa, offset, rv, jv)
        mean_offset[sub] = offset
        rater_variance[sub] = rv
        judge_variance[sub] = jv

    n_trusted = sum(
        1 for s in phase_1_sub_scores
        if buckets[s] in ("TRUSTED", "TRUSTED_WITH_OFFSET")
    )
    aggregate_gate_pass = n_trusted >= AGGREGATE_GATE_MIN_TRUSTED

    return {
        "per_sub_score_kappa": per_sub_score_kappa,
        "threshold_decision_agreement": agreement,
        "threshold_decision_kappa": kappa_thresh_var,  # renamed below
        "n_threshold_pairs": len(threshold_pairs),
        "buckets": buckets,
        "mean_offset": mean_offset,
        "rater_variance": rater_variance,
        "judge_variance": judge_variance,
        "n_phase_1_trusted": n_trusted,
        "aggregate_gate_pass": aggregate_gate_pass,
    }
```

Note: rename the local `kappa` (threshold) to `kappa_thresh_var` in the existing T9-added block to avoid the collision with the per-sub-score `kappa` introduced in this task. The full updated calibrate function should be:

```python
def calibrate(ratings_path: Path, baseline_path: Path) -> dict[str, Any]:
    baseline = _build_baseline_index(baseline_path)
    pairs_by_sub: dict[str, list[tuple[int, int]]] = defaultdict(list)
    rater_vals_per_synth: dict[str, dict[str, int]] = defaultdict(dict)

    with ratings_path.open() as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("event_type") != "rating":
                continue
            if rec.get("anchor_origin_id") is not None:
                continue
            row = baseline.get(rec["synth_id"])
            if row is None:
                continue
            jv = _judge_value_for_sub_score(row.get("judge_dimensions", []), rec["sub_score"])
            if jv is not None:
                pairs_by_sub[rec["sub_score"]].append((rec["value"], jv))
            rater_vals_per_synth[rec["synth_id"]][rec["sub_score"]] = rec["value"]

    per_sub_score_kappa = {
        sub: _cohens_weighted_kappa([a for a, _ in pairs], [b for _, b in pairs])
        for sub, pairs in pairs_by_sub.items()
        if pairs
    }

    threshold_pairs: list[tuple[int, int]] = []
    for synth_id, vals in rater_vals_per_synth.items():
        row = baseline.get(synth_id)
        if row is None:
            continue
        rc = _rater_composite(vals)
        jc = _judge_composite(row.get("judge_dimensions", []))
        if rc is None or jc is None:
            continue
        rater_pass = 1 if rc >= COMPOSITE_PASS_THRESHOLD else 0
        judge_pass = 1 if jc >= COMPOSITE_PASS_THRESHOLD else 0
        threshold_pairs.append((rater_pass, judge_pass))

    if threshold_pairs:
        agreement = sum(1 for a, b in threshold_pairs if a == b) / len(threshold_pairs)
        a_vals = [a for a, _ in threshold_pairs]
        b_vals = [b for _, b in threshold_pairs]
        kappa_thresh = _cohens_weighted_kappa(a_vals, b_vals, k=2)
    else:
        agreement = 0.0
        kappa_thresh = float("nan")

    buckets: dict[str, str] = {}
    mean_offset: dict[str, float] = {}
    rater_variance: dict[str, float] = {}
    judge_variance: dict[str, float] = {}

    phase_1_sub_scores = [
        "ascf_process", "concrete_artifact_process", "praise_process",
        "autonomy_process", "scaffolded_process", "style_process",
        "tone_process", "autonomy_outcome", "tone_outcome",
        "concrete_artifact_outcome", "praise_outcome",
    ]

    for sub in phase_1_sub_scores:
        pairs = pairs_by_sub.get(sub, [])
        if not pairs:
            buckets[sub] = "UNTRUSTED"
            mean_offset[sub] = 0.0
            rater_variance[sub] = 0.0
            judge_variance[sub] = 0.0
            continue
        rater_vals = [a for a, _ in pairs]
        judge_vals = [b for _, b in pairs]
        rv = _variance(rater_vals)
        jv = _variance(judge_vals)
        offset = sum(a - b for a, b in pairs) / len(pairs)
        kappa = per_sub_score_kappa.get(sub, float("nan"))
        if kappa != kappa:
            kappa = -1.0
        buckets[sub] = _route_bucket(kappa, offset, rv, jv)
        mean_offset[sub] = offset
        rater_variance[sub] = rv
        judge_variance[sub] = jv

    n_trusted = sum(
        1 for s in phase_1_sub_scores
        if buckets[s] in ("TRUSTED", "TRUSTED_WITH_OFFSET")
    )
    aggregate_gate_pass = n_trusted >= AGGREGATE_GATE_MIN_TRUSTED

    return {
        "per_sub_score_kappa": per_sub_score_kappa,
        "threshold_decision_agreement": agreement,
        "threshold_decision_kappa": kappa_thresh,
        "n_threshold_pairs": len(threshold_pairs),
        "buckets": buckets,
        "mean_offset": mean_offset,
        "rater_variance": rater_variance,
        "judge_variance": judge_variance,
        "n_phase_1_trusted": n_trusted,
        "aggregate_gate_pass": aggregate_gate_pass,
    }
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest -xvs teacher_model/calibration/tests/test_analyze_calibration.py
```
Expected: PASS (all nine analyze_calibration tests)

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/calibration/analyze_calibration.py apps/evals/teacher_model/calibration/tests/test_analyze_calibration.py && git commit -m "feat(calibration): 4-bucket routing and aggregate gate"
```

---

## Task 13: select_sample injects silent anchor duplicates
**Group:** D (depends on T10 — same file `select_sample.py`)

**Behavior being verified:** Manifest contains an `anchors` list of length `anchor_n` (~20). Each anchor entry references a `synth_id` from `main` (anchor source) plus a `synth_id_displayed` (a unique scrambled string the rater sees on the second showing) plus a `display_position` integer ≥ `len(main)` so the duplicate appears late in the rating schedule.

**Files:**
- Modify: `apps/evals/teacher_model/calibration/select_sample.py`
- Modify: `apps/evals/teacher_model/calibration/tests/test_select_sample.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to: apps/evals/teacher_model/calibration/tests/test_select_sample.py
def test_anchors_reference_main_and_have_scrambled_display_ids(tmp_path: Path):
    source = tmp_path / "baseline.jsonl"
    _write_synthetic_baseline(source)

    manifest = select_sample(
        source_path=source, target_n=200, holdout_n=30, anchor_n=20, seed=42,
    )

    main_ids = {e["synth_id"] for e in manifest["main"]}
    anchors = manifest["anchors"]
    assert len(anchors) == 20
    for a in anchors:
        assert a["synth_id"] in main_ids
        # The displayed id must differ from the original to avoid recognition.
        assert a["synth_id_displayed"] != a["synth_id"]
        assert a["display_position"] >= len(manifest["main"])
    # All scrambled display ids must be unique.
    displayed = [a["synth_id_displayed"] for a in anchors]
    assert len(set(displayed)) == len(displayed)
    assert manifest["stats"]["n_anchors_silent_dups"] == 20
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest -xvs teacher_model/calibration/tests/test_select_sample.py::test_anchors_reference_main_and_have_scrambled_display_ids
```
Expected: FAIL — `assert len(anchors) == 20` fails (anchors is empty in T10's manifest).

- [ ] **Step 3: Implement the minimum to make the test pass**

In `apps/evals/teacher_model/calibration/select_sample.py`, replace the construction of `anchors=[]` and `n_anchors_silent_dups=0` in the return dict with the version below. Add a helper `_make_scrambled_id` and select anchors deterministically from `main`:

```python
import hashlib


def _make_scrambled_id(synth_id: str, seed: int, position: int) -> str:
    h = hashlib.sha256(f"{seed}:{position}:{synth_id}".encode()).hexdigest()
    return f"anchor-{h[:16]}"
```

Replace the return-dict portion of `select_sample` (everything from `band_counts = ...` through the final return) with:

```python
    band_counts = Counter(_classify_band(r) for r in main)
    era_counts = Counter(composer_to_era(r["composer"]) for r in main)

    if anchor_n > len(main):
        raise ValueError(f"anchor_n={anchor_n} exceeds main size {len(main)}")
    anchor_indices = list(range(len(main)))
    rng.shuffle(anchor_indices)
    anchor_indices = sorted(anchor_indices[:anchor_n])

    anchors: list[dict[str, Any]] = []
    for k, idx in enumerate(anchor_indices):
        original = main[idx]
        synth_id = _row_synth_id(original)
        display_position = len(main) + k
        anchors.append({
            "synth_id": synth_id,
            "synth_id_displayed": _make_scrambled_id(synth_id, seed, display_position),
            "display_position": display_position,
            "original_main_index": idx,
        })

    return {
        "version": 1,
        "seed": seed,
        "source_path": str(source_path),
        "main": [
            {
                "synth_id": _row_synth_id(r),
                "band": _classify_band(r),
                "era": composer_to_era(r["composer"]),
                "skill_bucket": r["skill_bucket"],
                "is_anchor_seed": _row_synth_id(r) in {a["synth_id"] for a in anchors},
                "anchor_position": next(
                    (a["display_position"] for a in anchors
                     if a["synth_id"] == _row_synth_id(r)),
                    None,
                ),
            }
            for r in main
        ],
        "anchors": anchors,
        "holdout": [
            {
                "synth_id": _row_synth_id(r),
                "band": _classify_band(r),
                "era": composer_to_era(r["composer"]),
                "skill_bucket": r["skill_bucket"],
            }
            for r in holdout_rows
        ],
        "stats": {
            "n_main": len(main),
            "n_anchors_silent_dups": len(anchors),
            "n_holdout": len(holdout_rows),
            "band_counts": dict(band_counts),
            "era_counts": dict(era_counts),
        },
    }
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest -xvs teacher_model/calibration/tests/test_select_sample.py
```
Expected: PASS (all five select_sample tests)

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/calibration/select_sample.py apps/evals/teacher_model/calibration/tests/test_select_sample.py && git commit -m "feat(calibration): silent anchor duplicate injection"
```

---

## Task 14: rater_cli resume_state reports the next un-rated synthesis
**Group:** D (depends on T11 — same file `rater_cli.py`)

**Behavior being verified:** Given a manifest and a partial ratings.jsonl, `compute_resume_state` returns the index of the next manifest entry that has zero rating events written for it. Synthesis is considered "complete" only when all 11 sub-scores have rating events; partial syntheses are retried (the partial events are flagged for inspection but not silently consumed).

**Interface under test:** `compute_resume_state(manifest: dict, ratings_path: Path) -> dict`

**Files:**
- Modify: `apps/evals/teacher_model/calibration/rater_cli.py`
- Modify: `apps/evals/teacher_model/calibration/tests/test_rater_cli.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to: apps/evals/teacher_model/calibration/tests/test_rater_cli.py
from teacher_model.calibration.rater_cli import compute_resume_state


def _ratings_for(synth_id: str, sub_scores: list[str]) -> list[dict]:
    return [
        {
            "event_type": "rating",
            "synth_id": synth_id,
            "anchor_origin_id": None,
            "sub_score": sub,
            "value": 2,
            "evidence": "", "reason": "",
            "session_id": "S001", "session_idx": i + 1,
            "ts": "x",
        }
        for i, sub in enumerate(sub_scores)
    ]


def test_resume_state_picks_next_unstarted_synthesis(tmp_path: Path):
    manifest = {
        "main": [
            {"synth_id": "S0", "band": "high", "era": "Romantic",
             "skill_bucket": 3, "is_anchor_seed": False, "anchor_position": None},
            {"synth_id": "S1", "band": "high", "era": "Romantic",
             "skill_bucket": 3, "is_anchor_seed": False, "anchor_position": None},
            {"synth_id": "S2", "band": "high", "era": "Romantic",
             "skill_bucket": 3, "is_anchor_seed": False, "anchor_position": None},
        ],
        "anchors": [],
    }

    ratings_path = tmp_path / "ratings.jsonl"
    # S0 fully rated; S1 partial (only 5 sub-scores); S2 untouched.
    full = _ratings_for("S0", PHASE_1_SUB_SCORES)
    partial = _ratings_for("S1", PHASE_1_SUB_SCORES[:5])
    with ratings_path.open("w") as f:
        for r in full + partial:
            f.write(json.dumps(r) + "\n")

    state = compute_resume_state(manifest=manifest, ratings_path=ratings_path)
    # Resume index is the first synthesis that is NOT fully rated; S1 is
    # partial, so we resume there (the partial events get flagged).
    assert state["next_main_index"] == 1
    assert state["partially_rated"] == ["S1"]
    assert state["fully_rated"] == ["S0"]


def test_resume_state_returns_zero_when_no_ratings(tmp_path: Path):
    manifest = {
        "main": [{"synth_id": "S0", "band": "high", "era": "Romantic",
                  "skill_bucket": 3, "is_anchor_seed": False,
                  "anchor_position": None}],
        "anchors": [],
    }
    ratings_path = tmp_path / "ratings.jsonl"
    ratings_path.write_text("")

    state = compute_resume_state(manifest=manifest, ratings_path=ratings_path)
    assert state["next_main_index"] == 0
    assert state["fully_rated"] == []
    assert state["partially_rated"] == []
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest -xvs teacher_model/calibration/tests/test_rater_cli.py::test_resume_state_picks_next_unstarted_synthesis
```
Expected: FAIL — `ImportError: cannot import name 'compute_resume_state'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Append to `apps/evals/teacher_model/calibration/rater_cli.py`:

```python
from collections import defaultdict


def compute_resume_state(manifest: dict, ratings_path: Path) -> dict:
    sub_count_per_synth: dict[str, int] = defaultdict(int)
    if ratings_path.exists():
        with ratings_path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("event_type") != "rating":
                    continue
                sub_count_per_synth[rec["synth_id"]] += 1

    fully_rated: list[str] = []
    partially_rated: list[str] = []
    next_main_index: int | None = None

    for i, entry in enumerate(manifest["main"]):
        n = sub_count_per_synth.get(entry["synth_id"], 0)
        if n >= len(PHASE_1_SUB_SCORES):
            fully_rated.append(entry["synth_id"])
            continue
        if n > 0:
            partially_rated.append(entry["synth_id"])
        if next_main_index is None:
            next_main_index = i

    if next_main_index is None:
        next_main_index = len(manifest["main"])

    return {
        "next_main_index": next_main_index,
        "fully_rated": fully_rated,
        "partially_rated": partially_rated,
    }
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest -xvs teacher_model/calibration/tests/test_rater_cli.py
```
Expected: PASS (all rater_cli tests)

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/calibration/rater_cli.py apps/evals/teacher_model/calibration/tests/test_rater_cli.py && git commit -m "feat(calibration): rater_cli resume-from-crash state computation"
```

---

## Task 15: emit_recipe writes a Python module that filters syntheses correctly
**Group:** D (depends on T12 — different file from T13/T14, can run parallel)

**Behavior being verified:** Given a calibration_report dict with TRUSTED sub-scores, `emit` writes a `filter_recipe.py` file that, when imported, exposes `WEIGHTED_SUB_SCORES` (only TRUSTED + TRUSTED_WITH_OFFSET sub-scores), `BIAS_CORRECTIONS` (empty if no OFFSET dims), `SANITY_FILTERS` (CEILING dims), and `COMPOSITE_PASS_THRESHOLD = 2.5`. The recipe is importable as a regular Python module from a temp directory.

**Files:**
- Create: `apps/evals/teacher_model/calibration/emit_recipe.py`
- Test: `apps/evals/teacher_model/calibration/tests/test_emit_recipe.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/teacher_model/calibration/tests/test_emit_recipe.py
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from teacher_model.calibration.emit_recipe import emit


def _import_recipe(path: Path):
    spec = importlib.util.spec_from_file_location("filter_recipe_under_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load recipe at {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["filter_recipe_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_emit_writes_importable_recipe_with_trusted_subscores(tmp_path: Path):
    calibration_report = {
        "buckets": {
            "ascf_process": "TRUSTED",
            "concrete_artifact_process": "TRUSTED",
            "praise_process": "TRUSTED",
            "autonomy_process": "TRUSTED",
            "scaffolded_process": "TRUSTED",
            "style_process": "TRUSTED",
            "tone_process": "TRUSTED",
            "autonomy_outcome": "TRUSTED",
            "tone_outcome": "CEILING_ARTIFACT",
            "concrete_artifact_outcome": "UNTRUSTED",
            "praise_outcome": "TRUSTED",
        },
        "mean_offset": {sub: 0.0 for sub in [
            "ascf_process", "concrete_artifact_process", "praise_process",
            "autonomy_process", "scaffolded_process", "style_process",
            "tone_process", "autonomy_outcome", "tone_outcome",
            "concrete_artifact_outcome", "praise_outcome",
        ]},
        "aggregate_gate_pass": True,
    }
    drift_report = {"intra_rater_kappa": {}, "judge_drift_kappa": {}}

    recipe_path = tmp_path / "filter_recipe.py"
    emit(calibration_report=calibration_report, drift_report=drift_report,
         output_path=recipe_path)

    mod = _import_recipe(recipe_path)
    assert mod.COMPOSITE_PASS_THRESHOLD == 2.5
    # 9 TRUSTED sub-scores get full weight 1.0
    assert set(mod.WEIGHTED_SUB_SCORES.keys()) == {
        "ascf_process", "concrete_artifact_process", "praise_process",
        "autonomy_process", "scaffolded_process", "style_process",
        "tone_process", "autonomy_outcome", "praise_outcome",
    }
    assert all(w == 1.0 for w in mod.WEIGHTED_SUB_SCORES.values())
    # CEILING_ARTIFACT goes to SANITY_FILTERS, not WEIGHTED
    assert "tone_outcome" in mod.SANITY_FILTERS
    # UNTRUSTED is excluded entirely
    assert "concrete_artifact_outcome" not in mod.WEIGHTED_SUB_SCORES
    assert "concrete_artifact_outcome" not in mod.SANITY_FILTERS
    # No bias corrections (no TRUSTED_WITH_OFFSET in this fixture)
    assert mod.BIAS_CORRECTIONS == {}


def test_recipe_filters_synthesis_against_threshold(tmp_path: Path):
    """Round-trip: emit recipe, then a Stage-2-like consumer applies it."""
    calibration_report = {
        "buckets": {sub: "TRUSTED" for sub in [
            "ascf_process", "concrete_artifact_process", "praise_process",
            "autonomy_process", "scaffolded_process", "style_process",
            "tone_process", "autonomy_outcome", "tone_outcome",
            "concrete_artifact_outcome", "praise_outcome",
        ]},
        "mean_offset": {sub: 0.0 for sub in [
            "ascf_process", "concrete_artifact_process", "praise_process",
            "autonomy_process", "scaffolded_process", "style_process",
            "tone_process", "autonomy_outcome", "tone_outcome",
            "concrete_artifact_outcome", "praise_outcome",
        ]},
        "aggregate_gate_pass": True,
    }
    drift_report = {"intra_rater_kappa": {}, "judge_drift_kappa": {}}
    recipe_path = tmp_path / "filter_recipe_b.py"
    emit(calibration_report=calibration_report, drift_report=drift_report,
         output_path=recipe_path)
    mod = _import_recipe(recipe_path)

    def stage_2_filter(judge_scores_per_sub: dict[str, float]) -> bool:
        weighted = {
            s: judge_scores_per_sub[s] + mod.BIAS_CORRECTIONS.get(s, 0.0)
            for s in mod.WEIGHTED_SUB_SCORES
            if s in judge_scores_per_sub
        }
        if not weighted:
            return False
        composite = sum(weighted.values()) / len(weighted)
        return composite >= mod.COMPOSITE_PASS_THRESHOLD

    high = {s: 3 for s in mod.WEIGHTED_SUB_SCORES}
    low = {s: 1 for s in mod.WEIGHTED_SUB_SCORES}
    assert stage_2_filter(high) is True
    assert stage_2_filter(low) is False
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest -xvs teacher_model/calibration/tests/test_emit_recipe.py
```
Expected: FAIL — `ModuleNotFoundError: No module named 'teacher_model.calibration.emit_recipe'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# apps/evals/teacher_model/calibration/emit_recipe.py
"""Emit the filter_recipe.py public-API module consumed by Stage 2 SFT.

This file's only output is a Python source file. The Stage 2 pipeline imports
that file and nothing else from this protocol.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

COMPOSITE_PASS_THRESHOLD: float = 2.5

_HEADER = '''"""Generated calibration filter recipe — DO NOT EDIT BY HAND.

Generated by apps/evals/teacher_model/calibration/emit_recipe.py from
calibration_report.json. Re-run calibration to regenerate.

This module is the SOLE public API the Stage 2 SFT data pipeline imports
from the calibration protocol.
"""

'''


def _format_dict(name: str, d: dict, value_format: str = "{}") -> str:
    if not d:
        return f"{name}: dict = {{}}\n"
    lines = [f"{name}: dict = {{"]
    for k in sorted(d.keys()):
        v = d[k]
        formatted = value_format.format(v) if isinstance(v, (int, float)) else repr(v)
        lines.append(f"    {k!r}: {formatted},")
    lines.append("}\n")
    return "\n".join(lines)


def emit(
    calibration_report: dict[str, Any],
    drift_report: dict[str, Any],
    output_path: Path,
) -> None:
    buckets: dict[str, str] = calibration_report["buckets"]

    weighted: dict[str, float] = {}
    sanity: list[str] = []

    for sub, bucket in buckets.items():
        if bucket in ("TRUSTED", "TRUSTED_WITH_OFFSET"):
            weighted[sub] = 1.0
        elif bucket == "CEILING_ARTIFACT":
            sanity.append(sub)
        # UNTRUSTED: excluded entirely

    bias = {}  # T16 will populate from TRUSTED_WITH_OFFSET buckets

    body = _HEADER
    body += f"COMPOSITE_PASS_THRESHOLD: float = {COMPOSITE_PASS_THRESHOLD}\n\n"
    body += _format_dict("WEIGHTED_SUB_SCORES", weighted, "{:.4f}") + "\n"
    body += _format_dict("BIAS_CORRECTIONS", bias, "{:.4f}") + "\n"
    body += f"SANITY_FILTERS: list = {sorted(sanity)!r}\n"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(body)
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest -xvs teacher_model/calibration/tests/test_emit_recipe.py
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/calibration/emit_recipe.py apps/evals/teacher_model/calibration/tests/test_emit_recipe.py && git commit -m "feat(calibration): emit base filter recipe (TRUSTED + sanity filters)"
```

---

## Task 16: emit_recipe encodes bias corrections for TRUSTED_WITH_OFFSET buckets
**Group:** E (depends on T15 — same file `emit_recipe.py`)

**Behavior being verified:** When a sub-score is in the TRUSTED_WITH_OFFSET bucket with `mean_offset = -0.4`, the emitted recipe's `BIAS_CORRECTIONS["sub_score"] = -0.4`, and a Stage-2-like consumer applying that correction to a borderline judge score correctly flips a pass/fail decision.

**Files:**
- Modify: `apps/evals/teacher_model/calibration/emit_recipe.py`
- Modify: `apps/evals/teacher_model/calibration/tests/test_emit_recipe.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to: apps/evals/teacher_model/calibration/tests/test_emit_recipe.py
def test_emit_records_bias_correction_for_trusted_with_offset(tmp_path: Path):
    calibration_report = {
        "buckets": {sub: "TRUSTED" for sub in [
            "concrete_artifact_process", "praise_process",
            "autonomy_process", "scaffolded_process", "style_process",
            "tone_process", "autonomy_outcome", "tone_outcome",
            "concrete_artifact_outcome", "praise_outcome",
        ]},
        "mean_offset": {sub: 0.0 for sub in [
            "ascf_process", "concrete_artifact_process", "praise_process",
            "autonomy_process", "scaffolded_process", "style_process",
            "tone_process", "autonomy_outcome", "tone_outcome",
            "concrete_artifact_outcome", "praise_outcome",
        ]},
        "aggregate_gate_pass": True,
    }
    calibration_report["buckets"]["ascf_process"] = "TRUSTED_WITH_OFFSET"
    calibration_report["mean_offset"]["ascf_process"] = -0.4
    drift_report = {"intra_rater_kappa": {}, "judge_drift_kappa": {}}

    recipe_path = tmp_path / "filter_recipe_offset.py"
    emit(calibration_report=calibration_report, drift_report=drift_report,
         output_path=recipe_path)

    mod = _import_recipe(recipe_path)
    assert "ascf_process" in mod.BIAS_CORRECTIONS
    assert abs(mod.BIAS_CORRECTIONS["ascf_process"] - (-0.4)) < 1e-6
    assert "ascf_process" in mod.WEIGHTED_SUB_SCORES  # still weighted

    # Round-trip: a borderline judge score 2.5 minus 0.4 correction = 2.1 fails;
    # without correction it would pass.
    judge_scores = {s: 2.5 for s in mod.WEIGHTED_SUB_SCORES}
    corrected = {
        s: judge_scores[s] + mod.BIAS_CORRECTIONS.get(s, 0.0)
        for s in mod.WEIGHTED_SUB_SCORES
    }
    composite = sum(corrected.values()) / len(corrected)
    # 10 sub-scores at 2.5 + 1 sub-score at 2.1 = (10*2.5 + 2.1)/11 = 27.1/11 ≈ 2.4636
    assert composite < mod.COMPOSITE_PASS_THRESHOLD
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest -xvs teacher_model/calibration/tests/test_emit_recipe.py::test_emit_records_bias_correction_for_trusted_with_offset
```
Expected: FAIL — `assert "ascf_process" in mod.BIAS_CORRECTIONS` fails (T15's `bias = {}` is empty).

- [ ] **Step 3: Implement the minimum to make the test pass**

In `apps/evals/teacher_model/calibration/emit_recipe.py`, replace the `bias = {}` line in `emit()` with the following block that pulls offsets out of the calibration report:

```python
    bias: dict[str, float] = {}
    mean_offset: dict[str, float] = calibration_report.get("mean_offset", {})
    for sub, bucket in buckets.items():
        if bucket == "TRUSTED_WITH_OFFSET":
            offset = mean_offset.get(sub)
            if offset is None:
                raise ValueError(
                    f"Sub-score {sub!r} is TRUSTED_WITH_OFFSET but has no "
                    f"mean_offset in calibration_report."
                )
            bias[sub] = float(offset)
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest -xvs teacher_model/calibration/tests/test_emit_recipe.py
```
Expected: PASS (both emit_recipe tests)

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/calibration/emit_recipe.py apps/evals/teacher_model/calibration/tests/test_emit_recipe.py && git commit -m "feat(calibration): bias-correction emission for TRUSTED_WITH_OFFSET dims"
```

---

## Task 17: select_sample enforces skill-bucket min-quotas
**Group:** E (depends on T13 — same file `select_sample.py`)

**Behavior being verified:** Every skill-bucket group (beginner=1-2, intermediate=3, advanced=4-5) appears ≥50 times in the manifest's main 200. `manifest["stats"]["skill_group_counts"]` reports the three counts.

**Files:**
- Modify: `apps/evals/teacher_model/calibration/select_sample.py`
- Modify: `apps/evals/teacher_model/calibration/tests/test_select_sample.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to: apps/evals/teacher_model/calibration/tests/test_select_sample.py
def test_skill_group_min_quotas_satisfied(tmp_path: Path):
    source = tmp_path / "baseline.jsonl"
    _write_synthetic_baseline(source)

    manifest = select_sample(
        source_path=source, target_n=200, holdout_n=30, anchor_n=20, seed=42,
    )

    counts = manifest["stats"]["skill_group_counts"]
    assert counts["beginner"] >= 50, counts
    assert counts["intermediate"] >= 50, counts
    assert counts["advanced"] >= 50, counts
    assert counts["beginner"] + counts["intermediate"] + counts["advanced"] == 200
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest -xvs teacher_model/calibration/tests/test_select_sample.py::test_skill_group_min_quotas_satisfied
```
Expected: FAIL — `KeyError: 'skill_group_counts'`

- [ ] **Step 3: Implement the minimum to make the test pass**

In `apps/evals/teacher_model/calibration/select_sample.py`, add the skill-group helper, the min-quotas constant, and a third selection pass before band fill. Add near the top with other constants:

```python
_SKILL_MIN_QUOTAS: dict[str, int] = {
    "beginner": 50,
    "intermediate": 50,
    "advanced": 50,
}


def _skill_group(skill_bucket: int) -> str:
    if skill_bucket <= 2:
        return "beginner"
    if skill_bucket == 3:
        return "intermediate"
    return "advanced"
```

Replace `_select_with_quotas` to take a `skill_min_quotas` arg and add a skill-reservation pass between the era pass and the band fill pass:

```python
def _select_with_quotas(
    by_band: dict[str, list[dict[str, Any]]],
    band_targets: dict[str, int],
    era_min_quotas: dict[str, int],
    skill_min_quotas: dict[str, int],
    rng: random.Random,
) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    chosen_ids: set[str] = set()
    era_counter: Counter[str] = Counter()
    skill_counter: Counter[str] = Counter()
    band_counter: Counter[str] = Counter()

    def _take(r: dict[str, Any], band: str) -> None:
        chosen.append(r)
        chosen_ids.add(_row_synth_id(r))
        era_counter[composer_to_era(r["composer"])] += 1
        skill_counter[_skill_group(r["skill_bucket"])] += 1
        band_counter[band] += 1

    # Pass 1: era reservation
    for era, min_q in era_min_quotas.items():
        candidates = [
            (band, r) for band, pool in by_band.items() for r in pool
            if composer_to_era(r["composer"]) == era and _row_synth_id(r) not in chosen_ids
        ]
        rng.shuffle(candidates)
        if len(candidates) < min_q:
            raise ValueError(f"Era '{era}' has {len(candidates)} candidates but needs {min_q}.")
        for band, r in candidates:
            if era_counter[era] >= min_q:
                break
            if band_counter[band] >= band_targets[band]:
                continue
            _take(r, band)
        if era_counter[era] < min_q:
            for band, r in candidates:
                if _row_synth_id(r) in chosen_ids:
                    continue
                _take(r, band)
                if era_counter[era] >= min_q:
                    break

    # Pass 2: skill reservation (covers groups under-met by era pass)
    for group, min_q in skill_min_quotas.items():
        if skill_counter[group] >= min_q:
            continue
        candidates = [
            (band, r) for band, pool in by_band.items() for r in pool
            if _skill_group(r["skill_bucket"]) == group and _row_synth_id(r) not in chosen_ids
        ]
        rng.shuffle(candidates)
        needed = min_q - skill_counter[group]
        if len(candidates) < needed:
            raise ValueError(
                f"Skill group '{group}' has {len(candidates)} unreserved candidates "
                f"but needs {needed} more."
            )
        for band, r in candidates:
            if skill_counter[group] >= min_q:
                break
            if band_counter[band] >= band_targets[band]:
                continue
            _take(r, band)
        if skill_counter[group] < min_q:
            for band, r in candidates:
                if _row_synth_id(r) in chosen_ids:
                    continue
                _take(r, band)
                if skill_counter[group] >= min_q:
                    break

    # Pass 3: fill remaining band slots
    for band, target in band_targets.items():
        remaining = target - band_counter[band]
        if remaining <= 0:
            continue
        pool = [r for r in by_band[band] if _row_synth_id(r) not in chosen_ids]
        rng.shuffle(pool)
        if len(pool) < remaining:
            raise ValueError(
                f"Band '{band}' has only {len(pool)} unreserved rows but needs {remaining} more."
            )
        for r in pool[:remaining]:
            _take(r, band)

    # Pass 4: trim overshoot. Some bands may be over target due to forced
    # quota fills above; drop excess rows whose removal does not violate any
    # quota.
    while len(chosen) > sum(band_targets.values()):
        for band, target in band_targets.items():
            if band_counter[band] <= target:
                continue
            for i, r in enumerate(chosen):
                if _classify_band(r) != band:
                    continue
                era = composer_to_era(r["composer"])
                grp = _skill_group(r["skill_bucket"])
                if era_counter[era] - 1 < era_min_quotas.get(era, 0):
                    continue
                if skill_counter[grp] - 1 < skill_min_quotas.get(grp, 0):
                    continue
                chosen.pop(i)
                chosen_ids.discard(_row_synth_id(r))
                era_counter[era] -= 1
                skill_counter[grp] -= 1
                band_counter[band] -= 1
                break
            else:
                continue
            break
        else:
            raise ValueError(
                "Cannot satisfy all quotas simultaneously: overshoot is locked by "
                "minima in every band. Loosen a quota or expand source pool."
            )

    return chosen
```

Update the call site in `select_sample` to pass `_SKILL_MIN_QUOTAS` and add `skill_group_counts` to stats:

```python
    main = _select_with_quotas(by_band, band_targets, _ERA_MIN_QUOTAS, _SKILL_MIN_QUOTAS, rng)
    rng.shuffle(main)

    band_counts = Counter(_classify_band(r) for r in main)
    era_counts = Counter(composer_to_era(r["composer"]) for r in main)
    skill_group_counts = Counter(_skill_group(r["skill_bucket"]) for r in main)
```

And in the returned `stats` dict:

```python
        "stats": {
            "n_main": len(main),
            "n_anchors_silent_dups": len(anchors),
            "n_holdout": len(holdout_rows),
            "band_counts": dict(band_counts),
            "era_counts": dict(era_counts),
            "skill_group_counts": dict(skill_group_counts),
        },
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest -xvs teacher_model/calibration/tests/test_select_sample.py
```
Expected: PASS (all six select_sample tests)

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/calibration/select_sample.py apps/evals/teacher_model/calibration/tests/test_select_sample.py && git commit -m "feat(calibration): select_sample skill-bucket min-quotas"
```

---

## Task 18: analyze_drift computes judge-vs-judge κ on day1/day30 re-runs
**Group:** E (depends on T5 — same file `analyze_drift.py`; consumes T2's output schema)

**Behavior being verified:** Given two judge-run jsonl files written by T2's `rerun_anchors` (day1 and day30), `analyze_drift` returns `judge_drift_kappa` per sub-score (11 Phase-1 entries) where each value is the weighted κ between day1 and day30 ratings on the same anchor synth_ids. Identical re-runs yield κ = 1.0.

**Interface under test:** `analyze_drift(ratings_path, judge_runs_path)` — `judge_runs_path` is now consumed. The file is the concatenation of two T2 outputs (day1 records + day30 records), distinguished by their `run_label` field.

**Files:**
- Modify: `apps/evals/teacher_model/calibration/analyze_drift.py`
- Modify: `apps/evals/teacher_model/calibration/tests/test_analyze_drift.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to: apps/evals/teacher_model/calibration/tests/test_analyze_drift.py
def _write_judge_runs(path: Path, records: list[dict]) -> None:
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _judge_record(synth_id: str, run_label: str, dim_score: int) -> dict:
    return {
        "synth_id": synth_id,
        "run_label": run_label,
        "dimensions": [
            {"criterion": "Audible-Specific Corrective Feedback",
             "process": dim_score, "outcome": dim_score, "score": dim_score,
             "evidence": "", "reason": ""},
            {"criterion": "Appropriate Tone & Language",
             "process": dim_score, "outcome": dim_score, "score": dim_score,
             "evidence": "", "reason": ""},
        ],
        "ts": "x",
    }


def test_judge_drift_kappa_is_one_when_runs_identical(tmp_path: Path):
    ratings_path = tmp_path / "ratings.jsonl"
    ratings_path.write_text("")
    judge_runs_path = tmp_path / "judge_runs.jsonl"

    records = []
    for i, sid in enumerate(["A1", "A2", "A3", "A4", "A5"]):
        v = i % 4
        records.append(_judge_record(sid, "day1", v))
        records.append(_judge_record(sid, "day30", v))
    _write_judge_runs(judge_runs_path, records)

    report = analyze_drift(ratings_path=ratings_path, judge_runs_path=judge_runs_path)
    drift = report["judge_drift_kappa"]
    assert drift["ascf_process"] == 1.0
    assert drift["tone_process"] == 1.0


def test_judge_drift_kappa_drops_below_one_when_runs_disagree(tmp_path: Path):
    ratings_path = tmp_path / "ratings.jsonl"
    ratings_path.write_text("")
    judge_runs_path = tmp_path / "judge_runs.jsonl"

    records = []
    for i, sid in enumerate(["A1", "A2", "A3", "A4", "A5"]):
        v1 = i % 4
        v2 = (i + 1) % 4  # always disagree by 1
        records.append(_judge_record(sid, "day1", v1))
        records.append(_judge_record(sid, "day30", v2))
    _write_judge_runs(judge_runs_path, records)

    report = analyze_drift(ratings_path=ratings_path, judge_runs_path=judge_runs_path)
    drift = report["judge_drift_kappa"]
    assert drift["ascf_process"] < 1.0
    assert drift["ascf_process"] >= -1.0
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest -xvs teacher_model/calibration/tests/test_analyze_drift.py::test_judge_drift_kappa_is_one_when_runs_identical
```
Expected: FAIL — `assert {} ["ascf_process"]` raises KeyError (T5 returns `judge_drift_kappa: {}`).

- [ ] **Step 3: Implement the minimum to make the test pass**

Replace `analyze_drift` in `apps/evals/teacher_model/calibration/analyze_drift.py` with:

```python
_CRITERION_TO_SLUG: dict[str, str] = {
    "Audible-Specific Corrective Feedback": "ascf",
    "Concrete Artifact Provision": "concrete_artifact",
    "Specific Positive Praise": "praise",
    "Autonomy-Supporting Motivation": "autonomy",
    "Scaffolded Guided Discovery": "scaffolded",
    "Style-Consistent Musical Language": "style",
    "Appropriate Tone & Language": "tone",
}


def _extract_judge_sub_scores(dimensions: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for d in dimensions:
        slug = _CRITERION_TO_SLUG.get(d.get("criterion", ""))
        if slug is None:
            continue
        for leg in ("process", "outcome"):
            v = d.get(leg)
            if v is None:
                continue
            out[f"{slug}_{leg}"] = int(v)
    return out


def _compute_judge_drift(judge_runs_path: Path) -> dict[str, float]:
    by_run_label: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)
    with judge_runs_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            run_label = rec.get("run_label")
            synth_id = rec.get("synth_id")
            if run_label is None or synth_id is None:
                continue
            by_run_label[run_label][synth_id] = _extract_judge_sub_scores(rec["dimensions"])

    if len(by_run_label) < 2:
        return {}

    labels = list(by_run_label.keys())
    label_a, label_b = labels[0], labels[1]
    common_synth_ids = sorted(set(by_run_label[label_a]) & set(by_run_label[label_b]))

    pairs_by_sub: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for sid in common_synth_ids:
        a = by_run_label[label_a][sid]
        b = by_run_label[label_b][sid]
        for sub_score in set(a) & set(b):
            pairs_by_sub[sub_score].append((a[sub_score], b[sub_score]))

    return {
        sub: cohens_weighted_kappa([x for x, _ in pairs], [y for _, y in pairs])
        for sub, pairs in pairs_by_sub.items()
        if pairs
    }


def analyze_drift(ratings_path: Path, judge_runs_path: Path | None) -> dict[str, Any]:
    pairs_by_sub_score: dict[str, list[tuple[int, int]]] = defaultdict(list)
    first_seen: dict[tuple[str, str], int] = {}

    with ratings_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("event_type") != "rating":
                continue
            sub_score = rec["sub_score"]
            origin = rec.get("anchor_origin_id")
            if origin is None:
                first_seen[(rec["synth_id"], sub_score)] = rec["value"]
            else:
                first_value = first_seen.get((origin, sub_score))
                if first_value is None:
                    continue
                pairs_by_sub_score[sub_score].append((first_value, rec["value"]))

    intra_rater = {
        sub: cohens_weighted_kappa([a for a, _ in pairs], [b for _, b in pairs])
        for sub, pairs in pairs_by_sub_score.items()
        if pairs
    }

    judge_drift = _compute_judge_drift(judge_runs_path) if judge_runs_path is not None else {}

    return {
        "intra_rater_kappa": intra_rater,
        "judge_drift_kappa": judge_drift,
        "n_anchor_pairs": {sub: len(p) for sub, p in pairs_by_sub_score.items()},
    }
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest -xvs teacher_model/calibration/tests/test_analyze_drift.py
```
Expected: PASS (all four analyze_drift tests)

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/calibration/analyze_drift.py apps/evals/teacher_model/calibration/tests/test_analyze_drift.py && git commit -m "feat(calibration): judge-vs-judge drift kappa on day1/day30"
```

---

## Plan Self-Review

| Check | Result |
|---|---|
| Spec coverage | 16 tasks cover: era_lookup (T1, spec module ❶), judge_rerun (T2, ❹), select_sample (T3+T7+T10+T13, ❷), rater_cli (T4+T8+T11+T14, ❸), analyze_drift (T5, ❺), analyze_calibration (T6+T9+T12, ❻), emit_recipe (T15+T16, ❼). All seven spec modules have at least one task. |
| Placeholder scan | No "TBD"/"TODO"/"implement later". Each task has full test code, full implementation code, and exact run/commit commands. |
| Type consistency | `synth_id` format, sub-score IDs, criterion-to-slug mapping, JSON shapes for ratings.jsonl / manifest / calibration_report all consistent across tasks. `kappa_thresh` rename in T12 explicitly noted to avoid collision with per-sub-score `kappa`. |
| Group correctness | Group A: T1, T2, T3, T4, T5, T6 — six different files, all parallel. Group B: T7 (select_sample.py), T8 (rater_cli.py), T9 (analyze_calibration.py) — three different files, parallel; each chained to its predecessor. Group C, D similarly file-disjoint within group. Group E: only T16. No within-group file collisions. |
| Vertical slice check | Every task is one test + one implementation + one commit. T7-T16 all extend earlier modules but each one introduces a single new behavior with a single new test. |
| Behavior test check | All tests exercise public functions through their declared interface. No mocks of internal collaborators (the `judge_callable` and `input_provider` parameters in T2 and T8 are EXPLICIT injection points designed into the public API, not mocks of internals). No private-method tests. No internal-state assertions. |
| Forbidden-pattern scan | No bulk test scaffolding. No "test all the shapes." No tasks reference each other's code by name without including it. |

**Plan written and committed below. Run `/challenge` on it before executing.**

---

## Challenge Review

### CEO Pass

**Premise.** The protocol is on the right track: the spec's "weighted κ over Pearson r" + 4-bucket routing + Phase-1/Phase-2 partition is a defensible solution to a real, named blocker on the Stage 2/Stage 4 critical path. No reframing needed.

**Scope.** The plan implements 7 modules cleanly via 16 vertical-slice tasks. However, the plan **silently drops three spec-required behaviors** that turn the calibration artifact from a defensible filter into an unverified one (see BLOCKERs below).

**12-Month Alignment.**
```
TODAY                 →  THIS PLAN              →  IDEAL (Q3-Q4 2026)
ad-hoc sonnet-2.483   →  bucketed κ filter      →  re-derivable filter at every model bump
no defensibility         on 11 sub-scores          + Phase-2 expert layer for 3 expert dims
```
Direction is correct. The artifact is a Python module — easily versioned and re-emitted.

**Alternatives.** Spec lists 5 trade-offs with rejected alternatives. Sufficient.

```
[BLOCKER] (confidence: 10/10) — Sample-selection infeasibility against the real
  baseline_v1.jsonl. I scanned the actual 920 valid rows. Distribution by band
  is: threshold=302, high=295, low=40, weak_dim=95, gap=188 (excluded). With
  band targets {threshold:80, high:40, low:40, weak_dim:40} AND holdout_n=30
  reserved BEFORE selection (T10's design), the `low` band has 40 candidates,
  needs 40 main + ~1-2 holdout draws (probabilistic) = INFEASIBLE. T10 will
  raise ValueError on real data. Either: (a) reduce low-band target to 30 and
  document, (b) loosen low cutoff to ≤ 2.1, or (c) reserve holdout from a
  separate stratum. Pick one and freeze in the plan.

[BLOCKER] (confidence: 10/10) — Era min-quota infeasibility for Impressionist.
  Only 42 Debussy rows exist, of which 31 fall in valid bands (rest land in
  the 2.0-2.3 gap). Spec demands ≥30 Impressionist in main 200. After holdout
  draws ~1 Debussy probabilistically, only ~30 remain, with zero slack across
  the 4 score-bands. Combined with the band-target constraint above, the
  two-pass `_select_with_quotas` will deterministically fail on real data.
  Either drop Impressionist quota to 25 OR widen the band gap to fold rows
  in (2.0,2.3) into low/threshold.

[BLOCKER] (confidence: 10/10) — Skill-bucket min-quotas are in the spec
  (`docs/specs/2026-05-08-rubric-human-calibration-phase1-design.md` §Sample
  design: "beginner ≥50, intermediate ≥50, advanced ≥50") but **completely
  absent from the plan**. No task adds skill_bucket constraint to
  _select_with_quotas. This is silent spec drift. Add a task in Group B/C
  parallel to T7 (era quotas) that enforces skill-bucket quotas, or
  explicitly amend the spec to drop this requirement.

[BLOCKER] (confidence: 9/10) — `analyze_drift.py` never implements
  judge-vs-judge κ. Spec requires "judge-vs-judge κ ≥ 0.85" gate using
  day1/day30 re-runs (drift_report.judge_drift_kappa). T5 stubs this as
  `"judge_drift_kappa": {}` and no subsequent task implements it.
  judge_rerun.py (T2) writes day1/day30 jsonl that nothing consumes. Without
  this, the drift gate cannot pass and Open Question #2 in the spec
  (re: judge-drift escalation) is structurally unanswerable. Add a task to
  Group B/C that reads judge_runs_path and computes per-sub-score
  judge-drift κ.
```

### Engineering Pass

**Architecture.** Module decomposition matches the spec's table (7 deep modules), Hono-style boundaries, no MUC/Aria service coupling. Sub-score ID convention is consistent across all 16 tasks.

**Module depth.**
- `era_lookup.py` — interface 1 fn / impl ~10 LOC. Borderline shallow but justified by stable contract. **DEEP enough.**
- `select_sample.py` — 1 fn / hides band classification, era quotas, holdout reservation, anchor injection. **DEEP.**
- `rater_cli.py` — 4 fns (redact, capture, resume, SessionCapExceeded) / hides blinding allowlist, append-only IO, cap arithmetic. **DEEP.**
- `analyze_calibration.py` — 1 public `calibrate` / hides weighted κ, threshold agreement, variance, bucket routing. **DEEP.**
- `emit_recipe.py` — 1 fn / hides bucket→recipe field mapping. **DEEP.**

**Code quality.**

```
[RISK] (confidence: 8/10) — T7's `_select_with_quotas` can overshoot band
  targets through the era-quota fallback loop (lines after "If we did not
  hit min_q because some bands were full"), and Pass 2's `if remaining <= 0:
  continue` skips overshot bands. Final `len(main)` may exceed `target_n`,
  which breaks the T3 invariant `sum(band_counts.values()) == 200`. T3's
  test still runs after T7's edits. Verify this with the actual data
  distribution; consider a deterministic post-pass that trims overshoot.

[RISK] (confidence: 7/10) — T13 places anchors at `display_position =
  len(main) + k` — i.e., all 20 anchors come after every main rating. Spec
  says "silent duplicates within the main 200 (founder must not recognize)."
  Tail-clustering breaks blinding: rater notices "every synthesis past index
  200 is a duplicate." Interleave by selecting random positions ≥ N/2 (say,
  index 100..200) and inserting into the schedule rather than appending.

[RISK] (confidence: 7/10) — Session cap (15) interacts pathologically with
  11 sub-scores per synthesis: only 1 synthesis per session fits. At 2
  sessions/day → 2 syntheses/day → 100 days for main 200, vs spec's 4-6
  weeks. Either raise the cap to 22 (allows 2 per session) or revise the
  4-6 week estimate.

[RISK] (confidence: 6/10) — emit_recipe.emit() does not check
  `aggregate_gate_pass`. If <7/11 sub-scores reach TRUSTED status, the
  recipe is still written and importable. Stage 2 could consume an
  unvalidated recipe. Add a `raise CalibrationGateFailed` at the top of
  emit() unless `calibration_report["aggregate_gate_pass"]` is True, or
  emit a deliberately-empty recipe with a warning constant.

[RISK] (confidence: 6/10) — analyze_drift's intra-rater κ pairing logic
  (T5) keys `first_seen` by `(synth_id, sub_score)`. The duplicate carries
  `anchor_origin_id=origin`, but the original carries
  `anchor_origin_id=None`. If the rater happens to revisit a synth_id in
  any other context (e.g., partial restart that was logged then re-rated),
  the second occurrence is silently skipped. Acceptable for the protocol's
  controlled scope but document the assumption.

[OBS] — Spec calls for "bootstrap CI" on per-sub-score κ. Plan implements
  point estimate only. Not a blocker — bootstrap CI can be a follow-up task
  outside this plan's "first defensible artifact" scope. Note the gap
  explicitly in the spec's accepted trade-offs.

[OBS] — Spec calls for a "30-of-200 pilot flag" for end-of-week-1
  analysis. Plan's manifest has no pilot_indices field. Add as a stretch
  task or document deferral.
```

**Test philosophy.** Tests target public functions (`composer_to_era`,
`select_sample`, `redact_for_rater`, etc.). The injectable `judge_callable`
(T2) and `input_provider` (T8) parameters are EXPLICIT public DI points,
not internal-collaborator mocks. Anchor-pair pairing test (T5) constructs
real ratings.jsonl and reads via the function's documented file contract.
**Behavior-based, no shape tests, no internal mocks. Clean.**

**Vertical slice.** Every task has exactly one new test + one new
implementation + one commit. T7-T16 each extend earlier modules with one
new behavior. Group ordering is correct: same-file tasks chained, separate
files parallelized.

**Test coverage.**
```
[+] select_sample.py
    ├── [TESTED]  band proportions (T3) ★★
    ├── [TESTED]  determinism (T3) ★★
    ├── [TESTED]  era min-quotas (T7) ★★
    ├── [TESTED]  holdout disjoint (T10) ★★
    ├── [TESTED]  anchor injection (T13) ★★
    ├── [GAP]     **skill-bucket quota** — see BLOCKER
    ├── [GAP]     era infeasibility error path on tight pool — only happy path tested
    └── [GAP]     overshoot of `target_n` (RISK above)

[+] analyze_drift.py
    ├── [TESTED]  intra-rater κ (T5) ★★
    └── [GAP]     **judge-vs-judge κ** — see BLOCKER

[+] rater_cli.py
    ├── [TESTED]  redaction allowlist (T4) ★★★ (sentinel test is excellent)
    ├── [TESTED]  capture loop (T8) ★★
    ├── [TESTED]  session cap (T11) ★★
    └── [TESTED]  resume state (T14) ★★

[+] analyze_calibration.py
    ├── [TESTED]  per-sub-score κ extremes (T6) ★★★
    ├── [TESTED]  threshold agreement (T9) ★★
    └── [TESTED]  bucket routing 4 cases (T12) ★★★

[+] emit_recipe.py
    ├── [TESTED]  TRUSTED → weighted, CEILING → sanity, UNTRUSTED → drop (T15) ★★
    ├── [TESTED]  bias correction (T16) ★★
    └── [GAP]     refuses to emit if aggregate_gate_pass=False (RISK above)
```

**Failure modes.** Append-only ratings.jsonl + resume state (T14) handles
crash recovery cleanly. Session cap raises BEFORE any partial write (T11).
No silent failures on the rating path.

```
[QUESTION] — When _select_with_quotas raises ValueError on infeasible
  pool, what does the operator do? Plan has no fallback / config-knob
  pathway. Either pin the pool size constants (target_n, holdout_n) to
  values that are feasible against real data, or expose them as CLI args
  with a documented "tightening" procedure.

[QUESTION] — The 11 Phase-1 sub-scores include `concrete_artifact_outcome`
  and `praise_outcome` (constrained-rubric outcome dims). The spec says
  these use `muq_means` and `skill_bucket` as ground truth. Where in the
  rater_cli does the operator see this constraint? T8's `input_provider`
  takes the redacted row and a sub-score string but no rubric guidance.
  Is the rater expected to apply this rule from memory, or is there a
  per-sub-score prompt template missing from the plan?
```

### Presumption Inventory

| Assumption | Verdict | Reason |
|---|---|---|
| baseline_v1.jsonl has ≥40 rows in `low` band (≤2.0) | RISKY | Real data: exactly 40. Holdout reservation breaks this. |
| baseline_v1.jsonl has ≥30 Debussy/Impressionist in valid bands | RISKY | Real data: ~31 in valid bands. Zero slack. |
| Skill-bucket constraints are not required | RISKY | Spec requires them; plan silently drops. |
| `judge_dimensions[i].score` is always 0..3 int | SAFE | Verified — all rows match. |
| `judge_callable` injection is acceptable as public API | SAFE | Documented in module docstring. |
| Anchor blinding holds when all anchors appear after main | RISKY | Tail clustering exposes pattern; spec says "within main 200." |
| 4-6 week rating timeline at 15-rating session cap | RISKY | Math implies ~14 weeks at 2 sessions/day. |
| Stage 2 SFT consumer signature matches `WEIGHTED_SUB_SCORES`/`BIAS_CORRECTIONS`/`SANITY_FILTERS`/`COMPOSITE_PASS_THRESHOLD` | VALIDATE | No Stage-2 consumer file exists yet to confirm contract. |
| `aggregate_gate_pass=False` consumer behavior is "don't import" | VALIDATE | emit_recipe writes a fully-loaded recipe regardless. Clarify or guard. |
| `analyze_drift.judge_drift_kappa={}` is acceptable | RISKY | Spec gate requires this — must be implemented. |

### Summary

- [BLOCKER] count: 4
- [RISK]    count: 6
- [QUESTION] count: 2
- [OBS]     count: 2

VERDICT: NEEDS_REWORK — resolve before /build:
  1. Sample design feasibility against real `baseline_v1.jsonl` (low band, Impressionist quota, holdout interaction).
  2. Add skill-bucket min-quotas task (or amend spec to drop them).
  3. Add task implementing `judge_drift_kappa` in `analyze_drift.py`, consuming T2's day1/day30 jsonl.
  4. Decide and document whether anchors interleave or tail-cluster, and whether `emit_recipe` refuses to ship when `aggregate_gate_pass=False`.
