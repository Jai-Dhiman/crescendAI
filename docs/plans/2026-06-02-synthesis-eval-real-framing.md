# Synthesis Eval Real-Framing Re-Baseline Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** Re-measure the teacher-synthesis ASCF baseline through the real production framing (`buildSynthesisFraming`, driven by the `SessionBrain` Durable Object) instead of a thin Python strawman, and delete the divergent Python framing ports so a single source of truth remains.
**Spec:** docs/specs/2026-06-02-synthesis-eval-real-framing-design.md
**Style:** Follow the project's coding standards (CLAUDE.md / AGENTS.md). Python: `uv`, not pip. Explicit exception handling, no silent fallbacks. No emojis.

---

## Context the build agent must know

- All eval code lives under `apps/evals/`. Run Python from that dir: `cd apps/evals && uv run pytest ...`.
- The DO-replay driver already exists and is unchanged: `run_recording` in `apps/evals/shared/pipeline_client.py:105` returns a `SessionResult` whose `.synthesis.text` is the real production synthesis (built by `buildSynthesisFraming`). `SessionResult` fields: `.synthesis` (a `SynthesisResult` with `.text: str`, `.is_fallback: bool`, `.eval_context: dict`) or `None`; `.errors: list[str]`; `.synthesis_latency_ms: int`; `.piece_identification` (a `PieceIdentification` with `.piece_id: str`) or `None`.
- The judge call is unchanged: `judge_synthesis_v2(synthesis_text, context, provider, model)` from `shared.judge` returns a `JudgeResultV2` with `.dimensions: list[DimensionScore]` (each `.criterion: str`, `.process: int|None`, `.outcome: int|None`, `.score: int|None`, `.evidence: str`, `.reason: str`), `.model: str`, `.latency_ms: float`.
- Provenance is unchanged: `make_run_provenance()` from `shared.provenance` returns a `RunProvenance` with `.run_id`, `.git_sha`, `.git_dirty`.
- The aggregator (`apps/evals/teaching_knowledge/scripts/aggregate.py`) reads JSONL rows and uses **only** `row["judge_dimensions"][].{ "criterion", "outcome" }` and skips any row where `row.get("error")` is truthy. Its output JSON has `dimensions: [{ "name", "mean_outcome", "n", ... }]` and `composite_mean`.
- The aggregator dossier test (`apps/evals/teacher_model/stage0/tests/test_aggregator.py`) holds the locked baseline as `_SONNET_BASELINE = { "dimensions": [{"name","mean_outcome","n"}...], "composite_mean": ... }`. The currently locked ASCF row is `{"name": "Audible-Specific Corrective Feedback", "mean_outcome": 1.387, "n": 504}`.
- Holdout: `apps/evals/teacher_model/stage0/data/stage0_holdout.jsonl`, 98 rows. Each row has `recording_id`, `composer`, `title`, `skill_bucket`, `piece_slug`, `briefing_path` (absolute path to the inference-cache JSON: `{ "recording_id", "chunks": [...], "total_duration_seconds" }`).

---

## Task Groups

```
Group A (parallel): Task 1, Task 2, Task 3
Group B (sequential, depends on A): Task 4, Task 5
Group C (parallel, depends on B): Task 6, Task 9
Group D (manual, depends on C): Task 7, Task 8
```

- **Group A** — independent deletions of the thin-framing duplicates and their tests. Different files, no overlap, fully parallel. `[SHIPS INDEPENDENTLY]`: after Group A the divergent framing source is gone (the drift is deleted) even before the new baseline is measured.
- **Group B** — build the new DO-path row-builder (Task 4) and the runner shell + CLI wiring (Task 5). Task 5 depends on Task 4's `build_do_row`. Both touch `run_eval.py`, so they are sequential, not parallel.
- **Group C** — repoint the two remaining `build_synthesis_user_msg` consumers onto the new DO path: `cli.py synthesis` (Task 6) and `regen_calibration_baseline.py` (Task 9). Both depend on Task 5's `run_do_baseline` / Task 4's `build_do_row` existing; they touch different files (`cli.py` vs `regen_calibration_baseline.py`) so they run in parallel.
- **Group D** — the live-DO measurement (Task 7) and the baseline re-lock + checklist update (Task 8). Manual: requires `wrangler dev`. Task 8 depends on Task 7's output file.

**Consumers of the deleted `build_synthesis_user_msg` (all four must be handled, none left silently broken):**

| File | Consumes | Handled by |
|---|---|---|
| `apps/evals/teaching_knowledge/run_eval.py` | defines it + calls it in `run()` | Task 1 (remove) |
| `apps/evals/teaching_knowledge/test_run_eval_blocks.py` | imports it | Task 3 (delete) |
| `apps/evals/tests/test_run_eval_style_injection.py` | imports it | Task 3 (delete) |
| `apps/evals/teacher_model/calibration/regen_calibration_baseline.py` | imports it (lines 64-67) + calls it (~line 119) | **Task 9 (repoint onto DO path)** |

`apps/evals/teacher_model/stage0/run_synthesis.py` does **not** import `build_synthesis_user_msg` (it builds its own user message via the injected `teacher_client`); it is deleted in Task 3 because it is the *second* thin-framing holdout runner, and `cli.py:11` imports it (handled by Task 6).

**Group A and Group B can run concurrently** (Group A deletes `bar_analysis_local.py`, `run_synthesis.py`, and 4 test files; Group B edits `run_eval.py`). The only shared file is `run_eval.py`, which Group A only touches in Task 1 (removing the `build_synthesis_user_msg` body + its imports). To avoid a write conflict, **Task 1 is the single owner of removing `build_synthesis_user_msg` from `run_eval.py`, and Group B Tasks 4/5 are sequenced after Task 1.** Therefore: Group A runs first (or Task 1 first within a merged A+B ordering); Tasks 4/5/6 follow.

Revised ordering to remove the `run_eval.py` write conflict:

```
Group A (parallel): Task 1 (run_eval.py), Task 2 (delete bar_analysis_local + 2 tests), Task 3 (delete run_synthesis + its test)
Group B (sequential, after A): Task 4 (run_eval.py: build_do_row), Task 5 (run_eval.py: run_do_baseline + CLI)
Group C (parallel, after B): Task 6 (cli.py repoint), Task 9 (regen_calibration_baseline.py repoint)
Group D (manual, after C): Task 7 (live measure), Task 8 (re-lock + checklist)
```

---

### Task 1: Remove the thin-framing user-message builder from `run_eval.py`

**Group:** A (parallel with Task 2, Task 3)

**Behavior being verified:** `run_eval.py` no longer exposes `build_synthesis_user_msg`, and importing the module no longer pulls in the deleted `bar_analysis_local` / `piece_score_map` framing helpers.

**Interface under test:** the `teaching_knowledge.run_eval` module's public symbols.

**Files:**
- Modify: `apps/evals/teaching_knowledge/run_eval.py`
- Test: `apps/evals/teaching_knowledge/tests/test_run_eval_no_thin_framing.py`

- [x] **Step 1: Write the failing test**

```python
# apps/evals/teaching_knowledge/tests/test_run_eval_no_thin_framing.py
from __future__ import annotations

import teaching_knowledge.run_eval as run_eval


def test_thin_framing_builder_is_gone() -> None:
    assert not hasattr(run_eval, "build_synthesis_user_msg")


def test_module_imports_without_bar_analysis_local() -> None:
    # Importing run_eval must not require the deleted bar_analysis_local module.
    import importlib

    mod = importlib.reload(run_eval)
    src = mod.__file__
    text = open(src).read()
    assert "bar_analysis_local" not in text
    assert "piece_score_map" not in text
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teaching_knowledge/tests/test_run_eval_no_thin_framing.py -q
```
Expected: FAIL — `assert not hasattr(run_eval, "build_synthesis_user_msg")` fails because the function still exists.

- [x] **Step 3: Implement the minimum to make the test pass**

In `apps/evals/teaching_knowledge/run_eval.py`:

1. Delete the entire `build_synthesis_user_msg` function (lines defining `def build_synthesis_user_msg(...)` through its `return "\n".join(parts)`).
2. The `from teaching_knowledge.bar_analysis_local import build_bar_analysis` and `from teaching_knowledge.piece_score_map import get_score_path_for_piece` imports live **inside** `build_synthesis_user_msg`, so they are removed with it. Confirm no remaining top-level reference to `bar_analysis_local` or `piece_score_map` exists in the file.
3. In `run(...)`, the call site `user_msg = build_synthesis_user_msg(muq_means, duration_seconds, meta, chunks=chunks)` and the `synthesis_client.complete(...)` block that follows it will be replaced wholesale in Task 5. For Task 1, leave the legacy `run(...)` body intact **except** remove the `build_synthesis_user_msg` call by raising a clear NotImplementedError so the legacy path cannot silently run the deleted builder:

```python
            try:
                raise NotImplementedError(
                    "Thin-framing synthesis path removed (issue #22). "
                    "Use run_do_baseline via --do-path."
                )
```

   Place this `raise` as the first statement inside the existing `try:` in `run(...)`, replacing the `user_msg = build_synthesis_user_msg(...)` line and the subsequent `synthesis_client.complete(...)` / `extract_teacher_response(...)` / judge block down to the matching `except Exception as exc:`. Keep the `except Exception as exc:` error-row handling as-is so the module still parses. (Task 5 removes this stub and wires `run_do_baseline`.)
4. Keep `aggregate_muq`, `extract_teacher_response`, `load_manifests`, `load_completed_ids`, `_build_row`, `_filter_cache_files_by_split`, and the `SCALER_MEAN`/`DIMS` constants — Task 4 reuses `aggregate_muq` and `_build_row` is still imported elsewhere.

- [x] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teaching_knowledge/tests/test_run_eval_no_thin_framing.py -q
```
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add apps/evals/teaching_knowledge/run_eval.py apps/evals/teaching_knowledge/tests/test_run_eval_no_thin_framing.py
git commit -m "refactor(evals): remove thin-framing build_synthesis_user_msg from run_eval (#22)"
```

---

### Task 2: Delete `bar_analysis_local` and its tests

**Group:** A (parallel with Task 1, Task 3)

**Behavior being verified:** the bar-analysis framing port and its tests no longer exist; nothing under `teaching_knowledge/tests` imports them.

**Interface under test:** filesystem + collectability of the test suite.

**Files:**
- Delete: `apps/evals/teaching_knowledge/bar_analysis_local.py`
- Delete: `apps/evals/teaching_knowledge/tests/test_bar_analysis_local.py`
- Delete: `apps/evals/teaching_knowledge/tests/test_run_eval_bar_analysis.py`
- Test: `apps/evals/teaching_knowledge/tests/test_no_bar_analysis_local.py`

- [x] **Step 1: Write the failing test**

```python
# apps/evals/teaching_knowledge/tests/test_no_bar_analysis_local.py
from __future__ import annotations

from pathlib import Path


def test_bar_analysis_local_module_deleted() -> None:
    here = Path(__file__).resolve().parents[1]  # teaching_knowledge/
    assert not (here / "bar_analysis_local.py").exists()


def test_bar_analysis_local_not_importable() -> None:
    import importlib

    try:
        importlib.import_module("teaching_knowledge.bar_analysis_local")
    except ModuleNotFoundError:
        return
    raise AssertionError("teaching_knowledge.bar_analysis_local should not be importable")
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teaching_knowledge/tests/test_no_bar_analysis_local.py -q
```
Expected: FAIL — `bar_analysis_local.py` still exists, so the first assertion fails.

- [x] **Step 3: Implement the minimum to make the test pass**

```bash
git rm apps/evals/teaching_knowledge/bar_analysis_local.py \
       apps/evals/teaching_knowledge/tests/test_bar_analysis_local.py \
       apps/evals/teaching_knowledge/tests/test_run_eval_bar_analysis.py
```

- [x] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teaching_knowledge/tests/test_no_bar_analysis_local.py -q
```
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add -A apps/evals/teaching_knowledge/
git commit -m "chore(evals): delete bar_analysis_local framing port + tests (#22)"
```

---

### Task 3: Delete the stage0 thin-framing synthesis runner and its tests

**Group:** A (parallel with Task 1, Task 2)

**Behavior being verified:** the second thin-framing holdout runner (`stage0/run_synthesis.py`) and its style-injection / blocks tests are gone and not importable.

**Interface under test:** filesystem + importability.

**Files:**
- Delete: `apps/evals/teacher_model/stage0/run_synthesis.py`
- Delete: `apps/evals/teacher_model/stage0/tests/test_run_synthesis.py`
- Delete: `apps/evals/teaching_knowledge/test_run_eval_blocks.py`
- Delete: `apps/evals/tests/test_run_eval_style_injection.py`
- Test: `apps/evals/teacher_model/stage0/tests/test_no_run_synthesis.py`

> Note: `test_run_eval_blocks.py` and `test_run_eval_style_injection.py` both import `build_synthesis_user_msg` (deleted in Task 1) and would fail collection otherwise; they test the deleted thin framing, so they are removed here.

- [x] **Step 1: Write the failing test**

```python
# apps/evals/teacher_model/stage0/tests/test_no_run_synthesis.py
from __future__ import annotations

import importlib
from pathlib import Path


def test_stage0_run_synthesis_module_deleted() -> None:
    here = Path(__file__).resolve().parents[1]  # stage0/
    assert not (here / "run_synthesis.py").exists()


def test_stage0_run_synthesis_not_importable() -> None:
    try:
        importlib.import_module("teacher_model.stage0.run_synthesis")
    except ModuleNotFoundError:
        return
    raise AssertionError("teacher_model.stage0.run_synthesis should not be importable")
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/stage0/tests/test_no_run_synthesis.py -q
```
Expected: FAIL — `run_synthesis.py` still exists.

- [x] **Step 3: Implement the minimum to make the test pass**

```bash
git rm apps/evals/teacher_model/stage0/run_synthesis.py \
       apps/evals/teacher_model/stage0/tests/test_run_synthesis.py \
       apps/evals/teaching_knowledge/test_run_eval_blocks.py \
       apps/evals/tests/test_run_eval_style_injection.py
```

- [x] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/stage0/tests/test_no_run_synthesis.py -q
```
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add -A apps/evals/teacher_model/stage0/ apps/evals/teaching_knowledge/ apps/evals/tests/
git commit -m "chore(evals): delete stage0 thin-framing run_synthesis + style/blocks tests (#22)"
```

---

### Task 4: `build_do_row` — turn a DO `SessionResult` into an aggregator JSONL row

**Group:** B (sequential, after Group A; depends on Task 1 having removed the thin path from `run_eval.py`)

**Behavior being verified:** given a captured DO `SessionResult` + holdout meta + a judge callable, the builder emits one aggregator-schema row (provenance-stamped, `judge_dimensions` populated, `piece_resolved` flagged), and maps a DO/WS failure into the `error` field instead of falling back to thin framing.

**Interface under test:** `build_do_row(...)` (a new public function in `run_eval.py`).

**Files:**
- Modify: `apps/evals/teaching_knowledge/run_eval.py`
- Test: `apps/evals/teaching_knowledge/tests/test_do_row.py`

- [x] **Step 1: Write the failing test**

```python
# apps/evals/teaching_knowledge/tests/test_do_row.py
from __future__ import annotations

from dataclasses import dataclass, field

from teaching_knowledge.run_eval import build_do_row
from shared.provenance import make_run_provenance


@dataclass
class _FakeDim:
    criterion: str
    process: int | None
    outcome: int | None
    score: int | None
    evidence: str = ""
    reason: str = ""


@dataclass
class _FakeJudgeResult:
    dimensions: list
    model: str = "fake-judge"
    latency_ms: float = 12.0


@dataclass
class _FakeSynthesis:
    text: str
    is_fallback: bool = False
    eval_context: dict = field(default_factory=dict)


@dataclass
class _FakeSessionResult:
    recording_id: str
    synthesis: object | None
    errors: list
    synthesis_latency_ms: int = 700
    piece_identification: object | None = None


@dataclass
class _FakePieceId:
    piece_id: str


_META = {
    "piece_slug": "fur_elise",
    "title": "Fur Elise",
    "composer": "Beethoven",
    "skill_bucket": 3,
}


def _judge_ok(synthesis_text, context, **kwargs):
    assert synthesis_text  # judge only called with real text
    return _FakeJudgeResult(
        dimensions=[
            _FakeDim("Audible-Specific Corrective Feedback", 2, 1, 1),
            _FakeDim("Specific Positive Praise", 3, 3, 3),
        ]
    )


def test_successful_session_yields_judged_row() -> None:
    prov = make_run_provenance()
    sr = _FakeSessionResult(
        recording_id="rec1",
        synthesis=_FakeSynthesis(text="Lovely phrasing, try softer pedaling."),
        errors=[],
        piece_identification=_FakePieceId("fur_elise"),
    )
    row = build_do_row(sr, _META, _judge_ok, prov)

    assert row["recording_id"] == "rec1"
    assert row["error"] == ""
    assert row["run_id"] == prov.run_id
    assert row["synthesis_text"] == "Lovely phrasing, try softer pedaling."
    assert row["piece_resolved"] is True
    crits = {d["criterion"]: d["outcome"] for d in row["judge_dimensions"]}
    assert crits["Audible-Specific Corrective Feedback"] == 1
    assert crits["Specific Positive Praise"] == 3


def test_do_failure_records_error_and_skips_judge() -> None:
    prov = make_run_provenance()
    sr = _FakeSessionResult(
        recording_id="rec2",
        synthesis=None,
        errors=["WebSocket error: connection refused"],
    )

    def _judge_must_not_run(*a, **k):  # noqa: ANN001, ANN002
        raise AssertionError("judge must not run when the DO failed")

    row = build_do_row(sr, _META, _judge_must_not_run, prov)

    assert row["recording_id"] == "rec2"
    assert "connection refused" in row["error"]
    assert row["judge_dimensions"] == []
    assert row["synthesis_text"] == ""
    assert row["piece_resolved"] is False


def test_unresolved_piece_flags_false_but_still_judges() -> None:
    prov = make_run_provenance()
    sr = _FakeSessionResult(
        recording_id="rec3",
        synthesis=_FakeSynthesis(text="Keep the line singing through the rests."),
        errors=[],
        piece_identification=None,  # piece not resolved -> Tier 2/3, bar_range null
    )
    row = build_do_row(sr, _META, _judge_ok, prov)
    assert row["error"] == ""
    assert row["piece_resolved"] is False
    assert len(row["judge_dimensions"]) == 2
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teaching_knowledge/tests/test_do_row.py -q
```
Expected: FAIL — `ImportError: cannot import name 'build_do_row' from 'teaching_knowledge.run_eval'`.

- [x] **Step 3: Implement the minimum to make the test pass**

Add to `apps/evals/teaching_knowledge/run_eval.py` (the `RunProvenance` type is already imported at the top of the file):

```python
def build_do_row(
    session_result,
    meta: dict[str, Any],
    judge_fn,
    provenance: RunProvenance,
    *,
    dry_run: bool = False,
    judge_provider: str = "workers-ai",
    judge_model: str = "@cf/google/gemma-4-26b-a4b-it",
) -> dict:
    """Build one aggregator-schema JSONL row from a DO SessionResult.

    No thin-framing fallback: a DO/WS failure (or missing synthesis) records the
    error verbatim and skips the judge. Piece resolution is reported via
    `piece_resolved` (False -> the DO ran Tier 2/3 with bar_range null).
    """
    recording_id = getattr(session_result, "recording_id", meta.get("recording_id", ""))
    piece_resolved = getattr(session_result, "piece_identification", None) is not None

    base = {
        "recording_id": recording_id,
        "run_id": provenance.run_id,
        "git_sha": provenance.git_sha,
        "git_dirty": provenance.git_dirty,
        "piece_slug": meta.get("piece_slug", ""),
        "title": meta.get("title", ""),
        "composer": meta.get("composer", ""),
        "skill_bucket": meta.get("skill_bucket", 3),
        "piece_resolved": piece_resolved,
        "synthesis_text": "",
        "synthesis_latency_ms": int(getattr(session_result, "synthesis_latency_ms", 0) or 0),
        "judge_dimensions": [],
        "judge_model": "",
        "judge_latency_ms": 0,
        "error": "",
    }

    synthesis = getattr(session_result, "synthesis", None)
    errors = getattr(session_result, "errors", []) or []
    if synthesis is None or not getattr(synthesis, "text", ""):
        base["error"] = "; ".join(errors) if errors else "DO returned no synthesis text"
        return base

    base["synthesis_text"] = synthesis.text

    if dry_run:
        base["judge_model"] = "dry_run"
        return base

    judge_ctx = {
        "piece_name": meta.get("title", ""),
        "composer": meta.get("composer", ""),
        "skill_level": meta.get("skill_bucket", 3),
    }
    jr = judge_fn(
        synthesis.text,
        judge_ctx,
        provider=judge_provider,
        model=judge_model,
    )
    base["judge_dimensions"] = [
        {
            "criterion": d.criterion,
            "process": getattr(d, "process", None),
            "outcome": getattr(d, "outcome", None),
            "score": getattr(d, "score", None),
            "evidence": getattr(d, "evidence", ""),
            "reason": getattr(d, "reason", ""),
        }
        for d in jr.dimensions
    ]
    base["judge_model"] = getattr(jr, "model", "")
    base["judge_latency_ms"] = round(getattr(jr, "latency_ms", 0.0))
    return base
```

Note: the test calls `build_do_row(sr, meta, judge_fn, prov)` with a judge that accepts `provider=`/`model=` kwargs (matching `judge_synthesis_v2`'s signature); `_judge_ok` ignores them via `**kwargs`.

- [x] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teaching_knowledge/tests/test_do_row.py -q
```
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add apps/evals/teaching_knowledge/run_eval.py apps/evals/teaching_knowledge/tests/test_do_row.py
git commit -m "feat(evals): build_do_row maps DO SessionResult to aggregator row (#22)"
```

---

### Task 5: `run_do_baseline` — drive the holdout through the DO and write JSONL

**Group:** B (sequential, after Task 4; same file `run_eval.py`)

**Behavior being verified:** the runner iterates holdout rows, loads each inference-cache briefing, drives it through the DO via an injected driver, judges via an injected judge, and writes one `build_do_row` row per recording — resuming over already-completed `recording_id`s. (Verified with injected fakes so no live DO is needed for the unit test; the live run is Task 7.)

**Interface under test:** `run_do_baseline(...)` (new public function in `run_eval.py`), with `driver` and `judge_fn` injected for testability.

**Files:**
- Modify: `apps/evals/teaching_knowledge/run_eval.py`
- Test: `apps/evals/teaching_knowledge/tests/test_run_do_baseline.py`

- [x] **Step 1: Write the failing test**

```python
# apps/evals/teaching_knowledge/tests/test_run_do_baseline.py
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from teaching_knowledge.run_eval import run_do_baseline


@dataclass
class _FakeDim:
    criterion: str
    process: int | None
    outcome: int | None
    score: int | None
    evidence: str = ""
    reason: str = ""


@dataclass
class _FakeJudgeResult:
    dimensions: list
    model: str = "fake-judge"
    latency_ms: float = 5.0


@dataclass
class _FakeSynthesis:
    text: str
    is_fallback: bool = False
    eval_context: dict = field(default_factory=dict)


@dataclass
class _FakeSessionResult:
    recording_id: str
    synthesis: object | None
    errors: list
    synthesis_latency_ms: int = 600
    piece_identification: object | None = None


def _judge(synthesis_text, context, **kwargs):
    return _FakeJudgeResult(dimensions=[_FakeDim("Audible-Specific Corrective Feedback", 2, 2, 2)])


def _write_briefing(path: Path, rid: str) -> None:
    path.write_text(json.dumps({
        "recording_id": rid,
        "total_duration_seconds": 120.0,
        "chunks": [{"chunk_index": 0, "predictions": {"dynamics": 0.5}}],
    }))


def _write_holdout(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_run_do_baseline_writes_one_row_per_recording(tmp_path: Path) -> None:
    b1 = tmp_path / "b1.json"
    b2 = tmp_path / "b2.json"
    _write_briefing(b1, "rA")
    _write_briefing(b2, "rB")
    holdout = tmp_path / "holdout.jsonl"
    _write_holdout(holdout, [
        {"recording_id": "rA", "title": "P1", "composer": "Bach", "skill_bucket": 3,
         "piece_slug": "bach_invention_1", "briefing_path": str(b1)},
        {"recording_id": "rB", "title": "P2", "composer": "Chopin", "skill_bucket": 4,
         "piece_slug": "chopin_ballade_1", "briefing_path": str(b2)},
    ])

    seen_recordings = []

    def _driver(wrangler_url, recording_cache, student_id, piece_query):
        seen_recordings.append((recording_cache["recording_id"], piece_query))
        return _FakeSessionResult(
            recording_id=recording_cache["recording_id"],
            synthesis=_FakeSynthesis(text=f"feedback for {recording_cache['recording_id']}"),
            errors=[],
        )

    out = tmp_path / "out.jsonl"
    run_do_baseline(
        holdout_path=holdout,
        out_path=out,
        wrangler_url="http://localhost:8787",
        judge_fn=_judge,
        driver=_driver,
    )

    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert {r["recording_id"] for r in rows} == {"rA", "rB"}
    # piece_slug is passed as the piece_query to set_piece
    assert ("rA", "bach_invention_1") in seen_recordings
    assert all(r["judge_dimensions"] for r in rows)


def test_run_do_baseline_resumes_completed_recordings(tmp_path: Path) -> None:
    b1 = tmp_path / "b1.json"
    _write_briefing(b1, "rA")
    holdout = tmp_path / "holdout.jsonl"
    _write_holdout(holdout, [
        {"recording_id": "rA", "title": "P1", "composer": "Bach", "skill_bucket": 3,
         "piece_slug": "bach_invention_1", "briefing_path": str(b1)},
    ])
    out = tmp_path / "out.jsonl"
    out.write_text(json.dumps({
        "recording_id": "rA", "synthesis_text": "already done",
        "judge_dimensions": [{"criterion": "x", "outcome": 2}], "error": "",
    }) + "\n")

    calls = []

    def _driver(wrangler_url, recording_cache, student_id, piece_query):
        calls.append(recording_cache["recording_id"])
        raise AssertionError("driver must not run for an already-completed recording")

    run_do_baseline(
        holdout_path=holdout,
        out_path=out,
        wrangler_url="http://localhost:8787",
        judge_fn=_judge,
        driver=_driver,
    )
    assert calls == []
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert len(rows) == 1
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teaching_knowledge/tests/test_run_do_baseline.py -q
```
Expected: FAIL — `ImportError: cannot import name 'run_do_baseline' from 'teaching_knowledge.run_eval'`.

- [x] **Step 3: Implement the minimum to make the test pass**

In `apps/evals/teaching_knowledge/run_eval.py`:

1. Leave the legacy `run(...)` function with its Task-1 `NotImplementedError` stub untouched — once `main()` routes `--do-path` to `run_do_baseline` (item 2 below), the legacy `run(...)` is no longer reachable for synthesis, and `main()`'s non-`--do-path` branch will hit the stub's clear error if invoked. Add `run_do_baseline` as a new top-level function:

```python
import asyncio


def _default_driver(wrangler_url, recording_cache, student_id, piece_query):
    """Default DO-replay driver: real run_recording over wrangler dev."""
    from shared.pipeline_client import run_recording

    return asyncio.run(
        run_recording(
            wrangler_url,
            recording_cache,
            student_id=student_id,
            piece_query=piece_query,
        )
    )


def run_do_baseline(
    holdout_path: Path,
    out_path: Path,
    wrangler_url: str,
    judge_fn,
    *,
    driver=_default_driver,
    limit: int | None = None,
    dry_run: bool = False,
    judge_provider: str = "workers-ai",
    judge_model: str = "@cf/google/gemma-4-26b-a4b-it",
    student_id: str = "eval-student-001",
) -> None:
    """Drive holdout recordings through the real SessionBrain DO and write JSONL.

    `driver(wrangler_url, recording_cache, student_id, piece_query) -> SessionResult`
    is injected so the orchestration is unit-testable without a live DO.
    The default driver calls shared.pipeline_client.run_recording.
    """
    holdout = [
        json.loads(line)
        for line in holdout_path.read_text().splitlines()
        if line.strip()
    ]
    if limit is not None:
        holdout = holdout[:limit]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed_ids(out_path)
    provenance = make_run_provenance()
    print(f"run_id: {provenance.run_id} | DO path | wrangler={wrangler_url}")

    processed = 0
    errors = 0

    with out_path.open("a") as fout:
        for entry in holdout:
            recording_id = entry["recording_id"]
            if recording_id in completed:
                continue

            meta = {
                "recording_id": recording_id,
                "piece_slug": entry.get("piece_slug", ""),
                "title": entry.get("title", ""),
                "composer": entry.get("composer", ""),
                "skill_bucket": int(entry.get("skill_bucket", 3)),
            }

            briefing = json.loads(Path(entry["briefing_path"]).read_text())
            recording_cache = {
                "recording_id": briefing.get("recording_id", recording_id),
                "chunks": briefing.get("chunks", []),
            }

            try:
                session_result = driver(
                    wrangler_url,
                    recording_cache,
                    student_id,
                    meta["piece_slug"] or None,
                )
                row = build_do_row(
                    session_result,
                    meta,
                    judge_fn,
                    provenance,
                    dry_run=dry_run,
                    judge_provider=judge_provider,
                    judge_model=judge_model,
                )
            except Exception as exc:  # driver/judge hard failure -> error row, never thin fallback
                row = {
                    "recording_id": recording_id,
                    "run_id": provenance.run_id,
                    "git_sha": provenance.git_sha,
                    "git_dirty": provenance.git_dirty,
                    "piece_slug": meta["piece_slug"],
                    "title": meta["title"],
                    "composer": meta["composer"],
                    "skill_bucket": meta["skill_bucket"],
                    "piece_resolved": False,
                    "synthesis_text": "",
                    "synthesis_latency_ms": 0,
                    "judge_dimensions": [],
                    "judge_model": "",
                    "judge_latency_ms": 0,
                    "error": str(exc)[:500],
                }

            if row.get("error"):
                errors += 1
            fout.write(json.dumps(row) + "\n")
            fout.flush()
            processed += 1
            print(f"[{processed}] {recording_id} | {meta['piece_slug']}"
                  + (f" | ERROR: {row['error'][:80]}" if row.get("error") else ""))

    print(f"Done. processed={processed} errors={errors} -> {out_path}")
```

   Note on `errors`: count a row as an error iff its final `row["error"]` is non-empty (whether the failure came from the `except` branch or from `build_do_row` recording a DO/no-synthesis error). Do NOT also increment in the `except` branch — that would double-count. Remove the `errors += 1` inside the `except` block above and rely solely on the single post-row `if row.get("error"): errors += 1` check shown here.

2. Add a `--do-path` flag and `--wrangler-url` to `main()` and route to `run_do_baseline` when set:

```python
    parser.add_argument("--do-path", action="store_true",
                        help="Drive synthesis through the real SessionBrain DO (requires wrangler dev).")
    parser.add_argument("--wrangler-url", default="http://localhost:8787",
                        help="wrangler dev base URL for --do-path (default: %(default)s).")
```

   In `main()`, before the legacy `run(...)` call:

```python
    if args.do_path:
        from shared.judge import judge_synthesis_v2
        judge_provider = "openrouter" if "/" in args.judge_model and not args.judge_model.startswith("@cf/") else "workers-ai"
        holdout = EVALS_ROOT / "teacher_model" / "stage0" / "data" / "stage0_holdout.jsonl"
        out = args.out if args.out is not None else (RESULTS_DIR / "baseline_v2_do.jsonl")
        run_do_baseline(
            holdout_path=holdout,
            out_path=out,
            wrangler_url=args.wrangler_url,
            judge_fn=judge_synthesis_v2,
            limit=args.limit,
            dry_run=args.dry_run,
            judge_provider=judge_provider,
            judge_model=args.judge_model,
        )
        return
```

- [x] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teaching_knowledge/tests/test_run_do_baseline.py -q
```
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add apps/evals/teaching_knowledge/run_eval.py apps/evals/teaching_knowledge/tests/test_run_do_baseline.py
git commit -m "feat(evals): run_do_baseline drives holdout through real DO synthesis (#22)"
```

---

### Task 6: Repoint `cli.py synthesis` onto the DO-path runner

**Group:** C (parallel with Task 9, after Group B; depends on `run_do_baseline` existing)

**Behavior being verified:** the stage0 `synthesis` subcommand no longer imports the deleted `run_synthesis`; invoking it routes through `run_do_baseline` (the real-framing path), **preserving the 9-dim `judge_extended` judge** so the downstream `build_dossier` capability mapping is unchanged.

> **Verified import structure (do not skip).** `cli.py` imports the deleted runner at **module top, line 11**: `from teacher_model.stage0.run_synthesis import run as run_synthesis` — NOT inside the dispatch branch. The current `synthesis` dispatch branch (~lines 133-147) calls `run_synthesis(holdout_path=..., out_path=args.out, teacher_client=client, judge_fn=judge_extended, system_prompt=_SYNTH_SYSTEM.read_text(), judge_provider=..., judge_model=..., limit=...)` — it uses `judge_extended` (the stage0 9-dim extended judge), **not** `judge_synthesis_v2`.
>
> **Sequencing consequence:** Task 3 deletes `run_synthesis.py`. From the moment Task 3 lands until this task (Task 6) lands, the module-top import at `cli.py:11` is unsatisfiable, so importing/collecting `cli.py` — and **any** test that imports it — fails. Two tests import `cli.py` at module top: the pre-existing `teacher_model/stage0/tests/test_cli.py` (`from teacher_model.stage0.cli import _build_parser, sample_main`) and the new `test_cli_synthesis_repointed.py` added here. For the new test this is exactly the intended Step-2 "verify it FAILS" (collection error). For the pre-existing `test_cli.py`, this is a **known, contained intermediate-state breakage**: under subagent-driven TDD each task runs only its own test command (Task 6 Step 2/4 run `test_cli_synthesis_repointed.py` only), and the first full-suite collection (`uv run pytest teaching_knowledge teacher_model/stage0`) is Task 8 Step 4 — which runs strictly **after** Task 6 has removed `cli.py:11` and restored collectability. The window therefore never overlaps a full-suite gate. **Guard for the build agent:** do not run the whole `teacher_model/stage0` suite between Task 3 and Task 6; if you must, expect `test_cli.py` + `test_cli_synthesis_repointed.py` collection errors until Task 6 lands.
>
> **Judge-swap resolution (BLOCKER 2):** do NOT swap to `judge_synthesis_v2`. The stage0 `synthesis` subcommand feeds `build_dossier`'s `synthesis_jsonl`, and `aggregator.py` reads 9-dim criteria including `Taste Defensibility` and `Adaptation Specificity` that only `judge_extended` emits (`judge_synthesis_v2` emits 7 dims and omits them, which would silently blank the Taste/Adaptation capability rows). `run_do_baseline`/`build_do_row` call the judge as `judge_fn(text, ctx, provider=, model=)`, and `judge_extended(synthesis_text, context, provider=, model=)` is call-compatible — so pass `judge_fn=judge_extended` explicitly to preserve the existing 9-dim measurement.

**Interface under test:** `teacher_model.stage0.cli` import + its `synthesis` dispatch.

**Files:**
- Modify: `apps/evals/teacher_model/stage0/cli.py`
- Test: `apps/evals/teacher_model/stage0/tests/test_cli_synthesis_repointed.py`

- [x] **Step 1: Write the failing test**

```python
# apps/evals/teacher_model/stage0/tests/test_cli_synthesis_repointed.py
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock


def test_cli_imports_without_run_synthesis() -> None:
    import importlib

    mod = importlib.import_module("teacher_model.stage0.cli")
    importlib.reload(mod)
    text = Path(mod.__file__).read_text()
    assert "run_synthesis" not in text
    assert "run_do_baseline" in text


def test_synthesis_subcommand_calls_run_do_baseline_with_judge_extended(tmp_path: Path) -> None:
    import teacher_model.stage0.cli as cli
    from teacher_model.stage0.judge_extended import judge_extended

    out = tmp_path / "synth.jsonl"
    argv = ["cli", "synthesis", "--provider", "anthropic",
            "--model", "claude-sonnet-4-6", "--out", str(out)]
    with mock.patch.object(sys, "argv", argv), \
         mock.patch("teaching_knowledge.run_eval.run_do_baseline") as m:
        cli.main()
    assert m.called
    kwargs = m.call_args.kwargs
    assert kwargs["out_path"] == out
    # BLOCKER 2: preserve the 9-dim stage0 judge; do NOT swap to judge_synthesis_v2.
    assert kwargs["judge_fn"] is judge_extended
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/stage0/tests/test_cli_synthesis_repointed.py -q
```
Expected: FAIL — `cli.py` still has the module-top import at line 11 (`from teacher_model.stage0.run_synthesis import run as run_synthesis`), and because Task 3 already deleted `run_synthesis.py`, importing/collecting `cli.py` raises `ModuleNotFoundError` at line 11 before either test body runs. (This is the anticipated intermediate-state failure described in the Verified-import-structure note above.)

- [x] **Step 3: Implement the minimum to make the test pass**

In `apps/evals/teacher_model/stage0/cli.py`:

1. Remove the module-top import at **line 11** exactly: `from teacher_model.stage0.run_synthesis import run as run_synthesis`. (This is the line that breaks `cli.py` collection once Task 3 deletes `run_synthesis.py`; it is NOT an in-branch import.) Keep the existing `from teacher_model.stage0.judge_extended import judge_extended` import (line 8) — it is now used by the repointed branch, so do not remove it.
2. Replace the `synthesis` dispatch branch body (~lines 133-147, the block guarded by `if args.cmd == "synthesis":` that currently constructs an `LLMClient` and calls `run_synthesis(...)`) with a `run_do_baseline` call that **preserves `judge_extended`**:

```python
    if args.cmd == "synthesis":
        from teaching_knowledge.run_eval import run_do_baseline

        run_do_baseline(
            holdout_path=_DATA_DIR / "stage0_holdout.jsonl",
            out_path=args.out,
            wrangler_url=getattr(args, "wrangler_url", "http://localhost:8787"),
            judge_fn=judge_extended,  # BLOCKER 2: keep the 9-dim stage0 judge, not judge_synthesis_v2
            limit=args.limit,
            judge_provider=args.judge_provider,
            judge_model=args.judge_model,
        )
        return
```

   The previous branch's `LLMClient` construction (`from teaching_knowledge.llm_client import LLMClient; client = LLMClient(...)`) and the `system_prompt=_SYNTH_SYSTEM.read_text()` / `teacher_client=client` arguments are dropped: the DO now owns the synthesis prompt (`buildSynthesisFraming`), so the local teacher client and `_SYNTH_SYSTEM` are no longer used by this branch. If removing this branch leaves `_SYNTH_SYSTEM` or the `LLMClient` import unused **anywhere else in `cli.py`**, leave them (other subcommands `tool`/`continuation` may use `LLMClient`); only remove an import this change orphans. Confirm by grepping `cli.py` for `_SYNTH_SYSTEM` and `LLMClient` after the edit.

3. If the `synthesis` subparser lacks `--wrangler-url`, add it where that subparser is built (search for `sub.add_parser("synthesis"`):

```python
    sp.add_argument("--wrangler-url", default="http://localhost:8787",
                    help="wrangler dev base URL (DO-path synthesis).")
```

   The subparser already defines `--provider`, `--model`, `--out`, `--judge-provider`, `--judge-model`, `--limit` (confirm; keep them — `--provider`/`--model` are now unused by this branch but harmless; do not remove them in this task to keep the diff surgical). Map `args.judge_provider`/`args.judge_model` through unchanged.

- [x] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/stage0/tests/test_cli_synthesis_repointed.py -q
```
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/stage0/cli.py apps/evals/teacher_model/stage0/tests/test_cli_synthesis_repointed.py
git commit -m "refactor(evals): repoint stage0 cli synthesis to run_do_baseline (#22)"
```

---

### Task 7: Measure the honest baseline through the live DO

**Group:** D (manual, after Group C). This task requires a live `wrangler dev` + local DB and CANNOT be unit-tested; it is the measurement step, executed by the build agent or maintainer.

**Behavior being verified:** the full holdout runs through the real DO and produces `baseline_v2_do.jsonl` + `baseline_v2_do_aggregate.json` with an honest ASCF `mean_outcome`.

**Interface under test:** the `--do-path` runner end-to-end + `scripts/aggregate.py`.

**Files:**
- Create (generated): `apps/evals/results/baseline_v2_do.jsonl`
- Create (generated): `apps/evals/results/baseline_v2_do_aggregate.json`

- [ ] **Step 1: Boot the DO and smoke-test one recording**

```bash
# Terminal 1: real DO + local DB (see EVAL_CHECKLIST.md for wrangler auth)
just api

# Terminal 2: confirm reachability, then run one recording end-to-end
curl -sf http://localhost:8787/health | head
cd apps/evals && uv run python -m teaching_knowledge.run_eval --do-path --limit 1 \
    --teacher-model claude-sonnet-4-6 \
    --judge-model '@cf/google/gemma-4-26b-a4b-it' \
    --out results/smoke_do_$(date -u +%Y%m%d).jsonl
```
Expected: stdout shows `[1] <recording_id> | <piece_slug>` with NO `ERROR:`; the smoke JSONL row has non-empty `synthesis_text`, populated `judge_dimensions`, and `synthesis_latency_ms` > 500 (a real Anthropic call). If `ERROR:` appears, STOP — fix wrangler auth / boot per `EVAL_CHECKLIST.md`; do not proceed to the full run with a broken DO.

- [ ] **Step 2: Run the full holdout through the DO**

```bash
cd apps/evals && uv run python -m teaching_knowledge.run_eval --do-path \
    --teacher-model claude-sonnet-4-6 \
    --judge-model '@cf/google/gemma-4-26b-a4b-it' \
    --out results/baseline_v2_do.jsonl
```
Expected: 98 rows written (resume-safe; rerun to fill any error rows after fixing their cause). Error rate must be under 5% for a valid baseline (the dossier aggregator gate refuses above 5%).

- [ ] **Step 3: Aggregate**

> **Precondition (BLOCKER 3 — verify before aggregating).** `scripts/aggregate.py` calls `aggregate_run(jsonl, dataset_index_path)`, and `aggregate_run` calls `_load_index(dataset_index_path)` **unconditionally** (`aggregate.py:67`); `_load_index` does `path.read_text()`, so an absent index file hard-crashes the step. The `--dataset-index` flag defaults to the **CWD-relative** path `teaching_knowledge/data/dataset_index.jsonl`, so this command MUST be run from `apps/evals/` (as shown). Verified facts for this repo: that file exists (890 rows) and covers **all 98** holdout `recording_id`s with `composer_era` + `skill_bucket` tags, so the `by_era`/`by_skill` breakdowns will be populated (not silently blank). Run this guard first and abort if it does not print `missing: 0`:

```bash
cd apps/evals && test -f teaching_knowledge/data/dataset_index.jsonl || \
  { echo "ABORT: dataset_index.jsonl missing — aggregate will crash"; exit 1; }
cd apps/evals && python3 -c "
import json
h={json.loads(l)['recording_id'] for l in open('teacher_model/stage0/data/stage0_holdout.jsonl') if l.strip()}
i={json.loads(l)['recording_id'] for l in open('teaching_knowledge/data/dataset_index.jsonl') if l.strip()}
print('missing:', len(h - i))
assert not (h - i), 'holdout ids missing from dataset_index -> era/skill breakdowns would be silently empty'
"
```

Then aggregate (run from `apps/evals/` so the relative `--dataset-index` default resolves; the explicit flag is shown for clarity):

```bash
cd apps/evals && uv run python -m teaching_knowledge.scripts.aggregate \
    results/baseline_v2_do.jsonl \
    --dataset-index teaching_knowledge/data/dataset_index.jsonl \
    --out results/baseline_v2_do_aggregate.json
cat results/baseline_v2_do_aggregate.json
```
Expected: a JSON object with `dimensions: [{ "name", "mean_outcome", "n", ... }]`, `composite_mean`, and **non-empty** `by_era`/`by_skill` (the precondition guarantees coverage). Record the `Audible-Specific Corrective Feedback` `mean_outcome` and `n` — these are the new locked ASCF numbers (may be above or below 1.387; either is success).

- [ ] **Step 4: Commit the generated artifacts**

```bash
git add apps/evals/results/baseline_v2_do.jsonl apps/evals/results/baseline_v2_do_aggregate.json
git commit -m "feat(evals): honest DO-path synthesis baseline over holdout (#22)"
```

---

### Task 8: Re-lock `_SONNET_BASELINE` and update the checklist

**Group:** D (manual, after Task 7; depends on `baseline_v2_do_aggregate.json`)

**Behavior being verified:** the aggregator dossier test passes against the new DO-path baseline, and `EVAL_CHECKLIST.md` documents the DO-path runbook (the thin-framing entrypoint is removed).

**Interface under test:** `build_dossier` against the re-locked `_SONNET_BASELINE`.

**Files:**
- Modify: `apps/evals/teacher_model/stage0/tests/test_aggregator.py`
- Modify: `apps/evals/EVAL_CHECKLIST.md`

- [ ] **Step 1: Update the locked baseline, then run the existing dossier tests as the verification**

Edit `_SONNET_BASELINE` in `apps/evals/teacher_model/stage0/tests/test_aggregator.py` so its `dimensions` (name / mean_outcome / n) and `composite_mean` are copied verbatim from `apps/evals/results/baseline_v2_do_aggregate.json` produced in Task 7. The seven capability dimensions and their order are unchanged; only the numeric `mean_outcome`/`n`/`composite_mean` values change. Example of the single ASCF row to replace (substitute the real measured values from the aggregate):

```python
        {"name": "Audible-Specific Corrective Feedback", "mean_outcome": <measured>, "n": <measured_n>},
```

Add a one-line provenance comment above `_SONNET_BASELINE`:

```python
# Re-locked 2026-06-02 from results/baseline_v2_do_aggregate.json — measured through
# the real SessionBrain DO (buildSynthesisFraming), NOT the deleted thin Python framing (#22).
```

- [ ] **Step 2: Run the dossier test suite — verify it PASSES with the new baseline**

```bash
cd apps/evals && uv run pytest teacher_model/stage0/tests/test_aggregator.py -q
```
Expected: PASS. (These tests build a dossier that reads `_SONNET_BASELINE` as the baseline aggregate; they assert dossier shape/gates, which remain valid for any honest baseline numbers. If a test hard-codes the old 1.387, update that assertion to the measured value in the same commit.)

- [ ] **Step 3: Update `EVAL_CHECKLIST.md`**

In `apps/evals/EVAL_CHECKLIST.md`:
1. Replace the "Synthesis-quality eval (`run_eval.py`) — no wrangler dependency" section: `run_eval.py` now REQUIRES `wrangler dev` (it drives the real DO via `--do-path`). State the prereqs (Cloudflare auth, `just api`, `curl -sf http://localhost:8787/health`).
2. Replace the "Producing the locked baseline" commands with the Task 7 DO-path commands (`--do-path`, `--out results/baseline_v2_do.jsonl`, then `scripts/aggregate.py ... --out results/baseline_v2_do_aggregate.json`).
3. Remove any reference to `build_synthesis_user_msg`, `bar_analysis_local`, or `stage0/run_synthesis.py` as a synthesis source. Add one line: "The locked ASCF baseline is measured through the real DO framing (`buildSynthesisFraming`); the thin Python framing port was deleted in #22."

- [ ] **Step 4: Full eval-suite collection sanity + commit**

```bash
cd apps/evals && uv run pytest teaching_knowledge teacher_model/stage0 -q
git add apps/evals/teacher_model/stage0/tests/test_aggregator.py apps/evals/EVAL_CHECKLIST.md
git commit -m "test(evals): re-lock SONNET baseline to DO-path measurement + update checklist (#22)"
```
Expected: the eval suite collects and passes (no import errors from the deleted modules; deleted-module guard tests green; dossier test green against the new baseline).

---

### Task 9: Repoint `regen_calibration_baseline.py` onto the DO path

**Group:** C (parallel with Task 6, after Group B; depends on `build_do_row`/`run_do_baseline` existing)

**Behavior being verified (BLOCKER 1):** the calibration-baseline regenerator no longer imports or calls the deleted `build_synthesis_user_msg` / `extract_teacher_response`; it produces its calibration rows through the real DO framing (`run_recording` -> `SessionResult` -> `build_do_row`), keeping it functional and on the single source of truth instead of silently dead behind its `except ImportError -> sys.exit(1)` guard.

> **Why repoint, not delete or scope-out (one line):** this is the calibration-baseline regenerator that feeds rubric calibration (r>=0.7, the teacher-finetune critical path); repointing it onto the DO path is the only option consistent with this plan's stated goal of "a single source of truth remains" — deleting it would drop a needed tool, and leaving it broken violates the project's orphan rule.
>
> **Judge note:** the calibration regen currently judges with `judge_synthesis_v2` (7-dim), NOT the stage0 9-dim `judge_extended`. This is intentional and preserved — the calibration baseline is the v2-rubric calibration corpus, separate from the stage0 dossier. So unlike Task 6, this task keeps `judge_synthesis_v2`.
>
> **DO-replay note:** like the original (which made real Anthropic synthesis calls), the repointed regen requires a live `wrangler dev`; the orchestration is verified with an injected fake driver (no live DO needed for the unit test), mirroring Task 5.

**Interface under test:** `teacher_model.calibration.regen_calibration_baseline` module import + a `_regen_row(...)` helper extracted for testability.

**Files:**
- Modify: `apps/evals/teacher_model/calibration/regen_calibration_baseline.py`
- Test: `apps/evals/teacher_model/calibration/tests/test_regen_calibration_repointed.py`

- [x] **Step 1: Write the failing test**

```python
# apps/evals/teacher_model/calibration/tests/test_regen_calibration_repointed.py
from __future__ import annotations

from pathlib import Path

import teacher_model.calibration.regen_calibration_baseline as regen


def test_regen_no_longer_uses_thin_framing_symbols() -> None:
    text = Path(regen.__file__).read_text()
    assert "build_synthesis_user_msg" not in text
    assert "extract_teacher_response" not in text


def test_regen_builds_row_through_build_do_row() -> None:
    # The repointed regen routes synthesis through build_do_row (the DO path),
    # so the module must reference it.
    text = Path(regen.__file__).read_text()
    assert "build_do_row" in text
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/calibration/tests/test_regen_calibration_repointed.py -q
```
Expected: FAIL — the module still imports/uses `build_synthesis_user_msg` and `extract_teacher_response` and does not reference `build_do_row`.

- [x] **Step 3: Implement the minimum to make the test pass**

In `apps/evals/teacher_model/calibration/regen_calibration_baseline.py`:

1. In the `try: from teaching_knowledge.run_eval import (...)` block (lines ~62-71), remove `build_synthesis_user_msg` and `extract_teacher_response` from the import list; add `build_do_row`. The DO path owns the prompt and computes its own signals, so the following also become orphaned by this change and must be removed: the `SESSION_SYNTHESIS_SYSTEM` import, the `aggregate_muq` import (DO path needs no MuQ means), the `LLMClient` import + the local `synthesis_client = LLMClient(...)` construction, and — once `build_do_row` replaces it — the `_build_row` import. Keep `load_manifests` (still used to resolve `meta`). After editing, grep the file for each removed name to confirm no dangling reference remains.
2. Add a `--wrangler-url` arg (default `http://localhost:8787`) to the existing `argparse` parser, and a module-level default driver mirroring Task 5's `_default_driver` (or import `run_do_baseline`'s `_default_driver` from `run_eval` if exported). Replace the per-recording synthesis block (the `muq_means = aggregate_muq(...)` / `build_synthesis_user_msg(...)` / `synthesis_client.complete(...)` / `extract_teacher_response(...)` / judge / `_build_row(...)` sequence inside the `for rid in sorted(remaining):` loop) with:

```python
            data = json.loads(cache_path.read_text())
            chunks = data.get("chunks", [])
            if not chunks:
                print(f"  SKIP {rid}: empty chunks")
                continue

            recording_cache = {"recording_id": rid, "chunks": chunks}
            try:
                session_result = driver(
                    args.wrangler_url,
                    recording_cache,
                    "calibration-student-001",
                    meta.get("piece_slug") or None,
                )
                row = build_do_row(
                    session_result,
                    meta,
                    judge_synthesis_v2,
                    provenance,
                    dry_run=args.dry_run,
                )
            except Exception as exc:
                print(f"  ERROR {rid}: {exc}")
                continue

            fout.write(json.dumps(row) + "\n")
            fout.flush()
            processed += 1
            print(f"  [{processed}/{len(remaining)}] {rid} ({meta['title'][:30]})")
```

   `build_do_row`'s default `judge_provider="workers-ai"` / `judge_model="@cf/google/gemma-4-26b-a4b-it"` match the calibration regen's existing Gemma judge; pass explicit values only if the regen pinned a different judge model (confirm against the current `judge_synthesis_v2(...)` call — it used defaults, so no override needed).
3. The `driver` is the injected/default DO-replay driver (same shape as Task 5: `driver(wrangler_url, recording_cache, student_id, piece_query) -> SessionResult`, default calls `shared.pipeline_client.run_recording` via `asyncio.run`). Define it at module level so the unit test can monkeypatch it if needed; the two tests above assert source content only, so the default driver need not run.

- [x] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/calibration/tests/test_regen_calibration_repointed.py -q
```
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/calibration/regen_calibration_baseline.py \
        apps/evals/teacher_model/calibration/tests/test_regen_calibration_repointed.py
git commit -m "refactor(evals): repoint calibration baseline regen onto DO path (#22)"
```

---

## Open Questions

- Q: Should `cli.py`'s `synthesis` subcommand be repointed to the DO-path runner, or removed?
  Resolved (autopilot default): **repointed** (Task 6) so the stage0 dossier flow keeps producing `synthesis_runs.jsonl` on real framing with the smallest blast radius.
- Q (BLOCKER 2): Does repointing `cli.py synthesis` swap the judge from `judge_extended` (9-dim) to `judge_synthesis_v2` (7-dim)?
  Resolved: **no swap.** `judge_extended` is preserved and passed explicitly to `run_do_baseline`. The stage0 `build_dossier` reads 9-dim criteria (`Taste Defensibility`, `Adaptation Specificity`) that only `judge_extended` emits; swapping to the 7-dim `judge_synthesis_v2` would silently blank the Taste/Adaptation capability rows. `judge_extended(text, ctx, provider=, model=)` is call-compatible with `build_do_row`'s judge invocation.
- Q (BLOCKER 1): What happens to `regen_calibration_baseline.py`, the 4th consumer of `build_synthesis_user_msg`?
  Resolved: **repointed onto the DO path** (Task 9), keeping it functional and on the single source of truth. It keeps its own `judge_synthesis_v2` (the v2-rubric calibration corpus is separate from the stage0 9-dim dossier).
- Q (BLOCKER 3): Does Task 7's aggregate step need `--dataset-index`?
  Resolved: **yes.** `aggregate_run` loads the index unconditionally; the default is CWD-relative, so the command must run from `apps/evals/`. Task 7 Step 3 now adds an existence + holdout-coverage precondition (verified: file present, all 98 ids covered) so `by_era`/`by_skill` are populated, not silently blank.
- Q: The legacy synthesis WS payload uses `isFallback` (camelCase) but `pipeline_client.py` reads `is_fallback` (snake_case), so `is_fallback` always reads `False` on the legacy DO path.
  Resolved (autopilot default): **out of scope for #22.** Cold-start holdout runs always expect a real synthesis, so a false `is_fallback=False` does not corrupt the baseline. `build_do_row` keys off `synthesis is None or empty text`, not `is_fallback`, so the bug cannot mask a DO failure. Note left for a follow-up issue.
- Q: Does the live-DO baseline land above or below 1.387?
  Resolved: **unknown and not a gate.** Whatever Task 7 measures is locked verbatim in Task 8.
- Q: Task 7/8 require a live `wrangler dev`; in a headless build environment they may not be runnable.
  Resolved (autopilot default): Groups A–C (the code change + deletions) are fully unit-tested and land independently. If the build environment lacks a live DO, Tasks 7–8 are deferred to a maintainer with the exact commands provided here; the spec's success criterion (honest measurement + drift deleted) is met by Groups A–C deleting the drift and Task 7 producing the number. The plan flags Tasks 7–8 as the human-in-the-loop measurement step.
```

---

## Challenge Review

Reviewed against actual source: `run_eval.py` (`build_synthesis_user_msg` 119-213, `run` 293-495, `main` 498-571, `load_completed_ids` 224-240, imports 19-55), `shared/pipeline_client.py` (`run_recording`, `SessionResult`), `teacher_model/stage0/cli.py` (synthesis dispatch), `scripts/aggregate.py` (`aggregate_run`), `test_aggregator.py` (`_SONNET_BASELINE`), the holdout JSONL + a briefing inference-cache file, and a full cross-repo grep for every soon-deleted symbol.

### CEO Pass

**Premise — sound.** The drift is real and verified. `build_synthesis_user_msg` injects a `bar_analysis` block (run_eval.py:150-165) and assembles `top_moments` from global `SCALER_MEAN` deviations, neither of which matches the production DO path. Re-measuring through `run_recording` (the already-proven DO-replay driver) and deleting the duplicate is the direct path. Approach B (gut-and-repoint) correctly refuses to keep two prompt implementations in sync.

**Scope — one undocumented consumer breaks the "single source of truth" claim.** See BLOCKER 1. Otherwise scope is tight (8 modify/delete targets + generated artifacts), the hardest problem (live-DO measurement) is not avoided — it is correctly isolated behind a fixture-testable `build_do_row` with the orchestration shell `run_do_baseline` smoke-tested manually.

**12-Month alignment — net positive.** Deleting three framing duplicates collapses a maintenance liability and makes the locked baseline an honest production measurement, which the teacher-finetune critical path (rubric calibration r>=0.7) depends on. The one debt it introduces: the baseline `n` drops from ~504-513 (train split) to <=98 (holdout) — see RISK 1.

**Alternatives — documented.** The spec's "Why not patch build_synthesis_user_msg" section records the rejected alternative. Adequate.

### Engineering Pass

**Architecture — matches the code.** Data flow verified: holdout row -> `briefing_path` JSON -> `recording_cache{recording_id,chunks}` -> `run_recording(?eval=true -> set_piece -> eval_chunk xN -> end_session -> synthesis)` -> `SessionResult.synthesis.text` -> `judge_fn` -> aggregator row -> `aggregate_run` -> re-locked `_SONNET_BASELINE`. `SessionResult` exposes `recording_id`, `synthesis`, `errors`, `synthesis_latency_ms`, `piece_identification` exactly as the plan's "Context" block states (pipeline_client.py:52-66). No `apps/api/` production code is touched. Security: no new untrusted-input-to-sink path; holdout paths are maintainer-controlled local files.

**Module depth.** `build_do_row` — interface 4 positional + 4 kw args, hides error-mapping/judge-flattening/provenance/piece_resolved/empty-guard. DEEP. `run_do_baseline` — interface 4 positional + 6 kw, hides iteration/cache-load/resume/asyncio.run orchestration. DEEP, with an injected `driver`/`judge_fn` seam that is the right testability boundary. No shallow modules.

**Test philosophy — clean.** `test_do_row.py` and `test_run_do_baseline.py` exercise the public functions with fake `SessionResult`/judge/driver dataclasses, asserting user-observable row fields (`error`, `judge_dimensions`, `piece_resolved`, resume behavior), not internal state. Driver/judge are injected external boundaries — legitimate to fake. The deletion-guard tests (Tasks 1-3) assert filesystem/importability — appropriate for "this thing is gone" behavior. No internal-collaborator mocking, no shape-only tests.

**Vertical-slice audit — passes.** Each task is one test -> one impl -> one commit. Task 5 writes two tests (writes-one-row + resumes-completed) in one commit; both exercise the single new `run_do_baseline` behavior and its resume branch — acceptable as one slice, not horizontal scaffolding. No task writes tests for a later task to implement.

**Failure modes.** `run_do_baseline` wraps each recording in try/except, writes an `error` row, flushes per-row, and is resume-safe via `load_completed_ids` (which — verified, run_eval.py:236 — only counts a row complete when `recording_id` AND `synthesis_text` are present AND `error` is falsy, so error rows are correctly reprocessed on rerun). No silent failures: every failure path produces a visible `error` row + stdout `ERROR:` line. Task 7 gate (error rate <5%) is enforced by the dossier aggregator. Good.

#### Findings

[BLOCKER] (confidence: 10/10) — **`teacher_model/calibration/regen_calibration_baseline.py` is a fourth, undocumented consumer of the deleted symbols and is absent from the File Structure table.** It imports `build_synthesis_user_msg` and `extract_teacher_response` from `run_eval` (lines 64-67) and calls `build_synthesis_user_msg(...)` at line 119. Task 1 deletes `build_synthesis_user_msg`; after that, running this script hits its `except ImportError -> sys.exit(1)` guard and becomes permanently dead — silently. No test imports it (confirmed: not referenced under `teacher_model/calibration/tests/`), so the suite stays green and the breakage is invisible until someone runs the calibration regen. The plan's stated goal is "a single source of truth remains" — this leaves a broken duplicate. The plan must decide and document: delete `regen_calibration_baseline.py`, repoint it onto `run_do_baseline`/`build_do_row`, or explicitly scope it out with a note that it is knowingly left broken. Add it to the File Structure table and to the relevant task either way. (Per project rule: orphans created by your change must be handled, not left silently broken.)

[BLOCKER] (confidence: 9/10) — **Task 6 misstates `cli.py`'s import structure, and its Step-3 edit instructions will not apply cleanly.** The plan says "Delete the top-level import `from teacher_model.stage0.run_synthesis import run as run_synthesis`" and "the `synthesis` dispatch ... imports `run_synthesis`" — but verified: cli.py imports it at module top (line 11, `from teacher_model.stage0.run_synthesis import run as run_synthesis`) AND the dispatch branch (lines 133-147) does NOT lazily import it; it calls `run_synthesis(...)` with a different signature (`teacher_client=`, `judge_fn=judge_extended`, `system_prompt=`, NOT `judge_fn=judge_synthesis_v2`). Two concrete consequences: (a) Task 3 deletes `run_synthesis.py`, so the module-top import at cli.py:11 breaks import/collection of cli.py the moment Task 3 lands and before Task 6 runs — any test that imports cli between Group A and Group C fails. Group ordering (A before C) does not save intermediate states; the `test_cli_synthesis_repointed.py` Step-2 "verify it FAILS" expects an import error, which is fine, but the plan must explicitly state that cli.py:11 (the module-top import) is the line removed, not a nonexistent in-branch import. (b) Task 6's replacement maps `args.judge_provider`/`args.judge_model` into `run_do_baseline`, but the current branch uses `judge_extended` (stage0's own judge), not `judge_synthesis_v2`. Switching the stage0 `synthesis` subcommand from `judge_extended` to `judge_synthesis_v2` silently changes what the stage0 dossier measures. The plan must either confirm `judge_synthesis_v2` is the intended judge for the repointed subcommand (and note the judge swap explicitly) or preserve `judge_extended`. Rewrite Task 6 Step 3 against the actual file: remove cli.py:11, replace the lines 133-147 branch body, and resolve the judge-swap question.

[BLOCKER] (confidence: 8/10) — **Task 7 Step 3's aggregate command will silently drop the per-era / per-skill breakdowns and may mis-key on an absent index, because `aggregate_run(jsonl, dataset_index_path)` requires a dataset index the command omits.** Verified: `scripts/aggregate.py:main` defaults `--dataset-index` to `teaching_knowledge/data/dataset_index.jsonl`; `aggregate_run` calls `_load_index(dataset_index_path)` unconditionally (aggregate.py:67) and uses `tags = index.get(rec_id)` for `by_era`/`by_skill`. If that default index file does not exist, `_load_index` raises on `path.read_text()` and the aggregate step crashes; if it exists but lacks the 98 holdout DO recording_ids, the era/skill breakdowns are silently empty (the top-line `mean_outcome`/`composite_mean` the plan locks are still correct, since tagging is guarded by `if tags:`). The plan's Task 7 command (`aggregate.py results/baseline_v2_do.jsonl --out ...`) neither passes `--dataset-index` nor states the default-file precondition. Add an explicit check that `teaching_knowledge/data/dataset_index.jsonl` exists and covers the holdout ids (or pass the correct index path / document that breakdowns are intentionally empty for the holdout baseline). Without this the manual measurement step can hard-crash or produce a baseline whose tag breakdowns are silently blank, undetected.

[RISK] (confidence: 9/10) — **The re-locked baseline `n` collapses from ~504-513 to <=98, ~5x less statistical power, and the plan does not flag it.** The old `_SONNET_BASELINE` was measured on the train split (n per dim 504-513); the DO-path runner drives only the 98-row holdout. Task 8 copies `n` verbatim, so this is correct mechanically, but the locked ASCF CI will widen materially and any downstream "did ASCF move" comparison against the old number is comparing different populations (train vs holdout) AND different framings simultaneously — two confounded variables. The spec explicitly says success is "honest measurement + drift deleted, NOT ASCF went up," which contains the damage, but the build/ship docs and `EVAL_CHECKLIST.md` update (Task 8 Step 3) must state plainly that the new baseline is holdout-only (n<=98) and is NOT numerically comparable to the old train-split 1.387. Fallback: if power matters, a follow-up can run the DO path over the train split too — but that is out of scope here.

[RISK] (confidence: 7/10) — **The `judge_provider` autodetect duplicated into `main()` (Task 5 item 2) and `cli.py` (Task 6) re-implements run_eval.py:line-in-`run` logic a third time.** The string `"openrouter" if "/" in judge_model and not judge_model.startswith("@cf/") else "workers-ai"` now appears in the legacy `run()`, in the new `main()` `--do-path` branch, and the cli passes `args.judge_provider` separately. Three copies of one rule is the DRY smell the issue is otherwise trying to kill. Not a blocker (each is one line), but extract a `_judge_provider_for(model)` helper while you are in the file. Watch during execution: if Task 5 and Task 6 diverge on this rule, the stage0 dossier and the `run_eval --do-path` path could pick different judge providers for the same model.

[OBS] — Task 5's `_default_driver` calls `asyncio.run(run_recording(...))` once per recording. `run_recording` caches an auth session via module-global `_auth_session` (pipeline_client.py:89), so per-recording `asyncio.run` is fine (no cross-loop reuse of an open ws). No action.

[OBS] — The `isFallback`/`is_fallback` camel/snake mismatch (Open Question 2) is correctly scoped out: `build_do_row` keys the failure decision off `synthesis is None or not synthesis.text` (Task 4 impl), never `is_fallback`, so a wrong `is_fallback=False` cannot mask a DO failure on the cold-start holdout. Verified against pipeline_client.py:236 (`is_fallback=response.get("is_fallback", False)`).

[OBS] — `test_aggregator.py` tests use `_SONNET_BASELINE` only as a fixture written to a temp file and fed to `build_dossier`; they assert dossier shape/gates (7 rows, error-rate gate, inconsistency flag), not the literal 1.387. Task 8's "copy verbatim, shape tests stay green" claim is verified correct — provided the DO aggregate yields all seven criterion names in the existing order (judge-dependent; acceptable for a manual step).

### Presumption Inventory

| Assumption | Verdict | Reason |
|---|---|---|
| Only 3 thin-framing duplicates consume `build_synthesis_user_msg` | RISKY | `regen_calibration_baseline.py` is a 4th, undocumented (BLOCKER 1). |
| cli.py imports `run_synthesis` in the synthesis branch | RISKY | It's a module-top import (line 11) with a different signature + `judge_extended` (BLOCKER 2). |
| `aggregate.py jsonl --out` needs no index | RISKY | Requires `--dataset-index` (default file); crashes if absent, blanks breakdowns if uncovered (BLOCKER 3). |
| `load_completed_ids` makes the runner resume-safe | SAFE | Verified run_eval.py:236 — needs recording_id+synthesis_text, error-free. |
| `SessionResult` exposes the fields the plan lists | SAFE | Verified pipeline_client.py:52-66. |
| `run_recording` passes `piece_slug` as `set_piece` query | SAFE | Verified pipeline_client.py:183-184. |
| Re-locking `_SONNET_BASELINE` keeps test_aggregator green | SAFE | Tests assert shape/gates, not 1.387 (verified). |
| Briefing inference-cache has `recording_id` + `chunks[].predictions` | SAFE | Verified on `-wyCOEoOBC4.json` (6 dims, midi_notes present). |
| `_build_row`/`aggregate_muq`/`load_manifests` must be kept | SAFE | Still imported by tag_dataset, sampler, provenance test, calibration. |

### Summary

[BLOCKER] count: 3
[RISK]    count: 2
[QUESTION] count: 0

VERDICT: NEEDS_REWORK — (1) `regen_calibration_baseline.py` is an undocumented 4th consumer of `build_synthesis_user_msg`/`extract_teacher_response` that Task 1 silently breaks; add it to the File Structure table and decide delete/repoint/scope-out. (2) Task 6 misdescribes cli.py's import structure (the `run_synthesis` import is at module top, line 11, not in the dispatch branch) and silently swaps the stage0 judge from `judge_extended` to `judge_synthesis_v2`; rewrite Task 6 against the actual file and resolve the judge swap. (3) Task 7 Step 3's `aggregate.py` command omits the required `--dataset-index`, which can hard-crash the manual measurement or silently blank the era/skill breakdowns; add the index precondition/flag.
