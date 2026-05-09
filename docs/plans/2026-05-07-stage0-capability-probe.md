# Stage 0 Capability Probe Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task). Tasks within a group touch non-overlapping files and have no inter-task dependencies. Do NOT start execution until `/challenge` returns `VERDICT: PROCEED`.

**Goal:** Produce a defensible capability dossier that scores base Qwen3.6-35B-A3B at-ceiling / mid-tier / absent across the seven teacher capabilities (judgment, taste, integration, voice, vocabulary, tool-calling, adaptation), within a 1-2 day budget, with results that set training dosage for Stages 2-3 and decide whether Stage 5 is needed.

**Spec:** `docs/specs/2026-05-07-stage0-capability-probe-design.md`

**Style:** Follow `CLAUDE.md` (project) and `apps/CLAUDE.md`. Python: explicit exceptions over silent fallbacks; no emojis; `uv` for deps. Test runner: `pytest`. All Stage 0 code lives under `apps/evals/teacher_model/stage0/` and imports relative to `apps/evals/` (matches `teaching_knowledge/run_eval.py` convention).

**Working directory for all `uv run` and `pytest` commands:** `apps/evals/`

---

## Task Groups

```
Group A (parallel, no inter-task deps):
  Task 1  — domain_knowledge_probe.py: add openrouter to --provider choices
  Task 2  — tier_classifier.py
  Task 3  — pin_tokenizer.py
  Task 4  — continuation_probe.py
  Task 5  — tool_scorer.py
  Task 6  — judge_extended.py + prompts/judge_v2_extended.txt
  Task 7  — sampler.py
  Task 8  — tool_probe_cases.jsonl loader + the 40-case file
  Task 9  — pyproject.toml: add stage0 optional-dependencies group

Group B (depends on Group A):
  Task 10 — aggregator.py             (uses tier_classifier from Task 2)

Group C (depends on Group A; tasks within group are parallel):
  Task 11 — run_synthesis.py          (uses judge_extended from Task 6)
  Task 12 — run_tool_probe.py         (uses tool_scorer from Task 5, cases from Task 8)
  Task 13 — run_continuation.py       (uses continuation_probe from Task 4 + tool_scorer from Task 5)

Group D (depends on Groups A, B, C):
  Task 14 — cli.py                    (wires sample / synthesis / tool / continuation / mcq / pin-tokenizer / aggregate subcommands)
```

A subagent for each task may only read the files referenced in its own task. Re-state any shared code inside the task body.

---

## Task 1: Add `openrouter` provider to domain_knowledge_probe.py

**Group:** A

**Behavior being verified:** the existing MCQ probe CLI accepts `--provider openrouter` without erroring at argparse time.

**Interface under test:** `domain_knowledge_probe.main()` argparse parser via `--help`-style construction.

**Files:**
- Modify: `apps/evals/teacher_model/domain_knowledge_probe.py`
- Test: `apps/evals/teacher_model/tests/test_domain_probe_cli.py` (new file)

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/teacher_model/tests/test_domain_probe_cli.py
"""Verify domain_knowledge_probe CLI accepts the openrouter provider."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def test_cli_accepts_openrouter_provider() -> None:
    """`--provider openrouter` must parse without argparse rejecting it."""
    repo_root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; sys.path.insert(0, '.'); "
                "from teacher_model import domain_knowledge_probe as m; "
                "p = argparse.ArgumentParser(); "
            ),
        ],
        check=False,
        capture_output=True,
    )
    # Real test: invoke argparse directly via the module's parser.
    # We re-create it the same way main() does and assert openrouter is allowed.
    sys.path.insert(0, str(repo_root / "apps" / "evals"))
    from teacher_model.domain_knowledge_probe import main  # noqa: F401

    # Build a parser identical to main()'s, parse with --provider openrouter.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider",
        choices=["workers-ai", "anthropic", "openrouter"],
        default="workers-ai",
    )
    args = parser.parse_args(["--provider", "openrouter"])
    assert args.provider == "openrouter"

    # Now actually verify the source file lists openrouter as a choice.
    src = (repo_root / "apps" / "evals" / "teacher_model" / "domain_knowledge_probe.py").read_text()
    assert '"openrouter"' in src, "openrouter must be in --provider choices"
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/tests/test_domain_probe_cli.py -x -q
```
Expected: FAIL — `AssertionError: openrouter must be in --provider choices` (the source string check fails because the file only lists `workers-ai` and `anthropic`).

- [ ] **Step 3: Implement the minimum to make the test pass**

Edit `apps/evals/teacher_model/domain_knowledge_probe.py`. In the existing argparse block (around line 240), change:

```python
    parser.add_argument(
        "--provider",
        choices=["workers-ai", "anthropic"],
        default="workers-ai",
        help="LLM provider to use (default: workers-ai)",
    )
```

to:

```python
    parser.add_argument(
        "--provider",
        choices=["workers-ai", "anthropic", "openrouter"],
        default="workers-ai",
        help="LLM provider to use (default: workers-ai)",
    )
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/tests/test_domain_probe_cli.py -x -q
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/domain_knowledge_probe.py apps/evals/teacher_model/tests/test_domain_probe_cli.py && git commit -m "feat(stage0): allow openrouter provider in domain_knowledge_probe CLI"
```

---

## Task 2: tier_classifier — Sonnet-anchored + absolute tier classification

**Group:** A

**Behavior being verified:** given a value, optional baseline, mode, and optional CI, returns the correct tier label, including compound labels when CI straddles a boundary.

**Interface under test:** `classify_tier(value, baseline, mode, ci) -> Tier`.

**Files:**
- Create: `apps/evals/teacher_model/stage0/__init__.py`
- Create: `apps/evals/teacher_model/stage0/tier_classifier.py`
- Create: `apps/evals/teacher_model/stage0/tests/__init__.py`
- Test: `apps/evals/teacher_model/stage0/tests/test_tier_classifier.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/teacher_model/stage0/tests/test_tier_classifier.py
"""Boundary tests for tier classification."""
from __future__ import annotations

import pytest

from teacher_model.stage0.tier_classifier import classify_tier


def test_relative_at_ceiling_within_quarter_of_baseline() -> None:
    assert classify_tier(value=2.80, baseline=2.84, mode="relative") == "at_ceiling"


def test_relative_mid_tier_quarter_to_three_quarters_below() -> None:
    assert classify_tier(value=2.30, baseline=2.84, mode="relative") == "mid_tier"


def test_relative_absent_more_than_three_quarters_below() -> None:
    assert classify_tier(value=1.50, baseline=2.84, mode="relative") == "absent"


def test_absolute_thresholds() -> None:
    assert classify_tier(value=2.50, baseline=None, mode="absolute") == "at_ceiling"
    assert classify_tier(value=2.00, baseline=None, mode="absolute") == "mid_tier"
    assert classify_tier(value=1.00, baseline=None, mode="absolute") == "absent"


def test_ci_straddling_ceiling_boundary_returns_compound_label() -> None:
    # value = 2.55, baseline = 2.84, point estimate is mid_tier (delta -0.29).
    # CI (2.45, 2.65) straddles the at_ceiling boundary at baseline-0.25 = 2.59.
    label = classify_tier(value=2.55, baseline=2.84, mode="relative", ci=(2.45, 2.65))
    assert label == "mid_tier_with_ceiling_overlap"


def test_ci_straddling_absent_boundary_returns_compound_label() -> None:
    # baseline = 2.84, absent boundary = 2.84 - 0.75 = 2.09. value = 2.15, CI overlaps 2.09.
    label = classify_tier(value=2.15, baseline=2.84, mode="relative", ci=(2.00, 2.30))
    assert label == "mid_tier_with_absent_overlap"


def test_relative_mode_requires_baseline() -> None:
    with pytest.raises(ValueError, match="baseline is required"):
        classify_tier(value=2.5, baseline=None, mode="relative")


def test_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="mode must be"):
        classify_tier(value=2.5, baseline=2.0, mode="weird")  # type: ignore[arg-type]
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/stage0/tests/test_tier_classifier.py -x -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'teacher_model.stage0.tier_classifier'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `apps/evals/teacher_model/stage0/__init__.py` with the single line:

```python
"""Stage 0 capability probe for the teacher-model finetune curriculum."""
```

Create `apps/evals/teacher_model/stage0/tests/__init__.py` as an empty file (touch).

Create `apps/evals/teacher_model/stage0/tier_classifier.py`:

```python
"""Sonnet-anchored + absolute tier classification.

Used by the Stage 0 dossier aggregator to label each capability as
at-ceiling / mid-tier / absent, with compound labels when a 95% CI
straddles a tier boundary.
"""
from __future__ import annotations

from typing import Literal

Tier = str  # "at_ceiling" | "mid_tier" | "absent" | compound variants
Mode = Literal["relative", "absolute"]

_RELATIVE_CEILING_DELTA = 0.25  # within this much of baseline => at_ceiling
_RELATIVE_ABSENT_DELTA = 0.75   # more than this below baseline => absent

_ABSOLUTE_CEILING = 2.5
_ABSOLUTE_ABSENT = 1.5


def _point_tier_relative(value: float, baseline: float) -> Tier:
    delta = baseline - value
    if delta <= _RELATIVE_CEILING_DELTA:
        return "at_ceiling"
    if delta <= _RELATIVE_ABSENT_DELTA:
        return "mid_tier"
    return "absent"


def _point_tier_absolute(value: float) -> Tier:
    if value >= _ABSOLUTE_CEILING:
        return "at_ceiling"
    if value >= _ABSOLUTE_ABSENT:
        return "mid_tier"
    return "absent"


def classify_tier(
    value: float,
    baseline: float | None,
    mode: Mode,
    ci: tuple[float, float] | None = None,
) -> Tier:
    """Classify a measurement into a tier label.

    Args:
        value: the point estimate (e.g. mean dim score, discipline %)
        baseline: Sonnet baseline value (required when mode='relative')
        mode: 'relative' (Sonnet-anchored) or 'absolute' (fixed thresholds)
        ci: optional (low, high) 95% CI; if it straddles a tier boundary,
            the returned label is compound (e.g. 'mid_tier_with_ceiling_overlap')

    Returns the tier label string.
    """
    if mode not in ("relative", "absolute"):
        raise ValueError(f"mode must be 'relative' or 'absolute', got {mode!r}")
    if mode == "relative" and baseline is None:
        raise ValueError("baseline is required when mode='relative'")

    if mode == "relative":
        assert baseline is not None  # narrowed by the check above
        point = _point_tier_relative(value, baseline)
        if ci is None:
            return point
        ceiling_boundary = baseline - _RELATIVE_CEILING_DELTA
        absent_boundary = baseline - _RELATIVE_ABSENT_DELTA
    else:
        point = _point_tier_absolute(value)
        if ci is None:
            return point
        ceiling_boundary = _ABSOLUTE_CEILING
        absent_boundary = _ABSOLUTE_ABSENT

    low, high = ci
    crosses_ceiling = low < ceiling_boundary < high
    crosses_absent = low < absent_boundary < high

    if point == "mid_tier" and crosses_ceiling:
        return "mid_tier_with_ceiling_overlap"
    if point == "mid_tier" and crosses_absent:
        return "mid_tier_with_absent_overlap"
    if point == "at_ceiling" and crosses_ceiling:
        return "at_ceiling_with_mid_overlap"
    if point == "absent" and crosses_absent:
        return "absent_with_mid_overlap"
    return point
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/stage0/tests/test_tier_classifier.py -x -q
```
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/stage0/__init__.py apps/evals/teacher_model/stage0/tier_classifier.py apps/evals/teacher_model/stage0/tests/__init__.py apps/evals/teacher_model/stage0/tests/test_tier_classifier.py && git commit -m "feat(stage0): tier_classifier with Sonnet-anchored + absolute modes and CI overlap"
```

---

## Task 3: pin_tokenizer — download + hash the Qwen tokenizer

**Group:** A

**Behavior being verified:** invoking `pin_tokenizer` with a public model id downloads the tokenizer files, computes a stable hash, writes a JSON pin record, and raises `MissingChatTemplateError` if no chat template is present.

**Interface under test:** `pin_tokenizer(model_id, out_path) -> TokenizerPin`.

**Files:**
- Create: `apps/evals/teacher_model/stage0/pin_tokenizer.py`
- Test: `apps/evals/teacher_model/stage0/tests/test_pin_tokenizer.py`

**Dependency note:** this task uses `transformers`. It is added to `pyproject.toml` by Task 9 as part of a `[teacher-model-stage0]` extra. The test below uses the lightweight `gpt2` tokenizer so the build agent can run it without pulling 35B weights. To install: `cd apps/evals && uv sync --extra teacher-model-stage0` after Task 9 commits.

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/teacher_model/stage0/tests/test_pin_tokenizer.py
"""Hash stability + mutation sensitivity + chat-template-required behavior."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from teacher_model.stage0.pin_tokenizer import (
    MissingChatTemplateError,
    pin_tokenizer,
)


def test_pin_is_stable_across_two_invocations(tmp_path: Path) -> None:
    """Pinning the same model id twice produces the same hash."""
    out1 = tmp_path / "pin1.json"
    out2 = tmp_path / "pin2.json"
    # gpt2 has a chat_template baked in via tokenizer_config in modern transformers;
    # if not present, the test below will surface MissingChatTemplateError instead.
    # We use a model id that ships a chat template: Qwen/Qwen2.5-0.5B-Instruct.
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    pin_a = pin_tokenizer(model_id=model_id, out_path=out1)
    pin_b = pin_tokenizer(model_id=model_id, out_path=out2)
    assert pin_a.sha256 == pin_b.sha256
    assert json.loads(out1.read_text())["sha256"] == pin_a.sha256


def test_pin_changes_when_a_source_file_is_mutated(tmp_path: Path) -> None:
    """Mutating one of the pinned files changes the hash."""
    out = tmp_path / "pin.json"
    pin_a = pin_tokenizer(model_id="Qwen/Qwen2.5-0.5B-Instruct", out_path=out)
    # Mutate the cached tokenizer_config.json file in the cache dir
    cache_dir = Path(pin_a.cache_dir)
    targets = list(cache_dir.rglob("tokenizer_config.json"))
    assert targets, "expected tokenizer_config.json in cache"
    target = targets[0]
    original = target.read_text()
    target.write_text(original + "\n")  # one-byte mutation
    try:
        pin_b = pin_tokenizer(
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            out_path=tmp_path / "pin2.json",
            cache_dir=cache_dir,
            force_local=True,
        )
        assert pin_a.sha256 != pin_b.sha256
    finally:
        target.write_text(original)


def test_missing_chat_template_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A model with no chat_template must raise MissingChatTemplateError."""
    # Use gpt2 which has no chat template by default in older snapshots.
    # If gpt2 ever ships a chat template, swap to another template-less model id.
    with pytest.raises(MissingChatTemplateError):
        pin_tokenizer(model_id="gpt2", out_path=tmp_path / "pin.json")
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/stage0/tests/test_pin_tokenizer.py -x -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'teacher_model.stage0.pin_tokenizer'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `apps/evals/teacher_model/stage0/pin_tokenizer.py`:

```python
"""Pin the tokenizer of a Hugging Face model by hashing its source files.

Stage 0 commits the resulting tokenizer_pin.json so Stage 1 can verify the
exact same tokenizer + chat template is in use during finetuning.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

from transformers import AutoTokenizer

# Files that participate in the pin. Anything else (model weights, optimizer state)
# is irrelevant to tokenization equivalence.
_PIN_FILE_PATTERNS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "added_tokens.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)


class MissingChatTemplateError(RuntimeError):
    """Raised when a model has no chat template; Stage 1 cannot proceed."""


@dataclass
class TokenizerPin:
    model_id: str
    sha256: str
    files: list[str]            # sorted relative paths included in the hash
    cache_dir: str              # absolute path to the local cache dir
    chat_template_present: bool


def _collect_pin_files(cache_dir: Path) -> list[Path]:
    """Find all files in cache_dir matching _PIN_FILE_PATTERNS, sorted by relpath."""
    matches: list[Path] = []
    for pattern in _PIN_FILE_PATTERNS:
        matches.extend(cache_dir.rglob(pattern))
    # Deduplicate (a single file matches at most one pattern, but rglob may revisit symlinks)
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in matches:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        unique.append(p)
    unique.sort(key=lambda p: str(p.relative_to(cache_dir)))
    return unique


def _hash_files(files: Iterable[Path], cache_dir: Path) -> str:
    h = hashlib.sha256()
    for f in files:
        rel = str(f.relative_to(cache_dir)).encode("utf-8")
        h.update(b"PATH:")
        h.update(rel)
        h.update(b"\nDATA:")
        h.update(f.read_bytes())
        h.update(b"\n---\n")
    return h.hexdigest()


def pin_tokenizer(
    model_id: str,
    out_path: Path,
    cache_dir: Path | None = None,
    force_local: bool = False,
) -> TokenizerPin:
    """Download the tokenizer, hash its source files, write a pin record.

    Args:
        model_id: HuggingFace model id, e.g. 'Qwen/Qwen3.6-35B-A3B-Instruct'.
        out_path: where to write the JSON pin record.
        cache_dir: optional override of the HF cache directory (used in tests).
        force_local: if True, do not re-download; require files already in cache_dir.

    Raises:
        MissingChatTemplateError: if the loaded tokenizer has no chat_template.
    """
    kwargs: dict = {}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    if force_local:
        kwargs["local_files_only"] = True

    tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)

    chat_template = getattr(tokenizer, "chat_template", None)
    if not chat_template:
        raise MissingChatTemplateError(
            f"Model {model_id!r} has no chat_template. "
            "Stage 1 requires a chat-template-native tool-call slot."
        )

    # Resolve the actual cache dir transformers used.
    # For from_pretrained, the files land under cache_dir/models--{org}--{name}/snapshots/<sha>/
    if cache_dir is None:
        from huggingface_hub import constants as hf_const

        resolved_cache = Path(hf_const.HF_HUB_CACHE)
    else:
        resolved_cache = Path(cache_dir)

    # Find the snapshot directory containing the loaded tokenizer files.
    snapshots = list((resolved_cache).rglob(f"models--{model_id.replace('/', '--')}/snapshots"))
    if not snapshots:
        raise RuntimeError(f"Could not locate cache snapshot for {model_id!r} under {resolved_cache}")
    snapshot_root = snapshots[0]
    snapshot_dirs = sorted(p for p in snapshot_root.iterdir() if p.is_dir())
    if not snapshot_dirs:
        raise RuntimeError(f"No snapshots present under {snapshot_root}")
    pin_dir = snapshot_dirs[-1]  # most recent

    files = _collect_pin_files(pin_dir)
    if not files:
        raise RuntimeError(
            f"No tokenizer source files found under {pin_dir}. "
            "Pinning requires at least one of: " + ", ".join(_PIN_FILE_PATTERNS)
        )

    sha = _hash_files(files, pin_dir)
    pin = TokenizerPin(
        model_id=model_id,
        sha256=sha,
        files=[str(f.relative_to(pin_dir)) for f in files],
        cache_dir=str(pin_dir),
        chat_template_present=True,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(pin), indent=2))
    return pin
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv sync --extra teacher-model-stage0 && uv run pytest teacher_model/stage0/tests/test_pin_tokenizer.py -x -q
```
Expected: PASS (3 tests). Note: requires network for the first run to download tokenizers; subsequent runs read from the HF cache.

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/stage0/pin_tokenizer.py apps/evals/teacher_model/stage0/tests/test_pin_tokenizer.py && git commit -m "feat(stage0): pin_tokenizer hashes HF tokenizer source files; rejects missing chat template"
```

---

## Task 4: continuation_probe — degeneracy classifier for post-tool-result follow-ups

**Group:** A

**Behavior being verified:** given an initial assistant turn (with tool call), a synthetic tool_result, and a follow-up assistant response, classify the continuation as one of {clean, refusal, repetition, format_collapse, empty} and load the canned tool_result fixture for a given tool name.

**Interface under test:** `score_continuation(initial_assistant, tool_result, follow_up_response) -> ContinuationResult`; `load_tool_result_fixture(tool_name) -> dict`.

**Files:**
- Create: `apps/evals/teacher_model/stage0/continuation_probe.py`
- Create: `apps/evals/teacher_model/stage0/data/continuation_fixtures.json`
- Test: `apps/evals/teacher_model/stage0/tests/test_continuation_probe.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/teacher_model/stage0/tests/test_continuation_probe.py
"""Degeneracy classification fixtures for the post-tool-result continuation probe."""
from __future__ import annotations

import pytest

from teacher_model.stage0.continuation_probe import (
    ContinuationResult,
    load_tool_result_fixture,
    score_continuation,
)

_INITIAL = (
    "Let me look that up for you."
    "<tool_call>{\"name\": \"search_catalog\", \"arguments\": {\"query\": \"Chopin Ballade 1\"}}</tool_call>"
)
_RESULT = {"matches": [{"pieceId": "chopin.ballades.1"}]}


def test_clean_continuation_classified_clean() -> None:
    follow_up = (
        "Great — I found Chopin's Ballade No. 1. Try the second theme around bar 68 "
        "for that singing tone you're going for."
    )
    out = score_continuation(_INITIAL, _RESULT, follow_up)
    assert isinstance(out, ContinuationResult)
    assert out.category == "clean"
    assert out.is_degenerate is False


def test_refusal_classified() -> None:
    follow_up = "I cannot continue. I am unable to help with this request."
    out = score_continuation(_INITIAL, _RESULT, follow_up)
    assert out.category == "refusal"
    assert out.is_degenerate is True


def test_repetition_classified_when_same_tool_call_re_emitted() -> None:
    follow_up = (
        "<tool_call>{\"name\": \"search_catalog\", \"arguments\": {\"query\": \"Chopin Ballade 1\"}}</tool_call>"
    )
    out = score_continuation(_INITIAL, _RESULT, follow_up)
    assert out.category == "repetition"
    assert out.is_degenerate is True


def test_format_collapse_classified_when_output_is_raw_json_dump() -> None:
    follow_up = '{"matches":[{"pieceId":"chopin.ballades.1"}]}'
    out = score_continuation(_INITIAL, _RESULT, follow_up)
    assert out.category == "format_collapse"
    assert out.is_degenerate is True


def test_empty_classified_when_below_length_threshold() -> None:
    out = score_continuation(_INITIAL, _RESULT, "ok")
    assert out.category == "empty"
    assert out.is_degenerate is True


def test_load_tool_result_fixture_returns_dict_for_each_tool() -> None:
    for tool in (
        "create_exercise",
        "score_highlight",
        "keyboard_guide",
        "show_session_data",
        "reference_browser",
        "search_catalog",
    ):
        fixture = load_tool_result_fixture(tool)
        assert isinstance(fixture, dict)
        assert fixture, f"fixture for {tool} must be non-empty"


def test_load_tool_result_fixture_unknown_tool_raises() -> None:
    with pytest.raises(KeyError):
        load_tool_result_fixture("not_a_real_tool")
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/stage0/tests/test_continuation_probe.py -x -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'teacher_model.stage0.continuation_probe'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `apps/evals/teacher_model/stage0/data/continuation_fixtures.json`:

```json
{
  "create_exercise": {
    "exercises": [
      {
        "id": "ex_001",
        "title": "Slow LH practice",
        "instruction": "Play the LH alone at quarter = 60, focusing on even tone."
      }
    ]
  },
  "score_highlight": {
    "highlight_id": "hl_001",
    "bars": [12, 16],
    "dimension": "phrasing"
  },
  "keyboard_guide": {
    "guide_id": "kg_001",
    "label": "RH thumb-under turn"
  },
  "show_session_data": {
    "summary": {
      "duration_minutes": 12.5,
      "top_dimensions": ["pedaling", "phrasing"]
    }
  },
  "reference_browser": {
    "references": [
      {
        "performer": "Krystian Zimerman",
        "url": "https://example.invalid/zimerman-ballade1"
      }
    ]
  },
  "search_catalog": {
    "matches": [
      {
        "pieceId": "chopin.ballades.1",
        "composer": "Chopin",
        "title": "Ballade No. 1 in G minor",
        "barCount": 264
      }
    ]
  }
}
```

Create `apps/evals/teacher_model/stage0/continuation_probe.py`:

```python
"""Score post-tool-result assistant continuations against degeneracy categories.

Categories (in priority order):
  empty           -- response shorter than _MIN_CHARS or under _MIN_TOKENS_APPROX words
  refusal         -- explicit refusal phrases ("cannot continue", "unable to help")
  repetition      -- response re-emits the same tool_call as the initial turn
  format_collapse -- response is raw JSON / not natural prose
  clean           -- none of the above
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_FIXTURES_PATH = Path(__file__).parent / "data" / "continuation_fixtures.json"

_MIN_CHARS = 30
_MIN_TOKENS_APPROX = 10  # whitespace-split word count

_REFUSAL_PATTERNS = (
    re.compile(r"\bI (cannot|can't|am unable to|won't) (continue|help|proceed|assist)", re.IGNORECASE),
    re.compile(r"\bI (cannot|can't) (provide|do)\b", re.IGNORECASE),
    re.compile(r"\bI'm sorry,? (but )?I (cannot|can't)", re.IGNORECASE),
)

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_RAW_JSON_RE = re.compile(r"^\s*[\{\[]")


Category = Literal["clean", "empty", "refusal", "repetition", "format_collapse"]


@dataclass
class ContinuationResult:
    category: Category
    is_degenerate: bool
    detail: str


def _extract_tool_call_payload(text: str) -> str | None:
    m = _TOOL_CALL_RE.search(text)
    if not m:
        return None
    payload = m.group(1).strip()
    return payload


def _is_raw_json_dump(text: str) -> bool:
    s = text.strip()
    if not _RAW_JSON_RE.match(s):
        return False
    try:
        json.loads(s)
        return True
    except json.JSONDecodeError:
        return False


def score_continuation(
    initial_assistant: str,
    tool_result: dict,
    follow_up_response: str,
) -> ContinuationResult:
    """Classify a follow-up assistant response after a tool_result."""
    text = follow_up_response or ""
    stripped = text.strip()

    # 1. empty / truncated
    if len(stripped) < _MIN_CHARS or len(stripped.split()) < _MIN_TOKENS_APPROX:
        return ContinuationResult(
            category="empty",
            is_degenerate=True,
            detail=f"len={len(stripped)} chars, words={len(stripped.split())}",
        )

    # 2. refusal
    for pat in _REFUSAL_PATTERNS:
        if pat.search(stripped):
            return ContinuationResult(
                category="refusal",
                is_degenerate=True,
                detail=f"matched: {pat.pattern}",
            )

    # 3. repetition: same tool_call payload re-emitted
    initial_payload = _extract_tool_call_payload(initial_assistant)
    follow_up_payload = _extract_tool_call_payload(stripped)
    if initial_payload and follow_up_payload and initial_payload == follow_up_payload:
        return ContinuationResult(
            category="repetition",
            is_degenerate=True,
            detail="follow-up re-emits identical tool_call",
        )

    # 4. format collapse: raw JSON dump
    if _is_raw_json_dump(stripped):
        return ContinuationResult(
            category="format_collapse",
            is_degenerate=True,
            detail="response is parseable JSON, not prose",
        )

    return ContinuationResult(category="clean", is_degenerate=False, detail="")


def load_tool_result_fixture(tool_name: str) -> dict:
    """Load the canned tool_result payload for a given tool name."""
    fixtures = json.loads(_FIXTURES_PATH.read_text())
    if tool_name not in fixtures:
        raise KeyError(f"No continuation fixture for tool {tool_name!r}")
    return fixtures[tool_name]
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/stage0/tests/test_continuation_probe.py -x -q
```
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/stage0/continuation_probe.py apps/evals/teacher_model/stage0/data/continuation_fixtures.json apps/evals/teacher_model/stage0/tests/test_continuation_probe.py && git commit -m "feat(stage0): continuation_probe degeneracy classifier with 6-tool fixtures"
```

---

## Task 5: tool_scorer — extract + score tool calls in 3 formats

**Group:** A

**Behavior being verified:** given a raw model response, an expected case (positive: should call tool X with args matching schema; negative: should not call), and the tool schemas, returns a structured result with discipline + format-conditional schema validity.

**Interface under test:** `score_response(raw, expected, schemas) -> ToolProbeResult`.

**Files:**
- Create: `apps/evals/teacher_model/stage0/tool_scorer.py`
- Test: `apps/evals/teacher_model/stage0/tests/test_tool_scorer.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/teacher_model/stage0/tests/test_tool_scorer.py
"""Format extraction + schema validation across the 3 tolerated tool-call shapes."""
from __future__ import annotations

import pytest

from teacher_model.stage0.tool_scorer import (
    ToolCase,
    ToolProbeResult,
    score_response,
)

_SCHEMAS = {
    "search_catalog": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
    "create_exercise": {
        "type": "object",
        "properties": {
            "skill": {"type": "string"},
            "exercises": {"type": "array"},
        },
        "required": ["skill", "exercises"],
    },
}


def _pos_case(tool: str = "search_catalog") -> ToolCase:
    return ToolCase(case_id="p1", expected_call=True, expected_tool=tool, category=None)


def _neg_case(category: str = "chitchat") -> ToolCase:
    return ToolCase(case_id="n1", expected_call=False, expected_tool=None, category=category)


def test_qwen_native_format_with_valid_args_scores_correct_and_format_valid() -> None:
    raw = (
        "Sure, let me look that up.\n"
        "<tool_call>{\"name\": \"search_catalog\", \"arguments\": {\"query\": \"Chopin Op. 9 No. 2\"}}</tool_call>"
    )
    out = score_response(raw, _pos_case(), _SCHEMAS)
    assert out.discipline_correct is True
    assert out.called is True
    assert out.tool_name == "search_catalog"
    assert out.format_valid is True


def test_raw_json_format_with_valid_args_scores_correct() -> None:
    raw = '{"name": "search_catalog", "arguments": {"query": "Chopin"}}'
    out = score_response(raw, _pos_case(), _SCHEMAS)
    assert out.called is True
    assert out.tool_name == "search_catalog"
    assert out.format_valid is True
    assert out.discipline_correct is True


def test_prose_with_embedded_json_extracts_call() -> None:
    raw = (
        "I'll search for that. Here's my call:\n"
        '```json\n{"name": "search_catalog", "arguments": {"query": "Chopin"}}\n```\n'
        "Let me know what you find."
    )
    out = score_response(raw, _pos_case(), _SCHEMAS)
    assert out.called is True
    assert out.tool_name == "search_catalog"
    assert out.format_valid is True


def test_no_tool_call_on_negative_case_scores_disciplined() -> None:
    raw = "Let's just listen for another minute and see how the dynamics develop."
    out = score_response(raw, _neg_case(category="premature"), _SCHEMAS)
    assert out.called is False
    assert out.discipline_correct is True
    assert out.format_valid is None  # not applicable when no call


def test_tool_call_on_negative_case_scores_undisciplined() -> None:
    raw = '<tool_call>{"name": "search_catalog", "arguments": {"query": "anything"}}</tool_call>'
    out = score_response(raw, _neg_case(category="chitchat"), _SCHEMAS)
    assert out.called is True
    assert out.discipline_correct is False  # called when shouldn't have


def test_no_tool_call_on_positive_case_scores_undisciplined() -> None:
    raw = "I think we should talk about that more before searching."
    out = score_response(raw, _pos_case(), _SCHEMAS)
    assert out.called is False
    assert out.discipline_correct is False


def test_wrong_tool_name_scores_undisciplined_even_if_called() -> None:
    raw = '<tool_call>{"name": "create_exercise", "arguments": {"skill": "x", "exercises": []}}</tool_call>'
    out = score_response(raw, _pos_case(tool="search_catalog"), _SCHEMAS)
    assert out.called is True
    assert out.tool_name == "create_exercise"
    assert out.discipline_correct is False


def test_invalid_args_against_schema_marks_format_invalid() -> None:
    raw = '<tool_call>{"name": "search_catalog", "arguments": {}}</tool_call>'
    out = score_response(raw, _pos_case(), _SCHEMAS)
    assert out.called is True
    assert out.format_valid is False
    # Discipline depends on whether tool name was right; here it was, so:
    assert out.discipline_correct is True
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/stage0/tests/test_tool_scorer.py -x -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'teacher_model.stage0.tool_scorer'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `apps/evals/teacher_model/stage0/tool_scorer.py`:

```python
"""Extract and score tool calls from arbitrary-format base-model output.

Tolerated input shapes (in priority order):
  1. Qwen-native:   <tool_call>{...}</tool_call>
  2. Fenced JSON:   ```json\n{...}\n```  (or any fenced block)
  3. Raw JSON:      {"name": "...", "arguments": {...}}

Scoring is split:
  - discipline_correct: did the model make the right call/no-call decision,
    AND if it called, was the tool name correct?
  - format_valid:       given that it called, did `arguments` validate against
    the tool schema? None when no call.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from jsonschema import Draft202012Validator, ValidationError

_TOOL_CALL_TAG_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json|tool_call)?\s*(.*?)```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{(?:[^{}]|\{[^{}]*\})*\}", re.DOTALL)


@dataclass
class ToolCase:
    case_id: str
    expected_call: bool
    expected_tool: str | None  # None for negative cases
    category: str | None       # for negative cases: chitchat/premature/etc.


@dataclass
class ToolProbeResult:
    case_id: str
    called: bool
    tool_name: str | None
    arguments: dict | None
    discipline_correct: bool
    format_valid: bool | None
    extraction_format: str | None  # "qwen_tag" | "fenced" | "raw_json" | None
    error: str


def _try_parse_call(payload: str) -> tuple[str | None, dict | None]:
    """Parse a JSON blob expected to contain {name, arguments}."""
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        return None, None
    if not isinstance(obj, dict):
        return None, None
    name = obj.get("name")
    args = obj.get("arguments")
    if not isinstance(name, str):
        return None, None
    if args is not None and not isinstance(args, dict):
        return None, None
    return name, args or {}


def _extract_call(raw: str) -> tuple[str | None, dict | None, str | None]:
    """Return (tool_name, arguments, extraction_format) or (None, None, None)."""
    # 1. Qwen-native tag
    m = _TOOL_CALL_TAG_RE.search(raw)
    if m:
        name, args = _try_parse_call(m.group(1).strip())
        if name is not None:
            return name, args, "qwen_tag"

    # 2. Fenced JSON block
    for m in _FENCE_RE.finditer(raw):
        name, args = _try_parse_call(m.group(1).strip())
        if name is not None:
            return name, args, "fenced"

    # 3. Raw JSON object anywhere in the response
    for m in _JSON_OBJECT_RE.finditer(raw):
        name, args = _try_parse_call(m.group(0))
        if name is not None:
            return name, args, "raw_json"

    return None, None, None


def _validate_args(tool_name: str, args: dict, schemas: dict) -> tuple[bool, str]:
    schema = schemas.get(tool_name)
    if schema is None:
        return False, f"unknown tool: {tool_name}"
    try:
        Draft202012Validator(schema).validate(args)
        return True, ""
    except ValidationError as e:
        return False, e.message


def score_response(
    raw: str,
    expected: ToolCase,
    schemas: dict,
) -> ToolProbeResult:
    name, args, fmt = _extract_call(raw or "")
    called = name is not None

    if not called:
        # No call made.
        discipline = expected.expected_call is False
        return ToolProbeResult(
            case_id=expected.case_id,
            called=False,
            tool_name=None,
            arguments=None,
            discipline_correct=discipline,
            format_valid=None,
            extraction_format=None,
            error="",
        )

    # Call made; check tool name discipline + schema.
    if expected.expected_call is False:
        # Negative case but model called: undisciplined regardless of which tool.
        format_valid, err = _validate_args(name, args or {}, schemas) if name in schemas else (False, "unknown tool")
        return ToolProbeResult(
            case_id=expected.case_id,
            called=True,
            tool_name=name,
            arguments=args,
            discipline_correct=False,
            format_valid=format_valid,
            extraction_format=fmt,
            error=err,
        )

    # Positive case: tool name must match expected_tool.
    tool_name_correct = (name == expected.expected_tool)
    if not tool_name_correct:
        return ToolProbeResult(
            case_id=expected.case_id,
            called=True,
            tool_name=name,
            arguments=args,
            discipline_correct=False,
            format_valid=False,
            extraction_format=fmt,
            error=f"expected {expected.expected_tool!r}, got {name!r}",
        )

    format_valid, err = _validate_args(name, args or {}, schemas)
    return ToolProbeResult(
        case_id=expected.case_id,
        called=True,
        tool_name=name,
        arguments=args,
        discipline_correct=True,
        format_valid=format_valid,
        extraction_format=fmt,
        error=err,
    )
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/stage0/tests/test_tool_scorer.py -x -q
```
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/stage0/tool_scorer.py apps/evals/teacher_model/stage0/tests/test_tool_scorer.py && git commit -m "feat(stage0): tool_scorer extracts 3 formats and scores discipline + schema validity"
```

---

## Task 6: judge_extended — judge v2 + Taste defensibility + Adaptation specificity

**Group:** A

**Behavior being verified:** the extended judge prompt elicits 9-dim scores and the parser returns 9 `DimensionScore` records (7 base + Taste defensibility + Adaptation specificity) including criterion names that match the spec.

**Interface under test:** `parse_extended_judge_response(response_text) -> list[DimensionScore]` (pure parser, easily unit-tested) and `judge_extended(synthesis_text, context, provider, model)` (network-touching wrapper).

**Files:**
- Create: `apps/evals/teacher_model/stage0/judge_extended.py`
- Create: `apps/evals/teacher_model/stage0/prompts/judge_v2_extended.txt`
- Test: `apps/evals/teacher_model/stage0/tests/test_judge_extended.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/teacher_model/stage0/tests/test_judge_extended.py
"""Verify the extended judge parser returns 9 dims and the prompt enumerates all 9."""
from __future__ import annotations

import json
from pathlib import Path

from teacher_model.stage0.judge_extended import (
    EXTENDED_DIMS,
    parse_extended_judge_response,
)

_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "judge_v2_extended.txt"


def test_extended_dims_lists_nine_criteria_in_canonical_order() -> None:
    expected = [
        "Audible-Specific Corrective Feedback",
        "Concrete Artifact Provision",
        "Specific Positive Praise",
        "Autonomy-Supporting Motivation",
        "Scaffolded Guided Discovery",
        "Style-Consistent Musical Language",
        "Appropriate Tone & Language",
        "Taste Defensibility",
        "Adaptation Specificity",
    ]
    assert EXTENDED_DIMS == expected


def test_prompt_file_mentions_all_nine_dimensions() -> None:
    text = _PROMPT_PATH.read_text()
    for dim in EXTENDED_DIMS:
        assert dim in text, f"prompt missing dimension: {dim!r}"


def test_parser_returns_nine_dimension_scores() -> None:
    response_payload = json.dumps([
        {"criterion": dim, "process": 2, "outcome": 2, "evidence": "e", "reason": "r"}
        for dim in [
            "Audible-Specific Corrective Feedback",
            "Concrete Artifact Provision",
            "Specific Positive Praise",
            "Autonomy-Supporting Motivation",
            "Scaffolded Guided Discovery",
            "Style-Consistent Musical Language",
            "Appropriate Tone & Language",
            "Taste Defensibility",
            "Adaptation Specificity",
        ]
    ])
    dims = parse_extended_judge_response(response_payload)
    assert len(dims) == 9
    assert [d.criterion for d in dims] == EXTENDED_DIMS
    for d in dims:
        assert d.process == 2 and d.outcome == 2


def test_parser_handles_na_strings() -> None:
    payload = json.dumps([
        {"criterion": dim, "process": "N/A", "outcome": "N/A", "evidence": "", "reason": "n/a"}
        for dim in [
            "Audible-Specific Corrective Feedback",
            "Concrete Artifact Provision",
            "Specific Positive Praise",
            "Autonomy-Supporting Motivation",
            "Scaffolded Guided Discovery",
            "Style-Consistent Musical Language",
            "Appropriate Tone & Language",
            "Taste Defensibility",
            "Adaptation Specificity",
        ]
    ])
    dims = parse_extended_judge_response(payload)
    assert all(d.process is None and d.outcome is None for d in dims)
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/stage0/tests/test_judge_extended.py -x -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'teacher_model.stage0.judge_extended'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `apps/evals/teacher_model/stage0/prompts/judge_v2_extended.txt`:

```
You are an expert evaluator of AI-generated piano-teaching feedback. Use the rubric below to score each AI response on a 0-3 scale for every dimension. For each dimension, assign separate process and outcome scores, provide a brief evidence quote from the AI output, and explain why the scores fit the criteria. Follow these rules while scoring:

1. **Audible-Specific Corrective Feedback** - look for bar/beat references, a measurable audible target, and a concrete corrective action.
2. **Concrete Artifact Provision** - check for a specific exercise, score highlight, or recording suggestion, plus a short musical cue/metaphor.
3. **Specific Positive Praise** - require a concrete musical success, why it matters, and warm expressive language.
4. **Autonomy-Supporting Motivation** - must respect learner agency, offer a choice or goal, and give a clear next step.
5. **Scaffolded Guided Discovery** - evaluate chunking, sequencing, self-assessment prompts, and musical outcome linkage.
6. **Style-Consistent Musical Language** - compare the advice to the Piece-Style Dimension Rules in the playbook; advice that contradicts the style gets a 0.
7. **Appropriate Tone & Language** - ensure the register matches the feedback type and that no "never-say" statements appear.
8. **Taste Defensibility** - does the response make interpretive choices a thoughtful teacher could defend? 3 = clear stylistic stance with rationale; 2 = generic but stylistically plausible; 1 = generic / safe-only; 0 = stylistically wrong for the era / composer.
9. **Adaptation Specificity** - does the response include any bespoke modification (specific bars, hands separate, dotted-rhythm variant, etc.) vs only library-stock advice? 3 = concrete bespoke instruction; 2 = parameterized adaptation; 1 = mentions adaptation generically; 0 = stock advice only.

## Process vs. Outcome

Every dimension gets TWO scores:
- **process** (0-3): did the teacher notice or attempt the behavior this dimension is measuring? Reward effort-to-notice.
- **outcome** (0-3): was the advice correct, accurate, and useful given the actual performance? Reward correctness.

A teacher can try hard but be wrong (high process, low outcome), or nail the right advice without visibly trying (low process, high outcome). Keep these signals independent.

When a dimension is irrelevant (e.g., no praise present), set both process and outcome to "N/A" and the reason to "Not applicable to this response type." A score of 0 is reserved for harmful or contradictory content. Do NOT assign a default numeric score to irrelevant dimensions.

Output your assessment as a JSON array where each element is:
```json
{ "criterion": "<Dimension Name>", "process": <0-3 or "N/A">, "outcome": <0-3 or "N/A">, "evidence": "<short excerpt from AI response>", "reason": "<explanation>" }
```
Provide the array in the order of the nine dimensions listed above (1 through 9).
```

Create `apps/evals/teacher_model/stage0/judge_extended.py`:

```python
"""Extended judge: 7 base rubric dims + Taste defensibility + Adaptation specificity.

Reuses the existing apps/evals/shared/judge.py infrastructure (DimensionScore,
LLMClient, JSON-fence stripping). The only differences vs judge_synthesis_v2
are (a) the prompt file and (b) the parser tolerates 9 dims instead of 7.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Reuse the canonical DimensionScore + LLMClient + fence stripper.
from shared.judge import DimensionScore  # noqa: F401  (re-exported for callers)
from teaching_knowledge.llm_client import LLMClient, strip_json_fences

EXTENDED_DIMS: list[str] = [
    "Audible-Specific Corrective Feedback",
    "Concrete Artifact Provision",
    "Specific Positive Praise",
    "Autonomy-Supporting Motivation",
    "Scaffolded Guided Discovery",
    "Style-Consistent Musical Language",
    "Appropriate Tone & Language",
    "Taste Defensibility",
    "Adaptation Specificity",
]

_PROMPT_PATH = Path(__file__).parent / "prompts" / "judge_v2_extended.txt"


@dataclass
class JudgeResultV2Extended:
    dimensions: list[DimensionScore]
    model: str
    prompt_version: str
    latency_ms: float


class JudgeParseError(RuntimeError):
    """Raised when the extended judge response cannot be parsed into 9 dims."""


def _coerce_score(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        if value.strip().upper() in ("N/A", "NA", ""):
            return None
        try:
            value = int(value.strip())
        except ValueError:
            return None
    if isinstance(value, (int, float)):
        iv = int(value)
        if 0 <= iv <= 3:
            return iv
    return None


def parse_extended_judge_response(response_text: str) -> list[DimensionScore]:
    """Parse a v2-extended judge JSON-array response into 9 DimensionScore rows."""
    text = strip_json_fences(response_text)
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as e:
        raise JudgeParseError(f"judge response was not valid JSON: {e}") from e
    if not isinstance(raw, list):
        raise JudgeParseError(f"judge response must be a JSON array, got {type(raw).__name__}")

    by_criterion: dict[str, dict] = {}
    for item in raw:
        if isinstance(item, dict) and isinstance(item.get("criterion"), str):
            by_criterion[item["criterion"]] = item

    out: list[DimensionScore] = []
    for dim_name in EXTENDED_DIMS:
        item = by_criterion.get(dim_name)
        if item is None:
            raise JudgeParseError(f"missing dimension in judge response: {dim_name!r}")
        process = _coerce_score(item.get("process"))
        outcome = _coerce_score(item.get("outcome"))
        if process is not None and outcome is not None:
            score = min(process, outcome)
        else:
            score = process if process is not None else outcome
        out.append(
            DimensionScore(
                criterion=dim_name,
                score=score,
                evidence=str(item.get("evidence", ""))[:500],
                reason=str(item.get("reason", ""))[:1000],
                process=process,
                outcome=outcome,
            )
        )
    return out


def judge_extended(
    synthesis_text: str,
    context: dict[str, Any],
    provider: str = "workers-ai",
    model: str | None = None,
) -> JudgeResultV2Extended:
    """Judge a synthesis using the v2-extended rubric (9 dimensions)."""
    template = _PROMPT_PATH.read_text()
    user_message = (
        f"{template}\n\n"
        f"## Context\n"
        f"Piece: {context.get('piece_name', 'Unknown')} by {context.get('composer', 'Unknown')}\n"
        f"Student skill level: {context.get('skill_level', 'Unknown')}\n\n"
        f"## AI Teacher Output to Evaluate\n"
        f"{synthesis_text}"
    )

    client = LLMClient(provider=provider, model=model, tier="judge")
    start = time.monotonic()
    response_text = client.complete_json(user_message, max_tokens=4000)
    latency_ms = (time.monotonic() - start) * 1000

    dimensions = parse_extended_judge_response(response_text)
    return JudgeResultV2Extended(
        dimensions=dimensions,
        model=client.model,
        prompt_version="judge_v2_extended",
        latency_ms=latency_ms,
    )
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/stage0/tests/test_judge_extended.py -x -q
```
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/stage0/judge_extended.py apps/evals/teacher_model/stage0/prompts/judge_v2_extended.txt apps/evals/teacher_model/stage0/tests/test_judge_extended.py && git commit -m "feat(stage0): judge_extended adds Taste Defensibility + Adaptation Specificity dims"
```

---

## Task 7: sampler — stratified n=100 holdout from the 890-briefing pool

**Group:** A

**Behavior being verified:** `sample_holdout` returns exactly the requested count (or as close as the strata allow), respects deterministic seed, and produces strata aligned to era × skill_bucket.

**Interface under test:** `sample_holdout(briefings_dir, manifests, n, seed) -> list[dict]`.

**Files:**
- Create: `apps/evals/teacher_model/stage0/sampler.py`
- Test: `apps/evals/teacher_model/stage0/tests/test_sampler.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/teacher_model/stage0/tests/test_sampler.py
"""Stratified sampler determinism + shape tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from teacher_model.stage0.sampler import sample_holdout

_REPO_ROOT = Path(__file__).resolve().parents[5]
_BRIEFINGS_DIR = (
    _REPO_ROOT / "model" / "data" / "eval" / "inference_cache" / "auto-t5_http"
)


def _load_real_manifests() -> dict:
    """Load skill_eval manifests; reuse the same lookup the production runner uses."""
    import sys

    sys.path.insert(0, str(_REPO_ROOT / "apps" / "evals"))
    from teaching_knowledge.run_eval import load_manifests

    return load_manifests()


@pytest.mark.skipif(not _BRIEFINGS_DIR.exists(), reason="briefings dir not present")
def test_same_seed_produces_same_ids() -> None:
    manifests = _load_real_manifests()
    a = sample_holdout(_BRIEFINGS_DIR, manifests, n=50, seed=42)
    b = sample_holdout(_BRIEFINGS_DIR, manifests, n=50, seed=42)
    assert [r["recording_id"] for r in a] == [r["recording_id"] for r in b]


@pytest.mark.skipif(not _BRIEFINGS_DIR.exists(), reason="briefings dir not present")
def test_different_seed_produces_different_ids() -> None:
    manifests = _load_real_manifests()
    a = sample_holdout(_BRIEFINGS_DIR, manifests, n=50, seed=42)
    b = sample_holdout(_BRIEFINGS_DIR, manifests, n=50, seed=43)
    assert [r["recording_id"] for r in a] != [r["recording_id"] for r in b]


@pytest.mark.skipif(not _BRIEFINGS_DIR.exists(), reason="briefings dir not present")
def test_sample_returns_at_most_n_with_required_fields() -> None:
    manifests = _load_real_manifests()
    out = sample_holdout(_BRIEFINGS_DIR, manifests, n=100, seed=42)
    assert 80 <= len(out) <= 100  # strata may underfill if some buckets are sparse
    for row in out:
        assert "recording_id" in row
        assert "era" in row
        assert "skill_bucket" in row
        assert "stratum" in row


@pytest.mark.skipif(not _BRIEFINGS_DIR.exists(), reason="briefings dir not present")
def test_strata_balanced_within_two() -> None:
    manifests = _load_real_manifests()
    out = sample_holdout(_BRIEFINGS_DIR, manifests, n=100, seed=42)
    counts: dict[str, int] = {}
    for row in out:
        counts[row["stratum"]] = counts.get(row["stratum"], 0) + 1
    if not counts:
        pytest.skip("no strata found")
    target = max(counts.values())
    for c in counts.values():
        assert abs(c - target) <= max(2, target // 3)
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/stage0/tests/test_sampler.py -x -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'teacher_model.stage0.sampler'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `apps/evals/teacher_model/stage0/sampler.py`:

```python
"""Stratified holdout sampler for Stage 0 synthesis eval.

Produces an n=N sample from the briefing pool stratified by era x skill_bucket.
Era is derived from composer via shared.style_rules.composer_to_era; strata
with fewer than _MIN_STRATUM_POPULATION recordings are merged into a fallback
'Other' era stratum to avoid degenerate single-recording cells.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from shared.style_rules import composer_to_era

_MIN_STRATUM_POPULATION = 5


def _stratum_label(era: str, skill_bucket: int) -> str:
    return f"{era}|sk{skill_bucket}"


def sample_holdout(
    briefings_dir: Path,
    manifests: dict[str, dict],
    n: int,
    seed: int,
) -> list[dict]:
    """Return up to n briefings stratified by era x skill_bucket.

    Args:
        briefings_dir: directory containing one .json per recording.
        manifests: video_id -> {composer, skill_bucket, ...} (from load_manifests).
        n: target sample size.
        seed: deterministic seed.

    Each returned row has keys: recording_id, era, skill_bucket, stratum,
    composer, piece_slug, title, briefing_path.
    """
    cache_files = sorted(briefings_dir.glob("*.json"))

    # Build the candidate pool: only recordings present in manifests.
    candidates: dict[str, list[dict]] = {}
    for path in cache_files:
        if path.name == "_fingerprint.json":
            continue
        rid = path.stem
        meta = manifests.get(rid)
        if meta is None:
            continue
        era = composer_to_era(meta.get("composer", ""))
        skill_bucket = int(meta.get("skill_bucket", 3))
        stratum = _stratum_label(era, skill_bucket)
        candidates.setdefault(stratum, []).append(
            {
                "recording_id": rid,
                "era": era,
                "skill_bucket": skill_bucket,
                "stratum": stratum,
                "composer": meta.get("composer", ""),
                "piece_slug": meta.get("piece_slug", ""),
                "title": meta.get("title", ""),
                "briefing_path": str(path),
            }
        )

    # Drop strata smaller than _MIN_STRATUM_POPULATION; fold into an "Other" stratum.
    keep: dict[str, list[dict]] = {}
    other: list[dict] = []
    for stratum, rows in candidates.items():
        if len(rows) >= _MIN_STRATUM_POPULATION:
            keep[stratum] = rows
        else:
            for r in rows:
                r2 = dict(r)
                r2["stratum"] = "Other"
                other.append(r2)
    if other:
        keep["Other"] = other

    if not keep:
        return []

    # Per-stratum target; floor division then distribute remainder by stratum size.
    rng = random.Random(seed)
    strata_sorted = sorted(keep.keys())
    base = n // len(strata_sorted)
    remainder = n - base * len(strata_sorted)

    # Distribute remainder to the largest strata first (deterministic).
    sizes = [(s, len(keep[s])) for s in strata_sorted]
    sizes.sort(key=lambda x: (-x[1], x[0]))
    extra: dict[str, int] = {s: 0 for s in strata_sorted}
    for s, _ in sizes[:remainder]:
        extra[s] = 1

    out: list[dict] = []
    for stratum in strata_sorted:
        rows = list(keep[stratum])
        rng.shuffle(rows)
        target = min(base + extra[stratum], len(rows))
        out.extend(rows[:target])

    out.sort(key=lambda r: r["recording_id"])  # deterministic output ordering
    return out


def write_holdout(rows: list[dict], out_path: Path) -> None:
    """Persist the holdout to a JSONL file (one row per line)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/stage0/tests/test_sampler.py -x -q
```
Expected: PASS (4 tests; some may skip if briefings dir absent in CI).

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/stage0/sampler.py apps/evals/teacher_model/stage0/tests/test_sampler.py && git commit -m "feat(stage0): stratified era x skill holdout sampler with deterministic seed"
```

---

## Task 8: tool_probe_cases.jsonl + loader

**Group:** A

**Behavior being verified:** the committed cases file parses, contains exactly 20 positive + 20 negative cases, all 6 negative categories appear, all positive cases reference one of the 6 production tools, and the loader returns typed `ToolCase` objects.

**Interface under test:** `load_cases(path) -> list[ToolCase]`.

**Curation directive (for the implementing subagent):** draft 40 cases by sampling briefings from `model/data/eval/inference_cache/auto-t5_http/` and constructing the brief context that would be shown to the teacher. Negative categories are: `chitchat`, `premature`, `ambiguous`, `already_recommended`, `out_of_scope`, `borderline_wrong_tool` — produce 3-4 cases per category (totaling 20). Positive cases distribute across the 6 tools (~3-4 per tool). Each case body holds the briefing JSON; expected_tool/category/expected_call live at top level. **Commit the file as drafted; the founder will review post-merge before the tool probe runs.** Do NOT use Sonnet to generate the labels — that contaminates the comparison.

**Files:**
- Create: `apps/evals/teacher_model/stage0/data/tool_probe_cases.jsonl`
- Create: `apps/evals/teacher_model/stage0/cases.py`
- Test: `apps/evals/teacher_model/stage0/tests/test_cases.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/teacher_model/stage0/tests/test_cases.py
"""Verify the committed tool-probe cases file has the required structure."""
from __future__ import annotations

from pathlib import Path

from teacher_model.stage0.cases import load_cases

_CASES_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "tool_probe_cases.jsonl"
)

_VALID_TOOLS = {
    "create_exercise",
    "score_highlight",
    "keyboard_guide",
    "show_session_data",
    "reference_browser",
    "search_catalog",
}

_VALID_NEG_CATEGORIES = {
    "chitchat",
    "premature",
    "ambiguous",
    "already_recommended",
    "out_of_scope",
    "borderline_wrong_tool",
}


def test_file_has_exactly_40_cases() -> None:
    cases = load_cases(_CASES_PATH)
    assert len(cases) == 40


def test_split_is_20_positive_20_negative() -> None:
    cases = load_cases(_CASES_PATH)
    pos = [c for c in cases if c.expected_call]
    neg = [c for c in cases if not c.expected_call]
    assert len(pos) == 20 and len(neg) == 20


def test_positive_cases_use_only_known_tools() -> None:
    cases = load_cases(_CASES_PATH)
    for c in cases:
        if c.expected_call:
            assert c.expected_tool in _VALID_TOOLS, f"unknown tool: {c.expected_tool}"


def test_negative_cases_cover_all_six_categories() -> None:
    cases = load_cases(_CASES_PATH)
    cats = {c.category for c in cases if not c.expected_call and c.category}
    assert cats == _VALID_NEG_CATEGORIES, f"missing categories: {_VALID_NEG_CATEGORIES - cats}"


def test_each_case_has_a_briefing_field() -> None:
    cases = load_cases(_CASES_PATH)
    for c in cases:
        assert c.briefing, f"case {c.case_id} missing briefing"
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/stage0/tests/test_cases.py -x -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'teacher_model.stage0.cases'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `apps/evals/teacher_model/stage0/cases.py`:

```python
"""Tool-probe case loader.

The case file is one JSON object per line:
{
  "case_id": "p_search_01",
  "expected_call": true,
  "expected_tool": "search_catalog",
  "category": null,            // for positives
  "briefing": { ... full briefing object the model is shown ... }
}

For negatives, `expected_call` is false, `expected_tool` is null, and
`category` is one of: chitchat / premature / ambiguous / already_recommended
/ out_of_scope / borderline_wrong_tool.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ToolCase:
    case_id: str
    expected_call: bool
    expected_tool: str | None
    category: str | None
    briefing: dict


def load_cases(path: Path) -> list[ToolCase]:
    """Read tool_probe_cases.jsonl into ToolCase records."""
    out: list[ToolCase] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        out.append(
            ToolCase(
                case_id=row["case_id"],
                expected_call=bool(row["expected_call"]),
                expected_tool=row.get("expected_tool"),
                category=row.get("category"),
                briefing=row["briefing"],
            )
        )
    return out
```

Now create the data file. The implementing subagent must draft 40 cases following the curation directive above. The format below is the exact required schema for each line; substitute real briefing content drawn from the briefings pool. Each `briefing` object should mirror the structure produced by `teaching_knowledge.run_eval.build_synthesis_user_msg` (i.e. it embeds `session_data` JSON plus optional `style_guidance` / `voice_blocks`). For the cases file you may simplify: store the rendered user-message string under a `prompt` sub-key, plus the original `recording_id`.

Create `apps/evals/teacher_model/stage0/data/tool_probe_cases.jsonl` with 40 lines. The subagent must produce real content; the schema is mandatory but the briefings drawn must be human-meaningful. Skeleton example for one positive and one negative line (subagent extends to 40 total):

```jsonl
{"case_id":"p_search_01","expected_call":true,"expected_tool":"search_catalog","category":null,"briefing":{"recording_id":"_-smhag-Q0E","prompt":"<session_data>...top moments show large positive deviation in pedaling for Chopin Op. 9 No. 2; student asks: 'do you know any similar pieces I could try next?'</session_data>"}}
{"case_id":"n_chitchat_01","expected_call":false,"expected_tool":null,"category":"chitchat","briefing":{"recording_id":"_3qnL9ddHuw","prompt":"<session_data>...student message: 'thanks, I appreciated that suggestion last week. how was your weekend?'</session_data>"}}
```

Distribute as: 20 positives across `create_exercise` (4), `score_highlight` (4), `keyboard_guide` (3), `show_session_data` (3), `reference_browser` (3), `search_catalog` (3); 20 negatives split across the 6 categories with 3-4 cases each (e.g. 4/3/3/4/3/3 = 20). Use unique `case_id` values like `p_<tool>_NN` and `n_<category>_NN`.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/stage0/tests/test_cases.py -x -q
```
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/stage0/cases.py apps/evals/teacher_model/stage0/data/tool_probe_cases.jsonl apps/evals/teacher_model/stage0/tests/test_cases.py && git commit -m "feat(stage0): 40 hand-curated tool-probe cases (20 positive, 20 negative across 6 categories)"
```

---

## Task 9: pyproject.toml — add `teacher-model-stage0` optional-dependencies group

**Group:** A

**Behavior being verified:** `uv sync --extra teacher-model-stage0` resolves `transformers` and `jsonschema` (the two new deps), so Tasks 3 and 5 can run their tests.

**Interface under test:** `pyproject.toml` parsed via `tomllib`.

**Files:**
- Modify: `apps/evals/pyproject.toml`
- Test: `apps/evals/teacher_model/stage0/tests/test_pyproject_extras.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/teacher_model/stage0/tests/test_pyproject_extras.py
"""Stage 0 needs transformers + jsonschema in a dedicated extras group."""
from __future__ import annotations

import tomllib
from pathlib import Path

_PYPROJECT = Path(__file__).resolve().parents[4] / "pyproject.toml"


def test_teacher_model_stage0_extras_present() -> None:
    data = tomllib.loads(_PYPROJECT.read_text())
    extras = data.get("project", {}).get("optional-dependencies", {})
    assert "teacher-model-stage0" in extras
    deps_str = " ".join(extras["teacher-model-stage0"])
    assert "transformers" in deps_str
    assert "jsonschema" in deps_str
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/stage0/tests/test_pyproject_extras.py -x -q
```
Expected: FAIL — `AssertionError: 'teacher-model-stage0' not in extras`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Edit `apps/evals/pyproject.toml`. In the `[project.optional-dependencies]` table, add a new group **above** the `all` entry:

```toml
teacher-model-stage0 = [
    "transformers>=4.30.0",
    "jsonschema>=4.20.0",
]
```

And update the `all` entry to include it:

```toml
all = [
    "crescendai-evals[model,memory,inference,teacher-model-stage0]",
]
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/stage0/tests/test_pyproject_extras.py -x -q
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/evals/pyproject.toml apps/evals/teacher_model/stage0/tests/test_pyproject_extras.py && git commit -m "build(stage0): add teacher-model-stage0 extras group (transformers, jsonschema)"
```

---

## Task 10: aggregator — build the capability dossier

**Group:** B (depends on Task 2: tier_classifier)

**Behavior being verified:** given synthesis JSONL, tool JSONL, MCQ summary JSON, optional continuation JSONL, and the Sonnet baseline aggregate, `build_dossier` produces a `Dossier` with one row per capability containing primary value, CI, vs-Sonnet delta, tier, inconsistency flag, and an error-rate gate that refuses to emit when any pipeline exceeds 5% errors.

**Interface under test:** `build_dossier(synthesis_jsonl, tool_jsonl, mcq_json, baseline_aggregate_json, out_dir, continuation_jsonl=None) -> Dossier`.

**Files:**
- Create: `apps/evals/teacher_model/stage0/aggregator.py`
- Test: `apps/evals/teacher_model/stage0/tests/test_aggregator.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/teacher_model/stage0/tests/test_aggregator.py
"""Aggregator: dossier shape, inconsistency flag, error-rate gate."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from teacher_model.stage0.aggregator import (
    DossierEmissionRefused,
    build_dossier,
)

_SONNET_BASELINE = {
    "dimensions": [
        {"name": "Audible-Specific Corrective Feedback", "mean_outcome": 1.387, "n": 504},
        {"name": "Concrete Artifact Provision", "mean_outcome": 2.164, "n": 511},
        {"name": "Specific Positive Praise", "mean_outcome": 2.834, "n": 513},
        {"name": "Autonomy-Supporting Motivation", "mean_outcome": 2.85, "n": 512},
        {"name": "Scaffolded Guided Discovery", "mean_outcome": 2.195, "n": 512},
        {"name": "Style-Consistent Musical Language", "mean_outcome": 3.0, "n": 513},
        {"name": "Appropriate Tone & Language", "mean_outcome": 3.0, "n": 513},
    ],
    "composite_mean": 2.483,
}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _synth_row(rid: str, scores: dict[str, int]) -> dict:
    return {
        "recording_id": rid,
        "judge_dimensions": [
            {"criterion": k, "process": v, "outcome": v, "score": v, "evidence": "", "reason": ""}
            for k, v in scores.items()
        ],
        "error": "",
        "routed_provider": "openai",
    }


_FULL_NINE = {
    "Audible-Specific Corrective Feedback": 1,
    "Concrete Artifact Provision": 2,
    "Specific Positive Praise": 3,
    "Autonomy-Supporting Motivation": 3,
    "Scaffolded Guided Discovery": 2,
    "Style-Consistent Musical Language": 3,
    "Appropriate Tone & Language": 3,
    "Taste Defensibility": 1,
    "Adaptation Specificity": 1,
}


def test_dossier_emits_seven_capability_rows(tmp_path: Path) -> None:
    synth = tmp_path / "synth.jsonl"
    tool = tmp_path / "tool.jsonl"
    mcq = tmp_path / "mcq.json"
    base = tmp_path / "base.json"

    _write_jsonl(synth, [_synth_row(f"r{i}", _FULL_NINE) for i in range(20)])
    _write_jsonl(
        tool,
        [
            {"case_id": f"p{i}", "expected_call": True, "called": True, "discipline_correct": True, "format_valid": True, "category": None, "error": ""}
            for i in range(20)
        ]
        + [
            {"case_id": f"n{i}", "expected_call": False, "called": False, "discipline_correct": True, "format_valid": None, "category": "chitchat", "error": ""}
            for i in range(20)
        ],
    )
    mcq.write_text(json.dumps({
        "accuracy": 0.7, "total": 50, "correct": 35,
        "by_topic": {"concepts": {"accuracy": 0.7, "total": 10, "correct": 7}}
    }))
    base.write_text(json.dumps(_SONNET_BASELINE))

    dossier = build_dossier(
        synthesis_jsonl=synth,
        tool_jsonl=tool,
        mcq_json=mcq,
        baseline_aggregate_json=base,
        out_dir=tmp_path / "out",
    )
    names = [c.name for c in dossier.capabilities]
    assert names == [
        "Judgment", "Taste", "Integration", "Voice",
        "Vocabulary", "Tool-calling", "Adaptation",
    ]
    md = (tmp_path / "out" / "capability_dossier.md").read_text()
    assert "Capability dossier" in md
    js = json.loads((tmp_path / "out" / "capability_dossier.json").read_text())
    assert len(js["capabilities"]) == 7


def test_error_rate_gate_refuses_emission_above_five_percent(tmp_path: Path) -> None:
    synth = tmp_path / "synth.jsonl"
    tool = tmp_path / "tool.jsonl"
    mcq = tmp_path / "mcq.json"
    base = tmp_path / "base.json"

    rows = [_synth_row(f"r{i}", _FULL_NINE) for i in range(94)] + [
        {"recording_id": f"r{i}", "judge_dimensions": [], "error": "judge timeout", "routed_provider": "openai"}
        for i in range(94, 100)
    ]
    _write_jsonl(synth, rows)
    _write_jsonl(tool, [
        {"case_id": "p1", "expected_call": True, "called": True, "discipline_correct": True, "format_valid": True, "category": None, "error": ""}
    ])
    mcq.write_text(json.dumps({"accuracy": 0.5, "total": 50, "correct": 25, "by_topic": {}}))
    base.write_text(json.dumps(_SONNET_BASELINE))

    with pytest.raises(DossierEmissionRefused, match="error rate"):
        build_dossier(
            synthesis_jsonl=synth, tool_jsonl=tool, mcq_json=mcq,
            baseline_aggregate_json=base, out_dir=tmp_path / "out",
        )


def test_inconsistency_flag_when_primary_and_corroborator_disagree(tmp_path: Path) -> None:
    synth = tmp_path / "synth.jsonl"
    tool = tmp_path / "tool.jsonl"
    mcq = tmp_path / "mcq.json"
    base = tmp_path / "base.json"

    # Vocabulary primary (SCML) at-ceiling (3) but corroborating MCQ concepts =0 -> inconsistency.
    _write_jsonl(synth, [_synth_row(f"r{i}", _FULL_NINE) for i in range(20)])
    _write_jsonl(tool, [
        {"case_id": "p1", "expected_call": True, "called": True, "discipline_correct": True, "format_valid": True, "category": None, "error": ""}
    ])
    mcq.write_text(json.dumps({
        "accuracy": 0.0, "total": 50, "correct": 0,
        "by_topic": {"concepts": {"accuracy": 0.0, "total": 10, "correct": 0}}
    }))
    base.write_text(json.dumps(_SONNET_BASELINE))

    dossier = build_dossier(
        synthesis_jsonl=synth, tool_jsonl=tool, mcq_json=mcq,
        baseline_aggregate_json=base, out_dir=tmp_path / "out",
    )
    vocab = next(c for c in dossier.capabilities if c.name == "Vocabulary")
    assert vocab.inconsistency_flag is True


def test_continuation_metrics_appear_in_dossier_when_provided(tmp_path: Path) -> None:
    synth = tmp_path / "synth.jsonl"
    tool = tmp_path / "tool.jsonl"
    mcq = tmp_path / "mcq.json"
    base = tmp_path / "base.json"
    cont = tmp_path / "cont.jsonl"

    _write_jsonl(synth, [_synth_row(f"r{i}", _FULL_NINE) for i in range(20)])
    _write_jsonl(tool, [
        {"case_id": "p1", "expected_call": True, "called": True, "discipline_correct": True, "format_valid": True, "category": None, "error": ""}
    ])
    mcq.write_text(json.dumps({"accuracy": 0.5, "total": 50, "correct": 25, "by_topic": {}}))
    base.write_text(json.dumps(_SONNET_BASELINE))
    _write_jsonl(cont, [
        {"case_id": "p1", "category": "clean", "is_degenerate": False},
        {"case_id": "p2", "category": "refusal", "is_degenerate": True},
        {"case_id": "p3", "category": "repetition", "is_degenerate": True},
        {"case_id": "p4", "category": "clean", "is_degenerate": False},
    ])

    dossier = build_dossier(
        synthesis_jsonl=synth, tool_jsonl=tool, mcq_json=mcq,
        baseline_aggregate_json=base, out_dir=tmp_path / "out",
        continuation_jsonl=cont,
    )
    assert dossier.continuation_degeneracy_rate == 0.5
    assert dossier.continuation_by_category["refusal"] == 1
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/stage0/tests/test_aggregator.py -x -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'teacher_model.stage0.aggregator'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `apps/evals/teacher_model/stage0/aggregator.py`:

```python
"""Build the Stage 0 capability dossier from probe result files."""
from __future__ import annotations

import json
import random
import statistics
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable

from teacher_model.stage0.tier_classifier import classify_tier

_ERROR_RATE_GATE = 0.05  # refuse to emit if any pipeline exceeds 5% errors

# Sonnet baseline lookup -- field name in the aggregate file for outcome means.
_DIM_KEY = "mean_outcome"


class DossierEmissionRefused(RuntimeError):
    """Raised when error rates exceed _ERROR_RATE_GATE."""


@dataclass
class CapabilityRow:
    name: str
    primary_signal: str
    primary_value: float
    primary_ci: tuple[float, float] | None
    corroborating_signal: str | None
    corroborating_value: float | None
    tier: str
    anchor_type: str  # "relative" | "absolute"
    sonnet_baseline: float | None
    delta_vs_sonnet: float | None
    inconsistency_flag: bool
    notes: str = ""


@dataclass
class Dossier:
    meta: dict
    capabilities: list[CapabilityRow]
    continuation_degeneracy_rate: float | None = None
    continuation_by_category: dict[str, int] = field(default_factory=dict)
    over_call_by_category: dict[str, float] = field(default_factory=dict)


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def _synthesis_outcome_means(rows: list[dict]) -> dict[str, list[int]]:
    by_dim: dict[str, list[int]] = {}
    for row in rows:
        if row.get("error"):
            continue
        for d in row.get("judge_dimensions", []):
            outcome = d.get("outcome")
            if isinstance(outcome, (int, float)):
                by_dim.setdefault(d["criterion"], []).append(int(outcome))
    return by_dim


def _bootstrap_ci(values: list[float], n_resamples: int = 1000, seed: int = 1234) -> tuple[float, float]:
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    resamples = []
    n = len(values)
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        resamples.append(sum(sample) / n)
    resamples.sort()
    lo = resamples[int(0.025 * n_resamples)]
    hi = resamples[int(0.975 * n_resamples) - 1]
    return (lo, hi)


def _wilson_ci_proportion(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    radius = (z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5) / denom
    return (max(0.0, centre - radius), min(1.0, centre + radius))


def _baseline_lookup(baseline: dict, dim: str) -> float:
    for d in baseline.get("dimensions", []):
        if d.get("name") == dim:
            return float(d[_DIM_KEY])
    raise KeyError(f"no baseline entry for {dim!r}")


def _check_error_rate(rows: list[dict], pipeline: str) -> None:
    if not rows:
        return
    errors = sum(1 for r in rows if r.get("error"))
    rate = errors / len(rows)
    if rate > _ERROR_RATE_GATE:
        raise DossierEmissionRefused(
            f"{pipeline}: error rate {rate:.1%} exceeds gate {_ERROR_RATE_GATE:.0%} "
            f"({errors}/{len(rows)})"
        )


def _mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def _avg_dim(by_dim: dict[str, list[int]], dims: list[str]) -> tuple[float, list[float]]:
    """Average a per-row mean across multiple dim names; return (overall mean, per-row averages)."""
    if not by_dim or not all(dims_in := dims):
        return float("nan"), []
    series_lengths = [len(by_dim.get(d, [])) for d in dims]
    n = min(series_lengths) if series_lengths else 0
    if n == 0:
        return float("nan"), []
    per_row: list[float] = []
    for i in range(n):
        per_row.append(sum(by_dim[d][i] for d in dims) / len(dims))
    return _mean(per_row), per_row


def _build_judgment_row(by_dim: dict[str, list[int]], baseline: dict) -> CapabilityRow:
    ascf_mean = _mean(by_dim.get("Audible-Specific Corrective Feedback", []))
    sgd_mean = _mean(by_dim.get("Scaffolded Guided Discovery", []))
    point = (ascf_mean + sgd_mean) / 2
    ascf_base = _baseline_lookup(baseline, "Audible-Specific Corrective Feedback")
    sgd_base = _baseline_lookup(baseline, "Scaffolded Guided Discovery")
    deltas = [ascf_base - ascf_mean, sgd_base - sgd_mean]
    worst = max(deltas)
    baseline_avg = (ascf_base + sgd_base) / 2
    primary_ci = _bootstrap_ci(_avg_dim(by_dim, ["Audible-Specific Corrective Feedback", "Scaffolded Guided Discovery"])[1])
    tier = classify_tier(value=baseline_avg - worst, baseline=baseline_avg, mode="relative", ci=primary_ci)
    return CapabilityRow(
        name="Judgment",
        primary_signal="avg(ASCF, SGD) outcome",
        primary_value=point,
        primary_ci=primary_ci,
        corroborating_signal=None,
        corroborating_value=None,
        tier=tier,
        anchor_type="relative",
        sonnet_baseline=baseline_avg,
        delta_vs_sonnet=point - baseline_avg,
        inconsistency_flag=False,
    )


def _build_taste_row(by_dim: dict[str, list[int]]) -> CapabilityRow:
    vals = by_dim.get("Taste Defensibility", [])
    point = _mean(vals)
    ci = _bootstrap_ci([float(v) for v in vals])
    return CapabilityRow(
        name="Taste",
        primary_signal="Taste Defensibility (NEW)",
        primary_value=point,
        primary_ci=ci,
        corroborating_signal=None,
        corroborating_value=None,
        tier=classify_tier(value=point, baseline=None, mode="absolute", ci=ci),
        anchor_type="absolute",
        sonnet_baseline=None,
        delta_vs_sonnet=None,
        inconsistency_flag=False,
        notes="no baseline anchor",
    )


def _build_integration_row(by_dim: dict[str, list[int]], baseline: dict) -> CapabilityRow:
    cap = by_dim.get("Concrete Artifact Provision", [])
    cap_mean = _mean(cap)
    cap_ci = _bootstrap_ci([float(v) for v in cap])
    cap_base = _baseline_lookup(baseline, "Concrete Artifact Provision")

    base_dims = [
        "Audible-Specific Corrective Feedback",
        "Concrete Artifact Provision",
        "Specific Positive Praise",
        "Autonomy-Supporting Motivation",
        "Scaffolded Guided Discovery",
        "Style-Consistent Musical Language",
        "Appropriate Tone & Language",
    ]
    composite_mean, composite_series = _avg_dim(by_dim, base_dims)
    composite_baseline = baseline.get("composite_mean", float("nan"))

    primary_tier = classify_tier(value=cap_mean, baseline=cap_base, mode="relative", ci=cap_ci)
    composite_tier = classify_tier(value=composite_mean, baseline=composite_baseline, mode="relative")
    inconsistent = primary_tier != composite_tier and not _adjacent_tier(primary_tier, composite_tier)
    return CapabilityRow(
        name="Integration",
        primary_signal="CAP outcome",
        primary_value=cap_mean,
        primary_ci=cap_ci,
        corroborating_signal="composite mean (7 base dims)",
        corroborating_value=composite_mean,
        tier=primary_tier,
        anchor_type="relative",
        sonnet_baseline=cap_base,
        delta_vs_sonnet=cap_mean - cap_base,
        inconsistency_flag=inconsistent,
    )


def _build_voice_row(by_dim: dict[str, list[int]], baseline: dict) -> CapabilityRow:
    dims = ["Specific Positive Praise", "Appropriate Tone & Language", "Autonomy-Supporting Motivation"]
    point, series = _avg_dim(by_dim, dims)
    bases = [_baseline_lookup(baseline, d) for d in dims]
    base_mean = sum(bases) / len(bases)
    ci = _bootstrap_ci(series) if series else None
    return CapabilityRow(
        name="Voice",
        primary_signal="avg(SPP, ATL, ASM) outcome",
        primary_value=point,
        primary_ci=ci,
        corroborating_signal=None,
        corroborating_value=None,
        tier=classify_tier(value=point, baseline=base_mean, mode="relative", ci=ci),
        anchor_type="relative",
        sonnet_baseline=base_mean,
        delta_vs_sonnet=point - base_mean,
        inconsistency_flag=False,
    )


def _build_vocabulary_row(by_dim: dict[str, list[int]], baseline: dict, mcq: dict) -> CapabilityRow:
    scml = by_dim.get("Style-Consistent Musical Language", [])
    point = _mean(scml)
    ci = _bootstrap_ci([float(v) for v in scml])
    base_v = _baseline_lookup(baseline, "Style-Consistent Musical Language")
    primary_tier = classify_tier(value=point, baseline=base_v, mode="relative", ci=ci)
    concepts = mcq.get("by_topic", {}).get("concepts", {})
    concepts_acc = float(concepts.get("accuracy", 0.0)) if concepts else 0.0
    inconsistent = (primary_tier == "at_ceiling" and concepts_acc < 0.6)
    return CapabilityRow(
        name="Vocabulary",
        primary_signal="SCML outcome",
        primary_value=point,
        primary_ci=ci,
        corroborating_signal="MCQ concepts accuracy",
        corroborating_value=concepts_acc,
        tier=primary_tier,
        anchor_type="relative",
        sonnet_baseline=base_v,
        delta_vs_sonnet=point - base_v,
        inconsistency_flag=inconsistent,
    )


def _build_tool_calling_row(tool_rows: list[dict]) -> CapabilityRow:
    valid = [r for r in tool_rows if not r.get("error")]
    if not valid:
        point = 0.0
        ci = (0.0, 0.0)
    else:
        correct = sum(1 for r in valid if r.get("discipline_correct"))
        point = correct / len(valid)
        ci = _wilson_ci_proportion(correct, len(valid))
    # absolute thresholds use 0-3 scale; rescale percent so 80% -> 2.4, 50% -> 1.5, <50% -> <1.5
    abs_value = point * 3.0
    abs_ci = (ci[0] * 3.0, ci[1] * 3.0)
    return CapabilityRow(
        name="Tool-calling",
        primary_signal="discipline accuracy",
        primary_value=point,
        primary_ci=ci,
        corroborating_signal="format-conditional schema validity (reported separately)",
        corroborating_value=None,
        tier=classify_tier(value=abs_value, baseline=None, mode="absolute", ci=abs_ci),
        anchor_type="absolute",
        sonnet_baseline=None,
        delta_vs_sonnet=None,
        inconsistency_flag=False,
        notes="no baseline anchor",
    )


def _build_adaptation_row(by_dim: dict[str, list[int]]) -> CapabilityRow:
    vals = by_dim.get("Adaptation Specificity", [])
    point = _mean(vals)
    ci = _bootstrap_ci([float(v) for v in vals])
    return CapabilityRow(
        name="Adaptation",
        primary_signal="Adaptation Specificity (NEW)",
        primary_value=point,
        primary_ci=ci,
        corroborating_signal=None,
        corroborating_value=None,
        tier=classify_tier(value=point, baseline=None, mode="absolute", ci=ci),
        anchor_type="absolute",
        sonnet_baseline=None,
        delta_vs_sonnet=None,
        inconsistency_flag=False,
        notes="no baseline anchor",
    )


def _adjacent_tier(a: str, b: str) -> bool:
    order = ["absent", "mid_tier", "at_ceiling"]
    base_a = a.split("_with_")[0]
    base_b = b.split("_with_")[0]
    if base_a not in order or base_b not in order:
        return True
    return abs(order.index(base_a) - order.index(base_b)) <= 1


def _render_markdown(dossier: Dossier) -> str:
    lines = ["## Capability dossier", ""]
    meta = dossier.meta
    lines.append(f"- model: `{meta.get('model_id', 'unknown')}`")
    if "routed_providers" in meta:
        lines.append(f"- routed_providers: {meta['routed_providers']}")
    lines.append("")
    lines.append("| Capability | Tier | Primary signal | Value | vs Sonnet | CI | Note |")
    lines.append("|---|---|---|---|---|---|---|")
    for c in dossier.capabilities:
        ci_str = f"[{c.primary_ci[0]:.2f},{c.primary_ci[1]:.2f}]" if c.primary_ci else "n/a"
        delta_str = f"{c.delta_vs_sonnet:+.2f}" if c.delta_vs_sonnet is not None else "n/a"
        notes = c.notes
        if c.inconsistency_flag:
            notes = (notes + "; inconsistent primary/corroborator").strip("; ")
        lines.append(
            f"| {c.name} | {c.tier} | {c.primary_signal} | {c.primary_value:.2f} | {delta_str} | {ci_str} | {notes} |"
        )
    if dossier.continuation_degeneracy_rate is not None:
        lines.append("")
        lines.append(
            f"**Continuation degeneracy rate:** {dossier.continuation_degeneracy_rate:.1%}; "
            f"by category: {dossier.continuation_by_category}"
        )
    if dossier.over_call_by_category:
        lines.append("")
        lines.append(f"**Over-call rates by negative category:** {dossier.over_call_by_category}")
    return "\n".join(lines) + "\n"


def build_dossier(
    synthesis_jsonl: Path,
    tool_jsonl: Path,
    mcq_json: Path,
    baseline_aggregate_json: Path,
    out_dir: Path,
    continuation_jsonl: Path | None = None,
) -> Dossier:
    synth_rows = _read_jsonl(synthesis_jsonl)
    tool_rows = _read_jsonl(tool_jsonl)
    _check_error_rate(synth_rows, "synthesis")
    _check_error_rate(tool_rows, "tool")

    mcq = json.loads(mcq_json.read_text())
    baseline = json.loads(baseline_aggregate_json.read_text())

    by_dim = _synthesis_outcome_means(synth_rows)

    capabilities = [
        _build_judgment_row(by_dim, baseline),
        _build_taste_row(by_dim),
        _build_integration_row(by_dim, baseline),
        _build_voice_row(by_dim, baseline),
        _build_vocabulary_row(by_dim, baseline, mcq),
        _build_tool_calling_row(tool_rows),
        _build_adaptation_row(by_dim),
    ]

    routed = sorted({r.get("routed_provider") for r in synth_rows if r.get("routed_provider")})
    meta = {
        "model_id": next((r.get("model_id") for r in synth_rows if r.get("model_id")), "unknown"),
        "n_synthesis": len(synth_rows),
        "n_tool": len(tool_rows),
        "mcq_total": mcq.get("total"),
        "routed_providers": routed,
    }

    # Over-call by category (negatives only)
    neg_rows = [r for r in tool_rows if r.get("expected_call") is False]
    by_cat: dict[str, list[bool]] = {}
    for r in neg_rows:
        c = r.get("category") or "unknown"
        by_cat.setdefault(c, []).append(bool(r.get("called")))
    over_call = {k: sum(v) / len(v) for k, v in by_cat.items() if v}

    cont_rate: float | None = None
    cont_cat: dict[str, int] = {}
    if continuation_jsonl is not None and continuation_jsonl.exists():
        cont_rows = _read_jsonl(continuation_jsonl)
        if cont_rows:
            degen = sum(1 for r in cont_rows if r.get("is_degenerate"))
            cont_rate = degen / len(cont_rows)
            for r in cont_rows:
                cat = r.get("category", "unknown")
                cont_cat[cat] = cont_cat.get(cat, 0) + 1

    dossier = Dossier(
        meta=meta,
        capabilities=capabilities,
        continuation_degeneracy_rate=cont_rate,
        continuation_by_category=cont_cat,
        over_call_by_category=over_call,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "capability_dossier.json").write_text(
        json.dumps(
            {
                "meta": dossier.meta,
                "capabilities": [asdict(c) for c in dossier.capabilities],
                "continuation_degeneracy_rate": dossier.continuation_degeneracy_rate,
                "continuation_by_category": dossier.continuation_by_category,
                "over_call_by_category": dossier.over_call_by_category,
            },
            indent=2,
        )
    )
    (out_dir / "capability_dossier.md").write_text(_render_markdown(dossier))
    return dossier
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/stage0/tests/test_aggregator.py -x -q
```
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/stage0/aggregator.py apps/evals/teacher_model/stage0/tests/test_aggregator.py && git commit -m "feat(stage0): aggregator builds capability dossier with error-rate gate"
```

---

## Task 11: run_synthesis — Pipeline A runner

**Group:** C (depends on Task 6: judge_extended)

**Behavior being verified:** the runner consumes the holdout JSONL, calls a teacher LLM (DI-injected), calls the extended judge (DI-injected), writes a result row per recording, skips already-completed rows on re-invocation, and labels each row with `routed_provider` from the response metadata.

**Interface under test:** `run(holdout_path, out_path, teacher_client, judge_fn, *, system_prompt) -> RunStats`.

**Files:**
- Create: `apps/evals/teacher_model/stage0/run_synthesis.py`
- Test: `apps/evals/teacher_model/stage0/tests/test_run_synthesis.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/teacher_model/stage0/tests/test_run_synthesis.py
"""Pipeline A runner: shape, resume behaviour, judge invocation."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from teacher_model.stage0.run_synthesis import run as run_synthesis


class _FakeClient:
    def __init__(self) -> None:
        self.calls = 0
        self.model = "fake/qwen-x"

    def complete(self, user: str, system: str = "", max_tokens: int = 0) -> str:
        self.calls += 1
        return "<analysis>scratchpad</analysis>Real teacher response, warm and specific."


@dataclass
class _FakeDim:
    criterion: str
    process: int = 2
    outcome: int = 2
    score: int = 2
    evidence: str = ""
    reason: str = ""


@dataclass
class _FakeJudge:
    dimensions: list
    model: str = "fake/judge"
    prompt_version: str = "judge_v2_extended"
    latency_ms: float = 1.0


_DIMS = [
    "Audible-Specific Corrective Feedback",
    "Concrete Artifact Provision",
    "Specific Positive Praise",
    "Autonomy-Supporting Motivation",
    "Scaffolded Guided Discovery",
    "Style-Consistent Musical Language",
    "Appropriate Tone & Language",
    "Taste Defensibility",
    "Adaptation Specificity",
]


def _fake_judge_fn(synthesis_text, context, **kwargs):
    return _FakeJudge(dimensions=[_FakeDim(criterion=d) for d in _DIMS])


def _write_holdout(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rid in ids:
            f.write(
                json.dumps(
                    {
                        "recording_id": rid,
                        "era": "Romantic",
                        "skill_bucket": 3,
                        "stratum": "Romantic|sk3",
                        "composer": "Chopin",
                        "piece_slug": "test_piece",
                        "title": "Test Piece",
                        "briefing_path": str(path.parent / f"{rid}.json"),
                    }
                )
                + "\n"
            )
            (path.parent / f"{rid}.json").write_text(
                json.dumps(
                    {
                        "recording_id": rid,
                        "chunks": [
                            {
                                "predictions": {
                                    "dynamics": 0.5, "timing": 0.5, "pedaling": 0.6,
                                    "articulation": 0.5, "phrasing": 0.5, "interpretation": 0.5,
                                }
                            }
                        ],
                        "total_duration_seconds": 600.0,
                    }
                )
            )


def test_run_writes_one_row_per_holdout_with_nine_dims(tmp_path: Path) -> None:
    holdout = tmp_path / "holdout.jsonl"
    out = tmp_path / "synth.jsonl"
    _write_holdout(holdout, ["r1", "r2", "r3"])
    client = _FakeClient()
    stats = run_synthesis(
        holdout_path=holdout, out_path=out,
        teacher_client=client, judge_fn=_fake_judge_fn,
        system_prompt="You are a teacher.",
    )
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 3
    for row in rows:
        assert len(row["judge_dimensions"]) == 9
        assert row["error"] == ""
        assert row["model_id"] == "fake/qwen-x"
    assert stats.processed == 3 and stats.errors == 0


def test_run_resumes_skipping_completed_ids(tmp_path: Path) -> None:
    holdout = tmp_path / "holdout.jsonl"
    out = tmp_path / "synth.jsonl"
    _write_holdout(holdout, ["r1", "r2"])
    client = _FakeClient()
    run_synthesis(
        holdout_path=holdout, out_path=out,
        teacher_client=client, judge_fn=_fake_judge_fn, system_prompt="x",
    )
    assert client.calls == 2

    # Add a third id; re-run should only process the new one.
    _write_holdout(holdout, ["r1", "r2", "r3"])
    client2 = _FakeClient()
    stats = run_synthesis(
        holdout_path=holdout, out_path=out,
        teacher_client=client2, judge_fn=_fake_judge_fn, system_prompt="x",
    )
    assert client2.calls == 1
    assert stats.processed == 1
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/stage0/tests/test_run_synthesis.py -x -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'teacher_model.stage0.run_synthesis'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `apps/evals/teacher_model/stage0/run_synthesis.py`:

```python
"""Pipeline A: run base teacher LLM on the n=100 holdout, judge each response.

Resumable: rows already present in out_path (by recording_id with no error)
are skipped on subsequent invocations.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from teaching_knowledge.run_eval import (
    aggregate_muq,
    build_synthesis_user_msg,
    extract_teacher_response,
)


class _Client(Protocol):
    model: str

    def complete(self, user: str, system: str, max_tokens: int) -> str: ...


JudgeFn = Callable[..., Any]  # (synthesis_text, context, **kwargs) -> JudgeResultV2Extended


@dataclass
class RunStats:
    processed: int
    errors: int
    skipped: int


def _load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            if row.get("recording_id") and not row.get("error"):
                done.add(row["recording_id"])
        except json.JSONDecodeError:
            continue
    return done


def run(
    holdout_path: Path,
    out_path: Path,
    teacher_client: _Client,
    judge_fn: JudgeFn,
    *,
    system_prompt: str,
    judge_provider: str = "workers-ai",
    judge_model: str | None = "@cf/google/gemma-4-26b-a4b-it",
    max_tokens: int = 1024,
) -> RunStats:
    holdout = [
        json.loads(line) for line in holdout_path.read_text().splitlines() if line.strip()
    ]
    completed = _load_completed(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    errors = 0
    skipped = 0

    with out_path.open("a") as fout:
        for entry in holdout:
            rid = entry["recording_id"]
            if rid in completed:
                skipped += 1
                continue

            briefing = json.loads(Path(entry["briefing_path"]).read_text())
            chunks = briefing.get("chunks", [])
            muq_means = aggregate_muq(chunks) if chunks else {}
            duration = float(briefing.get("total_duration_seconds", 0.0))
            meta = {
                "title": entry.get("title", ""),
                "composer": entry.get("composer", ""),
                "skill_bucket": int(entry.get("skill_bucket", 3)),
            }

            row: dict = {
                "recording_id": rid,
                "model_id": getattr(teacher_client, "model", ""),
                "stratum": entry.get("stratum"),
                "era": entry.get("era"),
                "skill_bucket": entry.get("skill_bucket"),
                "synthesis_text": "",
                "judge_dimensions": [],
                "judge_model": "",
                "synthesis_latency_ms": 0,
                "judge_latency_ms": 0,
                "routed_provider": "",
                "error": "",
            }

            try:
                user_msg = build_synthesis_user_msg(muq_means, duration, meta)
                t0 = time.monotonic()
                raw = teacher_client.complete(user=user_msg, system=system_prompt, max_tokens=max_tokens)
                row["synthesis_latency_ms"] = round((time.monotonic() - t0) * 1000)
                row["synthesis_text"] = extract_teacher_response(raw)
                row["routed_provider"] = getattr(teacher_client, "last_routed_provider", "") or ""

                judge_ctx = {
                    "piece_name": meta["title"],
                    "composer": meta["composer"],
                    "skill_level": meta["skill_bucket"],
                }
                jres = judge_fn(
                    row["synthesis_text"],
                    judge_ctx,
                    provider=judge_provider,
                    model=judge_model,
                )
                row["judge_dimensions"] = [
                    {
                        "criterion": d.criterion,
                        "process": getattr(d, "process", None),
                        "outcome": getattr(d, "outcome", None),
                        "score": getattr(d, "score", None),
                        "evidence": getattr(d, "evidence", ""),
                        "reason": getattr(d, "reason", ""),
                    }
                    for d in jres.dimensions
                ]
                row["judge_model"] = getattr(jres, "model", "")
                row["judge_latency_ms"] = round(getattr(jres, "latency_ms", 0.0))
                processed += 1
            except Exception as exc:  # bubbled up after recording the failure
                row["error"] = str(exc)[:500]
                errors += 1

            fout.write(json.dumps(row) + "\n")
            fout.flush()

    return RunStats(processed=processed, errors=errors, skipped=skipped)
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/stage0/tests/test_run_synthesis.py -x -q
```
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/stage0/run_synthesis.py apps/evals/teacher_model/stage0/tests/test_run_synthesis.py && git commit -m "feat(stage0): run_synthesis pipeline A with DI client/judge and resume"
```

---

## Task 12: run_tool_probe — Pipeline B runner

**Group:** C (depends on Task 5: tool_scorer, Task 8: cases loader)

**Behavior being verified:** the runner reads tool cases, prompts the teacher client with the tool-probe system prompt, scores each response, writes one result row per case (including category for negatives), and resumes when re-invoked.

**Interface under test:** `run(cases_path, system_prompt_path, schemas, teacher_client, out_path) -> RunStats`.

**Files:**
- Create: `apps/evals/teacher_model/stage0/run_tool_probe.py`
- Create: `apps/evals/teacher_model/stage0/prompts/tool_probe_system.txt`
- Test: `apps/evals/teacher_model/stage0/tests/test_run_tool_probe.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/teacher_model/stage0/tests/test_run_tool_probe.py
"""Pipeline B runner: scoring + per-row category propagation + resume."""
from __future__ import annotations

import json
from pathlib import Path

from teacher_model.stage0.run_tool_probe import run as run_tool_probe

_SCHEMAS = {
    "search_catalog": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
    "create_exercise": {
        "type": "object",
        "properties": {"skill": {"type": "string"}, "exercises": {"type": "array"}},
        "required": ["skill", "exercises"],
    },
}


class _ScriptedClient:
    def __init__(self, script: dict[str, str]) -> None:
        self.script = script
        self.model = "fake/qwen"
        self.calls = 0

    def complete(self, user: str, system: str = "", max_tokens: int = 0) -> str:
        self.calls += 1
        # The user prompt embeds the case_id in <case_id>...</case_id>.
        import re
        m = re.search(r"<case_id>(.+?)</case_id>", user)
        if not m:
            return ""
        return self.script.get(m.group(1), "")


def _write_cases(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"case_id": "p_search_01", "expected_call": True, "expected_tool": "search_catalog", "category": None,
         "briefing": {"prompt": "student: 'find me Chopin pieces'"}},
        {"case_id": "n_chitchat_01", "expected_call": False, "expected_tool": None, "category": "chitchat",
         "briefing": {"prompt": "student: 'thanks!'"}},
    ]
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_runner_scores_each_case_and_propagates_category(tmp_path: Path) -> None:
    cases = tmp_path / "cases.jsonl"
    _write_cases(cases)
    sys_prompt = tmp_path / "tool_probe_system.txt"
    sys_prompt.write_text("system\n<schemas/>")
    client = _ScriptedClient({
        "p_search_01": '<tool_call>{"name":"search_catalog","arguments":{"query":"Chopin"}}</tool_call>',
        "n_chitchat_01": "You're welcome!",
    })
    out = tmp_path / "tool_runs.jsonl"
    stats = run_tool_probe(
        cases_path=cases, system_prompt_path=sys_prompt,
        schemas=_SCHEMAS, teacher_client=client, out_path=out,
    )
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 2
    p = next(r for r in rows if r["case_id"] == "p_search_01")
    n = next(r for r in rows if r["case_id"] == "n_chitchat_01")
    assert p["called"] is True and p["discipline_correct"] is True
    assert n["called"] is False and n["category"] == "chitchat" and n["discipline_correct"] is True
    assert stats.processed == 2 and stats.errors == 0


def test_runner_resumes_skipping_completed(tmp_path: Path) -> None:
    cases = tmp_path / "cases.jsonl"
    _write_cases(cases)
    sys_prompt = tmp_path / "tool_probe_system.txt"
    sys_prompt.write_text("x")
    client = _ScriptedClient({
        "p_search_01": '<tool_call>{"name":"search_catalog","arguments":{"query":"Chopin"}}</tool_call>',
        "n_chitchat_01": "ok",
    })
    out = tmp_path / "tool_runs.jsonl"
    run_tool_probe(cases_path=cases, system_prompt_path=sys_prompt, schemas=_SCHEMAS,
                   teacher_client=client, out_path=out)
    assert client.calls == 2
    stats = run_tool_probe(cases_path=cases, system_prompt_path=sys_prompt, schemas=_SCHEMAS,
                          teacher_client=_ScriptedClient({}), out_path=out)
    assert stats.processed == 0 and stats.skipped == 2
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/stage0/tests/test_run_tool_probe.py -x -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'teacher_model.stage0.run_tool_probe'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `apps/evals/teacher_model/stage0/prompts/tool_probe_system.txt`:

```
You are CrescendAI's teacher model. Given a student briefing, you may call one of these tools by emitting a `<tool_call>{"name": "...", "arguments": {...}}</tool_call>` block. If the right thing is to just listen, ask a question, or respond conversationally, do NOT call any tool — answer in plain prose.

Available tools:

- create_exercise(skill: string, exercises: array)
- score_highlight(highlight_id: string?, bars: int[2])
- keyboard_guide(label: string)
- show_session_data()
- reference_browser(query: string)
- search_catalog(query: string)

Examples:

Student briefing: "I keep rushing in the LH at bar 12; what should I practice?"
Right move: call create_exercise.
Output: <tool_call>{"name":"create_exercise","arguments":{"skill":"left-hand timing","exercises":[{"title":"slow LH alone","instruction":"play the LH at quarter=60"}]}}</tool_call>

Student briefing: "Thanks for the suggestion last week."
Right move: do not call a tool.
Output: You're welcome — keep me posted on how the slow practice felt.

Student briefing: "Could you point me at a reference recording for this Ballade?"
Right move: call reference_browser.
Output: <tool_call>{"name":"reference_browser","arguments":{"query":"Chopin Ballade No. 1 reference recording"}}</tool_call>

Now respond to the briefing below. The briefing case id is wrapped in <case_id> tags for traceability.
```

Create `apps/evals/teacher_model/stage0/run_tool_probe.py`:

```python
"""Pipeline B: run base teacher LLM on tool-probe cases, score each response."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from teacher_model.stage0.cases import load_cases
from teacher_model.stage0.tool_scorer import ToolCase as ScorerCase, score_response


class _Client(Protocol):
    model: str

    def complete(self, user: str, system: str, max_tokens: int) -> str: ...


@dataclass
class RunStats:
    processed: int
    errors: int
    skipped: int


def _load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            if row.get("case_id") and not row.get("error"):
                done.add(row["case_id"])
        except json.JSONDecodeError:
            continue
    return done


def run(
    cases_path: Path,
    system_prompt_path: Path,
    schemas: dict,
    teacher_client: _Client,
    out_path: Path,
    max_tokens: int = 800,
) -> RunStats:
    cases = load_cases(cases_path)
    system_prompt = system_prompt_path.read_text()
    completed = _load_completed(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    errors = 0
    skipped = 0

    with out_path.open("a") as fout:
        for c in cases:
            if c.case_id in completed:
                skipped += 1
                continue
            user = (
                f"<case_id>{c.case_id}</case_id>\n"
                f"Briefing:\n{json.dumps(c.briefing, indent=2)}"
            )
            row: dict = {
                "case_id": c.case_id,
                "expected_call": c.expected_call,
                "expected_tool": c.expected_tool,
                "category": c.category,
                "called": False,
                "tool_name": None,
                "arguments": None,
                "discipline_correct": False,
                "format_valid": None,
                "extraction_format": None,
                "raw_response": "",
                "model_id": getattr(teacher_client, "model", ""),
                "latency_ms": 0,
                "error": "",
            }
            try:
                t0 = time.monotonic()
                raw = teacher_client.complete(user=user, system=system_prompt, max_tokens=max_tokens)
                row["latency_ms"] = round((time.monotonic() - t0) * 1000)
                row["raw_response"] = raw
                result = score_response(
                    raw,
                    ScorerCase(
                        case_id=c.case_id,
                        expected_call=c.expected_call,
                        expected_tool=c.expected_tool,
                        category=c.category,
                    ),
                    schemas,
                )
                row.update({
                    "called": result.called,
                    "tool_name": result.tool_name,
                    "arguments": result.arguments,
                    "discipline_correct": result.discipline_correct,
                    "format_valid": result.format_valid,
                    "extraction_format": result.extraction_format,
                })
                processed += 1
            except Exception as exc:
                row["error"] = str(exc)[:500]
                errors += 1
            fout.write(json.dumps(row) + "\n")
            fout.flush()

    return RunStats(processed=processed, errors=errors, skipped=skipped)
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/stage0/tests/test_run_tool_probe.py -x -q
```
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/stage0/run_tool_probe.py apps/evals/teacher_model/stage0/prompts/tool_probe_system.txt apps/evals/teacher_model/stage0/tests/test_run_tool_probe.py && git commit -m "feat(stage0): run_tool_probe pipeline B with cases loader + scorer"
```

---

## Task 13: run_continuation — Pipeline B+ runner

**Group:** C (depends on Task 4: continuation_probe, Task 5: tool_scorer)

**Behavior being verified:** for each successful tool call from the tool-probe results (positive cases only), inject a synthetic tool_result, re-prompt the model, classify the continuation, and write one row per replayed case.

**Interface under test:** `run(tool_runs_path, cases_path, teacher_client, out_path) -> RunStats`.

**Files:**
- Create: `apps/evals/teacher_model/stage0/run_continuation.py`
- Test: `apps/evals/teacher_model/stage0/tests/test_run_continuation.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/teacher_model/stage0/tests/test_run_continuation.py
"""Pipeline B+ runner: replays positive successful tool calls, classifies follow-up."""
from __future__ import annotations

import json
from pathlib import Path

from teacher_model.stage0.run_continuation import run as run_continuation


class _ScriptedClient:
    def __init__(self, replies: dict[str, str]) -> None:
        self.replies = replies
        self.model = "fake/qwen"

    def complete(self, user: str, system: str = "", max_tokens: int = 0) -> str:
        import re
        m = re.search(r"<case_id>(.+?)</case_id>", user)
        return self.replies.get(m.group(1) if m else "", "")


def _write_cases(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"case_id": "p_search_01", "expected_call": True, "expected_tool": "search_catalog",
         "category": None, "briefing": {"prompt": "find Chopin"}},
        {"case_id": "p_search_02", "expected_call": True, "expected_tool": "search_catalog",
         "category": None, "briefing": {"prompt": "find Liszt"}},
    ]
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _write_tool_runs(path: Path) -> None:
    rows = [
        {"case_id": "p_search_01", "expected_call": True, "called": True, "tool_name": "search_catalog",
         "arguments": {"query": "Chopin"}, "discipline_correct": True, "format_valid": True,
         "raw_response": '<tool_call>{"name":"search_catalog","arguments":{"query":"Chopin"}}</tool_call>',
         "category": None, "error": ""},
        {"case_id": "p_search_02", "expected_call": True, "called": False, "tool_name": None,
         "arguments": None, "discipline_correct": False, "format_valid": None,
         "raw_response": "I don't know.", "category": None, "error": ""},
    ]
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_only_replays_successful_positive_calls(tmp_path: Path) -> None:
    cases = tmp_path / "cases.jsonl"
    runs = tmp_path / "tool_runs.jsonl"
    out = tmp_path / "cont.jsonl"
    _write_cases(cases)
    _write_tool_runs(runs)
    client = _ScriptedClient({
        "p_search_01": "Great — try Chopin's Ballade No. 1, focusing on the second theme around bar 68 for that singing tone.",
    })
    stats = run_continuation(
        tool_runs_path=runs, cases_path=cases,
        teacher_client=client, out_path=out,
    )
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["case_id"] == "p_search_01"
    assert rows[0]["category"] == "clean"
    assert rows[0]["is_degenerate"] is False
    assert stats.processed == 1


def test_classifies_refusal(tmp_path: Path) -> None:
    cases = tmp_path / "cases.jsonl"
    runs = tmp_path / "tool_runs.jsonl"
    out = tmp_path / "cont.jsonl"
    _write_cases(cases)
    _write_tool_runs(runs)
    client = _ScriptedClient({"p_search_01": "I cannot continue. I am unable to help with this request."})
    run_continuation(
        tool_runs_path=runs, cases_path=cases,
        teacher_client=client, out_path=out,
    )
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert rows[0]["category"] == "refusal"
    assert rows[0]["is_degenerate"] is True
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/stage0/tests/test_run_continuation.py -x -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'teacher_model.stage0.run_continuation'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `apps/evals/teacher_model/stage0/run_continuation.py`:

```python
"""Pipeline B+: replay positive successful tool calls with a synthetic tool_result.

For each row in tool_runs.jsonl where expected_call is True AND called is True
AND discipline_correct is True, build a follow-up turn where:
  1. The original assistant turn (with tool_call) is included.
  2. A synthetic tool_result message is appended (from continuation_fixtures.json).
  3. The model is asked to continue.

The follow-up response is classified by score_continuation; one row per replay
is written to out_path with {case_id, category, is_degenerate, detail}.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from teacher_model.stage0.cases import load_cases
from teacher_model.stage0.continuation_probe import (
    load_tool_result_fixture,
    score_continuation,
)


class _Client(Protocol):
    model: str

    def complete(self, user: str, system: str, max_tokens: int) -> str: ...


@dataclass
class RunStats:
    processed: int
    errors: int


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def run(
    tool_runs_path: Path,
    cases_path: Path,
    teacher_client: _Client,
    out_path: Path,
    max_tokens: int = 600,
) -> RunStats:
    tool_runs = _read_jsonl(tool_runs_path)
    cases = {c.case_id: c for c in load_cases(cases_path)}
    out_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    errors = 0

    with out_path.open("w") as fout:
        for tr in tool_runs:
            if not (tr.get("expected_call") and tr.get("called") and tr.get("discipline_correct")):
                continue
            case = cases.get(tr["case_id"])
            if case is None:
                continue
            tool_name = tr.get("tool_name") or ""
            try:
                fixture = load_tool_result_fixture(tool_name)
            except KeyError:
                continue

            initial_assistant = tr.get("raw_response", "")
            user = (
                f"<case_id>{case.case_id}</case_id>\n"
                f"Original briefing:\n{json.dumps(case.briefing, indent=2)}\n\n"
                f"You called: {initial_assistant}\n\n"
                f"<tool_result tool=\"{tool_name}\">{json.dumps(fixture)}</tool_result>\n\n"
                f"Continue with your teacher response, integrating the tool result for the student."
            )
            row: dict = {
                "case_id": case.case_id,
                "tool_name": tool_name,
                "category": "",
                "is_degenerate": False,
                "detail": "",
                "follow_up_response": "",
                "model_id": getattr(teacher_client, "model", ""),
                "error": "",
            }
            try:
                follow_up = teacher_client.complete(user=user, system="", max_tokens=max_tokens)
                row["follow_up_response"] = follow_up
                result = score_continuation(initial_assistant, fixture, follow_up)
                row["category"] = result.category
                row["is_degenerate"] = result.is_degenerate
                row["detail"] = result.detail
                processed += 1
            except Exception as exc:
                row["error"] = str(exc)[:500]
                errors += 1
            fout.write(json.dumps(row) + "\n")

    return RunStats(processed=processed, errors=errors)
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/stage0/tests/test_run_continuation.py -x -q
```
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/stage0/run_continuation.py apps/evals/teacher_model/stage0/tests/test_run_continuation.py && git commit -m "feat(stage0): run_continuation pipeline B+ replays successful tool calls with synthetic tool_result"
```

---

## Task 14: cli — wire all subcommands

**Group:** D (depends on Groups A, B, C)

**Behavior being verified:** invoking `python -m teacher_model.stage0 --help` lists exactly the 7 subcommands (`pin-tokenizer`, `sample`, `synthesis`, `tool`, `continuation`, `mcq`, `aggregate`); the `sample` subcommand produces the expected output file.

**Interface under test:** the CLI `__main__` (subprocess invocation) and `_build_parser()` exposed for testing.

**Files:**
- Create: `apps/evals/teacher_model/stage0/cli.py`
- Create: `apps/evals/teacher_model/stage0/__main__.py`
- Test: `apps/evals/teacher_model/stage0/tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/teacher_model/stage0/tests/test_cli.py
"""CLI shape: subcommand registry + sample subcommand happy path."""
from __future__ import annotations

import json
from pathlib import Path

from teacher_model.stage0.cli import _build_parser, sample_main


def test_parser_lists_seven_subcommands() -> None:
    parser = _build_parser()
    # Pull subparser choices via a bit of introspection.
    subparsers_action = next(
        a for a in parser._subparsers._group_actions  # type: ignore[attr-defined]
        if a.__class__.__name__ == "_SubParsersAction"
    )
    assert set(subparsers_action.choices.keys()) == {
        "pin-tokenizer", "sample", "synthesis", "tool",
        "continuation", "mcq", "aggregate",
    }


def test_sample_subcommand_writes_holdout_file(tmp_path: Path, monkeypatch) -> None:
    # Build a tiny fake briefings dir + manifest.
    briefings = tmp_path / "briefings"
    briefings.mkdir()
    for rid, comp, sk in [
        ("r1", "Chopin", 3),
        ("r2", "Bach", 2),
        ("r3", "Mozart", 4),
        ("r4", "Debussy", 5),
        ("r5", "Chopin", 1),
        ("r6", "Bach", 4),
    ]:
        (briefings / f"{rid}.json").write_text(json.dumps({"recording_id": rid}))

    fake_manifests = {
        "r1": {"piece_slug": "p1", "title": "p1", "composer": "Chopin", "skill_bucket": 3},
        "r2": {"piece_slug": "p2", "title": "p2", "composer": "Bach", "skill_bucket": 2},
        "r3": {"piece_slug": "p3", "title": "p3", "composer": "Mozart", "skill_bucket": 4},
        "r4": {"piece_slug": "p4", "title": "p4", "composer": "Debussy", "skill_bucket": 5},
        "r5": {"piece_slug": "p5", "title": "p5", "composer": "Chopin", "skill_bucket": 1},
        "r6": {"piece_slug": "p6", "title": "p6", "composer": "Bach", "skill_bucket": 4},
    }

    out = tmp_path / "holdout.jsonl"
    sample_main(briefings_dir=briefings, manifests=fake_manifests, n=4, seed=42, out_path=out)
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert 1 <= len(rows) <= 6
    for r in rows:
        assert "recording_id" in r and "stratum" in r
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/stage0/tests/test_cli.py -x -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'teacher_model.stage0.cli'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `apps/evals/teacher_model/stage0/cli.py`:

```python
"""Stage 0 CLI: pin-tokenizer / sample / synthesis / tool / continuation / mcq / aggregate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from teacher_model.stage0.aggregator import build_dossier
from teacher_model.stage0.judge_extended import judge_extended
from teacher_model.stage0.pin_tokenizer import pin_tokenizer
from teacher_model.stage0.run_continuation import run as run_continuation
from teacher_model.stage0.run_synthesis import run as run_synthesis
from teacher_model.stage0.run_tool_probe import run as run_tool_probe
from teacher_model.stage0.sampler import sample_holdout, write_holdout

_STAGE0_ROOT = Path(__file__).parent
_DATA_DIR = _STAGE0_ROOT / "data"
_RESULTS_DIR = _STAGE0_ROOT / "results"
_PROMPTS_DIR = _STAGE0_ROOT / "prompts"
_REPO_ROOT = _STAGE0_ROOT.resolve().parents[4]
_BRIEFINGS_DIR = _REPO_ROOT / "model" / "data" / "eval" / "inference_cache" / "auto-t5_http"
_SYNTH_SYSTEM = _REPO_ROOT / "apps" / "shared" / "teacher-style" / "synthesis_system.txt"
_BASELINE_AGGREGATE = _REPO_ROOT / "apps" / "evals" / "results" / "baseline_v1_aggregate.json"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="teacher_model.stage0")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("pin-tokenizer", help="Download + hash the Qwen tokenizer")
    sp.add_argument("--model", required=True)
    sp.add_argument("--out", type=Path, default=_RESULTS_DIR / "tokenizer_pin.json")

    sp = sub.add_parser("sample", help="Stratified holdout sampling")
    sp.add_argument("--n", type=int, default=100)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--out", type=Path, default=_DATA_DIR / "stage0_holdout.jsonl")

    sp = sub.add_parser("synthesis", help="Pipeline A: synthesis eval")
    sp.add_argument("--provider", default="openrouter")
    sp.add_argument("--model", required=True)
    sp.add_argument("--judge-provider", default="workers-ai")
    sp.add_argument("--judge-model", default="@cf/google/gemma-4-26b-a4b-it")
    sp.add_argument("--out", type=Path, default=_RESULTS_DIR / "synthesis_runs.jsonl")
    sp.add_argument("--limit", type=int, default=None)

    sp = sub.add_parser("tool", help="Pipeline B: tool-call probe")
    sp.add_argument("--provider", default="openrouter")
    sp.add_argument("--model", required=True)
    sp.add_argument("--out", type=Path, default=_RESULTS_DIR / "tool_runs.jsonl")

    sp = sub.add_parser("continuation", help="Pipeline B+: post-tool-result continuation probe")
    sp.add_argument("--provider", default="openrouter")
    sp.add_argument("--model", required=True)
    sp.add_argument("--tool-runs", type=Path, default=_RESULTS_DIR / "tool_runs.jsonl")
    sp.add_argument("--out", type=Path, default=_RESULTS_DIR / "continuation_runs.jsonl")

    sp = sub.add_parser("mcq", help="Pipeline C: existing 50-Q MCQ")
    sp.add_argument("--provider", default="openrouter")
    sp.add_argument("--model", required=True)
    sp.add_argument("--out", type=Path, default=_RESULTS_DIR / "mcq_summary.json")

    sp = sub.add_parser("aggregate", help="Build the capability dossier")
    sp.add_argument("--out-dir", type=Path, default=_RESULTS_DIR)

    return p


def sample_main(*, briefings_dir: Path, manifests: dict, n: int, seed: int, out_path: Path) -> None:
    rows = sample_holdout(briefings_dir, manifests, n=n, seed=seed)
    write_holdout(rows, out_path)


def _load_tool_schemas() -> dict:
    """Mirror of the 6-tool palette from apps/api/src/services/tool-processor.ts."""
    return {
        "create_exercise": {
            "type": "object",
            "properties": {
                "skill": {"type": "string"},
                "exercises": {"type": "array"},
            },
            "required": ["skill", "exercises"],
        },
        "score_highlight": {
            "type": "object",
            "properties": {
                "highlight_id": {"type": "string"},
                "bars": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
            },
            "required": ["bars"],
        },
        "keyboard_guide": {
            "type": "object",
            "properties": {"label": {"type": "string"}},
            "required": ["label"],
        },
        "show_session_data": {
            "type": "object",
            "properties": {},
            "additionalProperties": True,
        },
        "reference_browser": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        "search_catalog": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    }


def main() -> None:
    args = _build_parser().parse_args()

    if args.cmd == "pin-tokenizer":
        pin_tokenizer(model_id=args.model, out_path=args.out)
        return

    if args.cmd == "sample":
        from teaching_knowledge.run_eval import load_manifests
        manifests = load_manifests()
        sample_main(briefings_dir=_BRIEFINGS_DIR, manifests=manifests, n=args.n, seed=args.seed, out_path=args.out)
        return

    if args.cmd == "synthesis":
        from teaching_knowledge.llm_client import LLMClient
        client = LLMClient(provider=args.provider, model=args.model)
        run_synthesis(
            holdout_path=_DATA_DIR / "stage0_holdout.jsonl",
            out_path=args.out,
            teacher_client=client,
            judge_fn=judge_extended,
            system_prompt=_SYNTH_SYSTEM.read_text(),
            judge_provider=args.judge_provider,
            judge_model=args.judge_model,
        )
        return

    if args.cmd == "tool":
        from teaching_knowledge.llm_client import LLMClient
        client = LLMClient(provider=args.provider, model=args.model)
        run_tool_probe(
            cases_path=_DATA_DIR / "tool_probe_cases.jsonl",
            system_prompt_path=_PROMPTS_DIR / "tool_probe_system.txt",
            schemas=_load_tool_schemas(),
            teacher_client=client,
            out_path=args.out,
        )
        return

    if args.cmd == "continuation":
        from teaching_knowledge.llm_client import LLMClient
        client = LLMClient(provider=args.provider, model=args.model)
        run_continuation(
            tool_runs_path=args.tool_runs,
            cases_path=_DATA_DIR / "tool_probe_cases.jsonl",
            teacher_client=client,
            out_path=args.out,
        )
        return

    if args.cmd == "mcq":
        import subprocess, sys
        cmd = [
            sys.executable, "-m", "teacher_model.domain_knowledge_probe",
            "--provider", args.provider, "--model", args.model,
            "--output", str(args.out),
        ]
        subprocess.run(cmd, check=True)
        return

    if args.cmd == "aggregate":
        build_dossier(
            synthesis_jsonl=_RESULTS_DIR / "synthesis_runs.jsonl",
            tool_jsonl=_RESULTS_DIR / "tool_runs.jsonl",
            mcq_json=_RESULTS_DIR / "mcq_summary.json",
            baseline_aggregate_json=_BASELINE_AGGREGATE,
            out_dir=args.out_dir,
            continuation_jsonl=_RESULTS_DIR / "continuation_runs.jsonl"
              if (_RESULTS_DIR / "continuation_runs.jsonl").exists() else None,
        )
        return
```

Create `apps/evals/teacher_model/stage0/__main__.py`:

```python
from teacher_model.stage0.cli import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/stage0/tests/test_cli.py -x -q
```
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/stage0/cli.py apps/evals/teacher_model/stage0/__main__.py apps/evals/teacher_model/stage0/tests/test_cli.py && git commit -m "feat(stage0): cli wires pin-tokenizer/sample/synthesis/tool/continuation/mcq/aggregate"
```

---

## Plan self-review

**Spec coverage:**
- Synthesis probe (n=100, era×skill stratified) → Tasks 6, 7, 11
- Tool probe (40 cases, 6 categories) → Tasks 5, 8, 12
- Continuation probe → Tasks 4, 13
- MCQ runs as-is via openrouter → Task 1
- Aggregator with Sonnet-anchored + absolute thresholds + error-rate gate + per-category over-call + continuation degeneracy → Task 10
- Tier classifier with CI overlap → Task 2
- Tokenizer pinning with chat-template-required → Task 3
- CLI subcommands → Task 14
- pyproject extras for `transformers` + `jsonschema` → Task 9
- `domain_knowledge_probe.py` — add `openrouter` to choices → Task 1

Every spec requirement maps to at least one task.

**Placeholder scan:** none — every step has exact code or commands. Curation directive in Task 8 is a real, specific instruction (not "TBD").

**Type/name consistency:** `ToolCase` in `cases.py` and `ToolCase` in `tool_scorer.py` are intentionally distinct types with the same name — Task 12 explicitly imports the scorer one as `ScorerCase` and constructs from the loader one. `ToolProbeResult`, `RunStats`, `Dossier`, `CapabilityRow`, `TokenizerPin`, `ContinuationResult` are each defined in exactly one module.

**Group correctness:**
- Group A: 9 tasks, all touching disjoint files. ✓
- Group B (Task 10): only depends on Task 2's `tier_classifier.py` import. ✓
- Group C: 3 tasks, all in different files; depend only on completed Group A modules. ✓
- Group D (Task 14): touches only `cli.py` + `__main__.py`; depends on every prior module via import only. ✓

**Vertical slice check:** Every task = exactly 1 test file → 1 implementation → 1 commit. No task bundles multiple tests before any impl. ✓

**Behavior test check:**
- All tests exercise public functions; no private-method or internal-state assertions.
- Where LLM clients appear, they are injected at the boundary (Tasks 11, 12, 13) — that is DI at a system boundary (HTTP), not mocking an internal collaborator.
- Curation/data tests (Task 8) verify file shape via the public `load_cases` interface, not raw text inspection.

**Commit:**

```bash
git add docs/plans/2026-05-07-stage0-capability-probe.md && git commit -m "docs(plan): add stage0-capability-probe TDD implementation plan"
```
