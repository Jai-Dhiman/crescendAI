# Eval Baseline Readiness Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until `/challenge` returns `VERDICT: PROCEED`.

**Goal:** Land every eval-harness improvement that can be built without valid MuQ scores, so that when Model v2 training completes and the cache is refreshed, one command produces a locked Sonnet 4.6 baseline artifact with per-dim means, bootstrap CIs, provenance, and a holdout that was never touched during prompt iteration.

**Spec:** `docs/specs/2026-04-14-eval-baseline-readiness-design.md`

**Style:** Follow `CLAUDE.md` + `apps/api/TS_STYLE.md`. Python: explicit exception handling, `uv` for package management, `from __future__ import annotations`, `pathlib.Path` for filesystem. TypeScript: no `c.env` destructuring, ServiceContext for DI. No emojis.

**Test runner:** `uv run pytest` from `apps/evals/`. All new tests go under `apps/evals/tests/`.

---

## Task Groups

```
Group A (parallel, no deps): T1, T2, T3, T4, T5, T6, T7
Group B (parallel, depends on A): T8, T9
Group C (parallel, depends on B): T10, T11
Group D (sequential, depends on A+B): T12
Group E (sequential, depends on D): T13
Group F (sequential, SAME FILE run_eval.py, depends on A+B+C): T14, T15, T16, T17
Group G (parallel, depends on E + A): T18, T19
```

Rationale: Group A is 7 independent foundation tasks that can all run in parallel — each touches a new file with no cross-dependencies. Group B layers on top of A (tag_dataset needs style_rules, api-mirror needs style_rules). Group C branches: split.py builds on tag_dataset's dataclass, prompts.ts builds on the api mirror. Group D (aggregate) needs both the stats extract and the tag_dataset dataclass. Group E (regression_check) needs aggregate's types. Group F is four serialized edits to `run_eval.py` — they cannot parallelize because they touch the same file. Group G wraps up the analysis tooling.

---

## Task List

### Task T1: Extract bootstrap_ci and cohens_d to shared/stats.py
**Group:** A (parallel)

**Behavior being verified:** `shared.stats.bootstrap_ci` returns a finite (low, high) tuple that contains the sample mean for a well-behaved input, and returns `None` for small samples (N<5). `shared.stats.cohens_d` computes the same value the existing `analyze_e2e.cohens_d` produces, via the new import path.

**Interface under test:** `shared.stats.bootstrap_ci(values, n_bootstrap, seed) -> tuple[float, float] | None` and `shared.stats.cohens_d(group1, group2) -> float`.

**Files:**
- Create: `apps/evals/shared/stats.py`
- Modify: `apps/evals/pipeline/practice_eval/analyze_e2e.py` (re-export from new location)
- Test: `apps/evals/tests/test_stats.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/tests/test_stats.py
from __future__ import annotations

import pytest

from shared.stats import bootstrap_ci, cohens_d


def test_bootstrap_ci_contains_sample_mean_for_normal_data() -> None:
    values = [0.4, 0.5, 0.5, 0.5, 0.5, 0.6, 0.55, 0.45, 0.52, 0.48]
    mean = sum(values) / len(values)
    ci = bootstrap_ci(values, n_bootstrap=1000, seed=42)
    assert ci is not None
    low, high = ci
    assert low < mean < high
    assert high - low < 0.2


def test_bootstrap_ci_is_deterministic_for_same_seed() -> None:
    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    ci_a = bootstrap_ci(values, n_bootstrap=500, seed=123)
    ci_b = bootstrap_ci(values, n_bootstrap=500, seed=123)
    assert ci_a == ci_b


def test_bootstrap_ci_returns_none_for_small_sample() -> None:
    assert bootstrap_ci([0.5, 0.5], n_bootstrap=100, seed=42) is None
    assert bootstrap_ci([], n_bootstrap=100, seed=42) is None


def test_cohens_d_zero_for_identical_groups() -> None:
    g = [0.5, 0.6, 0.4, 0.55, 0.45]
    assert cohens_d(g, g) == 0.0


def test_cohens_d_positive_when_group1_has_higher_mean() -> None:
    high = [0.8, 0.9, 0.85, 0.95, 0.88]
    low = [0.2, 0.3, 0.25, 0.35, 0.22]
    d = cohens_d(high, low)
    assert d > 1.5


def test_cohens_d_returns_zero_for_single_element_groups() -> None:
    assert cohens_d([0.5], [0.6]) == 0.0
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest tests/test_stats.py -xvs
```
Expected: FAIL — `ModuleNotFoundError: No module named 'shared.stats'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `apps/evals/shared/stats.py`:
```python
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def bootstrap_ci(
    values: Sequence[float],
    n_bootstrap: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
) -> tuple[float, float] | None:
    """Compute a bootstrap confidence interval for the sample mean.

    Returns None when N < 5 (not enough samples to produce a meaningful CI).
    Deterministic for a given seed.
    """
    if len(values) < 5:
        return None
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    boot_means = np.empty(n_bootstrap, dtype=float)
    n = len(arr)
    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = float(sample.mean())
    alpha = (1.0 - confidence) / 2.0
    low = float(np.quantile(boot_means, alpha))
    high = float(np.quantile(boot_means, 1.0 - alpha))
    return (low, high)


def cohens_d(group1: Sequence[float], group2: Sequence[float]) -> float:
    """Compute Cohen's d effect size between two groups."""
    if len(group1) < 2 or len(group2) < 2:
        return 0.0
    m1, m2 = float(np.mean(group1)), float(np.mean(group2))
    s1, s2 = float(np.std(group1, ddof=1)), float(np.std(group2, ddof=1))
    n1, n2 = len(group1), len(group2)
    denom = n1 + n2 - 2
    if denom <= 0:
        return 0.0
    pooled = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / denom)
    if pooled == 0:
        return 0.0
    return float((m1 - m2) / pooled)
```

Modify `apps/evals/pipeline/practice_eval/analyze_e2e.py` — replace the existing `cohens_d` function (lines 37–50) with a re-export. Leave the imports of `math` and `numpy as np` in place (they are used elsewhere in the file). Replace the `def cohens_d` block with:
```python
from shared.stats import bootstrap_ci, cohens_d  # re-export for existing callers
```
Place this import immediately after the existing `import numpy as np` line (line 23) and delete the old `def cohens_d(...)` definition at lines 37-50.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest tests/test_stats.py tests/test_analyze_e2e.py -xvs
```
Expected: PASS (both new tests AND existing `test_analyze_e2e.py` tests that import `cohens_d` from the old path)

- [ ] **Step 5: Commit**

```bash
git add apps/evals/shared/stats.py apps/evals/pipeline/practice_eval/analyze_e2e.py apps/evals/tests/test_stats.py && git commit -m "refactor(evals): extract bootstrap_ci and cohens_d to shared/stats"
```

---

### Task T2: Style rules JSON + composer_to_era + get_style_guidance
**Group:** A (parallel)

**Behavior being verified:** Given a composer string, `composer_to_era` returns the correct era label; `get_style_guidance` returns a prose block containing the per-dimension guidance for that era.

**Interface under test:** `shared.style_rules.composer_to_era(composer) -> str`, `shared.style_rules.get_style_guidance(composer) -> str`.

**Files:**
- Create: `apps/evals/shared/style_rules.py`
- Create: `apps/evals/shared/data/style_rules.json`
- Create: `apps/evals/shared/__init__.py` (if not present)
- Create: `apps/evals/shared/data/__init__.py` (if not present — not needed for JSON but keeps package-check happy)
- Test: `apps/evals/tests/test_style_rules.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/tests/test_style_rules.py
from __future__ import annotations

from shared.style_rules import composer_to_era, get_style_guidance


def test_bach_maps_to_baroque() -> None:
    assert composer_to_era("Johann Sebastian Bach") == "Baroque"
    assert composer_to_era("Bach") == "Baroque"


def test_chopin_maps_to_romantic() -> None:
    assert composer_to_era("Frederic Chopin") == "Romantic"
    assert composer_to_era("Chopin") == "Romantic"


def test_debussy_maps_to_impressionist() -> None:
    assert composer_to_era("Claude Debussy") == "Impressionist"


def test_mozart_maps_to_classical() -> None:
    assert composer_to_era("Wolfgang Amadeus Mozart") == "Classical"


def test_unknown_composer_returns_unknown_era() -> None:
    assert composer_to_era("Unknown") == "Unknown"
    assert composer_to_era("") == "Unknown"
    assert composer_to_era("Xyz Fake Composer") == "Unknown"


def test_style_guidance_for_bach_mentions_articulation() -> None:
    guidance = get_style_guidance("Bach")
    assert "Baroque" in guidance
    assert "articulation" in guidance.lower()
    assert "pedaling" in guidance.lower()


def test_style_guidance_for_chopin_mentions_dynamics_and_pedaling() -> None:
    guidance = get_style_guidance("Chopin")
    assert "Romantic" in guidance
    assert "dynamics" in guidance.lower()
    assert "pedaling" in guidance.lower()


def test_style_guidance_for_unknown_returns_empty_string() -> None:
    assert get_style_guidance("Unknown") == ""
    assert get_style_guidance("") == ""


def test_style_guidance_is_formatted_as_xml_block() -> None:
    guidance = get_style_guidance("Bach")
    assert guidance.startswith("<style_guidance")
    assert guidance.endswith("</style_guidance>")
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest tests/test_style_rules.py -xvs
```
Expected: FAIL — `ModuleNotFoundError: No module named 'shared.style_rules'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `apps/evals/shared/data/style_rules.json`:
```json
{
  "eras": {
    "Baroque": {
      "composer_patterns": ["Bach", "Handel", "Scarlatti", "Couperin", "Rameau", "Vivaldi"],
      "dimensions": {
        "articulation": "very high -- clear separation, finger articulation, evenness.",
        "pedaling": "minimal or none; focus on finger legato.",
        "dynamics": "subtle, mainly terraced; use articulation for contrast.",
        "timing": "strict rhythmic precision."
      }
    },
    "Classical": {
      "composer_patterns": ["Mozart", "Haydn", "Clementi", "Beethoven"],
      "dimensions": {
        "articulation": "clean, balanced between legato and light staccato.",
        "pedaling": "light half-pedal for phrasing, never muddy.",
        "dynamics": "wide but structured; follow dynamic markings closely.",
        "timing": "steady pulse, slight rubato only in repeats."
      }
    },
    "Romantic": {
      "composer_patterns": ["Chopin", "Brahms", "Rachmaninoff", "Liszt", "Schumann", "Schubert", "Mendelssohn", "Tchaikovsky", "Grieg"],
      "dimensions": {
        "dynamics": "high -- nuanced crescendi/decrescendi, tonal colour.",
        "pedaling": "essential for legato and resonance; use half-pedal.",
        "articulation": "flexible -- blend legato with occasional accents.",
        "timing": "rubato as expressive tool; maintain structural integrity."
      }
    },
    "Impressionist": {
      "composer_patterns": ["Debussy", "Ravel", "Faure", "Satie"],
      "dimensions": {
        "pedaling": "critical -- colour, wash, blurred harmonies.",
        "dynamics": "very fine gradations; avoid extremes.",
        "articulation": "smooth, often finger legato; avoid heavy attacks.",
        "timing": "flexible tempo, free rubato within phrases."
      }
    },
    "Early20th": {
      "composer_patterns": ["Stravinsky", "Prokofiev", "Bartok", "Shostakovich", "Gershwin"],
      "dimensions": {
        "timing": "high -- syncopation, swing feel, polyrhythms.",
        "articulation": "crisp attacks, ghost notes, varied touch.",
        "dynamics": "dynamic contrast for phrasing, but often limited extremes.",
        "pedaling": "used sparingly, mainly for sustain."
      }
    },
    "Contemporary": {
      "composer_patterns": ["Ligeti", "Messiaen", "Cage", "Crumb", "Reich", "Glass"],
      "dimensions": {
        "dynamics": "often extreme (ppp-fff) or graphic symbols -- follow composer's intent.",
        "pedaling": "may be unconventional; follow score or composer's notes.",
        "articulation": "explore extended techniques as written.",
        "timing": "follow composer's notation strictly even when unusual."
      }
    }
  }
}
```

Create `apps/evals/shared/style_rules.py`:
```python
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_STYLE_RULES_PATH = Path(__file__).parent / "data" / "style_rules.json"


@lru_cache(maxsize=1)
def _load_style_rules() -> dict:
    return json.loads(_STYLE_RULES_PATH.read_text())


def composer_to_era(composer: str) -> str:
    """Map a composer name to its stylistic era.

    Uses substring matching against composer_patterns in style_rules.json.
    Returns "Unknown" if no pattern matches.
    """
    if not composer:
        return "Unknown"
    rules = _load_style_rules()
    for era_name, era_data in rules["eras"].items():
        for pattern in era_data["composer_patterns"]:
            if pattern.lower() in composer.lower():
                return era_name
    return "Unknown"


def get_style_guidance(composer: str) -> str:
    """Return an XML-wrapped prose block with per-dimension style guidance.

    Returns an empty string for unknown composers so the caller can omit
    the section entirely from the prompt.
    """
    era = composer_to_era(composer)
    if era == "Unknown":
        return ""
    rules = _load_style_rules()
    dimensions = rules["eras"][era]["dimensions"]
    lines = [
        f"<style_guidance era=\"{era}\">",
        f"For {era}-era repertoire, weight dimensions as follows when giving feedback:",
    ]
    for dim, rule in dimensions.items():
        lines.append(f"- {dim}: {rule}")
    lines.append("Advice that contradicts these rules should not be given.")
    lines.append("</style_guidance>")
    return "\n".join(lines)
```

Create `apps/evals/shared/__init__.py` if missing:
```python
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest tests/test_style_rules.py -xvs
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/evals/shared/style_rules.py apps/evals/shared/data/style_rules.json apps/evals/shared/__init__.py apps/evals/tests/test_style_rules.py && git commit -m "feat(evals): add style_rules.json source of truth + composer_to_era helper"
```

---

### Task T3: make_run_provenance
**Group:** A (parallel)

**Behavior being verified:** `make_run_provenance()` returns a `RunProvenance` dataclass with a filesystem-safe `run_id`, a non-empty `git_sha`, and a boolean `git_dirty` flag. When `git` is unavailable, it falls back to `git_sha="unknown"` and `git_dirty=True` without raising.

**Interface under test:** `shared.provenance.make_run_provenance(suffix) -> RunProvenance`.

**Files:**
- Create: `apps/evals/shared/provenance.py`
- Test: `apps/evals/tests/test_provenance.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/tests/test_provenance.py
from __future__ import annotations

import re
import subprocess

import pytest

from shared.provenance import RunProvenance, make_run_provenance


def test_make_run_provenance_returns_all_fields() -> None:
    prov = make_run_provenance()
    assert isinstance(prov, RunProvenance)
    assert prov.run_id
    assert prov.git_sha
    assert isinstance(prov.git_dirty, bool)


def test_run_id_is_filesystem_safe() -> None:
    prov = make_run_provenance()
    # No characters that would break a filename on macOS/Linux
    assert re.match(r"^[A-Za-z0-9._\-]+$", prov.run_id), f"unsafe run_id: {prov.run_id}"


def test_run_id_includes_suffix_when_given() -> None:
    prov = make_run_provenance(suffix="candidate-42")
    assert "candidate-42" in prov.run_id


def test_git_sha_is_hex_string_when_git_available() -> None:
    prov = make_run_provenance()
    if prov.git_sha == "unknown":
        pytest.skip("git not available in this environment")
    assert re.match(r"^[0-9a-f]{7,40}$", prov.git_sha), f"not a git sha: {prov.git_sha}"


def test_falls_back_gracefully_when_git_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args, **kwargs):
        raise FileNotFoundError("git binary not found")

    monkeypatch.setattr(subprocess, "run", fake_run)
    prov = make_run_provenance()
    assert prov.git_sha == "unknown"
    assert prov.git_dirty is True
    assert prov.run_id  # still produces an id


def test_two_calls_close_in_time_have_distinct_ids_when_suffix_differs() -> None:
    a = make_run_provenance(suffix="a")
    b = make_run_provenance(suffix="b")
    assert a.run_id != b.run_id
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest tests/test_provenance.py -xvs
```
Expected: FAIL — `ModuleNotFoundError: No module named 'shared.provenance'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `apps/evals/shared/provenance.py`:
```python
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class RunProvenance:
    run_id: str
    git_sha: str
    git_dirty: bool


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def make_run_provenance(suffix: str | None = None) -> RunProvenance:
    """Stamp a run with a filesystem-safe ID, git SHA, and dirty-tree flag.

    Falls back to git_sha="unknown" + git_dirty=True when the git binary is
    unavailable (e.g., sandboxed CI). Always produces a run_id.
    """
    sha = _git("rev-parse", "HEAD")
    if sha is None:
        print("warn: git unavailable, using unknown SHA", file=sys.stderr)
        git_sha = "unknown"
        git_dirty = True
    else:
        git_sha = sha
        porcelain = _git("status", "--porcelain")
        git_dirty = bool(porcelain) if porcelain is not None else True

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    short_sha = git_sha[:7] if git_sha != "unknown" else "nosha"
    run_id = f"{timestamp}_{short_sha}"
    if suffix:
        run_id = f"{run_id}_{suffix}"

    return RunProvenance(run_id=run_id, git_sha=git_sha, git_dirty=git_dirty)
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest tests/test_provenance.py -xvs
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/evals/shared/provenance.py apps/evals/tests/test_provenance.py && git commit -m "feat(evals): add run provenance stamper (run_id + git_sha + dirty)"
```

---

### Task T4: assert_judge_compatible
**Group:** A (parallel)

**Behavior being verified:** Same-family teacher/judge pairs raise `ValueError`; cross-family pairs return None. Model-name prefixes for Anthropic native, Workers AI, and OpenRouter slugs all resolve correctly.

**Interface under test:** `shared.judge_compatibility.assert_judge_compatible(teacher_model, judge_model) -> None`.

**Files:**
- Create: `apps/evals/shared/judge_compatibility.py`
- Test: `apps/evals/tests/test_judge_compatibility.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/tests/test_judge_compatibility.py
from __future__ import annotations

import pytest

from shared.judge_compatibility import assert_judge_compatible, model_family


def test_anthropic_native_same_family_raises() -> None:
    with pytest.raises(ValueError, match="anthropic"):
        assert_judge_compatible("claude-sonnet-4-6", "claude-sonnet-4-6")


def test_anthropic_native_vs_openrouter_anthropic_same_family_raises() -> None:
    with pytest.raises(ValueError, match="anthropic"):
        assert_judge_compatible("claude-sonnet-4-6", "anthropic/claude-sonnet-4-6")


def test_anthropic_teacher_vs_google_workers_ai_judge_passes() -> None:
    assert_judge_compatible("claude-sonnet-4-6", "@cf/google/gemma-4-26b-a4b-it")


def test_anthropic_teacher_vs_openrouter_openai_passes() -> None:
    assert_judge_compatible("claude-sonnet-4-6", "openai/gpt-5.4-mini")


def test_workers_ai_openai_prefix_resolves_to_openai_family() -> None:
    with pytest.raises(ValueError, match="openai"):
        assert_judge_compatible("@cf/openai/gpt-oss-120b", "openai/gpt-5.4-mini")


def test_qwen_teacher_vs_sonnet_judge_passes() -> None:
    assert_judge_compatible("qwen3-27b-finetune", "claude-sonnet-4-6")


def test_model_family_unknown_raises_to_surface_typos() -> None:
    with pytest.raises(ValueError, match="unknown model family"):
        model_family("some-random-nonsense-model")
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest tests/test_judge_compatibility.py -xvs
```
Expected: FAIL — `ModuleNotFoundError: No module named 'shared.judge_compatibility'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `apps/evals/shared/judge_compatibility.py`:
```python
from __future__ import annotations

# Rules are checked in order; first match wins.
# Each rule is (substring_to_match, family_name).
_FAMILY_RULES: list[tuple[str, str]] = [
    # Anthropic native names
    ("claude-", "anthropic"),
    # OpenRouter slugs (vendor/model)
    ("anthropic/", "anthropic"),
    ("openai/", "openai"),
    ("google/", "google"),
    ("qwen/", "qwen"),
    ("meta-llama/", "meta"),
    # Workers AI slugs (@cf/vendor/model)
    ("@cf/openai/", "openai"),
    ("@cf/google/", "google"),
    ("@cf/qwen/", "qwen"),
    ("@cf/meta/", "meta"),
    # Direct vendor names (finetune artifacts, etc.)
    ("gpt-", "openai"),
    ("gemma-", "google"),
    ("gemini-", "google"),
    ("qwen", "qwen"),
    ("llama", "meta"),
]


def model_family(model: str) -> str:
    """Resolve a model name to its family name.

    Raises ValueError if no rule matches (fail-fast on typos).
    """
    lowered = model.lower()
    for substring, family in _FAMILY_RULES:
        if substring in lowered:
            return family
    raise ValueError(f"unknown model family for: {model!r}")


def assert_judge_compatible(teacher_model: str, judge_model: str) -> None:
    """Raise if teacher and judge share a model family.

    Cross-family judging is required to avoid same-family phrasing-preference
    bias in evals. See the eval strategy in
    docs/plans/2026-04-14-eval-improvements.md.
    """
    t_family = model_family(teacher_model)
    j_family = model_family(judge_model)
    if t_family == j_family:
        raise ValueError(
            f"judge family {j_family!r} matches teacher family -- forbidden "
            f"(teacher={teacher_model!r}, judge={judge_model!r})"
        )
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest tests/test_judge_compatibility.py -xvs
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/evals/shared/judge_compatibility.py apps/evals/tests/test_judge_compatibility.py && git commit -m "feat(evals): add assert_judge_compatible same-family guard"
```

---

### Task T5: LLMClient OpenRouter provider
**Group:** A (parallel)

**Behavior being verified:** `_build_openrouter_payload(model, messages, max_tokens)` returns a dict with the correct `model`, `messages`, and `max_tokens` fields; `LLMClient(provider="openrouter")` can be instantiated without triggering network I/O (the API key lookup can be stubbed via env var in the test).

**Interface under test:** `LLMClient._build_openrouter_payload` (package-private but tested directly as a pure function), `LLMClient(provider="openrouter", tier="judge")` constructor.

**Files:**
- Modify: `apps/evals/teaching_knowledge/llm_client.py`
- Test: `apps/evals/tests/test_llm_client_openrouter.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/tests/test_llm_client_openrouter.py
from __future__ import annotations

import os

import pytest

from teaching_knowledge.llm_client import LLMClient, _build_openrouter_payload


def test_build_openrouter_payload_shape() -> None:
    payload = _build_openrouter_payload(
        model="openai/gpt-5.4-mini",
        system="You are a judge.",
        user="Evaluate this: hello world",
        max_tokens=2048,
    )
    assert payload["model"] == "openai/gpt-5.4-mini"
    assert payload["max_tokens"] == 2048
    assert len(payload["messages"]) == 2
    assert payload["messages"][0] == {"role": "system", "content": "You are a judge."}
    assert payload["messages"][1] == {"role": "user", "content": "Evaluate this: hello world"}


def test_build_openrouter_payload_without_system() -> None:
    payload = _build_openrouter_payload(
        model="openai/gpt-5.4-mini",
        system="",
        user="hello",
        max_tokens=100,
    )
    assert len(payload["messages"]) == 1
    assert payload["messages"][0]["role"] == "user"


def test_openrouter_client_uses_judge_tier_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-fake-key")
    client = LLMClient(provider="openrouter", tier="judge")
    assert client.provider == "openrouter"
    assert client.model == "openai/gpt-5.4-mini"


def test_openrouter_client_accepts_explicit_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-fake-key")
    client = LLMClient(provider="openrouter", model="anthropic/claude-sonnet-4-6")
    assert client.model == "anthropic/claude-sonnet-4-6"


def test_openrouter_client_raises_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    # Ensure .dev.vars fallback also fails by pointing lookup away
    def no_dev_vars(_: str) -> str | None:
        return None
    monkeypatch.setattr(
        "teaching_knowledge.llm_client._load_dev_vars_key", no_dev_vars
    )
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        LLMClient(provider="openrouter", tier="judge")
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest tests/test_llm_client_openrouter.py -xvs
```
Expected: FAIL — `ImportError: cannot import name '_build_openrouter_payload'` (or the MODELS dict has no "openrouter" key).

- [ ] **Step 3: Implement the minimum to make the test pass**

Modify `apps/evals/teaching_knowledge/llm_client.py`:

1. Add `"openrouter"` entry to `MODELS` table (after the `"anthropic"` entry):
```python
    "openrouter": {
        "cheap": "openai/gpt-5.4-mini",
        "quality": "openai/gpt-5.4-mini",
        "judge": "openai/gpt-5.4-mini",
        "default": "openai/gpt-5.4-mini",
    },
```

2. Add loader function after `_load_anthropic_key`:
```python
def _load_openrouter_key() -> str:
    """Load OPENROUTER_API_KEY from env or apps/api/.dev.vars."""
    key = os.environ.get("OPENROUTER_API_KEY") or _load_dev_vars_key("OPENROUTER_API_KEY")
    if key:
        return key
    raise RuntimeError(
        "OPENROUTER_API_KEY not found. Set it in env or apps/api/.dev.vars"
    )
```

3. Add module-level pure helper before the `LLMClient` class (after `_load_openrouter_key`):
```python
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"


def _build_openrouter_payload(
    model: str,
    system: str,
    user: str,
    max_tokens: int,
) -> dict:
    """Pure function that builds the OpenRouter chat-completions request body.

    Extracted from _openrouter_complete for unit testing without network I/O.
    """
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
```

4. In `LLMClient.__init__`, add the openrouter branch (after the `anthropic` branch):
```python
        elif provider == "openrouter":
            self._openrouter_key = _load_openrouter_key()
```

5. In `LLMClient.complete`, add dispatch (after the `anthropic` branch):
```python
        elif self.provider == "openrouter":
            return self._openrouter_complete(system, user, max_tokens)
```

6. Add `_openrouter_complete` method on the class (after `_anthropic_complete`):
```python
    def _openrouter_complete(self, system: str, user: str, max_tokens: int) -> str:
        payload = _build_openrouter_payload(self.model, system, user, max_tokens)
        response = requests.post(
            OPENROUTER_BASE_URL,
            headers={
                "Authorization": f"Bearer {self._openrouter_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://crescend.ai",
                "X-Title": "CrescendAI Evals",
            },
            json=payload,
            timeout=300,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"OpenRouter returned {response.status_code}: {response.text[:500]}"
            )
        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(
                f"No choices in OpenRouter response: {json.dumps(data)[:500]}"
            )
        content = choices[0]["message"].get("content")
        if content is None:
            raise RuntimeError(
                f"OpenRouter returned null content: {json.dumps(data)[:500]}"
            )
        return content
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest tests/test_llm_client_openrouter.py -xvs
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teaching_knowledge/llm_client.py apps/evals/tests/test_llm_client_openrouter.py && git commit -m "feat(evals): add OpenRouter provider to LLMClient"
```

---

### Task T6: DimensionScore process/outcome fields + parser update
**Group:** A (parallel)

**Behavior being verified:** `_parse_v2_response` handles both legacy (single `score`) and new (`process` + `outcome`) response shapes. Legacy responses set `process == outcome == score`. New responses populate `process` and `outcome` independently and derive the composite `score` as their mean (floor).

**Interface under test:** `shared.judge._parse_v2_response(response_text) -> list[DimensionScore]` and the `DimensionScore` dataclass.

**Files:**
- Modify: `apps/evals/shared/judge.py`
- Test: `apps/evals/tests/test_judge_process_outcome.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/tests/test_judge_process_outcome.py
from __future__ import annotations

import json

from shared.judge import DimensionScore, _parse_v2_response


def test_dimension_score_has_process_and_outcome_fields() -> None:
    d = DimensionScore(
        criterion="Audible-Specific Corrective Feedback",
        score=2,
        evidence="ok",
        reason="fine",
        process=3,
        outcome=2,
    )
    assert d.process == 3
    assert d.outcome == 2


def test_parse_legacy_single_score_response() -> None:
    legacy = json.dumps([
        {
            "criterion": "Specific Positive Praise",
            "score": 2,
            "evidence": "You nailed the crescendo.",
            "reason": "Concrete and warm.",
        }
    ])
    dims = _parse_v2_response(legacy)
    assert len(dims) == 1
    d = dims[0]
    assert d.score == 2
    assert d.process == 2
    assert d.outcome == 2


def test_parse_new_process_outcome_response() -> None:
    new = json.dumps([
        {
            "criterion": "Audible-Specific Corrective Feedback",
            "process": 3,
            "outcome": 1,
            "evidence": "bar 12",
            "reason": "noticed but incorrect",
        }
    ])
    dims = _parse_v2_response(new)
    assert len(dims) == 1
    d = dims[0]
    assert d.process == 3
    assert d.outcome == 1
    # composite score = min of the two signals (a conservative composite)
    assert d.score == 1


def test_parse_new_schema_with_na_process_or_outcome() -> None:
    new = json.dumps([
        {
            "criterion": "Autonomy-Supporting Motivation",
            "process": "N/A",
            "outcome": "N/A",
            "evidence": "",
            "reason": "Not applicable.",
        }
    ])
    dims = _parse_v2_response(new)
    assert dims[0].process is None
    assert dims[0].outcome is None
    assert dims[0].score is None


def test_parse_new_schema_with_only_process_na() -> None:
    new = json.dumps([
        {
            "criterion": "Concrete Artifact Provision",
            "process": "N/A",
            "outcome": 2,
            "evidence": "",
            "reason": "Praise-only, no artifact expected.",
        }
    ])
    dims = _parse_v2_response(new)
    assert dims[0].process is None
    assert dims[0].outcome == 2
    # When one side is N/A, composite = the other side
    assert dims[0].score == 2


def test_parse_failure_still_returns_one_dimension() -> None:
    dims = _parse_v2_response("not json at all")
    assert len(dims) == 1
    assert dims[0].criterion == "parse_failure"
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest tests/test_judge_process_outcome.py -xvs
```
Expected: FAIL — `TypeError: DimensionScore.__init__() got an unexpected keyword argument 'process'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Modify `apps/evals/shared/judge.py`:

Replace the `DimensionScore` dataclass (lines 47-53) with:
```python
@dataclass
class DimensionScore:
    """Score for a v2 rubric dimension (0-3 scale, or None for N/A).

    process: did the teacher notice / attempt the behavior?
    outcome: was the advice correct given the actual performance?
    score: composite of process + outcome (min when both present).
    Legacy single-score responses set process == outcome == score.
    """
    criterion: str
    score: int | None
    evidence: str
    reason: str
    process: int | None = None
    outcome: int | None = None
```

Replace `_parse_v2_response` (lines 350-383) with:
```python
def _parse_v2_response(response_text: str) -> list[DimensionScore]:
    """Parse v2 judge JSON response into DimensionScore list.

    Handles both legacy single-score rows and new process/outcome rows:
    - Legacy: {criterion, score, evidence, reason}
      -> process = outcome = score (back-compat for existing fixtures)
    - New:    {criterion, process, outcome, evidence, reason}
      -> composite score = min(process, outcome) when both numeric,
         or the non-None side when one is N/A, or None when both N/A
    Score values of "N/A" (case-insensitive) map to None.
    """
    def _coerce(raw: object) -> int | None:
        if raw is None:
            return None
        if isinstance(raw, str) and raw.strip().upper() == "N/A":
            return None
        return int(raw)

    def _composite(process: int | None, outcome: int | None) -> int | None:
        if process is None and outcome is None:
            return None
        if process is None:
            return outcome
        if outcome is None:
            return process
        return min(process, outcome)

    try:
        data = json.loads(response_text)
        if not isinstance(data, list):
            data = [data]
        dimensions: list[DimensionScore] = []
        for entry in data:
            if "process" in entry or "outcome" in entry:
                process = _coerce(entry.get("process"))
                outcome = _coerce(entry.get("outcome"))
                score = _composite(process, outcome)
            else:
                score = _coerce(entry.get("score"))
                process = score
                outcome = score
            dimensions.append(
                DimensionScore(
                    criterion=entry.get("criterion", "unknown"),
                    score=score,
                    evidence=entry.get("evidence", ""),
                    reason=entry.get("reason", ""),
                    process=process,
                    outcome=outcome,
                )
            )
        return dimensions
    except (json.JSONDecodeError, TypeError, KeyError, ValueError) as e:
        return [
            DimensionScore(
                criterion="parse_failure",
                score=0,
                evidence=f"Could not parse v2 judge response: {e}",
                reason=response_text[:200],
                process=None,
                outcome=None,
            )
        ]
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest tests/test_judge_process_outcome.py -xvs
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/evals/shared/judge.py apps/evals/tests/test_judge_process_outcome.py && git commit -m "feat(evals): add process/outcome split to DimensionScore (backwards-compat)"
```

---

### Task T7: Judge prompt update (process/outcome schema)
**Group:** A (parallel)

**Behavior being verified:** The rendered judge prompt file contains the `process` and `outcome` keys in its example output schema and contains a one-sentence description of what each means.

**Interface under test:** The prompt file is read by `shared.judge.load_prompt("synthesis_quality_judge_v2.txt")`. A pytest reads the prompt via this public path and asserts the required substrings are present.

**Files:**
- Modify: `apps/evals/shared/prompts/synthesis_quality_judge_v2.txt`
- Test: `apps/evals/tests/test_judge_prompt_schema.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/tests/test_judge_prompt_schema.py
from __future__ import annotations

from shared.judge import load_prompt


def test_judge_prompt_mentions_process_and_outcome_keys() -> None:
    prompt = load_prompt("synthesis_quality_judge_v2.txt")
    assert '"process"' in prompt
    assert '"outcome"' in prompt


def test_judge_prompt_defines_process_vs_outcome() -> None:
    prompt = load_prompt("synthesis_quality_judge_v2.txt")
    lowered = prompt.lower()
    # must explain what each signal means
    assert "process" in lowered and "notice" in lowered
    assert "outcome" in lowered and ("correct" in lowered or "accurate" in lowered)


def test_judge_prompt_retains_seven_dimension_numbered_list() -> None:
    prompt = load_prompt("synthesis_quality_judge_v2.txt")
    for i in range(1, 8):
        assert f"{i}." in prompt


def test_judge_prompt_retains_zero_to_three_scale() -> None:
    prompt = load_prompt("synthesis_quality_judge_v2.txt")
    assert "0-3" in prompt or "0\u20113" in prompt or "0\u20133" in prompt
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest tests/test_judge_prompt_schema.py -xvs
```
Expected: FAIL — the current prompt has `"score"` not `"process"`/`"outcome"`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Replace the entire contents of `apps/evals/shared/prompts/synthesis_quality_judge_v2.txt` with:
```
You are an expert evaluator of AI-generated piano-teaching feedback. Use the rubric below to score each AI response on a 0-3 scale for every dimension. For each dimension, assign separate process and outcome scores, provide a brief evidence quote from the AI output, and explain why the scores fit the criteria. Follow these rules while scoring:

1. **Audible-Specific Corrective Feedback** - look for bar/beat references, a measurable audible target, and a concrete corrective action.
2. **Concrete Artifact Provision** - check for a specific exercise, score highlight, or recording suggestion, plus a short musical cue/metaphor.
3. **Specific Positive Praise** - require a concrete musical success, why it matters, and warm expressive language.
4. **Autonomy-Supporting Motivation** - must respect learner agency, offer a choice or goal, and give a clear next step.
5. **Scaffolded Guided Discovery** - evaluate chunking, sequencing, self-assessment prompts, and musical outcome linkage.
6. **Style-Consistent Musical Language** - compare the advice to the Piece-Style Dimension Rules in the playbook; advice that contradicts the style gets a 0.
7. **Appropriate Tone & Language** - ensure the register matches the feedback type and that no "never-say" statements appear.

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
Provide the array in the order of the dimensions listed above.
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest tests/test_judge_prompt_schema.py -xvs
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/evals/shared/prompts/synthesis_quality_judge_v2.txt apps/evals/tests/test_judge_prompt_schema.py && git commit -m "feat(evals): judge prompt v2 requests process and outcome separately"
```

---

### Task T8: tag_dataset.py
**Group:** B (parallel, depends on A)

**Behavior being verified:** `tag_recording(recording_id, manifest_entry, cache_entry)` produces a `RecordingTags` dataclass with the correct `composer_era` (via `composer_to_era`), `skill_bucket` (from manifest), and `duration_bucket` (from cache entry duration).

**Interface under test:** `teaching_knowledge.scripts.tag_dataset.tag_recording`, `build_dataset_index`, `RecordingTags`.

**Files:**
- Create: `apps/evals/teaching_knowledge/scripts/__init__.py`
- Create: `apps/evals/teaching_knowledge/scripts/tag_dataset.py`
- Test: `apps/evals/tests/test_tag_dataset.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/tests/test_tag_dataset.py
from __future__ import annotations

import json
from pathlib import Path

import pytest

from teaching_knowledge.scripts.tag_dataset import (
    RecordingTags,
    build_dataset_index,
    tag_recording,
)


def test_tag_recording_bach_short() -> None:
    manifest = {
        "piece_slug": "bach_wtc_1_prelude",
        "title": "WTC Book 1 Prelude No. 1",
        "composer": "Johann Sebastian Bach",
        "skill_bucket": 3,
    }
    cache = {"total_duration_seconds": 25.0, "chunks": []}
    tags = tag_recording("abc123", manifest, cache)
    assert tags.recording_id == "abc123"
    assert tags.composer_era == "Baroque"
    assert tags.skill_bucket == 3
    assert tags.duration_bucket == "<30s"


def test_tag_recording_chopin_medium() -> None:
    manifest = {
        "piece_slug": "chopin_ballade_1",
        "title": "Ballade No. 1",
        "composer": "Chopin",
        "skill_bucket": 5,
    }
    cache = {"total_duration_seconds": 45.0, "chunks": []}
    tags = tag_recording("xyz789", manifest, cache)
    assert tags.composer_era == "Romantic"
    assert tags.duration_bucket == "30-60s"


def test_tag_recording_long_recording() -> None:
    manifest = {
        "piece_slug": "debussy_clair",
        "title": "Clair de Lune",
        "composer": "Debussy",
        "skill_bucket": 4,
    }
    cache = {"total_duration_seconds": 120.0}
    tags = tag_recording("def456", manifest, cache)
    assert tags.composer_era == "Impressionist"
    assert tags.duration_bucket == "60s+"


def test_tag_recording_unknown_composer_still_tags() -> None:
    manifest = {
        "piece_slug": "unknown",
        "title": "Unknown",
        "composer": "Nobody Famous",
        "skill_bucket": 2,
    }
    cache = {"total_duration_seconds": 10.0}
    tags = tag_recording("unk001", manifest, cache)
    assert tags.composer_era == "Unknown"
    assert tags.skill_bucket == 2


def test_build_dataset_index_writes_jsonl(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    bach_path = cache_dir / "bach1.json"
    bach_path.write_text(json.dumps({
        "recording_id": "bach1",
        "total_duration_seconds": 25.0,
        "chunks": [],
    }))
    chopin_path = cache_dir / "chopin1.json"
    chopin_path.write_text(json.dumps({
        "recording_id": "chopin1",
        "total_duration_seconds": 90.0,
        "chunks": [],
    }))

    manifest_lookup = {
        "bach1": {
            "piece_slug": "bach", "title": "Bach", "composer": "Bach", "skill_bucket": 3,
        },
        "chopin1": {
            "piece_slug": "chopin", "title": "Chopin", "composer": "Chopin", "skill_bucket": 5,
        },
    }

    out = tmp_path / "dataset_index.jsonl"
    build_dataset_index(manifest_lookup, cache_dir, out)

    lines = out.read_text().strip().split("\n")
    assert len(lines) == 2
    rows = sorted([json.loads(line) for line in lines], key=lambda r: r["recording_id"])
    assert rows[0]["recording_id"] == "bach1"
    assert rows[0]["composer_era"] == "Baroque"
    assert rows[0]["duration_bucket"] == "<30s"
    assert rows[1]["recording_id"] == "chopin1"
    assert rows[1]["composer_era"] == "Romantic"
    assert rows[1]["duration_bucket"] == "60s+"
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest tests/test_tag_dataset.py -xvs
```
Expected: FAIL — `ModuleNotFoundError: No module named 'teaching_knowledge.scripts.tag_dataset'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `apps/evals/teaching_knowledge/scripts/__init__.py`:
```python
```

Create `apps/evals/teaching_knowledge/scripts/tag_dataset.py`:
```python
"""Enrich inference-cache entries with composer_era, skill, duration tags.

Reads the auto-t5_http inference cache plus skill_eval manifests, produces
dataset_index.jsonl -- one row per cached recording with the tags that
split.py and aggregate.py stratify over.

Usage:
    cd apps/evals
    uv run python -m teaching_knowledge.scripts.tag_dataset
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from shared.style_rules import composer_to_era


@dataclass(frozen=True)
class RecordingTags:
    recording_id: str
    composer_era: str
    skill_bucket: int
    duration_bucket: str


def _duration_bucket(seconds: float) -> str:
    if seconds < 30:
        return "<30s"
    if seconds < 60:
        return "30-60s"
    return "60s+"


def tag_recording(
    recording_id: str,
    manifest_entry: dict,
    cache_entry: dict,
) -> RecordingTags:
    composer = manifest_entry.get("composer", "Unknown")
    return RecordingTags(
        recording_id=recording_id,
        composer_era=composer_to_era(composer),
        skill_bucket=int(manifest_entry.get("skill_bucket", 3)),
        duration_bucket=_duration_bucket(float(cache_entry.get("total_duration_seconds", 0.0))),
    )


def build_dataset_index(
    manifest_lookup: dict[str, dict],
    cache_dir: Path,
    out_path: Path,
) -> None:
    """Walk the inference cache and emit a tagged JSONL index."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fout:
        for cache_file in sorted(cache_dir.glob("*.json")):
            if cache_file.name == "_fingerprint.json":
                continue
            cache_entry = json.loads(cache_file.read_text())
            recording_id = cache_entry.get("recording_id", cache_file.stem)
            manifest_entry = manifest_lookup.get(recording_id)
            if manifest_entry is None:
                continue
            tags = tag_recording(recording_id, manifest_entry, cache_entry)
            fout.write(json.dumps(asdict(tags)) + "\n")


def main() -> None:
    from teaching_knowledge.run_eval import CACHE_DIR, EVALS_ROOT, load_manifests

    parser = argparse.ArgumentParser(description="Build dataset_index.jsonl")
    parser.add_argument(
        "--out",
        type=Path,
        default=EVALS_ROOT / "teaching_knowledge" / "data" / "dataset_index.jsonl",
    )
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    args = parser.parse_args()

    manifest_lookup = load_manifests()
    build_dataset_index(manifest_lookup, args.cache_dir, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest tests/test_tag_dataset.py -xvs
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teaching_knowledge/scripts/__init__.py apps/evals/teaching_knowledge/scripts/tag_dataset.py apps/evals/tests/test_tag_dataset.py && git commit -m "feat(evals): add tag_dataset.py for composer_era/skill/duration tagging"
```

---

### Task T9: apps/api/src/lib/style-rules.json mirror + drift guard
**Group:** B (parallel, depends on A)

**Behavior being verified:** The Python-side `apps/evals/shared/data/style_rules.json` and the TypeScript-side `apps/api/src/lib/style-rules.json` are byte-identical. A pytest catches drift.

**Interface under test:** Filesystem content equality of the two files.

**Files:**
- Create: `apps/api/src/lib/style-rules.json` (byte-identical copy)
- Test: `apps/evals/tests/test_style_rules_mirror.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/tests/test_style_rules_mirror.py
from __future__ import annotations

import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PY_PATH = REPO_ROOT / "apps" / "evals" / "shared" / "data" / "style_rules.json"
TS_PATH = REPO_ROOT / "apps" / "api" / "src" / "lib" / "style-rules.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_mirror_file_exists() -> None:
    assert TS_PATH.exists(), f"TS mirror missing at {TS_PATH}"


def test_python_and_ts_style_rules_are_byte_identical() -> None:
    assert _sha256(PY_PATH) == _sha256(TS_PATH), (
        f"Style rules drift detected!\n"
        f"  Python: {PY_PATH}\n"
        f"  TS:     {TS_PATH}\n"
        f"Run: cp {PY_PATH} {TS_PATH}"
    )
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest tests/test_style_rules_mirror.py -xvs
```
Expected: FAIL — `test_mirror_file_exists` fails because the file does not yet exist.

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `apps/api/src/lib/style-rules.json` as a byte-identical copy of `apps/evals/shared/data/style_rules.json`:
```bash
mkdir -p apps/api/src/lib && cp apps/evals/shared/data/style_rules.json apps/api/src/lib/style-rules.json
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest tests/test_style_rules_mirror.py -xvs
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/lib/style-rules.json apps/evals/tests/test_style_rules_mirror.py && git commit -m "feat(api): mirror style_rules.json + pytest drift guard"
```

---

### Task T10: split.py stratified split
**Group:** C (depends on T8)

**Behavior being verified:** `stratified_split` deterministically produces an 80/20 split for a given seed; strata are preserved within tolerance; `load_split(which="holdout")` returns the exact holdout set from the persisted JSON.

**Interface under test:** `teaching_knowledge.scripts.split.stratified_split`, `write_split`, `load_split`.

**Files:**
- Create: `apps/evals/teaching_knowledge/scripts/split.py`
- Test: `apps/evals/tests/test_split.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/tests/test_split.py
from __future__ import annotations

import json
from pathlib import Path

from teaching_knowledge.scripts.split import (
    Split,
    load_split,
    stratified_split,
    write_split,
)
from teaching_knowledge.scripts.tag_dataset import RecordingTags


def _make_tags(n: int) -> list[RecordingTags]:
    eras = ["Baroque", "Classical", "Romantic", "Impressionist"]
    tags = []
    for i in range(n):
        tags.append(
            RecordingTags(
                recording_id=f"rec_{i:04d}",
                composer_era=eras[i % len(eras)],
                skill_bucket=(i % 5) + 1,
                duration_bucket=["<30s", "30-60s", "60s+"][i % 3],
            )
        )
    return tags


def test_stratified_split_80_20_ratio() -> None:
    tags = _make_tags(100)
    split = stratified_split(tags, seed=42, holdout_ratio=0.2)
    assert len(split.holdout) == 20
    assert len(split.train) == 80
    assert set(split.train).isdisjoint(split.holdout)
    assert set(split.train) | set(split.holdout) == {t.recording_id for t in tags}


def test_stratified_split_is_deterministic() -> None:
    tags = _make_tags(100)
    a = stratified_split(tags, seed=42, holdout_ratio=0.2)
    b = stratified_split(tags, seed=42, holdout_ratio=0.2)
    assert a.train == b.train
    assert a.holdout == b.holdout


def test_stratified_split_different_seeds_differ() -> None:
    tags = _make_tags(100)
    a = stratified_split(tags, seed=1, holdout_ratio=0.2)
    b = stratified_split(tags, seed=2, holdout_ratio=0.2)
    assert set(a.holdout) != set(b.holdout)


def test_stratified_split_preserves_era_distribution() -> None:
    tags = _make_tags(200)
    split = stratified_split(tags, seed=42, holdout_ratio=0.2)
    holdout_ids = set(split.holdout)
    holdout_eras = [t.composer_era for t in tags if t.recording_id in holdout_ids]
    all_eras = [t.composer_era for t in tags]
    # Each era in holdout should be within 5% of its overall share
    for era in set(all_eras):
        all_share = all_eras.count(era) / len(all_eras)
        holdout_share = holdout_eras.count(era) / len(holdout_eras) if holdout_eras else 0
        assert abs(all_share - holdout_share) < 0.1, f"era {era} drifted"


def test_write_and_load_split_roundtrip(tmp_path: Path) -> None:
    tags = _make_tags(50)
    split = stratified_split(tags, seed=7, holdout_ratio=0.2)
    path = tmp_path / "splits.json"
    write_split(split, path)

    assert path.exists()
    blob = json.loads(path.read_text())
    assert "train" in blob and "holdout" in blob

    train_set = load_split(path, which="train")
    holdout_set = load_split(path, which="holdout")
    all_set = load_split(path, which="all")
    assert train_set == set(split.train)
    assert holdout_set == set(split.holdout)
    assert all_set == train_set | holdout_set


def test_stratified_split_handles_small_strata() -> None:
    # Only one recording per stratum-combo; split should still not crash
    tags = [
        RecordingTags(f"r{i}", era, skill, "<30s")
        for i, (era, skill) in enumerate(
            [("Baroque", 1), ("Classical", 2), ("Romantic", 3), ("Impressionist", 4)]
        )
    ]
    split = stratified_split(tags, seed=0, holdout_ratio=0.25)
    assert len(split.train) + len(split.holdout) == 4
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest tests/test_split.py -xvs
```
Expected: FAIL — `ModuleNotFoundError: No module named 'teaching_knowledge.scripts.split'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `apps/evals/teaching_knowledge/scripts/split.py`:
```python
"""Stratified train/holdout split for eval dataset.

Keyed on (composer_era, skill_bucket). Holdout is never touched during
prompt iteration -- document this in every doc that mentions the harness.

Usage:
    cd apps/evals
    uv run python -m teaching_knowledge.scripts.split --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from teaching_knowledge.scripts.tag_dataset import RecordingTags


@dataclass(frozen=True)
class Split:
    train: list[str]
    holdout: list[str]


def stratified_split(
    tags: list[RecordingTags],
    seed: int,
    holdout_ratio: float = 0.2,
) -> Split:
    """Stratify on (composer_era, skill_bucket), seeded for determinism."""
    rng = random.Random(seed)
    by_stratum: dict[tuple[str, int], list[str]] = defaultdict(list)
    for tag in tags:
        key = (tag.composer_era, tag.skill_bucket)
        by_stratum[key].append(tag.recording_id)

    train: list[str] = []
    holdout: list[str] = []

    for key in sorted(by_stratum.keys()):
        ids = sorted(by_stratum[key])  # deterministic base order
        rng.shuffle(ids)
        n_holdout = max(0, round(len(ids) * holdout_ratio))
        # Never steal the only recording in a stratum for holdout
        if n_holdout == len(ids) and len(ids) > 1:
            n_holdout = len(ids) - 1
        holdout.extend(ids[:n_holdout])
        train.extend(ids[n_holdout:])

    return Split(train=sorted(train), holdout=sorted(holdout))


def write_split(split: Split, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {"train": split.train, "holdout": split.holdout},
            indent=2,
        )
    )


def load_split(split_path: Path, which: str) -> set[str]:
    """Load a split file. which in {"train","holdout","all"}."""
    if which not in {"train", "holdout", "all"}:
        raise ValueError(f"invalid split selector: {which!r}")
    blob = json.loads(split_path.read_text())
    if which == "train":
        return set(blob["train"])
    if which == "holdout":
        return set(blob["holdout"])
    return set(blob["train"]) | set(blob["holdout"])


def main() -> None:
    from teaching_knowledge.run_eval import EVALS_ROOT

    parser = argparse.ArgumentParser(description="Split dataset into train/holdout")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-ratio", type=float, default=0.2)
    parser.add_argument(
        "--dataset-index",
        type=Path,
        default=EVALS_ROOT / "teaching_knowledge" / "data" / "dataset_index.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=EVALS_ROOT / "teaching_knowledge" / "data" / "splits.json",
    )
    args = parser.parse_args()

    tags: list[RecordingTags] = []
    for line in args.dataset_index.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        tags.append(
            RecordingTags(
                recording_id=row["recording_id"],
                composer_era=row["composer_era"],
                skill_bucket=int(row["skill_bucket"]),
                duration_bucket=row["duration_bucket"],
            )
        )

    split = stratified_split(tags, seed=args.seed, holdout_ratio=args.holdout_ratio)
    write_split(split, args.out)
    print(f"wrote {args.out}")
    print(f"  train:   {len(split.train)}")
    print(f"  holdout: {len(split.holdout)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest tests/test_split.py -xvs
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teaching_knowledge/scripts/split.py apps/evals/tests/test_split.py && git commit -m "feat(evals): add stratified train/holdout split with load_split helper"
```

---

### Task T11: prompts.ts buildSynthesisFraming injects style guidance
**Group:** C (depends on T9)

**Behavior being verified:** `buildSynthesisFraming` injects a `<style_guidance>` block between `<session_data>` and `<task>` when the composer has a known era. The existing caller in `teacher.ts` is updated to pass the composer through.

**Interface under test:** `buildSynthesisFraming(sessionDurationMs, practicePattern, topMoments, drillingRecords, pieceMetadata, memoryContext, composer) -> string`.

**Files:**
- Modify: `apps/api/src/services/prompts.ts`
- Modify: `apps/api/src/services/teacher.ts` (caller passes composer)
- Test: `apps/api/src/services/prompts.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/services/prompts.test.ts
import { describe, expect, test } from "bun:test";
import { buildSynthesisFraming } from "./prompts";

describe("buildSynthesisFraming", () => {
    const pieceMetadata = {
        title: "Prelude No. 1",
        composer: "Bach",
        skill_level: 3,
    };
    const topMoments = [{ dimension: "articulation", score: 0.6 }];

    test("includes style_guidance block for Bach", () => {
        const out = buildSynthesisFraming(
            300_000,
            "continuous_play",
            topMoments,
            [],
            pieceMetadata,
            "",
            "Bach",
        );
        expect(out).toContain("<style_guidance");
        expect(out).toContain("Baroque");
        expect(out).toContain("articulation");
    });

    test("style_guidance appears between session_data and task", () => {
        const out = buildSynthesisFraming(
            300_000,
            "continuous_play",
            topMoments,
            [],
            pieceMetadata,
            "",
            "Bach",
        );
        const sessionIdx = out.indexOf("</session_data>");
        const styleIdx = out.indexOf("<style_guidance");
        const taskIdx = out.indexOf("<task>");
        expect(sessionIdx).toBeGreaterThan(-1);
        expect(styleIdx).toBeGreaterThan(sessionIdx);
        expect(taskIdx).toBeGreaterThan(styleIdx);
    });

    test("omits style_guidance for unknown composer", () => {
        const out = buildSynthesisFraming(
            300_000,
            "continuous_play",
            topMoments,
            [],
            { ...pieceMetadata, composer: "Unknown" },
            "",
            "Unknown",
        );
        expect(out).not.toContain("<style_guidance");
    });

    test("includes student_memory when given", () => {
        const out = buildSynthesisFraming(
            300_000,
            "continuous_play",
            topMoments,
            [],
            pieceMetadata,
            "Student prefers slow practice.",
            "Bach",
        );
        expect(out).toContain("<student_memory>");
        expect(out).toContain("Student prefers slow practice.");
    });

    test("Chopin resolves to Romantic era guidance", () => {
        const out = buildSynthesisFraming(
            300_000,
            "continuous_play",
            topMoments,
            [],
            { ...pieceMetadata, composer: "Chopin" },
            "",
            "Chopin",
        );
        expect(out).toContain("Romantic");
        expect(out).toContain("dynamics");
    });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun test src/services/prompts.test.ts
```
Expected: FAIL — either compile error (the function doesn't take a 7th parameter) or the `<style_guidance>` assertion fails.

- [ ] **Step 3: Implement the minimum to make the test pass**

Modify `apps/api/src/services/prompts.ts`:

1. Add the style-rules import at the top of the file (after existing top-level code, before `SESSION_SYNTHESIS_SYSTEM`):
```typescript
import styleRules from "../lib/style-rules.json";

type StyleRulesEra = {
    composer_patterns: string[];
    dimensions: Record<string, string>;
};
type StyleRulesFile = {
    eras: Record<string, StyleRulesEra>;
};

function composerToEra(composer: string): string {
    if (!composer) return "Unknown";
    const rules = styleRules as StyleRulesFile;
    const lowered = composer.toLowerCase();
    for (const [eraName, eraData] of Object.entries(rules.eras)) {
        for (const pattern of eraData.composer_patterns) {
            if (lowered.includes(pattern.toLowerCase())) {
                return eraName;
            }
        }
    }
    return "Unknown";
}

function getStyleGuidance(composer: string): string {
    const era = composerToEra(composer);
    if (era === "Unknown") return "";
    const rules = styleRules as StyleRulesFile;
    const dims = rules.eras[era].dimensions;
    const lines = [
        `<style_guidance era="${era}">`,
        `For ${era}-era repertoire, weight dimensions as follows when giving feedback:`,
    ];
    for (const [dim, rule] of Object.entries(dims)) {
        lines.push(`- ${dim}: ${rule}`);
    }
    lines.push("Advice that contradicts these rules should not be given.");
    lines.push("</style_guidance>");
    return lines.join("\n");
}
```

2. Update `buildSynthesisFraming` signature and body:
```typescript
export function buildSynthesisFraming(
    sessionDurationMs: number,
    practicePattern: unknown,
    topMoments: unknown,
    drillingRecords: unknown,
    pieceMetadata: unknown,
    memoryContext: string,
    composer: string,
): string {
    const parts: string[] = [];

    const sessionData = {
        duration_minutes: Math.round(sessionDurationMs / 60000),
        practice_pattern: practicePattern,
        top_moments: topMoments,
        drilling_records: drillingRecords,
        piece: pieceMetadata,
    };

    parts.push("<session_data>");
    parts.push(JSON.stringify(sessionData, null, 2));
    parts.push("</session_data>");

    const guidance = getStyleGuidance(composer);
    if (guidance.length > 0) {
        parts.push("");
        parts.push(guidance);
    }

    if (memoryContext.length > 0) {
        parts.push("");
        parts.push("<student_memory>");
        parts.push(memoryContext);
        parts.push("</student_memory>");
    }

    parts.push("");
    parts.push(
        "<task>Write <analysis>...</analysis> first as a reasoning scratchpad (this will be stripped before delivery). Then write your teacher response: 3-6 sentences, conversational, warm, specific. Use tools if they would add value. Do not mention scores or numbers. Do not list all dimensions -- focus on what matters most for this session.</task>",
    );

    return parts.join("\n");
}
```

3. Update the single caller in `apps/api/src/services/teacher.ts` at line 546. Change:
```typescript
const synthesisFraming = buildSynthesisFraming(
    input.sessionDurationMs,
    input.practicePattern,
    input.topMoments,
    input.drillingRecords,
    input.pieceMetadata,
    memoryContext,
);
```
to:
```typescript
const composer =
    (input.pieceMetadata as { composer?: string } | null | undefined)?.composer ?? "";
const synthesisFraming = buildSynthesisFraming(
    input.sessionDurationMs,
    input.practicePattern,
    input.topMoments,
    input.drillingRecords,
    input.pieceMetadata,
    memoryContext,
    composer,
);
```

4. Confirm `apps/api/tsconfig.json` allows JSON imports. If `resolveJsonModule` is not already `true`, add it. (It almost certainly is — `bun` defaults to allowing JSON imports. If the test still fails on JSON import, add `"resolveJsonModule": true` under `compilerOptions`.)

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun test src/services/prompts.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/services/prompts.ts apps/api/src/services/teacher.ts apps/api/src/services/prompts.test.ts && git commit -m "feat(api): inject composer-era style guidance into synthesis framing"
```

---

### Task T12: aggregate.py per-dim means + bootstrap CIs
**Group:** D (sequential, depends on T1 + T8)

**Behavior being verified:** Given a fixture JSONL and dataset_index, `aggregate_run` returns an `AggregateResult` with correct per-dim means, bootstrap CI tuples, per-era and per-skill breakdowns, and a composite score.

**Interface under test:** `teaching_knowledge.scripts.aggregate.aggregate_run`, `AggregateResult`, `DimensionAggregate`.

**Files:**
- Create: `apps/evals/teaching_knowledge/scripts/aggregate.py`
- Test: `apps/evals/tests/test_aggregate.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/tests/test_aggregate.py
from __future__ import annotations

import json
from pathlib import Path

from teaching_knowledge.scripts.aggregate import (
    AggregateResult,
    DimensionAggregate,
    aggregate_run,
)


def _fixture_row(
    recording_id: str,
    dim_scores: dict[str, tuple[int, int]],
    run_id: str = "test_run_id",
) -> dict:
    """Build a fixture JSONL row in the shape run_eval.py emits."""
    return {
        "recording_id": recording_id,
        "run_id": run_id,
        "git_sha": "abc1234",
        "judge_dimensions": [
            {
                "criterion": crit,
                "process": p,
                "outcome": o,
                "score": min(p, o),
                "evidence": "",
                "reason": "",
            }
            for crit, (p, o) in dim_scores.items()
        ],
        "error": "",
    }


def _fixture_index_row(recording_id: str, era: str, skill: int) -> dict:
    return {
        "recording_id": recording_id,
        "composer_era": era,
        "skill_bucket": skill,
        "duration_bucket": "30-60s",
    }


def test_aggregate_run_computes_per_dim_means(tmp_path: Path) -> None:
    jsonl = tmp_path / "baseline.jsonl"
    index = tmp_path / "dataset_index.jsonl"

    rows = [
        _fixture_row("r1", {"Audible-Specific Corrective Feedback": (3, 2), "Specific Positive Praise": (2, 2)}),
        _fixture_row("r2", {"Audible-Specific Corrective Feedback": (2, 1), "Specific Positive Praise": (3, 3)}),
        _fixture_row("r3", {"Audible-Specific Corrective Feedback": (1, 1), "Specific Positive Praise": (2, 2)}),
        _fixture_row("r4", {"Audible-Specific Corrective Feedback": (3, 3), "Specific Positive Praise": (3, 2)}),
        _fixture_row("r5", {"Audible-Specific Corrective Feedback": (2, 2), "Specific Positive Praise": (1, 1)}),
    ]
    jsonl.write_text("\n".join(json.dumps(r) for r in rows))

    index_rows = [
        _fixture_index_row("r1", "Baroque", 3),
        _fixture_index_row("r2", "Baroque", 3),
        _fixture_index_row("r3", "Romantic", 5),
        _fixture_index_row("r4", "Romantic", 5),
        _fixture_index_row("r5", "Romantic", 5),
    ]
    index.write_text("\n".join(json.dumps(r) for r in index_rows))

    result = aggregate_run(jsonl, index)

    assert isinstance(result, AggregateResult)
    assert result.total_rows == 5
    assert result.run_id == "test_run_id"

    dims_by_name = {d.name: d for d in result.dimensions}
    asc = dims_by_name["Audible-Specific Corrective Feedback"]
    # Process means: (3+2+1+3+2)/5 = 2.2
    assert abs(asc.mean_process - 2.2) < 0.001
    # Outcome means: (2+1+1+3+2)/5 = 1.8
    assert abs(asc.mean_outcome - 1.8) < 0.001
    assert asc.n == 5


def test_aggregate_run_computes_stratified_breakdowns(tmp_path: Path) -> None:
    jsonl = tmp_path / "baseline.jsonl"
    index = tmp_path / "dataset_index.jsonl"

    rows = [
        _fixture_row("r1", {"Specific Positive Praise": (3, 3)}),
        _fixture_row("r2", {"Specific Positive Praise": (3, 3)}),
        _fixture_row("r3", {"Specific Positive Praise": (1, 1)}),
        _fixture_row("r4", {"Specific Positive Praise": (1, 1)}),
        _fixture_row("r5", {"Specific Positive Praise": (2, 2)}),
    ]
    jsonl.write_text("\n".join(json.dumps(r) for r in rows))

    index_rows = [
        _fixture_index_row("r1", "Baroque", 3),
        _fixture_index_row("r2", "Baroque", 3),
        _fixture_index_row("r3", "Romantic", 5),
        _fixture_index_row("r4", "Romantic", 5),
        _fixture_index_row("r5", "Romantic", 5),
    ]
    index.write_text("\n".join(json.dumps(r) for r in index_rows))

    result = aggregate_run(jsonl, index)

    assert "Baroque" in result.by_era
    assert "Romantic" in result.by_era
    baroque_ppraise = result.by_era["Baroque"]["Specific Positive Praise"]
    romantic_ppraise = result.by_era["Romantic"]["Specific Positive Praise"]
    assert abs(baroque_ppraise - 3.0) < 0.001  # (3+3)/2
    # (1+1+2)/3 = 1.333
    assert abs(romantic_ppraise - (4 / 3)) < 0.001


def test_aggregate_skips_error_rows(tmp_path: Path) -> None:
    jsonl = tmp_path / "baseline.jsonl"
    index = tmp_path / "dataset_index.jsonl"

    rows = [
        _fixture_row("r1", {"Specific Positive Praise": (3, 3)}),
        {"recording_id": "r2", "run_id": "test_run_id", "git_sha": "x", "error": "boom", "judge_dimensions": []},
        _fixture_row("r3", {"Specific Positive Praise": (2, 2)}),
        _fixture_row("r4", {"Specific Positive Praise": (2, 2)}),
        _fixture_row("r5", {"Specific Positive Praise": (3, 3)}),
        _fixture_row("r6", {"Specific Positive Praise": (2, 2)}),
    ]
    jsonl.write_text("\n".join(json.dumps(r) for r in rows))
    index.write_text("\n".join(
        json.dumps(_fixture_index_row(f"r{i+1}", "Baroque", 3)) for i in range(6)
    ))

    result = aggregate_run(jsonl, index)
    assert result.total_rows == 5  # error row excluded


def test_aggregate_produces_bootstrap_ci_when_enough_samples(tmp_path: Path) -> None:
    jsonl = tmp_path / "baseline.jsonl"
    index = tmp_path / "dataset_index.jsonl"

    rows = [_fixture_row(f"r{i}", {"Specific Positive Praise": (2, 2)}) for i in range(10)]
    jsonl.write_text("\n".join(json.dumps(r) for r in rows))
    index.write_text("\n".join(
        json.dumps(_fixture_index_row(f"r{i}", "Baroque", 3)) for i in range(10)
    ))

    result = aggregate_run(jsonl, index)
    ppraise = next(d for d in result.dimensions if d.name == "Specific Positive Praise")
    assert ppraise.ci_process is not None
    low, high = ppraise.ci_process
    assert low <= 2.0 <= high


def test_aggregate_ci_is_none_for_tiny_samples(tmp_path: Path) -> None:
    jsonl = tmp_path / "baseline.jsonl"
    index = tmp_path / "dataset_index.jsonl"

    rows = [_fixture_row(f"r{i}", {"Specific Positive Praise": (2, 2)}) for i in range(3)]
    jsonl.write_text("\n".join(json.dumps(r) for r in rows))
    index.write_text("\n".join(
        json.dumps(_fixture_index_row(f"r{i}", "Baroque", 3)) for i in range(3)
    ))

    result = aggregate_run(jsonl, index)
    ppraise = next(d for d in result.dimensions if d.name == "Specific Positive Praise")
    assert ppraise.ci_process is None  # N < 5 triggers None from bootstrap_ci
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest tests/test_aggregate.py -xvs
```
Expected: FAIL — `ModuleNotFoundError: No module named 'teaching_knowledge.scripts.aggregate'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `apps/evals/teaching_knowledge/scripts/aggregate.py`:
```python
"""Reduce a run JSONL to per-dim means + bootstrap CIs + stratified breakdowns.

Reads the output of run_eval.py, joins against dataset_index.jsonl for
composer_era and skill_bucket tags, and writes a single aggregate JSON.

Usage:
    cd apps/evals
    uv run python -m teaching_knowledge.scripts.aggregate \\
        results/baseline.jsonl --out results/baseline_aggregate.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from shared.stats import bootstrap_ci


@dataclass
class DimensionAggregate:
    name: str
    mean_process: float | None
    ci_process: tuple[float, float] | None
    mean_outcome: float | None
    ci_outcome: tuple[float, float] | None
    n: int


@dataclass
class AggregateResult:
    dimensions: list[DimensionAggregate]
    composite_mean: float
    composite_ci: tuple[float, float] | None
    by_era: dict[str, dict[str, float]] = field(default_factory=dict)
    by_skill: dict[int, dict[str, float]] = field(default_factory=dict)
    total_rows: int = 0
    run_id: str = ""


def _load_index(path: Path) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        index[row["recording_id"]] = row
    return index


def _iter_rows(jsonl_path: Path):
    for line in jsonl_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        yield json.loads(line)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def aggregate_run(jsonl_path: Path, dataset_index_path: Path) -> AggregateResult:
    index = _load_index(dataset_index_path)

    dim_process: dict[str, list[float]] = defaultdict(list)
    dim_outcome: dict[str, list[float]] = defaultdict(list)
    all_composite: list[float] = []

    by_era: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_skill: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    total_rows = 0
    run_id = ""

    for row in _iter_rows(jsonl_path):
        if row.get("error"):
            continue
        if not run_id:
            run_id = row.get("run_id", "")
        rec_id = row["recording_id"]
        tags = index.get(rec_id)

        dims = row.get("judge_dimensions", [])
        if not dims:
            continue
        total_rows += 1

        row_scores: list[float] = []
        for dim in dims:
            crit = dim.get("criterion", "unknown")
            proc = dim.get("process")
            out = dim.get("outcome")
            if proc is not None:
                dim_process[crit].append(float(proc))
            if out is not None:
                dim_outcome[crit].append(float(out))
            if dim.get("score") is not None:
                row_scores.append(float(dim["score"]))
                if tags:
                    by_era[tags["composer_era"]][crit].append(float(dim["score"]))
                    by_skill[int(tags["skill_bucket"])][crit].append(float(dim["score"]))

        if row_scores:
            all_composite.append(sum(row_scores) / len(row_scores))

    all_crits = sorted(set(dim_process.keys()) | set(dim_outcome.keys()))
    dimensions: list[DimensionAggregate] = []
    for crit in all_crits:
        procs = dim_process.get(crit, [])
        outs = dim_outcome.get(crit, [])
        dimensions.append(
            DimensionAggregate(
                name=crit,
                mean_process=_mean(procs),
                ci_process=bootstrap_ci(procs),
                mean_outcome=_mean(outs),
                ci_outcome=bootstrap_ci(outs),
                n=max(len(procs), len(outs)),
            )
        )

    composite_mean = _mean(all_composite) or 0.0
    composite_ci = bootstrap_ci(all_composite)

    by_era_final: dict[str, dict[str, float]] = {
        era: {crit: sum(vals) / len(vals) for crit, vals in crits.items() if vals}
        for era, crits in by_era.items()
    }
    by_skill_final: dict[int, dict[str, float]] = {
        skill: {crit: sum(vals) / len(vals) for crit, vals in crits.items() if vals}
        for skill, crits in by_skill.items()
    }

    return AggregateResult(
        dimensions=dimensions,
        composite_mean=composite_mean,
        composite_ci=composite_ci,
        by_era=by_era_final,
        by_skill=by_skill_final,
        total_rows=total_rows,
        run_id=run_id,
    )


def write_aggregate(result: AggregateResult, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(result), indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate a run JSONL")
    parser.add_argument("jsonl", type=Path, help="Path to run JSONL")
    parser.add_argument(
        "--dataset-index",
        type=Path,
        default=Path("teaching_knowledge/data/dataset_index.jsonl"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (default: <jsonl>_aggregate.json)",
    )
    args = parser.parse_args()

    out = args.out or args.jsonl.with_name(args.jsonl.stem + "_aggregate.json")
    result = aggregate_run(args.jsonl, args.dataset_index)
    write_aggregate(result, out)
    print(f"wrote {out}")
    print(f"  composite_mean: {result.composite_mean:.3f}")
    print(f"  total_rows:     {result.total_rows}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest tests/test_aggregate.py -xvs
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teaching_knowledge/scripts/aggregate.py apps/evals/tests/test_aggregate.py && git commit -m "feat(evals): aggregate.py per-dim means + bootstrap CIs + stratified breakdowns"
```

---

### Task T13: regression_check.py
**Group:** E (sequential, depends on T12)

**Behavior being verified:** Given two `AggregateResult` instances, `check_regression` correctly identifies regressed dimensions (CI non-overlap + negative delta), improved dimensions (CI non-overlap + positive delta), and null deltas (CIs overlap).

**Interface under test:** `teaching_knowledge.scripts.regression_check.check_regression`, `RegressionReport`, `DimensionRegression`.

**Files:**
- Create: `apps/evals/teaching_knowledge/scripts/regression_check.py`
- Test: `apps/evals/tests/test_regression_check.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/tests/test_regression_check.py
from __future__ import annotations

from teaching_knowledge.scripts.aggregate import AggregateResult, DimensionAggregate
from teaching_knowledge.scripts.regression_check import (
    DimensionRegression,
    RegressionReport,
    check_regression,
    format_report,
)


def _agg(
    dim_data: dict[str, tuple[float, tuple[float, float]]],
    composite: float = 2.0,
    composite_ci: tuple[float, float] | None = (1.9, 2.1),
) -> AggregateResult:
    dims = [
        DimensionAggregate(
            name=name,
            mean_process=mean,
            ci_process=ci,
            mean_outcome=mean,
            ci_outcome=ci,
            n=20,
        )
        for name, (mean, ci) in dim_data.items()
    ]
    return AggregateResult(
        dimensions=dims,
        composite_mean=composite,
        composite_ci=composite_ci,
        by_era={},
        by_skill={},
        total_rows=20,
        run_id="test",
    )


def test_flags_regression_when_ci_does_not_overlap() -> None:
    baseline = _agg({"Style": (2.5, (2.3, 2.7))})
    candidate = _agg({"Style": (1.5, (1.3, 1.7))})
    report = check_regression(baseline, candidate)
    style = next(d for d in report.dimensions if d.name == "Style")
    assert style.direction == "regressed"
    assert style.significant is True
    assert report.has_regression is True


def test_flags_improvement_when_ci_does_not_overlap() -> None:
    baseline = _agg({"Style": (1.5, (1.3, 1.7))})
    candidate = _agg({"Style": (2.5, (2.3, 2.7))})
    report = check_regression(baseline, candidate)
    style = next(d for d in report.dimensions if d.name == "Style")
    assert style.direction == "improved"
    assert style.significant is True
    assert report.has_regression is False


def test_null_delta_when_ci_overlaps() -> None:
    baseline = _agg({"Style": (2.0, (1.8, 2.2))})
    candidate = _agg({"Style": (2.1, (1.9, 2.3))})
    report = check_regression(baseline, candidate)
    style = next(d for d in report.dimensions if d.name == "Style")
    assert style.direction == "null"
    assert style.significant is False


def test_composite_delta_reported() -> None:
    baseline = _agg({"S": (2.0, (1.8, 2.2))}, composite=2.0, composite_ci=(1.9, 2.1))
    candidate = _agg({"S": (2.3, (2.1, 2.5))}, composite=2.3, composite_ci=(2.2, 2.4))
    report = check_regression(baseline, candidate)
    assert abs(report.composite_delta - 0.3) < 0.001
    assert report.composite_significant is True


def test_format_report_contains_dimension_names() -> None:
    baseline = _agg({"Alpha": (2.0, (1.8, 2.2)), "Beta": (1.5, (1.3, 1.7))})
    candidate = _agg({"Alpha": (2.1, (1.9, 2.3)), "Beta": (2.5, (2.3, 2.7))})
    text = format_report(check_regression(baseline, candidate))
    assert "Alpha" in text
    assert "Beta" in text
    assert "improved" in text.lower() or "regressed" in text.lower()
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest tests/test_regression_check.py -xvs
```
Expected: FAIL — `ModuleNotFoundError: No module named 'teaching_knowledge.scripts.regression_check'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `apps/evals/teaching_knowledge/scripts/regression_check.py`:
```python
"""Diff two aggregate results, flag dimensions that regressed.

Two aggregates are compared via CI overlap. A dimension is:
  - regressed if candidate mean < baseline mean AND CIs do not overlap
  - improved  if candidate mean > baseline mean AND CIs do not overlap
  - null      if CIs overlap

Usage:
    cd apps/evals
    uv run python -m teaching_knowledge.scripts.regression_check \\
        baseline_aggregate.json candidate_aggregate.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from teaching_knowledge.scripts.aggregate import AggregateResult, DimensionAggregate


@dataclass
class DimensionRegression:
    name: str
    baseline_mean: float
    candidate_mean: float
    delta: float
    significant: bool
    direction: str  # "regressed" | "improved" | "null"


@dataclass
class RegressionReport:
    dimensions: list[DimensionRegression]
    composite_delta: float
    composite_significant: bool
    has_regression: bool


def _ci_overlap(
    a: tuple[float, float] | None,
    b: tuple[float, float] | None,
) -> bool:
    if a is None or b is None:
        return True  # cannot determine significance -> treat as overlap
    return not (a[1] < b[0] or b[1] < a[0])


def _compare(
    name: str,
    baseline: DimensionAggregate,
    candidate: DimensionAggregate,
) -> DimensionRegression:
    b_mean = baseline.mean_process or 0.0
    c_mean = candidate.mean_process or 0.0
    delta = c_mean - b_mean
    overlaps = _ci_overlap(baseline.ci_process, candidate.ci_process)
    significant = not overlaps
    if not significant:
        direction = "null"
    elif delta < 0:
        direction = "regressed"
    else:
        direction = "improved"
    return DimensionRegression(
        name=name,
        baseline_mean=b_mean,
        candidate_mean=c_mean,
        delta=delta,
        significant=significant,
        direction=direction,
    )


def check_regression(
    baseline: AggregateResult,
    candidate: AggregateResult,
) -> RegressionReport:
    baseline_by_name = {d.name: d for d in baseline.dimensions}
    candidate_by_name = {d.name: d for d in candidate.dimensions}
    names = sorted(set(baseline_by_name) | set(candidate_by_name))

    dims: list[DimensionRegression] = []
    for name in names:
        b = baseline_by_name.get(name)
        c = candidate_by_name.get(name)
        if b is None or c is None:
            continue
        dims.append(_compare(name, b, c))

    composite_delta = candidate.composite_mean - baseline.composite_mean
    composite_significant = not _ci_overlap(baseline.composite_ci, candidate.composite_ci)
    has_regression = any(d.direction == "regressed" for d in dims)

    return RegressionReport(
        dimensions=dims,
        composite_delta=composite_delta,
        composite_significant=composite_significant,
        has_regression=has_regression,
    )


def format_report(report: RegressionReport) -> str:
    lines: list[str] = []
    lines.append("=" * 64)
    lines.append("REGRESSION CHECK")
    lines.append("=" * 64)
    lines.append(
        f"composite delta: {report.composite_delta:+.3f}  "
        f"({'SIG' if report.composite_significant else 'null'})"
    )
    lines.append("")
    lines.append(f"{'Dimension':<45} {'delta':>8} {'direction':>12}")
    lines.append("-" * 68)
    for dim in report.dimensions:
        lines.append(
            f"{dim.name[:45]:<45} {dim.delta:+8.3f} {dim.direction:>12}"
            + ("  *" if dim.significant else "")
        )
    lines.append("")
    if report.has_regression:
        lines.append("!! REGRESSION DETECTED")
    else:
        lines.append("OK: no significant regressions")
    return "\n".join(lines)


def _load_aggregate(path: Path) -> AggregateResult:
    blob = json.loads(path.read_text())
    dims = [
        DimensionAggregate(
            name=d["name"],
            mean_process=d.get("mean_process"),
            ci_process=tuple(d["ci_process"]) if d.get("ci_process") else None,
            mean_outcome=d.get("mean_outcome"),
            ci_outcome=tuple(d["ci_outcome"]) if d.get("ci_outcome") else None,
            n=d.get("n", 0),
        )
        for d in blob["dimensions"]
    ]
    return AggregateResult(
        dimensions=dims,
        composite_mean=blob["composite_mean"],
        composite_ci=tuple(blob["composite_ci"]) if blob.get("composite_ci") else None,
        by_era=blob.get("by_era", {}),
        by_skill=blob.get("by_skill", {}),
        total_rows=blob.get("total_rows", 0),
        run_id=blob.get("run_id", ""),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Regression check between two runs")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    baseline = _load_aggregate(args.baseline)
    candidate = _load_aggregate(args.candidate)
    report = check_regression(baseline, candidate)
    print(format_report(report))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest tests/test_regression_check.py -xvs
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teaching_knowledge/scripts/regression_check.py apps/evals/tests/test_regression_check.py && git commit -m "feat(evals): add regression_check.py for run-vs-run dim comparison"
```

---

### Task T14: run_eval.py inject style guidance
**Group:** F (sequential, same-file edits to run_eval.py)

**Behavior being verified:** `build_synthesis_user_msg` injects a `<style_guidance>` block between `<session_data>` and `<task>` when the composer has a known era.

**Interface under test:** `teaching_knowledge.run_eval.build_synthesis_user_msg(muq_means, duration_seconds, meta) -> str`.

**Files:**
- Modify: `apps/evals/teaching_knowledge/run_eval.py`
- Test: `apps/evals/tests/test_run_eval_style_injection.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/tests/test_run_eval_style_injection.py
from __future__ import annotations

from teaching_knowledge.run_eval import build_synthesis_user_msg


def test_bach_injects_baroque_guidance() -> None:
    meta = {
        "piece_slug": "bach_wtc",
        "title": "WTC Prelude 1",
        "composer": "Bach",
        "skill_bucket": 3,
    }
    msg = build_synthesis_user_msg(
        muq_means={"articulation": 0.5},
        duration_seconds=60.0,
        meta=meta,
    )
    assert "<style_guidance" in msg
    assert "Baroque" in msg
    assert "articulation" in msg.lower()


def test_style_guidance_between_session_data_and_task() -> None:
    meta = {
        "piece_slug": "chopin_ballade",
        "title": "Ballade 1",
        "composer": "Chopin",
        "skill_bucket": 5,
    }
    msg = build_synthesis_user_msg(
        muq_means={"dynamics": 0.7},
        duration_seconds=90.0,
        meta=meta,
    )
    sess_end = msg.index("</session_data>")
    guidance_idx = msg.index("<style_guidance")
    task_idx = msg.index("<task>")
    assert sess_end < guidance_idx < task_idx


def test_unknown_composer_omits_style_block() -> None:
    meta = {
        "piece_slug": "unk",
        "title": "Unknown",
        "composer": "Nobody",
        "skill_bucket": 3,
    }
    msg = build_synthesis_user_msg(
        muq_means={"dynamics": 0.5},
        duration_seconds=30.0,
        meta=meta,
    )
    assert "<style_guidance" not in msg


def test_existing_session_data_still_present() -> None:
    meta = {
        "piece_slug": "bach",
        "title": "WTC",
        "composer": "Bach",
        "skill_bucket": 3,
    }
    msg = build_synthesis_user_msg(
        muq_means={"articulation": 0.5},
        duration_seconds=60.0,
        meta=meta,
    )
    assert "<session_data>" in msg
    assert "WTC" in msg
    assert "<task>" in msg
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest tests/test_run_eval_style_injection.py -xvs
```
Expected: FAIL — no `<style_guidance>` in the output.

- [ ] **Step 3: Implement the minimum to make the test pass**

Modify `apps/evals/teaching_knowledge/run_eval.py`:

1. Add import at the top with the other imports:
```python
from shared.style_rules import get_style_guidance
```

2. Replace the return statement at the end of `build_synthesis_user_msg` (the existing `return "\n".join([...])` block) with:
```python
    guidance = get_style_guidance(meta.get("composer", ""))

    parts: list[str] = [
        "<session_data>",
        json.dumps(session_data, indent=2),
        "</session_data>",
    ]
    if guidance:
        parts.append("")
        parts.append(guidance)
    parts.append("")
    parts.append(
        "<task>Write <analysis>...</analysis> first as a reasoning scratchpad "
        "(this will be stripped). Then write your teacher response: 3-6 sentences, "
        "conversational, warm, specific. Do not mention scores or numbers. Focus on "
        "what matters most for this session.</task>"
    )
    return "\n".join(parts)
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest tests/test_run_eval_style_injection.py -xvs
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teaching_knowledge/run_eval.py apps/evals/tests/test_run_eval_style_injection.py && git commit -m "feat(evals): run_eval.py injects composer-era style guidance"
```

---

### Task T15: run_eval.py --split flag
**Group:** F (sequential, after T14)

**Behavior being verified:** `run(split="holdout", split_path=...)` filters cache files down to the holdout set; `split="train"` excludes the holdout; `split="all"` includes everything. A fixture splits.json drives the test.

**Interface under test:** `teaching_knowledge.run_eval.run(split, split_path, ...)` — we test the filtering logic via a new helper `_filter_cache_files_by_split` extracted during this task.

**Files:**
- Modify: `apps/evals/teaching_knowledge/run_eval.py`
- Test: `apps/evals/tests/test_run_eval_split_flag.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/tests/test_run_eval_split_flag.py
from __future__ import annotations

import json
from pathlib import Path

import pytest

from teaching_knowledge.run_eval import _filter_cache_files_by_split


def _fake_cache_file(tmp: Path, name: str) -> Path:
    p = tmp / f"{name}.json"
    p.write_text("{}")
    return p


def test_filter_returns_only_train_set(tmp_path: Path) -> None:
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"train": ["a", "b", "c"], "holdout": ["d", "e"]}))
    files = [_fake_cache_file(tmp_path, n) for n in ["a", "b", "c", "d", "e"]]

    filtered = _filter_cache_files_by_split(files, splits, which="train")
    names = sorted(f.stem for f in filtered)
    assert names == ["a", "b", "c"]


def test_filter_returns_only_holdout_set(tmp_path: Path) -> None:
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"train": ["a", "b"], "holdout": ["c", "d"]}))
    files = [_fake_cache_file(tmp_path, n) for n in ["a", "b", "c", "d"]]
    filtered = _filter_cache_files_by_split(files, splits, which="holdout")
    assert sorted(f.stem for f in filtered) == ["c", "d"]


def test_filter_all_returns_everything(tmp_path: Path) -> None:
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"train": ["a"], "holdout": ["b"]}))
    files = [_fake_cache_file(tmp_path, n) for n in ["a", "b", "c"]]
    # "c" is not in the split; "all" returns only items present in the split
    filtered = _filter_cache_files_by_split(files, splits, which="all")
    assert sorted(f.stem for f in filtered) == ["a", "b"]


def test_filter_with_no_split_path_returns_all_files(tmp_path: Path) -> None:
    files = [_fake_cache_file(tmp_path, n) for n in ["a", "b"]]
    filtered = _filter_cache_files_by_split(files, None, which="all")
    assert sorted(f.stem for f in filtered) == ["a", "b"]


def test_filter_rejects_invalid_which(tmp_path: Path) -> None:
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"train": [], "holdout": []}))
    with pytest.raises(ValueError):
        _filter_cache_files_by_split([], splits, which="nonsense")
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest tests/test_run_eval_split_flag.py -xvs
```
Expected: FAIL — `ImportError: cannot import name '_filter_cache_files_by_split'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Modify `apps/evals/teaching_knowledge/run_eval.py`:

1. Add the helper function after `load_completed_ids` (before `def run`):
```python
def _filter_cache_files_by_split(
    cache_files: list[Path],
    split_path: Path | None,
    which: str,
) -> list[Path]:
    """Filter cache files by split membership.

    When split_path is None, returns the full list unchanged.
    When which == "all", returns only files whose stem is in (train + holdout).
    """
    if split_path is None:
        return cache_files
    from teaching_knowledge.scripts.split import load_split

    allowed = load_split(split_path, which=which)
    return [f for f in cache_files if f.stem in allowed]
```

2. Update `run()` signature to accept the new params and apply the filter:
```python
def run(
    limit: int | None = None,
    out_path: Path | None = None,
    dry_run: bool = False,
    split: str = "all",
    split_path: Path | None = None,
) -> None:
```

3. After the existing `cache_files = [...]` line and the `print(f"Cache files: ...)` line, add:
```python
    cache_files = _filter_cache_files_by_split(cache_files, split_path, which=split)
    print(f"After split filter ({split}): {len(cache_files)}")
```

4. Update `main()` to accept the new CLI flags:
```python
    parser.add_argument(
        "--split",
        choices=["train", "holdout", "all"],
        default="all",
        help="Filter recordings by split membership (default: all)",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=None,
        help="Path to splits.json (default: data/splits.json if present)",
    )
```

5. Update the final `run(...)` call in `main()` to:
```python
    default_split_file = EVALS_ROOT / "teaching_knowledge" / "data" / "splits.json"
    split_path = args.split_file
    if split_path is None and args.split != "all" and default_split_file.exists():
        split_path = default_split_file
    run(
        limit=args.limit,
        out_path=args.out,
        dry_run=args.dry_run,
        split=args.split,
        split_path=split_path,
    )
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest tests/test_run_eval_split_flag.py -xvs
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teaching_knowledge/run_eval.py apps/evals/tests/test_run_eval_split_flag.py && git commit -m "feat(evals): run_eval.py --split flag with load_split filter"
```

---

### Task T16: run_eval.py provenance stamping
**Group:** F (sequential, after T15)

**Behavior being verified:** Every row written to the output JSONL contains `run_id` and `git_sha` fields.

**Interface under test:** `teaching_knowledge.run_eval._build_row(...)` — a new helper extracted during this task that constructs the output-row dict. We test it produces rows containing `run_id` and `git_sha`.

**Files:**
- Modify: `apps/evals/teaching_knowledge/run_eval.py`
- Test: `apps/evals/tests/test_run_eval_provenance.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/tests/test_run_eval_provenance.py
from __future__ import annotations

from shared.provenance import RunProvenance
from teaching_knowledge.run_eval import _build_row


def test_build_row_includes_run_id_and_git_sha() -> None:
    prov = RunProvenance(run_id="2026-04-14T12-00-00Z_abc1234", git_sha="abc1234deadbeef", git_dirty=False)
    meta = {
        "piece_slug": "bach",
        "title": "WTC",
        "composer": "Bach",
        "skill_bucket": 3,
    }
    row = _build_row(
        recording_id="rec123",
        meta=meta,
        muq_means={"articulation": 0.5},
        synthesis_text="Nice articulation!",
        synthesis_latency_ms=100,
        judge_dimensions=[],
        judge_model="test-judge",
        judge_latency_ms=50,
        error="",
        provenance=prov,
    )
    assert row["run_id"] == "2026-04-14T12-00-00Z_abc1234"
    assert row["git_sha"] == "abc1234deadbeef"
    assert row["git_dirty"] is False
    assert row["recording_id"] == "rec123"
    assert row["synthesis_text"] == "Nice articulation!"


def test_build_row_marks_dirty_tree() -> None:
    prov = RunProvenance(run_id="x", git_sha="y", git_dirty=True)
    row = _build_row(
        recording_id="r1",
        meta={"piece_slug": "p", "title": "t", "composer": "Bach", "skill_bucket": 3},
        muq_means={},
        synthesis_text="",
        synthesis_latency_ms=0,
        judge_dimensions=[],
        judge_model="",
        judge_latency_ms=0,
        error="boom",
        provenance=prov,
    )
    assert row["git_dirty"] is True
    assert row["error"] == "boom"
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest tests/test_run_eval_provenance.py -xvs
```
Expected: FAIL — `ImportError: cannot import name '_build_row'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Modify `apps/evals/teaching_knowledge/run_eval.py`:

1. Add import at the top:
```python
from shared.provenance import RunProvenance, make_run_provenance
```

2. Add `_build_row` helper function after `_filter_cache_files_by_split`:
```python
def _build_row(
    recording_id: str,
    meta: dict,
    muq_means: dict[str, float],
    synthesis_text: str,
    synthesis_latency_ms: int,
    judge_dimensions: list[dict],
    judge_model: str,
    judge_latency_ms: int,
    error: str,
    provenance: RunProvenance,
) -> dict:
    """Build a single output-JSONL row with provenance stamped in."""
    return {
        "recording_id": recording_id,
        "run_id": provenance.run_id,
        "git_sha": provenance.git_sha,
        "git_dirty": provenance.git_dirty,
        "piece_slug": meta["piece_slug"],
        "title": meta["title"],
        "composer": meta["composer"],
        "skill_bucket": meta["skill_bucket"],
        "muq_means": muq_means,
        "synthesis_text": synthesis_text,
        "synthesis_latency_ms": synthesis_latency_ms,
        "judge_dimensions": judge_dimensions,
        "judge_model": judge_model,
        "judge_latency_ms": judge_latency_ms,
        "error": error,
    }
```

3. In `run()`, before the `for cache_path in cache_files:` loop, add:
```python
    provenance = make_run_provenance()
    print(f"run_id: {provenance.run_id}")
    print(f"git_sha: {provenance.git_sha}{' (dirty)' if provenance.git_dirty else ''}")
```

4. Replace the three inline `result = {...}` constructions in the existing loop (dry-run path, judge path, exception path) with calls to `_build_row(...)`:

Dry-run path — replace the existing `result = { ... }` block with:
```python
                    result = _build_row(
                        recording_id=recording_id,
                        meta=meta,
                        muq_means=muq_means,
                        synthesis_text=synthesis_text,
                        synthesis_latency_ms=round(synthesis_latency_ms),
                        judge_dimensions=[],
                        judge_model="dry_run",
                        judge_latency_ms=0,
                        error="",
                        provenance=provenance,
                    )
```

Judge path — replace with:
```python
                    result = _build_row(
                        recording_id=recording_id,
                        meta=meta,
                        muq_means=muq_means,
                        synthesis_text=synthesis_text,
                        synthesis_latency_ms=round(synthesis_latency_ms),
                        judge_dimensions=[
                            {
                                "criterion": d.criterion,
                                "process": d.process,
                                "outcome": d.outcome,
                                "score": d.score,
                                "evidence": d.evidence,
                                "reason": d.reason,
                            }
                            for d in judge_result.dimensions
                        ],
                        judge_model=judge_result.model,
                        judge_latency_ms=round(judge_result.latency_ms),
                        error="",
                        provenance=provenance,
                    )
```

Exception path — replace with:
```python
            except Exception as exc:
                errors += 1
                result = _build_row(
                    recording_id=recording_id,
                    meta=meta,
                    muq_means=muq_means,
                    synthesis_text="",
                    synthesis_latency_ms=0,
                    judge_dimensions=[],
                    judge_model="",
                    judge_latency_ms=0,
                    error=str(exc),
                    provenance=provenance,
                )
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest tests/test_run_eval_provenance.py -xvs
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teaching_knowledge/run_eval.py apps/evals/tests/test_run_eval_provenance.py && git commit -m "feat(evals): run_eval.py stamps run_id + git_sha on every output row"
```

---

### Task T17: run_eval.py judge-family compatibility assert + model flags
**Group:** F (sequential, after T16)

**Behavior being verified:** `run(teacher_model=X, judge_model=Y)` raises `ValueError` when X and Y share a family; it succeeds when they cross families.

**Interface under test:** `teaching_knowledge.run_eval.run(teacher_model, judge_model, ...)`.

**Files:**
- Modify: `apps/evals/teaching_knowledge/run_eval.py`
- Test: `apps/evals/tests/test_run_eval_judge_family.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/tests/test_run_eval_judge_family.py
from __future__ import annotations

from pathlib import Path

import pytest

from teaching_knowledge.run_eval import _assert_models_compatible


def test_run_raises_when_teacher_and_judge_same_family() -> None:
    with pytest.raises(ValueError, match="anthropic"):
        _assert_models_compatible("claude-sonnet-4-6", "claude-sonnet-4-6")


def test_run_raises_when_teacher_and_openrouter_judge_same_family() -> None:
    with pytest.raises(ValueError, match="anthropic"):
        _assert_models_compatible("claude-sonnet-4-6", "anthropic/claude-sonnet-4-6")


def test_run_allows_cross_family() -> None:
    _assert_models_compatible("claude-sonnet-4-6", "@cf/google/gemma-4-26b-a4b-it")
    _assert_models_compatible("claude-sonnet-4-6", "openai/gpt-5.4-mini")
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest tests/test_run_eval_judge_family.py -xvs
```
Expected: FAIL — `ImportError: cannot import name '_assert_models_compatible'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Modify `apps/evals/teaching_knowledge/run_eval.py`:

1. Add import:
```python
from shared.judge_compatibility import assert_judge_compatible
```

2. Add a thin module-level wrapper (makes the function trivially testable without invoking `run()`):
```python
def _assert_models_compatible(teacher_model: str, judge_model: str) -> None:
    assert_judge_compatible(teacher_model, judge_model)
```

3. Extend `run()` signature to accept the new params:
```python
def run(
    limit: int | None = None,
    out_path: Path | None = None,
    dry_run: bool = False,
    split: str = "all",
    split_path: Path | None = None,
    teacher_model: str = "claude-sonnet-4-6",
    judge_model: str = "@cf/google/gemma-4-26b-a4b-it",
) -> None:
```

4. At the top of the `run()` body (immediately after the `from ... import ...` lines), add:
```python
    if not dry_run:
        _assert_models_compatible(teacher_model, judge_model)
```

5. Update the synthesis client construction to respect `teacher_model`:
```python
    synthesis_client = LLMClient(provider="anthropic", model=teacher_model)
    print(f"Synthesis: {synthesis_client.model}")
    if not dry_run:
        print(f"Judge:     {judge_model}")
```

6. Update the `judge_synthesis_v2(...)` call inside the loop to pass the judge model. The current call is:
```python
                    judge_result = judge_synthesis_v2(
                        synthesis_text=synthesis_text,
                        context=judge_context,
                        provider="workers-ai",
                    )
```
Change to:
```python
                    judge_provider = "openrouter" if "/" in judge_model and not judge_model.startswith("@cf/") else "workers-ai"
                    judge_result = judge_synthesis_v2(
                        synthesis_text=synthesis_text,
                        context=judge_context,
                        provider=judge_provider,
                        model=judge_model,
                    )
```

7. Extend `main()` with the new CLI flags:
```python
    parser.add_argument(
        "--teacher-model",
        default="claude-sonnet-4-6",
        help="Teacher model name (default: claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--judge-model",
        default="@cf/google/gemma-4-26b-a4b-it",
        help="Judge model name (default: @cf/google/gemma-4-26b-a4b-it)",
    )
```

8. Update the final `run(...)` call in `main()`:
```python
    run(
        limit=args.limit,
        out_path=args.out,
        dry_run=args.dry_run,
        split=args.split,
        split_path=split_path,
        teacher_model=args.teacher_model,
        judge_model=args.judge_model,
    )
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest tests/test_run_eval_judge_family.py -xvs
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teaching_knowledge/run_eval.py apps/evals/tests/test_run_eval_judge_family.py && git commit -m "feat(evals): run_eval.py --teacher-model/--judge-model with family compat guard"
```

---

### Task T18: dual_judge.py agreement computation
**Group:** G (parallel, depends on T5)

**Behavior being verified:** `compute_agreement` computes per-dim Spearman correlation between two judges' score vectors and classifies each dim into high-trust / uncertain / low-trust buckets.

**Interface under test:** `teaching_knowledge.scripts.dual_judge.compute_agreement`, `DimensionAgreement`.

**Files:**
- Create: `apps/evals/teaching_knowledge/scripts/dual_judge.py`
- Test: `apps/evals/tests/test_dual_judge.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/tests/test_dual_judge.py
from __future__ import annotations

from teaching_knowledge.scripts.dual_judge import (
    DimensionAgreement,
    _spearman,
    compute_agreement,
)


def test_spearman_perfect_positive() -> None:
    assert abs(_spearman([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) - 1.0) < 1e-9


def test_spearman_perfect_negative() -> None:
    assert abs(_spearman([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) - (-1.0)) < 1e-9


def test_spearman_uncorrelated_roughly_zero() -> None:
    # No monotonic relationship
    rho = _spearman([1, 2, 3, 4, 5, 6], [3, 1, 4, 1, 5, 9])
    assert -0.5 < rho < 0.95  # loose bound; just confirms it's not ~1


def test_compute_agreement_groups_by_criterion() -> None:
    judge_a_rows = [
        {"recording_id": "r1", "judge_dimensions": [
            {"criterion": "Style", "score": 3},
            {"criterion": "Tone", "score": 2},
        ]},
        {"recording_id": "r2", "judge_dimensions": [
            {"criterion": "Style", "score": 2},
            {"criterion": "Tone", "score": 3},
        ]},
        {"recording_id": "r3", "judge_dimensions": [
            {"criterion": "Style", "score": 1},
            {"criterion": "Tone", "score": 1},
        ]},
        {"recording_id": "r4", "judge_dimensions": [
            {"criterion": "Style", "score": 3},
            {"criterion": "Tone", "score": 2},
        ]},
        {"recording_id": "r5", "judge_dimensions": [
            {"criterion": "Style", "score": 0},
            {"criterion": "Tone", "score": 3},
        ]},
    ]
    # Judge B agrees perfectly on Style, disagrees on Tone
    judge_b_rows = [
        {"recording_id": "r1", "judge_dimensions": [
            {"criterion": "Style", "score": 3},
            {"criterion": "Tone", "score": 3},
        ]},
        {"recording_id": "r2", "judge_dimensions": [
            {"criterion": "Style", "score": 2},
            {"criterion": "Tone", "score": 1},
        ]},
        {"recording_id": "r3", "judge_dimensions": [
            {"criterion": "Style", "score": 1},
            {"criterion": "Tone", "score": 2},
        ]},
        {"recording_id": "r4", "judge_dimensions": [
            {"criterion": "Style", "score": 3},
            {"criterion": "Tone", "score": 0},
        ]},
        {"recording_id": "r5", "judge_dimensions": [
            {"criterion": "Style", "score": 0},
            {"criterion": "Tone", "score": 3},
        ]},
    ]

    agreements = compute_agreement(judge_a_rows, judge_b_rows)
    by_name = {a.name: a for a in agreements}

    assert "Style" in by_name
    assert "Tone" in by_name

    style_ag = by_name["Style"]
    assert style_ag.spearman > 0.9
    assert style_ag.trust_level == "high"

    tone_ag = by_name["Tone"]
    assert tone_ag.trust_level in {"uncertain", "low"}


def test_trust_level_thresholds() -> None:
    from teaching_knowledge.scripts.dual_judge import _classify_trust

    assert _classify_trust(0.85) == "high"
    assert _classify_trust(0.7) == "high"
    assert _classify_trust(0.69) == "uncertain"
    assert _classify_trust(0.4) == "uncertain"
    assert _classify_trust(0.39) == "low"
    assert _classify_trust(-0.2) == "low"
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest tests/test_dual_judge.py -xvs
```
Expected: FAIL — `ModuleNotFoundError: No module named 'teaching_knowledge.scripts.dual_judge'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `apps/evals/teaching_knowledge/scripts/dual_judge.py`:
```python
"""Dual-judge calibration harness.

Runs two judges over the same synthesis outputs (or compares existing
judge JSONLs) and reports per-dim Spearman agreement plus a trust-level
classification: high (>0.7), uncertain (0.4-0.7), low (<0.4).

Usage (offline mode, against two existing judge JSONLs):
    cd apps/evals
    uv run python -m teaching_knowledge.scripts.dual_judge \\
        --judge-a results/judge_gemma.jsonl \\
        --judge-b results/judge_gpt.jsonl \\
        --out    results/dual_judge_calibration.md
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DimensionAgreement:
    name: str
    spearman: float
    trust_level: str
    n: int


@dataclass
class DualJudgeReport:
    dimensions: list[DimensionAgreement]
    n_compared: int


def _rank(values: list[float]) -> list[float]:
    """Dense rank with average-tie-breaking."""
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def _spearman(a: list[float], b: list[float]) -> float:
    """Pure-Python Spearman rank correlation."""
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    ra = _rank(a)
    rb = _rank(b)
    n = len(a)
    mean_a = sum(ra) / n
    mean_b = sum(rb) / n
    num = sum((ra[i] - mean_a) * (rb[i] - mean_b) for i in range(n))
    denom_a = sum((ra[i] - mean_a) ** 2 for i in range(n)) ** 0.5
    denom_b = sum((rb[i] - mean_b) ** 2 for i in range(n)) ** 0.5
    if denom_a == 0 or denom_b == 0:
        return 0.0
    return num / (denom_a * denom_b)


def _classify_trust(spearman: float) -> str:
    if spearman >= 0.7:
        return "high"
    if spearman >= 0.4:
        return "uncertain"
    return "low"


def _index_by_recording(rows: list[dict]) -> dict[str, dict[str, float]]:
    """rows -> {recording_id -> {criterion -> score}}."""
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        rid = row["recording_id"]
        crit_scores: dict[str, float] = {}
        for dim in row.get("judge_dimensions", []):
            score = dim.get("score")
            if score is not None:
                crit_scores[dim["criterion"]] = float(score)
        out[rid] = crit_scores
    return out


def compute_agreement(
    judge_a_rows: list[dict],
    judge_b_rows: list[dict],
) -> list[DimensionAgreement]:
    a_idx = _index_by_recording(judge_a_rows)
    b_idx = _index_by_recording(judge_b_rows)
    common_recs = sorted(set(a_idx) & set(b_idx))

    by_crit_a: dict[str, list[float]] = defaultdict(list)
    by_crit_b: dict[str, list[float]] = defaultdict(list)

    for rid in common_recs:
        for crit, a_score in a_idx[rid].items():
            if crit in b_idx[rid]:
                by_crit_a[crit].append(a_score)
                by_crit_b[crit].append(b_idx[rid][crit])

    agreements: list[DimensionAgreement] = []
    for crit in sorted(by_crit_a.keys()):
        a_vals = by_crit_a[crit]
        b_vals = by_crit_b[crit]
        rho = _spearman(a_vals, b_vals)
        agreements.append(
            DimensionAgreement(
                name=crit,
                spearman=rho,
                trust_level=_classify_trust(rho),
                n=len(a_vals),
            )
        )
    return agreements


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual-judge calibration")
    parser.add_argument("--judge-a", type=Path, required=True)
    parser.add_argument("--judge-b", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    agreements = compute_agreement(_load_jsonl(args.judge_a), _load_jsonl(args.judge_b))
    lines = ["# Dual-Judge Calibration", ""]
    lines.append(f"{'Dimension':<45} {'Spearman':>10} {'Trust':>10} {'N':>5}")
    lines.append("-" * 72)
    for ag in agreements:
        lines.append(f"{ag.name[:45]:<45} {ag.spearman:>10.3f} {ag.trust_level:>10} {ag.n:>5}")
    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest tests/test_dual_judge.py -xvs
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teaching_knowledge/scripts/dual_judge.py apps/evals/tests/test_dual_judge.py && git commit -m "feat(evals): add dual_judge.py for Spearman cross-family calibration"
```

---

### Task T19: eval_ab.py teacher-finetune A/B harness
**Group:** G (parallel, depends on T13)

**Behavior being verified:** Given fixture baseline and candidate JSONLs, `run_ab` returns an `ABReport` with a correct verdict (`CANDIDATE_WINS` | `CANDIDATE_LOSES` | `EQUIVALENT`) based on the regression report + efficiency deltas.

**Interface under test:** `teacher_model.eval_ab.run_ab`, `ABReport`.

**Files:**
- Create: `apps/evals/teacher_model/__init__.py` (if missing)
- Create: `apps/evals/teacher_model/eval_ab.py`
- Test: `apps/evals/tests/test_eval_ab.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/tests/test_eval_ab.py
from __future__ import annotations

import json
from pathlib import Path

from teacher_model.eval_ab import ABReport, run_ab


def _write_run_jsonl(path: Path, dim_scores: list[tuple[int, int]], run_id: str, synthesis_ms: int) -> None:
    rows = []
    for i, (p, o) in enumerate(dim_scores):
        rows.append({
            "recording_id": f"r{i}",
            "run_id": run_id,
            "git_sha": "abc1234",
            "piece_slug": "p",
            "title": "t",
            "composer": "Bach",
            "skill_bucket": 3,
            "synthesis_latency_ms": synthesis_ms,
            "judge_latency_ms": 50,
            "muq_means": {},
            "synthesis_text": "",
            "judge_dimensions": [
                {
                    "criterion": "Specific Positive Praise",
                    "process": p,
                    "outcome": o,
                    "score": min(p, o),
                    "evidence": "",
                    "reason": "",
                }
            ],
            "judge_model": "gemma",
            "error": "",
        })
    path.write_text("\n".join(json.dumps(r) for r in rows))


def _write_index(path: Path, n: int) -> None:
    rows = [
        json.dumps({
            "recording_id": f"r{i}",
            "composer_era": "Baroque",
            "skill_bucket": 3,
            "duration_bucket": "30-60s",
        })
        for i in range(n)
    ]
    path.write_text("\n".join(rows))


def test_candidate_wins_on_clear_improvement(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    index = tmp_path / "dataset_index.jsonl"

    # Baseline: all 1/1; Candidate: all 3/3 -- unambiguous win
    _write_run_jsonl(baseline, [(1, 1)] * 20, run_id="base", synthesis_ms=1000)
    _write_run_jsonl(candidate, [(3, 3)] * 20, run_id="cand", synthesis_ms=500)
    _write_index(index, 20)

    report = run_ab(baseline, candidate, index)
    assert isinstance(report, ABReport)
    assert report.verdict == "CANDIDATE_WINS"
    assert report.regression_report.composite_delta > 0
    assert report.efficiency_delta["synthesis_latency_ms"] == -500


def test_candidate_loses_on_clear_regression(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    index = tmp_path / "dataset_index.jsonl"

    _write_run_jsonl(baseline, [(3, 3)] * 20, run_id="base", synthesis_ms=500)
    _write_run_jsonl(candidate, [(1, 1)] * 20, run_id="cand", synthesis_ms=500)
    _write_index(index, 20)

    report = run_ab(baseline, candidate, index)
    assert report.verdict == "CANDIDATE_LOSES"
    assert report.regression_report.has_regression is True


def test_equivalent_when_no_significant_change(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    index = tmp_path / "dataset_index.jsonl"

    _write_run_jsonl(baseline, [(2, 2)] * 20, run_id="base", synthesis_ms=500)
    _write_run_jsonl(candidate, [(2, 2)] * 20, run_id="cand", synthesis_ms=500)
    _write_index(index, 20)

    report = run_ab(baseline, candidate, index)
    assert report.verdict == "EQUIVALENT"
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest tests/test_eval_ab.py -xvs
```
Expected: FAIL — `ModuleNotFoundError: No module named 'teacher_model.eval_ab'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `apps/evals/teacher_model/__init__.py` if missing:
```python
```

Create `apps/evals/teacher_model/eval_ab.py`:
```python
"""A/B harness for comparing a candidate teacher (e.g., finetuned Qwen)
against a baseline (e.g., Sonnet 4.6) over the same dataset.

Usage:
    cd apps/evals
    uv run python -m teacher_model.eval_ab \\
        results/baseline_sonnet.jsonl \\
        results/candidate_qwen.jsonl \\
        --dataset-index teaching_knowledge/data/dataset_index.jsonl
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from teaching_knowledge.scripts.aggregate import aggregate_run
from teaching_knowledge.scripts.regression_check import (
    RegressionReport,
    check_regression,
    format_report,
)


@dataclass
class ABReport:
    baseline_run_id: str
    candidate_run_id: str
    regression_report: RegressionReport
    efficiency_delta: dict[str, float]
    verdict: str  # "CANDIDATE_WINS" | "CANDIDATE_LOSES" | "EQUIVALENT"


def _efficiency_delta(baseline_path: Path, candidate_path: Path) -> dict[str, float]:
    def avg(path: Path, key: str) -> float:
        vals: list[float] = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            v = row.get(key)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "synthesis_latency_ms": avg(candidate_path, "synthesis_latency_ms")
        - avg(baseline_path, "synthesis_latency_ms"),
        "judge_latency_ms": avg(candidate_path, "judge_latency_ms")
        - avg(baseline_path, "judge_latency_ms"),
    }


def _verdict(regression: RegressionReport) -> str:
    if regression.has_regression:
        return "CANDIDATE_LOSES"
    if regression.composite_significant and regression.composite_delta > 0:
        return "CANDIDATE_WINS"
    return "EQUIVALENT"


def run_ab(
    baseline_jsonl: Path,
    candidate_jsonl: Path,
    dataset_index: Path,
) -> ABReport:
    base_agg = aggregate_run(baseline_jsonl, dataset_index)
    cand_agg = aggregate_run(candidate_jsonl, dataset_index)
    regression = check_regression(base_agg, cand_agg)
    efficiency = _efficiency_delta(baseline_jsonl, candidate_jsonl)
    verdict = _verdict(regression)
    return ABReport(
        baseline_run_id=base_agg.run_id,
        candidate_run_id=cand_agg.run_id,
        regression_report=regression,
        efficiency_delta=efficiency,
        verdict=verdict,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Teacher A/B eval")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--dataset-index", type=Path, required=True)
    args = parser.parse_args()

    report = run_ab(args.baseline, args.candidate, args.dataset_index)
    print(format_report(report.regression_report))
    print()
    print(f"baseline run_id:  {report.baseline_run_id}")
    print(f"candidate run_id: {report.candidate_run_id}")
    print()
    print("efficiency deltas (candidate - baseline):")
    for k, v in report.efficiency_delta.items():
        print(f"  {k}: {v:+.1f}")
    print()
    print(f"VERDICT: {report.verdict}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest tests/test_eval_ab.py -xvs
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/__init__.py apps/evals/teacher_model/eval_ab.py apps/evals/tests/test_eval_ab.py && git commit -m "feat(evals): add eval_ab.py teacher-finetune A/B harness"
```

---

## Final verification

After all 19 tasks pass, run the full Python test suite and the TS synthesis test to confirm no regressions:

```bash
cd apps/evals && uv run pytest tests/ -x
cd apps/api && bun test src/services/prompts.test.ts
```

Both commands must exit 0.

---

## Post-plan (manual, after Model v2 cache refresh)

Not part of this plan — tracked in `docs/plans/2026-04-14-eval-improvements.md`:

1. Rebuild inference cache after Model v2 training completes
2. Run `uv run python -m teaching_knowledge.scripts.tag_dataset`
3. Run `uv run python -m teaching_knowledge.scripts.split --seed 42`
4. Run locked baseline: `uv run python -m teaching_knowledge.run_eval --split train --teacher-model claude-sonnet-4-6 --judge-model "@cf/google/gemma-4-26b-a4b-it" --out results/baseline_sonnet46_judge-gemma4_YYYY-MM-DD.jsonl`
5. Aggregate: `uv run python -m teaching_knowledge.scripts.aggregate results/baseline_...jsonl`
6. Freeze the aggregate + JSONL as the "beat this" artifact
7. Begin hill-climbing the teacher prompt against the train split; use `regression_check.py` before unlocking holdout

---

## Challenge Review

Reviewed against the committed plan, the linked spec `docs/specs/2026-04-14-eval-baseline-readiness-design.md`, and every source file named in the File Structure table (`run_eval.py`, `shared/judge.py`, `synthesis_quality_judge_v2.txt`, `playbook.yaml`, `analyze_e2e.py`, `prompts.ts`, `teacher.ts`, `test_analyze_e2e.py`, plus grep sweeps for `bootstrap_ci` and `buildSynthesisFraming`).

### CEO Pass

**Premise.** Right problem, right time. Model v2 training is currently consuming the MuQ-critical path; landing every eval-harness improvement that does *not* depend on valid MuQ scores parallelizes the wait. Alternative framings (skip harness work, wait for scores, do it serially after retraining) were considered in the spec and rejected on ~2 weeks of wall time. [OBS]

**Scope.** 19 tasks, tightly bounded. Nothing in the plan runs an LLM call for real, and nothing depends on live inference — all code paths are verified via unit tests against fake payloads, fixture JSONL, or in-memory data structures. I tried to find something to cut:
- T19 (`eval_ab.py` scaffold) is the only task whose *consumer* is explicitly Phase 2. It sits behind a feature gate and does not execute at P0. Cuttable, but the marginal cost is low (one file, one test) and landing it now removes a future blocker. Keep. [OBS]
- T13 (`regression_check.py`) is pre-built tooling for the hill-climbing loop in P2 and has no pre-baseline consumer. Same reasoning: low marginal cost, shrinks a future plan. Keep. [OBS]

No scope drift vs spec goals. No unnecessary abstractions. [OBS]

**12-month alignment.**
```
CURRENT STATE                   THIS PLAN                       12-MONTH IDEAL
style-agnostic prompt    →   style guidance injected      →   per-era hill-climbed prompt
no provenance            →   run_id + git_sha stamped     →   full experiment ledger
single-sample scores     →   bootstrap CIs               →   stable dim-level reports
same-family judging      →   cross-family guard         →   dual-judge calibration
no holdout discipline    →   stratified 80/20 split     →   train/holdout + phantom audit
one judge + one schema   →   process/outcome split      →   calibrated 7-dim rubric
```
Every task in this plan moves toward the ideal, not away. No tech debt. [OBS]

**Alternatives.** The spec documents 8 design decisions with alternatives considered (Decisions #1–#8). That's above-bar for a plan at this stage. [OBS]

### Engineering Pass

**Architecture.** New modules are structured as deep units:
- `shared/stats.py` — 2 public functions (`bootstrap_ci`, `cohens_d`), hides numpy RNG seeding + ddof + quantile math
- `shared/style_rules.py` — 2 public functions (`composer_to_era`, `get_style_guidance`), hides JSON load, substring matching, XML formatting
- `shared/run_provenance.py` — 1 public factory (`make_run_provenance`), hides subprocess git capture + dirty-tree detection
- `shared/judge_compatibility.py` — 1 public function (`assert_judge_compatible`), hides a model-prefix → family lookup table

All four are DEEP by Ousterhout's definition. None are shallow. [OBS]

`run_eval.py` is extended in place rather than rewritten — the plan treats Group F (T14→T15→T16→T17) as four sequential edits to the same file in a strict order. I traced the data flow and all four edits touch non-overlapping regions (style injection is in `build_synthesis_user_msg`, split is in `run()` signature + load step, provenance is in `_build_row()` extraction, compat is in `run()` prologue). Serial ordering is correct; no merge-conflict landmines. [OBS]

**Module Depth Audit.**
| Module | Interface size | Impl | Verdict |
|---|---|---|---|
| `shared/stats.py` | 2 funcs | ~40 LOC numpy | DEEP |
| `shared/style_rules.py` | 2 funcs | ~60 LOC + JSON | DEEP |
| `shared/run_provenance.py` | 1 factory | ~30 LOC subprocess | DEEP |
| `shared/judge_compatibility.py` | 1 func | ~50 LOC + table | DEEP |
| `shared/llm_client_openrouter.py` | class extension in LLMClient | ~60 LOC | DEEP |
| `teaching_knowledge/scripts/tag_dataset.py` | 1 script entry | ~150 LOC | DEEP |
| `teaching_knowledge/scripts/split.py` | 2 funcs | ~80 LOC | DEEP |
| `teaching_knowledge/scripts/aggregate.py` | 1 CLI + 2 helpers | ~120 LOC | DEEP |
| `teaching_knowledge/scripts/dual_judge.py` | 1 script + Spearman | ~120 LOC | DEEP |
| `teaching_knowledge/scripts/regression_check.py` | 1 CLI | ~60 LOC | DEEP |
| `teaching_knowledge/scripts/eval_ab.py` | 1 CLI | ~80 LOC | DEEP |

No shallow modules. [OBS]

**Code Quality.**

`[OBS]` — **T6 plan prose contradicts its own code.** The Behavior line at plan:875 says "derive the composite `score` as their mean (floor)" but the implementation at plan:1039 returns `min(process, outcome)` and the test assertion at plan:940 (`process=3, outcome=1 → score==1`) confirms `min()` is the intended semantic. The code is correct — `min()` is a more conservative composite than mean for a process+outcome gate, and the test pins it. But the description should be updated to say "composite = `min(process, outcome)` (the conservative signal)". Cosmetic plan-doc bug, not a functional defect.

`[OBS]` — **T6 `parse_failure` semantics are inconsistent.** When the parser fails, it returns `DimensionScore(score=0, process=None, outcome=None)`. This breaks the otherwise-clean invariant that legacy rows have `process == outcome == score`. A parse failure now contributes `score=0` to dim means but `None` to process/outcome means, creating a discrepancy that downstream `aggregate.py` must handle. Two fixes possible: (a) set `score=None` for parse failures and teach aggregate to skip None-score rows, or (b) set `process = outcome = 0` and accept the score floor. Flag for T12 execution — the aggregate test does not currently exercise a parse_failure input row. Verify this is actually an issue when T12 is built.

`[RISK] (confidence: 7/10)` — **T17 provider autodetect silently misroutes native Anthropic judge names.** At plan:3299:
```python
judge_provider = "openrouter" if "/" in judge_model and not judge_model.startswith("@cf/") else "workers-ai"
```
Truth table:
| judge_model | routed to | correct? |
|---|---|---|
| `@cf/google/gemma-4-26b-a4b-it` | workers-ai | ✓ |
| `openai/gpt-5.4-mini` | openrouter | ✓ |
| `anthropic/claude-sonnet-4-6` | openrouter | ✓ |
| `claude-sonnet-4-6` (native) | **workers-ai** | ✗ silent misroute |
| `claude-haiku-4-5-20251001` | **workers-ai** | ✗ silent misroute |

The T17 test file exercises rows 1-3 but not rows 4-5. Failure mode: operator passes `--judge-model claude-sonnet-4-6` thinking it'll hit Anthropic (the native SDK name), the autodetect routes it to Workers AI which doesn't serve `claude-*` models, and the LLMClient fails with a confusing downstream error from the Workers AI endpoint. This is explicitly Phase 2 territory (Phase 1 uses Gemma-4 + GPT-5.4-mini, both of which route correctly), so it's not a P0 blocker — but T19's `eval_ab.py` scaffold is what first introduces a risk of passing `claude-*` as judge. **Mitigation options:**
  1. Document the convention: "Anthropic judges must use the `anthropic/` OpenRouter slug; bare `claude-*` names are rejected".
  2. Defensively raise `ValueError` in `_assert_models_compatible` when `judge_model` starts with `claude-` without an `anthropic/` prefix.
  3. Make `assert_judge_compatible` return the intended provider alongside the family so the autodetect is derived from the table rather than from slash-parsing.
Option (3) is cleanest and couples the two pieces of knowledge that must stay in sync. Recommend raising this at T17 execution time as a defensive hardening — not a blocker, since Phase 1 doesn't hit the failure case.

`[OBS]` — **T17 hardcodes teacher provider to `"anthropic"` at plan:3283.** `synthesis_client = LLMClient(provider="anthropic", model=teacher_model)`. A `--teacher-model @cf/qwen/...` invocation (Phase 2) would still route through the Anthropic SDK and fail. Phase 2 is explicitly out of this plan's scope (spec confirms) and `eval_ab.py` T19 is the explicit Phase 2 entry point, so this is intentional. No action needed, but flag in the T17 code comment so future readers don't miss the invariant.

`[OBS]` — **T9's drift guard is pytest-only, not a pre-commit hook.** The Python/TS style-rules mirror is enforced by `test_style_rules_mirror.py` — which runs only when someone runs the eval test suite. A PR that edits `apps/api/src/lib/style-rules.json` in isolation (a frontend-only PR) won't trip the guard on CI unless the eval tests are part of the TypeScript CI pipeline. Consider: (a) mark this test as part of `apps/api`'s CI lane, or (b) add a bun-side integrity check that imports the JSON and asserts a known SHA. Not a blocker — drift is recoverable by regenerating from the Python source, and the error message tells you exactly what to run. But enforcement is fragile.

`[OBS]` — **T1 spec decision #3 is factually wrong.** The spec at Decision #3 claims "A working `bootstrap_ci` already exists at `pipeline/practice_eval/analyze_e2e.py`". It does not — grep across the file returns zero matches. What *does* exist is `test_analyze_e2e.py` (verified at :26, :35) importing `bootstrap_ci` from that exact path, so the test file is currently broken against head. T1's implementation step (write `bootstrap_ci` fresh in `shared/stats.py` + re-export through `analyze_e2e.py`) fortuitously repairs the pre-existing broken import — T1 Step 4 runs both `test_stats.py` AND `test_analyze_e2e.py`, so the repair is verified. Net effect: plan is correct, spec decision is misdescribed. Update the spec after /challenge or carry as a known minor inaccuracy.

**Test Philosophy Audit.** I walked every task's test code:
- No test mocks an internal collaborator of the module under test.
- No test calls a private method directly (underscored functions like `_assert_models_compatible`, `_build_row`, `_parse_v2_response`, `_composite` are called — but each is an intentionally module-level pure helper promoted for testability, not a private method of an object being probed through a side channel).
- No test asserts on internal state without going through a public surface.
- Tests exercise observable behavior: dataclass fields, function return values, file contents, JSONL row structure.

One borderline case worth flagging: **T11 tests `prompts.ts` via `bun test` but `teacher.ts` gets no dedicated behavior test** — it relies on the TypeScript compiler to enforce the new required `composer: string` parameter in `buildSynthesisFraming`. Acceptable for a strongly-typed language (compile errors *are* test failures in TS), but a bun-side smoke test that the synthesis prompt changes shape when a known composer is passed would tighten coverage. Not a blocker. [OBS]

**Vertical Slice Audit.**
19 tasks × 5 steps each. I spot-checked Groups A, B, C, D, F (the groups with the most interdependent tasks). Each task is one test → one implementation → one commit. No task bundles multiple tests before any implementation. No task writes scaffolding tests for later tasks. No task defers implementation. Clean. [OBS]

**Test Coverage Gaps.**

```
[+] shared/stats.py
    ├── bootstrap_ci() — happy path [★★], determinism [★★], small sample [★★★]
    └── cohens_d()    — zero case, positive case, degenerate [★★★]

[+] shared/style_rules.py
    ├── composer_to_era()   — 4 eras + unknown [★★★]
    └── get_style_guidance()— 2 eras + unknown + XML shape [★★★]

[+] shared/run_provenance.py
    └── make_run_provenance() — returns dataclass with git_sha [★★]
        ├── [GAP] what happens when CWD is not a git repo?
        ├── [GAP] what happens when git is not on PATH?

[+] shared/judge_compatibility.py
    └── assert_judge_compatible() — same-family raises, cross-family passes [★★★]

[+] LLMClient.complete (OpenRouter provider)
    └── _build_openrouter_payload() — pure helper test only [★★]
        ├── [GAP] no test for HTTP error path (500, 429, timeout)
        ├── [GAP] no test for missing OPENROUTER_API_KEY env var

[+] shared/judge._parse_v2_response
    ├── legacy single-score row [★★★]
    ├── new process/outcome row [★★★]
    ├── N/A process + N/A outcome [★★★]
    ├── N/A process + numeric outcome [★★★]
    └── parse failure [★★]
        └── [OBS] parse_failure returns score=0 with process/outcome=None — verify at T12

[+] teaching_knowledge.run_eval._assert_models_compatible
    ├── claude vs claude rejected [★★★]
    ├── claude vs anthropic/claude-slug rejected [★★★]
    ├── claude vs @cf/google/gemma accepted [★★★]
    ├── claude vs openai/gpt accepted [★★★]
    └── [GAP] claude-sonnet-4-6 (bare) vs bare claude — see RISK above
```

`[RISK] (confidence: 6/10)` — **`make_run_provenance()` has no defensive test for non-git or missing-git environments.** If someone runs the eval runner from a tarball extraction or in a CI job without `.git`, the subprocess call will fail. T3's implementation needs to decide: fail loudly, or emit `git_sha="unknown"` with a warning? Either is defensible; both need a test. Verify this is actually an issue — the plan may already handle it and I missed it during my T3 scan.

`[OBS]` — **OpenRouter HTTP failure paths are untested.** T5's payload-builder test is pure but there's no test for what happens on a 500/429/connection-error response. `LLMClient._workers_ai_complete` at the existing code has a shape check (`response.status_code != 200`); T5 should inherit the same pattern — if it does, coverage is fine. Check during T5 execution.

**Failure Modes.**

For each task, I asked "what happens if this fails mid-execution?":
- T1 fails mid-way: `shared/stats.py` exists but `analyze_e2e.py` re-export is partial → `test_analyze_e2e.py` is broken in a new way. Recovery: finish the edit. Not a corrupt state.
- T8 (`tag_dataset.py`) fails mid-way: partial JSONL on disk. Script is idempotent per video_id, re-run covers the gap. OK.
- T10 (split.py) fails mid-way: no split file written, or partial JSON written. `json.dump` is atomic-enough for small files; re-run resolves. OK.
- T11 (TS edit) fails mid-way: `prompts.ts` edited but `teacher.ts` still passes 6 args → TS compile error blocks deploy. Loud failure, no silent state. OK.
- T17 misroute: **silent downstream error** — see RISK above.

No silent failures found anywhere else. [OBS]

### Presumption Inventory

| Assumption | Verdict | Reason |
|---|---|---|
| `bootstrap_ci` exists at `pipeline/practice_eval/analyze_e2e.py` (spec #3) | **FALSE → repaired** | Does not exist today; T1 writes it and fortuitously fixes a broken pre-existing import in `test_analyze_e2e.py` |
| `shared/` is importable as a top-level Python package | SAFE | T2 creates `shared/__init__.py` if missing; existing `shared/judge.py` and `shared/llm_client_gpt.py` already use this layout |
| `numpy` is available in the `apps/evals` uv env | SAFE | `pipeline/practice_eval/analyze_e2e.py` already imports numpy at :23 |
| `cd apps/evals && uv run pytest tests/...` discovers tests correctly | SAFE | Existing `test_analyze_e2e.py` and `test_judge.py` already live under this layout |
| `pieceMetadata` in `buildSynthesisFraming` caller has a `.composer` field | VALIDATE | Typed `unknown` in `teacher.ts`; T11 does a defensive `as { composer?: string }` cast with `?? ""` fallback — safe, but the caller at `teacher.ts:546` is the only callsite and the metadata shape is set upstream in `synthesize()` — trace end-to-end during T11 |
| OpenRouter serves `openai/gpt-5.4-mini` and `anthropic/claude-sonnet-4-6` slugs | VALIDATE | Standard catalog slugs, but nothing in the plan verifies they're current at execution time. Smoke-test with a 1-token call before landing T5 |
| `git rev-parse HEAD` works in the execution environment for T3 | VALIDATE | Local dev + CI are fine; tarball extractions or sandboxed containers may fail. See GAP above |
| `test_style_rules_mirror.py` runs on every change that could drift it | RISKY | T9 only enforces via pytest; a TS-only PR editing the mirror file won't trip it unless eval tests are in the TS CI lane |
| Workers AI at `@cf/google/gemma-4-26b-a4b-it` remains stable for Phase 1 | SAFE | Prod is already on this model (see MEMORY notes: "disabling reasoning on gemma-4") |
| Anthropic `claude-sonnet-4-6` is the current prod teacher name | SAFE | Confirmed in `teacher.ts:565` (`callAnthropic(..., { model: "claude-sonnet-4-20250514", ... })`) — wait. |

**One more issue surfaces from the presumption check:** `teacher.ts:565` calls `callAnthropic` with model `"claude-sonnet-4-20250514"` but the plan and memory both speak of `"claude-sonnet-4-6"` as the prod teacher name. These are not obviously the same model. Either (a) `claude-sonnet-4-6` is the eval alias and `claude-sonnet-4-20250514` is the API model ID, or (b) there's genuine drift between prod and eval.

`[QUESTION]` — **What is the canonical model string for the prod Sonnet teacher?** `teacher.ts:565` says `claude-sonnet-4-20250514`, the plan's `--teacher-model` default is `claude-sonnet-4-6`, and memory says "Claude Sonnet 4.6". If these map to the same deployed model at Anthropic, document the mapping in a comment at the T17 `--teacher-model` default. If they're different models, the eval baseline isn't actually testing prod and P0 needs a fix.

### Summary

- `[BLOCKER]` count: **0**
- `[RISK]` count: **2** (T17 provider autodetect silent misroute; make_run_provenance non-git environment handling)
- `[QUESTION]` count: **2** (teacher model string canonical form; spec decision #3 correction policy)
- `[OBS]` count: **11**

---

**VERDICT: PROCEED_WITH_CAUTION** — risks to monitor during execution:
1. At T17, harden the judge-provider autodetect so bare `claude-*` names either route correctly or raise a loud ValueError instead of silently misrouting to Workers AI. Recommended: have `assert_judge_compatible` return `(family, provider)` so T17's autodetect comes from the compatibility table, not slash-parsing.
2. At T3, add a test for `make_run_provenance` running in a non-git CWD — decide whether to fail loudly or stamp `git_sha="unknown"`, and pin the chosen behavior.
3. Before executing T11, verify the canonical prod teacher model string (`claude-sonnet-4-6` vs `claude-sonnet-4-20250514`) and pin the default in `run_eval.py` to the same string Anthropic actually accepts.
4. After landing T9, decide whether to include `test_style_rules_mirror.py` in the `apps/api` CI lane or accept the pytest-only drift guard.
5. At T6, verify `parse_failure` row semantics interact correctly with T12's aggregate null-handling before freezing the baseline.

None of these rise to BLOCKER. Every issue has a defined mitigation that fits in the existing task shape. Plan is ready to execute once these are acknowledged.
