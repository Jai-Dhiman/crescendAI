# Tasks 06-10 (Group B — Python `teacher_style.py` track, sequential)

All tasks in this file modify `apps/evals/shared/teacher_style.py` and `apps/evals/shared/test_teacher_style.py`. Run sequentially; cannot be parallel.

---

## Task 6: Python DSL evaluator — arithmetic + signal lookup
**Group:** B (depends on Task 1)

**Behavior being verified:** `evaluate(formula, signals)` evaluates `"1.5 * max_neg_dev"` against a signals dict and returns the expected float.

**Interface under test:** `apps/evals/shared/teacher_style.py::evaluate`.

**Files:**
- Create: `apps/evals/shared/teacher_style.py`
- Create: `apps/evals/shared/test_teacher_style.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/shared/test_teacher_style.py
import pytest
from shared.teacher_style import evaluate

SIGNALS = {
    "max_neg_dev": 0.2, "max_pos_dev": 0.0, "n_significant": 2,
    "drilling_present": False, "drilling_improved": False,
    "duration_min": 15.0, "mode_count": 1, "has_piece": True,
}


def test_evaluate_arithmetic():
    assert evaluate("1.5 * max_neg_dev + 0.3 * n_significant", SIGNALS) == pytest.approx(0.9)


def test_evaluate_signal_lookup():
    assert evaluate("max_neg_dev", SIGNALS) == pytest.approx(0.2)


def test_evaluate_unknown_signal_raises():
    with pytest.raises(ValueError, match="unknown signal"):
        evaluate("max_neg_dev + bogus", SIGNALS)
```

- [ ] **Step 2: Run test — verify it FAILS**

```
cd apps/evals && uv run pytest shared/test_teacher_style.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'shared.teacher_style'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# apps/evals/shared/teacher_style.py
"""Cluster selection + teacher-voice prompt formatting.

Mirrors apps/api/src/services/teacher_style.ts. Both implementations evaluate
the same DSL formulas against the same signals.
"""
from __future__ import annotations
import re
from typing import Any

ALLOWED_SIGNALS = {
    "max_neg_dev", "max_pos_dev", "n_significant",
    "drilling_present", "drilling_improved",
    "duration_min", "mode_count", "has_piece",
}


class _Tokenizer:
    _TOKEN_RE = re.compile(
        r"\s*(?:"
        r"(?P<number>\d+(?:\.\d+)?)"
        r"|(?P<ident>[A-Za-z_][A-Za-z0-9_]*)"
        r"|(?P<op><=|>=|==|!=|[+\-*/<>()])"
        r")"
    )

    def __init__(self, text: str) -> None:
        self.tokens: list[tuple[str, str]] = []
        pos = 0
        while pos < len(text):
            m = self._TOKEN_RE.match(text, pos)
            if not m or m.end() == pos:
                if text[pos].isspace():
                    pos += 1
                    continue
                raise ValueError(f"unexpected character at {pos}: {text[pos]!r}")
            kind = m.lastgroup
            self.tokens.append((kind, m.group(kind)))
            pos = m.end()
        self.i = 0

    def peek(self):
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def take(self):
        tok = self.tokens[self.i]
        self.i += 1
        return tok


def evaluate(formula: str, signals: dict[str, Any]) -> float:
    tok = _Tokenizer(formula)
    value = _expr(tok, signals)
    if tok.peek() is not None:
        raise ValueError(f"unexpected trailing tokens in: {formula!r}")
    return float(value)


def _expr(tok, sig):
    return _arith(tok, sig)


def _arith(tok, sig):
    left = _term(tok, sig)
    while tok.peek() and tok.peek()[1] in ("+", "-"):
        op = tok.take()[1]
        right = _term(tok, sig)
        left = left + right if op == "+" else left - right
    return left


def _term(tok, sig):
    left = _factor(tok, sig)
    while tok.peek() and tok.peek()[1] in ("*", "/"):
        op = tok.take()[1]
        right = _factor(tok, sig)
        left = left * right if op == "*" else left / right
    return left


def _factor(tok, sig):
    nxt = tok.peek()
    if nxt is None:
        raise ValueError("unexpected end of formula")
    kind, text = nxt
    if kind == "number":
        tok.take()
        return float(text)
    if kind == "ident":
        if text not in ALLOWED_SIGNALS:
            raise ValueError(f"unknown signal: {text}")
        tok.take()
        v = sig[text]
        return float(v) if not isinstance(v, bool) else (1.0 if v else 0.0)
    if kind == "op" and text == "(":
        tok.take()
        v = _expr(tok, sig)
        if not tok.peek() or tok.peek()[1] != ")":
            raise ValueError("missing close paren")
        tok.take()
        return v
    raise ValueError(f"unexpected token: {nxt}")
```

- [ ] **Step 4: Run test — verify it PASSES**

```
cd apps/evals && uv run pytest shared/test_teacher_style.py -v
```
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```
git add apps/evals/shared/teacher_style.py apps/evals/shared/test_teacher_style.py
git commit -m "feat(teacher-style): python DSL evaluator (arithmetic + signals)"
```

---

## Task 7: Python DSL — boolean conditionals
**Group:** B (depends on Task 6)

**Behavior:** `evaluate("1 if drilling_improved else 0", signals)` returns the conditional branch correctly.

**Files:** modify `teacher_style.py` and `test_teacher_style.py`.

- [ ] **Step 1: Write the failing test** — append:

```python
def test_evaluate_conditional_true_branch():
    sig = {**SIGNALS, "drilling_improved": True}
    assert evaluate("1.5 if drilling_improved else 0.0", sig) == pytest.approx(1.5)


def test_evaluate_conditional_false_branch():
    sig = {**SIGNALS, "drilling_improved": False}
    assert evaluate("1.5 if drilling_improved else 0.5", sig) == pytest.approx(0.5)


def test_evaluate_compound_with_and():
    sig = {**SIGNALS, "max_neg_dev": 0.05, "max_pos_dev": 0.05}
    assert evaluate("1 if max_neg_dev < 0.1 and max_pos_dev < 0.1 else 0", sig) == pytest.approx(1.0)


def test_evaluate_real_formula_technical_corrective():
    sig = {**SIGNALS, "max_neg_dev": 0.2, "n_significant": 2, "drilling_improved": False}
    formula = "1.5 * max_neg_dev + 0.3 * n_significant - 0.5 * (1 if drilling_improved else 0)"
    assert evaluate(formula, sig) == pytest.approx(0.9)
```

- [ ] **Step 2: Run test — verify it FAILS**

```
cd apps/evals && uv run pytest shared/test_teacher_style.py -v
```
Expected: FAIL — first conditional test raises `ValueError` from `_factor` because `if` is not a known signal.

- [ ] **Step 3: Implement the minimum to make the test pass** — replace `_expr` and add boolean helpers in `teacher_style.py`:

```python
def _expr(tok, sig):
    value = _arith(tok, sig)
    nxt = tok.peek()
    if nxt and nxt[0] == "ident" and nxt[1] == "if":
        tok.take()
        cond = _bool_expr(tok, sig)
        kw = tok.peek()
        if not kw or kw[1] != "else":
            raise ValueError("expected 'else' in conditional")
        tok.take()
        else_value = _arith(tok, sig)
        return value if cond else else_value
    return value


def _bool_expr(tok, sig):
    left = _bool_and(tok, sig)
    while tok.peek() and tok.peek()[0] == "ident" and tok.peek()[1] == "or":
        tok.take()
        right = _bool_and(tok, sig)
        left = left or right
    return left


def _bool_and(tok, sig):
    left = _bool_not(tok, sig)
    while tok.peek() and tok.peek()[0] == "ident" and tok.peek()[1] == "and":
        tok.take()
        right = _bool_not(tok, sig)
        left = left and right
    return left


def _bool_not(tok, sig):
    nxt = tok.peek()
    if nxt and nxt[0] == "ident" and nxt[1] == "not":
        tok.take()
        return not _cmp(tok, sig)
    return _cmp(tok, sig)


def _cmp(tok, sig):
    left = _arith(tok, sig)
    nxt = tok.peek()
    if nxt and nxt[1] in ("<", "<=", ">", ">=", "==", "!="):
        op = tok.take()[1]
        right = _arith(tok, sig)
        return {
            "<": left < right, "<=": left <= right,
            ">": left > right, ">=": left >= right,
            "==": left == right, "!=": left != right,
        }[op]
    return bool(left)
```

- [ ] **Step 4: Run test — verify it PASSES**

```
cd apps/evals && uv run pytest shared/test_teacher_style.py -v
```
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```
git add apps/evals/shared/teacher_style.py apps/evals/shared/test_teacher_style.py
git commit -m "feat(teacher-style): python DSL conditionals + boolean ops"
```

---

## Task 8: Python `select_clusters` — top-2 selection from playbook
**Group:** B (depends on Task 7)

**Behavior:** `select_clusters(signals)` reads the playbook YAML, evaluates each cluster's `triggers.score`, returns top-2 named clusters with priority tie-break and confidence-floor fallback.

**Files:** modify `teacher_style.py` and `test_teacher_style.py`.

- [ ] **Step 1: Write the failing test** — append:

```python
from shared.teacher_style import select_clusters


def test_select_clusters_negative_dev_picks_technical():
    sig = {"max_neg_dev": 0.25, "max_pos_dev": 0.0, "n_significant": 3,
           "drilling_present": False, "drilling_improved": False,
           "duration_min": 15.0, "mode_count": 1, "has_piece": True}
    sel = select_clusters(sig)
    assert "Technical" in sel.primary.name


def test_select_clusters_positive_dev_picks_praise():
    sig = {"max_neg_dev": 0.0, "max_pos_dev": 0.3, "n_significant": 1,
           "drilling_present": True, "drilling_improved": True,
           "duration_min": 25.0, "mode_count": 2, "has_piece": True}
    sel = select_clusters(sig)
    assert "Positive" in sel.primary.name or "praise" in sel.primary.name.lower()


def test_select_clusters_returns_two_distinct():
    sig = {"max_neg_dev": 0.25, "max_pos_dev": 0.0, "n_significant": 3,
           "drilling_present": False, "drilling_improved": False,
           "duration_min": 15.0, "mode_count": 1, "has_piece": True}
    sel = select_clusters(sig)
    assert sel.primary.name != sel.secondary.name


def test_select_clusters_fallback_when_all_low():
    sig = {"max_neg_dev": 0.0, "max_pos_dev": 0.0, "n_significant": 0,
           "drilling_present": False, "drilling_improved": False,
           "duration_min": 15.0, "mode_count": 1, "has_piece": False}
    sel = select_clusters(sig)
    assert "Technical" in sel.primary.name
    assert "Positive" in sel.secondary.name or "praise" in sel.secondary.name.lower()
```

- [ ] **Step 2: Run test — verify it FAILS**

```
cd apps/evals && uv run pytest shared/test_teacher_style.py -v
```
Expected: FAIL — `ImportError: cannot import name 'select_clusters'`.

- [ ] **Step 3: Implement the minimum to make the test pass** — append to `teacher_style.py`:

```python
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import yaml

PLAYBOOK_PATH = Path(__file__).resolve().parents[2] / "shared" / "teacher-style" / "playbook.yaml"

PRIORITY_ORDER = [
    "Technical-corrective", "Positive-encouragement", "Artifact-based",
    "Guided-discovery", "Motivational",
]
FALLBACK_PRIMARY_KEY = "Technical-corrective"
FALLBACK_SECONDARY_KEY = "Positive-encouragement"
CONFIDENCE_FLOOR = 0.3


@dataclass(frozen=True)
class ClusterRef:
    name: str
    score: float
    raw: dict


@dataclass(frozen=True)
class ClusterSelection:
    primary: ClusterRef
    secondary: ClusterRef


@lru_cache(maxsize=1)
def _load_playbook():
    return yaml.safe_load(PLAYBOOK_PATH.read_text())["teaching_playbook"]["clusters"]


def _priority_index(name: str) -> int:
    for i, key in enumerate(PRIORITY_ORDER):
        if key.lower() in name.lower():
            return i
    return len(PRIORITY_ORDER)


def _find_cluster(substring: str) -> dict:
    for cluster in _load_playbook():
        if substring.lower() in cluster["name"].lower():
            return cluster
    raise ValueError(f"no cluster matching {substring!r}")


def select_clusters(signals: dict[str, Any]) -> ClusterSelection:
    scored = [
        ClusterRef(name=c["name"], score=evaluate(c["triggers"]["score"], signals), raw=c)
        for c in _load_playbook()
    ]
    scored.sort(key=lambda c: (-c.score, _priority_index(c.name)))
    primary, secondary = scored[0], scored[1]
    if primary.score < CONFIDENCE_FLOOR and secondary.score < CONFIDENCE_FLOOR:
        fp = _find_cluster(FALLBACK_PRIMARY_KEY)
        fs = _find_cluster(FALLBACK_SECONDARY_KEY)
        primary = ClusterRef(name=fp["name"], score=0.0, raw=fp)
        secondary = ClusterRef(name=fs["name"], score=0.0, raw=fs)
    return ClusterSelection(primary=primary, secondary=secondary)
```

- [ ] **Step 4: Run test — verify it PASSES**

```
cd apps/evals && uv run pytest shared/test_teacher_style.py -v
```
Expected: PASS (11 tests).

- [ ] **Step 5: Commit**

```
git add apps/evals/shared/teacher_style.py apps/evals/shared/test_teacher_style.py
git commit -m "feat(teacher-style): python select_clusters with priority + fallback"
```

---

## Task 9: Python `format_teacher_voice_blocks`
**Group:** B (depends on Task 8)

**Behavior:** Given a `ClusterSelection`, return a string containing both `<teacher_voice>` and `<also_consider>` blocks with Register, Tone, and Exemplar.

**Files:** modify `teacher_style.py` and `test_teacher_style.py`.

- [ ] **Step 1: Write the failing test** — append:

```python
from shared.teacher_style import format_teacher_voice_blocks


def test_format_emits_both_blocks():
    sig = {"max_neg_dev": 0.25, "max_pos_dev": 0.0, "n_significant": 3,
           "drilling_present": False, "drilling_improved": False,
           "duration_min": 15.0, "mode_count": 1, "has_piece": True}
    out = format_teacher_voice_blocks(select_clusters(sig))
    assert "<teacher_voice" in out
    assert "<also_consider" in out
    assert "Register:" in out
    assert "Tone:" in out
    assert "Exemplar:" in out


def test_format_includes_cluster_attribute():
    sig = {"max_neg_dev": 0.25, "max_pos_dev": 0.0, "n_significant": 3,
           "drilling_present": False, "drilling_improved": False,
           "duration_min": 15.0, "mode_count": 1, "has_piece": True}
    out = format_teacher_voice_blocks(select_clusters(sig))
    assert 'cluster="' in out
```

- [ ] **Step 2: Run test — verify it FAILS**

```
cd apps/evals && uv run pytest shared/test_teacher_style.py -v
```
Expected: FAIL — `ImportError: cannot import name 'format_teacher_voice_blocks'`.

- [ ] **Step 3: Implement the minimum to make the test pass** — append:

```python
def _normalize_cluster_id(name: str) -> str:
    return (name.lower()
            .replace("‑", "-").replace("–", "-").replace("—", "-")
            .replace("“", "").replace("”", "")
            .replace(" / ", "-").replace(" ", "-").strip())


def _first_exemplar(cluster: dict) -> str:
    for ex in cluster.get("good_examples") or []:
        text = ex.get("text") if isinstance(ex, dict) else None
        if text:
            return str(text)
    return ""


def format_teacher_voice_blocks(selection: ClusterSelection) -> str:
    lines: list[str] = []
    p, s = selection.primary, selection.secondary

    p_id = _normalize_cluster_id(p.name)
    p_register = p.raw.get("language_patterns", {}).get("register", "")
    p_tone = p.raw.get("language_patterns", {}).get("tone", "")
    p_ex = _first_exemplar(p.raw)
    lines.append(f'<teacher_voice cluster="{p_id}">')
    lines.append(f"Register: {p_register}")
    lines.append(f"Tone: {p_tone}")
    if p_ex:
        lines.append(f"Exemplar: {p_ex}")
    lines.append("</teacher_voice>")

    s_id = _normalize_cluster_id(s.name)
    s_when = "; ".join(s.raw.get("when_to_use") or [])
    s_ex = _first_exemplar(s.raw)
    lines.append("")
    lines.append(f'<also_consider cluster="{s_id}">')
    if s_when:
        lines.append(f"Apply when: {s_when}")
    if s_ex:
        lines.append(f"Exemplar: {s_ex}")
    lines.append("</also_consider>")
    return "\n".join(lines)
```

- [ ] **Step 4: Run test — verify it PASSES**

```
cd apps/evals && uv run pytest shared/test_teacher_style.py -v
```
Expected: PASS (13 tests).

- [ ] **Step 5: Commit**

```
git add apps/evals/shared/teacher_style.py apps/evals/shared/test_teacher_style.py
git commit -m "feat(teacher-style): python format_teacher_voice_blocks"
```

---

## Task 10: Canonical parity fixtures
**Group:** B (depends on Task 9)

**Behavior:** A canonical fixtures JSON in `shared/teacher-style/test_fixtures.json` is consumed by Python tests and produces expected primary/secondary names. The TS track will read the same file.

**Files:**
- Create: `shared/teacher-style/test_fixtures.json`
- Modify: `apps/evals/shared/test_teacher_style.py`

- [ ] **Step 1: Write the failing test** — append:

```python
import json as _json
from pathlib import Path as _Path

_FIXTURES = _Path(__file__).resolve().parents[2] / "shared" / "teacher-style" / "test_fixtures.json"


def test_parity_fixtures_match_expected_primary():
    fixtures = _json.loads(_FIXTURES.read_text())
    assert len(fixtures) >= 4
    for f in fixtures:
        sel = select_clusters(f["signals"])
        assert f["expected_primary_substring"].lower() in sel.primary.name.lower(), (
            f"fixture {f['name']}: expected primary contains "
            f"{f['expected_primary_substring']!r}, got {sel.primary.name!r}"
        )
        assert f["expected_secondary_substring"].lower() in sel.secondary.name.lower(), (
            f"fixture {f['name']}: expected secondary contains "
            f"{f['expected_secondary_substring']!r}, got {sel.secondary.name!r}"
        )
```

- [ ] **Step 2: Run test — verify it FAILS**

```
cd apps/evals && uv run pytest shared/test_teacher_style.py -v
```
Expected: FAIL — `FileNotFoundError: shared/teacher-style/test_fixtures.json`.

- [ ] **Step 3: Implement the minimum to make the test pass** — create `shared/teacher-style/test_fixtures.json`:

```json
[
  {
    "name": "strong_negative_deviation",
    "signals": {
      "max_neg_dev": 0.25, "max_pos_dev": 0.0, "n_significant": 3,
      "drilling_present": false, "drilling_improved": false,
      "duration_min": 15.0, "mode_count": 1, "has_piece": true
    },
    "expected_primary_substring": "Technical",
    "expected_secondary_substring": "Artifact"
  },
  {
    "name": "drilling_improvement",
    "signals": {
      "max_neg_dev": 0.05, "max_pos_dev": 0.2, "n_significant": 1,
      "drilling_present": true, "drilling_improved": true,
      "duration_min": 25.0, "mode_count": 2, "has_piece": true
    },
    "expected_primary_substring": "Positive",
    "expected_secondary_substring": "Technical"
  },
  {
    "name": "long_multi_mode_session",
    "signals": {
      "max_neg_dev": 0.1, "max_pos_dev": 0.1, "n_significant": 2,
      "drilling_present": true, "drilling_improved": false,
      "duration_min": 45.0, "mode_count": 3, "has_piece": true
    },
    "expected_primary_substring": "Guided",
    "expected_secondary_substring": "Technical"
  },
  {
    "name": "all_signals_low_fallback",
    "signals": {
      "max_neg_dev": 0.0, "max_pos_dev": 0.0, "n_significant": 0,
      "drilling_present": false, "drilling_improved": false,
      "duration_min": 15.0, "mode_count": 1, "has_piece": false
    },
    "expected_primary_substring": "Technical",
    "expected_secondary_substring": "Positive"
  }
]
```

- [ ] **Step 4: Run test — verify it PASSES**

```
cd apps/evals && uv run pytest shared/test_teacher_style.py -v
```
Expected: PASS. If a fixture's expected substring doesn't match the actual selection, the formula in the spec is the source of truth — adjust the fixture's `expected_*` substring (not the formula) to reflect the spec's stated rules. Re-run until pass.

- [ ] **Step 5: Commit**

```
git add shared/teacher-style/test_fixtures.json apps/evals/shared/test_teacher_style.py
git commit -m "feat(teacher-style): canonical parity fixtures + py parity test"
```
