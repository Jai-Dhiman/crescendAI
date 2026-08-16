# Python Style Guide

## CrescendAI research and tooling Python

Scope: `model/`, `apps/evals/`, and repo-level `scripts/`. Mechanical
enforcement is ruff (`model/pyproject.toml` `[tool.ruff.lint]`), run per
changed line by `scripts/lint_changed.py` at pre-commit; each rule below names
its ruff code where one exists. Cite the id when reporting a violation.

Governing philosophy: minimum code that solves the problem. Nothing
speculative: no abstractions for single-use code, no unrequested flexibility,
no error handling for impossible scenarios. Explicit failures over silent
fallbacks -- in this codebase a silently-wrong number is worse than a crash,
because it corrupts a measurement instead of stopping one.

## 1. Exceptions

- **[PY-ERR-001]** (MUST) No `except Exception`/bare `except` that neither
  re-raises nor routes through `logging.exception`; a deliberate broad catch
  carries `# noqa: BLE001` plus a reason. Ruff: `BLE001`.
- **[PY-ERR-002]** (MUST) No `except: pass`. Swallow-and-continue on a real
  error is the exact silent fallback this repo bans; re-raise, log with a
  reason, or delete the try/except.
- **[PY-ERR-003]** (SHOULD) Re-raises inside `except` chain with
  `raise ... from e` (or `from None` deliberately). Ruff: `B904`.
- **[PY-ERR-004]** (SHOULD) Do not raise inside a `try` only to catch it in
  the same function's `except`; restructure as a direct conditional raise.
  Ruff: `TRY301`.

## 2. Structure

- **[PY-002]** (MUST) No mutable default arguments; mutable class/dataclass
  defaults use `field(default_factory=...)`. Ruff: `B006`, `B008`, `RUF012`.
- **[PY-003]** (SHOULD) `zip(...)` over iterables whose lengths must match
  passes `strict=True`; silent truncation on a length mismatch is a silent
  fallback. Ruff: `B905`.
- **[PY-004]** (SHOULD) No unused function/method parameters; dead parameters
  are speculative interface. Remove them or prefix `_`. Ruff: `ARG`.
- **[PY-005]** (SHOULD) File access goes through `pathlib`
  (`Path.open()`, not builtin `open()`/`os.path`). Ruff: `PTH`.
- **[PY-006]** (SHOULD) No assign-then-immediately-return, no `else` after
  `return`/`raise`. Ruff: `RET504`, `RET505`, `RET506`, `RET508`.
- **[PY-008]** (MUST) No stale `# noqa` suppressing a violation that no
  longer fires. Ruff: `RUF100` (auto-fixable).
- **[PY-009]** (SHOULD) New code uses `numpy.random.default_rng(seed)` passed
  explicitly, never the global `np.random.seed()` state -- implicit RNG state
  is invisible non-reproducibility in a codebase built on ex-ante gates.
  Ruff: `NPY002`.

## 3. CLI and paths

- **[PY-001]** (MUST) CLI default paths are `__file__`-anchored, never
  CWD-relative; `just` recipes shift CWD. (Predates this doc; previously
  sourced from the root CLAUDE.md.)
- **[PY-ML-002]** (SHOULD) Entry points under `model/scripts/`,
  `model/tools/`, and `model/src/**` default filesystem paths from
  `model/src/paths.py` rather than re-deriving their own `__file__` anchors;
  one path authority, not forty slightly different ones.

## 4. Measurement integrity

- **[PY-ML-001]** (MUST) Code handling missing transcription, audio, or
  ground truth raises a named exception describing the missing artifact. It
  never substitutes clean MIDI, synthetic audio, cached output, or a weaker
  model, and never returns a sentinel (`None`, `0`, `NaN`) that a downstream
  aggregator could average into a metric. (This is `model/CLAUDE.md` law made
  citable.)
- **[PY-ML-003]** (MAY) No new `@dataclass` config object for a script or
  function with exactly one call site; keyword arguments or module constants
  suffice until a second caller exists.

## 5. Deliberate non-rules

`TRY003` (long messages in plain `raise ValueError(...)`) stays permanently
ignored in ruff config: explicit inline messages are how errors self-document
here, and 700+ call sites already use the style deliberately.
