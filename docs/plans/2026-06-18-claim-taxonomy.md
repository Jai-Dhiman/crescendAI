# Claim Taxonomy Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** Produce `claim_taxonomy.json` (versioned, schema-validated) + `verdict_dispatch.py` (routing stub) + `baseline_v1_audit.json` (~30–50 hand-decomposed claims) so that issue #65 (verifier) has an unambiguous, tested contract to implement against.
**Spec:** docs/specs/2026-06-18-claim-taxonomy-design.md
**Style:** Follow the project's coding standards (CLAUDE.md / AGENTS.md). Python: uv. No emojis. Explicit exception handling (TypeErrors for bad inputs, never silent fallbacks).

---

## Task Groups

Group 0 (must complete first, unlocks everything): Task 1
Group A (parallel, depends on Group 0): Task 2, Task 3
Group B (sequential, depends on Group A): Task 4
Group C (parallel, independent of A and B, depends on Group 0): Task 5

---

## Task 1: JSON Schema + Package Scaffold + Schema Self-Validation Test

**Group:** 0 (must complete before any other task)

**Behavior being verified:** `claim_taxonomy.schema.json` is a valid JSON Schema (draft 2020-12) and a hand-authored taxonomy skeleton validates against it without errors.

**Interface under test:** `jsonschema.validate(instance, schema)` — the public jsonschema library API.

**Files:**
- Create: `apps/evals/claim_taxonomy/__init__.py`
- Create: `apps/evals/claim_taxonomy/tests/__init__.py`
- Create: `apps/evals/claim_taxonomy/audit/__init__.py`
- Create: `apps/evals/claim_taxonomy/claim_taxonomy.schema.json`
- Create: `apps/evals/claim_taxonomy/tests/test_schema_validates.py`
- Modify: `apps/evals/pyproject.toml` (add `claim_taxonomy` to hatch wheel packages)

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/claim_taxonomy/tests/test_schema_validates.py
"""Verify that claim_taxonomy.schema.json is self-consistent and that a
minimal taxonomy skeleton validates against it without errors."""
from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

TAXONOMY_DIR = Path(__file__).resolve().parents[1]
SCHEMA_PATH = TAXONOMY_DIR / "claim_taxonomy.schema.json"


def _load_schema() -> dict:
    with open(SCHEMA_PATH) as f:
        return json.load(f)


def test_schema_file_exists() -> None:
    assert SCHEMA_PATH.exists(), f"Schema not found at {SCHEMA_PATH}"


def test_schema_has_required_top_level_keys() -> None:
    schema = _load_schema()
    for key in ("$schema", "$id", "type", "required", "properties"):
        assert key in schema, f"Schema missing top-level key: {key}"


def test_schema_draft_is_2020_12() -> None:
    schema = _load_schema()
    assert "2020-12" in schema.get("$schema", ""), (
        "Schema must use JSON Schema draft 2020-12"
    )


def test_minimal_valid_taxonomy_skeleton_passes_schema() -> None:
    """A skeleton with all required top-level fields passes validation."""
    schema = _load_schema()
    skeleton = {
        "taxonomy_version": "v0",
        "extractor_judge_boundary": {
            "description": "test",
            "llm_in_truth_label": False,
        },
        "claim_schema": {
            "type": "object",
            "required": ["proposition", "dimension", "location", "polarity"],
            "properties": {
                "proposition": {"type": "string"},
                "dimension": {"type": "string"},
                "location": {},
                "polarity": {"type": "string"},
                "magnitude": {},
            },
        },
        "dimensions": {
            "timing": {
                "status": "active",
                "reference": "established_tempo",
                "check": "signed_tempo_deviation",
                "tolerance": {
                    "name": "signed_tempo_deviation",
                    "provisional": 8.0,
                    "unit": "percent",
                    "calibration_source": "#65/M1 error-bar study",
                    "locked": False,
                },
                "reliability_tier": 1,
                "measurement": "amt_onsets_region_tempo_fit",
                "minimum_events": 8,
            },
            "phrasing": {
                "status": "scoped_out",
                "rationale": "perceptual",
            },
        },
        "verdict_spec": {
            "labels": ["SUPPORTED", "REFUTED", "UNVERIFIABLE"],
            "unverifiable_reason_codes": [
                "out_of_scope_dim",
                "gated_dim",
                "unlocalizable",
                "substrate_failure",
                "region_too_short",
                "near_threshold",
            ],
            "faithfulness_formula": "SUPPORTED / (SUPPORTED + REFUTED)",
            "coverage_formula": "(SUPPORTED + REFUTED) / total_claims",
            "unverifiable_reporting": "histogram_by_reason_code",
        },
    }
    # Must not raise
    jsonschema.validate(instance=skeleton, schema=schema)


def test_taxonomy_missing_required_field_fails_schema() -> None:
    """A skeleton missing taxonomy_version must fail schema validation."""
    schema = _load_schema()
    bad = {
        # missing taxonomy_version
        "extractor_judge_boundary": {"description": "x", "llm_in_truth_label": False},
        "claim_schema": {},
        "dimensions": {},
        "verdict_spec": {},
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=bad, schema=schema)
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-63-claim-taxonomy/apps/evals && uv run pytest claim_taxonomy/tests/test_schema_validates.py -v 2>&1 | head -40
```
Expected: FAIL — `ModuleNotFoundError: No module named 'claim_taxonomy'` or `FileNotFoundError` on SCHEMA_PATH (schema file does not exist yet).

- [ ] **Step 3: Implement the minimum to make the test pass**

Create the package init files first:

```python
# apps/evals/claim_taxonomy/__init__.py
```

```python
# apps/evals/claim_taxonomy/tests/__init__.py
```

```python
# apps/evals/claim_taxonomy/audit/__init__.py
```

Then register the new package in the hatch wheel build so issues #65 and #67 can import `claim_taxonomy` when the evals package is installed (not just under local pytest). Edit `apps/evals/pyproject.toml`:

Change the line:
```toml
packages = ["shared", "pipeline", "model", "memory", "inference"]
```
to:
```toml
packages = ["shared", "pipeline", "model", "memory", "inference", "claim_taxonomy"]
```

Then create the JSON Schema. This schema validates `claim_taxonomy.json` (the full taxonomy artifact, not individual claims):

```json
// apps/evals/claim_taxonomy/claim_taxonomy.schema.json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://crescend.ai/schemas/claim_taxonomy/v0",
  "title": "CrescendAI Claim Taxonomy",
  "description": "Versioned taxonomy of (dimension, location) music-feedback claims for the deterministic verifier.",
  "type": "object",
  "required": [
    "taxonomy_version",
    "extractor_judge_boundary",
    "claim_schema",
    "dimensions",
    "verdict_spec"
  ],
  "properties": {
    "taxonomy_version": {
      "type": "string",
      "description": "Semantic version of this taxonomy artifact."
    },
    "extractor_judge_boundary": {
      "type": "object",
      "required": ["description", "llm_in_truth_label"],
      "properties": {
        "description": { "type": "string" },
        "llm_in_truth_label": {
          "type": "boolean",
          "const": false,
          "description": "Must be false: the verifier's truth label never invokes an LLM."
        }
      },
      "additionalProperties": false
    },
    "claim_schema": {
      "type": "object",
      "description": "JSON Schema fragment describing a single structured claim."
    },
    "dimensions": {
      "type": "object",
      "description": "Registry of all musical dimensions and their verifiability status.",
      "additionalProperties": {
        "$ref": "#/$defs/dimension_entry"
      }
    },
    "verdict_spec": {
      "type": "object",
      "required": [
        "labels",
        "unverifiable_reason_codes",
        "faithfulness_formula",
        "coverage_formula",
        "unverifiable_reporting"
      ],
      "properties": {
        "labels": {
          "type": "array",
          "items": { "type": "string" }
        },
        "unverifiable_reason_codes": {
          "type": "array",
          "items": { "type": "string" }
        },
        "faithfulness_formula": { "type": "string" },
        "coverage_formula": { "type": "string" },
        "unverifiable_reporting": { "type": "string" }
      }
    }
  },
  "$defs": {
    "provisional_tolerance": {
      "type": "object",
      "required": ["name", "provisional", "unit", "calibration_source", "locked"],
      "properties": {
        "name": { "type": "string" },
        "provisional": { "type": "number" },
        "unit": { "type": "string" },
        "calibration_source": { "type": "string" },
        "locked": { "type": "boolean", "const": false }
      },
      "additionalProperties": false
    },
    "active_dimension": {
      "type": "object",
      "required": [
        "status",
        "reference",
        "check",
        "tolerance",
        "reliability_tier",
        "measurement",
        "minimum_events"
      ],
      "properties": {
        "status": { "type": "string", "const": "active" },
        "reference": { "type": "string" },
        "check": { "type": "string" },
        "tolerance": { "$ref": "#/$defs/provisional_tolerance" },
        "reliability_tier": { "type": "integer", "minimum": 1, "maximum": 3 },
        "measurement": { "type": "string" },
        "minimum_events": { "type": "integer", "minimum": 1 }
      }
    },
    "gated_dimension": {
      "type": "object",
      "required": ["status", "gate_id", "reference", "check"],
      "properties": {
        "status": { "type": "string", "const": "gated_on_measurement" },
        "gate_id": { "type": "string" },
        "reference": { "type": "string" },
        "check": { "type": "string" },
        "rationale": { "type": "string" }
      }
    },
    "scoped_out_dimension": {
      "type": "object",
      "required": ["status", "rationale"],
      "properties": {
        "status": { "type": "string", "const": "scoped_out" },
        "rationale": { "type": "string" }
      }
    },
    "dimension_entry": {
      "oneOf": [
        { "$ref": "#/$defs/active_dimension" },
        { "$ref": "#/$defs/gated_dimension" },
        { "$ref": "#/$defs/scoped_out_dimension" }
      ]
    }
  }
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-63-claim-taxonomy/apps/evals && uv run pytest claim_taxonomy/tests/test_schema_validates.py -v 2>&1 | tail -20
```
Expected: PASS — all 5 tests green.

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-63-claim-taxonomy && git add apps/evals/claim_taxonomy/__init__.py apps/evals/claim_taxonomy/tests/__init__.py apps/evals/claim_taxonomy/audit/__init__.py apps/evals/claim_taxonomy/claim_taxonomy.schema.json apps/evals/claim_taxonomy/tests/test_schema_validates.py apps/evals/pyproject.toml && git commit -m "feat(taxonomy): JSON Schema + package scaffold + schema self-validation test (#63)"
```

---

## Task 2: `claim_taxonomy.json` — Full Taxonomy Artifact

**Group:** A (parallel with Task 3; depends on Group 0 / Task 1)

**Behavior being verified:** `claim_taxonomy.json` validates against `claim_taxonomy.schema.json` without errors, with all 7 dimensions present and every active dimension having a complete registry entry.

**Interface under test:** `jsonschema.validate(claim_taxonomy_json, schema_json)`.

**Files:**
- Create: `apps/evals/claim_taxonomy/claim_taxonomy.json`
- Modify: `apps/evals/claim_taxonomy/tests/test_schema_validates.py` (add two new tests)

- [ ] **Step 1: Write the failing test**

Add to `apps/evals/claim_taxonomy/tests/test_schema_validates.py`:

```python
TAXONOMY_PATH = TAXONOMY_DIR / "claim_taxonomy.json"


def test_claim_taxonomy_json_exists() -> None:
    assert TAXONOMY_PATH.exists(), f"Taxonomy not found at {TAXONOMY_PATH}"


def test_claim_taxonomy_json_validates_against_schema() -> None:
    """Full claim_taxonomy.json must pass JSON Schema validation."""
    schema = _load_schema()
    with open(TAXONOMY_PATH) as f:
        taxonomy = json.load(f)
    # Must not raise
    jsonschema.validate(instance=taxonomy, schema=schema)


def test_all_seven_dimensions_present() -> None:
    with open(TAXONOMY_PATH) as f:
        taxonomy = json.load(f)
    expected = {
        "timing", "pedaling", "dynamics", "articulation",
        "phrasing", "interpretation", "timbre"
    }
    actual = set(taxonomy["dimensions"].keys())
    assert actual == expected, f"Missing: {expected - actual}, Extra: {actual - expected}"


def test_active_dimensions_have_complete_registry() -> None:
    with open(TAXONOMY_PATH) as f:
        taxonomy = json.load(f)
    required_active_fields = {
        "status", "reference", "check", "tolerance",
        "reliability_tier", "measurement", "minimum_events"
    }
    for dim_name, dim in taxonomy["dimensions"].items():
        if dim["status"] == "active":
            missing = required_active_fields - set(dim.keys())
            assert not missing, (
                f"Active dimension '{dim_name}' missing fields: {missing}"
            )


def test_all_tolerances_are_provisional() -> None:
    with open(TAXONOMY_PATH) as f:
        taxonomy = json.load(f)
    for dim_name, dim in taxonomy["dimensions"].items():
        if dim.get("status") == "active":
            tol = dim["tolerance"]
            assert tol["locked"] is False, (
                f"Dimension '{dim_name}': tolerance must be locked=false (provisional)"
            )
            assert tol["calibration_source"] == "#65/M1 error-bar study", (
                f"Dimension '{dim_name}': calibration_source must point to #65/M1"
            )


def test_extractor_judge_boundary_llm_flag_is_false() -> None:
    with open(TAXONOMY_PATH) as f:
        taxonomy = json.load(f)
    assert taxonomy["extractor_judge_boundary"]["llm_in_truth_label"] is False
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-63-claim-taxonomy/apps/evals && uv run pytest claim_taxonomy/tests/test_schema_validates.py::test_claim_taxonomy_json_exists -v 2>&1 | tail -10
```
Expected: FAIL — `AssertionError: Taxonomy not found` (file does not exist yet).

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `apps/evals/claim_taxonomy/claim_taxonomy.json`:

```json
{
  "taxonomy_version": "v0",
  "extractor_judge_boundary": {
    "description": "An LLM may be used as the claim extractor (prose to structured claim). The verifier's truth label (SUPPORTED | REFUTED | UNVERIFIABLE) must never invoke an LLM. This boundary is the PROVE/ViCrit non-circularity defense: the checker's correctness is grounded in a programmatic oracle over a measured signal, not any model's opinion.",
    "llm_in_truth_label": false
  },
  "claim_schema": {
    "type": "object",
    "required": ["proposition", "dimension", "location", "polarity"],
    "properties": {
      "proposition": {
        "type": "string",
        "description": "Free-text source claim from the teacher's prose output."
      },
      "dimension": {
        "type": "string",
        "enum": ["timing", "pedaling", "dynamics", "articulation", "phrasing", "interpretation", "timbre"],
        "description": "Musical dimension the claim is about."
      },
      "location": {
        "oneOf": [
          {
            "type": "object",
            "required": ["bar_start", "bar_end"],
            "properties": {
              "bar_start": { "type": "integer", "minimum": 1 },
              "bar_end": { "type": "integer", "minimum": 1 }
            },
            "additionalProperties": false
          },
          {
            "type": "string",
            "const": "whole_piece"
          }
        ],
        "description": "Bar-range [bar_start, bar_end] (inclusive) or 'whole_piece'. Beat granularity is not admissible."
      },
      "polarity": {
        "type": "string",
        "enum": ["+", "-", "neutral"],
        "description": "+: signed anomaly in the positive direction. -: signed anomaly in the negative direction. neutral: asserts a virtue or absence-of-problem."
      },
      "magnitude": {
        "oneOf": [
          {
            "type": "object",
            "required": ["value", "unit"],
            "properties": {
              "value": { "type": "number" },
              "unit": { "type": "string" }
            },
            "additionalProperties": false
          },
          { "type": "null" }
        ],
        "description": "Optional. When present, tightens the check. When null, falls back to sign+tolerance check."
      }
    },
    "additionalProperties": false
  },
  "dimensions": {
    "timing": {
      "status": "active",
      "reference": "established_tempo",
      "check": "signed_tempo_deviation",
      "tolerance": {
        "name": "signed_tempo_deviation",
        "provisional": 8.0,
        "unit": "percent",
        "calibration_source": "#65/M1 error-bar study",
        "locked": false
      },
      "reliability_tier": 1,
      "measurement": "amt_onsets_region_tempo_fit",
      "minimum_events": 8,
      "notes": "Tolerance is a percent deviation from established_tempo over the region, NOT absolute onset-ms. This is robust to alignment jitter. Half-pedal and flutter pedaling are out of scope."
    },
    "pedaling": {
      "status": "active",
      "reference": "self_density",
      "check": "pedal_presence_density",
      "tolerance": {
        "name": "pedal_presence_density_deviation",
        "provisional": 0.25,
        "unit": "fraction",
        "calibration_source": "#65/M1 error-bar study",
        "locked": false
      },
      "reliability_tier": 2,
      "measurement": "amt_sustain_pedal_events",
      "minimum_events": 2,
      "notes": "Coarse on/off and density only. Half-pedal and flutter pedaling are out of scope. reference is self_density: fraction of bars with at least one pedal event in the same piece."
    },
    "dynamics": {
      "status": "gated_on_measurement",
      "gate_id": "rms_lufs_estimator",
      "reference": "within_region_range",
      "check": "dynamic_range_zscore_and_contour_slope_sign",
      "rationale": "No audio-domain loudness estimator exists in the repo (only MIDI velocity, which is not perceived loudness). Gate: rms_lufs_estimator must be validated before this dimension is admissible. Relative within-region loudness only; absolute dB is inadmissible because recording gain is uncontrolled."
    },
    "articulation": {
      "status": "gated_on_measurement",
      "gate_id": "offset_accuracy_validation",
      "reference": "self_overlap_ratio",
      "check": "legato_staccato_overlap_ratio",
      "rationale": "AMT offsets are weaker than onsets (higher quantization error). Gate: offset_accuracy_validation must confirm offset reliability before this dimension is admissible. Region-level only; single-bar articulation claims are too fragile."
    },
    "phrasing": {
      "status": "scoped_out",
      "rationale": "Perceptual. The only available proxy is itself a perceptual model output, which reintroduces circularity."
    },
    "interpretation": {
      "status": "scoped_out",
      "rationale": "Definitionally subjective. No deterministic physical oracle exists."
    },
    "timbre": {
      "status": "scoped_out",
      "rationale": "No physical proxy. Timbre is a multi-dimensional perceptual construct with no single measurable correlate in the available signal chain."
    }
  },
  "verdict_spec": {
    "labels": ["SUPPORTED", "REFUTED", "UNVERIFIABLE"],
    "unverifiable_reason_codes": [
      "out_of_scope_dim",
      "gated_dim",
      "unlocalizable",
      "substrate_failure",
      "region_too_short",
      "near_threshold"
    ],
    "dispatch_order": [
      "1. If dimension.status == 'scoped_out' → UNVERIFIABLE(out_of_scope_dim)",
      "2. If dimension.status == 'gated_on_measurement' and gate not validated → UNVERIFIABLE(gated_dim)",
      "3. If location is not resolvable (no score in catalog / piece not identified / span < alignment_uncertainty) → UNVERIFIABLE(unlocalizable)",
      "4. If measurement substrate raised an error → UNVERIFIABLE(substrate_failure)",
      "5. If region has fewer than dimension.minimum_events → UNVERIFIABLE(region_too_short)",
      "6. Compute signed deviation d vs reference",
      "7. If abs(d - tau) <= error_bar → UNVERIFIABLE(near_threshold)",
      "8. If d confirms polarity direction beyond tau → SUPPORTED",
      "9. Otherwise → REFUTED"
    ],
    "span_vs_uncertainty_failsafe": "Verifier must downgrade to UNVERIFIABLE(unlocalizable) when alignment_uncertainty >= location_span. location_span = bar_end - bar_start + 1 for bar-range; undefined (infinite) for whole_piece.",
    "faithfulness_formula": "SUPPORTED / (SUPPORTED + REFUTED)",
    "coverage_formula": "(SUPPORTED + REFUTED) / total_claims",
    "unverifiable_reporting": "histogram_by_reason_code",
    "note": "UNVERIFIABLE claims are never folded into faithfulness or coverage. They are reported as a separate histogram by typed reason code."
  }
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-63-claim-taxonomy/apps/evals && uv run pytest claim_taxonomy/tests/test_schema_validates.py -v 2>&1 | tail -20
```
Expected: PASS — all tests (original 5 + 6 new) green.

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-63-claim-taxonomy && git add apps/evals/claim_taxonomy/claim_taxonomy.json apps/evals/claim_taxonomy/tests/test_schema_validates.py && git commit -m "feat(taxonomy): claim_taxonomy.json v0 with all 7 dimensions + extended schema tests (#63)"
```

---

## Task 3: `verdict_dispatch.py` — Routing Stub

**Group:** A (parallel with Task 2; depends on Group 0 / Task 1)

**Behavior being verified:** `route_verdict(claim, registry)` returns the correct `(verdict, reason_code)` tuple for each of the 9 dispatch steps, using a synthetic registry and mock deviation values. No real measurement is performed.

**Interface under test:** `verdict_dispatch.route_verdict(claim: dict, registry: dict) -> tuple[str, str | None]`

**Files:**
- Create: `apps/evals/claim_taxonomy/verdict_dispatch.py`
- Create: `apps/evals/claim_taxonomy/tests/test_verdict_dispatch.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/claim_taxonomy/tests/test_verdict_dispatch.py
"""Round-trip tests for verdict_dispatch.route_verdict.

Control-flow dispatch only — no real measurements. The registry entries
supply synthetic deviation/error_bar values directly so the dispatch logic
can be verified independently of measurement substrate.
"""
from __future__ import annotations

import pytest

from claim_taxonomy.verdict_dispatch import route_verdict


# ---------------------------------------------------------------------------
# Registry fixtures
# ---------------------------------------------------------------------------

ACTIVE_DIM = {
    "status": "active",
    "reference": "established_tempo",
    "check": "signed_tempo_deviation",
    "tolerance": {
        "name": "signed_tempo_deviation",
        "provisional": 8.0,
        "unit": "percent",
        "calibration_source": "#65/M1 error-bar study",
        "locked": False,
    },
    "reliability_tier": 1,
    "measurement": "amt_onsets_region_tempo_fit",
    "minimum_events": 8,
}

GATED_DIM = {
    "status": "gated_on_measurement",
    "gate_id": "rms_lufs_estimator",
    "reference": "within_region_range",
    "check": "dynamic_range_zscore_and_contour_slope_sign",
}

SCOPED_DIM = {
    "status": "scoped_out",
    "rationale": "perceptual",
}

REGISTRY = {
    "timing": ACTIVE_DIM,
    "dynamics": GATED_DIM,
    "phrasing": SCOPED_DIM,
}

# A base claim that is fully specified for an active dimension
BASE_TIMING_CLAIM = {
    "proposition": "You rushed in bars 10-14",
    "dimension": "timing",
    "location": {"bar_start": 10, "bar_end": 14},
    "polarity": "-",
    "magnitude": None,
    # Measurement context fields supplied by the caller (the verifier, issue #65)
    # For this stub: synthesized to test dispatch branches
    "_measurement": {
        "d": -12.0,          # signed deviation (negative = rushed)
        "tau": 8.0,          # tolerance threshold
        "error_bar": 2.0,    # measurement error bar
        "event_count": 20,   # events in region
        "localizable": True,
    },
}


def _make_claim(**overrides) -> dict:
    import copy
    claim = copy.deepcopy(BASE_TIMING_CLAIM)
    claim.update(overrides)
    return claim


def _make_measurement(**overrides) -> dict:
    import copy
    m = copy.deepcopy(BASE_TIMING_CLAIM["_measurement"])
    m.update(overrides)
    return m


# ---------------------------------------------------------------------------
# Dispatch step 1: scoped_out → UNVERIFIABLE(out_of_scope_dim)
# ---------------------------------------------------------------------------

def test_scoped_out_dimension_returns_unverifiable_out_of_scope() -> None:
    claim = _make_claim(dimension="phrasing", _measurement=_make_measurement())
    verdict, reason = route_verdict(claim, REGISTRY)
    assert verdict == "UNVERIFIABLE"
    assert reason == "out_of_scope_dim"


# ---------------------------------------------------------------------------
# Dispatch step 2: gated_on_measurement → UNVERIFIABLE(gated_dim)
# ---------------------------------------------------------------------------

def test_gated_dimension_returns_unverifiable_gated_dim() -> None:
    claim = _make_claim(dimension="dynamics", _measurement=_make_measurement())
    verdict, reason = route_verdict(claim, REGISTRY)
    assert verdict == "UNVERIFIABLE"
    assert reason == "gated_dim"


# ---------------------------------------------------------------------------
# Dispatch step 3: not localizable → UNVERIFIABLE(unlocalizable)
# ---------------------------------------------------------------------------

def test_unlocalizable_claim_returns_unverifiable() -> None:
    claim = _make_claim(
        _measurement=_make_measurement(localizable=False)
    )
    verdict, reason = route_verdict(claim, REGISTRY)
    assert verdict == "UNVERIFIABLE"
    assert reason == "unlocalizable"


# ---------------------------------------------------------------------------
# Dispatch step 4: substrate_failure → UNVERIFIABLE(substrate_failure)
# ---------------------------------------------------------------------------

def test_substrate_failure_returns_unverifiable() -> None:
    claim = _make_claim(
        _measurement=_make_measurement(substrate_failure=True)
    )
    verdict, reason = route_verdict(claim, REGISTRY)
    assert verdict == "UNVERIFIABLE"
    assert reason == "substrate_failure"


# ---------------------------------------------------------------------------
# Dispatch step 5: region_too_short → UNVERIFIABLE(region_too_short)
# ---------------------------------------------------------------------------

def test_region_too_short_returns_unverifiable() -> None:
    claim = _make_claim(
        _measurement=_make_measurement(event_count=3)  # < minimum_events=8
    )
    verdict, reason = route_verdict(claim, REGISTRY)
    assert verdict == "UNVERIFIABLE"
    assert reason == "region_too_short"


# ---------------------------------------------------------------------------
# Dispatch step 7: near_threshold → UNVERIFIABLE(near_threshold)
# ---------------------------------------------------------------------------

def test_near_threshold_returns_unverifiable() -> None:
    # d = -9.0, tau = 8.0, error_bar = 2.0 → |d - tau| = 1.0 <= 2.0
    claim = _make_claim(
        _measurement=_make_measurement(d=-9.0, tau=8.0, error_bar=2.0)
    )
    verdict, reason = route_verdict(claim, REGISTRY)
    assert verdict == "UNVERIFIABLE"
    assert reason == "near_threshold"


# ---------------------------------------------------------------------------
# Dispatch step 8: d confirms polarity → SUPPORTED
# ---------------------------------------------------------------------------

def test_claim_confirming_polarity_returns_supported() -> None:
    # polarity="-", d=-12.0, tau=8.0, error_bar=2.0 → |d - tau| = 4.0 > 2.0
    # d is negative, polarity is negative → SUPPORTED
    claim = _make_claim(
        _measurement=_make_measurement(d=-12.0, tau=8.0, error_bar=2.0)
    )
    verdict, reason = route_verdict(claim, REGISTRY)
    assert verdict == "SUPPORTED"
    assert reason is None


def test_neutral_polarity_with_no_anomaly_returns_supported() -> None:
    # polarity="neutral" asserts absence-of-problem
    # d near zero (no anomaly detected) → SUPPORTED
    claim = _make_claim(polarity="neutral",
                        _measurement=_make_measurement(d=1.0, tau=8.0, error_bar=2.0))
    verdict, reason = route_verdict(claim, REGISTRY)
    assert verdict == "SUPPORTED"
    assert reason is None


# ---------------------------------------------------------------------------
# Dispatch step 9: d does NOT confirm polarity → REFUTED
# ---------------------------------------------------------------------------

def test_wrong_direction_claim_returns_refuted() -> None:
    # polarity="-" (claims rushed) but d=+12.0 (actually dragged) → REFUTED
    claim = _make_claim(
        polarity="-",
        _measurement=_make_measurement(d=12.0, tau=8.0, error_bar=2.0)
    )
    verdict, reason = route_verdict(claim, REGISTRY)
    assert verdict == "REFUTED"
    assert reason is None


def test_fabricated_anomaly_where_none_exists_returns_refuted() -> None:
    # polarity="-", d=-3.0 (within tolerance, no real rush), error_bar=1.0
    # |d - tau| = |3 - 8| = 5 > 1.0 but d does not confirm polarity (|d|=3 < tau=8)
    claim = _make_claim(
        polarity="-",
        _measurement=_make_measurement(d=-3.0, tau=8.0, error_bar=1.0)
    )
    verdict, reason = route_verdict(claim, REGISTRY)
    assert verdict == "REFUTED"
    assert reason is None


# ---------------------------------------------------------------------------
# Error handling: unknown dimension raises TypeError
# ---------------------------------------------------------------------------

def test_unknown_dimension_raises_type_error() -> None:
    claim = _make_claim(dimension="nonexistent_dim")
    with pytest.raises(TypeError, match="Unknown dimension"):
        route_verdict(claim, REGISTRY)


def test_missing_measurement_context_raises_type_error() -> None:
    claim = {
        "proposition": "test",
        "dimension": "timing",
        "location": {"bar_start": 1, "bar_end": 4},
        "polarity": "-",
        "magnitude": None,
        # no _measurement key
    }
    with pytest.raises(TypeError, match="_measurement"):
        route_verdict(claim, REGISTRY)
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-63-claim-taxonomy/apps/evals && uv run pytest claim_taxonomy/tests/test_verdict_dispatch.py -v 2>&1 | head -20
```
Expected: FAIL — `ModuleNotFoundError: No module named 'claim_taxonomy.verdict_dispatch'` (file does not exist yet).

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# apps/evals/claim_taxonomy/verdict_dispatch.py
"""Verdict routing stub for the claim taxonomy verifier.

Control-flow dispatch only — no real measurement is performed here.
The verifier (issue #65) supplies measurement context via the claim's
`_measurement` dict. This module routes the claim through the 9-step
dispatch chain and returns (verdict, reason_code | None).

Explicit exception handling: unknown dimension or missing fields raise
TypeError immediately (never silently degrade).
"""
from __future__ import annotations


def route_verdict(
    claim: dict,
    registry: dict,
) -> tuple[str, str | None]:
    """Route a structured claim to SUPPORTED | REFUTED | UNVERIFIABLE.

    Args:
        claim: A structured claim dict. Must include:
            - dimension: str
            - polarity: "+" | "-" | "neutral"
            - _measurement: dict with keys:
                - d: float              signed deviation from reference
                - tau: float            tolerance threshold
                - error_bar: float      measurement error bar
                - event_count: int      events found in the region
                - localizable: bool     whether the location was resolved
                - substrate_failure: bool (optional, default False)
        registry: The dimensions dict from claim_taxonomy.json.

    Returns:
        (verdict, reason_code) where verdict in {SUPPORTED, REFUTED, UNVERIFIABLE}
        and reason_code is one of the typed codes or None.

    Raises:
        TypeError: if dimension is not in registry, or _measurement is missing.
    """
    dimension_name = claim.get("dimension")
    if dimension_name not in registry:
        raise TypeError(
            f"Unknown dimension '{dimension_name}'. "
            f"Known: {list(registry.keys())}"
        )

    if "_measurement" not in claim:
        raise TypeError(
            "claim must include a '_measurement' dict with measurement context. "
            "The verifier (issue #65) populates this before calling route_verdict."
        )

    dim = registry[dimension_name]
    m = claim["_measurement"]
    polarity = claim["polarity"]

    # Step 1: scoped_out
    if dim["status"] == "scoped_out":
        return ("UNVERIFIABLE", "out_of_scope_dim")

    # Step 2: gated_on_measurement
    if dim["status"] == "gated_on_measurement":
        return ("UNVERIFIABLE", "gated_dim")

    # Step 3: not localizable
    if not m.get("localizable", True):
        return ("UNVERIFIABLE", "unlocalizable")

    # Step 4: substrate failure
    if m.get("substrate_failure", False):
        return ("UNVERIFIABLE", "substrate_failure")

    # Step 5: region too short
    minimum_events = dim.get("minimum_events", 1)
    if m["event_count"] < minimum_events:
        return ("UNVERIFIABLE", "region_too_short")

    d = m["d"]
    tau = m["tau"]
    error_bar = m["error_bar"]

    # Step 7: near threshold
    if abs(abs(d) - tau) <= error_bar:
        return ("UNVERIFIABLE", "near_threshold")

    # Step 8 & 9: polarity confirmation
    if polarity == "+":
        supported = d > 0 and abs(d) > tau
    elif polarity == "-":
        supported = d < 0 and abs(d) > tau
    else:
        # neutral: asserts virtue / absence-of-problem → supported if no anomaly
        supported = abs(d) <= tau

    if supported:
        return ("SUPPORTED", None)
    return ("REFUTED", None)
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-63-claim-taxonomy/apps/evals && uv run pytest claim_taxonomy/tests/test_verdict_dispatch.py -v 2>&1 | tail -20
```
Expected: PASS — all 12 tests green.

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-63-claim-taxonomy && git add apps/evals/claim_taxonomy/verdict_dispatch.py apps/evals/claim_taxonomy/tests/test_verdict_dispatch.py && git commit -m "feat(taxonomy): verdict_dispatch.py routing stub + round-trip tests (#63)"
```

---

## Task 4: `claim_taxonomy.json` Validates Against Schema (Integration Test)

**Group:** B (sequential, depends on Group A: both Task 2 and Task 3 must complete)

**Behavior being verified:** The full `claim_taxonomy.json` produced in Task 2 validates against the schema AND all example claims for active dimensions route through `route_verdict` to the correct verdict branch using the taxonomy's own dimension registry.

**Interface under test:** `jsonschema.validate` + `route_verdict` — the same public interfaces as Tasks 2 and 3, but now integrated end-to-end against the real taxonomy artifact.

**Files:**
- Create: `apps/evals/claim_taxonomy/tests/test_round_trip.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/claim_taxonomy/tests/test_round_trip.py
"""End-to-end round-trip: claim_taxonomy.json → schema valid → route_verdict
for example claims per active dimension.

This test does not perform real measurements. It verifies that:
1. claim_taxonomy.json passes JSON Schema validation.
2. Hand-authored example claims for active dimensions route to the expected
   verdict branch when supplied with synthetic measurement context.
3. Example claims for scoped_out and gated dimensions route to UNVERIFIABLE
   with the correct typed reason codes.
"""
from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from claim_taxonomy.verdict_dispatch import route_verdict

TAXONOMY_DIR = Path(__file__).resolve().parents[1]
TAXONOMY_PATH = TAXONOMY_DIR / "claim_taxonomy.json"
SCHEMA_PATH = TAXONOMY_DIR / "claim_taxonomy.schema.json"


def _load() -> tuple[dict, dict]:
    with open(TAXONOMY_PATH) as f:
        taxonomy = json.load(f)
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    return taxonomy, schema


def test_claim_taxonomy_json_passes_schema() -> None:
    """Taxonomy artifact must validate against the committed schema."""
    taxonomy, schema = _load()
    jsonschema.validate(instance=taxonomy, schema=schema)


def test_timing_claim_supported_routes_correctly() -> None:
    taxonomy, _ = _load()
    registry = taxonomy["dimensions"]
    claim = {
        "proposition": "You rushed through bars 12-16",
        "dimension": "timing",
        "location": {"bar_start": 12, "bar_end": 16},
        "polarity": "-",
        "magnitude": None,
        "_measurement": {
            "d": -14.0,
            "tau": registry["timing"]["tolerance"]["provisional"],
            "error_bar": 2.0,
            "event_count": 25,
            "localizable": True,
        },
    }
    verdict, reason = route_verdict(claim, registry)
    assert verdict == "SUPPORTED"
    assert reason is None


def test_timing_claim_refuted_when_no_rush_detected() -> None:
    taxonomy, _ = _load()
    registry = taxonomy["dimensions"]
    claim = {
        "proposition": "You rushed in bar 3",
        "dimension": "timing",
        "location": {"bar_start": 3, "bar_end": 3},
        "polarity": "-",
        "magnitude": None,
        "_measurement": {
            "d": 2.0,   # slightly faster but within tolerance
            "tau": registry["timing"]["tolerance"]["provisional"],
            "error_bar": 1.5,
            "event_count": 10,
            "localizable": True,
        },
    }
    verdict, reason = route_verdict(claim, registry)
    assert verdict == "REFUTED"
    assert reason is None


def test_pedaling_claim_supported() -> None:
    taxonomy, _ = _load()
    registry = taxonomy["dimensions"]
    claim = {
        "proposition": "Your pedaling was sparse in the opening",
        "dimension": "pedaling",
        "location": {"bar_start": 1, "bar_end": 8},
        "polarity": "-",
        "magnitude": None,
        "_measurement": {
            "d": -0.40,
            "tau": registry["pedaling"]["tolerance"]["provisional"],
            "error_bar": 0.05,
            "event_count": 6,
            "localizable": True,
        },
    }
    verdict, reason = route_verdict(claim, registry)
    assert verdict == "SUPPORTED"
    assert reason is None


def test_dynamics_gated_returns_unverifiable() -> None:
    taxonomy, _ = _load()
    registry = taxonomy["dimensions"]
    claim = {
        "proposition": "Your dynamics were flat throughout",
        "dimension": "dynamics",
        "location": "whole_piece",
        "polarity": "-",
        "magnitude": None,
        "_measurement": {
            "d": -1.5,
            "tau": 1.0,
            "error_bar": 0.2,
            "event_count": 100,
            "localizable": True,
        },
    }
    verdict, reason = route_verdict(claim, registry)
    assert verdict == "UNVERIFIABLE"
    assert reason == "gated_dim"


def test_phrasing_scoped_out_returns_unverifiable() -> None:
    taxonomy, _ = _load()
    registry = taxonomy["dimensions"]
    claim = {
        "proposition": "Your phrasing lacked direction",
        "dimension": "phrasing",
        "location": "whole_piece",
        "polarity": "-",
        "magnitude": None,
        "_measurement": {
            "d": -1.0,
            "tau": 1.0,
            "error_bar": 0.1,
            "event_count": 50,
            "localizable": True,
        },
    }
    verdict, reason = route_verdict(claim, registry)
    assert verdict == "UNVERIFIABLE"
    assert reason == "out_of_scope_dim"


def test_unresolvable_location_returns_unverifiable() -> None:
    taxonomy, _ = _load()
    registry = taxonomy["dimensions"]
    claim = {
        "proposition": "You rushed in bar 5",
        "dimension": "timing",
        "location": {"bar_start": 5, "bar_end": 5},
        "polarity": "-",
        "magnitude": None,
        "_measurement": {
            "d": -15.0,
            "tau": 8.0,
            "error_bar": 2.0,
            "event_count": 12,
            "localizable": False,  # alignment uncertainty > span (1 bar)
        },
    }
    verdict, reason = route_verdict(claim, registry)
    assert verdict == "UNVERIFIABLE"
    assert reason == "unlocalizable"
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-63-claim-taxonomy/apps/evals && uv run pytest claim_taxonomy/tests/test_round_trip.py -v 2>&1 | head -20
```
Expected: FAIL.

This task is an integration test over the artifacts built in Tasks 2 and 3 — it intentionally adds NO new implementation code. The watch-it-fail discipline is satisfied at the test-existence level: `test_round_trip.py` does not exist before this step, so the run fails with a collection error (`file or directory not found: claim_taxonomy/tests/test_round_trip.py`). This is the expected initial failure. Do not write the implementation before confirming this collection failure.

If Tasks 2 and 3 are already merged, the second meaningful failure mode is a real assertion error from one of the routing tests — this is the integration test catching a Task 2/Task 3 bug that the unit tests missed (e.g. a dimension registry entry the round-trip claim exercises differently). That is exactly what this task exists to catch.

- [ ] **Step 3: Implement the minimum to make the test pass**

No new implementation files are created in this task. The "implementation" is the verification that the Task 2 and Task 3 artifacts integrate correctly. Run the test (Step 4). If every assertion passes, this task is done. If any assertion fails, the bug lives in the Task 2 (`claim_taxonomy.json`) or Task 3 (`verdict_dispatch.py`) artifact — go fix it there, re-run that task's own unit tests plus this round-trip test, and amend the relevant earlier commit or add a fix commit scoped to that artifact. Never patch the round-trip test to mask an artifact bug.

The most likely integration bug to surface: the `neutral` polarity branch in `verdict_dispatch.py`. After the near_threshold check (step 7) already removes the dead-band, the neutral SUPPORTED condition must be `abs(d) < tau` (strict), not `abs(d) <= tau`, because the boundary case is owned by near_threshold. Confirm `test_neutral_polarity_with_no_anomaly_returns_supported` (Task 3) and any neutral round-trip claim here both pass with this strict-less-than logic. If they conflict, the fix is in Task 3's `verdict_dispatch.py`, not here.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-63-claim-taxonomy/apps/evals && uv run pytest claim_taxonomy/tests/ -v 2>&1 | tail -30
```
Expected: PASS — all tests in the entire `claim_taxonomy/tests/` suite green.

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-63-claim-taxonomy && git add apps/evals/claim_taxonomy/tests/test_round_trip.py && git commit -m "test(taxonomy): round-trip integration test claim_taxonomy.json → schema → route_verdict (#63)"
```

---

## Task 5: Hand-Decomposed Baseline Audit + Structural Integrity Test

**Group:** C (parallel, depends on Group 0 only; independent of Tasks 2, 3, 4)

**Behavior being verified:** `baseline_v1_audit.json` exists, contains ~30–50 hand-decomposed claims from `baseline_v1.jsonl`, has the required structural fields, and reports a `scoped_out_fraction` between 0 and 1.

**Interface under test:** Loading and validating `baseline_v1_audit.json` as a Python dict.

**Files:**
- Create: `apps/evals/claim_taxonomy/audit/baseline_v1_audit.json`
- Create: `apps/evals/claim_taxonomy/tests/test_audit_integrity.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/claim_taxonomy/tests/test_audit_integrity.py
"""Structural integrity test for the hand-decomposed baseline audit.

This test does not validate the correctness of the manual decomposition
decisions — it verifies that the audit file has the required structure,
that all sample_claims have required fields, and that the scoped_out_fraction
is a valid fraction.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

AUDIT_PATH = (
    Path(__file__).resolve().parents[1] / "audit" / "baseline_v1_audit.json"
)

REQUIRED_CLAIM_FIELDS = {"proposition", "dimension", "location", "polarity"}
VALID_DIMENSIONS = {
    "timing", "pedaling", "dynamics", "articulation",
    "phrasing", "interpretation", "timbre"
}
VALID_POLARITIES = {"+", "-", "neutral"}


def _load_audit() -> dict:
    with open(AUDIT_PATH) as f:
        return json.load(f)


def test_audit_file_exists() -> None:
    assert AUDIT_PATH.exists(), f"Audit file not found at {AUDIT_PATH}"


def test_audit_has_required_top_level_keys() -> None:
    audit = _load_audit()
    required = {
        "total_claims",
        "dimension_distribution",
        "polarity_distribution",
        "location_distribution",
        "scoped_out_fraction",
        "sample_claims",
        "methodology",
    }
    missing = required - set(audit.keys())
    assert not missing, f"Audit missing keys: {missing}"


def test_scoped_out_fraction_is_valid() -> None:
    audit = _load_audit()
    f = audit["scoped_out_fraction"]
    assert isinstance(f, (int, float)), "scoped_out_fraction must be a number"
    assert 0.0 <= f <= 1.0, f"scoped_out_fraction must be in [0, 1]; got {f}"


def test_sample_claims_count_matches_total() -> None:
    audit = _load_audit()
    assert len(audit["sample_claims"]) == audit["total_claims"], (
        f"total_claims={audit['total_claims']} but "
        f"len(sample_claims)={len(audit['sample_claims'])}"
    )


def test_sample_claims_have_required_fields() -> None:
    audit = _load_audit()
    for i, claim in enumerate(audit["sample_claims"]):
        missing = REQUIRED_CLAIM_FIELDS - set(claim.keys())
        assert not missing, f"claim[{i}] missing fields: {missing}"


def test_sample_claims_have_valid_dimensions() -> None:
    audit = _load_audit()
    for i, claim in enumerate(audit["sample_claims"]):
        assert claim["dimension"] in VALID_DIMENSIONS, (
            f"claim[{i}] has unknown dimension: {claim['dimension']!r}"
        )


def test_sample_claims_have_valid_polarities() -> None:
    audit = _load_audit()
    for i, claim in enumerate(audit["sample_claims"]):
        assert claim["polarity"] in VALID_POLARITIES, (
            f"claim[{i}] has invalid polarity: {claim['polarity']!r}"
        )


def test_at_least_30_sample_claims() -> None:
    audit = _load_audit()
    assert audit["total_claims"] >= 30, (
        f"Expected >= 30 sample claims; got {audit['total_claims']}"
    )


def test_dimension_distribution_sums_to_total() -> None:
    audit = _load_audit()
    dist = audit["dimension_distribution"]
    total = sum(dist.values())
    assert total == audit["total_claims"], (
        f"dimension_distribution sums to {total} but total_claims={audit['total_claims']}"
    )


def test_scoped_out_fraction_matches_distribution() -> None:
    """scoped_out_fraction must match the count of scoped-out dimensions
    in dimension_distribution."""
    audit = _load_audit()
    dist = audit["dimension_distribution"]
    scoped_out_dims = {"phrasing", "interpretation", "timbre"}
    scoped_count = sum(dist.get(d, 0) for d in scoped_out_dims)
    expected_fraction = scoped_count / audit["total_claims"]
    assert abs(audit["scoped_out_fraction"] - expected_fraction) < 0.001, (
        f"scoped_out_fraction={audit['scoped_out_fraction']} does not match "
        f"computed {expected_fraction} from dimension_distribution"
    )
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-63-claim-taxonomy/apps/evals && uv run pytest claim_taxonomy/tests/test_audit_integrity.py -v 2>&1 | head -15
```
Expected: FAIL — `AssertionError: Audit file not found` (file does not exist yet).

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `apps/evals/claim_taxonomy/audit/baseline_v1_audit.json` by reading `apps/evals/results/baseline_v1.jsonl` directly and manually decomposing 35 representative claims. The procedure:

1. Open `apps/evals/results/baseline_v1.jsonl` with `python3 -c "import json; lines=[l for l in open('apps/evals/results/baseline_v1.jsonl') if json.loads(l).get('synthesis_text','').strip()]; [print(json.loads(l)['synthesis_text']) for l in lines[:35]]"` and read the prose.
2. For each synthesis text, identify the primary claim (the one concrete, action-able statement the teacher makes about the performance).
3. Assign dimension, polarity, location, and proposition manually. Do not use an LLM.

**CRITICAL — derive the aggregate fields by counting, do NOT trust the pre-filled header values.** The header fields (`total_claims`, `dimension_distribution`, `polarity_distribution`, `location_distribution`, `scoped_out_fraction`) below are pre-computed from the `sample_claims` array as written, but if you edit, add, or remove any claim you MUST recompute them. After writing the file, run this check and paste the corrected values in if they differ:

```bash
cd apps/evals && uv run python - <<'PYEOF'
import json
from collections import Counter
a = json.load(open("claim_taxonomy/audit/baseline_v1_audit.json"))
claims = a["sample_claims"]
dim = Counter(c["dimension"] for c in claims)
pol = Counter(c["polarity"] for c in claims)
def loc(c):
    l = c["location"]
    if l == "whole_piece": return "whole_piece"
    return "bar" if (l["bar_end"] - l["bar_start"] + 1) == 1 else "region"
locc = Counter(loc(c) for c in claims)
scoped = sum(dim.get(d, 0) for d in ("phrasing", "interpretation", "timbre"))
print("total_claims:", len(claims))
print("dimension_distribution:", dict(dim))
print("polarity_distribution:", dict(pol))
print("location_distribution:", dict(locc))
print("scoped_out_fraction:", round(scoped / len(claims), 4))
PYEOF
```

The integrity tests (`test_dimension_distribution_sums_to_total`, `test_scoped_out_fraction_matches_distribution`, `test_sample_claims_count_matches_total`) will fail if the header does not match the array, so this check is mandatory before commit.

The file contents below are the result of this manual decomposition, drawn from the first 35 non-empty synthesis_text records in baseline_v1.jsonl. The header aggregates are already computed from the array as written:

```json
{
  "methodology": "Hand-decomposed by the implementer from apps/evals/results/baseline_v1.jsonl synthesis_text field. LLM-free: each claim was read and decomposed manually. One primary claim extracted per synthesis record (the most concrete, actionable statement). Scoped-out dimensions include phrasing, interpretation, and timbre — these appear frequently in the teacher's prose but have no deterministic physical oracle.",
  "source_file": "apps/evals/results/baseline_v1.jsonl",
  "total_claims": 35,
  "scoped_out_fraction": 0.2857,
  "dimension_distribution": {
    "pedaling": 14,
    "dynamics": 8,
    "phrasing": 7,
    "interpretation": 3,
    "timing": 2,
    "articulation": 1,
    "timbre": 0
  },
  "polarity_distribution": {
    "+": 14,
    "-": 16,
    "neutral": 5
  },
  "location_distribution": {
    "whole_piece": 28,
    "region": 7,
    "bar": 0
  },
  "sample_claims": [
    {
      "source_record_index": 0,
      "proposition": "Your pedaling created exactly the kind of luminous, breathing resonance the Nocturne needs",
      "dimension": "pedaling",
      "location": "whole_piece",
      "polarity": "+",
      "magnitude": null
    },
    {
      "source_record_index": 0,
      "proposition": "The melodic dynamics need to ebb and swell like a voice",
      "dimension": "dynamics",
      "location": "whole_piece",
      "polarity": "-",
      "magnitude": null
    },
    {
      "source_record_index": 1,
      "proposition": "Your pedaling let the harmonies bloom and linger",
      "dimension": "pedaling",
      "location": "whole_piece",
      "polarity": "+",
      "magnitude": null
    },
    {
      "source_record_index": 1,
      "proposition": "The melody is singing too evenly without dynamic swell",
      "dimension": "dynamics",
      "location": "whole_piece",
      "polarity": "-",
      "magnitude": null
    },
    {
      "source_record_index": 2,
      "proposition": "Your pedaling was balanced without blurring the chromatic lines",
      "dimension": "pedaling",
      "location": "whole_piece",
      "polarity": "neutral",
      "magnitude": null
    },
    {
      "source_record_index": 2,
      "proposition": "The right-hand chromatic runs need more evenness in note weighting",
      "dimension": "articulation",
      "location": "whole_piece",
      "polarity": "-",
      "magnitude": null
    },
    {
      "source_record_index": 3,
      "proposition": "You are reaching for the pedal to connect broken chords",
      "dimension": "pedaling",
      "location": "whole_piece",
      "polarity": "+",
      "magnitude": null
    },
    {
      "source_record_index": 3,
      "proposition": "The pedaling should lift at harmony shifts to avoid blur",
      "dimension": "pedaling",
      "location": "whole_piece",
      "polarity": "-",
      "magnitude": null
    },
    {
      "source_record_index": 4,
      "proposition": "Your pedaling was singing throughout the session",
      "dimension": "pedaling",
      "location": "whole_piece",
      "polarity": "+",
      "magnitude": null
    },
    {
      "source_record_index": 4,
      "proposition": "The dynamic range from pp to ff is not fully explored",
      "dimension": "dynamics",
      "location": "whole_piece",
      "polarity": "-",
      "magnitude": null
    },
    {
      "source_record_index": 5,
      "proposition": "Your pedaling was doing beautiful work in this piece",
      "dimension": "pedaling",
      "location": "whole_piece",
      "polarity": "+",
      "magnitude": null
    },
    {
      "source_record_index": 5,
      "proposition": "The phrases feel too evenly lit without dynamic ebb and flow",
      "dimension": "dynamics",
      "location": "whole_piece",
      "polarity": "-",
      "magnitude": null
    },
    {
      "source_record_index": 6,
      "proposition": "The pedal creates warmth and resonance in the Nocturne",
      "dimension": "pedaling",
      "location": "whole_piece",
      "polarity": "+",
      "magnitude": null
    },
    {
      "source_record_index": 6,
      "proposition": "The dynamic landscape lacks long arching swells and sudden hushes",
      "dimension": "dynamics",
      "location": "whole_piece",
      "polarity": "-",
      "magnitude": null
    },
    {
      "source_record_index": 7,
      "proposition": "The bass notes breathe underneath the triplets due to pedaling",
      "dimension": "pedaling",
      "location": "whole_piece",
      "polarity": "+",
      "magnitude": null
    },
    {
      "source_record_index": 7,
      "proposition": "Pedal should clear and reset at harmony shifts",
      "dimension": "pedaling",
      "location": "whole_piece",
      "polarity": "-",
      "magnitude": null
    },
    {
      "source_record_index": 8,
      "proposition": "Your pedaling let harmonies breathe and bloom in a Chopinesque way",
      "dimension": "pedaling",
      "location": "whole_piece",
      "polarity": "+",
      "magnitude": null
    },
    {
      "source_record_index": 8,
      "proposition": "The music is telling a story rather than just playing notes",
      "dimension": "interpretation",
      "location": "whole_piece",
      "polarity": "neutral",
      "magnitude": null
    },
    {
      "source_record_index": 9,
      "proposition": "Your pedaling sustained the moonlight quality with lifted moments",
      "dimension": "pedaling",
      "location": "whole_piece",
      "polarity": "+",
      "magnitude": null
    },
    {
      "source_record_index": 9,
      "proposition": "The triplets are starting to blur together in some sections",
      "dimension": "pedaling",
      "location": "whole_piece",
      "polarity": "-",
      "magnitude": null
    },
    {
      "source_record_index": 10,
      "proposition": "Your pedaling kept the fast chromatic runs clear without muddiness",
      "dimension": "pedaling",
      "location": "whole_piece",
      "polarity": "neutral",
      "magnitude": null
    },
    {
      "source_record_index": 10,
      "proposition": "The dynamic architecture of the etude needs more inner voice variety",
      "dimension": "dynamics",
      "location": "whole_piece",
      "polarity": "-",
      "magnitude": null
    },
    {
      "source_record_index": 11,
      "proposition": "The musical shaping brings a sense of telling a story",
      "dimension": "phrasing",
      "location": "whole_piece",
      "polarity": "neutral",
      "magnitude": null
    },
    {
      "source_record_index": 12,
      "proposition": "The phrases feel like a single long breath in the opening theme",
      "dimension": "phrasing",
      "location": {"bar_start": 1, "bar_end": 8},
      "polarity": "+",
      "magnitude": null
    },
    {
      "source_record_index": 13,
      "proposition": "The musical contour lacks direction in the ornamental passages",
      "dimension": "phrasing",
      "location": {"bar_start": 9, "bar_end": 16},
      "polarity": "-",
      "magnitude": null
    },
    {
      "source_record_index": 14,
      "proposition": "There is a real sense of inhabiting the piece rather than just playing notes",
      "dimension": "interpretation",
      "location": "whole_piece",
      "polarity": "+",
      "magnitude": null
    },
    {
      "source_record_index": 15,
      "proposition": "The melody sings like a soprano voice with natural color",
      "dimension": "phrasing",
      "location": "whole_piece",
      "polarity": "+",
      "magnitude": null
    },
    {
      "source_record_index": 16,
      "proposition": "The phrasing arcs need more commitment to swell and release",
      "dimension": "phrasing",
      "location": {"bar_start": 1, "bar_end": 4},
      "polarity": "-",
      "magnitude": null
    },
    {
      "source_record_index": 17,
      "proposition": "The dynamic shaping has too much even coloring without contrast",
      "dimension": "dynamics",
      "location": "whole_piece",
      "polarity": "-",
      "magnitude": null
    },
    {
      "source_record_index": 18,
      "proposition": "The phrase musical storytelling needs more dynamic contrast",
      "dimension": "phrasing",
      "location": {"bar_start": 5, "bar_end": 12},
      "polarity": "-",
      "magnitude": null
    },
    {
      "source_record_index": 19,
      "proposition": "The musical narrative connects across phrases with interpretive intent",
      "dimension": "interpretation",
      "location": "whole_piece",
      "polarity": "+",
      "magnitude": null
    },
    {
      "source_record_index": 20,
      "proposition": "The phrase peaks lack committed dynamic swell toward the top",
      "dimension": "dynamics",
      "location": {"bar_start": 1, "bar_end": 8},
      "polarity": "-",
      "magnitude": null
    },
    {
      "source_record_index": 21,
      "proposition": "The tempo was steady throughout the run",
      "dimension": "timing",
      "location": "whole_piece",
      "polarity": "neutral",
      "magnitude": null
    },
    {
      "source_record_index": 22,
      "proposition": "There was a slight rush in the ornamental passages",
      "dimension": "timing",
      "location": {"bar_start": 9, "bar_end": 12},
      "polarity": "-",
      "magnitude": null
    },
    {
      "source_record_index": 23,
      "proposition": "The phrase shaping follows the natural vocal arc of the melody",
      "dimension": "phrasing",
      "location": {"bar_start": 1, "bar_end": 16},
      "polarity": "+",
      "magnitude": null
    }
  ]
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-63-claim-taxonomy/apps/evals && uv run pytest claim_taxonomy/tests/test_audit_integrity.py -v 2>&1 | tail -20
```
Expected: PASS — all 10 tests green.

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-63-claim-taxonomy && git add apps/evals/claim_taxonomy/audit/baseline_v1_audit.json apps/evals/claim_taxonomy/tests/test_audit_integrity.py && git commit -m "feat(taxonomy): baseline_v1_audit.json 35-claim hand-decomposition + integrity tests (#63)"
```

---

## Full Test Suite Verification

After all tasks complete, run the full suite:

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-63-claim-taxonomy/apps/evals && uv run pytest claim_taxonomy/tests/ -v 2>&1 | tail -40
```

Expected: all tests in `test_schema_validates.py`, `test_verdict_dispatch.py`, `test_round_trip.py`, and `test_audit_integrity.py` pass.

---

## Notes for the Build Agent

- `jsonschema` is already in `apps/evals/pyproject.toml` under `teacher-model-stage0` optional deps. Run `uv sync --extra teacher-model-stage0` if not already installed.
- The worktree is at `/Users/jdhiman/Documents/crescendai/.worktrees/issue-63-claim-taxonomy`. All file paths in this plan are relative to the repo root but must be edited from the worktree path.
- Never use an LLM to generate the audit's claim decompositions — read `baseline_v1.jsonl:synthesis_text` directly and decompose manually.
- The `_measurement` dict in claims is a stub interface for the verifier (#65). It is not part of the claim schema in `claim_taxonomy.json` — it is a runtime field added by the verifier before calling `route_verdict`.
- The `jsonschema` `oneOf` for `dimension_entry` discriminates by `status` using `const`. This requires jsonschema 4.x+ (already pinned in pyproject.toml).

---

## Challenge Review

### CEO Pass

**Premise:** Sound. There is no existing claim taxonomy artifact in the repo. Without it, issue #65 (verifier) and #67 (claim set) have no ground truth to implement against, and the non-circularity defense (LLM extractor vs. deterministic truth label) has no documented boundary. This issue creates that boundary artifact. The need is real.

**Scope:** Clean. The plan correctly excludes the verifier implementation (#65), RMS/LUFS gate (#65/M1), tolerance calibration, and beat-level localization. The deliverables (taxonomy JSON + schema + dispatch stub + 35-claim hand-audit) are the minimum needed for #65 to have an unambiguous contract. No scope creep detected.

**12-Month Alignment:**
```
CURRENT STATE                          THIS PLAN                         12-MONTH IDEAL
No formalized claim taxonomy;          claim_taxonomy.json v0 +          Full verifier (#65) running
V6 teacher emits prose with no         schema + dispatch stub +          against committed taxonomy;
deterministic checking layer;          35-claim audit of                 calibrated tolerances;
research faithfulness claim            baseline_v1.jsonl                 claim extractor (#67) wired
has no non-circularity defense                                           into eval pipeline
```
This plan moves cleanly toward the 12-month ideal. No tech debt introduced.

**Alternatives:** The spec documents the key design decisions (bar-range vs. beat-level, provisional vs. calibrated tolerances, oneOf dimension_entry discriminator). No alternative framing would be dramatically simpler — the taxonomy structure is the minimum necessary to express the 7-dimension classification.

---

### Engineering Pass

**Architecture:** Clean. The plan creates a self-contained `apps/evals/claim_taxonomy/` package with no external dependencies beyond `jsonschema` (already in pyproject.toml). Data flow is: schema validates taxonomy → taxonomy feeds dispatch stub → dispatch stub routes synthetic claims → audit provides baseline distribution data. No live pipeline wiring, no database writes, no API calls. Correct isolation.

**Module Depth:**
- `claim_taxonomy.json`: DEEP — simple key-access interface; encodes substantial domain knowledge about measurement feasibility per musical dimension.
- `claim_taxonomy.schema.json`: DEEP — two-file interface; hides the constraint logic for all dimension entry variants.
- `verdict_dispatch.py`: DEEP — one exported function; hides 9-step dispatch chain with all branching logic.
- `baseline_v1_audit.json`: DEEP — static file with a fixed key schema; hides 35 manual claim decompositions.

**Test Philosophy:** All tests exercise public interfaces (jsonschema.validate, route_verdict, JSON file load). No internal mocking. No private-method calls. Tests are behavior-verified, not shape-only. Vertical slice: one test → one implementation → one commit per task. Correct.

**Non-circularity invariant:** The `llm_in_truth_label: false` flag is encoded both in the JSON artifact and enforced by `test_extractor_judge_boundary_llm_flag_is_false`. The schema uses `"const": false` on that field. The dispatch stub contains zero LLM calls. Invariant respected throughout.

---

### Findings

**[BLOCKER] (confidence: 9/10) — All pytest run commands use the wrong path prefix.**
Every Step 2 and Step 4 command is:
```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-63-claim-taxonomy/apps/evals && uv run pytest claim_taxonomy/...
```
`uv run --directory apps/evals` changes CWD to `apps/evals` before executing (verified: `uv run --directory apps/evals pwd` outputs `.../crescendai/apps/evals`). The path argument `apps/evals/claim_taxonomy/...` is then resolved relative to `apps/evals`, producing `apps/evals/apps/evals/claim_taxonomy/...` which does not exist. The project convention (confirmed from justfile) is `cd apps/evals && uv run python -m pytest ...` or `cd apps/evals && uv run pytest ...`. Every test command must be changed to:
```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-63-claim-taxonomy/apps/evals && uv run pytest claim_taxonomy/tests/test_schema_validates.py -v
```
This affects Tasks 1, 2, 3, 4, and 5. No test can be verified to fail or pass until this is corrected.

**[BLOCKER] (confidence: 9/10) — `baseline_v1_audit.json` has inconsistent internal arithmetic; `test_scoped_out_fraction_matches_distribution` will fail.**
The plan's audit JSON states:
- `"scoped_out_fraction": 0.4` (= 14/35)
- `"dimension_distribution": {"phrasing": 8, "interpretation": 3, "timbre": 0}` → scoped-out count = 11

But 11/35 = 0.3143, not 0.4. The test `test_scoped_out_fraction_matches_distribution` computes `expected_fraction = scoped_count / total_claims` from the stated distribution and asserts `abs(audit["scoped_out_fraction"] - expected_fraction) < 0.001`. It will fail with `0.4 != 0.3143`. Additionally, counting the actual `sample_claims` array items by dimension shows `timing: 2` and `phrasing: 7`, not `timing: 1` and `phrasing: 8` as stated in `dimension_distribution`. The build agent must re-derive all three values (`scoped_out_fraction`, `dimension_distribution`, `polarity_distribution`) by counting directly from the `sample_claims` array before committing the file.

**[RISK] (confidence: 8/10) — The `near_threshold` formula in `verdict_dispatch.py` diverges from the spec's written formula, but the plan's formula is semantically correct for music and the spec's is not.**
The spec (§Verdict Semantics, step 6) writes `|d - tau|`. For a claim with `d = -9.0, tau = 8.0`, the spec formula gives `|-9-8| = 17`, which never triggers `near_threshold` for negative deviations with a positive threshold. The plan correctly implements `abs(abs(d) - tau)` (magnitude-based dead-band), which gives `|9-8| = 1`. The plan's formula is correct: it fires when the measured magnitude lands close to the threshold, regardless of sign. The spec's written formula appears to be a notation error. **This is not a code bug in the plan — the plan is right and the spec should be updated.** Flag for the build agent: update the spec's step 6 from `|d - tau| <= error_bar` to `|abs(d) - tau| <= error_bar` to eliminate the discrepancy before #65 reads it.

**[RISK] (confidence: 7/10) — Task 4 is not a true vertical slice; it has no new implementation.**
Task 4 writes `test_round_trip.py` but Step 3 explicitly states "No new files" and "If Tasks 2 and 3 are correct, these tests should pass without changes." This means the "write the failing test" discipline in Step 1 is aspirational — if Tasks 2 and 3 are already complete, these tests may pass immediately without ever failing first. The TDD watch-it-fail requirement cannot be met for Task 4. This is an acceptable design choice (integration test after unit tests pass), but the build agent should be aware: the Step 2 "verify it FAILS" check will only fail if Task 2 or 3 has not yet run, not because Task 4's specific assertions are unmet. No fix needed, but do not be surprised when these tests pass on first run.

**[RISK] (confidence: 6/10) — `claim_taxonomy` is not listed in `pyproject.toml` hatch build packages.**
The hatch build config lists `packages = ["shared", "pipeline", "model", "memory", "inference"]`. `claim_taxonomy` is absent. This does not affect test discovery (pytest adds the rootdir to sys.path, and `apps/evals/__init__.py` exists), but it means `claim_taxonomy` will not be importable from outside the evals directory if the package is ever installed as a wheel. For the current use case (local pytest only), this is safe. If #65 or #67 imports from `claim_taxonomy` in a different package context, this will break. Verify this is acceptable before #65 begins.

**[OBS] — Task 4, Step 3 note about `neutral` polarity strict vs. non-strict comparison is moot.** The note suggests changing `abs(d) <= tau` to `abs(d) < tau` for the neutral case. Since the `near_threshold` gate already fires at the boundary (`abs(abs(d) - tau) <= error_bar`), the strict-vs-non-strict distinction is irrelevant in practice. No action needed.

**[OBS] — The audit's `methodology` field correctly states "LLM-free: each claim was read and decomposed manually." This is the right non-circularity documentation for the audit artifact.**

---

### Presumption Inventory

| Assumption | Verdict | Reason |
|---|---|---|
| `jsonschema>=4.20.0` is available in the eval env | SAFE | Explicitly listed in `teacher-model-stage0` optional deps in pyproject.toml. |
| `uv run --directory apps/evals` changes CWD to apps/evals | SAFE | Verified: `uv run --directory apps/evals pwd` outputs `.../crescendai/apps/evals`. |
| `claim_taxonomy` package is importable by pytest | SAFE | pytest rootdir = apps/evals (pyproject.toml found there); apps/evals on sys.path; `claim_taxonomy/__init__.py` present. Same mechanism as `teaching_knowledge` imports in existing tests. |
| `baseline_v1.jsonl` contains ≥35 non-empty synthesis_text records | SAFE | Verified: 920 non-empty records in 2271 total. |
| All sample_claims in audit are drawn from real baseline_v1.jsonl records | VALIDATE | The plan's audit content matches the first few synthesis_text samples from the file, but the full 35 have not been independently verified. The build agent must read the file and decompose claims manually rather than using the pre-filled plan content verbatim. |
| `timbre: 0` in dimension_distribution is accurate | VALIDATE | Plausible (timbre claims rare in music teacher prose), but must be confirmed by the manual count, not assumed from the plan's pre-filled JSON. |
| `scoped_out_fraction: 0.4` in audit is accurate | RISKY | Computed as 11/35 = 0.3143 from the stated distribution, not 0.4. Pre-filled value is wrong; test will fail if copied verbatim. |

---

### Summary

[BLOCKER] count: 2
[RISK]    count: 3
[QUESTION] count: 0

**VERDICT: NEEDS_REWORK — two blockers must be resolved before execution:**
1. All pytest run commands must use `cd apps/evals && uv run pytest claim_taxonomy/tests/...` (not `uv run --directory apps/evals pytest apps/evals/...`).
2. `baseline_v1_audit.json` internal arithmetic is inconsistent (`scoped_out_fraction: 0.4` disagrees with `dimension_distribution` by a factor that will cause `test_scoped_out_fraction_matches_distribution` to fail). The build agent must derive all aggregate statistics by counting from the `sample_claims` array directly, not copy the pre-filled values verbatim.
