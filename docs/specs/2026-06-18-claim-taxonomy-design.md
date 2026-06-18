# Claim Taxonomy Design

**Goal:** Define a versioned, schema-validated taxonomy of `(dimension, location)` music-feedback claims so that a deterministic, non-LLM verifier (issue #65) has a complete, unambiguous contract to implement against.

**Not in scope:**
- The verifier implementation itself (issue #65)
- The RMS/LUFS dynamics estimator or AMT offset-accuracy gate validation (issue #65/M1)
- Tolerance calibration to measured error bars (issue #65/M1)
- The LLM claim extractor at scale (issue #67)
- Any live-pipeline wiring changes (issue #64)
- Beat-level claim localization (no metric grid; score_beat is a 0.0 stub)

---

## Problem

The V6 teacher generates prose feedback claims like "you rushed in bars 12–16" or "your pedaling was sparse in the opening." There is currently no formalized, versioned artifact that says:
- which musical dimensions are checkable deterministically (and which are not)
- what the closed-form check and reference frame are for each admissible dimension
- what schema a structured claim must conform to
- what verdict logic routes a claim to SUPPORTED / REFUTED / UNVERIFIABLE
- where the extractor–judge boundary sits (non-circularity defense)

Without this artifact, issue #65 (verifier) and issue #67 (claim set) have no ground truth to implement against, and the research contribution (faithfulness rate without an LLM judge) has no documented non-circularity argument.

---

## Solution (from the user's perspective)

After this issue:
1. `apps/evals/claim_taxonomy/claim_taxonomy.json` exists, is versioned (`taxonomy_version: "v0"`), and passes JSON Schema validation (`apps/evals/claim_taxonomy/claim_taxonomy.schema.json`).
2. Every musical dimension has an explicit `status` field (`active | gated_on_measurement | scoped_out`); every active dimension has a complete `{reference, check, tolerance, reliability_tier}` block; every tolerance is `{provisional: <number>, locked: false, calibration_source: "#65/M1 error-bar study"}`.
3. The verdict dispatch logic (SUPPORTED / REFUTED / UNVERIFIABLE with typed reason codes) is encoded in the taxonomy artifact and tested via a round-trip stub test — no real measurement, control-flow dispatch only.
4. A hand-decomposed audit of ~30–50 claims drawn from `apps/evals/results/baseline_v1.jsonl` exists as `apps/evals/claim_taxonomy/audit/baseline_v1_audit.json`, with a tally of dimension/polarity/location distribution and a reported scoped-out fraction.

---

## Design

### Extractor–Judge Boundary (Non-Circularity Defense)

An LLM **may** be used as the claim extractor (prose → structured `{dimension, location, polarity, magnitude}`). The verifier's **truth label** (SUPPORTED / REFUTED / UNVERIFIABLE) must **never** invoke an LLM. This boundary is encoded as a top-level `extractor_judge_boundary` field in `claim_taxonomy.json` (prose explanation + machine-readable `llm_in_truth_label: false` flag).

This is the PROVE/ViCrit non-circularity defense: the checker's correctness is grounded in a programmatic oracle over a measured signal, not in any model's opinion.

### Claim Formalism

```
Claim = {
  proposition: string,           // free text, source of the claim
  dimension: enum,               // from the admissible dimension set
  location: {bar_start: int, bar_end: int} | "whole_piece",
  polarity: "+" | "-" | "neutral",
  magnitude: {value: number, unit: string} | null
}
```

- `polarity: "+"` = signed anomaly in the positive direction (e.g., too loud)
- `polarity: "-"` = signed anomaly in the negative direction (e.g., too quiet, rushed)
- `polarity: "neutral"` = asserts a virtue or absence-of-problem (e.g., "pedaling was clean")
- `magnitude` degrades gracefully: when present, it tightens the check (the threshold is compared to `magnitude.value`); when null, the verifier falls back to sign+tolerance check
- Thresholds live in the dimension registry (`claim_taxonomy.json:dimensions[dim].tolerance`), NOT in the claim

### Location Granularity

Three reliability tiers:
- `whole_piece` — always localizable; tier 1
- `region` — `bar_end - bar_start >= N` (N=4 by convention); tier 1
- `bar` — `bar_start == bar_end`; fragile; tier 2

Beat granularity is NOT admissible. `score_beat` is a 0.0 stub in `bar_analysis.rs:643`; no metric grid exists.

**Span-vs-uncertainty fail-safe:** the verifier must downgrade a claim to `UNVERIFIABLE(unlocalizable)` when the measured alignment uncertainty exceeds the claim's location span. This rule is encoded in the taxonomy's verdict spec, not deferred to the verifier's discretion.

### Admissible Dimensions

| Dimension | Status | Rationale |
|---|---|---|
| `timing` | `active` | AMT onsets → region_tempo_fit → signed_tempo_deviation; reference: established_tempo |
| `pedaling` | `active` | AMT sustain pedal events → pedal_presence_density (on/off + density only; half-pedal/flutter out of scope); reference: self_density |
| `dynamics` | `gated_on_measurement` | Gate: `rms_lufs_estimator` (does not yet exist in repo; see issue #65/M1). Relative within-region loudness only; absolute dB inadmissible (recording gain uncontrolled). Reference: within_region_range; check: dynamic_range_zscore + contour_slope_sign |
| `articulation` | `gated_on_measurement` | Gate: `offset_accuracy_validation` (AMT offsets weaker than onsets; must validate before use). Reference: self_overlap_ratio; check: legato_staccato_overlap_ratio; region-level only |
| `phrasing` | `scoped_out` | Perceptual; the only proxy is itself a perceptual model |
| `interpretation` | `scoped_out` | Definitionally subjective |
| `timbre` | `scoped_out` | No physical proxy |

### Verdict Semantics

Three labels: `SUPPORTED | REFUTED | UNVERIFIABLE`

**UNVERIFIABLE typed reason codes (explicit, never a silent bucket):**

| Code | Condition |
|---|---|
| `out_of_scope_dim` | `dimension.status == "scoped_out"` |
| `gated_dim` | `dimension.status == "gated_on_measurement"` and gate not validated |
| `unlocalizable` | no score in catalog / piece not identified / span < alignment_uncertainty |
| `substrate_failure` | AMT or alignment substrate raised an error |
| `region_too_short` | measured region has fewer events than minimum for a reliable estimate |
| `near_threshold` | `abs(abs(d) - tau) <= error_bar` — the deviation *magnitude* lands in the dead-band around `tau` (NOT around 0) |

**Verdict dispatch:**
1. If `dimension.status != "active"` → `UNVERIFIABLE(out_of_scope_dim | gated_dim)`
2. If location is not resolvable → `UNVERIFIABLE(unlocalizable)`
3. If substrate failed → `UNVERIFIABLE(substrate_failure)`
4. If region has fewer than minimum events → `UNVERIFIABLE(region_too_short)`
5. Compute signed deviation `d` vs reference
6. If `abs(abs(d) - tau) <= error_bar` → `UNVERIFIABLE(near_threshold)` (compare the deviation magnitude `abs(d)` to the threshold `tau`; the dead-band is centered on `tau`, not on 0)
7. If `d` confirms `polarity` direction beyond `tau` (`abs(d) > tau` and `sign(d)` matches polarity; for `neutral` polarity, `abs(d) < tau` with no anomaly) → `SUPPORTED`
8. Otherwise → `REFUTED`

**Headline metric:**

```
faithfulness = SUPPORTED / (SUPPORTED + REFUTED)
coverage     = (SUPPORTED + REFUTED) / total_claims
```

`UNVERIFIABLE` claims are reported separately as a histogram by typed reason code. They are **never** folded into faithfulness or coverage.

### Tolerance Structure

Each active dimension's tolerance ships provisional:

```json
{
  "name": "signed_tempo_deviation",
  "provisional": 8.0,
  "unit": "percent",
  "calibration_source": "#65/M1 error-bar study",
  "locked": false
}
```

`locked: false` means the value is a defensible prior, not a calibrated number. Final calibration is issue #65/M1 and is out of scope here.

Timing tolerance is expressed as **percent deviation from established tempo**, not absolute onset-ms. This is robust to alignment jitter.

---

## Modules

### `claim_taxonomy.json` + `claim_taxonomy.schema.json`

**Interface:** two static JSON files. `claim_taxonomy.json` is the versioned taxonomy artifact (claim schema + dimension registry + verdict spec + extractor-judge boundary statement). `claim_taxonomy.schema.json` is the JSON Schema (draft 2020-12) that validates it.

**Hides:** the decision machinery of which dimensions are admissible, why, and what the closed-form checks are. Downstream consumers (verifier #65, claim extractor #67) read the registry; they do not re-derive these decisions.

**Tested through:** `jsonschema.validate(claim_taxonomy, schema)` in Python; hand-authored example claims per active dimension parsed against the claim schema; round-trip verdict dispatch stub.

**Depth verdict:** DEEP — the interface is two files with a fixed top-level structure; the registry encodes substantial domain knowledge about measurement feasibility per dimension.

### `apps/evals/claim_taxonomy/verdict_dispatch.py`

**Interface:**
```python
def route_verdict(claim: dict, registry: dict) -> tuple[str, str | None]:
    """
    Returns ("SUPPORTED"|"REFUTED"|"UNVERIFIABLE", reason_code | None).
    Control-flow dispatch only — does NOT perform any measurement.
    Raises TypeError for unknown dimension or missing required fields.
    """
```

**Hides:** the verdict dispatch logic (the 8-step chain from the verdict spec above). The verifier (#65) will replace this stub with real measurement; this module only tests that the routing logic is correct independent of measurement.

**Tested through:** the public `route_verdict` function with hand-authored claims and a mock registry that supplies synthetic deviation/error_bar values. Never mocked internally.

**Depth verdict:** DEEP — the caller passes a claim + registry; the module encodes all dispatch logic; the verifier can swap in real measurements without touching this module.

### `apps/evals/claim_taxonomy/audit/baseline_v1_audit.json`

**Interface:** a static JSON file with keys: `total_claims`, `dimension_distribution`, `polarity_distribution`, `location_distribution`, `scoped_out_fraction`, `sample_claims` (array of ~30–50 hand-decomposed claim dicts).

**Hides:** the manual decomposition work (reading prose, identifying dimension + location + polarity).

**Tested through:** a pytest that loads the file and asserts `scoped_out_fraction` is present, that all `sample_claims` entries have the required schema fields, and that `total_claims == len(sample_claims)`.

**Depth verdict:** DEEP — the file is a research artifact; the test verifies structural integrity, not the content decisions.

---

## Verification Architecture

**Canonical success state:** `jsonschema.validate(claim_taxonomy_json, schema_json)` passes with no exceptions; `pytest apps/evals/claim_taxonomy/tests/` returns all green; the audit file reports a scoped-out fraction with a value between 0 and 1.

**Automated check:**
```bash
cd apps/evals && uv run pytest claim_taxonomy/tests/ -v
```

**Harness:** Task 1 is a verification harness (the JSON Schema + a schema-validation test) that must pass before any implementation begins. This is Task Group 0 in the plan.

---

## File Changes

| File | Change | Type |
|------|--------|------|
| `apps/evals/claim_taxonomy/__init__.py` | Empty package init | New |
| `apps/evals/claim_taxonomy/claim_taxonomy.json` | Versioned taxonomy artifact | New |
| `apps/evals/claim_taxonomy/claim_taxonomy.schema.json` | JSON Schema (draft 2020-12) validating the taxonomy | New |
| `apps/evals/claim_taxonomy/verdict_dispatch.py` | Verdict routing stub (control-flow only, no measurement) | New |
| `apps/evals/claim_taxonomy/audit/__init__.py` | Empty package init | New |
| `apps/evals/claim_taxonomy/audit/baseline_v1_audit.json` | Hand-decomposed claim audit (~30–50 claims from baseline_v1.jsonl) | New |
| `apps/evals/claim_taxonomy/tests/__init__.py` | Empty test package init | New |
| `apps/evals/claim_taxonomy/tests/test_schema_validates.py` | Schema self-validation test (Task 1) | New |
| `apps/evals/claim_taxonomy/tests/test_verdict_dispatch.py` | Round-trip verdict dispatch stub tests (Tasks 2–4) | New |
| `apps/evals/claim_taxonomy/tests/test_audit_integrity.py` | Audit file structural integrity test (Task 5) | New |

---

## Open Questions

- Q: Should `magnitude.unit` be an enum or a free string?  
  Default: free string for v0; an enum would require knowing all units in advance and would block extension. The schema enforces `type: string` only.

- Q: What is the minimum event count (`region_too_short` threshold) for timing vs. pedaling?  
  Default: timing = 8 onsets in region; pedaling = 2 pedal events. Both are provisional, calibration_source "#65/M1".
