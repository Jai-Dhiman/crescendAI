# Implementation Notes — Claim Taxonomy (#63)

Decisions, deviations, and tradeoffs made during build. Read this before running /review.

Baseline state: 431 tests collected, 5 pre-existing collection errors (missing optional deps unrelated to claim_taxonomy). jsonschema 4.26.0 installed via `uv sync --extra teacher-model-stage0`.

## Task 1: JSON Schema + Package Scaffold
- Schema draft 2020-12, dimension_entry oneOf discriminated on status const.
- Code review (P-IMPORTANT) fixes applied in commit 56518838: (1) added negative test test_llm_in_truth_label_true_fails_schema so the non-circularity const:false is load-bearing; (2) added additionalProperties:false to active/gated/scoped_out dimension defs. active_dimension gained optional "notes":string to preserve the Task 2 artifact's notes field.
- Commits: 716ad936 (initial), 56518838 (review fixes). 6 tests green.

## Task 2: claim_taxonomy.json v0
- All 7 dimensions: timing+pedaling active, dynamics+articulation gated_on_measurement, phrasing+interpretation+timbre scoped_out.
- All active tolerances locked:false, calibration_source="#65/M1 error-bar study" (provisional; final values are #65/M1 work, out of scope).
- CAUTION APPLIED: dispatch_order step 7 reads "abs(abs(d) - tau)" (double-abs), per /challenge PROCEED_WITH_CAUTION. Unicode arrows replaced with "->" for ASCII cleanliness (prose strings, no test asserts content).
- Code review MINOR fix (66d3177e): removed stale pedaling "Half-pedal and flutter pedaling are out of scope" sentence copy-pasted into timing.notes.
- Commits: d766ec4e (initial), 66d3177e (fix). 12 schema tests green.

## Task 3: verdict_dispatch.py routing stub
- Control-flow STUB only, zero LLM/network imports (non-circularity: truth label never invokes an LLM).
- 9-step dispatch; near_threshold = abs(abs(d) - tau) <= error_bar.
- Code review IMPORTANT fix (b9c7cca0): `localizable` is now a REQUIRED _measurement key (raises TypeError if absent) per the project's explicit-exception-handling standard; `substrate_failure` stays optional-with-default (deliberate). Also: missing-dimension-key guard, Step 6 explanatory comment, module-scope `import copy`.
- Commits: 1d57b74f (initial), b9c7cca0 (fix). 13 dispatch tests green.

## Task 4: round-trip integration test
- test_round_trip.py: 7 tests against the REAL artifact + REAL route_verdict (no synthetic registry). Pulls tau from the registry provisional tolerance so a tolerance edit breaks the test. No new implementation; artifacts unchanged.
- Commit: 8935ef4f.

## Task 5: baseline_v1_audit.json hand-decomposition
- 35 hand-decomposed claims, LLM-FREE (methodology field documents this). Aggregates derived from the array (verified, header matched array exactly, no correction needed).
- Code review MINOR strengthening (16efb4d0): added test_header_aggregates_match_sample_claims_array recomputing every aggregate from sample_claims (closes header-vs-header gap; enforces "never trust the hand-written header").
- Commits: 355d188c (initial), 16efb4d0 (test strengthening). 11 integrity tests green.

## Final state
- Full claim_taxonomy suite: 43 tests passing, 0 failures.
- Non-circularity invariant protected at 3 layers: schema const:false (+ negative test), artifact llm_in_truth_label:false, verdict_dispatch zero LLM calls.
