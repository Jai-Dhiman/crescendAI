# Implementation Notes — Claim Taxonomy (#63)

Decisions, deviations, and tradeoffs made during build. Read this before running /review.

Baseline state: 431 tests collected, 5 pre-existing collection errors (missing optional deps unrelated to claim_taxonomy). jsonschema 4.26.0 installed via `uv sync --extra teacher-model-stage0`.

## Task 1: JSON Schema + Package Scaffold
- Schema draft 2020-12, dimension_entry oneOf discriminated on status const.
- Code review (P-IMPORTANT) fixes applied in commit 56518838: (1) added negative test test_llm_in_truth_label_true_fails_schema so the non-circularity const:false is load-bearing; (2) added additionalProperties:false to active/gated/scoped_out dimension defs. active_dimension gained optional "notes":string to preserve the Task 2 artifact's notes field.
- Commits: 716ad936 (initial), 56518838 (review fixes). 6 tests green.
