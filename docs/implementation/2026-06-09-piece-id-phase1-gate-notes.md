# Implementation Notes

Decisions, deviations, and tradeoffs made during build. Read this before running /review.

## Baseline
- Rust crate `piece-identify`: 24 tests pass; 1 PRE-EXISTING failure `real_recording_test::identify_real_recording` (env-gated, requires NOTES_JSON; legacy test retained per additive-PR constraint). Not caused by this work.
- Python Task-1 verify (`test_parity_fixtures.py`): 1 passed against committed golden fixture.

## Build deviations & decisions

### Controller deviation (process)
No subagent-dispatch (Task) tool was available in this environment, so the controller executed the plan's tasks directly under the same TDD discipline (write test -> watch fail -> implement -> watch pass -> commit) rather than dispatching per-task implementer/reviewer subagents. All code is the plan's verbatim code except the fixes noted below.

### Task 11 / 13: pkg/ is gitignored
`apps/api/src/wasm/piece-identify/pkg/` is gitignored and was never tracked (0 files in git). The plan's `git add ...pkg` steps were dropped — the WASM pkg is a regenerated artifact (`bun run build:wasm`), not committed. Local `wrangler dev`/vitest load the regenerated pkg.

### Task 13: identify_piece returns a JSON string, not a serde object (REAL FIX)
The plan returned the result via `serde_wasm_bindgen::to_value(&IdentifyResult)`. Under the real WASM (workerd), this MISMARSHALED the `locked: bool` field — it came back as the string "pitch" (leaked from the input notes' field name). Root cause: serde-wasm-bindgen externref-table aliasing when the SAME call both deserializes a JS array (`notes_js`) and takes a `&str` arg (`artifact_json`) and then serializes a struct containing a bool. Fix: `identify_piece` now returns `Result<Option<String>, JsValue>` (JSON string via `serde_json::to_string`, `None -> undefined`); the bridge `JSON.parse`s it. The node mock test returns a JSON string; the bridge nullish-checks `== null` (handles both null and undefined). Both the real-WASM workerd test (locks + null) and the node forwarding test pass. The Rust per-query parity (Task 12) was unaffected (it tests `elastic_cost`/`margin_gate` directly, not the JS boundary).

### Task 4: piece_id != filename stem
The plan's reference-parity test looked up the generated piece by `jf.stem`, but the catalog's `piece_id` (e.g. `bach_prelude_c_wtc1`) differs from the filename stem (`bach.prelude.bwv_846`). Fixed the test to read `piece_id` from the JSON. Also: only 1 of 16 catalog scores is present in a fresh worktree (catalog is gitignored/offloaded), and that score (a Bach prelude) is fully arpeggiated — NO onsets collapse within 50ms (tol=0 vs tol=0.05 produce identical events). So Task 4 proves ordering + `pitch%12` masking fidelity over 280 events but does NOT exercise the chord-collapse boundary on this piece; the collapse boundary is covered by Task 2's chord unit test and Task 12's full 28-query parity.

### Task 12 / certified operating point
`certified_operating_point_holds` passes at exactly 14/16 in-catalog locks (0/12 OOD) — the carried-forward >=14 zero-slack caution did NOT trigger; no loosening to >=13 was needed. The per-query 1e-4 cost+margin+best-piece+lock parity (`rust_elastic_cost_matches_python_per_candidate`, `rust_margin_gate_matches_python_decision`) PASS on all 28 queries.

### Pre-existing failures (NOT caused by this work)
- Rust: `real_recording_test::identify_real_recording` is env-gated (needs NOTES_JSON) and panics without it — fails in baseline, retained legacy test. Crate result: 41 passed, 1 failed (this one).
- TS typecheck: 18 pre-existing errors in `src/harness/*` (catalog-is-code `node:fs/promises` resolution + phase2 binding types) — documented in MEMORY, unrelated to piece-id. No errors in any file this PR touched.
