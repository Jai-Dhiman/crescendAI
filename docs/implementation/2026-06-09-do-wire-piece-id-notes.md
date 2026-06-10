# Implementation Notes

Decisions, deviations, and tradeoffs made during build. Read this before running /review.

## Baseline (pre-build)
- typecheck: 18 pre-existing errors (11 src/harness/*, 7 unrelated test/script type errors). None in session-brain*, wasm-bridge*, or piece-identify.
- cargo test (piece-identify): 41 pass, 1 fail = `real_recording_test::identify_real_recording` (env NOTES_JSON not set) — DOCUMENTED pre-existing; will disappear after Task 5 deletes real_recording_test.rs.
- session-brain.schema.test.ts: 3 pass. wasm-bridge.test.ts: 4 pass. wasm-bridge.workerd.test.ts: 7 pass.

## Carry-forward cautions from /challenge (PROCEED_WITH_CAUTION)
1. accumulateAndIdentify MUST mutate caller state in place; NO readState/writeState/CAS of its own.
2. MAX_IDENTIFICATION_BUFFER=1200 keeps MOST RECENT notes (slice(-CAP)); add a truncation unit test.
3. Task 3 falsification probe: set PIECE_ID_MARGIN_THRESHOLD=-1, confirm ambiguous test FAILS, restore 0.0935, record in commit body.
4. chunk_ready (finalizeChunk) wiring ships covered only indirectly; ensure helper called identically from both sites.
