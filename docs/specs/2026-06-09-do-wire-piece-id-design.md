# DO Wire Piece-ID (v2 gate) Design

**Goal:** During a live (or eval-replay) session, the `SessionBrain` Durable Object identifies the piece a student is playing using the Phase-0-CERTIFIED v2 chroma-recall + elastic-DTW margin gate, locking only on a certified-confident match — replacing the uncertified legacy 3-stage (N-gram/rerank/DTW-confirm) pipeline, which is then deleted.

**Not in scope:**
- Re-locking the synthesis `_SONNET_BASELINE` (#28; credit-blocked, disjoint region).
- Any change to the synthesis / Anthropic teacher path. Verification stops before session-end synthesis.
- Remote R2 cleanup of stale `fingerprint/v1/*` keys (pre-beta, local-first; harmless).
- Changing the recall-certification status (still TA=0.875 on 16 recordings; the gate fails gracefully to Tier-3 on a miss — unchanged).
- `match_piece_text` / the `set_piece` text-query path (kept as-is).

## Problem

The certified gate already ships in Rust/WASM (`apps/api/src/wasm/piece-identify/{chroma,gate,identify}.rs`, exported as `identify_piece`, wrapped as `wasm-bridge.ts:identifyPiece`) but **nothing at runtime calls it**. The DO still runs the uncertified legacy path:

- `session-brain.ts:tryIdentifyPiece` (~line 2181) loads `fingerprint/v1/{ngram_index,rerank_features,catalog}.json` from R2 and runs `wasm.ngramRecall` → `wasm.rerankCandidates` → `wasm.dtwConfirm`, locking when rerank similarity ≥ 0.5 and DTW confirms.
- Accumulation (`finalizeChunk` ~line 938) keys off `identificationNoteCount` (a running integer) and passes only **the current chunk's** `perfNotes` to identification — not the accumulated cross-chunk buffer the gate was certified on (full-piece / ~90s windows).

Two consequences: (1) the runtime uses an uncertified discriminator the harness work (#26 Stage-0c/0d/0e/0f) proved inferior; (2) identification runs on a too-small single-chunk window, so even the legacy path under-recalls.

Additionally, the legacy surface is now dead weight: Rust modules `ngram.rs`/`rerank.rs`/`dtw_confirm.rs` + the `#[wasm_bindgen]` exports `ngram_recall`/`compute_rerank_features`/`rerank_candidates`/`dtw_confirm`, the legacy types in `types.rs`, the `real_recording_test.rs` cargo test, and the bridge wrappers/interfaces.

## Solution (from the user's perspective)

A student plays. As chunks arrive, the DO accumulates their AMT-transcribed notes into a bounded cross-chunk buffer. Once enough notes have accumulated, the DO runs the certified gate against the v2 catalog artifact. When the gate clears the certified margin threshold (0.0935), the DO locks the piece, sets `pieceIdentification = { pieceId, confidence: <margin>, method: "identify_v2" }`, and emits a `piece_identified` WebSocket event with the correct piece. On an ambiguous or out-of-catalog (OOD) performance it stays unknown (Tier-3) and keeps accumulating — never a false lock. Downstream score-context loading and bar analysis are unchanged: they consume `pieceId`, not the confidence magnitude.

## Design

### Approach

1. **State:** Replace the `identificationNoteCount` integer with an `identificationNoteBuffer` — a Zod-validated array of `{pitch, onset, offset, velocity}` perf-notes, capped at `MAX_IDENTIFICATION_BUFFER` (keep the most-recent N). The buffer length subsumes the old count.
2. **Deep module — `tryIdentifyPiece` v2:** Rewrite the method to: fetch `fingerprint/v2/piece_index.json` from `SCORES` as **text** (`.text()`), call `wasm.identifyPiece(buffer, artifactJson, PIECE_ID_MARGIN_THRESHOLD)`, and return a lock decision. Hides R2 IO + the WASM gate behind a small interface. Drops all `fingerprint/v1` loads and the DTW-confirm/score-context stage.
3. **Shared accumulation helper — `accumulateAndIdentify`:** Extract the "append perfNotes to the buffer (truncate), run identification on the accumulated buffer once it crosses `MIN_NOTES_FOR_IDENTIFICATION`, lock + emit `piece_identified` on success" logic into one private method that **mutates the passed-in state object and uses the passed-in `ws`**. Call it from **both** `finalizeChunk` (chunk_ready path) and `handleEvalChunk` (eval_chunk path).

   *Why this matters / correction to the original brief:* the brief assumed "both `eval_chunk` and `chunk_ready` reach `finalizeChunk`." They do **not** — `handleEvalChunk` (line 1217) is a separate method that today does no identification at all. Extracting the shared helper is what makes the eval-replay path identify pieces AND makes the behavior testable through the `eval_chunk` WS interface with **pure note input** (no MuQ/AMT mocks, no Anthropic call) — satisfying the hard "verification must not require Anthropic credits" constraint.

4. **Delete the legacy pipeline** only after nothing calls it: bridge wrappers + interfaces, Rust modules + exports + types, the `real_recording_test.rs` cargo test, and the legacy bridge test cases. Rebuild the WASM pkg. Keep `chroma.rs`/`gate.rs`/`identify.rs`/`identify_piece`/`text_match.rs`/`match_piece_text`/`parity_test.rs` and the v2 types.

### Key decisions & trade-offs

- **Shared helper over duplicating logic in two methods.** The alternative (inline the buffer/identify block in both `finalizeChunk` and `handleEvalChunk`) duplicates ~25 lines and the WS-emit shape, and drifts. One deep method with a `(state, perfNotes, ws)` interface is testable in isolation through the public `eval_chunk` interface. Trade-off: the helper mutates its `state` argument (consistent with how `finalizeChunk`/`handleEvalChunk` already thread a single in-memory state object through their read→mutate→write cycle); it does NOT do its own `readState`/`writeState`, so the caller's existing single write persists the buffer + lock atomically under the caller's concurrency discipline.
- **`MAX_IDENTIFICATION_BUFFER = 1200`.** The certified gate operated on full-piece / ~90s windows. AMT emits well under ~10 notes/sec for typical piano; 1200 notes comfortably exceeds a 90s identification window and `MIN_NOTES_FOR_IDENTIFICATION = 30`, while bounding DO state size (1200 × ~4 numbers). Keep the most-recent 1200 (a long session caps memory; the gate only needs a representative recent window, and once locked the buffer is no longer appended).
- **`confidence` now carries the gate margin, `method: "identify_v2"`.** The margin is a cost-gap (~0.1–1.0), not a 0–1 similarity. The 4 consumers (`hasPieceMatch` = `pieceLocked`; `loadScoreContext(pieceIdentification.pieceId)` ×2; `chunkSignal.hasPieceMatch`; `handleSetPiece` setting `pieceLocked`) consume `pieceId`/`pieceLocked` only — none branch on confidence magnitude — so the contract is preserved. `method` lets any future consumer distinguish v2 from the legacy `"fingerprint"`/text `"set_piece"` values.
- **Once locked, stop accumulating.** Both call sites guard on `!state.pieceLocked` before appending (matches the existing `finalizeChunk` guard), so a locked session does not grow the buffer. `handleSetPiece` (text-lock) sets `pieceLocked=true` with `pieceIdentification=null`; the new helper's `!pieceLocked` guard means a text-locked session won't auto-identify — preserving existing behavior (the brief's "resolved on next chunk" comment is legacy and already non-functional; not in scope to change).

### Concurrency

`finalizeChunk` already runs inside `ctx.blockConcurrencyWhile` and re-reads state at entry; the helper mutates that re-read state in-memory and the existing `writeState` persists it — no new await between the `SCORES.get` and the state write that isn't already inside the serialized region. `handleEvalChunk` is strictly sequential (ack-gated). The one new await is `this.env.SCORES.get(...).text()` inside the helper; per `apps/api/TS_STYLE.md` the surrounding method already owns the state object across this await (no concurrent writer in either path), matching the existing `loadScoreContext`/baseline-load awaits in the same methods. No `c.env` destructuring; structured `console.log(JSON.stringify(...))` on the R2-miss / parse-failure branches; no `any`.

## Modules

### `accumulateAndIdentify` (private, `session-brain.ts`)
- **Interface:** `private async accumulateAndIdentify(state: SessionState, perfNotes: PerfNote[], ws: WebSocket): Promise<void>` — appends notes to `state.identificationNoteBuffer` (truncating to the most-recent `MAX_IDENTIFICATION_BUFFER`), and when the buffer length ≥ `MIN_NOTES_FOR_IDENTIFICATION` runs identification on the accumulated buffer; on a locking result sets `state.pieceLocked`/`state.pieceIdentification` and emits the `piece_identified` WS event. No-op when `state.pieceLocked` or `perfNotes` is empty.
- **Hides:** buffer truncation policy, the MIN-notes threshold gate, the lock/emit sequencing, and the call into `tryIdentifyPiece`.
- **Tested through:** the `eval_chunk` WebSocket interface (drive the DO with `eval_chunk` messages carrying `midi_notes`; assert `pieceLocked`/`pieceIdentification.pieceId` in DO storage and the `piece_identified` WS send). No internal-state-only or private-method tests.
- **Depth verdict:** DEEP — one verb, hides cross-chunk accumulation + bounded retention + the certified-gate decision behind a 3-arg interface.

### `tryIdentifyPiece` v2 (private, `session-brain.ts`)
- **Interface:** `private async tryIdentifyPiece(buffer: PerfNote[]): Promise<{ pieceId; composer; title; confidence; method } | null>` — loads the v2 artifact text from `SCORES` and runs the WASM gate; returns the lock payload only when `result.locked`, else `null`.
- **Hides:** R2 fetch of `fingerprint/v2/piece_index.json` as text, the `wasm.identifyPiece` call, the threshold constant, and the locked-vs-unlocked decision (a non-null-but-unlocked WASM result returns `null` here).
- **Tested through:** indirectly via `accumulateAndIdentify` (the `eval_chunk` interface) — locking artifact → lock; ambiguous/OOD artifact → no lock.
- **Depth verdict:** DEEP — collapses R2 + WASM + threshold into "give me the buffer, I'll tell you if we can certifiably lock."

## Verification Architecture

- **Canonical success state:** Driving the DO via `eval_chunk` WS messages whose accumulated `midi_notes` match an in-catalog piece's chord-events sets `state.pieceLocked === true`, `state.pieceIdentification.pieceId === <expected>`, `state.pieceIdentification.method === "identify_v2"`, and sends a `piece_identified` WS frame for that piece. An ambiguous/OOD performance leaves `pieceLocked === false`/`pieceIdentification === null`. Cross-chunk: the lock fires only after the buffer crosses `MIN_NOTES_FOR_IDENTIFICATION` across multiple `eval_chunk` messages, not on the first sub-threshold chunk.
- **Automated check:** `cd apps/api && bun run test src/do/session-brain.piece-id.test.ts` (new workerd integration test, real WASM + miniflare R2) green; `cargo test` in `piece-identify` green post-deletion; `bun run typecheck` clean; full `bun run test` (workerd) + `bun run test:scripts` (node) green with no new failures beyond the documented pre-existing catalog/harness ones.
- **Harness:** **Task Group 0** — seed a minimal `fingerprint/v2/piece_index.json` into the miniflare `SCORES` R2 binding inside the test and drive the DO via `runInDurableObject` + `webSocketMessage` (the pattern already used by `session-brain.canary.test.ts`). The locking fixture (a 2-piece artifact: `exact` sharing the query chord-events + chroma, `decoy` disjoint; plus a 4-note C-E-G-C query) is the one already proven in `wasm-bridge.workerd.test.ts:identifyPiece (real WASM)` and is reused here.
- **Manual (documented, not in CI — must NOT trigger Anthropic):** `just fingerprint && just seed-fingerprint && just dev-light`; start a session; feed an in-catalog performance; confirm the `piece_identified` WS event names the correct piece; feed an OOD piece and confirm NO lock. STOP before session-end synthesis to avoid the Anthropic teacher call.

## File Changes

| File | Change | Type |
|------|--------|------|
| `apps/api/src/do/session-brain.schema.ts` | Add `identificationNoteBuffer` (Zod perf-note array, `.default([])`); remove `identificationNoteCount`; update `createInitialState` | Modify |
| `apps/api/src/do/session-brain.ts` | Add `PIECE_ID_MARGIN_THRESHOLD` + `MAX_IDENTIFICATION_BUFFER`; add `accumulateAndIdentify`; rewrite `tryIdentifyPiece` to v2; call helper from `finalizeChunk` + `handleEvalChunk`; drop legacy imports | Modify |
| `apps/api/src/do/session-brain.piece-id.test.ts` | New workerd integration test (seed v2 R2, drive `eval_chunk`, assert lock/no-lock/cross-chunk) | New |
| `apps/api/src/services/wasm-bridge.ts` | Remove `ngramRecall`/`rerankCandidates`/`dtwConfirm` wrappers + `NgramIndex`/`RerankFeatures`/`NgramCandidate`/`RerankResult`/`DtwConfirmResult` interfaces | Modify |
| `apps/api/src/services/wasm-bridge.test.ts` | Remove the `ngramRecall` mock test case (keep `identifyPiece`) | Modify |
| `apps/api/src/services/wasm-bridge.workerd.test.ts` | Remove the `ngramRecall` real-WASM case (keep `identifyPiece`) | Modify |
| `apps/api/src/wasm/piece-identify/src/lib.rs` | Remove `mod ngram/rerank/dtw_confirm`, `mod real_recording_test`, and the 4 legacy `#[wasm_bindgen]` exports | Modify |
| `apps/api/src/wasm/piece-identify/src/types.rs` | Remove `NgramIndex`/`RerankFeatures`/`NgramCandidate`/`RerankResult`/`DtwConfirmResult` | Modify |
| `apps/api/src/wasm/piece-identify/src/{ngram,rerank,dtw_confirm,real_recording_test}.rs` | Delete | Delete |
| `apps/api/src/wasm/piece-identify/pkg/*` | Rebuild via `bun run build:wasm` | Modify |
| `justfile` | `test-piece-id`: drop the now-vacuous `-- --skip real_recording` filter | Modify |

## Open Questions

- Q: Should `handleSetPiece` (text-lock) be allowed to subsequently auto-identify and overwrite the text lock?  Default: No — preserve current behavior; the `!pieceLocked` guard blocks it. Out of scope for #26.
- Q: Cap value 1200 — exact number?  Default: 1200. Documented rationale above; safe to tune later without interface change.
