# Continuity-Aware Chroma Score-Follower Design

**Goal:** Prevent the chroma-DTW score-follower from teleporting to wrong score positions across chunks by adding positional continuity, separation-margin confidence, and a silence gate — all behind the same one-call WASM interface.

**Not in scope:**
- 88-bin or pitch-resolved chroma features
- AMT note-level symbolic alignment
- parangonar / (n)ASAP millisecond-level evaluation harness
- Non-monotonic within-session rehearsal model or Match-format multiplicity
- Decoupling from the existing AMT gate in the DO

---

## Problem

The current `align_chunk_chroma` in `apps/api/src/wasm/score-analysis/src/chroma_dtw.rs` is a stateless, free-start subsequence DTW. Three failures reproduced on a bucket-1 amateur Chopin Ballade 1 recording:

1. **Teleport:** The opening 15 s chunk locked to bars 261–262 of a 264-bar score. Free-start search with no positional prior is underdetermined when the query is short or contains repeated harmonic material.
2. **Mean-cost is not a confidence signal:** Correct and wrong alignments both produce costs ~0.17–0.21 because cost scales with player skill, not location correctness. There is no reliable way for the DO to know which is which.
3. **Silence locks:** Near-silent chunks (rms~0.01) produce confident-looking wrong locks (cost 0.049 → wrong bar 48) because L2-normalized chroma from silence is nearly uniform, and a uniform column matches cheaply to any score region.

The audio chroma is L2-normalized per column at the MuQ source (`apps/inference/muq/chroma.py`), so raw energy is gone by the time bytes reach the DTW.

---

## Solution (from the user's perspective)

After this change:
- A practice session that starts at bar 1 stays near bar 1 on subsequent chunks; it no longer jumps to the end of the score.
- A session that genuinely jumps (rehearsal loop to an earlier passage) is correctly re-anchored when the new location is unambiguous.
- Silent intro chunks do not produce a bar-map output; the DO emits Tier 3 (scores only) for those chunks and preserves the last good position.
- The bar-range output fed to `buildBarAnalysisFacts` and the synthesis prompt is more trustworthy.

---

## Design

### Approach: single-DP-fill with two readouts + in-WASM arbitration

The existing last-audio-row accumulated-cost array `d[last_audio_row, j]` is already comparable across all score positions `j` after one DP fill. No second pass is needed.

From that array, two candidates are read:

- **Global best:** `argmin d[last, j]` over all `j`
- **In-band best:** `argmin d[last, j]` over `j ∈ [expected − band_back, expected + band_fwd]`

**Separation margin** (skill-invariant confidence) for each candidate: the gap between the candidate cost and the minimum cost outside a disjoint neighborhood of ±`neighbor_frames` around the winner. Both costs scale with player skill, so the gap is invariant to skill level.

**Uniformity gate (pre-DP):** Fraction of chunk columns whose normalized chroma is sufficiently non-uniform (peaky). Below `uniformity_min` → `status="abstained"`, DP is skipped entirely. This catches silence.

**Arbitration (warm = `expected_score_frame >= 0` and within score bounds):**
- Cold (`expected_score_frame < 0` or beyond score): take global best; if its margin < `margin_min` → abstained, else aligned.
- Warm: prefer in-band; if in-band margin >= `margin_min` → aligned; else if global is clearly elsewhere and clears `margin_min` → relocalized; else abstained.

**State machine in the DO (`session-brain.ts`):**
- `aligned` or `relocalized`: update `expectedScoreFrame = end_score_frame`, Tier 1, feed bar range to `buildBarAnalysisFacts`.
- `abstained`: leave `expectedScoreFrame` unchanged (preserve last good), emit Tier 3, no bar range to teacher.
- Thrown WASM error (existing try/catch → null): Tier 3, same as before.

All tuning constants (`band_back_frames`, `band_fwd_frames`, `margin_min`, `uniformity_min`) are call parameters, so behavior is adjustable from the DO without a WASM rebuild.

### Key decision: continuity vs rehearsal jumps

Band-first search with a separation-margin-triggered global relocalization fallback. Abstain when no location is unambiguous. This handles:
- Normal forward play → in-band match, aligned
- Rehearsal jump to a new unambiguous region → relocalized
- Ambiguous / repeated material → abstained (Tier 3, teacher gets no bar hint)

### Critical risk

The separation margin must actually separate right-from-wrong alignment on real amateur chunks. This is validated in Task 0 (the gating harness): the amateur opening chunk that previously teleported must have a lower (worse) margin than the correct cold-start position. If the margins do not separate, the arbitration logic is unfounded and the plan must be revisited before any implementation proceeds.

---

## Modules

### `chroma_dtw.rs` (DEEP — gains continuity + confidence + arbitration behind one call)

**Interface:**
```
pub fn chroma_dtw_native(
    audio_f32: &[f32],
    n_audio: u32,
    score_bars: &[ScoreBar],
    frame_rate_hz: f32,
    decim_hz: f32,
    expected_score_frame: i32,   // -1 = cold start
    band_back_frames: u32,
    band_fwd_frames: u32,
    margin_min: f32,
    uniformity_min: f32,
) -> Result<BarMapChroma, String>
```

WASM entry point `align_chunk_chroma` mirrors these parameters.

**Hides:** full n_a × n_s DP matrix, two-readout argmin, disjoint-neighborhood margin computation, uniformity fraction gate, arbitration state machine, backtrack, bar_per_frame decimation.

**Depth verdict:** DEEP — 5 new parameters expand the API minimally while hiding all new complexity inside one function.

### `types.rs` (BarMapChroma extension)

**Interface:** adds `end_score_frame: u32`, `confidence: f32`, `status: String` to `BarMapChroma`.

**Hides:** nothing — pure data struct. This is intentionally shallow; depth lives in `chroma_dtw.rs`.

### `session-brain.schema.ts` (additive schema change)

**Interface:** adds `expectedScoreFrame: z.number().int().default(-1)` to `sessionStateSchema` and `createInitialState`.

**Hides:** Zod parsing/validation of the new field.

**Depth verdict:** SHALLOW — justified, this is a data schema not a logic module.

### `wasm-bridge.ts` (BarMapChroma interface + alignChunkChroma signature)

**Interface:** `BarMapChroma` gains `end_score_frame`, `confidence`, `status`; `alignChunkChroma` gains 4 new parameters.

**Hides:** nothing — forwarding wrapper. SHALLOW — justified, it's a typed forwarding layer.

### `session-brain.ts` (DO: state + status switch only)

**Interface:** reads/writes `expectedScoreFrame`; dispatches on `status`; calls `alignChunkChroma` with new args.

**Hides:** session state persistence across chunks; Tier 1/3 routing by status.

**Depth verdict:** SHALLOW at the new-code level — justified, the DO is already an orchestrator, not a logic module.

---

## Verification Architecture

**Canonical success state:** The Rust integration test `continuity_teleport_regression` passes: the amateur opening chunk (`cs_000`) with `expected_score_frame` set to an early-bar position locks to bars 1–5, not 261–262. The cold-start non-regression test also passes: `cs_111` with `expected=-1` still lands bars 25–40.

**Automated check (Task 0 gating):** Run the fixture generator on both recordings, compute separation margins for the teleport-wrong alignment and the correct cold-start alignment, and assert that the wrong alignment has a clearly lower (or negative) margin while the correct one has a positive margin. If this assertion fails, the plan stops — the fundamental assumption is invalidated.

**Full automated check (Tasks 1–N):**
```bash
cd apps/api/src/wasm/score-analysis && cargo test
cd apps/api && bun run test -- --run
```

**Harness:** Task 0 is the gating harness — it generates fixtures and validates the margin hypothesis before any Rust implementation is written.

---

## File Changes

| File | Change | Type |
|------|--------|------|
| `apps/api/src/wasm/score-analysis/tests/fixtures/generate.py` | Add amateur `cs_000` fixture + pro `cs_111` margin-validation section | Modify |
| `apps/api/src/wasm/score-analysis/tests/fixtures/ballade1_amateur_cs000/audio_chroma.bin` | New fixture binary | New |
| `apps/api/src/wasm/score-analysis/tests/fixtures/ballade1_amateur_cs000/score_bars.json` | New fixture score bars | New |
| `apps/api/src/wasm/score-analysis/tests/fixtures/ballade1_amateur_cs000/expected.json` | New fixture expected values (with margin bounds) | New |
| `apps/api/src/wasm/score-analysis/src/chroma_dtw.rs` | Add uniformity gate, two-readout margin, arbitration, new native + WASM parameters | Modify |
| `apps/api/src/wasm/score-analysis/src/types.rs` | Add `end_score_frame`, `confidence`, `status` to `BarMapChroma` | Modify |
| `apps/api/src/wasm/score-analysis/tests/chroma_dtw_roundtrip.rs` | Add 5 new integration tests (teleport, silence, relocalize, low-margin, margin gate) | Modify |
| `apps/api/src/services/wasm-bridge.ts` | Extend `BarMapChroma` interface; extend `alignChunkChroma` signature | Modify |
| `apps/api/src/do/session-brain.schema.ts` | Add `expectedScoreFrame` field to schema + `createInitialState` | Modify |
| `apps/api/src/do/session-brain.ts` | Read/write `expectedScoreFrame`; pass new args; dispatch on `status` | Modify |
| `apps/api/src/do/session-brain.unit.test.ts` | Add tests: frame carried, abstain preserves frame, reset on piece re-id | Modify |
| `apps/api/src/services/wasm-bridge.test.ts` | Extend `alignChunkChroma` forwarding test for 9 args | Modify |

---

## Open Questions

- Q: What value should `band_back_frames` / `band_fwd_frames` default to in the DO?
  Default: 150 frames back (3 s at 50 Hz, catches tempo variation) and 300 frames forward (6 s, assumes at most 6 s of advancement per 15 s chunk for slow performers).
- Q: What should `margin_min` default to?
  Default: 0.02 (empirically chosen to clear the gap seen between correct and teleport alignments; Task 0 will validate this).
- Q: What should `uniformity_min` default to?
  Default: 0.3 (fraction of frames that must be non-uniform; chunks with < 30% peaky columns are treated as silence).
- Q: What size neighborhood for the disjoint-margin computation?
  Default: ±50 frames (±1 s at 50 Hz), matching a typical score phrase.
