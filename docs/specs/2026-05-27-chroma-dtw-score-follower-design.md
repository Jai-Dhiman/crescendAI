# Chroma-DTW Score Follower Design

**Goal:** Replace the broken note-level score follower in `apps/api/src/wasm/score-analysis/src/score_follower.rs` with a stateless chroma-based subsequence DTW that aligns each 15-second audio chunk to the score in Rust WASM, producing a bar range and a 5 Hz per-frame bar mapping suitable for live cursor following.

**Not in scope:**
- Per-note alignment (`align_notes_in_window`) — deferred until AMT is redeployed.
- Score-cursor frontend wiring — the new `bar_per_frame` field reaches the WebSocket payload, but the cursor implementation in `apps/web/src/lib/score-cursor.ts` is not modified by this work.
- Benchmark harness against matchmaker / synctoolbox — a separate brainstorm.
- AMT pipeline reactivation.

## Problem

`apps/api/src/wasm/score-analysis/src/score_follower.rs` implements note-level symbolic DTW: it compares AMT-produced note onsets against score notes. Two failures:

1. **Effectively broken.** On Chopin Ballade 1 cold-start at 111s, the current Rust DTW lands at bars 25-29 with a 7% pitch-match rate (verified by spike at `apps/inference/score-align-spike/spike.py`). On uniformly-textured Romantic repertoire it gets stuck at bars 1-3.
2. **Currently unused in production.** Per project memory, AMT isn't deployed; all sessions are Tier 3. The note-DTW therefore receives no input — Tier 1 bar-aligned analysis hasn't run on any real chunk.

Score-cursor following on the web client cannot ship until alignment is reliable. The spike (`apps/inference/score-align-spike/spike.py`) demonstrated that an offline chroma-based subsequence DTW solves the cold-start case (bars 30-34 instead of 1-4), handles drilling deterministically, and runs in ~450 ms per 15s chunk on a laptop CPU — well within the async budget.

## Solution (from the user's perspective)

When a pianist records a practice chunk, the score-cursor in the web app advances through the score in time with their playing. Drilling the same passage three times is tracked correctly each time (no manual reset). Starting a session mid-piece works without a warm-up phase. Failed alignments leave the cursor frozen rather than misleading the teacher LLM with a wrong bar number.

## Design

Stateless per-chunk alignment. For each 15s chunk:

1. MuQ Python endpoint, which already decoded the audio for quality scoring, computes a 12-row chroma matrix at 50 Hz via `librosa.feature.chroma_cqt`, L2-normalizes per column with a 1e-3 floor, base64-encodes the row-major float32 bytes, and returns them alongside the existing 6 scores.
2. The `SessionBrain` DO base64-decodes the chroma bytes and passes them to a new Rust WASM entry point `align_chunk_chroma(audio_bytes, n_frames, score_bars, frame_rate_hz, decim_hz)`.
3. WASM builds the score chroma matrix from the score JSON's note list (no FFT — just incrementing `chroma[pitch%12]` over each note's frame window, then L2-normalizing), runs subsequence DTW with cosine distance and the standard monotonic step pattern, backtracks, decimates the score-axis warping path from 50 Hz to 5 Hz, and maps each decimated score frame to its bar number via `bars[].start_seconds`.
4. WASM returns `BarMapChroma { bar_min, bar_max, cost, bar_per_frame }`. The DO stores it in the accumulator slot where the old `BarMap` lived and forwards `bar_per_frame` to the WebSocket client.

### Key decisions

- **Audio chroma in Python, score chroma in Rust.** Audio is already decoded in MuQ; shipping PCM bytes to a CF Worker just to redecode and FFT in WASM would waste a kilobyte-per-millisecond round trip. Score chroma is so cheap to build (no FFT, ~50 ms for a 9-minute piece) that recomputing it in WASM on every call beats any caching scheme — sidesteps R2 versioning, ingest pipeline coupling, and cache invalidation.
- **Stateless per-chunk.** No `last_known_bar` continuity. Drilling becomes a free property (consecutive chunks aligning to overlapping bar ranges) rather than a heuristic. Cold-starts work without warm-up.
- **Per-note alignment split off.** Chroma DTW physically cannot produce per-note timing deviations — that's harmonic, not onset, data. The future `align_notes_in_window(perf_notes, score_bars, bar_range)` is a separate WASM function for the AMT path, not in scope here.
- **150-byte richer output (`bar_per_frame`) over 24-byte minimal.** Cost is ~3 lines of Rust (decimate the existing warping path); benefit is live score-cursor following. UI no longer needs per-chunk timestamp arithmetic to know where the player is.

### Trade-offs chosen

- **Within-chunk jumps (T4 in the spike) are not handled.** The standard DTW step pattern is monotonic, so a chunk containing a passage switch will be mapped to one coherent bar region rather than two. Acceptable because realistic 15s chunks rarely contain mid-chunk jumps; the HMM-based fix is deferred until real-data evidence demands it.
- **Score chroma is rebuilt every call (no R2 cache).** Worth ~50 ms per chunk to avoid an entire caching subsystem.
- **`analyze_tier1` (per-note expression analysis) is dark in production until AMT lands.** This is a continuation of the existing state, not a regression.

## Modules

**`apps/inference/muq/chroma.py :: chroma_feature(y, sr) -> (bytes, int)`** (DEEP)
- Interface: takes a mono float32 waveform and sample rate; returns `(row_major_f32_bytes, n_frames)`.
- Hides: librosa chroma_cqt call, hop=441 frame-rate enforcement, 1e-3 floor, L2-per-column normalization, dtype cast, row-major byte serialization.
- Tested through: synthetic-waveform pytest assertions on output shape, dtype, frame count, and dominant pitch class.

**`apps/api/src/wasm/score-analysis/src/chroma_dtw.rs :: align_chunk_chroma(...) -> BarMapChroma`** (DEEP)
- Interface: one `#[wasm_bindgen]` entry point taking `(audio_bytes: &[u8], n_audio: u32, score_bars_js: JsValue, frame_rate_hz: f32, decim_hz: f32)` and returning `Result<JsValue, JsValue>` carrying a `BarMapChroma`.
- Hides: score-chroma construction from note list, subseq cosine DTW with backtracking, warping-path decimation, score-frame → bar lookup.
- Tested through: cargo tests calling `align_chunk_chroma` directly with checked-in `(audio_chroma.bin, score_bars.json, expected.json)` fixtures generated by Task Group 0; no private-function tests.

**`apps/api/src/services/wasm-bridge.ts :: alignChunkChroma(...)`** (DEEP)
- Interface: `alignChunkChroma(audioChromaBytes: Uint8Array, chromaFrames: number, scoreBars: ScoreBar[], frameRateHz: number, decimHz: number): BarMapChroma`.
- Hides: WASM module loading, JsValue marshaling, error-string-to-Error conversion, byte-slice handoff.
- Tested through: vitest unit test that feeds a fixture-derived `Uint8Array` and a known score and asserts the resulting `BarMapChroma` matches the expected bar range.

## Verification Architecture

- **Canonical success state:** `align_chunk_chroma` on the checked-in Chopin Ballade 1 audio[0..120s] fixture returns `bar_min ∈ [1, 5]`, `bar_max ∈ [30, 35]`, `bar_per_frame.len() == 60`, monotone non-decreasing, `cost < 0.25`.
- **Automated check:** `(cd apps/api/src/wasm/score-analysis && cargo test chroma_dtw_roundtrip)` exits zero.
- **Harness:** Task Group 0 — a Python script `apps/api/src/wasm/score-analysis/tests/fixtures/generate.py` adapted from `apps/inference/score-align-spike/spike.py`. Takes `(audio_wav, score_json, start_s, dur_s)` and emits three files into `tests/fixtures/{slug}/`: `audio_chroma.bin` (raw f32 LE, row-major 12×N), `score_bars.json` (the bars slice copied out of the score), `expected.json` (`{bar_min_lo, bar_min_hi, bar_max_lo, bar_max_hi, n_frames, decim_n, cost_hi}`). Generator commits two fixtures: Chopin Ballade 1 forward-2min and Chopin Ballade 1 cold-start-15s-at-111s. Rust tests load these and assert. The generator is run once at fixture-creation time and is not part of CI.

## File Changes

| File | Change | Type |
|------|--------|------|
| `apps/inference/muq/chroma.py` | `chroma_feature(y, sr) -> (bytes, int)` helper | New |
| `apps/inference/muq/test_chroma.py` | pytest for `chroma_feature` | New |
| `apps/inference/muq/handler.py` | Add `chroma_b64`, `chroma_frames`, `chroma_frame_rate_hz` to success response (lines 144-173 region) | Modify |
| `apps/inference/muq/requirements.txt` | Add `pytest` if absent | Modify |
| `apps/api/src/wasm/score-analysis/src/chroma_dtw.rs` | `align_chunk_chroma` entry point + private helpers | New |
| `apps/api/src/wasm/score-analysis/src/types.rs` | Add `BarMapChroma` struct; delete `FollowerState` | Modify |
| `apps/api/src/wasm/score-analysis/src/lib.rs` | Register `chroma_dtw` mod, export `align_chunk_chroma`, remove `align_chunk` and `score_follower` references | Modify |
| `apps/api/src/wasm/score-analysis/src/score_follower.rs` | Delete | Delete |
| `apps/api/src/wasm/score-analysis/tests/fixtures/generate.py` | Fixture generator | New |
| `apps/api/src/wasm/score-analysis/tests/fixtures/ballade1_forward_2min/{audio_chroma.bin,score_bars.json,expected.json}` | Committed fixture | New |
| `apps/api/src/wasm/score-analysis/tests/fixtures/ballade1_coldstart_111s/{audio_chroma.bin,score_bars.json,expected.json}` | Committed fixture | New |
| `apps/api/src/services/inference.ts` | Extend `MuqResult` and `callMuqEndpoint` with `chromaBytes` + `chromaFrames` | Modify |
| `apps/api/src/services/wasm-bridge.ts` | Add `alignChunkChroma` + `BarMapChroma`; delete `alignChunk` + `FollowerState` + `AlignChunkResult` | Modify |
| `apps/api/src/services/wasm-bridge.test.ts` | Vitest for `alignChunkChroma` against fixture | New |
| `apps/api/src/do/session-brain.ts` | Replace both `wasm.alignChunk(...)` sites with `wasm.alignChunkChroma(...)`; remove `followerState` reads | Modify |
| `apps/api/src/do/session-brain.schema.ts` | Drop `followerState` from `sessionStateSchema` and `createInitialState`; bump `version` field meaning (existing field, no migration code needed — DO Zod parse defaults absent field to undefined and we delete it) | Modify |
| `apps/api/src/do/session-brain.unit.test.ts` | Add chunk-handler tests for chroma path (happy, null-chroma, contract-mismatch) | Modify |
| `apps/api/src/wasm/score-analysis/Cargo.toml` | Add `[dev-dependencies] serde_json` if not already a regular dep (already is — verify) | Verify only |

## Open Questions

- **Q:** Should `align_chunk_chroma` also return the raw `cost` per-frame for richer confidence display?  **Default:** No — return mean path cost only. UI can decide thresholds; per-frame cost is YAGNI until a real consumer asks.
- **Q:** Pytest in the MuQ image — does the production Dockerfile need to install it?  **Default:** No — `pytest` goes in `requirements.txt` under a dev-only convention, but the test runs locally via `uv run pytest apps/inference/muq/test_chroma.py`. The handler itself doesn't import pytest, so the production image is unaffected even if it's installed.
