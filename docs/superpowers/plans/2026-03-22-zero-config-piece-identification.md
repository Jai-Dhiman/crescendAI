# Zero-Config Piece Identification - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace ByteDance AMT with Aria-AMT, split inference into independent MuQ + AMT endpoints with parallel dispatch, and add multi-signal MIDI-based piece identification so students never need to name what they're playing.

**Architecture:** Two-phase delivery. Phase 2a splits the combined HF inference endpoint into MuQ-only (15s) and Aria-AMT (30s) endpoints, with the Durable Object orchestrating parallel fan-out and audio buffering. Phase 2b adds automatic piece identification via N-gram pitch fingerprinting, statistical reranking, and DTW confirmation against the 242-piece score library.

**Tech Stack:** Python (Aria-AMT, HF Inference Endpoints), Rust/WASM (Cloudflare Workers, Durable Objects), `aria-amt` library, `futures::join!` for parallel dispatch, R2 for fingerprint index storage.

**Spec:** `docs/superpowers/specs/2026-03-22-zero-config-piece-identification-design.md`

---

## File Structure

### Phase 2a: Aria-AMT Swap + Split Endpoints

```
apps/inference/
  amt_handler.py              # NEW: Aria-AMT endpoint handler
  handler.py                  # MODIFY: Remove ByteDance, MuQ-only
  models/transcription.py     # DELETE: ByteDance wrapper

apps/api/src/practice/
  session.rs                  # MODIFY: Audio buffer, parallel dispatch, dual-response processing

apps/api/
  wrangler.toml               # MODIFY: Add AMT endpoint URL binding
```

### Phase 2b: Multi-Signal Piece Identification

```
model/src/score_library/
  fingerprint.py              # NEW: N-gram index + rerank feature computation
  cli.py                      # MODIFY: Add fingerprint command

apps/api/src/practice/
  piece_identify.rs           # NEW: Multi-signal piece identification service
  session.rs                  # MODIFY: Accumulated notes, piece ID loop, lock-in
  score_context.rs            # MODIFY: Load fingerprint index + rerank features from R2
  mod.rs                      # MODIFY: Add piece_identify module

apps/api/migrations/
  0007_piece_requests_method.sql  # NEW: Add match_method column
```

**Key existing files (read-only references):**
- `apps/api/src/practice/score_follower.rs` -- `PerfNote` (lines 14-19), `PerfPedalEvent` (lines 23-26), `FollowerState`, `align_chunk()`
- `apps/api/src/practice/piece_match.rs` -- `match_piece()` (line 96), `MatchResult` (line 11), `CatalogPiece` (line 4)
- `apps/api/src/practice/score_context.rs` -- `resolve_piece()`, `load_catalog()` (line 85), `ScoreContext` (line 74)
- `apps/api/src/practice/analysis.rs` -- `analyze_chunk()`, Tier 1/2/3 degradation
- `apps/inference/models/transcription.py` -- ByteDance wrapper (to understand output format, lines 97-112)

---

## Task Dependency Chain

```
Phase 2a:
  Task 1 (AMT handler) -----+
  Task 2 (MuQ refactor) ----+---> Task 4 (DO orchestration) ---> Task 5 (validation)
  Task 3 (wrangler config) -+

Phase 2b:
  Task 6 (fingerprint precomp) --+---> Task 8 (DO integration) ---> Task 10 (E2E validation)
  Task 7 (piece_identify.rs) ----+
  Task 9 (D1 migration) ---------+
```

---

# Phase 2a: Aria-AMT Swap + Split Endpoints

## Task 1: Aria-AMT Endpoint Handler

**Files:**
- Create: `apps/inference/amt_handler.py`
- Create: `apps/inference/test_amt_handler.py`

- [ ] **Step 1: Write the Aria-AMT handler skeleton**

Create `apps/inference/amt_handler.py`. Key design:

- Class `EndpointHandler` with `__init__` (loads Aria-AMT model) and `__call__` (transcribes audio)
- Accepts two audio fields: `context_audio` (optional, previous chunk) + `chunk_audio` (required, current chunk)
- Both are WebM/Opus encoded bytes -- decode to 16kHz mono PCM via ffmpeg subprocess
- Concatenate context + chunk in PCM space, run Aria-AMT seq2seq inference
- Deduplicate: only return notes with onset >= context_duration (the context window is for boundary accuracy but those notes were already counted in the previous chunk)
- Adjust timestamps so returned notes are relative to current chunk start (onset -= context_duration)
- Output format matches `PerfNote`/`PerfPedalEvent` structs: `{"midi_notes": [{"pitch": int, "onset": float, "offset": float, "velocity": int}], "pedal_events": [{"time": float, "value": int}], "transcription_info": {...}}`

Note: The exact `aria-amt` Python API (imports, function names, MidiDict field structure) must be verified against the installed package during implementation. Read `model/.venv/lib/python3.12/site-packages/aria/inference/model_cuda.py` and `ariautils/midi.py` to confirm correct API calls.

- [ ] **Step 2: Write unit test for output format**

Create `apps/inference/test_amt_handler.py` with tests for:
- `test_output_format_has_required_fields`: Validate note fields (pitch: int, onset: float, offset: float, velocity: int) and pedal fields (time: float, value: int) match Rust struct expectations
- `test_context_deduplication`: Given 30s of notes with 15s context_duration, only notes after context_duration should be returned, with timestamps adjusted

These tests validate the format contract without requiring GPU or model loading.

- [ ] **Step 3: Run tests to verify format expectations**

Run: `cd apps/inference && python -m pytest test_amt_handler.py -v`
Expected: PASS

- [ ] **Step 4: Verify Aria-AMT API against installed package**

Read the actual `aria-amt` source to confirm import paths, model loading, `transcribe_audio()` call signature, and `MidiDict` field names. Adjust handler accordingly.

- [ ] **Step 5: Commit**

```bash
git add apps/inference/amt_handler.py apps/inference/test_amt_handler.py
git commit -m "feat: add Aria-AMT endpoint handler with context audio support"
```

---

## Task 2: Refactor MuQ Handler (Remove ByteDance)

**Files:**
- Modify: `apps/inference/handler.py` (lines 71-79, 150-172, 178-180, 192-193)
- Delete: `apps/inference/models/transcription.py`

- [ ] **Step 1: Remove ByteDance initialization from handler.py**

Remove the `TranscriptionModel` import and initialization from `__init__()` (lines 71-79).

- [ ] **Step 2: Remove ByteDance transcription call from `__call__()`**

Remove the AMT call block (lines ~150-172) and the AMT fields from the response dict (lines ~178-180, 192-193). Response should only contain `predictions`, `model_info`, `audio_duration_seconds`, `processing_time_ms`.

- [ ] **Step 3: Delete the ByteDance wrapper**

```bash
rm apps/inference/models/transcription.py
```

Verify no remaining imports: `grep -r "transcription" apps/inference/ --include="*.py" -l`

- [ ] **Step 4: Verify handler module imports cleanly**

```bash
cd apps/inference && python -c "import handler; print('OK')"
```

- [ ] **Step 5: Commit**

```bash
git add apps/inference/handler.py
git rm apps/inference/models/transcription.py
git commit -m "refactor: remove ByteDance AMT from MuQ handler, MuQ-only endpoint"
```

---

## Task 3: Wrangler Configuration

**Files:**
- Modify: `apps/api/wrangler.toml`

- [ ] **Step 1: Add AMT endpoint URL binding**

Add `HF_AMT_ENDPOINT` near existing `HF_INFERENCE_ENDPOINT` (line 23):

```toml
HF_AMT_ENDPOINT = ""  # Set after deploying Aria-AMT HF endpoint
```

- [ ] **Step 2: Commit**

```bash
git add apps/api/wrangler.toml
git commit -m "config: add HF_AMT_ENDPOINT binding for split inference"
```

---

## Task 4: DO Orchestration -- Parallel Dispatch + Audio Buffer

**Files:**
- Modify: `apps/api/src/practice/session.rs`

This is the largest task. Refactors how the DO calls inference and processes results.

- [ ] **Step 1: Add audio buffer field to SessionState**

Add to `SessionState` struct (after line 65):

```rust
/// Encoded WebM bytes from previous chunk (NOT persisted to durable storage).
/// Used to provide 30s context window for Aria-AMT.
previous_chunk_audio: Option<Vec<u8>>,
```

Initialize as `None` in Default impl.

- [ ] **Step 2: Add AMT endpoint URL to environment bindings**

Follow existing pattern for `HF_INFERENCE_ENDPOINT` access. Add `HF_AMT_ENDPOINT` with the same approach.

- [ ] **Step 3: Create helper functions for split endpoint calls**

Add two async functions:

- `call_muq_endpoint(env, chunk_audio) -> Result<MuqResponse>`: POST 15s audio to MuQ endpoint, parse predictions.
- `call_amt_endpoint(env, context_audio, chunk_audio) -> Result<AmtResponse>`: POST two audio fields to AMT endpoint, parse midi_notes + pedal_events.

Define `MuqResponse` and `AmtResponse` structs matching each endpoint's JSON output.

- [ ] **Step 4: Refactor chunk processing to parallel dispatch**

In `handle_chunk_ready`, replace single inference call with:

```rust
let context = self.state.borrow().previous_chunk_audio.clone();

let (muq_result, amt_result) = futures::join!(
    call_muq_endpoint(&env, &chunk_audio),
    call_amt_endpoint(&env, context.as_deref(), &chunk_audio),
);

// Store current chunk for next iteration
self.state.borrow_mut().previous_chunk_audio = Some(chunk_audio.to_vec());
```

Process MuQ result first (STOP classification, teaching moment candidate). Then process AMT result (score following, bar analysis, teaching moment enrichment). If AMT fails, proceed with Tier 3.

- [ ] **Step 5: Refactor response processing for split responses**

Split existing `process_inference_result` into separate MuQ processing (STOP, teaching moments) and AMT processing (score following, analysis) methods.

- [ ] **Step 6: Verify previous_chunk_audio is NOT persisted**

Check persistence logic (~line 140-146). Ensure `previous_chunk_audio` is excluded. Add comment explaining this is intentional (too large, too transient, DO eviction results in graceful degradation to single-chunk AMT).

- [ ] **Step 7: Test with wrangler dev**

```bash
cd apps/api && npx wrangler dev
```

Verify: MuQ called correctly, AMT graceful failure (endpoint not yet deployed), audio buffer stores/retrieves, Tier 3 degradation works.

- [ ] **Step 8: Commit**

```bash
git add apps/api/src/practice/session.rs
git commit -m "feat: parallel MuQ + AMT dispatch with audio buffer in DO"
```

---

## Task 5: Phase 2a Validation

**Files:** No new files. Deployment + testing.

- [ ] **Step 1: Deploy Aria-AMT endpoint to HuggingFace**

1. Create new HF Inference Endpoint
2. Upload `amt_handler.py` as custom handler + checkpoint
3. Select smallest GPU tier with int8 support
4. Test with curl
5. Update `wrangler.toml` with actual endpoint URL

- [ ] **Step 2: Run comparison -- ByteDance vs Aria-AMT**

Process same audio through both. Compare: note count, onset accuracy, offset accuracy, pedal detection. Document results.

- [ ] **Step 3: Verify score follower with Aria-AMT output**

Run DTW score follower on Aria-AMT notes for a known piece. Verify bar alignment matches or exceeds ByteDance quality.

- [ ] **Step 4: Commit final endpoint URL**

```bash
git add apps/api/wrangler.toml
git commit -m "config: set Aria-AMT endpoint URL after deployment"
```

---

# Phase 2b: Multi-Signal Piece Identification

## Task 6: Fingerprint Precomputation

**Files:**
- Create: `model/src/score_library/fingerprint.py`
- Modify: `model/src/score_library/cli.py`
- Create: `model/tests/score_library/test_fingerprint.py`

- [ ] **Step 1: Write failing tests for N-gram extraction and feature computation**

Create `model/tests/score_library/test_fingerprint.py` with:
- `test_extract_pitch_trigrams_basic`: 5 pitches -> 3 trigrams
- `test_extract_pitch_trigrams_too_short`: <3 pitches -> empty list
- `test_compute_rerank_features_shape`: Output is exactly 128 floats
- `test_compute_rerank_features_pitch_class_histogram`: All-C notes -> features[0] == 1.0, features[1:12] == 0.0

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd model && python -m pytest tests/score_library/test_fingerprint.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement fingerprint.py**

Create `model/src/score_library/fingerprint.py` with:

- `extract_pitch_trigrams(pitches: list[int]) -> list[tuple]`: Extract consecutive pitch trigrams
- `compute_rerank_features(notes: list[dict]) -> list[float]`: 128-dim feature vector:
  - [0:12] pitch class histogram (normalized)
  - [12:37] interval histogram (-12 to +12 semitones, normalized)
  - [37:41] pitch range (min, max, mean, std, scaled to 0-1)
  - [41:66] IOI histogram (25 bins at 50ms, normalized)
  - [66:78] velocity histogram (12 bins, normalized)
  - [78:82] velocity stats (min, max, mean, std, scaled)
  - [82:128] reserved / zero-padded
- `build_ngram_index(scores_dir: Path) -> dict`: Inverted index: trigram_key -> [(piece_id, bar_number), ...]
- `build_rerank_features(scores_dir: Path) -> dict[str, list[float]]`: Compute 128-dim features for all 242 scores

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd model && python -m pytest tests/score_library/test_fingerprint.py -v`
Expected: PASS

- [ ] **Step 5: Add `fingerprint` command to CLI**

In `model/src/score_library/cli.py`, add subcommand that calls `build_ngram_index()` and `build_rerank_features()`, writes to `data/fingerprints/ngram_index.json` and `data/fingerprints/rerank_features.json`, prints file sizes.

- [ ] **Step 6: Run fingerprinting on actual score library**

```bash
cd model && python -m score_library.cli fingerprint --scores-dir data/scores --output-dir data/fingerprints
```

Note actual file sizes. Validate N-gram index is < 20MB.

- [ ] **Step 7: Upload fingerprints to R2**

Upload to `fingerprints/v1/ngram_index.json` and `fingerprints/v1/rerank_features.json`.

- [ ] **Step 8: Commit**

```bash
git add model/src/score_library/fingerprint.py model/src/score_library/cli.py \
        model/tests/score_library/test_fingerprint.py model/data/fingerprints/
git commit -m "feat: score fingerprinting (N-gram index + rerank features)"
```

---

## Task 7: Piece Identification Service (Rust)

**Files:**
- Create: `apps/api/src/practice/piece_identify.rs`
- Modify: `apps/api/src/practice/mod.rs`

- [ ] **Step 1: Define data structures**

Create `apps/api/src/practice/piece_identify.rs` with:

- `NgramIndex`: Deserialized from JSON. Map from trigram key ("p1,p2,p3") to Vec<(piece_id, bar_number)>.
- `RerankFeatures`: Deserialized from JSON. Map from piece_id to Vec<f64> (128-dim).
- `PieceIdentification`: Result struct with piece_id, confidence, method.
- Constants: `LOCK_THRESHOLD = 0.3`, `MAX_CANDIDATES = 10`.

- [ ] **Step 2: Implement N-gram recall (Stage 1)**

`ngram_recall(notes, index) -> Vec<(piece_id, hit_count)>`: Extract pitch trigrams from accumulated notes, look up in inverted index, count hits per piece, return top-10 by hit count.

- [ ] **Step 3: Implement rerank (Stage 2)**

- `compute_rerank_features(notes: &[PerfNote]) -> Vec<f64>`: Mirror the Python 128-dim feature computation in Rust. Must produce identical features for the same input.
- `cosine_similarity(a, b) -> f64`: Standard dot-product / magnitude cosine.
- `rerank_candidates(notes, candidates, features) -> Vec<(piece_id, similarity)>`: Compute performance features, cosine against pre-computed score features, return top-2.

- [ ] **Step 4: Implement top-level identify_piece function**

`identify_piece(notes, index, features) -> Option<PieceIdentification>`: Run Stage 1 + Stage 2. Return top candidate. DTW confirmation (Stage 3) happens in session.rs because it requires async R2 access.

- [ ] **Step 5: Register module in mod.rs**

Add `pub mod piece_identify;` to `apps/api/src/practice/mod.rs`.

- [ ] **Step 6: Commit**

```bash
git add apps/api/src/practice/piece_identify.rs apps/api/src/practice/mod.rs
git commit -m "feat: multi-signal piece identification (N-gram + rerank + DTW)"
```

---

## Task 8: DO Integration -- Piece Identification Loop

**Files:**
- Modify: `apps/api/src/practice/session.rs`
- Modify: `apps/api/src/practice/score_context.rs`

- [ ] **Step 1: Add piece identification state to SessionState**

Add: `accumulated_notes: Vec<PerfNote>`, `piece_identification: Option<PieceIdentification>`, `piece_locked: bool`, `ngram_index: Option<Arc<NgramIndex>>`, `rerank_features: Option<Arc<RerankFeatures>>`.

- [ ] **Step 2: Add fingerprint loading functions to score_context.rs**

Add `load_ngram_index(env) -> Result<NgramIndex>` and `load_rerank_features(env) -> Result<RerankFeatures>` that fetch from R2 at `fingerprints/v1/`.

- [ ] **Step 3: Add identification loop to AMT response processing**

After appending AMT notes to `accumulated_notes`:
1. If `piece_locked`: skip identification, go to score following
2. If `accumulated_notes.len() > 200`: set `piece_locked = true`, log to `piece_requests`
3. Lazy-load N-gram index + rerank features from R2 (cache in session state)
4. Call `identify_piece()` with accumulated notes
5. If candidate returned: load score from R2, run `align_chunk()` for DTW confirmation
6. If DTW confirms (cost < threshold): lock in piece, load full ScoreContext
7. If no match: continue, try again next chunk

- [ ] **Step 4: Preserve text fallback path**

When student names a piece via chat, set `piece_identification` and `piece_locked` through the same mechanism. Both automatic (MIDI-based) and explicit (text-based) paths converge on the same lock-in state.

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/practice/session.rs apps/api/src/practice/score_context.rs
git commit -m "feat: piece identification loop in DO with lazy fingerprint loading"
```

---

## Task 9: D1 Migration

**Files:**
- Create: `apps/api/migrations/0007_piece_requests_method.sql`
- Modify: `apps/api/src/practice/score_context.rs`

- [ ] **Step 1: Create migration**

```sql
ALTER TABLE piece_requests ADD COLUMN match_method TEXT;
```

- [ ] **Step 2: Apply migration locally**

```bash
cd apps/api && npx wrangler d1 migrations apply DB --local
```

- [ ] **Step 3: Update log_piece_request to include match_method**

Add `match_method: Option<&str>` parameter to `log_piece_request()` in `score_context.rs`. Include in INSERT statement.

- [ ] **Step 4: Commit**

```bash
git add apps/api/migrations/0007_piece_requests_method.sql apps/api/src/practice/score_context.rs
git commit -m "feat: add match_method to piece_requests table"
```

---

## Task 10: End-to-End Validation

**Files:** No new files.

- [ ] **Step 1: Known piece identification test**

Play a known piece (e.g., Chopin Nocturne Op. 9 No. 2). Verify identification within 3 chunks (~45s), bar alignment activates, Tier 1 observations reference bar numbers.

- [ ] **Step 2: Wrong notes robustness test**

Play with ~10% deliberate mistakes. Verify identification still converges (may take more chunks).

- [ ] **Step 3: Unknown piece negative test**

Play something not in the library. Verify: no false positive, cutoff fires at ~200 notes, teacher asks "What are you playing?", Tier 2 observations delivered throughout.

- [ ] **Step 4: Partial match test**

Start from the middle of a known piece. Verify N-gram + subsequence DTW still matches (free start position in DTW).

- [ ] **Step 5: Deploy to production**

```bash
cd apps/api && npx wrangler d1 migrations apply DB --remote
cd apps/api && npx wrangler deploy
```

- [ ] **Step 6: Commit any final fixes**

```bash
git add -A
git commit -m "fix: E2E validation fixes for zero-config piece identification"
```
