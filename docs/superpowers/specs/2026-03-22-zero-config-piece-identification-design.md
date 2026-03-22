# Zero-Config Piece Identification with Aria-AMT

Replace ByteDance AMT with Aria-AMT (Whisper-based, 49M params, SOTA accuracy), split inference into two independent endpoints, and add multi-signal MIDI-based piece identification so students never need to name what they're playing.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| AMT engine | Aria-AMT (EleutherAI) | MAESTRO F1 0.86 vs ByteDance 0.38. Offset F1 0.91 vs 0.56 (critical for pedaling). Robust to diverse recording environments via augmentation training. Same ecosystem as Aria symbolic encoder. |
| Endpoint architecture | Split: MuQ (15s) + Aria-AMT (30s) | Independent scaling, independent deployment, parallel execution, failure isolation. Each model gets optimal input size. |
| Audio buffering | DO ring buffer (previous chunk) | HF endpoints are stateless. DO already holds session state. Client stays simple (15s chunks unchanged). |
| Piece matching strategy | Multi-signal: N-gram recall -> embedding rerank -> DTW confirm+align | Mirrors RAG retrieval pattern. N-gram for recall (fast, structural), embedding for precision (semantic), DTW for confirmation and bar alignment. |
| Embedding inference | Pre-computed vectors in R2. No live GPU cost. | 242 scores x 512-dim = ~500KB. Cosine similarity is trivial in the worker. |
| Temporal accumulation | Accumulated note buffer in DO | Not per-chunk. DO appends AMT notes across chunks. Confidence rises naturally as buffer grows. No weighting logic needed. |
| Give-up policy | Hard cutoff at ~200 notes (~60-90s) | Student can name piece via chat (existing text matcher as fallback). Avoids wasting cycles on pieces not in library. |
| Piece ID vs score following | Independent concerns | Piece ID locks in the "what." DTW handles the "where" and degrades independently to Tier 2 when alignment fails (drilling, jumping). |
| First chunk behavior | MuQ fires immediately, AMT skips or runs degraded | First observation can fire on Tier 3 (scores only). AMT catches up on chunk 2 with full 30s window. |

## Phasing

### Phase 2a: Aria-AMT Swap + Split Endpoints (~1 week)

Better transcription, parallel execution, cleaner architecture. No change to piece identification flow.

### Phase 2b: Multi-Signal Piece Identification (~1-1.5 weeks)

True zero-config. Student plays, system identifies the piece from the music itself.

Phase 2a is independently shippable. Phase 2b can slip without blocking beta (existing "what are you playing?" conversational fallback covers the gap).

---

## Architecture

### Current State

```
Client --15s audio--> CF Worker --> HF Endpoint (MuQ + ByteDance AMT)
                                        | sequential
                                        |-- MuQ scoring (~1s)
                                        |-- ByteDance AMT (~0.8s)
                                        |
                                    Response: scores + midi_notes
```

### Target State

```
Client --15s audio--> DO (session brain)
                        |
                        | stores audio in ring buffer
                        |
                        |--15s--> HF Endpoint: MuQ
                        |         LoRA rank-32, 15s chunks
                        |         Returns: 6-dim scores
                        |
                        |--30s--> HF Endpoint: Aria-AMT
                                  Whisper-based, 49M params
                                  30s with overlap context
                                  Returns: MIDI notes + pedal
                        |
                        v
                  DO processes responses as they arrive
                  (MuQ first --> STOP classification)
                  (AMT second --> piece ID + score following + analysis)
```

### Audio Ring Buffer

The DO keeps raw audio bytes from the previous chunk (~1.4MB at 24kHz mono). When chunk N arrives:

1. Dispatch chunk N (15s) to MuQ immediately
2. Concatenate chunk N-1 + chunk N (30s) and dispatch to Aria-AMT
3. On chunk 1: skip AMT or send 15s degraded (benchmark during implementation)

### Response Processing Order

1. MuQ result arrives (~1s) -> STOP classification -> queue teaching moment candidate
2. AMT result arrives (~1-2s) -> append notes to accumulated buffer -> piece identification (if not locked) -> score following -> bar-aligned analysis -> enrich teaching moment with bar context
3. Apply observation throttle -> if ready, fire subagent -> teacher -> WebSocket push

### Failure Isolation

- AMT endpoint down: Tier 3 (MuQ scores only). Log alert. Session continues.
- MuQ endpoint down: Session cannot function. Surface error to user.
- Independent failure domains. No cascading.

---

## Phase 2a: Aria-AMT Swap

### Aria-AMT Endpoint

New HF inference endpoint handler (`apps/inference/amt_handler.py`):

- Model: `aria-amt` piano-medium-double checkpoint (49M params)
- Input: raw audio bytes (24kHz mono, up to 30s)
- Processing: Aria-AMT resamples to 16kHz, log-mel spectrogram, seq2seq decodes to MIDI tokens
- Native output: `(on: pitch), (onset: ms), (velocity: v), (off: pitch), (onset: ms)` tokens
- Handler decodes tokens into note-list format matching existing Rust structs

Output format (compatible with `PerfNote`/`PerfPedalEvent`):

```json
{
  "midi_notes": [
    {"pitch": 60, "onset": 0.12, "offset": 0.45, "velocity": 78}
  ],
  "pedal_events": [
    {"time": 0.10, "value": 127}
  ],
  "transcription_info": {
    "note_count": 145,
    "pitch_range": [36, 96],
    "pedal_event_count": 8,
    "transcription_time_ms": 420
  }
}
```

Deployment: separate HF endpoint on smallest GPU that fits 49M params with int8 quantization.

### MuQ Endpoint Refactor

Simplify existing `apps/inference/handler.py`:

- Remove all ByteDance/TranscriptionModel code
- Remove `models/transcription.py`
- Remove `piano-transcription-inference` dependency
- Response returns only `predictions` and `model_info` (no `midi_notes`, `pedal_events`, `transcription_info`)

### DO Orchestration Changes

Session state additions:

```rust
previous_chunk_audio: Option<Vec<u8>>,  // raw audio from last chunk
```

Parallel dispatch on chunk arrival:

```rust
let (muq_result, amt_result) = tokio::join!(
    call_muq_endpoint(&chunk_audio),
    call_amt_endpoint(&combined_audio),
);
```

Fallback: if AMT request fails or times out (3s), proceed with MuQ scores only (Tier 3).

### Validation

Before shipping Phase 2a:

- Record 5-10 practice sessions with ByteDance AMT (current production)
- Re-process same audio through Aria-AMT
- Compare: note count, onset accuracy, offset accuracy, pedal event detection
- Verify `score_follower.rs` DTW works equally well or better on Aria-AMT output
- Verify bar-aligned analysis quality is maintained or improved

---

## Phase 2b: Multi-Signal Piece Identification

### Offline Precomputation

One-time batch jobs, re-run when score library changes.

**N-gram fingerprint index:**
- For each of 242 score MIDIs, extract all pitch trigrams (3 consecutive pitches)
- Build inverted index: `trigram -> [(piece_id, position_in_score), ...]`
- Store in R2 at `fingerprints/v1/ngram_index.json` (<5MB)
- Load into DO memory on first use, cache for session lifetime

**Aria-Embedding vectors:**
- Run each score MIDI through existing `aria_embeddings.py` (TransformerEMB, 512-dim)
- 242 vectors x 512 dims = ~500KB
- Store in R2 at `fingerprints/v1/embeddings.bin`
- Load into DO memory alongside N-gram index

New CLI stage in `model/src/score_library/cli.py`: `fingerprint` generates both artifacts and uploads to R2.

### Piece Identification Pipeline

Three-stage retrieval, runs in the DO on each AMT response (until piece is locked):

```
Accumulated MIDI notes
  |
  v
[Stage 1: N-gram recall] -- pitch trigrams vs precomputed index
  |                          histogram of offsets for position-aware matching
  |                          ~1ms, returns top-10 candidates
  v
[Stage 2: Embedding rerank] -- lightweight proxy (pitch-class + interval histograms)
  |                            cosine similarity vs pre-computed vectors
  |                            ~5ms, reranks to top-2
  v
[Stage 3: DTW confirm+align] -- subsequence DTW on top-1 (existing score_follower.rs)
  |                              ~50ms, returns bar alignment + cost
  v
Match? --yes--> ScoreContext activated (Tier 1 analysis)
  |
  no
  v
Try again next chunk (or give up at 200 notes)
```

**Stage 2 detail:** For beta, use a lightweight proxy rather than live Aria-Embedding inference (no GPU in worker). Compute pitch-class histogram + interval histogram from accumulated notes (~128-dim feature vector). Compare against pre-computed feature vectors for top-10 candidates. If retrieval accuracy is insufficient, add live embedding extraction to the AMT endpoint later (AMT and Embedding share the same tokenizer, ~50ms additional latency).

### DO Integration

Session state additions:

```rust
accumulated_notes: Vec<PerfNote>,
piece_match: Option<PieceMatch>,
piece_locked: bool,
identification_attempts: u32,
ngram_index: Option<Arc<NgramIndex>>,
score_embeddings: Option<Arc<ScoreEmbeddings>>,
```

On each AMT response:

1. Append new notes to `accumulated_notes`
2. If `piece_locked`: skip to score following
3. If `accumulated_notes.len() > 200`: give up, set `piece_locked = true`, log to `piece_requests`
4. Run `identify_piece(accumulated_notes, ...)`
5. If confident match: set `piece_match`, `piece_locked = true`, load `ScoreContext` from R2, initialize score follower
6. If no match: continue, try again next chunk

**Text fallback preserved:** If student names a piece via chat, existing `piece_match.rs` text matcher sets `piece_match` and `piece_locked` through the same path. Both identification methods (automatic MIDI-based, explicit text-based) converge on the same lock-in mechanism.

### Graceful Degradation Tiers

| State | Tier | Teacher context |
|-------|------|-----------------|
| No AMT yet (chunk 1, MuQ only) | 3 | Scores only |
| AMT arrived, no piece match | 2 | Scores + absolute MIDI analysis |
| Piece matched, DTW aligned | 1 | Scores + bar-aligned analysis + score context + reference comparison |
| Piece matched, DTW struggling | 1->2 | Scores + piece context, no bar numbers |
| Cutoff reached, no match | 2 | Scores + absolute MIDI. Teacher asks "What are you playing?" once. |

---

## Error Handling & Edge Cases

### Edge Cases

**Student switches pieces mid-session:** Piece ID stays locked. Score follower DTW cost spikes, degrades to Tier 2. Acceptable for beta. Post-beta optimization: detect piece switches via sustained DTW failure + N-gram profile divergence.

**Student drills a passage repeatedly:** DTW re-anchoring handles repetition. May degrade to Tier 2 temporarily. Piece ID unaffected (independent concern).

**Very short session (< 15s):** MuQ fires, Tier 3 observation. No AMT, no piece ID attempt.

**Piece not in library:** N-gram recall returns no confident candidates. After 200 notes, cutoff fires. Teacher asks once via chat. Tier 2 for remainder.

**Similar pieces in library:** N-gram may return multiple candidates. DTW confirmation disambiguates via actual note sequence alignment.

**Audio quality too poor for AMT:** Few/no notes returned. Buffer stays small, never reaches confidence. Stays on Tier 2/3.

### Failure Recovery

| Failure | Impact | Recovery |
|---------|--------|----------|
| Aria-AMT endpoint down | No MIDI, no piece ID, no bar alignment | Tier 3 (MuQ scores only). Log alert. |
| Aria-AMT timeout (>3s) | Same as above for this chunk | Retry next chunk. DO doesn't block on AMT. |
| MuQ endpoint down | No scores, no STOP | Session cannot deliver observations. Surface error. |
| N-gram index failed to load | No automatic piece ID | Fall back to text-based matching. Log error. |
| Score embeddings failed to load | No embedding rerank | N-gram -> DTW directly (skip rerank). |
| R2 score JSON unavailable | Piece matched, can't load score | Tier 2 (know piece name, no bar analysis). |
| DTW confirmation fails all candidates | N-gram/embedding disagreed with DTW | No lock-in. Try again next chunk with more notes. |

### Observability

- **Metrics:** piece identification rate, time-to-match (which chunk locked in), AMT latency (p50/p95), false match rate
- **Logging:** each identification attempt with candidate list, scores, outcome
- **piece_requests table:** extended with `match_method` field (`"text"` vs `"ngram+dtw"`) and confidence

---

## Files Changed

### Phase 2a

| File | Change |
|------|--------|
| `apps/inference/amt_handler.py` | **New.** Aria-AMT endpoint handler. |
| `apps/inference/handler.py` | Remove ByteDance AMT code. MuQ-only. |
| `apps/inference/models/transcription.py` | **Delete.** |
| `apps/api/src/practice/session.rs` | Add `previous_chunk_audio`, parallel dispatch, dual-response processing. |
| `apps/api/wrangler.toml` | Add Aria-AMT endpoint URL binding. |

### Phase 2b

| File | Change |
|------|--------|
| `model/src/score_library/cli.py` | Add `fingerprint` stage (N-gram index + embedding vectors). |
| `model/src/score_library/fingerprint.py` | **New.** N-gram extraction + Aria-Embedding batch computation. |
| `apps/api/src/practice/piece_identify.rs` | **New.** Multi-signal piece identification service. |
| `apps/api/src/practice/session.rs` | Add accumulated_notes, piece_match state, identification loop. |
| `apps/api/src/practice/piece_match.rs` | Preserved as text fallback. No changes. |
| `apps/api/src/practice/score_context.rs` | Add fingerprint index + embedding loading from R2. |

---

## Key References

- [Aria-AMT paper](https://www.alexander-spangher.com/papers/aria_amt.pdf): Whisper-based architecture, MAESTRO/MAPS benchmarks, bootstrapping, augmentation
- [Aria paper (ISMIR 2025)](https://arxiv.org/abs/2506.23869): SimCLR contrastive embedding training, 512-dim, MIR classification benchmarks
- [Aria-MIDI dataset (ICLR 2025)](https://arxiv.org/abs/2504.15071): 1.2M transcriptions, data pipeline, quality control via DTW
- [Dynamic N-gram fingerprinting (ISMIR)](https://transactions.ismir.net/articles/10.5334/tismir.70): MRR 0.853 on 29K pieces, sub-second runtime
- [EleutherAI/aria-amt GitHub](https://github.com/EleutherAI/aria-amt): Model weights, CLI, inference code
