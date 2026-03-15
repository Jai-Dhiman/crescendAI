# Phase 1b: Cloud AMT Service

> **Status:** DESIGNED (not implemented)
> **Depends on:** Nothing (independent of 1a, but same HF endpoint)
> **Enables:** 1c (Score Following -- needs performance MIDI to align against score MIDI)
> **Pipeline phase:** Phase 1 -- Score Infrastructure

---

## Purpose

Add Automatic Music Transcription (AMT) to the existing HF inference endpoint so that every audio chunk returns both MuQ quality scores and a parsed performance MIDI note list. This is the bridge between audio and symbolic: the AMT output feeds score following (1c), which aligns the student's performance to bar numbers in the score.

Without AMT, the system can evaluate *how well* the student plays but cannot say *where* in the score they are playing. With AMT, the teacher can say "in bars 12-16, the crescendo is timid" instead of "dynamics score 0.35."

## Scope

**In scope:**

- Add ByteDance `piano-transcription-inference` to the existing HF inference endpoint
- Both MuQ and AMT run on every request, sequential execution
- Return parsed note list (`[{pitch, onset, offset, velocity}]`) alongside MuQ scores
- Graceful degradation: AMT failure returns scores with `midi_notes: null`
- Update Dockerfile and requirements for the new dependency

**Out of scope:**

- API worker changes (worker receives new response fields but needs no code changes until 1c)
- Score following / DTW alignment (Phase 1c)
- Streaming/real-time AMT (we process 15s chunks, same as MuQ)
- AMT model fine-tuning or alternatives (ByteDance is validated and sufficient)

## Validation Status

ByteDance AMT has already been validated in Layer 1 experiments:

- **MAESTRO (studio audio):** 0% pairwise accuracy drop vs ground-truth MIDI (50 recordings, 107 pairs)
- **YouTube (mediocre audio):** 79.9% A1-vs-S2 agreement (50 recordings, 1,225 pairs, all dims > 72%)
- **Validation code:** `model/scripts/validate_youtube_amt.py`
- **Library:** `piano-transcription-inference` (ByteDance, optional dependency in pyproject.toml)

## Response Format

### Current response (MuQ only)

```json
{
  "predictions": {"dynamics": 0.65, "timing": 0.72, "pedaling": 0.58, "articulation": 0.61, "phrasing": 0.55, "interpretation": 0.68},
  "model_info": {"name": "A1-Max", "type": "muq_lora", "pairwise": 0.808, "architecture": "MuQ-L9-12-LoRA-r32", "ensemble_folds": 4},
  "audio_duration_seconds": 15.0,
  "processing_time_ms": 450
}
```

### New response (MuQ + AMT)

```json
{
  "predictions": {"dynamics": 0.65, "timing": 0.72, "pedaling": 0.58, "articulation": 0.61, "phrasing": 0.55, "interpretation": 0.68},
  "midi_notes": [
    {"pitch": 60, "onset": 0.12, "offset": 0.45, "velocity": 78},
    {"pitch": 64, "onset": 0.13, "offset": 0.44, "velocity": 72},
    {"pitch": 67, "onset": 0.50, "offset": 0.82, "velocity": 85}
  ],
  "transcription_info": {
    "note_count": 47,
    "pitch_range": [33, 96],
    "transcription_time_ms": 380
  },
  "model_info": {"name": "A1-Max", "type": "muq_lora", "pairwise": 0.808, "architecture": "MuQ-L9-12-LoRA-r32", "ensemble_folds": 4},
  "audio_duration_seconds": 15.0,
  "processing_time_ms": 830
}
```

### On AMT failure

```json
{
  "predictions": {"dynamics": 0.65, "timing": 0.72, "pedaling": 0.58, "articulation": 0.61, "phrasing": 0.55, "interpretation": 0.68},
  "midi_notes": null,
  "amt_error": "Transcription failed: no piano detected",
  "transcription_info": null,
  "model_info": {"name": "A1-Max", ...},
  "audio_duration_seconds": 15.0,
  "processing_time_ms": 520
}
```

Design choices:

- **`midi_notes` is always present.** List of notes on success, `null` on failure. Distinguishes from `[]` (silence -- no notes detected in audio).
- **Notes sorted by `onset`.** Each note: `pitch` (MIDI 0-127), `onset`/`offset` (seconds from chunk start), `velocity` (0-127).
- **`transcription_info` provides summary stats.** `note_count` and `pitch_range` for quick validation without scanning notes.
- **`processing_time_ms` reflects total.** Individual timing in `transcription_info.transcription_time_ms`.
- **Existing consumers unaffected.** The new fields are additive -- anything reading only `predictions` works unchanged.

## Implementation Architecture

```
STARTUP (container init, ~10s)
+----------------------------------------------+
|  Load MuQ + A1-Max heads (existing, ~5s)     |
|  Load ByteDance PianoTranscription (NEW, ~5s) |
|  Both on CUDA, ~1.5GB total VRAM             |
+----------------------------------------------+

REQUEST (per 15s audio chunk, <2s total)
+----------------------------------------------+
|  1. Parse request + load audio    (existing)  |
|  2. MuQ -> A1-Max 6-dim scores    (~0.4s)     |
|  3. ByteDance AMT -> temp MIDI    (~0.7s) NEW |
|  4. Parse MIDI -> note list       (<0.1s) NEW |
|  5. Build combined response                   |
+----------------------------------------------+
```

### Execution model

Sequential: MuQ first, then AMT. CUDA operations on a single GPU serialize at the hardware level, so threading would not produce real parallelism. Sequential is simpler and total latency (~1.1s) fits within the 2s budget.

### Error isolation

AMT runs in a try/except block after MuQ scoring completes. If AMT raises any exception:

1. MuQ scores are preserved (already computed)
2. `midi_notes` is set to `null`
3. `amt_error` is set to the error message
4. The response returns 200 (not 500) -- this is graceful degradation, not a failure

### Module structure

**New file: `apps/inference/models/transcription.py`**

Isolates all AMT logic from the handler:

```
class TranscriptionError(Exception): ...

class TranscriptionModel:
    def __init__(self, device: str):
        # Load ByteDance PianoTranscription model
        # ~200MB weights, downloads on first init

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> list[dict]:
        # 1. Write audio to temp WAV (ByteDance API requires file path)
        # 2. Run PianoTranscription.transcribe(wav_path, midi_path)
        # 3. Parse MIDI with pretty_midi
        # 4. Extract notes: [{pitch, onset, offset, velocity}]
        # 5. Clean up temp files
        # 6. Return sorted note list
        # Raises TranscriptionError on any failure
```

**Modified: `apps/inference/handler.py`**

Minimal changes:

- Import `TranscriptionModel` and `TranscriptionError`
- In `__init__`: initialize `self._transcription_model = TranscriptionModel(device="cuda")`
- In `__call__`: after MuQ scoring, call `self._transcription_model.transcribe(audio, 24000)` wrapped in try/except
- Build response with new fields

**Modified: `apps/inference/requirements.txt`**

Add:
```
piano-transcription-inference
pretty-midi>=0.2.10
```

**Modified: `apps/inference/Dockerfile`**

Add dependency installation. Optionally pre-download ByteDance model weights at build time for faster cold starts:
```dockerfile
RUN python -c "from piano_transcription_inference import PianoTranscription; PianoTranscription(device='cpu')"
```

## Testing

### Unit tests

- **Note parsing:** Given a known MIDI file, verify `TranscriptionModel.transcribe()` returns correct note list (pitch, onset, offset, velocity values match expected)
- **Empty audio:** Silent audio -> `midi_notes: []` (empty list, not null)
- **AMT failure:** Force transcription error -> response has `midi_notes: null`, `amt_error` set, `predictions` still present
- **Response schema:** Verify all new fields present and correctly typed

### Integration test

- Send real 15s piano audio to local endpoint
- Verify response contains both `predictions` (6 non-zero scores) and `midi_notes` (non-empty list)
- Verify `processing_time_ms` < 2000

### Latency validation

- Benchmark on the HF endpoint GPU (T4 or A10G)
- MuQ alone: ~0.4s baseline
- MuQ + AMT: should be < 1.5s for 15s chunk
- If total exceeds 2s, investigate AMT model optimization (lower resolution, batch processing)

## Performance Budget

| Component | Latency | VRAM |
|-----------|---------|------|
| MuQ + A1-Max | ~400ms | ~1.2GB |
| ByteDance AMT | ~700ms | ~300MB |
| MIDI parsing | <100ms | negligible |
| **Total** | **~1.2s** | **~1.5GB** |

Current HF endpoint runs on a GPU with at least 16GB VRAM. Adding ~300MB for AMT is well within budget.

## Deployment

1. Update `requirements.txt` and `Dockerfile`
2. Test locally with sample audio
3. Push to HF model repo (triggers endpoint rebuild)
4. Verify endpoint returns both `predictions` and `midi_notes`
5. Monitor latency for first 24 hours

Rollback: if AMT causes issues, the graceful degradation means scores continue working. To fully revert, redeploy with the previous Dockerfile.

## Future Enhancements

| Enhancement | When | Notes |
|-------------|------|-------|
| AMT model caching | If cold start > 15s | Pre-download weights in Dockerfile (described above) |
| Note confidence scores | When score following needs it | ByteDance outputs per-note probabilities, currently discarded |
| Pedal event transcription | When pedal analysis (1d) needs it | ByteDance also transcribes pedal events, can add to response |
| Alternative AMT models | If ByteDance accuracy insufficient | Kong et al. (2021) or newer transformer-based transcribers |
