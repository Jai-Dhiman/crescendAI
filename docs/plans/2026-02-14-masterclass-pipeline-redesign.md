# Masterclass Pipeline Redesign: Transcribe-First Architecture

## Problem

The current pipeline fails on 3 of 5 test videos. Three distinct failure modes:

1. **Playing detection fails entirely** (Zimerman, Perahia): Audio feature thresholds (spectral centroid >1500Hz, harmonic ratio >0.5) don't recognize piano in these recordings. Zero playing segments detected means zero teaching moments.
2. **Stopping point detection too conservative** (Arie Vardi): 20 playing segments exist but only 2 transitions captured as teaching moments in a 65-minute masterclass.
3. **Whisper hallucinates its initial prompt** ("Teacher gives feedback on student performance") during non-speech sections. The LLM then fabricates analysis from hallucinated text with confidence 0.7.

Root cause: the pipeline chains fragile audio classification (playing/talking) into transition detection into transcription. Each failure compounds.

## Solution

Invert the pipeline. Transcribe the full audio first, then identify teaching moments from the text.

## New Pipeline

```
Discover -> Download -> Transcribe (API) -> Identify (LLM) -> Export
```

### Stage 1: Discover (unchanged)

Import curated video URLs from `sources.yaml`. Same metadata extraction via yt-dlp.

### Stage 2: Download (unchanged)

Extract audio via yt-dlp. 16kHz mono WAV output.

### Stage 3: Transcribe (rewritten)

**Default: OpenAI Whisper API**

- Endpoint: `POST https://api.openai.com/v1/audio/transcriptions`
- Model: `whisper-1`
- Response format: `verbose_json` with `timestamp_granularities: ["word", "segment"]`
- Language: `en`

**Handling the 25MB file size limit:**

Chunk audio into ~10-minute segments with 30-second overlap (same pattern as current pipeline). Upload each chunk. Merge transcripts using existing deduplication logic (keep segment further from chunk boundary).

**Fallback: Local Whisper (`--local` flag)**

Retain current local Whisper implementation behind a CLI flag. Useful for offline use or cost savings. Default is API.

**Output:** `data/transcripts/{video_id}.json` -- same schema as today but with richer word-level timestamps from the API.

**Cost:** ~$0.006/min. 100 videos at 45 min avg = ~$27.

### Stage 4: Identify (new, replaces Segment + Extract)

Two-pass LLM approach:

**Pass 1 -- Moment Detection:**

Send the full transcript (with timestamps) to the LLM. Task: identify every point where the teacher stops the student to give feedback.

Input: full transcript text with timestamps.
Output: JSON array of `{ timestamp, brief_description }` pairs.

This is a simpler task than full extraction -- the LLM only needs to find the boundaries, not analyze the content. Lower chance of missing moments or hallucinating.

**Pass 2 -- Moment Extraction:**

For each moment identified in Pass 1, send a ~2-minute context window (transcript before and after the stop timestamp) to the LLM. Task: extract structured `TeachingMoment` JSON.

Output: same `TeachingMoment` schema as today.

**Playing boundary estimation:**

Without audio segmentation, estimate `playing_before_start` and `playing_before_end` from transcript gaps. The last silence gap (no transcript words) before the feedback starts likely corresponds to student playing. Use word-level timestamps to find these gaps.

**Edge cases handled by text-based approach:**
- Teacher demonstrations (plays piano mid-feedback): transcript shows gaps during playing, LLM groups them as one moment
- Conversational masterclasses (Zander style): LLM distinguishes general discussion from specific performance feedback
- Teacher talking over playing: still captured in transcript, LLM can identify the feedback
- Multiple students: LLM detects transitions from context clues

### Stage 5: Export (unchanged)

Same `all_moments.jsonl` consolidated output.

## Implementation Scope

### Rewritten:
- `transcribe.rs` -- add OpenAI Whisper API client, keep local Whisper behind `--local` flag
- `segment.rs` + `extract.rs` -- merged into new `identify.rs` with two-pass LLM logic
- `llm_client.rs` -- extend with OpenAI API auth headers (API key)
- `pipeline.rs` / `main.rs` -- update stage definitions (segment + extract become identify)
- `config.rs` -- add API key config, model selection

### Unchanged:
- `discovery.rs`
- `download.rs`
- Output schema (TeachingMoment)

### Kept but optional:
- `audio_features.rs` -- no longer on critical path, available for future enrichment
- `segment.rs` -- audio segmentation code preserved but not used by default pipeline

### Added:
- Whisper API client (HTTP multipart upload)
- Transcript-based playing boundary estimation
- Two-pass LLM prompts (moment detection + moment extraction)

### Pipeline state:
- Stages: `discover`, `download`, `transcribe`, `identify`, `export`
- Same `pipeline_state.jsonl` approach

## Constraints

- Language: Rust (all infrastructure already in place)
- Cloud APIs acceptable (OpenAI Whisper, LLM of choice)
- Local Whisper retained as fallback
- 50-100 videos target scale
- Primary goal: maximize correctly-identified STOP moments
- Explicit exception handling, no silent fallbacks
