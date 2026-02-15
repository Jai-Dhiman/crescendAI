# Masterclass Pipeline Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the fragile audio-segmentation-based pipeline with a transcribe-first architecture that uses the Whisper API for transcription and a two-pass LLM for teaching moment identification.

**Architecture:** Download audio, transcribe entire audio via OpenAI Whisper API (with local fallback), then use an LLM in two passes (detect moments from full transcript, then extract structured data per moment). Audio segmentation is removed from the critical path.

**Tech Stack:** Rust, tokio, reqwest (multipart), OpenAI Whisper API, OpenAI-compatible LLM API, serde, clap

---

### Task 1: Add reqwest multipart support to Cargo.toml

**Files:**

- Modify: `tools/masterclass-pipeline/Cargo.toml`

**Step 1: Add multipart feature to reqwest**

In `Cargo.toml`, change:

```toml
reqwest = { version = "0.12", features = ["json"] }
```

to:

```toml
reqwest = { version = "0.12", features = ["json", "multipart"] }
```

This is needed for uploading audio files to the Whisper API.

**Step 2: Verify it compiles**

Run: `cd tools/masterclass-pipeline && cargo check`
Expected: compiles with no errors

**Step 3: Commit**

```bash
git add tools/masterclass-pipeline/Cargo.toml
git commit -m "add reqwest multipart feature for Whisper API uploads"
```

---

### Task 2: Add Identify stage to pipeline schemas

**Files:**

- Modify: `tools/masterclass-pipeline/src/schemas.rs:129-147`

**Step 1: Add Identify variant to PipelineStage**

Replace the `PipelineStage` enum and its `Display` impl at lines 127-147:

```rust
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum PipelineStage {
    Discover,
    Download,
    Transcribe,
    Segment,    // kept for backward compat with existing state files
    Extract,    // kept for backward compat with existing state files
    Identify,   // new: replaces Segment + Extract
}

impl std::fmt::Display for PipelineStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PipelineStage::Discover => write!(f, "discover"),
            PipelineStage::Download => write!(f, "download"),
            PipelineStage::Transcribe => write!(f, "transcribe"),
            PipelineStage::Segment => write!(f, "segment"),
            PipelineStage::Extract => write!(f, "extract"),
            PipelineStage::Identify => write!(f, "identify"),
        }
    }
}
```

**Step 2: Update stage dependency chain in store.rs**

In `tools/masterclass-pipeline/src/store.rs`, function `get_videos_needing_stage` at line 199-205, add the Identify stage's prerequisite:

```rust
let prev_stage = match stage {
    PipelineStage::Discover => None,
    PipelineStage::Download => Some(PipelineStage::Discover),
    PipelineStage::Transcribe => Some(PipelineStage::Download),
    PipelineStage::Segment => Some(PipelineStage::Transcribe),
    PipelineStage::Extract => Some(PipelineStage::Segment),
    PipelineStage::Identify => Some(PipelineStage::Transcribe), // depends on transcribe, not segment
};
```

Also update `status_summary` at line 326-332 to include the new stage:

```rust
let stages = [
    PipelineStage::Discover,
    PipelineStage::Download,
    PipelineStage::Transcribe,
    PipelineStage::Segment,
    PipelineStage::Extract,
    PipelineStage::Identify,
];
```

And update the Display impl for `StatusSummary` at line 381:

```rust
let stage_order = ["discover", "download", "transcribe", "segment", "extract", "identify"];
```

**Step 3: Verify it compiles**

Run: `cd tools/masterclass-pipeline && cargo check`
Expected: compiles (there will be unused variant warnings for Identify, which is fine for now)

**Step 4: Commit**

```bash
git add tools/masterclass-pipeline/src/schemas.rs tools/masterclass-pipeline/src/store.rs
git commit -m "add Identify pipeline stage to schemas and store"
```

---

### Task 3: Extend LLM client to support OpenAI API authentication

**Files:**

- Modify: `tools/masterclass-pipeline/src/llm_client.rs`

The current `LlmClient` sends to an OpenAI-compatible API without auth headers. We need to support `Authorization: Bearer <key>` for both the OpenAI Whisper API and cloud LLM APIs.

**Step 1: Add api_key field and update constructor**

Replace the `LlmClient` struct and `new()` at lines 8-70 of `llm_client.rs`:

```rust
pub struct LlmClient {
    client: reqwest::Client,
    base_url: String,
    model: String,
    api_key: Option<String>,
}

// ... keep Request/Response types unchanged (lines 14-45) ...

impl LlmClient {
    pub fn new(base_url: Option<&str>, model: &str, api_key: Option<String>) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(600))
            .build()?;

        Ok(Self {
            client,
            base_url: base_url
                .unwrap_or(DEFAULT_BASE_URL)
                .trim_end_matches('/')
                .to_string(),
            model: model.to_string(),
            api_key,
        })
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn client(&self) -> &reqwest::Client {
        &self.client
    }

    pub fn api_key(&self) -> Option<&str> {
        self.api_key.as_deref()
    }
```

**Step 2: Add auth header to send_request**

In the `send_request` private function (around line 133), add the authorization header:

```rust
async fn send_request(&self, request: &ChatRequest) -> Result<String> {
    let url = format!("{}/v1/chat/completions", self.base_url);

    let mut req = self.client
        .post(&url)
        .header("Content-Type", "application/json");

    if let Some(ref key) = self.api_key {
        req = req.header("Authorization", format!("Bearer {}", key));
    }

    let response = req
        .json(request)
        .send()
        .await?;
    // ... rest unchanged
```

**Step 3: Update all call sites**

In `main.rs` line 171 and `pipeline.rs` line 263, the `LlmClient::new` calls need the new `api_key` parameter. For now, pass `None` -- we'll wire up the config in a later task:

```rust
// main.rs line 171
let client = llm_client::LlmClient::new(Some(&cli.llm_url), &cli.llm_model, None)?;

// pipeline.rs line 263
let client = llm_client::LlmClient::new(Some(&self.llm_url), &self.llm_model, None)?;
```

**Step 4: Verify it compiles**

Run: `cd tools/masterclass-pipeline && cargo check`
Expected: compiles with no errors

**Step 5: Commit**

```bash
git add tools/masterclass-pipeline/src/llm_client.rs tools/masterclass-pipeline/src/main.rs tools/masterclass-pipeline/src/pipeline.rs
git commit -m "add API key auth support to LLM client"
```

---

### Task 4: Add Whisper API transcription to transcribe.rs

**Files:**

- Modify: `tools/masterclass-pipeline/src/transcribe.rs`

This is the largest task. We add a new async function `transcribe_video_api` that sends audio chunks to the OpenAI Whisper API, while keeping the existing local `transcribe_video` function.

**Step 1: Add the API transcription function**

Add the following at the end of `transcribe.rs` (after the existing `apply_corrections` function at ~line 595):

```rust
// --- Whisper API transcription ---

use reqwest::multipart;
use std::path::Path;

/// Transcribe a video using the OpenAI Whisper API.
/// Chunks the audio into ~10-minute segments, uploads each, merges results.
pub async fn transcribe_video_api(
    client: &reqwest::Client,
    api_key: &str,
    store: &MasterclassStore,
    video_id: &str,
) -> Result<Transcript> {
    let audio_path = store.audio_path(video_id);
    anyhow::ensure!(audio_path.exists(), "Audio file not found: {}", audio_path.display());

    let samples = read_wav_samples(&audio_path)?;
    let total_samples = samples.len();
    let sample_rate = crate::config::SAMPLE_RATE as usize;

    let chunk_duration = crate::config::CHUNK_DURATION_SECS as usize;
    let chunk_overlap = crate::config::CHUNK_OVERLAP_SECS as usize;
    let chunk_samples = chunk_duration * sample_rate;
    let overlap_samples = chunk_overlap * sample_rate;
    let step_samples = chunk_samples - overlap_samples;

    let mut all_segments: Vec<TranscriptSegment> = Vec::new();
    let mut chunk_idx = 0usize;
    let mut offset = 0usize;

    while offset < total_samples {
        let end = (offset + chunk_samples).min(total_samples);
        let chunk = &samples[offset..end];
        let chunk_start_secs = offset as f64 / sample_rate as f64;

        tracing::info!(
            "  Transcribing chunk {} ({}s - {}s)",
            chunk_idx + 1,
            chunk_start_secs as u64,
            (end as f64 / sample_rate as f64) as u64,
        );

        let chunk_segments = transcribe_chunk_api(
            client,
            api_key,
            chunk,
            sample_rate as u32,
        ).await?;

        // Offset timestamps by chunk start
        for mut seg in chunk_segments {
            seg.start += chunk_start_secs;
            seg.end += chunk_start_secs;
            for tok in &mut seg.tokens {
                tok.start += chunk_start_secs;
                tok.end += chunk_start_secs;
            }
            all_segments.push(seg);
        }

        offset += step_samples;
        chunk_idx += 1;
    }

    // Deduplicate overlapping regions (reuse existing logic)
    if chunk_idx > 1 {
        all_segments = deduplicate_overlaps(
            all_segments,
            chunk_samples as f64 / sample_rate as f64,
            overlap_samples as f64 / sample_rate as f64,
        );
    }

    // Renumber segment IDs
    for (i, seg) in all_segments.iter_mut().enumerate() {
        seg.id = i as u32;
    }

    let transcript = Transcript {
        video_id: video_id.to_string(),
        model: "whisper-1-api".to_string(),
        language: "en".to_string(),
        transcribed_at: chrono::Utc::now().to_rfc3339(),
        segments: all_segments,
    };

    store.save_transcript(&transcript)?;
    Ok(transcript)
}

/// Write samples to a temporary WAV file for upload.
fn write_temp_wav(samples: &[f32], sample_rate: u32) -> Result<tempfile::NamedTempFile> {
    let tmp = tempfile::NamedTempFile::new()?;
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::new(std::io::BufWriter::new(tmp.reopen()?), spec)?;
    for &s in samples {
        let sample_i16 = (s * 32767.0).clamp(-32768.0, 32767.0) as i16;
        writer.write_sample(sample_i16)?;
    }
    writer.finalize()?;
    Ok(tmp)
}

/// Transcribe a single audio chunk via the Whisper API.
async fn transcribe_chunk_api(
    client: &reqwest::Client,
    api_key: &str,
    samples: &[f32],
    sample_rate: u32,
) -> Result<Vec<TranscriptSegment>> {
    let tmp_wav = write_temp_wav(samples, sample_rate)?;
    let wav_bytes = std::fs::read(tmp_wav.path())?;

    let file_part = multipart::Part::bytes(wav_bytes)
        .file_name("audio.wav")
        .mime_str("audio/wav")?;

    let form = multipart::Form::new()
        .part("file", file_part)
        .text("model", "whisper-1")
        .text("response_format", "verbose_json")
        .text("timestamp_granularities[]", "word")
        .text("timestamp_granularities[]", "segment")
        .text("language", "en");

    let response = client
        .post("https://api.openai.com/v1/audio/transcriptions")
        .header("Authorization", format!("Bearer {}", api_key))
        .multipart(form)
        .send()
        .await
        .context("Whisper API request failed")?;

    let status = response.status();
    let body = response.text().await?;

    if !status.is_success() {
        anyhow::bail!("Whisper API error ({}): {}", status, body);
    }

    let api_response: WhisperApiResponse = serde_json::from_str(&body)
        .context("Failed to parse Whisper API response")?;

    // Convert API response to our TranscriptSegment format
    let mut segments = Vec::new();
    if let Some(api_segments) = api_response.segments {
        for (i, api_seg) in api_segments.iter().enumerate() {
            let tokens: Vec<TranscriptToken> = api_response
                .words
                .as_ref()
                .map(|words| {
                    words
                        .iter()
                        .filter(|w| w.start >= api_seg.start && w.end <= api_seg.end)
                        .map(|w| TranscriptToken {
                            text: w.word.clone(),
                            start: w.start,
                            end: w.end,
                            probability: 1.0, // API doesn't give per-word probability in same format
                        })
                        .collect()
                })
                .unwrap_or_default();

            segments.push(TranscriptSegment {
                id: i as u32,
                text: apply_corrections(&api_seg.text),
                start: api_seg.start,
                end: api_seg.end,
                tokens,
            });
        }
    }

    Ok(segments)
}

#[derive(Deserialize)]
struct WhisperApiResponse {
    #[allow(dead_code)]
    text: String,
    segments: Option<Vec<WhisperApiSegment>>,
    words: Option<Vec<WhisperApiWord>>,
}

#[derive(Deserialize)]
struct WhisperApiSegment {
    text: String,
    start: f64,
    end: f64,
}

#[derive(Deserialize)]
struct WhisperApiWord {
    word: String,
    start: f64,
    end: f64,
}
```

**Step 2: Add tempfile dependency**

In `Cargo.toml`, add under `[dependencies]`:

```toml
tempfile = "3"
```

**Step 3: Verify it compiles**

Run: `cd tools/masterclass-pipeline && cargo check`
Expected: compiles (the new function is not called yet, so there may be dead_code warnings)

**Step 4: Commit**

```bash
git add tools/masterclass-pipeline/src/transcribe.rs tools/masterclass-pipeline/Cargo.toml
git commit -m "add Whisper API transcription with chunked upload"
```

---

### Task 5: Create the identify module (two-pass LLM moment detection)

**Files:**

- Create: `tools/masterclass-pipeline/src/identify.rs`

This is the core new module that replaces both `segment.rs` and `extract.rs`.

**Step 1: Create identify.rs**

Create `tools/masterclass-pipeline/src/identify.rs`:

```rust
use anyhow::{Context, Result};
use serde::Deserialize;

use crate::llm_client::LlmClient;
use crate::schemas::*;
use crate::store::MasterclassStore;

/// Identify teaching moments from a transcript using a two-pass LLM approach.
/// Pass 1: Detect all moments (timestamps + descriptions) from the full transcript.
/// Pass 2: Extract structured TeachingMoment data for each detected moment.
pub async fn identify_teaching_moments(
    client: &LlmClient,
    store: &MasterclassStore,
    video_id: &str,
) -> Result<Vec<TeachingMoment>> {
    let video = store
        .get_video(video_id)?
        .with_context(|| format!("Video metadata not found for {}", video_id))?;

    let transcript = store
        .load_transcript(video_id)?
        .with_context(|| format!("Transcript not found for {}. Run transcribe first.", video_id))?;

    if transcript.segments.is_empty() {
        tracing::info!("Empty transcript for {}, skipping identification", video_id);
        store.save_teaching_moments(video_id, &[])?;
        return Ok(Vec::new());
    }

    // Pass 1: Detect moments from full transcript
    tracing::info!("Pass 1: Detecting teaching moments in {}", video_id);
    let detected = detect_moments(client, &video, &transcript).await?;

    if detected.is_empty() {
        tracing::info!("No teaching moments detected in {}", video_id);
        store.save_teaching_moments(video_id, &[])?;
        return Ok(Vec::new());
    }

    tracing::info!("Detected {} potential teaching moments", detected.len());

    // Pass 2: Extract structured data for each moment
    tracing::info!("Pass 2: Extracting structured data for each moment");
    let total_stops = detected.len() as u32;
    let mut all_moments = Vec::new();

    for (idx, detection) in detected.iter().enumerate() {
        tracing::info!("  Moment {}/{}", idx + 1, total_stops);

        let context_window = build_context_window(&transcript, detection.timestamp, 120.0);
        let playing_bounds = estimate_playing_bounds(&transcript, detection.timestamp);

        let extraction = extract_moment(
            client,
            &video,
            &context_window,
            detection,
            total_stops,
        )
        .await;

        match extraction {
            Ok(extracted) => {
                let moment = TeachingMoment {
                    moment_id: uuid::Uuid::new_v4().to_string(),
                    video_id: video_id.to_string(),
                    video_title: video.title.clone(),
                    teacher: video.teacher.clone().unwrap_or_else(|| "Unknown".to_string()),

                    stop_timestamp: detection.timestamp,
                    feedback_start: detection.timestamp,
                    feedback_end: detection.timestamp + extracted.duration_estimate.unwrap_or(30.0),
                    playing_before_start: playing_bounds.0,
                    playing_before_end: playing_bounds.1,

                    transcript_text: context_window.clone(),
                    feedback_summary: extracted.feedback_summary,
                    musical_dimension: extracted.musical_dimension,
                    secondary_dimensions: extracted.secondary_dimensions,
                    severity: extracted.severity,
                    feedback_type: extracted.feedback_type,

                    piece: extracted.piece.or_else(|| video.pieces.first().cloned()),
                    composer: extracted.composer.or_else(|| video.composers.first().cloned()),
                    passage_description: extracted.passage_description,
                    student_level: extracted.student_level,

                    stop_order: (idx as u32) + 1,
                    total_stops,
                    time_spent_seconds: extracted.duration_estimate.unwrap_or(30.0),
                    demonstrated: extracted.demonstrated,

                    extracted_at: chrono::Utc::now().to_rfc3339(),
                    extraction_model: client.model().to_string(),
                    confidence: extracted.confidence,
                };
                all_moments.push(moment);
            }
            Err(e) => {
                tracing::error!("  Moment {}: extraction failed: {}", idx + 1, e);
            }
        }
    }

    if all_moments.is_empty() && !detected.is_empty() {
        anyhow::bail!(
            "Extracted 0 moments from {} detected stops -- LLM output format not usable",
            detected.len()
        );
    }

    store.save_teaching_moments(video_id, &all_moments)?;
    Ok(all_moments)
}

// --- Pass 1: Moment Detection ---

#[derive(Deserialize, Debug)]
pub struct DetectedMoment {
    pub timestamp: f64,
    pub description: String,
}

async fn detect_moments(
    client: &LlmClient,
    video: &VideoMetadata,
    transcript: &Transcript,
) -> Result<Vec<DetectedMoment>> {
    let formatted_transcript = format_transcript_with_timestamps(transcript);

    let system_prompt = r#"You analyze piano masterclass transcripts to find teaching moments.

A "teaching moment" is when the teacher stops the student's playing to give feedback, make a correction, offer a suggestion, demonstrate something, or explain a musical concept related to what was just played.

NOT teaching moments: casual conversation, introductions, applause, the teacher simply saying "good" without elaboration, or pure performance segments.

You MUST respond with ONLY a JSON array. No explanation, no markdown fences.

Each element: {"timestamp": <seconds as number>, "description": "<brief description of the feedback>"}

Example:
[
  {"timestamp": 245.5, "description": "Teacher stops student to correct dynamics in the opening phrase"},
  {"timestamp": 512.0, "description": "Teacher demonstrates how to voice the melody over accompaniment"}
]

If there are no teaching moments, respond with an empty array: []"#;

    let user_prompt = format!(
        "Video: {}\nTeacher: {}\n\nFull transcript with timestamps:\n\n{}\n\nIdentify every teaching moment. Respond with ONLY the JSON array.",
        video.title,
        video.teacher.as_deref().unwrap_or("Unknown"),
        formatted_transcript,
    );

    // The full transcript may be long. If it exceeds ~80k chars, chunk it.
    let max_transcript_len = 80_000;
    if user_prompt.len() <= max_transcript_len {
        return detect_moments_single(client, &system_prompt, &user_prompt).await;
    }

    // Chunk the transcript into overlapping windows and detect in each
    tracing::info!("  Transcript too long ({}), chunking for detection", user_prompt.len());
    let mut all_detected = Vec::new();
    let segments = &transcript.segments;
    let window_duration = 600.0; // 10-minute windows
    let overlap_duration = 60.0;  // 1-minute overlap

    let total_duration = segments.last().map(|s| s.end).unwrap_or(0.0);
    let mut window_start = 0.0;

    while window_start < total_duration {
        let window_end = (window_start + window_duration).min(total_duration);
        let window_segments: Vec<&TranscriptSegment> = segments
            .iter()
            .filter(|s| s.start >= window_start && s.end <= window_end + overlap_duration)
            .collect();

        if window_segments.is_empty() {
            window_start += window_duration - overlap_duration;
            continue;
        }

        let window_text: String = window_segments
            .iter()
            .map(|s| format!("[{:.1}s] {}", s.start, s.text.trim()))
            .collect::<Vec<_>>()
            .join("\n");

        let chunk_prompt = format!(
            "Video: {}\nTeacher: {}\nSection: {:.0}s - {:.0}s\n\nTranscript:\n\n{}\n\nIdentify every teaching moment in this section. Respond with ONLY the JSON array.",
            video.title,
            video.teacher.as_deref().unwrap_or("Unknown"),
            window_start,
            window_end,
            window_text,
        );

        match detect_moments_single(client, system_prompt, &chunk_prompt).await {
            Ok(moments) => all_detected.extend(moments),
            Err(e) => tracing::warn!("  Detection failed for window {:.0}-{:.0}s: {}", window_start, window_end, e),
        }

        window_start += window_duration - overlap_duration;
    }

    // Deduplicate moments that are within 10 seconds of each other
    all_detected.sort_by(|a, b| a.timestamp.partial_cmp(&b.timestamp).unwrap());
    let mut deduped = Vec::new();
    for moment in all_detected {
        if deduped.last().map_or(true, |prev: &DetectedMoment| {
            (moment.timestamp - prev.timestamp).abs() > 10.0
        }) {
            deduped.push(moment);
        }
    }

    Ok(deduped)
}

async fn detect_moments_single(
    client: &LlmClient,
    system_prompt: &str,
    user_prompt: &str,
) -> Result<Vec<DetectedMoment>> {
    let response = client.message(system_prompt, user_prompt).await?;
    parse_detected_moments(&response)
}

fn parse_detected_moments(response: &str) -> Result<Vec<DetectedMoment>> {
    let trimmed = response.trim();

    // Find the JSON array
    let json_str = if trimmed.starts_with('[') {
        find_json_array(trimmed)?
    } else if trimmed.starts_with("```") {
        let inner = trimmed
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim();
        find_json_array(inner)?
    } else {
        let start = trimmed.find('[').context("No JSON array found in response")?;
        find_json_array(&trimmed[start..])?
    };

    let moments: Vec<DetectedMoment> = serde_json::from_str(json_str)
        .with_context(|| format!("Failed to parse detected moments JSON: {}...", &json_str[..json_str.len().min(300)]))?;

    Ok(moments)
}

fn find_json_array(s: &str) -> Result<&str> {
    anyhow::ensure!(s.starts_with('['), "String does not start with [");

    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape_next = false;

    for (i, ch) in s.char_indices() {
        if escape_next {
            escape_next = false;
            continue;
        }
        match ch {
            '\\' if in_string => escape_next = true,
            '"' => in_string = !in_string,
            '[' if !in_string => depth += 1,
            ']' if !in_string => {
                depth -= 1;
                if depth == 0 {
                    return Ok(&s[..=i]);
                }
            }
            _ => {}
        }
    }

    anyhow::bail!("Unterminated JSON array")
}

// --- Pass 2: Moment Extraction ---

#[derive(Deserialize)]
struct RawExtraction {
    feedback_summary: String,
    musical_dimension: String,
    #[serde(default)]
    secondary_dimensions: Vec<String>,
    severity: String,
    feedback_type: String,
    piece: Option<String>,
    composer: Option<String>,
    passage_description: Option<String>,
    student_level: Option<String>,
    #[serde(default)]
    demonstrated: bool,
    #[serde(default = "default_confidence")]
    confidence: f32,
    #[serde(default)]
    duration_estimate: Option<f64>,
}

fn default_confidence() -> f32 {
    0.7
}

async fn extract_moment(
    client: &LlmClient,
    video: &VideoMetadata,
    context_window: &str,
    detection: &DetectedMoment,
    total_stops: u32,
) -> Result<RawExtraction> {
    let system_prompt = build_extraction_system_prompt();
    let user_prompt = format!(
        r#"Video: {}
Teacher: {}
Total teaching moments in this masterclass: {}
Detected moment at {:.1}s: {}

Transcript context around this moment:
{}

Respond with a single JSON object describing this teaching moment."#,
        video.title,
        video.teacher.as_deref().unwrap_or("Unknown"),
        total_stops,
        detection.timestamp,
        detection.description,
        context_window,
    );

    // Try up to 2 times
    for attempt in 0..2 {
        let prompt = if attempt == 0 {
            user_prompt.clone()
        } else {
            format!(
                "{}\n\nYour previous response was not valid JSON. Respond with ONLY the JSON object, nothing else.",
                user_prompt
            )
        };

        match client.message(&system_prompt, &prompt).await {
            Ok(response) => match parse_extraction(&response) {
                Ok(extracted) => return Ok(extracted),
                Err(e) => {
                    tracing::warn!("  Parse failed (attempt {}): {}", attempt + 1, e);
                }
            },
            Err(e) => {
                tracing::error!("  LLM request failed: {}", e);
            }
        }
    }

    anyhow::bail!("Failed to extract moment after retries")
}

fn build_extraction_system_prompt() -> String {
    r#"You analyze piano masterclass transcripts. When given a moment where a teacher stopped a student to give feedback, you extract structured information as JSON.

You MUST respond with ONLY a single JSON object (not an array). No explanation, no markdown fences, just the JSON.

The JSON object must have exactly these fields:

{
  "feedback_summary": "1-2 sentence summary of what the teacher said",
  "musical_dimension": "one of: dynamics, timing, articulation, pedaling, tone_color, phrasing, voicing, interpretation, technique, structure",
  "secondary_dimensions": [],
  "severity": "one of: minor, moderate, significant, critical",
  "feedback_type": "one of: correction, suggestion, demonstration, praise, explanation, comparison",
  "piece": null,
  "composer": null,
  "passage_description": null,
  "student_level": null,
  "demonstrated": false,
  "confidence": 0.7,
  "duration_estimate": 30.0
}

Dimension definitions:
- dynamics: volume, loud/soft, crescendo, forte, piano
- timing: tempo, rhythm, rubato, rushing, dragging
- articulation: legato, staccato, accents, note connection
- pedaling: sustain pedal, damper, una corda
- tone_color: timbre, sound quality, bright/dark/warm
- phrasing: musical line, breathing, shape, direction
- voicing: balance between hands, bringing out melody
- interpretation: expression, emotion, style, character
- technique: finger/hand/arm position, physical approach
- structure: form, sections, development

The "duration_estimate" field should be your estimate in seconds of how long the teacher's feedback lasts, based on the amount of transcript text."#
        .to_string()
}

fn parse_extraction(response: &str) -> Result<RawExtraction> {
    let trimmed = response.trim();

    let json_str = if trimmed.starts_with('{') {
        find_json_object(trimmed)?
    } else if trimmed.starts_with("```") {
        let inner = trimmed
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim();
        find_json_object(inner)?
    } else {
        let start = trimmed.find('{').context("No JSON object found in response")?;
        find_json_object(&trimmed[start..])?
    };

    let extraction: RawExtraction = serde_json::from_str(json_str).with_context(|| {
        format!("Failed to parse JSON: {}...", &json_str[..json_str.len().min(300)])
    })?;

    Ok(extraction)
}

fn find_json_object(s: &str) -> Result<&str> {
    anyhow::ensure!(s.starts_with('{'), "String does not start with {{");

    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape_next = false;

    for (i, ch) in s.char_indices() {
        if escape_next {
            escape_next = false;
            continue;
        }
        match ch {
            '\\' if in_string => escape_next = true,
            '"' => in_string = !in_string,
            '{' if !in_string => depth += 1,
            '}' if !in_string => {
                depth -= 1;
                if depth == 0 {
                    return Ok(&s[..=i]);
                }
            }
            _ => {}
        }
    }

    anyhow::bail!("Unterminated JSON object")
}

// --- Helper functions ---

fn format_transcript_with_timestamps(transcript: &Transcript) -> String {
    transcript
        .segments
        .iter()
        .map(|s| format!("[{:.1}s] {}", s.start, s.text.trim()))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Build a context window of transcript text around a given timestamp.
/// Returns the formatted transcript text within +/- window_secs of the timestamp.
fn build_context_window(transcript: &Transcript, timestamp: f64, window_secs: f64) -> String {
    let start = (timestamp - window_secs).max(0.0);
    let end = timestamp + window_secs;

    transcript
        .segments
        .iter()
        .filter(|s| s.end >= start && s.start <= end)
        .map(|s| format!("[{:.1}s] {}", s.start, s.text.trim()))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Estimate playing boundaries from transcript gaps.
/// Finds the largest gap in transcript coverage before the stop timestamp,
/// which likely corresponds to student playing.
fn estimate_playing_bounds(transcript: &Transcript, stop_timestamp: f64) -> (f64, f64) {
    let lookback = 120.0; // Look back up to 2 minutes
    let search_start = (stop_timestamp - lookback).max(0.0);

    // Find transcript segments in the lookback window
    let relevant_segments: Vec<&TranscriptSegment> = transcript
        .segments
        .iter()
        .filter(|s| s.end >= search_start && s.start <= stop_timestamp)
        .collect();

    if relevant_segments.is_empty() {
        // No transcript at all before stop -- assume playing filled the gap
        return (search_start, stop_timestamp);
    }

    // Find the largest gap between consecutive segments
    let mut best_gap_start = search_start;
    let mut best_gap_end = search_start;
    let mut best_gap_duration = 0.0;

    // Check gap before first relevant segment
    if let Some(first) = relevant_segments.first() {
        let gap = first.start - search_start;
        if gap > best_gap_duration {
            best_gap_start = search_start;
            best_gap_end = first.start;
            best_gap_duration = gap;
        }
    }

    // Check gaps between consecutive segments
    for window in relevant_segments.windows(2) {
        let gap_start = window[0].end;
        let gap_end = window[1].start;
        let gap = gap_end - gap_start;
        if gap > best_gap_duration && gap > 3.0 {
            // Minimum 3s gap to count as playing
            best_gap_start = gap_start;
            best_gap_end = gap_end;
            best_gap_duration = gap;
        }
    }

    // Check gap between last segment and stop timestamp
    if let Some(last) = relevant_segments.last() {
        let gap = stop_timestamp - last.end;
        if gap > best_gap_duration && gap > 3.0 {
            best_gap_start = last.end;
            best_gap_end = stop_timestamp;
            best_gap_duration = gap;
        }
    }

    // If no significant gap found, use a default window
    if best_gap_duration < 3.0 {
        let default_start = (stop_timestamp - 10.0).max(0.0);
        return (default_start, stop_timestamp);
    }

    (best_gap_start, best_gap_end)
}
```

**Step 2: Register the module in main.rs**

Add `mod identify;` to the module declarations at the top of `main.rs` (after `mod extract;` on line 6):

```rust
mod identify;
```

**Step 3: Verify it compiles**

Run: `cd tools/masterclass-pipeline && cargo check`
Expected: compiles with warnings about unused functions (the module isn't wired into the CLI yet)

**Step 4: Commit**

```bash
git add tools/masterclass-pipeline/src/identify.rs tools/masterclass-pipeline/src/main.rs
git commit -m "add identify module with two-pass LLM moment detection"
```

---

### Task 6: Wire up CLI and pipeline for the new stages

**Files:**

- Modify: `tools/masterclass-pipeline/src/main.rs`
- Modify: `tools/masterclass-pipeline/src/pipeline.rs`

**Step 1: Add CLI args for API key and --local flag**

In `main.rs`, add to the `Cli` struct (after the `piece` field at line 55):

```rust
    /// OpenAI API key (for Whisper API and cloud LLMs). Also reads OPENAI_API_KEY env var.
    #[arg(long, env = "OPENAI_API_KEY", global = true)]
    openai_api_key: Option<String>,

    /// Use local Whisper model instead of API
    #[arg(long, global = true)]
    local: bool,
```

**Step 2: Add Identify command to Commands enum**

In the `Commands` enum (after `Extract` at line 76):

```rust
    /// Identify teaching moments from transcript (replaces segment+extract)
    Identify,
```

**Step 3: Add Identify command handler in main()**

In the `match cli.command` block (after the `Extract` handler, before `Commands::Run`), add:

```rust
        Commands::Identify => {
            let api_key = cli.openai_api_key.as_deref()
                .ok_or_else(|| anyhow::anyhow!("OpenAI API key required. Set --openai-api-key or OPENAI_API_KEY env var."))?;
            let videos = get_videos(&store, &schemas::PipelineStage::Identify, cli.force, cli.max_videos, cli.piece.as_deref())?;
            tracing::info!("Identifying teaching moments in {} videos", videos.len());
            if !videos.is_empty() {
                let client = llm_client::LlmClient::new(Some(&cli.llm_url), &cli.llm_model, Some(api_key.to_string()))?;
                for video_id in &videos {
                    if cli.dry_run {
                        tracing::info!("[dry-run] Would identify moments in {}", video_id);
                        continue;
                    }
                    match identify::identify_teaching_moments(&client, &store, video_id).await {
                        Ok(moments) => {
                            tracing::info!("Identified {} moments in {}", moments.len(), video_id);
                            store.mark_stage_complete(video_id, &schemas::PipelineStage::Identify)?;
                        }
                        Err(e) => {
                            tracing::error!("Failed to identify moments in {}: {}", video_id, e);
                            store.mark_stage_failed(video_id, &schemas::PipelineStage::Identify, &e.to_string())?;
                        }
                    }
                }
            }
        }
```

**Step 4: Update the Transcribe command to support --local flag**

Replace the `Commands::Transcribe` handler to support both API and local:

```rust
        Commands::Transcribe => {
            let videos = get_videos(&store, &schemas::PipelineStage::Transcribe, cli.force, cli.max_videos, cli.piece.as_deref())?;
            tracing::info!("Transcribing {} videos", videos.len());
            if !videos.is_empty() {
                if cli.local {
                    // Local Whisper
                    let model_path = cli.data_dir.join("models").join(format!("ggml-{}.bin", cli.whisper_model));
                    let ctx = transcribe::load_whisper_context(&model_path)?;
                    for video_id in &videos {
                        if cli.dry_run {
                            tracing::info!("[dry-run] Would transcribe {} (local)", video_id);
                            continue;
                        }
                        match transcribe::transcribe_video(&ctx, &store, video_id) {
                            Ok(_) => store.mark_stage_complete(video_id, &schemas::PipelineStage::Transcribe)?,
                            Err(e) => {
                                tracing::error!("Failed to transcribe {}: {}", video_id, e);
                                store.mark_stage_failed(video_id, &schemas::PipelineStage::Transcribe, &e.to_string())?;
                            }
                        }
                    }
                } else {
                    // Whisper API
                    let api_key = cli.openai_api_key.as_deref()
                        .ok_or_else(|| anyhow::anyhow!("OpenAI API key required for Whisper API. Set --openai-api-key or OPENAI_API_KEY env var. Use --local for local Whisper."))?;
                    let http_client = reqwest::Client::builder()
                        .timeout(std::time::Duration::from_secs(600))
                        .build()?;
                    for video_id in &videos {
                        if cli.dry_run {
                            tracing::info!("[dry-run] Would transcribe {} (API)", video_id);
                            continue;
                        }
                        match transcribe::transcribe_video_api(&http_client, api_key, &store, video_id).await {
                            Ok(_) => store.mark_stage_complete(video_id, &schemas::PipelineStage::Transcribe)?,
                            Err(e) => {
                                tracing::error!("Failed to transcribe {}: {}", video_id, e);
                                store.mark_stage_failed(video_id, &schemas::PipelineStage::Transcribe, &e.to_string())?;
                            }
                        }
                    }
                }
            }
        }
```

**Step 5: Update Pipeline::run() to use identify instead of segment+extract by default**

In `pipeline.rs`, add `openai_api_key` and `local` fields to `Pipeline` struct:

```rust
pub struct Pipeline {
    store: MasterclassStore,
    data_dir: PathBuf,
    whisper_model: String,
    llm_model: String,
    llm_url: String,
    force: bool,
    max_videos: Option<usize>,
    dry_run: bool,
    piece_filter: Option<String>,
    openai_api_key: Option<String>,
    local: bool,
}
```

Update the `Pipeline::new()` constructor to accept these new fields, and update the `run()` method to use transcribe API + identify instead of local transcribe + segment + extract when not in local mode.

Update the `Pipeline::run()` method:

```rust
    pub async fn run(self) -> Result<PipelineReport> {
        let mut stages = Vec::new();

        if let Some(ref piece) = self.piece_filter {
            tracing::info!("=== Piece filter: {} ===", piece);
        }

        // Stage 1: Discover
        tracing::info!("=== Stage: Discover ===");
        stages.push(self.run_discover().await?);

        // Stage 2: Download
        tracing::info!("=== Stage: Download ===");
        stages.push(self.run_download().await?);

        // Stage 3: Transcribe
        tracing::info!("=== Stage: Transcribe ===");
        if self.local {
            stages.push(self.run_transcribe_local()?);
        } else {
            stages.push(self.run_transcribe_api().await?);
        }

        if self.local {
            // Legacy path: segment + extract
            tracing::info!("=== Stage: Segment ===");
            stages.push(self.run_segment()?);

            tracing::info!("=== Stage: Extract ===");
            stages.push(self.run_extract().await?);
        } else {
            // New path: identify
            tracing::info!("=== Stage: Identify ===");
            stages.push(self.run_identify().await?);
        }

        Ok(PipelineReport { stages })
    }
```

Add the new methods `run_transcribe_api()` and `run_identify()`:

```rust
    async fn run_transcribe_api(&self) -> Result<StageReport> {
        let videos = self.get_stage_videos(&PipelineStage::Transcribe)?;
        if videos.is_empty() {
            return Ok(StageReport {
                stage: "transcribe".to_string(),
                processed: 0, succeeded: 0, failed: 0, skipped: 0,
            });
        }

        let api_key = self.openai_api_key.as_deref()
            .ok_or_else(|| anyhow::anyhow!("OpenAI API key required"))?;
        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(600))
            .build()?;

        let mut succeeded = 0;
        let mut failed = 0;

        for video_id in &videos {
            if self.dry_run {
                tracing::info!("[dry-run] Would transcribe {} (API)", video_id);
                continue;
            }
            match transcribe::transcribe_video_api(&http_client, api_key, &self.store, video_id).await {
                Ok(_) => {
                    self.store.mark_stage_complete(video_id, &PipelineStage::Transcribe)?;
                    succeeded += 1;
                }
                Err(e) => {
                    tracing::error!("Transcription (API) failed for {}: {}", video_id, e);
                    self.store.mark_stage_failed(video_id, &PipelineStage::Transcribe, &e.to_string())?;
                    failed += 1;
                }
            }
        }

        Ok(StageReport {
            stage: "transcribe".to_string(),
            processed: videos.len(), succeeded, failed, skipped: 0,
        })
    }

    async fn run_identify(&self) -> Result<StageReport> {
        let videos = self.get_stage_videos(&PipelineStage::Identify)?;
        if videos.is_empty() {
            return Ok(StageReport {
                stage: "identify".to_string(),
                processed: 0, succeeded: 0, failed: 0, skipped: 0,
            });
        }

        let api_key = self.openai_api_key.as_ref().map(|k| k.clone());
        let client = llm_client::LlmClient::new(Some(&self.llm_url), &self.llm_model, api_key)?;
        let mut succeeded = 0;
        let mut failed = 0;

        for video_id in &videos {
            if self.dry_run {
                tracing::info!("[dry-run] Would identify moments in {}", video_id);
                continue;
            }
            match identify::identify_teaching_moments(&client, &self.store, video_id).await {
                Ok(moments) => {
                    tracing::info!("Identified {} moments in {}", moments.len(), video_id);
                    self.store.mark_stage_complete(video_id, &PipelineStage::Identify)?;
                    succeeded += 1;
                }
                Err(e) => {
                    tracing::error!("Identification failed for {}: {}", video_id, e);
                    self.store.mark_stage_failed(video_id, &PipelineStage::Identify, &e.to_string())?;
                    failed += 1;
                }
            }
        }

        Ok(StageReport {
            stage: "identify".to_string(),
            processed: videos.len(), succeeded, failed, skipped: 0,
        })
    }

    fn run_transcribe_local(&self) -> Result<StageReport> {
        // This is the existing run_transcribe() method, renamed
        // ... same implementation as current run_transcribe() ...
    }
```

Also update the `Pipeline::new()` call in `main.rs` `Commands::Run` to pass the new fields:

```rust
        Commands::Run => {
            let pipe = pipeline::Pipeline::new(
                store,
                cli.data_dir.clone(),
                cli.whisper_model.clone(),
                cli.llm_model.clone(),
                cli.llm_url.clone(),
                cli.force,
                cli.max_videos,
                cli.dry_run,
                cli.piece.clone(),
                cli.openai_api_key.clone(),
                cli.local,
            );
```

**Step 6: Verify it compiles**

Run: `cd tools/masterclass-pipeline && cargo check`
Expected: compiles with no errors

**Step 7: Commit**

```bash
git add tools/masterclass-pipeline/src/main.rs tools/masterclass-pipeline/src/pipeline.rs
git commit -m "wire up identify and Whisper API stages to CLI and pipeline"
```

---

### Task 7: End-to-end test with one video

**Files:**

- No new files

This is a manual verification step. Run the new pipeline on one video to verify everything works.

**Step 1: Clear existing state for one video to reprocess**

Pick one of the problematic videos (Zimerman: ALDzxU452gA had 0 moments with old pipeline).

Run:

```bash
cd tools/masterclass-pipeline
# Retranscribe and identify one video
OPENAI_API_KEY=<your key> cargo run -- transcribe --force --piece "Ballade" --max-videos 1
```

**Step 2: Verify transcript was created**

Check that `data/transcripts/<video_id>.json` was updated and contains real transcript text (not hallucinated "Teacher gives feedback" text).

**Step 3: Run identify stage**

```bash
OPENAI_API_KEY=<your key> cargo run -- identify --piece "Ballade" --max-videos 1
```

**Step 4: Verify moments were extracted**

Check `data/teaching_moments/<video_id>.jsonl` contains real teaching moments with meaningful summaries.

**Step 5: Compare with old pipeline**

The old pipeline got 0 moments for Zimerman. Count how many the new pipeline finds. Any number > 0 is an improvement.

**Step 6: Commit if working**

```bash
git add -A
git commit -m "verify new pipeline works end-to-end on test video"
```

---

### Task 8: Run full pipeline on all videos

**Files:**

- No new files

**Step 1: Run the full pipeline**

```bash
cd tools/masterclass-pipeline
OPENAI_API_KEY=<your key> cargo run -- run --force
```

This will:

1. Discover all videos
2. Download audio
3. Transcribe via Whisper API
4. Identify teaching moments via two-pass LLM
5. All results saved to data/

**Step 2: Export and verify**

```bash
cargo run -- export --output all_moments.jsonl
python verify_moments.py data/
```

Review the output. Check:

- All 5 videos now produce teaching moments (especially Zimerman and Perahia which had 0 before)
- No hallucinated "Teacher gives feedback on student performance" text
- No duplicate moments
- Moments have reasonable timestamps and durations

**Step 3: Commit results**

```bash
git add all_moments.jsonl
git commit -m "re-extract all teaching moments with transcribe-first pipeline"
```
