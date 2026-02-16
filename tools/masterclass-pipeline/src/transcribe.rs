use anyhow::{Context, Result};
use std::path::Path;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::config;
use crate::schemas::*;
use crate::store::MasterclassStore;

const CORRECTIONS: &[(&str, &str)] = &[
    ("robot o", "rubato"),
    ("rub auto", "rubato"),
    ("ruba to", "rubato"),
    ("leg auto", "legato"),
    ("lega to", "legato"),
    ("stack auto", "staccato"),
    ("staka to", "staccato"),
    ("show pan", "Chopin"),
    ("chop in", "Chopin"),
    ("four tay", "forte"),
    ("for tay", "forte"),
    ("piano see mo", "pianissimo"),
    ("for tis see mo", "fortissimo"),
    ("mezzo for tay", "mezzo forte"),
    ("cresh endo", "crescendo"),
    ("dim in u endo", "diminuendo"),
    ("rit are dan do", "ritardando"),
    ("a chel er an do", "accelerando"),
    ("sfor zan do", "sforzando"),
    ("arpe gio", "arpeggio"),
    ("ar peggio", "arpeggio"),
];

pub async fn download_model(data_dir: &Path, model_name: &str) -> Result<()> {
    let models_dir = data_dir.join("models");
    std::fs::create_dir_all(&models_dir)?;

    let filename = format!("ggml-{}.bin", model_name);
    let model_path = models_dir.join(&filename);

    if model_path.exists() {
        let size = std::fs::metadata(&model_path)?.len();
        if size > 0 {
            tracing::info!("Model already exists at {} ({:.0} MB)", model_path.display(), size as f64 / 1_048_576.0);
            return Ok(());
        }
    }

    let url = crate::config::whisper_model_url(model_name);
    tracing::info!("Downloading Whisper model from {}", url);
    tracing::info!("This may take a while for large models...");

    let client = reqwest::Client::new();
    let response = client
        .get(&url)
        .send()
        .await
        .with_context(|| format!("Failed to download model from {}", url))?;

    anyhow::ensure!(
        response.status().is_success(),
        "Failed to download model: HTTP {}",
        response.status()
    );

    let bytes = response
        .bytes()
        .await
        .with_context(|| "Failed to read model response body")?;

    let tmp_path = model_path.with_extension("bin.tmp");
    std::fs::write(&tmp_path, &bytes)?;
    std::fs::rename(&tmp_path, &model_path)?;

    tracing::info!(
        "Model saved to {} ({:.0} MB)",
        model_path.display(),
        bytes.len() as f64 / 1_048_576.0
    );

    Ok(())
}

pub fn load_whisper_context(model_path: &Path) -> Result<WhisperContext> {
    anyhow::ensure!(
        model_path.exists(),
        "Whisper model not found at {}. Run 'masterclass-pipeline setup' first.",
        model_path.display()
    );

    let ctx = WhisperContext::new_with_params(
        model_path.to_str().with_context(|| "Invalid model path")?,
        WhisperContextParameters::default(),
    )
    .map_err(|e| anyhow::anyhow!("Failed to load Whisper model: {}", e))?;

    Ok(ctx)
}

pub fn transcribe_video(
    ctx: &WhisperContext,
    store: &MasterclassStore,
    video_id: &str,
) -> Result<Transcript> {
    let audio_path = store.audio_path(video_id);
    anyhow::ensure!(
        audio_path.exists(),
        "Audio file not found for {}. Run download first.",
        video_id
    );

    tracing::info!("Transcribing {}", video_id);

    let samples = read_wav_samples(&audio_path)?;
    let total_duration = samples.len() as f64 / config::SAMPLE_RATE as f64;
    tracing::info!(
        "Loaded {:.1}s of audio ({} samples)",
        total_duration,
        samples.len()
    );

    let segments = transcribe_chunked(ctx, &samples)?;

    tracing::info!(
        "Transcribed {} segments from {}",
        segments.len(),
        video_id
    );

    let transcript = Transcript {
        video_id: video_id.to_string(),
        model: "whisper-large-v3".to_string(),
        language: "en".to_string(),
        transcribed_at: chrono::Utc::now().to_rfc3339(),
        segments,
    };

    store.save_transcript(&transcript)?;
    Ok(transcript)
}

/// Transcribe audio in chunks to avoid Whisper hallucination loops.
///
/// Splits audio into CHUNK_DURATION_SECS chunks with CHUNK_OVERLAP_SECS overlap,
/// transcribes each independently with a fresh decoder state, detects and retries
/// hallucinated chunks, then stitches and deduplicates the results.
fn transcribe_chunked(
    ctx: &WhisperContext,
    samples: &[f32],
) -> Result<Vec<TranscriptSegment>> {
    let sample_rate = config::SAMPLE_RATE as f64;
    let total_duration = samples.len() as f64 / sample_rate;

    let chunk_dur = config::CHUNK_DURATION_SECS;
    let overlap = config::CHUNK_OVERLAP_SECS;

    // Calculate chunk boundaries
    let mut boundaries: Vec<(f64, f64)> = Vec::new();
    let mut start = 0.0;
    while start < total_duration {
        let end = (start + chunk_dur).min(total_duration);
        boundaries.push((start, end));
        start += chunk_dur - overlap;
        if end >= total_duration {
            break;
        }
    }

    tracing::info!(
        "Splitting {:.0}s audio into {} chunks ({:.0}s each, {:.0}s overlap)",
        total_duration,
        boundaries.len(),
        chunk_dur,
        overlap
    );

    let mut all_segments: Vec<TranscriptSegment> = Vec::new();

    for (chunk_idx, &(chunk_start, chunk_end)) in boundaries.iter().enumerate() {
        let start_sample = (chunk_start * sample_rate) as usize;
        let end_sample = ((chunk_end * sample_rate) as usize).min(samples.len());
        let chunk_samples = &samples[start_sample..end_sample];

        tracing::info!(
            "Chunk {}/{}: {:.0}s - {:.0}s ({:.0}s)",
            chunk_idx + 1,
            boundaries.len(),
            chunk_start,
            chunk_end,
            chunk_end - chunk_start
        );

        let mut chunk_segments = transcribe_chunk(ctx, chunk_samples, false)?;

        // Detect repetition loops in this chunk
        let hallucinated_ranges = detect_repetition(&chunk_segments);

        if !hallucinated_ranges.is_empty() {
            let hallucinated_count: usize = hallucinated_ranges
                .iter()
                .map(|&(s, e)| e - s)
                .sum();
            tracing::warn!(
                "Chunk {}: detected {} hallucinated segments in {} runs, retrying with no_context",
                chunk_idx + 1,
                hallucinated_count,
                hallucinated_ranges.len()
            );

            // Retry with no_context to break the loop
            let retry_segments = transcribe_chunk(ctx, chunk_samples, true)?;
            let retry_hallucinated = detect_repetition(&retry_segments);

            if retry_hallucinated.is_empty() {
                chunk_segments = retry_segments;
                tracing::info!("Chunk {}: retry succeeded, no hallucination", chunk_idx + 1);
            } else {
                // Still hallucinated - drop the bad segments from the retry
                let retry_hall_count: usize = retry_hallucinated
                    .iter()
                    .map(|&(s, e)| e - s)
                    .sum();
                tracing::warn!(
                    "Chunk {}: retry still has {} hallucinated segments, dropping them",
                    chunk_idx + 1,
                    retry_hall_count
                );
                chunk_segments = drop_hallucinated(retry_segments, &retry_hallucinated);
            }
        }

        // Offset timestamps by chunk start time
        for seg in &mut chunk_segments {
            seg.start += chunk_start;
            seg.end += chunk_start;
            for tok in &mut seg.tokens {
                tok.start += chunk_start;
                tok.end += chunk_start;
            }
        }

        all_segments.extend(chunk_segments);
    }

    // Deduplicate segments in overlap regions
    let deduped = deduplicate_overlaps(&all_segments, &boundaries);

    // Renumber segment IDs sequentially
    let final_segments: Vec<TranscriptSegment> = deduped
        .into_iter()
        .enumerate()
        .map(|(i, mut seg)| {
            seg.id = i as u32;
            seg
        })
        .collect();

    Ok(final_segments)
}

/// Transcribe a single chunk of audio with anti-hallucination parameters.
fn transcribe_chunk(
    ctx: &WhisperContext,
    chunk_samples: &[f32],
    no_context: bool,
) -> Result<Vec<TranscriptSegment>> {
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_language(Some("en"));
    params.set_token_timestamps(true);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_special(false);
    params.set_print_timestamps(false);

    // Anti-hallucination parameters
    params.set_entropy_thold(2.4);
    params.set_no_speech_thold(0.6);
    params.set_suppress_blank(true);
    params.set_suppress_nst(true);
    params.set_initial_prompt("Piano masterclass. Teacher gives feedback on student performance.");

    if no_context {
        params.set_no_context(true);
    }

    // Fresh state per chunk guarantees decoder reset
    let mut state = ctx
        .create_state()
        .map_err(|e| anyhow::anyhow!("Failed to create whisper state: {}", e))?;

    state
        .full(params, chunk_samples)
        .map_err(|e| anyhow::anyhow!("Whisper transcription failed: {}", e))?;

    let num_segments = state.full_n_segments();
    let mut segments = Vec::new();

    for i in 0..num_segments {
        let segment = state
            .get_segment(i)
            .with_context(|| format!("Segment {} out of bounds", i))?;

        let text = segment
            .to_str_lossy()
            .map_err(|e| anyhow::anyhow!("Failed to get segment {} text: {}", i, e))?
            .into_owned();

        let start_cs = segment.start_timestamp();
        let end_cs = segment.end_timestamp();

        let start = start_cs as f64 / 100.0;
        let end = end_cs as f64 / 100.0;

        let num_tokens = segment.n_tokens();

        let mut tokens = Vec::new();
        for j in 0..num_tokens {
            let token = match segment.get_token(j) {
                Some(t) => t,
                None => continue,
            };

            let token_text = match token.to_str_lossy() {
                Ok(t) => t.into_owned(),
                Err(_) => continue,
            };

            if token_text.starts_with("<|") || token_text.starts_with('[') {
                continue;
            }

            let token_data = token.token_data();
            let prob = token.token_probability();

            tokens.push(TranscriptToken {
                text: token_text,
                start: token_data.t0 as f64 / 100.0,
                end: token_data.t1 as f64 / 100.0,
                probability: prob,
            });
        }

        let corrected_text = apply_corrections(&text);

        segments.push(TranscriptSegment {
            id: i as u32,
            text: corrected_text,
            start,
            end,
            tokens,
        });
    }

    Ok(segments)
}

/// Detect runs of repetitive segments that indicate hallucination loops.
///
/// Returns a list of (start_idx, end_idx) ranges where consecutive segments
/// have >SIMILARITY_THRESHOLD text similarity for >= REPETITION_THRESHOLD in a row.
fn detect_repetition(segments: &[TranscriptSegment]) -> Vec<(usize, usize)> {
    if segments.len() < config::REPETITION_THRESHOLD {
        return Vec::new();
    }

    let mut ranges: Vec<(usize, usize)> = Vec::new();
    let mut run_start = 0;
    let mut run_len = 1;

    for i in 1..segments.len() {
        let sim = text_similarity(&segments[i - 1].text, &segments[i].text);
        if sim >= config::SIMILARITY_THRESHOLD {
            run_len += 1;
        } else {
            if run_len >= config::REPETITION_THRESHOLD {
                ranges.push((run_start, run_start + run_len));
            }
            run_start = i;
            run_len = 1;
        }
    }
    if run_len >= config::REPETITION_THRESHOLD {
        ranges.push((run_start, run_start + run_len));
    }

    ranges
}

/// Drop segments that fall within hallucinated ranges.
fn drop_hallucinated(
    segments: Vec<TranscriptSegment>,
    hallucinated: &[(usize, usize)],
) -> Vec<TranscriptSegment> {
    segments
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !hallucinated.iter().any(|&(s, e)| *i >= s && *i < e))
        .map(|(_, seg)| seg)
        .collect()
}

/// Normalized text similarity using longest common substring ratio.
/// Returns 0.0..1.0 where 1.0 means identical texts.
fn text_similarity(a: &str, b: &str) -> f64 {
    let a = a.trim().to_lowercase();
    let b = b.trim().to_lowercase();

    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    if a == b {
        return 1.0;
    }

    let a_bytes = a.as_bytes();
    let b_bytes = b.as_bytes();
    let max_len = a_bytes.len().max(b_bytes.len());

    // Longest common substring length via DP
    let mut prev = vec![0u16; b_bytes.len() + 1];
    let mut curr = vec![0u16; b_bytes.len() + 1];
    let mut lcs_len: u16 = 0;

    for i in 1..=a_bytes.len() {
        for j in 1..=b_bytes.len() {
            if a_bytes[i - 1] == b_bytes[j - 1] {
                curr[j] = prev[j - 1] + 1;
                if curr[j] > lcs_len {
                    lcs_len = curr[j];
                }
            } else {
                curr[j] = 0;
            }
        }
        std::mem::swap(&mut prev, &mut curr);
        curr.iter_mut().for_each(|v| *v = 0);
    }

    lcs_len as f64 / max_len as f64
}

/// Deduplicate segments from overlapping chunk regions.
///
/// For segments that fall in overlap zones, prefer the segment from the chunk
/// where it is further from the chunk boundary (more "central" to that chunk).
fn deduplicate_overlaps(
    segments: &[TranscriptSegment],
    boundaries: &[(f64, f64)],
) -> Vec<TranscriptSegment> {
    if boundaries.len() <= 1 {
        return segments.to_vec();
    }

    // For each pair of adjacent chunks, the overlap region is:
    //   [next_chunk_start, prev_chunk_end]
    // i.e., [boundaries[i+1].0, boundaries[i].1]
    let mut overlap_regions: Vec<(f64, f64)> = Vec::new();
    for i in 0..boundaries.len() - 1 {
        let overlap_start = boundaries[i + 1].0;
        let overlap_end = boundaries[i].1;
        if overlap_end > overlap_start {
            overlap_regions.push((overlap_start, overlap_end));
        }
    }

    if overlap_regions.is_empty() {
        return segments.to_vec();
    }

    // For each overlap region, compute the midpoint - segments before midpoint
    // belong to the earlier chunk, segments at/after midpoint to the later chunk.
    let midpoints: Vec<f64> = overlap_regions
        .iter()
        .map(|&(s, e)| (s + e) / 2.0)
        .collect();

    let mut result: Vec<TranscriptSegment> = Vec::new();

    for seg in segments {
        let seg_mid = (seg.start + seg.end) / 2.0;
        let mut dominated = false;

        for (region_idx, &(ovl_start, ovl_end)) in overlap_regions.iter().enumerate() {
            if seg_mid >= ovl_start && seg_mid <= ovl_end {
                // This segment is in an overlap region.
                // Check if there's already a segment in result that covers similar time.
                let midpoint = midpoints[region_idx];

                // For the first chunk's copy: keep if seg_mid < midpoint
                // For the second chunk's copy: keep if seg_mid >= midpoint
                // But we need to know which chunk this segment came from.
                // Since segments are ordered by time and chunks are processed sequentially,
                // if we already have a segment close to this one, this is a duplicate.
                let has_duplicate = result.iter().any(|existing| {
                    let existing_mid = (existing.start + existing.end) / 2.0;
                    (existing_mid - seg_mid).abs() < 1.0
                        && text_similarity(&existing.text, &seg.text) > 0.5
                });

                if has_duplicate {
                    // Check which one to keep based on distance from midpoint
                    // The segment closer to the center of its chunk is preferred.
                    // For simplicity: earlier segments are from chunk N,
                    // later duplicates from chunk N+1. Keep whichever is further
                    // from the overlap midpoint (i.e., more central to its chunk).
                    let dist_from_mid = (seg_mid - midpoint).abs();
                    let existing_idx = result.iter().position(|existing| {
                        let existing_mid = (existing.start + existing.end) / 2.0;
                        (existing_mid - seg_mid).abs() < 1.0
                            && text_similarity(&existing.text, &seg.text) > 0.5
                    });
                    if let Some(idx) = existing_idx {
                        let existing_mid = (result[idx].start + result[idx].end) / 2.0;
                        let existing_dist = (existing_mid - midpoint).abs();
                        if dist_from_mid > existing_dist {
                            // New segment is further from boundary, prefer it
                            result[idx] = seg.clone();
                        }
                    }
                    dominated = true;
                    break;
                }
            }
        }

        if !dominated {
            result.push(seg.clone());
        }
    }

    result
}

fn read_wav_samples(path: &Path) -> Result<Vec<f32>> {
    let reader = hound::WavReader::open(path)
        .with_context(|| format!("Failed to open WAV file: {}", path.display()))?;

    let spec = reader.spec();
    tracing::debug!(
        "WAV: {}Hz, {} channels, {} bits, {:?}",
        spec.sample_rate,
        spec.channels,
        spec.bits_per_sample,
        spec.sample_format
    );

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<Result<Vec<f32>, _>>()
            .with_context(|| "Failed to read float samples")?,
        hound::SampleFormat::Int => {
            let max_val = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .collect::<Result<Vec<i32>, _>>()
                .with_context(|| "Failed to read int samples")?
                .into_iter()
                .map(|s| s as f32 / max_val)
                .collect()
        }
    };

    // If stereo, downmix to mono
    if spec.channels == 2 {
        let mono: Vec<f32> = samples
            .chunks(2)
            .map(|pair| {
                if pair.len() == 2 {
                    (pair[0] + pair[1]) / 2.0
                } else {
                    pair[0]
                }
            })
            .collect();
        Ok(mono)
    } else {
        Ok(samples)
    }
}

fn apply_corrections(text: &str) -> String {
    let mut result = text.to_string();
    for (wrong, correct) in CORRECTIONS {
        let lower = result.to_lowercase();
        let wrong_lower = wrong.to_lowercase();
        if let Some(pos) = lower.find(&wrong_lower) {
            let end = pos + wrong.len();
            result = format!("{}{}{}", &result[..pos], correct, &result[end..]);
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Whisper API transcription
// ---------------------------------------------------------------------------

use reqwest::multipart;

#[derive(serde::Deserialize)]
struct WhisperApiResponse {
    #[allow(dead_code)]
    text: String,
    segments: Option<Vec<WhisperApiSegment>>,
    words: Option<Vec<WhisperApiWord>>,
}

#[derive(serde::Deserialize)]
struct WhisperApiSegment {
    text: String,
    start: f64,
    end: f64,
}

#[derive(serde::Deserialize)]
struct WhisperApiWord {
    word: String,
    start: f64,
    end: f64,
}

/// Transcribe a video using the OpenAI Whisper API with chunked upload.
///
/// This mirrors the chunking strategy of the local `transcribe_video` function
/// (CHUNK_DURATION_SECS chunks with CHUNK_OVERLAP_SECS overlap), but sends each
/// chunk to the OpenAI API instead of running Whisper locally.
pub async fn transcribe_video_api(
    client: &reqwest::Client,
    api_key: &str,
    store: &MasterclassStore,
    video_id: &str,
) -> Result<Transcript> {
    let audio_path = store.audio_path(video_id);
    anyhow::ensure!(
        audio_path.exists(),
        "Audio file not found for {}. Run download first.",
        video_id
    );

    tracing::info!("Transcribing {} via Whisper API", video_id);

    let samples = read_wav_samples(&audio_path)?;
    let sample_rate = config::SAMPLE_RATE;
    let total_duration = samples.len() as f64 / sample_rate as f64;
    tracing::info!(
        "Loaded {:.1}s of audio ({} samples)",
        total_duration,
        samples.len()
    );

    let chunk_dur = config::CHUNK_DURATION_SECS;
    let overlap = config::CHUNK_OVERLAP_SECS;

    // Calculate chunk boundaries (same logic as local transcription)
    let mut boundaries: Vec<(f64, f64)> = Vec::new();
    let mut start = 0.0;
    while start < total_duration {
        let end = (start + chunk_dur).min(total_duration);
        boundaries.push((start, end));
        start += chunk_dur - overlap;
        if end >= total_duration {
            break;
        }
    }

    tracing::info!(
        "Splitting {:.0}s audio into {} chunks ({:.0}s each, {:.0}s overlap)",
        total_duration,
        boundaries.len(),
        chunk_dur,
        overlap
    );

    let mut all_segments: Vec<TranscriptSegment> = Vec::new();

    for (chunk_idx, &(chunk_start, chunk_end)) in boundaries.iter().enumerate() {
        let start_sample = (chunk_start * sample_rate as f64) as usize;
        let end_sample = ((chunk_end * sample_rate as f64) as usize).min(samples.len());
        let chunk_samples = &samples[start_sample..end_sample];

        tracing::info!(
            "Chunk {}/{}: {:.0}s - {:.0}s ({:.0}s)",
            chunk_idx + 1,
            boundaries.len(),
            chunk_start,
            chunk_end,
            chunk_end - chunk_start
        );

        let mut chunk_segments =
            transcribe_chunk_api(client, api_key, chunk_samples, sample_rate).await?;

        // Offset timestamps by chunk start time
        for seg in &mut chunk_segments {
            seg.start += chunk_start;
            seg.end += chunk_start;
            for tok in &mut seg.tokens {
                tok.start += chunk_start;
                tok.end += chunk_start;
            }
        }

        all_segments.extend(chunk_segments);
    }

    // Deduplicate segments in overlap regions
    let deduped = deduplicate_overlaps(&all_segments, &boundaries);

    // Renumber segment IDs sequentially
    let final_segments: Vec<TranscriptSegment> = deduped
        .into_iter()
        .enumerate()
        .map(|(i, mut seg)| {
            seg.id = i as u32;
            seg
        })
        .collect();

    tracing::info!(
        "Transcribed {} segments from {} via API",
        final_segments.len(),
        video_id
    );

    let transcript = Transcript {
        video_id: video_id.to_string(),
        model: "whisper-1-api".to_string(),
        language: "en".to_string(),
        transcribed_at: chrono::Utc::now().to_rfc3339(),
        segments: final_segments,
    };

    store.save_transcript(&transcript)?;
    Ok(transcript)
}

/// Write audio samples to a temporary WAV file for upload to the API.
fn write_temp_wav(samples: &[f32], sample_rate: u32) -> Result<tempfile::NamedTempFile> {
    let temp_file = tempfile::Builder::new()
        .suffix(".wav")
        .tempfile()
        .with_context(|| "Failed to create temporary WAV file")?;

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(temp_file.path(), spec)
        .with_context(|| "Failed to create WAV writer for temp file")?;

    for &sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let int_sample = (clamped * i16::MAX as f32) as i16;
        writer
            .write_sample(int_sample)
            .with_context(|| "Failed to write sample to temp WAV")?;
    }

    writer
        .finalize()
        .with_context(|| "Failed to finalize temp WAV file")?;

    Ok(temp_file)
}

pub const WHISPER_API_BASE_URL: &str = "https://api.openai.com";

/// Transcribe a single audio chunk via the OpenAI Whisper API.
async fn transcribe_chunk_api(
    client: &reqwest::Client,
    api_key: &str,
    samples: &[f32],
    sample_rate: u32,
) -> Result<Vec<TranscriptSegment>> {
    let temp_wav = write_temp_wav(samples, sample_rate)?;

    let wav_bytes = std::fs::read(temp_wav.path())
        .with_context(|| "Failed to read temp WAV file for upload")?;

    let file_part = multipart::Part::bytes(wav_bytes)
        .file_name("audio.wav")
        .mime_str("audio/wav")
        .with_context(|| "Failed to set MIME type on multipart part")?;

    let form = multipart::Form::new()
        .part("file", file_part)
        .text("model", "whisper-1")
        .text("response_format", "verbose_json")
        .text("timestamp_granularities[]", "word")
        .text("timestamp_granularities[]", "segment")
        .text("language", "en");

    let response = client
        .post("https://api.openai.com/v1/audio/transcriptions")
        .bearer_auth(api_key)
        .multipart(form)
        .send()
        .await
        .with_context(|| "Failed to send request to Whisper API")?;

    anyhow::ensure!(
        response.status().is_success(),
        "Whisper API returned error: HTTP {} - {}",
        response.status(),
        response
            .text()
            .await
            .unwrap_or_else(|_| "no body".to_string())
    );

    let api_response: WhisperApiResponse = response
        .json()
        .await
        .with_context(|| "Failed to parse Whisper API response")?;

    let api_segments = api_response.segments.unwrap_or_default();
    let api_words = api_response.words.unwrap_or_default();

    let mut segments: Vec<TranscriptSegment> = Vec::new();

    for (i, api_seg) in api_segments.iter().enumerate() {
        // Collect words that fall within this segment's time range
        let seg_tokens: Vec<TranscriptToken> = api_words
            .iter()
            .filter(|w| w.start >= api_seg.start && w.end <= api_seg.end)
            .map(|w| TranscriptToken {
                text: w.word.clone(),
                start: w.start,
                end: w.end,
                probability: 1.0,
            })
            .collect();

        let corrected_text = apply_corrections(&api_seg.text);

        segments.push(TranscriptSegment {
            id: i as u32,
            text: corrected_text,
            start: api_seg.start,
            end: api_seg.end,
            tokens: seg_tokens,
        });
    }

    Ok(segments)
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- text_similarity --

    #[test]
    fn text_similarity_identical() {
        assert_eq!(text_similarity("hello world", "hello world"), 1.0);
    }

    #[test]
    fn text_similarity_both_empty() {
        assert_eq!(text_similarity("", ""), 1.0);
    }

    #[test]
    fn text_similarity_one_empty() {
        assert_eq!(text_similarity("hello", ""), 0.0);
        assert_eq!(text_similarity("", "hello"), 0.0);
    }

    #[test]
    fn text_similarity_partial_match() {
        let sim = text_similarity("the quick brown fox", "the slow brown fox");
        assert!(sim > 0.3 && sim < 1.0, "sim = {}", sim);
    }

    #[test]
    fn text_similarity_case_insensitive() {
        assert_eq!(text_similarity("Hello", "hello"), 1.0);
    }

    // -- apply_corrections --

    #[test]
    fn apply_corrections_rubato() {
        assert!(apply_corrections("robot o phrase").contains("rubato"));
    }

    #[test]
    fn apply_corrections_no_match() {
        let input = "this has no known errors";
        assert_eq!(apply_corrections(input), input);
    }

    #[test]
    fn apply_corrections_chopin() {
        assert!(apply_corrections("show pan nocturne").contains("Chopin"));
    }

    // -- detect_repetition --

    fn make_seg(id: u32, text: &str) -> TranscriptSegment {
        TranscriptSegment {
            id,
            text: text.to_string(),
            start: id as f64,
            end: id as f64 + 1.0,
            tokens: vec![],
        }
    }

    #[test]
    fn detect_repetition_no_repeats() {
        let segments = vec![
            make_seg(0, "hello world"),
            make_seg(1, "something different"),
            make_seg(2, "another thing"),
        ];
        assert!(detect_repetition(&segments).is_empty());
    }

    #[test]
    fn detect_repetition_three_consecutive() {
        let segments = vec![
            make_seg(0, "normal text"),
            make_seg(1, "hallucinated text"),
            make_seg(2, "hallucinated text"),
            make_seg(3, "hallucinated text"),
            make_seg(4, "normal again"),
        ];
        let ranges = detect_repetition(&segments);
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0], (1, 4));
    }

    #[test]
    fn detect_repetition_multiple_runs() {
        let segments = vec![
            make_seg(0, "repeat A"),
            make_seg(1, "repeat A"),
            make_seg(2, "repeat A"),
            make_seg(3, "break"),
            make_seg(4, "repeat B"),
            make_seg(5, "repeat B"),
            make_seg(6, "repeat B"),
        ];
        let ranges = detect_repetition(&segments);
        assert_eq!(ranges.len(), 2);
    }

    #[test]
    fn detect_repetition_too_few_segments() {
        let segments = vec![make_seg(0, "a"), make_seg(1, "a")];
        assert!(detect_repetition(&segments).is_empty());
    }

    // -- drop_hallucinated --

    #[test]
    fn drop_hallucinated_removes_ranges() {
        let segments = vec![
            make_seg(0, "keep"),
            make_seg(1, "drop"),
            make_seg(2, "drop"),
            make_seg(3, "keep"),
        ];
        let result = drop_hallucinated(segments, &[(1, 3)]);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].text, "keep");
        assert_eq!(result[1].text, "keep");
    }

    #[test]
    fn drop_hallucinated_empty_ranges() {
        let segments = vec![make_seg(0, "keep"), make_seg(1, "keep")];
        let result = drop_hallucinated(segments, &[]);
        assert_eq!(result.len(), 2);
    }

    // -- deduplicate_overlaps --

    #[test]
    fn deduplicate_overlaps_single_boundary() {
        let segments = vec![make_seg(0, "first chunk content")];
        let boundaries = vec![(0.0, 300.0)];
        let result = deduplicate_overlaps(&segments, &boundaries);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn deduplicate_overlaps_two_boundaries() {
        let boundaries = vec![(0.0, 300.0), (270.0, 600.0)];
        let mut seg1 = make_seg(0, "overlap text here");
        seg1.start = 280.0;
        seg1.end = 285.0;
        let mut seg2 = make_seg(1, "overlap text here");
        seg2.start = 280.5;
        seg2.end = 285.5;
        let segments = vec![seg1, seg2];

        let result = deduplicate_overlaps(&segments, &boundaries);
        assert_eq!(result.len(), 1, "duplicate should be removed");
    }
}
