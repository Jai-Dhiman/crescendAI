use anyhow::{Context, Result};
use std::path::Path;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

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

    // Read WAV file
    let samples = read_wav_samples(&audio_path)?;
    tracing::info!(
        "Loaded {:.1}s of audio ({} samples)",
        samples.len() as f64 / crate::config::SAMPLE_RATE as f64,
        samples.len()
    );

    // Configure whisper
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_language(Some("en"));
    params.set_token_timestamps(true);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_special(false);
    params.set_print_timestamps(false);

    // Run transcription
    let mut state = ctx
        .create_state()
        .map_err(|e| anyhow::anyhow!("Failed to create whisper state: {}", e))?;

    state
        .full(params, &samples)
        .map_err(|e| anyhow::anyhow!("Whisper transcription failed: {}", e))?;

    // Extract segments and tokens
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

        // Timestamps are in centiseconds (hundredths of a second)
        let start_cs = segment.start_timestamp();
        let end_cs = segment.end_timestamp();

        let start = start_cs as f64 / 100.0;
        let end = end_cs as f64 / 100.0;

        // Extract token-level data
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

            // Skip special tokens (they start with '<|' or '[')
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
        // Case-insensitive replacement
        let lower = result.to_lowercase();
        let wrong_lower = wrong.to_lowercase();
        if let Some(pos) = lower.find(&wrong_lower) {
            let end = pos + wrong.len();
            result = format!("{}{}{}", &result[..pos], correct, &result[end..]);
        }
    }
    result
}
