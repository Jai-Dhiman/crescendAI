use anyhow::{Context, Result};

use crate::audio_features;
use crate::config::SAMPLE_RATE;
use crate::schemas::*;
use crate::store::MasterclassStore;

/// Minimum segment duration in seconds to prevent flickering.
const MIN_SEGMENT_DURATION: f64 = 2.0;

/// Maximum gap between Playing and Talking to count as a stopping point.
const MAX_STOP_GAP: f64 = 3.0;

/// Energy threshold (dB) below which we consider silence.
const SILENCE_THRESHOLD_DB: f32 = -40.0;

/// Spectral centroid threshold (Hz) to distinguish piano from speech.
/// Piano typically has higher spectral centroid than speech.
const PIANO_CENTROID_THRESHOLD: f32 = 1500.0;

/// Harmonic ratio threshold to detect piano (piano has strong harmonic content).
const PIANO_HARMONIC_THRESHOLD: f32 = 0.5;

pub fn segment_video(store: &MasterclassStore, video_id: &str) -> Result<SegmentationResult> {
    let audio_path = store.audio_path(video_id);
    anyhow::ensure!(
        audio_path.exists(),
        "Audio file not found for {}",
        video_id
    );

    // Load transcript for speech detection
    let transcript = store
        .load_transcript(video_id)?
        .with_context(|| format!("Transcript not found for {}. Run transcribe first.", video_id))?;

    tracing::info!("Segmenting audio for {}", video_id);

    // Read audio samples
    let reader = hound::WavReader::open(&audio_path)
        .with_context(|| format!("Failed to open WAV: {}", audio_path.display()))?;
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<Result<Vec<f32>, _>>()
            .with_context(|| "Failed to read WAV samples")?,
        hound::SampleFormat::Int => {
            let max_val = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .collect::<Result<Vec<i32>, _>>()
                .with_context(|| "Failed to read WAV samples")?
                .into_iter()
                .map(|s| s as f32 / max_val)
                .collect()
        }
    };

    // Compute audio features
    let features = audio_features::compute_features(&samples);
    let num_frames = features.rms_db.len();

    // Build speech presence map from transcript tokens
    let speech_map = build_speech_map(&transcript, &features.frame_times);

    // Classify each frame
    let mut frame_labels: Vec<SegmentLabel> = Vec::with_capacity(num_frames);
    for i in 0..num_frames {
        let energy_db = features.rms_db[i];
        let centroid = features.spectral_centroid[i];
        let harmonic = features.harmonic_ratio[i];
        let has_speech = speech_map[i];

        let label = classify_frame(energy_db, centroid, harmonic, has_speech);
        frame_labels.push(label);
    }

    // Convert frame labels to timed segments
    let raw_segments = frames_to_segments(&frame_labels, &features);

    // Post-process: enforce minimum duration and merge adjacent same-label
    let merged = merge_short_segments(raw_segments);
    let segments = merge_adjacent_same_label(merged);

    // Find stopping points (Playing -> Talking transitions)
    let stopping_points = find_stopping_points(&segments);

    tracing::info!(
        "Found {} segments, {} stopping points for {}",
        segments.len(),
        stopping_points.len(),
        video_id
    );

    let result = SegmentationResult {
        video_id: video_id.to_string(),
        segments,
        stopping_points,
        segmented_at: chrono::Utc::now().to_rfc3339(),
    };

    store.save_segmentation(&result)?;
    Ok(result)
}

/// Build a boolean map indicating speech presence at each frame time.
/// Uses transcript tokens with probability > 0.5 as speech indicators.
fn build_speech_map(transcript: &Transcript, frame_times: &[f64]) -> Vec<bool> {
    let mut speech = vec![false; frame_times.len()];

    for segment in &transcript.segments {
        // Use segment-level timing as fallback
        let seg_start = segment.start;
        let seg_end = segment.end;

        // Check if any tokens have good probability
        let has_confident_tokens = segment
            .tokens
            .iter()
            .any(|t| t.probability > 0.5);

        if !has_confident_tokens && segment.text.trim().is_empty() {
            continue;
        }

        // Mark frames in this segment as speech
        for (i, &t) in frame_times.iter().enumerate() {
            if t >= seg_start && t <= seg_end {
                speech[i] = true;
            }
        }
    }

    speech
}

/// Classify a single frame based on features and speech presence.
fn classify_frame(
    energy_db: f32,
    centroid: f32,
    harmonic_ratio: f32,
    has_speech: bool,
) -> SegmentLabel {
    let is_low_energy = energy_db < SILENCE_THRESHOLD_DB;

    if is_low_energy {
        return SegmentLabel::Silence;
    }

    let is_piano_like = centroid > PIANO_CENTROID_THRESHOLD && harmonic_ratio > PIANO_HARMONIC_THRESHOLD;

    match (has_speech, is_piano_like) {
        (true, true) => SegmentLabel::Mixed,
        (true, false) => SegmentLabel::Talking,
        (false, true) => SegmentLabel::Playing,
        (false, false) => {
            // No speech detected and not piano-like -- could be quiet passage or ambient
            if energy_db > -25.0 {
                // Moderate energy without speech: likely playing
                SegmentLabel::Playing
            } else {
                SegmentLabel::Silence
            }
        }
    }
}

/// Convert per-frame labels into continuous segments with features.
fn frames_to_segments(
    labels: &[SegmentLabel],
    features: &audio_features::AudioFeatures,
) -> Vec<AudioSegmentLabel> {
    if labels.is_empty() {
        return Vec::new();
    }

    let mut segments = Vec::new();
    let mut current_label = &labels[0];
    let mut seg_start = features.frame_times[0];
    let mut seg_energy_sum = 0.0f32;
    let mut seg_centroid_sum = 0.0f32;
    let mut seg_frame_count = 0u32;

    for i in 0..labels.len() {
        if &labels[i] != current_label {
            // Emit segment
            let avg_energy = seg_energy_sum / seg_frame_count.max(1) as f32;
            let avg_centroid = seg_centroid_sum / seg_frame_count.max(1) as f32;
            segments.push(AudioSegmentLabel {
                start: seg_start,
                end: features.frame_times[i],
                label: current_label.clone(),
                confidence: 0.8,
                energy_db: avg_energy,
                spectral_centroid_hz: avg_centroid,
            });

            current_label = &labels[i];
            seg_start = features.frame_times[i];
            seg_energy_sum = 0.0;
            seg_centroid_sum = 0.0;
            seg_frame_count = 0;
        }

        seg_energy_sum += features.rms_db[i];
        seg_centroid_sum += features.spectral_centroid[i];
        seg_frame_count += 1;
    }

    // Emit final segment
    if seg_frame_count > 0 {
        let last_time = features.frame_times.last().copied().unwrap_or(seg_start);
        // Add one frame duration to the end
        let frame_duration = crate::config::HOP_SIZE as f64 / SAMPLE_RATE as f64;
        let avg_energy = seg_energy_sum / seg_frame_count as f32;
        let avg_centroid = seg_centroid_sum / seg_frame_count as f32;
        segments.push(AudioSegmentLabel {
            start: seg_start,
            end: last_time + frame_duration,
            label: current_label.clone(),
            confidence: 0.8,
            energy_db: avg_energy,
            spectral_centroid_hz: avg_centroid,
        });
    }

    segments
}

/// Merge segments shorter than MIN_SEGMENT_DURATION into their neighbors.
fn merge_short_segments(mut segments: Vec<AudioSegmentLabel>) -> Vec<AudioSegmentLabel> {
    if segments.len() <= 1 {
        return segments;
    }

    let mut changed = true;
    while changed {
        changed = false;
        let mut i = 0;
        while i < segments.len() {
            let duration = segments[i].end - segments[i].start;
            if duration < MIN_SEGMENT_DURATION && segments.len() > 1 {
                // Merge with the neighbor that has the same label, or the longer neighbor
                if i > 0 && (i >= segments.len() - 1 || segments[i - 1].label == segments[i].label)
                {
                    segments[i - 1].end = segments[i].end;
                    segments.remove(i);
                    changed = true;
                    continue;
                } else if i < segments.len() - 1 {
                    segments[i + 1].start = segments[i].start;
                    segments.remove(i);
                    changed = true;
                    continue;
                }
            }
            i += 1;
        }
    }

    segments
}

/// Merge consecutive segments with the same label.
fn merge_adjacent_same_label(segments: Vec<AudioSegmentLabel>) -> Vec<AudioSegmentLabel> {
    if segments.is_empty() {
        return segments;
    }

    let mut merged = Vec::new();
    let mut current = segments[0].clone();

    for seg in segments.into_iter().skip(1) {
        if seg.label == current.label {
            current.end = seg.end;
            // Weighted average of features
            let total_dur = current.end - current.start;
            let seg_dur = seg.end - seg.start;
            if total_dur > 0.0 {
                let w = seg_dur as f32 / total_dur as f32;
                current.energy_db = current.energy_db * (1.0 - w) + seg.energy_db * w;
                current.spectral_centroid_hz =
                    current.spectral_centroid_hz * (1.0 - w) + seg.spectral_centroid_hz * w;
            }
        } else {
            merged.push(current);
            current = seg;
        }
    }
    merged.push(current);

    merged
}

/// Identify stopping points: transitions from Playing to Talking
/// (allowing up to MAX_STOP_GAP seconds of Silence between them).
fn find_stopping_points(segments: &[AudioSegmentLabel]) -> Vec<StoppingPoint> {
    let mut stops = Vec::new();

    for i in 0..segments.len() {
        if segments[i].label != SegmentLabel::Playing {
            continue;
        }

        let playing_seg = &segments[i];

        // Look for a Talking segment after this Playing segment (allow Silence gap)
        let mut talking_idx = None;
        let mut gap_end = playing_seg.end;

        for j in (i + 1)..segments.len() {
            match segments[j].label {
                SegmentLabel::Talking | SegmentLabel::Mixed => {
                    if segments[j].start - gap_end <= MAX_STOP_GAP {
                        talking_idx = Some(j);
                    }
                    break;
                }
                SegmentLabel::Silence => {
                    gap_end = segments[j].end;
                    if gap_end - playing_seg.end > MAX_STOP_GAP {
                        break;
                    }
                }
                SegmentLabel::Playing => break,
            }
        }

        if let Some(talk_i) = talking_idx {
            let talking_seg = &segments[talk_i];

            // Find the end of the talking section (may span multiple talking segments)
            let mut talk_end = talking_seg.end;
            for k in (talk_i + 1)..segments.len() {
                match segments[k].label {
                    SegmentLabel::Talking | SegmentLabel::Mixed => {
                        talk_end = segments[k].end;
                    }
                    SegmentLabel::Silence => {
                        // Allow short silence within a feedback section
                        if segments[k].end - segments[k].start < 5.0 {
                            continue;
                        }
                        break;
                    }
                    SegmentLabel::Playing => break,
                }
            }

            stops.push(StoppingPoint {
                timestamp: playing_seg.end,
                playing_start: playing_seg.start,
                playing_end: playing_seg.end,
                talking_start: talking_seg.start,
                talking_end: talk_end,
            });
        }
    }

    stops
}
