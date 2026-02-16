use anyhow::{Context, Result};
use serde::Deserialize;

use crate::llm_client::LlmClient;
use crate::schemas::*;
use crate::store::MasterclassStore;

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

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
        tracing::info!("Transcript has no segments for {}, skipping", video_id);
        store.save_teaching_moments(video_id, &[])?;
        return Ok(Vec::new());
    }

    // Pass 1: detect moments from the full transcript
    let all_detections = detect_moments(client, &transcript).await?;

    // Filter out moments with invalid timestamps
    let detections: Vec<_> = all_detections
        .into_iter()
        .filter(|m| {
            if m.timestamp < 0.0 || m.timestamp > video.duration_seconds {
                tracing::warn!(
                    "Dropping moment at {:.1}s: outside video duration (0..{:.0}s)",
                    m.timestamp,
                    video.duration_seconds
                );
                false
            } else {
                true
            }
        })
        .collect();

    tracing::info!(
        "Pass 1 detected {} teaching moments in {}",
        detections.len(),
        video_id
    );

    if detections.is_empty() {
        store.save_teaching_moments(video_id, &[])?;
        return Ok(Vec::new());
    }

    let total_stops = detections.len() as u32;
    let mut all_moments = Vec::new();

    // Pass 2: extract structured data for each detected moment
    for (idx, detection) in detections.iter().enumerate() {
        tracing::info!(
            "  Pass 2: extracting moment {}/{} at {:.1}s",
            idx + 1,
            total_stops,
            detection.timestamp
        );

        let context_window = build_context_window(&transcript, detection.timestamp, 120.0);

        if context_window.is_empty() {
            tracing::warn!(
                "  Moment {}: empty context window at {:.1}s, skipping",
                idx + 1,
                detection.timestamp
            );
            continue;
        }

        let (playing_start, playing_end) =
            estimate_playing_bounds(&transcript, detection.timestamp);

        let raw = match extract_moment(client, &video, &context_window, detection, total_stops)
            .await
        {
            Ok(r) => r,
            Err(e) => {
                tracing::error!(
                    "  Moment {}: extraction failed: {}",
                    idx + 1,
                    e
                );
                continue;
            }
        };

        let duration = raw.duration_estimate.unwrap_or(30.0).clamp(5.0, 300.0);

        let moment = TeachingMoment {
            moment_id: uuid::Uuid::new_v4().to_string(),
            video_id: video_id.to_string(),
            video_title: video.title.clone(),
            teacher: video
                .teacher
                .clone()
                .unwrap_or_else(|| "Unknown".to_string()),

            stop_timestamp: detection.timestamp,
            feedback_start: detection.timestamp,
            feedback_end: detection.timestamp + duration,
            playing_before_start: playing_start,
            playing_before_end: playing_end,

            transcript_text: context_window.clone(),
            feedback_summary: raw.feedback_summary,
            musical_dimension: raw.musical_dimension,
            secondary_dimensions: raw.secondary_dimensions,
            severity: raw.severity,
            feedback_type: raw.feedback_type,

            piece: raw.piece.or_else(|| video.pieces.first().cloned()),
            composer: raw.composer.or_else(|| video.composers.first().cloned()),
            passage_description: raw.passage_description,
            student_level: raw.student_level,

            stop_order: (idx + 1) as u32,
            total_stops,
            time_spent_seconds: duration,
            demonstrated: raw.demonstrated,

            extracted_at: chrono::Utc::now().to_rfc3339(),
            extraction_model: client.model().to_string(),
            confidence: raw.confidence,
        };

        all_moments.push(moment);
    }

    if all_moments.is_empty() && total_stops > 0 {
        anyhow::bail!(
            "Extracted 0 moments from {} detected stops -- LLM output format not usable",
            total_stops
        );
    }

    store.save_teaching_moments(video_id, &all_moments)?;
    Ok(all_moments)
}

// ---------------------------------------------------------------------------
// Pass 1 -- Moment detection
// ---------------------------------------------------------------------------

#[derive(Deserialize, Debug)]
pub struct DetectedMoment {
    pub timestamp: f64,
    pub description: String,
}

async fn detect_moments(
    client: &LlmClient,
    transcript: &Transcript,
) -> Result<Vec<DetectedMoment>> {
    let formatted = format_transcript_with_timestamps(transcript);

    if formatted.len() > 80_000 {
        // Chunk into 10-minute windows with 1-minute overlap
        tracing::info!(
            "Transcript too long ({} chars), chunking into 10-minute windows",
            formatted.len()
        );
        let mut all_moments = Vec::new();
        let duration = transcript
            .segments
            .last()
            .map(|s| s.end)
            .unwrap_or(0.0);

        let window_secs = 600.0; // 10 minutes
        let overlap_secs = 60.0; // 1 minute
        let mut start = 0.0_f64;

        while start < duration {
            let end = (start + window_secs).min(duration);
            let chunk: Vec<&TranscriptSegment> = transcript
                .segments
                .iter()
                .filter(|s| s.start >= start && s.start < end)
                .collect();

            if !chunk.is_empty() {
                let chunk_text: String = chunk
                    .iter()
                    .map(|s| format!("[{:.1}s] {}", s.start, s.text.trim()))
                    .collect::<Vec<_>>()
                    .join("\n");

                match detect_moments_single(client, &chunk_text).await {
                    Ok(moments) => all_moments.extend(moments),
                    Err(e) => {
                        tracing::warn!(
                            "Chunk {:.0}-{:.0}s detection failed: {}",
                            start,
                            end,
                            e
                        );
                    }
                }
            }

            start += window_secs - overlap_secs;
        }

        // Deduplicate moments within 10 seconds of each other
        all_moments.sort_by(|a, b| a.timestamp.partial_cmp(&b.timestamp).unwrap());
        let mut deduped: Vec<DetectedMoment> = Vec::new();
        for m in all_moments {
            if deduped
                .last()
                .map(|prev: &DetectedMoment| (m.timestamp - prev.timestamp).abs() > 10.0)
                .unwrap_or(true)
            {
                deduped.push(m);
            }
        }

        Ok(deduped)
    } else {
        let mut moments = detect_moments_single(client, &formatted).await?;
        moments.sort_by(|a, b| a.timestamp.partial_cmp(&b.timestamp).unwrap());
        Ok(moments)
    }
}

async fn detect_moments_single(
    client: &LlmClient,
    transcript_text: &str,
) -> Result<Vec<DetectedMoment>> {
    let system_prompt = r#"You are an expert at analyzing piano masterclass transcripts. Your task is to find all teaching moments -- places where a teacher stops a student to give feedback.

A teaching moment IS:
- Teacher stops the student to give a correction
- Teacher provides specific musical feedback or suggestion
- Teacher demonstrates how a passage should be played
- Teacher explains a musical concept or technique
- Teacher gives detailed guidance on interpretation, dynamics, phrasing, etc.

A teaching moment is NOT:
- Casual conversation or greetings
- Introductions or applause
- Simply saying "good" or "very nice" without elaboration
- Administrative comments (page turns, stand adjustments, etc.)
- Audience questions unrelated to the performance

Respond with a JSON array of objects, each with:
- "timestamp": the approximate time in seconds where the teaching moment begins
- "description": a brief description of what the teacher is addressing

Example response:
[
  {"timestamp": 45.2, "description": "Teacher stops student to address dynamics in the opening phrase"},
  {"timestamp": 123.8, "description": "Teacher demonstrates proper pedaling technique for the transition"}
]

If there are no teaching moments, respond with an empty array: []"#;

    let user_prompt = format!(
        "Find all teaching moments in this piano masterclass transcript:\n\n{}",
        transcript_text
    );

    let response = client.message(system_prompt, &user_prompt).await?;
    parse_detected_moments(&response)
}

fn parse_detected_moments(response: &str) -> Result<Vec<DetectedMoment>> {
    let trimmed = response.trim();

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
        let start = trimmed
            .find('[')
            .with_context(|| "No JSON array found in response")?;
        find_json_array(&trimmed[start..])?
    };

    let moments: Vec<DetectedMoment> = serde_json::from_str(json_str)
        .with_context(|| {
            format!(
                "Failed to parse detected moments JSON: {}...",
                &json_str[..json_str.len().min(300)]
            )
        })?;

    Ok(moments)
}

/// Find the first complete JSON array (matching brackets) in the string.
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

// ---------------------------------------------------------------------------
// Pass 2 -- Moment extraction
// ---------------------------------------------------------------------------

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
Total teaching stops in this masterclass: {}
Detected moment at: {:.1}s
Detection description: {}

Transcript context around this moment:
{}"#,
        video.title,
        video.teacher.as_deref().unwrap_or("Unknown"),
        total_stops,
        detection.timestamp,
        detection.description,
        context_window,
    );

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
                    tracing::warn!(
                        "  Extraction parse failed (attempt {}): {}",
                        attempt + 1,
                        e
                    );
                }
            },
            Err(e) => {
                tracing::error!("  LLM request failed (attempt {}): {}", attempt + 1, e);
            }
        }
    }

    anyhow::bail!(
        "Failed to extract moment at {:.1}s after retries",
        detection.timestamp
    )
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

The "duration_estimate" field should be your estimate of how many seconds the teacher spends on this teaching moment (typically 10-120 seconds)."#
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
        let start = trimmed
            .find('{')
            .with_context(|| "No JSON object found in response")?;
        find_json_object(&trimmed[start..])?
    };

    let extraction: RawExtraction = serde_json::from_str(json_str).with_context(|| {
        format!(
            "Failed to parse JSON: {}...",
            &json_str[..json_str.len().min(300)]
        )
    })?;

    Ok(extraction)
}

/// Find the first complete JSON object (matching braces) in the string.
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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn format_transcript_with_timestamps(transcript: &Transcript) -> String {
    transcript
        .segments
        .iter()
        .map(|s| format!("[{:.1}s] {}", s.start, s.text.trim()))
        .collect::<Vec<_>>()
        .join("\n")
}

fn build_context_window(
    transcript: &Transcript,
    timestamp: f64,
    window_secs: f64,
) -> String {
    let start = (timestamp - window_secs).max(0.0);
    let end = timestamp + window_secs;

    transcript
        .segments
        .iter()
        .filter(|s| s.start >= start && s.start <= end)
        .map(|s| format!("[{:.1}s] {}", s.start, s.text.trim()))
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn estimate_playing_bounds(transcript: &Transcript, stop_timestamp: f64) -> (f64, f64) {
    let lookback = 120.0;
    let start_bound = (stop_timestamp - lookback).max(0.0);

    // Collect segments in the lookback window, sorted by start time
    let mut relevant: Vec<&TranscriptSegment> = transcript
        .segments
        .iter()
        .filter(|s| s.start >= start_bound && s.end <= stop_timestamp)
        .collect();

    relevant.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap());

    // Find the largest gap (>3 seconds) between consecutive segments
    let mut largest_gap = 0.0_f64;
    let mut gap_start = stop_timestamp - 10.0;
    let mut gap_end = stop_timestamp;

    for pair in relevant.windows(2) {
        let gap = pair[1].start - pair[0].end;
        if gap > 3.0 && gap > largest_gap {
            largest_gap = gap;
            gap_start = pair[0].end;
            gap_end = pair[1].start;
        }
    }

    if largest_gap <= 3.0 {
        // No significant gap found, use default
        ((stop_timestamp - 10.0).max(0.0), stop_timestamp)
    } else {
        (gap_start, gap_end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- find_json_array --

    #[test]
    fn find_json_array_valid() {
        let result = find_json_array(r#"[{"a": 1}, {"b": 2}]"#).unwrap();
        assert_eq!(result, r#"[{"a": 1}, {"b": 2}]"#);
    }

    #[test]
    fn find_json_array_nested() {
        let result = find_json_array(r#"[[1, 2], [3]]"#).unwrap();
        assert_eq!(result, r#"[[1, 2], [3]]"#);
    }

    #[test]
    fn find_json_array_strings_with_bracket() {
        let result = find_json_array(r#"["a]b", "c"]"#).unwrap();
        assert_eq!(result, r#"["a]b", "c"]"#);
    }

    #[test]
    fn find_json_array_unterminated() {
        assert!(find_json_array("[1, 2, 3").is_err());
    }

    #[test]
    fn find_json_array_not_array() {
        assert!(find_json_array("{\"a\": 1}").is_err());
    }

    // -- parse_detected_moments --

    #[test]
    fn parse_detected_moments_clean_json() {
        let input = r#"[{"timestamp": 45.2, "description": "dynamics"}, {"timestamp": 100.0, "description": "pedal"}]"#;
        let moments = parse_detected_moments(input).unwrap();
        assert_eq!(moments.len(), 2);
        assert!((moments[0].timestamp - 45.2).abs() < 0.01);
    }

    #[test]
    fn parse_detected_moments_code_fenced() {
        let input = "```json\n[{\"timestamp\": 10.0, \"description\": \"test\"}]\n```";
        let moments = parse_detected_moments(input).unwrap();
        assert_eq!(moments.len(), 1);
    }

    #[test]
    fn parse_detected_moments_leading_text() {
        let input = "Here are the moments:\n[{\"timestamp\": 20.0, \"description\": \"feedback\"}]";
        let moments = parse_detected_moments(input).unwrap();
        assert_eq!(moments.len(), 1);
    }

    #[test]
    fn parse_detected_moments_empty_array() {
        let moments = parse_detected_moments("[]").unwrap();
        assert!(moments.is_empty());
    }

    #[test]
    fn parse_detected_moments_malformed() {
        assert!(parse_detected_moments("not json at all").is_err());
    }

    // -- format_transcript_with_timestamps --

    #[test]
    fn format_transcript_empty() {
        let transcript = Transcript {
            video_id: "v1".into(),
            model: "m".into(),
            language: "en".into(),
            transcribed_at: "now".into(),
            segments: vec![],
        };
        assert_eq!(format_transcript_with_timestamps(&transcript), "");
    }

    fn make_seg(id: u32, text: &str, start: f64, end: f64) -> TranscriptSegment {
        TranscriptSegment {
            id,
            text: text.to_string(),
            start,
            end,
            tokens: vec![],
        }
    }

    fn make_transcript(segments: Vec<TranscriptSegment>) -> Transcript {
        Transcript {
            video_id: "v1".into(),
            model: "m".into(),
            language: "en".into(),
            transcribed_at: "now".into(),
            segments,
        }
    }

    #[test]
    fn format_transcript_single_segment() {
        let transcript = make_transcript(vec![make_seg(0, "hello", 1.0, 2.0)]);
        let result = format_transcript_with_timestamps(&transcript);
        assert!(result.contains("[1.0s] hello"));
    }

    #[test]
    fn format_transcript_multiple_segments() {
        let transcript = make_transcript(vec![
            make_seg(0, "first", 0.0, 1.0),
            make_seg(1, "second", 1.0, 2.0),
        ]);
        let result = format_transcript_with_timestamps(&transcript);
        assert!(result.contains("[0.0s] first"));
        assert!(result.contains("[1.0s] second"));
    }

    // -- build_context_window --

    #[test]
    fn build_context_window_segments_in_range() {
        let transcript = make_transcript(vec![
            make_seg(0, "before", 10.0, 15.0),
            make_seg(1, "target", 50.0, 55.0),
            make_seg(2, "after", 90.0, 95.0),
        ]);
        let result = build_context_window(&transcript, 50.0, 10.0);
        assert!(result.contains("target"));
        assert!(!result.contains("before"));
        assert!(!result.contains("after"));
    }

    #[test]
    fn build_context_window_empty_transcript() {
        let transcript = make_transcript(vec![]);
        let result = build_context_window(&transcript, 50.0, 10.0);
        assert!(result.is_empty());
    }

    #[test]
    fn build_context_window_no_segments_in_range() {
        let transcript = make_transcript(vec![make_seg(0, "far away", 500.0, 505.0)]);
        let result = build_context_window(&transcript, 50.0, 10.0);
        assert!(result.is_empty());
    }

    // -- estimate_playing_bounds --

    #[test]
    fn estimate_playing_bounds_no_segments() {
        let transcript = make_transcript(vec![]);
        let (start, end) = estimate_playing_bounds(&transcript, 100.0);
        assert!((start - 90.0).abs() < 0.01);
        assert!((end - 100.0).abs() < 0.01);
    }

    #[test]
    fn estimate_playing_bounds_clear_gap() {
        let transcript = make_transcript(vec![
            make_seg(0, "talk", 30.0, 35.0),
            make_seg(1, "play", 45.0, 90.0),
            make_seg(2, "stop", 91.0, 95.0),
        ]);
        let (start, end) = estimate_playing_bounds(&transcript, 100.0);
        // Largest gap is between seg 0 end (35) and seg 1 start (45) = 10s gap
        assert!((start - 35.0).abs() < 0.01);
        assert!((end - 45.0).abs() < 0.01);
    }

    #[test]
    fn estimate_playing_bounds_lookback_limit() {
        let transcript = make_transcript(vec![
            make_seg(0, "very early", 5.0, 10.0),
        ]);
        // stop at 200 => lookback starts at 80, seg at 5-10 is outside lookback
        let (start, end) = estimate_playing_bounds(&transcript, 200.0);
        assert!((start - 190.0).abs() < 0.01);
        assert!((end - 200.0).abs() < 0.01);
    }

    // -- find_json_object (from identify.rs) --

    #[test]
    fn find_json_object_valid() {
        let result = find_json_object(r#"{"key": "value"}"#).unwrap();
        assert_eq!(result, r#"{"key": "value"}"#);
    }

    #[test]
    fn find_json_object_nested() {
        let result = find_json_object(r#"{"a": {"b": 1}}"#).unwrap();
        assert_eq!(result, r#"{"a": {"b": 1}}"#);
    }

    #[test]
    fn find_json_object_escaped_braces_in_string() {
        let result = find_json_object(r#"{"a": "hello {world}"}"#).unwrap();
        assert_eq!(result, r#"{"a": "hello {world}"}"#);
    }

    #[test]
    fn find_json_object_unterminated() {
        assert!(find_json_object(r#"{"key": "value""#).is_err());
    }

    // -- timestamp validation (parse_detected_moments allows negative, caller filters) --

    #[test]
    fn parse_detected_moments_negative_timestamp() {
        let input = r#"[{"timestamp": -5.0, "description": "bad"}, {"timestamp": 10.0, "description": "good"}]"#;
        let moments = parse_detected_moments(input).unwrap();
        // Parser doesn't filter; caller does
        assert_eq!(moments.len(), 2);
        assert!(moments[0].timestamp < 0.0);
    }

    // -- duration clamping --

    #[test]
    fn duration_clamp_zero() {
        let clamped: f64 = 0.0_f64.clamp(5.0, 300.0);
        assert_eq!(clamped, 5.0);
    }

    #[test]
    fn duration_clamp_large() {
        let clamped: f64 = 5000.0_f64.clamp(5.0, 300.0);
        assert_eq!(clamped, 300.0);
    }

    #[test]
    fn duration_clamp_none_default() {
        let val: Option<f64> = None;
        let clamped = val.unwrap_or(30.0).clamp(5.0, 300.0);
        assert_eq!(clamped, 30.0);
    }

    #[test]
    fn duration_clamp_normal() {
        let clamped: f64 = 45.0_f64.clamp(5.0, 300.0);
        assert_eq!(clamped, 45.0);
    }
}
