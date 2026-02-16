use anyhow::{Context, Result};
use serde::Deserialize;

use crate::llm_client::LlmClient;
use crate::schemas::*;
use crate::store::MasterclassStore;

pub async fn extract_teaching_moments(
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

    let segmentation = store
        .load_segmentation(video_id)?
        .with_context(|| format!("Segmentation not found for {}. Run segment first.", video_id))?;

    if segmentation.stopping_points.is_empty() {
        tracing::info!("No stopping points found for {}, skipping extraction", video_id);
        store.save_teaching_moments(video_id, &[])?;
        return Ok(Vec::new());
    }

    let total_stops = segmentation.stopping_points.len() as u32;
    tracing::info!(
        "Extracting from {} stopping points in {}",
        total_stops,
        video_id
    );

    let system_prompt = build_system_prompt();
    let mut all_moments = Vec::new();

    // Process one stop at a time -- much more reliable for local models
    for (idx, stop) in segmentation.stopping_points.iter().enumerate() {
        let stop_order = idx as u32;
        let ctx = build_stop_context(stop, stop_order, &transcript);

        tracing::info!("  Stop {}/{}", idx + 1, total_stops);

        let user_prompt = build_single_stop_prompt(&video, &ctx, total_stops);

        let mut raw: Option<RawExtraction> = None;

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
                Ok(response) => {
                    match parse_single_extraction(&response) {
                        Ok(extracted) => {
                            raw = Some(extracted);
                            break;
                        }
                        Err(e) => {
                            tracing::warn!(
                                "  Stop {}: parse failed (attempt {}): {}",
                                idx + 1,
                                attempt + 1,
                                e
                            );
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("  Stop {}: LLM request failed: {}", idx + 1, e);
                }
            }
        }

        if let Some(extracted) = raw {
            let moment = TeachingMoment {
                moment_id: uuid::Uuid::new_v4().to_string(),
                video_id: video_id.to_string(),
                video_title: video.title.clone(),
                teacher: video.teacher.clone().unwrap_or_else(|| "Unknown".to_string()),

                stop_timestamp: stop.timestamp,
                feedback_start: stop.talking_start,
                feedback_end: stop.talking_end,
                playing_before_start: stop.playing_start,
                playing_before_end: stop.playing_end,

                transcript_text: ctx.transcript_text.clone(),
                feedback_summary: extracted.feedback_summary,
                musical_dimension: extracted.musical_dimension,
                secondary_dimensions: extracted.secondary_dimensions,
                severity: extracted.severity,
                feedback_type: extracted.feedback_type,

                piece: extracted.piece.or_else(|| video.pieces.first().cloned()),
                composer: extracted.composer.or_else(|| video.composers.first().cloned()),
                passage_description: extracted.passage_description,
                student_level: extracted.student_level,

                stop_order: stop_order + 1,
                total_stops,
                time_spent_seconds: stop.talking_end - stop.talking_start,
                demonstrated: extracted.demonstrated,

                extracted_at: chrono::Utc::now().to_rfc3339(),
                extraction_model: client.model().to_string(),
                confidence: extracted.confidence,
            };

            all_moments.push(moment);
        } else {
            tracing::error!("  Stop {}: failed after retries, skipping", idx + 1);
        }
    }

    if all_moments.is_empty() && total_stops > 0 {
        anyhow::bail!(
            "Extracted 0 moments from {} stopping points -- LLM output format not usable",
            total_stops
        );
    }

    store.save_teaching_moments(video_id, &all_moments)?;
    Ok(all_moments)
}

struct StopContext {
    transcript_text: String,
}

fn build_stop_context(
    stop: &StoppingPoint,
    _stop_order: u32,
    transcript: &Transcript,
) -> StopContext {
    let mut text_parts = Vec::new();
    for seg in &transcript.segments {
        if seg.end >= stop.talking_start && seg.start <= stop.talking_end {
            text_parts.push(seg.text.trim().to_string());
        }
    }

    let context_start = (stop.playing_start - 30.0).max(0.0);
    let mut pre_context = Vec::new();
    for seg in &transcript.segments {
        if seg.end >= context_start && seg.start <= stop.playing_start {
            pre_context.push(seg.text.trim().to_string());
        }
    }

    let transcript_text = if pre_context.is_empty() {
        text_parts.join(" ")
    } else {
        format!(
            "[Before stop] {} [Teacher feedback] {}",
            pre_context.join(" "),
            text_parts.join(" ")
        )
    };

    StopContext { transcript_text }
}

fn build_system_prompt() -> String {
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
  "confidence": 0.7
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
- structure: form, sections, development"#
        .to_string()
}

fn build_single_stop_prompt(
    video: &VideoMetadata,
    ctx: &StopContext,
    total_stops: u32,
) -> String {
    format!(
        r#"Video: {}
Teacher: {}
Total stops in this masterclass: {}

Transcript of this teaching moment:
{}

Respond with a single JSON object describing this teaching moment."#,
        video.title,
        video.teacher.as_deref().unwrap_or("Unknown"),
        total_stops,
        ctx.transcript_text,
    )
}

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
}

fn default_confidence() -> f32 {
    0.7
}

fn parse_single_extraction(response: &str) -> Result<RawExtraction> {
    let trimmed = response.trim();

    // Find the JSON object in the response
    let json_str = if trimmed.starts_with('{') {
        // Find matching closing brace
        find_json_object(trimmed)?
    } else if trimmed.starts_with("```") {
        // Strip code fences
        let inner = trimmed
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim();
        find_json_object(inner)?
    } else {
        // Try to find a JSON object somewhere in the text
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

#[cfg(test)]
mod tests {
    use super::*;

    // -- find_json_object --

    #[test]
    fn find_json_object_valid() {
        let result = find_json_object(r#"{"key": "value"}"#).unwrap();
        assert_eq!(result, r#"{"key": "value"}"#);
    }

    #[test]
    fn find_json_object_nested() {
        let result = find_json_object(r#"{"a": {"b": [1, 2]}}"#).unwrap();
        assert_eq!(result, r#"{"a": {"b": [1, 2]}}"#);
    }

    #[test]
    fn find_json_object_escaped_braces() {
        let result = find_json_object(r#"{"a": "x}y"}"#).unwrap();
        assert_eq!(result, r#"{"a": "x}y"}"#);
    }

    #[test]
    fn find_json_object_unterminated() {
        assert!(find_json_object(r#"{"key": "value""#).is_err());
    }

    // -- parse_single_extraction --

    #[test]
    fn parse_single_extraction_valid() {
        let input = r#"{"feedback_summary": "Work on dynamics", "musical_dimension": "dynamics", "secondary_dimensions": [], "severity": "moderate", "feedback_type": "correction", "piece": null, "composer": null, "passage_description": null, "student_level": null, "demonstrated": false, "confidence": 0.8}"#;
        let result = parse_single_extraction(input).unwrap();
        assert_eq!(result.feedback_summary, "Work on dynamics");
        assert_eq!(result.musical_dimension, "dynamics");
    }

    #[test]
    fn parse_single_extraction_code_fenced() {
        let input = "```json\n{\"feedback_summary\": \"test\", \"musical_dimension\": \"timing\", \"severity\": \"minor\", \"feedback_type\": \"suggestion\"}\n```";
        let result = parse_single_extraction(input).unwrap();
        assert_eq!(result.musical_dimension, "timing");
    }

    #[test]
    fn parse_single_extraction_embedded_in_prose() {
        let input = "Here is the analysis:\n{\"feedback_summary\": \"Nice phrasing\", \"musical_dimension\": \"phrasing\", \"severity\": \"minor\", \"feedback_type\": \"praise\"}\nThat's my analysis.";
        let result = parse_single_extraction(input).unwrap();
        assert_eq!(result.musical_dimension, "phrasing");
    }

    #[test]
    fn parse_single_extraction_missing_fields() {
        let input = r#"{"feedback_summary": "test"}"#;
        assert!(parse_single_extraction(input).is_err());
    }
}
