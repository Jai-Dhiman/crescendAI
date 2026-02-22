use anyhow::{Context, Result};
use serde::Deserialize;

use crate::config;
use crate::llm_client::{LlmClient, ResponseFormat};
use crate::schemas::*;
use crate::store::MasterclassStore;

/// Check if text contains any forbidden physical-mechanism vocabulary.
///
/// Uses word-boundary-aware matching: the characters immediately before and after
/// the match must be non-alphanumeric (or string boundaries). This prevents
/// "warm" from matching "arm" or "fingertip" from matching "finger".
///
/// Returns the first matching forbidden term, or None if clean.
pub fn contains_forbidden_vocabulary(text: &str) -> Option<&'static str> {
    let text_lower = text.to_lowercase();
    for &forbidden in config::FORBIDDEN_OPEN_DESCRIPTION_WORDS {
        let forbidden_lower = forbidden.to_lowercase();
        let mut search_from = 0;
        while let Some(pos) = text_lower[search_from..].find(&forbidden_lower) {
            let abs_pos = search_from + pos;
            let end_pos = abs_pos + forbidden_lower.len();

            // Check word boundary before match
            let before_ok = abs_pos == 0
                || !text_lower.as_bytes()[abs_pos - 1].is_ascii_alphanumeric();
            // Check word boundary after match
            let after_ok = end_pos >= text_lower.len()
                || !text_lower.as_bytes()[end_pos].is_ascii_alphanumeric();

            if before_ok && after_ok {
                return Some(forbidden);
            }

            search_from = abs_pos + 1;
        }
    }
    None
}

/// Build a short correction prompt for a forbidden vocabulary violation.
pub fn build_forbidden_word_correction_prompt(
    original: &str,
    forbidden_term: &str,
) -> String {
    format!(
        "Your open_description '{}' contains forbidden physical term '{}'. \
         Rewrite to name the AUDIBLE MUSICAL QUALITY a listener would hear. \
         2-5 words. No physical terms. Respond with ONLY the corrected string.",
        original, forbidden_term
    )
}

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
    let response_format = build_response_format();
    let mut all_moments = Vec::new();

    // Compute stop groups (shared playing_before windows)
    let group_ids = compute_stop_groups(&segmentation.stopping_points);
    let group_sizes = compute_group_sizes(&group_ids);
    let group_positions = compute_group_positions(&group_ids);

    // Track extracted feedback summaries for group context
    let mut extracted_summaries: Vec<Option<String>> = vec![None; total_stops as usize];

    // Process one stop at a time
    for (idx, stop) in segmentation.stopping_points.iter().enumerate() {
        let stop_order = idx as u32;
        let group_id = group_ids[idx];
        let group_size = group_sizes[idx];
        let stop_in_group = group_positions[idx];

        // Build group context from previously extracted stops in same group
        let group_context = if stop_in_group > 1 {
            let mut prior_summaries = Vec::new();
            for prev_idx in 0..idx {
                if group_ids[prev_idx] == group_id {
                    if let Some(ref summary) = extracted_summaries[prev_idx] {
                        prior_summaries.push(format!(
                            "- Point {}: {}",
                            group_positions[prev_idx],
                            summary
                        ));
                    }
                }
            }
            if prior_summaries.is_empty() {
                None
            } else {
                Some(format!(
                    "This is teaching point {} of {} from the same interruption.\n\
                     Previous points from this interruption:\n\
                     {}\n\n\
                     Focus on what NEW aspect the teacher is addressing in this specific point.",
                    stop_in_group,
                    group_size,
                    prior_summaries.join("\n")
                ))
            }
        } else {
            None
        };

        let ctx = build_stop_context(stop, &transcript, group_context.as_deref());

        tracing::info!(
            "  Stop {}/{} (group {}, {}/{})",
            idx + 1,
            total_stops,
            group_id,
            stop_in_group,
            group_size
        );

        let user_prompt = build_single_stop_prompt(&video, &ctx, stop_order + 1, total_stops);

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

            match client
                .message_structured(
                    &system_prompt,
                    &prompt,
                    Some(response_format.clone()),
                    Some(0.0),
                )
                .await
            {
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

        if let Some(mut extracted) = raw {
            // Check open_description for forbidden vocabulary and attempt correction
            if let Some(ref desc) = extracted.open_description {
                if let Some(forbidden_term) = contains_forbidden_vocabulary(desc) {
                    tracing::warn!(
                        "  Stop {}: open_description '{}' contains forbidden term '{}'",
                        idx + 1,
                        desc,
                        forbidden_term
                    );
                    let correction_prompt =
                        build_forbidden_word_correction_prompt(desc, forbidden_term);
                    match client
                        .message_structured(
                            "You rewrite descriptions to name audible musical qualities only.",
                            &correction_prompt,
                            None,
                            Some(0.0),
                        )
                        .await
                    {
                        Ok(corrected) => {
                            let corrected = corrected.trim().trim_matches('"').to_string();
                            if contains_forbidden_vocabulary(&corrected).is_none() {
                                tracing::info!(
                                    "  Stop {}: corrected '{}' -> '{}'",
                                    idx + 1,
                                    desc,
                                    corrected
                                );
                                extracted.open_description = Some(corrected);
                            } else {
                                tracing::warn!(
                                    "  Stop {}: correction '{}' still forbidden, keeping original",
                                    idx + 1,
                                    corrected
                                );
                            }
                        }
                        Err(e) => {
                            tracing::warn!(
                                "  Stop {}: correction LLM call failed: {}, keeping original",
                                idx + 1,
                                e
                            );
                        }
                    }
                }
            }

            extracted_summaries[idx] = Some(extracted.feedback_summary.clone());

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
                musical_dimension: config::normalize_dimension(&extracted.musical_dimension),
                secondary_dimensions: extracted
                    .secondary_dimensions
                    .iter()
                    .map(|d| config::normalize_dimension(d))
                    .collect(),
                severity: extracted.severity,
                feedback_type: extracted.feedback_type,

                piece: extracted.piece.or_else(|| video.pieces.first().cloned()),
                composer: extracted.composer.or_else(|| video.composers.first().cloned()),
                passage_description: extracted.passage_description,
                student_level: extracted.student_level
                    .or_else(|| video.student_level.clone()),

                stop_order: stop_order + 1,
                total_stops,
                stop_group: group_id,
                stop_in_group,
                group_size,
                time_spent_seconds: stop.talking_end - stop.talking_start,
                demonstrated: extracted.demonstrated,

                extracted_at: chrono::Utc::now().to_rfc3339(),
                extraction_model: client.model().to_string(),
                confidence: extracted.confidence,

                open_description: extracted.open_description,
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
    group_context: Option<String>,
}

fn build_stop_context(
    stop: &StoppingPoint,
    transcript: &Transcript,
    group_context: Option<&str>,
) -> StopContext {
    // Teacher feedback window
    let mut feedback_parts = Vec::new();
    for seg in &transcript.segments {
        if seg.end >= stop.talking_start && seg.start <= stop.talking_end {
            feedback_parts.push(seg.text.trim().to_string());
        }
    }

    // Earlier context: 120s before playing_start (broader teaching flow)
    let earlier_start = (stop.playing_start - config::PRE_CONTEXT_SECS).max(0.0);
    let before_start = (stop.playing_start - 30.0).max(0.0);
    let mut earlier_context = Vec::new();
    for seg in &transcript.segments {
        if seg.end >= earlier_start && seg.start < before_start {
            earlier_context.push(seg.text.trim().to_string());
        }
    }

    // Immediate pre-context: 30s before playing_start
    let mut before_context = Vec::new();
    for seg in &transcript.segments {
        if seg.end >= before_start && seg.start <= stop.playing_start {
            before_context.push(seg.text.trim().to_string());
        }
    }

    // Post-context: 30s after talking_end
    let post_end = stop.talking_end + config::POST_CONTEXT_SECS;
    let mut after_context = Vec::new();
    for seg in &transcript.segments {
        if seg.start >= stop.talking_end && seg.end <= post_end {
            after_context.push(seg.text.trim().to_string());
        }
    }

    // Build labeled transcript
    let mut parts = Vec::new();
    if !earlier_context.is_empty() {
        parts.push(format!("[Earlier context] {}", earlier_context.join(" ")));
    }
    if !before_context.is_empty() {
        parts.push(format!("[Before stop] {}", before_context.join(" ")));
    }
    parts.push(format!("[Teacher feedback] {}", feedback_parts.join(" ")));
    if !after_context.is_empty() {
        parts.push(format!("[After feedback] {}", after_context.join(" ")));
    }

    StopContext {
        transcript_text: parts.join("\n"),
        group_context: group_context.map(|s| s.to_string()),
    }
}

fn compute_stop_groups(stops: &[StoppingPoint]) -> Vec<u32> {
    let mut groups = Vec::with_capacity(stops.len());
    let mut current_group = 1u32;

    for (i, stop) in stops.iter().enumerate() {
        if i == 0 {
            groups.push(current_group);
            continue;
        }
        let prev = &stops[i - 1];
        // Same playing window = same group
        if (stop.playing_start - prev.playing_start).abs() < 1.0
            && (stop.playing_end - prev.playing_end).abs() < 1.0
        {
            groups.push(current_group);
        } else {
            current_group += 1;
            groups.push(current_group);
        }
    }
    groups
}

fn compute_group_sizes(group_ids: &[u32]) -> Vec<u32> {
    let mut sizes = vec![0u32; group_ids.len()];
    for (i, &gid) in group_ids.iter().enumerate() {
        let count = group_ids.iter().filter(|&&g| g == gid).count() as u32;
        sizes[i] = count;
    }
    sizes
}

fn compute_group_positions(group_ids: &[u32]) -> Vec<u32> {
    let mut positions = Vec::with_capacity(group_ids.len());
    for (i, &gid) in group_ids.iter().enumerate() {
        let pos = group_ids[..i].iter().filter(|&&g| g == gid).count() as u32 + 1;
        positions.push(pos);
    }
    positions
}

fn build_system_prompt() -> String {
    r#"You are a musicologist specializing in piano pedagogy research. You analyze masterclass transcripts to identify what MUSICAL QUALITY a teacher is addressing when they stop a student.

Teachers often explain through physical metaphor (arm weight, finger position, wrist rotation). Your expertise is translating these into the AUDIBLE MUSICAL QUALITY the teacher wants to change in the sound.

Respond with ONLY a single JSON object. No explanation, no markdown fences, just the JSON.

Required JSON fields:
{
  "open_description": "2-5 word description of the musical quality being addressed",
  "feedback_summary": "1-2 sentence summary of the teacher's feedback and its musical purpose",
  "musical_dimension": "one of: dynamics, timing, articulation, pedaling, tone_color, phrasing, voicing, interpretation, technique, structure",
  "secondary_dimensions": [],
  "severity": "one of: minor, moderate, significant, critical",
  "feedback_type": "one of: correction, suggestion, demonstration, praise, explanation, comparison",
  "piece": null,
  "composer": null,
  "passage_description": "specific passage, measure, or section being addressed",
  "student_level": null,
  "demonstrated": false,
  "confidence": 0.85
}

## OPEN_DESCRIPTION RULES (most important field)

This field names the AUDIBLE QUALITY a listener would perceive in the recording.

CONSTRAINT: Could a blind listener identify this quality from the recording alone?
If yes, the description is correct. If it describes something visible or physical, rewrite to name the sonic result.

Before writing open_description, determine:
1. Is the teacher describing a physical action or a musical result?
2. If physical, what change in SOUND is the physical action intended to produce?
3. Name that sonic change in 2-5 words.

PREFERRED VOCABULARY: singing, resonant, percussive, bright, dark, warm, round, legato, connected, flowing, smooth, choppy, arc, direction, climax, contour, line, projection, prominence, blend, separation, layering, momentum, rubato, flexibility, steadiness, urgency, clarity, muddy, distinct, transparent, gradient, contrast, shaping, tapering, depth, warmth, brilliance, evenness

FORBIDDEN in open_description: finger, hand, arm, wrist, elbow, shoulder, weight, rotation, movement, position, posture, key depth, key speed, sliding, knuckle, pad, tip, bench, sitting, body

EXAMPLES (showing physical-to-musical translation):
  Teacher: "Use more arm weight, let gravity do the work"
  -> open_description: "singing tone depth"
  (NOT "arm weight transfer" -- that describes the mechanism, not the sound)

  Teacher: "Slide the finger pad across, don't strike from above"
  -> open_description: "smooth connected tone"
  (NOT "finger pad sliding motion" -- not audible)

  Teacher: "Your left hand is drowning out the melody"
  -> open_description: "melodic voice projection"

  Teacher: "Don't wait before the C, carry through"
  -> open_description: "phrase continuity at resolution"

  Teacher: "The syncopations should have suppressed excitement"
  -> open_description: "restrained syncopation intensity"

## DIMENSION DEFINITIONS
- dynamics: Volume level, crescendo/diminuendo, dynamic contrast, projection
- timing: Tempo, rhythm, rubato, rushing, dragging, pulse consistency
- articulation: Legato, staccato, accents, note connection, attack character
- pedaling: Sustain pedal usage, clarity vs blur, damper timing
- tone_color: Timbre quality, brightness/darkness, warmth, resonance
- phrasing: Musical line shape, direction, breathing, arc, continuity
- voicing: Balance between voices/hands, melodic prominence, inner voice clarity
- interpretation: Expressive character, emotion, stylistic authenticity, musical intent
- technique: Physical approach serving a musical goal (use when the musical result is inseparable from the physical method)
- structure: Formal awareness, section transitions, thematic connections

## SEVERITY CALIBRATION
- minor: Polishing refinement. Student is close. Brief comment, <20 seconds.
- moderate: Noticeable issue affecting the musical result. Teacher explains but moves on. 20-60 seconds.
- significant: Meaningfully compromises the passage. Teacher dwells on it, may demonstrate. 60+ seconds.
- critical: Fundamental misunderstanding. Teacher stops immediately and spends extended time re-teaching.

## FEEDBACK_TYPE GUIDANCE
- correction: Teacher identifies something wrong and asks for change
- suggestion: Teacher proposes an alternative approach, student choice implied
- demonstration: Teacher plays or sings to show the desired result
- praise: Teacher affirms something done well
- explanation: Teacher explains a concept without requesting immediate change
- comparison: Teacher contrasts two approaches (better/worse, different interpretations)"#
        .to_string()
}

fn build_single_stop_prompt(
    video: &VideoMetadata,
    ctx: &StopContext,
    stop_order: u32,
    total_stops: u32,
) -> String {
    let piece_info = video
        .pieces
        .first()
        .map(|p| format!("Piece: {}\n", p))
        .unwrap_or_default();
    let composer_info = video
        .composers
        .first()
        .map(|c| format!("Composer: {}\n", c))
        .unwrap_or_default();
    let level_info = video
        .student_level
        .as_ref()
        .map(|l| format!("Student level: {}\n", l))
        .unwrap_or_default();
    let group_info = ctx
        .group_context
        .as_ref()
        .map(|g| format!("\n{}\n", g))
        .unwrap_or_default();

    format!(
        r#"Video: {title}
Teacher: {teacher}
{piece}{composer}{level}This is stop {stop_order} of {total_stops} in this masterclass.
{group}
Transcript of this teaching moment:
{transcript}

Respond with a single JSON object describing this teaching moment."#,
        title = video.title,
        teacher = video.teacher.as_deref().unwrap_or("Unknown"),
        piece = piece_info,
        composer = composer_info,
        level = level_info,
        stop_order = stop_order,
        total_stops = total_stops,
        group = group_info,
        transcript = ctx.transcript_text,
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
    #[serde(default)]
    open_description: Option<String>,
}

fn default_confidence() -> f32 {
    0.7
}

fn build_response_format() -> ResponseFormat {
    ResponseFormat {
        format_type: "json_schema".to_string(),
        json_schema: Some(serde_json::json!({
            "name": "teaching_moment_extraction",
            "strict": true,
            "schema": {
                "type": "object",
                "properties": {
                    "open_description": { "type": "string" },
                    "feedback_summary": { "type": "string" },
                    "musical_dimension": {
                        "type": "string",
                        "enum": [
                            "dynamics", "timing", "articulation", "pedaling",
                            "tone_color", "phrasing", "voicing", "interpretation",
                            "technique", "structure"
                        ]
                    },
                    "secondary_dimensions": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "dynamics", "timing", "articulation", "pedaling",
                                "tone_color", "phrasing", "voicing", "interpretation",
                                "technique", "structure"
                            ]
                        }
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["minor", "moderate", "significant", "critical"]
                    },
                    "feedback_type": {
                        "type": "string",
                        "enum": [
                            "correction", "suggestion", "demonstration",
                            "praise", "explanation", "comparison"
                        ]
                    },
                    "piece": { "type": ["string", "null"] },
                    "composer": { "type": ["string", "null"] },
                    "passage_description": { "type": ["string", "null"] },
                    "student_level": { "type": ["string", "null"] },
                    "demonstrated": { "type": "boolean" },
                    "confidence": { "type": "number" }
                },
                "required": [
                    "open_description", "feedback_summary", "musical_dimension",
                    "secondary_dimensions", "severity", "feedback_type",
                    "piece", "composer", "passage_description", "student_level",
                    "demonstrated", "confidence"
                ],
                "additionalProperties": false
            }
        })),
    }
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

    // -- compute_stop_groups --

    fn make_stop(playing_start: f64, playing_end: f64, talking_start: f64) -> StoppingPoint {
        StoppingPoint {
            timestamp: talking_start,
            playing_start,
            playing_end,
            talking_start,
            talking_end: talking_start + 30.0,
        }
    }

    #[test]
    fn stop_groups_all_different() {
        let stops = vec![
            make_stop(10.0, 20.0, 25.0),
            make_stop(50.0, 60.0, 65.0),
            make_stop(100.0, 110.0, 115.0),
        ];
        let groups = compute_stop_groups(&stops);
        assert_eq!(groups, vec![1, 2, 3]);
    }

    #[test]
    fn stop_groups_shared_window() {
        let stops = vec![
            make_stop(10.0, 20.0, 25.0),
            make_stop(10.0, 20.0, 55.0),
            make_stop(10.0, 20.0, 85.0),
            make_stop(50.0, 60.0, 65.0),
        ];
        let groups = compute_stop_groups(&stops);
        assert_eq!(groups, vec![1, 1, 1, 2]);
    }

    #[test]
    fn stop_groups_sizes_and_positions() {
        let stops = vec![
            make_stop(10.0, 20.0, 25.0),
            make_stop(10.0, 20.0, 55.0),
            make_stop(50.0, 60.0, 65.0),
        ];
        let groups = compute_stop_groups(&stops);
        let sizes = compute_group_sizes(&groups);
        let positions = compute_group_positions(&groups);
        assert_eq!(sizes, vec![2, 2, 1]);
        assert_eq!(positions, vec![1, 2, 1]);
    }

    #[test]
    fn stop_groups_near_threshold() {
        // 0.5s difference should still be same group (< 1.0 threshold)
        let stops = vec![
            make_stop(10.0, 20.0, 25.0),
            make_stop(10.5, 20.5, 55.0),
        ];
        let groups = compute_stop_groups(&stops);
        assert_eq!(groups, vec![1, 1]);
    }

    #[test]
    fn stop_groups_empty() {
        let stops: Vec<StoppingPoint> = vec![];
        let groups = compute_stop_groups(&stops);
        assert!(groups.is_empty());
    }

    // -- contains_forbidden_vocabulary --

    #[test]
    fn forbidden_vocab_clean_description() {
        assert!(contains_forbidden_vocabulary("singing tone depth").is_none());
    }

    #[test]
    fn forbidden_vocab_arm_detected() {
        assert_eq!(
            contains_forbidden_vocabulary("arm weight transfer"),
            Some("arm")
        );
    }

    #[test]
    fn forbidden_vocab_warm_does_not_match_arm() {
        assert!(contains_forbidden_vocabulary("warm resonance").is_none());
    }

    #[test]
    fn forbidden_vocab_key_depth_multi_word() {
        assert_eq!(
            contains_forbidden_vocabulary("key depth adjustment"),
            Some("key depth")
        );
    }

    #[test]
    fn forbidden_vocab_hand_detected() {
        assert_eq!(
            contains_forbidden_vocabulary("left hand evenness"),
            Some("hand")
        );
    }

    #[test]
    fn forbidden_vocab_fingertip_does_not_match_finger() {
        // "fingertip" has no word boundary after "finger" -- 't' follows
        assert!(contains_forbidden_vocabulary("fingertip precision").is_none());
    }

    #[test]
    fn forbidden_vocab_case_insensitive() {
        assert_eq!(
            contains_forbidden_vocabulary("Arm Weight"),
            Some("arm")
        );
    }

    // -- build_forbidden_word_correction_prompt --

    #[test]
    fn correction_prompt_contains_terms() {
        let prompt = build_forbidden_word_correction_prompt("arm weight transfer", "arm");
        assert!(prompt.contains("arm weight transfer"));
        assert!(prompt.contains("arm"));
        assert!(prompt.contains("AUDIBLE MUSICAL QUALITY"));
    }
}
