use crate::models::{Citation, CitedFeedback, Performance, PerformanceDimensions, RetrievalResult, SourceType};
use regex::Regex;

#[cfg(feature = "ssr")]
use serde::Deserialize;
#[cfg(feature = "ssr")]
use worker::Env;

/// The LLM model to use for feedback generation
#[cfg(feature = "ssr")]
const LLM_MODEL: &str = "@cf/meta/llama-3.3-70b-instruct-fp8-fast";

/// Response format from Workers AI text generation
#[cfg(feature = "ssr")]
#[derive(Deserialize)]
struct TextGenerationResponse {
    response: String,
}

/// Format a source reference for the prompt
fn format_source_reference(result: &RetrievalResult, index: usize) -> String {
    let chunk = &result.chunk;
    let source_info = match chunk.source_type {
        SourceType::Book | SourceType::Letter | SourceType::Journal => {
            let mut info = format!("{} by {}", chunk.source_title, chunk.source_author);
            if let Some(page) = chunk.page_number {
                info.push_str(&format!(", p.{}", page));
            }
            info
        }
        SourceType::Masterclass => {
            let mut info = format!("{} - {}", chunk.source_author, chunk.source_title);
            if let Some(ts) = chunk.timestamp_start {
                let mins = (ts / 60.0).floor() as i32;
                let secs = (ts % 60.0).floor() as i32;
                info.push_str(&format!(" ({:02}:{:02})", mins, secs));
            }
            info
        }
    };

    format!("[{}] {}\n\"{}\"", index + 1, source_info, chunk.text)
}

/// Build the LLM prompt for cited feedback generation
fn build_feedback_prompt(
    performance: &Performance,
    dimensions: &PerformanceDimensions,
    retrieved_chunks: &[RetrievalResult],
) -> String {
    let mut prompt = String::new();

    // System instruction
    prompt.push_str("You are an expert piano teacher providing feedback on a student's performance.\n\n");

    // Performance context
    prompt.push_str("## Performance Analysis\n");
    prompt.push_str(&format!("Performer: {}\n", performance.performer));
    prompt.push_str(&format!("Piece: {} by {}\n\n", performance.piece_title, performance.composer));

    // Dimension scores
    prompt.push_str("Dimension Scores (0.0-1.0 scale):\n");
    for (label, score) in dimensions.to_labeled_vec() {
        prompt.push_str(&format!("- {}: {:.2}\n", label, score));
    }
    prompt.push('\n');

    // Reference sources
    prompt.push_str("## Reference Sources\n");
    for (i, result) in retrieved_chunks.iter().enumerate() {
        prompt.push_str(&format_source_reference(result, i));
        prompt.push_str("\n\n");
    }

    // Instructions
    prompt.push_str("## Instructions\n");
    prompt.push_str("Write 2-3 paragraphs of personalized feedback.\n\n");
    prompt.push_str("CRITICAL: Include inline citations using [1], [2], etc. when referencing ");
    prompt.push_str("advice from the sources. Every specific piece of advice should be grounded ");
    prompt.push_str("in a source.\n\n");
    prompt.push_str("Focus on:\n");
    prompt.push_str("1. One specific strength to celebrate (with citation if applicable)\n");
    prompt.push_str("2. One or two areas for improvement with actionable practice suggestions\n");
    prompt.push_str("3. A composer-specific insight relevant to this piece\n\n");
    prompt.push_str("Write in an encouraging but specific tone. Be concrete about what you ");
    prompt.push_str("observed in the performance data.");

    prompt
}

/// Parse citation markers [N] from LLM response and map to source metadata
fn parse_citations(response_text: &str, retrieved_chunks: &[RetrievalResult]) -> Vec<i32> {
    let citation_regex = Regex::new(r"\[(\d+)\]").unwrap();
    let mut cited_numbers: Vec<i32> = Vec::new();

    for cap in citation_regex.captures_iter(response_text) {
        if let Some(num_str) = cap.get(1) {
            if let Ok(num) = num_str.as_str().parse::<i32>() {
                // Only include valid citation numbers (1-indexed, within range)
                if num >= 1 && num <= retrieved_chunks.len() as i32 && !cited_numbers.contains(&num) {
                    cited_numbers.push(num);
                }
            }
        }
    }

    cited_numbers.sort();
    cited_numbers
}

/// Convert plain text with [N] markers to HTML with citation links
fn text_to_html_with_citations(text: &str) -> String {
    let citation_regex = Regex::new(r"\[(\d+)\]").unwrap();

    citation_regex
        .replace_all(text, |caps: &regex::Captures| {
            let num = &caps[1];
            format!(
                "<button class=\"citation-marker\" data-citation=\"{}\" aria-label=\"View source {}\">[{}]</button>",
                num, num, num
            )
        })
        .into_owned()
}

/// Generate cited feedback using RAG and LLM
///
/// This function:
/// 1. Builds a prompt with performance data and retrieved pedagogy chunks
/// 2. Calls Workers AI Llama 3.3 70B for feedback generation
/// 3. Parses citation markers from the response
/// 4. Returns structured CitedFeedback with HTML, plain text, and citation metadata
#[cfg(feature = "ssr")]
pub async fn generate_cited_feedback(
    env: &Env,
    performance: &Performance,
    dimensions: &PerformanceDimensions,
    retrieved_chunks: &[RetrievalResult],
) -> Result<CitedFeedback, String> {
    // Build the prompt
    let prompt = build_feedback_prompt(performance, dimensions, retrieved_chunks);

    // Call Workers AI
    let ai = env.ai("AI").map_err(|e| format!("Failed to get AI binding: {:?}", e))?;

    let request = serde_json::json!({
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.7
    });

    let result = ai
        .run(LLM_MODEL, request)
        .await
        .map_err(|e| format!("Workers AI LLM call failed: {:?}", e))?;

    // Parse the response
    let response: TextGenerationResponse = serde_json::from_value(result)
        .map_err(|e| format!("Failed to parse LLM response: {:?}", e))?;

    let plain_text = response.response.trim().to_string();

    // Parse citations from response
    let cited_numbers = parse_citations(&plain_text, retrieved_chunks);

    // Build citation metadata for each cited source
    let citations: Vec<Citation> = cited_numbers
        .iter()
        .filter_map(|&num| {
            let idx = (num - 1) as usize;
            retrieved_chunks.get(idx).map(|result| {
                Citation::from_chunk(&result.chunk, num)
            })
        })
        .collect();

    // Convert to HTML with citation buttons
    let html = text_to_html_with_citations(&plain_text);

    Ok(CitedFeedback {
        html,
        plain_text,
        citations,
    })
}

/// Fallback to template-based feedback when RAG is unavailable
pub fn generate_fallback_feedback(
    performance: &Performance,
    dimensions: &PerformanceDimensions,
) -> CitedFeedback {
    let plain_text = generate_template_feedback(performance, dimensions);

    CitedFeedback {
        html: plain_text.clone(),
        plain_text,
        citations: Vec::new(),
    }
}

/// Generate a chat response to a user question using RAG
///
/// This function:
/// 1. Builds a prompt with the question and retrieved pedagogy chunks
/// 2. Calls Workers AI Llama 3.3 70B for answer generation
/// 3. Parses citation markers from the response
/// 4. Returns structured CitedFeedback with answer and citations
#[cfg(feature = "ssr")]
pub async fn generate_chat_response(
    env: &Env,
    question: &str,
    performance: Option<&Performance>,
    retrieved_chunks: &[RetrievalResult],
) -> Result<CitedFeedback, String> {
    // Build the chat prompt
    let prompt = build_chat_prompt(question, performance, retrieved_chunks);

    // Call Workers AI
    let ai = env.ai("AI").map_err(|e| format!("Failed to get AI binding: {:?}", e))?;

    let request = serde_json::json!({
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 512,
        "temperature": 0.7
    });

    let result = ai
        .run(LLM_MODEL, request)
        .await
        .map_err(|e| format!("Workers AI LLM call failed: {:?}", e))?;

    // Parse the response
    let response: TextGenerationResponse = serde_json::from_value(result)
        .map_err(|e| format!("Failed to parse LLM response: {:?}", e))?;

    let plain_text = response.response.trim().to_string();

    // Parse citations from response
    let cited_numbers = parse_citations(&plain_text, retrieved_chunks);

    // Build citation metadata
    let citations: Vec<Citation> = cited_numbers
        .iter()
        .filter_map(|&num| {
            let idx = (num - 1) as usize;
            retrieved_chunks.get(idx).map(|result| {
                Citation::from_chunk(&result.chunk, num)
            })
        })
        .collect();

    // Convert to HTML with citation buttons
    let html = text_to_html_with_citations(&plain_text);

    Ok(CitedFeedback {
        html,
        plain_text,
        citations,
    })
}

/// Build the LLM prompt for chat response generation
fn build_chat_prompt(
    question: &str,
    performance: Option<&Performance>,
    retrieved_chunks: &[RetrievalResult],
) -> String {
    let mut prompt = String::new();

    // System instruction
    prompt.push_str("You are an expert piano teacher answering a student's question.\n\n");

    // Performance context if available
    if let Some(perf) = performance {
        prompt.push_str("## Context\n");
        prompt.push_str(&format!("The student is working on: {} by {}\n\n", perf.piece_title, perf.composer));
    }

    // Question
    prompt.push_str("## Question\n");
    prompt.push_str(question);
    prompt.push_str("\n\n");

    // Reference sources
    if !retrieved_chunks.is_empty() {
        prompt.push_str("## Reference Sources\n");
        for (i, result) in retrieved_chunks.iter().enumerate() {
            prompt.push_str(&format_source_reference(result, i));
            prompt.push_str("\n\n");
        }
    }

    // Instructions
    prompt.push_str("## Instructions\n");
    prompt.push_str("Provide a helpful, concise answer to the question.\n");
    prompt.push_str("Include inline citations using [1], [2], etc. when referencing advice from the sources.\n");
    prompt.push_str("Be specific and actionable. Write 1-2 paragraphs maximum.");

    prompt
}

/// Template-based feedback (original implementation)
fn generate_template_feedback(
    performance: &Performance,
    dimensions: &PerformanceDimensions,
) -> String {
    let mut scores: Vec<(&str, f64)> = vec![
        ("timing precision", dimensions.timing),
        (
            "articulation control",
            (dimensions.articulation_length + dimensions.articulation_touch) / 2.0,
        ),
        (
            "pedaling technique",
            (dimensions.pedal_amount + dimensions.pedal_clarity) / 2.0,
        ),
        (
            "tonal variety",
            (dimensions.timbre_variety + dimensions.timbre_depth) / 2.0,
        ),
        ("dynamic expression", dimensions.dynamics_range),
        ("dramatic intensity", dimensions.drama),
        ("musical imagination", dimensions.mood_imagination),
        ("interpretive depth", dimensions.interpretation_overall),
    ];
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let top_strength = scores[0].0;
    let second_strength = scores[1].0;
    let growth_area = scores.last().unwrap().0;

    let performer_style = match performance.performer.as_str() {
        "Vladimir Horowitz" => "legendary virtuosity and dramatic flair",
        "Martha Argerich" => "fiery temperament and electrifying energy",
        "Glenn Gould" => "intellectual clarity and unique artistic vision",
        "Krystian Zimerman" => "meticulous attention to detail and tonal refinement",
        "Evgeny Kissin" => "passionate intensity and technical brilliance",
        "Maurizio Pollini" => "structural clarity and controlled power",
        _ => "distinctive artistic voice",
    };

    let overall_score = dimensions.interpretation_overall;
    let quality_descriptor = if overall_score >= 0.90 {
        "exceptional"
    } else if overall_score >= 0.80 {
        "impressive"
    } else if overall_score >= 0.70 {
        "solid"
    } else {
        "developing"
    };

    format!(
        "This {} interpretation of {} by {} demonstrates {}. \
        The performance shows particular strength in {}, which brings out the \
        emotional depth of the piece beautifully. The {} also contributes significantly \
        to the overall musical narrative.\n\n\
        The recording captures {}'s {}, particularly evident in the way phrases \
        are shaped and the natural ebb and flow of the musical line. \
        For continued growth, focusing on {} could add even more nuance \
        to this already compelling interpretation.\n\n\
        Overall, this is a performance that rewards careful listening and \
        demonstrates a deep understanding of {}'s musical language.",
        quality_descriptor,
        performance.piece_title,
        performance.performer,
        performer_style,
        top_strength,
        second_strength,
        performance.performer,
        performer_style,
        growth_area,
        performance.composer
    )
}
