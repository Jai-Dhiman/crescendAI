use worker::*;
use serde::{Deserialize, Serialize};
use crate::ast_mock::{MertAnalysisResult, mock_mert_analysis};
use crate::db::{get_recording, get_analysis_result, insert_analysis_result};

#[derive(Debug, Serialize, Deserialize)]
pub struct FeedbackRequest {
    pub user_id: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FeedbackResponse {
    pub recording_id: String,
    pub overall_assessment: OverallAssessment,
    pub temporal_feedback: Vec<TemporalFeedbackItem>,
    pub practice_recommendations: PracticeRecommendations,
    pub metadata: FeedbackMetadata,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OverallAssessment {
    pub strengths: Vec<String>,
    pub areas_for_improvement: Vec<String>,
    pub overall_score: f64,
    pub skill_level_estimate: String,
    pub dimension_scores: DimensionScoreSummary,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DimensionScoreSummary {
    pub technical_average: f64,
    pub interpretive_average: f64,
    pub note_accuracy: f64,
    pub rhythmic_precision: f64,
    pub dynamics_control: f64,
    pub articulation_quality: f64,
    pub pedaling_technique: f64,
    pub tone_quality: f64,
    pub phrasing: f64,
    pub expressiveness: f64,
    pub musical_interpretation: f64,
    pub stylistic_appropriateness: f64,
    pub overall_musicality: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TemporalFeedbackItem {
    pub start_time: f64,
    pub end_time: f64,
    pub bar_range: Option<String>,
    pub observations: Vec<String>,
    pub specific_suggestions: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PracticeRecommendations {
    pub immediate_focus: Vec<String>,
    pub long_term_goals: Vec<String>,
    pub exercises: Vec<String>,
    pub estimated_improvement_timeline: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FeedbackMetadata {
    pub model_version: String,
    pub analysis_timestamp: i64,
    pub confidence: f64,
}

pub async fn generate_feedback_handler(
    mut req: Request,
    ctx: RouteContext<()>,
) -> Result<Response> {
    // Get recording ID from URL params
    let recording_id = ctx.param("id")
        .ok_or_else(|| Error::RustError("Missing recording_id parameter".to_string()))?;

    // Parse request body
    let body: FeedbackRequest = req.json().await
        .map_err(|e| Error::RustError(format!("Invalid request body: {}", e)))?;

    let env = &ctx.env;

    console_log!("Generating feedback for recording: {}", recording_id);

    // 1. Retrieve recording metadata
    let recording = get_recording(env, recording_id).await
        .map_err(|e| Error::RustError(format!("Failed to get recording: {}", e)))?;

    // 2. Get or generate analysis results
    let db = env.d1("DB")?;
    let analysis = match get_analysis_result(&db, recording_id).await? {
        Some(existing_analysis) => {
            console_log!("Using existing analysis for recording {}", recording_id);
            existing_analysis
        }
        None => {
            console_log!("Generating mock analysis for recording {}", recording_id);
            let duration = recording.duration.unwrap_or(120.0);
            let mock_analysis = mock_mert_analysis(recording_id, duration)?;

            // Store the analysis for future use
            insert_analysis_result(&db, recording_id, &mock_analysis).await?;

            mock_analysis
        }
    };

    // 3. Generate structured feedback from analysis
    let feedback = generate_structured_feedback(&analysis, &body.user_id);

    // 4. Store feedback in database (feedback_history table)
    let feedback_id = format!("feedback-{}", uuid::Uuid::new_v4());
    let created_at = Date::now().as_millis() as i64;

    let feedback_json = serde_json::to_string(&feedback)
        .map_err(|e| Error::RustError(format!("Failed to serialize feedback: {}", e)))?;

    db.prepare("INSERT INTO feedback_history (id, recording_id, session_id, feedback_text, citations, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6)")
        .bind(&[
            wasm_bindgen::JsValue::from_str(&feedback_id),
            wasm_bindgen::JsValue::from_str(recording_id),
            wasm_bindgen::JsValue::NULL,
            wasm_bindgen::JsValue::from_str(&feedback_json),
            wasm_bindgen::JsValue::from_str("[]"), // Empty citations for now
            wasm_bindgen::JsValue::from_f64(created_at as f64),
        ])?
        .run()
        .await?;

    console_log!("Feedback generated and stored: {}", feedback_id);

    // Return feedback as JSON
    Response::from_json(&feedback)
}

fn generate_structured_feedback(analysis: &MertAnalysisResult, _user_id: &str) -> FeedbackResponse {
    let scores = &analysis.overall_scores;

    // Calculate averages
    let technical_average = (scores.note_accuracy + scores.rhythmic_precision +
                            scores.dynamics_control + scores.articulation_quality +
                            scores.pedaling_technique + scores.tone_quality) / 6.0;

    let interpretive_average = (scores.phrasing + scores.expressiveness +
                               scores.musical_interpretation + scores.stylistic_appropriateness +
                               scores.overall_musicality) / 5.0;

    // Identify strengths (scores > 75)
    let mut strengths = Vec::new();
    if scores.note_accuracy > 75.0 {
        strengths.push(format!("Excellent note accuracy ({:.0}/100)", scores.note_accuracy));
    }
    if scores.rhythmic_precision > 75.0 {
        strengths.push(format!("Strong rhythmic precision ({:.0}/100)", scores.rhythmic_precision));
    }
    if scores.dynamics_control > 75.0 {
        strengths.push(format!("Good control of dynamics ({:.0}/100)", scores.dynamics_control));
    }
    if scores.tone_quality > 75.0 {
        strengths.push(format!("Beautiful tone quality ({:.0}/100)", scores.tone_quality));
    }
    if scores.expressiveness > 75.0 {
        strengths.push(format!("Expressive playing ({:.0}/100)", scores.expressiveness));
    }
    if scores.phrasing > 75.0 {
        strengths.push(format!("Excellent phrasing ({:.0}/100)", scores.phrasing));
    }

    if strengths.is_empty() {
        strengths.push("Solid overall foundation in piano technique".to_string());
    }

    // Identify areas for improvement (scores < 60)
    let mut areas_for_improvement = Vec::new();
    if scores.note_accuracy < 60.0 {
        areas_for_improvement.push(format!("Note accuracy needs work ({:.0}/100)", scores.note_accuracy));
    }
    if scores.rhythmic_precision < 60.0 {
        areas_for_improvement.push(format!("Rhythmic precision could be improved ({:.0}/100)", scores.rhythmic_precision));
    }
    if scores.dynamics_control < 60.0 {
        areas_for_improvement.push(format!("Focus on dynamic control ({:.0}/100)", scores.dynamics_control));
    }
    if scores.articulation_quality < 60.0 {
        areas_for_improvement.push(format!("Articulation needs attention ({:.0}/100)", scores.articulation_quality));
    }
    if scores.pedaling_technique < 60.0 {
        areas_for_improvement.push(format!("Pedaling technique requires practice ({:.0}/100)", scores.pedaling_technique));
    }
    if scores.tone_quality < 60.0 {
        areas_for_improvement.push(format!("Work on tone production ({:.0}/100)", scores.tone_quality));
    }
    if scores.phrasing < 60.0 {
        areas_for_improvement.push(format!("Develop phrasing skills ({:.0}/100)", scores.phrasing));
    }
    if scores.expressiveness < 60.0 {
        areas_for_improvement.push(format!("Enhance musical expressiveness ({:.0}/100)", scores.expressiveness));
    }

    if areas_for_improvement.is_empty() {
        areas_for_improvement.push("Continue refining interpretive nuances".to_string());
    }

    // Generate practice recommendations based on weakest areas
    let mut immediate_focus = Vec::new();
    let mut exercises = Vec::new();

    if scores.note_accuracy < 70.0 {
        immediate_focus.push("Practice sections slowly with metronome to improve accuracy".to_string());
        exercises.push("Hanon exercises for finger independence".to_string());
    }
    if scores.rhythmic_precision < 70.0 {
        immediate_focus.push("Work on rhythm with subdivisions and metronome".to_string());
        exercises.push("Clapping and counting exercises for complex rhythms".to_string());
    }
    if scores.dynamics_control < 70.0 {
        immediate_focus.push("Practice dynamic gradations (crescendo/diminuendo)".to_string());
        exercises.push("Graduated dynamics exercises across full keyboard range".to_string());
    }
    if scores.pedaling_technique < 70.0 {
        immediate_focus.push("Focus on pedaling technique and timing".to_string());
        exercises.push("Pedaling exercises with sustained chords".to_string());
    }

    if immediate_focus.is_empty() {
        immediate_focus.push("Continue developing musical interpretation".to_string());
        exercises.push("Listen to recordings by master pianists".to_string());
    }

    // Long-term goals
    let long_term_goals = vec![
        "Develop consistent technical accuracy across all tempos".to_string(),
        "Build interpretive depth and musical understanding".to_string(),
        "Master pedaling for various musical styles".to_string(),
    ];

    // Build temporal feedback
    let temporal_feedback: Vec<TemporalFeedbackItem> = analysis
        .temporal_segments
        .iter()
        .map(|seg| {
            let mut observations = seg.issues.clone();
            if observations.is_empty() {
                observations.push("Generally well-executed".to_string());
            }

            let suggestions = generate_suggestions_for_segment(seg);

            TemporalFeedbackItem {
                start_time: seg.start_time,
                end_time: seg.end_time,
                bar_range: seg.bar_range.clone(),
                observations,
                specific_suggestions: suggestions,
            }
        })
        .collect();

    FeedbackResponse {
        recording_id: analysis.recording_id.clone(),
        overall_assessment: OverallAssessment {
            strengths,
            areas_for_improvement,
            overall_score: scores.overall_quality,
            skill_level_estimate: estimate_skill_level(scores.overall_quality),
            dimension_scores: DimensionScoreSummary {
                technical_average,
                interpretive_average,
                note_accuracy: scores.note_accuracy,
                rhythmic_precision: scores.rhythmic_precision,
                dynamics_control: scores.dynamics_control,
                articulation_quality: scores.articulation_quality,
                pedaling_technique: scores.pedaling_technique,
                tone_quality: scores.tone_quality,
                phrasing: scores.phrasing,
                expressiveness: scores.expressiveness,
                musical_interpretation: scores.musical_interpretation,
                stylistic_appropriateness: scores.stylistic_appropriateness,
                overall_musicality: scores.overall_musicality,
            },
        },
        temporal_feedback,
        practice_recommendations: PracticeRecommendations {
            immediate_focus,
            long_term_goals,
            exercises,
            estimated_improvement_timeline: estimate_timeline(scores.overall_quality),
        },
        metadata: FeedbackMetadata {
            model_version: analysis.metadata.model_version.clone(),
            analysis_timestamp: Date::now().as_millis() as i64,
            confidence: analysis.metadata.confidence_score,
        },
    }
}

fn generate_suggestions_for_segment(segment: &crate::ast_mock::TemporalSegment) -> Vec<String> {
    let mut suggestions = Vec::new();

    if segment.issues.iter().any(|i| i.contains("Note accuracy")) {
        suggestions.push("Practice this passage hands separately at half tempo".to_string());
    }
    if segment.issues.iter().any(|i| i.contains("Rhythmic")) {
        suggestions.push("Count aloud while playing to solidify rhythm".to_string());
    }
    if segment.issues.iter().any(|i| i.contains("Dynamics")) {
        suggestions.push("Mark dynamic intentions in score and exaggerate initially".to_string());
    }
    if segment.issues.iter().any(|i| i.contains("Pedaling")) {
        suggestions.push("Practice pedal changes without hands first".to_string());
    }

    if suggestions.is_empty() {
        suggestions.push("Continue practicing for consistency".to_string());
    }

    suggestions
}

fn estimate_skill_level(overall_score: f64) -> String {
    match overall_score {
        s if s >= 85.0 => "Advanced".to_string(),
        s if s >= 70.0 => "Intermediate-Advanced".to_string(),
        s if s >= 55.0 => "Intermediate".to_string(),
        s if s >= 40.0 => "Early Intermediate".to_string(),
        _ => "Beginner".to_string(),
    }
}

fn estimate_timeline(overall_score: f64) -> String {
    match overall_score {
        s if s >= 75.0 => "1-2 weeks of focused practice for refinement".to_string(),
        s if s >= 60.0 => "2-4 weeks with consistent daily practice".to_string(),
        s if s >= 45.0 => "4-8 weeks of dedicated practice".to_string(),
        _ => "2-3 months of foundational work recommended".to_string(),
    }
}
