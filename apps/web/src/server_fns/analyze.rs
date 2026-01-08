use leptos::prelude::*;

use crate::models::{AnalysisResult, Performance};
use crate::services::{generate_teacher_feedback, get_performance_dimensions, get_practice_tips};

#[server(AnalyzePerformance, "/api")]
pub async fn analyze_performance(id: String) -> Result<AnalysisResult, ServerFnError> {
    let performance = Performance::find_by_id(&id)
        .ok_or_else(|| ServerFnError::new("Performance not found"))?;

    let dimensions = get_performance_dimensions(&id).await;
    let practice_tips = get_practice_tips(&performance, &dimensions).await;
    let teacher_feedback = generate_teacher_feedback(&performance, &dimensions).await;

    Ok(AnalysisResult {
        performance_id: id,
        dimensions,
        teacher_feedback,
        practice_tips,
    })
}
