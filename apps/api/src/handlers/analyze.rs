use crate::models::{AnalysisResult, Performance};
use crate::services::{generate_teacher_feedback, get_performance_dimensions, get_practice_tips};
use crate::utils::{error_response, json_response};
use worker::{Request, Response, Result, RouteContext};

/// POST /api/analyze/:id
/// Runs the analysis pipeline for a performance.
///
/// Pipeline steps:
/// 1. Validate performance ID exists
/// 2. Check KV cache for existing result (TODO: when KV is configured)
/// 3. Call RunPod for 19-dimension inference (currently mocked)
/// 4. Query Vectorize for RAG context (currently mocked)
/// 5. Call Workers AI for feedback generation (currently mocked)
/// 6. Cache result in KV (TODO: when KV is configured)
/// 7. Return AnalysisResult
pub async fn handle_analyze<D>(
    _req: Request,
    ctx: RouteContext<D>,
) -> Result<Response> {
    // Get performance ID from URL
    let id = match ctx.param("id") {
        Some(id) => id,
        None => return error_response("Missing performance ID", 400),
    };

    // Validate that the performance exists
    let performance = match Performance::find_by_id(id) {
        Some(p) => p,
        None => return error_response("Performance not found", 404),
    };

    // TODO: Check KV cache for existing result
    // let kv = ctx.kv("ANALYSIS_CACHE")?;
    // if let Some(cached) = kv.get(&format!("analysis:{}", id)).json::<AnalysisResult>().await? {
    //     return json_response(&cached);
    // }

    // Step 1: Get performance dimensions from RunPod (currently mocked)
    let dimensions = get_performance_dimensions(id).await;

    // Step 2: Get practice tips from Vectorize RAG (currently mocked)
    let practice_tips = get_practice_tips(&performance, &dimensions).await;

    // Step 3: Generate teacher feedback from Workers AI (currently mocked)
    let teacher_feedback = generate_teacher_feedback(&performance, &dimensions).await;

    // Build the analysis result
    let result = AnalysisResult {
        performance_id: id.to_string(),
        dimensions,
        teacher_feedback,
        practice_tips,
    };

    // TODO: Cache result in KV
    // kv.put(&format!("analysis:{}", id), &result)?
    //     .expiration_ttl(86400) // 24 hours
    //     .execute()
    //     .await?;

    json_response(&result)
}
