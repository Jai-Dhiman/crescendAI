use worker::*;
use serde::{Deserialize, Serialize};

mod handlers;
mod security;
mod storage;
mod processing;
mod modal_client;
mod utils;

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct JobStatus {
    pub job_id: String,
    pub status: String,
    pub progress: f32,
    pub error: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct AnalysisResult {
    pub id: String,
    pub status: String,
    pub file_id: String,
    pub analysis: AnalysisData,
    pub insights: Vec<String>,
    pub created_at: String,
    pub processing_time: Option<f32>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct ComparisonResult {
    pub id: String,
    pub status: String,
    pub file_id: String,
    pub model_a: ModelResult,
    pub model_b: ModelResult,
    pub created_at: String,
    pub total_processing_time: Option<f32>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct ModelResult {
    pub model_name: String,
    pub model_type: String,
    pub analysis: AnalysisData,
    pub insights: Vec<String>,
    pub processing_time: Option<f32>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct UserPreference {
    pub comparison_id: String,
    pub preferred_model: String, // "model_a" or "model_b"
    pub feedback: Option<String>,
    pub created_at: String,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct AnalysisData {
    pub rhythm: f32,
    pub pitch: f32,
    pub dynamics: f32,
    pub tempo: f32,
    pub articulation: f32,
    pub expression: f32,
    pub technique: f32,
    pub timing: f32,
    pub phrasing: f32,
    pub voicing: f32,
    pub pedaling: f32,
    pub hand_coordination: f32,
    pub musical_understanding: f32,
    pub stylistic_accuracy: f32,
    pub creativity: f32,
    pub listening: f32,
    pub overall_performance: f32,
    pub stage_presence: f32,
    pub repertoire_difficulty: f32,
}

#[event(fetch)]
pub async fn main(req: Request, env: Env, _ctx: worker::Context) -> Result<Response> {
    console_error_panic_hook::set_once();
    
    let router = Router::new();
    
    router
        // CORS preflight handler for all routes
        .options("/api/v1/upload", handle_options)
        .options("/api/v1/analyze", handle_options)
        .options("/api/v1/compare", handle_options)
        .options("/api/v1/job/:id", handle_options)
        .options("/api/v1/result/:id", handle_options)
        .options("/api/v1/comparison/:id", handle_options)
        .options("/api/v1/preference", handle_options)
        .options("/api/v1/health", handle_options)
        // Main API routes with CORS
        .post_async("/api/v1/upload", upload_with_cors)
        .post_async("/api/v1/analyze", analyze_with_cors)
        .post_async("/api/v1/compare", compare_with_cors)
        .get_async("/api/v1/job/:id", job_status_with_cors)
        .get_async("/api/v1/result/:id", result_with_cors)
        .get_async("/api/v1/comparison/:id", comparison_result_with_cors)
        .post_async("/api/v1/preference", preference_with_cors)
        .get("/api/v1/health", |_req, _ctx| {
            let mut response = Response::from_json(&serde_json::json!({
                "status": "healthy",
                "message": "CrescendAI API is running"
            }))?;
            add_cors_headers(&mut response);
            Ok(response)
        })
        .run(req, env)
        .await
}

/// Add CORS headers to a response
fn add_cors_headers(response: &mut Response) {
    let headers = response.headers_mut();
    
    // Allow all origins in development (restrict in production)
    headers.set("Access-Control-Allow-Origin", "*").unwrap();
    headers.set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS").unwrap();
    headers.set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With").unwrap();
    headers.set("Access-Control-Max-Age", "86400").unwrap(); // 24 hours
}

/// Handle OPTIONS preflight requests
fn handle_options(_req: Request, _ctx: RouteContext<()>) -> Result<Response> {
    let mut response = Response::empty()?;
    add_cors_headers(&mut response);
    Ok(response)
}

/// CORS wrapper for upload handler
async fn upload_with_cors(req: Request, ctx: RouteContext<()>) -> Result<Response> {
    match handlers::upload_audio(req, ctx).await {
        Ok(mut response) => {
            add_cors_headers(&mut response);
            Ok(response)
        }
        Err(e) => {
            let mut error_response = Response::error("Upload failed", 500)?;
            add_cors_headers(&mut error_response);
            Err(e)
        }
    }
}

/// CORS wrapper for analyze handler
async fn analyze_with_cors(req: Request, ctx: RouteContext<()>) -> Result<Response> {
    match handlers::analyze_audio(req, ctx).await {
        Ok(mut response) => {
            add_cors_headers(&mut response);
            Ok(response)
        }
        Err(e) => {
            let mut error_response = Response::error("Analysis failed", 500)?;
            add_cors_headers(&mut error_response);
            Err(e)
        }
    }
}

/// CORS wrapper for job status handler
async fn job_status_with_cors(req: Request, ctx: RouteContext<()>) -> Result<Response> {
    match handlers::get_job_status(req, ctx).await {
        Ok(mut response) => {
            add_cors_headers(&mut response);
            Ok(response)
        }
        Err(e) => {
            let mut error_response = Response::error("Job status fetch failed", 404)?;
            add_cors_headers(&mut error_response);
            Err(e)
        }
    }
}

/// CORS wrapper for result handler
async fn result_with_cors(req: Request, ctx: RouteContext<()>) -> Result<Response> {
    match handlers::get_analysis_result(req, ctx).await {
        Ok(mut response) => {
            add_cors_headers(&mut response);
            Ok(response)
        }
        Err(e) => {
            let mut error_response = Response::error("Result fetch failed", 404)?;
            add_cors_headers(&mut error_response);
            Err(e)
        }
    }
}

/// CORS wrapper for compare handler
async fn compare_with_cors(req: Request, ctx: RouteContext<()>) -> Result<Response> {
    match handlers::compare_models(req, ctx).await {
        Ok(mut response) => {
            add_cors_headers(&mut response);
            Ok(response)
        }
        Err(e) => {
            let mut error_response = Response::error("Comparison failed", 500)?;
            add_cors_headers(&mut error_response);
            Err(e)
        }
    }
}

/// CORS wrapper for comparison result handler
async fn comparison_result_with_cors(req: Request, ctx: RouteContext<()>) -> Result<Response> {
    match handlers::get_comparison_result(req, ctx).await {
        Ok(mut response) => {
            add_cors_headers(&mut response);
            Ok(response)
        }
        Err(e) => {
            let mut error_response = Response::error("Comparison result fetch failed", 404)?;
            add_cors_headers(&mut error_response);
            Err(e)
        }
    }
}

/// CORS wrapper for preference handler
async fn preference_with_cors(req: Request, ctx: RouteContext<()>) -> Result<Response> {
    match handlers::save_user_preference(req, ctx).await {
        Ok(mut response) => {
            add_cors_headers(&mut response);
            Ok(response)
        }
        Err(e) => {
            let mut error_response = Response::error("Preference save failed", 500)?;
            add_cors_headers(&mut error_response);
            Err(e)
        }
    }
}
