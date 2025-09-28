use worker::*;
use serde::{Deserialize, Serialize};

mod handlers;
mod security;
mod storage;
mod processing;
mod utils;
mod audio_dsp;
mod monitoring;

use security::{validate_api_key, get_client_ip, RateLimiter, secure_error_response};
use monitoring::{RequestLogger, HealthChecker, SystemInfo};

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
    pub processing_time: f64,
    pub dimensions: Option<Vec<f32>>,
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
        // Main API routes with authentication and CORS
        .post_async("/api/v1/upload", secure_upload_handler)
        .post_async("/api/v1/analyze", secure_analyze_handler)
        .post_async("/api/v1/compare", secure_compare_handler)
        .get_async("/api/v1/job/:id", secure_job_status_handler)
        .get_async("/api/v1/result/:id", secure_result_handler)
        .get_async("/api/v1/comparison/:id", secure_comparison_result_handler)
        .post_async("/api/v1/preference", secure_preference_handler)
        // Health endpoint without authentication (for monitoring)
        .get_async("/api/v1/health", basic_health_handler)
        // Detailed health endpoint (requires authentication)
        .get_async("/api/v1/health/detailed", secure_detailed_health_handler)
        // System info endpoint (requires authentication)
        .get_async("/api/v1/info", secure_system_info_handler)
        .run(req, env)
        .await
}

/// Get allowed origins based on environment variables
fn get_allowed_origins_from_env(env: Option<&Env>) -> Vec<String> {
    if let Some(env) = env {
        if let Ok(origins_string) = env.var("ALLOWED_ORIGINS") {
            return origins_string
                .to_string()
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
        }
    }
    
    // Safe defaults if environment variable is not set
    vec![
        "https://crescendai.com".to_string(),
        "https://app.crescendai.com".to_string(),
        "https://www.crescendai.com".to_string(),
        "http://localhost:3000".to_string(),  // Development
        "http://localhost:5173".to_string(),  // Vite dev server
    ]
}

/// Get allowed origins (legacy function for compatibility)
fn get_allowed_origins() -> Vec<String> {
    get_allowed_origins_from_env(None)
}

/// Add CORS headers to a response with proper origin validation
fn add_cors_headers(response: &mut Response, allowed_origins: &[String]) -> Result<()> {
    let headers = response.headers_mut();
    
    // Set a safe default origin (first in the list)
    let origin = allowed_origins.first()
        .map(|s| s.as_str())
        .unwrap_or("https://crescendai.com");
    
    headers.set("Access-Control-Allow-Origin", origin)
        .map_err(|_| worker::Error::RustError("Failed to set CORS origin".to_string()))?;
    headers.set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        .map_err(|_| worker::Error::RustError("Failed to set CORS methods".to_string()))?;
    headers.set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-API-Key, X-Requested-With")
        .map_err(|_| worker::Error::RustError("Failed to set CORS headers".to_string()))?;
    headers.set("Access-Control-Max-Age", "86400")
        .map_err(|_| worker::Error::RustError("Failed to set CORS max age".to_string()))?; // 24 hours
    headers.set("Access-Control-Allow-Credentials", "true")
        .map_err(|_| worker::Error::RustError("Failed to set CORS credentials".to_string()))?;
    
    Ok(())
}

/// Authentication and security validation
async fn validate_request_security(req: &Request, env: &Env) -> Result<()> {
    // Validate API key first
    validate_api_key(req, env)?;
    
    // Check rate limiting
    if let Ok(kv) = env.kv("METADATA") {
        let rate_limiter = RateLimiter::new(kv, 60, 100); // 100 requests per minute
        let client_ip = get_client_ip(req);
        rate_limiter.check_rate_limit(&client_ip).await?;
    }
    
    Ok(())
}

/// Handle OPTIONS preflight requests
fn handle_options(req: Request, _ctx: RouteContext<()>) -> Result<Response> {
    let mut response = Response::empty()?;
    let allowed_origins = get_allowed_origins();
    
    // Get the origin from request and validate it
    if let Ok(Some(origin)) = req.headers().get("Origin") {
        if allowed_origins.contains(&origin) {
            let headers = response.headers_mut();
            headers.set("Access-Control-Allow-Origin", &origin)
                .map_err(|_| worker::Error::RustError("Failed to set CORS origin".to_string()))?;
        }
    }
    
    add_cors_headers(&mut response, &allowed_origins)?;
    Ok(response)
}

/// Secure upload handler with authentication and CORS
async fn secure_upload_handler(req: Request, ctx: RouteContext<()>) -> Result<Response> {
    // Get allowed origins for CORS and check development mode
    let allowed_origins = get_allowed_origins_from_env(Some(&ctx.env));
    let is_development = ctx.env.var("ENVIRONMENT")
        .map(|env| env.to_string() == "development")
        .unwrap_or(false);
    
    // Validate security first
    if let Err(security_error) = validate_request_security(&req, &ctx.env).await {
        let mut error_response = secure_error_response(&security_error, is_development);
        add_cors_headers(&mut error_response, &allowed_origins).ok();
        return Ok(error_response);
    }
    
    match handlers::upload_audio(req, ctx).await {
        Ok(mut response) => {
            add_cors_headers(&mut response, &allowed_origins).ok();
            Ok(response)
        }
        Err(e) => {
            let mut error_response = secure_error_response(&e, is_development);
            add_cors_headers(&mut error_response, &allowed_origins).ok();
            Ok(error_response)
        }
    }
}

/// Secure analyze handler with authentication and CORS
async fn secure_analyze_handler(req: Request, ctx: RouteContext<()>) -> Result<Response> {
    // Get allowed origins for CORS and check development mode
    let allowed_origins = get_allowed_origins_from_env(Some(&ctx.env));
    let is_development = ctx.env.var("ENVIRONMENT")
        .map(|env| env.to_string() == "development")
        .unwrap_or(false);
    
    // Validate security first
    if let Err(security_error) = validate_request_security(&req, &ctx.env).await {
        let mut error_response = secure_error_response(&security_error, is_development);
        add_cors_headers(&mut error_response, &allowed_origins).ok();
        return Ok(error_response);
    }
    
    match handlers::analyze_audio(req, ctx).await {
        Ok(mut response) => {
            add_cors_headers(&mut response, &allowed_origins).ok();
            Ok(response)
        }
        Err(e) => {
            let mut error_response = secure_error_response(&e, is_development);
            add_cors_headers(&mut error_response, &allowed_origins).ok();
            Ok(error_response)
        }
    }
}

/// Secure job status handler with authentication and CORS
async fn secure_job_status_handler(req: Request, ctx: RouteContext<()>) -> Result<Response> {
    // Get allowed origins for CORS and check development mode
    let allowed_origins = get_allowed_origins_from_env(Some(&ctx.env));
    let is_development = ctx.env.var("ENVIRONMENT")
        .map(|env| env.to_string() == "development")
        .unwrap_or(false);
    
    // Validate security first
    if let Err(security_error) = validate_request_security(&req, &ctx.env).await {
        let mut error_response = secure_error_response(&security_error, is_development);
        add_cors_headers(&mut error_response, &allowed_origins).ok();
        return Ok(error_response);
    }
    
    match handlers::get_job_status(req, ctx).await {
        Ok(mut response) => {
            add_cors_headers(&mut response, &allowed_origins).ok();
            Ok(response)
        }
        Err(e) => {
            let mut error_response = secure_error_response(&e, is_development);
            add_cors_headers(&mut error_response, &allowed_origins).ok();
            Ok(error_response)
        }
    }
}

/// Secure result handler with authentication and CORS
async fn secure_result_handler(req: Request, ctx: RouteContext<()>) -> Result<Response> {
    // Get allowed origins for CORS and check development mode
    let allowed_origins = get_allowed_origins_from_env(Some(&ctx.env));
    let is_development = ctx.env.var("ENVIRONMENT")
        .map(|env| env.to_string() == "development")
        .unwrap_or(false);
    
    // Validate security first
    if let Err(security_error) = validate_request_security(&req, &ctx.env).await {
        let mut error_response = secure_error_response(&security_error, is_development);
        add_cors_headers(&mut error_response, &allowed_origins).ok();
        return Ok(error_response);
    }
    
    match handlers::get_analysis_result(req, ctx).await {
        Ok(mut response) => {
            add_cors_headers(&mut response, &allowed_origins).ok();
            Ok(response)
        }
        Err(e) => {
            let mut error_response = secure_error_response(&e, is_development);
            add_cors_headers(&mut error_response, &allowed_origins).ok();
            Ok(error_response)
        }
    }
}

/// Secure compare handler with authentication and CORS
async fn secure_compare_handler(req: Request, ctx: RouteContext<()>) -> Result<Response> {
    // Get allowed origins for CORS and check development mode
    let allowed_origins = get_allowed_origins_from_env(Some(&ctx.env));
    let is_development = ctx.env.var("ENVIRONMENT")
        .map(|env| env.to_string() == "development")
        .unwrap_or(false);
    
    // Validate security first
    if let Err(security_error) = validate_request_security(&req, &ctx.env).await {
        let mut error_response = secure_error_response(&security_error, is_development);
        add_cors_headers(&mut error_response, &allowed_origins).ok();
        return Ok(error_response);
    }
    
    match handlers::compare_models(req, ctx).await {
        Ok(mut response) => {
            add_cors_headers(&mut response, &allowed_origins).ok();
            Ok(response)
        }
        Err(e) => {
            let mut error_response = secure_error_response(&e, is_development);
            add_cors_headers(&mut error_response, &allowed_origins).ok();
            Ok(error_response)
        }
    }
}

/// Secure comparison result handler with authentication and CORS
async fn secure_comparison_result_handler(req: Request, ctx: RouteContext<()>) -> Result<Response> {
    // Get allowed origins for CORS and check development mode
    let allowed_origins = get_allowed_origins_from_env(Some(&ctx.env));
    let is_development = ctx.env.var("ENVIRONMENT")
        .map(|env| env.to_string() == "development")
        .unwrap_or(false);
    
    // Validate security first
    if let Err(security_error) = validate_request_security(&req, &ctx.env).await {
        let mut error_response = secure_error_response(&security_error, is_development);
        add_cors_headers(&mut error_response, &allowed_origins).ok();
        return Ok(error_response);
    }
    
    match handlers::get_comparison_result(req, ctx).await {
        Ok(mut response) => {
            add_cors_headers(&mut response, &allowed_origins).ok();
            Ok(response)
        }
        Err(e) => {
            let mut error_response = secure_error_response(&e, is_development);
            add_cors_headers(&mut error_response, &allowed_origins).ok();
            Ok(error_response)
        }
    }
}

/// Secure preference handler with authentication and CORS
async fn secure_preference_handler(req: Request, ctx: RouteContext<()>) -> Result<Response> {
    // Get allowed origins for CORS and check development mode
    let allowed_origins = get_allowed_origins_from_env(Some(&ctx.env));
    let is_development = ctx.env.var("ENVIRONMENT")
        .map(|env| env.to_string() == "development")
        .unwrap_or(false);
    
    // Validate security first
    if let Err(security_error) = validate_request_security(&req, &ctx.env).await {
        let mut error_response = secure_error_response(&security_error, is_development);
        add_cors_headers(&mut error_response, &allowed_origins).ok();
        return Ok(error_response);
    }
    
    match handlers::save_user_preference(req, ctx).await {
        Ok(mut response) => {
            add_cors_headers(&mut response, &allowed_origins).ok();
            Ok(response)
        }
        Err(e) => {
            let mut error_response = secure_error_response(&e, is_development);
            add_cors_headers(&mut error_response, &allowed_origins).ok();
            Ok(error_response)
        }
    }
}

/// Secure webhook handler for Modal callbacks with signature validation
async fn secure_modal_webhook_handler(req: Request, ctx: RouteContext<()>) -> Result<Response> {
    // Get allowed origins for CORS and check development mode
    let allowed_origins = get_allowed_origins_from_env(Some(&ctx.env));
    let is_development = ctx.env.var("ENVIRONMENT")
        .map(|env| env.to_string() == "development")
        .unwrap_or(false);
    
    // Webhooks don't need API key authentication, but DO need signature validation
    // The signature validation is handled inside the modal_client::handle_modal_webhook function
    
    match modal_client::handle_modal_webhook(req, ctx).await {
        Ok(mut response) => {
            add_cors_headers(&mut response, &allowed_origins).ok();
            Ok(response)
        }
        Err(e) => {
            let mut error_response = secure_error_response(&e, is_development);
            add_cors_headers(&mut error_response, &allowed_origins).ok();
            Ok(error_response)
        }
    }
}

/// Basic health check handler (no authentication required)
async fn basic_health_handler(req: Request, ctx: RouteContext<()>) -> Result<Response> {
    let mut logger = RequestLogger::new(&req);
    logger.log_request_start();
    
    let allowed_origins = get_allowed_origins_from_env(Some(&ctx.env));
    
    // Get KV store for health checker
    let kv_store = ctx.env.kv("METADATA").ok();
    let health_checker = HealthChecker::new(kv_store);
    
    // Perform basic health check
    let health_result = health_checker.basic_health_check().await;
    
    let status_code = if health_result.status == "healthy" { 200 } else { 503 };
    
    let mut response = Response::from_json(&serde_json::json!({
        "status": health_result.status,
        "message": "CrescendAI API health check",
        "timestamp": health_result.timestamp,
        "checks": health_result.checks,
        "version": env!("CARGO_PKG_VERSION")
    }))?;
    
    response = response.with_status(status_code);
    add_cors_headers(&mut response, &allowed_origins)?;
    
    logger.log_request_complete(status_code, None);
    Ok(response)
}

/// Detailed health check handler (requires authentication)
async fn secure_detailed_health_handler(req: Request, ctx: RouteContext<()>) -> Result<Response> {
    let mut logger = RequestLogger::new(&req);
    logger.log_request_start();
    
    let allowed_origins = get_allowed_origins_from_env(Some(&ctx.env));
    let is_development = ctx.env.var("ENVIRONMENT")
        .map(|env| env.to_string() == "development")
        .unwrap_or(false);
    
    // Validate security first
    if let Err(security_error) = validate_request_security(&req, &ctx.env).await {
        let mut error_response = secure_error_response(&security_error, is_development);
        add_cors_headers(&mut error_response, &allowed_origins).ok();
        logger.log_error(&format!("{}", security_error), "authentication", None);
        return Ok(error_response);
    }
    
    // Get KV store for health checker
    let kv_store = ctx.env.kv("METADATA").ok();
    let health_checker = HealthChecker::new(kv_store);
    
    // Perform detailed health check
    let health_result = health_checker.detailed_health_check().await;
    
    let status_code = match health_result.status.as_str() {
        "healthy" => 200,
        "degraded" => 200, // Still operational but with warnings
        _ => 503,
    };
    
    let mut response = Response::from_json(&serde_json::json!({
        "status": health_result.status,
        "message": "CrescendAI API detailed health check",
        "timestamp": health_result.timestamp,
        "checks": health_result.checks,
        "details": health_result.details,
        "version": env!("CARGO_PKG_VERSION"),
        "build_info": {
            "target": "wasm32-unknown-unknown",
            "runtime": "cloudflare-workers"
        }
    }))?;
    
    response = response.with_status(status_code);
    add_cors_headers(&mut response, &allowed_origins)?;
    
    logger.log_request_complete(status_code, None);
    Ok(response)
}

/// System info handler (requires authentication)
async fn secure_system_info_handler(req: Request, ctx: RouteContext<()>) -> Result<Response> {
    let mut logger = RequestLogger::new(&req);
    logger.log_request_start();
    
    let allowed_origins = get_allowed_origins_from_env(Some(&ctx.env));
    let is_development = ctx.env.var("ENVIRONMENT")
        .map(|env| env.to_string() == "development")
        .unwrap_or(false);
    
    // Validate security first
    if let Err(security_error) = validate_request_security(&req, &ctx.env).await {
        let mut error_response = secure_error_response(&security_error, is_development);
        add_cors_headers(&mut error_response, &allowed_origins).ok();
        logger.log_error(&format!("{}", security_error), "authentication", None);
        return Ok(error_response);
    }
    
    // Get system information
    let system_info = SystemInfo::get_system_info();
    let runtime_metrics = SystemInfo::get_runtime_metrics();
    
    let mut response = Response::from_json(&serde_json::json!({
        "system": system_info,
        "metrics": runtime_metrics,
        "environment": {
            "development_mode": is_development,
            "allowed_origins": allowed_origins
        }
    }))?;
    
    add_cors_headers(&mut response, &allowed_origins)?;
    
    logger.log_request_complete(200, None);
    Ok(response)
}
