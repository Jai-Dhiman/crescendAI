use axum::{routing::get, Router};
use http_body_util::BodyExt;
use leptos::prelude::*;
use leptos_axum::{generate_route_list, LeptosRoutes};
use tower::ServiceExt;
use worker::{event, console_log, Context, Env, HttpRequest, Result};

use crate::models::{AnalysisResult, CitedFeedback, ModelResult, Performance, PerformanceDimensions, UploadedPerformance};
use crate::services::{
    generate_chat_response, generate_cited_feedback, generate_fallback_feedback,
    get_performance_dimensions_from_hf, get_practice_tips, retrieve_for_analysis, retrieve_for_chat,
    upload_audio, get_audio, validate_content_type, MAX_FILE_SIZE,
};
use crate::{api, shell::shell, state::AppState, App};

fn router(app_state: AppState) -> Router {
    let leptos_options = app_state.leptos_options.clone();
    let routes = generate_route_list(App);

    // API routes with our AppState
    // Note: /api/analyze/:id is handled directly in the event handler for Workers bindings
    let api_router = Router::new()
        .route("/api/performances", get(api::list_performances))
        .route("/api/performances/:id", get(api::get_performance))
        .route("/health", get(|| async { "OK" }))
        .with_state(app_state.clone());

    // Leptos SSR routes - the context provides our state to components
    let leptos_router = Router::new()
        .leptos_routes_with_context(
            &leptos_options,
            routes,
            {
                let state = app_state.clone();
                move || provide_context(state.clone())
            },
            {
                let leptos_options = leptos_options.clone();
                move || shell(leptos_options.clone())
            },
        )
        .with_state(leptos_options);

    // Merge API routes first (so they take priority), then Leptos routes
    api_router.merge(leptos_router)
}

/// Generate model variants for the analysis result.
/// In production, these would come from actual model inference.
fn generate_model_variants(base: &PerformanceDimensions) -> Vec<ModelResult> {
    let symbolic = PerformanceDimensions {
        timing: (base.timing * 1.05).min(1.0),
        articulation_length: (base.articulation_length * 0.95).min(1.0),
        articulation_touch: (base.articulation_touch * 0.98).min(1.0),
        pedal_amount: (base.pedal_amount * 0.85).min(1.0),
        pedal_clarity: (base.pedal_clarity * 0.88).min(1.0),
        timbre_variety: (base.timbre_variety * 0.75).min(1.0),
        timbre_depth: (base.timbre_depth * 0.78).min(1.0),
        timbre_brightness: (base.timbre_brightness * 0.80).min(1.0),
        timbre_loudness: (base.timbre_loudness * 0.82).min(1.0),
        dynamics_range: base.dynamics_range,
        tempo: (base.tempo * 1.02).min(1.0),
        space: base.space,
        balance: base.balance,
        drama: (base.drama * 0.95).min(1.0),
        mood_valence: base.mood_valence,
        mood_energy: base.mood_energy,
        mood_imagination: (base.mood_imagination * 0.92).min(1.0),
        interpretation_sophistication: base.interpretation_sophistication,
        interpretation_overall: (base.interpretation_overall * 0.96).min(1.0),
    };

    let audio = PerformanceDimensions {
        timing: (base.timing * 0.97).min(1.0),
        articulation_length: base.articulation_length,
        articulation_touch: (base.articulation_touch * 1.02).min(1.0),
        pedal_amount: (base.pedal_amount * 1.08).min(1.0),
        pedal_clarity: (base.pedal_clarity * 1.05).min(1.0),
        timbre_variety: (base.timbre_variety * 1.12).min(1.0),
        timbre_depth: (base.timbre_depth * 1.10).min(1.0),
        timbre_brightness: (base.timbre_brightness * 1.08).min(1.0),
        timbre_loudness: (base.timbre_loudness * 1.05).min(1.0),
        dynamics_range: (base.dynamics_range * 1.03).min(1.0),
        tempo: (base.tempo * 0.98).min(1.0),
        space: (base.space * 1.02).min(1.0),
        balance: (base.balance * 1.01).min(1.0),
        drama: (base.drama * 1.04).min(1.0),
        mood_valence: (base.mood_valence * 1.02).min(1.0),
        mood_energy: (base.mood_energy * 1.03).min(1.0),
        mood_imagination: (base.mood_imagination * 1.05).min(1.0),
        interpretation_sophistication: (base.interpretation_sophistication * 1.02).min(1.0),
        interpretation_overall: (base.interpretation_overall * 1.03).min(1.0),
    };

    vec![
        ModelResult {
            model_name: "PercePiano".to_string(),
            model_type: "Symbolic".to_string(),
            r_squared: 0.395,
            dimensions: symbolic,
        },
        ModelResult {
            model_name: "MERT-330M".to_string(),
            model_type: "Audio".to_string(),
            r_squared: 0.433,
            dimensions: audio,
        },
        ModelResult {
            model_name: "Late Fusion".to_string(),
            model_type: "Fusion".to_string(),
            r_squared: 0.510,
            dimensions: base.clone(),
        },
    ]
}

/// Find an uploaded performance by probing R2 for the file with different extensions.
///
/// Upload IDs have format: "upload-{timestamp}-{random}"
/// R2 keys have format: "user-uploads/{timestamp}-{random}.{extension}"
async fn find_uploaded_performance(env: &Env, upload_id: &str) -> Option<Performance> {
    let bucket = env.bucket("BUCKET").ok()?;

    // Extract the file identifier from the upload ID
    let file_id = upload_id.strip_prefix("upload-")?;

    // Try common audio extensions
    let extensions = ["wav", "mp3", "m4a", "webm"];

    for ext in extensions {
        let key = format!("user-uploads/{}.{}", file_id, ext);
        // Check if file exists by attempting to get its metadata
        if let Ok(Some(_)) = bucket.head(&key).await {
            console_log!("Found uploaded file: {}", key);
            return Some(Performance {
                id: upload_id.to_string(),
                composer: "Unknown".to_string(),
                piece_title: "Your Recording".to_string(),
                performer: "Uploaded".to_string(),
                thumbnail_url: "/images/upload-placeholder.svg".to_string(),
                audio_url: format!("/r2/{}", key),
                duration_seconds: 0,
                year_recorded: None,
                description: Some("Your uploaded recording".to_string()),
            });
        }
    }

    console_log!("Could not find uploaded file for ID: {}", upload_id);
    None
}

/// Handle full performance analysis with RAG-based feedback.
///
/// This handler returns a complete `AnalysisResult` including:
/// - Performance dimensions
/// - Model variants (Symbolic, Audio, Fusion)
/// - Teacher feedback with citations (via RAG)
/// - Practice tips
///
/// Error handling follows hybrid approach:
/// - Critical errors (D1 binding fails) -> HTTP 500
/// - Soft errors (empty retrieval, LLM fails) -> Fallback to template feedback
async fn handle_full_analyze(
    env: &Env,
    performance_id: &str,
) -> http::Response<axum::body::Body> {
    use axum::body::Body;
    use http::{Response, StatusCode};

    // 1. Get performance - either from demos or uploaded recordings
    let performance = if performance_id.starts_with("upload-") {
        // Handle uploaded recording - probe R2 for the file
        match find_uploaded_performance(env, performance_id).await {
            Some(p) => p,
            None => {
                return Response::builder()
                    .status(StatusCode::NOT_FOUND)
                    .header("Content-Type", "application/json")
                    .body(Body::from(r#"{"error":"Uploaded recording not found"}"#))
                    .unwrap();
            }
        }
    } else {
        // Demo performance - look up from static list
        match Performance::find_by_id(performance_id) {
            Some(p) => p,
            None => {
                return Response::builder()
                    .status(StatusCode::NOT_FOUND)
                    .header("Content-Type", "application/json")
                    .body(Body::from(r#"{"error":"Performance not found"}"#))
                    .unwrap();
            }
        }
    };

    // 2. Get absolute audio URL for HF inference
    // HF needs a publicly accessible URL to download the audio
    let audio_url = if performance.audio_url.starts_with("http") {
        performance.audio_url.clone()
    } else {
        // Convert relative URL to absolute using PUBLIC_URL
        let public_url = env
            .var("PUBLIC_URL")
            .map(|v| v.to_string())
            .unwrap_or_else(|_| "https://crescend.ai".to_string());
        format!("{}{}", public_url.trim_end_matches('/'), performance.audio_url)
    };

    console_log!("Using audio URL for HF: {}", audio_url);

    // 3. Get performance dimensions from HF inference
    let hf_result = match get_performance_dimensions_from_hf(env, &audio_url, performance_id).await {
        Ok(result) => result,
        Err(e) => {
            console_log!("HF inference failed: {}", e);
            return Response::builder()
                .status(StatusCode::SERVICE_UNAVAILABLE)
                .header("Content-Type", "application/json")
                .body(Body::from(format!(r#"{{"error":"Analysis service unavailable: {}"}}"#, e)))
                .unwrap();
        }
    };

    let dimensions = hf_result.raw_dimensions;
    let calibrated_dimensions = hf_result.calibrated_dimensions;
    let calibration_context = hf_result.calibration_context;

    // 3. Generate model variants
    let models = generate_model_variants(&dimensions);

    // 4. Get practice tips
    let practice_tips = get_practice_tips(&performance, &dimensions).await;

    // 5. Get RAG feedback with hybrid error handling
    // Use calibrated dimensions for feedback generation (more interpretable)
    let teacher_feedback = match env.d1("DB") {
        Ok(db) => {
            match retrieve_for_analysis(env, &db, &performance, &calibrated_dimensions).await {
                Ok(chunks) if !chunks.is_empty() => {
                    console_log!("Retrieved {} pedagogy chunks for RAG", chunks.len());
                    match generate_cited_feedback(env, &performance, &calibrated_dimensions, &chunks, calibration_context.as_deref()).await {
                        Ok(feedback) => {
                            console_log!(
                                "Generated RAG feedback with {} citations",
                                feedback.citations.len()
                            );
                            feedback
                        }
                        Err(e) => {
                            console_log!("LLM generation failed: {}. Using fallback.", e);
                            generate_fallback_feedback(&performance, &calibrated_dimensions)
                        }
                    }
                }
                Ok(_) => {
                    console_log!("No chunks retrieved. Using fallback feedback.");
                    generate_fallback_feedback(&performance, &calibrated_dimensions)
                }
                Err(e) => {
                    console_log!("RAG retrieval failed: {:?}. Using fallback.", e);
                    generate_fallback_feedback(&performance, &calibrated_dimensions)
                }
            }
        }
        Err(e) => {
            // Critical error: D1 binding failed
            console_log!("D1 binding failed: {:?}. Returning 500.", e);
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(
                    r#"{"error":"Database connection failed"}"#,
                ))
                .unwrap();
        }
    };

    // 6. Build and return full analysis result
    let result = AnalysisResult {
        performance_id: performance_id.to_string(),
        dimensions,
        calibrated_dimensions,
        calibration_context,
        models,
        teacher_feedback,
        practice_tips,
    };

    let json = serde_json::to_string(&result).unwrap_or_else(|_| "{}".to_string());

    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Body::from(json))
        .unwrap()
}

/// Chat request payload
#[derive(serde::Deserialize)]
struct ChatRequest {
    performance_id: String,
    question: String,
}

/// Chat response payload
#[derive(serde::Serialize)]
struct ChatApiResponse {
    answer: String,
    citations: Vec<crate::models::Citation>,
}

/// Handle audio file upload to R2
///
/// Accepts multipart/form-data with:
/// - file: audio file (mp3, wav, m4a, webm)
/// - title: optional string for the recording title
///
/// Returns UploadedPerformance JSON on success.
async fn handle_upload(
    env: &Env,
    req: HttpRequest,
) -> http::Response<axum::body::Body> {
    use axum::body::Body;
    use http::{Response, StatusCode};

    // Get content type header (clone it since we'll consume req later)
    let content_type = req
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();

    // Check if it's multipart
    if !content_type.contains("multipart/form-data") {
        return Response::builder()
            .status(StatusCode::BAD_REQUEST)
            .header("Content-Type", "application/json")
            .body(Body::from(r#"{"error":"Content-Type must be multipart/form-data"}"#))
            .unwrap();
    }

    // Get R2 bucket
    let bucket = match env.bucket("BUCKET") {
        Ok(b) => b,
        Err(e) => {
            console_log!("R2 bucket binding failed: {:?}", e);
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Storage unavailable"}"#))
                .unwrap();
        }
    };

    // Read body
    let body_bytes = match req.into_body().collect().await {
        Ok(b) => b.to_bytes().to_vec(),
        Err(e) => {
            console_log!("Failed to read body: {:?}", e);
            return Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Failed to read request body"}"#))
                .unwrap();
        }
    };

    // Check size
    if body_bytes.len() > MAX_FILE_SIZE {
        return Response::builder()
            .status(StatusCode::PAYLOAD_TOO_LARGE)
            .header("Content-Type", "application/json")
            .body(Body::from(format!(
                r#"{{"error":"File too large. Max size: {} MB"}}"#,
                MAX_FILE_SIZE / 1024 / 1024
            )))
            .unwrap();
    }

    // Parse multipart boundary from content type
    let boundary = content_type
        .split("boundary=")
        .nth(1)
        .map(|s| s.trim_matches('"'))
        .unwrap_or("");

    if boundary.is_empty() {
        return Response::builder()
            .status(StatusCode::BAD_REQUEST)
            .header("Content-Type", "application/json")
            .body(Body::from(r#"{"error":"Missing multipart boundary"}"#))
            .unwrap();
    }

    // Simple multipart parsing - extract the file part
    let (file_data, file_content_type, title) = match parse_multipart(&body_bytes, boundary) {
        Ok(result) => result,
        Err(e) => {
            console_log!("Multipart parse error: {}", e);
            return Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .header("Content-Type", "application/json")
                .body(Body::from(format!(r#"{{"error":"{}"}}"#, e)))
                .unwrap();
        }
    };

    // Validate content type
    if !validate_content_type(&file_content_type) {
        return Response::builder()
            .status(StatusCode::UNSUPPORTED_MEDIA_TYPE)
            .header("Content-Type", "application/json")
            .body(Body::from(
                r#"{"error":"Unsupported audio format. Allowed: MP3, WAV, M4A, WebM"}"#,
            ))
            .unwrap();
    }

    // Upload to R2
    let upload_result = match upload_audio(&bucket, file_data.clone(), &file_content_type).await {
        Ok(r) => r,
        Err(e) => {
            console_log!("R2 upload failed: {}", e);
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(format!(r#"{{"error":"Upload failed: {}"}}"#, e)))
                .unwrap();
        }
    };

    // Generate ID from key
    let id = upload_result
        .key
        .split('/')
        .last()
        .unwrap_or(&upload_result.key)
        .split('.')
        .next()
        .unwrap_or("upload")
        .to_string();

    let response = UploadedPerformance {
        id: format!("upload-{}", id),
        audio_url: upload_result.url,
        r2_key: upload_result.key,
        title: title.unwrap_or_else(|| "My Recording".to_string()),
        file_size_bytes: file_data.len(),
        content_type: file_content_type,
    };

    let json = serde_json::to_string(&response).unwrap_or_else(|_| "{}".to_string());

    console_log!("Upload complete: {}", response.id);

    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Body::from(json))
        .unwrap()
}

/// Simple multipart form parser
/// Returns (file_data, content_type, title)
fn parse_multipart(body: &[u8], boundary: &str) -> Result<(Vec<u8>, String, Option<String>), String> {
    let boundary_bytes = format!("--{}", boundary);
    let body_str = String::from_utf8_lossy(body);

    let mut file_data: Option<Vec<u8>> = None;
    let mut file_content_type = "audio/mpeg".to_string();
    let mut title: Option<String> = None;

    // Split by boundary
    let parts: Vec<&str> = body_str.split(&boundary_bytes).collect();

    for part in parts {
        if part.is_empty() || part == "--" || part == "--\r\n" {
            continue;
        }

        // Find the header/body separator
        let header_end = part.find("\r\n\r\n").or_else(|| part.find("\n\n"));
        if header_end.is_none() {
            continue;
        }
        let header_end = header_end.unwrap();
        let headers = &part[..header_end];
        let body_start = if part[header_end..].starts_with("\r\n\r\n") {
            header_end + 4
        } else {
            header_end + 2
        };

        // Check if this is the file field
        if headers.contains("name=\"file\"") || headers.contains("name=\"audio\"") {
            // Extract content type
            if let Some(ct_line) = headers.lines().find(|l| l.to_lowercase().starts_with("content-type:")) {
                file_content_type = ct_line
                    .split(':')
                    .nth(1)
                    .map(|s| s.trim().to_string())
                    .unwrap_or_else(|| "audio/mpeg".to_string());
            }

            // Get the binary data - need to work with original bytes
            let part_start = body
                .windows(part.len().min(100))
                .position(|w| {
                    let w_str = String::from_utf8_lossy(w);
                    part.starts_with(&w_str[..w_str.len().min(50)])
                });

            if let Some(start) = part_start {
                let data_start = start + body_start;
                // Find the next boundary or end
                let next_boundary = format!("\r\n--{}", boundary);
                let data_end = body[data_start..]
                    .windows(next_boundary.len())
                    .position(|w| w == next_boundary.as_bytes())
                    .map(|p| data_start + p)
                    .unwrap_or(body.len());

                file_data = Some(body[data_start..data_end].to_vec());
            }
        } else if headers.contains("name=\"title\"") {
            // Extract title
            let body_end = part.len() - if part.ends_with("\r\n") { 2 } else { 0 };
            if body_start < body_end {
                title = Some(part[body_start..body_end].trim().to_string());
            }
        }
    }

    match file_data {
        Some(data) if !data.is_empty() => Ok((data, file_content_type, title)),
        _ => Err("No audio file found in request".to_string()),
    }
}

/// Serve R2 audio files
async fn handle_r2_serve(
    env: &Env,
    key: &str,
) -> http::Response<axum::body::Body> {
    use axum::body::Body;
    use http::{Response, StatusCode};

    let bucket = match env.bucket("BUCKET") {
        Ok(b) => b,
        Err(_) => {
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .body(Body::empty())
                .unwrap();
        }
    };

    match get_audio(&bucket, key).await {
        Ok(data) => {
            let content_type = if key.ends_with(".mp3") {
                "audio/mpeg"
            } else if key.ends_with(".wav") {
                "audio/wav"
            } else if key.ends_with(".m4a") {
                "audio/mp4"
            } else if key.ends_with(".webm") {
                "audio/webm"
            } else {
                "application/octet-stream"
            };

            Response::builder()
                .status(StatusCode::OK)
                .header("Content-Type", content_type)
                .header("Cache-Control", "public, max-age=3600")
                .body(Body::from(data))
                .unwrap()
        }
        Err(_) => Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(Body::empty())
            .unwrap(),
    }
}

/// Handle chat question with RAG-based response
async fn handle_chat(
    env: &Env,
    body: &[u8],
) -> http::Response<axum::body::Body> {
    use axum::body::Body;
    use http::{Response, StatusCode};

    // Parse request body
    let request: ChatRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => {
            console_log!("Failed to parse chat request: {:?}", e);
            return Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Invalid request body"}"#))
                .unwrap();
        }
    };

    // Get performance for context (optional - may not exist)
    let performance = Performance::find_by_id(&request.performance_id);

    // Get D1 database
    let db = match env.d1("DB") {
        Ok(db) => db,
        Err(e) => {
            console_log!("D1 binding failed: {:?}", e);
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Database connection failed"}"#))
                .unwrap();
        }
    };

    // Retrieve relevant pedagogy chunks
    let chunks = match retrieve_for_chat(env, &db, &request.question, performance.as_ref()).await {
        Ok(c) => c,
        Err(e) => {
            console_log!("Chat retrieval failed: {:?}", e);
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Retrieval failed"}"#))
                .unwrap();
        }
    };

    console_log!("Retrieved {} chunks for chat question", chunks.len());

    // Generate response with LLM
    let feedback = match generate_chat_response(env, &request.question, performance.as_ref(), &chunks).await {
        Ok(f) => f,
        Err(e) => {
            console_log!("Chat generation failed: {:?}", e);
            // Return a simple fallback response
            CitedFeedback {
                html: "I apologize, but I'm having trouble generating a response right now. Please try again.".to_string(),
                plain_text: "I apologize, but I'm having trouble generating a response right now. Please try again.".to_string(),
                citations: vec![],
            }
        }
    };

    let response = ChatApiResponse {
        answer: feedback.plain_text,
        citations: feedback.citations,
    };

    let json = serde_json::to_string(&response).unwrap_or_else(|_| r#"{"answer":"Error","citations":[]}"#.to_string());

    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Body::from(json))
        .unwrap()
}

#[event(fetch)]
async fn fetch(
    req: HttpRequest,
    env: Env,
    _ctx: Context,
) -> Result<http::Response<axum::body::Body>> {
    console_error_panic_hook::set_once();

    // Handle API endpoints directly (bypasses Send requirement for Workers bindings)
    let path = req.uri().path();
    let method = req.method();

    // Full analysis endpoint with RAG feedback
    if path.starts_with("/api/analyze/") && method == http::Method::POST {
        let performance_id = path.trim_start_matches("/api/analyze/");
        if !performance_id.is_empty() {
            return Ok(handle_full_analyze(&env, performance_id).await);
        }
    }

    // Chat endpoint with RAG-based Q&A
    if path == "/api/chat" && method == http::Method::POST {
        // Read the request body using http_body_util
        let body = req
            .into_body()
            .collect()
            .await
            .map(|b| b.to_bytes().to_vec())
            .unwrap_or_default();
        return Ok(handle_chat(&env, &body).await);
    }

    // Audio upload endpoint
    if path == "/api/upload" && method == http::Method::POST {
        return Ok(handle_upload(&env, req).await);
    }

    // Serve R2 uploaded audio files
    if path.starts_with("/r2/") {
        let key = path.trim_start_matches("/r2/");
        if !key.is_empty() {
            return Ok(handle_r2_serve(&env, key).await);
        }
    }

    let leptos_options = LeptosOptions::builder()
        .output_name("crescendai")
        .site_pkg_dir("pkg")
        .build();

    let state = AppState::new(env, leptos_options);
    let app = router(state);

    Ok(app.oneshot(req).await?)
}
