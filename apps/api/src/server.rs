//! Cloudflare Workers entry point.
//!
//! Routes most requests through the Axum Router (see `routes.rs`).
//! Two carve-outs bypass the router because they need raw `worker::Response`:
//!   1. WebSocket upgrade -- must preserve the JS `webSocket` property
//!   2. Streaming chat -- returns a `ReadableStream` for true SSE

use http_body_util::BodyExt;
use tower_http::cors::{AllowOrigin, CorsLayer};
use tower_service::Service;
use worker::{console_error, event, Context, Env, HttpRequest, Result};

use crate::routes::router;
use crate::state::AppState;

#[event(fetch)]
async fn fetch(
    req: HttpRequest,
    env: Env,
    _ctx: Context,
) -> Result<http::Response<axum::body::Body>> {
    console_error_panic_hook::set_once();

    let path = req.uri().path().to_string();
    let method = req.method().clone();

    // --- Carve-out 1: WebSocket upgrade ---
    // Returns worker::Response converted to http::Response via the "axum" feature.
    if path.starts_with("/api/practice/ws/") && method == http::Method::GET {
        let worker_resp = handle_ws_upgrade(&path, &env, req).await?;
        return Ok(worker_resp.into());
    }

    // --- Carve-out 2: Streaming chat ---
    // Returns worker::Response with ReadableStream body for true SSE.
    if path == "/api/chat" && method == http::Method::POST {
        let worker_resp = handle_chat_stream(&env, req).await?;
        return Ok(worker_resp.into());
    }

    // --- Everything else through Axum Router ---
    let allowed_origin = env
        .var("ALLOWED_ORIGIN")
        .map_or_else(|_| "http://localhost:3000".to_string(), |v| v.to_string());

    // NOTE: allow_credentials(true) cannot be combined with wildcard (Any)
    // allow_headers or allow_origin. Must list explicit headers.
    let cors = CorsLayer::new()
        .allow_origin(
            allowed_origin
                .parse::<http::HeaderValue>()
                .map_or_else(|_| AllowOrigin::any(), AllowOrigin::exact),
        )
        .allow_methods([
            http::Method::GET,
            http::Method::POST,
            http::Method::OPTIONS,
            http::Method::DELETE,
        ])
        .allow_headers([
            http::header::CONTENT_TYPE,
            http::header::AUTHORIZATION,
            http::header::COOKIE,
        ])
        .allow_credentials(true);

    let state = AppState::from_env(env);
    let mut app = router(state).layer(cors);

    Ok(app.call(req).await.unwrap_or_else(|err| {
        console_error!("Router error: {err}");
        http::Response::builder()
            .status(500)
            .header("content-type", "application/json")
            .body(axum::body::Body::from(r#"{"error":"internal"}"#))
            .unwrap_or_default()
    }))
}

// ---------------------------------------------------------------------------
// Carve-out 1: WebSocket upgrade
// ---------------------------------------------------------------------------

/// Forward WebSocket upgrade to the `PRACTICE_SESSION` Durable Object.
///
/// Must bypass the Axum router because the CF runtime needs the JS-level
/// `webSocket` property preserved on the `worker::Response` object.
async fn handle_ws_upgrade(path: &str, env: &Env, req: HttpRequest) -> Result<worker::Response> {
    let session_id = path.trim_start_matches("/api/practice/ws/");
    if session_id.is_empty() || session_id.contains('/') {
        return worker::Response::error("invalid session id", 400);
    }

    // Extract token from cookie or Bearer header and verify JWT.
    let token = req
        .headers()
        .get("cookie")
        .and_then(|v| v.to_str().ok())
        .and_then(|c| {
            c.split(';')
                .find_map(|p| p.trim().strip_prefix("token=").map(String::from))
        })
        .or_else(|| {
            req.headers()
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .and_then(|a| a.strip_prefix("Bearer ").map(String::from))
        });

    let student_id = match token {
        Some(t) => {
            let secret = env
                .secret("JWT_SECRET")
                .map(|s| s.to_string().into_bytes())
                .map_err(|e| worker::Error::RustError(format!("JWT_SECRET: {e}")))?;
            match crate::auth::jwt::verify(&t, &secret) {
                Ok(claims) => claims.sub,
                Err(_) => return worker::Response::error("Unauthorized", 401),
            }
        }
        None => return worker::Response::error("Unauthorized", 401),
    };

    // Extract conversationId from query params.
    let conv_id = req
        .uri()
        .query()
        .unwrap_or("")
        .split('&')
        .find_map(|pair| {
            let (k, v) = pair.split_once('=')?;
            (k == "conversationId").then(|| v.to_string())
        })
        .unwrap_or_default();

    let namespace = env.durable_object("PRACTICE_SESSION")?;
    let stub = namespace.id_from_name(session_id)?.get_stub()?;
    let url = format!(
        "https://do.internal/ws/{session_id}?student_id={student_id}&conversation_id={conv_id}"
    );
    let mut worker_req = worker::Request::new(&url, worker::Method::Get)?;
    worker_req.headers_mut()?.set("Upgrade", "websocket")?;

    stub.fetch_with_request(worker_req).await
}

// ---------------------------------------------------------------------------
// Carve-out 2: Streaming chat
// ---------------------------------------------------------------------------

/// Delegate to the streaming chat handler and wrap with CORS headers.
///
/// `handle_chat_stream` returns a `worker::Response` with a `ReadableStream`
/// body for true SSE token-by-token streaming. We add CORS headers and
/// return it directly.
async fn handle_chat_stream(env: &Env, req: HttpRequest) -> Result<worker::Response> {
    let headers = req.headers().clone();
    let origin = headers
        .get("origin")
        .and_then(|v| v.to_str().ok())
        .map(std::string::ToString::to_string);
    let body = req
        .into_body()
        .collect()
        .await
        .map(|b| b.to_bytes().to_vec())
        .unwrap_or_default();

    let mut resp = crate::services::chat::handle_chat_stream(env, &headers, &body).await;

    let allowed_origin = match origin.as_deref() {
        Some(o) if o == "https://crescend.ai" || o == "http://localhost:3000" => o,
        _ => "https://crescend.ai",
    };
    let _ = resp
        .headers_mut()
        .set("Access-Control-Allow-Origin", allowed_origin);
    let _ = resp
        .headers_mut()
        .set("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
    let _ = resp.headers_mut().set(
        "Access-Control-Allow-Headers",
        "Content-Type, Authorization, Cookie",
    );
    let _ = resp
        .headers_mut()
        .set("Access-Control-Allow-Credentials", "true");

    Ok(resp)
}
