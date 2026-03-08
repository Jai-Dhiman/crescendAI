use http_body_util::BodyExt;
use worker::{event, Context, Env, HttpRequest, Result};

/// Add CORS headers to a response with origin allowlist.
fn with_cors(response: http::Response<axum::body::Body>, origin: Option<&str>) -> http::Response<axum::body::Body> {
    let (mut parts, body) = response.into_parts();
    let allowed_origin = match origin {
        Some(o) if o == "https://crescend.ai" || o == "http://localhost:3000" => o,
        _ => "https://crescend.ai",
    };
    parts.headers.insert(
        http::header::ACCESS_CONTROL_ALLOW_ORIGIN,
        allowed_origin.parse().unwrap(),
    );
    parts.headers.insert(
        http::header::ACCESS_CONTROL_ALLOW_METHODS,
        "GET, POST, DELETE, OPTIONS".parse().unwrap(),
    );
    parts.headers.insert(
        http::header::ACCESS_CONTROL_ALLOW_HEADERS,
        "Content-Type, Authorization, Cookie".parse().unwrap(),
    );
    parts.headers.insert(
        http::header::HeaderName::from_static("access-control-allow-credentials"),
        "true".parse().unwrap(),
    );
    http::Response::from_parts(parts, body)
}

#[event(fetch)]
async fn fetch(
    req: HttpRequest,
    env: Env,
    _ctx: Context,
) -> Result<http::Response<axum::body::Body>> {
    console_error_panic_hook::set_once();

    let path = req.uri().path().to_string();
    let method = req.method().clone();
    let origin = req
        .headers()
        .get("origin")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    // Handle CORS preflight
    if method == http::Method::OPTIONS {
        return Ok(with_cors(
            http::Response::builder()
                .status(http::StatusCode::NO_CONTENT)
                .body(axum::body::Body::empty())
                .unwrap(),
            origin.as_deref(),
        ));
    }

    // Apple auth endpoint
    if path == "/api/auth/apple" && method == http::Method::POST {
        let body = req
            .into_body()
            .collect()
            .await
            .map(|b| b.to_bytes().to_vec())
            .unwrap_or_default();
        return Ok(with_cors(
            crate::auth::handle_apple_auth(&env, &body).await,
            origin.as_deref(),
        ));
    }

    // Auth me endpoint (authenticated)
    if path == "/api/auth/me" && method == http::Method::GET {
        let headers = req.headers().clone();
        return Ok(with_cors(
            crate::auth::handle_auth_me(&env, &headers).await,
            origin.as_deref(),
        ));
    }

    // Debug auth endpoint (dev only -- returns 404 in production)
    if path == "/api/auth/debug" && method == http::Method::POST {
        return Ok(with_cors(
            crate::auth::handle_debug_auth(&env).await,
            origin.as_deref(),
        ));
    }

    // Auth signout endpoint
    if path == "/api/auth/signout" && method == http::Method::POST {
        return Ok(with_cors(
            crate::auth::handle_signout(),
            origin.as_deref(),
        ));
    }

    // Goal extraction endpoint (authenticated)
    if path == "/api/extract-goals" && method == http::Method::POST {
        let headers = req.headers().clone();
        let body = req
            .into_body()
            .collect()
            .await
            .map(|b| b.to_bytes().to_vec())
            .unwrap_or_default();
        return Ok(with_cors(
            crate::services::goals::handle_extract_goals(&env, &headers, &body).await,
            origin.as_deref(),
        ));
    }

    // Sync endpoint (authenticated)
    if path == "/api/sync" && method == http::Method::POST {
        let headers = req.headers().clone();
        let body = req
            .into_body()
            .collect()
            .await
            .map(|b| b.to_bytes().to_vec())
            .unwrap_or_default();
        return Ok(with_cors(
            crate::services::sync::handle_sync(&env, &headers, &body).await,
            origin.as_deref(),
        ));
    }

    // Ask endpoint -- two-stage teacher pipeline (authenticated)
    if path == "/api/ask" && method == http::Method::POST {
        let headers = req.headers().clone();
        let body = req
            .into_body()
            .collect()
            .await
            .map(|b| b.to_bytes().to_vec())
            .unwrap_or_default();
        return Ok(with_cors(
            crate::services::ask::handle_ask(&env, &headers, &body).await,
            origin.as_deref(),
        ));
    }

    // Ask elaborate endpoint -- "Tell me more" follow-up (authenticated)
    if path == "/api/ask/elaborate" && method == http::Method::POST {
        let headers = req.headers().clone();
        let body = req
            .into_body()
            .collect()
            .await
            .map(|b| b.to_bytes().to_vec())
            .unwrap_or_default();
        return Ok(with_cors(
            crate::services::ask::handle_elaborate(&env, &headers, &body).await,
            origin.as_deref(),
        ));
    }

    // List conversations (authenticated)
    if path == "/api/conversations" && method == http::Method::GET {
        let headers = req.headers().clone();
        return Ok(with_cors(
            crate::services::chat::handle_list_conversations(&env, &headers).await,
            origin.as_deref(),
        ));
    }

    // Get single conversation with messages (authenticated)
    if path.starts_with("/api/conversations/") && method == http::Method::GET {
        let conversation_id = path.trim_start_matches("/api/conversations/");
        if !conversation_id.is_empty() && !conversation_id.contains('/') {
            let headers = req.headers().clone();
            return Ok(with_cors(
                crate::services::chat::handle_get_conversation(&env, &headers, conversation_id).await,
                origin.as_deref(),
            ));
        }
    }

    // Delete conversation (authenticated)
    if path.starts_with("/api/conversations/") && method == http::Method::DELETE {
        let conversation_id = path.trim_start_matches("/api/conversations/");
        if !conversation_id.is_empty() && !conversation_id.contains('/') {
            let headers = req.headers().clone();
            return Ok(with_cors(
                crate::services::chat::handle_delete_conversation(&env, &headers, conversation_id).await,
                origin.as_deref(),
            ));
        }
    }

    // Chat endpoint -- streaming teacher conversation (authenticated)
    if path == "/api/chat" && method == http::Method::POST {
        let headers = req.headers().clone();
        let body = req
            .into_body()
            .collect()
            .await
            .map(|b| b.to_bytes().to_vec())
            .unwrap_or_default();
        return Ok(with_cors(
            crate::services::chat::handle_chat_stream(&env, &headers, &body).await,
            origin.as_deref(),
        ));
    }

    // Health check
    if path == "/health" {
        return Ok(with_cors(
            http::Response::builder()
                .status(http::StatusCode::OK)
                .body(axum::body::Body::from("OK"))
                .unwrap(),
            origin.as_deref(),
        ));
    }

    // All other routes return 404
    Ok(with_cors(
        http::Response::builder()
            .status(http::StatusCode::NOT_FOUND)
            .header("Content-Type", "application/json")
            .body(axum::body::Body::from(r#"{"error":"Not found"}"#))
            .unwrap(),
        origin.as_deref(),
    ))
}
