use http_body_util::BodyExt;
use worker::{event, Context, Env, HttpRequest, Result};

/// Convert an http::Response<axum::body::Body> into a worker::Response.
/// This collects the body into bytes and rebuilds the response for the Workers runtime.
async fn into_worker_response(resp: http::Response<axum::body::Body>) -> Result<worker::Response> {
    let (parts, body) = resp.into_parts();
    let bytes = body
        .collect()
        .await
        .map(|b| b.to_bytes().to_vec())
        .unwrap_or_default();

    let headers = worker::Headers::new();
    for (name, value) in parts.headers.iter() {
        if let Ok(v) = value.to_str() {
            let _ = headers.set(name.as_str(), v);
        }
    }

    worker::Response::from_bytes(bytes)
        .map(|r| r.with_status(parts.status.as_u16()).with_headers(headers))
}

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
) -> Result<worker::Response> {
    console_error_panic_hook::set_once();

    let path = req.uri().path().to_string();
    let method = req.method().clone();
    let origin = req
        .headers()
        .get("origin")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    // WebSocket upgrade for practice sessions -- forward to Durable Object.
    // Must return worker::Response directly (not http::Response) to preserve
    // the webSocket property on the JS Response object.
    if path.starts_with("/api/practice/ws/") && method == http::Method::GET {
        let session_id = path.trim_start_matches("/api/practice/ws/");
        if !session_id.is_empty() && !session_id.contains('/') {
            let namespace = env.durable_object("PRACTICE_SESSION")?;
            let stub = namespace.id_from_name(session_id)?.get_stub()?;
            let url = format!("https://do.internal/ws/{}", session_id);
            let mut worker_req = worker::Request::new(&url, worker::Method::Get)?;
            worker_req.headers_mut()?.set("Upgrade", "websocket")?;
            return stub.fetch_with_request(worker_req).await;
        }
    }

    // Practice session start (authenticated)
    if path == "/api/practice/start" && method == http::Method::POST {
        let headers = req.headers().clone();
        return into_worker_response(with_cors(
            crate::practice::start::handle_start(&env, &headers).await,
            origin.as_deref(),
        )).await;
    }

    // Upload audio chunk to R2 (authenticated)
    if path == "/api/practice/chunk" && method == http::Method::POST {
        let headers = req.headers().clone();
        let query_string = req.uri().query().map(|q| q.to_string());
        let body = req
            .into_body()
            .collect()
            .await
            .map(|b| b.to_bytes().to_vec())
            .unwrap_or_default();

        let query = query_string.unwrap_or_default();
        let params: std::collections::HashMap<String, String> = query
            .split('&')
            .filter_map(|pair| {
                let mut parts = pair.splitn(2, '=');
                Some((parts.next()?.to_string(), parts.next()?.to_string()))
            })
            .collect();

        let session_id = params.get("sessionId").map(|s| s.as_str()).unwrap_or("");
        let chunk_index = params.get("chunkIndex").map(|s| s.as_str()).unwrap_or("0");

        if session_id.is_empty() {
            return into_worker_response(with_cors(
                http::Response::builder()
                    .status(http::StatusCode::BAD_REQUEST)
                    .header("Content-Type", "application/json")
                    .body(axum::body::Body::from(r#"{"error":"Missing sessionId"}"#))
                    .unwrap(),
                origin.as_deref(),
            )).await;
        }

        return into_worker_response(with_cors(
            crate::practice::upload::handle_upload_chunk(&env, &headers, body, session_id, chunk_index).await,
            origin.as_deref(),
        )).await;
    }

    // Handle CORS preflight
    if method == http::Method::OPTIONS {
        return into_worker_response(with_cors(
            http::Response::builder()
                .status(http::StatusCode::NO_CONTENT)
                .body(axum::body::Body::empty())
                .unwrap(),
            origin.as_deref(),
        )).await;
    }

    // Apple auth endpoint
    if path == "/api/auth/apple" && method == http::Method::POST {
        let body = req
            .into_body()
            .collect()
            .await
            .map(|b| b.to_bytes().to_vec())
            .unwrap_or_default();
        return into_worker_response(with_cors(
            crate::auth::handle_apple_auth(&env, &body).await,
            origin.as_deref(),
        )).await;
    }

    // Google auth endpoint
    if path == "/api/auth/google" && method == http::Method::POST {
        let body = req
            .into_body()
            .collect()
            .await
            .map(|b| b.to_bytes().to_vec())
            .unwrap_or_default();
        return into_worker_response(with_cors(
            crate::auth::handle_google_auth(&env, &body).await,
            origin.as_deref(),
        )).await;
    }

    // Auth me endpoint (authenticated)
    if path == "/api/auth/me" && method == http::Method::GET {
        let headers = req.headers().clone();
        return into_worker_response(with_cors(
            crate::auth::handle_auth_me(&env, &headers).await,
            origin.as_deref(),
        )).await;
    }

    // Debug auth endpoint (dev only -- returns 404 in production)
    if path == "/api/auth/debug" && method == http::Method::POST {
        return into_worker_response(with_cors(
            crate::auth::handle_debug_auth(&env).await,
            origin.as_deref(),
        )).await;
    }

    // Auth signout endpoint
    if path == "/api/auth/signout" && method == http::Method::POST {
        return into_worker_response(with_cors(
            crate::auth::handle_signout(&env),
            origin.as_deref(),
        )).await;
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
        return into_worker_response(with_cors(
            crate::services::goals::handle_extract_goals(&env, &headers, &body).await,
            origin.as_deref(),
        )).await;
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
        return into_worker_response(with_cors(
            crate::services::sync::handle_sync(&env, &headers, &body).await,
            origin.as_deref(),
        )).await;
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
        return into_worker_response(with_cors(
            crate::services::ask::handle_ask(&env, &headers, &body).await,
            origin.as_deref(),
        )).await;
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
        return into_worker_response(with_cors(
            crate::services::ask::handle_elaborate(&env, &headers, &body).await,
            origin.as_deref(),
        )).await;
    }

    // List conversations (authenticated)
    if path == "/api/conversations" && method == http::Method::GET {
        let headers = req.headers().clone();
        return into_worker_response(with_cors(
            crate::services::chat::handle_list_conversations(&env, &headers).await,
            origin.as_deref(),
        )).await;
    }

    // Get single conversation with messages (authenticated)
    if path.starts_with("/api/conversations/") && method == http::Method::GET {
        let conversation_id = path.trim_start_matches("/api/conversations/");
        if !conversation_id.is_empty() && !conversation_id.contains('/') {
            let headers = req.headers().clone();
            return into_worker_response(with_cors(
                crate::services::chat::handle_get_conversation(&env, &headers, conversation_id).await,
                origin.as_deref(),
            )).await;
        }
    }

    // Delete conversation (authenticated)
    if path.starts_with("/api/conversations/") && method == http::Method::DELETE {
        let conversation_id = path.trim_start_matches("/api/conversations/");
        if !conversation_id.is_empty() && !conversation_id.contains('/') {
            let headers = req.headers().clone();
            return into_worker_response(with_cors(
                crate::services::chat::handle_delete_conversation(&env, &headers, conversation_id).await,
                origin.as_deref(),
            )).await;
        }
    }

    // Memory extraction eval endpoint (authenticated)
    if path == "/api/memory/extract-chat" && method == http::Method::POST {
        let headers = req.headers().clone();
        let body = req
            .into_body()
            .collect()
            .await
            .map(|b| b.to_bytes().to_vec())
            .unwrap_or_default();
        return into_worker_response(with_cors(
            crate::services::memory::handle_extract_chat(&env, &headers, &body).await,
            origin.as_deref(),
        )).await;
    }

    // Memory store-facts endpoint (authenticated, benchmark/eval)
    if path == "/api/memory/store-facts" && method == http::Method::POST {
        let headers = req.headers().clone();
        let body = req
            .into_body()
            .collect()
            .await
            .map(|b| b.to_bytes().to_vec())
            .unwrap_or_default();
        return into_worker_response(with_cors(
            crate::services::memory::handle_store_facts(&env, &headers, &body).await,
            origin.as_deref(),
        )).await;
    }

    // Memory search endpoint (authenticated, hybrid retrieval)
    if path == "/api/memory/search" && method == http::Method::POST {
        let headers = req.headers().clone();
        let body = req
            .into_body()
            .collect()
            .await
            .map(|b| b.to_bytes().to_vec())
            .unwrap_or_default();
        return into_worker_response(with_cors(
            crate::services::memory::handle_search_facts(&env, &headers, &body).await,
            origin.as_deref(),
        )).await;
    }

    // Memory clear-benchmark endpoint (authenticated, cleanup)
    if path == "/api/memory/clear-benchmark" && method == http::Method::POST {
        let headers = req.headers().clone();
        let body = req
            .into_body()
            .collect()
            .await
            .map(|b| b.to_bytes().to_vec())
            .unwrap_or_default();
        return into_worker_response(with_cors(
            crate::services::memory::handle_clear_benchmark(&env, &headers, &body).await,
            origin.as_deref(),
        )).await;
    }

    // Teaching moment selection endpoint (authenticated)
    if path == "/api/practice/teaching-moment" && method == http::Method::POST {
        let headers = req.headers().clone();
        let body = req
            .into_body()
            .collect()
            .await
            .map(|b| b.to_bytes().to_vec())
            .unwrap_or_default();
        return into_worker_response(with_cors(
            crate::services::teaching_moment_handler::handle_teaching_moment(&env, &headers, &body).await,
            origin.as_deref(),
        )).await;
    }

    // Chat endpoint -- streaming teacher conversation (authenticated)
    // Returns worker::Response directly (not axum) to enable true token-by-token streaming.
    if path == "/api/chat" && method == http::Method::POST {
        let headers = req.headers().clone();
        let body = req
            .into_body()
            .collect()
            .await
            .map(|b| b.to_bytes().to_vec())
            .unwrap_or_default();
        let mut resp = crate::services::chat::handle_chat_stream(&env, &headers, &body).await;
        let allowed_origin = match origin.as_deref() {
            Some(o) if o == "https://crescend.ai" || o == "http://localhost:3000" => o,
            _ => "https://crescend.ai",
        };
        let _ = resp.headers_mut().set("Access-Control-Allow-Origin", allowed_origin);
        let _ = resp.headers_mut().set("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
        let _ = resp.headers_mut().set("Access-Control-Allow-Headers", "Content-Type, Authorization, Cookie");
        let _ = resp.headers_mut().set("Access-Control-Allow-Credentials", "true");
        return Ok(resp);
    }

    // Score library: GET /api/scores/:piece_id/data
    if path.starts_with("/api/scores/") && path.ends_with("/data") && method == http::Method::GET {
        let piece_id = path.trim_start_matches("/api/scores/").trim_end_matches("/data");
        if !piece_id.is_empty() && !piece_id.contains('/') {
            return into_worker_response(with_cors(
                crate::services::scores::handle_get_piece_data(&env, piece_id).await,
                origin.as_deref(),
            )).await;
        }
    }

    // Score library: GET /api/scores/:piece_id
    if path.starts_with("/api/scores/") && method == http::Method::GET {
        let piece_id = path.trim_start_matches("/api/scores/");
        if !piece_id.is_empty() && !piece_id.contains('/') {
            return into_worker_response(with_cors(
                crate::services::scores::handle_get_piece(&env, piece_id).await,
                origin.as_deref(),
            )).await;
        }
    }

    // Score library: GET /api/scores?composer=X (list)
    if path == "/api/scores" && method == http::Method::GET {
        let query = req.uri().query().unwrap_or("");
        let composer = query.split('&').find_map(|pair| {
            let mut parts = pair.splitn(2, '=');
            if parts.next() == Some("composer") {
                parts.next().map(|v| {
                    // Basic percent-decoding for composer names
                    let replaced = v.replace('+', " ");
                    let mut result = String::new();
                    let mut chars = replaced.chars();
                    while let Some(c) = chars.next() {
                        if c == '%' {
                            let hex: String = chars.by_ref().take(2).collect();
                            if let Ok(byte) = u8::from_str_radix(&hex, 16) {
                                result.push(byte as char);
                            } else {
                                result.push('%');
                                result.push_str(&hex);
                            }
                        } else {
                            result.push(c);
                        }
                    }
                    result
                })
            } else {
                None
            }
        });
        return into_worker_response(with_cors(
            crate::services::scores::handle_list_pieces(&env, composer.as_deref()).await,
            origin.as_deref(),
        )).await;
    }

    // Health check
    if path == "/health" {
        return into_worker_response(with_cors(
            http::Response::builder()
                .status(http::StatusCode::OK)
                .body(axum::body::Body::from("OK"))
                .unwrap(),
            origin.as_deref(),
        )).await;
    }

    // All other routes return 404
    into_worker_response(with_cors(
        http::Response::builder()
            .status(http::StatusCode::NOT_FOUND)
            .header("Content-Type", "application/json")
            .body(axum::body::Body::from(r#"{"error":"Not found"}"#))
            .unwrap(),
        origin.as_deref(),
    )).await
}
