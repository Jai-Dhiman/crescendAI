use worker::{console_log, Env};

/// Handle POST /api/practice/chunk?sessionId=X&chunkIndex=N
/// Body: raw audio bytes (WebM/Opus)
pub async fn handle_upload_chunk(
    env: &Env,
    headers: &http::HeaderMap,
    body: Vec<u8>,
    session_id: &str,
    chunk_index: &str,
) -> http::Response<axum::body::Body> {
    use axum::body::Body;
    use http::{Response, StatusCode};

    // Auth
    let _student_id = match crate::auth::verify_auth_header(headers, env) {
        Ok(id) => id,
        Err(err_response) => return err_response,
    };

    if body.is_empty() {
        return Response::builder()
            .status(StatusCode::BAD_REQUEST)
            .header("Content-Type", "application/json")
            .body(Body::from(r#"{"error":"Empty body"}"#))
            .unwrap();
    }

    let r2_key = format!("sessions/{}/chunks/{}.webm", session_id, chunk_index);

    let bucket = match env.bucket("CHUNKS") {
        Ok(b) => b,
        Err(e) => {
            console_log!("Failed to get R2 bucket: {:?}", e);
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Storage unavailable"}"#))
                .unwrap();
        }
    };

    match bucket.put(&r2_key, body).execute().await {
        Ok(_) => {
            let resp = serde_json::json!({
                "r2Key": r2_key,
                "sessionId": session_id,
                "chunkIndex": chunk_index,
            });
            Response::builder()
                .status(StatusCode::OK)
                .header("Content-Type", "application/json")
                .body(Body::from(serde_json::to_string(&resp).unwrap()))
                .unwrap()
        }
        Err(e) => {
            console_log!("R2 put failed: {:?}", e);
            Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Failed to store chunk"}"#))
                .unwrap()
        }
    }
}
