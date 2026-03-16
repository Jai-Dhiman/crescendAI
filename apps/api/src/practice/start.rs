use wasm_bindgen::JsValue;
use worker::Env;

pub async fn handle_start(
    env: &Env,
    headers: &http::HeaderMap,
) -> http::Response<axum::body::Body> {
    use axum::body::Body;
    use http::{Response, StatusCode};

    let _student_id = match crate::auth::verify_auth_header(headers, env) {
        Ok(id) => id,
        Err(err_response) => return err_response,
    };

    // Pre-warm HF inference endpoint (fire-and-forget)
    // This wakes the endpoint from scale-to-zero so it's ready by the time
    // the first 15s audio chunk arrives.
    prewarm_hf_endpoint(env);

    let session_id = crate::services::ask::generate_uuid();

    let resp = serde_json::json!({
        "sessionId": session_id,
    });

    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Body::from(serde_json::to_string(&resp).unwrap()))
        .unwrap()
}

/// Fire-and-forget ping to HF endpoint to trigger scale-up from zero.
/// Does not block the response — the endpoint wakes in the background.
fn prewarm_hf_endpoint(env: &Env) {
    let endpoint = match env.var("HF_INFERENCE_ENDPOINT") {
        Ok(v) => v.to_string(),
        Err(_) => return,
    };
    let token = match env.secret("HF_TOKEN") {
        Ok(v) => v.to_string(),
        Err(_) => return,
    };

    // Spawn a non-awaited fetch — we don't care about the response
    wasm_bindgen_futures::spawn_local(async move {
        let headers = worker::Headers::new();
        let _ = headers.set("Authorization", &format!("Bearer {}", token));
        let _ = headers.set("Content-Type", "application/json");

        let mut init = worker::RequestInit::new();
        init.with_method(worker::Method::Post);
        init.with_headers(headers);
        // Minimal JSON body — will get 400 but triggers the scale-up
        init.with_body(Some(JsValue::from_str("{}")));

        let req = match worker::Request::new_with_init(&endpoint, &init) {
            Ok(r) => r,
            Err(_) => return,
        };
        let result = worker::Fetch::Request(req).send().await;
        match result {
            Ok(resp) => {
                let status = resp.status_code();
                if status == 503 {
                    worker::console_log!("HF pre-warm: endpoint is cold (503), wake-up triggered");
                } else {
                    worker::console_log!("HF pre-warm: endpoint responded with {}", status);
                }
            }
            Err(e) => {
                worker::console_log!("HF pre-warm failed: {:?}", e);
            }
        }
    });
}
