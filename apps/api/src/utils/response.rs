use serde::Serialize;
use worker::{Headers, Response, Result};

/// Creates CORS headers for cross-origin requests.
pub fn cors_headers() -> Result<Headers> {
    let headers = Headers::new();
    headers.set("Access-Control-Allow-Origin", "*")?;
    headers.set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")?;
    headers.set("Access-Control-Allow-Headers", "Content-Type")?;
    headers.set("Content-Type", "application/json")?;
    Ok(headers)
}

/// Creates a JSON response with CORS headers.
pub fn json_response<T: Serialize>(data: &T) -> Result<Response> {
    let headers = cors_headers()?;
    Ok(Response::from_json(data)?.with_headers(headers))
}

/// Creates an error response with CORS headers.
pub fn error_response(message: &str, status: u16) -> Result<Response> {
    #[derive(Serialize)]
    struct ErrorResponse {
        error: String,
        status: u16,
    }

    let headers = cors_headers()?;
    let error = ErrorResponse {
        error: message.to_string(),
        status,
    };

    Ok(Response::from_json(&error)?
        .with_headers(headers)
        .with_status(status))
}

/// Creates a CORS preflight response for OPTIONS requests.
pub fn cors_preflight() -> Result<Response> {
    let headers = cors_headers()?;
    Ok(Response::empty()?.with_headers(headers).with_status(204))
}
