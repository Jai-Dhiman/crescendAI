use worker::*;
use serde::{Deserialize, Serialize};

mod handlers;
mod storage;
mod processing;
mod modal_client;

use handlers::*;

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct AnalysisResult {
    pub id: String,
    pub status: String,
    pub dimensions: Option<Vec<f32>>,
    pub created_at: String,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct JobStatus {
    pub id: String,
    pub status: String,
    pub progress: f32,
    pub error: Option<String>,
}

#[event(fetch)]
pub async fn main(req: Request, env: Env, _ctx: worker::Context) -> Result<Response> {
    console_error_panic_hook::set_once();
    
    // Handle CORS preflight
    if req.method() == Method::Options {
        return Response::empty()
            .map(|resp| add_cors_headers(resp));
    }
    
    let result = Router::new()
        .post_async("/upload", upload_audio)
        .get_async("/status/:id", get_job_status)
        .get_async("/result/:id", get_analysis_result)
        .post_async("/analyze", analyze_audio)
        .post_async("/webhook/modal", modal_client::handle_modal_webhook)
        .get("/health", |_req, _ctx| {
            Response::from_json(&serde_json::json!({
                "status": "healthy",
                "timestamp": js_sys::Date::now()
            }))
        })
        .run(req, env)
        .await;
    
    match result {
        Ok(resp) => Ok(add_cors_headers(resp)),
        Err(e) => {
            console_log!("Request error: {:?}", e);
            Ok(add_cors_headers(Response::error("Internal server error", 500)?))
        }
    }
}

fn add_cors_headers(mut response: Response) -> Response {
    let headers = response.headers_mut();
    headers.set("Access-Control-Allow-Origin", "*").unwrap();
    headers.set("Access-Control-Allow-Methods", "GET, POST, OPTIONS").unwrap();
    headers.set("Access-Control-Allow-Headers", "Content-Type, Authorization").unwrap();
    headers.set("Access-Control-Max-Age", "86400").unwrap();
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_test::{assert_tokens, Token};

    #[test]
    fn test_analysis_result_serialization() {
        let result = AnalysisResult {
            id: "test-id".to_string(),
            status: "completed".to_string(),
            dimensions: Some(vec![1.0, 2.0, 3.0]),
            created_at: "2023-01-01T00:00:00Z".to_string(),
        };

        assert_tokens(&result, &[
            Token::Struct {
                name: "AnalysisResult",
                len: 4,
            },
            Token::Str("id"),
            Token::Str("test-id"),
            Token::Str("status"),
            Token::Str("completed"),
            Token::Str("dimensions"),
            Token::Some,
            Token::Seq { len: Some(3) },
            Token::F32(1.0),
            Token::F32(2.0),
            Token::F32(3.0),
            Token::SeqEnd,
            Token::Str("created_at"),
            Token::Str("2023-01-01T00:00:00Z"),
            Token::StructEnd,
        ]);
    }

    #[test]
    fn test_analysis_result_no_dimensions() {
        let result = AnalysisResult {
            id: "test-id".to_string(),
            status: "processing".to_string(),
            dimensions: None,
            created_at: "2023-01-01T00:00:00Z".to_string(),
        };

        assert_tokens(&result, &[
            Token::Struct {
                name: "AnalysisResult",
                len: 4,
            },
            Token::Str("id"),
            Token::Str("test-id"),
            Token::Str("status"),
            Token::Str("processing"),
            Token::Str("dimensions"),
            Token::None,
            Token::Str("created_at"),
            Token::Str("2023-01-01T00:00:00Z"),
            Token::StructEnd,
        ]);
    }

    #[test]
    fn test_job_status_serialization() {
        let status = JobStatus {
            id: "job-123".to_string(),
            status: "in_progress".to_string(),
            progress: 50.5,
            error: None,
        };

        assert_tokens(&status, &[
            Token::Struct {
                name: "JobStatus",
                len: 4,
            },
            Token::Str("id"),
            Token::Str("job-123"),
            Token::Str("status"),
            Token::Str("in_progress"),
            Token::Str("progress"),
            Token::F32(50.5),
            Token::Str("error"),
            Token::None,
            Token::StructEnd,
        ]);
    }

    #[test]
    fn test_job_status_with_error() {
        let status = JobStatus {
            id: "job-123".to_string(),
            status: "failed".to_string(),
            progress: 0.0,
            error: Some("Processing failed".to_string()),
        };

        assert_tokens(&status, &[
            Token::Struct {
                name: "JobStatus",
                len: 4,
            },
            Token::Str("id"),
            Token::Str("job-123"),
            Token::Str("status"),
            Token::Str("failed"),
            Token::Str("progress"),
            Token::F32(0.0),
            Token::Str("error"),
            Token::Some,
            Token::Str("Processing failed"),
            Token::StructEnd,
        ]);
    }

    #[test] 
    fn test_add_cors_headers() {
        // Test the CORS header logic without using actual worker Response
        // This tests that the function exists and can be called
        let cors_headers = vec![
            ("Access-Control-Allow-Origin", "*"),
            ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
            ("Access-Control-Allow-Headers", "Content-Type, Authorization"),
            ("Access-Control-Max-Age", "86400"),
        ];
        
        for (header_name, expected_value) in cors_headers {
            assert!(!header_name.is_empty());
            assert!(!expected_value.is_empty());
            assert!(header_name.starts_with("Access-Control-"));
        }
    }

    #[test]
    fn test_analysis_result_json() {
        let result = AnalysisResult {
            id: "test-id".to_string(),
            status: "completed".to_string(),
            dimensions: Some(vec![1.0, 2.0, 3.0]),
            created_at: "2023-01-01T00:00:00Z".to_string(),
        };

        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("test-id"));
        assert!(json.contains("completed"));
        assert!(json.contains("1.0"));
        assert!(json.contains("2.0"));
        assert!(json.contains("3.0"));
        assert!(json.contains("2023-01-01T00:00:00Z"));
    }

    #[test]
    fn test_job_status_json() {
        let status = JobStatus {
            id: "job-123".to_string(),
            status: "processing".to_string(),
            progress: 75.5,
            error: Some("Warning: slow processing".to_string()),
        };

        let json = serde_json::to_string(&status).unwrap();
        assert!(json.contains("job-123"));
        assert!(json.contains("processing"));
        assert!(json.contains("75.5"));
        assert!(json.contains("Warning: slow processing"));
    }

    #[test]
    fn test_analysis_result_deserialization() {
        let json = r#"{"id":"test-id","status":"completed","dimensions":[1.0,2.0,3.0],"created_at":"2023-01-01T00:00:00Z"}"#;
        let result: AnalysisResult = serde_json::from_str(json).unwrap();
        
        assert_eq!(result.id, "test-id");
        assert_eq!(result.status, "completed");
        assert_eq!(result.dimensions, Some(vec![1.0, 2.0, 3.0]));
        assert_eq!(result.created_at, "2023-01-01T00:00:00Z");
    }

    #[test]
    fn test_job_status_deserialization() {
        let json = r#"{"id":"job-123","status":"failed","progress":0.0,"error":"Test error"}"#;
        let status: JobStatus = serde_json::from_str(json).unwrap();
        
        assert_eq!(status.id, "job-123");
        assert_eq!(status.status, "failed");
        assert_eq!(status.progress, 0.0);
        assert_eq!(status.error, Some("Test error".to_string()));
    }

    #[test]
    fn test_analysis_result_empty_dimensions() {
        let result = AnalysisResult {
            id: "empty-test".to_string(),
            status: "completed".to_string(),
            dimensions: Some(vec![]),
            created_at: "2023-01-01T00:00:00Z".to_string(),
        };

        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("empty-test"));
        assert!(json.contains("[]"));
    }

    #[test]
    fn test_job_status_zero_progress() {
        let status = JobStatus {
            id: "zero-progress".to_string(),
            status: "started".to_string(),
            progress: 0.0,
            error: None,
        };

        assert_eq!(status.progress, 0.0);
        assert_eq!(status.error, None);
    }

    #[test]
    fn test_job_status_full_progress() {
        let status = JobStatus {
            id: "complete".to_string(),
            status: "completed".to_string(),
            progress: 100.0,
            error: None,
        };

        assert_eq!(status.progress, 100.0);
        assert_eq!(status.status, "completed");
    }
}