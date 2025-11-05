// Test module declarations for CrescendAI comprehensive test suite

// WASM test configuration
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

// Import all test modules
pub mod security_tests;
pub mod performance_tests;
pub mod integration_tests;
pub mod temporal_analysis_tests;

// Phase 7 Task 7.1 - New integration tests
pub mod v1_api_tests;
pub mod rag_tests;
pub mod cache_tests;

// Test utilities and common functions
pub mod test_utils {
    use worker::*;
    use serde_json::{json, Value};
    use std::collections::HashMap;

    /// Create mock request for testing WASM environment
    pub fn create_mock_request(method: &str, path: &str, headers: Option<HashMap<String, String>>) -> worker::Result<Request> {
        use web_sys::{Request as WebRequest, RequestInit, Headers as WebHeaders};
        // js_sys::Object not needed for this test
        use wasm_bindgen::JsValue;

        // Create RequestInit object
        let mut request_init = RequestInit::new();
        request_init.set_method(method);

        // Create headers if provided
        if let Some(header_map) = headers {
            let web_headers = WebHeaders::new().map_err(|_| "Failed to create headers".to_string())?;
            for (key, value) in header_map {
                web_headers.set(&key, &value).map_err(|_| "Failed to set header".to_string())?;
            }
            request_init.set_headers(&JsValue::from(web_headers));
        }

        // Create the web request
        let web_request = WebRequest::new_with_str_and_init(path, &request_init)
            .map_err(|_| "Failed to create web request".to_string())?;

        // Convert to worker Request
        Ok(Request::from_raw(web_request)
            .map_err(|_| worker::Error::RustError("Failed to convert to worker request".to_string()))?)
    }

    /// Create mock environment for testing
    pub fn create_mock_env() -> MockEnv {
        MockEnv::new()
    }

    /// Mock environment for testing
    pub struct MockEnv {
        pub vars: HashMap<String, String>,
        pub secrets: HashMap<String, String>,
    }

    impl MockEnv {
        pub fn new() -> Self {
            let mut vars = HashMap::new();
            let mut secrets = HashMap::new();
            
            // Set default test values
            vars.insert("ENVIRONMENT".to_string(), "test".to_string());
            vars.insert("ALLOWED_ORIGINS".to_string(), "https://crescendai.com,http://localhost:3000".to_string());
            secrets.insert("API_KEY".to_string(), "test-api-key-12345".to_string());
            secrets.insert("MODAL_WEBHOOK_SECRET".to_string(), "test-webhook-secret".to_string());
            
            Self { vars, secrets }
        }
        
        pub fn set_var(&mut self, key: &str, value: &str) {
            self.vars.insert(key.to_string(), value.to_string());
        }
        
        pub fn set_secret(&mut self, key: &str, value: &str) {
            self.secrets.insert(key.to_string(), value.to_string());
        }
        
        pub fn get_var(&self, key: &str) -> Option<&String> {
            self.vars.get(key)
        }
        
        pub fn get_secret(&self, key: &str) -> Option<&String> {
            self.secrets.get(key)
        }
    }

    /// Validate test response structure
    pub fn validate_response_structure(response: &Value, expected_fields: &[&str]) -> bool {
        for field in expected_fields {
            if !response.get(field).is_some() {
                return false;
            }
        }
        true
    }

    /// Create test audio data
    pub fn create_test_audio_data(format: &str, duration_seconds: u32) -> Vec<u8> {
        match format {
            "wav" => create_wav_data(duration_seconds),
            "mp3" => create_mp3_data(duration_seconds),
            "flac" => create_flac_data(duration_seconds),
            "aac" => create_aac_data(duration_seconds),
            _ => vec![],
        }
    }

    fn create_wav_data(duration_seconds: u32) -> Vec<u8> {
        let sample_rate = 44100;
        let samples = sample_rate * duration_seconds * 2; // Stereo
        let mut data = Vec::new();
        
        // WAV header
        data.extend_from_slice(b"RIFF");
        data.extend_from_slice(&[(samples * 2 + 36) as u8, 0, 0, 0]);
        data.extend_from_slice(b"WAVE");
        data.extend_from_slice(b"fmt ");
        data.extend_from_slice(&[16, 0, 0, 0]);
        data.extend_from_slice(&[1, 0]);
        data.extend_from_slice(&[2, 0]);
        data.extend_from_slice(&[sample_rate as u8, (sample_rate >> 8) as u8, 0, 0]);
        data.extend_from_slice(&[0, 0, 0, 0]);
        data.extend_from_slice(&[4, 0]);
        data.extend_from_slice(&[16, 0]);
        data.extend_from_slice(b"data");
        data.extend_from_slice(&[(samples * 2) as u8, 0, 0, 0]);
        
        // Audio samples (sine wave)
        for i in 0..samples {
            let sample = ((i as f64 * 440.0 * 2.0 * std::f64::consts::PI / sample_rate as f64).sin() * 32767.0) as i16;
            data.extend_from_slice(&sample.to_le_bytes());
        }
        
        data
    }

    fn create_mp3_data(_duration_seconds: u32) -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(b"ID3");
        data.extend_from_slice(&[3, 0, 0]);
        data.extend_from_slice(&[0, 0, 0, 0]);
        data.extend_from_slice(&[0xFF, 0xFB]);
        data.extend_from_slice(&[0u8; 100]);
        data
    }

    fn create_flac_data(_duration_seconds: u32) -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(b"fLaC");
        data.extend_from_slice(&[0u8; 38]);
        data.extend_from_slice(&[0u8; 100]);
        data
    }

    fn create_aac_data(_duration_seconds: u32) -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(&[0xFF, 0xF1]);
        data.extend_from_slice(&[0u8; 100]);
        data
    }

    /// Validate analysis results structure
    pub fn validate_analysis_results(results: &Value) -> worker::Result<()> {
        let required_fields = vec![
            "id", "status", "analysis"
        ];

        for field in required_fields {
            if !results.get(field).is_some() {
                return Err(worker::Error::RustError(format!("Missing required field: {}", field)));
            }
        }

        // Validate analysis data structure
        let analysis = results.get("analysis").unwrap();
        let analysis_fields = vec![
            "rhythm", "pitch", "dynamics", "tempo", "articulation",
            "expression", "technique", "timing", "phrasing", "voicing",
            "pedaling", "hand_coordination", "musical_understanding",
            "stylistic_accuracy", "creativity", "listening",
            "overall_performance", "stage_presence", "repertoire_difficulty"
        ];

        for field in analysis_fields {
            if let Some(value) = analysis.get(field) {
                if let Some(score) = value.as_f64() {
                    if score < 0.0 || score > 10.0 {
                        return Err(worker::Error::RustError(format!("Score {} is out of range [0-10]: {}", field, score)));
                    }
                } else {
                    return Err(worker::Error::RustError(format!("Field {} is not a number", field)));
                }
            } else {
                return Err(worker::Error::RustError(format!("Missing analysis field: {}", field)));
            }
        }

        Ok(())
    }

    /// Create test comparison results
    pub fn create_test_comparison_results() -> Value {
        json!({
            "id": "test-comparison-id",
            "status": "completed",
            "model_a": {
                "model_name": "ast-v1",
                "model_type": "base",
                "analysis": create_test_analysis_data(),
                "insights": ["Model A insight 1", "Model A insight 2"],
                "processing_time": 2.3
            },
            "model_b": {
                "model_name": "ast-v2",
                "model_type": "large",
                "analysis": create_test_analysis_data(),
                "insights": ["Model B insight 1", "Model B insight 2"],
                "processing_time": 3.1
            },
            "created_at": "2024-01-01T00:00:00Z",
            "total_processing_time": 5.4
        })
    }

    /// Create test analysis data
    pub fn create_test_analysis_data() -> Value {
        json!({
            "rhythm": 8.5,
            "pitch": 7.8,
            "dynamics": 8.2,
            "tempo": 7.9,
            "articulation": 8.1,
            "expression": 7.7,
            "technique": 8.3,
            "timing": 8.0,
            "phrasing": 7.9,
            "voicing": 8.1,
            "pedaling": 7.8,
            "hand_coordination": 8.2,
            "musical_understanding": 8.0,
            "stylistic_accuracy": 7.9,
            "creativity": 7.5,
            "listening": 8.1,
            "overall_performance": 8.0,
            "stage_presence": 7.8,
            "repertoire_difficulty": 8.5
        })
    }

    /// Performance assertion helpers
    pub fn assert_response_time(actual_ms: f64, max_ms: f64, operation: &str) {
        assert!(
            actual_ms <= max_ms,
            "{} took {:.2}ms, should be under {:.2}ms",
            operation, actual_ms, max_ms
        );
    }

    pub fn assert_memory_usage(actual_mb: usize, max_mb: usize, operation: &str) {
        assert!(
            actual_mb <= max_mb,
            "{} used {}MB memory, should be under {}MB",
            operation, actual_mb, max_mb
        );
    }

    pub fn assert_cache_hit_rate(hits: u32, total: u32, min_rate: f64, operation: &str) {
        let actual_rate = hits as f64 / total as f64;
        assert!(
            actual_rate >= min_rate,
            "{} cache hit rate {:.2}% should be at least {:.2}%",
            operation, actual_rate * 100.0, min_rate * 100.0
        );
    }

    /// Security testing helpers
    pub fn create_malicious_payload(attack_type: &str) -> Vec<u8> {
        match attack_type {
            "path_traversal" => b"../../etc/passwd".to_vec(),
            "script_injection" => b"<script>alert('xss')</script>".to_vec(),
            "executable" => vec![0x4d, 0x5a], // MZ header
            "oversized" => vec![0u8; 100 * 1024 * 1024], // 100MB
            _ => vec![],
        }
    }

    pub fn create_invalid_api_keys() -> Vec<String> {
        vec![
            "".to_string(),
            "invalid-key".to_string(),
            "Bearer wrong-key".to_string(),
            "expired-key-12345".to_string(),
            "../../../secrets/api-key".to_string(),
        ]
    }

    pub fn create_unauthorized_origins() -> Vec<String> {
        vec![
            "https://malicious-site.com".to_string(),
            "http://evil.example.com".to_string(),
            "https://fake-crescendai.com".to_string(),
            "null".to_string(),
            "".to_string(),
        ]
    }
}

// Test configuration and setup
pub mod test_config {
    pub const TEST_API_KEY: &str = "test-api-key-12345";
    pub const TEST_WEBHOOK_SECRET: &str = "test-webhook-secret-67890";
    pub const MAX_API_LATENCY_MS: f64 = 100.0;
    pub const MAX_PROCESSING_TIME_MS: f64 = 2000.0;
    pub const MAX_MEMORY_USAGE_MB: usize = 128;
    pub const MIN_CACHE_HIT_RATE: f64 = 0.8;
    
    pub fn get_test_environment() -> std::collections::HashMap<String, String> {
        let mut env = std::collections::HashMap::new();
        env.insert("ENVIRONMENT".to_string(), "test".to_string());
        env.insert("API_KEY".to_string(), TEST_API_KEY.to_string());
        env.insert("MODAL_WEBHOOK_SECRET".to_string(), TEST_WEBHOOK_SECRET.to_string());
        env.insert("ALLOWED_ORIGINS".to_string(), "https://crescendai.com,http://localhost:3000".to_string());
        env
    }
}

// Test summary and reporting
pub mod test_reporting {
    use serde_json::{json, Value};
    
    pub struct TestSummary {
        pub total_tests: u32,
        pub passed_tests: u32,
        pub failed_tests: u32,
        pub skipped_tests: u32,
        pub test_suites: Vec<TestSuiteResult>,
    }

    pub struct TestSuiteResult {
        pub name: String,
        pub tests_run: u32,
        pub tests_passed: u32,
        pub duration_ms: f64,
        pub errors: Vec<String>,
    }

    impl TestSummary {
        pub fn new() -> Self {
            Self {
                total_tests: 0,
                passed_tests: 0,
                failed_tests: 0,
                skipped_tests: 0,
                test_suites: Vec::new(),
            }
        }

        pub fn add_suite_result(&mut self, result: TestSuiteResult) {
            self.total_tests += result.tests_run;
            self.passed_tests += result.tests_passed;
            self.failed_tests += result.tests_run - result.tests_passed;
            self.test_suites.push(result);
        }

        pub fn generate_report(&self) -> Value {
            json!({
                "summary": {
                    "total_tests": self.total_tests,
                    "passed_tests": self.passed_tests,
                    "failed_tests": self.failed_tests,
                    "skipped_tests": self.skipped_tests,
                    "success_rate": if self.total_tests > 0 { 
                        self.passed_tests as f64 / self.total_tests as f64 
                    } else { 
                        0.0 
                    }
                },
                "test_suites": self.test_suites.iter().map(|suite| {
                    json!({
                        "name": suite.name,
                        "tests_run": suite.tests_run,
                        "tests_passed": suite.tests_passed,
                        "duration_ms": suite.duration_ms,
                        "success_rate": if suite.tests_run > 0 {
                            suite.tests_passed as f64 / suite.tests_run as f64
                        } else {
                            0.0
                        },
                        "errors": suite.errors
                    })
                }).collect::<Vec<_>>(),
                "timestamp": js_sys::Date::now()
            })
        }
    }
}