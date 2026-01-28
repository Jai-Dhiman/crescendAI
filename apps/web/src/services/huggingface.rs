//! HuggingFace Inference Endpoints integration for audio-only piano performance analysis.
//!
//! Uses MuQ layers 9-12 with 4-fold Pianoteq ensemble for audio analysis (R² = 0.537).

use crate::models::PerformanceDimensions;
use serde::{Deserialize, Serialize};

#[cfg(feature = "ssr")]
use worker::{Env, Fetch, Headers, Method, Request, RequestInit, Url};

/// Request to HuggingFace Inference Endpoint
#[derive(Debug, Serialize)]
struct HFInferenceRequest {
    inputs: HFInputs,
    parameters: HFParameters,
}

#[derive(Debug, Serialize)]
struct HFInputs {
    audio_url: String,
    performance_id: String,
}

#[derive(Debug, Serialize)]
struct HFParameters {
    return_intermediate: bool,
    max_duration_seconds: u32,
}

/// Response from HuggingFace Inference Endpoint (audio-only model)
#[derive(Debug, Deserialize)]
struct HFInferenceResponse {
    predictions: Option<HFPredictions>,
    calibrated_predictions: Option<HFPredictions>,
    calibration_context: Option<String>,
    error: Option<HFError>,
    #[allow(dead_code)]
    model_info: Option<HFModelInfo>,
    #[allow(dead_code)]
    processing_time_ms: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct HFError {
    code: String,
    message: String,
}

#[derive(Debug, Deserialize)]
struct HFModelInfo {
    #[allow(dead_code)]
    name: String,
    #[allow(dead_code)]
    r2: f64,
}

/// Audio-only predictions (19 dimensions)
#[derive(Debug, Clone, Deserialize)]
struct HFPredictions {
    timing: f64,
    articulation_length: f64,
    articulation_touch: f64,
    pedal_amount: f64,
    pedal_clarity: f64,
    timbre_variety: f64,
    timbre_depth: f64,
    timbre_brightness: f64,
    timbre_loudness: f64,
    dynamics_range: f64,
    tempo: f64,
    space: f64,
    balance: f64,
    drama: f64,
    mood_valence: f64,
    mood_energy: f64,
    mood_imagination: f64,
    interpretation_sophistication: f64,
    interpretation_overall: f64,
}

impl From<HFPredictions> for PerformanceDimensions {
    fn from(p: HFPredictions) -> Self {
        PerformanceDimensions {
            timing: p.timing,
            articulation_length: p.articulation_length,
            articulation_touch: p.articulation_touch,
            pedal_amount: p.pedal_amount,
            pedal_clarity: p.pedal_clarity,
            timbre_variety: p.timbre_variety,
            timbre_depth: p.timbre_depth,
            timbre_brightness: p.timbre_brightness,
            timbre_loudness: p.timbre_loudness,
            dynamics_range: p.dynamics_range,
            tempo: p.tempo,
            space: p.space,
            balance: p.balance,
            drama: p.drama,
            mood_valence: p.mood_valence,
            mood_energy: p.mood_energy,
            mood_imagination: p.mood_imagination,
            interpretation_sophistication: p.interpretation_sophistication,
            interpretation_overall: p.interpretation_overall,
        }
    }
}

/// Result from HuggingFace inference including both raw and calibrated predictions
#[cfg(feature = "ssr")]
pub struct HFInferenceResult {
    /// Raw model predictions (0-1 scale)
    pub raw_dimensions: PerformanceDimensions,
    /// Calibrated predictions relative to MAESTRO professional benchmarks
    pub calibrated_dimensions: PerformanceDimensions,
    /// Context explaining calibration methodology
    pub calibration_context: Option<String>,
}

/// Get performance dimensions from HuggingFace Inference Endpoint
///
/// Requires HF_API_TOKEN and HF_INFERENCE_ENDPOINT environment variables.
/// Returns both raw and calibrated predictions.
#[cfg(feature = "ssr")]
pub async fn get_performance_dimensions_from_hf(
    env: &Env,
    audio_url: &str,
    performance_id: &str,
) -> Result<HFInferenceResult, String> {
    // Get HF configuration from environment
    let hf_token = env
        .secret("HF_API_TOKEN")
        .map_err(|_| "HF_API_TOKEN not configured")?
        .to_string();

    let endpoint_url = env
        .var("HF_INFERENCE_ENDPOINT")
        .map_err(|_| "HF_INFERENCE_ENDPOINT not configured")?
        .to_string();

    if endpoint_url.is_empty() {
        return Err("HF_INFERENCE_ENDPOINT is empty".to_string());
    }

    // Build request
    let request_body = HFInferenceRequest {
        inputs: HFInputs {
            audio_url: audio_url.to_string(),
            performance_id: performance_id.to_string(),
        },
        parameters: HFParameters {
            return_intermediate: false,
            max_duration_seconds: 300,
        },
    };

    let body_json = serde_json::to_string(&request_body)
        .map_err(|e| format!("Failed to serialize request: {:?}", e))?;

    // Make request to HF endpoint
    let headers = Headers::new();
    headers
        .set("Authorization", &format!("Bearer {}", hf_token))
        .map_err(|e| format!("Failed to set auth header: {:?}", e))?;
    headers
        .set("Content-Type", "application/json")
        .map_err(|e| format!("Failed to set content-type: {:?}", e))?;

    let url: Url = endpoint_url
        .parse()
        .map_err(|e| format!("Invalid endpoint URL: {:?}", e))?;

    let mut init = RequestInit::new();
    init.with_method(Method::Post);
    init.with_headers(headers);
    init.with_body(Some(body_json.into()));

    let request = Request::new_with_init(url.as_str(), &init)
        .map_err(|e| format!("Failed to create request: {:?}", e))?;

    let mut response = Fetch::Request(request)
        .send()
        .await
        .map_err(|e| format!("HF inference request failed: {:?}", e))?;

    let status = response.status_code();
    let response_text = response
        .text()
        .await
        .map_err(|e| format!("Failed to read response: {:?}", e))?;

    if status != 200 {
        return Err(format!("HF endpoint returned status {}: {}", status, response_text));
    }

    let hf_response: HFInferenceResponse = serde_json::from_str(&response_text)
        .map_err(|e| format!("Failed to parse response: {:?} - body: {}", e, response_text))?;

    if let Some(error) = hf_response.error {
        return Err(format!(
            "HF inference error: {} - {}",
            error.code, error.message
        ));
    }

    let raw_predictions = hf_response.predictions
        .ok_or("No predictions in response")?;

    // Use calibrated predictions if available, otherwise fall back to raw
    let calibrated_predictions = hf_response.calibrated_predictions
        .unwrap_or_else(|| raw_predictions.clone());

    Ok(HFInferenceResult {
        raw_dimensions: raw_predictions.clone().into(),
        calibrated_dimensions: calibrated_predictions.into(),
        calibration_context: hf_response.calibration_context,
    })
}
