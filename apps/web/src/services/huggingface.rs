use crate::models::PerformanceDimensions;
use serde::{Deserialize, Serialize};

#[cfg(feature = "ssr")]
use worker::{Env, Fetch, Headers, Method, Request, RequestInit, Url};

/// HuggingFace Inference Endpoints configuration
#[cfg(feature = "ssr")]
const HF_INFERENCE_TIMEOUT_MS: u64 = 120000; // 2 minutes

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

/// Response from HuggingFace Inference Endpoint
#[derive(Debug, Deserialize)]
struct HFInferenceResponse {
    predictions: Option<HFPredictions>,
    error: Option<HFError>,
    #[allow(dead_code)]
    processing_time_ms: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct HFError {
    code: String,
    message: String,
}

#[derive(Debug, Deserialize)]
struct HFPredictions {
    fusion: HFDimensions,
    #[allow(dead_code)]
    audio: Option<HFDimensions>,
    #[allow(dead_code)]
    symbolic: Option<HFDimensions>,
}

#[derive(Debug, Deserialize)]
struct HFDimensions {
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

impl From<HFDimensions> for PerformanceDimensions {
    fn from(d: HFDimensions) -> Self {
        PerformanceDimensions {
            timing: d.timing,
            articulation_length: d.articulation_length,
            articulation_touch: d.articulation_touch,
            pedal_amount: d.pedal_amount,
            pedal_clarity: d.pedal_clarity,
            timbre_variety: d.timbre_variety,
            timbre_depth: d.timbre_depth,
            timbre_brightness: d.timbre_brightness,
            timbre_loudness: d.timbre_loudness,
            dynamics_range: d.dynamics_range,
            tempo: d.tempo,
            space: d.space,
            balance: d.balance,
            drama: d.drama,
            mood_valence: d.mood_valence,
            mood_energy: d.mood_energy,
            mood_imagination: d.mood_imagination,
            interpretation_sophistication: d.interpretation_sophistication,
            interpretation_overall: d.interpretation_overall,
        }
    }
}

/// Get performance dimensions from HuggingFace Inference Endpoint
///
/// If HF is not configured, falls back to mock data.
#[cfg(feature = "ssr")]
pub async fn get_performance_dimensions_from_hf(
    env: &Env,
    audio_url: &str,
    performance_id: &str,
) -> Result<PerformanceDimensions, String> {
    // Get HF configuration from environment
    let hf_token = match env.secret("HF_API_TOKEN") {
        Ok(token) => token.to_string(),
        Err(_) => {
            // Fall back to mock data if HF not configured
            return Ok(get_performance_dimensions(performance_id).await);
        }
    };

    let endpoint_url = match env.var("HF_INFERENCE_ENDPOINT") {
        Ok(url) => {
            let url_str = url.to_string();
            if url_str.is_empty() {
                return Ok(get_performance_dimensions(performance_id).await);
            }
            url_str
        }
        Err(_) => {
            return Ok(get_performance_dimensions(performance_id).await);
        }
    };

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
    let mut headers = Headers::new();
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

    let response_text = response
        .text()
        .await
        .map_err(|e| format!("Failed to read response: {:?}", e))?;

    let hf_response: HFInferenceResponse = serde_json::from_str(&response_text)
        .map_err(|e| format!("Failed to parse response: {:?}", e))?;

    if let Some(error) = hf_response.error {
        return Err(format!(
            "HF inference error: {} - {}",
            error.code, error.message
        ));
    }

    match hf_response.predictions {
        Some(preds) => Ok(preds.fusion.into()),
        None => Err("No predictions in response".to_string()),
    }
}

/// Fallback mock implementation (used when HF not configured)
pub async fn get_performance_dimensions(performance_id: &str) -> PerformanceDimensions {
    let seed = performance_id
        .bytes()
        .fold(0u32, |acc, b| acc.wrapping_add(b as u32));
    let variation = |base: f64, offset: u32| -> f64 {
        let v = ((seed.wrapping_add(offset) % 100) as f64) / 500.0 - 0.1;
        (base + v).clamp(0.5, 0.98)
    };

    match performance_id {
        "horowitz-chopin-ballade-1" => PerformanceDimensions {
            timing: 0.92,
            articulation_length: 0.88,
            articulation_touch: 0.94,
            pedal_amount: 0.85,
            pedal_clarity: 0.82,
            timbre_variety: 0.95,
            timbre_depth: 0.93,
            timbre_brightness: 0.88,
            timbre_loudness: 0.90,
            dynamics_range: 0.96,
            tempo: 0.87,
            space: 0.84,
            balance: 0.91,
            drama: 0.97,
            mood_valence: 0.72,
            mood_energy: 0.94,
            mood_imagination: 0.93,
            interpretation_sophistication: 0.95,
            interpretation_overall: 0.94,
        },
        "argerich-prokofiev-toccata" => PerformanceDimensions {
            timing: 0.94,
            articulation_length: 0.91,
            articulation_touch: 0.89,
            pedal_amount: 0.72,
            pedal_clarity: 0.88,
            timbre_variety: 0.86,
            timbre_depth: 0.84,
            timbre_brightness: 0.93,
            timbre_loudness: 0.95,
            dynamics_range: 0.92,
            tempo: 0.96,
            space: 0.78,
            balance: 0.87,
            drama: 0.94,
            mood_valence: 0.65,
            mood_energy: 0.98,
            mood_imagination: 0.88,
            interpretation_sophistication: 0.91,
            interpretation_overall: 0.92,
        },
        "gould-bach-goldberg-aria" => PerformanceDimensions {
            timing: 0.96,
            articulation_length: 0.94,
            articulation_touch: 0.92,
            pedal_amount: 0.55,
            pedal_clarity: 0.95,
            timbre_variety: 0.78,
            timbre_depth: 0.88,
            timbre_brightness: 0.82,
            timbre_loudness: 0.68,
            dynamics_range: 0.72,
            tempo: 0.65,
            space: 0.94,
            balance: 0.96,
            drama: 0.62,
            mood_valence: 0.85,
            mood_energy: 0.58,
            mood_imagination: 0.92,
            interpretation_sophistication: 0.97,
            interpretation_overall: 0.94,
        },
        "zimerman-chopin-ballade-4" => PerformanceDimensions {
            timing: 0.95,
            articulation_length: 0.93,
            articulation_touch: 0.96,
            pedal_amount: 0.88,
            pedal_clarity: 0.91,
            timbre_variety: 0.92,
            timbre_depth: 0.94,
            timbre_brightness: 0.86,
            timbre_loudness: 0.82,
            dynamics_range: 0.93,
            tempo: 0.84,
            space: 0.90,
            balance: 0.95,
            drama: 0.88,
            mood_valence: 0.78,
            mood_energy: 0.82,
            mood_imagination: 0.95,
            interpretation_sophistication: 0.96,
            interpretation_overall: 0.95,
        },
        "kissin-rachmaninoff-prelude" => PerformanceDimensions {
            timing: 0.91,
            articulation_length: 0.86,
            articulation_touch: 0.88,
            pedal_amount: 0.92,
            pedal_clarity: 0.78,
            timbre_variety: 0.89,
            timbre_depth: 0.91,
            timbre_brightness: 0.87,
            timbre_loudness: 0.93,
            dynamics_range: 0.95,
            tempo: 0.92,
            space: 0.82,
            balance: 0.88,
            drama: 0.93,
            mood_valence: 0.68,
            mood_energy: 0.96,
            mood_imagination: 0.86,
            interpretation_sophistication: 0.88,
            interpretation_overall: 0.90,
        },
        "pollini-beethoven-appassionata" => PerformanceDimensions {
            timing: 0.93,
            articulation_length: 0.90,
            articulation_touch: 0.87,
            pedal_amount: 0.82,
            pedal_clarity: 0.86,
            timbre_variety: 0.88,
            timbre_depth: 0.92,
            timbre_brightness: 0.84,
            timbre_loudness: 0.91,
            dynamics_range: 0.94,
            tempo: 0.89,
            space: 0.85,
            balance: 0.93,
            drama: 0.95,
            mood_valence: 0.62,
            mood_energy: 0.94,
            mood_imagination: 0.87,
            interpretation_sophistication: 0.94,
            interpretation_overall: 0.93,
        },
        _ => PerformanceDimensions {
            timing: variation(0.85, 1),
            articulation_length: variation(0.82, 2),
            articulation_touch: variation(0.84, 3),
            pedal_amount: variation(0.78, 4),
            pedal_clarity: variation(0.80, 5),
            timbre_variety: variation(0.83, 6),
            timbre_depth: variation(0.85, 7),
            timbre_brightness: variation(0.81, 8),
            timbre_loudness: variation(0.82, 9),
            dynamics_range: variation(0.86, 10),
            tempo: variation(0.84, 11),
            space: variation(0.79, 12),
            balance: variation(0.85, 13),
            drama: variation(0.83, 14),
            mood_valence: variation(0.72, 15),
            mood_energy: variation(0.84, 16),
            mood_imagination: variation(0.81, 17),
            interpretation_sophistication: variation(0.84, 18),
            interpretation_overall: variation(0.85, 19),
        },
    }
}
