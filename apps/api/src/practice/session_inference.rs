use wasm_bindgen::JsValue;
use worker::*;

use super::session::{
    base64_encode, sleep_ms, AmtResponse, MuqResponse, PracticeSession, HF_RETRY_DELAYS_ENDING_MS,
    HF_RETRY_DELAYS_MS,
};
use crate::practice::dims::DIMS_6;

impl PracticeSession {
    // --- External service calls ---

    pub(crate) async fn fetch_audio_from_r2(
        &self,
        r2_key: &str,
    ) -> std::result::Result<Vec<u8>, String> {
        let bucket = self
            .env
            .bucket("CHUNKS")
            .map_err(|e| format!("R2 binding failed: {:?}", e))?;
        let object = bucket
            .get(r2_key)
            .execute()
            .await
            .map_err(|e| format!("R2 get failed: {:?}", e))?;
        let object = object.ok_or_else(|| format!("R2 object not found: {}", r2_key))?;
        let bytes = object
            .body()
            .ok_or_else(|| "R2 object has no body".to_string())?
            .bytes()
            .await
            .map_err(|e| format!("R2 read failed: {:?}", e))?;
        Ok(bytes)
    }

    /// Call the MuQ-only endpoint. Sends raw WebM audio bytes, returns 6-dim predictions.
    pub(crate) async fn call_muq_endpoint(
        &self,
        audio_bytes: &[u8],
    ) -> std::result::Result<MuqResponse, String> {
        // MuQ endpoint uses the same transport as the existing HF inference endpoint:
        // POST raw audio bytes with Content-Type: audio/webm;codecs=opus
        let endpoint = self
            .env
            .var("HF_INFERENCE_ENDPOINT")
            .map_err(|e| format!("HF_INFERENCE_ENDPOINT not set: {:?}", e))?
            .to_string();
        let token = self
            .env
            .secret("HF_TOKEN")
            .map_err(|e| format!("HF_TOKEN not set: {:?}", e))?
            .to_string();

        let mut last_err = String::new();
        let delays = if self.inner.borrow().session_ending {
            HF_RETRY_DELAYS_ENDING_MS
        } else {
            HF_RETRY_DELAYS_MS
        };

        for attempt in 0..=delays.len() {
            let headers = worker::Headers::new();
            headers
                .set("Content-Type", "audio/webm;codecs=opus")
                .map_err(|e| format!("{:?}", e))?;
            headers
                .set("Authorization", &format!("Bearer {}", token))
                .map_err(|e| format!("{:?}", e))?;

            let mut init = worker::RequestInit::new();
            init.with_method(worker::Method::Post);
            init.with_headers(headers);
            init.with_body(Some(JsValue::from(js_sys::Uint8Array::from(audio_bytes))));

            let request = worker::Request::new_with_init(&endpoint, &init)
                .map_err(|e| format!("MuQ request creation failed: {:?}", e))?;

            let mut response = match worker::Fetch::Request(request).send().await {
                Ok(r) => r,
                Err(e) => {
                    last_err = format!("MuQ fetch failed: {:?}", e);
                    if attempt < delays.len() {
                        let delay = delays[attempt];
                        console_log!(
                            "MuQ fetch failed (attempt {}), retrying in {}s: {}",
                            attempt + 1,
                            delay / 1000,
                            last_err
                        );
                        sleep_ms(delay).await;
                        continue;
                    }
                    return Err(last_err);
                }
            };

            let status = response.status_code();
            if status == 503 || status == 429 {
                let body = response.text().await.unwrap_or_default();
                last_err = format!("MuQ returned {}: {}", status, body);
                if attempt < delays.len() {
                    let delay = delays[attempt];
                    console_log!(
                        "MuQ {} (attempt {}), retrying in {}s",
                        status,
                        attempt + 1,
                        delay / 1000
                    );
                    sleep_ms(delay).await;
                    continue;
                }
                return Err(last_err);
            }

            if status != 200 {
                let body = response.text().await.unwrap_or_default();
                return Err(format!("MuQ returned {}: {}", status, body));
            }

            let body_text = response
                .text()
                .await
                .map_err(|e| format!("MuQ response read failed: {:?}", e))?;

            let muq: MuqResponse = serde_json::from_str(&body_text).map_err(|e| {
                format!(
                    "MuQ response parse failed: {:?} - body: {}",
                    e,
                    &body_text[..body_text.len().min(200)]
                )
            })?;

            // Validate all 6 dimensions present
            let dim_count = DIMS_6
                .iter()
                .filter(|dim| muq.predictions.contains_key(**dim))
                .count();
            if dim_count < 6 {
                return Err(format!("MuQ returned only {} dimensions", dim_count));
            }

            if attempt > 0 {
                console_log!("MuQ inference succeeded after {} retries", attempt);
            }

            return Ok(muq);
        }

        Err(last_err)
    }

    /// Call the Aria-AMT endpoint. Sends JSON with base64-encoded audio fields.
    /// Returns transcribed MIDI notes and pedal events.
    pub(crate) async fn call_amt_endpoint(
        &self,
        context_audio: Option<&[u8]>,
        chunk_audio: &[u8],
    ) -> std::result::Result<AmtResponse, String> {
        let endpoint = self
            .env
            .var("HF_AMT_ENDPOINT")
            .map_err(|e| format!("HF_AMT_ENDPOINT not set: {:?}", e))?
            .to_string();

        if endpoint.is_empty() {
            return Err("HF_AMT_ENDPOINT is empty (not yet deployed)".to_string());
        }

        let token = self
            .env
            .secret("HF_TOKEN")
            .map_err(|e| format!("HF_TOKEN not set: {:?}", e))?
            .to_string();

        // Build JSON payload with base64-encoded audio
        let chunk_b64 = base64_encode(chunk_audio);
        let context_b64 = context_audio.map(base64_encode);

        let payload = serde_json::json!({
            "chunk_audio": chunk_b64,
            "context_audio": context_b64,
        });
        let payload_str = payload.to_string();

        let mut last_err = String::new();
        let delays = if self.inner.borrow().session_ending {
            HF_RETRY_DELAYS_ENDING_MS
        } else {
            HF_RETRY_DELAYS_MS
        };

        for attempt in 0..=delays.len() {
            let headers = worker::Headers::new();
            headers
                .set("Content-Type", "application/json")
                .map_err(|e| format!("{:?}", e))?;
            headers
                .set("Authorization", &format!("Bearer {}", token))
                .map_err(|e| format!("{:?}", e))?;

            let mut init = worker::RequestInit::new();
            init.with_method(worker::Method::Post);
            init.with_headers(headers);
            init.with_body(Some(JsValue::from_str(&payload_str)));

            let request = worker::Request::new_with_init(&endpoint, &init)
                .map_err(|e| format!("AMT request creation failed: {:?}", e))?;

            let mut response = match worker::Fetch::Request(request).send().await {
                Ok(r) => r,
                Err(e) => {
                    last_err = format!("AMT fetch failed: {:?}", e);
                    if attempt < delays.len() {
                        let delay = delays[attempt];
                        console_log!(
                            "AMT fetch failed (attempt {}), retrying in {}s: {}",
                            attempt + 1,
                            delay / 1000,
                            last_err
                        );
                        sleep_ms(delay).await;
                        continue;
                    }
                    return Err(last_err);
                }
            };

            let status = response.status_code();
            if status == 503 || status == 429 {
                let body = response.text().await.unwrap_or_default();
                last_err = format!("AMT returned {}: {}", status, body);
                if attempt < delays.len() {
                    let delay = delays[attempt];
                    console_log!(
                        "AMT {} (attempt {}), retrying in {}s",
                        status,
                        attempt + 1,
                        delay / 1000
                    );
                    sleep_ms(delay).await;
                    continue;
                }
                return Err(last_err);
            }

            if status != 200 {
                let body = response.text().await.unwrap_or_default();
                return Err(format!("AMT returned {}: {}", status, body));
            }

            let body_text = response
                .text()
                .await
                .map_err(|e| format!("AMT response read failed: {:?}", e))?;

            let amt: AmtResponse = serde_json::from_str(&body_text).map_err(|e| {
                format!(
                    "AMT response parse failed: {:?} - body: {}",
                    e,
                    &body_text[..body_text.len().min(200)]
                )
            })?;

            if attempt > 0 {
                console_log!("AMT inference succeeded after {} retries", attempt);
            }

            return Ok(amt);
        }

        Err(last_err)
    }
}
