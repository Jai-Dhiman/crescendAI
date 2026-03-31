use wasm_bindgen::JsValue;
use worker::{console_log, js_sys, wasm_bindgen};

use super::error::PracticeError;
use super::{
    base64_encode, sleep_ms, AmtResponse, MuqResponse, PracticeSession, HF_RETRY_DELAYS_ENDING_MS,
    HF_RETRY_DELAYS_MS,
};
use crate::practice::dims::DIMS_6;

impl PracticeSession {
    // --- External service calls ---

    pub(crate) async fn fetch_audio_from_r2(
        &self,
        r2_key: &str,
    ) -> std::result::Result<Vec<u8>, PracticeError> {
        let bucket = self
            .env
            .bucket("CHUNKS")
            .map_err(|e| PracticeError::Storage(format!("R2 binding: {e:?}")))?;
        let object = bucket
            .get(r2_key)
            .execute()
            .await
            .map_err(|e| PracticeError::Storage(format!("R2 get: {e:?}")))?;
        let object = object
            .ok_or_else(|| PracticeError::Storage(format!("R2 object not found: {r2_key}")))?;
        let bytes = object
            .body()
            .ok_or_else(|| PracticeError::Storage("R2 object has no body".into()))?
            .bytes()
            .await
            .map_err(|e| PracticeError::Storage(format!("R2 read: {e:?}")))?;
        Ok(bytes)
    }

    /// Call the MuQ-only endpoint. Sends raw `WebM` audio bytes, returns 6-dim predictions.
    pub(crate) async fn call_muq_endpoint(
        &self,
        audio_bytes: &[u8],
    ) -> std::result::Result<MuqResponse, PracticeError> {
        let endpoint = self
            .env
            .var("HF_INFERENCE_ENDPOINT")
            .map_err(|e| PracticeError::Inference(format!("HF_INFERENCE_ENDPOINT not set: {e:?}")))?
            .to_string();
        let token = self
            .env
            .secret("HF_TOKEN")
            .map_err(|e| PracticeError::Inference(format!("HF_TOKEN not set: {e:?}")))?
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
                .map_err(|e| PracticeError::Inference(format!("{e:?}")))?;
            headers
                .set("Authorization", &format!("Bearer {token}"))
                .map_err(|e| PracticeError::Inference(format!("{e:?}")))?;

            let mut init = worker::RequestInit::new();
            init.with_method(worker::Method::Post);
            init.with_headers(headers);
            init.with_body(Some(JsValue::from(js_sys::Uint8Array::from(audio_bytes))));

            let request = worker::Request::new_with_init(&endpoint, &init)
                .map_err(|e| PracticeError::Inference(format!("MuQ request creation: {e:?}")))?;

            let mut response = match worker::Fetch::Request(request).send().await {
                Ok(r) => r,
                Err(e) => {
                    last_err = format!("MuQ fetch failed: {e:?}");
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
                    return Err(PracticeError::Inference(last_err));
                }
            };

            let status = response.status_code();
            if status == 503 || status == 429 {
                let body = response.text().await.unwrap_or_default();
                last_err = format!("MuQ returned {status}: {body}");
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
                return Err(PracticeError::Inference(last_err));
            }

            if status != 200 {
                let body = response.text().await.unwrap_or_default();
                return Err(PracticeError::Inference(format!(
                    "MuQ returned {status}: {body}"
                )));
            }

            let body_text = response
                .text()
                .await
                .map_err(|e| PracticeError::Inference(format!("MuQ response read: {e:?}")))?;

            let muq: MuqResponse = serde_json::from_str(&body_text).map_err(|e| {
                PracticeError::Inference(format!(
                    "MuQ response parse: {:?} - body: {}",
                    e,
                    crate::truncate_str(&body_text, 200)
                ))
            })?;

            // Validate all 6 dimensions present
            let dim_count = DIMS_6
                .iter()
                .filter(|dim| muq.predictions.contains_key(**dim))
                .count();
            if dim_count < 6 {
                return Err(PracticeError::Inference(format!(
                    "MuQ returned only {dim_count} dimensions"
                )));
            }

            if attempt > 0 {
                console_log!("MuQ inference succeeded after {} retries", attempt);
            }

            return Ok(muq);
        }

        Err(PracticeError::Inference(last_err))
    }

    /// Call the AMT container via service binding (production) or direct HTTP (local dev).
    /// Sends JSON with base64-encoded audio, returns transcribed MIDI notes and pedal events.
    pub(crate) async fn call_amt_endpoint(
        &self,
        context_audio: Option<&[u8]>,
        chunk_audio: &[u8],
    ) -> std::result::Result<AmtResponse, PracticeError> {
        // In local dev, AMT_LOCAL_URL points to a local AMT server
        let local_url = self.env.var("AMT_LOCAL_URL").ok().map(|v| v.to_string());

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
            // Fetch returns (status_code, body_text) or an error.
            let fetch_result: std::result::Result<(u16, String), PracticeError> =
                if let Some(ref url_base) = local_url {
                    // Local dev: direct HTTP to local AMT server
                    let headers = worker::Headers::new();
                    headers
                        .set("Content-Type", "application/json")
                        .map_err(|e| PracticeError::Inference(format!("{e:?}")))?;
                    let mut init = worker::RequestInit::new();
                    init.with_method(worker::Method::Post);
                    init.with_headers(headers);
                    init.with_body(Some(JsValue::from_str(&payload_str)));
                    let url = format!("{url_base}/transcribe");
                    let request = worker::Request::new_with_init(&url, &init)
                        .map_err(|e| PracticeError::Inference(format!("AMT request: {e:?}")))?;
                    match worker::Fetch::Request(request).send().await {
                        Ok(mut r) => {
                            let status = r.status_code();
                            let body = r.text().await.unwrap_or_default();
                            Ok((status, body))
                        }
                        Err(e) => Err(PracticeError::Inference(format!("AMT fetch: {e:?}"))),
                    }
                } else {
                    // Production: CF service binding to AMT container
                    let fetcher = self
                        .env
                        .service("AMT_SERVICE")
                        .map_err(|e| PracticeError::Inference(format!("AMT_SERVICE binding: {e:?}")))?;
                    let headers = worker::Headers::new();
                    headers
                        .set("Content-Type", "application/json")
                        .map_err(|e| PracticeError::Inference(format!("{e:?}")))?;
                    let mut init = worker::RequestInit::new();
                    init.with_method(worker::Method::Post);
                    init.with_headers(headers);
                    init.with_body(Some(JsValue::from_str(&payload_str)));
                    let request = worker::Request::new_with_init(
                        "https://amt-service/transcribe",
                        &init,
                    )
                    .map_err(|e| PracticeError::Inference(format!("AMT request: {e:?}")))?;
                    match fetcher.fetch_request(request).await {
                        Ok(r) => {
                            let status = r.status().as_u16();
                            let body = match http_body_util::BodyExt::collect(r.into_body()).await {
                                Ok(collected) => {
                                    String::from_utf8_lossy(&collected.to_bytes()).into_owned()
                                }
                                Err(_) => String::new(),
                            };
                            Ok((status, body))
                        }
                        Err(e) => Err(PracticeError::Inference(format!("AMT fetch: {e:?}"))),
                    }
                };

            match fetch_result {
                Err(e) => {
                    last_err = e.to_string();
                    if attempt < delays.len() {
                        sleep_ms(delays[attempt]).await;
                        continue;
                    }
                    return Err(PracticeError::Inference(last_err));
                }
                Ok((status, body)) => {
                    if status == 503 || status == 429 {
                        last_err = format!("AMT returned {status}: {body}");
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
                        return Err(PracticeError::Inference(last_err));
                    }

                    if status != 200 {
                        return Err(PracticeError::Inference(format!(
                            "AMT returned {status}: {body}"
                        )));
                    }

                    let amt: AmtResponse = serde_json::from_str(&body).map_err(|e| {
                        PracticeError::Inference(format!(
                            "AMT response parse: {:?} - body: {}",
                            e,
                            crate::truncate_str(&body, 200)
                        ))
                    })?;

                    if attempt > 0 {
                        console_log!("AMT inference succeeded after {} retries", attempt);
                    }

                    return Ok(amt);
                }
            }
        }

        Err(PracticeError::Inference(last_err))
    }
}
