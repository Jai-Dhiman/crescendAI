use serde::{Deserialize, Serialize};
use worker::*;

use crate::AnalysisData;
use crate::knowledge_base::{query_top_k, KBChunk};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct RepertoireInfo {
    pub composer: String,
    pub piece: String,
    pub difficulty: Option<u8>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct UserContext {
    pub goals: Vec<String>,
    pub practice_time_per_day_minutes: u32,
    pub constraints: Vec<String>,
    pub repertoire_info: Option<RepertoireInfo>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct TutorRecommendation {
    pub title: String,
    pub detail: String,
    pub applies_to: Vec<String>,
    pub practice_plan: Vec<String>,
    pub estimated_time_minutes: u32,
    pub citations: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct TutorCitation {
    pub id: String,
    pub title: String,
    pub source: String,
    pub url: Option<String>,
    pub sections: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct TutorFeedback {
    pub recommendations: Vec<TutorRecommendation>,
    pub citations: Vec<TutorCitation>,
}

fn pick_weakest_dimensions(analysis: &AnalysisData, n: usize) -> Vec<(String, f32)> {
    let mut pairs = vec![
        ("timing_stable_unstable".to_string(), analysis.timing_stable_unstable),
        ("articulation_short_long".to_string(), analysis.articulation_short_long),
        ("articulation_soft_hard".to_string(), analysis.articulation_soft_hard),
        ("pedal_sparse_saturated".to_string(), analysis.pedal_sparse_saturated),
        ("pedal_clean_blurred".to_string(), analysis.pedal_clean_blurred),
        ("timbre_even_colorful".to_string(), analysis.timbre_even_colorful),
        ("timbre_shallow_rich".to_string(), analysis.timbre_shallow_rich),
        ("timbre_bright_dark".to_string(), analysis.timbre_bright_dark),
        ("timbre_soft_loud".to_string(), analysis.timbre_soft_loud),
        ("dynamic_sophisticated_raw".to_string(), analysis.dynamic_sophisticated_raw),
        ("dynamic_range_little_large".to_string(), analysis.dynamic_range_little_large),
        ("music_making_fast_slow".to_string(), analysis.music_making_fast_slow),
        ("music_making_flat_spacious".to_string(), analysis.music_making_flat_spacious),
        ("music_making_disproportioned_balanced".to_string(), analysis.music_making_disproportioned_balanced),
        ("music_making_pure_dramatic".to_string(), analysis.music_making_pure_dramatic),
        ("emotion_mood_optimistic_dark".to_string(), analysis.emotion_mood_optimistic_dark),
        ("emotion_mood_low_high_energy".to_string(), analysis.emotion_mood_low_high_energy),
        ("emotion_mood_honest_imaginative".to_string(), analysis.emotion_mood_honest_imaginative),
        ("interpretation_unsatisfactory_convincing".to_string(), analysis.interpretation_unsatisfactory_convincing),
    ];
    pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    pairs.into_iter().take(n).collect()
}

fn round3(v: f32) -> f32 { (v * 1000.0).round() / 1000.0 }

fn compact_scores(analysis: &AnalysisData) -> serde_json::Value {
    serde_json::json!({
        "timing_stable_unstable": round3(analysis.timing_stable_unstable),
        "articulation_short_long": round3(analysis.articulation_short_long),
        "articulation_soft_hard": round3(analysis.articulation_soft_hard),
        "pedal_sparse_saturated": round3(analysis.pedal_sparse_saturated),
        "pedal_clean_blurred": round3(analysis.pedal_clean_blurred),
        "timbre_even_colorful": round3(analysis.timbre_even_colorful),
        "timbre_shallow_rich": round3(analysis.timbre_shallow_rich),
        "timbre_bright_dark": round3(analysis.timbre_bright_dark),
        "timbre_soft_loud": round3(analysis.timbre_soft_loud),
        "dynamic_sophisticated_raw": round3(analysis.dynamic_sophisticated_raw),
        "dynamic_range_little_large": round3(analysis.dynamic_range_little_large),
        "music_making_fast_slow": round3(analysis.music_making_fast_slow),
        "music_making_flat_spacious": round3(analysis.music_making_flat_spacious),
        "music_making_disproportioned_balanced": round3(analysis.music_making_disproportioned_balanced),
        "music_making_pure_dramatic": round3(analysis.music_making_pure_dramatic),
        "emotion_mood_optimistic_dark": round3(analysis.emotion_mood_optimistic_dark),
        "emotion_mood_low_high_energy": round3(analysis.emotion_mood_low_high_energy),
        "emotion_mood_honest_imaginative": round3(analysis.emotion_mood_honest_imaginative),
        "interpretation_unsatisfactory_convincing": round3(analysis.interpretation_unsatisfactory_convincing),
    })
}

fn build_retrieval_query(analysis: &AnalysisData, user_ctx: &UserContext) -> String {
    let weakest = pick_weakest_dimensions(analysis, 4)
        .into_iter()
        .map(|(k, _)| k)
        .collect::<Vec<_>>()
        .join(", ");
    let goals = if user_ctx.goals.is_empty() { "".to_string() } else { format!("; Goals: {}", user_ctx.goals.join(", ")) };
    let constraints = if user_ctx.constraints.is_empty() { "".to_string() } else { format!("; Constraints: {}", user_ctx.constraints.join(", ")) };
    let rep = if let Some(r) = &user_ctx.repertoire_info {
        if r.composer.is_empty() && r.piece.is_empty() { "".to_string() } else { format!("; Piece: {} {}", r.composer, r.piece) }
    } else { "".to_string() };
    format!("Focus dims: {}{}{}{}", weakest, goals, constraints, rep)
}

fn snippet(s: &str, max_len: usize) -> String {
    if s.len() <= max_len { s.to_string() } else { format!("{}…", &s[..max_len]) }
}

fn build_prompt_chunks(chunks: &[KBChunk]) -> Vec<serde_json::Value> {
    chunks.iter().map(|c| {
        serde_json::json!({
            "id": c.id,
            "title": c.title,
            "tags": c.tags,
            "source": c.source,
            "url": c.url,
            "snippet": snippet(&c.text, 220)
        })
    }).collect()
}

pub async fn call_llm(env: &Env, system: &str, user: &str, temperature: f32, max_tokens: u32) -> Result<String> {
    // Try Cloudflare AI first via REST
    if let (Ok(account_id), Ok(cf_model), Ok(cf_token)) = (
        env.var("CF_ACCOUNT_ID"), env.var("TUTOR_CF_MODEL"), env.secret("CF_API_TOKEN")
    ) {
        let url = format!(
            "https://api.cloudflare.com/client/v4/accounts/{}/ai/run/{}",
            account_id.to_string(), cf_model.to_string()
        );
        // Try chat-like payload
        let payload = serde_json::json!({
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        });
        let mut headers = Headers::new();
        headers.set("Authorization", &format!("Bearer {}", cf_token.to_string())).ok();
        headers.set("Content-Type", "application/json").ok();
        let mut init = RequestInit::new();
        init.with_method(Method::Post);
        init.with_headers(headers);
        init.with_body(Some(serde_json::to_string(&payload).map_err(|e| worker::Error::RustError(e.to_string()))?.into()));
        let req = Request::new_with_init(&url, &init)?;
        let mut resp = Fetch::Request(req).send().await?;
        if resp.status_code() / 100 == 2 {
            // Accept multiple possible shapes
            let text: String = resp.text().await?;
            // Try common fields
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) {
                if let Some(s) = v.pointer("/result/response").and_then(|x| x.as_str()) {
                    return Ok(s.to_string());
                }
                if let Some(s) = v.pointer("/result/output_text").and_then(|x| x.as_str()) {
                    return Ok(s.to_string());
                }
                if let Some(s) = v.pointer("/result/text").and_then(|x| x.as_str()) {
                    return Ok(s.to_string());
                }
            }
        }
        // fall through to OpenAI
    }

    // OpenAI fallback
    let openai_key = env.secret("OPENAI_API_KEY")
        .map_err(|_| worker::Error::RustError("OPENAI_API_KEY not configured".to_string()))?
        .to_string();
    let model = env.var("TUTOR_OPENAI_MODEL")
        .map(|v| v.to_string())
        .unwrap_or_else(|_| "gpt-5-nano-2025-08-07".to_string());
    let url = "https://api.openai.com/v1/chat/completions";
    
    // Newer models have different parameter requirements
    let is_new_model = model.starts_with("gpt-5") || model.starts_with("o1") || model.starts_with("o3");
    
    let mut payload = serde_json::json!({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
    });
    
    // Only add temperature for older models (newer models only support default temperature=1)
    if !is_new_model {
        payload["temperature"] = serde_json::json!(temperature);
    }
    
    // Use max_completion_tokens for newer models, max_tokens for older ones
    if is_new_model {
        payload["max_completion_tokens"] = serde_json::json!(max_tokens);
    } else {
        payload["max_tokens"] = serde_json::json!(max_tokens);
    }
    
    #[cfg(not(test))]
    worker::console_log!("OpenAI request: model={}, temp={}, max_tokens={}", model, temperature, max_tokens);
    
    let mut headers = Headers::new();
    headers.set("Authorization", &format!("Bearer {}", openai_key)).ok();
    headers.set("Content-Type", "application/json").ok();
    let mut init = RequestInit::new();
    init.with_method(Method::Post);
    init.with_headers(headers);
    init.with_body(Some(serde_json::to_string(&payload).map_err(|e| worker::Error::RustError(e.to_string()))?.into()));
    let req = Request::new_with_init(url, &init)?;
    let mut resp = Fetch::Request(req).send().await?;
    if resp.status_code() / 100 != 2 {
        // Try to get error details
        let error_body = resp.text().await.unwrap_or_else(|_| "(no error body)".to_string());
        #[cfg(not(test))]
        worker::console_log!("OpenAI error response ({}): {}", resp.status_code(), error_body);
        return Err(worker::Error::RustError(format!("OpenAI chat HTTP {}: {}", resp.status_code(), error_body)));
    }
    let v: serde_json::Value = resp.json().await?;
    let content = v.pointer("/choices/0/message/content")
        .and_then(|x| x.as_str())
        .ok_or_else(|| worker::Error::RustError("OpenAI: missing content".to_string()))?;
    Ok(content.to_string())
}

fn schema_repair_request(original: &str) -> String {
    format!("Please return ONLY valid JSON per the schema previously described. If the following is invalid, repair it without adding prose.\n\n{}", original)
}

fn validate_and_normalize(mut feedback: TutorFeedback) -> TutorFeedback {
    // Drop citations not present in citations list
    let valid_ids: std::collections::HashSet<String> = feedback.citations.iter().map(|c| c.id.clone()).collect();
    for rec in feedback.recommendations.iter_mut() {
        rec.citations.retain(|cid| valid_ids.contains(cid));
        if rec.estimated_time_minutes == 0 { rec.estimated_time_minutes = 10; }
        if rec.estimated_time_minutes > 120 { rec.estimated_time_minutes = 60; }
    }
    feedback
}

pub async fn generate_feedback(env: &Env, analysis: &AnalysisData, user_ctx: &UserContext, k: usize) -> Result<TutorFeedback> {
    // Build cache key
    let compact = compact_scores(analysis);
    let cache_key_input = serde_json::json!({
        "scores": compact,
        "ctx": user_ctx,
        "k": k,
        "model": env.var("TUTOR_CF_MODEL").ok().map(|v| v.to_string()),
    }).to_string();
    let cache_key = crate::utils::compute_etag_from_str(&cache_key_input);

    // Try KV cache
    if let Ok(kv) = env.kv("CRESCENDAI_METADATA") {
        if let Ok(Some(cached)) = kv.get(&format!("tutor:{}", cache_key)).text().await {
            if let Ok(tf) = serde_json::from_str::<TutorFeedback>(&cached) {
                return Ok(tf);
            }
        }
    }

    // Build retrieval query and fetch KB chunks (optional if Vectorize not configured)
    let retrieval_query = build_retrieval_query(analysis, user_ctx);
    let retrieved = if env.var("CF_ACCOUNT_ID").is_ok() && env.secret("CF_API_TOKEN").is_ok() {
        match query_top_k(env, &retrieval_query, if k == 0 { 3 } else { k }).await {
            Ok(v) => v,
            Err(_) => vec![],
        }
    } else {
        vec![]
    };

    // Construct prompt
    let sys = "You are a piano pedagogy tutor. Return ONLY valid JSON per the schema. The schema is:\n{\n  \"recommendations\": [\n    { \"title\": string, \"detail\": string, \"applies_to\": string[], \"practice_plan\": string[], \"estimated_time_minutes\": number, \"citations\": string[] }\n  ],\n  \"citations\": [\n    { \"id\": string, \"title\": string, \"source\": string, \"url\"?: string, \"sections\": string[] }\n  ]\n}";
    let user = serde_json::json!({
        "scores": compact,
        "user_context": user_ctx,
        "retrieved": build_prompt_chunks(&retrieved),
        "instruction": "Produce 2–4 concise, actionable recommendations with citations."
    }).to_string();

    // Call LLM
    let temperature = env.var("TUTOR_TEMPERATURE").ok().and_then(|v| v.to_string().parse::<f32>().ok()).unwrap_or(0.3);
    let max_tokens = env.var("TUTOR_MAX_TOKENS").ok().and_then(|v| v.to_string().parse::<u32>().ok()).unwrap_or(300);
    let first = call_llm(env, sys, &user, temperature, max_tokens).await?;

    // Parse JSON
    let mut parsed: Result<TutorFeedback> = serde_json::from_str::<TutorFeedback>(&first)
        .map_err(|e| worker::Error::RustError(format!("JSON parse error: {}", e)));

    if parsed.is_err() {
        // Attempt one repair
        let repair_prompt = schema_repair_request(&first);
        let repaired = call_llm(env, sys, &repair_prompt, 0.0, max_tokens).await?;
        parsed = serde_json::from_str::<TutorFeedback>(&repaired)
            .map_err(|e| worker::Error::RustError(format!("JSON parse error (repair): {}", e)));
    }

    let mut feedback = parsed?;

    // Validate and normalize
    feedback = validate_and_normalize(feedback);

    // Store in KV cache
    if let Ok(kv) = env.kv("CRESCENDAI_METADATA") {
        if let Ok(json) = serde_json::to_string(&feedback) {
            let _ = kv.put(&format!("tutor:{}", cache_key), &json)?.expiration_ttl(86400).execute().await; // 24h
        }
    }

    Ok(feedback)
}
