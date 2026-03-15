//! Score context loading: fetches score JSON and reference profiles from R2,
//! loads the piece catalog from D1, and logs piece requests for demand tracking.

use wasm_bindgen::JsValue;
use worker::{console_error, Env};

use crate::practice::piece_match::{CatalogPiece, MatchResult, match_piece};

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ScoreNote {
    pub pitch: u8,
    pub pitch_name: String,
    pub velocity: u8,
    pub onset_tick: u32,
    pub onset_seconds: f64,
    pub duration_ticks: u32,
    pub duration_seconds: f64,
    pub track: u8,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ScorePedalEvent {
    #[serde(rename = "type")]
    pub event_type: String,
    pub tick: u32,
    pub seconds: f64,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ScoreBar {
    pub bar_number: u32,
    pub start_tick: u32,
    pub start_seconds: f64,
    pub time_signature: String,
    pub notes: Vec<ScoreNote>,
    pub pedal_events: Vec<ScorePedalEvent>,
    pub note_count: u32,
    pub pitch_range: Vec<u8>,
    pub mean_velocity: u8,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ScoreData {
    pub piece_id: String,
    pub composer: String,
    pub title: String,
    pub key_signature: Option<String>,
    pub time_signatures: Vec<serde_json::Value>,
    pub tempo_markings: Vec<serde_json::Value>,
    pub total_bars: u32,
    pub bars: Vec<ScoreBar>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ReferenceBar {
    pub bar_number: u32,
    pub velocity_mean: f64,
    pub velocity_std: f64,
    pub onset_deviation_mean_ms: f64,
    pub onset_deviation_std_ms: f64,
    pub pedal_duration_mean_beats: Option<f64>,
    pub pedal_changes: Option<u32>,
    pub note_duration_ratio_mean: f64,
    pub performer_count: u32,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ReferenceProfile {
    pub piece_id: String,
    pub performer_count: u32,
    pub bars: Vec<ReferenceBar>,
}

#[derive(Debug, Clone)]
pub struct ScoreContext {
    pub piece_id: String,
    pub composer: String,
    pub title: String,
    pub score: ScoreData,
    pub reference: Option<ReferenceProfile>,
    pub match_confidence: f64,
}

/// Load the piece catalog from D1 (piece_id, composer, title only).
pub async fn load_catalog(env: &Env) -> Vec<CatalogPiece> {
    let db = match env.d1("DB") {
        Ok(db) => db,
        Err(e) => {
            console_error!("D1 binding failed for catalog load: {:?}", e);
            return vec![];
        }
    };

    let stmt = match db
        .prepare("SELECT piece_id, composer, title FROM pieces ORDER BY composer, title")
        .bind(&[])
    {
        Ok(s) => s,
        Err(e) => {
            console_error!("D1 catalog bind failed: {:?}", e);
            return vec![];
        }
    };

    let rows = match stmt.all().await {
        Ok(r) => r,
        Err(e) => {
            console_error!("D1 catalog query failed: {:?}", e);
            return vec![];
        }
    };

    let results: Vec<serde_json::Value> = rows.results().unwrap_or_default();

    results
        .into_iter()
        .filter_map(|row| {
            let piece_id = row.get("piece_id").and_then(|v| v.as_str()).map(String::from)?;
            let composer = row.get("composer").and_then(|v| v.as_str()).map(String::from)?;
            let title = row.get("title").and_then(|v| v.as_str()).map(String::from)?;
            Some(CatalogPiece { piece_id, composer, title })
        })
        .collect()
}

/// Fetch and deserialize score JSON from R2 at `scores/v1/{piece_id}.json`.
pub async fn load_score(env: &Env, piece_id: &str) -> Result<ScoreData, String> {
    let bucket = env
        .bucket("SCORES")
        .map_err(|e| format!("SCORES R2 binding failed: {:?}", e))?;

    let key = format!("scores/v1/{}.json", piece_id);

    let object = bucket
        .get(&key)
        .execute()
        .await
        .map_err(|e| format!("R2 get failed for {}: {:?}", key, e))?
        .ok_or_else(|| format!("Score not found in R2: {}", key))?;

    let bytes = object
        .body()
        .ok_or_else(|| format!("R2 object {} has no body", key))?
        .bytes()
        .await
        .map_err(|e| format!("R2 body read failed for {}: {:?}", key, e))?;

    serde_json::from_slice::<ScoreData>(&bytes)
        .map_err(|e| format!("Score JSON parse failed for {}: {:?}", piece_id, e))
}

/// Fetch and deserialize reference profile from R2 at `references/v1/{piece_id}.json`.
/// Returns None if the object does not exist (reference profiles are optional).
pub async fn load_reference(env: &Env, piece_id: &str) -> Option<ReferenceProfile> {
    let bucket = match env.bucket("SCORES") {
        Ok(b) => b,
        Err(e) => {
            console_error!("SCORES R2 binding failed for reference {}: {:?}", piece_id, e);
            return None;
        }
    };

    let key = format!("references/v1/{}.json", piece_id);

    let object = match bucket.get(&key).execute().await {
        Ok(Some(obj)) => obj,
        Ok(None) => return None,
        Err(e) => {
            console_error!("R2 get failed for reference {}: {:?}", key, e);
            return None;
        }
    };

    let bytes = match object.body() {
        Some(body) => match body.bytes().await {
            Ok(b) => b,
            Err(e) => {
                console_error!("R2 body read failed for reference {}: {:?}", key, e);
                return None;
            }
        },
        None => {
            console_error!("R2 reference object {} has no body", key);
            return None;
        }
    };

    match serde_json::from_slice::<ReferenceProfile>(&bytes) {
        Ok(profile) => Some(profile),
        Err(e) => {
            console_error!("Reference JSON parse failed for {}: {:?}", piece_id, e);
            None
        }
    }
}

/// Log a piece request to D1 for demand tracking.
///
/// Inserts into `piece_requests` table. Uses fire-and-forget pattern: errors are
/// logged but do not propagate to the caller.
pub async fn log_piece_request(
    env: &Env,
    query: &str,
    student_id: &str,
    match_result: Option<&MatchResult>,
) {
    let db = match env.d1("DB") {
        Ok(db) => db,
        Err(e) => {
            console_error!("D1 binding failed for piece_requests log: {:?}", e);
            return;
        }
    };

    let now = js_sys::Date::new_0().to_iso_string().as_string().unwrap_or_default();
    let id = crate::services::ask::generate_uuid();

    let (matched_piece_id, confidence) = match match_result {
        Some(m) => (
            JsValue::from_str(&m.piece_id),
            JsValue::from_f64(m.confidence),
        ),
        None => (JsValue::NULL, JsValue::NULL),
    };

    let stmt = db.prepare(
        "INSERT INTO piece_requests (id, student_id, query, matched_piece_id, confidence, created_at) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
    );

    let bound = match stmt.bind(&[
        JsValue::from_str(&id),
        JsValue::from_str(student_id),
        JsValue::from_str(query),
        matched_piece_id,
        confidence,
        JsValue::from_str(&now),
    ]) {
        Ok(s) => s,
        Err(_) => return,
    };

    let _ = bound.run().await;
}

/// Orchestrate piece resolution: load catalog, match, log, load score + reference.
///
/// Returns None if no match is found above the confidence threshold or if the
/// score data cannot be loaded from R2.
pub async fn resolve_piece(
    env: &Env,
    query: &str,
    student_id: &str,
) -> Option<ScoreContext> {
    let catalog = load_catalog(env).await;

    let match_result = match_piece(query, &catalog);

    log_piece_request(env, query, student_id, match_result.as_ref()).await;

    let matched = match_result?;

    let score = match load_score(env, &matched.piece_id).await {
        Ok(s) => s,
        Err(e) => {
            console_error!("Failed to load score for {}: {}", matched.piece_id, e);
            return None;
        }
    };

    let reference = load_reference(env, &matched.piece_id).await;

    Some(ScoreContext {
        piece_id: matched.piece_id.clone(),
        composer: matched.composer.clone(),
        title: matched.title.clone(),
        score,
        reference,
        match_confidence: matched.confidence,
    })
}
