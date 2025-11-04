use worker::*;
use wasm_bindgen::JsValue;
use serde::{Deserialize, Serialize};
use crate::ast_mock::MertAnalysisResult;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResultRecord {
    pub id: String,
    pub recording_id: String,
    pub scores: String,              // JSON
    pub temporal_segments: String,   // JSON
    pub uncertainty: String,          // JSON
    pub processing_time_ms: i64,
    pub model_version: String,
    pub created_at: i64,
}

pub async fn insert_analysis_result(
    db: &D1Database,
    recording_id: &str,
    analysis: &MertAnalysisResult,
) -> Result<String> {
    let id = format!("analysis-{}", uuid::Uuid::new_v4());
    let created_at = Date::now().as_millis() as i64;

    // Serialize to JSON
    let scores = serde_json::to_string(&analysis.overall_scores)
        .map_err(|e| Error::RustError(format!("Failed to serialize scores: {}", e)))?;

    let temporal_segments = serde_json::to_string(&analysis.temporal_segments)
        .map_err(|e| Error::RustError(format!("Failed to serialize temporal segments: {}", e)))?;

    let uncertainty = serde_json::to_string(&analysis.uncertainty)
        .map_err(|e| Error::RustError(format!("Failed to serialize uncertainty: {}", e)))?;

    let stmt = db.prepare(
        "INSERT INTO analysis_results
         (id, recording_id, scores, temporal_segments, processing_time_ms, model_version, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)"
    );

    stmt.bind(&[
        JsValue::from_str(&id),
        JsValue::from_str(recording_id),
        JsValue::from_str(&scores),
        JsValue::from_str(&temporal_segments),
        JsValue::from_f64(analysis.metadata.processing_time_ms as f64),
        JsValue::from_str(&analysis.metadata.model_version),
        JsValue::from_f64(created_at as f64),
    ])?
    .run()
    .await?;

    Ok(id)
}

pub async fn get_analysis_result(
    db: &D1Database,
    recording_id: &str,
) -> Result<Option<MertAnalysisResult>> {
    let stmt = db.prepare(
        "SELECT recording_id, scores, temporal_segments, processing_time_ms, model_version, created_at
         FROM analysis_results
         WHERE recording_id = ?1"
    );

    let result = stmt
        .bind(&[JsValue::from_str(recording_id)])?
        .first::<AnalysisResultRow>(None)
        .await?;

    if let Some(row) = result {
        // Parse JSON fields
        let overall_scores = serde_json::from_str(&row.scores)
            .map_err(|e| Error::RustError(format!("Failed to parse scores: {}", e)))?;

        let temporal_segments: Vec<crate::ast_mock::TemporalSegment> = serde_json::from_str(&row.temporal_segments)
            .map_err(|e| Error::RustError(format!("Failed to parse temporal segments: {}", e)))?;

        // Reconstruct uncertainty from scores if not stored separately
        // For now, we'll create mock uncertainty since the old schema doesn't have it
        let uncertainty = crate::ast_mock::UncertaintyEstimates::new_mock(&overall_scores);

        let metadata = crate::ast_mock::AnalysisMetadata {
            model_version: row.model_version,
            processing_time_ms: row.processing_time_ms as u64,
            audio_duration_seconds: 0.0, // Not stored in DB, would need to get from recording
            num_segments: temporal_segments.len(),
            confidence_score: 0.85, // Default value
        };

        Ok(Some(MertAnalysisResult {
            recording_id: row.recording_id,
            overall_scores,
            temporal_segments,
            uncertainty,
            metadata,
        }))
    } else {
        Ok(None)
    }
}

pub async fn delete_analysis_result(
    db: &D1Database,
    recording_id: &str,
) -> Result<bool> {
    let stmt = db.prepare("DELETE FROM analysis_results WHERE recording_id = ?1");

    let result = stmt
        .bind(&[JsValue::from_str(recording_id)])?
        .run()
        .await?;

    Ok(result.success())
}

#[derive(Debug, Deserialize)]
struct AnalysisResultRow {
    recording_id: String,
    scores: String,
    temporal_segments: String,
    processing_time_ms: i64,
    model_version: String,
    created_at: i64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analysis_result_record_serialization() {
        let record = AnalysisResultRecord {
            id: "analysis-123".to_string(),
            recording_id: "recording-456".to_string(),
            scores: r#"{"note_accuracy":85.0}"#.to_string(),
            temporal_segments: "[]".to_string(),
            uncertainty: r#"{"aleatoric_uncertainty":5.0}"#.to_string(),
            processing_time_ms: 1500,
            model_version: "MERT-330M-v1.0".to_string(),
            created_at: 1234567890,
        };

        let json = serde_json::to_string(&record).unwrap();
        assert!(json.contains("analysis-123"));
        assert!(json.contains("recording-456"));
    }
}
