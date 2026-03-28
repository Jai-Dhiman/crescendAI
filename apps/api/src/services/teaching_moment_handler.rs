//! HTTP handler for POST /api/practice/teaching-moment.
//!
//! Receives scored chunks + student context, runs STOP classification
//! and teaching moment selection, returns the top-1 moment (or positive
//! fallback when no issues are detected).

use axum::extract::{Json, State};

use crate::auth::AuthUser;
use crate::error::Result;
use crate::state::AppState;

use crate::services::teaching_moments::{
    RecentObservation, ScoredChunk, StudentBaselines, TeachingMoment,
};

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TeachingMomentRequest {
    pub chunks: Vec<ChunkInput>,
    pub baselines: BaselinesInput,
    #[serde(default)]
    pub recent_observations: Vec<RecentObservationInput>,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ChunkInput {
    pub chunk_index: usize,
    pub scores: ScoresInput,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ScoresInput {
    pub dynamics: f64,
    pub timing: f64,
    pub pedaling: f64,
    pub articulation: f64,
    pub phrasing: f64,
    pub interpretation: f64,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct BaselinesInput {
    pub dynamics: f64,
    pub timing: f64,
    pub pedaling: f64,
    pub articulation: f64,
    pub phrasing: f64,
    pub interpretation: f64,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RecentObservationInput {
    pub dimension: String,
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TeachingMomentResponse {
    pub teaching_moment: Option<TeachingMoment>,
    /// "`need_more`" when too few chunks, "positive" when no issues found,
    /// "corrective" when a teaching moment was identified, "none" should not occur.
    pub result_type: String,
    /// Human-readable message for the "`need_more`" case.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

/// Handle POST /api/practice/teaching-moment
#[worker::send]
pub async fn handle_teaching_moment(
    State(_state): State<AppState>,
    _auth: AuthUser,
    Json(request): Json<TeachingMomentRequest>,
) -> Result<Json<TeachingMomentResponse>> {
    // Convert input types to domain types
    let chunks: Vec<ScoredChunk> = request
        .chunks
        .iter()
        .map(|c| ScoredChunk {
            chunk_index: c.chunk_index,
            scores: [
                c.scores.dynamics,
                c.scores.timing,
                c.scores.pedaling,
                c.scores.articulation,
                c.scores.phrasing,
                c.scores.interpretation,
            ],
        })
        .collect();

    let baselines = StudentBaselines {
        dynamics: request.baselines.dynamics,
        timing: request.baselines.timing,
        pedaling: request.baselines.pedaling,
        articulation: request.baselines.articulation,
        phrasing: request.baselines.phrasing,
        interpretation: request.baselines.interpretation,
    };

    let recent: Vec<RecentObservation> = request
        .recent_observations
        .iter()
        .map(|o| RecentObservation {
            dimension: o.dimension.clone(),
        })
        .collect();

    // Run selection
    let result =
        crate::services::teaching_moments::select_teaching_moment(&chunks, &baselines, &recent);

    let response = match result {
        None => TeachingMomentResponse {
            teaching_moment: None,
            result_type: "need_more".to_string(),
            message: Some(
                "I need a bit more to listen to -- keep playing and ask me again.".to_string(),
            ),
        },
        Some(ref moment) if moment.is_positive => TeachingMomentResponse {
            teaching_moment: result,
            result_type: "positive".to_string(),
            message: None,
        },
        Some(_) => TeachingMomentResponse {
            teaching_moment: result,
            result_type: "corrective".to_string(),
            message: None,
        },
    };

    Ok(Json(response))
}
