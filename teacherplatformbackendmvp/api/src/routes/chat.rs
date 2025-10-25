use axum::{
    extract::{State, Extension},
    response::{sse::{Event, KeepAlive, Sse}, IntoResponse},
    Json,
};
use futures::stream::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use uuid::Uuid;

use crate::{
    auth::jwt::JwtClaims,
    errors::{AppError, Result},
    ingestion::generate_embeddings,
    llm::{LLMChunk, SimulatedLLM},
    search::{hybrid_search, UserContext},
    state::AppState,
};

/// Request to perform a RAG query
#[derive(Debug, Deserialize)]
pub struct QueryRequest {
    pub query: String,
    pub session_id: Option<Uuid>,
}

/// Perform a RAG query with streaming response
pub async fn rag_query(
    Extension(claims): Extension<JwtClaims>,
    State(state): State<AppState>,
    Json(payload): Json<QueryRequest>,
) -> Result<Sse<impl Stream<Item = std::result::Result<Event, Infallible>>>> {
    let user_id = Uuid::parse_str(&claims.sub)?;

    // Validate query
    if payload.query.trim().is_empty() {
        return Err(AppError::BadRequest("Query cannot be empty".to_string()));
    }

    // For MVP, create a simple mock embedding for the query
    // In production, this would call Workers AI
    let query_embedding = create_mock_query_embedding(&payload.query);

    // Create user context for search filtering
    let user_context = UserContext {
        user_id,
        is_public_only: false,
    };

    // Perform hybrid search to get relevant chunks
    let search_results = hybrid_search(
        &state.pool,
        query_embedding,
        &payload.query,
        &user_context,
        10, // Get top 10 from hybrid search
    )
    .await?;

    // Take top 3 for LLM context
    let top_chunks: Vec<_> = search_results.into_iter().take(3).collect();

    // Create LLM and get streaming response
    let llm = SimulatedLLM::new();
    let llm_stream = llm.query_stream(&payload.query, top_chunks);

    // Convert to SSE stream
    let sse_stream = llm_stream.map(|chunk_result| {
        match chunk_result {
            Ok(chunk) => {
                // Serialize the chunk to JSON
                let json = serde_json::to_string(&chunk).unwrap_or_else(|_| "{}".to_string());
                Ok(Event::default().data(json))
            }
            Err(_) => {
                // Error chunk
                let error_json = serde_json::json!({
                    "type": "error",
                    "message": "An error occurred while processing your query"
                });
                Ok(Event::default().data(error_json.to_string()))
            }
        }
    });

    Ok(Sse::new(sse_stream).keep_alive(KeepAlive::default()))
}

/// Create a mock embedding for a query
/// In production, this would call Workers AI
fn create_mock_query_embedding(query: &str) -> Vec<f32> {
    const EMBEDDING_DIM: usize = 768;

    let mut embedding = vec![0.0; EMBEDDING_DIM];
    let bytes = query.as_bytes();

    for (i, val) in embedding.iter_mut().enumerate() {
        let idx = i % bytes.len().max(1);
        let byte_val = bytes.get(idx).copied().unwrap_or(0) as f32;
        *val = (byte_val / 255.0) * 2.0 - 1.0;
    }

    // Add variation based on position
    for (i, val) in embedding.iter_mut().enumerate() {
        *val += (i as f32 / EMBEDDING_DIM as f32) * 0.1;
    }

    // L2 normalization
    let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if magnitude > 0.0 {
        for val in &mut embedding {
            *val /= magnitude;
        }
    }

    embedding
}

/// Health check for chat system
pub async fn chat_health() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "ok",
        "message": "Chat system operational"
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_mock_query_embedding() {
        let embedding = create_mock_query_embedding("test query");
        assert_eq!(embedding.len(), 768);

        // Check normalization
        let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((magnitude - 1.0).abs() < 0.001);
    }
}
