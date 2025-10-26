use axum::{
    extract::{State, Extension, Path},
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
    llm::{LLMChunk, WorkersAILLM},
    models::{CreateSessionRequest, SessionResponse, SessionListResponse, SessionMessagesResponse},
    search::{hybrid_search_with_rerank, UserContext},
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

    // Check if Workers AI client is available
    let workers_ai = state
        .workers_ai
        .as_ref()
        .ok_or_else(|| {
            AppError::Internal(
                "Workers AI not configured. Please set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_WORKERS_AI_API_TOKEN".to_string()
            )
        })?;

    // Try to get cached embedding first
    let query_embedding = if let Some(cached) = state.cache.get_embedding(&payload.query).await.ok().flatten() {
        tracing::debug!("Using cached embedding for query");
        cached
    } else {
        // Generate query embedding using Workers AI
        tracing::debug!("Generating query embedding for: {}", payload.query);
        let embedding = workers_ai
            .generate_embedding(&payload.query)
            .await
            .map_err(|e| {
                tracing::error!("Failed to generate query embedding: {}", e);
                AppError::Internal(format!("Failed to generate query embedding: {}", e))
            })?;

        // Cache the embedding for future requests
        if let Err(e) = state.cache.put_embedding(&payload.query, embedding.clone()).await {
            tracing::warn!("Failed to cache embedding: {}", e);
        }

        embedding
    };

    // Create user context for search filtering
    let user_context = UserContext {
        user_id,
        is_public_only: false,
    };

    // Create filter string for cache key
    let filter_str = format!("user:{}", user_id);

    // Try to get cached search results
    let search_results = if let Some(cached) = state.cache.get_search_results(&payload.query, &filter_str).await.ok().flatten() {
        tracing::debug!("Using cached search results");
        cached
    } else {
        // Perform hybrid search with re-ranking to get relevant chunks
        tracing::debug!("Performing hybrid search with re-ranking");
        let results = hybrid_search_with_rerank(
            &state.pool,
            query_embedding,
            &payload.query,
            &user_context,
            workers_ai,
            10, // Get top 10 after re-ranking
        )
        .await?;

        // Cache the search results
        if let Err(e) = state.cache.put_search_results(&payload.query, &filter_str, &results).await {
            tracing::warn!("Failed to cache search results: {}", e);
        }

        results
    };

    if search_results.is_empty() {
        return Err(AppError::NotFound(
            "No relevant information found in the knowledge base".to_string(),
        ));
    }

    // Take top 5 for LLM context (increased from 3 for better answers)
    let top_chunks: Vec<_> = search_results.into_iter().take(5).collect();
    tracing::info!(
        "Found {} relevant chunks for query",
        top_chunks.len()
    );

    // Create LLM and get streaming response
    let llm = WorkersAILLM::new(workers_ai.clone());
    let llm_stream = llm.query_stream(&payload.query, top_chunks).await?;

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

/// Health check for chat system
pub async fn chat_health() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "ok",
        "message": "Chat system operational"
    }))
}

/// Create a new chat session
pub async fn create_session(
    Extension(claims): Extension<JwtClaims>,
    State(state): State<AppState>,
    Json(payload): Json<CreateSessionRequest>,
) -> Result<Json<SessionResponse>> {
    let user_id = Uuid::parse_str(&claims.sub)?;

    // Validate project_id if provided
    if let Some(project_id) = payload.project_id {
        // Verify project exists and user has access
        let project_exists: Option<(Uuid,)> = sqlx::query_as(
            "SELECT id FROM projects WHERE id = $1"
        )
        .bind(project_id)
        .fetch_optional(&state.pool)
        .await?;

        if project_exists.is_none() {
            return Err(AppError::NotFound("Project not found".to_string()));
        }
    }

    // Create session
    let session: SessionResponse = sqlx::query_as!(
        SessionResponse,
        r#"
        INSERT INTO chat_sessions (user_id, project_id, title)
        VALUES ($1, $2, $3)
        RETURNING
            id,
            user_id,
            project_id,
            title,
            created_at,
            updated_at,
            NULL::bigint as message_count
        "#,
        user_id,
        payload.project_id,
        payload.title
    )
    .fetch_one(&state.pool)
    .await?;

    tracing::info!("Created chat session {} for user {}", session.id, user_id);

    Ok(Json(session))
}

/// List user's chat sessions
pub async fn list_sessions(
    Extension(claims): Extension<JwtClaims>,
    State(state): State<AppState>,
) -> Result<Json<SessionListResponse>> {
    let user_id = Uuid::parse_str(&claims.sub)?;

    let sessions: Vec<SessionResponse> = sqlx::query_as!(
        SessionResponse,
        r#"
        SELECT
            s.id,
            s.user_id,
            s.project_id,
            s.title,
            s.created_at,
            s.updated_at,
            COUNT(m.id) as message_count
        FROM chat_sessions s
        LEFT JOIN chat_messages m ON m.session_id = s.id
        WHERE s.user_id = $1
        GROUP BY s.id, s.user_id, s.project_id, s.title, s.created_at, s.updated_at
        ORDER BY s.updated_at DESC
        LIMIT 50
        "#,
        user_id
    )
    .fetch_all(&state.pool)
    .await?;

    let total = sessions.len() as i64;

    Ok(Json(SessionListResponse { sessions, total }))
}

/// Get a specific session with its messages
pub async fn get_session(
    Extension(claims): Extension<JwtClaims>,
    State(state): State<AppState>,
    Path(session_id): Path<Uuid>,
) -> Result<Json<SessionMessagesResponse>> {
    let user_id = Uuid::parse_str(&claims.sub)?;

    // Get session details
    let session: SessionResponse = sqlx::query_as!(
        SessionResponse,
        r#"
        SELECT
            s.id,
            s.user_id,
            s.project_id,
            s.title,
            s.created_at,
            s.updated_at,
            COUNT(m.id) as message_count
        FROM chat_sessions s
        LEFT JOIN chat_messages m ON m.session_id = s.id
        WHERE s.id = $1 AND s.user_id = $2
        GROUP BY s.id, s.user_id, s.project_id, s.title, s.created_at, s.updated_at
        "#,
        session_id,
        user_id
    )
    .fetch_optional(&state.pool)
    .await?
    .ok_or_else(|| AppError::NotFound("Session not found".to_string()))?;

    // Get messages for this session
    let messages = sqlx::query_as!(
        crate::models::ChatMessage,
        r#"
        SELECT
            id,
            session_id,
            role,
            content,
            sources,
            confidence as "confidence: f32",
            created_at
        FROM chat_messages
        WHERE session_id = $1
        ORDER BY created_at ASC
        "#,
        session_id
    )
    .fetch_all(&state.pool)
    .await?;

    Ok(Json(SessionMessagesResponse { session, messages }))
}

/// Delete a chat session
pub async fn delete_session(
    Extension(claims): Extension<JwtClaims>,
    State(state): State<AppState>,
    Path(session_id): Path<Uuid>,
) -> Result<impl IntoResponse> {
    let user_id = Uuid::parse_str(&claims.sub)?;

    let result = sqlx::query!(
        "DELETE FROM chat_sessions WHERE id = $1 AND user_id = $2",
        session_id,
        user_id
    )
    .execute(&state.pool)
    .await?;

    if result.rows_affected() == 0 {
        return Err(AppError::NotFound("Session not found".to_string()));
    }

    tracing::info!("Deleted chat session {} for user {}", session_id, user_id);

    Ok(Json(serde_json::json!({
        "message": "Session deleted successfully"
    })))
}

/// Request to store a chat message
#[derive(Debug, Deserialize)]
pub struct StoreMessageRequest {
    pub session_id: Uuid,
    pub role: String, // "user" or "assistant"
    pub content: String,
    pub sources: Option<serde_json::Value>,
    pub confidence: Option<f32>,
}

/// Store a chat message (called by frontend after streaming completes)
pub async fn store_message(
    Extension(claims): Extension<JwtClaims>,
    State(state): State<AppState>,
    Json(payload): Json<StoreMessageRequest>,
) -> Result<Json<crate::models::ChatMessage>> {
    let user_id = Uuid::parse_str(&claims.sub)?;

    // Verify the session belongs to the user
    let session_exists: Option<(Uuid,)> = sqlx::query_as(
        "SELECT id FROM chat_sessions WHERE id = $1 AND user_id = $2"
    )
    .bind(payload.session_id)
    .bind(user_id)
    .fetch_optional(&state.pool)
    .await?;

    if session_exists.is_none() {
        return Err(AppError::NotFound("Session not found".to_string()));
    }

    // Validate role
    if payload.role != "user" && payload.role != "assistant" {
        return Err(AppError::BadRequest("Role must be 'user' or 'assistant'".to_string()));
    }

    // Validate confidence if provided
    if let Some(conf) = payload.confidence {
        if !(0.0..=1.0).contains(&conf) {
            return Err(AppError::BadRequest("Confidence must be between 0.0 and 1.0".to_string()));
        }
    }

    // Store the message
    let message = sqlx::query_as!(
        crate::models::ChatMessage,
        r#"
        INSERT INTO chat_messages (session_id, role, content, sources, confidence)
        VALUES ($1, $2, $3, $4, $5::real)
        RETURNING
            id,
            session_id,
            role,
            content,
            sources,
            confidence as "confidence: f32",
            created_at
        "#,
        payload.session_id,
        payload.role,
        payload.content,
        payload.sources,
        payload.confidence
    )
    .fetch_one(&state.pool)
    .await?;

    // Update session's updated_at timestamp
    sqlx::query!(
        "UPDATE chat_sessions SET updated_at = NOW() WHERE id = $1",
        payload.session_id
    )
    .execute(&state.pool)
    .await?;

    tracing::info!(
        "Stored {} message in session {}",
        payload.role,
        payload.session_id
    );

    Ok(Json(message))
}
